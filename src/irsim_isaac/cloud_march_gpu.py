"""The infrared cloud march on the GPU: the same quadrature as the CPU deck, on Warp.

WX.26 (ADR 0190) made :class:`~irsim.atmosphere.weather_fx.WeatherFxDeck` read weather-fx's
``Cloudscape`` -- a function of position, several texture lookups per sample -- and left the
march as the CPU reference: 33 s for a quarter-resolution frame before edge refinement, about
twenty minutes a frame in production. This module is the follow-up that ADR named. It is the
same march line for line -- the slab span, the step count from the finest pitch, the hashed
stratified jitter, the Schwarzschild sum with the band's ratio, the drop-out at the band's
opacity -- with the density evaluated by weather-fx's own Warp ``cloud_density``, the function
the visible per-pixel layer marches. The CPU deck stays the oracle
(``tests/integration/test_cloud_march_gpu.py`` holds the two together); nothing of the physics
lives here, only the arithmetic's placement.

Engine glue: Warp and the submodule's GPU module are imported at call time, after the
``irsim_isaac.env`` probe, so the module imports without either (``tests/unit/test_layering.py``).

docs/physics-model.md §7.5, §7.6; docs/clouds-in-the-infrared.md.
"""

from __future__ import annotations

# The Warp kernel below is typed in Warp's own vocabulary (`wp.array(dtype=...)`, textures),
# which is what Warp compiles from and what mypy cannot read; nothing else in the file is exempt.
# mypy: disable-error-code="valid-type, name-defined, untyped-decorator, no-untyped-def"
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from irsim.atmosphere.cloud_deck import MarchResult
from irsim.atmosphere.weather_fx import (
    OPAQUE_OPTICAL_DEPTH,
    OccludedMarch,
    WeatherFxDeck,
    ensure_weather_fx_on_path,
    ray_hash,
    shell_span,
)

__all__ = ["GpuWeatherFxDeck", "gpu_march_available", "RADIANCE_TABLE_POINTS"]

#: Points on which ``radiance_at_height`` is tabulated between the layer's base and top for the
#: kernel's linear interpolation. The band LUT is itself linear in temperature at 0.05 K and the
#: in-cloud lapse is 6.5 K/km, so 256 points over a 4 km layer is 0.1 K between knots.
RADIANCE_TABLE_POINTS = 256

_KERNELS: dict[str, Any] = {}


def _warp() -> Any:
    from irsim_isaac.env import ensure_warp_on_path

    ensure_warp_on_path()
    import warp as wp

    wp.init()
    return wp


def gpu_march_available() -> bool:
    """Warp with a CUDA device, and the submodule's GPU march to borrow the density from."""
    try:
        ensure_weather_fx_on_path()
        from weather_fx.gpu import warp_available

        if not warp_available():
            return False
        wp = _warp()
        return bool(wp.get_cuda_device_count() > 0)
    except Exception:
        return False


def _kernel() -> Any:
    """Build (once) the march kernel over weather-fx's ``cloud_density``."""
    if "march" in _KERNELS:
        return _KERNELS["march"]
    wp = _warp()
    ensure_weather_fx_on_path()
    from weather_fx.gpu import cloud_march as cm

    @wp.kernel
    def march_ir(
        origin: wp.array(dtype=wp.vec3),
        direction: wp.array(dtype=wp.vec3),
        near: wp.array(dtype=float),
        width: wp.array(dtype=float),
        jitter: wp.array(dtype=float),
        active: wp.array(dtype=int),
        steps: int,
        extinction: float,
        ratio: float,
        opaque_depth: float,
        table: wp.array(dtype=float),
        table_h0: float,
        table_dh: float,
        table_n: int,
        L: cm.Layer,
        weather: wp.Texture2D,
        shape: wp.Texture3D,
        detail: wp.Texture3D,
        patches: wp.Texture3D,
        out_optical: wp.array(dtype=float),
        out_transmittance: wp.array(dtype=float),
        out_radiance: wp.array(dtype=float),
        out_range_sum: wp.array(dtype=float),
        out_range_weight: wp.array(dtype=float),
        out_height_sum: wp.array(dtype=float),
        out_height_weight: wp.array(dtype=float),
    ):
        i = wp.tid()
        optical = float(0.0)  # noqa: UP018 -- Warp's dynamic variable
        transmittance = float(1.0)  # noqa: UP018 -- Warp's dynamic variable
        # The sky march weights its emission height by the *visible* transmittance (the CPU
        # deck's `march`, kept as it is: this is its twin, not its reviewer); the march to the
        # hit weights by the band's. Both are carried.
        visible = float(1.0)  # noqa: UP018 -- Warp's dynamic variable
        radiance = float(0.0)  # noqa: UP018 -- Warp's dynamic variable
        range_sum = float(0.0)  # noqa: UP018 -- Warp's dynamic variable
        range_weight = float(0.0)  # noqa: UP018 -- Warp's dynamic variable
        height_sum = float(0.0)  # noqa: UP018 -- Warp's dynamic variable
        height_weight = float(0.0)  # noqa: UP018 -- Warp's dynamic variable
        if active[i] != 0:
            o = origin[i]
            d = direction[i]
            w = width[i]
            j = jitter[i]
            t0 = near[i]
            for k in range(steps):
                t = t0 + (float(k) + j) * w
                p = o + d * t
                # Height above the curved ground, as the visible kernel and the CPU deck take it.
                alt = cm.altitude_of(p)
                rho = cm.cloud_density(p[0], alt, p[2], L, weather, shape, detail, patches)
                sigma = rho * extinction
                d_tau = sigma * w
                absorbed = 1.0 - wp.exp(-ratio * d_tau)
                # march_to's weight: what this sample's emission contributes at the sensor.
                weight = transmittance * absorbed
                # The emission's radiance at this height, from the host's table.
                u = (alt - table_h0) / table_dh
                u = wp.clamp(u, 0.0, float(table_n - 1))
                lo = int(wp.floor(u))
                hi = wp.min(lo + 1, table_n - 1)
                f = u - float(lo)
                value = table[lo] * (1.0 - f) + table[hi] * f
                radiance += weight * value
                range_sum += weight * t
                range_weight += weight
                # march's weight: extinction times the visible transmittance back along the ray.
                hw = sigma * visible * w
                height_sum += hw * alt
                height_weight += hw
                optical += d_tau
                transmittance = transmittance * (1.0 - absorbed)
                visible = visible * wp.exp(-d_tau)
                if optical * ratio >= opaque_depth:
                    break
        out_optical[i] = optical
        out_transmittance[i] = transmittance
        out_radiance[i] = radiance
        out_range_sum[i] = range_sum
        out_range_weight[i] = range_weight
        out_height_sum[i] = height_sum
        out_height_weight[i] = height_weight

    _KERNELS["march"] = march_ir
    return march_ir


@dataclass
class GpuWeatherFxDeck(WeatherFxDeck):
    """:class:`~irsim.atmosphere.weather_fx.WeatherFxDeck` whose two marches run on the GPU.

    Wraps a weather-fx ``Cloudscape`` (the only source with a Warp density); a ``CloudField``
    is refused, since its grid has no GPU evaluation upstream, and the caller keeps the CPU deck
    for it. Everything a sky model or a bridge reads -- base, top, cover, pitch, origin, the
    band's ratio, the density scale -- is the parent's; only :meth:`march` and :meth:`march_to`
    are replaced, and they take the same arguments and return the same dataclasses.
    """

    device: str = "cuda:0"
    _renderer: Any = field(default=None, init=False, repr=False)
    _layer_key: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if not hasattr(self.field, "kernel_constants"):
            raise TypeError(
                "GpuWeatherFxDeck marches a weather-fx Cloudscape; a CloudField has no GPU "
                "density upstream -- keep the CPU WeatherFxDeck for it"
            )

    def _textures(self) -> Any:
        """weather-fx's ``CloudRenderer`` holds the cloudscape's textures and layer struct on
        the device; it is built once per deck."""
        if self._renderer is None:
            ensure_weather_fx_on_path()
            from weather_fx.gpu import cloud_march as cm

            self._renderer = cm.CloudRenderer(self.field, device=self.device)
        return self._renderer

    # -- the two marches ------------------------------------------------------------------

    def _launch(
        self,
        elevation_rad: Any,
        azimuth_rad: Any,
        range_m: Any,
        radiance_at_height: Callable[[NDArray[np.float64]], Any] | None,
        *,
        origin_m: tuple[float, float, float] | None,
        steps: int | None,
        to_hit: bool,
    ) -> tuple[Any, ...]:
        wp = _warp()
        renderer = self._textures()
        el = np.asarray(elevation_rad, dtype=np.float64)
        az = np.asarray(azimuth_rad, dtype=np.float64)
        rng = np.asarray(range_m, dtype=np.float64)
        el, az, rng = np.broadcast_arrays(el, az, rng)
        shape = el.shape
        direction = np.stack(
            [np.cos(el) * np.sin(az), np.sin(el), -np.cos(el) * np.cos(az)], axis=-1
        )
        origin = np.zeros(direction.shape, dtype=np.float64)
        origin[...] = np.asarray(self.origin_m if origin_m is None else origin_m, dtype=np.float64)
        near, far = shell_span(
            np.asarray(origin, dtype=np.float64),
            np.asarray(direction, dtype=np.float64),
            self.base_m,
            self.top_m,
        )
        far = np.minimum(far, self.max_range_m)
        if to_hit:
            end = np.minimum(np.minimum(far, near + self.max_path_m), rng)
            hit = end > near
            span = np.where(hit, end - near, 0.0)
        else:
            hit = far > near
            span = np.where(hit, np.minimum(far - near, self.max_path_m), 0.0)
        zeros = np.zeros(shape, dtype=np.float64)
        if not np.any(hit) or self.field.cover <= 0.0:
            return (zeros, np.ones(shape), zeros, zeros, zeros, zeros, zeros)
        n = max(int(steps if steps is not None else (self.steps or self.steps_for(span))), 1)
        jitter = ray_hash(direction)
        width = span / n
        base, top = float(self.base_m), float(self.top_m)
        heights = np.asarray(np.linspace(base, top, RADIANCE_TABLE_POINTS), dtype=np.float64)
        if radiance_at_height is None:
            table = np.zeros(heights.shape, dtype=np.float64)
        else:
            table = np.asarray(radiance_at_height(heights), dtype=np.float64)
            table = np.asarray(np.broadcast_to(table, heights.shape), dtype=np.float64)
        count = int(np.prod(shape))

        def f32(a: Any) -> Any:
            return wp.array(np.ascontiguousarray(a.reshape(-1), dtype=np.float32), dtype=float)

        def vec(a: Any) -> Any:
            return wp.array(np.ascontiguousarray(a.reshape(-1, 3), dtype=np.float32), dtype=wp.vec3)

        with wp.ScopedDevice(renderer.device):
            outs = [wp.zeros(count, dtype=float) for _ in range(7)]
            wp.launch(
                _kernel(),
                dim=count,
                inputs=[
                    vec(origin),
                    vec(direction),
                    f32(near),
                    f32(width),
                    f32(jitter),
                    wp.array(np.ascontiguousarray(hit.reshape(-1).astype(np.int32)), dtype=int),
                    int(n),
                    float(self.field.extinction_per_m) * float(self.density_scale),
                    float(self.od_ratio),
                    float(OPAQUE_OPTICAL_DEPTH),
                    f32(table),
                    float(heights[0]),
                    float(heights[1] - heights[0]),
                    int(RADIANCE_TABLE_POINTS),
                    renderer.layer,
                    renderer.weather,
                    renderer.shape,
                    renderer.detail,
                    renderer.patches,
                ]
                + outs,
            )
            wp.synchronize_device(renderer.device)
            host = [np.asarray(o.numpy(), dtype=np.float64).reshape(shape) for o in outs]
        return tuple(host)

    def march(
        self,
        elevation_rad: Any,
        azimuth_rad: Any,
        *,
        origin_m: tuple[float, float, float] | None = None,
        steps: int | None = None,
    ) -> MarchResult:
        """The sky march (optical depth and emission height), as the CPU deck's."""
        optical, _t, _r, _rs, _rw, height_sum, height_weight = self._launch(
            elevation_rad, azimuth_rad, np.inf, None, origin_m=origin_m, steps=steps, to_hit=False
        )
        above_base = np.where(
            height_weight > 1e-12,
            height_sum / np.maximum(height_weight, 1e-12) - self.base_m,
            0.0,
        )
        return MarchResult(optical_depth=optical, emission_height_m=np.maximum(above_base, 0.0))

    def march_to(
        self,
        elevation_rad: Any,
        azimuth_rad: Any,
        range_m: Any,
        radiance_at_height: Callable[[NDArray[np.float64]], Any],
        *,
        origin_m: tuple[float, float, float] | None = None,
        steps: int | None = None,
    ) -> OccludedMarch:
        """The march to each ray's hit (AT.14), as the CPU deck's, with ``radiance_at_height``
        tabulated on :data:`RADIANCE_TABLE_POINTS` heights across the layer."""
        optical, transmittance, radiance, range_sum, range_weight, _h, _w = self._launch(
            elevation_rad,
            azimuth_rad,
            range_m,
            radiance_at_height,
            origin_m=origin_m,
            steps=steps,
            to_hit=True,
        )
        emission_range = np.where(
            range_weight > 1e-12, range_sum / np.maximum(range_weight, 1e-12), 0.0
        )
        return OccludedMarch(optical, transmittance, radiance, emission_range)

    def __repr__(self) -> str:  # pragma: no cover - diagnostics
        return (
            f"GpuWeatherFxDeck(device={self.device!r}, base_m={self.base_m:.0f}, "
            f"cover={self.cover:.2f})"
        )
