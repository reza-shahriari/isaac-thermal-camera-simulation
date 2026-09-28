"""Point-target radiometry: analytic injection of targets smaller than one native pixel (MS.6).

A target of area A_t at range R fills the fraction φ = A_t f² / (R² A_pix) of a native pixel's
footprint. Rasterising it on the k× supersampled grid is badly biased below a few pixels (the
measured flux error per sub-pixel phase is recorded in ADR 0071), so below one native pixel the
target is injected analytically. Its **excess radiance over the background** the pixel would
otherwise see (the sky beyond it along the same ray, ADR 0071) is

    ΔL = φ · Σ_k w_k τ_k(R, θ) [L_t − L_beyond,k(R, θ)]          (layered atmosphere, per class)
       = φ · τ(R) [L_t − L_air]                                 (grey atmosphere)
       = φ · [L_t − L_background]                               (no atmosphere)

and the excess power on the pixel is Ω_eff(F) τ_opt A_d ΔL with Ω_eff = π/(4F² + 1) imported
from irsim.optics.aperture (CLAUDE.md #5) -- the fill-fraction form equals the resolved-path
formula at φ = 1 by construction. The excess is splatted bilinearly at the sub-pixel position on
the k× grid *before* stage 3, so MS.4's optical PSF and the same box downsample the resolved
side uses spread it; the splat conserves flux at every phase.

docs/physics-model.md §8.1, §8.3, §2; ADR 0071
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from irsim.atmosphere.layered import LayeredAtmosphere
from irsim.atmosphere.model import Atmosphere
from irsim.config.sensor import SensorSpec
from irsim.optics.aperture import pixel_power
from irsim.radiometry.lut import BandLUT, Quantity

__all__ = [
    "HANDOFF_FILL_FRACTION",
    "PointTarget",
    "fill_fraction",
    "excess_radiance",
    "excess_power",
    "splat",
    "inject_point_targets",
]

HANDOFF_FILL_FRACTION = 1.0  # below one native pixel: inject analytically (ADR 0071)


@dataclass(frozen=True)
class PointTarget:
    """An unresolved target: projected area, range, in-band radiance (already ε L_B + (1−ε) L_env),
    image position in native pixel coordinates (0.5, 0.5 = centre of the first pixel), and the
    ray elevation for the atmosphere."""

    area_m2: float
    range_m: float
    radiance: float
    position_px: tuple[float, float]  # (x, y)
    elevation_rad: float = 0.0

    def __post_init__(self) -> None:
        if self.area_m2 <= 0.0 or self.range_m <= 0.0 or self.radiance < 0.0:
            raise ValueError("area and range must be positive, radiance non-negative")


def fill_fraction(
    area_m2: float, range_m: float, focal_length_m: float, pixel_area_m2: float
) -> float:
    """φ = A_t f² / (R² A_pix): the target's share of one native pixel's footprint."""
    if min(area_m2, range_m, focal_length_m, pixel_area_m2) <= 0.0:
        raise ValueError("all geometric quantities must be positive")
    return area_m2 * focal_length_m**2 / (range_m**2 * pixel_area_m2)


def excess_radiance(
    target: PointTarget,
    sensor: SensorSpec,
    atmosphere: Atmosphere | LayeredAtmosphere | None,
    band: str,
    t_s: float,
    lut: BandLUT,
    quantity: Quantity = "lb",
    background_radiance: float | None = None,
) -> float:
    """ΔL averaged over the native pixel (band radiance units), per the module docstring.

    ``background_radiance`` is what the pixel holds *after stage 2* -- sky, cloud, sea, terrain
    or another target, each already carried through its own path. Given, the excess is

        ΔL = φ [ τ_B(R) L_t + L_path(R) − L_pixel ],

    the target replacing its share of the pixel with its own radiance seen through R of air (the
    blend `rotor_veil` argues for). With ``None``, and always for the grey L1 model, the layered
    branch falls back to the clear column beyond R -- the same number on a clear-sky pixel and
    wrong on every other one (AT.22): over a cloud the old form subtracted a clear sky that was
    not there and made a sub-pixel drone about twice as bright over its background as it should be.
    """
    phi = fill_fraction(
        target.area_m2, target.range_m, sensor.optics.focal_length_mm * 1e-3, sensor.pixel_area_m2
    )
    if phi >= HANDOFF_FILL_FRACTION:
        raise ValueError(
            f"target fills {phi:.2f} native pixels: rasterise it on the G-buffer instead "
            "(the point-target path is for phi < 1, ADR 0071)"
        )
    if atmosphere is None:
        return phi * (target.radiance - (background_radiance or 0.0))
    if isinstance(atmosphere, LayeredAtmosphere):
        es = atmosphere.exponential_sum(band, t_s)
        tau_k = atmosphere.class_transmittances(band, t_s, target.range_m, target.elevation_rad)
        if background_radiance is not None:
            tau_band = float(np.dot(es.weights, tau_k))
            path = atmosphere.path_radiance(
                band, t_s, target.range_m, target.elevation_rad, quantity
            )
            return phi * (tau_band * target.radiance + path - background_radiance)
        beyond = atmosphere.sky_beyond_per_class(
            band, t_s, target.range_m, target.elevation_rad, quantity
        )
        return phi * float(np.dot(es.weights * tau_k, target.radiance - beyond))
    # The grey L1 model keeps the clear-column form whatever the pixel holds. Its path radiance
    # tends to L_B(T_air) while the sky pixels beside it come from `SkyModel` (ADR 0050), so the
    # two are not one column and "what is beyond R" cannot be read off the pixel: at long range
    # (L_pixel - L_path) / tau goes negative. That is the grey model's own inconsistency (spec
    # issue S28, AT.5), not something a point target can repair.
    state = atmosphere.state(t_s)
    tau = float(np.exp(-state.gamma_per_m[band] * target.range_m))
    l_air = float(lut.lookup(np.float64(state.t_air_k), quantity)[()])
    return phi * tau * (target.radiance - l_air)


def excess_power(excess_radiance_value: float, sensor: SensorSpec, cos4: float = 1.0) -> float:
    """Φ_excess = Ω_eff(F) τ_opt A_d ΔL (W or photons s⁻¹) -- the aperture factor from one place."""
    return float(
        pixel_power(
            np.asarray(excess_radiance_value, dtype=np.float64),
            sensor.optics.f_number,
            sensor.optics.transmittance,
            sensor.detector_active_area_m2,
            cos4,
        )
    )


def splat(
    radiance_ss: NDArray[np.floating],
    excess: float,
    position_px: tuple[float, float],
    supersample: int,
) -> NDArray[np.floating]:
    """Add a native-pixel-averaged excess ΔL at a sub-pixel position on the k× grid: bilinear over
    the four nearest supersample cells with total k² ΔL, so the box downsample returns ΔL spread
    over at most four native pixels and the sum over the frame is exactly ΔL at every phase."""
    out = np.array(radiance_ss, dtype=np.float64, copy=True)
    k = int(supersample)
    x = float(position_px[0]) * k - 0.5  # supersample cell centres sit at (i + 0.5)/k native
    y = float(position_px[1]) * k - 0.5
    x0, y0 = int(np.floor(x)), int(np.floor(y))
    fx, fy = x - x0, y - y0
    h, w = out.shape[:2]
    total = excess * k * k
    for dy, wy in ((0, 1.0 - fy), (1, fy)):
        for dx, wx in ((0, 1.0 - fx), (1, fx)):
            i, j = y0 + dy, x0 + dx
            if 0 <= i < h and 0 <= j < w and wx * wy > 0.0:
                out[i, j] += total * wx * wy
            elif wx * wy > 0.0:
                raise ValueError(f"point target at {position_px} falls outside the frame")
    return np.asarray(out, dtype=radiance_ss.dtype)


def inject_point_targets(
    radiance_ss: NDArray[np.floating],
    targets: Sequence[PointTarget],
    sensor: SensorSpec,
    atmosphere: Atmosphere | LayeredAtmosphere | None,
    band: str,
    t_s: float,
    lut: BandLUT,
    quantity: Quantity,
    supersample: int,
    background: NDArray[np.floating] | None = None,
) -> NDArray[np.floating]:
    """Inject every target on the post-stage-2 k× radiance plane. The background the target
    occults is the plane's own value at the target's pixel -- whatever is there (AT.22): it used
    to be read only without an atmosphere, and with one the clear column beyond R stood in for
    it, cloud, sea or terrain notwithstanding."""
    out: Any = radiance_ss
    src = radiance_ss if background is None else background
    k = int(supersample)
    for t in targets:
        j = min(max(int(t.position_px[0] * k), 0), src.shape[1] - 1)
        i = min(max(int(t.position_px[1] * k), 0), src.shape[0] - 1)
        bg = float(src[i, j])
        d_l = excess_radiance(t, sensor, atmosphere, band, t_s, lut, quantity, bg)
        out = splat(out, d_l, t.position_px, supersample)
    return np.asarray(out, dtype=radiance_ss.dtype)
