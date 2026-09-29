"""Read `isaac-weather-fx`'s cloud field through this project's own deck contract.

The submodule at ``third_party/isaac-weather-fx`` owns the sky: where the sun and the moon are,
what the cloud looks like, what kind of day it is. This module is the seam that lets the infrared
band integrate **that** cloud instead of a second one of its own.

**Why a seam and not a rewrite.** `SkyModel.radiance_field_from_deck` asks a deck for exactly two
things per ray -- the optical depth it accumulated and the height the emission came from -- and
does the radiometry itself. That contract is narrow enough to satisfy from a foreign density
field, so the whole of this project's infrared physics (the band LUT, the clear-sky column, the
atmosphere in front of the cloud, the apparent-temperature inversion) is reused unchanged. What
changes is only *which array* says where the cloud is.

**And that is the point.** Before this, the infrared band marched
:class:`~irsim.atmosphere.cloud_deck.CloudDeck` while the visible companion baked a dome from a
different object, and the two agreed only as well as two implementations of the same idea ever
do -- measured at 26 % of pixels against 57 % at one point (AT.15). Now both bands sample one
``CloudField``: the visible dome through weather-fx's own renderer backend, the infrared band
through this adapter. They cannot disagree about where a cloud is, because there is one cloud.

**The engine-free rule still holds.** ``weather_fx.core`` is pure Python and numpy with no
Omniverse imports -- that is a property its own test suite enforces -- so importing it here does
not breach CLAUDE.md #1, which is about engine modules. The import is still done lazily and with
a message that says what to do, because the submodule may not be initialised.
"""

from __future__ import annotations

import math
import pathlib
import sys
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from irsim.atmosphere.cloud import CLOUD_OD_RATIO
from irsim.atmosphere.cloud_deck import MarchResult

__all__ = [
    "WEATHER_FX_PACKAGE",
    "OccludedMarch",
    "WeatherFxDeck",
    "cloud_field_from_spec",
    "ensure_weather_fx_on_path",
    "ray_hash",
    "weather_fx_available",
]

#: Where the submodule keeps its importable package. It is a Kit extension rather than a wheel,
#: so the path is the extension directory and not a site-packages entry.
WEATHER_FX_PACKAGE = (
    pathlib.Path(__file__).resolve().parents[3]
    / "third_party"
    / "isaac-weather-fx"
    / "exts"
    / "weather.fx"
)


def ensure_weather_fx_on_path() -> None:
    """Put the submodule's package directory on ``sys.path`` if it is not already importable.

    The same pattern as :func:`irsim_isaac.env.ensure_warp_on_path`: a dependency that ships as an
    extension directory rather than as an installed distribution has to be found rather than
    imported. Idempotent, and a no-op once the package is installed properly.
    """
    try:
        import weather_fx.core  # noqa: F401

        return
    except ImportError:
        pass
    if not WEATHER_FX_PACKAGE.is_dir():
        raise ImportError(
            f"the isaac-weather-fx submodule is not checked out at {WEATHER_FX_PACKAGE}.\n"
            "  run: git submodule update --init third_party/isaac-weather-fx"
        )
    path = str(WEATHER_FX_PACKAGE)
    if path not in sys.path:
        sys.path.insert(0, path)


def weather_fx_available() -> bool:
    """Whether the submodule can be imported, without raising if it cannot."""
    try:
        ensure_weather_fx_on_path()
        import weather_fx.core  # noqa: F401
    except ImportError:
        return False
    return True


def cloud_field_from_spec(
    *,
    cover: float,
    genus: str = "cumulus",
    base_m: float = 0.0,
    temperature_c: float = 20.0,
    dewpoint_c: float = 10.0,
    thickness_m: float = 0.0,
    optical_depth: float = 0.0,
    feature_m: float = 400.0,
    cells: int = 256,
    levels: int = 48,
    cell_m: float = 60.0,
    seed: int = 0,
) -> Any:
    """Build a ``weather_fx.core.clouds.CloudField`` from plain numbers.

    The arguments are exactly the ``clouds`` section of a weather-fx state, so a scene config and
    the extension's own UI describe a cloud the same way and nothing has to be translated twice.
    A ``base_m`` of zero means "compute it from the temperature and dew point", which is why a
    whole field of cumulus has its bases on one level.
    """
    ensure_weather_fx_on_path()
    from weather_fx.core.clouds import CloudField, cloud_profile, lifting_condensation_level_m

    base = float(base_m)
    if base <= 0.0:
        base = lifting_condensation_level_m(float(temperature_c), float(dewpoint_c))
    return CloudField(
        cover=float(cover),
        base_m=base,
        profile=cloud_profile(str(genus)),
        cell_m=float(cell_m),
        cells=int(cells),
        levels=int(levels),
        seed=int(seed),
        thickness_m=float(thickness_m),
        optical_depth=float(optical_depth),
        feature_m=float(feature_m),
    )


#: Transmittance below which a ray is opaque for every purpose this project has: e^-12 of the
#: sky behind it is 1e-5 of a radiance the band LUT resolves to 1e-4.
OPAQUE_TRANSMITTANCE = math.exp(-12.0)
#: The same limit as an optical depth, for a march that accumulates depth rather than product.
OPAQUE_OPTICAL_DEPTH = 12.0


def ray_hash(direction: Any) -> NDArray[np.float64]:
    """A per-ray offset in [0, 1), decorrelated between neighbouring rays.

    The classic fractional-sine hash on the direction vector. What matters is not its quality as
    a random number but that it is **not smooth in the direction**: a jitter that varies slowly
    across the frame moves every neighbouring ray's samples together, and the quadrature error
    then has the same sign over whole regions -- which is the ring pattern this replaces.
    Deterministic, so a frame is reproducible.
    """
    d = np.asarray(direction, dtype=np.float64)
    phase = d[..., 0] * 12.9898 + d[..., 1] * 78.233 + d[..., 2] * 37.719
    value = np.sin(phase) * 43758.5453123
    return np.asarray(value - np.floor(value), dtype=np.float64)


@dataclass(frozen=True)
class OccludedMarch:
    """What a ray took from the cloud between the camera and **whatever it hit** (AT.14).

    The sky-only :class:`~irsim.atmosphere.cloud_deck.MarchResult` answers "how much cloud, and
    where did the emission come from" for a ray that ends at infinity. This answers the
    Schwarzschild question for a ray that ends at a range -- a drone, a wing, the sea -- so the
    surface behind the cloud can be attenuated by it and the cloud's own emission laid in front:
    ``L = transmittance · L_hit + path_radiance`` (docs/clouds-in-the-infrared.md).
    """

    #: Visible optical depth accumulated between the camera and the hit (not the band's).
    optical_depth: NDArray[np.float64]
    #: The **band's** transmittance through that cloud, ``exp(−od_ratio · τ_vis)`` sample by
    #: sample. 1 where the ray met no cloud short of its hit.
    transmittance: NDArray[np.float64]
    #: The band radiance the cloud emits toward the camera along the ray, each sample at its own
    #: height's temperature and attenuated by the cloud in front of it. 0 without cloud.
    path_radiance: NDArray[np.float64]
    #: Range from the camera at which that emission is centred, weighted by what reaches the
    #: sensor, metres -- what the air in front of the cloud is integrated over. 0 without cloud.
    emission_range_m: NDArray[np.float64]

    @property
    def emissivity(self) -> NDArray[np.float64]:
        return np.asarray(1.0 - self.transmittance, dtype=np.float64)


@dataclass
class WeatherFxDeck:
    """A weather-fx ``CloudField``, presented through this project's deck march contract.

    Satisfies what :meth:`irsim.atmosphere.sky.SkyModel.radiance_field_from_deck` needs -- a
    ``march`` returning optical depth and emission height -- so the infrared radiometry is
    untouched and only the geometry comes from elsewhere.
    """

    field: Any
    #: Ratio of the band's extinction to the visible one. A cloud carries one optical depth and
    #: each band derives what it needs from it (ADR 0126), which is the same convention the
    #: native deck uses; it is kept here so an infrared caller can override it per band.
    od_ratio: float = CLOUD_OD_RATIO
    #: Steps the march takes when a caller fixes them. ``None`` -- the default -- sizes the march
    #: from the geometry instead (:meth:`steps_for`), which is what an infrared frame wants: a
    #: ray at 18 degrees crosses three times the slab a ray at 60 degrees does.
    steps: int | None = None
    #: Bounds on the adaptive step count. The floor matches weather-fx's own march; the cap is
    #: what keeps a horizon-grazing frame from costing minutes.
    min_steps: int = 64
    max_steps: int = 512
    #: Longest path followed through the slab, metres. Beyond it the transmittance of anything
    #: this field can hold has underflowed, and a level ray would otherwise march forever.
    max_path_m: float = 12_000.0
    #: Where the rays start, in the field's own Y-up frame, metres (AT.30): the camera, as the
    #: dome bake and the volume tiles place it (the anchor less the wind's drift, through
    #: ``stage_to_field``). A march that passes no ``origin_m`` starts here, so the sky pixels,
    #: the geometry pixels and the visible dome all read the field from one point.
    origin_m: tuple[float, float, float] = (0.0, 0.0, 0.0)

    @property
    def base_m(self) -> float:
        return float(self.field.base_m)

    @property
    def top_m(self) -> float:
        return float(self.field.top_m)

    @property
    def thickness_m(self) -> float:
        return float(self.field.thickness_m)

    @property
    def optical_depth(self) -> float:
        return float(self.field.optical_depth)

    @property
    def cover(self) -> float:
        """The sky fraction the field actually covers, measured rather than requested."""
        return float(self.field.measured_cover)

    @property
    def sample_pitch_m(self) -> float:
        """The finest spacing the grid carries: the smaller of a cell and a level."""
        return float(min(self.field.cell_m, self.field.thickness_m / self.field.levels))

    def steps_for(self, span_m: Any) -> int:
        """Steps that put two samples per grid pitch along the longest ray in ``span_m``.

        Two per pitch is the Nyquist spacing of a trilinear field: one per pitch left the band
        emissivity 0.019 out at the 99th percentile against a dense reference on the test deck,
        two leaves it under 0.01, which is a quarter kelvin of cloud. One count for the whole
        array, because a vectorised march wants one loop; it is set by the longest crossing, so
        the shallowest ray in a frame is sampled at that spacing and the steep ones finer.
        Bounded by :attr:`min_steps` and :attr:`max_steps`.
        """
        arr = np.asarray(span_m, dtype=np.float64)
        longest = float(np.max(arr)) if arr.size else 0.0
        want = math.ceil(2.0 * longest / max(self.sample_pitch_m, 1.0))
        return int(min(self.max_steps, max(self.min_steps, want)))

    def march(
        self,
        elevation_rad: Any,
        azimuth_rad: Any,
        *,
        origin_m: tuple[float, float, float] | None = None,
        steps: int | None = None,
    ) -> MarchResult:
        """Integrate the field along each ray and answer in this project's own terms.

        **The array is shared; the quadrature is this band's own (ADR 0146).** weather-fx's
        ``CloudField.march`` places its samples geometrically along the ray and offsets them by
        a jitter that is a *smooth* function of the ray direction, which is right for a dome
        baked once and read at a texel's blur, and wrong for a per-pixel infrared frame: the
        sampling error is then coherent across neighbouring pixels and prints as rings and bands
        through every cloud -- measured at 3.6 K at the 99th percentile against a 512-step
        reference, in concentric contours. This march is **stratified and uniform** -- one sample
        per grid pitch along the longest ray, every ray offset by a hash of its own direction --
        so what error remains is white and below the detector's noise, and the same density
        array is integrated, so the two bands still cannot disagree about *where* the cloud is.

        **The two direction conventions agree, and that is checked rather than assumed.** Both
        projects place a direction as ``[cos el sin az, sin el, -cos el cos az]`` -- +Y up, -Z at
        azimuth zero, +X at ninety. weather-fx writes it that way in ``BodyPosition.direction``;
        this project writes it that way in ``irsim_isaac.visible_sky.stage_direction``. A test
        pins the pair, because a silent disagreement here rotates the whole cloud field about
        the observer and looks entirely plausible.

        The emission height is reported **above the cloud base**, because the temperature this
        project lapses from is the base's; weather-fx reports heights above the ground.
        """
        el = np.asarray(elevation_rad, dtype=np.float64)
        az = np.asarray(azimuth_rad, dtype=np.float64)
        el, az = np.broadcast_arrays(el, az)
        direction = np.stack(
            [
                np.cos(el) * np.sin(az),
                np.sin(el),
                -np.cos(el) * np.cos(az),
            ],
            axis=-1,
        )
        origin = np.zeros(direction.shape, dtype=np.float64)
        origin[...] = np.asarray(self.origin_m if origin_m is None else origin_m, dtype=np.float64)

        near, far = self.field.slab_span(origin, direction)
        hit = far > near
        span = np.where(hit, np.minimum(far - near, self.max_path_m), 0.0)
        shape = span.shape
        optical = np.zeros(shape, dtype=np.float64)
        height_sum = np.zeros(shape, dtype=np.float64)
        height_weight = np.zeros(shape, dtype=np.float64)
        if not np.any(hit) or self.field.cover <= 0.0:
            return MarchResult(optical_depth=optical, emission_height_m=height_sum)

        n = int(steps if steps is not None else (self.steps or self.steps_for(span)))
        n = max(n, 1)
        jitter = ray_hash(direction)
        width = span / n
        transmittance = np.ones(shape, dtype=np.float64)
        extinction = float(self.field.extinction_per_m)
        dx, dy, dz = direction[..., 0], direction[..., 1], direction[..., 2]
        ox, oy, oz = origin[..., 0], origin[..., 1], origin[..., 2]
        # Only the rays that can still see are sampled. A ray that has gone opaque -- which in a
        # cumulus field is most of the cloudy ones within a few hundred metres of entry -- adds
        # nothing the sensor can tell apart, so it drops out of the gather; on a broken-cloud
        # frame that is about half the work.
        active = hit.copy()
        for k in range(n):
            idx = np.nonzero(active)
            if idx[0].size == 0:
                break
            t = near[idx] + (k + jitter[idx]) * width[idx]
            px, py, pz = ox[idx] + dx[idx] * t, oy[idx] + dy[idx] * t, oz[idx] + dz[idx] * t
            sigma = np.asarray(self.field.density(px, py, pz), dtype=np.float64) * extinction
            d_tau = sigma * width[idx]
            # What reaches the sensor from this sample: extinction times the transmittance back
            # along the ray. Weighting by extinction alone would average in cloud it cannot see.
            weight = sigma * transmittance[idx] * width[idx]
            height_sum[idx] += weight * py
            height_weight[idx] += weight
            optical[idx] += d_tau
            transmittance[idx] = transmittance[idx] * np.exp(-d_tau)
            # Opaque in the *band*, not in the visible: the band's depth is `od_ratio` times
            # the visible one, and dropping out at visible e^-12 capped an opaque cloud's LWIR
            # emissivity at 1 - e^-6 (AT.14 found it against the finite-range march).
            active[idx] = optical[idx] * float(self.od_ratio) < OPAQUE_OPTICAL_DEPTH

        above_base = np.where(
            height_weight > 1e-12,
            height_sum / np.maximum(height_weight, 1e-12) - self.base_m,
            0.0,
        )
        return MarchResult(
            optical_depth=optical,
            emission_height_m=np.maximum(above_base, 0.0),
        )

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
        """Integrate absorption **and emission** along each ray up to ``range_m`` (AT.14).

        The same stratified, hashed quadrature as :meth:`march`, ended at the ray's own hit
        instead of the far side of the slab, and carrying the band along: every sample
        absorbs ``1 − e^{−od_ratio σ Δs}`` of what is behind it and emits that fraction of
        ``radiance_at_height(height_m)`` toward the camera, attenuated by the cloud already
        crossed. That is the Schwarzschild integral for a non-scattering medium, discretised
        per sample and marched from the camera outward (docs/clouds-in-the-infrared.md; Petty
        ch. 8). Scattering enters only through ``od_ratio`` -- the band's absorption is a fixed
        fraction of the visible extinction -- which is the standard operational approximation
        (Chou et al. 1999) and the same one the sky pixels already make.

        ``range_m`` is per ray and may be ``inf`` (a sky pixel); a ray whose hit lies short of
        the slab meets no cloud and answers transmittance 1, radiance 0. ``radiance_at_height``
        takes heights above the ground in metres and returns the band radiance the cloud emits
        there; the sky model supplies its own base temperature and in-cloud lapse.
        """
        el = np.asarray(elevation_rad, dtype=np.float64)
        az = np.asarray(azimuth_rad, dtype=np.float64)
        rng = np.asarray(range_m, dtype=np.float64)
        el, az, rng = np.broadcast_arrays(el, az, rng)
        direction = np.stack(
            [np.cos(el) * np.sin(az), np.sin(el), -np.cos(el) * np.cos(az)], axis=-1
        )
        origin = np.zeros(direction.shape, dtype=np.float64)
        origin[...] = np.asarray(self.origin_m if origin_m is None else origin_m, dtype=np.float64)

        near, far = self.field.slab_span(origin, direction)
        end = np.minimum(np.minimum(far, near + self.max_path_m), rng)
        hit = end > near
        span = np.where(hit, end - near, 0.0)
        shape = span.shape
        optical = np.zeros(shape, dtype=np.float64)
        transmittance = np.ones(shape, dtype=np.float64)
        radiance = np.zeros(shape, dtype=np.float64)
        range_sum = np.zeros(shape, dtype=np.float64)
        range_weight = np.zeros(shape, dtype=np.float64)
        if not np.any(hit) or self.field.cover <= 0.0:
            return OccludedMarch(optical, transmittance, radiance, range_sum)

        n = max(int(steps if steps is not None else (self.steps or self.steps_for(span))), 1)
        jitter = ray_hash(direction)
        width = span / n
        extinction = float(self.field.extinction_per_m)
        ratio = float(self.od_ratio)
        dx, dy, dz = direction[..., 0], direction[..., 1], direction[..., 2]
        ox, oy, oz = origin[..., 0], origin[..., 1], origin[..., 2]
        active = hit.copy()
        for k in range(n):
            idx = np.nonzero(active)
            if idx[0].size == 0:
                break
            t = near[idx] + (k + jitter[idx]) * width[idx]
            px, py, pz = ox[idx] + dx[idx] * t, oy[idx] + dy[idx] * t, oz[idx] + dz[idx] * t
            d_tau = np.asarray(self.field.density(px, py, pz), dtype=np.float64) * (
                extinction * width[idx]
            )
            absorbed = -np.expm1(-ratio * d_tau)
            # What this sample contributes at the sensor: its own emission, absorbed here,
            # through the cloud already crossed. The weight is the same quantity, so the
            # emission range is where the radiance the sensor sees actually came from.
            weight = transmittance[idx] * absorbed
            radiance[idx] += weight * np.asarray(radiance_at_height(py), dtype=np.float64)
            range_sum[idx] += weight * t
            range_weight[idx] += weight
            optical[idx] += d_tau
            transmittance[idx] = transmittance[idx] * (1.0 - absorbed)
            active[idx] = transmittance[idx] > OPAQUE_TRANSMITTANCE

        emission_range = np.where(
            range_weight > 1e-12, range_sum / np.maximum(range_weight, 1e-12), 0.0
        )
        return OccludedMarch(optical, transmittance, radiance, emission_range)

    def adequate_steps(self, elevation_rad: Any) -> int:
        """How many steps this field wants for the shallowest ray in the array.

        Present so a caller written against the native deck keeps working; it is
        :meth:`steps_for` evaluated on the slab crossing of the lowest ray.
        """
        el = np.abs(np.asarray(elevation_rad, dtype=np.float64))
        positive = el[el > 0.0]
        lowest = float(np.min(positive)) if positive.size else 0.5 * math.pi
        lowest = max(lowest, math.radians(2.0))
        return self.steps_for(min(self.thickness_m / math.sin(lowest), self.max_path_m))
