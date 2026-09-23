"""SkyModel: sky radiance vs elevation, the tilt-integrated sky, the broadband sky (ADR 0044).

Built on the layered atmosphere (MS.1): the clear sky is the column's own emission,
L_clear,B(θ) = L_path,B(∞, θ). One object serves both paths (ADR 0035):

* ``radiance(θ_el)``            -- in-band sky radiance for the image (the sky background, the
                                   reflected term): a 1-D elevation LUT (0.5° grid, linear) over
                                   the layered model, the *fast path* the Isaac adapter calls;
* ``effective_radiance(β)``     -- L_sky,eff for a plane of tilt β (0 = facing up): the cosine-
                                   weighted mean over the sky the plane sees, from a 1-D tilt LUT
                                   whose azimuth integral is analytic;
* ``broadband_downwelling(...)`` -- delegated to M6.5's clear-sky emissivity relation (never the
                                   in-band T_sky, which is not a broadband quantity).

Cloud (MS.3, ADR 0070): with cloud fraction c from the shared WeatherSeries, the base at the
lifting condensation level of the same weather and T_base by the preset's lapse rate, the
blend is L_sky = (1 − c ε) L_clear + c ε L_B(T_base) (at c = 1 the sky is L_B(T_base) at every
angle and tilt; with RH = 1 the base is at the surface and the sky reads T_air).
``radiance_field`` adds the seeded 1/f^β structure per pixel.

Where ε comes from is the preset's choice (ADR 0126). ``clouds.tau`` gives one ε = 1 − τ for every
cloud at zero range — the original, kept bit-identical. ``clouds.optical_depth`` gives the cloud a
*visible optical depth* instead: the flux quantities (``radiance``, ``effective_radiance``) take
its diffusivity-weighted emissivity, and the **image** path (``radiance_field``) takes the ray's
own slant emissivity and puts the cloud at the LCL, so the air in front of it attenuates its
excess over the clear sky as 1/sin θ. That last part is what stops every opaque pixel in a frame
carrying one identical value.

The spec's T_sky = T_air − ΔT (1 − c) cos^q(θ_zen) (§5.3 a) is **not authored** here: ``fit_cos_q``
derives (ΔT, q) from the layered model and reports the fit error, and ADR 0044 records that the
form cannot follow the layered model within 0.5 K over 5°–90° (the sky warms toward the horizon
as a saturating column, not as a power of cos θ_zen), so the elevation LUT is the fast path.

Constructor takes the LayeredAtmosphere (which holds the one WeatherSeries) and exposes it as
``.weather`` for the Scene's identity check; a scalar air temperature is refused (CLAUDE.md #6).

docs/physics-model.md §5.3(a), §7.1; ADR 0126
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares

from irsim.atmosphere.cloud import (
    DIFFUSIVITY_FACTOR,
    CloudField,
    cloud_airmass,
    cloud_base_temperature_k,
    cloud_emissivity,
    cloud_radiance,
    cloud_radiance_at_range,
    generate_cloud_field,
    lifting_condensation_level_m,
)
from irsim.atmosphere.layered import LayeredAtmosphere
from irsim.atmosphere.skylight import DiffuseSkylight
from irsim.config.environment import EnvironmentSpec
from irsim.radiometry.lut import BandLUT, Quantity
from irsim.thermal.longwave import EmissivityFormula, longwave_down_from_sample
from irsim.thermal.weather import WeatherSeries

__all__ = ["SkyModel", "CosQFit", "azimuth_kernel", "ELEVATION_GRID_DEG", "TILT_GRID_DEG"]

ELEVATION_GRID_DEG = np.arange(0.0, 90.0 + 1e-9, 0.5)
TILT_GRID_DEG = np.arange(0.0, 180.0 + 1e-9, 1.0)
ELEVATION_FLOOR_RAD = math.radians(0.05)  # the flat-earth horizon integral needs θ > 0


def azimuth_kernel(a: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.float64]:
    """∫₀^{2π} max(0, a cos φ + b) dφ in closed form (a ≥ 0)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    out = np.where(b >= a, 2.0 * np.pi * b, 0.0)
    partial = (b > -a) & (b < a)
    with np.errstate(invalid="ignore", divide="ignore"):
        phi0 = np.arccos(np.clip(-b / np.where(a > 0, a, 1.0), -1.0, 1.0))
    out = np.where(partial, 2.0 * b * phi0 + 2.0 * a * np.sin(phi0), out)
    return np.asarray(out, dtype=np.float64)


@dataclass(frozen=True)
class CosQFit:
    """T_sky(θ_zen) = T_air − ΔT cos^q θ_zen fitted to the layered model over 5°–90°."""

    delta_t_k: float
    q: float
    max_error_k: float  # over the fit range, against the layered model
    rms_error_k: float


class SkyModel:
    def __init__(
        self,
        atmosphere: LayeredAtmosphere,
        environment: EnvironmentSpec,
        band: str,
        lut: BandLUT,
        quantity: Quantity = "lb",
        skylight: DiffuseSkylight | None = None,
    ) -> None:
        if not isinstance(atmosphere, LayeredAtmosphere):
            raise TypeError(
                "SkyModel takes the LayeredAtmosphere bound to the scene's WeatherSeries, "
                f"not {type(atmosphere).__name__} (CLAUDE.md #6: never a scalar T_air or a path)"
            )
        if not isinstance(environment, EnvironmentSpec):
            raise TypeError("environment must be an EnvironmentSpec (irsim.config.environment)")
        if band not in atmosphere.preset.bands:
            raise ValueError(f"band {band!r} not in the atmosphere preset")
        self._atm = atmosphere
        self._env = environment
        self._band = band
        self._lut = lut
        self._q: Quantity = quantity
        #: Scattered sunlight (M11.10, ADR 0086). ``None`` is a purely thermal sky, which is what
        #: this class was and is right for an emissive band -- in LWIR the scattered term is 1.6e-8
        #: of the column's own emission. In a reflective band leaving it out renders a **black
        #: sky**, which is backwards: there the daytime sky is the brightest thing in the frame.
        self._skylight = skylight
        if skylight is not None and skylight.quantity != quantity:
            raise ValueError(
                f"the skylight is in the {skylight.quantity!r} form and this sky model runs on "
                f"{quantity!r}; the two differ by ~1e19 and a mismatch renders a plausible sky "
                "with the wrong brightness"
            )
        self._cache: dict[float, tuple[NDArray[np.float64], NDArray[np.float64]]] = {}
        #: (tau to the cloud base, L beyond it) on the elevation grid; see `_cloud_path`.
        self._path_cache: dict[float, tuple[NDArray[np.float64], NDArray[np.float64]]] = {}

    # -- identity -------------------------------------------------------------------------
    @property
    def weather(self) -> WeatherSeries:
        return self._atm.weather

    @property
    def atmosphere(self) -> LayeredAtmosphere:
        return self._atm

    @property
    def band(self) -> str:
        return self._band

    @property
    def environment(self) -> EnvironmentSpec:
        return self._env

    @property
    def quantity(self) -> Quantity:
        return self._q

    @property
    def lut(self) -> BandLUT:
        return self._lut

    @property
    def skylight(self) -> DiffuseSkylight | None:
        return self._skylight

    # -- clear-sky elevation LUT ----------------------------------------------------------
    def _clear_lut(self, t_s: float) -> NDArray[np.float64]:
        """L_clear(θ) on the elevation grid (the layered column emission), cached per time."""
        key = float(t_s)
        if key not in self._cache:
            values = np.array(
                [
                    self._atm.sky_radiance(
                        self._band, key, max(math.radians(d), ELEVATION_FLOOR_RAD), self._q
                    )
                    for d in ELEVATION_GRID_DEG
                ]
            )
            if self._skylight is not None:
                # Isotropic, so it is a constant added at every elevation, and it flows into the
                # tilt LUT below and into every consumer of `clear_radiance` -- the sky background
                # *and* the reflected environment term -- from this one place.
                values = values + float(self._skylight.radiance(self.weather.at(key).dhi_w_m2)[()])
            tilt = self._tilt_lut(values)
            self._cache[key] = (values, tilt)
        return self._cache[key][0]

    # -- clouds (MS.3, ADR 0070) ------------------------------------------------------------
    def cloud_base_m(self, t_s: float) -> float:
        sample = self.weather.at(t_s)
        c = self._env.clouds
        return lifting_condensation_level_m(
            sample.t_air_k, sample.rh_fraction, c.min_base_m, c.max_base_m
        )

    def cloud_base_temperature_k(self, t_s: float) -> float:
        sample = self.weather.at(t_s)
        lapse = self._atm.preset.profile.lapse_rate_k_per_m
        return cloud_base_temperature_k(sample.t_air_k, self.cloud_base_m(t_s), lapse)

    def _flux_emissivity(self) -> float:
        """The cloud's emissivity to a **hemispherically integrated flux**, which is what the
        uniform blend in :meth:`radiance` and :meth:`effective_radiance` needs.

        For an ``optical_depth`` preset this is Shaw & Nugent's published ``1 − exp(−0.79 τ_vis)``
        exactly, because the diffusivity factor is the flux's own airmass. For a ``tau`` preset it
        is the authored ``1 − τ``, unchanged.
        """
        clouds = self._env.clouds
        if clouds.optical_depth is None:
            return float(clouds.emissivity)
        return float(cloud_emissivity(clouds.optical_depth, DIFFUSIVITY_FACTOR)[()])

    def _cloud(self, t_s: float) -> tuple[float, float]:
        """(effective covered fraction c ε, the cloud's own radiance L_B(T_base))."""
        sample = self.weather.at(t_s)
        l_base = float(
            self._lut.lookup(np.float64(self.cloud_base_temperature_k(t_s)), self._q)[()]
        )
        return sample.cloud_fraction * self._flux_emissivity(), l_base

    def _cloud_path(self, t_s: float) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """``(τ(0 → base, θ), L_beyond(base, θ))`` on the elevation grid, cached per time.

        The slant range to a plane-parallel deck at ``z_base`` is ``z_base / sin θ``, so both
        quantities are functions of elevation alone and go on the same 0.5° grid the clear-sky
        radiance already uses. 181 evaluations of the layered model per time step, once.

        ``L_beyond`` is the column emission from the base outwards *as seen from the base*, which
        is exactly what an opaque cloud occults; it is the layered model's own ``sky_beyond`` and
        not an approximation of it, so a cloud of emissivity 0 returns the clear sky identically.
        """
        key = float(t_s)
        if key not in self._path_cache:
            base_m = self.cloud_base_m(key)
            tau = np.empty(ELEVATION_GRID_DEG.size)
            beyond = np.empty(ELEVATION_GRID_DEG.size)
            for i, deg in enumerate(ELEVATION_GRID_DEG):
                el = max(math.radians(float(deg)), ELEVATION_FLOOR_RAD)
                # A base at the surface (saturated air) is the overcast limit: the cloud is at
                # zero range, nothing attenuates it, and this reduces to the old behaviour.
                slant = base_m / math.sin(el) if base_m > 0.0 else 0.0
                # The band transmittance Sum_k w_k tau_k, paired with `sky_beyond`'s tau_k-weighted
                # mean: the two are built so that L_path(R) + tau_band L_beyond = L_clear holds
                # **exactly** (both sides are Sum_k w_k L_sky,k), so eps = 0 and eps = 1 are exact
                # and the k-distribution approximation only enters in between.
                tau[i] = (
                    float(self._atm.transmittance(self._band, key, slant, el)[()])
                    if slant > 0.0
                    else 1.0
                )
                beyond[i] = (
                    self._atm.sky_beyond(self._band, key, slant, el, self._q)
                    if slant > 0.0
                    else self._atm.sky_radiance(self._band, key, el, self._q)
                )
            self._path_cache[key] = (tau, beyond)
        return self._path_cache[key]

    def cloud_field(self, t_s: float, shape: tuple[int, int], seed: int) -> CloudField:
        """The seeded 1/f^β structure covering the weather's cloud fraction of an image plane."""
        return generate_cloud_field(
            shape, self._env.clouds.beta, self.weather.at(t_s).cloud_fraction, seed
        )

    def cloud_ray_emissivity(self, elevation_rad: Any, density: Any) -> NDArray[np.float64]:
        """LWIR emissivity of the cloud along each ray, from its visible OD and the ray's airmass.

        Only defined for an ``optical_depth`` preset; a ``tau`` preset has one emissivity and no
        ray dependence, and :attr:`CloudSpec.emissivity` is the answer there.
        """
        optical_depth = self._env.clouds.optical_depth
        if optical_depth is None:
            raise ValueError(
                "this environment preset authors clouds.tau, which has no per-ray emissivity"
            )
        d = np.asarray(density, dtype=np.float64)
        return cloud_emissivity(optical_depth * d * cloud_airmass(elevation_rad), 1.0)

    def radiance_field(self, t_s: float, elevation_rad: Any, density: Any) -> NDArray[np.float64]:
        """Per-pixel sky radiance with structured cloud.

        ``density`` is `SkyFixedCloud.density`'s 0-to-1 depth along each ray, or a boolean
        coverage mask, which is that quantity at its two ends.

        Which model runs is the preset's choice (ADR 0126). Under ``clouds.tau`` a ray at full
        depth reads ``ε L_B(T_base) + τ L_clear`` exactly as it always did. Under
        ``clouds.optical_depth`` the emissivity comes from the ray's own slant optical depth and
        the cloud is placed at the LCL, so the air in front of it attenuates it.
        """
        clear = self.clear_radiance(t_s, elevation_rad)
        _, l_base = self._cloud(t_s)
        if self._env.clouds.optical_depth is None:
            return cloud_radiance(clear, l_base, self._env.clouds.tau, density)
        deg = np.degrees(np.asarray(elevation_rad, dtype=np.float64))
        tau_grid, beyond_grid = self._cloud_path(t_s)
        return cloud_radiance_at_range(
            clear,
            np.interp(deg, ELEVATION_GRID_DEG, beyond_grid),
            np.interp(deg, ELEVATION_GRID_DEG, tau_grid),
            l_base,
            self.cloud_ray_emissivity(elevation_rad, density),
        )

    def apparent_temperature_field(
        self, t_s: float, elevation_rad: Any, density: Any
    ) -> NDArray[np.float64]:
        return np.asarray(
            self._lut.apparent_temperature(
                self.radiance_field(t_s, elevation_rad, density), self._q
            ),
            dtype=np.float64,
        )

    def clear_radiance(self, t_s: float, elevation_rad: Any) -> NDArray[np.float64]:
        """Clear-sky column emission at elevation(s), linear on the 0.5° LUT (the fast path)."""
        deg = np.degrees(np.asarray(elevation_rad, dtype=np.float64))
        if np.any(deg < 0.0) or np.any(deg > 90.0 + 1e-9):
            raise ValueError("elevation must lie in [0, 90] degrees")
        return np.asarray(
            np.interp(deg, ELEVATION_GRID_DEG, self._clear_lut(t_s)), dtype=np.float64
        )

    def radiance(self, t_s: float, elevation_rad: Any) -> NDArray[np.float64]:
        """L_sky,B(θ) with the uniform cloud blend (1 − c ε) L_clear + c ε L_B(T_base): the
        expectation over the structured field, used for the LUTs and the tilt-integrated sky."""
        c, l_cloud = self._cloud(t_s)
        return np.asarray(
            (1.0 - c) * self.clear_radiance(t_s, elevation_rad) + c * l_cloud, dtype=np.float64
        )

    def apparent_temperature_k(self, t_s: float, elevation_rad: Any) -> NDArray[np.float64]:
        """T_sky(θ): what the adapter writes into temperature_k under the sky mask."""
        return np.asarray(
            self._lut.apparent_temperature(self.radiance(t_s, elevation_rad), self._q),
            dtype=np.float64,
        )

    # -- tilt LUT -------------------------------------------------------------------------
    @staticmethod
    def _tilt_lut(clear: NDArray[np.float64]) -> NDArray[np.float64]:
        """L_sky,eff(β) = ∫ L(θ) K(β, θ) cos θ dθ / ∫ K cos θ dθ with K the analytic azimuth
        kernel and L piecewise-linear on the elevation grid: composite Gauss–Legendre per grid
        interval, the interval holding the kernel's kink (θ = β or 180° − β) split there."""
        theta = np.radians(ELEVATION_GRID_DEG)
        out = np.empty(TILT_GRID_DEG.size)
        for i, beta_deg in enumerate(TILT_GRID_DEG):
            beta = math.radians(beta_deg)
            kinks = [k for k in (beta, math.pi - beta) if 0.0 < k < math.pi / 2]
            nodes, weights = _gauss_nodes(theta, kinks)
            kernel = azimuth_kernel(math.sin(beta) * np.cos(nodes), math.cos(beta) * np.sin(nodes))
            kernel = kernel * np.cos(nodes)
            norm = float(np.sum(weights * kernel))
            radiance = np.interp(nodes, theta, clear)
            out[i] = float(np.sum(weights * kernel * radiance)) / norm if norm > 0.0 else clear[-1]
        return out

    def effective_clear_radiance(self, t_s: float, tilt_rad: Any) -> NDArray[np.float64]:
        self._clear_lut(t_s)
        tilt = self._cache[float(t_s)][1]
        deg = np.degrees(np.asarray(tilt_rad, dtype=np.float64))
        if np.any(deg < 0.0) or np.any(deg > 180.0 + 1e-9):
            raise ValueError("tilt must lie in [0, 180] degrees (0 = facing up)")
        return np.asarray(np.interp(deg, TILT_GRID_DEG, tilt), dtype=np.float64)

    def effective_radiance(self, t_s: float, tilt_rad: Any) -> NDArray[np.float64]:
        """L_sky,eff(β) with the cloud blend; c = 1 gives L_B(T_air) at every tilt."""
        c, l_cloud = self._cloud(t_s)
        return np.asarray(
            (1.0 - c) * self.effective_clear_radiance(t_s, tilt_rad) + c * l_cloud, dtype=np.float64
        )

    def effective_radiance_from_sky_view(
        self, t_s: float, sky_view_factor: Any
    ) -> NDArray[np.float64]:
        """The same, indexed by an unoccluded sky-view factor V_s = (1 + cos β)/2 (M7.13)."""
        v = np.clip(np.asarray(sky_view_factor, dtype=np.float64), 0.0, 1.0)
        return self.effective_radiance(t_s, np.arccos(np.clip(2.0 * v - 1.0, -1.0, 1.0)))

    # -- broadband ------------------------------------------------------------------------
    def broadband_downwelling(
        self,
        t_s: float,
        sky_view_factor: Any,
        t_surround_k: Any,
        formula: EmissivityFormula = "brunt",
    ) -> NDArray[np.float64]:
        """Q_LW↓ for the energy balance -- M6.5's relation on the same weather sample (ADR 0035)."""
        return longwave_down_from_sample(
            self.weather.at(t_s), sky_view_factor, t_surround_k, formula
        )

    # -- the spec's form, derived --------------------------------------------------------
    def fit_cos_q(self, t_s: float, min_elevation_deg: float = 5.0) -> CosQFit:
        """Fit T_air − ΔT cos^q θ_zen to the clear-sky apparent temperature over
        [min_elevation, 90°]; the errors say how far the §5.3(a) form is from the column model."""
        t_air = self.weather.at(t_s).t_air_k
        sel = min_elevation_deg <= ELEVATION_GRID_DEG
        el = np.radians(ELEVATION_GRID_DEG[sel])
        t_sky = np.asarray(
            self._lut.apparent_temperature(self._clear_lut(t_s)[sel], self._q), dtype=np.float64
        )
        cos_zen = np.sin(el)  # cos θ_zen = sin θ_el

        def residual(x: NDArray[np.float64]) -> NDArray[np.float64]:
            dt, q = x
            return np.asarray(t_air - dt * cos_zen**q - t_sky, dtype=np.float64)

        res = least_squares(residual, x0=np.array([60.0, 0.5]), bounds=([0.0, 0.05], [200.0, 5.0]))
        err = residual(res.x)
        return CosQFit(
            float(res.x[0]),
            float(res.x[1]),
            float(np.max(np.abs(err))),
            float(np.sqrt(np.mean(err**2))),
        )


_GL_X, _GL_W = np.polynomial.legendre.leggauss(6)


def _gauss_nodes(
    knots: NDArray[np.float64], extra: list[float]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Composite 6-point Gauss–Legendre nodes/weights over consecutive knots (plus extra
    breakpoints): exact to ~1e-12 for a smooth kernel times a piecewise-linear factor."""
    pts = np.unique(np.concatenate([knots, np.asarray(extra, dtype=np.float64)]))
    a = pts[:-1]
    b = pts[1:]
    half = 0.5 * (b - a)
    mid = 0.5 * (a + b)
    nodes = mid[:, None] + half[:, None] * _GL_X[None, :]
    weights = half[:, None] * _GL_W[None, :]
    return nodes.ravel(), weights.ravel()
