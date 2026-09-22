"""The sea surface as a background: Cox-Munk slope statistics and apparent temperature vs angle.

The maritime analogue of :mod:`irsim.atmosphere.sky` (MS.2). Above the horizon a background pixel
takes ``T_sky(theta)``; below it, this module says what the sea reads, and the answer is not one
number. Water is a Fresnel reflector and the viewing geometry is brutal: a camera 20 m above the
surface sees the sea at 5 km at 0.23 degrees of depression, which is **89.8 degrees of incidence**,
where the band-effective emissivity has fallen from 0.988 to under 0.15. Nearly all visible sea is
a mirror of the sky, so its apparent temperature runs from about SST looking straight down to
within a couple of kelvin of the sky at the horizon. A vessel warmer than the sky and cooler than
the water therefore reads **dark against near water and bright against far water**, and where that
contrast null sits governs detection across range (ADR 0078).

**Why this is analytic and not geometry.** At 3 km a Boson pixel spans 2.6 m and contains thousands
of independent wave facets. One normal per pixel is not a coarse version of the right answer, it is
a different quantity. What the detector integrates is the *distribution* of facet orientations
inside the pixel, which Cox & Munk give in closed form from the wind alone.

**The quadrature.** A facet tilted by ``alpha`` in the vertical plane does two things at once: it
changes the incidence angle to ``(90 deg - depression) - alpha``, and it swings the reflected ray to
elevation ``depression + 2 alpha``. Both are integrated over the tilt distribution together, so the
effective emissivity near the horizon rises above the flat-surface value (a tilted facet presents
less grazing incidence) at the same time as the reflected sky broadens. The weights sum to one by
construction, which is what makes the isothermal identity hold for *any* emissivity -- the test
that catches a mis-weighted reflection when a plausible-looking gradient would not.

**What is approximated, in order of how much it matters.** (1) No wave shadowing or inter-facet
reflection: at extreme grazing a Gaussian slope model over-counts facets that would in reality be
hidden behind the wave in front, and this is worst in exactly the near-horizon band a low camera
cares about most. ADR 0078 records that this error is not currently bounded. (2) A reflected ray
aimed below the horizon hits the sea again; it is charged ``L_sky(0)`` instead, which is defensible
only because the sea near the horizon is itself nearly a sky mirror. (3) The slope distribution is
taken isotropic, the mean of Cox-Munk's upwind and crosswind variances; the anisotropy needs the
view-to-wind azimuth and is deferred. (4) The path from camera to surface is treated as horizontal,
which for any camera under 100 m means an elevation under 3 degrees over a homogeneous column.

docs/physics-model.md §5.3, §4.2, §7.4; roadmap MM.2, MM.3; ADR 0078, ADR 0079
"""

from __future__ import annotations

import math
from datetime import timedelta
from typing import Any

import numpy as np
from numpy.polynomial.hermite_e import hermegauss
from numpy.typing import NDArray

from irsim.atmosphere.sky import SkyModel
from irsim.materials.nk import NKTable, band_directional_emissivity
from irsim.radiometry.lut import Quantity
from irsim.radiometry.spectral_response import SpectralResponse
from irsim.thermal.longwave import longwave_down_from_sample
from irsim.thermal.sea_skin import (
    DEFAULT_SEA_SKIN,
    SeaSkinParams,
    cool_skin_deficit_k,
    net_longwave_up_w_m2,
    warm_layer_k,
)
from irsim.thermal.solar import solar_loading, sun_direction, sun_position_utc

__all__ = [
    "EARTH_RADIUS_M",
    "COX_MUNK",
    "INCIDENCE_GRID_DEG",
    "slope_variance",
    "horizon_depression_rad",
    "slant_range_m",
    "SeaModel",
]

#: Mean Earth radius, for the horizon geometry only (not a radiometric constant).
EARTH_RADIUS_M = 6_371_000.0

#: Cox & Munk (1954) clean-sea slope variances against wind speed at 12.5 m, dimensionless
#: (variance of the slope tan(alpha), not of the angle). Keyed by component.
COX_MUNK: dict[str, tuple[float, float]] = {
    # component: (intercept, coefficient on U in m/s)
    "upwind": (0.0, 3.16e-3),
    "crosswind": (3.0e-3, 1.92e-3),
}

#: Incidence-angle grid for the band-effective emissivity LUT. 0.25° is fine enough that
#: linear interpolation stays under 1e-3 even through the last degree before grazing, where
#: ε falls by roughly 0.08 per degree.
INCIDENCE_GRID_DEG = np.arange(0.0, 90.0 + 1e-9, 0.25)

#: Points in the profile LUT, which is tabulated against **slant range, not depression angle**.
#: Angle is the wrong coordinate: d(delta) has a square-root singularity at the horizon, so a
#: grid in angle (however fine, and however geometric) leaves a cusp that refinement barely
#: touches -- measured 52 mK of interpolation error at 192 points and still 41 mK at 768, both
#: of them a visible fraction of a 50 mK NETD. In range the same profile is smooth, because
#: what varies out there is tau(d) and L_path(d). 192 points then cost 0.26 s to build, land under
#: 13 mK, and answer two million pixels in 34 ms.
_PROFILE_POINTS = 192

#: Nodes for the facet-tilt quadrature. 15 Gauss-Hermite nodes reach past 4 sigma, and the
#: integrand is smooth in the tilt, so this is convergence rather than a compromise.
_TILT_NODES = 15


def slope_variance(wind_speed_m_s: Any) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Cox-Munk (upwind, crosswind) slope variances for a clean sea at ``wind_speed_m_s``.

    The two components are returned rather than one isotropic number because they are what a
    direction-aware model will need, and because their ratio is a property of the sea rather than
    of this code.

    **They do not add up to the familiar total, and that is not a bug.** Cox & Munk fitted the
    isotropic total independently as ``3.0e-3 + 5.12e-3 U``, while the component fits sum to
    ``3.0e-3 + 5.08e-3 U`` -- the two regressions were run separately on the same slicks data and
    disagree by 0.8 % in variance, which is 0.4 % in RMS slope and far inside their quoted scatter.
    This function returns the components; anything wanting the published isotropic total should say
    so explicitly rather than summing these.
    """
    u = np.asarray(wind_speed_m_s, dtype=np.float64)
    if np.any(u < 0.0):
        raise ValueError("wind speed must be non-negative")
    up = COX_MUNK["upwind"][0] + COX_MUNK["upwind"][1] * u
    cross = COX_MUNK["crosswind"][0] + COX_MUNK["crosswind"][1] * u
    return np.asarray(up, dtype=np.float64), np.asarray(cross, dtype=np.float64)


def horizon_depression_rad(camera_height_m: float) -> float:
    """Depression of the geometric horizon below the local horizontal, for a spherical Earth."""
    if camera_height_m <= 0.0:
        raise ValueError("camera height above the sea must be positive")
    h = float(camera_height_m)
    return math.acos(EARTH_RADIUS_M / (EARTH_RADIUS_M + h))


def slant_range_m(camera_height_m: float, depression_rad: Any) -> NDArray[np.float64]:
    """Distance from a camera at ``camera_height_m`` to the sea, along a ray at ``depression_rad``.

    The exact spherical solution, not ``h / sin(delta)``: the flat-earth form underestimates the
    range at the horizon by a factor of two, and the horizon is where every long-range maritime
    target sits. Rays shallower than the horizon raise -- they never meet the surface.
    """
    delta = np.asarray(depression_rad, dtype=np.float64)
    horizon = horizon_depression_rad(camera_height_m)
    if np.any(delta < horizon - 1e-12):
        raise ValueError(
            f"depression {np.min(delta):.6g} rad is above the horizon at "
            f"{horizon:.6g} rad for a camera {camera_height_m} m up: the ray never hits the sea"
        )
    if np.any(delta > 0.5 * np.pi + 1e-12):
        raise ValueError("depression must not exceed 90 degrees (straight down)")
    r_c = EARTH_RADIUS_M + float(camera_height_m)
    disc = np.maximum((r_c * np.sin(delta)) ** 2 - (r_c**2 - EARTH_RADIUS_M**2), 0.0)
    return np.asarray(r_c * np.sin(delta) - np.sqrt(disc), dtype=np.float64)


class SeaModel:
    """Apparent temperature of the sea against depression angle, on one shared WeatherSeries.

    Takes the :class:`~irsim.atmosphere.sky.SkyModel` rather than a sky radiance, because the sea
    *is* mostly reflected sky and the two must never disagree about the weather (CLAUDE.md #6). The
    wind that roughens the surface comes from that same series.
    """

    def __init__(
        self,
        sky: SkyModel,
        nk_table: NKTable,
        response: SpectralResponse,
        bulk_sst_k: float,
        camera_height_m: float = 20.0,
        quantity: Quantity = "lb",
        skin: SeaSkinParams = DEFAULT_SEA_SKIN,
        latitude_deg: float | None = None,
        longitude_deg: float | None = None,
        solar_absorptivity: float = 0.94,
    ) -> None:
        if not isinstance(sky, SkyModel):
            raise TypeError(
                "SeaModel takes the scene's SkyModel, so the sea and the sky it reflects share "
                f"one WeatherSeries (CLAUDE.md #6), not {type(sky).__name__}"
            )
        if sky.quantity != quantity:
            raise ValueError(
                f"the sky model is in the {sky.quantity!r} form and this sea model runs on "
                f"{quantity!r}. The sea is mostly *reflected sky*, so the two radiances are added "
                "together -- and `lb` and `lb_q` differ by about 1e19, so a mismatch does not "
                "produce a slightly wrong sea, it produces one whose apparent temperature pins at "
                "the LUT ceiling. Pass the camera's own quantity (SensorSpec.quantity, ADR 0021)."
            )
        if bulk_sst_k <= 0.0:
            raise ValueError("bulk_sst_k must be a positive absolute temperature")
        if (latitude_deg is None) != (longitude_deg is None):
            raise ValueError(
                "latitude_deg and longitude_deg come as a pair: the diurnal warm layer needs a "
                "sun elevation, and half a site does not give one"
            )
        if not 0.0 <= solar_absorptivity <= 1.0:
            raise ValueError("solar_absorptivity must lie in [0, 1]")
        self._sky = sky
        self._table = nk_table
        self._response = response
        self._bulk_sst_k = float(bulk_sst_k)
        self._camera_height_m = float(camera_height_m)
        self._skin = skin
        self._site = (
            None if latitude_deg is None else (float(latitude_deg), float(longitude_deg or 0.0))
        )
        self._alpha_sol = float(solar_absorptivity)
        self._q: Quantity = quantity
        self._eps_lut: NDArray[np.float64] | None = None
        self._profile: dict[float, tuple[NDArray[np.float64], NDArray[np.float64]]] = {}

    # -- identity ---------------------------------------------------------------------------
    @property
    def sky(self) -> SkyModel:
        return self._sky

    @property
    def weather(self) -> Any:
        return self._sky.weather

    @property
    def camera_height_m(self) -> float:
        return self._camera_height_m

    @property
    def bulk_sst_k(self) -> float:
        return self._bulk_sst_k

    @property
    def horizon_rad(self) -> float:
        return horizon_depression_rad(self._camera_height_m)

    # -- surface state ----------------------------------------------------------------------
    def absorbed_solar_w_m2(self, t_s: float) -> float:
        """Shortwave the sea absorbs, from the shared weather's DNI/DHI and the sun's elevation.

        Zero without a site, and zero at night. The surface is horizontal, so the direct beam is
        weighted by ``sin(elevation)`` and the whole diffuse component arrives.
        """
        if self._site is None:
            return 0.0
        sample = self._sky.weather.at(t_s)
        when = self._sky.weather.epoch_utc + timedelta(seconds=float(t_s))
        sun = sun_position_utc(self._site[0], self._site[1], when)
        if sun.elevation_deg <= 0.0:
            return 0.0
        q = solar_loading(
            np.array([0.0, 0.0, 1.0]),
            sun_direction(sun.elevation_deg, sun.azimuth_deg),
            sample.dni_w_m2,
            sample.dhi_w_m2,
            1.0,
        )
        return float(self._alpha_sol * float(q))

    def net_longwave_up_w_m2(self, t_s: float) -> float:
        """Net longwave leaving the surface, from the scene's own sky (M6.5).

        Evaluated at the **bulk** temperature rather than at the skin, which is the one place this
        model is knowingly not self-consistent: the skin is what radiates, and it depends on this
        flux. The circularity is worth naming and not worth iterating -- a 0.2 K difference in the
        radiating temperature moves sigma T^4 by 0.3 %, which moves the deficit by well under a
        millikelvin, three orders below the 50 mK the sensor can see.

        The sea sees the whole sky (V_s = 1) and has nothing else around it, so the surroundings
        term is the sky term.
        """
        sample = self._sky.weather.at(t_s)
        down = float(longwave_down_from_sample(sample, 1.0, sample.t_air_k))
        return float(net_longwave_up_w_m2(self._bulk_sst_k, down))

    def latent_up_w_m2(self, t_s: float) -> float:
        """Evaporative heat loss from the bulk sea to the scene's air, W m⁻² (PH.1).

        At sea the latent flux is usually the largest term in the net heat loss (the sea-skin
        module's own docstring says so, and omitted it). ``ρ_a C_E U (q_sat(SST) − q_a)`` with
        the bulk SST as the evaporating temperature -- the same knowingly-uniterated choice the
        longwave term makes -- and no salinity reduction of ``q_sat`` (2 %, below the bulk
        coefficient's own spread).
        """
        from irsim.thermal.latent import (
            bulk_conductance_kg_m2_s,
            latent_heat_flux_w_m2,
            specific_humidity_kg_kg,
        )

        sample = self._sky.weather.at(t_s)
        q_air = float(specific_humidity_kg_kg(sample.t_air_k, sample.rh_fraction))
        g_e = float(bulk_conductance_kg_m2_s(sample.wind_speed_m_s))
        # Clamped at zero: a sea colder than the air's dew point collects condensation, which
        # deposits its latent heat *at* the skin -- a warm-layer-like mechanism the cool-skin
        # model does not carry -- and letting it cancel the longwave loss here would report a
        # skin that has stopped cooling for the wrong reason.
        return max(0.0, float(latent_heat_flux_w_m2(self._bulk_sst_k, q_air, g_e)))

    def net_heat_up_w_m2(self, t_s: float) -> float:
        """The net heat leaving the sea that the cool skin conducts: longwave plus latent."""
        return self.net_longwave_up_w_m2(t_s) + self.latent_up_w_m2(t_s)

    def cool_skin_deficit_k(self, t_s: float) -> float:
        """How far the skin sits below the bulk, from the shared weather's wind (MM.4, PH.1)."""
        wind = float(self._sky.weather.at(t_s).wind_speed_m_s)
        return float(cool_skin_deficit_k(self.net_heat_up_w_m2(t_s), wind, self._skin))

    def warm_layer_k(self, t_s: float) -> float:
        """How far a calm, sunlit afternoon lifts the skin above the bulk (MM.4)."""
        wind = float(self._sky.weather.at(t_s).wind_speed_m_s)
        return float(warm_layer_k(self.absorbed_solar_w_m2(t_s), wind, self._skin))

    def skin_temperature_k(self, t_s: float) -> float:
        """T_skin = T_bulk - dT_cool(U, Q_net) + dT_warm(Q_sw, U) -- what the camera sees (MM.4).

        The bulk SST is the authored scenario input because that is the number a maritime scenario
        actually has; the skin is derived from it here, on the scene's own weather, so a calm-sea
        skin cannot end up under a 15 m/s wind.
        """
        return self._bulk_sst_k - self.cool_skin_deficit_k(t_s) + self.warm_layer_k(t_s)

    def tilt_sigma(self, t_s: float) -> float:
        """RMS facet tilt (radians) from the shared weather's wind, isotropic approximation."""
        wind = float(self._sky.weather.at(t_s).wind_speed_m_s)
        up, cross = slope_variance(wind)
        return float(math.sqrt(0.5 * (float(up) + float(cross))))

    def _emissivity_lut(self) -> NDArray[np.float64]:
        """ε_B on :data:`INCIDENCE_GRID_DEG`, built once.

        A band average is a Simpson quadrature over the spectral response, and the facet integral
        wants one per tilt node per pixel -- millions of them. The same trick the SkyModel uses for
        elevation applies here: tabulate against incidence angle once and interpolate. A test
        checks the interpolation against direct evaluation, because the grid has to be fine enough
        where ε is steepest, which is the last degree before grazing.
        """
        if self._eps_lut is None:
            self._eps_lut = band_directional_emissivity(
                self._table, self._response, np.cos(np.radians(INCIDENCE_GRID_DEG))
            )
        return self._eps_lut

    def emissivity(self, cos_incidence: Any) -> NDArray[np.float64]:
        """Band-effective Fresnel emissivity of water at the given incidence cosine (MM.1)."""
        theta_deg = np.degrees(np.arccos(np.clip(np.abs(np.asarray(cos_incidence)), 0.0, 1.0)))
        return np.asarray(
            np.interp(theta_deg, INCIDENCE_GRID_DEG, self._emissivity_lut()), dtype=np.float64
        )

    # -- the profile ------------------------------------------------------------------------
    def facet_quadrature(self, t_s: float) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Facet tilts and their weights. The weights sum to 1 exactly -- that is the invariant
        the isothermal identity rests on, so it is exposed rather than hidden in the integral."""
        sigma = self.tilt_sigma(t_s)
        x, w = hermegauss(_TILT_NODES)
        return np.asarray(sigma * x, dtype=np.float64), np.asarray(
            w / math.sqrt(2.0 * math.pi), dtype=np.float64
        )

    def surface_radiance(self, t_s: float, depression_rad: Any) -> NDArray[np.float64]:
        """Radiance leaving the sea toward the camera, before the atmospheric path.

        ``<eps Lb(T_skin) + (1 - eps) L_sky>`` over the facet tilt distribution, with the emissivity
        and the reflected sky elevation both moving with the tilt.
        """
        delta = np.atleast_1d(np.asarray(depression_rad, dtype=np.float64))
        alpha, weight = self.facet_quadrature(t_s)
        lb_skin = float(self._sky.lut.lookup(self.skin_temperature_k(t_s), self._q))

        # incidence measured from the surface normal; |cos| handles a facet tilted past the ray
        cos_i = np.abs(np.cos(0.5 * np.pi - delta[:, None] - alpha[None, :]))
        eps = self.emissivity(cos_i)
        # the reflected ray leaves at (depression + 2 alpha) above the horizontal; a ray aimed
        # below the horizon hits the sea again and is charged the horizon sky (module docstring)
        reflected_el = np.clip(delta[:, None] + 2.0 * alpha[None, :], 0.0, 0.5 * np.pi)
        l_sky = self._sky.radiance(t_s, reflected_el)

        integrand = eps * lb_skin + (1.0 - eps) * l_sky
        out = np.asarray(np.sum(integrand * weight[None, :], axis=1), dtype=np.float64)
        return out.reshape(np.shape(depression_rad)) if np.shape(depression_rad) else out

    def radiance(self, t_s: float, depression_rad: Any) -> NDArray[np.float64]:
        """Sea radiance as the camera sees it: the surface through the intervening air.

        The path is taken horizontal (elevation 0) -- for any camera under 100 m every visible sea
        patch is within 3 degrees of the horizontal, over a column that is homogeneous across that
        height difference.
        """
        surface = np.atleast_1d(self.surface_radiance(t_s, depression_rad))
        delta = np.atleast_1d(np.asarray(depression_rad, dtype=np.float64))
        distance = slant_range_m(self._camera_height_m, delta)
        atm = self._sky.atmosphere
        band = self._sky.band

        out = np.empty(delta.shape, dtype=np.float64)
        for i, d in enumerate(distance):
            tau = float(atm.transmittance(band, t_s, float(d), 0.0))
            path = atm.path_radiance(band, t_s, float(d), 0.0, self._q)
            out[i] = tau * surface[i] + path
        return out.reshape(np.shape(depression_rad)) if np.shape(depression_rad) else out

    def profile(self, t_s: float) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """(depression angles, apparent temperatures) on the geometric grid, cached per time.

        The reason this exists: :meth:`radiance` runs a path-radiance quadrature per angle, and a
        supersampled frame asks for millions of angles, most of them within a few thousandths of a
        degree of each other. Tabulating once per render time and interpolating is the same move
        :class:`~irsim.atmosphere.sky.SkyModel` makes for elevation, and without it a single frame
        takes longer than the whole rest of the pipeline put together.
        """
        key = float(t_s)
        if key not in self._profile:
            horizon = self.horizon_rad
            near = float(slant_range_m(self._camera_height_m, 0.5 * np.pi))
            far = float(slant_range_m(self._camera_height_m, horizon))
            ranges = np.geomspace(near, far, _PROFILE_POINTS)
            # Back out the depression each range corresponds to, so the profile is still evaluated
            # by the exact path and only its *sampling* is in range.
            r_c = EARTH_RADIUS_M + self._camera_height_m
            sin_delta = np.clip(
                (ranges**2 + r_c**2 - EARTH_RADIUS_M**2) / (2.0 * r_c * ranges), -1.0, 1.0
            )
            angles = np.arcsin(sin_delta)
            angles[0] = 0.5 * np.pi
            angles[-1] = horizon
            self._profile[key] = (
                ranges,
                np.asarray(
                    self._sky.lut.apparent_temperature(self.radiance(key, angles), self._q),
                    dtype=np.float64,
                ),
            )
        return self._profile[key]

    def apparent_temperature_k(self, t_s: float, depression_rad: Any) -> NDArray[np.float64]:
        """T_sea(delta): what the bridge writes into ``temperature_k`` below the horizon.

        Interpolated on the cached profile in **log slant range**, not in angle: d(delta) has a
        square-root singularity at the horizon, so an angle grid keeps a cusp that refining does
        not remove (52 mK at 192 points, still 41 mK at 768). In range it is smooth and the same
        192 points land under 13 mK.
        """
        ranges, values = self.profile(t_s)
        delta = np.clip(np.asarray(depression_rad, dtype=np.float64), self.horizon_rad, 0.5 * np.pi)
        d = np.clip(slant_range_m(self._camera_height_m, delta), ranges[0], ranges[-1])
        return np.asarray(np.interp(np.log(d), np.log(ranges), values), dtype=np.float64)

    # -- the validity envelope (SE.1) -------------------------------------------------------
    def view_zenith_rad(self, depression_rad: Any) -> NDArray[np.float64]:
        """Zenith angle at the sea for a ray at ``depression_rad`` -- the axis the published
        in-situ validation is indexed by (:mod:`irsim.atmosphere.sea_envelope`)."""
        from irsim.atmosphere.sea_envelope import view_zenith_rad

        return view_zenith_rad(self._camera_height_m, depression_rad)

    def beyond_envelope(self, depression_rad: Any) -> NDArray[np.bool_]:
        """True where this ray views the sea past the angle published radiometry reaches (SE.1).

        The model still answers there -- it has to, because a shore camera has almost no pixels
        anywhere else -- but the answer is not backed by a measurement. ADR 0118.
        """
        from irsim.atmosphere.sea_envelope import beyond_envelope

        return beyond_envelope(self._camera_height_m, depression_rad)

    def envelope_report(self, depression_rad: Any, mask: Any | None = None) -> Any:
        """How much of a frame's sea lies outside the validated envelope (SE.1)."""
        from irsim.atmosphere.sea_envelope import envelope_report

        return envelope_report(self._camera_height_m, depression_rad, mask)

    def apparent_temperature_exact_k(self, t_s: float, depression_rad: Any) -> NDArray[np.float64]:
        """The un-tabulated profile: the oracle :meth:`apparent_temperature_k` is tested against."""
        return np.asarray(
            self._sky.lut.apparent_temperature(self.radiance(t_s, depression_rad), self._q),
            dtype=np.float64,
        )

    def __repr__(self) -> str:  # pragma: no cover - diagnostics
        return (
            f"SeaModel(band={self._sky.band!r}, sst={self._bulk_sst_k:.2f} K, "
            f"h={self._camera_height_m:.1f} m, horizon={math.degrees(self.horizon_rad):.3f} deg)"
        )
