"""The sea as a background: Cox-Munk slopes and apparent temperature vs angle (MM.2, MM.3).

The load-bearing test here is :func:`test_isothermal_identity_holds_at_every_angle_and_wind`. It
holds for *any* emissivity, so it catches a mis-weighted reflection — the failure a plausible
looking gradient would hide, and the one no amount of looking at a rendered frame would reveal.

The rest of the file documents a profile shape that is **not** the simple monotone ramp from SST at
nadir to sky at the horizon that this milestone was planned around. Two effects fight:

* looking further down raises the incidence emissivity (0.11 at 89°, 0.99 at nadir), which pulls
  the sea toward its own temperature;
* looking further down also reflects sky from *higher* elevations, which is colder (290 K at the
  horizon, 226 K at zenith).

The second wins between roughly 2° and 15° of depression and the first wins outside it, so the
profile has a **minimum a few degrees below the horizon**, not a monotone slope. Adding the
atmospheric path then pulls the far field back toward air temperature, because at 4 km there is
more air than sea in the ray. Tests pin the shape rather than the wrong prediction.

docs/physics-model.md §5.3, §4.2, §7.4; ADR 0078
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from irsim.atmosphere import LayeredAtmosphere, SkyModel, load_atmosphere_preset
from irsim.atmosphere.sea import (
    SeaModel,
    horizon_depression_rad,
    slant_range_m,
    slope_variance,
)
from irsim.config.environment import load_environment_preset
from irsim.materials.nk import band_directional_emissivity, load_nk_table
from irsim.radiometry.lut import BandLUT
from irsim.radiometry.spectral_response import load_spectral_response
from irsim.thermal.weather import WeatherSample, WeatherSeries

# GT.1: this whole module is the slow tier -- a validation bench or an end-to-end frame rather
# than a unit test. `make test` skips it; `make test-slow` and `make check` run it.
pytestmark = pytest.mark.slow

BOSON_RESPONSE = "data/spectra/responses/boson_vox.csv"
SST_K = 290.0
T_AIR_K = 291.0
CAMERA_HEIGHT_M = 20.0


@pytest.fixture(scope="module")
def rig():
    response = load_spectral_response(BOSON_RESPONSE)
    return response, BandLUT.build(response), load_nk_table("water")


def _sea(rig, wind=5.0, sst=SST_K, cloud=0.0, height=CAMERA_HEIGHT_M, **kwargs):
    response, lut, table = rig
    weather = WeatherSeries.constant(
        WeatherSample(T_AIR_K, 0.5, wind, cloud, 0.0, 0.0, 23000.0, 0.0), 3600.0
    )
    atm = LayeredAtmosphere(load_atmosphere_preset("us_standard_clear"), weather, {"lwir": lut})
    sky = SkyModel(atm, load_environment_preset("clear_dry"), "lwir", lut)
    return sky, SeaModel(sky, table, response, bulk_sst_k=sst, camera_height_m=height, **kwargs)


# -- Cox-Munk (MM.2) ----------------------------------------------------------------------


def test_cox_munk_slope_variance_reproduces_the_published_coefficients() -> None:
    """sigma_u^2 = 3.16e-3 U and sigma_c^2 = 3.0e-3 + 1.92e-3 U, the component fits.

    The components sum to 3.0e-3 + 5.08e-3 U, which is **not** Cox & Munk's separately fitted
    isotropic total 3.0e-3 + 5.12e-3 U. The two regressions were run independently on the same
    data and disagree by 0.8 % in variance. This test pins that gap rather than papering over it,
    because a future contributor who "fixes" the components to add up will have silently replaced
    the measured anisotropy with an isotropic assumption.
    """
    for u in (0.0, 3.0, 10.0, 20.0):
        up, cross = slope_variance(u)
        assert float(up) == pytest.approx(3.16e-3 * u, abs=1e-15)
        assert float(cross) == pytest.approx(3.0e-3 + 1.92e-3 * u, abs=1e-15)
        assert float(up + cross) == pytest.approx(3.0e-3 + 5.08e-3 * u, abs=1e-15)
        published_isotropic = 3.0e-3 + 5.12e-3 * u
        assert float(up + cross) == pytest.approx(published_isotropic, rel=0.01)

    up0, cross0 = slope_variance(0.0)
    assert float(up0 + cross0) == pytest.approx(3.0e-3, abs=1e-15)  # a glassy sea is not a mirror
    up10, cross10 = slope_variance(10.0)
    assert float(up10) > float(cross10)  # waves are steeper along the wind than across it
    assert math.degrees(math.atan(math.sqrt(float(up10 + cross10)))) == pytest.approx(13.1, abs=0.1)


def test_negative_wind_is_refused() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        slope_variance(-1.0)


# -- geometry (MM.3) ----------------------------------------------------------------------


def test_horizon_depression_and_the_range_to_it() -> None:
    """0.045° at 2 m, 0.144° at 20 m — the whole sea lives in the first degree below horizontal.

    And the range at the horizon is sqrt(2 h R_e), which is **twice** what the flat-earth h/sin(δ)
    gives at the same angle. That factor of two lands directly on the atmospheric path length to
    every long-range maritime target, which is why the spherical solution is not a nicety.
    """
    assert math.degrees(horizon_depression_rad(2.0)) == pytest.approx(0.0454, abs=1e-4)
    assert math.degrees(horizon_depression_rad(20.0)) == pytest.approx(0.1436, abs=1e-4)

    horizon = horizon_depression_rad(20.0)
    exact = float(slant_range_m(20.0, horizon))
    assert exact == pytest.approx(math.sqrt(2.0 * 20.0 * 6_371_000.0), rel=1e-4)
    assert exact == pytest.approx(2.0 * 20.0 / math.sin(horizon), rel=2e-3)  # flat-earth halves it

    assert float(slant_range_m(20.0, math.radians(90.0))) == pytest.approx(20.0, abs=1e-6)
    assert float(slant_range_m(20.0, math.radians(1.0))) == pytest.approx(1152.0, rel=0.01)


def test_a_ray_above_the_horizon_never_hits_the_sea() -> None:
    with pytest.raises(ValueError, match="never hits the sea"):
        slant_range_m(20.0, math.radians(0.10))


# -- the identity that makes the reflection trustworthy ------------------------------------


@pytest.mark.parametrize("sst", [275.0, 290.0, 305.0])
@pytest.mark.parametrize("wind", [0.0, 10.0, 20.0])
def test_isothermal_identity_holds_at_every_angle_and_wind(rig, sst, wind) -> None:
    """Sky at the sea's own temperature ⇒ the sea reads its own temperature, to 1 mK.

    This must hold for *any* emissivity, because ε L + (1−ε) L = L. So it tests the one thing the
    emissivity cannot mask: whether the facet weights sum to one and the reflection is mixed in the
    right proportion. A model that double-counted a tilt node, or normalised the Gaussian wrongly,
    or clipped weight away at the horizon, fails here and nowhere else.
    """
    _, lut, _ = rig
    sky, sea = _sea(rig, wind=wind, sst=sst)
    # The enclosure is isothermal with the surface that radiates, which is the **skin** -- since
    # MM.4 that is no longer the bulk SST. Holding the sky at the bulk instead leaves exactly the
    # cool-skin deficit as a residual (0.09 K at 275 K), which would look like a broken quadrature
    # and is not one.
    skin = sea.skin_temperature_k(0.0)
    assert skin < sst
    constant = float(lut.lookup(skin, "lb"))
    sky.radiance = lambda t_s, elevation: np.full(np.shape(elevation), constant)  # type: ignore[method-assign]

    depression = np.radians(np.array([0.15, 0.5, 1.0, 5.0, 30.0, 60.0, 90.0]))
    reading = lut.apparent_temperature(sea.surface_radiance(0.0, depression), "lb")
    assert np.max(np.abs(np.asarray(reading, dtype=np.float64) - skin)) < 1e-3


def test_facet_weights_sum_to_one(rig) -> None:
    """The invariant the identity rests on, asserted directly so a failure is legible."""
    for wind in (0.0, 5.0, 20.0):
        _, sea = _sea(rig, wind=wind)
        alpha, weight = sea.facet_quadrature(0.0)
        assert float(np.sum(weight)) == pytest.approx(1.0, abs=1e-12)
        assert float(np.sum(weight * alpha)) == pytest.approx(0.0, abs=1e-12)
        sigma = sea.tilt_sigma(0.0)
        assert float(np.sum(weight * alpha**2)) == pytest.approx(sigma**2, rel=1e-9)


# -- the profile --------------------------------------------------------------------------


def test_nadir_reads_just_below_the_sea_surface_temperature(rig) -> None:
    """Straight down, ε = 0.988, so the sea reads ~0.5 K under SST — the reflected cold sky.

    The offset is checked against (1−ε)(Lb(T_sea) − L_sky(90°))/(dLb/dT), not against a
    remembered number: it is the reflection term, and it must scale with the sky it reflects.
    """
    _, lut, _ = rig
    sky, sea = _sea(rig)
    nadir = math.radians(90.0)

    reading = float(np.atleast_1d(sea.apparent_temperature_k(0.0, nadir))[0])
    assert reading < SST_K
    # Two effects now separate the reading from the authored bulk SST, and they are checked apart
    # because they have different sizes and different causes: the reflected cold sky (~0.53 K,
    # below) and MM.4's cool skin, which moves the radiating temperature itself -- ~0.11 K at
    # 5 m/s on the longwave loss alone, ~0.31 K once PH.1 added the latent flux the sea-skin
    # module had always said was the larger term.
    skin = sea.skin_temperature_k(0.0)
    deficit = SST_K - skin
    assert 0.05 < deficit < 0.50, deficit
    assert skin - reading == pytest.approx(0.53, abs=0.15)

    eps = float(np.atleast_1d(sea.emissivity(1.0))[0])
    l_sea = float(lut.lookup(skin, "lb"))
    l_sky = float(np.atleast_1d(sky.radiance(0.0, nadir))[0])
    dl_dt = float(lut.lookup(skin, "dlb_dt"))
    predicted = (1.0 - eps) * (l_sea - l_sky) / dl_dt
    assert skin - reading == pytest.approx(predicted, rel=0.25)


def test_the_profile_has_a_minimum_rather_than_a_monotone_ramp(rig) -> None:
    """Warmest at nadir, coldest a few degrees down, warming again toward the horizon.

    The planned prediction for this milestone was a monotone ramp from SST at nadir to sky at the
    horizon. It is wrong, and the measured shape is the interesting one: the emissivity rising with
    depression and the reflected sky cooling with elevation pull in opposite directions, so the
    surface radiance turns over. This test pins the turn-over so a refactor cannot quietly restore
    the monotone version.
    """
    _, lut, _ = rig
    _, sea = _sea(rig)
    depression = np.radians(np.array([0.15, 0.3, 1.0, 2.0, 5.0, 15.0, 45.0, 90.0]))
    surface = np.asarray(
        lut.apparent_temperature(sea.surface_radiance(0.0, depression), "lb"), dtype=np.float64
    )

    assert not np.all(np.diff(surface) > 0.0)  # not the monotone ramp
    argmin = int(np.argmin(surface))
    assert 0 < argmin < surface.size - 1  # an interior minimum
    assert 2.0 <= math.degrees(depression[argmin]) <= 15.0
    # 289.47 before PH.1; the latent flux joined the cool skin's net loss and the skin fell
    # ~0.3 K (the latent term is of the longwave term's size at 5 m/s and 50 % RH).
    assert surface[-1] == pytest.approx(289.16, abs=0.3)  # nadir, warmest
    assert surface[argmin] == pytest.approx(283.7, abs=0.5)
    assert surface[-1] - surface[argmin] > 4.0  # a real, findable feature, not noise


def test_the_atmosphere_pulls_the_far_field_back_toward_air_temperature(rig) -> None:
    """At 4 km there is more air than sea in the ray, so the path fills the grazing sea back in.

    Without this the near-horizon sea would read its surface value; with it, the observed profile
    near the horizon rises toward T_air. It is why range, not just angle, sets maritime background
    contrast — and why the sea profile cannot be tabulated against angle alone for a moving camera.
    """
    _, lut, _ = rig
    _, sea = _sea(rig)
    near_horizon = math.radians(0.3)

    surface = float(
        np.atleast_1d(lut.apparent_temperature(sea.surface_radiance(0.0, near_horizon), "lb"))[0]
    )
    observed = float(np.atleast_1d(sea.apparent_temperature_k(0.0, near_horizon))[0])

    assert float(slant_range_m(CAMERA_HEIGHT_M, near_horizon)) == pytest.approx(4068.0, rel=0.02)
    assert observed > surface + 1.5
    assert observed < T_AIR_K


def test_roughening_the_sea_flattens_the_profile(rig) -> None:
    """A 15 m/s sea has a shallower nadir-to-minimum span than a glassy one.

    Tilted facets present less grazing incidence (raising ε, pulling the sea toward its own
    temperature) and reflect a broader slice of sky, and both smooth the feature. Flat calm is not
    a mirror either — Cox-Munk's intercept keeps 3.0e-3 of slope variance at zero wind.
    """
    _, lut, _ = rig
    depression = np.radians(np.array([0.3, 1.0, 2.0, 5.0, 15.0, 45.0, 90.0]))

    spans = []
    for wind in (0.0, 15.0):
        _, sea = _sea(rig, wind=wind)
        profile = np.asarray(
            lut.apparent_temperature(sea.surface_radiance(0.0, depression), "lb"), dtype=np.float64
        )
        spans.append(float(np.max(profile) - np.min(profile)))

    calm, rough = spans
    assert rough < calm
    assert rough < 0.75 * calm


def test_overcast_flattens_the_profile_only_as_far_as_the_cloud_base_is_warm(rig) -> None:
    """Cloud compresses the span, but not to zero, and the reason is checkable.

    A sea seen from 20 m reflects the sky near the horizon. Under overcast that sky is the cloud
    base read *through the air in front of it* (ADR 0126's range term on the plane-parallel
    blend, ADR 0146): on this rig the base is 1320 m up at 5.0 °C, and the reflected sky runs
    from 17.3 °C at 0.3° to 7.2 °C at the zenith, where the clear sky runs from 17.3 °C to
    −47 °C. So the reflected term still varies, a little, and the span closes to what that
    variation and the emissivity leave: **1.27 K against 5.62 K clear**, a ratio of 0.23. It
    would be near-total only if the base were at the surface. Cloud comes from the
    WeatherSeries, never from the environment preset (CLAUDE.md #6).
    """
    _, lut, _ = rig
    depression = np.radians(np.array([0.3, 1.0, 2.0, 5.0, 15.0, 45.0, 90.0]))

    spans = []
    for cloud in (0.0, 1.0):
        _, sea = _sea(rig, cloud=cloud)
        profile = np.asarray(
            lut.apparent_temperature(sea.surface_radiance(0.0, depression), "lb"), dtype=np.float64
        )
        spans.append(float(np.max(profile) - np.min(profile)))

    clear, overcast = spans
    assert overcast < clear
    assert 0.12 < overcast / clear < 0.45


def test_the_emissivity_lut_matches_direct_band_averaging(rig) -> None:
    """The 0.25° incidence LUT is an optimisation, so it has to be shown not to be an error."""
    response, _, table = rig
    _, sea = _sea(rig)
    angles = np.array([0.37, 13.1, 44.9, 71.3, 83.7, 87.2, 89.6])
    mu = np.cos(np.radians(angles))
    assert (
        np.max(np.abs(sea.emissivity(mu) - band_directional_emissivity(table, response, mu))) < 1e-3
    )


# -- construction guards ------------------------------------------------------------------


def test_the_sea_cannot_be_built_on_a_different_weather(rig) -> None:
    """The sea is mostly reflected sky; the two disagreeing about weather breaks #6."""
    response, _, table = rig
    with pytest.raises(TypeError, match="WeatherSeries"):
        SeaModel(object(), table, response, bulk_sst_k=290.0)  # type: ignore[arg-type]


def test_a_sky_in_the_other_radiance_form_is_refused(rig) -> None:
    """The bug this caught, in the one place it can be caught (ADR 0021).

    The sea is mostly *reflected sky*, so the two radiances are added together -- and `lb` and
    `lb_q` differ by about 1e19. A sea in energy form reflecting a sky in photon form therefore
    does not produce a slightly wrong sea: it produces one whose apparent temperature pins at the
    LUT ceiling, 1000 K, over the entire water surface. That is exactly what a maritime MWIR
    render did, uniformly and silently, because the script built the `SeaModel` without a quantity
    and got the `lb` default while the camera ran on `lb_q`.
    """
    response, _, table = rig
    sky, _ = _sea(rig)
    assert sky.quantity == "lb"
    with pytest.raises(ValueError, match="1e19"):
        SeaModel(sky, table, response, bulk_sst_k=290.0, quantity="lb_q")
    # ...and the matching pair is accepted, so the guard is about agreement and not about a form.
    assert SeaModel(sky, table, response, bulk_sst_k=290.0, quantity="lb") is not None


def test_the_skin_is_derived_from_the_bulk_and_never_authored_beside_it(rig) -> None:
    """MM.4: the deficit is computed from the scene's own wind and flux, not passed in.

    It used to be a constructor argument defaulting to zero, which meant every maritime scene
    rendered a skin exactly equal to its bulk SST unless somebody remembered to type a number --
    and a number typed there could contradict the wind the same scene was using to roughen the
    surface. There is now nowhere to type it.
    """
    response, _, table = rig
    sky, sea = _sea(rig)
    with pytest.raises(TypeError):
        SeaModel(sky, table, response, bulk_sst_k=290.0, cool_skin_k=0.2)
    with pytest.raises(ValueError, match="positive absolute temperature"):
        SeaModel(sky, table, response, bulk_sst_k=0.0)
    with pytest.raises(ValueError, match="come as a pair"):
        SeaModel(sky, table, response, bulk_sst_k=290.0, latitude_deg=57.0)
    assert sea.skin_temperature_k(0.0) < sea.bulk_sst_k


def test_the_profile_lut_matches_the_exact_evaluation(rig) -> None:
    """The depression LUT is what makes a frame renderable, so it has to be shown not to be a lie.

    Without it the sea costs a path-radiance quadrature per pixel, and a supersampled Boson frame
    asks for 2.6 million of them — one frame then takes longer than the rest of the pipeline put
    together. The grid is geometric and the interpolation is in log angle, because the structure is
    compressed against the horizon; interpolating linearly in the angle would flatten the cold band
    the whole model exists to produce.
    """
    _, sea = _sea(rig)
    horizon = sea.horizon_rad
    angles = np.geomspace(horizon * 1.01, 0.5 * np.pi * 0.99, 77)

    lut = sea.apparent_temperature_k(0.0, angles)
    exact = sea.apparent_temperature_exact_k(0.0, angles)
    assert np.max(np.abs(lut - exact)) < 0.02  # 20 mK against a 50 mK NETD


def test_the_profile_lut_is_far_cheaper_than_the_exact_path(rig) -> None:
    """The LUT exists to answer a frame's worth of angles; the exact path cannot.

    Stated as a **ratio against the exact path on the same machine**, not as a wall-clock bound.
    An absolute bound is a test of the machine as much as of the code: this assertion was
    `< 1.0 s` and failed twice on a workstation at load average 12 while passing in isolation,
    which tells a reader nothing about the LUT. A ratio cancels the load, because both halves are
    slowed by the same amount.
    """
    import time

    _, sea = _sea(rig)
    sea.profile(0.0)  # build once, as a render would during settling

    # The exact path is far too slow for a frame, so it is timed on a sample and scaled.
    sample = np.geomspace(sea.horizon_rad, 0.5 * np.pi, 200)
    start = time.perf_counter()
    sea.apparent_temperature_exact_k(0.0, sample)
    per_angle_exact = (time.perf_counter() - start) / sample.size

    angles = np.geomspace(sea.horizon_rad, 0.5 * np.pi, 2_000_000)
    start = time.perf_counter()
    out = sea.apparent_temperature_k(0.0, angles)
    per_angle_lut = (time.perf_counter() - start) / angles.size

    assert out.shape == angles.shape
    # Measured at about 470x per angle on a loaded workstation. The bound is set well below that
    # rather than at it: the ratio still moves with cache behaviour, and a threshold sitting on the
    # measurement is the same fragility as the wall-clock bound this replaced, one level up.
    assert per_angle_exact / per_angle_lut > 100.0
