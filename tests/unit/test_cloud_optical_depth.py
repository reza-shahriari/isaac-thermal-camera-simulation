"""A cloud has an optical depth and a distance (AT.11, ADR 0126).

Before this step a cloud was a switch: the environment preset authored one transmittance, every
covered ray took ``eps L_B(T_base) + tau L_clear`` with the same ``eps``, and ``L_B(T_base)`` was
evaluated as though the cloud were at zero range. Measured on a rendered 640x512 aerial frame,
**55 % of the pixels fell inside a single 2.3 K histogram bin** -- one flat level covering most of
the picture, which is what "the clouds are always white" describes.

Two things are wrong there and this file pins both.

* A cloud's LWIR emissivity follows from its **liquid water path**, not from a preset: Shaw &
  Nugent 2013 (Eur. J. Phys. 34 S111) §4 give it as ``1 - exp(-0.79 tau_vis)`` for a flux, and
  that 0.79 is ``1.58 x 0.5`` -- the diffusivity factor times the LWIR/visible optical depth
  ratio. Decomposed that way one relation serves a flux *and* a ray, because a ray brings its own
  airmass instead of the diffusivity.
* A cloud is **kilometres away**, and the air in front of it attenuates its excess over the clear
  sky by ``tau(R)``, which runs as ``1/sin(theta)``. Shaw & Nugent's figure 7(a) is the
  observation this reproduces: in a calibrated all-sky LWIR image "the overall spatial pattern is
  dominated by the variation of atmospheric path length with angle from the zenith", not by the
  clouds.

The load-bearing test here is `test_a_cloud_of_zero_emissivity_is_the_clear_sky_to_the_bit`,
because every other claim is a perturbation of it: if the clear-sky limit drifted, so would every
frame with a gap in the cloud.
"""

from __future__ import annotations

import math
import pathlib

import numpy as np
import pytest
import yaml

from irsim.atmosphere.cloud import (
    CLOUD_AIRMASS_FLOOR_DEG,
    CLOUD_OD_RATIO,
    DIFFUSIVITY_FACTOR,
    OPAQUE_OPTICAL_DEPTH,
    cloud_airmass,
    cloud_base_temperature_k,
    cloud_emissivity,
    cloud_radiance,
    cloud_radiance_at_range,
    cloud_reflectance,
    generate_sky_cloud,
)
from irsim.atmosphere.cloud_deck import generate_cloud_deck
from irsim.config.environment import CloudSpec, load_environment_preset
from irsim.config.loader import load_sensor_config
from irsim.radiometry.lut_files import load_band_lut_for_config, load_band_response_for_config
from irsim.scene import Scene

REPO = pathlib.Path(__file__).resolve().parents[2]
SENSOR_YAML = REPO / "configs" / "sensors" / "flir_boson_640_lwir.yaml"
PHANTOM_SCENE = REPO / "configs" / "scenes" / "phantom3_outbound_pointwise.yaml"
BAND = "lwir"

#: The frame the Phantom outbound clip films: a 24.75 degree vertical field aimed 20 degrees up.
AIM_ELEVATION_DEG = 20.0
VFOV_DEG = 24.752


def _scene(path: pathlib.Path) -> Scene:
    sensor = load_sensor_config(SENSOR_YAML)
    lut = load_band_lut_for_config(sensor, REPO / "data" / "lut")
    response = load_band_response_for_config(sensor, REPO / "data")
    return Scene.from_file(
        path,
        {BAND: lut},
        responses={BAND: response},
        quantity=sensor.sensor.quantity,
    )


@pytest.fixture(scope="module")
def phantom() -> Scene:
    return _scene(PHANTOM_SCENE)


# -- the published relation, decomposed ---------------------------------------------------


def test_the_flux_emissivity_is_the_published_one_to_the_bit() -> None:
    """eps(tau, diffusivity) == 1 - exp(-0.79 tau), Shaw & Nugent 2013 §4.

    Not an approximation of it: 0.79 is exactly 1.58 x 0.5, so the two constants this module
    authors reproduce the one constant the paper quotes. Moving either by a percent fails here,
    which is the point -- they are not free parameters dressed up as physics.
    """
    tau = np.array([0.0, 0.05, 0.28, 1.0, 2.0, 4.0, 12.0])
    published = 1.0 - np.exp(-0.79 * tau)
    assert np.allclose(cloud_emissivity(tau, DIFFUSIVITY_FACTOR), published, rtol=0.0, atol=1e-15)
    product = DIFFUSIVITY_FACTOR * CLOUD_OD_RATIO
    assert product == pytest.approx(0.79, abs=1e-12)


def test_a_ray_brings_its_own_airmass_and_a_flux_brings_the_diffusivity() -> None:
    """The decomposition is what lets one relation serve both, so the two must differ correctly."""
    zenith = float(cloud_emissivity(1.0, float(cloud_airmass(math.pi / 2)[()])))
    flux = float(cloud_emissivity(1.0))
    oblique = float(cloud_emissivity(1.0, float(cloud_airmass(math.radians(20.0))[()])))
    # A ray straight up crosses one deck thickness; a flux crosses 1.58; a 20 degree ray 2.92.
    assert zenith == pytest.approx(1.0 - math.exp(-0.5), abs=1e-12)
    assert flux > zenith
    assert oblique > flux
    assert float(cloud_airmass(math.radians(20.0))[()]) == pytest.approx(2.9238, abs=1e-3)


def test_the_airmass_is_one_over_sin_and_is_bounded_at_the_horizon() -> None:
    el = np.radians(np.array([90.0, 30.0, 10.0, 3.0, CLOUD_AIRMASS_FLOOR_DEG, 0.5, 0.0]))
    m = cloud_airmass(el)
    assert m[0] == pytest.approx(1.0, abs=1e-12)
    assert m[1] == pytest.approx(2.0, abs=1e-12)
    assert np.all(np.diff(m[:4]) > 0.0)  # steeper is shorter, monotonically
    # Unbounded 1/sin overflows on the horizon row of a frame; the floor is the same value there.
    ceiling = 1.0 / math.sin(math.radians(CLOUD_AIRMASS_FLOOR_DEG))
    assert np.all(m[4:] == pytest.approx(ceiling, abs=1e-9))
    assert np.all(np.isfinite(m))


def test_an_optically_thin_cloud_is_thin_and_a_thick_one_is_a_blackbody() -> None:
    """The saturation is the physics, and it is why authoring one tau was wrong.

    Shaw & Nugent: cloud optical depth can be retrieved from a radiance measurement "up to a
    value of approximately 4", beyond which a cloud emits as a blackbody. Both ends must hold, or
    the model is a switch again with extra arithmetic.
    """
    thin = float(cloud_emissivity(0.3))
    opaque = float(cloud_emissivity(OPAQUE_OPTICAL_DEPTH))
    assert 0.15 < thin < 0.25  # visible from the ground, but mostly sky behind it
    assert opaque > 0.95
    # Doubling an already-opaque cloud changes almost nothing, which is why it cannot be retrieved
    assert float(cloud_emissivity(2.0 * OPAQUE_OPTICAL_DEPTH)) - opaque < 0.05


def test_emissivity_refuses_a_negative_depth_and_a_non_positive_path() -> None:
    with pytest.raises(ValueError, match="cannot be negative"):
        cloud_emissivity(np.array([1.0, -0.001]))
    with pytest.raises(ValueError, match="path factor"):
        cloud_emissivity(1.0, 0.0)


# -- the cloud at a range ------------------------------------------------------------------


def test_a_cloud_of_zero_emissivity_is_the_clear_sky_to_the_bit() -> None:
    """The limit every gap in every cloud takes. Not approximately: `np.where`, not arithmetic."""
    rng = np.random.default_rng(3)
    clear = rng.uniform(5.0, 40.0, 512)
    beyond = rng.uniform(1.0, 30.0, 512)
    tau = rng.uniform(0.0, 1.0, 512)
    out = cloud_radiance_at_range(clear, beyond, tau, 44.0, np.zeros(512))
    assert np.array_equal(out, clear)


def test_an_opaque_cloud_reads_the_path_in_front_of_it_plus_its_own_emission() -> None:
    """eps = 1 gives L_path(R) + tau(R) L_base, which is an emitter inserted at R and nothing else.

    Written through the clear-sky identity rather than through L_path, so the test would catch a
    sign error in the `(l_base - l_beyond)` difference that a direct restatement would not.
    """
    clear, beyond, tau, l_base = 22.0, 9.0, 0.63, 31.0
    out = float(cloud_radiance_at_range(clear, beyond, tau, l_base, 1.0)[()])
    l_path = clear - tau * beyond  # the definition of `sky_beyond`
    assert out == pytest.approx(l_path + tau * l_base, abs=1e-12)


def test_the_layered_models_own_identity_holds_so_the_clear_limit_is_exact(
    phantom: Scene,
) -> None:
    """L_path(R, θ) + τ_band(R, θ) · L_beyond(R, θ) == L_sky(θ), exactly.

    `cloud_radiance_at_range` is built on this and would be silently wrong without it -- the
    band transmittance is a weighted sum over spectral classes and ``sky_beyond`` is a
    τ_k-weighted mean, and it is only because they are weighted with the *same* weights that the
    two limits are exact rather than k-distribution approximations.
    """
    atm = phantom.sky_models[BAND].atmosphere
    t0 = phantom.t0_s
    for el_deg, distance in ((90.0, 1000.0), (20.0, 3000.0), (5.0, 12000.0)):
        el = math.radians(el_deg)
        tau = float(atm.transmittance(BAND, t0, distance, el)[()])
        beyond = atm.sky_beyond(BAND, t0, distance, el)
        path = atm.path_radiance(BAND, t0, distance, el)
        assert path + tau * beyond == pytest.approx(atm.sky_radiance(BAND, t0, el), rel=1e-12)


# -- what it does to a frame ---------------------------------------------------------------


def _frame_angles(rows: int = 512, cols: int = 640) -> tuple[np.ndarray, np.ndarray]:
    half = 0.5 * VFOV_DEG
    el = np.radians(np.linspace(AIM_ELEVATION_DEG - half, AIM_ELEVATION_DEG + half, rows))
    az = np.radians(np.linspace(-15.5, 15.5, cols))
    return el[:, None] * np.ones((1, cols)), az[None, :] * np.ones((rows, 1))


def test_the_cloud_in_a_frame_is_no_longer_one_flat_level(phantom: Scene) -> None:
    """The complaint, measured. A cloud at zero range with one emissivity is one number.

    The comparison is made on the *same* scene and the *same* field, with only the preset's
    opacity model swapped, so nothing but the model under test can move the result.
    """
    sky = phantom.sky_models[BAND]
    env = phantom.environment
    assert env is not None and env.clouds.optical_depth is not None
    el, az = _frame_angles()
    field = generate_sky_cloud(
        env.clouds.beta, float(phantom.weather.at(phantom.t0_s).cloud_fraction), 7
    )
    density = field.density(el, az)
    # The cloud's **core**: rays at the deck's full depth. Those are the pixels the old model
    # collapsed onto one number; its fringe always had a ramp, because (1 - tau) d is a ramp.
    core = density >= 1.0

    graded = np.asarray(sky.apparent_temperature_field(phantom.t0_s, el, density))
    # The old model, evaluated by hand from the same pieces: one emissivity, cloud at zero range.
    _, l_base = sky._cloud(phantom.t0_s)
    flat = np.asarray(
        sky.lut.apparent_temperature(
            cloud_radiance(sky.clear_radiance(phantom.t0_s, el), l_base, 0.0, density), sky.quantity
        )
    )
    assert core.mean() > 0.3, "the test frame must actually contain a cloud core"
    flat_spread = float(flat[core].max() - flat[core].min())
    graded_spread = float(graded[core].max() - graded[core].min())
    assert flat_spread < 1e-3, "an opaque cloud at zero range is one number, by construction"
    assert graded_spread > 0.8, f"the cloud is still flat: {graded_spread:.3f} K across the frame"


def test_the_cloud_is_cooler_where_the_air_in_front_of_it_is_thicker(phantom: Scene) -> None:
    """The sign of the range term, which is the one thing that could be backwards.

    The intervening air is *warmer* than the cloud base, so a longer path pulls an opaque cloud's
    apparent temperature **up** toward the air, not down -- and the longest path is at the lowest
    elevation. Getting this backwards would still produce a gradient, and a plausible one.
    """
    sky = phantom.sky_models[BAND]
    el = np.radians(np.array([8.0, 20.0, 32.0, 90.0]))
    opaque = np.ones(el.shape)
    t = np.asarray(sky.apparent_temperature_field(phantom.t0_s, el, opaque))
    t_base = sky.cloud_base_temperature_k(phantom.t0_s)
    t_air = float(phantom.weather.at(phantom.t0_s).t_air_k)
    assert t_air > t_base, "this anchor needs a cloud base colder than the surface air"
    assert np.all(np.diff(t) < 0.0), f"apparent T must fall as the path shortens: {t}"
    # Bounded by the two ends it interpolates: never above the air, never below the base.
    assert t_base - 0.5 < t.min() and t.max() < t_air + 0.5


def test_a_tau_preset_is_untouched_by_any_of_this() -> None:
    """Every scene authored before ADR 0126 renders bit-identically.

    The `tau` branch must still call `cloud_radiance` with the authored transmittance, so this
    compares the model's own output against that function directly rather than against a stored
    number, which would rot.
    """
    scene = _scene(REPO / "configs" / "scenes" / "quad_flight_clear_noon.yaml")
    sky = scene.sky_models[BAND]
    env = scene.environment
    assert env is not None and env.clouds.optical_depth is None
    el = np.radians(np.linspace(1.0, 89.0, 97))
    density = np.linspace(0.0, 1.0, 97)
    _, l_base = sky._cloud(scene.t0_s)
    expected = cloud_radiance(sky.clear_radiance(scene.t0_s, el), l_base, env.clouds.tau, density)
    assert np.array_equal(sky.radiance_field(scene.t0_s, el, density), expected)
    with pytest.raises(ValueError, match="no per-ray emissivity"):
        sky.cloud_ray_emissivity(el, density)


# -- what a preset may say -----------------------------------------------------------------


def test_a_preset_authors_one_opacity_or_the_other_but_never_both() -> None:
    assert CloudSpec(tau=0.2).emissivity == pytest.approx(0.8)
    assert CloudSpec(optical_depth=3.0).optical_depth == pytest.approx(3.0)
    with pytest.raises(ValueError, match="not both"):
        CloudSpec(tau=0.2, optical_depth=3.0)
    # An unauthored tau keeps its default and does not count as a second answer.
    assert CloudSpec(optical_depth=3.0).tau == pytest.approx(0.0)


def test_an_optical_depth_preset_refuses_to_hand_out_one_emissivity() -> None:
    """Returning 1 - 0 = 1 would render a thin cirrus as an opaque deck, silently."""
    thin = CloudSpec(optical_depth=0.3)
    with pytest.raises(AttributeError, match="cloud_emissivity"):
        _ = thin.emissivity


def test_the_shipped_presets_say_what_they_mean() -> None:
    cumulus = load_environment_preset("scattered_cumulus")
    assert cumulus.clouds.optical_depth is not None
    # Fair-weather cumulus, and thick enough to be a blackbody in the core -- which is real, and
    # is why the interesting part of a cloud in this band is its edge.
    assert cumulus.clouds.optical_depth > OPAQUE_OPTICAL_DEPTH
    # `clear_dry` is untouched, so every scene that names it is unchanged.
    assert load_environment_preset("clear_dry").clouds.optical_depth is None
    scene = yaml.safe_load(PHANTOM_SCENE.read_text(encoding="utf-8"))
    assert scene["scene"]["environment_preset"] == "scattered_cumulus"


def test_the_cloud_base_temperature_matches_the_papers_modelled_altostratus() -> None:
    """A public anchor for the one number the whole cloud reads: T at the base altitude.

    Shaw & Nugent model an opaque altostratus with a 2.4 km base in the 1976 US Standard
    Atmosphere and report a mean temperature of **271.6 K**. The environmental lapse rate gives
    272.55 K for the same base from the standard's 288.15 K surface -- within a kelvin, which is
    as close as a constant lapse rate gets to a layered standard atmosphere.

    Their cirrus anchor is deliberately *not* asserted: a 10 km base in the same model reads
    216.8 K and a constant 6.5 K/km gives 223.15 K, 6.3 K out, because the real profile bends
    toward the tropopause. High cloud needs an authored base, not the LCL, and this repo has no
    scene with one.
    """
    altostratus = cloud_base_temperature_k(288.15, 2400.0, 0.0065)
    assert altostratus == pytest.approx(271.6, abs=1.5)


# -- what the visible band does with the same optical depth ---------------------------------


def test_a_cloud_returns_more_light_the_deeper_it_is_and_never_more_than_all_of_it() -> None:
    """The two-stream reflectance, which is what stops a marched cloud drawing as a flat shape.

    Three properties, each of which would be a visible defect if it failed: it is zero where
    there is no cloud (so a cloud edge meets clear sky continuously), it rises monotonically
    with optical depth (so a cumulus has an inside), and it stays inside [0, 1) however deep the
    cloud (so a conservatively scattering layer never returns more light than reached it).
    """
    tau = np.array([0.0, 0.5, 1.0, 3.0, 10.0, 30.0, 100.0, 1e6])
    mu0 = math.cos(math.radians(40.0))
    r = cloud_reflectance(tau, mu0)
    assert float(r[0]) == 0.0
    assert np.all(np.diff(r) > 0.0)
    assert np.all((r >= 0.0) & (r < 1.0))
    assert float(r[-1]) > 0.99, "an arbitrarily deep cloud is arbitrarily close to white"
    # The published form, restated independently rather than recomputed from the same expression.
    g = 0.85
    expected = [(1 - g) * t / (2 * mu0 + (1 - g) * t) for t in tau]
    assert np.allclose(r, expected, atol=0.0)


def test_the_reflectance_is_a_spread_and_not_one_value_over_a_real_frame() -> None:
    """The whole reason it replaced a constant: over a marched frame it must actually vary.

    A constant reflectance draws every cloudy pixel the same, which once the opacity saturates
    is a grey cut-out with a hard edge -- what the dome did before it marched the deck.
    """
    deck = generate_cloud_deck(
        beta=2.8, cloud_fraction=0.45, seed=7, base_m=1071.0, optical_depth=12.0
    )
    el = np.radians(np.linspace(8.0, 32.0, 96))[:, None] * np.ones((1, 120))
    az = np.radians(np.linspace(-15.0, 15.0, 120))[None, :] * np.ones((96, 1))
    tau = deck.march(el, az).optical_depth
    r = cloud_reflectance(tau, math.cos(math.radians(40.0)))
    cloudy = tau > 0.5
    assert cloudy.mean() > 0.2, "the test frame must contain cloud"
    lo, hi = np.percentile(r[cloudy], [5, 95])
    assert hi - lo > 0.25, f"only {hi - lo:.2f} of reflectance across a frame of cumulus"
    assert float(r[cloudy].max()) > 0.8 and float(r[cloudy].min()) < 0.3
