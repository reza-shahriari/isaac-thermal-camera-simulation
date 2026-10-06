"""The two bands' agreement about the cloud is measured, not judged (WX.26, §7.5 criterion 3).

ADR 0190 shipped the infrared reading the cloudscape with the rendered pair "judged by eye".
These tests pin the measurement that replaced that: the score itself on planes with a known
answer, the knobs that would make the bands disagree (the visible density scale, which now
reaches the deck), the cirrus genus the infrared had no path for, and the temperature law a
cloud far above the mixed layer needs. The in-engine half -- the score on frames both cameras
rendered -- is `render_phantom4.py --cloud-tier pixel` and `scripts/cloud_band_agreement.py`.
"""

from __future__ import annotations

import dataclasses
import inspect
import pathlib

import numpy as np
import pytest

from irsim.atmosphere.cloud import CLOUD_OD_RATIO
from irsim.atmosphere.weather_fx import (
    OPAQUE_OPTICAL_DEPTH,
    WeatherFxDeck,
    ensure_weather_fx_on_path,
    weather_fx_available,
)
from irsim.io.truth import truth_planes
from irsim.radiometry.constants import DRY_ADIABATIC_LAPSE_K_PER_M
from irsim.validation.cloud_bands import (
    VISIBLE_TRANSMITTANCE_FLOOR,
    band_emissivity_from_visible,
    cloud_band_agreement,
)

REPO = pathlib.Path(__file__).resolve().parents[2]
DRIVER = REPO / "scripts" / "render_phantom4.py"
needs_weather_fx = pytest.mark.skipif(
    not weather_fx_available(), reason="the isaac-weather-fx submodule is not checked out"
)
SMALL = dict(weather_cells=128, shape_cells=32, detail_cells=24)


# --- the score --------------------------------------------------------------------------------


def test_the_visible_transmittance_implies_the_band_emissivity_through_the_ratio() -> None:
    """tau_B = r tau_vis, so T_B = T_vis ** r: the one number that links the two bands."""
    t_vis = np.array([1.0, 0.5, 0.25, 0.01])
    eps = band_emissivity_from_visible(t_vis, 0.5)
    assert np.allclose(eps, 1.0 - np.sqrt(t_vis))
    assert np.allclose(band_emissivity_from_visible(t_vis, 1.0), 1.0 - t_vis)
    with pytest.raises(ValueError):
        band_emissivity_from_visible(t_vis, 0.0)


def test_identical_clouds_score_perfectly_and_a_shifted_one_does_not() -> None:
    rng = np.random.default_rng(1)
    t_vis = np.clip(rng.uniform(0.0, 1.0, (64, 96)) ** 3, 0.0, 1.0)
    t_ir = np.power(t_vis, CLOUD_OD_RATIO)
    same = cloud_band_agreement(t_ir, t_vis)
    assert (
        same.iou == 1.0
        and same.p95_emissivity_error == 0.0
        and same.correlation == pytest.approx(1.0)
    )
    assert same.passes()
    # A cloud mask moved by 8 columns: masks overlap less than fully, the emissivity error grows.
    moved = cloud_band_agreement(np.roll(t_ir, 8, axis=1), t_vis)
    assert moved.iou < 0.9 and moved.p95_emissivity_error > 0.1 and not moved.passes()
    assert moved.pixels == 64 * 96
    # A pure-sky pair: no cloud either side is perfect agreement with nothing to correlate.
    clear = cloud_band_agreement(np.ones((8, 8)), np.ones((8, 8)))
    assert clear.iou == 1.0 and clear.p95_emissivity_error == 0.0 and np.isnan(clear.correlation)
    assert clear.cloud_fraction_ir == 0.0 and clear.cloud_fraction_vis == 0.0


def test_below_the_visible_floor_an_opaque_infrared_cloud_is_agreement() -> None:
    """weather-fx's march ends a ray at T = 0.004; the band's emissivity there is 1 - 0.004**0.5
    = 0.937. An infrared march that went on to 0.999 is not 0.06 wrong, it is what the floor
    implies or more; an infrared march that stopped *short* of it is."""
    t_vis = np.full((4, 4), VISIBLE_TRANSMITTANCE_FLOOR)
    implied = 1.0 - VISIBLE_TRANSMITTANCE_FLOOR**CLOUD_OD_RATIO
    deeper = cloud_band_agreement(np.full((4, 4), 1e-3), t_vis)
    assert deeper.p95_emissivity_error == 0.0 and deeper.iou == 1.0
    shallower = cloud_band_agreement(np.full((4, 4), 1.0 - (implied - 0.1)), t_vis)
    assert shallower.p95_emissivity_error == pytest.approx(0.1, abs=1e-9)


def test_the_score_masks_and_refuses_mismatched_planes() -> None:
    t_vis = np.ones((6, 6))
    t_vis[:3] = 0.1
    t_ir = np.power(t_vis, 0.5)
    t_ir[5, 5] = np.nan
    lower = cloud_band_agreement(t_ir, t_vis, mask=np.arange(6)[:, None] >= 3)
    assert lower.pixels == 17 and lower.cloud_fraction_ir == 0.0
    with pytest.raises(ValueError):
        cloud_band_agreement(np.ones((6, 6)), np.ones((6, 7)))


# --- what reaches the frame -------------------------------------------------------------------


def test_the_truth_planes_carry_the_visible_transmittance_beside_the_marched_one() -> None:
    k = 2
    ir = np.linspace(0.0, 1.0, 4 * k * 6 * k).reshape(4 * k, 6 * k)
    vis = ir**2
    planes = truth_planes({"cloud_transmittance": ir, "cloud_transmittance_vis": vis}, (4, 6))
    assert planes.planes["cloud_transmittance_vis"].shape == (4, 6)
    assert planes.planes["cloud_transmittance_vis"].dtype == np.float32
    assert np.allclose(planes.planes["cloud_transmittance_vis"], vis[1::2, 1::2])
    # Never invented: without the companion's plane there is none.
    assert "cloud_transmittance_vis" not in truth_planes({"cloud_transmittance": ir}, (4, 6)).planes


def test_the_camera_takes_the_companion_s_transmittance_and_resamples_a_shrunk_layer() -> None:
    from irsim_isaac.pipeline.ir_camera import IrCamera

    assert inspect.signature(IrCamera).parameters["visible_cloud_transmittance"].default is None

    class Stub:
        def __init__(self, array: object) -> None:
            self.visible_cloud_transmittance = lambda: array

    same = np.random.default_rng(0).uniform(0.0, 1.0, (8, 12)).astype(np.float32)
    assert np.array_equal(IrCamera._visible_cloud_plane(Stub(same), (8, 12)), same)  # type: ignore[arg-type]
    # A layer drawn at 3/4 scale is stretched over the frame texel for texel, like its quad.
    small = np.arange(6 * 9, dtype=np.float32).reshape(6, 9)
    big = IrCamera._visible_cloud_plane(Stub(small), (8, 12))  # type: ignore[arg-type]
    assert big.shape == (8, 12) and big[0, 0] == small[0, 0] and big[-1, -1] == small[-1, -1]
    assert IrCamera._visible_cloud_plane(Stub(None), (8, 12)) is None  # type: ignore[arg-type]
    # The whole ray (no scene depth): an object's pixels hold the cloud *behind* the object,
    # which the renderer hides, so they are left out; with the depth cut they are kept.
    sky = np.ones((8, 12), dtype=bool)
    sky[3:5, 4:6] = False
    whole = IrCamera._visible_cloud_plane(Stub((same, False)), (8, 12), sky_mask=sky)  # type: ignore[arg-type]
    assert np.isnan(whole[~sky]).all() and np.array_equal(whole[sky], same[sky])
    cut = IrCamera._visible_cloud_plane(Stub((same, True)), (8, 12), sky_mask=sky)  # type: ignore[arg-type]
    assert np.array_equal(cut, same)
    with pytest.raises(ValueError):
        IrCamera._visible_cloud_plane(Stub(np.ones(5)), (8, 12))  # type: ignore[arg-type]


def test_the_driver_stores_the_companion_plane_and_scores_every_frame() -> None:
    src = DRIVER.read_text(encoding="utf-8")
    assert '"--cloud-stride"' in src, "cloud_stride was read off args without being an argument"
    assert "visible_cloud_transmittance=(" in src
    assert "cam.cloud_band_agreement()" in src
    assert '"cloud_band_agreement": _agreement_summary(agreements)' in src


# --- the knobs that would make the bands disagree ---------------------------------------------


def _cloudscape(genus: str, **changes):  # type: ignore[no-untyped-def]
    ensure_weather_fx_on_path()
    from weather_fx.core import cloudscape as cs

    profile = dataclasses.replace(cs.CLOUDSCAPE_TYPES[genus], **changes)
    base = profile.base_default_m if not profile.convective else 1200.0
    return cs.Cloudscape(
        cover=0.4,
        base_m=base,
        profile=profile,
        seed=5,
        mixed_layer_top_m=None if profile.convective else 1200.0,
        **SMALL,
    )


def _rays() -> tuple[np.ndarray, np.ndarray]:
    el = np.radians(np.linspace(15.0, 75.0, 16))[:, None]
    az = np.radians(np.linspace(0.0, 350.0, 36))[None, :]
    return np.broadcast_arrays(el, az)


@needs_weather_fx
def test_the_visible_density_scale_reaches_both_marches() -> None:
    """weather-fx's `clouds.density_scale` multiplies the visible march's extinction; left out of
    the deck the bands marched clouds of different depth (docs/weather/README.md §7). Doubling it
    doubles every ray's visible optical depth in both marches and the column's."""
    scape = _cloudscape("stratocumulus")
    el, az = _rays()
    one = WeatherFxDeck(scape, origin_m=(0.0, 0.0, 0.0))
    two = WeatherFxDeck(scape, origin_m=(0.0, 0.0, 0.0), density_scale=2.0)
    assert two.optical_depth == pytest.approx(2.0 * one.optical_depth)
    # Exactly double on every ray that stays short of the band's opacity, where the march
    # stops accumulating (OPAQUE_OPTICAL_DEPTH); never less anywhere.
    sky_1, sky_2 = one.march(el, az), two.march(el, az)
    short = sky_2.optical_depth * one.od_ratio < 0.9 * OPAQUE_OPTICAL_DEPTH
    assert short.sum() > 50
    assert np.allclose(
        sky_2.optical_depth[short], 2.0 * sky_1.optical_depth[short], rtol=1e-9, atol=1e-12
    )
    # A doubled ray that stopped early is opaque in the band (it may stop on a smaller
    # overshoot than the single one did, so `>=` everywhere is not the claim).
    assert np.all(sky_2.optical_depth[~short] * one.od_ratio >= 0.9 * OPAQUE_OPTICAL_DEPTH)
    hit_1 = one.march_to(el, az, 3000.0, lambda h: np.ones_like(h))
    hit_2 = two.march_to(el, az, 3000.0, lambda h: np.ones_like(h))
    short = hit_2.optical_depth * one.od_ratio < 0.9 * OPAQUE_OPTICAL_DEPTH
    assert np.allclose(
        hit_2.optical_depth[short], 2.0 * hit_1.optical_depth[short], rtol=1e-9, atol=1e-12
    )
    assert np.all(hit_2.transmittance <= hit_1.transmittance + 1e-12)


@needs_weather_fx
def test_a_cirrus_reaches_the_infrared_high_thin_and_above_the_mixed_layer() -> None:
    """The one genus the cloudscape had no profile for. The deck reads it like any other: a base
    at 9 km, a layer the sky shows through (the march's emissivity is well under 1 on most cloudy
    rays), the mixed layer's top reported for the temperature law."""
    scape = _cloudscape("cirrus")
    deck = WeatherFxDeck(scape, origin_m=(0.0, 0.0, 0.0))
    assert deck.base_m == 9000.0 and deck.top_m == 10_500.0
    assert deck.mixed_layer_top_m == 1200.0
    assert WeatherFxDeck(_cloudscape("cumulus")).mixed_layer_top_m is None
    el, az = _rays()
    march = deck.march_to(el, az, np.inf, lambda h: np.ones_like(h))
    eps = march.emissivity
    assert (eps > 0.05).mean() > 0.2, "a 40 % cirrus field puts cloud on many rays"
    cloudy = eps[eps > 0.05]
    assert np.median(cloudy) < 0.9, "cirrus is optically thin: the sky shows through"
    assert np.all(march.emission_range_m[eps > 0.05] >= 9000.0 / np.sin(el.max()) * 0.99)


@needs_weather_fx
def test_a_cloud_above_the_mixed_layer_reads_the_environment_s_temperature(
    aerial_cloudy_sky,
) -> None:  # type: ignore[no-untyped-def]
    """Dry-adiabatic to the condensation level, the environment's lapse above it (S67). Lapsed dry
    all the way, a 9 km cirrus base read about 30 K colder than the air there."""
    sky = aerial_cloudy_sky
    t_air = sky.weather.at(0.0).t_air_k
    env = sky._atm.preset.profile.lapse_rate_k_per_m
    lcl = 1200.0
    # Below or at the mixed layer's top nothing changes.
    for base in (300.0, 1200.0):
        assert sky.cloud_base_temperature_k(0.0, base, mixed_layer_top_m=lcl) == pytest.approx(
            t_air - DRY_ADIABATIC_LAPSE_K_PER_M * base, abs=1e-9
        )
    high = sky.cloud_base_temperature_k(0.0, 9000.0, mixed_layer_top_m=lcl)
    assert high == pytest.approx(
        t_air - DRY_ADIABATIC_LAPSE_K_PER_M * lcl - env * (9000.0 - lcl), abs=1e-9
    )
    dry = sky.cloud_base_temperature_k(0.0, 9000.0)
    assert 20.0 < high - dry < 35.0
    # The deck carries the law to the sky model: an opaque cirrus emits at the environment's
    # temperature of its level, not the dry parcel's.
    scape = _cloudscape("cirrus", extinction_per_m=0.5)
    deck = WeatherFxDeck(scape, origin_m=(0.0, 0.0, 0.0))
    el = np.radians(np.full((4, 12), 60.0))
    az = np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False)[None, :] * np.ones((4, 1))
    march = sky.cloud_occlusion(0.0, el, az, deck, np.inf)
    opaque = march.transmittance < 1e-3
    assert opaque.any()
    t_read = sky._lut.apparent_temperature(
        np.asarray(march.path_radiance[opaque], dtype=np.float64), sky._q
    )
    assert float(np.max(t_read)) <= high + 1e-6
    assert float(np.min(t_read)) > high - env * 1500.0 - 1e-6


def test_a_headless_pixel_capture_draws_one_frame_at_full_resolution_and_a_fine_step() -> None:
    """The layer's sixteen-frame history ghosted earlier poses into every capture (frames are
    separate exposures 104 s apart) and averaged away the depth cut at the drone: measured p95
    emissivity error 0.10-0.56 against the infrared march. A headless capture draws one frame,
    at the G-buffer's own grid, with a step that resolves the patches' detail."""
    from irsim_isaac.weather_fx_stage import pixel_layer_settings

    settings = pixel_layer_settings()
    assert settings["layer_accumulate"] == 1
    assert settings["layer_scale"] == 1.0
    assert settings["layer_step_m"] <= 10.0
    # The march converged (ADR 0191): a wisp integrated, not sampled once.
    assert settings["layer_thin_tau"] <= 0.05 and settings["layer_step_growth"] <= 0.005
    assert settings["layer_max_steps"] >= 4096
    src = (REPO / "src" / "irsim_isaac" / "weather_fx_stage.py").read_text(encoding="utf-8")
    assert 'state.with_updates("clouds", **pixel_layer_settings())' in src


def test_rays_the_infrared_stops_short_on_are_left_out() -> None:
    """The deck ends a ray after 12 km inside the layer, the visible march after 80 km of range:
    below asin(thickness / 12 km) the infrared never crossed the whole layer."""
    from irsim.validation.cloud_bands import unreached_ray_mask

    el = np.radians(np.array([[2.0, 10.0, 14.0, 15.0, 30.0]]))
    keep = unreached_ray_mask(el, thickness_m=3000.0, max_path_m=12000.0)
    assert keep.tolist() == [[False, False, False, True, True]]
    assert unreached_ray_mask(el, thickness_m=0.0, max_path_m=12000.0).all()
    assert unreached_ray_mask(el, thickness_m=3000.0, max_path_m=np.inf).all()
    assert unreached_ray_mask(el, thickness_m=3000.0, max_path_m=40000.0).tolist() == [
        [False, True, True, True, True]
    ]


def test_the_layer_is_a_shell_round_the_planet() -> None:
    """A cloud layer sits at a height above the *curved* ground. Straight up it is the slab;
    at 10 degrees a 9 km base is 1.2 km nearer than the flat slab says, which is where the
    visible kernel marched a different cirrus from the infrared (ADR 0191). The two bands now
    share one geometry: weather-fx's kernel and this deck use one radius."""
    from irsim.atmosphere.sea import EARTH_RADIUS_M
    from irsim.atmosphere.weather_fx import altitude_m, shell_span

    o = np.array([0.0, 1.5, 0.0])
    up = np.array([0.0, 1.0, 0.0])
    near, far = shell_span(o, up, 910.0, 3910.0)
    assert near == pytest.approx(908.5, abs=1e-6) and far == pytest.approx(3908.5, abs=1e-6)
    el = np.radians(10.0)
    d = np.array([0.0, np.sin(el), -np.cos(el)])
    near, far = shell_span(o, d, 9000.0, 10500.0)
    flat = (9000.0 - 1.5) / np.sin(el)
    assert 1000.0 < flat - near < 1400.0
    p = o + near * d
    assert altitude_m(*p) == pytest.approx(9000.0, abs=1e-3)
    assert altitude_m(*(o + far * d)) == pytest.approx(10500.0, abs=1e-3)
    # A ray that meets the ground meets no cloud; one from above the layer enters at the top.
    down = np.array([0.0, -0.5, -np.sqrt(0.75)])
    near, far = shell_span(np.array([0.0, 100.0, 0.0]), down, 910.0, 3910.0)
    assert far <= near
    near, far = shell_span(np.array([0.0, 5000.0, 0.0]), down, 910.0, 3910.0)
    assert near == pytest.approx((5000.0 - 3910.0) / 0.5, rel=1e-3)
    assert altitude_m(*(np.array([0.0, 5000.0, 0.0]) + far * down)) == pytest.approx(910.0, abs=0.5)
    import re

    kernel = (
        REPO
        / "third_party"
        / "isaac-weather-fx"
        / "exts"
        / "weather.fx"
        / "weather_fx"
        / "gpu"
        / "cloud_march.py"
    )
    if kernel.exists():  # the two radii are one number (the kernel imports Warp, so read it)
        radius = re.search(r"^PLANET_RADIUS_M = ([\d_.]+)", kernel.read_text(), re.M)
        assert radius is not None and float(radius.group(1)) == EARTH_RADIUS_M


def test_a_wisp_of_a_few_pixels_is_not_judged_by_its_iou() -> None:
    """A wisp covering 0.1 % of the frame that sits at the mask threshold in both bands (band
    emissivity 0.52 against 0.48) has an IoU of zero and an emissivity error of 0.04: the
    emissivity criterion is what judges it. The same wisp at 1 % of the frame is judged by both."""
    t_vis = np.ones((100, 100))
    t_vis[50:52, 50:54] = (1.0 - 0.48) ** 2  # the visible implies band emissivity 0.48
    t_ir = np.ones((100, 100))
    t_ir[50:52, 50:54] = 1.0 - 0.52
    small = cloud_band_agreement(t_ir, t_vis)
    assert small.iou == 0.0 and small.p95_emissivity_error == pytest.approx(0.04, abs=1e-9)
    assert max(small.cloud_fraction_ir, small.cloud_fraction_vis) < 0.01
    assert small.passes()
    assert not small.passes(p95_max=0.01)
    t_vis[40:60, 40:50] = (1.0 - 0.48) ** 2
    t_ir[40:60, 40:50] = 1.0 - 0.52
    large = cloud_band_agreement(t_ir, t_vis)
    assert large.cloud_fraction_ir >= 0.01 and large.iou == 0.0 and not large.passes()


@needs_weather_fx
def test_no_cloud_is_marched_beyond_the_visible_kernel_s_range() -> None:
    """A 0.7-degree ray meets a 9 km cirrus shell at 266 km. weather-fx's kernel ends every ray
    at 80 km; so does the deck, or the infrared shows a cirrus the visible never marched to
    (measured: 34 % against 19 % of a frame at 10 degrees)."""
    scape = _cloudscape("cirrus", extinction_per_m=0.5)
    deck = WeatherFxDeck(scape, origin_m=(0.0, 1.5, 0.0))
    assert deck.max_range_m == 80_000.0
    el = np.radians(np.array([0.7, 4.2, 30.0]))[:, None]
    az = np.radians(np.linspace(0.0, 350.0, 36))[None, :]
    el, az = np.broadcast_arrays(el, az)
    march = deck.march_to(el, az, np.inf, lambda h: np.ones_like(h))
    assert np.all(march.transmittance[:2] == 1.0), "beyond 80 km: no cloud, as the visible has none"
    assert (march.transmittance[2] < 0.5).any(), "at 18 km the same shell is marched"
    sky = deck.march(el, az)
    assert np.all(sky.optical_depth[:2] == 0.0) and (sky.optical_depth[2] > 1.0).any()
