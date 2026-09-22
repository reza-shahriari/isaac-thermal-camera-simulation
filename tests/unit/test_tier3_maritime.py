"""Tier 3 maritime phenomenology on the synthetic sea fixture (MM.8).

The sea half of the ir-sim-testing skill's Tier 3 list, run through the whole engine-free chain --
MM.2's Cox-Munk slopes, MM.3's depression profile, MM.4's skin temperature, MS.1's layered
atmosphere, MS.2's sky above the horizon -- on a frame rather than on the models directly.
`tests/unit/test_sea_surface.py` checks `SeaModel`; this file checks what a *camera* reads.

⚠️ **Two of MM.8's three written criteria are backwards, and these tests measure rather than
assert them.** Both assume the sea behaves like an overcast ground scene, and MM.3 had already
found it does not (ADR 0078):

* "sea near the horizon reads colder than sea close in" -- it reads **warmer**, by 2.9 K here.
  The profile's coldest point is an interior trough about 5 degrees down, so a near-horizon frame
  sits on the *rising* side of it (1.2 K of the span), and the atmospheric path then roughly
  doubles that because the far ray is 7 km of air pulling toward T_air. The criterion is true only
  of sea past the trough, not of the band a shore or mast camera works in.
* "overcast collapses that span below 20 % of clear" -- it collapses to **57 %**. An overcast sky
  stops the reflection varying with angle, but the cloud base is still colder than the water and
  the atmospheric path does not care about cloud at all.

The third -- polarity flips with range -- holds, and is the one that matters for detection.

**And all three are statements about an extrapolation.** SE.1 measured this frame against the
angles published in-situ radiometry actually covers: **1.0000 of its sea is outside them.** The
50 deg validated limit is crossed 31 m from a 20 m camera and this frame starts at 530 m, so
nothing here is backed by a measurement of a sea. The isothermal identity holds throughout and
is not evidence to the contrary -- it holds for a wrong angular emissivity too (ADR 0118).

docs/physics-model.md §15 (Tier 3), §5.3; ADR 0078, ADR 0080
"""

from __future__ import annotations

import copy
import math
import pathlib

import numpy as np
import pytest
import yaml

from irsim.atmosphere.layered import LayeredAtmosphere
from irsim.atmosphere.library import load_atmosphere_preset
from irsim.atmosphere.sea import SeaModel, slant_range_m
from irsim.atmosphere.sea_envelope import VALIDATED_ZENITH_DEG
from irsim.atmosphere.sky import SkyModel
from irsim.config.environment import load_environment_preset
from irsim.config.gbuffer import UNMAPPED_MATERIAL_ID, GBuffer
from irsim.config.sensor import SensorConfig
from irsim.materials.library import MaterialLibrary
from irsim.materials.nk import load_nk_table
from irsim.materials.table import MaterialTable
from irsim.pipeline import PipelineConfig, PipelineState, run_frame
from irsim.radiometry.spectral_response import load_spectral_response
from irsim.thermal.weather import WeatherSample, WeatherSeries
from irsim.validation.aerial_scene import SceneTarget
from irsim.validation.maritime_scene import build_maritime_gbuffer

# GT.1: this whole module is the slow tier -- a validation bench or an end-to-end frame rather
# than a unit test. `make test` skips it; `make test-slow` and `make check` run it.
pytestmark = pytest.mark.slow

REPO = pathlib.Path(__file__).resolve().parents[2]
RESPONSE = REPO / "data" / "spectra" / "responses" / "boson_vox.csv"
SST_K = 290.0
T_AIR_K = 291.0
CAMERA_HEIGHT_M = 20.0
#: Boresight just below the horizon: the frame then spans 0.19-2.15 deg of depression, which is
#: 7 km of sea down to 530 m -- the range band a shore or mast camera actually works in.
BORESIGHT_DEG = -1.0
#: A vessel between the near sea (285.4 K) and the far sea (288.3 K), so its contrast changes sign.
VESSEL_T_K = 289.5


def _sensor(focal_mm: float | None = None):  # type: ignore[no-untyped-def]
    """The Boson's pitch and optics on a 64x48 crop; `focal_mm` widens it for the shape test."""
    raw = yaml.safe_load((REPO / "configs" / "sensors" / "flir_boson_640_lwir.yaml").read_text())
    d = copy.deepcopy(raw)
    d["sensor"]["fpa"].update(width=64, height=48)
    d["sensor"]["optics"]["supersample_factor"] = 1
    if focal_mm is not None:
        d["sensor"]["optics"]["focal_length_mm"] = focal_mm
    return SensorConfig.model_validate(d)


@pytest.fixture(scope="module")
def materials(tophat_lwir_lut) -> MaterialTable:  # type: ignore[no-untyped-def]
    del tophat_lwir_lut
    return MaterialTable.from_library(MaterialLibrary.load(), "lwir")


@pytest.fixture(scope="module")
def response():  # type: ignore[no-untyped-def]
    return load_spectral_response(RESPONSE)


def _build(  # type: ignore[no-untyped-def]
    lut,
    response,
    materials,
    *,
    cloud: float = 0.0,
    wind: float = 5.0,
    sst: float = SST_K,
    focal_mm: float | None = None,
    boresight_deg: float = BORESIGHT_DEG,
    targets=(),
):
    """One maritime frame's worth of models, scene and pipeline config."""
    sensor_cfg = _sensor(focal_mm)
    weather = WeatherSeries.constant(
        WeatherSample(T_AIR_K, 0.5, wind, cloud, 0.0, 0.0, 23000.0, 0.0), 3600.0
    )
    atmosphere = LayeredAtmosphere(
        load_atmosphere_preset("us_standard_clear"), weather, {"lwir": lut}
    )
    sky = SkyModel(atmosphere, load_environment_preset("clear_dry"), "lwir", lut)
    sea = SeaModel(
        sky,
        load_nk_table("water"),
        response,
        bulk_sst_k=sst,
        camera_height_m=CAMERA_HEIGHT_M,
    )
    scene = build_maritime_gbuffer(
        sensor_cfg.sensor,
        sky,
        sea,
        materials,
        boresight_elevation_deg=boresight_deg,
        targets=list(targets),
    )
    cfg = PipelineConfig.from_sensor(
        sensor_cfg,
        materials,
        lut=lut,
        noise_enabled=False,
        psf_enabled=False,
        atmosphere=atmosphere,
        sky=sky,
        sensor_seed=11,
    )
    return sky, sea, scene, cfg


def _render(scene, cfg):  # type: ignore[no-untyped-def]
    out = run_frame(dict(scene.planes), cfg, PipelineState(housing_temp_k=cfg.t_housing_cal_k))
    assert out.apparent_t is not None
    return out


def _column(scene, out):  # type: ignore[no-untyped-def]
    """(rendered apparent T, depression rad) down the water pixels of the centre column."""
    col = scene.shape[1] // 2
    rows = scene.sea_rows(col)
    return out.apparent_t[rows, col].astype(np.float64), scene.depression_rad[rows, col]


# -- the fixture itself ----------------------------------------------------------------------


def test_the_fixture_puts_sea_and_sky_both_under_the_background_mask(
    tophat_lwir_lut, response, materials
) -> None:  # type: ignore[no-untyped-def]
    """Both halves are computed backgrounds, so both carry distance 0 and the UNMAPPED id.

    The sea's apparent temperature already contains its own atmospheric path (`SeaModel.radiance`
    applies tau and the path radiance over the slant range). A non-zero distance here would send
    stage 2 over the same kilometres again -- on a 7 km horizon ray that is not a small error, and
    it would look like a plausibly hazier sea rather than like a bug.
    """
    _, sea, scene, _ = _build(tophat_lwir_lut, response, materials)
    g = GBuffer.from_dict(dict(scene.planes))
    assert g.sky_mask is not None
    planes = scene.planes
    assert np.all(planes["distance_m"] == 0.0)
    assert np.all(planes["material_id"] == UNMAPPED_MATERIAL_ID)
    assert scene.sea_mask.any() and (~scene.sea_mask).any(), "the frame must straddle the horizon"
    assert np.all(scene.sky_mask), "with no vessel every pixel is background"

    # the horizon is the spherical one, not zero: three Boson pixels' worth of difference
    horizon_deg = scene.metadata["horizon_depression_deg"]
    assert horizon_deg == pytest.approx(math.degrees(sea.horizon_rad), rel=1e-12)
    assert 0.14 < horizon_deg < 0.15
    assert np.all(scene.depression_rad[scene.sea_mask] >= sea.horizon_rad - 1e-12)


def test_a_sea_reflecting_a_different_sky_is_refused(tophat_lwir_lut, response, materials) -> None:  # type: ignore[no-untyped-def]
    """CLAUDE.md #6 at its most literal: one horizon cannot have two weathers across it."""
    sky_a, sea_a, _, _ = _build(tophat_lwir_lut, response, materials)
    sky_b, _, _, _ = _build(tophat_lwir_lut, response, materials, cloud=1.0)
    with pytest.raises(ValueError, match="same SkyModel"):
        build_maritime_gbuffer(_sensor().sensor, sky_b, sea_a, materials)
    assert sky_a is not sky_b


# -- the sea profile through the whole chain --------------------------------------------------


def test_the_rendered_sea_is_an_identity_on_the_sea_model(
    tophat_lwir_lut, response, materials
) -> None:  # type: ignore[no-untyped-def]
    """Rendered apparent temperature on water equals MM.3's own profile, to under a millikelvin.

    The maritime twin of the sky identity, and the test that would catch a double-counted
    atmospheric path: stage 2 charges the background nothing, so the only path in the answer is
    the one `SeaModel` applied itself. Compared against `apparent_temperature_exact_k`, the
    *un-tabulated* profile, so the frame is not being checked against the same interpolation it
    was built from.
    """
    _, sea, scene, cfg = _build(tophat_lwir_lut, response, materials)
    rendered, depression = _column(scene, _render(scene, cfg))
    exact = np.asarray(sea.apparent_temperature_exact_k(0.0, depression))
    err_mk = float(np.max(np.abs(rendered - exact))) * 1e3
    assert err_mk < 5.0, f"sea profile is off by {err_mk:.2f} mK -- is the path counted twice?"


def test_the_near_horizon_sea_reads_warmer_than_the_close_sea_not_colder(
    tophat_lwir_lut, response, materials
) -> None:  # type: ignore[no-untyped-def]
    """MM.8's first criterion, measured and **inverted**, with the mechanism separated out.

    The criterion said the horizon reads colder. Across this frame it reads 2.9 K **warmer**, and
    two separate things make it so. The frame sits on the horizon side of the profile's cold
    trough (the next test), where the surface is *already* warmer toward the horizon -- worth
    1.2 K here. The atmospheric path then roughly doubles that, because the near-horizon ray is
    7 km of air pulling toward T_air and the close ray is 530 m pulling almost not at all.

    The criterion is only true of the sea past the trough, where looking further down really does
    approach SST; it is not true of the near-horizon band a shore or mast camera actually works in.
    """
    _, sea, scene, cfg = _build(tophat_lwir_lut, response, materials)
    rendered, depression = _column(scene, _render(scene, cfg))

    far, near = rendered[0], rendered[-1]  # row 0 is nearest the horizon
    assert float(np.degrees(depression[0])) < 0.25
    assert float(np.degrees(depression[-1])) > 2.0
    assert far > near, "the far sea reads warmer, not colder"
    assert far - near == pytest.approx(2.89, abs=0.29), (far, near)  # the analytic amount, 10 %

    # Split the two contributions. The surface alone already leans this way; the path amplifies it.
    surface = np.asarray(
        tophat_lwir_lut.apparent_temperature(
            sea.surface_radiance(0.0, np.asarray([depression[0], depression[-1]])), "lb"
        ),
        dtype=np.float64,
    )
    surface_span = float(surface[0] - surface[1])
    assert surface_span > 0.0, "the surface already reads warmer toward the horizon here"
    assert 2.0 < (far - near) / surface_span < 3.0, (far - near, surface_span)

    far_range = float(slant_range_m(CAMERA_HEIGHT_M, float(depression[0])))
    near_range = float(slant_range_m(CAMERA_HEIGHT_M, float(depression[-1])))
    assert far_range > 10.0 * near_range
    assert far < T_AIR_K, "the path pulls toward air temperature, it does not overshoot it"


def test_the_profile_has_a_cold_trough_a_few_degrees_below_the_horizon(
    tophat_lwir_lut, response, materials
) -> None:  # type: ignore[no-untyped-def]
    """Seen in a frame: the minimum is interior, so neither end of the sea is its coldest part.

    Two effects fight across the profile (ADR 0078). Looking further down raises the incidence
    emissivity, pulling the sea toward its own temperature; looking further down also reflects sky
    from higher elevations, which is colder. The second wins for the first few degrees and the
    first wins after. A model that produced a monotone ramp -- the shape this milestone was
    planned around -- would pass every end-to-end check and be wrong in the middle of the frame.
    """
    _, _, scene, cfg = _build(
        tophat_lwir_lut, response, materials, focal_mm=1.0, boresight_deg=-10.0
    )
    rendered, depression = _column(scene, _render(scene, cfg))
    deg = np.degrees(depression)
    assert deg.min() < 1.5 and deg.max() > 20.0, (deg.min(), deg.max())

    i_min = int(np.argmin(rendered))
    assert 0 < i_min < len(rendered) - 1, "the coldest sea is neither end of the frame"
    assert 2.0 < float(deg[i_min]) < 15.0, float(deg[i_min])
    assert rendered[0] - rendered[i_min] > 1.0, "a real trough, not a wobble"
    assert rendered[-1] - rendered[i_min] > 1.0


def test_overcast_compresses_the_span_but_nowhere_near_to_nothing(
    tophat_lwir_lut, response, materials
) -> None:  # type: ignore[no-untyped-def]
    """MM.8's second criterion, measured at **57 %** of clear rather than under 20 %.

    An overcast sky is the cloud base's temperature at every elevation, so the reflected term
    stops varying with angle -- but the base is still colder than the water, so the surface span
    only closes by the fraction (T_sea - T_base)/(T_sea - T_sky_clear), and the *atmospheric path*
    that dominates this frame's span does not care about cloud at all. `test_sea_surface.py`
    already measured the surface-only reduction at 5-50 %; this is the same finding seen through
    a camera.
    """
    spans = []
    for cloud in (0.0, 1.0):
        _, _, scene, cfg = _build(tophat_lwir_lut, response, materials, cloud=cloud)
        rendered, _ = _column(scene, _render(scene, cfg))
        spans.append(float(rendered[0] - rendered[-1]))
    clear, overcast = spans
    assert overcast < clear, "cloud must flatten the sea, not steepen it"
    assert 0.4 < overcast / clear < 0.8, (clear, overcast, overcast / clear)


def test_roughening_the_sea_flattens_the_profile_in_the_frame(
    tophat_lwir_lut, response, materials
) -> None:  # type: ignore[no-untyped-def]
    """The wind reaches the picture through Cox-Munk, not through a roughness knob.

    A tilted facet presents a less grazing incidence than the flat surface would, so a rough sea
    has a higher effective emissivity near the horizon and looks more like its own temperature.
    The same `WeatherSeries` sets the skin temperature (MM.4), so this is also the check that
    turning the wind up does not move the two in contradictory directions.
    """
    spans = []
    for wind in (1.0, 15.0):
        _, _, scene, cfg = _build(tophat_lwir_lut, response, materials, wind=wind)
        rendered, _ = _column(scene, _render(scene, cfg))
        spans.append(float(rendered[0] - rendered[-1]))
    calm, rough = spans
    assert rough < calm, (calm, rough)


# -- the criterion that holds: polarity vs range ----------------------------------------------


def _vessel_contrast(  # type: ignore[no-untyped-def]
    tophat_lwir_lut, response, materials, depression_deg: float, vessel_t_k: float
) -> tuple[float, float]:
    """(contrast in K against the sea it sits on, its slant range)."""
    _, _, empty, cfg_empty = _build(tophat_lwir_lut, response, materials)
    bare = _render(empty, cfg_empty)
    col = empty.shape[1] // 2
    row = int(np.argmin(np.abs(np.degrees(empty.depression_rad[:, col]) - depression_deg)))
    range_m = float(slant_range_m(CAMERA_HEIGHT_M, math.radians(depression_deg)))
    target = SceneTarget("painted_composite", vessel_t_k, 12.0, range_m, (float(col), float(row)))
    _, _, scene, cfg = _build(tophat_lwir_lut, response, materials, targets=[target])
    assert scene.resolved, "the vessel should be resolved at both ranges tested"
    _, (i0, j0, i1, j1) = scene.resolved[0]
    out = _render(scene, cfg)
    vessel = float(np.mean(out.apparent_t[i0:i1, j0:j1]))
    background = float(np.mean(bare.apparent_t[i0:i1, j0:j1]))
    return vessel - background, range_m


def test_a_vessels_contrast_polarity_flips_with_range(tophat_lwir_lut, response, materials) -> None:  # type: ignore[no-untyped-def]
    """MM.8's third criterion, and the one that holds: bright near, dark far, same vessel.

    This is the maritime detection problem in one assertion. A vessel warmer than the near sea and
    cooler than the path-warmed far sea reads **bright against close water and dark against distant
    water**, so a detector trained on one range band sees the opposite sign in the other, and
    somewhere between them the contrast passes through zero and the target is invisible at any
    sensitivity. Nothing about the sensor changes across that span -- only the background.
    """
    near, near_range = _vessel_contrast(tophat_lwir_lut, response, materials, 2.0, VESSEL_T_K)
    far, far_range = _vessel_contrast(tophat_lwir_lut, response, materials, 0.18, VESSEL_T_K)

    assert far_range > 10.0 * near_range, (near_range, far_range)
    assert near > 0.0 > far, f"no polarity flip: {near:+.2f} K near, {far:+.2f} K far"
    # both sides comfortably above a 50 mK NETD, so the flip is a signature and not a rounding
    assert abs(near) > 0.5 and abs(far) > 0.5, (near, far)


def test_a_hot_vessel_stays_bright_at_every_range(tophat_lwir_lut, response, materials) -> None:  # type: ignore[no-untyped-def]
    """The control. The flip is a property of *this* vessel temperature, not of range itself.

    Without it the previous test would also pass on a pipeline that simply lost contrast with
    range and went negative from noise.
    """
    near, _ = _vessel_contrast(tophat_lwir_lut, response, materials, 2.0, 310.0)
    far, _ = _vessel_contrast(tophat_lwir_lut, response, materials, 0.18, 310.0)
    assert near > 0.0 and far > 0.0, (near, far)
    assert near > far, "contrast still falls with range, it just never changes sign"


def test_the_report_prints_how_much_of_the_frame_is_outside_the_validated_envelope(
    tophat_lwir_lut, response, materials
) -> None:  # type: ignore[no-untyped-def]
    """SE.1: **1.0000** of this frame's sea is past the angle published radiometry reaches.

    Every other test in this file measures what the sea model says. This one measures whether
    anyone has checked it at the angles being asked. The answer for the band a shore or mast
    camera works in is *none of them*: at a 20 m eye height the 50° validated limit is crossed
    at 31 m of slant range, and this frame starts at 530 m.

    So the findings above -- the 2.9 K inversion, the cold trough, the 57 % overcast collapse --
    are all statements about an extrapolation. They are self-consistent and the isothermal
    identity holds across them, which is exactly the reassurance SE.1's acceptance warns is not
    worth much here: the identity holds for a wrong angular emissivity too. ADR 0118.
    """
    _, sea, scene, cfg = _build(tophat_lwir_lut, response, materials)
    report = scene.envelope_report()
    print(report.summary())

    assert report.n_samples == int(scene.sea_mask.sum()) > 0
    assert report.fraction_beyond == 1.0, "no pixel of this frame is inside the envelope"
    assert report.zenith_min_deg > VALIDATED_ZENITH_DEG
    assert report.envelope_range_m == pytest.approx(31.1, rel=0.01)

    # the near end of the frame is still an order of magnitude beyond where the envelope ends
    _, depression = _column(scene, _render(scene, cfg))
    nearest = float(slant_range_m(CAMERA_HEIGHT_M, float(depression[-1])))
    assert nearest / report.envelope_range_m > 15.0, nearest

    # and the model still answers there rather than refusing -- the flag is a record, not a gate
    assert np.all(np.isfinite(sea.apparent_temperature_k(0.0, depression)))
