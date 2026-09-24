"""Roadmap `SE.2`: the maritime stage and the illumination bundle, in sim.

Two mechanisms reach a rendered maritime frame and neither could be checked without the engine,
because both are about what the *renderer* hands back rather than about arithmetic on a synthetic
G-buffer.

**The water is background, not geometry.** ADR 0078 renders the sea's radiometry analytically --
`SeaModel` gives the apparent temperature at each ray's own depression angle, from the angular
emissivity of water and the sky it reflects -- while the water *mesh* exists only so the visible
companion frame has a surface. The two are joined by `background_prim_paths`: the water prims'
pixels are added to `sky_mask`, which is what routes them past the material table and into the
background branch. Engine-free tests can drive that branch with a mask they wrote themselves;
what they cannot do is establish that the renderer's instance ids resolve to the prim paths the
stage authored, which is the step the wiring actually depends on. If they did not, the sea would
render as whatever material the resolver guessed for a water prim, which is a plausible picture
and wrong by tens of kelvin.

**The horizon is a claim with no free parameters.** With the camera at a known height and tilt and
a known focal length, the sea's edge lands at one row: the dip `acos(R_e / (R_e + h))` below the
horizontal, projected through the lens. At 20 m of eye height that dip is 0.144 degrees -- about
9 native pixels on this camera, small but far outside the tolerance here -- so the assertion also
distinguishes the curved sea from a flat one, which is the difference between a 16 km horizon and
an 8 km one (`_author_water`).

**Every Isaac render used to be emission only** (M10.22, ADR 0084). The illumination bundle is
what makes a reflective band see anything at all, and "the bundle reaches the render path" is
precisely an in-sim statement. Measured here on the same stage: with the bundle the SWIR frame
carries signal, without it the same geometry under the same sun is black.

docs/physics-model.md §5.3, §11.2; ADR 0078 (the sea model), ADR 0084 (the illumination bundle),
ADR 0014 addendum (the position frame the elevations come from).
"""

from __future__ import annotations

import copy
import math
import pathlib
from typing import Any

import numpy as np
import pytest
import yaml

REPO = pathlib.Path(__file__).resolve().parents[2]
BOSON_YAML = REPO / "configs" / "sensors" / "flir_boson_640_lwir.yaml"
SWIR_YAML = REPO / "configs" / "sensors" / "example_swir_ingaas_640.yaml"
SCENE_YAML = REPO / "configs" / "scenes" / "vessel_transit_clear_day.yaml"

WIDTH, HEIGHT = 320, 256
#: 8 mm, not the Boson's 14 mm, for the reason `test_aerial_demo_isaac` gives: the whole scene has
#: to fit in a 256-row frame, and here that means the horizon plus enough sea below it to carry a
#: depression gradient worth measuring.
FOCAL_MM = 8.0
SUPERSAMPLE = 2
#: Down. The horizon then sits in the upper third and most of the picture is sea.
TILT_DEG = -5.0
CAMERA_HEIGHT_M = 20.0

#: A coarser sea than the driver renders (420 x 448). The claims here are about where the mesh's
#: edge falls and which prims are masked, not about wave shape, and 160 x 160 puts the horizon on
#: the same ring while costing a quarter of the vertices.
WATER_RINGS, WATER_SECTORS = 160, 160

#: The horizon's row follows from the geometry alone. Three pixels covers the softening from the
#: PSF and the box filter and the half-ring the tessellation quantises the edge to -- not any
#: freedom in where the sea is.
HORIZON_TOLERANCE_PX = 3.0

#: The band LUT's own lower bound (`RADIOMETRIC_RANGE_K` is the ADC's; this is the table's).
#: An apparent temperature clamps here when the radiance under it is essentially zero, which is
#: what an undeclared sea comes back as.
LUT_FLOOR_K = 200.0

#: Half a kelvin between the rendered sea and `SeaModel`'s own curve -- ten times this camera's
#: NETD, and loose enough for the box filter and the PSF to average a band's edge without the
#: tolerance becoming the claim.
SEA_PROFILE_TOLERANCE_K = 0.5

#: Depression bands the sea profile is sampled in, degrees below the horizontal. The near edge is
#: not zero: the first band under the horizon is where the slant path is longest and the analytic
#: profile changes fastest, so a bin starting at the horizon itself averages over the steepest
#: part of the curve and reports a number that is about the binning.
PROFILE_BANDS_DEG: tuple[tuple[float, float], ...] = (
    (0.5, 1.5),
    (1.5, 4.0),
    (4.0, 10.0),
    (10.0, 25.0),
)


def _sensor(path: pathlib.Path) -> Any:
    """The shipped config at a small format, with its data paths resolved.

    `model_validate` is used rather than `load_sensor_config` because the format and focal length
    are overridden here, and it does **not** resolve `band.spectral_response` against the data
    root -- that is `load_sensor_config`'s job. Every other integration file gets away with the
    raw relative path because nothing it builds reads the response file; `SeaModel` does.
    """
    from irsim.config.loader import resolve_data_dir
    from irsim.config.sensor import SensorConfig

    raw = copy.deepcopy(yaml.safe_load(path.read_text()))
    raw["sensor"]["fpa"].update(width=WIDTH, height=HEIGHT)
    raw["sensor"]["optics"]["focal_length_mm"] = FOCAL_MM
    raw["sensor"]["optics"]["supersample_factor"] = SUPERSAMPLE
    response = pathlib.Path(raw["sensor"]["band"]["spectral_response"])
    if not response.is_absolute():
        raw["sensor"]["band"]["spectral_response"] = str(resolve_data_dir() / response)
    return SensorConfig.model_validate(raw)


def _scene(band_id: str, lut: Any, quantity: str = "lb") -> Any:
    """The maritime scene in the band's own radiometric form (ADR 0021).

    The quantity is not a detail to leave defaulted: a photon-counting FPA runs on `lb_q`, and a
    sky model built in `lb` reflecting into it is wrong by about 1e19. `Scene` refuses the
    mismatch rather than rendering it, which is how this test found out.

    No skylight is passed. The driver loads one for a reflective band, and should -- without it
    the sky is black and a sunlit target sits on nothing -- but here the *control* is a camera
    with no illumination bundle, and a bright sky would light up both frames and hide exactly the
    difference the control exists to show.
    """
    from irsim.scene import Scene

    return Scene.from_file(SCENE_YAML, {band_id: lut}, quantity=quantity)


def _sea_model(scene: Any, sensor: Any, quantity: Any) -> Any:
    """`SeaModel` on the scene's own sky and weather -- one weather object, as #6 requires."""
    from irsim.atmosphere.sea import SeaModel
    from irsim.materials.nk import load_nk_table
    from irsim.radiometry.spectral_response import load_spectral_response

    spec = sensor.sensor
    environment = scene.environment
    assert environment is not None and environment.ground.mode == "sea", (
        "the maritime scene must use a sea environment preset, or nothing below the horizon is "
        "water and this whole file is measuring a flat ground plane"
    )
    return SeaModel(
        scene.sky_models[spec.band.band_id],
        load_nk_table("water"),
        load_spectral_response(str(spec.band.spectral_response)),
        bulk_sst_k=float(environment.ground.bulk_sst_k or 0.0),
        camera_height_m=CAMERA_HEIGHT_M,
        quantity=quantity,
        latitude_deg=scene.spec.site.latitude_deg,
        longitude_deg=scene.spec.site.longitude_deg,
    )


def _open_camera(
    demo: Any,
    sensor: Any,
    scene: Any,
    lut: Any,
    *,
    background_paths: tuple[str, ...],
    sea: Any = None,
    illumination: Any = None,
) -> Any:
    """`IrCamera` on an already-authored maritime stage.

    The stage is a parameter rather than something this builds, because `build_maritime_demo`
    calls `new_stage()`: authoring a second one would pull the first camera's render product out
    from under it. Every camera in this module therefore looks at the same sea.
    """
    from irsim.materials.library import MaterialLibrary
    from irsim.materials.mapping import MaterialResolver, load_mapping_rules
    from irsim.materials.table import MaterialTable
    from irsim.pipeline.core import PipelineConfig
    from irsim_isaac.pipeline.ir_camera import IrCamera
    from irsim_isaac.pipeline.materials_usd import prim_records

    band_id = sensor.sensor.band.band_id
    table = MaterialTable.from_library(MaterialLibrary.load(), band_id)
    resolver = MaterialResolver(load_mapping_rules(), list(table.names))
    resolutions = resolver.resolve_all(prim_records(root="/World/Targets"))
    pipeline = PipelineConfig.from_sensor(
        sensor, table, lut, sky=scene.sky_models[band_id], atmosphere=scene.layered
    )
    return IrCamera(
        sensor,
        scene,
        pipeline=pipeline,
        prim_to_target=demo.prim_to_target,
        resolutions=resolutions,
        camera_path=demo.camera_path,
        illumination=illumination,
        strict_materials=False,
        strict_thermal_nodes=False,
        sea=sea,
        background_prim_paths=background_paths,
    ).open(settle_frames=8)


@pytest.fixture(scope="module")
def tophat_swir_lut(tmp_path_factory: pytest.TempPathFactory) -> Any:
    """BandLUT for an exact 1.0-1.7 um top-hat -- the InGaAs example sensor's span."""
    from irsim.radiometry.lut import BandLUT
    from irsim.radiometry.spectral_response import load_spectral_response

    p = tmp_path_factory.mktemp("swirlut") / "tophat_1_1p7.csv"
    p.write_text("# exact top-hat\n1.0,1.0\n1.7,1.0\n")
    return BandLUT.build(load_spectral_response(p))


@pytest.fixture(scope="module")
def maritime(simulation_app: Any, tophat_lwir_lut: Any, tophat_swir_lut: Any) -> Any:
    """One maritime stage, four cameras on it, every frame captured before anything is closed.

    Four, because three of the claims below need a control taken on the *same* geometry: the mask
    wiring is only a claim if the identical camera without `background_prim_paths` leaves those
    pixels alone, and the illumination bundle is only a claim if the identical camera without it
    renders dark.
    """
    del simulation_app
    from irsim_isaac.maritime_demo import build_maritime_demo
    from irsim_isaac.pipeline.illumination_isaac import SceneIllumination

    lwir_sensor = _sensor(BOSON_YAML)
    lwir_scene = _scene("lwir", tophat_lwir_lut)
    swir_sensor = _sensor(SWIR_YAML)
    swir_scene = _scene(
        swir_sensor.sensor.band.band_id, tophat_swir_lut, swir_sensor.sensor.quantity
    )

    demo = build_maritime_demo(
        camera_tilt_deg=TILT_DEG,
        camera_height_m=CAMERA_HEIGHT_M,
        water_rings=WATER_RINGS,
        water_sectors=WATER_SECTORS,
        wind_speed_m_s=float(lwir_scene.weather.at(lwir_scene.t0_s).wind_speed_m_s),
    )
    assert demo.errors == {}, demo.errors
    assert demo.water_paths, "the stage authored no water; there is nothing to mask"

    sea = _sea_model(lwir_scene, lwir_sensor, lwir_sensor.sensor.quantity)
    captured: dict[str, Any] = {"demo": demo, "sea": sea, "scene": lwir_scene}

    masked = _open_camera(
        demo, lwir_sensor, lwir_scene, tophat_lwir_lut, background_paths=demo.water_paths, sea=sea
    )
    captured["masked_out"] = masked.get_outputs()
    captured["masked"] = masked.last_frame
    captured["supersample"] = masked.config.supersample
    masked.close()

    unmasked = _open_camera(
        demo, lwir_sensor, lwir_scene, tophat_lwir_lut, background_paths=(), sea=sea
    )
    captured["unmasked_out"] = unmasked.get_outputs()
    captured["unmasked"] = unmasked.last_frame
    unmasked.close()

    lit = _open_camera(
        demo,
        swir_sensor,
        swir_scene,
        tophat_swir_lut,
        background_paths=demo.water_paths,
        illumination=SceneIllumination.for_camera(
            swir_sensor, swir_scene, swir_sensor.sensor.quantity
        ),
    )
    captured["swir_lit"] = lit.get_outputs()
    captured["swir_frame"] = lit.last_frame
    lit.close()

    dark = _open_camera(
        demo, swir_sensor, swir_scene, tophat_swir_lut, background_paths=demo.water_paths
    )
    captured["swir_dark"] = dark.get_outputs()
    dark.close()

    yield captured


def _native(mask: np.ndarray, k: int) -> np.ndarray:
    """A supersampled boolean plane reduced to the detector grid: a native pixel counts as the
    thing when **every** sub-sample is, which is the conservative reading for a coverage claim."""
    if k == 1:
        return mask
    h, w = mask.shape[0] // k, mask.shape[1] // k
    return mask[: h * k, : w * k].reshape(h, k, w, k).all(axis=(1, 3))


def _ids_for(frame: Any, paths: tuple[str, ...]) -> np.ndarray:
    """The pixels whose instance id resolves to one of ``paths``, on the supersampled grid."""
    from irsim_isaac.pipeline.material_ids import labels_to_paths

    resolved = labels_to_paths(frame.labels)
    wanted = [i for i, path in resolved.items() if path in paths]
    assert wanted, f"the renderer resolved no instance id to any of {paths}"
    ids = np.asarray(frame.instance_id)
    return np.isin(ids, np.asarray(wanted, dtype=ids.dtype))


# --- the water is background ------------------------------------------------------------------


def test_the_renderer_resolves_the_water_prims_the_stage_authored(maritime: Any) -> None:
    """`background_prim_paths` is matched against resolved prim paths, so they must resolve.

    This is the step no engine-free test reaches. `IrCamera` looks the frame's instance ids up in
    the renderer's own label payload and keeps the ones whose path is in the set the stage handed
    it; a label payload that names the prims differently -- or not at all -- silently selects
    nothing, and the sea then renders as whatever material the resolver guessed for it.
    """
    water = _ids_for(maritime["masked"], maritime["demo"].water_paths)
    assert water.any(), "no pixel resolved to a water prim"
    assert water.mean() > 0.25, (
        f"only {100 * water.mean():.1f} % of the frame is water; the camera is tilted "
        f"{TILT_DEG:+.0f} deg and most of the picture should be sea"
    )


def test_the_water_joins_the_sky_mask_and_the_vessels_do_not(maritime: Any) -> None:
    """Exactly the water prims are added to the background, with the control that makes it a claim.

    Two halves. The water must be masked -- otherwise `SeaModel` never supplies its radiance --
    and the vessels must not, because a masked hull would take the sea's apparent temperature and
    the target would vanish into the surface it floats on, which is the one failure a maritime
    camera exists to avoid.
    """
    frame = maritime["masked"]
    demo = maritime["demo"]
    water = _ids_for(frame, demo.water_paths)
    sky_mask = np.asarray(frame.planes["sky_mask"], dtype=bool)
    assert np.all(sky_mask[water]), "a water pixel was left out of the background"

    vessel_paths = tuple(demo.prim_to_target)
    vessels = _ids_for(frame, vessel_paths)
    assert vessels.any(), "no vessel was rendered, so the negative half proves nothing"
    assert not np.any(sky_mask[vessels]), "a vessel prim was swept into the background"


def test_without_background_prim_paths_the_sea_is_marked_not_measured(maritime: Any) -> None:
    """The control, and the reason the declaration is not redundant.

    Withhold `background_prim_paths` and the water prim is ordinary geometry with no thermal node
    and no material, because it is not a solver surface: it is background, and the declaration is
    how the camera is told so. Stage 1 then has nothing to build a radiance from.

    What made this worth a test rather than a comment is that **the mask still looks right**.
    `sky_mask` covers the water either way -- 99.9 % of the frame in both runs -- because the
    water prim resolves to no material and `debug_unmapped` (on by default) sweeps unmapped
    geometry into the background so it can be painted magenta. Until `IG.17` the radiometric
    branch got no such marking, and the apparent temperature came back as **200.1 K**: the band
    LUT's own floor, where an apparent temperature lands when the radiance under it is nearly
    zero. That is a number, in kelvin, in range, and it reads as cold water rather than as an
    error, so an inspection that checked the mask reported the sea correctly handled.

    It is now NaN, which is the point: a frame containing unmapped geometry is not a measurement
    of that geometry, and nothing should be able to average it into one by accident. The declared
    run beside it is finite and warm, which is what says the marking is about the declaration and
    not about the water.
    """
    frame = maritime["unmasked"]
    out = maritime["unmasked_out"]
    assert out.apparent_t is not None
    k = maritime["supersample"]
    t = np.asarray(out.apparent_t, dtype=np.float64)
    water = _native(_ids_for(frame, maritime["demo"].water_paths), k)
    assert water.any()
    assert np.isnan(t[water]).all(), (
        f"the undeclared sea reads {np.nanmean(t[water]):.2f} K on "
        f"{int(np.isfinite(t[water]).sum())} pixel(s); `IG.17` marks unmapped geometry NaN on "
        "the planes that claim physical units, so a finite value here means the marking was lost"
    )
    declared_plane = np.asarray(maritime["masked_out"].apparent_t, dtype=np.float64)
    declared_water = _native(_ids_for(maritime["masked"], maritime["demo"].water_paths), k)
    declared = float(declared_plane[declared_water].mean())
    assert np.isfinite(declared_plane[declared_water]).all()
    assert declared > LUT_FLOOR_K + 50.0, (
        f"the declared sea reads {declared:.2f} K: the declaration is supposed to be the "
        "difference between a sea and a hole in the picture"
    )


# --- the horizon ------------------------------------------------------------------------------


def horizon_row() -> float:
    """Where the sea's edge lands, in native pixels, from the geometry and nothing else.

    `MaritimeDemoScene.horizon_elevation_deg` is the dip below the horizontal minus the camera's
    tilt, so it is already relative to boresight; the row is `cy - f_px tan(elevation)` because
    elevation rises up the frame and rows count down it.
    """
    from irsim_isaac.maritime_demo import EARTH_RADIUS_M

    dip = -math.degrees(math.acos(EARTH_RADIUS_M / (EARTH_RADIUS_M + CAMERA_HEIGHT_M)))
    elevation = dip - TILT_DEG
    return HEIGHT / 2.0 - (FOCAL_MM / 0.012) * math.tan(math.radians(elevation))


def test_the_sea_ends_where_the_curved_earth_puts_it(maritime: Any) -> None:
    """The topmost water row is the horizon, and the horizon is the **dip**, not the tilt.

    Two claims, and the second is the one worth having. The first is that the sea's edge lands
    within a few pixels of `cy - f_px tan(elevation)`, with the elevation coming from the camera's
    height and tilt and nothing else. The second is that the measurement sits closer to the curved
    prediction than to the flat one -- a comparison rather than a tolerance, because at 20 m of
    eye height the dip is only 0.144 degrees and the two predictions are 1.7 native pixels apart
    on this camera. A tolerance loose enough to be robust would not separate them; asking which is
    nearer does, and that is the claim: the sea in this frame is curved, with its horizon at 16 km
    where a flat one would put it at 8 km.
    """
    frame = maritime["masked"]
    k = maritime["supersample"]
    water = _native(_ids_for(frame, maritime["demo"].water_paths), k)
    rows = np.flatnonzero(water.any(axis=1))
    assert rows.size, "no native row is wholly water"
    measured = float(rows[0])
    curved = horizon_row()
    flat = HEIGHT / 2.0 + (FOCAL_MM / 0.012) * math.tan(math.radians(TILT_DEG))
    assert 0.0 < curved < HEIGHT, "the stage should put the horizon inside the frame"
    assert abs(measured - curved) < HORIZON_TOLERANCE_PX, (
        f"the sea begins at row {measured:.0f}, the curved geometry says {curved:.1f}"
    )
    assert abs(measured - curved) * 2.0 < abs(measured - flat), (
        f"row {measured:.0f} is {abs(measured - curved):.2f} px from the curved horizon "
        f"({curved:.2f}) and {abs(measured - flat):.2f} px from the flat one ({flat:.2f}); "
        "the frame does not distinguish them, so this claim is not supported"
    )


# --- the sea profile --------------------------------------------------------------------------


def test_the_rendered_sea_is_the_analytic_sea(maritime: Any) -> None:
    """Apparent temperature over water falls monotonically with depression, onto ADR 0078's curve.

    The direction is the first thing to get right and is not the one a reader guesses. Angular
    emissivity alone would warm the sea as the ray steepens, and it does; but the **slant path**
    runs the other way and wins. A ray grazing towards the horizon crosses tens of kilometres of
    air at 298.6 K and arrives carrying mostly path radiance, so the near-horizon sea reads *above*
    the 290.0 K bulk SST; a ray steeply down crosses a few hundred metres and reads the water. The
    profile therefore falls from about 293 K at half a degree to about 290 K at twenty, and a model
    that had only the emissivity term would have it backwards.

    Each band is also held against `SeaModel.apparent_temperature_k` evaluated at the band's own
    mid-depression, to half a kelvin -- ten times the camera's NETD. That is the end-to-end claim:
    not that the rendered sea is smooth, but that it is the analytic profile the code computes,
    having travelled through the instance ids, the background mask, the bridge, the atmosphere and
    the radiometric inverse without being replaced by something plausible on the way.
    """
    frame = maritime["masked"]
    out = maritime["masked_out"]
    assert out.apparent_t is not None, "the LWIR camera should emit an apparent-temperature plane"
    k = maritime["supersample"]
    t = np.asarray(out.apparent_t, dtype=np.float64)
    water = _native(_ids_for(frame, maritime["demo"].water_paths), k)
    depression = -np.degrees(np.asarray(frame.elevation_rad, dtype=np.float64))
    if k > 1:
        depression = depression[::k, ::k][: t.shape[0], : t.shape[1]]

    sea, scene = maritime["sea"], maritime["scene"]
    means, model = [], []
    for lo, hi in PROFILE_BANDS_DEG:
        band = water & (depression >= lo) & (depression < hi)
        assert band.sum() > 200, f"only {int(band.sum())} water pixels between {lo} and {hi} deg"
        means.append(float(t[band].mean()))
        mid = math.radians(0.5 * (lo + hi))
        model.append(float(np.atleast_1d(sea.apparent_temperature_k(scene.t0_s, mid))[0]))

    assert all(b < a for a, b in zip(means, means[1:], strict=False)), (
        f"the sea profile is not monotone in depression: {[f'{m:.2f}' for m in means]}"
    )
    for (lo, hi), rendered, analytic in zip(PROFILE_BANDS_DEG, means, model, strict=True):
        assert abs(rendered - analytic) < SEA_PROFILE_TOLERANCE_K, (
            f"{lo}-{hi} deg: rendered {rendered:.3f} K, SeaModel says {analytic:.3f} K"
        )
    assert means[0] - means[-1] > 2.0, (
        f"the whole profile spans {means[0] - means[-1]:.3f} K; the angular model is not being "
        "exercised by a frame that flat"
    )
    bulk = float(scene.environment.ground.bulk_sst_k)
    assert means[0] > bulk, "the near-horizon sea should read above the bulk SST, on path radiance"
    assert means[-1] == pytest.approx(bulk, abs=1.5), (
        f"steeply down the sea should approach its own {bulk:.1f} K, not {means[-1]:.2f} K"
    )


# --- the illumination bundle ------------------------------------------------------------------


def test_a_reflective_band_is_black_without_the_illumination_bundle(maritime: Any) -> None:
    """The M10.22 claim, on a rendered frame: without the bundle every Isaac render was emission
    only, which is right to a fraction of a percent in LWIR and *black* in SWIR.

    The same stage, the same sun, the same geometry; the only difference is whether
    `SceneIllumination.for_camera` was passed to the camera. Asserted as a ratio rather than an
    absolute level because the absolute level is a property of the scene's sun angle, while the
    ratio is a property of the wiring.
    """
    lit = np.asarray(maritime["swir_lit"].signal_dn, dtype=np.float64)
    dark = np.asarray(maritime["swir_dark"].signal_dn, dtype=np.float64)
    assert lit.shape == dark.shape
    spread_lit = float(lit.max() - lit.min())
    spread_dark = float(dark.max() - dark.min())
    assert spread_lit > 1.0, "the illuminated SWIR frame carries no structure at all"
    assert spread_lit > 10.0 * max(spread_dark, 1e-9), (
        f"SWIR spans {spread_lit:.3g} DN with the bundle and {spread_dark:.3g} without it; "
        "the bundle is not reaching the render path"
    )
