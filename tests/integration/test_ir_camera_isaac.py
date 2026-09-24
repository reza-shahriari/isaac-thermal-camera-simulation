"""Roadmap M10.9a-ii: the camera closed against the renderer -- the lens, then a whole frame.

Two things need the engine and nothing else does.

**The lens.** ``optics.distortion`` becomes a USD applied API schema, and whether the renderer
honours it is not decidable from the schema registry. A grid of small quads at known world
positions is rendered twice, once with the Boson's zero coefficients and once with a visible
barrel term, and each quad's measured centroid is compared with
``irsim.optics.projection.project_usd``. That is the row's criterion: < 0.2 px.

Measuring to a fifth of a pixel off a **binary** instance-id mask takes care, and the first
attempt got it wrong. A rectangle's included-pixel set snaps to whole pixels, so its centroid
lands on a multiple of half a pixel no matter how large the rectangle is -- averaging over more
pixels does not help. These quads sit at thirds of a pixel, and every one of them came back a
flat 1/3 px off: a rasterisation artefact indistinguishable, by eye, from a lens error. The grid
is therefore rendered 8x supersampled and the centroids divided back down, which puts the
quantisation at 1/16 of a native pixel. The quads are kept small for a different reason: so that
the centroid of a distorted quad and the distortion of its centroid agree to ~0.007 px.

**A frame.** The five-prim ``material_probe`` stage carries one prim of each mapping-precedence
class including a deliberate miss, so one render exercises the id → material → emissivity chain,
the id → temperature table, the sky background and the UNMAPPED overlay at once.
"""

from __future__ import annotations

import copy
import pathlib
from typing import Any

import numpy as np
import pytest
import yaml

REPO = pathlib.Path(__file__).resolve().parents[2]
BOSON_YAML = REPO / "configs" / "sensors" / "flir_boson_640_lwir.yaml"
SCENE_YAML = REPO / "configs" / "scenes" / "sky_target_clear_day.yaml"

GRID_RES = (320, 256)
GRID_FOCAL_MM = 8.0
#: The grid is rendered supersampled and its centroids converted back to native pixels.
#: A binary instance-id mask carries no sub-pixel information: the included-pixel set of a
#: rectangle snaps to whole pixels, so its centroid lands on a multiple of half a pixel whatever
#: the rectangle's size. Measured at 1x that quantises the centroid to 0.5 px and the quads here,
#: whose true centres fall on thirds of a pixel, come out a flat 1/3 px off -- a rasterisation
#: artefact that looks exactly like a lens error. At 8x it is 1/16 of a native pixel, comfortably
#: inside the budget, and supersampling is what the renderer is asked for anyway (SS8.3).
GRID_SUPERSAMPLE = 8
GRID_Z_M = -5.0
#: Kept inside the 1.2 m half-width the 8 mm lens sees at 5 m, with margin: a quad clipped by the
#: frame edge has a centroid that is not its image, which reads as a lens error and is not one.
GRID_OFFSETS = (-0.8, -0.4, 0.0, 0.4, 0.8)
GRID_HALF_M = 0.06

#: The M10.9a budget: a rendered grid must reproject through the config model to better than this.
REPROJECTION_BUDGET_PX = 0.2

#: A visible barrel lens: 2.1 px inward at the outermost quad of this grid camera -- ten times the
#: reprojection budget, so a renderer that ignored the schema could not pass by accident.
BARREL_COEFFS = [-0.28, 0.09, 0.0, 0.0, 0.0]

#: How far the barrel lens must move the outer quads for the schema to have been honoured at all.
MIN_BARREL_SHIFT_PX = 1.5

TARGET_K = {
    "Body": 317.25,
    "Window": 268.40,
    "Road": 301.00,
    "Trim": 295.55,
    "Mystery": 330.10,
}


def make_sensor(
    width: int,
    height: int,
    focal_mm: float,
    coeffs: list[float] | None = None,
    supersample: int = 1,
) -> Any:
    """The Boson config resized, with a given lens: f/1 VOx optics on an arbitrary format."""
    from irsim.config.sensor import SensorConfig

    raw = yaml.safe_load(BOSON_YAML.read_text())
    d = copy.deepcopy(raw)
    d["sensor"]["fpa"].update(width=width, height=height)
    d["sensor"]["optics"]["focal_length_mm"] = focal_mm
    d["sensor"]["optics"]["supersample_factor"] = supersample
    d["sensor"]["optics"]["distortion"] = {
        "model": "brown_conrady",
        "coeffs": coeffs if coeffs is not None else [0.0] * 5,
    }
    return SensorConfig.model_validate(d)


def grid_sensor(coeffs: list[float]) -> Any:
    """A 320×256 camera on an 8 mm lens (13.5° half-field), 8x supersampled, with these coeffs."""
    return make_sensor(GRID_RES[0], GRID_RES[1], GRID_FOCAL_MM, coeffs, GRID_SUPERSAMPLE)


def build_grid_stage(camera_path: str) -> dict[str, tuple[float, float, float]]:
    """A grid of small quads at known camera-space positions; the camera sits at the origin."""
    import omni.usd
    from pxr import Gf, UsdGeom, UsdLux

    ctx = omni.usd.get_context()
    ctx.new_stage()
    stage = ctx.get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    stage.DefinePrim("/World", "Xform")
    stage.DefinePrim("/World/Grid", "Xform")
    UsdLux.DistantLight.Define(stage, "/World/Light").CreateIntensityAttr(1000.0)

    centres: dict[str, tuple[float, float, float]] = {}
    for i, x in enumerate(GRID_OFFSETS):
        for j, y in enumerate(GRID_OFFSETS):
            path = f"/World/Grid/Q_{i}_{j}"
            mesh = UsdGeom.Mesh.Define(stage, path)
            h = GRID_HALF_M
            mesh.GetPointsAttr().Set(
                [
                    Gf.Vec3f(x - h, y - h, GRID_Z_M),
                    Gf.Vec3f(x + h, y - h, GRID_Z_M),
                    Gf.Vec3f(x + h, y + h, GRID_Z_M),
                    Gf.Vec3f(x - h, y + h, GRID_Z_M),
                ]
            )
            mesh.GetFaceVertexCountsAttr().Set([4])
            mesh.GetFaceVertexIndicesAttr().Set([0, 1, 2, 3])
            mesh.GetNormalsAttr().Set([Gf.Vec3f(0, 0, 1)] * 4)
            mesh.GetDoubleSidedAttr().Set(True)
            mesh.GetSubdivisionSchemeAttr().Set("none")
            centres[path] = (x, y, GRID_Z_M)
    UsdGeom.Camera.Define(stage, camera_path)
    return centres


def centroids(ids_plane: Any, labels: dict[int, str]) -> dict[str, tuple[float, float]]:
    """Centroid (u, v) in pixel-edge units of every labelled prim present in the id image."""
    ids = np.asarray(ids_plane)
    rows, cols = np.mgrid[0 : ids.shape[0], 0 : ids.shape[1]]
    out: dict[str, tuple[float, float]] = {}
    for ident, path in labels.items():
        mask = ids == ident
        n = int(mask.sum())
        if n < 64:  # too few pixels for a centroid to mean anything
            continue
        out[path] = (float(cols[mask].mean()) + 0.5, float(rows[mask].mean()) + 0.5)
    return out


def render_grid(coeffs: list[float]) -> dict[str, Any]:
    """Author the camera from a config with ``coeffs``, render the grid, report what was seen."""
    import omni.replicator.core as rep
    import omni.usd

    from irsim_isaac.geometry_probe import configure_renderer
    from irsim_isaac.pipeline.gbuffer_isaac import AovReader
    from irsim_isaac.pipeline.ir_camera import author_camera, camera_optics
    from irsim_isaac.pipeline.material_ids import labels_from_payload

    camera_path = "/World/IrCamera"
    centres = build_grid_stage(camera_path)
    stage = omni.usd.get_context().get_stage()
    sensor = grid_sensor(coeffs)
    optics = camera_optics(sensor.sensor)
    authored = author_camera(stage, camera_path, optics, sensor.sensor.optics.distortion)

    configure_renderer()
    rp = rep.create.render_product(camera_path, optics.resolution)
    path = getattr(rp, "path", None) or str(rp)
    reader = AovReader(
        path,
        device="cpu",
        required=("instance",),
        expected_shape=(optics.resolution[1], optics.resolution[0]),
    ).attach(settle_frames=24)
    aovs = reader.read()
    labels = labels_from_payload(aovs.device_handles["instance"])
    result = {
        "authored": authored,
        "optics": optics,
        "sensor": sensor,
        "centres": centres,
        "measured": {
            path: (u / GRID_SUPERSAMPLE, v / GRID_SUPERSAMPLE)
            for path, (u, v) in centroids(aovs.instance_id, labels).items()
        },
    }
    reader.detach()
    return result


def reprojection_errors(rendered: dict[str, Any], spec: Any = None) -> dict[str, float]:
    """|measured − predicted| in pixels, per quad, through ``spec`` (default: the rendered lens)."""
    from irsim.optics.projection import Intrinsics, project_usd

    sensor = rendered["sensor"].sensor
    intrinsics = Intrinsics.from_sensor(sensor, 1)  # centroids were converted back to native
    spec = sensor.optics.distortion if spec is None else spec
    errors: dict[str, float] = {}
    for path, (u_meas, v_meas) in rendered["measured"].items():
        centre = rendered["centres"].get(path)
        if centre is None:
            continue
        u, v = project_usd(np.array([centre]), intrinsics, spec)
        errors[path] = float(np.hypot(u[0] - u_meas, v[0] - v_meas))
    return errors


# --- the lens -------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def pinhole_grid(simulation_app: Any) -> dict[str, Any]:
    del simulation_app
    return render_grid([0.0] * 5)


@pytest.fixture(scope="module")
def barrel_grid(pinhole_grid: dict[str, Any]) -> dict[str, Any]:
    del pinhole_grid  # ordering only: both rebuild the stage
    return render_grid(BARREL_COEFFS)


def test_the_camera_prim_carries_the_configured_optics(pinhole_grid: dict[str, Any]) -> None:
    """focalLength, aperture and the whole distortion schema, as the sensor config says."""
    authored = pinhole_grid["authored"]
    optics = pinhole_grid["optics"]
    assert authored["focalLength"] == pytest.approx(GRID_FOCAL_MM)
    assert authored["horizontalAperture"] == pytest.approx(GRID_RES[0] * 0.012)
    assert authored["schema"] == "OmniLensDistortionOpenCvPinholeAPI"
    assert str(authored["model"]) == "opencvPinhole"
    prefix = "omni:lensdistortion:opencvPinhole:"
    # The schema describes the grid actually rendered, which is the supersampled one: its fx and
    # imageSize scale with k while the aperture, a physical size, does not.
    assert authored[prefix + "imageSize"] == optics.resolution
    assert authored[prefix + "fx"] == pytest.approx(GRID_FOCAL_MM / 0.012 * GRID_SUPERSAMPLE)


def test_every_quad_is_resolved_and_visible(pinhole_grid: dict[str, Any]) -> None:
    """All 25 quads must reach the id image, or the reprojection is measuring a subset."""
    assert len(pinhole_grid["measured"]) == len(GRID_OFFSETS) ** 2


def test_undistorted_grid_reprojects_within_the_budget(pinhole_grid: dict[str, Any]) -> None:
    """With coeffs = 0 the render is a pinhole and the oracle must hit every quad to < 0.2 px."""
    errors = reprojection_errors(pinhole_grid)
    assert errors, "no quad was matched to a predicted position"
    assert max(errors.values()) < REPROJECTION_BUDGET_PX, errors


def test_barrel_coefficients_actually_move_the_image(
    pinhole_grid: dict[str, Any], barrel_grid: dict[str, Any]
) -> None:
    """The renderer must honour the schema: the field edge moves inward by ~13 px, not 0.

    Without this the reprojection test above would pass on a build that ignored the distortion
    attributes entirely -- a zero-coefficient lens and a discarded lens look identical.
    """
    shifts = {
        path: float(np.hypot(*(np.subtract(barrel_grid["measured"][path], centre_uv))))
        for path, centre_uv in pinhole_grid["measured"].items()
        if path in barrel_grid["measured"]
    }
    assert shifts
    assert max(shifts.values()) > MIN_BARREL_SHIFT_PX, shifts
    centre_path = f"/World/Grid/Q_{len(GRID_OFFSETS) // 2}_{len(GRID_OFFSETS) // 2}"
    assert shifts[centre_path] < 0.5, "the optical axis must not move"


def test_the_pinhole_model_cannot_explain_the_barrel_render(barrel_grid: dict[str, Any]) -> None:
    """Reprojecting the distorted render through a zero-coefficient lens must fail, and badly.

    Together with the test below this is what makes the reprojection claim mean something: the
    distortion is large enough to break the wrong model, and the configured model absorbs it to
    inside the budget. Either half alone could pass on a renderer that discarded the schema.
    """
    from irsim.config.sensor import DistortionSpec

    errors = reprojection_errors(
        barrel_grid, DistortionSpec(model="brown_conrady", coeffs=[0.0] * 5)
    )
    assert max(errors.values()) > MIN_BARREL_SHIFT_PX, errors


def test_distorted_grid_reprojects_within_the_budget(barrel_grid: dict[str, Any]) -> None:
    """The row's criterion: coeffs [−0.28, 0.09, 0, 0, 0] reproject through the config
    model to better than 0.2 px."""
    errors = reprojection_errors(barrel_grid)
    assert errors, "no quad was matched to a predicted position"
    assert max(errors.values()) < REPROJECTION_BUDGET_PX, errors


# --- a whole frame --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def camera(simulation_app: Any, tophat_lwir_lut: Any) -> Any:
    """An ``IrCamera`` on the five-prim material stage, with one thermal node per quad."""
    del simulation_app
    from irsim.config.scene import SceneConfig, TargetSpec, load_scene_config
    from irsim.materials.library import MaterialLibrary
    from irsim.materials.mapping import MaterialResolver, load_mapping_rules
    from irsim.materials.table import MaterialTable
    from irsim.pipeline.core import PipelineConfig
    from irsim.scene import Scene
    from irsim_isaac.material_probe import build_material_scene
    from irsim_isaac.pipeline.ir_camera import IrCamera
    from irsim_isaac.pipeline.materials_usd import prim_records

    built = build_material_scene(resolution=256)
    assert built.errors == {}, built.errors
    # The camera is re-authored from the sensor config, so the sensor must be built to see what
    # the probe scene was laid out for: same resolution, same focal length in pixels. Derived
    # from the scene rather than written down, so the two cannot drift apart.
    focal_mm = built.focal_px * 0.012

    config = load_scene_config(SCENE_YAML)
    targets = [
        TargetSpec(name=n.lower(), solver="prescribed", schedule_s=[0.0, 7200.0], schedule_k=[t, t])
        for n, t in TARGET_K.items()
    ]
    spec = config.scene.model_copy(update={"targets": targets})
    scene = Scene.from_config(
        SceneConfig(schema_version=config.schema_version, scene=spec), {"lwir": tophat_lwir_lut}
    )

    library = MaterialLibrary.load()
    table = MaterialTable.from_library(library, "lwir")
    resolver = MaterialResolver(load_mapping_rules(), list(table.names))
    resolutions = resolver.resolve_all(prim_records(root="/World/Targets"))

    sensor = make_sensor(built.resolution, built.resolution, focal_mm)
    pipeline = PipelineConfig.from_sensor(
        sensor, table, tophat_lwir_lut, sky=scene.sky_models["lwir"], atmosphere=scene.layered
    )
    cam = IrCamera(
        sensor,
        scene,
        pipeline=pipeline,
        prim_to_target={t.prim_path: n.lower() for n, t in built.targets.items()},
        resolutions=resolutions,
        camera_path=built.camera_path,
        strict_materials=False,
        strict_thermal_nodes=False,
    )
    yield cam.open(settle_frames=16)
    cam.close()


def test_outputs_have_the_declared_dtypes_and_the_native_shape(camera: Any) -> None:
    """§12.2: float32 radiance, float32 apparent T, uint16 DN, RGBA8 -- at the detector grid."""
    out = camera.get_outputs()
    h, w = camera.sensor.sensor.fpa_shape
    assert out.radiance is not None and out.radiance.dtype == np.float32
    assert out.apparent_t is not None and out.apparent_t.dtype == np.float32
    assert out.dn16 is not None and out.dn16.dtype == np.uint16
    assert out.display8 is not None and out.display8.dtype == np.uint8
    assert out.radiance.shape == (h, w)
    assert out.apparent_t.shape == (h, w)
    assert out.display8.shape == (h, w, 4)
    # `IG.17`: NaN is the marking for a pixel whose material never resolved, and nowhere else.
    # A frame that is finite everywhere is what this stage should produce -- every prim on it is
    # mapped or deliberately not -- so the assertion is that the two masks agree exactly, rather
    # than that the plane is finite. The weaker form would pass a plane of NaN.
    from irsim_isaac.pipeline.material_ids import fold_mask_to_native

    frame = camera.last_frame
    assert frame is not None
    unmapped = fold_mask_to_native(frame.unmapped, camera.config.supersample)
    assert np.array_equal(np.isnan(out.apparent_t), unmapped)
    assert np.array_equal(np.isnan(out.radiance), unmapped)
    assert np.all(np.isfinite(out.apparent_t[~unmapped]))


def test_no_plane_is_float16_anywhere(camera: Any) -> None:
    """CLAUDE.md #2 at the engine boundary: 0.25 K spacing against a 50 mK NETD."""
    planes = camera.planes()
    for name, plane in planes.items():
        assert np.asarray(plane).dtype != np.float16, name


def prim_apparent_t(camera: Any, out: Any, name: str) -> float | None:
    """Median apparent temperature over the pixels of one named prim, or None if not visible."""
    frame = camera.last_frame
    assert frame is not None and out.apparent_t is not None
    ident = next((i for i, path in frame.labels.items() if path.endswith(f"/{name}")), None)
    if ident is None:
        return None
    mask = frame.instance_id == ident
    if int(mask.sum()) < 16:
        return None
    return float(np.median(out.apparent_t[mask]))


def test_high_emissivity_prims_read_back_their_authored_temperature(camera: Any) -> None:
    """A near-black surface radiates itself, so T_app is its own temperature within a few kelvin.

    ``Body`` maps to ``car_paint_black`` and ``Road`` to ``asphalt_dry`` (eps 0.94), so the small
    remaining gap is the reflected sky through (1 - eps) and the short atmospheric path. What is
    under test is that the id -> temperature table put *each authored value on its own prim*: a
    swapped or averaged table misses by tens of kelvin, since the five prims span 268-330 K.
    """
    out = camera.get_outputs()
    for name in ("Body", "Road"):
        seen = prim_apparent_t(camera, out, name)
        assert seen is not None, f"{name} is not visible"
        assert abs(seen - TARGET_K[name]) < 8.0, f"{name}: saw {seen:.2f} K, wrote {TARGET_K[name]}"


def test_the_aluminium_prims_show_their_environment_instead_of_themselves(camera: Any) -> None:
    """eps = 0.09 is a mirror: ``Trim`` and ``Window`` read nearly alike despite 27 K between them.

    This is the phenomenology that makes an IR image unlike a visible one, and the sharpest check
    available that emissivity actually reaches the radiance kernel. The two prims are the same
    material at 295.55 K and 268.40 K; a chain that ignored the material table would report that
    27.15 K difference nearly intact. Through eps = 0.09 it collapses to about 0.09 of itself,
    because both surfaces are showing the same sky and ground.

    Note which way it goes: a mirror does not read *cold*, it reads *its environment*. The 268 K
    window is colder than what it reflects, so it reads warmer than it is -- the opposite of the
    cold-roof case, and from the same physics.
    """
    out = camera.get_outputs()
    seen = {n: prim_apparent_t(camera, out, n) for n in ("Trim", "Window", "Body", "Road")}
    assert all(v is not None for v in seen.values()), seen

    authored_spread = abs(TARGET_K["Trim"] - TARGET_K["Window"])
    mirror_spread = abs(seen["Trim"] - seen["Window"])  # type: ignore[operator]
    assert mirror_spread < 0.25 * authored_spread, seen

    # The control: at eps ~ 0.94 the authored spread survives, so the collapse above is the
    # emissivity and not something flattening every difference in the chain.
    control_spread = abs(seen["Body"] - seen["Road"])  # type: ignore[operator]
    assert control_spread > 0.5 * abs(TARGET_K["Body"] - TARGET_K["Road"]), seen


def test_the_unmapped_prim_is_painted_magenta_in_the_display_branch_only(camera: Any) -> None:
    """ADR 0047: a prim nobody mapped is loud in the picture and untouched in the radiometry."""
    from irsim_isaac.pipeline.material_ids import MAGENTA_RGBA

    out = camera.get_outputs()
    frame = camera.last_frame
    assert frame is not None and frame.unmapped.any(), "the Mystery prim should be UNMAPPED"
    assert out.display8 is not None
    painted = np.all(out.display8 == np.asarray(MAGENTA_RGBA, dtype=np.uint8), axis=-1)
    assert painted.any()
    assert out.dn16 is not None and np.all(out.dn16[painted] < np.iinfo(np.uint16).max)


def test_the_clock_advances_so_a_sequence_is_a_sequence(camera: Any) -> None:
    """Each capture moves render time by one frame period; the FFC and the drift depend on it."""
    t0 = camera.t_rel_s
    camera.get_outputs()
    camera.get_outputs()
    assert camera.t_rel_s == pytest.approx(t0 + 2.0 / 60.0, abs=1e-9)
