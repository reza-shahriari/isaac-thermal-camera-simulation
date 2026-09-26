"""Does a cloud volume load on this build, and does it reach any AOV? (AT.13, ADR 0140)

    ./python.sh scripts/probe_cloud_volume.py
    ./python.sh scripts/probe_cloud_volume.py --nanovdb      # the old writer, for comparison
    ./python.sh scripts/probe_cloud_volume.py --voxel-m 25   # a finer grid

Two questions, and they are separate. **Does it load at all** -- ADR 0127 wrote a NanoVDB that
Warp read back correctly and the renderer's IndeX plugin refused, so `--cloud-volume` has been off
by default ever since. **Does it reach an AOV** -- the infrared band is computed from AOVs, and a
volume is not a surface, so a ray through cloud may report no hit at all and an aircraft behind a
cloud would then be unoccluded in LWIR while being hidden in the visible.

Both are answered the same way: **render the scene twice, once with the volume and once without,
and difference the frames.** A volume that did not load and a volume that is invisible to an
annotator produce exactly the same array as no volume at all, and nothing in a single frame
distinguishes either from a cloud that is simply thin. A difference is not open to that reading.

The stage is the smallest thing that can answer it: a bright backdrop, a cloud in front of it, a
camera looking through. Nothing else, so anything that changes is the cloud.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))

parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
)
parser.add_argument("--out", default="captures/cloud_volume_probe")
parser.add_argument("--voxel-m", type=float, default=50.0)
parser.add_argument("--resolution", type=int, nargs=2, default=(640, 480), metavar=("W", "H"))
parser.add_argument("--spp", type=int, default=64, help="path-traced samples per pixel")
parser.add_argument("--settle", type=int, default=12, help="orchestrator steps before a read")
parser.add_argument("--nanovdb", action="store_true", help="write with the old Warp path instead")
parser.add_argument("--no-material", action="store_true", help="leave the volume unshaded")
parser.add_argument("--cloud-fraction", type=float, default=0.5)
parser.add_argument("--seed", type=int, default=7)
parser.add_argument("--gpu", default=None, help="GPU index; default is the project's A6000")
parser.add_argument("--verbose", action="store_true", help="carb logging at info, for the loader")
parser.add_argument(
    "--rendermode",
    default="PathTracing",
    help="`/rtx/rendermode`. Kit 110 silently ignores a mode it does not know (ADR 0014 found "
    "this with RaytracedLighting), so rendering the same scene in two modes and finding the "
    "frames byte-identical is how you learn the setting did nothing.",
)
parser.add_argument(
    "--no-isvolume",
    action="store_true",
    help="leave `primvars:isVolume` off the cube (the 24 Sep control rendered as a lit fog "
    "without it; with it and the ptvol settings on, it rendered black)",
)
parser.add_argument(
    "--noise-cube",
    action="store_true",
    help="a cube with `isVolume` and the procedural OmniVolumeWorleyNoise material at its "
    "defaults -- no file anywhere, so it tests whether a heterogeneous volume lights at all",
)
parser.add_argument(
    "--pt-bounces",
    type=int,
    default=None,
    help="set /rtx/pathtracing/maxBounces and ptvol/maxBounces (the UI allows up to 63); the "
    "renderer's default of four starves a thick scattering slab",
)
parser.add_argument(
    "--density-scale",
    type=float,
    default=None,
    help="override `volume_density_scale` on the cube variants (tests the unit the renderer "
    "reads the grid in: per metre, or per something else)",
)
parser.add_argument(
    "--dome-light",
    action="store_true",
    help="add a uniform dome light beside the sun, in case volumes are not lit by distant lights",
)
parser.add_argument(
    "--texture-box",
    action="store_true",
    help="like --texture-cube, but the bounds are a mesh box with its vertices at the VDB's "
    "world bounds and **no transform**, so the prim's local space is world space. If the "
    "renderer samples the density texture in local space, a unit cube scaled by 5200 magnifies "
    "the grid 5200-fold and puts the camera inside cloud everywhere -- which is what a frame "
    "that darkens edge to edge, at any density scale, looks like.",
)
parser.add_argument(
    "--texture-cube",
    action="store_true",
    help="the documented recipe: a cube with `isVolume` on the VDB's bounds, and the VDB given "
    "to OmniVolumeDensity as its `volume_density_texture`",
)
parser.add_argument(
    "--cube",
    action="store_true",
    help="a cube of constant density with the same volume material and no VDB at all -- which "
    "separates 'the material does not exist on this build' from 'the grid is refused'",
)
args = parser.parse_args()
args._volume_path = None
args._volume_bounds = None

if args.gpu is not None:
    os.environ["IRSIM_GPU"] = str(args.gpu)

t_boot = time.time()
from isaacsim import SimulationApp  # noqa: E402

from irsim_isaac.env import simulation_app_config  # noqa: E402

app = SimulationApp(simulation_app_config())
BOOT_S = time.time() - t_boot

import carb  # noqa: E402

if args.verbose:
    _settings = carb.settings.get_settings()
    _settings.set("/log/level", "Info")
    _settings.set("/log/enableStandardStreamOutput", True)
import numpy as np  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import omni.usd  # noqa: E402
from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade  # noqa: E402

from irsim.atmosphere.cloud_deck import generate_cloud_deck  # noqa: E402
from irsim_isaac.cloud_volume import (  # noqa: E402
    author_cloud_volume,
    write_nanovdb,
    write_openvdb,
)

#: Extensions that own VDB loading and volume rendering. **None of them is in
#: `isaacsim.exp.base.python.kit`**, which is the app every render driver in this project boots,
#: so a `UsdVol.Volume` in the stage has had nothing to read its file. That is a far duller
#: explanation than the NanoVDB version mismatch ADR 0127 chased, and it fails identically:
#: no error, no warning, and a frame with no cloud in it.
VOLUME_EXTENSIONS = ("omni.volume", "omni.hydra.index", "omni.index")

#: AOVs the infrared path cares about, plus the colour that says whether the volume drew at all.
#: `distance_to_camera` is the one that decides AT.14: if a volume does not write depth, cloud in
#: front of an aircraft cannot occlude it in a band computed from AOVs.
AOVS = (
    "LdrColor",
    # Linear radiance, before the tonemapper and before any exposure adaptation. A bright volume
    # in one corner pulls the auto-exposure down over the whole frame, and the LDR then reports
    # a darker *backdrop* where the only change was a cloud: measured, 227 -> 52 on the backdrop
    # below a texture-cube. HDR cannot be fooled that way.
    "HdrColor",
    "distance_to_camera",
    "instance_id_segmentation",
    "semantic_segmentation",
    "normals",
)

#: Path-traced AOV contributions. `volumesAOV` is this build's own switch for "shading from VDB
#: volumes" and is the first thing to check when a volume loads and renders black.
VOLUME_SETTINGS = {
    # **The master switch.** Path-traced volumes are the "Non-uniform Volumes" frame of the render
    # settings, and that frame is gated by this one boolean (`omni.rtx.settings.core`,
    # `pt_widgets.NonUniformVolumesSettingsFrame._frame_setting_path`). ADR 0140's four runs set
    # every setting *inside* the frame and never the frame itself, which is why a volume with a
    # file and a cube without one were equally invisible.
    "/rtx/pathtracing/ptvol/enabled": True,
    "/rtx/pathtracing/ptvol/transmittanceMethod": 0,
    # The UI's own tooltips: "increase to more than 32 for highly scattering volumes like clouds".
    "/rtx/pathtracing/ptvol/maxCollisionCount": 128,
    "/rtx/pathtracing/ptvol/maxLightCollisionCount": 64,
    "/rtx/pathtracing/ptvol/maxBounces": 8,
    "/rtx/pathtracing/volumesAOV": True,
    "/rtx/pathtracing/maxVolumeBounces": 8,
    "/rtx/rtpt/maxVolumeBounces": 8,
    "/rtx/raytracing/globalVolumetricEffects/enabled": True,
}

#: The backdrop sits at **negative** Z because a USD camera with no rotation looks down -Z. The
#: first version of this probe put it at +Z, rendered an empty grey frame twice, and reported
#: "the volume did not load" -- which was true of the backdrop as well and of nothing else.
#: `backdrop_visible` in the report exists so that mistake cannot be made quietly again.
BACKDROP_Z_M = -9000.0
CAMERA_M = (0.0, 700.0, 3200.0)
CAMERA_PITCH_DEG = 12.0


def _density_cube(stage):
    """A box with the volume material and a constant density, carrying no field at all.

    The control for the whole probe. If this renders as fog then `OmniVolumeDensity` exists here
    and whatever is wrong is about the grid; if it renders as nothing then the material is
    missing and no VDB was ever going to help.
    """
    from pxr import Gf, Sdf, UsdGeom, UsdShade

    if args.texture_box and args._volume_bounds is not None:
        low, high = args._volume_bounds
        cube = UsdGeom.Mesh.Define(stage, "/World/Cloud")
        x0, y0, z0 = low
        x1, y1, z1 = high
        corners = [
            (x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
            (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1),
        ]  # fmt: skip
        cube.CreatePointsAttr([Gf.Vec3f(*c) for c in corners])
        cube.CreateFaceVertexCountsAttr([4] * 6)
        cube.CreateFaceVertexIndicesAttr(
            [0, 3, 2, 1, 4, 5, 6, 7, 0, 1, 5, 4, 1, 2, 6, 5, 2, 3, 7, 6, 3, 0, 4, 7]
        )
        cube.CreateExtentAttr([Gf.Vec3f(*low), Gf.Vec3f(*high)])
    else:
        cube = UsdGeom.Cube.Define(stage, "/World/Cloud")
        cube.CreateSizeAttr(1.0)
        cube.CreateExtentAttr([Gf.Vec3f(-0.5, -0.5, -0.5), Gf.Vec3f(0.5, 0.5, 0.5)])
    half = 1200.0
    xform = UsdGeom.Xformable(cube.GetPrim())
    if args.texture_box:
        pass
    elif args.texture_cube and args._volume_bounds is not None:
        low, high = args._volume_bounds
        centre = [0.5 * (a + b) for a, b in zip(low, high, strict=True)]
        size = [b - a for a, b in zip(low, high, strict=True)]
        xform.AddTranslateOp().Set(Gf.Vec3d(*centre))
        xform.AddScaleOp().Set(Gf.Vec3f(*size))
    else:
        xform.AddTranslateOp().Set(Gf.Vec3d(0.0, 1800.0, 0.0))
        xform.AddScaleOp().Set(Gf.Vec3f(4.0 * half, 2.0 * half, 4.0 * half))

    material = UsdShade.Material.Define(stage, "/World/Cloud/Material")
    shader = UsdShade.Shader.Define(stage, "/World/Cloud/Material/Shader")
    shader.CreateImplementationSourceAttr(UsdShade.Tokens.sourceAsset)
    if args.noise_cube:
        shader.SetSourceAsset(Sdf.AssetPath("OmniVolumeNoise.mdl"), "mdl")
        shader.SetSourceAssetSubIdentifier("OmniVolumeWorleyNoise", "mdl")
        # Its defaults are a dense grey medium at a 25-unit feature size; scale the feature to
        # the cube so the noise is visible from three kilometres.
        shader.CreateInput("size", Sdf.ValueTypeNames.Float).Set(600.0)
        shader.CreateInput("volume_density_scale", Sdf.ValueTypeNames.Float).Set(0.01)
        shader.CreateInput("volume_albedo", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.9, 0.9, 0.9))
    else:
        shader.SetSourceAsset(Sdf.AssetPath("OmniVolumeDensity.mdl"), "mdl")
        shader.SetSourceAssetSubIdentifier("OmniVolumeDensity", "mdl")
        # The material's real inputs (see `author_cloud_volume`). 0.004 per metre across 2.4 km
        # of cube is an optical depth near ten: opaque, but a path survives it.
        shader.CreateInput("volume_density_scale", Sdf.ValueTypeNames.Float).Set(0.004)
        shader.CreateInput("volume_albedo", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(1.0, 1.0, 1.0))
    if args.texture_cube or args.texture_box:
        # The documented recipe: a cube that says `isVolume`, with the VDB handed to the material
        # as its density texture. The cube is the bounds; the file is the density inside them.
        shader.CreateInput("volume_density_texture", Sdf.ValueTypeNames.Asset).Set(
            Sdf.AssetPath(str(args._volume_path))
        )
        shader.CreateInput("volume_density_scale", Sdf.ValueTypeNames.Float).Set(1.0)
    if args.density_scale is not None:
        shader.CreateInput("volume_density_scale", Sdf.ValueTypeNames.Float).Set(
            float(args.density_scale)
        )
    # Kit binds an MDL material through all three of its outputs; a stage authored by Composer
    # carries `mdl:surface`, `mdl:displacement` and `mdl:volume` on the same shader.
    for output in (
        material.CreateSurfaceOutput("mdl"),
        material.CreateDisplacementOutput("mdl"),
        material.CreateVolumeOutput("mdl"),
    ):
        output.ConnectToSource(shader.ConnectableAPI(), "out")
    UsdShade.MaterialBindingAPI(cube.GetPrim()).Bind(material)
    # A mesh only becomes a procedural volume for the path tracer when it says so; without this
    # primvar the cube is an opaque box with a volume material bound to it, which renders as
    # nothing in particular.
    if not args.no_isvolume:
        UsdGeom.PrimvarsAPI(cube.GetPrim()).CreatePrimvar("isVolume", Sdf.ValueTypeNames.Bool).Set(
            True
        )
    return cube.GetPrim()


def _resolve_mdl(name: str):
    """Where, if anywhere, the volume material lives. A `UsdShade.Shader` pointing at an MDL the
    resolver cannot find binds to nothing and the prim renders as if unshaded -- which for a
    volume means invisible, silently."""
    from pxr import Ar

    resolver = Ar.GetResolver()
    resolved = resolver.Resolve(name)
    paths = carb.settings.get_settings().get("/renderer/mdl/searchPaths/templates") or []
    return {"resolved": str(resolved) or None, "search_paths": list(paths)}


def enable_volume_extensions():
    """Turn on whatever this build has for volumes, and report what took."""
    import omni.kit.app

    manager = omni.kit.app.get_app().get_extension_manager()
    out = {}
    for name in VOLUME_EXTENSIONS:
        was = manager.is_extension_enabled(name)
        if not was:
            try:
                manager.set_extension_enabled_immediate(name, True)
            except Exception as exc:  # noqa: BLE001
                out[name] = f"enable failed: {type(exc).__name__}: {exc}"
                continue
        out[name] = (
            "already on"
            if was
            else ("enabled" if manager.is_extension_enabled(name) else "refused")
        )
    return out


def build_stage(volume_path, use_material: bool):
    """A backdrop, a cloud, a camera and one light. Returns (stage, camera, cloud prim)."""
    omni.usd.get_context().new_stage()
    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    stage.DefinePrim("/World", "Xform")

    # A backdrop behind the cloud, wide enough to fill the frame, and bright, so that anything
    # the cloud does to it is a large signal rather than a subtle one.
    backdrop = UsdGeom.Mesh.Define(stage, "/World/Backdrop")
    half = 20000.0
    backdrop.CreatePointsAttr(
        [
            Gf.Vec3f(-half, -half, BACKDROP_Z_M),
            Gf.Vec3f(half, -half, BACKDROP_Z_M),
            Gf.Vec3f(half, half, BACKDROP_Z_M),
            Gf.Vec3f(-half, half, BACKDROP_Z_M),
        ]
    )
    backdrop.CreateFaceVertexCountsAttr([4])
    backdrop.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    backdrop.CreateExtentAttr(
        [Gf.Vec3f(-half, -half, BACKDROP_Z_M - 1.0), Gf.Vec3f(half, half, BACKDROP_Z_M + 1.0)]
    )
    backdrop.CreateDisplayColorAttr([Gf.Vec3f(0.85, 0.2, 0.2)])
    material = UsdShade.Material.Define(stage, "/World/Backdrop/Mat")
    shader = UsdShade.Shader.Define(stage, "/World/Backdrop/Mat/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.85, 0.2, 0.2))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(1.0)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI(backdrop.GetPrim()).Bind(material)

    light = UsdLux.DistantLight.Define(stage, "/World/Sun")
    light.CreateIntensityAttr(3000.0)
    light.CreateAngleAttr(0.53)
    UsdGeom.Xformable(light.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3f(-35.0, 25.0, 0.0))
    if args.dome_light:
        dome = UsdLux.DomeLight.Define(stage, "/World/Dome")
        dome.CreateIntensityAttr(1000.0)
        dome.CreateColorAttr(Gf.Vec3f(0.6, 0.7, 1.0))

    cloud = None
    # `volume_path is None` means "the control stage": nothing under test in it. The cube has to
    # honour that too -- the first version built it into the "without" stage as well, so the cube
    # run differenced the cube against itself, read zero, and was written up as "the material is
    # as invisible as the grid" (ADR 0140). It was not; the control was wrong.
    if volume_path is None:
        pass
    elif args.cube or args.texture_cube or args.texture_box or args.noise_cube:
        args._volume_path = volume_path.path
        low = volume_path.min_world_m
        args._volume_bounds = (
            tuple(float(v) for v in low),
            tuple(float(low[i] + volume_path.shape[i] * volume_path.voxel_m) for i in range(3)),
        )
        cloud = _density_cube(stage)
    else:
        cloud = author_cloud_volume(stage, "/World/Cloud", volume_path, with_material=use_material)

    camera = UsdGeom.Camera.Define(stage, "/World/Camera")
    camera.CreateFocalLengthAttr(24.0)
    camera.CreateHorizontalApertureAttr(36.0)
    camera.CreateClippingRangeAttr(Gf.Vec2f(1.0, 100000.0))
    xform = UsdGeom.Xformable(camera.GetPrim())
    xform.AddTranslateOp().Set(Gf.Vec3d(*CAMERA_M))
    xform.AddRotateXYZOp().Set(Gf.Vec3f(CAMERA_PITCH_DEG, 0.0, 0.0))
    return stage, "/World/Camera", cloud


def capture(camera_path, resolution, settle, spp):
    """Every AOV in one pass, as plain arrays."""
    settings = carb.settings.get_settings()
    settings.set("/rtx/rendermode", args.rendermode)
    settings.set("/rtx/pathtracing/spp", int(spp))
    settings.set("/rtx/pathtracing/totalSpp", int(spp))
    settings.set("/rtx/post/aa/op", 0)
    # No auto-exposure: the LDR frame is compared across captures, and adaptation to a bright
    # volume would change the backdrop in a frame where only the cloud changed.
    settings.set("/rtx/post/histogram/enabled", False)
    for key, value in VOLUME_SETTINGS.items():
        settings.set(key, value)
    if args.pt_bounces is not None:
        settings.set("/rtx/pathtracing/maxBounces", int(args.pt_bounces))
        settings.set("/rtx/pathtracing/ptvol/maxBounces", int(args.pt_bounces))
        settings.set("/rtx/pathtracing/maxSpecularAndTransmissionBounces", int(args.pt_bounces))

    product = rep.create.render_product(camera_path, tuple(resolution))
    path = product.path if hasattr(product, "path") else str(product)
    annotators = {}
    for name in AOVS:
        try:
            anno = rep.AnnotatorRegistry.get_annotator(name, device="cpu")
            anno.attach(path)
            annotators[name] = anno
        except Exception as exc:  # noqa: BLE001
            print(f"  [{name}] attach failed: {type(exc).__name__}: {exc}")
    for _ in range(settle):
        rep.orchestrator.step(rt_subframes=1)
    out = {}
    for name, anno in annotators.items():
        try:
            out[name] = _as_array(anno.get_data())
        except Exception as exc:  # noqa: BLE001
            print(f"  [{name}] read failed: {type(exc).__name__}: {exc}")
        finally:
            with contextlib.suppress(Exception):
                anno.detach(path)
    with contextlib.suppress(Exception):
        product.destroy()
    return out


def _as_array(data):
    if isinstance(data, dict):
        for key in ("data", "info"):
            if key in data and hasattr(data[key], "shape"):
                return np.asarray(data[key])
        for value in data.values():
            if hasattr(value, "shape"):
                return np.asarray(value)
    return np.asarray(data)


def changed_fraction(with_cloud, without_cloud):
    """Fraction of pixels the volume moved, and by how much at the 99th percentile.

    An integer AOV (a segmentation id) has no meaningful difference, only equality, so the two
    kinds are kept apart rather than run through one expression -- which is how the first version
    of this raised a `NameError` on `instance_id_segmentation` and, because Isaac's fast shutdown
    ends the process inside `app.close()`, printed no traceback and exited 0.
    """
    if with_cloud is None or without_cloud is None:
        return {"status": "missing"}
    a, b = np.asarray(with_cloud), np.asarray(without_cloud)
    if a.shape != b.shape:
        return {"status": "shape_mismatch", "with": list(a.shape), "without": list(b.shape)}

    entry = {"status": "ok", "dtype": str(a.dtype), "shape": list(a.shape)}
    if a.dtype.kind in "fc":
        finite = np.isfinite(a) & np.isfinite(b)
        delta = np.zeros(a.shape, dtype=np.float64)
        delta[finite] = np.abs(a[finite].astype(np.float64) - b[finite].astype(np.float64))
        scale = max(float(np.max(np.abs(b[finite]))) if finite.any() else 1.0, 1e-9)
        moved = delta > 1e-3 * scale
        entry["p99_abs_delta"] = round(float(np.percentile(delta, 99)), 5)
        entry["max_abs_delta"] = round(float(delta.max()), 5)
    else:
        moved = a != b
    while moved.ndim > 2:
        moved = moved.any(axis=-1)
    entry["fraction_changed"] = round(float(moved.mean()), 5)
    return entry


def main() -> int:
    out_dir = args.out if os.path.isabs(args.out) else os.path.join(os.path.dirname(HERE), args.out)
    os.makedirs(out_dir, exist_ok=True)

    deck = generate_cloud_deck(
        beta=2.8,
        cloud_fraction=args.cloud_fraction,
        seed=args.seed,
        base_m=1200.0,
        optical_depth=25.0,
        min_elevation_deg=25.0,
    )
    writer = write_nanovdb if args.nanovdb else write_openvdb
    suffix = ".nvdb" if args.nanovdb else ".vdb"
    t0 = time.time()
    volume = writer(deck, os.path.join(out_dir, "cloud" + suffix), voxel_m=args.voxel_m)
    write_s = time.time() - t0
    print(f"wrote {volume.path.name}: {volume.caption}  [{write_s:.1f} s]")

    report = {
        "boot_s": round(BOOT_S, 1),
        "rendermode": args.rendermode,
        "writer": "nanovdb" if args.nanovdb else "openvdb",
        "volume": {
            "file": str(volume.path),
            "caption": volume.caption,
            "voxel_m": volume.voxel_m,
            "shape": list(volume.shape),
            "min_world_m": list(volume.min_world_m),
            "active_voxels": volume.active_voxels,
            "write_s": round(write_s, 2),
        },
        "material": not args.no_material,
        "settings_requested": VOLUME_SETTINGS,
        "aovs": {},
    }

    report["extensions"] = enable_volume_extensions()
    report["mdl_search"] = _resolve_mdl("OmniVolumeDensity.mdl")
    print("OmniVolumeDensity.mdl resolves to:", report["mdl_search"])
    print("volume extensions:", report["extensions"])

    print("rendering with the volume ...")
    _, camera, cloud = build_stage(volume, not args.no_material)
    report["cloud_prim_valid"] = bool(cloud and cloud.IsValid())
    if cloud and cloud.IsValid():
        bound = UsdGeom.Imageable(cloud).ComputeWorldBound(0.0, UsdGeom.Tokens.default_)
        box = bound.ComputeAlignedBox()
        report["cloud_world_bound"] = {
            "min": [round(float(v), 1) for v in box.GetMin()],
            "max": [round(float(v), 1) for v in box.GetMax()],
            "empty": bool(box.IsEmpty()),
        }
        print(f"  cloud bound: {report['cloud_world_bound']}")
    with_cloud = capture(camera, args.resolution, args.settle, args.spp)

    print("rendering without it ...")
    _, camera, _ = build_stage(None, False)
    without_cloud = capture(camera, args.resolution, args.settle, args.spp)

    # **The noise floor, measured rather than assumed.** A real-time render is temporally
    # accumulated and denoised, so two captures of an identical stage are not identical arrays --
    # and the first version of this probe called that difference a volume, reporting that 43 % of
    # the frame had changed between two pictures the eye cannot tell apart. The same scene is
    # therefore rendered a second time and its own change measured; anything the volume claims
    # has to beat it.
    print("rendering without it again, for the noise floor ...")
    _, camera, _ = build_stage(None, False)
    repeat = capture(camera, args.resolution, args.settle, args.spp)

    settings = carb.settings.get_settings()
    report["settings_readback"] = {k: settings.get(k) for k in VOLUME_SETTINGS}
    report["settings_readback"]["/rtx/rendermode"] = settings.get("/rtx/rendermode")
    report["settings_readback"]["/rtx/pathtracing/maxBounces"] = settings.get(
        "/rtx/pathtracing/maxBounces"
    )

    for name in AOVS:
        entry = changed_fraction(with_cloud.get(name), without_cloud.get(name)) or {}
        noise = changed_fraction(repeat.get(name), without_cloud.get(name)) or {}
        entry["noise_floor"] = noise.get("fraction_changed")
        entry["noise_p99"] = noise.get("p99_abs_delta")
        signal = entry.get("fraction_changed") or 0.0
        floor = noise.get("fraction_changed")
        entry["above_noise"] = bool(
            floor is not None and signal > 0.05 and signal > 3.0 * max(float(floor), 1e-4)
        )
        report["aovs"][name] = entry

    # **Is the scene even in frame?** A probe that renders nothing twice reports a difference of
    # zero, which reads exactly like a volume that failed to load. The backdrop is deliberately a
    # colour nothing else in the stage is, so its presence is a check on the camera rather than on
    # the thing under test.
    # Depth, not colour: the backdrop is tone-mapped to a pale pink rather than the red it was
    # authored as, and a colour test tuned on the authored value calls a perfectly good frame
    # empty. A finite distance is what "there is geometry in front of the camera" means.
    plain_depth = without_cloud.get("distance_to_camera")
    if plain_depth is not None:
        depth = np.asarray(plain_depth, dtype=np.float64)
        report["backdrop_visible"] = round(float((np.isfinite(depth) & (depth < 1e5)).mean()), 4)
        report["backdrop_distance_m"] = round(float(np.nanmin(depth)), 1)
    else:
        report["backdrop_visible"] = 0.0
    if report["backdrop_visible"] < 0.05:
        print(
            "\n!! the backdrop is not in frame, so this run says nothing about the volume.\n"
            f"   only {report['backdrop_visible']:.1%} of the frame is backdrop; aim the camera."
        )

    colour = report["aovs"].get("LdrColor") or {}
    loaded = bool(colour.get("above_noise"))
    report["volume_loaded"] = loaded
    depth = report["aovs"].get("distance_to_camera") or {}
    report["volume_reaches_depth"] = bool(depth.get("above_noise"))

    for name, arr in with_cloud.items():
        np.save(os.path.join(out_dir, f"with_{name}.npy"), arr)
    for name, arr in without_cloud.items():
        np.save(os.path.join(out_dir, f"without_{name}.npy"), arr)
    for name, arr in repeat.items():
        np.save(os.path.join(out_dir, f"repeat_{name}.npy"), arr)
    _save_png(with_cloud.get("LdrColor"), os.path.join(out_dir, "with_cloud.png"))
    hdr = with_cloud.get("HdrColor")
    if hdr is not None:
        # One fixed tone curve for every run, anchored on the *control* frame's brightest
        # percentile, so a picture from one run can be compared with a picture from another.
        ref = without_cloud.get("HdrColor")
        linear = np.asarray(hdr, dtype=np.float64)[..., :3]
        anchor = np.asarray(ref if ref is not None else hdr, dtype=np.float64)[..., :3]
        scale = max(float(np.nanpercentile(anchor, 99.5)), 1e-9)
        mapped = np.clip(linear / scale, 0.0, 1.0) ** (1.0 / 2.2)
        _save_png((mapped * 255.0).astype(np.uint8), os.path.join(out_dir, "with_cloud_hdr.png"))
        H = linear.shape[0]
        report["hdr_means"] = {
            "with_top_half": [round(float(v), 4) for v in linear[: H // 2].reshape(-1, 3).mean(0)],
            "with_bottom_quarter": [
                round(float(v), 4) for v in linear[3 * H // 4 :].reshape(-1, 3).mean(0)
            ],
            "without_top_half": [
                round(float(v), 4) for v in anchor[: H // 2].reshape(-1, 3).mean(0)
            ],
            "with_p99_9": round(float(np.nanpercentile(linear, 99.9)), 4),
            "without_p99_9": round(float(np.nanpercentile(anchor, 99.9)), 4),
        }
        print("hdr means:", report["hdr_means"])
    _save_png(without_cloud.get("LdrColor"), os.path.join(out_dir, "without_cloud.png"))

    path = os.path.join(out_dir, "report.json")
    with open(path, "w") as handle:
        json.dump(report, handle, indent=2)

    print("\n--- what the render said -------------------------------------------")
    print(f"backdrop in frame    : {report['backdrop_visible']:.1%}")
    print(
        f"volume loaded at all : {loaded}   (LdrColor changed on "
        f"{colour.get('fraction_changed')} of the frame)"
    )
    for name in AOVS:
        entry = report["aovs"][name] or {"status": "missing"}
        print(
            f"  {name:28s} {str(entry.get('status')):14s} changed "
            f"{entry.get('fraction_changed')}  noise {entry.get('noise_floor')}  "
            f"-> {'SIGNAL' if entry.get('above_noise') else 'noise'}"
        )
    print(f"\nreport: {path}")
    if report["backdrop_visible"] < 0.05:
        return 3
    return 0 if loaded else 2


def _save_png(array, path):
    if array is None:
        return
    a = np.asarray(array)
    if a.ndim == 3 and a.shape[-1] >= 3:
        a = a[..., :3]
    if a.dtype != np.uint8:
        a = np.clip(a, 0.0, 1.0) * 255 if a.dtype.kind == "f" else a
        a = a.astype(np.uint8)
    import struct
    import zlib

    h, w = a.shape[:2]
    raw = b"".join(b"\x00" + a[y].tobytes() for y in range(h))

    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data))

    with open(path, "wb") as handle:
        handle.write(
            b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 6))
            + chunk(b"IEND", b"")
        )


# `app.close()` ends the process from inside itself when `fastShutdown` is on, so an exception
# raised in `main` and re-raised past a bare `finally` never prints and the shell sees exit 0.
# The traceback is therefore printed *here*, before anything is allowed to close.
code = 3
try:
    code = main()
except BaseException:  # noqa: BLE001
    import traceback

    traceback.print_exc()
    sys.stdout.flush()
    sys.stderr.flush()
finally:
    app.close()
sys.exit(code)
