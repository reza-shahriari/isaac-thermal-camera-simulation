#!/usr/bin/env python3
"""Roadmap AI.2 (in-engine half): render an **imported** third-party aircraft in LWIR.

    IRSIM_GPU=0 python.sh scripts/render_phantom4.py --frames 24 --range-m 4.0

Every other render driver in this repo films geometry this project authored in Python. This one
films a 62 MB DJI Phantom 4 Pro FBX that nobody here modelled, prepared on the CPU by
``scripts/prep_asset.py`` (ADR 0128) and referenced onto the stage as-is. What makes that possible
is not a new renderer path but a new *mapping* rung: the asset's 21 source material names resolve
through ``configs/assets/phantom4.yaml`` instead of through globs that were written for cars.

**The asset's frame is the stage's frame.** The prepared USD is Z-up at ``metersPerUnit = 1`` --
the prep tool applied ``scale_to_metres`` once -- and this stage is authored Z-up to match, so
nothing here transforms the aircraft. The other aerial drivers are Y-up because the geometry they
author is; an imported asset brings its own convention and the stage bends to it rather than the
other way round, because a second scale or a silent axis swap is exactly what ADR 0128 exists to
stop. The aircraft sits where the archive puts it, near y = +0.49 m.

**Temperature is per prim here, not per cell.** The scene's mesh fields are solved (234,923 cells)
but `IrCamera` takes planar `SurfaceBinding`s only, and `MeshPointBridge` -- the object that turns
a world position into a cell on a real mesh -- has never been driven by a render: its only driver,
``scripts/quad_flight_mesh.py``, runs on a synthetic G-buffer. Wiring it into the camera is the
remainder of `AI.2` and is called out in the summary this writes rather than papered over. What
*is* per prim is already the thing an imported asset could not do at all yesterday: 41 prims, each
with its own emissivity and its own temperature.

Materials come from the asset map, so `Copper` reads as a coated winding at eps 0.90 rather than
falling through to UNMAPPED, and `Metal_Matte` reads as a painted housing rather than as a mirror.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
from typing import Any

REPO = pathlib.Path(__file__).resolve().parents[1]

#: Where the prepared asset's prims land once referenced.
ASSET_ROOT = "/World/Targets/phantom4"

#: Source material name -> thermal target. The asset is grouped by material, so this is the only
#: granularity available: the `Copper` prims are the motor windings and the rest is airframe.
TARGET_BY_MATERIAL: dict[str, str] = {"copper": "motor"}
DEFAULT_TARGET = "airframe"

parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
parser.add_argument("--asset", default="phantom4")
parser.add_argument("--scene", default="configs/scenes/phantom4_pointwise.yaml")
parser.add_argument("--sensor", default="configs/sensors/flir_boson_640_lwir.yaml")
parser.add_argument("--out", default="outputs/phantom4")
parser.add_argument("--frames", type=int, default=24)
parser.add_argument("--fps", type=float, default=12.0)
parser.add_argument("--interval-s", type=float, default=60.0, help="scene seconds between frames")
parser.add_argument("--range-m", type=float, default=4.0, help="camera distance from the aircraft")
parser.add_argument("--elevation-deg", type=float, default=18.0)
parser.add_argument("--azimuth-deg", type=float, default=35.0)
parser.add_argument("--rt-subframes", type=int, default=8)
parser.add_argument("--settle", type=int, default=16)
parser.add_argument("--span", default=None, help="display span 'loC,hiC'; default from the solve")


def main(argv: list[str] | None = None) -> int:
    args = parser.parse_args(argv)
    usd = REPO / "data" / "assets" / args.asset / f"{args.asset}.usdc"
    if not usd.exists():
        print(
            f"no prepared asset at {usd}\n"
            f"  run: python scripts/prep_asset.py --asset {args.asset} --emit-mesh",
            file=sys.stderr,
        )
        return 2

    from isaacsim import SimulationApp

    from irsim_isaac.env import simulation_app_config

    app = SimulationApp(simulation_app_config())
    status = 1
    try:
        status = _render(args, usd)
    except BaseException:  # noqa: BLE001 - print BEFORE closing: close() ends in os._exit
        import traceback

        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        status = 1
    app.close(exit_code=status)
    return status


def _render(args: Any, usd: pathlib.Path) -> int:
    import numpy as np
    import omni.usd
    from pxr import Gf, UsdGeom

    from irsim.atmosphere.skylight import skylight_for_sensor
    from irsim.config.loader import load_sensor_config
    from irsim.io.png import write_png
    from irsim.materials.library import MaterialLibrary
    from irsim.materials.mapping import MaterialResolver, load_asset_mapping, load_mapping_rules
    from irsim.materials.table import MaterialTable
    from irsim.pipeline.core import PipelineConfig
    from irsim.radiometry.lut_files import (
        load_band_lut_for_config,
        load_band_response_for_config,
    )
    from irsim.scene import Scene
    from irsim_eval.video import encode_mp4, ffmpeg_available
    from irsim_isaac.aircraft_pass import look_at_quaternion
    from irsim_isaac.pipeline.ir_camera import IrCamera
    from irsim_isaac.pipeline.materials_usd import prim_records

    out = REPO / args.out
    out.mkdir(parents=True, exist_ok=True)

    # --- the stage: Z-up, to match the asset rather than transform it ---------------------
    ctx = omni.usd.get_context()
    ctx.new_stage()
    stage = ctx.get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    stage.DefinePrim("/World", "Xform")
    stage.DefinePrim("/World/Targets", "Xform")
    target = stage.DefinePrim(ASSET_ROOT, "Xform")
    target.GetReferences().AddReference(str(usd))

    records = prim_records(stage, root=ASSET_ROOT)
    if not records:
        print(f"the reference produced no prims under {ASSET_ROOT}", file=sys.stderr)
        return 3

    # --- materials, through the per-asset map (ADR 0128) ------------------------------------
    sensor = load_sensor_config(REPO / args.sensor)
    band = sensor.sensor.band.band_id
    lut = load_band_lut_for_config(sensor, REPO / "data" / "lut")
    response = load_band_response_for_config(sensor, REPO / "data")
    skylight = skylight_for_sensor(sensor, "lb")
    scene = Scene.from_file(
        REPO / args.scene,
        {band: lut},
        responses={band: response},
        quantity="lb",
        skylights={band: skylight},
    )
    library = MaterialLibrary.load()
    table = MaterialTable.from_library(library, band)
    asset_map = load_asset_mapping(args.asset, known_materials=library.names)
    resolver = MaterialResolver(
        load_mapping_rules(known_materials=library.names), list(table.names), asset=asset_map
    )
    resolutions = resolver.resolve_all(records)
    unmapped = [r.path for r in resolutions if not r.mapped]
    print(f"\nmaterials: {len(resolutions) - len(unmapped)}/{len(resolutions)} prims mapped")
    if unmapped:
        print(f"  UNMAPPED (would render magenta): {len(unmapped)}")
        for path in unmapped[:5]:
            print(f"    {path}")

    # --- every prim needs a thermal node, or the camera refuses it --------------------------
    prim_to_target = {}
    for record in records:
        name = (record.material_name or "").lower()
        prim_to_target[record.path] = TARGET_BY_MATERIAL.get(name, DEFAULT_TARGET)
    n_motor = sum(1 for v in prim_to_target.values() if v == "motor")
    print(
        print(
            f"thermal nodes: {n_motor} prim(s) on `motor`, "
            f"{len(prim_to_target) - n_motor} on `airframe`"
        )
    )

    # --- a camera looking at the aircraft ----------------------------------------------------
    cache = UsdGeom.BBoxCache(__import__("pxr").Usd.TimeCode.Default(), ["default"])
    box = cache.ComputeWorldBound(target).ComputeAlignedRange()
    centre = np.asarray([box.GetMidpoint()[i] for i in range(3)], dtype=np.float64)
    el, az = math.radians(args.elevation_deg), math.radians(args.azimuth_deg)
    offset = args.range_m * np.asarray(
        [math.cos(el) * math.sin(az), math.cos(el) * math.cos(az), math.sin(el)]
    )
    eye = centre + offset
    q = look_at_quaternion(centre - eye)
    camera = UsdGeom.Camera.Define(stage, "/World/IrCamera")
    xform = UsdGeom.Xformable(camera)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(*(float(v) for v in eye)))
    xform.AddOrientOp().Set(Gf.Quatf(float(q[0]), Gf.Vec3f(*(float(v) for v in q[1:]))))
    camera.GetClippingRangeAttr().Set(Gf.Vec2f(0.01, 1.0e5))
    fpa = sensor.sensor.fpa
    hfov_rad = 2.0 * math.atan(
        0.5 * fpa.width * fpa.pitch_um * 1e-3 / sensor.sensor.optics.focal_length_mm
    )
    span_px = fpa.width * (0.41 / (2.0 * args.range_m * math.tan(0.5 * hfov_rad)))
    print(
        f"camera: {args.range_m:.2f} m from centre {centre.round(3).tolist()}, "
        f"{fpa.width}x{fpa.height}, {math.degrees(hfov_rad):.1f} deg HFOV, "
        f"aircraft ~{span_px:.0f} px across"
    )

    pipeline = PipelineConfig.from_sensor(
        sensor,
        table,
        lut,
        sky=scene.sky_models[band],
        atmosphere=scene.layered,
    )
    cam = IrCamera(
        sensor,
        scene,
        pipeline=pipeline,
        prim_to_target=prim_to_target,
        resolutions=resolutions,
        camera_path="/World/IrCamera",
        stage=stage,
        up_axis="Z",
        frame_period_s=args.interval_s,
        strict_patch_coverage=False,
        strict_materials=True,
        strict_thermal_nodes=True,
    )
    cam.open(settle_frames=args.settle, rt_subframes=args.rt_subframes)

    lo_hi = None
    if args.span:
        lo_hi = tuple(float(v) + 273.15 for v in args.span.split(","))

    frames: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    for index in range(args.frames):
        outputs = cam.get_outputs(rt_subframes=args.rt_subframes)
        if outputs.apparent_t is None:
            raise RuntimeError("the sensor config produced no apparent_t plane")
        t_app = np.asarray(outputs.apparent_t, dtype=np.float32)
        frames.append(t_app)
        # `material_id` lives on the *supersampled* grid (ADR 0014: ids are never filtered, so
        # anti-aliasing happens by supersampling them and filtering radiance afterwards). Fold it
        # down with `any`, because a display pixel showing any aircraft at all is an aircraft
        # pixel -- averaging ids would invent a material that is not in the scene.
        last = cam.last_frame
        mat = None if last is None else getattr(last, "material_id", None)
        if mat is None:
            hit = t_app
        else:
            m = np.asarray(mat) > 0
            k = m.shape[0] // t_app.shape[0]
            if k > 1:
                m = m.reshape(t_app.shape[0], k, t_app.shape[1], k).any(axis=(1, 3))
            hit = t_app[m]
        rows.append(
            {
                "frame": index,
                "t_s": float(cam.last_frame_t_s or 0.0),
                "target_min_c": float(hit.min() - 273.15) if hit.size else None,
                "target_max_c": float(hit.max() - 273.15) if hit.size else None,
                "target_px": int(hit.size),
                "samples": hit.astype(np.float64).tolist() if hit.size < 40000 else [],
            }
        )
        if index in (0, args.frames - 1):
            r = rows[-1]
            print(
                f"  frame {index:3d}  t+{r['t_s'] - rows[0]['t_s']:7.0f}s  "
                f"aircraft {r['target_px']:6d} px  "
                f"{r['target_min_c']:6.1f} .. {r['target_max_c']:6.1f} C"
            )
        np.save(out / f"frame_{index:06d}_apparent_t.npy", t_app)

    stack = np.stack(frames)
    if lo_hi is None:
        # From the aircraft's own pixels. A span taken over the whole frame is a span over sky:
        # this scene is ~99 % background, so the percentiles land inside it and the target
        # saturates to one flat white shape -- a picture of the sky's noise, not of an aircraft.
        target_pixels = np.concatenate([np.asarray(r["samples"]) for r in rows])
        lo_hi = (
            float(np.percentile(target_pixels, 1.0)),
            float(np.percentile(target_pixels, 99.0)),
        )
    lo, hi = lo_hi
    print(f"display span: {lo - 273.15:.1f} .. {hi - 273.15:.1f} C, white-hot grayscale")

    frames_dir = out / "frames"
    frames_dir.mkdir(exist_ok=True)
    for index, plane in enumerate(frames):
        eight = np.clip((plane - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
        write_png(frames_dir / f"frame_{index:06d}.png", (eight * 255.0).astype(np.uint8))

    video = None
    if ffmpeg_available():
        video = out / f"{args.asset}_ir.mp4"
        encode_mp4(str(frames_dir / "frame_*.png"), video, fps=args.fps)
        print(f"wrote {video}")
    else:
        print("ffmpeg not available: stills written, no video", file=sys.stderr)
    summary = {
        "asset": args.asset,
        "usd": str(usd),
        "prims": len(records),
        "mapped": len(resolutions) - len(unmapped),
        "unmapped": len(unmapped),
        "frames": args.frames,
        "display_span_c": [lo - 273.15, hi - 273.15],
        "video": str(video) if video else None,
        "rows": [{k: v for k, v in r.items() if k != "samples"} for r in rows],
        "temperature_granularity": (
            "per prim. The scene's 234,923 mesh cells are solved but IrCamera takes planar "
            "SurfaceBindings only; wiring MeshPointBridge into the camera is AI.2's remainder."
        ),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {out}/summary.json")
    print(f"frame span across the clip: {stack.min() - 273.15:.1f} .. {stack.max() - 273.15:.1f} C")
    return 0


if __name__ == "__main__":
    sys.exit(main())
