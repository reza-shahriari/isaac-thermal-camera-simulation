#!/usr/bin/env python3
"""IG.20: render a phase-P patch scene in Isaac Sim, from its scene config alone.

Phase P's reference scenes are thermal scenes: world-frame patches, each naming the prim its
field is drawn on, solved and tested engine-free on a synthetic G-buffer. Their rows all said the
rendered frame "is IG.2's", and none had ever been rendered, because none had a stage. This
driver builds one from the config (:mod:`irsim_isaac.patch_stage`): the config is rotated into
the Y-up stage frame (a rotation, so every cell solves to the same temperature -- the unit test
holds that), every patched surface becomes a quad carrying its own ``thermal:material``, a camera
is placed from a per-scene preset or the command line, and :class:`IrCamera` draws each quad from
its solved field, per pixel, with ``strict_patch_coverage`` so a pixel that fell back to a
per-prim temperature raises instead of looking like physics.

The frames are a **time-lapse** of the scene's own clock: ``--frames`` captures ``--interval-s``
apart from the config's start, played at ``--fps``. The infrared clip is on one fixed span taken
from the solved cells over the whole run (so two frames are comparable), the camera's own AGC is
filmed beside it, and the visible companion -- dome, sun and shadows from the same NOAA sun the
solver used -- is paired with the infrared.

Run with the Isaac interpreter (docs/decisions/0002), after `make luts`:
    IRSIM_GPU=0 python.sh scripts/render_patch_scene.py --scene wall_half_in_sun
    IRSIM_GPU=0 python.sh scripts/render_patch_scene.py --scene wet_road_noon
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import sys
import time
from typing import Any

REPO = pathlib.Path(__file__).resolve().parents[1]

#: Per-scene camera placements, in the config's own ENU metres (x east, y north, z up), and the
#: time-lapse each scene is about. Plain data so the defaults resolve before Kit boots.
PRESETS: dict[str, dict[str, Any]] = {
    # West-south-west and high: over the neighbour's roof (4 m) into the 2 m alley, so the west
    # wall is seen with the neighbour's shadow across it and the caster standing in front of it.
    # 18:00-20:00 local: the shadow climbs the wall as the sun sets behind the neighbour.
    "wall_half_in_sun": {
        "eye": (-18.0, -7.0, 17.0),
        "look": (0.0, 3.0, 2.5),
        "frames": 48,
        "interval_s": 150.0,
    },
    # North-west and above the road: the wet west half, the dry east half, and the wall's shadow
    # across the south rows, over the two hours the film takes to dry.
    "wet_road_noon": {
        "eye": (-5.0, 21.0, 11.0),
        "look": (6.0, 5.0, 0.0),
        "frames": 48,
        "interval_s": 150.0,
    },
    # PH.6: the tailpipe and its plume from behind and to the side at bumper height, over the
    # first four minutes of a drive -- the load steps from idle to cruise at two minutes and the
    # gas leaving the pipe heats with it. Rendered once per band: the plume is CO2/H2O emission,
    # bright in MWIR and faint in LWIR, from one authored cone. The scene's two targets have no
    # patches, so their prims are authored here: a box for the shell, a short pipe at the
    # plume's own origin (ENU; kind, centre, size, material, node, axis).
    "car_exhaust_plume": {
        "eye": (-4.8, -4.6, 1.1),
        "look": (-2.7, 0.0, 0.45),
        "frames": 25,
        "interval_s": 10.0,
        "parts": (
            ("shell", "box", (0.0, 0.0, 0.85), (4.2, 1.8, 1.3), "car_paint_white", "shell", "X"),
            (
                "tailpipe",
                "cylinder",
                (-2.1, 0.0, 0.28),
                (0.3, 0.09, 0.09),
                "iron_weathered",
                "tailpipe",
                "X",
            ),
        ),
    },
}

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--scene", required=True, help="a name under configs/scenes/, or a path")
parser.add_argument("--out", default=None, help="default outputs/<scene>_<band>")
parser.add_argument("--sensor", default=str(REPO / "configs/sensors/flir_boson_640_lwir.yaml"))
parser.add_argument("--frames", type=int, default=None)
parser.add_argument("--interval-s", type=float, default=None, help="scene seconds per frame")
parser.add_argument("--eye", type=float, nargs=3, default=None, help="camera, ENU metres")
parser.add_argument("--look", type=float, nargs=3, default=None, help="aim point, ENU metres")
parser.add_argument("--fps", type=float, default=10.0, help="playback rate of the encoded video")
parser.add_argument("--settle", type=int, default=16)
parser.add_argument("--rt-subframes", type=int, default=8)
parser.add_argument("--no-rgb", action="store_true")
parser.add_argument("--no-chain", action="store_true", help="ideal camera: no M9 sensor chain")
parser.add_argument("--span-c", type=float, nargs=2, default=None, help="fixed IR span, deg C")
parser.add_argument("--plane-stride", type=int, default=8)
parser.add_argument("--keep-frames", action="store_true")
parser.add_argument("--float-format", default="npy", choices=("npy", "exr"))
args = parser.parse_args()

scene_path = pathlib.Path(args.scene)
if not scene_path.suffix:
    scene_path = REPO / "configs" / "scenes" / f"{args.scene}.yaml"
preset = PRESETS.get(scene_path.stem, {})
for key, default in (("frames", 48), ("interval_s", 150.0)):
    if getattr(args, key) is None:
        setattr(args, key, preset.get(key, default))
if args.eye is None or args.look is None:
    if not preset:
        parser.error(f"{scene_path.stem} has no camera preset: pass --eye and --look (ENU metres)")
    args.eye = args.eye or list(preset["eye"])
    args.look = args.look or list(preset["look"])

t_boot = time.time()
from isaacsim import SimulationApp  # noqa: E402

from irsim_isaac.env import simulation_app_config  # noqa: E402

app = SimulationApp(simulation_app_config())
boot_s = time.time() - t_boot


def main() -> int:
    import numpy as np
    import omni.usd
    from pxr import Gf, UsdGeom

    from irsim.atmosphere.skylight import skylight_for_sensor
    from irsim.config.loader import band_hash, config_hash, load_sensor_config
    from irsim.config.scene import load_scene_config
    from irsim.io.dataset import FrameWriter
    from irsim.io.png import write_png
    from irsim.isp.palette import palette_table, quantise_display
    from irsim.materials.library import MaterialLibrary
    from irsim.materials.mapping import MaterialResolver, load_mapping_rules
    from irsim.materials.table import MaterialTable
    from irsim.pipeline.core import PipelineConfig
    from irsim.radiometry.lut_files import (
        load_band_lut_for_config,
        load_band_response_for_config,
    )
    from irsim.scene import Scene
    from irsim_eval.video import encode_mp4, ffmpeg_available, overlay_readout, with_margin
    from irsim_isaac.aircraft_pass import look_at_quaternion
    from irsim_isaac.airframe import Part, author_parts
    from irsim_isaac.patch_stage import (
        author_patch_prims,
        enu_to_stage,
        patch_bindings,
        patch_prims,
        to_stage_frame,
        with_fallback_node,
    )
    from irsim_isaac.pipeline.ir_camera import IrCamera
    from irsim_isaac.pipeline.materials_usd import prim_records
    from irsim_isaac.pipeline.point_bridge import SurfaceBinding
    from irsim_isaac.stage import DOME_HEIGHT, author_environment, bind_visible_look
    from irsim_isaac.visible_sky import dome_spec_from_scene

    sensor = load_sensor_config(args.sensor)
    spec = sensor.sensor
    band = spec.band.band_id
    out_dir = pathlib.Path(args.out or f"outputs/{scene_path.stem}_{band}")
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    lut = load_band_lut_for_config(sensor, REPO / "data" / "lut")
    response = load_band_response_for_config(sensor, REPO / "data")
    skylight = skylight_for_sensor(sensor, spec.quantity)

    config = with_fallback_node(to_stage_frame(load_scene_config(scene_path)))
    scene = Scene.from_config(
        config,
        {band: lut},
        responses={band: response},
        quantity=spec.quantity,
        skylights={band: skylight},
    )
    prims = patch_prims(config)
    films = [p.path for p in prims if p.companion_only]
    print(
        f"{config.scene.name}: {len(prims) - len(films)} patch prims, {len(films)} film sheet(s), "
        f"{len(scene.surface_fields)} solved surfaces"
    )

    # --- the stage ------------------------------------------------------------------------------
    ctx = omni.usd.get_context()
    ctx.new_stage()
    stage = ctx.get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    stage.DefinePrim("/World", "Xform")
    stage.DefinePrim("/World/Looks", "Scope")
    # heading 0: the rotated config declares north on the stage's -Z, which is the heading the
    # dome and the sun light are aimed with -- one sun for the solver, the dome and the shadows.
    dome = None if args.no_rgb else dome_spec_from_scene(scene, heading_deg=0.0)
    author_environment(stage, dome, out_dir / "env_dome.exr", DOME_HEIGHT)
    prim_to_node = author_patch_prims(stage, prims)
    # Targets with no patch (PH.6's shell and tailpipe) get simple prims from the preset, in the
    # config's ENU and rotated like everything else: sizes swap Y and Z, a Y axis becomes Z.
    parts = []
    for name, kind, centre, size, material, node, axis in preset.get("parts", ()):
        parts.append(
            Part(
                name=name,
                kind=kind,
                centre_m=enu_to_stage(centre),
                size_m=(float(size[0]), float(size[2]), float(size[1])),
                material=material,
                thermal_node=node,
                axis={"X": "X", "Y": "Z", "Z": "Y"}[axis],
            )
        )
    if parts:
        prim_to_node.update(
            author_parts(stage, "/World/Props", tuple(parts), look_binder=bind_visible_look)
        )

    eye = np.asarray(enu_to_stage(args.eye), dtype=np.float64)
    look = np.asarray(enu_to_stage(args.look), dtype=np.float64)
    q = look_at_quaternion(look - eye)
    camera_path = "/World/IrCamera"
    cam = UsdGeom.Camera.Define(stage, camera_path)
    xform = UsdGeom.Xformable(cam)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(*(float(v) for v in eye)))
    xform.AddOrientOp().Set(Gf.Quatf(float(q[0]), Gf.Vec3f(*(float(v) for v in q[1:]))))
    cam.GetClippingRangeAttr().Set(Gf.Vec2f(0.05, 1.0e5))

    table = MaterialTable.from_library(MaterialLibrary.load(), band)
    resolver = MaterialResolver(load_mapping_rules(), list(table.names))
    records = [r for r in prim_records(root="/World") if r.path in prim_to_node]
    resolutions = resolver.resolve_all(records)
    unresolved = [r.path for r in resolutions if not r.mapped]
    if unresolved:
        print(f"unmapped prims: {unresolved}", file=sys.stderr)
        return 1

    try:
        pipeline = PipelineConfig.from_sensor(
            sensor,
            table,
            lut,
            sky=scene.sky_models[band],
            atmosphere=scene.layered,
            flat_field_enabled=True,
        )
    except ValueError as exc:
        print(f"{spec.name}: no flat field -- {exc}", file=sys.stderr)
        pipeline = PipelineConfig.from_sensor(
            sensor,
            table,
            lut,
            sky=scene.sky_models[band],
            atmosphere=scene.layered,
            flat_field_enabled=False,
        )
    if not args.no_chain and pipeline.calibration is not None:
        from irsim.pipeline.sensor_chain import attach_sensor_chain

        pipeline = attach_sensor_chain(pipeline, scene.weather, t0_s=scene.t0_s)

    bindings = [SurfaceBinding(path, fld) for path, fld in patch_bindings(scene)]
    camera = IrCamera(
        sensor,
        scene,
        pipeline=pipeline,
        prim_to_target=prim_to_node,
        resolutions=resolutions,
        camera_path=camera_path,
        surface_fields=bindings,
        capture_rgb=not args.no_rgb,
        heading_deg=0.0,
        strict_materials=True,
        strict_thermal_nodes=True,
        strict_patch_coverage=True,
        frame_period_s=float(args.interval_s),
        companion_only_prim_paths=films,
    ).open(settle_frames=args.settle)

    # --- one fixed span over the whole run, from the solved cells --------------------------------
    span_s = args.frames * args.interval_s
    probe = Scene.from_config(
        config, {band: lut}, responses={band: response}, quantity=spec.quantity
    )
    lo_k, hi_k = np.inf, -np.inf
    for t_rel in np.linspace(0.0, span_s, 16):
        for fld in probe.surface_fields.values():
            fld.advance_to(probe.t0_s + float(t_rel))
            cells = np.asarray(fld.temperature_at(probe.t0_s + float(t_rel)))
            lo_k, hi_k = min(lo_k, float(cells.min())), max(hi_k, float(cells.max()))
    if args.span_c:
        lo_k, hi_k = (float(v) + 273.15 for v in args.span_c)
    elif np.isfinite(lo_k):
        lo_k, hi_k = lo_k - 1.0, hi_k + 1.0
    # else: no solved surfaces (a plume scene): the span is taken off the first frame below
    print(f"display: fixed {lo_k - 273.15:.1f} .. {hi_k - 273.15:.1f} C, white-hot")
    gray = palette_table("gray")

    writer = FrameWriter(
        directory=out_dir,
        config_hash=config_hash(sensor),
        band_hash=band_hash(sensor),
        quantity=pipeline.quantity,
        float_format=args.float_format,
        start_utc=scene.weather.start_utc,
        stride=args.plane_stride,
        metadata={"scene": scene_path.name, "sensor": spec.name, "interval_s": args.interval_s},
    )

    # the readout quotes the largest surfaces, which are what a viewer finds in the frame
    sizes = {
        n: p.n_u * p.n_v * p.du_m * p.dv_m
        for n, p in scene.patches.items()
        if n in scene.surface_fields
    }
    gauges = sorted(sizes, key=lambda n: -sizes[n])[:5]
    local_offset_h = 2.0  # the committed weather files are local = UTC+2
    history: list[dict[str, Any]] = []
    t_render = time.time()
    for index in range(args.frames):
        t_rel = camera.t_rel_s
        outputs = camera.get_outputs(rt_subframes=args.rt_subframes)
        t_abs = scene.t0_s + t_rel
        cells = {
            n: np.asarray(f.temperature_at(t_abs), dtype=np.float64)
            for n, f in scene.surface_fields.items()
        }
        row: dict[str, Any] = {
            "frame": index,
            "t_rel_s": round(t_rel, 1),
            **{f"{n}_mean_k": round(float(c.mean()), 3) for n, c in cells.items()},
            **{f"{n}_span_k": round(float(c.max() - c.min()), 3) for n, c in cells.items()},
            "coverage_px": {k: int(v) for k, v in camera.patch_coverage.items()},
        }
        if outputs.apparent_t is not None:
            t_app = np.asarray(outputs.apparent_t)
            row["t_app_p1_k"] = round(float(np.nanpercentile(t_app, 1)), 3)
            row["t_app_p99_k"] = round(float(np.nanpercentile(t_app, 99)), 3)
        history.append(row)

        if writer.wants(index):
            truth = camera.truth()
            extra = {} if truth is None else dict(truth.planes)
            rgb = None if camera.last_frame is None else camera.last_frame.rgb
            if rgb is not None:
                extra["rgb"] = rgb
            writer.write(
                outputs,
                frame_index=index,
                t_s=camera.last_frame_t_s,
                extra_planes=extra,
                legends=None if truth is None else truth.legends,
                extra_metadata={"surface_mean_k": {n: float(c.mean()) for n, c in cells.items()}},
            )

        when = config.scene.start_utc + dt.timedelta(seconds=t_rel, hours=local_offset_h)
        lines = [
            f"{config.scene.name}   {when:%H:%M} local   T+{int(t_rel) // 60:3d} min",
            f"1 frame / {args.interval_s:g} s   solved per cell, {len(scene.surface_fields)} "
            "surfaces",
        ]
        values = {f"{n[:9]:9s}": float(cells[n].mean()) for n in gauges}
        for node in dict.fromkeys(prim_to_node.values()):
            if node in scene.targets and node != "patch_fallback":
                values[f"{node[:9]:9s}"] = float(camera.bridge.temperatures()[node])
        plane = np.asarray(outputs.apparent_t, dtype=np.float64)
        if not np.isfinite(lo_k):
            lo_k = float(np.nanpercentile(plane, 0.5)) - 1.0
            hi_k = float(np.nanpercentile(plane, 99.9)) + 1.0
            print(f"display: fixed {lo_k - 273.15:.1f} .. {hi_k - 273.15:.1f} C from frame 0")
        ir = gray[quantise_display((plane - lo_k) / max(hi_k - lo_k, 1e-6))]
        ir_frame = overlay_readout(
            with_margin(ir),
            [*lines, "LWIR apparent temperature, gray white-hot"],
            values,
            (lo_k, hi_k),
            palette=gray,
        )
        agc_frame = overlay_readout(
            with_margin(np.asarray(outputs.display8)),
            [*lines, f"camera {spec.isp.agc}"],
            values,
            (lo_k, hi_k),
        )
        write_png(frames_dir / f"ir_{index:05d}.png", ir_frame)
        write_png(frames_dir / f"agc_{index:05d}.png", agc_frame)
        if camera.last_frame is not None and camera.last_frame.rgb is not None:
            rgb = np.asarray(camera.last_frame.rgb, dtype=np.uint8)[..., :3]
            write_png(frames_dir / f"rgb_{index:05d}.png", np.ascontiguousarray(rgb))
            top = ir_frame.shape[0] - rgb.shape[0]
            padded = np.pad(rgb, ((top, 0), (0, 0), (0, 0))) if top > 0 else rgb
            write_png(
                frames_dir / f"pair_{index:05d}.png",
                np.ascontiguousarray(np.concatenate([ir_frame, padded], axis=1)),
            )
        if index % 8 == 0 or index == args.frames - 1:
            print(
                f"  frame {index:3d}  {when:%H:%M}  "
                + "  ".join(f"{n} {cells[n].mean() - 273.15:5.1f}C" for n in gauges[:4])
            )
    render_s = time.time() - t_render
    camera.close()

    videos: dict[str, str] = {}
    if ffmpeg_available():
        for kind in ("ir", "agc", "rgb", "pair"):
            if any(frames_dir.glob(f"{kind}_*.png")):
                videos[kind] = str(
                    encode_mp4(
                        str(frames_dir / f"{kind}_*.png"),
                        out_dir / f"{scene_path.stem}_{kind}.mp4",
                        fps=args.fps,
                    )
                )
        if not args.keep_frames:
            for png in frames_dir.glob("*.png"):
                png.unlink()
            frames_dir.rmdir()

    summary = {
        "boot_s": round(boot_s, 2),
        "render_s": round(render_s, 2),
        "scene": scene_path.name,
        "sensor": spec.name,
        "frames": args.frames,
        "interval_s": args.interval_s,
        "fps": args.fps,
        "eye_enu_m": list(args.eye),
        "look_enu_m": list(args.look),
        "display_span_k": [lo_k, hi_k],
        "film_sheets": films,
        "config_hash": config_hash(sensor),
        "videos": videos,
        "history": history,
    }
    path = out_dir / "summary.json"
    path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"\n{args.frames} frames in {render_s:.0f} s\n{videos}\nwrote {path}")
    return 0


try:
    status = main()
except Exception as exc:  # noqa: BLE001 - a demo crash should still report, not hang
    import traceback

    traceback.print_exc()
    print(f"render failed: {type(exc).__name__}: {exc}", file=sys.stderr)
    status = 1
finally:
    sys.stdout.flush()
    sys.stderr.flush()
    app.close(exit_code=status if isinstance(status, int) else 1)
