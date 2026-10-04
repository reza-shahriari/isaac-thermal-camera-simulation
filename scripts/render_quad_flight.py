#!/usr/bin/env python3
"""Roadmap M10.20: film a heavy-lift quadrotor flying a mission, in LWIR (ADR 0074).

One resolved multirotor at 20 m -- 105 px across the span, 6 px per motor bell -- against sky,
through the whole camera model, while ADR 0072's heat sources take the motors from ambient to
+45 K and back down over a 30-minute flight. The four hot bells, the warm speed controllers and
the warm battery are separate objects in the image, which is what a thermal sensor actually keys
on in a drone.

**This is a time-lapse, and that is a physics decision, not a shortcut.** The node law is
T = T_air(t) + dT_max u(t)^2 -- a steady-state relation with no thermal time constant (ADR 0072).
It is only defensible where the throttle moves slowly against a real motor's minutes-scale
response, so the flight is authored over 1800 s and the camera takes one frame every few seconds
of it. Every stage is told the truth about the interval, so the FFC fires on its real schedule and
the temporal noise decorrelates between frames exactly as it would in a real time-lapse. Running
the same profile at 60 Hz would look smoother and would be a lie about what metal can do.

Run with the Isaac interpreter (docs/decisions/0002), after `make luts`:
    python.sh scripts/render_quad_flight.py --frames 300 --rgb
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from typing import Any

REPO = pathlib.Path(__file__).resolve().parents[1]

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--out", default="outputs/quad_flight")
parser.add_argument("--frames", type=int, default=300)
parser.add_argument("--sensor", default=str(REPO / "configs/sensors/flir_boson_640_lwir.yaml"))
parser.add_argument("--scene", default=str(REPO / "configs/scenes/quad_flight_clear_noon.yaml"))
parser.add_argument(
    "--interval-s",
    type=float,
    default=6.0,
    help="scene seconds per captured frame. frames x interval must fit inside the flight profile",
)
parser.add_argument("--fps", type=float, default=30.0, help="playback rate of the encoded video")
parser.add_argument("--range-m", type=float, default=20.0)
parser.add_argument("--tilt-deg", type=float, default=15.0)
parser.add_argument("--heading-deg", type=float, default=110.0)
parser.add_argument("--settle", type=int, default=16)
parser.add_argument("--rt-subframes", type=int, default=8)
# IG.13: a driver that writes only 8-bit PNGs cannot support a radiometric claim (ADR 0068), and
# for most of this project's life this one wrote nothing else. The float32 planes and their
# sidecar are the *frame*; the video beside them is the look.
parser.add_argument("--float-format", default="npy", choices=("npy", "exr"))
parser.add_argument(
    "--plane-stride",
    type=int,
    default=1,
    help="write every Nth frame's float32 planes; 0 writes none, and makes no radiometric claim",
)
parser.add_argument("--rgb", action="store_true", help="also film the companion visible frame")
parser.add_argument("--no-chain", action="store_true", help="ideal camera: no M9 sensor chain")
parser.add_argument("--no-dome", action="store_true", help="untextured grey dome (ADR 0073)")
parser.add_argument(
    "--no-rotors",
    action="store_true",
    help="omit the propeller veils (ADR 0081), for the before/after comparison",
)
parser.add_argument(
    "--span-c",
    type=float,
    nargs=2,
    default=None,
    metavar=("LO", "HI"),
    help="fixed apparent-temperature span, Celsius. Default: spanned on the target nodes",
)
parser.add_argument(
    "--palette",
    default=None,
    help="palette for the fixed-span video. Default: whatever the sensor's ISP is configured with",
)
parser.add_argument(
    "--no-flat-field",
    action="store_true",
    help="skip the camera's flat-field correction, so cos^4 vignetting shows (ME.8 ablation)",
)
parser.add_argument(
    "--cloud-seed",
    type=int,
    default=None,
    help="structured cloud in BOTH bands, from the shared weather's cloud fraction (ADR 0076)",
)
parser.add_argument("--no-overlay", action="store_true", help="no burnt-in readout")
parser.add_argument("--keep-frames", action="store_true", help="keep the PNG sequence")
parser.add_argument(
    "--integration-ms",
    type=float,
    default=None,
    help="override a photon FPA's integration time, in ms. A camera has an exposure control and "
    "these configs carry one default each; a daylight reflective-band scene can saturate a "
    "low-light exposure by a hundred times. Changes the config hash, as it should -- it is a "
    "different configuration.",
)

args = parser.parse_args()

t_boot = time.time()
from isaacsim import SimulationApp  # noqa: E402

from irsim_isaac.env import simulation_app_config  # noqa: E402

app = SimulationApp(simulation_app_config())
boot_s = time.time() - t_boot


def throttle_at(scene: object, t_rel_s: float) -> float:
    """The flight profile's throttle at a scene-relative time, read back off the scene config."""
    import numpy as np

    for target in scene.spec.targets:  # type: ignore[attr-defined]
        if target.solver == "heat_source" and target.throttle_s:
            return float(np.interp(t_rel_s, target.throttle_s, target.throttle))
    return 0.0


def main() -> int:
    import numpy as np

    from irsim.atmosphere.cloud import generate_sky_cloud
    from irsim.atmosphere.skylight import skylight_for_sensor
    from irsim.config.loader import (
        band_hash,
        config_hash,
        load_sensor_config,
        with_integration_time_ms,
    )
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
    from irsim_eval.video import (
        encode_mp4,
        ffmpeg_available,
        overlay_readout,
    )
    from irsim_isaac.display_span import DisplaySpan, span_from_dn16, span_from_nodes
    from irsim_isaac.pipeline.illumination_isaac import SceneIllumination
    from irsim_isaac.pipeline.ir_camera import IrCamera
    from irsim_isaac.pipeline.materials_usd import prim_records
    from irsim_isaac.quad_flight import build_quad_flight, tracking_pose
    from irsim_isaac.visible_sky import dome_spec_from_scene

    out_dir = pathlib.Path(args.out)
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    sensor = load_sensor_config(args.sensor)
    if args.integration_ms is not None:
        try:
            sensor = with_integration_time_ms(sensor, args.integration_ms)
        except ValueError as exc:
            print(f"--integration-ms: {exc}", file=sys.stderr)
            return 1
    spec = sensor.sensor
    # ADR 0021: a photon FPA runs the whole chain on the photon table, and the sky model, the
    # atmosphere and the target solvers inside the Scene have to be built in the same form. Asked
    # of the sensor rather than assumed, because a mis-formed scene is out by ~1e19 and the AGC
    # hides it.
    quantity = sensor.sensor.quantity
    lut = load_band_lut_for_config(sensor, REPO / "data" / "lut")
    # The camera's own R(lambda), for the layered atmosphere's spectral-class split. It is
    # loaded beside the LUT because the two must describe one camera (AT.2).
    response = load_band_response_for_config(sensor, REPO / "data")
    # Scattered sunlight in the sky (M11.10, ADR 0086). `None` for an emissive band, so an LWIR
    # render is bit-identical to what it was; in a reflective band, without it the sky renders
    # black and the sunlit target sits on nothing, which is backwards.
    skylight = skylight_for_sensor(sensor, quantity)
    scene = Scene.from_file(
        args.scene,
        {spec.band.band_id: lut},
        responses={spec.band.band_id: response},
        quantity=quantity,
        skylights={spec.band.band_id: skylight},
    )

    span_s = args.frames * args.interval_s
    print(
        f"\n{spec.name}: {spec.fpa.width}x{spec.fpa.height}, "
        f"{args.frames} frames every {args.interval_s:g} s = {span_s:.0f} s of flight, "
        f"played at {args.fps:g} fps -> {args.frames / args.fps:.1f} s of video"
    )

    # One field, both bands: the infrared background samples it per ray and the dome bakes the
    # same object into its texture, so the pair cannot show cloud in different parts of the sky.
    cloud = None
    if args.cloud_seed is not None and scene.environment is not None:
        cloud = generate_sky_cloud(
            scene.environment.clouds.beta,
            float(scene.weather.at(scene.t0_s).cloud_fraction),
            int(args.cloud_seed),
        )
        print(
            f"cloud: seed {args.cloud_seed}, covering {cloud.fraction:.1%} of the sky "
            f"at beta = {cloud.beta} (from the shared weather)"
        )
    dome = None
    if not args.no_dome:
        dome = dome_spec_from_scene(scene, cloud=cloud, heading_deg=args.heading_deg)
        print(
            f"environment: sun {dome.sun_elevation_deg:.1f} deg elevation, "
            f"{dome.sun_azimuth_deg - args.heading_deg:+.1f} deg off boresight, "
            f"turbidity {dome.turbidity:.2f}"
        )
    stage = build_quad_flight(
        camera_tilt_deg=args.tilt_deg,
        range_m=args.range_m,
        dome=dome,
        dome_texture_path=out_dir / "env_dome.exr",
    )
    if stage.errors:
        print(f"stage errors: {stage.errors}", file=sys.stderr)

    ifov_mrad = 1e3 * spec.fpa.pitch_um * 1e-3 / spec.optics.focal_length_mm
    across = stage.pixels_across(ifov_mrad)
    print(
        f"at {stage.range_m:.0f} m: span "
        f"{1e3 * stage.spec.span_m / stage.range_m / ifov_mrad:.0f} px, "
        f"motor {across['motor_0']:.1f} px, esc {across['esc_0']:.1f} px, "
        f"battery {across['battery']:.1f} px"
    )
    missing = set(stage.thermal_nodes()) - set(scene.targets)
    if missing:
        print(f"scene has no solver for: {sorted(missing)}", file=sys.stderr)
        return 1

    table = MaterialTable.for_sensor(MaterialLibrary.load(), spec)
    resolver = MaterialResolver(load_mapping_rules(), list(table.names))
    resolutions = resolver.resolve_all(prim_records(root="/World/Targets"))
    unresolved = [r.path for r in resolutions if not r.mapped]
    if unresolved:
        print(f"unmapped prims: {unresolved}", file=sys.stderr)

    # Flat-field on, as on any real camera. `PipelineConfig` defaults it *off* -- the radiometric
    # branch divides cos^4 out analytically and does not need it -- but the display branch does,
    # and without it the 8-bit picture carries 21 % of vignetting from centre to corner that
    # plateau equalisation then stretches into dark corners (M9.12). No real thermal clip looks
    # like that.
    # **The flat field is refused when its hot calibration point is outside the ADC.**
    # `calibrate_flat_field` defaults to ADR 0021's -40..+200 C, which is a *bolometer*
    # range: the modelled InSb camera fills its well at 366 K, so a 473 K calibration point
    # drives it 16x over and the two-point fit becomes an extrapolation that comes back as
    # inverted vignetting. Falling back to no flat field, loudly, is better than a picture
    # with a calibration artefact in it that reads as a lens problem.
    try:
        pipeline = PipelineConfig.from_sensor(
            sensor,
            table,
            lut,
            sky=scene.sky_models[spec.band.band_id],
            atmosphere=scene.layered,
            flat_field_enabled=not args.no_flat_field,
        )
    except ValueError as exc:
        print(f"{spec.name}: no flat field -- {exc}", file=sys.stderr)
        pipeline = PipelineConfig.from_sensor(
            sensor,
            table,
            lut,
            sky=scene.sky_models[spec.band.band_id],
            atmosphere=scene.layered,
            flat_field_enabled=False,
        )

    # The M9 chain is a *bolometer* chain: its NUC residual is authored in mK/K and converted to
    # DN through the radiometric calibration, which a photon FPA has none of (ADR 0056, M11.6).
    # A photon camera in this repository is shutterless by configuration anyway, so there is no
    # FFC to model either. Skipping it is the honest behaviour and is announced; failing here
    # would make every reflective-band render impossible for a reason that is not about the band.
    if not args.no_chain:
        if pipeline.calibration is None:
            print(
                f"{spec.name}: no M9 sensor chain -- a photon FPA has no radiometric calibration "
                "to convert the NUC residual into DN (ADR 0056), and this camera is shutterless, "
                "so there is no FFC to freeze. Defects and 3-D noise still apply.",
                file=sys.stderr,
            )
        else:
            from irsim.pipeline.sensor_chain import attach_sensor_chain

            pipeline = attach_sensor_chain(pipeline, scene.weather, t0_s=scene.t0_s)

    camera = IrCamera(
        sensor,
        scene,
        pipeline=pipeline,
        prim_to_target=stage.prim_to_target,
        resolutions=resolutions,
        camera_path=stage.camera_path,
        # IG.6: the multirotor flies and its rotor discs are re-posed every frame from the
        # throttle, so the tracker reads each rendered leaf's own transform -- for the articulated
        # discs that is not the root's answer (ADR 0077, and `MotionTracker`'s class docstring).
        moving_prim_paths=[stage.quad_path],
        capture_rgb=args.rgb,
        illumination=SceneIllumination.for_camera(
            sensor, scene, pipeline.quantity, heading_deg=args.heading_deg
        ),
        heading_deg=args.heading_deg,
        strict_materials=False,
        strict_thermal_nodes=False,
        cloud_seed=args.cloud_seed,
        rotor_mounts=None if args.no_rotors else stage.rotor_mounts(0.0),
        # One capture every `interval_s` of scene time: a time-lapse camera, with every stage
        # told the truth about the gap (ADR 0074).
        frame_period_s=args.interval_s,
    ).open(settle_frames=args.settle)

    import omni.usd
    from pxr import Gf, UsdGeom

    usd_stage = omni.usd.get_context().get_stage()
    quad = UsdGeom.Xformable(usd_stage.GetPrimAtPath(stage.quad_path))
    translate_op, rotate_op = quad.GetOrderedXformOps()[:2]

    # **The main video is a fixed span, not the camera's AGC, and that is the whole point.**
    # Both of the §11.3 AGC modes rescale themselves from the current frame's own histogram, so a
    # target whose temperature is the subject comes out looking the same in every frame -- the
    # gain follows the target and cancels exactly the change being filmed. Worse, plateau
    # equalisation allocates display codes by *population*, and a target covering under 1 % of the
    # frame gets almost none of them: measured on this scene, 1217 of the object's pixels landed
    # in the top ten codes, so the whole airframe was one flat white shape. A real operator
    # switches to a manual span for this, and so do we.
    #
    # The span is taken from the **target's own nodes** over the whole sequence rather than from
    # ambient, because eight bits is 256 levels and spending half of them on sky-to-ambient leaves
    # every part of the target squeezed into the rest. Sky clips to black, deliberately: it is the
    # region with least to see in and it was costing the target all of its contrast.
    probe = Scene.from_file(
        args.scene,
        {spec.band.band_id: lut},
        responses={spec.band.band_id: response},
        quantity=quantity,
        skylights={spec.band.band_id: skylight},
    )
    try:
        node_samples = [probe.advance_targets(float(t), 0.0) for t in np.linspace(0.0, span_s, 64)]
    except ValueError as exc:
        # Better here than 200 frames into a render: the solvers refuse a time outside their
        # schedule rather than extrapolating one, so a capture window longer than the scene's
        # profile is a configuration error and this is the first place it can be seen.
        print(
            f"the capture window (frames x interval = {span_s:.0f} s) runs past the scene's "
            f"own schedule -- shorten it or extend the profile: {exc}",
            file=sys.stderr,
        )
        return 1
    # **In a reflective band there is no temperature to span.** §12.1 switches
    # `apparent_temperature` off for SWIR and NIR because inverting L_B does not give a scene
    # temperature when the signal is reflected sunlight, so the kelvin span above has no plane to
    # apply to. Those bands span the raw ADC instead, between percentiles of the first captured
    # frame, held fixed for the sequence -- a manual span, not a slow AGC (M10.23).
    emissive = spec.outputs.apparent_temperature
    span: DisplaySpan | None = None
    if args.span_c:
        if not emissive:
            print(
                "--span-c is a temperature span and this band has no apparent temperature; "
                "drop it and the render spans the ADC instead",
                file=sys.stderr,
            )
            return 1
        if float(args.span_c[1]) <= float(args.span_c[0]):
            print("--span-c must be increasing", file=sys.stderr)
            return 1
        span = DisplaySpan(
            kind="apparent_t",
            low=float(args.span_c[0]) + 273.15,
            high=float(args.span_c[1]) + 273.15,
        )
    elif emissive:
        span = span_from_nodes(node_samples)
    palette_name = args.palette or spec.isp.palette
    palette = palette_table(palette_name)
    if span is not None:
        print(
            f"display: fixed {span.caption}, {palette_name} palette (from the sensor ISP); "
            f"the camera's own {spec.isp.agc} output is filmed alongside"
        )
        for name, value in sorted(node_samples[-1].items()):
            print(
                f"    {name:10s} {value - 273.15:7.1f} C -> display code {span.code_for(value):3d}"
            )
    else:
        print(
            f"display: {palette_name} palette; the fixed span is taken from the first frame's ADC "
            "percentiles once it exists (this band has no apparent temperature to span)"
        )
    # IG.13: the float32 planes and their sidecar, bound once (ADR 0068). Every run-constant
    # field lives here so the loop body cannot quietly disagree with itself frame to frame.
    writer = FrameWriter(
        directory=out_dir,
        config_hash=config_hash(sensor),
        band_hash=band_hash(sensor),
        quantity=pipeline.quantity,
        float_format=args.float_format,
        start_utc=scene.weather.start_utc,
        stride=args.plane_stride,
        metadata={
            "scene": pathlib.Path(args.scene).name,
            "sensor": spec.name,
            "range_m": stage.range_m,
            "interval_s": args.interval_s,
        },
    )
    history: list[dict[str, float]] = []
    planes_written = 0
    t_render = time.time()
    for index in range(args.frames):
        t_rel = camera.t_rel_s
        throttle = throttle_at(scene, t_rel)
        translate, rotate = tracking_pose(
            t_rel, throttle, range_m=stage.range_m, camera_tilt_deg=stage.camera_tilt_deg
        )
        translate_op.Set(Gf.Vec3d(*translate))
        rotate_op.Set(Gf.Vec3f(*rotate))
        if not args.no_rotors:
            # The discs turn with the throttle the thermal model is reading, so the rotor a
            # viewer sees and the motor temperature they see come from one flight.
            camera.rotor_mounts.update(stage.rotor_mounts(throttle))

        outputs = camera.get_outputs(rt_subframes=args.rt_subframes)
        temps = camera.bridge.temperatures()
        if span is None:
            # First frame in a reflective band: the span is the picture's own ADC percentiles,
            # taken once and then held, so the video does not breathe.
            span = span_from_dn16(np.asarray(outputs.dn16))
            print(f"display: fixed {span.caption} from the first frame's ADC percentiles")
        record: dict[str, float] = {
            "frame": index,
            "t_rel_s": round(t_rel, 3),
            "throttle": round(throttle, 4),
            **{f"{k}_k": round(v, 3) for k, v in temps.items()},
        }
        if outputs.apparent_t is not None:
            t_app = np.asarray(outputs.apparent_t)
            record["t_app_min_k"] = round(float(t_app.min()), 3)
            record["t_app_max_k"] = round(float(t_app.max()), 3)
        if outputs.dn16 is not None:
            dn = np.asarray(outputs.dn16)
            record["dn16_min"] = float(dn.min())
            record["dn16_max"] = float(dn.max())
        history.append(record)

        # The frame itself. `wants` is asked first so the companion RGB is not copied out of the
        # renderer on a frame whose planes are not being kept.
        if writer.wants(index):
            # scene truth under each pixel centre, for the frame viewer (`python -m irsim_viewer`)
            truth = camera.truth()
            extra = {} if truth is None else dict(truth.planes)
            if args.rgb and camera.last_frame is not None and camera.last_frame.rgb is not None:
                extra["rgb"] = camera.last_frame.rgb
            writer.write(
                outputs,
                frame_index=index,
                t_s=camera.last_frame_t_s,
                extra_planes=extra,
                legends=None if truth is None else truth.legends,
                extra_metadata={
                    "throttle": round(throttle, 4),
                    "node_temperatures_k": {k: round(float(v), 3) for k, v in temps.items()},
                },
            )
            planes_written += 1

        # Fixed span -> palette. `quantise_display` is the ISP's own rounding, so the mapping is
        # the one the display branch uses and not a second, subtly different one.
        spanned = palette[quantise_display(span.scale(outputs))]
        caption_span = span.caption

        # Defaults bind this frame's values at definition time; a closure over the loop variables
        # would be evaluated later and is the classic way to caption every frame with the last
        # frame's numbers.
        bar = (span.low, span.high) if span.is_temperature else None

        def readout(
            image: Any,
            caption: str,
            t: float = t_rel,
            u: float = throttle,
            node: dict = temps,
            scale: Any = bar,
        ) -> Any:
            minutes, seconds = divmod(int(t), 60)
            return overlay_readout(
                image,
                [
                    f"T+{minutes:02d}:{seconds:02d}   throttle {u * 100:3.0f}%",
                    f"air {node['airframe'] - 273.15:5.1f}C   motor {node['motor'] - 273.15:5.1f}C",
                    f"1 frame / {args.interval_s:g} s time-lapse   {caption}",
                ],
                {
                    "motor  ": node["motor"],
                    "esc    ": node["esc"],
                    "battery": node["battery"],
                    "air    ": node["airframe"],
                },
                scale,
                bare=args.no_overlay,
            )

        write_png(
            frames_dir / f"ir_{index:05d}.png",
            readout(spanned, f"{caption_span} {palette_name}"),
        )
        write_png(
            frames_dir / f"agc_{index:05d}.png",
            readout(np.asarray(outputs.display8), f"camera {spec.isp.agc}"),
        )

        if args.rgb and camera.last_frame is not None and camera.last_frame.rgb is not None:
            write_png(
                frames_dir / f"rgb_{index:05d}.png",
                np.ascontiguousarray(np.asarray(camera.last_frame.rgb)[..., :3]),
            )
        if index % 25 == 0 or index == args.frames - 1:
            level = (
                f"T_app {record['t_app_min_k']:.1f}..{record['t_app_max_k']:.1f} K"
                if "t_app_min_k" in record
                else f"DN {record.get('dn16_min', 0):.0f}..{record.get('dn16_max', 0):.0f}"
            )
            print(
                f"  frame {index:4d}  T+{t_rel:7.1f}s  u={throttle:4.2f}  "
                f"motor {temps['motor'] - 273.15:5.1f}C  {level}"
            )
    render_s = time.time() - t_render
    camera.close()

    videos = {}
    if ffmpeg_available():
        videos["ir"] = str(
            encode_mp4(str(frames_dir / "ir_*.png"), out_dir / "quad_flight_ir.mp4", fps=args.fps)
        )
        videos["agc"] = str(
            encode_mp4(str(frames_dir / "agc_*.png"), out_dir / "quad_flight_agc.mp4", fps=args.fps)
        )
        if args.rgb and any(frames_dir.glob("rgb_*.png")):
            videos["rgb"] = str(
                encode_mp4(
                    str(frames_dir / "rgb_*.png"), out_dir / "quad_flight_rgb.mp4", fps=args.fps
                )
            )
        if not args.keep_frames:
            for png in frames_dir.glob("*.png"):
                png.unlink()
            frames_dir.rmdir()
    else:
        print("ffmpeg not found: keeping the PNG sequence", file=sys.stderr)

    peak = max(history, key=lambda row: row["motor_k"])
    summary = {
        "boot_s": round(boot_s, 2),
        "render_s": round(render_s, 2),
        "frames": args.frames,
        "interval_s": args.interval_s,
        "flight_span_s": span_s,
        "fps": args.fps,
        "sensor": spec.name,
        "scene": pathlib.Path(args.scene).name,
        "config_hash": config_hash(sensor),
        "band_hash": band_hash(sensor),
        "range_m": stage.range_m,
        "ifov_mrad": round(ifov_mrad, 4),
        "pixels_across": {k: round(v, 2) for k, v in across.items()},
        "motor_peak_k": peak["motor_k"],
        "motor_peak_at_s": peak["t_rel_s"],
        "motor_swing_k": round(peak["motor_k"] - min(r["motor_k"] for r in history), 3),
        "display_span": {"kind": span.kind, "low": span.low, "high": span.high},
        "float_format": args.float_format,
        "plane_stride": args.plane_stride,
        "plane_frames": planes_written,
        "palette": palette_name,
        "videos": videos,
        "history": history,
    }
    path = out_dir / "summary.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)
    print(
        f"\n{args.frames} frames in {render_s:.0f} s. motor peaked at "
        f"{peak['motor_k'] - 273.15:.1f} C at T+{peak['t_rel_s']:.0f} s, "
        f"swing {summary['motor_swing_k']:.1f} K\n{videos}\nwrote {path}"
    )
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
