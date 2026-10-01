#!/usr/bin/env python3
"""Roadmap M10.21: film a light jet flying a low pass, in LWIR (ADR 0075).

The companion to `render_quad_flight.py`, and the physics it shows is the opposite shape. A
multirotor's heat is four motors a sixteenth of its span across, which you resolve and watch warm
over half an hour. A jet's heat is an exhaust nozzle a *fortieth* of its span across but hundreds
of kelvin above ambient, and a skin warmed nearly uniformly by aerodynamic heating. So nothing
about the target changes during a pass -- what changes is **aspect**: the nozzles are hidden behind
their own nacelles from the front and fully exposed from the rear, which is why an aircraft's
infrared signature varies by a large factor around the clock.

**Real time, not a time-lapse.** ADR 0074 filmed the quadrotor as a time-lapse because the process
was thermal and slow. This process is geometric and fast: a ten-second pass captured at 30 Hz and
played at 30 fps, with no speed-up to declare. Range sweeps 790 -> 250 -> 790 m from the track
alone, so the same run exercises the inverse-square fall-off and the atmospheric path against a
target of known size and known temperature.

The mount tracks the aircraft. At 150 m/s and 250 m range it crosses a 33 degree field in well
under a second, and a stabilised line of sight is also what keeps the *target* still on the focal
plane -- the smear goes onto the featureless sky behind it instead.

Run with the Isaac interpreter (docs/decisions/0002), after `make luts`:
    python.sh scripts/render_aircraft_pass.py --frames 300 --rgb
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
parser.add_argument("--out", default="outputs/aircraft_pass")
parser.add_argument("--frames", type=int, default=300)
parser.add_argument("--sensor", default=str(REPO / "configs/sensors/flir_boson_640_lwir.yaml"))
parser.add_argument("--scene", default=str(REPO / "configs/scenes/aircraft_pass_clear_noon.yaml"))
parser.add_argument(
    "--interval-s",
    type=float,
    default=1.0 / 30.0,
    help="scene seconds per captured frame. 1/30 with --fps 30 is real time",
)
parser.add_argument("--fps", type=float, default=30.0, help="playback rate of the encoded video")
parser.add_argument("--speed-m-s", type=float, default=150.0, help="true airspeed along the track")
parser.add_argument("--altitude-m", type=float, default=150.0)
parser.add_argument("--offset-m", type=float, default=200.0, help="how far ahead the track crosses")
parser.add_argument("--tilt-deg", type=float, default=25.0)
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
    "--span-c",
    type=float,
    nargs=2,
    default=None,
    metavar=("LO", "HI"),
    help="fixed apparent-temperature span, Celsius. Default: spanned on the target nodes",
)
parser.add_argument("--palette", default=None, help="default: the sensor ISP's own palette")
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


def main() -> int:
    import numpy as np
    import omni.usd
    from pxr import Gf, UsdGeom

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
    from irsim_isaac.aircraft_pass import (
        PassTrack,
        build_aircraft_pass,
        look_at_quaternion,
    )
    from irsim_isaac.display_span import DisplaySpan, span_from_dn16, span_from_nodes
    from irsim_isaac.pipeline.illumination_isaac import SceneIllumination
    from irsim_isaac.pipeline.ir_camera import IrCamera
    from irsim_isaac.pipeline.materials_usd import prim_records
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
    track = PassTrack(
        speed_m_s=args.speed_m_s,
        altitude_m=args.altitude_m,
        offset_m=args.offset_m,
        cpa_time_s=0.5 * span_s,
    )
    print(
        f"\n{spec.name}: {spec.fpa.width}x{spec.fpa.height}, {args.frames} frames every "
        f"{args.interval_s * 1e3:.1f} ms = {span_s:.1f} s of flight, played at {args.fps:g} fps "
        f"-> {args.frames / args.fps:.1f} s of video "
        f"({'real time' if abs(args.interval_s * args.fps - 1.0) < 1e-6 else 'not real time'})"
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
    stage = build_aircraft_pass(
        camera_tilt_deg=args.tilt_deg,
        track=track,
        dome=dome,
        dome_texture_path=out_dir / "env_dome.exr",
    )
    if stage.errors:
        print(f"stage errors: {stage.errors}", file=sys.stderr)

    ifov_mrad = 1e3 * spec.fpa.pitch_um * 1e-3 / spec.optics.focal_length_mm
    first, cpa, last = (track.sample(t) for t in (0.0, track.cpa_time_s, span_s))
    print(
        f"pass: range {first.range_m:.0f} -> {cpa.range_m:.0f} -> {last.range_m:.0f} m, "
        f"aspect {first.aspect_deg:.0f} -> {cpa.aspect_deg:.0f} -> {last.aspect_deg:.0f} deg; "
        f"span {cpa.pixels_across(stage.spec.span_m, ifov_mrad):.0f} px and nozzle "
        f"{cpa.pixels_across(stage.spec.nozzle_diameter_m, ifov_mrad):.1f} px at closest approach"
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
        # IG.6: the aircraft flies and the pedestal slews, so both halves of the relative motion
        # are real. The camera is tracked unconditionally; this names the target.
        moving_prim_paths=[stage.aircraft_path],
        capture_rgb=args.rgb,
        illumination=SceneIllumination.for_camera(
            sensor, scene, pipeline.quantity, heading_deg=args.heading_deg
        ),
        heading_deg=args.heading_deg,
        strict_materials=False,
        strict_thermal_nodes=False,
        cloud_seed=args.cloud_seed,
        frame_period_s=args.interval_s,
    ).open(settle_frames=args.settle)

    usd_stage = omni.usd.get_context().get_stage()
    aircraft = UsdGeom.Xformable(usd_stage.GetPrimAtPath(stage.aircraft_path))
    translate_op = aircraft.GetOrderedXformOps()[0]
    camera_aim_op = UsdGeom.Xformable(
        usd_stage.GetPrimAtPath(stage.camera_path)
    ).GetOrderedXformOps()[0]

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
    # **In a reflective band there is no temperature to span** (M10.23): §12.1 switches
    # `apparent_temperature` off for SWIR and NIR, so the kelvin span has no plane to apply to and
    # those bands span the raw ADC between percentiles of the first captured frame instead.
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
    rate_hz = 1.0 / args.interval_s
    capture_label = (
        f"{rate_hz:.0f} Hz capture" if rate_hz >= 1.0 else f"1 frame / {args.interval_s:g} s"
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
            "speed_m_s": args.speed_m_s,
            "altitude_m": args.altitude_m,
        },
    )
    history: list[dict[str, float]] = []
    planes_written = 0
    t_render = time.time()
    for index in range(args.frames):
        t_rel = camera.t_rel_s
        sample = track.sample(t_rel)
        # The aircraft flies its real track and the pedestal slews onto it. `refresh_pose` is not
        # optional: the pose behind every pixel's ray is cached, so without it the geometry would
        # follow the new aim while the elevations and the sky temperature stayed at the old one.
        translate_op.Set(Gf.Vec3d(*(float(v) for v in sample.position_m)))
        aim = look_at_quaternion(sample.position_m)
        camera_aim_op.Set(Gf.Quatf(float(aim[0]), Gf.Vec3f(*(float(v) for v in aim[1:]))))
        camera.refresh_pose()

        outputs = camera.get_outputs(rt_subframes=args.rt_subframes)
        temps = camera.bridge.temperatures()
        if span is None:
            span = span_from_dn16(np.asarray(outputs.dn16))
            print(f"display: fixed {span.caption} from the first frame's ADC percentiles")

        # How much of the frame each node actually occupies, straight off the id plane: the only
        # honest way to say whether the nozzle is in view, since the renderer decides occlusion.
        last = camera.last_frame
        by_node: dict[str, int] = {}
        if last is not None:
            for ident, path in last.labels.items():
                node = stage.prim_to_target.get(str(path))
                if node is not None:
                    by_node[node] = by_node.get(node, 0) + int((last.instance_id == ident).sum())
        history.append(
            {
                "frame": index,
                "t_rel_s": round(t_rel, 4),
                "range_m": round(sample.range_m, 2),
                "aspect_deg": round(sample.aspect_deg, 2),
                "elevation_deg": round(
                    float(np.degrees(np.arcsin(sample.position_m[1] / sample.range_m))), 2
                ),
                "nozzle_face_expected": sample.shows_nozzle_face(),
                "nozzle_px": by_node.get("nozzle", 0),
                "skin_px": by_node.get("skin", 0),
                **{f"{k}_k": round(v, 3) for k, v in temps.items()},
                **(
                    {"t_app_max_k": round(float(np.asarray(outputs.apparent_t).max()), 3)}
                    if outputs.apparent_t is not None
                    else {"dn16_max": float(np.asarray(outputs.dn16).max())}
                ),
            }
        )

        # The frame itself. `wants` is asked first so the companion RGB is not copied out of the
        # renderer on a frame whose planes are not being kept.
        if writer.wants(index):
            # scene truth under each pixel centre, for the frame viewer (`python -m irsim_viewer`)
            truth = camera.truth()
            extra = {} if truth is None else dict(truth.planes)
            if args.rgb and last is not None and last.rgb is not None:
                extra["rgb"] = last.rgb
            writer.write(
                outputs,
                frame_index=index,
                t_s=camera.last_frame_t_s,
                extra_planes=extra,
                legends=None if truth is None else truth.legends,
                extra_metadata={
                    "range_m": round(sample.range_m, 3),
                    "aspect_deg": round(sample.aspect_deg, 3),
                    "node_temperatures_k": {k: round(float(v), 3) for k, v in temps.items()},
                },
            )
            planes_written += 1

        spanned = palette[quantise_display(span.scale(outputs))]
        caption_span = span.caption
        bar = (span.low, span.high) if span.is_temperature else None

        # The nozzle line reports the **measured** pixel count off the id plane, not a predicate.
        # "Occluded" would be wrong at the beam: side-on you see the nozzle's cylindrical wall
        # even though its hot aft face is edge-on, and the count is what actually varies.
        def readout(
            image: Any,
            caption: str,
            s: Any = sample,
            node: dict = temps,
            px: int = 0,
            scale: Any = bar,
        ) -> Any:
            return overlay_readout(
                image,
                [
                    f"T+{s.t_s:04.1f}s   range {s.range_m:6.0f} m   aspect {s.aspect_deg:5.1f} deg",
                    f"skin {node['skin'] - 273.15:5.1f}C   nozzle {node['nozzle'] - 273.15:5.0f}C"
                    f"   nozzle seen {px:3d} px",
                    f"{capture_label}   {caption}",
                ],
                {
                    "nozzle ": node["nozzle"],
                    "nacelle": node["nacelle"],
                    "skin   ": node["skin"],
                },
                scale,
                bare=args.no_overlay,
            )

        write_png(
            frames_dir / f"ir_{index:05d}.png",
            readout(
                spanned,
                f"{caption_span} {palette_name}",
                px=by_node.get("nozzle", 0),
            ),
        )
        write_png(
            frames_dir / f"agc_{index:05d}.png",
            readout(
                np.asarray(outputs.display8),
                f"camera {spec.isp.agc}",
                px=by_node.get("nozzle", 0),
            ),
        )
        if args.rgb and last is not None and last.rgb is not None:
            write_png(
                frames_dir / f"rgb_{index:05d}.png",
                np.ascontiguousarray(np.asarray(last.rgb)[..., :3]),
            )
        if index % 25 == 0 or index == args.frames - 1:
            print(
                f"  frame {index:4d}  T+{t_rel:5.2f}s  R={sample.range_m:6.0f} m  "
                f"aspect={sample.aspect_deg:5.1f} deg  nozzle={by_node.get('nozzle', 0):4d} px  "
                f"level {history[-1].get('t_app_max_k', history[-1].get('dn16_max', 0)):.1f}"
            )
    render_s = time.time() - t_render
    camera.close()

    videos = {}
    if ffmpeg_available():
        for tag, name in (("ir", "aircraft_pass_ir"), ("agc", "aircraft_pass_agc")):
            videos[tag] = str(
                encode_mp4(str(frames_dir / f"{tag}_*.png"), out_dir / f"{name}.mp4", fps=args.fps)
            )
        if args.rgb and any(frames_dir.glob("rgb_*.png")):
            videos["rgb"] = str(
                encode_mp4(
                    str(frames_dir / "rgb_*.png"), out_dir / "aircraft_pass_rgb.mp4", fps=args.fps
                )
            )
        if not args.keep_frames:
            for png in frames_dir.glob("*.png"):
                png.unlink()
            frames_dir.rmdir()
    else:
        print("ffmpeg not found: keeping the PNG sequence", file=sys.stderr)

    head_on = [r for r in history if r["aspect_deg"] < 60.0]
    tail_on = [r for r in history if r["aspect_deg"] > 120.0]
    summary = {
        "boot_s": round(boot_s, 2),
        "render_s": round(render_s, 2),
        "frames": args.frames,
        "interval_s": args.interval_s,
        "fps": args.fps,
        "real_time": abs(args.interval_s * args.fps - 1.0) < 1e-6,
        "sensor": spec.name,
        "scene": pathlib.Path(args.scene).name,
        "config_hash": config_hash(sensor),
        "band_hash": band_hash(sensor),
        "track": {
            "speed_m_s": track.speed_m_s,
            "altitude_m": track.altitude_m,
            "offset_m": track.offset_m,
            "cpa_range_m": round(track.cpa_range_m(), 2),
        },
        "ifov_mrad": round(ifov_mrad, 4),
        "display_span": {"kind": span.kind, "low": span.low, "high": span.high},
        "float_format": args.float_format,
        "plane_stride": args.plane_stride,
        "plane_frames": planes_written,
        "palette": palette_name,
        "nozzle_px_head_on": max((r["nozzle_px"] for r in head_on), default=0),
        "nozzle_px_tail_on": max((r["nozzle_px"] for r in tail_on), default=0),
        "history": history,
    }
    path = out_dir / "summary.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)
    print(
        f"\n{args.frames} frames in {render_s:.0f} s. nozzle pixels: "
        f"{summary['nozzle_px_head_on']} at head aspect, {summary['nozzle_px_tail_on']} at tail "
        f"aspect\n{videos}\nwrote {path}"
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
