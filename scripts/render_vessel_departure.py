#!/usr/bin/env python3
"""Roadmap MM.7: film a vessel getting under way and steaming out to the horizon, in LWIR.

The maritime counterpart of `render_aircraft_pass.py`, and the film the static maritime demo
cannot make. Two things happen at once and they pull in opposite directions:

* the vessel **recedes**, so its hull shrinks from a couple of hundred pixels to a handful, its
  funnel goes sub-pixel within a few minutes, and the atmosphere puts more and more warm air
  between it and the camera;
* its **engine warms**, from a cold uptake alongside to cruise power, climbing 150 K over a few
  minutes with the funnel's own time constant.

So the target's signature gets *stronger* while the target gets *smaller* — which is the regime an
infrared search set actually works in, and one no static frame shows.

The camera does not track it. Against a fixed boresight the vessel climbs through the frame as its
depression angle shrinks toward the horizon, crossing the sea's own angular gradient on the way
(ADR 0078): it starts against near water reading close to the SST and ends against far water that
is mostly warm air. A tracking mount would hold it still and throw that away.

The sea itself is analytic, as everywhere in this lane: the water mesh is visible-band geometry
that occludes and sets the horizon, while every pixel of it takes its apparent temperature from
`SeaModel` at that ray's own depression angle.

Run with the Isaac interpreter (docs/decisions/0002), after `make luts`:
    python.sh scripts/render_vessel_departure.py --frames 240 --rgb --out outputs/vessel_departure
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import time
from typing import Any

REPO = pathlib.Path(__file__).resolve().parents[1]

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--out", default="outputs/vessel_departure")
parser.add_argument("--frames", type=int, default=160)
parser.add_argument("--sensor", default=str(REPO / "configs/sensors/flir_boson_640_lwir.yaml"))
parser.add_argument("--scene", default=str(REPO / "configs/scenes/vessel_departure_clear_day.yaml"))
parser.add_argument(
    "--interval-s",
    type=float,
    default=6.0,
    help="scene seconds per captured frame. A departure takes minutes, so this is a time lapse: "
    "6 s per frame at 24 fps plays 16 minutes of sea time in under 7 seconds",
)
parser.add_argument("--fps", type=float, default=24.0, help="playback rate of the encoded video")
parser.add_argument("--speed-m-s", type=float, default=6.0, help="the vessel's speed (11.7 knots)")
parser.add_argument("--start-range-m", type=float, default=250.0)
parser.add_argument(
    "--length-m",
    type=float,
    default=90.0,
    help="length overall. A coaster rather than a launch, so its funnel (7 % of LOA) is "
    "still a pixel or so across at 6 km -- the engine signature is the point of the film",
)
parser.add_argument("--offset-m", type=float, default=0.0, help="lateral offset of the track")
parser.add_argument("--camera-height-m", type=float, default=20.0)
parser.add_argument("--tilt-deg", type=float, default=-3.0, help="negative looks DOWN at the sea")
parser.add_argument("--heading-deg", type=float, default=120.0, help="compass bearing of the view")
parser.add_argument("--settle", type=int, default=16)
parser.add_argument("--rt-subframes", type=int, default=8)
# IG.13: a driver that writes only 8-bit PNGs cannot support a radiometric claim (ADR 0068), and
# for most of this project's life this one wrote nothing else. The float32 planes and their
# sidecar are the *frame*; the video beside them is the look.
parser.add_argument("--float-format", default="npy", choices=("npy", "exr"))
parser.add_argument(
    "--integration-ms",
    type=float,
    default=None,
    help="override a photon FPA's integration time, in ms. A camera has an exposure control and "
    "these configs carry one default each; a daylight reflective-band scene can saturate a "
    "low-light exposure by a hundred times. Changes the config hash, as it should -- it is a "
    "different camera",
)
parser.add_argument(
    "--plane-stride",
    type=int,
    default=1,
    help="write every Nth frame's float32 planes; 0 writes none, and makes no radiometric claim",
)
parser.add_argument("--rgb", action="store_true", help="also film the companion visible frame")
parser.add_argument("--no-chain", action="store_true", help="ideal camera: no M9 sensor chain")
parser.add_argument("--no-water", action="store_true", help="no water geometry (ADR 0078 ablation)")
parser.add_argument("--no-dome", action="store_true", help="untextured grey dome (ADR 0073)")
parser.add_argument("--no-flat-field", action="store_true", help="skip the flat-field correction")
parser.add_argument("--cloud-seed", type=int, default=None, help="structured cloud (ADR 0076)")
parser.add_argument("--water-rings", type=int, default=420)
parser.add_argument("--water-sectors", type=int, default=448)
parser.add_argument("--water-half-angle-deg", type=float, default=26.0)
parser.add_argument(
    "--span-c",
    type=float,
    nargs=2,
    default=None,
    metavar=("LO", "HI"),
    help="fixed apparent-temperature span, Celsius. Default: spanned on the target nodes",
)
parser.add_argument("--palette", default=None, help="default: the sensor ISP's own palette")
parser.add_argument("--no-overlay", action="store_true", help="no burnt-in readout")
parser.add_argument("--keep-frames", action="store_true", help="keep the PNG sequence")
args = parser.parse_args()

t_boot = time.time()
from isaacsim import SimulationApp  # noqa: E402

from irsim_isaac.env import simulation_app_config  # noqa: E402

app = SimulationApp(simulation_app_config())
boot_s = time.time() - t_boot


def main() -> int:
    import numpy as np
    from pxr import Gf, UsdGeom

    from irsim.atmosphere.sea import SeaModel
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
    from irsim.materials.nk import load_nk_table
    from irsim.materials.table import MaterialTable
    from irsim.pipeline.core import PipelineConfig
    from irsim.radiometry.lut_files import (
        load_band_lut_for_config,
        load_band_response_for_config,
    )
    from irsim.radiometry.spectral_response import load_spectral_response
    from irsim.scene import Scene
    from irsim_eval.video import (
        encode_mp4,
        ffmpeg_available,
        overlay_readout,
        target_span_k,
    )
    from irsim_isaac.maritime_demo import DepartureTrack, build_maritime_demo
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
    lut = load_band_lut_for_config(sensor, REPO / "data" / "lut")
    # The camera's own R(lambda), for the layered atmosphere's spectral-class split. It is
    # loaded beside the LUT because the two must describe one camera (AT.2).
    response = load_band_response_for_config(sensor, REPO / "data")
    # Which table this camera runs on -- `lb` for a bolometer, `lb_q` for a photon FPA. The sky
    # model must be built in the same form the pipeline reads it in, or `PipelineConfig`
    # refuses the pair; without this line the scene was always built in `lb` and the driver
    # ran in its own default band only.
    quantity = spec.quantity
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

    environment = scene.environment
    if environment is None or environment.ground.mode != "sea":
        print(f"{args.scene} needs a sea environment preset (ground.mode 'sea')", file=sys.stderr)
        return 2
    bulk_sst_k = float(environment.ground.bulk_sst_k or 0.0)

    span_s = args.frames * args.interval_s
    track = DepartureTrack(
        start_range_m=args.start_range_m,
        speed_m_s=args.speed_m_s,
        camera_height_m=args.camera_height_m,
        offset_m=args.offset_m,
    )

    sky = scene.sky_models[spec.band.band_id]
    sea = SeaModel(
        sky,
        load_nk_table("water"),
        load_spectral_response(str(spec.band.spectral_response)),
        bulk_sst_k=bulk_sst_k,
        camera_height_m=args.camera_height_m,
    )
    wind_m_s = float(scene.weather.at(scene.t0_s).wind_speed_m_s)
    ifov_mrad = 1e3 * spec.fpa.pitch_um * 1e-3 / spec.optics.focal_length_mm

    print(
        f"\ndeparture: {args.length_m:.0f} m vessel from {track.start_range_m:.0f} m at "
        f"{track.speed_m_s:.1f} m/s, {args.frames} frames x {args.interval_s:g} s = "
        f"{span_s / 60.0:.1f} min of sea time, played at {args.fps:g} fps "
        f"({span_s / args.fps / args.interval_s:.0f}x real time)"
    )
    print(
        f"sea: bulk SST {bulk_sst_k:.2f} K, camera {args.camera_height_m:.0f} m up, horizon "
        f"{math.degrees(sea.horizon_rad):.4f} deg down; wind {wind_m_s:.1f} m/s -> rms facet tilt "
        f"{math.degrees(sea.tilt_sigma(scene.t0_s)):.2f} deg"
    )
    for t in (0.0, span_s * 0.25, span_s * 0.5, span_s):
        print(
            f"  T+{t / 60.0:5.1f} min  R={track.range_m(t):6.0f} m  "
            f"depression {track.depression_deg(t):6.3f} deg  "
            f"hull {track.pixels_across(t, args.length_m, ifov_mrad):6.1f} px  "
            f"stack {track.pixels_across(t, 0.07 * args.length_m, ifov_mrad):5.2f} px"
        )

    dome = None
    if not args.no_dome:
        dome = dome_spec_from_scene(
            scene, heading_deg=args.heading_deg, camera_height_m=args.camera_height_m
        )
        dome = type(dome)(**{**dome.__dict__, "ground_albedo": (0.020, 0.035, 0.050)})

    stage = build_maritime_demo(
        camera_tilt_deg=args.tilt_deg,
        camera_height_m=args.camera_height_m,
        vessels=(
            (
                "vessel",
                track.start_range_m,
                args.length_m,
                args.offset_m,
                "car_paint_white",
                "hull",
            ),
        ),
        dome=dome,
        dome_texture_path=out_dir / "env_dome.exr",
        water=not args.no_water,
        water_rings=args.water_rings,
        water_sectors=args.water_sectors,
        water_half_angle_deg=args.water_half_angle_deg,
        # Bow-away: a departing vessel shows its stern, not its side.
        heading_deg=90.0,
        wind_dir_deg=35.0,
        # Same wind Cox-Munk reads, so the picture and the radiometry agree (CLAUDE.md #6).
        wind_speed_m_s=wind_m_s,
    )
    if stage.errors:
        print(f"stage errors: {stage.errors}", file=sys.stderr)

    import omni.usd

    usd_stage = omni.usd.get_context().get_stage()
    vessel = stage.vessels["vessel"]
    translate_op = UsdGeom.Xformable(
        usd_stage.GetPrimAtPath(vessel.root_path)
    ).GetOrderedXformOps()[0]

    table = MaterialTable.from_library(MaterialLibrary.load(), spec.band.band_id)
    resolver = MaterialResolver(load_mapping_rules(), list(table.names))
    resolutions = resolver.resolve_all(prim_records(root="/World/Targets"))

    pipeline = PipelineConfig.from_sensor(
        sensor,
        table,
        lut,
        sky=sky,
        atmosphere=scene.layered,
        flat_field_enabled=not args.no_flat_field,
    )
    # ADR 0056: the M9 chain's NUC residual is authored in millikelvin and needs a radiometric
    # calibration to reach DN. A photon FPA has none (M11.6), and a photon camera in this
    # repository is shutterless by configuration anyway, so there is no FFC to model either.
    # Skipping it is the honest behaviour and is announced; failing here would make every
    # reflective-band render impossible for a reason that is not about the band.
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

    # The display span comes from the target's OWN nodes over the whole sequence, not from ambient:
    # eight bits is 256 levels and a sea-and-sky scene spans a hundred kelvin, so spanning ambient
    # would leave the whole vessel in a few codes. Here it matters more than usual, because the
    # funnel climbs 150 K during the film and a span picked off the first frame would clip it.
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
        print(
            f"the capture window ({span_s:.0f} s) runs past the scene's own schedule -- shorten it "
            f"or extend the profile: {exc}",
            file=sys.stderr,
        )
        return 1
    if args.span_c:
        span_c = (float(args.span_c[0]), float(args.span_c[1]))
        span_k = (span_c[0] + 273.15, span_c[1] + 273.15)
    else:
        # The funnel is deliberately LEFT OUT of the span and allowed to clip white. Spanning it
        # in is the obvious thing and it ruins the picture: the funnel climbs 150 K during the
        # film, so a span that holds it puts the sea, the hull and the deck -- everything with
        # structure in it -- inside about fifteen of the 256 codes, and the opening frames come
        # out nearly black. Clipping the hottest thing in the scene is also what a real thermal
        # image of a ship under way does.
        span_k = target_span_k(
            [{k: v for k, v in sample.items() if k != "stack"} for sample in node_samples]
        )
        span_c = (span_k[0] - 273.15, span_k[1] - 273.15)
    palette_name = args.palette or spec.isp.palette
    palette = palette_table(palette_name)
    print(
        f"display: fixed span {span_c[0]:.1f} to {span_c[1]:.1f} C, {palette_name} palette; "
        f"the funnel runs {node_samples[0]['stack'] - 273.15:.0f} to "
        f"{node_samples[-1]['stack'] - 273.15:.0f} C across the film and clips white on purpose"
    )

    camera = IrCamera(
        sensor,
        scene,
        pipeline=pipeline,
        prim_to_target=stage.prim_to_target,
        resolutions=resolutions,
        camera_path=stage.camera_path,
        # IG.6: the vessel steams away from a static camera, so all of the relative motion
        # is the target's. At 6 m/s and 250 m it is well under a pixel a frame -- which is
        # the honest answer for this scene, not a reason to leave the plane off.
        moving_prim_paths=[vessel.root_path],
        capture_rgb=args.rgb,
        strict_materials=False,
        strict_thermal_nodes=False,
        cloud_seed=args.cloud_seed,
        frame_period_s=args.interval_s,
        sea=sea,
        background_prim_paths=stage.water_paths,
    ).open(settle_frames=args.settle)

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
            "camera_height_m": args.camera_height_m,
            "interval_s": args.interval_s,
        },
    )
    history: list[dict[str, float]] = []
    planes_written = 0
    t_render = time.time()
    for index in range(args.frames):
        t_rel = camera.t_rel_s
        translate_op.Set(Gf.Vec3d(*track.position_m(t_rel)))

        outputs = camera.get_outputs(rt_subframes=args.rt_subframes)
        temps = camera.bridge.temperatures()
        t_app = np.asarray(outputs.apparent_t)

        # How much of the frame each node occupies, straight off the id plane: the only honest way
        # to say whether the funnel is still resolved, since the renderer decides occlusion.
        last = camera.last_frame
        by_node: dict[str, int] = {}
        if last is not None:
            for ident, path in last.labels.items():
                node = stage.prim_to_target.get(str(path))
                if node is not None:
                    by_node[node] = by_node.get(node, 0) + int((last.instance_id == ident).sum())

        range_m = track.range_m(t_rel)
        history.append(
            {
                "frame": index,
                "t_rel_s": round(t_rel, 3),
                "range_m": round(range_m, 1),
                "depression_deg": round(track.depression_deg(t_rel), 4),
                "sea_behind_k": round(
                    float(
                        np.atleast_1d(
                            sea.apparent_temperature_k(
                                scene.t0_s + t_rel, math.radians(track.depression_deg(t_rel))
                            )
                        )[0]
                    ),
                    3,
                ),
                "hull_px": by_node.get("hull", 0),
                "stack_px": by_node.get("stack", 0),
                **{f"{k}_k": round(v, 3) for k, v in temps.items()},
                "t_app_max_k": round(float(t_app.max()), 3),
            }
        )

        # The frame itself. `wants` is asked first so the companion RGB is not copied out of the
        # renderer on a frame whose planes are not being kept.
        if writer.wants(index):
            extra = {}
            if args.rgb and last is not None and last.rgb is not None:
                extra["rgb"] = last.rgb
            writer.write(
                outputs,
                frame_index=index,
                t_s=camera.last_frame_t_s,
                extra_planes=extra,
                extra_metadata={
                    "range_m": round(range_m, 3),
                    "depression_deg": round(track.depression_deg(t_rel), 4),
                    "node_temperatures_k": {k: round(float(v), 3) for k, v in temps.items()},
                },
            )
            planes_written += 1

        scaled = (t_app - span_k[0]) / (span_k[1] - span_k[0])
        spanned = palette[quantise_display(scaled)]

        def readout(image: Any, caption: str, row: dict = history[-1], node: dict = temps) -> Any:
            return overlay_readout(
                image,
                [
                    f"T+{row['t_rel_s'] / 60.0:05.1f} min   range {row['range_m']:6.0f} m   "
                    f"{row['depression_deg']:5.3f} deg down",
                    f"funnel {node['stack'] - 273.15:5.0f}C   hull {node['hull'] - 273.15:4.1f}C"
                    f"   sea behind {row['sea_behind_k'] - 273.15:4.1f}C",
                    f"hull {row['hull_px']:5.0f} px   funnel {row['stack_px']:4.0f} px   {caption}",
                ],
                {
                    "funnel": node["stack"],
                    "deck  ": node["superstructure"],
                    "hull  ": node["hull"],
                },
                span_k,
                bare=args.no_overlay,
            )

        write_png(
            frames_dir / f"ir_{index:05d}.png",
            readout(spanned, f"span {span_c[0]:.0f}-{span_c[1]:.0f}C {palette_name}"),
        )
        write_png(
            frames_dir / f"agc_{index:05d}.png",
            readout(np.asarray(outputs.display8), f"camera {spec.isp.agc}"),
        )
        if args.rgb and last is not None and last.rgb is not None:
            write_png(
                frames_dir / f"rgb_{index:05d}.png",
                np.ascontiguousarray(np.asarray(last.rgb)[..., :3]),
            )
        if index % 20 == 0 or index == args.frames - 1:
            print(
                f"  frame {index:4d}  T+{t_rel / 60.0:5.1f} min  R={range_m:6.0f} m  "
                f"funnel {temps['stack'] - 273.15:5.0f} C / {by_node.get('stack', 0):4d} px  "
                f"hull {by_node.get('hull', 0):5d} px"
            )
    render_s = time.time() - t_render
    camera.close()

    videos = {}
    if ffmpeg_available():
        for tag, name in (("ir", "vessel_departure_ir"), ("agc", "vessel_departure_agc")):
            videos[tag] = str(
                encode_mp4(str(frames_dir / f"{tag}_*.png"), out_dir / f"{name}.mp4", fps=args.fps)
            )
        if args.rgb and any(frames_dir.glob("rgb_*.png")):
            videos["rgb"] = str(
                encode_mp4(
                    str(frames_dir / "rgb_*.png"),
                    out_dir / "vessel_departure_rgb.mp4",
                    fps=args.fps,
                )
            )
        if not args.keep_frames:
            for png in frames_dir.glob("*.png"):
                png.unlink()
            frames_dir.rmdir()
    else:
        print("ffmpeg not found: keeping the PNG sequence", file=sys.stderr)

    first, last_row = history[0], history[-1]
    summary = {
        "boot_s": round(boot_s, 2),
        "render_s": round(render_s, 2),
        "frames": args.frames,
        "interval_s": args.interval_s,
        "fps": args.fps,
        "sensor": spec.name,
        "resolution": [spec.fpa.width, spec.fpa.height],
        "ifov_mrad": round(ifov_mrad, 4),
        "scene": pathlib.Path(args.scene).name,
        "config_hash": config_hash(sensor),
        "band_hash": band_hash(sensor),
        "float_format": args.float_format,
        "plane_stride": args.plane_stride,
        "plane_frames": planes_written,
        "track": {
            "start_range_m": track.start_range_m,
            "end_range_m": round(track.range_m(span_s), 1),
            "speed_m_s": track.speed_m_s,
            "camera_height_m": args.camera_height_m,
            "horizon_depression_deg": round(math.degrees(sea.horizon_rad), 5),
            "horizon_time_s": round(track.horizon_time_s(), 1),
        },
        "sea": {"bulk_sst_k": bulk_sst_k, "wind_m_s": round(wind_m_s, 3)},
        "engine": {
            "funnel_start_k": first["stack_k"],
            "funnel_end_k": last_row["stack_k"],
            "funnel_rise_k": round(last_row["stack_k"] - first["stack_k"], 2),
            "stack_px_start": first["stack_px"],
            "stack_px_end": last_row["stack_px"],
        },
        "display_span_c": list(span_c),
        "palette": palette_name,
        "videos": videos,
        "history": history,
    }
    path = out_dir / "summary.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)
    print(
        f"\n{args.frames} frames in {render_s:.1f} s -> {out_dir}\n"
        f"funnel {first['stack_k'] - 273.15:.0f} -> {last_row['stack_k'] - 273.15:.0f} C while the "
        f"hull went {first['hull_px']} -> {last_row['hull_px']} px\nwrote {path}"
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
    app.close(exit_code=status if isinstance(status, int) else 1)
