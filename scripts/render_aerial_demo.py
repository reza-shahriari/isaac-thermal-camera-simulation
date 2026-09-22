#!/usr/bin/env python3
"""Roadmap M10.19: render the phase-1 aerial demo stage through `IrCamera` and export the frames.

This is the end of the M10 chain as a single command: a USD stage with small targets against sky,
the Boson configuration, the scene's one WeatherSeries feeding the atmosphere, the sky model and
the target solvers, the whole sensor chain, and four outputs per frame on disk in physical units
(M10.10a). It is the thing to run when the question is "what does the camera see".

There is no sky *geometry* in the stage. Infrared background rays take T_sky(theta) from the sky
model at their own elevation, or T_ground below the horizon, because every colour AOV on this
build is float16 and would quantise the sky to ~100 mK against a 50 mK NETD (ADR 0014, ADR 0060).

With --rgb the companion visible frame is rendered against a generated environment dome (ADR
0073): a Preetham daylight sky plus a hazed Lambertian terrain, from the scene's own NOAA sun
position and the shared weather's visibility and irradiance, with a distant light for the solar
disc. It is a light, not geometry, so it changes no infrared pixel -- run --no-dome to see that
for yourself, and to see how little of the visible frame geometry alone carries.

Run with the Isaac interpreter (docs/decisions/0002), after `make luts`:
    python.sh scripts/render_aerial_demo.py --frames 8 --out outputs/aerial_demo
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

REPO = pathlib.Path(__file__).resolve().parents[1]

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--out", default="outputs/aerial_demo")
parser.add_argument("--frames", type=int, default=4)
parser.add_argument("--sensor", default=str(REPO / "configs/sensors/flir_boson_640_lwir.yaml"))
parser.add_argument("--scene", default=str(REPO / "configs/scenes/sky_target_clear_day.yaml"))
parser.add_argument("--tilt-deg", type=float, default=8.0)
parser.add_argument("--t0", type=float, default=0.0, help="scene time of the first frame, seconds")
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
parser.add_argument("--settle", type=int, default=16)
parser.add_argument(
    "--rt-subframes",
    type=int,
    default=8,
    help="path-traced subframes accumulated per capture; the multi-band sweep sets this",
)
parser.add_argument("--fps", type=float, default=30.0, help="playback rate of the encoded video")
parser.add_argument("--no-chain", action="store_true", help="ideal camera: no M9 sensor chain")
parser.add_argument(
    "--rgb",
    action="store_true",
    help="also capture the visible-light frame from the same camera (registered RGB/IR pair)",
)
parser.add_argument(
    "--heading-deg",
    type=float,
    default=0.0,
    help="compass bearing the camera looks along; 0 is north. Places the sun in the visible frame",
)
parser.add_argument(
    "--camera-height-m",
    type=float,
    default=2.0,
    help="height above the terrain painted on the dome; sets how wide the horizon haze band is",
)
parser.add_argument(
    "--no-dome", action="store_true", help="untextured grey dome (ADR 0073 ablation)"
)
parser.add_argument(
    "--no-flat-field",
    action="store_true",
    help="skip the camera's flat-field correction, so cos^4 vignetting shows (ME.8 ablation)",
)
parser.add_argument(
    "--no-point-targets",
    action="store_true",
    help="render sub-pixel targets as geometry instead of injecting them (ADR 0071 ablation)",
)
args = parser.parse_args()

t_boot = time.time()
from isaacsim import SimulationApp  # noqa: E402

from irsim_isaac.env import simulation_app_config  # noqa: E402

app = SimulationApp(simulation_app_config())
boot_s = time.time() - t_boot


def main() -> int:
    import numpy as np

    from irsim.atmosphere.skylight import skylight_for_sensor
    from irsim.config.loader import (
        band_hash,
        config_hash,
        load_sensor_config,
        with_integration_time_ms,
    )
    from irsim.io import write_frame
    from irsim.materials.library import MaterialLibrary
    from irsim.materials.mapping import MaterialResolver, load_mapping_rules
    from irsim.materials.table import MaterialTable
    from irsim.pipeline.core import PipelineConfig
    from irsim.radiometry.lut_files import (
        load_band_lut_for_config,
        load_band_response_for_config,
    )
    from irsim.scene import Scene
    from irsim_eval.video import encode_mp4, ffmpeg_available
    from irsim_isaac.aerial_demo import analytic_targets, build_aerial_demo, describe
    from irsim_isaac.pipeline.ir_camera import IrCamera
    from irsim_isaac.pipeline.materials_usd import prim_records
    from irsim_isaac.visible_sky import dome_spec_from_scene

    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

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
    # model has to be built in the same form the pipeline will read it in, and until this line
    # existed the scene was always built in `lb`, so this script ran on LWIR and refused every
    # reflective band at `PipelineConfig.from_sensor`. The two other aerial drivers took this
    # with M11.10; this one was missed.
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

    # The dome reads the sun, the visibility and the irradiance off the scene, so the companion
    # frame cannot end up showing a different hour of a different day than the infrared one.
    dome = None
    if not args.no_dome:
        dome = dome_spec_from_scene(
            scene, heading_deg=args.heading_deg, camera_height_m=args.camera_height_m
        )
        print(
            f"\nenvironment dome: sun at {dome.sun_elevation_deg:.1f} deg elevation, "
            f"{dome.sun_azimuth_deg:.1f} deg azimuth "
            f"({dome.sun_azimuth_deg - args.heading_deg:+.1f} deg from boresight); "
            f"turbidity {dome.turbidity:.2f} at {dome.visibility_m / 1e3:.0f} km visibility; "
            f"DNI {dome.dni_w_m2:.0f} W/m2, DHI {dome.dhi_w_m2:.0f} W/m2"
        )
    demo = build_aerial_demo(
        camera_tilt_deg=args.tilt_deg, dome=dome, dome_texture_path=out_dir / "env_dome.exr"
    )
    if demo.errors:
        print(f"stage errors: {demo.errors}", file=sys.stderr)

    # Below one native pixel the renderer samples geometry and gets a phase-dependent fraction of
    # the flux (ADR 0071), so those targets are hidden and injected analytically instead (MS.6).
    # `analytic_targets` hides them in the same call that produces their specs, so the two paths
    # cannot both claim a target.
    ifov_mrad = 1e3 * spec.fpa.pitch_um * 1e-3 / spec.optics.focal_length_mm
    analytic = [] if args.no_point_targets else analytic_targets(demo, ifov_mrad)

    table = MaterialTable.from_library(MaterialLibrary.load(), spec.band.band_id)
    resolver = MaterialResolver(load_mapping_rules(), list(table.names))
    resolutions = resolver.resolve_all(prim_records(root="/World/Targets"))
    unresolved = [r.path for r in resolutions if not r.mapped]
    if unresolved:
        print(f"unmapped prims: {unresolved}", file=sys.stderr)

    pipeline = PipelineConfig.from_sensor(
        sensor,
        table,
        lut,
        sky=scene.sky_models[spec.band.band_id],
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

            # The chain's housing node reads ambient, so it takes the scene's own WeatherSeries
            # -- the same object the atmosphere, the sky and the target solvers hold (CLAUDE.md #6).
            pipeline = attach_sensor_chain(pipeline, scene.weather, t0_s=scene.t0_s)

    rows = describe(demo, ifov_mrad)
    print(
        f"\n{spec.name}: {spec.fpa.width}x{spec.fpa.height}, IFOV {ifov_mrad:.3f} mrad, "
        f"{spec.optics.supersample_factor}x supersampled, boresight "
        f"{demo.boresight_elevation_deg():.1f} deg above the horizon"
    )
    injected = {t.name for t in analytic}
    for row in rows:
        flag = "  ANALYTIC (MS.6)" if row["name"] in injected else ""
        if row["subpixel"] and row["name"] not in injected:
            flag = "  SUB-PIXEL, rendered anyway (--no-point-targets)"
        print(
            f"  {row['name']:12s} {row['range_m']:7.0f} m  {row['mrad']:6.3f} mrad  "
            f"{row['pixels']:6.2f} px  el {row['elevation_deg']:5.1f} deg  "
            f"{row['material']}{flag}"
        )

    camera = IrCamera(
        sensor,
        scene,
        pipeline=pipeline,
        prim_to_target=demo.prim_to_target,
        resolutions=resolutions,
        analytic_targets=analytic,
        camera_path=demo.camera_path,
        capture_rgb=args.rgb,
        strict_materials=False,
        strict_thermal_nodes=False,
    ).open(settle_frames=args.settle)

    written = []
    t_render = time.time()
    for index in range(args.frames):
        outputs = camera.get_outputs(rt_subframes=args.rt_subframes)
        # The companion visible frame, already box-filtered onto the IR pixel grid, so the pair is
        # registered by construction rather than by calibration. An `extra_plane`, not an output:
        # nothing in the radiometric chain reads it, and the sidecar's unit string says so.
        extra = {}
        if args.rgb and camera.last_frame is not None and camera.last_frame.rgb is not None:
            extra["rgb"] = camera.last_frame.rgb
        elif args.rgb and camera.rgb_problem:
            print(f"  RGB not captured: {camera.rgb_problem}", file=sys.stderr)
        record = write_frame(
            out_dir,
            outputs,
            frame_index=index,
            t_s=camera.last_frame_t_s,
            start_utc=scene.weather.start_utc,
            config_hash=config_hash(sensor),
            band_hash=band_hash(sensor),
            quantity=pipeline.quantity,
            float_format=args.float_format,
            extra_planes=extra,
            extra_metadata={
                "scene": pathlib.Path(args.scene).name,
                "camera_tilt_deg": demo.camera_tilt_deg,
                "camera_heading_deg": args.heading_deg,
                "targets": rows,
                "analytic_targets": sorted(injected),
            },
        )
        written.append(record)
        if outputs.apparent_t is not None:
            t_app = np.asarray(outputs.apparent_t)
            print(
                f"  frame {index}: T_app {t_app.min():.2f} .. {t_app.max():.2f} K "
                f"(median {np.median(t_app):.2f})"
            )
    render_s = time.time() - t_render
    camera.close()

    # The clip. The owner reviews a render by watching it, and this driver -- the aerial lane's
    # own -- produced still frames only. `write_frame` already numbers the display frames
    # zero-padded, so the sequence encodes straight off what is on disk: no second PNG written and
    # nothing deleted afterwards, because here the frames *are* the dataset.
    videos: dict[str, str] = {}
    if ffmpeg_available():
        for tag, pattern in (("ir", "frame_*_display8.png"), ("rgb", "frame_*_rgb.png")):
            if not any(out_dir.glob(pattern)):
                continue  # no RGB was captured, so there is nothing to encode
            videos[tag] = str(
                encode_mp4(str(out_dir / pattern), out_dir / f"aerial_demo_{tag}.mp4", fps=args.fps)
            )
    else:
        print("ffmpeg not found: the frames are on disk, unencoded", file=sys.stderr)

    summary = {
        "boot_s": round(boot_s, 2),
        "render_s": round(render_s, 2),
        "frames": len(written),
        "fps": args.fps,
        "videos": videos,
        "sensor": spec.name,
        "resolution": [spec.fpa.width, spec.fpa.height],
        "supersample": spec.optics.supersample_factor,
        "ifov_mrad": round(ifov_mrad, 4),
        "dome": None
        if dome is None
        else {
            "sun_elevation_deg": round(dome.sun_elevation_deg, 3),
            "sun_azimuth_deg": round(dome.sun_azimuth_deg, 3),
            "heading_deg": args.heading_deg,
            "turbidity": round(dome.turbidity, 3),
            "visibility_m": dome.visibility_m,
            "dni_w_m2": round(dome.dni_w_m2, 2),
            "dhi_w_m2": round(dome.dhi_w_m2, 2),
        },
        "targets": rows,
        "analytic": sorted(injected),
        "files": [str(p.name) for r in written for p in r.files.values()],
    }
    path = out_dir / "summary.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)
    print(f"\n{len(written)} frames in {render_s:.1f} s -> {out_dir}\nwrote {path}")
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
