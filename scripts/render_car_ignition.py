#!/usr/bin/env python3
"""Roadmap MP.4b: film a cold car starting its engine, in LWIR, with a per-pixel temperature field.

This is the scene ADR 0087 exists for. Every frame this project rendered before it carried **one
temperature per prim**, so a bonnet was one flat value however its engine was doing. Here the
bonnet is still one USD prim, and by the end of the run it is several kelvin hotter over the block
than at its wings -- a gradient no per-prim bridge can produce.

Three mechanisms, and they are worth telling apart when you look at the output:

* the **bonnet** gains a gradient in minutes (thin steel, tau about 9 min), shaped by ADR 0088's
  configuration factor to the bay below it -- computed from the bay's dimensions, not authored;
* the **asphalt** under the car warms by a few tenths of a kelvin over the same period (tau about
  an hour), which at a 50 mK NETD is still several noise-equivalent steps, with a stripe under the
  exhaust run;
* the **wheels do essentially nothing**, and that is the physics rather than a gap. §6.6 makes tyre
  heating flexing work and brake heating kinetic energy, and a car idling in a car park is doing
  neither. The readout prints their rise beside the bonnet's so the contrast is stated.

The scene is shot at 03:00 local under thick overcast on purpose. Under cloud the sky radiates near
air temperature, so every surface starts within a kelvin or two of ambient and frame 0 is genuinely
flat. Under a CLEAR night the car would already have left a warm car-shaped patch on the road
before the key turned, by blocking the cold sky rather than by being warm (ADR 0088) -- real, and
worth filming, but not a demonstration of engine heat.

It is a **time-lapse**: the whole chain is told that `--frame-period` seconds really passed between
captures, so the FFC fires on its real schedule and the fixed pattern drifts by a real amount
(ADR 0074). The default covers 25 minutes in 26 frames.

Run with the Isaac interpreter (docs/decisions/0002), after `make luts`:
    python.sh scripts/render_car_ignition.py --frames 26 --rgb --out outputs/car_ignition
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
parser.add_argument("--out", default="outputs/car_ignition")
parser.add_argument("--frames", type=int, default=26)
parser.add_argument("--sensor", default=str(REPO / "configs/sensors/flir_boson_640_lwir.yaml"))
parser.add_argument(
    "--scene", default=str(REPO / "configs/scenes/car_ignition_overcast_night.yaml")
)
parser.add_argument(
    "--frame-period",
    type=float,
    default=60.0,
    help="scene seconds between captures. The engine bay's time constant is 750 s, so anything "
    "much under a minute films the same picture many times",
)
parser.add_argument("--depression-deg", type=float, default=45.0)
parser.add_argument("--range-m", type=float, default=27.0)
parser.add_argument("--azimuth-deg", type=float, default=22.0)
parser.add_argument("--ground-cell-m", type=float, default=0.30)
parser.add_argument("--bonnet-cell-m", type=float, default=0.07)
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
    "--rt-subframes",
    type=int,
    default=8,
    help="path-traced subframes accumulated per capture; the multi-band sweep sets this",
)
parser.add_argument("--settle", type=int, default=16)
parser.add_argument("--no-chain", action="store_true", help="ideal camera: no M9 sensor chain")
parser.add_argument("--no-flat-field", action="store_true")
parser.add_argument(
    "--no-fields",
    action="store_true",
    help="render WITHOUT the point-wise fields -- the pre-ADR-0087 picture, for comparison. The "
    "bonnet goes flat and the road loses its patch; nothing else changes",
)
parser.add_argument(
    "--span-c",
    nargs=2,
    type=float,
    metavar=("LOW", "HIGH"),
    help="a FIXED display span in Celsius, written alongside the camera's own AGC output. Worth "
    "using here: the scene's global AGC spans the whole frame, and the bonnet sits at the top of "
    "it, so the gradient this demo exists to show is clipped to display codes 253-255. Spanning "
    "the bonnet's own range instead puts the gradient across the full 0-255 axis. Defaults to a "
    "span taken from the bonnet's first and last temperatures",
)
parser.add_argument(
    "--palette",
    default="gray",
    help="palette for the fixed-span image. Grey by default",
)
parser.add_argument("--fps", type=float, default=10.0, help="playback rate of the encoded videos")
parser.add_argument(
    "--no-overlay",
    action="store_true",
    help="write the bare pictures, with no caption block or temperature gauge",
)
parser.add_argument("--rgb", action="store_true", help="also capture the companion visible frame")
args = parser.parse_args()

sys.path.insert(0, str(REPO / "src"))
_t0 = time.time()
from isaacsim import SimulationApp  # noqa: E402

from irsim_isaac.env import simulation_app_config  # noqa: E402

app = SimulationApp(simulation_app_config(renderer="RayTracedLighting"))
boot_s = time.time() - _t0


def main() -> int:
    import numpy as np
    import omni.usd

    from irsim.atmosphere.skylight import skylight_for_sensor
    from irsim.config.loader import (
        band_hash,
        config_hash,
        load_sensor_config,
        with_integration_time_ms,
    )
    from irsim.io import write_frame
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
    from irsim_eval.video import encode_mp4, ffmpeg_available, overlay_readout
    from irsim_isaac.car_demo import CameraSetup, build_car_demo, describe, grids_source
    from irsim_isaac.display_span import DisplaySpan
    from irsim_isaac.pipeline.ir_camera import IrCamera
    from irsim_isaac.pipeline.materials_usd import prim_records

    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(exist_ok=True)

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

    # The fields are sized from the camera's own frustum, so the road patch cannot end up smaller
    # than the frame -- which would make the MP.3 overlay raise half an hour into a Kit session.
    hfov, vfov = _fields_of_view(spec)
    camera_setup = CameraSetup(
        depression_deg=args.depression_deg,
        range_m=args.range_m,
        azimuth_deg=args.azimuth_deg,
    )
    stage = omni.usd.get_context().get_stage()
    demo = build_car_demo(
        scene,
        camera=camera_setup,
        vfov_deg=vfov,
        hfov_deg=hfov,
        ground_cell_m=args.ground_cell_m,
        bonnet_cell_m=args.bonnet_cell_m,
        stage=stage,
    )

    table = MaterialTable.from_library(MaterialLibrary.load(), spec.band.band_id)
    resolver = MaterialResolver(load_mapping_rules(), list(table.names))
    resolutions = resolver.resolve_all(prim_records(root="/World"))
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

            pipeline = attach_sensor_chain(pipeline, scene.weather, t0_s=scene.t0_s)

    road = demo.ground_field.patch
    bonnet = demo.bonnet_field.patch
    print(f"  {grids_source(scene)}")
    print(
        f"\n{spec.name}: {spec.fpa.width}x{spec.fpa.height}, {hfov:.1f} x {vfov:.1f} deg, "
        f"camera {args.depression_deg:.0f} deg down at {args.range_m:.0f} m"
    )
    print(
        f"  bonnet field: {bonnet.n_u} x {bonnet.n_v} cells at {bonnet.du_m * 100:.0f} mm\n"
        f"  road field:   {road.n_u} x {road.n_v} cells at {road.du_m * 100:.0f} mm "
        f"({road.extent_m[0]:.0f} x {road.extent_m[1]:.0f} m)"
    )
    if args.no_fields:
        print("  --no-fields: rendering the pre-ADR-0087 picture, one temperature per prim")

    camera = IrCamera(
        sensor,
        scene,
        pipeline=pipeline,
        prim_to_target=demo.prim_to_target,
        resolutions=resolutions,
        camera_path=demo.camera_path,
        surface_fields=() if args.no_fields else demo.surface_bindings(),
        frame_period_s=args.frame_period,
        capture_rgb=args.rgb,
        strict_materials=False,
        strict_thermal_nodes=False,
    ).open(settle_frames=args.settle)

    # FIXED spans, not the camera's AGC -- and **two** of them, because no single linear span
    # shows both features. The bonnet spans about 6 K by the end and the road patch under 1 K, so
    # a span wide enough for the bonnet renders the road flat black and one tight enough for the
    # road clips the bonnet white. That is not a modelling problem; it is why a thermal operator
    # changes range, and why plateau equalisation exists. The camera's own AGC output is written
    # beside both, and on this scene it clips the bonnet to display codes 253-255 -- exactly what
    # a real core with global AGC does.
    #
    # They are anchored on the **first frame's own median apparent temperature**, not on T_air.
    # The two differ by about 0.9 K here: asphalt at eps 0.95 reflects a slightly cooler sky and
    # 27 m of atmosphere sits in front of it, so a span centred on the air temperature puts the
    # whole picture below its floor and renders it black. Anchoring on what the camera actually
    # reads is also what a real operator does with a spot meter.
    palette = palette_table(args.palette)
    bonnet_rise = _predicted_bonnet_rise_k(scene, args.frames * args.frame_period)
    spans: dict[str, DisplaySpan] = {}
    if args.span_c:
        low_k, high_k = (float(v) + 273.15 for v in args.span_c)
        spans["spanned"] = DisplaySpan(kind="apparent_t", low=low_k, high=high_k)

    print(
        f"\n{'t/s':>6} {'T_air':>7} {'bay':>7} {'exh':>7} {'bonnet hi':>9} {'grad':>6} "
        f"{'road patch':>10} {'tyre rise':>9} | scene, kelvin"
    )
    written, rows = [], []
    t_render = time.time()
    for index in range(args.frames):
        # Before the capture: `get_outputs` advances the camera's clock on the way out, so
        # afterwards `t_rel_s` is the *next* frame's time. On a time-lapse that is a whole frame
        # period, and the readout would label every row with the state of the row after it.
        t_abs = scene.t0_s + camera.t_rel_s
        outputs = camera.get_outputs(rt_subframes=args.rt_subframes)
        row = describe(demo, scene, t_abs)
        rows.append(row)
        print(
            f"{row['t_s']:6.0f} {row['t_air_k']:7.2f} {row['engine_bay_k']:7.2f} "
            f"{row['exhaust_pipe_k']:7.2f} {row['bonnet_hot_k']:9.2f} "
            f"{row['bonnet_gradient_k']:6.2f} {row['road_patch_k']:10.3f} "
            f"{row['tyre_rise_k']:9.3f}"
        )
        # What the CAMERA sees of the road patch, which is not what the field holds: the warmest
        # cells are directly under the car and a 45-degree view cannot see them. Only the fringe
        # reaches the sensor, so this number is the honest one to quote about an image.
        row["visible_road_patch_k"] = _visible_road_patch_k(camera, outputs)

        extra = {}
        if outputs.apparent_t is not None:
            if not spans:
                base = float(np.median(np.asarray(outputs.apparent_t)))
                spans = {
                    # The bonnet's own range: the gradient across one prim, the milestone's claim.
                    "span_car": DisplaySpan(
                        kind="apparent_t", low=base - 0.5, high=base + max(bonnet_rise, 2.0) + 0.5
                    ),
                    # A 1.6 K window: the road's patch and its falloff, the bonnet clipped white.
                    "span_road": DisplaySpan(kind="apparent_t", low=base - 0.4, high=base + 1.2),
                }
                for name, sp in spans.items():
                    print(f"       display {name}: {sp.caption} (anchored on {base:.2f} K)")
            # `quantise_display` is the ISP's own rounding, so the fixed-span pictures go through
            # the same mapping the camera's display branch uses and not a second, similar one.
            for name, sp in spans.items():
                extra[name] = palette[quantise_display(sp.scale(outputs))]

            # The video frames. Defaults bind this frame's numbers at definition time; a closure
            # over the loop variables would be evaluated later, which is the classic way to
            # caption every frame with the last frame's values.
            def readout(image: Any, caption: str, sp: Any, r: dict = row) -> Any:
                minutes, seconds = divmod(int(r["t_s"]), 60)
                key = "OFF" if r["engine_bay_k"] - r["t_air_k"] < 0.05 else "RUNNING"
                return overlay_readout(
                    np.asarray(image, dtype=np.uint8),
                    [
                        f"T+{minutes:02d}:{seconds:02d}   engine {key}",
                        f"bonnet {r['bonnet_gradient_k']:4.2f} K across one prim"
                        f"   wheels {r['tyre_rise_k']:+.2f} K",
                        f"1 frame / {args.frame_period:g} s time-lapse   {caption}",
                    ],
                    {
                        "bay    ": r["engine_bay_k"],
                        "bonnet ": r["bonnet_hot_k"],
                        "wing   ": r["bonnet_cold_k"],
                        "road   ": r["road_near_k"],
                        "air    ": r["t_air_k"],
                    },
                    (sp.low, sp.high) if sp is not None else None,
                    bare=args.no_overlay,
                )

            for name, sp in spans.items():
                write_png(
                    frames_dir / f"{name}_{index:05d}.png",
                    readout(extra[name], f"{sp.caption} {args.palette}", sp),
                )
            write_png(
                frames_dir / f"agc_{index:05d}.png",
                readout(np.asarray(outputs.display8), f"camera {spec.isp.agc}", None),
            )
        if args.rgb and camera.last_frame is not None and camera.last_frame.rgb is not None:
            extra["rgb"] = camera.last_frame.rgb
            write_png(
                frames_dir / f"rgb_{index:05d}.png",
                np.ascontiguousarray(np.asarray(camera.last_frame.rgb)[..., :3]),
            )
        elif args.rgb and camera.rgb_problem:
            print(f"  RGB not captured: {camera.rgb_problem}", file=sys.stderr)
        written.append(
            write_frame(
                out_dir,
                outputs,
                frame_index=index,
                t_s=t_abs,
                start_utc=scene.weather.start_utc,
                config_hash=config_hash(sensor),
                band_hash=band_hash(sensor),
                quantity=pipeline.quantity,
                float_format=args.float_format,
                extra_planes=extra,
                extra_metadata={
                    "scene": pathlib.Path(args.scene).name,
                    "point_wise_fields": not args.no_fields,
                    "frame_period_s": args.frame_period,
                    "thermal": row,
                    # PT.19: pixels each bound patch took; a zero is a binding nothing consumed.
                    "patch_coverage": camera.patch_coverage,
                },
            )
        )
        if outputs.apparent_t is not None:
            t_app = np.asarray(outputs.apparent_t)
            print(
                f"       rendered T_app {t_app.min():.2f} .. {t_app.max():.2f} K "
                f"(median {np.median(t_app):.2f})"
            )
    render_s = time.time() - t_render
    camera.close()

    first, last = rows[0], rows[-1]
    print(
        f"\nover {last['t_s'] - first['t_s']:.0f} s:"
        f"\n  bonnet gradient  {first['bonnet_gradient_k']:.3f} -> "
        f"{last['bonnet_gradient_k']:.2f} K across ONE prim "
        f"({last['bonnet_gradient_k'] / 0.05:.0f} NETD) -- the milestone's claim"
        f"\n  road patch       {first['road_patch_k']:.3f} -> {last['road_patch_k']:.3f} K in the"
        f" field, but only {last['visible_road_patch_k']:+.3f} K "
        f"({abs(last['visible_road_patch_k']) / 0.05:.0f} NETD) reaches the camera: the warmest"
        f" road is UNDER the car, and a {args.depression_deg:.0f} deg view cannot see it"
        f"\n  wheels           {last['tyre_rise_k']:+.3f} K -- a stationary car does not heat its"
        f" tyres; that is §6.6, not a gap"
    )
    videos: dict[str, str] = {}
    if ffmpeg_available():
        streams = [*spans, "agc"] + (["rgb"] if args.rgb else [])
        for name in streams:
            if not list(frames_dir.glob(f"{name}_*.png")):
                continue
            videos[name] = str(
                encode_mp4(
                    str(frames_dir / f"{name}_*.png"),
                    out_dir / f"car_ignition_{name}.mp4",
                    fps=args.fps,
                )
            )
        for name, path in videos.items():
            print(f"  {name:9s} -> {path}")
    else:
        print("ffmpeg not found: keeping the PNG sequence in frames/", file=sys.stderr)

    summary = {
        "boot_s": round(boot_s, 2),
        "videos": videos,
        "fps": args.fps,
        "render_s": round(render_s, 2),
        "frames": len(written),
        "sensor": spec.name,
        "point_wise_fields": not args.no_fields,
        "bonnet_cells": [bonnet.n_u, bonnet.n_v],
        "road_cells": [road.n_u, road.n_v],
        "thermal": rows,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {len(written)} frames to {out_dir} in {render_s:.1f} s")
    return 0


def _visible_road_patch_k(camera: object, outputs: object) -> float:
    """Near-road minus far-road in the **rendered** frame, in kelvin.

    The field's own `road_patch_k` is the hottest cell under the car against the far road. The
    camera cannot see that cell -- the car is standing on it -- so at a 45 degree depression only
    the fringe reaches the sensor and the visible contrast is a small fraction of it. Reporting
    both is the point: one is what the model holds and the other is what an image would show.
    """
    import numpy as np

    frame = camera.last_frame  # type: ignore[attr-defined]
    plane = getattr(outputs, "apparent_t", None)
    if frame is None or plane is None or frame.instance_id is None:
        return float("nan")
    t = np.asarray(plane, dtype=np.float64)
    ids = np.asarray(frame.instance_id)
    if ids.shape != t.shape:
        # The id plane is supersampled and the output plane is native. Ids never blend (ADR 0014),
        # so this takes the sample at each native pixel's block centre rather than box-filtering.
        fy, fx = ids.shape[0] // t.shape[0], ids.shape[1] // t.shape[1]
        if fy < 1 or fx < 1 or (fy * t.shape[0], fx * t.shape[1]) != ids.shape:
            return float("nan")
        ids = ids[fy // 2 :: fy, fx // 2 :: fx]
    road_id = np.bincount(ids.ravel()).argmax()  # the road fills the frame
    road = ids == road_id
    car = (~road) & (ids != 0)
    if not road.any() or not car.any():
        return float("nan")
    ys, xs = np.nonzero(car)
    grid_y, grid_x = np.mgrid[0 : t.shape[0], 0 : t.shape[1]]
    d = np.hypot(grid_y - ys.mean(), grid_x - xs.mean())
    near = road & (d < np.percentile(d[road], 12))
    far = road & (d > np.percentile(d[road], 80))
    if not near.any() or not far.any():
        return float("nan")
    return float(t[near].mean() - t[far].mean())


def _predicted_bonnet_rise_k(scene: object, duration_s: float) -> float:
    """§6.6's own arithmetic for how far the bonnet climbs, so the span is fixed before frame 1.

    Taken from the model rather than measured off the sequence, because the span has to be the
    same for every frame and frame 0 is a picture of a flat car. Only the white end of the span
    depends on it, so an imperfect estimate costs contrast and never correctness.
    """
    from irsim.thermal.vehicle import VEHICLE_HEAT_SOURCES, first_order_rise

    spec = VEHICLE_HEAT_SOURCES["engine_bay"]
    bay_rise = float(
        first_order_rise(max(duration_s, 1.0), 0.45 * spec.delta_t_max_k, spec.tau_rise_s)
    )
    # The skin reaches a fraction of the bay's rise; 0.23 is measured off this scene's own
    # solve (5.8 K of skin under 25 K of bay) and only sets where the white end of the span
    # falls, so an imperfect guess costs contrast and never correctness.
    return 0.23 * bay_rise


def _fields_of_view(spec: object) -> tuple[float, float]:
    """Horizontal and vertical field of view in degrees, from the sensor's own geometry."""
    import math

    fpa = spec.fpa  # type: ignore[attr-defined]
    optics = spec.optics  # type: ignore[attr-defined]
    f_mm = optics.focal_length_mm
    width_mm = fpa.width * fpa.pitch_um * 1e-3
    height_mm = fpa.height * fpa.pitch_um * 1e-3
    return (
        2.0 * math.degrees(math.atan(0.5 * width_mm / f_mm)),
        2.0 * math.degrees(math.atan(0.5 * height_mm / f_mm)),
    )


try:
    status = main()
except Exception as exc:  # noqa: BLE001 - a demo crash should still report, not hang
    import traceback

    traceback.print_exc()
    print(f"render failed: {type(exc).__name__}: {exc}", file=sys.stderr)
    status = 1
finally:
    app.close(exit_code=status if isinstance(status, int) else 1)
