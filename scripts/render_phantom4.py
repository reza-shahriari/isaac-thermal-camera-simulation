#!/usr/bin/env python3
"""Roadmap AI.2 (in-engine half): fly an **imported** third-party aircraft, in LWIR and in colour.

    IRSIM_GPU=0 python.sh scripts/render_phantom4.py --frames 96

Every other render driver in this repo films geometry this project authored in Python. This one
films a 62 MB DJI Phantom 4 Pro FBX that nobody here modelled, prepared on the CPU by
``scripts/prep_asset.py`` (ADR 0128) and referenced onto the stage as-is. What makes that possible
is not a new renderer path but a new *mapping* rung: the asset's 21 source material names resolve
through ``configs/assets/phantom4.yaml`` instead of through globs that were written for cars.

**The aircraft flies, and the pair is filmed together.** One circuit of a lemniscate
(:class:`irsim_isaac.asset_flight.FigureEightTrack`) in front of a ground observer, over the
aircraft's whole 28-minute mission: the slant range swings 2.2:1, the aspect sweeps through both
broadsides and tail-on, and the elevation never falls within ten degrees of the horizon -- this
scene authors no terrain, so an aircraft below the horizon would be backed by sky-model radiance
at near-air temperature instead of cold sky. The infrared frame and its visible companion come
out of the **same** capture (ADR 0073), box-filtered onto the same pixel grid, so the pair is
registered by construction and either one can be used to check the other.

**The mount is derived, not typed in.** The prepared USD is Z-up at ``metersPerUnit = 1`` and the
scene config says so in its own ``world_frame:`` block; the stage is the renderer's Y-up, because
the dome, the sun and the sky model all read the stage's +Y as up and its -Z as north. The single
rotation between the two is computed from that block by
:func:`irsim_isaac.asset_flight.world_frame_to_stage` rather than written as a ``rotateX 90``,
because a hand-written axis swap is how an aircraft ends up mirrored, or rolled, or lit from the
wrong quarter with nothing to complain about. An earlier revision of this driver authored the
stage Z-up instead and aimed the camera with the default Y-up ``look_at_quaternion``: the picture
came back rolled, the aircraft apparently pitched over, and nothing in the physics was wrong.

**Temperature is per prim here, not per cell.** The scene's mesh fields are solved (234,923 cells)
but `IrCamera` takes planar `SurfaceBinding`s only, and `MeshPointBridge` -- the object that turns
a world position into a cell on a real mesh -- has never been driven by a render: its only driver,
``scripts/quad_flight_mesh.py``, runs on a synthetic G-buffer. Wiring it into the camera is the
remainder of `AI.2` and is called out in the summary this writes rather than papered over.

The solver's frame stays the asset's. The scene's patches are authored in the archive's own
Z-up coordinates and are *not* bound to the camera here, so the mount rotation moves the rendered
aircraft and touches nothing the thermal solve depends on. The day a `MeshPointBridge` does reach
the camera, that binding has to carry this rotation with it.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import pathlib
import sys
import time
from typing import Any

REPO = pathlib.Path(__file__).resolve().parents[1]

#: Where the prepared asset's prims land once referenced.
ASSET_ROOT = "/World/Targets/phantom4"

#: Part name -> thermal target (AI.5, ADR 0138). The asset used with ``--asset phantom4_parts``
#: has one prim per **part**, so a target can finally name hardware instead of a material.
#:
#: What this replaces, and why the old form could not be rescued. Until AI.5 this was
#: ``TARGET_BY_MATERIAL = {"copper": "motor"}`` with everything else on ``airframe`` -- two thermal
#: nodes for an aircraft with four motors, a battery and four propellers. It could not be made
#: finer, because in the source asset a prim is a *material group*: only 25 of 41 prims are more
#: than 90 % one part and those carry just 34.7 % of the area, and the prim containing the battery
#: holds nine parts of which the battery is 26 %. So the geometry was regrouped instead
#: (``prep_asset.py --emit-parts``), and this map keys on the result.
TARGET_BY_PART: dict[str, str] = {
    # The four motors are the hottest surfaces on a flying quadcopter: §6.6's aerial node peaks at
    # +45 K over ambient at full throttle (ADR 0072). Four nodes, not one.
    "motor_front_left": "motor",
    "motor_front_right": "motor",
    "motor_rear_left": "motor",
    "motor_rear_right": "motor",
    # The mounts carry the ESC wiring out of the arms; the ESC node peaks at +30 K.
    "motor_mount_front_left": "esc",
    "motor_mount_front_right": "esc",
    "motor_mount_rear_left": "esc",
    "motor_mount_rear_right": "esc",
    # The pack: +15 K, large mass, and **enclosed by the shell**, so the camera sees it only
    # through what it warms. It exists in the model for the first time as of ADR 0138.
    "battery": "battery",
    # Everything else takes the airframe's own energy balance. The propellers are named separately
    # in the scene because they are 1 mm of ABS moving fastest through their own boundary layer.
    "propeller_front_left": "airframe",
    "propeller_front_right": "airframe",
    "propeller_rear_left": "airframe",
    "propeller_rear_right": "airframe",
}
DEFAULT_TARGET = "airframe"

#: Which way the airframe's nose points **in the asset's own axes**, before the mount rotation.
#: Measured from the archive, and corrected in `AI.5` after the first measurement proved wrong by
#: **28.6 degrees**. The correction matters: this vector is what a yaw of zero means, so an error
#: here flies the whole aircraft crabbed through every frame of every clip.
#:
#: **What the first measurement got wrong.** It read the gimbal camera's offset from the airframe
#: centroid as "-53 mm" and concluded the nose was -Y. That took only the *y* component. The
#: camera body (`camera_static`) and its `Crystal` lens elements sit at offset (-32, -52) mm --
#: bearing **-121.5 deg**, not -90.
#:
#: **What it is measured from now.** The four rotor stations, which are the airframe's own axis of
#: symmetry and are far better conditioned than one small component's centroid. They sit at
#: r = 185 mm (spread 0.6 mm) and bearings -73.73, -163.50, +16.45 and +106.39 deg -- 90 deg
#: apart to within 0.23 deg, which is the asset's own modelling tolerance and four hundred
#: times smaller than the error being corrected. The two front arms bisect at **-118.61 deg**,
#: and the camera agrees with that to
#: 2.9 deg, which is the cross-check. This is the X configuration a Phantom 4 has: the camera
#: points forward between the two front arms, not along one of them.
#:
#: **The old docstring's supporting argument does not hold either**, and it is worth writing down
#: so nobody restores it. It reasoned that DJI puts red status LEDs on the front arms and green on
#: the rear, so a green lamp aft of the centre of mass marks the tail. In *this* asset all four
#: arms carry the same `red_light` material, equal area at each station, and the `Green_light` it
#: relied on is a 0.07 cm2 speck at r = 77 mm near the body -- not an arm LED at all.
#:
#: Read the stations back with `scripts/prep_asset.py --emit-components`; ADR 0138 records the
#: measurement and `tests/unit/test_phantom4_orientation.py` pins it.
NOSE_IN_ASSET = (-0.4789, -0.8779, 0.0)

parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
parser.add_argument("--asset", default="phantom4")
parser.add_argument("--scene", default="configs/scenes/phantom4_pointwise.yaml")
parser.add_argument("--sensor", default="configs/sensors/flir_boson_640_lwir.yaml")
parser.add_argument("--out", default="outputs/phantom4")
parser.add_argument("--frames", type=int, default=96)
parser.add_argument("--fps", type=float, default=12.0)
parser.add_argument(
    "--mission-s",
    type=float,
    default=1666.0,
    help="scene seconds the clip covers; one circuit of the track per clip",
)
parser.add_argument("--laps", type=float, default=1.0, help="circuits of the eight per clip")
# --- framing -------------------------------------------------------------------------------------
# A Phantom 4 is 464 mm tip to tip and this sensor is a 30.7 deg HFOV Boson 640, so the aircraft
# spans 640 * 0.464 / (2 R tan 15.35 deg) pixels: 108 px at 5 m, 271 px at 2 m. The defaults below
# keep the old flypast; bring `--centre-range-m` in to about 2 m for a close pass where a 27 mm
# motor can is ~16 px and per-part temperature is actually legible.
parser.add_argument("--centre-range-m", type=float, default=None, help="track centre distance")
parser.add_argument("--half-width-m", type=float, default=None, help="half-width of the eight")
parser.add_argument("--altitude-low-m", type=float, default=None, help="lowest track altitude")
parser.add_argument("--altitude-high-m", type=float, default=None, help="highest track altitude")
parser.add_argument("--rt-subframes", type=int, default=16)
parser.add_argument("--settle", type=int, default=16)
parser.add_argument("--span", default=None, help="display span 'loC,hiC'; default from the solve")
parser.add_argument("--no-rgb", action="store_true", help="infrared only; no visible companion")
# --- the weather comes from the isaac-weather-fx submodule --------------------------------------
# One weather, both bands: the visible dome, the sun and the moon are authored by weather-fx's own
# SkyEffect, and the infrared band marches the *same* CloudField through `WeatherFxDeck`. Before
# this the two bands ran two cloud models and agreed only as well as two implementations ever do.
# One weather, both *halves* too (ADR 0136): the surfaces are integrated through a series
# synthesised from the same state, so the sun in the frame is the sun that warmed the paint.
parser.add_argument(
    "--weather",
    default=None,
    help="weather-fx regime name (clear, fair_cumulus, broken_cumulus, overcast, haze, rain, "
    "fog, snow, storm), or 'random'. Omit for the project's own dome.",
)
parser.add_argument("--weather-seed", type=int, default=None, help="seed for --weather random")
parser.add_argument("--weather-preset", default=None, help="a weather-fx JSON preset to load")
parser.add_argument(
    "--weather-hour",
    type=float,
    default=None,
    help="override the hour (UTC) of whatever weather was drawn, to film one day twice",
)
# The scene config still names a measured CSV, and `--weather-csv` is how you get it back:
# a validation run wants the day that was actually recorded, not a synthesised one.
parser.add_argument(
    "--weather-csv",
    action="store_true",
    help="keep the scene's measured weather file for the thermal solve, and let weather-fx "
    "drive the sky only",
)
parser.add_argument("--no-overlay", action="store_true", help="bare frames, no readout")
# IG.13: the float32 planes and their sidecar are the *frame*; the videos beside them are the
# look. An 8-bit display PNG has had the AGC, the palette and a 256-level quantisation applied to
# it and none of those invert (ADR 0068), so a clip on its own supports no radiometric claim.
parser.add_argument("--float-format", default="npy", choices=("npy", "exr"))
parser.add_argument(
    "--plane-stride",
    type=int,
    default=8,
    help="write every Nth frame's float32 planes; 0 writes none, and makes no radiometric claim",
)


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


def _mount_matrix(rotation: Any, centre: Any, yaw_deg: float, position: Any) -> Any:
    """The asset's local-to-world 4x4, in USD's row-vector convention.

    ``stage = position + Ryaw @ rotation @ (asset - centre)``: recentre the archive on its own
    centroid, rotate its world frame onto the stage's, yaw it to point where it is going, and put
    it on the track. USD multiplies a row vector on the left, so every matrix here is transposed
    relative to that reading and the translation lives in the fourth row.
    """
    import numpy as np
    from pxr import Gf

    yaw = math.radians(yaw_deg)
    # About +Y, positive from -Z toward +X: the same sense as `FigureEightTrack.yaw_deg`.
    spin = np.asarray(
        [
            [math.cos(yaw), 0.0, -math.sin(yaw)],
            [0.0, 1.0, 0.0],
            [math.sin(yaw), 0.0, math.cos(yaw)],
        ],
        dtype=np.float64,
    )
    linear = (spin @ np.asarray(rotation, dtype=np.float64)).T
    offset = np.asarray(position, dtype=np.float64) - np.asarray(centre, dtype=np.float64) @ linear
    m = np.eye(4, dtype=np.float64)
    m[:3, :3] = linear
    m[3, :3] = offset
    return Gf.Matrix4d(*(float(v) for v in m.reshape(-1)))


def _render(args: Any, usd: pathlib.Path) -> int:  # noqa: PLR0915 - one driver, read top to bottom
    import numpy as np
    import omni.usd
    from pxr import Gf, Usd, UsdGeom

    from irsim.atmosphere.skylight import skylight_for_sensor
    from irsim.config.loader import band_hash, config_hash, load_sensor_config
    from irsim.config.scene import load_scene_config
    from irsim.io.dataset import FrameWriter
    from irsim.io.png import write_png
    from irsim.isp.palette import palette_table, quantise_display
    from irsim.materials.library import MaterialLibrary
    from irsim.materials.mapping import MaterialResolver, load_asset_mapping, load_mapping_rules
    from irsim.materials.table import MaterialTable
    from irsim.pipeline.core import PipelineConfig
    from irsim.radiometry.lut_files import (
        load_band_lut_for_config,
        load_band_response_for_config,
    )
    from irsim.scene import Scene
    from irsim.thermal.weather_fx_series import weather_series_from_state
    from irsim_eval.video import encode_mp4, ffmpeg_available, overlay_readout
    from irsim_isaac.aircraft_pass import look_at_quaternion
    from irsim_isaac.asset_flight import FigureEightTrack, world_frame_to_stage
    from irsim_isaac.pipeline.ir_camera import IrCamera
    from irsim_isaac.pipeline.materials_usd import prim_records
    from irsim_isaac.stage import author_environment
    from irsim_isaac.visible_sky import dome_spec_from_scene
    from irsim_isaac.weather_fx_stage import author_weather_fx_sky, weather_state

    out = REPO / args.out
    out.mkdir(parents=True, exist_ok=True)
    frames_dir = out / "frames"
    frames_dir.mkdir(exist_ok=True)

    # --- the scene, first: its `world_frame:` is what the mount rotation is derived from --------
    sensor = load_sensor_config(REPO / args.sensor)
    band = sensor.sensor.band.band_id
    lut = load_band_lut_for_config(sensor, REPO / "data" / "lut")
    response = load_band_response_for_config(sensor, REPO / "data")
    skylight = skylight_for_sensor(sensor, "lb")
    # The *config* first, not the scene: the weather-fx state is drawn from the site and the
    # clock it names, and the scene is then built around whichever weather won (below).
    scene_config = load_scene_config(REPO / args.scene)
    spec = scene_config.scene
    scene_start = spec.start_utc
    world = spec.world_frame
    mount = world_frame_to_stage(world.up, world.north)
    print(
        f"mount: scene world_frame up={tuple(world.up)} north={tuple(world.north)} "
        f"-> stage +Y up, -Z north"
    )

    track_kwargs = {
        k: v
        for k, v in (
            ("centre_range_m", args.centre_range_m),
            ("half_width_m", args.half_width_m),
            ("altitude_low_m", args.altitude_low_m),
            ("altitude_high_m", args.altitude_high_m),
        )
        if v is not None
    }
    track = FigureEightTrack(**track_kwargs)
    # The camera advances its clock *before* it hands back a frame, so frame i is at
    # (i + 1) * interval and the last one lands exactly on `--mission-s`. Dividing by
    # `frames - 1` instead overshoots by one interval, and the throttle schedule -- which refuses
    # a time past the end of the mission rather than extrapolating one -- is right to object.
    interval_s = args.mission_s / max(args.frames, 1)

    # --- the stage: the renderer's Y-up, with the asset rotated onto it -------------------------
    ctx = omni.usd.get_context()
    ctx.new_stage()
    stage = ctx.get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    stage.DefinePrim("/World", "Xform")
    stage.DefinePrim("/World/Targets", "Xform")
    target = stage.DefinePrim(ASSET_ROOT, "Xform")
    target.GetReferences().AddReference(str(usd))
    mount_op = UsdGeom.Xformable(target)
    mount_op.ClearXformOpOrder()
    transform_op = mount_op.AddTransformOp()

    records = prim_records(stage, root=ASSET_ROOT)
    if not records:
        print(f"the reference produced no prims under {ASSET_ROOT}", file=sys.stderr)
        return 3

    # The archive's own centroid, read with the mount at identity so the number means what it says.
    transform_op.Set(Gf.Matrix4d(1.0))
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default"])
    box = cache.ComputeWorldBound(target).ComputeAlignedRange()
    centre = np.asarray([box.GetMidpoint()[i] for i in range(3)], dtype=np.float64)
    span_m = float(max(box.GetMax()[i] - box.GetMin()[i] for i in range(3)))
    # The archive's nose axis, carried through the mount, is what a yaw of zero has to mean.
    nose_yaw_deg = float(
        math.degrees(
            math.atan2(*(lambda v: (v[0], -v[2]))(mount @ np.asarray(NOSE_IN_ASSET, float)))
        )
    )
    print(f"asset: centroid {centre.round(3).tolist()}, {span_m:.3f} m across")

    # --- the environment ------------------------------------------------------------------------
    weather_sky = None
    if args.weather or args.weather_preset:
        # weather-fx owns the sky. `author_weather_fx_sky` writes the dome, the sun and the moon
        # into the session layer through the extension's own effect -- the same code its UI panel
        # drives -- and hands back the cloud field for the infrared band to march.
        regime = None if args.weather in (None, "random") else args.weather
        state = weather_state(
            seed=args.weather_seed if (args.weather == "random" or regime) else args.weather_seed,
            regime=regime,
            preset_json=args.weather_preset,
            latitude_deg=spec.site.latitude_deg,
            longitude_deg=spec.site.longitude_deg,
            date_utc=spec.start_utc.strftime("%Y-%m-%d"),
            hour_utc=(
                args.weather_hour
                if args.weather_hour is not None
                else spec.start_utc.hour + spec.start_utc.minute / 60.0
            ),
        )
        # `--weather-hour` films this scene at another hour, so the *scene's* clock moves with
        # it. Left where the config put it, the frame would be lit at one hour and its surfaces
        # integrated to another, and nothing downstream can tell.
        scene_start = dt.datetime.strptime(state.sky.date_utc, "%Y-%m-%d").replace(
            tzinfo=dt.timezone.utc
        ) + dt.timedelta(hours=float(state.sky.hour_utc))
        if scene_start != spec.start_utc:
            spec = spec.model_copy(update={"start_utc": scene_start})
            scene_config = scene_config.model_copy(update={"scene": spec})
            print(f"  scene clock moved to {scene_start:%Y-%m-%d %H:%M} UTC")

        weather_sky = author_weather_fx_sky(stage, state, texture_dir=out)
        print(f"weather-fx: {weather_sky.describe()}")
        stats = weather_sky.stats()
        print(
            f"  sun {stats['sun_elevation_deg']:+.1f} deg / {stats['sun_azimuth_deg']:.0f} deg, "
            f"moon {stats['moon_elevation_deg']:+.1f} deg {stats['moon_phase']} "
            f"({stats['moon_lux']:.3f} lx), daylight {stats['daylight']:.3f}"
        )
        if weather_sky.deck is not None:
            print(
                f"  cloud: {weather_sky.deck.cover * 100:.0f} % cover, base "
                f"{weather_sky.deck.base_m:.0f} m, top {weather_sky.deck.top_m:.0f} m, "
                f"tau {weather_sky.deck.optical_depth:.0f} -- marched by BOTH bands"
            )

    # --- the weather the *surfaces* see, which must be the weather the sky shows ----------------
    # CLAUDE.md #6 is about one object reaching both consumers; this is the other half of it. If
    # weather-fx is drawing an overcast June morning and the solver is integrating a March day out
    # of a CSV, both halves are individually plausible and the frame is simply wrong. So unless
    # `--weather-csv` asks for the measured file, the series is synthesised from the same state.
    weather_override = None
    if weather_sky is not None and not args.weather_csv:
        weather_override = weather_series_from_state(
            weather_sky.state,
            # The cover the field actually built, not the one that was asked for: a thresholded
            # noise field lands where it lands, and the sky emissivity should use what the camera
            # is looking at.
            measured_cover=None if weather_sky.deck is None else weather_sky.deck.cover,
        )
        sample = weather_override.at_datetime(scene_start)
        print(
            f"  surface weather from weather-fx: air {sample.t_air_k - 273.15:.1f} C, "
            f"RH {sample.rh_fraction * 100:.0f} %, wind {sample.wind_speed_m_s:.1f} m/s, "
            f"DNI {sample.dni_w_m2:.0f} + DHI {sample.dhi_w_m2:.0f} W/m2, "
            f"visibility {sample.visibility_m / 1000.0:.1f} km"
        )
        if sample.precip_mm_h > 0.0:
            print(f"    precipitation {sample.precip_mm_h:.2f} mm/h")
    elif weather_sky is not None:
        print(f"  surface weather from {spec.weather_file} (--weather-csv)")

    scene = Scene.from_config(
        scene_config,
        {band: lut},
        responses={band: response},
        quantity="lb",
        weather_override=weather_override,
        skylights={band: skylight},
    )

    # --- or this project's own dome, for the companion frame only (ADR 0073) ---------------------
    dome = None
    if weather_sky is None and not args.no_rgb:
        # Baked once, at mid-mission: the sun moves 7 degrees across a 28-minute clip, which is
        # below what a baked latlong texture would show, and re-baking it 96 times would cost more
        # than the render. `heading_deg=0` is not a default here -- the mount put the scene's own
        # north on the stage's -Z, which is exactly what heading zero means.
        dome = dome_spec_from_scene(
            scene,
            t_rel_s=0.5 * args.mission_s,
            heading_deg=0.0,
            camera_height_m=float(track.observer_m[1]),
        )
        print(
            f"environment: sun {dome.sun_elevation_deg:.1f} deg elevation, "
            f"{dome.sun_azimuth_deg:.1f} deg azimuth, turbidity {dome.turbidity:.2f}"
        )
    if weather_sky is None:
        try:
            author_environment(stage, dome, out / "env_dome.exr", 512)
        except Exception as exc:  # noqa: BLE001 - a dark companion must not stop the render
            print(f"environment: {type(exc).__name__}: {exc}", file=sys.stderr)

    # --- materials, through the per-asset map (ADR 0128) ----------------------------------------
    library = MaterialLibrary.load()
    table = MaterialTable.from_library(library, band)
    asset_map = load_asset_mapping(args.asset, known_materials=library.names)
    resolver = MaterialResolver(
        load_mapping_rules(known_materials=library.names), list(table.names), asset=asset_map
    )
    resolutions = resolver.resolve_all(records)
    unmapped = [r.path for r in resolutions if not r.mapped]
    print(f"materials: {len(resolutions) - len(unmapped)}/{len(resolutions)} prims mapped")
    for path in unmapped[:5]:
        print(f"  UNMAPPED (renders magenta): {path}")

    # --- every prim needs a thermal node, or the camera refuses it ------------------------------
    # A part-split asset names its prims after parts, so the leaf name is the key. A prim the map
    # does not name falls to `airframe`, which is what every shell, arm and leg wants anyway.
    prim_to_target = {
        record.path: TARGET_BY_PART.get(record.path.rsplit("/", 1)[-1], DEFAULT_TARGET)
        for record in records
    }
    by_node: dict[str, int] = {}
    for value in prim_to_target.values():
        by_node[value] = by_node.get(value, 0) + 1
    summary = ", ".join(f"{n} prim(s) on `{k}`" for k, n in sorted(by_node.items()))
    print(f"thermal nodes: {summary}")
    if set(prim_to_target.values()) == {DEFAULT_TARGET}:
        print(
            "  NOTE: every prim is on the default node. With --asset phantom4 that is expected "
            "(its prims are material groups); use --asset phantom4_parts for per-part heat.",
            file=sys.stderr,
        )

    # --- the camera: on the ground at the observer, slewing to hold the aircraft ----------------
    eye = np.asarray(track.observer_m, dtype=np.float64)
    camera = UsdGeom.Camera.Define(stage, "/World/IrCamera")
    camera_xform = UsdGeom.Xformable(camera)
    camera_xform.ClearXformOpOrder()
    camera_xform.AddTranslateOp().Set(Gf.Vec3d(*(float(v) for v in eye)))
    aim_op = camera_xform.AddOrientOp()
    camera.GetClippingRangeAttr().Set(Gf.Vec2f(0.01, 1.0e5))

    fpa = sensor.sensor.fpa
    hfov_rad = 2.0 * math.atan(
        0.5 * fpa.width * fpa.pitch_um * 1e-3 / sensor.sensor.optics.focal_length_mm
    )

    def aim_at(position: Any) -> None:
        q = look_at_quaternion(np.asarray(position, dtype=np.float64) - eye)
        aim_op.Set(Gf.Quatf(float(q[0]), Gf.Vec3f(*(float(v) for v in q[1:]))))

    def place(phase: float) -> tuple[Any, float, float]:
        """Put the aircraft on the track at ``phase`` and slew the camera onto it.

        Returns the position, the slant range and the **aspect**: the angle between where the nose
        points and where the observer is, which is what decides how much of the aircraft the
        camera can see. 0 is head-on, +-90 broadside, 180 tail-on.
        """
        position = track.position_m(phase)
        heading = float(track.yaw_deg(phase))
        transform_op.Set(_mount_matrix(mount, centre, heading - nose_yaw_deg, position))
        aim_at(position)
        to_observer = eye - position
        bearing = math.degrees(math.atan2(to_observer[0], -to_observer[2]))
        aspect = (heading - bearing + 180.0) % 360.0 - 180.0
        return position, float(track.range_m(phase)), aspect

    phases = (np.arange(args.frames, dtype=np.float64) / args.frames) * args.laps
    ranges = track.range_m(phases % 1.0)
    px_across = fpa.width * span_m / (2.0 * ranges * math.tan(0.5 * hfov_rad))
    print(
        f"camera: {fpa.width}x{fpa.height}, {math.degrees(hfov_rad):.1f} deg HFOV, on the ground "
        f"at {eye.round(2).tolist()}\n"
        f"track: range {ranges.min():.1f}..{ranges.max():.1f} m, "
        f"elevation {track.elevation_deg(phases % 1.0).min():.1f}.."
        f"{track.elevation_deg(phases % 1.0).max():.1f} deg, "
        f"aircraft {px_across.min():.0f}..{px_across.max():.0f} px across"
    )
    place(0.0)

    pipeline = PipelineConfig.from_sensor(
        sensor, table, lut, sky=scene.sky_models[band], atmosphere=scene.layered
    )
    cam = IrCamera(
        sensor,
        scene,
        pipeline=pipeline,
        prim_to_target=prim_to_target,
        resolutions=resolutions,
        camera_path="/World/IrCamera",
        stage=stage,
        up_axis="Y",
        capture_rgb=not args.no_rgb,
        # The aircraft really moves between frames, so the synthesised `motion_px` of IG.6 has a
        # true premise here -- unlike `render_quad_outbound`, where the target is static and only
        # the mount slews.
        moving_prim_paths=[ASSET_ROOT],
        heading_deg=0.0,
        # The infrared band marches the same cloud the dome was baked from. `None` leaves the
        # background clear, which is what a cloudless weather means.
        weather_fx_clouds=None if weather_sky is None else weather_sky.deck,
        frame_period_s=interval_s,
        strict_patch_coverage=False,
        strict_materials=True,
        strict_thermal_nodes=True,
    )
    cam.open(settle_frames=args.settle, rt_subframes=args.rt_subframes)

    # IG.13: every run-constant field bound once, so the loop body cannot quietly disagree with
    # itself frame to frame. `--plane-stride` thins what reaches disk without thinning the render,
    # and records itself in every sidecar, so a gap in the numbering is not a render that died.
    writer = FrameWriter(
        directory=out,
        config_hash=config_hash(sensor),
        band_hash=band_hash(sensor),
        quantity=pipeline.quantity,
        float_format=args.float_format,
        start_utc=scene.weather.start_utc,
        stride=args.plane_stride,
        metadata={
            "scene": pathlib.Path(args.scene).name,
            "sensor": pathlib.Path(args.sensor).name,
            "asset": args.asset,
            "interval_s": interval_s,
            "track": "lemniscate, one circuit per clip",
        },
    )
    planes_written = 0

    planes: list[Any] = []
    hits: list[Any] = []
    rgbs: list[Any] = []
    rows: list[dict[str, Any]] = []
    started = time.time()
    for index in range(args.frames):
        phase = float(phases[index] % 1.0)
        position, slant, aspect = place(phase)
        # Move the mount, then tell the camera it moved: every ray direction, sky elevation and
        # view cosine is built from a cached pose, and a camera that slews without refreshing
        # renders geometry from here and radiometry from where it used to be.
        cam.refresh_pose()

        outputs = cam.get_outputs(rt_subframes=args.rt_subframes)
        if outputs.apparent_t is None:
            raise RuntimeError("the sensor config produced no apparent_t plane")
        t_app = np.asarray(outputs.apparent_t, dtype=np.float32)
        planes.append(t_app)

        # `material_id` lives on the *supersampled* grid (ADR 0014: ids are never filtered, so
        # anti-aliasing happens by supersampling them and filtering radiance afterwards). Fold it
        # down with `any`, because a display pixel showing any aircraft at all is an aircraft
        # pixel -- averaging ids would invent a material that is not in the scene.
        last = cam.last_frame
        mat = None if last is None else getattr(last, "material_id", None)
        if mat is None:
            hit = t_app.reshape(-1)
        else:
            mask = np.asarray(mat) > 0
            k = mask.shape[0] // t_app.shape[0]
            if k > 1:
                mask = mask.reshape(t_app.shape[0], k, t_app.shape[1], k).any(axis=(1, 3))
            hit = t_app[mask]
        hits.append(hit)

        rgb = None if last is None else getattr(last, "rgb", None)
        if not args.no_rgb and rgb is None and index == 0:
            print(f"companion frame: {cam.rgb_problem}", file=sys.stderr)
        rgbs.append(None if rgb is None else np.ascontiguousarray(np.asarray(rgb)[..., :3]))

        temperatures = {k: float(v) for k, v in cam.bridge.temperatures().items()}
        node_k = {k: round(v, 3) for k, v in temperatures.items()}
        rows.append(
            {
                "frame": index,
                "t_rel_s": round(float(cam.t_rel_s), 2),
                "range_m": round(slant, 3),
                "aspect_deg": round(aspect, 2),
                "elevation_deg": round(float(track.elevation_deg(phase)), 2),
                "px_across": round(float(px_across[index]), 1),
                "target_px": int(hit.size),
                "target_min_c": float(hit.min() - 273.15) if hit.size else None,
                "target_max_c": float(hit.max() - 273.15) if hit.size else None,
                "node_c": {k: round(v - 273.15, 2) for k, v in temperatures.items()},
                "position_m": [round(float(v), 3) for v in position],
            }
        )
        if writer.wants(index):
            writer.write(
                outputs,
                frame_index=index,
                t_s=cam.last_frame_t_s,
                extra_planes={} if rgbs[-1] is None else {"rgb": rgbs[-1]},
                extra_metadata={
                    "range_m": round(slant, 3),
                    "aspect_deg": round(aspect, 2),
                    "elevation_deg": round(float(track.elevation_deg(phase)), 2),
                    "node_temperatures_k": node_k,
                },
            )
            planes_written += 1
        if index % 12 == 0 or index == args.frames - 1:
            r = rows[-1]
            print(
                f"  frame {index:4d}  T+{r['t_rel_s']:7.0f}s  R={slant:5.1f} m  "
                f"{r['px_across']:5.1f} px  aircraft {r['target_px']:6d} px  "
                f"{r['target_min_c']:6.1f} .. {r['target_max_c']:6.1f} C  "
                f"motor {r['node_c'].get('motor', float('nan')):5.1f} C"
            )
    render_s = time.time() - started
    cam.close()
    per_frame = render_s / max(args.frames, 1)
    print(f"rendered {args.frames} frames in {render_s:.0f} s ({per_frame:.1f} s/frame)")

    # --- the display span, from the aircraft's own pixels ---------------------------------------
    # A span taken over the whole frame is a span over sky: this scene is ~98 % background, so the
    # percentiles land inside it and the target saturates to one flat white shape -- a picture of
    # the sky's noise, not of an aircraft. Fixed across the clip, not per frame: a per-frame AGC
    # would rescale from each histogram and cancel the warm-up being filmed.
    if args.span:
        lo, hi = (float(v) + 273.15 for v in args.span.split(","))
    else:
        target_pixels = np.concatenate([h for h in hits if h.size])
        lo = float(np.percentile(target_pixels, 1.0))
        hi = float(np.percentile(target_pixels, 99.0))
    print(f"display span: {lo - 273.15:.1f} .. {hi - 273.15:.1f} C, white-hot grayscale")
    gray = palette_table("gray")

    for index, plane in enumerate(planes):
        row = rows[index]
        eight = gray[quantise_display((plane - lo) / max(hi - lo, 1e-6))]
        minutes, seconds = divmod(int(row["t_rel_s"]), 60)
        readout = overlay_readout(
            eight,
            [
                f"T+{minutes:02d}:{seconds:02d}   range {row['range_m']:5.1f} m   "
                f"span {row['px_across']:5.1f} px   el {row['elevation_deg']:4.1f} deg",
                f"aspect {row['aspect_deg']:+6.1f} deg   "
                + "   ".join(f"{k} {v:5.1f}C" for k, v in sorted(row["node_c"].items())),
                f"1 frame / {interval_s:.0f} s   LWIR apparent temperature, gray white-hot",
            ],
            {k: v + 273.15 for k, v in row["node_c"].items()},
            (lo, hi),
            bare=args.no_overlay,
            palette=gray,
        )
        write_png(frames_dir / f"ir_{index:05d}.png", readout)
        if rgbs[index] is not None:
            write_png(frames_dir / f"rgb_{index:05d}.png", rgbs[index])
            pair = np.concatenate([readout, np.asarray(rgbs[index])], axis=1)
            write_png(frames_dir / f"pair_{index:05d}.png", np.ascontiguousarray(pair))

    videos: dict[str, str] = {}
    if ffmpeg_available():
        for kind in ("ir", "rgb", "pair"):
            if not any(frames_dir.glob(f"{kind}_*.png")):
                continue
            path = out / f"{args.asset}_{kind}.mp4"
            encode_mp4(str(frames_dir / f"{kind}_*.png"), path, fps=args.fps)
            videos[kind] = str(path)
            print(f"wrote {path}")
    else:
        print("ffmpeg not available: stills written, no video", file=sys.stderr)

    summary = {
        "asset": args.asset,
        "usd": str(usd),
        "prims": len(records),
        "mapped": len(resolutions) - len(unmapped),
        "unmapped": len(unmapped),
        "frames": args.frames,
        "planes_written": planes_written,
        "config_hash": config_hash(sensor),
        "band_hash": band_hash(sensor),
        "float_format": args.float_format,
        "plane_stride": args.plane_stride,
        "interval_s": interval_s,
        "seconds_per_frame": round(render_s / max(args.frames, 1), 2),
        "mount_rotation": [[float(v) for v in r] for r in mount],
        "track": {
            "shape": "lemniscate of Bernoulli, one circuit per clip",
            "observer_m": list(track.observer_m),
            "centre_range_m": track.centre_range_m,
            "half_width_m": track.half_width_m,
            "altitude_m": [track.altitude_low_m, track.altitude_high_m],
            "range_m": [float(ranges.min()), float(ranges.max())],
            "px_across": [float(px_across.min()), float(px_across.max())],
        },
        "display_span_c": [lo - 273.15, hi - 273.15],
        "videos": videos,
        "rows": rows,
        "temperature_granularity": (
            "per prim. The scene's 234,923 mesh cells are solved but IrCamera takes planar "
            "SurfaceBindings only; wiring MeshPointBridge into the camera is AI.2's remainder."
        ),
        "weather_fx": (
            None
            if weather_sky is None
            else {"state": weather_sky.state.to_dict(), **weather_sky.stats()}
        ),
        # ADR 0136: a reader has to be able to tell a measured day from a synthesised one, and
        # the file format cannot say it -- both are a WeatherSeries by the time anything reads
        # them. The state above is what makes a synthesised day reproducible.
        "weather_source": (
            {
                "surface": "measured CSV: " + spec.weather_file,
                "sky": "weather-fx" if weather_sky is not None else "irsim dome (ADR 0073)",
            }
            if weather_override is None
            else {
                "surface": "synthesised from the weather-fx state (ADR 0136)",
                "sky": "weather-fx, the same state",
                "content_hash": weather_override.content_hash,
            }
        ),
        "companion_frame": (
            "not captured (--no-rgb)"
            if args.no_rgb
            else (cam.rgb_problem or "captured with the infrared frame, same pixel grid")
        ),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {out}/summary.json")
    stack = np.stack(planes)
    print(f"frame span across the clip: {stack.min() - 273.15:.1f} .. {stack.max() - 273.15:.1f} C")
    return 0


if __name__ == "__main__":
    sys.exit(main())
