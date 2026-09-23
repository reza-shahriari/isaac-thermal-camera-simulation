#!/usr/bin/env python3
"""Roadmap PT.9 / IG.2: film a quadrotor going away from the camera, point-wise (ADR 0123).

A ground camera watches one quadrotor from 12 m to 150 m while the sun and the throttle heat it
together. The airframe is **not one temperature**: the deck, the belly and two arms each carry a
field, so a single prim shows a gradient -- the deck's own shadow lies across the inner half of
each arm, 27 K of it -- and the clip shows that gradient survive, blur and finally collapse into
a handful of pixels as the target recedes. That collapse is the thing the aerial lane exists to
measure, and it cannot be seen in a frame that gives the whole airframe one number.

**Sun and throttle, both.** The sun reaches each cell through its own beam, its own occluders and
its own sky view; the throttle heats the motors, the speed controllers and the pack, and drives
the skin's convective speed (ADR 0109) -- which is what collapses the deck's 29 K excess over air
to 14 K once the aircraft is moving, and lets it come back on landing.

**The camera moves and the aircraft does not.** See :mod:`irsim_isaac.quad_outbound`: an occluder
in a moving frame is refused by the thermal core, so flying the aircraft would cost this target
its self-shadowing, which is the whole point-wise signature. Relative motion is all a camera sees.

**Sky only.** No ground plane and no sky geometry (ADR 0060); the boresight sits 20 deg up, which
is more than half the vertical field, so the horizon is never in frame. The driver refuses to
render rather than quietly putting the analytic ground into the bottom of a sky-target picture.

Run with the Isaac interpreter (docs/decisions/0002), after `make luts`:
    IRSIM_GPU=0 python.sh scripts/render_quad_outbound.py --frames 300 --rgb
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

#: The airframes this driver can fly, and the scene and track each one belongs with. Held as
#: plain data so the defaults resolve before Kit boots; the geometry is imported inside `main`.
#:
#: **The track is part of the airframe, not a global.** A Phantom is a fifth of the heavy-lift
#: frame across, so flying it to 150 m would put it under three pixels for most of the clip --
#: the honest picture of a small drone at long range, and not one anybody can watch. Each
#: aircraft therefore carries the range band at which it is the subject of the frame.
AIRFRAMES: dict[str, dict[str, Any]] = {
    "heavy_lift": {
        "scene": "quad_outbound_pointwise.yaml",
        "near_m": 12.0,
        "far_m": 150.0,
        "close_near_m": 2.6,
        "close_far_m": 4.2,
        "out": "outputs/quad_outbound",
    },
    "phantom3": {
        "scene": "phantom3_outbound_pointwise.yaml",
        "near_m": 6.0,
        "far_m": 80.0,
        "close_near_m": 1.4,
        "close_far_m": 2.2,
        "out": "outputs/phantom3_outbound",
    },
}

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--airframe",
    default="phantom3",
    choices=sorted(AIRFRAMES),
    help="which aircraft to fly. Each carries its own scene config and range band",
)
parser.add_argument("--out", default=None)
parser.add_argument("--frames", type=int, default=300)
parser.add_argument("--sensor", default=str(REPO / "configs/sensors/flir_boson_640_lwir.yaml"))
parser.add_argument("--scene", default=None)
parser.add_argument(
    "--interval-s",
    type=float,
    default=6.0,
    help="scene seconds per captured frame. frames x interval must fit inside the flight profile",
)
parser.add_argument("--fps", type=float, default=30.0, help="playback rate of the encoded video")
parser.add_argument("--near-m", type=float, default=None)
parser.add_argument("--far-m", type=float, default=None)
# **The close-up is a different measurement, not a nicer picture.** The outbound track answers
# "when does this target stop being detectable"; at 80 m the aircraft is nine pixels and its
# per-part structure is gone by construction, which is the honest answer and tells you nothing
# about the parts. This holds the aircraft filling the frame for the whole mission so the only
# thing changing is temperature -- the radiometric read, with the colour bar beside it.
parser.add_argument(
    "--close-up",
    action="store_true",
    help="fill the frame with the aircraft and film the temperatures instead of the range",
)
parser.add_argument(
    "--elevation-deg",
    type=float,
    default=None,
    help="boresight elevation. Default: irsim_isaac.quad_outbound.AIM_ELEVATION_DEG",
)
parser.add_argument("--settle", type=int, default=16)
parser.add_argument("--rt-subframes", type=int, default=8)
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
parser.add_argument("--no-rotors", action="store_true", help="omit the propeller veils (ADR 0081)")
parser.add_argument(
    "--no-fields",
    action="store_true",
    help="the pre-ADR-0087 picture: one temperature per prim, for the side-by-side comparison",
)
parser.add_argument(
    "--lenient-coverage",
    action="store_true",
    help="warn instead of raising when a patch does not cover every pixel of its prim",
)
parser.add_argument(
    "--span-c",
    type=float,
    nargs=2,
    default=None,
    metavar=("LO", "HI"),
    help="fixed apparent-temperature span, Celsius. Default: spanned on the target's own cells",
)
parser.add_argument("--palette", default=None, help="default: the sensor ISP's own palette")
parser.add_argument("--no-flat-field", action="store_true")
# **Cloud is on by default** (ADR 0076). One `SkyFixedCloud` is seeded from the scene's own
# weather and sampled by BOTH the infrared background, per ray, and the visible dome's texture,
# so the pair cannot show cloud in different parts of the sky. On a weather file whose cloud
# fraction is near zero this costs nothing and draws nothing; on a broken sky it is the clutter
# a sky-background detector actually has to live with.
parser.add_argument("--cloud-seed", type=int, default=7)
parser.add_argument("--no-cloud", action="store_true", help="a bare sky, for the comparison")
# `generate_sky_cloud` defaults to a half-degree grid, which is the whole sky at a resolution
# suited to a wide survey. Through a 31 x 25 degree field at 0.86 mrad per pixel, half a degree
# is **ten pixels**, and the coverage mask reads as blocks rather than as cloud. A sixth of a
# degree puts a grid cell at about three pixels, which is under the PSF and so invisible as
# structure. It costs 4.7 MB and one second.
parser.add_argument("--cloud-cells-per-deg", type=float, default=6.0)
parser.add_argument("--no-overlay", action="store_true", help="no burnt-in readout")
parser.add_argument("--keep-frames", action="store_true", help="keep the PNG sequence")
parser.add_argument("--integration-ms", type=float, default=None)

args = parser.parse_args()

_AIRFRAME = AIRFRAMES[args.airframe]
if args.scene is None:
    args.scene = str(REPO / "configs" / "scenes" / _AIRFRAME["scene"])
if args.out is None:
    args.out = _AIRFRAME["out"]
if args.close_up:
    # Sized so the widest extent fills most of the frame at both ends: a slow push out that
    # still never lets the aircraft get small. The exact pair is checked against the sensor
    # below, because a longer lens would overflow the frame and a shorter one would not fill it.
    if args.near_m is None:
        args.near_m = _AIRFRAME["close_near_m"]
    if args.far_m is None:
        args.far_m = _AIRFRAME["close_far_m"]
    if args.out == _AIRFRAME["out"]:
        args.out = _AIRFRAME["out"] + "_closeup"
if args.near_m is None:
    args.near_m = _AIRFRAME["near_m"]
if args.far_m is None:
    args.far_m = _AIRFRAME["far_m"]
if args.no_cloud:
    args.cloud_seed = None

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
    from irsim_eval.video import encode_mp4, ffmpeg_available, overlay_readout
    from irsim_isaac.display_span import DisplaySpan, span_from_dn16
    from irsim_isaac.pipeline.illumination_isaac import SceneIllumination
    from irsim_isaac.pipeline.ir_camera import IrCamera
    from irsim_isaac.pipeline.materials_usd import prim_records
    from irsim_isaac.pipeline.point_bridge import bindings_from_scene
    from irsim_isaac.quad_outbound import (
        AIM_ELEVATION_DEG,
        OutboundTrack,
        build_quad_outbound,
    )

    if args.airframe == "phantom3":
        from irsim_isaac.phantom3 import PHANTOM_3 as PARTS
        from irsim_isaac.phantom3 import TIP_TO_TIP_M as SPAN_M
        from irsim_isaac.phantom3 import rotor_mounts

        # A Phantom's props nearly touch, so tip to tip is what a camera sees of a flying one --
        # 589 mm against the 350 mm diagonal DJI quotes. The readout says which it is quoting.
        span_label = "prop tip to tip"
    else:
        from irsim_isaac.quad_outbound import POINTWISE_QUAD as PARTS
        from irsim_isaac.quad_outbound import TIP_TO_TIP_M as SPAN_M
        from irsim_isaac.quad_outbound import rotor_mounts

        # Tip to tip for this one too, so the two aircraft are quoted on the same measurement.
        # Its motors are 0.84 m apart -- a plus-configuration frame, not an X.
        span_label = "prop tip to tip"
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
    quantity = spec.quantity
    lut = load_band_lut_for_config(sensor, REPO / "data" / "lut")
    response = load_band_response_for_config(sensor, REPO / "data")
    skylight = skylight_for_sensor(sensor, quantity)

    def load_scene() -> Scene:
        return Scene.from_file(
            args.scene,
            {spec.band.band_id: lut},
            responses={spec.band.band_id: response},
            quantity=quantity,
            skylights={spec.band.band_id: skylight},
        )

    scene = load_scene()

    span_s = args.frames * args.interval_s
    track = OutboundTrack(
        near_m=args.near_m,
        far_m=args.far_m,
        duration_s=span_s,
        elevation_deg=AIM_ELEVATION_DEG if args.elevation_deg is None else args.elevation_deg,
        # A wider walk round the aircraft when it fills the frame: at 80 m the aspect hardly
        # matters, but at 2 m which side the sun is on is most of the picture.
        azimuth_sweep_deg=140.0 if args.close_up else 55.0,
    )

    ifov_mrad = 1e3 * spec.fpa.pitch_um * 1e-3 / spec.optics.focal_length_mm
    vfov_deg = math.degrees(2.0 * math.atan(0.5 * spec.fpa.height * ifov_mrad * 1e-3))
    hfov_deg = math.degrees(2.0 * math.atan(0.5 * spec.fpa.width * ifov_mrad * 1e-3))
    # **Refused, not clipped.** There is no ground plane on this stage, so ground in the picture
    # would be ADR 0060's analytic T_ground painted across the bottom of a sky-target frame -- a
    # horizon that is not a horizon, in the one scene whose whole claim is that it is sky-only.
    if track.sees_horizon(vfov_deg):
        print(
            f"this camera's {vfov_deg:.1f} deg vertical field reaches the horizon from a "
            f"{track.elevation_deg:.1f} deg boresight. Raise --elevation-deg above "
            f"{0.5 * vfov_deg:.1f} deg, or use a longer lens.",
            file=sys.stderr,
        )
        return 1

    print(
        f"\n{spec.name}: {spec.fpa.width}x{spec.fpa.height}, {hfov_deg:.1f} x {vfov_deg:.1f} deg, "
        f"{args.frames} frames every {args.interval_s:g} s = {span_s:.0f} s of flight, "
        f"played at {args.fps:g} fps -> {args.frames / args.fps:.1f} s of video"
    )
    if args.close_up:
        # Fill, and fit. The widest extent at the near end must sit inside the frame with a
        # margin, and at the far end must still be most of it -- otherwise this is an outbound
        # clip wearing a close-up's name.
        fills = track.pixels_across(0.0, SPAN_M, ifov_mrad) / spec.fpa.width
        keeps = track.pixels_across(span_s, SPAN_M, ifov_mrad) / spec.fpa.width
        print(
            f"close-up: the aircraft is {fills:.0%} of the frame width at {track.near_m:g} m "
            f"and {keeps:.0%} at {track.far_m:g} m"
        )
        if fills > 0.92:
            print(
                f"the aircraft overflows the frame at {track.near_m:g} m ({fills:.0%} of the "
                "width). Raise --near-m, or use a shorter lens.",
                file=sys.stderr,
            )
            return 1
        if keeps < 0.25:
            print(
                f"the aircraft is only {keeps:.0%} of the frame by {track.far_m:g} m, which is "
                "an outbound clip and not a close-up. Lower --far-m.",
                file=sys.stderr,
            )
            return 1

    print(
        f"track: {track.near_m:.0f} m -> {track.far_m:.0f} m, boresight "
        f"{track.elevation_deg:.0f} deg up ({track.elevation_deg - 0.5 * vfov_deg:.1f} deg of "
        f"margin to the horizon), target {track.target_altitude_m(0.0):.0f} m -> "
        f"{track.target_altitude_m(span_s):.0f} m above the camera"
    )
    print(
        f"{args.airframe}: {SPAN_M:.3f} m {span_label}, "
        f"{track.pixels_across(0.0, SPAN_M, ifov_mrad):.0f} px at the start -> "
        f"{track.pixels_across(span_s, SPAN_M, ifov_mrad):.1f} px at the end"
    )

    cloud = None
    if args.cloud_seed is not None and scene.environment is not None:
        fraction = float(scene.weather.at(scene.t0_s).cloud_fraction)
        per_deg = max(1.0, float(args.cloud_cells_per_deg))
        cloud = generate_sky_cloud(
            scene.environment.clouds.beta,
            fraction,
            int(args.cloud_seed),
            n_elevation=int(round(90.0 * per_deg)),
            n_azimuth=int(round(360.0 * per_deg)),
        )
        print(
            f"cloud: seed {args.cloud_seed}, covering {cloud.fraction:.1%} of the sky at "
            f"beta = {scene.environment.clouds.beta} (from the shared weather's {fraction:.2f}), "
            f"{per_deg:g} cells/deg = {1e3 / per_deg / ifov_mrad * 1e-3 * 17.45:.1f} px per cell; "
            "the infrared background samples it per ray and the dome bakes the same object"
        )
        if fraction < 0.1:
            print(
                f"  note: this scene's weather carries only {fraction:.2f} of cloud, so a "
                f"{hfov_deg:.0f} x {vfov_deg:.0f} deg field will rarely contain any",
                file=sys.stderr,
            )
    dome = None
    if not args.no_dome:
        # heading 0: the scene's `world_frame:` puts north on the stage's -Z, so the dome, the sun
        # light and the scene's own solar geometry are all in the one frame.
        dome = dome_spec_from_scene(scene, cloud=cloud, heading_deg=0.0)
        print(
            f"environment: sun {dome.sun_elevation_deg:.1f} deg elevation, "
            f"{dome.sun_azimuth_deg:.1f} deg azimuth, turbidity {dome.turbidity:.2f}"
        )

    stage = build_quad_outbound(
        track=track,
        parts=PARTS,
        span_m=SPAN_M,
        dome=dome,
        dome_texture_path=out_dir / "env_dome.exr",
    )
    if stage.errors:
        print(f"stage errors: {stage.errors}", file=sys.stderr)
    missing = set(stage.thermal_nodes()) - set(scene.targets)
    if missing:
        print(f"scene has no solver for: {sorted(missing)}", file=sys.stderr)
        return 1

    table = MaterialTable.from_library(MaterialLibrary.load(), spec.band.band_id)
    resolver = MaterialResolver(load_mapping_rules(), list(table.names))
    resolutions = resolver.resolve_all(prim_records(root="/World/Targets"))
    unresolved = [r.path for r in resolutions if not r.mapped]
    if unresolved:
        print(f"unmapped prims: {unresolved}", file=sys.stderr)

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

    if not args.no_chain:
        if pipeline.calibration is None:
            print(
                f"{spec.name}: no M9 sensor chain -- a photon FPA has no radiometric calibration "
                "to convert the NUC residual into DN (ADR 0056), and this camera is shutterless.",
                file=sys.stderr,
            )
        else:
            from irsim.pipeline.sensor_chain import attach_sensor_chain

            pipeline = attach_sensor_chain(pipeline, scene.weather, t0_s=scene.t0_s)

    surface_tilts = {srf.name: float(srf.tilt_deg) for srf in scene.spec.thermal.surfaces}
    bindings = [] if args.no_fields else bindings_from_scene(scene)
    if args.no_fields:
        print("  --no-fields: rendering the pre-ADR-0087 picture, one temperature per prim")
    else:
        print(
            "  fields: "
            + ", ".join(
                f"{name} {scene.patches[name].n_u}x{scene.patches[name].n_v} "
                f"at {scene.patches[name].du_m * 1e3:.0f} mm"
                for name in sorted(scene.patches)
            )
        )

    camera = IrCamera(
        sensor,
        scene,
        pipeline=pipeline,
        prim_to_target=stage.prim_to_target,
        resolutions=resolutions,
        camera_path=stage.camera_path,
        surface_fields=bindings,
        capture_rgb=args.rgb,
        illumination=SceneIllumination.for_camera(
            sensor, scene, pipeline.quantity, heading_deg=0.0
        ),
        heading_deg=0.0,
        strict_materials=False,
        strict_thermal_nodes=False,
        strict_patch_coverage=not args.lenient_coverage,
        cloud_seed=args.cloud_seed,
        rotor_mounts=None if args.no_rotors else rotor_mounts(0.0),
        # IG.6's `motion_px` is deliberately not requested. The aircraft is static in the stage
        # and the camera slews; the tracker reads prim transforms, so it would report zero motion
        # for a target that is in fact crossing the frame, and a zero smear asserted from the
        # wrong premise is worse than no smear declared.
        frame_period_s=args.interval_s,
    ).open(settle_frames=args.settle)

    # **The span is taken from the target's own cells, over the whole sequence.** A per-frame AGC
    # rescales itself from the current histogram, so a target whose temperature is the subject
    # comes out looking the same in every frame -- the gain follows the target and cancels the
    # change being filmed. And it must span the *cells*, not the per-prim nodes: on this scene the
    # deck runs 29 K above air while every node sits within a kelvin of it, so a node-derived span
    # renders the one gradient this clip exists to show as a single flat code.
    probe = load_scene()
    lo_k, hi_k = np.inf, -np.inf
    # The **skin** alone, tracked separately from the powered nodes. A span taken over everything
    # is set by the motors, and on this aircraft they run forty kelvin above a skin whose whole
    # structure is four -- so the airframe lands in the bottom eight display codes and the frame
    # that was supposed to show per-part temperature shows a silhouette.
    cell_lo, cell_hi = np.inf, -np.inf
    for t_rel in np.linspace(0.0, span_s, 64):
        nodes = probe.advance_targets(float(t_rel), 0.0)
        lo_k = min(lo_k, min(nodes.values()))
        hi_k = max(hi_k, max(nodes.values()))
        for name in probe.surface_fields:
            fld = probe.surface_fields[name]
            fld.advance_to(probe.t0_s + float(t_rel))
            cells = np.asarray(fld.temperature_at(probe.t0_s + float(t_rel)))
            cell_lo = min(cell_lo, float(cells.min()))
            cell_hi = max(cell_hi, float(cells.max()))
    lo_k, hi_k = min(lo_k, cell_lo), max(hi_k, cell_hi)

    emissive = spec.outputs.apparent_temperature
    # **Two fixed spans, because no single linear one shows both features**, and for the same
    # reason `render_car_ignition` carries two. The motors reach tens of kelvin over air, so a
    # span wide enough for them puts the whole skin in the bottom few display codes; a span tight
    # enough for the skin clips every powered part to white. That is not a modelling problem --
    # it is why a thermal operator changes range, and it is the honest picture of a drone: the
    # hot parts and the cold airframe are not on one scale.
    #
    # The tight span is taken on the solved **cells** over the whole mission, not on the air
    # temperature: how far the skin sits from air is the measurement, so a span defined from air
    # would move the picture whenever the weather did and two frames of the same aircraft would
    # stop being comparable.
    spans: dict[str, DisplaySpan] = {}
    if args.span_c:
        if not emissive:
            print("--span-c is a temperature span and this band has none", file=sys.stderr)
            return 1
        if float(args.span_c[1]) <= float(args.span_c[0]):
            print("--span-c must be increasing", file=sys.stderr)
            return 1
        spans["ir"] = DisplaySpan(
            kind="apparent_t",
            low=float(args.span_c[0]) + 273.15,
            high=float(args.span_c[1]) + 273.15,
        )
    elif emissive:
        # A kelvin of headroom each way, so the extremes are not sitting on codes 0 and 255.
        spans["ir"] = DisplaySpan(kind="apparent_t", low=float(lo_k) - 1.0, high=float(hi_k) + 1.0)
        # The skin span is taken on the **solved cells over the whole mission**, half a kelvin
        # either side, so the airframe's own structure fills the ramp and the powered parts clip
        # white -- which is the correct reading of a part that is off this scale, not a loss.
        # It is anchored on the cells rather than on air because how far the skin sits from air
        # is the measurement; a span defined from air would move the picture whenever the
        # weather did, and two frames of the same aircraft would not be comparable.
        spans["skin"] = DisplaySpan(
            kind="apparent_t", low=float(cell_lo) - 0.5, high=float(cell_hi) + 0.5
        )
    palette_name = args.palette or spec.isp.palette
    palette = palette_table(palette_name)
    for kind, one in spans.items():
        print(f"display[{kind}]: fixed {one.caption}, {palette_name} palette")

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
            "near_m": track.near_m,
            "far_m": track.far_m,
            "elevation_deg": track.elevation_deg,
            "interval_s": args.interval_s,
            "point_wise": not args.no_fields,
        },
    )

    # The readout names the scene's **own** surfaces rather than a fixed list, so an aircraft
    # whose parts are called something else still gets a legible frame. The two quoted on the
    # caption line are the up-facing and down-facing pair -- the contrast that says what the
    # sun did to this airframe -- picked off the declared tilt rather than off the name.
    # The *largest* surface on each side, not the alphabetically first: on both scenes an arm
    # sorts before the main skin, and an arm is half in its own shadow, so naming it here would
    # caption the frame with a shaded strip while calling it the sunlit side.
    def widest(names: list[str]) -> list[str]:
        cells = {
            n: scene.patches[n].n_u * scene.patches[n].n_v for n in names if n in scene.patches
        }
        return [max(cells, key=lambda n: cells[n])] if cells else []

    up_facing = widest([n for n, srf in surface_tilts.items() if srf < 90.0])
    down_facing = widest([n for n, srf in surface_tilts.items() if srf >= 90.0])
    sunlit_shaded = (up_facing + down_facing) or sorted(scene.surface_fields)[:2]
    gauge_fields = (
        sunlit_shaded + [n for n in sorted(scene.surface_fields) if n not in sunlit_shaded][:1]
    )

    history: list[dict[str, float]] = []
    planes_written = 0
    t_render = time.time()
    for index in range(args.frames):
        t_rel = camera.t_rel_s
        throttle = throttle_at(scene, t_rel)
        range_m = track.range_at(t_rel)
        # Move the mount, then tell the camera it moved: the ray directions, the sky elevations
        # and every view cosine are built from a cached pose, and a slewing camera that does not
        # refresh renders geometry from here and radiometry from where it used to be.
        stage.aim_camera(t_rel)
        camera.refresh_pose()
        if not args.no_rotors:
            camera.rotor_mounts.update(rotor_mounts(throttle))

        outputs = camera.get_outputs(rt_subframes=args.rt_subframes)
        temps = camera.bridge.temperatures()
        if not spans:
            # A reflective band has no apparent temperature to span, so the picture's own ADC
            # percentiles are taken once and then held. A manual span, not a slow AGC.
            spans["ir"] = span_from_dn16(np.asarray(outputs.dn16))
            print(f"display[ir]: fixed {spans['ir'].caption} from the first frame's ADC")
        cells = {
            name: float(np.mean(np.asarray(fld.temperature_at(scene.t0_s + t_rel))))
            for name, fld in scene.surface_fields.items()
        }
        record: dict[str, float] = {
            "frame": index,
            "t_rel_s": round(t_rel, 3),
            "range_m": round(range_m, 3),
            "throttle": round(throttle, 4),
            "span_px": round(1e3 * SPAN_M / range_m / ifov_mrad, 2),
            **{f"{k}_k": round(v, 3) for k, v in temps.items()},
            **{f"cell_{k}_k": round(v, 3) for k, v in cells.items()},
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

        if writer.wants(index):
            extra = {}
            if args.rgb and camera.last_frame is not None and camera.last_frame.rgb is not None:
                extra["rgb"] = camera.last_frame.rgb
            writer.write(
                outputs,
                frame_index=index,
                t_s=camera.last_frame_t_s,
                extra_planes=extra,
                extra_metadata={
                    "throttle": round(throttle, 4),
                    "range_m": round(range_m, 3),
                    "node_temperatures_k": {k: round(float(v), 3) for k, v in temps.items()},
                    "cell_mean_k": {k: round(v, 3) for k, v in cells.items()},
                },
            )
            planes_written += 1

        def readout(
            image: Any,
            caption: str,
            scale: Any,
            ramp: Any = None,
            t: float = t_rel,
            u: float = throttle,
            r: float = range_m,
            cell: dict = cells,
            node: dict = temps,
        ) -> Any:
            minutes, seconds = divmod(int(t), 60)
            return overlay_readout(
                image,
                [
                    f"T+{minutes:02d}:{seconds:02d}   range {r:6.1f} m   "
                    f"span {1e3 * SPAN_M / r / ifov_mrad:5.1f} px",
                    f"throttle {u * 100:3.0f}%   "
                    + "   ".join(f"{n} {cell[n] - 273.15:5.1f}C" for n in sunlit_shaded),
                    f"1 frame / {args.interval_s:g} s time-lapse   {caption}",
                ],
                {**{f"{n:7.7s}": cell[n] for n in gauge_fields}, "motor  ": node["motor"]},
                scale,
                bare=args.no_overlay,
                palette=ramp,
            )

        for kind, one in spans.items():
            write_png(
                frames_dir / f"{kind}_{index:05d}.png",
                readout(
                    palette[quantise_display(one.scale(outputs))],
                    f"{one.caption} {palette_name}",
                    (one.low, one.high) if one.is_temperature else None,
                    # The colour bar goes beside a **fixed** span only. Under either AGC mode the
                    # mapping is rebuilt from each frame's own histogram, so one bar would be
                    # wrong for every frame but one -- and the AGC video not having a scale is
                    # exactly what distinguishes a picture of contrast from a measurement.
                    palette if one.is_temperature else None,
                ),
            )
        write_png(
            frames_dir / f"agc_{index:05d}.png",
            readout(np.asarray(outputs.display8), f"camera {spec.isp.agc}", None),
        )
        if args.rgb and camera.last_frame is not None and camera.last_frame.rgb is not None:
            write_png(
                frames_dir / f"rgb_{index:05d}.png",
                np.ascontiguousarray(np.asarray(camera.last_frame.rgb)[..., :3]),
            )
        if index % 25 == 0 or index == args.frames - 1:
            print(
                f"  frame {index:4d}  T+{t_rel:7.1f}s  R={range_m:6.1f}m  u={throttle:4.2f}  "
                + "  ".join(f"{n} {cells[n] - 273.15:5.1f}C" for n in sunlit_shaded)
                + f"  motor {temps['motor'] - 273.15:5.1f}C"
            )
    render_s = time.time() - t_render
    camera.close()

    videos = {}
    if ffmpeg_available():
        for kind in (*spans, "agc", "rgb"):
            if not any(frames_dir.glob(f"{kind}_*.png")):
                continue
            videos[kind] = str(
                encode_mp4(
                    str(frames_dir / f"{kind}_*.png"),
                    out_dir / f"{args.airframe}_outbound_{kind}.mp4",
                    fps=args.fps,
                )
            )
        if not args.keep_frames:
            for png in frames_dir.glob("*.png"):
                png.unlink()
            frames_dir.rmdir()
    else:
        print("ffmpeg not found: keeping the PNG sequence", file=sys.stderr)

    first, last = history[0], history[-1]
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
        "point_wise": not args.no_fields,
        "near_m": track.near_m,
        "far_m": track.far_m,
        "elevation_deg": track.elevation_deg,
        "vfov_deg": round(vfov_deg, 3),
        "horizon_margin_deg": round(track.elevation_deg - 0.5 * vfov_deg, 3),
        "ifov_mrad": round(ifov_mrad, 4),
        "span_px_first": first["span_px"],
        "span_px_last": last["span_px"],
        "airframe": args.airframe,
        "span_m": SPAN_M,
        "span_label": span_label,
        "cloud_seed": args.cloud_seed,
        "sunlit_minus_shaded": sunlit_shaded,
        "sunlit_minus_shaded_k_first": round(
            first[f"cell_{sunlit_shaded[0]}_k"] - first[f"cell_{sunlit_shaded[-1]}_k"], 3
        ),
        "sunlit_minus_shaded_k_last": round(
            last[f"cell_{sunlit_shaded[0]}_k"] - last[f"cell_{sunlit_shaded[-1]}_k"], 3
        ),
        "patch_coverage": {k: round(float(v), 4) for k, v in camera.patch_coverage.items()},
        "display_spans": {
            k: {"kind": v.kind, "low": v.low, "high": v.high} for k, v in spans.items()
        },
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
        f"\n{args.frames} frames in {render_s:.0f} s. "
        f"target {first['span_px']:.0f} px -> {last['span_px']:.1f} px across; "
        f"{sunlit_shaded[0]} - {sunlit_shaded[-1]} "
        f"{summary['sunlit_minus_shaded_k_first']:.1f} K\n{videos}\nwrote {path}"
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
