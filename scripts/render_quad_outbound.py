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

**The reference clip (EV.14, ADR 0168).** ``--reference`` turns the time-lapse into what the
public aerial sets look like: ten seconds of continuous video at the sensor's own frame rate, the
heavy-lift quad at 50 m to 250 m (27 px to 5.3 px across), fair-weather cumulus behind it
(``configs/scenes/aerial_reference_clip.yaml``), and a second, distant drone 1.2 km beyond it
that stays under a pixel for the whole clip and therefore goes through the analytic sub-pixel
path (ADR 0071). The capture interval is ``1 / frame_rate_hz``, so the bolometer's smear over its
window and the chain's temporal noise run at their real rate rather than once per six seconds.

**The clear-sky exit (EV.16).** ``--clear-exit`` is continuous video too, against a sky with no
cloud in it at all (``configs/scenes/aerial_clear_exit.yaml``): the heavy-lift quad from 12 m to
150 m in 15 s, then the mount stops tracking and the aircraft crosses the edge of the field at
15 m/s and leaves, then a second of empty sky (:func:`irsim_isaac.quad_outbound.clear_exit_track`).
Every frame records where the aircraft *should* be in the picture beside how many pixels the
renderer actually gave it and where they sit, so the frame edge is checked from the render and
not assumed. ``--far-m 30 --outbound-s 3`` is the same exit at close range, where the aircraft is
44 px across and is visibly cut by the edge on its way out.

Run with the Isaac interpreter (docs/decisions/0002), after `make luts`:
    IRSIM_GPU=0 python.sh scripts/render_quad_outbound.py --frames 300 --rgb
    IRSIM_GPU=0 python.sh scripts/render_quad_outbound.py --reference --rgb
    IRSIM_GPU=0 python.sh scripts/render_quad_outbound.py --clear-exit --rgb
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
    default=None,
    choices=sorted(AIRFRAMES),
    help="which aircraft to fly. Each carries its own scene config and range band "
    "(default: phantom3, or heavy_lift with --reference)",
)
parser.add_argument(
    "--reference",
    action="store_true",
    help="EV.14: 10 s of continuous video at the sensor's frame rate, 50-250 m, cumulus behind "
    "the target and a sub-pixel companion 1.2 km beyond it",
)
parser.add_argument(
    "--clear-exit",
    action="store_true",
    help="EV.16: continuous video in a cloudless sky, 12-150 m, then out of the frame sideways",
)
parser.add_argument(
    "--outbound-s",
    type=float,
    default=15.0,
    help="--clear-exit: seconds from near to far before the aircraft leaves the frame",
)
parser.add_argument(
    "--exit-speed-m-s",
    type=float,
    default=15.0,
    help="--clear-exit: the aircraft's speed across the line of sight as it leaves",
)
parser.add_argument(
    "--hold-s",
    type=float,
    default=1.0,
    help="--clear-exit: seconds of empty sky after the aircraft has gone",
)
parser.add_argument("--out", default=None)
parser.add_argument(
    "--frames", type=int, default=None, help="default 300; --reference: 10 s; --clear-exit: all"
)
parser.add_argument("--sensor", default=str(REPO / "configs/sensors/flir_boson_640_lwir.yaml"))
parser.add_argument("--scene", default=None)
parser.add_argument(
    "--interval-s",
    type=float,
    default=None,
    help="scene seconds per captured frame. frames x interval must fit inside the flight profile "
    "(default 6; --reference: one frame period of the sensor)",
)
parser.add_argument(
    "--fps",
    type=float,
    default=30.0,
    help="playback rate of the encoded video (--reference without --fps: the sensor's own rate)",
)
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
# **A cloud with a top** (AT.12): the field laid on a horizontal deck at the LCL, each column
# given its own top, marched per ray for the infrared band. The visible dome bakes the same deck,
# so the two bands cannot disagree about where the cloud is (ADR 0076).
parser.add_argument(
    "--cloud-deck",
    action="store_true",
    help="cloud with a top: marched per ray for LWIR, the same deck baked into the dome (AT.12)",
)
# Additionally write the deck as a NanoVDB volume and put it in the stage as real participating
# geometry, clearing the dome's cloud. **Blocked on this build** -- Isaac Sim 6.1's IndeX plugin
# refuses the grid ("unable to create VDB subset") even after the version stamp is matched -- so
# it is off by default and the render says so. Kept because it is the artefact `AT.13` probes
# with, and because the failure is loud: the companion frame comes back with no cloud in it.
parser.add_argument(
    "--cloud-volume",
    action="store_true",
    help="also author the deck as a UsdVol.Volume (AT.12; refused by this build's IndeX plugin)",
)
parser.add_argument(
    "--cloud-voxel-m",
    type=float,
    default=25.0,
    help="voxel size of the cloud volume, metres (default 25)",
)
parser.add_argument("--no-overlay", action="store_true", help="no burnt-in readout")
parser.add_argument("--keep-frames", action="store_true", help="keep the PNG sequence")
parser.add_argument("--integration-ms", type=float, default=None)

args = parser.parse_args()

#: EV.14's sub-pixel companion: how far beyond the aircraft it holds station, how far above the
#: boresight it sits, and the area it shows the camera -- a 350 mm-class quad seen from below,
#: ESTIMATED as its body plus arms. At 1.25-1.5 km it fills under a tenth of a Boson pixel.
COMPANION_BEYOND_M = 1200.0
COMPANION_OFFSET_DEG = 6.0
COMPANION_AREA_M2 = 0.06

#: The continuous clips: captured at the sensor's own frame period, not as a time-lapse.
CONTINUOUS = bool(args.reference or args.clear_exit)
if args.airframe is None:
    args.airframe = "heavy_lift" if CONTINUOUS else "phantom3"
if args.reference and args.close_up:
    parser.error("--reference is the standoff clip; --close-up is the opposite measurement")
if args.clear_exit and (args.reference or args.close_up):
    parser.error("--clear-exit is its own clip; it does not combine with --reference or --close-up")
if args.clear_exit and args.airframe != "heavy_lift":
    parser.error("--clear-exit flies the heavy-lift frame its scene describes")
_AIRFRAME = AIRFRAMES[args.airframe]
if args.scene is None:
    args.scene = str(
        REPO
        / "configs"
        / "scenes"
        / (
            "aerial_reference_clip.yaml"
            if args.reference
            else ("aerial_clear_exit.yaml" if args.clear_exit else _AIRFRAME["scene"])
        )
    )
if args.out is None:
    args.out = (
        "outputs/aerial_reference_clip"
        if args.reference
        else ("outputs/aerial_clear_exit" if args.clear_exit else _AIRFRAME["out"])
    )
if args.clear_exit:
    # Literal here because Kit has not booted yet; `clear_exit_track` owns the same numbers and
    # `test_aerial_clear_exit.py` holds the two together.
    if args.near_m is None:
        args.near_m = 12.0
    if args.far_m is None:
        args.far_m = 150.0
    # The weather carries no cloud, so a cloud field would cover nothing; not generating one says
    # so in the sidecar rather than leaving a seed that drew an empty sky.
    args.no_cloud = True
if args.reference:
    # The reference band, not the airframe's own: the sets it is compared with film at standoff.
    # Literal here because Kit has not booted yet; `reference_track` owns the same numbers and
    # `test_aerial_reference_clip.py` holds the two together.
    if args.near_m is None:
        args.near_m = 50.0
    if args.far_m is None:
        args.far_m = 250.0
    # A cloud with a top, so there is cloud *behind* the target along the ray (AT.12, AT.14).
    if not args.no_cloud:
        args.cloud_deck = True
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
    args.cloud_deck = False
    args.cloud_volume = False
if args.cloud_volume:
    args.cloud_deck = True
    # The RTX renderer documents volume rendering under `PathTracing` only, and `--rgb` is what
    # puts this build into it. Implied rather than required, because a volume authored into a
    # stage the renderer will not integrate is a silent no-op: the frame comes back with no cloud
    # in it and nothing says why.
    args.rgb = True

t_boot = time.time()
from isaacsim import SimulationApp  # noqa: E402

from irsim_isaac.env import render_device, simulation_app_config  # noqa: E402

app = SimulationApp(simulation_app_config())
boot_s = time.time() - t_boot


def throttle_at(scene: object, t_rel_s: float) -> float:
    """The flight profile's throttle at a scene-relative time, read back off the scene config."""
    import numpy as np

    for target in scene.spec.targets:  # type: ignore[attr-defined]
        if target.solver == "heat_source" and target.throttle_s:
            return float(np.interp(t_rel_s, target.throttle_s, target.throttle))
    return 0.0


def exit_summary(history: list[dict[str, Any]], track: Any, width_px: int) -> dict[str, Any]:
    """EV.16: when the renderer stopped drawing the aircraft, against when the track says it left.

    ``last_drawn_frame`` and ``first_empty_frame`` are read off the instance plane; the track's own
    times sit beside them. The sky's apparent-temperature spread over the empty frames is the
    no-target baseline: the clear sky's elevation gradient across the field plus noise, and
    nothing else.
    """
    drawn = [h["frame"] for h in history if h.get("target_px", 0.0) > 0.0]
    last_drawn = max(drawn) if drawn else None
    empty = [h for h in history if last_drawn is not None and h["frame"] > last_drawn]
    out: dict[str, Any] = {
        "outbound_s": track.duration_s,
        "exit_s": round(track.exit_s, 4),
        "exit_pan_deg": round(track.exit_pan_deg, 4),
        "hold_s": track.hold_s,
        "last_drawn_frame": last_drawn,
        "last_drawn_t_s": None if last_drawn is None else history[last_drawn]["t_rel_s"],
        "first_empty_frame": empty[0]["frame"] if empty else None,
        "empty_frames": len(empty),
        "track_exit_end_t_s": round(track.duration_s + track.exit_s, 4),
    }
    if empty and "t_app_p001_k" in empty[0]:
        # 0.1 / 99.9 percentiles: the extremes are defect pixels, not sky.
        out["empty_sky_t_app_p001_p999_k"] = [
            round(min(h["t_app_p001_k"] for h in empty), 3),
            round(max(h["t_app_p999_k"] for h in empty), 3),
        ]
    # How far the drawn columns' centre sits from the pinhole's prediction, while the aircraft is
    # wholly in frame. A few tenths of a pixel is the rasteriser; more is a frame-convention bug.
    whole = [
        abs(0.5 * (h["target_cols_px"][0] + h["target_cols_px"][1]) - h["target_u_px"])
        for h in history
        if "target_cols_px" in h
        and h["target_cols_px"][0] > 1.0
        and h["target_cols_px"][1] < width_px - 1.0
    ]
    if whole:
        out["centre_error_px_median"] = round(float(sorted(whole)[len(whole) // 2]), 3)
        out["centre_error_px_max"] = round(float(max(whole)), 3)
    return out


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
    from irsim_isaac.pipeline.material_ids import BACKGROUND_INSTANCE_ID
    from irsim_isaac.pipeline.materials_usd import prim_records
    from irsim_isaac.pipeline.point_bridge import bindings_from_scene
    from irsim_isaac.quad_outbound import (
        AIM_ELEVATION_DEG,
        REFERENCE_DURATION_S,
        REFERENCE_SWEEP_DEG,
        OutboundTrack,
        build_quad_outbound,
        clear_exit_track,
        companion_angle_deg,
        distant_companion_position,
        target_image_angles_deg,
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
    from irsim_isaac.cloud_volume import author_cloud_volume, write_nanovdb
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
    # EV.14: continuous video at the camera's own rate, unless a caller asked for something else.
    if CONTINUOUS:
        rate = float(spec.fpa.frame_rate_hz)
        if args.interval_s is None:
            args.interval_s = 1.0 / rate
        if args.frames is None and args.reference:
            args.frames = int(round(REFERENCE_DURATION_S / args.interval_s))
        # The literal default stays 30 (the sweep's watch-length test reads it); the reference
        # clip plays at the camera's own rate unless the command line asked for another.
        if not any(a == "--fps" or a.startswith("--fps=") for a in sys.argv[1:]):
            args.fps = rate
    if args.frames is None and not args.clear_exit:
        args.frames = 300
    if args.interval_s is None:
        args.interval_s = 6.0
    ifov_mrad = 1e3 * spec.fpa.pitch_um * 1e-3 / spec.optics.focal_length_mm
    vfov_deg = math.degrees(2.0 * math.atan(0.5 * spec.fpa.height * ifov_mrad * 1e-3))
    hfov_deg = math.degrees(2.0 * math.atan(0.5 * spec.fpa.width * ifov_mrad * 1e-3))
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

    exit_track = None
    if args.clear_exit:
        try:
            exit_track = clear_exit_track(
                hfov_deg,
                ifov_mrad,
                near_m=args.near_m,
                far_m=args.far_m,
                outbound_s=args.outbound_s,
                exit_speed_m_s=args.exit_speed_m_s,
                hold_s=args.hold_s,
            )
        except ValueError as exc:
            print(f"--clear-exit: {exc}", file=sys.stderr)
            return 1
        if args.frames is None:
            args.frames = int(math.ceil(exit_track.end_s / args.interval_s)) + 1
    span_s = args.frames * args.interval_s
    track = exit_track or OutboundTrack(
        near_m=args.near_m,
        far_m=args.far_m,
        duration_s=span_s,
        elevation_deg=AIM_ELEVATION_DEG if args.elevation_deg is None else args.elevation_deg,
        # A wider walk round the aircraft when it fills the frame: at 80 m the aspect hardly
        # matters, but at 2 m which side the sun is on is most of the picture. The reference
        # clip slews a few degrees in ten seconds, as a tracking mount does (EV.14).
        azimuth_sweep_deg=REFERENCE_SWEEP_DEG
        if args.reference
        else (140.0 if args.close_up else 55.0),
    )

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
    deck = None
    volume = None
    if args.cloud_deck:
        if args.cloud_seed is None or scene.environment is None:
            print("--cloud-deck needs a cloud seed and an environment preset", file=sys.stderr)
            return 1
        sky_model = scene.sky_models.get(spec.band.band_id)
        if sky_model is None or scene.environment.clouds.optical_depth is None:
            print(
                "--cloud-deck needs a scene whose environment preset authors "
                "clouds.optical_depth (ADR 0126); clouds.tau is a plane-parallel sheet with no "
                "top to give it",
                file=sys.stderr,
            )
            return 1
        deck = sky_model.cloud_deck(scene.t0_s, int(args.cloud_seed))
        march_steps = deck.adequate_steps(
            np.radians(np.array([track.elevation_deg - 0.5 * vfov_deg]))
        )
        print(
            f"cloud deck: base {deck.base_m:.0f} m, tops to {deck.top_m:.0f} m, "
            f"tiling every {2 * deck.half_extent_m / 1e3:.1f} km, visible OD "
            f"{deck.optical_depth:g} at full depth, {deck.depth.shape[0]}^2 cells of "
            f"{deck.cell_m:g} m, mean column {deck.depth[deck.depth > 0].mean():.2f} of the "
            f"deck over {100 * (deck.depth > 0).mean():.0f}% of it, "
            f"{march_steps} march steps at the frame's lowest ray"
        )
        if args.cloud_volume:
            volume = write_nanovdb(
                deck,
                out_dir / "cloud_deck.nvdb",
                voxel_m=args.cloud_voxel_m,
                device=render_device(),
            )
            print(f"cloud volume: {volume.caption}")
    dome = None
    if not args.no_dome:
        # heading 0: the scene's `world_frame:` puts north on the stage's -Z, so the dome, the sun
        # light and the scene's own solar geometry are all in the one frame.
        #
        # **The dome's cloud is switched off when the volume carries it**, or the frame would show
        # both: a painted cloud at infinity and the real one in front of it, in the same places,
        # which reads as haze rather than as a mistake.
        # The dome bakes whichever cloud the infrared band is reading: the deck when there is
        # one, the hemispherical field otherwise, and **nothing** when a volume is carrying it --
        # or the frame would show cloud twice, in the same places, which reads as haze.
        dome = dome_spec_from_scene(
            scene,
            cloud=None if deck is not None else cloud,
            deck=None if args.cloud_volume else deck,
            heading_deg=0.0,
        )
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
    if volume is not None and deck is not None:
        import omni.usd

        prim = author_cloud_volume(omni.usd.get_context().get_stage(), "/World/Cloud", volume)
        print(f"cloud volume: {prim.GetPath()} -> {volume.path.name}")
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

    # EV.14: the sub-pixel companion, injected analytically (ADR 0071). It has no prim on the
    # stage, so there is nothing to hide and nothing the renderer could double-count; its
    # temperature is the scene's per-prim airframe node, a drone holding station in the same air.
    analytic = []
    companion: dict[str, Any] | None = None
    if args.reference:
        from irsim.pipeline.point_target import fill_fraction
        from irsim_isaac.pipeline.ir_camera import AnalyticTarget

        where = distant_companion_position(track, COMPANION_BEYOND_M, COMPANION_OFFSET_DEG)
        seen = [companion_angle_deg(track, t, where) for t in np.linspace(0.0, span_s, 11)]
        phis = [
            fill_fraction(
                COMPANION_AREA_M2, r, spec.optics.focal_length_mm * 1e-3, spec.pixel_area_m2
            )
            for r, _ in seen
        ]
        companion = {
            "position_m": [round(x, 3) for x in where],
            "area_m2": COMPANION_AREA_M2,
            "range_m": [round(min(r for r, _ in seen), 1), round(max(r for r, _ in seen), 1)],
            "off_boresight_deg": [
                round(min(a for _, a in seen), 2),
                round(max(a for _, a in seen), 2),
            ],
            "fill_fraction": [round(min(phis), 4), round(max(phis), 4)],
        }
        print(
            f"companion: {COMPANION_AREA_M2} m^2 at {companion['range_m'][0]:.0f}-"
            f"{companion['range_m'][1]:.0f} m, {companion['off_boresight_deg'][0]:.1f}-"
            f"{companion['off_boresight_deg'][1]:.1f} deg off the boresight, fill "
            f"{companion['fill_fraction'][0]:.3f}-{companion['fill_fraction'][1]:.3f} of a pixel"
        )
        analytic.append(
            AnalyticTarget(
                name="companion",
                world_position=where,
                area_m2=COMPANION_AREA_M2,
                material="carbon_fibre",
                thermal_node="airframe",
            )
        )

    camera = IrCamera(
        sensor,
        scene,
        pipeline=pipeline,
        analytic_targets=analytic,
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
        cloud_deck=args.cloud_deck,
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
        # **A third span, for the sky the aircraft is flying against**, on the outbound clip only
        # (in the close-up the airframe is most of the frame and there is little sky to read).
        #
        # Neither span above reaches it. `ir` starts at the coolest airframe node and `skin` is
        # tighter still, so cloud at 15 C and a clear zenith at -27 C both land on code 0 and the
        # only picture carrying the sky is the camera's own AGC. That one is a plateau-
        # equalising AGC with DDE, and on this scene it is given a frame where 60 % of the pixels
        # sit inside half a kelvin of each other: it hands that plateau an enormous local gain,
        # so what it draws over the cloud is amplified NETD, not cloud. Measured on
        # `frame_000120`: 0.44 K of real spread inside the cloud came out as **43 display codes**
        # of speckle.
        #
        # This span is linear over the scene's own sky-to-target range, so 60 K falls across 256
        # codes at 0.23 K each -- four times the sensor's NETD, which puts the noise back under
        # the quantisation where a real thermal sky image has it.
        sky_model = probe.sky_models.get(spec.band.band_id)
        if sky_model is not None and not args.close_up:
            top_el = math.radians(track.elevation_deg) + 0.5 * math.radians(vfov_deg)
            # The **clear** sky at the top of the frame, not the cloud-blended average: with 45 %
            # coverage the blend sits 20 K above the coldest thing in the picture, and a span
            # starting there puts every gap in the cloud on display code 0 -- which is the defect
            # this span exists to remove.
            coldest = float(
                np.min(
                    sky_model.lut.apparent_temperature(
                        sky_model.clear_radiance(probe.t0_s, np.array([top_el])),
                        sky_model.quantity,
                    )
                )
            )
            # Top of the span is the **skin**, not the hottest node: stretching it to the motors
            # at 65 C spends half the display on four bells and leaves the cloud and the airframe
            # sharing the lower half. The motors clip white instead, which is the correct reading
            # of a part that is off this scale and the same choice `skin` makes.
            spans["sky"] = DisplaySpan(
                kind="apparent_t",
                low=coldest - 2.0,
                high=max(float(cell_hi), coldest + 10.0) + 2.0,
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
            if args.clear_exit:
                # The extremes are the chain's defect pixels -- a dead one reads the LUT's 200 K
                # floor on every frame -- so the empty-sky baseline is taken on percentiles.
                lo, hi = np.percentile(t_app, [0.1, 99.9])
                record["t_app_p001_k"] = round(float(lo), 3)
                record["t_app_p999_k"] = round(float(hi), 3)
        if outputs.dn16 is not None:
            dn = np.asarray(outputs.dn16)
            record["dn16_min"] = float(dn.min())
            record["dn16_max"] = float(dn.max())
        status = ""
        if track.exit_s > 0.0:
            # EV.16: where the pinhole puts the aircraft's centre, beside what the renderer drew.
            # The two are independent -- one is the track's arithmetic, the other the instance
            # plane -- so a frame edge in the wrong place, or a target still drawn after it has
            # gone, shows as a disagreement in the sidecar rather than as a plausible picture.
            h_deg, v_deg = target_image_angles_deg(track, t_rel)
            ifov_rad = ifov_mrad * 1e-3
            record["pan_deg"] = round(track.pan_deg_at(t_rel), 4)
            record["target_u_px"] = round(
                0.5 * spec.fpa.width + math.tan(math.radians(h_deg)) / ifov_rad, 2
            )
            record["target_v_px"] = round(
                0.5 * spec.fpa.height - math.tan(math.radians(v_deg)) / ifov_rad, 2
            )
            if camera.last_frame is not None:
                ids = np.asarray(camera.last_frame.instance_id)
                k = ids.shape[1] / float(spec.fpa.width)
                drawn = ids != BACKGROUND_INSTANCE_ID
                record["target_px"] = round(float(drawn.sum()) / (k * k), 2)
                cols = np.flatnonzero(drawn.any(axis=0))
                if cols.size:
                    record["target_cols_px"] = [
                        round(float(cols[0]) / k, 1),
                        round(float(cols[-1] + 1) / k, 1),
                    ]
                if track.pan_deg_at(t_rel) > 0.0:
                    status = "   out of frame" if not cols.size else "   leaving the frame"
        if analytic:
            # EV.14: where the companion landed, or where it fell outside the frame. Recomputed
            # rather than cached by the camera, which keeps only the ones it could not see.
            on = camera.point_targets()
            if on:
                record["companion_px"] = [round(x, 2) for x in on[0].position_px]
            for name, (u, v) in camera.last_offscreen_targets:
                record[f"{name}_offscreen_px"] = [round(u, 1), round(v, 1)]
        if outputs.display8 is not None:
            # EV.14: what the camera's own AGC did this frame, for the frame-to-frame drift.
            record["display8_mean"] = round(float(np.mean(np.asarray(outputs.display8))), 3)
        history.append(record)

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
            edge: str = status,
        ) -> Any:
            minutes, seconds = divmod(float(t), 60.0)
            return overlay_readout(
                image,
                [
                    f"T+{int(minutes):02d}:{seconds:05.2f}   range {r:6.1f} m   "
                    f"span {1e3 * SPAN_M / r / ifov_mrad:5.1f} px{edge}",
                    f"throttle {u * 100:3.0f}%   "
                    + "   ".join(f"{n} {cell[n] - 273.15:5.1f}C" for n in sunlit_shaded),
                    (
                        f"continuous, {args.fps:g} Hz   {caption}"
                        if CONTINUOUS
                        else f"1 frame / {args.interval_s:g} s time-lapse   {caption}"
                    ),
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
                    out_dir
                    / (
                        f"aerial_reference_{kind}.mp4"
                        if args.reference
                        else (
                            f"aerial_clear_exit_{kind}.mp4"
                            if args.clear_exit
                            else f"{args.airframe}_outbound_{kind}.mp4"
                        )
                    ),
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
    # EV.14: the camera AGC's frame-to-frame drift -- the number SC.10's damping is to be judged
    # on. Per-frame AGC has no memory, so this is what the stream does without it.
    means = np.array([h["display8_mean"] for h in history if "display8_mean" in h])
    steps = np.abs(np.diff(means)) if means.size > 1 else np.zeros(1)
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
        "cloud_deck": None
        if deck is None
        else {
            "base_m": deck.base_m,
            "top_m": deck.top_m,
            "tile_m": 2.0 * deck.half_extent_m,
            "cell_m": deck.cell_m,
            "optical_depth": deck.optical_depth,
        },
        "cloud_volume": None
        if volume is None
        else {
            "path": volume.path.name,
            "voxel_m": volume.voxel_m,
            "shape": list(volume.shape),
            "min_world_m": list(volume.min_world_m),
        },
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
        "reference": bool(args.reference),
        "clear_exit": exit_summary(history, track, spec.fpa.width) if args.clear_exit else None,
        "companion": companion,
        "display8_mean_step_median": round(float(np.median(steps)), 4),
        "display8_mean_step_p95": round(float(np.percentile(steps, 95)), 4),
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
