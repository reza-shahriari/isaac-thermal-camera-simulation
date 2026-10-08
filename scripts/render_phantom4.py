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

**Temperature is per cell on the mesh (`AI.2`, closed).** The scene's mesh fields are solved and
`IrCamera` takes them as ``mesh_fields=``: `MeshPointBridge` (WM.3) asks each bound prim's own
triangles for the closest point to a pixel's world position and reads that cell. The binding
names ``frame=ASSET_ROOT``, the Xform the asset is mounted by, so the solver keeps the archive's
own Z-up coordinates and the mount rotation travels with the binding rather than with the solve.
This paragraph used to say the opposite -- that the bridge had never been driven by a render --
and stayed that way after the code moved (corrected by `SC.29`).

**Shutter-referenced, like the camera (`SC.29`).** The flat field and the M9 chain are attached
as in every other driver; ``--no-flat-field`` and ``--no-chain`` take them off again.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import math
import pathlib
import re
import sys
import time
from collections.abc import Collection
from typing import Any

REPO = pathlib.Path(__file__).resolve().parents[1]

#: Where the prepared asset's prims land once referenced.
ASSET_ROOT = "/World/Targets/phantom4"

#: Part name -> thermal-node candidates, in order of preference (AI.5, ADR 0138). The asset used
#: with ``--asset phantom4_parts`` has one prim per **part**, so a target can name hardware
#: instead of a material. A part's *own name* is always tried first and is not listed here:
#: :func:`target_for_part` takes the first candidate the scene actually defines, so a scene that
#: gives each motor its own node (`phantom4_perpart.yaml`) gets four motor nodes and a scene with
#: one `motor` node (`phantom4_parts.yaml`) gets one, from the same map and the same asset.
#:
#: What this replaces, and why the old form could not be rescued. Until AI.5 this was
#: ``TARGET_BY_MATERIAL = {"copper": "motor"}`` with everything else on ``airframe`` -- two thermal
#: nodes for an aircraft with four motors, a battery and four propellers. It could not be made
#: finer, because in the source asset a prim is a *material group*: only 25 of 41 prims are more
#: than 90 % one part and those carry just 34.7 % of the area, and the prim containing the battery
#: holds nine parts of which the battery is 26 %. So the geometry was regrouped instead
#: (``prep_asset.py --emit-parts``), and this map keys on the result.
TARGET_BY_PART: dict[str, tuple[str, ...]] = {
    # The four motors are the hottest surfaces on a flying quadcopter: §6.6's aerial node peaks at
    # +45 K over ambient at full throttle (ADR 0072). Four prims; four nodes where the scene has
    # them, one where it does not.
    "motor_front_left": ("motor",),
    "motor_front_right": ("motor",),
    "motor_rear_left": ("motor",),
    "motor_rear_right": ("motor",),
    # The mounts are the plastic over the ESC in each arm; the ESC node peaks at +30 K. A mount's
    # pixels come from its own mesh field wherever a cell covers them, so this is the fallback.
    "motor_mount_front_left": ("esc_front_left", "esc"),
    "motor_mount_front_right": ("esc_front_right", "esc"),
    "motor_mount_rear_left": ("esc_rear_left", "esc"),
    "motor_mount_rear_right": ("esc_rear_right", "esc"),
    # The pack: +15 K, large mass, and **enclosed by the shell**, so the camera sees it only
    # through what it warms. It exists in the model for the first time as of ADR 0138.
    "battery": ("battery",),
    # The propellers are 1 mm of ABS spinning above the arm they are bolted to, and are mesh
    # surfaces in every scene that flies this asset; the fallback is the arm, then the airframe.
    "propeller_front_left": ("arms", "airframe"),
    "propeller_front_right": ("arms", "airframe"),
    "propeller_rear_left": ("arms", "airframe"),
    "propeller_rear_right": ("arms", "airframe"),
}
DEFAULT_TARGET = "airframe"


_NUMBERED = re.compile(r"_\d{3}$")


def part_of_leaf(leaf: str, defined: Collection[str], objects: Collection[str] = ()) -> str:
    """The part a prim's leaf name stands for: itself, or itself without a trailing ``_NNN``.

    Blender's USD exporter numbers a Mesh prim when a mesh datablock of the same name already
    exists in the file (``skin_Head`` → ``skin_Head_001``), so the part a scene named is not
    always the leaf letter for letter. The unnumbered name wins only when the scene or an object
    actually defines it; a part whose own name ends in three digits is left alone.
    """
    if _NUMBERED.search(leaf):
        bare = _NUMBERED.sub("", leaf)
        candidates = [bare, *(f"{obj}.{bare}" for obj in objects), *TARGET_BY_PART.get(bare, ())]
        if any(c in defined for c in candidates):
            return bare
    return leaf


def target_for_part(part: str, defined: Collection[str], objects: Collection[str] = ()) -> str:
    """The thermal node a prim named after ``part`` renders from: the part's own name if the
    scene defines it, else the candidates of :data:`TARGET_BY_PART` in order, then
    :data:`DEFAULT_TARGET`. With ``objects`` (TC.13, `thermal.objects:`), a part solved as one of
    an object's surfaces -- ``<object>.<part>`` -- comes first: the solve, not the driver's map,
    is then what says how fine the nodes are. Nothing here is checked against the scene; the
    bridge refuses a name it does not have, which is the guard that matters.
    """
    for obj in objects:
        if f"{obj}.{part}" in defined:
            return f"{obj}.{part}"
    for candidate in (part, *TARGET_BY_PART.get(part, ())):
        if candidate in defined:
            return candidate
    return DEFAULT_TARGET


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


def _agreement_summary(agreements: list[Any]) -> dict[str, object] | None:
    """The run's worst and typical band agreement (WX.26): what the step's bar is checked on."""
    if not agreements:
        return None
    from statistics import median

    from irsim.validation.cloud_bands import IOU_BAR, P95_BAR

    ious = [a.iou for a in agreements]
    p95s = [a.p95_emissivity_error for a in agreements]
    return {
        "frames": len(agreements),
        "iou_min": round(min(ious), 4),
        "iou_median": round(float(median(ious)), 4),
        "p95_emissivity_error_max": round(max(p95s), 4),
        "p95_emissivity_error_median": round(float(median(p95s)), 4),
        "mean_emissivity_error_median": round(
            float(median([a.mean_emissivity_error for a in agreements])), 4
        ),
        "all_pass": all(a.passes() for a in agreements),
        "bar": f"IoU >= {IOU_BAR} and p95 |d emissivity| <= {P95_BAR} per frame (ADR 0191)",
    }


def _cloud_overrides(items: list[str]) -> dict[str, object]:
    """``--cloud-set`` as a weather-fx ``clouds`` section: numbers where they parse as numbers
    (weather-fx coerces each to its parameter's own type), names otherwise."""
    out: dict[str, object] = {"enabled": True}
    for item in items:
        name, _, value = item.partition("=")
        try:
            out[name] = float(value)
        except ValueError:
            out[name] = value
    return out


parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
parser.add_argument("--asset", default="phantom4")
parser.add_argument("--scene", default="configs/scenes/phantom4_pointwise.yaml")
parser.add_argument("--sensor", default="configs/sensors/flir_boson_640_lwir.yaml")
parser.add_argument("--out", default="outputs/phantom4")
parser.add_argument("--frames", type=int, default=96)
parser.add_argument(
    "--integration-ms",
    type=float,
    default=None,
    help="override a photon FPA's integration time, in ms (as render_quad_flight.py): a daylight "
    "reflective-band scene saturates a low-light exposure; changes the config hash",
)
parser.add_argument(
    "--focus-m",
    default=None,
    help="focus the lens at this distance in metres, or 'inf' (EV.21): a tracking camera's focus "
    "is set once and the drone is rarely at it. Sets a Hopkins defocus model if the sensor names "
    "none (irsim.config.loader.with_focus); changes the config hash. Default: the sensor's own",
)
parser.add_argument(
    "--corner-illumination",
    type=float,
    default=None,
    help="give the lens a mechanical vignetting falling to this relative illumination at the "
    "format's corner (EV.23; irsim.optics.vignetting.radial_vignetting_map, ESTIMATED): the "
    "off-axis pixels see more housing, so a drifted housing leaves a bowl. Default: the sensor's",
)
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
parser.add_argument(
    "--bridge-device",
    default="cpu",
    help=(
        "Warp device for the mesh bridge's closest-point query. 'cpu' is the default because "
        "this is a correctness path and the workstation is shared; 'cuda:0' is much faster on a "
        "quarter-million cells and gives the same answer -- the engine-free query is the oracle"
    ),
)
parser.add_argument(
    "--track",
    default="figure8",
    choices=("figure8", "outbound", "scatter", "wander"),
    help=(
        "figure8: one lemniscate circuit, the aspect sweep. outbound: a climbing run straight "
        "away from the observer, near to far, at a fixed elevation -- the detection question, "
        "where range is the only thing changing. scatter: no path -- every frame an "
        "independent range, elevation, bearing and heading (a detector's training set, with a "
        "YOLO label beside each frame). wander: the same variety as one continuous random "
        "flight, so the clip can be watched and the airframe is seen to warm up while it flies"
    ),
)
parser.add_argument("--scatter-seed", type=int, default=0, help="scatter, wander: the seed")
parser.add_argument("--wander-cycles", type=float, default=6.0, help="wander: swings per clip")
parser.add_argument("--elevation-low-deg", type=float, default=None, help="scatter: lowest")
parser.add_argument("--elevation-high-deg", type=float, default=None, help="scatter: highest")
parser.add_argument(
    "--aim-jitter",
    type=float,
    default=0.6,
    help="scatter: how far off the boresight the aircraft may sit, as a fraction of the half "
    "field of view on each axis (0 centres it in every frame, which no real mount does)",
)
parser.add_argument(
    "--lost-lock",
    default="",
    help="wander: runs of frames in which the camera loses the aircraft (EV.26), as "
    "'first:length:direction,...' -- direction in degrees, 0 right, 90 up, 180 left; the mount "
    "swings off over LOST_RAMP_FRAMES frames either side and holds the aircraft out of frame",
)
parser.add_argument(
    "--no-flat-field",
    action="store_true",
    help="skip the two-point flat field (SC.29): the picture then carries the housing bowl a "
    "real shuttered core removes",
)
parser.add_argument("--no-chain", action="store_true", help="ideal camera: no M9 sensor chain")
parser.add_argument(
    "--tilt-sigma-deg",
    type=float,
    default=0.0,
    help="scatter/wander: pitch and roll the aircraft (EV.25), a tilt of this scale in degrees "
    "(half-normal per scatter pose, a smooth wander per clip), capped at 25. 0 flies level",
)
parser.add_argument(
    "--housing-start-k",
    type=float,
    default=0.0,
    help="start the camera's housing this far from its settled temperature (EV.23): -4 is a "
    "camera just switched on, which warms through the clip and grows a dark-centred bowl since "
    "its power-up shutter; positive cools and grows a bright centre. 0: settled, as before",
)
parser.add_argument(
    "--nose-deg",
    type=float,
    default=None,
    help="the asset's nose as a bearing in its own XY plane, degrees from +x (default: the "
    "Phantom 4's NOSE_IN_ASSET); asset configs state it, e.g. -90 for a nose toward -y",
)
parser.add_argument(
    "--turn-deg",
    type=float,
    default=0.0,
    help="presentation yaw added to the flight heading, so a still shows the aircraft three-"
    "quarter on instead of tail-on (catalogue shots; leave 0 for any flight clip)",
)
parser.add_argument("--near-m", type=float, default=None, help="outbound: range at the start")
parser.add_argument("--far-m", type=float, default=None, help="outbound: range at the end")
parser.add_argument(
    "--elevation-deg", type=float, default=None, help="outbound: elevation held for the run"
)
# --- EV.20: the clear-sky exit, `render_quad_outbound.py --clear-exit` for the imported aircraft --
# The outbound run, then the mount stops and the aircraft strafes out of the field, then a second
# of the sky it left, at the camera's own 60 Hz, starting at cruise. Here the aircraft really moves
# and the camera holds still (asset_flight.StraightOutExitTrack); EV.16's quad leaves by a pan.
parser.add_argument(
    "--clear-exit",
    action="store_true",
    help="EV.20: out to --far-m, then strafe out of the field; 60 Hz, starts at cruise. Defaults "
    "to the solved aircraft under a cloudless sky (phantom4_clear_exit.yaml, phantom4_parts)",
)
parser.add_argument("--outbound-s", type=float, default=15.0, help="clear-exit: the outbound run")
parser.add_argument(
    "--exit-speed-m-s", type=float, default=10.0, help="clear-exit: the strafe speed"
)
parser.add_argument("--hold-s", type=float, default=1.0, help="clear-exit: empty sky at the end")
parser.add_argument(
    "--mission-start-s",
    type=float,
    default=None,
    help="mission second of the first frame (default 540 with --clear-exit: 210 s into the "
    "forward cruise, so the motors are at cruise temperature; 0 otherwise)",
)
parser.add_argument("--centre-range-m", type=float, default=None, help="track centre distance")
parser.add_argument("--half-width-m", type=float, default=None, help="half-width of the eight")
parser.add_argument("--altitude-low-m", type=float, default=None, help="lowest track altitude")
parser.add_argument("--altitude-high-m", type=float, default=None, help="highest track altitude")
parser.add_argument("--rt-subframes", type=int, default=16)
parser.add_argument(
    "--rgb-subframes",
    type=int,
    default=1,
    help="IG.19: expose the visible companion over the detector's window from this many "
    "sub-frames posed along the track (1 = one instant, as before)",
)
parser.add_argument("--settle", type=int, default=16)
parser.add_argument("--span", default=None, help="display span 'loC,hiC'; default from the solve")
parser.add_argument("--no-rgb", action="store_true", help="infrared only; no visible companion")
# The aircraft and the sky it flies against do not share a stretch. An outbound Phantom 4 spans
# 16.6..34.7 C while the same frame's clear sky reads -27 C and the cumulus in front of it +13 C,
# so a grayscale ramp fitted to the aircraft clips every cloud to black -- the infrared band looks
# empty beside a visible companion full of cloud, which reads as a missing model rather than as a
# display choice. This writes the same planes a second time, stretched to the whole frame, where
# the cloud is 40 K of structure and the aircraft is the part that saturates. Both are the same
# radiometry; neither is an extra render.
# A real camera shows the aircraft *and* the cloud in one frame, and it does it with its own AGC,
# not with a ramp. `SC.24`: the clip is **the camera's own `display8`**, frame by frame -- the
# config's ISP (the Boson's factory information-based equalisation since `SC.22`), run by the
# pipeline on the raw DN. It used to rebuild plateau equalisation over 65 536 float bins of
# apparent temperature, one LUT for the whole run, which is no camera: no bin ever reached the
# plateau, so it was full histogram equalisation and the drone got 3 grey codes. The camera
# re-maps every frame, as a real one does (it has no damping yet, `SC.10`), so the legend is what
# carries the warm-up. Display only -- the float32 planes are untouched.
parser.add_argument(
    "--agc-clip",
    action="store_true",
    help="also write a third clip: the camera's own 8-bit display (its ISP, per frame, §11.3), "
    "where the aircraft and the cloud share the frame as a real camera shows them",
)
parser.add_argument(
    "--sky-clip",
    action="store_true",
    help="also write a second clip stretched to the whole frame, showing the sky and cloud "
    "structure the aircraft's own span clips away",
)
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
# AT.31 (ADR 0169): one switch for both bands' clouds. `path_traced` draws weather-fx's 3-D volumes
# in the visible frame and marches the infrared to each hit, so a drone behind a cloud is dimmed in
# both; `real_time` keeps the visible cloud on the dome and skips that march, so it is dimmed in
# neither. The cloud is in the sky in both bands either way.
parser.add_argument(
    "--cloud-tier",
    default="path_traced",
    choices=("path_traced", "real_time", "pixel"),
    help="path_traced: 3-D cloud volumes and infrared occlusion; real_time: dome cloud, none; "
    "pixel: the cloudscape (simulated patches) marched per pixel in the visible band and read "
    "by the infrared march, occluding in both (WX.26)",
)
parser.add_argument(
    "--cloud-stride",
    type=int,
    default=1,
    help="march the infrared cloud on every Nth native pixel (edges are re-marched at full "
    "pitch either way, ADR 0181); 1 is every pixel. A speed knob for the CPU march only",
)
parser.add_argument(
    "--cpu-cloud-march",
    action="store_true",
    help="march the infrared cloud on the CPU reference even when Warp and a CUDA device are "
    "present (the GPU twin, irsim_isaac.cloud_march_gpu, is the default where it can run)",
)
parser.add_argument(
    "--freeze-clouds",
    action="store_true",
    help="hold the cloud field still for the whole clip. By default it moves down the state's "
    "wind (surface wind times clouds.wind_factor) as the clip's clock runs, in the visible "
    "layer and every infrared band's march alike (WX.28)",
)
parser.add_argument(
    "--cloud-steps",
    type=int,
    default=None,
    help="cap on the infrared cloud march's steps per ray (the deck's own default, 512, "
    "otherwise); 192 is two and a half times faster on the cloudscape and still three samples "
    "per 30 m on the longest ray",
)
parser.add_argument(
    "--prop-spin-deg",
    type=float,
    default=0.0,
    help="turn every `propeller_*` prim about its own hub by this many degrees per frame "
    "(front-left and rear-right one way, the other pair the other, as a quadrotor's do); "
    "0 leaves the propellers still. A still frame shows a blade, not a blurred disc.",
)
parser.add_argument(
    "--cloud-set",
    nargs="*",
    default=[],
    metavar="NAME=VALUE",
    help="weather-fx cloud settings laid over the drawn weather, by their own names: "
    "genus=cumulus cover=0.35 seed=3 towers=0.35 spacing_m=2400 base_m=900 ...",
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


#: EV.20's defaults: the solved aircraft (TC.13) under the cloudless sky, 5 -> 60 m, at cruise.
CLEAR_EXIT_SCENE = "configs/scenes/phantom4_clear_exit.yaml"
CLEAR_EXIT_ASSET = "phantom4_parts"
CLEAR_EXIT_NEAR_M = 5.0
CLEAR_EXIT_FAR_M = 60.0
CLEAR_EXIT_FPS = 60.0
CLEAR_EXIT_MISSION_START_S = 540.0


def apply_clear_exit_defaults(args: Any) -> None:
    """Fill in what ``--clear-exit`` means when the caller left it to the defaults.

    Only arguments still at the parser's own default are replaced, so an explicit ``--scene`` or
    ``--far-m`` wins. The frame count is not set here: it follows from the track's duration, which
    needs the aircraft's size and the lens (see ``_render``).
    """
    if not args.clear_exit:
        if args.mission_start_s is None:
            args.mission_start_s = 0.0
        return
    if args.scene == parser.get_default("scene"):
        args.scene = CLEAR_EXIT_SCENE
    if args.asset == parser.get_default("asset"):
        args.asset = CLEAR_EXIT_ASSET
    if args.near_m is None:
        args.near_m = CLEAR_EXIT_NEAR_M
    if args.far_m is None:
        args.far_m = CLEAR_EXIT_FAR_M
    if args.mission_start_s is None:
        args.mission_start_s = CLEAR_EXIT_MISSION_START_S
    if args.out == parser.get_default("out"):
        args.out = "outputs/phantom4_clear_exit"
    # The Boson's own frame rate: the exit is a second and a half, and at 12 Hz it is 18 frames.
    args.fps = CLEAR_EXIT_FPS
    args.track = "outbound"


def clear_exit_summary(
    rows: list[dict[str, Any]], planes: list[Any], track: Any, args: Any, *, frame_width: int
) -> dict[str, Any]:
    """EV.20's checks, from the frame records: drawn where predicted, and gone once gone.

    * **centre error** -- the drawn centroid against the pinhole's prediction for the aircraft's
      bounding-box centre, over frames where it is at least 30 px and clear of the frame's edge.
      A silhouette's centroid is not its box centre, so this is a few pixels up close and shrinks
      with range; what it rules out is the aircraft being drawn somewhere it is not.
    * **exit** -- the last frame anything was drawn, against the first frame the prediction puts
      the whole aircraft past the left edge; and that nothing is drawn after it.
    """
    import numpy as np

    drawn = [r for r in rows if r.get("drawn_px", 0) > 0]
    last_drawn = max((r["frame"] for r in drawn), default=-1)
    predicted_gone = next(
        (r["frame"] for r in rows if r["pred_u_px"] + r["pred_half_span_px"] < 0.0), None
    )
    measurable = [r for r in drawn if r["drawn_px"] >= 30 and not r.get("touches_edge", False)]
    errors = np.asarray(
        [
            math.hypot(r["drawn_u_px"] - r["pred_u_px"], r["drawn_v_px"] - r["pred_v_px"])
            for r in measurable
        ],
        dtype=np.float64,
    )
    spans = np.asarray([2.0 * r["pred_half_span_px"] for r in measurable], dtype=np.float64)
    empty = [i for i in range(last_drawn + 1, len(rows))]
    sky = (
        np.percentile(np.stack([planes[i] for i in empty]), [0.1, 99.9]).round(3).tolist()
        if empty
        else None
    )
    return {
        "outbound_s": track.outbound_s,
        "exit_speed_m_s": track.exit_speed_m_s,
        "exit_s": round(track.exit_s, 4),
        "hold_s": track.hold_s,
        "mission_start_s": args.mission_start_s,
        "fps": args.fps,
        "frame_width_px": frame_width,
        "last_drawn_frame": last_drawn,
        "predicted_gone_frame": predicted_gone,
        "drawn_after_exit": any(r.get("drawn_px", 0) > 0 for r in rows[last_drawn + 1 :]),
        "empty_frames": len(empty),
        "empty_sky_t_app_p001_p999_k": sky,
        "centre_error_frames": len(measurable),
        "centre_error_px_median": round(float(np.median(errors)), 3) if errors.size else None,
        "centre_error_px_p95": round(float(np.percentile(errors, 95)), 3) if errors.size else None,
        "centre_error_over_span_median": (
            round(float(np.median(errors / spans)), 4) if errors.size else None
        ),
    }


def main(argv: list[str] | None = None) -> int:
    args = parser.parse_args(argv)
    apply_clear_exit_defaults(args)
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


def _spin(yaw_deg: float) -> Any:
    """The 3x3 yaw :func:`_mount_matrix` applies: about +Y, positive from -Z toward +X."""
    import numpy as np

    yaw = math.radians(yaw_deg)
    return np.asarray(
        [
            [math.cos(yaw), 0.0, -math.sin(yaw)],
            [0.0, 1.0, 0.0],
            [math.sin(yaw), 0.0, math.cos(yaw)],
        ],
        dtype=np.float64,
    )


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


class _Propellers:
    """The asset's `propeller_*` prims, turned about their own hubs frame by frame.

    A quadrotor's propellers are the one part of it that moves in its own frame. Each prim gets
    a `spin` transform op at the end of its order (so it is applied first, in the mesh's own
    frame): a rotation about the asset's up axis through the propeller's bound centre. Diagonal
    pairs turn opposite ways. The thermal side is untouched: a propeller's temperature lives on
    its points, and the points go round with it.
    """

    def __init__(self, stage: Any, root: str, up_axis: str) -> None:
        from pxr import Gf, Usd, UsdGeom

        self._gf = Gf
        self._up = {"X": 0, "Y": 1, "Z": 2}[str(up_axis).upper()]
        self._props: list[tuple[Any, Any, float]] = []
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default"])
        root_prim = stage.GetPrimAtPath(root)
        for prim in Usd.PrimRange(root_prim) if root_prim else ():
            name = prim.GetName()
            if not name.startswith("propeller_") or not prim.IsA(UsdGeom.Xformable):
                continue
            if not any(child.IsA(UsdGeom.Mesh) for child in prim.GetChildren()):
                continue
            box = cache.ComputeLocalBound(prim).ComputeAlignedRange()
            hub = Gf.Vec3d(*[float(box.GetMidpoint()[i]) for i in range(3)])
            xf = UsdGeom.Xformable(prim)
            op = xf.AddTransformOp(opSuffix="spin")
            op.Set(Gf.Matrix4d(1.0))
            sense = -1.0 if name.endswith(("front_left", "rear_right")) else 1.0
            self._props.append((op, hub, sense))

    def __len__(self) -> int:
        return len(self._props)

    def set_angle(self, angle_deg: float) -> None:
        Gf = self._gf
        axis = Gf.Vec3d(*[1.0 if i == self._up else 0.0 for i in range(3)])
        for op, hub, sense in self._props:
            rotation = Gf.Matrix4d().SetRotate(Gf.Rotation(axis, sense * angle_deg))
            # Row-vector convention: translate to the origin, turn, translate back.
            m = Gf.Matrix4d().SetTranslate(-hub) * rotation * Gf.Matrix4d().SetTranslate(hub)
            op.Set(m)


def _render(args: Any, usd: pathlib.Path) -> int:  # noqa: PLR0915 - one driver, read top to bottom
    import numpy as np
    import omni.usd
    from pxr import Gf, Usd, UsdGeom

    from irsim.atmosphere.skylight import skylight_for_sensor
    from irsim.config.loader import (
        band_hash,
        config_hash,
        load_sensor_config,
        with_focus,
        with_integration_time_ms,
        with_vignetting_map,
    )
    from irsim.config.scene import load_scene_config
    from irsim.io.dataset import FrameWriter
    from irsim.io.labels import as_drawn, frame_labels, write_frame_labels
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
    from irsim_eval.video import (
        apparent_target_span_k,
        encode_mp4,
        ffmpeg_available,
        interior,
        overlay_readout,
        readout_margin,
        target_code_span,
        with_margin,
    )
    from irsim_isaac.aircraft_pass import look_at_quaternion
    from irsim_isaac.asset_flight import (
        FigureEightTrack,
        LostLock,
        ScatterTrack,
        StraightOutExitTrack,
        StraightOutTrack,
        WanderTrack,
        clear_exit_seconds,
        horizon_safe_aim,
        lost_lock_offset,
        lowest_framable_elevation_deg,
        tilt_matrix,
        world_frame_to_stage,
    )
    from irsim_isaac.pipeline.ir_camera import IrCamera
    from irsim_isaac.pipeline.material_ids import labels_to_paths
    from irsim_isaac.pipeline.materials_usd import prim_records
    from irsim_isaac.pipeline.mesh_bridge import MeshBinding
    from irsim_isaac.stage import author_environment
    from irsim_isaac.visible_sky import dome_spec_from_scene
    from irsim_isaac.weather_fx_stage import author_weather_fx_sky, weather_state

    out = REPO / args.out
    out.mkdir(parents=True, exist_ok=True)
    frames_dir = out / "frames"
    frames_dir.mkdir(exist_ok=True)

    # --- the scene, first: its `world_frame:` is what the mount rotation is derived from --------
    sensor = load_sensor_config(REPO / args.sensor)
    if args.integration_ms is not None:
        sensor = with_integration_time_ms(sensor, args.integration_ms)
    if args.focus_m is not None:
        focus = str(args.focus_m).strip().lower()
        sensor = with_focus(sensor, None if focus in ("inf", "infinity") else float(focus))
    if args.corner_illumination is not None:
        from irsim.optics.vignetting import radial_vignetting_map

        fpa_spec = sensor.sensor.fpa
        vignetting = out / "vignetting_map.npy"
        np.save(
            vignetting,
            radial_vignetting_map(fpa_spec.width, fpa_spec.height, args.corner_illumination),
        )
        sensor = with_vignetting_map(sensor, vignetting.resolve())
    band = sensor.sensor.band.band_id
    # ADR 0021: a photon FPA runs the whole chain on the photon table, so the sky, the atmosphere
    # and the target solvers are built in the sensor's own form. This was hard-coded "lb", which is
    # right only for the bolometer this driver was written for: a photon camera would have had its
    # scene built in energy units and its detector read in photons.
    quantity = sensor.sensor.quantity
    lut = load_band_lut_for_config(sensor, REPO / "data" / "lut")
    response = load_band_response_for_config(sensor, REPO / "data")
    skylight = skylight_for_sensor(sensor, quantity)
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

    if args.clear_exit:
        # The strafe's length needs the aircraft's size, which the stage has not loaded yet: this
        # one second is replaced by the solved one as soon as it has (below, after `span_m`).
        track: Any = StraightOutExitTrack(
            near_m=args.near_m,
            far_m=args.far_m,
            outbound_s=args.outbound_s,
            exit_speed_m_s=args.exit_speed_m_s,
            exit_s=1.0,
            hold_s=args.hold_s,
            **({} if args.elevation_deg is None else {"hold_elevation_deg": args.elevation_deg}),
        )
    elif args.track == "outbound":
        track_kwargs = {
            k: v
            for k, v in (
                ("near_m", args.near_m),
                ("far_m", args.far_m),
                ("hold_elevation_deg", args.elevation_deg),
            )
            if v is not None
        }
        track = StraightOutTrack(**track_kwargs)
    elif args.track == "scatter":
        track_kwargs = {
            k: v
            for k, v in (
                ("near_m", args.near_m),
                ("far_m", args.far_m),
                ("elevation_low_deg", args.elevation_low_deg),
                ("elevation_high_deg", args.elevation_high_deg),
            )
            if v is not None
        }
        track = ScatterTrack(
            count=args.frames,
            seed=args.scatter_seed,
            tilt_sigma_deg=args.tilt_sigma_deg,
            **track_kwargs,
        )
    elif args.track == "wander":
        track_kwargs = {
            k: v
            for k, v in (
                ("near_m", args.near_m),
                ("far_m", args.far_m),
                ("elevation_low_deg", args.elevation_low_deg),
                ("elevation_high_deg", args.elevation_high_deg),
            )
            if v is not None
        }
        track = WanderTrack(
            seed=args.scatter_seed,
            cycles=args.wander_cycles,
            tilt_sigma_deg=args.tilt_sigma_deg,
            **track_kwargs,
        )
    else:
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

    propellers = None
    if args.prop_spin_deg:
        asset_stage = Usd.Stage.Open(str(usd))
        propellers = _Propellers(stage, ASSET_ROOT, UsdGeom.GetStageUpAxis(asset_stage))
        print(f"propellers: {len(propellers)} turn {args.prop_spin_deg:g} deg per frame")

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
    if args.clear_exit:
        # EV.20: now the strafe can be solved -- the shortest that puts the whole aircraft three
        # pixels past the field's left edge -- and the clip's length, frame count and frame
        # period follow from it, at the camera's 60 Hz rather than spread over a mission.
        lens = sensor.sensor
        exit_hfov_deg = math.degrees(
            2.0
            * math.atan(
                0.5 * lens.fpa.width * lens.fpa.pitch_um * 1e-3 / lens.optics.focal_length_mm
            )
        )
        exit_ifov_mrad = lens.fpa.pitch_um / lens.optics.focal_length_mm
        track = dataclasses.replace(
            track,
            exit_s=clear_exit_seconds(
                track.far_m, exit_hfov_deg, exit_ifov_mrad, span_m, track.exit_speed_m_s
            ),
        )
        args.frames = max(int(round(track.duration_s * args.fps)), 2)
        args.mission_s = track.duration_s
        interval_s = track.duration_s / args.frames
        print(
            f"clear exit: out {track.near_m:.0f} -> {track.far_m:.0f} m "
            f"in {track.outbound_s:.1f} s, "
            f"strafe {track.exit_speed_m_s:.1f} m/s for {track.exit_s:.2f} s, hold "
            f"{track.hold_s:.1f} s: {args.frames} frames at {args.fps:.0f} Hz from mission "
            f"second {args.mission_start_s:.0f}"
        )
    # The archive's nose axis, carried through the mount, is what a yaw of zero has to mean.
    nose_in_asset = (
        NOSE_IN_ASSET
        if args.nose_deg is None
        else (math.cos(math.radians(args.nose_deg)), math.sin(math.radians(args.nose_deg)), 0.0)
    )
    nose_yaw_deg = float(
        math.degrees(
            math.atan2(*(lambda v: (v[0], -v[2]))(mount @ np.asarray(nose_in_asset, float)))
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
            overrides=({"clouds": _cloud_overrides(args.cloud_set)} if args.cloud_set else None),
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

        # AT.30: the dome is baked around the observer and the infrared march starts there too.
        weather_sky = author_weather_fx_sky(
            stage,
            state,
            texture_dir=out,
            anchor_m=tuple(float(v) for v in track.observer_m),
            tier=args.cloud_tier,
            camera_path="/World/IrCamera",
            gpu_march=False if args.cpu_cloud_march else None,
        )
        if weather_sky.deck is not None and args.cloud_steps:
            weather_sky.deck.max_steps = int(args.cloud_steps)
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
                f"tau {weather_sky.deck.optical_depth:.0f} -- marched by BOTH bands, "
                f"infrared on the {'GPU' if 'Gpu' in type(weather_sky.deck).__name__ else 'CPU'}"
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
        quantity=quantity,
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
            t_rel_s=args.mission_start_s + 0.5 * args.mission_s,
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
    # A part-split archive (`<asset>_parts`, ADR 0138) shares its source's material names, so it
    # takes the source's map unless it has one of its own (phantom4_parts does).
    map_name = args.asset
    if not (REPO / "configs" / "assets" / f"{map_name}.yaml").is_file() and map_name.endswith(
        "_parts"
    ):
        map_name = map_name.removesuffix("_parts")
    library = MaterialLibrary.load()
    asset_map = load_asset_mapping(map_name, known_materials=library.names)
    if asset_map.human is not None and asset_map.human.garments:
        # HU.5: a coloured garment is a dyed variant of its library material -- alpha_sol and the
        # NIR follow the colour, the long-wave does not -- made here, never written to the library
        from irsim.materials.colour import dress_asset_materials

        dressed, library = dress_asset_materials(asset_map.materials, asset_map.human, library)
        asset_map = asset_map.model_copy(update={"materials": dressed})
    table = MaterialTable.for_sensor(library, sensor)
    resolver = MaterialResolver(
        load_mapping_rules(known_materials=library.names), list(table.names), asset=asset_map
    )
    resolutions = resolver.resolve_all(records)
    unmapped = [r.path for r in resolutions if not r.mapped]
    print(f"materials: {len(resolutions) - len(unmapped)}/{len(resolutions)} prims mapped")
    for path in unmapped[:5]:
        print(f"  UNMAPPED (renders magenta): {path}")

    # --- every prim needs a thermal node, or the camera refuses it ------------------------------
    # A part-split asset names its prims after parts, so the leaf name is the key, and the scene's
    # own target names decide how fine the nodes are (`target_for_part`). A prim nothing names
    # falls to `airframe`, which is what a shell, an arm or a leg wants in a four-node scene.
    # TC.13: a solved object's parts are thermal *surfaces* named `<object>.<part>`; the bridge
    # resolves either kind, so both are on offer and the object's own surface wins for its parts.
    defined_targets = set(scene.targets) | set(scene.thermal_surfaces)
    prim_to_target = {
        record.path: target_for_part(
            part_of_leaf(
                record.path.rsplit("/", 1)[-1],
                defined_targets,
                tuple(scene.objects) + tuple(scene.humans),
            ),
            defined_targets,
            objects=tuple(scene.objects) + tuple(scene.humans),  # HU.4: `<human>.skin_<Segment>`
        )
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

    vfov_rad = 2.0 * math.atan(
        0.5 * fpa.height * fpa.pitch_um * 1e-3 / sensor.sensor.optics.focal_length_mm
    )
    # Scatter: where in the frame each aircraft sits, drawn once so a sub-frame re-pose lands on
    # the same boresight. The horizon check is the one `render_quad_outbound` makes -- this scene
    # authors no terrain (ADR 0060), so a frame that reaches the horizon has analytic ground in
    # it -- taken at the lowest boresight the jitter can produce.
    aim_offsets = np.zeros((max(args.frames, 1), 2), dtype=np.float64)
    half_fov = np.asarray([0.5 * hfov_rad, 0.5 * vfov_rad])
    if args.track in ("scatter", "wander"):
        if args.track == "scatter":
            aim_offsets = (
                np.random.default_rng(args.scatter_seed + 1).uniform(
                    -args.aim_jitter, args.aim_jitter, size=(max(args.frames, 1), 2)
                )
                * half_fov
            )
        # EV.10: the aim is raised wherever a frame would reach the horizon (`horizon_safe_aim`),
        # so the band is limited only by the lowest target the frame can hold above it.
        lowest = lowest_framable_elevation_deg(math.degrees(0.5 * vfov_rad))
        if track.elevation_low_deg < lowest:
            raise SystemExit(
                f"{args.track}: a target at {track.elevation_low_deg:.1f} deg cannot be framed "
                f"with the frame's bottom edge above the horizon in a "
                f"{math.degrees(vfov_rad):.1f} deg vertical field; the lowest is {lowest:.2f} deg"
            )

    def aim_at(position: Any, offset: Any = (0.0, 0.0)) -> None:
        view = np.asarray(position, dtype=np.float64) - eye
        if offset[0] or offset[1]:
            # Slew the boresight off the aircraft by (azimuth, elevation) so it lands off-centre.
            slant = float(np.linalg.norm(view))
            right = np.cross(view, [0.0, 1.0, 0.0])
            right /= max(float(np.linalg.norm(right)), 1e-12)
            up = np.cross(right, view / slant)
            view = view + slant * (math.tan(offset[0]) * right + math.tan(offset[1]) * up)
        q = look_at_quaternion(view)
        aim_op.Set(Gf.Quatf(float(q[0]), Gf.Vec3f(*(float(v) for v in q[1:]))))

    focal_px = sensor.sensor.optics.focal_length_mm / (fpa.pitch_um * 1e-3)

    def predicted_px(position: Any, aim: Any) -> tuple[float, float]:
        """Where a pinhole on ``aim_at``'s boresight puts ``position``, in pixel-edge coordinates.

        The same frame `look_at_quaternion` builds -- right is forward x +Y, up is right x
        forward -- so this is the prediction the drawn centroid is checked against (EV.20).
        """
        forward = np.asarray(aim, dtype=np.float64) - eye
        forward /= max(float(np.linalg.norm(forward)), 1e-12)
        right = np.cross(forward, [0.0, 1.0, 0.0])
        right /= max(float(np.linalg.norm(right)), 1e-12)
        up = np.cross(right, forward)
        d = np.asarray(position, dtype=np.float64) - eye
        depth = float(d @ forward)
        return (
            0.5 * fpa.width + focal_px * float(d @ right) / depth,
            0.5 * fpa.height - focal_px * float(d @ up) / depth,
        )

    def above_horizon(position: Any, offset: Any) -> tuple[float, float]:
        """``offset`` (azimuth, elevation; radians) with the elevation raised where the frame
        would reach the horizon (EV.10, :func:`horizon_safe_aim`)."""
        d = np.asarray(position, dtype=np.float64) - eye
        target_deg = math.degrees(math.atan2(float(d[1]), math.hypot(float(d[0]), float(d[2]))))
        raised = horizon_safe_aim(
            target_deg, math.degrees(float(offset[1])), math.degrees(0.5 * vfov_rad)
        )
        return float(offset[0]), math.radians(float(raised))

    # EV.26: the runs in which the camera has lost the aircraft, by frame; a wander clip's frame
    # i sits at phase i / (frames - 1), so a sub-frame re-pose lands between two frames.
    lost_runs = LostLock.parse(args.lost_lock)
    if lost_runs and args.track != "wander":
        raise SystemExit("--lost-lock needs --track wander: only its mount follows the aircraft")
    if lost_runs and max(r.first + r.length for r in lost_runs) > args.frames:
        raise SystemExit(
            f"--lost-lock {args.lost_lock!r} runs past the clip's {args.frames} frames"
        )

    def lost_frame(phase: float) -> float:
        return float(phase) * (max(args.frames, 2) - 1)

    def place(phase: float) -> tuple[Any, float, float]:
        """Put the aircraft on the track at ``phase`` and slew the camera onto it.

        Returns the position, the slant range and the **aspect**: the angle between where the nose
        points and where the observer is, which is what decides how much of the aircraft the
        camera can see. 0 is head-on, +-90 broadside, 180 tail-on.
        """
        position = track.position_m(phase)
        heading = float(track.yaw_deg(phase)) + args.turn_deg
        if args.track in ("scatter", "wander") and args.tilt_sigma_deg > 0.0:
            # EV.25: lean about the aircraft's own axes -- turned to north, tilted, turned back to
            # its heading -- which is the level mount exactly when the tilt is zero.
            pitch_deg, roll_deg = (float(v) for v in track.attitude_deg(phase))
            leaned = tilt_matrix(pitch_deg, roll_deg) @ _spin(-nose_yaw_deg) @ mount
            transform_op.Set(_mount_matrix(leaned, centre, heading, position))
        else:
            transform_op.Set(_mount_matrix(mount, centre, heading - nose_yaw_deg, position))
        if args.track == "scatter":
            aim_at(
                position,
                above_horizon(position, aim_offsets[int(round(phase * (max(args.frames, 2) - 1)))]),
            )
        elif args.track == "wander":
            # The mount lags and leads the aircraft smoothly, like a tracker, not frame by frame.
            # EV.26: inside a lost run it swings off the aircraft altogether -- added after the
            # horizon raise, whose guard is for a mount still on the aircraft. The swing is never
            # down, so it cannot bring the horizon into the frame.
            azimuth, elevation = above_horizon(
                position, track.aim_fraction(phase) * args.aim_jitter * half_fov
            )
            lost = lost_lock_offset(lost_runs, lost_frame(phase), args.aim_jitter) * half_fov
            aim_at(position, (azimuth + float(lost[0]), elevation + float(lost[1])))
        elif args.clear_exit:
            # EV.20: the mount follows the run, then stops; the aircraft strafes on without it.
            aim_at(track.aim_m(phase))
        else:
            aim_at(position)
        to_observer = eye - position
        bearing = math.degrees(math.atan2(to_observer[0], -to_observer[2]))
        aspect = (heading - bearing + 180.0) % 360.0 - 180.0
        return position, float(track.range_m(phase)), aspect

    # The lemniscate closes on itself, so its last frame must *not* repeat the first: dividing by
    # `frames` leaves the clip one step short of the loop point, which is what makes it playable
    # on repeat. An outbound run does not close, and stopping one step short of `far_m` would
    # quietly make the range band in the summary a range band the clip never reached -- so it
    # spans the endpoints inclusively instead.
    phases = (
        # EV.20: a clip in seconds, so frame i is posed at the instant the camera's clock gives
        # it, (i + 1) * interval -- the last frame lands exactly on the end of the hold.
        np.arange(1, args.frames + 1, dtype=np.float64) / args.frames
        if args.clear_exit
        else np.linspace(0.0, 1.0, max(args.frames, 2), dtype=np.float64)
        if args.track != "figure8"
        else (np.arange(args.frames, dtype=np.float64) / args.frames) * args.laps
    )

    # **The wrap belongs to the closed track only.** A lemniscate is periodic, so `--laps 2` has
    # to fold back into [0, 1); an outbound run is not, and folding its final phase of exactly 1.0
    # to 0.0 puts the *first* frame at the end of the clip -- the aircraft 80 m out, then abruptly
    # back at 4 m filling the frame. It renders, it encodes, and it is wrong in the one frame a
    # reader is most likely to freeze on.
    def at(phase: Any) -> Any:
        return phase if args.track != "figure8" else np.asarray(phase) % 1.0

    ranges = track.range_m(at(phases))
    px_across = fpa.width * span_m / (2.0 * ranges * math.tan(0.5 * hfov_rad))
    print(
        f"camera: {fpa.width}x{fpa.height}, {math.degrees(hfov_rad):.1f} deg HFOV, on the ground "
        f"at {eye.round(2).tolist()}\n"
        f"track: range {ranges.min():.1f}..{ranges.max():.1f} m, "
        f"elevation {track.elevation_deg(at(phases)).min():.1f}.."
        f"{track.elevation_deg(at(phases)).max():.1f} deg, "
        f"aircraft {px_across.min():.0f}..{px_across.max():.0f} px across"
    )
    place(0.0)

    # SC.29: shutter-referenced, like every other driver and like the camera. This driver alone
    # built the pipeline bare, so every Phantom 4 clip carried the un-flat-fielded housing bowl
    # (sky corners +1230 DN over the centre on frame 96, stretched to +90 display codes) that a
    # Boson's FFC removes. The retry without the flat field is for a camera whose ADC cannot hold
    # the hot calibration point (an InSb well at 366 K against ADR 0021's +200 C).
    try:
        pipeline = PipelineConfig.from_sensor(
            sensor,
            table,
            lut,
            sky=scene.sky_models[band],
            atmosphere=scene.layered,
            flat_field_enabled=not args.no_flat_field,
        )
    except ValueError as exc:
        print(f"{sensor.sensor.name}: no flat field -- {exc}", file=sys.stderr)
        pipeline = PipelineConfig.from_sensor(
            sensor,
            table,
            lut,
            sky=scene.sky_models[band],
            atmosphere=scene.layered,
            flat_field_enabled=False,
        )
    if not args.no_chain:
        if pipeline.calibration is None:
            print(
                f"{sensor.sensor.name}: no M9 sensor chain -- a photon FPA has no radiometric "
                "calibration to convert the NUC residual into DN (ADR 0056), and this camera is "
                "shutterless. Defects and 3-D noise still apply.",
                file=sys.stderr,
            )
        else:
            from irsim.pipeline.sensor_chain import attach_sensor_chain

            pipeline = attach_sensor_chain(
                pipeline,
                scene.weather,
                t0_s=scene.t0_s,
                housing_start_offset_k=args.housing_start_k,
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
        # AT.31: the tier decides the infrared half from the same switch as the visible one, and
        # the path-traced cloud volumes are drawn in the companion only -- the infrared has the
        # same cloud as its march, and to the G-buffer a volume box is just a box.
        cloud_occlusion=True if weather_sky is None else weather_sky.occludes,
        cloud_stride=args.cloud_stride,
        # WX.26: the companion's own cloud transmittance, stored beside the march's so the two
        # bands' agreement is measured per frame (`cloud_band_agreement` below).
        visible_cloud_transmittance=(
            None if weather_sky is None else weather_sky.visible_cloud_transmittance
        ),
        companion_only_prim_paths=() if weather_sky is None else weather_sky.companion_only_paths,
        frame_period_s=interval_s,
        # EV.20: the clip starts at a mission second, not on the pad; 0 for every other track.
        start_rel_s=args.mission_start_s,
        strict_patch_coverage=False,
        strict_materials=True,
        strict_thermal_nodes=True,
        # `AI.2`'s remainder, closed: the scene's mesh surfaces reach the frame. Until now the
        # solve ran and its cells reached no pixel -- `IrCamera` took planar `SurfaceBinding`s
        # only, so a quarter of a million solved cells were discarded and the render fell back to
        # one temperature per prim. `MeshPointBridge` (WM.3) asks the prim's own triangles for the
        # closest point to each pixel's world position, which is what an imported asset needs: a
        # Phantom 4's shell is not a rectangle and cannot be made into one.
        # `frame=ASSET_ROOT` is the mounting statement, and it belongs here rather than in the
        # scene config. The archive's vertices are the asset's own world-space metres and the
        # scene must keep calling that frame `world`: `scene_forcing` refuses a patch in a moving
        # frame outright, because shadow in one needs a pose the thermal core does not carry. On
        # *this* stage those same coordinates are the local frame of the Xform the aircraft is
        # flown by, so the bridge maps each pixel's world position back through it before asking
        # the triangles. Without it every query lands metres from the mesh and the 7 mm gate
        # rejects all of them -- which is not an error, just a frame that silently does nothing.
        mesh_fields=[
            MeshBinding(path, field, frame=ASSET_ROOT) for path, field in scene.mesh_bindings()
        ],
        device=args.bridge_device,
    )
    cam.open(settle_frames=args.settle, rt_subframes=args.rt_subframes)

    #: Peak pixels per mesh-bound prim over the whole run (AI.2), keyed by the prim's leaf name.
    mesh_coverage: dict[str, int] = {}
    #: The apparent-temperature spread *across* each mesh-bound prim, at the frame where that
    #: prim covered the most pixels. This is `AI.2`'s acceptance and it has to be measured on the
    #: rendered plane rather than on the solve: the solve has had a gradient since ADR 0132 and
    #: the question was always whether any of it reached a pixel. Only pixels **wholly** on the
    #: prim count -- a silhouette pixel is part sky, and at 4 m the sky is 47 K colder, so
    #: including the rim would report the background as structure.
    mesh_span: dict[str, dict[str, float]] = {}
    bound = 0 if cam.mesh_pointwise is None else len(cam.mesh_pointwise.prim_paths)
    cells = sum(f.patch.n_cells for _, f in scene.mesh_bindings())
    print(f"mesh bridge: {bound} prim(s), {cells} cells, device {args.bridge_device}")

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
            # EV.21: what the lens was focused at, and the model that made it matter
            "focus": sensor.sensor.optics.focus.model_dump(mode="json", exclude_defaults=True)
            or {"mode": "infinity"},
            "defocus_model": sensor.sensor.optics.mtf.defocus_model,
            # EV.23: how far from settled the housing started, K (0: settled)
            "housing_start_k": args.housing_start_k,
            "corner_illumination": args.corner_illumination,
            "asset": args.asset,
            "interval_s": interval_s,
            "track": "lemniscate, one circuit per clip",
        },
    )
    planes_written = 0

    planes: list[Any] = []
    hits: list[Any] = []
    #: the target's pixels per frame (`None` when the frame carried no material ids), and the
    #: camera's own 8-bit display per frame (`None` when the config turns `display_8` off)
    masks: list[Any] = []
    displays: list[Any] = []
    rgbs: list[Any] = []
    rows: list[dict[str, Any]] = []
    started = time.time()
    agreements: list[Any] = []
    cloud_drift: Any = None
    for index in range(args.frames):
        phase = float(at(phases[index]))
        position, slant, aspect = place(phase)
        if propellers is not None:
            propellers.set_angle(args.prop_spin_deg * index)
        # Move the mount, then tell the camera it moved: every ray direction, sky elevation and
        # view cosine is built from a cached pose, and a camera that slews without refreshing
        # renders geometry from here and radiometry from where it used to be.
        cam.refresh_pose()
        if args.track == "scatter":
            # Independent poses, not a flight: without this the jump from the last aircraft to
            # this one is smeared across the frame as if it had been flown during the exposure.
            cam.restart_motion()

        # IG.19: the companion's sub-frames pose the aircraft where it is `dt` seconds from this
        # frame's instant along the same track; `pose_at(0.0)` puts it back before the infrared.
        phase_step = float(phases[1] - phases[0]) if len(phases) > 1 else 0.0
        if args.track == "scatter":
            # The next pose is another aircraft somewhere else, not this one a moment later: the
            # target hovers through its own exposure.
            phase_step = 0.0

        def pose_at(dt: float, _p: float = float(phases[index]), _step: float = phase_step) -> None:
            place(float(at(_p + dt / interval_s * _step)))

        if weather_sky is not None and not args.freeze_clouds:
            # WX.28: the clip's clock carries the cloud down the wind. The camera's own clock
            # advances one `interval_s` per capture, so this frame is `index * interval_s` in.
            cloud_drift = weather_sky.advance_clouds(index * interval_s)
        outputs = cam.get_outputs(
            rt_subframes=args.rt_subframes, rgb_subframes=args.rgb_subframes, pose_at=pose_at
        )
        # A reflective band (SWIR, NIR) switches `apparent_temperature` off in its config --
        # inverting reflected sunlight through Planck gives a number that is not a temperature --
        # so the driver carries the band radiance instead. Every span, legend and statistic below
        # then describes radiance, not kelvin, and the summary says which (`plane_quantity`).
        if outputs.apparent_t is not None:
            t_app = np.asarray(outputs.apparent_t, dtype=np.float32)
        elif outputs.radiance is not None:
            t_app = np.asarray(outputs.radiance, dtype=np.float32)
        else:
            raise RuntimeError("the sensor config produced neither apparent_t nor radiance")
        planes.append(t_app)

        # `material_id` lives on the *supersampled* grid (ADR 0014: ids are never filtered, so
        # anti-aliasing happens by supersampling them and filtering radiance afterwards). Fold it
        # down with `any`, because a display pixel showing any aircraft at all is an aircraft
        # pixel -- averaging ids would invent a material that is not in the scene.
        # `AI.2`: how many pixels actually took a cell of the mesh solve, per prim, at the
        # frame where each prim was largest. The *last* frame would be the wrong record -- at
        # the far end of an outbound run the whole aircraft is a handful of pixels, so a peak
        # is what says the cells reached the picture at all.
        if cam.mesh_pointwise is not None and cam.last_frame is not None:
            bound = set(cam.mesh_pointwise.prim_paths)
            resolved = labels_to_paths(cam.last_frame.labels)
            ident_of = np.asarray(cam.last_frame.instance_id)
            fold = ident_of.shape[0] // t_app.shape[0]
            for path, taken in cam.mesh_pointwise.last_coverage.items():
                name = path.rsplit("/", 1)[-1]
                if int(taken) <= mesh_coverage.get(name, 0):
                    continue
                mesh_coverage[name] = int(taken)
                ids_here = [i for i, q in resolved.items() if q == path and q in bound]
                if not ids_here:
                    continue
                whole = np.isin(ident_of, np.asarray(ids_here, dtype=ident_of.dtype))
                if fold > 1:
                    whole = whole.reshape(t_app.shape[0], fold, t_app.shape[1], fold).all(
                        axis=(1, 3)
                    )
                if int(whole.sum()) < 8:
                    continue
                values = t_app[whole]
                mesh_span[name] = {
                    "frame": float(index),
                    "range_m": round(float(ranges[index]), 3),
                    "interior_px": float(int(whole.sum())),
                    "min_c": round(float(values.min()) - 273.15, 3),
                    "max_c": round(float(values.max()) - 273.15, 3),
                    "span_k": round(float(values.max() - values.min()), 3),
                }

        last = cam.last_frame
        mat = None if last is None else getattr(last, "material_id", None)
        if mat is None:
            mask = None
            hit = t_app.reshape(-1)
        else:
            mask = np.asarray(mat) > 0
            k = mask.shape[0] // t_app.shape[0]
            if k > 1:
                mask = mask.reshape(t_app.shape[0], k, t_app.shape[1], k).any(axis=(1, 3))
            hit = t_app[mask]
        hits.append(hit)
        masks.append(mask)
        exit_row: dict[str, Any] = {}
        if args.clear_exit:
            # EV.20: where the aircraft should be beside where it was drawn. Pixel-edge
            # coordinates on both sides, so a centred pixel is at +0.5 and not at 0.
            u_pred, v_pred = predicted_px(position, track.aim_m(phase))
            exit_row = {
                "clip_s": round(float(track.time_s(phase)), 4),
                "pred_u_px": round(u_pred, 3),
                "pred_v_px": round(v_pred, 3),
                "pred_half_span_px": round(0.5 * focal_px * span_m / slant, 3),
                "drawn_px": 0 if mask is None else int(mask.sum()),
            }
            if mask is not None and mask.any():
                rr, cc = np.nonzero(mask)
                exit_row.update(
                    drawn_u_px=round(float(cc.mean()) + 0.5, 3),
                    drawn_v_px=round(float(rr.mean()) + 0.5, 3),
                    touches_edge=bool(
                        cc.min() == 0
                        or cc.max() == mask.shape[1] - 1
                        or rr.min() == 0
                        or rr.max() == mask.shape[0] - 1
                    ),
                )
        displays.append(
            None
            if outputs.display8 is None
            else np.ascontiguousarray(np.asarray(outputs.display8, dtype=np.uint8)[..., :3])
        )

        rgb = None if last is None else getattr(last, "rgb", None)
        if not args.no_rgb and rgb is None and index == 0:
            print(f"companion frame: {cam.rgb_problem}", file=sys.stderr)
        rgbs.append(None if rgb is None else np.ascontiguousarray(np.asarray(rgb)[..., :3]))

        temperatures = {k: float(v) for k, v in cam.bridge.temperatures().items()}
        # TC.13: a part solved as a mesh member is drawn from its own cells, and its readout
        # must be those cells too. `bridge.temperatures()` reports the scene's per-surface
        # facet estimate, which the full solve does not drive: it showed the motors at a flat
        # 30 C while their rendered cells ran to 53 C. The area-weighted cell mean is the
        # number `tests/unit/test_phantom4_solved.py` measures the bells by.
        t_frame = scene.t0_s + float(cam.bridge.t_rel_s)
        for name, fld in scene.mesh_fields.items():
            if name in temperatures:
                cells = np.asarray(fld.temperature_at(t_frame), dtype=np.float64)
                temperatures[name] = float(fld.patch.area_weighted_mean(cells))
        node_k = {k: round(v, 3) for k, v in temperatures.items()}
        # WX.26: do the two bands agree about where the cloud is, in this frame? Measured on the
        # companion's transmittance against the infrared march's, never judged from the pictures.
        agreement = cam.cloud_band_agreement()
        if agreement is not None:
            agreements.append(agreement)
        rows.append(
            {
                "frame": index,
                "cloud_band_agreement": None if agreement is None else agreement.as_dict(),
                "t_rel_s": round(float(cam.t_rel_s), 2),
                "range_m": round(slant, 3),
                "aspect_deg": round(aspect, 2),
                "elevation_deg": round(float(track.elevation_deg(phase)), 2),
                "px_across": round(float(px_across[index]), 1),
                "target_px": int(hit.size),
                # EV.26: 1 inside a lost run, 0 on the aircraft, between on the swing.
                "lost_lock": round(
                    max((r.envelope(float(index)) for r in lost_runs), default=0.0), 3
                ),
                "target_min_c": float(hit.min() - 273.15) if hit.size else None,
                "target_max_c": float(hit.max() - 273.15) if hit.size else None,
                "node_c": {k: round(v - 273.15, 2) for k, v in temperatures.items()},
                "position_m": [round(float(v), 3) for v in position],
                **exit_row,
            }
        )
        if writer.wants(index):
            # What was there under each pixel centre -- true surface temperature, range, part,
            # thermal node, material -- beside what the camera reported, for the frame viewer
            # (`python -m irsim_viewer`) to read back at a clicked pixel.
            truth = cam.truth()
            extra = {} if truth is None else dict(truth.planes)
            if rgbs[-1] is not None:
                extra["rgb"] = rgbs[-1]
            writer.write(
                outputs,
                frame_index=index,
                t_s=cam.last_frame_t_s,
                extra_planes=extra,
                legends=None if truth is None else truth.legends,
                extra_metadata={
                    "range_m": round(slant, 3),
                    "aspect_deg": round(aspect, 2),
                    "attitude_deg": (
                        [round(float(v), 2) for v in track.attitude_deg(phase)]
                        if hasattr(track, "attitude_deg")
                        else None
                    ),
                    "elevation_deg": round(float(track.elevation_deg(phase)), 2),
                    "node_temperatures_k": node_k,
                },
            )
            planes_written += 1
            if (
                args.track in ("scatter", "wander")
                and truth is not None
                and "part_id" in truth.planes
            ):
                # EV.15's labels, wired: every part of the aircraft is one `drone` box, from the
                # truth plane and not from the pose, so a target cut by the frame edge is boxed
                # as far as it is in the picture and one that left it gets no box.
                legend = truth.legends["part_id"]
                names = [name for code, name in legend.items() if int(code) != 0]
                labels = frame_labels(
                    truth.planes["part_id"], legend, {"aircraft": names}, {"aircraft": "drone"}
                )
                write_frame_labels(out / "labels", f"frame_{index:06d}", labels, image_id=index)
                if outputs.display8 is not None:
                    # EV.22: the box a person would draw on this display frame -- what they can
                    # see of the target, with their margin -- beside the truth's tight one.
                    write_frame_labels(
                        out / "labels_drawn",
                        f"frame_{index:06d}",
                        as_drawn(
                            labels,
                            outputs.display8,
                            np.random.default_rng([args.scatter_seed, index, 22]),
                        ),
                        image_id=index,
                    )
        if index % 12 == 0 or index == args.frames - 1:
            r = rows[-1]
            # EV.20: once the aircraft has left the field a frame has no target pixels at all.
            drawn_c = (
                "  (not in frame)  "
                if r["target_min_c"] is None
                else f"{r['target_min_c']:6.1f} .. {r['target_max_c']:6.1f} C"
            )
            print(
                f"  frame {index:4d}  T+{r['t_rel_s']:7.0f}s  R={slant:5.1f} m  "
                f"{r['px_across']:5.1f} px  aircraft {r['target_px']:6d} px  {drawn_c}  "
                f"hottest {max(r['node_c'], key=r['node_c'].__getitem__)} "
                f"{max(r['node_c'].values()):5.1f} C"
            )
    render_s = time.time() - started
    cam.close()
    per_frame = render_s / max(args.frames, 1)
    print(f"rendered {args.frames} frames in {render_s:.0f} s ({per_frame:.1f} s/frame)")

    # --- the display span, from the aircraft's own pixels ---------------------------------------
    # A span taken over the whole frame is a span over sky: this scene is ~98 % background, so the
    # percentiles land inside it and the target saturates to one flat white shape -- a picture of
    # the sky's noise, not of an aircraft. Fixed across the clip, not per frame: a per-frame AGC
    # would rescale from each histogram and cancel the warm-up being filmed. The top is the
    # hottest target pixel (`SC.24`): a 99th percentile sat below the four motor bells and
    # clipped them to one white.
    if args.span:
        lo, hi = (float(v) + 273.15 for v in args.span.split(","))
    else:
        lo, hi = apparent_target_span_k(np.concatenate([h for h in hits if h.size]))
    print(f"display span: {lo - 273.15:.1f} .. {hi - 273.15:.1f} C, white-hot grayscale")
    gray = palette_table("gray")

    def write_display(
        kind: str,
        pair_kind: str,
        span_lo: float,
        span_hi: float,
        note: str,
        images: list[Any] | None = None,
    ) -> None:
        """Write one display clip's stills from the planes already in hand.

        `span_lo`/`span_hi` are the only thing that changes between calls: the radiometry is
        identical and the 8-bit frames are not (ADR 0068 -- a display PNG has had the stretch,
        the palette and a 256-level quantisation applied, none of which invert). `note` is the
        readout's third line, which is where the frame says which stretch it is. `images`, when
        given, are finished 8-bit frames used instead of the linear ramp (the camera's own
        display, the AGC clip); the span then only scales the legend's gauges, and the colour bar
        is left off because under an equaliser the brightness axis is no longer a temperature
        axis (`palette_scale`). The readout goes in a margin beside the frame (`with_margin`), so
        it covers no scene; the visible companion is padded by the same top margin to pair.
        """
        # EV.20: the margin is sized to this aircraft's readout. Nineteen long part names overran
        # the fixed 400 px and the colour bar sat on the right edge, so a drone leaving the field
        # disappeared under the readout before it reached the edge of the picture.
        scale = images is None and bool(sensor.sensor.outputs.apparent_temperature)
        margin, margin_right = readout_margin(rows[0]["node_c"], scale=scale)
        for index, plane in enumerate(planes):
            row = rows[index]
            if images is None:
                eight = gray[quantise_display((plane - span_lo) / max(span_hi - span_lo, 1e-6))]
            else:
                eight = images[index]
            minutes, seconds = divmod(int(row["t_rel_s"]), 60)
            readout = overlay_readout(
                eight if args.no_overlay else with_margin(eight, margin, right=margin_right),
                [
                    f"T+{minutes:02d}:{seconds:02d}   range {row['range_m']:5.1f} m   "
                    f"span {row['px_across']:5.1f} px   el {row['elevation_deg']:4.1f} deg",
                    # The nodes are **not** repeated here. The legend below the frame already
                    # draws every one of them with a bar and a value, and packing them into the
                    # header too overflows 640 px as soon as a scene has four -- which
                    # `phantom4_parts.yaml` does, so the line ran off the right edge reading
                    # `esc 20.9C   m`. A readout that is cropped is worse than one that is short.
                    f"aspect {row['aspect_deg']:+6.1f} deg   "
                    f"{len(row['node_c'])} thermal nodes, per-pixel on the bound parts",
                    note,
                ],
                {k: v + 273.15 for k, v in row["node_c"].items()},
                # A reflective band's planes are band radiance (see `plane_quantity`): a kelvin
                # gauge and a colour bar labelled "apparent T" beside them would print radiance as
                # a temperature, which is what the SWIR clip did. No span leaves the node values
                # as numbers and drops the scale (M10.23, `overlay_readout`).
                (span_lo, span_hi) if sensor.sensor.outputs.apparent_temperature else None,
                bare=args.no_overlay,
                palette=None if images is not None else gray,
            )
            write_png(frames_dir / f"{kind}_{index:05d}.png", readout)
            if rgbs[index] is not None:
                write_png(frames_dir / f"rgb_{index:05d}.png", rgbs[index])
                rgb = np.asarray(rgbs[index], dtype=np.uint8)[..., :3]
                top = readout.shape[0] - rgb.shape[0]
                rgb = np.pad(rgb, ((top, 0), (0, 0), (0, 0))) if top > 0 else rgb
                pair = np.concatenate([readout, rgb], axis=1)
                write_png(frames_dir / f"{pair_kind}_{index:05d}.png", np.ascontiguousarray(pair))

    write_display(
        "ir",
        "pair",
        lo,
        hi,
        f"1 frame / {interval_s:.0f} s   {band.upper()} "
        f"{'apparent temperature' if sensor.sensor.outputs.apparent_temperature else 'radiance'}"
        ", gray white-hot",
    )

    # The sky's own stretch. Taken over every frame rather than per frame, for the same reason the
    # aircraft's is: a per-frame AGC would rescale from each histogram and cancel the cloud field
    # drifting through the run.
    sky_span: list[float] | None = None
    if args.sky_clip:
        whole = np.concatenate([plane.reshape(-1) for plane in planes])
        sky_lo = float(np.percentile(whole, 1.0))
        sky_hi = float(np.percentile(whole, 99.0))
        sky_span = [sky_lo - 273.15, sky_hi - 273.15]
        print(f"sky span: {sky_span[0]:.1f} .. {sky_span[1]:.1f} C, the aircraft saturates")
        write_display(
            "sky",
            "skypair",
            sky_lo,
            sky_hi,
            f"1 frame / {interval_s:.0f} s   the same {band.upper()} frames, stretched to the sky",
        )

    # The camera's own display (`SC.24`): what its ISP made of each raw frame, nothing rebuilt.
    agc_isp: dict[str, Any] | None = None
    if args.agc_clip:
        if any(d is None for d in displays):
            raise RuntimeError(
                "--agc-clip shows the camera's own display8, and this sensor config turns "
                "outputs.display_8 off"
            )
        isp = sensor.sensor.isp
        agc_isp = {
            "agc": isp.agc,
            "plateau": isp.plateau,
            "linear_percent": isp.linear_percent,
            "max_gain": isp.max_gain,
            "detail_headroom": isp.detail_headroom,
        }
        for index, display in enumerate(displays):
            if masks[index] is not None:
                # interior pixels only: a rim pixel is half sky and reads the sky's codes
                rows[index]["target_codes_agc"] = target_code_span(display, interior(masks[index]))
        codes = [r["target_codes_agc"] for r in rows if r.get("target_px")]
        if codes:
            print(f"agc clip: camera {isp.agc}, target spans {min(codes)}-{max(codes)} codes")
        write_display(
            "agc",
            "agcpair",
            lo,
            hi,
            f"1 frame / {interval_s:.0f} s   camera display8: {isp.agc}, "
            f"linear {isp.linear_percent:.0%}, per frame (§11.3)",
            images=displays,
        )

    videos: dict[str, str] = {}
    if ffmpeg_available():
        for kind in ("ir", "rgb", "pair", "sky", "skypair", "agc", "agcpair"):
            if not any(frames_dir.glob(f"{kind}_*.png")):
                continue
            path = out / f"{args.asset}_{kind}.mp4"
            encode_mp4(str(frames_dir / f"{kind}_*.png"), path, fps=args.fps)
            videos[kind] = str(path)
            print(f"wrote {path}")
    else:
        print("ffmpeg not available: stills written, no video", file=sys.stderr)

    band_agreement = _agreement_summary(agreements)
    if band_agreement is not None:
        print(
            f"cloud band agreement over {band_agreement['frames']} frames: IoU min "
            f"{band_agreement['iou_min']:.3f} (median {band_agreement['iou_median']:.3f}), "
            f"p95 |d eps| max {band_agreement['p95_emissivity_error_max']:.3f}; "
            f"{'passes' if band_agreement['all_pass'] else 'FAILS'} the WX.26 bar"
        )
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
        "plane_quantity": (
            "apparent_temperature_k"
            if sensor.sensor.outputs.apparent_temperature
            else f"band_radiance_{sensor.sensor.quantity}"
        ),
        "sky_span_c": sky_span,
        "cloud_band_agreement": _agreement_summary(agreements),
        "cloud_band_mask": cam.cloud_band_mask_spec(),
        # WX.28: how far the wind carried the cloud by the last frame, metres in stage axes;
        # None when the clouds were frozen or there was no weather-fx sky.
        "cloud_drift_m": None if cloud_drift is None else [round(float(v), 1) for v in cloud_drift],
        "agc_isp": agc_isp,
        "interval_s": interval_s,
        "seconds_per_frame": round(render_s / max(args.frames, 1), 2),
        "mount_rotation": [[float(v) for v in r] for r in mount],
        "track": {
            **(
                {
                    "shape": "straight out, geometric in range, constant elevation",
                    "near_m": track.near_m,
                    "far_m": track.far_m,
                    "elevation_deg": track.hold_elevation_deg,
                }
                if isinstance(track, StraightOutTrack)
                else {
                    "shape": "straight out, then a strafe out of a fixed field (EV.20)",
                    "near_m": track.near_m,
                    "far_m": track.far_m,
                    "elevation_deg": track.hold_elevation_deg,
                }
                if isinstance(track, StraightOutExitTrack)
                else {
                    "shape": "scatter: independent range, elevation, bearing, heading per frame",
                    "seed": track.seed,
                    "near_m": track.near_m,
                    "far_m": track.far_m,
                    "elevation_deg": [track.elevation_low_deg, track.elevation_high_deg],
                    "aim_jitter": args.aim_jitter,
                }
                if isinstance(track, ScatterTrack)
                else {
                    "shape": "wander: one continuous seeded flight inside the bands",
                    "seed": track.seed,
                    "near_m": track.near_m,
                    "far_m": track.far_m,
                    "elevation_deg": [track.elevation_low_deg, track.elevation_high_deg],
                    "cycles": track.cycles,
                    "aim_jitter": args.aim_jitter,
                    "lost_lock": [[r.first, r.length, r.direction_deg] for r in lost_runs],
                    # EV.26: frames whose truth holds no aircraft pixel -- the empty frames,
                    # each labelled with an empty file.
                    "frames_without_aircraft": sum(1 for r in rows if not r["target_px"]),
                }
                if isinstance(track, WanderTrack)
                else {
                    "shape": "lemniscate of Bernoulli, one circuit per clip",
                    "centre_range_m": track.centre_range_m,
                    "half_width_m": track.half_width_m,
                    "altitude_m": [track.altitude_low_m, track.altitude_high_m],
                }
            ),
            "observer_m": list(track.observer_m),
            "range_m": [float(ranges.min()), float(ranges.max())],
            "px_across": [float(px_across.min()), float(px_across.max())],
        },
        "display_span_c": [lo - 273.15, hi - 273.15],
        "videos": videos,
        "rows": rows,
        "temperature_granularity": (
            f"per pixel on {len(mesh_coverage)} mesh-bound prims (AI.2): "
            f"{sum(mesh_coverage.values())} pixels took a cell of the scene's own solve, and the "
            f"widest gradient across one prim is "
            f"{max((v['span_k'] for v in mesh_span.values()), default=0.0):.2f} K. "
            "Unbound prims keep their per-prim node."
        ),
        "clear_exit": (
            clear_exit_summary(rows, planes, track, args, frame_width=fpa.width)
            if args.clear_exit
            else None
        ),
        "mesh_coverage_px": mesh_coverage,
        "mesh_span_across_prim": mesh_span,
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
