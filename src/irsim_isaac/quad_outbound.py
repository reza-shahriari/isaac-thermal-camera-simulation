"""A quadrotor filmed from the ground as it flies out, with a point-wise airframe (PT.9, IG.2).

:mod:`irsim_isaac.quad_flight` films a drone at one range and flies the *aircraft* past a fixed
camera. This stage answers the other half of the aerial question -- **what happens to a target as
it goes away** -- and it answers it with a field rather than a node: the deck, the belly and two
arms each carry one temperature per cell, so the picture shows a gradient across a single prim and
then shows that gradient collapse as the target falls below a pixel.

**The camera moves and the aircraft does not, and that is the physics, not the convenience.** A
patch authored in a moving prim's frame loses its shadows: `irsim.config.scene.OccluderSpec`
refuses any frame but ``world``, because a shadow cast in a moving frame needs that frame's pose
and the thermal core does not carry one. The deck's own shadow across the inner arms is the
point-wise signature of this target -- 27 K across one arm -- so it is the thing that must not be
given up. Keeping the airframe still in the stage frame keeps every occluder legal, and relative
motion is all a camera can see anyway: the target shrinks as 1/R and the slant path in front of it
grows exactly as it would if the drone were the thing moving.

**The frame is the stage's**, +X right, +Y up, -Z forward, and
``configs/scenes/quad_outbound_pointwise.yaml`` declares ``world_frame:`` so the sun is still
placed from the site and the clock. Every patch coordinate in that file is a coordinate of a prim
this module authors -- the shadow and the picture come from one description of the aircraft, and
`test_quad_outbound.py` asserts it cell by cell rather than trusting that they were kept in step.

**Nothing below the boresight is in frame.** :data:`AIM_ELEVATION_DEG` is larger than half of any
of this project's vertical fields, so the horizon never enters the picture and every pixel that is
not the aircraft is sky, computed at that ray's own elevation (ADR 0060). There is no ground plane
and no sky geometry on the stage.

docs/physics-model.md §6.6, §15 T3; ADR 0060 (the analytic background), ADR 0072 (the heat
sources), ADR 0073 (the dome), ADR 0087 (the field), ADR 0123 (this stage),
ADR 0168 (EV.14's continuous reference clip), roadmap EV.16 (the clear-sky exit clip)
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

from irsim.optics.rotor import RotorDisc
from irsim_isaac.airframe import Part, author_parts
from irsim_isaac.pipeline.rotor_isaac import RotorMount
from irsim_isaac.stage import DOME_HEIGHT, author_environment, bind_visible_look
from irsim_isaac.visible_sky import DomeSpec

__all__ = [
    "AIM_ELEVATION_DEG",
    "AIM_POINT_M",
    "REFERENCE_DURATION_S",
    "REFERENCE_FAR_M",
    "REFERENCE_NEAR_M",
    "REFERENCE_SWEEP_DEG",
    "OutboundTrack",
    "QuadOutboundStage",
    "POINTWISE_QUAD",
    "boresight",
    "build_quad_outbound",
    "clear_exit_track",
    "companion_angle_deg",
    "distant_companion_position",
    "look_direction",
    "pointwise_quad_parts",
    "reference_track",
    "rotor_mounts",
    "rotor_rpm",
    "target_image_angles_deg",
]

#: Boresight elevation, degrees above the horizontal, held for the whole run. Half the vertical
#: field of this project's widest camera is 12.6 deg (a 640x512 Boson through a 14 mm lens), so
#: 20 deg keeps the horizon out of frame with 7 deg to spare at every range. The value is checked
#: against the sensor at render time rather than trusted: a longer lens narrows the field and a
#: shorter one widens it, and a scene that quietly admitted the ground would look like weather.
AIM_ELEVATION_DEG = 20.0

#: What the camera is aimed at: the centre of the airframe, in the stage frame.
AIM_POINT_M = (0.0, 0.30, 0.0)

#: The rotor: a 12-inch two-blade prop, which is what a 0.84 m diagonal frame actually carries.
#: Adjacent motors are 0.594 m apart, so a 0.30 m disc clears its neighbour by 0.29 m.
QUAD_ROTOR = RotorDisc(radius_m=0.1524, root_m=0.02, chord_root_m=0.022, chord_tip_m=0.014)

#: Hub plane of each disc, stage-frame metres: 20 mm above the top of the motor bell (y = 0.37).
ROTOR_HUB_Y_M = 0.39

#: Where the four motors sit, in the stage frame. North is -Z (see the scene's ``world_frame:``),
#: so these are the N, E, S and W arms of the scene config, in that order.
_ARM_DIRECTIONS: tuple[tuple[str, float, float], ...] = (
    ("n", 0.0, -1.0),
    ("e", 1.0, 0.0),
    ("s", 0.0, 1.0),
    ("w", -1.0, 0.0),
)

#: Opposite motor axis to opposite motor axis, metres. **This frame is a plus, not an X**: its
#: four arms run due N, E, S and W (see :data:`_ARM_DIRECTIONS`), so two opposite motors are
#: ``2 x 0.42`` apart and not ``2 x 0.42 x sqrt 2``. The sqrt-2 form was authored here first, from
#: the X-quad convention every multirotor spec sheet quotes, and it overstated this aircraft by
#: 41 %; the number is now derived from the layout rather than asserted beside it.
SPAN_M = 2.0 * 0.42

#: Diagonally opposite propeller tips -- what a camera sees of a flying one, and what the readout
#: quotes while the discs are on. 0.84 m of airframe plus a rotor diameter.
TIP_TO_TIP_M = SPAN_M + 2.0 * QUAD_ROTOR.radius_m


def pointwise_quad_parts() -> tuple[Part, ...]:
    """Every prim of the aircraft, in the stage frame, at the scene config's own coordinates.

    The deck and the belly are **separate thin plates** rather than the top and bottom faces of
    the body box, because a patch binds to a prim and one prim cannot carry two fields facing
    opposite ways.

    **Every patched prim is 2 mm smaller than the patch bound to it**, and the body is smaller
    again. Two reasons, and both were measured rather than anticipated. A prim exactly the size of
    its patch loses its edge pixels: the sampled surface position lands a float's width outside
    the rectangle and `PointwiseTemperature` refuses the frame -- 150 deck pixels on the first
    render. And coincident faces z-fight, which in the companion visible frame is
    indistinguishable from a rendering fault in the infrared one. 2 mm is a fifth of a pixel at
    the near end of the track, so it costs the picture nothing.
    """
    parts: list[Part] = [
        # The body, inset inside the two plates. `airframe` rather than a field: a prim with no
        # patch still needs a temperature, which is what the per-prim solver is kept for.
        Part("body", "box", (0.0, 0.30, 0.0), (0.292, 0.114, 0.292), "carbon_fibre", "airframe"),
        # The two plates the fields bind to. 8 mm thick, centred on the patch planes at
        # y = 0.36 and y = 0.24, so each plane sits inside its own prim.
        Part("deck", "box", (0.0, 0.36, 0.0), (0.296, 0.008, 0.296), "carbon_fibre", "airframe"),
        Part("belly", "box", (0.0, 0.24, 0.0), (0.296, 0.008, 0.296), "carbon_fibre", "airframe"),
        # The pack, under the belly plate and not touching it.
        Part(
            "battery",
            "box",
            (0.0, 0.205, 0.0),
            (0.20, 0.06, 0.14),
            "painted_composite",
            "battery",
        ),
    ]
    for name, sx, sz in _ARM_DIRECTIONS:
        along_x = abs(sx) > 0.5
        # The arm: 0.30 m of tube from the body wall to the pod, 20 mm thick and centred on the
        # patch plane at y = 0.32. Only the north and east arms carry a field -- the scene
        # declares two, and the other two are the control that shows what a per-prim arm looks
        # like in the same frame.
        parts.append(
            Part(
                f"arm_{name}",
                "box",
                (0.30 * sx, 0.32, 0.30 * sz),
                (0.296, 0.02, 0.046) if along_x else (0.046, 0.02, 0.296),
                "carbon_fibre",
                "airframe",
            )
        )
        # The speed controller, on top of the arm at 45 % of the way out.
        parts.append(
            Part(
                f"esc_{name}",
                "box",
                (0.19 * sx, 0.34, 0.19 * sz),
                (0.07, 0.02, 0.05) if along_x else (0.05, 0.02, 0.07),
                "painted_composite",
                "esc",
            )
        )
        # The motor bell. Its top face is y = 0.37 and its inboard face is 30 mm short of the
        # axis, which is exactly where the scene's `pod_*_top` and `pod_*_s` occluders are.
        parts.append(
            Part(
                f"motor_{name}",
                "cylinder",
                (0.42 * sx, 0.34, 0.42 * sz),
                (0.06, 0.06, 0.06),
                # Anodised, not bare: bare aluminium is eps = 0.09 in this library and a hot bell
                # behind it would read barely above the reflected sky.
                "aircraft_aluminium_painted",
                "motor",
            )
        )
    return tuple(parts)


#: The aircraft, built once. A module-level constant so a test can walk it against the scene
#: config without an engine anywhere in sight.
POINTWISE_QUAD: tuple[Part, ...] = pointwise_quad_parts()


def rotor_rpm(throttle: float) -> float:
    """Disc speed from throttle, for a 12-inch prop.

    Idle to 9000 rpm, which puts the tip at 144 m/s at full throttle -- where props of this class
    actually live. ESTIMATED, and cosmetic: the veil's radiometry comes from the blade's own
    temperature and solidity, not from how fast it is turning.
    """
    return 1500.0 + 7500.0 * max(0.0, min(1.0, float(throttle)))


def rotor_mounts(throttle: float) -> dict[str, list[RotorMount]]:
    """The four discs, keyed by the prim they are bolted to (ADR 0081).

    Nothing is authored on the stage: a spinning rotor is a time-averaged occluder, not geometry,
    and :func:`irsim.pipeline.rotor_veil.inject_rotor_veils` composites it into the frame.
    """
    rpm = rotor_rpm(throttle)
    mounts = [
        RotorMount(
            disc=QUAD_ROTOR,
            offset_m=(0.42 * sx, ROTOR_HUB_Y_M, 0.42 * sz),
            axis=(0.0, 1.0, 0.0),
            rpm=rpm,
            thermal_node="airframe",
            # A quarter-turn apart so four identical pictures are not drawn; cosmetic at a
            # bolometer's shutter speed and visible at a cooled camera's.
            phase_rad=index * math.pi / 8.0,
        )
        for index, (_, sx, sz) in enumerate(_ARM_DIRECTIONS)
    ]
    return {"/World/Targets/quad": mounts}


@dataclass(frozen=True)
class OutboundTrack:
    """The camera's track: how far it is from the aircraft, and from which side.

    The range is **geometric**, not linear -- ``R(t) = near (far/near)^(t/T)``. A target's angular
    size goes as 1/R, so a linear recession spends most of the clip with the target already small
    and shrinking imperceptibly; a geometric one halves the target's size in equal intervals,
    which is what makes the collapse of the point-wise gradient readable frame by frame.

    The azimuth walks slowly around the aircraft so the deck, the belly and the arms are not seen
    from one fixed aspect for the whole run.

    **The exit (EV.16).** With ``exit_s > 0`` the mount stops tracking once ``duration_s`` is
    over: the camera holds its position and pans right through ``exit_pan_deg`` of azimuth over
    ``exit_s``, so the aircraft crosses the left edge of the frame and leaves. Then ``hold_s`` of
    empty sky. A pan about the vertical keeps the boresight's elevation, so :meth:`sees_horizon`
    still covers the whole run, and it is the picture a fixed camera takes of a drone flying out
    of its field sideways -- relative motion again, for the reason the module docstring gives.
    """

    near_m: float = 12.0
    far_m: float = 150.0
    duration_s: float = 1800.0
    elevation_deg: float = AIM_ELEVATION_DEG
    azimuth_start_deg: float = 20.0
    azimuth_sweep_deg: float = 55.0
    exit_s: float = 0.0
    exit_pan_deg: float = 0.0
    hold_s: float = 0.0

    def __post_init__(self) -> None:
        if self.near_m <= 0.0 or self.far_m <= 0.0:
            raise ValueError("both ranges must be positive")
        if self.far_m <= self.near_m:
            raise ValueError("far_m must be greater than near_m: this is an outbound track")
        if self.duration_s <= 0.0:
            raise ValueError("duration_s must be positive")
        if not 0.0 < self.elevation_deg < 90.0:
            raise ValueError("elevation_deg must lie strictly between the horizon and the zenith")
        if self.exit_s < 0.0 or self.hold_s < 0.0:
            raise ValueError("exit_s and hold_s cannot be negative")
        if (self.exit_s > 0.0) != (self.exit_pan_deg > 0.0):
            raise ValueError("an exit needs both a duration and a positive pan, or neither")

    @property
    def end_s(self) -> float:
        """Seconds from the first frame to the last: out, across the edge, and the empty sky."""
        return self.duration_s + self.exit_s + self.hold_s

    def pan_deg_at(self, t_rel_s: float) -> float:
        """How far the mount has panned right off the aircraft, degrees. 0 while it tracks."""
        if self.exit_s <= 0.0:
            return 0.0
        f = min(1.0, max(0.0, (float(t_rel_s) - self.duration_s) / self.exit_s))
        return float(self.exit_pan_deg * f)

    def range_at(self, t_rel_s: float) -> float:
        """Range in metres. Clamped at both ends, so a longer capture holds at ``far_m``."""
        f = min(1.0, max(0.0, float(t_rel_s) / self.duration_s))
        return float(self.near_m * (self.far_m / self.near_m) ** f)

    def azimuth_at(self, t_rel_s: float) -> float:
        f = min(1.0, max(0.0, float(t_rel_s) / self.duration_s))
        return float(self.azimuth_start_deg + self.azimuth_sweep_deg * f)

    def camera_position_m(self, t_rel_s: float) -> tuple[float, float, float]:
        """Where the camera stands, in the stage frame.

        Its ``y`` is well below zero at long range and that is not a mistake: with no ground plane
        on the stage, only the *relative* geometry exists, and this is the same picture as a
        camera on the ground watching an aircraft climb to ``R sin(elevation)`` above it. Nothing
        in the pipeline reads a stage coordinate as an altitude -- the sky, the atmosphere and
        every view cosine are built from each ray's own direction (ADR 0060).
        """
        r = self.range_at(t_rel_s)
        e = math.radians(self.elevation_deg)
        a = math.radians(self.azimuth_at(t_rel_s))
        horizontal = r * math.cos(e)
        return (
            AIM_POINT_M[0] - horizontal * math.sin(a),
            AIM_POINT_M[1] - r * math.sin(e),
            AIM_POINT_M[2] + horizontal * math.cos(a),
        )

    def target_altitude_m(self, t_rel_s: float) -> float:
        """How high the aircraft is above the camera, which is the number a viewer understands."""
        return self.range_at(t_rel_s) * math.sin(math.radians(self.elevation_deg))

    def pixels_across(self, t_rel_s: float, extent_m: float, ifov_mrad: float) -> float:
        """Native pixels an object of ``extent_m`` spans at this point on the track."""
        return 1e3 * float(extent_m) / self.range_at(t_rel_s) / float(ifov_mrad)

    def sees_horizon(self, vfov_deg: float) -> bool:
        """True if the bottom of the frame reaches or passes the horizon.

        The test a driver runs before it renders 300 frames of a scene with ground in it. The
        boresight elevation is constant along the track, so one comparison covers the whole run.
        """
        return self.elevation_deg - 0.5 * float(vfov_deg) <= 0.0


#: EV.14: the reference clip's range band, metres. The public aerial sets this project is compared
#: against (EV.3-EV.7) film drones at tens to hundreds of metres with the target 5-40 px across;
#: every other aerial clip here is a time-lapse at a few metres. 50 -> 250 m puts this frame's
#: 1.14 m tip-to-tip at 27 px through the Boson's 14 mm lens at the start and 5.3 px at the end.
REFERENCE_NEAR_M = 50.0
REFERENCE_FAR_M = 250.0

#: EV.14: seconds of flight the reference clip covers, at the sensor's own frame rate.
REFERENCE_DURATION_S = 10.0

#: EV.14: how far the camera walks round the aircraft over the clip, degrees. A tracking mount
#: follows a drone at 20 m/s through a few degrees of bearing in ten seconds, not the 55 degrees
#: the half-hour time-lapse sweeps; at 60 Hz this is 0.23 mrad per frame, a quarter of a pixel of
#: background slew per frame, which is what a pan-tilt mount tracking a drone shows.
REFERENCE_SWEEP_DEG = 8.0


def reference_track(duration_s: float = REFERENCE_DURATION_S) -> OutboundTrack:
    """EV.14's track: 50 -> 250 m in ``duration_s`` of continuous video, a few degrees of slew."""
    return OutboundTrack(
        near_m=REFERENCE_NEAR_M,
        far_m=REFERENCE_FAR_M,
        duration_s=duration_s,
        azimuth_sweep_deg=REFERENCE_SWEEP_DEG,
    )


def boresight(track: OutboundTrack, t_rel_s: float) -> tuple[float, float, float]:
    """Unit vector from the camera to :data:`AIM_POINT_M` at ``t_rel_s``, stage frame."""
    c = track.camera_position_m(t_rel_s)
    d = [AIM_POINT_M[i] - c[i] for i in range(3)]
    n = math.sqrt(sum(x * x for x in d))
    return (d[0] / n, d[1] / n, d[2] / n)


def look_direction(track: OutboundTrack, t_rel_s: float) -> tuple[float, float, float]:
    """Unit vector the camera looks along: the boresight, panned right by the exit (EV.16).

    Identical to :func:`boresight` while the mount tracks. During the exit it is the boresight
    turned about the stage's +Y, which is the vertical, so its elevation never changes.
    """
    d = boresight(track, t_rel_s)
    a = -math.radians(track.pan_deg_at(t_rel_s))  # a right pan is negative about +Y
    return (
        d[0] * math.cos(a) + d[2] * math.sin(a),
        d[1],
        -d[0] * math.sin(a) + d[2] * math.cos(a),
    )


def target_image_angles_deg(track: OutboundTrack, t_rel_s: float) -> tuple[float, float]:
    """Where the aircraft lies in the picture: (right of centre, above centre), degrees.

    The camera frame is :func:`irsim_isaac.aircraft_pass.look_at_quaternion`'s -- wings level,
    right = forward x up -- rebuilt here in plain arithmetic so a test can find the frame edge
    without an engine. The angles are atan of the camera-frame tangents, which is what a pinhole
    puts on the focal plane, so ``tan(h) / tan(hfov / 2)`` is the aircraft's place across the
    width.
    """
    f = look_direction(track, t_rel_s)
    right = (-f[2], 0.0, f[0])  # f x (0, 1, 0)
    rn = math.hypot(right[0], right[2])
    right = (right[0] / rn, 0.0, right[2] / rn)
    up = (
        right[1] * f[2] - right[2] * f[1],
        right[2] * f[0] - right[0] * f[2],
        right[0] * f[1] - right[1] * f[0],
    )
    c = track.camera_position_m(t_rel_s)
    v = [AIM_POINT_M[i] - c[i] for i in range(3)]
    z = sum(v[i] * f[i] for i in range(3))
    x = sum(v[i] * right[i] for i in range(3))
    y = sum(v[i] * up[i] for i in range(3))
    return math.degrees(math.atan2(x, z)), math.degrees(math.atan2(y, z))


#: EV.16: the clear-sky exit clip. 12 -> 150 m in 15 s is a peak recession of 25 m/s at the far
#: end of a geometric track (``far ln(far/near) / T``), the top of what a heavy-lift frame flies;
#: then a 15 m/s crossing out of the frame, and a second of empty sky.
CLEAR_EXIT_NEAR_M = 12.0
CLEAR_EXIT_FAR_M = 150.0
CLEAR_EXIT_OUTBOUND_S = 15.0
CLEAR_EXIT_SPEED_M_S = 15.0
CLEAR_EXIT_HOLD_S = 1.0

#: The mission second the clip starts at: cruise, throttle 0.66 and 15 m/s over the skin in the
#: scene's flight profile. At t = 0 the same profile is on the pad (throttle 0.15, 1 m/s), which
#: is the wrong thermal state for an aircraft filmed flying away at up to 25 m/s.
CLEAR_EXIT_MISSION_S = 900.0

#: How far past the frame edge the aircraft's nearest tip goes before the exit is over, in
#: native pixels. The optics' blur and the bolometer's lag both carry a target's signal a pixel or
#: two beyond its geometric edge; three puts the last of it outside the frame.
CLEAR_EXIT_MARGIN_PX = 3.0


def clear_exit_track(
    hfov_deg: float,
    ifov_mrad: float,
    *,
    near_m: float = CLEAR_EXIT_NEAR_M,
    far_m: float = CLEAR_EXIT_FAR_M,
    outbound_s: float = CLEAR_EXIT_OUTBOUND_S,
    exit_speed_m_s: float = CLEAR_EXIT_SPEED_M_S,
    hold_s: float = CLEAR_EXIT_HOLD_S,
    extent_m: float = TIP_TO_TIP_M,
    margin_px: float = CLEAR_EXIT_MARGIN_PX,
    sweep_deg: float = 8.0,
) -> OutboundTrack:
    """EV.16's track: out from ``near_m`` to ``far_m``, then sideways out of a ``hfov_deg`` field.

    The pan is **solved, not guessed**: the smallest azimuth turn that puts the aircraft's centre
    ``extent_m / 2`` plus ``margin_px`` pixels beyond the edge of the field, found by bisection on
    :func:`target_image_angles_deg`. Its rate is what a drone crossing at ``exit_speed_m_s`` shows
    a camera at the far range: ``v / (R cos e)`` of azimuth, because the horizontal distance to
    the aircraft is ``R cos e`` and not ``R``. The sweep is a tracking mount's few degrees, as on
    the reference clip, so the background moves slowly while the mount is following.
    """
    if exit_speed_m_s <= 0.0:
        raise ValueError("exit_speed_m_s must be positive")
    if not 0.0 < hfov_deg < 180.0:
        raise ValueError("hfov_deg must be a real field of view")
    base = OutboundTrack(
        near_m=near_m, far_m=far_m, duration_s=outbound_s, azimuth_sweep_deg=sweep_deg
    )
    edge = 0.5 * float(hfov_deg)
    half_extent = math.degrees(math.atan(0.5 * float(extent_m) / float(far_m)))
    clear_deg = edge + half_extent + float(margin_px) * 1e-3 * float(ifov_mrad) * 180.0 / math.pi

    def offset(pan: float) -> float:
        probe = OutboundTrack(
            near_m=near_m,
            far_m=far_m,
            duration_s=outbound_s,
            azimuth_sweep_deg=sweep_deg,
            exit_s=1.0,
            exit_pan_deg=pan,
        )
        return abs(target_image_angles_deg(probe, outbound_s + 1.0)[0])

    lo, hi = 0.0, 90.0
    if offset(hi) < clear_deg:
        raise ValueError(f"no pan under 90 deg clears a {hfov_deg:.1f} deg field")
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if offset(mid) < clear_deg else (lo, mid)
    rate_deg_s = math.degrees(
        float(exit_speed_m_s) / (float(far_m) * math.cos(math.radians(base.elevation_deg)))
    )
    return OutboundTrack(
        near_m=near_m,
        far_m=far_m,
        duration_s=outbound_s,
        azimuth_sweep_deg=sweep_deg,
        exit_s=hi / rate_deg_s,
        exit_pan_deg=hi,
        hold_s=hold_s,
    )


def distant_companion_position(
    track: OutboundTrack, beyond_m: float, offset_deg: float
) -> tuple[float, float, float]:
    """Where EV.14's sub-pixel companion hangs: ``beyond_m`` past the aircraft, ``offset_deg`` up.

    Placed from the track's **midpoint** view: along the camera-to-aircraft ray at mid-clip,
    tilted ``offset_deg`` towards the stage's up, at the aircraft's range plus ``beyond_m``. From
    a kilometre past the aircraft the ray to it swings far less than the boresight does, so it
    stays where it was put for the whole clip -- :func:`companion_angle_deg` is what a test reads
    to prove that rather than trust it. Static in the stage, like the aircraft: a distant drone
    holding station, which the sky-first lane needs as the sub-pixel case (ADR 0071).
    """
    if beyond_m <= 0.0:
        raise ValueError("beyond_m must be positive: the companion is behind the aircraft")
    t_mid = 0.5 * track.duration_s
    d = boresight(track, t_mid)
    up = (0.0, 1.0, 0.0)
    right = (d[1] * up[2] - d[2] * up[1], d[2] * up[0] - d[0] * up[2], d[0] * up[1] - d[1] * up[0])
    rn = math.sqrt(sum(x * x for x in right))
    right = (right[0] / rn, right[1] / rn, right[2] / rn)
    true_up = (
        right[1] * d[2] - right[2] * d[1],
        right[2] * d[0] - right[0] * d[2],
        right[0] * d[1] - right[1] * d[0],
    )
    o = math.radians(offset_deg)
    ray = tuple(math.cos(o) * d[i] + math.sin(o) * true_up[i] for i in range(3))
    reach = track.range_at(t_mid) + float(beyond_m)
    c = track.camera_position_m(t_mid)
    return (c[0] + reach * ray[0], c[1] + reach * ray[1], c[2] + reach * ray[2])


def companion_angle_deg(
    track: OutboundTrack, t_rel_s: float, position_m: tuple[float, float, float]
) -> tuple[float, float]:
    """The companion's range and its angle off the boresight at ``t_rel_s`` (metres, degrees)."""
    c = track.camera_position_m(t_rel_s)
    v = [position_m[i] - c[i] for i in range(3)]
    r = math.sqrt(sum(x * x for x in v))
    d = boresight(track, t_rel_s)
    cos_a = max(-1.0, min(1.0, sum(v[i] * d[i] for i in range(3)) / r))
    return r, math.degrees(math.acos(cos_a))


@dataclass
class QuadOutboundStage:
    """The built stage and everything a caller needs to fly the camera along the track."""

    camera_path: str
    quad_path: str
    track: OutboundTrack
    parts: tuple[Part, ...]
    prim_to_target: dict[str, str]
    #: The extent the readout quotes, metres. Defaults to this module's own heavy-lift span; a
    #: caller authoring another airframe passes that aircraft's own number, because "span" is a
    #: property of the machine and quoting one aircraft's against another's picture is how a
    #: pixel count silently becomes meaningless.
    span_m: float = SPAN_M
    errors: dict[str, str] = field(default_factory=dict)

    def thermal_nodes(self) -> tuple[str, ...]:
        """The solver names the scene has to define for this airframe to have a temperature."""
        return tuple(sorted({p.thermal_node for p in self.parts}))

    def pixels_across(self, t_rel_s: float, ifov_mrad: float) -> dict[str, float]:
        """Span and the largest and smallest part, in native pixels, at a point on the track."""
        largest = max(p.largest_dimension_m() for p in self.parts)
        smallest = min(p.largest_dimension_m() for p in self.parts)
        return {
            "span": self.track.pixels_across(t_rel_s, self.span_m, ifov_mrad),
            "largest_part": self.track.pixels_across(t_rel_s, largest, ifov_mrad),
            "smallest_part": self.track.pixels_across(t_rel_s, smallest, ifov_mrad),
        }

    def aim_camera(self, t_rel_s: float) -> None:
        """Put the camera where the track says it is at ``t_rel_s``, aimed at the aircraft -- or,
        during an exit (EV.16), panned off it along :func:`look_direction`.

        The aim is an azimuth/elevation construction (:func:`look_at_quaternion`), not the minimal
        rotation onto the boresight: the minimal rotation rolls the horizon as it slews, and a
        real two-axis pedestal does not.
        """
        import numpy as np
        import omni.usd
        from pxr import Gf, UsdGeom

        from irsim_isaac.aircraft_pass import look_at_quaternion

        stage = omni.usd.get_context().get_stage()
        eye = np.asarray(self.track.camera_position_m(t_rel_s), dtype=np.float64)
        # The boresight while tracking; panned off the aircraft during an exit (EV.16).
        q = look_at_quaternion(np.asarray(look_direction(self.track, t_rel_s), dtype=np.float64))
        xform = UsdGeom.Xformable(stage.GetPrimAtPath(self.camera_path))
        ops = {op.GetOpName(): op for op in xform.GetOrderedXformOps()}
        ops["xformOp:translate"].Set(Gf.Vec3d(*(float(v) for v in eye)))
        ops["xformOp:orient"].Set(Gf.Quatf(float(q[0]), Gf.Vec3f(*(float(v) for v in q[1:]))))


def build_quad_outbound(
    *,
    camera_path: str = "/World/IrCamera",
    quad_path: str = "/World/Targets/quad",
    track: OutboundTrack | None = None,
    parts: tuple[Part, ...] | None = None,
    span_m: float | None = None,
    dome: DomeSpec | None = None,
    dome_texture_path: str | os.PathLike[str] | None = None,
    dome_height: int = DOME_HEIGHT,
) -> QuadOutboundStage:
    """Author the stage: the environment dome, a camera on the track, and one static aircraft.

    ``parts`` chooses the airframe -- this module's own heavy-lift frame by default, or a named
    real aircraft such as :data:`irsim_isaac.phantom3.PHANTOM_3`. The stage itself does not care
    which: the scene config is what binds a field to a prim path, so an aircraft and the scene
    that describes it travel together and the stage is the thing they are both mounted on.

    No ground plane and no sky geometry (ADR 0060). The aircraft is the only geometry, so every
    pixel that is not the aircraft is sky -- and because the boresight stays above the horizon,
    every one of those pixels is sky at a positive elevation rather than ground below it.
    """
    import numpy as np
    import omni.usd
    from pxr import Gf, UsdGeom

    from irsim_isaac.aircraft_pass import look_at_quaternion

    the_track = OutboundTrack() if track is None else track
    the_parts = POINTWISE_QUAD if parts is None else parts

    ctx = omni.usd.get_context()
    ctx.new_stage()
    stage = ctx.get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    stage.DefinePrim("/World", "Xform")
    stage.DefinePrim("/World/Targets", "Xform")
    stage.DefinePrim("/World/Looks", "Scope")

    errors: dict[str, str] = {}
    try:
        author_environment(stage, dome, dome_texture_path, dome_height)
    except Exception as exc:  # noqa: BLE001 - a dark companion frame must not stop the IR render
        errors["environment"] = f"{type(exc).__name__}: {exc}"

    prim_to_target = author_parts(stage, quad_path, the_parts, look_binder=bind_visible_look)
    # The aircraft carries no transform at all: its prim coordinates *are* the scene config's
    # world coordinates, which is what lets every patch stay in the world frame and keep its
    # occluders (see this module's docstring).

    eye = np.asarray(the_track.camera_position_m(0.0), dtype=np.float64)
    q = look_at_quaternion(np.asarray(AIM_POINT_M, dtype=np.float64) - eye)
    camera = UsdGeom.Camera.Define(stage, camera_path)
    xform = UsdGeom.Xformable(camera)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(*(float(v) for v in eye)))
    xform.AddOrientOp().Set(Gf.Quatf(float(q[0]), Gf.Vec3f(*(float(v) for v in q[1:]))))
    camera.GetClippingRangeAttr().Set(Gf.Vec2f(0.05, 1.0e5))

    return QuadOutboundStage(
        camera_path=camera_path,
        quad_path=quad_path,
        track=the_track,
        parts=the_parts,
        prim_to_target=prim_to_target,
        span_m=SPAN_M if span_m is None else float(span_m),
        errors=errors,
    )
