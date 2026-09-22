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
sources), ADR 0073 (the dome), ADR 0087 (the field), ADR 0123 (this stage)
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
    "OutboundTrack",
    "QuadOutboundStage",
    "POINTWISE_QUAD",
    "build_quad_outbound",
    "pointwise_quad_parts",
    "rotor_mounts",
    "rotor_rpm",
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

#: Motor axis to motor axis across the diagonal, metres -- how multirotor frames are quoted.
SPAN_M = 2.0 * 0.42 * math.sqrt(2.0)


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
    """

    near_m: float = 12.0
    far_m: float = 150.0
    duration_s: float = 1800.0
    elevation_deg: float = AIM_ELEVATION_DEG
    azimuth_start_deg: float = 20.0
    azimuth_sweep_deg: float = 55.0

    def __post_init__(self) -> None:
        if self.near_m <= 0.0 or self.far_m <= 0.0:
            raise ValueError("both ranges must be positive")
        if self.far_m <= self.near_m:
            raise ValueError("far_m must be greater than near_m: this is an outbound track")
        if self.duration_s <= 0.0:
            raise ValueError("duration_s must be positive")
        if not 0.0 < self.elevation_deg < 90.0:
            raise ValueError("elevation_deg must lie strictly between the horizon and the zenith")

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


@dataclass
class QuadOutboundStage:
    """The built stage and everything a caller needs to fly the camera along the track."""

    camera_path: str
    quad_path: str
    track: OutboundTrack
    parts: tuple[Part, ...]
    prim_to_target: dict[str, str]
    errors: dict[str, str] = field(default_factory=dict)

    def thermal_nodes(self) -> tuple[str, ...]:
        """The solver names the scene has to define for this airframe to have a temperature."""
        return tuple(sorted({p.thermal_node for p in self.parts}))

    def pixels_across(self, t_rel_s: float, ifov_mrad: float) -> dict[str, float]:
        """Span, motor and deck in native pixels at a point on the track."""
        return {
            "span": self.track.pixels_across(t_rel_s, SPAN_M, ifov_mrad),
            "deck": self.track.pixels_across(t_rel_s, 0.30, ifov_mrad),
            "motor": self.track.pixels_across(t_rel_s, 0.06, ifov_mrad),
        }

    def aim_camera(self, t_rel_s: float) -> None:
        """Put the camera where the track says it is at ``t_rel_s``, aimed at the aircraft.

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
        q = look_at_quaternion(np.asarray(AIM_POINT_M, dtype=np.float64) - eye)
        xform = UsdGeom.Xformable(stage.GetPrimAtPath(self.camera_path))
        ops = {op.GetOpName(): op for op in xform.GetOrderedXformOps()}
        ops["xformOp:translate"].Set(Gf.Vec3d(*(float(v) for v in eye)))
        ops["xformOp:orient"].Set(Gf.Quatf(float(q[0]), Gf.Vec3f(*(float(v) for v in q[1:]))))


def build_quad_outbound(
    *,
    camera_path: str = "/World/IrCamera",
    quad_path: str = "/World/Targets/quad",
    track: OutboundTrack | None = None,
    dome: DomeSpec | None = None,
    dome_texture_path: str | os.PathLike[str] | None = None,
    dome_height: int = DOME_HEIGHT,
) -> QuadOutboundStage:
    """Author the stage: the environment dome, a camera on the track, and one static quadrotor.

    No ground plane and no sky geometry (ADR 0060). The aircraft is the only geometry, so every
    pixel that is not the aircraft is sky -- and because the boresight stays above the horizon,
    every one of those pixels is sky at a positive elevation rather than ground below it.
    """
    import numpy as np
    import omni.usd
    from pxr import Gf, UsdGeom

    from irsim_isaac.aircraft_pass import look_at_quaternion

    the_track = OutboundTrack() if track is None else track

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

    prim_to_target = author_parts(stage, quad_path, POINTWISE_QUAD, look_binder=bind_visible_look)
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
        parts=POINTWISE_QUAD,
        prim_to_target=prim_to_target,
        errors=errors,
    )
