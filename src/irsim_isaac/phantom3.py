"""A DJI Phantom 3, to its published dimensions (PT.9, ADR 0124).

:mod:`irsim_isaac.quad_outbound`'s ``POINTWISE_QUAD`` is a *class* of aircraft -- a 1.8 m
heavy-lift frame invented to be large enough to resolve. This module is a **named, real one**,
laid out from DJI's own published specification, because the question "what does a drone look
like in the infrared" has no answer until you say which drone, and the two most-quoted numbers
for this airframe pull the answer in opposite directions:

* **350 mm diagonal**, motor axis to motor axis, against the outbound heavy-lift frame's 840 mm.
  The Phantom is **2.4x smaller across its motors and 1.9x across its props** (589 mm tip to tip
  against 1145 mm), so it falls under a pixel at roughly half the range. Small is the whole
  difficulty of the anti-UAV problem and it is a property of the *aircraft*, not of the sensor.
* **A white ABS shell.** ``abs_plastic_white`` absorbs **0.25** of the short-wave flux against
  ``carbon_fibre``'s 0.90, so the same sun puts under a third as much heat into this skin. A
  Phantom in daylight is a far weaker target than a composite one, and reporting it otherwise
  would be flattering the simulator rather than describing the aircraft.

Published specification used (DJI Phantom 3 Professional / Advanced):

===========================  =========================================
diagonal (props excluded)    350 mm
propellers                   9450, 9.4 in (239 mm) diameter, 5.0 in pitch
take-off weight              1280 g
motors                       2312 brushless, ~28 mm bell
battery                      4480 mAh 4S LiPo, in the rear of the shell
===========================  =========================================

Everything else -- shell plan of 140 mm, a 62 mm body depth, the gimbal hanging under the nose,
the twin landing skids -- is **ESTIMATED from photographs against those four dimensions**, and
said so here rather than implied by the precision of the numbers. The frame is the stage's:
+X right, +Y up, **-Z forward**, so the aircraft's nose points along -Z and its two front arms
are the -Z pair.

**What the ESCs are not.** The heavy-lift frame carries its speed controllers on the arms, where
a camera sees them. A Phantom's are on the mainboard inside the shell, so they present no surface
at all; their heat reaches a camera only by conducting into the skin, and conducting a node's heat
into a field from a scene config still has no route (PT.9's remainder). The scene therefore
declares no ``esc`` target, which is the honest statement rather than putting an invisible node
in the config.

docs/physics-model.md §6.6, §15 T3; ADR 0072 (the heat sources), ADR 0081 (the rotor veil),
ADR 0123 (the outbound stage), ADR 0124 (this airframe)
"""

from __future__ import annotations

import math

from irsim.optics.rotor import RotorDisc
from irsim_isaac.airframe import Part
from irsim_isaac.pipeline.rotor_isaac import RotorMount

__all__ = [
    "DIAGONAL_M",
    "MOTOR_OFFSET_M",
    "PROP_DIAMETER_M",
    "SHELL_PLAN_M",
    "SPAN_M",
    "TIP_TO_TIP_M",
    "PHANTOM_3",
    "PHANTOM_ROTOR",
    "phantom3_parts",
    "rotor_mounts",
    "rotor_rpm",
]

#: DJI's published diagonal, motor axis to motor axis, propellers excluded. MEASURED (published).
DIAGONAL_M = 0.350

#: Each motor's offset on one body axis. The Phantom is an X quad, so the arms sit on the
#: diagonals and each motor is ``diagonal / (2 sqrt 2)`` out along both X and Z.
MOTOR_OFFSET_M = DIAGONAL_M / (2.0 * math.sqrt(2.0))

#: The 9450 propeller: 9.4 inches across. MEASURED (published).
PROP_DIAMETER_M = 0.2394

#: Diagonally opposite propeller tips -- the extent a camera actually sees when the discs are
#: turning, and the number the readout quotes, because a spinning Phantom is its props.
TIP_TO_TIP_M = DIAGONAL_M + PROP_DIAMETER_M

#: The number DJI quotes and the one to compare against another airframe's "span".
SPAN_M = DIAGONAL_M

#: Plan size of the central shell. **ESTIMATED** from photographs scaled on the 350 mm diagonal.
SHELL_PLAN_M = 0.140

#: Depth of the shell between its top and belly plates. ESTIMATED, as above.
SHELL_DEPTH_M = 0.072

#: Height of the aircraft's mid-plane in the stage frame. Arbitrary and shared with the scene
#: config: the aircraft is static (ADR 0123) and only *relative* geometry exists on this stage.
MID_Y_M = 0.300

#: The 9450, as a disc. 5.0 inches of pitch at 9.4 inches of diameter gives a geometric pitch
#: angle at 75 % radius of ``atan(5.0 / (2 pi * 0.75 * 4.7))`` = 12.7 degrees, which is where a
#: propeller of this class actually lives -- not a number chosen to look reasonable.
PHANTOM_ROTOR = RotorDisc(
    radius_m=0.5 * PROP_DIAMETER_M,
    root_m=0.012,
    blades=2,
    chord_root_m=0.020,
    chord_tip_m=0.012,
    pitch_deg=12.7,
)

#: Hub plane of each disc, stage-frame metres: 12 mm above the top of the motor bell.
ROTOR_HUB_Y_M = 0.344

#: The four arms, as (name, x sign, z sign). -Z is forward, so ``fl``/``fr`` are the front pair --
#: the ones a Phantom carries its red stripes on.
_ARMS: tuple[tuple[str, float, float], ...] = (
    ("fl", -1.0, -1.0),
    ("fr", 1.0, -1.0),
    ("rl", -1.0, 1.0),
    ("rr", 1.0, 1.0),
)

#: Arm geometry: a 90 mm stub from the shell corner out to the pod, 32 x 24 mm in section.
_ARM_LENGTH_M = 0.090
_ARM_RADIUS_M = 0.130  # centre of the arm box, along its own diagonal
_ARM_WIDTH_M = 0.032
_ARM_DEPTH_M = 0.024

#: The 2312 motor's bell. MEASURED (published stator 23 x 12 mm); the bell over it is ESTIMATED.
_MOTOR_DIAMETER_M = 0.028
_MOTOR_HEIGHT_M = 0.020


def phantom3_parts() -> tuple[Part, ...]:
    """Every prim of the aircraft, in the stage frame, at the scene config's own coordinates.

    The top shell and the belly are **separate thin plates** rather than faces of the body box,
    because a patch binds to a prim and one prim cannot carry two fields facing opposite ways.
    Each is 2 mm smaller than the patch bound to it, for the reason ADR 0123 measured: a prim
    exactly the size of its patch loses its edge pixels to a float's width of rounding.
    """
    half = 0.5 * SHELL_PLAN_M
    plate = SHELL_PLAN_M - 0.004  # the 2 mm inset, on each side
    top_y = MID_Y_M + 0.5 * SHELL_DEPTH_M
    belly_y = MID_Y_M - 0.5 * SHELL_DEPTH_M
    parts: list[Part] = [
        # The shell itself, inset inside both plates so no two faces are coplanar.
        Part(
            "shell",
            "box",
            (0.0, MID_Y_M, 0.0),
            (plate, SHELL_DEPTH_M - 0.010, plate),
            "abs_plastic_white",
            "airframe",
        ),
        Part(
            "shell_top",
            "box",
            (0.0, top_y, 0.0),
            (plate, 0.006, plate),
            "abs_plastic_white",
            "airframe",
        ),
        Part(
            "shell_belly",
            "box",
            (0.0, belly_y, 0.0),
            (plate, 0.006, plate),
            "abs_plastic_white",
            "airframe",
        ),
        # The Intelligent Flight Battery, in the rear of the shell with its back face proud of
        # it -- which is how it comes out, and the one dark surface on a white aircraft.
        Part(
            "battery",
            "box",
            (0.0, MID_Y_M + 0.010, half - 0.018),
            (0.076, 0.044, 0.060),
            "painted_composite",
            "battery",
        ),
        # The gimbal and its camera, hanging under the nose. Both `airframe`: the camera and its
        # radio do dissipate, but no solver here is given their power, and inventing one would
        # put an unsourced hot spot in the most recognisable part of the aircraft.
        Part(
            "gimbal_arm",
            "box",
            (0.0, belly_y - 0.025, -0.048),
            (0.022, 0.048, 0.022),
            "abs_plastic_white",
            "airframe",
        ),
        Part(
            "camera",
            "box",
            (0.0, belly_y - 0.056, -0.052),
            (0.046, 0.036, 0.052),
            "abs_plastic_white",
            "airframe",
        ),
    ]
    for side in (-1.0, 1.0):
        parts.append(
            Part(
                f"leg_{'l' if side < 0 else 'r'}",
                "box",
                (side * 0.072, belly_y - 0.042, 0.018),
                (0.016, 0.080, 0.016),
                "abs_plastic_white",
                "airframe",
            )
        )
        parts.append(
            Part(
                f"skid_{'l' if side < 0 else 'r'}",
                "box",
                (side * 0.072, belly_y - 0.083, 0.010),
                (0.020, 0.014, 0.120),
                "abs_plastic_white",
                "airframe",
            )
        )
    for name, sx, sz in _ARMS:
        # The arm as an un-rotated box along +X of the full stub length; the authoring step yaws
        # it onto its diagonal, so the layout arithmetic stays one-dimensional (as in
        # `irsim_isaac.quadrotor`).
        parts.append(
            Part(
                f"arm_{name}",
                "box",
                (
                    _ARM_RADIUS_M * sx / math.sqrt(2.0),
                    MID_Y_M,
                    _ARM_RADIUS_M * sz / math.sqrt(2.0),
                ),
                (_ARM_LENGTH_M, _ARM_DEPTH_M, _ARM_WIDTH_M),
                "abs_plastic_white",
                "airframe",
                rotate_xyz_deg=(0.0, math.degrees(math.atan2(-sz, sx)), 0.0),
            )
        )
        parts.append(
            Part(
                f"motor_{name}",
                "cylinder",
                (
                    MOTOR_OFFSET_M * sx,
                    MID_Y_M + 0.5 * _ARM_DEPTH_M + 0.5 * _MOTOR_HEIGHT_M,
                    MOTOR_OFFSET_M * sz,
                ),
                (_MOTOR_DIAMETER_M, _MOTOR_HEIGHT_M, _MOTOR_DIAMETER_M),
                # Anodised, not bare: bare aluminium is eps = 0.09 in this library and a hot bell
                # behind it would read barely above the reflected sky.
                "aircraft_aluminium_painted",
                "motor",
            )
        )
    return tuple(parts)


#: The aircraft, built once, so a test can walk it against the scene config with no engine present.
PHANTOM_3: tuple[Part, ...] = phantom3_parts()


def rotor_rpm(throttle: float) -> float:
    """Disc speed from throttle, for a 9450 on a 2312 motor.

    A Phantom hovers near 5500 rpm and tops out around 8500, where the tip does 107 m/s.
    ESTIMATED within those published bounds, and cosmetic: the veil's radiometry comes from the
    blade's own temperature and solidity, not from how fast it is turning.
    """
    return 4200.0 + 4300.0 * max(0.0, min(1.0, float(throttle)))


def rotor_mounts(
    throttle: float, root_path: str = "/World/Targets/quad"
) -> dict[str, list[RotorMount]]:
    """The four discs, keyed by the prim they are bolted to (ADR 0081).

    Nothing is authored on the stage: a spinning rotor is a time-averaged occluder, not geometry.
    Note how little room a Phantom has -- adjacent motors are 247 mm apart and the discs are
    239 mm across, so they clear each other by 8 mm. That is true of the real aircraft, and it is
    why 9450 is the largest propeller this frame takes.
    """
    rpm = rotor_rpm(throttle)
    return {
        root_path: [
            RotorMount(
                disc=PHANTOM_ROTOR,
                offset_m=(MOTOR_OFFSET_M * sx, ROTOR_HUB_Y_M, MOTOR_OFFSET_M * sz),
                axis=(0.0, 1.0, 0.0),
                rpm=rpm,
                thermal_node="airframe",
                phase_rad=index * math.pi / 8.0,
            )
            for index, (_, sx, sz) in enumerate(_ARMS)
        ]
    }
