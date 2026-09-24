"""Which way the imported Phantom 4 is facing, and why it is not a matter of opinion.

`scripts/render_phantom4.py` turns ``NOSE_IN_ASSET`` into the yaw that a heading of zero means, so
an error in it flies the aircraft crabbed through every frame of every clip while still rendering
something entirely plausible. It *was* wrong, by 28.6 degrees, from AI.2 until `AI.5`.

These tests are engine-free: the driver's module-level imports are all stdlib, so importing it
boots no Kit and touches no GPU. They re-derive the nose from the airframe's own geometry rather
than trusting the constant, which is the only way this class of error gets caught -- the wrong
value renders a perfectly good aircraft.

Roadmap `AI.5`; ADR 0138.
"""

from __future__ import annotations

import importlib.util
import math
import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]

# The asset's own centre, and the four rotor stations as `prep_asset.py --emit-components`
# measures them. Every bearing below is relative to the centre: the asset sits at y = +0.49 m in
# its own frame, so bearings taken about the origin would be meaningless.
CENTRE = (-0.0078, 0.4915)
STATIONS = {
    "front_left": (0.044, 0.314),
    "front_right": (-0.185, 0.439),
    "rear_left": (0.170, 0.544),
    "rear_right": (-0.060, 0.669),
}
# The gimbal camera body, `camera_static`. Used only as a cross-check -- it is one small component
# and the stations are four, so it does not get to set the answer.
CAMERA = (-0.040, 0.439)


def driver():
    """Import the render driver without running it."""
    path = REPO / "scripts" / "render_phantom4.py"
    spec = importlib.util.spec_from_file_location("render_phantom4", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def bearing_deg(point):
    return math.degrees(math.atan2(point[1] - CENTRE[1], point[0] - CENTRE[0]))


def test_the_four_rotor_stations_are_square_and_equidistant():
    """The premise everything else rests on: this really is an X quad, measured.

    If the stations were not 90 degrees apart at one radius, bisecting the front pair would not
    give the nose and the rest of this file would be meaningless.
    """
    radii = [math.dist(p, CENTRE) for p in STATIONS.values()]
    assert max(radii) - min(radii) < 0.001, "stations are not at one radius"
    assert 0.180 < min(radii) < 0.190, "a Phantom 4's 350 mm diagonal puts these at ~185 mm"

    bearings = sorted(bearing_deg(p) for p in STATIONS.values())
    gaps = [b - a for a, b in zip(bearings, bearings[1:], strict=False)]
    for gap in gaps:
        # 0.25 deg: the measured gaps are 89.766, 90.182 and 89.937. That is the asset's own
        # modelling tolerance, not measurement noise -- and it is four hundred times smaller than
        # the 28.6 deg error this file exists to prevent, so it constrains the bisector plenty.
        assert gap == pytest.approx(90.0, abs=0.25), f"stations are not square: {gaps}"


def test_the_nose_bisects_the_two_front_arms():
    """The correction itself.

    An X-configuration quadcopter's camera points forward *between* the two front arms, so the
    nose is their bisector. The driver's constant must be that direction.
    """
    module = driver()
    nose = module.NOSE_IN_ASSET
    assert len(nose) == 3
    assert nose[2] == 0.0, "the nose is horizontal in the asset's own frame"
    assert math.hypot(nose[0], nose[1]) == pytest.approx(1.0, abs=1e-3), "nose must be a unit"

    front = [STATIONS["front_left"], STATIONS["front_right"]]
    ux = sum((p[0] - CENTRE[0]) / math.dist(p, CENTRE) for p in front)
    uy = sum((p[1] - CENTRE[1]) / math.dist(p, CENTRE) for p in front)
    norm = math.hypot(ux, uy)
    expected = math.degrees(math.atan2(uy / norm, ux / norm))
    actual = math.degrees(math.atan2(nose[1], nose[0]))
    assert actual == pytest.approx(expected, abs=0.5), (
        f"nose bearing {actual:.2f} deg is not the front arms' bisector {expected:.2f} deg"
    )


def test_the_gimbal_camera_agrees_with_the_arms_to_within_three_degrees():
    """The independent cross-check, and the measurement the old constant botched.

    The camera hangs off the nose, so its offset from the airframe centroid points forward. Taking
    only that offset's **y** component -- which is what AI.2 did -- reads a bearing of -90 deg from
    a vector that actually points at -121.5 deg.
    """
    module = driver()
    nose_deg = math.degrees(math.atan2(module.NOSE_IN_ASSET[1], module.NOSE_IN_ASSET[0]))
    assert bearing_deg(CAMERA) == pytest.approx(nose_deg, abs=3.0)

    # the offset really is diagonal, so a y-only reading is not a rounding difference
    dx, dy = CAMERA[0] - CENTRE[0], CAMERA[1] - CENTRE[1]
    assert abs(dx) > 0.02, "a y-only reading would have been defensible if dx were negligible"
    y_only_deg = math.degrees(math.atan2(dy, 0.0))
    assert abs(y_only_deg - nose_deg) > 25.0, "the old reading was wrong by ~29 degrees"


def test_the_old_nose_value_is_not_restored():
    """-Y is the value this step removed. It is wrong by 28.6 degrees."""
    module = driver()
    nose_deg = math.degrees(math.atan2(module.NOSE_IN_ASSET[1], module.NOSE_IN_ASSET[0]))
    assert abs(nose_deg - (-90.0)) > 20.0, "NOSE_IN_ASSET is back to the uncorrected -Y"


def test_every_arm_carries_the_same_status_lamp_so_leds_cannot_mark_the_front():
    """Guards the reasoning, not just the number.

    The old docstring inferred the tail from DJI putting green LEDs aft and red forward. In this
    asset all four arms carry the same `red_light` material with equal area, so that inference
    cannot be made here -- and anyone tempted to redo it should find this test first.

    The areas are `prep_asset.py --emit-components` output for the four `red_light` components.
    """
    red_by_station = {
        "front_left": 0.00066,
        "front_right": 0.00066,
        "rear_left": 0.00066,
        "rear_right": 0.00066,
    }
    assert set(red_by_station) == set(STATIONS)
    assert len(set(red_by_station.values())) == 1, "red lamps differ between arms after all"
    # and the green lamp is a speck near the body, not an arm LED at r = 185 mm
    green_radius_m = 0.077
    assert green_radius_m < 0.5 * min(math.dist(p, CENTRE) for p in STATIONS.values())
