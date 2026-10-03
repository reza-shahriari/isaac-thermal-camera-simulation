"""``motion_px`` reaches a rendered frame, and the smear it causes reaches the image (IG.6).

M10.1b built the rigid-body synthesis and verified it in-sim to 0.1 px. Its only caller was that
test. `IrCamera.planes()` never set ``motion_px``, so ADR 0077's within-frame smear has never run
on a rendered frame -- with M9.8 and M10.1b both ticked and the documentation asserting otherwise.
This is the same failure ADR 0082 recorded for the membrane lag, one layer up: a mechanism that is
correct, tested, and unreachable.

The reason it survived is worth naming, because it is structural rather than careless. The
arithmetic in :mod:`irsim.optics.motion` is engine-free, but the *only* path to it ran through
:class:`MotionTracker`, whose `sample` read USD directly -- so the wiring could only be exercised
by a renderer, and a renderer is exactly what this project's fast suite does not have. IG.6 put
the USD read behind an injectable ``read``, and these tests drive the real `IrCamera.planes()`
over synthetic transforms on a CPU.

Three layers, each checking what it owns:

* the arithmetic, in pixels, against a hand-computed displacement;
* the wiring -- that the plane appears, on the right frame, with the right prims moving;
* the consequence -- §16's "lateral motion smears LWIR, not cooled MWIR", which is a statement
  about the *duty cycle* and has nothing to do with the band.

docs/physics-model.md §8.3 (``mtf_motion``), §9.2, §16; ADR 0077, ADR 0014 addendum.
"""

from __future__ import annotations

import math
import pathlib

import numpy as np
import pytest

from irsim.config.sensor import SensorConfig
from irsim.materials.mapping import Resolution
from irsim.optics.projection import Intrinsics
from irsim.optics.smear import apply_motion_smear, smear_duty
from irsim.pipeline.core import PipelineConfig
from irsim.pipeline.optics import motion_decay, motion_for_integration
from irsim.scene import Scene
from irsim_isaac.pipeline.gbuffer_isaac import RawAovs
from irsim_isaac.pipeline.ir_camera import IrCamera
from irsim_isaac.pipeline.motion_isaac import MotionTracker

REPO = pathlib.Path(__file__).resolve().parents[2]
SENSORS = REPO / "configs" / "sensors"

SCENE_YAML = REPO / "configs" / "scenes" / "sky_target_clear_day.yaml"
ROAD_ROOT = "/World"
ROAD_PATH = "/World/road"
TARGET_PATH = "/World/aircraft/fuselage"
ROOT_PATH = "/World/aircraft"
TARGET_ID = 4
CAMERA_PATH = "/World/IrCamera"
RANGE_M = 200.0


def _sensor(name: str) -> SensorConfig:
    from irsim.config.loader import load_sensor_config

    return load_sensor_config(SENSORS / name)


def _translation(dx: float = 0.0, dy: float = 0.0, dz: float = 0.0) -> np.ndarray:
    """A USD row-vector 4x4: translation lives in the **last row**, not the last column."""
    m = np.eye(4)
    m[3, :3] = (dx, dy, dz)
    return m


# --- 1. the arithmetic, in pixels --------------------------------------------------------------


def test_a_crossing_target_reports_the_pixels_it_crossed() -> None:
    """One metre of lateral motion at range R is f_px / R pixels. Checked against that, not
    against a stored array, so the test states the physics rather than a previous run.
    """
    sensor = _sensor("flir_boson_640_lwir.yaml").sensor
    intrinsics = Intrinsics.from_sensor(sensor, 1)
    f_px = float(intrinsics.fx_px)

    step_m = 1.0
    expected_px = f_px * step_m / RANGE_M

    tracker = MotionTracker([TARGET_PATH], CAMERA_PATH)
    poses = {TARGET_PATH: _translation(), CAMERA_PATH: _translation()}
    tracker.sample(read=lambda path: poses[path])
    assert not tracker.ready, "one pose is not a difference"

    poses[TARGET_PATH] = _translation(dx=step_m)
    assert tracker.sample(read=lambda path: poses[path])

    shape = (8, 10)
    # USD camera space: +Y up, -Z forward, so the surface sits at -RANGE_M in z.
    points = np.zeros((*shape, 3), dtype=np.float64)
    points[..., 2] = -RANGE_M
    ids = np.full(shape, TARGET_ID, dtype=np.int64)
    labels = {"0": "BACKGROUND", str(TARGET_ID): TARGET_PATH}

    motion = tracker.motion_px(points, ids, labels, intrinsics, sensor.optics.distortion)
    assert motion.shape == (*shape, 2) and motion.dtype == np.float32
    assert float(np.median(motion[..., 0])) == pytest.approx(expected_px, rel=1e-6)
    assert float(np.abs(motion[..., 1]).max()) == pytest.approx(0.0, abs=1e-9)


def test_a_prim_the_tracker_does_not_hold_reports_no_motion() -> None:
    """The safe default: a wrong guess would contribute plausible motion, which is worse."""
    sensor = _sensor("flir_boson_640_lwir.yaml").sensor
    intrinsics = Intrinsics.from_sensor(sensor, 1)
    tracker = MotionTracker([TARGET_PATH], CAMERA_PATH)
    poses = {TARGET_PATH: _translation(), CAMERA_PATH: _translation()}
    tracker.sample(read=lambda path: poses[path])
    poses[TARGET_PATH] = _translation(dx=5.0)
    tracker.sample(read=lambda path: poses[path])

    points = np.zeros((4, 4, 3), dtype=np.float64)
    points[..., 2] = -RANGE_M
    ids = np.full((4, 4), 9, dtype=np.int64)  # an id whose prim is not tracked
    motion = tracker.motion_px(
        points, ids, {"9": "/World/tree"}, intrinsics, sensor.optics.distortion
    )
    assert np.all(motion == 0.0)


def test_a_prim_that_appears_mid_sequence_reports_no_motion() -> None:
    """It has no previous pose. Zero is the honest answer; the identity default is what says so."""
    sensor = _sensor("flir_boson_640_lwir.yaml").sensor
    intrinsics = Intrinsics.from_sensor(sensor, 1)
    tracker = MotionTracker([], CAMERA_PATH)
    # The camera holds still in both frames; only the newly-appearing prim has moved, and it has
    # nothing to have moved *from*.
    tracker.sample(read=lambda p: _translation(), paths=[])
    tracker.sample(
        read=lambda p: _translation(dx=3.0) if p == TARGET_PATH else _translation(),
        paths=[TARGET_PATH],
    )

    points = np.zeros((4, 4, 3), dtype=np.float64)
    points[..., 2] = -RANGE_M
    ids = np.full((4, 4), TARGET_ID, dtype=np.int64)
    motion = tracker.motion_px(
        points, ids, {str(TARGET_ID): TARGET_PATH}, intrinsics, sensor.optics.distortion
    )
    assert np.all(motion == 0.0)


def test_the_camera_moving_is_motion_too() -> None:
    """A slewing camera over a static scene, which is half of what the aircraft stage does.

    Backwards-looking (where the surface point *was*), so a camera stepping +x makes the world
    appear to step -x. The sign is the one a smear kernel wants and is checked, not just the size.
    """
    sensor = _sensor("flir_boson_640_lwir.yaml").sensor
    intrinsics = Intrinsics.from_sensor(sensor, 1)
    f_px = float(intrinsics.fx_px)

    tracker = MotionTracker([TARGET_PATH], CAMERA_PATH)
    poses = {TARGET_PATH: _translation(), CAMERA_PATH: _translation()}
    tracker.sample(read=lambda path: poses[path])
    poses[CAMERA_PATH] = _translation(dx=1.0)
    tracker.sample(read=lambda path: poses[path])

    points = np.zeros((4, 4, 3), dtype=np.float64)
    points[..., 2] = -RANGE_M
    ids = np.full((4, 4), TARGET_ID, dtype=np.int64)
    motion = tracker.motion_px(
        points, ids, {str(TARGET_ID): TARGET_PATH}, intrinsics, sensor.optics.distortion
    )
    assert float(np.median(motion[..., 0])) == pytest.approx(-f_px / RANGE_M, rel=1e-6)


def test_a_rigid_child_agrees_with_its_root_and_an_articulated_one_does_not() -> None:
    """Why the tracker reads each rendered leaf, stated correctly.

    The obvious worry -- that a child offset from a rotating assembly's axis needs its own
    transform because its delta is a *conjugation* of the root's -- is **wrong**, and worth
    recording because it is the reasoning this step started from. The displacement is built as
    ``inv(cur) @ prev``, and a constant local offset cancels in the middle of that product:

        inv(L @ root_cur) @ (L @ root_prev) = inv(root_cur) @ inv(L) @ L @ root_prev
                                            = inv(root_cur) @ root_prev

    exactly, for any ``L``. So for a rigid assembly the root's matrices are not an approximation;
    they are the same number. The first half of this test measures that to float precision at a
    20-degree bank, where the conjugation argument predicted a 2 px gap.

    What the root cannot describe is a leaf with its own *changing* local transform -- and this
    project has one: ``render_quad_flight`` re-poses the rotor discs every frame from the
    throttle, so a disc's local pose differs between the two samples and ``L`` no longer cancels.
    Reading each rendered leaf covers both cases; reading the root covers only the first.
    """
    sensor = _sensor("flir_boson_640_lwir.yaml").sensor
    intrinsics = Intrinsics.from_sensor(sensor, 1)
    range_m, arm_m = 20.0, 0.5

    def rotation(deg: float) -> np.ndarray:
        theta = np.deg2rad(deg)
        m = np.eye(4)
        m[0, 0] = m[2, 2] = np.cos(theta)
        m[0, 2], m[2, 0] = np.sin(theta), -np.sin(theta)
        return m

    points = np.zeros((4, 4, 3), dtype=np.float64)
    points[..., 2] = -range_m
    ids = np.full((4, 4), TARGET_ID, dtype=np.int64)
    labels = {str(TARGET_ID): TARGET_PATH}

    def measure(prev: np.ndarray, cur: np.ndarray) -> float:
        tracker = MotionTracker([TARGET_PATH], CAMERA_PATH)
        for matrix in (prev, cur):
            tracker.sample(read=lambda path, m=matrix: m if path == TARGET_PATH else _translation())
        motion = tracker.motion_px(points, ids, labels, intrinsics, sensor.optics.distortion)
        return float(np.median(motion[..., 0]))

    # A rigid child of a banking airframe: the offset cancels, to float precision.
    bank = rotation(20.0)
    offset = _translation(dx=arm_m)
    from_root = measure(np.eye(4), bank)
    from_rigid_leaf = measure(offset @ np.eye(4), offset @ bank)
    assert from_rigid_leaf == pytest.approx(from_root, rel=1e-12), (from_rigid_leaf, from_root)
    assert abs(from_root) > 100.0, "the frames really are far apart; this is not a null comparison"

    # An articulated child -- a rotor disc turning in its own frame while the airframe banks.
    # Now the local transform differs between samples and there is nothing left to cancel.
    from_articulated = measure(offset @ np.eye(4), (offset @ rotation(35.0)) @ bank)
    assert abs(from_articulated - from_root) > 10.0, (from_articulated, from_root)


# --- 2. the wiring: the plane reaches the G-buffer --------------------------------------------


def _camera_frame_rig(aerial_sensor, aerial_materials, tophat_lwir_lut, **over):  # type: ignore[no-untyped-def]
    """The real `IrCamera` over a fake AOV reader and fake prim transforms, in the camera frame.

    This is the rig IG.6 exists to make possible. Before it, `MotionTracker.sample` read USD
    directly, so the only way to exercise the wiring was a renderer -- which is why a mechanism
    verified in-sim to 0.1 px could sit unreferenced for two milestones without anything turning
    red.

    ``position_frame="camera"`` because that is what the drivers use and what the synthesis is
    defined on: `Camera3dPositionSD` delivers camera space on this build (ADR 0014 addendum).
    """
    from test_camera_strictness import ROAD_ID, _FakeReader
    from test_camera_strictness import ROAD_PATH as RIG_ROAD

    scene = Scene.from_file(SCENE_YAML, {"lwir": tophat_lwir_lut})
    rows, columns = aerial_sensor.sensor.fpa_shape

    # One prim filling the frame, its surface a flat wall RANGE_M in front of the lens.
    position = np.zeros((rows, columns, 3), dtype=np.float32)
    position[..., 2] = -RANGE_M
    aovs = RawAovs(
        distance_m=np.full((rows, columns), RANGE_M, dtype=np.float32),
        normal=np.broadcast_to(
            np.array([0.0, 0.0, 1.0], dtype=np.float32), (rows, columns, 3)
        ).copy(),
        position=position,
        instance_id=np.full((rows, columns), ROAD_ID, dtype=np.uint32),
        device_handles={
            "instance": {"info": {"idToLabels": {"0": "BACKGROUND", str(ROAD_ID): RIG_ROAD}}}
        },
    )

    camera = IrCamera(
        aerial_sensor,
        scene,
        pipeline=PipelineConfig.from_sensor(aerial_sensor, aerial_materials, tophat_lwir_lut),
        prim_to_target={RIG_ROAD: "airframe"},
        resolutions=[
            Resolution(path=RIG_ROAD, material="aluminium", material_id=1, rule="override")
        ],
        position_frame="camera",
        up_axis="Y",
        strict_materials=False,
        strict_thermal_nodes=False,
        strict_patch_coverage=False,
        **over,
    )
    camera._up_axis = "Y"
    camera._camera_position = np.zeros(3)
    camera._camera_to_world = np.eye(4)
    camera._reader = _FakeReader(aovs)
    poses = {RIG_ROAD: _translation(), CAMERA_PATH: _translation()}
    camera._motion_read = lambda path: poses.get(path, _translation())
    return camera, poses, RIG_ROAD


@pytest.fixture()
def rig(aerial_sensor, aerial_materials, tophat_lwir_lut):  # type: ignore[no-untyped-def]
    return _camera_frame_rig(
        aerial_sensor, aerial_materials, tophat_lwir_lut, moving_prim_paths=["/World"]
    )


def test_the_first_frame_carries_no_motion_plane(rig) -> None:
    """Motion is a difference. One pose is not one, and a zero plane would claim otherwise."""
    camera, _, _ = rig
    assert "motion_px" not in camera.planes(step=False)


def test_the_second_frame_carries_one_and_it_is_the_prim_that_moved(rig) -> None:
    """The defect this step closes: `planes()` never set `motion_px` at all.

    The magnitude is the same f_px * dx / R the arithmetic layer checks, measured here through
    `IrCamera` instead -- so this fails if the wiring loses the plane *or* mangles it.
    """
    camera, poses, road = rig
    camera.planes(step=False)
    poses[road] = _translation(dx=1.0)
    planes = camera.planes(step=False)

    assert "motion_px" in planes, "the G-buffer still has no motion; the wiring is not connected"
    motion = np.asarray(planes["motion_px"])
    rows, columns = camera.sensor.sensor.fpa_shape
    assert motion.shape == (rows, columns, 2) and motion.dtype == np.float32

    expected = float(camera.optics.intrinsics.fx_px) * 1.0 / RANGE_M
    assert float(np.median(motion[..., 0])) == pytest.approx(expected, rel=1e-5)
    assert float(np.abs(motion[..., 1]).max()) == pytest.approx(0.0, abs=1e-6)


def test_a_prim_outside_the_declared_movers_stays_still(
    aerial_sensor, aerial_materials, tophat_lwir_lut
) -> None:  # type: ignore[no-untyped-def]
    """Declaring the movers is the contract; everything else is static by construction."""
    camera, poses, road = _camera_frame_rig(aerial_sensor, aerial_materials, tophat_lwir_lut)
    camera.planes(step=False)
    poses[road] = _translation(dx=1.0)
    planes = camera.planes(step=False)
    # The plane is still produced -- the camera is tracked unconditionally, and a slewing camera
    # smears a static scene -- but the prim that moved was never declared, so it contributes none.
    assert "motion_px" in planes
    assert float(np.abs(np.asarray(planes["motion_px"])).max()) == pytest.approx(0.0, abs=1e-9)


def test_the_camera_alone_smears_a_scene_with_no_declared_movers(
    aerial_sensor, aerial_materials, tophat_lwir_lut
) -> None:  # type: ignore[no-untyped-def]
    """`render_aircraft_pass` slews its pedestal; that is motion whether or not a prim moved."""
    camera, poses, _ = _camera_frame_rig(aerial_sensor, aerial_materials, tophat_lwir_lut)
    camera.planes(step=False)
    poses[CAMERA_PATH] = _translation(dx=1.0)
    motion = np.asarray(camera.planes(step=False)["motion_px"])
    expected = -float(camera.optics.intrinsics.fx_px) * 1.0 / RANGE_M
    assert float(np.median(motion[..., 0])) == pytest.approx(expected, rel=1e-5)


def test_world_space_positions_are_refused_rather_than_misread(
    aerial_sensor, aerial_materials, tophat_lwir_lut
) -> None:  # type: ignore[no-untyped-def]
    """The synthesis is defined on camera-space points and would double-apply the camera pose.

    Found by this test module: the rig was built in the world frame first and produced a plane of
    silent zeros, because every point read as being behind the lens. Costing the caller the plane
    is the right answer; giving them a wrong one is not.
    """
    camera, poses, road = _camera_frame_rig(
        aerial_sensor, aerial_materials, tophat_lwir_lut, moving_prim_paths=["/World"]
    )
    camera.position_frame = "world"
    camera.planes(step=False)
    poses[road] = _translation(dx=1.0)
    assert "motion_px" not in camera.planes(step=False)


def test_a_camera_that_was_never_opened_synthesises_nothing(
    aerial_sensor, aerial_materials, tophat_lwir_lut
) -> None:  # type: ignore[no-untyped-def]
    """No stage, no poses, no motion -- rather than a plane of invented zeros."""
    camera, poses, road = _camera_frame_rig(
        aerial_sensor, aerial_materials, tophat_lwir_lut, moving_prim_paths=["/World"]
    )
    camera._motion_read = None
    assert camera._stage is None
    poses[road] = _translation(dx=1.0)
    for _ in range(2):
        planes = camera.planes(step=False)
    assert "motion_px" not in planes


# --- 3. the consequence: LWIR smears, cooled MWIR does not -------------------------------------


def test_the_duty_cycle_is_what_separates_the_two_bands_not_the_band() -> None:
    """§16's "lateral motion smears LWIR, not cooled MWIR", reduced to its actual cause.

    A bolometer has no shutter: `integration_time_ms` is `None` for one on purpose, so it is
    sensitive for the whole frame period and smears over all of it. The cooled InSb integrates
    2 ms of a 16.7 ms frame and is idle for the rest. Nothing about 8-12 um versus 3-5 um enters
    into it -- a photon FPA run at 100 % duty would smear exactly as much, which is worth saying
    because the spec phrases the effect by band.
    """
    boson = _sensor("flir_boson_640_lwir.yaml").sensor
    insb = _sensor("example_mwir_insb_640.yaml").sensor
    assert boson.fpa.integration_time_ms is None
    assert insb.fpa.integration_time_ms == pytest.approx(2.0)
    assert boson.fpa.frame_rate_hz == insb.fpa.frame_rate_hz == 60

    lwir_duty = smear_duty(1.0 / 60.0, None)
    mwir_duty = smear_duty(1.0 / 60.0, 2.0e-3)
    assert lwir_duty == 1.0
    assert mwir_duty == pytest.approx(0.12, rel=1e-6)
    assert lwir_duty / mwir_duty == pytest.approx(8.333, rel=1e-3)


def test_a_crossing_target_smears_lwir_and_leaves_cooled_mwir_sharp() -> None:
    """The roadmap's acceptance, measured on an edge: 11 px/frame at the aircraft stage's rate.

    34 deg/s at the Boson's 0.86 mrad IFOV is about 11 pixels in a 1/60 s frame. The bolometer
    integrates all of it and the cooled MWIR 12 % of it, so the same scene velocity gives a
    ~11 px smear in one and ~1.3 px in the other. Measured as the 10-90 rise across a step,
    through `motion_for_integration` and `apply_motion_smear` -- the same two functions stage 3
    calls -- rather than by asserting the duty twice.
    """
    px_per_frame = 11.0
    edge = np.zeros((16, 96), dtype=np.float64)
    edge[:, 48:] = 1.0

    def rise_width(image: np.ndarray) -> float:
        """10-90 width of the edge profile, **interpolated**.

        The crossings are found between samples rather than snapped to them: the cooled detector's
        whole smear is about a pixel, and an integer search cannot resolve a one-pixel feature on
        a one-pixel grid -- it reported 2.0 for a 1.06 px ramp and 0.0 for a perfect step, which
        would have made this comparison a measurement of the estimator.
        """
        profile = np.asarray(image).mean(axis=0)
        columns = np.arange(profile.size, dtype=np.float64)
        lo = float(np.interp(0.1, profile, columns))
        hi = float(np.interp(0.9, profile, columns))
        return hi - lo

    def smeared_width(sensor_name: str) -> float:
        sensor = _sensor(sensor_name).sensor
        motion = np.zeros((*edge.shape, 2), dtype=np.float32)
        motion[..., 0] = px_per_frame
        scaled = motion_for_integration({"motion_px": motion}, sensor)
        assert scaled is not None
        # The pipeline's own kernel: the membrane's exponential for the bolometer, the shutter's
        # box for the photon FPA (ADR 0077 amendment).
        decay = motion_decay(sensor)
        return rise_width(np.asarray(apply_motion_smear(edge, scaled, 1.0, decay_frames=decay)))

    # The control: the same estimator on the same edge with nothing applied. A 10-90 width has a
    # floor of a pixel or two on a discretised step, so "sharp" has to be measured rather than
    # assumed to be zero -- otherwise the MWIR result gets compared against an ideal it cannot
    # reach and the test turns into an argument about the estimator.
    sharp = rise_width(edge)
    lwir = smeared_width("flir_boson_640_lwir.yaml")
    mwir = smeared_width("example_mwir_insb_640.yaml")

    # The bolometer's membrane turns a step into the exponential's CDF over the frame, so its
    # 10-90 width is the gap between the CDF's 10 % and 90 % points: with a = tau/T,
    # s_p = -a ln(1 - p (1 - e^(-1/a))) of a frame -- 7.7 px of 11 for the Boson's 8 ms, where a
    # box over the whole frame would give 0.8 w = 8.8 px. Stating it that way rather than
    # widening a tolerance around 11 keeps the test measuring the smear.
    a = 8.0e-3 * 60.0
    s10, s90 = (-a * math.log(1.0 - p * (1.0 - math.exp(-1.0 / a))) for p in (0.1, 0.9))
    assert lwir == pytest.approx((s90 - s10) * px_per_frame, abs=1.0), lwir
    assert lwir - sharp > 6.0, (lwir, sharp)

    # The cooled detector, on the same scene at the same velocity, is within a pixel of sharp.
    assert mwir - sharp == pytest.approx(0.8 * px_per_frame * 0.12, abs=0.3), (mwir, sharp)
    assert mwir - sharp < 1.5, (mwir, sharp)


def test_a_still_scene_costs_nothing() -> None:
    """`motion_px` is optional in M0.6 and a static frame must not pay for a convolution."""
    sensor = _sensor("flir_boson_640_lwir.yaml").sensor
    assert motion_for_integration({}, sensor) is None
