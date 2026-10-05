"""EV.20 -- the solved Phantom 4 leaves a cloudless sky (`render_phantom4.py --clear-exit`).

The geometry is checked here, engine-free: the run is :class:`StraightOutTrack`'s run unchanged,
the mount stops and the aircraft strafes out of a field that no longer moves, and the strafe is
exactly as long as it takes to put the whole aircraft past the edge. A strafe one per cent short
leaves the aircraft's near edge still in the frame, which is the failure a reader would see as
"the clip ends with the drone stuck on the border". The scene is checked to be the solved
aircraft unchanged under a sky with no cloud in it, and the driver's checks are run on records
whose answer is known.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import pathlib

import numpy as np
import pytest

from irsim.config.scene import load_scene_config
from irsim_isaac.asset_flight import StraightOutExitTrack, StraightOutTrack, clear_exit_seconds

REPO = pathlib.Path(__file__).resolve().parents[2]
SCENE = REPO / "configs" / "scenes" / "phantom4_clear_exit.yaml"
PARENT = REPO / "configs" / "scenes" / "phantom4_solved.yaml"

#: The Boson 640's field (640 x 12 um behind 14 mm: 30.7 deg), as the driver derives it.
HFOV_DEG = math.degrees(2.0 * math.atan(0.5 * 640 * 12e-3 / 14.0))
IFOV_MRAD = 12.0 / 14.0
#: The prepared Phantom 4's largest bounding-box side, about its tip-to-tip span.
EXTENT_M = 0.464


def driver():
    """Import the render driver without running it (its engine imports are inside functions)."""
    path = REPO / "scripts" / "render_phantom4.py"
    spec = importlib.util.spec_from_file_location("render_phantom4", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def track(**kwargs) -> StraightOutExitTrack:
    base = {"near_m": 5.0, "far_m": 60.0, "outbound_s": 15.0, "exit_speed_m_s": 10.0}
    base.update(kwargs)
    probe = StraightOutExitTrack(**base)
    seconds = clear_exit_seconds(
        probe.far_m, HFOV_DEG, IFOV_MRAD, EXTENT_M, probe.exit_speed_m_s, margin_px=3.0
    )
    return StraightOutExitTrack(**{**base, "exit_s": seconds})


def camera_angles_deg(t: StraightOutExitTrack, phase: float, point) -> tuple[float, float]:
    """(horizontal, vertical) angle of ``point`` off the mount's boresight at ``phase``.

    The driver's own frame -- right is forward x +Y, up is right x forward -- built independently
    of the track, so a track that moves its aim and its aircraft together cannot pass by agreeing
    with itself.
    """
    eye = np.asarray(t.observer_m, dtype=np.float64)
    forward = t.aim_m(phase) - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0.0, 1.0, 0.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    d = np.asarray(point, dtype=np.float64) - eye
    return (
        math.degrees(math.atan2(float(d @ right), float(d @ forward))),
        math.degrees(math.atan2(float(d @ up), float(d @ forward))),
    )


def phase_at(t: StraightOutExitTrack, seconds: float) -> float:
    return seconds / t.duration_s


def test_the_run_is_the_outbound_track_unchanged() -> None:
    """Up to ``outbound_s`` every position is :class:`StraightOutTrack`'s at the same fraction."""
    t = track()
    base = StraightOutTrack(near_m=5.0, far_m=60.0)
    for seconds in np.linspace(0.0, t.outbound_s, 31):
        assert np.allclose(
            t.position_m(phase_at(t, seconds)), base.position_m(seconds / t.outbound_s), atol=1e-9
        )
    assert float(t.range_m(phase_at(t, 0.0))) == pytest.approx(5.0, abs=1e-9)
    assert float(t.range_m(phase_at(t, t.outbound_s))) == pytest.approx(60.0, abs=1e-9)


def test_the_mount_follows_the_run_and_then_holds_still() -> None:
    """On the run the aircraft is on the boresight; after it, the aim does not move at all."""
    t = track()
    for seconds in np.linspace(0.0, t.outbound_s, 11):
        p = phase_at(t, seconds)
        assert camera_angles_deg(t, p, t.position_m(p)) == pytest.approx((0.0, 0.0), abs=1e-9)
    stopped = t.aim_m(phase_at(t, t.outbound_s))
    for seconds in np.linspace(t.outbound_s, t.duration_s, 11):
        assert np.allclose(t.aim_m(phase_at(t, seconds)), stopped, atol=1e-12)


def test_the_strafe_is_sideways_at_its_speed_and_keeps_its_depth() -> None:
    """-X at ``exit_speed_m_s``, perpendicular to the stopped boresight, the nose unturned."""
    t = track()
    eye = np.asarray(t.observer_m)
    forward = t.aim_m(1.0) - eye
    forward /= np.linalg.norm(forward)
    a, b = t.outbound_s + 0.2, t.outbound_s + 1.2
    step = t.position_m(phase_at(t, b)) - t.position_m(phase_at(t, a))
    assert np.allclose(step, [-t.exit_speed_m_s * (b - a), 0.0, 0.0], atol=1e-9)
    for seconds in (t.outbound_s, a, b, t.duration_s):
        depth = float((t.position_m(phase_at(t, seconds)) - eye) @ forward)
        assert depth == pytest.approx(60.0, abs=1e-9)
    assert np.all(t.yaw_deg(np.linspace(0.0, 1.0, 50)) == 0.0)


def test_the_solved_strafe_puts_the_whole_aircraft_just_past_the_edge() -> None:
    """At the end of the strafe the near wingtip is 3 px outside; 1 % sooner it is still inside.

    Measured with the camera frame above, not with the formula `clear_exit_seconds` uses.
    """
    t = track()
    edge_deg = 0.5 * HFOV_DEG
    margin_deg = math.degrees(3.0 * 1e-3 * IFOV_MRAD)

    def near_tip_deg(seconds: float) -> float:
        p = phase_at(t, seconds)
        tip = t.position_m(p) + np.asarray([0.5 * EXTENT_M, 0.0, 0.0])
        return camera_angles_deg(t, p, tip)[0]

    gone = t.outbound_s + t.exit_s
    assert near_tip_deg(gone) == pytest.approx(-(edge_deg + margin_deg), abs=1e-9)
    assert near_tip_deg(t.outbound_s + 0.99 * t.exit_s) > -edge_deg
    # and the hold keeps flying it further out, not back in
    assert near_tip_deg(t.duration_s) < near_tip_deg(gone)
    # 60 m * tan(15.6 deg) + 0.23 m at 10 m/s: about 1.7 s
    assert 1.5 < t.exit_s < 1.9


@pytest.mark.parametrize(
    "kwargs",
    [{"exit_speed_m_s": 0.0}, {"outbound_s": -1.0}, {"exit_s": 0.0}, {"hold_s": -0.1}],
)
def test_a_strafe_that_cannot_leave_is_refused(kwargs: dict) -> None:
    base = {"near_m": 5.0, "far_m": 60.0, "exit_s": 1.0}
    with pytest.raises(ValueError):
        StraightOutExitTrack(**{**base, **kwargs})
    with pytest.raises(ValueError):
        clear_exit_seconds(60.0, 180.0, IFOV_MRAD, EXTENT_M, 10.0)


def test_the_scene_is_the_solved_aircraft_under_a_cloudless_sky() -> None:
    """The thermal block, site and clock are the parent's; only the weather file differs."""
    clip = load_scene_config(SCENE).scene
    parent = load_scene_config(PARENT).scene
    assert clip.thermal.model_dump() == parent.thermal.model_dump()
    assert clip.site == parent.site
    assert clip.start_utc == parent.start_utc
    assert clip.weather_file == "weather/cloudless_midlat_summer_48h.csv"
    rows = (REPO / "data" / clip.weather_file).read_text().splitlines()
    header = next(i for i, line in enumerate(rows) if line.startswith("time_utc"))
    column = rows[header].split(",").index("cloud_fraction")
    cloud = [float(line.split(",")[column]) for line in rows[header + 1 :] if line]
    assert len(cloud) >= 48
    assert max(cloud) == 0.0


def test_the_flag_flies_the_solved_aircraft_at_cruise() -> None:
    """``--clear-exit`` alone means this scene, the part-split asset, 5 -> 60 m, 60 Hz, T+540.

    T+540 is inside the parent's forward cruise (T+330..570) and the whole clip fits in it.
    """
    module = driver()
    args = module.parser.parse_args(["--clear-exit"])
    module.apply_clear_exit_defaults(args)
    assert args.scene == "configs/scenes/phantom4_clear_exit.yaml"
    assert args.asset == "phantom4_parts"
    assert (args.near_m, args.far_m, args.fps) == (5.0, 60.0, 60.0)
    assert args.mission_start_s > 330.0
    assert args.mission_start_s + track().duration_s < 570.0
    plain = module.parser.parse_args([])
    module.apply_clear_exit_defaults(plain)
    assert plain.mission_start_s == 0.0
    assert plain.scene == module.parser.get_default("scene")
    explicit = module.parser.parse_args(["--clear-exit", "--far-m", "30"])
    module.apply_clear_exit_defaults(explicit)
    assert explicit.far_m == 30.0


def test_the_summary_sees_a_drone_drawn_where_predicted_and_then_gone() -> None:
    """Known records in, known verdict out: centroid error, last frame drawn, nothing after."""
    module = driver()
    rows = []
    for i in range(10):
        drawn = i < 7
        row = {
            "frame": i,
            "pred_u_px": 320.0 - 60.0 * i,
            "pred_v_px": 256.0,
            "pred_half_span_px": 10.0,
            "drawn_px": 200 if drawn else 0,
        }
        if drawn:
            row.update(drawn_u_px=row["pred_u_px"] + 0.3, drawn_v_px=256.4, touches_edge=i == 6)
        rows.append(row)
    planes = [np.full((4, 4), 260.0, dtype=np.float32) for _ in rows]
    t = track()
    args = argparse.Namespace(mission_start_s=540.0, fps=60.0)
    out = module.clear_exit_summary(rows, planes, t, args, frame_width=640)
    assert out["last_drawn_frame"] == 6
    assert out["predicted_gone_frame"] == 6  # 320 - 360 + 10 < 0
    assert out["drawn_after_exit"] is False
    assert out["empty_frames"] == 3
    assert out["centre_error_frames"] == 6  # the frame on the edge is left out
    assert out["centre_error_px_median"] == pytest.approx(0.5, abs=1e-9)
    assert out["empty_sky_t_app_p001_p999_k"] == [260.0, 260.0]
    rows[8]["drawn_px"] = 3
    again = module.clear_exit_summary(rows, planes, t, args, frame_width=640)
    assert again["last_drawn_frame"] == 8
