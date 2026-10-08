"""EV.26: a drone set has empty frames where the camera loses the drone, at the real set's rate.

In Anti-UAV-RGBT, 3,692 of 296,901 infrared frames (1.24 %) have no drone: 97 runs in 52 of 318
clips, median 14 frames (``scripts/absent_frames.py``). A synthetic set with a drone in every
frame teaches a detector that there always is one. ``generate_aerial_dataset.draw_lost_runs``
sizes a set's lost runs to the real share with real run lengths; ``render_phantom4.py
--lost-lock`` swings the mount off the aircraft for them (``irsim_isaac.asset_flight.LostLock``,
``lost_lock_offset``), and the truth plane, finding no aircraft, writes an empty label.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import itertools
import pathlib

import numpy as np
import pytest

from irsim_isaac.asset_flight import (
    LOST_CLEARANCE,
    LOST_RAMP_FRAMES,
    LostLock,
    lost_lock_offset,
)

REPO = pathlib.Path(__file__).resolve().parents[2]
CSV = REPO / "data" / "validation" / "anti_uav_absent_runs.csv"
SCENE = "configs/scenes/phantom4_pointwise.yaml"


def _planner():  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location(
        "generate_aerial_dataset", REPO / "scripts" / "generate_aerial_dataset.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _args(**overrides: object) -> argparse.Namespace:
    base = dict(
        seed=20261004, clear_runs=6, cloud_runs=0, frames_per_run=300, cloud_frames_per_run=50,
        near_m=20.0, far_m=90.0, focus="drawn", scene=SCENE, track="wander", absent_share="real",
    )  # fmt: skip
    base.update(overrides)
    return argparse.Namespace(**base)


def test_the_real_runs_are_the_measured_ones() -> None:
    """The committed file is the full set's measurement: 97 runs, 3,692 of 296,901 frames, 52 of
    318 clips, and its lengths have the measured median of 14 frames."""
    share, lengths = _planner().real_absent_runs()
    assert len(lengths) == 97 and sum(lengths) == 3692
    assert share == pytest.approx(3692 / 296901)
    lines = [x for x in CSV.read_text(encoding="utf-8").splitlines() if not x.startswith("#")]
    rows = list(csv.DictReader(lines))
    assert len({(r["split"], r["clip"]) for r in rows}) == 52
    assert float(np.median(lengths)) == 14.0


def test_the_mount_swings_off_smoothly_and_holds() -> None:
    """The envelope is 0 on the aircraft, 1 for every frame of the run, and climbs between in
    LOST_RAMP_FRAMES frames each side without a step: every frame outside a run is untouched."""
    run = LostLock(first=40, length=10, direction_deg=90.0)
    assert run.envelope(40 - LOST_RAMP_FRAMES) == 0.0
    assert run.envelope(49 + LOST_RAMP_FRAMES) == 0.0
    assert run.envelope(39) == 0.0 and run.envelope(50) == 0.0
    assert all(run.envelope(f) == 1.0 for f in range(40, 50))
    frames = np.linspace(30.0, 60.0, 3001)
    level = np.array([run.envelope(f) for f in frames])
    assert np.max(np.abs(np.diff(level))) < 0.02  # 0.01 frame steps: no jump anywhere
    rising = level[frames <= 40.0]
    assert np.all(np.diff(rising) >= 0.0)


@pytest.mark.parametrize("aim_jitter", [0.0, 0.6, 0.9])
def test_inside_a_run_the_aircraft_is_out_of_frame_whatever_the_wander_does(
    aim_jitter: float,
) -> None:
    """Held out, the boresight is past the frame edge on its larger axis by LOST_CLEARANCE
    half-fields whatever the wander adds, so an aircraft smaller than that cannot show; and it
    never swings down, which would take the frame toward the horizon."""
    run = LostLock(first=10, length=5, direction_deg=0.0)
    for direction in np.linspace(0.0, 180.0, 37):
        runs = (LostLock(run.first, run.length, float(direction)),)
        lost = lost_lock_offset(runs, 12.0, aim_jitter)
        assert lost[1] >= -1e-12
        for wander in itertools.product((-aim_jitter, aim_jitter), repeat=2):
            total = np.asarray(wander) + lost
            assert np.max(np.abs(total)) >= 1.0 + LOST_CLEARANCE - 1e-9
    assert np.all(lost_lock_offset((run,), 0.0, aim_jitter) == 0.0)


def test_the_runs_parse_as_the_planner_writes_them() -> None:
    runs = LostLock.parse("120:7:45.5, 30:2:180")
    assert runs == (LostLock(30, 2, 180.0), LostLock(120, 7, 45.5))
    assert LostLock.parse("") == ()
    with pytest.raises(ValueError):
        LostLock.parse("10:3:270")  # down, toward the horizon
    with pytest.raises(ValueError):
        LostLock.parse("10:0:90")


@pytest.mark.parametrize("clips", [6, 30, 100])
def test_a_set_loses_the_drone_for_the_real_share_of_its_frames(clips: int) -> None:
    """Whatever the set's size, its lost frames come to the real 1.24 % to the nearest frame, in
    runs drawn from the real lengths, each inside its clip with room for the swing and apart from
    the clip's other runs; and at 100 clips about the real share of clips (52 of 318) has one."""
    planner = _planner()
    runs = planner.plan_runs(_args(clear_runs=clips))
    share, lengths = planner.real_absent_runs()
    total = sum(r["frames"] for r in runs)
    lost = [(r["frames"], r.get("lost_lock", [])) for r in runs]
    placed = sum(n for _, clip in lost for _, n, _ in clip)
    assert abs(placed - share * total) <= 0.5
    cap = int(planner.LOST_MAX_SHARE * 300)
    allowed = {min(n, cap) for n in lengths}
    for frames, clip in lost:
        for (f1, n1, _), (f2, _, _) in itertools.pairwise(clip):
            assert f1 + n1 + 2 * LOST_RAMP_FRAMES <= f2
        for first, n, direction in clip:
            assert n in allowed and 0.0 <= direction <= 180.0
            assert first >= LOST_RAMP_FRAMES and first + n + LOST_RAMP_FRAMES <= frames
    if clips == 100:
        with_runs = sum(bool(clip) for _, clip in lost) / clips
        assert 0.08 < with_runs < 0.30, with_runs  # real: 52 / 318 = 0.16


def test_the_draw_is_seeded_and_leaves_every_other_draw_alone() -> None:
    """The lost runs have a stream of their own: the same seed gives the same runs, and turning
    them off changes nothing else about any clip."""
    planner = _planner()
    with_lost = planner.plan_runs(_args(clear_runs=30))
    assert with_lost == planner.plan_runs(_args(clear_runs=30))
    without = planner.plan_runs(_args(clear_runs=30, absent_share="0"))
    assert all("lost_lock" not in r for r in without)
    assert [{k: v for k, v in r.items() if k != "lost_lock"} for r in with_lost] == without


def test_scatter_poses_have_no_mount_to_lose_the_drone_with() -> None:
    runs = _planner().plan_runs(_args(clear_runs=30, track="scatter"))
    assert all("lost_lock" not in r for r in runs)


def test_a_share_outside_zero_to_one_is_refused() -> None:
    with pytest.raises(ValueError):
        _planner().absent_share("1.5")


def test_the_render_command_carries_the_runs() -> None:
    planner = _planner()
    run = {
        "name": "clear_000", "frames": 300, "weather": "clear", "weather_seed": 1,
        "weather_hour": 3.0, "scatter_seed": 2, "lost_lock": [[40, 14, 90.0], [200, 3, 12.5]],
    }  # fmt: skip
    args = argparse.Namespace(
        python="python", track="wander", wander_cycles=6.0, mission_s=1666.0, asset="phantom4",
        scene=SCENE, sensor="s.yaml", near_m=20.0, far_m=90.0, elevation_low_deg=1.0,
        elevation_high_deg=12.0, aim_jitter=0.6,
    )  # fmt: skip
    command = planner.render_command(args, run, pathlib.Path("out"))
    assert command[command.index("--lost-lock") + 1] == "40:14:90.0,200:3:12.5"
