"""EV.24: a drone set's clips cover the day, and a night clip is filmed in night air.

The first clip set drew its hours uniformly and five of six landed at night; and the weather
those clips were rendered under gave 22.0 degC at 09:45 and at 22:23, because a weather-fx state
re-timed to the clip's hour kept the air it was drawn with. ``generate_aerial_dataset.plan_runs``
now stratifies the hours in local solar time, and ``irsim_isaac.weather_fx_stage.weather_state``
re-times with weather-fx's ``move_clock``, which carries the air along the day's curve.
"""

from __future__ import annotations

import argparse
import importlib.util
import pathlib

import numpy as np
import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
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
        seed=20261004, clear_runs=6, cloud_runs=4, frames_per_run=300, cloud_frames_per_run=50,
        near_m=20.0, far_m=90.0, focus="drawn", scene=SCENE,
    )  # fmt: skip
    base.update(overrides)
    return argparse.Namespace(**base)


@pytest.mark.parametrize("clear_runs", [6, 8, 24])
def test_every_slot_of_the_day_holds_exactly_one_clip(clear_runs: int) -> None:
    """n clear clips put one clip in each 24/n-hour slot of local solar time: six clips are
    four-hour slots, so two of them are at night and two by day whatever the seed."""
    runs = [
        r for r in _planner().plan_runs(_args(clear_runs=clear_runs)) if r["name"][:5] == "clear"
    ]
    local = np.array([r["local_solar_hour"] for r in runs])
    slots = np.floor(local / (24.0 / clear_runs)).astype(int)
    assert sorted(slots.tolist()) == list(range(clear_runs))


def test_the_utc_hour_is_the_local_one_at_the_scenes_longitude() -> None:
    longitude = _planner().scene_longitude_deg(SCENE)
    assert longitude == pytest.approx(11.6)
    for run in _planner().plan_runs(_args()):
        utc = (run["local_solar_hour"] - longitude / 15.0) % 24.0
        assert run["weather_hour"] == pytest.approx(utc, abs=0.011)  # both rounded to 0.01 h


def test_the_weather_and_seeds_a_set_drew_before_are_unchanged() -> None:
    """The stratification takes its order from a stream of its own and its position in the slot
    from the old uniform draw, so the main stream -- weather, seeds -- is drawn as it was."""
    args = _args()
    rng = np.random.default_rng(args.seed)
    expected = []
    for kind, count in (("clear", args.clear_runs), ("cloud", args.cloud_runs)):
        for _ in range(count):
            regime = "clear" if kind == "clear" else str(rng.choice(_planner().CLOUD_REGIMES))
            weather_seed = int(rng.integers(0, 2**31 - 1))
            rng.uniform(0.0, 24.0)
            expected.append((regime, weather_seed, int(rng.integers(0, 2**31 - 1))))
    runs = _planner().plan_runs(args)
    assert [(r["weather"], r["weather_seed"], r["scatter_seed"]) for r in runs] == expected


def test_a_clear_night_clip_is_filmed_in_night_air() -> None:
    """The same drawn clear day at 02:00 and at 13:00 UTC over the Phantom 4 scene (11.6 deg E):
    the afternoon is the clear-day swing warmer, and the dew point is the same water."""
    from irsim_isaac.weather_fx_stage import weather_state

    def at(hour: float):  # type: ignore[no-untyped-def]
        return weather_state(
            seed=5, regime="clear", latitude_deg=48.1, longitude_deg=11.6,
            date_utc="2024-06-21", hour_utc=hour,
        )  # fmt: skip

    night, afternoon = at(2.0), at(13.0)
    assert afternoon.clouds.temperature_c - night.clouds.temperature_c > 9.0
    assert night.clouds.dewpoint_c == afternoon.clouds.dewpoint_c
