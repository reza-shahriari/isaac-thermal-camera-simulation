"""EV.21: a clip's lens focus, set from outside the sensor file, and the planner that draws it.

Real anti-UAV targets are soft -- a tracking camera's focus is set once and the drone is rarely at
it -- and irsim's renders were sharp because no camera named a defocus model.
:func:`irsim.config.loader.with_focus` focuses a camera and makes the focus matter;
``scripts/generate_aerial_dataset.py`` draws one focus per clip.
"""

from __future__ import annotations

import argparse
import importlib.util
import pathlib

import numpy as np
import pytest

from irsim.config.loader import config_hash, load_sensor_config, with_focus

REPO = pathlib.Path(__file__).resolve().parents[2]
TELEPHOTO = REPO / "configs" / "sensors" / "example_lwir_640_telephoto.yaml"


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
        seed=20261004, clear_runs=40, cloud_runs=4, frames_per_run=300,
        cloud_frames_per_run=50, near_m=20.0, far_m=90.0, focus="drawn",
    )  # fmt: skip
    base.update(overrides)
    return argparse.Namespace(**base)


def test_focusing_a_camera_names_a_defocus_model_and_a_new_hash() -> None:
    """A fixed focus with no defocus model would be sharp at every range; `with_focus` sets the
    Hopkins model, layered, and the config hash moves -- two focus settings are two cameras."""
    sensor = load_sensor_config(TELEPHOTO)
    assert sensor.sensor.optics.mtf.defocus_model == "none"
    near = with_focus(sensor, 35.0)
    far = with_focus(sensor, None)
    assert near.sensor.optics.focus.mode == "fixed"
    assert near.sensor.optics.focus.distance_m == 35.0
    assert far.sensor.optics.focus.mode == "infinity"
    for focused in (near, far):
        assert focused.sensor.optics.mtf.defocus_model == "hopkins"
        assert focused.sensor.optics.mtf.defocus_apply == "layered"
        assert focused.sensor.defocus_enabled
    assert len({config_hash(sensor), config_hash(near), config_hash(far)}) == 3


def test_a_model_the_sensor_already_names_is_kept() -> None:
    sensor = with_focus(load_sensor_config(TELEPHOTO), None, model="gaussian", apply="global")
    refocused = with_focus(sensor, 50.0)
    assert refocused.sensor.optics.mtf.defocus_model == "gaussian"
    assert refocused.sensor.optics.mtf.defocus_apply == "global"


def test_the_planner_draws_half_the_clips_at_infinity_and_the_rest_in_range() -> None:
    """44 clips: about half at infinity, the rest between the near range and ten times the far."""
    runs = _planner().plan_runs(_args())
    focus = [r["focus_m"] for r in runs]
    at_infinity = sum(f == "inf" for f in focus)
    assert 12 <= at_infinity <= 32
    distances = np.array([float(f) for f in focus if f != "inf"])
    assert distances.min() >= 20.0 and distances.max() <= 900.0
    assert _planner().plan_runs(_args()) == runs  # reproducible from the seed


def test_the_focus_draw_leaves_every_other_draw_as_it_was() -> None:
    """Its own random stream: a set planned before EV.21 keeps its weather, hours and seeds."""
    drawn = _planner().plan_runs(_args())
    fixed = _planner().plan_runs(_args(focus="inf"))
    assert all(f["focus_m"] == "inf" for f in fixed)
    strip = [{k: v for k, v in r.items() if k != "focus_m"} for r in drawn]
    assert strip == [{k: v for k, v in r.items() if k != "focus_m"} for r in fixed]


@pytest.mark.parametrize("focus", ["inf", "35.0", "sensor"])
def test_the_render_command_carries_the_clips_focus(focus: str) -> None:
    planner = _planner()
    run = {**planner.plan_runs(_args(clear_runs=1, cloud_runs=0))[0], "focus_m": focus}
    args = _args(
        python="python", track="wander", wander_cycles=6.0, mission_s=1666.0, asset="phantom4",
        scene="s.yaml", sensor="c.yaml", elevation_low_deg=6.0, elevation_high_deg=20.0,
        aim_jitter=0.6,
    )  # fmt: skip
    command = planner.render_command(args, run, pathlib.Path("out"))
    if focus == "sensor":
        assert "--focus-m" not in command
    else:
        assert command[command.index("--focus-m") + 1] == focus
