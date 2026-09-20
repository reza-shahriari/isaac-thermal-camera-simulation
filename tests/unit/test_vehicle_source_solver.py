"""MP.4a / ADR 0089 — §6.6's ground-vehicle heat sources, reachable from a scene.

The table and its integrator have existed since M6.14 and nothing outside their own unit tests
could reach them. The scene schema's `heat_source` is ADR 0072's *aerial* node, whose law is a
steady-state relation with no time constant at all -- so an engine bay authored through it shows
its full 65 K in the first frame after ignition, which is the one thing an engine bay does not do.

`test_the_rise_is_the_closed_form_and_not_a_step` is the test that separates the two. Everything
else guards the seams: the load profile lives on scene-relative time while the solver lives on the
weather's absolute axis, and getting that offset wrong reads a plausible temperature from the
wrong hour.

docs/physics-model.md §6.6; ADR 0038, ADR 0089
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest
from pydantic import ValidationError

from irsim.config.scene import SCENE_SCHEMA_VERSION, TargetSpec
from irsim.thermal.vehicle import (
    VEHICLE_HEAT_SOURCES,
    VehicleSourceSolver,
    first_order_rise,
    newton_cool,
)
from irsim.thermal.weather_io import synthetic_clear_day

ENGINE = VEHICLE_HEAT_SOURCES["engine_bay"]
T_AIR = 288.0


def _running(delta_t0_k: float = 0.0, t0_s: float = 0.0) -> VehicleSourceSolver:
    return VehicleSourceSolver(
        ENGINE, T_AIR, [t0_s, t0_s + 1e6], [1.0, 1.0], t0_s=t0_s, delta_t0_k=delta_t0_k
    )


def _march(solver: VehicleSourceSolver, until_s: float, dt_s: float, t0_s: float = 0.0) -> float:
    t = t0_s
    while t < t0_s + until_s - 1e-9:
        solver.advance(t, dt_s)
        t += dt_s
    return solver.temperature()


# ---------------------------------------------------------------------------------------------
# the law
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("elapsed_s", [60.0, 375.0, 750.0, 1500.0, 3000.0])
def test_the_rise_is_the_closed_form_and_not_a_step(elapsed_s: float) -> None:
    """Held to §6.6's own curve to 1 mK. A steady-state law fails this by up to 65 K."""
    solver = _running()
    _march(solver, elapsed_s, 5.0)
    expected = float(first_order_rise(elapsed_s, ENGINE.delta_t_max_k, ENGINE.tau_rise_s))
    assert solver.delta_t_k == pytest.approx(expected, abs=1e-3)
    assert solver.temperature() == pytest.approx(T_AIR + expected, abs=1e-3)


def test_one_time_constant_is_63_percent_and_not_the_whole_rise() -> None:
    solver = _running()
    _march(solver, ENGINE.tau_rise_s, 5.0)
    assert solver.delta_t_k == pytest.approx(ENGINE.delta_t_max_k * (1.0 - np.exp(-1.0)), abs=1e-3)
    assert solver.delta_t_k < 0.65 * ENGINE.delta_t_max_k


def test_the_answer_does_not_depend_on_the_step_size() -> None:
    """The exact exponential, which is why a render at 60 Hz and a time-lapse agree."""
    fine = _running()
    coarse = _running()
    _march(fine, 1200.0, 1.0)
    _march(coarse, 1200.0, 200.0)
    assert fine.delta_t_k == pytest.approx(coarse.delta_t_k, abs=1e-9)


def test_switching_off_cools_on_the_other_time_constant() -> None:
    """tau_cool is 1800 s against tau_rise's 750 s: an engine forgets far slower than it warms."""
    solver = VehicleSourceSolver(ENGINE, T_AIR, [0.0, 600.0, 600.1, 1e6], [1.0, 1.0, 0.0, 0.0])
    _march(solver, 600.0, 5.0)
    hot = solver.delta_t_k
    t = 600.0
    for _ in range(120):
        solver.advance(t, 5.0)
        t += 5.0
    expected = float(newton_cool(600.0, hot, ENGINE.tau_cool_s))
    assert solver.delta_t_k == pytest.approx(expected, rel=2e-3)
    assert solver.delta_t_k > 0.6 * hot, "1800 s of cooling is slow; 600 s barely dents it"


def test_a_partial_load_reaches_a_partial_rise() -> None:
    idle = VehicleSourceSolver(ENGINE, T_AIR, [0.0, 1e6], [0.35, 0.35])
    _march(idle, 20_000.0, 50.0)
    assert idle.delta_t_k == pytest.approx(0.35 * ENGINE.delta_t_max_k, abs=1e-2)


def test_a_node_that_has_just_parked_starts_hot_and_forgets() -> None:
    """delta_t0_k is what makes a daylight scene of a recently parked car possible at all."""
    parked = VehicleSourceSolver(ENGINE, T_AIR, [0.0, 1e6], [0.0, 0.0], delta_t0_k=40.0)
    assert parked.temperature() == pytest.approx(T_AIR + 40.0)
    _march(parked, ENGINE.tau_cool_s, 10.0)
    assert parked.delta_t_k == pytest.approx(40.0 * np.exp(-1.0), rel=2e-3)


def test_ambient_enters_in_exactly_one_place() -> None:
    """A source cannot drift off the shared weather (CLAUDE.md #6)."""
    warm = VehicleSourceSolver(ENGINE, 310.0, [0.0, 1e6], [1.0, 1.0])
    cold = VehicleSourceSolver(ENGINE, 260.0, [0.0, 1e6], [1.0, 1.0])
    _march(warm, 900.0, 10.0)
    _march(cold, 900.0, 10.0)
    assert warm.delta_t_k == pytest.approx(cold.delta_t_k, abs=1e-12)
    assert warm.temperature() - cold.temperature() == pytest.approx(50.0, abs=1e-12)


def test_a_weather_series_is_exposed_for_the_one_weather_guard() -> None:
    weather = synthetic_clear_day(datetime(2024, 6, 21, tzinfo=timezone.utc), hours=24.0)
    solver = VehicleSourceSolver(ENGINE, weather, [0.0, 1e6], [1.0, 1.0])
    assert solver.weather is weather
    assert VehicleSourceSolver(ENGINE, T_AIR, [0.0, 1e6], [1.0, 1.0]).weather is None


def test_the_load_profile_is_held_flat_not_extrapolated() -> None:
    solver = VehicleSourceSolver(ENGINE, T_AIR, [100.0, 200.0], [0.0, 1.0])
    assert solver.load_at(-500.0) == pytest.approx(0.0)
    assert solver.load_at(5000.0) == pytest.approx(1.0)
    assert solver.load_at(150.0) == pytest.approx(0.5)


def test_bad_profiles_raise() -> None:
    with pytest.raises(ValueError, match="load must lie"):
        VehicleSourceSolver(ENGINE, T_AIR, [0.0, 1.0], [0.0, 1.4])
    with pytest.raises(ValueError, match="strictly increasing"):
        VehicleSourceSolver(ENGINE, T_AIR, [1.0, 0.0], [0.0, 1.0])
    with pytest.raises(ValueError, match="equal length"):
        VehicleSourceSolver(ENGINE, T_AIR, [0.0, 1.0], [0.0])
    with pytest.raises(ValueError, match="non-negative"):
        VehicleSourceSolver(ENGINE, T_AIR, [0.0, 1.0], [0.0, 1.0], delta_t0_k=-1.0)


# ---------------------------------------------------------------------------------------------
# the config seam
# ---------------------------------------------------------------------------------------------


def test_the_scene_schema_accepts_a_vehicle_source() -> None:
    spec = TargetSpec(
        name="engine",
        solver="vehicle_source",
        source="engine_bay",
        load_s=[0.0, 30.0],
        load=[0.0, 1.0],
    )
    assert spec.source == "engine_bay"
    assert SCENE_SCHEMA_VERSION >= 6


def test_an_aerial_source_name_is_refused_for_a_vehicle_node() -> None:
    """`motor` is an AERIAL_HEAT_SOURCES row; accepting it here would silently use a drone."""
    with pytest.raises(ValidationError, match="source must be one of"):
        TargetSpec(name="engine", solver="vehicle_source", source="motor", load_s=[0.0], load=[1.0])


def test_the_two_node_kinds_do_not_take_each_others_profiles() -> None:
    with pytest.raises(ValidationError, match="load_s/load, not a throttle"):
        TargetSpec(
            name="engine",
            solver="vehicle_source",
            source="engine_bay",
            throttle_s=[0.0],
            throttle=[1.0],
        )
    with pytest.raises(ValidationError, match="only vehicle_source and engine take a load profile"):
        TargetSpec(
            name="motor",
            solver="heat_source",
            source="motor",
            throttle_s=[0.0],
            throttle=[1.0],
            load_s=[0.0],
            load=[1.0],
        )
    with pytest.raises(ValidationError, match="only vehicle_source and engine take a load profile"):
        TargetSpec(
            name="hull",
            solver="prescribed",
            schedule_s=[0.0],
            schedule_k=[300.0],
            load_s=[0.0],
            load=[1.0],
        )


def test_authoring_a_temperature_directly_is_refused() -> None:
    """The point of the solver is that the curve comes from the model, not from the YAML."""
    with pytest.raises(ValidationError, match="derives its curve"):
        TargetSpec(
            name="engine",
            solver="vehicle_source",
            source="engine_bay",
            load_s=[0.0],
            load=[1.0],
            schedule_s=[0.0],
            schedule_k=[400.0],
        )
