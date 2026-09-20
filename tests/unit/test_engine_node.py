"""TC.5 — the engine as a solved node, not a schedule.

§6.6 scripts the bay with τ_rise = 750 s and τ_cool = 1800 s. The measurements the survey
collected say otherwise about what happens *after* key-off: the block cools with a time constant
of hours (a coolant diagnostic reaches ambient in ~7 h), the bay air spikes because the fan-driven
airflow stops while the block does not, and the skins around the block keep warming for a minute
or two. The tests below are those measurements as bands, with the §6.6 schedule as the negative
control that fails the first of them.

docs/physics-model.md §6.6, §6.4; spec issue S43; ADR 0096, ADR 0100; roadmap TC.5.
"""

from __future__ import annotations

import pathlib

import pytest

from irsim.config.scene import TargetSpec, load_scene_config
from irsim.scene import Scene
from irsim.thermal.engine import EngineSolver, EngineSpec, engine_network
from irsim.thermal.vehicle import VEHICLE_HEAT_SOURCES, newton_cool
from irsim.thermal.weather import WeatherSample, WeatherSeries

REPO = pathlib.Path(__file__).resolve().parents[2]
OVERCAST = REPO / "configs/scenes/car_ignition_overcast_night.yaml"

AMBIENT_K = 300.15  # 27 °C
HOT_BLOCK_K = 366.15  # 93 °C: a thermostat's coolant at key-off


def _constant_weather(t_air_k: float = AMBIENT_K) -> WeatherSeries:
    return WeatherSeries.constant(
        WeatherSample(t_air_k, 0.5, 1.0, 1.0, 0.0, 0.0, 23000.0, 0.0), 12.0 * 3600.0
    )


# --- key-off: hours, not minutes ----------------------------------------------------------------


def test_the_block_cools_over_hours_and_a_schedule_does_not() -> None:
    """From 93 °C at 27 °C ambient: > 20 K above ambient after 1 h, within 1 K after 7 h.

    The §6.6 schedule (τ_cool = 1800 s) has the block 9 K above ambient after an hour, and fails.
    """
    net = engine_network(
        EngineSpec(), lambda t: 0.0, lambda t: AMBIENT_K, 0.0, initial_block_k=HOT_BLOCK_K
    )
    net.advance_to(3600.0, 60.0)
    after_1h = net.temperature("block") - AMBIENT_K
    net.advance_to(7.0 * 3600.0, 60.0)
    after_7h = net.temperature("block") - AMBIENT_K
    assert after_1h > 20.0, after_1h
    assert 0.0 < after_7h < 1.0, after_7h

    schedule = float(
        newton_cool(3600.0, HOT_BLOCK_K - AMBIENT_K, VEHICLE_HEAT_SOURCES["engine_bay"].tau_cool_s)
    )
    assert schedule < 20.0, "the negative control: a 1800 s schedule is nearly cold after an hour"


def test_the_steady_rise_at_full_load_is_inside_6_6_s_band() -> None:
    net = engine_network(EngineSpec(), lambda t: 1.0, lambda t: AMBIENT_K, 0.0)
    net.advance_to(6.0 * 3600.0, 60.0)
    rise = net.temperature("block") - AMBIENT_K
    assert 40.0 <= rise <= 90.0, rise
    idle = engine_network(EngineSpec(), lambda t: 0.45, lambda t: AMBIENT_K, 0.0)
    idle.advance_to(6.0 * 3600.0, 60.0)
    assert 0.0 < idle.temperature("block") - AMBIENT_K < rise
    # Heat in equals heat out at steady state, to the solver's own arithmetic.
    before = net.all_temperatures_k()[: net.n_free]
    net.advance(net.t_s, 60.0)
    assert abs(net.energy_residual_w(before, 60.0)) < 1e-6 * EngineSpec().heat_to_bay_w


def test_the_bay_air_overshoots_after_key_off() -> None:
    """The fan stops and the block does not: bay air jumps by 20–50 K within minutes."""
    key_off = 4.0 * 3600.0
    net = engine_network(
        EngineSpec(), lambda t: 1.0 if t <= key_off else 0.0, lambda t: AMBIENT_K, 0.0
    )
    net.advance_to(key_off - 60.0, 60.0)
    running = net.temperature("bay_air") - AMBIENT_K
    peak, t_peak = running, 0.0
    while net.t_s < key_off + 900.0:
        net.advance(net.t_s, 5.0)
        rise = net.temperature("bay_air") - AMBIENT_K
        if rise > peak:
            peak, t_peak = rise, net.t_s - key_off
    assert 20.0 <= peak - running <= 50.0, (peak, running)
    assert 30.0 <= t_peak <= 300.0, t_peak


def test_parts_lag_the_block_in_conductance_order() -> None:
    net = engine_network(EngineSpec(), lambda t: 1.0, lambda t: AMBIENT_K, 0.0)
    net.advance_to(1200.0, 30.0)
    t = net.temperatures_k
    assert t["block"] > t["mounts"] > t["subframe"] > AMBIENT_K
    assert t["block"] > t["bay_air"] > AMBIENT_K


# --- the scene ----------------------------------------------------------------------------------


def test_the_engine_solver_drops_into_the_scene_s_target_slot(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """``solver: engine`` replaces ADR 0089's adapter; the bonnet's forcing reads the bay air."""
    text = OVERCAST.read_text()
    old = "- name: engine_bay\n      solver: vehicle_source\n      source: engine_bay\n"
    assert text.count(old) == 1
    path = tmp_path / "engine.yaml"
    path.write_text(text.replace(old, "- name: engine_bay\n      solver: engine\n"))
    config = load_scene_config(path)
    assert [t.solver for t in config.scene.targets if t.name == "engine_bay"] == ["engine"]
    scene = Scene.from_file(path)
    engine = scene.targets["engine_bay"]
    assert isinstance(engine, EngineSolver) and engine.weather is scene.weather
    t_air = scene.weather.at(scene.t0_s).t_air_k
    assert engine.temperature() == pytest.approx(t_air) and engine.state.t_s == scene.t0_s
    # The key turns at 30 s: the block warms, the bay air with it, and the reported cavity
    # temperature is the bay air.
    out = scene.advance_targets(0.0, 600.0)
    assert engine.node_temperature_k("block") > t_air + 5.0
    assert out["engine_bay"] == engine.temperature() == engine.node_temperature_k("bay_air")
    assert engine.temperature() > t_air + 0.5  # the fan vents hard while the engine runs


@pytest.mark.slow
def test_the_bonnet_over_the_block_keeps_warming_after_key_off(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Key on at 30 s, off at 1200 s. A schedule with τ_cool starts the skin cooling at once; the
    solved engine keeps feeding the bay from the block's stored heat while the fan has stopped,
    and the bonnet over the block goes on warming.

    Measured: the bay air peaks 150 s after key-off; the bonnet over the block is still rising at
    +600 s and peaks ~23 min after key-off, because in this step the bonnet is not yet a loss
    path for the bay (TC.6 couples it) and the block is one lumped node without a fast skin
    (TC.7's manifold). The row's 60–120 s figure is the survey's manifold-skin number; the
    lower bound is what this step holds itself to.
    """
    import numpy as np

    from irsim.thermal.spatial_sources import patch_view_factors
    from irsim_isaac.car_demo import build_car_demo

    text = OVERCAST.read_text()
    old = "- name: engine_bay\n      solver: vehicle_source\n      source: engine_bay\n"
    schedule_old = "load_s: [0.0, 29.9, 30.0, 1800.0]\n      load:   [0.0,  0.0,  0.45,   0.45]"
    assert text.count(old) == 1 and text.count(schedule_old) >= 1
    path = tmp_path / "engine_off.yaml"
    path.write_text(
        text.replace(old, "- name: engine_bay\n      solver: engine\n").replace(
            schedule_old,
            "load_s: [0.0, 29.9, 30.0, 1200.0, 1200.1, 3000.0]\n"
            "      load:   [0.0,  0.0,  0.45,   0.45,   0.0,    0.0]",
            1,
        )
    )
    scene = Scene.from_file(path)
    demo = build_car_demo(scene, author=False, spin_up=False)
    bonnet = demo.bonnet_field
    over = patch_view_factors(bonnet.patch, demo.geometry.engine_bay())
    over = over >= 0.9 * over.max()
    engine = scene.targets["engine_bay"]

    def bonnet_over_block() -> float:
        return float(np.asarray(bonnet.temperature_at(bonnet.latest_t_s), float)[over].mean())

    for k in range(120):  # to key-off
        scene.advance_targets(10.0 * k, 10.0)
    bonnet.advance_to(scene.t0_s + 1200.0)
    at_key_off = bonnet_over_block()
    bay_at_key_off = engine.temperature()
    bay_peak, bay_peak_t = bay_at_key_off, 0.0
    for k in range(120, 180):  # ten minutes after
        scene.advance_targets(10.0 * k, 10.0)
        if engine.temperature() > bay_peak:
            bay_peak, bay_peak_t = engine.temperature(), 10.0 * (k + 1) - 1200.0
    bonnet.advance_to(scene.t0_s + 1800.0)
    assert 60.0 <= bay_peak_t <= 300.0 and bay_peak - bay_at_key_off > 2.0, (bay_peak_t, bay_peak)
    assert bonnet_over_block() > at_key_off + 0.5, "the bonnet over the block kept warming"


def test_the_engine_kind_is_validated_like_a_load_profile() -> None:
    TargetSpec(name="e", solver="engine", load_s=[0.0, 30.0], load=[0.0, 1.0])
    with pytest.raises(ValueError, match="no `source`"):
        TargetSpec(name="e", solver="engine", source="engine_bay", load_s=[0.0], load=[1.0])
    with pytest.raises(ValueError, match="needs load_s"):
        TargetSpec(name="e", solver="engine")
    with pytest.raises(ValueError, match="strictly increasing"):
        TargetSpec(name="e", solver="engine", load_s=[10.0, 5.0], load=[0.0, 1.0])
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        TargetSpec(name="e", solver="engine", load_s=[0.0, 5.0], load=[0.0, 1.5])
    with pytest.raises(ValueError, match="must be positive"):
        EngineSpec(block_mass_kg=0.0)
    with pytest.raises(ValueError, match="BAY"):
        EngineSpec(bay_fraction=0.5)
    with pytest.raises(ValueError, match="stands at"):
        EngineSolver(EngineSpec(), _constant_weather(), [0.0, 1.0], [0.0, 1.0], 0.0).advance(
            5.0, 1.0
        )
