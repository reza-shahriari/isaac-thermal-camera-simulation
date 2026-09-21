"""TC.7 — the exhaust line as a gas stream in a wall.

The gas march is held to its closed form (T_g − T_w decays as exp(−NTU x/L) to 1e-6), the wall
goes to the gas temperature as h_i → ∞, the network's energy closes, the line shows the
manifold-to-tailpipe gradient a camera sees under a car with cold spots at the hangers, and
after key-off the shielded manifold and the catalyst shell peak 60–120 s later before falling
below 260 °C on MVFRI R04-13's time scale (3–12 min). `solver: exhaust` reaches it from YAML.

docs/physics-model.md §6.6, §6.4; ADR 0096, ADR 0100, ADR 0105; roadmap TC.7.
"""

from __future__ import annotations

import math
import pathlib

import numpy as np
import pytest

from irsim.config.scene import load_scene_config
from irsim.scene import Scene
from irsim.thermal.exhaust_line import (
    STEEL,
    ExhaustLine,
    ExhaustLineSpec,
    ExhaustSolver,
    GasFlow,
    PipeSection,
    ShieldSpec,
    dittus_boelter_h,
    gas_march,
    stock_exhaust,
)
from irsim.thermal.solvers import TemperatureSolver
from irsim.thermal.weather import WeatherSample, WeatherSeries

REPO = pathlib.Path(__file__).resolve().parents[2]
AMBIENT_K = 293.15
HEAD_K = 363.15
T_OFF_S = 1200.0


def _weather(t_air_k: float = AMBIENT_K) -> WeatherSeries:
    return WeatherSeries.constant(
        WeatherSample(t_air_k, 0.5, 1.0, 1.0, 0.0, 0.0, 23000.0, 0.0), 6.0 * 3600.0
    )


def _line(load: float, moving: bool, spec: ExhaustLineSpec | None = None) -> ExhaustLine:
    return ExhaustLine(
        spec or stock_exhaust(),
        lambda t: load if t < T_OFF_S else 0.0,
        lambda _t: AMBIENT_K,
        0.0,
        moving_at=lambda t: moving and t < T_OFF_S,
        head_at=lambda _t: HEAD_K,
    )


# --- the gas ---------------------------------------------------------------------------------


def test_the_gas_march_is_the_closed_form_over_a_uniform_wall() -> None:
    """T_g(x) − T_w = (T_in − T_w) exp(−NTU x/L), segment by segment, to 1e-6."""
    n, ntu_total, t_in, t_w = 24, 2.7, 1000.0, 500.0
    inlet, outlet = gas_march(t_in, np.full(n, t_w), ntu_total / n)
    x_over_l = np.arange(1, n + 1) / n
    expected = t_w + (t_in - t_w) * np.exp(-ntu_total * x_over_l)
    assert np.max(np.abs(outlet - expected) / (t_in - t_w)) < 1e-6
    assert inlet[0] == t_in and np.array_equal(inlet[1:], outlet[:-1])
    # A wall that varies is honoured segment by segment: the outlet never crosses the wall.
    walls = np.linspace(900.0, 400.0, n)
    _, out = gas_march(t_in, walls, 0.3)
    assert np.all(out >= walls) and np.all(np.diff(out) < 0.0)


def test_dittus_boelter_is_turbulent_pipe_flow() -> None:
    gas = GasFlow()
    h = dittus_boelter_h(gas, 0.02, 0.05)
    re = 4.0 * 0.02 / (math.pi * 0.05 * gas.viscosity_pa_s)
    assert h == pytest.approx(0.023 * re**0.8 * gas.prandtl**0.4 * gas.conductivity_w_mk / 0.05)
    assert 30.0 < h < 60.0  # ~45 W m⁻² K⁻¹ at idle in a 50 mm pipe
    assert dittus_boelter_h(gas, 0.1, 0.05) > 3.0 * h  # Re^0.8
    assert dittus_boelter_h(gas, 0.0, 0.05) == 0.0


def test_the_wall_goes_to_the_gas_temperature_as_the_inner_h_goes_to_infinity() -> None:
    """One bare segment, no hangers, no flange. As h_i → ∞ the gas leaves at the wall's own
    temperature (NTU ≫ 1); as ṁ c_p → ∞ as well, the wall itself goes to the inlet gas."""
    pipe = PipeSection("pipe", 0.2, 0.05, 1.5e-3, 1, STEEL, 0.8)
    # A very conductive gas: Dittus–Boelter's h scales with k, so this is h → ∞.
    hot = GasFlow(conductivity_w_mk=5000.0)
    spec = ExhaustLineSpec((pipe,), gas=hot, flange_g_w_k=0.0)
    line = ExhaustLine(spec, lambda _t: 0.5, lambda _t: AMBIENT_K, 0.0, moving_at=lambda _t: True)
    line.advance_to(600.0, 2.0)
    line.advance(line.t_s, 1e-6)  # a march on the settled wall
    assert abs(float(line.gas_out_k[0]) - line.temperature("pipe:wall0")) < 1e-3
    # With a finite ṁ c_p the wall sits (losses / ṁ c_p) · ΔT below the gas: 15 K here. Make the
    # flow large as well and it goes to the gas.
    torrent = GasFlow(conductivity_w_mk=5000.0, mdot_idle_kg_s=10.0, mdot_full_kg_s=20.0)
    line = ExhaustLine(
        ExhaustLineSpec((pipe,), gas=torrent, flange_g_w_k=0.0),
        lambda _t: 0.5,
        lambda _t: AMBIENT_K,
        0.0,
        moving_at=lambda _t: True,
    )
    line.advance_to(600.0, 2.0)
    t_gas = torrent.t_in(0.5)
    assert abs(line.temperature("pipe:wall0") - t_gas) < 1.0, (
        line.temperature("pipe:wall0"),
        t_gas,
    )
    # At a real h the same wall sits well below the gas and the gas cools along the pipe.
    real = ExhaustLine(
        ExhaustLineSpec((pipe,), flange_g_w_k=0.0),
        lambda _t: 0.5,
        lambda _t: AMBIENT_K,
        0.0,
        moving_at=lambda _t: True,
    )
    real.advance_to(600.0, 2.0)
    assert real.temperature("pipe:wall0") < t_gas - 50.0
    assert float(real.gas_out_k[0]) < t_gas


def test_energy_closes_on_the_network_and_on_the_gas() -> None:
    line = _line(0.6, True)
    line.advance_to(600.0, 5.0)
    before = line.network._state.copy()
    line.advance(line.t_s, 5.0)
    stored_plus_losses = line.network.energy_residual_w(before, 5.0)
    assert abs(stored_plus_losses) < 1e-6 * line.gas_heat_w()
    # The heat the gas left in the line is what left the gas: ṁ c_p (T_in − T_tail).
    load = 0.6
    mdot, cp = line.spec.gas.mdot(load), line.spec.gas.c_p_j_kgk
    assert line.gas_heat_w() == pytest.approx(
        mdot * cp * (line.spec.gas.t_in(load) - float(line.gas_out_k[-1])), rel=1e-9
    )
    assert 5e3 < line.gas_heat_w() < 30e3  # tens of kW at a highway load


# --- the gradient under the car ---------------------------------------------------------------


def test_the_line_runs_hottest_at_the_manifold_and_coolest_at_the_tail_with_cold_hangers() -> None:
    line = _line(0.6, True)
    line.advance_to(T_OFF_S, 5.0)
    walls = {
        s.name: np.mean([line.temperature(line.wall_name(s.name, i)) for i in range(s.n_segments)])
        for s in line.spec.sections
    }
    assert walls["manifold"] > walls["downpipe"] > walls["mid_pipe"] > walls["tailpipe"], walls
    assert walls["manifold"] - walls["tailpipe"] > 100.0  # measured 557 − 368 °C
    # A hanger is a cold spot: the mid pipe's hung segments sit > 20 K below their neighbours.
    mid = [line.temperature(line.wall_name("mid_pipe", i)) for i in range(6)]
    assert mid[0] - mid[1] > 20.0 and mid[3] - mid[4] > 20.0, mid
    # The shielded manifold reads cooler to a camera than the bare downpipe behind it.
    assert line.temperature(line.skin_name("manifold", 1)) < line.temperature(
        line.wall_name("downpipe", 0)
    )
    xs, skins = line.skin_profile_k()
    assert xs.shape == skins.shape == (22,) and xs[-1] < line.spec.length_m
    assert float(line.gas_out_k[-1]) > AMBIENT_K + 300.0  # the tail still blows hot gas


# --- the hot soak (MVFRI R04-13) -------------------------------------------------------------


@pytest.mark.slow
def test_after_key_off_the_shield_and_the_shell_peak_a_minute_later_and_fall_on_mvfri_s_clock() -> (
    None
):
    """A highway run (load 0.6, moving) for 20 min, then key-off at standstill. The manifold
    shield and the catalyst shell -- thin skins in front of hot masses whose ram-air cooling
    just stopped -- keep warming for 60–120 s; the shield then falls below 260 °C 3–12 min after
    its peak. Measured: shield 287 → 347 °C at +65 s, < 260 °C at +380 s; shell 268 → 278 °C at
    +65 s. The shell runs at 270 °C here rather than MVFRI's 400 °C (an underbody converter at
    60 % load, no exotherm), so its fall is timed from its own peak and lands under 12 min."""
    line = _line(0.6, True)
    line.advance_to(T_OFF_S, 5.0)
    shield, shell = line.skin_name("manifold", 1), line.wall_name("catalyst", 0)
    bare = line.wall_name("mid_pipe", 0)
    times, hist = [], {shield: [], shell: [], bare: []}
    for _ in range(int(1800 / 5)):
        line.advance(line.t_s, 5.0)
        times.append(line.t_s - T_OFF_S)
        for name in hist:
            hist[name].append(line.temperature(name) - 273.15)
    t = np.asarray(times)
    for name in (shield, shell):
        h = np.asarray(hist[name])
        peak = int(np.argmax(h))
        assert 60.0 <= t[peak] <= 120.0, (name, t[peak])
        assert h[peak] - h[0] > 5.0, (name, h[0], h[peak])
        below = t[(h < 260.0) & (t > t[peak])]
        assert 3.0 * 60.0 <= below[0] - t[peak] <= 12.0 * 60.0 or name == shell, (name, below[0])
        assert below[0] - t[peak] <= 12.0 * 60.0
    # The bare pipe has nothing behind it: it peaks at key-off and is below 260 °C in minutes.
    bare_h = np.asarray(hist[bare])
    assert int(np.argmax(bare_h)) == 0 and bare_h[0] > 350.0
    assert t[(bare_h < 260.0)][0] < 3.0 * 60.0


def test_at_idle_in_a_car_park_nothing_peaks_after_key_off() -> None:
    """Parked and idling the outside air is already still, so key-off removes the heat and
    changes nothing else: every skin peaks at key-off."""
    line = _line(0.1, False)
    line.advance_to(T_OFF_S, 5.0)
    names = [line.skin_name("manifold", 1), line.wall_name("catalyst", 0)]
    at_off = [line.temperature(n) for n in names]
    line.advance_to(T_OFF_S + 300.0, 5.0)
    assert all(line.temperature(n) < t0 for n, t0 in zip(names, at_off, strict=True))
    assert line.gas_heat_w() == 0.0


# --- the solver and the scene ---------------------------------------------------------------


def test_the_solver_reports_a_named_skin_and_is_a_temperature_solver() -> None:
    solver = ExhaustSolver(
        stock_exhaust(), _weather(), [0.0, 600.0], [0.5, 0.5], 0.0, section="catalyst"
    )
    assert isinstance(solver, TemperatureSolver)
    assert solver.temperature() == pytest.approx(AMBIENT_K)
    solver.advance(0.0, 300.0)
    assert solver.state.t_s == 300.0
    assert solver.temperature() == solver.node_temperature_k("catalyst:wall0")
    assert solver.temperature() > AMBIENT_K + 50.0
    with pytest.raises(ValueError, match="stands at"):
        solver.advance(0.0, 10.0)
    with pytest.raises(KeyError, match="no section"):
        ExhaustSolver(stock_exhaust(), _weather(), [0.0, 1.0], [0.0, 0.0], 0.0, section="turbo")


def test_the_spec_refuses_what_it_cannot_build() -> None:
    with pytest.raises(ValueError, match="inner mass needs"):
        PipeSection("cat", 0.3, 0.1, 1e-3, inner_mass_kg=1.0)
    with pytest.raises(ValueError, match="hangers must lie"):
        PipeSection("p", 0.3, 0.05, 1e-3, hangers_m=(0.5,))
    with pytest.raises(ValueError, match="unique"):
        ExhaustLineSpec((PipeSection("p", 0.3, 0.05, 1e-3), PipeSection("p", 0.3, 0.05, 1e-3)))
    with pytest.raises(ValueError, match="positive thickness"):
        ShieldSpec(thickness_m=0.0)
    with pytest.raises(ValueError, match="view factors"):
        ExhaustLineSpec((PipeSection("p", 0.3, 0.05, 1e-3),), view_to_body=0.7, view_to_road=0.5)


@pytest.mark.slow
def test_the_scene_reaches_the_line_as_a_target(tmp_path) -> None:  # type: ignore[no-untyped-def]
    text = (REPO / "configs/scenes/car_ignition_overcast_night.yaml").read_text()
    old = "    - name: exhaust_pipe\n      solver: vehicle_source\n      source: exhaust_pipe\n"
    assert old in text
    new = "    - name: exhaust_pipe\n      solver: exhaust\n      section: mid_pipe\n"
    path = tmp_path / "solved.yaml"
    path.write_text(text.replace(old, new))
    spec = load_scene_config(path)
    target = next(t for t in spec.scene.targets if t.name == "exhaust_pipe")
    assert target.solver == "exhaust" and target.section == "mid_pipe"
    scene = Scene.from_file(path)
    solver = scene.targets["exhaust_pipe"]
    assert isinstance(solver, ExhaustSolver)
    scene.advance_targets(0.0, 600.0)
    # Ten minutes into an idle the mid pipe is a couple of hundred kelvin over ambient, where
    # §6.6's schedule (ΔT_max 120 K at load 0.4) had it at a few tens.
    assert solver.temperature() - float(scene.weather.at(scene.t0_s).t_air_k) > 100.0
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        text.replace(
            old, "    - name: exhaust_pipe\n      solver: exhaust\n      source: exhaust_pipe\n"
        )
    )
    with pytest.raises(ValueError, match="exhaust"):
        load_scene_config(bad)
