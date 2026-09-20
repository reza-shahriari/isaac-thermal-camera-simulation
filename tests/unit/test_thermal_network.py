"""TC.2 — a thermal network: nodes, links, boundaries, and the identity that catches a sign error.

Until this module nothing could connect two solved parts (spec issue S42). The tests here are
the closed forms the roadmap row names -- ΔT = Q/G on a fixed sink and Q/(hA) on a fluid, the two
link forms bit-identical, a link node answering a step exponentially -- and the one that matters
most: every tick, ``Σ C dT/dt`` equals the sources minus what flows into the fixed nodes, with
the flows evaluated exactly as the solver applied them. A conductor with the wrong sign fails it
on the first tick.

docs/physics-model.md §6.4, §6.6; spec issue S42; ADR 0094, ADR 0096; roadmap TC.2.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from irsim.radiometry.constants import SIGMA_SB
from irsim.thermal.network import (
    FixedNode,
    ImposedHeat,
    Link,
    LinkNode,
    Node,
    RadiationLink,
    ThermalNetwork,
)

AMBIENT_K = 300.0


def _settle(net: ThermalNetwork, hours: float, dt_s: float = 60.0) -> None:
    net.advance_to(net.t0_s + hours * 3600.0, dt_s)


# --- the closed forms ---------------------------------------------------------------------------


def test_a_heated_node_on_a_fixed_sink_settles_at_q_over_g() -> None:
    """ΔT = Q/G to 1e-6 relative: 500 W through 25 W/K (a 25 cm² bolted joint at 1e4)."""
    net = ThermalNetwork(
        nodes=[Node("bracket", capacity_j_k=200.0)],
        fixed=[FixedNode("block", AMBIENT_K)],
        links=[Link.from_contact("bracket", "block", h_c_w_m2_k=1e4, area_m2=25e-4)],
        sources=[ImposedHeat("bracket", 500.0)],
        initial_k=AMBIENT_K,
    )
    _settle(net, hours=2.0)
    delta = net.temperature("bracket") - AMBIENT_K
    assert delta == pytest.approx(500.0 / 25.0, rel=1e-6)


def test_convection_to_a_fluid_node_settles_at_q_over_ha() -> None:
    """The fluid is a node of its own here, itself convecting to fixed air: two resistors."""
    h, a, h_air, a_air = 12.0, 0.8, 8.0, 3.0
    net = ThermalNetwork(
        nodes=[Node("shield", 400.0), Node("bay_air", 360.0)],
        fixed=[FixedNode("air", AMBIENT_K)],
        links=[
            Link.convection("shield", "bay_air", h, a),
            Link.convection("bay_air", "air", h_air, a_air),
        ],
        sources=[ImposedHeat("shield", 120.0)],
        initial_k=AMBIENT_K,
    )
    _settle(net, hours=3.0)
    t_bay = net.temperature("bay_air")
    assert t_bay - AMBIENT_K == pytest.approx(120.0 / (h_air * a_air), rel=1e-6)
    assert net.temperature("shield") - t_bay == pytest.approx(120.0 / (h * a), rel=1e-6)


def test_the_two_link_forms_are_bit_identical() -> None:
    """``h_c·A`` and ``G`` are the same number to the solver, not two code paths."""
    h_c, area = 7.3e3, 1.7e-3
    as_g = ThermalNetwork(
        nodes=[Node("a", 50.0)],
        fixed=[FixedNode("b", AMBIENT_K)],
        links=[Link("a", "b", h_c * area)],
        sources=[ImposedHeat("a", 40.0)],
        initial_k=310.0,
    )
    as_contact = ThermalNetwork(
        nodes=[Node("a", 50.0)],
        fixed=[FixedNode("b", AMBIENT_K)],
        links=[Link.from_contact("a", "b", h_c, area)],
        sources=[ImposedHeat("a", 40.0)],
        initial_k=310.0,
    )
    as_convection = ThermalNetwork(
        nodes=[Node("a", 50.0)],
        fixed=[FixedNode("b", AMBIENT_K)],
        links=[Link.convection("a", "b", h_c, area)],
        sources=[ImposedHeat("a", 40.0)],
        initial_k=310.0,
    )
    for _ in range(50):
        for net in (as_g, as_contact, as_convection):
            net.advance(net.t_s, 7.0)
    assert as_g.temperature("a") == as_contact.temperature("a") == as_convection.temperature("a")


def test_a_link_node_answers_a_step_exponentially() -> None:
    """A rubber mount between a hot block and a cold subframe: τ = C/(4G), within 2 %.

    Fitted from the solver's own trajectory over one τ, against the closed form; and the series
    conductance through the mount is the one authored (steady flow = G ΔT).
    """
    g, c = 5.0, 1800.0  # W/K through the mount, J/K of rubber
    tau = c / (4.0 * g)
    net = ThermalNetwork(
        nodes=[],
        fixed=[FixedNode("block", 380.0), FixedNode("subframe", AMBIENT_K)],
        link_nodes=[LinkNode("mount", "block", "subframe", conductance_w_k=g, capacity_j_k=c)],
        initial_k=AMBIENT_K,
    )
    t_final = 0.5 * (380.0 + AMBIENT_K)
    times, temps = [], []
    dt = tau / 200.0
    while net.t_s < tau:
        net.advance(net.t_s, dt)
        times.append(net.t_s)
        temps.append(net.temperature("mount"))
    t = np.asarray(times)
    remaining = (t_final - np.asarray(temps)) / (t_final - AMBIENT_K)
    fitted_tau = -(float(np.polyfit(t, np.log(remaining), 1)[0]) ** -1)
    assert abs(fitted_tau - tau) / tau < 0.02, (fitted_tau, tau)

    _settle(net, hours=20 * tau / 3600.0, dt_s=dt)
    power = net.link_power_w()
    into_subframe = power[net.names.index("subframe")]
    assert into_subframe == pytest.approx(g * (380.0 - AMBIENT_K), rel=1e-6)


def test_a_radiation_link_reaches_the_t4_steady_state() -> None:
    """Q = εAFσ(T⁴ − T_s⁴) at steady state, exactly, though the step linearises per tick."""
    q, eps, area = 300.0, 0.9, 0.5
    net = ThermalNetwork(
        nodes=[Node("manifold", 900.0)],
        fixed=[FixedNode("enclosure", AMBIENT_K)],
        radiation=[RadiationLink("manifold", "enclosure", emissivity=eps, area_m2=area)],
        sources=[ImposedHeat("manifold", q)],
        initial_k=AMBIENT_K,
    )
    _settle(net, hours=6.0)
    expected = (AMBIENT_K**4 + q / (eps * area * SIGMA_SB)) ** 0.25
    assert net.temperature("manifold") == pytest.approx(expected, rel=1e-6)


# --- what conservation catches ------------------------------------------------------------------


def _bay() -> ThermalNetwork:
    """A small engine bay: block, mount, bracket, wing, bay air; free air and coolant fixed."""
    ambient = FixedNode("air", lambda t: AMBIENT_K + 3.0 * math.sin(t / 900.0))
    return ThermalNetwork(
        nodes=[
            Node.from_mass("block", 120.0, 500.0),
            Node.from_mass("bracket", 0.6, 470.0),
            Node.from_mass("wing", 4.0, 900.0),
            Node("bay_air", 360.0),
        ],
        fixed=[ambient, FixedNode("coolant", 363.0)],
        links=[
            # A dry, unpasted joint (the survey's 1 kW m⁻² K⁻¹ default) over 25 cm².
            Link.from_contact("block", "bracket", 1.0e3, 25e-4),
            # A small hanger to the wing: about a fastener's worth (Voller: ~1 W/K per small bolt).
            Link("bracket", "wing", 0.5),
            Link.convection("block", "bay_air", 15.0, 1.2),
            # Fan-driven bay air while the engine runs; still air after key-off at 1200 s.
            Link.convection("bracket", "bay_air", lambda t: 60.0 if t < 1200.0 else 5.0, 0.05),
            Link.convection("wing", "air", lambda t: 8.0 if t < 1200.0 else 4.0, 1.5),
            Link.convection("bay_air", "air", 6.0, 2.0),
            Link("block", "coolant", 40.0),
        ],
        radiation=[
            RadiationLink("block", "wing", emissivity=0.8, area_m2=0.3, view_factor=0.4),
        ],
        link_nodes=[LinkNode("mount", "block", "wing", conductance_w_k=2.0, capacity_j_k=900.0)],
        sources=[ImposedHeat("block", lambda t: 9000.0 if 30.0 <= t < 1200.0 else 0.0)],
        initial_k=AMBIENT_K,
    )


def test_energy_closes_every_tick_to_a_microwatt_in_a_kilowatt() -> None:
    """Σ C dT/dt = sources − heat into the fixed nodes, per tick, to 1e-6 relative."""
    net = _bay()
    for _ in range(40):
        before = np.asarray(net.all_temperatures_k()[: net.n_free])
        net.advance(net.t_s, 60.0)
        residual = net.energy_residual_w(before, 60.0)
        scale = max(1.0, float(np.sum(np.abs(net.imposed_w(net.t_s)))), 1e3)
        assert abs(residual) < 1e-6 * scale, (net.t_s, residual)


def test_a_sign_error_in_one_conductor_breaks_the_identity() -> None:
    """The negative control: flip one link's sign in the operator and the budget opens."""
    net = _bay()
    net.advance(net.t_s, 60.0)
    before = np.asarray(net.all_temperatures_k()[: net.n_free])
    net.advance(net.t_s, 60.0)
    assert abs(net.energy_residual_w(before, 60.0)) < 1e-3
    k = net._last_k.copy()  # the matrix the step applied
    i, j = net.names.index("block"), net.names.index("bracket")
    k[i, j] = k[j, i] = -k[i, j]
    with pytest.raises(ValueError, match="negative"):
        from irsim.thermal.conduction import ConductionOperator

        ConductionOperator(k, np.ones(k.shape[0]))
    # And a wrong-sign flow, computed by hand from the applied matrix, misses the budget.
    temps = net.all_temperatures_k()
    stored = float(np.sum(net.capacity_j_k * (temps[: net.n_free] - before) / 60.0))
    wrong_flow_into_fixed = -float(np.sum((k @ temps)[net.n_free :]))  # sign flipped on purpose
    residual = stored - (float(np.sum(net.imposed_w(net.t_s))) - wrong_flow_into_fixed)
    assert abs(residual) > 1.0


def test_parts_warm_in_conductance_order_and_the_bracket_lags_the_block() -> None:
    """Block → bracket → wing: the R2 picture, and a first check that the mount lags."""
    net = _bay()
    _settle(net, hours=0.25)  # 15 min after key-on at 30 s
    t = net.temperatures_k
    assert t["block"] > t["bracket"] > t["wing"] > AMBIENT_K - 3.5
    assert t["bracket"] - AMBIENT_K > 5.0, "a bolted bracket follows the block within minutes"
    assert t["mount"] < t["block"], "rubber lags"


def test_hot_soak_the_bracket_keeps_warming_after_the_block_s_heat_stops() -> None:
    """At key-off (1200 s) the imposed heat stops and the fan-driven convection with it; the
    bracket, still fed by the block's stored heat through its joint, *rises* for a minute or two
    before it cools -- the under-hood measurement every survey quotes (MVFRI R04-13: skins peak
    1-2 min after key-off), produced here by a callable h and a node with mass, not by a script.
    Measured: +3.7 K, peaking ~90 s after key-off."""
    net = _bay()
    net.advance_to(1200.0, 60.0)
    at_key_off = net.temperature("bracket")
    times, temps = [], []
    while net.t_s < 1500.0:
        net.advance(net.t_s, 10.0)
        times.append(net.t_s - 1200.0)
        temps.append(net.temperature("bracket"))
    peak = int(np.argmax(temps))
    assert temps[peak] - at_key_off > 2.0, (temps[peak], at_key_off)
    assert 30.0 <= times[peak] <= 180.0, times[peak]
    assert temps[-1] < temps[peak], "and then it cools"
    assert net.temperature("block") < 440.0  # coolant and air hold it under §6.6's band top


# --- stiffness, shapes and refusals -------------------------------------------------------------


def test_a_stiff_joint_stands_under_a_scene_tick() -> None:
    """A 50 g bracket on a 25 W/K bolt: τ ≈ 1 s. Sixty-second ticks converge, no ringing."""
    net = ThermalNetwork(
        nodes=[Node("bracket", 0.05 * 470.0)],
        fixed=[FixedNode("block", 390.0)],
        links=[Link("bracket", "block", 25.0)],
        initial_k=AMBIENT_K,
    )
    history = [net.advance(net.t_s, 60.0)[0] for _ in range(5)]
    assert all(b >= a for a, b in zip(history[:-1], history[1:], strict=True))
    assert history[-1] == pytest.approx(390.0, abs=1e-6)


def test_the_fixed_node_is_honoured_at_the_tick_s_end() -> None:
    """A node bolted tightly to a moving boundary tracks the boundary's end-of-tick value."""
    net = ThermalNetwork(
        nodes=[Node("skin", 1.0)],
        fixed=[FixedNode("core", lambda t: 300.0 + 0.1 * t)],
        links=[Link("skin", "core", 1e6)],
        initial_k=300.0,
    )
    net.advance(0.0, 10.0)
    assert net.temperature("skin") == pytest.approx(301.0, abs=1e-4)
    assert net.temperatures_k["core"] == pytest.approx(301.0)


def test_shapes_and_names_are_refused_loudly() -> None:
    with pytest.raises(ValueError, match="unknown node"):
        ThermalNetwork(nodes=[Node("a", 1.0)], links=[Link("a", "ghost", 1.0)])
    with pytest.raises(ValueError, match="unique"):
        ThermalNetwork(nodes=[Node("a", 1.0)], fixed=[FixedNode("a", 300.0)])
    with pytest.raises(ValueError, match="goes nowhere"):
        ThermalNetwork(
            nodes=[Node("a", 1.0)],
            fixed=[FixedNode("air", 300.0)],
            sources=[ImposedHeat("air", 5.0)],
        )
    with pytest.raises(ValueError, match="capacity must be positive"):
        Node("a", 0.0)
    with pytest.raises(ValueError, match="cannot join"):
        Link("a", "a", 1.0)
    with pytest.raises(ValueError, match="negative"):
        Link("a", "b", -1.0)
    with pytest.raises(ValueError, match="at least one node"):
        ThermalNetwork(nodes=[], fixed=[FixedNode("air", 300.0)])
    with pytest.raises(ValueError, match="missing"):
        ThermalNetwork(nodes=[Node("a", 1.0), Node("b", 1.0)], initial_k={"a": 300.0})
    with pytest.raises(ValueError, match="stands at"):
        ThermalNetwork(nodes=[Node("a", 1.0)]).advance(5.0, 1.0)
    with pytest.raises(ValueError, match="emissivity"):
        RadiationLink("a", "b", emissivity=0.0, area_m2=1.0)


def test_the_conductance_matrix_is_what_the_operator_checks() -> None:
    """The network hands `ConductionOperator` a symmetric, zero-diagonal matrix every tick."""
    net = _bay()
    k = net.conductances_w_k(net.t_s)
    assert np.array_equal(k, k.T) and not k.diagonal().any()
    assert net.n_free == 5 and net.n_fixed == 2 and net.names[-2:] == ("air", "coolant")
