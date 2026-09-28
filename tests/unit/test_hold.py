"""TC.12 -- evolve or freeze, per object.

`evolve: false` holds an object at its spun-up state; `freeze_at_s: t` solves it to t and holds
it from then on. A frozen object still radiates to its neighbours, which go on changing; unset,
every field, target and network is what it was, bit for bit. ADR 0158; roadmap TC.12.
"""

from __future__ import annotations

import numpy as np
import pytest

from irsim.thermal.facets import FacetForcing, FacetProperties
from irsim.thermal.field import ThermalField
from irsim.thermal.hold import HeldSolver, hold_from_s
from irsim.thermal.network import Link, Node, ThermalNetwork
from irsim.thermal.object_exchange import ExchangeBody, ObjectExchange
from irsim.thermal.solvers import NewtonCoolingSolver
from irsim.thermal.surface_field import PlanarPatch, PlanarThermalField
from irsim.thermal.weather import WeatherSample, WeatherSeries

EX, EY, EZ = np.eye(3)


def _props(n: int = 1, c: float = 2.0e4, eps: float = 0.9) -> FacetProperties:
    return FacetProperties(
        heat_capacity_j_m2_k=np.full(n, c),
        emissivity=np.full(n, eps),
        solar_absorptivity=np.full(n, 0.9),
    )


def _warming(t_s: float) -> FacetForcing:
    """A forcing that never lets the facet rest: warm air, strong sun."""
    return FacetForcing(
        t_air_k=300.0, h_w_m2_k=10.0, q_solar_w_m2=600.0, q_longwave_down_w_m2=350.0
    )


# --- the field ---------------------------------------------------------------------------------


def test_a_field_unset_is_the_field_it_was_and_held_it_stops_on_its_tick() -> None:
    plain = ThermalField(_props(), _warming, 0.0, np.array([280.0]), 10.0)
    same = ThermalField(_props(), _warming, 0.0, np.array([280.0]), 10.0)
    assert same.hold_from_s is None
    held = ThermalField(_props(), _warming, 0.0, np.array([280.0]), 10.0)
    held.hold_from_s = 100.0
    for f in (plain, same, held):
        f.advance_to(300.0)
    assert plain.state_hash() == same.state_hash()
    a = np.asarray(plain.temperature_at(100.0))
    assert np.array_equal(np.asarray(held.temperature_at(100.0)), a), "identical up to the hold"
    assert np.array_equal(np.asarray(held.temperature_at(300.0)), a), "and unchanged after it"
    assert plain.temperature_at(300.0)[0] > a[0] + 1.0, "the free field went on warming"
    assert held.n_ticks == plain.n_ticks, "held ticks are still ticks: the clock does not stop"


def test_a_freeze_between_ticks_holds_from_the_first_tick_past_it() -> None:
    held = ThermalField(_props(), _warming, 0.0, np.array([280.0]), 10.0)
    held.hold_from_s = 95.0
    held.advance_to(200.0)
    t100 = np.asarray(held.temperature_at(100.0))
    assert np.array_equal(np.asarray(held.temperature_at(200.0)), t100)
    assert held.temperature_at(90.0)[0] < t100[0], "still solving before the hold"


def test_one_facet_of_a_field_can_be_held_while_the_others_evolve() -> None:
    field = ThermalField(_props(3), _warming, 0.0, np.full(3, 280.0), 10.0)
    field.hold_from_s = np.array([np.inf, 50.0, 0.0])
    field.advance_to(200.0)
    t = np.asarray(field.temperature_at(200.0), dtype=np.float64)
    assert t[2] == 280.0, "held from t0: the spun-up state"
    assert t[1] == pytest.approx(float(field.temperature_at(50.0)[1]))
    assert t[0] > t[1] > t[2]


# --- the exchange keeps seeing a frozen body ---------------------------------------------------


def _road_and_pan() -> tuple[PlanarThermalField, PlanarThermalField, ObjectExchange]:
    ground = PlanarPatch(
        origin_m=np.array([-1.0, -1.0, 0.0]),
        u_axis=EX,
        v_axis=EY,
        n_u=6,
        n_v=6,
        du_m=1 / 3,
        dv_m=1 / 3,
        thickness_m=0.05,
    )
    hot = PlanarPatch(
        origin_m=np.array([-0.5, 0.5, 0.25]),
        u_axis=EX,
        v_axis=-EY,
        n_u=2,
        n_v=2,
        du_m=0.5,
        dv_m=0.5,
        thickness_m=0.002,
    )
    night = lambda t: FacetForcing(t_air_k=280.0, h_w_m2_k=8.0, q_longwave_down_w_m2=250.0)  # noqa: E731
    asphalt = PlanarThermalField(
        ground, _props(36, 2e5, 0.95), night, 0.0, np.full(36, 281.0), 10.0
    )
    steel = PlanarThermalField(hot, _props(4, 5e3, 0.9), night, 0.0, np.full(4, 320.0), 10.0)
    group = ObjectExchange(
        [ExchangeBody.from_planar("road", ground, 0.95), ExchangeBody.from_planar("pan", hot, 0.9)]
    )
    return group.register("road", asphalt), group.register("pan", steel), group  # type: ignore[return-value]


def test_a_frozen_pan_keeps_warming_the_road_under_it() -> None:
    """The owner's case: a warm engine at night, kept warm while the camera films. The pan is
    held at 320 K; the road beneath it goes on receiving the pan's radiation and warms, where
    the same road under a pan left to cool receives less every tick."""
    road_free, pan_free, _ = _road_and_pan()
    road_held, pan_held, _ = _road_and_pan()
    pan_held.field.hold_from_s = 0.0
    road_free.advance_to(1800.0)
    road_held.advance_to(1800.0)
    assert np.all(np.asarray(pan_held.temperature_at(1800.0)) == np.float32(320.0))
    assert np.asarray(pan_free.temperature_at(1800.0)).max() < 319.0, "the free pan cooled"
    centre = road_held.patch.cell_centres()
    under = (np.abs(centre[:, 0]) < 0.5) & (np.abs(centre[:, 1]) < 0.5)
    warm_held = np.asarray(road_held.temperature_at(1800.0), dtype=np.float64)
    warm_free = np.asarray(road_free.temperature_at(1800.0), dtype=np.float64)
    assert warm_held[under].mean() > warm_free[under].mean() + 0.05
    assert warm_held[under].mean() > 281.0, "the road under a held pan warms, at night"


# --- lumped targets -----------------------------------------------------------------------------


def _newton() -> NewtonCoolingSolver:
    weather = WeatherSeries.constant(
        WeatherSample(300.0, 0.3, 1.0, 0.0, 0.0, 0.0, 23000.0, 0.0), 7200.0
    )
    return NewtonCoolingSolver(350.0, 600.0, weather, t0_s=0.0)


def test_a_held_target_is_its_inner_solver_until_the_hold_and_constant_after() -> None:
    plain, inner = _newton(), _newton()
    held = HeldSolver(inner, hold_from_s=250.0)
    assert not held.frozen
    for t in range(0, 200, 50):
        assert held.advance(float(t), 50.0) == plain.advance(float(t), 50.0)
    assert not held.frozen
    # 200 -> 300 straddles the hold: the inner solver is stepped exactly to 250 and held there
    at_250 = held.advance(200.0, 100.0)
    plain.advance(200.0, 50.0)
    assert held.frozen and at_250 == pytest.approx(plain.temperature(), rel=1e-12)
    assert held.advance(300.0, 100.0) == at_250 and held.temperature() == at_250
    assert held.state.t_s == 400.0 and held.state.temperature_k == at_250
    assert held.weather is inner.weather, "the one-weather guard sees through the wrapper"


def test_hold_from_s_spells_both_switches() -> None:
    assert hold_from_s(100.0, True, None) is None
    assert hold_from_s(100.0, False, None) == 100.0
    assert hold_from_s(100.0, True, 30.0) == 130.0
    with pytest.raises(ValueError, match="negative"):
        hold_from_s(100.0, True, -1.0)


# --- the network ----------------------------------------------------------------------------------


def test_a_held_node_is_a_boundary_its_neighbour_keeps_moving_toward() -> None:
    def net() -> ThermalNetwork:
        return ThermalNetwork(
            nodes=[Node("hot", capacity_j_k=5000.0), Node("cold", capacity_j_k=5000.0)],
            links=[Link("hot", "cold", conductance_w_k=5.0)],
            t0_s=0.0,
            initial_k={"hot": 350.0, "cold": 290.0},
        )

    free, held = net(), net()
    held.hold("hot", 0.0)
    for _ in range(60):
        free.advance(free.t_s, 10.0)
        held.advance(held.t_s, 10.0)
    assert held.temperatures_k["hot"] == 350.0
    assert free.temperatures_k["hot"] < 340.0, "the free hot node cooled toward the cold one"
    assert held.temperatures_k["cold"] > free.temperatures_k["cold"], "a held source warms more"
    with pytest.raises(KeyError):
        held.hold("nobody", 0.0)
