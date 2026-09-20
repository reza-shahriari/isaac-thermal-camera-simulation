"""TC.1 — heat moves between facets, and the scene tick survives it.

Three claims, each with a way to fail. **Nothing changes without an operator**: a solver with no
conduction, or with a zero operator, must be bit-identical to the midpoint rule it always was.
**The implicit step is right**: a stiff RC ladder that forward Euler blows up at a 60 s tick settles
to its closed-form equilibrium to 1e-6 and its slow mode converges at first order (halve the tick,
halve the error — a scheme that was merely stable would not). **Energy closes**: for unequal cells
under a real forcing, `Σ C_i A_i ΔT_i` equals `dt Σ A_i F_i` to 1e-9, which a sign error in the
Laplacian, a missing area weight or an asymmetric matrix each break.

And the guard that never existed: a leaf-like facet at a scene tick now raises with the bound.

docs/physics-model.md §6.1, §6.4; ADR 0036, ADR 0094
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from irsim.thermal.conduction import ConductionOperator, explicit_bound_s
from irsim.thermal.facets import FacetForcing, FacetProperties, FacetSolver
from irsim.thermal.field import ThermalField
from irsim.thermal.surface_field import PlanarPatch, PlanarThermalField

DT_S = 60.0


def _props(capacity, emissivity=0.0, absorptivity=0.0) -> FacetProperties:  # type: ignore[no-untyped-def]
    c = np.asarray(capacity, dtype=np.float64)
    return FacetProperties(
        heat_capacity_j_m2_k=c,
        emissivity=np.full(c.shape, float(emissivity)),
        solar_absorptivity=np.full(c.shape, float(absorptivity)),
    )


def _chain(n: int, k_w_k: float, area: float = 1.0) -> ConductionOperator:
    """``n`` facets in a line, each linked to the next by ``k_w_k``."""
    rows = np.arange(n - 1)
    k = sp.coo_matrix((np.full(n - 1, k_w_k), (rows, rows + 1)), shape=(n, n))
    return ConductionOperator(k + k.T, np.full(n, area))


ADIABATIC = FacetForcing(t_air_k=290.0, h_w_m2_k=0.0)


# ---------------------------------------------------------------------------------------------
# nothing changes without an operator
# ---------------------------------------------------------------------------------------------


def _diurnal(t_s: float) -> FacetForcing:
    hour = (t_s / 3600.0) % 24.0
    return FacetForcing(
        t_air_k=288.0 + 6.0 * np.sin(np.pi * (hour - 8.0) / 12.0),
        h_w_m2_k=10.0,
        q_solar_w_m2=max(0.0, 900.0 * np.sin(np.pi * (hour - 6.0) / 12.0)),
        q_longwave_down_w_m2=320.0,
    )


def test_a_zero_operator_is_bit_identical_to_no_operator() -> None:
    """k → 0 is the midpoint rule, bit for bit: the IMEX path must add nothing at zero."""
    props = _props([101_200.0, 4_399.0, 2_500 * 840 * 0.005], emissivity=0.9, absorptivity=0.9)
    zero = ConductionOperator(sp.csr_matrix((3, 3)), np.ones(3))
    plain = FacetSolver(props, np.full(3, 288.0))
    imex = FacetSolver(props, np.full(3, 288.0), conduction=zero)
    for i in range(240):
        plain.advance(_diurnal(i * DT_S), DT_S)
        imex.advance(_diurnal(i * DT_S), DT_S)
    assert np.array_equal(plain.temperatures_k, imex.temperatures_k)


# ---------------------------------------------------------------------------------------------
# the implicit step is right
# ---------------------------------------------------------------------------------------------


def test_a_stiff_ladder_settles_to_its_equilibrium_where_forward_euler_diverges() -> None:
    """Three cells, links with τ ≈ 0.1 s, stepped at 60 s: 600× past the explicit limit."""
    capacity = np.array([100.0, 100.0, 100.0])  # J/m²/K, area 1 m² each
    ladder = _chain(3, k_w_k=500.0)  # τ = C/(2k) ≈ 0.1 s for the middle cell
    start = np.array([300.0, 280.0, 320.0])
    solver = FacetSolver(_props(capacity), start.copy(), conduction=ladder, guard=False)
    history = [start.copy()]
    for _ in range(6):
        history.append(solver.advance(ADIABATIC, DT_S).copy())
    equilibrium = float(np.mean(start))  # equal capacities, adiabatic: the mean survives
    assert np.allclose(history[-1], equilibrium, rtol=0.0, atol=1e-6)
    # Stable: the spread between the hottest and coldest cell shrinks every step and never
    # overshoots, and the total stored energy is conserved to roundoff. (A single cell need not
    # move monotonically: the end cell that starts *at* the mean is pulled off it by its one,
    # colder neighbour before the far end's heat arrives -- which is what conduction does.)
    for before, after in zip(history[:-1], history[1:], strict=True):
        assert np.ptp(after) < np.ptp(before)
        assert float(np.sum(capacity * after)) == pytest.approx(
            float(np.sum(capacity * before)), rel=1e-12
        )
    # And the explicit rule on the same problem is not just inaccurate; it explodes.
    explicit = start.copy()
    for _ in range(6):
        explicit = explicit + DT_S * ladder.power_in_w(explicit) / capacity
    assert np.abs(explicit - equilibrium).max() > 1e6


def test_the_slow_mode_converges_at_first_order_to_its_closed_form() -> None:
    """Two equal cells, τ = C/(2k) = 1 h: backward Euler's error must halve with the tick."""
    c, k = 3600.0, 0.5  # τ = 3600/(2·0.5) = 3600 s
    pair = _chain(2, k_w_k=k)
    start = np.array([310.0, 290.0])
    errors = []
    for dt in (120.0, 60.0, 30.0):
        solver = FacetSolver(_props([c, c]), start.copy(), conduction=pair, guard=False)
        steps = int(round(7200.0 / dt))
        for _ in range(steps):
            solver.advance(ADIABATIC, dt)
        exact = 300.0 + np.array([10.0, -10.0]) * np.exp(-7200.0 / 3600.0)
        errors.append(float(np.abs(solver.temperatures_k - exact).max()))
    assert errors[1] < 0.05  # ≈ 0.7 % of the 10 K amplitude at dt = 60 s
    assert 1.8 < errors[0] / errors[1] < 2.2
    assert 1.8 < errors[1] / errors[2] < 2.2


def test_the_operator_s_power_has_the_right_sign_and_conserves() -> None:
    op = _chain(2, k_w_k=3.0)
    power = op.power_in_w([300.0, 290.0])
    assert power[0] == pytest.approx(-30.0) and power[1] == pytest.approx(30.0)
    assert power.sum() == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------------------------
# energy closes, with unequal cells and a real forcing
# ---------------------------------------------------------------------------------------------


def test_energy_closes_to_a_part_in_a_billion_under_forcing() -> None:
    rng = np.random.default_rng(7)
    n = 50
    area = rng.uniform(0.02, 0.5, n)
    capacity = rng.uniform(2_000.0, 120_000.0, n)
    dense = rng.uniform(0.0, 5.0, (n, n)) * (rng.uniform(size=(n, n)) < 0.15)
    k = np.triu(dense, 1)
    op = ConductionOperator(sp.csr_matrix(k + k.T), area)
    props = _props(capacity, emissivity=0.93, absorptivity=0.85)
    forcing = FacetForcing(
        t_air_k=285.0,
        h_w_m2_k=rng.uniform(4.0, 25.0, n),
        q_solar_w_m2=rng.uniform(0.0, 800.0, n),
        q_longwave_down_w_m2=310.0,
        q_internal_w_m2=rng.uniform(-50.0, 200.0, n),
    )
    solver = FacetSolver(props, rng.uniform(280.0, 320.0, n), conduction=op)
    for _ in range(20):
        before = solver.temperatures_k
        # The explicit half evaluates the balance at the midpoint; reproduce it to book the
        # energy it actually deposited.
        half = before + 0.5 * DT_S * solver.net_flux(before, forcing) / capacity
        deposited = DT_S * float(np.sum(area * solver.net_flux(half, forcing)))
        after = solver.advance(forcing, DT_S)
        stored = float(np.sum(capacity * area * (after - before)))
        assert stored == pytest.approx(deposited, rel=1e-9)


# ---------------------------------------------------------------------------------------------
# the operator refuses what would invent energy
# ---------------------------------------------------------------------------------------------


def test_an_asymmetric_matrix_is_refused() -> None:
    k = sp.csr_matrix(np.array([[0.0, 2.0], [1.0, 0.0]]))
    with pytest.raises(ValueError, match="symmetric"):
        ConductionOperator(k, np.ones(2))


def test_a_negative_conductance_a_self_link_and_a_bad_area_are_refused() -> None:
    with pytest.raises(ValueError, match="negative"):
        ConductionOperator(sp.csr_matrix(np.array([[0.0, -1.0], [-1.0, 0.0]])), np.ones(2))
    with pytest.raises(ValueError, match="diagonal"):
        ConductionOperator(sp.csr_matrix(np.array([[1.0, 1.0], [1.0, 0.0]])), np.ones(2))
    with pytest.raises(ValueError, match="positive area"):
        _chain(2, 1.0, area=0.0)
    with pytest.raises(ValueError, match="facet areas"):
        ConductionOperator(sp.csr_matrix((3, 3)), np.ones(2))


def test_an_operator_of_the_wrong_size_is_refused_by_the_solver() -> None:
    with pytest.raises(ValueError, match="links 3 facets"):
        FacetSolver(_props([1.0, 1.0]), np.full(2, 290.0), conduction=_chain(3, 1.0))


# ---------------------------------------------------------------------------------------------
# the guard that never existed
# ---------------------------------------------------------------------------------------------


def test_a_leaf_at_a_scene_tick_raises_with_the_bound() -> None:
    """630 J m⁻² K⁻¹ at h = 30: the bound is ≈ 36 s and a 60 s tick used to diverge quietly."""
    leaf = _props([630.0], emissivity=0.96)
    windy = FacetForcing(t_air_k=290.0, h_w_m2_k=30.0)
    bound = float(explicit_bound_s(630.0, 30.0, 0.96, 290.0))
    assert 33.0 < bound < 38.0  # 2·630 / (30 + 5.3): the radiative term is a sixth of the loss
    with pytest.raises(ValueError, match=r"exceeds the §6.4 explicit bound"):
        FacetSolver(leaf, np.array([290.0])).advance(windy, 60.0)
    FacetSolver(leaf, np.array([290.0])).advance(windy, 20.0)  # inside the bound: fine
    FacetSolver(leaf, np.array([290.0]), guard=False).advance(windy, 60.0)  # opted out


def test_the_bound_is_evaluated_on_the_actual_h_not_a_worst_case() -> None:
    """Still air makes the same leaf steppable at 60 s; the guard must see that."""
    leaf = _props([630.0], emissivity=0.96)
    still = FacetForcing(t_air_k=290.0, h_w_m2_k=5.0)
    assert float(explicit_bound_s(630.0, 5.0, 0.96, 290.0)) > 60.0
    FacetSolver(leaf, np.array([290.0])).advance(still, 60.0)


# ---------------------------------------------------------------------------------------------
# through the fields
# ---------------------------------------------------------------------------------------------


def test_a_field_carries_the_operator_and_reuses_one_factorisation() -> None:
    props = _props([100.0, 100.0, 100.0])
    field = ThermalField(
        props,
        lambda _t: ADIABATIC,
        0.0,
        np.array([300.0, 280.0, 320.0]),
        DT_S,
        conduction=_chain(3, 500.0),
    )
    # The guard is on; an adiabatic forcing has no h, so nothing to trip.
    field.advance_to(6 * DT_S)
    assert np.allclose(field.temperature_at(6 * DT_S), 300.0, atol=1e-4)
    assert field.conduction is not None
    assert field._solver._factor is not None and field._solver._factor[0] == DT_S  # noqa: SLF001


def test_a_planar_field_takes_an_operator_over_its_cells() -> None:
    patch = PlanarPatch(
        origin_m=np.zeros(3),
        u_axis=np.array([1.0, 0.0, 0.0]),
        v_axis=np.array([0.0, 0.0, 1.0]),
        n_u=4,
        n_v=1,
        du_m=0.1,
        dv_m=0.1,
        thickness_m=0.05,
    )
    op = _chain(4, k_w_k=2.0, area=patch.cell_area_m2)
    field = PlanarThermalField(
        patch,
        _props(np.full(4, 4_399.0)),
        lambda _t: ADIABATIC,
        0.0,
        np.array([310.0, 300.0, 300.0, 290.0]),
        10.0,
        conduction=op,
    )
    field.advance_to(3600.0)
    temps = field.temperature_at(3600.0).astype(np.float64)
    assert temps.max() - temps.min() < 20.0 - 1e-6  # heat flowed from the hot end to the cold
    assert float(temps.mean()) == pytest.approx(300.0, abs=1e-4)  # and none was invented
