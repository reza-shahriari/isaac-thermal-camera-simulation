"""ADR 0087 — a temperature that varies across one surface, not one value per object.

The test that carries the weight here is `test_internal_heat_step_holds_the_predicted_gradient`.
Everything else in this file could pass on a field that quietly solved one cell and broadcast it;
that one cannot, because it holds **each cell** to its own §6.1 equilibrium root, and the two ends
of the patch have different roots by construction. If the spatial forcing were being averaged,
broadcast or dropped, the two ends would agree and the test would fail by ~10 K.

The second is `test_patch_claims_a_slab_not_a_rectangle`: a bonnet above a road projects into the
road's own (u, v) rectangle, so a patch that tested only its in-plane extent would hand the road's
temperature to the car and produce a picture that looks entirely reasonable.

docs/physics-model.md §6.1, §6.4; ADR 0087
"""

from __future__ import annotations

import numpy as np
import pytest

from irsim.thermal.balance import (
    SurfaceForcing,
    ThermalProperties,
    rk2_step,
    steady_state_temperature,
)
from irsim.thermal.facets import FacetForcing, FacetProperties
from irsim.thermal.surface_field import PlanarPatch, PlanarThermalField

# A thin painted steel panel: light enough that 6000 s is many time constants, so a steady-state
# comparison is a comparison and not a race.
PANEL = ThermalProperties(heat_capacity_j_m2_k=8_000.0, emissivity=0.92, solar_absorptivity=0.85)
T_AIR_K = 290.0
H_W_M2_K = 10.0
Q_LW_W_M2 = 300.0
TICK_S = 2.0


def _patch(n_u: int = 8, n_v: int = 4, **kwargs: object) -> PlanarPatch:
    return PlanarPatch(
        origin_m=np.zeros(3),
        u_axis=np.array([1.0, 0.0, 0.0]),
        v_axis=np.array([0.0, 1.0, 0.0]),
        n_u=n_u,
        n_v=n_v,
        du_m=0.25,
        dv_m=0.25,
        **kwargs,  # type: ignore[arg-type]
    )


def _properties(n_cells: int) -> FacetProperties:
    return FacetProperties.stack([PANEL] * n_cells)


def _field(patch: PlanarPatch, q_internal: np.ndarray | float, t0_k: float = T_AIR_K):
    def forcing_at(_t_s: float) -> FacetForcing:
        return FacetForcing(
            t_air_k=T_AIR_K,
            h_w_m2_k=H_W_M2_K,
            q_longwave_down_w_m2=Q_LW_W_M2,
            q_internal_w_m2=q_internal,
        )

    return PlanarThermalField(patch, _properties(patch.n_cells), forcing_at, 0.0, t0_k, TICK_S)


# ---------------------------------------------------------------------------------------------
# the point of the module
# ---------------------------------------------------------------------------------------------


def test_internal_heat_step_holds_the_predicted_gradient() -> None:
    """Each cell converges to *its own* §6.1 root, not to a shared one.

    Half the patch carries 250 W/m² of internal load and half carries none -- an engine under one
    end of a bonnet. After many time constants every cell must sit at the equilibrium
    `steady_state_temperature` predicts for the flux *that cell* sees.
    """
    patch = _patch()
    q = np.zeros(patch.n_cells)
    hot = patch.cell_centres()[:, 0] > 1.0  # the far half in u
    q[hot] = 250.0

    field = _field(patch, q)
    field.advance_to(6_000.0)
    temps = field.temperature_at(6_000.0)

    for load in (0.0, 250.0):
        expected = steady_state_temperature(
            PANEL,
            SurfaceForcing(
                t_air_k=T_AIR_K,
                h_w_m2_k=H_W_M2_K,
                q_longwave_down_w_m2=Q_LW_W_M2,
                q_internal_w_m2=load,
            ),
        )
        got = temps[q == load]
        assert np.allclose(got, expected, atol=1e-3), f"load {load}: {got.min()}..{got.max()}"

    # And the two halves really are far apart -- a uniform field would pass the loop above only
    # if both roots coincided, so state the separation the picture depends on.
    assert float(temps[hot].mean() - temps[~hot].mean()) > 9.0


def test_a_smooth_source_gives_a_smooth_monotone_gradient() -> None:
    """A falloff in q_internal must appear as a falloff in T, ordered the same way.

    Non-increasing everywhere, and *strictly* decreasing while the source is still above 1 % of
    its peak. Past that the cells differ by less than float32 can hold at 284 K, so demanding a
    strict fall out there would be demanding a number the boundary dtype cannot carry -- which
    is a statement about CLAUDE.md #2, not about the solver.
    """
    patch = _patch(n_u=16, n_v=1)
    u = patch.cell_centres()[:, 0]
    q = 300.0 * np.exp(-((u - u[0]) ** 2) / (2 * 0.5**2))

    field = _field(patch, q)
    field.advance_to(6_000.0)
    temps = field.temperature_at(6_000.0).astype(np.float64)

    assert np.all(np.diff(temps) <= 0.0), "temperature must not rise away from the source"
    lit = q > 0.01 * q.max()
    assert np.all(np.diff(temps[lit]) < 0.0), "a falling source must give a falling temperature"
    assert temps[0] - temps[-1] > 5.0


def test_uniform_patch_tracks_the_scalar_transient() -> None:
    """With uniform forcing every cell must follow the scalar midpoint solver step for step."""
    patch = _patch()
    field = _field(patch, 120.0, t0_k=275.0)
    forcing = SurfaceForcing(
        t_air_k=T_AIR_K,
        h_w_m2_k=H_W_M2_K,
        q_longwave_down_w_m2=Q_LW_W_M2,
        q_internal_w_m2=120.0,
    )
    scalar = 275.0
    for _ in range(300):
        scalar = float(rk2_step(scalar, TICK_S, PANEL, forcing))

    field.advance_to(300 * TICK_S)
    got = field.temperature_at(300 * TICK_S)
    assert np.allclose(got, scalar, atol=1e-4), f"{got.min()}..{got.max()} vs {scalar}"


# ---------------------------------------------------------------------------------------------
# the lookup
# ---------------------------------------------------------------------------------------------


def test_sample_is_exact_at_cell_centres() -> None:
    patch = _patch()
    values = np.arange(patch.n_cells, dtype=np.float64) + 300.0
    got = patch.sample(values, patch.cell_centres())
    assert np.allclose(got, values, atol=1e-12)


def test_sample_is_linear_between_cell_centres() -> None:
    """Halfway between two centres is the mean of the two -- no staircase across a surface."""
    patch = _patch(n_u=4, n_v=1)
    values = np.array([300.0, 310.0, 320.0, 330.0])
    centres = patch.cell_centres()
    midpoints = 0.5 * (centres[:-1] + centres[1:])
    assert np.allclose(patch.sample(values, midpoints), [305.0, 315.0, 325.0], atol=1e-12)


def test_cells_reshape_like_an_image() -> None:
    """``(n_v, n_u)`` with v slow: image[iv, iu] is the cell at that (u, v)."""
    patch = _patch(n_u=5, n_v=3)
    centres = patch.cell_centres().reshape(*patch.shape, 3)
    assert centres[0, 1, 0] == pytest.approx(1.5 * patch.du_m)  # one step along u
    assert centres[1, 0, 1] == pytest.approx(1.5 * patch.dv_m)  # one step along v


def test_patch_claims_a_slab_not_a_rectangle() -> None:
    """A bonnet 0.9 m above a road projects into the road's rectangle and must not be claimed."""
    road = _patch(n_u=8, n_v=8, thickness_m=0.1)
    on_road = np.array([[0.5, 0.5, 0.0]])
    bonnet = np.array([[0.5, 0.5, 0.9]])
    assert bool(road.contains(on_road)[0])
    assert not bool(road.contains(bonnet)[0]), "the road claimed a surface 0.9 m above it"

    values = np.full(road.n_cells, 305.0)
    assert np.isnan(road.sample(values, bonnet)[0])


def test_outside_the_patch_is_nan_not_an_edge_value() -> None:
    """The failure this guards is a patch that silently extends itself to the whole scene."""
    patch = _patch()
    values = np.full(patch.n_cells, 305.0)
    beyond = np.array([[10.0, 0.5, 0.0]])
    assert np.isnan(patch.sample(values, beyond)[0])
    assert patch.sample(values, beyond, fill=0.0)[0] == 0.0


# ---------------------------------------------------------------------------------------------
# the contract the renderer depends on
# ---------------------------------------------------------------------------------------------


def test_a_query_does_not_advance_the_solve() -> None:
    """A renderer asks many times per tick; the answer must not depend on how often.

    The query sits inside the two-tick window a spatial field keeps by default (PT.8, ADR 0093):
    a renderer asks for the tick it just advanced to, which is what this stands for.
    """
    patch = _patch()
    field = _field(patch, 200.0)
    field.advance_to(100.0)
    before = field.state_hash()
    points = patch.cell_centres()
    first = field.sample_at(99.0, points)
    for _ in range(20):
        field.sample_at(99.0, points)
    assert field.state_hash() == before
    assert np.array_equal(field.sample_at(99.0, points), first)


def test_temperature_narrows_to_float32_only_at_the_boundary() -> None:
    patch = _patch()
    field = _field(patch, 200.0)
    field.advance_to(10.0)
    assert field.temperature_at(10.0).dtype == np.float32
    assert field.sample_at(10.0, patch.cell_centres()).dtype == np.float32
    assert field.temperature_image(10.0).shape == patch.shape


def test_construction_rejects_what_would_silently_mislead() -> None:
    with pytest.raises(ValueError, match="perpendicular"):
        PlanarPatch(
            origin_m=np.zeros(3),
            u_axis=np.array([1.0, 0.0, 0.0]),
            v_axis=np.array([1.0, 1.0, 0.0]),
            n_u=2,
            n_v=2,
            du_m=0.1,
            dv_m=0.1,
        )
    with pytest.raises(ValueError, match="thickness_m must be positive"):
        _patch(thickness_m=0.0)
    patch = _patch()
    with pytest.raises(ValueError, match="patch has"):
        patch.sample(np.zeros(patch.n_cells + 1), patch.cell_centres())
    with pytest.raises(ValueError, match="cells"):
        PlanarThermalField(
            patch, _properties(patch.n_cells + 1), lambda _t: FacetForcing(290.0, 10.0), 0.0, 290.0
        )
