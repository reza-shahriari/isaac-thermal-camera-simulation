"""TC.3 — contactors and radiation between fields.

Two overlapping patches at h_c = 1e4 W m⁻² K⁻¹ must give a total conductance ``h_c · A_overlap``
and the *same* total after one grid is refined four times -- the property a conductance authored
per node cannot have, and the reason a contactor is defined by overlap area. The coupled solve
is one solver, so a joint whose time constant is a fraction of a second stands under a 60 s tick
and the stored energy of two patches joined by a contactor is conserved to solver precision.

Radiation reuses ADR 0088's view factors in both directions: the power an underbody radiates
onto the road equals the power the road's cells receive, and reciprocity is checked against an
independent quadrature from the body's side.

docs/physics-model.md §6.1, §6.4; ADR 0087, ADR 0088, ADR 0094, ADR 0099; roadmap TC.3.
"""

from __future__ import annotations

import numpy as np
import pytest

from irsim.radiometry.constants import SIGMA_SB
from irsim.thermal.coupling import (
    Contactor,
    CoupledFields,
    FieldMember,
    RadiationExchange,
    cell_overlap_areas,
    contactor_conductances,
)
from irsim.thermal.facets import FacetForcing, FacetProperties
from irsim.thermal.spatial_sources import RadiantRectangle
from irsim.thermal.surface_field import PlanarPatch, PlanarThermalField

EX = np.array([1.0, 0.0, 0.0])
EY = np.array([0.0, 1.0, 0.0])
EZ = np.array([0.0, 0.0, 1.0])
H_C = 1e4  # a new bolted ferrous joint (Voller & Tirovic 2007)


def _grid(origin, n_u, n_v, cell, u=EX, v=EY, thickness=0.02):  # type: ignore[no-untyped-def]
    return PlanarPatch(
        origin_m=np.asarray(origin, dtype=float),
        u_axis=u,
        v_axis=v,
        n_u=n_u,
        n_v=n_v,
        du_m=cell,
        dv_m=cell,
        thickness_m=thickness,
    )


def _props(n: int, c: float = 8_000.0, eps: float = 0.0) -> FacetProperties:
    return FacetProperties(
        heat_capacity_j_m2_k=np.full(n, c),
        emissivity=np.full(n, eps),
        solar_absorptivity=np.full(n, 0.5),
    )


def _adiabatic(_t: float) -> FacetForcing:
    return FacetForcing(t_air_k=300.0, h_w_m2_k=0.0)


# --- overlap ------------------------------------------------------------------------------------


def test_the_contactor_total_is_h_c_times_the_overlap_and_survives_refinement() -> None:
    """A 1 m² plate 1 mm above a 2 m × 2 m base, grids of 0.1 and 0.25 m, then 0.0625 m."""
    base = _grid([0.0, 0.0, 0.0], 20, 20, 0.1)
    plate = _grid([0.35, 0.55, 0.001], 4, 4, 0.25)
    fine = _grid([0.35, 0.55, 0.001], 16, 16, 0.0625)
    k_coarse = contactor_conductances(base, plate, H_C)
    k_fine = contactor_conductances(base, fine, H_C)
    assert k_coarse.shape == (400, 16) and k_fine.shape == (400, 256)
    assert abs(k_coarse.sum() - H_C * 1.0) < 1e-9 * H_C
    assert abs(k_fine.sum() - k_coarse.sum()) < 1e-9 * H_C
    # Every plate cell's row sums to its own area: nothing is lost or double-counted.
    per_cell = np.asarray(cell_overlap_areas(base, plate).sum(axis=0)).ravel()
    assert np.allclose(per_cell, 0.25 * 0.25, rtol=0.0, atol=1e-12)


def test_a_rotated_grid_needs_no_alignment() -> None:
    """A 1 m square turned 30° inside the base still overlaps exactly its own area."""
    base = _grid([0.0, 0.0, 0.0], 20, 20, 0.1)
    c, s = np.cos(np.radians(30.0)), np.sin(np.radians(30.0))
    turned = _grid([0.6, 0.3, 0.0], 8, 8, 0.125, u=np.array([c, s, 0.0]), v=np.array([-s, c, 0.0]))
    total = cell_overlap_areas(base, turned).sum()
    assert abs(total - 1.0) < 1e-9


def test_a_plate_overhanging_the_base_counts_only_what_touches_and_a_per_node_total_does_not() -> (
    None
):
    """Half the plate hangs off the base. The contactor sees 0.5 m²; a conductance authored per
    node (h_c times each plate cell's area, nearest base cell) sees the whole plate and, refined,
    a different number again."""
    base = _grid([0.0, 0.0, 0.0], 20, 20, 0.1)
    plate = _grid([1.5, 0.5, 0.0], 4, 4, 0.25)  # x from 1.5 to 2.5: half outside
    assert abs(cell_overlap_areas(base, plate).sum() - 0.5) < 1e-9
    per_node_coarse = H_C * plate.cell_area_m2 * plate.n_cells
    fine = _grid([1.5, 0.5, 0.0], 16, 16, 0.0625)
    per_node_fine = H_C * fine.cell_area_m2 * fine.n_cells
    assert per_node_coarse == pytest.approx(H_C * 1.0)  # the whole plate, overhang included
    assert per_node_fine == pytest.approx(per_node_coarse)  # and no better when refined
    assert abs(contactor_conductances(base, plate, H_C).sum() - H_C * 0.5) < 1e-9 * H_C


def test_non_parallel_far_apart_or_other_frame_patches_are_refused() -> None:
    base = _grid([0.0, 0.0, 0.0], 10, 10, 0.1)
    with pytest.raises(ValueError, match="parallel"):
        cell_overlap_areas(base, _grid([0.0, 0.0, 0.0], 4, 4, 0.1, u=EX, v=EZ))
    with pytest.raises(ValueError, match="apart"):
        cell_overlap_areas(base, _grid([0.0, 0.0, 0.9], 4, 4, 0.1))
    other = PlanarPatch(
        origin_m=np.zeros(3), u_axis=EX, v_axis=EY, n_u=2, n_v=2, du_m=0.1, dv_m=0.1, frame="/Car"
    )
    with pytest.raises(ValueError, match="frames"):
        cell_overlap_areas(base, other)
    with pytest.raises(ValueError, match="negative"):
        contactor_conductances(base, base, -1.0)


# --- the coupled solve --------------------------------------------------------------------------


def _members(t_plate: float = 350.0) -> tuple[FieldMember, FieldMember]:
    base = _grid([0.0, 0.0, 0.0], 20, 20, 0.1)
    plate = _grid([0.35, 0.55, 0.001], 4, 4, 0.25)
    return (
        FieldMember("base", base, _props(base.n_cells), _adiabatic, 300.0),
        FieldMember("plate", plate, _props(plate.n_cells), _adiabatic, t_plate),
    )


def test_a_hot_plate_on_a_cold_base_equalises_and_conserves_energy_at_a_scene_tick() -> None:
    """τ of a plate cell across the joint: C A / (h_c A) = 8000/1e4 = 0.8 s, under 60 s ticks."""
    coupled = CoupledFields(_members(), [Contactor("base", "plate", H_C)], t0_s=0.0, tick_s=60.0)
    e0 = coupled.stored_energy_j()
    plate, base = coupled.fields["plate"], coupled.fields["base"]
    assert float(plate.temperature_at(0.0).mean()) == 350.0
    plate.advance_to(3600.0)  # advancing one view advances both
    assert base.latest_t_s == 3600.0
    e1 = coupled.stored_energy_j()
    assert abs(e1 - e0) < 1e-9 * e0
    state = coupled.field.latest_state_k  # float64: the float32 query cannot show 1e-9
    t_base, t_plate = state[:400], state[400:]
    assert np.allclose(plate.temperature_at(3600.0), t_plate, atol=1e-3)
    # Each plate cell has equalised with the base cells it touches (neither patch conducts
    # laterally here), so the plate is between the two starting temperatures, well below 350 K.
    assert np.all((t_plate > 300.0) & (t_plate < 350.0)) and float(t_plate.mean()) < 330.0
    # The plate sits on the cells it overlaps and nowhere else: those base cells warmed, the rest
    # never received anything (no lateral operator in the base).
    touched = (
        np.asarray(cell_overlap_areas(_members()[0].patch, _members()[1].patch).sum(axis=1)).ravel()
        > 0
    )
    assert t_base[touched].min() > 300.0 + 1.0
    assert np.all(t_base[~touched] == 300.0)
    # Energy left the plate and arrived in the touched cells, to the joule.
    c_a = 8_000.0
    lost = c_a * 0.25 * 0.25 * float(np.sum(350.0 - t_plate))
    gained = c_a * 0.1 * 0.1 * float(np.sum(t_base[touched] - 300.0))
    assert lost == pytest.approx(gained, rel=1e-9)


def test_without_a_contactor_the_members_are_the_separate_fields_bit_for_bit() -> None:
    members = _members()
    coupled = CoupledFields(members, [], t0_s=0.0, tick_s=60.0)
    separate = [
        PlanarThermalField(m.patch, m.properties, m.forcing_at, 0.0, m.initial_k, 60.0)
        for m in members
    ]
    for t in (600.0, 1800.0):
        coupled.advance_to(t)
        for m, f in zip(members, separate, strict=True):
            f.advance_to(t)
            assert np.array_equal(coupled.fields[m.name].temperature_at(t), f.temperature_at(t))


def test_the_views_carry_the_bridge_s_interface() -> None:
    coupled = CoupledFields(_members(), [Contactor("base", "plate", H_C)], t0_s=0.0, tick_s=30.0)
    coupled.advance_to(90.0)
    view = coupled.fields["plate"]
    assert view.patch is coupled.members[1].patch and view.tick_s == 30.0 and view.t0_s == 0.0
    centres = view.patch.cell_centres()
    sampled = view.sample_at(90.0, centres)
    assert np.allclose(sampled, view.temperature_at(90.0), atol=1e-4)
    assert view.temperature_image(90.0).shape == view.patch.shape
    assert np.isnan(view.sample_at(90.0, np.array([[5.0, 5.0, 0.0]])))[0]
    with pytest.raises(ValueError, match="unknown member"):
        CoupledFields(_members(), [Contactor("base", "roof", H_C)], t0_s=0.0)
    with pytest.raises(ValueError, match="unique"):
        CoupledFields([_members()[0], _members()[0]], [], t0_s=0.0)


# --- radiation ----------------------------------------------------------------------------------


def test_the_power_an_underbody_radiates_onto_the_road_is_the_power_the_cells_receive() -> None:
    road = _grid([-3.0, -3.0, 0.0], 60, 60, 0.1, thickness=0.05)
    underbody = RadiantRectangle(
        centre_m=np.array([0.0, 0.0, 0.28]),
        u_axis=EX,
        v_axis=EY,
        half_u_m=0.82,
        half_v_m=1.9,
        emissivity=0.88,
    )
    exchange = RadiationExchange(underbody, road, surface_emissivity=0.95)
    t_body = 320.0
    leaving = exchange.intercepted_power_w(t_body)
    arriving = exchange.arriving_power_w(t_body)
    assert leaving == pytest.approx(arriving, rel=1e-6)
    # Reciprocity between two discretisations: the body's own quadrature agrees within 1 %.
    assert exchange.rect_to_patch_view_factor == pytest.approx(
        exchange.reverse_view_factor(), rel=1e-2
    )
    assert 0.9 < exchange.rect_to_patch_view_factor <= 1.0  # a low body over a wide road
    # And the cells' net term is ADR 0088's, sky included: with a black sky it is pure gain.
    q = exchange.cell_flux_w_m2(t_body)
    assert np.all(q >= 0.0) and q.max() == pytest.approx(
        exchange.view_factors.max() * 0.95 * 0.88 * SIGMA_SB * t_body**4
    )
    with pytest.raises(ValueError, match="positive"):
        exchange.intercepted_power_w(0.0)
