"""TC.16 -- mesh members, explicit contactors and footprint links in the coupled solve.

ADR 0099's coupled solve took planar patches only, because its contactors are rectangle
clipping. A motor can bolted to a mesh arm and a pack resting on a shell are the joins TC.11
needs, so a member may now be a `TriangleMeshPatch`, a contact may be an explicit matrix built
by proximity, and a lumped part may be bolted to the cells under its footprint. The identities
are the old ones: energy closes, no contactor is the separate fields bit for bit, and a
conductance is the number that was authored. Roadmap TC.16; ADR 0099, ADR 0112.
"""

from __future__ import annotations

import numpy as np
import pytest

from irsim.thermal.coupling import (
    Contactor,
    CoupledFields,
    ExplicitContactor,
    FieldMember,
    LumpedLink,
    LumpedMember,
    footprint_conductances,
    proximity_contactor,
)
from irsim.thermal.facets import FacetForcing, FacetProperties
from irsim.thermal.mesh_conduction import mesh_lateral_operator
from irsim.thermal.mesh_field import TriangleMeshField, TriangleMeshPatch
from irsim.thermal.raycast import cylinder_mesh
from irsim.thermal.surface_field import PlanarPatch

EX, EY, EZ = np.eye(3)


def _tube(centre=(0.0, 0.0, 0.0), radius=0.015, length=0.3, level=2) -> TriangleMeshPatch:
    soup = cylinder_mesh(np.asarray(centre), radius, length, axis=(1.0, 0.0, 0.0), n_phi=12, n_z=6)
    return TriangleMeshPatch.uniform(soup.vertices, soup.faces, level=level)


def _plate(origin=(-0.05, -0.05, 0.02), n=4, d=0.025) -> PlanarPatch:
    return PlanarPatch(
        origin_m=np.asarray(origin),
        u_axis=EX,
        v_axis=EY,
        n_u=n,
        n_v=n,
        du_m=d,
        dv_m=d,
        thickness_m=0.002,
    )


def _props(n: int, c: float = 3000.0) -> FacetProperties:
    """Cells that neither radiate nor see the sun: conduction is the only physics."""
    return FacetProperties(
        heat_capacity_j_m2_k=np.full(n, c), emissivity=np.zeros(n), solar_absorptivity=np.zeros(n)
    )


def _adiabatic(t_s: float) -> FacetForcing:
    return FacetForcing(t_air_k=300.0, h_w_m2_k=0.0)


def _member(name: str, patch, initial, c: float = 3000.0) -> FieldMember:  # type: ignore[no-untyped-def]
    n = patch.n_cells
    return FieldMember(name, patch, _props(n, c), _adiabatic, np.full(n, float(initial)))


# --- a mesh is a member -------------------------------------------------------------------------


def test_a_mesh_member_with_no_contactor_is_the_separate_mesh_field_bit_for_bit() -> None:
    tube = _tube()
    k = mesh_lateral_operator(tube, 7.0, 0.0015)
    initial = 300.0 + 20.0 * (tube.cell_centres()[:, 0] > 0.0)  # one half hot
    alone = TriangleMeshField(
        tube, _props(tube.n_cells), _adiabatic, 0.0, initial, 10.0, conduction=k
    )
    coupled = CoupledFields(
        [FieldMember("arm", tube, _props(tube.n_cells), _adiabatic, initial, conduction=k)],
        t0_s=0.0,
        tick_s=10.0,
    )
    alone.advance_to(600.0)
    coupled.advance_to(600.0)
    assert np.array_equal(alone.temperature_at(600.0), coupled.fields["arm"].temperature_at(600.0))
    assert coupled.fields["arm"].patch is tube
    assert np.ptp(np.asarray(alone.temperature_at(600.0))) < 20.0, "the tube conducted"


# --- an explicit contact between a mesh and a plate ---------------------------------------------


def test_a_plate_bolted_to_a_tube_by_proximity_conserves_energy_and_flows_downhill() -> None:
    """A 10 x 10 cm plate 5 mm over a 30 mm tube: the cells within 25 mm of each other share an
    authored 4 cm2 contact at 3000 W/m2/K. Stored energy is constant to 1e-9 and the hot
    plate cools into the tube, whose cells nearest the plate warm first."""
    tube, plate = _tube(), _plate()
    k_ab = proximity_contactor(plate, tube, 3000.0, 4e-4, gap_m=0.025)
    assert k_ab.sum() == pytest.approx(3000.0 * 4e-4, rel=1e-12)
    coupled = CoupledFields(
        [_member("plate", plate, 340.0), _member("tube", tube, 300.0)],
        [ExplicitContactor("plate", "tube", k_ab)],
        t0_s=0.0,
        tick_s=5.0,
    )
    before = coupled.stored_energy_j()
    coupled.advance_to(1200.0)
    assert coupled.stored_energy_j() == pytest.approx(before, rel=1e-9)
    plate_t = np.asarray(coupled.fields["plate"].temperature_at(1200.0), dtype=np.float64)
    tube_t = np.asarray(coupled.fields["tube"].temperature_at(1200.0), dtype=np.float64)
    assert plate_t.mean() < 340.0 and tube_t.mean() > 300.0
    near = np.asarray(k_ab.sum(axis=0)).ravel() > 0.0
    assert tube_t[near].mean() > tube_t[~near].mean() + 0.5, "the contact warms its own cells"


def test_proximity_refuses_parts_that_do_not_touch_and_clipping_refuses_a_mesh() -> None:
    tube, plate = _tube(), _plate(origin=(-0.05, -0.05, 0.5))
    with pytest.raises(ValueError, match="do not touch"):
        proximity_contactor(plate, tube, 3000.0, 4e-4, gap_m=0.025)
    with pytest.raises(ValueError, match="needs planar patches"):
        CoupledFields(
            [_member("plate", plate, 300.0), _member("tube", tube, 300.0)],
            [Contactor("plate", "tube", 3000.0)],
            t0_s=0.0,
        )
    with pytest.raises(ValueError, match="matrix is"):
        CoupledFields(
            [_member("plate", plate, 300.0), _member("tube", tube, 300.0)],
            [ExplicitContactor("plate", "tube", np.ones((2, 2)))],
            t0_s=0.0,
        )


# --- a hidden part on its footprint -------------------------------------------------------------


def test_a_footprint_sums_to_the_authored_conductance_and_takes_the_nearest_cells() -> None:
    tube = _tube()
    g = footprint_conductances(tube, (0.12, 0.0, 0.015), 3e-4, 2000.0)
    assert g.sum() == pytest.approx(2000.0 * 3e-4, rel=1e-12)
    touched = np.flatnonzero(g > 0.0)
    centres = tube.cell_centres()
    far = np.flatnonzero(g == 0.0)
    assert np.abs(centres[touched, 0] - 0.12).max() < np.abs(centres[far, 0] - 0.12).max()
    with pytest.raises(ValueError, match="larger than the whole part"):
        footprint_conductances(tube, (0.0, 0.0, 0.0), 10.0, 2000.0)


def test_a_hot_lumped_part_bolted_to_a_tube_warms_the_cells_under_it_first_and_keeps_warming() -> (
    None
):
    """The ESC in the arm (AI.11) as TC.11 will solve it: 12 g at 900 J/kg/K dissipating 2 W into
    a 3 cm2 footprint. Its heat reaches the cells under the footprint first, and after the
    dissipation stops the tube goes on warming from the part's stored heat."""
    tube = _tube()
    g = footprint_conductances(tube, (0.10, 0.0, 0.015), 3e-4, 2000.0)

    def esc_forcing(t_s: float) -> FacetForcing:
        return FacetForcing(
            t_air_k=300.0, h_w_m2_k=0.0, q_internal_w_m2=2.0 if t_s < 600.0 else 0.0
        )

    coupled = CoupledFields(
        [_member("arm", tube, 300.0)],
        t0_s=0.0,
        tick_s=5.0,
        lumped=[LumpedMember("esc", 0.012 * 900.0, esc_forcing, 300.0)],
        lumped_links=[LumpedLink("arm", "esc", 0.0, per_cell_w_k=g)],
    )
    coupled.advance_to(600.0)
    under = g > 0.0
    t600 = np.asarray(coupled.fields["arm"].temperature_at(600.0), dtype=np.float64)
    assert t600[under].mean() > t600[~under].mean() + 0.5
    assert coupled.node_temperature_k("esc") > t600.max()
    arm_600 = t600.mean()
    coupled.advance_to(1200.0)
    t1200 = np.asarray(coupled.fields["arm"].temperature_at(1200.0), dtype=np.float64)
    assert t1200.mean() > arm_600, "the arm keeps warming after the ESC stops"
    assert (
        coupled.node_temperature_k("esc") < coupled.fields["arm"].field.latest_state_k.max() + 50.0
    )
    # energy: 2 W for 600 s went in and nothing left
    stored = coupled.stored_energy_j() - (
        0.012 * 900.0 * 300.0 + float((3000.0 * tube.cell_area_m2 * 300.0).sum())
    )
    assert stored == pytest.approx(2.0 * 600.0, rel=1e-6)
