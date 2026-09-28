"""PT.26 -- in-plane conduction uses the in-plane conductivity.

`lateral_operator` and `mesh_lateral_operator` took `conductivity_w_mk`, the through-thickness
value a slab solver wants, so a carbon laminate (0.8 W/m/K through, ~7 along the fibres) spread
heat along its skin nine times too slowly and the shadow terminator on a quadrotor's arm fell
across one 25 mm cell where the fibres carry it over three. docs/physics-model.md §6.1, §6.7
(S41); ADR 0102, ADR 0112; roadmap PT.26.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from irsim.config.materials import ThermalSpec
from irsim.materials.library import MaterialLibrary
from irsim.thermal.conduction import lateral_operator
from irsim.thermal.surface_field import PlanarPatch

EX = np.array([1.0, 0.0, 0.0])
EY = np.array([0.0, 1.0, 0.0])


def _spec(**overrides: float) -> ThermalSpec:
    base = dict(
        density_kg_m3=1600.0,
        specific_heat_j_kgk=1050.0,
        conductivity_w_mk=0.8,
        thickness_m=0.0015,
        solar_absorptivity=0.9,
    )
    base.update(overrides)
    return ThermalSpec(**base)


def test_an_isotropic_material_reads_its_own_k() -> None:
    assert _spec().inplane_conductivity_w_mk == 0.8


def test_a_laminate_reads_its_in_plane_k_and_the_schema_guards_it() -> None:
    assert _spec(conductivity_inplane_w_mk=7.0).inplane_conductivity_w_mk == 7.0
    with pytest.raises(ValueError):
        _spec(conductivity_inplane_w_mk=0.0)


def test_the_carbon_fibre_entry_carries_the_in_plane_value() -> None:
    thermal = MaterialLibrary.load()["carbon_fibre"].spec.thermal
    assert thermal.conductivity_w_mk == pytest.approx(0.8)
    assert thermal.inplane_conductivity_w_mk == pytest.approx(7.0)


def test_the_lateral_operator_scales_with_the_in_plane_k() -> None:
    """Same patch, same thickness: every link conductance scales by k_inplane / k_through, and
    the 300 s diffusion length √(α t) goes from 12 mm to 35 mm on the carbon laminate."""
    patch = PlanarPatch(
        origin_m=np.zeros(3),
        u_axis=EX,
        v_axis=EY,
        n_u=12,
        n_v=2,
        du_m=0.025,
        dv_m=0.025,
        thickness_m=0.002,
    )
    thermal = _spec(conductivity_inplane_w_mk=7.0)
    through = lateral_operator(patch, thermal.conductivity_w_mk, thermal.thickness_m)
    inplane = lateral_operator(patch, thermal.inplane_conductivity_w_mk, thermal.thickness_m)
    ratio = inplane.conductance_w_k.toarray() / np.where(
        through.conductance_w_k.toarray() > 0.0, through.conductance_w_k.toarray(), 1.0
    )
    assert np.allclose(ratio[through.conductance_w_k.toarray() > 0.0], 7.0 / 0.8, rtol=1e-12)
    c = thermal.heat_capacity_j_m2_k
    for k, expect_mm in ((0.8, 12.0), (7.0, 35.5)):
        alpha = k * thermal.thickness_m / c
        assert math.sqrt(alpha * 300.0) * 1e3 == pytest.approx(expect_mm, abs=0.6)
