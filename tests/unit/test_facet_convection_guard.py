"""PT.25 -- free convection at the facet's own temperature, and the latent slope in the guard.

`SceneSurfaceForcing` evaluated ``h_free = c |T_s − T_air|^{1/3}`` at a fixed 300 K whatever a
facet's own temperature was, so a hot roof in still air was cooled as if it were at 300 K; and
§6.4's explicit bound counted ``h + 4εσT³`` only, while a wet surface at 10 m/s loses ~45 W m⁻² K⁻¹
more through the latent term's slope than through convection -- a step the guard passed could
diverge. docs/physics-model.md §6.2, §6.4; roadmap PT.25, PH.1.
"""

from __future__ import annotations

import numpy as np
import pytest

from irsim.thermal.convection import DEFAULT_CONVECTION
from irsim.thermal.facets import FacetForcing, FacetProperties, FacetSolver
from irsim.thermal.latent import (
    bulk_conductance_kg_m2_s,
    latent_flux_derivative_w_m2_k,
    latent_heat_flux_w_m2,
    specific_humidity_kg_kg,
)


def _properties(capacity: float = 20000.0) -> FacetProperties:
    return FacetProperties(
        heat_capacity_j_m2_k=np.array([capacity]),
        emissivity=np.array([0.9]),
        solar_absorptivity=np.array([0.9]),
    )


def test_free_convection_follows_the_facet_not_a_fixed_300_k() -> None:
    """Still air, a facet 60 K over air: §6.2 gives c·60^{1/3} = 5.88, above the a = 5 floor.
    The old forcing, at 300 K over 288 K air, gave the floor."""
    forcing = FacetForcing(t_air_k=288.15, h_w_m2_k=5.0, free_convection_c=DEFAULT_CONVECTION.c)
    hot = float(forcing.effective_h(np.array([348.15]), 1)[0])
    warm = float(forcing.effective_h(np.array([298.15]), 1)[0])
    assert hot == pytest.approx(1.5 * 60.0 ** (1.0 / 3.0), rel=1e-12)
    assert warm == 5.0, "below the 37 K crossover the forced floor wins, as before"
    solver = FacetSolver(_properties(), 348.15)
    fixed = FacetForcing(t_air_k=288.15, h_w_m2_k=5.0)
    assert float(solver.net_flux(np.array([348.15]), forcing)[0]) < float(
        solver.net_flux(np.array([348.15]), fixed)[0]
    ), "the hot facet loses more than the fixed-300 K coefficient allowed"


def test_a_producer_without_the_field_is_bit_identical() -> None:
    forcing = FacetForcing(t_air_k=288.15, h_w_m2_k=np.array([7.0]))
    assert np.array_equal(forcing.effective_h(np.array([400.0]), 1), np.array([7.0]))


def test_the_latent_slope_matches_a_finite_difference_of_the_flux() -> None:
    q_air = float(specific_humidity_kg_kg(293.15, 0.5))
    g_e = float(bulk_conductance_kg_m2_s(10.0))
    t = np.array([298.15])
    slope = float(latent_flux_derivative_w_m2_k(t, q_air, g_e)[0])
    fd = float(
        (latent_heat_flux_w_m2(t + 0.5, q_air, g_e) - latent_heat_flux_w_m2(t - 0.5, q_air, g_e))[0]
    )
    assert slope == pytest.approx(fd, rel=1e-3)
    assert 30.0 < slope < 60.0, f"~45 W/m²/K at 10 m/s for a fully wet surface; got {slope:.1f}"


def test_the_guard_counts_the_latent_slope() -> None:
    """A leaf-like facet (C = 630 J m⁻² K⁻¹), fully wet, at 10 m/s: the bound is
    2C/(h + 4εσT³ + ∂Q_L/∂T) ≈ 16 s where the old guard said 35 s. A 20 s step must raise."""
    q_air = float(specific_humidity_kg_kg(293.15, 0.5))
    g_e = float(bulk_conductance_kg_m2_s(10.0))
    h = float(5.0 + 4.0 * 10.0**0.8)
    wet = FacetForcing(
        t_air_k=293.15, h_w_m2_k=h, q_air_kg_kg=q_air, g_e_kg_m2_s=g_e, wet_fraction=1.0
    )
    dry = FacetForcing(t_air_k=293.15, h_w_m2_k=h)
    FacetSolver(_properties(630.0), 298.15).advance(dry, 20.0)  # passes: bound ≈ 35 s
    with pytest.raises(ValueError, match="explicit bound"):
        FacetSolver(_properties(630.0), 298.15).advance(wet, 20.0)
    FacetSolver(_properties(630.0), 298.15).advance(wet, 10.0)  # inside the wet bound
