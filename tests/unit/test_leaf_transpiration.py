"""Leaves transpire: the surface whose temperature its stomata set (PH.11).

The phenomenology, measured on the shipped broadleaf at 30 °C, 2 m/s, 800 W/m² of sun: a
**well-watered leaf sits 1.9 K below air** at a 3.2 kPa deficit and a **stressed one 7.0 K above**
it, under identical weather. That gap is the whole reason vegetation is worth modelling in an
infrared scene, and it is produced by one number -- the stomatal resistance -- and nothing else.

Four checks, in rising order of how much they could have failed:

* closing the stomata converges to the dry-surface reference (+7.48 K), exactly;
* a well-watered leaf goes **below** air as the air dries;
* the slope of (T_leaf − T_air) against vapour pressure deficit is **−1.84 °C/kPa**, inside
  Idso's published non-water-stressed band of [−3.8, −1.1];
* Campbell & Norman's closed form, written in **molar** units from the published constants,
  agrees with this project's SI mass-based balance to 0.10 K near air temperature.

⚠️ **The last one does not hold to 0.1 K everywhere, and this file says where it stops.** The
agreement degrades to 0.37 K by 4.6 K of departure from air, because the closed form linearises
both the saturation curve and σT⁴ about air temperature. Reporting only the favourable case would
have made an approximation look like an identity.

docs/physics-model.md §6.1; roadmap PH.11; ADR 0121
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from irsim.materials.hemispherical import total_hemispherical_emissivity
from irsim.materials.library import MaterialLibrary
from irsim.radiometry.constants import SIGMA_SB
from irsim.thermal.balance import SurfaceForcing, ThermalProperties, stability_limit_s
from irsim.thermal.latent import bulk_conductance_kg_m2_s
from irsim.thermal.vegetation import (
    C_P_MOLAR_J_MOL_K,
    IDSO_SLOPE_C_PER_KPA,
    M_AIR_KG_MOL,
    PSYCHROMETER_CONSTANT_PER_K,
    STRESSED_R_S_S_M,
    WELL_WATERED_R_S_S_M,
    leaf_boundary_conductance_mol_m2_s,
    leaf_forcing,
    leaf_steady_state_k,
    leaf_temperature_cn,
    leaf_time_constant_s,
    radiative_conductance_mol_m2_s,
    vapour_pressure_deficit_kpa,
)

pytestmark = pytest.mark.slow

T_AIR_K = 303.15
WIND_M_S = 2.0
SOLAR_W_M2 = 800.0
SKY_EMISSIVITY = 0.80


@pytest.fixture(scope="module")
def leaf() -> ThermalProperties:
    material = MaterialLibrary.load()["vegetation_leaf"]
    thermal = material.spec.thermal
    return ThermalProperties(
        heat_capacity_j_m2_k=thermal.heat_capacity_j_m2_k,
        emissivity=total_hemispherical_emissivity(material, 300.0).value,
        solar_absorptivity=thermal.solar_absorptivity,
    )


def _forcing(rh: float, r_s: float, solar: float = SOLAR_W_M2) -> SurfaceForcing:
    return leaf_forcing(T_AIR_K, rh, WIND_M_S, solar, SKY_EMISSIVITY * SIGMA_SB * T_AIR_K**4, r_s)


# -- the leaf's own boundary layer ---------------------------------------------------------------


def test_a_leaf_boundary_layer_is_an_order_below_the_projects_bulk_one() -> None:
    """33 s/m against 435 s/m at 2 m/s, and using the wrong one inverts the phenomenology.

    `bulk_conductance_kg_m2_s` describes a metres-deep surface layer over soil or sea. A 5 cm leaf
    in the same wind is thirteen times better ventilated, because the air only has to cross five
    centimetres. With the bulk value a well-watered leaf cannot transpire fast enough to go below
    air at all -- it reads +6.4 K at a 3.4 kPa deficit instead of −1.9 K, which is the wrong sign
    of the only effect this module exists to produce.
    """
    rho = 1.225
    g_va = leaf_boundary_conductance_mol_m2_s(WIND_M_S) / 0.93
    leaf_r_a = rho / (M_AIR_KG_MOL * g_va)
    bulk_r_a = rho / float(bulk_conductance_kg_m2_s(WIND_M_S))

    assert leaf_r_a == pytest.approx(32.9, abs=1.0)
    assert bulk_r_a == pytest.approx(435.0, abs=10.0)
    assert bulk_r_a / leaf_r_a > 10.0

    # and it scales as sqrt(u/d), which is the shape of the published correlation
    assert leaf_boundary_conductance_mol_m2_s(8.0) == pytest.approx(
        2.0 * leaf_boundary_conductance_mol_m2_s(2.0), rel=1e-12
    )
    assert leaf_boundary_conductance_mol_m2_s(2.0, 0.20) == pytest.approx(
        0.5 * leaf_boundary_conductance_mol_m2_s(2.0, 0.05), rel=1e-12
    )
    # still air still ventilates: the floor stands in for free convection
    assert leaf_boundary_conductance_mol_m2_s(0.0) > 0.0
    with pytest.raises(ValueError):
        leaf_boundary_conductance_mol_m2_s(-1.0)


def test_the_sensible_and_latent_paths_share_one_boundary_layer(leaf) -> None:  # type: ignore[no-untyped-def]
    """h and g_e are the same turbulence carrying two things; a forcing must not split them."""
    forcing = _forcing(0.5, WELL_WATERED_R_S_S_M)
    g_ha = leaf_boundary_conductance_mol_m2_s(WIND_M_S)
    assert forcing.h_w_m2_k == pytest.approx(C_P_MOLAR_J_MOL_K * g_ha, rel=1e-12)
    assert forcing.g_e_kg_m2_s == pytest.approx(M_AIR_KG_MOL * g_ha / 0.93, rel=1e-12)
    assert forcing.wet_fraction == 1.0
    del leaf


# -- TC.1's guard ---------------------------------------------------------------------------------


def test_a_leaf_breaks_a_sixty_second_explicit_tick(leaf) -> None:  # type: ignore[no-untyped-def]
    """PH.11's stated reason for solving the root instead of stepping it, measured.

    The leaf's 630 J m⁻² K⁻¹ and its well-ventilated boundary layer give a 15.4 s time constant,
    so the midpoint rule is stable to 30.8 s. A snowpack's limit is 3089 s and asphalt's 8825 s --
    the leaf is two orders of magnitude more restrictive than anything else in this library, which
    is why it is the material that forced the question.
    """
    forcing = _forcing(0.4, WELL_WATERED_R_S_S_M)
    assert leaf.heat_capacity_j_m2_k == pytest.approx(630.0)
    assert leaf_time_constant_s(leaf, forcing) == pytest.approx(15.4, abs=0.5)

    limit = stability_limit_s(T_AIR_K, leaf, forcing)
    assert limit == pytest.approx(30.8, abs=1.0)
    assert limit < 60.0, "a 60 s tick is past the explicit limit -- that is the guard"

    heavier = ThermalProperties(
        heat_capacity_j_m2_k=63000.0, emissivity=0.95, solar_absorptivity=0.5
    )
    assert stability_limit_s(T_AIR_K, heavier, forcing) > 100.0 * limit


# -- the four acceptance criteria ----------------------------------------------------------------


def test_closing_the_stomata_converges_to_the_dry_reference(leaf) -> None:  # type: ignore[no-untyped-def]
    """g_s → 0 must reproduce the surface that does not evaporate at all, and monotonically."""
    dry = SurfaceForcing(
        t_air_k=T_AIR_K,
        h_w_m2_k=_forcing(0.4, 0.0).h_w_m2_k,
        q_solar_w_m2=SOLAR_W_M2,
        q_longwave_down_w_m2=SKY_EMISSIVITY * SIGMA_SB * T_AIR_K**4,
    )
    reference = leaf_steady_state_k(leaf, dry) - T_AIR_K
    assert reference == pytest.approx(7.48, abs=0.1), "the dry leaf sits above air in sun"

    series = [
        leaf_steady_state_k(leaf, _forcing(0.4, r)) - T_AIR_K for r in (50.0, 100.0, 500.0, 5000.0)
    ]
    assert all(b > a for a, b in itertools.pairwise(series)), series
    closed = leaf_steady_state_k(leaf, _forcing(0.4, 1e7)) - T_AIR_K
    assert closed == pytest.approx(reference, abs=1e-2)


def test_a_well_watered_leaf_goes_below_air_as_the_air_dries(leaf) -> None:  # type: ignore[no-untyped-def]
    """The headline: −1.93 K at a 3.18 kPa deficit, where a stressed leaf is +7.0 K.

    Same sun, same wind, same air: only the stomatal resistance differs, and it is worth nine
    kelvin. A model that could not produce both signs would be useless for anything a crop water
    stress index is built on.
    """
    humid = leaf_steady_state_k(leaf, _forcing(0.85, WELL_WATERED_R_S_S_M)) - T_AIR_K
    dry_air = leaf_steady_state_k(leaf, _forcing(0.25, WELL_WATERED_R_S_S_M)) - T_AIR_K
    assert humid > 0.0, "in humid air even a watered leaf cannot shed enough to go below"
    assert dry_air < 0.0, "in dry air it does"
    assert dry_air == pytest.approx(-1.93, abs=0.2)

    stressed = leaf_steady_state_k(leaf, _forcing(0.25, STRESSED_R_S_S_M)) - T_AIR_K
    assert stressed == pytest.approx(7.0, abs=0.3)
    assert stressed - dry_air > 8.0, "the stomata are worth more than eight kelvin"


def test_the_stress_slope_lands_inside_idsos_published_baseline(leaf) -> None:  # type: ignore[no-untyped-def]
    """−1.84 °C/kPa against a published [−3.8, −1.1], and the band was not fitted to.

    Idso's non-water-stressed baselines are a linear fit of (T_canopy − T_air) against vapour
    pressure deficit for well-watered crops. This is the one number in the module compared against
    something outside the repo, and the model is asked whether it lands inside rather than being
    tuned until it does.
    """
    humidities = (0.85, 0.70, 0.55, 0.40, 0.25)
    deficits = [vapour_pressure_deficit_kpa(T_AIR_K, rh) for rh in humidities]
    departures = [
        leaf_steady_state_k(leaf, _forcing(rh, WELL_WATERED_R_S_S_M)) - T_AIR_K for rh in humidities
    ]
    slope, _ = np.polyfit(deficits, departures, 1)

    low, high = IDSO_SLOPE_C_PER_KPA
    assert low <= slope <= high, f"slope {slope:.3f} outside Idso's {low}..{high}"
    assert slope == pytest.approx(-1.84, abs=0.15)
    # linear enough for a slope to be the right summary -- Idso's baseline is a straight line
    residuals = np.asarray(departures) - np.polyval(np.polyfit(deficits, departures, 1), deficits)
    assert np.max(np.abs(residuals)) < 0.05, residuals

    # a stressed canopy's slope is much flatter, which is what makes the baseline diagnostic
    stressed = [
        leaf_steady_state_k(leaf, _forcing(rh, STRESSED_R_S_S_M)) - T_AIR_K for rh in humidities
    ]
    stressed_slope, _ = np.polyfit(deficits, stressed, 1)
    assert abs(stressed_slope) < 0.3 * abs(slope)


# -- the independent oracle ----------------------------------------------------------------------


def test_campbell_norman_and_this_projects_balance_are_the_same_equation(leaf) -> None:  # type: ignore[no-untyped-def]
    """Where both linearisations vanish they agree to 4e-10 K -- the bisection tolerance.

    With no net isothermal radiation and saturated air, the closed form's two terms are each
    identically zero and the balance's root is air temperature. Any disagreement here would be a
    units error in the molar bridge rather than an approximation, so this is the test that says
    the two formulations are the same physics before the finite-departure test says how far apart
    their approximations drift.
    """
    forcing = leaf_forcing(T_AIR_K, 1.0, WIND_M_S, 0.0, SIGMA_SB * T_AIR_K**4, WELL_WATERED_R_S_S_M)
    ours = leaf_steady_state_k(leaf, forcing)
    closed = leaf_temperature_cn(forcing, leaf, 1.0)
    assert closed == pytest.approx(T_AIR_K, abs=1e-12)
    assert ours == pytest.approx(T_AIR_K, abs=1e-8)
    assert abs(ours - closed) < 1e-8


def test_the_closed_form_agrees_to_a_tenth_of_a_kelvin_near_air_and_drifts_beyond(leaf) -> None:  # type: ignore[no-untyped-def]
    """PH.11's fourth criterion, with the range over which it is true stated rather than assumed.

    0.10 K while the leaf is within about a kelvin of air; 0.37 K by 4.6 K of departure. The floor
    is the unit bridge (our exact ``q = ε e/(p − (1−ε)e)`` against the molar side's ``q ≈ ε e/p``,
    1.6 % of the deficit at 30 °C); the growth is Campbell & Norman linearising both the
    saturation curve and σT⁴ about air temperature. Quoting 0.1 K without the qualifier would have
    turned an approximation into a claimed identity.
    """
    gaps = {}
    for rh in (0.85, 0.55, 0.25):
        for r_s in (50.0, WELL_WATERED_R_S_S_M, 300.0):
            forcing = _forcing(rh, r_s)
            ours = leaf_steady_state_k(leaf, forcing)
            gaps[abs(ours - T_AIR_K)] = abs(ours - leaf_temperature_cn(forcing, leaf, rh))

    near = [gap for departure, gap in gaps.items() if departure < 1.0]
    far = [gap for departure, gap in gaps.items() if departure > 4.0]
    assert near and far, sorted(gaps)
    assert max(near) < 0.12, near
    assert max(gaps.values()) < 0.4, gaps
    assert max(far) > 2.0 * min(near), "the gap grows with departure, which is the linearisation"

    # and it grows with departure rather than randomly
    departures = np.array(sorted(gaps))
    ordered = np.array([gaps[d] for d in departures])
    assert np.corrcoef(departures, ordered)[0, 1] > 0.8, (departures, ordered)


def test_the_molar_constants_reproduce_the_published_ones() -> None:
    """29.105 J/mol/K and 6.594e-4 /K -- **derived**, and each about 1 % under the published pair.

    Campbell & Norman tabulate 29.3 and 6.66e-4. Deriving both from this project's own SI
    constants rather than pasting theirs keeps the molar oracle tied to the balance it checks; the
    cost is a 0.67 % gap on c_p (this project carries dry-air 1005 J/kg/K, they use a moist-air
    value) and 0.99 % on γ. Both are asserted **as the derived values with the published ones
    named**, because a test that loosened its tolerance until 29.105 counted as 29.3 would be
    hiding the only thing here worth knowing.
    """
    c_p_molar, gamma = C_P_MOLAR_J_MOL_K, PSYCHROMETER_CONSTANT_PER_K
    assert c_p_molar == pytest.approx(29.105, abs=0.001)
    assert gamma == pytest.approx(6.594e-4, rel=1e-3)
    c_p_gap = 29.3 / C_P_MOLAR_J_MOL_K - 1.0
    gamma_gap = 6.66e-4 / PSYCHROMETER_CONSTANT_PER_K - 1.0
    assert c_p_gap == pytest.approx(0.0067, abs=5e-4)
    assert gamma_gap == pytest.approx(0.0099, abs=5e-4)
    # radiation as a conductance is the same order as a still-air leaf's boundary layer
    g_r = radiative_conductance_mol_m2_s(300.0, 0.97)
    assert g_r == pytest.approx(0.19, abs=0.02)
    assert 0.02 < g_r / leaf_boundary_conductance_mol_m2_s(WIND_M_S) < 0.5


def test_a_leaf_forcing_without_a_boundary_layer_is_refused(leaf) -> None:  # type: ignore[no-untyped-def]
    """A zero vapour conductance is not a still leaf, it is a missing forcing."""
    from irsim.thermal.vegetation import conductances_from_forcing

    broken = SurfaceForcing(t_air_k=T_AIR_K, h_w_m2_k=10.0)
    with pytest.raises(ValueError, match="positive bulk vapour conductance"):
        conductances_from_forcing(broken, leaf.emissivity)
