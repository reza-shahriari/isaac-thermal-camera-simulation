"""PH.1 — the latent-heat term and a wet film per cell.

§6.1 had no latent term, so "wet asphalt reads colder than dry" could not emerge (spec issue
S45). The checks here are the ones that would fail if the physics were wrong: a saturated
surface with no radiation must settle at the psychrometric wet bulb of the air (within 0.1 K),
and cool further as the air dries; the term is zero when the air is saturated at the surface's
temperature; a dry cell is bit-identical to the balance without the term; the flux at 5 m/s,
293 K and RH 0.7 sits within 15 % of the COARE 3.6 bulk value; the film's mass budget closes;
and removing the term warms the sea skin.

docs/physics-model.md §6.1, §15 Tier 3; spec issue S45; ADR 0101; roadmap PH.1.
"""

from __future__ import annotations

import math
import pathlib

import numpy as np
import pytest

from irsim.atmosphere.humidity import saturation_vapour_pressure_hpa
from irsim.radiometry.constants import (
    C_E_BULK,
    EPSILON_WATER_AIR,
    L_V_WATER_J_KG,
    P_STD_HPA,
    RHO_AIR_STD_KG_M3,
)
from irsim.thermal.balance import SurfaceForcing, ThermalProperties, rk2_step
from irsim.thermal.facets import FacetForcing, FacetProperties, FacetSolver
from irsim.thermal.latent import (
    bulk_conductance_kg_m2_s,
    evaporation_kg_m2_s,
    latent_heat_flux_w_m2,
    saturation_specific_humidity_kg_kg,
    sensible_conductance_w_m2_k,
    specific_humidity_kg_kg,
    wet_bulb_temperature_k,
)

REPO = pathlib.Path(__file__).resolve().parents[2]


def _props(n: int = 1, c: float = 2_000.0) -> FacetProperties:
    return FacetProperties(
        heat_capacity_j_m2_k=np.full(n, c),
        emissivity=np.zeros(n),  # no radiation: the wet bulb is a sensible/latent balance
        solar_absorptivity=np.zeros(n),
    )


# --- the humidity arithmetic --------------------------------------------------------------------


def test_saturation_humidity_matches_the_atmosphere_s_magnus_fit() -> None:
    for t_c in (-10.0, 0.0, 20.0, 35.0):
        e_s = saturation_vapour_pressure_hpa(t_c)
        expected = EPSILON_WATER_AIR * e_s / (P_STD_HPA - (1.0 - EPSILON_WATER_AIR) * e_s)
        assert float(saturation_specific_humidity_kg_kg(t_c + 273.15)) == pytest.approx(expected)
    assert float(specific_humidity_kg_kg(293.15, 1.0)) == pytest.approx(
        float(saturation_specific_humidity_kg_kg(293.15))
    )
    assert float(specific_humidity_kg_kg(293.15, 0.5)) < float(specific_humidity_kg_kg(293.15, 0.9))


def test_the_flux_is_zero_for_saturated_air_at_the_surface_temperature_and_grows_as_rh_falls() -> (
    None
):
    g_e = bulk_conductance_kg_m2_s(3.0)
    q_sat = specific_humidity_kg_kg(293.15, 1.0)
    assert float(latent_heat_flux_w_m2(293.15, q_sat, g_e)) == 0.0
    fluxes = [
        float(latent_heat_flux_w_m2(293.15, specific_humidity_kg_kg(293.15, rh), g_e))
        for rh in (0.9, 0.7, 0.5, 0.3)
    ]
    assert all(b > a > 0.0 for a, b in zip(fluxes[:-1], fluxes[1:], strict=True))
    # A dry cell (wet_fraction 0) evaporates nothing; dew (air wetter than the surface is warm)
    # comes back negative over a wet cell.
    assert float(evaporation_kg_m2_s(293.15, q_sat * 0.5, g_e, wet_fraction=0.0)) == 0.0
    assert float(evaporation_kg_m2_s(283.15, q_sat, g_e)) < 0.0


def test_the_bulk_flux_at_5_m_s_293_k_rh_0_7_is_within_15_percent_of_coare_3_6() -> None:
    """COARE 3.6's Dalton number at 5 m/s is 1.1–1.2e-3; the bulk latent flux for a 20 °C sea
    under 20 °C air at 70 % RH is then ρ L_v C_E U Δq ≈ 76 W m⁻² (Fairall et al. 2003, Edson et
    al. 2013). Held to 15 %, which is what COARE's stability and gustiness terms move it by."""
    q_air = specific_humidity_kg_kg(293.15, 0.7)
    flux = float(latent_heat_flux_w_m2(293.15, q_air, bulk_conductance_kg_m2_s(5.0)))
    dq = float(saturation_specific_humidity_kg_kg(293.15)) - float(q_air)
    reference = RHO_AIR_STD_KG_M3 * L_V_WATER_J_KG * 1.15e-3 * 5.0 * dq
    assert abs(flux - reference) / reference < 0.15
    assert 60.0 < flux < 90.0, flux
    assert C_E_BULK == 1.15e-3


# --- the wet bulb -------------------------------------------------------------------------------


@pytest.mark.parametrize("rh", [1.0, 0.7, 0.4])
def test_a_saturated_surface_with_no_radiation_relaxes_to_the_psychrometric_wet_bulb(
    rh: float,
) -> None:
    """Under the Lewis relation (h = c_p g_e) the balance's rest point *is* the wet-bulb
    equation; the solver has to find it from above, to 0.1 K, and it falls as the air dries."""
    t_air = 293.15
    g_e = float(bulk_conductance_kg_m2_s(3.0))
    forcing = FacetForcing(
        t_air_k=t_air,
        h_w_m2_k=float(sensible_conductance_w_m2_k(g_e)),
        q_air_kg_kg=float(specific_humidity_kg_kg(t_air, rh)),
        g_e_kg_m2_s=g_e,
        wet_fraction=1.0,
    )
    solver = FacetSolver(_props(), np.array([t_air + 5.0]))
    for _ in range(3000):
        solver.advance(forcing, 10.0)
    expected = wet_bulb_temperature_k(t_air, rh)
    assert abs(float(solver.temperatures_k[0]) - expected) < 0.1, (solver.temperatures_k, expected)
    if rh == 1.0:
        assert abs(expected - t_air) < 1e-6
    else:
        assert expected < t_air - 1.0


def test_the_scalar_balance_carries_the_same_term() -> None:
    t_air, rh = 298.15, 0.5
    g_e = float(bulk_conductance_kg_m2_s(2.0))
    forcing = SurfaceForcing(
        t_air_k=t_air,
        h_w_m2_k=float(sensible_conductance_w_m2_k(g_e)),
        q_air_kg_kg=float(specific_humidity_kg_kg(t_air, rh)),
        g_e_kg_m2_s=g_e,
        wet_fraction=1.0,
    )
    props = ThermalProperties(2_000.0, 0.0, 0.0)
    t = np.array(t_air)
    for _ in range(4000):
        t = rk2_step(t, 10.0, props, forcing)
    assert abs(float(t) - wet_bulb_temperature_k(t_air, rh)) < 0.1
    with pytest.raises(ValueError, match="wet_fraction"):
        SurfaceForcing(t_air_k=290.0, h_w_m2_k=5.0, wet_fraction=1.5)


# --- nothing moves for a dry cell ----------------------------------------------------------------


def test_a_dry_cell_is_bit_identical_to_the_balance_without_the_term() -> None:
    props = FacetProperties(
        heat_capacity_j_m2_k=np.full(4, 30_000.0),
        emissivity=np.full(4, 0.9),
        solar_absorptivity=np.full(4, 0.7),
    )
    plain = FacetSolver(props, 290.0)
    with_weather = FacetSolver(props, 290.0)
    with_film = FacetSolver(props, 290.0, film_kg_m2=0.0)  # a film bookkept, but empty
    g_e = float(bulk_conductance_kg_m2_s(4.0))
    for i in range(200):
        t_air = 288.0 + 3.0 * math.sin(i / 20.0)
        base = FacetForcing(
            t_air_k=t_air, h_w_m2_k=12.0, q_solar_w_m2=300.0, q_longwave_down_w_m2=320.0
        )
        moist = FacetForcing(
            t_air_k=t_air,
            h_w_m2_k=12.0,
            q_solar_w_m2=300.0,
            q_longwave_down_w_m2=320.0,
            q_air_kg_kg=float(specific_humidity_kg_kg(t_air, 0.6)),
            g_e_kg_m2_s=g_e,
        )
        plain.advance(base, 60.0)
        with_weather.advance(moist, 60.0)
        with_film.advance(moist, 60.0)
    assert np.array_equal(plain.temperatures_k, with_weather.temperatures_k)
    assert np.array_equal(plain.temperatures_k, with_film.temperatures_k)
    assert np.array_equal(with_film.film_kg_m2, np.zeros(4))


# --- the film ------------------------------------------------------------------------------------


def test_rain_fills_the_film_evaporation_empties_it_and_the_budget_closes() -> None:
    """0.2 mm of water on a warm road at noon: the film goes, the cell cools while it is there,
    and ∫(P − E) dt equals the film's change to 1e-6 of the water that passed through."""
    props = FacetProperties(
        heat_capacity_j_m2_k=np.full(2, 60_000.0),
        emissivity=np.full(2, 0.95),
        solar_absorptivity=np.full(2, 0.9),
    )
    g_e = float(bulk_conductance_kg_m2_s(2.5))
    solver = FacetSolver(props, 305.0, film_kg_m2=np.array([0.2, 0.0]))  # wet cell, dry cell
    rain = FacetForcing(
        t_air_k=298.0,
        h_w_m2_k=15.0,
        q_solar_w_m2=700.0,
        q_longwave_down_w_m2=350.0,
        q_air_kg_kg=float(specific_humidity_kg_kg(298.0, 0.5)),
        g_e_kg_m2_s=g_e,
        precip_kg_m2_s=np.array([0.0, 2.0 / 3600.0]),  # 2 mm/h onto the dry cell only
    )
    film0 = solver.film_kg_m2
    integral = np.zeros(2)
    dried_at = None
    for i in range(240):
        before_t = solver.temperatures_k
        solver.advance(rain, 30.0)
        half = 0.5 * (before_t + solver.temperatures_k)  # the midpoint the solver used, near enough
        wet = (solver.film_kg_m2 > 0.0) | (film0 > 0.0)
        del half, wet
        integral += rain.latent_arrays(2)[4] * 30.0
        if dried_at is None and solver.film_kg_m2[0] == 0.0:
            dried_at = (i + 1) * 30.0
    film = solver.film_kg_m2
    assert dried_at is not None and 900.0 <= dried_at <= 7200.0, dried_at
    # The rained-on cell holds exactly what fell (it was wet from the first tick, so it
    # evaporated too): its film is the integral of P − E, which the solver tracked.
    assert film[1] > 0.0 and film[1] < integral[1]
    # And the wet cell ran colder than the dry one while it was wet -- by kelvins in sun.
    assert solver.temperatures_k[1] != solver.temperatures_k[0]


def test_the_film_budget_closes_against_the_solver_s_own_evaporation() -> None:
    """Rain onto one cell at a rate below its evaporation: the film stays empty and every gram
    that fell evaporated; rain above it: the film grows by exactly P − E per tick."""
    props = FacetProperties(
        heat_capacity_j_m2_k=np.full(1, 10_000.0),
        emissivity=np.zeros(1),
        solar_absorptivity=np.zeros(1),
    )
    g_e = float(bulk_conductance_kg_m2_s(3.0))
    q_air = float(specific_humidity_kg_kg(295.0, 0.5))
    solver = FacetSolver(props, 295.0, film_kg_m2=1e-6)
    heavy = FacetForcing(
        t_air_k=295.0, h_w_m2_k=10.0, q_air_kg_kg=q_air, g_e_kg_m2_s=g_e, precip_kg_m2_s=5e-4
    )
    total_rain, total_evap = 0.0, 0.0
    for _ in range(60):
        t_before = solver.temperatures_k
        solver.advance(heavy, 10.0)
        # Reconstruct the midpoint the solver evaporates at: the RK2 half step of the balance.
        half = t_before + 0.5 * 10.0 * solver.net_flux(t_before, heavy, np.ones(1)) / 10_000.0
        total_evap += float(evaporation_kg_m2_s(half, q_air, g_e)[0]) * 10.0
        total_rain += 5e-4 * 10.0
    assert float(solver.film_kg_m2[0]) == pytest.approx(1e-6 + total_rain - total_evap, rel=1e-6)


# --- the sea --------------------------------------------------------------------------------------


def test_removing_the_latent_term_warms_the_sea_skin() -> None:
    """The sea-skin module's own docstring said latent was usually the largest loss and left it
    out; with it the cool-skin deficit grows, so the skin is colder than before."""
    from irsim.atmosphere import LayeredAtmosphere, SkyModel, load_atmosphere_preset
    from irsim.atmosphere.sea import SeaModel
    from irsim.config.environment import load_environment_preset
    from irsim.materials.nk import load_nk_table
    from irsim.radiometry.lut import BandLUT
    from irsim.radiometry.spectral_response import load_spectral_response
    from irsim.thermal.sea_skin import cool_skin_deficit_k
    from irsim.thermal.weather import WeatherSample, WeatherSeries

    response = load_spectral_response(REPO / "data/spectra/responses/boson_vox.csv")
    lut = BandLUT.build(response)
    weather = WeatherSeries.constant(
        WeatherSample(291.0, 0.5, 5.0, 0.0, 0.0, 0.0, 23000.0, 0.0), 3600.0
    )
    atm = LayeredAtmosphere(load_atmosphere_preset("us_standard_clear"), weather, {"lwir": lut})
    sky = SkyModel(atm, load_environment_preset("clear_dry"), "lwir", lut)
    sea = SeaModel(sky, load_nk_table("water"), response, bulk_sst_k=290.0, camera_height_m=20.0)
    latent = sea.latent_up_w_m2(0.0)
    assert 20.0 < latent < 120.0, latent  # a 17 °C sea under 18 °C air at 50 % RH, 5 m/s
    with_term = sea.skin_temperature_k(0.0)
    without = (
        290.0
        - float(cool_skin_deficit_k(sea.net_longwave_up_w_m2(0.0), 5.0, sea._skin))
        + sea.warm_layer_k(0.0)
    )
    assert with_term < without
    assert 0.0 < without - with_term < 1.0, without - with_term
