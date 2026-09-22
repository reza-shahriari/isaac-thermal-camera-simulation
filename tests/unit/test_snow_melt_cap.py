"""Snow: the surface that cannot get warmer than freezing, and where the surplus goes (PH.10).

The headline, measured: a snow cell under **+200 W/m² of net gain stays at exactly 273.15 K** --
bit-exact, for as long as there is snow -- and sheds **2.1557 mm of water equivalent per hour**.
The same balance on any other material would put the surface tens of kelvin above air.

The second phenomenology, also measured: with snow's ε_hemi of 0.9874, a **clear calm night drives
the surface 12.73 K below air**, a breezy one only 4.75 K, and **overcast exactly 0.00 K** -- full
overcast is a blackbody at air temperature, so the radiative term vanishes identically and the
surface sits on the air. No special case produces that; it falls out of the ordinary balance with
the library's emissivity.

⚠️ **The interesting test here is `test_the_flux_is_read_at_the_melt_point_not_at_a_midpoint`.**
The obvious implementation clamps after an RK2 step, which evaluates emission and convection at a
temperature the surface never reaches and under-reports the melt by a quarter of a percent every
step -- small, one-signed, and cumulative over a season. That test measures the gap rather than
trusting the argument.

docs/physics-model.md §6.1; roadmap PH.10; ADR 0120
"""

from __future__ import annotations

import numpy as np
import pytest

from irsim.materials.hemispherical import total_hemispherical_emissivity
from irsim.materials.library import MaterialLibrary
from irsim.radiometry.constants import L_F_WATER_J_KG, SIGMA_SB
from irsim.thermal.balance import SurfaceForcing, ThermalProperties, net_flux, rk2_step
from irsim.thermal.longwave import longwave_down_from_sample
from irsim.thermal.snow import (
    ALPINE_TIER4_MAE_K,
    T_MELT_K,
    melt_capped_step,
    melt_rate_kg_m2_s,
    water_equivalent_mm,
)
from irsim.thermal.weather import WeatherSample

pytestmark = pytest.mark.slow

#: A snowpack's areal heat capacity, from the library material: 300 kg/m³ × 2100 J/kgK × 0.10 m.
SNOWPACK_C_J_M2_K = 63000.0
T_AIR_NIGHT_K = 270.0


def _snow_properties(emissivity: float = 0.99) -> ThermalProperties:
    return ThermalProperties(
        heat_capacity_j_m2_k=SNOWPACK_C_J_M2_K,
        emissivity=emissivity,
        solar_absorptivity=0.15,
    )


def _forcing_with_net_at_cap(
    target_w_m2: float, properties: ThermalProperties, t_air_k: float = 275.0, h: float = 5.0
) -> SurfaceForcing:
    """A forcing whose net flux at exactly the melt point is ``target_w_m2``.

    Solved rather than tuned, so the acceptance number is a statement about the melt arithmetic
    and not about how well a hand-picked longwave happened to land.
    """
    emitted = properties.emissivity * SIGMA_SB * T_MELT_K**4
    convected = h * (T_MELT_K - t_air_k)
    lw_down = (target_w_m2 + emitted + convected) / properties.emissivity
    return SurfaceForcing(
        t_air_k=t_air_k, h_w_m2_k=h, q_solar_w_m2=0.0, q_longwave_down_w_m2=lw_down
    )


# -- the cap -----------------------------------------------------------------------------------


def test_two_hundred_watts_holds_the_surface_at_freezing_and_melts_2_2_mm_an_hour() -> None:
    """PH.10's first acceptance criterion, end to end over an hour of one-minute steps."""
    props = _snow_properties()
    forcing = _forcing_with_net_at_cap(200.0, props)
    assert float(net_flux(T_MELT_K, props, forcing)) == pytest.approx(200.0, abs=1e-9)

    t = np.array(T_MELT_K)
    total_kg = 0.0
    for _ in range(60):
        step = melt_capped_step(t, 60.0, props, forcing)
        t = step.temperature_k
        total_kg += float(step.melt_kg_m2)

    assert float(t) == T_MELT_K, "the cap is exact, not approximate"
    assert float(water_equivalent_mm(total_kg)) == pytest.approx(2.156, abs=0.01)
    # and it is exactly the flux over the latent heat, not a fitted number
    assert total_kg == pytest.approx(200.0 * 3600.0 / L_F_WATER_J_KG, rel=1e-12)


def test_the_flux_is_read_at_the_melt_point_not_at_a_midpoint() -> None:
    """The negative control: clamping after an RK2 step under-reports the melt, one-signed.

    RK2 evaluates the balance at a midpoint half a step above the cap. That temperature does not
    exist -- the surface is melting and isothermal -- and emission and convection there are both
    larger, so the surplus comes out too small. The error is tiny per step and always in the same
    direction, which is the kind that survives review and shows up as a season of missing melt.
    """
    props = _snow_properties()
    forcing = _forcing_with_net_at_cap(200.0, props)
    dt = 60.0

    correct = float(melt_capped_step(np.array(T_MELT_K), dt, props, forcing).melt_kg_m2)

    # what a clamp-after-step implementation would have produced
    stepped = float(rk2_step(np.array(T_MELT_K), dt, props, forcing))
    naive = props.heat_capacity_j_m2_k * (stepped - T_MELT_K) / L_F_WATER_J_KG

    assert naive < correct, "the naive form is biased low, which is the point"
    shortfall = (correct - naive) / correct
    assert 1e-3 < shortfall < 1e-2, shortfall
    # the exact answer is the flux at the cap; the naive one is the flux at the midpoint
    assert correct == pytest.approx(200.0 * dt / L_F_WATER_J_KG, rel=1e-12)


def test_a_step_that_carries_a_cell_through_the_cap_conserves_energy() -> None:
    """A cell below freezing that warms past it splits its step: part sensible, part latent.

    Discarding the overshoot instead would lose energy silently at every thaw, and a scene that
    crossed freezing twice a day would lose it twice a day.
    """
    props = _snow_properties()
    forcing = _forcing_with_net_at_cap(200.0, props)
    start = T_MELT_K - 0.05  # close enough that one minute carries it through

    uncapped = float(rk2_step(np.array(start), 60.0, props, forcing))
    assert uncapped > T_MELT_K, "the fixture must actually cross the cap"

    step = melt_capped_step(np.array(start), 60.0, props, forcing)
    assert float(step.temperature_k) == T_MELT_K
    latent_j = float(step.melt_kg_m2) * L_F_WATER_J_KG
    sensible_j = props.heat_capacity_j_m2_k * (T_MELT_K - start)
    total_j = props.heat_capacity_j_m2_k * (uncapped - start)
    assert latent_j + sensible_j == pytest.approx(total_j, rel=1e-12)


def test_below_freezing_the_cap_changes_nothing() -> None:
    """A cold snow surface must step exactly as any other surface does -- bit for bit."""
    props = _snow_properties()
    forcing = SurfaceForcing(t_air_k=260.0, h_w_m2_k=5.0, q_longwave_down_w_m2=200.0)
    for start in (240.0, 255.0, 272.0):
        step = melt_capped_step(np.array(start), 60.0, props, forcing)
        assert float(step.temperature_k) == float(rk2_step(np.array(start), 60.0, props, forcing))
        assert float(step.melt_kg_m2) == 0.0
        assert not bool(step.melted)


def test_refreezing_reports_no_melt_rather_than_negative_melt() -> None:
    """A negative surplus would otherwise create snow out of a cooling surface."""
    assert float(melt_rate_kg_m2_s(-500.0)) == 0.0
    assert float(melt_rate_kg_m2_s(0.0)) == 0.0
    assert float(melt_rate_kg_m2_s(334.0)) == pytest.approx(1e-3, rel=1e-12)

    props = _snow_properties()
    cooling = _forcing_with_net_at_cap(-50.0, props)
    step = melt_capped_step(np.array(T_MELT_K), 60.0, props, cooling)
    assert float(step.melt_kg_m2) == 0.0
    assert float(step.temperature_k) < T_MELT_K, "it cools off the cap instead"


def test_melt_stops_when_the_snow_runs_out_and_the_surplus_then_warms_the_ground() -> None:
    """A finite pack: what cannot melt has to go somewhere, and it goes into temperature.

    Without this the last millimetre of snow would hold a surface at freezing forever, which is
    exactly the artefact a melt cap is supposed to avoid producing.
    """
    props = _snow_properties()
    forcing = _forcing_with_net_at_cap(200.0, props)
    available = 0.001  # kg/m², about a micron of water equivalent

    step = melt_capped_step(np.array(T_MELT_K), 60.0, props, forcing, available_kg_m2=available)
    assert float(step.melt_kg_m2) == pytest.approx(available)
    assert float(step.temperature_k) > T_MELT_K, "bare ground is free to warm"

    unlimited = melt_capped_step(np.array(T_MELT_K), 60.0, props, forcing)
    leftover_j = (float(unlimited.melt_kg_m2) - available) * L_F_WATER_J_KG
    assert float(step.temperature_k) - T_MELT_K == pytest.approx(
        leftover_j / props.heat_capacity_j_m2_k, rel=1e-12
    )
    with pytest.raises(ValueError, match="negative"):
        melt_capped_step(np.array(T_MELT_K), 60.0, props, forcing, available_kg_m2=-1.0)


def test_a_field_of_cells_takes_both_branches_at_once() -> None:
    """Vectorised: cold cells step, capped cells melt, and neither leaks into the other."""
    props = _snow_properties()
    forcing = _forcing_with_net_at_cap(200.0, props)
    start = np.array([250.0, 265.0, T_MELT_K, T_MELT_K])

    step = melt_capped_step(start, 60.0, props, forcing)
    assert step.temperature_k.shape == start.shape
    assert np.all(step.temperature_k <= T_MELT_K + 1e-12)
    assert list(map(bool, step.melted)) == [False, False, True, True]
    # the two cold cells stepped exactly as the bare solver would
    bare = np.asarray(rk2_step(start, 60.0, props, forcing))
    assert np.allclose(step.temperature_k[:2], bare[:2], rtol=0, atol=0)


# -- the night, which needs no special case ------------------------------------------------------


def test_a_clear_calm_night_drives_snow_below_air_and_overcast_does_not() -> None:
    """PH.10's second criterion: 12.73 K of depression clear and calm, 0.00 K overcast.

    The overcast figure is exactly zero and that is not a coincidence or a tolerance -- a fully
    overcast sky radiates as a blackbody at air temperature, so absorbed and emitted longwave
    cancel identically at T_s = T_air and the surface has nowhere else to go. Wind then only
    matters when there *is* a radiative imbalance to mix away, which the breezy clear case shows.
    """
    library = MaterialLibrary.load()
    eps = total_hemispherical_emissivity(library["snow"], T_AIR_NIGHT_K).value
    props = _snow_properties(eps)

    results = {}
    for label, cloud, wind in (
        ("clear calm", 0.0, 0.5),
        ("clear breezy", 0.0, 4.0),
        ("overcast calm", 1.0, 0.5),
    ):
        sample = WeatherSample(T_AIR_NIGHT_K, 0.75, wind, cloud, 0.0, 0.0, 23000.0, 0.0)
        forcing = SurfaceForcing(
            t_air_k=T_AIR_NIGHT_K,
            h_w_m2_k=2.0 + 4.0 * wind,
            q_longwave_down_w_m2=float(longwave_down_from_sample(sample, 1.0, T_AIR_NIGHT_K)),
        )
        t = np.array(T_AIR_NIGHT_K)
        for _ in range(8 * 60):  # eight hours of night
            t = melt_capped_step(t, 60.0, props, forcing).temperature_k
        results[label] = float(t) - T_AIR_NIGHT_K

    assert results["clear calm"] == pytest.approx(-12.73, abs=0.3)
    assert results["clear breezy"] == pytest.approx(-4.75, abs=0.3)
    assert results["overcast calm"] == pytest.approx(0.0, abs=0.01)
    # ordering, stated as the relation so it survives a change in any one number
    assert results["clear calm"] < results["clear breezy"] < results["overcast calm"]
    # the overcast case is not merely smaller, it is radiatively closed: absorbed and emitted
    # longwave cancel at T_s = T_air, so there is no imbalance for wind or capacity to act on
    assert abs(results["overcast calm"]) < 1e-6


def test_the_library_snow_emissivity_sits_in_the_band_the_step_asked_for() -> None:
    """PH.10 asked for ε 0.98-0.99 with the library's angular fall-off. It is 0.9874.

    Checked hemispherically and not per-band, because the hemispherical value is what the balance
    uses -- and snow's angular model pulls it below the 0.99 the LWIR band declares, which is the
    fall-off the row named.
    """
    library = MaterialLibrary.load()
    snow = library["snow"]
    hemi = total_hemispherical_emissivity(snow, T_AIR_NIGHT_K)
    assert 0.98 <= hemi.value <= 0.99, hemi.value
    assert hemi.value == pytest.approx(0.9874, abs=1e-3)
    # the angular fall-off really is what puts it under the band value
    assert hemi.value < snow.spec.optical.emissivity_per_band["lwir"]


def test_the_alpine_tier4_band_is_recorded_and_not_claimed() -> None:
    """The ESSD 16 (2024) bar is written down; no check in this repo evaluates against it.

    Recorded the same way SE.1 records the sea's published envelope: a number a future Tier 4
    comparison must be judged against, stored now so it cannot be invented later to fit whatever
    the model happens to produce.
    """
    low, high = ALPINE_TIER4_MAE_K
    assert low == pytest.approx(0.7) and high == pytest.approx(1.3)
    assert low < high
