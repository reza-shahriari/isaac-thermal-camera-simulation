"""PH.4 — a gas slab in radiance space: the volumetric emitter a surface pipeline lacked.

The tests are the row's own limits and the one phenomenon that makes the feature worth having:
κL → 0 returns the background exactly and κL → ∞ returns B_b(T_g); an opaque soot flame reads its
own temperature in MWIR and LWIR alike, while a CO₂/H₂O slab of the same temperature reads hot
in MWIR and nearly invisible in LWIR -- which no grey emissivity knob can produce, so a slab is
authored as (T_gas, p_CO₂, p_H₂O, f_soot, L) and never as an emissivity; a gas colder than what
lies behind it gives negative contrast; and at range the excess is attenuated exactly as MS.6's
point target is, per class.

The CO₂/H₂O tables here are **synthetic** -- shaped like a real one (MWIR ≫ LWIR) to test the
kernel, not numbers about any gas. `PH.5` generates the real ones offline from HITEMP.

docs/physics-model.md §2, §8.1, §8.3; spec issue S46; roadmap PH.4 (the ADR is PH.13's).
"""

from __future__ import annotations

import math
import pathlib

import numpy as np
import pytest

from irsim.atmosphere import Atmosphere, LayeredAtmosphere, load_atmosphere_preset
from irsim.pipeline.gas_slab import (
    GAS_T_MAX_K,
    GAS_T_MIN_K,
    GasBandTables,
    GasSlab,
    SpeciesAbsorption,
    gas_band_radiance,
    slab_excess_radiance,
    slab_optical_depth,
    slab_radiance,
    slab_transmittance,
    soot_band_kappa_per_m,
)
from irsim.radiometry.band_integration import band_radiance
from irsim.radiometry.constants import SOOT_RAYLEIGH_C0
from irsim.radiometry.lut import BandLUT
from irsim.radiometry.spectral_response import SpectralResponse, load_spectral_response
from irsim.thermal import WeatherSample, WeatherSeries

REPO = pathlib.Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def lwir(tmp_path_factory: pytest.TempPathFactory) -> SpectralResponse:
    p = tmp_path_factory.mktemp("resp") / "lwir.csv"
    p.write_text("# exact top-hat\n7.5,1.0\n13.5,1.0\n")
    return load_spectral_response(p)


@pytest.fixture(scope="module")
def mwir(tmp_path_factory: pytest.TempPathFactory) -> SpectralResponse:
    p = tmp_path_factory.mktemp("resp") / "mwir.csv"
    p.write_text("# exact top-hat\n3.0,1.0\n5.0,1.0\n")
    return load_spectral_response(p)


def _tables(band: str, kappa_co2: float, kappa_h2o: float) -> GasBandTables:
    """A synthetic, temperature-flat table: the kernel is under test, not the gas."""
    grid = np.array([300.0, 2500.0])
    return GasBandTables(
        band,
        {
            "co2": SpeciesAbsorption(grid, np.full(2, kappa_co2)),
            "h2o": SpeciesAbsorption(grid, np.full(2, kappa_h2o)),
        },
    )


def _apparent_k(response: SpectralResponse, radiance: float) -> float:
    """Invert B_b(T) by bisection over the guard range -- the oracle, not a LUT."""
    lo, hi = 200.0, 3000.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if float(band_radiance(response, mid)[0]) < radiance:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# --- the limits ---------------------------------------------------------------------------------


def test_an_empty_path_returns_the_background_exactly(lwir) -> None:  # type: ignore[no-untyped-def]
    slab = GasSlab(t_gas_k=1200.0, length_m=1.0, f_soot=1e-30)  # κL ~ 1e-24: τ is exactly 1.0
    background = np.array([1.5, 20.0, 33.3])
    assert slab_transmittance(slab, lwir) == 1.0
    assert np.array_equal(slab_radiance(slab, lwir, background), background)


def test_an_opaque_flame_returns_the_gas_s_own_planck_radiance(lwir, mwir) -> None:  # type: ignore[no-untyped-def]
    slab = GasSlab(t_gas_k=1200.0, length_m=1.0, f_soot=1e-3)  # κL ~ 1e3: τ underflows to 0
    for response in (lwir, mwir):
        assert slab_transmittance(slab, response) == 0.0
        expected = float(band_radiance(response, 1200.0)[0])
        assert float(slab_radiance(slab, response, 0.0)) == expected
        assert float(slab_radiance(slab, response, 1e4)) == expected
        assert gas_band_radiance(response, 1200.0) == expected


def test_a_flame_is_hotter_than_the_lut_can_say_and_the_quadrature_agrees_with_it_below(
    lwir,
) -> None:  # type: ignore[no-untyped-def]
    lut = BandLUT.build(lwir)
    assert lut.out_of_range(np.float64(1200.0))
    assert gas_band_radiance(lwir, 800.0) == pytest.approx(
        float(lut.lookup(np.float64(800.0))[()]), rel=2e-6
    )
    assert gas_band_radiance(lwir, 1500.0) > gas_band_radiance(lwir, 800.0)


# --- the phenomenon: bands are not interchangeable, and an emissivity knob cannot say so ------


def test_soot_is_greyish_and_an_opaque_soot_flame_reads_its_temperature_in_both_bands(
    lwir, mwir
) -> None:  # type: ignore[no-untyped-def]
    """κ_soot ∝ 1/λ, so the band means differ by about the wavelength ratio; once opaque, both
    cameras read T_g to a millikelvin."""
    k_l = soot_band_kappa_per_m(lwir, 1e-6, 1300.0)
    k_m = soot_band_kappa_per_m(mwir, 1e-6, 1300.0)
    assert 2.0 < k_m / k_l < 3.0  # ~10 µm against ~4 µm
    # The band mean of 1/λ is bracketed by the band's edges.
    assert SOOT_RAYLEIGH_C0 * 1e-6 / 13.5e-6 < k_l < SOOT_RAYLEIGH_C0 * 1e-6 / 7.5e-6
    assert SOOT_RAYLEIGH_C0 * 1e-6 / 5.0e-6 < k_m < SOOT_RAYLEIGH_C0 * 1e-6 / 3.0e-6

    thick = GasSlab(t_gas_k=1300.0, length_m=2.0, f_soot=3e-5)
    for response in (lwir, mwir):
        assert slab_optical_depth(thick, response) > 15.0
        t_apparent = _apparent_k(response, float(slab_radiance(thick, response, 0.0)))
        assert abs(t_apparent - 1300.0) < 1e-3


def test_a_co2_h2o_slab_reads_hot_in_mwir_and_cold_in_lwir_and_a_grey_knob_cannot(
    lwir, mwir
) -> None:  # type: ignore[no-untyped-def]
    """The same gas, the same temperature, two cameras: 900 K of apparent-temperature difference.

    A grey emissivity applied to both bands reads within a few kelvin of the same temperature in
    both -- which is why a slab is authored by its species and never by an emissivity.
    """
    background_k = 290.0
    slab = GasSlab(t_gas_k=1400.0, length_m=1.0, p_co2_atm=0.1, p_h2o_atm=0.1)
    tables = {"lwir": _tables("lwir", 0.05, 0.05), "mwir": _tables("mwir", 20.0, 10.0)}
    apparent = {}
    for name, response in (("lwir", lwir), ("mwir", mwir)):
        behind = float(band_radiance(response, background_k)[0])
        radiance = float(slab_radiance(slab, response, behind, tables[name]))
        apparent[name] = _apparent_k(response, radiance)
    assert apparent["mwir"] > 1300.0, apparent
    # 1 % band emissivity of a 1400 K gas still reads tens of kelvin over a 290 K background.
    assert apparent["lwir"] < 400.0, apparent
    assert apparent["mwir"] - apparent["lwir"] > 900.0

    # The negative control: one emissivity for both bands.
    grey = {}
    for name, response in (("lwir", lwir), ("mwir", mwir)):
        behind = float(band_radiance(response, background_k)[0])
        radiance = 0.6 * float(band_radiance(response, 1400.0)[0]) + 0.4 * behind
        grey[name] = _apparent_k(response, radiance)
    assert abs(grey["mwir"] - grey["lwir"]) < 150.0, grey


def test_a_gas_colder_than_what_lies_behind_it_gives_negative_contrast(mwir) -> None:  # type: ignore[no-untyped-def]
    """Steam in front of a hot wall: the slab absorbs more than it emits."""
    slab = GasSlab(t_gas_k=400.0, length_m=1.0, p_h2o_atm=0.5)
    tables = _tables("mwir", 0.0, 3.0)
    behind = float(band_radiance(mwir, 600.0)[0])
    assert float(slab_radiance(slab, mwir, behind, tables)) < behind
    assert (
        slab_excess_radiance(
            slab, mwir, None, "mwir", 0.0, BandLUT.build(mwir), 0.0, tables=tables, behind=behind
        )
        < 0.0
    )


# --- at range -----------------------------------------------------------------------------------


def _weather(t_air: float = 288.15) -> WeatherSeries:
    return WeatherSeries.constant(
        WeatherSample(t_air, 0.3, 1.0, 0.0, 0.0, 0.0, 23000.0, 0.0), 3600.0
    )


def test_the_grey_atmosphere_attenuates_the_excess_from_the_slab_s_range(mwir) -> None:  # type: ignore[no-untyped-def]
    """ΔL = τ(R) · [L_slab − L_air], with L_slab built on L_air behind the slab."""
    lut = BandLUT.build(mwir)
    atmosphere = Atmosphere(load_atmosphere_preset("us_standard_clear"), _weather(), {"mwir": lut})
    slab = GasSlab(t_gas_k=1000.0, length_m=0.5, f_soot=5e-7)
    for range_m in (0.0, 200.0, 2000.0):
        got = slab_excess_radiance(slab, mwir, atmosphere, "mwir", 0.0, lut, range_m)
        state = atmosphere.state(0.0)
        tau = math.exp(-state.gamma_per_m["mwir"] * range_m)
        l_air = float(lut.lookup(np.float64(state.t_air_k))[()])
        l_slab = float(slab_radiance(slab, mwir, l_air))
        assert got == pytest.approx(tau * (l_slab - l_air), rel=1e-12)
    at_0 = slab_excess_radiance(slab, mwir, atmosphere, "mwir", 0.0, lut, 0.0)
    at_2k = slab_excess_radiance(slab, mwir, atmosphere, "mwir", 0.0, lut, 2000.0)
    assert 0.0 < at_2k < at_0


def test_the_layered_atmosphere_attenuates_per_class_like_the_point_target(lwir) -> None:  # type: ignore[no-untyped-def]
    lut = BandLUT.build(lwir)
    layered = LayeredAtmosphere(
        load_atmosphere_preset("us_standard_clear"), _weather(), {"lwir": lut}
    )
    slab = GasSlab(t_gas_k=900.0, length_m=1.0, f_soot=2e-6)
    range_m, elevation = 1500.0, 0.05
    got = slab_excess_radiance(slab, lwir, layered, "lwir", 0.0, lut, range_m, elevation)
    es = layered.exponential_sum("lwir", 0.0)
    tau_k = layered.class_transmittances("lwir", 0.0, range_m, elevation)
    beyond = layered.sky_beyond_per_class("lwir", 0.0, range_m, elevation, "lb")
    emitted = 1.0 - slab_transmittance(slab, lwir)
    b_gas = gas_band_radiance(lwir, 900.0)
    expected = float(np.dot(es.weights * tau_k, emitted * (b_gas - beyond)))
    assert got == pytest.approx(expected, rel=1e-12)
    # Something opaque behind the slab at its own range replaces the sky beyond, per class.
    wall = float(lut.lookup(np.float64(320.0))[()])
    with_wall = slab_excess_radiance(
        slab, lwir, layered, "lwir", 0.0, lut, range_m, elevation, behind=wall
    )
    assert with_wall == pytest.approx(
        float(np.sum(es.weights * tau_k)) * emitted * (b_gas - wall), rel=1e-12
    )


def test_the_photon_quantity_mixes_photon_radiances(mwir) -> None:  # type: ignore[no-untyped-def]
    slab = GasSlab(t_gas_k=1100.0, length_m=1.0, f_soot=1e-6)
    tau = slab_transmittance(slab, mwir)
    b_q = gas_band_radiance(mwir, 1100.0, "lb_q")
    assert b_q > 1e15  # photons, not watts
    assert float(slab_radiance(slab, mwir, 0.0, quantity="lb_q")) == pytest.approx(
        (1.0 - tau) * b_q
    )


# --- the guards ---------------------------------------------------------------------------------


def test_the_slab_is_authored_by_species_and_guarded() -> None:
    with pytest.raises(ValueError, match="outside"):
        GasSlab(t_gas_k=GAS_T_MIN_K - 1.0, length_m=1.0, f_soot=1e-6)
    with pytest.raises(ValueError, match="outside"):
        GasSlab(t_gas_k=GAS_T_MAX_K + 1.0, length_m=1.0, f_soot=1e-6)
    with pytest.raises(ValueError, match="empty air"):
        GasSlab(t_gas_k=1000.0, length_m=1.0)
    with pytest.raises(ValueError, match="negative"):
        GasSlab(t_gas_k=1000.0, length_m=1.0, p_co2_atm=-0.1)
    with pytest.raises(ValueError, match="1 atm"):
        GasSlab(t_gas_k=1000.0, length_m=1.0, p_h2o_atm=1.5)
    with pytest.raises(ValueError, match="path length"):
        GasSlab(t_gas_k=1000.0, length_m=0.0, f_soot=1e-6)
    with pytest.raises(TypeError):
        GasSlab(t_gas_k=1000.0, length_m=1.0, f_soot=1e-6, emissivity=0.5)  # type: ignore[call-arg]


def test_a_species_without_a_table_names_the_step_that_makes_it(mwir) -> None:  # type: ignore[no-untyped-def]
    slab = GasSlab(t_gas_k=1000.0, length_m=1.0, p_co2_atm=0.1)
    with pytest.raises(ValueError, match="PH.5"):
        slab_transmittance(slab, mwir)
    only_h2o = GasBandTables("mwir", {"h2o": _tables("mwir", 0.0, 1.0).species["h2o"]})
    with pytest.raises(ValueError, match="CO2.*PH.5"):
        slab_transmittance(slab, mwir, only_h2o)


def test_tables_do_not_extrapolate_and_refuse_float16() -> None:
    table = SpeciesAbsorption(np.array([500.0, 1500.0]), np.array([1.0, 3.0]))
    assert table.at(1000.0) == 2.0
    with pytest.raises(ValueError, match="not extrapolated"):
        table.at(2000.0)
    with pytest.raises(TypeError, match="float16"):
        SpeciesAbsorption(np.array([500.0, 1500.0]), np.array([1.0, 3.0], dtype=np.float16))
    with pytest.raises(ValueError, match="increasing"):
        SpeciesAbsorption(np.array([1500.0, 500.0]), np.array([1.0, 3.0]))
    with pytest.raises(ValueError, match="negative"):
        SpeciesAbsorption(np.array([500.0, 1500.0]), np.array([-1.0, 3.0]))
