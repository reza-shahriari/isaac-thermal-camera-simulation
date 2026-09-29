"""Koschmieder aerosol extinction and the band crossovers (M8.4, ADR 0049): the visibility
identity, the clear-air limit, fog ordering, the humid/fog LWIR-SWIR crossover from weather
alone, and the exact humidity sensitivity."""

from __future__ import annotations

import math

import numpy as np
import pytest

from irsim.atmosphere.extinction import (
    FOG_VISIBILITY_M,
    extinction_per_band,
    gamma_aerosol,
    gamma_aerosol_visible,
    regime_for_visibility,
    transmittance_per_band,
)
from irsim.atmosphere.humidity import absolute_humidity_g_m3
from irsim.atmosphere.library import load_atmosphere_preset
from irsim.radiometry.constants import KOSCHMIEDER


@pytest.mark.parametrize("vis", [50.0, 200.0, 1500.0, 23000.0])
def test_koschmieder_identity(vis: float) -> None:
    """ratio = 1: the transmittance over one visibility length is the 2 % contrast threshold."""
    assert math.exp(-gamma_aerosol(vis, 1.0) * vis) == pytest.approx(0.02, rel=1e-12)
    assert pytest.approx(math.log(1 / 0.02), rel=1e-3) == KOSCHMIEDER


def test_clear_air_limit_and_rayleigh_ceiling() -> None:
    preset = load_atmosphere_preset("us_standard_clear")
    assert gamma_aerosol_visible(math.inf) == 0.0
    g_inf = extinction_per_band(preset, 288.15, 0.46, math.inf)
    w = absolute_humidity_g_m3(288.15, 0.46)
    for band, coeffs in preset.bands.items():
        # AT.27: LWIR carries a self-continuum square; every other band's beta2 is 0
        assert g_inf[band] == pytest.approx(
            coeffs.gamma0_per_m
            + coeffs.beta_per_m_per_g_m3 * w
            + coeffs.beta2_per_m_per_g2_m6 * w * w,
            rel=1e-12,
        )
    # MOR is defined on the total extinction: past the Rayleigh-limited visibility no aerosol
    ray = preset.bands["visible"].gamma0_per_m
    v_max = KOSCHMIEDER / ray
    assert 300e3 < v_max < 400e3
    assert gamma_aerosol_visible(v_max * 1.01, ray) == 0.0
    assert gamma_aerosol_visible(v_max * 0.5, ray) == pytest.approx(ray, rel=1e-12)
    with pytest.raises(ValueError):
        gamma_aerosol_visible(0.0)
    with pytest.raises(ValueError):
        gamma_aerosol(100.0, -0.1)


def test_fog_ordering_and_ranges_at_200m() -> None:
    fog = load_atmosphere_preset("fog_light_200m")
    tau = transmittance_per_band(fog, 283.15, 1.0, 200.0, 200.0)
    assert tau["lwir"] > tau["mwir"] > tau["swir"] > tau["nir"] > tau["visible"], tau
    assert 0.35 <= tau["lwir"] <= 0.60 and 0.02 <= tau["visible"] <= 0.10, tau
    dense = load_atmosphere_preset("fog_dense_50m")
    assert transmittance_per_band(dense, 283.15, 1.0, 50.0, 200.0)["visible"] < 1e-4


def test_humid_and_fog_crossover_from_weather_alone() -> None:
    """One preset, weather only: humid clear air hurts LWIR more than SWIR (water-vapour
    continuum); fog hurts SWIR more than LWIR. Reproducing this crossover is the point (§7.2)."""
    preset = load_atmosphere_preset("us_standard_clear")
    humid = transmittance_per_band(preset, 303.15, 0.80, 23000.0, 200.0)
    assert humid["lwir"] < humid["swir"], humid
    fog = transmittance_per_band(preset, 283.15, 1.0, 200.0, 200.0)
    assert fog["lwir"] > fog["swir"], fog
    assert (
        regime_for_visibility(200.0) == "droplet"
        and regime_for_visibility(FOG_VISIBILITY_M) == "aerosol"
    )


def test_humidity_step_changes_gamma_by_exactly_beta_dw_and_monotone() -> None:
    preset = load_atmosphere_preset("midlat_summer_humid")
    g1 = extinction_per_band(preset, 293.15, 0.30, 23000.0)
    g2 = extinction_per_band(preset, 303.15, 0.80, 23000.0)
    w1, w2 = absolute_humidity_g_m3(293.15, 0.30), absolute_humidity_g_m3(303.15, 0.80)
    lwir = preset.bands["lwir"]
    # AT.27: the step is beta1 dw + beta2 (w2^2 - w1^2), the linear law's beta dw at beta2 = 0
    expect = lwir.beta_per_m_per_g_m3 * (w2 - w1) + lwir.beta2_per_m_per_g2_m6 * (w2**2 - w1**2)
    assert g2["lwir"] - g1["lwir"] == pytest.approx(expect, rel=1e-12)
    assert lwir.beta2_per_m_per_g2_m6 > 0.0 and g2["lwir"] - g1[
        "lwir"
    ] > lwir.beta_per_m_per_g_m3 * (w2 - w1)
    # monotone in visibility and in distance
    vis = [50.0, 200.0, 1000.0, 5000.0, 23000.0, math.inf]
    taus = [transmittance_per_band(preset, 288.15, 0.5, v, 200.0)["lwir"] for v in vis]
    assert np.all(np.diff(taus) > 0.0), taus
    ds = [0.0, 10.0, 200.0, 5000.0]
    tau_d = [transmittance_per_band(preset, 288.15, 0.5, 23000.0, d)["mwir"] for d in ds]
    assert tau_d[0] == 1.0 and np.all(np.diff(tau_d) < 0.0)
