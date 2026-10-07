"""AT.27 -- the LWIR water-vapour term grows with w², not only w.

§7.3's γ = γ₀ + β w is linear in absolute humidity; the 8–12 µm self-broadened continuum goes
as the vapour's density times its own partial pressure, w² at a given temperature. LWIR now
carries a β₂ w² term sized from the MT_CKD continuum magnitude, with β₁ refit so the clear-dry
anchor (us_standard_clear at 288 K, RH 0.46) is unchanged; the humid rows still land in the
§7.2 band, and every air wetter than the anchor absorbs more than the linear law said, by a
margin that grows with w. Spec issue S57; ADR 0160; roadmap AT.27.

AT.37 (ADR 0200) refit β₂ against ARM's imager and gave the square its own column: w(h)² falls
at half the water's scale height, so a vertical path through the square holds half the linear
term's column, which the last test pins on the layered model itself.
"""

from __future__ import annotations

import math

import pytest

from irsim.atmosphere.humidity import absolute_humidity_g_m3, gamma_molecular
from irsim.atmosphere.library import load_atmosphere_preset

BETA_OLD = 4.61e-5
W_DRY = absolute_humidity_g_m3(288.15, 0.46)  # 5.90 g/m3, the clear-dry anchor
W_HUMID = absolute_humidity_g_m3(303.15, 0.80)  # 24.3 g/m3, the §7.2 humid row's own condition
W_TROPICAL = absolute_humidity_g_m3(300.0, 0.74)  # 19 g/m3, the AFGL tropical sea-level value


def _lwir() -> tuple[float, float, float]:
    c = load_atmosphere_preset("us_standard_clear").bands["lwir"]
    return c.gamma0_per_m, c.beta_per_m_per_g_m3, c.beta2_per_m_per_g2_m6


def _tau_200m(w: float) -> float:
    g0, b1, b2 = _lwir()
    return math.exp(-200.0 * gamma_molecular(w, g0, b1, b2))


def test_the_law_is_quadratic_and_beta2_zero_is_the_old_law_bit_for_bit() -> None:
    g0, b1 = 1.4e-4, 4.4e-5
    for w in (0.0, 3.0, 12.5, 30.0):
        assert gamma_molecular(w, g0, b1) == gamma_molecular(w, g0, b1, 0.0) == g0 + b1 * w
    b2 = 1e-6
    assert gamma_molecular(10.0, g0, b1, b2) == pytest.approx(
        g0 + b1 * 10.0 + b2 * 100.0, rel=1e-15
    )
    dw = 1e-3
    fd = (gamma_molecular(10.0 + dw, g0, b1, b2) - gamma_molecular(10.0 - dw, g0, b1, b2)) / (
        2 * dw
    )
    assert fd == pytest.approx(b1 + 2 * b2 * 10.0, rel=1e-8), "the slope grows with w"
    with pytest.raises(ValueError, match="beta2"):
        gamma_molecular(1.0, g0, b1, -1e-9)


def test_lwir_carries_the_continuum_and_the_other_bands_do_not() -> None:
    for name in ("us_standard_clear", "tropical", "midlat_summer_humid", "haze", "fog_dense_50m"):
        preset = load_atmosphere_preset(name)
        assert preset.bands["lwir"].beta2_per_m_per_g2_m6 > 0.0, name
        for band in ("mwir", "swir", "nir", "visible"):
            assert preset.bands[band].beta2_per_m_per_g2_m6 == 0.0, (name, band)


def test_the_clear_dry_anchor_is_the_number_it_was_and_humid_air_stays_in_band() -> None:
    g0, b1, b2 = _lwir()
    old = g0 + BETA_OLD * W_DRY
    assert gamma_molecular(W_DRY, g0, b1, b2) == pytest.approx(old, rel=2e-3), "refit to 3 digits"
    # the §7.2 humid row at its own condition, and the tropical check, both land in 0.72–0.85
    assert 0.72 <= _tau_200m(W_HUMID) <= 0.85, _tau_200m(W_HUMID)
    assert 0.72 <= _tau_200m(W_TROPICAL) <= 0.85, _tau_200m(W_TROPICAL)
    assert 0.90 <= _tau_200m(W_DRY) <= 0.96


def test_above_the_anchor_the_square_absorbs_more_and_the_excess_grows_with_w() -> None:
    """Anchored at the clear-dry row, the quadratic sits above the line for every wetter air:
    γ_quad − γ_lin = β₂ w (w − w_dry), zero at the anchor and growing with w. Mid-latitude summer
    air (12 g/m³) is 0.4 % murkier over 200 m than the line said, the humid row 2.6 %, a 35 g/m³
    monsoon afternoon 5.9 % -- the under-absorption S57 named, at the size AT.37's fit to ARM's
    imager allows (β₂ = 3.0e-7, a third of AT.27's MT_CKD-sized 8.6e-7; ADR 0200)."""
    g0, b1, b2 = _lwir()

    def linear(w: float) -> float:
        return g0 + BETA_OLD * w

    def quad(w: float) -> float:
        return gamma_molecular(w, g0, b1, b2)

    excess = [quad(w) - linear(w) for w in (W_DRY, 12.0, W_HUMID, 35.0)]
    assert abs(excess[0]) < 1e-7, "the anchor itself"
    assert 0.0 < excess[1] < excess[2] < excess[3]
    assert excess[3] == pytest.approx(b2 * 35.0 * (35.0 - W_DRY), rel=0.05)
    assert 0.99 < _tau_200m(12.0) / math.exp(-200.0 * linear(12.0)) < 1.0
    # the continuum's share of the water term at the humid row: a real fraction, not a nudge
    share = b2 * W_HUMID**2 / (quad(W_HUMID) - g0)
    assert 0.10 < share < 0.2, share


@pytest.mark.filterwarnings("ignore:LayeredAtmosphere has no spectral response")
def test_the_square_s_column_is_half_the_linear_part_s() -> None:
    """AT.37. The water classes carry γ_lin e^{−h/H_w} + γ_sq e^{−2h/H_w} up the column; the
    layered model keeps one exponential per class whose surface value is the sum and whose
    height holds the column exact. A vertical ray's water-class optical depth therefore equals
    ``scale · (γ_lin H_w + γ_sq H_w / 2)`` to rounding; a band whose β₂ is 0 keeps H_w itself."""
    import numpy as np

    from irsim.atmosphere.layered import LayeredAtmosphere, classes_for
    from irsim.thermal.weather import WeatherSample, WeatherSeries

    preset = load_atmosphere_preset("us_standard_clear")
    weather = WeatherSeries.constant(
        WeatherSample(303.15, 0.80, 1.0, 0.0, 0.0, 0.0, 23000.0, 0.0), 3600.0
    )
    atm = LayeredAtmosphere(preset, weather, {}, {})
    es = atm.exponential_sum("lwir", 0.0)
    water = np.array([c.kind == "water" for c in classes_for("lwir")])
    assert water.any()
    g0, b1, b2 = _lwir()
    gamma_lin = g0 + b1 * W_HUMID
    gamma_sq = b2 * W_HUMID**2
    h_w = preset.profile.water_vapour_scale_height_m
    column = es.optical_depths(np.inf, math.pi / 2.0)[water] - es.gamma_aerosol * (
        preset.profile.aerosol_scale_height_m
    )
    scale = es.gamma_0[water] / (gamma_lin + gamma_sq)
    expected = scale * (gamma_lin * h_w + 0.5 * gamma_sq * h_w)
    assert np.allclose(column, expected, rtol=1e-9)
    # half the square's column is a real change at the humid row, not a rounding
    assert float((expected / (scale * (gamma_lin + gamma_sq) * h_w)).max()) < 0.95

    mwir = atm.exponential_sum("mwir", 0.0)
    mwir_water = np.array([c.kind == "water" for c in classes_for("mwir")])
    assert np.all(mwir.scale_heights_m[mwir_water] == h_w)  # β₂ = 0: unchanged, bit for bit
