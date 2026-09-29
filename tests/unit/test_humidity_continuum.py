"""AT.27 -- the LWIR water-vapour term grows with w², not only w.

§7.3's γ = γ₀ + β w is linear in absolute humidity; the 8–12 µm self-broadened continuum goes
as the vapour's density times its own partial pressure, w² at a given temperature. LWIR now
carries a β₂ w² term sized from the MT_CKD continuum magnitude, with β₁ refit so the clear-dry
anchor (us_standard_clear at 288 K, RH 0.46) is unchanged; the humid rows still land in the
§7.2 band, and every air wetter than the anchor absorbs more than the linear law said, by a
margin that grows with w. Spec issue S57; ADR 0160; roadmap AT.27.
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
    air (12 g/m³) is 1 % murkier over 200 m than the line said, the humid row 7 %, a 35 g/m³
    monsoon afternoon 16 % -- which is the under-absorption S57 named."""
    g0, b1, b2 = _lwir()

    def linear(w: float) -> float:
        return g0 + BETA_OLD * w

    def quad(w: float) -> float:
        return gamma_molecular(w, g0, b1, b2)

    excess = [quad(w) - linear(w) for w in (W_DRY, 12.0, W_HUMID, 35.0)]
    assert abs(excess[0]) < 1e-7, "the anchor itself"
    assert 0.0 < excess[1] < excess[2] < excess[3]
    assert excess[3] == pytest.approx(b2 * 35.0 * (35.0 - W_DRY), rel=0.05)
    assert 0.98 < _tau_200m(12.0) / math.exp(-200.0 * linear(12.0)) < 1.0
    # the continuum's share of the water term at the humid row is a real fraction, not a nudge
    share = b2 * W_HUMID**2 / (quad(W_HUMID) - g0)
    assert 0.25 < share < 0.6, share
