"""The air is one air whatever camera looks through it (AT.35, ADR 0177).

`LayeredAtmosphere` splits a band into spectral classes (AT.10) and scales the non-opaque ones so
the band's τ(200 m) is the preset's grey value. That scale used to be solved per **camera**, so
every camera's τ(200 m) was forced to the preset's, and a camera with less absorbing edge in its
range bought a murkier window: an 8-12 µm camera read τ(5 km) 0.311 against 0.588 for 7.5-13.5 µm.
It is now solved once on the band's nominal top-hat and every camera in the band uses that γ.

The tests state the physics that must follow, not the numbers: a narrower camera that sheds
absorbing classes transmits *more*, one that adds them transmits less, and the nominal camera is
still anchored to the preset exactly.

docs/physics-model.md §7.1, §7.4; ADR 0113, ADR 0177
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from irsim.atmosphere.extinction import gamma_aerosol_visible, gamma_molecular
from irsim.atmosphere.layered import (
    ANCHOR_DISTANCE_M,
    LayeredAtmosphere,
    class_weights,
    classes_for,
)
from irsim.atmosphere.library import load_atmosphere_preset
from irsim.config.bands import ANCHOR_BAND
from irsim.radiometry.spectral_response import SpectralResponse
from irsim.thermal import WeatherSample, WeatherSeries

PRESETS = ("us_standard_clear", "midlat_summer_humid", "midlat_winter_dry", "tropical", "haze")
DISTANCES_M = np.array([200.0, 500.0, 1000.0, 2000.0, 5000.0, 10000.0])


def _top_hat(lo: float, hi: float) -> SpectralResponse:
    return SpectralResponse(np.array([lo, hi]), np.array([1.0, 1.0]), f"<{lo}-{hi}>", "")


def _weather(t_k: float = 288.15, rh: float = 0.5) -> WeatherSeries:
    return WeatherSeries.constant(WeatherSample(t_k, rh, 1.0, 0.0, 0.0, 0.0, 23000.0, 0.0), 3600.0)


def _tau(preset: str, lo: float, hi: float, weather: WeatherSeries | None = None) -> np.ndarray:
    atm = LayeredAtmosphere(
        load_atmosphere_preset(preset), weather or _weather(), None, {"lwir": _top_hat(lo, hi)}
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.asarray(atm.transmittance("lwir", 0.0, DISTANCES_M), dtype=np.float64)


@pytest.mark.parametrize("preset", PRESETS)
def test_a_window_only_camera_sees_further_than_the_nominal_band(preset: str) -> None:
    """8-12 µm sheds the 7.5-8 and 12-13.5 µm edges; it must transmit more at every range."""
    nominal = _tau(preset, 7.5, 13.5)
    window = _tau(preset, 8.0, 12.0)
    assert np.all(window >= nominal), (window, nominal)
    assert window[4] > nominal[4] + 0.02  # 5 km: a margin, not a rounding


@pytest.mark.parametrize("preset", PRESETS)
def test_a_camera_reaching_into_the_water_band_sees_less(preset: str) -> None:
    """6-13 µm adds the opaque 6.3 µm water band and more edge: lower τ at every range, and at
    200 m lower by at least the opaque class's share of the band (it is gone by 200 m)."""
    nominal = _tau(preset, 7.5, 13.5)
    wide = _tau(preset, 6.0, 13.0)
    assert np.all(wide < nominal)
    weights = class_weights("lwir", _top_hat(6.0, 13.0))
    names = [c.name for c in classes_for("lwir", _top_hat(6.0, 13.0))]
    opaque_share = float(weights[names.index("h2o_6p3")])
    assert opaque_share == pytest.approx(0.103, abs=5e-3)
    assert nominal[0] - wide[0] >= opaque_share * nominal[0] * 0.95


@pytest.mark.parametrize("preset", PRESETS)
def test_the_nominal_band_is_still_anchored_to_the_preset(preset: str) -> None:
    """The anchor the presets were written for: the nominal top-hat's free classes average to the
    preset's grey τ at 200 m, exactly as before AT.35."""
    weather = _weather()
    sample = weather.at(0.0)
    p = load_atmosphere_preset(preset)
    c = p.bands["lwir"]
    v = p.bands[ANCHOR_BAND]
    g_mol = gamma_molecular(
        sample.absolute_humidity_g_m3,
        c.gamma0_per_m,
        c.beta_per_m_per_g_m3,
        c.beta2_per_m_per_g2_m6,
    )
    g_vis = gamma_molecular(
        sample.absolute_humidity_g_m3,
        v.gamma0_per_m,
        v.beta_per_m_per_g_m3,
        v.beta2_per_m_per_g2_m6,
    )
    g_aer = c.aerosol_ratio_to_visible * gamma_aerosol_visible(sample.visibility_m, g_vis)
    grey = math.exp(-(g_mol + g_aer) * ANCHOR_DISTANCE_M)
    # The nominal LWIR band has no opaque class, so its whole τ(200 m) is the free classes'.
    assert _tau(preset, 7.5, 13.5, weather)[0] == pytest.approx(grey, rel=1e-9)


def test_the_scale_is_the_air_s_not_the_camera_s() -> None:
    """Two cameras share each class's γ: their exponential sums agree class by class."""
    weather = _weather()
    preset = load_atmosphere_preset("us_standard_clear")
    sums = {}
    for lo, hi in ((7.5, 13.5), (8.0, 12.0), (7.0, 14.0)):
        atm = LayeredAtmosphere(preset, weather, None, {"lwir": _top_hat(lo, hi)})
        es = atm.exponential_sum("lwir", 0.0)
        sums[(lo, hi)] = dict(zip(es.names, es.gamma_0, strict=True))
    for name in ("lines", "window"):
        values = {round(float(s[name]), 15) for s in sums.values() if name in s}
        assert len(values) == 1, (name, values)
