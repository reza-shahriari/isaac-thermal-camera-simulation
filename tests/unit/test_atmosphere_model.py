"""Atmosphere bound to the shared WeatherSeries (M8.5): object-only constructor, T_air identity
with the thermal solver's ambient, the exact humidity sensitivity, the humid/fog crossovers from
weather alone, air radiance and the per-pixel kernel, regime mismatch."""

from __future__ import annotations

import math
from datetime import datetime, timezone

import numpy as np
import pytest

from irsim.atmosphere import Atmosphere, load_atmosphere_preset
from irsim.atmosphere.humidity import absolute_humidity_g_m3
from irsim.radiometry.lut import BandLUT
from irsim.thermal import NewtonCoolingSolver, WeatherSample, WeatherSeries, synthetic_clear_day

EPOCH = datetime(2024, 6, 21, tzinfo=timezone.utc)


def _weather_step() -> WeatherSeries:
    """Hours 0-1: 20 C, 30 %, 23 km; hours 2-3: 30 C, 80 %, 23 km; hours 4-5: 10 C, 100 %, 200 m."""
    t = np.arange(6) * 3600.0
    return WeatherSeries.from_arrays(
        EPOCH,
        t,
        t_air_k=np.array([293.15, 293.15, 303.15, 303.15, 283.15, 283.15]),
        rh_fraction=np.array([0.30, 0.30, 0.80, 0.80, 1.0, 1.0]),
        wind_speed_m_s=np.full(6, 2.0),
        cloud_fraction=np.array([0.1, 0.1, 0.1, 0.1, 1.0, 1.0]),
        dni_w_m2=np.zeros(6),
        dhi_w_m2=np.zeros(6),
        visibility_m=np.array([23000.0, 23000.0, 23000.0, 23000.0, 200.0, 200.0]),
        precip_mm_h=np.zeros(6),
    )


def test_constructor_takes_the_object_only(tophat_lwir_lut: BandLUT) -> None:
    preset = load_atmosphere_preset("us_standard_clear")
    with pytest.raises(TypeError, match="never a path"):
        Atmosphere(preset, "/data/weather/clear.csv", {"lwir": tophat_lwir_lut})  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        Atmosphere("us_standard_clear", _weather_step())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="does not define"):
        Atmosphere(preset, _weather_step(), {"thz": tophat_lwir_lut})
    atm = Atmosphere(preset, _weather_step(), {"lwir": tophat_lwir_lut})
    assert atm.bands == ("lwir", "mwir", "nir", "swir", "visible") and atm.lut_bands == ("lwir",)


def test_t_air_identity_with_the_thermal_solver_at_every_hour(tophat_lwir_lut: BandLUT) -> None:
    """CLAUDE.md #6 as physics: the air the path radiates at is the air the object convects to."""
    w = synthetic_clear_day(EPOCH, hours=48.0, utc_offset_h=2.0)
    atm = Atmosphere(load_atmosphere_preset("midlat_summer_humid"), w, {"lwir": tophat_lwir_lut})
    solver = NewtonCoolingSolver(300.0, 600.0, w)
    assert atm.weather is solver.weather is w
    for h in range(0, 49):
        t = h * 3600.0
        st = atm.state(t)
        assert abs(st.t_air_k - w.at(t).t_air_k) < 1e-9
        assert abs(st.t_air_k - solver.ambient_at(t)) < 1e-9
        assert st.l_air["lwir"] == pytest.approx(
            float(tophat_lwir_lut.lookup(st.t_air_k)[()]), rel=1e-7
        )
    assert atm.air_radiance("lwir", 7 * 3600.0) == atm.state(7 * 3600.0).l_air["lwir"]
    with pytest.raises(KeyError, match="no LUT"):
        atm.air_radiance("mwir", 0.0)


def test_rh_step_changes_gamma_by_exactly_beta_dw() -> None:
    preset = load_atmosphere_preset("us_standard_clear")
    atm = Atmosphere(preset, _weather_step())
    g1, g2 = atm.gamma("lwir", 1800.0), atm.gamma("lwir", 2.5 * 3600.0)
    w1, w2 = absolute_humidity_g_m3(293.15, 0.30), absolute_humidity_g_m3(303.15, 0.80)
    lwir = preset.bands["lwir"]
    # AT.27: beta1 dw + beta2 (w2^2 - w1^2); the linear law's beta dw when beta2 = 0
    expect = lwir.beta_per_m_per_g_m3 * (w2 - w1) + lwir.beta2_per_m_per_g2_m6 * (w2**2 - w1**2)
    assert g2 - g1 == pytest.approx(expect, rel=1e-12)
    assert atm.state(1800.0).w_g_m3 == pytest.approx(absolute_humidity_g_m3(293.15, 0.30))


def test_humid_and_fog_crossovers_from_weather_alone() -> None:
    """One preset, one weather series: humid air makes LWIR worse than SWIR, fog makes it
    better -- the §7.2 crossover the simulator must reproduce."""
    atm = Atmosphere(load_atmosphere_preset("us_standard_clear"), _weather_step())
    humid = 2.5 * 3600.0
    fog = 4.5 * 3600.0
    tau = lambda band, t: float(atm.transmittance(band, t, 200.0))  # noqa: E731
    assert tau("lwir", humid) < tau("swir", humid)
    assert tau("lwir", fog) > tau("swir", fog)
    assert tau("lwir", fog) < tau("lwir", humid) < tau("lwir", 0.0)
    assert not atm.state(humid).regime_mismatch and atm.state(fog).regime_mismatch
    fog_preset = Atmosphere(load_atmosphere_preset("fog_light_200m"), _weather_step())
    assert fog_preset.state(fog).regime_mismatch is False and fog_preset.state(0.0).regime_mismatch
    assert float(atm.transmittance("lwir", 0.0, math.inf)) == 0.0


def test_apply_is_the_beer_lambert_kernel(tophat_lwir_lut: BandLUT) -> None:
    w = WeatherSeries.constant(WeatherSample(290.0, 0.5, 1.0, 0.0, 0.0, 0.0, 5000.0, 0.0), 3600.0)
    atm = Atmosphere(load_atmosphere_preset("haze"), w, {"lwir": tophat_lwir_lut})
    l_air = atm.air_radiance("lwir", 0.0)
    lb = np.full((4, 4), l_air, dtype=np.float32)
    d = np.array([[0.0, 10.0, 200.0, 5000.0]] * 4, dtype=np.float32)
    out = atm.apply("lwir", 0.0, lb, d)
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, l_air, rtol=1e-6)  # isothermal invariance
    hot = np.full((4, 4), float(tophat_lwir_lut.lookup(310.0)[()]), dtype=np.float32)
    out_hot = atm.apply("lwir", 0.0, hot, d)
    contrast = (out_hot - out) / (hot - lb)
    np.testing.assert_allclose(contrast, atm.transmittance("lwir", 0.0, d), rtol=1e-5)
    assert atm.air_radiance("lwir", 0.0, "lb_q") == pytest.approx(
        float(tophat_lwir_lut.lookup(290.0, "lb_q")[()])
    )
