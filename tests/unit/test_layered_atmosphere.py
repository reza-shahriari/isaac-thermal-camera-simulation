"""Layered slant-path atmosphere (MS.1): horizontal identity with M8.1, isothermal invariance at
every elevation, the exact exponential sum of M8.7's piecewise spectra, the fitter, the 200 m
anchor for every preset, the R13 sky anchors, convergence along the ray, and stage-2 wiring."""

from __future__ import annotations

import copy
import math
import pathlib
from typing import Any

import numpy as np
import pytest
import yaml

from irsim.atmosphere import (
    Atmosphere,
    ExponentialSum,
    LayeredAtmosphere,
    class_weights,
    exponential_sum_from_piecewise,
    fit_exponential_sum,
    load_atmosphere_preset,
)
from irsim.atmosphere.beer_lambert import transmittance
from irsim.atmosphere.layered import ANCHOR_DISTANCE_M, classes_for
from irsim.atmosphere.spectral import band_transmittance_spectral
from irsim.config.sensor import SensorConfig
from irsim.materials import MaterialTable
from irsim.pipeline import PipelineConfig, PipelineState, run_frame
from irsim.radiometry.lut import BandLUT
from irsim.radiometry.spectral_response import SpectralResponse, load_spectral_response
from irsim.thermal import WeatherSample, WeatherSeries

REPO = pathlib.Path(__file__).resolve().parents[2]
BOSON = yaml.safe_load((REPO / "configs" / "sensors" / "flir_boson_640_lwir.yaml").read_text())
FIT = {
    "us_standard_clear": (288.15, 0.46, 23000.0),
    "midlat_summer_humid": (303.15, 0.80, 23000.0),
    "midlat_winter_dry": (272.2, 0.76, 23000.0),
    "tropical": (300.0, 0.74, 23000.0),
    "haze": (288.15, 0.46, 1500.0),
    "fog_light_200m": (283.15, 1.0, 200.0),
    "fog_dense_50m": (283.15, 1.0, 50.0),
}


def _weather(t: float, rh: float, vis: float) -> WeatherSeries:
    return WeatherSeries.constant(WeatherSample(t, rh, 1.0, 0.0, 0.0, 0.0, vis, 0.0), 3600.0)


def _tophat(tmp_path: pathlib.Path, lo: float, hi: float) -> SpectralResponse:
    p = tmp_path / f"tophat_{lo}_{hi}.csv"
    p.write_text(f"# exact top-hat\n{lo},1.0\n{hi},1.0\n")
    return load_spectral_response(p)


def test_single_class_horizontal_reproduces_beer_lambert_to_1e12() -> None:
    es = ExponentialSum(np.array([1.0]), np.array([4e-4]), np.array([2000.0]), 0.0, 1200.0)
    d = np.array([0.0, 10.0, 200.0, 5000.0])
    np.testing.assert_allclose(es.transmittance(d, 0.0), transmittance(d, 4e-4), rtol=1e-12)
    assert float(es.transmittance(np.inf, 0.0)) == 0.0
    lb = lambda h: np.full(h.shape, 42.0)  # noqa: E731
    for dd in (10.0, 200.0, 5000.0):
        assert es.path_radiance(dd, 0.0, lb) == pytest.approx(
            (1 - math.exp(-4e-4 * dd)) * 42.0, rel=1e-12
        )
    assert es.path_radiance(math.inf, 0.0, lb) == pytest.approx(42.0, rel=1e-12)
    # the visible band has one class, so the layered model IS the grey model there
    w = _weather(288.15, 0.5, 5000.0)
    preset = load_atmosphere_preset("haze")
    grey, layered = Atmosphere(preset, w), LayeredAtmosphere(preset, w)
    np.testing.assert_allclose(
        layered.transmittance("visible", 0.0, d), grey.transmittance("visible", 0.0, d), rtol=1e-12
    )


@pytest.mark.parametrize("deg", [0.0, 5.0, 15.0, 45.0, 90.0])
def test_isothermal_atmosphere_invariance_at_every_elevation(
    tmp_path: pathlib.Path, deg: float
) -> None:
    """Lapse rate 0: a 300 K blackbody beyond the whole column plus the column's own emission
    equals L_B(300 K) -- τ(∞) L + L_sky = L_B(T_air) -- at every elevation."""
    r = _tophat(tmp_path, 3.0, 5.0)
    lut = BandLUT.build(r)
    preset = load_atmosphere_preset("us_standard_clear")
    iso = preset.model_copy(
        update={"profile": preset.profile.model_copy(update={"lapse_rate_k_per_m": 0.0})}
    )
    atm = LayeredAtmosphere(iso, _weather(300.0, 0.4, 23000.0), {"mwir": lut}, {"mwir": r})
    th = math.radians(deg)
    lb = float(lut.lookup(np.float64(300.0))[()])
    total = (
        atm.sky_radiance("mwir", 0.0, th) + float(atm.transmittance("mwir", 0.0, math.inf, th)) * lb
    )
    assert abs(total / lb - 1.0) < 1e-6, total / lb - 1.0
    for d in (10.0, 2000.0, 50000.0):
        out = float(atm.apply("mwir", 0.0, np.float64(lb), d, th))
        assert abs(out / lb - 1.0) < 1e-6


def test_piecewise_spectra_of_m8_7_are_exact_exponential_sums(tmp_path: pathlib.Path) -> None:
    """The grey model was off by 9 % / 42 % at 1 km on these (ADR 0048); the exponential sum
    reproduces the spectral quadrature to round-off, including at 5 km. (Half-open sub-bands,
    the convention of exponential_sum_from_piecewise.)"""
    cases = [
        (_tophat(tmp_path, 8.0, 12.0), lambda lam: np.where(lam < 10.0, 2e-4, 1.2e-3), (10.0,)),
        (
            _tophat(tmp_path, 3.0, 5.0),
            lambda lam: np.where((lam >= 4.2) & (lam < 4.4), 5.0, 3e-4),
            (4.2, 4.4),
        ),
    ]
    d = np.array([50.0, 300.0, 1000.0, 5000.0])
    for resp, gamma, breaks in cases:
        es = exponential_sum_from_piecewise(resp, gamma, breaks)
        exact = band_transmittance_spectral(resp, gamma, d)
        np.testing.assert_allclose(es.transmittance(d, 0.0), exact, rtol=1e-9)
        assert abs(float(es.transmittance(5000.0, 0.0)) / float(exact[-1]) - 1.0) < 0.05
        # uniform-temperature path radiance follows from tau exactly: (1 - tau) L_B
        lb = lambda h: np.full(h.shape, 1.0)  # noqa: E731
        assert es.path_radiance(5000.0, 0.0, lb) == pytest.approx(1.0 - float(exact[-1]), rel=1e-9)


def test_fitter_recovers_a_two_level_curve_of_growth() -> None:
    w_true, g_true = np.array([0.45, 0.55]), np.array([2e-4, 1.2e-3])
    d = np.linspace(20.0, 6000.0, 60)
    tau = np.exp(-np.outer(d, g_true)) @ w_true
    w, g = fit_exponential_sum(d, tau, 2)
    np.testing.assert_allclose(g, g_true, rtol=1e-5)
    np.testing.assert_allclose(w, w_true, atol=1e-4)
    w1, g1 = fit_exponential_sum(d, np.exp(-3e-4 * d), 1)
    assert w1[0] == 1.0 and g1[0] == pytest.approx(3e-4, rel=1e-12)
    with pytest.raises(ValueError):
        fit_exponential_sum(d, tau * 2.0, 2)


@pytest.mark.parametrize("name", sorted(FIT))
def test_horizontal_200m_anchor_matches_the_grey_preset(name: str) -> None:
    t, rh, vis = FIT[name]
    preset = load_atmosphere_preset(name)
    w = _weather(t, rh, vis)
    grey, layered = Atmosphere(preset, w), LayeredAtmosphere(preset, w)
    for band in ("lwir", "mwir", "swir", "nir", "visible"):
        es = layered.exponential_sum(band, 0.0)
        free = np.array([not c.opaque for c in classes_for(band)])
        tau_free = float(
            np.sum(
                es.weights[free]
                * np.exp(-(es.gamma_0[free] + es.gamma_aerosol) * ANCHOR_DISTANCE_M)
            )
        )
        assert tau_free / float(es.weights[free].sum()) == pytest.approx(
            float(grey.transmittance(band, 0.0, ANCHOR_DISTANCE_M)), rel=1e-9
        )
        assert 0.0 < float(es.transmittance(ANCHOR_DISTANCE_M, 0.0)) <= 1.0
        assert abs(float(class_weights(band).sum()) - 1.0) < 1e-12


@pytest.mark.slow  # GT.1: over a second on its own
def test_r13_sky_anchors_and_elevation_shape(tmp_path: pathlib.Path) -> None:
    """R13 (Sensors 21:7067, Tucson, clear, low humidity): FLIR T1020 (7.5-14 um) sky at
    -40 C by 15 deg elevation; the TELOPS M1k (2.2-5.5 um) sky far warmer. Thermal-only, so the
    MWIR value is bounded below +10 C: R13's midday MWIR includes scattered sunlight (ADR 0071)."""
    responses = {"lwir": _tophat(tmp_path, 7.5, 14.0), "mwir": _tophat(tmp_path, 2.2, 5.5)}
    luts = {b: BandLUT.build(r) for b, r in responses.items()}
    atm = LayeredAtmosphere(
        load_atmosphere_preset("us_standard_clear"),
        _weather(288.15, 0.20, 23000.0),
        luts,
        responses,
    )
    lwir15 = atm.apparent_sky_temperature_k("lwir", 0.0, math.radians(15.0)) - 273.15
    mwir15 = atm.apparent_sky_temperature_k("mwir", 0.0, math.radians(15.0)) - 273.15
    assert lwir15 == pytest.approx(-40.0, abs=5.0), lwir15
    assert -5.0 <= mwir15 <= 10.0 and mwir15 - lwir15 > 30.0, (lwir15, mwir15)
    degs = [0.5, 2.0, 5.0, 15.0, 30.0, 60.0, 90.0]
    for band in ("lwir", "mwir"):
        t = [atm.apparent_sky_temperature_k(band, 0.0, math.radians(d)) for d in degs]
        assert all(b < a for a, b in zip(t[:-1], t[1:], strict=True)), "coldest at zenith"
        assert abs(t[0] - 288.15) < 8.0, "the horizon sky approaches T_air"
    with pytest.raises(TypeError, match="never a path"):
        LayeredAtmosphere(load_atmosphere_preset("haze"), "/weather.csv")  # type: ignore[arg-type]


@pytest.mark.parametrize("deg", [5.0, 15.0, 45.0, 90.0])
def test_target_converges_along_the_ray_and_excess_is_monotone(
    tmp_path: pathlib.Path, deg: float
) -> None:
    """L'(d) = τ(d) L_t + L_path(d) converges to τ(∞) L_t + L_sky(θ) (a target beyond the column
    is still seen through the window: L_sky alone is the limit only for an opaque column) and the
    excess over the sky decreases monotonically with range."""
    r = _tophat(tmp_path, 7.5, 13.5)
    lut = BandLUT.build(r)
    atm = LayeredAtmosphere(
        load_atmosphere_preset("us_standard_clear"),
        _weather(288.15, 0.5, 23000.0),
        {"lwir": lut},
        {"lwir": r},
    )
    th = math.radians(deg)
    lt = float(lut.lookup(np.float64(300.0))[()])
    sky = atm.sky_radiance("lwir", 0.0, th)
    limit = float(atm.transmittance("lwir", 0.0, math.inf, th)) * lt + sky
    ds = [100.0, 1000.0, 5000.0, 20000.0, 60000.0, 200000.0]
    seen = [float(atm.apply("lwir", 0.0, np.float64(lt), d, th)) for d in ds]
    excess = [s - sky for s in seen]
    assert all(b < a for a, b in zip(excess[:-1], excess[1:], strict=True)), excess
    dl = float(lut.lookup(np.float64(300.0), "dlb_dt")[()])
    assert abs(seen[-1] - limit) / dl * 1e3 < 5.0, "within 5 mK of the limit at 200 km"
    assert excess[-1] > 0.0 and excess[0] / (lt - sky) == pytest.approx(1.0, abs=0.05)


def _config(lut: BandLUT, atmosphere: Any) -> PipelineConfig:
    d = copy.deepcopy(BOSON)
    d["sensor"]["fpa"].update(width=16, height=8)
    d["sensor"]["optics"]["supersample_factor"] = 1
    return PipelineConfig.from_sensor(
        SensorConfig.model_validate(d),
        MaterialTable.constant(1.0),
        lut=lut,
        noise_enabled=False,
        psf_enabled=False,
        atmosphere=atmosphere,
    )


def _gbuffer(t_k: float, distance_m: float) -> dict[str, np.ndarray]:
    shape = (8, 16)
    return {
        "temperature_k": np.full(shape, t_k, np.float32),
        "normal_dot_view": np.ones(shape, np.float32),
        "distance_m": np.full(shape, distance_m, np.float32),
        "material_id": np.ones(shape, np.int32),
        "sky_view_factor": np.ones(shape, np.float32),
    }


def test_stage_two_with_the_layered_model(tophat_lwir_lut: BandLUT) -> None:
    preset = load_atmosphere_preset("us_standard_clear")
    w = _weather(290.0, 0.5, 23000.0)
    grey, layered = Atmosphere(preset, w), LayeredAtmosphere(preset, w, {"lwir": tophat_lwir_lut})
    for atm in (grey, layered):
        cfg = _config(tophat_lwir_lut, atm)
        out = run_frame(
            _gbuffer(290.0, 3000.0), cfg, PipelineState(housing_temp_k=cfg.t_housing_cal_k)
        )
        assert out.apparent_t is not None
        assert np.max(np.abs(out.apparent_t.astype(np.float64) - 290.0)) * 1e3 < 1.0, "isothermal"
    at_anchor = []
    at_range = []
    for atm in (grey, layered):
        cfg = _config(tophat_lwir_lut, atm)
        a = run_frame(
            _gbuffer(310.0, ANCHOR_DISTANCE_M),
            cfg,
            PipelineState(housing_temp_k=cfg.t_housing_cal_k),
        )
        b = run_frame(
            _gbuffer(310.0, 3000.0), cfg, PipelineState(housing_temp_k=cfg.t_housing_cal_k)
        )
        assert a.apparent_t is not None and b.apparent_t is not None
        at_anchor.append(float(a.apparent_t[4, 8]))
        at_range.append(float(b.apparent_t[4, 8]))
    assert abs(at_anchor[0] - at_anchor[1]) * 1e3 < 5.0, (
        "LWIR has no opaque class: identical at 200 m"
    )
    assert at_range[1] > at_range[0] + 0.1, "curve of growth: more contrast survives at 3 km"
    with pytest.raises(ValueError, match="grey L1"):
        d = copy.deepcopy(BOSON)
        d["sensor"]["fpa"].update(width=16, height=8)
        d["sensor"]["optics"]["supersample_factor"] = 1
        PipelineConfig.from_sensor(
            SensorConfig.model_validate(d),
            MaterialTable.constant(1.0),
            lut=tophat_lwir_lut,
            atmosphere=layered,
            tau_override=0.8,
        )
