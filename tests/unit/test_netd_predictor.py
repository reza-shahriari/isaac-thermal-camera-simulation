"""NETD predictor and datasheet figures of merit (M4.5, ADR 0024)."""

from __future__ import annotations

import copy
import math
import pathlib
from typing import Any

import pytest
import yaml

from irsim.config.sensor import SensorConfig, SensorSpec
from irsim.detector import (
    NoiseBudget,
    bolometer_floors,
    photoelectrons,
    predict_netd_k,
    signal_derivative_per_k,
)
from irsim.detector.figures_of_merit import (
    d_star_cm_hz_w,
    datasheet_over_photon_count_ratio,
    enbw_first_order_hz,
    enbw_sampled_iir_hz,
    nep_w,
    netd_datasheet_form_k,
    netd_photon_count_form_k,
)
from irsim.detector.params import PhotonParams, fpa_params_from_config
from irsim.optics import pixel_power
from irsim.radiometry.lut import BandLUT
from irsim.radiometry.planck import band_photon_radiance_tophat, band_radiance_tophat

REPO = pathlib.Path(__file__).resolve().parents[2]
BOSON = yaml.safe_load((REPO / "configs" / "sensors" / "flir_boson_640_lwir.yaml").read_text())


def _boson(**optics: Any) -> SensorSpec:
    d = copy.deepcopy(BOSON)
    d["sensor"]["optics"].update(optics)
    return SensorConfig.model_validate(d).sensor


def _mwir(
    read_noise_e: float | None = 30.0, t_int_ms: float = 5.0, f_number: float = 2.0, **fpa: Any
) -> SensorSpec:
    d = copy.deepcopy(BOSON)
    d["sensor"]["band"].update(lambda_min_um=3.0, lambda_max_um=5.0, regime="mixed")
    d["sensor"]["optics"].update(f_number=f_number, transmittance=0.9)
    d["sensor"]["fpa"] = {
        "type": "photon",
        "width": 640,
        "height": 512,
        "pitch_um": 15.0,
        "fill_factor": 1.0,
        "frame_rate_hz": 30,
        "bit_depth": 14,
        "quantum_efficiency": 0.7,
        "well_capacity_e": 1.0e7,
        "integration_time_ms": t_int_ms,
        "dark_current_model": "arrhenius",
        "read_noise_e": read_noise_e,
        **fpa,
    }
    return SensorConfig.model_validate(d).sensor


# --- figures of merit ---------------------------------------------------------------------


def test_d_star_known_answer_and_identities() -> None:
    """NEP 1e-12 W, A_d = (12 µm)² = 1.44e-6 cm², Δf = 30 Hz → D* = 6.573e9 cm·Hz^½/W."""
    a_d = (12e-6) ** 2
    d = d_star_cm_hz_w(a_d, 30.0, 1e-12)
    assert d == pytest.approx(6.573e9, rel=1e-4)
    assert d == pytest.approx(math.sqrt(1.44e-6 * 30.0) / 1e-12, rel=1e-9)
    assert nep_w(2e-7, -8.4e5) == pytest.approx(2e-7 / 8.4e5, rel=1e-12)
    assert d * 1e-12 / math.sqrt(30.0) == pytest.approx(math.sqrt(1.44e-6), rel=1e-12)


def test_datasheet_form_over_photon_count_form_is_0_8_at_f1() -> None:
    """For the same system the paraxial 4F² form gives 4F²/(4F²+1) = 0.8 of the +1-form NETD at
    F/1 (the spec's "20 % at F/1.0"): a datasheet figure in that convention is 1.25x low."""
    assert datasheet_over_photon_count_ratio(1.0) == pytest.approx(0.8, rel=1e-9)
    assert 1.0 / datasheet_over_photon_count_ratio(1.0) == pytest.approx(1.25, rel=1e-9)
    a_d, l_prime, nep = 1.296e-10, 0.8, 3e-12  # l_prime: radiance derivative W/m2/sr/K
    paraxial = netd_datasheet_form_k(1.0, nep, a_d, math.pi * l_prime)  # M' = pi L'
    plus_one = netd_photon_count_form_k(nep, a_d * (math.pi / 5.0) * l_prime)
    assert paraxial / plus_one == pytest.approx(0.8, rel=1e-9)


def test_enbw_of_sampled_iir_tends_to_1_over_4tau() -> None:
    tau = 10e-3
    assert enbw_first_order_hz(tau) == 25.0
    assert abs(enbw_sampled_iir_hz(tau, tau / 1000.0) / 25.0 - 1.0) < 1e-2
    # at 60 Hz with tau = 10 ms the sampled window is ~18 % narrower -- reported, not hidden
    assert 0.78 < enbw_sampled_iir_hz(tau, 1.0 / 60.0) / 25.0 < 0.86


# --- predictor ------------------------------------------------------------------------------


@pytest.mark.parametrize("t", [250.0, 300.0, 373.0, 500.0])
def test_bolometer_derivative_matches_finite_difference_of_the_transfer(
    tophat_lwir_lut: BandLUT, t: float
) -> None:
    """dS/dT from the derivative table vs a central FD (1 mK) of the transfer on the closed-form
    top-hat radiance (float64; the float32 radiance table cannot be differenced over 2 mK)."""
    s = _boson()
    d = signal_derivative_per_k(t, s, tophat_lwir_lut)
    dt = 1e-3
    hi = float(
        pixel_power(band_radiance_tophat(7.5, 13.5, t + dt), 1.0, 0.92, s.detector_active_area_m2)
    )
    lo = float(
        pixel_power(band_radiance_tophat(7.5, 13.5, t - dt), 1.0, 0.92, s.detector_active_area_m2)
    )
    assert abs(d / ((hi - lo) / (2 * dt)) - 1.0) < 1e-5


@pytest.mark.parametrize("t", [250.0, 300.0, 373.0, 500.0])
def test_photon_derivative_matches_finite_difference_of_photoelectrons(
    tophat_mwir_lut: BandLUT, t: float
) -> None:
    s = _mwir()
    p = fpa_params_from_config(SensorConfig(sensor=s))
    assert isinstance(p, PhotonParams)
    d = signal_derivative_per_k(t, s, tophat_mwir_lut)
    dt = 1e-3
    hi = float(photoelectrons(band_photon_radiance_tophat(3.0, 5.0, t + dt), p, 2.0, 0.9))
    lo = float(photoelectrons(band_photon_radiance_tophat(3.0, 5.0, t - dt), p, 2.0, 0.9))
    assert abs(d / ((hi - lo) / (2 * dt)) - 1.0) < 1e-5


def test_netd_falls_with_scene_temperature(
    tophat_lwir_lut: BandLUT, tophat_mwir_lut: BandLUT
) -> None:
    """Scene-independent bolometer noise: NETD(373)/NETD(300) == dLb_dT(300)/dLb_dT(373) to 1e-9,
    and for 7.5-13.5 µm that is within 15 % of the published 23/39 = 0.59 [R9]."""
    s = _boson()
    budget = NoiseBudget(kind="bolometer", sigma_gaussian=3e-12)
    ratio = predict_netd_k(373.0, s, tophat_lwir_lut, budget) / predict_netd_k(
        300.0, s, tophat_lwir_lut, budget
    )
    lut = tophat_lwir_lut
    derivative_ratio = float(lut.lookup(300.0, "dlb_dt")[()]) / float(
        lut.lookup(373.0, "dlb_dt")[()]
    )
    assert abs(ratio / derivative_ratio - 1.0) < 1e-9
    assert ratio < 1.0 and abs(ratio / (23.0 / 39.0) - 1.0) < 0.15, ratio
    m = _mwir()
    photon = NoiseBudget(kind="photon", sigma_gaussian=30.0)
    assert predict_netd_k(373.0, m, tophat_mwir_lut, photon) < predict_netd_k(
        300.0, m, tophat_mwir_lut, photon
    )


def test_shot_limited_scaling(tophat_mwir_lut: BandLUT) -> None:
    """σ_read = 0: NETD == √N_e / (∂N_e/∂T) to 1e-12; doubling t_int → NETD/√2."""
    from irsim.detector.netd import mean_signal

    budget = NoiseBudget(kind="photon", sigma_gaussian=0.0)
    s5, s10 = _mwir(t_int_ms=5.0), _mwir(t_int_ms=10.0)
    n_e = mean_signal(300.0, s5, tophat_mwir_lut)
    expected = math.sqrt(n_e) / signal_derivative_per_k(300.0, s5, tophat_mwir_lut)
    assert predict_netd_k(300.0, s5, tophat_mwir_lut, budget) == pytest.approx(expected, rel=1e-12)
    assert 1e6 < n_e < 1e8
    ratio = predict_netd_k(300.0, s10, tophat_mwir_lut, budget) / predict_netd_k(
        300.0, s5, tophat_mwir_lut, budget
    )
    assert ratio == pytest.approx(1.0 / math.sqrt(2.0), rel=1e-9)


def test_aperture_convention_in_netd(tophat_mwir_lut: BandLUT) -> None:
    """Read-limited NETD(F=2)/NETD(F=1) == 17/5 = 3.4 (4F² would give 4.0); shot-limited √3.4."""
    read = NoiseBudget(kind="photon", sigma_gaussian=1e9)  # read ≫ shot (3e3 e-): 1e-11 relative
    s = _mwir()
    r = predict_netd_k(300.0, s, tophat_mwir_lut, read, f_number=2.0) / predict_netd_k(
        300.0, s, tophat_mwir_lut, read, f_number=1.0
    )
    assert r == pytest.approx(3.4, rel=1e-9)
    shot = NoiseBudget(kind="photon", sigma_gaussian=0.0)
    r2 = predict_netd_k(300.0, s, tophat_mwir_lut, shot, f_number=2.0) / predict_netd_k(
        300.0, s, tophat_mwir_lut, shot, f_number=1.0
    )
    assert r2 == pytest.approx(math.sqrt(3.4), rel=1e-9)


def test_boson_floors_are_finite_and_plausible(tophat_lwir_lut: BandLUT) -> None:
    """Best-estimate VOx constants give finite first-principles floors in [5, 500] mK, and a
    Boson-class D* of order 1e8-1e10 cm·Hz^½/W; the anchor (M4.6) overrides the magnitude.

    ENBW is 1/(4 tau), so it is set by the membrane and by nothing else: at the datasheet's
    nominal 8 ms (SC.3, ADR 0091) that is 31.25 Hz, where the 10 ms this project carried as
    ESTIMATED gave 25 Hz. A faster membrane passes more bandwidth and therefore has a *higher*
    temperature-fluctuation floor. With the generic 1e-7 W/K conductance that floor was 74.5 mK,
    above the 50 mK the datasheet claims -- the direction nobody expects, and a sign the number
    was a placeholder; SC.34 authors VOx's G_th as 2e-8 W/K (ESTIMATED from the literature's
    1-3e-8) and the floor is 33.3 mK, under the anchor as a floor should be. The anchor (M4.6)
    still overrides the magnitude; this pins the floor so a change to the constants is seen.
    """
    s = _boson()
    floors = bolometer_floors(s, tophat_lwir_lut)
    assert floors["enbw_hz"] == pytest.approx(31.25)
    assert floors["enbw_hz"] == pytest.approx(1.0 / (4.0 * s.fpa.thermal_time_constant_ms * 1e-3))
    assert floors["netd_temperature_fluctuation_k"] == pytest.approx(0.0333, rel=1e-2)  # SC.34
    for key in ("netd_temperature_fluctuation_k", "netd_johnson_k"):
        assert 1e-3 < floors[key] < 0.5, (key, floors[key])
    total = math.hypot(floors["netd_temperature_fluctuation_k"], floors["netd_johnson_k"])
    assert 5e-3 < total < 0.5, total
    assert floors["responsivity_v_per_w"] == pytest.approx(4.2e6, rel=1e-9)  # 5x: G_th 2e-8, SC.34
    nep = 0.05 * 0.8 * signal_derivative_per_k(300.0, s, tophat_lwir_lut)  # NEP at the 50 mK anchor
    d_star = d_star_cm_hz_w(s.detector_active_area_m2, floors["enbw_hz"], nep)
    assert 1e8 < d_star < 1e10, d_star


def test_budget_validation() -> None:
    with pytest.raises(ValueError):
        NoiseBudget(kind="photon", sigma_gaussian=-1.0)
    with pytest.raises(ValueError):
        netd_photon_count_form_k(1.0, 0.0)
    with pytest.raises(ValueError):
        d_star_cm_hz_w(1e-10, 0.0, 1e-12)
