"""Roadmap M10.4, engine-free half: the Warp stage module imports without Warp, refuses what the
CPU oracle refuses, and keys its device tables by content (so an identical table is never
re-uploaded and a rebuilt one always is). The kernel itself is compared to the oracle in
`tests/integration/test_kernels_vs_reference.py`."""

from __future__ import annotations

import pathlib
from typing import Any

import numpy as np
import pytest

from irsim.atmosphere import Atmosphere, load_atmosphere_preset
from irsim.atmosphere.beer_lambert import transmittance
from irsim.atmosphere.layered import LayeredAtmosphere
from irsim.materials import MaterialTable
from irsim.pipeline import PipelineConfig, PipelineState
from irsim.pipeline.radiance import band_radiance_stage
from irsim.radiometry.lut import BandLUT
from irsim.thermal import WeatherSample, WeatherSeries
from irsim_isaac.pipeline import warp_stages as ws


def test_module_imports_without_warp_and_entry_points_say_so() -> None:
    if ws.has_warp_module():
        pytest.skip("Warp is importable here; the no-Warp path is exercised in CI")
    with pytest.raises(RuntimeError, match="Warp"):
        ws.band_radiance_warp(
            np.full((2, 2), 300.0, np.float32),
            np.ones((2, 2), np.int32),
            MaterialTable.constant(1.0),
            _tiny_lut(),
        )


def test_equivalence_registry_names_the_cpu_oracle() -> None:
    cpu, gpu = ws.EQUIVALENCE_STAGES["band_radiance"]
    assert cpu is band_radiance_stage
    assert gpu is ws.band_radiance_stage_warp
    assert ws.WarpBandRadianceStage.name == "band_radiance"


def _tiny_lut() -> BandLUT:
    n = 5
    t = np.linspace(200.0, 1000.0, n)
    lb = (t / 300.0).astype(np.float32)
    return BandLUT(200.0, 1000.0, n, lb, lb, lb, lb, band_hash="tiny")


def test_validate_material_ids_mirrors_the_oracle() -> None:
    table = MaterialTable.from_mapping({1: 0.95, 2: 0.5})
    ids = np.array([[1, 2], [2, 1]], np.int32)
    ws.validate_material_ids(ids, table)  # fine
    with pytest.raises(ValueError, match="UNMAPPED"):
        ws.validate_material_ids(np.array([[0, 1]], np.int32), table)
    with pytest.raises(ValueError, match="table has"):
        ws.validate_material_ids(np.array([[1, 7]], np.int32), table)
    with pytest.raises(TypeError):
        ws.validate_material_ids(np.array([[1.0, 2.0]]), table)
    # under the sky mask the id is ignored, exactly as MaterialTable.emissivity_for does
    sky = np.array([[True, False]])
    ws.validate_material_ids(np.array([[0, 1]], np.int32), table, sky)
    with pytest.raises(ValueError, match="sky_mask"):
        ws.validate_material_ids(ids, table, np.array([[1, 0]], np.int32))


def test_tables_key_tracks_content_not_identity() -> None:
    lut = _tiny_lut()
    a = MaterialTable.from_mapping({1: 0.9})
    b = MaterialTable.from_mapping({1: 0.9})
    c = MaterialTable.from_mapping({1: 0.8})
    k_a = ws.tables_key(lut, a, "lb", "cuda:0")
    assert k_a == ws.tables_key(lut, b, "lb", "cuda:0")
    assert k_a != ws.tables_key(lut, c, "lb", "cuda:0")
    assert k_a != ws.tables_key(lut, a, "lb", "cpu")
    assert k_a != ws.tables_key(lut, a, "lb_q", "cuda:0") or np.array_equal(lut.lb, lut.lb_q)


# ---- M10.5: stage 2 and stage 3, the halves that need no GPU ----------------------------------


def _boson(**fpa: Any) -> Any:
    import yaml

    from irsim.config.sensor import SensorConfig

    repo = pathlib.Path(__file__).resolve().parents[2]
    raw = yaml.safe_load((repo / "configs/sensors/flir_boson_640_lwir.yaml").read_text())
    raw["sensor"]["fpa"].update(fpa)
    return SensorConfig.model_validate(raw)


def _weather() -> WeatherSeries:
    return WeatherSeries.constant(
        WeatherSample(288.15, 0.46, 1.0, 0.0, 0.0, 0.0, 23000.0, 0.0), 3600.0
    )


def _config(lut: BandLUT, **kw: Any) -> PipelineConfig:
    return PipelineConfig.from_sensor(
        _boson(width=16, height=16), MaterialTable.from_mapping({1: 0.95}), lut=lut, **kw
    )


def test_no_atmosphere_means_the_identity_stage_on_both_paths() -> None:
    config = _config(_tiny_lut())
    assert config.atmosphere is None
    assert ws.atmosphere_terms(config, PipelineState()) is None


def test_grey_terms_reproduce_the_beer_lambert_oracle(tophat_lwir_lut: BandLUT) -> None:
    """The host must read tau(d) off the model, not re-derive it: compare the terms' own
    transmittance with `irsim.atmosphere.beer_lambert.transmittance` at the same distances."""
    lut = tophat_lwir_lut
    atmosphere = Atmosphere(load_atmosphere_preset("us_standard_clear"), _weather())
    config = _config(lut, atmosphere=atmosphere)
    terms = ws.atmosphere_terms(config, PipelineState())
    assert terms is not None and terms.n_terms == 1

    gamma = atmosphere.state(0.0).gamma_per_m["lwir"]
    d = np.array([0.0, 1.0, 50.0, 200.0, 5000.0, np.inf])
    assert terms.transmittance(d) == pytest.approx(transmittance(d, gamma), rel=1e-6)
    t_air = atmosphere.state(0.0).t_air_k
    assert terms.l_air == pytest.approx(float(lut.lookup(np.float64(t_air), "lb")[()]), rel=1e-12)


def test_layered_terms_reproduce_the_exponential_sum_including_the_infinite_path(
    tophat_lwir_lut: BandLUT,
) -> None:
    lut = tophat_lwir_lut
    atmosphere = LayeredAtmosphere(
        load_atmosphere_preset("us_standard_clear"), _weather(), {"lwir": lut}
    )
    config = _config(lut, atmosphere=atmosphere)
    terms = ws.atmosphere_terms(config, PipelineState())
    assert terms is not None and terms.n_terms == 3, "LWIR has three spectral classes (MS.1)"

    d = np.array([0.0, 50.0, 200.0, 5000.0, np.inf])
    expected = atmosphere.transmittance("lwir", 0.0, d, 0.0)
    assert terms.transmittance(d) == pytest.approx(expected, rel=1e-6, abs=1e-12)
    assert terms.tau_at_inf == 0.0, "an infinite horizontal path is opaque in the layered model"
    assert terms.l_air == pytest.approx(atmosphere.air_radiance("lwir", 0.0, "lb"), rel=1e-12)


def test_tau_override_is_carried_as_a_constant_and_ignores_distance(
    tophat_lwir_lut: BandLUT,
) -> None:
    lut = tophat_lwir_lut
    config = _config(
        lut,
        atmosphere=Atmosphere(load_atmosphere_preset("us_standard_clear"), _weather()),
        tau_override=0.7,
    )
    terms = ws.atmosphere_terms(config, PipelineState())
    assert terms is not None and terms.tau_const == pytest.approx(0.7)
    tau = terms.transmittance(np.array([0.0, 500.0, np.inf]))
    assert np.allclose(tau, 0.7)


def test_atmosphere_terms_refuse_an_inconsistent_set() -> None:
    ok = np.array([0.5, 0.5], np.float32)
    with pytest.raises(ValueError, match="sum to"):
        ws.AtmosphereTerms(np.array([0.5, 0.9], np.float32), ok, 1.0, 0.0)
    with pytest.raises(ValueError, match="non-negative"):
        ws.AtmosphereTerms(ok, np.array([1e-4, -1e-4], np.float32), 1.0, 0.0)
    with pytest.raises(ValueError, match="matching"):
        ws.AtmosphereTerms(ok, np.array([1e-4], np.float32), 1.0, 0.0)


def test_validate_distance_mirrors_the_oracle() -> None:
    shape = (2, 2)
    ws.validate_distance(np.array([[1.0, np.inf], [0.0, 5.0]]), shape)
    with pytest.raises(ValueError, match="non-negative"):
        ws.validate_distance(np.array([[1.0, -1.0], [0.0, 5.0]]), shape)
    with pytest.raises(ValueError, match="non-negative"):
        ws.validate_distance(np.array([[1.0, np.nan], [0.0, 5.0]]), shape)
    with pytest.raises(ValueError, match="shape"):
        ws.validate_distance(np.ones((3, 3)), shape)
    with pytest.raises(ValueError, match="sky_mask"):
        ws.validate_distance(np.ones(shape), shape, np.ones(shape, np.int32))


def test_optics_terms_take_every_scalar_from_irsim_optics() -> None:
    """Non-negotiable #5 in its positive form: the aperture factor the kernel is handed is the
    one `irsim.optics.aperture` computes, and the housing power is `housing_power_axis`'s value
    (ADR 0145: the kernel subtracts L_h before scaling by RI, then adds A_d Omega_eff L_h)."""
    from irsim.optics.aperture import aperture_factor
    from irsim.optics.self_emission import housing_power_axis
    from irsim.optics.stage import optics_field

    spec = _boson(width=8, height=8).sensor
    terms = ws.optics_terms(spec, 12.5, 4, None)
    f, tau, a_d = spec.optics.f_number, spec.optics.transmittance, spec.detector_active_area_m2
    assert terms.factor == pytest.approx(aperture_factor(f) * tau, rel=1e-15)
    assert terms.phi_housing == pytest.approx(housing_power_axis(a_d, f, tau, 12.5), rel=1e-15)
    assert terms.lb_housing == 12.5
    assert terms.area_m2 == pytest.approx(a_d, rel=1e-15)
    assert np.array_equal(terms.cos4, optics_field(spec).astype(np.float32))
    assert terms.supersample == 4 and terms.psf is None


def test_optics_terms_refuse_an_even_sided_psf() -> None:
    spec = _boson(width=8, height=8).sensor
    with pytest.raises(ValueError, match="odd sides"):
        ws.optics_terms(spec, 1.0, 4, np.ones((4, 4)))
    with pytest.raises(ValueError, match="supersample"):
        ws.OpticsTerms(1.0, 1.0, 0.0, 0.0, 0, np.ones((2, 2), np.float32), None)


def test_the_kernel_source_never_recomputes_the_aperture_factor() -> None:
    """M10.5 asks for this explicitly. `tests/unit/test_aperture_guard.py` already walks every
    module in src/irsim_isaac; this points the same scanner at the kernel file by name, so the
    failure names stage 3 rather than "some module" if a future edit inlines pi/(4F^2+1)."""
    from test_aperture_guard import aperture_expressions

    source = pathlib.Path(ws.__file__).read_text(encoding="utf-8")
    assert aperture_expressions(source) == []
    assert "4.0 * f" not in source and "4.f*F*F" not in source


def test_equivalence_registry_covers_every_landed_stage() -> None:
    from irsim.pipeline.atmosphere import atmosphere_stage
    from irsim.pipeline.optics import optics_stage

    assert set(ws.EQUIVALENCE_STAGES) == set(ws.EQUIVALENCE_OUTPUT)
    assert ws.EQUIVALENCE_STAGES["atmosphere"] == (atmosphere_stage, ws.atmosphere_stage_warp)
    assert ws.EQUIVALENCE_STAGES["optics"] == (optics_stage, ws.optics_stage_warp)
    assert ws.EQUIVALENCE_OUTPUT["optics"] == "flux", "stage 3 leaves pixel power, not radiance"
    assert ws.WarpAtmosphereStage.name == "atmosphere" and ws.WarpOpticsStage.name == "optics"


# ---- M10.6: stage 4, the half that needs no GPU -----------------------------------------------


def _photon_config(lut: BandLUT) -> PipelineConfig:
    """The Boson file with a cooled MWIR photon FPA swapped in -- the other stage-4 path."""
    import copy

    import yaml

    repo = pathlib.Path(__file__).resolve().parents[2]
    cfg = copy.deepcopy(
        yaml.safe_load((repo / "configs/sensors/flir_boson_640_lwir.yaml").read_text())
    )
    cfg["sensor"]["band"].update(lambda_min_um=3.0, lambda_max_um=5.0, regime="mixed")
    cfg["sensor"]["fpa"] = {
        "type": "photon",
        "width": 8,
        "height": 8,
        "pitch_um": 15.0,
        "fill_factor": 1.0,
        "frame_rate_hz": 30,
        "bit_depth": 14,
        "quantum_efficiency": 0.8,
        "well_capacity_e": 1.0e6,
        "integration_time_ms": 5.0,
        "dark_current_model": "arrhenius",
    }
    from irsim.config.sensor import SensorConfig

    return PipelineConfig.from_sensor(
        SensorConfig.model_validate(cfg), MaterialTable.from_mapping({1: 0.95}), lut=lut
    )


def test_bolometer_terms_are_the_oracle_transfer_and_alpha(tophat_lwir_lut: BandLUT) -> None:
    from irsim.detector.lowpass import alpha_for

    config = _config(tophat_lwir_lut)
    terms = ws.detector_terms(config)
    transfer = config.detector.transfer
    assert terms.kind == "bolometer"
    assert terms.gain_dn_per_w == pytest.approx(transfer.gain_dn_per_w, rel=1e-15)
    assert terms.offset_w == pytest.approx(transfer.offset_w, rel=1e-15)
    expected = alpha_for(config.fpa.frame_dt_s, config.fpa.thermal_time_constant_s)
    assert terms.alpha == pytest.approx(expected, rel=1e-15)
    # the §9.2 number the kernel actually applies -- not the "0.6 frames" of spec issue S8
    assert terms.alpha == pytest.approx(0.875, abs=0.002)  # 8 ms at 60 Hz (SC.3, was 0.811)
    assert terms.dn_max == config.fpa.dn_max


def test_photon_terms_fold_the_transfer_into_three_scalars(tophat_mwir_lut: BandLUT) -> None:
    config = _photon_config(tophat_mwir_lut)
    terms = ws.detector_terms(config)
    fpa = config.fpa
    budget = config.detector.budget
    assert terms.kind == "photon"
    assert terms.qe_t_int == pytest.approx(fpa.quantum_efficiency * fpa.integration_time_s)
    assert terms.offset_e == pytest.approx(budget.dark_electrons + budget.background_electrons)
    assert terms.dn_per_electron == pytest.approx(2**fpa.bit_depth / fpa.well_capacity_e)
    assert terms.alpha == 0.0, "a photon FPA has no membrane to lag"


def test_detector_terms_refuse_an_impossible_set() -> None:
    with pytest.raises(ValueError, match="unknown FPA type"):
        ws.DetectorTerms(kind="ccd", dn_max=1023)
    with pytest.raises(ValueError, match="blend weight"):
        ws.DetectorTerms(kind="bolometer", dn_max=1023, alpha=0.0)
    with pytest.raises(ValueError, match="blend weight"):
        ws.DetectorTerms(kind="bolometer", dn_max=1023, alpha=1.5)


def test_validate_flux_mirrors_the_detector_guards() -> None:
    good = np.full((2, 2), 1e-8, np.float32)
    assert ws.validate_flux(good, "bolometer").dtype == np.float32
    with pytest.raises(ValueError, match="negative"):
        ws.validate_flux(np.full((2, 2), -1e-9, np.float32), "bolometer")
    with pytest.raises(TypeError):
        ws.validate_flux(good.astype(np.float16), "bolometer")
    # a photon rate handed to a bolometer is the form mix-up §9.2 warns about
    with pytest.raises(ValueError, match="photon rate"):
        ws.validate_flux(np.full((2, 2), 1e12, np.float32), "bolometer")
    ws.validate_flux(np.full((2, 2), 1e12, np.float32), "photon")  # fine in photons/s


def test_the_device_state_is_owned_by_the_pipeline_state(tophat_lwir_lut: BandLUT) -> None:
    """ADR 0052 again, on the device side: one owner so a cold start clears everything at once."""
    state = PipelineState()
    first = ws.warp_pipeline_state(state, "cuda:0")
    assert state.buffers[ws.DEVICE_STATE_KEY] is first
    assert ws.warp_pipeline_state(state, "cuda:0") is first, "one per state x device"
    other = ws.warp_pipeline_state(state, "cpu")
    assert other is not first and other.device == "cpu"
    assert first.iir_ptr is None, "nothing allocated until a frame runs"
    first.reset()
    assert first.iir_ptr is None


def test_stage_four_is_registered_against_its_oracle() -> None:
    from irsim.pipeline.detector import detector_stage

    assert ws.EQUIVALENCE_STAGES["detector"] == (detector_stage, ws.detector_stage_warp)
    assert ws.EQUIVALENCE_OUTPUT["detector"] == "signal_dn"
    assert ws.WarpDetectorStage.name == "detector"
