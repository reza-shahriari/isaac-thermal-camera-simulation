"""SC.18: the FFC snapshots the shutter and the housing; the residual is radial (§11.2).

At a shutter event the camera re-measures its offset on the closed shutter, so a frame of the
shutter reads uniform. What survives afterwards is (i) the housing's drift since the event,
weighted toward the corners by (1 − τ RI)/(τ RI), and (ii) any gain-map error, scaled by the scene
*minus the shutter*. Before SC.18 the FFC recorded only T_FPA and the residual was a white field
proportional to ΔT_FPA, so a uniform scene stayed flat whatever the housing did.
"""

from __future__ import annotations

import copy
import pathlib

import numpy as np
import pytest
import yaml

from irsim.config.sensor import SensorConfig, SensorSpec
from irsim.isp.nuc import TwoPointNuc
from irsim.materials.table import MaterialTable
from irsim.noise.nuc_residual import NucResidual
from irsim.optics import aperture_factor, apply_optics, optics_field
from irsim.optics.stage import shutter_flux
from irsim.pipeline import PipelineConfig, PipelineState, attach_sensor_chain, run_frame
from irsim.radiometry.lut import BandLUT

REPO = pathlib.Path(__file__).resolve().parents[2]
BOSON = yaml.safe_load((REPO / "configs" / "sensors" / "flir_boson_640_lwir.yaml").read_text())
SHAPE = (48, 64)


def _pipeline(lut: BandLUT, **optics: object) -> PipelineConfig:
    d = copy.deepcopy(BOSON)
    d["sensor"]["fpa"].update(width=SHAPE[1], height=SHAPE[0])
    # a short lens so the small array spans a real field (corner cos⁴ ≈ 0.8)
    d["sensor"]["optics"].update({"supersample_factor": 1, "focal_length_mm": 1.0, **optics})
    return PipelineConfig.from_sensor(
        SensorConfig.model_validate(d),
        MaterialTable.constant(1.0),
        lut=lut,
        noise_enabled=False,
        sensor_seed=7,
    )


def _dn(cfg: PipelineConfig, t_scene: float, t_housing: float) -> np.ndarray:
    s: SensorSpec = cfg.sensor.sensor
    lb = lambda t: float(cfg.lut.lookup(np.float64(t), cfg.quantity)[()])  # noqa: E731
    flux = apply_optics(np.full(SHAPE, lb(t_scene)), s, lb(t_housing), supersample=1)
    return np.asarray(cfg.detector.noiseless_signal_dn(flux), dtype=np.float64)


def _shutter_dn(cfg: PipelineConfig, t_shutter: float, t_housing: float) -> np.ndarray:
    lb = lambda t: float(cfg.lut.lookup(np.float64(t), cfg.quantity)[()])  # noqa: E731
    flux = shutter_flux(cfg.sensor.sensor, lb(t_shutter), lb(t_housing))
    return np.asarray(cfg.detector.noiseless_signal_dn(flux), dtype=np.float64)


def _factory(cfg: PipelineConfig) -> TwoPointNuc:
    t_cal = cfg.t_housing_cal_k
    return TwoPointNuc.calibrate(
        _dn(cfg, 233.0, t_cal), _dn(cfg, 473.0, t_cal), restore_pedestal=True
    )


# --- the operator ---------------------------------------------------------------------------


def test_a_refreshed_offset_makes_the_shutter_frame_uniform() -> None:
    rng = np.random.default_rng(1)
    lo = 1000.0 + rng.normal(0.0, 30.0, SHAPE)
    hi = lo + 4000.0 * (1.0 + rng.normal(0.0, 0.05, SHAPE))
    nuc = TwoPointNuc.calibrate(lo, hi, restore_pedestal=True)
    shutter = 2500.0 + rng.normal(0.0, 40.0, SHAPE) + np.linspace(0.0, 60.0, SHAPE[1])
    fresh = nuc.refreshed(shutter)
    out = fresh.apply(shutter).astype(np.float64)
    assert np.ptp(out) < 1e-2
    assert out.mean() == pytest.approx(
        float(nuc.apply(shutter).astype(np.float64).mean()), abs=1e-2
    )
    assert np.array_equal(fresh.gain, nuc.gain) and fresh.pedestal == nuc.pedestal


def test_the_shutter_flux_is_the_scene_form_with_no_lens() -> None:
    """Φ_sh = A_d Ω_eff [L_h + RI (L_sh − L_h)]: flat when the shutter is at the housing
    temperature, and A_d Ω_eff L_sh on axis whatever τ_opt is."""
    d = copy.deepcopy(BOSON)
    d["sensor"]["fpa"].update(width=SHAPE[1], height=SHAPE[0])
    d["sensor"]["optics"].update(focal_length_mm=1.0)
    s = SensorConfig.model_validate(d).sensor
    same = shutter_flux(s, 5.0, 5.0).astype(np.float64)
    assert np.ptp(same) / same.mean() < 1e-6
    ri = optics_field(s).astype(np.float64)
    phi = shutter_flux(s, 4.0, 6.0).astype(np.float64)
    axis = s.detector_active_area_m2 * aperture_factor(s.optics.f_number)
    np.testing.assert_allclose(phi, axis * (6.0 + ri * (4.0 - 6.0)), rtol=1e-6)


# --- the bowl -------------------------------------------------------------------------------


def _corrected(cfg: PipelineConfig, t_scene: float, t_ffc_housing: float, t_housing: float):  # type: ignore[no-untyped-def]
    nuc = _factory(cfg).refreshed(_shutter_dn(cfg, 300.0, t_ffc_housing))
    return nuc.apply(_dn(cfg, t_scene, t_housing)).astype(np.float64)


def test_the_shutter_removes_the_housing_mismatch_the_factory_left(boson_lut: BandLUT) -> None:
    """At the event, a housing 4 K off its calibration value shades a factory-corrected frame;
    the refreshed offset makes the same frame flat."""
    cfg = _pipeline(boson_lut)
    t_h = cfg.t_housing_cal_k - 4.0
    factory = _factory(cfg).apply(_dn(cfg, 260.0, t_h)).astype(np.float64)
    fresh = _corrected(cfg, 260.0, t_h, t_h)
    assert np.ptp(factory) > 5.0, np.ptp(factory)
    assert np.ptp(fresh) < 0.05 * np.ptp(factory), (np.ptp(fresh), np.ptp(factory))


def test_a_housing_that_cooled_since_the_event_leaves_a_bright_centre(boson_lut: BandLUT) -> None:
    """The §11.2 middle term, to 5 %: corner − centre = (1/τ)(1/RI_c − 1/RI_0) ΔL_h, in
    scene-radiance units, and negative for a housing 2 K cooler than when the shutter closed."""
    cfg = _pipeline(boson_lut)
    s = cfg.sensor.sensor
    tau = s.optics.transmittance
    lb = lambda t: float(cfg.lut.lookup(np.float64(t), cfg.quantity)[()])  # noqa: E731
    t1, t2 = 298.0, 296.0
    img = _corrected(cfg, 260.0, t1, t2)
    # the display's DN per unit scene radiance, measured, so the test holds in any units
    slope = (_corrected(cfg, 280.0, t1, t2).mean() - img.mean()) / (lb(280.0) - lb(260.0))
    err = img / slope

    ri = optics_field(s).astype(np.float64)
    h, w = SHAPE
    c = (slice(h // 2 - 1, h // 2 + 1), slice(w // 2 - 1, w // 2 + 1))
    expected = (1.0 / tau) * (1.0 / ri[0, 0] - 1.0 / ri[c].mean()) * (lb(t2) - lb(t1))
    got = err[0, 0] - err[c].mean()
    assert got < 0.0
    assert got == pytest.approx(expected, rel=0.05), (got, expected)

    # monotonic in radius: the azimuthal mean falls from the centre outward
    yy, xx = np.mgrid[:h, :w]
    r = np.hypot(xx + 0.5 - w / 2, yy + 0.5 - h / 2)
    edges = np.linspace(0.0, r.max() + 1e-9, 8)
    prof = [err[(r >= a) & (r < b)].mean() for a, b in zip(edges[:-1], edges[1:], strict=True)]
    assert all(b < a for a, b in zip(prof[:-1], prof[1:], strict=True)), prof


# --- the gain residual ----------------------------------------------------------------------


def test_the_gain_residual_scales_with_scene_minus_shutter(boson_lut: BandLUT) -> None:
    cfg = _pipeline(boson_lut)
    nuc_spec = cfg.sensor.sensor.nuc.model_copy(update={"residual_offset_mk_per_k": 0.0})
    res = NucResidual(nuc=nuc_spec, shape=SHAPE, dn_per_k=100.0, sensor_seed=3)
    ref = _shutter_dn(cfg, 300.0, 300.0).astype(np.float32)
    # a scene exactly at the shutter's signal carries no gain residual, at any ΔT
    np.testing.assert_array_equal(res.apply(ref, 2.0, ref), ref)

    def spread(t_scene: float) -> float:
        x = _dn(cfg, t_scene, 300.0).astype(np.float32)
        return float(np.std(res.apply(x, 2.0, ref).astype(np.float64) - x))

    sky, room = spread(250.0), spread(300.0)
    assert sky > 10.0 * room, (sky, room)
    # the pre-SC.18 form multiplied the whole signal, so the room scene carried the most
    x = _dn(cfg, 300.0, 300.0).astype(np.float32)
    assert float(np.std(res.apply(x, 2.0).astype(np.float64) - x)) > 10.0 * room


# --- end to end -----------------------------------------------------------------------------


def test_the_camera_flat_fields_on_its_shutter_at_power_up(boson_lut: BandLUT) -> None:
    d = copy.deepcopy(BOSON)
    d["sensor"]["fpa"].update(width=SHAPE[1], height=SHAPE[0])
    d["sensor"]["optics"].update(supersample_factor=1, focal_length_mm=1.0)
    cfg = PipelineConfig.from_sensor(
        SensorConfig.model_validate(d),
        MaterialTable.constant(1.0),
        lut=boson_lut,
        noise_enabled=False,
        flat_field_enabled=True,
        sensor_seed=7,
    )
    cfg = attach_sensor_chain(
        cfg,
        ambient_provider=lambda t: 285.0,  # the housing settles far from its calibration value
        defects_enabled=False,
        residual_enabled=False,
    )
    planes = {
        "temperature_k": np.full(SHAPE, 250.0, dtype=np.float32),
        "material_id": np.ones(SHAPE, dtype=np.int32),
        "distance_m": np.zeros(SHAPE, dtype=np.float32),
    }
    state = PipelineState()
    assert cfg.chain is not None and cfg.chain.shutter_dn is None
    out = run_frame(planes, cfg, state)
    assert cfg.chain.shutter_dn is not None and cfg.chain.display_nuc is not None
    assert abs(state.housing_temp_k - cfg.t_housing_cal_k) > 5.0
    dn16 = np.asarray(out.dn16, dtype=np.float64)
    factory = cfg.flat_field.apply(dn16).astype(np.float64)  # type: ignore[union-attr]
    fresh = cfg.chain.display_nuc.apply(dn16).astype(np.float64)
    assert np.ptp(fresh) < 0.1 * np.ptp(factory), (np.ptp(fresh), np.ptp(factory))
