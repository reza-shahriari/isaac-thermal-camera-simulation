"""SC.31 -- the information histogram sees detail, not noise.

`information.py` weights histogram bins by the summed |high-pass| of a 5×5 bilateral residual.
On a noisy frame that residual is mostly temporal noise, so the weighting is the plain histogram
in disguise and a small target's transition bins get nothing: measured on a 60 K clear-sky
gradient at σ_TVH = 8.5 DN, a 3 px target at 0 °C fell from 132 codes of contrast noise-free to
70. `detail_threshold_sigma` gates the detail measure at k × σ_TVH -- the detector's own temporal
noise in DN, handed to the display branch by the pipeline (`PipelineState.sigma_tvh_dn`), never
estimated from the frame, which quantisation and a sky's curvature both fool. Gated at the
Boson's 3 σ the target is 132 again, a target outside the sky's range or of one pixel moves by a
few codes, a noise-free frame is bit-identical (its σ is 0) and k = 0 is the old operator bit for
bit. Roadmap SC.31; ADR 0152 (note updated).
"""

from __future__ import annotations

import copy
import pathlib

import numpy as np
import pytest
import yaml

from irsim.config.loader import load_sensor_config
from irsim.config.sensor import ISP_OPTIONAL_DEFAULTS, IspSpec, SensorConfig
from irsim.isp.display import isp_config_hash, run_display_branch
from irsim.isp.information import agc_information
from irsim.materials.table import MaterialTable
from irsim.pipeline import PipelineConfig, PipelineState, run_frame

REPO = pathlib.Path(__file__).resolve().parents[2]
BOSON_YAML = REPO / "configs" / "sensors" / "flir_boson_640_lwir.yaml"
DN_PER_K = 170.0  # the Boson near 300 K, as test_agc_information measures it
FRAME = (512, 640)
SIGMA_DN = 8.5  # 50 mK of temporal noise at 170 DN/K


def _clear_sky(noise_dn: float, size_px: int, span_c: float, t_target_c: float, seed: int = 0):  # type: ignore[no-untyped-def]
    """A clear sky ``span_c`` kelvin colder at the top than at the horizon, ``noise_dn`` of
    temporal noise, and a ``size_px``-square target at ``t_target_c`` in the middle -- a drone at
    standoff is 1 to 9 of 327 680 pixels."""
    rng = np.random.default_rng(seed)
    height, width = FRAME
    yy = np.arange(height, dtype=np.float64)[:, None]
    t_c = np.broadcast_to(-47.0 + span_c * yy / (height - 1), FRAME).copy()
    target = np.zeros(FRAME, dtype=bool)
    r0, c0 = 256 - size_px // 2, 320 - size_px // 2
    target[r0 : r0 + size_px, c0 : c0 + size_px] = True
    t_c[target] = t_target_c
    dn = (t_c + 60.0) * DN_PER_K + rng.normal(0.0, noise_dn, FRAME)
    return np.clip(np.round(dn), 0, 2**16 - 1).astype(np.uint16), target


def _codes(isp: IspSpec, dn: np.ndarray, sigma_dn: float) -> np.ndarray:
    return (
        run_display_branch(dn, isp, 16, noise_sigma_dn=sigma_dn).display8[..., 0].astype(np.int64)
    )


def _contrast(codes: np.ndarray, target: np.ndarray) -> float:
    """Median target code minus the median of the 15 × 15 ring of sky around it."""
    ring = np.zeros(FRAME, dtype=bool)
    ring[249:264, 313:328] = True
    ring &= ~target
    return float(np.median(codes[target]) - np.median(codes[ring]))


@pytest.fixture(scope="module")
def boson() -> IspSpec:
    return load_sensor_config("boson640").sensor.isp


def test_the_gate_is_off_by_default_and_the_boson_sets_three_sigma(boson: IspSpec) -> None:
    assert ISP_OPTIONAL_DEFAULTS["detail_threshold_sigma"] == 0.0
    assert IspSpec.model_fields["detail_threshold_sigma"].default == 0.0
    assert boson.detail_threshold_sigma == 3.0
    plain = boson.model_copy(update={"detail_threshold_sigma": 0.0})
    older = IspSpec.model_validate(
        {k: v for k, v in plain.model_dump().items() if k != "detail_threshold_sigma"}
    )
    assert isp_config_hash(plain, 16) == isp_config_hash(older, 16), "0 leaves every hash alone"
    assert isp_config_hash(boson, 16) != isp_config_hash(plain, 16)


def test_a_three_pixel_target_keeps_its_codes_at_the_configured_netd(boson: IspSpec) -> None:
    """The defect, measured, and its repair: on the 60 K sky the noise-free operator gives the
    3 px target at 0 °C 132 codes over its ring and the noisy one 70; gated at 3 σ_TVH it is the
    noise-free figure again. A 1 px target and one hotter than the whole sky, whose codes never
    depended on the weighting, move by a few codes at most."""
    ungated = boson.model_copy(update={"detail_threshold_sigma": 0.0})
    clean, target = _clear_sky(0.0, 3, 60.0, 0.0)
    noisy, _ = _clear_sky(SIGMA_DN, 3, 60.0, 0.0)
    reference = _contrast(_codes(ungated, clean, 0.0), target)
    before = _contrast(_codes(ungated, noisy, SIGMA_DN), target)
    after = _contrast(_codes(boson, noisy, SIGMA_DN), target)
    assert reference >= 120.0, reference
    assert before < 0.6 * reference, (before, reference)  # the defect: half the codes gone
    assert after >= 120.0 and abs(after - reference) <= 6.0, (before, after, reference)
    for size, t_c in ((1, 0.0), (1, 20.0), (3, 20.0)):
        noisy, target = _clear_sky(SIGMA_DN, size, 60.0, t_c)
        a = _contrast(_codes(ungated, noisy, SIGMA_DN), target)
        b = _contrast(_codes(boson, noisy, SIGMA_DN), target)
        assert a >= 40.0 and b >= 40.0 and abs(a - b) <= 6.0, (size, t_c, a, b)


def test_a_noise_free_frame_is_bit_identical_and_zero_is_the_old_operator(boson: IspSpec) -> None:
    """σ_TVH = 0 (noise off) switches the gate off whatever k says; k = 0 is the old operator
    however noisy the frame; the quantisation residual of a clean frame is never read as noise."""
    ungated = boson.model_copy(update={"detail_threshold_sigma": 0.0})
    for span in (20.0, 60.0):
        clean, _ = _clear_sky(0.0, 3, span, 0.0)
        assert np.array_equal(_codes(ungated, clean, 0.0), _codes(boson, clean, 0.0))
    noisy, _ = _clear_sky(SIGMA_DN, 3, 60.0, 0.0)
    y0 = agc_information(noisy, 0.07, 16, linear_percent=0.2)
    y1 = agc_information(
        noisy, 0.07, 16, linear_percent=0.2, detail_threshold_sigma=0.0, noise_sigma_dn=SIGMA_DN
    )
    assert np.array_equal(y0, y1)
    y2 = agc_information(
        noisy, 0.07, 16, linear_percent=0.2, detail_threshold_sigma=3.0, noise_sigma_dn=SIGMA_DN
    )
    assert not np.array_equal(y0, y2), "the gate does act when both are set"
    with pytest.raises(ValueError, match="detail_threshold_sigma"):
        agc_information(noisy, 0.07, 16, detail_threshold_sigma=-1.0)
    with pytest.raises(ValueError, match="noise_sigma_dn"):
        agc_information(noisy, 0.07, 16, noise_sigma_dn=-1.0)


def _pipeline(lut, noise_enabled: bool, k: float) -> PipelineConfig:  # type: ignore[no-untyped-def]
    d = copy.deepcopy(yaml.safe_load(BOSON_YAML.read_text()))
    d["sensor"]["fpa"].update(width=80, height=64)
    d["sensor"]["optics"].update(supersample_factor=1)
    d["sensor"]["isp"].update(detail_threshold_sigma=k)
    sensor = SensorConfig.model_validate(d)
    return PipelineConfig.from_sensor(
        sensor, MaterialTable.constant(1.0), lut=lut, sensor_seed=7, noise_enabled=noise_enabled
    )


def test_the_pipeline_hands_the_display_the_detector_s_own_sigma(boson_lut) -> None:  # type: ignore[no-untyped-def]
    """`PipelineState.sigma_tvh_dn` is the mean per-pixel σ the detector put into the frame:
    about 50 mK × ∂DN/∂T for the Boson, and exactly 0 with noise off -- where the gated and the
    ungated camera then render the same display bit for bit."""
    shape = (64, 80)
    planes = {
        "temperature_k": np.linspace(280.0, 320.0, 80, dtype=np.float32)[None, :].repeat(64, 0),
        "material_id": np.ones(shape, dtype=np.int32),
        "distance_m": np.zeros(shape, dtype=np.float32),
    }
    noisy = run_frame(planes, _pipeline(boson_lut, True, 3.0), state := PipelineState())
    assert 4.0 < state.sigma_tvh_dn < 20.0, state.sigma_tvh_dn  # 50 mK at ~170 DN/K
    assert noisy.display8 is not None
    outs = []
    for k in (0.0, 3.0):
        state = PipelineState()
        outs.append(run_frame(planes, _pipeline(boson_lut, False, k), state))
        assert state.sigma_tvh_dn == 0.0
    assert np.array_equal(outs[0].display8, outs[1].display8)
