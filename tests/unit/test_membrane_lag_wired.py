"""The bolometer membrane IIR on `run_frame`'s own path (M9.13, §9.2, ADR 0052).

`BolometerLowPass` has existed since M9.1 and `irsim.pipeline.detector.detector_stage` has driven
it since, but **`run_frame` never called either**. Its stage 4 is `_detector_signal`, which reaches
`Detector.response(flux, frame_index, sensor_seed)` -- a signature with no `dt` and no state, so it
could not have carried a lag whatever it wanted to. Every frame sequence this simulator has
produced therefore came from a bolometer with a thermal time constant of zero: no trail behind a
moving target, and §15 Tier 3's "lateral motion smears LWIR, not cooled MWIR" true only in the
within-frame half ADR 0077 supplied.

The documentation asserted the opposite, which is why this went unseen: `sensor_chain`'s own
module docstring lists stage 4 as "detector + membrane IIR", and M9.8's roadmap title is "wire IIR,
FPA node, housing, drift, defects, NUC residual, FFC into run_frame". Six of the seven landed.

These tests are therefore mostly *identities against the closed form*, not tolerances: the point is
not that frames now lag but that they lag by exactly the amount §9.2 specifies, and that the photon
path still does not.
"""

from __future__ import annotations

import copy
import math
import pathlib
from typing import Any

import numpy as np
import pytest
import yaml

from irsim.config.sensor import SensorConfig
from irsim.materials.table import MaterialTable
from irsim.pipeline import PipelineConfig, PipelineState, run_frame
from irsim.pipeline.detector import IIR_STATE_KEY, LAST_T_KEY, lag_interval_s
from irsim.radiometry.lut import BandLUT

REPO = pathlib.Path(__file__).resolve().parents[2]
BOSON = yaml.safe_load((REPO / "configs" / "sensors" / "flir_boson_640_lwir.yaml").read_text())

SHAPE = (16, 20)
#: Read off the committed config, not written here, so a correction to it moves these tests
#: rather than leaving them asserting a detector the repository no longer models. SC.3 took it
#: from an ESTIMATED 10 ms to [R24]'s nominal 8 ms.
TAU_S = float(BOSON["sensor"]["fpa"]["thermal_time_constant_ms"]) * 1e-3
FPS = 60.0
DT_S = 1.0 / FPS
#: exp(-dt/tau) at 60 Hz on the 8 ms membrane: the ratio of successive step-response residuals.
DECAY = math.exp(-DT_S / TAU_S)
#: 1 - exp(-dt/tau): the fraction of a step the first frame shows. §9.2's "0.6 frames" phrase is
#: tau/dt and not this (spec issue S8).
ALPHA = 1.0 - DECAY


def _config(lut: BandLUT, *, photon: bool = False, **over: Any) -> PipelineConfig:
    d = copy.deepcopy(BOSON)
    d["sensor"]["fpa"].update(width=SHAPE[1], height=SHAPE[0], frame_rate_hz=FPS)
    d["sensor"]["optics"].update(supersample_factor=1)
    if photon:
        d["sensor"]["fpa"].update(
            type="photon",
            quantum_efficiency=0.7,
            well_capacity_e=5e6,
            integration_time_ms=4.0,
            dark_current_model="fixed",
            thermal_time_constant_ms=None,
            tcr_per_k=None,
            g_th_w_per_k=None,
        )
    for block, updates in over.items():
        d["sensor"][block].update(updates)
    return PipelineConfig.from_sensor(
        SensorConfig.model_validate(d),
        MaterialTable.constant(1.0),
        lut=lut,
        noise_enabled=False,
        sensor_seed=20260914,
    )


def _planes(t_k: float) -> dict[str, np.ndarray]:
    return {
        "temperature_k": np.full(SHAPE, t_k, dtype=np.float32),
        "material_id": np.ones(SHAPE, dtype=np.int32),
        "distance_m": np.zeros(SHAPE, dtype=np.float32),
    }


def _step_sequence(
    config: PipelineConfig, n: int, *, settle: int = 6, cold: float = 300.0, hot: float = 310.0
) -> list[float]:
    """Mean stage-4 signal for ``n`` frames after a cold→hot step, from a settled start."""
    state = PipelineState(housing_temp_k=config.t_housing_cal_k)
    for k in range(settle):
        state.t_s = k * DT_S
        run_frame(_planes(cold), config, state)
    out = []
    for k in range(n):
        state.t_s = (settle + k) * DT_S
        frame = run_frame(_planes(hot), config, state)
        assert frame.signal_dn is not None
        out.append(float(np.asarray(frame.signal_dn).mean()))
    return out


# --------------------------------------------------------------------------------------------
# The step response, against the closed form
# --------------------------------------------------------------------------------------------


def test_the_step_response_is_the_closed_form_in_signal_space(tophat_lwir_lut: BandLUT) -> None:
    """(S_k − S_0)/(S_∞ − S_0) = 1 − e^(−(k+1)·dt/τ), to 1e-6.

    In **signal** space, where the IIR actually runs. In apparent temperature the same sequence is
    a little different because L(T) is not linear, which is the reason this assertion is not
    written on the Kelvin plane.
    """
    config = _config(tophat_lwir_lut)
    cold = _step_sequence(config, 1, settle=6, cold=300.0, hot=300.0)[0]
    seq = _step_sequence(config, 12)
    final = seq[-1]
    for k, value in enumerate(seq[:8]):
        expected = 1.0 - math.exp(-(k + 1) * DT_S / TAU_S)
        assert (value - cold) / (final - cold) == pytest.approx(expected, abs=1e-6)


def test_the_first_frame_shows_exactly_alpha_of_the_step(tophat_lwir_lut: BandLUT) -> None:
    """α = 1 − e^(−dt/τ) = 0.8111 at 60 Hz on a 10 ms membrane."""
    config = _config(tophat_lwir_lut)
    cold = _step_sequence(config, 1, cold=300.0, hot=300.0)[0]
    seq = _step_sequence(config, 12)
    # 8 ms at 60 Hz. It was 0.811124 at the 10 ms this project carried as ESTIMATED, so the
    # corrected detector is *faster* -- it shows more of a step in its first frame, not less.
    assert pytest.approx(8.0e-3) == TAU_S
    assert pytest.approx(0.875486, abs=1e-6) == ALPHA
    assert (seq[0] - cold) / (seq[-1] - cold) == pytest.approx(ALPHA, abs=1e-6)


def test_successive_residuals_fall_by_exactly_the_decay(tophat_lwir_lut: BandLUT) -> None:
    """The trailing profile is a geometric series with ratio e^(−dt/τ) — §15 Tier 3's tail.

    The tolerance is **derived, not chosen**. The signal plane is float32 and each residual is a
    difference of two numbers near 10 120 DN, so by the fourth term the residual is ~2 DN and
    carries only three significant figures: a fixed `rel=1e-4` passes on the first two ratios and
    fails on the fourth for no physical reason. Scaling by float32's own precision on each residual
    keeps the whole tail under test and says why the later terms are looser.
    """
    seq = _step_sequence(_config(tophat_lwir_lut), 14)
    final = seq[-1]
    residuals = [final - v for v in seq[:6]]
    eps = float(np.finfo(np.float32).eps)
    for i in range(4):
        ratio = residuals[i + 1] / residuals[i]
        tolerance = max(5e-5, 8.0 * eps * abs(final) / abs(residuals[i + 1]))
        assert ratio == pytest.approx(DECAY, rel=tolerance)
    # The first term is where float32 still has room: measured 1.1e-5 relative, against a
    # fourth-term floor two decades looser.
    assert residuals[1] / residuals[0] == pytest.approx(DECAY, rel=5e-5)


def test_a_cooled_photon_detector_has_no_tail_at_all(tophat_lwir_lut: BandLUT) -> None:
    """The other half of "lateral motion smears LWIR, not cooled MWIR": it settles in one frame."""
    seq = _step_sequence(_config(tophat_lwir_lut, photon=True), 6)
    assert seq[0] == pytest.approx(seq[-1], rel=1e-9)


def test_a_longer_time_constant_lags_more(tophat_lwir_lut: BandLUT) -> None:
    """Guard: the lag is driven by the configured τ and not by a constant."""
    fast = _step_sequence(_config(tophat_lwir_lut, fpa={"thermal_time_constant_ms": 2.0}), 4)
    slow = _step_sequence(_config(tophat_lwir_lut, fpa={"thermal_time_constant_ms": 40.0}), 4)
    cold = _step_sequence(_config(tophat_lwir_lut), 1, cold=300.0, hot=300.0)[0]
    assert (fast[0] - cold) > (slow[0] - cold)
    assert (fast[0] - cold) / (fast[-1] - cold) == pytest.approx(
        1.0 - math.exp(-DT_S / 2.0e-3), abs=1e-6
    )


# --------------------------------------------------------------------------------------------
# What must not have changed
# --------------------------------------------------------------------------------------------


def test_the_first_frame_of_a_sequence_adopts_its_input(tophat_lwir_lut: BandLUT) -> None:
    """A core staring at a scene is already in equilibrium with it, so a single frame is unlagged.

    This is what kept every single-frame golden bit-identical when the lag was wired in: starting
    the membrane from zero would put a frame-long ramp at the head of every exported sequence.
    """
    config = _config(tophat_lwir_lut)
    one = run_frame(_planes(310.0), config, PipelineState(housing_temp_k=config.t_housing_cal_k))
    settled = _step_sequence(config, 20)[-1]
    assert one.signal_dn is not None
    assert float(np.asarray(one.signal_dn).mean()) == pytest.approx(settled, rel=1e-9)


def test_a_still_scene_is_unaffected_however_many_frames_run(tophat_lwir_lut: BandLUT) -> None:
    """Nothing moving, nothing to lag: the IIR is the identity on a constant input."""
    config = _config(tophat_lwir_lut)
    state = PipelineState(housing_temp_k=config.t_housing_cal_k)
    frames = []
    for k in range(8):
        state.t_s = k * DT_S
        out = run_frame(_planes(300.0), config, state)
        assert out.radiance is not None
        frames.append(np.asarray(out.radiance).copy())
    for f in frames[1:]:
        assert np.array_equal(f, frames[0])


def test_the_state_lives_in_the_pipeline_buffers(tophat_lwir_lut: BandLUT) -> None:
    """ADR 0052's single-owner rule: the membrane state belongs to `PipelineState.buffers`."""
    config = _config(tophat_lwir_lut)
    state = PipelineState(housing_temp_k=config.t_housing_cal_k)
    assert IIR_STATE_KEY not in state.buffers
    run_frame(_planes(300.0), config, state)
    assert state.buffers[IIR_STATE_KEY].dtype == np.float32
    assert state.buffers[IIR_STATE_KEY].shape == SHAPE


def test_a_photon_detector_keeps_no_membrane_state(tophat_lwir_lut: BandLUT) -> None:
    config = _config(tophat_lwir_lut, photon=True)
    state = PipelineState(housing_temp_k=config.t_housing_cal_k)
    run_frame(_planes(300.0), config, state)
    assert IIR_STATE_KEY not in state.buffers


# --------------------------------------------------------------------------------------------
# The interval, which a time-lapse gets wrong
# --------------------------------------------------------------------------------------------


def test_the_interval_falls_back_to_the_frame_rate(tophat_lwir_lut: BandLUT) -> None:
    """A caller that never advances `t_s` gets the detector's own frame period."""
    config = _config(tophat_lwir_lut)
    state = PipelineState()
    assert lag_interval_s(config, state) == pytest.approx(DT_S)
    assert lag_interval_s(config, state) == pytest.approx(DT_S)  # elapsed 0 -> still the fallback


def test_a_time_lapse_is_told_its_real_interval(tophat_lwir_lut: BandLUT) -> None:
    """ADR 0074 films one frame per six seconds of scene time, and the membrane must know.

    A 10 ms membrane settles completely across six seconds. Driving it at 1/60 s instead would
    leave `e^(−dt/τ)` = **19 %** of a scene six seconds old in the picture — a ghost, and a
    flattering one, since it smooths exactly the change being filmed.
    """
    config = _config(tophat_lwir_lut)
    state = PipelineState()
    lag_interval_s(config, state)
    state.t_s = 6.0
    assert lag_interval_s(config, state) == pytest.approx(6.0)
    assert state.buffers[LAST_T_KEY] == pytest.approx(6.0)


def test_a_six_second_gap_settles_completely(tophat_lwir_lut: BandLUT) -> None:
    """End to end: the same step, filmed as a time-lapse, arrives whole in one frame."""
    config = _config(tophat_lwir_lut)
    state = PipelineState(housing_temp_k=config.t_housing_cal_k)
    state.t_s = 0.0
    run_frame(_planes(300.0), config, state)
    state.t_s = 6.0
    lapsed = run_frame(_planes(310.0), config, state)
    state.t_s = 12.0
    settled = run_frame(_planes(310.0), config, state)
    assert lapsed.signal_dn is not None and settled.signal_dn is not None
    assert float(np.asarray(lapsed.signal_dn).mean()) == pytest.approx(
        float(np.asarray(settled.signal_dn).mean()), rel=1e-9
    )


# --------------------------------------------------------------------------------------------
# Order: the lag is before the noise (ADR 0052)
# --------------------------------------------------------------------------------------------


def test_the_lag_does_not_filter_the_noise(tophat_lwir_lut: BandLUT) -> None:
    """Filtering *after* noise would cut the per-frame temporal variance by α/(2 − α) ≈ 0.68.

    ADR 0052 quantifies it and this is the check that the order survived being wired in: on a
    settled uniform scene the frame-to-frame temporal standard deviation must be the detector's
    own σ, not a filtered version of it, or the M4.6 NETD anchor is silently broken.
    """
    config = _config(tophat_lwir_lut)
    noisy = PipelineConfig.from_sensor(
        config.sensor,
        MaterialTable.constant(1.0),
        lut=tophat_lwir_lut,
        noise_enabled=True,
        sensor_seed=20260914,
    )
    state = PipelineState(housing_temp_k=noisy.t_housing_cal_k)
    series = []
    for k in range(240):
        state.t_s = k * DT_S
        out = run_frame(_planes(300.0), noisy, state)
        assert out.signal_dn is not None
        series.append(np.asarray(out.signal_dn).copy())
    stack = np.stack(series[40:])
    measured = float(stack.std(axis=0).mean())
    sigma = float(noisy.detector.sigma_signal_dn)
    filtered = sigma * math.sqrt(ALPHA / (2.0 - ALPHA))
    assert measured == pytest.approx(sigma, rel=0.05)
    assert abs(measured - sigma) < abs(measured - filtered)
