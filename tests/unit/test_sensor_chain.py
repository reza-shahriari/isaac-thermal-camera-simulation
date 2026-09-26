"""The wired M9 sensor chain (M9.8): the noise budget, and the order of §11.1.

The headline test is ``test_the_spatial_noise_budget_is_two_mechanisms_not_four``. Every M9
mechanism makes the image drift or speckle, so the real hazard in assembling them is not that one
is wrong but that two describe the same physics. On a uniform scene 179 s after an FFC the
post-correction spatial noise must be

    sqrt(sigma_V^2 + sigma_H^2 + sigma_VH^2 + sigma_residual^2)

within 10 % -- two mechanisms in quadrature, not four added up. A chain that double-counted would
still produce an entirely plausible thermal image, which is why this is a test and not an
inspection.

The rest pins the order and the couplings: the housing node drives the self-emission, the FPA node
drives the residual through the FFC controller's ΔT_eff, the drifting pattern reaches stage 5, the
defects land before replacement, and a frozen frame is stale in *every* output rather than only in
the picture.

docs/physics-model.md §11.1, §11.2, §10.3, §10.4. ADR 0058.
"""

from __future__ import annotations

import copy
import pathlib
from typing import Any

import numpy as np
import pytest
import yaml

from irsim.config.sensor import SensorConfig
from irsim.materials.table import MaterialTable
from irsim.noise.defects import DefectKind
from irsim.pipeline import PipelineConfig, PipelineState, attach_sensor_chain, run_frame
from irsim.radiometry.lut import BandLUT

REPO = pathlib.Path(__file__).resolve().parents[2]
BOSON = yaml.safe_load((REPO / "configs" / "sensors" / "flir_boson_640_lwir.yaml").read_text())
SEED = 20260912
SHAPE = (64, 80)
#: These tests step real seconds -- 179 s of drift, a whole FFC interval -- so the camera is run
#: at 6 Hz rather than 60. Every quantity here is a function of elapsed *time*, not of frame
#: count, so the physics is unchanged and the suite stays inside its 30 s budget. The one thing
#: that is frame-quantised, the FFC freeze, is checked at the real 60 Hz in test_ffc_controller.
TEST_FPS = 6.0


#: A focal-plane node that responds on the timescale of these tests. The committed Boson has no
#: `fpa_temp_mode` at all, so T_FPA falls back to the housing -- whose 900 s body lag damps a
#: 179 s drive down to a tenth of the air's excursion. That fallback is right for a camera, and
#: wrong for a test that needs the residual to have grown, so these configs give the FPA its own
#: (much shorter) time constant, which is what a real focal plane has relative to its housing.
FAST_FPA = {"fpa_temp_mode": "coupled", "fpa_tau_s": 20.0, "fpa_self_heating_k": 2.0}


def _config(lut: BandLUT, **over: Any) -> PipelineConfig:
    d = copy.deepcopy(BOSON)
    d["sensor"]["fpa"].update(width=SHAPE[1], height=SHAPE[0], frame_rate_hz=TEST_FPS)
    d["sensor"]["optics"].update(supersample_factor=1)
    for block, updates in over.items():
        d["sensor"][block].update(updates)
    sensor = SensorConfig.model_validate(d)
    materials = MaterialTable.constant(1.0)
    return PipelineConfig.from_sensor(sensor, materials, lut=lut, sensor_seed=SEED)


def _uniform_planes(t_k: float = 300.0) -> dict[str, np.ndarray]:
    return {
        "temperature_k": np.full(SHAPE, t_k, dtype=np.float32),
        "material_id": np.ones(SHAPE, dtype=np.int32),
        "distance_m": np.zeros(SHAPE, dtype=np.float32),
    }


def _run_to(config: PipelineConfig, seconds: float, t_k: float = 300.0):  # type: ignore[no-untyped-def]
    """Run frames at the sensor's own rate up to ``seconds``; return the last Outputs."""
    fps = config.sensor.sensor.fpa.frame_rate_hz
    n = int(round(seconds * fps))
    state = PipelineState()
    planes = _uniform_planes(t_k)
    out = None
    for _ in range(n):
        state.t_s = state.frame_index / fps
        out = run_frame(planes, config, state)
    assert out is not None
    return out, state


# -- the budget ------------------------------------------------------------------------------


def _flat_field_reference(config: PipelineConfig) -> np.ndarray:
    """The noiseless response of the same camera to the same uniform scene.

    Fixed-pattern noise is a *deviation from the expected response*, and the expected response is
    not flat: `vignetting_cos4` alone puts a 0.9 % cos⁴ falloff across this array, which is a
    larger spatial standard deviation than the entire noise budget. Measuring the raw std of a
    frame would therefore be measuring the lens. A real NUC measurement subtracts a flat-field
    reference for exactly this reason, and so does this one.
    """
    from dataclasses import replace

    ideal = replace(config, chain=None, noise_enabled=False)
    state = PipelineState()
    return np.asarray(run_frame(_uniform_planes(), ideal, state).signal_dn, dtype=np.float64)


def _spatial_std(config: PipelineConfig, state: PipelineState, n_frames: int = 100) -> float:
    """Spatial standard deviation of a frame-averaged plane, the way FPN is actually measured.

    Two corrections to a naive std, both necessary. The flat-field reference is subtracted, so the
    lens is not counted as noise. And frames are averaged: a single frame's spatial std also
    contains the *temporal* per-pixel noise, which in one frame is indistinguishable from fixed
    pattern. Averaging 100 frames divides the temporal terms by about ten and leaves the fixed
    pattern and the NUC residual, which is what the budget is about. Over 100 frames at 60 Hz the
    drift (tau = 120 s) and ΔT_FPA each move by about a per cent, so the thing measured is still
    the thing budgeted.
    """
    reference = _flat_field_reference(config)
    fps = config.sensor.sensor.fpa.frame_rate_hz
    planes = _uniform_planes()
    total = None
    for _ in range(n_frames):
        state.t_s = state.frame_index / fps
        out = run_frame(planes, config, state)
        # An FFC inside the window would average two different residual levels together, and a
        # freeze would stop the temporal noise averaging down at all. Both would quietly
        # invalidate the measurement, so refuse rather than report a number.
        assert not out.report.ffc.fired, "an FFC fired inside the measurement window"
        assert not out.report.ffc.frozen, "the measurement window overlaps an FFC freeze"
        plane = np.asarray(out.signal_dn, dtype=np.float64)
        total = plane if total is None else total + plane
    assert total is not None
    return float((total / n_frames - reference).std())


@pytest.mark.parametrize(
    ("drift_k_per_s", "comparable"),
    [
        (0.05, False),  # the roadmap's case: 179 s of real drift, residual-dominated
        (0.00176, True),  # tuned so the residual and the 3-D spatial term are the same size
    ],
)
@pytest.mark.slow  # GT.1: over a second on its own
def test_the_spatial_noise_budget_is_two_mechanisms_not_four(
    tophat_lwir_lut: BandLUT, drift_k_per_s: float, comparable: bool
) -> None:
    """Post-NUC spatial noise is the 3-D spatial terms in quadrature with the NUC residual, ±10 %.

    Two mechanisms, not four. The residual contributes two *terms* because §2's model is both
    multiplicative and additive (g_ij and o_ij), and at a realistic ΔT_FPA they are comparable in
    size -- but they are one mechanism with one parameter set, reset by one event.

    Run at two drift rates on purpose. At the roadmap's 0.05 K/s the residual is nearly thirty
    times the 3-D spatial term, so the budget is dominated by one mechanism and a spurious third
    could hide inside it. The second rate is tuned to make the two the same size, which is where
    the quadrature is actually under test: adding them linearly would give 2σ against √2 σ, a
    41 % error and four times the tolerance.

    Defects are off for this measurement. They are a localised artefact, not part of the spatial
    budget, and a few hundred replaced pixels would move a global standard deviation without
    saying anything about double-counting. Everything else runs.
    """
    lut = tophat_lwir_lut
    config = attach_sensor_chain(
        _config(lut, fpa=FAST_FPA),
        ambient_provider=lambda t: 293.15 + drift_k_per_s * t,
        defects_enabled=False,
    )
    n_average = 100
    fps = config.sensor.sensor.fpa.frame_rate_hz
    # Land the averaging window so it *ends* at 179 s -- just before the first shutter event at
    # 180 s, with the residual at its largest.
    out, state = _run_to(config, 179.0 - n_average / fps)
    measured = _spatial_std(config, state, n_average)

    chain = config.chain
    delta_t = chain.ffc.delta_t_eff_k
    assert delta_t > 0.1, "the FPA must actually have drifted, or the test proves nothing"

    # Each budgeted contribution from its own model, never from the image.
    sigma_tvh = float(np.asarray(config.detector.response(out.flux, 0, SEED).sigma_dn).mean())
    ratios = config.sensor.sensor.noise.ratios_3d
    sigma_3d = sigma_tvh * float(np.sqrt(ratios.v**2 + ratios.h**2 + ratios.vh**2))
    # SC.18: the gain residual acts on the signal *relative to the closed shutter* the offset was
    # last measured on, so it scales with the rms of signal − shutter, not with the whole signal.
    assert chain.shutter_dn is not None
    above_shutter = float(
        np.sqrt(np.mean((np.asarray(out.signal_dn, np.float64) - chain.shutter_dn) ** 2))
    )
    sigma_res = float(
        np.hypot(
            float(chain.residual.offset_dn(delta_t).std()),
            float(chain.residual.gain(delta_t).std()) * above_shutter,
        )
    )
    budget = float(np.hypot(sigma_3d, sigma_res))
    assert measured == pytest.approx(budget, rel=0.10)

    if comparable:
        ratio = sigma_res / sigma_3d
        assert 0.5 < ratio < 2.0, f"the two terms are meant to be comparable here, got {ratio:.2f}"
        # The discriminating part: a linear sum would be ~41 % high, four times the tolerance.
        assert measured != pytest.approx(sigma_3d + sigma_res, rel=0.10)


def test_an_ideal_camera_has_no_residual_and_no_drift(tophat_lwir_lut: BandLUT) -> None:
    """`nuc.mode: ideal` is the control the budget above is measured against."""
    config = attach_sensor_chain(
        _config(tophat_lwir_lut, nuc={"mode": "ideal"}, fpa=FAST_FPA),
        ambient_provider=lambda t: 293.15 + 0.05 * t,
        defects_enabled=False,
    )
    _run_to(config, 60.0)
    assert config.chain.ffc.delta_t_eff_k == 0.0
    assert float(config.chain.residual.offset_dn(config.chain.ffc.delta_t_eff_k).std()) == 0.0


def test_no_chain_is_the_ideal_camera(tophat_lwir_lut: BandLUT) -> None:
    """`chain=None` must leave run_frame exactly as M8 left it -- the goldens depend on it."""
    config = _config(tophat_lwir_lut)
    assert config.chain is None
    out, _ = _run_to(config, 1.0)
    assert out.report is None


# -- the couplings ---------------------------------------------------------------------------


def test_the_housing_node_drives_the_self_emission(tophat_lwir_lut: BandLUT) -> None:
    """A warming housing raises the measured signal on an unchanged scene (§8.2, ADR 0016)."""
    lut = tophat_lwir_lut
    cold = attach_sensor_chain(
        _config(lut, noise={"netd_mk_at_300k": 1e-6}),
        ambient_provider=lambda t: 280.0,
        defects_enabled=False,
        residual_enabled=False,
    )
    warm = attach_sensor_chain(
        _config(lut, noise={"netd_mk_at_300k": 1e-6}),
        ambient_provider=lambda t: 320.0,
        defects_enabled=False,
        residual_enabled=False,
    )
    a, _ = _run_to(cold, 2.0)
    b, _ = _run_to(warm, 2.0)
    assert cold.chain.housing.temperature_k < warm.chain.housing.temperature_k
    assert float(np.mean(b.signal_dn)) > float(np.mean(a.signal_dn))


def test_the_report_carries_the_nodes_and_the_ffc_state(tophat_lwir_lut: BandLUT) -> None:
    config = attach_sensor_chain(
        _config(tophat_lwir_lut, fpa=FAST_FPA), ambient_provider=lambda t: 293.15 + 0.05 * t
    )
    out, _ = _run_to(config, 5.0)
    assert out.report is not None
    assert out.report.t_housing_k > 0.0
    assert out.report.t_fpa_k > 0.0
    assert out.report.delta_t_fpa_k >= 0.0
    assert out.report.defects_active > 0


def test_the_drifting_pattern_reaches_stage_five(tophat_lwir_lut: BandLUT) -> None:
    """The pattern breathes frame to frame, so the spatial noise is not frozen across a run."""
    config = attach_sensor_chain(
        _config(tophat_lwir_lut),
        ambient_provider=lambda t: 293.15,
        defects_enabled=False,
        residual_enabled=False,
    )
    first = np.array(config.chain.fixed_pattern.vh, copy=True)
    _run_to(config, 30.0)
    later = config.chain.fixed_pattern.vh
    assert not np.array_equal(first, later)
    # but still the same distribution: the drift is stationary (ADR 0054)
    assert float(later.std()) == pytest.approx(float(first.std()), rel=0.10)


def test_defects_are_injected_and_then_replaced(tophat_lwir_lut: BandLUT) -> None:
    """The stuck values must not survive into the output, and the map must be non-empty."""
    config = attach_sensor_chain(
        _config(tophat_lwir_lut, noise={"netd_mk_at_300k": 1e-6}),
        ambient_provider=lambda t: 293.15,
        residual_enabled=False,
    )
    chain = config.chain
    assert chain.bad_pixels.count > 0
    dead = chain.bad_pixels.mask_of(DefectKind.DEAD)
    assert dead.any()

    out, _ = _run_to(config, 1.0)
    signal = np.asarray(out.signal_dn, dtype=np.float64)
    # A dead pixel would read 0 if it were not replaced; the scene here is far from the floor.
    assert float(signal[dead].min()) > 0.5 * float(np.median(signal))


def test_switching_defects_off_changes_only_the_defective_pixels(
    tophat_lwir_lut: BandLUT,
) -> None:
    lut = tophat_lwir_lut
    on = attach_sensor_chain(
        _config(lut), ambient_provider=lambda t: 293.15, residual_enabled=False
    )
    off = attach_sensor_chain(
        _config(lut),
        ambient_provider=lambda t: 293.15,
        defects_enabled=False,
        residual_enabled=False,
    )
    a, _ = _run_to(on, 1.0)
    b, _ = _run_to(off, 1.0)
    differing = np.asarray(a.signal_dn) != np.asarray(b.signal_dn)
    assert differing.any()
    # Only pixels in, or adjacent to, the defect map may differ (replacement touches the defect
    # itself; its neighbours are read but not written).
    mask = on.chain.bad_pixels.mask
    assert not np.any(differing & ~mask)


# -- the freeze, end to end -------------------------------------------------------------------


def test_a_frozen_frame_is_stale_in_every_output(tophat_lwir_lut: BandLUT) -> None:
    """The FFC holds the corrected plane, so the radiometric branch goes stale with the picture.

    A camera that froze only the 8-bit image while its apparent-temperature output kept updating
    would be a camera nobody has ever built, and would let a validation pipeline silently see
    through an artefact the perception stack cannot.
    """
    # A short interval so the event lands inside a manageable run.
    config = attach_sensor_chain(
        _config(tophat_lwir_lut, nuc={"ffc_interval_s": 2.0, "ffc_freeze_ms": 700.0}),
        ambient_provider=lambda t: 293.15,
        defects_enabled=False,
        residual_enabled=False,
    )
    fps = config.sensor.sensor.fpa.frame_rate_hz
    state = PipelineState()
    seen: list[tuple[bool, np.ndarray, np.ndarray]] = []
    for _ in range(int(round(5.0 * fps))):
        state.t_s = state.frame_index / fps
        # A scene that changes every frame, so a held output is unmistakable.
        planes = _uniform_planes(300.0 + 0.5 * state.frame_index)
        out = run_frame(planes, config, state)
        seen.append((out.report.ffc.frozen, np.asarray(out.signal_dn), np.asarray(out.apparent_t)))

    frozen = [i for i, (f, _, _) in enumerate(seen) if f]
    assert frozen, "no FFC fired in the run"

    # Group into freeze events: the window is long enough to contain more than one.
    events: list[list[int]] = [[frozen[0]]]
    for i in frozen[1:]:
        if i == events[-1][-1] + 1:
            events[-1].append(i)
        else:
            events.append([i])
    assert events

    freeze_frames = config.chain.ffc.freeze_frames
    for run in events:
        assert len(run) == freeze_frames
        reference_signal, reference_t = seen[run[0]][1], seen[run[0]][2]
        for i in run[1:]:
            assert np.array_equal(seen[i][1], reference_signal)
            assert np.array_equal(seen[i][2], reference_t)
        after = run[-1] + 1
        if after < len(seen):
            assert not np.array_equal(seen[after][1], reference_signal)


def test_the_residual_collapses_at_the_ffc(tophat_lwir_lut: BandLUT) -> None:
    """Across the event ΔT_eff returns to zero, so the residual does too (§11.2)."""
    config = attach_sensor_chain(
        _config(
            tophat_lwir_lut,
            nuc={"ffc_interval_s": 2.0, "ffc_freeze_ms": 0.0},
            fpa={"fpa_temp_mode": "ambient"},  # tracks the air, so 1 s of drive is 2 K
        ),
        ambient_provider=lambda t: 293.15 + 2.0 * t,
        defects_enabled=False,
    )
    fps = config.sensor.sensor.fpa.frame_rate_hz
    state = PipelineState()
    planes = _uniform_planes()
    before = after = None
    for _ in range(int(round(2.4 * fps))):
        state.t_s = state.frame_index / fps
        fires_next = config.chain.ffc.fires_on(state.frame_index)
        if fires_next:
            before = config.chain.ffc.delta_t_eff_k
        run_frame(planes, config, state)
        if fires_next:
            after = config.chain.ffc.delta_t_eff_k
            break
    assert before is not None and before > 0.5
    assert after == 0.0


# -- the temporal filter -----------------------------------------------------------------------


def test_the_temporal_filter_is_the_identity(tophat_lwir_lut: BandLUT) -> None:
    """ADR 0058: §11.1 names a stage it never defines, so it stays an identity until ME.5.

    Inventing a coefficient would change sigma_TVH -- the quantity NETD is measured from -- by an
    amount nobody could later distinguish from a detector-model error.
    """
    config = attach_sensor_chain(
        _config(tophat_lwir_lut), ambient_provider=lambda t: 293.15, defects_enabled=False
    )
    plane = np.random.default_rng(0).standard_normal(SHAPE).astype(np.float32) * 100.0 + 20_000.0
    assert np.array_equal(config.chain._temporal_filter(plane), plane)


# -- guards ------------------------------------------------------------------------------------


def test_attaching_without_a_calibration_is_refused(tophat_lwir_lut: BandLUT) -> None:
    """ADR 0056's conversion needs the camera's transfer; a photon FPA has none until M11.6."""
    from dataclasses import replace

    config = replace(_config(tophat_lwir_lut), calibration=None)
    with pytest.raises(ValueError, match="radiometric calibration"):
        attach_sensor_chain(config)


def test_the_chain_is_reproducible_for_a_seed(tophat_lwir_lut: BandLUT) -> None:
    lut = tophat_lwir_lut
    outs = []
    for _ in range(2):
        config = attach_sensor_chain(
            _config(lut, fpa=FAST_FPA), ambient_provider=lambda t: 293.15 + 0.05 * t
        )
        out, _ = _run_to(config, 5.0)
        outs.append(np.asarray(out.signal_dn))
    assert np.array_equal(outs[0], outs[1])
