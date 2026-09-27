"""M9.9 — Tier 3 sensor-chain phenomenology: what the camera does to the picture (§15 T3).

The radiometry is checked elsewhere. This bench is about the four things a *camera* does that a
radiometer does not, each of which is visible in an image and none of which is subtle once you
know to look:

* **AGC collapse.** One hot object takes the display span and everything else goes flat. This is
  the single most recognisable artefact of real thermal video and the one a simulator most often
  omits, because a radiometrically perfect frame scaled linearly looks fine.
* **The FFC freeze.** A shuttered core stops, holds its last frame for a few hundred milliseconds
  and resumes with its pattern reset. In a clip it is a visible hitch.
* **The membrane trail.** An uncooled bolometer smears a moving target across frames; a cooled
  photon detector does not. The trail is a geometric series and its ratio is `exp(−dt/τ)`.
* **Saturation.** A very hot source pins the ADC at full scale. It must clip, never wrap.

Two of the row's criteria are **not** covered here and are named rather than quietly dropped: the
aerial edge-asymmetry band comes from ME.4's measurements of real imagery, and the ME.6
real-vs-synthetic comparison needs ME.6. Both wait on public data this machine cannot fetch.

docs/physics-model.md §15 T3, §11.2, §11.3, §9.2, §16.4 step 9
"""

from __future__ import annotations

import copy
import math
import pathlib
from typing import Any

import numpy as np
import pytest
import yaml

from irsim.config.sensor import IspSpec, SensorConfig
from irsim.isp.display import run_display_branch
from irsim.materials.table import MaterialTable
from irsim.pipeline import PipelineConfig, PipelineState, attach_sensor_chain, run_frame
from irsim.radiometry.lut import BandLUT

# GT.1: this whole module is the slow tier -- a validation bench or an end-to-end frame rather
# than a unit test. `make test` skips it; `make test-slow` and `make check` run it.
pytestmark = pytest.mark.slow

REPO = pathlib.Path(__file__).resolve().parents[2]
BOSON = yaml.safe_load((REPO / "configs" / "sensors" / "flir_boson_640_lwir.yaml").read_text())
SEED = 20260915
SHAPE = (48, 64)
PAIR_COLD_K, PAIR_WARM_K = 300.0, 303.0
EXHAUST_K = 450.0
EXHAUST_FRACTION = 0.05
#: The Boson's SC.22 factory controls zeroed (ADR 0152): these tests compare AGC *operators*, and
#: an override naming only `agc` would otherwise inherit Linear Percent 20 % and Max Gain 1.38.
BARE_OPERATOR: dict[str, Any] = {"linear_percent": 0.0, "max_gain": 0.0, "detail_headroom": 0.0}


def _config(lut: BandLUT, fps: float = 60.0, **over: Any) -> PipelineConfig:
    d = copy.deepcopy(BOSON)
    d["sensor"]["fpa"].update(width=SHAPE[1], height=SHAPE[0], frame_rate_hz=fps)
    d["sensor"]["optics"].update(supersample_factor=1)
    for block, updates in over.items():
        d["sensor"][block].update(updates)
    return PipelineConfig.from_sensor(
        SensorConfig.model_validate(d), MaterialTable.constant(1.0), lut=lut, sensor_seed=SEED
    )


def _planes(temperature: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "temperature_k": np.asarray(temperature, dtype=np.float32),
        "material_id": np.ones(temperature.shape, dtype=np.int32),
        "distance_m": np.zeros(temperature.shape, dtype=np.float32),
    }


# ---------------------------------------------------------------------------------------------
# AGC collapse
# ---------------------------------------------------------------------------------------------


def _pair_scene(with_exhaust: bool) -> np.ndarray:
    """A 300/303 K pair filling the frame, optionally with a 5 % patch of 450 K exhaust."""
    scene = np.full(SHAPE, PAIR_COLD_K, dtype=np.float32)
    scene[:, SHAPE[1] // 2 :] = PAIR_WARM_K
    if with_exhaust:
        # a compact blob rather than a block: a saturated rectangle is clipped away by plateau
        # equalisation on its own, which would make the comparison a test of the blob's shape
        rows, cols = np.mgrid[0 : SHAPE[0], 0 : SHAPE[1]]
        area = EXHAUST_FRACTION * SHAPE[0] * SHAPE[1]
        sigma = math.sqrt(area / (2.0 * math.pi))
        blob = np.exp(
            -0.5 * (((rows - SHAPE[0] * 0.25) ** 2 + (cols - SHAPE[1] * 0.25) ** 2) / sigma**2)
        )
        scene = scene + (EXHAUST_K - PAIR_COLD_K) * blob.astype(np.float32)
    return scene


def _pair_contrast(lut: BandLUT, isp_over: dict[str, Any], with_exhaust: bool) -> float:
    """Mean display8 difference between the two halves of the pair, away from the exhaust."""
    isp_over = {**BARE_OPERATOR, **isp_over}
    config = _config(lut, isp=isp_over)
    out = run_frame(_planes(_pair_scene(with_exhaust)), config, PipelineState())
    spec = IspSpec.model_validate({**BOSON["sensor"]["isp"], **isp_over})
    display = run_display_branch(out.dn16, spec, config.sensor.sensor.fpa.bit_depth)
    grey = np.asarray(display.display8[..., 0], dtype=np.float64)
    # bottom half only: the exhaust blob sits in the top-left quadrant, and the point of the
    # measurement is what the AGC did to the rest of the frame, not to the neighbourhood of the
    # thing that caused it.
    lower = grey[SHAPE[0] // 2 :, :]
    return float(lower[:, SHAPE[1] // 2 :].mean() - lower[:, : SHAPE[1] // 2].mean())


@pytest.mark.parametrize(
    ("isp_over", "floor", "ceiling", "label"),
    [
        ({"agc": "linear"}, 0.0, 0.5, "linear"),
        ({"agc": "plateau_equalization", "plateau": 0.012}, 0.5, None, "plateau"),
        (
            {"agc": "plateau_local", "plateau": 0.012, "agc_tiles": (8, 8)},
            0.8,
            None,
            "plateau_local",
        ),
    ],
)
def test_a_hot_exhaust_collapses_linear_agc_and_not_the_plateau_variants(
    tophat_lwir_lut: BandLUT,
    isp_over: dict[str, Any],
    floor: float,
    ceiling: float | None,
    label: str,
) -> None:
    """A 5 % 450 K patch against a 300/303 K pair: how much of the 3 K survives the display.

    C0 is the same scene's contrast with no exhaust in it. Linear AGC hands the whole span to the
    exhaust and the 3 K pair falls to a fraction of a code; plateau equalisation gives the
    exhaust's few pixels the few codes their *population* deserves and keeps almost all of it.
    """
    c0 = _pair_contrast(tophat_lwir_lut, isp_over, with_exhaust=False)
    assert c0 > 1.0, f"{label}: the pair is not visible even without an exhaust (C0 = {c0:.3f})"
    ratio = _pair_contrast(tophat_lwir_lut, isp_over, with_exhaust=True) / c0
    assert ratio > floor - 1e-9, f"{label}: kept {ratio:.4f} of C0, wanted > {floor}"
    if ceiling is not None:
        assert ratio < ceiling, f"{label}: kept {ratio:.4f} of C0, wanted < {ceiling}"


def test_the_collapse_is_an_ordering_and_not_three_separate_numbers(
    tophat_lwir_lut: BandLUT,
) -> None:
    """linear ≪ plateau ≤ local. If that ordering inverts, one of them is doing the wrong job."""
    ratios = {}
    for label, over in (
        ("linear", {"agc": "linear"}),
        ("plateau", {"agc": "plateau_equalization", "plateau": 0.012}),
        ("local", {"agc": "plateau_local", "plateau": 0.012, "agc_tiles": (8, 8)}),
    ):
        c0 = _pair_contrast(tophat_lwir_lut, over, with_exhaust=False)
        ratios[label] = _pair_contrast(tophat_lwir_lut, over, with_exhaust=True) / c0
    assert ratios["linear"] < 0.1 * ratios["plateau"], ratios
    assert ratios["local"] >= ratios["plateau"] - 0.05, ratios


# ---------------------------------------------------------------------------------------------
# the FFC freeze
# ---------------------------------------------------------------------------------------------


def _flat_field_reference(config: PipelineConfig) -> np.ndarray:
    """The same camera's noiseless response to the same uniform scene.

    Fixed pattern is a *deviation from the expected response*, and the expected response is not
    flat: `vignetting_cos4` alone puts a cos⁴ falloff across the array that is a larger spatial
    standard deviation than the whole noise budget. Measuring a raw frame's std would be
    measuring the lens.
    """
    from dataclasses import replace

    ideal = replace(config, chain=None, noise_enabled=False)
    return np.asarray(
        run_frame(
            _planes(np.full(SHAPE, 300.0, dtype=np.float32)), ideal, PipelineState()
        ).signal_dn,
        dtype=np.float64,
    )


def test_the_shutter_freezes_exactly_42_frames_and_resets_the_pattern(
    tophat_lwir_lut: BandLUT,
) -> None:
    """60 Hz, 180 s, 700 ms: ``round(0.7 · 60)`` = **42** held frames.

    A viewer sees **43** bit-identical pictures, because the frame the shutter held is itself the
    first of the run. Both numbers are asserted: 42 is what the controller reports and 43 is what
    the clip looks like, and confusing them is an off-by-one that only shows up when someone
    counts frames in a video.

    The pattern reset is measured on **frame-averaged, flat-field-subtracted** planes, for the
    reason M9.8 gives: a single frame's spatial std is dominated by per-pixel *temporal* noise
    (13.6 DN at this camera's NETD) and by the lens, both of which survive an FFC untouched and
    would hide the thing being measured.
    """
    # A 60 s shutter interval rather than the committed 180 s, and its own focal-plane node.
    # The freeze length is set by the frame rate and `ffc_freeze_ms`, not by the interval, so the
    # 42/43 counts are the camera's real ones; the interval only decides how long the suite
    # spends growing a residual. The FPA node is M9.8's `FAST_FPA` and it is necessary, not a
    # convenience: the committed Boson has no `fpa_temp_mode`, so T_FPA falls back to the housing
    # and its 900 s lag damps the drive to a tenth -- right for a camera, and useless for a test
    # that needs the residual to have grown enough to see it removed.
    config = attach_sensor_chain(
        _config(
            tophat_lwir_lut,
            fps=60.0,
            nuc={"ffc_interval_s": 60.0},
            fpa={"fpa_temp_mode": "coupled", "fpa_tau_s": 20.0, "fpa_self_heating_k": 2.0},
        ),
        ambient_provider=lambda t: 293.15 + 0.15 * t,
    )
    fps = 60.0
    reference = _flat_field_reference(config)
    state = PipelineState()
    planes = _planes(np.full(SHAPE, 300.0, dtype=np.float32))
    for _ in range(int(round(58.5 * fps))):
        state.t_s = state.frame_index / fps
        run_frame(planes, config, state)

    frames: list[np.ndarray] = []
    signals: list[np.ndarray] = []
    frozen_flags: list[bool] = []
    fired_at: int | None = None
    for _ in range(240):
        state.t_s = state.frame_index / fps
        out = run_frame(planes, config, state)
        if out.report.ffc.fired and fired_at is None:
            fired_at = len(frames)
        frames.append(np.asarray(out.dn16))
        signals.append(np.asarray(out.signal_dn, dtype=np.float64) - reference)
        frozen_flags.append(bool(out.report.ffc.frozen))

    assert fired_at is not None and fired_at > 40, "no FFC fired inside the window"
    assert sum(frozen_flags) == 42, f"{sum(frozen_flags)} frozen frames"

    run_start = fired_at - 1  # the live frame the shutter held
    identical = 1
    while run_start + identical < len(frames) and np.array_equal(
        frames[run_start], frames[run_start + identical]
    ):
        identical += 1
    assert identical == 43, f"{identical} bit-identical frames"

    n_average = 40
    pre = np.mean(signals[fired_at - n_average : fired_at], axis=0).std()
    post = np.mean(signals[-n_average:], axis=0).std()
    assert post < 0.10 * pre, f"post-FFC pattern std {post:.4f} vs pre-FFC {pre:.4f}"


# ---------------------------------------------------------------------------------------------
# the membrane trail
# ---------------------------------------------------------------------------------------------


def _trailing_series(config: PipelineConfig, n_frames: int = 8) -> tuple[np.ndarray, float]:
    """A hot bar crosses one pixel and leaves; the residual it leaves behind, and the signal level.

    The signal level is returned because the residuals are *differences* of numbers that size, so
    it is what sets the float32 floor on any ratio taken from them.
    """
    fps = float(config.sensor.sensor.fpa.frame_rate_hz)
    state = PipelineState()
    cold = np.full(SHAPE, 290.0, dtype=np.float32)
    hot = cold.copy()
    hot[:, SHAPE[1] // 2] = 340.0
    watched = (SHAPE[0] // 2, SHAPE[1] // 2)
    # settle on the cold scene, then one hot frame, then cold again
    for _ in range(20):
        state.t_s = state.frame_index / fps
        run_frame(_planes(cold), config, state)
    state.t_s = state.frame_index / fps
    run_frame(_planes(hot), config, state)
    # 24 extra frames past the ones measured. The baseline has to be the *settled* value: taking
    # the last measured frame instead leaves ~0.19 of its own residual in it, which shifts every
    # term by the same small amount and biases the later ratios low -- 2e-4 by the third, which
    # is exactly the size of the effect this test is trying to resolve.
    tail = []
    for _ in range(n_frames + 24):
        state.t_s = state.frame_index / fps
        out = run_frame(_planes(cold), config, state)
        tail.append(float(np.asarray(out.signal_dn)[watched]))
    baseline = tail[-1]
    return (
        np.asarray([v - baseline for v in tail[:n_frames]], dtype=np.float64),
        abs(baseline),
    )


def test_the_bolometer_trail_is_a_geometric_series_with_the_membrane_ratio(
    tophat_lwir_lut: BandLUT,
) -> None:
    """Ratio `exp(−dt/τ_th)` = 0.188876 for a 10 ms membrane at 60 Hz.

    The tolerance is **derived, not chosen.** The trail is a difference of two float32 signals
    around 10 000 DN, so each residual carries an absolute error of about ε₃₂·S ≈ 1e-3 DN
    whatever its own size. By the fourth term the residual is 7.7 DN and that error is 1.3e-4 of
    it — larger than the roadmap's flat 1e-4, for no physical reason. So each ratio is held to
    its own float32 floor, and the roadmap's 1e-4 is the floor rather than the ceiling.
    """
    from dataclasses import replace

    config = replace(_config(tophat_lwir_lut, fps=60.0), noise_enabled=False)
    tau_s = config.sensor.sensor.fpa.thermal_time_constant_ms * 1e-3
    expected = math.exp(-(1.0 / 60.0) / tau_s)
    series, signal_level = _trailing_series(config)
    assert series[0] > 100.0, f"no trail at all ({series[0]:.4f} DN)"

    eps32 = float(np.finfo(np.float32).eps)
    checked = 0
    for i in range(len(series) - 1):
        if series[i] <= 0.0 or series[i + 1] <= 0.0:
            break
        ratio = series[i + 1] / series[i]
        # propagated float32 error on a ratio of two residuals of a signal of this size
        floor = 2.0 * eps32 * signal_level * (1.0 + ratio) / series[i]
        assert ratio == pytest.approx(expected, abs=max(1e-4, floor)), (
            i,
            ratio,
            expected,
            floor,
            series,
        )
        checked += 1
    assert checked >= 5, f"only {checked} terms were above the noise floor"
    # the first term, where float32 is nowhere near binding, is held to the roadmap's figure flat
    assert series[1] / series[0] == pytest.approx(expected, abs=1e-4)


def test_a_cooled_photon_detector_leaves_no_trail() -> None:
    """§16's "lateral motion smears LWIR, not cooled MWIR", in the across-frame half."""
    from irsim.config.loader import load_sensor_config
    from irsim.radiometry.lut_files import load_band_lut_for_config

    cfg = load_sensor_config(REPO / "configs/sensors/example_mwir_insb_640.yaml", REPO / "data")
    lut = load_band_lut_for_config(cfg, REPO / "data" / "lut", REPO / "data")
    d = cfg.model_dump()
    d["sensor"]["fpa"].update(width=SHAPE[1], height=SHAPE[0])
    d["sensor"]["optics"].update(supersample_factor=1)
    small = SensorConfig.model_validate(d)
    config = PipelineConfig.from_sensor(
        small, MaterialTable.constant(1.0), lut=lut, sensor_seed=SEED, noise_enabled=False
    )
    assert small.sensor.fpa.thermal_time_constant_ms is None
    series, signal_level = _trailing_series(config)
    assert abs(series[0]) < 1e-6 * max(signal_level, 1.0), (
        f"a cooled detector left a trail of {series[0]:.6f} DN on {signal_level:.1f} DN"
    )


# ---------------------------------------------------------------------------------------------
# saturation
# ---------------------------------------------------------------------------------------------


def test_a_1000_k_source_clips_at_full_scale_without_wrapping(tophat_lwir_lut: BandLUT) -> None:
    """uint16 arithmetic that wraps turns the hottest thing in the scene into the coldest."""
    config = _config(tophat_lwir_lut, fps=60.0)
    scene = np.full(SHAPE, 300.0, dtype=np.float32)
    scene[8:16, 8:16] = 1000.0
    out = run_frame(_planes(scene), config, PipelineState())
    dn = np.asarray(out.dn16)
    assert dn.dtype == np.uint16
    patch = dn[8:16, 8:16]
    assert int(patch.min()) == 65535, f"hot patch min {patch.min()}"
    background = dn[32:, 32:]
    assert int(background.max()) < 65535
    assert int(patch.min()) > int(background.max()), "the hottest region is not the brightest"
    # monotone all the way up: no fold-over anywhere between ambient and the clip
    ramp = np.linspace(300.0, 1200.0, SHAPE[1], dtype=np.float32)
    row = run_frame(_planes(np.tile(ramp, (SHAPE[0], 1))), config, PipelineState())
    values = np.asarray(row.dn16, dtype=np.int64)[SHAPE[0] // 2]
    assert np.all(np.diff(values) >= 0), "DN is not monotone in scene temperature"
    assert int(values[-1]) == 65535


# ---------------------------------------------------------------------------------------------
# what this bench does not cover
# ---------------------------------------------------------------------------------------------


def test_the_two_open_criteria_are_named_rather_than_silently_absent() -> None:
    """The Tier 3 sensor-chain bench left two criteria open, and they must stay tracked.

    Both -- an aerial edge-asymmetry band and the real-vs-synthetic comparison -- need statistics
    measured from public real imagery, which is not obtainable on this machine (no network; and
    ADR 0003 records that most of the indexed sets need a manual request anyway). This test exists
    so that "open" has something executable behind it rather than a note.

    Roadmap revision 4 retired the legacy `M9.9` id and re-lanes the remainder as `SC.14`; the
    legacy ids `ME.4` and `ME.6` remain the citations for the two criteria themselves. Pinning the
    row rather than the prose is what caught the drop: the revision had lost both criteria
    entirely, and this assertion is the only thing that noticed.
    """
    roadmap = (REPO / "docs" / "roadmap.md").read_text(encoding="utf-8")
    row = next(line for line in roadmap.splitlines() if line.startswith("| SC.14 "))
    assert "ME.4" in row and "ME.6" in row, "the open dependencies are not named in the row"
    assert "✅" not in row, "SC.14 must stay open while both criteria are unmeasured"
