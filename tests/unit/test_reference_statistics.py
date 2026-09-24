"""ME.5: the reference report's arithmetic, on synthetic input.

The report generator is tested without a byte of anyone's dataset, which is the point of keeping
the aggregation in `irsim_eval.reference` and the archive handling in the script. What is pinned
here is the discipline rather than the numbers: **no value reaches a report without N, a confidence
interval and a floor**, a value at its floor is marked as an upper bound, and anything that could
not be measured appears as a refusal with a reason rather than as an absence.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib

import numpy as np
import pytest

from irsim.validation.flat import robust_noise_scale
from irsim_eval.reference import (
    BOOTSTRAP_SEED,
    COLLAPSED_SCALE_CODES,
    REPORT_SCHEMA_VERSION,
    Section,
    measure_clip,
    refusal,
    summarise,
)

REPO = pathlib.Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "reference_statistics.py"
RNG = np.random.default_rng(20260915)


def _script():  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location("reference_statistics", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _smooth(field: np.ndarray, radius: int) -> np.ndarray:
    """Box-smooth by rolling, then normalise -- broadband, non-periodic structure with no SciPy."""
    out = np.zeros_like(field)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            out += np.roll(np.roll(field, dy, axis=0), dx, axis=1)
    out /= (2 * radius + 1) ** 2
    return out / max(float(np.std(out)), 1e-12)


def _scene(shape: tuple[int, int] = (128, 128)) -> np.ndarray:
    """Textured ground under flat sky -- the shape of the clips this report is written for.

    The two halves are in tension and the geometry is how it is resolved. Phase correlation needs
    two-dimensional, non-periodic structure to lock onto, or a still clip reads as moving: a pure
    gradient is degenerate along its own axis and a sinusoid aliases to the wrong peak. ME.2b's
    window finder needs the opposite -- a patch with no scene in it at all. A featureless frame
    cannot supply both, so the structure goes in a band at the top and the rest is left flat, which
    is also what a camera pointed just above a treeline actually sees.
    """
    field = np.random.default_rng(4).normal(0.0, 1.0, shape)
    scene = np.full(shape, 110.0)
    band = shape[0] // 3
    scene[:band] += 25.0 * _smooth(field, 3)[:band]
    return scene


def _static_cube(sigma: float, frames: int = 60) -> np.ndarray:
    """A still scene with a fixed pattern and per-pixel temporal noise -- a working camera."""
    scene = _scene()
    fixed = RNG.normal(0.0, 1.5, scene.shape)
    return np.clip(scene + fixed + RNG.normal(0.0, sigma, (frames, *scene.shape)), 0, 255).astype(
        np.uint8
    )


def _flattened_cube(frames: int = 60) -> np.ndarray:
    """What a codec leaves: the scene, still drifting, with the noise and the pattern gone."""
    scene = np.round(_scene())
    return np.stack([np.roll(scene, i // 10, axis=1) for i in range(frames)]).astype(np.uint8)


# --- the discipline ----------------------------------------------------------------------------


def test_a_statistic_always_carries_n_a_ci_and_a_floor() -> None:
    stat = summarise("x", "codes", [1.0, 1.2, 0.9, 1.1], floor=0.289, floor_label="quantiser")
    assert stat.n == 4
    assert stat.ci_low < stat.mean < stat.ci_high
    assert stat.floor == pytest.approx(0.289) and stat.floor_label == "quantiser"
    assert stat.spread > 0.0
    assert set(stat.as_dict()) >= {"n", "ci_low", "ci_high", "floor", "limited", "median", "spread"}


def test_an_empty_sample_is_refused_rather_than_reported_as_zero() -> None:
    """The failure this prevents: a band with no clips behind it printing 0 and reading as a
    measurement of zero rather than as an absence of data."""
    with pytest.raises(ValueError, match="Refusal"):
        summarise("x", "codes", [])
    with pytest.raises(ValueError, match="non-finite"):
        summarise("x", "codes", [1.0, np.nan])
    with pytest.raises(ValueError, match="omission"):
        refusal("x", "   ")


def test_a_value_at_its_floor_is_marked_as_an_upper_bound() -> None:
    at_floor = summarise("x", "codes", [0.4, 0.5, 0.45], floor=0.289)
    clear = summarise("x", "codes", [4.0, 5.0, 4.5], floor=0.289)
    assert at_floor.limited and "≤" in at_floor.as_row() and "⚠" in at_floor.as_row()
    assert not clear.limited and "≤" not in clear.as_row()


def test_the_bootstrap_is_deterministic_and_brackets_the_mean() -> None:
    """A report that changed in the third decimal on every run could not be diffed, which is most
    of what a reference report is for."""
    values = RNG.gamma(2.0, 1.5, 40)
    first = summarise("x", "codes", values, seed=BOOTSTRAP_SEED)
    again = summarise("x", "codes", values, seed=BOOTSTRAP_SEED)
    assert (first.ci_low, first.ci_high) == (again.ci_low, again.ci_high)
    assert summarise("x", "codes", values, seed=BOOTSTRAP_SEED + 1).ci_low != first.ci_low
    assert first.ci_low < first.mean < first.ci_high
    # A single clip has no spread to resample, and says so rather than inventing an interval.
    one = summarise("x", "codes", [2.0])
    assert one.n == 1 and one.ci_low == one.ci_high == one.mean and one.spread == 0.0


def test_a_section_renders_every_refusal() -> None:
    section = Section("T", "intro")
    section.statistics.append(summarise("a", "codes", [1.0, 2.0]))
    section.refusals.append(refusal("b", "the labels are MATLAB objects", n_refused=365))
    text = section.as_markdown()
    assert "the labels are MATLAB objects" in text and "365 clips" in text
    assert set(section.as_dict()) == {"title", "intro", "statistics", "refusals"}


# --- the measurement ---------------------------------------------------------------------------


def test_a_static_noisy_clip_is_measured_and_a_moving_one_is_gated_out() -> None:
    """ME.1b's gate in its report role: a per-pixel temporal statistic on a panning clip measures
    the pan, so a moving clip contributes nothing to the noise table and says why."""
    still = measure_clip(
        _static_cube(4.0), name="still", fps=30.0, range_ambiguity=8.0, signal_path="recorder"
    )
    assert still.static and not still.skipped
    assert still.values["sigma_tvh_codes"] == pytest.approx(4.0, rel=0.25)
    assert still.values["noise_scale_codes"] > 1.0
    # Both rulers are reported: the single-frame one carries the temporal noise, the temporal
    # median only the fixed pattern, and they differ by roughly the fixed pattern's own size.
    assert still.values["spatial_noise_scale_codes"] < still.values["noise_scale_codes"]

    cube = _static_cube(4.0)
    moving = np.stack([np.roll(f, 2 * i, axis=1) for i, f in enumerate(cube)])
    verdict = measure_clip(
        moving, name="moving", fps=30.0, range_ambiguity=8.0, signal_path="recorder"
    )
    assert not verdict.static and "moving" in verdict.skipped
    assert "sigma_tvh_codes" not in verdict.values


def test_a_codec_flattened_clip_reports_the_collapse_rather_than_a_number() -> None:
    """The headline finding of the reference set, reproduced synthetically.

    When the compression removes the sensor's noise, `robust_noise_scale` -- the ruler every other
    threshold is written in units of -- falls onto the quantiser. No window can then meet a
    threshold expressed in units of it, and the honest report is that the measurement is
    impossible, not that the noise is small.
    """
    flattened = _flattened_cube()
    result = measure_clip(
        flattened, name="flat", fps=30.0, range_ambiguity=8.0, signal_path="recorder"
    )
    assert result.values["noise_scale_codes"] <= 1.0
    assert "noise_floor_collapsed" in result.skipped
    assert "sigma_tvh_codes" not in result.values


def test_the_collapse_threshold_is_the_estimator_s_own_one_code_value() -> None:
    """The constant is pinned against the function it describes, not against a remembered number.

    `robust_noise_scale` of any image whose neighbours differ by at most one code is exactly
    1.4826/sqrt(2) = 1.0484, whatever the image is: the MAD of the first differences is one code
    and the rest is the estimator's constants. If those constants ever change, this fails rather
    than letting the report's collapse test quietly stop matching the estimator.
    """
    rows, cols = np.mgrid[0:64, 0:64]
    one_code = 100.0 + ((rows + cols) % 2)  # neighbours differ by exactly one code, everywhere
    assert robust_noise_scale(one_code) == pytest.approx(COLLAPSED_SCALE_CODES, rel=1e-12)
    assert 1.04 < COLLAPSED_SCALE_CODES < 1.05, COLLAPSED_SCALE_CODES
    # A flatter image reads lower, never higher, so the test is one-sided and cannot be fooled by
    # a clip that happens to have no structure in one axis: `robust_noise_scale` takes the minimum.
    assert robust_noise_scale(np.round(_scene())) <= COLLAPSED_SCALE_CODES


def test_a_clip_too_short_to_measure_is_an_error_not_a_silent_empty_row() -> None:
    with pytest.raises(ValueError, match="at least four frames"):
        measure_clip(
            np.zeros((2, 16, 16), np.uint8),
            name="x",
            fps=30.0,
            range_ambiguity=0.0,
            signal_path="recorder",
        )


# --- the report ---------------------------------------------------------------------------------


def test_the_json_payload_keeps_its_shape(tmp_path: pathlib.Path) -> None:
    """The JSON is the machine-readable half; ME.6 reads it, so its layout is a contract."""
    module = _script()
    measurements = [
        measure_clip(
            _static_cube(4.0),
            name=f"c{i}",
            fps=30.0,
            range_ambiguity=8.0,
            signal_path="recorder",
        )
        for i in range(3)
    ]

    class _Dataset:
        title = "t"
        licence = "CC0-1.0"
        sensor = "s"

    sections = module._sections(measurements, _Dataset())
    provenance = {
        "archive": "a.zip",
        "archive_sha256": "0" * 64,
        "clips_found": 3,
        "clips_measured": 3,
        "failures": [],
    }
    payload = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "dataset": "synthetic",
        "provenance": provenance,
        "sections": [s.as_dict() for s in sections],
    }
    round_tripped = json.loads(json.dumps(payload, sort_keys=True))
    assert round_tripped["schema_version"] == REPORT_SCHEMA_VERSION
    assert {s["title"] for s in round_tripped["sections"]}
    for section in round_tripped["sections"]:
        for stat in section["statistics"]:
            assert {"name", "unit", "n", "mean", "ci_low", "ci_high", "floor"} <= set(stat)
            assert stat["n"] >= 1
        for item in section["refusals"]:
            assert item["reason"].strip()
    markdown = module._render("synthetic", _Dataset(), sections, provenance)
    assert "0" * 64 in markdown, "the archive hash must be in the report a reader diffs"
    assert "Not measured" in markdown


def test_a_display_set_is_refused_the_measurements_its_path_cannot_carry() -> None:
    """XD.2's run-time half: the same clip, measured down two paths, yields two different reports.

    The frames are identical -- this is not about the data being worse. It is about an AGC having
    rescaled every frame on its own content, which moves temporal noise into the scene's variance
    and back, so a decomposition of them describes the ISP. The refusal is recorded rather than the
    statistic quietly missing, because a report that simply omitted it would read as a set that was
    measured and came out quiet.
    """
    cube = _static_cube(4.0)
    recorder = measure_clip(cube, name="c", fps=30.0, range_ambiguity=8.0, signal_path="recorder")
    display = measure_clip(cube, name="c", fps=30.0, range_ambiguity=8.0, signal_path="display")

    assert "sigma_tvh_codes" in recorder.values
    assert "sigma_tvh_codes" not in display.values
    assert "psd_line_fraction_kv0" not in display.values
    assert "noise_3d_wrong_signal_path" in display.skipped
    assert "temporal_psd_wrong_signal_path" in display.skipped
    assert display.signal_path == "display"
    # What survives an ISP still gets measured: a run of identical frames is identical whatever
    # mapped it, and the blockiness of the storage path is about the storage path.
    assert "freeze_count" in display.values
    assert "blockiness_z" in display.values


def test_an_undocumented_clip_keeps_only_what_needs_no_provenance() -> None:
    """`unknown` is the most common value in the index and the easiest to over-read.

    It is not a weaker `display`: nothing about the file is known, so the shutter statistic goes
    too -- an undocumented path may have dropped or duplicated frames of its own, which is
    indistinguishable from a freeze.
    """
    result = measure_clip(
        _static_cube(4.0), name="c", fps=30.0, range_ambiguity=8.0, signal_path="unknown"
    )
    assert "freeze_count" not in result.values
    assert "sigma_tvh_codes" not in result.values
    assert "ffc_freeze_wrong_signal_path" in result.skipped
    assert "blockiness_z" in result.values  # the storage path is still the storage path


def test_a_signal_path_outside_the_vocabulary_is_an_error() -> None:
    """A typo would otherwise silently disable every gated measurement at once."""
    with pytest.raises(ValueError, match="unknown signal path"):
        measure_clip(_static_cube(4.0), name="c", fps=30.0, range_ambiguity=8.0, signal_path="y16")  # type: ignore[arg-type]
