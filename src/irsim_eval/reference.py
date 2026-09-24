"""Reference statistics: the bands the simulator has to land in (ME.5, §15 T4, ADR 0068).

Everything here is a *band*, not a number. A statistic measured on somebody else's video is an
estimate over a finite sample of clips that have been through an 8-bit conversion and a lossy codec,
so three things travel with every value and none of them is optional:

* **N** -- how many clips it was measured on, and how many were *refused* and why;
* **a confidence interval** -- a percentile bootstrap over clips, because the clip-to-clip spread
  dominates the within-clip one and the distributions are not normal (freeze lengths are integers,
  noise components are bounded below by zero);
* **its floor** -- the level below which the quantiser or the codec, not the camera, is what is
  being measured. A value at or under its floor is reported as ``limited``, which means *this is an
  upper bound on what the codec left*, not a measurement of the sensor.

:func:`summarise` is the only way a number enters a report, so none of the three can be skipped.

The module is engine-free and dataset-free: it takes arrays and returns summaries, so the report
generator is testable on synthetic input without a single byte of anyone's data.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from irsim.validation.signal_path import SIGNAL_PATHS, SignalPath, allows

__all__ = [
    "REPORT_SCHEMA_VERSION",
    "COLLAPSED_SCALE_CODES",
    "ClipMeasurement",
    "measure_clip",
    "require_known_path",
    "Section",
    "BOOTSTRAP_RESAMPLES",
    "BOOTSTRAP_SEED",
    "Statistic",
    "Refusal",
    "summarise",
    "refusal",
]


def require_known_path(signal_path: str) -> None:
    """Raise unless ``signal_path`` is one of the four words the vocabulary has."""
    if signal_path not in SIGNAL_PATHS:
        raise ValueError(f"unknown signal path {signal_path!r}; known: {list(SIGNAL_PATHS)}")


def _collapsed_scale() -> float:
    """The noise scale of data that is nothing but a one-code lattice.

    **Derived by running the estimator, not written down.** A checkerboard of two adjacent codes
    makes every first difference exactly one code, so this is whatever
    :func:`~irsim.validation.flat.robust_noise_scale` returns for "neighbours differ by at most one
    code" -- currently 1.0484, from the MAD's 1.4826 and the sqrt(2) of differencing two pixels.
    Deriving it means it cannot drift away from the function it describes if those constants are
    ever refined.
    """
    from irsim.validation.flat import robust_noise_scale

    rows, cols = np.mgrid[0:8, 0:8]
    return float(robust_noise_scale(((rows + cols) % 2).astype(np.float64)))


#: At or below this the ruler is the quantiser rather than the sensor, and every threshold written
#: in units of it has stopped meaning anything -- so the honest report is that the measurement is
#: impossible, not that the noise is small.
COLLAPSED_SCALE_CODES = _collapsed_scale()

#: Bumped when the report's JSON layout changes in a way a reader has to know about.
REPORT_SCHEMA_VERSION = 1

#: Fixed so the report regenerates byte-for-byte. A bootstrap is a random procedure and a report
#: that changed in the third decimal on every run could not be diffed, which is most of what a
#: reference report is for.
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 20260915


@dataclass(frozen=True)
class Statistic:
    """One measured band: what, from how many, how uncertain, and against what floor."""

    name: str
    unit: str
    n: int
    mean: float
    median: float
    ci_low: float
    ci_high: float
    spread: float  # clip-to-clip standard deviation, not the CI half-width
    floor: float | None
    floor_label: str | None
    limited: bool
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def as_row(self) -> str:
        """One markdown table row. A limited statistic is marked in the value, not in a footnote."""
        value = f"{self.mean:.4g}"
        if self.limited:
            value = f"≤ {value} ⚠"
        floor = "--" if self.floor is None else f"{self.floor:.4g}"
        if self.floor_label:
            floor += f" ({self.floor_label})"
        return (
            f"| {self.name} | {value} | {self.ci_low:.4g} – {self.ci_high:.4g} | "
            f"{self.median:.4g} | {self.spread:.4g} | {self.n} | {floor} | {self.note} |"
        )


@dataclass(frozen=True)
class Refusal:
    """A statistic that was *not* measured, and the reason, which is itself a result.

    A report that silently omitted what it could not do would read as though the set supported
    everything it lists. Most of what this project learned from the public data is in here.
    """

    name: str
    reason: str
    n_refused: int = 0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _bootstrap_ci(values: NDArray[np.float64], resamples: int, seed: int) -> tuple[float, float]:
    """Percentile bootstrap of the mean over clips, 95 %."""
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, values.size, size=(resamples, values.size))
    means = values[draws].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def summarise(
    name: str,
    unit: str,
    values: Any,
    *,
    floor: float | None = None,
    floor_label: str | None = None,
    margin: float = 2.0,
    note: str = "",
    seed: int = BOOTSTRAP_SEED,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> Statistic:
    """Summarise per-clip values into a band. Refuses an empty sample rather than reporting 0.

    ``margin`` is ADR 0023's rule, applied here to the *aggregate*: a mean within ``margin`` floors
    of the floor is flagged ``limited``, because at that level the quantiser's or the codec's
    contribution is the same size as the thing being measured and subtracting one from the other
    would be arithmetic on two numbers that are not independent.
    """
    array = np.asarray(values, dtype=np.float64).ravel()
    if array.size == 0:
        raise ValueError(f"{name}: nothing to summarise; report a Refusal instead of an empty band")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name}: {int((~np.isfinite(array)).sum())} non-finite values")
    mean = float(array.mean())
    if array.size == 1:
        lo = hi = mean
    else:
        lo, hi = _bootstrap_ci(array, resamples, seed)
    return Statistic(
        name=name,
        unit=unit,
        n=int(array.size),
        mean=mean,
        median=float(np.median(array)),
        ci_low=lo,
        ci_high=hi,
        spread=float(array.std(ddof=1)) if array.size > 1 else 0.0,
        floor=None if floor is None else float(floor),
        floor_label=floor_label,
        limited=floor is not None and mean <= margin * float(floor),
        note=note,
    )


def refusal(name: str, reason: str, n_refused: int = 0) -> Refusal:
    if not reason.strip():
        raise ValueError(f"{name}: a refusal without a reason is an omission")
    return Refusal(name=name, reason=reason.strip(), n_refused=int(n_refused))


@dataclass
class Section:
    """A group of statistics under one heading, plus what could not be measured in it."""

    title: str
    intro: str
    statistics: list[Statistic] = field(default_factory=list)
    refusals: list[Refusal] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "intro": self.intro,
            "statistics": [s.as_dict() for s in self.statistics],
            "refusals": [r.as_dict() for r in self.refusals],
        }

    def as_markdown(self) -> str:
        lines = [f"### {self.title}", "", self.intro, ""]
        if self.statistics:
            lines += [
                "| statistic | value | 95 % CI | median | clip σ | N | floor | note |",
                "|---|---|---|---|---|---|---|---|",
                *[s.as_row() for s in self.statistics],
                "",
            ]
        for r in self.refusals:
            count = f" ({r.n_refused} clips)" if r.n_refused else ""
            lines += [f"**Not measured -- {r.name}{count}.** {r.reason}", ""]
        return "\n".join(lines)


def math_is_finite(value: float) -> bool:
    """Small helper kept public so the report generator never writes a NaN into a table."""
    return math.isfinite(value)


# --- per-clip measurement ----------------------------------------------------------------------


@dataclass(frozen=True)
class ClipMeasurement:
    """Everything one clip contributes, or the reason it contributes nothing.

    ``skipped`` is not a failure: a moving clip is *correctly* excluded from every per-pixel
    temporal statistic (ME.1b's gate), and a clip with no flat window is correctly excluded from
    every noise statistic. Both are counted into the report so the N of each band is explained.
    """

    name: str
    frames: int
    fps: float
    static: bool
    max_displacement_px: float
    range_ambiguity_codes: float
    flat_regions: int
    signal_path: str = "unknown"
    skipped: tuple[str, ...] = ()
    values: dict[str, float] = field(default_factory=dict)


def measure_clip(
    frames: Any,
    *,
    name: str,
    fps: float,
    range_ambiguity: float,
    signal_path: SignalPath,
    flat_size: int = 32,
    static_threshold_px: float = 1.0,
) -> ClipMeasurement:
    """Run every frame-only Tier 4 analyser this clip's signal path supports.

    ``signal_path`` has no default, for the same reason ``conversion`` has none in
    :func:`~irsim.validation.display_signature.agc_signature` (XD.2): a default would be a claim
    about somebody else's data made by whoever wrote this function, and the whole difficulty of
    Tier 4 is that the claim belongs to the set. A caller that does not know writes ``"unknown"``
    and gets the freeze count and the blockiness, which is genuinely all an undocumented file
    supports.

    The order is the one the analysers themselves require and it is not interchangeable:

    1. **the static gate first** (ME.1b). Every per-pixel temporal statistic -- the 3-D
       decomposition, the temporal noise, the one-pole shape, the replaced-pixel map -- assumes the
       scene behind a pixel does not move. Running them on a panning clip measures the pan.
    2. **a flat window next** (ME.2b). Noise is only measurable where the scene is not; a window
       is chosen against the clip's own noise scale so the thresholds mean the same thing whatever
       the DN8 range turned out to be.
    3. **the codec floor with every noise number**, never after the fact.

    The frame rate comes from the file, never from the index: this set's core runs at 60 Hz and its
    clips are stored at 30, and a one-pole fit at the wrong rate returns a time constant wrong by
    that factor while looking entirely plausible.

    Everything the path refuses is recorded in ``skipped`` as ``<measurement>_wrong_signal_path``
    rather than dropped. A refusal is a result here -- most of what this project learned from the
    public data is which measurements its files cannot carry -- and a report that simply omitted
    them would read as though the set had been measured and come out quiet.
    """
    from irsim.validation.codec import blockiness, codec_floor, flag_codec_limited
    from irsim.validation.flat import (
        find_flat_regions,
        representative_frame,
        robust_noise_scale,
        temporal_noise,
    )
    from irsim.validation.noise import decompose_3d, spatial_psd, temporal_shape
    from irsim.validation.shutter import find_freezes
    from irsim_eval.motion import classify_clip

    require_known_path(signal_path)

    cube = np.asarray(frames)
    if cube.ndim != 3 or cube.shape[0] < 4:
        raise ValueError(f"{name}: need a (T, H, W) cube of at least four frames, got {cube.shape}")
    verdict = classify_clip(
        cube[:: max(1, cube.shape[0] // 24)], max_displacement_px=static_threshold_px
    )
    values: dict[str, float] = {}
    skipped: list[str] = []
    for measurement in ("ffc_freeze", "noise_3d", "spatial_psd", "temporal_psd"):
        if not allows(measurement, signal_path):
            skipped.append(f"{measurement}_wrong_signal_path")

    # Freezes need only frames and are the one statistic a moving clip still supports: a run of
    # identical frames is identical whether or not the camera was panning before it.
    freezes: list[Any] = []
    if allows("ffc_freeze", signal_path):
        try:
            freezes = list(find_freezes(cube, fps=fps))
        except ValueError:
            # `find_freezes` judges each still run against the clip's own median gap, so a clip
            # that never changes at all has no reference to call a run still against and it
            # refuses. That is the right answer -- such a clip is one long freeze or a recording
            # fault, and counting it as either would be a guess -- so it is recorded as
            # unmeasurable rather than as zero.
            skipped.append("freeze_not_measurable")
        else:
            values["freeze_count"] = float(len(freezes))
    if freezes:
        values["freeze_length_frames"] = float(np.mean([f.frames for f in freezes]))
        values["freeze_pattern_change"] = float(np.mean([f.pattern_change for f in freezes]))

    # The noise scale is measured on every clip, moving or not, because it is the *ruler* every
    # other threshold is expressed in (ADR 0023 addendum) and it is the one number that says
    # whether this clip has any sensor noise left in it at all. At or below one code the ruler is
    # the quantiser, no window can pass a threshold written in units of it, and the flat-region
    # finder refusing is the correct behaviour rather than a tuning problem.
    # Measured on **one frame**, not on the temporal median: the median is what `find_flat_regions`
    # scores flatness against (it deletes a moving target), but it also averages the per-pixel
    # temporal noise down by sqrt(T), so a clip with healthy temporal noise and no fixed pattern
    # would read as collapsed. The single-frame scale carries both and is the one that answers
    # "is there any sensor noise left in this file at all". Both are reported.
    scale = float(robust_noise_scale(cube[cube.shape[0] // 2]))
    values["noise_scale_codes"] = scale
    values["spatial_noise_scale_codes"] = float(robust_noise_scale(representative_frame(cube)))
    regions = find_flat_regions(cube, size=flat_size, limit=4)
    if not verdict.is_static:
        skipped.append("moving")
    if scale <= COLLAPSED_SCALE_CODES + 1e-9:
        skipped.append("noise_floor_collapsed")
    elif not regions:
        skipped.append("no_flat_window")
    if verdict.is_static and regions and allows("noise_3d", signal_path):
        best = min(regions, key=lambda r: r.structure)
        window = cube[:, best.row : best.row + best.size, best.col : best.col + best.size]
        window = window.astype(np.float64)
        decomposition = decompose_3d(window)
        floor = codec_floor(window)
        flagged = flag_codec_limited(decomposition, floor)
        measurable = set(flagged.measurable())
        values["codec_floor_codes"] = float(floor.sigma_floor)
        values["quantiser_step_codes"] = float(floor.step)
        for component in ("t", "v", "h", "tv", "th", "vh", "tvh"):
            sigma = float(getattr(decomposition, component))
            values[f"sigma_{component}_codes"] = sigma
            values[f"sigma_{component}_measurable"] = float(component in measurable)
        values["measurable_components"] = float(len(measurable))
        temporal = temporal_noise(window)
        values["temporal_sigma_codes"] = float(temporal.median)
        values["temporal_outlier_fraction"] = float(temporal.outlier_fraction)
        if allows("spatial_psd", signal_path):
            psd = spatial_psd(window)
            values["psd_line_fraction_kv0"] = float(psd.fraction_on_kv0())
            values["psd_line_fraction_kh0"] = float(psd.fraction_on_kh0())
        if allows("temporal_psd", signal_path):
            shape = temporal_shape(window, dt_s=1.0 / fps)
            if shape.tau_s is not None and math.isfinite(shape.tau_s):
                values["temporal_tau_ms"] = float(shape.tau_s * 1e3)
            values["temporal_drift_fraction"] = float(shape.drift_fraction)
    block = blockiness(cube[: min(64, cube.shape[0])].astype(np.float64))
    values["blockiness_z"] = float(max(block.z_h, block.z_v))
    return ClipMeasurement(
        name=name,
        frames=int(cube.shape[0]),
        fps=float(fps),
        static=bool(verdict.is_static),
        max_displacement_px=float(verdict.max_displacement_px),
        range_ambiguity_codes=float(range_ambiguity),
        flat_regions=len(regions),
        signal_path=signal_path,
        skipped=tuple(skipped),
        values=values,
    )
