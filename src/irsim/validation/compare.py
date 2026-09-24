"""Real versus synthetic, in the display domain (ME.6, §15 T4; ADR 0068).

ADR 0068 replaced §15's radiometric Tier 4 targets with DN8 ones, because the public data for sky
targets is 8-bit, post-AGC and codec-compressed and there is nothing radiometric to compare
against. This module is the comparison those targets are checked with, and every statistic in it
is chosen to survive that path:

* **histogram EMD** in codes -- the earth-mover distance between two 8-bit histograms, which is the
  L1 distance between their cumulative distributions and reads directly as "how many codes of
  shift would it take to turn one into the other";
* **contrast ratio** (target − ring) / σ_ring -- a ratio, so an unknown AGC gain cancels;
* **PSD shape** above the codec floor -- a ratio of normalised radial spectra, so the level
  cancels and only the shape is compared, and bins the codec owns are not compared at all;
* **ESF width** (10-90 % rise) -- a length in pixels, which no display mapping changes.

**The one target that is not here.** §15 asks for apparent temperature within 2 K of a measurement.
Nothing in the public data is radiometric, so there is no measurement to be within 2 K *of*. It is
reported as `untestable` by :class:`Tier4Report` in every run rather than quietly dropped, because
a missing row reads as a passing row.

docs/physics-model.md §15 T4, §10.2, §8.3; ADR 0068
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "auc_from_scores",
    "DN8_MAX",
    "Tier4Targets",
    "Check",
    "Tier4Report",
    "histogram_emd",
    "contrast_ratio",
    "psd_shape_ratio",
    "esf_width_px",
    "compare_frames",
    "PairedStatistic",
    "paired_statistic",
    "compare_clips",
]

#: Everything here is measured on the display byte, so the axis is 0-255 codes.
DN8_MAX = 255

Verdict = Literal["pass", "fail", "untestable"]


@dataclass(frozen=True)
class Tier4Targets:
    """ADR 0068's DN8 acceptance targets, in one place so a report cannot invent its own.

    These are *replacements* for §15's radiometric targets, not translations of them: there is no
    conversion from "2 K of apparent temperature" to "8 codes of histogram shift" that does not
    require the AGC nobody documented. They were chosen for what the public data can support.
    """

    #: (target − ring)/σ_ring must agree within this fraction. 25 % is loose on purpose: the ring
    #: statistic is taken on a handful of pixels around a small target and its own sampling error
    #: is a good part of that.
    contrast_ratio_rel: float = 0.25
    #: Earth-mover distance between the DN8 histograms, in codes. Eight codes is about 3 % of the
    #: range and is smaller than the colour-range ambiguity measured on the reference set (ME.5),
    #: which is the honest floor on any absolute-level agreement.
    histogram_emd_codes: float = 8.0
    #: Radial PSD shapes may differ by at most this factor. ⚠️ **ADR 0068 proposed 2.0 and it is
    #: too loose**, measured on a smooth scene with per-pixel noise: matched noise reads
    #: 1.05-1.08, a **3x** noise mismatch reads 1.77-1.93 and only a **5x** one reaches 3.2. A 2.0
    #: threshold therefore passes a simulator whose noise is three times wrong. 1.5 separates both
    #: sides with a 1.6x margin; 2x noise reads 1.30-1.36 and slips through, which is the honest
    #: statement of this statistic's sensitivity rather than a threshold tuned to a wish.
    psd_shape_ratio: float = 1.5
    #: 10-90 % edge rise, relative.
    esf_width_rel: float = 0.20
    #: A discriminator that separates real from synthetic better than this has found a gap.
    discriminator_auc: float = 0.70


@dataclass(frozen=True)
class Check:
    """One target, its measured value, and whether it was met.

    ``untestable`` is a first-class verdict, not an error and not a pass: §15's radiometric bias
    target has no measurement to be checked against on public data, and a report that omitted it
    would read as though it had passed.
    """

    name: str
    measured: float | None
    target: float | None
    verdict: Verdict
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Tier4Report:
    checks: list[Check] = field(default_factory=list)
    #: Per-clip-pair distributions behind the whole-frame checks, keyed by check name, when the
    #: report came from :func:`compare_clips`. The note carries the same numbers as prose; this is
    #: what a reader who wants a different aggregation rule needs, and prose cannot be re-reduced.
    paired: dict[str, Any] = field(default_factory=dict)

    @property
    def failed(self) -> list[Check]:
        return [c for c in self.checks if c.verdict == "fail"]

    @property
    def untestable(self) -> list[Check]:
        return [c for c in self.checks if c.verdict == "untestable"]

    @property
    def passed(self) -> bool:
        """True when nothing failed. An untestable check does **not** make a report fail -- and it
        does not make it silent either: :meth:`as_markdown` lists every one."""
        return not self.failed

    def add(
        self,
        name: str,
        measured: float | None,
        target: float | None,
        *,
        ok: bool | None = None,
        note: str = "",
    ) -> Check:
        if ok is None:
            verdict: Verdict = "untestable"
        else:
            verdict = "pass" if ok else "fail"
        check = Check(name=name, measured=measured, target=target, verdict=verdict, note=note)
        self.checks.append(check)
        return check

    def as_markdown(self) -> str:
        rows = ["| check | measured | target | verdict | note |", "|---|---|---|---|---|"]
        mark = {"pass": "PASS", "fail": "**FAIL**", "untestable": "untestable"}
        for c in self.checks:
            measured = "--" if c.measured is None else f"{c.measured:.4g}"
            target = "--" if c.target is None else f"{c.target:.4g}"
            rows.append(f"| {c.name} | {measured} | {target} | {mark[c.verdict]} | {c.note} |")
        return "\n".join(rows)


def _as_frame(image: Any, what: str) -> NDArray[np.float64]:
    arr = np.asarray(image)
    if arr.dtype == np.float16:
        raise TypeError(f"{what} is float16 (non-negotiable #2)")
    out = np.asarray(arr, dtype=np.float64)
    if out.ndim != 2:
        raise ValueError(f"{what} must be (H, W), got {out.shape}")
    return out


def histogram_emd(a: Any, b: Any, bins: int = DN8_MAX + 1) -> float:
    """Earth-mover distance between two DN8 histograms, **in codes**.

    For one-dimensional distributions the Wasserstein-1 distance is the L1 distance between the
    cumulative distributions, which needs no optimisation and has a unit a reader can act on: a
    value of 8 means the two pictures differ by the equivalent of shifting every pixel eight codes.

    Both inputs are normalised to unit mass first, so two frames of different size are comparable
    -- a real clip and a render are never the same number of pixels.
    """
    if bins < 2:
        raise ValueError("need at least two bins")
    edges = np.linspace(0.0, float(DN8_MAX), bins + 1)
    hist_a, _ = np.histogram(_as_frame(a, "a"), bins=edges)
    hist_b, _ = np.histogram(_as_frame(b, "b"), bins=edges)
    if hist_a.sum() == 0 or hist_b.sum() == 0:
        raise ValueError("a histogram is empty; both frames must have pixels inside 0-255")
    cdf_a = np.cumsum(hist_a / hist_a.sum())
    cdf_b = np.cumsum(hist_b / hist_b.sum())
    width = float(DN8_MAX) / bins
    return float(np.sum(np.abs(cdf_a - cdf_b)) * width)


def contrast_ratio(frame: Any, box: Any, ring_margin: int = 4) -> float:
    """(mean_target − mean_ring)/σ_ring for one box -- :mod:`irsim.validation.targets`' SCR.

    Re-exported through this module so a Tier 4 comparison has one import, and defined by
    delegation so the two can never drift into two definitions of contrast.
    """
    from irsim.validation.targets import target_statistics

    return float(target_statistics(frame, box, ring_margin=ring_margin).scr)


def psd_shape_ratio(a: Any, b: Any, *, floor_fraction: float = 0.0) -> float:
    """Largest ratio between two radially averaged PSD **shapes**, above a floor.

    Each spectrum is divided by its own total power first, so an AGC's gain cancels and only the
    distribution of power over spatial frequency is compared -- the thing a wrong noise model or a
    wrong MTF actually changes.

    ``floor_fraction`` drops bins carrying less than that fraction of the *reference* spectrum's
    peak, for the case where the codec owns the top of the band (ME.2b measured x264 at CRF 18
    removing 95 % of a clip's temporal noise, and comparing those bins would be comparing two
    encoders).

    ⚠️ **It defaults to zero, and a fraction-of-peak floor is the wrong shape of filter for this.**
    A scene with low-frequency structure has a steeply falling spectrum, so "small" and
    "high-frequency" are the same bins -- which are exactly the ones the noise lives in. Measured:
    excluding bins below 1 % of the peak took a 3x noise mismatch from **1.90 to 1.06**, i.e. it
    removed the entire signal the statistic exists to detect. A real codec floor is an *absolute*
    power level, so pass a fraction computed from one rather than a round number.
    """
    from irsim.validation.noise import spatial_psd

    psd_a = spatial_psd(_as_frame(a, "a")[None, ...])
    psd_b = spatial_psd(_as_frame(b, "b")[None, ...])
    ref = np.asarray(psd_a.radial, dtype=np.float64)
    other = np.asarray(psd_b.radial, dtype=np.float64)
    if ref.shape != other.shape:
        raise ValueError(
            f"radial spectra have different lengths ({ref.size} vs {other.size}); the two frames "
            "must be the same size for a shape comparison to mean anything"
        )
    ref = ref / max(float(ref.sum()), 1e-300)
    other = other / max(float(other.sum()), 1e-300)
    keep = ref >= floor_fraction * float(ref.max())
    if not keep.any():
        raise ValueError("floor_fraction excluded every bin")
    top = np.maximum(ref[keep], other[keep])
    bottom = np.minimum(ref[keep], other[keep])
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = np.where(bottom > 0.0, top / bottom, np.inf)
    return float(np.max(ratios))


def esf_width_px(profile: Any, low: float = 0.1, high: float = 0.9) -> float:
    """10-90 % rise distance of an edge profile, in pixels, by linear interpolation.

    A length, so no display mapping changes it -- which is what makes it usable on data that has
    been through somebody's AGC. Monotonicity is not required (a real ESF overshoots after a DDE),
    so the crossings are taken as the **first** time the profile passes each level going from its
    own minimum to its own maximum.
    """
    values = np.asarray(profile, dtype=np.float64).ravel()
    if values.size < 5:
        raise ValueError("an edge profile needs at least five samples")
    if not 0.0 < low < high < 1.0:
        raise ValueError("levels must satisfy 0 < low < high < 1")
    lo, hi = float(values.min()), float(values.max())
    if hi - lo <= 0.0:
        raise ValueError("the profile is flat; there is no edge in it")
    normalised = (values - lo) / (hi - lo)
    rising = normalised if normalised[-1] >= normalised[0] else normalised[::-1]

    def crossing(level: float) -> float:
        above = np.flatnonzero(rising >= level)
        if above.size == 0:
            raise ValueError(f"the profile never reaches {level:.0%}")
        i = int(above[0])
        if i == 0:
            return 0.0
        y0, y1 = rising[i - 1], rising[i]
        return float(i - 1 + (level - y0) / (y1 - y0)) if y1 > y0 else float(i)

    return crossing(high) - crossing(low)


def compare_frames(
    real: Any,
    synthetic: Any,
    *,
    targets: Tier4Targets | None = None,
    boxes: tuple[Any, Any] | None = None,
    profiles: tuple[Any, Any] | None = None,
    psd_floor_fraction: float = 0.0,
    auc: float | None = None,
    report: Tier4Report | None = None,
) -> Tier4Report:
    """Run every DN8 check that the given inputs support, and record the ones they do not.

    ``boxes``, ``profiles`` and ``auc`` are optional because they need things the frames alone do
    not carry -- an annotation, an edge, a trained discriminator. A check whose inputs are missing
    is recorded as ``untestable`` with the reason, never skipped: on this project's data most of
    Tier 4 is untestable, and a report that hid that would be the most misleading thing it could do.
    """
    goals = targets or Tier4Targets()
    out = report or Tier4Report()

    emd = histogram_emd(real, synthetic)
    out.add(
        "histogram EMD (DN8 codes)",
        emd,
        goals.histogram_emd_codes,
        ok=emd <= goals.histogram_emd_codes,
        note="L1 between the cumulative distributions; reads as codes of shift",
    )

    ratio = psd_shape_ratio(real, synthetic, floor_fraction=psd_floor_fraction)
    out.add(
        "PSD shape ratio",
        ratio,
        goals.psd_shape_ratio,
        ok=ratio <= goals.psd_shape_ratio,
        note=(
            "radial, normalised, every bin compared"
            if psd_floor_fraction <= 0.0
            else f"radial, normalised, bins below {psd_floor_fraction:.2%} of peak excluded"
        ),
    )

    if boxes is None:
        out.add(
            "contrast ratio agreement",
            None,
            goals.contrast_ratio_rel,
            note="no annotation boxes supplied; on the reference set the labels are MATLAB MCOS "
            "objects with no Python reader (ME.5), so this is untestable there",
        )
    else:
        real_scr = contrast_ratio(real, boxes[0])
        synth_scr = contrast_ratio(synthetic, boxes[1])
        rel = abs(synth_scr - real_scr) / max(abs(real_scr), 1e-12)
        out.add(
            "contrast ratio agreement",
            rel,
            goals.contrast_ratio_rel,
            ok=rel <= goals.contrast_ratio_rel,
            note=f"real {real_scr:.3g} vs synthetic {synth_scr:.3g}",
        )

    if profiles is None:
        out.add("ESF width agreement", None, goals.esf_width_rel, note="no edge profiles supplied")
    else:
        w_real = esf_width_px(profiles[0])
        w_synth = esf_width_px(profiles[1])
        rel = abs(w_synth - w_real) / max(abs(w_real), 1e-12)
        out.add(
            "ESF width agreement",
            rel,
            goals.esf_width_rel,
            ok=rel <= goals.esf_width_rel,
            note=f"10-90 % rise: real {w_real:.3g} px vs synthetic {w_synth:.3g} px",
        )

    if auc is None:
        out.add(
            "discriminator AUC",
            None,
            goals.discriminator_auc,
            note="no discriminator supplied (irsim_eval.discriminator)",
        )
    else:
        out.add(
            "discriminator AUC",
            float(auc),
            goals.discriminator_auc,
            ok=float(auc) <= goals.discriminator_auc,
            note="0.5 is indistinguishable; 1.0 is trivially separable",
        )

    out.add(
        "apparent-temperature bias (K)",
        None,
        2.0,
        note="§15 asks for 2 K against a measurement. The public data is 8-bit, post-AGC and "
        "lossy-coded, so there is no radiometric measurement to be within 2 K of. Open until a "
        "radiometric capture exists (ADR 0003, ADR 0068)",
    )
    return out


def auc_from_scores(scores: Any, labels: Any) -> float:
    """Area under the ROC curve, by the rank (Mann-Whitney) identity, ties handled.

    Kept here rather than in the discriminator so the *metric* is engine-free and dependency-free
    and the classifier is the only thing that needs the extra stack.
    """
    s = np.asarray(scores, dtype=np.float64).ravel()
    y = np.asarray(labels).ravel().astype(bool)
    if s.size != y.size:
        raise ValueError("scores and labels differ in length")
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError("AUC needs at least one sample of each class")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(s.size, dtype=np.float64)
    ranks[order] = np.arange(1, s.size + 1, dtype=np.float64)
    # Average ranks within ties, so a classifier that outputs one constant scores exactly 0.5
    # rather than whatever the sort order happened to be.
    unique, inverse, counts = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.zeros(unique.size, dtype=np.float64)
    np.add.at(sums, inverse, ranks)
    ranks = (sums / counts)[inverse]
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


# --- clips, not a composite nobody photographed (EV.1) ------------------------------------------


@dataclass(frozen=True)
class PairedStatistic:
    """One whole-frame statistic measured over every clip pair, with its own controls.

    ``median`` is the reported value and ``low``/``high`` are the 10th and 90th percentiles over
    the pairs, so a reader sees the spread the aggregation hid. ``fraction_passing`` is there
    because a median that passes while half the pairs fail is a different situation from one where
    all of them pass, and the verdict alone cannot tell them apart.

    ``within_real`` and ``within_synthetic`` are the same statistic measured **inside** each set,
    over its own clip pairs. They are the control that says whether the between-set number means
    anything: if a set's own clips differ from each other by more than the target, then no
    simulator can meet that target against it and the honest report is that the target is below the
    data's own spread, not that the simulator failed.
    """

    name: str
    n_pairs: int
    median: float
    low: float
    high: float
    fraction_passing: float
    within_real: float | None = None
    within_synthetic: float | None = None

    @property
    def within(self) -> float | None:
        """The larger of the two within-set spreads -- the floor under any between-set value."""
        values = [v for v in (self.within_real, self.within_synthetic) if v is not None]
        return max(values) if values else None

    @property
    def below_the_sets_own_spread(self) -> bool:
        """True when the two sets differ by no more than one set differs from itself.

        This is the strongest form of agreement these statistics can express: a difference smaller
        than the within-set spread is not evidence of a gap, whatever the target says.
        """
        floor = self.within
        return floor is not None and self.median <= floor

    def as_note(self) -> str:
        """The one line the report carries, so the aggregation is never implicit."""
        parts = [
            f"median of {self.n_pairs} clip pair{'s' if self.n_pairs != 1 else ''} "
            f"(10-90 %: {self.low:.4g}-{self.high:.4g}); "
            f"{self.fraction_passing:.0%} of pairs meet the target"
        ]
        floor = self.within
        if floor is not None:
            verdict = "at or under" if self.below_the_sets_own_spread else "above"
            parts.append(
                f"each set's own clip pairs read {floor:.4g}, so the between-set value is "
                f"{verdict} the sets' own clip-to-clip spread"
            )
        return "; ".join(parts)


def _pairwise(frames_a: Any, frames_b: Any, statistic: Any) -> NDArray[np.float64]:
    """``statistic`` over every (a, b) pair. Same list twice gives the within-set pairs, i < j."""
    a = list(frames_a)
    b = list(frames_b)
    if not a or not b:
        raise ValueError("a paired statistic needs at least one clip on each side")
    same = a is b or (len(a) == len(b) and all(x is y for x, y in zip(a, b, strict=True)))
    if same:
        values = [statistic(a[i], a[j]) for i in range(len(a)) for j in range(i + 1, len(a))]
        # One clip has no pair to differ from, which is not a spread of zero -- it is no
        # measurement at all, and returning zero would claim a perfectly uniform set.
        return np.asarray(values, dtype=np.float64)
    return np.asarray([statistic(x, y) for x in a for y in b], dtype=np.float64)


def paired_statistic(
    name: str,
    real_mosaics: Any,
    synthetic_mosaics: Any,
    statistic: Any,
    target: float,
) -> PairedStatistic:
    """Measure ``statistic`` over every clip pair, and inside each set as its own control."""
    between = _pairwise(real_mosaics, synthetic_mosaics, statistic)
    real_list, synth_list = list(real_mosaics), list(synthetic_mosaics)
    within_real = _pairwise(real_list, real_list, statistic)
    within_synth = _pairwise(synth_list, synth_list, statistic)
    return PairedStatistic(
        name=name,
        n_pairs=int(between.size),
        median=float(np.median(between)),
        low=float(np.percentile(between, 10.0)),
        high=float(np.percentile(between, 90.0)),
        fraction_passing=float(np.mean(between <= target)),
        within_real=float(np.median(within_real)) if within_real.size else None,
        within_synthetic=float(np.median(within_synth)) if within_synth.size else None,
    )


def compare_clips(
    real_mosaics: Any,
    synthetic_mosaics: Any,
    *,
    targets: Tier4Targets | None = None,
    boxes: tuple[Any, Any] | None = None,
    profiles: tuple[Any, Any] | None = None,
    psd_floor_fraction: float = 0.0,
    auc: float | None = None,
    report: Tier4Report | None = None,
) -> Tier4Report:
    """Compare two sets **clip by clip** rather than as one pooled composite (EV.1).

    :func:`compare_frames` takes one frame from each side, which is right when there is one clip on
    each side and wrong the moment there is more than one. The report used to build its single
    frame by taking the temporal median of every clip's frames stacked together, and that composite
    is an image neither set contains: measured on this project's own matched clips the per-clip
    mosaic means were 63.2, 58.3, 67.1, 93.3, 113.1 and 102.0 codes, a **55-code span**, against an
    8-code histogram target. The pooled number was therefore reporting the clip-to-clip variation
    of the synthetic set and calling it a gap.

    **The aggregation is stated, not implied.** Each whole-frame statistic is measured on every
    (real clip, synthetic clip) mosaic pair; the reported value is the **median** over those pairs,
    because these statistics are distances, bounded below by zero and skewed, and one anomalous
    clip should not carry a verdict. The 10-90 % span, the fraction of pairs meeting the target and
    each set's own clip-to-clip spread all travel in the note, so a reader can apply a stricter
    rule without re-running anything.

    With one clip on each side this reduces exactly to :func:`compare_frames` for the two
    whole-frame checks: one pair, and its median is its value.
    """
    goals = targets or Tier4Targets()
    out = report or Tier4Report()

    emd = paired_statistic(
        "histogram EMD (DN8 codes)",
        real_mosaics,
        synthetic_mosaics,
        histogram_emd,
        goals.histogram_emd_codes,
    )
    out.add(
        emd.name,
        emd.median,
        goals.histogram_emd_codes,
        ok=emd.median <= goals.histogram_emd_codes,
        note="L1 between the cumulative distributions, in codes of shift; " + emd.as_note(),
    )
    out.paired[emd.name] = emd

    def _psd(a: Any, b: Any) -> float:
        return psd_shape_ratio(a, b, floor_fraction=psd_floor_fraction)

    psd = paired_statistic(
        "PSD shape ratio", real_mosaics, synthetic_mosaics, _psd, goals.psd_shape_ratio
    )
    out.add(
        psd.name,
        psd.median,
        goals.psd_shape_ratio,
        ok=psd.median <= goals.psd_shape_ratio,
        note=(
            (
                "radial, normalised, every bin compared; "
                if psd_floor_fraction <= 0.0
                else f"radial, normalised, bins below {psd_floor_fraction:.2%} of peak excluded; "
            )
            + psd.as_note()
        ),
    )
    out.paired[psd.name] = psd

    # The remaining checks need an annotation, an edge or a trained discriminator rather than a
    # second frame, so they are unchanged by the pooling fix and are delegated as they were. A
    # representative mosaic is passed only to satisfy the signature; neither check reads it when
    # its own input is absent, and when the input is present it is the caller's pairing.
    first_real = list(real_mosaics)[0]
    first_synth = list(synthetic_mosaics)[0]
    _skip = {"histogram EMD (DN8 codes)", "PSD shape ratio"}
    rest = compare_frames(
        first_real,
        first_synth,
        targets=goals,
        boxes=boxes,
        profiles=profiles,
        psd_floor_fraction=psd_floor_fraction,
        auc=auc,
    )
    out.checks.extend(c for c in rest.checks if c.name not in _skip)
    return out
