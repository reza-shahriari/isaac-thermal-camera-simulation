"""Which clips a Tier 4 comparison is allowed to use, and why the rest were refused (EV.2).

roadmap EV.2; ADR 0068, ADR 0023 addendum; docs/physics-model.md §15 Tier 4.

The acceptance report used to take the first six clips of the archive in alphabetical order. The
project had already built three gates and applied none of them here, and ME.5 had already measured
what that costs on the reference set: **81 of 365 clips are moving**, so every per-pixel temporal
statistic is invalid on them; **306 of 365 have a robust noise scale at or below one code**, so the
codec removed the sensor's noise and the ruler every threshold is written in is the quantiser; and
only **28 clips support a noise table at all**. Taking six alphabetically means taking six clips
that are, on those proportions, most likely unusable -- and that is exactly the condition under
which a discriminator separates the two sets on ``noise_scale`` for reasons that have nothing to do
with the simulator.

So admission is a decision made once, in one place, with the same three gates the rest of the
project uses and in the order they require:

1. **static** (:func:`irsim_eval.motion.classify_clip`) -- a per-pixel temporal statistic on a
   panning clip measures the pan;
2. **the noise scale** -- at or below one code the ruler is the quantiser rather than the sensor,
   no window can pass a threshold written in units of it, and the honest answer is that the
   measurement is impossible rather than that the noise is small;
3. **a flat window** (:func:`irsim.validation.flat.find_flat_regions`) -- noise is only measurable
   where the scene is not.

**A refusal is a result and is returned, not dropped.** Most of what this project learned from the
public data is which clips it cannot use, and a report that quietly kept the survivors would state
a sample size it did not have.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    "ADMITTED",
    "Admission",
    "admit_clip",
    "admit_clips",
    "admission_summary",
]

#: The reason field of a clip that passed every gate.
ADMITTED = "admitted"


@dataclass(frozen=True)
class Admission:
    """One clip's verdict, with the numbers behind it so a reader can disagree with the gate."""

    name: str
    admitted: bool
    reason: str
    frames: int
    static: bool
    max_displacement_px: float
    noise_scale_codes: float
    flat_regions: int

    def as_dict(self) -> dict[str, Any]:
        from dataclasses import asdict

        return asdict(self)


def admit_clip(
    frames: Any,
    *,
    name: str,
    flat_size: int = 32,
    static_threshold_px: float = 1.0,
) -> Admission:
    """Run the three gates over one clip and return the verdict with its numbers.

    The gates run in the order above and the **first** failure is the reason: a moving clip's
    flat-window count is not interesting, because the window finder judges flatness against the
    temporal median and a pan has already invalidated that.
    """
    from irsim.validation.flat import find_flat_regions, robust_noise_scale
    from irsim_eval.motion import classify_clip
    from irsim_eval.reference import COLLAPSED_SCALE_CODES

    cube = np.asarray(frames)
    if cube.ndim != 3 or cube.shape[0] < 4:
        raise ValueError(f"{name}: need a (T, H, W) cube of at least four frames, got {cube.shape}")

    verdict = classify_clip(
        cube[:: max(1, cube.shape[0] // 24)], max_displacement_px=static_threshold_px
    )
    # One frame, not the temporal median: the median averages per-pixel temporal noise down by
    # sqrt(T), so a clip with healthy noise would read as collapsed (ME.5's own note).
    scale = float(robust_noise_scale(cube[cube.shape[0] // 2]))
    collapsed = scale <= COLLAPSED_SCALE_CODES + 1e-9
    # Skip the window search when the ruler has already collapsed: no window can meet a threshold
    # written in units of a scale that is the quantiser, so the search can only waste time.
    regions = [] if collapsed else find_flat_regions(cube, size=flat_size, limit=4)

    if not verdict.is_static:
        reason = "moving"
    elif collapsed:
        reason = "noise_floor_collapsed"
    elif not regions:
        reason = "no_flat_window"
    else:
        reason = ADMITTED
    return Admission(
        name=name,
        admitted=reason == ADMITTED,
        reason=reason,
        frames=int(cube.shape[0]),
        static=bool(verdict.is_static),
        max_displacement_px=float(verdict.max_displacement_px),
        noise_scale_codes=scale,
        flat_regions=len(regions),
    )


def admit_clips(
    clips: Any,
    names: Any,
    *,
    want: int | None = None,
    flat_size: int = 32,
    static_threshold_px: float = 1.0,
) -> tuple[list[list[Any]], list[Admission]]:
    """Admit clips until ``want`` of them pass, returning the survivors **and** every verdict.

    ``want`` is a count of *admitted* clips, not of clips looked at. That is the behaviour change
    EV.2 is about: stopping after the first six clips in the archive means stopping after six
    clips that are, on the reference set's own proportions, most likely unusable. Scanning stops
    as soon as enough have passed, so the cost is paid only for what the refusals make necessary.
    """
    kept: list[list[Any]] = []
    verdicts: list[Admission] = []
    for clip, name in zip(list(clips), list(names), strict=True):
        verdict = admit_clip(
            clip, name=name, flat_size=flat_size, static_threshold_px=static_threshold_px
        )
        verdicts.append(verdict)
        if verdict.admitted:
            kept.append(list(clip))
            if want is not None and len(kept) >= want:
                break
    return kept, verdicts


def admission_summary(verdicts: Any) -> str:
    """One line naming how many were admitted and what refused the rest.

    Printed with every report, because the sample size a Tier 4 number rests on is part of the
    number, and "six clips" and "six of forty-one, the rest codec-flattened" are different claims.
    """
    items = list(verdicts)
    if not items:
        return "no clips examined"
    admitted = [v for v in items if v.admitted]
    counts: dict[str, int] = {}
    for verdict in items:
        if not verdict.admitted:
            counts[verdict.reason] = counts.get(verdict.reason, 0) + 1
    line = f"{len(admitted)} of {len(items)} clips admitted"
    if counts:
        refusals = ", ".join(f"{n} {reason}" for reason, n in sorted(counts.items()))
        line += f" ({refusals})"
    return line
