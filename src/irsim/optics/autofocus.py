"""Focus that moves: contrast-detection autofocus and range tracking (`OC.9`, ADR 0129).

`OC.4`'s three modes all hold the lens still. Real thermal cores mostly do too -- a fixed lens set
at hyperfocal -- but the motorised ones focus by **passive contrast detection**: there is no phase
sensor and no rangefinder, so the servo moves the lens, measures how sharp the picture got, and
climbs. That behaviour is worth modelling rather than shortcutting, because its artefacts are
distinctive: a lag behind a closing target, a brief hunt when contrast is flat, and a focus that
settles on whatever part of the scene carries the most edge energy rather than on the target.

`track` is the other half: a payload told what to look at, which focuses on that object's range and
ignores the rest of the frame. It is not autofocus at all -- it is a rangefinder the simulator
happens to have exactly, since the G-buffer knows every pixel's distance.

**Why not simply focus on the median range.** That is `OC.5`'s kernel choice and it is not a focus
mechanism: it presumes the camera knows the scene's depth, which a real passive imager does not.
Using it as "autofocus" would hide precisely the artefacts this models.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

__all__ = ["AutofocusServo", "focus_measure", "track_distance_m"]

FloatArray = NDArray[np.float64]


def focus_measure(plane: NDArray[np.floating]) -> float:
    """Normalised Tenengrad: mean squared gradient magnitude over the squared mean.

    The normalisation matters. A bare gradient sum rises with scene radiance, so an unnormalised
    measure would "focus" by finding the hottest part of a sequence rather than the sharpest, and a
    diurnal run would drift the lens all day. Dividing by the squared mean makes it a contrast
    measure, invariant to a scale change of the whole plane.
    """
    x = np.asarray(plane, dtype=np.float64)
    if x.ndim != 2 or x.size < 4:
        raise ValueError("focus measure needs a 2-D plane")
    gy, gx = np.gradient(x)
    mean = float(np.mean(x))
    if mean == 0.0:
        return 0.0
    return float(np.mean(gx * gx + gy * gy) / (mean * mean))


def track_distance_m(
    distance_m: object,
    id_plane: object,
    target_id: int,
    sky_mask: object = None,
) -> float | None:
    """Median range of the pixels carrying ``target_id``; ``None`` when the target is not in frame.

    The median, not the mean, for the same reason `OC.5` uses one: an object straddling a depth
    edge, or a few pixels of mis-segmentation onto the background, should not drag the focus.
    """
    d = np.asarray(distance_m, dtype=np.float64)
    ids = np.asarray(id_plane)
    hit = ids == target_id
    if sky_mask is not None:
        hit &= ~np.asarray(sky_mask, dtype=bool)
    hit &= np.isfinite(d) & (d > 0.0)
    return float(np.median(d[hit])) if np.any(hit) else None


@dataclass
class AutofocusServo:
    """A damped contrast-detection focus servo, stepped once per frame.

    Each step probes three lens positions -- where it is, and one multiplicative step either side --
    and moves toward the sharpest, by ``damping`` of the way. ``hysteresis`` is the fractional
    improvement in the focus measure below which it does not move at all, which is what stops a
    servo hunting on a static scene: without it, quantisation and noise alone keep it stepping.

    The probe is multiplicative because depth of field is: a 10 % step is the same optical move at
    5 m and at 50 m, where a fixed metre step is either invisible or wild.
    """

    distance_m: float
    step_ratio: float = 0.12
    damping: float = 0.6
    hysteresis: float = 0.005
    min_distance_m: float = 0.5
    max_distance_m: float = 1.0e6
    #: How far the probe step may widen when the servo stalls, and narrow once it is moving.
    max_step_ratio: float = 0.6
    min_step_ratio: float = 0.03
    #: Focus measures from the last :meth:`update`, in probe order, for tests and reports.
    last_scores: tuple[float, float, float] = field(default=(0.0, 0.0, 0.0))
    #: The live probe step, which is the initial one until the servo starts adapting it.
    step: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 < self.step_ratio < 1.0:
            raise ValueError("step_ratio must be a fraction between 0 and 1")
        if not 0.0 < self.damping <= 1.0:
            raise ValueError("damping must be in (0, 1]")
        if self.hysteresis < 0.0:
            raise ValueError("hysteresis cannot be negative")
        self.distance_m = float(np.clip(self.distance_m, self.min_distance_m, self.max_distance_m))
        self.step = self.step_ratio if self.step <= 0.0 else self.step

    def probes(self) -> tuple[float, float, float]:
        """(nearer, here, farther) -- the three lens positions this step will measure."""
        lo = max(self.min_distance_m, self.distance_m / (1.0 + self.step))
        hi = min(self.max_distance_m, self.distance_m * (1.0 + self.step))
        return (lo, self.distance_m, hi)

    def update(self, evaluate: Callable[[float], float]) -> float:
        """Probe, decide, move, and return the new focus distance.

        ``evaluate(distance_m)`` renders the frame at that focus and returns its focus measure. It
        is called three times per step, which is the servo's real cost and is why this is a mode a
        scene asks for rather than the default.
        """
        lo, here, hi = self.probes()
        scores = (evaluate(lo), evaluate(here), evaluate(hi))
        self.last_scores = scores
        best = int(np.argmax(scores))
        centre = scores[1]
        gain = 0.0 if centre <= 0.0 else (scores[best] - centre) / centre
        if best == 1 or gain < self.hysteresis:
            # Stalled. Far from focus the contrast measure is nearly flat -- a small lens move
            # barely changes a large blur -- so a fixed step gets stuck a long way from the target.
            # Real cores answer this by widening the search, which is what a viewer sees as
            # hunting, and it is modelled here rather than hidden behind a step that always works.
            self.step = min(self.max_step_ratio, self.step * 1.6)
            return self.distance_m
        self.step = max(self.min_step_ratio, self.step * 0.8)
        target = (lo, here, hi)[best]
        moved = self.distance_m * (target / self.distance_m) ** self.damping
        self.distance_m = float(np.clip(moved, self.min_distance_m, self.max_distance_m))
        return self.distance_m
