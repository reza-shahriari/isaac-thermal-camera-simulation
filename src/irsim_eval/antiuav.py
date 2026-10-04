"""Anti-UAV RGBT, reduced to the frames a sky-only simulator can be judged against.

The validation phase trains one small detector three ways -- on real frames, on irsim renders, and
on both -- and scores all three on the same real test frames (docs/physics-model.md §15 T5;
:data:`irsim_eval.detection.PROTOCOLS`). That comparison only measures the simulator if the real
frames contain what the simulator renders. irsim's aerial lane renders a drone against sky and
cloud; roughly half of Anti-UAV RGBT's infrared frames also contain tower blocks, cranes and
wires, and a detector's errors on those are a statement about buildings the simulator never drew.
So every frame with anything but sky, cloud and the drone is removed from **all three** splits
before any detector sees one.

The camera pans, so a clip is not sky-only or cluttered as a whole: the decision is per frame. It
is made by a small classifier trained on hand labels (``data/validation/
anti_uav_rgbt_sky_labels.csv`` -- five frames from each of the 318 clips, labelled ``sky``,
``clutter`` or ``unsure``), and what lives here is the part of that pipeline that is arithmetic:
which frames were labelled, which frames are scored, the rule that turns classifier scores into a
keep/drop decision, and the box conversion. The classifier itself needs torch and lives in
``scripts/build_anti_uav_sky.py``.

**The rule is deliberately one-sided.** A cluttered frame that slips through contaminates the
measurement; a sky frame that is dropped only costs data, and there are about 300 000 frames. So
the threshold is high and a frame must also be flanked by frames the classifier does not call
clutter (:func:`accept_sky_frames`).
"""

from __future__ import annotations

import csv
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "SPLITS",
    "SkyLabel",
    "labelled_frames",
    "load_sky_labels",
    "sampled_frames",
    "accept_sky_frames",
    "yolo_line",
]

#: The publisher's own splits, kept as they are: the split is by clip, so no frame of a test clip
#: is ever a neighbour of a training frame.
SPLITS: tuple[str, ...] = ("train", "val", "test")

LabelName = Literal["sky", "clutter", "unsure"]


@dataclass(frozen=True)
class SkyLabel:
    """One hand label: does this infrared frame show anything but sky, cloud and the drone?"""

    split: str
    clip: str
    frame: int
    label: LabelName


def labelled_frames(n_frames: int) -> tuple[int, ...]:
    """The five frame indices of a clip that were shown to the labeller.

    Start, quarters and end -- a panning camera is in a different place at each. The last one is
    ``n - 2`` rather than ``n - 1`` because the final frame of several clips does not decode.
    """
    if n_frames < 2:
        raise ValueError(f"a clip needs at least 2 frames to be labelled, got {n_frames}")
    n = n_frames
    return (0, n // 4, n // 2, 3 * n // 4, n - 2)


def load_sky_labels(path: Path | str) -> list[SkyLabel]:
    """Read the hand-label file. An unknown label is an error, not a third class."""
    out: list[SkyLabel] = []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            label = row["label"]
            if label not in ("sky", "clutter", "unsure"):
                raise ValueError(f"{path}: unknown label {label!r} for {row['clip']}")
            out.append(SkyLabel(row["split"], row["clip"], int(row["frame"]), label))  # type: ignore[arg-type]
    return out


def sampled_frames(n_frames: int, stride: int) -> NDArray[np.int64]:
    """Every ``stride``-th frame index. At 20 Hz neighbouring frames are near-duplicates, and a
    detector trained on all of them has seen a fifth as much as its image count claims."""
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")
    return np.arange(0, n_frames, stride, dtype=np.int64)


def accept_sky_frames(
    p_sky: NDArray[np.floating] | Sequence[float],
    threshold: float = 0.99,
    neighbour_floor: float = 0.5,
) -> NDArray[np.bool_]:
    """Which frames of one clip, in time order, are kept as sky-only.

    A frame is kept when its own score reaches ``threshold`` **and** neither temporal neighbour
    falls below ``neighbour_floor``. The second clause is what a per-frame classifier cannot give
    by itself: a building enters the frame over several frames, and the frame on which the
    classifier first notices is later than the frame on which the building first appears. The
    first and last frame have one neighbour and are judged on that one.
    """
    p = np.asarray(p_sky, dtype=np.float64)
    if p.ndim != 1:
        raise ValueError(f"p_sky must be one clip's scores in time order, got shape {p.shape}")
    if not 0.0 <= neighbour_floor <= threshold <= 1.0:
        raise ValueError("need 0 <= neighbour_floor <= threshold <= 1")
    if p.size and (np.any(p < 0.0) or np.any(p > 1.0) or not np.all(np.isfinite(p))):
        raise ValueError("p_sky must be probabilities in [0, 1]")
    keep = p >= threshold
    not_clutter = p >= neighbour_floor
    keep[1:] &= not_clutter[:-1]
    keep[:-1] &= not_clutter[1:]
    return keep


def yolo_line(
    rect: Sequence[float], width: int, height: int, class_id: int = 0, min_side_px: float = 1.0
) -> str | None:
    """An Anti-UAV ``gt_rect`` (x, y, w, h in pixels, top-left origin) as a YOLO label line.

    The box is clipped to the image first -- the publisher's boxes run off the edge when the drone
    does -- and ``None`` is returned when less than ``min_side_px`` of it is left on either axis,
    which is also what an absent target's all-zero rect gives.
    """
    if len(rect) != 4:
        return None
    x, y, w, h = (float(v) for v in rect)
    x0, y0 = max(x, 0.0), max(y, 0.0)
    x1, y1 = min(x + w, float(width)), min(y + h, float(height))
    bw, bh = x1 - x0, y1 - y0
    if bw < min_side_px or bh < min_side_px:
        return None
    cx, cy = (x0 + x1) / 2.0 / width, (y0 + y1) / 2.0 / height
    return f"{class_id} {cx:.6f} {cy:.6f} {bw / width:.6f} {bh / height:.6f}"
