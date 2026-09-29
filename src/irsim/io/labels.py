"""Training labels from the planes that already exist (EV.15).

docs/physics-model.md §15 (evaluation); ADR 0134 (the truth planes), ADR 0073 (the frame pair).

A written frame carries float32 planes, a ``part_id`` truth plane whose legend names every part
the renderer resolved, and a sidecar -- but no boxes, masks or classes, although the instance
plane is exact. This module derives them, per frame, from those planes and nothing else:

* a **target** is a named set of parts (``{"quad": ["arm_n", "motor_front_left", ...]}``), its
  **class** the asset entry it came from;
* its **mask** is the pixels whose ``part_id`` names one of its parts, its **box** the mask's
  extent to the pixel (``xyxy`` in pixels, COCO ``xywh`` and normalised YOLO ``cxcywh`` derived
  from it, never the other way round);
* a target under a pixel -- the point-target path -- is labelled a **point** with its fill
  fraction ``phi`` instead of a box, because a box drawn around a sub-pixel target would be a
  claim the frame cannot support;
* **visibility** is the share of the target's mask that lies inside the frame; occlusion by other
  bodies is not measured here (the instance plane holds only what is seen) and is reported as
  ``null``, not as 1.

Files: ``<stem>_labels.json`` (one COCO-shaped document per frame: ``images``, ``annotations``,
``categories``), ``<stem>.txt`` (YOLO, one line per box), and a run-level ``split.json`` naming
which frames are train, val and test. Everything is pure NumPy and JSON; a public COCO loader
reads the per-frame documents as they are (and a run merge is a list concatenation).
"""

from __future__ import annotations

import json
import os
import pathlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "Box",
    "FrameLabels",
    "PointLabel",
    "frame_labels",
    "split_manifest",
    "write_frame_labels",
    "write_split_manifest",
]


@dataclass(frozen=True)
class Box:
    """One resolved target: its mask's extent in pixels, inclusive-exclusive ``x1, y1, x2, y2``."""

    target: str
    category: str
    x1: int
    y1: int
    x2: int
    y2: int
    pixels: int
    visibility: float | None
    mask_rle: list[int]

    @property
    def xywh(self) -> tuple[int, int, int, int]:
        return self.x1, self.y1, self.x2 - self.x1, self.y2 - self.y1

    def yolo(self, width: int, height: int) -> tuple[float, float, float, float]:
        """Normalised ``cx, cy, w, h`` in [0, 1]."""
        x, y, w, h = self.xywh
        return (x + 0.5 * w) / width, (y + 0.5 * h) / height, w / width, h / height


@dataclass(frozen=True)
class PointLabel:
    """A sub-pixel target: where it sits and how much of its pixel it fills (``phi``)."""

    target: str
    category: str
    x_px: float
    y_px: float
    phi: float


@dataclass(frozen=True)
class FrameLabels:
    width: int
    height: int
    boxes: tuple[Box, ...] = ()
    points: tuple[PointLabel, ...] = ()
    categories: tuple[str, ...] = ()
    extra: dict[str, Any] = field(default_factory=dict)

    def coco(self, image_id: int, file_name: str) -> dict[str, Any]:
        """One COCO-shaped document for this frame (a run is the concatenation)."""
        cat_id = {c: i + 1 for i, c in enumerate(self.categories)}
        annotations: list[dict[str, Any]] = []
        for k, b in enumerate(self.boxes):
            x, y, w, h = b.xywh
            annotations.append(
                {
                    "id": image_id * 10_000 + k + 1,
                    "image_id": image_id,
                    "category_id": cat_id[b.category],
                    "bbox": [x, y, w, h],
                    "area": int(b.pixels),
                    "iscrowd": 0,
                    "segmentation": {"size": [self.height, self.width], "counts": b.mask_rle},
                    "attributes": {"target": b.target, "visibility": b.visibility},
                }
            )
        for k, p in enumerate(self.points):
            annotations.append(
                {
                    "id": image_id * 10_000 + 5_000 + k + 1,
                    "image_id": image_id,
                    "category_id": cat_id[p.category],
                    "bbox": [p.x_px, p.y_px, 0.0, 0.0],
                    "area": 0,
                    "iscrowd": 0,
                    "attributes": {"target": p.target, "point": True, "phi": p.phi},
                }
            )
        return {
            "images": [
                {"id": image_id, "file_name": file_name, "width": self.width, "height": self.height}
            ],
            "annotations": annotations,
            "categories": [{"id": i, "name": c} for c, i in cat_id.items()],
        }

    def yolo_lines(self) -> list[str]:
        cat_id = {c: i for i, c in enumerate(self.categories)}
        out = []
        for b in self.boxes:
            cx, cy, w, h = b.yolo(self.width, self.height)
            out.append(f"{cat_id[b.category]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        return out


def _rle(mask: NDArray[np.bool_]) -> list[int]:
    """COCO's uncompressed RLE: run lengths of 0s and 1s in column-major (Fortran) order."""
    flat = np.asarray(mask, dtype=bool).ravel(order="F")
    if flat.size == 0:
        return []
    change = np.flatnonzero(np.diff(flat.astype(np.int8))) + 1
    edges = np.concatenate([[0], change, [flat.size]])
    runs = np.diff(edges).tolist()
    if flat[0]:
        runs = [0, *runs]  # COCO runs start with the count of 0s
    return [int(r) for r in runs]


def frame_labels(
    part_id: Any,
    legend: Mapping[int, str],
    targets: Mapping[str, Sequence[str]],
    category_of: Mapping[str, str],
    *,
    points: Sequence[tuple[str, float, float, float]] = (),
    frame_shape: tuple[int, int] | None = None,
) -> FrameLabels:
    """Boxes, masks and points for one frame from its ``part_id`` plane and legend.

    ``targets`` names each target's parts as the legend names them; a target none of whose parts
    appears in the frame gets no box (it is not in the picture, and a zero-size box would be a
    lie). ``points`` are ``(target, x_px, y_px, phi)`` for sub-pixel targets, labelled as points.
    ``frame_shape`` (rows, cols) lets a plane larger than the frame -- a padded render -- report
    the visible share of each mask.
    """
    ids = np.asarray(part_id)
    if ids.ndim != 2:
        raise ValueError("part_id is a 2-D integer plane")
    if not np.issubdtype(ids.dtype, np.integer):
        raise TypeError("part_id must be an integer plane")
    height, width = ids.shape if frame_shape is None else frame_shape
    code_of = {name: code for code, name in legend.items()}
    boxes: list[Box] = []
    categories: list[str] = []

    def category(target: str) -> str:
        try:
            cat = category_of[target]
        except KeyError as exc:
            raise KeyError(f"target {target!r} has no category") from exc
        if cat not in categories:
            categories.append(cat)
        return cat

    for target, parts in targets.items():
        codes = [code_of[p] for p in parts if p in code_of]
        missing = [p for p in parts if p not in code_of]
        if missing and not codes:
            raise KeyError(f"target {target!r}: none of its parts {list(parts)} is in the legend")
        mask = np.isin(ids, codes) if codes else np.zeros(ids.shape, dtype=bool)
        cat = category(target)
        inside = mask[:height, :width]
        n_inside = int(inside.sum())
        if n_inside == 0:
            continue
        rows = np.flatnonzero(inside.any(axis=1))
        cols = np.flatnonzero(inside.any(axis=0))
        total = int(mask.sum())
        visibility = None if frame_shape is None else float(n_inside / total)
        boxes.append(
            Box(
                target,
                cat,
                int(cols[0]),
                int(rows[0]),
                int(cols[-1]) + 1,
                int(rows[-1]) + 1,
                n_inside,
                visibility,
                _rle(inside),
            )
        )
    point_labels = []
    for target, x, y, phi in points:
        if not 0.0 < phi < 1.0:
            raise ValueError(f"point target {target!r}: phi must lie in (0, 1), got {phi}")
        point_labels.append(PointLabel(target, category(target), float(x), float(y), float(phi)))
    return FrameLabels(width, height, tuple(boxes), tuple(point_labels), tuple(categories))


def write_frame_labels(
    directory: str | os.PathLike[str], stem: str, labels: FrameLabels, *, image_id: int
) -> tuple[pathlib.Path, pathlib.Path]:
    """``<stem>_labels.json`` (COCO) and ``<stem>.txt`` (YOLO) beside the frame's planes."""
    out = pathlib.Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    coco = out / f"{stem}_labels.json"
    coco.write_text(json.dumps(labels.coco(image_id, f"{stem}.png"), indent=2, sort_keys=True))
    yolo = out / f"{stem}.txt"
    yolo.write_text("\n".join(labels.yolo_lines()) + ("\n" if labels.boxes else ""))
    return coco, yolo


def split_manifest(
    stems: Sequence[str], *, val_fraction: float = 0.1, test_fraction: float = 0.1, seed: int = 0
) -> dict[str, list[str]]:
    """A deterministic train/val/test split of a run's frames by a seeded shuffle.

    Frames of one run are one recording: a temporal split (contiguous blocks) would be the
    honest choice for a detector evaluated on unseen conditions, but the split here is a
    bookkeeping default, seeded so it reproduces, and a study can pass its own.
    """
    if not 0.0 <= val_fraction < 1.0 or not 0.0 <= test_fraction < 1.0:
        raise ValueError("fractions must lie in [0, 1)")
    if val_fraction + test_fraction >= 1.0:
        raise ValueError("val and test together must leave something to train on")
    order = list(stems)
    rng = np.random.default_rng(seed)
    rng.shuffle(order)
    n = len(order)
    n_val, n_test = int(round(n * val_fraction)), int(round(n * test_fraction))
    return {
        "test": sorted(order[:n_test]),
        "val": sorted(order[n_test : n_test + n_val]),
        "train": sorted(order[n_test + n_val :]),
    }


def write_split_manifest(
    directory: str | os.PathLike[str], split: Mapping[str, Sequence[str]], categories: Sequence[str]
) -> pathlib.Path:
    path = pathlib.Path(directory) / "split.json"
    path.write_text(
        json.dumps(
            {"split": {k: list(v) for k, v in split.items()}, "categories": list(categories)},
            indent=2,
            sort_keys=True,
        )
    )
    return path
