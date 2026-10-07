#!/usr/bin/env python3
"""EV.22: how a set's boxes sit on what is visible, real against rendered.

    python scripts/box_convention.py datasets/anti_uav_sky:test datasets/irsim_sky_v1:train \\
        --as-drawn datasets/irsim_sky_v1:train --out outputs/box_convention/report.json

Each argument is a frame set, as in ``scripts/target_sharpness.py``. For every box at least
``--min-width`` px wide it measures the box over the target's visible extent
(:func:`irsim.io.labels.seen_extent`: the pixels past half the target's contrast within one
box size), per axis, and the box's own width, height and aspect, as quantiles. A set named with
``--as-drawn`` is measured a second time with its boxes replaced by the ones a person would draw
(:func:`irsim.io.labels.as_drawn`), which is how the convention is checked on frames rendered
before it existed. ``irsim.io.labels.DRAWN_MARGIN`` is this script's real-set measurement.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

QUANTILES = (10, 50, 90)


def measure(spec: str, args: argparse.Namespace, drawn: bool) -> tuple[str, dict[str, object]]:
    import cv2
    from target_sharpness import frames_of

    from irsim.io.labels import Box, FrameLabels, as_drawn, seen_extent

    name, pairs = frames_of(spec)
    pairs = [(i, lab) for i, lab in pairs if lab.exists() and lab.read_text().strip()]
    rng = np.random.default_rng(args.seed)
    if len(pairs) > args.limit:
        pairs = [pairs[k] for k in sorted(rng.choice(len(pairs), args.limit, replace=False))]
    rows = []
    for k, (image_path, label) in enumerate(pairs):
        frame = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if frame is None:
            continue
        h, w = frame.shape
        cx, cy, bw, bh = (float(v) for v in label.read_text().split()[1:5])
        x1, y1 = max(0, int(round((cx - bw / 2) * w))), max(0, int(round((cy - bh / 2) * h)))
        x2, y2 = min(w, int(round((cx + bw / 2) * w))), min(h, int(round((cy + bh / 2) * h)))
        if x2 - x1 < args.min_width or y2 <= y1:
            continue
        box = Box("t", "drone", x1, y1, x2, y2, 0, None, [])
        if drawn:
            labels = FrameLabels(w, h, (box,), (), ("drone",))
            box = as_drawn(labels, frame, np.random.default_rng([args.seed, k])).boxes[0]
        vx1, vy1, vx2, vy2 = seen_extent(frame, box)
        width, height = box.x2 - box.x1, box.y2 - box.y1
        rows.append((width / (vx2 - vx1), height / (vy2 - vy1), width, height, width / height))
    if not rows:
        return name, {"boxes": 0}
    a = np.asarray(rows)

    def q(col: int) -> dict[str, float]:
        return {
            f"p{k}": round(float(v), 3)
            for k, v in zip(QUANTILES, np.percentile(a[:, col], QUANTILES), strict=True)
        }

    return (name + (" (as drawn)" if drawn else "")), {
        "boxes": len(rows),
        "box_over_visible_width": q(0),
        "box_over_visible_height": q(1),
        "width_px": q(2),
        "height_px": q(3),
        "aspect": q(4),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sets", nargs="*")
    parser.add_argument("--as-drawn", action="append", default=[])
    parser.add_argument("--out", default=None)
    parser.add_argument("--limit", type=int, default=800)
    parser.add_argument("--min-width", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    report = dict(measure(s, args, False) for s in args.sets)
    report.update(dict(measure(s, args, True) for s in args.as_drawn))
    text = json.dumps(report, indent=1)
    if args.out:
        out = pathlib.Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
