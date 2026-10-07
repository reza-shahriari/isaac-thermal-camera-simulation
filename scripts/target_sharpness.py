#!/usr/bin/env python3
"""EV.21: how soft the drone is, measured the same way on real frames and on renders.

    python scripts/target_sharpness.py datasets/anti_uav_sky:test datasets/irsim_sky_v1:train \\
        outputs/focus_inf --out outputs/target_sharpness/report.json

Each argument is a frame set: a YOLO layout (``<root>:<split>``, images under
``<root>/images/<split>``, labels beside them under ``labels``) or one render run of
``scripts/render_phantom4.py`` (``frame_*_display8.png`` with ``labels/frame_*.txt``). For every
labelled target at least ``--min-width`` px wide and ``--min-contrast`` DN from its background it
reads :func:`irsim.validation.targets.edge_width_px` -- contrast over the steepest gradient, a ratio
any linear display mapping cancels -- and reports its quantiles with the equivalent Gaussian blur
(:func:`~irsim.validation.targets.blur_sigma_px`). Up to ``--limit`` frames a set, drawn with a
fixed seed. Reads the 8-bit PNGs with OpenCV.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

QUANTILES = (10, 25, 50, 75, 90)


def frames_of(spec: str) -> tuple[str, list[tuple[pathlib.Path, pathlib.Path]]]:
    """``(name, [(image, label), ...])`` for a YOLO split or a render run."""
    root, _, split = spec.partition(":")
    path = pathlib.Path(root)
    if split:
        images = sorted((path / "images" / split).glob("*.png"))
        pairs = [(p, path / "labels" / split / f"{p.stem}.txt") for p in images]
        return f"{path.name}:{split}", pairs
    images = sorted(path.glob("frame_*_display8.png"))
    pairs = [(p, path / "labels" / f"{p.name.removesuffix('_display8.png')}.txt") for p in images]
    return path.name, pairs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sets", nargs="+")
    parser.add_argument("--out", default=None, help="write the report here as JSON")
    parser.add_argument("--limit", type=int, default=1500)
    parser.add_argument("--min-width", type=float, default=16.0)
    parser.add_argument("--min-contrast", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    import cv2

    from irsim.validation.targets import Box, blur_sigma_px, edge_width_px

    report = {}
    for spec in args.sets:
        name, pairs = frames_of(spec)
        pairs = [(i, lab) for i, lab in pairs if lab.exists() and lab.read_text().strip()]
        rng = np.random.default_rng(args.seed)
        if len(pairs) > args.limit:
            pairs = [pairs[k] for k in sorted(rng.choice(len(pairs), args.limit, replace=False))]
        widths = []
        for image_path, label in pairs:
            frame = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if frame is None:
                continue
            h, w = frame.shape
            cx, cy, bw, bh = (float(v) for v in label.read_text().split()[1:5])
            if bw * w < args.min_width:
                continue
            x0, y0 = int(round((cx - bw / 2) * w)), int(round((cy - bh / 2) * h))
            x1, y1 = int(round((cx + bw / 2) * w)), int(round((cy + bh / 2) * h))
            x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
            if x1 <= x0 or y1 <= y0:
                continue
            box = Box(x0, y0, x1 - x0, y1 - y0)
            image = frame.astype(np.float64)
            inside = image[box.slice()]
            if np.ptp(inside) < args.min_contrast:
                continue
            try:
                widths.append(edge_width_px(image, box))
            except ValueError:
                continue
        if not widths:
            report[name] = {"targets": 0}
            continue
        q = np.percentile(widths, QUANTILES)
        report[name] = {
            "targets": len(widths),
            "edge_width_px": {
                f"p{k}": round(float(v), 3) for k, v in zip(QUANTILES, q, strict=True)
            },
            "blur_sigma_px": {
                f"p{k}": round(blur_sigma_px(v), 3) for k, v in zip(QUANTILES, q, strict=True)
            },
        }
    text = json.dumps(report, indent=1)
    if args.out:
        out = pathlib.Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
