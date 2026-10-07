#!/usr/bin/env python3
"""EV.23: the radial bowl a clear sky shows, measured the same way on real frames and on renders.

    python scripts/sky_bowl.py datasets/anti_uav_sky:test datasets/irsim_sky_v1:train \\
        outputs/shade_run --out outputs/sky_bowl/report.json

Each argument is a frame set, as in ``scripts/target_sharpness.py`` (a YOLO ``<root>:<split>``
or one ``render_phantom4.py`` run). Every frame is fitted with
:func:`irsim.validation.radial.radial_fit` -- a plane (the sky's own gradient) plus a paraboloid
about the centre (the bowl) -- with the target's box grown by its own size, and the top
``--readout-rows`` rows, masked out: a real frame's burnt-in readout lives there, and the same
crop is applied to every set so they are compared alike. Reported per set: the signed depth
(``|k|`` over the frame's 1-99 % span, positive for a dark centre on a white-hot display), the
share of frames that are dark-centred, the paraboloid's share of the variance the plane leaves,
and the bowl over the plane's vertical slope.
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

QUANTILES = (10, 25, 50, 75, 90)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sets", nargs="+")
    parser.add_argument("--out", default=None)
    parser.add_argument("--limit", type=int, default=400)
    parser.add_argument("--readout-rows", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    import cv2
    from target_sharpness import frames_of

    from irsim.validation.radial import radial_fit

    report = {}
    for spec in args.sets:
        name, pairs = frames_of(spec)
        rng = np.random.default_rng(args.seed)
        if len(pairs) > args.limit:
            pairs = [pairs[k] for k in sorted(rng.choice(len(pairs), args.limit, replace=False))]
        depth, share, ratio = [], [], []
        for image_path, label in pairs:
            frame = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if frame is None:
                continue
            image = frame.astype(np.float64)
            h, w = image.shape
            mask = np.ones(image.shape, dtype=bool)
            mask[: args.readout_rows, :] = False
            text = label.read_text().split() if label.exists() else []
            if len(text) >= 5:
                cx, cy, bw, bh = (float(v) for v in text[1:5])
                y0, y1 = max(0, int((cy - bh) * h)), min(h, int((cy + bh) * h))
                x0, x1 = max(0, int((cx - bw) * w)), min(w, int((cx + bw) * w))
                mask[y0:y1, x0:x1] = False
            fit = radial_fit(image, mask=mask)
            depth.append(float(np.sign(fit.k)) * fit.relative_depth)
            share.append(fit.radial_share)
            ratio.append(abs(fit.k) / max(abs(fit.plane[2]), 1e-9))
        if not depth:
            report[name] = {"frames": 0}
            continue
        report[name] = {
            "frames": len(depth),
            "signed_depth": {
                f"p{k}": round(float(v), 3)
                for k, v in zip(QUANTILES, np.percentile(depth, QUANTILES), strict=True)
            },
            "dark_centre_share": round(float(np.mean(np.asarray(depth) > 0.0)), 3),
            "radial_share_p50": round(float(np.median(share)), 3),
            "bowl_over_vertical_slope_p50": round(float(np.median(ratio)), 3),
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
