#!/usr/bin/env python3
"""EV.26: where the real Anti-UAV set has no drone in the frame, and for how long.

    python scripts/absent_frames.py [--root datasets/Anti-UAV-RGBT] \\
        [--out data/validation/anti_uav_absent_runs.csv]

Each Anti-UAV-RGBT clip carries ``infrared.json`` with an ``exist`` flag per frame (0: no drone
in the picture, so no box). This reads every clip of every split, at the full 20 fps, and writes
one row per run of absent frames: its split, clip, first frame and length. The comment header
records the totals the dataset planner sizes its empty frames from
(``generate_aerial_dataset.py``: the share, and run lengths resampled from these rows).

The labels are the dataset's own (its licence is unstated, ``data/validation/README.md``). Only
counts and frame indices are derived here; no frame or box is copied.
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]


def absent_runs(exist: list[int]) -> list[tuple[int, int]]:
    """``(first frame, length)`` of each run of zeros in a clip's ``exist`` flags."""
    runs, start = [], None
    for i, flag in enumerate(exist):
        if not flag and start is None:
            start = i
        elif flag and start is not None:
            runs.append((start, i - start))
            start = None
    if start is not None:
        runs.append((start, len(exist) - start))
    return runs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(REPO / "datasets" / "Anti-UAV-RGBT"))
    parser.add_argument(
        "--out", default=str(REPO / "data" / "validation" / "anti_uav_absent_runs.csv")
    )
    args = parser.parse_args()

    root = pathlib.Path(args.root)
    rows, frames, clips, clips_with = [], 0, 0, 0
    for split in ("train", "val", "test"):
        for path in sorted((root / split).glob("*/infrared.json")):
            exist = [int(v) for v in json.loads(path.read_text(encoding="utf-8"))["exist"]]
            runs = absent_runs(exist)
            frames += len(exist)
            clips += 1
            clips_with += bool(runs)
            rows.extend((split, path.parent.name, first, length) for first, length in runs)
    if not clips:
        raise SystemExit(f"no infrared.json under {root}")
    absent = sum(r[3] for r in rows)
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as fh:
        fh.write(
            "# Anti-UAV-RGBT infrared, all splits (Jiang et al. 2021; labels' licence unstated): "
            "runs of frames whose exist flag is 0, read at the full 20 fps by "
            "scripts/absent_frames.py.\n"
            f"# clips={clips} clips_with_absent={clips_with} frames={frames} "
            f"absent_frames={absent}\n"
        )
        writer = csv.writer(fh)
        writer.writerow(["split", "clip", "first_frame", "length"])
        writer.writerows(rows)
    print(
        f"{absent} of {frames} frames absent ({absent / frames:.2%}) in {len(rows)} runs, "
        f"{clips_with} of {clips} clips"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
