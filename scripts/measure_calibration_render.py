#!/usr/bin/env python3
"""SC.33's in-engine round trip: is a rendered frame distorted by exactly the camera's calibration?

`scripts/render_patch_scene.py --scene calibration_checkerboard` films a heated checkerboard
through `configs/sensors/example_boson_640_wide_calibrated.yaml`, whose ``calibration.geometric``
block (OpenCV pinhole + Brown-Conrady, ADR 0164) is authored onto the Isaac camera's lens schema.
If the renderer applies that lens, the straight rows of board corners come out bowed, and
``cv2.undistort`` with the **same** numbers -- nothing fitted, nothing tuned -- makes them straight
again. If the renderer used a different lens, or none, the undistorted rows stay bowed (or bow
the other way), which is the failure this script exists to catch.

For each image -- the infrared apparent temperature and the visible companion -- the script finds
the 12 x 10 inner corners (OpenCV's sector-based detector), fits a total-least-squares line to
every row and every column of corners, and reports the worst perpendicular residual in pixels,
before and after undistortion. It also reports how far the infrared corners sit from the visible
ones, which is the companion's registration through the same lens.

    python scripts/measure_calibration_render.py outputs/calibration_checkerboard_lwir

Writes ``calibration_roundtrip.json`` and ``calibration_roundtrip.png`` (distorted | undistorted,
both bands, with the fitted corner rows drawn) into the run directory.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any

import cv2
import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from irsim.config.loader import load_sensor_config  # noqa: E402

PATTERN = (12, 10)  # inner corners of the 13 x 11 board


def _to_u8(plane: np.ndarray) -> np.ndarray:
    finite = np.isfinite(plane)
    lo, hi = np.percentile(plane[finite], (1.0, 99.0))
    out = np.clip((np.where(finite, plane, lo) - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    return np.asarray(out * 255.0, dtype=np.uint8)


def _corners(gray: np.ndarray) -> np.ndarray | None:
    ok, found = cv2.findChessboardCornersSB(gray, PATTERN, flags=cv2.CALIB_CB_EXHAUSTIVE)
    if not ok:
        return None
    return np.asarray(found, dtype=np.float64).reshape(PATTERN[1], PATTERN[0], 2)


def _worst_line_residual_px(grid: np.ndarray) -> float:
    """Largest perpendicular distance of a corner from its row's or column's fitted line."""
    worst = 0.0
    for line in [*grid, *grid.transpose(1, 0, 2)]:
        centred = line - line.mean(axis=0)
        # the line's normal is the smallest singular direction
        normal = np.linalg.svd(centred, full_matrices=False)[2][-1]
        worst = max(worst, float(np.abs(centred @ normal).max()))
    return worst


def _measure(gray: np.ndarray, k: np.ndarray, dist: np.ndarray) -> dict[str, Any]:
    straight = cv2.undistort(gray, k, dist)
    before, after = _corners(gray), _corners(straight)
    return {
        "image": gray,
        "undistorted": straight,
        "corners": before,
        "corners_undistorted": after,
        "bow_px": None if before is None else _worst_line_residual_px(before),
        "bow_undistorted_px": None if after is None else _worst_line_residual_px(after),
    }


def _draw(gray: np.ndarray, grid: np.ndarray | None) -> np.ndarray:
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if grid is not None:
        for row in grid:
            a, b = row[0], row[-1]
            cv2.line(rgb, tuple(int(v) for v in a), tuple(int(v) for v in b), (0, 0, 255), 1)
        for x, y in grid.reshape(-1, 2):
            cv2.circle(rgb, (int(round(x)), int(round(y))), 2, (0, 255, 0), -1)
    return rgb


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", help="the render_patch_scene.py output directory")
    parser.add_argument(
        "--sensor", default=str(REPO / "configs/sensors/example_boson_640_wide_calibrated.yaml")
    )
    parser.add_argument("--frame", type=int, default=0)
    args = parser.parse_args()
    run = pathlib.Path(args.run)

    geo = load_sensor_config(args.sensor).sensor.calibration
    if geo is None or geo.geometric is None:
        print(f"{args.sensor} has no calibration.geometric block", file=sys.stderr)
        return 2
    g = geo.geometric
    # this repository's pixel-edge convention against OpenCV's pixel-centre one: the same half
    # pixel `tests/unit/test_calibration_from_file.py` shifts by
    k = np.array([[g.fx_px, 0.0, g.cx_px - 0.5], [0.0, g.fy_px, g.cy_px - 0.5], [0.0, 0.0, 1.0]])
    dist = np.array([g.k1, g.k2, g.p1, g.p2, g.k3])

    stem = run / f"frame_{args.frame:06d}"
    ir = _measure(_to_u8(np.load(f"{stem}_apparent_t.npy")), k, dist)
    rgb_path = pathlib.Path(f"{stem}_rgb.png")
    vis = None
    if rgb_path.is_file():
        vis = _measure(cv2.cvtColor(cv2.imread(str(rgb_path)), cv2.COLOR_BGR2GRAY), k, dist)

    result: dict[str, Any] = {
        "sensor": pathlib.Path(args.sensor).name,
        "frame": args.frame,
        "k1_k2": [g.k1, g.k2],
        "ir_bow_px": ir["bow_px"],
        "ir_bow_undistorted_px": ir["bow_undistorted_px"],
    }
    if vis is not None:
        result["rgb_bow_px"] = vis["bow_px"]
        result["rgb_bow_undistorted_px"] = vis["bow_undistorted_px"]
        if ir["corners"] is not None and vis["corners"] is not None:
            offset = np.linalg.norm(ir["corners"] - vis["corners"], axis=-1)
            result["ir_to_rgb_corner_offset_px"] = {
                "mean": float(offset.mean()),
                "max": float(offset.max()),
            }
    print(json.dumps(result, indent=2))
    (run / "calibration_roundtrip.json").write_text(json.dumps(result, indent=2) + "\n")

    rows = [
        np.concatenate(
            [_draw(m["image"], m["corners"]), _draw(m["undistorted"], m["corners_undistorted"])],
            axis=1,
        )
        for m in (ir, vis)
        if m is not None
    ]
    cv2.imwrite(str(run / "calibration_roundtrip.png"), np.concatenate(rows, axis=0))
    return 0 if ir["corners"] is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
