"""Measure how far the two bands agree about the cloud, on a rendered directory (WX.26).

    python scripts/cloud_band_agreement.py outputs/phantom4_cloudscape/cumulus_towers [more dirs]
        [--json out.json] [--png]

Reads every frame that saved both ``cloud_transmittance`` (the infrared march's band
transmittance) and ``cloud_transmittance_vis`` (the visible companion's own march, the same
cloudscape), scores them with :func:`irsim.validation.cloud_bands.cloud_band_agreement`, and
prints one row per frame plus the directory's worst case against the step's bar: cloud masks at
IoU >= 0.9 and the band emissivity within 0.05 at the 95th percentile. ``--png`` writes, beside
each frame, a three-panel picture -- the marched band emissivity, the one the visible implies,
and their absolute difference -- for the eye to check what the numbers say.

Engine-free: numpy on saved planes. The planes come from ``render_phantom4.py --cloud-tier pixel``.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from irsim.validation.cloud_bands import (  # noqa: E402
    band_emissivity_from_visible,
    cloud_band_agreement,
    unreached_ray_mask,
)


def _load(path: pathlib.Path) -> np.ndarray:
    if path.suffix == ".npy":
        return np.asarray(np.load(path), dtype=np.float64)
    if path.suffix == ".exr":
        from irsim.io.exr import read_exr

        return np.asarray(read_exr(path), dtype=np.float64)
    raise ValueError(f"unknown float plane format: {path}")


def _plane(directory: pathlib.Path, stem: str, key: str) -> pathlib.Path | None:
    for ext in (".npy", ".exr"):
        candidate = directory / f"{stem}_{key}{ext}"
        if candidate.exists():
            return candidate
    return None


def _mask_spec(directory: pathlib.Path) -> dict[str, float] | None:
    """The run's own mask (`summary.json`, written by the driver): rays the infrared march's
    in-layer path cap never crossed the layer on are not judged."""
    summary = directory / "summary.json"
    if not summary.exists():
        return None
    spec = json.loads(summary.read_text()).get("cloud_band_mask")
    return spec if isinstance(spec, dict) else None


def score_directory(directory: pathlib.Path, *, od_ratio: float, png: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    spec = _mask_spec(directory)
    for meta in sorted(directory.glob("frame_*.json")):
        stem = meta.stem
        ir_path = _plane(directory, stem, "cloud_transmittance")
        vis_path = _plane(directory, stem, "cloud_transmittance_vis")
        if ir_path is None or vis_path is None:
            continue
        t_ir, t_vis = _load(ir_path), _load(vis_path)
        mask = None
        el_path = _plane(directory, stem, "elevation_rad")
        if spec is not None and el_path is not None:
            mask = unreached_ray_mask(
                _load(el_path), thickness_m=spec["thickness_m"], max_path_m=spec["max_path_m"]
            )
        result = cloud_band_agreement(t_ir, t_vis, od_ratio=od_ratio, mask=mask)
        row = {"frame": stem, **result.as_dict(), "passes": result.passes()}
        rows.append(row)
        if png:
            from irsim.io.png import write_png

            e_ir = 1.0 - np.clip(t_ir, 0.0, 1.0)
            e_vis = band_emissivity_from_visible(t_vis, od_ratio)
            diff = np.abs(e_ir - e_vis)
            panel = np.concatenate([e_ir, e_vis, np.minimum(diff * 5.0, 1.0)], axis=1)
            write_png(
                directory / f"{stem}_cloud_band_agreement.png",
                (panel * 255.0 + 0.5).astype(np.uint8),
            )
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("directories", nargs="+", type=pathlib.Path)
    parser.add_argument("--od-ratio", type=float, default=0.5, help="the band's tau_B / tau_vis")
    parser.add_argument("--json", type=pathlib.Path, default=None)
    parser.add_argument("--png", action="store_true")
    args = parser.parse_args(argv)
    report: dict[str, Any] = {}
    worst_all = True
    for directory in args.directories:
        rows = score_directory(directory, od_ratio=args.od_ratio, png=args.png)
        if not rows:
            print(f"{directory}: no frame saved both cloud planes")
            continue
        print(f"{directory} ({len(rows)} frames)")
        head = f"  {'frame':<14} {'IoU':>6} {'p95|de|':>8} {'mean|de|':>9} {'corr':>6}"
        print(head + f" {'cloud ir':>9} {'cloud vis':>10}")
        for r in rows:
            print(
                f"  {r['frame']:<14} {r['iou']:6.3f} {r['p95_emissivity_error']:8.3f} "
                f"{r['mean_emissivity_error']:9.4f} {r['correlation']:6.3f} "
                f"{r['cloud_fraction_ir']:9.3f} {r['cloud_fraction_vis']:10.3f}"
                + ("" if r["passes"] else "   FAIL")
            )
        iou_min = min(r["iou"] for r in rows)
        p95_max = max(r["p95_emissivity_error"] for r in rows)
        ok = all(r["passes"] for r in rows)
        worst_all &= ok
        verdict = "passes" if ok else "FAILS"
        print(f"  worst: IoU {iou_min:.3f}, p95 |d eps| {p95_max:.3f} -> {verdict} the bar")
        report[str(directory)] = {
            "frames": rows,
            "iou_min": iou_min,
            "p95_max": p95_max,
            "passes": ok,
        }
    if args.json:
        args.json.write_text(json.dumps(report, indent=2))
        print(f"wrote {args.json}")
    return 0 if worst_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
