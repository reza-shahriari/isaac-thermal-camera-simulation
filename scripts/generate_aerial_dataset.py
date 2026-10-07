#!/usr/bin/env python3
"""Render a drone-against-sky detection set with irsim, in the layout a detector trains on.

    python scripts/generate_aerial_dataset.py --out datasets/irsim_sky --clear-runs 6 --cloud-runs 2

Each **run** is one launch of ``scripts/render_phantom4.py`` under one weather, and by default one
**clip**: a continuous random flight (``--track wander``, :class:`irsim_isaac.asset_flight.
WanderTrack`) in which range, elevation, bearing, heading and the aircraft's place in the frame
all drift inside their bands while the mission clock runs, so the clip can be watched and the
airframe is seen cold, warming and hot. ``--track scatter`` gives unrelated poses instead
(:class:`~irsim_isaac.asset_flight.ScatterTrack`): more variety per frame, nothing to watch.

What changes **between** runs is
what one launch cannot change -- the hour of the day, the cloud regime and its seed -- because a
scene has one weather and the thermal solve and the sky both read it (CLAUDE.md non-negotiable 6).

The frames kept are the camera's own 8-bit display (its AGC, white-hot), which is what the public
anti-UAV sets are, and the label beside each is the box of the aircraft's own pixels from the
truth ``part_id`` plane (roadmap EV.15). Output::

    <out>/images/train/<run>_<frame>.png      8-bit grey
    <out>/labels/train/<run>_<frame>.txt      YOLO, one class (drone); empty = aircraft off frame
    <out>/videos/<run>_{ir,agc}.mp4           each clip: fixed span, and the camera's own display
    <out>/manifest.json                       every run's arguments, frame count and seconds

Everything lands in ``train``: the validation phase selects and scores on **real** frames
(``scripts/train_detector.py``), so a synthetic val split would be a split nobody reads.

Run with the Isaac interpreter (docs/decisions/0002), pinned to one GPU with ``IRSIM_GPU``. A run
that already has its ``summary.json`` is collected, not re-rendered, so an interrupted set resumes.

docs/physics-model.md §15 T5; roadmap EV.10 (viewpoint distribution), EV.15 (labels), EV.21 (focus)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
#: Cloud regimes of the weather-fx submodule that put structure behind the target without
#: precipitation in front of it.
CLOUD_REGIMES = ("fair_cumulus", "broken_cumulus", "overcast")
#: EV.21: the share of clips whose lens is focused at infinity; the rest are focused at a distance
#: drawn log-uniformly from ``near_m`` to ``FOCUS_FAR_FACTOR * far_m``. Real anti-UAV targets are
#: soft (edge width median 4.1 px against 1.9 for the old renders; ADR 0204), and a 50 mm F/1
#: lens at infinity puts the drone's blur in that range; a clip focused near the drone's own range
#: keeps some sharp frames too, which a detector should also have seen.
FOCUS_INFINITY_SHARE = 0.5
FOCUS_FAR_FACTOR = 10.0


def draw_focus(rng: np.random.Generator, near_m: float, far_m: float) -> str:
    """One clip's focus, as ``render_phantom4.py --focus-m`` takes it: ``inf`` or metres."""
    if rng.uniform() < FOCUS_INFINITY_SHARE:
        return "inf"
    lo, hi = np.log(near_m), np.log(FOCUS_FAR_FACTOR * far_m)
    return f"{float(np.exp(rng.uniform(lo, hi))):.1f}"


def plan_runs(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Every run's weather, hour and seeds, drawn once from ``--seed`` so a set reproduces."""
    rng = np.random.default_rng(args.seed)
    # A stream of its own, so adding the focus draw left every earlier set's weather, hours and
    # seeds exactly as they were.
    focus_rng = np.random.default_rng([args.seed, 21])
    runs: list[dict[str, Any]] = []
    for kind, count, frames in (
        ("clear", args.clear_runs, args.frames_per_run),
        ("cloud", args.cloud_runs, args.cloud_frames_per_run),
    ):
        for i in range(count):
            runs.append(
                {
                    "name": f"{kind}_{i:03d}",
                    "frames": frames,
                    "weather": "clear" if kind == "clear" else str(rng.choice(CLOUD_REGIMES)),
                    "weather_seed": int(rng.integers(0, 2**31 - 1)),
                    # Any hour: the real sets were filmed by day and by night, and a sunlit
                    # airframe and one that has cooled to the night air are different targets.
                    "weather_hour": round(float(rng.uniform(0.0, 24.0)), 2),
                    "scatter_seed": int(rng.integers(0, 2**31 - 1)),
                    "focus_m": (
                        draw_focus(focus_rng, args.near_m, args.far_m)
                        if args.focus == "drawn"
                        else args.focus
                    ),
                }
            )
    return runs


def render_command(args: argparse.Namespace, run: dict[str, Any], out: Path) -> list[str]:
    return [
        args.python,
        str(REPO / "scripts" / "render_phantom4.py"),
        "--track", args.track,
        "--wander-cycles", str(args.wander_cycles),
        "--mission-s", str(args.mission_s),
        "--agc-clip",
        "--asset", args.asset,
        "--scene", args.scene,
        "--sensor", args.sensor,
        "--frames", str(run["frames"]),
        "--scatter-seed", str(run["scatter_seed"]),
        "--near-m", str(args.near_m),
        "--far-m", str(args.far_m),
        "--elevation-low-deg", str(args.elevation_low_deg),
        "--elevation-high-deg", str(args.elevation_high_deg),
        "--aim-jitter", str(args.aim_jitter),
        "--weather", run["weather"],
        "--weather-seed", str(run["weather_seed"]),
        "--weather-hour", str(run["weather_hour"]),
        # The fast tier: the cloud is in the sky behind the target in both tiers, and the slow
        # one's extra is occlusion of a target *inside* cloud, at several times the cost.
        "--cloud-tier", "real_time",
        "--no-rgb", "--no-overlay",
        "--plane-stride", "1",
        *([] if run.get("focus_m", "sensor") == "sensor" else ["--focus-m", str(run["focus_m"])]),
        "--out", str(out),
    ]  # fmt: skip


def collect(run_dir: Path, name: str, out: Path) -> int:
    """Copy one run's display frames and labels into the YOLO layout. Returns frames collected."""
    import cv2

    images, labels = out / "images" / "train", out / "labels" / "train"
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)
    n = 0
    for display in sorted(run_dir.glob("frame_*_display8.png")):
        stem = display.name.removesuffix("_display8.png")
        label = run_dir / "labels" / f"{stem}.txt"
        if not label.exists():
            continue
        frame = cv2.imread(str(display), cv2.IMREAD_UNCHANGED)
        if frame.ndim == 3:
            frame = frame[..., 0]  # grey written as RGBA: every colour channel is the same plane
        cv2.imwrite(str(images / f"{name}_{stem}.png"), frame)
        (labels / f"{name}_{stem}.txt").write_text(label.read_text())
        n += 1
    videos = out / "videos"
    videos.mkdir(parents=True, exist_ok=True)
    for kind in ("ir", "agc"):
        clip = next(iter(sorted(run_dir.glob(f"*_{kind}.mp4"))), None)
        if clip is not None:
            (videos / f"{name}_{kind}.mp4").write_bytes(clip.read_bytes())
    return n


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--work", type=Path, default=None, help="per-run render directories")
    parser.add_argument("--python", default=sys.executable, help="the Isaac interpreter")
    parser.add_argument("--clear-runs", type=int, default=6)
    parser.add_argument("--cloud-runs", type=int, default=0)
    parser.add_argument("--track", default="wander", choices=("wander", "scatter"))
    parser.add_argument("--wander-cycles", type=float, default=6.0)
    parser.add_argument("--mission-s", type=float, default=1666.0, help="scene seconds per run")
    parser.add_argument("--frames-per-run", type=int, default=300)
    parser.add_argument("--cloud-frames-per-run", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20261004)
    parser.add_argument("--asset", default="phantom4")
    parser.add_argument("--scene", default="configs/scenes/phantom4_pointwise.yaml")
    parser.add_argument("--sensor", default="configs/sensors/example_lwir_640_telephoto.yaml")
    parser.add_argument("--near-m", type=float, default=20.0)
    parser.add_argument("--far-m", type=float, default=90.0)
    parser.add_argument("--elevation-low-deg", type=float, default=6.0)
    parser.add_argument("--elevation-high-deg", type=float, default=20.0)
    parser.add_argument("--aim-jitter", type=float, default=0.6)
    parser.add_argument(
        "--focus",
        default="drawn",
        help="the lens focus per clip: 'drawn' (EV.21: half at infinity, half at a distance; "
        "draw_focus), 'inf', a distance in metres, or 'sensor' for the sensor file's own",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print every planned run (weather, hour, focus, seeds) and its command; "
        "render nothing",
    )
    args = parser.parse_args(argv)

    out: Path = args.out
    work: Path = args.work or (REPO / "outputs" / "aerial_dataset" / out.name)
    if args.dry_run:
        for run in plan_runs(args):
            print(json.dumps(run))
            print("  " + " ".join(render_command(args, run, work / run["name"])))
        return 0
    work.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    records = []
    for run in plan_runs(args):
        run_dir = work / run["name"]
        command = render_command(args, run, run_dir)
        seconds = 0.0
        if not (run_dir / "summary.json").exists():
            started = time.time()
            with open(work / f"{run['name']}.log", "w") as log:
                status = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
            seconds = time.time() - started
            if status.returncode != 0 or not (run_dir / "summary.json").exists():
                print(f"{run['name']}: render failed, see {work / run['name']}.log", flush=True)
                records.append({**run, "failed": True})
                continue
        collected = collect(run_dir, run["name"], out)
        records.append({**run, "collected": collected, "render_s": round(seconds, 1)})
        print(f"{run['name']}: {collected} frames ({run['weather']}, {seconds:.0f} s)", flush=True)

    manifest = {
        "generator": "scripts/generate_aerial_dataset.py",
        "arguments": {k: str(v) for k, v in vars(args).items()},
        "frames": sum(r.get("collected", 0) for r in records),
        "runs": records,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"{manifest['frames']} frames in {out}")
    return 0 if all(not r.get("failed") for r in records) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
