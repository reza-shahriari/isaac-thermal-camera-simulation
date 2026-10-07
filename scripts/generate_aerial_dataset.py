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

docs/physics-model.md §15 T5; roadmap EV.10 (viewpoint distribution), EV.15 (labels),
EV.21 (focus), EV.23 (lens shading), EV.24 (hours)
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
#: EV.23: each clip's camera starts its housing this far from settled, K, uniform. Real clear-sky
#: frames are dark-centred in 71 % of cases (a housing that warmed since its shutter); -4 K is a
#: camera just switched on, and a housing in the sun runs further. Three quarters of the band is
#: warming. ESTIMATED; ADR 0206.
HOUSING_START_K = (-8.0, 3.0)
#: EV.23: each clip's lens falls to this relative illumination at the corner, uniform: real fast IR
#: lenses vignette mechanically as well as by cos^4, and the bowl a drift leaves grows with it.
#: ESTIMATED; ADR 0206.
CORNER_ILLUMINATION = (0.6, 0.9)
#: EV.25: the airframes a clip may fly, as (asset, scene, nose bearing in the asset's XY plane,
#: degrees from +x; None for the Phantom 4's measured NOSE_IN_ASSET). The Phantom 4 flies its
#: solved per-cell scene; the library's other multirotors fly `zoo_hero.yaml`, whose generic
#: heat sources serve any part-split asset (motors hot, the pack warm, the shell near air).
#: Noses are the ones their asset configs state; the last field is the asset's span as
#: `render_phantom4.py` measures it ("m across"), which scales the clip's range band so every
#: airframe covers the apparent sizes the band was set for (the real boxes are 34-83 px wide).
AIRFRAMES: tuple[tuple[str, str, float | None, float], ...] = (
    ("phantom4", "configs/scenes/phantom4_pointwise.yaml", None, 0.618),
    ("dji_mini_3_pro_parts", "configs/scenes/zoo_hero.yaml", -129.0, 0.391),
    ("dji_avata_2_parts", "configs/scenes/zoo_hero.yaml", -90.0, 0.212),
    ("dji_fpv_parts", "configs/scenes/zoo_hero.yaml", -90.0, 0.330),
    ("dji_inspire_3_parts", "configs/scenes/zoo_hero.yaml", -90.0, 0.917),
    ("dji_matrice_100_parts", "configs/scenes/zoo_hero.yaml", -90.0, 0.690),
    ("dji_matrice_300_rtk_parts", "configs/scenes/zoo_hero.yaml", -90.0, 1.032),
)
#: The span the range band (``--near-m``/``--far-m``) is stated for: the Phantom 4's.
REFERENCE_SPAN_M = 0.618


def scene_longitude_deg(scene: str | None) -> float:
    """The scene's site longitude, which turns a local solar hour into the UTC hour the renderer
    takes; 0 when no scene is named (UTC is then solar time)."""
    if not scene:
        return 0.0
    import yaml

    path = Path(scene) if Path(scene).is_absolute() else REPO / scene
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    return float(config["scene"]["site"]["longitude_deg"])


def draw_airframe(index: int, args: argparse.Namespace) -> dict[str, object]:
    """One clip's airframe (EV.25): the asset, the scene it flies in and its nose. With
    ``--airframes all`` the caller passes the clip's place in a shuffled cycle through
    :data:`AIRFRAMES`, so a set of n clips flies min(n, 7) different aircraft; a name keeps that
    one for every clip."""
    chosen = getattr(args, "airframes", "all")
    if chosen != "all":
        names = [a[0] for a in AIRFRAMES]
        if chosen not in names:
            raise ValueError(f"unknown airframe {chosen!r}; known: {names}")
        index = names.index(chosen)
    asset, scene, nose, span = AIRFRAMES[index]
    scale = span / REFERENCE_SPAN_M
    return {
        "asset": asset,
        "scene": scene,
        "nose_deg": nose,
        "near_m": round(args.near_m * scale, 2),
        "far_m": round(args.far_m * scale, 2),
    }


def draw_camera(rng: np.random.Generator, args: argparse.Namespace) -> dict[str, float | None]:
    """One clip's camera unit (EV.23): how far from settled its housing starts, and how much its
    lens vignettes. Both are drawn even when fixed, so the stream does not depend on the flags."""
    housing = round(float(rng.uniform(*HOUSING_START_K)), 2)
    corner = round(float(rng.uniform(*CORNER_ILLUMINATION)), 3)
    chosen_housing = getattr(args, "housing_start", "drawn")
    chosen_corner = getattr(args, "corner_illumination", "drawn")
    return {
        "housing_start_k": housing if chosen_housing == "drawn" else float(chosen_housing),
        "corner_illumination": (
            corner
            if chosen_corner == "drawn"
            else (None if chosen_corner == "sensor" else float(chosen_corner))
        ),
    }


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
    hour_rng = np.random.default_rng([args.seed, 24])
    camera_rng = np.random.default_rng([args.seed, 23])
    airframe_rng = np.random.default_rng([args.seed, 26])
    longitude = scene_longitude_deg(getattr(args, "scene", None))
    runs: list[dict[str, Any]] = []
    for kind, count, frames in (
        ("clear", args.clear_runs, args.frames_per_run),
        ("cloud", args.cloud_runs, args.cloud_frames_per_run),
    ):
        # EV.24: the hours are stratified in the site's local solar time -- run i of n falls in
        # its own 24/n-hour slot, the slots in a shuffled order -- so a set of six is not five
        # nights and a morning, as the first one was. The position inside the slot is the old
        # uniform draw, so the main stream (and every seed after it) is unchanged.
        slots = hour_rng.permutation(count) if count else np.zeros(0, dtype=int)
        cycle = np.concatenate(
            [airframe_rng.permutation(len(AIRFRAMES)) for _ in range(count // len(AIRFRAMES) + 1)]
        )
        for i in range(count):
            weather = "clear" if kind == "clear" else str(rng.choice(CLOUD_REGIMES))
            weather_seed = int(rng.integers(0, 2**31 - 1))
            within = float(rng.uniform(0.0, 24.0)) / 24.0
            local = (float(slots[i]) + within) * 24.0 / count
            runs.append(
                {
                    "name": f"{kind}_{i:03d}",
                    "frames": frames,
                    "weather": weather,
                    "weather_seed": weather_seed,
                    # Every hour: the real sets were filmed by day and by night, and a sunlit
                    # airframe and one that has cooled to the night air are different targets.
                    "weather_hour": round((local - longitude / 15.0) % 24.0, 2),
                    "local_solar_hour": round(local, 2),
                    "scatter_seed": int(rng.integers(0, 2**31 - 1)),
                    "focus_m": (
                        draw_focus(focus_rng, args.near_m, args.far_m)
                        if args.focus == "drawn"
                        else args.focus
                    ),
                    **draw_camera(camera_rng, args),
                    **draw_airframe(int(cycle[i]), args),
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
        "--asset", run.get("asset", args.asset),
        "--scene", run.get("scene", args.scene),
        *([] if run.get("nose_deg") is None else ["--nose-deg", str(run["nose_deg"])]),
        "--tilt-sigma-deg", str(getattr(args, "tilt_sigma_deg", 0.0)),
        "--sensor", args.sensor,
        "--frames", str(run["frames"]),
        "--scatter-seed", str(run["scatter_seed"]),
        "--near-m", str(run.get("near_m", args.near_m)),
        "--far-m", str(run.get("far_m", args.far_m)),
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
        "--housing-start-k", str(run.get("housing_start_k", 0.0)),
        *(
            []
            if run.get("corner_illumination") is None
            else ["--corner-illumination", str(run["corner_illumination"])]
        ),
        "--out", str(out),
    ]  # fmt: skip


def collect(run_dir: Path, name: str, out: Path, boxes: str = "drawn") -> int:
    """Copy one run's display frames and labels into the YOLO layout. Returns frames collected.

    ``boxes``: ``drawn`` takes the box a person would draw (EV.22, ``labels_drawn``) where the run
    wrote one, ``tight`` the truth mask's extent (``labels``).
    """
    import cv2

    images, labels = out / "images" / "train", out / "labels" / "train"
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)
    n = 0
    for display in sorted(run_dir.glob("frame_*_display8.png")):
        stem = display.name.removesuffix("_display8.png")
        label = run_dir / "labels" / f"{stem}.txt"
        drawn = run_dir / "labels_drawn" / f"{stem}.txt"
        if boxes == "drawn" and drawn.exists():
            label = drawn
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
    # EV.10: the real sky-only frames' mount elevation reads 0.2-10 deg by eye off the burnt-in
    # readout (EV.19 report); the aim is raised wherever a frame would show the horizon.
    parser.add_argument("--elevation-low-deg", type=float, default=1.0)
    parser.add_argument("--elevation-high-deg", type=float, default=12.0)
    parser.add_argument("--aim-jitter", type=float, default=0.6)
    parser.add_argument(
        "--focus",
        default="drawn",
        help="the lens focus per clip: 'drawn' (EV.21: half at infinity, half at a distance; "
        "draw_focus), 'inf', a distance in metres, or 'sensor' for the sensor file's own",
    )
    parser.add_argument(
        "--airframes",
        default="all",
        help="'all' draws an airframe per clip from AIRFRAMES (EV.25); an asset name keeps one",
    )
    parser.add_argument(
        "--tilt-sigma-deg",
        type=float,
        default=8.0,
        help="the aircraft's pitch and roll scale per clip, degrees (EV.25; ESTIMATED: a camera "
        "multirotor cruises at 5-20 deg of tilt); 0 flies level",
    )
    parser.add_argument(
        "--boxes",
        default="drawn",
        choices=("drawn", "tight"),
        help="the label box: 'drawn' as a person would (EV.22: what is visible, plus their "
        "margin), or 'tight', the truth mask's extent",
    )
    parser.add_argument(
        "--housing-start",
        default="drawn",
        help="the camera housing's start offset from settled per clip, K: 'drawn' (EV.23, "
        "HOUSING_START_K) or a number; 0 is the settled camera",
    )
    parser.add_argument(
        "--corner-illumination",
        default="drawn",
        help="the lens's relative illumination at the corner per clip: 'drawn' (EV.23, "
        "CORNER_ILLUMINATION), a number in (0, 1], or 'sensor' for the sensor file's own",
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
        collected = collect(run_dir, run["name"], out, args.boxes)
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
