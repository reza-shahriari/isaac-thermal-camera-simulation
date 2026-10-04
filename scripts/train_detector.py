#!/usr/bin/env python3
r"""Train one small detector under one of §15's transfer protocols, and score it on real frames.

    python scripts/train_detector.py --protocol real_to_real --real <dir> --out <run>
    python scripts/train_detector.py --protocol synthetic_to_real --real <dir> \
        --synthetic <dir> --out <run>
    python scripts/train_detector.py --protocol mixed_to_real --real <dir> \
        --synthetic <dir> --out <run>

``--real`` is the sky-only Anti-UAV set written by ``scripts/build_anti_uav_sky.py`` and
``--synthetic`` an irsim render set in the same YOLO layout (``images/{train,val}``,
``labels/{train,val}``). **Whatever the protocol trains on, the checkpoint is selected on the real
validation split and the number reported is on the real test split** -- the three arms differ in
their training images and in nothing else: same model, same image size, same epochs, same seed,
same augmentation. A sim-to-real comparison in which the arms were also tuned differently measures
the tuning.

The detector is YOLO11n through ``ultralytics`` (AGPL-3.0) -- the owner's choice for this phase,
which settles roadmap open question 3 for local measurement. Nothing trained here is shipped: the
real frames carry no stated licence (ADR 0068).

**This script needs torch and ultralytics**, which the default environment does not install;
``src/irsim`` may never import them (``tests/unit/test_layering.py``).

docs/physics-model.md §15 T5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from irsim_eval.detection import PROTOCOLS

#: The three protocols this phase runs. ``real_to_synthetic`` is scored by pointing ``--real`` at a
#: render set's test split and is not a training arrangement of its own.
TRAINS_ON: dict[str, tuple[str, ...]] = {
    "real_to_real": ("real",),
    "synthetic_to_real": ("synthetic",),
    "mixed_to_real": ("real", "synthetic"),
}


def _dataset_yaml(out: Path, train_dirs: list[Path], real: Path) -> Path:
    """The ultralytics dataset file for one arm: its training images, the real val and test."""
    out.mkdir(parents=True, exist_ok=True)
    lines = ["train:"]
    lines += [f"  - {(d / 'images' / 'train').resolve()}" for d in train_dirs]
    lines += [
        f"val: {(real / 'images' / 'val').resolve()}",
        f"test: {(real / 'images' / 'test').resolve()}",
        "names:",
        "  0: drone",
        "",
    ]
    path = out / "data.yaml"
    path.write_text("\n".join(lines))
    return path


def _metrics(result: Any) -> dict[str, float]:
    box = result.box
    return {
        "precision": float(box.mp),
        "recall": float(box.mr),
        "mAP50": float(box.map50),
        "mAP50_95": float(box.map),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--protocol", choices=sorted(TRAINS_ON), required=True)
    parser.add_argument("--real", type=Path, required=True)
    parser.add_argument("--synthetic", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--weights", default="yolo11n.pt", help="starting checkpoint")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=20261004)
    args = parser.parse_args(argv)

    sources = {"real": args.real, "synthetic": args.synthetic}
    missing = [s for s in TRAINS_ON[args.protocol] if sources[s] is None]
    if missing:
        parser.error(f"protocol {args.protocol} needs --{' and --'.join(missing)}")
    print(f"protocol {args.protocol}: {PROTOCOLS[args.protocol]}", file=sys.stderr)  # type: ignore[index]

    try:
        from ultralytics import YOLO
    except ModuleNotFoundError:
        raise SystemExit("this script needs torch and ultralytics installed") from None

    out: Path = args.out.resolve()
    data = _dataset_yaml(out, [sources[s] for s in TRAINS_ON[args.protocol]], args.real)
    model = YOLO(args.weights)
    model.train(
        data=str(data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        seed=args.seed,
        deterministic=True,
        project=str(out),
        name="train",
        exist_ok=True,
        plots=False,
    )
    best = YOLO(str(out / "train" / "weights" / "best.pt"))
    common = {"data": str(data), "imgsz": args.imgsz, "batch": args.batch, "device": args.device}
    scores = {
        split: _metrics(
            best.val(
                split=split, project=str(out), name=split, exist_ok=True, plots=False, **common
            )
        )
        for split in ("val", "test")
    }
    report = {
        "protocol": args.protocol,
        "trained_on": [str(sources[s]) for s in TRAINS_ON[args.protocol]],
        "real": str(args.real),
        "weights": args.weights,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "seed": args.seed,
        "real_val": scores["val"],
        "real_test": scores["test"],
    }
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
