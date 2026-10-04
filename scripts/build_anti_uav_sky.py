#!/usr/bin/env python3
"""Build the sky-only Anti-UAV detection set the validation phase trains and tests on.

    python scripts/build_anti_uav_sky.py cv      # how good is the sky/clutter classifier?
    python scripts/build_anti_uav_sky.py build   # train it on every label, filter, export YOLO
    python scripts/build_anti_uav_sky.py score frame.png clip.mp4 ...   # p(sky) with saved weights

Anti-UAV RGBT's infrared frames are about half sky and half city. irsim renders a drone against
sky and cloud, so frames with buildings, cranes or wires are removed from the train, val **and**
test splits before any detector is trained (see :mod:`irsim_eval.antiuav` for why and for the
keep/drop rule). The decision is per frame, by a ResNet-18 trained on the hand labels in
``data/validation/anti_uav_rgbt_sky_labels.csv``.

``cv`` is the measurement that says whether the filter can be trusted: five folds **split by
clip**, so the classifier is always scored on clips it has not seen. ``build`` then trains on every
label -- the classifier is a curation tool, not the detector under test -- scores every
``--stride``-th frame of every clip, and writes::

    <out>/images/{train,val,test}/<clip>_<frame>.png     8-bit grey, as decoded
    <out>/labels/{train,val,test}/<clip>_<frame>.txt     YOLO, one class (drone); empty = no drone
    <out>/anti_uav_sky.yaml                              the ultralytics dataset file
    <out>/scores.json                                    every sampled frame's p(sky)
    <out>/manifest.json                                  counts, threshold, label-file hash

``score`` is the classifier on its own, for frames this script did not build: it loads
``<out>/sky_classifier.pt`` (a torchvision ResNet-18 ``state_dict`` with a one-logit head; input
640x512 8-bit grey, see :func:`_to_input`) and prints one ``path frame p_sky`` line per image, or
per ``--stride``-th frame of a video.

**This script needs torch, torchvision and OpenCV**, which the default environment does not
install; ``src/irsim`` may never import them (``tests/unit/test_layering.py``).

**Licence.** Anti-UAV RGBT states no licence for its frames. Measuring on one machine is what this
does; the output directory is under the git-ignored ``datasets/`` and nothing in it, and no
detector trained on it, may be published (ADR 0068).

docs/physics-model.md §15 T5
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np

from irsim_eval.antiuav import (
    SPLITS,
    accept_sky_frames,
    load_sky_labels,
    sampled_frames,
    yolo_line,
)

REPO = Path(__file__).resolve().parents[1]
LABELS = REPO / "data/validation/anti_uav_rgbt_sky_labels.csv"
#: Frames either side of a labelled one that inherit its label for training (0.2 s and 0.4 s at
#: 20 Hz): the scene has not changed, the noise and the drone's position have.
NEIGHBOUR_OFFSETS = (-8, -4, 0, 4, 8)
EPOCHS = 8
BATCH = 32


def _deps() -> tuple[Any, Any, Any]:
    try:
        import cv2
        import torch
        import torchvision
    except ModuleNotFoundError as exc:
        raise SystemExit(f"this script needs torch, torchvision and OpenCV ({exc})") from None
    return cv2, torch, torchvision


def _read_frames(cv2: Any, video: Path, frames: list[int]) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out = []
    for f in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, min(max(f, 0), n - 2))
        ok, img = cap.read()
        if not ok:
            raise RuntimeError(f"{video}: frame {f} does not decode")
        out.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    return out


def _to_input(torch: Any, x: Any, device: str) -> Any:
    """uint8 grey (N, H, W) -> the 3-channel float the ImageNet-pretrained stem expects. Full
    640x512: at half resolution a rooftop in a bottom corner disappears, and that is the clutter
    the classifier most often has to find."""
    return x.to(device).float().div(255).sub(0.45).div(0.225).unsqueeze(1).repeat(1, 3, 1, 1)


def _fit(torch: Any, torchvision: Any, x: np.ndarray, y: np.ndarray, device: str) -> Any:
    nn = torch.nn
    xt, yt = torch.from_numpy(x), torch.from_numpy(y.astype(np.float32))
    model = torchvision.models.resnet18(weights="IMAGENET1K_V1")
    model.fc = nn.Linear(512, 1)
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), 3e-4, weight_decay=1e-2)
    steps = EPOCHS * ((len(xt) + BATCH - 1) // BATCH)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=steps)
    for _ in range(EPOCHS):
        model.train()
        perm = torch.randperm(len(xt))
        for i in range(0, len(perm), BATCH):
            b = perm[i : i + BATCH]
            xb, yb = _to_input(torch, xt[b], device), yt[b].to(device)
            if random.random() < 0.5:
                xb = xb.flip(3)
            # Gain and offset jitter: the camera's AGC moves both between clips, and neither says
            # anything about whether a building is in the frame.
            gain = 1 + 0.2 * (torch.rand(len(b), 1, 1, 1, device=device) - 0.5)
            offset = 0.3 * (torch.rand(len(b), 1, 1, 1, device=device) - 0.5)
            loss = nn.functional.binary_cross_entropy_with_logits(
                model(xb * gain + offset).squeeze(1), yb
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
    return model.eval()


def _predict(torch: Any, model: Any, x: np.ndarray, device: str) -> np.ndarray:
    xt = torch.from_numpy(x)
    with torch.no_grad():
        parts = [
            torch.sigmoid(model(_to_input(torch, xt[i : i + BATCH], device)).squeeze(1)).cpu()
            for i in range(0, len(xt), BATCH)
        ]
    return torch.cat(parts).numpy().astype(np.float64) if parts else np.zeros(0)


def _seed(torch: Any, seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def _load_labelled(cv2: Any, src: Path) -> tuple[np.ndarray, np.ndarray, list[tuple[str, str]]]:
    """Every confident label with its temporal neighbours: (N, 5, H, W) uint8, (N,) sky?, clips."""
    labels = [lab for lab in load_sky_labels(LABELS) if lab.label != "unsure"]
    x = np.stack(
        [
            np.stack(
                _read_frames(
                    cv2,
                    src / lab.split / lab.clip / "infrared.mp4",
                    [lab.frame + o for o in NEIGHBOUR_OFFSETS],
                )
            )
            for lab in labels
        ]
    )
    y = np.array([lab.label == "sky" for lab in labels])
    return x, y, [(lab.split, lab.clip) for lab in labels]


def cross_validate(args: argparse.Namespace) -> dict[str, Any]:
    cv2, torch, torchvision = _deps()
    _seed(torch, args.seed)
    x, y, clip_of = _load_labelled(cv2, args.src)
    clips = sorted(set(clip_of))
    random.shuffle(clips)
    fold_of = {c: i % 5 for i, c in enumerate(clips)}
    fold = np.array([fold_of[c] for c in clip_of])
    centre = NEIGHBOUR_OFFSETS.index(0)
    p = np.zeros(len(y))
    for k in range(5):
        tr, te = np.where(fold != k)[0], np.where(fold == k)[0]
        model = _fit(
            torch,
            torchvision,
            x[tr].reshape(-1, *x.shape[2:]),
            np.repeat(y[tr], x.shape[1]),
            args.device,
        )
        p[te] = _predict(torch, model, x[te, centre], args.device)
        print(f"fold {k}: accuracy {np.mean((p[te] > 0.5) == y[te]):.4f}", file=sys.stderr)
    keep = p >= args.threshold
    result = {
        "labelled_frames": len(y),
        "accuracy_at_0.5": float(np.mean((p > 0.5) == y)),
        "threshold": args.threshold,
        "clutter_frames": int(np.sum(~y)),
        "clutter_accepted": int(np.sum(keep & ~y)),
        "sky_frames": int(np.sum(y)),
        "sky_kept_fraction": float(np.sum(keep & y) / np.sum(y)),
    }
    print(json.dumps(result, indent=2))
    return result


def build(args: argparse.Namespace) -> None:
    cv2, torch, torchvision = _deps()
    _seed(torch, args.seed)
    out: Path = args.out
    x, y, _ = _load_labelled(cv2, args.src)
    model = _fit(
        torch, torchvision, x.reshape(-1, *x.shape[2:]), np.repeat(y, x.shape[1]), args.device
    )
    del x
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "sky_classifier.pt")

    scores: dict[str, dict[str, list[float]]] = {}
    counts: dict[str, dict[str, int]] = {}
    for split in SPLITS:
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        c = {"clips": 0, "clips_kept": 0, "sampled": 0, "kept": 0, "kept_with_drone": 0}
        scores[split] = {}
        for clip_dir in sorted((args.src / split).iterdir()):
            truth = json.loads((clip_dir / "infrared.json").read_text())
            cap = cv2.VideoCapture(str(clip_dir / "infrared.mp4"))
            frames: list[np.ndarray] = []
            while True:
                ok, img = cap.read()
                if not ok:
                    break
                frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
            n = min(len(frames), len(truth["exist"]))
            idx = sampled_frames(n, args.stride)
            p = _predict(torch, model, np.stack([frames[i] for i in idx]), args.device)
            keep = accept_sky_frames(p, args.threshold, args.neighbour_floor)
            scores[split][clip_dir.name] = [round(float(v), 4) for v in p]
            c["clips"] += 1
            c["clips_kept"] += int(keep.any())
            c["sampled"] += len(idx)
            for i in idx[keep]:
                h, w = frames[i].shape
                line = yolo_line(truth["gt_rect"][i], w, h) if truth["exist"][i] else None
                stem = f"{clip_dir.name}_{i:05d}"
                cv2.imwrite(str(out / "images" / split / f"{stem}.png"), frames[i])
                (out / "labels" / split / f"{stem}.txt").write_text(f"{line}\n" if line else "")
                c["kept"] += 1
                c["kept_with_drone"] += int(line is not None)
        counts[split] = c
        print(split, c, file=sys.stderr)

    (out / "anti_uav_sky.yaml").write_text(
        f"path: {out.resolve()}\ntrain: images/train\nval: images/val\ntest: images/test\n"
        "names:\n  0: drone\n"
    )
    (out / "scores.json").write_text(json.dumps(scores))
    manifest = {
        "source": "Anti-UAV RGBT (Jiang et al. 2021, arXiv 2101.08466), infrared stream",
        "licence": "unstated -- local measurement only, do not redistribute (ADR 0068)",
        "labels_sha256": hashlib.sha256(LABELS.read_bytes()).hexdigest(),
        "stride": args.stride,
        "threshold": args.threshold,
        "neighbour_floor": args.neighbour_floor,
        "seed": args.seed,
        "counts": counts,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


def load_classifier(torch: Any, torchvision: Any, weights: Path, device: str) -> Any:
    """The saved sky/clutter classifier, ready to score: ResNet-18, one logit, eval mode."""
    model = torchvision.models.resnet18(weights=None)
    model.fc = torch.nn.Linear(512, 1)
    model.load_state_dict(torch.load(weights, map_location=device))
    return model.to(device).eval()


def score(args: argparse.Namespace) -> None:
    cv2, torch, torchvision = _deps()
    weights = args.weights or args.out / "sky_classifier.pt"
    model = load_classifier(torch, torchvision, weights, args.device)
    for path in args.inputs:
        frames: list[tuple[int, np.ndarray]] = []
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is not None:
            frames.append((0, image))
        else:
            cap = cv2.VideoCapture(str(path))
            index = 0
            while True:
                ok, img = cap.read()
                if not ok:
                    break
                if index % args.stride == 0:
                    frames.append((index, cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)))
                index += 1
        if not frames:
            raise SystemExit(f"{path}: neither an image nor a video OpenCV can read")
        # The classifier was trained at 640x512 and a corner rooftop is a few pixels of it.
        stack = np.stack(
            [cv2.resize(f, (640, 512), interpolation=cv2.INTER_AREA) for _, f in frames]
        )
        for (index, _), p in zip(frames, _predict(torch, model, stack, args.device), strict=True):
            print(f"{path} {index} {p:.4f}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("command", choices=("cv", "build", "score"))
    parser.add_argument("inputs", nargs="*", type=Path, help="score: images or videos")
    parser.add_argument("--weights", type=Path, default=None, help="score: default <out>'s")
    parser.add_argument("--src", type=Path, default=REPO / "datasets/Anti-UAV-RGBT")
    parser.add_argument("--out", type=Path, default=REPO / "datasets/anti_uav_sky")
    parser.add_argument("--stride", type=int, default=5, help="keep every Nth frame (20 Hz source)")
    parser.add_argument("--threshold", type=float, default=0.99)
    parser.add_argument("--neighbour-floor", type=float, default=0.5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20261004)
    args = parser.parse_args(argv)
    if args.command == "cv":
        cross_validate(args)
    elif args.command == "score":
        score(args)
    else:
        build(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
