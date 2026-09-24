#!/usr/bin/env python3
"""Roadmap ME.6: the Tier 4 acceptance report. Exits non-zero when a target is exceeded.

    python scripts/validation_report.py --real <dir> --synthetic <dir>
    python scripts/validation_report.py --self-test

Frames are 8-bit ``.npy`` or ``.png`` files -- the display domain, because ADR 0068 established
that is the only domain the public sky-target data exists in. Every DN8 target from
:class:`irsim.validation.compare.Tier4Targets` is checked, plus the discriminator's gap score, and
**everything that cannot be measured is printed as `untestable` rather than omitted**: on this
project's data most of Tier 4 is untestable, and a report that hid that would be the single most
misleading thing it could produce.

``--self-test`` runs the report on synthetic-versus-itself, which is the control: two halves of one
distribution must pass every check and give an AUC near 0.5. It needs no data and is what the unit
test drives.

Needs no GPU, no Isaac Sim and no network.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from dataclasses import asdict
from typing import Any

import numpy as np

from irsim.validation.compare import Tier4Report, Tier4Targets, compare_clips
from irsim_eval.discriminator import gap_score

REPO = pathlib.Path(__file__).resolve().parents[1]

#: Which step a failing check points at first. ME.6's exit criterion asks that every failing metric
#: name the step responsible, and this is that mapping -- a judgement, written down, so a reader can
#: disagree with it rather than guess. It is deliberately **not** a list of physics steps: on
#: public 8-bit lossy data the signal path is the first suspect for three of the five, and blaming
#: the radiometry for an encoder's work is the single easiest mistake to make here.
ATTRIBUTION: dict[str, str] = {
    "histogram EMD (DN8 codes)": (
        "M12.1's recorder model first (`agc: linear` standing in for an unverified Y16 -> 8-bit "
        "conversion), then the scene priors, which ME.5 could not measure at all. The radiometry "
        "is the last suspect: an unknown display mapping can move a histogram by tens of codes "
        "without any physics changing."
    ),
    "PSD shape ratio": (
        "M12.1's codec settings and MS.3's cloud model. ME.2b measured x264 at CRF 18 removing "
        "95 % of a clip's temporal noise, so a mismatch here is an encoder mismatch until the "
        "encoder is matched; after that it is the spatial structure of the modelled sky."
    ),
    "contrast ratio agreement": (
        "M11.3/M11.10 (how much light reaches the target) and the target's own material."
    ),
    "ESF width agreement": "M10.1b/M9.13 (motion and the membrane) and the optics MTF cascade.",
    "discriminator AUC": (
        "read the heaviest features in the note. `noise_scale` or `spectrum_degenerate` at the top "
        "means the *signal path* separates the sets -- M12.1's recorder and codec -- not the "
        "physics. A physics gap shows up as `psd_slope`, `gradient_*` or `lag1_autocorrelation`."
    ),
}


def _load_sequences(directory: pathlib.Path, limit: int | None) -> list[list[np.ndarray]]:
    """Frames from the ME.1 canonical layout, **grouped by clip** -- one list per sequence.

    This is what `generate_matched_scenario.py` writes, so a synthetic set needs no conversion
    step between being rendered and being compared, and nothing can go wrong in one.

    The grouping is the whole point of EV.1: a clip is the unit a statistic may be measured on,
    and flattening the sequences into one list is what let the report median a composite image
    that neither set contains. ``limit`` counts frames across the set, as it always did.
    """
    from irsim_eval.data import read_sequence

    roots = sorted(d for d in directory.iterdir() if (d / "sequence.json").is_file())
    if (directory / "sequence.json").is_file() and not roots:
        roots = [directory]
    clips: list[list[np.ndarray]] = []
    seen = 0
    for root in roots:
        clip: list[np.ndarray] = []
        for frame in read_sequence(root):
            clip.append(np.asarray(frame.image))
            seen += 1
            if limit is not None and seen >= limit:
                break
        if clip:
            clips.append(clip)
        if limit is not None and seen >= limit:
            break
    return clips


def _load_dataset_clips(name: str, limit: int, max_frames: int) -> list[list[np.ndarray]]:
    """Decode real clips straight out of the indexed archive -- no conversion step on disk.

    Reads exactly what ME.5 measured: the luma plane, at the rate the container states (never the
    index's), with the missing colour-range flag left alone rather than guessed at.
    """
    import tempfile
    import zipfile

    from irsim_eval.decode import decode_gray8
    from irsim_eval.fetch import target_dir

    archives = sorted(target_dir(name).glob("*.zip"))
    if not archives:
        raise SystemExit(
            f"no archive for {name}; fetch it with scripts/fetch_validation_data.py --set {name}"
        )
    clips: list[list[np.ndarray]] = []
    with zipfile.ZipFile(archives[0]) as zf:
        members = sorted(
            n for n in zf.namelist() if "/Video_IR/" in n and n.lower().endswith(".mp4")
        )[:limit]
        with tempfile.TemporaryDirectory() as tmp:
            scratch = pathlib.Path(tmp) / "clip.mp4"
            for member in members:
                scratch.write_bytes(zf.read(member))
                clips.append(list(decode_gray8(scratch, max_frames=max_frames)))
    return clips


def _load_frames(directory: pathlib.Path, limit: int | None) -> list[list[np.ndarray]]:
    """Clips from a directory. A bare directory of frames is **one** clip, and says so.

    That is not a workaround: a folder of loose frames carries no clip boundaries, so the honest
    reading is that it is a single clip, and the report then says "1 clip pair" rather than
    implying a distribution it does not have.
    """
    if (directory / "sequence.json").is_file() or any(
        (d / "sequence.json").is_file() for d in directory.iterdir() if d.is_dir()
    ):
        return _load_sequences(directory, limit)
    paths = sorted(p for p in directory.iterdir() if p.suffix.lower() in {".npy", ".png"})
    if limit is not None:
        paths = paths[:limit]
    if not paths:
        raise SystemExit(f"no .npy or .png frames under {directory}")
    frames: list[np.ndarray] = []
    for path in paths:
        if path.suffix.lower() == ".npy":
            array = np.load(path)
        else:
            from irsim_eval.data import _decode_png  # the validation extra lives there

            array = _decode_png(path)
        array = np.asarray(array)
        if array.ndim == 3:
            array = array[..., 0]  # a display frame stored RGBA; the channels are equal in grey
        if array.dtype != np.uint8:
            raise SystemExit(f"{path.name}: Tier 4 is measured on 8-bit frames, got {array.dtype}")
        frames.append(array)
    return [frames]


def _patches(frames: list[np.ndarray], size: int, per_frame: int, seed: int) -> list[np.ndarray]:
    """Fixed-size patches on a deterministic grid, so a re-run compares the same pixels."""
    rng = np.random.default_rng(seed)
    out: list[np.ndarray] = []
    for frame in frames:
        h, w = frame.shape
        if h < size or w < size:
            raise SystemExit(f"a {size}x{size} patch does not fit in a {h}x{w} frame")
        for _ in range(per_frame):
            y = int(rng.integers(0, h - size + 1))
            x = int(rng.integers(0, w - size + 1))
            out.append(frame[y : y + size, x : x + size].astype(np.float64))
    return out


def _mosaic(frames: list[np.ndarray]) -> np.ndarray:
    """One frame standing for the set, for the whole-frame statistics.

    The temporal median rather than the mean: it is what ME.2b's window finder already uses, and a
    target that moves disappears from it, so the histogram and the spectrum describe the
    *background* the two sets are really being compared on.
    """
    shapes = {f.shape for f in frames}
    if len(shapes) != 1:
        raise SystemExit(f"frames are not all the same size: {sorted(shapes)}")
    return np.median(np.stack(frames).astype(np.float64), axis=0)


def _self_test_frames(
    seed: int = 20260915, n: int = 24, clips: int = 3
) -> tuple[list[list[np.ndarray]], list[list[np.ndarray]]]:
    """Two halves of one distribution: the control every acceptance report needs.

    If the report cannot pass this, its thresholds are wrong and nothing it says about a real
    comparison means anything.
    """
    rng = np.random.default_rng(seed)
    base = rng.normal(0.0, 1.0, (64, 64))
    smooth = np.zeros_like(base)
    for dy in range(-4, 5):
        for dx in range(-4, 5):
            smooth += np.roll(np.roll(base, dy, 0), dx, 1)
    smooth /= smooth.std()
    scene = 110.0 + 12.0 * smooth
    # Several clips a side, since EV.1 made the clip the unit: a control with one clip each would
    # exercise neither the pairing nor the within-set spread, which is most of what there is to
    # get wrong. Every clip is drawn from the same distribution, so the control's own answer is
    # still "these two sets are the same".
    made = [
        [
            np.clip(scene + rng.normal(0.0, 3.0, scene.shape), 0, 255).astype(np.uint8)
            for _ in range(n)
        ]
        for _ in range(2 * clips)
    ]
    return made[:clips], made[clips:]


def _run(
    real: list[list[np.ndarray]],
    synthetic: list[list[np.ndarray]],
    *,
    patch: int,
    per_frame: int,
    seed: int,
    targets: Tier4Targets,
) -> Tier4Report:
    """Compare two sets of clips. The whole-frame checks are per clip pair, never pooled (EV.1)."""
    flat_real = [f for clip in real for f in clip]
    flat_synthetic = [f for clip in synthetic for f in clip]
    result = gap_score(
        _patches(flat_real, patch, per_frame, seed),
        _patches(flat_synthetic, patch, per_frame, seed + 1),
        seed=seed,
    )
    report = compare_clips(
        [_mosaic(clip) for clip in real],
        [_mosaic(clip) for clip in synthetic],
        targets=targets,
        auc=result.auc,
    )
    for check in report.checks:
        if check.name == "discriminator AUC":
            report.checks[report.checks.index(check)] = type(check)(
                name=check.name,
                measured=check.measured,
                target=check.target,
                verdict=check.verdict,
                note=(
                    f"{result.n_real} vs {result.n_synthetic} patches, {result.folds}-fold "
                    f"cross-validated; {result.z:+.2f} null sigma from chance "
                    f"(sigma = {result.null_sigma:.3f}); heaviest features: {result.top()}"
                ),
            )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real", type=pathlib.Path)
    parser.add_argument(
        "--real-from-set",
        default=None,
        help="decode real clips straight out of an indexed archive (e.g. halmstad_drone_detection)",
    )
    parser.add_argument("--real-clips", type=int, default=6)
    parser.add_argument("--real-frames-per-clip", type=int, default=60)
    parser.add_argument("--synthetic", type=pathlib.Path)
    parser.add_argument("--self-test", action="store_true", help="synthetic vs itself: the control")
    parser.add_argument("--limit", type=int, default=None, help="frames per set")
    parser.add_argument("--patch", type=int, default=32)
    parser.add_argument("--patches-per-frame", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--out", type=pathlib.Path, default=None, help="write markdown + json here")
    args = parser.parse_args(argv)

    if args.self_test:
        real, synthetic = _self_test_frames(args.seed)
        label = "self-test (synthetic vs itself)"
    else:
        if not args.synthetic or not (args.real or args.real_from_set):
            parser.error(
                "--synthetic and one of --real / --real-from-set are required "
                "unless --self-test is given"
            )
        if args.real_from_set:
            real = _load_dataset_clips(
                args.real_from_set, args.real_clips, args.real_frames_per_clip
            )
            source = f"{args.real_from_set} ({args.real_clips} clips)"
        else:
            real = _load_frames(args.real, args.limit)
            source = str(args.real)
        synthetic = _load_frames(args.synthetic, args.limit)
        label = f"{source} vs {args.synthetic}"

    report = _run(
        real,
        synthetic,
        patch=args.patch,
        per_frame=args.patches_per_frame,
        seed=args.seed,
        targets=Tier4Targets(),
    )
    print(f"# Tier 4 acceptance -- {label}\n")
    print(report.as_markdown())
    print()
    if report.failed:
        print("Where each failure points first:\n")
        for check in report.failed:
            print(f"- **{check.name}** -- {ATTRIBUTION.get(check.name, 'unattributed')}")
        print()
    if report.untestable:
        print(f"{len(report.untestable)} check(s) untestable on this input:")
        for check in report.untestable:
            print(f"  - {check.name}: {check.note}")
        print()
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        body = [f"# Tier 4 acceptance -- {label}", "", report.as_markdown(), ""]
        if report.failed:
            body += ["## Where each failure points first", ""]
            body += [
                f"- **{c.name}** -- {ATTRIBUTION.get(c.name, 'unattributed')}"
                for c in report.failed
            ]
            body += [""]
        if report.untestable:
            body += ["## Not testable on this input", ""]
            body += [f"- **{c.name}** -- {c.note}" for c in report.untestable]
            body += [""]
        (args.out / "tier4-acceptance.md").write_text("\n".join(body), "utf-8")
        payload: dict[str, Any] = {
            "label": label,
            "checks": [c.as_dict() for c in report.checks],
            "attribution": {c.name: ATTRIBUTION.get(c.name) for c in report.failed},
            # The per-clip-pair distribution behind each whole-frame check (EV.1). The verdict is
            # the median; this is what a reader who wants a stricter rule needs, and it cannot be
            # recovered from the note's prose.
            "paired": {k: asdict(v) for k, v in report.paired.items()},
            "passed": report.passed,
        }
        (args.out / "tier4-acceptance.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True), "utf-8"
        )
        print(f"wrote {args.out / 'tier4-acceptance'}.md and .json")
    if report.failed:
        print(f"FAILED: {', '.join(c.name for c in report.failed)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
