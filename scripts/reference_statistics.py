#!/usr/bin/env python3
"""Roadmap ME.5: measure the public reference sets and write the bands the simulator must land in.

    python scripts/reference_statistics.py --list
    python scripts/reference_statistics.py --set halmstad_drone_detection
    python scripts/reference_statistics.py --set halmstad_drone_detection --limit 20

Writes ``docs/validation/reference-stats-<date>.md`` and the matching ``.json``. Both regenerate
deterministically from the same bytes: the clips are visited in sorted order, the bootstrap is
seeded, and the archive's SHA-256 goes in the header so a reader can tell whether a number was
measured on the same data they have.

Every statistic carries N, a 95 % confidence interval and its floor, and everything that could
*not* be measured is written down with the reason, because on this data that is most of the result.

Needs ffmpeg on PATH. Needs no GPU, no Isaac Sim and no network.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import sys
import tempfile
import zipfile
from typing import Any

import numpy as np

from irsim_eval.decode import decode_gray8, ffmpeg_available, probe_clip, range_ambiguity_codes
from irsim_eval.fetch import sha256_of, target_dir
from irsim_eval.manifest import load_manifest
from irsim_eval.reference import (
    BOOTSTRAP_RESAMPLES,
    REPORT_SCHEMA_VERSION,
    ClipMeasurement,
    Section,
    measure_clip,
    refusal,
    summarise,
)

REPO = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO / "docs" / "validation"

#: Which archive member holds the infrared clips, per set. Only the sets this script can open.
IR_MEMBER_HINT = {"halmstad_drone_detection": "/Video_IR/"}


def _archive(name: str) -> pathlib.Path:
    directory = target_dir(name)
    archives = sorted(directory.glob("*.zip"))
    if not archives:
        raise SystemExit(
            f"no archive under {directory}. Fetch it first:\n"
            f"    python scripts/fetch_validation_data.py --set {name}"
        )
    return archives[0]


def _measure_set(name: str, limit: int | None) -> tuple[list[ClipMeasurement], dict[str, Any]]:
    archive = _archive(name)
    hint = IR_MEMBER_HINT[name]
    # The set's own declaration decides which analysers run. Read from the index rather than
    # passed in, so the report cannot be generated with a path its own data does not have.
    signal_path = load_manifest().datasets[name].signal_path
    measurements: list[ClipMeasurement] = []
    failures: list[dict[str, str]] = []
    with zipfile.ZipFile(archive) as zf:
        members = sorted(n for n in zf.namelist() if hint in n and n.lower().endswith(".mp4"))
        if limit is not None:
            members = members[:limit]
        with tempfile.TemporaryDirectory() as tmp:
            scratch = pathlib.Path(tmp) / "clip.mp4"
            for index, member in enumerate(members, 1):
                short = member.rsplit("/", 1)[-1]
                scratch.write_bytes(zf.read(member))
                try:
                    info = probe_clip(scratch)
                    frames = decode_gray8(scratch)
                    measurements.append(
                        measure_clip(
                            frames,
                            name=short,
                            fps=info.fps,
                            range_ambiguity=range_ambiguity_codes(frames),
                            signal_path=signal_path,
                        )
                    )
                except Exception as error:  # noqa: BLE001 - a bad clip is data, not a crash
                    failures.append({"clip": short, "error": f"{type(error).__name__}: {error}"})
                if index % 25 == 0 or index == len(members):
                    print(f"  {index}/{len(members)} clips", file=sys.stderr)
    provenance = {
        "archive": archive.name,
        "archive_sha256": sha256_of(archive),
        "clips_found": len(members),
        "clips_measured": len(measurements),
        "signal_path": signal_path,
        "failures": failures,
    }
    return measurements, provenance


def _write_cache(
    path: pathlib.Path, name: str, measurements: list[ClipMeasurement], provenance: dict[str, Any]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "dataset": name,
                "provenance": provenance,
                "clips": [
                    {
                        "name": m.name,
                        "frames": m.frames,
                        "fps": m.fps,
                        "static": m.static,
                        "max_displacement_px": m.max_displacement_px,
                        "range_ambiguity_codes": m.range_ambiguity_codes,
                        "flat_regions": m.flat_regions,
                        "signal_path": m.signal_path,
                        "skipped": list(m.skipped),
                        "values": m.values,
                    }
                    for m in measurements
                ],
            },
            indent=1,
            sort_keys=True,
        ),
        "utf-8",
    )


def _load_cache(
    path: pathlib.Path, limit: int | None
) -> tuple[list[ClipMeasurement] | None, dict[str, Any] | None]:
    if not path.is_file():
        return None, None
    payload = json.loads(path.read_text("utf-8"))
    clips = payload["clips"][:limit] if limit is not None else payload["clips"]
    return [
        ClipMeasurement(
            name=c["name"],
            frames=c["frames"],
            fps=c["fps"],
            static=c["static"],
            max_displacement_px=c["max_displacement_px"],
            range_ambiguity_codes=c["range_ambiguity_codes"],
            flat_regions=c["flat_regions"],
            signal_path=c.get("signal_path", "unknown"),
            skipped=tuple(c["skipped"]),
            values=c["values"],
        )
        for c in clips
    ], payload["provenance"]


def _values(measurements: list[ClipMeasurement], key: str) -> np.ndarray:
    """Finite per-clip values for one key.

    Non-finite entries are dropped here rather than allowed to reach `summarise`, which refuses
    them -- an analyser can legitimately return a NaN (a pattern-change ratio whose denominator is
    a context with no variation, for instance), and one such clip should cost that clip, not the
    whole report. :func:`_note_dropped` puts the count in the table so the drop is visible.
    """
    raw = np.array([m.values[key] for m in measurements if key in m.values], dtype=np.float64)
    return raw[np.isfinite(raw)]


def _note_dropped(measurements: list[ClipMeasurement], key: str, note: str) -> str:
    raw = np.array([m.values[key] for m in measurements if key in m.values], dtype=np.float64)
    dropped = int((~np.isfinite(raw)).sum())
    if not dropped:
        return note
    tail = f"{dropped} clip{'s' if dropped > 1 else ''} dropped as non-finite"
    return f"{note}; {tail}" if note else tail


def _sections(measurements: list[ClipMeasurement], dataset: Any) -> list[Section]:
    n_total = len(measurements)
    usable = [m for m in measurements if not m.skipped]
    collapsed = sum("noise_floor_collapsed" in m.skipped for m in measurements)
    moving = sum(not m.static for m in measurements)
    no_window = sum("no_flat_window" in m.skipped for m in measurements)
    floor_codes = float(np.median(_values(usable, "codec_floor_codes"))) if usable else 0.289

    signal = Section(
        "The signal path, measured rather than assumed",
        "Two properties of these files bound every number below, and neither is in the index.",
    )
    signal.statistics.append(
        summarise(
            "stored frame rate",
            "Hz",
            [m.fps for m in measurements],
            note="the core runs at 60 Hz; the files are stored at this rate, and a one-pole "
            "temporal fit run at the index's rate is wrong by the ratio while looking plausible",
        )
    )
    signal.statistics.append(
        summarise(
            "colour-range ambiguity",
            "DN8 codes",
            [m.range_ambiguity_codes for m in measurements],
            note="`color_range` is unflagged on every clip, so limited- and full-range readings "
            "of the same file differ by this much. Compare it with the noise scale below: it is "
            "several times larger, which is why only scale-free statistics are reported",
        )
    )
    signal.statistics.append(
        summarise(
            "8x8 blockiness",
            "z",
            _values(measurements, "blockiness_z"),
            note="per-gap null (ADR 0023 addendum); a positive z is the codec's block grid",
        )
    )

    noise = Section(
        "Noise -- and the reason most of it is not there",
        "The headline is negative and it is the most important result in this report.",
    )
    noise.statistics.append(
        summarise(
            "robust noise scale",
            "DN8 codes",
            _values(measurements, "noise_scale_codes"),
            floor=1.0,
            floor_label="one code",
            margin=1.0,
            note="MAD of first differences about their own median, so a gradient cancels exactly. "
            "This is the ruler every other threshold is written in units of",
        )
    )
    if usable:
        for component, label in [
            ("tvh", "sigma_TVH (temporal pixel)"),
            ("vh", "sigma_VH (fixed pattern)"),
            ("t", "sigma_T (frame-to-frame)"),
            ("h", "sigma_H (column)"),
            ("v", "sigma_V (row)"),
        ]:
            values = _values(usable, f"sigma_{component}_codes")
            if values.size:
                fraction = float(np.mean(_values(usable, f"sigma_{component}_measurable")))
                noise.statistics.append(
                    summarise(
                        label,
                        "DN8 codes",
                        values,
                        floor=floor_codes,
                        floor_label="quantiser",
                        note=f"above the floor on {fraction * 100:.0f} % of the usable clips",
                    )
                )
        noise.statistics.append(
            summarise(
                "measurable components",
                "of 7",
                _values(usable, "measurable_components"),
                note="how many of the NVESD seven clear their own floor by ADR 0023's margin",
            )
        )
    noise.refusals.append(
        refusal(
            "every per-pixel noise statistic, on most of the set",
            f"Of {n_total} clips, {len(usable)} support the noise table above. The exclusions "
            "overlap, so these counts do not partition the set: "
            f"**{moving}** are moving, and ME.1b's gate correctly keeps every per-pixel temporal "
            "statistic off a clip whose scene is sliding across the pixels; "
            f"**{collapsed}** have a robust noise scale at or below *one code*, which means the "
            "codec has removed the sensor's noise entirely -- no window can meet a threshold "
            "written in units of a ruler that has collapsed onto the quantiser, and the "
            "flat-region finder refusing is the correct answer rather than a tuning problem; "
            f"**{no_window}** have a usable ruler but no window flat enough against it. "
            "Every value in the table is an upper bound on what the codec left, not a "
            "measurement of the camera.",
            n_refused=n_total - len(usable),
        )
    )

    structure = Section(
        "Spatial and temporal structure",
        "Scale-free by construction, so the missing range flag does not touch them.",
    )
    for key, label, note in [
        ("psd_line_fraction_kv0", "PSD energy on k_v = 0", "the column-noise line"),
        ("psd_line_fraction_kh0", "PSD energy on k_h = 0", "the row-noise line"),
        (
            "temporal_tau_ms",
            "one-pole time constant",
            "fitted on the flat window, DC and Nyquist dropped",
        ),
        (
            "temporal_drift_fraction",
            "drift fraction",
            "how much of the fit is a trend rather than a pole; above ~0.5 the tau is "
            "not trustworthy",
        ),
    ]:
        values = _values(usable, key)
        if values.size:
            structure.statistics.append(
                summarise(label, "fraction" if "fraction" in key else "ms", values, note=note)
            )

    shutter = Section(
        "The shutter signature",
        "What the FFC leaves in a ten-second clip.",
    )
    freezes = _values(measurements, "freeze_count")
    if freezes.size:
        shutter.statistics.append(
            summarise("freeze events per clip", "count", freezes, note="runs of still frames")
        )
    lengths = _values(measurements, "freeze_length_frames")
    if lengths.size:
        shutter.statistics.append(
            summarise("freeze length", "frames", lengths, note="mean over the events in a clip")
        )
        changes = _values(measurements, "freeze_pattern_change")
        if changes.size:
            shutter.statistics.append(
                summarise(
                    "pattern change across a freeze",
                    "ratio",
                    changes,
                    note=_note_dropped(
                        measurements,
                        "freeze_pattern_change",
                        "below ~2.5 the freeze is a stall, above ~5 a shutter (ME.3a)",
                    ),
                )
            )
    shutter.refusals.append(
        refusal(
            "FFC interval distribution",
            "The clips are ten seconds long and `freeze_intervals_s` refuses anything under 180 s "
            "-- the Boson's own schedule -- so this set gives freeze *length* and can never give "
            "an interval. That is a property of the publication, not of the analyser.",
        )
    )
    shutter.refusals.append(
        refusal(
            "between-FFC pattern growth",
            "`fit_pattern_growth` needs the resets to fit sigma^2(t) between them. A ten-second "
            "clip at a 180 s schedule contains at most one, so there is no growth curve in it.",
        )
    )

    targets = Section(
        "Target statistics",
        "This is where the reference set stops being usable, and the reason is worth recording.",
    )
    targets.refusals.append(
        refusal(
            "target size, SCR, contrast polarity, edge spread and smear",
            "All five need the annotation boxes. The labels ship as MATLAB Video Labeler "
            "`groundTruth` objects -- MCOS class instances inside `__function_workspace__` -- "
            "which `scipy.io.loadmat` returns as an opaque blob and no Python reader decodes. "
            "The index said a Python decoder ships with the dataset; **it does not** (the "
            "repository ships a MATLAB script, `Create_a_dataset_from_videos_and_labels.m`), and "
            "the index has been corrected. Decoding MCOS is possible and is not worth its own "
            "failure mode here: until someone runs the MATLAB script and exports boxes, every "
            "box-dependent Tier 4 comparison is open.",
            n_refused=n_total,
        )
    )
    targets.refusals.append(
        refusal(
            "sky elevation profile and cloud clutter slope",
            "Both need a labelled sky region and, for the profile, a visible horizon. Selecting "
            "either automatically would be inventing an annotation and calling it data. The "
            "analysers are implemented and tested (ME.4); what is missing is a region list.",
        )
    )
    targets.refusals.append(
        refusal(
            "size and SCR versus range",
            "Needs LRDDv3's range labels, which sit behind an access request citing export "
            "control. Unavailable, not unimplemented.",
        )
    )
    return [signal, noise, structure, shutter, targets]


def _render(name: str, dataset: Any, sections: list[Section], provenance: dict[str, Any]) -> str:
    today = dt.date.today().isoformat()
    head = [
        f"# Reference statistics -- `{name}` ({today})",
        "",
        "Generated by `scripts/reference_statistics.py`. Do not edit by hand.",
        "",
        "These are the **bands the simulator has to land in** for a Tier 4 comparison (§15 T4, "
        "ADR 0068), measured on public imagery in the display domain, because that is the only "
        "domain this data exists in.",
        "",
        "**Read every value as an upper bound until its floor says otherwise.** A value marked "
        "`≤ … ⚠` is within two floors of the quantiser or the codec, which means the number "
        "describes what the compression left rather than what the camera did.",
        "",
        "| provenance | |",
        "|---|---|",
        f"| dataset | {dataset.title} |",
        f"| licence | {dataset.licence} |",
        f"| sensor | {dataset.sensor} |",
        f"| archive | `{provenance['archive']}` |",
        f"| archive SHA-256 | `{provenance['archive_sha256']}` |",
        "| clips found / measured | "
        f"{provenance['clips_found']} / {provenance['clips_measured']} |",
        f"| bootstrap | {BOOTSTRAP_RESAMPLES} resamples, seeded |",
        f"| report schema | {REPORT_SCHEMA_VERSION} |",
        "",
    ]
    if provenance["failures"]:
        head += [f"{len(provenance['failures'])} clips failed to decode; see the JSON.", ""]
    return "\n".join(head + [s.as_markdown() for s in sections])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set", dest="name", default="halmstad_drone_detection")
    parser.add_argument("--limit", type=int, default=None, help="measure only the first N clips")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--list", action="store_true", help="print the sets this script can open")
    parser.add_argument(
        "--cache",
        default=None,
        help="path to a measurement cache (default: <out>/.<set>-clips.json). Measuring 365 "
        "clips takes half an hour, so the per-clip pass is written here before anything is "
        "aggregated and reused on the next run unless --remeasure is given.",
    )
    parser.add_argument("--remeasure", action="store_true", help="ignore any measurement cache")
    args = parser.parse_args(argv)

    manifest = load_manifest()
    if args.list:
        for name in manifest.datasets:
            ready = "yes" if name in IR_MEMBER_HINT else "no reader"
            print(f"{name:28s} {ready}")
        return 0
    if args.name not in IR_MEMBER_HINT:
        raise SystemExit(f"no archive reader for {args.name!r}; known: {sorted(IR_MEMBER_HINT)}")
    if not ffmpeg_available():
        raise SystemExit("ffmpeg and ffprobe must be on PATH to decode the clips")

    dataset = manifest.datasets[args.name]
    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = pathlib.Path(args.cache) if args.cache else out_dir / f".{args.name}-clips.json"
    measurements, provenance = (
        _load_cache(cache, args.limit) if not args.remeasure else (None, None)
    )
    if measurements is None:
        print(f"measuring {args.name}...", file=sys.stderr)
        measurements, provenance = _measure_set(args.name, args.limit)
        # Written *before* aggregating: the measurement pass is the expensive half and a bug in
        # the arithmetic below should cost a second of re-aggregation, not another half hour of
        # decoding. This is a cache of somebody else's data, so it is gitignored with the data.
        _write_cache(cache, args.name, measurements, provenance)
    else:
        print(f"reusing {cache} ({len(measurements)} clips); --remeasure to redo", file=sys.stderr)
    if not measurements:
        raise SystemExit("no clip could be measured; nothing to report")
    sections = _sections(measurements, dataset)

    stem = f"reference-stats-{dt.date.today().isoformat()}"
    (out_dir / f"{stem}.md").write_text(_render(args.name, dataset, sections, provenance), "utf-8")
    payload = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "dataset": args.name,
        "generated_utc": dt.datetime.now(dt.timezone.utc).date().isoformat(),
        "provenance": provenance,
        "sections": [s.as_dict() for s in sections],
        "clips": [
            {
                "name": m.name,
                "frames": m.frames,
                "fps": m.fps,
                "static": m.static,
                "skipped": list(m.skipped),
                "values": m.values,
            }
            for m in measurements
        ],
    }
    (out_dir / f"{stem}.json").write_text(json.dumps(payload, indent=2, sort_keys=True), "utf-8")
    print(f"wrote {out_dir / stem}.md and .json", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
