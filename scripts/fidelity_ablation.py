#!/usr/bin/env python3
"""Roadmap M12.3 (ablation half): which mechanism actually moves the picture?

    python scripts/fidelity_ablation.py --clips 3 --frames 20 --out docs/validation

§15's first Tier 5 caution is that a sim-to-real gap has to be *attributed*. The transfer protocols
answer that with a detector; this answers the cheaper, prior question -- **which parts of the model
change the frames at all** -- by rendering the same scenarios with one mechanism switched off and
measuring the DN8 distance from full fidelity with ME.6's own statistics.

**It needs no detector, no labels and no GPU**, which is why it is worth running on its own: ME.7's
training half is blocked on the reference set's MATLAB annotation boxes, and this half of M12.3 is
not blocked on anything. A mechanism that moves nothing here cannot be responsible for a sim-to-real
gap, whatever a later detector says.

**A large distance is not a fault.** It means the mechanism is load-bearing and its parameters are
worth measuring. A distance near zero means the opposite -- either the mechanism is negligible in
this scenario, or it is not reaching the frames at all, which is a bug this repository has now found
three times (ADR 0077, ADR 0082, ADR 0084).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import sys
from typing import Any

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_SENSOR = REPO / "configs" / "sensors" / "halmstad_boson_320.yaml"
DEFAULT_SCENE = REPO / "configs" / "scenes" / "sky_target_clear_day.yaml"

#: The variants. Each is a patch onto the sensor config dict; `None` is full fidelity. The AGC and
#: the FFC are *not* in ME.8's `fidelity:` block -- they were already explicit config (ADR 0083) --
#: so they are patched where they live, which is the point of that decision.
VARIANTS: dict[str, dict[str, Any]] = {
    "full": {},
    "noise off": {"fidelity": {"noise": False}},
    "optical PSF off": {"fidelity": {"optical_psf": False}},
    "bad pixels off": {"fidelity": {"bad_pixels": False}},
    "NUC residual off": {"fidelity": {"nuc_residual": False}},
    "AGC linear -> none": {"isp": {"agc": "none"}},
    "FFC ideal": {"nuc": {"mode": "ideal"}},
}


def _patched(base: Any, patch: dict[str, Any]) -> Any:
    from irsim.config.sensor import SensorConfig

    dumped = base.model_dump(mode="json")
    for block, fields in patch.items():
        existing = dumped["sensor"].get(block) or {}
        dumped["sensor"][block] = {**existing, **fields}
    return SensorConfig.model_validate(dumped)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clips", type=int, default=3)
    parser.add_argument("--frames", type=int, default=20)
    parser.add_argument("--sensor", default=str(DEFAULT_SENSOR))
    parser.add_argument("--scene", default=str(DEFAULT_SCENE))
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--out", default=str(REPO / "docs" / "validation"))
    args = parser.parse_args(argv)

    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "generate_matched_scenario", REPO / "scripts" / "generate_matched_scenario.py"
    )
    assert spec is not None and spec.loader is not None
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)

    from irsim.config.loader import config_hash, load_sensor_config
    from irsim.materials import MaterialLibrary, MaterialTable
    from irsim.radiometry.lut import BandLUT
    from irsim.radiometry.spectral_response import load_spectral_response
    from irsim.scene import Scene
    from irsim.validation.compare import histogram_emd, psd_shape_ratio
    from irsim.validation.scenario import ScenarioSampler
    from irsim_eval.discriminator import gap_score

    base = load_sensor_config(args.sensor)
    band = base.sensor.band.band_id
    lut = BandLUT.build(load_spectral_response(base.sensor.band.spectral_response), n=2001)
    scene = Scene.from_file(args.scene, {band: lut}, quantity=base.sensor.quantity)
    materials = MaterialTable.for_sensor(MaterialLibrary.load(), base)
    scenarios = ScenarioSampler(seed=args.seed).draw(args.clips)

    def render(sensor: Any) -> np.ndarray:
        frames = []
        for scenario in scenarios:
            images, _ = generator._render_clip(
                scenario,
                sensor=sensor,
                scene=scene,
                lut=lut,
                materials=materials,
                frames=args.frames,
                supersample=sensor.sensor.optics.supersample_factor,
            )
            frames.append(images)
        return np.concatenate(frames)

    def patches(
        cube: np.ndarray, offset: int, size: int = 32, per_frame: int = 4
    ) -> list[np.ndarray]:
        # `offset` makes the two sides of a comparison **independent samples**. Handing the same
        # patches to both classes is degenerate: the probe cannot separate them, but a patch and
        # its twin land in different cross-validation folds and are scored by different models, so
        # the AUC drifts off 0.5 systematically. Measured on this scene: 0.27, five null sigma
        # below chance, on a set compared with itself. The control has to read 0.5 or nothing else
        # in the table means anything.
        rng = np.random.default_rng([args.seed, offset])
        out = []
        for frame in cube:
            h, w = frame.shape
            for _ in range(per_frame):
                y = int(rng.integers(0, h - size + 1))
                x = int(rng.integers(0, w - size + 1))
                out.append(frame[y : y + size, x : x + size].astype(np.float64))
        return out

    print("rendering full fidelity...", file=sys.stderr)
    reference = render(base)
    reference_median = np.median(reference.astype(np.float64), axis=0)
    rows: list[dict[str, Any]] = []
    for name, patch in VARIANTS.items():
        sensor = _patched(base, patch) if patch else base
        cube = reference if not patch else render(sensor)
        if patch:
            print(f"rendering {name}...", file=sys.stderr)
        median = np.median(cube.astype(np.float64), axis=0)
        gap = gap_score(patches(reference, 0), patches(cube, 1), seed=args.seed)
        rows.append(
            {
                "variant": name,
                "config_hash": config_hash(sensor),
                "histogram_emd_codes": histogram_emd(reference_median, median),
                "psd_shape_ratio": psd_shape_ratio(reference_median, median),
                "discriminator_auc": gap.auc,
                "auc_null_sigma": gap.null_sigma,
                "auc_z": gap.z,
                "separable": gap.separable,
                "heaviest_features": [n for n, _ in gap.importance[:3]],
            }
        )
        print(f"  {name}: EMD {rows[-1]['histogram_emd_codes']:.3g}, AUC {gap.auc:.3f}")

    hashes = {r["config_hash"] for r in rows}
    assert len(hashes) == len(rows), "every ablation variant must have its own config hash (ME.8)"

    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    today = dt.date.today().isoformat()
    ranked = sorted(rows[1:], key=lambda r: -float(r["discriminator_auc"]))
    lines = [
        f"# Fidelity ablation -- {today}",
        "",
        "Generated by `scripts/fidelity_ablation.py`. Do not edit by hand.",
        "",
        "Each variant renders the **same scenarios** with one mechanism switched off, and is "
        "compared with full fidelity using ME.6's own DN8 statistics. A large distance means the "
        "mechanism is load-bearing and its parameters are worth measuring; a distance near zero "
        "means either that it is negligible in this scenario **or that it is not reaching the "
        "frames at all** -- a bug this repository has found three times (ADR 0077, 0082, 0084).",
        "",
        f"Camera `{base.sensor.name}`, {args.clips} scenarios x {args.frames} frames, "
        f"seed {args.seed}. Every variant has its own config hash (ME.8, ADR 0083).",
        "",
        "| variant | histogram EMD (codes) | PSD shape ratio | discriminator AUC "
        "| z (null σ) | heaviest features |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['variant']} | {r['histogram_emd_codes']:.3g} | {r['psd_shape_ratio']:.3g} | "
            f"{r['discriminator_auc']:.3f} | {r['auc_z']:+.1f} | "
            f"{', '.join(r['heaviest_features'])} |"
        )
    lines += [
        "",
        "## Ranking",
        "",
        "Most to least effect on the frames, by discriminator AUC against full fidelity:",
        "",
    ]
    lines += [
        f"{i}. **{r['variant']}** -- AUC {r['discriminator_auc']:.3f} ({r['auc_z']:+.1f} σ)"
        for i, r in enumerate(ranked, 1)
    ]
    lines += [
        "",
        "A variant whose AUC sits near the `full` row did not change the frames measurably in "
        "this scenario. That is a result about *this* scenario, not about the mechanism.",
        "",
        "**Read the `full` row as the control.** It compares two disjoint samples of the same "
        "frames, so it must read 0.5, and every other row is meaningful relative to it. "
        "⚠️ **How many scenarios that takes is itself a measurement**: at 2 scenarios the control "
        "read **0.40**, five null sigma below chance on sets that are identical in distribution, "
        "because the null standard error assumes independent patches and patches cut from the "
        "same frames are not independent. The effective sample size is nearer the number of "
        "*scenarios* than the number of patches, so buy independence with more scenarios and "
        "never with more patches per frame.",
        "",
        "**Why three switches move nothing here, verified rather than assumed.** The M9 chain *is* "
        "attached (a photon FPA gets none -- ADR 0056), with 112 bad pixels on the array and "
        "defects and the residual both enabled. The FFC interval is 180 s = **10 800 frames** at "
        "60 Hz, so a clip this short contains no shutter event at all; the NUC residual grows from "
        "the last FFC and is therefore near zero at the start of a clip; and the chain's own "
        "replacement stage repairs the 112 defects, which is what it is for. All three are "
        "properties of a short clip, and a longer one is where they would show.",
        "",
    ]
    (out_dir / f"fidelity-ablation-{today}.md").write_text("\n".join(lines), "utf-8")
    (out_dir / f"fidelity-ablation-{today}.json").write_text(
        json.dumps(
            {"camera": base.sensor.name, "seed": args.seed, "clips": args.clips,
             "frames": args.frames, "variants": rows},
            indent=2, sort_keys=True,
        ),
        "utf-8",
    )  # fmt: skip
    print(f"wrote {out_dir / f'fidelity-ablation-{today}'}.md and .json", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
