#!/usr/bin/env python3
"""Probe (HU.10): does a skinned walking body reach the AOVs as it moves?

    IRSIM_GPU=0 python.sh scripts/probe_isaac_skel.py \\
        --usd outputs/humans/man_adult/walk/man_adult_walk.usdc

The walk and its answer come from ``scripts/animate_human.py`` (Blender). This renders chosen frames
in Kit through the production :class:`~irsim_isaac.pipeline.gbuffer_isaac.AovReader` and holds
each segment's pixels to Blender's centroids (:mod:`irsim_isaac.skel_probe`). It writes
``<usd>.probe.json`` and exits 0 only if every criterion of roadmap HU.10 holds.

docs/physics-model.md §13.3; roadmap HU.10; ADR 0199.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--usd", type=pathlib.Path, required=True)
    ap.add_argument(
        "--frames",
        default="1,9,17,25,33,41,49,57,65",
        help="USD time codes to render (Blender's frames, from 1), or 'all'",
    )
    ap.add_argument(
        "--scene",
        type=pathlib.Path,
        default=None,
        help="a scene whose `solver: human` target is this body: paint each frame's segments with "
        "its JOS-3 skin temperatures (white-hot) and encode a video beside the USD",
    )
    a = ap.parse_args(argv)
    expected_path = a.usd.with_suffix(".expected.json")
    expected = json.loads(expected_path.read_text())
    frames = (
        list(range(1, int(expected["frames"]) + 1))
        if a.frames == "all"
        else [int(f) for f in a.frames.split(",")]
    )

    from isaacsim import SimulationApp

    from irsim_isaac.env import simulation_app_config

    app = SimulationApp(simulation_app_config())
    status = 1
    try:
        import numpy as np

        from irsim_isaac.skel_probe import run_probe

        npz = np.load(a.usd.with_suffix(".vertices.npz"))
        vertices = {name: npz[name] for name in npz.files}
        report = run_probe(str(a.usd.resolve()), expected, frames=frames, vertices=vertices)
        print(report.render())
        out = a.usd.with_suffix(".probe.json")
        out.write_text(
            json.dumps(
                {
                    "frames": report.frames,
                    "ids_stable": report.ids_stable,
                    "id_changes": report.id_changes,
                    "dtypes": report.dtypes,
                    "worst_absolute_m": report.worst_absolute_m,
                    "worst_displacement_m": report.worst_displacement_m,
                    "worst_surface_p95_m": report.worst_surface_p95_m,
                    "moved_segments": report.moved_segments,
                    "control_failures": report.control_failures,
                    "hand_travel_m": report.hand_travel_m,
                    "per_segment": report.per_segment,
                    "passed": report.passed,
                },
                indent=1,
            )
            + "\n"
        )
        status = 0 if report.passed else 1
        if a.scene is not None:
            _paint(report, a.scene, a.usd.parent / "temperature")
    finally:
        app.close(exit_code=status)
    return status


def _paint(report, scene_path: pathlib.Path, out: pathlib.Path) -> None:  # type: ignore[no-untyped-def]
    """Each frame's segments in their JOS-3 skin temperature: the per-prim lookup, made visible.

    This is the temperature plane the IR pipeline keys by prim (ADR 0014), not a camera image: no
    radiance, no optics, no noise. White-hot between the coldest and warmest segment.
    """
    import numpy as np
    from PIL import Image

    from irsim.config.humans import JOS3_SEGMENTS
    from irsim.scene import Scene
    from irsim_eval.video import encode_mp4, ffmpeg_available

    scene = Scene.from_file(scene_path)
    temps = scene.advance_targets(0.0, 30.0)
    (target,) = [t for t in scene.humans]
    seg_k = np.array([temps[f"{target}.skin_{s}"] for s in JOS3_SEGMENTS], dtype=np.float32)
    lo, hi = float(seg_k.min()), float(seg_k.max())
    out.mkdir(parents=True, exist_ok=True)
    for r in report.readings:
        idx = r.segment_index
        plane = np.zeros(idx.shape, dtype=np.float32)
        body = idx >= 0
        plane[body] = (seg_k[idx[body]] - lo) / max(hi - lo, 1e-6) * 0.85 + 0.15
        Image.fromarray((plane * 255).astype(np.uint8)).save(out / f"frame_{r.frame:03d}.png")
    print(f"painted {len(report.readings)} frames, skin {lo - 273.15:.1f}-{hi - 273.15:.1f} C")
    if ffmpeg_available() and len(report.readings) > 1:
        path = out.parent / "walk_temperature.mp4"
        encode_mp4(str(out / "frame_*.png"), path, fps=30)
        print(f"wrote {path}")


if __name__ == "__main__":
    sys.exit(main())
