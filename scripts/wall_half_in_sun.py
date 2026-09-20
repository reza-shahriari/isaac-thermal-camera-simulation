#!/usr/bin/env python
"""The R1 reference scene, engine-free: one building's faces at their own temperatures (PT.20).

Loads `configs/scenes/wall_half_in_sun.yaml`, prints every surface's mean and lit/shaded step at
the scene start, lifts the neighbour's shadow and follows the once-shaded concrete for half an
hour, and writes the west wall as a **frame** -- a synthetic G-buffer over the wall's plane
(instance ids and world positions, the same planes the Isaac path supplies) passed through
`PointwiseTemperature.apply`, so the pixels are exactly what a render would carry before the
radiometry. Grayscale, white-hot, one PNG and one float32 .npy. The rendered frame is IG.2's.

Usage: python scripts/wall_half_in_sun.py [--out DIR] [--minutes N]
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from irsim.scene import Scene  # noqa: E402
from irsim_isaac.pipeline.point_bridge import (  # noqa: E402
    PointwiseTemperature,
    bindings_from_scene,
)

SCENE = REPO / "configs" / "scenes" / "wall_half_in_sun.yaml"


def west_wall_gbuffer(px_per_m: int = 16) -> tuple[np.ndarray, np.ndarray, dict[str, str]]:
    """A camera square-on to the west wall: 8 m wide (both halves), 6 m high."""
    w, h = 6 * px_per_m, 6 * px_per_m
    ys = (np.arange(w) + 0.5) / px_per_m  # 0..6 m along +y, left to right as seen from the west
    zs = 6.0 - (np.arange(h) + 0.5) / px_per_m  # top row is the top of the wall
    yy, zz = np.meshgrid(ys, zs)
    positions = np.stack([np.zeros_like(yy), yy, zz], axis=-1)
    ids = np.full((h, w), 7, dtype=np.int32)
    return ids, positions, {"7": "/World/Building/west"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=pathlib.Path, default=REPO / "out" / "wall_half_in_sun")
    parser.add_argument("--minutes", type=int, default=30, help="minutes to follow after t0")
    args = parser.parse_args()

    scene = Scene.from_file(SCENE)
    t0 = scene.t0_s
    sun = scene.solar_terms(t0)
    print(f"{SCENE.name}: sun elevation {sun.elevation_deg:.1f} deg, DNI {sun.dni_w_m2:.0f} W/m2")
    for name, fld in scene.surface_fields.items():
        temps = np.asarray(fld.temperature_at(t0), dtype=np.float64)
        lit = fld.field.forcing_at.cell_visibility(t0) == 1.0
        line = f"  {name:14s} mean {temps.mean():7.2f} K  lit {lit.mean():4.2f}"
        if 0.0 < lit.mean() < 1.0:
            line += f"  lit - shaded {temps[lit].mean() - temps[~lit].mean():5.2f} K"
        print(line)

    # The frame: the west wall through the point bridge, from a synthetic G-buffer.
    ids, positions, labels = west_wall_gbuffer()
    bridge = PointwiseTemperature(
        bindings_from_scene(scene), known_paths=set(scene.patch_prims.values())
    )
    plane = np.full(ids.shape, 300.0, dtype=np.float32)
    frame = bridge.apply(plane, ids, labels, positions, t0, strict=True)
    args.out.mkdir(parents=True, exist_ok=True)
    np.save(args.out / "west_wall_t0_k.npy", frame)
    lo, hi = float(frame.min()), float(frame.max())
    gray = ((frame - lo) / max(hi - lo, 1e-6) * 255.0).astype(np.uint8)
    try:
        from PIL import Image

        Image.fromarray(gray, mode="L").save(args.out / "west_wall_t0_whitehot.png")
        print(
            f"  frame: {args.out / 'west_wall_t0_whitehot.png'} ({lo:.1f}..{hi:.1f} K, white-hot)"
        )
    except ImportError:  # pragma: no cover - PIL is the eval stack's, not the core's
        print(f"  frame: {args.out / 'west_wall_t0_k.npy'} (PIL not installed; no PNG)")

    # Lift the neighbour's shadow and follow the once-shaded concrete.
    fld = scene.surface_fields["west_concrete"]
    temps = np.asarray(fld.temperature_at(t0), dtype=np.float64)
    lit = fld.field.forcing_at.cell_visibility(t0) == 1.0
    step = temps[lit].mean() - temps[~lit].mean()
    fld.field.forcing_at.occluders = tuple(
        o for n, o in scene.occluders.items() if not n.startswith("nbr")
    )
    for minute in sorted({m for m in (10, 20, args.minutes) if m <= args.minutes}):
        t = t0 + 60.0 * minute
        fld.advance_to(t)
        now = np.asarray(fld.temperature_at(t), dtype=np.float64)
        print(
            f"  +{minute:2d} min after the shadow lifts: once-shaded concrete still "
            f"{now[lit].mean() - now[~lit].mean():.2f} K cooler (the step was {step:.2f} K)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
