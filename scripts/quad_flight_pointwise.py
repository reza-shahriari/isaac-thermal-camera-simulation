#!/usr/bin/env python
"""The aerial scenario, regenerated point-wise and engine-free (PT.9).

Loads `configs/scenes/quad_flight_pointwise.yaml`, flies the 30-minute mission, prints the deck,
the belly and the arms at each phase of it, and writes the quadrotor in plan as a **frame** -- a
synthetic G-buffer over each patch's own plane (instance ids and world positions, the same planes
the Isaac path supplies) passed through `PointwiseTemperature.apply`, so the pixels are exactly
what a render would carry before the radiometry. Grayscale, white-hot, one PNG and one float32
.npy per phase. The rendered frame is IG.2's.

Usage: python scripts/quad_flight_pointwise.py [--out DIR] [--at SECONDS ...]
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

SCENE = REPO / "configs" / "scenes" / "quad_flight_pointwise.yaml"
PHASES = ((0.0, "on the pad"), (400.0, "climbing"), (1000.0, "hard climb"), (1750.0, "landed"))
SURFACES = ("deck", "belly", "arm_n", "arm_e")


def plan_gbuffer(px_per_m: int = 400) -> tuple[np.ndarray, np.ndarray, dict[str, str]]:
    """A camera straight above the quadrotor: the deck and the two arms it can see."""
    half = 0.5
    n = int(2 * half * px_per_m)
    xs = -half + (np.arange(n) + 0.5) / px_per_m
    ys = half - (np.arange(n) + 0.5) / px_per_m  # the top row is north
    xx, yy = np.meshgrid(xs, ys)
    ids = np.zeros((n, n), dtype=np.int32)
    zz = np.zeros_like(xx)
    deck = (np.abs(xx) <= 0.15) & (np.abs(yy) <= 0.15)
    ids[deck], zz[deck] = 1, 0.36
    arm_n = (np.abs(xx) <= 0.025) & (yy > 0.15) & (yy <= 0.45)
    ids[arm_n], zz[arm_n] = 2, 0.32
    arm_e = (np.abs(yy) <= 0.025) & (xx > 0.15) & (xx <= 0.45)
    ids[arm_e], zz[arm_e] = 3, 0.32
    positions = np.stack([xx, yy, zz], axis=-1)
    labels = {"1": "/World/Quad/deck", "2": "/World/Quad/arm_n", "3": "/World/Quad/arm_e"}
    return ids, positions, labels


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=pathlib.Path, default=REPO / "out" / "quad_flight_pointwise")
    parser.add_argument("--at", type=float, nargs="*", help="seconds after t0 to report and draw")
    args = parser.parse_args()

    scene = Scene.from_file(SCENE)
    phases = [(t, f"t+{t:.0f}s") for t in args.at] if args.at else list(PHASES)
    sun = scene.solar_terms(scene.t0_s)
    print(f"{SCENE.name}: sun elevation {sun.elevation_deg:.1f} deg, DNI {sun.dni_w_m2:.0f} W/m2")
    header = "  " + "".join(f"{n:>22s}" for n in SURFACES)
    print(f"  {'phase':<14s}{'speed':>7s}{'air':>7s}" + header)

    ids, positions, labels = plan_gbuffer()
    bridge = PointwiseTemperature(
        bindings_from_scene(scene), known_paths=set(scene.patch_prims.values())
    )
    args.out.mkdir(parents=True, exist_ok=True)
    for rel, label in phases:
        t = scene.t0_s + rel
        air = float(scene.weather.at(t).t_air_k)
        speed = scene.thermal.forcing_at.orientations[0].speed_at(t)
        cells = []
        for name in SURFACES:
            field = scene.surface_fields[name]
            field.advance_to(t)
            v = np.asarray(field.temperature_at(t), dtype=np.float64) - 273.15
            lit = field.field.forcing_at.cell_visibility(t)
            cells.append(f"{v.mean():7.1f} ({v.max() - v.min():4.1f} K, lit {lit.mean():4.2f})")
        print(f"  {label:<14s}{speed:7.1f}{air - 273.15:7.1f}  " + "".join(cells))

        plane = np.full(ids.shape, air, dtype=np.float32)
        frame = bridge.apply(plane, ids, labels, positions, t, strict=True)
        stem = f"plan_t{int(rel):04d}"
        np.save(args.out / f"{stem}_k.npy", frame)
        lo, hi = float(frame.min()), float(frame.max())
        gray = (np.clip((frame - lo) / max(hi - lo, 1e-6), 0.0, 1.0) * 255.0).astype(np.uint8)
        try:
            from PIL import Image

            Image.fromarray(gray, mode="L").save(args.out / f"{stem}_whitehot.png")
            print(f"      frame {args.out / (stem + '_whitehot.png')} ({lo:.1f}..{hi:.1f} K)")
        except ImportError:  # pragma: no cover - PIL is the eval stack's, not the core's
            print(f"      frame {args.out / (stem + '_k.npy')} (PIL not installed; no PNG)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
