#!/usr/bin/env python
"""The cabin reference scene, engine-free: a parked car and the air behind its panels (PT.15).

Loads `configs/scenes/parked_car_cabin.yaml`, prints every panel, the cabin air and the
adiabatic-backed bonnet that is the control, follows them from noon into the night, and writes
the car in plan as a **frame** -- a synthetic G-buffer over the roof's plane (instance ids and world
positions, the same planes the Isaac path supplies) passed through `PointwiseTemperature.apply`,
so the pixels are exactly what a render would carry before the radiometry. Grayscale, white-hot,
one PNG and one float32 .npy. The rendered frame is IG.2's.

Usage: python scripts/parked_car_cabin.py [--out DIR] [--hours N]
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

SCENE = REPO / "configs" / "scenes" / "parked_car_cabin.yaml"
PANELS = ("roof", "bonnet", "door_east", "door_west", "glazing")


def plan_gbuffer(px_per_m: int = 60) -> tuple[np.ndarray, np.ndarray, dict[str, str]]:
    """A camera straight above the car: the roof, the bonnet and the windscreen in one frame.

    The three near-horizontal panels are the ones a plan view sees; the doors are vertical and
    do not appear. Each pixel carries the world position **on that panel's own plane**, which is
    what the position AOV supplies in the engine, so the bridge samples the right cell -- for the
    windscreen that means following its 30 deg slope, not the ground.
    """
    x0, x1, y0, y1 = -1.0, 1.0, -2.0, 2.2
    w, h = int((x1 - x0) * px_per_m), int((y1 - y0) * px_per_m)
    xs = x0 + (np.arange(w) + 0.5) / px_per_m
    ys = y1 - (np.arange(h) + 0.5) / px_per_m  # the top row is the front of the car
    xx, yy = np.meshgrid(xs, ys)
    ids = np.zeros((h, w), dtype=np.int32)
    zz = np.zeros_like(xx)

    roof = (np.abs(xx) <= 0.7) & (np.abs(yy) <= 0.8)
    ids[roof], zz[roof] = 3, 1.45
    bonnet = (np.abs(xx) <= 0.7) & (yy > 0.8) & (yy <= 2.0)
    ids[bonnet], zz[bonnet] = 4, 1.0
    glazing = (np.abs(xx) <= 0.8) & (yy >= -1.9) & (yy < -0.87)
    ids[glazing] = 5
    zz[glazing] = 0.95 + (yy[glazing] + 1.9) * (0.5 / 0.8660254)  # up the 30 deg slope

    positions = np.stack([xx, yy, zz], axis=-1)
    labels = {"3": "/World/Car/roof", "4": "/World/Car/bonnet", "5": "/World/Car/glazing"}
    return ids, positions, labels


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=pathlib.Path, default=REPO / "out" / "parked_car_cabin")
    parser.add_argument("--hours", type=int, default=14, help="hours to follow after t0")
    args = parser.parse_args()

    scene = Scene.from_file(SCENE)
    coupled = scene.surface_fields["cabin"]
    t0 = scene.t0_s
    sun = scene.solar_terms(t0)
    print(f"{SCENE.name}: sun elevation {sun.elevation_deg:.1f} deg, DNI {sun.dni_w_m2:.0f} W/m2")
    # The frame first: a field keeps only its last few ticks (ADR 0093), so t0 has to be
    # read before the run below walks the scene into the night.
    ids, positions, labels = plan_gbuffer()
    bridge = PointwiseTemperature(
        bindings_from_scene(scene), known_paths=set(scene.patch_prims.values())
    )
    # Everything the bridge does not bind keeps the plane's value: the air, which is roughly
    # what the ground around a parked car reads and keeps the cold windscreen visible against it.
    plane = np.full(ids.shape, float(scene.weather.at(t0).t_air_k), dtype=np.float32)
    frame = bridge.apply(plane, ids, labels, positions, t0, strict=True)
    args.out.mkdir(parents=True, exist_ok=True)
    np.save(args.out / "plan_t0_k.npy", frame)
    lo, hi = float(frame.min()), float(frame.max())
    gray = (np.clip((frame - lo) / max(hi - lo, 1e-6), 0.0, 1.0) * 255.0).astype(np.uint8)
    try:
        from PIL import Image

        Image.fromarray(gray, mode="L").save(args.out / "plan_t0_whitehot.png")
        print(f"  frame: {args.out / 'plan_t0_whitehot.png'} ({lo:.1f}..{hi:.1f} K, white-hot)")
    except ImportError:  # pragma: no cover - PIL is the eval stack's, not the core's
        print(f"  frame: {args.out / 'plan_t0_k.npy'} (PIL not installed; no PNG)")
    print("  hour   air   cabin    roof  bonnet(adiabatic)  roof-bonnet  door_e  door_w  glazing")
    for hour in sorted({0, 6, args.hours}):
        t = t0 + 3600.0 * hour
        coupled.advance_to(t)
        scene.surface_fields["bonnet"].advance_to(t)
        air = float(scene.weather.at(t).t_air_k) - 273.15
        means = {
            n: float(np.mean(scene.surface_fields[n].temperature_at(t))) - 273.15 for n in PANELS
        }
        cabin = coupled.node_temperature_k("cabin") - 273.15
        print(
            f"  +{hour:2d} h {air:6.1f} {cabin:7.1f} {means['roof']:7.1f} {means['bonnet']:13.1f} "
            f"{means['roof'] - means['bonnet']:+15.2f} {means['door_east']:7.1f} "
            f"{means['door_west']:7.1f} {means['glazing']:8.1f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
