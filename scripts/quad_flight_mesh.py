#!/usr/bin/env python
"""The quadrotor's arms as tubes, engine-free (WM.7).

Loads `configs/scenes/quad_flight_mesh.yaml`, flies the 30-minute mission, and writes each arm
**unrolled** -- the tube cut along its underside and laid flat, angle across and length down -- so
the gradient around the circumference is a picture rather than a number. The pixels come from the
same `MeshPointBridge` a render uses: a world position per pixel, a closest-point query, a cell.

`quad_flight_pointwise.yaml` models the same arms as flat strips, because ADR 0087's projection is
all a patch can do. That scene reports one temperature across the whole width of an arm; this one
reports the 31 K between its sunlit crown and its shaded underside.

Usage: python scripts/quad_flight_mesh.py [--out DIR] [--frames N]
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from irsim.scene import Scene  # noqa: E402
from irsim_eval.video import annotate, encode_mp4  # noqa: E402
from irsim_isaac.pipeline.mesh_bridge import MeshBinding, MeshPointBridge  # noqa: E402

SCENE = REPO / "configs" / "scenes" / "quad_flight_mesh.yaml"
PHASES = ((0.0, "on the pad"), (400.0, "climbing"), (1000.0, "hard climb"), (1750.0, "landed"))
#: The mission, at the same 6 s cadence the Isaac driver films it at.
INTERVAL_S = 6.0


def unrolled_gbuffer(spec, px_per_m: int = 600, n_phi: int = 180):  # type: ignore[no-untyped-def]
    """A cylinder cut along its underside and laid flat: ``(ids, positions, labels)``.

    The angle runs across the image starting from straight down, so the shaded underside is at
    both edges and the sunlit crown is in the middle. This is the mesh's own surface, sampled --
    not a camera view, which would show one side at a time and hide the thing worth seeing.
    """
    axis = np.asarray(spec.axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    helper = np.eye(3)[int(np.argmin(np.abs(axis)))]
    u = np.cross(axis, helper)
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    # Start the cut at -z (the underside) so the crown lands mid-frame.
    down = np.array([0.0, 0.0, -1.0])
    phi0 = float(np.arctan2(np.dot(down, v), np.dot(down, u)))

    n_z = max(2, int(round(spec.length_m * px_per_m)))
    phi = phi0 + np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)
    z = np.linspace(-0.5 * spec.length_m, 0.5 * spec.length_m, n_z)
    zz, pp = np.meshgrid(z, phi, indexing="ij")
    ring = spec.radius_m * (np.cos(pp)[..., None] * u + np.sin(pp)[..., None] * v)
    positions = np.asarray(spec.centre_m, dtype=np.float64) + zz[..., None] * axis + ring
    ids = np.full(positions.shape[:2], 1, dtype=np.int32)
    return ids, positions, {"1": spec.prim_path}


def white_hot(frame: np.ndarray, span: tuple[float, float]) -> np.ndarray:
    lo, hi = span
    return ((np.clip(frame, lo, hi) - lo) / max(hi - lo, 1e-6) * 255.0).astype(np.uint8)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=pathlib.Path, default=REPO / "outputs" / "quad_flight_mesh")
    parser.add_argument("--frames", type=int, default=300, help="0 writes the phase stills only")
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args()

    scene = Scene.from_file(SCENE)
    specs = {
        s.name: s.mesh for s in scene.spec.thermal.surfaces if getattr(s, "mesh", None) is not None
    }
    sun = scene.solar_terms(scene.t0_s)
    print(f"{SCENE.name}: sun elevation {sun.elevation_deg:.1f} deg, DNI {sun.dni_w_m2:.0f} W/m2")
    inventory = ", ".join(f"{n} ({f.patch.n_cells} cells)" for n, f in scene.mesh_fields.items())
    print(f"  meshes: {inventory}")

    bridge = MeshPointBridge(
        [MeshBinding(path, fld) for path, fld in scene.mesh_bindings()],
        known_paths=list(scene.mesh_prims.values()),
    )
    buffers = {name: unrolled_gbuffer(spec) for name, spec in specs.items()}
    args.out.mkdir(parents=True, exist_ok=True)

    # --- one forward pass over the mission -------------------------------------------------
    #
    # A spatial field holds two ticks (ADR 0093), so it can only answer near where it has been
    # advanced to and time may not run backwards. Everything this script writes therefore comes
    # from a **single** pass in time order -- the table line, the phase stills and the video
    # frame for one instant are produced together, rather than replaying the mission three times.
    from PIL import Image

    span = (296.0, 336.0)
    frames_dir = args.out / "frames"
    if args.frames > 0:
        frames_dir.mkdir(parents=True, exist_ok=True)
        for stale in frames_dir.glob("*.png"):
            stale.unlink()
    video_name = next(iter(buffers))

    header = f"  {'phase':14s}{'air':>7s}{'deck':>8s}{'belly':>8s}"
    for name in specs:
        header += f"{name + ' crown':>14s}{name + ' under':>14s}"
    print(header)

    n_frames = max(args.frames, 1)
    phase_at = {int(round(rel / INTERVAL_S)): label for rel, label in PHASES}
    for index in range(n_frames):
        rel = index * INTERVAL_S
        t = scene.t0_s + rel
        for fld in scene.mesh_fields.values():
            fld.advance_to(t)
        label = phase_at.get(index)

        if label is not None:
            air = float(scene.weather.at(t).t_air_k) - 273.15
            line = f"  {label:14s}{air:7.1f}"
            for surface in ("deck", "belly"):
                planar = scene.surface_fields[surface]
                planar.advance_to(t)
                line += f"{float(np.mean(np.asarray(planar.temperature_at(t)))) - 273.15:8.1f}"
            for fld in scene.mesh_fields.values():
                cells = np.asarray(fld.temperature_at(t), dtype=np.float64)
                up = fld.patch.cell_normal[:, 2]
                line += f"{cells[up > 0.7].mean() - 273.15:14.1f}"
                line += f"{cells[up < -0.7].mean() - 273.15:14.1f}"
            print(line)

        for name, (ids, positions, labels) in buffers.items():
            if label is None and name != video_name:
                continue
            plane = np.full(ids.shape, 300.0, dtype=np.float32)
            frame = bridge.apply(plane, ids, labels, positions, t, strict=True)
            if label is not None:
                stem = f"{name}_unrolled_t{int(rel):04d}"
                np.save(args.out / f"{stem}_k.npy", frame)
                Image.fromarray(white_hot(frame, span), mode="L").save(args.out / f"{stem}.png")
            if name == video_name and args.frames > 0:
                gray = white_hot(frame, span)
                rgb = annotate(
                    np.dstack([gray, gray, gray]),
                    [
                        f"{name} unrolled   T+{int(rel // 60):02d}:{int(rel % 60):02d}",
                        f"crown {float(frame.max()) - 273.15:5.1f} C   "
                        f"underside {float(frame.min()) - 273.15:5.1f} C   "
                        f"span {float(np.ptp(frame)):4.1f} K",
                        f"white-hot, fixed span {span[0] - 273.15:.0f}..{span[1] - 273.15:.0f} C",
                    ],
                    size=14,
                )
                Image.fromarray(rgb).save(frames_dir / f"frame_{index:05d}.png")

    if args.frames > 0:
        video = encode_mp4(
            str(frames_dir / "*.png"), args.out / f"{video_name}_unrolled.mp4", fps=args.fps
        )
        print(f"  video: {video}")
    print(f"\nall output under {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
