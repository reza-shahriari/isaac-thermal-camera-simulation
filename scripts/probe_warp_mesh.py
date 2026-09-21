#!/usr/bin/env python3
"""Roadmap WM.1: can Warp recover exact per-pixel (face, u, v) on this build?

ADR 0087 chose to parameterise point-wise surface temperature by projecting the per-pixel world
position onto a **planar patch**, and recorded curved geometry as a limit that could only be
raised by a renderer capability this build does not have -- a UV AOV or a per-triangle id AOV.
This script tests the third way: do not ask the renderer for the parameterisation, *derive* it,
by asking Warp for the closest point on the prim's own mesh to the position the position AOV
already carries.

It answers four questions, and the fourth is the one WM.3 actually rides on:

1. Does `mesh_query_point_no_sign` return a usable (face, u, v) on this Warp, and does
   `mesh_eval_position` invert it?
2. **What barycentric convention does it use?** Getting this wrong samples the wrong cell of the
   wrong triangle and looks like plausible noise, so the convention is measured, not assumed.
3. Does it agree with a brute-force NumPy oracle -- the same oracle WM.3 would be tested against?
4. ADR 0014 measured the position AOV as good to **3.4 mm**. Under that much error in the query
   point, how often does the recovered *face* change, and how far does the sampled point move?
   A face that flips between neighbours is harmless if the surface position barely moves; it is a
   defect if it jumps across the mesh.

Never raises on a probe failure -- failures are data (the discipline of `probe_isaac_geometry.py`).
No Kit boot: Warp imports from `python.sh` directly (ADR 0014 addendum), so this runs in seconds.

    python.sh scripts/probe_warp_mesh.py --out outputs/isaac_probe/warp_mesh
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

# CUDA's default order is FASTEST_FIRST, which puts the workstation's 5090 at index 0 and reverses
# `nvidia-smi`'s PCI order -- so `cuda:0` would select the card the owner works on. Set before any
# CUDA library is imported, or it has no effect.
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import numpy as np  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from irsim.thermal.raycast import TriangleSoup, box_mesh  # noqa: E402
from irsim_isaac.env import ensure_warp_on_path  # noqa: E402

#: ADR 0014's measured accuracy of `Camera3dPositionSD` / `PtWorldPos` on this build.
POSITION_BUDGET_M = 3.4e-3


# --- meshes with a known answer ------------------------------------------------------------------


def uv_sphere(radius_m: float, n_theta: int, n_phi: int) -> TriangleSoup:
    """A UV sphere, so that the closest point to an exterior point is analytic on the *sphere*.

    The mesh is a chord approximation of it, short by the sagitta `sagitta_m` computes, which is
    reported beside every sphere residual rather than folded into a tolerance.
    """
    theta = np.linspace(0.0, np.pi, n_theta + 1)
    phi = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)
    tt, pp = np.meshgrid(theta, phi, indexing="ij")
    vertices = radius_m * np.stack(
        [np.sin(tt) * np.cos(pp), np.sin(tt) * np.sin(pp), np.cos(tt)], axis=-1
    ).reshape(-1, 3)
    faces = []
    for i in range(n_theta):
        for j in range(n_phi):
            a = i * n_phi + j
            b = i * n_phi + (j + 1) % n_phi
            c = (i + 1) * n_phi + j
            d = (i + 1) * n_phi + (j + 1) % n_phi
            faces.append([a, c, d])
            faces.append([a, d, b])
    return TriangleSoup(vertices, np.asarray(faces, dtype=np.int64))


def sagitta_m(radius_m: float, n_theta: int, n_phi: int) -> float:
    """How far below the sphere a chord triangle can fall.

    Set by the triangle's **longest** angular extent, which is the diagonal of the
    (dtheta, dphi) cell -- not dtheta alone. Using the polar step by itself understates it by
    about a factor of two on a square-ish sphere and would let a real error hide under it.
    """
    diagonal = float(np.hypot(np.pi / n_theta, 2.0 * np.pi / n_phi))
    return float(radius_m * (1.0 - np.cos(0.5 * diagonal)))


# --- the oracle: brute-force closest point on a triangle soup -------------------------------------


def closest_point_numpy(
    soup: TriangleSoup, points: np.ndarray, chunk: int = 256
) -> tuple[np.ndarray, np.ndarray]:
    """``(closest (P, 3), face (P,))`` by testing every triangle. Ericson's region test.

    Deliberately the dumbest correct implementation: it is the oracle, so it must be obviously
    right rather than fast.
    """
    v = soup.vertices
    a, b, c = v[soup.faces[:, 0]], v[soup.faces[:, 1]], v[soup.faces[:, 2]]
    ab, ac = b - a, c - a
    out = np.zeros_like(points)
    which = np.zeros(len(points), dtype=np.int64)
    for start in range(0, len(points), chunk):
        p = points[start : start + chunk][:, None, :]
        ap, bp, cp = p - a, p - b, p - c
        d1, d2 = (ab * ap).sum(-1), (ac * ap).sum(-1)
        d3, d4 = (ab * bp).sum(-1), (ac * bp).sum(-1)
        d5, d6 = (ab * cp).sum(-1), (ac * cp).sum(-1)
        va, vb, vc = d3 * d6 - d5 * d4, d5 * d2 - d1 * d6, d1 * d4 - d3 * d2
        with np.errstate(divide="ignore", invalid="ignore"):
            denom = 1.0 / (va + vb + vc)
            interior = a + ab * (vb * denom)[..., None] + ac * (vc * denom)[..., None]
            t_ab = np.nan_to_num(d1 / (d1 - d3))[..., None]
            t_ac = np.nan_to_num(d2 / (d2 - d6))[..., None]
            t_bc = np.nan_to_num((d4 - d3) / ((d4 - d3) + (d5 - d6)))[..., None]
        candidate = np.where(np.isfinite(interior), interior, a)
        # The regions, applied outermost-first so the vertex cases win over the edges.
        candidate = np.where(
            ((va <= 0.0) & ((d4 - d3) >= 0.0) & ((d5 - d6) >= 0.0))[..., None],
            b + (c - b) * t_bc,
            candidate,
        )
        candidate = np.where(
            ((vb <= 0.0) & (d2 >= 0.0) & (d6 <= 0.0))[..., None], a + ac * t_ac, candidate
        )
        candidate = np.where(
            ((vc <= 0.0) & (d1 >= 0.0) & (d3 <= 0.0))[..., None], a + ab * t_ab, candidate
        )
        candidate = np.where(((d6 >= 0.0) & (d5 <= d6))[..., None], c, candidate)
        candidate = np.where(((d3 >= 0.0) & (d4 <= d3))[..., None], b, candidate)
        candidate = np.where(((d1 <= 0.0) & (d2 <= 0.0))[..., None], a, candidate)
        distance2 = ((candidate - p) ** 2).sum(-1)
        best = np.argmin(distance2, axis=1)
        rows = np.arange(len(best))
        out[start : start + chunk] = candidate[rows, best]
        which[start : start + chunk] = best
    return out, which


# --- the Warp side -------------------------------------------------------------------------------


def build_kernel():  # type: ignore[no-untyped-def]
    """Import Warp and compile the query kernel. Returns `(wp, kernel)` or `(None, reason)`."""
    try:
        ensure_warp_on_path()
        import warp as wp

        wp.init()
    except Exception as error:  # pragma: no cover - probe
        return None, f"{type(error).__name__}: {error}"

    @wp.kernel
    def query(  # type: ignore[no-untyped-def]
        mesh: wp.uint64,
        points: wp.array(dtype=wp.vec3),
        max_dist: float,
        hit: wp.array(dtype=wp.int32),
        face: wp.array(dtype=wp.int32),
        bary: wp.array(dtype=wp.vec2),
        evaluated: wp.array(dtype=wp.vec3),
    ):
        i = wp.tid()
        result = wp.mesh_query_point_no_sign(mesh, points[i], max_dist)
        if result.result:
            hit[i] = 1
            face[i] = result.face
            bary[i] = wp.vec2(result.u, result.v)
            evaluated[i] = wp.mesh_eval_position(mesh, result.face, result.u, result.v)

    return wp, query


def run_query(wp, kernel, soup: TriangleSoup, points: np.ndarray, device: str):  # type: ignore[no-untyped-def]
    mesh = wp.Mesh(
        points=wp.array(soup.vertices.astype(np.float32), dtype=wp.vec3, device=device),
        indices=wp.array(soup.faces.astype(np.int32).flatten(), dtype=wp.int32, device=device),
    )
    n = len(points)
    hit = wp.zeros(n, dtype=wp.int32, device=device)
    face = wp.zeros(n, dtype=wp.int32, device=device)
    bary = wp.zeros(n, dtype=wp.vec2, device=device)
    evaluated = wp.zeros(n, dtype=wp.vec3, device=device)
    query_points = wp.array(points.astype(np.float32), dtype=wp.vec3, device=device)
    # One unmeasured launch first. Warp compiles a module on its first use on a device, and
    # folding a 1.8 s compile into the first mesh's timing reported 2 189 queries/s for a mesh
    # that actually runs at two million.
    wp.launch(
        kernel,
        dim=n,
        inputs=[mesh.id, query_points, 1.0e6, hit, face, bary, evaluated],
        device=device,
    )
    wp.synchronize_device(device)
    started = time.perf_counter()
    wp.launch(
        kernel,
        dim=n,
        inputs=[mesh.id, query_points, 1.0e6, hit, face, bary, evaluated],
        device=device,
    )
    wp.synchronize_device(device)
    elapsed = time.perf_counter() - started
    return (
        hit.numpy().astype(bool),
        face.numpy().astype(np.int64),
        bary.numpy().astype(np.float64),
        evaluated.numpy().astype(np.float64),
        elapsed,
    )


# --- the questions -------------------------------------------------------------------------------


def barycentric_convention(
    soup: TriangleSoup, face: np.ndarray, bary: np.ndarray, evaluated: np.ndarray
) -> dict[str, float]:
    """Which (u, v) -> vertex mapping reproduces `mesh_eval_position`? Measured, not assumed."""
    v = soup.vertices
    a, b, c = v[soup.faces[face, 0]], v[soup.faces[face, 1]], v[soup.faces[face, 2]]
    u, w = bary[:, 0:1], bary[:, 1:2]
    candidates = {
        "(1-u-v)*v0 + u*v1 + v*v2": (1.0 - u - w) * a + u * b + w * c,
        "u*v0 + v*v1 + (1-u-v)*v2": u * a + w * b + (1.0 - u - w) * c,
        "(1-u-v)*v0 + v*v1 + u*v2": (1.0 - u - w) * a + w * b + u * c,
    }
    return {
        name: float(np.max(np.linalg.norm(point - evaluated, axis=-1)))
        for name, point in candidates.items()
    }


def surface_samples(soup: TriangleSoup, count: int, seed: int) -> np.ndarray:
    """Points *on* the mesh: a uniform barycentric sample of uniformly chosen faces."""
    rng = np.random.default_rng(seed)
    v = soup.vertices
    face = rng.integers(0, soup.n_faces, size=count)
    r1, r2 = rng.random(count), rng.random(count)
    root = np.sqrt(r1)
    w0, w1, w2 = 1.0 - root, root * (1.0 - r2), root * r2
    return (
        w0[:, None] * v[soup.faces[face, 0]]
        + w1[:, None] * v[soup.faces[face, 1]]
        + w2[:, None] * v[soup.faces[face, 2]]
    )


def probe_mesh(wp, kernel, name: str, soup: TriangleSoup, count: int, device: str, seed: int):  # type: ignore[no-untyped-def]
    report: dict[str, object] = {
        "faces": soup.n_faces,
        "vertices": int(soup.vertices.shape[0]),
        "queries": count,
        "device": device,
    }
    on_surface = surface_samples(soup, count, seed)
    hit, face, bary, evaluated, elapsed = run_query(wp, kernel, soup, on_surface, device)
    report["hit_fraction"] = float(hit.mean())
    report["seconds"] = elapsed
    report["queries_per_second"] = float(count / elapsed) if elapsed > 0 else None

    # 1. the round trip: a point already on the surface must come back as itself
    residual = np.linalg.norm(evaluated - on_surface, axis=-1)
    report["round_trip_max_m"] = float(residual.max())
    report["round_trip_p99_m"] = float(np.percentile(residual, 99))
    report["round_trip_within_budget"] = bool(residual.max() <= POSITION_BUDGET_M)

    # 2. the convention
    report["barycentric_residual_m"] = barycentric_convention(soup, face, bary, evaluated)

    # 3. the oracle. Subsampled: it is O(queries x faces) by construction, and a few thousand
    # points already pin agreement to four decimal places.
    sample = slice(0, min(2000, count))
    oracle_point, oracle_face = closest_point_numpy(soup, on_surface[sample])
    report["oracle_sample"] = int(len(oracle_face))
    same_face = face[sample] == oracle_face
    gap = np.linalg.norm(evaluated[sample] - oracle_point, axis=-1)
    report["oracle_same_face_fraction"] = float(same_face.mean())
    report["oracle_position_max_m"] = float(gap.max())
    # Where the two disagree on the face, do they still agree on the point? A query point on a
    # shared edge has two equally correct answers, and that is not an error.
    if (~same_face).any():
        report["oracle_disagreement_position_max_m"] = float(gap[~same_face].max())

    # 4. the position budget: how does a 3.4 mm error in the query point land?
    rng = np.random.default_rng(seed + 1)
    direction = rng.normal(size=on_surface.shape)
    direction /= np.linalg.norm(direction, axis=-1, keepdims=True)
    jittered = on_surface + direction * POSITION_BUDGET_M
    _, face_j, _, evaluated_j, _ = run_query(wp, kernel, soup, jittered, device)
    moved = np.linalg.norm(evaluated_j - evaluated, axis=-1)
    report["budget_face_changed_fraction"] = float((face_j != face).mean())
    report["budget_position_moved_max_m"] = float(moved.max())
    report["budget_position_moved_mean_m"] = float(moved.mean())
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=pathlib.Path, default=REPO / "outputs/isaac_probe/warp_mesh")
    parser.add_argument("--queries", type=int, default=20000)
    parser.add_argument(
        "--device",
        default="cpu",
        help="warp device. 'cpu' is the default on purpose: this is a correctness probe and "
        "the other card is in use. Under CUDA_DEVICE_ORDER=PCI_BUS_ID, cuda:0 is the A6000",
    )
    parser.add_argument("--seed", type=int, default=20260921)
    args = parser.parse_args(argv)

    wp, kernel = build_kernel()
    if wp is None:
        print(f"WM.1: Warp unavailable -- {kernel}")
        report = {"available": False, "reason": kernel}
    else:
        print(f"WM.1: warp {wp.config.version}, device {args.device}")
        radius = 0.25
        meshes = {
            # A box is the clean case: its faces are planar, so the mesh *is* the surface and the
            # residual has nothing but float32 in it. The spheres are the curved case ADR 0087
            # wrote off, at two resolutions, so the chord error is visible as a trend.
            "box_0.4m": (box_mesh((0.0, 0.0, 0.0), (0.4, 0.4, 0.4)), None),
            "sphere_r0.25_64x32": (uv_sphere(radius, 32, 64), sagitta_m(radius, 32, 64)),
            "sphere_r0.25_128x64": (uv_sphere(radius, 64, 128), sagitta_m(radius, 64, 128)),
        }
        report = {"available": True, "warp": str(wp.config.version), "meshes": {}}
        for name, (soup, sagitta) in meshes.items():
            result = probe_mesh(wp, kernel, name, soup, args.queries, args.device, args.seed)
            if sagitta is not None:
                result["chord_sagitta_m"] = sagitta
            report["meshes"][name] = result  # type: ignore[index]
            print(f"\n--- {name}: {soup.n_faces} faces, {args.queries} queries ---")
            print(f"  hit fraction            {result['hit_fraction']:.4f}")
            print(
                f"  round trip              max {1e6 * result['round_trip_max_m']:8.2f} um   "
                f"p99 {1e6 * result['round_trip_p99_m']:8.2f} um   "
                f"(budget {1e3 * POSITION_BUDGET_M:.1f} mm: "
                f"{'PASS' if result['round_trip_within_budget'] else 'FAIL'})"
            )
            print("  barycentric convention:")
            for convention, error in result["barycentric_residual_m"].items():  # type: ignore[union-attr]
                mark = "  <-- this one" if error < 1e-6 else ""
                print(f"    {convention:28s} max {1e6 * error:10.2f} um{mark}")
            print(
                f"  vs NumPy oracle         same face {result['oracle_same_face_fraction']:.4f}   "
                f"position max {1e6 * result['oracle_position_max_m']:.2f} um"
            )
            print(
                f"  under 3.4 mm of position error: face changes on "
                f"{100.0 * result['budget_face_changed_fraction']:.1f} % of queries, "
                f"and the point moves {1e3 * result['budget_position_moved_mean_m']:.2f} mm "
                f"on average (max {1e3 * result['budget_position_moved_max_m']:.2f} mm)"
            )
            print(f"  throughput              {result['queries_per_second']:,.0f} queries/s")

    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "warp_mesh_probe.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
