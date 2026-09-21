#!/usr/bin/env python3
"""Roadmap WM.1, the in-Kit half: a `wp.Mesh` built from a **prim's own** points and indices.

`probe_warp_mesh.py` settles the query itself against analytic geometry with no Kit boot. This
script answers the other half of WM.1's sentence -- "build a `wp.Mesh` from a prim's own points
and indices" -- because the gap between a NumPy triangle soup and a USD prim is where the work
actually is:

* **USD meshes are not necessarily triangles.** `faceVertexCounts` is per face, and an asset
  authored as quads gives 4s. `wp.Mesh` takes triangles only, so something must triangulate, and
  the face indices Warp returns are then indices into *that* triangulation, not into the prim's
  faces. WM.2's per-face cell resolution has to be defined on whichever of the two it means.
* **A gprim has no points at all.** `UsdGeom.Sphere` is an analytic primitive; there is no
  `points` attribute to hand Warp until something tessellates it.
* **Points are in local space.** They need the prim's local-to-world transform before they can
  meet a world-space position AOV.

Each of those is a way to get a plausible-looking wrong answer rather than an error. Failures are
data: this script reports and does not raise.

    python.sh scripts/probe_warp_prim.py --out outputs/isaac_probe/warp_mesh
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from irsim_isaac.env import simulation_app_config  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--out", type=pathlib.Path, default=REPO / "outputs/isaac_probe/warp_mesh")
parser.add_argument("--queries", type=int, default=8000)
parser.add_argument("--seed", type=int, default=20260921)
args = parser.parse_args()

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp(simulation_app_config())

import numpy as np  # noqa: E402
from pxr import Gf, Usd, UsdGeom  # noqa: E402

from irsim.thermal.raycast import TriangleSoup  # noqa: E402


def _load_mesh_probe():  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location(
        "probe_warp_mesh", REPO / "scripts" / "probe_warp_mesh.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mesh_probe = _load_mesh_probe()


# --- authoring ------------------------------------------------------------------------------------


def author_quad_cube(stage: Usd.Stage, path: str, size: float, translate) -> None:  # type: ignore[no-untyped-def]
    """A cube as six **quads**, which is how a modelling package exports one."""
    h = 0.5 * size
    corners = [
        (-h, -h, -h), (h, -h, -h), (h, h, -h), (-h, h, -h),
        (-h, -h, h), (h, -h, h), (h, h, h), (-h, h, h),
    ]  # fmt: skip
    quads = [
        (0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4),
        (2, 3, 7, 6), (0, 4, 7, 3), (1, 2, 6, 5),
    ]  # fmt: skip
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr([Gf.Vec3f(*c) for c in corners])
    mesh.CreateFaceVertexCountsAttr([4] * len(quads))
    mesh.CreateFaceVertexIndicesAttr([i for quad in quads for i in quad])
    UsdGeom.Xformable(mesh).AddTranslateOp().Set(Gf.Vec3d(*translate))


def author_triangle_sphere(stage: Usd.Stage, path: str, soup: TriangleSoup, translate) -> None:  # type: ignore[no-untyped-def]
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr([Gf.Vec3f(*p) for p in soup.vertices])
    mesh.CreateFaceVertexCountsAttr([3] * soup.n_faces)
    mesh.CreateFaceVertexIndicesAttr([int(i) for i in soup.faces.flatten()])
    UsdGeom.Xformable(mesh).AddTranslateOp().Set(Gf.Vec3d(*translate))


# --- ingestion ------------------------------------------------------------------------------------


def soup_from_prim(prim: Usd.Prim, time=None):  # type: ignore[no-untyped-def]
    """``(TriangleSoup, note)`` in **world** space, or ``(None, why not)``."""
    time = Usd.TimeCode.Default() if time is None else time
    mesh = UsdGeom.Mesh(prim)
    if not mesh:
        return None, f"{prim.GetTypeName()} is not a UsdGeom.Mesh: no points attribute to read"
    points = mesh.GetPointsAttr().Get(time)
    counts = mesh.GetFaceVertexCountsAttr().Get(time)
    indices = mesh.GetFaceVertexIndicesAttr().Get(time)
    if points is None or counts is None or indices is None:
        return None, "the mesh declares no points/faceVertexCounts/faceVertexIndices"

    local = np.asarray([[p[0], p[1], p[2]] for p in points], dtype=np.float64)
    # Local -> world. A prim under a translated parent whose points were handed to Warp raw would
    # query a mesh sitting at the origin while the position AOV reports the world, and every hit
    # would be wrong by the translation with no error anywhere.
    matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(time)
    m = np.asarray([[matrix[r][c] for c in range(4)] for r in range(4)], dtype=np.float64)
    world = local @ m[:3, :3] + m[3, :3]

    counts = np.asarray(counts, dtype=np.int64)
    indices = np.asarray(indices, dtype=np.int64)
    faces = []
    cursor = 0
    for n in counts:
        ring = indices[cursor : cursor + n]
        cursor += int(n)
        # Fan triangulation. Correct for the convex polygons a cube's faces are; a concave
        # n-gon would need ear clipping, and WM.2 should say which it assumes.
        for k in range(1, int(n) - 1):
            faces.append([int(ring[0]), int(ring[k]), int(ring[k + 1])])
    note = (
        f"{len(counts)} faces, counts {sorted(set(int(c) for c in counts))} -> "
        f"{len(faces)} triangles"
    )
    return TriangleSoup(world, np.asarray(faces, dtype=np.int64)), note


# --- the probe -----------------------------------------------------------------------------------


def main() -> int:
    report: dict[str, object] = {"prims": {}}
    wp, kernel = mesh_probe.build_kernel()
    if wp is None:
        print(f"WM.1 (prim): warp unavailable -- {kernel}")
        report["available"] = False
        report["reason"] = kernel
    else:
        report["available"] = True
        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        author_quad_cube(stage, "/World/QuadCube", 0.4, (1.0, -2.0, 0.5))
        author_triangle_sphere(
            stage, "/World/TriSphere", mesh_probe.uv_sphere(0.25, 32, 64), (-3.0, 4.0, 1.5)
        )
        UsdGeom.Sphere.Define(stage, "/World/Gprim").CreateRadiusAttr(0.25)

        for path in ("/World/QuadCube", "/World/TriSphere", "/World/Gprim"):
            entry: dict[str, object] = {}
            soup, note = soup_from_prim(stage.GetPrimAtPath(path))
            entry["ingest"] = note
            print(f"\n--- {path} ---\n  {note}")
            if soup is None:
                entry["usable"] = False
                report["prims"][path] = entry  # type: ignore[index]
                continue
            entry["usable"] = True
            entry["triangles"] = soup.n_faces
            lo, hi = soup.bounds_m
            entry["world_bounds_m"] = [lo.tolist(), hi.tolist()]
            print(f"  world bounds {np.round(lo, 3).tolist()} .. {np.round(hi, 3).tolist()}")

            points = mesh_probe.surface_samples(soup, args.queries, args.seed)
            hit, face, bary, evaluated, _ = mesh_probe.run_query(wp, kernel, soup, points, "cpu")
            residual = np.linalg.norm(evaluated - points, axis=-1)
            entry["hit_fraction"] = float(hit.mean())
            entry["round_trip_max_m"] = float(residual.max())
            entry["round_trip_within_budget"] = bool(residual.max() <= mesh_probe.POSITION_BUDGET_M)
            conventions = mesh_probe.barycentric_convention(soup, face, bary, evaluated)
            best = min(conventions, key=lambda k: conventions[k])
            entry["barycentric_convention"] = best
            entry["barycentric_residual_m"] = conventions
            sample = slice(0, min(2000, args.queries))
            oracle_point, oracle_face = mesh_probe.closest_point_numpy(soup, points[sample])
            entry["oracle_same_face_fraction"] = float((face[sample] == oracle_face).mean())
            entry["oracle_position_max_m"] = float(
                np.max(np.linalg.norm(evaluated[sample] - oracle_point, axis=-1))
            )
            print(
                f"  hit {entry['hit_fraction']:.4f}   round trip max "
                f"{1e6 * entry['round_trip_max_m']:.2f} um   "
                f"({'PASS' if entry['round_trip_within_budget'] else 'FAIL'} vs 3.4 mm)"
            )
            print(f"  convention {best}  ({1e6 * conventions[best]:.3f} um)")
            print(
                f"  vs oracle: same face {entry['oracle_same_face_fraction']:.4f}, "
                f"position max {1e6 * entry['oracle_position_max_m']:.2f} um"
            )
            report["prims"][path] = entry  # type: ignore[index]

    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "warp_prim_probe.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nwrote {path}")
    return 0


try:
    code = main()
except Exception as error:  # pragma: no cover - probe
    print(f"probe failed: {type(error).__name__}: {error}")
    code = 1
finally:
    app.close()

raise SystemExit(code)
