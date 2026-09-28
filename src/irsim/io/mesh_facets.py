"""Is this mesh smooth enough to image? Facet angles, and the angle a camera can resolve.

The `low2high` skill moves a mesh in either direction — a phone-game model up to smooth geometry, a
heavy CAD tessellation down to a workable one — and both directions need the same yardstick: is a
curved surface approximated finely enough that the camera cannot see the facets? Roadmap `AI.10`,
ADR 0155.

**Why it matters in both bands of the RGB–IR pair.** A coarse facet is a step in the surface
normal. In the visible companion that is a shading stripe. In the infrared it is worse than
cosmetic: directional emissivity and the reflected-sky term both follow the normal (§4.3, §8), so
each facet reports a slightly different apparent temperature and a smooth hub renders as bands —
a structure a detector trained on the frames can learn and real footage will never show. And at
the silhouette a facet is a polygon corner against the sky, visible whenever the corner's
deviation from the true curve exceeds the pixel footprint.

**What is measured.** The dihedral angle across every interior edge, after welding coincident
vertices: glTF and most game exports split vertices along every UV seam and hard edge, and an
unwelded mesh would report every seam as a boundary rather than as the curve it is. Edges are then
read in three bands:

* **smooth** — at or below ``smooth_max_deg``: the curve is resolved;
* **faceted** — between that and ``crease_min_deg``: too coarse to be a curve, too gentle to be a
  designed edge. This is the band that is the defect;
* **crease** — at or above ``crease_min_deg``: a real edge (a box corner, a flange).

The crease boundary is a heuristic and is reported as one. A low-poly cylinder of 8 segments has
45° edges and is faceted; a hex nut has 60° edges and is not — geometry alone cannot tell a coarse
curve from a designed polygon, which is why the default sits at 60° and why the skill confirms the
faceted edges by looking at a render rather than by trusting this number alone.

**Where ``smooth_max_deg`` comes from.** A chord spanning angle α on a circle of radius r misses
the arc by the sagitta ``s = r (1 − cos(α/2))``. The facet is invisible at the silhouette when
``s`` is under half the pixel footprint ``R·IFOV`` at the closest range the scene flies it
(:func:`max_facet_angle_deg`). A 10° default holds ``s ≤ 0.0038 r`` — sub-pixel for a 5 cm hub
seen at 1 mrad from 0.5 m and beyond — and the skill derives the real number from the scene.
"""

from __future__ import annotations

import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "DEFAULT_CREASE_MIN_DEG",
    "DEFAULT_SMOOTH_MAX_DEG",
    "EdgeDihedrals",
    "FacetReport",
    "edge_dihedrals",
    "edge_sagitta_m",
    "facet_report",
    "glb_meshes",
    "max_facet_angle_deg",
    "sagitta_m",
    "weld_vertices",
]

#: A 10° chord misses its arc by 0.38 % of the radius; see the module docstring.
DEFAULT_SMOOTH_MAX_DEG = 10.0
#: At or above this an edge is read as designed. 60° keeps a hex nut out of the defect band and
#: puts a 6-segment cylinder in it only just — the ambiguity is geometric, not a tuning choice.
DEFAULT_CREASE_MIN_DEG = 60.0


def sagitta_m(radius_m: float, facet_angle_deg: float) -> float:
    """How far a chord spanning ``facet_angle_deg`` misses a circle of ``radius_m``."""
    return radius_m * (1.0 - math.cos(math.radians(facet_angle_deg) / 2.0))


def max_facet_angle_deg(radius_m: float, range_m: float, ifov_rad: float) -> float:
    """The coarsest facet angle whose silhouette error stays under half a pixel.

    Solves ``sagitta(r, α) = R·IFOV / 2`` for α. A curve smaller than half a pixel across needs
    no facets at all to look right, so the answer is capped at 180°.
    """
    if radius_m <= 0.0 or range_m <= 0.0 or ifov_rad <= 0.0:
        raise ValueError("radius, range and IFOV must be positive")
    half_pixel = 0.5 * range_m * ifov_rad
    ratio = 1.0 - half_pixel / radius_m
    if ratio <= -1.0:
        return 180.0
    return min(180.0, math.degrees(2.0 * math.acos(max(-1.0, ratio))))


def weld_vertices(
    vertices_m: NDArray[np.float64], faces: NDArray[np.intp], tol_m: float = 1e-6
) -> tuple[NDArray[np.float64], NDArray[np.intp]]:
    """Merge vertices closer than ``tol_m`` (grid snap), returning the welded mesh.

    Grid snapping can miss two points that straddle a cell boundary; with ``tol_m`` far below the
    mesh's own edge lengths that only leaves a split that the edge census reports as a boundary,
    which is the conservative direction (a missed curve, never an invented one).
    """
    keys = np.round(np.asarray(vertices_m, dtype=np.float64) / tol_m).astype(np.int64)
    _, first, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    welded = np.asarray(vertices_m, dtype=np.float64)[first]
    return welded, inverse.reshape(-1)[np.asarray(faces, dtype=np.intp)]


@dataclass(frozen=True)
class EdgeDihedrals:
    """Per-interior-edge dihedral angles and lengths, plus the edge census."""

    angle_deg: NDArray[np.float64]
    length_m: NDArray[np.float64]
    #: Mean distance from the shared edge to the two faces' opposite vertices: the facet width a
    #: curve is being approximated with, across this edge.
    width_m: NDArray[np.float64]
    boundary_edges: int
    nonmanifold_edges: int
    degenerate_faces: int


def edge_dihedrals(
    vertices_m: NDArray[np.float64], faces: NDArray[np.intp], weld_tol_m: float = 1e-6
) -> EdgeDihedrals:
    """The angle between the two face normals across every edge shared by exactly two faces.

    0° is flat, 90° a box corner. Game exports are often inconsistently wound, and a flipped face
    would read 180° − θ. So a flipped neighbour is detected — two consistently wound faces
    traverse their shared edge in opposite directions, a flipped pair in the same one — and its
    normal negated before the angle is taken, which reports the true angle either way.
    """
    verts, tris = weld_vertices(vertices_m, faces, weld_tol_m)
    a, b, c = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
    normals = np.cross(b - a, c - a)
    norm = np.linalg.norm(normals, axis=1)
    good = norm > 0.0
    degenerate = int((~good).sum())
    tris, normals = tris[good], normals[good] / norm[good, None]

    # Directed edges (u -> v) per face; key each by its undirected pair.
    u = tris[:, [0, 1, 2]].reshape(-1)
    v = tris[:, [1, 2, 0]].reshape(-1)
    face_of = np.repeat(np.arange(len(tris)), 3)
    lo, hi = np.minimum(u, v), np.maximum(u, v)
    order = np.lexsort((hi, lo))
    lo, hi, u, face_of = lo[order], hi[order], u[order], face_of[order]
    starts = np.flatnonzero(np.r_[True, (lo[1:] != lo[:-1]) | (hi[1:] != hi[:-1])])
    counts = np.diff(np.r_[starts, len(lo)])

    pairs = starts[counts == 2]
    f1, f2 = face_of[pairs], face_of[pairs + 1]
    cos = np.einsum("ij,ij->i", normals[f1], normals[f2])
    # Consistently wound neighbours traverse a shared edge in opposite directions; the same
    # direction means one face is flipped, and its normal must be negated to compare shapes.
    flipped = u[pairs] == u[pairs + 1]
    cos = np.where(flipped, -cos, cos)
    angle = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
    p0, p1 = verts[lo[pairs]], verts[hi[pairs]]
    edge = p1 - p0
    length = np.linalg.norm(edge, axis=1)
    width = 0.5 * (
        _opposite_height(verts, tris[f1], p0, edge, length)
        + _opposite_height(verts, tris[f2], p0, edge, length)
    )
    return EdgeDihedrals(
        angle_deg=angle,
        length_m=length,
        width_m=width,
        boundary_edges=int((counts == 1).sum()),
        nonmanifold_edges=int((counts > 2).sum()),
        degenerate_faces=degenerate,
    )


def _opposite_height(
    verts: NDArray[np.float64],
    tris: NDArray[np.intp],
    p0: NDArray[np.float64],
    edge: NDArray[np.float64],
    length: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Distance from each triangle's vertex farthest off the edge line to that line."""
    safe = np.where(length > 0.0, length, 1.0)[:, None]
    heights = [
        np.linalg.norm(np.cross(verts[tris[:, k]] - p0, edge), axis=1) / safe[:, 0]
        for k in range(3)
    ]
    return np.asarray(np.max(np.stack(heights), axis=0), dtype=np.float64)


def edge_sagitta_m(
    angle_deg: NDArray[np.float64], width_m: NDArray[np.float64]
) -> NDArray[np.float64]:
    """How far the facets meeting at an edge stand off the curve they approximate.

    Two facets of width ``w`` meeting at angle θ are chords of a circle of radius
    ``w / (2 sin(θ/2))``, and each misses its arc by ``r (1 − cos(θ/2)) = (w/2) tan(θ/4)``. So the
    edge carries its own silhouette error and no radius has to be supplied: a sub-pixel screw head
    of 45° facets and a coarse 45° fuselage are told apart by this, not by the angle.
    """
    return np.asarray(0.5 * width_m * np.tan(np.radians(angle_deg) / 4.0), dtype=np.float64)


@dataclass(frozen=True)
class FacetReport:
    """How much of a mesh's edge length sits in each band, and whether it passes."""

    name: str
    faces: int
    smooth_max_deg: float
    crease_min_deg: float
    edge_length_m: float
    faceted_length_m: float
    max_faceted_deg: float
    boundary_edges: int
    nonmanifold_edges: int
    allowed_fraction: float
    max_sagitta_m: float | None = None

    @property
    def faceted_fraction(self) -> float:
        return 0.0 if self.edge_length_m == 0.0 else self.faceted_length_m / self.edge_length_m

    @property
    def passed(self) -> bool:
        return self.faceted_fraction <= self.allowed_fraction

    def render(self) -> str:
        verdict = "PASS" if self.passed else "FACETED"
        worst = f", worst {self.max_faceted_deg:.1f} deg" if self.faceted_length_m else ""
        rule = (
            f"sagitta > {1e3 * self.max_sagitta_m:.3g} mm"
            if self.max_sagitta_m is not None
            else f"({self.smooth_max_deg:g}, {self.crease_min_deg:g}) deg"
        )
        return (
            f"{verdict:8s} {self.name}: {self.faces:,} faces, "
            f"{100 * self.faceted_fraction:.1f} % of edge length {rule}{worst}; "
            f"{self.boundary_edges} boundary / {self.nonmanifold_edges} non-manifold edges"
        )


def facet_report(
    name: str,
    vertices_m: NDArray[np.float64],
    faces: NDArray[np.intp],
    *,
    smooth_max_deg: float = DEFAULT_SMOOTH_MAX_DEG,
    crease_min_deg: float = DEFAULT_CREASE_MIN_DEG,
    allowed_fraction: float = 0.01,
    weld_tol_m: float = 1e-6,
    max_sagitta_m: float | None = None,
) -> FacetReport:
    """Measure one mesh against the smooth/faceted/crease bands.

    Two readings of "faceted", below the crease angle either way:

    * **with a scene** (``max_sagitta_m`` = half the pixel footprint at the closest range): an
      edge is faceted when its own silhouette error (:func:`edge_sagitta_m`) exceeds that —
      the physical criterion, which lets sub-pixel detail be as coarse as it likes;
    * **without one**: an edge is faceted when its angle exceeds ``smooth_max_deg`` — a
      scale-free proxy that flags small coarse details the camera may never resolve.

    Weighted by edge **length**, not count: a dense patch of tiny smooth edges should not dilute
    one long faceted seam down the side of a fuselage.
    """
    if not 0.0 <= smooth_max_deg < crease_min_deg <= 180.0:
        raise ValueError("need 0 <= smooth_max_deg < crease_min_deg <= 180")
    d = edge_dihedrals(vertices_m, faces, weld_tol_m)
    below_crease = d.angle_deg < crease_min_deg
    if max_sagitta_m is None:
        band = below_crease & (d.angle_deg > smooth_max_deg)
    else:
        if max_sagitta_m <= 0.0:
            raise ValueError("max_sagitta_m must be positive")
        band = below_crease & (edge_sagitta_m(d.angle_deg, d.width_m) > max_sagitta_m)
    return FacetReport(
        name=name,
        faces=int(np.asarray(faces).shape[0]),
        smooth_max_deg=smooth_max_deg,
        crease_min_deg=crease_min_deg,
        edge_length_m=float(d.length_m.sum()),
        faceted_length_m=float(d.length_m[band].sum()),
        max_faceted_deg=float(d.angle_deg[band].max()) if band.any() else 0.0,
        boundary_edges=d.boundary_edges,
        nonmanifold_edges=d.nonmanifold_edges,
        allowed_fraction=allowed_fraction,
        max_sagitta_m=max_sagitta_m,
    )


_GLB_COMPONENT: dict[int, type[np.generic]] = {
    5121: np.uint8,
    5123: np.uint16,
    5125: np.uint32,
    5126: np.float32,
}
_GLB_WIDTH = {"SCALAR": 1, "VEC3": 3}


def glb_meshes(path: Path) -> list[tuple[str, NDArray[np.float64], NDArray[np.intp]]]:
    """``(name, vertices, faces)`` for every triangle primitive in a binary glTF.

    Read straight from the GLB container with ``json`` and NumPy, so the facet check needs neither
    Blender nor a mesh library. Vertices are in each mesh's own frame: node transforms are not
    applied, which leaves every dihedral angle unchanged (rigid motion and uniform scale preserve
    angles) but would distort them under a non-uniform node scale — the report is per mesh, and
    such a node is rare enough in CAD and game exports to be named in the skill rather than
    handled here.
    """
    data = Path(path).read_bytes()
    if data[:4] != b"glTF":
        raise ValueError(f"{path}: not a binary glTF (.glb)")
    (json_len,) = struct.unpack("<I", data[12:16])
    gltf = json.loads(data[20 : 20 + json_len])
    binary = data[20 + json_len + 8 :]

    def read(index: int) -> NDArray[np.generic]:
        acc = gltf["accessors"][index]
        view = gltf["bufferViews"][acc["bufferView"]]
        dtype = np.dtype(_GLB_COMPONENT[acc["componentType"]])
        width = _GLB_WIDTH[acc["type"]]
        stride = view.get("byteStride")
        if stride and stride != dtype.itemsize * width:
            raise ValueError(f"{path}: interleaved buffer views are not supported")
        start = view.get("byteOffset", 0) + acc.get("byteOffset", 0)
        return np.frombuffer(binary, dtype=dtype, count=acc["count"] * width, offset=start)

    out: list[tuple[str, NDArray[np.float64], NDArray[np.intp]]] = []
    for m, mesh in enumerate(gltf.get("meshes", [])):
        # One entry per mesh, its primitives concatenated: a CAD writer emits every B-rep face as
        # its own primitive, and measured apart the edges *between* faces would never be seen.
        verts_all: list[NDArray[np.float64]] = []
        faces_all: list[NDArray[np.intp]] = []
        offset = 0
        for prim in mesh.get("primitives", []):
            if prim.get("mode", 4) != 4 or "indices" not in prim:
                continue  # points, lines and strips carry no facets
            verts = read(prim["attributes"]["POSITION"]).reshape(-1, 3).astype(np.float64)
            faces = read(prim["indices"]).astype(np.intp).reshape(-1, 3) + offset
            verts_all.append(verts)
            faces_all.append(faces)
            offset += len(verts)
        if faces_all:
            out.append(
                (mesh.get("name") or f"mesh_{m}", np.vstack(verts_all), np.vstack(faces_all))
            )
    return out
