"""View factors between arbitrary meshes by ray casting, occlusion included (TC.9).

docs/physics-model.md §6.1 (the surround a surface exchanges with) and §6.6 (heat sources that
radiate onto what faces them); ADR 0088 (the closed form this generalises).

ADR 0088's ``corner_view_factor`` is Howell C-11 -- a point facing a *parallel* rectangle -- and
raises on anything else, so a downloaded car cannot radiate onto the road beside it and a battery
cannot warm the shell around it. This module is the general case: every face of every body casts
the same hemispherical quadrature the sky-view code traces (``skyview.sub_rays``, 1740 rays of
known solid angle, rotated into the face's own frame), and the view factor from face *i* to face
*j* is the cosine-weighted share of *i*'s rays whose **nearest** hit lies on *j*:

    F_ij = (1/π) Σ_k ω_k cos θ_k [nearest hit of ray k is face j],

averaged over sample points spread evenly across *i* (each face is cut into ``subdivide²``
congruent sub-triangles and a ray fan leaves each centroid). Occlusion is not a separate step: a
body between two others simply *is* the nearest hit. Whatever misses every body reaches the sky,
and that share is returned too, because TC.10 needs it -- a body that radiates onto a surface also
hides the sky it was seeing (ADR 0088's rule, now for any shape).

It is a NumPy reference, deliberately: clear, tested against Howell's closed forms and against the
identities every view factor must satisfy (reciprocity ``A_i F_ij = A_j F_ji``, and rows of a
closed enclosure summing to one). Speed is not the goal; a scene traces this once at build.

Sign convention: a face radiates from the side its winding normal points to, like every
``TriangleSoup`` in ``raycast`` -- an enclosure's walls face inward, a body's skin faces outward.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from irsim.thermal.raycast import HIT_EPS_M, TriangleSoup
from irsim.thermal.skyview import sub_rays

__all__ = [
    "RadiantMesh",
    "ViewFactors",
    "first_hit",
    "howell_coaxial_discs",
    "howell_parallel_rectangles",
    "howell_perpendicular_rectangles",
    "view_factors",
]

#: How far a ray's origin is lifted off its own face along the normal, so the face it leaves
#: cannot be its own nearest hit; ``HIT_EPS_M`` keeps the neighbours it shares an edge with out.
LIFT_M = 1.0e-6
_CHUNK = 2048


@dataclass(frozen=True)
class RadiantMesh:
    """One body: a triangle soup whose faces radiate from their winding-normal side."""

    name: str
    soup: TriangleSoup

    @property
    def n_faces(self) -> int:
        return self.soup.n_faces

    @property
    def corners(self) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        v, f = self.soup.vertices, self.soup.faces
        return v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]

    @property
    def face_areas_m2(self) -> NDArray[np.float64]:
        a, b, c = self.corners
        return np.asarray(0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1))

    @property
    def face_normals(self) -> NDArray[np.float64]:
        a, b, c = self.corners
        n = np.cross(b - a, c - a)
        length = np.linalg.norm(n, axis=1, keepdims=True)
        if np.any(length <= 0.0):
            raise ValueError(f"{self.name!r} has a degenerate face (zero area)")
        return np.asarray(n / length)

    @property
    def area_m2(self) -> float:
        return float(self.face_areas_m2.sum())


@dataclass(frozen=True)
class ViewFactors:
    """The traced factors: ``face`` is ``(n, n)`` with ``face[i, j] = F_i→j`` over every face of
    every body in order, ``sky`` is ``(n,)`` -- the share of face *i*'s hemisphere that reaches
    nothing -- and ``body`` aggregates by area: ``F_A→B = Σ_{i∈A} A_i Σ_{j∈B} F_ij / A_A``.
    Per face ``Σ_j face[i, j] + sky[i] + back[i] = 1`` to the quadrature's rounding."""

    names: tuple[str, ...]
    owner: NDArray[np.int64]
    areas_m2: NDArray[np.float64]
    face: NDArray[np.float64]
    sky: NDArray[np.float64]
    #: ``(n,)`` the share of face *i*'s rays that landed on the *back* of some face: an opaque
    #: body's unmodelled side, which blocks the ray but radiates nothing here. A real two-sided
    #: panel is two faces with opposite windings, and then every hit is a front hit.
    back: NDArray[np.float64]

    @property
    def n_faces(self) -> int:
        return int(self.face.shape[0])

    @property
    def body(self) -> NDArray[np.float64]:
        n_bodies = len(self.names)
        out = np.zeros((n_bodies, n_bodies))
        power = self.areas_m2[:, None] * self.face  # A_i F_ij, the "power" weight
        for a in range(n_bodies):
            rows = self.owner == a
            area = float(self.areas_m2[rows].sum())
            for b in range(n_bodies):
                cols = self.owner == b
                out[a, b] = power[np.ix_(rows, cols)].sum() / area
        return out

    def _by_body(self, per_face: NDArray[np.float64]) -> NDArray[np.float64]:
        out = np.zeros(len(self.names))
        for a in range(len(self.names)):
            rows = self.owner == a
            out[a] = float((self.areas_m2[rows] * per_face[rows]).sum() / self.areas_m2[rows].sum())
        return out

    @property
    def body_sky(self) -> NDArray[np.float64]:
        """``(n_bodies,)`` area-weighted share of each body's hemisphere that reaches the sky."""
        return self._by_body(self.sky)

    @property
    def body_back(self) -> NDArray[np.float64]:
        """``(n_bodies,)`` area-weighted share that lands on the back of an opaque face."""
        return self._by_body(self.back)

    def reciprocity_defect(self, level: str = "body") -> float:
        """``max |A_i F_ij − A_j F_ji| / max(A_i F_ij)``: zero for an exact solution.

        ``level="body"`` compares bodies, which is what the quadrature is accurate at;
        ``"face"`` compares single faces, where a 1740-ray fan from two sample points is a far
        coarser estimate and the defect is a measure of that resolution, not of the physics.
        """
        if level == "body":
            areas = np.array([self.areas_m2[self.owner == a].sum() for a in range(len(self.names))])
            power = areas[:, None] * self.body
        elif level == "face":
            power = self.areas_m2[:, None] * self.face
        else:
            raise ValueError("level must be 'body' or 'face'")
        scale = float(power.max()) if power.size else 1.0
        if scale <= 0.0:
            return 0.0
        return float(np.abs(power - power.T).max() / scale)


def first_hit(
    soup: TriangleSoup, origins: Any, directions: Any
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Nearest intersection of each ray with the soup: ``(distance (n,), face (n,))``.

    Möller-Trumbore, every ray against every triangle, like ``TriangleSoup.blocked`` -- but
    keeping the *nearest* hit and its face rather than stopping at the first, because a view
    factor needs to know which face a ray lands on, not only that it landed. A ray that misses
    has distance ``inf`` and face ``-1``.
    """
    o_all = np.atleast_2d(np.asarray(origins, dtype=np.float64))
    d_all = np.broadcast_to(np.atleast_2d(np.asarray(directions, dtype=np.float64)), o_all.shape)
    n = o_all.shape[0]
    distance = np.full(n, np.inf)
    face = np.full(n, -1, dtype=np.int64)
    v0 = soup.vertices[soup.faces[:, 0]]
    e1 = soup.vertices[soup.faces[:, 1]] - v0
    e2 = soup.vertices[soup.faces[:, 2]] - v0
    for start in range(0, n, _CHUNK):
        stop = min(start + _CHUNK, n)
        o, d = o_all[start:stop], d_all[start:stop]
        pvec = np.cross(d[:, None, :], e2[None, :, :])
        det = np.einsum("mj,nmj->nm", e1, pvec)
        parallel = np.abs(det) < 1e-12
        inv_det = np.where(parallel, 0.0, 1.0 / np.where(parallel, 1.0, det))
        tvec = o[:, None, :] - v0[None, :, :]
        u = np.einsum("nmj,nmj->nm", tvec, pvec) * inv_det
        qvec = np.cross(tvec, e1[None, :, :])
        v = np.einsum("nj,nmj->nm", d, qvec) * inv_det
        t = np.einsum("mj,nmj->nm", e2, qvec) * inv_det
        ok = (~parallel) & (u >= 0.0) & (v >= 0.0) & (u + v <= 1.0) & (t > HIT_EPS_M)
        t = np.where(ok, t, np.inf)
        nearest = np.argmin(t, axis=1)
        best = t[np.arange(t.shape[0]), nearest]
        hit = np.isfinite(best)
        distance[start:stop] = best
        face[start:stop] = np.where(hit, nearest, -1)
    return distance, face


def _tangent_frame(normal: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Two unit tangents completing ``normal`` to a right-handed frame."""
    helper = np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    t1 = np.cross(normal, helper)
    t1 /= np.linalg.norm(t1)
    t2 = np.cross(normal, t1)
    return t1, t2


def _sample_points(a: Any, b: Any, c: Any, subdivide: int) -> NDArray[np.float64]:
    """Centroids of the ``subdivide²`` congruent sub-triangles of ``abc``, all of equal area."""
    pts = []
    m = subdivide
    for i in range(m):
        for j in range(m - i):
            # the "upright" sub-triangle with corner (i, j) on the barycentric lattice
            u, v = (i + 1.0 / 3.0) / m, (j + 1.0 / 3.0) / m
            pts.append(a + u * (b - a) + v * (c - a))
            if j < m - i - 1:  # the inverted one beside it
                u, v = (i + 2.0 / 3.0) / m, (j + 2.0 / 3.0) / m
                pts.append(a + u * (b - a) + v * (c - a))
    return np.asarray(pts)


def view_factors(
    meshes: Sequence[RadiantMesh], *, subdivide: int = 2, lift_m: float = LIFT_M
) -> ViewFactors:
    """Trace ``F_ij`` between every face of every body, with occlusion, plus the sky share.

    ``subdivide`` cuts each face into ``subdivide²`` equal sub-triangles and casts one ray fan
    (1740 rays) from each centroid, so a face's factor is a proper area average rather than its
    centroid's alone. With ``2``, the default, a unit square meshed 4 × 4 reproduces Howell's
    parallel and perpendicular rectangles to 0.3 % and a 16-gon disc the coaxial discs to 0.2 %
    (equal-area radii); the cost is ``Σ faces × subdivide² × 1740`` ray-triangle tests against
    every triangle, so a 300-face pair takes a minute and a 30-face pair a second.
    """
    if not meshes:
        raise ValueError("view_factors needs at least one body")
    if subdivide < 1:
        raise ValueError("subdivide must be >= 1")
    names = tuple(m.name for m in meshes)
    if len(set(names)) != len(names):
        raise ValueError("every body needs its own name")
    # one soup of everything, with the owner of each face
    vertices, faces, owner = [], [], []
    offset = 0
    for k, m in enumerate(meshes):
        vertices.append(m.soup.vertices)
        faces.append(m.soup.faces + offset)
        owner.append(np.full(m.n_faces, k, dtype=np.int64))
        offset += m.soup.vertices.shape[0]
    world = TriangleSoup(np.concatenate(vertices), np.concatenate(faces))
    owner_arr = np.concatenate(owner)
    areas = np.concatenate([m.face_areas_m2 for m in meshes])
    normals = np.concatenate([m.face_normals for m in meshes])
    n = world.n_faces

    directions_up, omegas, _ = sub_rays()  # z-up hemisphere, Σ ω = 2π
    weights_up = omegas * directions_up[:, 2] / math.pi  # ω cos θ / π, Σ = 1 in the limit
    weights_up = weights_up / weights_up.sum()  # the quadrature's own unit, so rows close exactly

    face_matrix = np.zeros((n, n))
    sky = np.zeros(n)
    back = np.zeros(n)
    a_all, b_all, c_all = (
        world.vertices[world.faces[:, 0]],
        world.vertices[world.faces[:, 1]],
        world.vertices[world.faces[:, 2]],
    )
    for i in range(n):
        normal = normals[i]
        t1, t2 = _tangent_frame(normal)
        dirs = (
            directions_up[:, 0:1] * t1 + directions_up[:, 1:2] * t2 + directions_up[:, 2:3] * normal
        )
        points = _sample_points(a_all[i], b_all[i], c_all[i], subdivide) + lift_m * normal
        n_pts = points.shape[0]
        origins = np.repeat(points, dirs.shape[0], axis=0)
        rays = np.tile(dirs, (n_pts, 1))
        _, hit = first_hit(world, origins, rays)
        w = np.tile(weights_up, n_pts) / n_pts
        landed = hit >= 0
        # a front hit arrives against the face's normal; a back hit is an opaque body's far side
        front = landed.copy()
        front[landed] = np.einsum("kj,kj->k", rays[landed], normals[hit[landed]]) < 0.0
        np.add.at(face_matrix[i], hit[front], w[front])
        back[i] = float(w[landed & ~front].sum())
        sky[i] = float(w[~landed].sum())
    return ViewFactors(
        names=names, owner=owner_arr, areas_m2=areas, face=face_matrix, sky=sky, back=back
    )


# --- Howell's closed forms, the oracles ---------------------------------------------------------


def howell_parallel_rectangles(a_m: float, b_m: float, c_m: float) -> float:
    """C-11: two identical, aligned ``a × b`` rectangles a distance ``c`` apart, F_1→2."""
    x, y = a_m / c_m, b_m / c_m
    x2, y2 = x * x, y * y
    term = (
        0.5 * math.log((1.0 + x2) * (1.0 + y2) / (1.0 + x2 + y2))
        + x * math.sqrt(1.0 + y2) * math.atan(x / math.sqrt(1.0 + y2))
        + y * math.sqrt(1.0 + x2) * math.atan(y / math.sqrt(1.0 + x2))
        - x * math.atan(x)
        - y * math.atan(y)
    )
    return 2.0 / (math.pi * x * y) * term


def howell_perpendicular_rectangles(h_m: float, w_m: float, l_m: float) -> float:
    """C-14: rectangles ``w × l`` (radiating, 1) and ``h × l`` (2) sharing the edge ``l`` at a
    right angle, F_1→2."""
    h, w = h_m / l_m, w_m / l_m
    h2, w2 = h * h, w * w
    r2 = h2 + w2
    term = (
        w * math.atan(1.0 / w)
        + h * math.atan(1.0 / h)
        - math.sqrt(r2) * math.atan(1.0 / math.sqrt(r2))
        + 0.25
        * math.log(
            ((1.0 + w2) * (1.0 + h2) / (1.0 + r2))
            * (w2 * (1.0 + r2) / ((1.0 + w2) * r2)) ** w2
            * (h2 * (1.0 + r2) / ((1.0 + h2) * r2)) ** h2
        )
    )
    return term / (math.pi * w)


def howell_coaxial_discs(r1_m: float, r2_m: float, h_m: float) -> float:
    """C-41: coaxial parallel discs of radii ``r1`` (radiating) and ``r2``, ``h`` apart, F_1→2."""
    big_r1, big_r2 = r1_m / h_m, r2_m / h_m
    s = 1.0 + (1.0 + big_r2 * big_r2) / (big_r1 * big_r1)
    return 0.5 * (s - math.sqrt(s * s - 4.0 * (big_r2 / big_r1) ** 2))
