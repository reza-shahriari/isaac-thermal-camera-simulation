"""Lateral conduction between the cells of a triangle mesh (WM.6).

docs/physics-model.md §6.1 (the conduction term), §6.4 (the integrator and its bound);
ADR 0094 (the operator and its backward Euler step), ADR 0102 (`PT.11`, the same term on a
rectangular grid), ADR 0110 (the field these cells belong to), ADR 0112 (this choice).

`PT.11` gave a planar patch's cells a link to their four neighbours and `ADR 0111` measured what
its absence costs on a mesh: the quadrotor arm's 22.1 K crown-to-underside span is an upper bound,
and the same ring solved with conduction is **26 %** smaller. This module supplies the missing
links, as the same `ConductionOperator` a patch uses, so the implicit step, the prefactorisation
and the energy-conservation guarantee are reused rather than restated.

**The link.** Two cells sharing an edge of length ``w``, their centroids a perpendicular distance
``d_a`` and ``d_b`` from it, get the two-point flux

    G = k δ w / (d_a + d_b)          [W/K]

which is the same ``k δ w / d`` `PT.11` uses on a rectangle -- there ``d_a = d_b`` is half a cell
and the formula is identical. Within a face this collapses to a closed form that does not depend
on the level at all: a sub-edge parallel to the parent's side ``E`` carries

    G = (3/4) k δ |E|² / A_face

because the barycentric cut is affine, so every sub-triangle is similar to its parent and the
ratio ``w / d`` is scale-free. Refining a face therefore adds cells without changing how fast heat
crosses it, which is the property a discretisation of a continuum has to have.

**Why not the cotan weight.** The consistent finite-volume scheme on a triangulation puts the node
at the **circumcentre**, not the centroid, because then the line joining two nodes is perpendicular
to the edge they share. Its weight is the dual of the familiar cotan Laplacian,

    G = 2 k δ / (cot θ_a + cot θ_b)

and it is **negative** whenever the two angles opposite the shared edge sum past π. Inside one
face the two sub-triangles across an edge are congruent, so ``θ_a = θ_b`` is the parent's own angle
opposite that side and the weight is ``k δ tan Θ``: negative for every **obtuse** face, and
singular for a right-angled one, where the circumcentre lies on the shared edge itself. A negative
conductance breaks the discrete maximum principle -- a cell leaves the range spanned by its
neighbours and the forcing, and renders as a bright speck indistinguishable from a bad pixel.

So this module ships the centroid form, which is non-negative by construction, and
`circumcentric_conductance` exists as the comparison the tests hold it against. The two agree
**exactly** on an equilateral face (centroid and circumcentre coincide there, and
``(3/4)|E|²/A = tan 60° = √3``); they part company as the face is skewed, and `ADR 0112` records
what that costs.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

from irsim.thermal.conduction import ConductionOperator

__all__ = [
    "circumcentric_conductance",
    "edge_spans",
    "mesh_lateral_operator",
    "within_face_links",
]


def _face_edges(vertices_m: NDArray[np.float64], faces: NDArray[np.intp]) -> Any:
    """``(lengths (n_faces, 3), areas (n_faces,))``; side ``s`` is the one opposite vertex ``s``."""
    v = vertices_m[faces]
    e0 = np.linalg.norm(v[:, 2] - v[:, 1], axis=-1)
    e1 = np.linalg.norm(v[:, 0] - v[:, 2], axis=-1)
    e2 = np.linalg.norm(v[:, 1] - v[:, 0], axis=-1)
    area = 0.5 * np.linalg.norm(np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]), axis=-1)
    return np.stack([e0, e1, e2], axis=1), area


def within_face_links(patch: Any, face: int) -> list[tuple[int, int, int]]:
    """``(cell_a, cell_b, side)`` for every interior sub-edge of one face.

    ``side`` names the parent side the sub-edge is parallel to, which is what sets its length and
    therefore its conductance. Each interior sub-edge is incident to exactly one **inverted**
    sub-triangle, and an inverted one has exactly three upright neighbours, so enumerating the
    inverted cells enumerates the edges once each and none twice.
    """
    level = int(patch.levels[face])
    base = int(patch.face_offsets[face])

    def local(i: int, j: int, inverted: int) -> int:
        return 2 * level * j - j * j + 2 * i + inverted

    links: list[tuple[int, int, int]] = []
    for j in range(level):
        for i in range(level - j - 1):
            down = base + local(i, j, 1)
            # The sub-edge shared with the upright at (i, j) runs parallel to side 2 (v0-v1);
            # with (i+1, j) parallel to side 0 (v1-v2); with (i, j+1) parallel to side 1 (v2-v0).
            links.append((base + local(i, j, 0), down, 2))
            links.append((down, base + local(i + 1, j, 0), 0))
            links.append((down, base + local(i, j + 1, 0), 1))
    return links


def edge_spans(patch: Any, face: int, side: int) -> list[tuple[int, float, float]]:
    """``(cell, s_lo, s_hi)`` for the cells along one side of one face.

    ``s`` runs 0 → 1 from the first to the second vertex of that side, the sides being
    ``(v1, v2)``, ``(v2, v0)`` and ``(v0, v1)`` for ``side`` 0, 1 and 2 -- each opposite the
    vertex of the same index. The cells along a side are always the upright ones, ``level`` of
    them, each covering exactly one ``1/level`` of it.
    """
    level = int(patch.levels[face])
    base = int(patch.face_offsets[face])

    def local(i: int, j: int) -> int:
        return 2 * level * j - j * j + 2 * i

    out: list[tuple[int, float, float]] = []
    for m in range(level):
        lo, hi = m / level, (m + 1) / level
        if side == 0:  # w0 = 0, from v1 (w1 = 1) to v2 (w1 = 0): cell (0, j) holds w1 in [j, j+1]/k
            out.append((base + local(0, level - 1 - m), lo, hi))
        elif side == 1:  # w1 = 0, from v2 (w0 = 0) to v0 (w0 = 1)
            out.append((base + local(m, 0), lo, hi))
        else:  # w2 = 0, from v0 (w0 = 1) to v1 (w0 = 0)
            out.append((base + local(level - 1 - m, m), lo, hi))
    return out


def circumcentric_conductance(patch: Any, face: int, side: int) -> float:
    """``tan Θ`` for the parent angle opposite ``side``: the consistent weight, per ``k δ``.

    The finite-volume scheme that *is* consistent puts its node at the circumcentre, and its
    weight for an edge shared by two triangles is ``2 / (cot θ_a + cot θ_b)``. Inside one face the
    two sub-triangles across a sub-edge are congruent, so both angles are the parent's own and
    this collapses to ``tan Θ``. **Negative for an obtuse face** and singular for a right-angled
    one; exported so the tests can show what the shipped weight is chosen over rather than assert
    it. Not used by :func:`mesh_lateral_operator`.
    """
    v = np.asarray(patch.vertices_m)[np.asarray(patch.faces)[face]]
    other = [(side + 1) % 3, (side + 2) % 3]
    a = v[other[0]] - v[side]
    b = v[other[1]] - v[side]
    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    return float(np.tan(np.arccos(np.clip(cos, -1.0, 1.0))))


def mesh_lateral_operator(
    patch: Any, conductivity_w_mk: float, thickness_m: float
) -> ConductionOperator | None:
    """Conduction between a mesh's cells, within each face and across every shared mesh edge.

    Returns ``None`` for ``k δ = 0`` -- the operator-free field, bit for bit -- rather than a
    matrix of zeros, as :func:`~irsim.thermal.conduction.lateral_operator` does for a patch.

    Faces of **different levels** may meet: the cells along a shared mesh edge are matched by the
    length they actually overlap, in the same way `TC.3`'s contactors join two patches whose grids
    do not line up. A mesh edge with only one face (an open boundary) conducts to nothing, and one
    with more than two is skipped: heat sharing between three sheets at a seam is not a two-point
    flux and guessing at it would be worse than leaving it out.
    """
    if conductivity_w_mk < 0.0 or thickness_m < 0.0:
        raise ValueError("conductivity and thickness cannot be negative")
    k_delta = float(conductivity_w_mk) * float(thickness_m)
    if k_delta == 0.0:
        return None

    faces = np.asarray(patch.faces, dtype=np.intp)
    lengths, areas = _face_edges(np.asarray(patch.vertices_m, dtype=np.float64), faces)
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []

    # -- within each face ---------------------------------------------------------------------
    # G = (3/4) k δ |E_side|² / A. Level-free: the barycentric cut is affine, so refining a face
    # multiplies the edge count and divides w and d alike, and the rate heat crosses the face is
    # unchanged -- which is what a discretisation of a continuum has to do.
    for face in range(patch.n_faces):
        scale = 0.75 * k_delta / float(areas[face])
        for a, b, side in within_face_links(patch, face):
            rows.append(a)
            cols.append(b)
            vals.append(scale * float(lengths[face, side]) ** 2)

    # -- across the mesh's own edges ------------------------------------------------------------
    shared: dict[tuple[int, int], list[tuple[int, int, bool]]] = defaultdict(list)
    for face in range(patch.n_faces):
        for side in range(3):
            p = int(faces[face, (side + 1) % 3])
            q = int(faces[face, (side + 2) % 3])
            shared[(min(p, q), max(p, q))].append((face, side, p > q))

    for (p, q), incident in shared.items():
        if len(incident) != 2:
            continue
        length = float(np.linalg.norm(patch.vertices_m[q] - patch.vertices_m[p]))
        spans = []
        depths = []
        for face, side, flipped in incident:
            cells = edge_spans(patch, face, side)
            if flipped:  # this face walks the edge from q to p, so its parameter runs backwards
                cells = [(c, 1.0 - hi, 1.0 - lo) for c, lo, hi in cells]
            spans.append(cells)
            # A boundary cell's centroid sits a third of its own height from the mesh edge, and
            # that height is the face's height for this side divided by the level.
            level = int(patch.levels[face])
            depths.append(2.0 * float(areas[face]) / (3.0 * level * length))
        for cell_a, lo_a, hi_a in spans[0]:
            for cell_b, lo_b, hi_b in spans[1]:
                overlap = min(hi_a, hi_b) - max(lo_a, lo_b)
                if overlap <= 0.0:
                    continue
                rows.append(cell_a)
                cols.append(cell_b)
                vals.append(k_delta * overlap * length / (depths[0] + depths[1]))

    n = patch.n_cells
    if not rows:
        return None
    upper = sp.coo_matrix((np.asarray(vals), (rows, cols)), shape=(n, n))
    return ConductionOperator(upper + upper.T, patch.cell_area_m2)
