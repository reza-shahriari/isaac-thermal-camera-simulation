"""A surface temperature field on a **triangle mesh**, for the geometry a plane cannot cover.

docs/physics-model.md §6.1 (the balance), §6.4 (the fixed tick); ADR 0087 and its WM.1 addendum;
roadmap WM.2.

:mod:`~irsim.thermal.surface_field` gives a near-planar surface a field by projecting the
per-pixel world position onto a rectangle. That covers a road, a bonnet, a roof and a deck, and
ADR 0087 recorded what it does not cover: a wheel, a tyre, an exhaust pipe, a mast or a fuselage
is not near-planar and still takes one temperature for the whole prim. In LWIR that is the
dominant modelling error for any of them larger than a few pixels.

`WM.1` measured the way out. The renderer cannot transport a surface parameterisation -- there is
no UV AOV and no per-triangle id on this build -- but it does not have to: a closest-point query
against the prim's own mesh **derives** a triangle and its barycentric coordinates from the world
position the renderer already carries, to 0.13 µm against a 3.4 mm position budget. This module is
the engine-free half of what that unlocks: the cells, their geometry and their solve. `WM.3` is
the bridge that feeds it a face and a barycentric pair per pixel.

**Cells per face, at a per-face resolution -- Ptex style.** A face at level ``k`` carries ``k²``
congruent sub-triangles cut by the barycentric grid, so resolution follows the thermal gradient
and not the tessellation: a flat door panel exported as two big triangles can carry a hundred
cells each, while a hundred small triangles around a wheel arch can carry one apiece. There is no
atlas, no unwrap and no seam, because the parameterisation is the mesh's own.

**What this module deliberately does not do.**

* **The sample is piecewise constant within a cell**, where `PlanarPatch.sample` is bilinear.
  Smoothing across a mesh needs edge adjacency -- which cell of the neighbouring face sits across
  this edge, and in which orientation -- and that is `WM.4`'s, together with the per-cell normals
  and the ray-traced sky view. Until then the control is the cell size, exactly as it is for the
  solve.
* **Normals are per face, not per vertex.** A cell's incidence comes from its own triangle, which
  is already the point: one shared patch normal is what a curved surface cannot have. A smooth
  (vertex-interpolated) normal would be a better *shading* normal and a worse *facet* normal, and
  §6.1 balances a facet.
* **Cells do not conduct to each other.** `PT.11`'s lateral operator is built on a rectangular
  grid's four-neighbour edges; the mesh equivalent is a cotan Laplacian, and `WM.6` exists because
  a naive one breaks the discrete maximum principle on an obtuse imported mesh.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from irsim.thermal.facets import FacetForcing, FacetProperties
from irsim.thermal.field import DEFAULT_TICK_S, ThermalField
from irsim.thermal.surface_field import DEFAULT_KEEP_TICKS

__all__ = [
    "TriangleMeshPatch",
    "TriangleMeshField",
    "DEFAULT_LEVEL",
    "closest_point_on_mesh",
]

#: Sub-triangles per edge on a face nobody gave a resolution for. One cell per face is the
#: per-prim behaviour restricted to a triangle, which is the honest default: a caller who has not
#: said where the gradient is does not get one invented for them.
DEFAULT_LEVEL = 1


def _subdivision(level: int) -> tuple[NDArray[np.intp], NDArray[np.intp], NDArray[np.intp]]:
    """``(i, j, inverted)`` for every sub-triangle of a face at ``level``, in cell order.

    Row ``j`` (towards vertex 1) holds ``k - j`` upright sub-triangles and ``k - j - 1`` inverted
    ones, interleaved, so the local index is ``2kj - j² + 2i + inverted`` and the face's cells run
    ``0 .. k² - 1``. Keeping generation and indexing in one place is deliberate: they are inverses
    of each other, and a disagreement between them would put a cell's temperature on a different
    part of the triangle without ever raising.
    """
    i_out: list[int] = []
    j_out: list[int] = []
    inverted: list[int] = []
    for j in range(level):
        upright = level - j
        for i in range(upright):
            i_out.append(i)
            j_out.append(j)
            inverted.append(0)
            if i < upright - 1:
                i_out.append(i)
                j_out.append(j)
                inverted.append(1)
    return (
        np.asarray(i_out, dtype=np.intp),
        np.asarray(j_out, dtype=np.intp),
        np.asarray(inverted, dtype=np.intp),
    )


def closest_point_on_mesh(
    vertices_m: Any, faces: Any, points: Any, *, chunk: int = 256
) -> tuple[NDArray[np.intp], NDArray[np.float64], NDArray[np.float64]]:
    """``(face, barycentric (P, 3), distance_m)`` of the closest point on a triangle soup.

    Ericson's region test over every triangle, chunked over the query points. Deliberately the
    dumbest correct implementation: `WM.3` is specified to hold Warp's `mesh_query_point_no_sign`
    to this, so it has to be obviously right rather than fast, and an oracle that shared an
    optimisation with the thing it certifies would certify the optimisation too.

    It is also the fallback when Warp is not installed, which is what keeps the mesh bridge
    runnable on the plain-CPython gate.
    """
    v = np.asarray(vertices_m, dtype=np.float64)
    f = np.asarray(faces, dtype=np.intp)
    q = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    ab, ac = b - a, c - a
    best_face = np.zeros(len(q), dtype=np.intp)
    best_point = np.zeros((len(q), 3), dtype=np.float64)
    best_d2 = np.full(len(q), np.inf)
    for start in range(0, len(q), chunk):
        p = q[start : start + chunk][:, None, :]
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
        cand = np.where(np.isfinite(interior), interior, a)
        # The edge and vertex regions, applied so the vertex cases win over the edges.
        cand = np.where(
            ((va <= 0.0) & ((d4 - d3) >= 0.0) & ((d5 - d6) >= 0.0))[..., None],
            b + (c - b) * t_bc,
            cand,
        )
        cand = np.where(((vb <= 0.0) & (d2 >= 0.0) & (d6 <= 0.0))[..., None], a + ac * t_ac, cand)
        cand = np.where(((vc <= 0.0) & (d1 >= 0.0) & (d3 <= 0.0))[..., None], a + ab * t_ab, cand)
        cand = np.where(((d6 >= 0.0) & (d5 <= d6))[..., None], c, cand)
        cand = np.where(((d3 >= 0.0) & (d4 <= d3))[..., None], b, cand)
        cand = np.where(((d1 <= 0.0) & (d2 <= 0.0))[..., None], a, cand)
        dist2 = ((cand - p) ** 2).sum(-1)
        pick = np.argmin(dist2, axis=1)
        rows = np.arange(len(pick))
        best_face[start : start + chunk] = pick
        best_point[start : start + chunk] = cand[rows, pick]
        best_d2[start : start + chunk] = dist2[rows, pick]
    return best_face, barycentric_of(v, f, best_face, best_point), np.sqrt(best_d2)


def barycentric_of(vertices_m: Any, faces: Any, face: Any, points: Any) -> NDArray[np.float64]:
    """``(P, 3)`` weights of each face's three vertices for a point lying in that face's plane."""
    v = np.asarray(vertices_m, dtype=np.float64)
    f = np.asarray(faces, dtype=np.intp)[np.asarray(face, dtype=np.intp)]
    p = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    e0, e1, e2 = b - a, c - a, p - a
    d00 = (e0 * e0).sum(-1)
    d01 = (e0 * e1).sum(-1)
    d11 = (e1 * e1).sum(-1)
    d20 = (e2 * e0).sum(-1)
    d21 = (e2 * e1).sum(-1)
    denom = d00 * d11 - d01 * d01
    with np.errstate(divide="ignore", invalid="ignore"):
        w1 = np.nan_to_num((d11 * d20 - d01 * d21) / denom)
        w2 = np.nan_to_num((d00 * d21 - d01 * d20) / denom)
    return np.stack([1.0 - w1 - w2, w1, w2], axis=-1)


@dataclass(frozen=True)
class TriangleMeshPatch:
    """A triangle mesh whose faces are subdivided into thermal cells, in one named frame.

    ``frame`` is a name and not a transform, as :class:`~irsim.thermal.surface_field.PlanarPatch`
    uses it: ``"world"`` for fixed geometry, a prim path for something that moves with its object.
    The caller puts query points -- here, faces and barycentric weights -- in the same frame.

    Barycentric weights are named ``(w0, w1, w2)`` for the three vertices of the face, and they
    sum to one. They are **not** Warp's ``(u, v)`` without translation: `WM.1` measured that
    Warp's pair weights vertices 0 and 1 with ``1 - u - v`` on vertex 2, so
    ``(w0, w1, w2) = (u, v, 1 - u - v)``. :meth:`cell_of_uv` does that conversion in one place so
    no caller has to remember it.
    """

    vertices_m: NDArray[np.float64]
    faces: NDArray[np.intp]
    levels: NDArray[np.intp]
    frame: str = "world"

    #: Derived, built once in __post_init__.
    _offsets: NDArray[np.intp] = dataclass_field(init=False, repr=False)
    _cell_face: NDArray[np.intp] = dataclass_field(init=False, repr=False)
    _cell_w: NDArray[np.float64] = dataclass_field(init=False, repr=False)
    _face_area: NDArray[np.float64] = dataclass_field(init=False, repr=False)
    _face_normal: NDArray[np.float64] = dataclass_field(init=False, repr=False)

    def __post_init__(self) -> None:
        v = np.asarray(self.vertices_m, dtype=np.float64)
        f = np.asarray(self.faces, dtype=np.intp)
        if v.ndim != 2 or v.shape[1] != 3 or v.shape[0] < 3:
            raise ValueError(f"vertices_m must be (n >= 3, 3), got {v.shape}")
        if f.ndim != 2 or f.shape[1] != 3 or f.shape[0] < 1:
            raise ValueError(f"faces must be (m >= 1, 3), got {f.shape}")
        if f.min() < 0 or f.max() >= v.shape[0]:
            raise ValueError("a face indexes a vertex that does not exist")
        levels = np.asarray(self.levels, dtype=np.intp).reshape(-1)
        if levels.size == 1:
            levels = np.full(f.shape[0], int(levels[0]), dtype=np.intp)
        if levels.shape != (f.shape[0],):
            raise ValueError(f"levels has shape {levels.shape}, expected ({f.shape[0]},)")
        if np.any(levels < 1):
            raise ValueError("every face needs a level of at least 1")
        object.__setattr__(self, "vertices_m", v)
        object.__setattr__(self, "faces", f)
        object.__setattr__(self, "levels", levels)

        counts = levels.astype(np.int64) ** 2
        offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.intp)
        object.__setattr__(self, "_offsets", offsets)

        a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
        cross = np.cross(b - a, c - a)
        norm = np.linalg.norm(cross, axis=-1)
        if np.any(norm <= 0.0):
            raise ValueError(f"{int(np.sum(norm <= 0.0))} face(s) are degenerate (zero area)")
        object.__setattr__(self, "_face_area", 0.5 * norm)
        object.__setattr__(self, "_face_normal", cross / norm[:, None])

        # Cell -> face, and each cell's centroid in its face's barycentric coordinates. Built by
        # level so a mesh of one resolution walks its subdivision table once.
        cell_face = np.repeat(np.arange(f.shape[0], dtype=np.intp), counts)
        cell_w = np.zeros((int(offsets[-1]), 3), dtype=np.float64)
        for level in np.unique(levels):
            i, j, inv = _subdivision(int(level))
            k = float(level)
            # Upright centroid is a third of a cell in from its corner, inverted two thirds.
            w0 = (i + np.where(inv == 1, 2.0 / 3.0, 1.0 / 3.0)) / k
            w1 = (j + np.where(inv == 1, 2.0 / 3.0, 1.0 / 3.0)) / k
            block = np.stack([w0, w1, 1.0 - w0 - w1], axis=-1)
            for face in np.flatnonzero(levels == level):
                cell_w[offsets[face] : offsets[face + 1]] = block
        object.__setattr__(self, "_cell_face", cell_face)
        object.__setattr__(self, "_cell_w", cell_w)

    # -- geometry ---------------------------------------------------------------------------

    @property
    def n_faces(self) -> int:
        return int(self.faces.shape[0])

    @property
    def n_cells(self) -> int:
        return int(self._offsets[-1])

    @property
    def cell_face(self) -> NDArray[np.intp]:
        """``(n_cells,)`` the face each cell belongs to."""
        return self._cell_face

    @property
    def face_offsets(self) -> NDArray[np.intp]:
        """``(n_faces + 1,)`` where each face's cells start."""
        return self._offsets

    @property
    def face_area_m2(self) -> NDArray[np.float64]:
        return self._face_area

    @property
    def face_normal(self) -> NDArray[np.float64]:
        """``(n_faces, 3)`` unit normals, by the right-hand rule on the winding."""
        return self._face_normal

    @property
    def cell_area_m2(self) -> NDArray[np.float64]:
        """``(n_cells,)``. Every sub-triangle of a face has the face's area over ``k²`` -- the
        barycentric cut is affine, so the pieces are congruent however skewed the triangle is."""
        return np.asarray(self._face_area[self._cell_face] / self.levels[self._cell_face] ** 2)

    @property
    def area_m2(self) -> float:
        return float(self._face_area.sum())

    @property
    def cell_normal(self) -> NDArray[np.float64]:
        """``(n_cells, 3)``: its own face's normal. The whole point -- one shared normal is what
        a curved surface cannot have, and it is what sets each cell's solar incidence."""
        return np.asarray(self._face_normal[self._cell_face])

    def cell_centres(self) -> NDArray[np.float64]:
        """``(n_cells, 3)`` cell centroids in the patch's frame."""
        v = self.vertices_m[self.faces[self._cell_face]]
        return np.asarray(np.einsum("cv,cvx->cx", self._cell_w, v))

    # -- the lookup ---------------------------------------------------------------------------

    def cell_of(self, face: Any, w0: Any, w1: Any) -> NDArray[np.intp]:
        """``(...,)`` cell index for a face and barycentric weights of vertices 0 and 1.

        A point on an edge or a vertex belongs to one of the cells that share it, chosen without
        raising: a closest-point query lands exactly on an edge often enough that refusing would
        be refusing a legitimate pixel.
        """
        f = np.asarray(face, dtype=np.intp)
        if np.any(f < 0) or np.any(f >= self.n_faces):
            raise IndexError(f"face index outside 0..{self.n_faces - 1}")
        k = self.levels[f].astype(np.float64)
        x = np.clip(np.asarray(w0, dtype=np.float64), 0.0, 1.0) * k
        y = np.clip(np.asarray(w1, dtype=np.float64), 0.0, 1.0) * k
        i = np.floor(x)
        j = np.floor(y)
        # Upright when the fractional parts stay inside the sub-triangle, inverted past its
        # hypotenuse -- which is the same test as `floor(w0 k) + floor(w1 k) + floor(w2 k)`
        # equalling k - 1 or k - 2, without needing the third weight.
        inverted = ((x - i) + (y - j)) > 1.0
        limit = k - 1.0 - inverted
        j = np.clip(j, 0.0, np.maximum(limit, 0.0))
        i = np.clip(i, 0.0, np.maximum(limit - j, 0.0))
        local = 2.0 * k * j - j * j + 2.0 * i + inverted
        return np.asarray(self._offsets[f] + local.astype(np.intp), dtype=np.intp)

    def cell_of_uv(self, face: Any, u: Any, v: Any) -> NDArray[np.intp]:
        """:meth:`cell_of` from **Warp's** pair, which weights vertices 0 and 1 (`WM.1`).

        The conversion lives here and nowhere else. `probe_warp_mesh.py` measured that reading
        Warp's (u, v) as the weights of vertices 1 and 2 -- the form a person writes down first --
        misses by 0.56 m on a 0.4 m box, silently, so it is not a convention to retype per caller.
        """
        return self.cell_of(face, u, v)

    def locate(
        self, points: Any
    ) -> tuple[NDArray[np.intp], NDArray[np.float64], NDArray[np.float64]]:
        """``(face, barycentric, distance_m)`` for arbitrary points, by brute force.

        The engine-free route to a cell: what `WM.3`'s Warp query does per pixel, done here in
        NumPy so the mesh field is usable -- and testable -- with no renderer and no Warp.
        """
        return closest_point_on_mesh(self.vertices_m, self.faces, points)

    def cell_at(self, points: Any) -> tuple[NDArray[np.intp], NDArray[np.float64]]:
        """``(cell, distance_m)`` for arbitrary points."""
        face, bary, distance = self.locate(points)
        return self.cell_of(face, bary[:, 0], bary[:, 1]), distance

    def sample(
        self, values: Any, face: Any, w0: Any, w1: Any, *, fill: float = float("nan")
    ) -> NDArray[np.float64]:
        """Piecewise-constant sample of a ``(n_cells,)`` array. ``face < 0`` takes ``fill``."""
        flat = np.asarray(values, dtype=np.float64).reshape(-1)
        if flat.size != self.n_cells:
            raise ValueError(f"values has {flat.size} cells, mesh has {self.n_cells}")
        f = np.asarray(face, dtype=np.intp)
        hit = f >= 0
        cell = self.cell_of(np.where(hit, f, 0), w0, w1)
        return np.asarray(np.where(hit, flat[cell], fill))

    def face_mean(self, values: Any) -> NDArray[np.float64]:
        """``(n_faces,)`` mean over each face's own cells. They are equal-area, so this is the
        area-weighted mean of the face."""
        flat = np.asarray(values, dtype=np.float64).reshape(-1)
        if flat.size != self.n_cells:
            raise ValueError(f"values has {flat.size} cells, mesh has {self.n_cells}")
        totals = np.add.reduceat(flat, self._offsets[:-1])
        return np.asarray(totals / (self.levels.astype(np.float64) ** 2))

    def area_weighted_mean(self, values: Any) -> float:
        """One number for the whole prim: what the per-prim path carried before this module."""
        flat = np.asarray(values, dtype=np.float64).reshape(-1)
        if flat.size != self.n_cells:
            raise ValueError(f"values has {flat.size} cells, mesh has {self.n_cells}")
        areas = self.cell_area_m2
        return float(np.sum(flat * areas) / np.sum(areas))

    # -- construction --------------------------------------------------------------------------

    @classmethod
    def uniform(
        cls, vertices_m: Any, faces: Any, level: int = DEFAULT_LEVEL, *, frame: str = "world"
    ) -> TriangleMeshPatch:
        """Every face at the same level."""
        f = np.asarray(faces, dtype=np.intp)
        return cls(
            np.asarray(vertices_m, dtype=np.float64),
            f,
            np.full(f.shape[0], int(level), dtype=np.intp),
            frame,
        )

    @classmethod
    def from_soup(cls, soup: Any, level: int = DEFAULT_LEVEL, *, frame: str = "world"):  # type: ignore[no-untyped-def]
        """From an :class:`~irsim.thermal.raycast.TriangleSoup` -- the same pair of arrays the
        occlusion path already builds, so one mesh can shade and be solved."""
        return cls.uniform(soup.vertices, soup.faces, level, frame=frame)

    @classmethod
    def by_cell_size(
        cls,
        vertices_m: Any,
        faces: Any,
        cell_m: float,
        *,
        max_level: int = 32,
        frame: str = "world",
    ) -> TriangleMeshPatch:
        """Ptex style: each face gets the level that puts its cells near ``cell_m`` across.

        A face of area ``A`` at level ``k`` has cells of area ``A/k²``, so ``k = sqrt(A)/cell_m``
        rounded up and at least 1. Big faces get many cells and small ones get few, which is what
        keeps a wheel arch's hundred small triangles from costing a hundred times a door panel.
        """
        if cell_m <= 0.0:
            raise ValueError(f"cell_m must be positive, got {cell_m}")
        v = np.asarray(vertices_m, dtype=np.float64)
        f = np.asarray(faces, dtype=np.intp)
        a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
        area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=-1)
        level = np.clip(np.ceil(np.sqrt(area) / float(cell_m)), 1, int(max_level))
        return cls(v, f, level.astype(np.intp), frame)


class TriangleMeshField:
    """A :class:`TriangleMeshPatch` solved on a fixed tick and queried per face.

    The twin of :class:`~irsim.thermal.surface_field.PlanarThermalField`, and thin for the same
    reason: the tick schedule, the between-tick interpolation and the rule that a query never
    advances the solve belong to :class:`~irsim.thermal.field.ThermalField` and are reused rather
    than restated. What this adds is the spatial half on a mesh instead of a rectangle.
    """

    def __init__(
        self,
        patch: TriangleMeshPatch,
        properties: FacetProperties,
        forcing_at: Callable[[float], FacetForcing],
        t0_s: float,
        initial_k: Any,
        tick_s: float = DEFAULT_TICK_S,
        *,
        keep_ticks: int | None = DEFAULT_KEEP_TICKS,
        on_tick: Callable[[float, NDArray[np.float64]], None] | None = None,
        conduction: Any = None,
        film_kg_m2: Any = None,
    ) -> None:
        if properties.n_facets != patch.n_cells:
            raise ValueError(
                f"properties describe {properties.n_facets} facets but the mesh has "
                f"{patch.n_cells} cells"
            )
        state = np.asarray(initial_k, dtype=np.float64)
        if state.ndim == 0:
            state = np.full(patch.n_cells, float(state))
        self.patch = patch
        self.field = ThermalField(
            properties,
            forcing_at,
            t0_s,
            state,
            tick_s,
            keep_ticks=keep_ticks,
            on_tick=on_tick,
            conduction=conduction,
            film_kg_m2=film_kg_m2,
        )

    # -- delegation --------------------------------------------------------------------------

    @property
    def t0_s(self) -> float:
        return self.field.t0_s

    @property
    def tick_s(self) -> float:
        return self.field.tick_s

    @property
    def latest_t_s(self) -> float:
        return self.field.latest_t_s

    @property
    def n_ticks(self) -> int:
        return self.field.n_ticks

    @property
    def n_held(self) -> int:
        return self.field.n_held

    def advance_to(self, t_s: float) -> None:
        """Produce ticks up to ``t_s``. The **only** method that changes anything."""
        self.field.advance_to(t_s)

    def state_hash(self) -> str:
        return self.field.state_hash()

    # -- the query ---------------------------------------------------------------------------

    def temperature_at(self, t_s: float) -> NDArray[np.float32]:
        """``(n_cells,)`` float32 cell temperatures."""
        return self.field.temperature_at(t_s)

    def face_temperature_k(self, t_s: float) -> NDArray[np.float64]:
        """``(n_faces,)`` each face's own mean -- the per-facet picture, one step coarser."""
        return self.patch.face_mean(np.asarray(self.temperature_at(t_s), dtype=np.float64))

    def mean_temperature_k(self, t_s: float) -> float:
        """The area-weighted mean over the whole mesh: the one number this field replaces."""
        return self.patch.area_weighted_mean(np.asarray(self.temperature_at(t_s), dtype=np.float64))

    def sample_at(
        self, t_s: float, face: Any, w0: Any, w1: Any, *, fill: float = float("nan")
    ) -> NDArray[np.float32]:
        """Temperature at (face, barycentric) points, float32 (CLAUDE.md #2)."""
        return np.asarray(
            self.patch.sample(self.temperature_at(t_s), face, w0, w1, fill=fill),
            dtype=np.float32,
        )

    def sample_at_uv(
        self, t_s: float, face: Any, u: Any, v: Any, *, fill: float = float("nan")
    ) -> NDArray[np.float32]:
        """:meth:`sample_at` from Warp's ``(u, v)`` pair (`WM.1`'s convention)."""
        return self.sample_at(t_s, face, u, v, fill=fill)

    def sample_at_points(
        self,
        t_s: float,
        points: Any,
        *,
        max_distance_m: float | None = None,
        fill: float = float("nan"),
    ) -> NDArray[np.float32]:
        """Temperature at arbitrary points, located by brute force in NumPy.

        ``max_distance_m`` rejects a point that is not really on this mesh: it takes ``fill``
        rather than the temperature of whatever happened to be nearest. Without it, a pixel that
        hit the road would quietly read the car's bonnet because the bonnet was the closest mesh
        to it.
        """
        cell, distance = self.patch.cell_at(points)
        values = np.asarray(self.temperature_at(t_s), dtype=np.float64)[cell]
        if max_distance_m is not None:
            values = np.where(distance <= float(max_distance_m), values, fill)
        return np.asarray(values, dtype=np.float32)

    def __repr__(self) -> str:  # pragma: no cover - diagnostics
        return (
            f"TriangleMeshField(frame={self.patch.frame!r}, {self.patch.n_faces} faces, "
            f"{self.patch.n_cells} cells, t={self.latest_t_s:.1f}s)"
        )
