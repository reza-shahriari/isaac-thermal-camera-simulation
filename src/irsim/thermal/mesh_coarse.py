"""Coarse thermal cells on an over-tessellated mesh (TC.13, ADR 0165).

docs/physics-model.md §6.3, §12.3. :class:`~irsim.thermal.mesh_field.TriangleMeshPatch` puts at
least one cell on every face, which is right for a mesh sized to the physics and wrong for a
product-visualisation asset whose faces are far smaller than any thermal gradient: the Phantom
4's shells carry 180-220 k faces each and its motor bells 160 k, so a full solve of the airframe
would be 1.5 million cells for a body that a few thousand describe. ADR 0143 lumped those parts
at ``T_air + offset`` for exactly this reason.

This module is the other way round: faces are **clustered** into cells of about ``cell_m``. A
face joins the cluster of the voxel its centroid falls in, split by the octant of its normal so
the inner and outer skins of a thin shell, which share voxels, never share a cell. A cell's area
is the exact sum of its faces' areas (nothing is decimated, no area moves), its centroid and
normal are the area-weighted means, and the closest-point lookup a pixel or a contact makes
resolves a face straight to its cluster, so the field, the bridge, the contactors and the
forcing all take a :class:`CoarseMeshPatch` where they take a fine one.

Conduction between clusters is a finite-volume link across every mesh edge shared by faces of
two different clusters: ``k δ · L_edge / d`` with ``d`` the distance between the two cluster
centroids (floored at half a cell, since two centroids can sit closer than the cells they
represent). Within a cluster the temperature is one value, which is the approximation: a
gradient finer than ``cell_m`` cannot be represented, and a caller who needs one lowers
``cell_m`` rather than asking this patch to invent it.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field as dataclass_field
from typing import Any

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

from irsim.thermal.conduction import ConductionOperator
from irsim.thermal.mesh_field import closest_point_on_mesh

__all__ = ["CoarseMeshPatch", "coarse_lateral_operator", "coarsen_mesh"]


@dataclass(frozen=True)
class CoarseMeshPatch:
    """A triangle mesh whose faces are grouped into thermal cells of about ``cell_m``.

    The geometry stays the mesh's own (``vertices_m``, ``faces``): only the *cells* are coarse.
    ``face_cluster`` maps every face to its cell.
    """

    vertices_m: NDArray[np.float64]
    faces: NDArray[np.intp]
    face_cluster: NDArray[np.intp]
    cell_m: float
    frame: str = "world"

    _n_cells: int = dataclass_field(init=False, repr=False)
    _face_area: NDArray[np.float64] = dataclass_field(init=False, repr=False)
    _face_normal: NDArray[np.float64] = dataclass_field(init=False, repr=False)
    _cell_area: NDArray[np.float64] = dataclass_field(init=False, repr=False)
    _cell_normal: NDArray[np.float64] = dataclass_field(init=False, repr=False)
    _cell_centre: NDArray[np.float64] = dataclass_field(init=False, repr=False)

    def __post_init__(self) -> None:
        v = np.asarray(self.vertices_m, dtype=np.float64)
        f = np.asarray(self.faces, dtype=np.intp)
        c = np.asarray(self.face_cluster, dtype=np.intp).reshape(-1)
        if v.ndim != 2 or v.shape[1] != 3 or v.shape[0] < 3:
            raise ValueError(f"vertices_m must be (n >= 3, 3), got {v.shape}")
        if f.ndim != 2 or f.shape[1] != 3 or f.shape[0] < 1:
            raise ValueError(f"faces must be (m >= 1, 3), got {f.shape}")
        if c.shape != (f.shape[0],):
            raise ValueError(f"face_cluster has shape {c.shape}, expected ({f.shape[0]},)")
        n = int(c.max()) + 1 if c.size else 0
        if c.min() < 0 or len(np.unique(c)) != n:
            raise ValueError("face_cluster must number the clusters densely from 0")
        if not self.cell_m > 0.0:
            raise ValueError("cell_m must be positive")
        a, b, cc = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
        cross = np.cross(b - a, cc - a)
        norm = np.linalg.norm(cross, axis=-1)
        if np.any(norm <= 0.0):
            raise ValueError(f"{int(np.sum(norm <= 0.0))} face(s) are degenerate (zero area)")
        face_area = 0.5 * norm
        face_normal = cross / norm[:, None]
        centroid = (a + b + cc) / 3.0
        cell_area = np.bincount(c, weights=face_area, minlength=n)
        cell_centre = (
            np.stack(
                [np.bincount(c, weights=face_area * centroid[:, k], minlength=n) for k in range(3)],
                axis=1,
            )
            / cell_area[:, None]
        )
        summed = np.stack(
            [np.bincount(c, weights=face_area * face_normal[:, k], minlength=n) for k in range(3)],
            axis=1,
        )
        length = np.linalg.norm(summed, axis=-1)
        # a cluster whose normals cancel (only possible across the octant split's boundary) takes
        # its largest face's normal rather than a zero vector
        largest = np.zeros(n, dtype=np.intp)
        order = np.argsort(face_area)
        largest[c[order]] = order
        fallback = face_normal[largest]
        unit = np.where(
            length[:, None] > 1e-12, summed / np.maximum(length, 1e-12)[:, None], fallback
        )
        object.__setattr__(self, "vertices_m", v)
        object.__setattr__(self, "faces", f)
        object.__setattr__(self, "face_cluster", c)
        object.__setattr__(self, "_n_cells", n)
        object.__setattr__(self, "_face_area", face_area)
        object.__setattr__(self, "_face_normal", face_normal)
        object.__setattr__(self, "_cell_area", cell_area)
        object.__setattr__(self, "_cell_normal", unit)
        object.__setattr__(self, "_cell_centre", cell_centre)

    # -- geometry (the TriangleMeshPatch surface the rest of the code reads) --------------------

    @property
    def n_faces(self) -> int:
        return int(self.faces.shape[0])

    @property
    def n_cells(self) -> int:
        return self._n_cells

    @property
    def levels(self) -> NDArray[np.intp]:
        """Every face carries one (coarse) cell: a level of 1, for code that asks."""
        return np.ones(self.n_faces, dtype=np.intp)

    @property
    def face_area_m2(self) -> NDArray[np.float64]:
        return self._face_area

    @property
    def face_normal(self) -> NDArray[np.float64]:
        return self._face_normal

    @property
    def cell_area_m2(self) -> NDArray[np.float64]:
        """``(n_cells,)``: the exact sum of each cluster's face areas."""
        return self._cell_area

    @property
    def area_m2(self) -> float:
        return float(self._face_area.sum())

    @property
    def cell_normal(self) -> NDArray[np.float64]:
        """``(n_cells, 3)`` area-weighted mean normals, unit length."""
        return self._cell_normal

    @property
    def cell_face(self) -> NDArray[np.intp]:
        raise NotImplementedError(
            "a CoarseMeshPatch's cells span many faces, so there is no one face per cell; the "
            "per-face object exchange (ADR 0090) needs a fine TriangleMeshPatch"
        )

    def cell_centres(self) -> NDArray[np.float64]:
        """``(n_cells, 3)`` area-weighted centroids of each cluster's faces."""
        return self._cell_centre

    def cells_per_face(self) -> NDArray[np.intp]:
        """``(n_faces,)``: the cell each face belongs to (``face_cluster`` under the patch name)."""
        return self.face_cluster

    # -- the lookup ---------------------------------------------------------------------------

    def cell_of(self, face: Any, w0: Any, w1: Any) -> NDArray[np.intp]:
        """``(...,)`` the cell of ``face``; the barycentric weights do not matter here."""
        f = np.asarray(face, dtype=np.intp)
        if np.any(f < 0) or np.any(f >= self.n_faces):
            raise IndexError(f"face index outside 0..{self.n_faces - 1}")
        return np.asarray(self.face_cluster[f], dtype=np.intp)

    def cell_of_uv(self, face: Any, u: Any, v: Any) -> NDArray[np.intp]:
        return self.cell_of(face, u, v)

    def locate(
        self, points: Any
    ) -> tuple[NDArray[np.intp], NDArray[np.float64], NDArray[np.float64]]:
        return closest_point_on_mesh(self.vertices_m, self.faces, points)

    def cell_at(self, points: Any) -> tuple[NDArray[np.intp], NDArray[np.float64]]:
        face, bary, distance = self.locate(points)
        return self.cell_of(face, bary[:, 0], bary[:, 1]), distance

    def sample(
        self, values: Any, face: Any, w0: Any, w1: Any, *, fill: float = float("nan")
    ) -> NDArray[np.float64]:
        flat = np.asarray(values, dtype=np.float64).reshape(-1)
        if flat.size != self.n_cells:
            raise ValueError(f"values has {flat.size} cells, mesh has {self.n_cells}")
        f = np.asarray(face, dtype=np.intp)
        hit = f >= 0
        cell = self.cell_of(np.where(hit, f, 0), w0, w1)
        return np.asarray(np.where(hit, flat[cell], fill))

    def face_mean(self, values: Any) -> NDArray[np.float64]:
        """``(n_faces,)``: a face reads its cluster's value."""
        flat = np.asarray(values, dtype=np.float64).reshape(-1)
        if flat.size != self.n_cells:
            raise ValueError(f"values has {flat.size} cells, mesh has {self.n_cells}")
        return np.asarray(flat[self.face_cluster])

    def area_weighted_mean(self, values: Any) -> float:
        flat = np.asarray(values, dtype=np.float64).reshape(-1)
        if flat.size != self.n_cells:
            raise ValueError(f"values has {flat.size} cells, mesh has {self.n_cells}")
        return float(np.sum(flat * self._cell_area) / np.sum(self._cell_area))


def coarsen_mesh(
    vertices_m: Any, faces: Any, cell_m: float, *, frame: str = "world"
) -> CoarseMeshPatch:
    """Cluster ``faces`` into cells of about ``cell_m``: by the voxel of each face's centroid and
    the octant of its normal (so the two skins of a thin shell stay apart)."""
    if cell_m <= 0.0:
        raise ValueError(f"cell_m must be positive, got {cell_m}")
    v = np.asarray(vertices_m, dtype=np.float64)
    f = np.asarray(faces, dtype=np.intp)
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    cross = np.cross(b - a, c - a)
    centroid = (a + b + c) / 3.0
    voxel = np.floor(centroid / float(cell_m)).astype(np.int64)
    octant = (cross >= 0.0).astype(np.int64)
    key = np.concatenate([voxel, octant], axis=1)
    _, cluster = np.unique(key, axis=0, return_inverse=True)
    return CoarseMeshPatch(
        v, f, np.asarray(cluster, dtype=np.intp).reshape(-1), float(cell_m), frame
    )


def coarse_lateral_operator(
    patch: CoarseMeshPatch, conductivity_w_mk: float, thickness_m: float
) -> ConductionOperator | None:
    """Conduction between a coarse patch's cells across every mesh edge two cells share.

    Each shared mesh edge between faces of different clusters contributes ``k δ L / d``, with
    ``L`` the edge length and ``d`` the distance between the two clusters' centroids (at least
    ``cell_m / 2``). Summed per pair, it is the finite-volume conductance of a sheet of thickness
    δ between two control volumes whose common boundary is those edges. Returns ``None`` for
    ``k δ = 0``, as :func:`~irsim.thermal.mesh_conduction.mesh_lateral_operator` does.
    """
    if conductivity_w_mk < 0.0 or thickness_m < 0.0:
        raise ValueError("conductivity and thickness cannot be negative")
    k_delta = float(conductivity_w_mk) * float(thickness_m)
    if k_delta == 0.0:
        return None
    f = patch.faces
    n_f = f.shape[0]
    # every directed edge of every face, keyed by its sorted vertex pair
    ends = np.stack([f[:, [1, 2]], f[:, [2, 0]], f[:, [0, 1]]], axis=1).reshape(-1, 2)
    owner = np.repeat(np.arange(n_f, dtype=np.intp), 3)
    key = np.sort(ends, axis=1)
    _, inverse, counts = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    inverse = inverse.reshape(-1)
    # edges shared by exactly two faces: pair them
    shared = counts[inverse] == 2
    order = np.argsort(inverse[shared], kind="stable")
    e_idx = inverse[shared][order]
    e_owner = owner[shared][order]
    e_ends = key[shared][order]
    first = e_idx[0::2] == e_idx[1::2]
    if not np.all(first):  # pragma: no cover - the count filter guarantees pairs
        raise AssertionError("edge pairing failed")
    fa, fb = e_owner[0::2], e_owner[1::2]
    ca, cb = patch.face_cluster[fa], patch.face_cluster[fb]
    across = ca != cb
    if not np.any(across):
        return None
    ca, cb = ca[across], cb[across]
    length = np.linalg.norm(
        patch.vertices_m[e_ends[0::2][across, 0]] - patch.vertices_m[e_ends[0::2][across, 1]],
        axis=-1,
    )
    centres = patch.cell_centres()
    d = np.linalg.norm(centres[ca] - centres[cb], axis=-1)
    d = np.maximum(d, 0.5 * patch.cell_m)
    g = k_delta * length / d
    n = patch.n_cells
    lo, hi = np.minimum(ca, cb), np.maximum(ca, cb)
    upper = sp.coo_matrix((g, (lo, hi)), shape=(n, n)).tocsr()
    return ConductionOperator(upper + upper.T, patch.cell_area_m2)
