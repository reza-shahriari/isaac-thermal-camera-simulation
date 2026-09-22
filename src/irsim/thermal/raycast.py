"""Occlusion from geometry, and the sun's finite disc (PT.22).

docs/physics-model.md §5.2, §6.1; ADR 0095 (occluders in the scene), ADR 0107 (this module).

PT.18 shadowed the beam with authored rectangles and PT.21 the sky with the same ones. Two
things a rectangle cannot do, and this module can:

* **Geometry that is not a rectangle, and geometry that is not the surface's own.** An
  ``(origins, directions) -> hit`` adapter puts the occlusion test behind one call, so the same
  balance runs against authored rectangles, against a triangle soup built from a prim's points,
  or against both at once. A *neighbour's* mesh shades a wall that its own mesh never can, which
  is the whole content of "the query walks every opaque prim": self-shadowing is what a surface's
  own normal already gives, and the interesting shadow comes from something else in the scene.
* **A shadow edge that is a ramp, not a step.** The sun is a disc about 0.53 deg across, so an
  edge standing ``d`` above a surface throws a penumbra ``d tan(0.53 deg)`` wide -- 9.3 mm per
  metre of standoff. Sampling the disc with 7 or 19 rays turns the binary visibility into a
  **sunlit fraction**, which is what multiplies the direct beam. Inside a 25 cm cell at a 2 m
  standoff the ramp is sub-cell and invisible; on a car's panel gaps, a louvre, or a wall a hand's
  breadth from its neighbour, it is the edge.

**The rectangles stay the oracle.** :class:`RectangleOccluders` is `cell_shadow`'s exact
ray-rectangle test generalised to per-ray origins *and* per-ray directions, and every mesh
result in the tests is held against it. :class:`TriangleSoup` is Moeller-Trumbore in NumPy, with
an axis-aligned box reject per soup -- no acceleration structure and no dependency, which is
right for the tens-to-thousands of triangles a scene's occluders amount to. A BVH backend
(trimesh with embreex, or Warp on the GPU) drops in behind the same protocol when a scene needs
one, and must live outside ``src/irsim`` because it would add a dependency (CLAUDE.md #1).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from irsim.radiometry.constants import SOLAR_DISC_DIAMETER_DEG
from irsim.thermal.shadow import ShadowRectangle
from irsim.thermal.surface_field import PlanarPatch

__all__ = [
    "AnyOccluders",
    "MeshOccluders",
    "Occluders",
    "RectangleOccluders",
    "TriangleSoup",
    "box_mesh",
    "cylinder_mesh",
    "disc_visibility",
    "penumbra_width_m",
    "rectangle_mesh",
    "solar_disc_rays",
    "sphere_mesh",
    "sunlit_fraction",
]

#: Rays that start exactly on a surface would hit it at t = 0; this is how far along a ray a hit
#: has to be to count, in metres. A millimetre of a shadow is a millimetre nobody can see.
HIT_EPS_M = 1e-6

#: Rays per chunk in the mesh test: the intermediate is (rays x triangles x 3) float64.
_CHUNK = 4096


@runtime_checkable
class Occluders(Protocol):
    """Anything that can answer "does this ray hit something?" for a batch of rays."""

    def blocked(self, origins: Any, directions: Any, max_distance_m: float = math.inf) -> Any:
        """``(n,)`` bool: True where the ray from ``origins[i]`` along ``directions[i]`` is
        stopped before ``max_distance_m``. Directions need not be unit length; ``max_distance_m``
        is measured in units of the direction vector when it is not."""
        ...


@dataclass(frozen=True)
class RectangleOccluders:
    """The oracle: `cell_shadow`'s exact ray-rectangle test, per-ray origin and direction."""

    rectangles: tuple[ShadowRectangle, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "rectangles", tuple(self.rectangles))

    def blocked(
        self, origins: Any, directions: Any, max_distance_m: float = math.inf
    ) -> NDArray[np.bool_]:
        o = np.atleast_2d(np.asarray(origins, dtype=np.float64))
        d = np.broadcast_to(np.atleast_2d(np.asarray(directions, dtype=np.float64)), o.shape)
        hit = np.zeros(o.shape[0], dtype=bool)
        for rect in self.rectangles:
            denominator = d @ rect.normal
            # A ray running along the occluder's plane can graze it but not be stopped by it.
            parallel = np.abs(denominator) < 1e-12
            safe = np.where(parallel, 1.0, denominator)
            t = ((rect.centre_m - o) @ rect.normal) / safe
            inside_range = (~parallel) & (t > HIT_EPS_M) & (t < max_distance_m)
            offset = o + t[:, None] * d - rect.centre_m
            inside = (np.abs(offset @ rect.u_axis) <= rect.half_u_m) & (
                np.abs(offset @ rect.v_axis) <= rect.half_v_m
            )
            hit |= inside_range & inside
        return hit


@dataclass(frozen=True)
class TriangleSoup:
    """A prim's triangles in the patch's frame: ``vertices (n, 3)`` and ``faces (m, 3)``."""

    vertices: NDArray[np.float64]
    faces: NDArray[np.int64]
    _lo: NDArray[np.float64] = field(init=False, repr=False)
    _hi: NDArray[np.float64] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        v = np.asarray(self.vertices, dtype=np.float64)
        f = np.asarray(self.faces, dtype=np.int64)
        if v.ndim != 2 or v.shape[1] != 3 or v.shape[0] < 3:
            raise ValueError(f"vertices must be (n >= 3, 3), got {v.shape}")
        if f.ndim != 2 or f.shape[1] != 3 or f.shape[0] < 1:
            raise ValueError(f"faces must be (m >= 1, 3), got {f.shape}")
        if f.min() < 0 or f.max() >= v.shape[0]:
            raise ValueError("a face indexes a vertex that does not exist")
        object.__setattr__(self, "vertices", v)
        object.__setattr__(self, "faces", f)
        object.__setattr__(self, "_lo", v.min(axis=0))
        object.__setattr__(self, "_hi", v.max(axis=0))

    @property
    def n_faces(self) -> int:
        return int(self.faces.shape[0])

    @property
    def bounds_m(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        return self._lo, self._hi

    def _slab_reject(self, o: NDArray[np.float64], d: NDArray[np.float64], far: float) -> Any:
        """``(n,)`` bool: rays whose segment cannot reach this soup's box at all."""
        with np.errstate(divide="ignore", invalid="ignore"):
            inv = 1.0 / d
            t0 = (self._lo - o) * inv
            t1 = (self._hi - o) * inv
        near = np.nanmax(np.minimum(t0, t1), axis=1)
        far_hit = np.nanmin(np.maximum(t0, t1), axis=1)
        return ~((far_hit >= np.maximum(near, HIT_EPS_M)) & (near < far))

    def blocked(
        self, origins: Any, directions: Any, max_distance_m: float = math.inf
    ) -> NDArray[np.bool_]:
        o_all = np.atleast_2d(np.asarray(origins, dtype=np.float64))
        d_all = np.broadcast_to(
            np.atleast_2d(np.asarray(directions, dtype=np.float64)), o_all.shape
        )
        hit = np.zeros(o_all.shape[0], dtype=bool)
        far = float(max_distance_m)
        v0 = self.vertices[self.faces[:, 0]]
        e1 = self.vertices[self.faces[:, 1]] - v0
        e2 = self.vertices[self.faces[:, 2]] - v0
        for start in range(0, o_all.shape[0], _CHUNK):
            stop = min(start + _CHUNK, o_all.shape[0])
            o, d = o_all[start:stop], d_all[start:stop]
            live = ~self._slab_reject(o, d, far)
            if not live.any():
                continue
            o, d = o[live], d[live]
            # Moeller-Trumbore, every ray against every triangle.
            pvec = np.cross(d[:, None, :], e2[None, :, :])
            det = np.einsum("mj,nmj->nm", e1, pvec)
            parallel = np.abs(det) < 1e-12
            inv_det = np.where(parallel, 0.0, 1.0 / np.where(parallel, 1.0, det))
            tvec = o[:, None, :] - v0[None, :, :]
            u = np.einsum("nmj,nmj->nm", tvec, pvec) * inv_det
            qvec = np.cross(tvec, e1[None, :, :])
            v = np.einsum("nj,nmj->nm", d, qvec) * inv_det
            t = np.einsum("mj,nmj->nm", e2, qvec) * inv_det
            ok = (
                (~parallel) & (u >= 0.0) & (v >= 0.0) & (u + v <= 1.0) & (t > HIT_EPS_M) & (t < far)
            )
            found = np.zeros(live.shape[0], dtype=bool)
            found[live] = ok.any(axis=1)
            hit[start:stop] = found
        return hit


@dataclass(frozen=True)
class MeshOccluders:
    """Every opaque prim in the scene, as triangle soups: a ray is blocked if any one stops it."""

    meshes: tuple[TriangleSoup, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "meshes", tuple(self.meshes))

    def blocked(
        self, origins: Any, directions: Any, max_distance_m: float = math.inf
    ) -> NDArray[np.bool_]:
        o = np.atleast_2d(np.asarray(origins, dtype=np.float64))
        hit = np.zeros(o.shape[0], dtype=bool)
        for mesh in self.meshes:
            remaining = ~hit
            if not remaining.any():
                break
            found = mesh.blocked(o[remaining], np.atleast_2d(directions), max_distance_m)
            hit[remaining] = found
        return hit


@dataclass(frozen=True)
class AnyOccluders:
    """Several occluders of any kind behind one query -- rectangles beside meshes."""

    parts: tuple[Any, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "parts", tuple(self.parts))

    def blocked(
        self, origins: Any, directions: Any, max_distance_m: float = math.inf
    ) -> NDArray[np.bool_]:
        o = np.atleast_2d(np.asarray(origins, dtype=np.float64))
        hit = np.zeros(o.shape[0], dtype=bool)
        for part in self.parts:
            hit |= np.asarray(part.blocked(o, directions, max_distance_m), dtype=bool)
        return hit


# --- geometry ----------------------------------------------------------------------------------


def rectangle_mesh(rect: ShadowRectangle) -> TriangleSoup:
    """The two triangles of a `ShadowRectangle`: the same surface, the other representation."""
    u = rect.u_axis * rect.half_u_m
    v = rect.v_axis * rect.half_v_m
    corners = np.stack(
        [rect.centre_m - u - v, rect.centre_m + u - v, rect.centre_m + u + v, rect.centre_m - u + v]
    )
    return TriangleSoup(corners, np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64))


def box_mesh(centre_m: Any, size_m: Any) -> TriangleSoup:
    """The twelve triangles of an axis-aligned box: `shadow.box_faces` as one soup."""
    c = np.asarray(centre_m, dtype=np.float64).reshape(3)
    size = np.asarray(size_m, dtype=np.float64).reshape(3)
    if np.any(size <= 0.0):
        raise ValueError(f"a box needs three positive extents, got {tuple(size)}")
    signs = np.array([[i, j, k] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)], dtype=float)
    vertices = c + 0.5 * size * signs
    # Vertex n has bits (x, y, z) in the order the product above generates them.
    faces = np.array(
        [
            [0, 1, 3],
            [0, 3, 2],  # x = lo
            [4, 7, 5],
            [4, 6, 7],  # x = hi
            [0, 4, 5],
            [0, 5, 1],  # y = lo
            [2, 3, 7],
            [2, 7, 6],  # y = hi
            [0, 2, 6],
            [0, 6, 4],  # z = lo
            [1, 5, 7],
            [1, 7, 3],  # z = hi
        ],
        dtype=np.int64,
    )
    return TriangleSoup(vertices, faces)


def sphere_mesh(centre_m: Any, radius_m: float, n_theta: int = 16, n_phi: int = 32) -> TriangleSoup:
    """A UV sphere: the curved case, for an occluder or for a `WM.2` temperature field.

    A chord approximation, short of the true sphere by `radius (1 - cos(diagonal/2))` where the
    diagonal is the angular extent of one quad -- about 0.6 mm on a 0.25 m sphere at 32x64. Poles
    are single vertices, so the first and last rings are triangles and the rest are split quads.
    """
    c = np.asarray(centre_m, dtype=np.float64).reshape(3)
    if radius_m <= 0.0:
        raise ValueError(f"a sphere needs a positive radius, got {radius_m}")
    if n_theta < 2 or n_phi < 3:
        raise ValueError(f"a sphere needs n_theta >= 2 and n_phi >= 3, got {n_theta}, {n_phi}")
    theta = np.linspace(0.0, np.pi, n_theta + 1)
    phi = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)
    tt, pp = np.meshgrid(theta, phi, indexing="ij")
    vertices = c + radius_m * np.stack(
        [np.sin(tt) * np.cos(pp), np.sin(tt) * np.sin(pp), np.cos(tt)], axis=-1
    ).reshape(-1, 3)
    rows = np.arange(n_theta)[:, None]
    cols = np.arange(n_phi)[None, :]
    a = (rows * n_phi + cols).ravel()
    b = (rows * n_phi + (cols + 1) % n_phi).ravel()
    d = ((rows + 1) * n_phi + cols).ravel()
    e = ((rows + 1) * n_phi + (cols + 1) % n_phi).ravel()
    faces = np.concatenate([np.stack([a, d, e], axis=1), np.stack([a, e, b], axis=1)], axis=0)
    # Degenerate triangles at the two poles, where a whole ring collapses to one point.
    v = vertices[faces]
    keep = np.linalg.norm(np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]), axis=-1) > 1e-15
    return TriangleSoup(vertices, np.asarray(faces[keep], dtype=np.int64))


def cylinder_mesh(
    centre_m: Any,
    radius_m: float,
    length_m: float,
    axis: Any = (0.0, 0.0, 1.0),
    n_phi: int = 24,
    n_z: int = 8,
    *,
    capped: bool = True,
) -> TriangleSoup:
    """A cylinder about ``axis``, centred on ``centre_m``: a pipe, an arm, a mast, a motor bell.

    The shape ADR 0087 named as Hard, and the one `WM.2`'s field exists for -- its temperature
    varies around the circumference as well as along the length, which no single patch normal can
    express. ``capped`` closes the ends with a triangle fan; an open tube is the right model for a
    pipe seen from outside and a closed one for a bell.
    """
    c = np.asarray(centre_m, dtype=np.float64).reshape(3)
    if radius_m <= 0.0 or length_m <= 0.0:
        raise ValueError(f"a cylinder needs positive radius and length, got {radius_m}, {length_m}")
    if n_phi < 3 or n_z < 1:
        raise ValueError(f"a cylinder needs n_phi >= 3 and n_z >= 1, got {n_phi}, {n_z}")
    w = np.asarray(axis, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(w))
    if norm <= 0.0:
        raise ValueError("a cylinder's axis has zero length")
    w = w / norm
    # Any perpendicular pair will do; take the world axis least aligned with w so the cross
    # product is well conditioned rather than nearly zero.
    helper = np.eye(3)[int(np.argmin(np.abs(w)))]
    u = np.cross(w, helper)
    u /= np.linalg.norm(u)
    v = np.cross(w, u)

    phi = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)
    z = np.linspace(-0.5 * length_m, 0.5 * length_m, n_z + 1)
    zz, pp = np.meshgrid(z, phi, indexing="ij")
    ring = radius_m * (np.cos(pp)[..., None] * u + np.sin(pp)[..., None] * v)
    vertices = (c + zz[..., None] * w + ring).reshape(-1, 3)

    rows, cols = np.arange(n_z)[:, None], np.arange(n_phi)[None, :]
    a = (rows * n_phi + cols).ravel()
    b = (rows * n_phi + (cols + 1) % n_phi).ravel()
    d = ((rows + 1) * n_phi + cols).ravel()
    e = ((rows + 1) * n_phi + (cols + 1) % n_phi).ravel()
    # Wound so the side normals point **outward**. This is not cosmetic: a cell's solar term is
    # `max(0, n·s)`, so an inward normal makes the sunlit crown of a tube the cold side and the
    # shaded underside the hot one -- a picture that looks like physics and is upside down.
    faces = [np.stack([a, e, d], axis=1), np.stack([a, b, e], axis=1)]
    if capped:
        low = c - 0.5 * length_m * w
        high = c + 0.5 * length_m * w
        base = vertices.shape[0]
        vertices = np.concatenate([vertices, low[None, :], high[None, :]], axis=0)
        cols_flat = np.arange(n_phi)
        nxt = (cols_flat + 1) % n_phi
        faces.append(np.stack([np.full(n_phi, base), nxt, cols_flat], axis=1))
        top = n_z * n_phi
        faces.append(np.stack([np.full(n_phi, base + 1), top + cols_flat, top + nxt], axis=1))
    return TriangleSoup(vertices, np.concatenate(faces, axis=0).astype(np.int64))


# --- the sun's disc ------------------------------------------------------------------------------


def penumbra_width_m(standoff_m: float) -> float:
    """``d tan(0.53 deg)``: how wide the ramp is under an edge ``d`` above the surface."""
    return float(standoff_m) * math.tan(math.radians(SOLAR_DISC_DIAMETER_DEG))


def solar_disc_rays(
    direction: Any, n_rays: int = 7
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """``(directions (n, 3), weights (n,))`` sampling the sun's disc about ``direction``.

    Concentric rings at ``r_k = R k / K`` for ``K`` rings, with the **outermost ring exactly on
    the limb** -- that is what makes the sampled penumbra as wide as the geometric one rather
    than a fraction of it. 1 ray (the centre, which is PT.18's hard edge), 7 (K = 1), 19 (K = 2),
    37 (K = 3) and so on. Weights are the fraction of a uniform disc each ray stands for and sum
    to 1; limb darkening is not modelled, which narrows the ramp's shoulders and not its width.
    """
    s = np.asarray(direction, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(s))
    if norm <= 0.0:
        raise ValueError("direction has zero length")
    s = s / norm
    rings = _rings_for(n_rays)
    if rings == 0:
        return s.reshape(1, 3), np.ones(1)
    # An orthonormal basis across the beam; the branch keeps it well conditioned near the poles.
    helper = np.array([0.0, 0.0, 1.0]) if abs(s[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    a = np.cross(s, helper)
    a /= np.linalg.norm(a)
    b = np.cross(s, a)
    half_angle = math.radians(0.5 * SOLAR_DISC_DIAMETER_DEG)
    directions = [s]
    weights = [1.0 / (4.0 * rings * rings)]
    for k in range(1, rings + 1):
        radius = math.tan(half_angle) * k / rings
        share = (2.0 * k if k < rings else rings - 0.25) / (rings * rings)
        count = 6 * k
        for i in range(count):
            phi = 2.0 * math.pi * i / count
            ray = s + radius * (math.cos(phi) * a + math.sin(phi) * b)
            directions.append(ray / np.linalg.norm(ray))
            weights.append(share / count)
    return np.asarray(directions), np.asarray(weights)


def _rings_for(n_rays: int) -> int:
    """``n_rays = 1 + 3K(K+1)``: 1, 7, 19, 37, 61 ... rays for K = 0, 1, 2, 3, 4 rings."""
    k = 0
    while 1 + 3 * k * (k + 1) < n_rays:
        k += 1
    if 1 + 3 * k * (k + 1) != n_rays:
        allowed = [1 + 3 * j * (j + 1) for j in range(5)]
        raise ValueError(f"n_rays must be one of {allowed} (concentric rings), got {n_rays}")
    return k


def sunlit_fraction(
    patch: PlanarPatch,
    sun_direction: Any,
    occluders: Occluders | Sequence[ShadowRectangle],
    n_rays: int = 7,
    *,
    lift_m: float = 0.0,
) -> NDArray[np.float64]:
    """``(n_cells,)`` in [0, 1]: how much of the sun's disc each cell of a **planar patch** sees.

    The cell geometry half of :func:`disc_visibility` -- centres from the patch, lifted along its
    one normal. A mesh cell lifts along its own face's normal instead and calls the other
    function directly (`WM.4`).
    """
    centres = patch.cell_centres()
    if lift_m:
        centres = centres + lift_m * patch.normal
    return disc_visibility(centres, sun_direction, occluders, n_rays)


def disc_visibility(
    points: Any,
    sun_direction: Any,
    occluders: Occluders | Sequence[ShadowRectangle],
    n_rays: int = 7,
) -> NDArray[np.float64]:
    """``(n,)`` in [0, 1]: how much of the sun's disc each of ``points`` can see.

    ``occluders`` is anything with ``blocked`` (or a plain sequence of `ShadowRectangle`s, which
    is wrapped). ``n_rays = 1`` is `cell_shadow`'s hard edge exactly, to the bit. Like
    `cell_shadow`, a point on a surface facing away from the sun is returned as lit: ``max(0,
    n.s)`` in `solar_loading` is what zeroes the beam there, and duplicating it here would make a
    self-shadowing test look like an occlusion result.
    """
    query = occluders if hasattr(occluders, "blocked") else RectangleOccluders(tuple(occluders))
    directions, weights = solar_disc_rays(sun_direction, n_rays)
    centres = np.atleast_2d(np.asarray(points, dtype=np.float64))
    n = centres.shape[0]
    lit = np.ones(n)
    blocked_rays = np.zeros(n, dtype=np.int64)
    for ray, weight in zip(directions, weights, strict=True):
        blocked = np.asarray(query.blocked(centres, ray), dtype=bool)
        lit[blocked] -= weight
        blocked_rays += blocked
    # The ends are snapped rather than summed: the weights are floats, so a fully shaded cell
    # lands within 1e-15 of zero and a fully lit one within 1e-15 of one. Snapping keeps a cell
    # that nothing shades bit-identical to the per-prim solve (PT.17) whatever `n_rays` is.
    full = len(directions)
    lit = np.where(blocked_rays == 0, 1.0, np.where(blocked_rays == full, 0.0, lit))
    return np.clip(lit, 0.0, 1.0)
