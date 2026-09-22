"""What a mesh cell can actually see: sky view and shadow, traced against geometry (WM.4).

docs/physics-model.md §5.4 (the diffuse term and the shadow S), §6.1 (where V_s scales both the
diffuse solar and the longwave down); ADR 0088's own "revisit when", ADR 0104 (the dome),
ADR 0110 (the field this serves).

`WM.7` gave every mesh cell its own face normal, which fixed the **cosine**: the far side of a
tube is dark because it faces away, not because anything was traced. Two terms stayed analytic,
and on a curved or cluttered surface both are geometry:

* the **sky view**, taken as the tilt's isotropic ``V_s = (1 + n·up)/2`` -- which is the
  *unobstructed* answer, and a cell tucked under an airframe is not unobstructed; and
* the **beam**, gated by the surface's one ``shaded`` flag, so a cell that another part of the
  same mesh, or the body above it, hides still saw the whole sun.

Both are traced here through the `Occluders` protocol `PT.22` put behind the beam -- so a shadow
and a sky view cannot disagree about where the occluders are -- and on `PT.21`'s Tregenza
quadrature, unchanged and shared, so a mesh cell and a patch cell cannot disagree about what a
direction is worth either.

**A convex mesh is skipped, and that is exact rather than an optimisation.** Every ray leaving a
convex body's surface into its own outward hemisphere leaves and never returns, so for a convex
mesh with nothing else around it the traced factor *is* ``(1 + n·up)/2`` and the traced beam *is*
``max(0, n·s)``. :func:`is_convex` tests exactly that -- every vertex on or behind every face's
plane -- and the pipes, arms, masts and bells this lane was written for are convex, so they cost
nothing and keep the numbers `WM.7` measured, bit for bit.

**Per-face normals, still.** A smooth vertex normal is a better *shading* normal and a worse
*facet* normal, and §6.1 balances a facet. `WM.6`'s adjacency is what would change that.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from irsim.thermal.frames import ENU, WorldFrame
from irsim.thermal.raycast import (
    AnyOccluders,
    Occluders,
    RectangleOccluders,
    TriangleSoup,
    disc_visibility,
)
from irsim.thermal.shadow import ShadowRectangle
from irsim.thermal.skyview import sub_rays

__all__ = [
    "LIFT_M",
    "cell_occluders",
    "cell_origins",
    "is_convex",
    "mesh_soup",
    "mesh_sky_view",
    "mesh_sunlit_fraction",
]

#: How far off its own surface a cell's rays start, in metres. `TriangleSoup.blocked` counts a
#: hit past ``HIT_EPS_M`` = 1 µm; a cell's own face cannot stop a ray leaving its centroid (that
#: hit is at t = 0), but a face sharing an edge with it sits within float noise of the ray, and
#: acne on an edge would read as a shadow nothing casts. Ten times the hit epsilon, and three
#: orders below the 15 mm radius of the arms this was written for.
LIFT_M = 1.0e-5

#: Faces per chunk in the convexity test; the intermediate is (faces x vertices) float64.
_CONVEX_CHUNK = 256


def mesh_soup(patch: Any) -> TriangleSoup:
    """The patch's own triangles, as something that can stop a ray."""
    return TriangleSoup(patch.vertices_m, np.asarray(patch.faces, dtype=np.int64))


def is_convex(soup: TriangleSoup, tol_m: float | None = None) -> bool:
    """True when every vertex lies on or behind every face's plane.

    The test is the definition of a convex polyhedron with outward-wound faces, and it is what
    licenses skipping the trace: a ray leaving a convex body into the outward hemisphere of the
    face it left cannot come back. An **inward**-wound mesh fails this, which is the right
    answer -- its normals are wrong and nothing downstream should trust them.

    ``tol_m`` defaults to 1e-9 of the mesh's own extent, so it scales with the object rather
    than calling a 30 m hull non-convex for the float noise a 15 mm tube would not show.
    """
    v = soup.vertices
    lo, hi = soup.bounds_m
    extent = float(np.max(hi - lo))
    tol = 1e-9 * extent if tol_m is None else float(tol_m)
    v0 = v[soup.faces[:, 0]]
    e1 = v[soup.faces[:, 1]] - v0
    e2 = v[soup.faces[:, 2]] - v0
    normals = np.cross(e1, e2)
    normals /= np.linalg.norm(normals, axis=-1)[:, None]
    for start in range(0, soup.n_faces, _CONVEX_CHUNK):
        stop = min(start + _CONVEX_CHUNK, soup.n_faces)
        # Signed distance of every vertex from every face's plane, in this chunk.
        ahead = (v[None, :, :] - v0[start:stop, None, :]) * normals[start:stop, None, :]
        if float(np.max(ahead.sum(axis=-1))) > tol:
            return False
    return True


def cell_origins(patch: Any, lift_m: float = LIFT_M) -> NDArray[np.float64]:
    """``(n_cells, 3)`` ray origins: cell centroids lifted along **their own** face normals."""
    return np.asarray(patch.cell_centres() + float(lift_m) * patch.cell_normal)


def cell_occluders(
    patch: Any,
    occluders: Occluders | Sequence[ShadowRectangle] = (),
    *,
    self_occluding: bool | None = None,
    soup: TriangleSoup | None = None,
) -> Occluders | None:
    """What a cell of ``patch`` has to trace against, or ``None`` when nothing can stop a ray.

    ``self_occluding`` defaults to "trace the mesh against itself unless it is convex", which is
    exact (see the module docstring). Pass ``True`` to force the trace -- useful as a test's
    control, since a convex mesh must give the same answer either way -- or ``False`` to state
    that only the scene's occluders matter.
    """
    parts: list[Any] = []
    if occluders is not None and (hasattr(occluders, "blocked") or len(tuple(occluders))):
        parts.append(
            occluders if hasattr(occluders, "blocked") else RectangleOccluders(tuple(occluders))
        )
    mesh = mesh_soup(patch) if soup is None else soup
    trace_self = (not is_convex(mesh)) if self_occluding is None else bool(self_occluding)
    if trace_self:
        parts.append(mesh)
    if not parts:
        return None
    return parts[0] if len(parts) == 1 else AnyOccluders(tuple(parts))


def mesh_sky_view(
    patch: Any,
    occluders: Occluders | Sequence[ShadowRectangle] = (),
    *,
    frame: WorldFrame = ENU,
    self_occluding: bool | None = None,
    lift_m: float = LIFT_M,
) -> NDArray[np.float64]:
    """``(n_cells,)`` sky view factors: each cell's own ``V_s``, less where the dome is hidden.

    The patch is in the scene's **world** frame and ``frame`` says how that relates to ENU; the
    dome is built in ENU and rotated, so a Y-up scene and a Z-up one trace the same rays.

        SVF_c = V_s(n_c) · Σ_i ω_i cos θ_ci vis_ci / Σ_i ω_i cos θ_ci

    with the open quadrature accumulated in the same order as the gated one, so an unobstructed
    cell's ratio is exactly 1.0 and its factor is exactly ``(1 + n·up)/2`` -- the number `WM.7`
    used, to the bit. Each cell has its own normal and therefore its own quadrature, which is the
    part a planar patch does not need.

    Costs ``n_cells × 1740`` rays once. Nothing here moves with time, so a scene traces it at
    build and the field carries the array.
    """
    normals_world = np.asarray(patch.cell_normal, dtype=np.float64)
    enu = np.asarray(frame.to_enu(normals_world), dtype=np.float64)
    analytic = np.clip(0.5 * (1.0 + enu[:, 2]), 0.0, 1.0)
    soup = mesh_soup(patch)
    query = cell_occluders(patch, occluders, self_occluding=self_occluding, soup=soup)
    if query is None:
        return np.asarray(analytic)

    directions_enu, omegas, _ = sub_rays()
    directions = np.asarray(frame.to_world(directions_enu), dtype=np.float64)
    origins = cell_origins(patch, lift_m)
    cosines = normals_world @ directions.T
    seen = np.zeros(patch.n_cells)
    total = np.zeros(patch.n_cells)
    for j in range(directions.shape[0]):
        above = cosines[:, j] > 0.0
        if not above.any():
            continue
        weight = omegas[j] * cosines[above, j]
        total[above] += weight
        blocked = np.asarray(query.blocked(origins[above], directions[j]), dtype=bool)
        seen[above] += weight * ~blocked
    out = np.zeros(patch.n_cells)
    open_sky = total > 0.0  # a cell whose whole hemisphere is below the horizon sees no sky
    out[open_sky] = analytic[open_sky] * (seen[open_sky] / total[open_sky])
    return out


def mesh_sunlit_fraction(
    patch: Any,
    sun_direction_world: Any,
    occluders: Occluders | Sequence[ShadowRectangle] = (),
    n_rays: int = 1,
    *,
    self_occluding: bool | None = None,
    lift_m: float = LIFT_M,
) -> NDArray[np.float64]:
    """``(n_cells,)`` in [0, 1]: how much of the sun's disc each cell sees.

    ``n_rays = 1`` is the hard edge; 7, 19 or 37 make the terminator a ramp `d·tan(0.53°)` wide
    (`PT.22`). A cell facing away from the sun comes back lit, as everywhere else in this
    package: ``max(0, n·s)`` in `solar_loading` is what zeroes its beam, and doing it twice would
    make a cosine look like an occlusion.
    """
    query = cell_occluders(patch, occluders, self_occluding=self_occluding)
    if query is None:
        return np.ones(patch.n_cells)
    return disc_visibility(cell_origins(patch, lift_m), sun_direction_world, query, n_rays)
