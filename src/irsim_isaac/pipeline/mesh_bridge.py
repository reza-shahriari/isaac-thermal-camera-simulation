"""Per-pixel temperature from a mesh field, keyed by the position AOV (`WM.3`).

docs/physics-model.md §13.1, §13.3; ADR 0087 and its WM.1 addendum; roadmap WM.3.

:mod:`~irsim_isaac.pipeline.point_bridge` overwrites a prim's pixels from a **planar patch** by
projecting the world position onto a rectangle. This module does the same job for the geometry a
rectangle cannot cover -- a wheel, a tyre, an exhaust pipe, a mast, a fuselage -- by asking for the
closest point on the prim's own mesh instead of projecting onto a plane. `WM.1` measured the query
that makes it possible: on this build a closest-point query returns a triangle and its barycentric
coordinates from a world position to 0.13 µm, against ADR 0014's 3.4 mm position budget.

It is **additive**, exactly as the planar bridge is: it takes a finished plane and overwrites only
the pixels of prims that have a mesh field bound to them, so a prim with no binding keeps the
per-instance value it always had and every existing scene renders bit-identically.

**Warp is an accelerator here, not the authority.** The engine-free
:func:`~irsim.thermal.mesh_field.closest_point_on_mesh` is the oracle the tests hold Warp to, and
it is also the fallback when Warp is absent -- so this bridge runs on the plain-CPython gate and a
machine with no GPU produces the same frame, slower. `device` defaults to Warp's **CPU** device for
the same reason: this is a correctness path, and the workstation's other card belongs to somebody.

**A hit beyond the tolerance raises rather than snapping.** A pixel whose closest point on this
mesh is further away than the position AOV's own error did not land on this prim -- the id plane
and the position plane disagree, or the bound mesh is not the geometry that was rendered. Snapping
it to the nearest cell would paint a wheel with the temperature of whatever part of it happened to
be closest, which looks like physics.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from irsim.thermal.mesh_field import TriangleMeshField, closest_point_on_mesh
from irsim_isaac.pipeline.material_ids import BACKGROUND_INSTANCE_ID, labels_to_paths

# The frame rule (PT.5: a patch authored in a prim's own frame rides that prim) has one
# implementation, in the planar bridge, and is imported rather than restated -- two copies
# would be two places to get a moving prim's transform wrong.
from irsim_isaac.pipeline.point_bridge import _into_frame

__all__ = ["MeshBinding", "MeshPointBridge", "DEFAULT_MAX_DISTANCE_M", "warp_available"]

#: How far a pixel's closest point on the bound mesh may be before the bridge refuses it. ADR
#: 0014 measured the position AOV at 3.4 mm; a mesh is also a chord approximation of whatever it
#: represents, so the budget is doubled rather than set to the AOV's figure exactly. A pixel
#: beyond it is not a slightly noisy hit, it is a different surface.
DEFAULT_MAX_DISTANCE_M = 7.0e-3

_KERNEL: Any = None
_WARP: Any = None


def _build_kernel() -> tuple[Any, Any]:
    """Import Warp and compile the query kernel once. ``(None, reason)`` if it is unavailable."""
    global _KERNEL, _WARP
    if _KERNEL is not None:
        return _WARP, _KERNEL
    try:
        from irsim_isaac.env import ensure_warp_on_path

        ensure_warp_on_path()
        import warp as wp

        wp.init()
    except Exception as error:  # pragma: no cover - depends on the machine
        return None, f"{type(error).__name__}: {error}"

    @wp.kernel  # type: ignore[untyped-decorator]
    def query(  # type: ignore[no-untyped-def]
        mesh: wp.uint64,
        points: wp.array(dtype=wp.vec3),  # type: ignore[valid-type]
        max_dist: float,
        hit: wp.array(dtype=wp.int32),  # type: ignore[valid-type]
        face: wp.array(dtype=wp.int32),  # type: ignore[valid-type]
        bary: wp.array(dtype=wp.vec2),  # type: ignore[valid-type]
        evaluated: wp.array(dtype=wp.vec3),  # type: ignore[valid-type]
    ):
        i = wp.tid()
        result = wp.mesh_query_point_no_sign(mesh, points[i], max_dist)
        if result.result:
            hit[i] = 1
            face[i] = result.face
            # WM.1 measured the convention: these weight vertices 0 and 1, with 1 - u - v on 2.
            bary[i] = wp.vec2(result.u, result.v)
            evaluated[i] = wp.mesh_eval_position(mesh, result.face, result.u, result.v)

    _WARP, _KERNEL = wp, query
    return wp, query


def warp_available() -> bool:
    """True when the Warp path can be used. The NumPy path runs either way."""
    return _build_kernel()[0] is not None


@dataclass(frozen=True)
class MeshBinding:
    """One prim's mesh temperature field.

    ``field.patch.frame`` is a name this module **checks** the same way the planar bridge does: a
    mesh authored in a prim's local frame cannot be queried with world positions, and the two are
    indistinguishable from the array shapes alone.
    """

    prim_path: str
    field: TriangleMeshField

    def __post_init__(self) -> None:
        if not self.prim_path:
            raise ValueError("a binding needs a prim path")


class MeshPointBridge:
    """Overwrites the mesh-backed prims of a per-instance temperature plane with their fields."""

    def __init__(
        self,
        bindings: Sequence[MeshBinding],
        known_paths: Sequence[str] | None = None,
        *,
        device: str = "cpu",
        max_distance_m: float = DEFAULT_MAX_DISTANCE_M,
        use_warp: bool = True,
    ) -> None:
        if max_distance_m <= 0.0:
            raise ValueError("max_distance_m must be positive")
        self.bindings = tuple(bindings)
        self.device = device
        self.max_distance_m = float(max_distance_m)
        self.use_warp = bool(use_warp)
        self._by_path: dict[str, list[TriangleMeshField]] = {}
        for binding in self.bindings:
            self._by_path.setdefault(binding.prim_path, []).append(binding.field)
        #: Pixels each bound prim path took in the last :meth:`apply`, as `PT.19` records them.
        self.last_coverage: dict[str, int] = dict.fromkeys(self._by_path, 0)
        self.known_paths: frozenset[str] | None = (
            None if known_paths is None else frozenset(known_paths)
        )
        if self.known_paths is not None:
            unknown = sorted(p for p in self._by_path if p not in self.known_paths)
            if unknown:
                raise ValueError(
                    f"bound prim path(s) {unknown} are not among the stage's prims "
                    f"{sorted(self.known_paths)}; check the `prim_path` of each mesh field -- a "
                    "binding nothing consumes would leave its prim at the per-instance fallback"
                )
        self._meshes: dict[int, Any] = {}

    @property
    def prim_paths(self) -> tuple[str, ...]:
        return tuple(self._by_path)

    @property
    def local_frames(self) -> tuple[str, ...]:
        """The non-world frames this bridge needs a ``world_from_local`` matrix for."""
        return tuple(
            sorted(
                {
                    field.patch.frame
                    for fields in self._by_path.values()
                    for field in fields
                    if field.patch.frame != "world"
                }
            )
        )

    def advance_to(self, t_s: float) -> None:
        """Push every bound field to ``t_s``. The only method that changes anything."""
        for fields in self._by_path.values():
            for field in fields:
                field.advance_to(t_s)

    # -- the query -----------------------------------------------------------------------------

    def locate(
        self, field: TriangleMeshField, points: NDArray[np.float64]
    ) -> tuple[NDArray[np.intp], NDArray[np.float64], NDArray[np.float64]]:
        """``(face, barycentric, distance_m)``, through Warp where it is available.

        The two routes are held to agreement by `tests/unit/test_mesh_bridge.py`; this method is
        the only place that chooses between them, so a caller cannot get one by accident.
        """
        if not self.use_warp or len(points) == 0:
            return closest_point_on_mesh(field.patch.vertices_m, field.patch.faces, points)
        wp, kernel = _build_kernel()
        if wp is None:
            return closest_point_on_mesh(field.patch.vertices_m, field.patch.faces, points)

        key = id(field.patch)
        mesh = self._meshes.get(key)
        if mesh is None:
            mesh = wp.Mesh(
                points=wp.array(
                    np.asarray(field.patch.vertices_m, dtype=np.float32),
                    dtype=wp.vec3,
                    device=self.device,
                ),
                indices=wp.array(
                    np.asarray(field.patch.faces, dtype=np.int32).flatten(),
                    dtype=wp.int32,
                    device=self.device,
                ),
            )
            self._meshes[key] = mesh

        n = len(points)
        hit = wp.zeros(n, dtype=wp.int32, device=self.device)
        face = wp.zeros(n, dtype=wp.int32, device=self.device)
        bary = wp.zeros(n, dtype=wp.vec2, device=self.device)
        evaluated = wp.zeros(n, dtype=wp.vec3, device=self.device)
        wp.launch(
            kernel,
            dim=n,
            inputs=[
                mesh.id,
                wp.array(np.asarray(points, dtype=np.float32), dtype=wp.vec3, device=self.device),
                1.0e6,
                hit,
                face,
                bary,
                evaluated,
            ],
            device=self.device,
        )
        wp.synchronize_device(self.device)
        found = hit.numpy().astype(bool)
        faces = face.numpy().astype(np.intp)
        uv = bary.numpy().astype(np.float64)
        surface = evaluated.numpy().astype(np.float64)
        distance = np.linalg.norm(surface - points, axis=-1)
        # A miss is impossible with an infinite search radius on a non-empty mesh, but a mesh that
        # failed to build would return zeros and read as face 0 at distance |p|. Push those to the
        # tolerance check rather than letting them sample cell 0.
        distance = np.where(found, distance, np.inf)
        weights = np.stack([uv[:, 0], uv[:, 1], 1.0 - uv[:, 0] - uv[:, 1]], axis=-1)
        return faces, weights, distance

    def apply(
        self,
        plane: Any,
        instance_ids: Any,
        id_to_labels: Mapping[Any, Any] | None,
        positions_world: Any,
        t_s: float,
        *,
        strict: bool = True,
        world_from_local: Mapping[str, Any] | None = None,
    ) -> NDArray[np.float32]:
        """Return ``plane`` with every bound prim's pixels replaced by its mesh field's values.

        Untouched when no bound prim is on screen, which is what makes attaching this to an
        existing scene a no-op until a mesh field is authored for something in it.
        """
        out = np.array(plane, dtype=np.float32, copy=True)
        ids = np.asarray(instance_ids)
        if not np.issubdtype(ids.dtype, np.integer):
            raise TypeError(f"instance_ids must be an integer plane, got {ids.dtype}")
        if ids.shape != out.shape:
            raise ValueError(f"instance_ids {ids.shape} does not match the plane {out.shape}")
        points = np.asarray(positions_world, dtype=np.float64)
        if points.shape != (*out.shape, 3):
            raise ValueError(f"positions_world {points.shape} does not match the plane {out.shape}")

        paths = labels_to_paths(id_to_labels)
        coverage = dict.fromkeys(self._by_path, 0)
        for ident, path in paths.items():
            if ident == BACKGROUND_INSTANCE_ID:
                continue
            fields = self._by_path.get(path)
            if not fields:
                continue
            mask = ids == ident
            if not mask.any():
                continue
            selected = points[mask]
            best = np.full(selected.shape[0], np.inf)
            values = np.zeros(selected.shape[0], dtype=np.float32)
            filled = np.zeros(selected.shape[0], dtype=bool)
            for field in fields:
                query = _into_frame(field.patch.frame, selected, world_from_local)
                face, bary, distance = self.locate(field, query)
                cell = field.patch.cell_of(face, bary[:, 0], bary[:, 1])
                sampled = np.asarray(field.temperature_at(t_s), dtype=np.float64)[cell]
                # Several meshes may share a prim; the nearest surface wins, so a bonnet and a
                # wing bound to one body do not depend on the order they were listed in.
                closer = (distance <= self.max_distance_m) & (distance < best)
                best = np.where(closer, distance, best)
                values = np.where(closer, sampled.astype(np.float32), values)
                filled |= closer
            if not filled.all() and strict:
                worst = float(np.min(best[~filled])) if np.isfinite(best[~filled]).any() else None
                raise ValueError(
                    f"{int((~filled).sum())} pixels of {path!r} are further than "
                    f"{1e3 * self.max_distance_m:.1f} mm from every mesh bound to it"
                    + (f" (nearest {1e3 * worst:.1f} mm)" if worst is not None else "")
                    + ". The id plane and the position plane disagree, or the bound mesh is not "
                    "the geometry that was rendered; snapping them to the nearest cell would "
                    "paint the surface with whatever part of it happened to be closest."
                )
            if not filled.all():
                values[~filled] = out[mask][~filled]
            out[mask] = values
            coverage[path] = int(filled.sum())
        self.last_coverage = coverage
        return out

    def __repr__(self) -> str:  # pragma: no cover - diagnostics
        return f"MeshPointBridge({len(self.bindings)} bindings over {len(self._by_path)} prims)"
