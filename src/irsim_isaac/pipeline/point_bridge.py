"""Per-pixel temperature from a solved surface field, keyed by the position AOV.

docs/physics-model.md §13.1, §13.3; roadmap MP.3; ADR 0087 (the field and why the position AOV is
the parameterisation), ADR 0014 (what this build actually transports per pixel).

`AerialThermalBridge.temperature_plane` gathers `table[instance_id]`: one temperature per prim,
which is the defect ADR 0087 exists to fix. This module is the other half of that fix on the render
path, and it is deliberately **additive** -- it takes a finished plane and overwrites only the
pixels belonging to prims that have a field bound to them. A prim with no binding keeps the
per-instance value it always had, so every existing scene renders bit-identically.

The lookup is the one ADR 0014 leaves available. Temperature cannot cross a colour AOV (all fp16),
but `Camera3dPositionSD` carries a **float32 position good to 3.4 mm**, which M2.4 established is
**camera** space on this build -- so :func:`world_positions` applies the same rotation
`ray_directions` does, from the same `camera_to_world`, rather than deriving a second one that
could disagree with it by twice the camera's tilt without raising anything.

**A pixel that lands on a bound prim but outside all of its patches raises.** It is the symptom of
a patch authored smaller than the geometry it is meant to cover, and the alternative -- taking the
prim's fallback value, or the nearest cell -- produces a frame in which part of a bonnet is a field
and part of it is a flat patch, with a seam that looks like physics.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from irsim.optics.motion import transform_points
from irsim.thermal.surface_field import PlanarThermalField
from irsim_isaac.pipeline.gbuffer_isaac import _as_f64_plane
from irsim_isaac.pipeline.material_ids import BACKGROUND_INSTANCE_ID, labels_to_paths

__all__ = ["SurfaceBinding", "PointwiseTemperature", "world_positions", "bindings_from_scene"]


def world_positions(
    position: Any,
    *,
    frame: str,
    camera_position: Any = None,
    camera_to_world: Any = None,
) -> NDArray[np.float64]:
    """``(H, W, 3)`` world-space surface positions from the position AOV.

    The world-space counterpart of :func:`~irsim_isaac.pipeline.gbuffer_isaac.ray_directions`,
    which needs only the *direction*: here the magnitude matters, because it is what selects a cell.
    ``frame="camera"`` (what M2.4 measured on this build) rotates by ``camera_to_world`` and adds
    the camera's own position; ``frame="world"`` passes through.
    """
    # The same validation `ray_directions` applies, from the same helper rather than a second
    # copy: the annotator delivers (H, W, 4) with an unused alpha, and a private reimplementation
    # here would be one more place for the channel count or the float16 rule to drift.
    pos = _as_f64_plane("position", position, 3)
    if frame == "world":
        return pos
    if frame != "camera":
        raise ValueError(f"unknown position frame {frame!r}")
    if camera_position is None:
        raise ValueError("frame='camera' needs camera_position to reach world space")
    out = pos
    if camera_to_world is not None:
        rot = np.asarray(camera_to_world, dtype=np.float64)
        if rot.shape == (4, 4):
            rot = rot[:3, :3]
        if rot.shape != (3, 3):
            raise ValueError(f"camera_to_world must be 3x3 or 4x4, got {rot.shape}")
        out = out @ rot.T
    return np.asarray(out + np.asarray(camera_position, dtype=np.float64).reshape(3))


def _into_frame(
    frame: str, points_world: NDArray[np.float64], world_from_local: Mapping[str, Any] | None
) -> NDArray[np.float64]:
    """World points expressed in ``frame``, which is ``"world"`` or a prim path (PT.5).

    ``surface_field`` has documented a prim path as a legal frame since MP.1 -- "``world`` for a
    road, a prim path for a panel that moves with its object" -- while this bridge refused every
    frame but ``world``. The docstring promised what the code declined, and every aerial and
    maritime target moves, so the promise was the useful half.

    Two conventions have to be right at once and both are borrowed rather than restated.

    **Direction.** The matrix is **world-from-local**, which is what USD's
    ``ComputeLocalToWorldTransform`` returns, and it is inverted *here* rather than at the call
    site: a caller that had to remember to invert would eventually not, and the failure is a patch
    that tracks its prim backwards -- smooth, plausible, and wrong by twice the prim's offset.

    **Row versus column.** USD matrices are row-vector, ``p' = p @ M``, with the translation in the
    last **row**. Getting that backwards transposes every rotation and still produces a plausible
    picture, which is the mistake ADR 0014's M10.19 addendum records costing this project a
    164-row horizon. So the multiply is :func:`irsim.optics.motion.transform_points`, the one
    place the convention already lives, rather than a second copy of it here.
    """
    if frame == "world":
        return points_world
    if not world_from_local or frame not in world_from_local:
        raise KeyError(
            f"a patch is authored in frame {frame!r} but no world_from_local matrix was supplied "
            "for it; pass one per moving prim (PointwiseTemperature.local_frames lists them)"
        )
    matrix = np.asarray(world_from_local[frame], dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"world_from_local[{frame!r}] must be 4x4, got {matrix.shape}")
    return transform_points(points_world, np.linalg.inv(matrix))


@dataclass(frozen=True)
class SurfaceBinding:
    """One prim's temperature field. Several may share a prim -- a bonnet and a roof, say.

    ``field.patch.frame`` is a name the caller sets and this module **checks**: a patch authored in
    a prim's local frame cannot be sampled with world positions, and the two are indistinguishable
    from the array shapes alone.
    """

    prim_path: str
    field: PlanarThermalField

    def __post_init__(self) -> None:
        if not self.prim_path:
            raise ValueError("a binding needs a prim path")


def bindings_from_scene(scene: Any) -> list[SurfaceBinding]:
    """Every patched surface the scene config bound to a prim, as `IrCamera` takes them (PT.17).

    The core exposes ``(prim path, field)`` pairs (`Scene.surface_bindings`) because it may not
    import this module (CLAUDE.md #1); this is the one line that turns them into bindings, so a
    render driver reads its fields off the scene instead of building grids in Python.
    """
    return [SurfaceBinding(path, fld) for path, fld in scene.surface_bindings()]


class PointwiseTemperature:
    """Overwrites the patch-backed prims of a per-instance temperature plane with their fields."""

    def __init__(self, bindings: Sequence[SurfaceBinding]) -> None:
        self.bindings = tuple(bindings)
        self._by_path: dict[str, list[PlanarThermalField]] = {}
        for binding in self.bindings:
            self._by_path.setdefault(binding.prim_path, []).append(binding.field)

    @property
    def prim_paths(self) -> tuple[str, ...]:
        return tuple(self._by_path)

    @property
    def local_frames(self) -> tuple[str, ...]:
        """The non-world frames this bridge needs a ``world_from_local`` matrix for (PT.5).

        A caller reads these off the bridge rather than guessing which prims move: the frame name
        is a prim path, and supplying a matrix for it is what lets a patch ride its object.
        """
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
        """Return ``plane`` with every bound prim's pixels replaced by its field's own values.

        Untouched when no bound prim is on screen, which is what makes attaching this to an
        existing scene a no-op until a patch is authored for something in it.
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
            filled = np.zeros(selected.shape[0], dtype=bool)
            values = np.zeros(selected.shape[0], dtype=np.float32)
            for field in fields:
                pending = ~filled
                if not pending.any():
                    break
                # A patch authored in a prim's own frame rides that prim (PT.5). The points are
                # world; the patch is not; so the points come back through the prim's transform
                # before they meet the grid.
                query = _into_frame(field.patch.frame, selected[pending], world_from_local)
                sampled = field.sample_at(t_s, query)
                hit = np.isfinite(sampled)
                idx = np.flatnonzero(pending)[hit]
                values[idx] = sampled[hit]
                filled[idx] = True
            if not filled.all() and strict:
                raise ValueError(
                    f"{int((~filled).sum())} pixels of {path!r} fall outside every patch bound to "
                    "it. A patch authored smaller than its geometry renders part of a surface as "
                    "a field and part as a flat value, with a seam that looks like physics; widen "
                    "the patch or its thickness_m rather than letting them take a fallback."
                )
            if not filled.all():
                # Non-strict: keep the per-instance value, which is what the pixel had before.
                values[~filled] = out[mask][~filled]
            out[mask] = values
        return out

    def __repr__(self) -> str:  # pragma: no cover - diagnostics
        return (
            f"PointwiseTemperature({len(self.bindings)} bindings over {len(self._by_path)} prims)"
        )
