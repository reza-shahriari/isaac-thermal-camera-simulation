"""Measure the open model: which objects are parts, their area per thermal material, their size.

Blender data goes in through ``foreach_get`` and the arithmetic happens in :mod:`.coverage`, so a
2.5-million-face model is measured in array operations rather than a Python loop per face.
"""

import bpy
import numpy as np

from . import coverage, naming
from .library_state import emissivity_lwir


def part_objects(context: bpy.types.Context) -> list[bpy.types.Object]:
    """The parts: every mesh object in the scene that renders. The export writes exactly these."""
    return [ob for ob in context.scene.objects if ob.type == "MESH" and not ob.hide_render]


def _area_scale(ob: bpy.types.Object) -> float:
    """Local-to-world area factor. Exact for uniform scale; the geometric mean otherwise, which is
    close enough for a coverage percentage (``scripts/prep_asset.py`` audits the exported USD)."""
    det = abs(ob.matrix_world.to_3x3().determinant())
    return det ** (2.0 / 3.0)


def part_stats(ob: bpy.types.Object) -> coverage.PartStats:
    me = ob.data
    n = len(me.polygons)
    areas = np.empty(n, dtype=np.float32)
    slots = np.empty(n, dtype=np.int32)
    if n:
        me.polygons.foreach_get("area", areas)
        me.polygons.foreach_get("material_index", slots)
    materials = [s.material for s in ob.material_slots]
    return coverage.part_stats(
        ob.name,
        areas.astype(np.float64) * _area_scale(ob),
        slots,
        [m.name if m else None for m in materials],
        [m.irsim_material if m else None for m in materials],
    )


def summary(context: bpy.types.Context) -> coverage.Summary:
    return coverage.summarize(
        [part_stats(ob) for ob in part_objects(context)], emissivity_lwir(context)
    )


def world_extent(objects: list[bpy.types.Object]) -> tuple[float, float, float]:
    """Size along world X, Y, Z in metres, measured on the **vertices**.

    Not on ``Object.bound_box``: that is a local box, and transforming its corners through a
    rotated parent chain and re-taking min/max inflates it -- the mistake that once turned the
    Phantom 4's 41 x 46 cm into 60 x 62 cm (``configs/assets/phantom4_parts.yaml``).
    """
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    for ob in objects:
        me = ob.data
        n = len(me.vertices)
        if not n:
            continue
        co = np.empty(n * 3, dtype=np.float32)
        me.vertices.foreach_get("co", co)
        m = np.array(ob.matrix_world, dtype=np.float64)
        world = co.reshape(-1, 3).astype(np.float64) @ m[:3, :3].T + m[:3, 3]
        lo = np.minimum(lo, world.min(axis=0))
        hi = np.maximum(hi, world.max(axis=0))
    if not np.all(np.isfinite(lo)):
        return (0.0, 0.0, 0.0)
    ext = hi - lo
    return (float(ext[0]), float(ext[1]), float(ext[2]))


def used_materials(objects: list[bpy.types.Object]) -> set[bpy.types.Material]:
    """Materials on at least one face of the given parts (an unused slot does not count)."""
    used: set[bpy.types.Material] = set()
    for ob in objects:
        me = ob.data
        n = len(me.polygons)
        if not n or not ob.material_slots:
            continue
        slots = np.empty(n, dtype=np.int32)
        me.polygons.foreach_get("material_index", slots)
        for index in np.unique(slots):
            if 0 <= index < len(ob.material_slots):
                mat = ob.material_slots[int(index)].material
                if mat is not None:
                    used.add(mat)
    return used


def refresh(context: bpy.types.Context) -> coverage.Summary:
    """Recompute the checklist into ``Scene.irsim`` for the panel to show."""
    settings = context.scene.irsim
    parts = part_objects(context)
    result = summary(context)
    settings.checked = True
    settings.coverage = 100.0 * result.coverage
    settings.total_area_m2 = result.total_area_m2
    settings.part_count = len(parts)
    settings.unassigned_count = len(result.unassigned_parts)
    settings.size_x_m, settings.size_y_m, settings.size_z_m = world_extent(parts)

    settings.issues.clear()
    for part in result.unassigned_parts:
        issue = settings.issues.add()
        issue.name = part.name
        issue.kind = "UNASSIGNED"
        issue.area_m2 = part.unassigned_area_m2
        share = part.unassigned_area_m2 / part.total_area_m2 if part.total_area_m2 else 1.0
        materials = ", ".join(sorted(part.unassigned_by_material_m2))
        issue.detail = f"{100.0 * share:.0f} % of it has no thermal material ({materials})"
    for part_name, thermal, eps in result.mirror_parts:
        issue = settings.issues.add()
        issue.name = part_name
        issue.kind = "MIRROR"
        issue.detail = (
            f"{thermal}: LWIR emissivity {eps:.2f}, so it will show mostly reflected "
            "surroundings. Right only if this surface really is polished metal"
        )
    for ob in parts:
        if naming.is_default_name(ob.name):
            issue = settings.issues.add()
            issue.name = ob.name
            issue.kind = "NAME"
            issue.detail = "a default name: rename the part to what it is, e.g. propeller_left"
    return result
