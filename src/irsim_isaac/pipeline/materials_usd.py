"""Walk a USD stage into the engine-free :class:`~irsim.materials.mapping.PrimRecord` dump.

docs/physics-model.md §13.3; roadmap M10.2; ADR 0047 (precedence and the coverage gate).

ADR 0047 fixed the mapping precedence -- explicit override, then semantic class, then a glob on
the bound material's name, then a recorded miss -- and deliberately kept the resolver engine-free
by having it consume plain records. This module is the other half: the only place that opens a USD
stage and produces those records. Nothing in ``irsim`` imports USD, and this module imports it
inside functions, so ``import irsim_isaac`` still works on a machine with no Isaac Sim.

``pxr`` is provided by Kit and is **not importable outside a running Kit application** on this
build (the same is true of ``warp`` -- ADR 0014), so the USD path needs the engine even though
nothing about reading a stage is inherently graphical. The two-step flow exists for that reason:
dump the records inside Kit once, then audit them anywhere.

**The override attribute.** ``thermal:material`` on the prim, holding a library material name. It
is authored in the asset (or by a scene-prep script) and beats every heuristic, which is what makes
a wrong glob fixable without editing the rules for everyone. It is a plain USD attribute rather
than an applied schema so that any DCC can write it without our schema installed.
"""

from __future__ import annotations

import json
import os
import pathlib
from dataclasses import dataclass
from typing import Any

from irsim.materials.mapping import PrimRecord

__all__ = [
    "THERMAL_MATERIAL_ATTR",
    "MATERIAL_BIND_FAMILY",
    "StageWalk",
    "walk_stage",
    "prim_records",
    "bound_material_name",
    "semantic_class_of",
    "override_of",
    "dump_prim_records",
]

#: The per-prim override attribute (ADR 0047 precedence rule 1).
THERMAL_MATERIAL_ATTR = "thermal:material"

#: USD's reserved subset family for material assignment. A ``GeomSubset`` in any other
#: family (an artist's selection set, a simulation group) says nothing about substance.
MATERIAL_BIND_FAMILY = "materialBind"


def bound_material_name(prim: Any) -> str | None:
    """Name of the material bound to this prim, or None when nothing is bound.

    The *name* is used, not the path, because the glob rules in ``configs/materials/mapping.yaml``
    match how artists name materials (``Car_Paint_Red``, ``Glass_Clear``), which survives being
    re-parented far better than an absolute path does.
    """
    from pxr import UsdShade

    try:
        binding = UsdShade.MaterialBindingAPI(prim)
        material = binding.ComputeBoundMaterial()[0]
    except Exception:  # noqa: BLE001 - an unbindable prim is a miss, not a crash
        return None
    if not material or not material.GetPrim().IsValid():
        return None
    return str(material.GetPrim().GetName())


def semantic_class_of(prim: Any) -> str | None:
    """The prim's ``class`` semantic label, via the 6.x API with the legacy schema as fallback."""
    try:
        from isaacsim.core.experimental.utils.semantics import get_labels

        labels = get_labels(prim)
        if isinstance(labels, dict):
            value = labels.get("class")
            if value:
                return str(value[0] if isinstance(value, (list, tuple)) else value)
    except Exception:  # noqa: BLE001 - fall through to the legacy schema
        pass
    try:
        from pxr import Semantics

        for name in prim.GetAppliedSchemas():
            if not name.startswith("SemanticsAPI:"):
                continue
            api = Semantics.SemanticsAPI.Get(prim, name.split(":", 1)[1])
            if api and api.GetSemanticTypeAttr().Get() == "class":
                data = api.GetSemanticDataAttr().Get()
                if data:
                    return str(data)
    except Exception:  # noqa: BLE001
        return None
    return None


def override_of(prim: Any) -> str | None:
    """The ``thermal:material`` override, or None. An empty string counts as absent."""
    attr = prim.GetAttribute(THERMAL_MATERIAL_ATTR)
    if not attr or not attr.IsValid():
        return None
    value = attr.Get()
    return str(value) if value else None


@dataclass(frozen=True)
class StageWalk:
    """What one walk of a stage found: the records, and what is not sayable in them.

    ``records`` is the audit's input. The two report lists exist because a ``materialBind``
    subset is a fact about the asset that a flat list of per-prim records cannot express, and
    silence about it is expensive in both directions (`AI.4`).
    """

    #: One per surface, in traversal order. With ``expand_subsets`` a subset is a surface.
    records: tuple[PrimRecord, ...]
    #: Meshes whose material varies across their own faces. One instance id is transported per
    #: prim (ADR 0014), so the renderer gives every one of these a *single* material.
    subset_meshes: tuple[str, ...]
    #: Of those, the ones that also bind a material at the mesh level. Blender writes slot 0
    #: there as a documented Hydra workaround, so reading it maps the whole mesh to whichever
    #: material happened to be first in the artist's stack.
    shadowed: tuple[str, ...]


def _material_bind_subsets(prim: Any) -> list[Any]:
    from pxr import UsdShade

    try:
        return list(UsdShade.MaterialBindingAPI(prim).GetMaterialBindSubsets())
    except Exception:  # noqa: BLE001 - a prim that cannot carry subsets simply has none
        return []


def walk_stage(
    stage: Any = None,
    *,
    root: str = "/",
    include_invisible: bool = False,
    expand_subsets: bool = False,
) -> StageWalk:
    """Every renderable prim under ``root`` as an engine-free record, subsets accounted for.

    Only geometry is reported -- a ``Gprim`` (mesh, sphere, cube, ...). Xforms, scopes, cameras,
    lights and the material prims themselves are not surfaces and would dilute the coverage
    fraction ADR 0047 gates on, making an asset look better mapped than it is.

    ``expand_subsets`` chooses which question is being asked, and the two answers are different:

    * **False** (the default, and what every render driver wants) -- one record per prim, from
      the prim's own binding. That is what the renderer can actually transport, because the
      instance-id plane carries one id per prim and a material table is indexed by it.
    * **True** (what an audit wants) -- one record per ``materialBind`` subset, from the
      subset's binding, with the mesh-level binding deliberately ignored. That is what the asset
      says, and on a multi-material mesh it is the only reading that is true.

    Visibility is decided before subsets are read: an invisible mesh contributes nothing either
    way, rather than contributing surfaces that are not in the picture.
    """
    import omni.usd
    from pxr import Usd, UsdGeom

    if stage is None:
        stage = omni.usd.get_context().get_stage()
    start = stage.GetPrimAtPath(root)
    if not start or not start.IsValid():
        raise ValueError(f"no prim at {root!r}")

    records: list[PrimRecord] = []
    subset_meshes: list[str] = []
    shadowed: list[str] = []
    for prim in Usd.PrimRange(start):
        if not prim.IsA(UsdGeom.Gprim):
            continue
        if not include_invisible:
            imageable = UsdGeom.Imageable(prim)
            if imageable and imageable.ComputeVisibility() == UsdGeom.Tokens.invisible:
                continue
        subsets = _material_bind_subsets(prim)
        if subsets:
            subset_meshes.append(str(prim.GetPath()))
            if bound_material_name(prim) is not None:
                shadowed.append(str(prim.GetPath()))
        if subsets and expand_subsets:
            for subset in subsets:
                child = subset.GetPrim()
                records.append(
                    PrimRecord(
                        path=str(child.GetPath()),
                        material_name=bound_material_name(child),
                        # A subset inherits neither: semantics and the override are authored on
                        # the prim, and reading the mesh's would assert them of one face group.
                        semantic_class=semantic_class_of(prim),
                        override=override_of(prim),
                    )
                )
            continue
        records.append(
            PrimRecord(
                path=str(prim.GetPath()),
                material_name=bound_material_name(prim),
                semantic_class=semantic_class_of(prim),
                override=override_of(prim),
            )
        )
    return StageWalk(
        records=tuple(records),
        subset_meshes=tuple(subset_meshes),
        shadowed=tuple(shadowed),
    )


def prim_records(
    stage: Any = None,
    *,
    root: str = "/",
    include_invisible: bool = False,
    expand_subsets: bool = False,
) -> list[PrimRecord]:
    """:meth:`walk_stage`'s records alone, for callers that have nothing to do with the report."""
    walk = walk_stage(
        stage, root=root, include_invisible=include_invisible, expand_subsets=expand_subsets
    )
    return list(walk.records)


def dump_prim_records(records: list[PrimRecord], path: str | os.PathLike[str]) -> pathlib.Path:
    """Write the records as the JSON ``scripts/audit_materials.py`` already consumes."""
    out = pathlib.Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "path": r.path,
            "material_name": r.material_name,
            "semantic_class": r.semantic_class,
            "override": r.override,
        }
        for r in records
    ]
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out
