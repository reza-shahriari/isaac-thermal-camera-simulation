"""One prim per material, so a multi-material mesh can be rendered as what it is (`AI.6`).

A USD mesh may carry several materials, one per ``materialBind`` ``GeomSubset``, and a great many
third-party assets do -- a building whose facade is glass, precast concrete and metal cladding is
one mesh in every architectural model this project has looked at. `AI.4` taught the stage walk to
*read* those subsets, which made the audit true. It did not make the render true, and could not:
the instance-id plane carries one id per prim (ADR 0014) and the material table is indexed by it,
so one prim is one material no matter what the asset says. A three-material wall renders as one
substance, and the audit's coverage figure then describes an asset the frame cannot reproduce.

This module plans the repair. Splitting the mesh so that one prim is one material is the same
regrouping `AI.5` does for functional parts (ADR 0138) with a different input: a face's material
slot instead of a face's connected component. The geometry work is Blender's
(``scripts/prep_asset.py``), because this package may not import USD or ``bpy``; what is here is
the arithmetic that decides whether the split is worth doing and what it will be called.

**The number that justifies the pass is an area, not a count.** A mesh with two materials where
the second covers four faces of nine thousand is not worth a prim; the same mesh split 60/40 is
two thirds wrong when read from slot 0. :attr:`MeshSplit.misassigned_area` is that fraction --
the share of a mesh's area whose material is *not* the one a slot-0 reader would give it -- and
:attr:`MaterialSplitReport.misassigned_area` is the same figure over the whole asset. A split that
does not move it is a split that bought nothing.

**Names are derived and must survive a round trip.** A piece is named ``<mesh>_<material>``,
sanitised to a USD identifier, and a collision is resolved with a numeric suffix rather than left
to whichever piece is exported last -- two prims with one name is how geometry disappears.

docs/physics-model.md §13.3; roadmap `AI.6`; ADR 0128 (the asset pipeline), ADR 0138 (regrouping
geometry rather than approximating the assignment), ADR 0047 (the mapping the split feeds).
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "MeshFaces",
    "SlotShare",
    "MeshSplit",
    "MaterialSplitReport",
    "piece_name",
    "plan_material_split",
    "usd_identifier",
]

#: Anything USD will not take in a prim name. USD identifiers are C-like: letters, digits and
#: underscores, not starting with a digit.
_NOT_IDENTIFIER = re.compile(r"[^A-Za-z0-9_]")


def usd_identifier(name: str) -> str:
    """``name`` as a USD-safe prim name, deterministically.

    Blender and USD disagree about what a name may contain -- a Blender object may be called
    ``Facade.001`` or ``wall glass`` and USD may not -- and the exporter's own mangling is not
    documented as stable. Deriving the name here means the prim path a scene config is authored
    against is the one this project chose, not the one an exporter happened to produce.
    """
    cleaned = _NOT_IDENTIFIER.sub("_", name.strip())
    if not cleaned or cleaned == "_" * len(cleaned):
        return "unnamed"
    return f"_{cleaned}" if cleaned[0].isdigit() else cleaned


def piece_name(mesh: str, material: str) -> str:
    """What one mesh's faces of one material are called after the split.

    ``<mesh>_<material>``, except when the material's name already begins with the mesh's, which
    is the common case in an asset organised by part: a mesh called ``Facade`` painted with
    ``Facade_Glass_Clear`` would otherwise become ``Facade_Facade_Glass_Clear``. A material with
    no name at all falls back to the mesh's own, so a face painted with nothing is still
    addressable rather than silently unnamed.
    """
    stem = usd_identifier(mesh)
    painted = usd_identifier(material) if material.strip() else ""
    if not painted:
        return stem
    if painted.lower().startswith(stem.lower()):
        return painted
    return usd_identifier(f"{stem}_{painted}")


@dataclass(frozen=True)
class MeshFaces:
    """One mesh as the splitter needs it: a material slot and an area for every face.

    ``slot_of_face`` and ``area_of_face`` are parallel per-face arrays, in the source's own face
    order. ``materials`` is the slot table: ``materials[slot_of_face[i]]`` is face *i*'s material.
    Areas are square metres, because the decision this module makes is about area.
    """

    name: str
    slot_of_face: NDArray[np.integer]
    area_of_face: NDArray[np.floating]
    materials: Sequence[str]

    def __post_init__(self) -> None:
        if self.slot_of_face.shape != self.area_of_face.shape:
            raise ValueError(
                f"{self.name}: {self.slot_of_face.shape} slots for {self.area_of_face.shape} areas"
            )
        if self.slot_of_face.size and int(self.slot_of_face.max()) >= len(self.materials):
            raise ValueError(
                f"{self.name}: face references slot {int(self.slot_of_face.max())} of "
                f"{len(self.materials)} materials"
            )
        if self.slot_of_face.size and int(self.slot_of_face.min()) < 0:
            raise ValueError(f"{self.name}: negative material slot")
        # float16 would quantise a square millimetre to nothing at a square metre's magnitude,
        # and these areas are summed over hundreds of thousands of faces (CLAUDE.md #2).
        if self.area_of_face.dtype == np.float16:
            raise ValueError(f"{self.name}: face areas must not be float16")


@dataclass(frozen=True)
class SlotShare:
    """What one material holds on one mesh."""

    material: str
    piece: str
    faces: int
    area_m2: float


@dataclass(frozen=True)
class MeshSplit:
    """The plan for one mesh: what it would become, and what leaving it whole costs."""

    mesh: str
    slots: tuple[SlotShare, ...]

    @property
    def multi_material(self) -> bool:
        return len(self.slots) > 1

    @property
    def area_m2(self) -> float:
        return float(sum(s.area_m2 for s in self.slots))

    @property
    def dominant(self) -> SlotShare:
        """The slot a reader of the mesh-level binding would give the whole mesh.

        Blender writes the *first* slot as the mesh binding (ADR 0128), and the first slot is
        the first one any face uses, which is what ``slots`` is ordered by -- not the largest.
        The distinction matters: on the committed fixture the slot-0 material is a third of the
        wall, and calling the largest slot dominant would understate the error.
        """
        return self.slots[0]

    @property
    def misassigned_area(self) -> float:
        """Fraction of this mesh's area whose material is not :attr:`dominant`'s."""
        total = self.area_m2
        return 0.0 if total <= 0.0 else 1.0 - self.dominant.area_m2 / total


@dataclass(frozen=True)
class MaterialSplitReport:
    """Every mesh's plan, and the asset-wide figure that says whether to run it."""

    meshes: tuple[MeshSplit, ...]

    @property
    def multi_material(self) -> tuple[MeshSplit, ...]:
        return tuple(m for m in self.meshes if m.multi_material)

    @property
    def prims_before(self) -> int:
        return len(self.meshes)

    @property
    def prims_after(self) -> int:
        return sum(len(m.slots) for m in self.meshes)

    @property
    def area_m2(self) -> float:
        return float(sum(m.area_m2 for m in self.meshes))

    @property
    def misassigned_area(self) -> float:
        """Fraction of the **asset's** area that renders as the wrong material without the split.

        This is the one number worth quoting. A count of multi-material meshes says nothing: an
        asset can have forty of them and be 0.1 % wrong, or one and be two thirds wrong.
        """
        total = self.area_m2
        if total <= 0.0:
            return 0.0
        wrong = sum(m.area_m2 - m.dominant.area_m2 for m in self.meshes)
        return float(wrong / total)

    def render(self) -> str:
        """A short report for the prep tool's log."""
        lines = [
            f"material split: {self.prims_before} mesh(es) -> {self.prims_after} prim(s); "
            f"{len(self.multi_material)} carry more than one material",
            f"  {self.misassigned_area:.1%} of {self.area_m2:.4f} m2 renders as the wrong "
            f"material without the split",
        ]
        for mesh in sorted(self.multi_material, key=lambda m: -m.misassigned_area)[:8]:
            shares = ", ".join(f"{s.material} {s.area_m2:.4f} m2" for s in mesh.slots)
            lines.append(f"  {mesh.mesh}: {mesh.misassigned_area:.1%} wrong -- {shares}")
        return "\n".join(lines)


def plan_material_split(meshes: Iterable[MeshFaces]) -> MaterialSplitReport:
    """Plan one prim per material for every mesh, with unique names across the whole asset.

    A slot no face uses is dropped: an asset's material slots outnumber the materials it actually
    paints with, and an empty prim is geometry the renderer pays for and nobody sees.
    """
    plans: list[MeshSplit] = []
    taken: dict[str, int] = {}
    for mesh in meshes:
        # First-use order, not sorted and not slot order: the mesh-level binding Blender writes
        # is the first slot a face uses, and `dominant` depends on this being that one.
        seen: list[int] = []
        for slot in mesh.slot_of_face.tolist():
            if slot not in seen:
                seen.append(int(slot))
        shares: list[SlotShare] = []
        for slot in seen:
            face_mask = mesh.slot_of_face == slot
            material = str(mesh.materials[slot])
            stem = piece_name(mesh.name, material)
            count = taken.get(stem, 0)
            taken[stem] = count + 1
            shares.append(
                SlotShare(
                    material=material,
                    piece=stem if count == 0 else f"{stem}_{count:d}",
                    faces=int(face_mask.sum()),
                    area_m2=float(mesh.area_of_face[face_mask].sum()),
                )
            )
        plans.append(MeshSplit(mesh=mesh.name, slots=tuple(shares)))
    return MaterialSplitReport(meshes=tuple(plans))
