"""What an imported asset's geometry *is*, as opposed to what it is made of.

A third-party asset is grouped the way the artist who sold it needed it grouped, which is by
**material**. The Phantom 4 of ADR 0128 is the worst case and the honest one: forty-one prims, all
named ``GeometryNode_<n>``, one of which is "all the white plastic" spread over 987 disconnected
shells. That grouping is correct for a renderer -- one draw call per material -- and useless for
everything this project wants to say about an aircraft, because none of the sentences are about
materials:

* the **battery** is the aircraft's largest stored-energy source and the thing a thermal camera
  finds first on a quadcopter that has been flying;
* the **motors** are the hottest surfaces, and there are four of them with four different duty
  histories, not one;
* the **propellers** are thin, have almost no thermal inertia, and move through their own
  boundary layer faster than anything else on the airframe.

None of those is a material and none of them is a prim. Before this module the Phantom 4 scene
declared a ``battery`` heat source and bound it to **no geometry at all** -- the target existed,
nothing in the asset was the battery, and nothing said so. That failure is silent by construction:
a scene with an unbound target renders a perfectly plausible aircraft that is missing its largest
heat source. :attr:`PartReport.empty_parts` exists so that it cannot happen again.

**Parts are recovered from connected components, not from names.** Splitting the mesh into
connected shells and clustering them by radius, height and size recovers real hardware, because a
propeller is a separate shell from the motor it is bolted to even when both are "white plastic".
Measured on the Phantom 4: 31,068 components over 2,486,459 faces, from which four rotor stations
fall out at r = 185 mm exactly 90 deg apart -- a 370 mm diagonal against DJI's published 350 mm --
each carrying a 27 mm motor can, a stator, a mount and two propeller blades offset +/-12 mm.

**The selectors are data.** A part is a name, a thermal target, and a predicate over component
statistics, authored in the asset's own YAML beside its material map. Nothing here knows what a
quadcopter is, which is the point: the next asset is a boat, and it must not need a new module.
The component statistics themselves come from the Blender worker in ``scripts/prep_asset.py``, so
this module stays engine-free and is tested against synthetic components.

**First match wins.** Parts are tried in declaration order and a component joins the first one that
accepts it, so a YAML reads most-specific-first: the battery block before the shell that encloses
it. The alternative -- best match, by some score -- would make the outcome depend on a ranking
nobody authored.

docs/physics-model.md §6.6; roadmap `AI.5`; ADR 0128 (the asset pipeline), ADR 0132 (the thermal
mesh is not the render mesh), ADR 0137 (the geometry budget).
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

__all__ = [
    "Component",
    "PartAssignment",
    "PartReport",
    "PartSelector",
    "PartSpec",
    "PartsConfig",
    "assign_parts",
]

#: Fraction of an asset's **area** that must land in a named part for the decomposition to pass.
#: Area rather than component count: an asset's components are overwhelmingly tiny shells (the
#: Phantom 4's median component is under a square millimetre) and counting them would let a
#: decomposition that misses the entire upper shell pass on the strength of a thousand screws.
DEFAULT_PART_COVERAGE = 0.95


@dataclass(frozen=True)
class Component:
    """One connected shell of an imported mesh, measured in the asset's own metres.

    ``lo``/``hi`` are the axis-aligned bounds of the shell's vertices *after* the asset's
    ``scale_to_metres`` has been applied, in the asset's own frame -- the same frame the asset
    config's part selectors are authored in. Measuring from vertices rather than from a local
    bounding box is not a detail: transforming the eight corners of a local box under a rotated
    parent chain and re-taking min/max inflates the Phantom 4 from 41 x 46 cm to 60 x 62.
    """

    index: int
    faces: int
    area_m2: float
    centroid: tuple[float, float, float]
    lo: tuple[float, float, float]
    hi: tuple[float, float, float]
    material_name: str | None = None

    def __post_init__(self) -> None:
        if self.faces <= 0:
            raise ValueError(f"component {self.index} has no faces")
        if self.area_m2 < 0.0:
            raise ValueError(f"component {self.index} has negative area")
        for name in ("centroid", "lo", "hi"):
            if len(getattr(self, name)) != 3:
                raise ValueError(f"component {self.index}: {name} must be a 3-vector")
        if any(h < lo for lo, h in zip(self.lo, self.hi, strict=True)):
            raise ValueError(f"component {self.index}: hi is below lo")

    @property
    def extent(self) -> tuple[float, float, float]:
        """The shell's size along each axis."""
        return (self.hi[0] - self.lo[0], self.hi[1] - self.lo[1], self.hi[2] - self.lo[2])

    def radius_m(self, centre: Sequence[float]) -> float:
        """Planar distance from the asset's centre, in the plane normal to the asset's up axis.

        Rotor stations, arms and landing gear are all placed by this, because a multirotor is
        laid out on a circle and an asset frame's up axis is the one the scene declares.
        """
        return math.hypot(self.centroid[0] - centre[0], self.centroid[1] - centre[1])


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class PartSelector(_Frozen):
    """A predicate over :class:`Component` statistics. Every field given must hold.

    All lengths are metres in the asset's own frame. An omitted field does not constrain, so an
    empty selector matches everything -- which is what a trailing catch-all part wants and what
    anything else should be prevented from being by :attr:`PartReport.greedy_parts`.
    """

    #: Source material names, matched case-insensitively. Empty means "any material".
    materials: list[str] = Field(default_factory=list)
    #: Planar station centre; pair with :attr:`within_m` to select one rotor station's hardware.
    near_xy: tuple[float, float] | None = None
    within_m: float | None = Field(default=None, gt=0.0)
    #: Radius band measured from the asset centre, for arms and gear that are neither body nor tip.
    r_min_m: float | None = Field(default=None, ge=0.0)
    r_max_m: float | None = Field(default=None, gt=0.0)
    z_min_m: float | None = None
    z_max_m: float | None = None
    area_min_m2: float | None = Field(default=None, ge=0.0)
    area_max_m2: float | None = Field(default=None, gt=0.0)
    faces_min: int | None = Field(default=None, ge=1)
    faces_max: int | None = Field(default=None, ge=1)
    #: Height of the shell itself. A landing-gear strut is 95 mm tall and a skid foot is 10 mm;
    #: they sit at the same radius and are made of the same plastic, so only extent separates them.
    extent_z_min_m: float | None = Field(default=None, ge=0.0)
    extent_z_max_m: float | None = Field(default=None, gt=0.0)

    @field_validator("materials")
    @classmethod
    def _lower(cls, v: list[str]) -> list[str]:
        return [s.strip().lower() for s in v if s.strip()]

    def accepts(self, component: Component, centre: Sequence[float]) -> bool:
        """Whether this selector claims ``component``."""
        if self.materials:
            name = (component.material_name or "").lower()
            if name not in self.materials:
                return False
        if self.near_xy is not None or self.within_m is not None:
            if self.near_xy is None or self.within_m is None:
                raise ValueError("near_xy and within_m must be given together")
            d = math.hypot(
                component.centroid[0] - self.near_xy[0], component.centroid[1] - self.near_xy[1]
            )
            if d > self.within_m:
                return False
        if self.r_min_m is not None or self.r_max_m is not None:
            r = component.radius_m(centre)
            if self.r_min_m is not None and r < self.r_min_m:
                return False
            if self.r_max_m is not None and r > self.r_max_m:
                return False
        z = component.centroid[2]
        if self.z_min_m is not None and z < self.z_min_m:
            return False
        if self.z_max_m is not None and z > self.z_max_m:
            return False
        if self.area_min_m2 is not None and component.area_m2 < self.area_min_m2:
            return False
        if self.area_max_m2 is not None and component.area_m2 > self.area_max_m2:
            return False
        if self.faces_min is not None and component.faces < self.faces_min:
            return False
        if self.faces_max is not None and component.faces > self.faces_max:
            return False
        ez = component.extent[2]
        if self.extent_z_min_m is not None and ez < self.extent_z_min_m:
            return False
        if self.extent_z_max_m is not None and ez > self.extent_z_max_m:  # noqa: SIM103
            return False
        return True


class PartSpec(_Frozen):
    """One named part of an asset, and the thermal target its geometry should carry."""

    name: str = Field(min_length=1)
    #: The scene target this part's cells are solved by -- ``battery``, ``motor``, ``airframe``.
    #: ``None`` means the part is named and measured but left to the scene's default target.
    target: str | None = None
    select: PartSelector = Field(default_factory=PartSelector)


class PartsConfig(_Frozen):
    """An asset's part decomposition: where its centre is, and what its parts are."""

    #: The asset's own centre, measured from the geometry and authored here. Radius selectors are
    #: relative to it, so an asset that does not sit at its own origin -- the Phantom 4 sits at
    #: y = +0.49 m -- still selects correctly.
    centre: tuple[float, float, float] = (0.0, 0.0, 0.0)
    parts: list[PartSpec] = Field(default_factory=list)
    coverage_threshold: float = Field(default=DEFAULT_PART_COVERAGE, ge=0.0, le=1.0)

    @field_validator("parts")
    @classmethod
    def _unique_names(cls, v: list[PartSpec]) -> list[PartSpec]:
        seen = [p.name.lower() for p in v]
        if len(set(seen)) != len(seen):
            raise ValueError("duplicate part name in asset parts")
        return v

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(p.name for p in self.parts)

    @property
    def targets(self) -> frozenset[str]:
        return frozenset(p.target for p in self.parts if p.target is not None)


@dataclass(frozen=True)
class PartAssignment:
    """Which part one component landed in, and which selector claimed it."""

    index: int
    part: str | None  # None on a component no part accepted
    target: str | None
    area_m2: float
    faces: int

    @property
    def assigned(self) -> bool:
        return self.part is not None


@dataclass(frozen=True)
class PartReport:
    """What :func:`assign_parts` found, and whether the decomposition is usable.

    The three failure modes are different and are reported separately, because they have different
    fixes and only two of them are the author's fault:

    * **uncovered area** -- geometry no part claims. Fix by adding a part or widening a selector.
    * **empty parts** -- a part that claimed nothing. This is the Phantom 4's unbound ``battery``,
      and it is the one that used to be silent. Fix by correcting the selector; a part that names
      hardware the asset does not model should be deleted rather than left matching nothing.
    * **greedy parts** -- a part that claimed more than a stated ceiling of the asset's area, which
      is how an over-wide selector announces itself before it swallows the aircraft.
    """

    total_area_m2: float
    total_faces: int
    threshold: float
    area_by_part: dict[str, float] = field(default_factory=dict)
    faces_by_part: dict[str, int] = field(default_factory=dict)
    empty_parts: tuple[str, ...] = ()
    unassigned_area_m2: float = 0.0
    unassigned_faces: int = 0

    @property
    def coverage(self) -> float:
        """Fraction of the asset's area that landed in a named part."""
        if self.total_area_m2 <= 0.0:
            return 0.0
        return 1.0 - self.unassigned_area_m2 / self.total_area_m2

    @property
    def passed(self) -> bool:
        """True when the asset is covered **and** every declared part found geometry."""
        return self.coverage >= self.threshold and not self.empty_parts

    def greedy_parts(self, ceiling: float = 0.60) -> tuple[str, ...]:
        """Parts holding more than ``ceiling`` of the asset's area.

        Not a gate: a hull genuinely is most of a boat. It is a prompt to look, because an
        over-wide selector and a large part are indistinguishable from their area alone.
        """
        if self.total_area_m2 <= 0.0:
            return ()
        return tuple(
            n for n, a in self.area_by_part.items() if a / self.total_area_m2 > float(ceiling)
        )

    def render(self) -> str:
        """A human-readable summary, in the shape of the material audit's."""
        lines = [
            f"parts: {len(self.area_by_part)} named, "
            f"coverage {self.coverage * 100:.1f}% of {self.total_area_m2:.5f} m2 "
            f"(threshold {self.threshold * 100:.0f}%)",
        ]
        for name, area in sorted(self.area_by_part.items(), key=lambda kv: -kv[1]):
            share = 100.0 * area / self.total_area_m2 if self.total_area_m2 > 0.0 else 0.0
            lines.append(
                f"  {name:<28} {area:9.5f} m2  {share:5.1f}%  {self.faces_by_part[name]:>8d} faces"
            )
        if self.unassigned_faces:
            lines.append(
                f"  {'<unassigned>':<28} {self.unassigned_area_m2:9.5f} m2  "
                f"{100.0 * self.unassigned_area_m2 / self.total_area_m2:5.1f}%  "
                f"{self.unassigned_faces:>8d} faces"
            )
        for name in self.empty_parts:
            lines.append(f"  EMPTY: part {name!r} matched no geometry")
        return "\n".join(lines)


def assign_parts(
    components: Iterable[Component], config: PartsConfig
) -> tuple[list[PartAssignment], PartReport]:
    """Assign each component to the first part whose selector accepts it.

    Returns the per-component assignments and a :class:`PartReport`. A component no part accepts is
    returned with ``part=None`` rather than dropped, so the caller can see what it is rather than
    only how much of it there was.
    """
    parts = list(config.parts)
    centre = config.centre
    assignments: list[PartAssignment] = []
    area_by_part: dict[str, float] = {p.name: 0.0 for p in parts}
    faces_by_part: dict[str, int] = {p.name: 0 for p in parts}
    total_area = 0.0
    total_faces = 0
    unassigned_area = 0.0
    unassigned_faces = 0

    for c in components:
        total_area += c.area_m2
        total_faces += c.faces
        hit: PartSpec | None = None
        for spec in parts:
            if spec.select.accepts(c, centre):
                hit = spec
                break
        if hit is None:
            unassigned_area += c.area_m2
            unassigned_faces += c.faces
            assignments.append(PartAssignment(c.index, None, None, c.area_m2, c.faces))
            continue
        area_by_part[hit.name] += c.area_m2
        faces_by_part[hit.name] += c.faces
        assignments.append(PartAssignment(c.index, hit.name, hit.target, c.area_m2, c.faces))

    empty = tuple(p.name for p in parts if faces_by_part[p.name] == 0)
    report = PartReport(
        total_area_m2=total_area,
        total_faces=total_faces,
        threshold=config.coverage_threshold,
        area_by_part=area_by_part,
        faces_by_part=faces_by_part,
        empty_parts=empty,
        unassigned_area_m2=unassigned_area,
        unassigned_faces=unassigned_faces,
    )
    return assignments, report


# ------------------------------------------------------------------------------------------------
# Splitting a prepared archive into per-part meshes.
#
# This runs on the **project interpreter**, not inside Blender: the archive is already world-space
# triangles (ADR 0132), so connected components can be recovered from the face arrays with numpy
# and no DCC is involved. That matters because selectors are iterated -- a part decomposition is
# tuned by looking at the report and adjusting a threshold -- and a loop that needs a 60 MB FBX
# re-imported each time is a loop nobody runs.
# ------------------------------------------------------------------------------------------------


def _shell_of_vertex(n_vertices: int, faces: Any) -> Any:
    """Union-find over triangle edges: which connected shell each vertex belongs to."""
    import numpy as np

    parent = np.arange(n_vertices, dtype=np.int64)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = int(parent[x])
        return int(x)

    for tri in faces:
        a = find(int(tri[0]))
        for other in (int(tri[1]), int(tri[2])):
            b = find(other)
            if a != b:
                parent[b] = a
    return np.array([find(i) for i in range(n_vertices)], dtype=np.int64)


def _xyz(point: Any) -> tuple[float, float, float]:
    """A 3-vector as the fixed-length tuple :class:`Component` declares.

    A generator expression over a NumPy row gives `tuple[float, ...]`, which is not the same type
    and is not checkable: nothing then stops a 2-vector or a 4-vector reaching a field documented
    as a point in the asset's frame. Unpacking is the check.
    """
    x, y, z = (float(v) for v in point)
    return (x, y, z)


def mesh_components(name: str, vertices_m: Any, faces: Any, material_name: str | None) -> Any:
    """The connected shells of one prim's triangles, as :class:`Component` statistics.

    Areas are the shells' own, computed from the triangles rather than carried over from the prim,
    because a prim's area says nothing about how it divides between the parts inside it.
    """
    import numpy as np

    v = np.asarray(vertices_m, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    if not len(f):
        return []
    shell = _shell_of_vertex(len(v), f)
    face_shell = shell[f[:, 0]]
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    face_area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=-1)

    out: list[tuple[Component, Any]] = []
    for root in np.unique(face_shell):
        mask = face_shell == root
        used = np.unique(f[mask])
        pts = v[used]
        out.append(
            (
                Component(
                    index=len(out),
                    faces=int(mask.sum()),
                    area_m2=float(face_area[mask].sum()),
                    centroid=_xyz(pts.mean(axis=0)),
                    lo=_xyz(pts.min(axis=0)),
                    hi=_xyz(pts.max(axis=0)),
                    material_name=material_name,
                ),
                mask,
            )
        )
    return out


def split_by_part(meshes: Any, config: PartsConfig) -> tuple[dict[str, Any], PartReport]:
    """Regroup a prepared asset's per-prim triangles into per-**part** meshes.

    Takes an :class:`irsim.io.assets.AssetMeshes` and returns ``{part name: AssetMesh}`` plus the
    decomposition report. The returned meshes are ordinary ``AssetMesh`` objects keyed by part
    name, so a scene binds ``mesh: {asset: phantom4, prim: battery}`` through exactly the machinery
    that already binds a prim -- no new scene concept, and the solver is unchanged.

    A part's ``material_name`` is that of whichever source material contributes the most area to
    it, which is the right answer for the parts that matter (a propeller is white ABS, a motor can
    is matte metal) and an arbitrary one for a part that genuinely spans materials.
    """
    import numpy as np

    from irsim.io.assets import AssetMesh

    per_part_faces: dict[str, list[tuple[Any, Any]]] = {}
    per_part_material: dict[str, dict[str, float]] = {}
    all_components: list[Component] = []
    running = 0

    for name in sorted(meshes):
        mesh = meshes[name]
        for component, mask in mesh_components(
            name, mesh.vertices_m, mesh.faces, mesh.material_name
        ):
            stat = Component(
                index=running,
                faces=component.faces,
                area_m2=component.area_m2,
                centroid=component.centroid,
                lo=component.lo,
                hi=component.hi,
                material_name=component.material_name,
            )
            running += 1
            all_components.append(stat)
            hit = None
            for spec in config.parts:
                if spec.select.accepts(stat, config.centre):
                    hit = spec
                    break
            if hit is None:
                continue
            per_part_faces.setdefault(hit.name, []).append(
                (mesh.vertices_m, np.asarray(mesh.faces)[mask])
            )
            if stat.material_name:
                bucket = per_part_material.setdefault(hit.name, {})
                bucket[stat.material_name] = bucket.get(stat.material_name, 0.0) + stat.area_m2

    _, report = assign_parts(all_components, config)

    out: dict[str, Any] = {}
    for part, chunks in per_part_faces.items():
        verts: list[Any] = []
        faces: list[Any] = []
        offset = 0
        for v, f in chunks:
            used, inverse = np.unique(f, return_inverse=True)
            verts.append(np.asarray(v, dtype=np.float64)[used])
            faces.append(inverse.reshape(f.shape).astype(np.intp) + offset)
            offset += len(used)
        v_all = np.concatenate(verts, axis=0)
        f_all = np.concatenate(faces, axis=0)
        materials = per_part_material.get(part, {})
        dominant = max(materials.items(), key=lambda kv: kv[1])[0] if materials else None
        area = float(report.area_by_part.get(part, 0.0))
        out[part] = AssetMesh(
            name=part,
            material_name=dominant,
            vertices_m=v_all,
            faces=f_all,
            area_m2=area,
            area_before_m2=area,
        )
    return out, report
