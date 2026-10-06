"""The body schema: a human is a fixed taxonomy labelled by its skeleton (HU.2).

docs/physics-model.md §6.1 (the person's two surfaces), §16.2 (skin and clothing materials);
roadmap HU.2; ADR 0122 (a person is two surfaces); ADR 0138 (what this does *not* do);
``docs/research/2026-10-06-humans-in-the-simulation.md``.

An imported drone is decomposed by discovery -- connected components, clustered and named
(ADR 0138). A human cannot be: the body is one connected shell. It does not need to be, either,
because a human's parts are **known in advance and identical for every human**. They are the
seventeen segments JOS-3 solves (Takahashi et al. 2021, doi:10.1016/j.enbuild.2020.110575), each
bare or under a garment, plus hair and eyes. So a human mesh is *labelled* onto one taxonomy, and
the labeller is the skeleton: every rigged humanoid already carries per-vertex bone weights, and
the bones have names (Mixamo, Rigify, SMPL-X, MPFB) that map onto the segments.

This module is that taxonomy's loader: ``configs/humans/body_schema.yaml`` holds the segments,
the region a bone may name when one bone drives two segments (the spine drives Chest *and*
Back), the garment slots and what each covers, and one bone map per rig. The loader refuses a
schema that is not JOS-3's seventeen by name, a rig that leaves a deforming bone unmapped, a
target that is not a segment, and a map whose left and right halves disagree -- because every
one of those is an error a plausible render would hide: a forearm labelled as a hand still
renders a warm arm.

:func:`segment_for_bone` is the one lookup every prep tool uses (HU.3, HU.9): prefix stripped,
exact, then pattern, then without a trailing ``.NNN`` (a Rigify DEF- limb is two bones). A bone
that deforms nothing (``HeadTop_End``) answers ``None``; a bone the rig does not know is an
error that names it.

The asset side (``kind: human`` in ``configs/assets/*.yaml``) is :class:`HumanSpec`: the
phenotype JOS-3 takes -- sex, age, height, mass -- and the garments, each on a slot of this
schema with a library material, an insulation in clo and, for the RGB companion and the
reflective bands, a colour. Colour never reaches a long-wave emissivity (Zhang, Hu & Zhang 2009,
J. Textile Inst. 100:90): it sets solar absorption and NIR reflectance in HU.5, nothing else.
"""

from __future__ import annotations

import fnmatch
import os
import pathlib
import re
from collections.abc import Iterable
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

__all__ = [
    "BODY_SCHEMA_PATH",
    "BODY_SCHEMA_VERSION",
    "JOS3_SEGMENTS",
    "BodySchema",
    "GarmentSlot",
    "GarmentSpec",
    "HumanSpec",
    "Phenotype",
    "RigMap",
    "Segment",
    "Split",
    "load_body_schema",
    "segment_for_bone",
]

BODY_SCHEMA_VERSION = 1
BODY_SCHEMA_PATH = (
    pathlib.Path(__file__).resolve().parents[3] / "configs" / "humans" / "body_schema.yaml"
)

#: JOS-3's ``BODY_NAMES`` (src/jos3/matrix.py), in its order and spelling. The schema file must
#: carry exactly these, because HU.4 binds the solver's per-segment skin temperatures by name.
#: "Shoulder" is the upper arm, "Arm" the forearm, "Leg" the lower leg -- the 65MN convention.
JOS3_SEGMENTS: tuple[str, ...] = (
    "Head",
    "Neck",
    "Chest",
    "Back",
    "Pelvis",
    "LShoulder",
    "LArm",
    "LHand",
    "RShoulder",
    "RArm",
    "RHand",
    "LThigh",
    "LLeg",
    "LFoot",
    "RThigh",
    "RLeg",
    "RFoot",
)

Side = Literal["left", "right", "centre"]
_TRAILING_INDEX = re.compile(r"\.\d{3}$")


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Segment(_Frozen):
    """One JOS-3 segment: its side, its mirror, and JOS-3's standard local surface area."""

    name: str = Field(min_length=1)
    side: Side
    mirror: str | None = None
    area_m2: float = Field(gt=0.0)

    @model_validator(mode="after")
    def _sided_segments_have_mirrors(self) -> Segment:
        if (self.side == "centre") != (self.mirror is None):
            raise ValueError(
                f"segment {self.name!r}: a left or right segment names its mirror and a centre "
                f"segment has none"
            )
        return self


class Split(_Frozen):
    """How a region resolves to a segment per face: by the sign of one body coordinate."""

    split: Literal["forward"]
    positive: str
    negative: str


class GarmentSlot(_Frozen):
    covers: list[str] = Field(min_length=1)


class RigMap(_Frozen):
    """One skeleton's deforming bones and the segment or region each drives."""

    source: str = Field(min_length=1)
    strip_prefixes: list[str] = Field(default_factory=list)
    ignore_prefixes: list[str] = Field(default_factory=list)
    mirror_tokens: list[tuple[str, str]] = Field(default_factory=list)
    bones: dict[str, str] = Field(min_length=1)
    patterns: dict[str, str] = Field(default_factory=dict)
    non_deforming: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _no_bone_is_both(self) -> RigMap:
        both = sorted(set(self.bones) & set(self.non_deforming))
        if both:
            raise ValueError(f"bones listed as deforming and as non-deforming: {both}")
        return self

    def _strip(self, bone: str) -> str:
        for p in self.strip_prefixes:
            if bone.startswith(p):
                return bone[len(p) :]
        return bone

    def is_ignored(self, bone: str) -> bool:
        """A control, organisational or widget bone a generated rig carries beside the DEF- ones."""
        return any(bone.startswith(p) for p in self.ignore_prefixes)

    def target(self, bone: str) -> str | None:
        """The segment or region ``bone`` drives; ``None`` for a bone that deforms nothing.

        Raises ``KeyError`` naming the bone when the rig does not know it.
        """
        name = self._strip(bone)
        if name in self.non_deforming:
            return None
        if name in self.bones:
            return self.bones[name]
        for pat, tgt in self.patterns.items():
            if fnmatch.fnmatchcase(name, pat):
                return tgt
        base = _TRAILING_INDEX.sub("", name)
        if base != name:
            if base in self.non_deforming:
                return None
            if base in self.bones:
                return self.bones[base]
            for pat, tgt in self.patterns.items():
                if fnmatch.fnmatchcase(base, pat):
                    return tgt
        raise KeyError(f"bone {bone!r} is not in this rig's map")

    def mirror_name(self, bone: str) -> str | None:
        """The other side's bone name by the rig's side tokens, or ``None`` for a centre bone."""
        for left, right in self.mirror_tokens:
            if left in bone:
                return bone.replace(left, right)
            if right in bone:
                return bone.replace(right, left)
        return None


class BodySchema(_Frozen):
    """The taxonomy every human in irsim is labelled onto. Validated against JOS-3's names."""

    name: str = Field(min_length=1)
    source: str = Field(min_length=1)
    segments: list[Segment]
    regions: dict[str, Split] = Field(default_factory=dict)
    layers: list[str] = Field(min_length=1)
    garment_slots: dict[str, GarmentSlot]
    rigs: dict[str, RigMap]

    # -- the checks ---------------------------------------------------------------------------

    @model_validator(mode="after")
    def _is_jos3(self) -> BodySchema:
        names = [s.name for s in self.segments]
        missing = [n for n in JOS3_SEGMENTS if n not in names]
        extra = [n for n in names if n not in JOS3_SEGMENTS]
        if missing or extra:
            raise ValueError(
                f"body schema is not JOS-3's seventeen segments: missing {missing}, extra {extra}"
            )
        if names != list(JOS3_SEGMENTS):
            raise ValueError("body schema segments are not in JOS-3's order")
        by_name = {s.name: s for s in self.segments}
        for s in self.segments:
            if s.mirror is not None:
                m = by_name.get(s.mirror)
                if m is None or m.mirror != s.name or m.side == s.side:
                    raise ValueError(
                        f"segment {s.name!r}: mirror {s.mirror!r} does not mirror it back"
                    )
        return self

    @model_validator(mode="after")
    def _regions_slots_and_layers(self) -> BodySchema:
        names = self.segment_names
        for region, split in self.regions.items():
            if region in names:
                raise ValueError(f"region {region!r} shadows a segment")
            for t in (split.positive, split.negative):
                if t not in names:
                    raise ValueError(f"region {region!r} resolves to {t!r}, not a segment")
        for slot, g in self.garment_slots.items():
            bad = [c for c in g.covers if c not in names]
            if bad:
                raise ValueError(f"garment slot {slot!r} covers {bad}, which are not segments")
        if "skin" not in self.layers:
            raise ValueError("the layers must include 'skin'")
        return self

    @model_validator(mode="after")
    def _every_rig_covers_the_body(self) -> BodySchema:
        for rig_name, rig in self.rigs.items():
            bad = {b: t for b, t in {**rig.bones, **rig.patterns}.items() if t not in self.targets}
            if bad:
                raise ValueError(f"rig {rig_name!r} maps bones to unknown targets: {bad}")
            missing = self.segment_names - self.coverage(rig_name)
            if missing:
                raise ValueError(f"rig {rig_name!r} drives no bone for segments {sorted(missing)}")
            asym = self.asymmetries(rig_name)
            if asym:
                raise ValueError(f"rig {rig_name!r} is not mirror-symmetric: {asym}")
        return self

    # -- the answers --------------------------------------------------------------------------

    @property
    def segment_names(self) -> frozenset[str]:
        return frozenset(s.name for s in self.segments)

    @property
    def targets(self) -> frozenset[str]:
        """Everything a bone may be mapped to: a segment or a region."""
        return self.segment_names | frozenset(self.regions)

    def segment(self, name: str) -> Segment:
        for s in self.segments:
            if s.name == name:
                return s
        raise KeyError(f"unknown segment {name!r}; the schema has {list(JOS3_SEGMENTS)}")

    def resolve(self, target: str) -> frozenset[str]:
        """The segments a bone target can land on: itself, or a region's two sides."""
        if target in self.regions:
            r = self.regions[target]
            return frozenset((r.positive, r.negative))
        if target in self.segment_names:
            return frozenset((target,))
        raise KeyError(f"{target!r} is neither a segment nor a region")

    def coverage(self, rig: str) -> frozenset[str]:
        """Every segment some bone of ``rig`` can drive."""
        out: set[str] = set()
        for t in {**self.rigs[rig].bones, **self.rigs[rig].patterns}.values():
            out |= self.resolve(t)
        return frozenset(out)

    def mirror_target(self, target: str) -> str:
        if target in self.regions:
            return target
        s = self.segment(target)
        return s.mirror or s.name

    def asymmetries(self, rig: str) -> list[str]:
        """Bones whose mirror bone drives something other than the mirror segment."""
        r = self.rigs[rig]
        out = []
        for bone, tgt in r.bones.items():
            other = r.mirror_name(bone)
            if other is None:
                continue
            try:
                other_tgt = r.target(other)
            except KeyError:
                out.append(f"{bone} -> {tgt} but {other} is not in the map")
                continue
            if other_tgt != self.mirror_target(tgt):
                out.append(f"{bone} -> {tgt} but {other} -> {other_tgt}")
        return out

    def check_bones(self, rig: str, bones: Iterable[str]) -> dict[str, str | None]:
        """Label every bone of a real armature, or raise naming the ones the rig does not know."""
        r = self.rigs[rig]
        labels: dict[str, str | None] = {}
        unknown = []
        for b in bones:
            if r.is_ignored(b):
                labels[b] = None
                continue
            try:
                labels[b] = r.target(b)
            except KeyError:
                unknown.append(b)
        if unknown:
            raise ValueError(f"rig {rig!r} does not map bones {unknown}")
        return labels


class _BodyFile(_Frozen):
    schema_version: int
    body: BodySchema

    @field_validator("schema_version")
    @classmethod
    def _version(cls, v: int) -> int:
        if v != BODY_SCHEMA_VERSION:
            raise ValueError(f"body schema_version {v} != {BODY_SCHEMA_VERSION}")
        return v


def load_body_schema(path: str | os.PathLike[str] | None = None) -> BodySchema:
    """Read the body schema; the committed one by default."""
    p = pathlib.Path(path) if path is not None else BODY_SCHEMA_PATH
    return _BodyFile.model_validate(yaml.safe_load(p.read_text(encoding="utf-8"))).body


def segment_for_bone(schema: BodySchema, rig: str, bone: str) -> str | None:
    """The segment or region a bone drives under ``rig``; ``None`` if it deforms nothing."""
    if rig not in schema.rigs:
        raise KeyError(f"unknown rig {rig!r}; the schema has {sorted(schema.rigs)}")
    r = schema.rigs[rig]
    if r.is_ignored(bone):
        return None
    return r.target(bone)


# ---------------------------------------------------------------------------------------------
# The asset side: `kind: human`
# ---------------------------------------------------------------------------------------------


class Phenotype(_Frozen):
    """What JOS-3 takes to size a body: its `sex`, `age`, `height` and `weight`."""

    sex: Literal["male", "female"]
    age_y: float = Field(gt=0.0, le=120.0)
    height_m: float = Field(ge=0.4, le=2.5)
    mass_kg: float = Field(ge=2.0, le=300.0)


class GarmentSpec(_Frozen):
    """One garment on one slot: a library material, its insulation, and its colour.

    ``clo`` is the garment's own intrinsic insulation (ISO 9920; 1 clo = 0.155 m² K W⁻¹).
    ``colour_rgb`` is linear reflectance in 0..1 for the RGB companion; HU.5 derives solar
    absorptivity and NIR reflectance from it and leaves the long-wave emissivity alone.
    ``covers`` narrows or widens the slot's default coverage; it must name segments.
    """

    material: str = Field(min_length=1)
    clo: float = Field(ge=0.0, le=5.0)
    colour_rgb: tuple[float, float, float] | None = None
    covers: list[str] | None = None

    @field_validator("colour_rgb")
    @classmethod
    def _unit_interval(
        cls, v: tuple[float, float, float] | None
    ) -> tuple[float, float, float] | None:
        if v is not None and not all(0.0 <= c <= 1.0 for c in v):
            raise ValueError(f"colour_rgb {v} is not in 0..1 (linear reflectance)")
        return v


class HumanSpec(_Frozen):
    """The `human:` block of a `kind: human` asset."""

    phenotype: Phenotype
    garments: dict[str, GarmentSpec] = Field(default_factory=dict)
    #: Which asset axis the body faces, for the Chest/Back split. Blender's front view looks
    #: along +Y, so a character facing the viewer faces -Y; MPFB exports that way.
    forward_axis: Literal["+x", "-x", "+y", "-y"] = "-y"
    rig: str | None = None

    @property
    def materials(self) -> frozenset[str]:
        return frozenset(g.material for g in self.garments.values())

    def check_against(self, schema: BodySchema) -> None:
        """Refuse a garment on a slot the body has not got, or covering what is not a segment."""
        for slot, g in self.garments.items():
            if slot not in schema.garment_slots:
                raise ValueError(
                    f"garment on unknown slot {slot!r}; the body schema has "
                    f"{sorted(schema.garment_slots)}"
                )
            if g.covers is not None:
                bad = [c for c in g.covers if c not in schema.segment_names]
                if bad:
                    raise ValueError(f"garment {slot!r} covers {bad}, which are not segments")
        if self.rig is not None and self.rig not in schema.rigs:
            raise ValueError(f"unknown rig {self.rig!r}; the body schema has {sorted(schema.rigs)}")

    def coverage(self, schema: BodySchema, slot: str) -> frozenset[str]:
        """The segments the garment on ``slot`` covers: its own list, else the slot's default."""
        g = self.garments[slot]
        if g.covers is not None:
            return frozenset(g.covers)
        return frozenset(schema.garment_slots[slot].covers)
