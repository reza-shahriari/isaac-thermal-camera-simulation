"""An occupation's outfit is data: a kit of generated garments and equipment (HU.11).

docs/physics-model.md §6.1 (the person's surfaces), §16.2 (clothing materials); roadmap HU.11;
ADR 0196 (an occupation is garments plus equipment), ADR 0198 (this file).

MakeHuman's CC0 clothes packs hold a T-shirt, trousers, shoes and hats; they hold no hi-vis vest,
no plate carrier, no turnout coat and no breathing cylinder. HU.7 and HU.8 cut those from the body
itself in ``scripts/make_human.py``, one Python function per occupation. That does not scale to
occupations, and the roadmap's criterion for HU.11 is that an occupation is "a YAML of garments
and parts with no new code". So the pieces are described here and cut by one generic routine.

A **kit** (``configs/humans/kits/<name>.yaml``) lists:

* ``clothes`` -- MakeHuman clothes assets to dress a slot with, as ``--garment SLOT=ASSET`` did;
* ``pieces`` -- generated pieces, each named ``garment_<slot>`` or ``equipment_<name>`` exactly as
  the labelled asset and its config name them, with an RGB base colour and one shape:

  - ``shell``: the body's own skin faces whose dominant bone matches one of ``bones`` (shell-style
    patterns over the rig's bone names), pushed ``offset_m`` out along the normals. It keeps the
    faces between horizontal planes at ``bands`` (fractions of stature), or between planes
    perpendicular to each bone at ``rings`` (fractions of that bone's length, head to tail; a
    little past 0 or 1 reaches over the joint). A ring is how a band goes round a forearm that is
    not vertical in the rest pose.
  - ``box`` / ``cylinder``: a rigid shape standing ``behind`` an earlier piece (on its back, the
    body facing -y), centred at ``centre_z`` of stature, ``gap_m`` clear of it -- a pack, a
    breathing cylinder.

What a piece *is* to the cameras -- its library material, its insulation -- is not in the kit. It
is in the asset's ``human:`` block, which owns the physics (ADR 0195, ADR 0196); the kit owns the
geometry only. :meth:`HumanKit.check_against` holds the two together: every pattern names a bone
of the rig, every garment a slot of the body, and an asset built from a kit wears exactly its
slots and carries exactly its equipment (``tests/unit/test_kits.py``).
"""

from __future__ import annotations

import fnmatch
import os
import pathlib
import re

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from irsim.config.humans import BodySchema, HumanSpec

__all__ = [
    "KIT_SCHEMA_VERSION",
    "KITS_DIR",
    "HumanKit",
    "KitBox",
    "KitCylinder",
    "KitPiece",
    "KitShell",
    "kit_names",
    "load_kit",
]

KIT_SCHEMA_VERSION = 1
KITS_DIR = pathlib.Path(__file__).resolve().parents[3] / "configs" / "humans" / "kits"
_PIECE_NAME = re.compile(r"^(garment|equipment)_[a-z][a-z0-9_]*$")


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


def _ordered(pairs: list[tuple[float, float]], lo: float, hi: float, what: str) -> None:
    for a, b in pairs:
        if not (lo <= a < b <= hi):
            raise ValueError(f"{what} ({a}, {b}) must satisfy {lo} <= from < to <= {hi}")


class KitShell(_Frozen):
    """Skin faces of the matched bones, between planes, pushed ``offset_m`` out."""

    bones: list[str] = Field(min_length=1)
    offset_m: float = Field(gt=0.0, le=0.2)
    bands: list[tuple[float, float]] | None = None
    rings: list[tuple[float, float]] | None = None

    @model_validator(mode="after")
    def _one_kind_of_cut(self) -> KitShell:
        if (self.bands is None) == (self.rings is None):
            raise ValueError("a shell is cut by `bands` (stature) or `rings` (along a bone), one")
        if self.bands is not None:
            _ordered(self.bands, -0.05, 1.05, "band")
        if self.rings is not None:
            _ordered(self.rings, -0.25, 1.25, "ring")
        return self


class KitBox(_Frozen):
    """A rigid box on the back of an earlier piece: width (x), depth (y), height (z), metres."""

    size_m: tuple[float, float, float]
    centre_z: float = Field(gt=0.0, lt=1.0)
    behind: str
    gap_m: float = Field(default=0.0, ge=0.0, le=0.2)

    @field_validator("size_m")
    @classmethod
    def _positive(cls, v: tuple[float, float, float]) -> tuple[float, float, float]:
        if not all(0.0 < s <= 2.0 for s in v):
            raise ValueError(f"box size {v} must be positive metres")
        return v


class KitCylinder(_Frozen):
    """A rigid upright cylinder on the back of an earlier piece."""

    radius_m: float = Field(gt=0.0, le=0.5)
    length_m: float = Field(gt=0.0, le=2.0)
    centre_z: float = Field(gt=0.0, lt=1.0)
    behind: str
    gap_m: float = Field(default=0.0, ge=0.0, le=0.2)
    #: segments round the circumference
    vertices: int = Field(default=32, ge=8, le=256)


class KitPiece(_Frozen):
    """One generated piece: ``garment_<slot>`` or ``equipment_<name>``, one shape, one colour."""

    name: str
    colour_rgb: tuple[float, float, float]
    shell: KitShell | None = None
    box: KitBox | None = None
    cylinder: KitCylinder | None = None

    @field_validator("name")
    @classmethod
    def _named(cls, v: str) -> str:
        if not _PIECE_NAME.match(v):
            raise ValueError(f"piece {v!r} must be named garment_<slot> or equipment_<name>")
        return v

    @field_validator("colour_rgb")
    @classmethod
    def _unit(cls, v: tuple[float, float, float]) -> tuple[float, float, float]:
        if not all(0.0 <= c <= 1.0 for c in v):
            raise ValueError(f"colour_rgb {v} is not in 0..1")
        return v

    @model_validator(mode="after")
    def _one_shape(self) -> KitPiece:
        shapes = [s for s in (self.shell, self.box, self.cylinder) if s is not None]
        if len(shapes) != 1:
            raise ValueError(f"piece {self.name!r} needs exactly one of shell, box, cylinder")
        return self

    @property
    def kind(self) -> str:
        return self.name.split("_", 1)[0]

    @property
    def key(self) -> str:
        """The slot of a garment, the name of an equipment item."""
        return self.name.split("_", 1)[1]

    @property
    def behind(self) -> str | None:
        shape = self.box or self.cylinder
        return None if shape is None else shape.behind


class HumanKit(_Frozen):
    """An occupation's outfit: MakeHuman clothes per slot plus generated pieces."""

    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    rig: str = "mpfb_game_engine"
    clothes: dict[str, str] = Field(default_factory=dict)
    pieces: list[KitPiece] = Field(default_factory=list)

    @model_validator(mode="after")
    def _consistent(self) -> HumanKit:
        names = [p.name for p in self.pieces]
        dup = sorted({n for n in names if names.count(n) > 1})
        if dup:
            raise ValueError(f"kit {self.name!r}: pieces {dup} appear twice")
        both = sorted(set(self.clothes) & {p.key for p in self.pieces if p.kind == "garment"})
        if both:
            raise ValueError(
                f"kit {self.name!r}: slots {both} are both a MakeHuman asset and a piece"
            )
        seen: set[str] = set()
        for p in self.pieces:
            if p.behind is not None and p.behind not in seen:
                raise ValueError(
                    f"kit {self.name!r}: {p.name!r} stands behind {p.behind!r}, which is not an "
                    "earlier piece"
                )
            seen.add(p.name)
        return self

    @property
    def garment_slots(self) -> frozenset[str]:
        """Every slot the kit dresses, from MakeHuman clothes or a generated piece."""
        return frozenset(self.clothes) | {p.key for p in self.pieces if p.kind == "garment"}

    @property
    def equipment(self) -> frozenset[str]:
        return frozenset(p.key for p in self.pieces if p.kind == "equipment")

    def check_against(self, schema: BodySchema) -> None:
        """Refuse a slot the body has not got, or a bone pattern that matches no bone of the rig."""
        if self.rig not in schema.rigs:
            raise ValueError(f"kit {self.name!r}: unknown rig {self.rig!r}")
        bones = list(schema.rigs[self.rig].bones)
        for slot in sorted(self.garment_slots):
            if slot not in schema.garment_slots:
                raise ValueError(
                    f"kit {self.name!r}: slot {slot!r} is not one of {sorted(schema.garment_slots)}"
                )
        for p in self.pieces:
            if p.shell is None:
                continue
            for pattern in p.shell.bones:
                if not fnmatch.filter(bones, pattern):
                    raise ValueError(
                        f"kit {self.name!r}: {p.name!r} cuts by bone {pattern!r}, which matches "
                        f"no bone of rig {self.rig!r}"
                    )

    def check_asset(self, human: HumanSpec) -> None:
        """An asset built from this kit wears exactly its slots and carries its equipment."""
        worn, carried = set(human.garments), set(human.equipment)
        if worn != set(self.garment_slots):
            raise ValueError(
                f"kit {self.name!r} dresses {sorted(self.garment_slots)}, the asset {sorted(worn)}"
            )
        if carried != set(self.equipment):
            raise ValueError(
                f"kit {self.name!r} carries {sorted(self.equipment)}, the asset {sorted(carried)}"
            )


class _KitFile(_Frozen):
    schema_version: int
    kit: HumanKit

    @field_validator("schema_version")
    @classmethod
    def _version(cls, v: int) -> int:
        if v != KIT_SCHEMA_VERSION:
            raise ValueError(f"kit schema_version {v} != {KIT_SCHEMA_VERSION}")
        return v


def kit_names() -> list[str]:
    """The committed kits, by name."""
    return sorted(p.stem for p in KITS_DIR.glob("*.yaml"))


def load_kit(name_or_path: str | os.PathLike[str]) -> HumanKit:
    """Read a kit by name (``configs/humans/kits/<name>.yaml``) or by path."""
    p = pathlib.Path(name_or_path)
    if p.suffix != ".yaml":
        p = KITS_DIR / f"{name_or_path}.yaml"
    kit = _KitFile.model_validate(yaml.safe_load(p.read_text(encoding="utf-8"))).kit
    if p.parent == KITS_DIR and kit.name != p.stem:
        raise ValueError(f"{p.name}: kit name {kit.name!r} differs from the file's")
    return kit
