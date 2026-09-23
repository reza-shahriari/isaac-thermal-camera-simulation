"""Engine-free material-mapping resolver: which thermal material does an imported prim get?

No asset ships with thermal properties (thermal-materials skill, §13.3). The resolver assigns a
library material to each prim record by precedence

    1. explicit override   (a ``thermal:material`` USD attribute -- must name a library material)
    2. asset material map  (exact material name, from ``configs/assets/<asset>.yaml``; ADR 0128)
    3. semantic class      (``car_body`` → car_paint_black, from ``configs/materials/mapping.yaml``)
    4. material-name pattern (ordered case-insensitive globs: ``*glass*`` → glass_windshield)
    5. loud miss           (material id 0 = UNMAPPED, **recorded**, never silently defaulted)

Rung 2 exists because a *global* glob is a statement about every asset and an imported asset needs
statements about itself. ``*white*`` → ``car_paint_white`` is a reasonable default for a car park
and wrong for a moulded drone shell, which is ``abs_plastic_white`` -- same optics, half the areal
heat capacity, so it swings twice as fast. Authoring that as a global pattern would change every
other scene; authoring it per asset changes one. ADR 0047's own "Revisit when" clause named this
file ("then a per-asset mapping file"); ADR 0128 records the precedence and why it sits above the
semantic rung.

Ids come from the packed table's sorted-name order (M7.18), so they are stable across loads and
identical to what the kernel table uses. The prim records are plain data (path, material name,
semantic class, override) -- the Isaac adapter dumps them to JSON and this module never imports
an engine; ``scripts/audit_materials.py`` reports coverage on such a dump (ADR 0047).

docs/physics-model.md §13.3; thermal-materials skill "The USD import problem"
"""

from __future__ import annotations

import fnmatch
import os
import pathlib
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from irsim.materials.table import UNMAPPED_MATERIAL_ID, UNMAPPED_NAME

__all__ = [
    "MAPPING_PATH",
    "MAPPING_SCHEMA_VERSION",
    "DEFAULT_COVERAGE_THRESHOLD",
    "PrimRecord",
    "PatternRule",
    "MappingRules",
    "AssetMapping",
    "Resolution",
    "AuditReport",
    "MaterialResolver",
    "load_mapping_rules",
    "load_asset_mapping",
    "audit",
]

MAPPING_PATH = (
    pathlib.Path(__file__).resolve().parents[3] / "configs" / "materials" / "mapping.yaml"
)
ASSETS_DIR = pathlib.Path(__file__).resolve().parents[3] / "configs" / "assets"
MAPPING_SCHEMA_VERSION = 1
ASSET_SCHEMA_VERSION = 1
DEFAULT_COVERAGE_THRESHOLD = 0.95  # ADR 0047
Rule = Literal["override", "asset", "semantic", "pattern", "miss"]


@dataclass(frozen=True)
class PrimRecord:
    """One renderable prim as the adapter dumps it (engine-free)."""

    path: str
    material_name: str | None = None
    semantic_class: str | None = None
    override: str | None = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PrimRecord:
        unknown = set(d) - {"path", "material_name", "semantic_class", "override"}
        if unknown or "path" not in d:
            raise ValueError(f"prim record needs 'path' and only known keys; got {sorted(d)}")
        return cls(
            path=str(d["path"]),
            material_name=None if d.get("material_name") is None else str(d["material_name"]),
            semantic_class=None if d.get("semantic_class") is None else str(d["semantic_class"]),
            override=None if d.get("override") is None else str(d["override"]),
        )


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class PatternRule(_Frozen):
    match: str = Field(min_length=1)  # glob on the prim's material name, case-insensitive
    material: str = Field(min_length=1)


class MappingRules(_Frozen):
    semantic: dict[str, str] = Field(default_factory=dict)
    patterns: list[PatternRule] = Field(default_factory=list)
    coverage_threshold: float = Field(default=DEFAULT_COVERAGE_THRESHOLD, ge=0.0, le=1.0)

    @field_validator("patterns")
    @classmethod
    def _unique_patterns(cls, v: list[PatternRule]) -> list[PatternRule]:
        seen = [p.match.lower() for p in v]
        if len(set(seen)) != len(seen):
            raise ValueError("duplicate pattern in mapping rules")
        return v

    @property
    def targets(self) -> frozenset[str]:
        return frozenset(self.semantic.values()) | frozenset(p.material for p in self.patterns)


class MappingConfig(_Frozen):
    schema_version: int
    mapping: MappingRules

    @field_validator("schema_version")
    @classmethod
    def _version(cls, v: int) -> int:
        if v != MAPPING_SCHEMA_VERSION:
            raise ValueError(f"mapping schema_version {v} != {MAPPING_SCHEMA_VERSION}")
        return v


def load_mapping_rules(
    path: str | os.PathLike[str] | None = None, known_materials: Iterable[str] | None = None
) -> MappingRules:
    """Read mapping.yaml; with ``known_materials`` given, every target must be one of them."""
    p = pathlib.Path(path) if path is not None else MAPPING_PATH
    rules = MappingConfig.model_validate(yaml.safe_load(p.read_text(encoding="utf-8"))).mapping
    if known_materials is not None:
        unknown = rules.targets - set(known_materials)
        if unknown:
            raise ValueError(f"{p}: rules target unknown materials {sorted(unknown)}")
    return rules


class AssetMapping(_Frozen):
    """Per-asset truth about one imported model (ADR 0128).

    ``materials`` maps a **source** material name -- the name the DCC or the FBX carried, matched
    exactly and case-insensitively -- onto a library material. It is not a glob: an asset map is
    authored after looking at the asset, so a miss here should be a miss, not a near-match.

    ``scale_to_metres`` records the factor the source units need. It is metadata for the prep tool
    and the reviewer, not something the resolver applies; it lives here because it is a fact about
    this asset that would otherwise be known only to whoever ran the importer once.
    """

    name: str = Field(min_length=1)
    source_file: str | None = None
    scale_to_metres: float = Field(default=1.0, gt=0.0)
    materials: dict[str, str] = Field(default_factory=dict)

    @field_validator("materials")
    @classmethod
    def _no_case_collisions(cls, v: dict[str, str]) -> dict[str, str]:
        lowered = [k.lower() for k in v]
        if len(set(lowered)) != len(lowered):
            raise ValueError("asset material map has two keys differing only by case")
        return v

    @property
    def targets(self) -> frozenset[str]:
        return frozenset(self.materials.values())

    def lookup(self, material_name: str | None) -> str | None:
        """The library material for a source material name, or None."""
        if material_name is None:
            return None
        target = material_name.lower()
        for key, value in self.materials.items():
            if key.lower() == target:
                return value
        return None


class AssetConfig(_Frozen):
    schema_version: int
    asset: AssetMapping

    @field_validator("schema_version")
    @classmethod
    def _version(cls, v: int) -> int:
        if v != ASSET_SCHEMA_VERSION:
            raise ValueError(f"asset schema_version {v} != {ASSET_SCHEMA_VERSION}")
        return v


def load_asset_mapping(
    path: str | os.PathLike[str], known_materials: Iterable[str] | None = None
) -> AssetMapping:
    """Read ``configs/assets/<asset>.yaml``.

    A bare name (``"phantom4"``) is resolved inside :data:`ASSETS_DIR`, so a scene or a CLI can
    name an asset without knowing the layout. With ``known_materials`` given, every target must be
    one of them -- the same guard :func:`load_mapping_rules` applies, for the same reason: a
    mapping that names a material nobody has is a miss that looks like a hit.
    """
    p = pathlib.Path(path)
    if not p.suffix and not p.exists():
        p = ASSETS_DIR / f"{p.name}.yaml"
    asset = AssetConfig.model_validate(yaml.safe_load(p.read_text(encoding="utf-8"))).asset
    if known_materials is not None:
        unknown = asset.targets - set(known_materials)
        if unknown:
            raise ValueError(f"{p}: asset map targets unknown materials {sorted(unknown)}")
    return asset


@dataclass(frozen=True)
class Resolution:
    path: str
    material: str | None  # None on a miss
    material_id: int  # UNMAPPED_MATERIAL_ID on a miss
    rule: Rule
    matched: str | None = None  # the semantic class / pattern / override that fired

    @property
    def mapped(self) -> bool:
        return self.material_id != UNMAPPED_MATERIAL_ID


class MaterialResolver:
    """Rules + the table's id order → resolutions with recorded misses."""

    def __init__(
        self,
        rules: MappingRules,
        names: Sequence[str],
        asset: AssetMapping | None = None,
    ) -> None:
        if not names or names[0] != UNMAPPED_NAME:
            raise ValueError("names must be the packed table's names (index 0 = UNMAPPED)")
        unknown = rules.targets - set(names[1:])
        if unknown:
            raise ValueError(f"mapping rules target materials not in the table: {sorted(unknown)}")
        if asset is not None:
            unknown_asset = asset.targets - set(names[1:])
            if unknown_asset:
                raise ValueError(
                    f"asset {asset.name!r} maps to materials not in the table: "
                    f"{sorted(unknown_asset)}"
                )
        self._rules = rules
        self._asset = asset
        self._names = tuple(names)
        self._ids = {name: i for i, name in enumerate(self._names)}
        self.misses: list[Resolution] = []

    @property
    def rules(self) -> MappingRules:
        return self._rules

    @property
    def asset(self) -> AssetMapping | None:
        return self._asset

    def id_for(self, material: str) -> int:
        return self._ids[material]

    def resolve(self, prim: PrimRecord) -> Resolution:
        if prim.override is not None:
            if prim.override not in self._ids or prim.override == UNMAPPED_NAME:
                raise ValueError(
                    f"{prim.path}: override {prim.override!r} is not a library material "
                    f"(have {self._names[1:]})"
                )
            return Resolution(
                prim.path, prim.override, self._ids[prim.override], "override", prim.override
            )
        if self._asset is not None:
            material = self._asset.lookup(prim.material_name)
            if material is not None:
                return Resolution(
                    prim.path, material, self._ids[material], "asset", prim.material_name
                )
        if prim.semantic_class is not None and prim.semantic_class in self._rules.semantic:
            material = self._rules.semantic[prim.semantic_class]
            return Resolution(
                prim.path, material, self._ids[material], "semantic", prim.semantic_class
            )
        if prim.material_name is not None:
            name = prim.material_name.lower()
            for rule in self._rules.patterns:
                if fnmatch.fnmatchcase(name, rule.match.lower()):
                    return Resolution(
                        prim.path, rule.material, self._ids[rule.material], "pattern", rule.match
                    )
        miss = Resolution(prim.path, None, UNMAPPED_MATERIAL_ID, "miss", None)
        self.misses.append(miss)
        return miss

    def resolve_all(self, prims: Iterable[PrimRecord]) -> list[Resolution]:
        return [self.resolve(p) for p in prims]


@dataclass(frozen=True)
class AuditReport:
    total: int
    mapped: int
    threshold: float
    misses_by_material_name: dict[str, int] = field(default_factory=dict)
    miss_paths: tuple[str, ...] = ()
    by_rule: dict[str, int] = field(default_factory=dict)

    @property
    def coverage(self) -> float:
        return self.mapped / self.total if self.total else 1.0

    @property
    def passed(self) -> bool:
        return self.coverage >= self.threshold

    def render(self) -> str:
        lines = [
            f"materials audit: {self.mapped}/{self.total} prims mapped, "
            f"coverage {self.coverage:.1%} (threshold {self.threshold:.0%})"
        ]
        lines.append("  by rule: " + ", ".join(f"{k}={v}" for k, v in sorted(self.by_rule.items())))
        if self.misses_by_material_name:
            lines.append("  UNMAPPED (material id 0, rendered magenta), grouped by material name:")
            for name, n in sorted(
                self.misses_by_material_name.items(), key=lambda kv: (-kv[1], kv[0])
            ):
                lines.append(f"    {n:5d}  {name}")
        lines.append(
            "  RESULT: " + ("PASS" if self.passed else "FAIL -- add mapping rules or overrides")
        )
        return "\n".join(lines)


def audit(
    prims: Iterable[PrimRecord], resolver: MaterialResolver, threshold: float | None = None
) -> AuditReport:
    thr = resolver.rules.coverage_threshold if threshold is None else float(threshold)
    prims = list(prims)
    resolutions = resolver.resolve_all(prims)
    by_rule = Counter(r.rule for r in resolutions)
    misses = [p for p, r in zip(prims, resolutions, strict=True) if not r.mapped]
    grouped = Counter((p.material_name or "<no material>") for p in misses)
    return AuditReport(
        total=len(prims),
        mapped=sum(1 for r in resolutions if r.mapped),
        threshold=thr,
        misses_by_material_name=dict(grouped),
        miss_paths=tuple(p.path for p in misses),
        by_rule={str(k): int(v) for k, v in by_rule.items()},
    )
