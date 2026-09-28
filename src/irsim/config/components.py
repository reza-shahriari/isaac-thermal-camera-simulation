"""The component library: ``configs/components/<name>.yaml`` (AI.12).

docs/physics-model.md §6.6 (heat sources inside a body), §12.3; ADR 0072 (the instantaneous
nodes these replace), AI.11 (the hidden parts that name them), TC.11 (the solve that uses them).

A hidden part of an asset -- a brushless motor, an ESC, a LiPo pack, a flight controller, a
piston engine, a small turbine, an exhaust line -- is a lump of mass that turns some of the
power it takes into heat. This library holds, per component, the four numbers a thermal node
needs and where each came from: **mass** and **specific heat** (its capacity), the **rated
dissipation** (heat at rated power) and the **idle dissipation** (heat when it is on but not
working), plus the faces it heats and the library material its case is made of. Every entry
carries its provenance: MEASURED when the number is a published figure for this component,
ESTIMATED when it is an engineering default or a figure for a similar part.

**The consistency guard is the point.** An entry states its rated input power and its cited
efficiency as well as its dissipation, and the loader refuses an entry whose dissipation is not
``P_in · (1 − η)`` to 5 %: a number typed from memory that disagrees with the efficiency it
cites is exactly the error a plausible-looking render would hide. It likewise refuses an entry
without mass (a source with no thermal inertia is the instantaneous node being retired) or
without a source, and a case material the material library does not have.
"""

from __future__ import annotations

import os
import pathlib
from collections.abc import Iterable, Mapping
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

__all__ = [
    "COMPONENTS_DIR",
    "COMPONENT_SCHEMA_VERSION",
    "DISSIPATION_TOLERANCE",
    "ComponentLibrary",
    "ComponentSpec",
    "load_component",
    "load_component_library",
]

COMPONENT_SCHEMA_VERSION = 1
COMPONENTS_DIR = pathlib.Path(__file__).resolve().parents[3] / "configs" / "components"
#: How far the stated rated dissipation may sit from ``P_in (1 − η)`` before the entry is refused.
DISSIPATION_TOLERANCE = 0.05

Status = Literal["MEASURED", "ESTIMATED"]
Kind = Literal[
    "motor", "esc", "battery", "controller", "piston_engine", "turbine", "exhaust", "other"
]


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ComponentSpec(_Frozen):
    """One component: its capacity, what it dissipates, what it heats, and where that came from."""

    name: str = Field(min_length=1)
    kind: Kind = "other"
    description: str = ""
    mass_kg: float = Field(gt=0.0)
    specific_heat_j_kgk: float = Field(gt=0.0)
    #: Electrical or fuel power into the component at its rating, W.
    rated_power_in_w: float = Field(ge=0.0)
    #: The share of that power that leaves as useful work (shaft, thrust, delivered charge).
    efficiency: float = Field(ge=0.0, le=1.0)
    #: Heat at the rating, W -- must equal ``rated_power_in_w · (1 − efficiency)`` to 5 %.
    dissipation_rated_w: float = Field(ge=0.0)
    #: Heat when powered but not working (a controller's quiescent draw, an idling engine), W.
    dissipation_idle_w: float = Field(default=0.0, ge=0.0)
    #: The faces or parts this component heats, by role: what TC.11 bolts it to.
    heats: list[str] = Field(default_factory=list)
    #: The library material of its case, or ``None`` when it is buried and radiates to nothing.
    material: str | None = None
    status: Status
    source: str = Field(min_length=20)

    @model_validator(mode="after")
    def _dissipation_matches_the_efficiency(self) -> ComponentSpec:
        expect = self.rated_power_in_w * (1.0 - self.efficiency)
        if expect == 0.0 and self.dissipation_rated_w == 0.0:
            return self
        scale = max(expect, self.dissipation_rated_w)
        if abs(self.dissipation_rated_w - expect) > DISSIPATION_TOLERANCE * scale:
            raise ValueError(
                f"component {self.name!r}: dissipation_rated_w {self.dissipation_rated_w:g} W is "
                f"not rated_power_in_w × (1 − efficiency) = {expect:g} W within 5 %; one of the "
                "three numbers is wrong, and the source should say which"
            )
        if self.dissipation_idle_w > self.dissipation_rated_w and self.dissipation_rated_w > 0.0:
            raise ValueError(f"component {self.name!r}: idle dissipation exceeds rated")
        return self

    @property
    def capacity_j_k(self) -> float:
        """``m · c_p``, J/K: the thermal inertia a hidden part inherits."""
        return float(self.mass_kg) * float(self.specific_heat_j_kgk)

    def dissipation_w(self, duty: float) -> float:
        """Heat at ``duty`` ∈ [0, 1] of rated: idle at 0, rated at 1, linear between."""
        d = min(max(float(duty), 0.0), 1.0)
        return float(
            self.dissipation_idle_w + d * (self.dissipation_rated_w - self.dissipation_idle_w)
        )


class _ComponentFile(_Frozen):
    schema_version: int
    component: ComponentSpec

    @field_validator("schema_version")
    @classmethod
    def _version(cls, v: int) -> int:
        if v != COMPONENT_SCHEMA_VERSION:
            raise ValueError(f"component schema_version {v} != {COMPONENT_SCHEMA_VERSION}")
        return v


class ComponentLibrary(Mapping[str, ComponentSpec]):
    """Every component under one directory, by name."""

    def __init__(self, components: Mapping[str, ComponentSpec], path: pathlib.Path) -> None:
        self._components = dict(components)
        self.path = path

    def __getitem__(self, name: str) -> ComponentSpec:
        try:
            return self._components[name]
        except KeyError as exc:
            raise KeyError(
                f"unknown component {name!r}; the library has {sorted(self._components)}"
            ) from exc

    def __iter__(self):  # type: ignore[no-untyped-def]
        return iter(self._components)

    def __len__(self) -> int:
        return len(self._components)

    def check_materials(self, known_materials: Iterable[str]) -> None:
        """Refuse an entry whose case material the material library does not have."""
        known = set(known_materials)
        for c in self._components.values():
            if c.material is not None and c.material not in known:
                raise ValueError(
                    f"component {c.name!r} names material {c.material!r}, which the material "
                    f"library does not have"
                )


def load_component(path: str | os.PathLike[str]) -> ComponentSpec:
    """One ``<name>.yaml``; the file's stem must be the component's name."""
    p = pathlib.Path(path)
    spec = _ComponentFile.model_validate(yaml.safe_load(p.read_text(encoding="utf-8"))).component
    if spec.name != p.stem:
        raise ValueError(f"{p}: the file is named {p.stem!r} but the component {spec.name!r}")
    return spec


def load_component_library(path: str | os.PathLike[str] | None = None) -> ComponentLibrary:
    """The project's library by default; a directory for a test or a study's own parts."""
    d = pathlib.Path(path) if path is not None else COMPONENTS_DIR
    files = sorted(d.glob("*.yaml"))
    if not files:
        raise FileNotFoundError(f"no component files under {d}")
    return ComponentLibrary({c.name: c for c in (load_component(f) for f in files)}, d)
