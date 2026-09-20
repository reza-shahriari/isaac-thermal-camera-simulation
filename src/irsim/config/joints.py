"""The joint-conductance table: ``configs/thermal/joints.yaml`` (TC.4, ADR 0097).

A scene's ``links:`` may name a joint (``h_c`` in W m⁻² K⁻¹, multiplied by the link's footprint)
or a fastener (``G`` in W/K, multiplied by a count) instead of typing a number, so the value and
its provenance live in one place. Every entry says whether it is MEASURED (a published
measurement of this kind of joint) or ESTIMATED (an engineering default, or a measurement under
other conditions), and names its source.

**The range guard is the point.** A contact conductance is 1e2–1e6 W m⁻² K⁻¹ across the whole
literature (Holman's 2e3–2e5 sits inside it); a per-K typo such as ``1e-3`` would turn a bolted
joint into an insulator and the scene would render plausibly with the bracket cold. The loader
refuses it here and the scene schema refuses it again on an inline ``h_c_w_m2_k``.

docs/physics-model.md §6.4; spec issue S42; survey 2026-09-18 (Voller & Tirovic 2007, DSPE,
Hasselström & Nilsson 2012).
"""

from __future__ import annotations

import os
import pathlib
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

__all__ = [
    "H_C_RANGE_W_M2_K",
    "G_RANGE_W_K",
    "JOINTS_SCHEMA_VERSION",
    "JOINTS_PATH",
    "FastenerSpec",
    "JointSpec",
    "JointTable",
    "load_joint_table",
]

JOINTS_SCHEMA_VERSION = 1
JOINTS_PATH = pathlib.Path(__file__).resolve().parents[3] / "configs" / "thermal" / "joints.yaml"

#: Contact conductance, W m⁻² K⁻¹: below 1e2 is a gap, above 1e6 is a weld.
H_C_RANGE_W_M2_K = (1e2, 1e6)
#: Per-fastener conductance, W/K: a small bolt is ~1, a heavy through-bolt tens.
G_RANGE_W_K = (1e-3, 1e3)

Status = Literal["MEASURED", "ESTIMATED"]


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


def check_h_c(value: float, what: str) -> float:
    lo, hi = H_C_RANGE_W_M2_K
    if not lo <= value <= hi:
        raise ValueError(
            f"{what}: h_c = {value} W m^-2 K^-1 is outside {lo:g}..{hi:g}; a contact conductance "
            "below 1e2 is a gap and above 1e6 a weld -- check for a per-K typo (1e-3 for 1e3)"
        )
    return float(value)


def check_g(value: float, what: str) -> float:
    lo, hi = G_RANGE_W_K
    if not lo <= value <= hi:
        raise ValueError(f"{what}: G = {value} W/K is outside {lo:g}..{hi:g} for one fastener")
    return float(value)


class JointSpec(_Frozen):
    """A contact conductance per unit area, with where the number came from."""

    h_c_w_m2_k: float
    status: Status
    source: str = Field(min_length=10)

    @field_validator("h_c_w_m2_k")
    @classmethod
    def _range(cls, v: float) -> float:
        return check_h_c(v, "joint")


class FastenerSpec(_Frozen):
    """A total conductance per fastener, with where the number came from."""

    g_w_k: float
    status: Status
    source: str = Field(min_length=10)

    @field_validator("g_w_k")
    @classmethod
    def _range(cls, v: float) -> float:
        return check_g(v, "fastener")


class JointTable(_Frozen):
    schema_version: int
    joints: dict[str, JointSpec] = Field(default_factory=dict)
    fasteners: dict[str, FastenerSpec] = Field(default_factory=dict)

    @field_validator("schema_version")
    @classmethod
    def _version(cls, v: int) -> int:
        if v != JOINTS_SCHEMA_VERSION:
            raise ValueError(f"joints schema_version {v} != {JOINTS_SCHEMA_VERSION}")
        return v

    def joint(self, name: str) -> JointSpec:
        try:
            return self.joints[name]
        except KeyError as exc:
            raise KeyError(f"unknown joint {name!r}; the table has {sorted(self.joints)}") from exc

    def fastener(self, name: str) -> FastenerSpec:
        try:
            return self.fasteners[name]
        except KeyError as exc:
            raise KeyError(
                f"unknown fastener {name!r}; the table has {sorted(self.fasteners)}"
            ) from exc


def load_joint_table(path: str | os.PathLike[str] | None = None) -> JointTable:
    """The project's table by default; a path for a test or a study's own values."""
    p = pathlib.Path(path) if path is not None else JOINTS_PATH
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    return JointTable.model_validate(raw)
