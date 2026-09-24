"""Material schema: one YAML per material, ``source`` required, author one optical property.

docs/physics-model.md §4.1, §12.3:

    schema_version: 1
    material:
      name: car_paint_black
      source: literature                # measured | literature | estimated -- drives trust
      thermal: {density_kg_m3, specific_heat_j_kgk, conductivity_w_mk, thickness_m,
                solar_absorptivity}
      optical:
        spectral_emissivity: spectra/materials/car_paint_black.csv     # exactly ONE of these four:
        # spectral_reflectance / emissivity_per_band / reflectance_per_band
        transmittance_per_band: {nir: 0, swir: 0, mwir: 0, lwir: 0}   # optional, default 0
        roughness_per_band: {nir: 0.35, swir: 0.30, mwir: 0.18, lwir: 0.12}
        angular_model: {type: fresnel, n_k_file: nk/acrylic_paint.csv}
                     | {type: empirical, a, p} | {type: constant}

Non-negotiable #4 as read for τ > 0 materials (ADR 0040): **one** of {ε, ρ} is authored (spectral
file or scalar per band) and τ may be authored per band; the remaining quantity is always derived
(ρ = 1 − ε − τ or ε = 1 − ρ − τ) by :mod:`irsim.materials.library`, and ε + τ > 1 (or ρ + τ > 1)
is refused. Authoring both ε and ρ in any form is an error. Band keys are free strings (a material
may declare a band no current sensor has); consumers ask for the bands they need.

docs/physics-model.md §4.1, §4.4, §12.3, §16.2, CLAUDE.md #4
"""

from __future__ import annotations

import re
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator, model_validator

__all__ = [
    "MATERIAL_SCHEMA_VERSION",
    "ThermalSpec",
    "FresnelAngular",
    "EmpiricalAngular",
    "ConstantAngular",
    "AngularModel",
    "OpticalSpec",
    "MaterialSpec",
    "MaterialConfig",
    "SURFACE_TREATMENTS",
    "SurfaceTreatment",
]

MATERIAL_SCHEMA_VERSION = 2

#: §4.5's surface states. Emissivity is a property of the **surface**, not of the substance under
#: it: one measurement campaign on aluminium window profiles reports 0.834-0.856 for anodised
#: exterior surfaces and 0.055-0.82 across untreated cavities in the same frames [R39] -- one
#: metal, one paper, a fifteen-fold spread. So "aluminium" is not a material specification, and a
#: file that does not name its state is not either.
#:
#: The first seven are §4.5's own list. ``as_manufactured`` and ``natural`` are added for the
#: substances that have no *treatment*: a moulded plastic, a woven fabric and a cast tyre leave the
#: works in the state they will keep, and skin, snow, soil and water were never treated at all.
#: Both are still *states* and still have to be chosen, because the alternative -- an optional
#: field, or a free string -- is the field being left blank on exactly the materials whose state is
#: least obvious.
SURFACE_TREATMENTS: tuple[str, ...] = (
    "polished",
    "machined",
    "oxidised",
    "anodised",
    "painted",
    "sandblasted",
    "weathered",
    "as_manufactured",
    "natural",
)
SurfaceTreatment = Literal[
    "polished",
    "machined",
    "oxidised",
    "anodised",
    "painted",
    "sandblasted",
    "weathered",
    "as_manufactured",
    "natural",
]
_BAND_KEY = re.compile(r"^[a-z][a-z0-9_]*$")
Fraction = Annotated[float, Field(ge=0.0, le=1.0)]


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ThermalSpec(_Frozen):
    """§6.2 / §12.3 thermal block (area-normalised capacity = ρ c_p δ)."""

    density_kg_m3: float = Field(gt=0.0)
    specific_heat_j_kgk: float = Field(gt=0.0)
    conductivity_w_mk: float = Field(gt=0.0)
    thickness_m: float = Field(gt=0.0)
    solar_absorptivity: Fraction

    @property
    def heat_capacity_j_m2_k(self) -> float:
        return self.density_kg_m3 * self.specific_heat_j_kgk * self.thickness_m

    @property
    def thermal_inertia(self) -> float:
        """P = sqrt(k ρ c_p), J m⁻² K⁻¹ s⁻½ (§6.3)."""
        return float(
            (self.conductivity_w_mk * self.density_kg_m3 * self.specific_heat_j_kgk) ** 0.5
        )


class FresnelAngular(_Frozen):
    type: Literal["fresnel"]
    n_k_file: str = Field(min_length=1)  # relative to the data root (data/nk/)


class EmpiricalAngular(_Frozen):
    """Level B: ε(θ) = ε₀ [1 − a (1 − cos θ)^p]."""

    type: Literal["empirical"]
    a: float = Field(ge=0.0, le=1.0)
    p: float = Field(gt=0.0, le=12.0)


class ConstantAngular(_Frozen):
    type: Literal["constant"]


AngularModel = Annotated[
    FresnelAngular | EmpiricalAngular | ConstantAngular, Field(discriminator="type")
]


def _check_band_keys(values: dict[str, float] | None, what: str) -> dict[str, float] | None:
    if values is None:
        return None
    if not values:
        raise ValueError(f"{what} must name at least one band")
    for key in values:
        if not _BAND_KEY.match(key):
            raise ValueError(f"{what}: band key {key!r} is not a lower-case identifier")
    return values


class OpticalSpec(_Frozen):
    spectral_emissivity: str | None = None
    spectral_reflectance: str | None = None
    emissivity_per_band: dict[str, Fraction] | None = None
    reflectance_per_band: dict[str, Fraction] | None = None
    transmittance_per_band: dict[str, Fraction] | None = None
    roughness_per_band: dict[str, Fraction] | None = None
    #: Where ``transmittance_per_band`` came from: ``derived`` (it is the Beer-Lambert
    #: transmittance of ``angular_model``'s n/k table over ``thermal.thickness_m``, and
    #: `tests/unit/test_transmittance_derivation.py` checks that it is) or
    #: ``authored: <spec issue>`` (it is not, and the named issue says why). Omitted means
    #: ``derived`` for a material that has a table. The point is that a number which cannot
    #: be recomputed from checked-in data has to *say so*, naming a live issue, rather than
    #: sitting beside a table that looks like its source (spec issue S13).
    transmittance_derivation: str | None = None
    angular_model: AngularModel = ConstantAngular(type="constant")

    @field_validator(
        "emissivity_per_band",
        "reflectance_per_band",
        "transmittance_per_band",
        "roughness_per_band",
    )
    @classmethod
    def _bands(cls, v: dict[str, float] | None, info: ValidationInfo) -> dict[str, float] | None:
        return _check_band_keys(v, str(info.field_name))

    @field_validator("transmittance_derivation")
    @classmethod
    def _derivation(cls, v: str | None) -> str | None:
        """``derived``, or ``authored: <issue>`` naming a row of `docs/spec-issues.md`."""
        if v is None:
            return None
        text = v.strip()
        if text == "derived":
            return text
        match = re.fullmatch(r"authored:\s*([ST]\d+)", text)
        if match is None:
            raise ValueError(
                f"transmittance_derivation must be 'derived' or 'authored: <issue>' with an "
                f"issue id like S13, got {v!r}. A number that cannot be recomputed from "
                "checked-in data has to name the issue that explains why."
            )
        return f"authored: {match.group(1)}"

    @model_validator(mode="after")
    def _exactly_one_authored(self) -> OpticalSpec:
        authored = [
            name
            for name in (
                "spectral_emissivity",
                "spectral_reflectance",
                "emissivity_per_band",
                "reflectance_per_band",
            )
            if getattr(self, name) is not None
        ]
        if len(authored) != 1:
            raise ValueError(
                "author exactly one optical property -- spectral_emissivity, spectral_reflectance, "
                f"emissivity_per_band or reflectance_per_band -- got {authored or 'none'} "
                "(CLAUDE.md #4: the other is derived, never authored)"
            )
        for name in ("spectral_emissivity", "spectral_reflectance"):
            path = getattr(self, name)
            if path is not None and not path.strip():
                raise ValueError(f"{name} must be a file path")
        if self.transmittance_derivation is not None and self.transmittance_per_band is None:
            raise ValueError(
                "transmittance_derivation describes where transmittance_per_band came from, "
                "and this material authors no transmittance_per_band"
            )
        scalar = self.emissivity_per_band or self.reflectance_per_band
        if scalar is not None and self.transmittance_per_band is not None:
            for band, tau in self.transmittance_per_band.items():
                if band in scalar and scalar[band] + tau > 1.0 + 1e-12:
                    which = "emissivity" if self.emissivity_per_band else "reflectance"
                    raise ValueError(
                        f"band {band!r}: {which} {scalar[band]} + transmittance {tau} > 1 "
                        "violates Kirchhoff closure (CLAUDE.md #4)"
                    )
        return self

    @property
    def authored(self) -> Literal["emissivity", "reflectance"]:
        if self.spectral_emissivity is not None or self.emissivity_per_band is not None:
            return "emissivity"
        return "reflectance"

    @property
    def spectral_file(self) -> str | None:
        return self.spectral_emissivity or self.spectral_reflectance

    def transmittance(self, band: str) -> float:
        return float((self.transmittance_per_band or {}).get(band, 0.0))

    @property
    def declared_bands(self) -> frozenset[str]:
        keys: set[str] = set()
        for d in (
            self.emissivity_per_band,
            self.reflectance_per_band,
            self.transmittance_per_band,
            self.roughness_per_band,
        ):
            if d:
                keys.update(d)
        return frozenset(keys)


class MaterialSpec(_Frozen):
    """One substance **in one surface state** (§4.5).

    ``surface_treatment`` is required and has no default, which is the whole of `AT.17`. A default
    would be a state somebody did not choose, and the failure it guards is silent: this library
    shipped `bare_aluminium` described as *polished* and valued as *oxidised* -- 0.04 against 0.09
    at 8 µm, a factor of 2.25 in the term that decides whether a surface reports itself or the sky
    -- and nothing could see the contradiction because neither number had a state beside it.
    """

    name: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    source: Literal["measured", "literature", "estimated"]
    #: The surface state the optics were measured on, from §4.5's vocabulary.
    surface_treatment: SurfaceTreatment
    reference: str = ""
    description: str = ""
    thermal: ThermalSpec
    optical: OpticalSpec


class MaterialConfig(_Frozen):
    schema_version: int
    material: MaterialSpec

    @field_validator("schema_version")
    @classmethod
    def _version(cls, v: int) -> int:
        if v != MATERIAL_SCHEMA_VERSION:
            raise ValueError(f"material schema_version {v} != {MATERIAL_SCHEMA_VERSION}")
        return v
