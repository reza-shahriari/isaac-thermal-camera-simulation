"""Material schema: one YAML per material, ``source`` required, author one optical quantity.

docs/physics-model.md §4.1, §12.3:

    schema_version: 2
    material:
      name: car_paint_black
      source: literature                # measured | literature | estimated -- drives trust
      thermal: {density_kg_m3, specific_heat_j_kgk, conductivity_w_mk, thickness_m,
                solar_absorptivity}
      optical:
        # ONE quantity -- emissivity or reflectance -- in any of three forms, layered (ADR 0175):
        spectral_emissivity: spectra/materials/car_paint_black.csv   # a curve, or a list of
        #   segments: [lw.csv, {file: sw.csv, quantity: reflectance}] (an opaque complement)
        emissivity_per_band: {mwir: 0.88}     # where the curve has no data, by band
        emissivity: 0.9                       # one grey value for everything else
        # -- or the same three keys spelled reflectance --
        transmittance_per_band: {nir: 0, swir: 0, mwir: 0, lwir: 0}   # optional, default 0
        roughness_per_band: {nir: 0.35, swir: 0.30, mwir: 0.18, lwir: 0.12}
        angular_model: {type: fresnel, n_k_file: nk/acrylic_paint.csv}
                     | {type: empirical, a, p} | {type: constant}

**Three forms, one quantity.** A camera's value is resolved wavelength by wavelength under its
own response (:meth:`irsim.materials.library.Material.band_properties`): the curve where it has
data, else the per-band value of the camera's band, else the grey value. Any non-empty subset of
the three may be authored; a material with only a grey value is grey in every band, one with only
a per-band table is today's four numbers, one with only a curve is a spectral material. What
remains forbidden is authoring a number twice: a per-band value for a band whose whole nominal
range the curve already covers is refused at load (the curve would silently overrule it).

Non-negotiable #4 as read for τ > 0 materials (ADR 0040): **one** of {ε, ρ} is authored (in any
of its forms) and τ may be authored per band; the remaining quantity is always derived
(ρ = 1 − ε − τ or ε = 1 − ρ − τ) by :mod:`irsim.materials.library`, and ε + τ > 1 (or ρ + τ > 1)
is refused. Authoring both ε and ρ is an error -- except that one *segment* of a curve may be
tabulated as the opaque complement (``quantity: reflectance`` inside ``spectral_emissivity``),
because short-wave libraries measure reflectance and long-wave ones emission; the library refuses
that segment wherever the material authors τ > 0. Band keys are free strings (a material may
declare a band no current sensor has); consumers ask for the bands they need.

docs/physics-model.md §4.1, §4.4, §12.3, §16.2, CLAUDE.md #4; ADR 0175
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
    "SpectrumSegment",
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
    #: PT.26: the in-plane conductivity a lateral operator needs. A laminate conducts along its
    #: fibres ten times better than through its thickness (CFRP: ~7 against 0.8 W/m/K), and
    #: `conductivity_w_mk` is the through-thickness value the slab solver wants. ``None`` is
    #: isotropic: the same number, which is every material written before this field existed.
    conductivity_inplane_w_mk: float | None = Field(default=None, gt=0.0)
    thickness_m: float = Field(gt=0.0)
    solar_absorptivity: Fraction

    @property
    def inplane_conductivity_w_mk(self) -> float:
        """k along the surface: the in-plane value when authored, else the through-thickness one."""
        if self.conductivity_inplane_w_mk is not None:
            return float(self.conductivity_inplane_w_mk)
        return float(self.conductivity_w_mk)

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
    #: AT.25 (ADR 0163). ``magnitude``: the authored ε_B scales the table's Fresnel *shape*
    #: (Level A as it was; a dielectric's shape barely moves). ``effective_nk``: the table's
    #: n and k are scaled by one factor until the band's normal-incidence Fresnel emissivity
    #: *is* the authored ε_B, and the angular law is that conductor's own -- for a bare, oxidised
    #: or anodised metal whose authored ε_B sits well above the clean metal's, so the scaled
    #: shape no longer clips to 1 at the limb.
    fit: Literal["magnitude", "effective_nk"] = "magnitude"


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


class SpectrumSegment(_Frozen):
    """One file of a segmented curve (ADR 0175).

    ``quantity`` says what the file tabulates. When it is the key's own quantity the segment is
    read as it stands; when it is the other one the file is the opaque complement (ε = 1 − ρ),
    which is how a short-wave reflectance measurement joins a long-wave emission one.
    """

    file: str = Field(min_length=1)
    quantity: Literal["emissivity", "reflectance"]


SpectralSource = str | list[str | SpectrumSegment]


class OpticalSpec(_Frozen):
    spectral_emissivity: SpectralSource | None = None
    spectral_reflectance: SpectralSource | None = None
    #: One grey value of the authored quantity, used wherever neither the curve nor the
    #: per-band table says anything (ADR 0175). Alone, it makes the material grey in every band.
    emissivity: Fraction | None = None
    reflectance: Fraction | None = None
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
        eps = [
            n
            for n in ("spectral_emissivity", "emissivity_per_band", "emissivity")
            if getattr(self, n) is not None
        ]
        rho = [
            n
            for n in ("spectral_reflectance", "reflectance_per_band", "reflectance")
            if getattr(self, n) is not None
        ]
        if bool(eps) == bool(rho):
            raise ValueError(
                "author exactly one optical quantity -- emissivity or reflectance, as a curve "
                "(spectral_*), per band (*_per_band) and/or one grey value -- got "
                f"{(eps + rho) or 'none'} (CLAUDE.md #4: the other is derived, never authored)"
            )
        for name in ("spectral_emissivity", "spectral_reflectance"):
            source = getattr(self, name)
            if source is None:
                continue
            if isinstance(source, list) and not source:
                raise ValueError(f"{name} must name at least one file")
            for seg in source if isinstance(source, list) else [source]:
                if isinstance(seg, str) and not seg.strip():
                    raise ValueError(f"{name} must be a file path")
        if self.transmittance_derivation is not None and self.transmittance_per_band is None:
            raise ValueError(
                "transmittance_derivation describes where transmittance_per_band came from, "
                "and this material authors no transmittance_per_band"
            )
        which = "emissivity" if eps else "reflectance"
        scalar = self.emissivity_per_band or self.reflectance_per_band
        grey = self.emissivity if self.emissivity is not None else self.reflectance
        for band, tau in (self.transmittance_per_band or {}).items():
            value = scalar.get(band) if scalar is not None else None
            # The grey value fills a band only where the table is silent, so that is the pair
            # closure has to hold for.
            if value is None:
                value = grey
            if value is not None and value + tau > 1.0 + 1e-12:
                raise ValueError(
                    f"band {band!r}: {which} {value} + transmittance {tau} > 1 "
                    "violates Kirchhoff closure (CLAUDE.md #4)"
                )
        return self

    @property
    def authored(self) -> Literal["emissivity", "reflectance"]:
        if (
            self.spectral_emissivity is not None
            or self.emissivity_per_band is not None
            or self.emissivity is not None
        ):
            return "emissivity"
        return "reflectance"

    @property
    def spectral_segments(self) -> tuple[SpectrumSegment, ...]:
        """The authored curve as segments, each naming the quantity its file tabulates."""
        source = self.spectral_emissivity or self.spectral_reflectance
        if source is None:
            return ()
        own = self.authored
        items = source if isinstance(source, list) else [source]
        return tuple(
            SpectrumSegment(file=s, quantity=own) if isinstance(s, str) else s for s in items
        )

    @property
    def spectral_file(self) -> str | None:
        """The curve's first file, or ``None`` when the material authors no curve."""
        segments = self.spectral_segments
        return segments[0].file if segments else None

    @property
    def band_table(self) -> dict[str, float] | None:
        """The per-band table of the authored quantity, if any."""
        return self.emissivity_per_band or self.reflectance_per_band

    @property
    def grey_value(self) -> float | None:
        """The grey value of the authored quantity, if any."""
        return self.emissivity if self.emissivity is not None else self.reflectance

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
