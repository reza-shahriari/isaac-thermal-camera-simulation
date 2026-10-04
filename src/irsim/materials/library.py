"""Material library: load ``configs/materials/*.yaml``, derive the third optical quantity (§4.1).

For a material and a band the library returns the band-effective triple (ε_B, ρ_B, τ_B) with
ε_B + ρ_B + τ_B = 1 by construction (ADR 0040, ADR 0046). The authored quantity is the
Planck-weighted band average (ADR 0010, the only sanctioned reduction) of a spectrum assembled
wavelength by wavelength under the camera's own response (ADR 0175): the curve where a segment
has data, else the band's per-band value, else the grey value -- so a 6-13 µm camera and a
7.5-13.5 µm one read different numbers off the same curve, and a material without a curve reads
its band number exactly as before. τ_B is the authored per-band transmittance (0 when absent);
the remaining quantity is derived. An authored value that leaves no room (ε_B + τ_B > 1) is refused
here, after the band average, so a spectral file cannot smuggle a closure violation through.

Band averages need a spectral response: pass the sensor's (``SpectralResponse``) or let the
library use the nominal top-hat of a registry band (``NOMINAL_RANGES_UM``). A band the registry
does not know (``lwir_wide``) is fine for scalar tables and needs an explicit response for
spectral ones.

docs/physics-model.md §4.1, §4.4, §12.3, CLAUDE.md #4
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import yaml
from numpy.typing import NDArray

from irsim.config.bands import NOMINAL_RANGES_UM
from irsim.config.loader import file_sha256, resolve_data_dir
from irsim.config.materials import FresnelAngular, MaterialConfig, MaterialSpec
from irsim.materials.spectra import PropertySpectrum, SpectralCurve, load_property_spectrum
from irsim.radiometry.band_average import WeightingForm, band_average
from irsim.radiometry.spectral_response import SpectralResponse

__all__ = [
    "MATERIAL_DIR",
    "CLOSURE_TOL",
    "BandProperties",
    "Material",
    "MaterialLibrary",
    "nominal_response",
]

MATERIAL_DIR = pathlib.Path(__file__).resolve().parents[3] / "configs" / "materials"
CLOSURE_TOL = 1e-6


@dataclass(frozen=True)
class BandProperties:
    """ε + ρ + τ = 1 on band-effective values; ``authored`` says which one came from the file."""

    emissivity: float
    reflectance: float
    transmittance: float
    authored: Literal["emissivity", "reflectance"]
    #: Fraction of the band's Planck × response weight the curve supplied (ADR 0175); the rest
    #: came from the per-band or grey value. 0 for a material without a curve. Report it: below
    #: 1 the value carries the flat-within-the-gap assumption.
    curve_fraction: float = 0.0

    def __post_init__(self) -> None:
        for name in ("emissivity", "reflectance", "transmittance"):
            v = getattr(self, name)
            if not 0.0 <= v <= 1.0:
                raise ValueError(f"{name} = {v} outside [0, 1]")
        closure = self.emissivity + self.reflectance + self.transmittance
        if abs(closure - 1.0) > CLOSURE_TOL:
            raise ValueError(f"Kirchhoff closure ε + ρ + τ = {closure} != 1 (CLAUDE.md #4)")


def nominal_response(band: str) -> SpectralResponse:
    """A unit top-hat over the registry's nominal range for a standard band id."""
    if band not in NOMINAL_RANGES_UM:
        raise KeyError(
            f"band {band!r} has no nominal range; pass the sensor's SpectralResponse explicitly"
        )
    lo, hi = NOMINAL_RANGES_UM[band]
    return SpectralResponse(
        wavelength_um=np.array([lo, hi]),
        response=np.array([1.0, 1.0]),
        source_path=f"<nominal top-hat {band}>",
        sha256="",
    )


@dataclass(frozen=True)
class Material:
    spec: MaterialSpec
    path: pathlib.Path
    curve: SpectralCurve | None  # the authored spectral quantity, if any
    n_k_path: pathlib.Path | None

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def spectrum(self) -> PropertySpectrum | None:
        """The curve's file when it is one plain file (the pre-ADR-0175 form), else ``None``."""
        if self.curve is None or len(self.curve.segments) != 1 or self.curve.complement[0]:
            return None
        return self.curve.segments[0]

    def band_properties(
        self,
        band: str,
        response: SpectralResponse | None = None,
        form: WeightingForm = "energy",
        t_ref_k: float = 300.0,
    ) -> BandProperties:
        """(ε_B, ρ_B, τ_B), closed to 1e-6; raises when the authored values leave no room.

        Pass the camera's ``response`` (its ``Band.response``) so the value is that camera's;
        without one a registry band uses its nominal top-hat. ADR 0175 gives the layering.
        """
        optical = self.spec.optical
        tau = optical.transmittance(band)
        table = optical.band_table or {}
        fill = table.get(band, optical.grey_value)
        curve_fraction = 0.0
        if self.curve is None or (
            response is None and band not in NOMINAL_RANGES_UM and band in table
        ):
            if fill is None:
                raise KeyError(f"{self.name}: no {optical.authored} authored for band {band!r}")
            authored_value = float(fill)
        else:
            resp = response if response is not None else nominal_response(band)
            lo, hi = resp.support_um
            # threshold 0: the whole response file must be covered, not just where R is
            # appreciable. A library material is reused across cameras, so the stricter rule is
            # the right one here -- and it is the *same* rule, in one place (M7.3).
            if self.curve.covers_interval(lo, hi):
                curve = self.curve

                def spectrum(grid: NDArray[np.float64]) -> NDArray[np.float64]:
                    return curve.values(grid)

                curve_fraction = 1.0
            elif fill is None:
                raise ValueError(
                    f"{self.name}: the curve covers {self.curve.spans_text()} um, band "
                    f"{band!r} needs {lo}-{hi} um: extend the "
                    f"table, or author a per-band or grey {optical.authored} to fill the rest "
                    "(ADR 0175) rather than extrapolate"
                )
            else:
                curve = self.curve
                filler = float(fill)

                def spectrum(grid: NDArray[np.float64]) -> NDArray[np.float64]:
                    v = curve.values(grid)
                    return np.where(np.isnan(v), filler, v)

                def coverage(grid: NDArray[np.float64]) -> NDArray[np.float64]:
                    return curve.covered(grid).astype(np.float64)

                # Linear in the spectrum, so the indicator's band average *is* the curve's share.
                curve_fraction = band_average(resp, coverage, t_ref_k, form)
            authored_value = float(band_average(resp, spectrum, t_ref_k, form))
        if authored_value + tau > 1.0 + CLOSURE_TOL:
            raise ValueError(
                f"{self.name} band {band!r}: {optical.authored} {authored_value:.4f} + "
                f"transmittance {tau:.4f} > 1 violates Kirchhoff closure (CLAUDE.md #4)"
            )
        derived = max(0.0, 1.0 - authored_value - tau)
        if optical.authored == "emissivity":
            return BandProperties(authored_value, derived, tau, "emissivity", curve_fraction)
        return BandProperties(derived, authored_value, tau, "reflectance", curve_fraction)

    @property
    def declared_bands(self) -> frozenset[str]:
        return self.spec.optical.declared_bands

    def content_hash(self) -> str:
        """SHA-256 of the validated spec with data file paths replaced by their content hashes."""
        dump: dict[str, Any] = self.spec.model_dump(mode="json")
        opt = dump["optical"]
        if self.curve is not None:
            hashes = [file_sha256(seg.path) for seg in self.curve.segments]
            for key in ("spectral_emissivity", "spectral_reflectance"):
                source = opt.get(key)
                if isinstance(source, str):
                    opt[key] = hashes[0]
                elif isinstance(source, list):
                    opt[key] = [
                        h if isinstance(item, str) else {**item, "file": h}
                        for h, item in zip(hashes, source, strict=True)
                    ]
        if self.n_k_path is not None:
            opt["angular_model"]["n_k_file"] = file_sha256(self.n_k_path)
        canonical = json.dumps(dump, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _resolve(raw: str, data_dir: pathlib.Path, what: str) -> pathlib.Path:
    p = pathlib.Path(raw).expanduser()
    if not p.is_absolute():
        p = data_dir / p
    p = p.resolve()
    if not p.is_file():
        raise FileNotFoundError(f"{what} = {raw!r} resolves to {p}, which does not exist")
    return p


def _load_curve(spec: MaterialSpec, root: pathlib.Path) -> SpectralCurve | None:
    """Load the curve's segments and refuse what the schema could not see without the files.

    Two refusals need the wavelengths (ADR 0175): a per-band value for a registry band whose whole
    nominal range the curve already covers (a second authoring the curve would silently
    overrule), and a complement segment (ε = 1 − ρ) over a band where the material authors τ > 0
    (there the complement is 1 − ρ − τ, not 1 − ρ).
    """
    optical = spec.optical
    segments = optical.spectral_segments
    if not segments:
        return None
    files = tuple(
        load_property_spectrum(_resolve(seg.file, root, f"{spec.name}.optical.spectral_*"))
        for seg in segments
    )
    curve = SpectralCurve(
        segments=files,
        complement=tuple(seg.quantity != optical.authored for seg in segments),
    )
    nominal: dict[str, tuple[float, float]] = {str(k): v for k, v in NOMINAL_RANGES_UM.items()}
    for band in sorted(optical.band_table or {}):
        if band in nominal and curve.covers_interval(*nominal[band]):
            lo, hi = nominal[band]
            raise ValueError(
                f"{spec.name}: {optical.authored}_per_band[{band!r}] is authored twice -- the "
                f"curve already covers {band}'s nominal {lo}-{hi} um and would overrule it. "
                "Delete the per-band value (ADR 0175)"
            )
    for seg_file, flip in zip(files, curve.complement, strict=True):
        if not flip:
            continue
        s_lo, s_hi = seg_file.support_um
        for band, tau in (optical.transmittance_per_band or {}).items():
            rng = nominal.get(band)
            if tau > 0.0 and rng is not None and rng[0] < s_hi and rng[1] > s_lo:
                raise ValueError(
                    f"{spec.name}: {seg_file.path.name} is an opaque complement but the material "
                    f"transmits {tau} in {band!r}, which it overlaps (CLAUDE.md #4)"
                )
    return curve


def load_material(
    path: str | os.PathLike[str], data_dir: str | os.PathLike[str] | None = None
) -> Material:
    p = pathlib.Path(path)
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "materials" in raw:
        raise ValueError(
            f"{p}: the 'materials:' wrapper is not the file layout -- one YAML per material with a "
            "top-level 'material:' block (ADR 0040)"
        )
    spec = MaterialConfig.model_validate(raw).material
    if p.parent.resolve() == MATERIAL_DIR.resolve() and spec.name != p.stem:
        raise ValueError(f"material name {spec.name!r} does not match file name {p.stem!r}")
    root = resolve_data_dir(data_dir)
    curve = _load_curve(spec, root)
    n_k = None
    if isinstance(spec.optical.angular_model, FresnelAngular):
        n_k = _resolve(
            spec.optical.angular_model.n_k_file, root, f"{spec.name}.angular_model.n_k_file"
        )
    return Material(spec=spec, path=p.resolve(), curve=curve, n_k_path=n_k)


class MaterialLibrary(Mapping[str, Material]):
    """All materials of a directory, by name."""

    def __init__(self, materials: Mapping[str, Material]) -> None:
        self._materials = dict(materials)

    @classmethod
    def load(
        cls,
        material_dir: str | os.PathLike[str] | None = None,
        data_dir: str | os.PathLike[str] | None = None,
    ) -> MaterialLibrary:
        root = pathlib.Path(material_dir) if material_dir is not None else MATERIAL_DIR
        materials = {}
        for path in sorted(root.glob("*.yaml")):
            if path.stem == "mapping":
                continue
            m = load_material(path, data_dir)
            if m.name in materials:
                raise ValueError(f"duplicate material name {m.name!r}")
            materials[m.name] = m
        if not materials:
            raise FileNotFoundError(f"no material YAML files under {root}")
        return cls(materials)

    def __getitem__(self, name: str) -> Material:
        try:
            return self._materials[name]
        except KeyError:
            raise KeyError(f"unknown material {name!r}; have {sorted(self._materials)}") from None

    def __iter__(self) -> Iterator[str]:
        return iter(self._materials)

    def __len__(self) -> int:
        return len(self._materials)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._materials)

    def content_hash(self) -> str:
        h = hashlib.sha256()
        for name in sorted(self._materials):
            h.update(name.encode())
            h.update(self._materials[name].content_hash().encode())
        return h.hexdigest()
