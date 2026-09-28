"""Property spectra on disk: λ (µm), value in [0, 1] -- ε(λ), ρ(λ) or τ(λ) tables (§12.3).

Same text format as the spectral-response files (``#`` comments, ``lambda_um,value`` rows,
increasing wavelength) but a *property* is a fraction, not a normalised response: it is never
rescaled to a peak of one. Files live under ``data/spectra/materials/`` (ADR 0040).
"""

from __future__ import annotations

import os
import pathlib
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from irsim.radiometry.band_average import (
    T_REF_K,
    Spectrum,
    WeightingForm,
    band_average,
    tabulated,
)
from irsim.radiometry.solar import SolarSpectrum
from irsim.radiometry.spectral_response import SpectralResponse

#: The wavelengths a solar absorptance is averaged over. Ground-level sunlight carries about 98 % of
#: its energy between these two; outside them the atmosphere has already taken it (below 0.3 µm,
#: ozone) or there is almost nothing left to take (beyond 4 µm).
SOLAR_RANGE_UM = (0.3, 4.0)

__all__ = [
    "PropertySpectrum",
    "load_property_spectrum",
    "band_effective",
    "weighting_for_fpa",
    "solar_absorptance",
    "SOLAR_RANGE_UM",
]


@dataclass(frozen=True)
class PropertySpectrum:
    wavelength_um: NDArray[np.float64]
    values: NDArray[np.float64]
    path: pathlib.Path

    @property
    def as_callable(self) -> Spectrum:
        return tabulated(self.wavelength_um, self.values)

    @property
    def support_um(self) -> tuple[float, float]:
        return float(self.wavelength_um[0]), float(self.wavelength_um[-1])

    def covers(self, response: SpectralResponse, threshold: float = 1e-3) -> bool:
        """Does this curve span everywhere the response actually responds?

        Judged on the response's ``support(threshold)`` rather than on its file extent: a
        response file padded with zeros out to 7.0 µm does not need ε(λ) there, and requiring it
        would refuse perfectly good data. ``threshold = 0`` asks for the file extent instead,
        which is the stricter rule the material library applies to authored curves.
        """
        lo_r, hi_r = response.support_um if threshold <= 0.0 else response.support(threshold)
        lo_s, hi_s = self.support_um
        return lo_s <= lo_r and hi_s >= hi_r

    def band_effective(
        self,
        response: SpectralResponse,
        t_ref_k: float = T_REF_K,
        form: WeightingForm = "energy",
        threshold: float = 1e-3,
    ) -> float:
        """Planck-weighted band-effective value of this property under ``response`` (ADR 0010).

        Refuses a response the curve does not cover. Silently extrapolating the last tabulated
        value across a band edge is the failure that matters here: it is invisible, it biases in
        whichever direction the curve happened to be heading, and for a material whose ε falls
        off a cliff at the band edge -- glass, most paints -- it is a large error that looks like
        a small one.
        """
        if not self.covers(response, threshold):
            lo_r, hi_r = response.support_um if threshold <= 0.0 else response.support(threshold)
            raise ValueError(
                f"{self.path.name} covers {self.support_um[0]}-{self.support_um[1]} um but the "
                f"response needs {lo_r:.3f}-{hi_r:.3f} um; extend the table rather than letting "
                f"the band edge be extrapolated"
            )
        return band_average(response, self.as_callable, t_ref_k, form)


def load_property_spectrum(path: str | os.PathLike[str]) -> PropertySpectrum:
    p = pathlib.Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"property spectrum {p} does not exist")
    lam: list[float] = []
    val: list[float] = []
    for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        parts = [c.strip() for c in text.replace(";", ",").split(",")]
        if len(parts) != 2:
            raise ValueError(f"{p} line {i}: expected 'lambda_um,value', got {text!r}")
        try:
            lam.append(float(parts[0]))
            val.append(float(parts[1]))
        except ValueError as exc:
            raise ValueError(f"{p} line {i}: non-numeric entry {text!r}") from exc
    if len(lam) < 2:
        raise ValueError(f"{p}: at least two rows are needed")
    wl = np.asarray(lam, dtype=np.float64)
    v = np.asarray(val, dtype=np.float64)
    if np.any(np.diff(wl) <= 0.0):
        raise ValueError(f"{p}: wavelengths must be strictly increasing")
    if np.any(v < 0.0) or np.any(v > 1.0):
        raise ValueError(f"{p}: a property spectrum lies in [0, 1] (got {v.min()}..{v.max()})")
    if wl[0] <= 0.0:
        raise ValueError(f"{p}: wavelengths must be positive micrometres")
    return PropertySpectrum(wavelength_um=wl, values=v, path=p.resolve())


def weighting_for_fpa(fpa_type: str) -> WeightingForm:
    """Which Planck weighting a band-effective property should use for this detector.

    A bolometer absorbs power, so it averages under B(λ, T); a photon detector counts photons, so
    it averages under B_q(λ, T). The two differ by hc/λ inside the integral, which for a sloped
    spectrum is a real difference and not a convention -- and the sign of the difference is fixed:
    photon weighting leans towards the long-wave end of the band, so a *falling* ε(λ) always
    averages lower under photon weighting than under energy weighting.
    """
    if fpa_type == "bolometer":
        return "energy"
    if fpa_type == "photon":
        return "photon"
    raise ValueError(f"unknown FPA type {fpa_type!r}")


def band_effective(
    curve: PropertySpectrum,
    response: SpectralResponse,
    t_ref_k: float = T_REF_K,
    form: WeightingForm = "energy",
) -> float:
    """Free-function spelling of :meth:`PropertySpectrum.band_effective`."""
    return curve.band_effective(response, t_ref_k, form)


def solar_absorptance(
    absorptance: PropertySpectrum,
    solar: SolarSpectrum,
    range_um: tuple[float, float] = SOLAR_RANGE_UM,
) -> float:
    """The fraction of sunlight an opaque surface absorbs: ∫α(λ)E(λ)dλ / ∫E(λ)dλ (§5.4, §6.2).

    For an opaque surface Kirchhoff holds wavelength by wavelength, α(λ) = ε(λ) = 1 − ρ(λ), so the
    curve passed in is the surface's spectral emissivity (or one minus its reflectance). This is
    the number ``thermal.solar_absorptivity`` stands for, and for a material that authors a
    spectral curve it is *computed* from that curve instead of being typed next to it.

    ``solar`` is normally the AM1.5 direct-normal table. That table is modelled (ADR 0064) and is
    trusted to a few percent above 0.7 µm only; in the visible its shape can be off by more, which
    moves the *weights* of this average and not the curve. The average is refused when the curve
    does not reach across ``range_um``, for the same reason band averages are: an extrapolated
    edge biases silently.
    """
    lo, hi = range_um
    s_lo, s_hi = absorptance.support_um
    if s_lo > lo or s_hi < hi:
        raise ValueError(
            f"{absorptance.path.name} covers {s_lo}-{s_hi} um; a solar absorptance needs "
            f"{lo}-{hi} um"
        )
    grid = solar.wavelength_um[(solar.wavelength_um >= lo) & (solar.wavelength_um <= hi)]
    e = np.interp(grid, solar.wavelength_um, solar.values)
    a = np.interp(grid, absorptance.wavelength_um, absorptance.values)
    return float(np.trapezoid(a * e, grid) / np.trapezoid(e, grid))
