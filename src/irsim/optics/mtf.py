"""MTF cascade components (docs/physics-model.md §8.3, [R20], [R21]).

    MTF_sys(ξ) = MTF_diff · MTF_aberr · MTF_det · MTF_motion · MTF_defocus · MTF_elec

Units: spatial frequency ξ in **cycles per mm** on the focal plane, lengths in mm. Conventions
fixed by ADR 0059: one Gaussian stands for aberration + defocus (fitted from a slant edge);
MTF_elec = 1; MTF_det is *provided by the supersample box filter* in the pipeline and must not be
applied again as a PSF; MTF_motion applies to photon detectors only (bolometer smear is the τ_th
IIR, M9.1). ``np.sinc`` is the normalised sinc, sin(πx)/(πx): the argument is ``w·ξ``, not
``π·w·ξ`` (the double-π bug turns 0.637 at Nyquist into 0.198).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "cutoff_frequency_cyc_per_mm",
    "nyquist_frequency_cyc_per_mm",
    "mtf_diffraction",
    "mtf_detector",
    "mtf_motion",
    "mtf_gaussian",
    "mtf_system",
    "aberration_sigma_for_mtf",
]

FloatArray = NDArray[np.float64]


def cutoff_frequency_cyc_per_mm(wavelength_um: float, f_number: float) -> float:
    """ξ_c = 1/(λ F) with λ in mm: 100 cyc/mm for 10 µm at F/1 (§8.3)."""
    if wavelength_um <= 0.0 or f_number <= 0.0:
        raise ValueError("wavelength and f-number must be positive")
    return 1.0 / (wavelength_um * 1e-3 * f_number)


def nyquist_frequency_cyc_per_mm(pitch_um: float) -> float:
    """ξ_N = 1/(2p): 41.667 cyc/mm for a 12 µm pitch (§8.3)."""
    if pitch_um <= 0.0:
        raise ValueError("pitch must be positive")
    return 1.0 / (2.0 * pitch_um * 1e-3)


def mtf_diffraction(xi_cyc_per_mm: object, wavelength_um: float, f_number: float) -> FloatArray:
    """Incoherent circular-aperture diffraction MTF: (2/π)[arccos x − x√(1−x²)] with x = ξ/ξ_c,
    and 0 beyond the cut-off."""
    xi = np.abs(np.asarray(xi_cyc_per_mm, dtype=np.float64))
    x = np.clip(xi / cutoff_frequency_cyc_per_mm(wavelength_um, f_number), 0.0, 1.0)
    mtf = 2.0 / np.pi * (np.arccos(x) - x * np.sqrt(1.0 - x * x))
    return np.asarray(np.where(xi < cutoff_frequency_cyc_per_mm(wavelength_um, f_number), mtf, 0.0))


def mtf_detector(xi_cyc_per_mm: object, width_mm: float) -> FloatArray:
    """|sinc(w ξ)| for a square detector of width w (first zero at ξ = 1/w)."""
    if width_mm <= 0.0:
        raise ValueError("detector width must be positive")
    xi = np.asarray(xi_cyc_per_mm, dtype=np.float64)
    return np.asarray(np.abs(np.sinc(width_mm * xi)))  # np.sinc(x) = sin(pi x)/(pi x)


def mtf_motion(xi_cyc_per_mm: object, velocity_mm_per_s: float, t_int_s: float) -> FloatArray:
    """|sinc(v t_int ξ)| for linear image-plane motion during the integration; 1 when v t = 0."""
    if t_int_s < 0.0:
        raise ValueError("integration time cannot be negative")
    xi = np.asarray(xi_cyc_per_mm, dtype=np.float64)
    smear = abs(velocity_mm_per_s) * t_int_s
    if smear == 0.0:
        return np.ones_like(xi)
    return np.asarray(np.abs(np.sinc(smear * xi)))


def mtf_gaussian(xi_cyc_per_mm: object, sigma_mm: float) -> FloatArray:
    """exp(−2π² σ² ξ²): the MTF of a Gaussian PSF of standard deviation σ (aberration/defocus)."""
    if sigma_mm < 0.0:
        raise ValueError("sigma cannot be negative")
    xi = np.asarray(xi_cyc_per_mm, dtype=np.float64)
    return np.asarray(np.exp(-2.0 * np.pi**2 * sigma_mm**2 * xi**2))


def mtf_system(
    xi_cyc_per_mm: object,
    wavelength_um: float,
    f_number: float,
    detector_width_mm: float | None = None,
    sigma_mm: float | None = None,
    velocity_mm_per_s: float | None = None,
    t_int_s: float | None = None,
) -> FloatArray:
    """Product of the enabled terms (MTF_elec = 1). Pass ``detector_width_mm`` only when the
    detector footprint is *not* already provided by the supersample box filter (ADR 0059)."""
    xi = np.asarray(xi_cyc_per_mm, dtype=np.float64)
    mtf = mtf_diffraction(xi, wavelength_um, f_number)
    if detector_width_mm is not None:
        mtf = mtf * mtf_detector(xi, detector_width_mm)
    if sigma_mm is not None and sigma_mm > 0.0:
        mtf = mtf * mtf_gaussian(xi, sigma_mm)
    if velocity_mm_per_s is not None and t_int_s is not None:
        mtf = mtf * mtf_motion(xi, velocity_mm_per_s, t_int_s)
    return np.asarray(mtf, dtype=np.float64)


def aberration_sigma_for_mtf(
    target_mtf: float, xi_cyc_per_mm: float, wavelength_um: float, f_number: float
) -> float:
    """The σ (in **µm**) whose Gaussian, times diffraction, gives ``target_mtf`` at ``ξ``.

    The inverse of the lens half of :func:`mtf_system`, and it exists so a config can carry a
    *derived* number rather than a pasted one. A datasheet quotes one figure -- "MTF at Nyquist,
    nominal, on-axis" -- and that figure is the **whole lens**, diffraction included; authoring
    ``aberration_sigma_um`` therefore means solving

        MTF_diff(ξ) · exp(−2π² σ² ξ²) = target

    for σ, which is what this does. Doing it by hand once and writing the answer into a YAML is
    how a number stops being checkable: nobody can tell later whether 1.65 µm came from the
    datasheet or from a fit to a golden.

    The detector footprint is deliberately **not** included. It is the box filter's (ADR 0059) and
    the datasheet figure is the lens alone, so folding it in here would double-count it.

    Raises when the target is unreachable: above the diffraction limit at that frequency no
    aberration can help, and a lens quoted above its own diffraction MTF is a datasheet to
    re-read rather than a σ to solve for.
    """
    if not 0.0 < target_mtf < 1.0:
        raise ValueError("target_mtf must lie in (0, 1)")
    if xi_cyc_per_mm <= 0.0:
        raise ValueError("the frequency must be positive")
    diffraction = float(mtf_diffraction(xi_cyc_per_mm, wavelength_um, f_number))
    if diffraction <= target_mtf:
        raise ValueError(
            f"diffraction alone gives {diffraction:.4f} at {xi_cyc_per_mm:g} cyc/mm, which is at "
            f"or below the {target_mtf:g} asked for: no aberration Gaussian can raise an MTF, so "
            "either the figure is for a different frequency or the f-number and wavelength are"
        )
    ratio = target_mtf / diffraction
    sigma_mm = float(np.sqrt(-np.log(ratio) / (2.0 * np.pi**2 * xi_cyc_per_mm**2)))
    return sigma_mm * 1e3
