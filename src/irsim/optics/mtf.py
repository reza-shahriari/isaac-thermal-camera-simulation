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
    "DEFOCUS_MODELS",
    "band_average_otf",
    "bessel_j1",
    "lens_otf",
    "mtf_defocus_geometric",
    "mtf_defocus_hopkins",
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


# --- defocus (OC.2) ----------------------------------------------------------------------------
# docs/physics-model.md §8.3 names MTF_defocus and declines to define it ("in practice fit a
# Gaussian ... rather than deriving it"). These are the two models that *do* derive it, plus the
# Gaussian kept as the cheap ablation. `w020_um` comes from `irsim.optics.defocus` (OC.1).
#
# Hopkins is the whole lens, diffraction included: at W020 = 0 it reduces analytically to
# `mtf_diffraction`, which `test_defocus_otf.py` checks to 1e-9. It therefore REPLACES the
# diffraction factor in the cascade rather than multiplying onto it. The aberration Gaussian
# (ADR 0117) is separate and still multiplies, because it was fitted at the in-focus condition.

#: Simpson nodes for the Hopkins and Bessel quadratures. The Hopkins integrand oscillates about
#: a/(2π) times over the interval and a = 8π W020 s/λ, so 2049 nodes hold ~1e-9 out to W020 = 5λ.
_QUAD_NODES = 2049


def _simpson(y: FloatArray, dx: float) -> FloatArray:
    """Composite Simpson along the last axis; ``y`` must have an odd number of samples."""
    w = np.ones(y.shape[-1])
    w[1:-1:2], w[2:-1:2] = 4.0, 2.0
    return np.asarray(np.tensordot(y, w, axes=([-1], [0])) * dx / 3.0)


def bessel_j1(x: object) -> FloatArray:
    """J₁ by its integral representation, (1/π)∫₀^π cos(θ − x sin θ) dθ.

    NumPy has no Bessel functions and SciPy is not a dependency of the physics core, so this is
    the quadrature rather than a rational approximation -- it is exact to the node count and has
    no validity band to get wrong.
    """
    xa = np.asarray(x, dtype=np.float64)
    theta = np.linspace(0.0, np.pi, _QUAD_NODES)
    integrand = np.cos(theta - xa[..., None] * np.sin(theta))
    return np.asarray(_simpson(integrand, float(theta[1] - theta[0])) / np.pi)


def mtf_defocus_geometric(xi_cyc_per_mm: object, blur_circle_um: float) -> FloatArray:
    """The OTF of a uniform disk of diameter ``c``: ``2 J₁(π c ξ)/(π c ξ)``, signed.

    Geometric optics only: valid once W020 > 2λ, which
    :func:`irsim.optics.defocus.geometric_regime_blur_um` puts at 168 µm for a Boson at F/1.0.
    Below that it understates the contrast badly -- 0.173 against Hopkins' 0.356 at Nyquist for a
    target at 10 m -- and it is kept as a selectable model, not as a default.
    """
    if blur_circle_um < 0.0:
        raise ValueError("blur-circle diameter cannot be negative")
    xi = np.asarray(xi_cyc_per_mm, dtype=np.float64)
    if blur_circle_um == 0.0:
        return np.ones_like(xi)
    x = np.pi * (blur_circle_um * 1e-3) * xi
    out = np.ones_like(x)
    nz = x != 0.0
    out[nz] = 2.0 * bessel_j1(x[nz]) / x[nz]
    return out


def mtf_defocus_hopkins(
    xi_cyc_per_mm: object, wavelength_um: float, f_number: float, w020_um: float
) -> FloatArray:
    """Hopkins' defocus OTF for a circular pupil -- **diffraction and defocus in one term**::

        OTF(s) = 4/(πa) ∫₀^√(1−s²) sin[a(√(1−y²) − s)] dy,   a = 8π W020 s / λ

    with ``s = ξ/ξ_c`` and ``ξ_c = 1/(λF)``. Substituting ``y = √(1−s²) u`` puts the s-dependent
    limit into the integrand so one fixed quadrature grid serves every frequency.

    The result is the **OTF**, not the MTF: it goes negative past the first zero, which is the
    contrast reversal (spurious resolution) a real defocused lens shows and a Gaussian never can.
    """
    if w020_um < 0.0:
        raise ValueError("W020 cannot be negative")
    xi = np.abs(np.asarray(xi_cyc_per_mm, dtype=np.float64))
    s = np.clip(xi / cutoff_frequency_cyc_per_mm(wavelength_um, f_number), 0.0, 1.0)
    if w020_um == 0.0:
        return mtf_diffraction(xi, wavelength_um, f_number)
    a = 8.0 * np.pi * w020_um * s / float(wavelength_um)
    root = np.sqrt(np.clip(1.0 - s * s, 0.0, None))
    u = np.linspace(0.0, 1.0, _QUAD_NODES)
    inner = np.sqrt(np.clip(1.0 - (root[..., None] * u) ** 2, 0.0, None)) - s[..., None]
    integral = _simpson(np.sin(a[..., None] * inner), float(u[1] - u[0])) * root
    with np.errstate(divide="ignore", invalid="ignore"):
        otf = np.where(a > 0.0, 4.0 / (np.pi * np.where(a > 0.0, a, 1.0)) * integral, 1.0)
    return np.asarray(np.where(xi < cutoff_frequency_cyc_per_mm(wavelength_um, f_number), otf, 0.0))


#: The defocus models `OC.4` will expose as a config switch. `none` is the present behaviour.
DEFOCUS_MODELS = ("none", "gaussian", "geometric", "hopkins")


def lens_otf(
    xi_cyc_per_mm: object,
    wavelength_um: float,
    f_number: float,
    w020_um: float = 0.0,
    model: str = "hopkins",
) -> FloatArray:
    """The lens OTF -- diffraction and defocus together -- under the selected model.

    ``hopkins`` is the one term; the others multiply their defocus factor onto
    :func:`mtf_diffraction`, which is what makes them wrong in the transition band rather than
    merely cheaper. ``gaussian`` matches the disk's second moment, σ = c/4, and is the ablation
    that shows what the fitted-Gaussian convention costs.
    """
    if model not in DEFOCUS_MODELS:
        raise ValueError(f"unknown defocus model {model!r}; expected one of {DEFOCUS_MODELS}")
    xi = np.asarray(xi_cyc_per_mm, dtype=np.float64)
    if model == "hopkins":
        return mtf_defocus_hopkins(xi, wavelength_um, f_number, w020_um)
    diffraction = mtf_diffraction(xi, wavelength_um, f_number)
    if model == "none" or w020_um == 0.0:
        return diffraction
    blur_um = 8.0 * float(f_number) * float(w020_um)
    if model == "geometric":
        return np.asarray(diffraction * mtf_defocus_geometric(xi, blur_um))
    return np.asarray(diffraction * mtf_gaussian(xi, blur_um / 4.0 * 1e-3))


def band_average_otf(
    xi_cyc_per_mm: object,
    wavelength_um: object,
    weight: object,
    f_number: float,
    w020_um: float = 0.0,
    model: str = "hopkins",
) -> FloatArray:
    """The lens OTF averaged across the band, weighted by ``weight`` on ``wavelength_um``.

    **Defocus itself is achromatic.** In Hopkins' ``a = 8π W020 s/λ`` with ``s = ξλF`` the
    wavelength cancels, leaving ``a = 8π W020 ξ F``; the blur circle ``c = 8 F W020`` has no λ in
    it either. So averaging does not smear the defocus zeros -- they sit at fixed frequencies.
    What is chromatic is diffraction, whose cut-off runs 133 to 74 cyc/mm across a 7.5-13.5 µm
    band. Measured at W020 = 2λ: the averaged curve differs from the 10.5 µm one by under 0.3 %
    wherever the OTF still carries contrast, and reshapes only the far tail. Worth having for the
    cut-off, not the rescue the defocus model was planned around (`test_defocus_otf.py`).

    The weight is the camera's own R(λ) (`SpectralResponse.resampled`), **not** R(λ) times the
    scene's spectral radiance. That is an approximation: the true weight depends on what is in
    front of the camera, so a correct-per-scene OTF would have to be rebuilt per frame. It is the
    convention every published MTF bench uses, and the error it carries is bounded by how much the
    in-band radiance slope tilts the average -- see the ADR.
    """
    lam = np.asarray(wavelength_um, dtype=np.float64)
    w = np.asarray(weight, dtype=np.float64)
    if lam.shape != w.shape or lam.ndim != 1 or lam.size < 2:
        raise ValueError("wavelength and weight must be matching 1-D arrays of at least two points")
    if np.any(lam <= 0.0) or np.any(w < 0.0) or not np.any(w > 0.0):
        raise ValueError("wavelengths must be positive, weights non-negative and not all zero")
    xi = np.asarray(xi_cyc_per_mm, dtype=np.float64)
    stack = np.stack([lens_otf(xi, float(x), f_number, w020_um, model) for x in lam], axis=-1)
    return np.asarray(np.trapezoid(stack * w, lam, axis=-1) / np.trapezoid(w, lam))
