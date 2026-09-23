"""Optical PSF synthesis from the MTF and its application at the supersampled pitch.

The PSF is built from the **optical** part of the cascade only -- MTF_diff · MTF_gauss -- as the
inverse Fourier transform of that radially symmetric, real, non-negative-definite transfer
function sampled at the supersampled pitch p/k. The detector footprint is *not* in it (the box
downsample supplies MTF_det, ADR 0020/0059), nor is motion (photon-only, applied separately;
bolometer smear is the IIR). Kernel: odd size, normalised to Σ = 1, tiny negative FFT ringing
clipped, radially symmetric. ``apply_psf`` convolves by FFT in float64 with edge replication and
casts back to the input dtype (float32 minimum; float16 refused). Convolving a uniform field
leaves it unchanged, so radiance is conserved.

docs/physics-model.md §8.3, §2 (MTF ∗ L)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from irsim.optics.mtf import band_average_otf, lens_otf, mtf_diffraction, mtf_gaussian

__all__ = [
    "optical_psf",
    "apply_psf",
    "psf_radius_samples",
    "defocus_psf",
    "defocus_radius_samples",
]

FloatArray = NDArray[np.float64]


def psf_radius_samples(
    wavelength_um: float, f_number: float, sigma_um: float, sample_pitch_um: float
) -> int:
    """Kernel half-width: the larger of 6 Airy radii (1.22 λF) and 6σ, at least 4 samples."""
    airy_um = 1.22 * wavelength_um * f_number
    extent_um = max(6.0 * airy_um, 6.0 * sigma_um)
    return max(4, int(np.ceil(extent_um / sample_pitch_um)))


def optical_psf(
    wavelength_um: float,
    f_number: float,
    sigma_aberr_um: float,
    pitch_um: float,
    supersample: int = 1,
    radius: int | None = None,
) -> FloatArray:
    """Radially symmetric PSF kernel (float64, odd size, Σ = 1) sampled at pitch/supersample."""
    if pitch_um <= 0.0 or supersample < 1 or sigma_aberr_um < 0.0:
        raise ValueError("pitch must be positive, supersample >= 1, sigma non-negative")
    sample_um = pitch_um / supersample
    r = (
        radius
        if radius is not None
        else psf_radius_samples(wavelength_um, f_number, sigma_aberr_um, sample_um)
    )
    n = 4 * r + 1  # generous FFT grid so the truncation ringing is negligible
    f = np.fft.fftfreq(n, d=sample_um * 1e-3)  # cycles per mm
    fx, fy = np.meshgrid(f, f, indexing="xy")
    fr = np.sqrt(fx * fx + fy * fy)
    mtf = mtf_diffraction(fr, wavelength_um, f_number)
    if sigma_aberr_um > 0.0:
        mtf = mtf * mtf_gaussian(fr, sigma_aberr_um * 1e-3)
    psf = np.real(np.fft.fftshift(np.fft.ifft2(mtf)))
    c = n // 2
    kernel = psf[c - r : c + r + 1, c - r : c + r + 1]
    kernel = np.clip(kernel, 0.0, None)
    total = kernel.sum()
    if total <= 0.0:
        raise ValueError("degenerate PSF")
    return np.asarray(kernel / total, dtype=np.float64)


def apply_psf(image_ss: object, kernel: FloatArray) -> NDArray[np.floating]:
    """Convolve a (H, W) image with the kernel by FFT (edge-replicated padding); dtype preserved."""
    x = np.asarray(image_ss)
    if x.dtype == np.float16:
        raise TypeError("image is float16 (non-negotiable #2)")
    if not np.issubdtype(x.dtype, np.floating):
        raise TypeError("image must be a float array")
    if x.ndim != 2:
        raise ValueError("apply_psf works on one (H, W) plane")
    k = np.asarray(kernel, dtype=np.float64)
    if k.ndim != 2 or k.shape[0] % 2 == 0 or k.shape[1] % 2 == 0:
        raise ValueError("kernel must be 2-D with odd sides")
    ry, rx = k.shape[0] // 2, k.shape[1] // 2
    padded = np.pad(x.astype(np.float64), ((ry, ry), (rx, rx)), mode="edge")
    h, w = padded.shape
    # place the kernel centre on the origin of the padded grid (circular convolution then equals
    # the linear one inside the crop because the padding is at least the kernel radius)
    kern = np.zeros((h, w), dtype=np.float64)
    kern[: k.shape[0], : k.shape[1]] = k
    # np.roll is shape-preserving, but numpy 2.x types it as widening the shape parameter
    # from (int, int) to (int, ...), so the 2-D type is restated rather than lost.
    kern = np.asarray(np.roll(kern, (-ry, -rx), axis=(0, 1)), dtype=np.float64).reshape(h, w)
    fk = np.fft.rfft2(kern)
    out = np.fft.irfft2(np.fft.rfft2(padded) * fk, s=(h, w))
    out = out[ry : ry + x.shape[0], rx : rx + x.shape[1]]
    return np.asarray(out, dtype=x.dtype)


# --- defocus (OC.3) ----------------------------------------------------------------------------


def defocus_radius_samples(
    wavelength_um: float, f_number: float, sigma_um: float, w020_um: float, sample_pitch_um: float
) -> int:
    """Kernel half-width once defocus is in play: the in-focus rule, or 0.75 c, whichever is larger.

    The geometric disk has diameter ``c = 8 F W020`` and is hard-edged, so a half-width of ``c/2``
    truncates exactly at the edge where the energy still is; 0.75 c leaves room for the diffraction
    fringes around it.
    """
    in_focus = psf_radius_samples(wavelength_um, f_number, sigma_um, sample_pitch_um)
    blur_um = 8.0 * float(f_number) * float(w020_um)
    return max(in_focus, int(np.ceil(0.75 * blur_um / sample_pitch_um)))


def defocus_psf(  # noqa: PLR0913
    wavelength_um: float,
    f_number: float,
    w020_um: float,
    sigma_aberr_um: float,
    pitch_um: float,
    supersample: int = 1,
    model: str = "hopkins",
    band: tuple[NDArray[np.float64], NDArray[np.float64]] | None = None,
    radius: int | None = None,
) -> FloatArray:
    """PSF kernel (float64, odd, Σ = 1) for a lens defocused by ``w020_um`` waves of path error.

    The in-focus case (``w020_um = 0``, ``model="hopkins"``) reproduces :func:`optical_psf` to
    7e-7 of the peak -- the radial-interpolation floor, not a modelling difference, since Hopkins
    reduces to the diffraction MTF there (ADR 0129). Switching the pipeline onto this function
    therefore changes nothing physical until a focus distance is authored, but it is **not**
    bit-identical, so `OC.5` must either keep :func:`optical_psf` on the in-focus path or accept a
    golden refresh and say so.

    ``band`` is ``(wavelengths_um, weights)`` for :func:`~irsim.optics.mtf.band_average_otf`; it
    costs one OTF evaluation per wavelength and, per ADR 0129, moves the contrast-carrying part of
    the result by under 0.3 %, so it is optional.

    The OTF is evaluated on a **radial** grid and interpolated onto the 2-D frequency plane. The
    Hopkins quadrature is 2049 nodes per sample, and a defocused kernel can be hundreds of samples
    across; evaluating it per 2-D cell would be quadratically wasteful for a radially symmetric
    function.
    """
    if pitch_um <= 0.0 or supersample < 1 or sigma_aberr_um < 0.0 or w020_um < 0.0:
        raise ValueError("pitch > 0, supersample >= 1, sigma and W020 non-negative")
    sample_um = pitch_um / supersample
    r = (
        radius
        if radius is not None
        else defocus_radius_samples(wavelength_um, f_number, sigma_aberr_um, w020_um, sample_um)
    )
    n = 4 * r + 1
    f = np.fft.fftfreq(n, d=sample_um * 1e-3)  # cycles per mm
    fx, fy = np.meshgrid(f, f, indexing="xy")
    fr = np.sqrt(fx * fx + fy * fy)

    radial = np.linspace(0.0, float(fr.max()), max(2048, 4 * n))
    if band is None:
        otf_radial = lens_otf(radial, wavelength_um, f_number, w020_um, model)
    else:
        otf_radial = band_average_otf(radial, band[0], band[1], f_number, w020_um, model)
    otf = np.interp(fr, radial, otf_radial)
    if sigma_aberr_um > 0.0:
        otf = otf * mtf_gaussian(fr, sigma_aberr_um * 1e-3)

    psf = np.real(np.fft.fftshift(np.fft.ifft2(otf)))
    c = n // 2
    kernel = np.clip(psf[c - r : c + r + 1, c - r : c + r + 1], 0.0, None)
    total = kernel.sum()
    if total <= 0.0:
        raise ValueError("degenerate PSF")
    return np.asarray(kernel / total, dtype=np.float64)


class DefocusKernelBank:
    """Cached PSF kernels on a quantised W020 grid (`OC.5`, ADR 0129).

    A kernel costs a 2049-node Simpson quadrature per radial sample, which is fine once per scene
    and not fine once per frame of a thousand-frame sequence. W020 is quantised so that the blur
    circle it describes moves by less than a quarter of a supersample cell between neighbours --
    below what the box filter can resolve, so the quantisation is invisible in the output -- and the
    kernel for each quantised value is built once and kept.
    """

    def __init__(  # noqa: PLR0913
        self,
        wavelength_um: float,
        f_number: float,
        sigma_aberr_um: float,
        pitch_um: float,
        supersample: int = 1,
        model: str = "hopkins",
        band: tuple[NDArray[np.float64], NDArray[np.float64]] | None = None,
    ) -> None:
        self.wavelength_um, self.f_number = float(wavelength_um), float(f_number)
        self.sigma_aberr_um, self.pitch_um = float(sigma_aberr_um), float(pitch_um)
        self.supersample, self.model, self.band = int(supersample), model, band
        sample_um = self.pitch_um / self.supersample
        #: c = 8 F W020, so a quarter-cell step in c is this step in W020.
        self.step_um = sample_um / (4.0 * 8.0 * self.f_number)
        self._cache: dict[int, FloatArray] = {}

    def quantise(self, w020_um: float) -> float:
        """The grid value a request lands on -- exposed so a test can assert the step, not guess."""
        return round(max(0.0, float(w020_um)) / self.step_um) * self.step_um

    def kernel_for(self, w020_um: float) -> FloatArray:
        key = int(round(max(0.0, float(w020_um)) / self.step_um))
        cached = self._cache.get(key)
        if cached is None:
            cached = defocus_psf(
                self.wavelength_um,
                self.f_number,
                key * self.step_um,
                self.sigma_aberr_um,
                self.pitch_um,
                self.supersample,
                self.model,
                self.band,
            )
            self._cache[key] = cached
        return cached

    def __len__(self) -> int:
        return len(self._cache)
