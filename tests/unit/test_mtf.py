"""MTF cascade and PSF synthesis (MS.4, ADR 0059): the §8.3 worked case, independent pupil
autocorrelation and FFT checks of each term, the Airy ring, kernel properties and conservation."""

from __future__ import annotations

import numpy as np
import pytest

from irsim.optics import (
    apply_psf,
    cutoff_frequency_cyc_per_mm,
    mtf_detector,
    mtf_diffraction,
    mtf_gaussian,
    mtf_motion,
    mtf_system,
    nyquist_frequency_cyc_per_mm,
    optical_psf,
)


def test_worked_case_12um_f1_10um() -> None:
    xi_c = cutoff_frequency_cyc_per_mm(10.0, 1.0)
    xi_n = nyquist_frequency_cyc_per_mm(12.0)
    assert xi_c == pytest.approx(100.0, rel=1e-12) and xi_n == pytest.approx(41.667, abs=1e-3)
    assert float(mtf_diffraction(xi_n, 10.0, 1.0)) == pytest.approx(0.4853, rel=5e-3)
    assert float(mtf_detector(xi_n, 0.012)) == pytest.approx(0.6366, rel=1e-3)
    assert float(mtf_detector(xi_n, 0.012)) > 0.5, "the double-pi bug gives 0.198"


def test_diffraction_equals_pupil_autocorrelation() -> None:
    """Independent route: |FFT of the Airy PSF| = normalised autocorrelation of a circular pupil."""
    n, samples_per_radius = 1024, 100
    yy, xx = np.mgrid[-n // 2 : n // 2, -n // 2 : n // 2]
    pupil = (np.sqrt(xx**2 + yy**2) <= samples_per_radius).astype(np.float64)
    otf = np.abs(np.fft.fftshift(np.fft.ifft2(np.abs(np.fft.fft2(pupil)) ** 2)))
    otf /= otf.max()
    line = otf[n // 2, n // 2 :]  # radial line, in units where the cut-off is 2 R samples
    x = np.arange(line.size) / (2.0 * samples_per_radius)
    lam, f = 10.0, 1.0
    xi = x * cutoff_frequency_cyc_per_mm(lam, f)
    analytic = mtf_diffraction(xi, lam, f)
    mask = x <= 0.95
    assert np.max(np.abs(line[mask] - analytic[mask])) < 2e-3
    assert np.all(mtf_diffraction(np.array([100.0, 150.0]), lam, f) == 0.0)


def test_detector_and_gaussian_terms_match_fft_of_their_psfs() -> None:
    w = 0.012
    assert abs(float(mtf_detector(1.0 / w, w))) < 1e-12
    assert float(mtf_detector(0.5 / w, w)) == pytest.approx(2.0 / np.pi, rel=1e-12)
    # box of width w sampled finely: |FFT| vs sinc
    m = 4000  # samples across the box: the DFT of a sampled box is a Dirichlet kernel, ~1/m off
    dx = w / m
    n = 1 << 17
    box = np.zeros(n)
    box[:m] = 1.0 / m
    spec = np.abs(np.fft.rfft(box))
    freq = np.fft.rfftfreq(n, d=dx)
    keep = freq < 2.5 / w
    assert np.max(np.abs(spec[keep] - mtf_detector(freq[keep], w))) < 2e-6
    sigma = 0.008
    g = np.exp(-0.5 * ((np.arange(n) - n // 2) * dx / sigma) ** 2)
    g /= g.sum()
    gspec = np.abs(np.fft.rfft(np.fft.ifftshift(g)))
    assert np.max(np.abs(gspec[keep] - mtf_gaussian(freq[keep], sigma))) < 1e-6


def test_motion_and_cascade_properties() -> None:
    assert abs(float(mtf_motion(1.0 / (0.5 * 0.01), 0.5, 0.01))) < 1e-12
    assert np.all(mtf_motion(np.linspace(0, 100, 5), 0.0, 0.01) == 1.0)
    assert np.all(mtf_motion(np.linspace(0, 100, 5), 0.5, 0.0) == 1.0)
    xi = np.linspace(0.0, 120.0, 601)
    sysm = mtf_system(xi, 10.0, 1.0, detector_width_mm=0.012, sigma_mm=0.004)
    assert sysm[0] == pytest.approx(1.0)
    parts = np.minimum(mtf_diffraction(xi, 10.0, 1.0), mtf_detector(xi, 0.012))
    assert np.all(sysm <= parts + 1e-12)
    first_zero = np.argmax(mtf_detector(xi, 0.012) < 1e-9)
    assert np.all(np.diff(sysm[:first_zero]) <= 1e-12)


def test_psf_kernel_properties_and_airy_ring() -> None:
    k = optical_psf(10.0, 1.0, 0.0, 12.0, supersample=8)
    assert k.dtype == np.float64 and k.shape[0] % 2 == 1 and k.shape == k.T.shape
    assert abs(k.sum() - 1.0) < 1e-9 and k.min() >= 0.0
    assert np.max(np.abs(k - k.T)) < 1e-12 and np.max(np.abs(k - k[::-1, ::-1])) < 1e-12
    # first dark ring at 1.22 lambda F = 12.2 um = 8.13 samples at 1.5 um/sample (8x of 12 um)
    c = k.shape[0] // 2
    line = k[c, c:]
    first_min = int(np.argmin(line[: int(3.0 * 8.13)]) if line.size > 24 else 0)
    # find the first local minimum along the radius
    d = np.diff(line)
    first_min = int(np.argmax(d > 0))  # first index where the profile starts rising
    assert abs(first_min / 8.13 - 1.0) < 0.03, first_min


def test_apply_psf_conserves_uniform_and_dtypes() -> None:
    k = optical_psf(10.5, 1.0, 5.0, 12.0, supersample=4)
    uni = np.full((64, 96), 55.49, dtype=np.float32)
    out = apply_psf(uni, k)
    assert out.dtype == np.float32 and np.max(np.abs(out.astype(np.float64) / 55.49 - 1.0)) < 1e-6
    rng = np.random.default_rng(2)
    img = rng.random((200, 240)).astype(np.float32)
    blurred = apply_psf(img, k)
    # edge replication keeps the mean up to the border band (kernel radius / image size)
    assert abs(float(blurred.mean()) / float(img.mean()) - 1.0) < 2e-3
    assert blurred.std() < img.std()
    with pytest.raises(TypeError, match="float16"):
        apply_psf(img.astype(np.float16), k)
    with pytest.raises(ValueError):
        apply_psf(img, k[:-1])


def test_a_black_silhouette_on_a_bright_sky_stays_non_negative() -> None:
    """The NIR crash of 2026-09-26: FFT round-off (~1e-16 of the brightest pixel) turned a black
    airframe beside a 3e20 photon-unit sky into -2e5, and the photon detector refused it. A
    non-negative image through a non-negative kernel must come out non-negative, and nothing but
    the round-off may change: the bright field is untouched to float32 precision."""
    import numpy as np

    from irsim.config.loader import load_sensor_config
    from irsim.optics.psf import apply_psf, optical_psf

    s = load_sensor_config("configs/sensors/example_nir_si_1280.yaml").sensor
    k = optical_psf(
        s.reference_wavelength_um,
        s.optics.f_number,
        s.optics.mtf.aberration_sigma_um,
        s.fpa.pitch_um,
        s.optics.supersample_factor,
    )
    img = np.full((256, 320), 3.0e20, dtype=np.float32)
    img[100:160, 120:200] = 0.0
    out = apply_psf(img, k)
    assert float(out.min()) >= 0.0
    assert float(out[0:40, 0:40].min()) == pytest.approx(3.0e20, rel=1e-6)
    # Deep inside the silhouette only round-off remains: below 1e-12 of the sky, not light.
    assert float(out[125:135, 150:170].max()) < 1e-12 * 3.0e20
