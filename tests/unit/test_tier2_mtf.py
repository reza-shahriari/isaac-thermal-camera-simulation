"""Tier 2 MTF bench (MS.5 / VAL-16): the hotplate/aluminium step edge at 4x through the Boson
optical PSF and the box filter, measured by the slant-edge estimator against the analytic
cascade; the supersampled path aliases while a native-resolution blur does not; the PSF-before-
box order matters; the edge contrast is the radiometric one.

**The PSF is the shipped camera's, aberration included** (`SC.4`). Until then every assertion here
was written against a lens whose second factor was identically zero -- a *diffraction-limited*
Boson, 10 % sharper at Nyquist than the one FLIR sells. The sigma is read from
`configs/sensors/flir_boson_640_lwir.yaml` rather than typed here, so this bench and the renderer
cannot disagree about which camera they are measuring."""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from irsim.config.loader import load_sensor_config
from irsim.optics import (
    apply_psf,
    box_downsample,
    mtf_detector,
    mtf_diffraction,
    nyquist_frequency_cyc_per_mm,
    optical_psf,
)
from irsim.optics.mtf import mtf_gaussian
from irsim.radiometry.lut import BandLUT
from irsim.validation import (
    compare_absolute,
    load_measured_pairs,
    measured_path,
    slant_edge_mtf,
)

# GT.1: this whole module is the slow tier -- a validation bench or an end-to-end frame rather
# than a unit test. `make test` skips it; `make test-slow` and `make check` run it.
pytestmark = pytest.mark.slow

LAMBDA_UM, F, PITCH_UM, K = 10.5, 1.0, 12.0, 4
#: The shipped Boson's own aberration Gaussian (`SC.4`), read rather than typed.
SIGMA_UM = float(
    load_sensor_config(
        pathlib.Path(__file__).resolve().parents[2]
        / "configs"
        / "sensors"
        / "flir_boson_640_lwir.yaml"
    ).sensor.optics.mtf.aberration_sigma_um
)


def _lens_mtf(xi_cyc_per_mm: object) -> np.ndarray:
    """Diffraction times the aberration Gaussian: the lens alone, without the detector box."""
    return np.asarray(
        mtf_diffraction(xi_cyc_per_mm, LAMBDA_UM, F) * mtf_gaussian(xi_cyc_per_mm, SIGMA_UM * 1e-3)
    )


@pytest.fixture(scope="module")
def edge_radiance(tophat_lwir_lut: BandLUT):  # type: ignore[no-untyped-def]
    """Radiance (epsilon = 1 for both materials) of a 512x512-at-4x step edge: 373 K hotplate on the
    right of a 5.5 deg tilted edge, 293 K sheet on the left."""
    n = 128 * K
    yy, xx = np.mgrid[0:n, 0:n]
    edge_x = 64.0 * K + yy * np.tan(np.radians(5.5))
    hot = (xx + 0.5) > edge_x
    t = np.where(hot, 373.0, 293.0).astype(np.float32)
    return tophat_lwir_lut.lookup(t)


MEASURED_MTF = measured_path("mtf", "flir_boson_640_lwir")


@pytest.mark.skipif(not MEASURED_MTF.is_file(), reason=f"no measured MTF at {MEASURED_MTF}")
def test_against_measured_mtf(edge_radiance: np.ndarray) -> None:
    """When a slant-edge bench file exists, MTF is compared **absolutely** (M12.4).

    MTF is already a ratio normalised to 1 at DC, so there is no scale left to fit and a
    discrepancy is a discrepancy. Compared only up to Nyquist: above it the estimate is aliasing,
    not resolution.
    """
    psf = optical_psf(LAMBDA_UM, F, SIGMA_UM, PITCH_UM, supersample=K)
    res = slant_edge_mtf(
        box_downsample(apply_psf(edge_radiance, psf), K), oversample=4, pitch_um=PITCH_UM
    )
    freq, measured = load_measured_pairs(MEASURED_MTF)
    keep = freq <= nyquist_frequency_cyc_per_mm(PITCH_UM)
    simulated = np.interp(freq[keep], res.freq_cyc_per_mm, res.mtf)
    result = compare_absolute(simulated, measured[keep], tolerance=1.0)
    assert float(np.max(np.abs(simulated - measured[keep]))) < 0.05, result.describe()


def test_measured_mtf_at_nyquist_matches_cascade(edge_radiance: np.ndarray) -> None:
    psf = optical_psf(LAMBDA_UM, F, SIGMA_UM, PITCH_UM, supersample=K)
    native = box_downsample(apply_psf(edge_radiance, psf), K)
    res = slant_edge_mtf(native, oversample=4, pitch_um=PITCH_UM)
    xi_n = nyquist_frequency_cyc_per_mm(PITCH_UM)
    expected_n = float(_lens_mtf(xi_n) * mtf_detector(xi_n, PITCH_UM * 1e-3))
    # `SC.4`'s acceptance: FLIR's 42 % lens at Nyquist, cascaded with the ideal 12 um box
    # (sinc = 0.637). Diffraction-only -- what this bench measured before the second factor was
    # authored -- gives 0.294, which sits inside this band too: the band alone does not catch the
    # missing factor, and `tests/unit/test_lens_mtf.py` is where the sharp check lives.
    assert expected_n == pytest.approx(0.27, abs=0.03)
    measured_n = res.at(0.5)
    assert abs(measured_n - expected_n) < 0.05, (measured_n, expected_n)
    f_mm = res.freq_cyc_per_mm
    keep = f_mm <= xi_n
    cascade = _lens_mtf(f_mm[keep]) * mtf_detector(f_mm[keep], PITCH_UM * 1e-3)
    assert np.max(np.abs(res.mtf[keep] - cascade)) < 0.05


def test_supersampled_path_carries_the_detector_footprint_and_aliasing(
    edge_radiance: np.ndarray,
) -> None:
    """The supersampled + box path measures the full cascade (0.27 at Nyquist) and its estimated
    MTF above Nyquist follows |MTF_lens · sinc| -- real aliasing energy the estimator resolves from
    the edge tilt. A native-resolution point-sampled render blurred by the optical PSF lacks the
    detector footprint: it measures close to the lens alone instead (the wrong path of §8.3)."""
    psf_ss = optical_psf(LAMBDA_UM, F, SIGMA_UM, PITCH_UM, supersample=K)
    ss_path = box_downsample(apply_psf(edge_radiance, psf_ss), K)
    psf_native = optical_psf(LAMBDA_UM, F, SIGMA_UM, PITCH_UM, supersample=1)
    native_path = apply_psf(_native_edge(edge_radiance), psf_native)
    a = slant_edge_mtf(ss_path, oversample=4, pitch_um=PITCH_UM)
    b = slant_edge_mtf(native_path, oversample=4, pitch_um=PITCH_UM)
    xi_n = nyquist_frequency_cyc_per_mm(PITCH_UM)
    lens_only = float(_lens_mtf(xi_n))
    # the native-pitch PSF kernel is itself sampled coarsely, so its transfer at Nyquist is a
    # folded sum a little above the analytic lens value: the wrong path reads close to the LENS
    # alone, not the 0.27 of the full cascade
    assert abs(b.at(0.5) - lens_only) < 0.08, (b.at(0.5), lens_only)
    assert a.at(0.5) < b.at(0.5) - 0.1
    f_mm = a.freq_cyc_per_mm
    band = (f_mm > xi_n) & (f_mm < 0.9 / (PITCH_UM * 1e-3))  # between Nyquist and the sinc zero
    predicted = _lens_mtf(f_mm[band]) * mtf_detector(f_mm[band], PITCH_UM * 1e-3)
    assert np.max(np.abs(a.mtf[band] - predicted)) < 0.05
    assert float(np.trapezoid(a.mtf[band], f_mm[band])) > 0.0


def _native_edge(edge_radiance: np.ndarray) -> np.ndarray:
    """The wrong path: render at native resolution (point-sample the centre supersample)."""
    return np.ascontiguousarray(edge_radiance[K // 2 :: K, K // 2 :: K])


def test_psf_after_box_changes_the_mtf(edge_radiance: np.ndarray) -> None:
    psf_ss = optical_psf(LAMBDA_UM, F, SIGMA_UM, PITCH_UM, supersample=K)
    right = box_downsample(apply_psf(edge_radiance, psf_ss), K)
    psf_native = optical_psf(LAMBDA_UM, F, SIGMA_UM, PITCH_UM, supersample=1)
    wrong = apply_psf(box_downsample(edge_radiance, K), psf_native)
    a, b = slant_edge_mtf(right, oversample=4), slant_edge_mtf(wrong, oversample=4)
    keep = a.freq_cyc_per_px <= 0.5
    assert np.max(np.abs(a.mtf[keep] - b.mtf[keep])) > 0.02


def test_edge_contrast_matches_radiometry(
    edge_radiance: np.ndarray, tophat_lwir_lut: BandLUT
) -> None:
    psf = optical_psf(LAMBDA_UM, F, SIGMA_UM, PITCH_UM, supersample=K)
    native = box_downsample(apply_psf(edge_radiance, psf), K)
    contrast = float(native[:, -8:].mean() - native[:, :8].mean())
    expected = float(tophat_lwir_lut.lookup(373.0)[()] - tophat_lwir_lut.lookup(293.0)[()])
    assert abs(contrast / expected - 1.0) < 0.01
