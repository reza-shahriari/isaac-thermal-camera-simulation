"""Stage 3 — optics: supersampled scene radiance → in-band power on each detector pixel.

Fixed order (docs/physics-model.md §2, §8.1-§8.3, §13.4; ADR 0020):

    1. optical PSF (diffraction · aberration Gaussian) at the supersampled pitch, if given
    2. box-mean downsample k× → native grid            (detector footprint MTF + aliasing)
    3. × π τ_opt / (4F² + 1) · RI_ij · A_d               (aperture, relative illumination, area)
    4. + A_d Ω_eff (1 − τ_opt RI_ij) L_B(T_housing)     (lens emission + out-of-cone housing view)

so Φ_ij = A_d Ω_eff [L_h + τ_opt RI_ij (L_scene − L_h)]: the relative illumination multiplies the
scene radiance *minus* the housing radiance (§8.2 revised 2026-09-26, ADR 0145). RI_ij is cos⁴θ
times the measured map named by ``optics.vignetting_map``, when there is one. On axis this is the
old single-lens form exactly; a uniform scene colder than the housing is darkest on axis.

Output Φ is float32 in W (or photons s⁻¹ if the input was photon radiance). ``invert_optics``
undoes 3-4 at the native grid and returns the scene band radiance the radiometric branch
inverts to apparent temperature (M3.10). The band radiance of the housing L_B(T_housing) is an
input: the caller looks it up in the band LUT so this stage stays band-agnostic.

docs/physics-model.md §2, §8.1, §8.2, §8.3, §13.4
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from irsim.config.sensor import SensorSpec
from irsim.optics.aperture import aperture_factor, fpa_irradiance
from irsim.optics.psf import apply_psf
from irsim.optics.sampling import box_downsample
from irsim.optics.self_emission import housing_power_field
from irsim.optics.smear import apply_motion_smear
from irsim.optics.vignetting import cos4_field, load_vignetting_map

__all__ = ["optics_field", "apply_optics", "invert_optics", "shutter_flux"]


def optics_field(sensor: SensorSpec, supersample: int = 1) -> NDArray[np.float32]:
    """The relative illumination RI_ij: cos⁴ (or ones) times the measured map, if configured.

    The measured map is native-grid data, so it is only accepted at ``supersample=1``; the
    optics stage applies RI after the box downsample, where the native grid is what it needs.
    """
    measured = None
    if sensor.optics.vignetting_map is not None:
        if supersample != 1:
            raise ValueError("a measured vignetting map is native-grid data; use supersample=1")
        measured = load_vignetting_map(
            sensor.optics.vignetting_map, sensor.fpa.width, sensor.fpa.height
        )
    return cos4_field(
        sensor.fpa.width,
        sensor.fpa.height,
        sensor.fpa.pitch_um,
        sensor.optics.focal_length_mm,
        supersample=supersample,
        enabled=sensor.optics.vignetting_cos4,
        measured_map=measured,
    )


def _check_native(radiance: NDArray[np.floating], sensor: SensorSpec, what: str) -> None:
    if radiance.shape[:2] != (sensor.fpa.height, sensor.fpa.width):
        raise ValueError(
            f"{what} shape {radiance.shape[:2]} != detector grid "
            f"{(sensor.fpa.height, sensor.fpa.width)}"
        )


def apply_optics(
    radiance_ss: NDArray[np.floating],
    sensor: SensorSpec,
    lb_housing: float,
    supersample: int | None = None,
    psf: NDArray[np.float64] | None = None,
    motion_px: NDArray[np.floating] | None = None,
) -> NDArray[np.float32]:
    """Scene band radiance on the k× grid → pixel power Φ (H, W) float32.

    ``psf`` (a kernel from :func:`irsim.optics.psf.optical_psf` at the same k) is applied first
    when given, then ``motion_px`` -- the within-frame smear of §8.3's ``mtf_motion``, which the
    cascade has described since M5 and nothing applied. Both are convolutions laid down during the
    integration and they commute, so the order between them is arbitrary; what is *not* arbitrary
    is that both come before the box filter, which is the detector sampling the result.

    ``motion_px`` is the supersampled displacement per frame, already scaled by the integration
    duty (:func:`irsim.optics.smear.smear_duty`) by the caller -- a bolometer integrates the whole
    frame, a cooled photon detector a fraction of it.
    """
    k = sensor.optics.supersample_factor if supersample is None else supersample
    blurred = apply_psf(radiance_ss, psf) if psf is not None else radiance_ss
    if motion_px is not None:
        blurred = apply_motion_smear(blurred, motion_px, 1.0)
    radiance = box_downsample(blurred, k)
    _check_native(radiance, sensor, "downsampled radiance")
    a_d = sensor.detector_active_area_m2
    f, tau = sensor.optics.f_number, sensor.optics.transmittance
    ri = optics_field(sensor)
    irradiance = fpa_irradiance(radiance, f, tau, ri)
    phi_housing = housing_power_field(a_d, f, tau, lb_housing, ri)
    phi = irradiance.astype(np.float64) * a_d + phi_housing
    return np.asarray(phi, dtype=np.float32)


def shutter_flux(sensor: SensorSpec, lb_shutter: float, lb_housing: float) -> NDArray[np.float32]:
    """Pixel power with the flat-field shutter closed (§11.2 revised 2026-09-26, SC.18).

    The shutter sits between the lens and the focal plane and fills each pixel's cone at
    L_B(T_shutter); the out-of-cone view of the housing is unchanged. Referenced to the axis as
    `apply_optics` is (ADR 0145):

        Φ_sh,ij = A_d Ω_eff [L_h + RI_ij (L_shutter − L_h)]

    which is `apply_optics` on a uniform scene with τ = 1, no lens emission and no blur.
    """
    ri = optics_field(sensor).astype(np.float64)
    a_d, f = sensor.detector_active_area_m2, sensor.optics.f_number
    axis = a_d * aperture_factor(f)
    phi = axis * (lb_housing + ri * (lb_shutter - lb_housing))
    return np.asarray(phi, dtype=np.float32)


def invert_optics(
    phi: NDArray[np.floating], sensor: SensorSpec, lb_housing: float
) -> NDArray[np.float32]:
    """Pixel power → equivalent scene band radiance at the native grid (float32).

    Removes the field-weighted housing term and divides out RI, so an off-axis pixel reports the
    radiance it actually saw; the radiometric branch therefore recovers T everywhere, not only on
    axis. ``lb_housing`` is what the camera *believes* the housing is (its calibration value), so a
    housing that has drifted reads back as a shading that grows toward the corners (§8.2).
    """
    p = np.asarray(phi)
    if p.dtype == np.float16:
        raise TypeError("pixel power is float16 (non-negotiable #2)")
    _check_native(p, sensor, "pixel power")
    a_d = sensor.detector_active_area_m2
    f, tau = sensor.optics.f_number, sensor.optics.transmittance
    ri = optics_field(sensor).astype(np.float64)
    phi_housing = housing_power_field(a_d, f, tau, lb_housing, ri)
    denom = a_d * aperture_factor(f) * tau * ri
    return np.asarray((p.astype(np.float64) - phi_housing) / denom, dtype=np.float32)
