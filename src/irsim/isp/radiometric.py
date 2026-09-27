"""ISP radiometric branch: DN → scene band radiance → apparent temperature.

A radiometric camera exposes a linear branch beside the AGC image (docs/physics-model.md §11.1,
§3.3). Its calibration is the two-blackbody SITF of the *ideal* chain -- the detector transfer
and the optics with the housing at its **calibration temperature** (ADR 0021): the camera assumes
that housing temperature when it inverts, which is exactly why a housing drift reads as a scene
temperature error (§8.2, ADR 0016) until the next flat-field correction.

Two routes exist and are kept distinct:

* the **float32 route** used for the pipeline's ``apparent_t`` output: scene radiance is
  recovered from the un-quantised float32 signal, so the < 1 mK claims are meaningful;
* the **DN16 route** (``apparent_temperature_from_dn``): the same calibration applied to the
  quantised uint16 image, a *validation* output whose error is bounded by half an LSB.

``T_apparent`` is L_B⁻¹(L) via the band LUT and is never routed through the kinetic temperature
(§3.3: the gap *is* the product). ``outputs.apparent_temperature: false`` omits the branch --
the caller emits no key, never zeros.

docs/physics-model.md §3.3, §11.1, §12.2 (outputs)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from irsim.config.sensor import SensorSpec
from irsim.detector.bolometer import BolometerTransfer
from irsim.optics.stage import apply_optics, invert_optics
from irsim.radiometry.lut import BandLUT, Quantity

__all__ = [
    "RadiometricCalibration",
    "apparent_temperature",
    "apparent_temperature_from_dn",
    "dn_per_kelvin",
]

Float32Array = NDArray[np.float32]


@dataclass(frozen=True)
class RadiometricCalibration:
    """The calibrated DN ↔ scene-radiance transfer of one camera (ADR 0021)."""

    sensor: SensorSpec
    transfer: BolometerTransfer
    lb_housing_cal: float  # band radiance of the housing at calibration (the NUC reference level)

    @classmethod
    def from_scene_range(
        cls,
        sensor: SensorSpec,
        lut: BandLUT,
        t_min_k: float,
        t_max_k: float,
        t_housing_cal_k: float,
        quantity: Quantity = "lb",
    ) -> RadiometricCalibration:
        """Size the DN gain so blackbodies at [t_min, t_max] span the ADC, housing at T_cal."""
        if not t_min_k < t_max_k:
            raise ValueError("t_min_k must be below t_max_k")
        return cls.from_radiance_range(
            sensor,
            lut,
            float(lut.lookup(t_min_k, quantity)[()]),
            float(lut.lookup(t_max_k, quantity)[()]),
            t_housing_cal_k,
            quantity,
        )

    @classmethod
    def from_radiance_range(
        cls,
        sensor: SensorSpec,
        lut: BandLUT,
        l_min: float,
        l_max: float,
        t_housing_cal_k: float,
        quantity: Quantity = "lb",
    ) -> RadiometricCalibration:
        """Size the DN gain so uniform scenes at band radiance [l_min, l_max] span the ADC.

        The radiance form exists for the floor (§11.1, S53, `SC.23`): DN 0 at *zero* scene
        radiance, which no temperature in the band LUT reaches -- the table starts at 200 K and
        the model's clear winter zenith sky is colder than that. ``l_min = 0`` is legal and still
        lands on a positive flux, because the housing fills the rest of the pixel's view.
        """
        if not 0.0 <= l_min < l_max:
            raise ValueError("need 0 <= l_min < l_max")
        lb_housing = float(lut.lookup(t_housing_cal_k, quantity)[()])
        phi = [
            float(
                apply_optics(
                    np.full(sensor.fpa_shape, value, dtype=np.float32),
                    sensor,
                    lb_housing,
                    supersample=1,
                )[sensor.fpa.height // 2, sensor.fpa.width // 2]
            )
            for value in (l_min, l_max)
        ]
        transfer = BolometerTransfer.from_power_range(phi[0], phi[1], sensor.fpa.bit_depth)
        return cls(sensor=sensor, transfer=transfer, lb_housing_cal=lb_housing)

    def radiance_from_signal(self, signal_dn: NDArray[np.floating]) -> Float32Array:
        """Un-quantised signal (DN units, float32) → scene band radiance (float32)."""
        phi = self.transfer.power_from_signal_w(signal_dn)
        return invert_optics(phi, self.sensor, self.lb_housing_cal)

    def radiance_from_dn(self, dn16: NDArray[np.integer]) -> Float32Array:
        """Quantised DN → scene band radiance; error bounded by half an LSB (ADR 0021)."""
        dn = np.asarray(dn16)
        if not np.issubdtype(dn.dtype, np.integer):
            raise TypeError(f"dn16 must be an integer image, got {dn.dtype}")
        # an ADC floor sits on average half an LSB below the signal: reconstruct at the bin centre
        return self.radiance_from_signal((dn.astype(np.float64) + 0.5).astype(np.float32))

    def signal_from_radiance(self, radiance: NDArray[np.floating]) -> Float32Array:
        """Scene band radiance at the native grid → un-quantised signal (the forward SITF)."""
        phi = apply_optics(radiance, self.sensor, self.lb_housing_cal, supersample=1)
        return self.transfer.signal_dn(phi)


def apparent_temperature(
    radiance: NDArray[np.floating], lut: BandLUT, quantity: Quantity = "lb"
) -> Float32Array:
    """T_app = L_B⁻¹(L) (§3.3), float32; float16 refused by the LUT."""
    return lut.apparent_temperature(radiance, quantity)


def apparent_temperature_from_dn(
    dn16: NDArray[np.integer], calibration: RadiometricCalibration, lut: BandLUT
) -> Float32Array:
    """The DN16 validation route: quantised image → radiance → T_app (½-LSB bound)."""
    return lut.apparent_temperature(calibration.radiance_from_dn(dn16))


def dn_per_kelvin(
    calibration: RadiometricCalibration,
    lut: BandLUT,
    t_k: float,
    step_k: float = 0.5,
    quantity: Quantity = "lb",
) -> float:
    """∂DN/∂T of the calibrated transfer at ``t_k`` (DN per kelvin of scene temperature).

    This is the conversion reference the NUC residual uses to turn a millikelvin figure into a
    fixed number of DN, once (M9.6, ADR 0056). A central difference over ``step_k`` rather than an
    analytic derivative, because the transfer is a composition of the LUT, the optics stage and the
    detector and a finite difference is the same object the SITF bench measures.

    It is a strong function of temperature -- that dependence is exactly why a residual specified
    in kelvin and *applied* in kelvin would be wrong everywhere but one point (non-negotiable #3).
    """
    if not step_k > 0.0:
        raise ValueError("step_k must be positive")
    lo, hi = float(t_k) - float(step_k), float(t_k) + float(step_k)
    if lo <= 0.0:
        raise ValueError("the difference step reaches non-positive temperature")
    radiances = np.array([float(lut.lookup(t, quantity)[()]) for t in (lo, hi)], dtype=np.float64)
    shape = calibration.sensor.fpa_shape
    signals = [
        float(
            calibration.signal_from_radiance(np.full(shape, value, dtype=np.float32))[
                shape[0] // 2, shape[1] // 2
            ]
        )
        for value in radiances
    ]
    return (signals[1] - signals[0]) / (2.0 * float(step_k))
