"""Scattered sunlight: the sky's own brightness in a reflective band (M11.10, §5.3, §5.4).

**The gap this fills.** :class:`~irsim.atmosphere.sky.SkyModel` is a *thermal* sky: the clear sky
is the atmospheric column's own emission, which is what an LWIR camera sees and is essentially
zero at 0.9 µm. Rendering a NIR or SWIR scene with it produced a **black sky with a brightly
sunlit target on it** -- backwards, because in those bands the daytime sky is the brightest thing
in the frame and an aircraft is usually a dark silhouette against it. The reflected-solar term
(M11.3) lit the target and nothing lit the sky.

**The model, and what it costs.** The sky's diffuse radiance in a band is

    L_sky,scatter = f_B . DHI / pi

where DHI is the diffuse horizontal irradiance carried by the scene's own ``WeatherSeries`` and
f_B is the fraction of the diffuse spectrum that lands inside the band, per watt of DHI. The
factor of pi is not a fudge: for an **isotropic** sky, integrating L cos(theta) over the hemisphere
gives exactly L.pi, so this form reproduces the measured DHI by construction. That is the whole
justification for isotropy here -- it is the one angular distribution that is guaranteed right in
the integral, and the integral is the quantity the weather file actually measured.

It is wrong in the distribution, and knowingly so. A real clear sky is brighter near the horizon
(a longer scattering path) and much brighter within ~25 deg of the sun (the aureole, forward Mie
scattering). An isotropic sky therefore under-renders the circumsolar region and over-renders the
zenith. ADR 0086 records the size of that and why a Preetham-style distribution was not adopted.

**The diffuse spectrum is not the direct one.** Rayleigh scattering goes as lambda^-4, so skylight
is far bluer than the beam that produced it -- which matters enormously here, because it means a
NIR band receives a much smaller share of DHI than of DNI. The diffuse shape is taken as the solar
spectrum weighted by lambda^-4, normalised away by the ratio below, so only the *shape* of the
Rayleigh law is used and its absolute scale never enters.

docs/physics-model.md §5.3, §5.4; ADR 0086
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from irsim.radiometry.lut import Quantity
from irsim.radiometry.solar import SolarSpectrum, load_solar_spectrum

__all__ = ["RAYLEIGH_EXPONENT", "DiffuseSkylight", "diffuse_shape", "skylight_for_sensor"]

#: Rayleigh's wavelength dependence. The cross-section goes as lambda^-4, so the *spectral
#: radiance* of singly-scattered skylight is the solar spectrum times lambda^-4. Aerosol
#: scattering is much greyer (Angstrom exponents of 0.5-1.5), so a pure Rayleigh weighting
#: **understates** a hazy sky's near-infrared share; ADR 0086 records the bound.
RAYLEIGH_EXPONENT = 4.0


def diffuse_shape(spectrum: SolarSpectrum, exponent: float = RAYLEIGH_EXPONENT) -> SolarSpectrum:
    """The solar spectrum reweighted into a single-scattering Rayleigh diffuse shape.

    Only the shape survives: :meth:`DiffuseSkylight.for_sensor` divides the in-band integral by the
    broadband one, so any constant multiplying this cancels exactly and no calibration is implied.
    """
    wavelength = np.asarray(spectrum.wavelength_um, dtype=np.float64)
    if np.any(wavelength <= 0.0):
        raise ValueError("wavelengths must be positive")
    weighted = np.asarray(spectrum.irradiance_w_m2_um, dtype=np.float64) * wavelength ** (
        -float(exponent)
    )
    return SolarSpectrum(
        wavelength_um=wavelength,
        values=weighted,
        source_path=spectrum.source_path,
        sha256=spectrum.sha256,
    )


@dataclass(frozen=True)
class DiffuseSkylight:
    """Isotropic scattered-sunlight radiance for one band, per watt of DHI.

    ``per_dhi`` is in the band's own units divided by W m^-2: multiply by the weather's DHI to get
    a radiance, which :meth:`radiance` does. Holding the *ratio* rather than an absolute integral
    is what makes the Rayleigh weighting a shape rather than a calibration.
    """

    per_dhi: float
    quantity: Quantity
    band_id: str
    spectrum_sha256: str
    spectrum_path: str
    exponent: float = RAYLEIGH_EXPONENT
    #: In-band irradiance per W m^-2 of broadband **direct** beam (AT.20): the ground-level
    #: solar spectrum's own in-band share, unweighted, because the beam is not Rayleigh-shaped.
    #: A ratio like :attr:`per_dhi`, and in the same quantity. ``None`` on a skylight built
    #: before AT.20, which then cannot light the ground or a cloud.
    per_dni: float | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.per_dhi) or self.per_dhi < 0.0:
            raise ValueError(f"per_dhi must be finite and non-negative, got {self.per_dhi}")
        if self.per_dni is not None and (not math.isfinite(self.per_dni) or self.per_dni < 0.0):
            raise ValueError(f"per_dni must be finite and non-negative, got {self.per_dni}")

    @classmethod
    def from_spectrum(
        cls,
        spectrum: SolarSpectrum,
        response: Any,
        quantity: Quantity,
        band_id: str = "",
        exponent: float = RAYLEIGH_EXPONENT,
    ) -> DiffuseSkylight:
        shape = diffuse_shape(spectrum, exponent)
        in_band = float(shape.band(response, quantity))
        broadband = float(np.trapezoid(shape.irradiance_w_m2_um, shape.wavelength_um))
        if broadband <= 0.0:
            raise ValueError("the diffuse shape integrates to zero; check the spectrum file")
        direct = float(np.trapezoid(spectrum.irradiance_w_m2_um, spectrum.wavelength_um))
        return cls(
            per_dhi=in_band / broadband / math.pi,
            per_dni=float(spectrum.band(response, quantity)) / direct,
            quantity=quantity,
            band_id=band_id,
            spectrum_sha256=spectrum.sha256,
            spectrum_path=spectrum.source_path,
            exponent=exponent,
        )

    @classmethod
    def for_sensor(
        cls,
        sensor: Any,
        quantity: Quantity,
        data_dir: str | os.PathLike[str] | None = None,
        spectrum_file: str = "spectra/solar/direct_normal_am1p5.csv",
        exponent: float = RAYLEIGH_EXPONENT,
    ) -> DiffuseSkylight:
        """Integrate the Rayleigh-weighted ground-level spectrum against this sensor's own R(λ).

        The **ground-level** spectrum, not the top-of-atmosphere one: skylight is made by
        scattering the beam that has already been through the column, so the absorption bands the
        beam lost are missing from the sky too.
        """
        from irsim.config.loader import resolve_data_dir
        from irsim.radiometry.spectral_response import load_spectral_response

        root = resolve_data_dir(data_dir)
        spectrum = load_solar_spectrum(root / spectrum_file)
        response = load_spectral_response(sensor.sensor.band.spectral_response)
        return cls.from_spectrum(spectrum, response, quantity, sensor.sensor.band.band_id, exponent)

    def horizontal_irradiance(
        self, dni_w_m2: float, sun_elevation_deg: float, dhi_w_m2: float
    ) -> float:
        """E_B on a horizontal plane: the beam's ``per_dni`` share + the sky's Rayleigh share.

        ``E_B = per_dni · DNI · sin(el) + π · per_dhi · DHI`` -- the weather's own GHI split into
        its two spectral shapes, because a NIR band takes a far larger share of the beam than of
        the blue sky. Zero with the sun at or below the horizon. Needs :attr:`per_dni`.
        """
        if self.per_dni is None:
            raise ValueError(
                "this skylight carries no direct-beam fraction (per_dni); rebuild it with "
                "DiffuseSkylight.from_spectrum / for_sensor"
            )
        if dni_w_m2 < 0.0 or dhi_w_m2 < 0.0:
            raise ValueError("irradiance cannot be negative")
        beam = float(dni_w_m2) * max(math.sin(math.radians(float(sun_elevation_deg))), 0.0)
        return self.per_dni * beam + math.pi * self.per_dhi * float(dhi_w_m2)

    def radiance(self, dhi_w_m2: Any) -> NDArray[np.float64]:
        """f_B . DHI / pi -- the isotropic sky radiance this band sees from scattered sunlight."""
        dhi = np.asarray(dhi_w_m2, dtype=np.float64)
        if np.any(dhi < 0.0):
            raise ValueError("diffuse horizontal irradiance cannot be negative")
        return np.asarray(self.per_dhi * dhi, dtype=np.float64)


def skylight_for_sensor(
    sensor: Any,
    quantity: Quantity,
    data_dir: str | os.PathLike[str] | None = None,
    **kwargs: Any,
) -> DiffuseSkylight | None:
    """:class:`DiffuseSkylight` for a reflective or mixed band, **None for an emissive one**.

    The scattered term is physically present in every band; in LWIR it is 1.6e-8 of the column's
    own emission, which is two orders below float32's spacing there. Returning ``None`` rather
    than a negligible number is what keeps every emissive render bit-identical to what it was
    before this model existed -- the same rule §5.2's regime gate applies to the solar term.
    """
    if sensor.sensor.band.regime == "emissive":
        return None
    return DiffuseSkylight.for_sensor(sensor, quantity, data_dir, **kwargs)
