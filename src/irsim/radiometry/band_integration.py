"""Reference band integration: ∫R(λ) B(λ, T) dλ by composite Simpson quadrature.

This is the oracle. It is float64, written for clarity, vectorised over temperature, and every
faster path (the LUT, the GPU kernels) is tested against it -- never the other way round. It
uses NumPy only, so the oracle does not depend on a library's adaptive tolerances.

Grid (§3.2 b, amended by AT.24): **edge-aligned to both ends** of the response file's support --
the first node is the first sample and the last node is the last sample, never a step beyond it --
and **odd-length** as composite Simpson requires, by choosing an even number of intervals and
letting the spacing shrink below the nominal 0.01 µm to fit. The old grid kept 0.01 µm exactly
and padded one node past the support when the interval count came out odd; there ``resampled``
returns 0, so when the file ends on a non-zero sample Simpson's 1-4-1 weights straddled a cliff
(an exact 0.75–1.0 µm top-hat came out **+33 %** at 300 K and +54 % at 200 K, and the registry's
nominal NIR band is that case). The spacing is also capped at a fraction of the band's short
edge (:data:`FINE_GRID_FRACTION`): in the Wien tail the integrand changes by e^{c2 Δλ / λ² T}
per step, which at 0.75 µm and 300 K is a factor 2.4 per 0.01 µm, so the spec's "0.01 µm is
ample" holds above ~2 µm only (spec issue S58). Measured against a 0.0005 µm quadrature, the
shipped NIR response went from 28 mK to 2.6 mK worst case over 200–1000 K and the SWIR from
4.7 to 0.2 mK; LWIR and MWIR are bit-identical, their grids being unchanged.
The response is resampled onto this grid by linear interpolation and is zero outside the file's
support.

Four quantities, one per LUT table (§13.5): band radiance (energy and photon form) and the
band-integrated thermal derivative in both forms (§3.4, §9.4).

docs/physics-model.md §3.2 (b), §3.4, §9.4
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from irsim.radiometry.planck import (
    d_spectral_photon_radiance_dT,
    d_spectral_radiance_dT,
    spectral_photon_radiance,
    spectral_radiance,
)
from irsim.radiometry.spectral_response import RESAMPLE_DL_UM, SpectralResponse

__all__ = [
    "FINE_GRID_FRACTION",
    "quadrature_grid",
    "simpson",
    "band_radiance",
    "band_photon_radiance",
    "d_band_radiance_dT",
    "d_band_photon_radiance_dT",
]

FloatArray = NDArray[np.float64]

#: The grid spacing is at most this fraction of the band's short edge (AT.24): 0.005 puts two
#: hundred nodes per unit of ln λ, which holds a Wien-tail integrand to < 3 mK at 200 K in NIR.
#: Above 2 µm the nominal 0.01 µm is the tighter of the two and nothing changes.
FINE_GRID_FRACTION = 0.005


def quadrature_grid(response: SpectralResponse, dl_um: float = RESAMPLE_DL_UM) -> FloatArray:
    """Odd-length uniform grid from the response's first sample to its last, exactly.

    The spacing is the largest value not above ``dl_um`` (nor above
    ``FINE_GRID_FRACTION`` times the short edge) that divides the support into an even number
    of intervals, so the last node is the last sample and Simpson never sees a node where the
    response has been zeroed by extrapolation.
    """
    lo, hi = response.support_um
    step = min(float(dl_um), FINE_GRID_FRACTION * lo)
    n_intervals = int(np.ceil((hi - lo) / step - 1e-9))
    if n_intervals % 2 == 1:
        n_intervals += 1
    n_intervals = max(n_intervals, 2)
    return lo + (hi - lo) / n_intervals * np.arange(n_intervals + 1, dtype=np.float64)


def simpson(y: FloatArray, dx: float) -> FloatArray:
    """Composite Simpson over the last axis of ``y`` (odd length, uniform spacing ``dx``)."""
    n = y.shape[-1]
    if n < 3 or n % 2 == 0:
        raise ValueError(f"Simpson needs an odd number of samples >= 3, got {n}")
    return np.asarray(
        dx
        / 3.0
        * (y[..., 0] + y[..., -1] + 4.0 * y[..., 1:-1:2].sum(-1) + 2.0 * y[..., 2:-1:2].sum(-1)),
        dtype=np.float64,
    )


def _integrate(
    response: SpectralResponse,
    temperature_k: FloatArray | float,
    spectral_fn: object,
) -> FloatArray:
    grid = quadrature_grid(response)
    weights = response.resampled(grid)
    t = np.atleast_1d(np.asarray(temperature_k, dtype=np.float64))
    spectral = spectral_fn(grid[None, :], t[:, None])  # type: ignore[operator]
    out = simpson(weights[None, :] * spectral, float(grid[1] - grid[0]))
    return np.asarray(out, dtype=np.float64)


def band_radiance(response: SpectralResponse, temperature_k: FloatArray | float) -> FloatArray:
    """L_B(T) = ∫R(λ) B(λ, T) dλ in W m^-2 sr^-1, for each temperature. §3.2 (b)."""
    return _integrate(response, temperature_k, spectral_radiance)


def band_photon_radiance(
    response: SpectralResponse, temperature_k: FloatArray | float
) -> FloatArray:
    """L_{B,q}(T) = ∫R(λ) B_q(λ, T) dλ in photons s^-1 m^-2 sr^-1. §3.2 (b), §9.1."""
    return _integrate(response, temperature_k, spectral_photon_radiance)


def d_band_radiance_dT(response: SpectralResponse, temperature_k: FloatArray | float) -> FloatArray:
    """(∂L_B/∂T) = ∫R(λ) ∂B/∂T dλ in W m^-2 sr^-1 K^-1: what bolometer NETD divides by. §3.4."""
    return _integrate(response, temperature_k, d_spectral_radiance_dT)


def d_band_photon_radiance_dT(
    response: SpectralResponse, temperature_k: FloatArray | float
) -> FloatArray:
    """(∂L_{B,q}/∂T) in photons s^-1 m^-2 sr^-1 K^-1: what photon-detector NETD divides by. §9.4."""
    return _integrate(response, temperature_k, d_spectral_photon_radiance_dT)
