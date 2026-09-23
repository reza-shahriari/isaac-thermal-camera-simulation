"""Band extinction of a water-droplet cloud, from Mie theory (`PH.9`; §7.4).

What a thermal camera sees of a steam plume is **the droplets, not the vapour**. Water vapour is
close to transparent through an 8-14 um window over a plume-sized path -- `PH.5`'s tables give the
number and `tests/unit/test_droplets.py` asserts it -- so a plume that is visible in LWIR is
visible because steam has condensed. That is the whole reason this module exists beside the gas
slab rather than inside it.

The extinction of a droplet population follows from one geometric identity. For droplets of
effective radius ``r`` at number density ``N``,

    beta_ext = N pi r^2 Q_ext          LWC = N (4/3) pi r^3 rho_water

and eliminating ``N`` gives the standard cloud-optics form

    beta_ext = 3 LWC Q_ext / (4 rho_water r)

so the only spectroscopy is ``Q_ext``, and that comes from :mod:`irsim.atmosphere.mie` on
Segelstein's ``data/nk/water.csv``. Note what the identity says: extinction goes as ``1/r`` at
fixed water content, so the same kilogram of water spread over smaller droplets is **more** opaque.
A plume does not become transparent as it condenses further.

``Q_ext`` is averaged over the camera's own response weighted by the Planck emission of the
droplets, which is the same weighting :func:`~irsim.pipeline.gas_slab.soot_band_kappa_per_m` uses:
the quantity that matters is the extinction seen by the photons this camera is actually collecting.
"""

from __future__ import annotations

import functools
import os

import numpy as np
from numpy.typing import NDArray

from irsim.materials.nk import NKTable, load_nk_table
from irsim.radiometry.band_integration import quadrature_grid, simpson
from irsim.radiometry.planck import spectral_radiance
from irsim.radiometry.spectral_response import SpectralResponse

__all__ = ["droplet_band_kappa_per_m", "droplet_q_ext", "water_refractive_index"]

#: Liquid water, kg/m^3. Its temperature dependence over 273-373 K is under 5 % and is ignored;
#: the droplet size distribution is a far larger unknown.
RHO_WATER_KG_M3 = 1000.0


@functools.lru_cache(maxsize=4)
def _water_table(data_dir: str | None = None) -> NKTable:
    return load_nk_table("water", data_dir)


def water_refractive_index(
    wavelength_um: NDArray[np.float64], data_dir: str | os.PathLike[str] | None = None
) -> NDArray[np.complex128]:
    """``m = n + ik`` for liquid water (Segelstein 1981), on the given wavelengths."""
    n, k = _water_table(str(data_dir) if data_dir is not None else None).at(wavelength_um)
    return np.asarray(n + 1j * k, dtype=np.complex128)


def droplet_q_ext(
    radius_um: float,
    response: SpectralResponse,
    t_droplet_k: float,
    data_dir: str | os.PathLike[str] | None = None,
) -> float:
    """Band-mean extinction efficiency of a droplet, weighted by ``R(lambda) B(lambda, T)``.

    This is where the two bands part company. A 10 um droplet is ``x = 16`` against MWIR and
    ``x = 6`` against LWIR, and ``Q_ext`` has not settled to its geometric limit of 2 at the
    latter -- so the same cloud is optically thicker to the shorter-wave camera, which is what
    `PH.9` asks to be shown rather than assumed.
    """
    if radius_um <= 0.0:
        raise ValueError("droplet radius must be positive")
    from irsim.atmosphere.mie import mie_efficiencies, size_parameter

    grid = quadrature_grid(response)
    m = water_refractive_index(grid, data_dir)
    q = np.array(
        [
            mie_efficiencies(size_parameter(radius_um, float(lam)), complex(mi))[0]
            for lam, mi in zip(grid, m, strict=True)
        ],
        dtype=np.float64,
    )
    weights = response.resampled(grid) * spectral_radiance(grid, np.float64(t_droplet_k))
    dx = float(grid[1] - grid[0])
    return float(simpson(weights * q, dx) / simpson(weights, dx))


def droplet_band_kappa_per_m(
    response: SpectralResponse,
    lwc_kg_m3: float,
    radius_um: float,
    t_droplet_k: float,
    data_dir: str | os.PathLike[str] | None = None,
) -> float:
    """``beta_ext = 3 LWC Q_ext / (4 rho_water r)``, 1/m. Zero water content gives zero."""
    if lwc_kg_m3 < 0.0:
        raise ValueError("liquid water content cannot be negative")
    if lwc_kg_m3 == 0.0:
        return 0.0
    q_ext = droplet_q_ext(radius_um, response, t_droplet_k, data_dir)
    radius_m = radius_um * 1e-6
    return float(3.0 * lwc_kg_m3 * q_ext / (4.0 * RHO_WATER_KG_M3 * radius_m))
