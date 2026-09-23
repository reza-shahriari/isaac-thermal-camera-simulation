"""Mie scattering by a sphere (`PH.9`; docs/physics-model.md §7.4).

`PH.9` was written expecting "`cloud.py`'s Mie tables". There are none: `cloud.py` carries a single
measured band ratio (:data:`~irsim.atmosphere.cloud.CLOUD_OD_RATIO`) and says in as many words that
deriving that ratio from droplet Mie theory is not in scope. What the repository does have is
``data/nk/water.csv`` -- Segelstein's complex refractive index of liquid water over 0.65-15.6 um,
which is the *input* to Mie theory and covers both infrared bands. So the tables are computed here
rather than read.

Why it cannot be waved away with ``Q_ext = 2``. The geometric-optics limit is only reached for a
size parameter ``x = 2 pi r / lambda`` well above ten. A steam droplet of 5-20 um radius sits at
``x = 3-13`` in LWIR and ``x = 8-32`` in MWIR -- the first is squarely in the resonance region where
``Q_ext`` is neither 2 nor small, and the two bands land in different parts of the curve. That
difference **is** the measurement `PH.9` asks for, so approximating it away would answer the
question with the assumption.

The algorithm is Bohren & Huffman's ``BHMIE`` (*Absorption and Scattering of Light by Small
Particles*, 1983, §4.8) with the standard downward recursion for the logarithmic derivative, which
is the part that is unstable upward for an absorbing sphere. No SciPy: the Riccati-Bessel functions
are built by their own recurrences, as `irsim.optics.mtf` does for ``J1``.
"""

from __future__ import annotations

import numpy as np

__all__ = ["mie_efficiencies", "size_parameter"]


def size_parameter(radius_um: float, wavelength_um: float) -> float:
    """``x = 2 pi r / lambda`` -- the only geometry Mie theory needs."""
    if radius_um <= 0.0 or wavelength_um <= 0.0:
        raise ValueError("radius and wavelength must be positive")
    return float(2.0 * np.pi * radius_um / wavelength_um)


def mie_efficiencies(x: float, m: complex) -> tuple[float, float]:
    """``(Q_ext, Q_sca)`` for a sphere of size parameter ``x`` and refractive index ``m = n + ik``.

    The absorption efficiency is the difference. ``m`` takes ``k >= 0`` for an absorbing sphere,
    which is the convention ``data/nk`` is written in.

    The series is truncated at Wiscombe's ``x + 4 x^(1/3) + 2`` terms, and the logarithmic
    derivative ``D_n(mx)`` is seeded 15 orders above that and recurred **downward**: upward is
    unstable once ``k`` is appreciable, which for water in LWIR it is (``k = 0.05`` at 10 um,
    ``0.2`` at 12 um), so the obvious direction is the wrong one here.
    """
    if x <= 0.0:
        raise ValueError("size parameter must be positive")
    if m.imag < 0.0:
        raise ValueError("refractive index takes k >= 0 for an absorbing sphere")
    y = m * x
    n_terms = int(round(x + 4.0 * x ** (1.0 / 3.0) + 2.0))
    n_down = int(max(n_terms, abs(y))) + 15

    # D_n(y), downward from a zero seed: the recurrence forgets the seed within a few orders.
    d = np.zeros(n_down + 1, dtype=np.complex128)
    for n in range(n_down, 0, -1):
        rn = n / y
        d[n - 1] = rn - 1.0 / (d[n] + rn)

    psi_prev, psi = np.cos(x), np.sin(x)  # psi_{-1}, psi_0
    chi_prev, chi = -np.sin(x), np.cos(x)
    xi = complex(psi, -chi)
    q_ext = q_sca = 0.0
    for n in range(1, n_terms + 1):
        psi_next = (2.0 * n - 1.0) / x * psi - psi_prev
        chi_next = (2.0 * n - 1.0) / x * chi - chi_prev
        xi_next = complex(psi_next, -chi_next)
        dn = d[n]
        a_n = ((dn / m + n / x) * psi_next - psi) / ((dn / m + n / x) * xi_next - xi)
        b_n = ((dn * m + n / x) * psi_next - psi) / ((dn * m + n / x) * xi_next - xi)
        weight = 2.0 * n + 1.0
        q_ext += weight * (a_n.real + b_n.real)
        q_sca += weight * (abs(a_n) ** 2 + abs(b_n) ** 2)
        psi_prev, psi = psi, psi_next
        chi_prev, chi = chi, chi_next
        xi = xi_next
    scale = 2.0 / (x * x)
    return float(scale * q_ext), float(scale * q_sca)
