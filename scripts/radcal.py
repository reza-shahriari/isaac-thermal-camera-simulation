#!/usr/bin/env python3
"""RadCal's CO2 and H2O narrow-band absorption coefficients, ported to NumPy (roadmap PH.5).

Not a runnable script: the oracle `scripts/generate_gas_luts.py` integrates to produce the
committed ``kappa_b(T)`` tables. It lives under `scripts/` and not under `src/irsim/` on purpose --
`docs/decisions/0098` fixed that the pipeline *reads* a table and never derives one, so nothing in
the physics core may reach a spectral model at render time.

**What this is.** A transcription of the ``CO2`` and ``H2O`` subroutines of FDS ``Source/rcal.f90``
(RadCal; W. L. Grosshandler, NIST Technical Note 1402, 1993; public domain), restricted to the
quantity RadCal calls ``SDWEAK``: the narrow-band **mean absorption coefficient in the weak-line
limit**, in cm-1 per atm of partial pressure, referred to STP density as NASA SP-3080 does.

**What this deliberately is not.** RadCal's line-structure machinery -- the Goody/Malkmus/Elsasser
fits, the Curtis-Godson averaging, the Doppler growth curve -- is *not* ported. Those convert the
weak-line coefficient into a narrow-band transmittance, and the model of ADR 0098 does not want a
transmittance: it wants one band-mean coefficient to put inside one exponential. Taking the
weak-line coefficient makes that model exact where it should be (optically thin) and conservative
where it cannot be (a thick band transmits more than a single exponential says, because the lines
saturate before the gaps between them do). ``GasSlab``'s docstring states the same envelope, and
`PH.13` records the ~8 % radiance figure the survey attaches to it.

The two species reach their coefficient by different routes, which is RadCal's own split:

* **H2O** is entirely tabulated -- ``SD(6, 376)``, bilinear in (temperature, wavenumber). The
  routine is the interpolation and nothing else.
* **CO2** is closed-form for the 2.0, 2.7, 4.3 and 10 um bands (vibration-rotation sums over the
  vibrational quantum numbers, from Penner's *Quantitative Molecular Spectroscopy*, 1955) and
  tabulated for 15 um -- ``SD15(6, 80)``.

Both tables are checked in under ``data/spectra/radcal/`` by `scripts/fetch_radcal_tables.py`,
which carries their provenance.

docs/physics-model.md §8.1, §8.3; ADR 0098.
"""

from __future__ import annotations

import functools
import math
import pathlib

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "CO2_SUPPORT_CM1",
    "H2O_SUPPORT_CM1",
    "RadcalTables",
    "kappa_co2_cm_atm",
    "kappa_h2o_cm_atm",
    "load_radcal_tables",
]

# --- RadCal's own constants ---------------------------------------------------------------------

Q2 = 1.4388  # h c / k_B, cm K -- RadCal's spelling of the second radiation constant
T0 = 300.0  # the reference temperature every band intensity is authored at
BE = 0.391635  # CO2 rotational constant, cm-1
OM1, OM2, OM3 = 1354.91, 673.0, 2396.49  # CO2 fundamentals: (100), (010), (001)
A_ROT = 0.0030875  # the rotation-vibration interaction constant RadCal calls A
X13, X23, X33 = -19.37, -12.53, -12.63  # CO2 anharmonicity constants
TWENTY_EPSILON = 20.0 * float(np.finfo(np.float64).eps)
UNDERFLOW = 78.0  # RadCal's own exponent guards, kept so a ported result matches term for term
#: RadCal stops summing over v3 once a term adds less than this fraction of the running total.
#: Ported rather than dropped: without it the sum keeps a tail RadCal never counted, and a table
#: that disagrees with the reference code by a tenth of a percent is a table nobody can check.
CONVERGED = 1.0e-4

#: Where each species' model says anything at all. Outside these, RadCal *sets* the coefficient to
#: zero -- which is a statement about fire radiation, where the truncated overtones carry no heat,
#: and not a statement that a SWIR or NIR camera would see nothing. The generator turns that
#: distinction into a per-band support fraction so a band outside the model is refused rather than
#: quietly handed zeros.
CO2_SUPPORT_CM1 = (500.0, 5725.0)
H2O_SUPPORT_CM1 = (50.0, 9300.0)

DEFAULT_TABLE_DIR = pathlib.Path(__file__).resolve().parent.parent / "data" / "spectra" / "radcal"


class RadcalTables:
    """The two tabulated arrays plus their grids, read once from ``data/spectra/radcal``."""

    def __init__(self, root: pathlib.Path | None = None) -> None:
        root = pathlib.Path(root) if root is not None else DEFAULT_TABLE_DIR
        self.h2o_wavenumbers, self.h2o_temperatures, self.sd = _read(root / "h2o_sd.csv")
        self.co2_wavenumbers, self.co2_temperatures, self.sd15 = _read(root / "co2_sd15.csv")
        if self.sd.shape != (6, 376) or self.sd15.shape != (6, 80):
            raise ValueError(
                f"RadCal tables have shapes {self.sd.shape} and {self.sd15.shape}; the ported "
                "index arithmetic assumes (6, 376) and (6, 80). Re-run fetch_radcal_tables.py."
            )


Triple = tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]


def _read(path: pathlib.Path) -> Triple:
    text = path.read_text(encoding="utf-8").splitlines()
    header = next(line for line in text if not line.startswith("#"))
    columns = header.split(",")[1:]
    temperatures = np.array([float(c.strip().rstrip("K").lstrip("k_")) for c in columns])
    rows = [
        [float(v) for v in line.split(",")]
        for line in text
        if line and not line.startswith("#") and line is not header
    ]
    body = np.array(rows, dtype=np.float64)
    return body[:, 0], temperatures, body[:, 1:].T


@functools.lru_cache(maxsize=4)
def load_radcal_tables(root: str | None = None) -> RadcalTables:
    return RadcalTables(pathlib.Path(root) if root else None)


# --- H2O: the table is the model -----------------------------------------------------------------


def kappa_h2o_cm_atm(
    omega_cm1: NDArray[np.float64], temp_k: float, tables: RadcalTables | None = None
) -> NDArray[np.float64]:
    """H2O's weak-line coefficient, cm-1 atm-1, by RadCal's bilinear interpolation of ``SD``.

    Ported from ``SUBROUTINE H2O`` in ``rcal.f90``. The temperature index arithmetic is RadCal's,
    not a general searchsorted: its grid is 300/600/1000/1500/2000/2500 K and the first two gaps
    are 300 and 400 K rather than 500, which the ``I == 1`` / ``I == 2`` special cases encode.
    """
    sd = (tables or load_radcal_tables()).sd
    omega = np.asarray(omega_cm1, dtype=np.float64)
    out = np.zeros_like(omega)
    inside = (omega >= H2O_SUPPORT_CM1[0]) & (omega < H2O_SUPPORT_CM1[1])
    if not inside.any():
        return out

    ttemp = min(max(temp_k, 300.0), 2499.99)
    i = math.floor(ttemp / 500.0) + 1
    if i == 2 and ttemp < 600.0:
        i = 1
    if i > 2:
        tt = (ttemp - (i - 1) * 500.0) / 500.0
    elif i == 1:
        tt = (ttemp - 300.0) / 300.0
    else:
        tt = (ttemp - 600.0) / 400.0

    j = ((omega[inside] - 25.0) // 25.0).astype(int)  # 1-based Fortran column
    ww = (omega[inside] - (25.0 + 25.0 * j)) / 25.0
    tw = tt * ww
    lo, hi = i - 1, i  # 0-based rows for I and I+1
    out[inside] = (
        sd[lo, j - 1] * (1.0 - tt - ww + tw)
        + sd[hi, j - 1] * (tt - tw)
        + sd[lo, j] * (ww - tw)
        + sd[hi, j] * tw
    )
    return out


# --- CO2: closed form per band, plus one table ---------------------------------------------------


def _partition_function(
    wavenumbers: tuple[float, ...], degeneracy: tuple[float, ...], temp: float
) -> float:
    q = 1.0
    for wn, g in zip(wavenumbers, degeneracy, strict=True):
        q *= (1.0 - math.exp(-Q2 * wn / temp)) ** g
    return q


def _approx_vib_rot(
    transitions: tuple[tuple[float, ...], ...],
    line_strength: tuple[float, ...],
    centres: tuple[float, ...],
    temp: float,
    omega: float,
) -> float:
    """``APPROX_VIB_ROT``: just-overlapping lines, Penner (1955) eq. 11-44."""
    total = 0.0
    fundamentals = (OM1, OM2, OM3)
    for row, strength, centre in zip(transitions, line_strength, centres, strict=True):
        transition_wn = sum(r * f for r, f in zip(row, fundamentals, strict=True))
        a_tot = (
            strength
            * (T0 / temp)
            * (1.0 - math.exp(-Q2 * transition_wn / temp))
            / _partition_function(fundamentals, row, temp)
        )
        total += (
            a_tot
            * (Q2 / temp / (4.0 * BE))
            * abs(omega - centre)
            * math.exp(-Q2 / (temp * 4.0 * BE) * (omega - centre) ** 2)
        )
    return total * temp / 273.0


def _vibration_sum(omega: float, temp: float, alpha: float, om_prime: float, second: bool) -> float:
    """One pass of RadCal's ``L120``/``L102``/``L101`` triple loop over (v, v3).

    ``second`` selects the second transition pair of the 2.7 um band, which shifts the band
    origin; ``alpha`` and ``om_prime`` carry the pair's integrated intensity and combination
    frequency. Returns the *unnormalised* sum -- the caller applies ``T/273``.
    """
    om12 = 0.5 * (0.5 * OM1 + OM2)
    aa = (
        alpha
        * BE
        * Q2
        / (
            A_ROT
            * (1.0 - math.exp(-OM3 * Q2 / T0))
            * (1.0 - math.exp(-om12 * Q2 / T0)) ** 3
            * (1.0 + math.exp(-om12 * Q2 / T0))
            * (1.0 - math.exp(-om_prime * Q2 / T0))
        )
    )
    bb = (
        (1.0 - math.exp(-Q2 * omega / temp))
        * (1.0 - math.exp(-Q2 * OM3 / temp))
        * (1.0 - math.exp(-om12 * Q2 / temp)) ** 3
        * (1.0 + math.exp(-om12 * Q2 / temp))
        * (1.0 - math.exp(-Q2 * om_prime / temp))
    )
    cc = aa * bb * omega * T0 / temp**2

    total = 0.0
    for jj in range(1, 21):
        v = float(jj - 1)
        even = jj % 2 == 0
        g = 0.25 * (v + 1.0) * (v + 3.0) if even else 0.25 * (v + 2.0) ** 2
        odd_bar = -1.0 + (v + 3.0) * (v + 4.0) / (6.0 * (v + 2.0))
        v_bar1 = -1.0 + (v + 5.0) / 6.0 if even else odd_bar
        for kk in range(1, 11):
            v3 = float(kk - 1)
            qq = (v3 + 1.0) * g * math.exp(-(v3 * OM3 + v * om12) * Q2 / temp) * (v_bar1 + 1.0)
            gam = BE - A_ROT * (v3 + 1.0)
            if second:
                om_vv3 = 3715.0 - 47.0 * v3 if v <= TWENTY_EPSILON else 3728.0 - 5.0 * v - 47.0 * v3
            else:
                om_vv3 = (
                    3613.0 - 47.0 * v3 if v <= TWENTY_EPSILON else 3598.0 - 18.0 * v - 47.0 * v3
                )
            term = _line_pair(omega, temp, cc, qq, gam, om_vv3)
            if term is None:
                break
            total += term
            if term <= 0.0 or term / total < CONVERGED:
                break
    return total


def _line_pair(
    omega: float, temp: float, cc: float, qq: float, gam: float, om_vv3: float
) -> float | None:
    """The body shared by every closed-form CO2 band; ``None`` means RadCal's ``CYCLE``."""
    delta = A_ROT * (omega - om_vv3)
    if gam * gam <= delta:
        return None
    d = 2.0 * math.sqrt(gam * gam - delta)
    om_v_bar = om_vv3 * (1.0 - math.exp(-om_vv3 * Q2 / temp))
    f1 = gam - 0.5 * d
    f2 = gam + 0.5 * d
    ee = Q2 * gam / (A_ROT * A_ROT * temp)
    unflo1 = ee * delta * (1.0 + 0.5 * A_ROT / gam)
    if unflo1 <= -UNDERFLOW or ee * 2.0 * gam * f1 >= UNDERFLOW:
        return None
    ff = math.exp(unflo1)
    s_minus = cc * qq / om_v_bar * abs(f1) * ff * math.exp(-ee * 2.0 * gam * f1)
    unflo3 = ee * 2.0 * gam * f2
    s_plus = 0.0 if unflo3 >= UNDERFLOW else cc * qq / om_v_bar * abs(f2) * ff * math.exp(-unflo3)
    return (s_minus + s_plus) / d


def _co2_scalar(omega: float, temp: float, tables: RadcalTables) -> float:
    """``SUBROUTINE CO2``, branch for branch, returning ``SDWEAK`` in cm-1 atm-1."""
    if omega > 5725.0:
        return 0.0

    if omega > 4550.0:  # 2.0 um: (000)-(041), (000)-(121), (000)-(201)
        return _approx_vib_rot(
            ((0.0, 4.0, 1.0), (1.0, 2.0, 1.0), (2.0, 0.0, 1.0)),
            (0.272, 1.01, 0.426),  # cm-2 atm-1 at 300 K, Penner & Varanasi JQSRT 4, 799 (1964)
            (4860.5, 4983.5, 5109.0),
            temp,
            omega,
        )

    if omega > 3800.0:
        return 0.0

    if omega > 3050.0:  # 2.7 um: (000)-(021)+(010)-(031), then (000)-(101)+(010)-(111)
        total = _vibration_sum(omega, temp, 28.5, 2.0 * OM2 + OM3, second=False)
        total += _vibration_sum(omega, temp, 42.3, OM1 + OM3, second=True)
        return total * temp / 273.0 if total > 0.0 else 0.0

    if omega > 2474.0:
        return 0.0

    if omega > 1975.0:  # 4.3 um
        if omega <= 2395.0:
            return _co2_dipole(omega, temp) * temp / 273.0
        total = _vibration_sum(omega, temp, 28.5, 2.0 * OM2 + OM3, second=False)
        total += _vibration_sum(omega, temp, 42.3, OM1 + OM3, second=True)
        return total * temp / 273.0 if total > 0.0 else 0.0

    if omega > 1100.0:
        return 0.0

    if omega > 880.0:  # 10 um: (100)-(001) and (020)-(001)
        oma = OM3
        omb = (OM1 + 2.0 * OM2) / 2.0
        scale = (
            (T0 / temp)
            * math.exp(Q2 * omb * (1.0 / T0 - 1.0 / temp))
            * (1.0 - math.exp(-Q2 * (oma - omb) / temp))
            / ((1.0 - math.exp(-Q2 * oma / temp)) * (1.0 - math.exp(-omb * Q2 / temp)))
        )
        total = 0.0
        for a_tot, centre in ((0.0219, 960.8), (0.0532, 1063.6)):
            total += (
                a_tot
                * scale
                * Q2
                / (4.0 * BE * temp)
                * abs(omega - centre)
                * math.exp(-Q2 / (4.0 * BE * temp) * (omega - centre) ** 2)
            )
        return total * temp / 273.0

    if omega > 500.0:  # 15 um, tabulated
        return _co2_15um(omega, temp, tables)

    return 0.0


def _co2_dipole(omega: float, temp: float) -> float:
    """The dipole approximation RadCal uses below the 4.3 um band head (``L202A``)."""
    x_bar = 0.5 * (0.5 * X13 + X23)
    om12 = 0.5 * (0.5 * OM1 + OM2)
    om_prime = OM3
    aa = (
        2700.0
        * BE
        * Q2
        / (
            A_ROT
            * (1.0 - math.exp(-OM3 * Q2 / T0))
            * (1.0 - math.exp(-om12 * Q2 / T0)) ** 3
            * (1.0 + math.exp(-om12 * Q2 / T0))
            * (1.0 - math.exp(-om_prime * Q2 / T0))
        )
    )
    bb = (
        (1.0 - math.exp(-Q2 * omega / temp))
        * (1.0 - math.exp(-Q2 * OM3 / temp))
        * (1.0 - math.exp(-om12 * Q2 / temp)) ** 3
        * (1.0 + math.exp(-om12 * Q2 / temp))
        * (1.0 - math.exp(-Q2 * om_prime / temp))
    )
    cc = aa * bb * omega * T0 / temp**2

    total = 0.0
    for jj in range(1, 21):
        v = float(jj - 1)
        g = 0.25 * (v + 1.0) * (v + 3.0) if jj % 2 == 0 else 0.25 * (v + 2.0) ** 2
        for kk in range(1, 11):
            v3 = float(kk - 1)
            qq = (v3 + 1.0) * g * math.exp(-(v3 * OM3 + v * om12) * Q2 / temp)
            gam = BE - A_ROT * (v3 + 1.0)
            om_vv3 = OM3 + 0.5 * X13 + X23 + 2.0 * X33 + x_bar * v + 2.0 * X33 * v3
            term = _line_pair(omega, temp, cc, qq, gam, om_vv3)
            if term is None:
                break
            total += term
            if term <= 0.0 or term / total < CONVERGED:
                break
    return total


def _co2_15um(omega: float, temp: float, tables: RadcalTables) -> float:
    """Bilinear in ``SD15``; its temperature grid is 300/600/1200/1500/1800/2400 K, not H2O's."""
    ttemp = min(max(temp, 300.0), 2399.99)
    j = (math.floor(omega) - 495) // 5  # 1-based Fortran column
    ww = (omega - (495.0 + 5.0 * j)) / 5.0
    i = int(ttemp // 300.0)
    if i < 2:
        tt = (ttemp - i * 300.0) / 300.0
    elif ttemp < 1200.0:
        i, tt = 2, (ttemp - 600.0) / 600.0
    elif i > 5:
        i, tt = 5, (ttemp - 1800.0) / 600.0
    else:
        tt = (ttemp - i * 300.0) / 300.0
        if i >= 4:
            i -= 1
    tw = tt * ww
    sd15 = tables.sd15
    return float(
        sd15[i - 1, j - 1] * (1.0 - tt - ww + tw)
        + sd15[i, j - 1] * (tt - tw)
        + sd15[i - 1, j] * (ww - tw)
        + sd15[i, j] * tw
    )


def kappa_co2_cm_atm(
    omega_cm1: NDArray[np.float64], temp_k: float, tables: RadcalTables | None = None
) -> NDArray[np.float64]:
    """CO2's weak-line coefficient, cm-1 atm-1, over an array of wavenumbers at one temperature."""
    loaded = tables or load_radcal_tables()
    omega = np.asarray(omega_cm1, dtype=np.float64)
    flat = omega.reshape(-1)
    out = np.array([_co2_scalar(float(w), float(temp_k), loaded) for w in flat])
    return np.maximum(out, 0.0).reshape(omega.shape)
