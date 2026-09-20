"""A gas slab in radiance space: hot gas and soot as a volumetric emitter in a surface pipeline.

Roadmap PH.4.

docs/physics-model.md §2 (the master equation, surface-only until this), §8.1, §8.3; spec issue
S46 (the §7.5 the model lacks); ADR 0098.

§2 describes what a surface emits and reflects. A flame, an exhaust plume or a hot-gas volume is
none of that: it emits and absorbs **along the ray**, and painting it as a hot prim gives a grey
emitter that is wrong in every band -- a CO₂/H₂O plume is bright in MWIR and nearly transparent in
LWIR, which no emissivity knob can express. The operator that is right for a band camera is the
one FDS uses for flame radiation, read per band (FDS Tech. Ref. eq. C.3):

    L_b = τ_b · L_behind + (1 − τ_b) · B_b(T_g),      τ_b = exp(−κ_b(T_g) · L)

with ``L_behind`` whatever the ray would have seen without the slab (the sky, a wall, an exhaust
pipe), ``B_b`` the band-integrated Planck radiance at the gas temperature, and ``κ_b`` the
band-mean absorption coefficient of the mixture over the path length ``L``:

    κ_b = κ_CO₂,b(T_g) · p_CO₂ + κ_H₂O,b(T_g) · p_H₂O + κ_soot,b(T_g)

The gas coefficients are per atmosphere of partial pressure and come from **tables generated
offline** from HITEMP through RADIS, or from RadCal (`PH.5`, open question 14) -- a table is
data this module reads and never derives, because HITRAN-regime coefficients scaled to flame
temperature are wrong by the bands HITEMP adds. Soot needs no table: in the small-particle limit
``κ_λ = C0 f_v / λ`` (`SOOT_RAYLEIGH_C0`), and its band mean is the Planck-weighted ``⟨1/λ⟩`` over
the band's own R(λ), evaluated here by the same quadrature the band LUTs use.

**A slab is authored as (T_gas, p_CO₂, p_H₂O, f_soot, L), never as an emissivity.** The band
emissivity ``1 − τ_b`` is an *output* that differs per band, which is the whole phenomenon. The
gas temperature is guarded to 300–2500 K: below is not a hot gas, above is outside every table
and RadCal's envelope.

**At range**, the slab is a resolved feature at the target's distance, attenuated by the rest of
the path exactly as MS.6's point target is (`irsim.pipeline.point_target`): per class of the
layered atmosphere, ``ΔL = Σ_k w_k τ_k(R) (1 − τ_b) (B_b − L_behind,k)``; for the grey model
``ΔL = τ(R) (L_slab − L_air)``. The excess is *negative* when the gas is colder than what lies
behind it -- steam against a hot wall.

**What this is not.** A band-mean coefficient inside one exponential is exact in the optically
thin and the opaque limits and approximate between them (the true band transmittance is the
R·B-weighted mean of ``exp(−κ_λ L)``, which a narrow-band code like RadCal evaluates and this does
not); the survey's RadCal envelope for 20–200 cm paths at 800–1800 K is about 8 % in radiance,
and that is the fidelity this feature claims -- a phenomenology term, not a 10 mK one. No
scattering, no temperature gradient along the ray, no buoyancy or flicker. ADR 0098 records all
of it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from irsim.atmosphere.layered import LayeredAtmosphere
from irsim.atmosphere.model import Atmosphere
from irsim.radiometry.band_integration import (
    band_photon_radiance,
    band_radiance,
    quadrature_grid,
    simpson,
)
from irsim.radiometry.constants import SOOT_RAYLEIGH_C0
from irsim.radiometry.lut import BandLUT
from irsim.radiometry.planck import spectral_radiance
from irsim.radiometry.spectral_response import SpectralResponse

__all__ = [
    "GAS_T_MAX_K",
    "GAS_T_MIN_K",
    "GasBandTables",
    "GasSlab",
    "SlabQuantity",
    "SpeciesAbsorption",
    "gas_band_radiance",
    "slab_excess_radiance",
    "slab_optical_depth",
    "slab_radiance",
    "slab_transmittance",
    "soot_band_kappa_per_m",
]

#: The guard: below 300 K it is not a hot gas; above 2500 K it is outside every table.
GAS_T_MIN_K = 300.0
GAS_T_MAX_K = 2500.0

SlabQuantity = Literal["lb", "lb_q"]


@dataclass(frozen=True)
class GasSlab:
    """A homogeneous hot-gas volume along the ray: temperature, species, soot and path length.

    ``p_co2_atm`` and ``p_h2o_atm`` are partial pressures in atmospheres; ``f_soot`` is the soot
    volume fraction (a sooty flame is 1e-7 to 1e-5); ``length_m`` is the path through the gas.
    """

    t_gas_k: float
    length_m: float
    p_co2_atm: float = 0.0
    p_h2o_atm: float = 0.0
    f_soot: float = 0.0

    def __post_init__(self) -> None:
        if not GAS_T_MIN_K <= self.t_gas_k <= GAS_T_MAX_K:
            raise ValueError(
                f"gas temperature {self.t_gas_k} K is outside {GAS_T_MIN_K:g}-{GAS_T_MAX_K:g} K: "
                "below is not a hot gas, above is outside every absorption table"
            )
        if self.length_m <= 0.0:
            raise ValueError("a slab needs a positive path length")
        for name in ("p_co2_atm", "p_h2o_atm", "f_soot"):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} cannot be negative")
        if self.p_co2_atm > 1.0 or self.p_h2o_atm > 1.0:
            raise ValueError("a partial pressure above 1 atm is not a gas at ambient pressure")
        if self.p_co2_atm == 0.0 and self.p_h2o_atm == 0.0 and self.f_soot == 0.0:
            raise ValueError(
                "a slab with no CO2, no H2O and no soot is empty air: author at least one species"
            )

    @property
    def species(self) -> tuple[str, ...]:
        """The gas species with a non-zero partial pressure -- what needs a table."""
        return tuple(
            name for name, p in (("co2", self.p_co2_atm), ("h2o", self.p_h2o_atm)) if p > 0.0
        )


@dataclass(frozen=True)
class SpeciesAbsorption:
    """``κ_b(T)`` for one species in one band, 1/(m·atm), on a temperature grid.

    Generated offline (`PH.5`) and read here; linear in T between grid points, and a query
    outside the grid raises rather than extrapolating a line-by-line result nobody computed.
    """

    temperatures_k: NDArray[np.float64]
    kappa_per_m_atm: NDArray[np.float64]

    def __post_init__(self) -> None:
        t = np.asarray(self.temperatures_k, dtype=np.float64).reshape(-1)
        k = np.asarray(self.kappa_per_m_atm, dtype=np.float64).reshape(-1)
        if np.asarray(self.kappa_per_m_atm).dtype == np.float16:
            raise TypeError("float16 absorption tables are refused (CLAUDE.md #2)")
        if t.shape != k.shape or t.shape[0] < 2:
            raise ValueError("a table needs matching temperature and kappa arrays of length >= 2")
        if np.any(np.diff(t) <= 0.0):
            raise ValueError("table temperatures must be strictly increasing")
        if np.any(k < 0.0):
            raise ValueError("an absorption coefficient cannot be negative")
        object.__setattr__(self, "temperatures_k", t)
        object.__setattr__(self, "kappa_per_m_atm", k)

    def at(self, t_gas_k: float) -> float:
        t = self.temperatures_k
        if not t[0] <= t_gas_k <= t[-1]:
            raise ValueError(
                f"T = {t_gas_k} K is outside the table's {t[0]:g}-{t[-1]:g} K; the table is a "
                "line-by-line result and is not extrapolated"
            )
        return float(np.interp(t_gas_k, t, self.kappa_per_m_atm))


@dataclass(frozen=True)
class GasBandTables:
    """The species tables for one band: ``{"co2": ..., "h2o": ...}``, either may be absent."""

    band: str
    species: Mapping[str, SpeciesAbsorption]

    def kappa_per_m(self, slab: GasSlab) -> float:
        """The gas part of κ_b at the slab's temperature, 1/m, from its partial pressures."""
        total = 0.0
        for name, pressure in (("co2", slab.p_co2_atm), ("h2o", slab.p_h2o_atm)):
            if pressure <= 0.0:
                continue
            table = self.species.get(name)
            if table is None:
                raise ValueError(
                    f"the slab carries {name.upper()} but no κ_b(T) table is loaded for band "
                    f"{self.band!r}; `PH.5` generates the tables (`scripts/generate_gas_luts.py`) "
                    "-- a coefficient scaled from ambient HITRAN data would be wrong at flame "
                    "temperature and is not substituted"
                )
            total += table.at(slab.t_gas_k) * pressure
        return total


def soot_band_kappa_per_m(response: SpectralResponse, f_soot: float, t_gas_k: float) -> float:
    """Band-mean soot absorption, 1/m: ``C0 f_v ⟨1/λ⟩`` weighted by R(λ) B(λ, T_g) over the band.

    Rayleigh soot absorbs as ``C0 f_v / λ``; the band mean is the emission-weighted average of
    that over the camera's own response, so it is larger in MWIR than in LWIR by about the ratio
    of the bands' wavelengths -- soot is grey-ish, not grey.
    """
    if f_soot < 0.0:
        raise ValueError("f_soot cannot be negative")
    if f_soot == 0.0:
        return 0.0
    grid = quadrature_grid(response)
    weights = response.resampled(grid) * spectral_radiance(grid, np.float64(t_gas_k))
    dx = float(grid[1] - grid[0])
    mean_inverse_um = float(simpson(weights / grid, dx) / simpson(weights, dx))
    return SOOT_RAYLEIGH_C0 * f_soot * mean_inverse_um * 1e6  # 1/µm → 1/m


def slab_optical_depth(
    slab: GasSlab, response: SpectralResponse, tables: GasBandTables | None = None
) -> float:
    """``κ_b(T_g) · L`` for the mixture; ``tables`` may be omitted for a soot-only slab."""
    kappa = soot_band_kappa_per_m(response, slab.f_soot, slab.t_gas_k)
    if slab.species:
        if tables is None:
            raise ValueError(
                f"the slab carries {', '.join(s.upper() for s in slab.species)} and no band tables "
                "were given; `PH.5` generates them"
            )
        kappa += tables.kappa_per_m(slab)
    return kappa * slab.length_m


def slab_transmittance(
    slab: GasSlab, response: SpectralResponse, tables: GasBandTables | None = None
) -> float:
    """``τ_b = exp(−κ_b L)``: 1 for an empty path, 0 for an opaque flame."""
    return float(np.exp(-slab_optical_depth(slab, response, tables)))


def gas_band_radiance(
    response: SpectralResponse, t_gas_k: float, quantity: SlabQuantity = "lb"
) -> float:
    """``B_b(T_g)`` by quadrature -- the band LUT stops at 1000 K and a flame does not."""
    if quantity == "lb":
        return float(band_radiance(response, t_gas_k)[0])
    return float(band_photon_radiance(response, t_gas_k)[0])


def slab_radiance(
    slab: GasSlab,
    response: SpectralResponse,
    behind: Any,
    tables: GasBandTables | None = None,
    quantity: SlabQuantity = "lb",
) -> NDArray[np.float64]:
    """``τ_b L_behind + (1 − τ_b) B_b(T_g)`` for whatever lies behind the slab (any shape)."""
    tau = slab_transmittance(slab, response, tables)
    b_gas = gas_band_radiance(response, slab.t_gas_k, quantity)
    if np.asarray(behind).dtype == np.float16:
        raise TypeError("float16 radiance is refused (CLAUDE.md #2)")
    l_behind = np.asarray(behind, dtype=np.float64)
    return np.asarray(tau * l_behind + (1.0 - tau) * b_gas, dtype=np.float64)


def slab_excess_radiance(
    slab: GasSlab,
    response: SpectralResponse,
    atmosphere: Atmosphere | LayeredAtmosphere | None,
    band: str,
    t_s: float,
    lut: BandLUT,
    range_m: float,
    elevation_rad: float = 0.0,
    tables: GasBandTables | None = None,
    quantity: SlabQuantity = "lb",
    behind: float | None = None,
) -> float:
    """The slab's excess over what the pixel would otherwise see, attenuated from ``range_m``.

    ``behind`` is the radiance the ray meets beyond the slab at the slab's own range (a wall,
    a pipe); ``None`` means the sky beyond, per class for the layered model and ``L_B(T_air)``
    for the grey one -- the same convention as `point_target.excess_radiance`, so the two
    features attenuate identically along the same path.
    """
    if range_m < 0.0:
        raise ValueError("range cannot be negative")
    tau_b = slab_transmittance(slab, response, tables)
    b_gas = gas_band_radiance(response, slab.t_gas_k, quantity)
    emitted = 1.0 - tau_b
    if atmosphere is None:
        l_behind = 0.0 if behind is None else float(behind)
        return float(emitted * (b_gas - l_behind))
    if isinstance(atmosphere, LayeredAtmosphere):
        es = atmosphere.exponential_sum(band, t_s)
        tau_k = atmosphere.class_transmittances(band, t_s, range_m, elevation_rad)
        beyond = (
            atmosphere.sky_beyond_per_class(band, t_s, range_m, elevation_rad, quantity)
            if behind is None
            else np.full(es.weights.shape, float(behind))
        )
        return float(np.dot(es.weights * tau_k, emitted * (b_gas - beyond)))
    state = atmosphere.state(t_s)
    tau = float(np.exp(-state.gamma_per_m[band] * range_m))
    l_behind = (
        float(lut.lookup(np.float64(state.t_air_k), quantity)[()])
        if behind is None
        else float(behind)
    )
    return float(tau * emitted * (b_gas - l_behind))
