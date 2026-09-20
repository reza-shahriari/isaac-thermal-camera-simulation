"""The latent-heat term: evaporation from a wet surface, and the film it draws on (PH.1).

docs/physics-model.md §6.1 (which had no latent term; spec issue S45), §15 Tier 3 ("wet asphalt
reads colder than dry" was asked for and could not emerge); ADR 0101.

A wet surface loses heat as its water evaporates: ``Q_L = L_v E`` with the bulk-aerodynamic
evaporation rate

    E = ρ_a (q_sat(T_s) − q_a) / (r_a + r_s),      r_a = 1 / (C_E U)

in kg m⁻² s⁻¹ -- ``q_sat(T_s)`` the saturation specific humidity at the *surface's* temperature,
``q_a`` the air's, ``r_a`` the aerodynamic resistance the wind sets and ``r_s`` a surface
resistance that is zero for a free water film and positive for stomata (vegetation, PH.11) or a
drying soil. The term depends on the surface's own temperature, like emission, so it lives in the
balance (`irsim.thermal.balance.net_flux`) and not in the forcing; the forcing carries what the
weather knows -- ``q_a``, the bulk conductance ``g_e = ρ_a C_E U`` and the rain rate -- and the
solver carries a **film** per cell in kg m⁻² that rain fills and evaporation empties. A cell with
no film and no declared wetness evaluates exactly the balance it always did.

**The wet bulb is the check.** With no radiation, a saturated surface settles where sensible
gain balances latent loss, ``h (T_a − T_w) = L_v g_e (q_sat(T_w) − q_a)``, and when ``h`` and
``g_e`` obey the Lewis relation (``h = c_p g_e``, the same turbulence carrying heat and vapour)
that is the psychrometric wet-bulb equation. :func:`wet_bulb_temperature_k` solves it directly,
so the solver's transient has an algebraic answer to land on.

Saturation follows the same Magnus/Bolton fit `irsim.atmosphere.humidity` uses, vectorised
here because a field asks for ten thousand cells at once.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from irsim.atmosphere.humidity import MAGNUS_A_HPA, MAGNUS_B, MAGNUS_C_C
from irsim.radiometry.constants import (
    C_E_BULK,
    C_P_AIR_J_KGK,
    EPSILON_WATER_AIR,
    L_V_WATER_J_KG,
    P_STD_HPA,
    RHO_AIR_STD_KG_M3,
)

__all__ = [
    "MIN_WIND_M_S",
    "bulk_conductance_kg_m2_s",
    "evaporation_kg_m2_s",
    "latent_heat_flux_w_m2",
    "saturation_specific_humidity_kg_kg",
    "sensible_conductance_w_m2_k",
    "specific_humidity_kg_kg",
    "wet_bulb_temperature_k",
]

#: Below this the bulk formula's ``C_E U`` would vanish while free convection does not; the
#: floor stands in for the gustiness term COARE adds. ESTIMATED.
MIN_WIND_M_S = 0.5

# The same Magnus/Bolton fit `irsim.atmosphere.humidity` uses, so the air and the surface agree.
_MAGNUS_A = MAGNUS_A_HPA
_MAGNUS_B = MAGNUS_B
_MAGNUS_C = MAGNUS_C_C


def saturation_specific_humidity_kg_kg(
    temperature_k: Any, pressure_hpa: float = P_STD_HPA
) -> NDArray[np.float64]:
    """``q_sat = ε e_s / (p − (1 − ε) e_s)`` with the Magnus ``e_s(T)``; any shape."""
    t_c = np.asarray(temperature_k, dtype=np.float64) - 273.15
    if np.any(t_c < -80.0) or np.any(t_c > 100.0):
        raise ValueError("temperature outside -80..100 C: not a surface this fit is for")
    e_s = _MAGNUS_A * np.exp(_MAGNUS_B * t_c / (t_c + _MAGNUS_C))
    return np.asarray(EPSILON_WATER_AIR * e_s / (pressure_hpa - (1.0 - EPSILON_WATER_AIR) * e_s))


def specific_humidity_kg_kg(
    t_air_k: Any, rh_fraction: Any, pressure_hpa: float = P_STD_HPA
) -> NDArray[np.float64]:
    """The air's specific humidity from its temperature and relative humidity."""
    rh = np.asarray(rh_fraction, dtype=np.float64)
    if np.any((rh < 0.0) | (rh > 1.0)):
        raise ValueError("relative humidity must lie in [0, 1]")
    t_c = np.asarray(t_air_k, dtype=np.float64) - 273.15
    e = rh * _MAGNUS_A * np.exp(_MAGNUS_B * t_c / (t_c + _MAGNUS_C))
    return np.asarray(EPSILON_WATER_AIR * e / (pressure_hpa - (1.0 - EPSILON_WATER_AIR) * e))


def bulk_conductance_kg_m2_s(
    wind_speed_m_s: Any,
    c_e: float = C_E_BULK,
    rho_air_kg_m3: float = RHO_AIR_STD_KG_M3,
    min_wind_m_s: float = MIN_WIND_M_S,
) -> NDArray[np.float64]:
    """``g_e = ρ_a C_E max(U, U_min)``, kg m⁻² s⁻¹ per unit specific-humidity difference."""
    u = np.maximum(np.asarray(wind_speed_m_s, dtype=np.float64), min_wind_m_s)
    if np.any(u < 0.0):
        raise ValueError("wind speed cannot be negative")
    return np.asarray(rho_air_kg_m3 * c_e * u)


def sensible_conductance_w_m2_k(g_e_kg_m2_s: Any) -> NDArray[np.float64]:
    """The Lewis relation: ``h = c_p g_e`` -- the same turbulence carries heat and vapour."""
    return np.asarray(C_P_AIR_J_KGK * np.asarray(g_e_kg_m2_s, dtype=np.float64))


def evaporation_kg_m2_s(
    t_surface_k: Any,
    q_air_kg_kg: Any,
    g_e_kg_m2_s: Any,
    wet_fraction: Any = 1.0,
    r_s_s_m: Any = 0.0,
    rho_air_kg_m3: float = RHO_AIR_STD_KG_M3,
) -> NDArray[np.float64]:
    """``E = wet · ρ_a (q_sat(T_s) − q_a) / (r_a + r_s)`` with ``r_a = ρ_a / g_e``.

    Negative when the air is wetter than the surface is warm -- dew, which *adds* to the film
    and warms the surface by the latent heat it releases -- but only over the wet fraction: a
    dry cell neither evaporates nor collects dew here (dew onto dry surfaces is a later step).
    """
    wet = np.asarray(wet_fraction, dtype=np.float64)
    if np.any((wet < 0.0) | (wet > 1.0)):
        raise ValueError("wet_fraction must lie in [0, 1]")
    g_e = np.asarray(g_e_kg_m2_s, dtype=np.float64)
    r_s = np.asarray(r_s_s_m, dtype=np.float64)
    if np.any(g_e < 0.0) or np.any(r_s < 0.0):
        raise ValueError("conductances and resistances cannot be negative")
    dq = saturation_specific_humidity_kg_kg(t_surface_k) - np.asarray(q_air_kg_kg, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        r_a = np.where(g_e > 0.0, rho_air_kg_m3 / np.where(g_e > 0.0, g_e, 1.0), np.inf)
        rate = np.where(np.isfinite(r_a), rho_air_kg_m3 * dq / (r_a + r_s), 0.0)
    return np.asarray(wet * rate)


def latent_heat_flux_w_m2(
    t_surface_k: Any,
    q_air_kg_kg: Any,
    g_e_kg_m2_s: Any,
    wet_fraction: Any = 1.0,
    r_s_s_m: Any = 0.0,
) -> NDArray[np.float64]:
    """``Q_L = L_v E``, W m⁻², positive when the surface loses heat by evaporating."""
    return np.asarray(
        L_V_WATER_J_KG
        * evaporation_kg_m2_s(t_surface_k, q_air_kg_kg, g_e_kg_m2_s, wet_fraction, r_s_s_m)
    )


def wet_bulb_temperature_k(t_air_k: float, rh_fraction: float) -> float:
    """The psychrometric wet bulb: ``c_p (T_a − T_w) = L_v (q_sat(T_w) − q_a)``, by bisection.

    Where a saturated surface with no radiation settles under the Lewis relation, whatever the
    wind: the wind sets how fast it gets there, not where.
    """
    if not 0.0 <= rh_fraction <= 1.0:
        raise ValueError("relative humidity must lie in [0, 1]")
    q_a = float(specific_humidity_kg_kg(t_air_k, rh_fraction))
    lo, hi = t_air_k - 60.0, t_air_k
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        residual = C_P_AIR_J_KGK * (t_air_k - mid) - L_V_WATER_J_KG * (
            float(saturation_specific_humidity_kg_kg(mid)) - q_a
        )
        if residual > 0.0:  # too cold: sensible gain exceeds latent loss, so T_w is higher
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)
