"""Exterior convection coefficient with wind and vehicle speed (§6.2).

    h = max(h_free, h_forced),   h_free = c |T_s − T_air|^{1/3},   h_forced = a + b v_rel^n
    v_rel = |v_wind| + |v_vehicle|

with a ≈ 5, b ≈ 4, n ≈ 0.8 (SI, m s⁻¹) and c ≈ 1.5 -- the §6.2 engineering form, in the
family of the Jürges/McAdams flat-plate correlations (h ≈ 5.7 + 3.8 v) and the turbulent
natural-convection ΔT^{1/3} law. h is the largest single uncertainty in surface temperature
prediction (DIRSIG [R3]); it is *derived* from conditions here, never authored per material.
Vehicle speed enters through the relative air speed: a car at 28 m s⁻¹ sees h ≈ 62 W m⁻² K⁻¹,
twelve times the parked value (ADR 0033 for the scalar speed composition).

docs/physics-model.md §6.2
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "ConvectionParams",
    "DEFAULT_CONVECTION",
    "relative_air_speed",
    "free_convection",
    "forced_convection",
    "convection_coefficient",
    "free_forced_crossover_k",
]


@dataclass(frozen=True)
class ConvectionParams:
    """h_forced = a + b v^n  (W m⁻² K⁻¹, v in m s⁻¹);  h_free = c |ΔT|^{1/3}."""

    a: float = 5.0
    b: float = 4.0
    n: float = 0.8
    c: float = 1.5

    def __post_init__(self) -> None:
        if min(self.a, self.b, self.n, self.c) <= 0.0:
            raise ValueError("convection parameters a, b, n, c must be positive")


DEFAULT_CONVECTION = ConvectionParams()


def relative_air_speed(wind_speed_m_s: Any, vehicle_speed_m_s: Any = 0.0) -> NDArray[np.float64]:
    """|v_wind| + |v_vehicle|: scalar composition, the upper bound when headings are unknown."""
    v = np.abs(np.asarray(wind_speed_m_s, dtype=np.float64)) + np.abs(
        np.asarray(vehicle_speed_m_s, dtype=np.float64)
    )
    return np.asarray(v, dtype=np.float64)


def free_convection(
    delta_t_k: Any, params: ConvectionParams = DEFAULT_CONVECTION
) -> NDArray[np.float64]:
    """c |ΔT|^{1/3} (turbulent natural convection; W m⁻² K⁻¹)."""
    h = params.c * np.cbrt(np.abs(np.asarray(delta_t_k, dtype=np.float64)))
    return np.asarray(h, dtype=np.float64)


def forced_convection(
    relative_speed_m_s: Any,
    params: ConvectionParams = DEFAULT_CONVECTION,
    *,
    density_ratio: float = 1.0,
) -> NDArray[np.float64]:
    """a + b (ρ_r v)^n (W m⁻² K⁻¹); v ≥ 0.

    ``density_ratio`` is ``ρ(z)/ρ(0)`` (PT.28): the turbulent correlation the parameters fit
    goes as the mass flux ``ρ v``, so thinner air blows less. At 1 this is the sea-level law
    bit for bit.
    """
    v = np.asarray(relative_speed_m_s, dtype=np.float64)
    if np.any(v < 0.0):
        raise ValueError("relative air speed must be non-negative")
    if not 0.0 < density_ratio <= 1.0:
        raise ValueError("density_ratio is rho(z)/rho(0) in (0, 1]")
    if density_ratio == 1.0:
        return params.a + params.b * v**params.n
    return params.a + params.b * (density_ratio * v) ** params.n


def convection_coefficient(
    delta_t_k: Any,
    wind_speed_m_s: Any,
    vehicle_speed_m_s: Any = 0.0,
    params: ConvectionParams = DEFAULT_CONVECTION,
) -> NDArray[np.float64]:
    """h = max(h_free(ΔT), h_forced(|wind| + |vehicle|)), broadcasting over arrays."""
    v_rel = relative_air_speed(wind_speed_m_s, vehicle_speed_m_s)
    return np.maximum(free_convection(delta_t_k, params), forced_convection(v_rel, params))


def free_forced_crossover_k(params: ConvectionParams = DEFAULT_CONVECTION) -> float:
    """|ΔT| at which free convection overtakes the calm-air forced value: (a/c)³ (37.0 K)."""
    return float((params.a / params.c) ** 3)
