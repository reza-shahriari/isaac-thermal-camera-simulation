"""§6.1's surface energy balance, its RK2 step, and the steady state it relaxes to.

    C dT/dt = α_sol Q_sol + Q_LW↓ − ε σ T⁴ − h (T − T_air) + q_int

Everything on the right already exists in this package -- §6.4's solar loading, §6.5's downwelling
longwave, §6.6's convection -- and this module is the bookkeeping that makes them one equation.
Two things it deliberately does **not** do:

* **It does not let a scene author an emissivity.** `ThermalProperties.from_material` takes ε from
  the material's *optical* data through M7.8's total hemispherical emissivity, which is the only
  number §6.1 can correctly use (ADR 0043). A `thermal:` block that could carry its own ε would
  let one scene radiate at 0.95 while the camera sees 0.88, and nothing would catch it.
* **It absorbs the downwelling longwave with ε, exactly as §6.1 writes it** — the term is
  `ε Q_LW↓`, "absorbed sky/env", not `Q_LW↓`. Kirchhoff: a surface absorbs the same fraction of
  incident longwave that it emits. Dropping that ε is invisible on a painted surface (10 % of one
  term) and catastrophic on a metal, where it hands a panel with ε = 0.09 the full ~320 W m⁻² of
  sky radiation while letting it emit only a tenth of a blackbody — a 19 K error at 03:00, which
  is how this was found. `ε Q_LW↓ − ε σ T⁴` is algebraically the net-exchange form
  `ε σ (ε_sky T_air⁴ − T⁴)`; the two terms are kept apart here only because Q_LW↓ arrives from
  §6.5 already carrying the sky's own emissivity, cloud and view factor.

The integrator is **RK2 (midpoint)**, per ADR 0036. The equation is stiff in the sense that a thin
panel's time constant is seconds while a scene's tick is minutes, so the step is guarded rather
than made adaptive: `stability_limit_s` returns the explicit bound and the caller is expected to
respect it or to use the analytic steady state.

docs/physics-model.md §6.1, §6.2, §15 T1; ADR 0036, ADR 0043
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from irsim.radiometry.constants import SIGMA_SB

__all__ = [
    "ThermalProperties",
    "SurfaceForcing",
    "net_flux",
    "rk2_step",
    "stability_limit_s",
    "steady_state_temperature",
    "linearised_time_constant_s",
]


@dataclass(frozen=True)
class ThermalProperties:
    """The material side of §6.1: what the surface is, not what is happening to it.

    ``emissivity`` is the **total hemispherical** value (ADR 0043) and is not authored here --
    :meth:`from_material` is the supported way to obtain one.
    """

    heat_capacity_j_m2_k: float
    emissivity: float
    solar_absorptivity: float

    def __post_init__(self) -> None:
        if not self.heat_capacity_j_m2_k > 0.0:
            raise ValueError("areal heat capacity must be positive")
        if not 0.0 <= self.emissivity <= 1.0:
            raise ValueError("emissivity must lie in [0, 1]")
        if not 0.0 <= self.solar_absorptivity <= 1.0:
            raise ValueError("solar absorptivity must lie in [0, 1]")

    @classmethod
    def from_material(
        cls, material: Any, temperature_k: float = 300.0, **kwargs: Any
    ) -> ThermalProperties:
        """Build from a library material, taking ε from its **optical** data (ADR 0043).

        The §12.3 ``thermal:`` block deliberately has no emissivity field, and this is why: ε is a
        property of the same surface the camera looks at, and a second authored copy of it is a
        second chance to disagree.
        """
        from irsim.materials.hemispherical import total_hemispherical_emissivity

        thermal = material.spec.thermal
        return cls(
            heat_capacity_j_m2_k=thermal.heat_capacity_j_m2_k,
            emissivity=total_hemispherical_emissivity(material, temperature_k, **kwargs).value,
            solar_absorptivity=thermal.solar_absorptivity,
        )


@dataclass(frozen=True)
class SurfaceForcing:
    """The environment side of §6.1: what is happening to the surface, at one instant."""

    t_air_k: float
    h_w_m2_k: float
    q_solar_w_m2: float = 0.0
    q_longwave_down_w_m2: float = 0.0
    q_internal_w_m2: float = 0.0
    #: The fraction of the surface's own emission that leaves for good. Below 1 when a grey body
    #: facing the surface reflects part of it back (ADR 0088 addendum, PT.7): a body of
    #: emissivity ε_r over view factor F returns F (1 − ε_r) ε of what the surface emits, so
    #: ``emission_factor = 1 − F (1 − ε_r) ε``. Exactly 1 under an open sky.
    emission_factor: float = 1.0

    def __post_init__(self) -> None:
        if self.t_air_k <= 0.0:
            raise ValueError("air temperature must be positive (kelvin, not celsius)")
        if not 0.0 <= self.emission_factor <= 1.0:
            raise ValueError("emission_factor must lie in [0, 1]")
        if self.h_w_m2_k < 0.0:
            raise ValueError("convection coefficient cannot be negative")
        if self.q_solar_w_m2 < 0.0 or self.q_longwave_down_w_m2 < 0.0:
            raise ValueError("incident fluxes cannot be negative")


def net_flux(
    temperature_k: Any, properties: ThermalProperties, forcing: SurfaceForcing
) -> NDArray[np.float64]:
    """The right-hand side of §6.1 in W m⁻², **not** divided by C."""
    t = np.asarray(temperature_k, dtype=np.float64)
    if np.any(t <= 0.0):
        raise ValueError("temperature must be positive (kelvin)")
    absorbed = properties.solar_absorptivity * forcing.q_solar_w_m2
    absorbed_longwave = properties.emissivity * forcing.q_longwave_down_w_m2
    emitted = forcing.emission_factor * properties.emissivity * SIGMA_SB * t**4
    convected = forcing.h_w_m2_k * (t - forcing.t_air_k)
    return np.asarray(absorbed + absorbed_longwave - emitted - convected + forcing.q_internal_w_m2)


def rk2_step(
    temperature_k: Any, dt_s: float, properties: ThermalProperties, forcing: SurfaceForcing
) -> NDArray[np.float64]:
    """One midpoint (RK2) step of C dT/dt = net_flux (ADR 0036).

    Midpoint rather than forward Euler for a reason that shows up in the numbers rather than in
    the order: the T⁴ term makes the derivative fall as the surface warms, so Euler consistently
    **overshoots** a warming surface and undershoots a cooling one, biasing a diurnal cycle's
    amplitude in a way that averaging over the day does not cancel.
    """
    if dt_s <= 0.0:
        raise ValueError("dt_s must be positive")
    t0 = np.asarray(temperature_k, dtype=np.float64)
    capacity = properties.heat_capacity_j_m2_k
    half = t0 + 0.5 * dt_s * net_flux(t0, properties, forcing) / capacity
    return np.asarray(t0 + dt_s * net_flux(half, properties, forcing) / capacity)


def linearised_time_constant_s(
    temperature_k: float, properties: ThermalProperties, forcing: SurfaceForcing
) -> float:
    """τ = C / (h + 4 ε σ T³): the relaxation time of the linearised balance (§6.2).

    The radiative term is linearised about ``temperature_k`` and behaves like a second convection
    coefficient of 4εσT³ -- about 5.5 W m⁻² K⁻¹ at 300 K for ε = 0.9, which is the same size as
    free convection. A model that ignored it would predict a surface twice as slow as it is.
    """
    denominator = forcing.h_w_m2_k + 4.0 * properties.emissivity * SIGMA_SB * temperature_k**3
    if denominator <= 0.0:
        raise ValueError("a surface with no convection and no emissivity has no time constant")
    return float(properties.heat_capacity_j_m2_k / denominator)


def stability_limit_s(
    temperature_k: float, properties: ThermalProperties, forcing: SurfaceForcing
) -> float:
    """Largest explicit step that stays stable: 2τ for the midpoint rule on this linearisation.

    Reported rather than enforced inside :func:`rk2_step`, because a caller stepping a thin panel
    with a scene-sized tick usually wants the **steady state** instead, not a smaller step.
    """
    return 2.0 * linearised_time_constant_s(temperature_k, properties, forcing)


def steady_state_temperature(
    properties: ThermalProperties,
    forcing: SurfaceForcing,
    bracket: tuple[float, float] = (150.0, 900.0),
    tolerance_k: float = 1e-9,
) -> float:
    """The root of ``net_flux`` in T, by bisection on a monotone function.

    ``net_flux`` is strictly decreasing in T (both −εσT⁴ and −h(T − T_air) are), so the root is
    unique and bisection cannot land on the wrong one -- which is the reason for preferring it to
    a Newton iteration that would be faster and could walk off a flat region at low ε.

    ε = 0 and h > 0 gives exactly T_air + (α Q_sol + q_int)/h, and the test holds it to 1e-9.
    """
    lo, hi = bracket
    if not 0.0 < lo < hi:
        raise ValueError("bracket must be an increasing pair of positive temperatures")
    f_lo = float(net_flux(lo, properties, forcing))
    f_hi = float(net_flux(hi, properties, forcing))
    if f_lo < 0.0 or f_hi > 0.0:
        raise ValueError(
            f"equilibrium is outside the bracket {bracket}: net flux is {f_lo:.3f} at {lo} K and "
            f"{f_hi:.3f} at {hi} K"
        )
    while hi - lo > tolerance_k:
        mid = 0.5 * (lo + hi)
        if float(net_flux(mid, properties, forcing)) > 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)
