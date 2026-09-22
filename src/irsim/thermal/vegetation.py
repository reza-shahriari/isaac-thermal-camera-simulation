"""Leaves transpire: the surface whose temperature is set by its stomata (PH.11).

A leaf is the opposite of every other surface in this library. It has almost no heat capacity --
630 J m⁻² K⁻¹ for the shipped broadleaf, against 63 000 for a snowpack and 180 000 for asphalt --
so it has no thermal memory at all and sits wherever the instantaneous balance puts it. And it
controls its own evaporation: the stomatal resistance ``r_s`` in `PH.1`'s latent term is not a
property of the weather but of the plant, and moving it from 50 to 5000 s/m swings the same leaf
under the same sun from **below** air temperature to well above it.

That is the whole phenomenology, and it is why vegetation is worth modelling at all in an infrared
scene: a well-watered canopy reads cooler than the air on a hot dry afternoon, a stressed one reads
warmer, and the difference is the basis of every crop water stress index ever flown.

**The capacity is small enough to break the solver, and that is `TC.1`'s guard.** A leaf's own
boundary layer gives ``h = 34.8 W m⁻² K⁻¹`` at 2 m/s, so with radiation its conductance is about
40 W m⁻² K⁻¹ and its time constant is **15.4 s** -- the midpoint rule is stable only to **30.8 s**,
and a scene ticking at 60 s would oscillate and diverge. (A snowpack's limit is 3089 s and
asphalt's 8825 s; the leaf is two orders more restrictive than anything else here.)
:func:`leaf_steady_state_k` therefore solves the balance's root directly rather than stepping it:
a surface with a 15 s memory has nothing to remember across a minute-long tick, so the steady
state *is* the answer -- the same move `TC.1` made for a conduction ladder whose fast mode sits
below the tick.

**A leaf's boundary layer is not the project's bulk one**, and getting that wrong is what makes a
leaf model produce the opposite phenomenology. See
:func:`leaf_boundary_conductance_mol_m2_s`.

**The oracle is Campbell & Norman, in different units on purpose.** :func:`leaf_temperature_cn`
implements the closed form of *An Introduction to Environmental Biophysics* (2nd ed., §14.5) in
**molar** units -- conductances in mol m⁻² s⁻¹, vapour as a mole fraction, the psychrometer
constant γ = c_p/λ -- while this project's balance is in SI mass units with a bulk aerodynamic
``E = ρ_a Δq / (r_a + r_s)``. Neither is derived from the other, so agreement between them is
evidence rather than an algebraic identity.

**They agree to 0.10 K near air temperature and to 0.37 K at 4.6 K of departure**, and the two
parts of that have different causes. The floor of about 0.1 K is the unit bridge: our specific
humidity uses the exact ``q = ε e/(p − (1−ε) e)`` while the molar side's ``Δq = (M_w/M_a) Δw``
is the ``q ≈ ε e/p`` form, worth 1.6 % of the deficit at 30 °C. The growth with departure is
Campbell & Norman's own double linearisation -- of the saturation curve *and* of σT⁴, both about
air temperature -- which is exactly what a closed form buys its closed form with. Where both
vanish (no net isothermal radiation, saturated air) the two agree to **4e-10 K**, which is the
bisection tolerance and says the formulations are the same equation.

docs/physics-model.md §6.1; roadmap PH.11; ADR 0121
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from irsim.atmosphere.humidity import saturation_vapour_pressure_hpa
from irsim.radiometry.constants import (
    C_P_AIR_J_KGK,
    L_V_WATER_J_KG,
    P_STD_HPA,
    RHO_AIR_STD_KG_M3,
    SIGMA_SB,
)
from irsim.thermal.balance import SurfaceForcing, ThermalProperties, net_flux

__all__ = [
    "M_AIR_KG_MOL",
    "M_WATER_KG_MOL",
    "C_P_MOLAR_J_MOL_K",
    "PSYCHROMETER_CONSTANT_PER_K",
    "WELL_WATERED_R_S_S_M",
    "STRESSED_R_S_S_M",
    "IDSO_SLOPE_C_PER_KPA",
    "LeafConductances",
    "LEAF_WIDTH_M",
    "leaf_boundary_conductance_mol_m2_s",
    "leaf_forcing",
    "vapour_pressure_deficit_kpa",
    "saturation_slope_per_k",
    "radiative_conductance_mol_m2_s",
    "leaf_temperature_cn",
    "leaf_steady_state_k",
    "leaf_time_constant_s",
]

#: Molar masses, kg mol⁻¹ (IUPAC). They are what bridge this module's molar oracle to the
#: project's mass-based bulk formulation, so they are named rather than folded into a factor.
M_AIR_KG_MOL = 0.028_96
M_WATER_KG_MOL = 0.018_015

#: Molar heat capacity of air at constant pressure, J mol⁻¹ K⁻¹ -- ``c_p M_air``. **Derived, and
#: it comes out at 29.105 against the 29.3 Campbell & Norman tabulate**, 0.67 % low: this project
#: carries dry-air ``c_p = 1005`` while they use a moist-air value. Deriving it rather than
#: pasting theirs keeps the molar oracle tied to the SI balance it is meant to check, and the gap
#: is left visible because it is smaller than the 1.4 outdoor-turbulence factor's own uncertainty.
C_P_MOLAR_J_MOL_K = C_P_AIR_J_KGK * M_AIR_KG_MOL

#: The thermodynamic psychrometer constant γ = c_p / λ in K⁻¹, both in molar units. Comes out at
#: 6.594e-4 against Campbell & Norman's tabulated 6.66e-4, **0.99 % low** -- the c_p above plus
#: their λ at 20 °C against this project's `L_V_WATER_J_KG`. Also derived rather than pasted.
PSYCHROMETER_CONSTANT_PER_K = C_P_MOLAR_J_MOL_K / (L_V_WATER_J_KG * M_WATER_KG_MOL)

#: Stomatal resistance of a well-watered broadleaf in daylight, s m⁻¹. Leaf-level values run
#: 50-200 s/m open and above 2000 s/m closed; these two bracket the regimes PH.11 asks about.
#: ESTIMATED from the standard range, not from a measurement of any particular species.
WELL_WATERED_R_S_S_M = 100.0
STRESSED_R_S_S_M = 5000.0

#: Idso's non-water-stressed baseline: the slope of (T_leaf − T_air) against vapour pressure
#: deficit for a well-watered canopy, °C per kPa. Published baselines across crops fall in this
#: band. **Recorded as an acceptance band, not fitted to** -- the model is asked whether it lands
#: inside it, and a model that had been tuned to it would prove nothing.
IDSO_SLOPE_C_PER_KPA = (-3.8, -1.1)

#: The root search's bracket around air temperature, K. A leaf cannot be 60 K from the air it
#: sits in under any forcing this project produces, and a bracket that wide makes the bisection
#: cheap while still failing loudly if the balance is broken rather than silently returning an edge.
_BRACKET_K = 60.0
_ROOT_TOL_K = 1e-9


@dataclass(frozen=True)
class LeafConductances:
    """The molar conductances Campbell & Norman's closed form needs, all mol m⁻² s⁻¹.

    Built from this project's SI forcing so the oracle describes the *same* boundary layer, which
    is the only way its agreement means anything. The conversions are the two molar masses and
    nothing else:

    * heat: ``h = c_p,molar g_Ha``  →  ``g_Ha = h / c_p,molar``
    * vapour: our ``E = ρ_a Δq / (r_a + r_s)`` and CN's ``E = M_w g_v Δw`` with
      ``Δq = (M_w/M_a) Δw`` give ``g_v = ρ_a / (M_a (r_a + r_s))``
    """

    heat_mol_m2_s: float
    vapour_mol_m2_s: float
    radiative_mol_m2_s: float

    @property
    def heat_plus_radiative_mol_m2_s(self) -> float:
        """``g_Hr``: the leaf loses sensible heat and radiation through parallel paths."""
        return self.heat_mol_m2_s + self.radiative_mol_m2_s

    @property
    def gamma_star_per_k(self) -> float:
        """``γ* = γ g_Hr / g_v`` -- the apparent psychrometer constant of *this* leaf.

        A closed stomate drives ``g_v`` to zero and γ* to infinity, which is what collapses the
        closed form onto the dry-surface answer.
        """
        if self.vapour_mol_m2_s <= 0.0:
            return math.inf
        return (
            PSYCHROMETER_CONSTANT_PER_K * self.heat_plus_radiative_mol_m2_s / (self.vapour_mol_m2_s)
        )


#: Characteristic width of the shipped broadleaf, m. A leaf's boundary layer is set by how far
#: air has to travel across it, so this is the single geometric number the conductance needs.
#: ESTIMATED for a generic broadleaf.
LEAF_WIDTH_M = 0.05

#: Campbell & Norman (2nd ed., eq. 7.30) forced-convection coefficient for a flat leaf, and the
#: 1.4 factor for outdoor turbulence they add on top of the laminar plate result.
_BOUNDARY_COEFFICIENT = 0.135
_OUTDOOR_TURBULENCE = 1.4

#: Vapour diffuses slightly faster than heat, so the boundary layer conducts vapour a little
#: better: ``g_va = g_Ha / 0.93`` (Campbell & Norman §7.5).
_VAPOUR_OVER_HEAT = 1.0 / 0.93


def leaf_boundary_conductance_mol_m2_s(
    wind_m_s: float, leaf_width_m: float = LEAF_WIDTH_M
) -> float:
    """``g_Ha = 1.4 · 0.135 √(u/d)`` -- a leaf's boundary layer, not a field's (C&N eq. 7.30).

    **This is not the bulk conductance the rest of the project uses**, and the difference is not
    a refinement. `bulk_conductance_kg_m2_s` describes a metres-deep surface layer over soil or
    sea and gives an aerodynamic resistance near 490 s/m at 2 m/s; a 5 cm leaf in the same wind
    has about **33 s/m**, fifteen times smaller, because the air only has to cross five
    centimetres. Using the bulk value makes a leaf unable to transpire, so it sits above air even
    when well watered -- the opposite of the phenomenology PH.11 exists to produce.
    """
    if wind_m_s < 0.0 or leaf_width_m <= 0.0:
        raise ValueError("wind must be non-negative and leaf width positive")
    # A still-air leaf still loses heat by free convection; the floor stands in for it.
    u = max(float(wind_m_s), 0.1)
    return _OUTDOOR_TURBULENCE * _BOUNDARY_COEFFICIENT * math.sqrt(u / float(leaf_width_m))


def leaf_forcing(
    t_air_k: float,
    rh_fraction: float,
    wind_m_s: float,
    q_solar_w_m2: float,
    q_longwave_down_w_m2: float,
    r_s_s_m: float,
    *,
    leaf_width_m: float = LEAF_WIDTH_M,
    rho_air_kg_m3: float = RHO_AIR_STD_KG_M3,
) -> SurfaceForcing:
    """A :class:`SurfaceForcing` whose sensible and latent paths share one leaf boundary layer.

    Building the two separately is how a leaf model goes quietly wrong: ``h`` and ``g_e`` are the
    same turbulence carrying two different things, and a forcing that drew them from different
    formulations would violate the Lewis relation this project's wet-bulb check rests on.
    """
    from irsim.thermal.latent import specific_humidity_kg_kg

    g_ha = leaf_boundary_conductance_mol_m2_s(wind_m_s, leaf_width_m)
    g_va = g_ha * _VAPOUR_OVER_HEAT
    return SurfaceForcing(
        t_air_k=float(t_air_k),
        h_w_m2_k=C_P_MOLAR_J_MOL_K * g_ha,
        q_solar_w_m2=float(q_solar_w_m2),
        q_longwave_down_w_m2=float(q_longwave_down_w_m2),
        q_air_kg_kg=float(specific_humidity_kg_kg(t_air_k, rh_fraction)),
        # invert `g_v = rho / (M_a r_a)` for the g_e the bulk latent term wants
        g_e_kg_m2_s=M_AIR_KG_MOL * g_va,
        wet_fraction=1.0,
        r_s_s_m=float(r_s_s_m),
    )


def radiative_conductance_mol_m2_s(t_air_k: float, emissivity: float) -> float:
    """``g_r = 4 ε σ T³ / c_p,molar``: radiation as a conductance, so it can sit beside the wind.

    This is the linearisation of εσT⁴ about the air temperature, and at 300 K with ε = 0.97 it is
    about 0.19 mol m⁻² s⁻¹ -- the same order as a leaf's boundary layer in still air. A model
    that left it out would make a leaf far too free to depart from air temperature.
    """
    return 4.0 * emissivity * SIGMA_SB * float(t_air_k) ** 3 / C_P_MOLAR_J_MOL_K


def conductances_from_forcing(
    forcing: SurfaceForcing, emissivity: float, rho_air_kg_m3: float = RHO_AIR_STD_KG_M3
) -> LeafConductances:
    """Translate this project's SI forcing into the molar conductances the oracle wants."""
    if forcing.g_e_kg_m2_s <= 0.0:
        raise ValueError("a transpiring leaf needs a positive bulk vapour conductance g_e")
    r_a = rho_air_kg_m3 / forcing.g_e_kg_m2_s
    return LeafConductances(
        heat_mol_m2_s=forcing.h_w_m2_k / C_P_MOLAR_J_MOL_K,
        vapour_mol_m2_s=rho_air_kg_m3 / (M_AIR_KG_MOL * (r_a + forcing.r_s_s_m)),
        radiative_mol_m2_s=radiative_conductance_mol_m2_s(forcing.t_air_k, emissivity),
    )


def vapour_pressure_deficit_kpa(t_air_k: float, rh_fraction: float) -> float:
    """``e_s(T_a) − e_a`` in kPa -- the dryness a leaf actually responds to."""
    if not 0.0 <= rh_fraction <= 1.0:
        raise ValueError("relative humidity must be a fraction in [0, 1]")
    e_s = saturation_vapour_pressure_hpa(float(t_air_k) - 273.15)
    return float(e_s * (1.0 - rh_fraction) / 10.0)


def saturation_slope_per_k(t_air_k: float, pressure_hpa: float = P_STD_HPA) -> float:
    """``s = (de_s/dT)/p``: the saturation *mole fraction* slope, K⁻¹.

    Differentiated numerically from the same Magnus fit the rest of the project uses, rather than
    from a second analytic derivative that could drift away from it.
    """
    t_c = float(t_air_k) - 273.15
    delta = 0.01
    de = saturation_vapour_pressure_hpa(t_c + delta) - saturation_vapour_pressure_hpa(t_c - delta)
    return float(de / (2.0 * delta) / pressure_hpa)


def leaf_temperature_cn(
    forcing: SurfaceForcing,
    properties: ThermalProperties,
    rh_fraction: float,
    *,
    pressure_hpa: float = P_STD_HPA,
    rho_air_kg_m3: float = RHO_AIR_STD_KG_M3,
) -> float:
    """Campbell & Norman's closed form for leaf temperature (2nd ed., §14.5).

        T_L − T_a = γ*/(s + γ*) · R_ni/(c_p g_Hr)  −  D/(s + γ*)

    with ``R_ni`` the net **isothermal** radiation -- what the leaf would absorb minus what it
    would emit *at air temperature* -- and ``D`` the vapour mole-fraction deficit. The first term
    is how far radiation pushes the leaf up; the second is how far evaporation pulls it down, and
    a closed stomate sends γ* to infinity so the second vanishes and the first becomes the whole
    dry-surface answer.
    """
    g = conductances_from_forcing(forcing, properties.emissivity, rho_air_kg_m3)
    t_air = float(forcing.t_air_k)
    r_ni = (
        properties.solar_absorptivity * forcing.q_solar_w_m2
        + properties.emissivity * forcing.q_longwave_down_w_m2
        - properties.emissivity * SIGMA_SB * t_air**4
        + forcing.q_internal_w_m2
    )
    s = saturation_slope_per_k(t_air, pressure_hpa)
    gamma_star = g.gamma_star_per_k
    deficit = vapour_pressure_deficit_kpa(t_air, rh_fraction) * 10.0 / pressure_hpa

    radiative_term = r_ni / (C_P_MOLAR_J_MOL_K * g.heat_plus_radiative_mol_m2_s)
    if math.isinf(gamma_star):
        # closed stomata: γ*/(s+γ*) → 1 and D/(s+γ*) → 0, so the leaf is a dry surface
        return t_air + radiative_term
    return float(
        t_air + gamma_star / (s + gamma_star) * radiative_term - deficit / (s + gamma_star)
    )


def leaf_steady_state_k(
    properties: ThermalProperties, forcing: SurfaceForcing, *, t_air_k: float | None = None
) -> float:
    """The root of this project's own balance: ``net_flux(T) = 0``.

    Solved rather than stepped. A leaf's time constant is about 25 s (see
    :func:`leaf_time_constant_s`), so an explicit 60 s tick is past the midpoint rule's stability
    limit and would oscillate -- `TC.1`'s guard. A surface with nothing to remember across a tick
    has a steady state that *is* its answer.

    Bisection rather than Newton: the balance is monotone decreasing in T (emission, convection
    and evaporation all rise with it), so bisection cannot fail, and the derivative Newton would
    need is exactly the thing a future latent term might make ugly.
    """
    air = float(forcing.t_air_k if t_air_k is None else t_air_k)

    def residual(t: float) -> float:
        return float(np.asarray(net_flux(np.array(t), properties, forcing)).reshape(()))

    lo, hi = air - _BRACKET_K, air + _BRACKET_K
    f_lo, f_hi = residual(lo), residual(hi)
    if f_lo < 0.0 or f_hi > 0.0:
        raise ValueError(
            f"the leaf balance has no root within {_BRACKET_K} K of air at {air:.1f} K "
            f"(residuals {f_lo:.3g} and {f_hi:.3g}); the forcing is not a leaf's"
        )
    while hi - lo > _ROOT_TOL_K:
        mid = 0.5 * (lo + hi)
        if residual(mid) > 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def leaf_time_constant_s(properties: ThermalProperties, forcing: SurfaceForcing) -> float:
    """``C / (h + 4 ε σ T³)`` at air temperature -- why a leaf cannot be stepped at 60 s."""
    denominator = (
        forcing.h_w_m2_k + 4.0 * properties.emissivity * SIGMA_SB * float(forcing.t_air_k) ** 3
    )
    if denominator <= 0.0:
        raise ValueError("a leaf with no convection and no emissivity has no time constant")
    return float(properties.heat_capacity_j_m2_k / denominator)


def stress_slope_c_per_kpa(
    properties: ThermalProperties,
    make_forcing: Any,
    deficits_kpa: Any,
    rh_for: Any,
) -> NDArray[np.float64]:
    """(T_leaf − T_air) at each vapour pressure deficit, for fitting the Idso slope.

    Takes builders rather than a forcing so the caller controls how the deficit is produced --
    raising it by drying the air at fixed temperature is not the same experiment as raising it by
    warming the air, and conflating the two is how a baseline slope goes wrong.
    """
    out = []
    for d in np.atleast_1d(np.asarray(deficits_kpa, dtype=np.float64)):
        forcing = make_forcing(float(d))
        out.append(leaf_steady_state_k(properties, forcing) - float(forcing.t_air_k))
    return np.asarray(out, dtype=np.float64)
