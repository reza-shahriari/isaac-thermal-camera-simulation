"""People: skin and clothing are two temperatures on one body (PH.12).

A person is the most common target an infrared camera is pointed at, and modelling one as a single
temperature gets the most important thing about them wrong. **Bare skin and clothed torso are not
close.** At 0 °C in still air, 1 clo: skin sits at **34.07 °C** and the outside of the clothing at
**13.96 °C**, a **20.1 K** step across one body -- four hundred times a 50 mK NETD, and the reason
a face and hands are the brightest things in a winter street scene while the coat between them is
nearly background. Indoors at 22 °C in 0.5 clo the same person's step is only **4.9 K**, so how
much of a person stands out is a property of the weather and not of the person.

**Two different models, because they are two different problems.** Skin is thermoregulated: the
body holds it at ``35.7 − 0.028 (M − W)`` °C almost regardless of the weather, so it is authored
from the metabolic rate and not solved. Clothing is an ordinary passive surface with an insulation
between it and the skin, and it *is* solved -- ISO 7730's implicit balance,

    t_cl = t_sk − I_cl [ 3.96e-8 f_cl ((t_cl+273)⁴ − (t_r+273)⁴) + f_cl h_c (t_cl − t_a) ]

where ``f_cl`` is how much clothing enlarges the radiating area and ``h_c`` takes the larger of
free and forced convection. ``t_cl`` appears on both sides, and the residual is strictly
decreasing in it, so it is found by **bisection**. ISO 7730 is usually written as a fixed-point
iteration and an undamped one **diverges for a coat in wind** -- at 1 m/s the quartic and the
``h_c`` term together push the map's slope past −1, and damping by a half only postpones it.

⚠️ **This is an indoor comfort standard used outdoors.** ISO 7730's Annex A states validity for
air temperatures of 10-30 °C, so every winter figure above is an extrapolation. It is the right
extrapolation to make -- nothing else published gives a clothing surface temperature -- but the
numbers past the floor are the equation's, not the standard's.

**Why ISO 7730's own form and not this project's balance.** The standard's equation is the
reference every comfort model in the world is checked against, and its coefficients (3.96e-8 for
the radiative term, the 1.290/0.645 area factors, the 2.38/12.1 convection pair) are fitted
together as a set. Re-deriving it from :mod:`irsim.thermal.balance` would produce something close
but no longer comparable to anything published, and the point of a person is that a published
number exists for them.

**What this is not.** No sweating, no shivering, no skin blood flow, no radiant asymmetry, no
posture. Those live in a two-node comfort model; this is the surface an infrared camera sees, at a
steady state, which is a smaller question. ADR 0122.

docs/physics-model.md §6.1, §16.2; roadmap PH.12; ADR 0122; ISO 7730:2005 §4
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "MET_W_M2",
    "CLO_M2K_W",
    "STEFAN_ISO_7730",
    "SEATED_MET",
    "WALKING_MET",
    "skin_temperature_c",
    "clothing_area_factor",
    "convective_coefficient_w_m2_k",
    "clothing_temperature_c",
    "HumanSurfaces",
    "human_surfaces",
]

#: One met, the metabolic rate of a seated person at rest, W m⁻² of body surface (ISO 7730 §3.2).
MET_W_M2 = 58.15

#: One clo of thermal insulation, m² K W⁻¹ (ISO 7730 §3.3). A business suit is about 1 clo.
CLO_M2K_W = 0.155

#: The radiative coefficient ISO 7730 writes into its clothing balance, W m⁻² K⁻⁴. It is
#: ``ε σ`` with ε = 0.97 folded in -- 0.97 × 5.67e-8 = 5.50e-8 -- times the 0.72 effective
#: radiation area factor of a standing human, giving 3.96e-8. **Kept as the standard's single
#: number rather than rebuilt from `SIGMA_SB`**, because the coefficients of this equation were
#: fitted together and mixing one of ours into it would leave a balance comparable to nothing.
STEFAN_ISO_7730 = 3.96e-8

SEATED_MET = 1.0
WALKING_MET = 2.0

#: Insulation below which ISO 7730 uses the first area-factor branch, m² K W⁻¹ (= 0.5 clo).
_AREA_FACTOR_BREAK = 0.078

#: Bisection controls. Halving a ~90 K bracket to 1e-9 needs about 37 passes; the cap is far
#: above that so it can only be reached by a bracket that is not shrinking, which is a bug.
_MAX_ITERATIONS = 200
_TOLERANCE_C = 1e-9


def skin_temperature_c(metabolic_w_m2: float, work_w_m2: float = 0.0) -> float:
    """``t_sk = 35.7 − 0.028 (M − W)`` °C -- the thermoregulated set point (ISO 7730 §4).

    Authored rather than solved, and that is the physics and not a shortcut: a healthy body holds
    its skin near this value across a wide range of weather by varying blood flow and sweating,
    so a passive energy balance on skin would answer a question the body does not ask. Working
    harder *lowers* it -- a walking person's skin is 1.6 K cooler than a seated one's, because
    more of the heat is being carried away by sweat and blood flow than by the surface.
    """
    if metabolic_w_m2 <= 0.0:
        raise ValueError("metabolic rate must be positive")
    if work_w_m2 < 0.0 or work_w_m2 > metabolic_w_m2:
        raise ValueError("external work must lie between zero and the metabolic rate")
    return 35.7 - 0.028 * (float(metabolic_w_m2) - float(work_w_m2))


def clothing_area_factor(insulation_m2k_w: float) -> float:
    """``f_cl``: how much clothing enlarges the radiating and convecting area (ISO 7730 §4).

    A two-branch fit, breaking at 0.5 clo. It never falls below 1: clothing cannot make a person
    smaller, and a formula that allowed it would quietly reduce a naked person's heat loss.
    """
    i_cl = float(insulation_m2k_w)
    if i_cl < 0.0:
        raise ValueError("clothing insulation cannot be negative")
    if i_cl <= _AREA_FACTOR_BREAK:
        return 1.00 + 1.290 * i_cl
    return 1.05 + 0.645 * i_cl


def convective_coefficient_w_m2_k(t_cl_c: float, t_air_c: float, air_speed_m_s: float) -> float:
    """``h_c = max(2.38 |Δt|^0.25, 12.1 √v)`` -- free or forced, whichever is winning.

    The maximum rather than a sum, which is ISO 7730's own choice and the same one
    :mod:`irsim.thermal.convection` makes for surfaces: the two mechanisms do not add, one of
    them sets the boundary layer.
    """
    if air_speed_m_s < 0.0:
        raise ValueError("air speed cannot be negative")
    free = 2.38 * abs(float(t_cl_c) - float(t_air_c)) ** 0.25
    forced = 12.1 * math.sqrt(float(air_speed_m_s))
    return float(max(free, forced))


def clothing_temperature_c(
    t_air_c: float,
    t_radiant_c: float,
    insulation_m2k_w: float,
    metabolic_w_m2: float = MET_W_M2,
    air_speed_m_s: float = 0.1,
    work_w_m2: float = 0.0,
) -> float:
    """Solve ISO 7730's implicit clothing-surface balance for ``t_cl`` (ISO 7730 §4).

    ``t_radiant_c`` is the mean radiant temperature of the surroundings, which outdoors is well
    below air temperature under a clear sky and is the term that makes a coat read cold.

    With ``insulation_m2k_w = 0`` the equation collapses to ``t_cl = t_sk`` exactly -- no
    bisection, no residual -- which is the limit PH.12 asks for, returned exactly rather than
    approached to a tolerance.
    """
    t_sk = skin_temperature_c(metabolic_w_m2, work_w_m2)
    i_cl = float(insulation_m2k_w)
    if i_cl == 0.0:
        return t_sk
    f_cl = clothing_area_factor(i_cl)
    t_a, t_r = float(t_air_c), float(t_radiant_c)

    def residual(t_cl: float) -> float:
        """``t_sk − I_cl(radiative + convective) − t_cl``: zero at the solution.

        Strictly decreasing in ``t_cl`` -- every term on the left falls as the surface warms and
        the ``−t_cl`` falls with it -- which is what makes bisection safe here. ISO 7730 is
        usually written as a fixed-point iteration, and an undamped one **diverges** for a coat
        in wind: at 1 m/s the quartic and the ``h_c`` term together give the map a slope past
        −1, and even damping by a half only postpones it. A bracketed root has no such failure.
        """
        h_c = convective_coefficient_w_m2_k(t_cl, t_a, air_speed_m_s)
        radiative = STEFAN_ISO_7730 * f_cl * ((t_cl + 273.0) ** 4 - (t_r + 273.0) ** 4)
        convective = f_cl * h_c * (t_cl - t_a)
        return t_sk - i_cl * (radiative + convective) - t_cl

    # The surface lies between the coldest thing it touches and the skin: it cannot be warmer
    # than the body heating it, nor colder than both the air and the sky cooling it.
    lo = min(t_a, t_r, t_sk) - 1.0
    hi = max(t_sk, t_a, t_r) + 1.0
    if residual(lo) < 0.0 or residual(hi) > 0.0:
        raise RuntimeError(
            f"ISO 7730 clothing balance has no root between {lo:.1f} and {hi:.1f} C for "
            f"t_air={t_a} C, t_r={t_r} C, I_cl={i_cl} m2K/W"
        )
    for _ in range(_MAX_ITERATIONS):
        mid = 0.5 * (lo + hi)
        if hi - lo < _TOLERANCE_C:
            return mid
        if residual(mid) > 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


@dataclass(frozen=True)
class HumanSurfaces:
    """The two temperatures one person puts in front of a camera, in kelvin."""

    skin_k: float
    clothing_k: float

    #: ``t_sk − t_cl`` in kelvin: the step across the body, and the whole point of two patches.
    @property
    def step_k(self) -> float:
        return self.skin_k - self.clothing_k


def human_surfaces(
    t_air_c: float,
    t_radiant_c: float | None = None,
    clo: float = 1.0,
    met: float = SEATED_MET,
    air_speed_m_s: float = 0.1,
    work_w_m2: float = 0.0,
) -> HumanSurfaces:
    """Both surface temperatures for one person, from clo and met rather than from W m⁻² K.

    ``t_radiant_c`` defaults to the air temperature, which is the indoor assumption and the one
    ISO 7730 is written for. An outdoor scene should pass the sky's own radiant temperature; that
    is the difference between a coat reading 20 °C and the same coat reading 16 °C.
    """
    radiant = float(t_air_c) if t_radiant_c is None else float(t_radiant_c)
    metabolic = float(met) * MET_W_M2
    insulation = float(clo) * CLO_M2K_W
    return HumanSurfaces(
        skin_k=skin_temperature_c(metabolic, work_w_m2) + 273.15,
        clothing_k=clothing_temperature_c(
            t_air_c, radiant, insulation, metabolic, air_speed_m_s, work_w_m2
        )
        + 273.15,
    )
