"""Snow: a surface that cannot get warmer than freezing, and what happens to the surplus (PH.10).

Every other surface in this project answers ``C dT/dt = net_flux``. Snow does not, and the
difference is not a refinement: a melting snowfield under 200 W/m² of net gain stays at **exactly
273.15 K** for as long as there is snow, while the same balance on any other material would put it
40 K above air by mid-afternoon. A model without the cap renders a spring snowfield as a hot patch,
which is the opposite of what a thermal camera sees.

**Where the energy goes.** At the cap the surface is isothermal and every surplus joule goes into
fusion instead of into temperature:

    m_melt = Q_net Δt / L_f,      L_f = 334 kJ/kg

so 200 W/m² melts 2.16 kg/m² per hour -- **2.2 mm of water equivalent** -- and moves the surface
not at all. The arithmetic is worth keeping in view because it is the whole phenomenology: a
snowfield's apparent temperature carries information about whether it is melting, not about how
much sun is on it.

**Two regimes, and the second one is the reason this is not a clamp.**

* A cell *at* the cap with a positive balance melts isothermally. The flux is evaluated **at the
  melt point**, not at some midpoint above it, because the surface never goes above it. Clamping
  after an RK2 step instead would evaluate emission and convection at a temperature that does not
  exist and under-report the melt by about 0.25 % per step -- small, one-signed, and avoidable.
* A cell *below* the cap that a step would carry through it has genuinely warmed part of the way.
  Its surplus enthalpy ``C (T_stepped − T_melt)`` becomes melt and the temperature lands on the
  cap. This conserves energy exactly across the transition rather than discarding the overshoot.

**Below freezing the cap does nothing at all**, and that matters as much as the melting case. Snow
has ε ≈ 0.99 in LWIR, so on a clear calm night it radiates into a sky tens of kelvin colder than
the air and settles several kelvin *below* air temperature; under overcast the sky is close to air
temperature and the depression nearly vanishes. That contrast is the second thing PH.10 asks for,
and it comes out of the ordinary balance with snow's emissivity -- no special case.

**What is not modelled here.** Snow is treated as one surface with one temperature: no cold
content through a pack depth, no liquid water retained and refrozen, no densification, no
albedo ageing (the material's 0.15 solar absorptivity is fresh snow and does not decay), and
sublimation only insofar as a caller supplies a latent term. ADR 0120 records why, and what each
would take.

docs/physics-model.md §6.1; roadmap PH.10; ADR 0120
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from irsim.radiometry.constants import L_F_WATER_J_KG, T_MELT_WATER_K
from irsim.thermal.balance import SurfaceForcing, ThermalProperties, net_flux, rk2_step

__all__ = [
    "T_MELT_K",
    "WATER_DENSITY_KG_M3",
    "ALPINE_TIER4_MAE_K",
    "SnowStep",
    "melt_capped_step",
    "melt_rate_kg_m2_s",
    "water_equivalent_mm",
]

#: The cap. Named here as well as in `constants` so a reader of this module sees it.
T_MELT_K = T_MELT_WATER_K

#: For turning a melted mass into a depth of water equivalent, kg m⁻³.
WATER_DENSITY_KG_M3 = 1000.0

#: Tier 4 acceptance band for snow-surface temperature against the alpine ESSD 16 (2024) series,
#: in K of mean absolute error. **Recorded, not run** -- the series is not fetched and no check
#: in this repo evaluates against it. It is here so that when one is written the bar is the
#: published instrument's rather than an aspiration invented at the time.
ALPINE_TIER4_MAE_K = (0.7, 1.3)

#: A cell within this of the melt point is treated as being *at* it, so that a surface which has
#: just been capped takes the isothermal branch on the following step instead of stepping a
#: hundredth of a millikelvin and being capped again.
CAP_TOLERANCE_K = 1e-9


@dataclass(frozen=True)
class SnowStep:
    """One step's outcome: where the surface landed and how much water it shed."""

    temperature_k: NDArray[np.float64]
    #: Mass melted during the step, kg m⁻². Zero wherever the surface stayed below freezing.
    melt_kg_m2: NDArray[np.float64]

    @property
    def melted(self) -> NDArray[np.bool_]:
        return np.asarray(self.melt_kg_m2 > 0.0)

    def water_equivalent_mm(self) -> NDArray[np.float64]:
        """The melt as a depth of liquid water, mm."""
        return water_equivalent_mm(self.melt_kg_m2)


def water_equivalent_mm(melt_kg_m2: Any) -> NDArray[np.float64]:
    """kg m⁻² of melt as mm of water equivalent. A kilogram per square metre is a millimetre."""
    return np.asarray(
        np.asarray(melt_kg_m2, dtype=np.float64) / WATER_DENSITY_KG_M3 * 1000.0,
        dtype=np.float64,
    )


def melt_rate_kg_m2_s(net_flux_w_m2: Any) -> NDArray[np.float64]:
    """Melt rate from a surplus flux at the melt point: ``Q / L_f``, never negative.

    A negative surplus is refreezing, which this model does not carry (there is no liquid water
    store to refreeze), so it reports no melt rather than a negative one -- a negative melt would
    silently create snow.
    """
    q = np.asarray(net_flux_w_m2, dtype=np.float64)
    return np.asarray(np.maximum(q, 0.0) / L_F_WATER_J_KG, dtype=np.float64)


def melt_capped_step(
    temperature_k: Any,
    dt_s: float,
    properties: ThermalProperties,
    forcing: SurfaceForcing,
    *,
    melt_point_k: float = T_MELT_K,
    available_kg_m2: Any | None = None,
) -> SnowStep:
    """One balance step with the melting cap (§6.1 with a phase change).

    ``available_kg_m2`` is the snow actually present. When the surplus would melt more than there
    is, only what exists melts and **the rest goes back into temperature** -- the surface is bare
    ground now and is free to warm. Leaving it out means an unlimited pack, which is the right
    default for a scene showing a snowfield and the wrong one for a patch that is about to
    disappear.
    """
    if dt_s <= 0.0:
        raise ValueError("dt_s must be positive")
    t0 = np.asarray(temperature_k, dtype=np.float64)
    if np.any(t0 <= 0.0):
        raise ValueError("temperature must be positive (kelvin)")
    capacity = properties.heat_capacity_j_m2_k

    # -- regime 1: already at the cap. The flux is read *at* the melt point, because that is the
    # only temperature this surface has; an RK2 midpoint above it does not exist.
    at_cap_flux = np.asarray(
        net_flux(np.full(np.shape(t0) or (1,), melt_point_k), properties, forcing),
        dtype=np.float64,
    ).reshape(np.shape(t0) or ())
    melting = (t0 >= melt_point_k - CAP_TOLERANCE_K) & (at_cap_flux > 0.0)

    # -- regime 2: everything else steps normally, and a step that carried a cell through the cap
    # hands its surplus enthalpy over rather than discarding it.
    stepped = np.asarray(rk2_step(t0, dt_s, properties, forcing), dtype=np.float64)
    overshot = ~melting & (stepped > melt_point_k)

    surplus_j = np.where(melting, at_cap_flux * dt_s, 0.0) + np.where(
        overshot, capacity * (stepped - melt_point_k), 0.0
    )
    melt = surplus_j / L_F_WATER_J_KG

    temperature = np.where(melting | overshot, melt_point_k, stepped)

    if available_kg_m2 is not None:
        available = np.asarray(available_kg_m2, dtype=np.float64)
        if np.any(available < 0.0):
            raise ValueError("available snow mass cannot be negative")
        limited = np.minimum(melt, available)
        # the energy that found no snow to melt warms the surface instead
        leftover_j = (melt - limited) * L_F_WATER_J_KG
        temperature = temperature + np.where(capacity > 0.0, leftover_j / capacity, 0.0)
        melt = limited

    return SnowStep(
        temperature_k=np.asarray(temperature, dtype=np.float64),
        melt_kg_m2=np.asarray(melt, dtype=np.float64),
    )
