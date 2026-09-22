"""Fire: what a flame radiates onto a surface, and how hot the air above it is (roadmap PH.7).

docs/physics-model.md §6.1 (where both terms enter the balance), §6.6 (the source rows this
replaces for fire); ADR 0088 (the configuration factor these reuse), ADR 0090 (the clamp),
ADR 0114 (the flame's own appearance, which is `PH.4`'s slab).

A fire reaches a surface by two routes and this module is the pair of them. Neither is the
flame's *image* -- that is a soot slab through `irsim.pipeline.plume`, and it is deliberately a
separate thing, because a camera and a wall do not see the same fire.

**Radiation: a flame is authored by what it emits, not by how hot it is.** The obvious model is
`ε σ T⁴` on a hot rectangle, which is what
:func:`~irsim.thermal.spatial_sources.occluded_longwave_flux` does for an engine bay. It is wrong
for a flame, and not by a little. A flame's emissive power is
set by its soot, its thickness and how much of it is hidden behind smoke, and the temperature and
emissivity that would reproduce it are not separately knowable: 1200 K at ε = 1 is 118 kW/m², and
so is 1500 K at ε = 0.41. Fire protection therefore authors the product directly, as **surface
emissive power** -- :data:`SEP_UNOBSCURED_W_M2` for the luminous part of a pool fire,
:data:`SEP_SMOKE_OBSCURED_W_M2` for the sooty upper region that shrouds a large one -- and a
cell's absorbed flux is

    q_int = F · (α · SEP − ε · L_occluded)

with ``F`` the same configuration factor ADR 0088 already computes, clamped by ADR 0090 where
several radiators overlap. **α is not ε.** A flame radiates mostly between 1 and 5 µm, where a
painted surface can absorb far more than its own long-wave emissivity suggests, so the absorbed
term takes the surface's absorptivity *for the flame's spectrum* and only the sky it blocks takes
ε. Using one number for both is the easy mistake and it is wrong in the direction that matters --
it under-predicts what the fire delivers.

**Convection: the air above a fire is not the weather's air.** A facet over a fire convects to
the plume, not to ambient, and the difference is hundreds of kelvin. Heskestad's correlation gives
the centreline excess above the flame tip,

    ΔT₀ = 9.1 (T_∞ / (g c_p² ρ_∞²))^(1/3) · Q_c^(2/3) · (z − z₀)^(−5/3)

with ``Q_c`` the **convective** heat release in kilowatts and ``z₀`` the virtual origin. It
describes the buoyant plume *above* the flame; inside the continuous flame there is no such decay,
so below the mean flame height the value is held at the tip's. That is not a smoothing choice: it
makes the profile continuous at the tip by construction, and the tip value is
**independent of Q and D** at about 450 K, which is Heskestad's own statement of what the mean
flame height means (the height at which the centreline excess has fallen to roughly 500 K). It is
also the cheapest test this module has, and it is in the suite.

Off the axis the excess falls as a Gaussian in radius. That half-width is the one **ESTIMATED**
number here; everything else is a published correlation.

**Deliberately not modelled:** wind tilt, a plume that is not vertical, flame pulsation, radiation
from the smoke layer as distinct from the flame, and any feedback from the heated surface back to
the fire. Each would change the geometry rather than these two terms.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from irsim.radiometry.constants import (
    C_P_AIR_J_KGK,
    G_STANDARD_M_S2,
    RHO_AIR_STD_KG_M3,
    T_STD_ICAO_K,
)

__all__ = [
    "HESKESTAD_COEFFICIENT",
    "PLUME_SPREAD",
    "SEP_SMOKE_OBSCURED_W_M2",
    "SEP_UNOBSCURED_W_M2",
    "PoolFire",
    "centreline_rise_k",
    "flame_flux_w_m2",
    "flame_height_m",
    "plume_air_temperature_k",
    "virtual_origin_m",
]

#: Surface emissive power of the luminous, unobscured part of a hydrocarbon pool fire, W/m².
#: 100-170 kW/m² across the fuels and pool sizes in Mudan's review and Considine's correlations;
#: the midpoint is the default. ESTIMATED to that spread -- it is a *range* in the literature, not
#: a constant, and a scene that cares should author its own.
SEP_UNOBSCURED_W_M2: Final = 1.35e5
#: The sooty upper region of a large pool fire, which radiates far less because the smoke that
#: shrouds it is cooler than the flame behind it: 30-50 kW/m². Above about 3 m of pool diameter
#: this covers most of the flame's height, which is why a bigger fire is not proportionally more
#: dangerous to stand beside.
SEP_SMOKE_OBSCURED_W_M2: Final = 4.0e4

#: Heskestad's 9.1, with ``c_p`` in kJ/(kg·K) and ``Q_c`` in kW as the correlation is published.
HESKESTAD_COEFFICIENT: Final = 9.1
#: Gaussian half-width of the temperature excess, as a fraction of the height above the virtual
#: origin. The velocity profile's is about 0.12 and the temperature profile is some 10 % wider
#: (Heskestad, SFPE Handbook, "Fire Plumes, Flame Height and Air Entrainment"). **ESTIMATED**: it
#: is the one fitted number in this module, and it only sets how quickly the plume's heat falls
#: away from the axis, never the centreline value the tests pin.
PLUME_SPREAD: Final = 0.13

#: Above this the number was almost certainly written in watts. A 25 MW pool fire -- ten metres
#: of burning hydrocarbon -- has Q_c near 17 500 kW, so nothing physical reaches here, while a
#: 1 MW room fire mistakenly given in watts arrives as 700 000.
Q_C_KW_CEILING = 1.0e5


def _air_density_kg_m3(t_air_k: float) -> float:
    """Ambient density at constant pressure, from the standard value at ICAO's 15 °C."""
    return RHO_AIR_STD_KG_M3 * T_STD_ICAO_K / float(t_air_k)


@dataclass(frozen=True)
class PoolFire:
    """A vertical axisymmetric fire: a pool of diameter ``D`` releasing ``Q_c`` convectively.

    ``q_c_kw`` is the **convective** part of the heat release, which is what drives the plume;
    ``convective_fraction`` turns it back into the total the flame-height and virtual-origin
    correlations want (0.7 is the usual value for a well-ventilated hydrocarbon fire, and a sooty,
    heavily radiating one is lower). ``base_m`` is the centre of the pool and ``axis`` points the
    way the plume rises.
    """

    diameter_m: float
    q_c_kw: float
    base_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    axis: tuple[float, float, float] = (0.0, 0.0, 1.0)
    convective_fraction: float = 0.7
    sep_w_m2: float = SEP_UNOBSCURED_W_M2

    def __post_init__(self) -> None:
        if self.diameter_m <= 0.0:
            raise ValueError("a pool fire needs a positive diameter")
        if self.q_c_kw <= 0.0:
            raise ValueError("a fire needs a positive convective heat release")
        if self.q_c_kw > Q_C_KW_CEILING:
            raise ValueError(
                f"q_c_kw = {self.q_c_kw:g} is {self.q_c_kw / 1000.0:.0f} MW of *convective* heat "
                "release, which no pool fire reaches; Heskestad's correlation takes kilowatts and "
                "this looks like watts. Divide by 1000."
            )
        if not 0.0 < self.convective_fraction <= 1.0:
            raise ValueError("convective_fraction must lie in (0, 1]")
        if self.sep_w_m2 < 0.0:
            raise ValueError("surface emissive power cannot be negative")
        axis = np.asarray(self.axis, dtype=np.float64).reshape(3)
        norm = float(np.linalg.norm(axis))
        if norm <= 0.0:
            raise ValueError("the plume axis must have a direction")
        object.__setattr__(self, "axis", tuple(float(v) for v in axis / norm))
        if self.flame_height_m <= 0.0:
            raise ValueError(
                f"Q_c = {self.q_c_kw:g} kW over a {self.diameter_m:g} m pool gives a mean flame "
                f"height of {self.flame_height_m:.2f} m -- the flame does not clear the pool rim "
                "and Heskestad's correlation does not describe it"
            )

    @property
    def q_kw(self) -> float:
        """Total heat release, which the height correlations take (``Q_c`` is the plume's part)."""
        return self.q_c_kw / self.convective_fraction

    @property
    def flame_height_m(self) -> float:
        return flame_height_m(self.q_kw, self.diameter_m)

    @property
    def virtual_origin_m(self) -> float:
        return virtual_origin_m(self.q_kw, self.diameter_m)


def flame_height_m(q_kw: float, diameter_m: float) -> float:
    """Heskestad's mean flame height ``L = 0.235 Q^(2/5) − 1.02 D``, metres above the pool."""
    return float(0.235 * float(q_kw) ** 0.4 - 1.02 * float(diameter_m))


def virtual_origin_m(q_kw: float, diameter_m: float) -> float:
    """``z₀ = 0.083 Q^(2/5) − 1.02 D``: where a point source would have to sit to match the plume.

    Usually **below** the pool for a fire of any size, which is why heights are measured from the
    base and this is subtracted rather than added.
    """
    return float(0.083 * float(q_kw) ** 0.4 - 1.02 * float(diameter_m))


def centreline_rise_k(height_m: Any, fire: PoolFire, t_air_k: float) -> NDArray[np.float64]:
    """``ΔT₀(z)`` on the plume axis, kelvin above ambient, ``z`` measured from the pool.

    Held at the flame tip's value below the mean flame height: the correlation describes the
    buoyant plume above the flame, and inside the continuous flame there is no ``z^(−5/3)`` decay
    to describe. That makes the profile continuous at the tip by construction, and the tip value
    falls out **independent of Q and D** at about 450 K -- which is what Heskestad's flame height
    means.
    """
    z = np.asarray(height_m, dtype=np.float64)
    if np.any(z < 0.0):
        raise ValueError("height is measured upward from the pool and cannot be negative")
    rho = _air_density_kg_m3(t_air_k)
    c_p_kj = C_P_AIR_J_KGK / 1000.0  # the correlation is published with c_p in kJ/(kg K)
    group = (float(t_air_k) / (G_STANDARD_M_S2 * c_p_kj**2 * rho**2)) ** (1.0 / 3.0)
    reach = np.maximum(z, fire.flame_height_m) - fire.virtual_origin_m
    return np.asarray(
        HESKESTAD_COEFFICIENT * group * fire.q_c_kw ** (2.0 / 3.0) * reach ** (-5.0 / 3.0)
    )


def plume_air_temperature_k(points_m: Any, fire: PoolFire, t_air_k: float) -> NDArray[np.float64]:
    """The air temperature a facet at each point convects to: ambient plus the plume's excess.

    Gaussian in radius about the axis, with the half-width :data:`PLUME_SPREAD` × the height above
    the virtual origin and never narrower than the pool that feeds it. Points level with or below
    the pool are ambient -- the plume rises, and a surface beside a fire is heated by radiation,
    which is the other half of this module.
    """
    points = np.asarray(points_m, dtype=np.float64)
    if points.shape[-1] != 3:
        raise ValueError("points must be (..., 3) in the same frame as the fire's base")
    axis = np.asarray(fire.axis, dtype=np.float64)
    offset = points - np.asarray(fire.base_m, dtype=np.float64)
    z = offset @ axis
    radius = np.linalg.norm(offset - z[..., None] * axis, axis=-1)

    above = z > 0.0
    safe_z = np.where(above, z, 0.0)
    rise = centreline_rise_k(safe_z, fire, t_air_k)
    half_width = np.maximum(PLUME_SPREAD * (safe_z - fire.virtual_origin_m), 0.5 * fire.diameter_m)
    shape = np.exp(-((radius / half_width) ** 2))
    return np.asarray(float(t_air_k) + np.where(above, rise * shape, 0.0))


def flame_flux_w_m2(
    view_factors: Any,
    sep_w_m2: float,
    absorptivity: Any,
    *,
    surface_emissivity: Any = 0.0,
    longwave_down_w_m2: Any = 0.0,
    sky_view: Any = 1.0,
) -> NDArray[np.float64]:
    """``F (α SEP − ε L_occluded)``: what a flame deposits in a cell, as a ``q_internal`` term.

    The two coefficients differ on purpose. ``absorptivity`` is the surface's absorptivity **for
    the flame's spectrum**, which peaks between 1 and 5 µm; ``surface_emissivity`` is its own
    long-wave emissivity, and only the sky the flame blocks is weighted by it. They are the same
    number only for a grey surface, and a painted one is not.

    The occlusion term defaults to nothing, which is right for the common geometry -- a flame
    stands *beside* a wall rather than over it, and takes away no sky worth counting. Where a
    flame does stand over a surface, pass the downwelling and the cell's sky view, exactly as
    :func:`~irsim.thermal.spatial_sources.occluded_longwave_flux` does, or the term invents energy.
    """
    f = np.asarray(view_factors, dtype=np.float64)
    alpha = np.asarray(absorptivity, dtype=np.float64)
    eps = np.asarray(surface_emissivity, dtype=np.float64)
    if np.any(f < 0.0) or np.any(f > 1.0):
        raise ValueError("view factors must lie in [0, 1]")
    if np.any(alpha < 0.0) or np.any(alpha > 1.0):
        raise ValueError("absorptivity must lie in [0, 1]")
    if sep_w_m2 < 0.0:
        raise ValueError("surface emissive power cannot be negative")
    l_down = np.asarray(longwave_down_w_m2, dtype=np.float64)
    v_s = np.asarray(sky_view, dtype=np.float64)
    occluded = np.where(v_s > 0.0, l_down / np.where(v_s > 0.0, v_s, 1.0), 0.0)
    return np.asarray(f * (alpha * float(sep_w_m2) - eps * occluded))
