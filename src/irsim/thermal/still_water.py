"""Still water: lakes, ponds and puddles (PH.3).

docs/physics-model.md §6.1, §16.2; ADR 0080 (the sea's cool skin), ADR 0101 (the latent term),
ADR 0108 (this module).

`sea_skin.py` models the ocean: a bulk temperature a scenario authors, a conductive sublayer that
only ever cools, and a warm layer for the afternoon sun. Standing fresh water on land is the same
molecular sublayer over a different body, and two things about it are not the sea's:

* **The skin can be warmer than the bulk, by the sublayer's own mechanism.** ADR 0080 clamps the
  sea's deficit at zero on purpose, because there the warming term is *absorbed sunlight*, which
  is deposited metres down and cannot reverse a gradient across a 2 mm film. On a humid overcast
  night over a pond the flux that reverses is longwave (absorbed in ~20 µm of water) and
  condensation (deposited exactly at the interface): both land **in** the sublayer, so the
  gradient genuinely inverts and the skin runs above the bulk. Clamping here would throw away
  the one night-time signature that separates a pond from a cold plate.
* **A pond is a lumped body, not a boundary condition.** Its mixed layer is
  ``C = rho c d`` per square metre -- 20 mm of water is 83 kJ m^-2 K^-1, twenty times a 10 cm
  asphalt slab -- so it is the thermal mass that makes a puddle read cold at dawn and lag the
  road all afternoon. That capacity, water's emissivity and absorptivity, and a full water film
  for the latent term are all **per-cell** quantities, which is why a puddle is a *region of a
  road's patch* (`water:` on a surface) rather than a surface of its own.

**What a camera sees is not the kinetic temperature.** Water's Fresnel emissivity falls from 0.99
at nadir toward zero at grazing, so an off-nadir look mixes in a reflection of a cold sky:
:func:`apparent_temperature_k` is that mix, and it is the difference between a puddle that reads
like wet asphalt and one that reads like a hole in the road.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from irsim.radiometry.constants import (
    DENSITY_AIR_SEA_LEVEL,
    DENSITY_FRESH_WATER,
    KINEMATIC_VISCOSITY_FRESH_WATER,
    SPECIFIC_HEAT_FRESH_WATER,
    THERMAL_CONDUCTIVITY_FRESH_WATER,
)

__all__ = [
    "DEFAULT_STILL_WATER",
    "StillWaterParams",
    "apparent_temperature_k",
    "friction_velocity_m_s",
    "mixed_layer_capacity_j_m2_k",
    "skin_offset_k",
    "skin_temperature_k",
    "sublayer_thickness_m",
]


@dataclass(frozen=True)
class StillWaterParams:
    """The parameterisation choices for a still-water skin, kept apart from the constants.

    Everything in :mod:`irsim.radiometry.constants` is a property of water or air; everything
    here is a modelling decision with a defensible range. Saunders (1967) put ``lambda`` between
    5 and 10 and later work makes it a function of the surface buoyancy flux; 7 is mid-range.
    ``max_thickness_m`` is the free-convection bound the Saunders form needs as the wind drops
    (the same device and the same reason as ADR 0080's). Together they put the sublayer at
    **0.67 mm at 8.2 m/s and 3.61 mm at 0.8 m/s**, which is the 0.7-3.6 mm band the survey
    quotes for standing water; that is a calibration of two free parameters inside the
    literature's own range, and it is stated here rather than buried.
    """

    saunders_lambda: float = 7.0
    max_thickness_m: float = 3.8e-3
    drag_coefficient: float = 1.3e-3

    def __post_init__(self) -> None:
        if not 1.0 <= self.saunders_lambda <= 20.0:
            raise ValueError("saunders_lambda outside any defensible range (Saunders: 5-10)")
        if self.max_thickness_m <= 0.0 or self.drag_coefficient <= 0.0:
            raise ValueError("max_thickness_m and drag_coefficient must be positive")


DEFAULT_STILL_WATER = StillWaterParams()


def friction_velocity_m_s(
    wind_speed_m_s: Any, params: StillWaterParams = DEFAULT_STILL_WATER
) -> NDArray[np.float64]:
    """``u* = U sqrt(rho_air C_D / rho_water)``: the wind stress, carried into the water."""
    u = np.asarray(wind_speed_m_s, dtype=np.float64)
    if np.any(u < 0.0):
        raise ValueError("wind_speed_m_s cannot be negative")
    return np.asarray(
        u * np.sqrt(DENSITY_AIR_SEA_LEVEL * params.drag_coefficient / DENSITY_FRESH_WATER)
    )


def sublayer_thickness_m(
    wind_speed_m_s: Any, params: StillWaterParams = DEFAULT_STILL_WATER
) -> NDArray[np.float64]:
    """``delta = delta_max tanh(lambda nu / (u* delta_max))``, the conductive film's thickness.

    Saunders' form where the surface is wind-stirred, smoothly bounded where it is not: the tanh
    agrees with ``lambda nu / u*`` to third order while the Saunders term is small, so the
    thickness stays strictly decreasing in wind instead of acquiring a corner at the crossover.
    """
    u_star = friction_velocity_m_s(wind_speed_m_s, params)
    with np.errstate(divide="ignore"):
        saunders = params.saunders_lambda * KINEMATIC_VISCOSITY_FRESH_WATER / u_star
    limit = params.max_thickness_m
    return np.asarray(limit * np.tanh(np.where(np.isfinite(saunders), saunders, np.inf) / limit))


def skin_offset_k(
    q_net_up_w_m2: Any, wind_speed_m_s: Any, params: StillWaterParams = DEFAULT_STILL_WATER
) -> NDArray[np.float64]:
    """``T_skin - T_bulk = -Q_net delta / k_w``, **signed**, with ``Q_net`` positive upward.

    Negative where heat leaves the water, which is the usual cool skin; positive where the net
    flux at the interface points *into* it, which a humid overcast night over standing water
    really does produce. Unlike the sea's (ADR 0080) this is not clamped, and the reason is
    physical rather than tidy: see the module docstring.
    """
    q = np.asarray(q_net_up_w_m2, dtype=np.float64)
    delta = sublayer_thickness_m(wind_speed_m_s, params)
    return np.asarray(-q * delta / THERMAL_CONDUCTIVITY_FRESH_WATER)


def skin_temperature_k(
    bulk_k: Any,
    q_net_up_w_m2: Any,
    wind_speed_m_s: Any,
    params: StillWaterParams = DEFAULT_STILL_WATER,
) -> NDArray[np.float64]:
    """``T_bulk + skin_offset``: the temperature the radiance comes from, not the mixed layer's."""
    bulk = np.asarray(bulk_k, dtype=np.float64)
    if np.any(bulk <= 0.0):
        raise ValueError("bulk_k must be positive kelvin")
    return np.asarray(bulk + skin_offset_k(q_net_up_w_m2, wind_speed_m_s, params))


def mixed_layer_capacity_j_m2_k(depth_m: Any) -> NDArray[np.float64]:
    """``rho c d``: the areal heat capacity of a pond, puddle or lake's mixed layer."""
    d = np.asarray(depth_m, dtype=np.float64)
    if np.any(d <= 0.0):
        raise ValueError("a water body needs a positive depth")
    return np.asarray(DENSITY_FRESH_WATER * SPECIFIC_HEAT_FRESH_WATER * d)


def apparent_temperature_k(
    lut: Any, kinetic_k: Any, emissivity: Any, sky_radiance: Any, quantity: str = "lb"
) -> NDArray[np.float64]:
    """What the camera reads: ``L^-1(eps L_B(T) + (1 - eps) L_sky)`` in the LUT's own band.

    ``emissivity`` is the band-effective *directional* value at the look angle -- for water,
    :func:`irsim.materials.nk.band_directional_emissivity` on the Segelstein table -- and
    ``sky_radiance`` is what the surface reflects, in the units ``quantity`` tabulates.
    Kirchhoff supplies the reflected share as ``1 - eps``, which for opaque water is exact.
    """
    eps = np.asarray(emissivity, dtype=np.float64)
    if np.any((eps < 0.0) | (eps > 1.0)):
        raise ValueError("emissivity must lie in [0, 1]")
    emitted = np.asarray(
        lut.lookup(np.asarray(kinetic_k, dtype=np.float64), quantity), dtype=np.float64
    )
    mixed = eps * emitted + (1.0 - eps) * np.asarray(sky_radiance, dtype=np.float64)
    return np.asarray(lut.apparent_temperature(mixed, quantity), dtype=np.float64)
