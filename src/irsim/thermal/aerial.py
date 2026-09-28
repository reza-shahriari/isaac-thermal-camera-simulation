"""Aerial-target thermal signature: airframe at air temperature, heat sources above it.

An aircraft in flight is not a ground scene. It has no solar-loaded slab in contact with soil, no
diurnal storage and no sky-view geometry worth solving: forced convection at flight speed pins the
skin to within a degree or two of the air it is flying through, and everything a thermal camera
actually keys on -- the motors, the speed controllers, the battery pack, the exhaust -- is an
*internal dissipation* whose temperature is set by throttle, not by weather. So the target model
here is a set of **prescribed** nodes (M6.6 ``PrescribedSolver``), not an energy balance:

    T_node(t) = T_air(t) + ΔT(u(t)),      ΔT(u) = ΔT_max · u^n            (§6.6, ESTIMATED)

with u ∈ [0, 1] the throttle fraction. n = 2 is the ohmic reading: winding and MOSFET loss go as
I²R and current is roughly proportional to throttle, so dissipation -- and, at a fixed convective
conductance, the steady rise above ambient -- goes as u². The ΔT_max values are **ESTIMATED**
(ADR 0072): no public dataset of instrumented multirotor motor temperatures was available, and
they are the parameters a Tier 4 comparison against public aerial IR imagery should re-fit first.

The airframe node follows the shared ``WeatherSeries`` (CLAUDE.md #6: it is injected, never loaded
here), optionally with a small constant offset for a sun-soaked upper surface.

**That airframe treatment is a multirotor's, and it does not carry to a fast aircraft.** It says
the skin sits at air temperature because forced convection at *multirotor* speed pins it there,
which is true at 20 m/s and false at 200. A body moving through air is heated by the stagnation of
its own boundary layer, and the skin settles not at the free-stream temperature but at the
**adiabatic wall (recovery) temperature**

    T_r = T_air (1 + r (gamma - 1)/2 M^2),      r = Pr^(1/3) ~ 0.89 (turbulent)      (ADR 0075)

which at M = 0.5 is about 15 K above ambient and at M = 0.8 about 34 K. Applying the multirotor
airframe model to a jet understates its whole skin by that much -- uniformly, plausibly, and in the
direction that makes the target harder to detect than it is. :func:`ram_skin_solver` is the node
for anything fast enough for that to matter.

Nothing in ``docs/physics-model.md`` covers this: §6.6 is an automotive table and the spec has no
aircraft section at all. ADR 0075 records the extension and its sources.

Everything returns a ``PrescribedSolver``, so an aerial target drops into the same
``Scene``/solver plumbing as a ground surface. The schedule is refined until piecewise-linear
interpolation between its nodes reproduces the analytic law to a stated tolerance, which is what
makes "prescribed" a discretisation of a model rather than a table someone typed.

docs/physics-model.md §6.6, §16.2; ADR 0072
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from irsim.radiometry.constants import (
    GAMMA_AIR,
    ISA_LAPSE_RATE_K_PER_M,
    ISA_SEA_LEVEL_T_K,
    PRANDTL_AIR,
    R_SPECIFIC_AIR,
    STANDARD_GRAVITY_M_S2,
)
from irsim.thermal.solvers import PrescribedSolver
from irsim.thermal.weather import WeatherSeries

__all__ = [
    "air_density_ratio",
    "air_temperature_at_altitude_k",
    "RECOVERY_FACTOR_TURBULENT",
    "RECOVERY_FACTOR_LAMINAR",
    "speed_of_sound_m_s",
    "mach_number",
    "recovery_temperature_k",
    "ram_skin_solver",
    "HeatSource",
    "MOTOR",
    "ESC",
    "BATTERY",
    "AERIAL_HEAT_SOURCES",
    "throttle_profile",
    "node_temperature",
    "refine_nodes",
    "prescribed_from_schedule",
    "heat_source_solver",
    "airframe_solver",
]

FloatArray = NDArray[np.float64]

# Linear interpolation of a C² function on a step h errs by at most h²/8·max|f''|, and for a
# quadratic the worst point is the segment midpoint -- so bisecting on the midpoint residual is
# both the cheapest test and, for the u² law, the exact one.
DEFAULT_TOLERANCE_K = 1e-3
MAX_SCHEDULE_NODES = 1 << 16

#: Recovery factor r in T_r = T_air (1 + r (gamma-1)/2 M^2). For a flat plate the boundary-layer
#: result is r = Pr^(1/2) laminar and r = Pr^(1/3) turbulent; a full-scale aircraft skin at flight
#: Reynolds number is turbulent over almost all of its wetted area, so the turbulent value is the
#: default. The two differ by about 6 %, which at M = 0.8 is 2 K -- worth knowing, not worth
#: agonising over. (Standard boundary-layer result; e.g. White, *Viscous Fluid Flow*, ch. 7.)
RECOVERY_FACTOR_TURBULENT = PRANDTL_AIR ** (1.0 / 3.0)
RECOVERY_FACTOR_LAMINAR = PRANDTL_AIR**0.5


@dataclass(frozen=True)
class HeatSource:
    """One dissipating node on an aerial target: ΔT above air = ``delta_t_max_k`` · u^``exponent``.

    ``delta_t_max_k`` is the steady rise at full throttle in still air at the reference condition;
    it is ESTIMATED (ADR 0072). ``exponent`` = 2 is ohmic loss with current ∝ throttle.
    """

    name: str
    delta_t_max_k: float
    exponent: float = 2.0
    reference: str = "ESTIMATED (ADR 0072)"

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("heat source needs a name")
        if not self.delta_t_max_k >= 0.0 or not math.isfinite(self.delta_t_max_k):
            raise ValueError(f"{self.name}: delta_t_max_k must be finite and non-negative")
        if not self.exponent > 0.0 or not math.isfinite(self.exponent):
            raise ValueError(f"{self.name}: exponent must be positive and finite")

    def delta_t_k(self, throttle: object) -> FloatArray:
        """ΔT above air (K) at throttle fraction u ∈ [0, 1]. Vectorised, analytic, float64."""
        u = np.asarray(throttle, dtype=np.float64)
        if np.any(u < 0.0) or np.any(u > 1.0):
            raise ValueError(f"{self.name}: throttle must lie in [0, 1]")
        return np.asarray(self.delta_t_max_k * u**self.exponent, dtype=np.float64)


# ESTIMATED (ADR 0072). Ordering is the physically defensible part: the motor windings are the
# hottest node, the ESC MOSFETs sit below them, and the pack -- large thermal mass, spread over
# many cells -- is the coolest of the three. A Tier 4 fit against public aerial IR should move the
# magnitudes and keep the ordering.
MOTOR = HeatSource("motor", delta_t_max_k=45.0, exponent=2.0)
ESC = HeatSource("esc", delta_t_max_k=30.0, exponent=2.0)
BATTERY = HeatSource("battery", delta_t_max_k=15.0, exponent=2.0)

AERIAL_HEAT_SOURCES: dict[str, HeatSource] = {s.name: s for s in (MOTOR, ESC, BATTERY)}


def throttle_profile(times_s: object, throttle: object) -> Callable[[FloatArray], FloatArray]:
    """A piecewise-linear u(t) from breakpoints, clamped to the end values outside the range."""
    t = np.asarray(times_s, dtype=np.float64)
    u = np.asarray(throttle, dtype=np.float64)
    if t.ndim != 1 or t.size < 1 or u.shape != t.shape:
        raise ValueError("throttle profile needs 1-D times and values of equal length")
    if t.size > 1 and np.any(np.diff(t) <= 0.0):
        raise ValueError("throttle breakpoint times must be strictly increasing")
    if np.any(u < 0.0) or np.any(u > 1.0):
        raise ValueError("throttle must lie in [0, 1]")
    if not np.all(np.isfinite(t)):
        raise ValueError("throttle breakpoint times must be finite")

    def u_of_t(t_query: FloatArray) -> FloatArray:
        return np.asarray(np.interp(np.asarray(t_query, dtype=np.float64), t, u), dtype=np.float64)

    return u_of_t


def node_temperature(
    source: HeatSource,
    weather: WeatherSeries,
    throttle: Callable[[FloatArray], FloatArray],
) -> Callable[[FloatArray], FloatArray]:
    """The analytic node law T(t) = T_air(t) + ΔT(u(t)) as a vectorised callable.

    ``weather`` is the scene's shared series (CLAUDE.md #6) -- T_air comes from it, never from a
    file read here. This is the oracle the prescribed schedule is a discretisation of.
    """

    def t_of_t(t_query: FloatArray) -> FloatArray:
        t = np.asarray(t_query, dtype=np.float64)
        t_air = np.asarray(weather.interpolate(t)["t_air_k"], dtype=np.float64)
        return np.asarray(t_air + source.delta_t_k(throttle(t)), dtype=np.float64)

    return t_of_t


def refine_nodes(
    f: Callable[[FloatArray], FloatArray],
    nodes_s: object,
    tolerance_k: float = DEFAULT_TOLERANCE_K,
    max_nodes: int = MAX_SCHEDULE_NODES,
) -> FloatArray:
    """Bisect ``nodes_s`` until linear interpolation of ``f`` errs by < ``tolerance_k`` everywhere.

    Each pass evaluates every segment midpoint and splits the segments that miss the tolerance;
    for the u^n law the midpoint is the worst point of the segment, so the returned grid carries a
    genuine bound, not a sampled one.
    """
    t = np.unique(np.asarray(nodes_s, dtype=np.float64))
    if t.ndim != 1 or t.size < 2:
        raise ValueError("refinement needs at least two distinct node times")
    if not tolerance_k > 0.0:
        raise ValueError("tolerance_k must be positive")
    while True:
        y = f(t)
        mid = 0.5 * (t[:-1] + t[1:])
        err = np.abs(f(mid) - 0.5 * (y[:-1] + y[1:]))
        split = err > tolerance_k
        if not np.any(split):
            return np.asarray(t, dtype=np.float64)
        if t.size + int(split.sum()) > max_nodes:
            raise ValueError(
                f"schedule needs more than {max_nodes} nodes to reach {tolerance_k} K; "
                "the profile is probably discontinuous"
            )
        t = np.unique(np.concatenate([t, mid[split]]))


def prescribed_from_schedule(
    f: Callable[[FloatArray], FloatArray],
    nodes_s: object,
    tolerance_k: float = DEFAULT_TOLERANCE_K,
) -> PrescribedSolver:
    """A ``PrescribedSolver`` sampling ``f`` on a grid refined to ``tolerance_k`` (§6.6, M6.6)."""
    t = refine_nodes(f, nodes_s, tolerance_k)
    return PrescribedSolver(t, f(t))


def _base_nodes(weather: WeatherSeries, times_s: object) -> FloatArray:
    """Union of the profile breakpoints and the weather samples inside their span."""
    t = np.unique(np.asarray(times_s, dtype=np.float64))
    if t.size < 2:
        raise ValueError("need at least two distinct breakpoint times")
    inside = weather.time_s[(weather.time_s > t[0]) & (weather.time_s < t[-1])]
    return np.asarray(np.unique(np.concatenate([t, inside])), dtype=np.float64)


def heat_source_solver(
    source: HeatSource,
    weather: WeatherSeries,
    times_s: object,
    throttle: object,
    tolerance_k: float = DEFAULT_TOLERANCE_K,
    *,
    altitude_agl_m: float = 0.0,
    lapse_rate_k_per_m: float = ISA_LAPSE_RATE_K_PER_M,
) -> PrescribedSolver:
    """Motor / ESC / battery node as a prescribed schedule over a throttle profile (§6.6).

    The returned solver reproduces T_air(t) + ΔT_max u(t)^n to ``tolerance_k`` at every instant in
    the profile's span, including between its nodes. With ``altitude_agl_m`` the base is the air
    at that height (PT.28); at zero it is the surface series, bit for bit.
    """
    u_of_t = throttle_profile(times_s, throttle)
    base_law = node_temperature(source, weather, u_of_t)
    if altitude_agl_m == 0.0:
        return prescribed_from_schedule(base_law, _base_nodes(weather, times_s), tolerance_k)

    def law(t_query: FloatArray) -> FloatArray:
        return np.asarray(
            air_temperature_at_altitude_k(base_law(t_query), altitude_agl_m, lapse_rate_k_per_m)
        )

    return prescribed_from_schedule(law, _base_nodes(weather, times_s), tolerance_k)


def airframe_solver(
    weather: WeatherSeries,
    times_s: object | None = None,
    offset_k: float = 0.0,
    tolerance_k: float = DEFAULT_TOLERANCE_K,
    *,
    altitude_agl_m: float = 0.0,
    lapse_rate_k_per_m: float = ISA_LAPSE_RATE_K_PER_M,
) -> PrescribedSolver:
    """The airframe node: T_air(t) + ``offset_k``, from the scene's shared weather (CLAUDE.md #6).

    Forced convection at flight speed holds an unpowered skin within a degree or so of the air;
    ``offset_k`` carries a sun-soaked upper surface or an unmodelled internal soak, and is 0 by
    default so an airframe is air temperature unless someone says otherwise.
    """
    if not math.isfinite(offset_k):
        raise ValueError("offset_k must be finite")
    span = weather.time_s if times_s is None else np.asarray(times_s, dtype=np.float64)

    def law(t_query: FloatArray) -> FloatArray:
        t_air = weather.interpolate(np.asarray(t_query, dtype=np.float64))["t_air_k"]
        t_here = air_temperature_at_altitude_k(t_air, altitude_agl_m, lapse_rate_k_per_m)
        return np.asarray(t_here + offset_k, dtype=np.float64)

    return prescribed_from_schedule(law, _base_nodes(weather, span), tolerance_k)


# --- aerodynamic heating of a moving skin (ADR 0075) -----------------------------------------


def air_temperature_at_altitude_k(
    t_air_k: object, altitude_agl_m: float, lapse_rate_k_per_m: float
) -> FloatArray:
    """``T_air − Γ z``: the air a target flies in, from the surface series and the preset's Γ.

    PT.28. The weather file is the surface; a target at 500 m AGL under a 6.5 K/km lapse sits
    in air 3.25 K colder, and everything that reads T_air for it -- the airframe's own
    temperature, a heat source's base, the speed of sound the ram recovery divides by -- reads
    this instead. ``z = 0`` returns the input bit for bit.
    """
    if not math.isfinite(altitude_agl_m) or altitude_agl_m < 0.0:
        raise ValueError("altitude_agl_m must be finite and non-negative")
    if not math.isfinite(lapse_rate_k_per_m) or lapse_rate_k_per_m < 0.0:
        raise ValueError("lapse_rate_k_per_m must be finite and non-negative")
    t = np.asarray(t_air_k, dtype=np.float64)
    if altitude_agl_m == 0.0:
        return np.asarray(t, dtype=np.float64)
    out = t - lapse_rate_k_per_m * altitude_agl_m
    if np.any(out <= 0.0):
        raise ValueError("the lapse rate cools the column below zero kelvin")
    return np.asarray(out, dtype=np.float64)


def air_density_ratio(
    altitude_m: float,
    *,
    lapse_rate_k_per_m: float = ISA_LAPSE_RATE_K_PER_M,
    t0_k: float = ISA_SEA_LEVEL_T_K,
) -> float:
    """``ρ(z)/ρ(0)`` of a hydrostatic column with a constant lapse rate (PT.28).

    ``(1 − Γ z / T₀)^(g/(R Γ) − 1)`` for Γ > 0 and the barometric ``exp(−g z / (R T₀))`` for
    an isothermal column; the ISA values give 0.742 at 3 km and 0.338 at 10 km. The forced
    convection scales its ``v`` by this, because ``h ∝ (ρ v)^n`` for the turbulent
    correlation the default parameters fit. Exactly 1 at ``z = 0``.
    """
    if not math.isfinite(altitude_m) or altitude_m < 0.0:
        raise ValueError("altitude_m must be finite and non-negative")
    if altitude_m == 0.0:
        return 1.0
    g_over_r = STANDARD_GRAVITY_M_S2 / R_SPECIFIC_AIR
    if lapse_rate_k_per_m <= 0.0:
        return float(math.exp(-g_over_r * altitude_m / t0_k))
    base = 1.0 - lapse_rate_k_per_m * altitude_m / t0_k
    if base <= 0.0:
        raise ValueError("altitude above the top of a constant-lapse column")
    return float(base ** (g_over_r / lapse_rate_k_per_m - 1.0))


def speed_of_sound_m_s(t_air_k: object) -> FloatArray:
    """a = sqrt(gamma R T) for dry air, m/s. 340.29 m/s at the ISA sea-level 288.15 K."""
    t = np.asarray(t_air_k, dtype=np.float64)
    if np.any(t <= 0.0):
        raise ValueError("air temperature must be positive kelvin")
    return np.asarray(np.sqrt(GAMMA_AIR * R_SPECIFIC_AIR * t), dtype=np.float64)


def mach_number(speed_m_s: object, t_air_k: object) -> FloatArray:
    """Flight Mach number from true airspeed and the *shared weather's* air temperature.

    Taking the speed of sound from the same ``WeatherSeries`` everything else reads is the point:
    a config that named a Mach number directly would silently mean a different airspeed on a cold
    day than on a warm one (CLAUDE.md #6).
    """
    v = np.asarray(speed_m_s, dtype=np.float64)
    if np.any(v < 0.0):
        raise ValueError("airspeed must be non-negative")
    return np.asarray(v / speed_of_sound_m_s(t_air_k), dtype=np.float64)


def recovery_temperature_k(
    t_air_k: object,
    mach: object,
    recovery_factor: float = RECOVERY_FACTOR_TURBULENT,
) -> FloatArray:
    """Adiabatic wall temperature T_r = T_air (1 + r (gamma-1)/2 M^2), kelvin (ADR 0075).

    The temperature an *unheated, uncooled* skin settles at in a moving airstream: the boundary
    layer brings the flow to rest against the surface and recovers a fraction ``r`` of its kinetic
    energy as heat. It is an equilibrium, not a transient, so it needs no time constant -- which is
    what makes it usable as a prescribed node.

    It is also an **upper bound on the skin alone**: a real airframe loses heat by radiation to a
    cold sky and gains it from the sun and from anything hot inside, none of which is here. For a
    cruising aircraft those are small against the ram term; for a slow one they are not, and then
    :func:`airframe_solver` with an offset is the more honest model.
    """
    if not 0.0 < recovery_factor <= 1.0:
        raise ValueError("the recovery factor lies in (0, 1]")
    t = np.asarray(t_air_k, dtype=np.float64)
    m = np.asarray(mach, dtype=np.float64)
    if np.any(m < 0.0):
        raise ValueError("Mach number must be non-negative")
    return np.asarray(t * (1.0 + recovery_factor * 0.5 * (GAMMA_AIR - 1.0) * m**2))


def ram_skin_solver(
    weather: WeatherSeries,
    speed_m_s: float,
    times_s: object | None = None,
    recovery_factor: float = RECOVERY_FACTOR_TURBULENT,
    tolerance_k: float = DEFAULT_TOLERANCE_K,
    *,
    altitude_agl_m: float = 0.0,
    lapse_rate_k_per_m: float = ISA_LAPSE_RATE_K_PER_M,
) -> PrescribedSolver:
    """An aircraft skin at its recovery temperature, on the scene's shared weather (ADR 0075).

    Constant true airspeed: a flypast lasts seconds and an aircraft does not change speed
    appreciably in that time, whereas the *air* it flies through does change temperature over the
    weather file -- so T_air comes from the series at every node and the airspeed does not.
    """
    if not math.isfinite(speed_m_s) or speed_m_s < 0.0:
        raise ValueError("airspeed must be finite and non-negative")
    span = weather.time_s if times_s is None else np.asarray(times_s, dtype=np.float64)

    def law(t_query: FloatArray) -> FloatArray:
        t_air = np.asarray(
            weather.interpolate(np.asarray(t_query, dtype=np.float64))["t_air_k"],
            dtype=np.float64,
        )
        # PT.28: the air up there, and the speed of sound in it -- the Mach number rises in
        # cold air at the same true airspeed, and the recovery rides on the colder base
        t_here = air_temperature_at_altitude_k(t_air, altitude_agl_m, lapse_rate_k_per_m)
        return recovery_temperature_k(t_here, mach_number(speed_m_s, t_here), recovery_factor)

    return prescribed_from_schedule(law, _base_nodes(weather, span), tolerance_k)
