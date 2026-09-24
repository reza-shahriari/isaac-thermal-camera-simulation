"""A driving vehicle's wheels: the disc that stores braking energy, the tyre that flexes (`TC.8`).

§6.6 gives a ground vehicle's heat sources as a table of steady rises over ambient, and ADR 0089
made that table reachable from a scene through `vehicle_source`. It also said, in as many words,
what this module is for:

    *Tyres and brakes are reachable but should usually not be used through this kind.* §6.6 gives
    the tyre as a **speed** relation and the brake disc as an **energy deposit**, and
    ``vehicle.py`` implements both properly. Driving them from a duty fraction instead is a
    coarser statement.

Both implementations then sat with **no caller at all**: `brake_temperature_rise_k` and
`tyre_delta_t_k` were exercised by their own unit tests and by nothing else, so no frame this
project has ever rendered contained a warm brake. That is what `TC.8` closes, along ADR 0089's own
"revisit when": a driving vehicle, with the two models driven from a :class:`VehicleState` trace.

**The two laws are different shapes and stay different shapes.**

* A **brake disc** is an energy deposit (ADR 0038). Braking from v1 to v2 puts
  ``f · ½ m (v1² − v2²)`` into the discs, so the rise is arithmetic and only the *cool-down* is a
  time constant. A stop from twice the speed deposits exactly four times as much, which is the
  property that makes a braking cue carry how hard the car actually braked.
* A **tyre** is a speed relation with a long constant. §6.6's "+10 … +35 K, rises with speed" is
  a steady state, and the tyre's own τ is 20-30 minutes, so what a frame shows is the tyre's
  *history* of speed and not its current one.

**A parked car has cold wheels, and the relation does not say so by itself.** `tyre_delta_t_k`
returns its lower bound, +10 K, at zero speed, because §6.6's range describes a *rolling* tyre and
its bottom end is a tyre rolling slowly -- not a tyre standing still. Tyre heating is flexing work,
so a stationary tyre has no source at all. The target here is therefore **zero** when the vehicle
is stopped, and the tyre forgets its heat over `tau_cool` (1800 s) rather than instantly: a car
that has just pulled up still has warm tyres, and one that parked an hour ago does not. Reading
the relation literally at v = 0 would give every car in every car park a +10 K wheel.

docs/physics-model.md §6.6; roadmap `TC.8`; ADR 0038 (brakes are an energy deposit), ADR 0089
(the vehicle source solver, and its "revisit when"), ADR 0088 (a radiator facing a panel).
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from irsim.thermal.vehicle import (
    VEHICLE_HEAT_SOURCES,
    SourceHistory,
    VehicleState,
    brake_temperature_rise_k,
    tyre_delta_t_k,
)

__all__ = [
    "FRONT_AXLE_BRAKE_FRACTION",
    "WheelSpec",
    "WheelTemperatures",
    "WheelHistory",
    "DriveCycle",
    "brake_energy_share",
    "passenger_car_wheels",
]

#: Front/rear braking split for an ordinary passenger car under firm braking. Weight transfer puts
#: most of the work on the front axle; 0.65/0.35 is the usual design figure and is **ESTIMATED**
#: here, like every other number §6.6 quotes as a range. It matters to a picture: the front discs
#: run hotter than the rear ones by nearly a factor of two, and a model that split the energy
#: evenly would render four identical wheels, which is not what a thermal camera sees behind a car
#: that has just come off a motorway.
FRONT_AXLE_BRAKE_FRACTION = 0.65


def brake_energy_share(corner: str, front_fraction: float = FRONT_AXLE_BRAKE_FRACTION) -> float:
    """This corner's share of the vehicle's braking energy, from its name's axle.

    ``corner`` ends in ``_f*`` or ``_r*`` (``wheel_fl``, ``rr``, ...). Left and right split evenly:
    a straight-line stop loads both sides alike, and cornering while braking is a manoeuvre this
    model does not carry.
    """
    if not 0.0 < front_fraction < 1.0:
        raise ValueError("front_fraction must lie in (0, 1)")
    axle = corner.rsplit("_", 1)[-1][:1].lower()
    if axle not in ("f", "r"):
        raise ValueError(f"cannot tell which axle {corner!r} is on; name it like 'wheel_fl'")
    return float((front_fraction if axle == "f" else 1.0 - front_fraction) / 2.0)


@dataclass(frozen=True)
class WheelSpec:
    """One corner: the disc that stores the braking energy and its share of it.

    ``disc_mass_kg`` and ``disc_specific_heat_j_kgk`` are ADR 0038's worked example -- an 8 kg
    ventilated disc in cast iron. ``share`` is this corner's fraction of the vehicle's braking
    energy, normally from :func:`brake_energy_share`.
    """

    name: str
    share: float
    disc_mass_kg: float = 8.0
    disc_specific_heat_j_kgk: float = 500.0
    fraction_to_discs: float = 0.9

    def __post_init__(self) -> None:
        if not 0.0 < self.share <= 1.0:
            raise ValueError("share must lie in (0, 1]")
        if self.disc_mass_kg <= 0.0 or self.disc_specific_heat_j_kgk <= 0.0:
            raise ValueError("a disc needs a positive mass and specific heat")
        if not 0.0 < self.fraction_to_discs <= 1.0:
            raise ValueError("fraction_to_discs must lie in (0, 1]")


@dataclass(frozen=True)
class WheelTemperatures:
    """One corner at one instant, as rises over ambient."""

    t_s: float
    disc_delta_t_k: float
    tyre_delta_t_k: float

    def disc_k(self, t_air_k: float) -> float:
        return float(t_air_k + self.disc_delta_t_k)

    def tyre_k(self, t_air_k: float) -> float:
        return float(t_air_k + self.tyre_delta_t_k)


@dataclass
class WheelHistory:
    """One corner walked along a :class:`VehicleState` trace.

    The disc integrates deposits and cooling; the tyre relaxes toward the speed relation through
    :class:`SourceHistory`, so §6.6's exact-exponential step exists in one place (ADR 0089).
    """

    mass_kg: float
    wheel: WheelSpec
    disc_delta_t_k: float = 0.0
    _tyre: SourceHistory = field(init=False, repr=False)
    _last: VehicleState | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.mass_kg <= 0.0:
            raise ValueError("mass_kg must be positive")
        self._tyre = SourceHistory(VEHICLE_HEAT_SOURCES["tyre"])

    @property
    def tyre_delta_t_k(self) -> float:
        return float(self._tyre.delta_t_k)

    def step(self, state: VehicleState) -> WheelTemperatures:
        """Advance this corner to ``state``.

        The disc cools across the interval **before** the deposit lands, because the deposit is
        the work done arriving at ``state`` and the cooling is what happened on the way there.
        The alternative -- deposit, then cool the whole sum -- would let a hard stop lose part of
        its own energy to a time constant it had not yet spent.
        """
        previous, self._last = self._last, state
        spec = VEHICLE_HEAT_SOURCES["brake_disc"]
        if previous is not None:
            dt = state.t_s - previous.t_s
            if dt < 0.0:
                raise ValueError("a vehicle trace must move forward in time")
            self.disc_delta_t_k *= math.exp(-dt / spec.tau_cool_s)
            # A deposit needs braking *and* a speed that actually fell. A trace that slows on
            # engine braking alone puts nothing in the discs, which is the difference between a
            # car lifting off and a car stopping.
            if state.braking and state.speed_m_s < previous.speed_m_s:
                self.disc_delta_t_k += self.wheel.share * brake_temperature_rise_k(
                    self.mass_kg,
                    previous.speed_m_s,
                    state.speed_m_s,
                    self.wheel.disc_mass_kg,
                    self.wheel.disc_specific_heat_j_kgk,
                    self.wheel.fraction_to_discs,
                )
        # Zero when stopped: tyre heating is flexing work, and `tyre_delta_t_k`'s +10 K floor
        # describes a tyre rolling slowly, not one standing still. See this module's docstring.
        target = 0.0 if state.speed_m_s <= 0.0 else float(tyre_delta_t_k(state.speed_m_s))
        self._tyre.step_to_target(state, target)
        return WheelTemperatures(state.t_s, self.disc_delta_t_k, self.tyre_delta_t_k)


@dataclass
class DriveCycle:
    """Every corner of one vehicle, walked along one trace.

    The corners share the trace and nothing else: each carries its own disc temperature, because
    the front axle takes roughly twice the energy of the rear and that difference is the whole
    reason to model four wheels rather than one.
    """

    mass_kg: float
    wheels: Sequence[WheelSpec]
    histories: dict[str, WheelHistory] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        names = [w.name for w in self.wheels]
        if len(set(names)) != len(names):
            raise ValueError(f"two corners share a name: {names}")
        total = sum(w.share for w in self.wheels)
        if total > 1.0 + 1e-9:
            raise ValueError(f"the corners claim {total:.3f} of the braking energy, which is > 1")
        self.histories = {w.name: WheelHistory(self.mass_kg, w) for w in self.wheels}

    def step(self, state: VehicleState) -> dict[str, WheelTemperatures]:
        return {name: h.step(state) for name, h in self.histories.items()}

    def run(self, trace: Iterable[VehicleState]) -> list[dict[str, WheelTemperatures]]:
        """Walk a whole trace, returning every corner at every sample."""
        return [self.step(state) for state in trace]


def passenger_car_wheels(
    front_fraction: float = FRONT_AXLE_BRAKE_FRACTION, **spec: float
) -> tuple[WheelSpec, ...]:
    """The four corners of an ordinary car, named as ``car_demo`` names its wheel prims."""
    return tuple(
        WheelSpec(name=name, share=brake_energy_share(name, front_fraction), **spec)
        for name in ("wheel_fl", "wheel_fr", "wheel_rl", "wheel_rr")
    )
