"""§6.6's active heat sources: scripted, not predicted, and where most of the signal lives.

§6.6 is explicit that these are "scripted, not predicted", so this module is a set of *schedules*
driven by a :class:`VehicleState` trace rather than a combustion model. That is a deliberate
fidelity choice and not a shortcut: predicting an exhaust tip temperature needs an engine map, a
catalyst model and a flow solver, and the result would still be authored parameters wearing a
physics costume. A schedule with §6.6's own ΔT ranges and time constants is honest about what it is.

Three laws, each chosen because it has a closed form to test against:

* **First-order warm-up and cool-down.** `T = T_air + ΔT_max (1 − e^{−t/τ})` while the source runs,
  Newton cooling after it stops. §6.6 lists a time constant for every row, so this is the law the
  table is already describing.
* **Brakes are an energy deposit, not a schedule.** A braking event dumps ½m(v₁² − v₂²) into the
  discs, so ΔT = E/(m c_p) is arithmetic, and only the *cool-down* is a time constant. Scripting a
  brake temperature directly would make it independent of how hard the car actually braked.
* **Tyres follow speed, not time.** §6.6's "+10 … +35 K, rises with speed" is a steady-state
  relation like the aerial heat sources of ADR 0072 — the contact patch reaches a temperature that
  depends on how fast it is being flexed, with a long time constant to get there.

**Everything here is ESTIMATED.** The defaults are §6.6's own table midpoints; no row is a
measurement, and the module says so in one place rather than in nine.

docs/physics-model.md §6.6; ADR 0038
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "VehicleState",
    "VehicleSourceSolver",
    "SourceHistory",
    "HeatSourceSpec",
    "VEHICLE_HEAT_SOURCES",
    "HUMAN_BODY",
    "first_order_rise",
    "newton_cool",
    "brake_temperature_rise_k",
    "tyre_delta_t_k",
    "source_temperature_k",
]


@dataclass(frozen=True)
class VehicleState:
    """What the vehicle is doing at one instant. The only input the schedules take."""

    t_s: float
    speed_m_s: float = 0.0
    accel_m_s2: float = 0.0
    ignition: bool = False
    braking: bool = False

    def __post_init__(self) -> None:
        if self.speed_m_s < 0.0:
            raise ValueError("speed must be non-negative (use a heading, not a sign)")


@dataclass(frozen=True)
class HeatSourceSpec:
    """One §6.6 row: how hot it gets, how fast, and how fast it forgets.

    ``delta_t_max_k`` is the steady rise over ambient at full load; ``tau_rise_s`` and
    ``tau_cool_s`` are the warm-up and cool-down constants. All ESTIMATED from §6.6's table.
    """

    name: str
    delta_t_max_k: float
    tau_rise_s: float
    tau_cool_s: float
    load_exponent: float = 1.0

    def __post_init__(self) -> None:
        if self.delta_t_max_k < 0.0:
            raise ValueError("delta_t_max_k must be non-negative")
        if self.tau_rise_s <= 0.0 or self.tau_cool_s <= 0.0:
            raise ValueError("time constants must be positive")


#: §6.6's table, as midpoints of its quoted ranges. ESTIMATED, every row.
VEHICLE_HEAT_SOURCES: dict[str, HeatSourceSpec] = {
    "engine_bay": HeatSourceSpec("engine_bay", 65.0, 750.0, 1800.0),
    "exhaust_manifold": HeatSourceSpec("exhaust_manifold", 250.0, 240.0, 600.0),
    "catalytic_converter": HeatSourceSpec("catalytic_converter", 200.0, 390.0, 900.0),
    "exhaust_pipe": HeatSourceSpec("exhaust_pipe", 120.0, 360.0, 700.0),
    "exhaust_tip": HeatSourceSpec("exhaust_tip", 90.0, 360.0, 500.0),
    "brake_disc": HeatSourceSpec("brake_disc", 225.0, 30.0, 300.0),
    "tyre": HeatSourceSpec("tyre", 22.5, 1200.0, 1800.0),
}

#: §6.6's clothed-human row: "+8 … +15 K, effective ε ≈ 0.98; face is warmest". The face runs at
#: the top of the range and covered skin at the bottom, which is the ordering a detector sees.
HUMAN_BODY: dict[str, float] = {"face": 15.0, "hands": 12.0, "clothed_torso": 8.0}


def first_order_rise(t_s: Any, delta_t_max_k: float, tau_s: float) -> NDArray[np.float64]:
    """ΔT(t) = ΔT_max (1 − e^{−t/τ}); exactly (1 − 1/e) of the way there at t = τ."""
    if tau_s <= 0.0:
        raise ValueError("tau_s must be positive")
    t = np.asarray(t_s, dtype=np.float64)
    if np.any(t < 0.0):
        raise ValueError("elapsed time cannot be negative")
    return np.asarray(delta_t_max_k * (1.0 - np.exp(-t / tau_s)))


def newton_cool(t_s: Any, delta_t0_k: float, tau_s: float) -> NDArray[np.float64]:
    """ΔT(t) = ΔT₀ e^{−t/τ}. §6.6: "Newton cooling is exactly right here"."""
    if tau_s <= 0.0:
        raise ValueError("tau_s must be positive")
    t = np.asarray(t_s, dtype=np.float64)
    if np.any(t < 0.0):
        raise ValueError("elapsed time cannot be negative")
    return np.asarray(delta_t0_k * np.exp(-t / tau_s))


def brake_temperature_rise_k(
    mass_kg: float,
    speed_from_m_s: float,
    speed_to_m_s: float,
    disc_mass_kg: float,
    disc_specific_heat_j_kgk: float = 500.0,
    fraction_to_discs: float = 0.9,
) -> float:
    """ΔT = f · ½ m (v₁² − v₂²) / (m_disc c_p) — arithmetic, not a schedule.

    Scripting a brake temperature directly would make it independent of how hard the car actually
    braked, which is the one thing a braking cue is supposed to carry. ``fraction_to_discs``
    accounts for the rest going into the pads, the tyres and the air (ESTIMATED).
    """
    if speed_to_m_s > speed_from_m_s:
        raise ValueError("braking means slowing down")
    for name, value in (("mass_kg", mass_kg), ("disc_mass_kg", disc_mass_kg)):
        if value <= 0.0:
            raise ValueError(f"{name} must be positive")
    if not 0.0 < fraction_to_discs <= 1.0:
        raise ValueError("fraction_to_discs must lie in (0, 1]")
    energy = 0.5 * mass_kg * (speed_from_m_s**2 - speed_to_m_s**2)
    return float(fraction_to_discs * energy / (disc_mass_kg * disc_specific_heat_j_kgk))


def tyre_delta_t_k(
    speed_m_s: Any,
    reference_speed_m_s: float = 30.0,
    spec: HeatSourceSpec | None = None,
) -> NDArray[np.float64]:
    """§6.6's "+10 … +35 K, rises with speed", as a steady-state relation in u = v/v_ref.

    Like ADR 0072's aerial heat sources this has **no time constant in it**: the tyre reaches the
    temperature its current speed implies. That is defensible only where the speed varies slowly
    against the tyre's own 10–30 min constant, and a trace that swings the speed in seconds will
    produce a temperature swing no rubber could follow.
    """
    source = VEHICLE_HEAT_SOURCES["tyre"] if spec is None else spec
    if reference_speed_m_s <= 0.0:
        raise ValueError("reference_speed_m_s must be positive")
    v = np.asarray(speed_m_s, dtype=np.float64)
    if np.any(v < 0.0):
        raise ValueError("speed must be non-negative")
    u = np.clip(v / reference_speed_m_s, 0.0, 1.0)
    return np.asarray(10.0 + (source.delta_t_max_k * 2.0 - 10.0 - 10.0) * u**source.load_exponent)


@dataclass
class SourceHistory:
    """Integrates one source's first-order response along a :class:`VehicleState` trace."""

    spec: HeatSourceSpec
    delta_t_k: float = 0.0
    _last_t_s: float | None = field(default=None, repr=False)

    def step(self, state: VehicleState, load: float = 1.0) -> float:
        """Advance to ``state.t_s`` under a load fraction in [0, 1]; return ΔT over ambient."""
        if not 0.0 <= load <= 1.0:
            raise ValueError("load must lie in [0, 1]")
        return self.step_to_target(state, self.spec.delta_t_max_k * load**self.spec.load_exponent)

    def step_to_target(self, state: VehicleState, target_k: float) -> float:
        """Advance to ``state.t_s`` toward an explicit steady rise; return ΔT over ambient.

        :meth:`step` derives its target from a duty fraction, which is the right statement for an
        engine bay. A tyre is not a duty fraction: §6.6 gives it as a relation in *speed*, and
        `TC.8` drives it from one along a :class:`VehicleState` trace. Both go through this
        method, so the exact-exponential integration and the rise/cool choice by **direction**
        exist once. A second copy of that step is a second copy of §6.6 that can drift (ADR 0089).
        """
        if target_k < 0.0:
            raise ValueError("a rise over ambient cannot be negative")
        if self._last_t_s is None:
            self._last_t_s = state.t_s
            return self.delta_t_k
        dt = state.t_s - self._last_t_s
        if dt < 0.0:
            raise ValueError("a vehicle trace must move forward in time")
        self._last_t_s = state.t_s
        tau = self.spec.tau_rise_s if target_k > self.delta_t_k else self.spec.tau_cool_s
        # exact exponential step, so the result does not depend on the trace's sample spacing
        alpha = 1.0 - math.exp(-dt / tau)
        self.delta_t_k += alpha * (target_k - self.delta_t_k)
        return self.delta_t_k


def source_temperature_k(t_air_k: float, delta_t_k: float) -> float:
    """T = T_air + ΔT. The one place ambient enters, so a source cannot drift off the weather."""
    return float(t_air_k + delta_t_k)


class VehicleSourceSolver:
    """§6.6's first-order rise and cool for one vehicle heat source, as a `TemperatureSolver`.

    The §6.6 table and its integrator (:class:`SourceHistory`) have existed since M6.14 and have
    been reachable only from their own unit tests. The scene schema's `heat_source` solver is ADR
    0072's **aerial** node, whose law is a *steady-state* relation with no time constant at all,
    and an engine bay whose entire character is a 750 s warm-up is precisely what that cannot
    represent. Writing the rise out as a `prescribed` schedule instead -- what the maritime scene
    had to do for its funnel -- moves a number the model already knows into a YAML file, where it
    stops tracking the model and becomes a number somebody typed.

    This class is therefore **only an adapter**: the law, and the exact-exponential step that makes
    the answer independent of the sample spacing, stay in `SourceHistory`. In particular the choice
    between tau_rise and tau_cool stays there too, and it is made by *direction* -- a source whose
    load has just dropped is cooling toward a lower target even though the key is still turned,
    which is the physically meaningful reading and not the one "is the engine on" would give.

    Ambient enters through :func:`source_temperature_k` and nowhere else, so a source cannot drift
    off the shared weather (CLAUDE.md #6); passing the `WeatherSeries` exposes it on `.weather` for
    the `Scene`'s identity assertion.

    ``delta_t0_k`` is the rise the node already carries at ``t0_s``: 0 for a vehicle that has stood
    overnight -- the case the MP.4 demo films -- and non-zero for one that has just parked, which
    is the far more common thing to photograph in daylight.

    docs/physics-model.md §6.6; ADR 0038, ADR 0089
    """

    def __init__(
        self,
        spec: HeatSourceSpec,
        ambient: Any,
        load_s: Any,
        load: Any,
        t0_s: float = 0.0,
        delta_t0_k: float = 0.0,
    ) -> None:
        from irsim.thermal.solvers import SolverState
        from irsim.thermal.weather import WeatherSeries

        self.spec = spec
        times = np.asarray(load_s, dtype=np.float64)
        loads = np.asarray(load, dtype=np.float64)
        if times.ndim != 1 or times.size < 1 or loads.shape != times.shape:
            raise ValueError("load profile needs 1-D times and loads of equal length")
        if times.size > 1 and np.any(np.diff(times) <= 0.0):
            raise ValueError("load profile times must be strictly increasing")
        if np.any(loads < 0.0) or np.any(loads > 1.0):
            raise ValueError("load must lie in [0, 1]")
        if delta_t0_k < 0.0:
            raise ValueError("delta_t0_k must be non-negative")
        self._times = times
        self._loads = loads

        self._weather: WeatherSeries | None = None
        if isinstance(ambient, WeatherSeries):
            self._weather = ambient
            self._ambient: Any = lambda t: ambient.at(t).t_air_k
        elif callable(ambient):
            self._ambient = ambient
        else:
            const = float(ambient)
            self._ambient = lambda _t: const

        self._history = SourceHistory(spec, delta_t_k=float(delta_t0_k))
        # Prime the integrator's clock: its first step only records a time (there is no interval
        # to integrate yet), and leaving that to the first `advance` would silently drop one tick.
        self._history.step(VehicleState(t_s=float(t0_s)), load=self.load_at(float(t0_s)))
        self._state = SolverState(
            float(t0_s), source_temperature_k(self.ambient_at(float(t0_s)), self.delta_t_k)
        )

    # -- inputs ------------------------------------------------------------------------------

    @property
    def weather(self) -> Any:
        """The shared WeatherSeries when the ambient is the weather's air temperature."""
        return self._weather

    @property
    def delta_t_k(self) -> float:
        """The rise over ambient the node carries -- the actual state variable."""
        return float(self._history.delta_t_k)

    def ambient_at(self, t_s: float) -> float:
        return float(self._ambient(float(t_s)))

    def load_at(self, t_s: float) -> float:
        """The load profile, held flat outside its own span rather than extrapolated."""
        return float(np.interp(float(t_s), self._times, self._loads))

    # -- the step ----------------------------------------------------------------------------

    def advance(self, t_s: float, dt_s: float) -> float:
        """Step to ``t_s + dt_s``. The load is taken at the step **midpoint**, so a step that
        spans the moment the key turns is second order rather than depending on which side of the
        event the sample happens to land."""
        from irsim.thermal.solvers import SolverState

        if dt_s < 0.0:
            raise ValueError("dt_s must be non-negative")
        t_s = float(t_s)
        dt = float(dt_s)
        self._history.step(VehicleState(t_s=t_s + dt), load=self.load_at(t_s + 0.5 * dt))
        self._state = SolverState(
            t_s + dt, source_temperature_k(self.ambient_at(t_s + dt), self.delta_t_k)
        )
        return self._state.temperature_k

    def temperature(self) -> float:
        return self._state.temperature_k

    @property
    def state(self) -> Any:
        return self._state

    def __repr__(self) -> str:  # pragma: no cover - diagnostics
        return (
            f"VehicleSourceSolver({self.spec.name!r}, dT={self.delta_t_k:.2f} K, "
            f"T={self._state.temperature_k:.2f} K)"
        )
