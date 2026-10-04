"""TemperatureSolver protocol, PrescribedSolver and NewtonCoolingSolver (§6.5, §6.6).

Pluggable solvers behind one interface (thermal-solver skill): ``advance(t, dt)`` steps the
node from t to t + dt and returns the new temperature, ``temperature()`` reads it, ``state`` is
the (time, temperature) pair. Temperatures are float64 throughout; a float32 input is widened
at the boundary, never narrowed (CLAUDE.md #2).

* ``PrescribedSolver`` -- a scripted or measured schedule (engine bay, exhaust, brakes, tyres;
  §6.6 "scripted, not predicted"). Linear interpolation, nodes exact, no extrapolation.
* ``NewtonCoolingSolver`` -- dT/dt = −(T − T∞)/τ with the **exact exponential update**
  T(t+dt) = T∞ + (T(t) − T∞) e^{−dt/τ}, unconditionally stable and free of overshoot at any dt.
  T∞ over a step is taken at the step midpoint (second order for smooth ambient; exact for an
  ambient that steps on a tick boundary). The ambient may be a constant, a callable of time, or
  the shared ``WeatherSeries`` (its T_air), in which case ``.weather`` exposes the object so the
  Scene can assert it is the one everybody else uses (CLAUDE.md #6).

**Scope limit (§6.5):** Newton cooling is the linearised, radiation-free, solar-free special
case of the energy balance. It is right for scripted actors and for objects relaxing with no
solar input (a parked car, residual engine heat, a body in shade). It is wrong for anything
under solar load or exchanging with a cold sky -- panels, roads, roofs -- which need the
two-node environment solver (M6.7+).

docs/physics-model.md §6.5, §6.6
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np

from irsim.thermal.weather import WeatherSeries

__all__ = [
    "SolverState",
    "TemperatureSolver",
    "PrescribedSolver",
    "NewtonCoolingSolver",
    "SOLVER_TYPES",
]


@dataclass(frozen=True)
class SolverState:
    t_s: float
    temperature_k: float


@runtime_checkable
class TemperatureSolver(Protocol):
    """One temperature node advanced on the thermal tick."""

    def advance(self, t_s: float, dt_s: float) -> float:
        """Step from t_s to t_s + dt_s; return the temperature at t_s + dt_s (kelvin)."""
        ...

    def temperature(self) -> float: ...

    @property
    def state(self) -> SolverState: ...


def _as_float64(value: Any, what: str) -> float:
    arr = np.asarray(value)
    if arr.shape != ():
        raise ValueError(f"{what} must be a scalar")
    if not np.issubdtype(arr.dtype, np.floating) and not np.issubdtype(arr.dtype, np.integer):
        raise TypeError(f"{what} must be numeric, got {arr.dtype}")
    out = float(np.asarray(arr, dtype=np.float64))
    if not math.isfinite(out):
        raise ValueError(f"{what} must be finite")
    return out


class PrescribedSolver:
    """Follows a (time, temperature) schedule exactly at the nodes, linearly between them."""

    def __init__(self, times_s: Any, temperatures_k: Any, t0_s: float | None = None) -> None:
        t = np.asarray(times_s, dtype=np.float64)
        temp = np.asarray(temperatures_k, dtype=np.float64)
        if t.ndim != 1 or t.size < 1 or temp.shape != t.shape:
            raise ValueError("schedule needs 1-D times and temperatures of equal length")
        if t.size > 1 and np.any(np.diff(t) <= 0.0):
            raise ValueError("schedule times must be strictly increasing")
        if not np.all(np.isfinite(t)) or not np.all(np.isfinite(temp)) or np.any(temp <= 0.0):
            raise ValueError("schedule must be finite with positive kelvin temperatures")
        self._t = t
        self._temp = temp
        self._t.setflags(write=False)
        self._temp.setflags(write=False)
        start = float(t[0]) if t0_s is None else float(t0_s)
        self._state = SolverState(start, self._lookup(start))

    def _lookup(self, t_s: float) -> float:
        if t_s < self._t[0] or t_s > self._t[-1]:
            raise ValueError(
                f"t = {t_s} s outside the prescribed schedule [{self._t[0]}, {self._t[-1]}] s"
            )
        return float(np.interp(t_s, self._t, self._temp))

    def covers(self, t_s: float) -> bool:
        """Whether ``t_s`` lies inside the schedule, where :meth:`temperature_at` answers."""
        return bool(self._t[0] <= float(t_s) <= self._t[-1])

    def temperature_at(self, t_s: float) -> float:
        """The schedule's temperature at ``t_s``, without moving the solver (ADR 0187's mounts)."""
        return self._lookup(float(t_s))

    def advance(self, t_s: float, dt_s: float) -> float:
        if dt_s < 0.0:
            raise ValueError("dt_s must be non-negative")
        t_new = float(t_s) + float(dt_s)
        self._state = SolverState(t_new, self._lookup(t_new))
        return self._state.temperature_k

    def temperature(self) -> float:
        return self._state.temperature_k

    @property
    def state(self) -> SolverState:
        return self._state

    @property
    def weather(self) -> WeatherSeries | None:
        return None


class NewtonCoolingSolver:
    """dT/dt = −(T − T∞)/τ by the exact exponential update; see the module docstring."""

    def __init__(
        self,
        t0_k: Any,
        tau_s: float,
        ambient: float | WeatherSeries | Callable[[float], float],
        t0_s: float = 0.0,
    ) -> None:
        if not tau_s > 0.0 or not math.isfinite(tau_s):
            raise ValueError("tau_s must be a positive finite time constant")
        self._tau = float(tau_s)
        self._weather: WeatherSeries | None = None
        if isinstance(ambient, WeatherSeries):
            self._weather = ambient
            self._ambient: Callable[[float], float] = lambda t: ambient.at(t).t_air_k
        elif callable(ambient):
            self._ambient = ambient
        else:
            const = _as_float64(ambient, "ambient")
            self._ambient = lambda _t: const
        self._state = SolverState(float(t0_s), _as_float64(t0_k, "t0_k"))

    @property
    def tau_s(self) -> float:
        return self._tau

    @property
    def weather(self) -> WeatherSeries | None:
        """The shared WeatherSeries when the ambient is the weather's air temperature."""
        return self._weather

    def ambient_at(self, t_s: float) -> float:
        return _as_float64(self._ambient(float(t_s)), "ambient temperature")

    def advance(self, t_s: float, dt_s: float) -> float:
        if dt_s < 0.0:
            raise ValueError("dt_s must be non-negative")
        t_s = float(t_s)
        dt = float(dt_s)
        t_inf = self.ambient_at(t_s + 0.5 * dt)
        decay = math.exp(-dt / self._tau)
        t_new = t_inf + (self._state.temperature_k - t_inf) * decay
        self._state = SolverState(t_s + dt, t_new)
        return t_new

    def temperature(self) -> float:
        return self._state.temperature_k

    @property
    def state(self) -> SolverState:
        return self._state


# Static conformance: mypy checks each class against the Protocol here.
SOLVER_TYPES: tuple[type[TemperatureSolver], ...] = (PrescribedSolver, NewtonCoolingSolver)
