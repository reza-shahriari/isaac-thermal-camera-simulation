"""Evolve or freeze, per object (TC.12): a solver that stops at a chosen moment and holds.

docs/physics-model.md §6.5 (the node solvers), §12.3 (the scene's thermal block); ADR 0158.

A warm engine at night, kept warm while the camera films; a road whose diurnal history is the
point of the scene while the car on it is a prop -- both need a per-object switch between
*evolving* (the solve runs) and *frozen* (the object stays where the solve left it), and both
need the frozen object to go on radiating to and convecting with its neighbours. Two spellings,
one rule: ``evolve: false`` holds an object at its spun-up state from t₀; ``freeze_at_s: t``
solves it to ``t`` seconds after the scene start and holds it from then on.

Fields (`irsim.thermal.field.ThermalField`) and networks (`irsim.thermal.network`) carry the
hold themselves, on their own tick. A lumped target (`irsim.thermal.solvers`) is wrapped by
:class:`HeldSolver` here: it steps the inner solver exactly up to the hold and answers with the
held temperature after it, so the scene's ``advance_targets`` loop, the radiators that read a
target's temperature and every consumer of the one-weather rule see the same object they did.
Unset, nothing here is constructed and every scene is what it was, bit for bit.
"""

from __future__ import annotations

from typing import Any

from irsim.thermal.solvers import SolverState

__all__ = ["HeldSolver", "hold_from_s"]


def hold_from_s(t0_s: float, evolve: bool, freeze_at_s: float | None) -> float | None:
    """The absolute time an object stops evolving, or ``None`` for one that never does.

    ``evolve: false`` is a hold from t₀; ``freeze_at_s`` is a hold from ``t₀ + t``; both is
    refused by the schema, and neither is ``None``.
    """
    if not evolve:
        return float(t0_s)
    if freeze_at_s is not None:
        if freeze_at_s < 0.0:
            raise ValueError("freeze_at_s is seconds after the scene start and cannot be negative")
        return float(t0_s) + float(freeze_at_s)
    return None


class HeldSolver:
    """A `TemperatureSolver` that evolves until ``hold_from_s`` and then keeps its temperature.

    Steps that end before the hold are passed through untouched, so before the hold the wrapper
    is invisible (bit for bit); a step that straddles the hold advances the inner solver exactly
    to it, and from then on every step returns the held temperature without touching the inner
    solver. ``weather`` is forwarded so the one-weather guard (CLAUDE.md #6) sees through.
    """

    def __init__(self, inner: Any, hold_from_s: float) -> None:
        self._inner = inner
        self.hold_from_s = float(hold_from_s)
        self._held_k: float | None = None
        self._t_s = float(inner.state.t_s)
        if self._t_s >= self.hold_from_s - 1e-9:
            self._held_k = float(inner.temperature())

    @property
    def inner(self) -> Any:
        return self._inner

    @property
    def frozen(self) -> bool:
        return self._held_k is not None

    @property
    def weather(self) -> Any:
        return getattr(self._inner, "weather", None)

    def advance(self, t_s: float, dt_s: float) -> float:
        if dt_s < 0.0:
            raise ValueError("dt_s must not be negative")
        t_end = float(t_s) + float(dt_s)
        if self._held_k is None:
            if t_end <= self.hold_from_s + 1e-9:
                out = float(self._inner.advance(t_s, dt_s))
                self._t_s = t_end
                if t_end >= self.hold_from_s - 1e-9:
                    self._held_k = out
                return out
            # the step straddles the hold: solve exactly to it, then hold
            remaining = self.hold_from_s - float(t_s)
            if remaining > 0.0:
                self._inner.advance(t_s, remaining)
            self._held_k = float(self._inner.temperature())
        self._t_s = t_end
        return self._held_k

    def temperature(self) -> float:
        return self._held_k if self._held_k is not None else float(self._inner.temperature())

    @property
    def state(self) -> SolverState:
        return SolverState(self._t_s, self.temperature())

    def __getattr__(self, item: str) -> Any:
        return getattr(self._inner, item)

    def __repr__(self) -> str:  # pragma: no cover - diagnostics
        return f"HeldSolver({self._inner!r}, hold_from_s={self.hold_from_s})"
