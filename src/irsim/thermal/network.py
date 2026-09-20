"""A thermal network: lumped nodes, the links between them, and the boundaries they see (TC.2).

docs/physics-model.md §6.4 (the integrator), §6.6 (the sources this will replace); spec issue
S42 (the §6.8 the model does not have); ADR 0094 (the implicit step), ADR 0096 (this module).

Nothing in the package could connect two solved parts. An engine warmed the bonnet by radiation
onto a parallel panel and never the mounts, brackets or wings it is bolted to -- the owner's
second requirement, and the one every under-hood measurement is about: after key-off the metal
around the block keeps warming for minutes, from the block. That needs a **network**: lumped
nodes with a heat capacity in J/K, links in W/K between them, and the three boundary kinds a
MuSES/TAITherm model uses --

* a **fixed node**, whose temperature is imposed (ambient air, deep ground, a coolant held by
  its thermostat) and which absorbs whatever flows into it;
* **imposed heat** in watts on a node (an engine's loss fraction × power, a resistor);
* **convection to a fluid node** through ``hA``, where the fluid is a node of its own (bay air,
  which has a small capacity and overshoots after key-off) or a fixed one (free air).

Links are what the literature reports: a contact conductance ``h_c·A`` for a bolted joint
(Voller & Tirovic 2007: 7--67 kW m⁻² K⁻¹ ferrous, halve for corrosion) or a total ``G`` in W/K
per fastener or hanger. Both are the same number to the solver and are built to be bit-identical.
A **link node** is a link with mass of its own -- a rubber engine mount -- expanded into a node
with half the resistance on either side, so the series conductance is the one authored and the
mount lags with its own τ. A **radiation link** exchanges ``ε A F σ (T_a⁴ − T_b⁴)``.

**The step.** Everything linear is stepped by backward Euler on `ConductionOperator`'s Laplacian
(ADR 0094 -- L-stable, so a bolt at 25 W/K on a 50 g bracket, τ ≈ 1 s, stands under a 60 s
tick), with fixed nodes eliminated exactly: their temperature at the *end* of the tick is
known, so it moves to the right-hand side. A radiation link is linearised at the start of the
tick, ``G = ε A F σ (T_a² + T_b²)(T_a + T_b)``, which is the exact conductance for the flux at
those temperatures; the matrix therefore changes per tick and is rebuilt per tick, which for a
network of tens of nodes costs nothing. The linearisation converges to the exact steady state
(at steady state the linearised and the true flux coincide) and is first order in dt on the
transient, like the rest of the step.

**What the network conserves, and how a test can see it.** Because the step is one linear solve,
``Σ_i C_i (T_iⁿ⁺¹ − T_iⁿ)/dt = Σ Q_i − (heat into fixed nodes)`` holds to solver precision every
tick, with the flows evaluated exactly as the solver applied them (`link_power_w`). A sign
error in any conductor breaks that identity on the first tick.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from irsim.radiometry.constants import SIGMA_SB
from irsim.thermal.conduction import ConductionOperator

__all__ = [
    "FixedNode",
    "ImposedHeat",
    "Link",
    "LinkNode",
    "Node",
    "RadiationLink",
    "ThermalNetwork",
]

Scalar = float | Callable[[float], float]


def _at(value: Scalar, t_s: float) -> float:
    return float(value(t_s)) if callable(value) else float(value)


@dataclass(frozen=True)
class Node:
    """A lumped part: everything at one temperature, holding ``capacity_j_k`` joules per kelvin."""

    name: str
    capacity_j_k: float

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("a node needs a name")
        if self.capacity_j_k <= 0.0:
            raise ValueError(f"node {self.name!r}: capacity must be positive (J/K)")

    @classmethod
    def from_mass(cls, name: str, mass_kg: float, specific_heat_j_kgk: float) -> Node:
        return cls(name, mass_kg * specific_heat_j_kgk)


@dataclass(frozen=True)
class FixedNode:
    """A boundary whose temperature is imposed: ambient air, deep ground, a thermostatted coolant.

    ``temperature_k`` may be a callable of scene time, so the ambient can follow the weather.
    """

    name: str
    temperature_k: Scalar

    def at(self, t_s: float) -> float:
        return _at(self.temperature_k, t_s)


@dataclass(frozen=True)
class ImposedHeat:
    """Heat in watts delivered to a node -- an engine's loss fraction × power, a heater."""

    node: str
    power_w: Scalar

    def at(self, t_s: float) -> float:
        return _at(self.power_w, t_s)


@dataclass(frozen=True)
class Link:
    """A conductor between two nodes, in W/K. Both authoring forms land here.

    ``from_contact`` is ``h_c · A`` (a bolted joint, a gasket, a contactor) and ``convection`` is
    ``h · A`` to a fluid node; they are the same multiplication, so a link written either way is
    bit-identical to one written as a total ``G``. ``conductance_w_k`` may be a callable of time
    for a coefficient that changes -- forced to natural convection when a vehicle stops.
    """

    a: str
    b: str
    conductance_w_k: Scalar

    def __post_init__(self) -> None:
        if self.a == self.b:
            raise ValueError(f"a link cannot join {self.a!r} to itself")
        if not callable(self.conductance_w_k) and float(self.conductance_w_k) < 0.0:
            raise ValueError(f"link {self.a!r}-{self.b!r}: conductance cannot be negative")

    @staticmethod
    def _times_area(coefficient: Scalar, area_m2: float, what: str) -> Scalar:
        if area_m2 <= 0.0:
            raise ValueError(f"{what} needs a positive area")
        if callable(coefficient):
            h_of_t = coefficient
            return lambda t_s: float(h_of_t(t_s)) * area_m2
        if float(coefficient) < 0.0:
            raise ValueError(f"{what} needs a non-negative coefficient")
        return float(coefficient) * area_m2

    @classmethod
    def from_contact(cls, a: str, b: str, h_c_w_m2_k: Scalar, area_m2: float) -> Link:
        """``h_c · A``: a bolted joint, a gasket, a contactor footprint."""
        return cls(a, b, cls._times_area(h_c_w_m2_k, area_m2, "a contact"))

    @classmethod
    def convection(cls, node: str, fluid: str, h_w_m2_k: Scalar, area_m2: float) -> Link:
        """``h · A`` to a fluid node; ``h`` may be a callable (forced → natural at speed 0)."""
        return cls(node, fluid, cls._times_area(h_w_m2_k, area_m2, "convection"))

    def at(self, t_s: float) -> float:
        return _at(self.conductance_w_k, t_s)


@dataclass(frozen=True)
class LinkNode:
    """A link with mass of its own: a rubber mount, a hanger, a gasket thick enough to lag.

    Expands to a node of ``capacity_j_k`` in the middle of the link with **twice** the authored
    conductance on each side, so the series conductance ``a``→``b`` is exactly the one authored
    and the node answers a step on either side with τ = C / (4 G).
    """

    name: str
    a: str
    b: str
    conductance_w_k: float
    capacity_j_k: float

    def expand(self) -> tuple[Node, Link, Link]:
        if self.conductance_w_k <= 0.0:
            raise ValueError(f"link node {self.name!r}: conductance must be positive")
        half = 2.0 * self.conductance_w_k
        return (
            Node(self.name, self.capacity_j_k),
            Link(self.a, self.name, half),
            Link(self.name, self.b, half),
        )


@dataclass(frozen=True)
class RadiationLink:
    """Grey exchange ``ε A F σ (T_a⁴ − T_b⁴)`` between two nodes, linearised per tick.

    ``emissivity`` is the effective exchange emissivity (1/(1/ε_a + 1/ε_b − 1) for parallel
    plates; ε_a for a small body in a large enclosure) and ``view_factor`` is F_ab from ``a``'s
    area; reciprocity makes the flow the same from either end.
    """

    a: str
    b: str
    emissivity: float
    area_m2: float
    view_factor: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 < self.emissivity <= 1.0:
            raise ValueError("emissivity must lie in (0, 1]")
        if self.area_m2 <= 0.0 or not 0.0 < self.view_factor <= 1.0:
            raise ValueError("a radiation link needs a positive area and a view factor in (0, 1]")

    def conductance_w_k(self, t_a_k: float, t_b_k: float) -> float:
        """The conductance that reproduces the T⁴ flux exactly at (T_a, T_b)."""
        return float(
            self.emissivity
            * self.area_m2
            * self.view_factor
            * SIGMA_SB
            * (t_a_k * t_a_k + t_b_k * t_b_k)
            * (t_a_k + t_b_k)
        )


@dataclass
class ThermalNetwork:
    """The nodes, boundaries and links, stepped together by backward Euler (ADR 0096).

    ``initial_k`` is one temperature for every free node or a mapping by name. Fixed nodes are
    read from their own definitions and never stored.
    """

    nodes: Sequence[Node]
    fixed: Sequence[FixedNode] = ()
    links: Sequence[Link] = ()
    radiation: Sequence[RadiationLink] = ()
    sources: Sequence[ImposedHeat] = ()
    link_nodes: Sequence[LinkNode] = ()
    t0_s: float = 0.0
    initial_k: Any = 293.15
    #: The `WeatherSeries` an ``"ambient"`` fixed node reads, if any, so the scene's one-weather
    #: guard (CLAUDE.md #6) can see that this network follows the same series as everything else.
    weather: Any = None
    _names: list[str] = field(init=False, repr=False)
    _index: dict[str, int] = field(init=False, repr=False)
    _capacity: NDArray[np.float64] = field(init=False, repr=False)
    _state: NDArray[np.float64] = field(init=False, repr=False)
    _t_s: float = field(init=False, repr=False)
    _links: tuple[Link, ...] = field(init=False, repr=False)
    _last_k: NDArray[np.float64] | None = field(init=False, repr=False, default=None)

    def __post_init__(self) -> None:
        nodes = list(self.nodes)
        links = list(self.links)
        for ln in self.link_nodes:
            node, left, right = ln.expand()
            nodes.append(node)
            links.extend((left, right))
        free = [n.name for n in nodes]
        bound = [f.name for f in self.fixed]
        names = free + bound
        if len(set(names)) != len(names):
            raise ValueError(f"node names must be unique across nodes and fixed nodes: {names}")
        if not free:
            raise ValueError("a network needs at least one node with capacity")
        self._names = names
        self._index = {name: i for i, name in enumerate(names)}
        self._capacity = np.array([n.capacity_j_k for n in nodes], dtype=np.float64)
        for link in links:
            self._check_names(link.a, link.b, "link")
        for rad in self.radiation:
            self._check_names(rad.a, rad.b, "radiation link")
        for src in self.sources:
            if src.node not in self._index:
                raise ValueError(f"imposed heat names unknown node {src.node!r}")
            if src.node in bound:
                raise ValueError(
                    f"imposed heat on fixed node {src.node!r} goes nowhere: its temperature is "
                    "imposed, so the power would be silently discarded"
                )
        self._links = tuple(links)
        n_free = len(free)
        if isinstance(self.initial_k, Mapping):
            missing = [n for n in free if n not in self.initial_k]
            if missing:
                raise ValueError(f"initial temperatures missing for {missing}")
            state = np.array([float(self.initial_k[n]) for n in free], dtype=np.float64)
        else:
            state = np.full(n_free, float(self.initial_k), dtype=np.float64)
        if np.any(state <= 0.0):
            raise ValueError("initial temperatures must be positive kelvin")
        self._state = state
        self._t_s = float(self.t0_s)

    # -- shape ----------------------------------------------------------------------------------

    def _check_names(self, a: str, b: str, what: str) -> None:
        for name in (a, b):
            if name not in self._index:
                raise ValueError(f"{what} names unknown node {name!r}; known: {self._names}")

    @property
    def names(self) -> tuple[str, ...]:
        """Free nodes first (in declaration order, link nodes after), then fixed nodes."""
        return tuple(self._names)

    @property
    def n_free(self) -> int:
        return int(self._capacity.shape[0])

    @property
    def n_fixed(self) -> int:
        return len(self.fixed)

    @property
    def capacity_j_k(self) -> NDArray[np.float64]:
        return np.asarray(self._capacity.copy())

    @property
    def t_s(self) -> float:
        return self._t_s

    # -- state ----------------------------------------------------------------------------------

    def _fixed_at(self, t_s: float) -> NDArray[np.float64]:
        return np.array([f.at(t_s) for f in self.fixed], dtype=np.float64)

    def all_temperatures_k(self, t_s: float | None = None) -> NDArray[np.float64]:
        """Free-node state followed by the fixed nodes' values at ``t_s`` (default: now)."""
        t = self._t_s if t_s is None else float(t_s)
        return np.concatenate([self._state, self._fixed_at(t)])

    @property
    def temperatures_k(self) -> dict[str, float]:
        """Every node by name, fixed nodes at the current time."""
        return dict(zip(self._names, self.all_temperatures_k().tolist(), strict=True))

    def temperature(self, name: str) -> float:
        i = self._index[name]
        return (
            float(self._state[i]) if i < self.n_free else self.fixed[i - self.n_free].at(self._t_s)
        )

    # -- the operator at one instant ---------------------------------------------------------------

    def conductances_w_k(
        self, t_s: float, temperatures_k: Any | None = None
    ) -> NDArray[np.float64]:
        """The symmetric (n_all × n_all) conductance matrix the step applies at ``t_s``.

        Conductive links evaluated at ``t_s``; radiation links linearised at ``temperatures_k``
        (default: the current state), which is the exact conductance for the flux at those
        temperatures.
        """
        temps = (
            self.all_temperatures_k(t_s)
            if temperatures_k is None
            else np.asarray(temperatures_k, dtype=np.float64)
        )
        n = len(self._names)
        k = np.zeros((n, n), dtype=np.float64)
        for link in self._links:
            i, j = self._index[link.a], self._index[link.b]
            g = link.at(t_s)
            if g < 0.0:
                raise ValueError(f"link {link.a!r}-{link.b!r} returned a negative conductance")
            k[i, j] += g
            k[j, i] += g
        for rad in self.radiation:
            i, j = self._index[rad.a], self._index[rad.b]
            g = rad.conductance_w_k(float(temps[i]), float(temps[j]))
            k[i, j] += g
            k[j, i] += g
        return k

    def imposed_w(self, t_s: float) -> NDArray[np.float64]:
        """Imposed heat per free node, W, at ``t_s``."""
        q = np.zeros(self.n_free, dtype=np.float64)
        for src in self.sources:
            q[self._index[src.node]] += src.at(t_s)
        return q

    def link_power_w(self, t_s: float | None = None) -> NDArray[np.float64]:
        """Net power into every node (free then fixed), W, at the current state.

        Evaluated with the same conductances the last/next step applies, so the energy identity
        a test checks is the solver's own arithmetic rather than a re-derivation of it.
        """
        t = self._t_s if t_s is None else float(t_s)
        temps = self.all_temperatures_k(t)
        if t_s is None and self._last_k is not None:
            k = self._last_k  # the matrix the last step applied, radiation linearisation included
        else:
            k = self.conductances_w_k(t, temps)
        op = ConductionOperator(k, np.ones(len(self._names)))
        return op.power_in_w(temps)

    # -- the step ---------------------------------------------------------------------------------

    def advance(self, t_s: float, dt_s: float) -> NDArray[np.float64]:
        """Backward Euler from ``t_s`` to ``t_s + dt_s``; returns the free-node temperatures.

        ``(C/dt + L_ff) Tⁿ⁺¹_f = (C/dt) Tⁿ_f + Q(tⁿ⁺¹) − L_fb T_b(tⁿ⁺¹)`` with ``L = diag(K·1) − K``
        from `ConductionOperator` (which is also what checks the matrix). Fixed nodes enter at
        their end-of-tick value, so a boundary that moves during the tick is honoured exactly at
        the tick's end rather than lagged by one step.
        """
        if dt_s <= 0.0:
            raise ValueError("dt_s must be positive")
        if abs(float(t_s) - self._t_s) > 1e-9:
            raise ValueError(f"the network stands at t = {self._t_s} s, not {t_s} s")
        t1 = self._t_s + float(dt_s)
        n_free = self.n_free
        temps = self.all_temperatures_k(t1)  # radiation linearised at the start-of-tick state
        temps[:n_free] = self._state
        k = self.conductances_w_k(t1, temps)
        op = ConductionOperator(k, np.ones(len(self._names)))
        lap = np.asarray(op.laplacian.toarray(), dtype=np.float64)
        l_ff = lap[:n_free, :n_free]
        l_fb = lap[:n_free, n_free:]
        c_dt = self._capacity / float(dt_s)
        matrix = np.diag(c_dt) + l_ff
        rhs = c_dt * self._state + self.imposed_w(t1) - l_fb @ self._fixed_at(t1)
        self._state = np.asarray(np.linalg.solve(matrix, rhs), dtype=np.float64)
        self._t_s = t1
        self._last_k = k
        return np.asarray(self._state.copy())

    def advance_to(self, t_s: float, dt_s: float) -> None:
        """Take equal ticks of ``dt_s`` until ``t_s`` (the last one shorter if needed)."""
        while self._t_s < float(t_s) - 1e-9:
            self.advance(self._t_s, min(float(dt_s), float(t_s) - self._t_s))

    def energy_residual_w(self, before: NDArray[np.float64], dt_s: float) -> float:
        """``Σ C ΔT/dt − (Σ Q − heat into fixed nodes)`` for the tick just taken, W.

        Zero to solver precision for a correct step. ``before`` is the free-node state at the
        start of the tick.
        """
        stored = float(np.sum(self._capacity * (self._state - np.asarray(before)) / dt_s))
        power = self.link_power_w()
        into_fixed = float(np.sum(power[self.n_free :]))
        return stored - (float(np.sum(self.imposed_w(self._t_s))) - into_fixed)
