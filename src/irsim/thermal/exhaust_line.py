"""The exhaust line as a gas stream in a wall (TC.7).

docs/physics-model.md §6.6 (whose exhaust rows -- manifold, catalyst, pipe, tip -- become
sections of one line here), §6.4; ADR 0096 (the network), ADR 0100 (the engine as a solved
node, whose load this line follows), ADR 0105 (this module).

§6.6 scripts the manifold, the pipe and the tip as three independent schedules with their own
ΔT_max and τ. A camera under a car sees one thing those three cannot produce: a **gradient**
along the line, hottest at the manifold and cooling toward the tip, that moves with load and
that, after key-off, *rises* for a minute or two at the shielded manifold and the catalyst can
before it falls. This module solves it:

* **The gas** is a quasi-one-dimensional stream: ``ṁ c_p dT_g/dx = −h_i π D (T_g − T_w)``.
  Marched segment by segment it is exact for a wall that is uniform over the segment,
  ``T_g,out − T_w = (T_g,in − T_w) exp(−NTU)`` with ``NTU = h_i π D Δx / (ṁ c_p)``, and the
  heat it leaves in the segment is ``ṁ c_p (T_g,in − T_g,out) = G_eff (T_g,in − T_w)`` with
  ``G_eff = ṁ c_p (1 − e^{−NTU})``. That last form is what the network sees: a link of
  conductance ``G_eff`` from the wall node to a fixed node at the segment's inlet gas
  temperature, implicit in the wall (ADR 0096) and lagged one tick in the upstream gas. The
  inner ``h_i`` is Dittus–Boelter on the gas properties of the spec.
* **The wall** is a node per segment (steel or cast iron, ``ρ c π D δ Δx``), convecting to
  ambient outside with a coefficient that is *forced* while the car moves and *natural* when it
  stands (underbody air is ram air, not fan air), radiating to the floor pan and the road, and
  hung from the body through hangers of about 1 W/K each.
* **A catalyst** has an inner mass -- the monolith -- that the gas actually touches, joined to
  the can through the mat; the can never sees the gas directly. A **heat shield** is a thin
  node outside the wall, joined to it by radiation across the gap and cooled by the same
  outside air; where a shield is fitted the camera sees the shield.

The hot soak follows: at key-off the gas stops, the outside cooling drops from forced to
natural, and the shield and the can -- thin skins in front of hot masses -- warm toward the
metal behind them before everything cools. **Every number is ESTIMATED** as §6.6's were; the
acceptance bands are the survey's (MVFRI R04-13 post-stop exhaust temperatures).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from irsim.thermal.network import FixedNode, Link, Node, RadiationLink, ThermalNetwork
from irsim.thermal.solvers import SolverState

__all__ = [
    "ExhaustLine",
    "ExhaustLineSpec",
    "ExhaustSolver",
    "GasFlow",
    "PipeSection",
    "ShieldSpec",
    "dittus_boelter_h",
    "gas_march",
    "stock_exhaust",
]

#: Steel and grey cast iron, the two pipe materials, ρ (kg/m³) and c (J/kg/K).
STEEL = (7800.0, 470.0)
CAST_IRON = (7200.0, 460.0)


@dataclass(frozen=True)
class ShieldSpec:
    """A heat shield: sheet ``thickness_m`` outside the wall across an air gap.

    ``gap_h_w_m2_k`` is the conduction across the still-air gap (k_air / gap ≈ 0.03 / 0.01 =
    3 W m⁻² K⁻¹ for a centimetre); radiation across it is a `RadiationLink` at the shield's
    inner emissivity. The shield is what a camera sees.
    """

    thickness_m: float = 0.5e-3
    gap_h_w_m2_k: float = 3.0
    emissivity_inner: float = 0.5
    emissivity_outer: float = 0.6

    def __post_init__(self) -> None:
        if self.thickness_m <= 0.0 or self.gap_h_w_m2_k < 0.0:
            raise ValueError("a shield needs a positive thickness and a non-negative gap h")


@dataclass(frozen=True)
class PipeSection:
    """One run of the line: manifold, downpipe, catalyst, mid pipe, silencer, tailpipe.

    ``hangers_m`` are positions along the section (from its inlet) where a fastener carries the
    pipe to the body at ``hanger_g_w_k`` each. ``inner_mass_kg`` > 0 makes this a catalyst or
    silencer: the gas touches the inner mass over ``inner_area_m2`` and the wall only through
    ``inner_g_w_k`` (the mat, the baffles' welds).
    """

    name: str
    length_m: float
    diameter_m: float
    wall_m: float
    n_segments: int = 1
    material: tuple[float, float] = STEEL
    emissivity: float = 0.8
    hangers_m: tuple[float, ...] = ()
    hanger_g_w_k: float = 1.0
    inner_mass_kg: float = 0.0
    inner_specific_heat_j_kgk: float = 1000.0
    inner_area_m2: float = 0.0
    inner_g_w_k: float = 0.0
    #: Where the gas still touches the wall of a section with an inner mass: the inlet and
    #: outlet cones of a converter, the end plates of a silencer, m² over the whole section.
    wall_gas_area_m2: float = 0.0
    shield: ShieldSpec | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("a section needs a name")
        if min(self.length_m, self.diameter_m, self.wall_m) <= 0.0 or self.n_segments < 1:
            raise ValueError(f"section {self.name!r}: length, diameter, wall and segments > 0")
        if not 0.0 < self.emissivity <= 1.0:
            raise ValueError(f"section {self.name!r}: emissivity in (0, 1]")
        if any(not 0.0 <= p <= self.length_m for p in self.hangers_m):
            raise ValueError(f"section {self.name!r}: hangers must lie along the section")
        if self.inner_mass_kg > 0.0 and (self.inner_area_m2 <= 0.0 or self.inner_g_w_k <= 0.0):
            raise ValueError(
                f"section {self.name!r}: an inner mass needs a gas contact area and a "
                "conductance to the wall"
            )

    @property
    def segment_length_m(self) -> float:
        return self.length_m / self.n_segments

    @property
    def wall_capacity_j_k(self) -> float:
        """One segment's wall, ``ρ c π D δ Δx``."""
        rho, c = self.material
        return rho * c * math.pi * self.diameter_m * self.wall_m * self.segment_length_m

    @property
    def outer_area_m2(self) -> float:
        """One segment's outside skin area."""
        return math.pi * (self.diameter_m + 2.0 * self.wall_m) * self.segment_length_m

    @property
    def inner_area_per_segment_m2(self) -> float:
        return math.pi * self.diameter_m * self.segment_length_m

    @property
    def has_inner(self) -> bool:
        return self.inner_mass_kg > 0.0


@dataclass(frozen=True)
class GasFlow:
    """Exhaust gas: mass flow and inlet temperature against engine load, and its properties.

    A 2 L petrol engine breathes ~0.015 kg/s at idle and ~0.15 kg/s at rated power; the
    manifold inlet runs ~300 °C at idle and ~850 °C at full load. Both linear in load between
    the two ends, zero flow at key-off. Properties are air-like at ~700 K (ESTIMATED; c_p of the
    burnt gas is a little above air's).
    """

    mdot_idle_kg_s: float = 0.015
    mdot_full_kg_s: float = 0.15
    t_in_idle_k: float = 573.15
    t_in_full_k: float = 1123.15
    c_p_j_kgk: float = 1100.0
    viscosity_pa_s: float = 3.3e-5
    conductivity_w_mk: float = 0.05
    prandtl: float = 0.70

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            if float(getattr(self, name)) <= 0.0:
                raise ValueError(f"GasFlow.{name} must be positive")

    def mdot(self, load: float) -> float:
        if load <= 0.0:
            return 0.0
        return self.mdot_idle_kg_s + (self.mdot_full_kg_s - self.mdot_idle_kg_s) * load

    def t_in(self, load: float) -> float:
        return self.t_in_idle_k + (self.t_in_full_k - self.t_in_idle_k) * max(0.0, load)


def dittus_boelter_h(gas: GasFlow, mdot_kg_s: float, diameter_m: float) -> float:
    """Inside-wall convection ``Nu = 0.023 Re^0.8 Pr^0.4`` (turbulent; laminar floor Nu = 3.66)."""
    if mdot_kg_s <= 0.0:
        return 0.0
    re = 4.0 * mdot_kg_s / (math.pi * diameter_m * gas.viscosity_pa_s)
    nu = max(3.66, 0.023 * re**0.8 * gas.prandtl**0.4)
    return float(nu * gas.conductivity_w_mk / diameter_m)


def gas_march(
    t_in_k: float, t_wall_k: Any, ntu: Any
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Segment-by-segment gas temperatures: ``(inlet of each segment, outlet of each segment)``.

    Exact for a wall uniform over each segment: ``T_out − T_w = (T_in − T_w) e^{−NTU}``.
    """
    walls = np.asarray(t_wall_k, dtype=np.float64)
    ntus = np.broadcast_to(np.asarray(ntu, dtype=np.float64), walls.shape)
    inlet = np.empty_like(walls)
    outlet = np.empty_like(walls)
    t = float(t_in_k)
    for i in range(walls.size):
        inlet[i] = t
        t = walls[i] + (t - walls[i]) * math.exp(-float(ntus[i]))
        outlet[i] = t
    return inlet, outlet


@dataclass(frozen=True)
class ExhaustLineSpec:
    """The line as sections in order, the gas, and how the outside cools it."""

    sections: tuple[PipeSection, ...]
    gas: GasFlow = GasFlow()
    #: Outside convection with ram air under a moving car / at standstill, W m⁻² K⁻¹.
    h_forced_w_m2_k: float = 40.0
    h_natural_w_m2_k: float = 8.0
    #: Where the outside radiation goes: the floor pan above and the road below (both at the
    #: body / ambient temperature here), as view factors of the outer skin.
    view_to_body: float = 0.5
    view_to_road: float = 0.4
    #: The flange: the manifold's conductance to the engine head, W/K -- the gasket (~20 W/K
    #: over four ports) in series with the runners' own cast section (~0.3 W/K each over 10 cm),
    #: so the runners' conduction rules and the manifold can run red while the head sits at the
    #: coolant temperature. One way: the head is a boundary the line reads, the engine does not
    #: feel the line.
    flange_g_w_k: float = 1.5

    def __post_init__(self) -> None:
        if not self.sections:
            raise ValueError("a line needs at least one section")
        names = [s.name for s in self.sections]
        if len(set(names)) != len(names):
            raise ValueError(f"section names must be unique: {names}")
        if self.view_to_body + self.view_to_road > 1.0:
            raise ValueError("the skin's view factors cannot exceed 1 together")
        if min(self.h_forced_w_m2_k, self.h_natural_w_m2_k) <= 0.0 or self.flange_g_w_k < 0.0:
            raise ValueError("outside h must be positive and the flange conductance non-negative")

    @property
    def length_m(self) -> float:
        return float(sum(s.length_m for s in self.sections))


def stock_exhaust() -> ExhaustLineSpec:
    """A mid-size petrol car's line: shielded cast manifold, downpipe, catalyst, mid pipe,
    silencer, tailpipe -- 3.6 m in twenty-two segments. ESTIMATED throughout."""
    return ExhaustLineSpec(
        sections=(
            PipeSection("manifold", 0.30, 0.040, 5e-3, 2, CAST_IRON, 0.85, shield=ShieldSpec()),
            PipeSection("downpipe", 0.50, 0.050, 1.5e-3, 3, STEEL, 0.8, hangers_m=(0.45,)),
            # A 1.2 L cordierite monolith (~0.5 kg/L bulk), its mat ~2.5 W/K to the shell, the
            # gas on the shell only at the two cones; the shell is what a camera sees from below.
            PipeSection(
                "catalyst",
                0.30,
                0.110,
                1.5e-3,
                2,
                STEEL,
                0.8,
                inner_mass_kg=0.6,
                inner_specific_heat_j_kgk=1000.0,
                inner_area_m2=2.5,
                inner_g_w_k=2.5,
                wall_gas_area_m2=0.05,
            ),
            PipeSection("mid_pipe", 1.20, 0.050, 1.5e-3, 6, STEEL, 0.8, hangers_m=(0.3, 0.9)),
            PipeSection(
                "silencer",
                0.50,
                0.180,
                1.0e-3,
                3,
                STEEL,
                0.8,
                hangers_m=(0.25,),
                inner_mass_kg=2.0,
                inner_specific_heat_j_kgk=470.0,
                inner_area_m2=0.6,
                inner_g_w_k=6.0,
            ),
            PipeSection("tailpipe", 0.80, 0.050, 1.5e-3, 6, STEEL, 0.8, hangers_m=(0.4,)),
        )
    )


@dataclass
class ExhaustLine:
    """The line's network and the gas march that drives it, stepped together.

    ``load_at`` is the engine's duty fraction (0 = key off), ``moving_at`` whether ram air
    flows under the car, ``ambient_at`` the air and body temperature, ``head_at`` the engine
    head the manifold is bolted to (``None``: no flange).
    """

    spec: ExhaustLineSpec
    load_at: Callable[[float], float]
    ambient_at: Callable[[float], float]
    t0_s: float
    moving_at: Callable[[float], bool] | None = None
    head_at: Callable[[float], float] | None = None
    initial_k: Any = None
    network: ThermalNetwork = field(init=False, repr=False)
    _segments: list[tuple[PipeSection, int]] = field(init=False, repr=False)
    _gas_in: NDArray[np.float64] = field(init=False, repr=False)
    _gas_g: NDArray[np.float64] = field(init=False, repr=False)
    _wall_g: NDArray[np.float64] = field(init=False, repr=False)
    _gas_out: NDArray[np.float64] = field(init=False, repr=False)
    _gas_heat_w: float = field(init=False, repr=False, default=0.0)

    def __post_init__(self) -> None:
        self._segments = [(s, i) for s in self.spec.sections for i in range(s.n_segments)]
        n = len(self._segments)
        ambient0 = float(self.ambient_at(self.t0_s))
        start = ambient0 if self.initial_k is None else float(self.initial_k)
        self._gas_in = np.full(n, start)
        self._gas_out = np.full(n, start)
        self._gas_g = np.zeros(n)
        self._wall_g = np.zeros(n)
        nodes: list[Node] = []
        fixed: list[FixedNode] = [FixedNode("ambient", self.ambient_at)]
        links: list[Link] = []
        radiation: list[RadiationLink] = []
        initial: dict[str, float] = {}
        if self.head_at is not None:
            fixed.append(FixedNode("head", self.head_at))
        for k, (section, i) in enumerate(self._segments):
            wall = self.wall_name(section.name, i)
            nodes.append(Node(wall, section.wall_capacity_j_k))
            initial[wall] = start
            # The gas: a fixed node at the segment's inlet temperature, a link of G_eff.
            fixed.append(FixedNode(f"gas:{k}", self._gas_reader(k)))
            contact = wall
            if section.has_inner:
                inner = self.inner_name(section.name, i)
                nodes.append(
                    Node(
                        inner,
                        section.inner_mass_kg
                        * section.inner_specific_heat_j_kgk
                        / section.n_segments,
                    )
                )
                initial[inner] = start
                links.append(Link(inner, wall, section.inner_g_w_k / section.n_segments))
                contact = inner
            links.append(Link(contact, f"gas:{k}", self._gas_link(k)))
            if section.has_inner and section.wall_gas_area_m2 > 0.0:
                links.append(Link(wall, f"gas:{k}", self._wall_gas_link(k)))
            # Outside: the wall or its shield convects and radiates.
            skin = wall
            if section.shield is not None:
                shield = self.shield_name(section.name, i)
                rho, c = STEEL
                nodes.append(
                    Node(shield, rho * c * section.shield.thickness_m * section.outer_area_m2)
                )
                initial[shield] = start
                links.append(
                    Link.from_contact(
                        wall, shield, section.shield.gap_h_w_m2_k, section.outer_area_m2
                    )
                )
                eps_gap = 1.0 / (
                    1.0 / section.emissivity + 1.0 / section.shield.emissivity_inner - 1.0
                )
                radiation.append(RadiationLink(wall, shield, eps_gap, section.outer_area_m2))
                skin = shield
                eps_out = section.shield.emissivity_outer
            else:
                eps_out = section.emissivity
            links.append(Link.convection(skin, "ambient", self._h_outside, section.outer_area_m2))
            radiation.append(
                RadiationLink(
                    skin, "ambient", eps_out, section.outer_area_m2, self.spec.view_to_body
                )
            )
            radiation.append(
                RadiationLink(
                    skin, "ambient", eps_out, section.outer_area_m2, self.spec.view_to_road
                )
            )
            # Hangers along this segment.
            lo, hi = i * section.segment_length_m, (i + 1) * section.segment_length_m
            n_hangers = sum(
                1 for p in section.hangers_m if lo <= p < hi or (p == hi == section.length_m)
            )
            if n_hangers:
                links.append(Link(wall, "ambient", n_hangers * section.hanger_g_w_k))
            if k == 0 and self.head_at is not None and self.spec.flange_g_w_k > 0.0:
                links.append(Link(wall, "head", self.spec.flange_g_w_k))
        self.network = ThermalNetwork(
            nodes=nodes,
            fixed=fixed,
            links=links,
            radiation=radiation,
            t0_s=self.t0_s,
            initial_k=initial,
        )
        self._march(self.t0_s)

    # --- names -------------------------------------------------------------------------------

    @staticmethod
    def wall_name(section: str, i: int) -> str:
        return f"{section}:wall{i}"

    @staticmethod
    def inner_name(section: str, i: int) -> str:
        return f"{section}:inner{i}"

    @staticmethod
    def shield_name(section: str, i: int) -> str:
        return f"{section}:shield{i}"

    def skin_name(self, section: str, i: int = 0) -> str:
        """What a camera sees on segment ``i`` of ``section``: its shield, else its wall."""
        sec = self.section(section)
        return (
            self.shield_name(section, i) if sec.shield is not None else self.wall_name(section, i)
        )

    def section(self, name: str) -> PipeSection:
        for s in self.spec.sections:
            if s.name == name:
                return s
        raise KeyError(f"no section {name!r}; the line has {[s.name for s in self.spec.sections]}")

    @property
    def n_segments(self) -> int:
        return len(self._segments)

    # --- the gas ------------------------------------------------------------------------------

    def _gas_reader(self, k: int) -> Callable[[float], float]:
        return lambda _t: float(self._gas_in[k])

    def _gas_link(self, k: int) -> Callable[[float], float]:
        return lambda _t: float(self._gas_g[k])

    def _wall_gas_link(self, k: int) -> Callable[[float], float]:
        return lambda _t: float(self._wall_g[k])

    def _h_outside(self, t_s: float) -> float:
        moving = self.moving_at is not None and bool(self.moving_at(t_s))
        return self.spec.h_forced_w_m2_k if moving else self.spec.h_natural_w_m2_k

    def contact_temperatures_k(self) -> NDArray[np.float64]:
        """The temperature the gas sees in each segment: the inner mass where there is one."""
        out = np.empty(self.n_segments)
        for k, (section, i) in enumerate(self._segments):
            name = (
                self.inner_name(section.name, i)
                if section.has_inner
                else self.wall_name(section.name, i)
            )
            out[k] = self.network.temperature(name)
        return out

    def ntu(self, mdot_kg_s: float) -> NDArray[np.float64]:
        """Per-segment NTU at this flow: ``h_i A / (ṁ c_p)`` over the gas-contact area."""
        if mdot_kg_s <= 0.0:
            return np.zeros(self.n_segments)
        out = np.empty(self.n_segments)
        for k, (section, _i) in enumerate(self._segments):
            h = dittus_boelter_h(self.spec.gas, mdot_kg_s, section.diameter_m)
            area = (
                section.inner_area_m2 / section.n_segments
                if section.has_inner
                else section.inner_area_per_segment_m2
            )
            out[k] = h * area / (mdot_kg_s * self.spec.gas.c_p_j_kgk)
        return out

    def wall_ntu(self, mdot_kg_s: float) -> NDArray[np.float64]:
        """NTU of the gas on the *wall* of a section that has an inner mass (its cones)."""
        out = np.zeros(self.n_segments)
        if mdot_kg_s <= 0.0:
            return out
        for k, (section, _i) in enumerate(self._segments):
            if section.has_inner and section.wall_gas_area_m2 > 0.0:
                h = dittus_boelter_h(self.spec.gas, mdot_kg_s, section.diameter_m)
                area = section.wall_gas_area_m2 / section.n_segments
                out[k] = h * area / (mdot_kg_s * self.spec.gas.c_p_j_kgk)
        return out

    def _march(self, t_s: float) -> None:
        load = float(self.load_at(t_s))
        if not 0.0 <= load <= 1.0:
            raise ValueError(f"load must lie in [0, 1], got {load} at t = {t_s}")
        mdot = self.spec.gas.mdot(load)
        if mdot <= 0.0:
            self._gas_g[:] = 0.0
            self._wall_g[:] = 0.0
            self._gas_in[:] = self.contact_temperatures_k()
            self._gas_out[:] = self._gas_in
            self._gas_heat_w = 0.0
            return
        ntu = self.ntu(mdot)
        wall_ntu = self.wall_ntu(mdot)
        # Where the gas touches both the inner mass and the wall, the segment's outlet follows
        # the conductance-weighted mean of the two, and each gets its own share of ṁ c_p.
        walls = np.array(
            [self.network.temperature(self.wall_name(s.name, i)) for s, i in self._segments]
        )
        contact = self.contact_temperatures_k()
        total = ntu + wall_ntu
        mixed = np.where(
            total > 0.0,
            (ntu * contact + wall_ntu * walls) / np.where(total > 0.0, total, 1.0),
            contact,
        )
        self._gas_in, self._gas_out = gas_march(self.spec.gas.t_in(load), mixed, total)
        mc = mdot * self.spec.gas.c_p_j_kgk
        share = np.where(
            total > 0.0, (1.0 - np.exp(-total)) / np.where(total > 0.0, total, 1.0), 1.0
        )
        self._gas_g = mc * ntu * share
        self._wall_g = mc * wall_ntu * share
        self._gas_heat_w = float(
            np.sum(self._gas_g * (self._gas_in - contact))
            + np.sum(self._wall_g * (self._gas_in - walls))
        )

    @property
    def gas_in_k(self) -> NDArray[np.float64]:
        return np.asarray(self._gas_in.copy())

    @property
    def gas_out_k(self) -> NDArray[np.float64]:
        return np.asarray(self._gas_out.copy())

    def gas_heat_w(self) -> float:
        """Heat the gas left in the line on the last march, ``ṁ c_p (T_in − T_tail)`` exactly."""
        return self._gas_heat_w

    # --- stepping -----------------------------------------------------------------------------

    @property
    def t_s(self) -> float:
        return self.network.t_s

    def advance(self, t_s: float, dt_s: float) -> None:
        """One tick: march the gas on the walls as they stand, then step the network."""
        self._march(t_s)
        self.network.advance(t_s, dt_s)

    def advance_to(self, t_end_s: float, dt_s: float) -> None:
        while self.network.t_s < t_end_s - 1e-9:
            step = min(dt_s, t_end_s - self.network.t_s)
            self.advance(self.network.t_s, step)

    def temperature(self, name: str) -> float:
        return self.network.temperature(name)

    def skin_profile_k(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """``(x along the line at segment centres, skin temperature)`` -- the gradient seen."""
        xs, ts = [], []
        x0 = 0.0
        for section in self.spec.sections:
            for i in range(section.n_segments):
                xs.append(x0 + (i + 0.5) * section.segment_length_m)
                ts.append(self.network.temperature(self.skin_name(section.name, i)))
            x0 += section.length_m
        return np.asarray(xs), np.asarray(ts)


class ExhaustSolver:
    """A `TemperatureSolver` over the line, for the scene's ``targets`` (ADR 0105).

    ``temperature()`` reports the skin of ``section`` (its first segment), so the target drops
    into the slot §6.6's ``exhaust_pipe`` schedule filled; every node stays readable through
    :meth:`node_temperature_k` and the gradient through :attr:`line`.
    """

    def __init__(
        self,
        spec: ExhaustLineSpec,
        weather: Any,
        load_times_s: Any,
        load: Any,
        t0_s: float,
        *,
        section: str = "mid_pipe",
        segment: int = 0,
        moving_at: Callable[[float], bool] | None = None,
        head_at: Callable[[float], float] | None = None,
        initial_k: float | None = None,
        tick_s: float = 5.0,
    ) -> None:
        times = np.asarray(load_times_s, dtype=np.float64)
        duty = np.asarray(load, dtype=np.float64)
        if times.shape != duty.shape or times.ndim != 1 or times.size == 0:
            raise ValueError("load_times_s and load must be one-dimensional and the same length")
        if np.any(np.diff(times) <= 0.0):
            raise ValueError("load_times_s must be strictly increasing")
        if np.any((duty < 0.0) | (duty > 1.0)):
            raise ValueError("load must lie in [0, 1]")
        if tick_s <= 0.0:
            raise ValueError("tick_s must be positive")
        self.spec = spec
        self.weather = weather
        self.tick_s = float(tick_s)
        self._times = times
        self._duty = duty
        self.line = ExhaustLine(
            spec,
            self.load_at,
            lambda t: float(weather.at(t).t_air_k),
            float(t0_s),
            moving_at=moving_at,
            head_at=head_at,
            initial_k=initial_k,
        )
        self._reports = self.line.skin_name(section, segment)
        self._state = SolverState(float(t0_s), self.temperature())

    def load_at(self, t_s: float) -> float:
        return float(np.interp(t_s, self._times, self._duty))

    def node_temperature_k(self, name: str) -> float:
        return self.line.temperature(name)

    def temperature(self) -> float:
        return self.line.temperature(self._reports)

    @property
    def state(self) -> SolverState:
        return self._state

    def advance(self, t_s: float, dt_s: float) -> float:
        if abs(float(t_s) - self.line.t_s) > 1e-9:
            raise ValueError(f"the exhaust line stands at t = {self.line.t_s} s, not {t_s} s")
        self.line.advance_to(float(t_s) + float(dt_s), self.tick_s)
        self._state = SolverState(self.line.t_s, self.temperature())
        return self.temperature()

    def __repr__(self) -> str:  # pragma: no cover - diagnostics
        return f"ExhaustSolver({self._reports}={self.temperature():.1f} K, t={self.line.t_s:.0f} s)"
