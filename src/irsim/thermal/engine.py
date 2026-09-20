"""The engine as a solved node, not a schedule (TC.5).

docs/physics-model.md §6.6 (whose ranges become acceptance bands here), §6.4; spec issue S43;
ADR 0096 (the network), ADR 0100 (this module; supersedes ADR 0089's adapter for the bay).

§6.6 scripts the engine bay as ``T = T_air + ΔT_max (1 − e^{−t/τ})`` with τ_rise = 750 s and
τ_cool = 1800 s. A schedule cannot depend on ambient, airflow, load history or the metal bolted to
the block, and it gets the one thing every under-hood measurement agrees on wrong: after key-off
the block cools with a time constant of **hours** (a coolant-temperature diagnostic fits
ECT − ambient to a·exp(−t/τ) and reaches ambient in ~7 h, i.e. τ ≈ 1.7 h), while the skins around
it keep *warming* for a minute or two because the fan-driven airflow stopped and the block did
not. This module replaces the schedule with a small thermal network (ADR 0096):

* **block** -- the block, head and coolant as one lumped node (``m c`` of iron plus glycol), with
  imposed heat ``P_rated · load(t) · block_fraction``: the heat the combustion puts into the
  block and coolant, about the mechanical output at the same load (the thirds rule: a third
  work, a third coolant, a third exhaust). A **thermostat** holds it: once the block passes
  ``thermostat_k`` while running, a large conductance to a radiator node at that temperature
  opens and the coolant loop carries the surplus away; at key-off the loop stops and the block
  cools only through the bay -- which is why a block warms up in minutes and cools in hours;
* **bay_air** -- a fluid node of small capacity that the block convects into and that vents to
  ambient. Both conductances switch: **forced** (fan and ram air) while the engine runs or the
  car moves, **natural** at key-off and standstill. That switch is the hot soak;
* **mounts** -- a rubber link node between block and subframe (SAE 2016-01-0192: a first-order
  lag with its own mass), and the **subframe**, which convects to ambient;
* radiation from the block to the bay's walls and down to the road, as links to ambient.

**Every number is ESTIMATED**, as §6.6's were; what changed is that they are masses, areas and
coefficients that can be measured for a vehicle rather than a rise and a time constant that
cannot be derived. The acceptance bands are §6.6's own: a steady rise of +40…+90 K at full load,
and the survey's key-off figures (> 20 K above ambient after 1 h, within 1 K after 7 h, bay air
overshooting by 20–50 K).

**What the bonnet sees.** ADR 0088's radiator under the bonnet is the hot *mass* in the bay --
the block and what is bolted to it -- so :attr:`EngineSolver.temperature` reports the **block**
as the cavity temperature the bonnet field's forcing carries (TC.6 corrected TC.5's first
choice of the bay air, which at idle is vented within a few kelvin of ambient and would leave
the bonnet flat). The bay air stays readable through :meth:`EngineSolver.node_temperature_k`
and is what a skin's *underside* convects with, a term the bonnet does not yet carry.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from irsim.thermal.network import (
    FixedNode,
    ImposedHeat,
    Link,
    LinkNode,
    Node,
    RadiationLink,
    ThermalNetwork,
)
from irsim.thermal.solvers import SolverState

__all__ = ["EngineSpec", "EngineSolver", "engine_network"]

#: Specific heats, J kg⁻¹ K⁻¹: grey iron / aluminium alloy block as one figure, 50/50 glycol.
C_P_BLOCK_J_KGK = 500.0
C_P_COOLANT_J_KGK = 3400.0
C_P_AIR_J_KGK = 1005.0
RHO_AIR_KG_M3 = 1.16


@dataclass(frozen=True)
class EngineSpec:
    """A mid-size petrol engine in a car park. ESTIMATED throughout; see the module docstring."""

    #: Block, head, manifolds, ancillaries that share the block's temperature, kg.
    block_mass_kg: float = 110.0
    #: Coolant in the block, head and hoses (the radiator's share is outside the bay), kg.
    coolant_mass_kg: float = 7.0
    #: Skin area exchanging with the bay air by convection, m².
    block_area_m2: float = 1.4
    #: Block-to-bay-air convection with the fan running / at standstill, W m⁻² K⁻¹.
    h_forced_w_m2_k: float = 25.0
    h_natural_w_m2_k: float = 7.5
    #: Bay air volume, m³, and its vent conductance to ambient with the fan / by leakage, W/K.
    bay_volume_m3: float = 0.35
    vent_forced_w_k: float = 200.0
    vent_natural_w_k: float = 5.0
    #: Rated mechanical power, W, and the heat into block + coolant as a fraction of it at the
    #: same load (the thirds rule; ESTIMATED). ``load`` is the duty fraction of rated power: an
    #: idle in a car park is ~0.1, a climb at full throttle 1.0.
    rated_power_w: float = 90e3
    block_fraction: float = 0.9
    #: The thermostat: the coolant temperature at which it starts to open, K, the band over
    #: which a wax element opens fully, K, and the coolant loop's conductance to the radiator
    #: when fully open (a large number: the loop moves tens of kilowatts per few kelvin). Closed
    #: -- zero -- below the setpoint and whenever the engine is off.
    thermostat_k: float = 363.15
    thermostat_band_k: float = 6.0
    thermostat_g_w_k: float = 5000.0
    #: Radiation: block emissivity, and the view of the bay walls and of the road below.
    block_emissivity: float = 0.85
    view_to_walls: float = 0.6
    view_to_road: float = 0.35
    #: Rubber mounts as one link node: series conductance block → subframe, W/K, and mass, J/K.
    mounts_g_w_k: float = 6.0
    mounts_capacity_j_k: float = 900.0
    #: The subframe: mass, kg, and its convective loss to ambient, W/K.
    subframe_mass_kg: float = 25.0
    subframe_loss_w_k: float = 6.4

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            if float(getattr(self, name)) <= 0.0:
                raise ValueError(f"EngineSpec.{name} must be positive")
        if not self.block_fraction <= 1.5:
            raise ValueError(
                "block_fraction is the heat into block and coolant per unit of mechanical power; "
                "the thirds rule puts it near 1, and 1.5 is already a poor engine"
            )
        if self.view_to_walls + self.view_to_road > 1.0:
            raise ValueError("the block's view factors cannot exceed 1 together")

    @property
    def block_capacity_j_k(self) -> float:
        return self.block_mass_kg * C_P_BLOCK_J_KGK + self.coolant_mass_kg * C_P_COOLANT_J_KGK

    @property
    def bay_air_capacity_j_k(self) -> float:
        return RHO_AIR_KG_M3 * self.bay_volume_m3 * C_P_AIR_J_KGK

    @property
    def heat_to_block_w(self) -> float:
        """Heat into the block and coolant at full load, W."""
        return self.rated_power_w * self.block_fraction


def engine_network(
    spec: EngineSpec,
    load_at: Callable[[float], float],
    ambient_at: Callable[[float], float],
    t0_s: float,
    *,
    initial_block_k: float | None = None,
    moving_at: Callable[[float], bool] | None = None,
) -> ThermalNetwork:
    """The engine's network at ``t0_s``, its nodes at ambient unless the block is told otherwise.

    ``load_at`` is the duty fraction in [0, 1] (0 = key off); ``moving_at`` says whether ram
    air is present when the engine is off (a coasting car) -- default never. Forced convection
    holds while ``load > 0`` or the car moves; otherwise natural.
    """

    def running(t_s: float) -> bool:
        return float(load_at(t_s)) > 0.0 or (moving_at is not None and bool(moving_at(t_s)))

    def h_block(t_s: float) -> float:
        return spec.h_forced_w_m2_k if running(t_s) else spec.h_natural_w_m2_k

    def vent(t_s: float) -> float:
        return spec.vent_forced_w_k if running(t_s) else spec.vent_natural_w_k

    def power(t_s: float) -> float:
        load = float(load_at(t_s))
        if not 0.0 <= load <= 1.0:
            raise ValueError(f"load must lie in [0, 1], got {load} at t = {t_s}")
        return spec.heat_to_block_w * load

    # The thermostat reads the block at the start of each tick (the same linearisation the
    # radiation links use) and opens proportionally over its band while the engine runs. A
    # proportional element rather than a switch, because a switch evaluated a tick late
    # bang-bangs by Q dt / C per tick; the band makes the fixed point attracting.
    holder: dict[str, ThermalNetwork] = {}

    def thermostat(t_s: float) -> float:
        net = holder.get("net")
        if net is None or not running(t_s):
            return 0.0
        opening = (net.temperature("block") - spec.thermostat_k) / spec.thermostat_band_k
        return spec.thermostat_g_w_k * float(min(1.0, max(0.0, opening)))

    ambient0 = float(ambient_at(t0_s))
    block0 = ambient0 if initial_block_k is None else float(initial_block_k)
    net = ThermalNetwork(
        nodes=[
            Node("block", spec.block_capacity_j_k),
            Node("bay_air", spec.bay_air_capacity_j_k),
            Node.from_mass("subframe", spec.subframe_mass_kg, C_P_BLOCK_J_KGK),
        ],
        fixed=[FixedNode("ambient", ambient_at), FixedNode("radiator", spec.thermostat_k)],
        links=[
            Link.convection("block", "bay_air", h_block, spec.block_area_m2),
            Link("bay_air", "ambient", vent),
            Link("subframe", "ambient", spec.subframe_loss_w_k),
            Link("block", "radiator", thermostat),
        ],
        radiation=[
            RadiationLink(
                "block", "ambient", spec.block_emissivity, spec.block_area_m2, spec.view_to_walls
            ),
            RadiationLink(
                "block", "ambient", spec.block_emissivity, spec.block_area_m2, spec.view_to_road
            ),
        ],
        link_nodes=[
            LinkNode("mounts", "block", "subframe", spec.mounts_g_w_k, spec.mounts_capacity_j_k)
        ],
        sources=[ImposedHeat("block", power)],
        t0_s=t0_s,
        initial_k={"block": block0, "bay_air": ambient0, "subframe": ambient0, "mounts": ambient0},
    )
    holder["net"] = net
    return net


class EngineSolver:
    """A `TemperatureSolver` over the engine network, for the scene's ``targets`` (ADR 0100).

    ``temperature()`` is the **block** -- the radiating mass ADR 0088's bay radiator stands for
    -- so it drops into the slot ADR 0089's `vehicle_source` adapter filled and the bonnet
    field's forcing is unchanged. The bay air and every other node are readable through
    :meth:`node_temperature_k`.
    """

    def __init__(
        self,
        spec: EngineSpec,
        weather: Any,
        load_times_s: Any,
        load: Any,
        t0_s: float,
        *,
        initial_block_k: float | None = None,
        tick_s: float = 10.0,
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
        self.network = engine_network(
            spec,
            self.load_at,
            lambda t: float(weather.at(t).t_air_k),
            float(t0_s),
            initial_block_k=initial_block_k,
        )
        self._state = SolverState(float(t0_s), self.temperature())

    def load_at(self, t_s: float) -> float:
        return float(np.interp(t_s, self._times, self._duty))

    def node_temperature_k(self, name: str) -> float:
        return self.network.temperature(name)

    def temperature(self) -> float:
        return self.network.temperature("block")

    @property
    def state(self) -> SolverState:
        return self._state

    def advance(self, t_s: float, dt_s: float) -> float:
        """Step the network to ``t_s + dt_s`` in sub-ticks of ``tick_s`` (the last one shorter)."""
        if abs(float(t_s) - self.network.t_s) > 1e-9:
            raise ValueError(f"the engine stands at t = {self.network.t_s} s, not {t_s} s")
        self.network.advance_to(float(t_s) + float(dt_s), self.tick_s)
        self._state = SolverState(self.network.t_s, self.temperature())
        return self.temperature()

    def __repr__(self) -> str:  # pragma: no cover - diagnostics
        return (
            f"EngineSolver(block={self.node_temperature_k('block'):.1f} K, "
            f"bay_air={self.temperature():.1f} K, t={self.network.t_s:.0f} s)"
        )
