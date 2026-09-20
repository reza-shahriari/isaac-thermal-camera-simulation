"""An N-layer stack through the thickness of every cell (PT.12).

docs/physics-model.md §6.4 (the two-node model this generalises); ADR 0036 (the two-node
choice), ADR 0094 (the implicit step), ADR 0099 (coupled fields as one solver), ADR 0103.

A field's cell was one node: the whole slab at one temperature, with an adiabatic back. That is
right for a 1 mm skin and wrong for 0.3 m of asphalt, whose surface swings by tens of kelvin in
a day while its base barely moves -- lumped, the slab hardly cools at night because its whole
0.3 m of heat is holding the surface up. MuSES evaluates properties per thermal node through
the element thickness; Fraunhofer's models stored 2 + N temperatures per triangle and used ten
layers. `two_node.py` is the N = 2 case with a resistive back, and this module is the same
physics for any N:

    C_i dT_i/dt = [surface balance on layer 0]
                  + (T_{i−1} − T_i)/R_{i−1,i} − (T_i − T_{i+1})/R_{i,i+1}

with ``R_{i,i+1} = δ_i/(2k_i) + δ_{i+1}/(2k_{i+1})``, §6.4's centre-to-centre resistance
(`two_node.contact_resistance`), and an optional deep boundary ``(T_deep, R_deep)`` under the
last layer, adiabatic by default.

**Nothing new is integrated.** Each layer is a member of a `CoupledFields` (ADR 0099) with the
same grid: layer 0 carries the surface balance and the material's ε and α, the layers below
carry ε = α = 0 and no convection, and consecutive layers are joined by a contactor whose
``h_c = 1/R`` -- so the stack is stepped by ADR 0094's IMEX scheme, the vertical conduction
implicit like every other link, and a 60 s tick stands over a 5 mm layer whose explicit limit
would be seconds. The surface layer is the field a prim binds (`fields[name]`); the layers
below are bookkeeping that never reaches a pixel. Lateral conduction (PT.11) runs in every layer
with that layer's own k δ.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from irsim.thermal.balance import ThermalProperties
from irsim.thermal.conduction import lateral_operator
from irsim.thermal.coupling import Contactor, CoupledFields, FieldMember
from irsim.thermal.facets import FacetForcing, FacetProperties, spin_up
from irsim.thermal.field import DEFAULT_TICK_S
from irsim.thermal.surface_field import DEFAULT_KEEP_TICKS, PlanarPatch
from irsim.thermal.two_node import NodeLayer, contact_resistance

__all__ = ["LayerStack", "layered_field"]


@dataclass(frozen=True)
class LayerStack:
    """The layers under a surface, top first, and what lies beneath the last one.

    ``back_resistance_m2k_w`` / ``deep_temperature_k`` are `TwoNodeProperties`' R₂d and T_deep:
    infinite and ``None`` for an adiabatic back (ADR 0036's default, so nothing leaks to a
    temperature nobody chose).
    """

    layers: tuple[NodeLayer, ...]
    back_resistance_m2k_w: float = math.inf
    deep_temperature_k: float | None = None

    def __post_init__(self) -> None:
        if not self.layers:
            raise ValueError("a stack needs at least one layer")
        if self.back_resistance_m2k_w <= 0.0:
            raise ValueError("back resistance must be positive (use inf for an adiabatic back)")
        if math.isfinite(self.back_resistance_m2k_w) and self.deep_temperature_k is None:
            raise ValueError("a finite back resistance needs a deep_temperature_k (ADR 0036)")
        if self.deep_temperature_k is not None and self.deep_temperature_k <= 0.0:
            raise ValueError("deep_temperature_k must be positive (kelvin)")

    @classmethod
    def uniform(
        cls,
        n_layers: int,
        conductivity_w_mk: float,
        density_kg_m3: float,
        specific_heat_j_kgk: float,
        thickness_m: float,
        **deep: Any,
    ) -> LayerStack:
        """One material cut into ``n_layers`` equal slices -- what a scene's ``layers:`` means."""
        if n_layers < 1:
            raise ValueError("n_layers must be at least 1")
        dz = thickness_m / n_layers
        layer = NodeLayer(dz, conductivity_w_mk, density_kg_m3, specific_heat_j_kgk)
        return cls(tuple(layer for _ in range(n_layers)), **deep)

    @property
    def n_layers(self) -> int:
        return len(self.layers)

    @property
    def thickness_m(self) -> float:
        return float(sum(layer.thickness_m for layer in self.layers))

    def contact_conductances_w_m2_k(self) -> tuple[float, ...]:
        """``1/R_{i,i+1}`` between consecutive layers, §6.4's centre-to-centre form."""
        return tuple(
            1.0 / contact_resistance(a, b)
            for a, b in zip(self.layers[:-1], self.layers[1:], strict=True)
        )


def _sub_layer_forcing(h_w_m2_k: float, t_deep_k: float | None) -> Callable[[float], FacetForcing]:
    """A layer below the surface sees no sky and no air -- only, at the bottom, the deep node."""
    t_ref = 300.0 if t_deep_k is None else float(t_deep_k)

    def at(_t_s: float) -> FacetForcing:
        return FacetForcing(t_air_k=t_ref, h_w_m2_k=h_w_m2_k)

    return at


def layered_field(
    name: str,
    patch: PlanarPatch,
    optical: ThermalProperties,
    stack: LayerStack,
    forcing_at: Callable[[float], FacetForcing],
    t0_s: float,
    initial_k: Any,
    tick_s: float = DEFAULT_TICK_S,
    *,
    lateral: bool = True,
    keep_ticks: int | None = DEFAULT_KEEP_TICKS,
    on_tick: Callable[[float, NDArray[np.float64]], None] | None = None,
    spin_up_hours: float | None = None,
    spin_up_hash: str = "",
    wrap: Callable[[Any], Any] | None = None,
) -> CoupledFields:
    """The stack as one coupled solve; ``fields[name]`` is the surface layer a prim binds.

    ``optical`` supplies ε and α (its capacity is ignored: each layer's is ``ρ c δ`` of its own).
    ``initial_k`` is the surface's starting state (scalar or per cell); the layers below start
    at the same temperature. With ``spin_up_hours`` the whole stack is integrated through the
    hours before ``t0_s`` (through ``wrap``, the scene's weather wrap) so the base carries the
    days before, which is the point of having one.
    """
    n = patch.n_cells
    initial = np.broadcast_to(np.asarray(initial_k, dtype=np.float64), (n,)).astype(np.float64)
    conductances = stack.contact_conductances_w_m2_k()
    members: list[FieldMember] = []
    contactors: list[Contactor] = []
    for i, layer in enumerate(stack.layers):
        surface = i == 0
        bottom = i == stack.n_layers - 1
        props = FacetProperties(
            heat_capacity_j_m2_k=np.full(n, layer.heat_capacity_j_m2_k),
            emissivity=np.full(n, float(optical.emissivity) if surface else 0.0),
            solar_absorptivity=np.full(n, float(optical.solar_absorptivity) if surface else 0.0),
        )
        if surface:
            forcing = forcing_at
        else:
            h_deep = 0.0
            if bottom and math.isfinite(stack.back_resistance_m2k_w):
                h_deep = 1.0 / stack.back_resistance_m2k_w
            forcing = _sub_layer_forcing(h_deep, stack.deep_temperature_k)
        if surface and bottom and math.isfinite(stack.back_resistance_m2k_w):
            raise ValueError(
                "a single-layer stack with a deep boundary is the two-node model's substrate-less "
                "case; add a layer or drop the back resistance"
            )
        member_name = name if surface else f"{name}:layer{i}"
        conduction = (
            lateral_operator(patch, layer.conductivity_w_mk, layer.thickness_m) if lateral else None
        )
        members.append(FieldMember(member_name, patch, props, forcing, initial, conduction))
        if i > 0:
            previous = name if i == 1 else f"{name}:layer{i - 1}"
            contactors.append(Contactor(previous, member_name, conductances[i - 1]))
    coupled = CoupledFields(
        members, contactors, t0_s=t0_s, tick_s=tick_s, keep_ticks=keep_ticks, on_tick=on_tick
    )
    if spin_up_hours is None:
        return coupled
    # Spin the whole stack up on its own forcing and operator, then rebuild it from that state.
    forcing_all = coupled._forcing if wrap is None else wrap(coupled._forcing)
    spun = spin_up(
        coupled.field.properties,
        forcing_all,
        spin_up_hash,
        t0_s,
        hours=float(spin_up_hours),
        dt_s=60.0,
        conduction=coupled.field.conduction,
    ).temperatures_k
    rebuilt = [
        FieldMember(
            m.name, m.patch, m.properties, m.forcing_at, spun[i * n : (i + 1) * n], m.conduction
        )
        for i, m in enumerate(members)
    ]
    return CoupledFields(
        rebuilt, contactors, t0_s=t0_s, tick_s=tick_s, keep_ticks=keep_ticks, on_tick=on_tick
    )
