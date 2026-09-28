"""A scene's main object, fully solved from its asset (TC.11).

docs/physics-model.md §6.4 (the coupled step), §6.6 (heat sources inside a body), §12.3; ADR 0099
(the coupled solve), ADR 0128 (parts), AI.11 (contacts and hidden parts), TC.16 (mesh members).

Until now an object's heat was authored as offsets and instantaneous nodes -- a motor at
``T_air + ΔT_max·u²`` (ADR 0072), an airframe at ``T_air + offset`` -- and its parts never
touched. ``solve: full`` builds the object's thermal model from what its asset already says:
every shown part is a mesh member with its material's own areal capacity and in-plane
conduction; every hidden part (AI.11) is a lumped member with its mass and c_p, dissipating its
rated watts scaled by the object's duty; every contact is a conductance -- two shown parts by
the cells that stand within a gap of each other, a hidden part by the cells under its footprint,
two hidden parts directly -- priced by the joints table; and, when the scene switches object
exchange on, the parts radiate to each other inside the same solve. All of it is one
`CoupledFields` (one implicit step), so a bolted joint cannot ring and a motor's heat reaches
its mount, its arm and the skin around it in the order the conductances dictate.

No offsets, no instantaneous ΔT(u): a motor's temperature now lags its throttle by its own
time constant and stays hot after it stops. Slow is acceptable; a scene builds this once.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from irsim.thermal.coupling import (
    CoupledFields,
    ExplicitContactor,
    FieldMember,
    LumpedLink,
    LumpedMember,
    footprint_conductances,
    proximity_contactor,
)
from irsim.thermal.facets import FacetForcing

__all__ = ["FullSolve", "build_full_solve", "constant_duty"]


def constant_duty(value: float = 1.0) -> Callable[[float], float]:
    """A duty that never changes: every hidden part at ``value`` of its rated dissipation."""
    v = float(value)
    return lambda t_s: v


@dataclass(frozen=True)
class FullSolve:
    """What `build_full_solve` returns: the coupled solve and how its members map to parts."""

    name: str
    coupled: CoupledFields
    shown: tuple[str, ...]
    hidden: tuple[str, ...]

    def field(self, part: str) -> Any:
        """The `PatchView` of a shown part (what a mesh bridge binds)."""
        return self.coupled.fields[part]

    def node_temperature_k(self, hidden: str) -> float:
        return self.coupled.node_temperature_k(hidden)


def build_full_solve(
    name: str,
    parts: Any,
    mesh_fields: Mapping[str, Any],
    table: Any,
    *,
    t0_s: float,
    tick_s: float,
    duty: Callable[[float], float] | None = None,
    gap_m: float,
    exchange: Any = None,
    keep_ticks: int | None = None,
) -> FullSolve:
    """One coupled solve for an object: its shown parts, hidden parts and contacts.

    ``parts`` is the asset's `PartsConfig` (AI.11); ``mesh_fields`` maps each shown part's
    name to the `TriangleMeshField` the scene already built for it (its patch, properties,
    forcing, spun-up state and conduction are taken from there, so the coupled solve starts
    where the separate one did); ``table`` is the `JointTable`; ``duty`` scales every hidden
    part's ``dissipation_w`` (1 is rated, 0 is off); ``gap_m`` is how close two shown parts'
    cells must stand to share a contact; ``exchange`` is an `ObjectExchange` over the shown
    parts in this order, or ``None``.

    A hidden part without ``mass_kg`` and ``specific_heat_j_kgk`` is refused here: the
    component library (AI.12) that would supply them does not exist yet, and a node without
    capacity is the instantaneous source this step retires.
    """
    duty_fn = duty if duty is not None else constant_duty()
    shown = [p.name for p in parts.parts if p.name in mesh_fields]
    missing = [p.name for p in parts.parts if p.name not in mesh_fields]
    if missing:
        raise ValueError(f"object {name!r}: no mesh field for parts {missing}")
    if not shown:
        raise ValueError(f"object {name!r}: an object needs at least one shown part")
    members: list[FieldMember] = []
    for part in shown:
        fld = mesh_fields[part]
        inner = fld.field
        members.append(
            FieldMember(
                part,
                fld.patch,
                inner.properties,
                inner.forcing_at,
                inner.latest_state_k,
                inner.conduction,
            )
        )
    by_name = {m.name: m for m in members}
    initial_mean = float(np.mean(np.concatenate([np.atleast_1d(m.initial_k) for m in members])))

    lumped: list[LumpedMember] = []
    for h in parts.hidden_parts:
        capacity = h.capacity_j_k
        if capacity is None:
            raise ValueError(
                f"object {name!r}: hidden part {h.name!r} has no mass_kg / specific_heat_j_kgk; "
                "give both (the component library, AI.12, is not here yet)"
            )
        watts = float(h.dissipation_w or 0.0)

        def forcing(t_s: float, _w: float = watts) -> FacetForcing:
            return FacetForcing(
                t_air_k=initial_mean, h_w_m2_k=0.0, q_internal_w_m2=_w * duty_fn(t_s)
            )

        lumped.append(LumpedMember(h.name, float(capacity), forcing, initial_mean))
    hidden_names = {h.name for h in parts.hidden_parts}

    contactors: list[ExplicitContactor] = []
    links: list[LumpedLink] = []
    for c in parts.contacts:
        h_c = float(table.joint(c.joint).h_c_w_m2_k)
        a_hidden, b_hidden = c.a in hidden_names, c.b in hidden_names
        if not a_hidden and not b_hidden:
            k_ab = proximity_contactor(
                by_name[c.a].patch, by_name[c.b].patch, h_c, c.area_m2, gap_m=gap_m
            )
            contactors.append(ExplicitContactor(c.a, c.b, k_ab))
        elif a_hidden and b_hidden:
            # two lumped parts: a unit-area link, so h · 1 is the whole conductance
            links.append(LumpedLink(c.a, c.b, h_c * c.area_m2))
        else:
            field, node = (c.b, c.a) if a_hidden else (c.a, c.b)
            spec = parts.hidden(node)
            g = footprint_conductances(by_name[field].patch, spec.centre_m, c.area_m2, h_c)
            links.append(LumpedLink(field, node, 0.0, per_cell_w_k=g))

    coupled = CoupledFields(
        members,
        contactors,
        t0_s=t0_s,
        tick_s=tick_s,
        keep_ticks=keep_ticks,
        lumped=lumped,
        lumped_links=links,
        exchange=exchange,
        exchange_members=tuple(shown) if exchange is not None else (),
    )
    return FullSolve(name, coupled, tuple(shown), tuple(h.name for h in parts.hidden_parts))
