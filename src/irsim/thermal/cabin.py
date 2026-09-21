"""A lumped cabin-air node, and the greenhouse it makes of a parked car (§6.6, ADR 0038).

A vehicle panel is not a slab on the ground: what is behind it is a closed volume of air that the
sun has been heating through the glass all afternoon. Give the panel an adiabatic back and its
roof comes out at ambient-plus-solar; give it the cabin and the roof is warmer, because it is
losing heat into air that is already hotter than the outside.

    C_cab dT_cab/dt = τ_glz A_glz Q_sol
                      + Σ_p A_p (T_p − T_cab)/R_p
                      + ṁ c_p (T_air − T_cab)

Three terms, one per mechanism §6.6's "five regimes" names: solar gain **through** the glazing (not
absorbed by it -- that is the glass panel's own balance), conduction from every panel that bounds
the cabin, and infiltration, the slow air exchange that stops a sealed car reaching absurd
temperatures.

**The cabin is a boundary condition for the panels and the panels are a source for the cabin**, so
the pair is solved together. Stepping them alternately -- panels against yesterday's cabin, then
cabin against today's panels -- is stable at a 1 s tick and wrong at a 60 s one, in the direction
that under-predicts the greenhouse, which is the effect the node exists to produce.

Every parameter here is ESTIMATED from §6.6's ranges and from the geometry of an ordinary car; none
is a measurement.

**Reaching a scene (PT.15, ADR 0106).** `CabinNode` steps its own panels as lumps. A scene's
panels are *fields* -- cells with their own shadows and sky -- so there the cabin is a
:class:`~irsim.thermal.coupling.LumpedMember` of the panels' `CoupledFields`: the same three
terms, the same conductances, on the one implicit operator that steps every cell. It has to
reproduce this module's coupled equilibrium to 0.1 K (the test holds it to that), or two
copies of one balance would drift apart. :func:`cabin_coupling` builds that member.

docs/physics-model.md §6.6; ADR 0038, ADR 0106
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from irsim.thermal.balance import SurfaceForcing, ThermalProperties, net_flux
from irsim.thermal.coupling import CoupledFields, FieldMember, LumpedLink, LumpedMember
from irsim.thermal.facets import FacetForcing, spin_up
from irsim.thermal.surface_field import DEFAULT_KEEP_TICKS

__all__ = [
    "CabinPanel",
    "CabinNode",
    "CabinState",
    "AIR_SPECIFIC_HEAT_J_KGK",
    "AIR_DENSITY_KG_M3",
    "cabin_coupling",
    "cabin_field",
]

AIR_SPECIFIC_HEAT_J_KGK = 1005.0
AIR_DENSITY_KG_M3 = 1.2


@dataclass(frozen=True)
class CabinPanel:
    """One panel bounding the cabin: its own surface physics plus how it couples inward."""

    name: str
    properties: ThermalProperties
    area_m2: float
    #: Conduction + interior-film resistance from the panel's node to the cabin air, m² K W⁻¹.
    #: ESTIMATED: ~0.13 for a bare metal panel (the interior film dominates), ~0.5 with trim.
    inner_resistance_m2k_w: float = 0.13

    def __post_init__(self) -> None:
        if self.area_m2 <= 0.0:
            raise ValueError("panel area must be positive")
        if self.inner_resistance_m2k_w <= 0.0:
            raise ValueError("inner resistance must be positive")


@dataclass(frozen=True)
class CabinState:
    cabin_k: float
    panels_k: NDArray[np.float64]


class CabinNode:
    """The cabin air, coupled to its panels, stepped together.

    ``glazing_area_m2`` × ``glazing_transmittance`` is the solar power that reaches the *interior*
    and heats the air; the glass's own absorbed share belongs to the glass panel's balance and is
    not counted twice.

    ⚠️ **The glazing must also appear in ``panels``**, and ``glazing_panel`` has to name it. Glass
    is both the cabin's solar inlet and one of its largest conduction paths -- single glazing is
    about 5.8 W m⁻² K⁻¹, which for 2.6 m² is 15 W K⁻¹ against 2 W K⁻¹ of infiltration. A cabin
    given the inlet and not the path reaches **102 °C** at noon instead of 80 °C: not obviously
    wrong, just wrong, which is why this is a constructor error rather than a note.
    """

    def __init__(
        self,
        panels: list[CabinPanel],
        volume_m3: float = 3.0,
        glazing_area_m2: float = 2.6,
        glazing_transmittance: float = 0.55,
        air_changes_per_hour: float = 2.0,
        interior_mass_j_k: float = 25_000.0,
        glazing_panel: str | None = None,
    ) -> None:
        if not panels:
            raise ValueError("a cabin needs at least one bounding panel")
        if volume_m3 <= 0.0 or glazing_area_m2 <= 0.0:
            raise ValueError("cabin volume and glazing area must be positive")
        if not 0.0 <= glazing_transmittance <= 1.0:
            raise ValueError("glazing transmittance must lie in [0, 1]")
        if air_changes_per_hour < 0.0:
            raise ValueError("air changes per hour cannot be negative")
        names = [p.name for p in panels]
        if glazing_panel is None or glazing_panel not in names:
            raise ValueError(
                f"glazing_panel must name one of {names}: the glass is both the cabin's solar "
                "inlet and one of its largest conduction paths, and a cabin given the inlet "
                "without the path runs ~22 K too hot (ADR 0038)"
            )
        self.glazing_panel = glazing_panel
        self.panels = panels
        self.volume_m3 = volume_m3
        self.glazing_area_m2 = glazing_area_m2
        self.glazing_transmittance = glazing_transmittance
        self.air_changes_per_hour = air_changes_per_hour
        # The air alone has almost no heat capacity -- 3 m³ is 3.6 kJ/K, which a sunbeam moves in
        # seconds. The trim, seats and dashboard are what actually store the cabin's heat, and
        # leaving them out gives a cabin that tracks the sun instantly and cools instantly too.
        self.interior_mass_j_k = interior_mass_j_k

    @property
    def capacity_j_k(self) -> float:
        return self.volume_m3 * AIR_DENSITY_KG_M3 * AIR_SPECIFIC_HEAT_J_KGK + self.interior_mass_j_k

    @property
    def infiltration_w_k(self) -> float:
        """ṁ c_p for the configured air-change rate, W K⁻¹."""
        mass_flow = self.air_changes_per_hour * self.volume_m3 * AIR_DENSITY_KG_M3 / 3600.0
        return mass_flow * AIR_SPECIFIC_HEAT_J_KGK

    def derivative(
        self, state: CabinState, forcing: SurfaceForcing
    ) -> tuple[float, NDArray[np.float64]]:
        """(dT_cab/dt, dT_panel/dt) for the coupled system."""
        panels = np.asarray(state.panels_k, dtype=np.float64)
        if panels.shape != (len(self.panels),):
            raise ValueError(f"expected {len(self.panels)} panel temperatures, got {panels.shape}")
        inward = np.array(
            [
                panel.area_m2 * (t - state.cabin_k) / panel.inner_resistance_m2k_w
                for panel, t in zip(self.panels, panels, strict=True)
            ]
        )
        solar_gain = self.glazing_transmittance * self.glazing_area_m2 * forcing.q_solar_w_m2
        infiltration = self.infiltration_w_k * (forcing.t_air_k - state.cabin_k)
        d_cabin = (solar_gain + float(inward.sum()) + infiltration) / self.capacity_j_k
        d_panels = np.array(
            [
                (float(net_flux(t, panel.properties, forcing)) - flux / panel.area_m2)
                / panel.properties.heat_capacity_j_m2_k
                for panel, t, flux in zip(self.panels, panels, inward, strict=True)
            ]
        )
        return d_cabin, d_panels

    def advance(self, state: CabinState, forcing: SurfaceForcing, dt_s: float) -> CabinState:
        """One midpoint step of the **coupled** system: cabin and panels move together."""
        if dt_s <= 0.0:
            raise ValueError("dt_s must be positive")
        d_cabin, d_panels = self.derivative(state, forcing)
        half = CabinState(
            state.cabin_k + 0.5 * dt_s * d_cabin, state.panels_k + 0.5 * dt_s * d_panels
        )
        d_cabin, d_panels = self.derivative(half, forcing)
        return CabinState(
            state.cabin_k + dt_s * d_cabin, np.asarray(state.panels_k + dt_s * d_panels)
        )

    def equilibrium(self, forcing: SurfaceForcing, guess_k: float | None = None) -> CabinState:
        """The coupled steady state, by relaxation on the residual of both equations.

        Found by stepping rather than by a root finder over the whole vector: the panel equations
        are the same monotone functions M6.7 bisects, and the cabin's is linear in T_cab given the
        panels, so successive substitution converges without needing a Jacobian.
        """
        base = forcing.t_air_k if guess_k is None else guess_k
        state = CabinState(base, np.full(len(self.panels), base))
        dt = 0.2 * min(
            p.properties.heat_capacity_j_m2_k * p.inner_resistance_m2k_w for p in self.panels
        )
        for _ in range(400_000):
            nxt = self.advance(state, forcing, dt)
            moved = max(
                abs(nxt.cabin_k - state.cabin_k),
                float(np.max(np.abs(nxt.panels_k - state.panels_k))),
            )
            state = nxt
            if moved < 1e-10:
                break
        return state

    def relaxation_time_constant_s(self, state: CabinState, forcing: SurfaceForcing) -> float:
        """C_cab / (Σ A/R + ṁ c_p): how fast the cabin forgets, with the panels held fixed.

        This is the lumped constant **with the panels pinned**, which is what the formula assumes.
        Let them move and the coupled system relaxes about 33 % slower -- a cooling cabin drags
        its panels down and they feed heat back -- so 9 minutes here is 12 in a scene. Either way
        it is minutes, not hours, which is why a car that has been standing an hour is not
        obviously warm and one that has been standing five minutes is.
        """
        conductance = sum(p.area_m2 / p.inner_resistance_m2k_w for p in self.panels)
        total = conductance + self.infiltration_w_k
        if total <= 0.0:
            raise ValueError("a cabin with no coupling has no time constant")
        return float(self.capacity_j_k / total)


def cabin_coupling(
    cabin: CabinNode,
    ambient_at: Callable[[float], float],
    glazing_solar_at: Callable[[float], float],
    initial_k: float,
    *,
    name: str = "cabin",
) -> tuple[LumpedMember, list[LumpedLink]]:
    """The cabin as a lumped member of its panels' coupled solve, plus the links to them.

    ``glazing_solar_at`` is the solar irradiance *incident* on the glazing (W m⁻²) -- the
    scene's own ``q_solar`` on that surface's tilt -- of which ``τ_glz A_glz`` reaches the
    interior. The member's forcing is `CabinNode`'s balance term for term: infiltration as a
    conductance ``ṁ c_p`` to the air, the transmitted sun as an imposed power, no radiation and
    no absorption of its own; the panels' inward conduction is the links, one per panel at
    ``1/R_p`` over the panel's cells. The link names are the panels' names, which must be the
    member names in the `CoupledFields` this is handed to.
    """

    def forcing(t_s: float) -> FacetForcing:
        return FacetForcing(
            t_air_k=float(ambient_at(t_s)),
            h_w_m2_k=cabin.infiltration_w_k,
            q_internal_w_m2=cabin.glazing_transmittance
            * cabin.glazing_area_m2
            * float(glazing_solar_at(t_s)),
        )

    member = LumpedMember(name, cabin.capacity_j_k, forcing, float(initial_k))
    links = [LumpedLink(p.name, name, 1.0 / p.inner_resistance_m2k_w) for p in cabin.panels]
    return member, links


def cabin_field(
    cabin: CabinNode,
    panels: Sequence[FieldMember],
    ambient_at: Callable[[float], float],
    glazing_solar_at: Callable[[float], float],
    t0_s: float,
    tick_s: float,
    *,
    name: str = "cabin",
    initial_k: float | None = None,
    keep_ticks: int | None = DEFAULT_KEEP_TICKS,
    spin_up_hours: float | None = None,
    spin_up_hash: str = "",
    wrap: Callable[[Any], Any] | None = None,
) -> CoupledFields:
    """The panels and the cabin behind them as **one** coupled solve (PT.15, ADR 0106).

    ``panels`` are the fields that bound the cabin, in the order `cabin.panels` names them;
    each keeps its own cells, forcing, shadow and lateral operator. The cabin joins them as a
    lumped member, so the whole thing is stepped by one implicit operator -- panels against
    this tick's cabin, cabin against this tick's panels, which is what ADR 0038 required and
    what an alternating step gets wrong at a 60 s tick.

    With ``spin_up_hours`` the coupled system is integrated through the hours before ``t0_s``
    (through ``wrap``, the scene's weather wrap), so the cabin starts the scene carrying the
    days before rather than at the air temperature.
    """
    names = [p.name for p in cabin.panels]
    if [m.name for m in panels] != names:
        raise ValueError(
            f"the coupled panels {[m.name for m in panels]} must be the cabin's own {names}, "
            "in order: the links are built from the cabin's panel list"
        )
    start = float(ambient_at(t0_s)) if initial_k is None else float(initial_k)
    member, links = cabin_coupling(cabin, ambient_at, glazing_solar_at, start, name=name)
    coupled = CoupledFields(
        panels,
        t0_s=t0_s,
        tick_s=tick_s,
        keep_ticks=keep_ticks,
        lumped=[member],
        lumped_links=links,
    )
    if spin_up_hours is None:
        return coupled
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
    rebuilt = []
    at = 0
    for m in panels:
        rebuilt.append(
            FieldMember(
                m.name,
                m.patch,
                m.properties,
                m.forcing_at,
                spun[at : at + m.patch.n_cells],
                m.conduction,
            )
        )
        at += m.patch.n_cells
    member, links = cabin_coupling(cabin, ambient_at, glazing_solar_at, float(spun[at]), name=name)
    return CoupledFields(
        rebuilt,
        t0_s=t0_s,
        tick_s=tick_s,
        keep_ticks=keep_ticks,
        lumped=[member],
        lumped_links=links,
    )
