# ADR 0096 — A lumped thermal network: backward Euler, fixed nodes eliminated, radiation linearised per tick

**Status:** Accepted
**Date:** 2026-09-20
Roadmap: TC.2 (§6.4, §6.6; spec issue S42)

## Context

Nothing in the package could connect two solved parts. `FacetSolver` steps independent surface
columns; `TwoNodeSolver` is one surface on one substrate; §6.6's vehicle sources are schedules.
An engine warmed the bonnet by radiation onto a parallel panel (ADR 0088) and never the mounts,
brackets or wings it is bolted to — the owner's second requirement on 2026-09-18. TC.1 supplied
the term that moves heat between facets and the implicit step that survives it (ADR 0094); what
was missing is the object a scene author and TC.4's schema can name: parts with mass, the joints
between them, and the boundaries they see.

The survey's anchors set the shape. Production under-hood models (MuSES / TAITherm) are
networks of lumped nodes with three boundary kinds — fixed temperature, imposed heat in watts,
convection to a fluid node — and links authored either as a contact conductance `h_c·A`
(Voller & Tirovic 2007: 7–67 kW m⁻² K⁻¹ for bolted ferrous joints) or as a total `G` in W/K per
fastener. A rubber mount lags with mass of its own. A block radiates to the wing above it.

## Options considered

1. **Extend `FacetSolver` with per-facet capacities in J/K and watts.** Its balance is the
   areal §6.1 surface balance with h, q_solar and sky terms; a bracket inside a bay has none of
   those, and overloading the facet with "sometimes areal, sometimes lumped" units is how a W/K
   ends up divided by an area twice.
2. **Explicit stepping of the network at its own sub-tick.** A 50 g bracket on a 25 W/K bolt has
   τ ≈ 1 s; a scene tick is 60 s. The fixed-tick contract exists so the render rate does not set
   the physics.
3. **A separate `ThermalNetwork` on `ConductionOperator`'s Laplacian, fully implicit.** Nodes in
   J/K, links in W/K, one backward Euler solve per tick, fixed nodes eliminated exactly.
4. **Newton iterations for the T⁴ radiation links.** Exact per tick, several solves per tick, and
   a Jacobian that TC.3's contactors would have to extend.

## Decision

Option 3, with radiation linearised rather than iterated.

* `irsim.thermal.network`: `Node(name, capacity_j_k)`, `FixedNode(name, T or T(t))`,
  `ImposedHeat(node, W or W(t))`, `Link(a, b, G or G(t))` with `Link.from_contact(h_c, A)` and
  `Link.convection(h, A)` as the same multiplication (bit-identical by construction),
  `LinkNode(name, a, b, G, C)` expanding to a node with `2G` on each side (series `G`, τ = C/4G),
  `RadiationLink(a, b, ε, A, F)`.
* The step: `(C/dt + L_ff) Tⁿ⁺¹_f = (C/dt) Tⁿ_f + Q(tⁿ⁺¹) − L_fb T_b(tⁿ⁺¹)`, with `L` from
  `ConductionOperator` (areas of 1, so its symmetry and sign checks apply to every matrix the
  network builds). Fixed nodes enter at their **end-of-tick** value, so a boundary that follows
  the weather is honoured at the tick's end rather than lagged by one step.
* A radiation link contributes `G = ε A F σ (T_a² + T_b²)(T_a + T_b)` evaluated at the
  start-of-tick temperatures — the exact conductance for the T⁴ flux at those temperatures. The
  matrix is therefore rebuilt every tick (tens of nodes: negligible). The scheme converges to the
  exact steady state, since there the linearised and true flux coincide, and is first order on
  the transient like the rest of the step.
* The network keeps the conductance matrix it last applied, so `link_power_w` and
  `energy_residual_w` report the flows **as the solver used them**. That is what makes the
  conservation test meaningful: `Σ C ΔT/dt = Σ Q − heat into fixed nodes` to solver precision
  every tick, and a sign error in any conductor fails it on the first tick.

## Consequences

**What follows from it:** TC.3's contactors are links between a field's cells and network nodes
(cell capacity = areal C × cell area); TC.4's `nodes:` / `links:` schema is a direct transcription
of these dataclasses; TC.5's engine is a node with imposed heat, a coolant fixed node and a bay-air
fluid node whose convection links carry a callable `h` for the forced-to-natural switch at key-off
— the hot soak already shows in this step's own bay test, where a bracket keeps warming after the
block's heat stops.

**What it costs:** first-order accuracy on slow modes, as ADR 0094; a dense solve per tick
(`numpy.linalg.solve`), fine for the tens of nodes a vehicle needs and to be revisited if a
network ever reaches thousands; radiation linearised at the tick's start, so a link whose
temperatures change by a large fraction within one tick lags by that fraction.

**Measured:** ΔT = Q/G and Q/(hA) to 1e-6; the three link forms bit-identical after 50 ticks; a
mount's τ fitted from the trajectory within 2 % of C/4G; the T⁴ steady state to 1e-6; energy
closed to 1e-6 of the imposed power on every tick of a seven-node bay with a moving ambient, a
switching h, a radiation link and a mount.

## Revisit when

A network needs hundreds of nodes (switch the dense solve to `splu`, re-factorising only when a
callable or a radiation link changes the matrix); or a validation needs second-order accuracy on
a transient (TR-BDF2, for the L-stability reason in ADR 0094); or a temperature-dependent
conductance (k(T) on an exhaust) arrives, which is the same per-tick rebuild this step already does.
