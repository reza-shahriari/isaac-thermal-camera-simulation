# ADR 0103 — An N-layer stack through every cell, as members of one coupled solve

**Status:** Accepted
**Date:** 2026-09-20
Roadmap: PT.12 (§6.4)

## Context

A field's cell was one node: the whole slab at one temperature with an adiabatic back. §6.4
gives the two-node model for exactly the case that breaks this — a fast surface on a slow
substrate — and ADR 0036 built it (`two_node.py`), but no field used it; every patched surface
in every scene was a single lump, so 0.3 m of asphalt held its surface up all night on the
whole slab's heat. Production tools do not lump: MuSES evaluates properties per thermal node
through the element thickness, Fraunhofer's models stored 2 + N temperatures per triangle and
used ten layers.

## Options considered

1. **Extend `LumpedTwoNodeSolver` to N nodes per cell and give fields a second solver.** A
   second integrator beside `FacetSolver`, with its own tick contract, guard and history ring,
   and an explicit step whose bound §6.4 evaluated wrongly by a factor of 400 (spec issue S39).
2. **A per-cell semi-analytic substrate** (a response function for the diurnal wave). Exact for
   a homogeneous half-space under a sinusoid, and nothing else: no layers, no deep boundary, no
   coupling to the parts a network hangs off the surface.
3. **Layers as members of `CoupledFields`.** Each layer is a member with the same grid: layer 0
   carries the surface balance and the material's ε and α, the layers below ε = α = 0 and no
   convection (the bottom one a linear exchange with a deep node when one is declared), and
   consecutive layers are joined by a contactor whose `h_c = 1/R_{i,i+1}` with
   `R = δ_i/(2k_i) + δ_{i+1}/(2k_{i+1})` — §6.4's centre-to-centre resistance, reused from
   `two_node.contact_resistance`. One implicit solve per tick (ADR 0094), the surface layer a
   `PatchView` the bridge binds, the layers below bookkeeping that never reaches a pixel.
   Chosen.

## Decision

Option 3, `irsim.thermal.layers.LayerStack` and `layered_field`, and `layers: N` on a patched
surface in the scene schema (default 1: the single node the field always was; N cuts the
material's thickness into N equal slices with an adiabatic back). Lateral conduction runs in
every layer with that layer's own `k δ`; the whole stack is spun up on its own forcing and
operator so the base carries the days before, which is the point of having one. A film on a
layered surface is refused for now (the film lives on the single-node solver).

## Consequences

**Measured:** a two-layer stack with a deep boundary reproduces `LumpedTwoNodeSolver` to under
1 mK on both nodes over 6 h of diurnal forcing at a 2 s tick (1.7 mK at 10 s: the IMEX side is
first order in dt); the contact conductances are `k/δ` for equal slices exactly; a 1 mm steel
skin peaks with the beam and a six-layer 0.3 m asphalt surface more than an hour later; the
single-node 0.3 m asphalt is warmer than the six-layer one by more than 2 K at 04:00, with the
stack's base warmer than its surface at night; `layers: 6` on a scene's road builds a six-member
solve whose surface view answers the bridge's interface, and `layers: 1` is the field it always
was.

**What it costs:** N times the cells in one sparse implicit system, and N times the spin-up
work; the two-node explicit bound is gone (the vertical link is implicit), so any layer thickness
is admissible and the accuracy question is the tick's alone (first order, ADR 0094).

**Error introduced:** the stack's first-order-in-dt on the vertical transient — bounded by the
two-node comparison at the tick a test chooses, and by the roadmap's own night-curve
criterion for the shipped 60 s tick. Layers of one material are equal slices; a real pavement's
graded base would want `LayerStack(layers=(...))` with its own thicknesses, which the API takes
and the schema does not yet spell.

## Revisit when

A scene needs a non-uniform stack from the config (a wearing course over a base over soil), or
a film on a layered surface (PH.2 on a layered road), or a deep boundary from the weather (the
soil temperature a day down), each a schema addition on this module.
