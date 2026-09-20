# ADR 0102 — Lateral conduction between a patch's cells, from the material's own k and δ

**Status:** Accepted
**Date:** 2026-09-20
Roadmap: PT.11 (§6.1, §6.4; spec issue S41)

## Context

A `PlanarThermalField`'s cells were independent §6.1 columns: heat spread across a panel only
insofar as the forcing spread. Over §6.6's 750 s engine-bay rise a signal diffuses
`√(k δ t / C)`: 270 mm in aluminium, 99 mm in steel, 16 mm in asphalt, against the 71–150 mm
cells the scenes use — so a metal panel rendered sharper than aluminium can be (spec issue
S41, and the owner's first requirement read the other way: one object, its parts *connected*).
TC.1 supplied the term and the implicit step (ADR 0094: `tick_s: 60` is 6.6× past the explicit
limit at 5 cm in aluminium); what remained was the operator and where its numbers come from.

## Options considered

1. **A per-cell Laplacian with an authored diffusivity per surface.** One more number a scene
   author types, beside the material that already carries `conductivity_w_mk` and `thickness_m`
   — the single-source rule ADR 0043 stated for emissivity, broken again.
2. **ADI or an explicit sub-tick.** ADI is exact for the rectangular grid but a second
   integrator; an explicit sub-tick at 6 s inside a 60 s tick is ten solves per tick of a
   term whose transient nobody images. ADR 0094 already made this choice.
3. **A `ConductionOperator` builder from the material.** `K = k δ · (shared side) / (gap)` per
   four-neighbour edge, in W/K — Fourier across a slab of thickness δ, the `k δ w / d` link
   `irsim.thermal.conduction`'s docstring names — built once per patch from the library
   material's `conductivity_w_mk` and `thickness_m`, and stepped by the IMEX scheme that
   already exists. `k δ = 0` returns `None`, the operator-free field bit for bit. Chosen.

## Decision

Option 3, `irsim.thermal.conduction.lateral_operator(patch, k, δ)`.

* Every patched surface in a scene gets the operator from its own material unless it says
  `lateral_conduction: false` (the per-material switch: it is the material that decides how
  much it matters, and the flag is for a comparison, not a default). The per-cell spin-up
  carries the operator too, so the starting state is the coupled one.
* The car demo's bonnet takes `conductivity_w_mk` and `thickness_m` (steel under paint,
  45 W/mK and 1.2 mm, the `car_paint_black` substrate) until PT.6 reads them from the material.
* Diagonal edges are not linked: a five-point Laplacian is second-order on a rectangular grid
  and the nine-point stencil buys nothing a camera resolves.

## Consequences

**Measured:** a 20 K step on a 2 m strip of 5 mm aluminium cells spreads as the semi-infinite
sheet's `erf(x / 2√(αt))` to within 0.6 % of the step at 60 s (1 s ticks; a test of the
operator, not of the tick); the operator is `k δ dv / du` along u and `k δ du / dv` along v,
exactly; `k → 0` is the operator-free field bit for bit; at 5 cm in aluminium the explicit
limit is `du² C / (4 k δ)` = 6.4 s and a 60 s implicit tick obeys the maximum principle and
conserves the mean where forward Euler on the same operator explodes; the scene builds asphalt's
operator at `k δ` = 0.0375 W/K per edge from the library and the switch removes it; the steel
bonnet's engine hot spot is 5 % smoother than the independent-column field with the same mean —
one 70 mm cell of spreading, which is what 99 mm over 750 s buys.

**What it costs:** one sparse factorisation per tick size per patched surface (a 10 812-cell
road: well under a second) and a back-substitution per tick and per spin-up step; the road's
spin-up grows by a few seconds. The bit-identity tests of PT.17 and PT.18 hold only without the
operator, which is what they now declare: with it, a lit cell beside a shaded one exchanges
heat, millikelvins for asphalt, and that is the physics.

**Error introduced:** ADR 0094's first-order-in-dt on the slow modes; nothing from the operator
itself on a rectangular grid. A patch in a moving frame conducts exactly as a fixed one — the
operator lives in the patch's own (u, v).

## Revisit when

A mesh field (WM.2) needs an intrinsic-Delaunay-safe cotan Laplacian (WM.6); or a validation
needs second-order time accuracy on the conduction transient (TR-BDF2, ADR 0094).
