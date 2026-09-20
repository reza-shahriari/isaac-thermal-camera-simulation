# ADR 0099 — Contactors by overlap area, and coupled fields as one solver

**Status:** Accepted
**Date:** 2026-09-20
Roadmap: TC.3 (§6.1, §6.4; spec issues S41, S42)

## Context

TC.1 gave one field a conduction term and TC.2 gave lumped parts a network. A vehicle is both:
a bonnet skin (a field) bolted, clipped and gasketed to the body (a field or a node) under it,
an underbody (a node's face) facing the road (a field). Two questions had to be settled before
TC.5's engine could warm the metal around it through a scene config.

*How is a joint between two grids authored?* The literature reports a contact conductance
`h_c` per unit area (Voller & Tirovic 2007, ADR 0097's table). Two patches rarely share a cell
size, an origin or an orientation — the car scenes' bonnet is 71 mm cells over a road of 300 mm
cells — and the survey's warning is explicit: a conductance authored per *node* scales with the
cell count, so refining one grid changes the joint.

*How are two coupled fields stepped?* A 5 mm skin cell on a 10⁴ W m⁻² K⁻¹ joint has a time
constant under a second; the scene tick is 60 s. An exchange stepped explicitly between two
separate fields blows up; a lagged exchange (each field seeing the other's last tick) rings at
the tick frequency, which on a rendered panel is a checkerboard that looks like fixed-pattern
noise (the Crank–Nicolson argument of ADR 0094, from the other side).

## Options considered

1. **Per-node links, nearest cell to nearest cell.** Simple, and wrong under refinement: a
   plate half overhanging its base is credited with its whole area, and quartering its cells
   leaves that unchanged (TC.3's negative control).
2. **Require aligned grids** (same cell size, same origin) so cells pair one to one. It makes
   every scene's road grid a function of its bonnet grid, and a rotated part impossible.
3. **Conductance by overlap area:** `K_ij = h_c · A_ij` with `A_ij` from clipping cell *j*'s
   rectangle against cell *i*'s in the first patch's plane (Sutherland–Hodgman on a convex
   quadrilateral, candidates prefiltered by the regular grid). Exact to clipping precision for
   any offset, cell size or in-plane rotation; the total is `h_c · A_overlap` and is invariant
   to refinement by construction. Chosen.
4. **Two fields exchanging heat as a forcing term** (explicit or lagged). Rejected above.
5. **One solver over the union of cells.** The members' cells are concatenated into one
   `ThermalField` with a block `ConductionOperator` — each member's own lateral operator on the
   diagonal, the contactors off it — and stepped by ADR 0094's IMEX scheme, which already
   handles any conductance implicitly. Each member becomes a `PatchView` over its slice. Chosen.

## Decision

Options 3 and 5, shipped as `irsim.thermal.coupling` (TC.3):

* `cell_overlap_areas(patch_a, patch_b, gap_tolerance_m)`: a sparse `(n_a, n_b)` matrix of
  shared areas; the patches must share a frame, be parallel, and lie within the tolerance along
  the normal (a bonnet 0.9 m above a road is not a joint and the call refuses).
  `contactor_conductances` multiplies by `h_c`.
* `CoupledFields(members, contactors, t0_s, tick_s)`: one `ThermalField`, one forcing (the
  members' forcings concatenated cell for cell), one block operator. `fields[name]` is a
  `PatchView` with `patch`, `advance_to`, `temperature_at`, `sample_at` — the interface the
  point bridge consumes — and advancing any view advances the whole system. With no contactor
  the members are the separate fields **bit for bit**.
* `RadiationExchange(rect, patch, surface_emissivity)`: ADR 0088's parallel-rectangle view
  factors read in both directions. The cells' term is `occluded_longwave_flux` (the body also
  blocks the sky); the body's loss to the patch is `ε_r σ T_r⁴ Σ_i A_i F_i`, which by
  reciprocity is `A_r F_r→patch ε_r σ T_r⁴`. `reverse_view_factor` evaluates `F_r→patch` by an
  independent quadrature from the body's side so reciprocity is checked between two
  discretisations, not one formula against itself.

## Consequences

**Easy now:** TC.5 charges an engine node for what its bay face radiates onto the bonnet and
what its underbody radiates onto the road, with the same object serving both; TC.6 declares a
bonnet-to-body contactor in the scene and the bonnet stops being a skin floating in air. PT.11's
lateral Laplacian drops onto a member's diagonal block unchanged.

**What it costs:** a coupled system is one linear solve per tick over every member's cells,
factorised once per tick size (ADR 0094); a road of 10 812 cells coupled to a bonnet of 437 is
still a sparse system with a few thousand off-diagonal entries. Clipping is pure Python over
candidate pairs: the car's road–bonnet overlap takes milliseconds.

**Error introduced:** none in the overlap (exact polygon clipping); the step's error is
ADR 0094's first-order-in-dt on the slow modes. Radiation between a body and a field remains
single-bounce, isothermal over the rectangle and parallel-only (ADR 0088's limits); the reverse
quadrature agrees with reciprocity to 1 % on a 60 × 60 road under a 1.64 × 3.8 m underbody, the
residual being the finite-cell discretisation of a differential-element formula on both sides.

**Measured:** a 1 m² plate on a 2 m base gives `h_c · 1.0` to 1e-9 at 0.25 m cells and the same
at 0.0625 m; a 30°-rotated square overlaps exactly its own area; a half-overhanging plate is
credited with 0.5 m² where a per-node total says 1.0 at either refinement; a 350 K plate on a
300 K base at h_c = 10⁴ (τ = 0.8 s) equalises under 60 s ticks with the stored energy conserved
and the joules lost by the plate equal to those gained by the cells it touches; the power an
underbody radiates onto the road equals the power the cells receive to 1e-6.

## Revisit when

A joint between non-parallel surfaces is needed (a door skin on a sill), which is a line contact
rather than an area and wants a different primitive; or the coupled system grows past what one
`splu` per tick size is comfortable with, at which point the block structure is what a Schur
complement per member would exploit.
