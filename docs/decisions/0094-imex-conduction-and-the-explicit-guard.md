# ADR 0094 — Conduction between facets: an IMEX step, and the explicit bound finally enforced

**Status:** Accepted
**Date:** 2026-09-20
Roadmap: TC.1 (§6.1, §6.4)

## Context

Every facet `FacetSolver` has ever stepped was an independent column. That is the root of two
things the owner asked for on 2026-09-18 and the plan could not deliver: heat does not spread
across one panel (a bonnet renders sharper than aluminium can be — PT.11's diffusion lengths are
270 mm for aluminium and 99 mm for steel over an engine's 750 s rise, against 71–150 mm cells),
and heat does not move between parts at all (an engine cannot warm the bracket bolted to it — the
`TC` lane). Both need a term that moves heat between facets, and both are blocked on the same
integrator problem: conduction is stiff. ADR 0036 measured the two-node case at a factor of 400
past the scene tick; a 5 cm aluminium cell is ~9 s against a 60 s tick; a bolted joint at
10⁴ W m⁻² K⁻¹ is a tenth of a second.

There was also a hole beside it: `FacetSolver` enforced no stability bound at all. `tick_s` up to
3600 s parsed, and a leaf at 630 J m⁻² K⁻¹ under 30 W m⁻² K⁻¹ of wind — bound ≈ 29 s — would have
diverged quietly during a 60 s spin-up. Nothing in a shipped scene tripped it; the next material
would have.

## Options considered

1. **Explicit conduction, smaller tick.** Two orders of magnitude more steps for a transient
   nobody images. The fixed-tick contract in `field.py` exists so the render rate does not set the
   physics; a conduction term that sets the tick is the same failure from the other side.
2. **Fully implicit, including the T⁴ radiation term.** Newton iterations every tick, and it
   changes the surface balance's integrator — ADR 0036 chose the midpoint rule over Euler on
   *bias* (explicit Euler inflates the diurnal swing that §6.3's crossover test measures), and an
   implicit scheme damps the same quantity. Nothing about the surface balance needs changing.
3. **Crank–Nicolson on the conduction term.** Second order, but only A-stable: a mode with
   dt/τ = 600 is multiplied by (1 − 300)/(1 + 300) ≈ −0.993 per tick. It does not blow up; it
   rings, for hours, at the tick frequency, with the sign flipping each step. On a rendered panel
   that is a checkerboard that looks like fixed-pattern noise.
4. **IMEX: the surface balance explicit on the midpoint rule, conduction backward Euler.**
   L-stable on the conduction term (the same mode is damped by 1/601 per tick and is gone after
   a few), the surface balance untouched, one linear solve per tick with a matrix that depends only
   on the tick size.

## Decision

Option 4.

* `irsim.thermal.conduction.ConductionOperator(conductance_w_k, area_m2)`: a symmetric sparse
  matrix of **link** conductances in W/K, zero diagonal, non-negative, plus the facet areas. The
  balance gains `Σ_j K_ij (T_j − T_i)` per facet, divided by `C_i A_i`. Links in W/K rather than
  W m⁻² K⁻¹ because that is what a joint (`h_c·A`), a contactor or a grid edge (`k δ w / d`) is;
  the areas make it conservative for **unequal** cells, and symmetry is checked at construction
  because an asymmetric matrix invents energy quietly.
* `FacetSolver(..., conduction=)` steps `T* = Tⁿ + ½ dt F(Tⁿ)/C`, `rhs = Tⁿ + dt F(T*)/C`, then
  solves `(I + dt D⁻¹ L) Tⁿ⁺¹ = rhs` with `L = diag(K·1) − K`, `D = diag(C A)`. The matrix is
  factorised (`scipy.sparse.linalg.splu`) once per tick size and back-substituted each step.
  Without an operator the step is the midpoint rule it always was, **bit for bit** — tested with a
  zero operator as well as with none.
* The explicit bound `2C/(h + 4εσT³)` is checked on every step against the forcing's *actual*
  `h` and the state's own `T`, and raises naming the facet and the numbers. `guard=False` opts out
  for a caller that knows why (a steady-state search). Conduction never trips it: it is implicit.
* `ThermalField` and `PlanarThermalField` take `conduction=` and pass it through unchanged.

## Consequences

**Easy:** PT.11's lateral Laplacian is a builder that returns an operator; TC.2's network is
capacities plus links; TC.3's contactors are links between two fields' cells. None of them touches
the integrator again. A 60 s tick stands over any conductance.

**What it costs:** first-order accuracy on the conduction term's *slow* modes. Measured: two cells
with τ = 1 h at dt = 60 s land 0.7 % of the initial 10 K contrast from the closed form after two
hours, and halving the tick halves the error. For a thermal camera that is far inside what the
forcing's own uncertainty (§6.2's h) contributes, and it is measured, not assumed.

**Error introduced:** the splitting error of IMEX — the explicit half sees `Tⁿ`, not `Tⁿ⁺¹`, on
the conduction-coupled part. It is the same O(dt) term as above and is bounded by the same test.

## Revisit when

A scene needs a temperature-dependent conductance (a phase change, TC's exhaust line where k(T)
matters), at which point the matrix changes per tick and the factorisation is per tick too — or a
second-order-in-time conduction answer is needed for a validation, which would mean TR-BDF2 rather
than Crank–Nicolson, for the L-stability reason above.
