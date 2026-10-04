# ADR 0187 — A motor is bolted to its arm through the implicit step, and the arms are tubes

**Status:** Accepted
**Date:** 2026-10-04
Roadmap: EV.16 (the clear-sky exit clip), PT.9's remainder ("conducting their heat into the skin
from a config still has no route")

## Context

The owner read the first cruise render of `aerial_clear_exit.yaml` cell by cell and found three
things no real quadrotor shows.

1. **Two arms 13 K colder than the other two.** The outbound scene solves `arm_n` and `arm_e` as
   fields and leaves `arm_s` and `arm_w` on the per-prim `airframe` node, which is the air
   temperature. That was a deliberate control for the PT.9 time-lapse. On a clip whose whole job is
   to look like a drone, it reads as a defect.
2. **An arm coldest under its motor.** The motor is the hottest part of the aircraft at cruise
   (46–48 °C, ADR 0072's throttle law), and the arm under it sat at 28 °C, in the motor's own
   shadow, because nothing carried the motor's heat anywhere. The motor law gives a temperature
   with no route into the skin.
3. **Arms 9 K too warm where the camera looks.** Each arm was a flat strip solved as a sunlit top
   face (40.4 °C at cruise). The ground camera sees the arm's underside, and ADR 0087's planar
   bridge painted the top's temperature onto it. The engine-free tube oracle
   (`quad_flight_mesh.yaml`) puts the same arm's crown at 34 °C and its underside at 31 °C at the
   same instant. A tube catches the beam over its projected width, a third of its girth, and loses
   heat around all of it.

## Options considered

**For the motor's heat:**

1. **Fold the joint into the convective term**, as `h (T_air − T) + g (T_m − T) = (h + g)(T_eff − T)`.
   This is exact for the linear terms, but it is wrong twice over. The free-convection rule (PT.25)
   takes `max(h, c |T − T_air|^⅓)` against the *air*. And the convective term is stepped
   explicitly, under §6.4's bound `2C/h`: a 1000 W m⁻² K⁻¹ joint on carbon's 2520 J m⁻² K⁻¹ allows
   5 s, against a 10 s tick and a 60 s spin-up step. Rejected.
2. **A lumped motor node in a `CoupledFields`** (the cabin's route). Implicit, and it would solve
   the motor rather than prescribe it. But it gives the motor a second temperature beside the
   `heat_source` target the renderer paints the bell with, and the two would disagree. Rejected
   until the motor is solved from watts (the TC.11 asset path already does that for an asset).
3. **A prescribed-temperature boundary in the implicit conduction step** (chosen). The cells
   under the footprint gain `g_i (T_m − T_i)`, with `g_i` from
   `coupling.footprint_conductances` (nearest cells first, `h_c · area` in total, the same rule the
   asset path uses for a hidden part). This rides ADR 0094's backward Euler:
   `(I + dt D⁻¹ (L + diag g)) Tⁿ⁺¹ = rhs + dt D⁻¹ g T_b`. It is one-way, which is right for a
   prescribed temperature, as TC.6's `follows:` nodes are.

**For the arms:** solve all four, as `mesh:` tubes (WM.7 and ADR 0110's 16 × 36 cut, ADR 0112),
bound to cylinder prims of the same radius and length (`quad_outbound.TUBE_ARM_QUAD`). The tube
geometry was already measured and shipped engine-free; `render_phantom4.py` proved the Isaac
mesh bridge.

## Decision

* `ConductionOperator.boundary_w_k` (per facet, W/K) and `FacetForcing.boundary_k` (the temperature
  behind it). `FacetSolver.advance` adds `boundary_rhs` inside the implicit solve. An operator with a
  boundary refuses a forcing without `boundary_k` rather than reading 0 K, and a zero boundary is no
  boundary, bit for bit.
* `irsim.thermal.mounts`: `Mount`, `mount_conductances`, `MountedForcing`. Outside the motor's
  schedule the boundary is the cell's air: the motor is off before take-off and after landing.
  `MountedForcing` wraps **outside** the weather wrap and reads the true instant, so the spin-up
  does not run yesterday's flight at the same time of day.
* Schema v21: `thermal.mounts: [{target, surface, centre_m, area_m2, joint | h_c_w_m2_k}]`. The
  target must be a `heat_source` and the surface a plain patch or mesh. A layered surface, a
  temperature map, a cabin panel and `object_exchange: true` are refused, because those solves
  rebuild the operator without the boundary.
* `aerial_clear_exit.yaml`: four tube arms, the south and west pod occluders, and four mounts at
  20 cm² on `dry_default` (1000 W m⁻² K⁻¹, ESTIMATED; the Inspire 3's mounts in the asset library
  use the same joint at 19 cm²).

## Consequences

**Measured at cruise** (mission 900 s, air 26.5 °C, motor 46.1 °C), engine-free:

| | flat strips, before | tubes with mounts, after |
|---|---|---|
| arms that are solved | 2 of 4 (the others at 26.5 °C) | 4 of 4, within 1 K of each other |
| crown: open span → under the motor | 40.4 → 28–30 °C (shadow) | 34–37 → 43.6–43.9 °C |
| underside (what the camera sees): open span → under the motor | painted 40.4 °C | 31–32 → 34.6–34.9 °C |

`tests/unit/test_motor_mounts.py` checks the following. A bolted cell relaxes at exactly
`g/(C A)`. One step balances to 1e-9, and a 238 τ step lands on `T_b` from below. The footprint
totals `h_c · area` on the tube, under the motor. The motor is off outside its mission. With the
mount, the arm is warmest under its motor on crown and underside alike, and never above the motor.
Without it, the same cells are colder than the open span (the control).

**Not modelled.** The motor is not cooled by the arm it heats (one-way), and it takes no sunlight:
ADR 0072's law is throttle only. The clamp is a contact conductance, not a part with mass. The
outbound and reference scenes keep their flat strips and their control arms; they are the PT.9
measurement and EV.14's reference, and moving them would move numbers those steps recorded.

## Revisit when

* The motor is solved from its losses (watts) instead of prescribed. The mount should then become a
  two-way link to a lumped node, which `CoupledFields` already supports.
* A mount is wanted on a layered surface or inside the object exchange.
