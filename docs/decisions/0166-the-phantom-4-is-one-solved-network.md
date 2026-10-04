# 0166 — The Phantom 4 is one solved network; ADR 0072's nodes and ADR 0143's offsets retire for it

Date: 2026-09-29

**Status:** Accepted (2026-09-29, TC.13). Supersedes, for the Phantom 4 demo, the prescribed
nodes of ADR 0072 and the airframe offsets of ADR 0143; both scenes remain in the catalogue.

## Context

ADR 0072 gave an aerial target's heat sources prescribed nodes, `T = T_air + ΔT_max · u²`, with
no time constant: a motor was 20 K over the air the instant the throttle said so and at the air
temperature the instant it stopped. ADR 0143 gave the Phantom 4's nineteen parts nineteen
temperatures on that model plus `T_air + offset` for the six airframe-family parts, and named the
fidelity it skipped: a solved shell, and heat that moves from a source into the parts around it.
TC.11 built the solve (`thermal.objects:`, `solve: full`: mesh members, lumped hidden parts,
AI.11 contacts, one implicit step) and could not be applied to the aircraft because one cell per
face made it 1.5 million cells (ADR 0165 removed that).

Putting the aircraft on the solve needed four decisions.

## Decisions

1. **The heat sources are hidden components inside the parts the camera sees.** Four
   `winding_*` hidden parts (`component: brushless_motor`) sit inside the four motor bells, a
   `pack` (`component: lipo_pack`) inside the battery case, beside the ESCs and the flight
   controller AI.11 already placed. Each takes mass, heat capacity and rated dissipation from the
   component library (AI.12). A winding meets its bell through the stator's press fit over the
   stator's circumference, ~10 cm² (ESTIMATED); the pack meets its case over 40 cm² (the case
   mesh is 57 cm² in all). Nothing dissipates on a visible part directly.
2. **Duty is the throttle cubed, scaled to the hover draw.** A component's dissipation is rated
   × duty. Electrical input to a rotor scales about as the cube of throttle (thrust ∝ u², power
   ∝ thrust^1.5) and the loss fraction is taken constant at the rating, so a station's duty is
   `0.694 u³` (ESTIMATED): the factor puts hover (u = 0.6) at 0.15 of the 300 W rating, the 45 W
   per motor the pack's 200 W hover draw implies, so a 60 W-rated motor dissipates 9 W at hover
   and 17 W in an orbit. ESCs follow their station, the pack the mean of the four, the flight
   controller runs at rated. `ObjectSpec.duties` carries a schedule per hidden part; before it,
   one duty served every hidden part and four motors on four throttle histories were one.
3. **The bells sit in their propeller's downwash.** `ObjectSpec.speeds` gives a visible part an
   airflow schedule (PT.9's `speed_s` / `speed_m_s`); each propeller, motor bell and mount takes
   its station's, on the perpart scene's scale. Without it a 9 W bell in still air ran to 100 °C.
4. **Every part is a coarse mesh member at 2 cm** (ADR 0165), 5,113 cells for the airframe,
   built in 15 s and flown through the 28-minute mission in under a second per hundred ticks.

## Consequences

* Measured on `phantom4_solved.yaml` (June noon, 26 °C air): the bells rise 1.4 K in the first
  minute of the climb and 14 K by its end -- a time constant, not a jump; they reach 42 °C at
  hover and 60-66 °C in the orbits; the rear pair leads the front by 10 K in nose-down cruise,
  the outer pair the inner by 8 K in each orbit, and the four are four temperatures; landed and
  idle they are still 43 °C ten minutes later and 37 °C at twenty, where ADR 0072's node was
  0.1 K over the air at once. The windings sit 7-15 K above their bells. The lower shell under
  the pack runs 2 K warmer than the rest of the shell, from inside. The same object, made
  adiabatic, stores exactly what its components dissipate (`tests/unit/test_phantom4_solved.py`).
* The spin-up is still per part (TC.11): the coupled solve starts from separately spun-up
  parts, and the first tick equalises the contacts -- the sunlit battery case drops 6 K into the
  pack and the shell at T+0. Known, and the next thing to fix on this object.
* `phantom4_perpart.yaml` stays as the prescribed baseline and its tests stand; the render
  driver maps a part to `<object>.<part>` when the scene solves an object, and to the old
  candidate list otherwise.
* Not measured here: the rendered frame (a run of `render_phantom4.py --scene phantom4_solved`),
  which the GPU lane owes.

## Correction, 2026-10-03

The first two figures above included a start-up transient. The object began from its parts'
separate spin-ups, and its contacts equalised on the first tick. Since ADR 0157's third
amendment the object is spun up whole. With that, the bells rise **0.5 K** in the climb's first
minute (not 1.4 K) and **12.5 K** by its end (not 14 K), as station means. The figures from
landing on are unchanged to 0.01 K. The bell-lag test held either way: it compares the
first-minute rise with the five-minute rise.
