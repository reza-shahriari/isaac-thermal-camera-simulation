# ADR 0109 — A patched surface's speed follows the mission

**Status:** Accepted
**Date:** 2026-09-21
Roadmap: PT.9 (§6.1, §6.6; engine-free half)

## Context

`SurfaceSpec.vehicle_speed_m_s` is one number per surface, and it is what turns free convection
into forced in `convection_coefficient`. That is right for a parked car and wrong for anything
that flies: a quadrotor spends the first two minutes of its mission on the pad and the next
twenty in the air, and the difference between those two convection regimes is **15 K on a
sunlit carbon deck**. Regenerating the aerial scenario point-wise (the owner's own next step
after phase P) was blocked on exactly that: with a constant speed the scene can show the deck
hot for the whole mission or cool for the whole mission, and neither is the flight.

ADR 0072's `airframe` solver sidestepped this by *asserting* the answer — the skin is at the air
temperature — which is the flying case stated as a law. It is a good approximation in cruise and
30 K wrong on the pad, and nothing in the scene could tell the two apart.

## Options considered

1. **A separate "on the ground" scene and an "in flight" scene.** Two configs for one mission,
   and the transition — which is the interesting part — appears in neither.
2. **Derive the speed from a target node** (`speed_from: {target: motor}`). A target is a
   temperature, not a velocity; reading a speed out of one would be inferring the flight from
   its thermal consequence, backwards.
3. **A speed schedule on the surface**, `speed_s` against `speed_m_s`, the same shape as a
   target's `throttle_s`/`throttle` and `load_s`/`load`, linearly interpolated and held flat
   outside. The mission is already authored as what the pilot did; this is one more column of
   it. Chosen.

## Decision

`SurfaceSpec.speed_s` / `speed_m_s` (schema v13) and `SurfaceOrientation.speed_schedule`, whose
`speed_at(t)` `SceneSurfaceForcing` calls once per evaluation in place of the constant. A surface
declares **one** speed authority — the constant or the schedule — and both together is a load
error, not a precedence rule. Held flat outside the schedule, so the spin-up before `t0` runs at
the first entry, which for a platform sitting on its pad is exactly right.

## Consequences

**Measured**, on `configs/scenes/quad_flight_pointwise.yaml`: the deck sits 29.3 K over air on
the pad, 13.8 K over it in the hard climb at 13 m/s, and 29.6 K over it again once landed. Freeze
the schedule at the pad's 1 m/s and the climb never cools — the negative control that shows the
collapse is the mission and not the sun moving. Every cell still holds its own equilibrium to
under 1 mK against `steady_state_temperature` on its own flux.

**What it does not do.** Rotor downwash is not modelled: the 1 m/s on the pad is a stand-in for
it and a real hover moves more air over a deck than that, so the pad excess above is an upper
bound. The schedule drives **convection only** — a skin fast enough for aerodynamic heating
needs the recovery temperature ADR 0075 derives, and a patched surface has no route to it yet,
which is why the aircraft-pass scene stays per-prim. The powered parts of the quadrotor
(motors, speed controllers, the pack) also stay per-prim: conducting their heat into the skin
from a config still has no route, and that is the remainder of `PT.9` beside `WM.3`'s curved
binding.

## Revisit when

A patched surface needs the ram recovery temperature (ADR 0075 in the forcing rather than in a
target), a speed that varies *across* a surface (a rotor disc), or the schedule to come from a
flight trace rather than from the config.
