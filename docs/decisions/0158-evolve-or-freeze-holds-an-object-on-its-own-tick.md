# 0158 — Evolve or freeze holds an object on its own tick, and a held object stays a boundary

Date: 2026-09-28
**Status:** Accepted
Roadmap: TC.12 (§6.5, §12.3); builds on ADR 0093 (the tick), ADR 0096 (the network), ADR 0157 (the exchange)

## Context

The owner asked for two controls with the same shape (`thermal-realism-switches-are-requirements`):
an object that does not evolve (a road whose history is the point of the scene while the car on
it is a prop), and an object frozen at a chosen moment (a warm engine at night, kept warm while
the camera films). In both cases the frozen object must go on radiating to and convecting with
its neighbours, and noise and motion go on. Unset, every scene must be bit-identical to today.

## Options considered

1. **Stop the object's clock.** No ticks after the hold; a query past it answers the last tick.
   Cheapest, but every consumer that walks ticks (the exchange group's lockstep, the on-tick
   recorder, the state hash) would see a member that has stopped producing, and the group would
   refuse it as out of step.
2. **Keep the clock, hold the state.** Every tick after the hold is produced as before -- into
   the ring, the hash and the hook -- carrying the same temperatures; no forcing is evaluated
   and no solve is taken. The object is still *there* at every tick, and its neighbours read it
   as they always did.
3. **Prescribe the object.** Replace the solver by a prescribed schedule at the frozen value.
   Loses the object's history before the hold and its identity (a network node cannot become a
   prescribed target without rewiring its links).

## Decision

Option 2, in three places that share one rule (`irsim.thermal.hold.hold_from_s`): `evolve: false`
holds from t₀, `freeze_at_s: t` holds from t₀ + t, and both at once is refused by the schema.

* **A field** (`ThermalField.hold_from_s`) holds on its own tick: ticks whose start is at or after
  the hold copy the state forward. A hold that falls between ticks therefore takes effect at the
  first tick past it, to the tick, which is the precision the field has anyway. Per facet as
  well as per field, so the scene's per-prim field can hold one surface while the others solve:
  the solver steps and the held facets are written back.
* **A lumped target** is wrapped by `HeldSolver`: steps before the hold pass through untouched,
  the step that straddles it advances the inner solver exactly to the hold, and every later step
  answers the held temperature. `weather` is forwarded so the one-weather guard sees through.
* **A network node** is held by giving it, for the solve, an infinite capacity and writing the
  held value back: the backward-Euler step then treats it as a boundary at its held temperature
  within the same tick, so heat still flows through its links exactly, and the energy identity
  counts what flows into it as it counts a fixed node.

**Precedence.** A surface's, target's or node's own `evolve` / `freeze_at_s` wins; unset, it takes
the scene's `thermal.evolve` / `thermal.freeze_at_s`; unset there too, it evolves. A cabin panel
and a prescribed map cannot be held (the cabin is one solve; a map is not solved) and are refused
by name.

**Verification** (`tests/unit/test_hold.py`, `test_scene_hold.py`): a field with no hold hashes
identically to one that never had the attribute; held, it is bit-identical up to the hold and
constant after while the free field goes on warming, with the same tick count; a hold between
ticks takes effect at the next tick; one facet of three holds while the other two evolve; a pan
held at 320 K over a road at night keeps the road under it warmer than the same pan left to
cool; a held target equals its inner solver step for step until the hold and is constant after;
a held network node stays put while its neighbour keeps moving toward it; at the scene level a
surface with `evolve: false` is constant, a target with `freeze_at_s` stops on time, the scene's
own switch holds everything, and no key gives the same state hashes.

## Consequences

* Frozen is non-physical by design: a held object's own energy balance is suspended (its stored
  energy does not change while heat still crosses its boundary). That is the point of a prop.
* Because the clock keeps running, a frozen field costs a tick's bookkeeping and nothing else.
* The exchange group's spin-up gap (ADR 0157) is the natural next use of the same hold: a scene
  can freeze a car that has "always" been there and let the road under it evolve.
