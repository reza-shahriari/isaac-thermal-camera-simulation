# 0089 — §6.6's vehicle heat sources need their own solver kind, not the aerial one

Date: 2026-09-15
**Status:** Accepted
Roadmap: MP.4a (§6.6)

## Context

`irsim.thermal.vehicle` has held §6.6's ground-vehicle table since M6.14 — `engine_bay` (+65 K,
τ_rise 750 s, τ_cool 1800 s), the exhaust chain, brake discs, tyres — together with
`SourceHistory`, which integrates the first-order rise and cool by an exact exponential step. None
of it has ever been reachable from a scene config. Every vehicle in this repository is either an
aerial node or a typed-out `prescribed` schedule.

Building MP.4's ignition demo made that concrete. The scene needs an engine bay that is at ambient
in frame 0 and warms over the following ten minutes, which is the whole subject of the film.

The two obvious ways to author it are both wrong:

1. **`solver: heat_source`.** This is ADR 0072's *aerial* node, and its law is a **steady-state
   relation with no time constant at all** — the schema's own docstring says so. Its temperature
   follows the throttle instantaneously, so an engine bay authored through it shows its full 65 K
   in the first frame after ignition. It also validates `source` against
   `AERIAL_HEAT_SOURCES`, so `engine_bay` is not even a legal name there.
2. **`solver: prescribed` with the curve written out.** This is what the maritime scene had to do
   for its funnel, and its own comment calls it a wart. It moves a number the model already knows
   into a YAML file, where it stops tracking the model and becomes a number somebody typed. The
   demo would then be showing an author's arithmetic, not §6.6.

## Decision

A sixth solver kind, `vehicle_source` (scene schema **v6**): `source` names a row of
`VEHICLE_HEAT_SOURCES` and `load_s` → `load` is the duty fraction over time.

`VehicleSourceSolver` is deliberately **only an adapter**. The law and the exact-exponential step
stay in `SourceHistory`, which already had them; duplicating the integration to give it a
`TemperatureSolver` interface would have created a second copy of §6.6 that could drift from the
first. In particular the choice between `τ_rise` and `τ_cool` stays in `SourceHistory`, where it is
made by **direction** rather than by whether the key is turned — a source whose load has just
dropped is cooling toward a lower target even while the engine runs, which is the physically
meaningful reading.

It is also the first solver that **cannot be pre-derived into a `PrescribedSolver`**, which is what
`build_target` does for every aerial node. A node with a time constant depends on its own history:
the same load profile started from a cold engine and from one that parked ten minutes ago gives
different curves, and `delta_t0_k` is how a scene says which.

## Consequences

**Easy:** a scene can now film a vehicle starting, idling, being switched off and cooling, with
§6.6's own numbers and no schedule to maintain. `delta_t0_k` additionally makes the far more common
daylight subject — a car that parked a few minutes ago and is forgetting its engine — a
one-parameter change rather than a different model.

**What it does not fix.** These rows are all `ESTIMATED` midpoints of §6.6's quoted ranges and
nothing here makes them measurements. A `vehicle_source` node produces §6.6's curve exactly; whether
§6.6's curve is a real engine bay is a Tier 4 question against public data, unchanged by this ADR.

**Tyres and brakes are reachable but should usually not be used through this kind.** §6.6 gives the
tyre as a *speed* relation and the brake disc as an *energy deposit*, and `vehicle.py` implements
both properly. Driving them from a duty fraction instead is a coarser statement, and for a
stationary vehicle the honest answer is that neither warms at all: tyre heating is flexing work and
brake heating is kinetic energy, so a car idling in a car park has cold wheels however long it
idles. The MP.4 demo says so in its readout rather than staging a rise nobody computed.

**Schema cost.** `SCENE_SCHEMA_VERSION` goes to 6 and the five committed demo configs are bumped
with it; `MIN_SCENE_SCHEMA_VERSION` stays at 4, so every older file still loads.

## Revisit when

A scene needs a driving vehicle, at which point the tyre and brake models want to be driven from a
`VehicleState` trace (speed, braking) rather than from a duty fraction — the trace type already
exists and `SourceHistory.step` already takes one.

## Addendum (`TC.8`, 2026-09-24): the trace arrives, and a parked car has cold wheels

"Revisit when a scene needs a driving vehicle" is now. `irsim.thermal.drive_cycle` walks a
`VehicleState` trace and drives the two models this ADR told scenes not to reach through
`vehicle_source`: the brake disc as an energy deposit and the tyre as a relation in speed. Until
it, `brake_temperature_rise_k` and `tyre_delta_t_k` had **no caller outside their own unit tests**,
so no frame this project has rendered has ever contained a warm brake.

Three decisions worth recording, because each has a plausible alternative:

**1. The integration still exists once.** `SourceHistory` gained `step_to_target`, and its old
`step` is now that method with a load-derived target. A tyre's target is not a duty fraction, so
the alternative was a second exact-exponential step inside the drive cycle — which is the second
copy of §6.6 this ADR's Decision section exists to prevent.

**2. The disc cools across the interval before the deposit lands.** The deposit is the work done
arriving at the sample; the cooling is what happened on the way there. Depositing first and then
cooling the sum lets a hard stop lose part of its own energy to a constant it has not yet spent,
and makes the answer depend on the trace's sample spacing — the property the exact-exponential
step exists to remove.

**3. A parked car has cold wheels, and §6.6 does not say so.** `tyre_delta_t_k` returns its lower
bound, +10 K, at zero speed, because §6.6's "+10 … +35 K, rises with speed" describes a *rolling*
tyre and its bottom end is a tyre rolling slowly. Tyre heating is flexing work, so a stationary
tyre has no source at all. The drive cycle therefore targets **zero** when the vehicle is stopped
and lets the tyre forget over its own τ_cool of 1800 s: a car that has just pulled up has warm
tyres and one that parked an hour ago does not. Reading the relation literally at v = 0 would give
every car in every car park a +10 K wheel — a feature a detector would learn and no camera sees.
This is a departure from §6.6 as written and is recorded here rather than left in the code.

The front/rear split is `FRONT_AXLE_BRAKE_FRACTION = 0.65`, ESTIMATED like every other number in
§6.6's table. It matters to a picture rather than to a total: the front discs run nearly twice as
hot as the rear, and an even split would render four identical wheels.

Measured on the demo car: forty minutes at 27 m/s puts the tyre **28.1 K** over ambient, and the
arch liner 0.12 m above the tread receives **285.9 W/m²** against the door's **0.11 W/m²** — some
23.8 K against 0.01 K through an ESTIMATED panel conductance. One stop from 30 m/s deposits
**162 K** into an 8 kg disc (ADR 0038's own figure) and four times that from 60 m/s.
