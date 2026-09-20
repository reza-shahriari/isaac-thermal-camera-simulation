# ADR 0100 — The engine as a solved node, not a schedule

**Status:** Accepted. Supersedes ADR 0089's `vehicle_source` adapter **for the engine bay**; the
other §6.6 rows (exhaust chain, brakes, tyres) keep it until TC.7.
**Date:** 2026-09-20
Roadmap: TC.5 (§6.6, §6.4; spec issue S43)

## Context

§6.6 scripts the engine bay: `T = T_air + ΔT_max (1 − e^{−t/τ})`, τ_rise = 750 s, τ_cool =
1800 s, "scripted, not predicted". ADR 0089 gave that schedule a solver kind and it carried both
car scenes. The 2026-09-18 survey put numbers beside it that a schedule cannot produce:

* after key-off the block and coolant cool with a time constant of **hours** — a cooling-system
  diagnostic (US 9,790,842) fits ECT − ambient to `a·exp(−t/τ)` and reaches ambient in ~7 h from
  93 °C at 27 °C, i.e. τ ≈ 1.7 h; the schedule's 1800 s has it cold in one;
* the bay air **spikes** at key-off (ThermoAnalytics: +40 K; 20–50 K for a passenger car) because
  forced convection stops while the high-mass parts keep radiating;
* a scripted rise cannot depend on ambient, airflow, load history or the metal bolted to the
  block — which is the owner's second requirement, an engine warming the parts around it.

TC.2 built the network and TC.4 its schema; TC.3 the couplings to fields. What was missing was
the engine itself as nodes and links, with heat in watts.

## Options considered

1. **Keep the schedule and add a key-off τ_cool of 1.7 h.** Fixes one number and nothing else:
   the bay air cannot spike, the block cannot warm a bracket, and the skins cannot lag.
2. **A full under-hood model** (block skin and core, manifold, catalyst, radiator loop with a
   thermostat, fan curve). Right and far beyond an M-sized step; every extra node is another
   estimated number.
3. **A four-node network**: block + coolant as one lumped node with imposed heat
   `P_rated · load · bay_fraction`; a bay-air fluid node the block convects into and that vents to
   ambient, both conductances switching forced → natural when the engine stops; rubber mounts as
   a link node to a subframe that convects to ambient; radiation from the block to the bay walls
   and the road. Every number ESTIMATED, but each a mass, an area or a coefficient that can be
   measured for a vehicle. Chosen.

## Decision

Option 3, `irsim.thermal.engine` (`EngineSpec`, `engine_network`, `EngineSolver`) and a seventh
target solver kind, `solver: engine`, taking `load_s` → `load` like `vehicle_source` and no
`source`.

* `EngineSolver.temperature()` reports the **bay air**: the cavity temperature ADR 0088's
  radiator carries onto the bonnet. The bonnet field's forcing is therefore unchanged and the
  post-key-off overshoot reaches the skin through the term it always used. The block and every
  other node are read through `node_temperature_k`.
* Forced convection (fan / ram air) holds while `load > 0` or the car moves; natural otherwise.
  That switch, and the block's mass, are the hot soak.
* The defaults were tuned to the survey's bands and nothing else: full-load rise +54 K (§6.6:
  +40…+90), key-off from 93 °C at 27 °C: +32.3 K after 1 h, +0.92 K after 7 h, bay-air overshoot
  +27 K peaking 145 s after key-off. `bay_fraction` (0.025 of 90 kW) is the least-founded
  number: the share of engine power that heats the bay rather than leaving by the tailpipe and
  the radiator's airflow. It is the one to measure first.

## Consequences

**Easy now:** TC.6 declares the car scenes' engine as `solver: engine` and links a bracket and
the bonnet to it; TC.7 hangs the exhaust line off the block node.

**What it costs:** a §12.3 scene author now sees a load schedule and no temperatures; the
schedule's `delta_t_max` and `tau` are gone for the bay, replaced by twelve estimated
parameters with a default set. `vehicle_source: engine_bay` still parses and behaves as before,
so no scene breaks; the migration is TC.6's.

**Error introduced, and one deviation from the row recorded plainly:** one lumped block has no
fast skin, and the bonnet is not yet a loss path for the bay air (that needs TC.6's coupling), so
after key-off the bay air stays elevated as long as the block does and the bonnet over the block
keeps warming for ~23 minutes before it turns, where the row asked for 60–120 s. The 60–120 s
figure is the survey's **manifold-skin** number (MVFRI R04-13) and belongs to TC.7's exhaust
line; this step holds itself to the lower bound — the bonnet keeps warming after key-off, which
the schedule could not do — and records the peak time as measured.

## Revisit when

TC.6 couples the bonnet (its skin becomes the bay's loss path and the overshoot shortens), TC.7
adds the manifold and catalyst skins, or a vehicle's measured under-hood data arrives, at which
point `EngineSpec`'s defaults become a fit rather than an estimate.

## Addendum 2026-09-20 (TC.6): the thermostat, the block as the cavity, and what still waits

Migrating the car scenes onto this engine showed three things the first cut had wrong or missing.

1. **The block needs its coolant loop.** With only the bay's share of power heating it, an
   idling block reached +11 K in twenty minutes and the bonnet over it 2.4 K of gradient; a real
   idling engine reaches its thermostat. `EngineSpec` now heats the block with
   `P_rated · load · block_fraction` (the thirds rule, `block_fraction = 0.9`) and holds it with
   a **thermostat**: a proportional element opening over `thermostat_band_k` (6 K) above
   `thermostat_k` (90 °C) onto a radiator node at that temperature with `thermostat_g_w_k`
   (5 kW/K) fully open, closed at key-off. Proportional rather than a switch, because a switch
   evaluated at the tick's start bang-bangs by `Q dt / C` per tick. `load` is now the duty
   fraction of rated power (an idle in a car park is 0.10, not the scripted rows' 0.45-of-ΔT).
   Measured: full load +78.5 K and flat to 1e-6 per tick, idle +65.5 K (the thermostat), key-off
   +32.2 K after 1 h and +0.91 K after 7 h, bay-air overshoot +40 K peaking 140 s after key-off.
2. **The target reports the block, not the bay air.** ADR 0088's radiator under the bonnet is the
   hot mass; the bay air at idle is vented to within ten kelvin of ambient and would leave the
   bonnet flat. `EngineSolver.temperature()` is the block; the bay air stays readable. The
   overcast scene's bonnet shows 22.8 K max–min at 1500 s (the clear scene 19.7 K), inside the
   row's 10–40 K.
3. **Nodes that hang off a target, one way.** The scene's `nodes:` gain `follows: {target, node}`,
   a boundary that reads a solved target's node each tick, and `links:` gain
   `switch`/`off_h_w_m2_k`, a convection that drops when the named engine stops. The bracket,
   mounts, subframe and wing hang off the engine's block without cooling it (a hundredfold mass
   ratio); parts warm block → bracket (30 W/K bolted) → mounts (12 W/K rubber) → wing → subframe.

**Still waiting:** the bonnet joined to the body by a contactor, and the bonnet skin as the bay's
loss path, need one implicit system across the network and a field (`CoupledFields` joins
fields; the network is its own solver). The row's "bonnet by contactor" is therefore not done;
the bonnet is coupled by radiation from the block, as before, and the deviation is recorded here
rather than in a passing test.

