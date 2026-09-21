# ADR 0108 — Still water as a region of a patch, with a signed skin

**Status:** Accepted
**Date:** 2026-09-21
Roadmap: PH.3 (§6.1, §16.2; spec issue S45)

## Context

`sea_skin.py` (ADR 0080) models the ocean and `configs/materials/water.yaml` carries the only
measured n/k table in the library, but nothing in the repository could put water **on land**. The
owner's requirement names water beside fire as a phenomenon the simulator has to have, and the
form it takes on a road is a puddle: a few tens of millimetres, a few metres across, on the same
prim as the asphalt around it.

Two things about standing fresh water are not the sea's. Its skin can be **warmer** than its
bulk: ADR 0080 clamps the cool-skin deficit at zero deliberately, because there the warming term
is absorbed sunlight, deposited metres down, which cannot reverse a gradient across a 2 mm film.
Over a pond on a humid overcast night the fluxes that reverse are longwave (absorbed in ~20 µm of
water) and condensation (deposited exactly at the interface); both land *in* the sublayer, so the
gradient genuinely inverts. And a puddle is not a scenario input like an SST — it is a body with
a mixed layer whose mass is what makes it lag the road.

## Options considered

1. **A puddle as its own `SurfaceSpec` with its own patch.** Two patches over the same geometry,
   two prims or one prim bound twice, and a terminator between them that neither solve knows
   about. Rejected: the row's own framing is "a region of the road".
2. **A two-layer stack, water over asphalt (`layers:` with a water top).** `LayerStack` is one
   material cut into equal slices; a stack of two different materials is an API the schema does
   not spell yet (ADR 0103's own revisit trigger), and it would still not give the wet cells
   their own optics.
3. **A per-cell override on the road's patch.** The cells inside the region keep the patch, the
   forcing, the shadow, the sky view and the lateral conduction they already had, and swap three
   numbers: areal capacity, emissivity and solar absorptivity — plus a film of the puddle's own
   depth, which is PH.2's machinery unchanged (ADR 0101) and gives the latent term something to
   evaporate. Chosen.

## Decision

`irsim.thermal.still_water` — `StillWaterParams`, `sublayer_thickness_m`, a **signed**
`skin_offset_k`, `skin_temperature_k`, `mixed_layer_capacity_j_m2_k`, and
`apparent_temperature_k` (the ε(θ)·B(T) + (1−ε(θ))·L_sky mix a camera actually reads) — plus
scene schema v12's `water:` on a patched surface: `depth_mm`, an optional `region_m` in the
patch's own metres, and the material (default `water`). Fresh-water ν, k, ρ and c join
`irsim.radiometry.constants` rather than reusing the seawater values, which differ by 4 %, 0.3 %
and 2.6 %: a reader should not have to discover that a road's puddle was modelled with brine.

**The puddle's capacity is the water's plus the substrate's.** The road under a puddle has not
gone away, and 20 mm of water over 50 mm of asphalt are coupled far more tightly than a day is
long. Dropping the asphalt's mass would be the larger error; lumping the two is the smaller one,
and for a deep pond the bed's share is negligible anyway.

**Saunders' λ = 7 and a 3.8 mm free-convection bound are a calibration, and are labelled one.**
Saunders (1967) puts λ between 5 and 10 and later work makes it a function of surface buoyancy
flux. These two values put the sublayer at 0.67 mm at 8.2 m/s and 3.61 mm at 0.8 m/s, which is
the 0.7–3.6 mm band the survey quotes for standing water. Two free parameters inside the
literature's own range, fixed against a stated band, said out loud in the docstring.

## Consequences

**Measured:** the sublayer spans 0.67–3.61 mm over 0.8–8.2 m/s and thins strictly with wind. A
pond at 288 K under a clear sky with 289 K air at 60 % humidity and 2 m/s loses 96 W m⁻² and its
skin sits **0.38 K** below the bulk, inside the row's [−0.5, −0.1] K. Overcast, with the air 3 K
warmer and 95 % humid, the net flux points *into* the water and the skin runs **+0.21 K** above
the bulk, where the sea's clamped form reports exactly zero and misses the sign. Kirchhoff closes
on water to 1e-12 from nadir to 89.5°. A puddle at 293 K under the shipped clear sky reads 0.55 K
below its kinetic temperature at nadir, 1.90 K at 60° and 4.40 K at 70°; overcast, the 60° deficit
collapses below 0.5 K. On PH.2's 14:00 road scene, 20 mm of water over half the patch opens 3.8 K
below the dry half and is 11.0 K below an hour later, having evaporated 0.85 of its 20 kg m⁻²;
the dry cells more than 2 m from the edge are bit-identical to the same scene without the water.

**Deviations from the row, both in the same direction and for the same reason.** The row asks for
"several kelvin colder at 60° off nadir" and "within 0.5 K at nadir"; this gives 1.90 K and
0.55 K, because the shipped clear-sky model puts the zenith at 226 K in the Boson band — colder
than the row assumed, which *deepens* the nadir deficit and moves "several kelvin" out to 70°.
The mechanism, its angular shape and its dependence on cloud are exactly as specified.

**What it costs and does not do.** The film is laid at t₀ and not spun up (ADR 0101's semantics),
so the spin-up sees water's optics and mass but not its evaporation; a puddle that has been
standing for days starts from the substrate's spun state. The skin offset is a function a caller
applies, not a term the field carries: a scene's puddle reports its **bulk**, and the ≤0.5 K skin
correction is available to whoever renders it. Lateral conduction across the puddle's edge uses
the *road's* k·δ, not water's. Water on a layered surface, and `water:` beside an authored
`film:`, are both refused rather than guessed at.

## Revisit when

A scene needs a pond with a bed at its own temperature (a bed conductance on the spec), a puddle
whose evaporation is spun up, the skin offset folded into what the bridge reports, or a stack of
two different materials — the last is ADR 0103's revisit trigger and would subsume option 2.
