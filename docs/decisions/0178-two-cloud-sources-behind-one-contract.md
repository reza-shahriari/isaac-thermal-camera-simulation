# ADR 0178 — Two cloud sources behind one contract, until one is accepted

**Status:** Accepted
**Date:** 2026-10-01
Implements the contract in `docs/physics-model.md` §7.5 ("Two sources, one contract"). Roadmap:
`WX.2`, with parity in `WX.7` and the choice in `WX.9`. Builds on
[ADR 0146](0146-a-cloud-in-the-infrared-has-one-base-a-dry-adiabatic-temperature-and-its-own-march.md)
(the infrared band's own march over the shared field) and
[ADR 0162](0162-the-cloud-is-marched-to-the-hit-for-every-pixel.md) (marched to the hit), and on
[ADR 0144](0144-a-vdb-cloud-renders-as-a-volume-box-with-the-density-as-a-texture.md) for how a
VDB reaches the path tracer.

## Context

The owner does not like the clouds. Diagnosed in the 2026-10-01 weather audit
(`docs/research/2026-10-01-weather-in-the-infrared.md`): the field is coarser than the 8–20 m
optical skin a cloud edge actually has, the dome lights thin edges dark, and the infrared march
sizes its steps by the 60 m grid and so steps over the edge detail.

There are two ways to get a cloud that looks like one, and they fail differently:

* **A procedural field** (`isaac-weather-fx`'s `CloudField`: a Perlin–Worley envelope, eroded
  at the edge by a 15 m detail texture) is controlled by cover, genus, base and drift, and tiles
  forever. Whether it can be made to *look* right is not yet shown.
* **Volume assets** (`HeroClouds`: an OpenVDB cloud such as the Disney cloud, CC BY-SA 3.0, or
  an LES snapshot such as LASSO's, CC BY 4.0) have the shape of a real or artist-made cloud by
  construction. Cover, genus and the field's edge are hard to control, the data is large, and
  today only the path tracer draws them: the infrared camera does not see a hero cloud at all.

The owner asked for both to be built to the same standard, so that one can be accepted later by
looking at them, rather than for a choice to be made now on argument.

## Options considered

1. **Procedural only.** Cheapest, one code path. But it bets the result on the option whose look
   is unproven, and it is not what was asked.
2. **Volume assets only.** Real shapes. But it gives up cover and genus as inputs. It would also
   need an infrared path for assets that does not exist yet, and it is not what was asked either.
3. **Both, behind one contract.** Each renderer (the visible dome, the path-traced volumes and the
   infrared march) reads a source only through the same few members, so either source can drive
   every band. A switch picks one per scene, and a side-by-side clip decides. Costs a second
   source to keep at parity until the choice is made.

## Decision

Option 3.

* **The contract** is what `WeatherFxDeck` and the two visible renderers may read:
  * the density in 0..1 at any point of the field frame;
  * the visible extinction at unit density, per metre;
  * the base and the top;
  * the genus and its microphysics, from which the band ratio comes (§7.5);
  * the drift as a function of time;
  * **the finest pitch the source carries.**
* **Both sources live upstream.** `CloudField` and `HeroClouds` stay in `isaac-weather-fx`, and
  this repository consumes them through the submodule, never through a copy.
* **The finest pitch lands first, because it is the one the infrared band already gets wrong.**
  `CloudField.finest_pitch_m` is the smallest of a cell, a level and the detail texture's own
  cell. `HeroClouds.finest_pitch_m` is one voxel. Both were added upstream in `11bf843`.
  `WeatherFxDeck.sample_pitch_m` now reads it, and the march places **three** samples per pitch
  (`samples_per_pitch`). Two would be the Nyquist spacing of a smooth trilinear field, but the
  cloud's boundary is clipped to a step, so the residual falls only as 1/n.
* **The path tracer's volume settings come from upstream's tables.**
  `path_traced_volume_settings()` reads weather-fx's `VOLUME_SETTINGS`, `VOLUME_FLOORS` and
  `VOLUME_BOUNCE_SETTINGS`, with `clouds.volume_bounces` (32) as the volume bounce floor; the
  renderer's own is 2. The bump renamed one of them (`MIN_PATH_BOUNCES`), which the engine
  would have found at render time. The unit gate now finds it first.
* **The rest of the parity is `WX.7`.** A hero source gets a deck, the infrared march, cloud
  occlusion and the real-time dome. The switch and the comparison clip are `WX.9`.
* **Both sources answer to the same tests.** These are the visible criteria of `WX.3` and the
  shape targets of `WX.6` (§7.5's table). A source that fails them is not a candidate.

## Consequences

* **The march resolves the edge.** Measured against a 4096-step reference of the same field,
  as the 99th-percentile error in band emissivity:

  | deck | sized by the grid (before) | by the finest pitch (after) |
  |---|---|---|
  | bridge test deck, three seeds (140 m cells, 25 m finest) | 0.019–0.024 | 0.006–0.007 |
  | production, 20°–60° frame (60 m cells, 12.5 m finest) | 0.018 | 0.009 |
  | production, 5°–30° frame | 0.012 | 0.005 |

  0.01 of emissivity is about a quarter of a kelvin of cloud.
* **It costs steps where the cap does not already bind.** On the production grid every frame
  now reaches the existing `max_steps` cap of 512. The steep frame went from 281 steps to 512,
  and its march from 2.6 s to 4.6 s for 82 000 rays (1.8×). The shallow frame was already at
  the cap and costs the same. So on production the spacing is the longest crossing over 512,
  up to 23 m on a 12 km path. That is coarser than the detail, and `WX.4` is what makes it
  enough: it filters the detail to the pixel, so at that range the detail is not there to miss.
* **Two sources cost upkeep** until `WX.9`. A fix to one renderer must hold for both, and the
  tests are written against the contract so that they can.
* **The error this ADR does not bound** is the look. Neither source is shown yet to pass §7.5's
  five visible criteria. That is `WX.3`'s to measure, not this ADR's to promise.

## Revisit when

* The owner accepts one source after `WX.9`. A follow-up ADR then retires the other or keeps it
  as an option, and this one is superseded.
* A third source is wanted, such as a cloud resolved by a radiative-transfer code. It joins
  through the same contract or the contract is wrong.
