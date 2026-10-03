# ADR 0180 — The visible cloud is lit by the sky, by single scattering, and by the two-stream field along each sun chord

**Status:** Accepted
**Date:** 2026-10-03
Implements `docs/physics-model.md` §7.5's criteria for the visible companion's cloud ("How the dome meets
them"). Roadmap: `WX.3`. The change is in `isaac-weather-fx` (`ac7b3b9`); this repository takes it as a
submodule bump. Builds on [ADR 0178](0178-two-cloud-sources-behind-one-contract.md): both cloud sources
are to be held to the same criteria.

## Context

The owner does not like the clouds; the 2026-10-01 audit found the visible lighting flat. Measured on
upstream before this change (35 % cumulus, 58° sun), against §7.5's criteria:

* **No sky light at all.** A cloud added light only from the sun or the moon. At night, or under a
  uniform sky with the sun off, a non-absorbing cloud rendered as a black shape. It fails the white
  furnace outright, and at civil twilight the clouds were black against the glow.
* **Brightness from one envelope.** The magnitude was the two-stream reflectance of the ray's own
  optical depth, multiplied by a shadow factor from Wrenninge-style octaves. A stand-in constant
  (`BASE_ENVELOPE = 1`) made every base as bright as a sunlit top. The from-below/from-above choice was
  made per ray, so a sunlit side seen from the ground was treated as a base.
* **A deck seen from above** averaged 0.92 of its own two-stream albedo, and some thin edges were 0.92
  of the sky behind them.
* **The light maps were biased.** They stored transmittance, and interpolating it next to a clear cell
  made a base look nearly unshadowed: 0.6 of the depth a fine integration gives.

## Options considered

1. **Keep the octaves and envelope, add an ambient term.** Cheapest. The furnace would pass, but bases
   stay as bright as tops and the per-ray face choice stays wrong.
2. **Precomputed multiple scattering**, a table over depth, density and angle (Bouthors 2008; Hillaire
   2016). This is the production answer, but it needs a precomputation per genus and sun, and a second
   code path to keep consistent with the infrared.
3. **Single scattering plus the two-stream field along each sun chord, plus the sky** (chosen). Every
   term is a closed form a reviewer can check against the plane-parallel limit. It needs one more light
   map, toward the dark side.
4. **Leave the look to the path tracer.** The volumes can carry it, but the real-time dome needs its own
   light, and the dome is what the companion frame uses in the real-time tier.

## Decision

Option 3, in `weather_fx.core.clouds.CloudField.march` and `weather_fx.core.sky._cloud_terms`. Every
step in-scatters, with the energy-conserving weight ϖ T (1 − e^{−Δτ}), three lights:

* **Sky and ground** (`MarchResult.ambient_above`, `ambient_below`): the step's height fraction splits
  its weight between the clear sky's hemispheric means above and below (`sky.cloud_ambient_rgb`). This
  is returned with or without a sun.
* **The sun once**: the HG phase function over 4 μ☉, times the transmittance toward the sun.
* **The sun many times**: the δ-Eddington two-stream streams along the sunlight's chord,
  `R(1 − f)` backward and `T f` forward. The ray reads the stream leaving the face it entered, judged
  from how the depth fraction f has changed since it entered the cloud. It in-scatters the source that
  stream implies, `S = I − dI/dτ`.

The supporting changes:

* The light maps store optical depth (`sun_optical_depth`), at full vertical and half horizontal
  resolution.
* `_cloud_terms(origin_m=)` lets an observer sit off the ground.
* `BASE_ENVELOPE` and the octave constants are gone.

## Consequences

* **The criteria, measured** (35 % cumulus, 58° sun; `tests/test_cloud_light.py` upstream):

  | criterion | before | after |
  |---|---|---|
  | white furnace | fails (a black cloud) | within 1e-4 |
  | thin edge vs the sky behind, minimum | 0.92 | 1.01 (median 1.9) |
  | thick base (τ > 30) vs sunlit flank | — | 0.82 at a 44° sun, 0.89 at 58° |
  | deck seen from above vs two-stream | 0.92 | 1.005 |
  | uniform layer, τ 0.5 / 4 / 50 | — | R above and T below, each within 3 % |

* **It costs about a third more per bake.** The production dome bake (1024 rows) went from 3.1 s to
  4.2 s, mostly for the second light map and the full vertical resolution. Both maps are cached per sun
  direction, so a clip at one time of day pays once.
* **Sunsets look like sunsets.** Sun-facing cloud faces at a 4° sun are far brighter than the grazed
  ground the incident meter exposes for. The meter now runs at its highlight cap, and the brightest
  percent of a sunset is brighter than noon's, as in a photograph exposed for the ground. Upstream's
  tone tests were restated to measure what they guard:
  * the bulk of a sunset frame stays under noon's (median 0.92; the old flat-yellow sunset was 1.32);
  * the metered highlights stay inside the meter's headroom (0.9 of it).
* **Not modelled, with the error it leaves:**
  * *Light leaking out of a finite cloud's sides.* The march is one-dimensional along each sun chord,
    so the base of a small cumulus at noon is as bright as two-stream says a layer of its depth is. For
    τ ≈ 10 that is half the sun, diffusely. Bases are dark only for thick cloud.
  * *A cloud's shadow on the air behind it.* A wisp shadowed by its own cloud near the sun reads 0.95 of
    the unshadowed aureole. The thin-edge criterion is applied outside 15° of the sun for that reason.

## Revisit when

* The comparison clip of `WX.9` shows small-cumulus bases too bright to accept. Side leakage, from
  horizontal optical-depth maps that are independent of the sun, is the next term to add.
* A precomputed multiple-scattering table becomes available for both cloud sources, as option 2.
