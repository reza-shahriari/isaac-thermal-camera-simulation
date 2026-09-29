# 0169 — Cloud occlusion follows the render path; the cloud itself is in both

Date: 2026-09-29
**Status:** Accepted
Roadmap: AT.31 (implements it), XD.6 (moved into phase A as the realism gate); amends ADR 0162.
Spec: `docs/physics-model.md` §7.5.

## Context

ADR 0162 made the infrared band march the shared weather-fx cloud to every pixel's own hit, so a
target behind or inside a cloud is attenuated by it. The visible companion only occludes a target
when the path tracer renders weather-fx's 3-D cloud volumes. In RTX real-time mode, and in every
headless irsim render today (`author_weather_fx_sky` attaches the sky effect only), the visible
cloud is painted on the dome at infinity and every target draws in front of it. The infrared band
therefore occludes where the visible band cannot, and the owner treats any disagreement between
the two bands as a defect.

The owner settled the trade on 2026-09-29: "about the realtime mode, its ok to dont have some of
the features to have better speed, so we dont need an object between the clouds, but the clouds
should apear and be realistic in ir view (i know path trace is better, it could be as good as
possible)".

## Decision

1. **Two tiers.** The path-traced tier is the full-fidelity one: 3-D volumes in the visible band,
   the march to the hit in the infrared, so a target behind or inside a cloud is occluded in both.
   The real-time tier keeps the cloud on the dome in the visible band and skips the infrared march
   to the hit: a target is occluded in neither band.
2. **Occlusion follows the render path.** One switch selects the tier for both bands, so they never
   disagree about occlusion. The infrared march to the hit costs a second pass over the frame, and
   the real-time tier saves exactly that.
3. **The cloud is never dropped.** In both tiers the infrared sky pixels are marched per pixel on
   the same field from the same origin, and are held to the same realism criteria (§7.5): an opaque
   cloud within 1 K of its level's temperature, edges as fringes, no sampling structure, the same
   cloud pixels as the visible band, and agreement with calibrated full-sky LWIR imagery.
4. **Realism is gated on real data inside the aerial phase.** `XD.6` (the ARM Infrared Cloud
   Imager) moves from the cross-cutting phase into phase A, and phase A's exit names the cloud.

## Consequences

* Until `AT.31` lands, the infrared band occludes in real-time renders where the visible band does
  not; that disagreement is now a known defect with a step, not an accepted state.
* A real-time frame of a target behind a cloud is wrong in the same way in both bands, which is
  the owner's stated trade and is what makes the pair comparable.
* The path-traced tier needs the path tracer and weather-fx's volume effect in the headless driver,
  which is `AT.31`'s in-engine half.

## Alternatives rejected

**Occlude in the infrared in both tiers.** More correct in the infrared alone, but the pair then
disagrees in every real-time frame with a target near cloud, and the pair is how results are read.

**Drop infrared clouds in the real-time tier.** Rejected by the owner: the cloud must appear and be
realistic in the infrared whatever the render mode.
