# ADR 0190 — The infrared march reads the cloudscape through the same deck, and a third cloud tier draws it in both bands

**Status:** Accepted
**Date:** 2026-10-05
Closes roadmap step `WX.26`. Builds on ADR 0178 (two cloud sources behind one contract), ADR 0185
(the cloud is a function marched per pixel) and ADR 0186 (it reaches the viewport on a quad).
`isaac-weather-fx` moves to `07c6615`.

## Context

The owner rejected the visible cloud ("a marshmallow in the sky") and it was rebuilt in
`isaac-weather-fx`: cumulus and towers grown by a fluid solver, placed by a weather map, marched
per camera pixel in RTX Real-Time and the path tracer alike, with the cloud in front of scene
surfaces drawn over them (`docs/weather/README.md`, the update at its top). That cloud is a
`Cloudscape`: a function of position. The infrared still marched the older `CloudField`, a grid
of cells, so the two bands showed different clouds. The owner's instruction for this step: the
visible look may change again, and the infrared must not have to be redone when it does.

## Options considered

1. **Derive the infrared from the visible frame** (cloud mask or brightness from the RGB image).
   It ties the band to the picture: every change to lighting, exposure or erosion would move the
   infrared, and a picture carries no height, so no temperature.
2. **A second infrared march written for the cloudscape.** Two marches to keep equal, and the
   sky model's radiometry would have to choose between them.
3. **The cloudscape behind the existing deck (chosen).** ADR 0178 already names what a cloud
   source must offer. `Cloudscape` now offers it upstream (`thickness_m`, `finest_pitch_m`,
   `optical_depth`, `slab_span`, beside the `density`, `extinction_per_m`, `base_m`, `top_m` it
   had), and `WeatherFxDeck` marches whichever source it is given.

## Decision

- `irsim.atmosphere.weather_fx.WeatherFxDeck` wraps a `CloudField` or a `Cloudscape`. Its march
  reads only: density in 0..1 at a point, the visible extinction at unit density, the layer's
  base, top and span along a ray, the finest pitch, and the measured cover. Nothing of the
  visible renderer (lighting, exposure, skin constants, the veil) is read. The radiometry is
  unchanged: emission at the temperature of the height the band's absorption comes from, the
  band's optical depth a fixed fraction of the visible one (ADR 0162), stage 2 composing the
  march to each pixel's hit.
- `irsim_isaac.weather_fx_stage` gains a third cloud tier, `pixel`, beside `path_traced` and
  `real_time` (ADR 0169). It sets weather-fx's render path to `pixel`, attaches its
  `CloudLayerEffect` for the infrared camera (redrawn on every app update), and returns a deck
  over the **same** `Cloudscape` object that layer marches. The tier occludes in both bands. The
  layer's two quads are companion-only prims: the G-buffer render does not see them.
- `cloudscape_from_spec` builds a cloudscape from plain numbers for engine-free callers.

## Consequences

- One cloud in both bands by construction: where the function has cloud, both see cloud.
  `tests/unit/test_cloudscape_in_the_infrared.py` holds the march to the function's own dense
  integral (cloud masks at IoU above 0.9, band emissivity within 0.05 at the 95th percentile) and
  shows that changing the visible skin constants leaves every ray's optical depth identical.
- A change to the visible look reaches the infrared only if it changes where the cloud is. New
  patches, new placement or new settings need no infrared work.
- The march is the CPU reference and it is slow on this source: 13 s for 20 000 rays, since
  every sample evaluates the function (several texture lookups) instead of reading a grid. A
  640 x 512 frame took about a minute and a half in a three-frame run. A GPU deck with the same
  interface is the follow-up; the visible side's Warp kernel already evaluates the same function
  and is tested against the numpy one.
- Not done: the agreement of cloud pixels between the two rendered frames was judged by eye on
  `outputs/phantom4_cloudscape/cumulus_towers/`, not measured as the IoU `WX.26` names; the
  default tier stays `path_traced`; the cloud's water content and temperature still come from
  height above the base, since the solver that grew the patches carries neither.
