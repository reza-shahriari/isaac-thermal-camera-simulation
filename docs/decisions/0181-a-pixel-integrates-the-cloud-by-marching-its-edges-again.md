# ADR 0181 — A pixel integrates the cloud by marching its edges again, not by blurring the cloud

**Status:** Accepted
**Date:** 2026-10-03
Implements `docs/physics-model.md` §7.5 ("the pixel integrates radiance") and the sampling rule of §7.6.
Roadmap: `WX.4`. The visible half is in `isaac-weather-fx` (`5853dcc`); the infrared half is
`irsim.atmosphere.cloud_deck.march_on_native_grid`. This **amends the plan WX.4 was written with**:
"detail filtered to the pixel".

## Context

A broken-cumulus field shows static toward the horizon in both bands: each pixel reads cloud or
sky by the luck of one ray. Measured on 35 % cumulus:

* **The visible camera**, rendered at one ray per pixel against a 4 × 4 supersampled reference, was
  26.6 % rms off in a band from 0.3° to 9.7°.
* **The infrared camera**, at a Boson's IFOV and marched once per native pixel as it always has been,
  was 0.0094 rms off in emissivity at 30°, against a march of every supersampled sample.

The roadmap's plan was to band-limit the field's density to each pixel's footprint: fade the 15 m
detail, and widen the edge ramp.

## Options considered

1. **Filter the density to the footprint (the plan).** Measured, it does not work.
   * With the footprint isotropic, the static went from 26.6 % to 28.1 %.
   * With the footprint projected onto the slab (×1/sin e, which near the horizon is the honest
     footprint), it went to 46.5 %, with a −17 % bias.

   Opacity is not linear in density. A half-covered pixel is half cloud and half sky; a half-dense
   cloud over the whole pixel, on a kilometres-long grazing ray, is still opaque.
2. **More march steps.** At 512 steps against 64 the static did not fall at 5°. It is spatial, not
   quadrature: steps help at 35° (19 % → 12 %), not at the horizon.
3. **Supersample every pixel.** Correct, but 4 × 4 is 16 times the rays everywhere.
4. **Supersample only where an edge crosses the pixel** (chosen). This is classic adaptive
   antialiasing: march once per pixel, flag the pixels whose value differs from a neighbour's, and
   march those again through several rays.

## Decision

Option 4, in each band's own pixel.

* **Infrared** (`march_on_native_grid`, used by `AerialThermalBridge._deck_temperature` and
  `cloud_occlusion`):
  * One ray per native pixel, interpolated as before.
  * A pixel is an edge where it differs from a neighbour by 0.5 K in sky temperature, or by 0.02 in
    the transmittance in front of a hit; the mask is dilated by one.
  * Edge pixels are marched again, at every sample below 15°, at one ray per 2 × 2 block through
    its centre above. Ranges are never averaged: they are each sample's own.
* **Visible dome** (`sky._integrate_cloud_edges`):
  * A texel is an edge where its transmittance differs from a neighbour's by 0.02.
  * Edge texels get `clouds.dome_edge_rays` rays per side, default 2, and twice that below 15°.
  * Transmittance and added light are averaged separately, so the composite stays premultiplied.

## Consequences

| | static energy, below 10° | above | rays |
|---|---|---|---|
| infrared, 5° / 30° (`tests/unit/test_pixel_integral.py`) | 0.1 % of one ray's | 0.5 % | 1.09× / 1.87× |
| visible dome, 256 rows | 9.1 % | about 12 % | 4.2× |

* **Infrared** stays within the roadmap's 2× budget. An edge pixel near the horizon now reads the
  fraction of it that is cloud, as a detector's box filter does.
* **The dome bake** costs 4.2 times the rays: 8.0 s at 1024 rows, up from 4.1 s. Upstream re-bakes as
  the clouds drift, so `dome_edge_rays = 1` turns it off for an interactive session that prefers
  the faster refresh.
* **The edge test is a threshold.** An edge with less contrast than 0.5 K, or 0.02 of
  transmittance, is still sampled once. That bounds the static that is left.
* **The field itself is unchanged.** Its 15 m detail is marched in full everywhere, so a near cloud
  keeps every metre of it.

## Revisit when

* The dome's resolution rises (`WX.5`) and the edge texels make the bake too slow. The integration
  could then run on the dome's coarser rows only.
* A renderer offers per-pixel accumulation for the dome. The path tracer already converges the
  volumes that way.
