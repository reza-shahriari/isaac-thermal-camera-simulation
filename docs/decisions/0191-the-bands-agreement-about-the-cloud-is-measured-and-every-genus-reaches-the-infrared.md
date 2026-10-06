# ADR 0191 — The bands' agreement about the cloud is measured per frame, every genus reaches the infrared, and the march runs on the GPU

**Status:** Accepted
**Date:** 2026-10-06
Completes roadmap step `WX.26` (ADR 0190 shipped it with its bar unmeasured). `isaac-weather-fx`
moves to `bd77d3d`. Spec issue `S67`.

## Context

ADR 0190 put the visible cloudscape behind the infrared deck and listed what it had not done:
the agreement of cloud pixels between the two rendered frames was judged by eye, not measured
as the IoU the step names; only cumulus with towers had been rendered in the infrared although
the state offers six genera; the CPU march cost about twenty minutes a 640 × 512 frame. The
owner's response: a judgement by eye is not acceptable where a number can be had; every kind
of cloud, including any added later, must appear in the infrared viewport; and the cloud's
temperature coming from height alone is fine while it does not damage the frame's accuracy.

Two defects turned up on the way. weather-fx's `clouds.density_scale` multiplied the visible
march's extinction on a private copy of the layer and never reached the deck, so the two bands
could be made to march clouds of different depth by a look knob — the very thing
`docs/weather/README.md` §7 forbids. And a cirrus base set at 9 km was lapsed dry-adiabatically
from the surface, 30 K colder than the air there, because §7.5's law is written for a cloud
standing on the condensation level of surface air.

## Options considered

1. **Compare the two rendered pictures** (a cloud mask from the RGB frame against the infrared
   frame). The RGB picture carries lighting, exposure, the veil and the sky's own colour; a mask
   from it would measure the tone mapping as much as the cloud, and could not be held to a
   physical tolerance.
2. **Compare the two marches' transmittance, pixel by pixel (chosen).** The visible per-pixel
   layer already integrates the same density to a transmittance per pixel; so does the infrared
   deck. Related by the one number that links the bands, `τ_B = r_B τ_vis`, they are the same
   physical quantity and can be scored in emissivity.
3. **For cirrus, keep the older grid field in the pixel tier.** The two bands would then draw
   different clouds for one genus, which is what the tier exists to prevent.
4. **For speed, coarsen the CPU march (`cloud_stride`).** It only moves the cost; edges are
   re-marched at full pitch either way, and a cumulus sky is mostly edges at a drone's range.

## Decision

- **The score.** `irsim.validation.cloud_bands.cloud_band_agreement` takes the infrared march's
  band transmittance and the visible march's transmittance on one grid, converts the visible to
  the band's emissivity through `1 − T_vis^{r_B}`, and reports the IoU of the two cloud masks at
  emissivity 0.5, the 95th-percentile and mean absolute emissivity error over the pixels either
  band put cloud on, and their correlation. Below the visible march's own floor (0.004) the
  visible says only "opaque", and an infrared emissivity at or above what the floor implies is
  agreement. Rays shallower than `asin(thickness / max_path)` are left out: the deck ends a ray
  after its in-layer path cap and the visible march does not, so there the infrared stopped short
  of cloud the visible went on to see — a difference of caps, not of cloud. The bar is IoU ≥ 0.9
  (the roadmap step's own number), judged where either band has at least 1 % of cloud — a wisp
  of a few hundred pixels swings the ratio with a one-pixel shift — and p95 ≤ 0.05 in band
  emissivity on every frame, the engine-free test's own tolerance.
- **The plane.** weather-fx's `CloudRenderer` keeps each frame's transmittance;
  `CloudLayerEffect.transmittance()` hands it to the host for the camera it follows; the
  infrared camera stores it as the `cloud_transmittance_vis` plane beside its own
  `cloud_transmittance`, and `render_phantom4.py` scores every frame into its rows and summary.
  `scripts/cloud_band_agreement.py` rescores a saved directory with no engine. The layer is drawn
  at the camera's full render resolution in the pixel tier (`layer_scale` 1), so the two planes
  share a grid.
- **Every genus.** The cloudscape gains a cirrus profile (ice, base 9 km by default, cover read
  against a 0.3 column since it is thin, stretched 6:1 along the field's x axis in the weather
  map and shape noise — kernel and numpy alike); `supports()` is now true for every genus the
  state offers, and the pixel tier takes them all. The deck reads cirrus as it reads the rest.
- **The temperature law for a cloud above the mixed layer.** §7.5 is amended (`S67`): dry
  adiabat to the LCL, the environment's lapse above it. The cloud source says which case it is
  — a non-convective cloudscape carries `mixed_layer_top_m`, a convective one `None` — and
  `SkyModel.cloud_base_temperature_k(mixed_layer_top_m=)` applies it in both the sky march and
  the march to the hit. Nothing changes for a cumulus, whose base is that level.
- **The density scale reaches the deck.** `WeatherFxDeck.density_scale`, set by
  `author_weather_fx_sky` from the state, multiplies the field's extinction in both marches and
  in the column depth.
- **The capture draws the visible layer converged.** The first in-engine measurement scored
  IoU 0.63–0.95 and p95 up to 0.60 and would not move with the layer's frame accumulation.
  Marching single rays three ways found the CPU deck, the GPU deck and the visible kernel
  agreeing on the cloud but the kernel's *viewport* settings far from the function: inside thin
  cloud it stretches a step to 0.8 of optical depth and takes the local density to hold over it
  ("a wisp is sampled, not resolved"), and its clear-air step grows 1.2 % per metre of range, so
  a 10 m wisp at 5 km is sampled once or skipped. Those are now weather-fx state parameters, the
  viewport keeps its speed, and a headless pixel-tier capture sets `layer_thin_tau` 0.02,
  `layer_step_growth` 0.001, `layer_max_steps` 8192, one frame, full resolution
  (`PIXEL_LAYER_SETTINGS`), at 60 ms for 640 × 512.
- **The infrared marches the layer as a shell round the planet, to the visible kernel's range.**
  With the visible converged the residual still sat at p95 0.06–0.08 on cumulus and 0.93 on
  cirrus at 10°. Both are geometry. The deck marched a flat slab; the visible kernel a shell
  (density at the altitude above the curved ground, entry and exit on spheres). The two agree to
  metres within 12 km and differ by 1.2 km on a cirrus base at 9 km seen at 10°, where the ray
  enters at 50 km — so the bands marched different cirrus. And the deck marched any cloud a ray
  reached, the kernel nothing beyond 80 km; at 0.7–4° a cirrus shell is 110–290 km away, so the
  infrared showed 34 % of a frame as cloud against the visible's 19 %. `WeatherFxDeck` now takes
  the altitude from the sphere (`altitude_m`, `shell_span`, weather-fx's own radius) and ends
  every ray at `max_range_m` = 80 km, in the CPU reference and the GPU twin alike; the in-layer
  path cap stays (12 km on the CPU, 40 km on the GPU in the pixel tier, with 4096 steps). After
  this the two integrators agree engine-free to p95 0.019 / 0.026 (cumulus, 10° / 30°) and
  0.025 / 0.012 (cirrus) at growth 0.003, and 0.008 / 0.005 at 0.001 — inside the engine-free
  test's 0.05, which therefore stays the bar.
- **The march on the GPU.** `irsim_isaac.cloud_march_gpu.GpuWeatherFxDeck` is the CPU deck's
  twin: the same slab span, step count, hashed stratified jitter, Schwarzschild sum, drop-out
  and emission-height weighting, with the density from weather-fx's own Warp `cloud_density`.
  The CPU deck stays the oracle (`tests/integration/test_cloud_march_gpu.py`); the stage uses
  the twin whenever Warp and a CUDA device are present and `--cpu-cloud-march` keeps the
  reference. This is a speed step taken after the CPU path was verified, as the project's rule
  on GPU work allows.
- **The cloud's temperature from height alone stays.** A cumulus base sits near the surface dew
  point and drifts by a few kelvin over a day as the dew point does; inside the cloud the moist
  adiabat holds to about a kelvin against entrainment. That is within §7.5 criterion 1's 1 K,
  so a solver-carried temperature (WX.8) is not needed for the frame's accuracy, only for a
  source that measures its own (LES).

## Consequences

- Measured, engine-free: the score is 1.0 / 0.0 on identical planes, falls on a shifted mask,
  masks and refuses mismatched shapes; the density scale doubles every ray's optical depth in
  both marches; a cirrus field reaches the deck at 9 km with a median emissivity under 0.9 on
  cloudy rays; a 9 km base reads the environment's temperature, 20–35 K warmer than the dry
  parcel's.
- Measured on the GPU: the twin agrees with the CPU deck to p99 < 0.02 in band emissivity on a
  fan of 4 608 rays, path radiance within 3 % at p95 on cloudy rays, emission height within 25 m
  and emission range within 40 m at p95; a full 640 × 512 frame of rays marches in 0.04 s at 512
  steps and 0.2 s at 2048, where the CPU took about 130 s before edge refinement. On single
  rays the CPU deck, the GPU deck and the visible kernel agree to three decimals.
- Measured engine-free on the same cloudscape and camera, before the renders: the visible
  layer at viewport settings against the infrared march is p95 0.14–0.26 (IoU 0.92–0.95);
  at the capture settings with the flat slab 0.06–0.08; with one shell and one range IoU
  0.996–0.998 and p95 0.007–0.019 (cumulus) and 0.975–0.998, 0.001–0.025 (cirrus) at 10°.
- Measured in the engine, six genera, both bands, the Phantom 4 lemniscate (numbers in the
  roadmap row and `TECHNICAL_REPORT.md`): see the gallery's cloud-genera entry.
- The score measures the two integrators against each other, not either against the world;
  the comparison with calibrated cloudy LWIR imagery (`XD.6` part 2) is still the only check
  of the cloud's radiance.
- The residual inside the bar is the two integrators' sampling of 10 m voxels, not a
  disagreement about the cloud; the viewport's own layer settings remain approximate on thin
  edges by design, and a reader comparing a viewport screenshot with an infrared frame should
  expect that.
- Measured in the engine on the six clips (`scripts/cloud_band_agreement.py` on the saved
  planes, re-rendered 2026-10-07 after the amendment below, the depth read active in all six):
  IoU min 0.852 (median 0.994), p95 band-emissivity error max 0.096 (median 0.010), mean
  error under 0.024 on every frame; 132 of 144 frames inside the bar. The twelve outside are
  eleven cumulus and congestus frames with under 24 % cloud, where the edge fringe sets the
  95th percentile, and one cirrus frame with 1.5 % cloud whose IoU is 0.883 at a p95 of 0.007
  and a correlation of 0.9996 -- thin cirrus straddling the ε = 0.5 cut, not a disagreement.
  On the drone's 30,494 pixels across the 144 frames, no pixel reads an infrared cloud
  transmittance under 0.5 (up to 70 a frame before the fix) and the visible plane reads 1.0.
- **Amended 2026-10-07: the drone pixels.** The first scoring of these clips (IoU median 0.991,
  132 of 144 frames inside the bar) compared unlike quantities on the drone's pixels, and this
  ADR first read that as the visible companion failing to hide the drone behind cloud. It was
  not: the drone flies 5–10 m away, under a base at 910 m, so no cloud is ever in front of it,
  and the RGB picture correctly draws it over the cloud. The visible plane was the layer's
  whole-ray transmittance, which on an object's pixels is the cloud *behind* the object. And
  the infrared had a real defect there: the coarse-then-interpolate march (ADR 0181) spread the
  sky's cloud across the silhouette onto up to 70 drone pixels per frame, since a uniform cloud
  behind the drone flagged no edge. Two fixes: weather-fx's layer now hands over the
  transmittance to the first surface where it read the scene's depth (`surface_transmittance`,
  and `depth` in its stats), the camera leaves object pixels out when it did not, and
  `march_on_native_grid` marches every sample of a pixel whose samples do not share one range
  to the hit (`tests/unit/test_pixel_integral.py`, red on the old march).
- The sky model's clear-air path to a cloud still takes the base height and a plane-parallel
  airmass (ADR 0126); only the cloud's geometry is on the sphere. Beyond 80 km the band's air is
  opaque in any case, which is why the range cap costs the infrared nothing it could show.
- A numpy dense integral of the cloudscape disagrees with all three marches on a few far rays
  (up to 0.15 in emissivity): the GPU texture's density and the numpy density differ locally
  there, within upstream's own p99 < 0.02 density bound over random points. The marches are
  held to each other and to the CPU deck, which is the project's oracle.
- Cirrus has no shadow and no ice-specific band ratio: 0.5 is the large-particle limit for
  60 µm crystals as for 10 µm droplets, which is defensible in the 8–14 µm window and not
  measured.

## Revisit when

`XD.6` part 2 compares the cloud radiance with the Infrared Cloud Imager's cloudy images; a
source carries its own temperature (LES, `WX.8`); a band other than LWIR renders cirrus, where
ice's absorption bands would want a ratio of their own.
