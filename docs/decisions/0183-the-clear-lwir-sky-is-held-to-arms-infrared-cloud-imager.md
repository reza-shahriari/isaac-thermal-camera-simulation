# 0183 — The clear LWIR sky is held to ARM's Infrared Cloud Imager

Date: 2026-10-03

**Status:** Accepted (2026-10-03, XD.6 part 1). Measures ADR 0071's layered sky; opens AT.37.

## Context

The layered clear sky (ADR 0071) was calibrated on one anchor: a dry Tucson sky at −40 °C at
15° elevation. Every later check of it was against other models. ADR 0169 made real calibrated
sky imagery the realism gate of the aerial phase (`XD.6`).

ARM's Infrared Cloud Imager deployment at the Southern Great Plains site (NWB Sensors, May–Dec
2023, doi:10.5439/3001561) is that imagery. The instrument is an upward-looking 7.3–14 µm
microbolometer. It records calibrated sky radiance in W/(m²·sr) per pixel, a cloud mask, and a
per-pixel azimuth and elevation map. Its intercomparison report holds it to ARM's AERI to
0.17 W/(m²·sr).

The owner fetched 92 images over five nights. Two things the comparison needs are not in those
files:

1. **The surface met and PWV.** The companion datastreams the owner ordered arrived empty, and
   for other dates. The instrument's processing prints both quantities on every frame of the
   deployment time-lapse.
2. **The camera's spectral response.** The report calls the unpublished response shape the main
   limitation of its own calibration.

## Decision

1. **Met is read off the time-lapse.** `scripts/ici_met_from_video.py` reads the title with a
   glyph-template reader trained on seven frames transcribed by eye. It finds the frame nearest
   each image, at most 60 s away. This is the instrument provider's own met, which their
   clear-sky model ran on. The values change smoothly minute to minute, as misread digits would
   not.
2. **Only a derived file is committed.** `data/validation/ici_sgp2023_clear_sky.csv` (15 kB)
   holds, per image: the clear-pixel median radiance in eight elevation bands, the cloud
   fraction, the met, and the instrument's own modelled clear sky in the same bands. The raw
   data stays outside git (2.3 MB per image, ARM terms: no use constraints, cite the DOI).
3. **The weather is the surface sample, and the column is matched.** The surface humidity is
   chosen so that `w0 · H_w` equals the measured PWV, with `H_w` the preset's water-vapour
   scale height. Where that needs RH > 1, the preset's `H_w` is too shallow; the RH is clamped
   and flagged.
4. **The band is a stand-in, and that is stated.** The comparison uses the ESTIMATED VOx
   response. A band change scales every sky, roughly alike. A misfit that grows with humidity
   cannot come from it, and the test shows the December result moves by under 7 % across the
   band choices.

## Consequences

The comparison over the 30 clear images, under the VOx stand-in, as model/measured. The ICI's
own model is shown beside it for contrast.

| Night | Surface T, PWV | irsim layered sky | ICI's own model |
|---|---|---|---|
| 11 Dec | 2.5 °C, 0.86 cm | 1.00–1.07 | 0.89–0.93 |
| 21 May | 20.8 °C, 1.41 cm | 1.03 (10°) to 1.18 (zenith) | 0.97–1.07 |
| 6 Aug | 27.0 °C, 3.96 cm | 1.24–1.28 | 1.01–1.03 |

- The dry sky is right; the humid sky is up to 28 % too bright. Splitting the band by spectral
  class shows where: from December to August the water-line class (7.8–8.3 and 12.5–13.2 µm)
  rises from emissivity 0.41 to 0.97, and the window class from 0.047 to 0.31. The humid
  window and lines absorb too much. AT.27's squared self-continuum term is the first suspect.
  With the dry anchor held, dropping it entirely (the pre-AT.27 line) brings August to
  1.06–1.10 and leaves December alone; but 21 May's zenith stays at 1.23 under every
  coefficient tried. So part of the misfit is the vertical profile — the column depth implied
  by the preset's `H_w` = 2 km, and the evening inversion — not the continuum. Fitting the
  continuum alone would bake a profile error into it, which ADR 0160 warned against.
- Re-fitting is **AT.37**, not this step. It moves every LWIR golden, and it first needs each
  night's measured surface dew point (ARM `sgpmetE13.b1` for those five dates) to pin the
  column depth independently of the PWV.
- `tests/unit/test_ici_clear_sky.py` holds the December night to 7 % at 20°–90°. It records the
  August night as a strict xfail that names AT.37, so the fix has to remove the marker.
- §15's radiometric Tier 4 target, which ADR 0068 reported as untestable, now has a radiometric
  reference for the sky.
- The cloudy images (ADR 0070's cloud clutter and the cloud-base radiance) are in the same
  files. They are XD.6's second part.
