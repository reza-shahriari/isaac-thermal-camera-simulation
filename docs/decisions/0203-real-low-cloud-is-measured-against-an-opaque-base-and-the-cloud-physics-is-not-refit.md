# ADR 0203 — Real low cloud is measured against an opaque base, and the cloud physics is not refit

**Status:** Accepted (XD.6 part 2). Extends ADR 0183.
**Date:** 2026-10-07

## Context

Renders of low cumulus looked brighter to the owner than clouds do in real LWIR cameras. ADR 0183
held the clear sky to ARM's calibrated Infrared Cloud Imager (ICI) at the Southern Great Plains
site, and ADR 0200 brought the humid night to 10 %. So the clear air the cloud is seen through is
measured. The open question was the cloud itself.

The 92 ICI NetCDF images of part 1 hold almost no low cloud; their cloudy ones are mostly cirrus.
The deployment's time-lapse does hold low cloud. Its frames draw calibrated radiance on a fixed
−5…35 W/(m²·sr) colour scale. Inverting the colour reproduces the NetCDF files to a median
0.13–0.24 W/(m²·sr) where both exist (`scripts/ici_timelapse_clouds.py`).

What a cloud should read depends on its base height and that height's air temperature. The ICI
has no ceilometer. Three free sources supply them:

- ERA5 picks hours whose cloud is low only (Open-Meteo archive).
- The ceilometers at Ponca City and Enid Woodring (ASOS, Iowa Mesonet, each about 37 km away)
  give the base. A minute keeps a base only when both report within 40 min and agree within 500 m.
- The reanalysis pressure-level profile gives the air temperature at that base
  (`scripts/ici_low_cloud_bases.py`).

The result is `data/validation/ici_sgp2023_low_cloud.csv`: 461 low-cloud minutes from June to
December 2023, 200 of them with an agreed base.

## Options considered

1. **Compare the rendered cloud with the measured one directly.** The renderer's cloud has its own
   base (a lifted parcel), cover and optical depth. A mismatch would not say which was wrong.
2. **Compare with the brightest a cloud at the observed base can be.** That is an opaque cloud at
   the base's air temperature, seen through the pinned clear column
   (`irsim.validation.ici.model_opaque_cloud`, the identity behind the sky model's own cloud blend,
   ADR 0126). Opaque cores at that temperature would reach it. The test is whether the brightest
   real pixels (the 99th percentile of the cloudy ones) do.
3. **Refit the cloud's emissivity or temperature to the measurement now.**

## Decision

Option 2, recorded as a measurement; option 3 is declined for now.

The scale saturates at 34.5 W/(m²·sr), −0.5 °C under the VOx stand-in band. So the comparison
reads saturation, not values. Cases are taken at 30° elevation and above, in minutes at least 10 %
cloudy. They are binned by how far the model's opaque base lies above the ceiling:

| Opaque base above the ceiling | Cases | Brightest real pixels saturate |
|---|---|---|
| 0–5 K | 272 | 0 % |
| 5–10 K | 218 | 0 % |
| 10–15 K | 163 | 28 % |
| 15–20 K | 265 | 76 % |
| 20 K and more | 23 | 83 % |

If real cores were opaque at the base's air temperature, the first two rows would all saturate.
None of 490 does. Saturation appears only once the model is more than 10 K above the ceiling.
Under this band, then, real low-cloud cores read about 10 K colder than an opaque cloud at the
ceilometer height's air.

## Consequences

- `tests/unit/test_ici_low_cloud.py` pins the derived file and this measurement (the 0 of 490;
  40–80 % beyond 10 K).
- The owner's "too bright" is consistent with it. A renderer whose cloud is opaque at its base's
  air temperature is at the bright edge of what the instrument saw.
- The cloud physics is **not** changed. Three things would each move the answer by several kelvin:
  - the ICI's spectral response is unpublished (ADR 0183); across plausible 8–14 µm band
    choices, the same cloud's brightness temperature moves by 6–8 K;
  - the ceilometers are 37 km away;
  - the hourly reanalysis puts the temperature at the base height but not inside the cloud.
  Refitting against an uncertainty that size would bake the band error into the cloud, as
  ADR 0160 warned for the continuum.
- The time-lapse frames and airport reports are not committed (`datasets/arm_icii23/`, ignored).
  The CSV is derived, and both scripts regenerate it.

## Revisit when

- ARM's AERI (`sgpaerich1C1`) spectra at these minutes are available. They give the cloud's
  radiance spectrally, so no band stand-in is needed. The site's own ceilometer (`sgpceilC1`) gives
  the base without the 37 km gap. Together they would decide whether the cloud's emissivity, its
  temperature or the band is off.
