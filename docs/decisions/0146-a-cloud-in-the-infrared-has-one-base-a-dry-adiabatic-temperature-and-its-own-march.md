# ADR 0146 — A cloud in the infrared has one base, a dry-adiabatic temperature, and the band's own march

**Status:** Accepted. Amends [ADR 0070](0070-cloud-clutter-model.md) (the base temperature law)
and [ADR 0130](0130-a-cumulus-has-a-size-and-a-shape-and-both-bands-march-it.md) (how the
infrared band integrates the shared field).
**Date:** 2026-09-26
Roadmap: AT.19.

## Context

The owner's judgement of the infrared clouds was that they are almost always brighter than they
should be, and smooth and blurred in a way no real LWIR footage shows. Both were measured on the
`phantom4_weather` clip and its frames.

**Brighter.** The clip's cloud read 17 °C in the apparent-temperature plane while the weather-fx
state that drew it said 12.9 °C air. Rebuilt on the CPU, the same field under the same state gives
an opaque cloud at **8.5 °C** (base 600 m, base temperature 9.0 °C) and a clear sky at −40 °C.
Under the scene's *CSV* weather instead — 26.3 °C air, RH 0.41 — the sky model puts irsim's own
lifting condensation level at **1792 m** and the cloud at **17.0 °C**, and the airframe node at
26.3 °C, which is the frame to the decimal. The clip was rendered on 23 September, one day before
`AT.16` made weather-fx's series the scene's weather; it is the two-weather defect ADR 0136
closed, never re-rendered. Two things underneath it are still wrong in today's code:

1. **Two bases.** `radiance_field_from_deck` took the emission heights from the deck's own base
   (600 m) but the base *temperature* and the air in front of it from a second LCL computed from
   the weather (1792 m in that run). Geometry and temperature agreed only while both came from
   one state.
2. **The wrong lapse to the base.** ADR 0070 lapsed the surface air to the base at the atmosphere
   preset's environmental rate, 6.5 K/km. The base is the LCL of *surface* air, which got there
   by cooling dry-adiabatically at g/c_p = 9.76 K/km; Espy's 125 m/K rule for the LCL is that
   rate minus the dew-point lapse, inverted, and at the base the parcel meets its dew point to
   0.1 K (a test now says so). The environmental rate overwarms the base by 3.3 K per kilometre —
   2 K at 600 m, 6 K at 1.8 km — always toward a brighter cloud. Inside the cloud the parcel is
   saturated and the moist rate, which 6.5 K/km is close to, stays.

**Smooth and blurred.** With one weather the cloud sits mid-grey against a black sky — the
physics — but the shapes are soft, and the 64-step march printed structure of its own: against a
512-step reference the frame differs by **3.6 K at the 99th percentile**, in concentric rings.
weather-fx's march offsets its samples by a jitter that is a *smooth* function of the ray
direction, so neighbouring pixels' errors share a sign (lag-one correlation of the residual along
a row: **0.84–0.90**). The remaining softness is the field itself: 60 m voxels read trilinearly,
30 % of cloudy pixels in the 0.1–0.9 emissivity fringe; that is the shared array and is not this
band's to change.

## Decision

1. **One base.** `SkyModel.radiance_field_from_deck` takes the base temperature and the slant
   path to the base at `deck.base_m`. `cloud_base_temperature_k` and `_cloud_path` accept a base;
   the LCL from the weather remains the default for the plane-parallel blend, and equals the
   deck's base whenever the deck was built from the same weather.
2. **Dry-adiabatic to the base.** `DRY_ADIABATIC_LAPSE_K_PER_M = 9.761e-3` and
   `DEW_POINT_LAPSE_K_PER_M = 1.8e-3` join `irsim.radiometry.constants` with their sources;
   `SkyModel.cloud_base_temperature_k` uses the former. The preset's lapse is kept for the
   emission level *inside* the cloud.
3. **The infrared band marches the shared array with its own quadrature.** `WeatherFxDeck.march`
   no longer calls weather-fx's march: it samples `CloudField.density` on a **uniform, stratified**
   path with **two samples per grid pitch** along the longest ray in the array (Nyquist for a
   trilinear field), each ray offset by a hash of its own direction, rays dropped once opaque.
   ADR 0130's "same march, same fidelity" becomes "same array; each band at its own quadrature",
   because the dome is baked once and read through a texel's blur while the infrared frame is
   read per pixel, and the two want different sampling.
4. **The plane-parallel blend carries the range too.** `SkyModel.radiance(θ)` -- the
   expectation over the field that the sea, the tilt LUTs and the uniform sky use -- was
   `(1 − cε) L_clear + cε L_B(T_base)`, the base read through no air at all. It is now ADR
   0126's `L_clear + cε τ(θ) (L_B(T_base) − L_beyond(θ))`, identical to the old form when the
   base is at the surface (τ = 1, `L_beyond` = the clear column) and different when it is not:
   a grazing ray no longer reads a 1.2 km base through seventy kilometres of air it cannot see
   through. The dry lapse made this visible -- a 1.2 km base under 288 K air is 276 K, colder
   than the clear horizon, and the range-free blend made an overcast *sea* widen its reflected
   profile instead of flattening it. `effective_radiance`, a tilt integral, keeps the flux form.

## What was measured

| | weather-fx march | this march |
|---|---|---|
| band-emissivity error, p99, real deck, 311 steps | 0.0025 | **0.0018** |
| same, test deck, 156 steps | 0.0068 | 0.0089 |
| lag-one correlation of the residual along a row | 0.84–0.90 | **0.39–0.56** |
| 640×512 frame, real deck, 18–44° elevation | 3.8 s at 64 steps | 10.1 s at 311 steps |
| opaque cumulus, 12.9 °C air, 600 m base | 8.5 °C (env. lapse: 9.0 °C base) | **7.1 °C** (dry: 7.1 °C base; the in-cloud cooling and the path warming balance) |
| a 1.5 km base at 273.5 K through 2.3 km of 288 K air at 40° | — | reads 276.6 K: the range term of ADR 0126, now visible |

Tests: `tests/unit/test_cloud_base_law.py` (the parcel meets its dew point at the base; the sky
model lapses dry-adiabatically; a marched deck is read at its own base) and the march tests in
`tests/unit/test_weather_fx_bridge.py` (agreement with a 2048-step reference; less banded than
what it replaces; the step count follows the longest crossing; the hash is not smooth).

## Consequences

* A cumulus reads about 2 K colder than before at a typical base, more for a high one, and the
  sky, the surfaces and the cloud can no longer be integrated from two weathers by construction.
* The infrared march costs about 2.5× what it did per frame, spent where the picture is read.
* **What this does not fix:** the shapes. The blobby, soft cloud is the 60 m grid, shared by both
  bands, and sharpening it means detail at sample time in weather-fx's field — the item its own
  roadmap lists as "a cheaper march so a finer grid is affordable". That is the next step, and it
  is a change to the shared array, not to either band.
* Under an overcast sky a sea seen from 20 m reflects the near-horizon sky, which now reads
  within a few kelvin of the air rather than at the base, so the reflected profile flattens to
  what the emissivity alone leaves; `test_sea_surface` states the new fraction and why.
* `test_cloud_clutter` pinned ADR 0070's 6.5 K/km and now pins g/c_p; its "uniform at every
  elevation" claim is made where it is exact (a saturated base at zero range) and its lifted
  base is asserted to read between the base and the air, warmer toward the horizon. The
  altostratus anchor in
  `test_cloud_optical_depth` (a 2.4 km base in a standard atmosphere, 271.6 K) is a stratiform
  cloud whose base is *not* the LCL of surface air and keeps the explicit environmental lapse it
  was always evaluated with.
* The `phantom4_weather` clip is re-rendered on today's code, so the site stops showing the
  two-weather frame.

## Alternatives rejected

**Patch weather-fx's jitter instead.** It is the right change for that project too, but the two
bands want different quadratures anyway, and a submodule pin is a heavier dependency for a
per-frame accuracy question than a forty-line march over an array the seam already exposes.

**Keep the environmental lapse and call 2 K within the envelope.** It is not within a 50 mK NETD,
it is systematic, and it is in the direction the owner sees.
