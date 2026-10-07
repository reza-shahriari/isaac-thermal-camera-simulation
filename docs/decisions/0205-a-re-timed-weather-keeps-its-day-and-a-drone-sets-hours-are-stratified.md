# ADR 0205 — A re-timed weather keeps its day, and a drone set's hours are stratified

**Status:** Accepted (EV.24)
**Date:** 2026-10-07

## Context

The first rendered clip set (`EV.19`) had two faults, both found in its own logs.

- **Five of its six clips were at night.** Each clip's hour was drawn uniformly, and this draw
  landed 22:12–22:57 four times and 02:06 once.
- **The air was 22.0 °C at 09:45 and at 22:23.**
  - weather-fx's random draw left the clear regime's air at the 22 °C default; it used its own
    seasonal temperature only for cloudy regimes.
  - The draw placed that temperature at a random hour. irsim then moved the state's clock to the
    clip's hour with a bare `with_updates("sky", hour_utc=…)`.
  - `diurnal_series` anchors the day on the state's temperature at the state's hour. So a 22:00
    clip got a day shifted up to put the afternoon's air at 22:00. A night clip's drone and sky
    differed from a day clip's by the sun alone.

## Decision

In weather-fx (`d0d55ff`):

- **The seasonal draw is the day's mean.** It is placed on the day's curve at the drawn local
  hour: `T = mean + ½·swing·sin(2π(h − 14.5 + 6)/24)`. The swing is 11 K on a clear day, damped
  by cloud and fog (`diurnal_swing_k`). This applies to the clear, cumulus, overcast and haze
  regimes. Rain, storm, snow and fog keep their saturated air as drawn. No random draw is added,
  so every other value a seed gives is unchanged.
- **`move_clock(state, **sky)` re-times a state and carries the air along the curve**, keeping
  the day's mean. The dew point stays, except in saturated air (rain, snow, fog under 1 km). There
  it moves with the air, so the spread that defines the weather is kept.

In irsim:

- `weather_state` applies the site and clock through `move_clock`.
- `generate_aerial_dataset.plan_runs` stratifies the hours in the scene's local solar time. Clip
  *i* of *n* falls in its own 24/*n*-hour slot, the slots in an order shuffled on a stream of its
  own. The position inside the slot is the old uniform draw, so the weather and seeds a set drew
  before are unchanged. Each run records `local_solar_hour` as well as the UTC `weather_hour`.

## Consequences

- The same clear day over the Phantom 4 scene (48.1° N, 11.6° E, seed 5) reads 19.0 °C at
  02:00 UTC and 29.9 °C at 13:00. The dew point is 16.0 °C at both.
- Six clips cover the day in four-hour slots. The default plan (seed 20261004) is at 0.4, 7.8,
  9.6, 15.7, 19.8 and 23.7 h local.
- Every render that draws a weather-fx regime and films it at another hour now has different air
  and a different cloud base: the LCL follows the spread. This includes the cloud clips in the
  gallery. Their published numbers came from their own runs and stay true of those runs. A re-run
  gives the corrected air.
- The diurnal model is weather-fx's sinusoid with an ESTIMATED 11 K clear swing. A site with a
  measured series should use it (`SceneSpec.weather_file`).

## Revisit when

- The real set's night clips can be told from its day clips (the burnt-in readout's clock). Then
  match the set's own day/night share, not a uniform day.
