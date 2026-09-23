# ADR 0136 — One weather state owns the sky *and* the surfaces, and a CSV is the escape hatch

**Status:** Accepted
**Date:** 2026-09-24
Completes the move begun in `edbf711` (the sky to `isaac-weather-fx`). Enforces CLAUDE.md
non-negotiable #6. Extends [ADR 0032](0032-weather-series-format-and-injection.md) (the weather
series and how it is injected). Related:
[ADR 0037](0037-solve-granularity-and-spin-up.md) (the spin-up window) and
[ADR 0130](0130-a-cumulus-has-a-size-and-a-shape-and-both-bands-march-it.md) (one cloud, both
bands). Roadmap: `AT.16`.

## Context

The sky moved to `isaac-weather-fx`: the sun, the moon, the turbidity and the cloud field that
both bands march are drawn from one `WeatherState` there. The thermal solver did not move. It
still read `SceneSpec.weather_file`, a measured 48-hour CSV from `data/weather/`.

So `render_phantom4.py --weather broken_cumulus` produced a frame lit by weather-fx's cloudy June
morning whose **surfaces** had been integrated through a different day out of a file — different
air temperature, different humidity, different irradiance, quite possibly a different season. Both
halves were internally consistent. Nothing in the pipeline can detect it, because neither half is
wrong on its own; the image is simply of a scene that does not exist.

That is precisely the failure CLAUDE.md #6 exists to prevent, and the mechanism it prescribes —
*one `WeatherSeries`, injected into both* — was already in place and being satisfied by the letter.
The single object reached every consumer. It just no longer described the sky the camera saw.

A second problem sat behind it. The CSV is a **measurement**: one recorded day at one place. It is
the right input for a validation run and the wrong input for a dataset, where the point is
coverage of conditions. Generating that coverage meant writing CSVs, and a CSV that was generated
rather than measured looks exactly like one that was measured.

## Options considered

**A. Keep the CSV authoritative and drive weather-fx from it.** Read the file, and set the
weather-fx state's temperature, dew point, cover and turbidity from the sample at the scene start.
Preserves every existing scene unchanged. Rejected: it inverts the ownership the rest of the
subsystem now has — weather-fx knows what a cloud base is, what a regime is, how rain and
turbidity couple — and the CSV carries no cloud genus, no site, no lunar phase, so most of what a
sky needs would still have to come from somewhere else. Two authorities again, in the other order.

**B. Delete `weather_file` and make weather-fx the only source.** Rejected outright: there is then
no path for measured weather at all, and Tier 4 validation against public data is the project's
only contact with reality.

**C. weather-fx owns the physical weather; the CSV stays as an explicit escape hatch.** Chosen.

## Decision

1. **weather-fx synthesises the series.** `weather_fx.core.meteorology.diurnal_series` turns one
   state into surface meteorology on a regular grid. The physics lives *there*, with its own
   tests, because that package owns what the air is doing.

2. **This project wraps, and validates.** `irsim.thermal.weather_fx_series.weather_series_from_state`
   does one thing: it puts the eight columns into a `WeatherSeries`. That is not ceremony — the
   `WeatherSeries` is where the ranges are checked, where the arrays become immutable float64, and
   where the content hash that keys every spin-up cache and golden comes from. A foreign array
   that entered the solver without passing through it would silently invalidate all of them.

3. **The injection point is the one that already existed.** The result is passed as
   `Scene.from_config(weather_override=...)`, the parameter written for sensitivity studies. No
   second path through the scene builder, and the one-weather guard is unchanged.

4. **`weather_file` stays, and stays the default for a scene config.** A scene that names a file
   still gets that file. Only a driver that has drawn a weather-fx state synthesises, and
   `--weather-csv` turns even that off, keeping the measured file for the solve while weather-fx
   still draws the sky. Both halves of a validation run therefore remain reachable.

5. **The state's clock is the anchor, and the scene's clock follows it.** The synthesised air
   temperature passes through the state's own value at the state's own hour, with a day of history
   in front of it (ADR 0037's window). When `--weather-hour` moves the state, the scene's
   `start_utc` moves with it; otherwise the sky would be lit for one hour and the surfaces
   integrated to another.

6. **The cloud fraction recorded is the one that was built, not the one that was asked for.** A
   thresholded noise field covers what it covers. The driver passes `WeatherFxDeck.cover`, so the
   sky emissivity the solver uses is the cover the camera is actually looking at.

7. **What is diurnal varies; what the state cannot justify varying is held constant and said so.**
   Air temperature, humidity and irradiance vary. Wind, cover, visibility and precipitation are
   constant across the series, because a single state carries one of each and inventing a diurnal
   wind would be fabrication dressed as physics. The module docstring carries that table.

## Consequences

* A scene rendered with `--weather` now has one weather in both senses: the sun in the frame and
  the sun that warmed the paint are the same sun.
* A dataset can be generated from integers. `random_state(seed)` gives a regime-coherent day, and
  the same seed gives the same `content_hash`, so spin-up caches and goldens survive.
* **Synthetic weather is not measured weather** and the distinction now has to be maintained by
  the person reading the output, not by the file format. A run's `summary.json` records the full
  weather-fx state, which is what makes a synthesised day reproducible and auditable; a reader who
  sees no `weather_fx` block is looking at a measured CSV.
* The diurnal model is crude on purpose: one sinusoid, one lag, a dew-point floor. It is not a
  boundary-layer model and will not reproduce a frontal passage, a sea breeze or a nocturnal jet.
  When a scene needs one of those, it needs a measured file.
* The swing, the turbidity-to-visibility curve and the regime weights are **estimated**. They are
  marked as such in weather-fx and are the first thing to replace when a real climatology for a
  site is available.
* `Scene.from_file` gained a `weather_override` passthrough so a driver does not have to load the
  config itself just to reach the injection point.
