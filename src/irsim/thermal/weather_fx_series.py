"""Build the scene's one ``WeatherSeries`` from an `isaac-weather-fx` state.

This is the second half of "weather-fx owns the weather" (ADR 0136). The first half
(:mod:`irsim.atmosphere.weather_fx`) moved the *sky* there: the sun, the moon, the cloud field that
both bands march. The thermal solver, meanwhile, still ran off a measured CSV, so a scene could
be rendered under weather-fx's overcast June morning while its surfaces were warming under a
March day from a file. Nothing detects that: both halves are individually plausible and the image
is merely wrong.

So the surface meteorology comes from the same state. weather-fx synthesises the series -- it owns
the physics of what the air is doing, and its
:func:`weather_fx.core.meteorology.diurnal_series` is tested there -- and this module does the one
thing that belongs on this side: it wraps the columns in a :class:`WeatherSeries`, which
range-checks them, makes them immutable, and gives them the content hash that keys every spin-up
cache and golden in this project.

**The CSV is not going away.** ``SceneSpec.weather_file`` remains the default and remains the only
path that carries *measured* weather, which is what a validation run needs. This is the other
branch: a synthetic series, reproducible from a seed, for the datasets where the point is coverage
of conditions rather than agreement with one recorded day. The choice is made where the scene is
built, by passing the result of this function as ``Scene.from_config(weather_override=...)`` --
the same injection point a sensitivity study already used, so the one-weather guard (CLAUDE.md #6)
holds unchanged.

docs/physics-model.md §6.4, §7.3
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np

from irsim.thermal.weather import WEATHER_FIELDS, WeatherSeries

__all__ = [
    "SPIN_UP_H",
    "synchronise_state",
    "weather_series_from_state",
]

#: How much history the series carries before the scene's own instant. A day is enough for
#: anything thinner than a wall to forget its initial condition (ADR 0021), and it is what the
#: measured fixtures in ``data/weather/`` provide, so a synthetic series and a measured one give a
#: solver the same amount of run-up.
SPIN_UP_H = 24.0


def _weather_fx_meteorology() -> Any:
    """Import weather-fx's meteorology module, lazily and with a message that says what to do.

    Lazy for two reasons. The submodule may not be checked out, and a missing optional dependency
    must not stop ``import irsim.thermal`` -- every scene that uses a CSV would break with it.
    And :mod:`irsim.atmosphere.weather_fx` cannot be imported at this module's scope without
    pulling in ``irsim.atmosphere.__init__``, which imports modules that import this one; the same
    cycle that :class:`~irsim.thermal.weather.WeatherSample` documents.
    """
    from irsim.atmosphere.weather_fx import ensure_weather_fx_on_path

    ensure_weather_fx_on_path()
    from weather_fx.core import meteorology

    return meteorology


def synchronise_state(state: Any, when: datetime) -> Any:
    """Return ``state`` with its clock moved to ``when`` (an aware UTC datetime).

    A weather-fx state carries its own date and hour, and a scene config carries ``start_utc``.
    When the two disagree the sun is in the wrong place *and* the thermal history is anchored at
    the wrong hour, and the render still succeeds -- so they are reconciled here, once, in the one
    direction that makes sense: the scene says when, the state says what.
    """
    if when.tzinfo is None or when.utcoffset() is None:
        raise ValueError("`when` must be timezone-aware (naive datetimes are ambiguous)")
    utc = when.astimezone(timezone.utc)
    hour = utc.hour + utc.minute / 60.0 + utc.second / 3600.0 + utc.microsecond / 3.6e9
    return state.with_updates("sky", date_utc=utc.strftime("%Y-%m-%d"), hour_utc=hour)


def weather_series_from_state(
    state: Any,
    *,
    start_utc: datetime | None = None,
    hours: float = 48.0,
    step_s: float = 1800.0,
    spin_up_h: float = SPIN_UP_H,
    swing_k: float | None = None,
    measured_cover: float | None = None,
) -> WeatherSeries:
    """Synthesise the scene's weather from a weather-fx ``WeatherState``.

    The state's own clock is the anchor: the air temperature passes through the state's value at
    the state's hour, and the series starts ``spin_up_h`` before it. So a scene whose
    ``start_utc`` has been written into the state (see :func:`synchronise_state`) gets a series
    that covers the scene start with a day of history in front of it and the rest behind, and
    ``Scene.from_config`` can take ``t0_s`` without meeting the extrapolation refusal.

    :param measured_cover: the cloud fraction to record, overriding the state's *requested* cover.
        Pass :attr:`~irsim.atmosphere.weather_fx.WeatherFxDeck.cover` when a cloud field has been
        built: a thresholded noise field covers what it covers, usually a few percent off what was
        asked for, and the thermal model's sky emissivity should use the cover the camera is
        actually looking at rather than the one in the config.
    """
    meteorology = _weather_fx_meteorology()
    start = _default_start(state, spin_up_h) if start_utc is None else start_utc
    series = meteorology.diurnal_series(
        state, start_utc=start, hours=float(hours), step_s=float(step_s), swing_k=swing_k
    )

    columns = {
        name: np.asarray(column, dtype=np.float64) for name, column in series.columns().items()
    }
    if set(columns) != set(WEATHER_FIELDS):
        # weather-fx and this project name the same eight quantities. If that ever stops being
        # true, say so here rather than letting `from_arrays` report it as a missing column.
        raise ValueError(
            "weather-fx returned columns "
            f"{sorted(columns)}; this project's weather has {sorted(WEATHER_FIELDS)}"
        )
    if measured_cover is not None:
        columns["cloud_fraction"] = np.full_like(
            columns["cloud_fraction"], float(np.clip(measured_cover, 0.0, 1.0))
        )
    return WeatherSeries.from_arrays(series.epoch_utc, series.time_s, **columns)


def _default_start(state: Any, spin_up_h: float) -> datetime:
    """The state's own instant, ``spin_up_h`` earlier."""
    if spin_up_h < 0.0:
        raise ValueError("spin_up_h must be non-negative")
    day = datetime.strptime(state.sky.date_utc, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    anchor = day + timedelta(hours=float(state.sky.hour_utc))
    return anchor - timedelta(hours=float(spin_up_h))
