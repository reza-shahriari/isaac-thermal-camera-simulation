"""Author an `isaac-weather-fx` sky onto a stage, and hand the *same* cloud to the infrared band.

This is the engine-side half of the seam whose engine-free half is
:mod:`irsim.atmosphere.weather_fx`. Together they are what makes "one weather, both bands" a
structural property rather than a discipline:

* the **visible** companion frame gets its dome, its sun and its moon from weather-fx's own
  ``SkyEffect`` -- the same code its UI panel and its Python API drive, not a reimplementation;
* the **infrared** band gets the very same :class:`CloudField` object, wrapped in
  :class:`~irsim.atmosphere.weather_fx.WeatherFxDeck`, and does this project's radiometry on it.

Before this, the two bands ran two cloud models that agreed only as well as two implementations
ever do. They are now one array read twice.

**Why the effect and not a copy of it.** ``SkyEffect`` needs almost nothing from Kit -- a stage
and an up axis -- so a four-line shim context lets a headless render driver use the real thing.
Copying its authoring into a driver would have been quicker and would have started drifting the
same afternoon: the dome's exposure, the sun's airmass law and the moon's intensity are all
decisions that belong to weather-fx and should change there once.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from irsim.atmosphere.weather_fx import WeatherFxDeck, ensure_weather_fx_on_path

__all__ = ["StageOnlyContext", "WeatherFxSky", "author_weather_fx_sky", "weather_state"]


@dataclass
class StageOnlyContext:
    """The smallest thing ``SkyEffect`` will accept in place of a ``WeatherContext``.

    The effect asks for a stage, an up axis and the live state. A render driver has all three and
    none of the rest of a Kit application, which is the whole reason this is four lines.
    """

    _stage: Any
    _up_axis: int
    _state: Any

    def stage(self) -> Any:
        return self._stage

    def up_axis(self) -> int:
        return self._up_axis

    @property
    def state(self) -> Any:
        return self._state

    def meters_per_unit(self) -> float:
        from pxr import UsdGeom

        return float(UsdGeom.GetStageMetersPerUnit(self._stage)) if self._stage else 1.0


@dataclass
class WeatherFxSky:
    """What a driver gets back: the resolved conditions, the deck for the infrared band, and a
    line it can print."""

    state: Any
    conditions: Any
    deck: WeatherFxDeck | None
    effect: Any

    def describe(self) -> str:
        return str(self.conditions.describe())

    def stats(self) -> dict[str, Any]:
        return dict(self.effect.stats())


def weather_state(
    *,
    seed: int | None = None,
    regime: str | None = None,
    preset_json: str | None = None,
    overrides: dict[str, Any] | None = None,
    latitude_deg: float | None = None,
    longitude_deg: float | None = None,
    date_utc: str | None = None,
    hour_utc: float | None = None,
) -> Any:
    """Build a weather-fx ``WeatherState`` for a render.

    Four ways in, in increasing order of specificity, and they compose: a random draw, a named
    regime, a saved JSON preset, and explicit overrides. The site and the clock are applied last
    so a driver can film the *same* weather at a different hour or a different latitude, which is
    the comparison a sensor study usually wants.
    """
    ensure_weather_fx_on_path()
    from weather_fx.core.random_weather import random_state
    from weather_fx.core.state import WeatherState

    if seed is not None or regime is not None:
        state = random_state(seed, regime=regime)
    else:
        state = WeatherState()
    if preset_json:
        import json
        import pathlib

        state = WeatherState.from_dict(
            json.loads(pathlib.Path(preset_json).read_text(encoding="utf-8")), base=state
        )
    for section, values in (overrides or {}).items():
        state = state.with_updates(section, **values)

    clock: dict[str, Any] = {}
    if latitude_deg is not None:
        clock["latitude_deg"] = float(latitude_deg)
    if longitude_deg is not None:
        clock["longitude_deg"] = float(longitude_deg)
    if date_utc is not None:
        clock["date_utc"] = str(date_utc)
    if hour_utc is not None:
        clock["hour_utc"] = float(hour_utc)
    if clock:
        state = state.with_updates("sky", **clock)
    return state


def author_weather_fx_sky(
    stage: Any,
    state: Any,
    *,
    texture_dir: Any = None,
    up_axis: int | None = None,
    od_ratio: float | None = None,
) -> WeatherFxSky:
    """Put weather-fx's sky on ``stage`` and return the cloud the infrared band should march.

    Everything authored lives under ``/WeatherFX`` in the **session layer**, so the caller's own
    stage is untouched and the effect's ``detach`` removes it cleanly.

    The returned ``deck`` is ``None`` for a cloudless state -- a clear sky needs no march, and
    passing a deck that covers nothing would cost a march per pixel to learn that.
    """
    ensure_weather_fx_on_path()
    from weather_fx.backends.viewport.sky import SkyEffect
    from weather_fx.core.sky import conditions_from_state

    if up_axis is None:
        from pxr import UsdGeom

        up_axis = 2 if UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.z else 1

    conditions = conditions_from_state(state)
    effect = SkyEffect(texture_dir=str(texture_dir) if texture_dir else None)
    effect.attach(StageOnlyContext(stage, int(up_axis), state))
    # `changed` names every section, because nothing has been applied to this stage yet and the
    # effect's own short-circuit would otherwise decide there was nothing to do.
    effect.apply_state(state, {"general", "sky", "clouds"})

    deck = None
    if conditions.cloud is not None:
        deck = WeatherFxDeck(conditions.cloud)
        if od_ratio is not None:
            deck.od_ratio = float(od_ratio)
    return WeatherFxSky(state=state, conditions=conditions, deck=deck, effect=effect)
