"""The scene's weather, synthesised from a weather-fx state instead of read from a CSV.

The joint tested here is narrow and the failure it guards against is not: a series that is
individually plausible but anchored at the wrong instant, or whose cloud fraction disagrees with
the cloud the camera is looking at, produces a scene where the sky and the surfaces describe two
different days. Both halves look right on their own.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from irsim.atmosphere.weather_fx import weather_fx_available
from irsim.thermal.weather import WEATHER_FIELDS, WeatherSeries

pytestmark = pytest.mark.skipif(
    not weather_fx_available(),
    reason="the isaac-weather-fx submodule is not checked out",
)


def _state(**sections):
    from irsim.atmosphere.weather_fx import ensure_weather_fx_on_path

    ensure_weather_fx_on_path()
    from weather_fx.core.state import WeatherState

    state = WeatherState()
    for name, values in sections.items():
        state = state.with_updates(name, **values)
    return state


def _fair_day(**sky):
    params = {
        "latitude_deg": 48.1,
        "longitude_deg": 11.6,
        "date_utc": "2024-06-21",
        "hour_utc": 10.0,
        "turbidity": 2.8,
    }
    params.update(sky)
    return _state(
        sky=params,
        clouds={"enabled": True, "cover": 0.38, "temperature_c": 22.0, "dewpoint_c": 12.0},
        wind={"speed_mps": 3.0},
    )


# --- the wrapper does what a WeatherSeries is for ----------------------------------------------


def test_the_result_is_a_validated_immutable_weather_series() -> None:
    from irsim.thermal.weather_fx_series import weather_series_from_state

    series = weather_series_from_state(_fair_day())
    assert isinstance(series, WeatherSeries)
    assert set(series.columns()) == set(WEATHER_FIELDS)
    for name in WEATHER_FIELDS:
        column = getattr(series, name)
        assert column.dtype == np.float64  # CLAUDE.md #2: never float16 in a temperature path
        assert not column.flags.writeable


def test_two_builds_from_one_state_are_the_same_weather() -> None:
    """The content hash keys every spin-up cache and golden in this project, so a synthetic
    series that is not reproducible quietly invalidates all of them on every run."""
    from irsim.thermal.weather_fx_series import weather_series_from_state

    first = weather_series_from_state(_fair_day())
    second = weather_series_from_state(_fair_day())
    assert first.content_hash == second.content_hash
    assert first == second


def test_a_different_state_is_a_different_weather() -> None:
    from irsim.thermal.weather_fx_series import weather_series_from_state

    fair = weather_series_from_state(_fair_day())
    warmer = weather_series_from_state(_fair_day().with_updates("clouds", temperature_c=22.01))
    assert fair.content_hash != warmer.content_hash


# --- the anchor ---------------------------------------------------------------------------------


def test_the_series_covers_the_scene_start_with_a_day_of_history_in_front_of_it() -> None:
    """``Scene.from_config`` refuses to extrapolate, so a series that starts at the scene start
    fails on the first spin-up step -- and a series centred on the wrong day fails at the render,
    which is much later and much harder to read."""
    from irsim.thermal.weather_fx_series import weather_series_from_state

    when = datetime(2024, 6, 21, 10, 0, tzinfo=timezone.utc)
    series = weather_series_from_state(_fair_day())
    assert series.start_utc == when - timedelta(hours=24.0)
    assert series.end_utc >= when
    # Asking for the scene's own instant must not raise.
    assert series.at_datetime(when).t_air_k > 0.0


def test_the_air_temperature_at_the_scene_start_is_the_state_s_own_temperature() -> None:
    from irsim.thermal.weather_fx_series import weather_series_from_state

    when = datetime(2024, 6, 21, 10, 0, tzinfo=timezone.utc)
    series = weather_series_from_state(_fair_day())
    assert series.at_datetime(when).t_air_k == pytest.approx(22.0 + 273.15, abs=1e-6)


def test_synchronise_state_moves_the_clock_to_the_scene_s_own_start() -> None:
    from irsim.thermal.weather_fx_series import synchronise_state, weather_series_from_state

    when = datetime(2024, 3, 2, 16, 30, tzinfo=timezone.utc)
    moved = synchronise_state(_fair_day(), when)
    assert moved.sky.date_utc == "2024-03-02"
    assert moved.sky.hour_utc == pytest.approx(16.5)
    series = weather_series_from_state(moved)
    assert series.at_datetime(when).t_air_k == pytest.approx(22.0 + 273.15, abs=1e-6)


def test_a_naive_datetime_is_refused_rather_than_assumed_to_be_utc() -> None:
    from irsim.thermal.weather_fx_series import synchronise_state

    with pytest.raises(ValueError, match="timezone-aware"):
        synchronise_state(_fair_day(), datetime(2024, 3, 2, 16, 30))


# --- the physics survives the wrapping -----------------------------------------------------------


def test_the_night_is_dark_and_the_day_is_not() -> None:
    from irsim.thermal.weather_fx_series import weather_series_from_state

    series = weather_series_from_state(_fair_day(), hours=48.0, step_s=900.0)
    assert float(series.dni_w_m2.min()) == 0.0
    assert float(series.dni_w_m2.max()) > 300.0

    # The coldest hour is before dawn and the warmest is in the afternoon, not the reverse --
    # in *solar* time, which at 11.6 degE runs three quarters of an hour ahead of UTC. Asserting
    # on the UTC hour instead would pass at Greenwich and fail everywhere else.
    def solar_hour(index: int) -> float:
        when = series.datetime_at(float(series.time_s[index]))
        return (when.hour + when.minute / 60.0 + 11.6 / 15.0) % 24.0

    assert 2.0 <= solar_hour(int(np.argmin(series.t_air_k))) <= 3.0
    assert 13.5 <= solar_hour(int(np.argmax(series.t_air_k))) <= 15.5


def test_the_thermal_model_sees_the_cover_the_camera_is_actually_looking_at() -> None:
    """A thresholded noise field covers what it covers. If the solver uses the requested cover
    and the render marches the built one, the sky emissivity and the picture disagree by a few
    percent for no reason anyone can find later."""
    from irsim.atmosphere.weather_fx import WeatherFxDeck, cloud_field_from_spec
    from irsim.thermal.weather_fx_series import weather_series_from_state

    deck = WeatherFxDeck(
        cloud_field_from_spec(
            cover=0.38,
            genus="cumulus",
            temperature_c=22.0,
            dewpoint_c=12.0,
            seed=5,
            cells=64,
            levels=24,
            cell_m=140.0,
        )
    )
    series = weather_series_from_state(_fair_day(), measured_cover=deck.cover)
    assert float(series.cloud_fraction[0]) == pytest.approx(deck.cover, abs=1e-9)
    assert float(series.cloud_fraction[0]) != pytest.approx(0.38, abs=1e-9)


def test_the_state_s_own_parameter_limits_are_the_first_guard() -> None:
    """weather-fx clamps every authored field to the range its ``param`` declares, so an absurd
    temperature never reaches this project at all. Worth pinning: it is the reason the wrapper
    below only has to catch what is *computed*."""
    absurd = _fair_day().with_updates("clouds", temperature_c=500.0)
    assert absurd.clouds.temperature_c <= 50.0


def test_a_computed_column_out_of_range_is_refused_at_the_seam_not_in_a_solver() -> None:
    """``WeatherSeries`` is where the ranges live, and this is the reason the columns are wrapped
    rather than handed on. A swing nobody clamps is the way an out-of-range column can still be
    produced from an in-range state, and it must fail here rather than five layers down."""
    from irsim.thermal.weather_fx_series import weather_series_from_state

    with pytest.raises(ValueError, match="t_air_k"):
        weather_series_from_state(_fair_day(), swing_k=500.0)


def test_a_random_regime_gives_a_usable_scene_weather() -> None:
    from irsim.atmosphere.weather_fx import ensure_weather_fx_on_path
    from irsim.thermal.weather_fx_series import weather_series_from_state

    ensure_weather_fx_on_path()
    from weather_fx.core.random_weather import random_state

    for seed in (0, 1, 2, 3, 4, 5, 6, 7):
        series = weather_series_from_state(random_state(seed))
        assert series.n > 2
        assert series.duration_s == pytest.approx(48.0 * 3600.0)
        sample = series.at(series.duration_s * 0.5)
        assert 0.0 <= sample.rh_fraction <= 1.0
        assert sample.visibility_m > 0.0


# --- it reaches a scene the way a CSV does --------------------------------------------------------


def test_it_drops_into_scene_from_config_where_the_csv_would_have_been() -> None:
    """The whole point: the injection point already exists, so weather-fx can own the weather
    without a second path through the scene builder and without a second series in play."""
    import inspect

    from irsim.scene import Scene

    signature = inspect.signature(Scene.from_config)
    assert "weather_override" in signature.parameters
