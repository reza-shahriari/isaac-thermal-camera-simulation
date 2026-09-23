"""The seam between this project and the isaac-weather-fx submodule.

The submodule owns where the cloud is; this project owns what the cloud radiates. The tests here
are about the *joint* between the two, which is where a silent disagreement would live: a rotated
azimuth convention, an emission height measured from the wrong datum, a band ratio applied twice.
Each of those produces a picture that looks like weather.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from irsim.atmosphere.weather_fx import (
    WeatherFxDeck,
    cloud_field_from_spec,
    weather_fx_available,
)

pytestmark = pytest.mark.skipif(
    not weather_fx_available(),
    reason="the isaac-weather-fx submodule is not checked out",
)

SMALL = dict(cells=64, levels=24, cell_m=140.0)


def deck(**overrides) -> WeatherFxDeck:
    kwargs = dict(cover=0.6, genus="cumulus", temperature_c=20.0, dewpoint_c=12.0, seed=5, **SMALL)
    kwargs.update(overrides)
    return WeatherFxDeck(cloud_field_from_spec(**kwargs))


# --- the two projects have to agree about which way is which ---------------------------------


def test_both_projects_place_a_direction_the_same_way() -> None:
    """**The silent one.** If the two conventions differ the whole cloud field is rotated about
    the observer: the infrared band puts cloud where the visible band puts sky, and every frame
    still looks like a sky. Checked against weather-fx's own vector, not against a copy of it."""
    from weather_fx.core.celestial import BodyPosition

    for elevation, azimuth in [(0.0, 0.0), (30.0, 90.0), (10.0, 217.0), (75.0, 300.0)]:
        theirs = BodyPosition(elevation, azimuth, 0.5).direction(
            up_axis=1, north_axis=2, north_sign=-1.0
        )
        el, az = math.radians(elevation), math.radians(azimuth)
        ours = np.array([math.cos(el) * math.sin(az), math.sin(el), -math.cos(el) * math.cos(az)])
        assert np.allclose(theirs, ours, atol=1e-12), (elevation, azimuth, theirs, ours)


def test_the_same_convention_is_the_one_the_visible_dome_uses() -> None:
    """`irsim_isaac.visible_sky.stage_direction` is what bakes the companion frame's sky, so it
    has to be the third member of the same agreement. Imported lazily: it is engine glue, but the
    function itself is pure numpy."""
    from irsim_isaac.visible_sky import stage_direction

    for elevation, azimuth in [(5.0, 12.0), (40.0, 190.0), (88.0, 45.0)]:
        el, az = math.radians(elevation), math.radians(azimuth)
        ours = np.array([math.cos(el) * math.sin(az), math.sin(el), -math.cos(el) * math.cos(az)])
        assert np.allclose(stage_direction(elevation, azimuth), ours, atol=1e-12)


# --- the deck contract ---------------------------------------------------------------------


def test_the_adapter_satisfies_what_the_sky_model_asks_of_a_deck() -> None:
    """Narrow on purpose: optical depth and emission height per ray, and nothing else."""
    result = deck().march(np.radians([[60.0, 25.0]]), np.radians([[0.0, 140.0]]))
    assert result.optical_depth.shape == (1, 2)
    assert result.emission_height_m.shape == (1, 2)
    assert np.all(result.optical_depth >= 0.0)
    assert np.all(np.isfinite(result.emissivity()))


def test_the_emission_height_is_measured_above_the_base_not_above_the_ground() -> None:
    """The temperature it is lapsed from is the cloud *base's*, so the datum has to match. Using
    the height above ground instead makes a 1250 m base emit 1250 m of lapse rate colder -- about
    eight kelvin, uniformly, which reads as a colder day rather than as a bug."""
    d = deck(cover=0.95)
    result = d.march(np.radians([[70.0, 30.0]]), np.radians([[0.0, 90.0]]))
    assert np.all(result.emission_height_m >= 0.0)
    assert np.all(result.emission_height_m <= d.thickness_m + 1.0)
    assert d.base_m > 500.0, "the fixture must have a base well above the ground for this to bite"


def test_a_ray_that_meets_no_cloud_reports_no_optical_depth_and_no_emissivity() -> None:
    clear = WeatherFxDeck(cloud_field_from_spec(cover=0.0, **SMALL))
    result = clear.march(np.radians([[45.0]]), np.radians([[0.0]]))
    assert result.optical_depth.item() == 0.0
    assert result.emissivity().item() == 0.0


def test_the_band_ratio_is_applied_once_and_is_the_project_s_own() -> None:
    """A cloud carries one optical depth and each band derives what it needs from it (ADR 0126).
    Applying the ratio in the adapter *and* in `MarchResult.emissivity` would square it."""
    from irsim.atmosphere.cloud import CLOUD_OD_RATIO

    d = deck()
    result = d.march(np.radians([[35.0]]), np.radians([[20.0]]))
    tau = result.optical_depth.item()
    assert result.emissivity().item() == pytest.approx(1.0 - math.exp(-CLOUD_OD_RATIO * tau))


def test_an_oblique_ray_gathers_more_cloud_than_a_vertical_one() -> None:
    """The property that makes a third dimension worth having at all."""
    d = deck(cover=0.95)
    up = d.march(np.radians([[89.0]]), np.radians([[0.0]])).optical_depth.item()
    slant = d.march(np.radians([[12.0]]), np.radians([[0.0]])).optical_depth.item()
    assert slant > up


def test_the_base_comes_from_the_dew_point_spread() -> None:
    """125 m per kelvin, so a scene states a temperature and a humidity rather than a height."""
    assert deck(temperature_c=20.0, dewpoint_c=12.0).base_m == pytest.approx(1000.0)
    assert deck(temperature_c=25.0, dewpoint_c=5.0).base_m == pytest.approx(2500.0)
    # An explicit base overrides the derivation, for a scene that knows its own ceiling.
    assert deck(base_m=800.0).base_m == pytest.approx(800.0)


def test_the_cover_that_comes_back_is_the_one_that_was_asked_for() -> None:
    assert deck(cover=0.3).cover == pytest.approx(0.3, abs=0.04)
    assert deck(cover=0.8).cover == pytest.approx(0.8, abs=0.04)


def test_a_missing_submodule_says_how_to_fix_it_rather_than_failing_to_import() -> None:
    """The failure a new checkout meets first, so the message has to carry the command."""
    import irsim.atmosphere.weather_fx as bridge

    original = bridge.WEATHER_FX_PACKAGE
    saved = {
        name: module
        for name, module in list(__import__("sys").modules.items())
        if name.startswith("weather_fx")
    }
    try:
        import sys

        for name in saved:
            del sys.modules[name]
        bridge.WEATHER_FX_PACKAGE = original.parent / "definitely-not-here"
        sys.path[:] = [p for p in sys.path if "weather.fx" not in p]
        with pytest.raises(ImportError, match="git submodule update --init"):
            bridge.ensure_weather_fx_on_path()
    finally:
        bridge.WEATHER_FX_PACKAGE = original
        __import__("sys").modules.update(saved)
        bridge.ensure_weather_fx_on_path()
