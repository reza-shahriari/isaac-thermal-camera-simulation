"""The cloud moves down the wind in a headless clip, in every band at once (WX.28).

weather-fx's manager carries the cloud field on its own clock in the viewport; a render driver
has no such clock, so until WX.28 every clip -- 1,666 s of mission in the dataset clips -- showed
one frozen sky. ``WeatherFxSky.advance_clouds`` evaluates weather-fx's pure drift law for the
clip's time and writes it into the one context the dome, the volumes and the per-pixel layer read,
and into the infrared deck's origin. These tests hold the sign (the cloud overhead now is the
cloud that was upwind), the speed (surface wind times ``clouds.wind_factor``), that the deck and
the visible side share one number, and that the driver calls it.
"""

from __future__ import annotations

import math
import pathlib
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from irsim.atmosphere.weather_fx import (
    WeatherFxDeck,
    ensure_weather_fx_on_path,
    weather_fx_available,
)

REPO = pathlib.Path(__file__).resolve().parents[2]
needs_weather_fx = pytest.mark.skipif(
    not weather_fx_available(), reason="the isaac-weather-fx submodule is not checked out"
)


def _state(speed: float = 6.0, direction_deg: float = 30.0, factor: float = 1.5) -> Any:
    return SimpleNamespace(
        wind=SimpleNamespace(speed_mps=speed, direction_deg=direction_deg),
        clouds=SimpleNamespace(wind_factor=factor),
    )


def _deck() -> WeatherFxDeck:
    ensure_weather_fx_on_path()
    from weather_fx.core import cloudscape as cs

    scape = cs.Cloudscape(
        cover=0.5,
        base_m=1200.0,
        profile=cs.CLOUDSCAPE_TYPES["cumulus"],
        seed=11,
        weather_cells=128,
        shape_cells=32,
        detail_cells=24,
    )
    return WeatherFxDeck(scape, origin_m=(0.0, 0.0, 0.0))


class _Recorder:
    """Stands in for weather-fx's sky and volume effects: counts the updates it is given."""

    def __init__(self) -> None:
        self.updates = 0

    def update(self, dt: float, t: float) -> None:
        self.updates += 1


def _sky(deck: Any, state: Any, anchor_m: tuple[float, float, float], up_axis: int) -> Any:
    from irsim_isaac.weather_fx_stage import StageOnlyContext, WeatherFxSky

    context = StageOnlyContext(None, up_axis, state, anchor_m=anchor_m)
    return WeatherFxSky(
        state=state,
        conditions=None,
        deck=deck,
        effect=_Recorder(),
        volumes=_Recorder(),
        context=context,
        anchor_m=anchor_m,
        up_axis=up_axis,
    )


@needs_weather_fx
@pytest.mark.parametrize("up_axis", [1, 2])
def test_the_drift_is_the_wind_aloft_times_the_clip_s_time(up_axis: int) -> None:
    """6 m/s at the surface, a wind factor of 1.5 and 400 s of clip carry the cloud 3.6 km
    toward 30 degrees from +X, horizontal in either up-axis convention -- weather-fx's own law,
    which the viewport draws, not a second copy of it here."""
    sky = _sky(None, _state(), (0.0, 0.0, 0.0), up_axis)
    drift = sky.advance_clouds(400.0)
    assert float(np.linalg.norm(drift)) == pytest.approx(6.0 * 1.5 * 400.0, rel=1e-12)
    assert drift[up_axis] == 0.0
    h0, h1 = (0, 2) if up_axis == 1 else (0, 1)
    assert math.degrees(math.atan2(drift[h1], drift[h0])) == pytest.approx(30.0)
    assert np.array_equal(sky.context.cloud_drift_m, drift)
    assert sky.context.time == 400.0
    assert np.array_equal(sky.advance_clouds(0.0), np.zeros(3))


@needs_weather_fx
def test_the_cloud_overhead_later_is_the_cloud_that_was_upwind() -> None:
    """Frozen turbulence: after t seconds the cloud the camera sees is the cloud that, at time
    zero, was v t upwind of it. A deck advanced 300 s marches exactly what an undrifted deck
    marches from the upwind point -- so the sign is downwind, not up -- and differs from what
    it marched at time zero."""
    from weather_fx.core.clouds import stage_to_field

    anchor = (250.0, 0.0, -400.0)
    deck = _deck()
    sky = _sky(deck, _state(), anchor, 1)
    el = np.radians(np.linspace(35.0, 90.0, 8))[:, None] * np.ones((1, 24))
    az = np.radians(np.linspace(0.0, 345.0, 24))[None, :] * np.ones((8, 1))

    sky.advance_clouds(0.0)
    before = deck.march(el, az).optical_depth.copy()
    drift = sky.advance_clouds(300.0)
    after = deck.march(el, az).optical_depth

    upwind = stage_to_field(np.asarray(anchor) - drift, 1)
    reference = _deck().march(el, az, origin_m=tuple(float(v) for v in upwind)).optical_depth
    assert np.array_equal(after, reference)
    assert np.count_nonzero(np.abs(after - before) > 1e-6) > after.size // 4


@needs_weather_fx
def test_the_visible_effects_and_the_deck_read_one_drift() -> None:
    """The dome and the volumes are told to update (they place themselves from the context's
    drift), and the deck's origin is the dome's own expression of that same drift; one call
    moves all of them, so no band can show the cloud somewhere the other does not."""
    from weather_fx.core.clouds import stage_to_field

    anchor = (10.0, 1.5, 20.0)
    deck = _deck()
    sky = _sky(deck, _state(speed=4.0, direction_deg=200.0), anchor, 1)
    drift = sky.advance_clouds(90.0)
    expected = stage_to_field(np.asarray(anchor) - sky.context.cloud_drift_m, 1)
    assert np.allclose(deck.origin_m, expected, rtol=0.0, atol=1e-9)
    assert np.array_equal(sky.context.cloud_drift_m, drift)
    assert sky.effect.updates == 1 and sky.volumes.updates == 1


def test_the_driver_moves_the_cloud_every_frame_unless_told_not_to() -> None:
    """`render_phantom4.py` advances the cloud by the frame's own time in the clip before each
    capture, keeps `--freeze-clouds` for a still sky, and records the last drift."""
    source = (REPO / "scripts" / "render_phantom4.py").read_text(encoding="utf-8")
    assert "--freeze-clouds" in source
    assert "weather_sky.advance_clouds(index * interval_s)" in source
    loop = source.index("for index in range(args.frames):")
    call = source.index("weather_sky.advance_clouds(index * interval_s)")
    capture = source.index("outputs = cam.get_outputs(", loop)
    assert loop < call < capture
    assert '"cloud_drift_m":' in source


@needs_weather_fx
def test_the_sky_model_marches_from_the_deck_s_own_origin(aerial_cloudy_sky: Any) -> None:
    """The infrared's two marches -- the sky pixels' and the occlusion to every hit -- go through
    the sky model, which defaulted ``origin_m`` to (0, 0, 0) and passed it on: that overrode the
    deck's origin, so the first drifting clip moved the visible cloud by 9.9 km and the infrared's
    not at all (IoU 0 by its last frame). With no origin given both now march from the deck's."""
    sky = aerial_cloudy_sky
    deck = _deck()
    deck.origin_m = (2500.0, 0.0, -1800.0)
    el = np.radians(np.linspace(40.0, 85.0, 6))[:, None] * np.ones((1, 16))
    az = np.radians(np.linspace(0.0, 337.5, 16))[None, :] * np.ones((6, 1))
    shifted = deck.march(el, az).optical_depth
    at_zero = deck.march(el, az, origin_m=(0.0, 0.0, 0.0)).optical_depth
    assert np.count_nonzero(np.abs(shifted - at_zero) > 1e-6) > shifted.size // 4

    occlusion = sky.cloud_occlusion(0.0, el, az, deck, np.inf)
    expected = deck.march_to(el, az, np.inf, lambda h: np.zeros_like(h)).transmittance
    assert np.allclose(occlusion.transmittance, expected, rtol=0.0, atol=1e-12)
    radiance = sky.radiance_field_from_deck(0.0, el, az, deck)
    reference = sky.radiance_field_from_deck(0.0, el, az, deck, origin_m=deck.origin_m)
    assert np.array_equal(radiance, reference)
    # an explicit origin still wins
    pinned = sky.cloud_occlusion(0.0, el, az, deck, np.inf, origin_m=(0.0, 0.0, 0.0))
    assert not np.allclose(pinned.transmittance, occlusion.transmittance)
