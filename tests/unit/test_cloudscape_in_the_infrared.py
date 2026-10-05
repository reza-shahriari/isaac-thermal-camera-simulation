"""The infrared march reads the cloudscape (WX.26, ADR 0178, ADR 0185).

The visible per-pixel renderer draws a weather-fx ``Cloudscape``: simulated cloud patches placed
by a weather map, a function of position. These tests are about the infrared reading that same
function through ``WeatherFxDeck``: the deck's contract holds for it as it does for a
``CloudField``, the infrared sees cloud exactly where the function has cloud, and nothing of the
cloud's visible look can reach the band except through where the cloud is.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from irsim.atmosphere.cloud import CLOUD_OD_RATIO
from irsim.atmosphere.weather_fx import (
    WeatherFxDeck,
    ensure_weather_fx_on_path,
    weather_fx_available,
)

needs_weather_fx = pytest.mark.skipif(
    not weather_fx_available(), reason="the isaac-weather-fx submodule is not checked out"
)

SMALL = dict(weather_cells=128, shape_cells=32, detail_cells=24)
BASE_M = 1200.0


def _patch(name: str, seed: int):  # type: ignore[no-untyped-def]
    """A few soft lumps on a floor: a small stand-in for a simulated patch."""
    from weather_fx.core import cloudscape as cs

    shape = (44, 24, 40)
    rng = np.random.default_rng(seed)
    x, y, z = np.meshgrid(*(np.arange(n) + 0.5 for n in shape), indexing="ij")
    density = np.zeros(shape)
    for _ in range(4):
        cx, cz = rng.uniform(12, shape[0] - 12), rng.uniform(12, shape[2] - 12)
        r, top = rng.uniform(5, 9), rng.uniform(10, 20)
        density = np.maximum(density, np.clip(1.6 - np.hypot(x - cx, z - cz) / r, 0, 1) * (y < top))
    return cs.CloudPatch(name, density.astype(np.float32), 50.0)


def _cloudscape(**profile_changes):  # type: ignore[no-untyped-def]
    ensure_weather_fx_on_path()
    from weather_fx.core import cloudscape as cs

    profile = dataclasses.replace(cs.CLOUDSCAPE_TYPES["cumulus"], **profile_changes)
    return cs.Cloudscape(
        cover=0.4,
        base_m=BASE_M,
        profile=profile,
        seed=5,
        patches=[_patch("a", 1), _patch("b", 2), _patch("c", 3)],
        **SMALL,
    )


@pytest.fixture(scope="module")
def scape():  # type: ignore[no-untyped-def]
    return _cloudscape()


def _rays() -> tuple[np.ndarray, np.ndarray]:
    el = np.radians(np.linspace(12.0, 75.0, 24))[:, None]
    az = np.radians(np.linspace(0.0, 355.0, 72))[None, :]
    return np.broadcast_arrays(el, az)


@needs_weather_fx
def test_the_deck_contract_holds_for_a_cloudscape(scape) -> None:  # type: ignore[no-untyped-def]
    """Base, top, thickness, pitch, cover and median optical depth all answer, as plain floats,
    and the cover is the measured one (a method on a cloudscape, a property on a cloud field)."""
    deck = WeatherFxDeck(scape)
    assert deck.base_m == BASE_M
    assert deck.top_m == pytest.approx(BASE_M + deck.thickness_m)
    assert deck.sample_pitch_m == 50.0
    assert 0.05 < deck.cover < 0.6
    assert deck.optical_depth > 0.4
    assert deck.steps_for(8000.0) == 480  # three samples per 50 m pitch


@needs_weather_fx
def test_the_infrared_sees_cloud_where_the_function_has(scape) -> None:  # type: ignore[no-untyped-def]
    """Ray by ray, the marched visible optical depth is the function's own density integrated
    along that ray on a dense reference: the same clouds in the same pixels. Intersection over
    union of the two cloud masks is above 0.9, the bar WX.26 sets against the visible frame."""
    deck = WeatherFxDeck(scape)
    el, az = _rays()
    marched = deck.march(el, az).optical_depth
    direction = np.stack([np.cos(el) * np.sin(az), np.sin(el), -np.cos(el) * np.cos(az)], axis=-1)
    near, far = scape.slab_span(np.zeros(3), direction)
    far = np.minimum(far, near + deck.max_path_m)
    n = 1536
    reference = np.zeros(el.shape)
    for k in range(n):
        t = near + (k + 0.5) / n * (far - near)
        p = direction * t[..., None]
        reference += scape.density(p[..., 0], p[..., 1], p[..., 2])
    reference *= (far - near) / n * scape.extinction_per_m
    a, b = marched > 1.0, reference > 1.0
    assert b.mean() > 0.1
    assert (a & b).sum() / (a | b).sum() > 0.9
    # And the band's emissivity agrees to a few hundredths where it matters.
    emissivity = 1.0 - np.exp(-CLOUD_OD_RATIO * marched)
    wanted = 1.0 - np.exp(-CLOUD_OD_RATIO * reference)
    assert np.percentile(np.abs(emissivity - wanted), 95) < 0.05


@needs_weather_fx
def test_the_cloud_emits_from_inside_its_own_layer(scape) -> None:  # type: ignore[no-untyped-def]
    """Where a ray meets cloud, the height the band's emission comes from lies between the base
    and the top; a clear ray has no optical depth at all."""
    deck = WeatherFxDeck(scape)
    el, az = _rays()
    m = deck.march(el, az)
    cloudy = m.optical_depth > 1.0
    assert np.all(m.emission_height_m[cloudy] >= 0.0)
    assert np.all(m.emission_height_m[cloudy] <= deck.thickness_m)
    assert (m.optical_depth == 0.0).mean() > 0.05


@needs_weather_fx
def test_a_target_under_the_base_is_clear_one_beyond_is_hidden(scape) -> None:  # type: ignore[no-untyped-def]
    """The occlusion march, on the cloudscape: 300 m away under a 1200 m base there is no cloud
    on any ray; at infinity the cloud on the ray is the sky march's own; and an opaque cloud
    emitting radiance 5 returns 5 times what it absorbed (the Schwarzschild integral closes)."""
    deck = WeatherFxDeck(scape)
    el, az = _rays()
    near = deck.march_to(el, az, 300.0, lambda h: np.full(np.shape(h), 5.0))
    assert np.all(near.transmittance == 1.0)
    assert np.all(near.path_radiance == 0.0)
    far = deck.march_to(el, az, np.inf, lambda h: np.full(np.shape(h), 5.0))
    sky = deck.march(el, az)
    hidden = sky.optical_depth > 20.0
    assert hidden.any()
    assert np.all(far.transmittance[hidden] < 1e-3)
    assert np.allclose(far.path_radiance, 5.0 * (1.0 - far.transmittance), rtol=1e-9, atol=1e-12)


@needs_weather_fx
def test_the_visible_look_does_not_reach_the_band(scape) -> None:  # type: ignore[no-untyped-def]
    """The march reads density and extinction only. A cloudscape whose visible skin constants
    differ (how crisp its edge is drawn, how bright its base) but whose density is the same
    function gives the infrared the same optical depth on every ray."""
    el, az = _rays()
    first = WeatherFxDeck(scape).march(el, az).optical_depth
    # ``edge_top`` and ``water_base`` shape the *function* genus only; with patches they are
    # visible-side constants the density never reads.
    other = _cloudscape(edge_top=5.0, water_base=0.9)
    assert np.array_equal(WeatherFxDeck(other).march(el, az).optical_depth, first)
