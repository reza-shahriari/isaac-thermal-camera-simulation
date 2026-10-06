"""A detector pixel integrates the cloud's radiance; it does not sample it (WX.4).

docs/physics-model.md §7.5. `march_on_native_grid` marches a supersampled frame once per native
pixel, as the camera always has, and marches again only the pixels a cloud edge crosses. These
tests hold it to what that is for: a smooth sky costs nothing extra and comes out as before, an
edge pixel's value is the fraction of it that is cloud, and the static near the horizon -- a
random pick of cloud or sky per pixel -- falls to a tenth of what it was at no more than twice
the rays.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from irsim.atmosphere.cloud_deck import (
    REFINE_EVERY_SAMPLE_BELOW_DEG,
    march_on_native_grid,
    resample_bilinear,
)

#: A Boson's 0.857 mrad IFOV, supersampled four times, as the camera renders it.
STRIDE = 4
PITCH = 0.8571e-3 / STRIDE


def _grid(rows: int, cols: int, elevation_deg: float):
    el = math.radians(elevation_deg) + (np.arange(rows) - 0.5 * rows)[:, None] * PITCH * np.ones(
        (1, cols)
    )
    az = (np.arange(cols) - 0.5 * cols)[None, :] * PITCH * np.ones((rows, 1))
    return el, az


class _Counted:
    """An evaluate that counts the rays it was asked for."""

    def __init__(self, fn):
        self.fn = fn
        self.rays = 0

    def __call__(self, el, az, *extras):
        self.rays += int(np.asarray(el).size)
        return self.fn(el, az, *extras)


def _native(a: np.ndarray) -> np.ndarray:
    r, c = a.shape
    return a.reshape(r // STRIDE, STRIDE, c // STRIDE, STRIDE).mean(axis=(1, 3))


def test_a_sky_with_no_edge_is_the_interpolation_and_costs_one_ray_a_pixel() -> None:
    el, az = _grid(64, 96, 30.0)
    smooth = _Counted(lambda e, a: (250.0 + 40.0 * np.asarray(e) + 3.0 * np.asarray(a),))
    (out,) = march_on_native_grid(smooth, el, az, STRIDE, threshold=0.5)
    coarse = smooth.fn(el[::STRIDE, ::STRIDE], az[::STRIDE, ::STRIDE])[0]
    assert np.array_equal(out, resample_bilinear(coarse, el.shape))
    assert smooth.rays == (64 // STRIDE) * (96 // STRIDE)


@pytest.mark.parametrize("elevation_deg", [5.0, 30.0])
def test_an_edge_pixel_reads_the_fraction_of_it_that_is_cloud(elevation_deg: float) -> None:
    """A vertical edge a third of the way into a pixel. Interpolated, that pixel reads whatever
    its neighbours' samples were. Refined, it reads the fraction of it that is cloud: exactly,
    marched at every sample near the horizon; to within one 2 x 2 block's share above it."""
    el, az = _grid(32, 64, elevation_deg)
    pixel = 7
    edge_az = az[0, pixel * STRIDE] + (STRIDE / 3.0) * PITCH

    def step(e, a):
        return ((np.asarray(a) > edge_az).astype(np.float64),)

    (out,) = march_on_native_grid(_Counted(step), el, az, STRIDE, threshold=0.5)
    truth = _native(step(el, az)[0])
    got = _native(out)
    tolerance = 1e-12 if elevation_deg < REFINE_EVERY_SAMPLE_BELOW_DEG else 0.25 + 1e-12
    assert np.max(np.abs(got[:, pixel] - truth[:, pixel])) <= tolerance
    # And every other pixel is exact, since the field is constant away from the edge.
    others = np.ones(got.shape[1], dtype=bool)
    others[pixel] = False
    assert np.allclose(got[:, others], truth[:, others])


def test_a_per_sample_input_reaches_the_rays_that_use_it() -> None:
    """The range to each sample's hit is not averaged across a geometry edge: the rays marched
    again carry their own samples' ranges."""
    el, az = _grid(32, 32, 5.0)
    rng = np.where(az > 0.0, 1000.0, np.inf)

    def ev(e, a, r):
        return (np.where(np.isfinite(r), 0.0, 1.0),)

    (out,) = march_on_native_grid(ev, el, az, STRIDE, threshold=0.5, extras=(rng,))
    assert np.array_equal(out, np.where(np.isfinite(rng), 0.0, 1.0))


def test_a_drone_the_coarse_samples_miss_keeps_its_own_range() -> None:
    """WX.26: a drone at 5 m in front of a cloud base at 910 m. The cloud is uniform, so the
    coarse samples -- all on the sky -- agree and flag no edge, and interpolation spread their
    cloud over the drone (70 pixels read opaque in the cumulus clip). A pixel whose samples do
    not share one range is marched at every sample instead, at 5 degrees and at 40 alike."""
    for elevation in (5.0, 40.0):
        el, az = _grid(32, 32, elevation)
        rng = np.full(el.shape, np.inf)
        rng[13:15, 17:18] = 5.0  # smaller than one coarse cell, off every coarse sample

        def ev(e, a, r):
            return (np.where(np.isfinite(r), 1.0, 0.02),)  # transmittance: clear to 5 m

        (out,) = march_on_native_grid(ev, el, az, STRIDE, threshold=0.02, extras=(rng,))
        assert np.array_equal(out[13:15, 17:18], np.ones((2, 1))), elevation
        assert np.allclose(out[np.isinf(rng)], 0.02)


@pytest.mark.slow
def test_the_horizon_static_falls_below_a_tenth_at_no_more_than_twice_the_rays() -> None:
    """The step's verification, on the shipped weather-fx field (35 % cumulus). Measured against a
    march of every sample, which is what the detector's box filter integrates: the static energy
    falls to 0.1 % of the interpolation's at 5 degrees for 1.09x its rays, and 0.5 % at 30 for
    1.87x. The bound is 10 % at 2x, the roadmap's."""
    from irsim.atmosphere.weather_fx import (
        WeatherFxDeck,
        cloud_field_from_spec,
        weather_fx_available,
    )

    if not weather_fx_available():
        pytest.skip("the isaac-weather-fx submodule is not checked out")
    deck = WeatherFxDeck(
        cloud_field_from_spec(
            cover=0.35, genus="cumulus", temperature_c=22.0, dewpoint_c=12.0, seed=17
        )
    )
    rows, cols = 256, 640
    for elevation_deg in (5.0, 30.0):
        el, az = _grid(rows, cols, elevation_deg)
        march = _Counted(lambda e, a: (deck.march(e, a).emissivity(),))
        reference = _native(march(el, az)[0])
        march.rays = 0
        coarse = march(el[::STRIDE, ::STRIDE], az[::STRIDE, ::STRIDE])[0]
        interpolated = _native(resample_bilinear(coarse, el.shape))
        native_rays = march.rays
        march.rays = 0
        (refined,) = march_on_native_grid(march, el, az, STRIDE, threshold=0.02)
        before = np.mean((interpolated - reference) ** 2)
        after = np.mean((_native(refined) - reference) ** 2)
        assert after < 0.1 * before, (elevation_deg, after / before)
        assert march.rays <= 2.0 * native_rays, (elevation_deg, march.rays / native_rays)
