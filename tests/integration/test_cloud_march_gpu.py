"""The GPU cloud march is the CPU deck, pixel for pixel (WX.26 follow-up).

`irsim_isaac.cloud_march_gpu.GpuWeatherFxDeck` runs the same quadrature as
`irsim.atmosphere.weather_fx.WeatherFxDeck` -- the same slab span, step count, hashed jitter,
Schwarzschild sum and drop-out -- with the density from weather-fx's Warp `cloud_density`. The
CPU deck is the oracle (ADR 0018): the two must agree to within the texture interpolation the
GPU density is already held to upstream (p99 0.02 in density), which this file holds on the
integrated quantities instead.

    $PYTHON -m pytest tests/integration/test_cloud_march_gpu.py -m gpu
"""

from __future__ import annotations

import dataclasses
import time
from typing import Any

import numpy as np
import pytest

pytestmark = pytest.mark.gpu
SMALL = dict(weather_cells=128, shape_cells=32, detail_cells=24)


@pytest.fixture(scope="module")
def decks() -> tuple[Any, Any]:
    from irsim.atmosphere.weather_fx import WeatherFxDeck, ensure_weather_fx_on_path
    from irsim_isaac.cloud_march_gpu import GpuWeatherFxDeck, gpu_march_available

    if not gpu_march_available():
        pytest.skip("no Warp with a CUDA device")
    ensure_weather_fx_on_path()
    from weather_fx.core import cloudscape as cs

    profile = dataclasses.replace(cs.CLOUDSCAPE_TYPES["cumulus"], extinction_per_m=0.05)
    scape = cs.Cloudscape(cover=0.45, base_m=1200.0, profile=profile, seed=7, **SMALL)
    origin = (0.0, 1.5, 0.0)
    return WeatherFxDeck(scape, origin_m=origin), GpuWeatherFxDeck(scape, origin_m=origin)


def _rays(n_el: int = 48, n_az: int = 96) -> tuple[np.ndarray, np.ndarray]:
    el = np.radians(np.linspace(8.0, 80.0, n_el))[:, None]
    az = np.radians(np.linspace(0.0, 360.0, n_az, endpoint=False))[None, :]
    return np.broadcast_arrays(el, az)


def test_the_sky_march_agrees_with_the_cpu_deck(decks: tuple[Any, Any]) -> None:
    cpu, gpu = decks
    el, az = _rays()
    a, b = cpu.march(el, az), gpu.march(el, az)
    eps_a, eps_b = 1.0 - np.exp(-0.5 * a.optical_depth), 1.0 - np.exp(-0.5 * b.optical_depth)
    assert (eps_a > 0.5).mean() > 0.1
    error = np.abs(eps_a - eps_b)
    assert np.percentile(error, 99) < 0.02 and error.mean() < 2e-3
    cloudy = eps_a > 0.1
    assert np.percentile(np.abs(a.emission_height_m - b.emission_height_m)[cloudy], 95) < 25.0


def test_the_march_to_the_hit_agrees_with_the_cpu_deck(decks: tuple[Any, Any]) -> None:
    cpu, gpu = decks
    el, az = _rays()
    rng = np.random.default_rng(2)
    ranges = np.where(
        rng.uniform(size=el.shape) < 0.5, np.inf, rng.uniform(1500.0, 6000.0, el.shape)
    )

    def radiance(h: np.ndarray) -> np.ndarray:
        return 10.0 - 0.0065 * (h - 1200.0) * 0.1  # a band radiance falling with height

    a = cpu.march_to(el, az, ranges, radiance)
    b = gpu.march_to(el, az, ranges, radiance)
    error = np.abs(a.transmittance - b.transmittance)
    assert np.percentile(error, 99) < 0.02 and error.mean() < 2e-3
    cloudy = a.transmittance < 0.9
    assert cloudy.any()
    rel = np.abs(a.path_radiance - b.path_radiance)[cloudy] / np.maximum(
        a.path_radiance[cloudy], 1e-6
    )
    assert np.percentile(rel, 95) < 0.03
    assert np.percentile(np.abs(a.emission_range_m - b.emission_range_m)[cloudy], 95) < 40.0
    # A hit short of the slab meets no cloud on either side.
    short = cpu.march_to(el, az, 100.0, radiance), gpu.march_to(el, az, 100.0, radiance)
    assert np.all(short[0].transmittance == 1.0) and np.all(short[1].transmittance == 1.0)


def test_the_gpu_march_is_fast_enough_for_a_frame(decks: tuple[Any, Any]) -> None:
    _, gpu = decks
    el, az = _rays(512, 640)
    gpu.march_to(el, az, np.inf, lambda h: np.ones_like(h))  # warm the kernel
    start = time.perf_counter()
    gpu.march_to(el, az, np.inf, lambda h: np.ones_like(h))
    seconds = time.perf_counter() - start
    print(f"GPU march of {el.size} rays: {seconds:.2f} s")
    assert seconds < 20.0
