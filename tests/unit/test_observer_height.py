"""AT.28 -- the camera has an altitude too.

`column_length` integrated from the surface, so a camera on a mast, a drone or an aircraft saw
surface-density air along its whole ray. The column now starts at the observer's height, a
descending ray gains density on the way down and stops at the ground, and the per-pixel plane
and the point-target path read the same thinner column. z_obs = 0 is bit for bit what it was.
docs/physics-model.md §7.1; roadmap AT.28, PT.28.
"""

from __future__ import annotations

import math
import pathlib

import numpy as np
import pytest
from scipy.integrate import quad

from irsim.atmosphere.layered import LayeredAtmosphere, column_length
from irsim.atmosphere.library import load_atmosphere_preset
from irsim.pipeline.atmosphere import apply_layered_gbuffer
from irsim.thermal.weather import WeatherSample, WeatherSeries

H = 2000.0


def _weather() -> WeatherSeries:
    return WeatherSeries.constant(
        WeatherSample(288.15, 0.46, 1.0, 0.0, 0.0, 0.0, 23000.0, 0.0), 3600.0
    )


def _column_by_quadrature(d: float, el: float, z0: float, h: float = H) -> float:
    st = math.sin(el)
    upper = d if st >= 0.0 else min(d, z0 / abs(st))
    return quad(lambda s: math.exp(-max(z0 + s * st, 0.0) / h), 0.0, upper, epsabs=0, epsrel=1e-12)[
        0
    ]


# --- the column -----------------------------------------------------------------------------------


def test_the_surface_camera_is_bit_identical() -> None:
    d = np.array([200.0, 5000.0, 20000.0, np.inf])
    for el in (0.0, -0.1, 0.05, 0.3, math.pi / 2):
        old = column_length(d, el, H)
        assert np.array_equal(column_length(d, el, H, 0.0), old)


@pytest.mark.parametrize("z0", [30.0, 3000.0])
@pytest.mark.parametrize("el_deg", [0.0, 1.0, 10.0, 45.0, 90.0, -1.0, -10.0, -45.0])
def test_air_to_air_matches_the_two_endpoint_integral_to_1e6(z0: float, el_deg: float) -> None:
    d = 5000.0
    el = math.radians(el_deg)
    expect = _column_by_quadrature(d, el, z0)
    got = float(column_length(d, el, H, z0))
    assert got == pytest.approx(expect, rel=1e-6), (z0, el_deg)


def test_a_ray_down_to_the_ground_is_denser_than_the_same_ray_up() -> None:
    z0, d = 3000.0, 4000.0
    down = float(column_length(d, math.radians(-30.0), H, z0))
    up = float(column_length(d, math.radians(30.0), H, z0))
    level = float(column_length(d, 0.0, H, z0))
    assert down > level > up
    assert level == pytest.approx(d * math.exp(-z0 / H), rel=1e-12), "horizontal at height"
    # and the descending ray stops at the ground: further range adds nothing
    far = float(column_length(40000.0, math.radians(-30.0), H, z0))
    assert far == pytest.approx(
        float(column_length(z0 / 0.5, math.radians(-30.0), H, z0)), rel=1e-12
    )
    with pytest.raises(ValueError):
        column_length(d, 0.0, H, -1.0)


# --- the model ------------------------------------------------------------------------------------

REPO = pathlib.Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def atmosphere() -> LayeredAtmosphere:
    from irsim.config.loader import load_sensor_config
    from irsim.radiometry.lut_files import load_band_lut_for_config

    sensor = load_sensor_config("boson640")
    lut = load_band_lut_for_config(sensor, REPO / "data" / "lut")
    return LayeredAtmosphere(
        load_atmosphere_preset("us_standard_clear"), _weather(), {"lwir": lut}, None
    )


def test_transmittance_from_altitude_is_higher_and_at_zero_unchanged(atmosphere) -> None:  # type: ignore[no-untyped-def]
    d = np.array([200.0, 5000.0])
    for el in (0.0, 0.3, -0.3):
        surface = atmosphere.transmittance("lwir", 0.0, d, el)
        assert np.array_equal(atmosphere.transmittance("lwir", 0.0, d, el, 0.0), surface)
    high = atmosphere.transmittance("lwir", 0.0, d, 0.0, 3000.0)
    assert np.all(high > atmosphere.transmittance("lwir", 0.0, d, 0.0))
    per_class = atmosphere.class_transmittances("lwir", 0.0, 5000.0, 0.0, 3000.0)
    assert np.all(per_class >= atmosphere.class_transmittances("lwir", 0.0, 5000.0, 0.0))


def test_path_radiance_from_altitude_is_lower_and_the_plane_agrees_with_the_point_path(
    atmosphere,
) -> None:  # type: ignore[no-untyped-def]
    """A camera at 3 km looking 45° down 5 km sees thinner, colder air than the surface tables
    say; the per-pixel plane (each class on its own equal-loss grid) and the exact quadrature
    the point-target path uses agree to a few tenths of a percent."""
    d, el = 5000.0, math.radians(-45.0)
    surface = atmosphere.path_radiance("lwir", 0.0, d, math.radians(45.0))
    high = atmosphere.path_radiance("lwir", 0.0, d, el, observer_height_m=3000.0)
    assert 0.0 < high < surface
    plane = atmosphere.path_radiance_plane("lwir", 0.0, np.array([d]), np.array([el]), "lb", 3000.0)
    assert float(plane[0]) == pytest.approx(high, rel=3.5e-3)
    level = atmosphere.path_radiance("lwir", 0.0, 5000.0, 0.0, observer_height_m=3000.0)
    plane_level = atmosphere.path_radiance_plane(
        "lwir", 0.0, np.array([5000.0]), np.array([0.0]), "lb", 3000.0
    )
    assert float(plane_level[0]) == pytest.approx(level, rel=1e-9), "horizontal: exact"
    beyond = atmosphere.sky_beyond_per_class("lwir", 0.0, d, math.radians(20.0), "lb", 3000.0)
    assert np.all(beyond >= 0.0)


def test_the_stage_takes_the_height_and_defaults_to_the_surface(atmosphere) -> None:  # type: ignore[no-untyped-def]
    radiance = np.full((4, 4), 8.0, dtype=np.float32)
    distance = np.full((4, 4), 3000.0, dtype=np.float32)
    elevation = np.full((4, 4), 0.2, dtype=np.float32)
    base = apply_layered_gbuffer(
        atmosphere, "lwir", 0.0, radiance, distance, "lb", elevation_rad=elevation
    )
    same = apply_layered_gbuffer(
        atmosphere,
        "lwir",
        0.0,
        radiance,
        distance,
        "lb",
        elevation_rad=elevation,
        observer_height_m=0.0,
    )
    assert np.array_equal(base, same)
    high = apply_layered_gbuffer(
        atmosphere,
        "lwir",
        0.0,
        radiance,
        distance,
        "lb",
        elevation_rad=elevation,
        observer_height_m=2000.0,
    )
    assert high.dtype == np.float32 and not np.array_equal(high, base)


@pytest.mark.parametrize(
    "z0,d,el_deg",
    [
        (3000.0, 5000.0, -20.0),
        (3000.0, 5000.0, 20.0),
        (300.0, 5000.0, -3.0),
        (1000.0, 20000.0, -10.0),
        (3000.0, math.inf, 30.0),
        (3000.0, math.inf, -30.0),
        (3000.0, math.inf, 0.0),
        (3000.0, 60000.0, 5.0),
        (8000.0, math.inf, -5.0),
        (8000.0, math.inf, 90.0),
        (200.0, math.inf, 1.0),
        (10000.0, math.inf, -1.0),
        (50.0, 200.0, -80.0),
    ],
)
def test_the_plane_tracks_the_exact_path_across_geometries(atmosphere, z0, d, el_deg) -> None:  # type: ignore[no-untyped-def]
    """Every geometry an aerial camera meets -- short and infinite rays, steep and grazing, up
    and down, from a mast to an airliner -- within 0.6 % of the 4000-step quadrature. The
    isothermal-at-mean-height plane tried first missed several of these by 1-4 %, and equal
    pieces of *distance* missed the 5° descent from 8 km by 3 %."""
    el = math.radians(el_deg)
    exact = atmosphere.path_radiance("lwir", 0.0, d, el, observer_height_m=z0)
    plane = float(
        atmosphere.path_radiance_plane("lwir", 0.0, np.array([d]), np.array([el]), "lb", z0)[0]
    )
    assert math.isfinite(plane) and plane > 0.0
    assert plane == pytest.approx(exact, rel=6e-3), (exact, plane)
