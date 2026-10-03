"""EV.16: the clear-sky exit clip -- a quadrotor out to 150 m, then out of the frame, engine-free.

The clip itself is rendered in Isaac Sim (`render_quad_outbound.py --clear-exit`); what can be
checked without the engine is everything that decides whether the clip shows what it claims:

  * **the sky is cloudless, not nearly so.** The inland weather's 0.05 is blended into every clear
    ray by `SkyModel.radiance`, so a "clear" render on it still carries a twentieth of a cloud.
    The fixture here is that day with the cloud set to zero and nothing else changed, and the sky
    radiance is checked to be colder for it -- a test that would pass on the old file would not
    be testing the switch;
  * **the aircraft is the outbound stage's**, cell for cell, so `test_quad_outbound.py`'s patch and
    prim agreement covers this scene too;
  * **the exit is where the track says.** The aircraft sits on the boresight while the mount
    tracks, crosses the left edge during the pan, and ends beyond it by half its own span plus the
    PSF margin. The camera-frame angles are checked against the quaternion the stage actually
    sets, so a sign slip in either cannot hide behind the other;
  * the horizon stays out of frame through the pan, and the driver's literals are the module's.
"""

from __future__ import annotations

import importlib.util
import math
import pathlib
import re

import numpy as np
import pytest
import yaml

from irsim.atmosphere.layered import LayeredAtmosphere
from irsim.atmosphere.library import load_atmosphere_preset
from irsim.atmosphere.sky import SkyModel
from irsim.config.environment import load_environment_preset
from irsim.config.loader import load_sensor_config
from irsim.config.scene import load_scene_config
from irsim.radiometry.lut_files import load_band_lut_for_config, load_band_response_for_config
from irsim.thermal.weather_io import load_weather_csv, write_weather_csv
from irsim_isaac.aircraft_pass import look_at_quaternion
from irsim_isaac.quad_outbound import (
    AIM_POINT_M,
    CLEAR_EXIT_FAR_M,
    CLEAR_EXIT_HOLD_S,
    CLEAR_EXIT_MARGIN_PX,
    CLEAR_EXIT_NEAR_M,
    CLEAR_EXIT_OUTBOUND_S,
    CLEAR_EXIT_SPEED_M_S,
    TIP_TO_TIP_M,
    OutboundTrack,
    boresight,
    clear_exit_track,
    look_direction,
    target_image_angles_deg,
)

REPO = pathlib.Path(__file__).resolve().parents[2]
SCENE = REPO / "configs" / "scenes" / "aerial_clear_exit.yaml"
OUTBOUND = REPO / "configs" / "scenes" / "quad_outbound_pointwise.yaml"
DRIVER = REPO / "scripts" / "render_quad_outbound.py"
BOSON = REPO / "configs" / "sensors" / "flir_boson_640_lwir.yaml"
CLOUDLESS = REPO / "data" / "weather" / "cloudless_midlat_summer_48h.csv"
INLAND = REPO / "data" / "weather" / "clear_midlat_summer_48h.csv"


@pytest.fixture(scope="module")
def boson():  # type: ignore[no-untyped-def]
    return load_sensor_config(BOSON).sensor


def _fields(spec) -> tuple[float, float, float]:  # type: ignore[no-untyped-def]
    """(ifov mrad, hfov deg, vfov deg) of a sensor, from its own pitch and focal length."""
    ifov = spec.fpa.pitch_um * 1e-6 / (spec.optics.focal_length_mm * 1e-3)
    hfov = math.degrees(2.0 * math.atan(0.5 * spec.fpa.width * ifov))
    vfov = math.degrees(2.0 * math.atan(0.5 * spec.fpa.height * ifov))
    return 1e3 * ifov, hfov, vfov


@pytest.fixture(scope="module")
def track(boson) -> OutboundTrack:  # type: ignore[no-untyped-def]
    ifov, hfov, _ = _fields(boson)
    return clear_exit_track(hfov, ifov)


def _sky(path: pathlib.Path) -> tuple[SkyModel, float]:
    """The scene's LWIR sky and start time, built the way `Scene.from_config` builds it.

    Not through `Scene` itself: that spins up every point-wise field for 48 h, which is twenty
    seconds a scene, and the sky does not depend on any of it.
    """
    sensor = load_sensor_config(BOSON)
    spec = load_scene_config(path).scene
    weather = load_weather_csv(REPO / "data" / spec.weather_file)
    lut = load_band_lut_for_config(sensor, REPO / "data" / "lut")
    layered = LayeredAtmosphere(
        load_atmosphere_preset(spec.atmosphere_preset),
        weather,
        {"lwir": lut},
        {"lwir": load_band_response_for_config(sensor, REPO / "data")},
    )
    sky = SkyModel(
        layered,
        load_environment_preset(spec.environment_preset),
        "lwir",
        lut,
        sensor.sensor.quantity,
    )
    return sky, weather.seconds_of(spec.start_utc)


# -- the sky -------------------------------------------------------------------------------------


def test_the_cloudless_fixture_is_its_generator_s_bit_for_bit(tmp_path: pathlib.Path) -> None:
    spec = importlib.util.spec_from_file_location(
        "gen_weather", REPO / "scripts/generate_weather.py"
    )
    assert spec is not None and spec.loader is not None
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    fresh = gen.build_cloudless()
    out = tmp_path / "regen.csv"
    write_weather_csv(fresh, out, comment=gen.CLOUDLESS_COMMENT)
    assert out.read_text() == CLOUDLESS.read_text()


def test_the_cloudless_day_is_the_inland_day_with_no_cloud() -> None:
    ours, theirs = load_weather_csv(CLOUDLESS), load_weather_csv(INLAND)
    assert np.all(ours.cloud_fraction == 0.0)
    assert np.all(theirs.cloud_fraction > 0.0)  # the reason this file exists
    for name in ("t_air_k", "rh_fraction", "wind_speed_m_s", "dni_w_m2", "dhi_w_m2"):
        np.testing.assert_array_equal(getattr(ours, name), getattr(theirs, name), err_msg=name)


def test_the_sky_behind_the_target_is_colder_without_the_twentieth_of_a_cloud() -> None:
    # The same day, the same elevations: only the cloud blend differs, and a cloud base is warmer
    # than the clear sky, so removing it must make every ray colder. If the scene still read the
    # inland file, or the blend were ignored, this would be equal.
    (ours, t0), (theirs, t0_theirs) = _sky(SCENE), _sky(OUTBOUND)
    assert t0 == t0_theirs
    elevations = np.radians(np.array([8.0, 20.0, 32.0, 60.0]))
    clear = np.asarray(ours.radiance(t0, elevations))
    hazed = np.asarray(theirs.radiance(t0, elevations))
    assert np.all(clear < hazed), (clear, hazed)
    assert float(ours.weather.at(t0).cloud_fraction) == 0.0


def test_the_aircraft_is_the_outbound_stage_s_cell_for_cell() -> None:
    ours = yaml.safe_load(SCENE.read_text())["scene"]
    theirs = yaml.safe_load(OUTBOUND.read_text())["scene"]
    assert ours["thermal"] == theirs["thermal"]
    assert ours["targets"] == theirs["targets"]
    assert ours["world_frame"] == theirs["world_frame"]
    assert ours["start_utc"] == theirs["start_utc"]
    assert ours["environment_preset"] == theirs["environment_preset"] == "clear_dry"
    assert ours["weather_file"] == "weather/cloudless_midlat_summer_48h.csv"


# -- the track -----------------------------------------------------------------------------------


def test_the_mount_tracks_until_the_far_end_and_then_lets_go(track: OutboundTrack) -> None:
    assert track.range_at(0.0) == CLEAR_EXIT_NEAR_M
    assert track.range_at(track.duration_s) == pytest.approx(CLEAR_EXIT_FAR_M)
    for t in np.linspace(0.0, track.duration_s, 31):
        assert track.pan_deg_at(t) == 0.0
        assert look_direction(track, t) == boresight(track, t)
        h, v = target_image_angles_deg(track, t)
        assert abs(h) < 1e-9 and abs(v) < 1e-9, (t, h, v)
    # during the exit the camera stands still and only turns
    end = track.duration_s + track.exit_s
    still = track.camera_position_m(track.duration_s)
    pans = [track.pan_deg_at(t) for t in np.linspace(track.duration_s, track.end_s, 41)]
    assert all(b >= a for a, b in zip(pans, pans[1:], strict=False))
    assert track.camera_position_m(end) == still
    assert track.pan_deg_at(track.end_s) == track.exit_pan_deg


def test_the_aircraft_leaves_by_the_left_edge_and_is_wholly_gone(boson, track) -> None:  # type: ignore[no-untyped-def]
    ifov, hfov, _ = _fields(boson)
    edge = 0.5 * hfov
    half = math.degrees(math.atan(0.5 * TIP_TO_TIP_M / CLEAR_EXIT_FAR_M))
    margin = CLEAR_EXIT_MARGIN_PX * ifov * 1e-3 * 180.0 / math.pi
    h_end, _ = target_image_angles_deg(track, track.duration_s + track.exit_s)
    # a right pan sends the aircraft left, and the pan is the smallest that clears it
    assert h_end < 0.0
    assert -h_end == pytest.approx(edge + half + margin, abs=1e-6)
    # it is still partly in frame a frame before the end of the exit: the crossing is filmed
    crossing = [
        target_image_angles_deg(track, t)[0]
        for t in np.linspace(track.duration_s, track.duration_s + track.exit_s, 200)
    ]
    straddling = [h for h in crossing if edge - half < -h < edge + half]
    assert len(straddling) >= 5, len(straddling)


def test_the_crossing_rate_is_a_drone_flying_across_the_line_of_sight(track) -> None:  # type: ignore[no-untyped-def]
    # azimuth rate v / (R cos e): the horizontal distance to the aircraft is R cos e, not R
    rate = math.degrees(
        CLEAR_EXIT_SPEED_M_S / (CLEAR_EXIT_FAR_M * math.cos(math.radians(track.elevation_deg)))
    )
    assert track.exit_pan_deg / track.exit_s == pytest.approx(rate, rel=1e-12)
    assert track.hold_s == CLEAR_EXIT_HOLD_S
    assert track.duration_s == CLEAR_EXIT_OUTBOUND_S


def test_the_horizon_stays_out_of_frame_through_the_pan(boson, track) -> None:  # type: ignore[no-untyped-def]
    _, _, vfov = _fields(boson)
    assert not track.sees_horizon(vfov)
    for t in np.linspace(track.duration_s, track.end_s, 21):
        d = look_direction(track, t)
        assert math.degrees(math.asin(d[1])) == pytest.approx(track.elevation_deg, abs=1e-9)


@pytest.mark.parametrize("fraction", [0.0, 0.3, 0.7, 1.0])
def test_the_image_angles_agree_with_the_quaternion_the_stage_sets(track, fraction) -> None:  # type: ignore[no-untyped-def]
    # Independent of `target_image_angles_deg`'s own basis: rotate the camera-to-aircraft vector
    # into the camera frame with the quaternion `aim_camera` writes, where USD looks along -Z.
    t = track.duration_s + fraction * track.exit_s
    w, x, y, z = look_at_quaternion(np.asarray(look_direction(track, t)))
    rot = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )
    v = np.asarray(AIM_POINT_M) - np.asarray(track.camera_position_m(t))
    cx, cy, cz = rot.T @ v
    h, vv = target_image_angles_deg(track, t)
    assert h == pytest.approx(math.degrees(math.atan2(cx, -cz)), abs=1e-9)
    assert vv == pytest.approx(math.degrees(math.atan2(cy, -cz)), abs=1e-9)


def test_the_aircraft_is_resolved_at_both_ends_of_the_outbound_leg(boson, track) -> None:  # type: ignore[no-untyped-def]
    ifov, _, _ = _fields(boson)
    near = track.pixels_across(0.0, TIP_TO_TIP_M, ifov)
    far = track.pixels_across(track.duration_s, TIP_TO_TIP_M, ifov)
    assert near < boson.fpa.width / 4  # in frame with room round it at 12 m
    assert far > 5.0  # still a shape, not a point, when it leaves
    assert near == pytest.approx(111.3, abs=0.1)
    assert far == pytest.approx(8.90, abs=0.01)


def test_the_clip_is_continuous_and_ends_on_empty_sky(boson, track) -> None:  # type: ignore[no-untyped-def]
    interval = 1.0 / float(boson.fpa.frame_rate_hz)
    frames = int(math.ceil(track.end_s / interval)) + 1  # the driver's own count
    empty = sum(1 for k in range(frames) if k * interval > track.duration_s + track.exit_s + 1e-9)
    assert empty >= int(CLEAR_EXIT_HOLD_S / interval) - 1, empty
    assert frames == 1126


def test_an_exit_needs_both_a_duration_and_a_pan() -> None:
    with pytest.raises(ValueError, match="both a duration and a positive pan"):
        OutboundTrack(exit_s=2.0)
    with pytest.raises(ValueError, match="both a duration and a positive pan"):
        OutboundTrack(exit_pan_deg=10.0)
    with pytest.raises(ValueError, match="negative"):
        OutboundTrack(hold_s=-1.0)
    with pytest.raises(ValueError, match="exit_speed_m_s"):
        clear_exit_track(30.0, 0.857, exit_speed_m_s=0.0)


def test_the_driver_flies_the_band_the_module_declares() -> None:
    src = DRIVER.read_text()
    assert f"args.near_m = {CLEAR_EXIT_NEAR_M}" in src
    assert f"args.far_m = {CLEAR_EXIT_FAR_M}" in src
    for flag, value in (
        ("--outbound-s", CLEAR_EXIT_OUTBOUND_S),
        ("--exit-speed-m-s", CLEAR_EXIT_SPEED_M_S),
        ("--hold-s", CLEAR_EXIT_HOLD_S),
    ):
        match = re.search(rf'"{flag}",\s*type=float,\s*default=([0-9.]+)', src)
        assert match, flag
        assert float(match.group(1)) == value, flag
