"""EV.14: the aerial reference clip -- continuous video at standoff, engine-free (ADR 0168).

The clip itself is rendered in Isaac Sim (`render_quad_outbound.py --reference`); what can be
checked without the engine is everything that decides whether the clip *is* the reference the
roadmap asked for, before a single frame is spent on it:

  * the target is 5-40 px across through the shipped Boson for the whole clip, which is the size
    band of the public aerial sets EV.3-EV.7 compare against;
  * the capture interval is the sensor's own frame period, so ten seconds is 600 frames -- a
    continuous clip, not a time-lapse;
  * the companion drone stays below one pixel *and* inside the frame from the first frame to
    the last, so the analytic sub-pixel path (ADR 0071) runs on every frame and never refuses;
  * there is cloud behind the target: the scene's weather carries cumulus at the start time, its
    environment authors the deck's optical depth, and the deck's base is above the aircraft;
  * the aircraft is the outbound stage's, cell for cell, so `test_quad_outbound.py`'s agreement
    between patches and prims covers this scene too.
"""

from __future__ import annotations

import math
import pathlib
import re

import pytest
import yaml

from irsim.config.loader import load_sensor_config
from irsim.pipeline.point_target import fill_fraction
from irsim.radiometry.lut_files import load_band_lut_for_config, load_band_response_for_config
from irsim.scene import Scene
from irsim_isaac.quad_outbound import (
    AIM_ELEVATION_DEG,
    REFERENCE_DURATION_S,
    REFERENCE_FAR_M,
    REFERENCE_NEAR_M,
    TIP_TO_TIP_M,
    companion_angle_deg,
    distant_companion_position,
    reference_track,
)

REPO = pathlib.Path(__file__).resolve().parents[2]
SCENE = REPO / "configs" / "scenes" / "aerial_reference_clip.yaml"
OUTBOUND = REPO / "configs" / "scenes" / "quad_outbound_pointwise.yaml"
DRIVER = REPO / "scripts" / "render_quad_outbound.py"
BOSON = REPO / "configs" / "sensors" / "flir_boson_640_lwir.yaml"


def _driver_constant(name: str) -> float:
    match = re.search(rf"^{name} = ([0-9.]+)$", DRIVER.read_text(), re.MULTILINE)
    assert match, f"{name} is gone from the driver"
    return float(match.group(1))


@pytest.fixture(scope="module")
def boson():  # type: ignore[no-untyped-def]
    return load_sensor_config(BOSON).sensor


def _ifov_rad(spec) -> float:  # type: ignore[no-untyped-def]
    return spec.fpa.pitch_um * 1e-6 / (spec.optics.focal_length_mm * 1e-3)


def test_the_target_stays_in_the_public_sets_size_band_for_the_whole_clip(boson) -> None:  # type: ignore[no-untyped-def]
    track = reference_track()
    ifov_mrad = 1e3 * _ifov_rad(boson)
    sizes = [
        track.pixels_across(t, TIP_TO_TIP_M, ifov_mrad)
        for t in [REFERENCE_DURATION_S * k / 100 for k in range(101)]
    ]
    assert min(sizes) >= 5.0 and max(sizes) <= 40.0, (min(sizes), max(sizes))
    # geometric recession: the target halves in size in equal intervals, so it is not "small for
    # nine seconds and large for one"
    assert sizes[0] == pytest.approx(26.7, abs=0.1)
    assert sizes[-1] == pytest.approx(5.34, abs=0.05)
    assert track.range_at(0.0) == REFERENCE_NEAR_M
    assert track.range_at(REFERENCE_DURATION_S) == pytest.approx(REFERENCE_FAR_M)


def test_the_driver_flies_the_same_band_the_module_declares() -> None:
    # The driver sets its near/far before Kit boots, as literals; the module is the owner.
    src = DRIVER.read_text()
    assert f"args.near_m = {REFERENCE_NEAR_M}" in src
    assert f"args.far_m = {REFERENCE_FAR_M}" in src


def test_ten_seconds_is_six_hundred_frames_at_the_boson_s_own_rate(boson) -> None:  # type: ignore[no-untyped-def]
    interval = 1.0 / float(boson.fpa.frame_rate_hz)
    assert interval == pytest.approx(1.0 / 60.0)
    assert int(round(REFERENCE_DURATION_S / interval)) == 600
    # a track slower than a pixel a frame of background slew: the tracking mount follows the
    # drone, so the sky moves slowly and the target stays put
    track = reference_track()
    slew_rad_per_frame = math.radians(track.azimuth_sweep_deg) / REFERENCE_DURATION_S * interval
    assert slew_rad_per_frame / _ifov_rad(boson) < 1.0


def test_the_horizon_never_enters_the_reference_frame(boson) -> None:  # type: ignore[no-untyped-def]
    vfov_deg = math.degrees(2.0 * math.atan(0.5 * boson.fpa.height * _ifov_rad(boson)))
    assert not reference_track().sees_horizon(vfov_deg)
    assert reference_track().elevation_deg == AIM_ELEVATION_DEG


def test_the_companion_is_under_a_pixel_and_in_frame_on_every_frame(boson) -> None:  # type: ignore[no-untyped-def]
    track = reference_track()
    beyond = _driver_constant("COMPANION_BEYOND_M")
    offset = _driver_constant("COMPANION_OFFSET_DEG")
    area = _driver_constant("COMPANION_AREA_M2")
    where = distant_companion_position(track, beyond, offset)
    half_vfov = math.degrees(math.atan(0.5 * boson.fpa.height * _ifov_rad(boson)))
    for k in range(601):
        t = REFERENCE_DURATION_S * k / 600
        r, angle = companion_angle_deg(track, t, where)
        phi = fill_fraction(area, r, boson.optics.focal_length_mm * 1e-3, boson.pixel_area_m2)
        # IrCamera.point_targets refuses phi >= 1: the whole clip must be analytic, not rendered
        assert phi < 0.1, (t, phi)
        # inside the frame with at least two degrees to spare, whichever way it lies
        assert angle < half_vfov - 2.0, (t, angle)
        # and it is behind the aircraft all along, so the aircraft never occludes it by chance
        assert r > track.range_at(t) + 0.8 * beyond
    # it sits where it was put: at mid-clip the angle off the boresight is exactly the offset
    _, mid = companion_angle_deg(track, 0.5 * REFERENCE_DURATION_S, where)
    assert mid == pytest.approx(offset, abs=1e-9)


def test_the_companion_refuses_to_be_placed_in_front_of_the_aircraft() -> None:
    with pytest.raises(ValueError, match="behind the aircraft"):
        distant_companion_position(reference_track(), 0.0, 6.0)


def test_there_is_cloud_behind_the_target() -> None:
    sensor = load_sensor_config(BOSON)
    scene = Scene.from_file(
        SCENE,
        {"lwir": load_band_lut_for_config(sensor, REPO / "data" / "lut")},
        responses={"lwir": load_band_response_for_config(sensor, REPO / "data")},
        quantity=sensor.sensor.quantity,
    )
    fraction = float(scene.weather.at(scene.t0_s).cloud_fraction)
    assert fraction >= 0.3, fraction
    assert scene.environment is not None
    assert scene.environment.clouds.optical_depth is not None
    # the deck the infrared marches has its base above the highest the aircraft ever is over the
    # camera, so the cloud is behind it on every ray and never between it and the camera
    deck = scene.sky_models["lwir"].cloud_deck(scene.t0_s, 7)
    top = reference_track().target_altitude_m(REFERENCE_DURATION_S)
    assert deck.base_m > top, (deck.base_m, top)
    assert (deck.depth > 0).mean() > 0.2


def test_the_aircraft_is_the_outbound_stage_s_cell_for_cell() -> None:
    ours = yaml.safe_load(SCENE.read_text())["scene"]
    theirs = yaml.safe_load(OUTBOUND.read_text())["scene"]
    assert ours["thermal"] == theirs["thermal"]
    assert ours["targets"] == theirs["targets"]
    assert ours["world_frame"] == theirs["world_frame"]
    assert ours["weather_file"] != theirs["weather_file"]
