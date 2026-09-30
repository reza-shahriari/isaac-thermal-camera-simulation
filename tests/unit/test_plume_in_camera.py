"""PH.6's in-engine half: a scene's world plumes reach `run_frame` in the camera's own frame.

`irsim.pipeline.plume` works in the OpenCV camera frame; the Isaac camera stores a USD camera-to-
world matrix. `IrCamera.plumes()` joins the two with :func:`world_to_opencv_rotation`, and the one
thing that can go wrong there is a flipped axis -- a plume drawn mirrored, or behind the camera.
So the rotation is held to the camera's *existing* projection path, point for point: a plume's
apex must land exactly where `world_to_camera` + `usd_camera_to_opencv` put the same world point,
for arbitrary poses, and a plume straight ahead of a camera is at +z.
"""

from __future__ import annotations

import numpy as np
import pytest

from irsim.optics.projection import usd_camera_to_opencv
from irsim.pipeline.plume import WorldPlume
from irsim_isaac.pipeline.ir_camera import world_to_camera, world_to_opencv_rotation


def _random_pose(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1.0
    return q, rng.normal(scale=10.0, size=3)


def _plume(origin, direction) -> WorldPlume:  # type: ignore[no-untyped-def]
    return WorldPlume(
        origin_m=tuple(origin),
        direction=tuple(direction),
        length_m=1.5,
        radius_tip_m=0.045,
        radius_end_m=0.22,
        mixing_length_m=0.7,
        t_tip_k=600.0,
        t_air_k=290.0,
        p_co2_atm=0.11,
        p_h2o_atm=0.12,
    )


def test_the_rotation_agrees_with_the_camera_s_own_projection_path() -> None:
    rng = np.random.default_rng(20260930)
    for _ in range(20):
        camera_to_world, eye = _random_pose(rng)
        origin = rng.normal(scale=5.0, size=3)
        direction = rng.normal(size=3)
        cone = (
            _plume(origin, direction).in_camera(world_to_opencv_rotation(camera_to_world), eye).cone
        )
        expect = usd_camera_to_opencv(world_to_camera(origin, eye, camera_to_world))
        np.testing.assert_allclose(cone.apex_m, expect, atol=1e-9)
        # directions go through the rotation alone (no translation)
        tip = usd_camera_to_opencv(world_to_camera(origin + direction, eye, camera_to_world))
        axis = (tip - expect) / np.linalg.norm(tip - expect)
        np.testing.assert_allclose(cone.axis, axis, atol=1e-9)
        r = world_to_opencv_rotation(camera_to_world)
        assert np.linalg.det(r) == pytest.approx(1.0)


def test_a_plume_straight_ahead_is_in_front_and_centred() -> None:
    # USD camera looking down world -Z from the origin with +Y up: camera_to_world is identity
    cone = (
        _plume((0.0, 0.0, -8.0), (1.0, 0.0, 0.0))
        .in_camera(world_to_opencv_rotation(np.eye(3)), np.zeros(3))
        .cone
    )
    assert cone.apex_m == pytest.approx((0.0, 0.0, 8.0))  # +z is forward in OpenCV
    assert cone.axis == pytest.approx((1.0, 0.0, 0.0))  # world +X stays screen right
    up = (
        _plume((0.0, 1.0, -8.0), (0.0, 1.0, 0.0))
        .in_camera(world_to_opencv_rotation(np.eye(3)), np.zeros(3))
        .cone
    )
    assert up.apex_m[1] < 0.0 and up.axis[1] == pytest.approx(-1.0)  # world up is screen up
