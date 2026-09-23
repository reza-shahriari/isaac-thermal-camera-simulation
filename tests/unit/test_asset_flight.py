"""The mount rotation and the flight path of an imported asset (AI.2).

Both are geometry a render driver would otherwise carry inline, and both fail silently: a mirrored
mount gives a convincing aircraft facing the wrong way, and a path that dips below the observer's
horizon puts near-air-temperature ground behind a warm target instead of cold sky.
"""

from __future__ import annotations

import numpy as np
import pytest

from irsim_isaac.asset_flight import FigureEightTrack, world_frame_to_stage

Z_UP_ENU = ((0.0, 0.0, 1.0), (0.0, 1.0, 0.0))
Y_UP_STAGE = ((0.0, 1.0, 0.0), (0.0, 0.0, -1.0))


def test_the_stage_s_own_frame_needs_no_rotation() -> None:
    """A scene already authored in stage axes must mount as the identity, not as a near-identity."""
    matrix = world_frame_to_stage(*Y_UP_STAGE)
    assert np.allclose(matrix, np.eye(3), atol=1e-12)


def test_a_z_up_enu_asset_lands_with_its_up_on_y_and_its_north_on_minus_z() -> None:
    matrix = world_frame_to_stage(*Z_UP_ENU)
    up, north = (np.asarray(v, dtype=np.float64) for v in Z_UP_ENU)
    assert np.allclose(matrix @ up, [0.0, 1.0, 0.0], atol=1e-12)
    assert np.allclose(matrix @ north, [0.0, 0.0, -1.0], atol=1e-12)
    # East is derived, and it is the axis a hand-written `rotateX 90` most often gets backwards.
    assert np.allclose(matrix @ np.cross(north, up), [1.0, 0.0, 0.0], atol=1e-12)


@pytest.mark.parametrize("frame", [Z_UP_ENU, Y_UP_STAGE, ((0.0, 0.0, 1.0), (1.0, 0.0, 0.0))])
def test_the_mount_is_a_rotation_and_never_a_mirror(frame: tuple) -> None:
    matrix = world_frame_to_stage(*frame)
    assert np.allclose(matrix @ matrix.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(matrix) == pytest.approx(1.0, abs=1e-12)


def test_a_north_tilted_out_of_the_horizontal_keeps_only_its_bearing() -> None:
    """Declaring north with a vertical component must not tilt the whole scene."""
    tilted = world_frame_to_stage((0.0, 0.0, 1.0), (0.0, 1.0, 0.4))
    assert np.allclose(tilted, world_frame_to_stage(*Z_UP_ENU), atol=1e-12)


def test_a_degenerate_world_frame_is_refused_rather_than_guessed() -> None:
    with pytest.raises(ValueError, match="parallel"):
        world_frame_to_stage((0.0, 0.0, 1.0), (0.0, 0.0, 3.0))
    with pytest.raises(ValueError, match="zero length"):
        world_frame_to_stage((0.0, 0.0, 0.0), (0.0, 1.0, 0.0))


def test_the_circuit_closes_so_a_looping_clip_does_not_jump() -> None:
    track = FigureEightTrack()
    assert np.allclose(track.position_m(0.0), track.position_m(1.0), atol=1e-9)


def test_the_eight_crosses_its_own_track_twice_at_the_centre() -> None:
    """A lemniscate, not an orbit: the quarter and three-quarter points are the same place."""
    track = FigureEightTrack()
    quarter, three_quarters = track.position_m(0.25), track.position_m(0.75)
    assert np.allclose(quarter[[0, 2]], three_quarters[[0, 2]], atol=1e-9)
    assert quarter[0] == pytest.approx(track.observer_m[0], abs=1e-9)
    assert quarter[2] == pytest.approx(track.observer_m[2] - track.centre_range_m, abs=1e-9)


def test_the_aspect_sweeps_a_full_turn_which_an_orbit_would_not() -> None:
    """The point of the shape. Yaw relative to the line of sight must cover both broadsides."""
    track = FigureEightTrack()
    phase = np.linspace(0.0, 1.0, 721, endpoint=False)
    position = track.position_m(phase)
    to_observer = np.asarray(track.observer_m) - position
    bearing = np.degrees(np.arctan2(to_observer[:, 0], -to_observer[:, 2]))
    aspect = (track.yaw_deg(phase) - bearing + 180.0) % 360.0 - 180.0
    # Both broadsides *and* tail-on. An orbit flown nose-first shows only +-90 for its whole
    # circuit, so 180 is the discriminator: it can only happen on a track that turns back.
    for wanted in (90.0, -90.0, 180.0):
        offset = (aspect - wanted + 180.0) % 360.0 - 180.0
        assert np.abs(offset).min() < 5.0, f"aspect {wanted} deg never occurs"


def test_the_range_swings_by_more_than_two_to_one() -> None:
    track = FigureEightTrack()
    ranges = track.range_m(np.linspace(0.0, 1.0, 721, endpoint=False))
    assert ranges.max() / ranges.min() > 2.0


def test_the_aircraft_never_drops_below_the_observer_s_horizon() -> None:
    """Below the horizon the background stops being sky, which is the whole scene's premise."""
    track = FigureEightTrack()
    elevation = track.elevation_deg(np.linspace(0.0, 1.0, 721, endpoint=False))
    assert elevation.min() > 5.0


def test_the_nose_follows_the_velocity_it_is_derived_from() -> None:
    track = FigureEightTrack()
    phase = np.linspace(0.0, 1.0, 37, endpoint=False)
    velocity = track.velocity(phase)
    nose = np.stack(
        [
            np.sin(np.radians(track.yaw_deg(phase))),
            np.zeros_like(phase),
            -np.cos(np.radians(track.yaw_deg(phase))),
        ],
        axis=-1,
    )
    horizontal = velocity.copy()
    horizontal[:, 1] = 0.0
    horizontal /= np.linalg.norm(horizontal, axis=-1, keepdims=True)
    assert np.allclose(nose, horizontal, atol=1e-6)
