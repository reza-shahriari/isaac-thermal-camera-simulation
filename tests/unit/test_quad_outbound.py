"""The outbound quadrotor stage, engine-free (PT.9/IG.2, ADR 0123).

Two things are checked here that nothing else can check, because they are agreements *between*
two files rather than properties of either one:

  * every patch cell and every occluder face in ``quad_outbound_pointwise.yaml`` lies on the prim
    :mod:`irsim_isaac.quad_outbound` authors for it. A patch that drifts off its prim does not
    fail -- ``PointwiseTemperature`` leaves those pixels at the prim's per-prim fallback and the
    frame looks entirely normal, with part of an airframe at one temperature. That is the failure
    mode ADR 0087 exists to remove, so it is asserted cell by cell.
  * the camera's boresight keeps the horizon out of frame at every sensor this project ships. The
    scene has no ground plane, so ground in the picture would not be ground -- it would be the
    analytic ``T_ground`` of ADR 0060 painted across the bottom of a sky-target frame.

The physics is checked against `quad_flight_pointwise.yaml`'s measured numbers: this scene is that
one carried into the stage frame, and if the port moved the answer the port is wrong.
"""

from __future__ import annotations

import math
import pathlib

import numpy as np
import pytest

from irsim.scene import Scene
from irsim_isaac.quad_outbound import (
    AIM_ELEVATION_DEG,
    AIM_POINT_M,
    POINTWISE_QUAD,
    SPAN_M,
    OutboundTrack,
    rotor_mounts,
    rotor_rpm,
)

REPO = pathlib.Path(__file__).resolve().parents[2]
SCENE_PATH = REPO / "configs" / "scenes" / "quad_outbound_pointwise.yaml"
QUAD_ROOT = "/World/Targets/quad"

#: The phases the engine-free oracle reports, and this scene has to agree with.
PHASES = ((0.0, "on the pad"), (400.0, "climbing"), (1000.0, "hard climb"), (1750.0, "landed"))


@pytest.fixture(scope="module")
def scene() -> Scene:
    assert SCENE_PATH.exists(), f"{SCENE_PATH} is gone; this test names it deliberately"
    return Scene.from_file(SCENE_PATH)


def _boxes() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Each prim's axis-aligned bounds in the stage frame. No part here is rotated."""
    out = {}
    for part in POINTWISE_QUAD:
        assert part.rotate_xyz_deg == (0.0, 0.0, 0.0), (
            f"{part.name} is rotated, so a bounding box is no longer its extent and the "
            "containment checks below would silently pass on the wrong volume"
        )
        centre = np.asarray(part.centre_m, dtype=np.float64)
        half = 0.5 * np.asarray(part.size_m, dtype=np.float64)
        out[f"{QUAD_ROOT}/{part.name}"] = (centre - half, centre + half)
    return out


def test_every_patch_binds_to_a_prim_that_exists(scene: Scene) -> None:
    boxes = _boxes()
    assert scene.patch_prims, "the scene declares no patch prims at all"
    for name, path in scene.patch_prims.items():
        assert path in boxes, (
            f"surface {name!r} binds to {path!r}, which this stage does not author. "
            f"authored: {sorted(boxes)}"
        )


def test_every_patch_cell_lies_inside_its_own_prim(scene: Scene) -> None:
    """The agreement that matters: a cell off its prim is a fallback pixel, not a failure."""
    boxes = _boxes()
    for name, path in scene.patch_prims.items():
        patch = scene.patches[name]
        lo, hi = boxes[path]
        origin = np.asarray(patch.origin_m, dtype=np.float64)
        u = np.asarray(patch.u_axis, dtype=np.float64)
        v = np.asarray(patch.v_axis, dtype=np.float64)
        u = u / np.linalg.norm(u)
        v = v / np.linalg.norm(v)
        iu, iv = np.meshgrid(np.arange(patch.n_u), np.arange(patch.n_v), indexing="xy")
        centres = (
            origin
            + ((iu + 0.5) * patch.du_m)[..., None] * u
            + ((iv + 0.5) * patch.dv_m)[..., None] * v
        )
        # Inside the prim's box, allowing the patch's own slab half-thickness off the plane: the
        # cell centre sits on the plane, the pixel it claims sits on the prim's surface.
        slack = patch.thickness_m
        inside = np.all(centres >= lo - slack - 1e-9, axis=-1) & np.all(
            centres <= hi + slack + 1e-9, axis=-1
        )
        assert inside.all(), (
            f"{inside.size - int(inside.sum())} of {inside.size} cells of {name!r} fall outside "
            f"{path!r} (bounds {lo} .. {hi})"
        )
        # And the plane itself is inside the prim's thickness, not merely within a slab of it.
        assert (lo[1] - 1e-9) <= origin[1] <= (hi[1] + 1e-9), (
            f"{name!r}'s plane at y = {origin[1]} is not inside {path!r} (y {lo[1]}..{hi[1]})"
        )


#: How far inside its patch every patched prim's surface must sit, metres. Measured, not chosen:
#: a prim exactly the size of its patch loses its edge pixels -- the sampled surface position
#: lands a float's width outside the rectangle and `PointwiseTemperature` refuses the frame, which
#: is what the first render of this scene did, on 150 deck pixels. 2 mm is a fifth of a pixel at
#: the near end of the track.
PRIM_INSET_M = 0.002


def test_every_patched_prim_is_strictly_inside_its_patch(scene: Scene) -> None:
    """The margin that makes the render possible, asserted where it can be read."""
    boxes = _boxes()
    for name, path in scene.patch_prims.items():
        patch = scene.patches[name]
        lo, hi = boxes[path]
        origin = np.asarray(patch.origin_m, dtype=np.float64)
        u = np.asarray(patch.u_axis, dtype=np.float64)
        v = np.asarray(patch.v_axis, dtype=np.float64)
        u = u / np.linalg.norm(u)
        v = v / np.linalg.norm(v)
        far = origin + patch.n_u * patch.du_m * u + patch.n_v * patch.dv_m * v
        p_lo = np.minimum(origin, far)
        p_hi = np.maximum(origin, far)
        # Only the two in-plane axes: the third is the patch's slab, checked above.
        in_plane = np.abs(u) + np.abs(v) > 0.5
        margin_lo = (lo - p_lo)[in_plane]
        margin_hi = (p_hi - hi)[in_plane]
        assert margin_lo.min() >= PRIM_INSET_M - 1e-9, f"{path} reaches {name!r}'s low edge"
        assert margin_hi.min() >= PRIM_INSET_M - 1e-9, f"{path} reaches {name!r}'s high edge"


def test_a_displaced_patch_is_caught(scene: Scene) -> None:
    """The negative control: the containment check above is not vacuous."""
    boxes = _boxes()
    lo, hi = boxes[f"{QUAD_ROOT}/deck"]
    moved = np.asarray(scene.patches["deck"].origin_m, dtype=np.float64) + np.array([1.0, 0.0, 0.0])
    assert not (np.all(moved >= lo) and np.all(moved <= hi))


#: How far an occluder's centre may sit from the nearest prim's surface, metres. The body box is
#: inset 2 mm inside the deck and belly plates so that no two faces are coplanar -- coincident
#: faces z-fight, and a flickering surface in the companion visible frame is indistinguishable
#: from a rendering fault in the infrared one. The occluders describe the aircraft's silhouette,
#: which is the plates' 0.30 m, so the body's own walls are exactly that inset away from them.
OCCLUDER_TOLERANCE_M = 0.005


def test_every_occluder_face_sits_on_a_prim(scene: Scene) -> None:
    """Each shadow caster is a face of a prim, so the shadow and the picture agree."""
    boxes = _boxes()
    assert scene.spec.thermal.occluders, "the scene declares no occluders"
    for occ in scene.spec.thermal.occluders:
        centre = np.asarray(occ.centre_m, dtype=np.float64)
        gaps = {
            path: float(np.linalg.norm(np.maximum(np.maximum(lo - centre, centre - hi), 0.0)))
            for path, (lo, hi) in boxes.items()
        }
        nearest = min(gaps, key=lambda k: gaps[k])
        assert gaps[nearest] <= OCCLUDER_TOLERANCE_M, (
            f"occluder {occ.name!r} at {centre} is {gaps[nearest] * 1e3:.1f} mm from the nearest "
            f"prim ({nearest}); it casts a shadow from somewhere the aircraft is not"
        )


def test_an_occluder_adrift_is_caught() -> None:
    """The negative control for the tolerance above: 5 mm is a gap, not a free pass."""
    boxes = _boxes()
    adrift = np.array([-0.15, 0.30, 0.0]) + np.array([0.0, 0.0, 0.40])
    gaps = [
        float(np.linalg.norm(np.maximum(np.maximum(lo - adrift, adrift - hi), 0.0)))
        for lo, hi in boxes.values()
    ]
    assert min(gaps) > OCCLUDER_TOLERANCE_M


def test_the_thermal_nodes_the_scene_defines_cover_every_prim(scene: Scene) -> None:
    needed = {part.thermal_node for part in POINTWISE_QUAD}
    assert needed <= set(scene.targets), f"no solver for {sorted(needed - set(scene.targets))}"


# --------------------------------------------------------------------------------------------
# the track
# --------------------------------------------------------------------------------------------


def test_range_is_monotone_and_geometric() -> None:
    track = OutboundTrack(near_m=12.0, far_m=150.0, duration_s=1800.0)
    ts = np.linspace(0.0, track.duration_s, 200)
    ranges = np.array([track.range_at(float(t)) for t in ts])
    assert np.all(np.diff(ranges) > 0.0), "an outbound track never comes closer"
    assert ranges[0] == pytest.approx(12.0)
    assert ranges[-1] == pytest.approx(150.0)
    # Geometric: the halfway point is the geometric mean, not the arithmetic one (81.0).
    assert track.range_at(900.0) == pytest.approx(math.sqrt(12.0 * 150.0), rel=1e-12)


def test_range_is_clamped_outside_the_track() -> None:
    track = OutboundTrack()
    assert track.range_at(-100.0) == pytest.approx(track.near_m)
    assert track.range_at(1e6) == pytest.approx(track.far_m)


def test_angular_size_falls_as_one_over_range() -> None:
    track = OutboundTrack(near_m=12.0, far_m=150.0)
    ifov = 0.857
    near = track.pixels_across(0.0, SPAN_M, ifov)
    far = track.pixels_across(track.duration_s, SPAN_M, ifov)
    assert near == pytest.approx(1e3 * SPAN_M / 12.0 / ifov)
    assert near / far == pytest.approx(150.0 / 12.0, rel=1e-12)
    # The whole reason for the clip: a target that is resolved at the start is not at the end.
    assert near > 50.0 and far < 10.0


def test_the_camera_stands_at_the_declared_range_and_elevation() -> None:
    track = OutboundTrack()
    aim = np.asarray(AIM_POINT_M, dtype=np.float64)
    for t in (0.0, 300.0, 900.0, 1800.0):
        eye = np.asarray(track.camera_position_m(t), dtype=np.float64)
        to_target = aim - eye
        assert float(np.linalg.norm(to_target)) == pytest.approx(track.range_at(t), rel=1e-12)
        elevation = math.degrees(math.asin(float(to_target[1]) / float(np.linalg.norm(to_target))))
        assert elevation == pytest.approx(track.elevation_deg, abs=1e-9)
        assert track.target_altitude_m(t) == pytest.approx(
            track.range_at(t) * math.sin(math.radians(track.elevation_deg))
        )


def test_the_horizon_is_never_in_frame_for_this_projects_cameras() -> None:
    """A 640x512 Boson through its 14 mm lens is the widest vertical field shipped here."""
    track = OutboundTrack()
    vfov_deg = math.degrees(2.0 * math.atan(0.5 * 512 * 12e-3 / 14.0))
    assert vfov_deg == pytest.approx(25.0, abs=0.5)
    assert not track.sees_horizon(vfov_deg)
    assert track.elevation_deg - 0.5 * vfov_deg > 5.0, "less than 5 deg of margin to the horizon"
    # The negative control: a genuinely wide camera does see it, and the driver must refuse.
    assert track.sees_horizon(2.0 * AIM_ELEVATION_DEG + 1.0)


def test_a_track_that_goes_the_wrong_way_is_refused() -> None:
    with pytest.raises(ValueError, match="outbound"):
        OutboundTrack(near_m=150.0, far_m=12.0)
    with pytest.raises(ValueError, match="zenith"):
        OutboundTrack(elevation_deg=95.0)


def test_rotor_speed_and_mounts_follow_the_throttle() -> None:
    assert rotor_rpm(0.0) < rotor_rpm(0.5) < rotor_rpm(1.0)
    assert rotor_rpm(2.0) == pytest.approx(rotor_rpm(1.0)), "throttle is clamped"
    mounts = rotor_mounts(0.8)
    assert set(mounts) == {QUAD_ROOT}
    discs = mounts[QUAD_ROOT]
    assert len(discs) == 4
    assert all(m.rpm == pytest.approx(rotor_rpm(0.8)) for m in discs)
    # Adjacent discs must not intersect: motors are 0.594 m apart on a 0.84 m frame.
    centres = np.array([m.offset_m for m in discs])
    gaps = [
        float(np.linalg.norm(centres[i] - centres[j])) for i in range(4) for j in range(i + 1, 4)
    ]
    assert min(gaps) > 2.0 * discs[0].disc.radius_m, "adjacent rotor discs overlap"


# --------------------------------------------------------------------------------------------
# the physics, against the oracle this scene was ported from
# --------------------------------------------------------------------------------------------


def _mean_c(scene: Scene, name: str, t_rel: float) -> float:
    field = scene.surface_fields[name]
    field.advance_to(scene.t0_s + t_rel)
    return float(np.mean(np.asarray(field.temperature_at(scene.t0_s + t_rel)))) - 273.15


def _span_k(scene: Scene, name: str, t_rel: float) -> float:
    field = scene.surface_fields[name]
    field.advance_to(scene.t0_s + t_rel)
    t = np.asarray(field.temperature_at(scene.t0_s + t_rel))
    return float(t.max() - t.min())


def test_the_deck_and_the_belly_are_not_the_same_surface(scene: Scene) -> None:
    """The headline of PT.9: one airframe, two temperatures, 29 K apart on the pad."""
    deck = _mean_c(scene, "deck", 0.0)
    belly = _mean_c(scene, "belly", 0.0)
    assert deck - belly > 20.0, f"deck {deck:.2f} C, belly {belly:.2f} C"
    # The belly sees no sky at all (tilt 180), so it sits within a couple of kelvin of the air --
    # which is why a drone read from below is nothing like the same drone read from above.
    air = float(scene.weather.at(scene.t0_s).t_air_k) - 273.15
    assert abs(belly - air) < 2.0


def test_an_arm_carries_the_decks_own_shadow(scene: Scene) -> None:
    """A gradient across one prim, from occluders that are faces of other prims."""
    for arm in ("arm_n", "arm_e"):
        assert _span_k(scene, arm, 0.0) > 10.0, f"{arm} is flat; the occluders are not biting"
    # The deck is on top of everything and is shaded by nothing, so it is the control: a uniform
    # surface in the same frame, proving the span above is shadow and not solver noise.
    # 1 mK, not zero: the field is stored float32, whose spacing at 330 K is 30 uK, and the
    # arms' span is four orders of magnitude above this floor.
    assert _span_k(scene, "deck", 0.0) < 1e-3


def test_the_flight_collapses_the_decks_excess_over_air(scene: Scene) -> None:
    """ADR 0109: the skin's convective speed follows the mission, and that is visible."""
    excess = {}
    for t_rel, label in PHASES:
        air = float(scene.weather.at(scene.t0_s + t_rel).t_air_k) - 273.15
        excess[label] = _mean_c(scene, "deck", t_rel) - air
    assert excess["on the pad"] > 25.0
    assert excess["hard climb"] < 0.6 * excess["on the pad"], excess
    assert excess["landed"] > 25.0, "back on the pad the deck heats up again"
