"""Engine-free tests for the Isaac geometry adapter (roadmap M10.1).

The adapter's arithmetic -- per-pixel ray directions, the two cosines, V_s, the motion scaling,
the sky mask -- needs no renderer, so it is verified here in milliseconds against closed forms
rather than inside a 35 s Kit boot. ``tests/integration/test_gbuffer_isaac.py`` then checks only
what genuinely needs the engine: that the AOVs exist, carry float32, and agree with the scene.

The sphere case is the one that catches the expensive bug. Taking the cosine against the optical
axis (or reading a view-space normal's z channel) is right at the principal point and wrong
everywhere else; ``test_sphere_cos_theta_needs_the_per_pixel_ray`` asserts both that the adapter
matches the closed form and that the axis shortcut would miss by more than the tolerance.
"""

from __future__ import annotations

import numpy as np
import pytest

from irsim.config.gbuffer import GBuffer
from irsim_isaac.pipeline.gbuffer_isaac import (
    RawAovs,
    geometry_planes,
    motion_px_per_frame,
    orient_to_viewer,
    ray_directions,
    to_gbuffer,
)

# --- analytic scenes ---------------------------------------------------------------------------

SPHERE_CENTRE = np.array([0.0, 0.0, -12.0])
SPHERE_RADIUS = 3.0
CAMERA_POSITION = np.array([0.0, 0.0, 0.0])


def _sphere_scene(n: int = 41) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Ray-trace a sphere analytically: (position, normal, distance, cos_theta_closed_form).

    Camera at the origin looking -Z. For a ray whose perpendicular distance from the centre is
    ``b``, the near-side hit has ``sin(theta) = b / R`` between the normal and the view direction,
    so ``cos(theta) = sqrt(1 - (b/R)^2)`` -- derived from the geometry, not from the adapter.
    """
    half = 0.32
    u, v = np.meshgrid(np.linspace(-half, half, n), np.linspace(-half, half, n), indexing="xy")
    direction = np.stack([u, v, -np.ones_like(u)], axis=2)
    direction /= np.linalg.norm(direction, axis=2, keepdims=True)

    oc = CAMERA_POSITION - SPHERE_CENTRE
    b = np.sum(direction * oc, axis=2)
    c = float(oc @ oc) - SPHERE_RADIUS**2
    disc = b * b - c
    hit = disc > 0.0
    t = np.where(hit, -b - np.sqrt(np.maximum(disc, 0.0)), np.inf)

    position = CAMERA_POSITION + direction * np.where(hit, t, 0.0)[..., None]
    normal = (position - SPHERE_CENTRE) / SPHERE_RADIUS
    # perpendicular distance from the sphere centre to the ray line
    impact = np.sqrt(np.maximum(float(oc @ oc) - b * b, 0.0))
    cos_theta = np.sqrt(np.maximum(1.0 - (impact / SPHERE_RADIUS) ** 2, 0.0))

    position = np.where(hit[..., None], position, 0.0)
    normal = np.where(hit[..., None], normal, 0.0)
    return position, normal, t, np.where(hit, cos_theta, np.nan)


def _plane_aovs(normal: tuple[float, float, float], distance: float = 5.0) -> RawAovs:
    """A 4x4 patch of one flat plane at a fixed distance, normal authored as given."""
    shape = (4, 4)
    n = np.broadcast_to(np.asarray(normal, dtype=np.float32), (*shape, 3)).copy()
    pos = np.zeros((*shape, 3), dtype=np.float32)
    pos[:, :, 2] = -distance
    return RawAovs(
        distance_m=np.full(shape, distance, dtype=np.float32),
        normal=n,
        position=pos,
        instance_id=np.ones(shape, dtype=np.uint32),
    )


# --- the cosine against the per-pixel ray ------------------------------------------------------


def test_sphere_cos_theta_needs_the_per_pixel_ray() -> None:
    position, normal, distance, expected = _sphere_scene()
    aovs = RawAovs(
        distance_m=distance.astype(np.float32),
        normal=normal.astype(np.float32),
        position=position.astype(np.float32),
    )
    planes = geometry_planes(aovs, camera_position=CAMERA_POSITION)

    hit = np.isfinite(expected)
    assert hit.sum() > 100
    got = planes.normal_dot_view[hit].astype(np.float64)
    assert np.max(np.abs(got - expected[hit])) < 1e-5

    # The shortcut this test exists to forbid: cosine against the optical axis (-Z), which is what
    # a view-space normal's z channel would give. It agrees at the centre and drifts off-axis.
    axis = np.abs(normal[:, :, 2])[hit]
    assert np.max(np.abs(axis - expected[hit])) > 0.05


def test_sphere_silhouette_and_nose_are_exact() -> None:
    position, normal, distance, expected = _sphere_scene(n=101)
    aovs = RawAovs(
        distance_m=distance.astype(np.float32),
        normal=normal.astype(np.float32),
        position=position.astype(np.float32),
    )
    planes = geometry_planes(aovs, camera_position=CAMERA_POSITION)
    hit = np.isfinite(expected)
    assert planes.normal_dot_view[hit].min() < 0.06  # silhouette -> grazing
    assert planes.normal_dot_view[50, 50] == pytest.approx(1.0, abs=1e-6)  # nose -> normal-on


def test_ray_directions_are_unit_and_point_away_from_the_camera() -> None:
    position, _, _, expected = _sphere_scene()
    ray = ray_directions(position, frame="world", camera_position=CAMERA_POSITION)
    hit = np.isfinite(expected)
    assert np.allclose(np.linalg.norm(ray[hit], axis=1), 1.0, atol=1e-6)
    assert np.all(ray[hit][:, 2] < 0.0)  # camera looks -Z


def test_camera_frame_positions_rotate_into_world_axes() -> None:
    """A camera-space position AOV gives the same cosines once rotated to world axes."""
    position, normal, distance, expected = _sphere_scene()
    # Camera yawed 90 deg about +Y: world = R @ camera.
    rot = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
    world_normal = normal @ rot.T
    aovs = RawAovs(
        distance_m=distance.astype(np.float32),
        normal=world_normal.astype(np.float32),
        position=position.astype(np.float32),  # still camera-space
    )
    planes = geometry_planes(aovs, position_frame="camera", camera_to_world=rot)
    hit = np.isfinite(expected)
    assert np.max(np.abs(planes.normal_dot_view[hit] - expected[hit])) < 1e-5


def test_two_sided_normals_are_flipped_into_the_viewing_hemisphere() -> None:
    """A double-sided quad may store the away-facing normal; the cosine must stay positive."""
    facing = _plane_aovs((0.0, 0.0, 1.0))
    away = _plane_aovs((0.0, 0.0, -1.0))
    assert np.allclose(
        geometry_planes(facing, camera_position=CAMERA_POSITION).normal_dot_view, 1.0
    )
    assert np.allclose(geometry_planes(away, camera_position=CAMERA_POSITION).normal_dot_view, 1.0)


def test_orient_to_viewer_returns_the_flipped_normal() -> None:
    ray = np.zeros((1, 1, 3))
    ray[..., 2] = -1.0
    normal, cos_theta = orient_to_viewer(np.array([[[0.0, 0.0, -2.0]]]), ray)
    assert np.allclose(normal, [0.0, 0.0, 1.0])  # unit length and flipped towards the camera
    assert cos_theta == pytest.approx(1.0)


# --- the sky-view factor (ADR 0045) ------------------------------------------------------------


@pytest.mark.parametrize(
    ("normal", "expected"),
    [((0.0, 1.0, 0.0), 1.0), ((0.0, 0.0, 1.0), 0.5), ((0.0, -1.0, 0.0), 0.0)],
    ids=["up", "vertical", "under"],
)
def test_sky_view_factor_of_up_vertical_and_under_planes(
    normal: tuple[float, float, float], expected: float
) -> None:
    """V_s = (1 + n.up)/2, not raw occlusion: a wall sees half the sky, a soffit none."""
    planes = geometry_planes(_plane_aovs(normal), up_axis="Y", camera_position=CAMERA_POSITION)
    assert np.allclose(planes.sky_view_factor, expected, atol=0.05)
    assert np.allclose(planes.normal_dot_up, 2.0 * expected - 1.0, atol=0.05)


def test_occlusion_scales_the_sky_view_factor() -> None:
    aovs = _plane_aovs((0.0, 1.0, 0.0))
    occluded = RawAovs(
        distance_m=aovs.distance_m,
        normal=aovs.normal,
        position=aovs.position,
        occlusion=np.full(aovs.shape, 0.25, dtype=np.float32),
        instance_id=aovs.instance_id,
    )
    assert np.allclose(
        geometry_planes(occluded, camera_position=CAMERA_POSITION).sky_view_factor, 0.25
    )


def test_z_up_stage_uses_the_z_axis() -> None:
    planes = geometry_planes(
        _plane_aovs((0.0, 0.0, 1.0)), up_axis="Z", camera_position=CAMERA_POSITION
    )
    assert np.allclose(planes.sky_view_factor, 1.0)
    with pytest.raises(ValueError, match="up_axis"):
        geometry_planes(_plane_aovs((0.0, 1.0, 0.0)), up_axis="X", camera_position=CAMERA_POSITION)


# --- motion ------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("convention", "raw", "expected"),
    [
        ("pixels", (3.0, 0.0), (3.0, 0.0)),
        ("ndc", (3.0 / 128.0, 0.0), (3.0, 0.0)),
        ("uv", (3.0 / 256.0, 0.0), (3.0, 0.0)),
    ],
)
def test_motion_conventions_all_report_three_pixels_per_frame(
    convention: str, raw: tuple[float, float], expected: tuple[float, float]
) -> None:
    plane = np.broadcast_to(np.asarray(raw, dtype=np.float32), (256, 256, 2))
    out = motion_px_per_frame(plane, convention=convention, shape=(256, 256))  # type: ignore[arg-type]
    assert out.dtype == np.float32
    assert np.allclose(out[..., 0], expected[0], atol=0.1)
    assert np.allclose(out[..., 1], expected[1], atol=0.1)


def test_ndc_motion_flips_the_y_axis() -> None:
    """NDC y points up, image y points down; a sign error here reverses smear direction."""
    plane = np.broadcast_to(np.asarray([0.0, 3.0 / 128.0], dtype=np.float32), (256, 256, 2))
    out = motion_px_per_frame(plane, convention="ndc", shape=(256, 256))
    assert np.allclose(out[..., 1], -3.0, atol=0.1)


# --- sky pixels, dtypes, and the M0.6 contract -------------------------------------------------


def test_sky_pixels_are_masked_and_left_finite() -> None:
    shape = (4, 4)
    distance = np.full(shape, 7.0, dtype=np.float32)
    distance[0, :] = np.inf
    ids = np.ones(shape, dtype=np.uint32)
    ids[1, :] = 0  # the other half of the same fact (ADR 0014)
    normal = np.zeros((*shape, 3), dtype=np.float32)
    normal[:, :, 1] = 1.0
    position = np.zeros((*shape, 3), dtype=np.float32)
    position[:, :, 2] = -7.0
    planes = geometry_planes(
        RawAovs(
            distance_m=distance,
            normal=normal,
            position=position,
            instance_id=ids,
            motion=np.full((*shape, 2), 2.0, dtype=np.float32),
        ),
        camera_position=CAMERA_POSITION,
        # Named, not defaulted: there is no safe default convention (IG.5).
        motion_convention="pixels",
    )
    assert planes.sky_mask[0].all() and planes.sky_mask[1].all()
    assert not planes.sky_mask[2:].any()
    # tau = 1 for a consumer that ignores the mask, and nothing non-finite reaches the kernels
    assert np.all(planes.distance_m[planes.sky_mask] == 0.0)
    assert np.isfinite(planes.distance_m).all()
    assert np.all(planes.motion_px is not None and planes.motion_px[planes.sky_mask] == 0.0)


def test_to_gbuffer_satisfies_the_m0_6_contract() -> None:
    shape = (8, 8)
    distance = np.full(shape, 30.0, dtype=np.float32)
    distance[0, 0] = np.inf
    normal = np.zeros((*shape, 3), dtype=np.float32)
    normal[:, :, 1] = 1.0
    position = np.zeros((*shape, 3), dtype=np.float32)
    position[:, :, 2] = -30.0
    planes = geometry_planes(
        RawAovs(
            distance_m=distance,
            normal=normal,
            position=position,
            instance_id=np.ones(shape, dtype=np.uint32),
            semantic_id=np.full(shape, 3, dtype=np.uint32),
            motion=np.zeros((*shape, 2), dtype=np.float32),
        ),
        camera_position=CAMERA_POSITION,
        motion_convention="pixels",
    )
    gbuf = to_gbuffer(
        planes,
        temperature_k=np.full(shape, 295.0, dtype=np.float32),
        material_id=np.full(shape, 4, dtype=np.int32),
        sky_temperature_k=np.float32(233.0),
    )
    assert isinstance(gbuf, GBuffer)
    assert gbuf.temperature_k.dtype == np.float32 and gbuf.distance_m.dtype == np.float32
    assert gbuf.material_id.dtype == np.int32
    assert gbuf.temperature_k[0, 0] == pytest.approx(233.0)  # sky took T_sky
    assert gbuf.temperature_k[1, 1] == pytest.approx(295.0)
    assert gbuf.sky_mask is not None and gbuf.sky_mask[0, 0]


def test_float16_normals_are_refused() -> None:
    """Inverted by IG.8. This test used to assert the opposite, on a false premise.

    Its docstring read "this build delivers fp16 normals (ADR 0014 addendum)". ADR 0014's addendum
    records `normals` as **float32 x4 at full resolution** and marks it *use*, and the build's
    Replicator registry registers it as float32. The fp16 plane in that table is `PtWorldNormal`,
    which `AovReader` rejects anyway for being half-resolution and all zero. The carve-out was
    rescuing nothing and spending CLAUDE.md non-negotiable #2 to do it.
    """
    aovs = _plane_aovs((0.0, 1.0, 0.0))
    half = RawAovs(
        distance_m=aovs.distance_m,
        normal=aovs.normal.astype(np.float16),
        position=aovs.position,
    )
    with pytest.raises(TypeError, match="float16"):
        geometry_planes(half, camera_position=CAMERA_POSITION)


def test_float32_normals_still_work() -> None:
    """The guard must refuse the dtype, not the channel: float32 normals go through untouched."""
    planes = geometry_planes(_plane_aovs((0.0, 1.0, 0.0)), camera_position=CAMERA_POSITION)
    assert planes.normal_dot_up.dtype == np.float32
    assert np.allclose(planes.sky_view_factor, 1.0, atol=1e-3)


def test_float16_occlusion_is_refused() -> None:
    """The other plane the carve-out covered. No AO AOV delivers here, so nothing is lost."""
    aovs = _plane_aovs((0.0, 1.0, 0.0))
    half = RawAovs(
        distance_m=aovs.distance_m,
        normal=aovs.normal,
        position=aovs.position,
        occlusion=np.full(aovs.distance_m.shape, 0.5, dtype=np.float16),
    )
    with pytest.raises(TypeError, match="float16"):
        geometry_planes(half, camera_position=CAMERA_POSITION)


def test_float16_motion_is_refused() -> None:
    """The third, reachable only by a caller that named a convention (IG.5)."""
    aovs = _plane_aovs((0.0, 1.0, 0.0))
    half = RawAovs(
        distance_m=aovs.distance_m,
        normal=aovs.normal,
        position=aovs.position,
        motion=np.zeros((*aovs.distance_m.shape, 2), dtype=np.float16),
    )
    with pytest.raises(TypeError, match="float16"):
        geometry_planes(half, camera_position=CAMERA_POSITION, motion_convention="pixels")


def test_float16_distance_is_refused() -> None:
    aovs = _plane_aovs((0.0, 1.0, 0.0))
    half = RawAovs(
        distance_m=aovs.distance_m.astype(np.float16),
        normal=aovs.normal,
        position=aovs.position,
    )
    with pytest.raises(TypeError, match="float16"):
        geometry_planes(half, camera_position=CAMERA_POSITION)


def test_float16_position_is_refused() -> None:
    """Position is distance in three components: the ray length it implies must stay float32."""
    aovs = _plane_aovs((0.0, 1.0, 0.0))
    half = RawAovs(
        distance_m=aovs.distance_m,
        normal=aovs.normal,
        position=aovs.position.astype(np.float16),
    )
    with pytest.raises(TypeError, match="float16"):
        geometry_planes(half, camera_position=CAMERA_POSITION)


def test_float16_facet_temperature_is_refused() -> None:
    planes = geometry_planes(_plane_aovs((0.0, 1.0, 0.0)), camera_position=CAMERA_POSITION)
    with pytest.raises(TypeError, match="float16"):
        to_gbuffer(
            planes,
            temperature_k=np.full(planes.shape, 295.0, dtype=np.float16),
            material_id=np.ones(planes.shape, dtype=np.int32),
        )


def test_non_integer_material_id_is_refused() -> None:
    planes = geometry_planes(_plane_aovs((0.0, 1.0, 0.0)), camera_position=CAMERA_POSITION)
    with pytest.raises(TypeError, match="integer"):
        to_gbuffer(
            planes,
            temperature_k=np.full(planes.shape, 295.0, dtype=np.float32),
            material_id=np.ones(planes.shape, dtype=np.float32),
        )


# --- which frame the position AOV is in (roadmap M2.4) -------------------------------------------

_PITCH_DEG = 20.0
_PROBE_CAMERA = np.array([2.0, 1.0, 5.0])
_PROBE_RESOLUTION = 32
_PROBE_FOCAL_PX = 40.0


def _pitch_rotation(degrees: float) -> np.ndarray:
    """``camera_to_world`` for a pitch about X, in ``ray_directions``' convention.

    That convention is ``v_world = v_camera @ rot.T``, so ``rot`` is the transpose of the USD
    row-vector matrix's upper-left 3x3 -- which for a rotation is the matrix that takes a camera
    vector to world as an ordinary column-vector product.
    """
    c, s = np.cos(np.radians(degrees)), np.sin(np.radians(degrees))
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _synthetic_position_aov(
    frame: str, rot: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A plane 9 m down the boresight, encoded in ``frame``: (position, distance, rays).

    The surface points come from the rays themselves, so the scene is exactly what the oracle
    assumes and any residual is the encoding, not the geometry.
    """
    from irsim_isaac.geometry_probe import pinhole_rays

    rays = pinhole_rays(_PROBE_RESOLUTION, _PROBE_FOCAL_PX, rot)
    distance = np.full(rays.shape[:2], 9.0) + 0.4 * rays[:, :, 0]  # not a constant-range sphere
    world = _PROBE_CAMERA + distance[..., None] * rays
    if frame == "world":
        position = world
    elif frame == "camera":
        position = (world - _PROBE_CAMERA) @ rot
    elif frame == "rotated_world":
        position = world @ rot
    else:  # pragma: no cover - guard against a typo in a parametrisation
        raise AssertionError(frame)
    return position, distance, rays


@pytest.mark.parametrize("frame", ["world", "camera", "rotated_world"])
def test_position_frame_residuals_name_the_frame_they_were_given(frame: str) -> None:
    """The discriminator M2.4 rests on: each encoding must be recognised, and only that one.

    ADR 0014 recorded the position AOV as world space and its M10.19 addendum as camera space,
    each measured on a scene degenerate in the other's axis -- and both readings are equally
    consistent with ``rotated_world``, the world point in camera **axes**. Distinguishing them
    needs a camera that is off the origin *and* rotated, so the margin below is asserted as well
    as the verdict: a verdict from a degenerate scene is not evidence.
    """
    from irsim_isaac.geometry_probe import position_frame_residuals

    rot = _pitch_rotation(_PITCH_DEG)
    position, distance, rays = _synthetic_position_aov(frame, rot)
    got = position_frame_residuals(
        position, distance, rays, camera_position=_PROBE_CAMERA, camera_to_world=rot
    )
    assert got["verdict"] == frame, got
    assert got["residual_m"][frame] < 1e-9, got
    assert got["margin_m"] > 0.5, got  # the runner-up is wrong by metres, not by rounding


def test_an_unrotated_camera_at_the_origin_cannot_tell_the_frames_apart() -> None:
    """Why ADR 0014 and its M10.19 addendum disagreed: both scenes were degenerate.

    With C = 0 and no rotation all three hypotheses decode to the same point, so every residual
    is zero and the margin is zero. The probe reports that rather than picking a winner by
    floating-point noise, which is what makes the in-sim verdict trustworthy.
    """
    from irsim_isaac.geometry_probe import pinhole_rays, position_frame_residuals

    rot = np.eye(3)
    rays = pinhole_rays(_PROBE_RESOLUTION, _PROBE_FOCAL_PX, rot)
    distance = np.full(rays.shape[:2], 9.0)
    got = position_frame_residuals(
        distance[..., None] * rays,
        distance,
        rays,
        camera_position=np.zeros(3),
        camera_to_world=rot,
    )
    assert max(got["residual_m"].values()) < 1e-9, got
    assert got["margin_m"] < 1e-9, got


def test_camera_and_rotated_world_are_separated_by_the_camera_offset() -> None:
    """The two readings the ADR holds differ by exactly |C|, so a camera at the origin cannot tell.

    This is the quantitative form of the degeneracy above: it is the camera's *translation* that
    separates these two, and its *rotation* that separates either from world space.
    """
    from irsim_isaac.geometry_probe import position_frame_residuals

    rot = _pitch_rotation(_PITCH_DEG)
    position, distance, rays = _synthetic_position_aov("camera", rot)
    got = position_frame_residuals(
        position, distance, rays, camera_position=_PROBE_CAMERA, camera_to_world=rot
    )
    offset = float(np.linalg.norm(_PROBE_CAMERA))
    assert abs(got["residual_m"]["rotated_world"] - offset) < 1e-9, got


def test_position_frame_residuals_ignore_sky_and_the_miss_sentinel() -> None:
    """Sky pixels carry ``inf`` distance and a miss sentinel 1000 m down the ray; neither is
    geometry.

    Scoring them would swamp the median with nonsense and could invert the verdict. The ray
    length is what excludes them -- deliberately not the sentinel's magnitude, which would also
    discard real targets in a long-range aerial scene.
    """
    from irsim_isaac.geometry_probe import position_frame_residuals

    rot = _pitch_rotation(_PITCH_DEG)
    position, distance, rays = _synthetic_position_aov("camera", rot)
    position, distance = position.copy(), distance.copy()
    position[0, :, :] = -1000.0
    distance[0, :] = np.inf
    distance[1, :] = np.inf
    got = position_frame_residuals(
        position, distance, rays, camera_position=_PROBE_CAMERA, camera_to_world=rot
    )
    assert got["sample_pixels"] == position.shape[0] * position.shape[1] - 2 * position.shape[1]
    assert got["verdict"] == "camera", got
    assert got["residual_m"]["camera"] < 1e-9, got
