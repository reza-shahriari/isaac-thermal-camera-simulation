"""MP.3 — the per-pixel temperature lookup, on synthetic G-buffers and no renderer.

Three things carry weight.

`test_a_rendered_surface_reproduces_the_field_it_was_solved_from` is the whole point: pixels
looking at a patch must read that patch's own cells, to well under the NETD. If the gather were
still per-instance the frame would be flat and this fails by the field's full span.

`test_an_unbound_prim_is_bit_identical` is what makes this safe to attach to existing scenes.

`test_camera_space_positions_reach_world_space` guards the frame conversion. M2.4 established the
position AOV is **camera** space on this build; reading it as world would displace every lookup by
the camera's own position -- tens of metres, so every pixel would fall outside its patch and the
strict path would raise rather than render something plausible. The test uses a camera that is both
moved *and* rotated, because the two readings are indistinguishable on a camera at the origin.

docs/physics-model.md §13.1, §13.3; ADR 0087, ADR 0014
"""

from __future__ import annotations

import numpy as np
import pytest

from irsim.optics.motion import transform_points
from irsim.thermal.facets import FacetForcing, FacetProperties
from irsim.thermal.surface_field import PlanarPatch, PlanarThermalField
from irsim_isaac.pipeline.point_bridge import (
    PointwiseTemperature,
    SurfaceBinding,
    world_positions,
)

EX = np.array([1.0, 0.0, 0.0])
EY = np.array([0.0, 1.0, 0.0])
EZ = np.array([0.0, 0.0, 1.0])
ROAD_ID, CAR_ID, BACKGROUND_ID = 3, 7, 0
LABELS = {"0": "BACKGROUND", "3": "/World/road", "7": "/World/car"}


def _patch(origin: np.ndarray, n_u: int = 8, n_v: int = 8, d: float = 0.5) -> PlanarPatch:
    return PlanarPatch(
        origin_m=origin, u_axis=EX, v_axis=EY, n_u=n_u, n_v=n_v, du_m=d, dv_m=d, thickness_m=0.2
    )


def _field_with_a_ramp(patch: PlanarPatch, lo: float = 290.0, hi: float = 320.0):
    """A field whose cells are a known ramp: solved for zero time, so the ramp is the answer."""
    initial = np.linspace(lo, hi, patch.n_cells)
    props = FacetProperties(
        heat_capacity_j_m2_k=np.full(patch.n_cells, 1e9),  # frozen: the initial state stands
        emissivity=np.full(patch.n_cells, 0.95),
        solar_absorptivity=np.full(patch.n_cells, 0.9),
    )
    field = PlanarThermalField(
        patch, props, lambda _t: FacetForcing(t_air_k=290.0, h_w_m2_k=0.0), 0.0, initial
    )
    field.advance_to(1.0)
    return field


def _scene(shape=(6, 6)):
    """A frame looking straight down at a road patch, with one prim filling it."""
    patch = _patch(np.zeros(3))
    field = _field_with_a_ramp(patch)
    centres = patch.cell_centres().reshape(*patch.shape, 3)
    # One pixel per cell, so the expected answer is the cell value exactly.
    ids = np.full(patch.shape, ROAD_ID, dtype=np.uint32)
    flat = np.full(patch.shape, 300.0, dtype=np.float32)
    return patch, field, centres, ids, flat


# ---------------------------------------------------------------------------------------------
# the point of the module
# ---------------------------------------------------------------------------------------------


def test_a_rendered_surface_reproduces_the_field_it_was_solved_from() -> None:
    patch, field, centres, ids, flat = _scene()
    bridge = PointwiseTemperature([SurfaceBinding("/World/road", field)])
    out = bridge.apply(flat, ids, LABELS, centres, 1.0)

    expected = field.temperature_image(1.0)
    assert np.allclose(out, expected, atol=1e-3), f"max err {np.abs(out - expected).max()}"
    # And the frame is genuinely not flat -- a per-instance gather would give 300 K everywhere.
    assert float(out.max() - out.min()) > 25.0
    assert out.dtype == np.float32


def test_between_cell_centres_the_frame_is_smooth() -> None:
    """A camera does not sample cell centres; a staircase would be visible on a bonnet."""
    patch, field, _, _, _ = _scene()
    span_u, span_v = patch.extent_m
    gu = np.linspace(0.05, span_u - 0.05, 40)
    gv = np.linspace(0.05, span_v - 0.05, 40)
    grid_v, grid_u = np.meshgrid(gv, gu, indexing="ij")
    points = np.stack([grid_u, grid_v, np.zeros_like(grid_u)], axis=-1)
    ids = np.full(points.shape[:2], ROAD_ID, dtype=np.uint32)
    flat = np.full(points.shape[:2], 300.0, dtype=np.float32)

    out = PointwiseTemperature([SurfaceBinding("/World/road", field)]).apply(
        flat, ids, LABELS, points, 1.0
    )
    steps = np.abs(np.diff(out.astype(np.float64), axis=1))
    assert steps.max() < 3.0 * np.median(steps[steps > 0]), "the sample staircases between cells"


def test_an_unbound_prim_is_bit_identical() -> None:
    """What makes this safe to attach: a scene with no patches renders exactly as before."""
    patch, field, centres, ids, flat = _scene()
    car_ids = np.full(patch.shape, CAR_ID, dtype=np.uint32)
    out = PointwiseTemperature([SurfaceBinding("/World/road", field)]).apply(
        flat, car_ids, LABELS, centres, 1.0
    )
    assert np.array_equal(out, flat)
    assert np.array_equal(PointwiseTemperature([]).apply(flat, ids, LABELS, centres, 1.0), flat)


def test_only_the_bound_prims_pixels_move() -> None:
    patch, field, centres, ids, flat = _scene()
    mixed = ids.copy()
    mixed[:, 4:] = CAR_ID  # the right half is a different, unbound prim
    out = PointwiseTemperature([SurfaceBinding("/World/road", field)]).apply(
        flat, mixed, LABELS, centres, 1.0
    )
    assert np.array_equal(out[:, 4:], flat[:, 4:])
    assert not np.array_equal(out[:, :4], flat[:, :4])


def test_two_patches_on_one_prim_are_tried_in_turn() -> None:
    """A car needs a bonnet and a roof; both are the same prim as far as the ids are concerned."""
    bonnet = _patch(np.zeros(3), n_u=4, n_v=4)
    roof = _patch(np.array([0.0, 0.0, 1.0]), n_u=4, n_v=4)
    f_bonnet = _field_with_a_ramp(bonnet, 300.0, 310.0)
    f_roof = _field_with_a_ramp(roof, 280.0, 285.0)

    points = np.concatenate(
        [bonnet.cell_centres().reshape(4, 4, 3), roof.cell_centres().reshape(4, 4, 3)], axis=0
    )
    ids = np.full((8, 4), CAR_ID, dtype=np.uint32)
    flat = np.full((8, 4), 300.0, dtype=np.float32)
    bridge = PointwiseTemperature(
        [SurfaceBinding("/World/car", f_bonnet), SurfaceBinding("/World/car", f_roof)]
    )
    out = bridge.apply(flat, ids, LABELS, points, 1.0)
    assert np.allclose(out[:4], f_bonnet.temperature_image(1.0), atol=1e-3)
    assert np.allclose(out[4:], f_roof.temperature_image(1.0), atol=1e-3)


# ---------------------------------------------------------------------------------------------
# the frame conversion
# ---------------------------------------------------------------------------------------------


def test_camera_space_positions_reach_world_space() -> None:
    """A camera both moved and rotated -- the case that separates the three readings (M2.4)."""
    angle = np.deg2rad(35.0)
    rot = np.array(
        [
            [np.cos(angle), 0.0, np.sin(angle)],
            [0.0, 1.0, 0.0],
            [-np.sin(angle), 0.0, np.cos(angle)],
        ]
    )
    cam = np.array([4.0, -2.0, 9.0])
    truth = np.array([[[1.0, 2.0, 3.0], [-0.5, 0.25, 7.0]]])
    camera_space = (truth - cam) @ rot  # the inverse of what world_positions must do

    got = world_positions(camera_space, frame="camera", camera_position=cam, camera_to_world=rot)
    assert np.allclose(got, truth, atol=1e-9)
    # Read as world it is wrong by the camera offset -- ~10 m, far outside any patch.
    naive = world_positions(camera_space, frame="world")
    assert np.linalg.norm(naive - truth, axis=-1).min() > 5.0


def test_world_frame_passes_through_and_a_bad_rotation_raises() -> None:
    pos = np.zeros((2, 2, 3))
    assert np.array_equal(world_positions(pos, frame="world"), pos)
    with pytest.raises(ValueError, match="camera_position"):
        world_positions(pos, frame="camera")
    with pytest.raises(ValueError, match="camera_to_world must be"):
        world_positions(pos, frame="camera", camera_position=np.zeros(3), camera_to_world=np.eye(2))
    with pytest.raises(ValueError, match="unknown position frame"):
        world_positions(pos, frame="clip")


# ---------------------------------------------------------------------------------------------
# what must not pass quietly
# ---------------------------------------------------------------------------------------------


def test_a_pixel_outside_every_patch_raises() -> None:
    """A patch authored smaller than its geometry gives a seam that looks like physics."""
    patch, field, centres, ids, flat = _scene()
    moved = centres.copy()
    moved[0, 0] += np.array([50.0, 0.0, 0.0])
    bridge = PointwiseTemperature([SurfaceBinding("/World/road", field)])
    with pytest.raises(ValueError, match="outside every patch"):
        bridge.apply(flat, ids, LABELS, moved, 1.0)

    lenient = bridge.apply(flat, ids, LABELS, moved, 1.0, strict=False)
    assert lenient[0, 0] == pytest.approx(300.0), "non-strict must fall back, not invent"


def test_a_patch_in_a_local_frame_is_accepted_and_declares_what_it_needs() -> None:
    """Inverted by PT.5: this used to assert the refusal.

    `surface_field` had documented a prim path as a legal frame since MP.1 — "`world` for a road, a
    prim path for a panel that moves with its object" — while this bridge refused everything but
    `world`. The docstring promised what the code declined, and every aerial and maritime target
    moves, so the promise was the useful half. Construction now succeeds and the bridge says which
    frames it needs a matrix for.
    """
    patch = PlanarPatch(
        origin_m=np.zeros(3),
        u_axis=EX,
        v_axis=EY,
        n_u=2,
        n_v=2,
        du_m=0.5,
        dv_m=0.5,
        frame="/World/car",
    )
    bridge = PointwiseTemperature([SurfaceBinding("/World/car", _field_with_a_ramp(patch))])
    assert bridge.local_frames == ("/World/car",)


def test_shape_and_dtype_mismatches_raise() -> None:
    patch, field, centres, ids, flat = _scene()
    bridge = PointwiseTemperature([SurfaceBinding("/World/road", field)])
    with pytest.raises(TypeError, match="integer plane"):
        bridge.apply(flat, ids.astype(np.float32), LABELS, centres, 1.0)
    with pytest.raises(ValueError, match="does not match the plane"):
        bridge.apply(flat, ids[:-1], LABELS, centres, 1.0)
    with pytest.raises(ValueError, match="does not match the plane"):
        bridge.apply(flat, ids, LABELS, centres[:-1], 1.0)


def test_advance_pushes_every_bound_field() -> None:
    patch, field, _, _, _ = _scene()
    other = _field_with_a_ramp(_patch(np.array([10.0, 0.0, 0.0])))
    bridge = PointwiseTemperature(
        [SurfaceBinding("/World/road", field), SurfaceBinding("/World/car", other)]
    )
    bridge.advance_to(60.0)
    assert field.latest_t_s >= 60.0
    assert other.latest_t_s >= 60.0


# ---------------------------------------------------------------------------------------------
# PT.5 — a patch that rides a moving prim
# ---------------------------------------------------------------------------------------------


def _pose(translation: np.ndarray, yaw_deg: float = 0.0) -> np.ndarray:
    """A world-from-local 4x4 in USD's **row-vector** convention: `p' = p @ M`.

    The translation goes in the last **row**, and the rotation block is the transpose of the
    column-convention one. Written this way on purpose: a test that built a column-convention
    matrix would pass against a column-convention bug and prove nothing about the renderer.
    """
    a = np.deg2rad(yaw_deg)
    rotation = np.array(
        [[np.cos(a), -np.sin(a), 0.0], [np.sin(a), np.cos(a), 0.0], [0.0, 0.0, 1.0]]
    )
    matrix = np.eye(4)
    matrix[:3, :3] = rotation.T
    matrix[3, :3] = translation
    return matrix


def _local_patch() -> PlanarPatch:
    return PlanarPatch(
        origin_m=np.zeros(3),
        u_axis=EX,
        v_axis=EY,
        n_u=8,
        n_v=8,
        du_m=0.5,
        dv_m=0.5,
        thickness_m=0.2,
        frame="/World/car",
    )


def test_a_material_point_keeps_its_cell_when_the_prim_moves() -> None:
    """The acceptance: translate 10 m and yaw 90°, and the same material point reads the same cell.

    A material point is fixed in the prim's own frame; the renderer reports it in world. If the
    bridge sampled the world position against a local patch — which is what it did before PT.5,
    modulo refusing outright — a prim that moved would slide across its own temperature field, and
    the result would be a smooth, plausible, completely wrong gradient.
    """
    patch = _local_patch()
    field = _field_with_a_ramp(patch)
    bridge = PointwiseTemperature([SurfaceBinding("/World/car", field)])

    local = patch.cell_centres().reshape(*patch.shape, 3)
    ids = np.full(patch.shape, CAR_ID, dtype=np.uint32)
    flat = np.full(patch.shape, 300.0, dtype=np.float32)

    poses = [
        _pose(np.zeros(3)),
        _pose(np.array([10.0, 0.0, 0.0])),
        _pose(np.array([10.0, -4.0, 0.0]), yaw_deg=90.0),
        _pose(np.array([-3.0, 7.5, 2.0]), yaw_deg=215.0),
    ]
    reference = None
    for pose in poses:
        world = transform_points(local, pose)
        out = bridge.apply(flat, ids, LABELS, world, 1.0, world_from_local={"/World/car": pose})
        if reference is None:
            reference = out
        # Sub-cell, and in fact far tighter: the transform is exact and only float64 rounding
        # separates the two. One cell here spans about 4 K of the ramp.
        assert np.max(np.abs(out.astype(np.float64) - reference.astype(np.float64))) < 1e-6


def test_ignoring_the_prim_transform_would_be_visibly_wrong() -> None:
    """The negative control that gives the test above its teeth."""
    patch = _local_patch()
    field = _field_with_a_ramp(patch)
    bridge = PointwiseTemperature([SurfaceBinding("/World/car", field)])
    local = patch.cell_centres().reshape(*patch.shape, 3)
    ids = np.full(patch.shape, CAR_ID, dtype=np.uint32)
    flat = np.full(patch.shape, 300.0, dtype=np.float32)

    pose = _pose(np.array([10.0, 0.0, 0.0]))
    world = transform_points(local, pose)

    right = bridge.apply(flat, ids, LABELS, world, 1.0, world_from_local={"/World/car": pose})
    # Sampling the world points against the local patch is what the old code would have had to do.
    wrong = bridge.apply(
        flat, ids, LABELS, world, 1.0, strict=False, world_from_local={"/World/car": np.eye(4)}
    )
    assert np.max(np.abs(right.astype(np.float64) - wrong.astype(np.float64))) > 5.0


def test_the_world_frame_path_is_bit_identical() -> None:
    """Every existing scene authors patches in world; PT.5 must not have moved any of them."""
    patch, field, centres, ids, flat = _scene()
    bridge = PointwiseTemperature([SurfaceBinding("/World/road", field)])
    without = bridge.apply(flat, ids, LABELS, centres, 1.0)
    with_matrices = bridge.apply(
        flat, ids, LABELS, centres, 1.0, world_from_local={"/World/car": _pose(np.ones(3))}
    )
    assert np.array_equal(without, with_matrices)
    assert bridge.local_frames == ()


def test_a_local_patch_without_a_matrix_says_which_one_is_missing() -> None:
    patch = _local_patch()
    bridge = PointwiseTemperature([SurfaceBinding("/World/car", _field_with_a_ramp(patch))])
    ids = np.full(patch.shape, CAR_ID, dtype=np.uint32)
    flat = np.full(patch.shape, 300.0, dtype=np.float32)
    points = patch.cell_centres().reshape(*patch.shape, 3)
    with pytest.raises(KeyError, match="/World/car"):
        bridge.apply(flat, ids, LABELS, points, 1.0)


def test_a_malformed_transform_is_refused() -> None:
    patch = _local_patch()
    bridge = PointwiseTemperature([SurfaceBinding("/World/car", _field_with_a_ramp(patch))])
    ids = np.full(patch.shape, CAR_ID, dtype=np.uint32)
    flat = np.full(patch.shape, 300.0, dtype=np.float32)
    points = patch.cell_centres().reshape(*patch.shape, 3)
    with pytest.raises(ValueError, match="4x4"):
        bridge.apply(flat, ids, LABELS, points, 1.0, world_from_local={"/World/car": np.eye(3)})


def test_the_transform_uses_usd_row_vector_convention() -> None:
    """The transpose trap, pinned: ADR 0014's M10.19 addendum records it costing 164 rows.

    A yawed prim is where the two conventions separate: `p @ M` and `M @ p` differ by the transpose
    of the rotation, and both produce a smooth, believable field. So the test asks for a pose with
    real yaw and checks that feeding the *column*-convention matrix lands somewhere else.
    """
    patch = _local_patch()
    field = _field_with_a_ramp(patch)
    bridge = PointwiseTemperature([SurfaceBinding("/World/car", field)])
    local = patch.cell_centres().reshape(*patch.shape, 3)
    ids = np.full(patch.shape, CAR_ID, dtype=np.uint32)
    flat = np.full(patch.shape, 300.0, dtype=np.float32)

    pose = _pose(np.array([2.0, -1.0, 0.0]), yaw_deg=40.0)
    world = transform_points(local, pose)

    right = bridge.apply(flat, ids, LABELS, world, 1.0, world_from_local={"/World/car": pose})
    transposed = bridge.apply(
        flat, ids, LABELS, world, 1.0, strict=False, world_from_local={"/World/car": pose.T}
    )
    assert np.max(np.abs(right.astype(np.float64) - transposed.astype(np.float64))) > 1.0
    # And the right one really does reproduce the field, so "different" is not "both wrong".
    assert np.allclose(right, field.temperature_image(1.0), atol=1e-3)


# ---------------------------------------------------------------------------------------------
# PT.17: the bindings come off the scene
# ---------------------------------------------------------------------------------------------


def test_bindings_from_scene_wrap_every_patched_surface_with_a_prim(tophat_lwir_lut) -> None:  # type: ignore[no-untyped-def]
    """One line between `Scene.surface_bindings` and `IrCamera(surface_fields=...)`."""
    import pathlib

    from irsim.scene import Scene
    from irsim_isaac.pipeline.point_bridge import SurfaceBinding, bindings_from_scene

    repo = pathlib.Path(__file__).resolve().parents[2]
    scene = Scene.from_file(
        repo / "configs/scenes/car_ignition_overcast_night.yaml", {"lwir": tophat_lwir_lut}
    )
    bindings = bindings_from_scene(scene)
    assert {b.prim_path for b in bindings} == {"/World/Road", "/World/Car/bonnet"}
    assert all(isinstance(b, SurfaceBinding) for b in bindings)
    assert PointwiseTemperature(bindings).prim_paths == ("/World/Road", "/World/Car/bonnet")


# ---------------------------------------------------------------------------------------------
# PT.19: an unconsumed binding is loud
# ---------------------------------------------------------------------------------------------


def test_a_misspelt_prim_path_raises_naming_it_and_correct_paths_are_bit_identical() -> None:
    """The failure this exists for: `/World/Car/bonet` bound, `/World/Car/bonnet` on the stage,
    and the bonnet rendering at its ambient fallback with a seam that looked like physics."""
    _patch, field, centres, ids, flat = _scene()
    stage = ["/World/road", "/World/car"]
    with pytest.raises(ValueError, match=r"/World/raod.*not among the stage's prims"):
        PointwiseTemperature([SurfaceBinding("/World/raod", field)], known_paths=stage)
    # Without a stage list the frame's labels are the only authority: strict raises, else warns.
    bad = PointwiseTemperature([SurfaceBinding("/World/raod", field)])
    with pytest.raises(ValueError, match=r"/World/raod.*absent from this frame's idToLabels"):
        bad.apply(flat, ids, LABELS, centres, 1.0)
    with pytest.warns(UserWarning, match="/World/raod"):
        lenient = bad.apply(flat, ids, LABELS, centres, 1.0, strict=False)
    assert np.array_equal(lenient, flat)  # nothing consumed: the plane is untouched
    assert bad.last_coverage == {"/World/raod": 0}
    # And a correctly spelt binding is exactly what it was before this check existed.
    good = PointwiseTemperature([SurfaceBinding("/World/road", field)], known_paths=stage)
    expected = PointwiseTemperature([SurfaceBinding("/World/road", field)]).apply(
        flat, ids, LABELS, centres, 1.0
    )
    assert np.array_equal(good.apply(flat, ids, LABELS, centres, 1.0), expected)
    assert good.last_coverage == {"/World/road": int(ids.size)}


def test_coverage_counts_the_pixels_each_binding_took_and_off_screen_is_not_an_error() -> None:
    _patch, field, centres, ids, flat = _scene()
    mixed = ids.copy()
    mixed[:, :3] = 7  # the car's id over half the frame; the road keeps the other half
    bridge = PointwiseTemperature(
        [SurfaceBinding("/World/road", field)], known_paths=["/World/road", "/World/car"]
    )
    bridge.apply(flat, mixed, LABELS, centres, 1.0)
    assert bridge.last_coverage == {"/World/road": int((mixed == 3).sum())}
    # A bound prim off screen this frame -- absent from the labels too -- is coverage 0, quietly,
    # because the stage list already said the path is real.
    car_only = np.full_like(ids, 7)
    out = bridge.apply(flat, car_only, {"0": "BACKGROUND", "7": "/World/car"}, centres, 1.0)
    assert np.array_equal(out, flat) and bridge.last_coverage == {"/World/road": 0}
