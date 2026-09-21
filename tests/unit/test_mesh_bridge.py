"""WM.3 — a mesh temperature field reaching the pixels of a curved prim.

`WM.2` put a field on a triangle mesh; this is the bridge that feeds it a face and a barycentric
pair per pixel, so a wheel, a tyre or an exhaust pipe stops being one temperature — the case
ADR 0087 lists as Hard.

The claims:

* Warp's `mesh_query_point_no_sign` and the brute-force NumPy oracle agree on the **face** and on
  the **sampled temperature** for 100 % of pixels on a curved fixture. The oracle is not an
  optimised twin: it tests every triangle.
* A prim with no binding keeps the per-instance value **bit-identically**, so attaching the
  bridge to an existing scene changes nothing until a mesh field is authored for something in it.
* A pixel further from the bound mesh than the tolerance **raises** rather than snapping to the
  nearest cell.

docs/physics-model.md §13.1; ADR 0087 (and its WM.1 addendum), ADR 0014; roadmap WM.3.
"""

from __future__ import annotations

import numpy as np
import pytest

from irsim.thermal.facets import FacetForcing, FacetProperties
from irsim.thermal.mesh_field import TriangleMeshField, TriangleMeshPatch, closest_point_on_mesh
from irsim.thermal.raycast import TriangleSoup, sphere_mesh
from irsim_isaac.pipeline.mesh_bridge import (
    DEFAULT_MAX_DISTANCE_M,
    MeshBinding,
    MeshPointBridge,
    warp_available,
)

T_AIR_K = 293.15
SUN = np.asarray([0.0, 0.0, 1.0])
SETTLE_S = 3600.0


def pipe_mesh(radius: float, length: float, n_phi: int = 24, n_z: int = 8) -> TriangleSoup:
    """An open cylinder along +x: the exhaust pipe the row names, and a shape whose temperature
    must vary around its circumference as well as along it."""
    phi = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)
    x = np.linspace(0.0, length, n_z + 1)
    xx, pp = np.meshgrid(x, phi, indexing="ij")
    vertices = np.stack([xx, radius * np.cos(pp), radius * np.sin(pp)], axis=-1).reshape(-1, 3)
    rows, cols = np.arange(n_z)[:, None], np.arange(n_phi)[None, :]
    a = (rows * n_phi + cols).ravel()
    b = (rows * n_phi + (cols + 1) % n_phi).ravel()
    c = ((rows + 1) * n_phi + cols).ravel()
    d = ((rows + 1) * n_phi + (cols + 1) % n_phi).ravel()
    faces = np.concatenate([np.stack([a, c, d], axis=1), np.stack([a, d, b], axis=1)], axis=0)
    return TriangleSoup(vertices, np.asarray(faces, dtype=np.int64))


def _sunlit_field(patch: TriangleMeshPatch, dni: float = 900.0) -> TriangleMeshField:
    n = patch.n_cells
    cos = np.maximum(patch.cell_normal @ SUN, 0.0)
    field = TriangleMeshField(
        patch,
        FacetProperties(np.full(n, 2000.0), np.full(n, 0.90), np.full(n, 0.60)),
        lambda _t: FacetForcing(
            t_air_k=T_AIR_K,
            h_w_m2_k=10.0,
            q_solar_w_m2=dni * cos,
            q_longwave_down_w_m2=300.0,
        ),
        0.0,
        290.0,
        1.0,
        keep_ticks=None,
    )
    field.advance_to(SETTLE_S)
    return field


def _on_mesh(patch: TriangleMeshPatch, count: int, seed: int) -> np.ndarray:
    """Points on the mesh itself, which is where the position AOV reports a hit: it reports where
    the ray met the *rendered* geometry, not where an ideal surface would have been."""
    rng = np.random.default_rng(seed)
    face = rng.integers(0, patch.n_faces, count)
    r1, r2 = rng.random(count), rng.random(count)
    root = np.sqrt(r1)
    weights = np.stack([1.0 - root, root * (1.0 - r2), root * r2], axis=-1)
    return np.asarray(np.einsum("pv,pvx->px", weights, patch.vertices_m[patch.faces[face]]))


# --- the bar: Warp against the oracle ---------------------------------------------------------


@pytest.mark.slow
def test_warp_and_the_numpy_oracle_agree_on_every_pixel_of_a_curved_prim() -> None:
    """The row's own criterion. A 0.25 m sphere, 4 000 query points on its own surface: the Warp
    query and a brute-force closest point over every triangle must name the same face and hand
    back the same temperature, for **all** of them. The oracle shares no code with the fast path
    -- it has no BVH and no early out -- so agreement is evidence and not a tautology."""
    if not warp_available():
        pytest.skip("warp is not installed on this machine")
    patch = TriangleMeshPatch.from_soup(sphere_mesh((0.0, 0.0, 0.0), 0.25, 16, 32), level=2)
    field = _sunlit_field(patch)
    points = _on_mesh(patch, 4000, seed=11)

    bridge = MeshPointBridge([MeshBinding("/World/Ball", field)])
    warp_face, warp_bary, warp_distance = bridge.locate(field, points)
    oracle_face, oracle_bary, oracle_distance = closest_point_on_mesh(
        patch.vertices_m, patch.faces, points
    )
    assert float((warp_face == oracle_face).mean()) == 1.0
    assert float(np.max(np.abs(warp_distance - oracle_distance))) < 1e-6

    temperatures = np.asarray(field.temperature_at(SETTLE_S), dtype=np.float64)
    warp_cell = patch.cell_of(warp_face, warp_bary[:, 0], warp_bary[:, 1])
    oracle_cell = patch.cell_of(oracle_face, oracle_bary[:, 0], oracle_bary[:, 1])
    assert np.array_equal(warp_cell, oracle_cell)
    assert np.array_equal(temperatures[warp_cell], temperatures[oracle_cell])
    # And the fixture really is curved: the sampled temperatures span the sphere's own gradient.
    assert float(np.ptp(temperatures[warp_cell])) > 25.0


@pytest.mark.slow
def test_the_warp_and_numpy_routes_produce_the_same_frame() -> None:
    """One switch chooses between them, so the whole `apply` is compared and not only `locate`."""
    if not warp_available():
        pytest.skip("warp is not installed on this machine")
    patch = TriangleMeshPatch.from_soup(sphere_mesh((0.0, 0.0, 0.0), 0.25, 16, 32), level=2)
    field = _sunlit_field(patch)
    ids, labels, positions, plane = _frame(patch, ident=4, path="/World/Ball")

    kwargs = dict(known_paths=["/World/Ball"])
    fast = MeshPointBridge([MeshBinding("/World/Ball", field)], use_warp=True, **kwargs)
    slow = MeshPointBridge([MeshBinding("/World/Ball", field)], use_warp=False, **kwargs)
    assert np.array_equal(
        fast.apply(plane, ids, labels, positions, SETTLE_S),
        slow.apply(plane, ids, labels, positions, SETTLE_S),
    )


# --- the bar: a curved prim stops being one temperature ----------------------------------------


def _frame(
    patch: TriangleMeshPatch, ident: int, path: str, size: int = 48, fallback: float = 300.0
):  # type: ignore[no-untyped-def]
    """A synthetic G-buffer over a prim: the instance-id and world-position planes Isaac supplies,
    built here by sampling the mesh so the test needs no renderer."""
    points = _on_mesh(patch, size * size, seed=5).reshape(size, size, 3)
    ids = np.full((size, size), ident, dtype=np.int32)
    labels = {str(ident): path, "0": "BACKGROUND"}
    plane = np.full((size, size), fallback, dtype=np.float32)
    return ids, labels, points, plane


@pytest.mark.slow
def test_an_exhaust_pipe_stops_being_one_temperature() -> None:
    """ADR 0087 lists a pipe as Hard: it is not near-planar, so the projection bridge leaves it at
    one value for the whole prim. Through the mesh bridge its pixels carry the gradient the solve
    produced -- 30 K around the circumference between the side facing the sun and the side facing
    away -- where the per-instance plane it replaced was flat to the bit."""
    patch = TriangleMeshPatch.from_soup(pipe_mesh(0.03, 0.8, 24, 8), level=1)
    field = _sunlit_field(patch)
    ids, labels, positions, plane = _frame(patch, ident=7, path="/World/Car/exhaust")

    bridge = MeshPointBridge(
        [MeshBinding("/World/Car/exhaust", field)], known_paths=["/World/Car/exhaust"]
    )
    out = bridge.apply(plane, ids, labels, positions, SETTLE_S)

    assert float(np.ptp(plane)) == 0.0  # what the prim carried before: one number
    assert float(np.ptp(out)) > 25.0  # and after: the pipe's own circumferential gradient
    assert bridge.last_coverage["/World/Car/exhaust"] == ids.size
    # Every pixel took a real cell temperature, not the fallback.
    assert not np.any(out == np.float32(300.0))


def test_an_unbound_prim_keeps_the_per_instance_value_bit_identically() -> None:
    """Additive, exactly as the planar bridge is. A frame with nothing bound in it comes back
    unchanged -- which is what makes attaching this to an existing scene a no-op."""
    patch = TriangleMeshPatch.from_soup(sphere_mesh((0.0, 0.0, 0.0), 0.25, 8, 16), level=1)
    field = _sunlit_field(patch)
    ids, labels, positions, plane = _frame(patch, ident=4, path="/World/Ball")

    # The bound prim is not the one on screen.
    labels_elsewhere = {"4": "/World/Other", "0": "BACKGROUND"}
    bridge = MeshPointBridge([MeshBinding("/World/Ball", field)])
    out = bridge.apply(plane, ids, labels_elsewhere, positions, SETTLE_S)
    assert np.array_equal(out, plane)
    assert bridge.last_coverage["/World/Ball"] == 0

    # Two prims on screen, one bound: the other's pixels are untouched to the bit.
    mixed_ids = np.where(np.arange(ids.size).reshape(ids.shape) % 2 == 0, 4, 9).astype(np.int32)
    mixed_plane = plane.copy()
    mixed_plane[mixed_ids == 9] = np.float32(271.35)
    out = MeshPointBridge([MeshBinding("/World/Ball", field)]).apply(
        mixed_plane, mixed_ids, {"4": "/World/Ball", "9": "/World/Road"}, positions, SETTLE_S
    )
    assert np.array_equal(out[mixed_ids == 9], mixed_plane[mixed_ids == 9])
    assert not np.array_equal(out[mixed_ids == 4], mixed_plane[mixed_ids == 4])


# --- the bar: a hit beyond the tolerance raises ----------------------------------------------


def test_a_pixel_too_far_from_the_mesh_raises_rather_than_snapping() -> None:
    """The id plane says this pixel is the ball; the position plane puts it 50 mm away. Something
    disagrees -- a stale transform, the wrong mesh bound, a prim that moved between the two AOVs
    -- and snapping it to the nearest cell would paint the ball with a plausible number."""
    patch = TriangleMeshPatch.from_soup(sphere_mesh((0.0, 0.0, 0.0), 0.25, 8, 16), level=1)
    field = _sunlit_field(patch)
    ids, labels, positions, plane = _frame(patch, ident=4, path="/World/Ball")
    positions = positions.copy()
    positions[0, 0] *= 1.2  # 50 mm off a 0.25 m sphere

    bridge = MeshPointBridge([MeshBinding("/World/Ball", field)])
    with pytest.raises(ValueError, match="further than"):
        bridge.apply(plane, ids, labels, positions, SETTLE_S)

    # Non-strict keeps the per-instance value for that pixel and nothing else changes.
    out = bridge.apply(plane, ids, labels, positions, SETTLE_S, strict=False)
    assert out[0, 0] == plane[0, 0]
    assert bridge.last_coverage["/World/Ball"] == ids.size - 1
    # A tolerance wide enough to accept it does, which shows the refusal is the tolerance and not
    # a failure to find the point at all.
    generous = MeshPointBridge([MeshBinding("/World/Ball", field)], max_distance_m=0.1)
    assert generous.apply(plane, ids, labels, positions, SETTLE_S)[0, 0] != plane[0, 0]


def test_the_tolerance_is_the_position_budget_and_must_be_positive() -> None:
    assert pytest.approx(7.0e-3) == DEFAULT_MAX_DISTANCE_M  # 2x ADR 0014's 3.4 mm
    with pytest.raises(ValueError, match="max_distance_m must be positive"):
        MeshPointBridge([], max_distance_m=0.0)


# --- parity with the planar bridge ---------------------------------------------------------------


def test_a_binding_to_a_prim_the_stage_does_not_know_raises_at_construction() -> None:
    """`PT.19`'s rule, carried over: a misspelt path is caught before any frame, not left as a
    prim quietly sitting at its per-instance fallback."""
    patch = TriangleMeshPatch.from_soup(sphere_mesh((0.0, 0.0, 0.0), 0.25, 8, 16), level=1)
    field = _sunlit_field(patch)
    with pytest.raises(ValueError, match="not among the stage's prims"):
        MeshPointBridge([MeshBinding("/World/Bal", field)], known_paths=["/World/Ball"])
    with pytest.raises(ValueError, match="a binding needs a prim path"):
        MeshBinding("", field)
    bridge = MeshPointBridge([MeshBinding("/World/Ball", field)], known_paths=["/World/Ball"])
    assert bridge.prim_paths == ("/World/Ball",)
    assert bridge.local_frames == ()


def test_a_mesh_in_a_prims_own_frame_needs_that_prims_transform() -> None:
    """PT.5's rule on a mesh: a field authored in a prim's local frame rides the prim, and world
    positions have to come back through its transform before they meet the mesh."""
    patch = TriangleMeshPatch.from_soup(
        sphere_mesh((0.0, 0.0, 0.0), 0.25, 8, 16), level=1, frame="/World/Ball"
    )
    field = _sunlit_field(patch)
    ids, labels, local, plane = _frame(patch, ident=4, path="/World/Ball")
    bridge = MeshPointBridge([MeshBinding("/World/Ball", field)])
    assert bridge.local_frames == ("/World/Ball",)

    shift = np.array([3.0, -1.0, 0.5])
    world_from_local = np.eye(4)
    world_from_local[3, :3] = shift  # row-vector convention, as transform_points uses
    world = local + shift

    out = bridge.apply(
        plane, ids, labels, world, SETTLE_S, world_from_local={"/World/Ball": world_from_local}
    )
    direct = bridge.apply(
        plane,
        ids,
        labels,
        local,
        SETTLE_S,
        world_from_local={"/World/Ball": np.eye(4)},
    )
    assert np.array_equal(out, direct)
    with pytest.raises(KeyError, match="no world_from_local matrix"):
        bridge.apply(plane, ids, labels, world, SETTLE_S)


def test_the_nearest_of_several_meshes_on_one_prim_wins() -> None:
    """A body with two mesh fields bound to it must not depend on the order they were listed."""
    near = TriangleMeshPatch.from_soup(sphere_mesh((0.0, 0.0, 0.0), 0.25, 8, 16), level=1)
    far = TriangleMeshPatch.from_soup(sphere_mesh((2.0, 0.0, 0.0), 0.25, 8, 16), level=1)
    hot = _sunlit_field(near, dni=900.0)
    cold = _sunlit_field(far, dni=0.0)
    ids, labels, positions, plane = _frame(near, ident=4, path="/World/Body")

    forward = MeshPointBridge(
        [MeshBinding("/World/Body", hot), MeshBinding("/World/Body", cold)]
    ).apply(plane, ids, labels, positions, SETTLE_S)
    backward = MeshPointBridge(
        [MeshBinding("/World/Body", cold), MeshBinding("/World/Body", hot)]
    ).apply(plane, ids, labels, positions, SETTLE_S)
    assert np.array_equal(forward, backward)
    assert float(np.ptp(forward)) > 25.0  # it took the near, sunlit sphere


def test_the_planes_are_checked_before_anything_is_sampled() -> None:
    patch = TriangleMeshPatch.from_soup(sphere_mesh((0.0, 0.0, 0.0), 0.25, 8, 16), level=1)
    field = _sunlit_field(patch)
    ids, labels, positions, plane = _frame(patch, ident=4, path="/World/Ball")
    bridge = MeshPointBridge([MeshBinding("/World/Ball", field)])
    with pytest.raises(TypeError, match="must be an integer plane"):
        bridge.apply(plane, ids.astype(np.float32), labels, positions, SETTLE_S)
    with pytest.raises(ValueError, match="does not match the plane"):
        bridge.apply(plane, ids[:-1], labels, positions, SETTLE_S)
    with pytest.raises(ValueError, match="does not match the plane"):
        bridge.apply(plane, ids, labels, positions[..., :2], SETTLE_S)
