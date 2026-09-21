"""PT.22 — shadows from geometry, neighbours included, and the sun's finite disc.

Three claims. The mesh path and the rectangle path agree cell-for-cell on a box occluder, so
swapping the representation cannot change the physics. A neighbouring block shades a wall that
the wall's own mesh never can, which is what "the query walks every opaque prim" means. And the
0.53 deg solar disc, sampled on 7 to 37 rays, makes the terminator a ramp `d tan(0.53 deg)`
wide -- 9.3 mm per metre of standoff, held to 10 % -- where a binary test gives a step of zero
width and fails outright.

docs/physics-model.md §5.2, §6.1; ADR 0095, ADR 0107; roadmap PT.22.
"""

from __future__ import annotations

import math
import pathlib

import numpy as np
import pytest

from irsim.config.scene import load_scene_config
from irsim.radiometry.constants import SOLAR_DISC_DIAMETER_DEG
from irsim.scene import Scene
from irsim.thermal.raycast import (
    AnyOccluders,
    MeshOccluders,
    Occluders,
    RectangleOccluders,
    TriangleSoup,
    box_mesh,
    penumbra_width_m,
    rectangle_mesh,
    solar_disc_rays,
    sunlit_fraction,
)
from irsim.thermal.shadow import ShadowRectangle, box_faces, cell_shadow
from irsim.thermal.surface_field import PlanarPatch

REPO = pathlib.Path(__file__).resolve().parents[2]
EX = np.array([1.0, 0.0, 0.0])
EY = np.array([0.0, 1.0, 0.0])
EZ = np.array([0.0, 0.0, 1.0])


def _ground(n: int = 20, cell: float = 0.1) -> PlanarPatch:
    half = 0.5 * n * cell
    return PlanarPatch(
        origin_m=np.array([-half, -half, 0.0]),
        u_axis=EX,
        v_axis=EY,
        n_u=n,
        n_v=n,
        du_m=cell,
        dv_m=cell,
        thickness_m=0.05,
    )


# --- the adapter, and the rectangles as the oracle ---------------------------------------------


def test_the_rectangle_occluder_is_cell_shadow_generalised_to_per_ray_origins() -> None:
    """The oracle: the same exact ray-rectangle test, so nothing about the physics moved."""
    patch = _ground()
    rect = ShadowRectangle(
        centre_m=np.array([0.0, 0.0, 1.0]), u_axis=EX, v_axis=EY, half_u_m=0.4, half_v_m=0.4
    )
    sun = np.array([0.3, 0.2, 1.0])
    sun /= np.linalg.norm(sun)
    query = RectangleOccluders((rect,))
    assert isinstance(query, Occluders)
    lit = 1.0 - query.blocked(patch.cell_centres(), sun).astype(float)
    assert np.array_equal(lit, cell_shadow(patch, sun, [rect]))
    assert 0.0 < lit.mean() < 1.0, "the fixture must actually cast a shadow"
    # One ray on the disc *is* the hard edge, to the bit -- so turning the disc on is the only
    # thing that changes a number, and leaving it off changes nothing.
    assert np.array_equal(sunlit_fraction(patch, sun, [rect], 1), cell_shadow(patch, sun, [rect]))
    # A ray that ends before the occluder is not blocked by it.
    near = query.blocked(patch.cell_centres(), sun, max_distance_m=0.1)
    assert not near.any()


def test_the_mesh_path_and_the_rectangle_path_agree_cell_for_cell_on_a_box() -> None:
    """A box as six rectangles and as twelve triangles: the same shadow, cell for cell."""
    patch = _ground()
    centre, size = np.array([0.0, 0.0, 1.5]), np.array([1.0, 1.0, 0.6])
    sun = np.array([0.3, 0.2, 1.0])
    sun /= np.linalg.norm(sun)
    rects = RectangleOccluders(box_faces(centre, size))
    mesh = MeshOccluders((box_mesh(centre, size),))
    from_rects = rects.blocked(patch.cell_centres(), sun)
    from_mesh = mesh.blocked(patch.cell_centres(), sun)
    assert np.array_equal(from_rects, from_mesh)
    assert 0.0 < from_mesh.mean() < 1.0, from_mesh.mean()
    # And from several directions, including a low sun that meets the box's side.
    for elevation in (10.0, 30.0, 60.0, 85.0):
        s = np.array([math.cos(math.radians(elevation)), 0.13, math.sin(math.radians(elevation))])
        assert np.array_equal(
            rects.blocked(patch.cell_centres(), s), mesh.blocked(patch.cell_centres(), s)
        ), elevation
    # A single rectangle's two triangles are the same surface too.
    rect = ShadowRectangle(
        centre_m=np.array([0.1, -0.2, 0.8]), u_axis=EX, v_axis=EY, half_u_m=0.3, half_v_m=0.5
    )
    assert np.array_equal(
        RectangleOccluders((rect,)).blocked(patch.cell_centres(), sun),
        MeshOccluders((rectangle_mesh(rect),)).blocked(patch.cell_centres(), sun),
    )


def test_a_neighbouring_block_shades_a_wall_its_own_mesh_cannot() -> None:
    """Self-shadowing is what a surface's own normal already gives; the interesting shadow comes
    from something else in the scene, and the query has to walk that too."""
    # A wall in the x-z plane at y = 0, facing west (-x); cells over its lower half.
    wall = PlanarPatch(
        origin_m=np.array([0.0, -1.5, 0.0]),
        u_axis=EY,
        v_axis=EZ,
        n_u=12,
        n_v=12,
        du_m=0.25,
        dv_m=0.25,
        thickness_m=0.15,
    )
    own = box_mesh([0.5, 0.0, 1.5], [1.0, 3.0, 3.0])  # the wall's own building
    neighbour = box_mesh([-2.0, 0.0, 1.0], [1.5, 3.0, 2.0])  # a lower block 1.5 m to the west
    sun = np.array([-1.0, 0.0, 0.55])  # a low western sun, straight onto the wall
    sun /= np.linalg.norm(sun)
    alone = sunlit_fraction(wall, sun, MeshOccluders((own,)), 1)
    together = sunlit_fraction(wall, sun, MeshOccluders((own, neighbour)), 1)
    assert alone.mean() == 1.0, "a wall's own box must not shade the wall"
    assert 0.0 < together.mean() < 1.0, together.mean()
    assert together[together == 0.0].size > 10
    # The shadow is on the lower cells: the block is shorter than the wall.
    uv = wall.local_coords(wall.cell_centres())
    assert together[uv[:, 1] < 1.0].mean() < together[uv[:, 1] > 2.0].mean()
    # Rectangles and meshes answer one query together.
    both = AnyOccluders(
        (MeshOccluders((own,)), RectangleOccluders(box_faces([-2.0, 0.0, 1.0], [1.5, 3.0, 2.0])))
    )
    assert np.array_equal(
        both.blocked(wall.cell_centres(), sun),
        MeshOccluders((own, neighbour)).blocked(wall.cell_centres(), sun),
    )


# --- the sun's disc ------------------------------------------------------------------------------


def test_the_disc_rays_cover_it_to_the_limb_and_weigh_it_correctly() -> None:
    for n in (1, 7, 19, 37):
        directions, weights = solar_disc_rays(EZ, n)
        assert directions.shape == (n, 3) and weights.shape == (n,)
        assert float(weights.sum()) == pytest.approx(1.0, abs=1e-12)
        assert np.all(weights > 0.0)
        assert np.allclose(np.linalg.norm(directions, axis=1), 1.0)
        angles = np.degrees(np.arccos(np.clip(directions @ EZ, -1.0, 1.0)))
        limb = 0.5 * SOLAR_DISC_DIAMETER_DEG
        assert angles.max() == pytest.approx(limb if n > 1 else 0.0, abs=1e-6)
    # It works about any direction, not just the pole the basis branches on.
    for axis in (EX, EY, EZ, np.array([0.4, -0.5, 0.77])):
        directions, _ = solar_disc_rays(axis, 19)
        assert directions[0] == pytest.approx(axis / np.linalg.norm(axis))
    with pytest.raises(ValueError, match="n_rays must be one of"):
        solar_disc_rays(EZ, 12)
    with pytest.raises(ValueError, match="zero length"):
        solar_disc_rays(np.zeros(3), 7)


@pytest.mark.parametrize("standoff_m", [0.5, 2.0, 6.0])
def test_the_penumbra_ramp_is_d_tan_half_a_degree_within_ten_per_cent(standoff_m: float) -> None:
    """An infinite edge ``d`` above the ground under an overhead sun. The partly lit cells are
    the penumbra, and they span `d tan(0.53 deg)` = 9.3 mm per metre. Measured at 2 m: 18.0 mm
    against 18.6 mm expected, the difference being the 1 mm cell the ramp is sampled on."""
    expected = penumbra_width_m(standoff_m)
    cell = expected / 20.0
    edge = ShadowRectangle(
        centre_m=np.array([0.0, 500.0, standoff_m]),
        u_axis=EX,
        v_axis=EY,
        half_u_m=500.0,
        half_v_m=500.0,
    )
    span = 4.0 * expected
    fine = PlanarPatch(
        origin_m=np.array([-0.5 * cell, -0.5 * span, 0.0]),
        u_axis=EX,
        v_axis=EY,
        n_u=1,
        n_v=int(span / cell),
        du_m=cell,
        dv_m=cell,
        thickness_m=0.01,
    )
    ys = fine.local_coords(fine.cell_centres())[:, 1]
    for n_rays in (7, 19, 37):
        f = sunlit_fraction(fine, EZ, [edge], n_rays)
        partial = (f > 1e-12) & (f < 1.0 - 1e-12)
        assert partial.any(), n_rays
        width = float(ys[partial].max() - ys[partial].min() + cell)
        assert abs(width - expected) / expected < 0.10, (n_rays, width, expected)
        # Fully lit on the open side, fully dark under the edge, monotone between.
        assert f[ys < ys[partial].min()].min() == 1.0
        assert f[ys > ys[partial].max()].max() == 0.0
        assert np.all(np.diff(f) <= 1e-12)
    # A binary test fails it: the step has no width at all.
    hard = sunlit_fraction(fine, EZ, [edge], 1)
    assert not ((hard > 1e-12) & (hard < 1.0 - 1e-12)).any()


def test_a_soup_refuses_what_it_cannot_index_and_rejects_rays_that_miss_its_box() -> None:
    with pytest.raises(ValueError, match="vertices must be"):
        TriangleSoup(np.zeros((2, 3)), np.array([[0, 1, 0]]))
    with pytest.raises(ValueError, match="indexes a vertex"):
        TriangleSoup(np.eye(3), np.array([[0, 1, 5]]))
    with pytest.raises(ValueError, match="three positive extents"):
        box_mesh([0.0, 0.0, 0.0], [1.0, 0.0, 1.0])
    soup = box_mesh([0.0, 0.0, 5.0], [1.0, 1.0, 1.0])
    lo, hi = soup.bounds_m
    assert soup.n_faces == 12 and np.allclose(hi - lo, 1.0)
    # Rays pointing away from the box, and rays in the wrong half-space, miss it.
    origins = np.zeros((4, 3))
    assert not soup.blocked(origins, np.array([0.0, 0.0, -1.0])).any()
    assert soup.blocked(origins, np.array([0.0, 0.0, 1.0])).all()


# --- the scene ---------------------------------------------------------------------------------


def test_a_scene_can_turn_the_disc_on_and_one_ray_is_the_shadow_it_always_had(tmp_path) -> None:  # type: ignore[no-untyped-def]
    text = (REPO / "configs/scenes/wall_half_in_sun.yaml").read_text()
    assert (
        load_scene_config(REPO / "configs/scenes/wall_half_in_sun.yaml").scene.thermal.penumbra_rays
        == 1
    )
    soft = tmp_path / "soft.yaml"
    soft.write_text(
        text.replace("    spin_up_hours: 48.0\n", "    spin_up_hours: 1.0\n    penumbra_rays: 19\n")
    )
    spec = load_scene_config(soft).scene
    assert spec.thermal.penumbra_rays == 19
    built = Scene.from_file(soft)
    forcing = built.surface_fields["west_concrete"].field.forcing_at
    assert forcing.penumbra_rays == 19
    visibility = forcing.cell_visibility(built.t0_s)
    # The neighbour is 2 m from the wall, so its penumbra is ~19 mm against 25 cm cells: a thin
    # band of partly lit cells along the terminator, and everything else still 0 or 1.
    partial = (visibility > 1e-9) & (visibility < 1.0 - 1e-9)
    assert partial.any() and partial.mean() < 0.2, partial.mean()
    assert visibility.min() == 0.0 and visibility.max() == 1.0

    bad = tmp_path / "bad.yaml"
    bad.write_text(text.replace("    spin_up_hours: 48.0\n", "    penumbra_rays: 12\n"))
    with pytest.raises(ValueError, match="penumbra_rays must be one of"):
        load_scene_config(bad)
    lonely = tmp_path / "lonely.yaml"
    lonely.write_text(
        (REPO / "configs/scenes/parked_car_cabin.yaml")
        .read_text()
        .replace("    spin_up_hours: 48.0\n", "    spin_up_hours: 48.0\n    penumbra_rays: 7\n")
    )
    with pytest.raises(ValueError, match="sharpens nothing without"):
        load_scene_config(lonely)


def test_the_numpy_soup_agrees_with_trimesh_where_trimesh_is_installed() -> None:
    """An independent intersector on the same triangles. `trimesh` is optional and not a
    dependency of the core (CLAUDE.md #1): the rectangles are the oracle, and this is a third
    opinion where the library happens to be present. It is skipped on the CI gate."""
    trimesh = pytest.importorskip("trimesh")
    soup = box_mesh([0.0, 0.0, 1.5], [1.0, 1.0, 0.6])
    mesh = trimesh.Trimesh(vertices=soup.vertices, faces=soup.faces, process=False)
    patch = _ground()
    origins = patch.cell_centres()
    rng = np.random.default_rng(22)
    for _ in range(4):
        sun = np.array([rng.uniform(-0.6, 0.6), rng.uniform(-0.6, 0.6), 1.0])
        sun /= np.linalg.norm(sun)
        theirs = mesh.ray.intersects_any(origins, np.broadcast_to(sun, origins.shape))
        assert np.array_equal(soup.blocked(origins, sun), np.asarray(theirs, dtype=bool))
