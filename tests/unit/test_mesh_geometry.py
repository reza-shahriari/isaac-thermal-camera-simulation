"""What a mesh cell can see, traced against geometry (WM.4).

`WM.7` gave each cell its own normal and left two terms analytic: the sky view as the tilt's
``(1 + n·up)/2``, and the beam as the surface's one ``shaded`` flag. Both are now traced. The
tests here are the ones that would fail if the trace were wrong in a way the picture would not
show -- an occluder that stops the wrong rays reads as plausible shading, and the only way to
tell is against an independent answer.

Three independent answers are used: the analytic ``(1 + n·up)/2`` where convexity makes it exact,
`PT.21`'s planar quadrature at the same point through a different occluder implementation, and
`PT.22`'s box-as-rectangles against the same box as triangles.
"""

from __future__ import annotations

import numpy as np
import pytest

from irsim.thermal.frames import ENU, WorldFrame
from irsim.thermal.mesh_field import TriangleMeshPatch
from irsim.thermal.mesh_geometry import (
    LIFT_M,
    cell_occluders,
    cell_origins,
    is_convex,
    mesh_sky_view,
    mesh_soup,
    mesh_sunlit_fraction,
)
from irsim.thermal.raycast import MeshOccluders, TriangleSoup, box_mesh, cylinder_mesh, sphere_mesh
from irsim.thermal.shadow import ShadowRectangle, box_faces
from irsim.thermal.skyview import sky_view_factors
from irsim.thermal.surface_field import PlanarPatch

UP = np.array([0.0, 0.0, 1.0])


def join(*soups: TriangleSoup) -> TriangleSoup:
    """Several soups as one, so a scene's parts can occlude each other in a single query."""
    vertices, faces, offset = [], [], 0
    for soup in soups:
        vertices.append(soup.vertices)
        faces.append(soup.faces + offset)
        offset += soup.vertices.shape[0]
    return TriangleSoup(np.concatenate(vertices), np.concatenate(faces))


def analytic_sky_view(patch: TriangleMeshPatch) -> np.ndarray:
    return np.clip(0.5 * (1.0 + patch.cell_normal @ UP), 0.0, 1.0)


# --- convexity: the licence for skipping the trace ---------------------------------------------


@pytest.mark.parametrize(
    "soup",
    [
        cylinder_mesh([0.0, 0.0, 1.0], 0.015, 0.30, [0.0, 1.0, 0.0], n_phi=24, n_z=4),
        cylinder_mesh([0.0, 0.0, 1.0], 0.015, 0.30, [0.0, 1.0, 0.0], n_phi=24, capped=False),
        sphere_mesh([0.0, 0.0, 1.0], 0.2, 8, 16),
        box_mesh([0.0, 0.0, 1.0], [0.4, 0.4, 0.4]),
    ],
    ids=["tube", "open tube", "sphere", "box"],
)
def test_tracing_a_convex_mesh_returns_the_analytic_factor_to_the_bit(soup) -> None:  # type: ignore[no-untyped-def]
    """The claim the auto-skip rests on, measured rather than argued.

    A ray leaving a convex body into the outward hemisphere of the face it left cannot come back,
    so the traced quadrature must reduce to ``(1 + n·up)/2`` exactly -- not nearly. It does,
    because the gated sum and the open sum are then the *same additions in the same order* and
    their ratio is 1.0 to the bit. If this ever drifted, skipping the trace would be an
    approximation nobody declared.
    """
    patch = TriangleMeshPatch.from_soup(soup, level=1)
    forced = mesh_sky_view(patch, self_occluding=True)
    assert np.array_equal(forced, analytic_sky_view(patch))
    assert np.array_equal(mesh_sky_view(patch), forced)  # and the skip agrees with the trace


def test_the_convexity_test_separates_a_tube_from_a_fold() -> None:
    tube = cylinder_mesh([0.0, 0.0, 1.0], 0.015, 0.30, [0.0, 1.0, 0.0], n_phi=24, n_z=4)
    assert is_convex(tube)
    # A wall beside a floor slab: the classic geometry where a cell cannot see half the sky.
    fold = join(
        box_mesh([0.0, 0.0, 1.0], [0.1, 2.0, 2.0]), box_mesh([1.0, 0.0, 0.05], [1.9, 2.0, 0.1])
    )
    assert not is_convex(fold)


def test_an_inward_wound_mesh_is_not_convex() -> None:
    """Winding is part of the test on purpose. An inside-out mesh has normals pointing into the
    body, and nothing downstream should be allowed to skip a trace on the strength of them --
    `cylinder_mesh` shipped wound inward once, and only a picture caught it."""
    soup = box_mesh([0.0, 0.0, 1.0], [0.4, 0.4, 0.4])
    flipped = TriangleSoup(soup.vertices, soup.faces[:, ::-1].copy())
    assert is_convex(soup)
    assert not is_convex(flipped)


# --- the sky view -------------------------------------------------------------------------------


def test_a_mesh_cell_and_a_patch_cell_agree_about_the_sky_at_the_same_point() -> None:
    """`PT.21`'s answer and `WM.4`'s, for one point under one wall, through two different
    occluder implementations -- `cell_shadow`'s ray-rectangle test and `RectangleOccluders`'.

    They share the quadrature and nothing else, so an error in either ray test shows up here.
    With the origin lift off, the two agree **to the bit**: same directions, same weights, same
    order of accumulation.
    """
    wall = ShadowRectangle(
        centre_m=np.array([0.0, 0.0, 1.5]),
        u_axis=np.array([0.0, 1.0, 0.0]),
        v_axis=np.array([0.0, 0.0, 1.0]),
        half_u_m=3.0,
        half_v_m=1.5,
    )
    patch = PlanarPatch(
        origin_m=np.array([0.2, -0.3, 0.0]),
        u_axis=np.array([1.0, 0.0, 0.0]),
        v_axis=np.array([0.0, 1.0, 0.0]),
        n_u=1,
        n_v=1,
        du_m=0.6,
        dv_m=0.6,
    )
    planar = sky_view_factors(
        patch, patch.normal, 0.5 * (1.0 + patch.normal[2]), (wall,), ENU.to_world
    )
    centre = patch.cell_centres()[0]
    triangle = TriangleMeshPatch(
        vertices_m=np.array(
            [centre + [0.4, -0.2, 0.0], centre + [0.0, 0.4, 0.0], centre + [-0.4, -0.2, 0.0]]
        ),
        faces=np.array([[0, 1, 2]]),
        levels=np.array([1]),
    )
    traced = mesh_sky_view(triangle, (wall,), self_occluding=False, lift_m=0.0)
    assert float(traced[0]) == float(planar[0])
    assert 0.5 < float(traced[0]) < 1.0  # it is genuinely shaded, not trivially equal at 1.0


def test_the_foot_of_a_tall_wall_sees_half_the_sky() -> None:
    """The analytic anchor: an infinite wall hides exactly half the dome from its own foot."""
    soup = join(
        box_mesh([-0.01, 0.0, 30.0], [0.02, 60.0, 60.0]),  # the wall
        box_mesh([0.02, 0.0, -0.005], [0.04, 60.0, 0.01]),  # a sliver of floor at its foot
    )
    patch = TriangleMeshPatch.from_soup(soup, level=1)
    centres = patch.cell_centres()
    floor_top = (patch.cell_normal @ UP > 0.99) & (centres[:, 2] > -0.01) & (centres[:, 0] > 0.0)
    assert floor_top.sum() >= 1
    assert np.allclose(mesh_sky_view(patch)[floor_top], 0.5, atol=0.01)


def test_the_edge_of_a_wide_overhang_sees_half_the_sky() -> None:
    soup = join(
        box_mesh([0.0, 0.0, 3.0], [60.0, 60.0, 0.02]),  # the slab overhead
        box_mesh([30.02, 0.0, -0.005], [0.04, 60.0, 0.01]),  # ground just past its edge
    )
    patch = TriangleMeshPatch.from_soup(soup, level=1)
    centres = patch.cell_centres()
    ground = (patch.cell_normal @ UP > 0.99) & (centres[:, 2] > -0.01) & (centres[:, 0] > 29.0)
    assert ground.sum() >= 1
    assert np.allclose(mesh_sky_view(patch)[ground], 0.5, atol=0.01)


def test_shelter_only_takes_sky_away_and_takes_more_of_it_closer_in() -> None:
    """Two properties one wall must have: the trace can never *add* sky, and a cell closer to
    the wall must see less of it. A sign error in the ray test passes neither."""
    soup = join(
        box_mesh([0.0, 0.0, 1.0], [0.1, 4.0, 2.0]),
        box_mesh([1.05, 0.0, 0.005], [2.0, 4.0, 0.01]),
    )
    patch = TriangleMeshPatch.from_soup(soup, level=3)
    svf = mesh_sky_view(patch)
    analytic = analytic_sky_view(patch)
    assert np.all(svf <= analytic + 1e-12)

    # The floor slab's top face only: the wall's own roof also faces up, and it stands 2 m clear
    # of everything, so sorting it in with the floor would compare two different surfaces.
    centres = patch.cell_centres()
    floor = (patch.cell_normal @ UP > 0.99) & (centres[:, 2] < 0.02) & (centres[:, 0] > 0.06)
    x, value = centres[floor, 0], svf[floor]
    edges = np.linspace(x.min(), x.max(), 5)
    means = [
        float(value[(x >= lo) & (x < hi)].mean())
        for lo, hi in zip(edges[:-1], edges[1:], strict=True)
    ]
    assert np.all(np.diff(means) > 0.0), means
    assert means[-1] - means[0] > 0.1, means


def test_a_cell_facing_straight_down_is_given_no_sky_rather_than_a_nan() -> None:
    """Its whole hemisphere is below the horizon, so the open quadrature is empty and the ratio
    would be 0/0. The answer is zero sky, and it has to be a number: it multiplies the longwave
    down, and one NaN there poisons a whole field's solve."""
    soup = box_mesh([0.0, 0.0, 1.0], [0.4, 0.4, 0.4])
    patch = TriangleMeshPatch.from_soup(soup, level=1)
    svf = mesh_sky_view(patch, self_occluding=True)
    down = patch.cell_normal @ UP < -0.99
    assert down.sum() >= 1
    assert np.all(np.isfinite(svf))
    assert np.all(svf[down] == 0.0)


def test_the_dome_is_rotated_into_the_scene_s_own_frame() -> None:
    """A Y-up scene and a Z-up one describing the same geometry must trace the same rays. The
    car scenes are Y-up; a dome built in ENU and never rotated would shade them sideways."""
    z_up = join(
        box_mesh([0.0, 0.0, 1.0], [0.1, 4.0, 2.0]), box_mesh([1.05, 0.0, 0.005], [2.0, 4.0, 0.01])
    )
    y_up = TriangleSoup(z_up.vertices[:, [0, 2, 1]].copy(), z_up.faces[:, ::-1].copy())
    a = mesh_sky_view(TriangleMeshPatch.from_soup(z_up, level=2))
    b = mesh_sky_view(
        TriangleMeshPatch.from_soup(y_up, level=2),
        frame=WorldFrame(up=np.array([0.0, 1.0, 0.0]), north=np.array([0.0, 0.0, -1.0])),
    )
    assert np.allclose(np.sort(a), np.sort(b), atol=1e-12)


# --- the beam -----------------------------------------------------------------------------------


def test_the_box_as_rectangles_and_the_box_as_triangles_cast_the_same_shadow() -> None:
    """`PT.22`'s cross-check, at mesh cell origins: the exact ray-rectangle oracle against
    Moeller-Trumbore on the same box. Two implementations, one shadow, cell for cell."""
    ground = TriangleMeshPatch.from_soup(box_mesh([0.0, 0.0, -0.05], [6.0, 6.0, 0.1]), level=4)
    rectangles = box_faces([0.0, 0.0, 1.0], [1.0, 1.0, 2.0])
    triangles = MeshOccluders((box_mesh([0.0, 0.0, 1.0], [1.0, 1.0, 2.0]),))
    sun = np.array([0.4, 0.2, 0.9])
    by_rect = mesh_sunlit_fraction(ground, sun, rectangles, self_occluding=False)
    by_mesh = mesh_sunlit_fraction(ground, sun, triangles, self_occluding=False)
    assert np.array_equal(by_rect, by_mesh)
    assert 0.0 < float(by_rect.mean()) < 1.0  # something really is shaded


def test_a_tube_under_a_plate_goes_dark_where_the_cosine_leaves_it_lit() -> None:
    """The headline. `WM.7`'s arm was dark on its underside because its normal faced away; its
    crown was lit everywhere, including the half of it tucked under the airframe. Here the plate
    covers the near half of the tube, and the cells it covers face the sun and receive none of
    it -- which is the difference between a cosine and a shadow."""
    tube = cylinder_mesh([0.0, 0.0, 0.3], 0.015, 0.4, [0.0, 1.0, 0.0], n_phi=24, n_z=4)
    patch = TriangleMeshPatch.from_soup(tube, level=1)
    plate = ShadowRectangle(
        centre_m=np.array([0.0, -0.1, 0.36]),
        u_axis=np.array([1.0, 0.0, 0.0]),
        v_axis=np.array([0.0, 1.0, 0.0]),
        half_u_m=0.2,
        half_v_m=0.1,
    )
    sun = np.array([0.0, 0.0, 1.0])
    lit = mesh_sunlit_fraction(patch, sun, (plate,))
    crown = patch.cell_normal @ UP > 0.7
    y = patch.cell_centres()[:, 1]
    under = crown & (y < -0.02)
    clear = crown & (y > 0.02)
    assert under.sum() > 0 and clear.sum() > 0
    assert np.all(lit[under] == 0.0), "a crown cell under the plate still saw the sun"
    assert np.all(lit[clear] == 1.0), "a crown cell past the plate lost the sun it should have"


def test_the_sun_s_disc_turns_the_terminator_into_a_ramp() -> None:
    """`PT.22`'s penumbra, on mesh cells: one ray gives a hard edge and every cell reads 0 or 1;
    19 rays put a band of partly-lit cells along it."""
    ground = TriangleMeshPatch.from_soup(box_mesh([0.0, 0.0, -0.05], [6.0, 6.0, 0.1]), level=4)
    rectangles = box_faces([0.0, 0.0, 2.0], [1.0, 1.0, 0.1])
    sun = np.array([0.3, 0.0, 0.95])
    hard = mesh_sunlit_fraction(ground, sun, rectangles, 1, self_occluding=False)
    soft = mesh_sunlit_fraction(ground, sun, rectangles, 19, self_occluding=False)
    assert not np.any((hard > 0.0) & (hard < 1.0))
    assert np.any((soft > 0.0) & (soft < 1.0))


def test_a_convex_mesh_with_nothing_around_it_traces_nothing() -> None:
    """`cell_occluders` returning None is what keeps every mesh shipped so far bit-identical to
    `WM.7`, and what keeps a scene's build from paying for a trace whose answer it knows."""
    patch = TriangleMeshPatch.from_soup(sphere_mesh([0.0, 0.0, 1.0], 0.2, 8, 16), level=1)
    assert cell_occluders(patch, ()) is None
    assert np.array_equal(mesh_sunlit_fraction(patch, [0.0, 0.0, 1.0]), np.ones(patch.n_cells))
    assert cell_occluders(patch, (), self_occluding=True) is not None


def test_the_origins_are_lifted_along_each_cell_s_own_normal() -> None:
    """One shared lift would push a tube's cells sideways off their own faces. The distance from
    the centroid is the same for every cell and the direction is not."""
    patch = TriangleMeshPatch.from_soup(
        cylinder_mesh([0.0, 0.0, 1.0], 0.015, 0.30, [0.0, 1.0, 0.0], n_phi=24, n_z=4), level=1
    )
    offset = cell_origins(patch) - patch.cell_centres()
    assert np.allclose(np.linalg.norm(offset, axis=-1), LIFT_M)
    unit = offset / np.linalg.norm(offset, axis=-1)[:, None]
    assert np.allclose(np.einsum("cx,cx->c", unit, patch.cell_normal), 1.0)
    assert float(np.ptp(unit[:, 2])) > 1.0  # the directions genuinely differ around the tube


def test_a_soup_of_the_patch_is_the_patch_s_own_triangles() -> None:
    soup = box_mesh([0.0, 0.0, 1.0], [0.4, 0.4, 0.4])
    patch = TriangleMeshPatch.from_soup(soup, level=3)
    again = mesh_soup(patch)
    assert np.array_equal(again.vertices, soup.vertices)
    assert np.array_equal(again.faces, soup.faces)
    assert again.n_faces == patch.n_faces  # subdivision is the field's, not the occluder's
