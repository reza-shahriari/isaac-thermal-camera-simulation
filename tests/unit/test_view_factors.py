"""TC.9 -- view factors between arbitrary meshes by ray casting, occlusion included.

ADR 0088's closed form is a point facing a parallel rectangle and raises on anything else, so a
car could not radiate onto the road and a battery could not warm the shell around it.
`irsim.thermal.view_factors` traces the general case, and these tests hold it to the oracles a
view factor has: Howell's closed forms for parallel rectangles (C-11), perpendicular rectangles
with a common edge (C-14) and coaxial discs (C-41) to 1 %; reciprocity A_i F_ij = A_j F_ji; a
closed enclosure whose rows sum to one; and a blocker that takes a view away.
docs/physics-model.md §6.1, §6.6; ADR 0088; roadmap TC.9.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from irsim.thermal.raycast import TriangleSoup, box_mesh
from irsim.thermal.view_factors import (
    RadiantMesh,
    first_hit,
    howell_coaxial_discs,
    howell_parallel_rectangles,
    howell_perpendicular_rectangles,
    view_factors,
)


def rectangle(centre, u, v, half_u: float, half_v: float, n: int = 4) -> TriangleSoup:  # type: ignore[no-untyped-def]
    """An ``n × n`` quad mesh of the rectangle; its faces radiate toward ``u × v``."""
    centre, u, v = (np.asarray(x, dtype=np.float64) for x in (centre, u, v))
    us = np.linspace(-half_u, half_u, n + 1)
    vs = np.linspace(-half_v, half_v, n + 1)
    vertices = np.array([centre + a * u + b * v for b in vs for a in us])
    faces = []
    for j in range(n):
        for i in range(n):
            p = j * (n + 1) + i
            faces += [[p, p + 1, p + n + 2], [p, p + n + 2, p + n + 1]]
    return TriangleSoup(vertices, np.array(faces))


def disc(centre, radius_m: float, facing_up: bool, n_r: int = 3, n_t: int = 16) -> TriangleSoup:  # type: ignore[no-untyped-def]
    """A fan-and-rings ``n_t``-gon disc in a horizontal plane."""
    centre = np.asarray(centre, dtype=np.float64)
    points = [centre]
    rings = []
    for k in range(1, n_r + 1):
        r = radius_m * k / n_r
        start = len(points)
        for m in range(n_t):
            angle = 2.0 * math.pi * m / n_t
            points.append(centre + np.array([r * math.cos(angle), r * math.sin(angle), 0.0]))
        rings.append(list(range(start, start + n_t)))
    faces = [[0, rings[0][m], rings[0][(m + 1) % n_t]] for m in range(n_t)]
    for k in range(1, n_r):
        inner, outer = rings[k - 1], rings[k]
        for m in range(n_t):
            m1 = (m + 1) % n_t
            faces += [[inner[m], outer[m], outer[m1]], [inner[m], outer[m1], inner[m1]]]
    idx = np.array(faces)
    if not facing_up:
        idx = idx[:, [0, 2, 1]]
    return TriangleSoup(np.array(points), idx)


def _equal_area_radius(radius_m: float, n_t: int) -> float:
    """The radius of the disc with the ``n_t``-gon's area, so the oracle sees what was meshed."""
    return radius_m * math.sqrt(n_t * math.sin(2.0 * math.pi / n_t) / (2.0 * math.pi))


# --- Howell's closed forms ---------------------------------------------------------------------


def test_parallel_squares_match_howell_c11_to_one_percent() -> None:
    """Unit squares half a unit apart: F = 0.4153 (C-11). A point-to-rectangle form would need
    an integral over the source; the trace is that integral."""
    a = RadiantMesh("a", rectangle([0, 0, 0], [1, 0, 0], [0, 1, 0], 0.5, 0.5))
    b = RadiantMesh("b", rectangle([0, 0, 0.5], [0, 1, 0], [1, 0, 0], 0.5, 0.5))  # faces down
    vf = view_factors([a, b])
    expect = howell_parallel_rectangles(1.0, 1.0, 0.5)
    assert vf.body[0, 1] == pytest.approx(expect, rel=1e-2)
    assert vf.body[1, 0] == pytest.approx(expect, rel=1e-2)
    assert vf.body_sky[0] == pytest.approx(1.0 - expect, rel=1e-2)


def test_perpendicular_rectangles_match_howell_c14_to_one_percent() -> None:
    """A 1 × 1 floor and a 0.5 × 1 wall on its edge: F_floor→wall = 0.1462 (C-14), and the
    wall's factor back is twice that by reciprocity, since it has half the area."""
    floor = RadiantMesh("floor", rectangle([0.5, 0.5, 0], [1, 0, 0], [0, 1, 0], 0.5, 0.5))
    wall = RadiantMesh("wall", rectangle([0, 0.5, 0.25], [0, 1, 0], [0, 0, 1], 0.5, 0.25))
    vf = view_factors([floor, wall])
    expect = howell_perpendicular_rectangles(0.5, 1.0, 1.0)
    assert vf.body[0, 1] == pytest.approx(expect, rel=1e-2)
    assert vf.body[1, 0] == pytest.approx(2.0 * expect, rel=1e-2)
    assert vf.reciprocity_defect() < 1e-2


@pytest.mark.slow
def test_coaxial_discs_match_howell_c41_to_one_percent() -> None:
    """Unit discs a unit apart: F = 0.382 (C-41) for true discs; the 16-gons meshed here have
    2.5 % less area, so the oracle is evaluated at the equal-area radius (0.987), where the
    trace lands within 1 %."""
    top = RadiantMesh("top", disc([0, 0, 1], 1.0, facing_up=False))
    bottom = RadiantMesh("bottom", disc([0, 0, 0], 1.0, facing_up=True))
    vf = view_factors([bottom, top])
    r = _equal_area_radius(1.0, 16)
    expect = howell_coaxial_discs(r, r, 1.0)
    assert vf.body[0, 1] == pytest.approx(expect, rel=1e-2)
    assert vf.reciprocity_defect() < 1e-2


# --- the identities ----------------------------------------------------------------------------


def test_a_closed_box_sums_every_row_to_one_and_sees_no_sky() -> None:
    """Inside a closed enclosure every ray lands somewhere: Σ_j F_ij = 1 to the quadrature's own
    rounding, and the sky share is exactly zero."""
    box = box_mesh([0.0, 0.0, 0.0], [1.0, 2.0, 0.5])
    inward = TriangleSoup(box.vertices, box.faces[:, [0, 2, 1]])
    vf = view_factors([RadiantMesh("box", inward)])
    rows = vf.face.sum(axis=1)
    assert np.allclose(rows, 1.0, atol=1e-12), rows
    assert np.all(vf.sky == 0.0)
    assert vf.body[0, 0] == pytest.approx(1.0, abs=1e-12)
    # single faces are two ray fans each, so face-level reciprocity is the fan's resolution:
    # 9 % here, against 0 at the body level (one body, trivially) -- reported, not the claim
    assert vf.reciprocity_defect("face") < 0.15
    assert np.all(vf.back == 0.0)


def test_reciprocity_holds_between_bodies_of_different_size() -> None:
    """A_i F_ij = A_j F_ji: a small plate under a large one. Body level, to 1 %; the face level
    is the ray fan's own resolution and is reported, not asserted tight."""
    small = RadiantMesh("small", rectangle([0, 0, 0], [1, 0, 0], [0, 1, 0], 0.25, 0.25, n=2))
    large = RadiantMesh("large", rectangle([0.3, 0.2, 0.6], [0, 1, 0], [1, 0, 0], 1.0, 1.0, n=4))
    vf = view_factors([small, large])
    a_small, a_large = small.area_m2, large.area_m2
    assert a_small * vf.body[0, 1] == pytest.approx(a_large * vf.body[1, 0], rel=1e-2)
    assert vf.reciprocity_defect() < 1e-2
    assert 0.0 < vf.body[0, 1] < 1.0 and vf.body[1, 0] < vf.body[0, 1]


# --- occlusion ---------------------------------------------------------------------------------


def test_a_blocker_between_two_plates_takes_the_view_and_the_sky() -> None:
    """The occluded case ADR 0088 could not express: a wide plate between two unit squares
    is the nearest hit for every ray, so F_a→b is exactly zero, F_a→blocker is what the squares
    exchanged plus the sky the blocker now hides, and what a loses in sky it gains in blocker."""
    a = RadiantMesh("a", rectangle([0, 0, 0], [1, 0, 0], [0, 1, 0], 0.5, 0.5))
    b = RadiantMesh("b", rectangle([0, 0, 1], [0, 1, 0], [1, 0, 0], 0.5, 0.5))
    open_pair = view_factors([a, b])
    blocker = RadiantMesh("blk", rectangle([0, 0, 0.5], [0, 1, 0], [1, 0, 0], 2.0, 2.0, n=2))
    blocked = view_factors([a, b, blocker])
    assert blocked.body[0, 1] == 0.0
    assert blocked.body[0, 2] > open_pair.body[0, 1]
    assert blocked.body_sky[0] < open_pair.body_sky[0]
    assert blocked.body[0, 2] + blocked.body_sky[0] == pytest.approx(1.0, abs=1e-12)
    # b faces down onto the blocker's *back*: the blocker is opaque, so b's rays stop there,
    # but a back is not a radiating face -- F_b→blk is zero and the share is reported as back,
    # which is what keeps A_b F_b→blk = A_blk F_blk→b (the blocker never sees b) exact
    assert blocked.body[1, 2] == 0.0 and blocked.body[1, 0] == 0.0
    assert blocked.body_back[1] == pytest.approx(blocked.body[0, 2], rel=1e-12)
    assert blocked.body_sky[1] + blocked.body_back[1] == pytest.approx(1.0, abs=1e-12)
    assert blocked.reciprocity_defect() < 1e-2  # a <-> blocker, two independent fans


def test_a_partial_blocker_removes_exactly_the_share_it_covers() -> None:
    """A strip over half of a's hemisphere toward b takes about half of F_a→b and gives it to
    itself, and the sum over everything a sees plus its sky is still one."""
    a = RadiantMesh("a", rectangle([0, 0, 0], [1, 0, 0], [0, 1, 0], 0.5, 0.5))
    b = RadiantMesh("b", rectangle([0, 0, 1], [0, 1, 0], [1, 0, 0], 0.5, 0.5))
    strip = RadiantMesh("strip", rectangle([0, 0.25, 0.5], [0, 1, 0], [1, 0, 0], 0.25, 2.0, n=2))
    vf = view_factors([a, b, strip])
    open_pair = view_factors([a, b])
    lost = open_pair.body[0, 1] - vf.body[0, 1]
    assert 0.3 * open_pair.body[0, 1] < lost < 0.7 * open_pair.body[0, 1], lost
    assert vf.body[0].sum() + vf.body_sky[0] == pytest.approx(1.0, abs=1e-12)


# --- the ray query underneath -------------------------------------------------------------------


def test_first_hit_returns_the_nearest_face_and_misses_honestly() -> None:
    soup = rectangle([0, 0, 1], [1, 0, 0], [0, 1, 0], 0.5, 0.5, n=1)
    far = rectangle([0, 0, 2], [1, 0, 0], [0, 1, 0], 0.5, 0.5, n=1)
    both = TriangleSoup(
        np.concatenate([soup.vertices, far.vertices]),
        np.concatenate([soup.faces, far.faces + soup.vertices.shape[0]]),
    )
    origins = np.array([[0.1, 0.1, 0.0], [0.1, 0.1, 1.5], [3.0, 3.0, 0.0]])
    distance, face = first_hit(both, origins, np.array([0.0, 0.0, 1.0]))
    assert distance[0] == pytest.approx(1.0) and face[0] in (0, 1)
    assert distance[1] == pytest.approx(0.5) and face[1] in (2, 3)
    assert math.isinf(distance[2]) and face[2] == -1


def test_the_bodies_are_checked() -> None:
    a = RadiantMesh("a", rectangle([0, 0, 0], [1, 0, 0], [0, 1, 0], 0.5, 0.5, n=1))
    with pytest.raises(ValueError, match="own name"):
        view_factors([a, a])
    with pytest.raises(ValueError, match="at least one"):
        view_factors([])
    with pytest.raises(ValueError, match="subdivide"):
        view_factors([a], subdivide=0)
    with pytest.raises(ValueError, match="degenerate"):
        collinear = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        _ = RadiantMesh("flat", TriangleSoup(collinear, np.array([[0, 1, 2]]))).face_normals
