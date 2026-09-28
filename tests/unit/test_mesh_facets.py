"""Facet angles are exact on analytic shapes, and the resolvable angle follows the sagitta (AI.10).

Every number here is geometry with a closed form: an N-segment cylinder's side edges are exactly
360/N degrees, its cap edges exactly 90°, a chord's miss is r(1 − cos α/2). A metric that is
wrong by a weld, a winding or a band boundary gets one of them wrong. ADR 0155.
"""

from __future__ import annotations

import json
import math
import struct

import numpy as np
import pytest

from irsim.io.mesh_facets import (
    edge_dihedrals,
    edge_sagitta_m,
    facet_report,
    glb_meshes,
    max_facet_angle_deg,
    sagitta_m,
)


def cylinder(n: int, radius: float = 0.05, height: float = 0.1, split: bool = False):
    """A closed N-segment cylinder, outward-wound; ``split`` unwelds it the way glTF does."""
    t = np.arange(n) * 2 * math.pi / n
    ring = np.stack([radius * np.cos(t), radius * np.sin(t)], axis=1)
    bottom = np.c_[ring, np.zeros(n)]
    top = np.c_[ring, np.full(n, height)]
    verts = np.vstack([bottom, top, [[0, 0, 0]], [[0, 0, height]]])
    cb, ct = 2 * n, 2 * n + 1
    faces = []
    for i in range(n):
        j = (i + 1) % n
        faces += [[i, j, n + j], [i, n + j, n + i]]  # side quad
        faces += [[cb, j, i]]  # bottom cap, facing -z
        faces += [[ct, n + i, n + j]]  # top cap, facing +z
    faces = np.array(faces)
    if split:
        verts = verts[faces.reshape(-1)]
        faces = np.arange(len(verts)).reshape(-1, 3)
    return verts.astype(np.float64), faces


def box():
    v = np.array([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)], dtype=np.float64)
    quads = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    faces = [[a, b, c] for a, b, c, _ in quads] + [[a, c, d] for a, _, c, d in quads]
    return v, np.array(faces)


# ------------------------------------------------------------------------ exact dihedral angles


@pytest.mark.parametrize("n", [6, 8, 12, 36])
def test_an_n_segment_cylinder_has_side_edges_of_exactly_360_over_n(n: int) -> None:
    d = edge_dihedrals(*cylinder(n))
    angles = np.round(d.angle_deg, 9)
    assert np.isclose(angles, 360.0 / n).sum() == n  # the n vertical seams between facets
    assert np.isclose(angles, 90.0).sum() == 2 * n  # the rim edges to each cap
    assert d.boundary_edges == 0 and d.nonmanifold_edges == 0


def test_a_glTF_style_split_mesh_measures_the_same_as_a_welded_one() -> None:
    # Unwelded, every edge would be a boundary and the metric would see no curve at all.
    welded = edge_dihedrals(*cylinder(12))
    split = edge_dihedrals(*cylinder(12, split=True))
    assert split.boundary_edges == 0
    np.testing.assert_allclose(np.sort(split.angle_deg), np.sort(welded.angle_deg), atol=1e-9)


def test_a_flipped_face_does_not_change_the_measured_angle() -> None:
    v, f = cylinder(12)
    f = f.copy()
    f[0] = f[0][::-1]  # one side triangle wound backwards, as game exports often are
    np.testing.assert_allclose(
        np.sort(edge_dihedrals(v, f).angle_deg),
        np.sort(edge_dihedrals(*cylinder(12)).angle_deg),
        atol=1e-9,
    )


def test_the_diagonal_of_a_flat_quad_is_zero_degrees() -> None:
    d = edge_dihedrals(*box())
    assert np.isclose(d.angle_deg, 0.0).sum() == 6  # one diagonal per face
    assert np.isclose(d.angle_deg, 90.0).sum() == 12  # the cube's edges


# ------------------------------------------------------------------------------- the bands


def test_a_low_poly_8_segment_cylinder_is_faceted_and_a_72_segment_one_is_not() -> None:
    coarse = facet_report("hub8", *cylinder(8))
    fine = facet_report("hub72", *cylinder(72))
    assert not coarse.passed and coarse.max_faceted_deg == pytest.approx(45.0)
    assert fine.passed and fine.faceted_length_m == 0.0


def test_a_cube_is_all_crease_and_passes() -> None:
    assert facet_report("cube", *box()).faceted_fraction == 0.0


def test_the_fraction_is_by_edge_length_not_by_count() -> None:
    # 8 seams of height 0.1 m in a cylinder whose rims are 8 x 2 x chord long.
    r, h, n = 0.05, 0.1, 8
    report = facet_report("hub8", *cylinder(n, radius=r, height=h))
    chord = 2 * r * math.sin(math.pi / n)
    diagonals = n * math.hypot(chord, h)  # flat quad diagonals, 0 deg
    spokes = 2 * n * r  # flat cap fan edges, 0 deg
    total = n * h + 2 * n * chord + diagonals + spokes
    assert report.faceted_fraction == pytest.approx(n * h / total, rel=1e-9)


# ------------------------------------------------------------------------ the sagitta bound


def test_sagitta_has_its_closed_form() -> None:
    assert sagitta_m(1.0, 10.0) == pytest.approx(1 - math.cos(math.radians(5)))
    assert sagitta_m(1.0, 180.0) == pytest.approx(1.0)  # a diameter misses by the radius


def test_the_resolvable_angle_puts_the_sagitta_at_exactly_half_a_pixel() -> None:
    r, rng, ifov = 0.05, 2.0, 1e-3
    alpha = max_facet_angle_deg(r, rng, ifov)
    assert sagitta_m(r, alpha) == pytest.approx(0.5 * rng * ifov, rel=1e-12)


def test_closer_or_finer_cameras_demand_finer_facets() -> None:
    assert max_facet_angle_deg(0.05, 1.0, 1e-3) < max_facet_angle_deg(0.05, 10.0, 1e-3)
    assert max_facet_angle_deg(0.05, 2.0, 0.5e-3) < max_facet_angle_deg(0.05, 2.0, 1e-3)


def test_a_curve_under_half_a_pixel_needs_no_facets() -> None:
    assert max_facet_angle_deg(0.001, 100.0, 1e-3) == 180.0


def test_bad_inputs_are_refused() -> None:
    with pytest.raises(ValueError):
        max_facet_angle_deg(0.0, 1.0, 1e-3)
    with pytest.raises(ValueError):
        facet_report("x", *box(), smooth_max_deg=60.0, crease_min_deg=10.0)


# ------------------------------------------------------------------------------ reading a .glb


def write_glb(path, parts: list[tuple[np.ndarray, np.ndarray]], name: str = "hub") -> None:
    """One glTF mesh whose primitives are ``parts`` — the way a CAD writer emits B-rep faces."""
    blob, views, accessors, prims = b"", [], [], []
    for verts, faces in parts:
        for array, kind, comp in [
            (verts.astype(np.float32), "VEC3", 5126),
            (faces.astype(np.uint32).reshape(-1), "SCALAR", 5125),
        ]:
            raw = array.tobytes()
            views.append({"buffer": 0, "byteOffset": len(blob), "byteLength": len(raw)})
            acc = {"bufferView": len(views) - 1, "componentType": comp, "type": kind}
            acc["count"] = len(array)
            accessors.append(acc)
            blob += raw
        prims.append(
            {"attributes": {"POSITION": len(accessors) - 2}, "indices": len(accessors) - 1}
        )
    doc = {
        "asset": {"version": "2.0"},
        "meshes": [{"name": name, "primitives": prims}],
        "bufferViews": views,
        "accessors": accessors,
        "buffers": [{"byteLength": len(blob)}],
    }
    js = json.dumps(doc).encode()
    js += b" " * (-len(js) % 4)
    blob += b"\0" * (-len(blob) % 4)
    body = (
        struct.pack("<I4s", len(js), b"JSON") + js + struct.pack("<I4s", len(blob), b"BIN\0") + blob
    )
    path.write_bytes(struct.pack("<4sII", b"glTF", 2, 12 + len(body)) + body)


def test_a_glb_mesh_split_into_primitives_is_measured_as_one_welded_solid(tmp_path) -> None:
    v, f = cylinder(12)
    sides = f[[i for i in range(len(f)) if i % 4 < 2]]
    caps = f[[i for i in range(len(f)) if i % 4 >= 2]]
    path = tmp_path / "hub.glb"
    write_glb(path, [(v, sides), (v, caps)])
    [(name, verts, faces)] = glb_meshes(path)
    d = edge_dihedrals(verts, faces)
    assert name == "hub"
    assert d.boundary_edges == 0  # the rims join across the two primitives
    np.testing.assert_allclose(
        np.sort(d.angle_deg), np.sort(edge_dihedrals(*cylinder(12)).angle_deg), atol=1e-4
    )


def test_a_file_that_is_not_a_glb_is_refused(tmp_path) -> None:
    path = tmp_path / "x.glb"
    path.write_bytes(b"not a gltf at all")
    with pytest.raises(ValueError, match="not a binary glTF"):
        glb_meshes(path)


# ------------------------------------------------- each edge's own silhouette error, no radius


@pytest.mark.parametrize("n", [6, 8, 24])
def test_an_edge_knows_its_sagitta_without_being_told_the_radius(n: int) -> None:
    # (w/2) tan(theta/4) with w the chord and theta = 360/n is exactly r (1 - cos(pi/n)).
    r = 0.07
    d = edge_dihedrals(*cylinder(n, radius=r))
    seams = np.isclose(d.angle_deg, 360.0 / n)
    s = edge_sagitta_m(d.angle_deg[seams], d.width_m[seams])
    np.testing.assert_allclose(s, r * (1 - math.cos(math.pi / n)), rtol=1e-9)


def test_a_coarse_screw_head_passes_where_a_coarse_fuselage_fails() -> None:
    # Both are 8-sided: the angle alone cannot tell them apart. At 2 m on a 1.36 mrad pixel the
    # screw's 0.08 mm miss is sub-pixel and the fuselage's 15 mm miss is not.
    half_pixel = 0.5 * 2.0 * 1.36e-3
    screw = facet_report("screw", *cylinder(8, radius=1e-3, height=2e-3), max_sagitta_m=half_pixel)
    body = facet_report("fuselage", *cylinder(8, radius=0.2, height=1.0), max_sagitta_m=half_pixel)
    assert screw.passed and screw.faceted_length_m == 0.0
    assert not body.passed
    # Without the scene, the scale-free angle band flags both.
    assert not facet_report("screw", *cylinder(8, radius=1e-3, height=2e-3)).passed


def test_a_designed_crease_is_never_faceted_whatever_its_size() -> None:
    assert facet_report("big cube", *box(), max_sagitta_m=1e-6).passed
