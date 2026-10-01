"""TC.13 (ADR 0165) -- coarse thermal cells on an over-tessellated mesh.

A fine tube (7,680 faces) coarsened at 2 cm keeps every square millimetre of its area, puts its
cell centroids on the surface to a fraction of a cell, points its normals outward, and sends a
pixel's face straight to its cluster. Its conduction is a sheet's: the heat crossing a plane of
a rod under a unit gradient is ``k δ · circumference`` to a few percent, and the operator is
symmetric with no self-links. Solved against the fine field on the same tube, the coarse field
tracks the fine one's cluster means to within a tenth of the transient's amplitude, at a
hundredth of the cells. TC.17 (ADR 0176): a planar-dissolved tube, whose faces run its whole
length, is bisected before it is clustered, and then conducts and makes contact as a sheet should.
"""

from __future__ import annotations

import numpy as np
import pytest

from irsim.thermal.facets import FacetForcing, FacetProperties
from irsim.thermal.mesh_coarse import CoarseMeshPatch, coarse_lateral_operator, coarsen_mesh
from irsim.thermal.mesh_conduction import mesh_lateral_operator
from irsim.thermal.mesh_field import TriangleMeshField, TriangleMeshPatch
from irsim.thermal.raycast import cylinder_mesh

RADIUS = 0.02
LENGTH = 0.40
TUBE = cylinder_mesh(
    np.array([0.0, 0.0, 0.0]),
    RADIUS,
    LENGTH,
    axis=(1.0, 0.0, 0.0),
    n_phi=96,
    n_z=40,
    capped=False,
)
CELL = 0.02


@pytest.fixture(scope="module")
def coarse() -> CoarseMeshPatch:
    return coarsen_mesh(TUBE.vertices, TUBE.faces, CELL)


def test_the_area_is_exact_the_cells_sit_on_the_skin_and_face_outward(
    coarse: CoarseMeshPatch,
) -> None:
    fine = TriangleMeshPatch.uniform(TUBE.vertices, TUBE.faces, 1)
    assert coarse.n_faces == fine.n_faces == 7680
    assert 40 < coarse.n_cells < fine.n_cells / 50, coarse.n_cells  # 80 on this tube
    assert coarse.area_m2 == pytest.approx(fine.area_m2, rel=1e-12)
    assert coarse.cell_area_m2.sum() == pytest.approx(fine.area_m2, rel=1e-12)
    assert np.all(coarse.cell_area_m2 > 0.0)
    centres = coarse.cell_centres()
    radial = np.linalg.norm(centres[:, 1:], axis=-1)
    assert np.all(np.abs(radial - RADIUS) < 0.25 * CELL), (
        "centroids within a quarter cell of the skin"
    )
    normals = coarse.cell_normal
    assert np.allclose(np.linalg.norm(normals, axis=-1), 1.0)
    outward = np.einsum("ij,ij->i", normals[:, 1:], centres[:, 1:]) / radial
    assert np.all(outward > 0.9), "every cluster normal points out of the tube"
    # a face resolves straight to its cluster, whatever the barycentric point
    face = np.arange(0, coarse.n_faces, 97)
    cells = coarse.cell_of(face, np.full(face.size, 0.2), np.full(face.size, 0.3))
    assert np.array_equal(cells, coarse.face_cluster[face])
    assert np.array_equal(coarse.face_mean(np.arange(coarse.n_cells)), coarse.face_cluster)
    with pytest.raises(NotImplementedError, match="fine TriangleMeshPatch"):
        _ = coarse.cell_face


def test_the_coarse_conduction_is_a_sheet_s(coarse: CoarseMeshPatch) -> None:
    """A unit axial gradient on a tube of thickness δ carries k δ · 2πr watts through any
    cross-section; the coarse operator's net flux across the mid-plane says the same to 10 %."""
    k, delta = 200.0, 0.001
    op = coarse_lateral_operator(coarse, k, delta)
    assert op is not None
    kmat = op.conductance_w_k
    assert kmat.shape == (coarse.n_cells, coarse.n_cells) and not kmat.diagonal().any()
    assert abs(kmat - kmat.T).max() == 0.0
    x = coarse.cell_centres()[:, 0]
    t = x.copy()  # 1 K per metre along the axis
    left = x < 0.0
    # heat flowing from the right half into the left half under T = x
    flux = 0.0
    coo = kmat.tocoo()
    for i, j, g in zip(coo.row, coo.col, coo.data, strict=True):
        if left[i] and not left[j]:
            flux += g * (t[j] - t[i])
    expected = k * delta * 2.0 * np.pi * RADIUS
    assert flux == pytest.approx(expected, rel=0.10), (flux, expected)
    assert coarse_lateral_operator(coarse, 0.0, delta) is None


def _field(patch: object, conduction: object, n: int, hot: np.ndarray) -> TriangleMeshField:
    props = FacetProperties(
        heat_capacity_j_m2_k=np.full(n, 2700.0),
        emissivity=np.zeros(n),
        solar_absorptivity=np.zeros(n),
    )
    return TriangleMeshField(
        patch,  # type: ignore[arg-type]
        props,
        lambda t: FacetForcing(t_air_k=290.0, h_w_m2_k=0.0),
        0.0,
        np.where(hot, 310.0, 290.0),
        5.0,
        conduction=conduction,  # type: ignore[arg-type]
    )


def test_the_coarse_field_tracks_the_fine_field_at_a_hundredth_of_the_cells(
    coarse: CoarseMeshPatch,
) -> None:
    """One end starts 20 K hot, adiabatic otherwise: conduction alone moves the heat along the
    tube. The fine solve's mean over each cluster and the coarse solve's cell agree to a tenth
    of the 20 K step at every tick, and both conserve the energy exactly."""
    fine = TriangleMeshPatch.uniform(TUBE.vertices, TUBE.faces, 1)
    k, delta = 200.0, 0.001
    hot_fine = fine.cell_centres()[:, 0] < -0.15
    hot_coarse = coarse.cell_centres()[:, 0] < -0.15
    f_field = _field(fine, mesh_lateral_operator(fine, k, delta), fine.n_cells, hot_fine)
    c_field = _field(coarse, coarse_lateral_operator(coarse, k, delta), coarse.n_cells, hot_coarse)
    area_f, area_c = fine.cell_area_m2, coarse.cell_area_m2
    e_f0 = float((2700.0 * area_f * f_field.temperature_at(0.0)).sum())
    e_c0 = float((2700.0 * area_c * c_field.temperature_at(0.0)).sum())
    spread: dict[float, float] = {}
    for t in (60.0, 300.0, 900.0):
        f_field.advance_to(t)
        c_field.advance_to(t)
        tf = np.asarray(f_field.temperature_at(t), dtype=np.float64)
        tc = np.asarray(c_field.temperature_at(t), dtype=np.float64)
        spread[t] = float(tc.max() - tc.min())
        # the fine field's area-weighted mean over each cluster's faces
        cluster_mean = np.bincount(
            coarse.face_cluster, weights=area_f * tf, minlength=coarse.n_cells
        )
        cluster_mean /= area_c
        assert np.abs(tc - cluster_mean).max() < 2.0, (t, np.abs(tc - cluster_mean).max())
        assert float((2700.0 * area_f * tf).sum()) == pytest.approx(
            e_f0, rel=1e-6
        )  # float32 read-back
        assert float((2700.0 * area_c * tc).sum()) == pytest.approx(e_c0, rel=1e-6)
    assert spread[60.0] > 5.0 and spread[900.0] < 1.0, spread  # the step spreads, then levels


# --- the scene ------------------------------------------------------------------------------------


def test_a_scene_solves_a_coarse_arm_next_to_a_fine_one(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """`coarse: true` on a mesh surface builds a `CoarseMeshPatch` (ADR 0165) that the scene
    spins up, solves and binds like any other: the north arm coarsened at 2 cm carries a tenth
    of the fine cells, sits within 1.5 K of the fine solve's mean at the same instant and shows
    the same crown-over-underside sign at a high sun."""
    import pathlib

    from irsim.scene import Scene

    repo = pathlib.Path(__file__).resolve().parents[2]
    text = (repo / "configs/scenes/quad_flight_mesh.yaml").read_text()
    text = text.replace("spin_up_hours: 48.0", "spin_up_hours: 1.0")
    head, arm, rest = text.partition("      - name: arm_n\n")
    arm_block, sep, tail = (arm + rest).partition("      - name: arm_e\n")
    assert "level: 1\n" in arm_block
    coarse_block = arm_block.replace("level: 1\n", "cell_m: 0.02\n          coarse: true\n")
    fine, coarse = tmp_path / "fine.yaml", tmp_path / "coarse.yaml"
    fine.write_text(head + arm_block + sep + tail)
    coarse.write_text(head + coarse_block + sep + tail)
    s_fine, s_coarse = Scene.from_file(fine), Scene.from_file(coarse)
    p_fine, p_coarse = s_fine.mesh_fields["arm_n"].patch, s_coarse.mesh_fields["arm_n"].patch
    assert isinstance(p_coarse, CoarseMeshPatch) and not isinstance(p_fine, CoarseMeshPatch)
    assert p_coarse.n_cells < p_fine.n_cells / 5, (p_coarse.n_cells, p_fine.n_cells)
    assert p_coarse.area_m2 == pytest.approx(p_fine.area_m2, rel=1e-9)
    t = s_fine.t0_s + 300.0
    for scene in (s_fine, s_coarse):
        scene.mesh_fields["arm_n"].advance_to(t)
    m_fine = s_fine.mesh_fields["arm_n"].mean_temperature_k(t)
    m_coarse = s_coarse.mesh_fields["arm_n"].mean_temperature_k(t)
    assert abs(m_fine - m_coarse) < 1.5, (m_fine, m_coarse)
    up = np.array([0.0, 0.0, 1.0])

    def contrast(scene) -> float:  # type: ignore[no-untyped-def]
        field = scene.mesh_fields["arm_n"]
        cells = np.asarray(field.temperature_at(t), dtype=np.float64)
        n_up = field.patch.cell_normal @ up
        return float(cells[n_up > 0.7].mean() - cells[n_up < -0.7].mean())

    c_fine, c_coarse = contrast(s_fine), contrast(s_coarse)
    assert np.sign(c_fine) == np.sign(c_coarse) and abs(c_fine) > 0.5
    assert abs(c_coarse - c_fine) < 0.5 * abs(c_fine) + 0.5, (c_fine, c_coarse)


# --- TC.17: a face longer than a cell is bisected before it is clustered -------------------------
#
# A planar-dissolved archive (the DJI Inspire 3's arm tubes) is the opposite of the tube above:
# each side facet is two triangles running the tube's whole length. A face joins one cell whole,
# so clustered as-is every side cell sits at the middle of the tube.

STRIP = cylinder_mesh(
    np.array([0.0, 0.0, 0.0]),
    0.01,
    0.30,
    axis=(1.0, 0.0, 0.0),
    n_phi=48,
    n_z=1,
    capped=True,
)


def _clustered_whole(vertices: np.ndarray, faces: np.ndarray, cell: float) -> CoarseMeshPatch:
    """The pre-TC.17 clustering, kept as the control: faces clustered as they come."""
    v, f = np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.intp)
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    voxel = np.floor((a + b + c) / 3.0 / cell).astype(np.int64)
    octant = (np.cross(b - a, c - a) >= 0.0).astype(np.int64)
    _, cluster = np.unique(np.concatenate([voxel, octant], axis=1), axis=0, return_inverse=True)
    return CoarseMeshPatch(v, f, np.asarray(cluster, dtype=np.intp).reshape(-1), cell)


def _edge_use(faces: np.ndarray) -> np.ndarray:
    key = np.sort(np.stack([faces, np.roll(faces, -1, axis=1)], axis=-1).reshape(-1, 2), axis=1)
    return np.unique(key, axis=0, return_counts=True)[1]


def test_long_faces_are_bisected_on_the_same_surface_without_a_seam() -> None:
    from irsim.thermal.mesh_coarse import split_long_edges
    from irsim.thermal.mesh_field import closest_point_on_mesh

    v, f = split_long_edges(STRIP.vertices, STRIP.faces, CELL)
    edges = np.concatenate(
        [np.linalg.norm(v[f[:, k]] - v[f[:, (k + 1) % 3]], axis=-1) for k in range(3)]
    )
    assert edges.max() <= CELL
    before = TriangleMeshPatch.uniform(STRIP.vertices, STRIP.faces, 1).area_m2
    after = TriangleMeshPatch.uniform(v, f, 1).area_m2
    assert after == pytest.approx(before, rel=1e-12), "no area moves"
    _, _, distance = closest_point_on_mesh(STRIP.vertices, STRIP.faces, v)
    assert distance.max() < 1e-12, "every new vertex lies on the surface it split"
    # the capped strip tube is closed; conforming refinement keeps every edge on exactly two faces
    assert set(_edge_use(STRIP.faces)) == {2} and set(_edge_use(f)) == {2}, "no T-junction"


def test_a_strip_tube_conducts_along_its_length_once_its_faces_are_split() -> None:
    """The sheet check of `test_the_coarse_conduction_is_a_sheet_s`, on the dissolved tube: under
    a unit axial gradient the flux across the mid-plane is k δ · 2πr. Clustered whole, the side
    cells sit at x = ±L/6, joined by diagonal edges the tube's whole length, and the operator is
    out by orders of magnitude (~460x too conductive here)."""
    k, delta, radius = 200.0, 0.001, 0.01

    def midplane_flux(patch: CoarseMeshPatch) -> float:
        op = coarse_lateral_operator(patch, k, delta)
        if op is None:
            return 0.0
        x = patch.cell_centres()[:, 0]
        coo = op.conductance_w_k.tocoo()
        left = x < 0.0
        return float(
            sum(
                g * (x[j] - x[i])
                for i, j, g in zip(coo.row, coo.col, coo.data, strict=True)
                if left[i] and not left[j]
            )
        )

    expected = k * delta * 2.0 * np.pi * radius
    split = coarsen_mesh(STRIP.vertices, STRIP.faces, CELL)
    whole = _clustered_whole(STRIP.vertices, STRIP.faces, CELL)
    side = np.abs(np.linalg.norm(split.cell_centres()[:, 1:], axis=-1) - radius) < 0.25 * CELL
    assert np.ptp(split.cell_centres()[side, 0]) > 0.25, "side cells run the tube's length"
    assert midplane_flux(split) == pytest.approx(expected, rel=0.10)
    control = midplane_flux(whole) / expected
    assert abs(np.log10(control)) > 1.0, f"control: whole strips are out by {control:.0f}x"


def test_a_contact_finds_the_ends_of_two_strip_tubes_that_meet() -> None:
    """Two dissolved tubes butt-jointed at x = 0, as an Inspire 3 arm meets its junction: the
    contactor finds cells within the gap at the joint and only there. Clustered whole, the
    nearest cells are a tube's half-length apart and the contact is refused."""
    from irsim.thermal.coupling import proximity_contactor

    def tube(x0: float) -> tuple[np.ndarray, np.ndarray]:
        t = cylinder_mesh(
            np.array([x0, 0.0, 0.0]),
            0.01,
            0.30,
            axis=(1.0, 0.0, 0.0),
            n_phi=48,
            n_z=1,
            capped=False,
        )
        return t.vertices, t.faces

    (va, fa), (vb, fb) = tube(-0.15), tube(0.15)
    a, b = coarsen_mesh(va, fa, CELL), coarsen_mesh(vb, fb, CELL)
    k_ab = proximity_contactor(a, b, 1000.0, 0.001, gap_m=1.5 * CELL).tocoo()
    assert k_ab.sum() == pytest.approx(1.0)  # h_c · A
    assert np.abs(a.cell_centres()[k_ab.row, 0]).max() < 2.0 * CELL
    assert np.abs(b.cell_centres()[k_ab.col, 0]).max() < 2.0 * CELL
    with pytest.raises(ValueError, match="do not touch"):
        proximity_contactor(
            _clustered_whole(va, fa, CELL),
            _clustered_whole(vb, fb, CELL),
            1000.0,
            0.001,
            gap_m=1.5 * CELL,
        )
