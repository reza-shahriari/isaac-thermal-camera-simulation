"""WM.2 — a temperature field on a triangle mesh, for the geometry a plane cannot cover.

ADR 0087 gave a near-planar surface a field by projecting the position AOV onto a rectangle and
recorded curved geometry as the limit: a sphere, a wheel or a pipe takes one temperature for the
whole prim. `WM.1` measured that the parameterisation can be derived from the mesh instead of
transported by the renderer; this module is the solve that sits behind it.

Two claims carry the step:

* a sphere under a directional sun holds **each face** to the equilibrium its own cos θ gives, to
  well under a millikelvin, where one facet for the whole sphere is a single number across a
  span of tens of kelvin; and
* the change is **provably a redistribution**: under uniform forcing the mesh reproduces the
  per-prim scalar solve exactly, and under varying forcing its area-weighted mean departs from
  the per-prim value only in the direction and by the amount the concavity of `T(q)` requires.

docs/physics-model.md §6.1, §6.4; ADR 0087 (and its WM.1 addendum); roadmap WM.2.
"""

from __future__ import annotations

import numpy as np
import pytest

from irsim.thermal.balance import SurfaceForcing, ThermalProperties, steady_state_temperature
from irsim.thermal.facets import FacetForcing, FacetProperties
from irsim.thermal.mesh_field import TriangleMeshField, TriangleMeshPatch
from irsim.thermal.raycast import box_mesh, sphere_mesh

# A thin painted metal skin: the time constant is C/(4εσT³ + h) ≈ 130 s, so four hours is the
# root and not a snapshot of a transient.
SKIN = ThermalProperties(heat_capacity_j_m2_k=2000.0, emissivity=0.90, solar_absorptivity=0.60)
T_AIR_K = 293.15
H_W_M2_K = 10.0
LW_DOWN_W_M2 = 300.0
DNI_W_M2 = 900.0
SUN = np.asarray([0.0, 0.0, 1.0])  # straight down onto the sphere's north pole
SETTLE_S = 4.0 * 3600.0


def _properties(n: int) -> FacetProperties:
    return FacetProperties(
        heat_capacity_j_m2_k=np.full(n, SKIN.heat_capacity_j_m2_k),
        emissivity=np.full(n, SKIN.emissivity),
        solar_absorptivity=np.full(n, SKIN.solar_absorptivity),
    )


def _expected_k(q_solar: np.ndarray, q_internal: np.ndarray | float = 0.0) -> np.ndarray:
    internal = np.broadcast_to(np.asarray(q_internal, dtype=np.float64), np.shape(q_solar))
    return np.asarray(
        [
            steady_state_temperature(
                SKIN,
                SurfaceForcing(
                    t_air_k=T_AIR_K,
                    h_w_m2_k=H_W_M2_K,
                    q_solar_w_m2=float(q),
                    q_longwave_down_w_m2=LW_DOWN_W_M2,
                    q_internal_w_m2=float(i),
                ),
            )
            for q, i in zip(np.ravel(q_solar), np.ravel(internal), strict=True)
        ]
    )


def _settled(patch: TriangleMeshPatch, q_solar, q_internal=0.0) -> np.ndarray:  # type: ignore[no-untyped-def]
    field = TriangleMeshField(
        patch,
        _properties(patch.n_cells),
        lambda _t: FacetForcing(
            t_air_k=T_AIR_K,
            h_w_m2_k=H_W_M2_K,
            q_solar_w_m2=q_solar,
            q_longwave_down_w_m2=LW_DOWN_W_M2,
            q_internal_w_m2=q_internal,
        ),
        0.0,
        np.full(patch.n_cells, 290.0),
        1.0,
        keep_ticks=None,
    )
    field.advance_to(SETTLE_S)
    return np.asarray(field.temperature_at(SETTLE_S), dtype=np.float64)


# --- the bar: every face at its own equilibrium --------------------------------------------------


@pytest.mark.slow
def test_a_sphere_holds_every_face_to_its_own_cosine_equilibrium() -> None:
    """The measurement ADR 0087's curved-geometry limit costs. A 0.25 m sphere under an overhead
    sun: the cap facing the sun absorbs 900 W/m², the terminator absorbs nothing, and each face
    settles on the root of its own §6.1 balance to under a millikelvin. One facet for the whole
    sphere would carry a single number across that entire span, which is the defect."""
    soup = sphere_mesh((0.0, 0.0, 0.0), 0.25, 24, 48)
    patch = TriangleMeshPatch.from_soup(soup, level=1)
    cos = np.maximum(patch.cell_normal @ SUN, 0.0)
    q_solar = DNI_W_M2 * cos

    settled = _settled(patch, q_solar)
    expected = _expected_k(q_solar)
    assert float(np.max(np.abs(settled - expected))) < 1e-3, float(
        np.max(np.abs(settled - expected))
    )

    # The span this replaces: sunlit cap to shaded side, on one prim. Measured 34.3 K --
    # 320.3 K where the sun is overhead, 286.0 K on the half that never sees it.
    span = float(settled.max() - settled.min())
    assert span > 30.0, span
    # And it really is the cosine that orders them, not the solver drifting.
    lit = cos > 0.0
    order = np.argsort(cos[lit])
    assert np.all(np.diff(settled[lit][order]) >= -1e-9)


@pytest.mark.slow
def test_one_facet_for_the_whole_sphere_misses_by_the_pole_to_terminator_span() -> None:
    """The negative control the row names. Solve the same sphere as a single facet on its
    area-weighted mean flux -- which is what every prim in this project carried before a field --
    and it lands one number inside a 34 K spread, 25 K under the cap and 9 K over the far side."""
    soup = sphere_mesh((0.0, 0.0, 0.0), 0.25, 24, 48)
    patch = TriangleMeshPatch.from_soup(soup, level=1)
    cos = np.maximum(patch.cell_normal @ SUN, 0.0)
    q_solar = DNI_W_M2 * cos
    settled = _settled(patch, q_solar)

    areas = patch.cell_area_m2
    mean_q = float(np.sum(q_solar * areas) / np.sum(areas))
    per_prim = float(_expected_k(np.array([mean_q]))[0])
    # Measured: cap 320.3 K, per-prim 295.0 K, shaded 286.0 K. The gap is asymmetric because
    # half the sphere's area never sees the sun and drags the area-weighted mean flux down, which
    # is precisely the averaging error one facet per prim commits.
    assert float(settled.max() - per_prim) > 20.0
    assert float(per_prim - settled.min()) > 8.0


# --- the bar: provably a redistribution ----------------------------------------------------------


@pytest.mark.slow
def test_uniform_forcing_reproduces_the_per_prim_solve_exactly() -> None:
    """The conservation claim in its strongest form. Give every cell the same flux and the mesh
    field must return the *same number* the one-facet solve returns -- not close, the same. A
    field that added energy, double-counted an area or mis-weighted a cell fails here before any
    gradient is involved."""
    patch = TriangleMeshPatch.from_soup(box_mesh((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)), level=3)
    q = np.full(patch.n_cells, 500.0)
    settled = _settled(patch, q)
    expected = float(_expected_k(np.array([500.0]))[0])
    # "The same number" against a float32 query (CLAUDE.md #2) means bit-identical in float32:
    # one ulp at 300 K is ~30 uK, so a float64 tolerance below that would be testing the storage
    # and not the solve. Every cell must equal the scalar root exactly at the stored precision.
    assert np.all(np.float32(settled) == np.float32(expected))
    assert patch.area_weighted_mean(settled) == pytest.approx(expected, abs=1e-4)


@pytest.mark.slow
def test_the_area_weighted_mean_departs_from_the_per_prim_value_only_by_the_concavity() -> None:
    """Under varying forcing the mean of the field and the field at the mean are not equal, and
    they should not be: T rises as roughly q^(1/4), so by Jensen the mean temperature is *below*
    the temperature of the mean flux, by an amount set by the curvature over the span. Asserting
    equality here would be asserting a linear balance; asserting the sign and a bound is the
    honest statement that nothing was created."""
    soup = sphere_mesh((0.0, 0.0, 0.0), 0.25, 24, 48)
    patch = TriangleMeshPatch.from_soup(soup, level=1)
    cos = np.maximum(patch.cell_normal @ SUN, 0.0)
    q_solar = DNI_W_M2 * cos
    settled = _settled(patch, q_solar)

    areas = patch.cell_area_m2
    mean_q = float(np.sum(q_solar * areas) / np.sum(areas))
    per_prim = float(_expected_k(np.array([mean_q]))[0])
    mesh_mean = patch.area_weighted_mean(settled)
    span = float(settled.max() - settled.min())

    assert mesh_mean < per_prim  # concave T(q): the mean of T is below T of the mean
    assert (per_prim - mesh_mean) < 0.15 * span, (per_prim - mesh_mean, span)

    # Energy: at steady state every cell's own balance closes, so the area-weighted net flux over
    # the whole mesh is zero to round-off. This is the statement that the redistribution moved
    # temperature around and did not invent any.
    sigma = 5.670374419e-8
    net = (
        SKIN.solar_absorptivity * q_solar
        + SKIN.emissivity * LW_DOWN_W_M2
        - SKIN.emissivity * sigma * settled**4
        - H_W_M2_K * (settled - T_AIR_K)
    )
    absorbed_w = float(np.sum(SKIN.solar_absorptivity * q_solar * areas))
    assert absorbed_w > 50.0  # the scale the residual is judged against, ~105 W here
    assert abs(float(np.sum(net * areas))) < 1e-6 * absorbed_w


# --- cells inside a face -------------------------------------------------------------------------


@pytest.mark.slow
def test_sub_face_cells_resolve_a_source_that_one_cell_per_face_cannot() -> None:
    """Per-face normals mean a directional sun varies between faces and not within one, so the
    sphere above never exercises the sub-face grid. A heat source under half of a large triangle
    does: at level 1 the face is one temperature, at level 12 the halves stand 20 K apart and
    every cell is on its own root."""
    vertices = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    faces = np.array([[0, 1, 2]])

    coarse = TriangleMeshPatch.uniform(vertices, faces, level=1)
    fine = TriangleMeshPatch.uniform(vertices, faces, level=12)
    assert coarse.n_cells == 1 and fine.n_cells == 144

    def heater(patch: TriangleMeshPatch) -> np.ndarray:
        # 400 W/m² under the half of the triangle nearer vertex 0.
        return np.where(patch.cell_centres()[:, 0] < 0.66, 400.0, 0.0)

    flat = _settled(coarse, 0.0, heater(coarse))
    assert flat.size == 1

    graded = _settled(fine, 0.0, heater(fine))
    assert float(graded.max() - graded.min()) > 20.0
    assert float(np.max(np.abs(graded - _expected_k(np.zeros(fine.n_cells), heater(fine))))) < 1e-3


# --- the geometry --------------------------------------------------------------------------------


def test_a_cell_index_round_trips_from_its_own_centroid_at_every_level() -> None:
    """Generation and lookup are inverses. If they drift apart a cell's temperature lands on a
    different part of the triangle, and nothing raises -- so this is pinned at several levels."""
    soup = box_mesh((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    for level in (1, 2, 3, 5, 8, 16):
        patch = TriangleMeshPatch.from_soup(soup, level)
        assert patch.n_cells == soup.n_faces * level**2
        w = patch._cell_w
        assert np.array_equal(
            patch.cell_of(patch.cell_face, w[:, 0], w[:, 1]), np.arange(patch.n_cells)
        )


def test_a_query_lands_in_the_sub_triangle_it_names() -> None:
    """Stronger than the round trip: uniformly sampled points over each face must fall inside the
    sub-triangle the lookup returns, or the index is self-consistent and geometrically wrong."""
    from irsim.thermal.mesh_field import _subdivision

    rng = np.random.default_rng(1)
    for level in (1, 3, 8, 16):
        patch = TriangleMeshPatch.from_soup(box_mesh((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)), level)
        n = 5000
        face = rng.integers(0, patch.n_faces, n)
        r1, r2 = rng.random(n), rng.random(n)
        root = np.sqrt(r1)
        w0, w1 = root * (1.0 - r2), root * r2
        local = patch.cell_of(face, w0, w1) - patch.face_offsets[face]
        i_t, j_t, inv_t = _subdivision(level)
        i, j, inv = i_t[local], j_t[local], inv_t[local]
        dx, dy = w0 * level - i, w1 * level - j
        eps = 1e-9
        upright = (dx >= -eps) & (dy >= -eps) & (dx + dy <= 1.0 + eps)
        inverted = (dx <= 1.0 + eps) & (dy <= 1.0 + eps) & (dx + dy >= 1.0 - eps)
        assert np.all(np.where(inv == 1, inverted, upright)), level


def test_cell_areas_sum_to_the_mesh_area_and_split_evenly_inside_a_face() -> None:
    soup = sphere_mesh((1.0, -2.0, 0.5), 0.25, 16, 32)
    patch = TriangleMeshPatch.from_soup(soup, level=4)
    assert patch.area_m2 == pytest.approx(float(patch.cell_area_m2.sum()), rel=1e-12)
    # 0.8 % under the analytic sphere: the chord deficit, not an error in the cell split.
    assert patch.area_m2 == pytest.approx(4.0 * np.pi * 0.25**2, rel=0.01)
    for face in (0, 7, patch.n_faces - 1):
        cells = patch.cell_area_m2[patch.face_offsets[face] : patch.face_offsets[face + 1]]
        assert cells.size == 16
        assert float(cells.std()) < 1e-18
        assert float(cells.sum()) == pytest.approx(float(patch.face_area_m2[face]), rel=1e-12)


def test_an_edge_or_a_vertex_picks_a_neighbouring_cell_instead_of_raising() -> None:
    """A closest-point query lands exactly on an edge often enough that refusing would refuse a
    legitimate pixel, so the corners of the barycentric domain must resolve."""
    patch = TriangleMeshPatch.uniform(
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]), np.array([[0, 1, 2]]), 4
    )
    corners = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.5, 0.5], [0.25, 0.75]])
    cells = patch.cell_of(np.zeros(len(corners), dtype=int), corners[:, 0], corners[:, 1])
    assert np.all((cells >= 0) & (cells < patch.n_cells))
    assert len(set(cells.tolist())) >= 4  # they are not all collapsing to one cell


def test_per_face_resolution_follows_face_size() -> None:
    """Ptex style: a big face gets many cells and a small one gets few, so the cell count follows
    the thermal gradient rather than the tessellation."""
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [0.0, 4.0, 0.0],
            [10.0, 10.0, 0.0],
            [10.1, 10.0, 0.0],
            [10.0, 10.1, 0.0],
        ]
    )
    faces = np.array([[0, 1, 2], [3, 4, 5]])
    patch = TriangleMeshPatch.by_cell_size(vertices, faces, cell_m=0.25)
    big, small = patch.levels
    assert big > 8 and small == 1, (big, small)
    assert patch.n_cells == int(big) ** 2 + 1
    with pytest.raises(ValueError, match="cell_m must be positive"):
        TriangleMeshPatch.by_cell_size(vertices, faces, cell_m=0.0)


def test_warps_uv_pair_is_translated_in_one_place() -> None:
    """`WM.1` measured Warp's (u, v) as the weights of vertices 0 and 1. Reading them as the
    weights of 1 and 2 misses by 0.56 m on a 0.4 m box, silently, so the conversion has exactly
    one home and this pins it."""
    patch = TriangleMeshPatch.uniform(
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]), np.array([[0, 1, 2]]), 4
    )
    face = np.zeros(3, dtype=int)
    u, v = np.array([0.8, 0.1, 0.1]), np.array([0.1, 0.8, 0.1])
    assert np.array_equal(patch.cell_of_uv(face, u, v), patch.cell_of(face, u, v))
    # The three corners of the barycentric domain must land on three different cells, which is
    # what fails if the pair is read against the wrong vertices.
    assert len(set(patch.cell_of_uv(face, u, v).tolist())) == 3


def test_the_mesh_refuses_what_it_cannot_solve() -> None:
    good = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    with pytest.raises(ValueError, match="faces must be"):
        TriangleMeshPatch.uniform(good, np.array([[0, 1]]), 2)
    with pytest.raises(ValueError, match="indexes a vertex that does not exist"):
        TriangleMeshPatch.uniform(good, np.array([[0, 1, 9]]), 2)
    with pytest.raises(ValueError, match="level of at least 1"):
        TriangleMeshPatch.uniform(good, np.array([[0, 1, 2]]), 0)
    with pytest.raises(ValueError, match="degenerate"):
        TriangleMeshPatch.uniform(
            np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
            np.array([[0, 1, 2]]),
            1,
        )
    patch = TriangleMeshPatch.uniform(good, np.array([[0, 1, 2]]), 2)
    with pytest.raises(IndexError, match="face index outside"):
        patch.cell_of(np.array([3]), np.array([0.2]), np.array([0.2]))
    with pytest.raises(ValueError, match="cells, mesh has"):
        patch.face_mean(np.zeros(3))
    with pytest.raises(ValueError, match="properties describe"):
        TriangleMeshField(
            patch, _properties(99), lambda _t: FacetForcing(290.0, 5.0), 0.0, 290.0, 1.0
        )


def test_a_missing_hit_takes_the_fill_rather_than_a_cell() -> None:
    """A pixel that hit nothing must not silently take cell 0's temperature."""
    patch = TriangleMeshPatch.uniform(
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]), np.array([[0, 1, 2]]), 2
    )
    values = np.arange(patch.n_cells, dtype=np.float64)
    out = patch.sample(values, np.array([0, -1]), np.array([0.2, 0.2]), np.array([0.2, 0.2]))
    assert out[0] == values[patch.cell_of(np.array([0]), 0.2, 0.2)[0]]
    assert np.isnan(out[1])


def test_the_field_delegates_the_never_mutate_on_query_rule() -> None:
    """Composed on `ThermalField` rather than restating it, so a renderer asking many times per
    tick cannot change the answer by asking -- the rule `PlanarThermalField` carries."""
    patch = TriangleMeshPatch.from_soup(box_mesh((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)), level=2)
    field = TriangleMeshField(
        patch,
        _properties(patch.n_cells),
        lambda _t: FacetForcing(t_air_k=T_AIR_K, h_w_m2_k=H_W_M2_K, q_solar_w_m2=400.0),
        0.0,
        290.0,
        1.0,
        keep_ticks=None,
    )
    field.advance_to(60.0)
    before = field.state_hash()
    first = np.asarray(field.temperature_at(60.0))
    for _ in range(5):
        assert np.array_equal(np.asarray(field.temperature_at(60.0)), first)
    assert field.state_hash() == before
    with pytest.raises(ValueError, match="past the last tick"):
        field.temperature_at(600.0)
    assert field.face_temperature_k(60.0).shape == (patch.n_faces,)
    assert float(field.mean_temperature_k(60.0)) == pytest.approx(float(first.mean()), rel=1e-12)
