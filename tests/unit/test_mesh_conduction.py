"""Lateral conduction between a mesh's cells (WM.6).

Two things can go wrong here and neither shows up as an error. A conductance of the wrong
*magnitude* renders a gradient that is simply too strong or too weak, and nothing complains; a
**negative** conductance breaks the discrete maximum principle and puts a cell outside the range
its neighbours and its forcing span, which arrives as a bright speck indistinguishable from a bad
pixel. The tests below measure the first against the fin equation's own answer and pin the second
shut.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from irsim.thermal.conduction import ConductionOperator
from irsim.thermal.mesh_conduction import (
    circumcentric_conductance,
    edge_spans,
    mesh_lateral_operator,
    within_face_links,
)
from irsim.thermal.mesh_field import TriangleMeshPatch
from irsim.thermal.raycast import cylinder_mesh

SIDE = 1.0
EQUILATERAL = np.array([[0.0, 0.0, 0.0], [SIDE, 0.0, 0.0], [0.5 * SIDE, SIDE * 3**0.5 / 2, 0.0]])
RIGHT = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
OBTUSE = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [-0.6, 0.35, 0.0]])
SKEWED = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.9, 0.45, 0.0]])

#: Carbon fibre, from configs/materials/carbon_fibre.yaml, and a still-air h at 300 K.
K_W_MK, DELTA_M, H_W_M2K = 0.80, 0.0015, 15.51


def one_face(vertices: np.ndarray, level: int) -> TriangleMeshPatch:
    return TriangleMeshPatch(
        vertices_m=vertices, faces=np.array([[0, 1, 2]]), levels=np.array([level])
    )


# --- the weight ---------------------------------------------------------------------------------


@pytest.mark.parametrize("level", [2, 3, 4, 7])
def test_an_equilateral_face_carries_the_cotan_weight_exactly(level: int) -> None:
    """The one geometry where the shipped scheme and the consistent one must agree.

    On an equilateral triangle the centroid *is* the circumcentre, so the two-point flux and the
    circumcentric weight are the same number: ``k δ tan 60° = √3 k δ``. If this ever drifted, the
    shipped weight would be an approximation of nothing in particular.
    """
    patch = one_face(EQUILATERAL, level)
    operator = mesh_lateral_operator(patch, K_W_MK, DELTA_M)
    assert operator is not None
    expected = K_W_MK * DELTA_M * circumcentric_conductance(patch, 0, 0)
    assert np.allclose(operator.conductance_w_k.data, expected, rtol=1e-12)
    assert np.isclose(expected, K_W_MK * DELTA_M * 3**0.5, rtol=1e-12)


def test_refining_a_face_adds_cells_without_changing_how_fast_heat_crosses_it() -> None:
    """The barycentric cut is affine, so ``w`` and ``d`` shrink together and ``w/d`` is scale-free.

    A discretisation whose conductance moved with the level would make the mesh resolution a
    physical parameter: refining a face for a sharper picture would change its temperature.
    """
    weights = []
    for level in (2, 4, 8):
        patch = one_face(SKEWED, level)
        operator = mesh_lateral_operator(patch, K_W_MK, DELTA_M)
        assert operator is not None
        weights.append(sorted(set(np.round(operator.conductance_w_k.data, 12))))
        assert len(within_face_links(patch, 0)) == 3 * level * (level - 1) // 2
    assert weights[0] == weights[1] == weights[2]


def test_the_link_and_span_bookkeeping_covers_each_sub_edge_once() -> None:
    """Every interior sub-edge touches exactly one inverted cell, and an inverted cell has three
    upright neighbours -- so counting inverted cells counts the edges, once each."""
    for level in (1, 2, 3, 5):
        patch = one_face(EQUILATERAL, level)
        links = within_face_links(patch, 0)
        assert len({tuple(sorted(link[:2])) for link in links}) == len(links)
        assert len(links) == 3 * level * (level - 1) // 2
        for side in range(3):
            spans = edge_spans(patch, 0, side)
            assert len(spans) == level
            assert np.isclose(sum(hi - lo for _, lo, hi in spans), 1.0)
            assert len({cell for cell, _, _ in spans}) == level


# --- consistency, and what it costs ---------------------------------------------------------------


def linear_field_residual(patch: TriangleMeshPatch, gradient: np.ndarray) -> float:
    """Largest net conduction power into an interior cell under a linear field, normalised.

    A linear field has ``∇²T = 0``, so a consistent scheme moves no net heat into any interior
    cell. What comes back is therefore the scheme's own error, not the field's.
    """
    operator = mesh_lateral_operator(patch, K_W_MK, DELTA_M)
    assert operator is not None
    temperatures = patch.cell_centres() @ np.asarray(gradient, dtype=np.float64)
    conductance = operator.conductance_w_k
    interior = np.asarray((conductance != 0).sum(axis=1)).ravel() == 3
    scale = float(np.abs(conductance.data).max()) * float(np.ptp(temperatures))
    return float(np.abs(operator.power_in_w(temperatures)[interior]).max() / scale)


def test_a_linear_field_is_exact_on_an_equilateral_face() -> None:
    for gradient in ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.7, -0.3, 0.0]):
        assert linear_field_residual(one_face(EQUILATERAL, 12), gradient) < 1e-14


@pytest.mark.parametrize(
    ("name", "vertices", "floor"),
    [("right", RIGHT, 0.01), ("obtuse", OBTUSE, 0.01), ("skewed", SKEWED, 0.02)],
)
def test_a_linear_field_is_not_exact_on_a_skewed_face_and_the_error_is_recorded(
    name: str, vertices: np.ndarray, floor: float
) -> None:
    """The measured price of the choice, asserted so it cannot grow unnoticed.

    A three-neighbour centroid stencil on a triangular lattice can be exact for linear fields
    **or** isotropic, not both, unless the triangles are equilateral: exactness forces all three
    weights equal, and equal weights on a skewed face give an anisotropic operator. The scheme
    keeps the isotropy and pays 1-5 % of scale in linear exactness; the alternative pays with
    negative weights (below).
    """
    residual = max(
        linear_field_residual(one_face(vertices, 12), g) for g in ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
    )
    assert floor <= residual < 0.06, f"{name}: {residual:.4f}"


# --- why not the cotan weight ---------------------------------------------------------------------


def test_the_circumcentric_weight_goes_negative_on_an_obtuse_face() -> None:
    """The defect `WM.6` exists for. Inside one face the two sub-triangles across a sub-edge are
    congruent, so the two angles opposite it are both the parent's own and the consistent weight
    is ``k δ tan Θ`` -- negative for every obtuse face, and singular for a right-angled one, where
    the circumcentre lies on the shared edge."""
    obtuse = one_face(OBTUSE, 3)
    weights = [circumcentric_conductance(obtuse, 0, side) for side in range(3)]
    assert min(weights) < 0.0, weights
    assert all(
        w > 0.0
        for w in (circumcentric_conductance(one_face(EQUILATERAL, 3), 0, s) for s in range(3))
    )


def test_an_operator_built_from_the_negative_weight_is_refused_outright() -> None:
    """`ConductionOperator` will not hold a negative conductance, so the cotan assembly cannot
    even be constructed on an obtuse face -- the failure is at build, not a speck in a frame."""
    patch = one_face(OBTUSE, 3)
    n = patch.n_cells
    rows, cols, vals = [], [], []
    for a, b, side in within_face_links(patch, 0):
        rows.append(a)
        cols.append(b)
        vals.append(K_W_MK * DELTA_M * circumcentric_conductance(patch, 0, side))
    upper = sp.coo_matrix((np.asarray(vals), (rows, cols)), shape=(n, n))
    with pytest.raises(ValueError, match="cannot be negative"):
        ConductionOperator(upper + upper.T, patch.cell_area_m2)


def test_no_cell_leaves_the_range_its_neighbours_and_the_forcing_span() -> None:
    """The discrete maximum principle, on the deliberately obtuse mesh, through the real step.

    One cell is started 50 K hot and everything else sits at ambient with no forcing. Backward
    Euler on a non-negative Laplacian is a convex combination, so every cell must stay inside
    [ambient, ambient + 50] at every tick and the hot cell must only cool. A negative weight
    overshoots on the first step and the overshoot is the bright speck.
    """
    patch = one_face(OBTUSE, 6)
    operator = mesh_lateral_operator(patch, K_W_MK, DELTA_M)
    assert operator is not None
    solve = operator.factorise(np.full(patch.n_cells, 100.0), 1.0)
    temperatures = np.full(patch.n_cells, 300.0)
    temperatures[patch.n_cells // 2] = 350.0
    peak = float(temperatures.max())
    for _ in range(8000):
        temperatures = solve.solve(temperatures)
        assert temperatures.min() >= 300.0 - 1e-9
        assert temperatures.max() <= peak + 1e-9
        peak = float(temperatures.max())
    # and it really spread rather than freezing: the cells are equal-area, so conduction alone
    # takes the whole face to the mean of what it started with.
    assert np.allclose(temperatures, 300.0 + 50.0 / patch.n_cells, atol=0.01)


def test_conduction_moves_heat_without_creating_it() -> None:
    """Symmetry is what makes the exchange conservative for cells of unequal area: the total
    ``Σ C A dT`` from conduction alone is zero, and the operator checks the symmetry at build."""
    patch = one_face(SKEWED, 5)
    operator = mesh_lateral_operator(patch, K_W_MK, DELTA_M)
    assert operator is not None
    rng = np.random.default_rng(20260922)
    temperatures = 300.0 + rng.normal(0.0, 5.0, patch.n_cells)
    assert abs(float(operator.power_in_w(temperatures).sum())) < 1e-9


# --- across the mesh's own edges --------------------------------------------------------------


def two_faces(level_a: int, level_b: int) -> TriangleMeshPatch:
    """A rhombus: two equilateral faces sharing one edge, wound consistently."""
    height = SIDE * 3**0.5 / 2
    vertices = np.array(
        [[0.0, 0.0, 0.0], [SIDE, 0.0, 0.0], [0.5 * SIDE, height, 0.0], [0.5 * SIDE, -height, 0.0]]
    )
    return TriangleMeshPatch(
        vertices_m=vertices,
        faces=np.array([[0, 1, 2], [0, 3, 1]]),
        levels=np.array([level_a, level_b]),
    )


def cross_edge_total(patch: TriangleMeshPatch) -> float:
    """Total conductance between the two faces' cells."""
    operator = mesh_lateral_operator(patch, K_W_MK, DELTA_M)
    assert operator is not None
    first = int(patch.face_offsets[1])
    return float(operator.conductance_w_k[:first, first:].sum())


def test_two_faces_conduct_across_the_edge_they_share() -> None:
    patch = two_faces(3, 3)
    operator = mesh_lateral_operator(patch, K_W_MK, DELTA_M)
    assert operator is not None
    # One connected component: heat can get from any cell to any other.
    n_components, _ = sp.csgraph.connected_components(operator.conductance_w_k, directed=False)
    assert n_components == 1
    assert cross_edge_total(patch) > 0.0


@pytest.mark.parametrize(("level_a", "level_b"), [(2, 2), (2, 3), (3, 2), (4, 7), (1, 5)])
def test_faces_of_different_levels_meet_by_the_length_they_actually_share(
    level_a: int, level_b: int
) -> None:
    """Non-conforming levels are matched by overlap, as `TC.3`'s contactors match two patches
    whose grids do not line up.

    What must hold is that the seam is covered **once**: the overlaps along it sum to its whole
    length and to no more, whatever either side was cut into. A matching that dropped a sliver
    would insulate part of the seam and one that double-counted would conduct twice through it,
    and neither shows up as anything but a slightly wrong picture.

    The total conductance itself is *not* an invariant and must not be asserted as one: the
    centroid of a finer cell sits closer to the seam, so the link across it is stiffer, exactly as
    a finite difference stiffens as its step shrinks. The closed form below carries that -- with
    equal-area faces the two depths are ``2A/(3 k E)`` and the total is
    ``(3/2) k δ E²/A · k_a k_b/(k_a + k_b)``.
    """
    patch = two_faces(level_a, level_b)
    edge = SIDE  # the shared edge (v0, v1) of the rhombus
    area = SIDE * SIDE * 3**0.5 / 4
    expected = 1.5 * K_W_MK * DELTA_M * edge**2 / area * (level_a * level_b) / (level_a + level_b)
    assert np.isclose(cross_edge_total(patch), expected, rtol=1e-12)


def test_an_open_boundary_conducts_to_nothing() -> None:
    """A single face has three sides and no neighbour on any of them; its cells conduct only to
    each other, and no link runs off the edge."""
    patch = one_face(EQUILATERAL, 4)
    operator = mesh_lateral_operator(patch, K_W_MK, DELTA_M)
    assert operator is not None
    assert operator.conductance_w_k.nnz == 2 * len(within_face_links(patch, 0))


def test_no_conductivity_means_no_operator_at_all() -> None:
    """`None`, not a matrix of zeros: the field then takes the operator-free path bit for bit."""
    patch = one_face(EQUILATERAL, 4)
    assert mesh_lateral_operator(patch, 0.0, DELTA_M) is None
    assert mesh_lateral_operator(patch, K_W_MK, 0.0) is None
    with pytest.raises(ValueError, match="cannot be negative"):
        mesh_lateral_operator(patch, -1.0, DELTA_M)


# --- against the fin equation -------------------------------------------------------------------


def tube_attenuation(n_phi: int, n_z: int, radius: float = 0.015, length: float = 0.40) -> float:
    """How much of a sunlit tube's crown-to-underside span survives lateral conduction.

    The steady fin problem ``L T + h A T = A q`` solved on the mesh, against the same forcing with
    no conduction at all. The mid-span only, so the open ends do not enter.
    """
    soup = cylinder_mesh([0, 0, 0], radius, length, [0, 1, 0], n_phi=n_phi, n_z=n_z, capped=False)
    patch = TriangleMeshPatch.from_soup(soup, level=1)
    operator = mesh_lateral_operator(patch, K_W_MK, DELTA_M)
    assert operator is not None
    normals = patch.cell_normal
    theta = np.arctan2(normals[:, 2], normals[:, 0])
    flux = 800.0 * np.maximum(np.cos(theta - 0.5 * np.pi), 0.0)  # a sunlit crown
    area = patch.cell_area_m2
    conducted = spla.spsolve((operator.laplacian + sp.diags(H_W_M2K * area)).tocsc(), area * flux)
    mid = np.abs(patch.cell_centres()[:, 1]) < 0.05
    return float(np.ptp(conducted[mid]) / np.ptp((flux / H_W_M2K)[mid]))


def test_a_tube_attenuates_its_gradient_as_the_fin_equation_says() -> None:
    """The headline. A sheet of ``k δ`` losing heat at ``h`` cannot hold a feature smaller than
    ``L = √(kδ/h)``, and a gradient of wavelength ``λ`` survives at ``1/(1 + (2πL/λ)²)`` -- the
    steady fin equation, not a rule of thumb. On carbon fibre round a 30 mm tube that is 0.744,
    and the operator reproduces it to **1 %** when the tessellation is cut near-square.
    """
    radius = 0.015
    circumference = 2.0 * np.pi * radius
    smoothing = (K_W_MK * DELTA_M / H_W_M2K) ** 0.5
    analytic = 1.0 / (1.0 + (2.0 * np.pi * smoothing / circumference) ** 2)
    assert np.isclose(analytic, 0.7441, atol=1e-4)
    # n_phi 48 gives a 1.96 mm arc; n_z 143 over 0.40 m gives a 2.80 mm ring: 1.43 across.
    assert np.isclose(tube_attenuation(48, 143), analytic, rtol=0.01)


def test_a_long_thin_tessellation_under_conducts_and_the_rule_says_so() -> None:
    """What the aspect ratio costs, measured -- the authoring rule `ADR 0112` states.

    The scheme is exact when a face's quad is about 1.4 times longer along the tube than around
    it, and degrades either side: long thin quads saturate at three quarters of the conductivity
    they should have, which would render a tube's gradient a quarter too strong.
    """
    circumference = 2.0 * np.pi * 0.015
    smoothing = (K_W_MK * DELTA_M / H_W_M2K) ** 0.5

    def effective(ratio: float) -> float:
        """``kδ_eff / kδ`` backed out of the attenuation."""
        length = 0.40
        n_phi = 48
        arc = circumference / n_phi
        n_z = max(4, int(round(length / (ratio * arc))))
        got = tube_attenuation(n_phi, n_z, length=length)
        recovered = circumference * (1.0 / got - 1.0) ** 0.5 / (2.0 * np.pi)
        return float((recovered / smoothing) ** 2)

    assert np.isclose(effective(1.4), 1.0, atol=0.01)
    assert 0.74 < effective(12.7) < 0.78
    assert 1.18 < effective(0.8) < 1.24
