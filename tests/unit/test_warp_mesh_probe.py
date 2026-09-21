"""WM.1 — the closest-point oracle, and Warp's query where Warp is installed.

`scripts/probe_warp_mesh.py` answers whether Warp can recover an exact per-pixel (face, u, v)
from the position AOV, which is the measurement ADR 0087's curved-geometry limit rests on. Two
things in it need pinning rather than eyeballing:

* the **brute-force NumPy oracle**, because WM.3 is specified to be tested against it and an
  oracle that is quietly wrong would certify a broken bridge; and
* the **barycentric convention** Warp returns, because the natural reading
  `(1-u-v)*v0 + u*v1 + v*v2` is **not** the one it uses, and on a 0.4 m box the difference is
  half a metre of sampling error that looks like plausible noise.

The Warp half skips where Warp is absent, so the plain-CPython gate (`make ci`) still runs.

docs/decisions/0087-point-wise-surface-temperature.md; ADR 0014 (the 3.4 mm position budget);
roadmap WM.1.
"""

from __future__ import annotations

import importlib.util
import pathlib

import numpy as np
import pytest

from irsim.thermal.raycast import box_mesh

REPO = pathlib.Path(__file__).resolve().parents[2]
PROBE = REPO / "scripts" / "probe_warp_mesh.py"


def _probe_module():  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location("probe_warp_mesh", PROBE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


probe = _probe_module()


# --- the oracle ----------------------------------------------------------------------------------


def test_the_oracle_finds_the_analytic_closest_point_on_a_box() -> None:
    """Outside an axis-aligned box the closest surface point is `clip(p, lo, hi)` exactly, so
    this is a test against arithmetic rather than against another implementation. It covers the
    face, edge and vertex regions of Ericson's test, which are where a closest-point routine
    goes wrong: a version that only ever projected onto the triangle's plane passes on the face
    samples and fails on the corners."""
    centre, size = np.array([0.1, -0.2, 0.3]), np.array([0.4, 0.6, 0.5])
    soup = box_mesh(centre, size)
    lo, hi = centre - 0.5 * size, centre + 0.5 * size

    rng = np.random.default_rng(7)
    points = np.concatenate(
        [
            centre + np.array([[1.0, 0.0, 0.0]]) * 0.9,  # off one face
            centre + np.array([[1.0, 1.0, 0.0]]) * 0.9,  # off one edge
            centre + np.array([[1.0, 1.0, 1.0]]) * 0.9,  # off one corner
            centre + rng.normal(scale=0.8, size=(200, 3)),
        ]
    )
    outside = np.any((points < lo) | (points > hi), axis=1)
    points = points[outside]
    assert len(points) > 100

    closest, face = probe.closest_point_numpy(soup, points)
    expected = np.clip(points, lo, hi)
    assert float(np.max(np.linalg.norm(closest - expected, axis=-1))) < 1e-12
    # The reported face must be one that actually contains the reported point.
    v = soup.vertices[soup.faces[face]]
    for point, triangle in zip(closest, v, strict=True):
        area = np.linalg.norm(np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0]))
        sub = sum(
            np.linalg.norm(np.cross(triangle[i] - point, triangle[j] - point))
            for i, j in ((0, 1), (1, 2), (2, 0))
        )
        assert abs(sub - area) < 1e-9 * max(area, 1e-9)


def test_the_oracle_lands_on_the_sphere_mesh_within_its_own_chord_error() -> None:
    """A UV sphere is a chord approximation, short of the true sphere by the sagitta. The oracle
    must land on the *mesh*, so its distance from the analytic sphere is bounded by that sagitta
    and by nothing looser -- which is what lets the probe quote a sphere residual at all."""
    radius, n_theta, n_phi = 0.25, 32, 64
    soup = probe.uv_sphere(radius, n_theta, n_phi)
    sagitta = probe.sagitta_m(radius, n_theta, n_phi)
    rng = np.random.default_rng(11)
    directions = rng.normal(size=(300, 3))
    directions /= np.linalg.norm(directions, axis=-1, keepdims=True)
    points = directions * (radius + 0.05)  # a shell outside the sphere

    closest, _ = probe.closest_point_numpy(soup, points)
    deficit = radius - np.linalg.norm(closest, axis=-1)
    assert float(deficit.min()) >= -1e-12  # never outside the true sphere
    assert float(deficit.max()) <= sagitta * 1.001, (float(deficit.max()), sagitta)


def test_surface_samples_land_on_the_mesh() -> None:
    soup = box_mesh((0.0, 0.0, 0.0), (0.4, 0.4, 0.4))
    points = probe.surface_samples(soup, 500, seed=3)
    closest, _ = probe.closest_point_numpy(soup, points)
    assert float(np.max(np.linalg.norm(closest - points, axis=-1))) < 1e-12


# --- Warp, where it is installed -----------------------------------------------------------------


@pytest.mark.slow
def test_warp_returns_the_convention_the_probe_records_and_a_micron_round_trip() -> None:
    """The measurement WM.1 exists to make, at small scale. Two claims:

    1. `mesh_eval_position(face, u, v)` reproduces a point already on the surface to far inside
       ADR 0014's 3.4 mm position budget -- so the parameterisation loses nothing.
    2. Warp's (u, v) are the weights of **v0 and v1**, with `1 - u - v` on v2. The two readings
       a person would write down first are wrong by centimetres to metres, and this test fails if
       a future Warp changes the convention rather than letting the bridge sample the wrong cell.
    """
    wp, kernel = probe.build_kernel()
    if wp is None:
        pytest.skip(f"warp unavailable: {kernel}")

    soup = box_mesh((0.0, 0.0, 0.0), (0.4, 0.4, 0.4))
    points = probe.surface_samples(soup, 2000, seed=5)
    hit, face, bary, evaluated, _ = probe.run_query(wp, kernel, soup, points, "cpu")
    assert hit.all()

    residual = np.linalg.norm(evaluated - points, axis=-1)
    assert float(residual.max()) < probe.POSITION_BUDGET_M
    assert float(residual.max()) < 1e-5  # measured 9.6e-8 m: float32, not the budget

    conventions = probe.barycentric_convention(soup, face, bary, evaluated)
    best = min(conventions, key=lambda k: conventions[k])
    assert best == "u*v0 + v*v1 + (1-u-v)*v2", conventions
    assert conventions[best] < 1e-6
    others = [v for k, v in conventions.items() if k != best]
    assert min(others) > 0.1, conventions  # the wrong readings miss by 0.5 m on a 0.4 m box

    # And Warp agrees with the oracle about *where on the surface* the point is, always -- even
    # on the queries where the two name different triangles, which a shared edge makes legitimate.
    oracle_point, _ = probe.closest_point_numpy(soup, points)
    assert float(np.max(np.linalg.norm(evaluated - oracle_point, axis=-1))) < 1e-6
