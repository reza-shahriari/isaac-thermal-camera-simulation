"""AI.3 -- what an imported asset costs, and how much of that cost buys anything.

Two separate claims are tested here and they fail for different reasons.

*Affordability* is a measurement someone else made: `GT.7` timed 102,400 cells over a 48-hour
spin-up at 11.5 s, against Fraunhofer's 1,313,410-triangle reference. If the budget drifts away
from that, the number stops meaning anything.

*Usefulness* is physics. A cell holds one temperature, and two cells closer together than
``sqrt(alpha dt)`` cannot hold different ones, because conduction erases the difference inside the
step. A test that only counted triangles would pass on a mesh with a million faces on a postage
stamp, which is the actual condition of the only imported asset this project has.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from irsim.io.asset_budget import (
    FASTEST_DIFFUSIVITY_M2_S,
    MIN_CELL_EDGE_M,
    REFERENCE_TICK_S,
    SLOWEST_DIFFUSIVITY_M2_S,
    BudgetReport,
    GeometryBudget,
    PrimBudget,
    conduction_length_m,
    measure_geometry,
)
from irsim.io.assets import AssetMesh, AssetMeshes
from irsim.materials.library import MaterialLibrary

SQUARE_V = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
SQUARE_F = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.intp)


def _mesh(name: str, faces: int, area_m2: float, material: str | None = "paint") -> AssetMesh:
    """A prim with a declared face count and area. The arrays are a placeholder: the budget reads
    the count and the area, which is exactly what the archive's manifest carries."""
    v = np.zeros((3, 3), dtype=np.float64)
    v[1, 0] = 1.0
    v[2, 1] = 1.0
    f = np.zeros((faces, 3), dtype=np.intp)
    f[:, 1] = 1
    f[:, 2] = 2
    return AssetMesh(
        name=name,
        material_name=material,
        vertices_m=v,
        faces=f,
        area_m2=area_m2,
        area_before_m2=area_m2,
    )


def _asset(*meshes: AssetMesh, name: str = "fixture") -> AssetMeshes:
    return AssetMeshes(name=name, meshes={m.name: m for m in meshes})


# --- the resolution floor is derived, not chosen -------------------------------------------------


def test_the_conduction_length_is_the_square_root_law():
    assert conduction_length_m(4.0e-6, 100.0) == pytest.approx(0.02)
    assert conduction_length_m(1.0, 1.0) == pytest.approx(1.0)
    # Four times the tick doubles the distance: it is a diffusion, not a velocity.
    assert conduction_length_m(1e-5, 400.0) == pytest.approx(2.0 * conduction_length_m(1e-5, 100.0))


def test_the_conduction_length_refuses_a_nonsense_argument():
    for bad in ((0.0, 60.0), (-1e-6, 60.0), (1e-6, 0.0), (1e-6, -60.0)):
        with pytest.raises(ValueError):
            conduction_length_m(*bad)


def test_the_floor_is_a_millimetre_and_the_ceiling_is_seven_centimetres():
    """Both ends matter. The floor is the point past which no material can use the geometry; the
    ceiling says an aluminium panel is already one temperature over most of its own width."""
    floor = MIN_CELL_EDGE_M
    assert floor == pytest.approx(conduction_length_m(SLOWEST_DIFFUSIVITY_M2_S, REFERENCE_TICK_S))
    assert 0.001 < floor < 0.0013
    fastest = conduction_length_m(FASTEST_DIFFUSIVITY_M2_S, REFERENCE_TICK_S)
    assert 0.06 < fastest < 0.08


def test_the_two_diffusivities_bracket_the_material_library():
    """The floor is only a floor if nothing in the library is slower. This walks the library, so
    adding an aerogel without revisiting the constant fails here rather than silently lowering
    the resolution everything is measured against."""
    library = MaterialLibrary.load()
    alphas = []
    for name in library.names:
        thermal = library.get(name).spec.thermal
        alphas.append(
            thermal.conductivity_w_mk / (thermal.density_kg_m3 * thermal.specific_heat_j_kgk)
        )
    assert min(alphas) == pytest.approx(SLOWEST_DIFFUSIVITY_M2_S, rel=0.01)
    assert max(alphas) == pytest.approx(FASTEST_DIFFUSIVITY_M2_S, rel=0.01)


# --- the arithmetic ------------------------------------------------------------------------------


def test_a_cell_edge_is_the_side_of_the_square_two_triangles_tile():
    assert PrimBudget("a", None, faces=2, area_m2=1.0).cell_edge_m == pytest.approx(1.0)
    assert PrimBudget("a", None, faces=8, area_m2=1.0).cell_edge_m == pytest.approx(0.5)
    # Four times the faces halves the edge, which is the only scaling a 2-D tiling can have.
    coarse = PrimBudget("a", None, faces=100, area_m2=3.0).cell_edge_m
    fine = PrimBudget("a", None, faces=400, area_m2=3.0).cell_edge_m
    assert fine == pytest.approx(0.5 * coarse)


def test_a_degenerate_prim_is_not_infinitely_fine():
    assert PrimBudget("a", None, faces=0, area_m2=1.0).cell_edge_m == 0.0
    assert PrimBudget("a", None, faces=10, area_m2=0.0).cell_edge_m == 0.0


def test_the_affordable_face_count_tiles_the_area_at_the_floor():
    budget = GeometryBudget()
    one_m2 = budget.affordable_faces(1.0)
    assert one_m2 == pytest.approx(2.0 / budget.min_cell_edge_m**2, rel=1e-6)
    # A surface of that many faces sits exactly at the floor, not below it.
    edge = PrimBudget("a", None, faces=one_m2, area_m2=1.0).cell_edge_m
    assert edge == pytest.approx(budget.min_cell_edge_m, rel=1e-4)
    assert budget.affordable_faces(0.0) == 0


def test_a_budget_refuses_nonsense_limits():
    for kwargs in ({"total_faces": 0}, {"faces_per_prim": -1}, {"min_cell_edge_m": 0.0}):
        with pytest.raises(ValueError):
            GeometryBudget(**kwargs)  # type: ignore[arg-type]


# --- the gate ------------------------------------------------------------------------------------


def test_a_modest_asset_passes_and_wastes_nothing():
    budget = GeometryBudget()
    # A tenth of a square metre carried at exactly the floor: every face is doing work.
    asset = _asset(_mesh("panel", budget.affordable_faces(0.1), 0.1))
    report = measure_geometry(asset, budget)
    assert report.passed
    assert report.wasted_faces == 0
    assert not report.too_fine
    assert "within budget" in report.render()


def test_the_whole_budget_buys_less_than_a_square_metre_at_the_floor():
    """Worth knowing before anyone reads "finer than the floor" as an instruction to refine. The
    affordable face count resolves about 0.8 m2 at a millimetre, so a vehicle or a building has
    to be solved *coarser* than the floor and the waste report is about geometry that is finer
    than even that -- not about geometry anyone should add."""
    budget = GeometryBudget()
    area_at_floor = budget.total_faces * budget.min_cell_edge_m**2 / 2.0
    assert 0.6 < area_at_floor < 1.0
    assert budget.affordable_faces(area_at_floor) == pytest.approx(budget.total_faces, rel=1e-3)


def test_too_many_faces_in_total_is_refused_even_when_every_prim_is_small():
    """The failure an import actually has: no single part is outrageous and the sum is."""
    budget = GeometryBudget(total_faces=1000, faces_per_prim=1000)
    asset = _asset(*[_mesh(f"p{i}", 200, 1.0) for i in range(8)])
    report = measure_geometry(asset, budget)
    assert not report.passed
    assert report.over_total
    assert not report.over_prim
    assert "REFUSED" in report.render()


def test_one_heavy_prim_is_refused_and_named():
    budget = GeometryBudget(total_faces=10_000_000, faces_per_prim=1000)
    asset = _asset(_mesh("light", 100, 1.0), _mesh("heavy", 5000, 1.0))
    report = measure_geometry(asset, budget)
    assert not report.passed
    assert not report.over_total
    assert [p.name for p in report.over_prim] == ["heavy"]
    rendered = report.render()
    assert "heavy" in rendered and "REFUSED" in rendered


def test_geometry_finer_than_the_physics_is_reported_but_not_refused():
    """A million faces on a postage stamp is a waste, not an error -- the lever that fixes it is
    the caller's. But nothing should be able to hide it."""
    budget = GeometryBudget()
    stamp = _mesh("stamp", 100_000, 2.7e-4)  # the Phantom 4's worst prim, to scale
    report = measure_geometry(_asset(stamp), budget)
    assert report.passed  # affordable
    assert [p.name for p in report.too_fine] == ["stamp"]
    assert report.wasted_faces > 0.99 * stamp.n_faces
    assert "finer than the floor" in report.render()


def test_the_report_is_ordered_by_what_costs_most():
    report = measure_geometry(_asset(_mesh("small", 10, 1.0), _mesh("big", 5000, 1.0)))
    lines = [ln for ln in report.render().splitlines() if "small" in ln or "big" in ln]
    assert lines and "big" in lines[0]


def test_an_empty_asset_measures_to_zero_rather_than_dividing_by_it():
    report = measure_geometry(_asset())
    assert report.total_faces == 0
    assert report.passed
    assert isinstance(report.render(), str)


# --- the real asset ------------------------------------------------------------------------------


def _phantom4():
    from irsim.io.assets import ASSETS_DIR

    archive = ASSETS_DIR / "phantom4" / "phantom4.meshes.npz"
    if not archive.exists():
        pytest.skip("no prepared phantom4 archive (data/assets is generated, not in git)")
    from irsim.io.assets import load_asset_meshes

    return load_asset_meshes(archive)


def test_the_only_imported_asset_this_project_has_is_over_budget():
    """The point of the step. Before this, nothing counted, so an asset too heavy to solve was
    discovered by waiting for the solve. The Phantom 4 is over, and by a knowable amount."""
    report = measure_geometry(_phantom4())
    assert report.total_faces > report.budget.total_faces
    assert not report.passed
    assert report.over_total


def test_three_quarters_of_the_phantom_4_s_faces_cannot_carry_a_gradient():
    """Its mesh came out of a renderer. One prim holds 100,926 faces over 2.7 cm2 -- a 73 um cell,
    fifteen times below the distance heat crosses in one tick in the slowest material here."""
    report = measure_geometry(_phantom4())
    share = report.wasted_faces / report.total_faces
    assert 0.6 < share < 0.95
    finest = min(p.cell_edge_m for p in report.prims if p.cell_edge_m > 0.0)
    assert finest < 0.2e-3
    assert len(report.too_fine) > 0.5 * len(report.prims)


def test_the_measured_area_and_prim_count_are_what_adr_0128_recorded():
    """A guard on the archive itself: 41 prims and a third of a square metre of aircraft. If the
    prep tool changes what it emits, this says so before a budget number is read as a regression."""
    report = measure_geometry(_phantom4())
    assert len(report.prims) == 41
    assert 0.25 < report.total_area_m2 < 0.35


def test_a_report_can_be_built_without_measuring_anything():
    """`BudgetReport` is a value, so a caller can construct one for a hypothetical budget."""
    report = BudgetReport(asset="hypothetical", budget=GeometryBudget(), prims=())
    assert report.passed and report.total_faces == 0
    assert math.isfinite(report.total_area_m2)


def _prep_module():
    import importlib.util
    import pathlib

    path = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "prep_asset.py"
    spec = importlib.util.spec_from_file_location("prep_asset_budget", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_prep_tool_refuses_an_over_budget_asset_and_can_be_told_not_to(capsys):
    """The gate, end to end, without Blender: `--skip-convert` audits what is already on disk.
    A budget nothing enforces is a comment."""
    _phantom4()  # skips when the generated archive is absent
    prep = _prep_module()

    assert prep.run_driver(["--asset", "phantom4", "--skip-convert"]) == 1
    refused = capsys.readouterr().out
    assert "geometry budget" in refused and "REFUSED" in refused

    assert prep.run_driver(["--asset", "phantom4", "--skip-convert", "--allow-over-budget"]) == 0
    allowed = capsys.readouterr().out
    # Still reported -- the override silences the failure, not the finding.
    assert "REFUSED" in allowed


def test_a_raised_budget_accepts_the_same_asset(capsys):
    _phantom4()
    prep = _prep_module()
    code = prep.run_driver(
        [
            "--asset",
            "phantom4",
            "--skip-convert",
            "--face-budget",
            "5000000",
            "--faces-per-prim",
            "500000",
        ]
    )
    assert code == 0
    assert "within budget" in capsys.readouterr().out
