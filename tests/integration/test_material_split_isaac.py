"""The whole `AI.6` pass on the committed fixture: Blender splits it, Kit reads the result.

`AI.4` left a gap it could only report: the audit reads ``materialBind`` subsets, the renderer
binds one material per prim (ADR 0014), and so a three-material wall renders as one substance
while the coverage figure says 100 %. This test is the closure of that gap end to end --
``scripts/prep_asset.py`` regroups the geometry in Blender, on the CPU, booting no Kit, and the
production stage walk then reads the output inside Kit and finds nothing left to report.

Two processes are involved for the reason ADR 0128 gives: Blender's interpreter carries a
complete OpenUSD and none of this project's dependencies, and Kit carries the renderer and no
``bpy``. The plan that passes between them is a file, and it is the same file a prep run writes.

Skipped without Blender; the Isaac half is skipped without Isaac Sim like every test here.
"""

from __future__ import annotations

import json
import pathlib
import shutil
import subprocess
from typing import Any

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
FIXTURE = REPO / "tests" / "unit" / "data" / "subset_building.usda"
PREP = REPO / "scripts" / "prep_asset.py"

#: What the fixture says, measured: six 1 m x 1 m faces of facade in three materials, a
#: two-material roof, a single-material plinth, and an invisible scaffold that is not imported.
EXPECTED_PIECES = {
    "Facade_Glass_Clear",
    "Facade_Concrete_Precast",
    "Facade_Metal_Cladding",
    "Roof_Asphalt_Felt",
    "Roof_Alumin_Flashing",
    "Plinth_Concrete_Base",
}


def _prep_module() -> Any:
    import importlib.util

    spec = importlib.util.spec_from_file_location("prep_asset_split_it", PREP)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def blender() -> str:
    found = shutil.which("blender")
    if found is None:
        pytest.skip("Blender not on PATH; it is this project's USD toolchain (ADR 0128)")
    return found


@pytest.fixture(scope="module")
def split(blender: str, tmp_path_factory: pytest.TempPathFactory) -> Any:
    """Measure, plan, split -- the three steps a `--emit-material-split` run performs."""
    prep = _prep_module()
    out = tmp_path_factory.mktemp("material_split")
    slots = out / "subset_building.material_slots.json"
    measure = prep.blender_material_slots_command(blender, FIXTURE, 1.0, slots)
    assert subprocess.run(measure, check=False, capture_output=True).returncode == 0

    report, plan = prep.plan_from_slots(slots)
    plan_path = out / "plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")

    usd = out / "subset_building_materials.usdc"
    cmd = prep.blender_material_split_command(blender, FIXTURE, 1.0, plan_path, usd)
    assert subprocess.run(cmd, check=False, capture_output=True).returncode == 0
    assert usd.exists()
    return report, usd


def test_the_plan_measures_what_leaving_the_mesh_whole_would_cost(split: Any) -> None:
    """Blender's own per-face material indices, planned engine-free.

    The USD importer puts a ``materialBind`` subset, an FBX material group and an OBJ ``usemtl``
    on one footing -- a per-face index into the object's slot table -- which is why this pass is
    one mechanism for every source format and reads no USD itself.
    """
    report, _ = split
    assert report.prims_before == 3
    assert report.prims_after == 6
    assert [m.mesh for m in report.multi_material] == ["Facade", "Roof"]
    assert report.area_m2 == pytest.approx(14.0)
    # 4 m2 of facade and 1 m2 of roof would render as the wrong material.
    assert report.misassigned_area == pytest.approx(5.0 / 14.0)


def test_the_split_asset_has_one_prim_per_material_and_nothing_left_to_report(
    simulation_app: Any, split: Any
) -> None:
    """The claim `AI.6` exists for, read by the production walk inside Kit.

    ``subset_meshes`` empty is the whole point: there is no mesh left whose material varies
    across its own faces, so nothing the audit can see is beyond what the renderer can address.
    """
    del simulation_app
    from pxr import Usd

    from irsim_isaac.pipeline.materials_usd import walk_stage

    _, usd = split
    stage = Usd.Stage.Open(str(usd))
    assert stage is not None
    walk = walk_stage(stage, root="/")
    assert walk.subset_meshes == ()
    assert walk.shadowed == ()
    assert {r.path.rsplit("/", 1)[-1] for r in walk.records} == EXPECTED_PIECES
    assert {r.material_name for r in walk.records} == EXPECTED_PIECES


def test_every_face_survives_the_regrouping(simulation_app: Any, split: Any) -> None:
    """Geometry is moved, not rebuilt, so the face count must be conserved exactly.

    A split that dropped faces would still audit at 100 % coverage -- the prims that remain are
    all mapped -- which is why this is measured rather than assumed.
    """
    del simulation_app
    from pxr import Usd, UsdGeom

    _, usd = split
    stage = Usd.Stage.Open(str(usd))
    faces = 0
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Mesh):
            counts = UsdGeom.Mesh(prim).GetFaceVertexCountsAttr().Get()
            faces += 0 if counts is None else len(counts)
    assert faces == 9  # six facade, two roof, one plinth


def test_the_materials_themselves_come_through(simulation_app: Any, split: Any) -> None:
    """Faces move with Blender's `separate`, which carries the slots and UVs with them.

    Rebuilding the mesh from arrays would be simpler and would drop the textures the visible
    companion frame depends on (ADR 0073), which no thermal test would catch.
    """
    del simulation_app
    from pxr import Usd, UsdShade

    _, usd = split
    stage = Usd.Stage.Open(str(usd))
    materials = {prim.GetName() for prim in stage.Traverse() if prim.IsA(UsdShade.Material)}
    assert materials >= EXPECTED_PIECES
