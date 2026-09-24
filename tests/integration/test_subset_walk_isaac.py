"""The `materialBind` subset branch of the stage walk, run on a real USD file (`AI.4`).

``scripts/prep_asset.py`` and ``irsim_isaac.pipeline.materials_usd`` both claim to read
``GeomSubset`` material assignments, and until this test neither claim had been exercised on a
USD file: every asset committed here had one material per mesh, so the branch was dead code that
looked alive. ``tests/unit/data/subset_building.usda`` is the fixture that runs it -- six faces
of facade split into glass, precast concrete and metal cladding, a two-material roof, a
single-material plinth, and an invisible scaffold that also carries subsets.

What needs Kit is only the walk: ``pxr`` is a Kit extension and is not importable outside a
running Kit application on this build (ADR 0014 addendum). What the resolver then does with the
records, and what a wrong reading costs in kelvin, is engine-free and lives in
``tests/unit/test_subset_materials.py``.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any

import pytest

FIXTURE = pathlib.Path(__file__).resolve().parents[1] / "unit" / "data" / "subset_building.usda"
DUMP = pathlib.Path(__file__).resolve().parents[1] / "unit" / "data" / "subset_building.prims.json"

FACADE = "/World/Building/Facade"
ROOF = "/World/Building/Roof"
PLINTH = "/World/Building/Plinth"
SCAFFOLD = "/World/Building/Scaffold"


@pytest.fixture(scope="module")
def stage(simulation_app: Any) -> Any:
    """The committed fixture, opened on its own stage rather than on the session's."""
    del simulation_app
    from pxr import Usd

    opened = Usd.Stage.Open(str(FIXTURE))
    assert opened is not None, f"could not open {FIXTURE}"
    return opened


def test_the_default_walk_reads_one_material_per_prim(stage: Any) -> None:
    """Unchanged behaviour, and it is the behaviour the renderer needs.

    The instance-id plane carries one id per prim (ADR 0014) and the material table is indexed by
    it, so a render driver can only transport one material per prim. The default reading is
    therefore the prim's own binding -- which on the facade is Blender's slot 0, and is wrong
    about two thirds of the wall. The point of this test is that the wrongness is *stated*: the
    walk reports the mesh rather than quietly resolving it.
    """
    from irsim_isaac.pipeline.materials_usd import walk_stage

    walk = walk_stage(stage, root="/World")
    by_path = {r.path: r.material_name for r in walk.records}
    assert set(by_path) == {FACADE, ROOF, PLINTH}
    assert by_path[FACADE] == "Facade_Glass_Clear"
    assert by_path[ROOF] is None  # no mesh-level binding at all: a miss, not a guess
    assert by_path[PLINTH] == "Plinth_Concrete_Base"
    assert walk.subset_meshes == (FACADE, ROOF)
    assert walk.shadowed == (FACADE,)


def test_expanding_subsets_gives_one_record_per_face_group(stage: Any) -> None:
    """What the asset actually says: six surfaces, five of them on two meshes."""
    from irsim_isaac.pipeline.materials_usd import walk_stage

    walk = walk_stage(stage, root="/World", expand_subsets=True)
    by_path = {r.path: r.material_name for r in walk.records}
    assert by_path == {
        f"{FACADE}/glazing": "Facade_Glass_Clear",
        f"{FACADE}/precast": "Facade_Concrete_Precast",
        f"{FACADE}/cladding": "Facade_Metal_Cladding",
        f"{ROOF}/felt": "Roof_Asphalt_Felt",
        f"{ROOF}/flashing": "Roof_Alumin_Flashing",
        PLINTH: "Plinth_Concrete_Base",
    }
    # The mesh-level binding is ignored, not preferred: `Facade_Glass_Clear` survives only on the
    # face group that authored it.
    assert sum(1 for v in by_path.values() if v == "Facade_Glass_Clear") == 1


def test_the_committed_dump_is_what_this_walk_produces(stage: Any) -> None:
    """The fixture the unit tests read is regenerable from the fixture they cite.

    A dump that drifted from its stage would keep every engine-free test passing while describing
    an asset that no longer exists -- the failure mode a committed pair exists to prevent.
    """
    from irsim_isaac.pipeline.materials_usd import walk_stage

    walk = walk_stage(stage, root="/World", expand_subsets=True)
    committed = json.loads(DUMP.read_text(encoding="utf-8"))
    assert [
        {
            "path": r.path,
            "material_name": r.material_name,
            "semantic_class": r.semantic_class,
            "override": r.override,
        }
        for r in walk.records
    ] == committed


def test_an_invisible_mesh_contributes_nothing_even_though_it_has_subsets(stage: Any) -> None:
    """Visibility first, subsets second -- and `include_invisible` still reaches it.

    A walk that read subsets before visibility would report the scaffold's two face groups as
    surfaces, inflating the coverage fraction ADR 0047 gates on with geometry no ray meets.
    """
    from irsim_isaac.pipeline.materials_usd import walk_stage

    visible = walk_stage(stage, root="/World", expand_subsets=True)
    assert not [r for r in visible.records if SCAFFOLD in r.path]
    assert SCAFFOLD not in visible.subset_meshes

    hidden = walk_stage(stage, root="/World", expand_subsets=True, include_invisible=True)
    assert [r.path for r in hidden.records if SCAFFOLD in r.path] == [
        f"{SCAFFOLD}/poles",
        f"{SCAFFOLD}/boards",
    ]


def test_the_materials_the_subsets_name_are_the_ones_the_file_authors(stage: Any) -> None:
    """A name, not a path (ADR 0047), and it comes off the subset rather than off the mesh."""
    from pxr import UsdShade

    from irsim_isaac.pipeline.materials_usd import MATERIAL_BIND_FAMILY

    facade = stage.GetPrimAtPath(FACADE)
    subsets = UsdShade.MaterialBindingAPI(facade).GetMaterialBindSubsets()
    assert [s.GetPrim().GetName() for s in subsets] == ["glazing", "precast", "cladding"]
    for subset in subsets:
        family = subset.GetPrim().GetAttribute("familyName").Get()
        assert family == MATERIAL_BIND_FAMILY
