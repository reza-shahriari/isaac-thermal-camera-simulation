"""One prim per material: the arithmetic that decides the split and names its pieces (`AI.6`).

`AI.4` taught the stage walk to read ``materialBind`` subsets, which made the *audit* true about
a multi-material mesh. The render stayed wrong, and by construction: the instance-id plane carries
one id per prim (ADR 0014), so one prim is one material whatever the asset says. The repair is to
regroup the geometry, and this module is the engine-free half of it -- what the pieces are called,
and how much area a render gets wrong without them.

The geometry work is Blender's and is exercised end to end in
``tests/integration/test_material_split_isaac.py`` on the committed fixture. What is here needs
neither Blender nor Kit.
"""

from __future__ import annotations

import numpy as np
import pytest

from irsim.io.asset_material_split import (
    MeshFaces,
    piece_name,
    plan_material_split,
    usd_identifier,
)

#: The committed fixture, as the Blender worker measures it: six 1 m x 1 m faces of facade in
#: three materials, a two-material roof, a single-material plinth.
FACADE = MeshFaces(
    name="Facade",
    slot_of_face=np.array([0, 0, 1, 1, 2, 2], dtype=np.int64),
    area_of_face=np.ones(6, dtype=np.float64),
    materials=["Facade_Glass_Clear", "Facade_Concrete_Precast", "Facade_Metal_Cladding"],
)
ROOF = MeshFaces(
    name="Roof",
    slot_of_face=np.array([0, 1], dtype=np.int64),
    area_of_face=np.ones(2, dtype=np.float64),
    materials=["Roof_Asphalt_Felt", "Roof_Alumin_Flashing"],
)
PLINTH = MeshFaces(
    name="Plinth",
    slot_of_face=np.zeros(1, dtype=np.int64),
    area_of_face=np.full(1, 6.0, dtype=np.float64),
    materials=["Plinth_Concrete_Base"],
)


def test_a_three_material_mesh_becomes_three_prims() -> None:
    """The count, and the areas that go with it."""
    report = plan_material_split([FACADE])
    (facade,) = report.meshes
    assert facade.multi_material
    assert [s.material for s in facade.slots] == list(FACADE.materials)
    assert [s.faces for s in facade.slots] == [2, 2, 2]
    assert [s.area_m2 for s in facade.slots] == [2.0, 2.0, 2.0]
    assert report.prims_before == 1
    assert report.prims_after == 3


def test_the_number_that_justifies_the_split_is_an_area_not_a_count() -> None:
    """Two thirds of the facade renders as the wrong substance without it.

    The asset-wide figure is what a prep run should be judged on: an asset can have forty
    multi-material meshes and be 0.1 % wrong, or one and be two thirds wrong.
    """
    (facade,) = plan_material_split([FACADE]).meshes
    assert facade.misassigned_area == pytest.approx(2.0 / 3.0)

    whole = plan_material_split([FACADE, ROOF, PLINTH])
    assert whole.area_m2 == pytest.approx(14.0)
    # 4 m2 of facade plus 1 m2 of roof, out of 14 m2.
    assert whole.misassigned_area == pytest.approx(5.0 / 14.0)
    assert whole.prims_before == 3
    assert whole.prims_after == 6
    assert [m.mesh for m in whole.multi_material] == ["Facade", "Roof"]


def test_the_dominant_slot_is_the_first_one_used_not_the_largest() -> None:
    """Blender writes the *first* slot as the mesh-level binding, and that is what a slot-0
    reader gets (ADR 0128). Calling the largest slot dominant would understate the error on
    exactly the asset that needs the split most.
    """
    lopsided = MeshFaces(
        name="Wall",
        slot_of_face=np.array([0, 1, 1, 1, 1], dtype=np.int64),
        area_of_face=np.ones(5, dtype=np.float64),
        materials=["Trim_Steel", "Wall_Brick"],
    )
    (wall,) = plan_material_split([lopsided]).meshes
    assert wall.dominant.material == "Trim_Steel"
    assert wall.misassigned_area == pytest.approx(0.8)
    # The negative control: reading the largest slot instead would report 20 %, which is the
    # error of a reader nobody has.
    largest = max(wall.slots, key=lambda s: s.area_m2)
    assert largest.material == "Wall_Brick"
    assert 1.0 - largest.area_m2 / wall.area_m2 == pytest.approx(0.2)


def test_a_single_material_mesh_is_one_prim_and_costs_nothing() -> None:
    report = plan_material_split([PLINTH])
    (plinth,) = report.meshes
    assert not plinth.multi_material
    assert plinth.misassigned_area == 0.0
    assert report.misassigned_area == 0.0
    assert plinth.slots[0].piece == "Plinth_Concrete_Base"


def test_a_slot_no_face_uses_becomes_no_prim() -> None:
    """An asset's slot table is longer than the materials it paints with, and an empty prim is
    geometry the renderer pays for and nobody sees."""
    sparse = MeshFaces(
        name="Panel",
        slot_of_face=np.array([2, 2], dtype=np.int64),
        area_of_face=np.ones(2, dtype=np.float64),
        materials=["unused_a", "unused_b", "Panel_Paint"],
    )
    (panel,) = plan_material_split([sparse]).meshes
    assert [s.material for s in panel.slots] == ["Panel_Paint"]


def test_piece_names_do_not_repeat_a_prefix_the_material_already_carries() -> None:
    assert piece_name("Facade", "Facade_Glass_Clear") == "Facade_Glass_Clear"
    assert piece_name("Wall", "Brick_Red") == "Wall_Brick_Red"
    assert piece_name("Wall", "") == "Wall"
    assert piece_name("wall 2", "brick.red") == "wall_2_brick_red"


def test_two_pieces_may_not_share_a_name() -> None:
    """Two prims with one name is how geometry disappears on export."""
    first = MeshFaces(
        name="Wall",
        slot_of_face=np.zeros(1, dtype=np.int64),
        area_of_face=np.ones(1, dtype=np.float64),
        materials=["Brick"],
    )
    second = MeshFaces(
        name="Wall",  # a second object of the same name: Blender allows it, USD does not
        slot_of_face=np.zeros(1, dtype=np.int64),
        area_of_face=np.ones(1, dtype=np.float64),
        materials=["Brick"],
    )
    pieces = [s.piece for m in plan_material_split([first, second]).meshes for s in m.slots]
    assert pieces == ["Wall_Brick", "Wall_Brick_1"]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Facade.001", "Facade_001"),
        ("wall glass", "wall_glass"),
        ("2_up", "_2_up"),
        ("", "unnamed"),
        ("...", "unnamed"),
    ],
)
def test_usd_identifiers_are_derived_here_rather_than_by_an_exporter(
    raw: str, expected: str
) -> None:
    """A scene config is authored against a prim path, so the path has to be one we chose."""
    assert usd_identifier(raw) == expected


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {"slot_of_face": np.zeros(3, dtype=np.int64), "area_of_face": np.ones(2)},
            "slots for",
        ),
        (
            {"slot_of_face": np.array([7], dtype=np.int64), "area_of_face": np.ones(1)},
            "references slot",
        ),
        (
            {"slot_of_face": np.array([-1], dtype=np.int64), "area_of_face": np.ones(1)},
            "negative material slot",
        ),
        (
            {
                "slot_of_face": np.zeros(1, dtype=np.int64),
                "area_of_face": np.ones(1, dtype=np.float16),
            },
            "float16",
        ),
    ],
)
def test_a_malformed_mesh_is_refused_rather_than_planned(
    kwargs: dict[str, object], message: str
) -> None:
    """Every one of these renders a plausible building made of the wrong things."""
    with pytest.raises(ValueError, match=message):
        MeshFaces(name="Wall", materials=["Brick"], **kwargs)  # type: ignore[arg-type]


def _prep_module() -> object:
    """`scripts/prep_asset.py` loaded by path: it is a script, not an importable package."""
    import importlib.util
    import pathlib

    path = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "prep_asset.py"
    spec = importlib.util.spec_from_file_location("prep_asset_split", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_both_blender_passes_carry_the_scale_and_the_source() -> None:
    """The two halves of the split talk through argv and files, across a process boundary.

    Blender's interpreter has neither pydantic nor `irsim`, so the scale crosses as text and a
    silently truncated one would rescale an entire asset without a word.
    """
    import pathlib

    prep = _prep_module()
    measure = prep.blender_material_slots_command(  # type: ignore[attr-defined]
        "blender", pathlib.Path("a.usda"), 1.0 / 3.0, pathlib.Path("out/slots.json")
    )
    split = prep.blender_material_split_command(  # type: ignore[attr-defined]
        "blender",
        pathlib.Path("a.usda"),
        1.0 / 3.0,
        pathlib.Path("out/plan.json"),
        pathlib.Path("out/a_materials.usdc"),
    )
    for cmd in (measure, split):
        assert "--background" in cmd and "--factory-startup" in cmd
        assert cmd[cmd.index("--") + 1 :][:2] == ["--source", "a.usda"]
        assert float(cmd[cmd.index("--scale") + 1]) == 1.0 / 3.0
    assert measure[-2:] == ["--out-material-slots", "out/slots.json"]
    assert split[-4:] == [
        "--split-material-plan",
        "out/plan.json",
        "--split-out-usd",
        "out/a_materials.usdc",
    ]
