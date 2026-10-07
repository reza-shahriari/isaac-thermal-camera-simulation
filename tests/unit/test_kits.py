"""HU.11 -- an occupation's outfit is a YAML kit, cut by one routine, with no new code.

The police vest (HU.7) and the soldier's plate carrier (HU.8) were Python functions in
``make_human.py``; they are now ``configs/humans/kits/*.yaml``, and rebuilding both from their kits
gave glTF files byte-identical to the functions' (checked when the kits were written; the
generator needs Blender, so it is not re-run here). What is tested here is what would be wrong
and still render: a kit dressing a slot its asset does not solve (the piece would render with no
temperature), equipment that sits *inside* the garment it is on (retroreflective trim hidden in
the coat), a bone pattern that matches no bone (an empty piece), and a piece standing behind one
not yet made.

docs/physics-model.md §6.1; roadmap HU.11; ADR 0196, ADR 0198.
"""

from __future__ import annotations

import pathlib

import pytest
import yaml
from pydantic import ValidationError

from irsim.config.humans import load_body_schema
from irsim.config.kits import HumanKit, kit_names, load_kit
from irsim.materials.mapping import load_asset_mapping

REPO = pathlib.Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def schema():  # type: ignore[no-untyped-def]
    return load_body_schema()


def _raw(name: str) -> dict:
    return yaml.safe_load((REPO / "configs" / "humans" / "kits" / f"{name}.yaml").read_text())


def test_the_three_occupations_are_kits() -> None:
    assert {"police_officer", "soldier", "firefighter"} <= set(kit_names())


@pytest.mark.parametrize("name", kit_names())
def test_every_kit_names_real_bones_and_slots_and_matches_its_asset(name: str, schema) -> None:  # type: ignore[no-untyped-def]
    kit = load_kit(name)
    kit.check_against(schema)
    asset = load_asset_mapping(name)
    assert asset.human is not None
    kit.check_asset(asset.human)


@pytest.mark.parametrize("name", kit_names())
def test_equipment_stands_proud_of_the_garment_it_is_on(name: str) -> None:
    """A trim band 1 mm inside its coat is invisible to every camera and still in the solve."""
    kit = load_kit(name)
    asset = load_asset_mapping(name)
    assert asset.human is not None
    shells = {p.name: p.shell for p in kit.pieces if p.shell is not None}
    for item, spec in asset.human.equipment.items():
        piece, under = shells.get(f"equipment_{item}"), shells.get(f"garment_{spec.on}")
        if piece is None or under is None:
            continue  # a rigid piece, or a garment from MakeHuman's packs
        if set(piece.bones) <= set(under.bones):
            assert piece.offset_m > under.offset_m, (item, spec.on)


def test_a_kit_that_cannot_be_built_is_refused(schema) -> None:  # type: ignore[no-untyped-def]
    base = _raw("soldier")["kit"]
    pack_first = dict(base, pieces=[base["pieces"][2], *base["pieces"][:2]])
    with pytest.raises(ValidationError, match="not an earlier piece"):
        HumanKit.model_validate(pack_first)
    twice = dict(base, clothes={**base["clothes"], "torso": "toigo_basic_tucked_t-shirt"})
    with pytest.raises(ValidationError, match="both a MakeHuman asset and a piece"):
        HumanKit.model_validate(twice)
    shell = {"bones": ["pelvis"], "offset_m": 0.01, "bands": [[0.5, 0.6]], "rings": [[0.1, 0.2]]}
    with pytest.raises(ValidationError, match="one"):
        HumanKit.model_validate(
            dict(base, pieces=[{"name": "equipment_x", "colour_rgb": [0, 0, 0], "shell": shell}])
        )
    with pytest.raises(ValidationError, match="garment_<slot> or equipment_<name>"):
        HumanKit.model_validate(
            dict(base, pieces=[{"name": "vest", "colour_rgb": [0, 0, 0], "box": {
                "size_m": [0.1, 0.1, 0.1], "centre_z": 0.5, "behind": "x"}}])
        )  # fmt: skip
    nobone = dict(base, pieces=[dict(base["pieces"][0], shell=dict(
        base["pieces"][0]["shell"], bones=["spine_09"]))])  # fmt: skip
    with pytest.raises(ValueError, match="matches no bone"):
        HumanKit.model_validate(nobone).check_against(schema)
    nohands = dict(base, clothes={**base["clothes"], "tail": "x"})
    with pytest.raises(ValueError, match="slot 'tail'"):
        HumanKit.model_validate(nohands).check_against(schema)
