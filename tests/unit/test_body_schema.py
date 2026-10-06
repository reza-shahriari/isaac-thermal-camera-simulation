"""HU.2 -- the body schema is data, and the data is checked.

A human is labelled onto JOS-3's seventeen segments by its skeleton, so the things that would
make a person render plausibly and wrongly are: a rig that leaves a bone unmapped (its vertices
fall off the body), a bone mapped to the wrong side (a warm left hand on the right), a region
that never resolves to a segment, a garment on a slot the body has not got. Each of those is
refused here, at load, by name.

docs/physics-model.md §6.1, §16.2; roadmap HU.2; ADR 0122, ADR 0138.
"""

from __future__ import annotations

import copy
import pathlib

import pytest
import yaml

from irsim.config.humans import (
    BODY_SCHEMA_PATH,
    JOS3_SEGMENTS,
    BodySchema,
    HumanSpec,
    load_body_schema,
    segment_for_bone,
)
from irsim.materials.library import MaterialLibrary
from irsim.materials.mapping import AssetMapping, load_asset_mapping

#: JOS-3's standard body is 1.87 m² (construction.py's comment on _BSAst).
JOS3_STANDARD_BSA_M2 = 1.87


@pytest.fixture(scope="module")
def schema() -> BodySchema:
    return load_body_schema()


@pytest.fixture()
def raw() -> dict:
    return yaml.safe_load(BODY_SCHEMA_PATH.read_text(encoding="utf-8"))


def _load(raw: dict, tmp_path: pathlib.Path) -> BodySchema:
    p = tmp_path / "body.yaml"
    p.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return load_body_schema(p)


# --- the committed schema --------------------------------------------------------------------


def test_the_segments_are_jos3s_seventeen_in_its_order(schema: BodySchema) -> None:
    assert [s.name for s in schema.segments] == list(JOS3_SEGMENTS)
    assert len(schema.segments) == 17


def test_the_standard_areas_sum_to_jos3s_body(schema: BodySchema) -> None:
    """_BSAst sums to 1.87 m²; a mistyped area would move the body's total."""
    total = sum(s.area_m2 for s in schema.segments)
    assert total == pytest.approx(JOS3_STANDARD_BSA_M2, abs=0.005)
    # the mirrored pairs carry the same area
    for s in schema.segments:
        if s.mirror:
            assert schema.segment(s.mirror).area_m2 == s.area_m2


def test_every_rig_drives_every_segment(schema: BodySchema) -> None:
    assert set(schema.rigs) >= {"mixamo", "rigify_human", "smplx", "mpfb_game_engine"}
    for rig in schema.rigs:
        assert schema.coverage(rig) == schema.segment_names, rig


def test_every_rig_is_mirror_symmetric(schema: BodySchema) -> None:
    for rig in schema.rigs:
        assert schema.asymmetries(rig) == [], rig


@pytest.mark.parametrize(
    ("rig", "count"),
    [("mixamo", 52), ("mpfb_game_engine", 52), ("smplx", 55), ("rigify_human", 159)],
)
def test_the_deforming_bone_lists_are_complete(schema: BodySchema, rig: str, count: int) -> None:
    """The sources' counts: Mixamo 65 − 13 leaves, GameEngine 53 − Root, SMPL-X 55, Rigify 159."""
    assert len(schema.rigs[rig].bones) == count


def test_the_torso_region_resolves_to_chest_and_back(schema: BodySchema) -> None:
    assert schema.resolve("torso") == {"Chest", "Back"}
    assert schema.resolve("LHand") == {"LHand"}
    with pytest.raises(KeyError):
        schema.resolve("Arm")


def test_garment_slots_cover_everything_but_the_neck(schema: BodySchema) -> None:
    """The Neck is bare by default: a scarf names it with `covers:`."""
    covered: set[str] = set()
    for slot in schema.garment_slots.values():
        covered |= set(slot.covers)
    assert covered == schema.segment_names - {"Neck"}


# --- the lookup ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("rig", "bone", "expected"),
    [
        ("mixamo", "mixamorig:LeftHandIndex3", "LHand"),
        ("mixamo", "mixamorig:RightForeArm", "RArm"),
        ("mixamo", "mixamorig_LeftUpLeg", "LThigh"),  # FBX importers rewrite the colon
        ("mixamo", "Spine1", "torso"),
        ("mixamo", "mixamorig:HeadTop_End", None),  # a leaf that deforms nothing
        ("mixamo", "mixamorig:LeftHandThumb4", None),
        ("rigify_human", "DEF-upper_arm.L", "LShoulder"),
        ("rigify_human", "DEF-upper_arm.L.001", "LShoulder"),  # the second limb piece
        ("rigify_human", "DEF-forearm.R.001", "RArm"),
        ("rigify_human", "DEF-spine.006", "Head"),  # an exact `.NNN` that is its own bone
        ("rigify_human", "DEF-spine.001", "Pelvis"),
        ("rigify_human", "DEF-f_index.02.L", "LHand"),
        ("rigify_human", "DEF-breast.L", "Chest"),
        ("rigify_human", "ORG-hand.L", None),  # a non-deforming organisational bone
        ("smplx", "left_knee", "LLeg"),  # the joint drives the segment distal to it
        ("smplx", "left_elbow", "LArm"),
        ("smplx", "right_hip", "RThigh"),
        ("mpfb_game_engine", "calf_r", "RLeg"),
        ("mpfb_game_engine", "ball_l", "LFoot"),
        ("mpfb_game_engine", "Root", None),
    ],
)
def test_segment_for_bone(schema: BodySchema, rig: str, bone: str, expected: str | None) -> None:
    assert segment_for_bone(schema, rig, bone) == expected


def test_an_unknown_bone_is_an_error_that_names_it(schema: BodySchema) -> None:
    with pytest.raises(KeyError, match="tail"):
        segment_for_bone(schema, "mixamo", "mixamorig:tail")
    with pytest.raises(KeyError, match="unknown rig"):
        segment_for_bone(schema, "unreal5", "spine_01")
    with pytest.raises(ValueError, match=r"does not map bones \['Tail'\]"):
        schema.check_bones("mixamo", ["Hips", "Tail"])


def test_check_bones_labels_a_whole_armature(schema: BodySchema) -> None:
    bones = ["mixamorig:" + b for b in schema.rigs["mixamo"].bones] + ["mixamorig:HeadTop_End"]
    labels = schema.check_bones("mixamo", bones)
    assert labels["mixamorig:HeadTop_End"] is None
    assert set(labels.values()) - {None} == schema.segment_names - {"Chest", "Back"} | {"torso"}


# --- refusals ---------------------------------------------------------------------------------


def test_a_schema_missing_a_segment_is_refused_naming_it(raw: dict, tmp_path: pathlib.Path) -> None:
    bad = copy.deepcopy(raw)
    bad["body"]["segments"] = [s for s in bad["body"]["segments"] if s["name"] != "LHand"]
    with pytest.raises(ValueError, match=r"missing \['LHand'\]"):
        _load(bad, tmp_path)


def test_a_renamed_segment_is_refused(raw: dict, tmp_path: pathlib.Path) -> None:
    bad = copy.deepcopy(raw)
    for s in bad["body"]["segments"]:
        if s["name"] == "LLeg":
            s["name"] = "LShin"
            s["mirror"] = "RLeg"
    with pytest.raises(ValueError, match="LShin"):
        _load(bad, tmp_path)


def test_a_rig_with_an_unmapped_segment_is_refused(raw: dict, tmp_path: pathlib.Path) -> None:
    bad = copy.deepcopy(raw)
    bones = bad["body"]["rigs"]["mixamo"]["bones"]
    for b in ("LeftFoot", "LeftToeBase", "RightFoot", "RightToeBase"):
        del bones[b]
    with pytest.raises(ValueError, match=r"mixamo.*LFoot.*RFoot"):
        _load(bad, tmp_path)


def test_a_bone_on_the_wrong_side_is_refused(raw: dict, tmp_path: pathlib.Path) -> None:
    bad = copy.deepcopy(raw)
    bad["body"]["rigs"]["smplx"]["bones"]["right_wrist"] = "LHand"
    with pytest.raises(ValueError, match="not mirror-symmetric.*right_wrist"):
        _load(bad, tmp_path)


def test_a_bone_mapped_to_nothing_known_is_refused(raw: dict, tmp_path: pathlib.Path) -> None:
    bad = copy.deepcopy(raw)
    bad["body"]["rigs"]["mpfb_game_engine"]["bones"]["spine_02"] = "Torso"
    with pytest.raises(ValueError, match="unknown targets"):
        _load(bad, tmp_path)


def test_a_slot_covering_a_non_segment_is_refused(raw: dict, tmp_path: pathlib.Path) -> None:
    bad = copy.deepcopy(raw)
    bad["body"]["garment_slots"]["hat"] = {"covers": ["Hair"]}
    with pytest.raises(ValueError, match="hat.*Hair"):
        _load(bad, tmp_path)


# --- the asset side: kind: human --------------------------------------------------------------


def _human_asset(tmp_path: pathlib.Path, **human) -> pathlib.Path:
    body = {
        "phenotype": {"sex": "female", "age_y": 31.0, "height_m": 1.66, "mass_kg": 61.0},
        "garments": {
            "torso": {"material": "cotton_clothing", "clo": 0.25, "colour_rgb": [0.1, 0.2, 0.7]},
            "legs": {"material": "cotton_clothing", "clo": 0.3},
        },
    }
    body.update(human)
    doc = {
        "schema_version": 1,
        "asset": {
            "name": "test_woman",
            "kind": "human",
            "materials": {"Skin": "human_skin"},
            "human": body,
        },
    }
    p = tmp_path / "test_woman.yaml"
    p.write_text(yaml.safe_dump(doc), encoding="utf-8")
    return p


def test_a_human_asset_loads_and_its_garments_count_as_targets(tmp_path: pathlib.Path) -> None:
    lib = MaterialLibrary.load()
    asset = load_asset_mapping(_human_asset(tmp_path), known_materials=lib.names)
    assert asset.kind == "human" and asset.human is not None
    assert asset.targets == {"human_skin", "cotton_clothing"}
    schema = load_body_schema()
    assert asset.human.coverage(schema, "torso") == {"Chest", "Back", "Pelvis"}


def test_a_garment_may_narrow_or_widen_its_slot(tmp_path: pathlib.Path) -> None:
    schema = load_body_schema()
    p = _human_asset(
        tmp_path,
        garments={
            "torso": {
                "material": "cotton_clothing",
                "clo": 0.3,
                "covers": ["Chest", "Back", "Pelvis", "Neck"],
            },
            "arms": {
                "material": "cotton_clothing",
                "clo": 0.1,
                "covers": ["LShoulder", "RShoulder"],
            },
        },
    )
    h = load_asset_mapping(p).human
    assert h is not None
    assert h.coverage(schema, "torso") == {"Chest", "Back", "Pelvis", "Neck"}  # a collar
    assert h.coverage(schema, "arms") == {"LShoulder", "RShoulder"}  # short sleeves


def test_a_garment_on_an_unknown_slot_is_refused(tmp_path: pathlib.Path) -> None:
    p = _human_asset(tmp_path, garments={"cape": {"material": "cotton_clothing", "clo": 0.5}})
    with pytest.raises(ValueError, match="unknown slot 'cape'"):
        load_asset_mapping(p)


def test_a_garment_covering_a_non_segment_is_refused(tmp_path: pathlib.Path) -> None:
    p = _human_asset(
        tmp_path, garments={"head": {"material": "cotton_clothing", "clo": 0.1, "covers": ["Face"]}}
    )
    with pytest.raises(ValueError, match=r"covers \['Face'\]"):
        load_asset_mapping(p)


def test_a_garment_of_an_unknown_material_is_refused(tmp_path: pathlib.Path) -> None:
    p = _human_asset(tmp_path, garments={"legs": {"material": "gore_tex", "clo": 0.4}})
    with pytest.raises(ValueError, match="gore_tex"):
        load_asset_mapping(p, known_materials=MaterialLibrary.load().names)


def test_kind_and_block_go_together() -> None:
    with pytest.raises(ValueError, match="go together"):
        AssetMapping(name="x", kind="human")
    spec = HumanSpec(phenotype={"sex": "male", "age_y": 8, "height_m": 1.28, "mass_kg": 26})
    with pytest.raises(ValueError, match="go together"):
        AssetMapping(name="x", kind="object", human=spec)
    assert AssetMapping(name="x", kind="human", human=spec).human is spec


@pytest.mark.parametrize(
    "phenotype",
    [
        {"sex": "male", "age_y": 0.0, "height_m": 1.7, "mass_kg": 70},
        {"sex": "male", "age_y": 30, "height_m": 3.0, "mass_kg": 70},
        {"sex": "female", "age_y": 30, "height_m": 1.7, "mass_kg": 1.0},
        {"sex": "other", "age_y": 30, "height_m": 1.7, "mass_kg": 70},
    ],
)
def test_an_impossible_phenotype_is_refused(phenotype: dict) -> None:
    with pytest.raises(ValueError):
        HumanSpec(phenotype=phenotype)


def test_a_colour_outside_reflectance_is_refused() -> None:
    with pytest.raises(ValueError, match="colour_rgb"):
        HumanSpec(
            phenotype={"sex": "male", "age_y": 30, "height_m": 1.7, "mass_kg": 70},
            garments={
                "torso": {"material": "cotton_clothing", "clo": 0.2, "colour_rgb": [255, 0, 0]}
            },
        )
