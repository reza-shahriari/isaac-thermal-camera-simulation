"""HU.3 -- labelling a skinned human by its skeleton, tested without Blender.

What would go wrong silently: a vertex given to a control bone's weight, a face voted to the
wrong side on a tie, a torso split about the origin instead of about itself, a body facing the
other way so Chest and Back swap. Each is a case here. The committed man (`man_adult`) is checked
through the asset loader: seventeen skin parts named for the seventeen segments, and a phenotype a
real adult could have.

docs/physics-model.md §6.1, §16.2; roadmap HU.3; ADR 0192.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from irsim.config.humans import JOS3_SEGMENTS, load_body_schema
from irsim.io.human_labels import AXES, face_labels, forward_check, split_region, vertex_labels
from irsim.materials.library import MaterialLibrary
from irsim.materials.mapping import load_asset_mapping

# labels used below: 0 Pelvis, 1 torso (a region), 2 Chest, 3 Back, 4 LLeg, 5 LFoot


def test_a_vertex_takes_its_heaviest_deforming_bone_and_ignores_control_bones() -> None:
    # groups: 0 pelvis bone, 1 spine bone, 2 a control bone (label -1), 3 shin bone
    weights = np.array(
        [
            [0.7, 0.3, 0.0, 0.0],  # pelvis
            [0.2, 0.8, 0.0, 0.0],  # torso
            [0.1, 0.2, 0.9, 0.0],  # the control bone is heaviest but does not count -> torso
            [0.0, 0.0, 1.0, 0.0],  # only a control weight -> unlabelled
            [0.0, 0.0, 0.0, 0.0],  # no weight at all -> unlabelled
        ]
    )
    labels = vertex_labels(weights, np.array([0, 1, -1, 4]))
    assert labels.tolist() == [0, 1, 1, -1, -1]


def test_a_face_votes_and_an_unlabelled_face_takes_its_nearest_neighbour() -> None:
    vl = np.array([0, 0, 1, 1, -1, -1, -1])
    faces = [[0, 1, 2], [0, 1, 2, 3], [4, 5, 6], [1, 2, 3]]
    centres = np.array([[0, 0, 0], [0, 0, 1], [0, 0, 1.1], [5, 5, 5]], dtype=float)
    out = face_labels(faces, vl, centres)
    # face 0: two votes for 0, one for 1 -> 0; face 1: a 2-2 tie -> the lowest label, 0; face 2:
    # no labelled vertex, nearest labelled centre is face 1 (0.1 away) -> 0; face 3: two for 1 -> 1
    assert out.tolist() == [0, 0, 0, 1]


def test_a_mesh_with_no_deforming_weight_is_refused() -> None:
    with pytest.raises(ValueError, match="no deforming weight"):
        face_labels([[0, 1, 2]], np.array([-1, -1, -1]), np.zeros((1, 3)))


def test_the_torso_splits_about_its_own_centre_not_the_origin() -> None:
    # a torso displaced to y = +10: splitting about the origin would make every face Chest
    centres = np.array(
        [[0, 10.5, 1], [0, 9.5, 1], [0, 10.2, 1], [0, 9.8, 1], [0, 0, 0]], dtype=float
    )
    fl = np.array([1, 1, 1, 1, 0])
    out = split_region(fl, centres, region=1, positive=2, negative=3, forward=AXES["+y"])
    assert out.tolist() == [2, 3, 2, 3, 0]
    # the other way round for a body facing -y
    out = split_region(fl, centres, region=1, positive=2, negative=3, forward=AXES["-y"])
    assert out.tolist() == [3, 2, 3, 2, 0]


def test_the_feet_say_which_way_the_body_faces() -> None:
    # a shin at the origin, a foot 0.13 m toward -y and 0.4 m lower: toes point -y
    centres = np.array([[0, 0, 0.4], [0, -0.13, 0.0]])
    fl = np.array([4, 5])
    assert forward_check(fl, centres, [(5, 4)], AXES["-y"]) == {5: pytest.approx(0.13)}
    assert forward_check(fl, centres, [(5, 4)], AXES["+y"]) == {5: pytest.approx(-0.13)}
    # the vertical offset never leaks into the answer
    assert forward_check(fl, centres, [(5, 4)], AXES["+x"]) == {5: pytest.approx(0.0)}


def test_the_generator_maps_makehumans_age_slider_to_years() -> None:
    """MakeHuman: 1 y -> 0.0, 11 y -> 0.1875, 25 y -> 0.5, 90 y -> 1.0; the script runs in
    Blender, so `bpy` is stubbed to import it."""
    import importlib.util
    import pathlib

    sys.modules.setdefault("bpy", types.ModuleType("bpy"))
    path = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "make_human.py"
    spec = importlib.util.spec_from_file_location("make_human", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.years_to_age_macro(11.0) == pytest.approx(0.1875)
    assert mod.years_to_age_macro(25.0) == pytest.approx(0.5)
    assert mod.age_macro_to_years(0.5) == pytest.approx(25.0)
    assert mod.age_macro_to_years(1.0) == pytest.approx(90.0)
    for years in (3.0, 8.0, 35.0, 70.0):
        assert mod.age_macro_to_years(mod.years_to_age_macro(years)) == pytest.approx(years)


# --- the committed man --------------------------------------------------------------------------


def test_the_man_is_a_human_asset_with_a_part_per_segment() -> None:
    asset = load_asset_mapping("man_adult", known_materials=MaterialLibrary.load().names)
    assert asset.kind == "human" and asset.human is not None and asset.parts is not None
    assert asset.parts.granularity == "object"
    names = {p.name for p in asset.parts.parts}
    assert {f"skin_{s}" for s in JOS3_SEGMENTS} <= names
    for p in asset.parts.parts:
        assert [o.lower() for o in p.select.objects] == [p.name.lower()], (
            "a human's part is the object the split named"
        )
    schema = load_body_schema()
    assert asset.human.rig in schema.rigs
    ph = asset.human.phenotype
    assert ph.sex == "male" and 1.5 <= ph.height_m <= 2.0 and 50.0 <= ph.mass_kg <= 110.0
    # mass is the stated BMI 23 on the measured stature, not a number typed from memory
    assert ph.mass_kg == pytest.approx(23.0 * ph.height_m**2, abs=0.1)
