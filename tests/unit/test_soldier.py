"""HU.8 -- a soldier, and camouflage whose near infrared is engineered, not dyed.

A dark green dye goes dark in the near infrared; foliage does not. That is why camouflage
specifications set NIR reflectance targets per pattern colour, and why the soldier's cloth is a
material of its own with `dye_optics: false` -- its colour tints the RGB companion only.

What would be wrong and invisible: a colour that quietly overwrote an engineered NIR value (the
uniform rendering as a black figure on a bright field in an NIR camera, which is what an olive
cotton does), a helmet that insulated nothing, a pack with no material. Each is a case here.

docs/physics-model.md §4.2, §6.1, §16.2; roadmap HU.8; ADR 0195, ADR 0196, ADR 0197.
"""

from __future__ import annotations

import pathlib

import pytest
import yaml

pytest.importorskip("jos3")

from irsim.config.humans import HumanSpec  # noqa: E402
from irsim.config.scene import SceneConfig  # noqa: E402
from irsim.materials.colour import coloured_material, dress_asset_materials  # noqa: E402
from irsim.materials.library import MaterialLibrary  # noqa: E402
from irsim.materials.mapping import load_asset_mapping  # noqa: E402
from irsim.scene import Scene  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[2]
OLIVE = (0.24, 0.26, 0.14)
#: Camouflage NIR targets over 700-1100 nm for a woodland pattern's colours (Yahaya et al. 2026,
#: doi:10.1177/15280837251319055): green 45-55 %, brown 25-40 %, beige 60-70 %, black 5-20 %.
#: A pattern of about 40 % green, 30 % brown, 20 % beige and 10 % black averages 0.37-0.50.
PATTERN_NIR_REFLECTANCE = (0.37, 0.50)


@pytest.fixture(scope="module")
def library() -> MaterialLibrary:
    return MaterialLibrary.load()


def _scene(stem: str) -> Scene:
    with open(REPO / "configs" / "scenes" / f"{stem}.yaml", encoding="utf-8") as fh:
        return Scene.from_config(SceneConfig.model_validate(yaml.safe_load(fh)))


def test_the_camouflage_matches_the_pattern_targets_and_a_dyed_olive_does_not(library) -> None:  # type: ignore[no-untyped-def]
    camo = library["camouflage_nir_compliant"].spec.optical.emissivity_per_band
    assert camo is not None
    rho_camo = 1.0 - camo["nir"]
    lo, hi = PATTERN_NIR_REFLECTANCE
    assert lo <= rho_camo <= hi
    # the same olive as a plain dye on cotton goes dark in the NIR: why camouflage is authored
    dyed = coloured_material(library["cotton_clothing"], OLIVE).spec.optical.emissivity_per_band
    assert dyed is not None
    assert 1.0 - dyed["nir"] < lo, "a dyed olive (0.30) falls below the pattern's band"
    # and the long-wave is cloth's, whatever the pattern (Zhang 2009; Belliveau 2020)
    assert camo["lwir"] == pytest.approx(0.88)


def test_a_garment_that_is_not_dyed_keeps_its_material(library) -> None:  # type: ignore[no-untyped-def]
    human = HumanSpec(
        phenotype={"sex": "male", "age_y": 25, "height_m": 1.73, "mass_kg": 72},
        garments={
            "arms": {
                "material": "camouflage_nir_compliant",
                "clo": 0.28,
                "colour_rgb": OLIVE,
                "dye_optics": False,
            },
            "legs": {"material": "cotton_clothing", "clo": 0.24, "colour_rgb": OLIVE},
        },
    )  # fmt: skip
    materials = {"garment_arms": "camouflage_nir_compliant", "garment_legs": "cotton_clothing"}
    dressed, _ = dress_asset_materials(materials, human, library)
    assert dressed["garment_arms"] == "camouflage_nir_compliant", "the colour must not dye it"
    assert dressed["garment_legs"] != "cotton_clothing", "an ordinary garment is still dyed"


def test_the_soldier_is_garments_and_equipment_with_a_material_each(library) -> None:  # type: ignore[no-untyped-def]
    asset = load_asset_mapping("soldier", known_materials=library.names)
    h = asset.human
    assert h is not None and asset.parts is not None
    assert set(h.garments) == {"head", "torso", "arms", "legs", "feet"}
    assert set(h.equipment) == {"belt", "pack"}
    for slot in ("torso", "arms", "legs", "head"):
        assert h.garments[slot].dye_optics is False, slot
    names = {p.name for p in asset.parts.parts}
    assert {"equipment_belt", "equipment_pack", "garment_head"} <= names


def test_in_winter_the_helmet_and_carrier_sit_near_the_air_and_the_face_does_not() -> None:
    scene = _scene("soldier_winter_night")
    t = scene.advance_targets(0.0, 30.0)
    air = scene.weather_at(0.0).t_air_k
    face = t["soldier.skin_Head"]
    for part in ("soldier.garment_head", "soldier.garment_torso"):
        assert air < t[part] < face - 2.0, (part, t[part] - 273.15, face - 273.15)
    # the carrier's 1.38 clo puts it nearer the air than the 0.28-clo sleeves
    assert t["soldier.garment_torso"] < t["soldier.garment_arms"]
    # the pack sits on the carrier: with no sun it is the carrier's temperature
    assert t["soldier.equipment_pack"] == pytest.approx(t["soldier.garment_torso"], abs=1e-9)
