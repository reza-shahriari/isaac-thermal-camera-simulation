"""HU.7 -- an occupation is garments plus equipment, and the vest's tape is not a mirror.

A police officer is the dressed body with a heavier torso (pullover, soft armour, hi-vis vest) and
two pieces of equipment with materials of their own: retroreflective tape on the vest and a leather
belt on the trousers. Equipment sits on a garment slot -- that garment's insulation over that
garment's skin -- and absorbs its own share of the sun.

What would be wrong and invisible: equipment on a slot nobody wears (no insulation to sit on), a
belt that took the trousers' colour of sun, armour that insulates nothing, and -- the one the row
itself got wrong -- tape authored as a long-wave mirror. Glass-bead retroreflective sheeting is
glass and polymer at its surface and is emissive at 8-14 µm (ADR 0196).

docs/physics-model.md §4.2, §6.1, §16.2; roadmap HU.7; ADR 0195, ADR 0196.
"""

from __future__ import annotations

import pathlib

import pytest
import yaml

pytest.importorskip("jos3")

from irsim.config.humans import HumanSpec, load_body_schema  # noqa: E402
from irsim.config.scene import SceneConfig  # noqa: E402
from irsim.materials.library import MaterialLibrary  # noqa: E402
from irsim.materials.mapping import load_asset_mapping  # noqa: E402
from irsim.scene import Scene  # noqa: E402
from irsim.thermal.human import CLO_M2K_W, clothing_temperature_c  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[2]
PHENO = {"sex": "male", "age_y": 35, "height_m": 1.72, "mass_kg": 70}


def _scene(stem: str) -> Scene:
    with open(REPO / "configs" / "scenes" / f"{stem}.yaml", encoding="utf-8") as fh:
        return Scene.from_config(SceneConfig.model_validate(yaml.safe_load(fh)))


# --- the schema --------------------------------------------------------------------------------


def test_equipment_sits_on_a_slot_the_person_wears() -> None:
    schema = load_body_schema()
    ok = HumanSpec(
        phenotype=PHENO,
        garments={"legs": {"material": "wool_clothing", "clo": 0.28}},
        equipment={"belt": {"material": "leather", "on": "legs"}},
    )
    ok.check_against(schema)
    assert ok.materials == {"wool_clothing", "leather"}
    bad = HumanSpec(phenotype=PHENO, equipment={"belt": {"material": "leather", "on": "legs"}})
    with pytest.raises(ValueError, match="no garment on"):
        bad.check_against(schema)


# --- the tape ----------------------------------------------------------------------------------


def test_the_tape_is_emissive_in_the_long_wave_and_bright_in_the_sun() -> None:
    """Not a mirror (the row's premise, corrected in ADR 0196): glass and polymer at 8-14 µm."""
    tape = MaterialLibrary.load()["retroreflective_tape"].spec
    eps = tape.optical.emissivity_per_band
    assert eps is not None
    assert eps["lwir"] >= 0.85 and eps["mwir"] >= 0.8
    assert eps["nir"] <= 0.5, "silver in the sun's bands"
    assert tape.thermal.solar_absorptivity < 0.5
    assert "ESTIMATED" in tape.reference


# --- the officer -------------------------------------------------------------------------------


def test_the_officer_is_garments_and_equipment_with_a_material_each() -> None:
    asset = load_asset_mapping("police_officer", known_materials=MaterialLibrary.load().names)
    h = asset.human
    assert h is not None and asset.parts is not None
    assert set(h.garments) == {"head", "torso", "arms", "legs", "feet"}
    assert set(h.equipment) == {"tape", "belt"}
    assert h.coverage(load_body_schema(), "torso") == {"Chest", "Back"}
    names = {p.name for p in asset.parts.parts}
    for item in h.equipment:
        assert f"equipment_{item}" in names
        assert asset.materials[f"equipment_{item}"] == h.equipment[item].material
    for slot in h.garments:
        assert asset.materials[f"garment_{slot}"] == h.garments[slot].material


def test_in_winter_the_armoured_torso_sits_nearer_the_air_than_the_sleeves() -> None:
    """ISO 7730's figure for each garment's clo on the JOS-3 skin beneath it, at -5 °C."""
    scene = _scene("police_winter_night")
    temps = scene.advance_targets(0.0, 30.0)
    body = scene.targets["officer"]
    air = scene.weather_at(0.0).t_air_k
    torso, arms = temps["officer.garment_torso"], temps["officer.garment_arms"]
    assert air < torso < arms, (torso - 273.15, arms - 273.15)
    # the figure: ISO 7730 with each garment's own clo and its own skin, by hand
    h = body.spec
    f = body.last_forcing
    schema = load_body_schema()
    areas = {s.name: s.area_m2 for s in schema.segments}
    skin = dict(zip([s.name for s in schema.segments], body.skin_k() - 273.15, strict=True))
    for slot, got in (("torso", torso), ("arms", arms)):
        cov = sorted(h.coverage(schema, slot))
        t_sk = sum(areas[s] * skin[s] for s in cov) / sum(areas[s] for s in cov)
        want = clothing_temperature_c(
            f.tdb_c, f.tr_longwave_c, h.garments[slot].clo * CLO_M2K_W,
            metabolic_w_m2=body.activity_met * 58.15, air_speed_m_s=f.v_m_s, t_skin_c=t_sk,
        )  # fmt: skip
        assert got - 273.15 == pytest.approx(want, abs=1e-6), slot


def test_at_night_the_tape_and_the_vest_are_one_temperature_and_at_noon_they_are_not() -> None:
    """The tape sits on the vest's insulation; only its share of the sun sets it apart."""
    night = _scene("police_winter_night").advance_targets(0.0, 30.0)
    assert night["officer.equipment_tape"] == pytest.approx(
        night["officer.garment_torso"], abs=1e-9
    )
    noon = _scene("police_summer_noon").advance_targets(0.0, 30.0)
    assert abs(noon["officer.equipment_tape"] - noon["officer.garment_torso"]) > 0.05
    # the belt is the trousers' insulation with leather's sun
    assert noon["officer.equipment_belt"] != pytest.approx(noon["officer.garment_legs"], abs=0.05)
