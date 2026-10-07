"""HU.11 -- a firefighter: turnout gear whose insulation is a manikin's, trim, and a cylinder.

The turnout's insulation is not a guess per slot. Kuklane et al. 2022 (Biology 11:1813,
doi:10.3390/biology11121813) measured Dutch structural firefighter ensembles zone by zone on a
thermal manikin at 0.18 m/s; for ensemble C6 (turnout coat and trousers over station wear, no
breathing set) the local total insulation, and the nude manikin's air layer in each zone, are in
their Table A1. The intrinsic insulation of a zone is the difference (ESTIMATED: it neglects the
clothing area factor). The test re-derives each slot's clo from those numbers, so an edited asset
that drifts from the source fails.

What else would be wrong and plausible: trim whose emissivity made it a long-wave mirror (ADR 0196
says it is not), a cylinder that did not take its own sun, a turnout that insulated like a shirt.

docs/physics-model.md §6.1, §16.2; roadmap HU.11; ADR 0196, ADR 0198.
"""

from __future__ import annotations

import pathlib

import pytest
import yaml

pytest.importorskip("jos3")

from irsim.config.scene import SceneConfig  # noqa: E402
from irsim.materials.library import MaterialLibrary  # noqa: E402
from irsim.materials.mapping import load_asset_mapping  # noqa: E402
from irsim.scene import Scene  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[2]
CLO = 0.155  # m2 K / W
#: Kuklane et al. 2022, Table A1: local total insulation I_T,i of ensemble C6 and of the nude
#: manikin (the air layer), m2 K / W.
C6 = {"chest": 0.730, "back": 0.529, "arms": 0.506, "hands": 0.259, "legs": 0.514, "feet": 0.345}
NUDE = {"chest": 0.113, "back": 0.097, "arms": 0.113, "hands": 0.101, "legs": 0.090, "feet": 0.108}


def _intrinsic_clo(*zones: str) -> float:
    return sum(C6[z] - NUDE[z] for z in zones) / len(zones) / CLO


def _scene(stem: str) -> Scene:
    with open(REPO / "configs" / "scenes" / f"{stem}.yaml", encoding="utf-8") as fh:
        return Scene.from_config(SceneConfig.model_validate(yaml.safe_load(fh)))


def test_the_turnout_insulation_is_the_manikins() -> None:
    h = load_asset_mapping("firefighter").human
    assert h is not None
    expected = {
        "torso": _intrinsic_clo("chest", "back"),
        "arms": _intrinsic_clo("arms"),
        "legs": _intrinsic_clo("legs"),
        "hands": _intrinsic_clo("hands"),
        "feet": _intrinsic_clo("feet"),
    }
    for slot, clo in expected.items():
        assert h.garments[slot].clo == pytest.approx(clo, abs=0.01), slot
    # a turnout is not a shirt: more than ten times a long-sleeve shirt's 0.25 on the arms
    assert h.garments["arms"].clo > 10 * 0.25


def test_the_shell_and_the_trim_are_emissive_in_the_long_wave() -> None:
    lib = MaterialLibrary.load()
    shell = lib["turnout_aramid_shell"].spec.optical.emissivity_per_band
    assert shell is not None
    # NISTIR 6299 Table 1: tau 0.044, r 0.09 against a 943 K source -> 0.866 absorbed
    assert shell["mwir"] == pytest.approx(1.0 - 0.044 - 0.09, abs=0.005)
    assert shell["lwir"] >= 0.85
    trim = lib["retroreflective_tape"].spec.optical.emissivity_per_band
    assert trim is not None and trim["lwir"] > 0.8


def test_at_night_the_trim_and_cylinder_are_the_coat_and_at_noon_the_sun_parts_them() -> None:
    night = _scene("firefighter_winter_night")
    t = night.advance_targets(0.0, 30.0)
    coat, face = t["firefighter.garment_torso"], t["firefighter.skin_Head"]
    assert night.weather_at(0.0).t_air_k < coat < face - 10.0
    for item in ("trim_coat", "cylinder"):
        assert t[f"firefighter.equipment_{item}"] == pytest.approx(coat, abs=1e-9), item
    noon = _scene("firefighter_summer_noon").advance_targets(0.0, 30.0)
    # the carbon cylinder (alpha 0.90) takes more sun than the silver-and-lime trim (0.35)
    assert noon["firefighter.equipment_cylinder"] > noon["firefighter.equipment_trim_coat"] + 1.0
