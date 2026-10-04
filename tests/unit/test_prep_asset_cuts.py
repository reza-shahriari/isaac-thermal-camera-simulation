"""An asset's ``cuts`` reach every Blender pass, or none.

A part is claimed piece by piece (ADR 0138), so one continuous moulding is one part however much
of the aircraft it spans: the DJI Avata 2's upper shell carries all four duct rings. ``cuts``
split such a piece along planes inside Blender, so what these tests can pin engine-free is the
contract around that: the part split indexes faces by the order the component pass saw them
in, so a pass that cuts while another does not would hand every part the wrong faces; the
component cache must recognise the cuts it was measured under; and a cut must say where it is.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib

import pytest
from pydantic import ValidationError

from irsim.materials.mapping import AssetMapping, CutSpec

REPO = pathlib.Path(__file__).resolve().parents[2]

CUT = {
    "point": [0.0, -0.0237, 0.0],
    "normal": [0.0, 1.0, 0.0],
    "materials": ["shell"],
    "x_min": 0.03,
}


@pytest.fixture(scope="module")
def prep():
    spec = importlib.util.spec_from_file_location("prep_asset", REPO / "scripts" / "prep_asset.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _every_pass(prep, cuts) -> list[list[str]]:
    src = pathlib.Path("x.glb")
    out = pathlib.Path("data/assets/x/x.usdc")
    return [
        prep.blender_command("blender", src, out, out.with_suffix(".json"), 1.0, cuts=cuts),
        prep.blender_split_command("blender", src, 1.0, out, out, out, cuts=cuts),
        prep.blender_material_slots_command("blender", src, 1.0, out, cuts=cuts),
        prep.blender_material_split_command("blender", src, 1.0, out, out, cuts=cuts),
    ]


def test_every_pass_cuts_alike(prep) -> None:
    cut = CutSpec(**CUT).model_dump(mode="json")
    for cmd in _every_pass(prep, [cut]):
        i = cmd.index("--cuts-json")
        assert json.loads(cmd[i + 1]) == [cut]  # the whole spec, filters included
        assert i > cmd.index("--")


def test_an_asset_that_cuts_nothing_keeps_its_argv(prep) -> None:
    for cmd in _every_pass(prep, []):
        assert "--cuts-json" not in cmd


def test_the_driver_hands_the_worker_the_configs_cuts_in_order(prep) -> None:
    a = AssetMapping(name="x", cuts=[CUT, {**CUT, "x_min": None, "x_max": -0.03}])
    dicts = prep._cut_dicts(a)
    # what the component cache records is JSON; a tuple here would never compare equal to it, and
    # every run would re-measure the components
    assert json.loads(json.dumps(dicts)) == dicts
    assert [d["x_min"] for d in dicts] == [0.03, None]  # order is meaning: each cuts what is left
    assert [d["x_max"] for d in dicts] == [None, -0.03]


def test_a_cut_defaults_to_none_and_needs_a_plane() -> None:
    assert AssetMapping(name="x").cuts == []
    with pytest.raises(ValidationError):
        CutSpec(point=(0.0, 0.0, 0.0), normal=(0.0, 0.0, 0.0))  # no direction
    with pytest.raises(ValidationError):
        CutSpec(point=(0.0, 0.0), normal=(1.0, 0.0, 0.0))  # not a point
    with pytest.raises(ValidationError):
        CutSpec(point=(0.0, 0.0, 0.0), normal=(1.0, 0.0, 0.0), area_min_m2=-1.0)
