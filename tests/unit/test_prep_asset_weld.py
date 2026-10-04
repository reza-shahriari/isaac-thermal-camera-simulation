"""An asset's ``weld_m`` reaches every Blender pass, or none (AI.18).

glTF splits a vertex at every UV and normal seam. Welded, the Liberty ship is 1,592 pieces;
unwelded, 11,953 shells, each a thermal island, since conduction crosses only shared edges
(ADR 0112). The weld runs inside Blender, so what these tests can pin engine-free is the argv
contract: the part split indexes faces by the order the component pass saw them in, so a pass
that welds while another does not regroups the wrong faces into every part, silently. Each
builder therefore has to carry the flag, and an asset that does not weld keeps its argv as before.
"""

from __future__ import annotations

import importlib.util
import pathlib

import pytest
from pydantic import ValidationError

from irsim.materials.mapping import AssetMapping

REPO = pathlib.Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def prep():
    spec = importlib.util.spec_from_file_location("prep_asset", REPO / "scripts" / "prep_asset.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _every_pass(prep, weld_m: float) -> list[list[str]]:
    src = pathlib.Path("x.glb")
    out = pathlib.Path("data/assets/x/x.usdc")
    return [
        prep.blender_command("blender", src, out, out.with_suffix(".json"), 1.0, weld_m=weld_m),
        prep.blender_split_command("blender", src, 1.0, out, out, out, weld_m=weld_m),
        prep.blender_material_slots_command("blender", src, 1.0, out, weld_m),
        prep.blender_material_split_command("blender", src, 1.0, out, out, weld_m=weld_m),
    ]


def test_every_pass_welds_alike(prep) -> None:
    for cmd in _every_pass(prep, 0.001):
        assert "--weld-m" in cmd, cmd
        i = cmd.index("--weld-m")
        assert float(cmd[i + 1]) == 0.001
        # a worker argument, after the separator Blender stops parsing at
        assert i > cmd.index("--")


def test_an_asset_that_does_not_weld_keeps_its_argv(prep) -> None:
    for cmd in _every_pass(prep, 0.0):
        assert "--weld-m" not in cmd


def test_the_weld_distance_keeps_its_precision(prep) -> None:
    (cmd, *_) = _every_pass(prep, 0.000_123_456_789)
    assert float(cmd[cmd.index("--weld-m") + 1]) == 0.000_123_456_789


def test_the_config_field_defaults_off_and_refuses_a_negative_distance() -> None:
    assert AssetMapping(name="x").weld_m == 0.0
    assert AssetMapping(name="x", weld_m=0.001).weld_m == 0.001
    with pytest.raises(ValidationError):
        AssetMapping(name="x", weld_m=-0.001)
