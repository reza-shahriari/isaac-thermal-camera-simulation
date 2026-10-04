"""An asset's ``rotate_deg`` reaches every Blender pass, or none.

Some artists pose a model in flight: the DJI FPV glTF arrives pitched 36 deg nose down. The library
holds it level, because a scene sets the attitude and every part selector (``near_xy``, heights)
is a position in the asset's frame. The turn runs inside Blender, so what these tests pin
engine-free is the argv contract: the part split indexes faces by the order the component pass
saw them in, and its selectors were measured in the turned frame, so a pass that turns while
another does not would hand every part the wrong faces, silently.
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


def _every_pass(prep, rotate_deg: tuple[float, float, float]) -> list[list[str]]:
    src = pathlib.Path("x.glb")
    out = pathlib.Path("data/assets/x/x.usdc")
    r = rotate_deg
    return [
        prep.blender_command("blender", src, out, out.with_suffix(".json"), 1.0, rotate_deg=r),
        prep.blender_split_command("blender", src, 1.0, out, out, out, rotate_deg=r),
        prep.blender_material_slots_command("blender", src, 1.0, out, rotate_deg=r),
        prep.blender_material_split_command("blender", src, 1.0, out, out, rotate_deg=r),
    ]


def test_every_pass_turns_alike(prep) -> None:
    for cmd in _every_pass(prep, (-36.0, 0.0, 90.0)):
        i = cmd.index("--rotate-deg")
        # all three angles, in X, Y, Z order -- a dropped one would turn about the wrong axis
        assert [float(v) for v in cmd[i + 1 : i + 4]] == [-36.0, 0.0, 90.0]
        assert i > cmd.index("--")


def test_an_unturned_asset_keeps_its_argv(prep) -> None:
    for cmd in _every_pass(prep, (0.0, 0.0, 0.0)):
        assert "--rotate-deg" not in cmd


def test_the_config_field_defaults_to_no_turn_and_takes_three_angles() -> None:
    assert AssetMapping(name="x").rotate_deg == (0.0, 0.0, 0.0)
    assert AssetMapping(name="x", rotate_deg=(-36, 0, 0)).rotate_deg == (-36.0, 0.0, 0.0)
    with pytest.raises(ValidationError):
        AssetMapping(name="x", rotate_deg=(-36.0, 0.0))
