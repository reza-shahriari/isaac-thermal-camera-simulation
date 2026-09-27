"""The library's side artefacts ride the existing Blender passes (AI.9, ADR 0150).

The owner's 2026-09-26 format decision: a library entry is master `.blend` + the `.usdc` scenes
reference, with an FBX as an interchange courtesy. These tests pin the argv contract between the
driver and the worker — the whole interface between the two interpreters — engine-free, the same
way the other `blender_command` tests do: if a flag is dropped from the argv, the artefact
silently never appears, and nothing downstream would notice until the cloud upload is missing
files.
"""

from __future__ import annotations

import importlib.util
import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def prep():
    spec = importlib.util.spec_from_file_location("prep_asset", REPO / "scripts" / "prep_asset.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_plain_pass_carries_both_side_exports_when_asked(prep) -> None:
    cmd = prep.blender_command(
        "blender",
        pathlib.Path("3d_models/x/gltf/scene.gltf"),
        pathlib.Path("data/assets/x/x.usdc"),
        pathlib.Path("data/assets/x/x.prims.json"),
        1.0,
        save_blend=pathlib.Path("data/assets/x/x.blend"),
        emit_fbx=pathlib.Path("data/assets/x/x.fbx"),
    )
    assert "--save-blend" in cmd and cmd[cmd.index("--save-blend") + 1].endswith("x.blend")
    assert "--emit-fbx" in cmd and cmd[cmd.index("--emit-fbx") + 1].endswith("x.fbx")


def test_the_part_split_pass_carries_them_too(prep) -> None:
    # The part-split scene IS the cleaned master when a parts: block exists, so the .blend/.fbx
    # of a decomposed asset must come from this pass, not from the pre-split import.
    cmd = prep.blender_split_command(
        "blender",
        pathlib.Path("3d_models/x/gltf/scene.gltf"),
        1.0,
        pathlib.Path("data/assets/x/x.components.json"),
        pathlib.Path("data/assets/x/x.part_assignment.json"),
        pathlib.Path("data/assets/x_parts/x_parts.usdc"),
        save_blend=pathlib.Path("data/assets/x_parts/x_parts.blend"),
        emit_fbx=pathlib.Path("data/assets/x_parts/x_parts.fbx"),
    )
    assert "--save-blend" in cmd
    assert "--emit-fbx" in cmd


def test_without_the_flags_the_argv_is_unchanged(prep) -> None:
    # Backwards compatibility: every existing caller and committed transcript stays valid.
    cmd = prep.blender_command(
        "blender",
        pathlib.Path("s.gltf"),
        pathlib.Path("o.usdc"),
        pathlib.Path("o.prims.json"),
        0.01,
    )
    assert "--save-blend" not in cmd
    assert "--emit-fbx" not in cmd
