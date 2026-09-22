"""The multi-band driver's bookkeeping (M10.24).

`scripts/render_multiband.py` runs three demo scenes x four bands as child processes, so almost
all of it needs Isaac Sim and a GPU. Two parts do not, and both are places where a partial run
silently misrepresents a complete output tree -- which is the failure mode this driver exists to
prevent, since its whole purpose is that the twelve renders are compared against one another.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import pathlib
import subprocess
from types import SimpleNamespace

import pytest

SCRIPT = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "render_multiband.py"

SCRIPTS = SCRIPT.resolve().parent
SCENE_CONFIGS = SCRIPT.resolve().parents[1] / "configs" / "scenes"


def _add_argument_calls(script: pathlib.Path):
    """Every ``parser.add_argument(...)`` in a driver, as AST nodes."""
    for node in ast.walk(ast.parse(script.read_text("utf-8"))):
        if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "add_argument":
            yield node


def _option_strings(script: pathlib.Path) -> set[str]:
    """Every ``--flag`` a driver's parser accepts."""
    return {
        str(a.value)
        for node in _add_argument_calls(script)
        for a in node.args
        if isinstance(a, ast.Constant) and str(a.value).startswith("--")
    }


def _fps_default(script: pathlib.Path) -> float | None:
    for node in _add_argument_calls(script):
        named = [a.value for a in node.args if isinstance(a, ast.Constant)]
        if "--fps" not in named:
            continue
        for keyword in node.keywords:
            if keyword.arg == "default" and isinstance(keyword.value, ast.Constant):
                return float(keyword.value.value)
    return None


def _scene_default(script: pathlib.Path) -> str | None:
    """The scene config a driver films when nobody passes ``--scene``."""
    for node in _add_argument_calls(script):
        named = [a.value for a in node.args if isinstance(a, ast.Constant)]
        if "--scene" not in named:
            continue
        for keyword in node.keywords:
            if keyword.arg == "default":
                # ``str(REPO / "configs/scenes/x.yaml")`` -- the filename is the only literal.
                for inner in ast.walk(keyword.value):
                    if isinstance(inner, ast.Constant) and str(inner.value).endswith(".yaml"):
                        return pathlib.PurePosixPath(str(inner.value)).name
    return None


def _filmed_scene(driver, entry) -> str | None:
    """Which scene config a `SCENES` row actually renders: its override, else the driver's own."""
    script, extra, _frames = entry
    if "--scene" in extra:
        return pathlib.PurePosixPath(extra[extra.index("--scene") + 1]).name
    return _scene_default(SCRIPTS / script)


@pytest.fixture(scope="module")
def driver():
    spec = importlib.util.spec_from_file_location("render_multiband", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write(root: pathlib.Path, payload: dict) -> None:
    (root / "index.json").write_text(json.dumps(payload), "utf-8")


def test_a_partial_rerun_keeps_the_scenes_it_did_not_touch(driver, tmp_path):
    """The defect: re-rendering one scene left an index claiming the other two did not exist."""
    _write(
        tmp_path,
        {
            "renders": {"drone/lwir": "ok", "airplane/lwir": "ok", "ship/lwir": "ok"},
            "contact_sheets": {"drone": "/d.png", "airplane": "/a.png", "ship": "/s.png"},
        },
    )
    merged = driver.merged_index(tmp_path, {"ship/lwir": "ok"}, {"ship": "/s2.png"})
    assert merged["renders"] == {"drone/lwir": "ok", "airplane/lwir": "ok", "ship/lwir": "ok"}
    assert merged["contact_sheets"]["drone"] == "/d.png"
    assert merged["contact_sheets"]["airplane"] == "/a.png"


def test_this_run_wins_where_the_two_overlap(driver, tmp_path):
    """A scene that rendered last time and failed this time must read failed, not ok.

    This is the half of the merge that a plain `previous | fresh` gets right and a plain
    `fresh | previous` gets exactly backwards -- a stale success outliving the failure that
    replaced it is worse than no manifest at all.
    """
    _write(tmp_path, {"renders": {"ship/mwir": "ok"}, "contact_sheets": {"ship": "/old.png"}})
    merged = driver.merged_index(tmp_path, {"ship/mwir": "failed"}, {"ship": "/new.png"})
    assert merged["renders"]["ship/mwir"] == "failed"
    assert merged["contact_sheets"]["ship"] == "/new.png"


def test_no_index_yet_is_an_empty_one(driver, tmp_path):
    merged = driver.merged_index(tmp_path, {"drone/nir": "ok"}, {})
    assert merged == {"renders": {"drone/nir": "ok"}, "contact_sheets": {}}


@pytest.mark.parametrize("payload", ["{not json", "[]", '"a string"', "null"])
def test_an_unreadable_index_does_not_lose_this_run(driver, tmp_path, payload):
    """A convenience file any run can rebuild must never be able to discard a completed render."""
    (tmp_path / "index.json").write_text(payload, "utf-8")
    merged = driver.merged_index(tmp_path, {"drone/swir": "ok"}, {"drone": "/d.png"})
    assert merged["renders"] == {"drone/swir": "ok"}
    assert merged["contact_sheets"] == {"drone": "/d.png"}


def test_the_merge_does_not_mutate_the_arguments(driver, tmp_path):
    _write(tmp_path, {"renders": {"ship/lwir": "ok"}, "contact_sheets": {}})
    status = {"drone/lwir": "ok"}
    driver.merged_index(tmp_path, status, {})
    assert status == {"drone/lwir": "ok"}


def test_every_scene_is_long_enough_to_watch(driver):
    """Each scene is encoded to video, so each count is a clip length.

    The ship's default was 8 frames -- a quarter of a second at 30 fps -- because it was written
    when the maritime script exported loose per-frame files and encoded nothing. The scenes need
    not agree on a count (the aerial ones fly, the maritime camera is static, and the car scenes
    are hour-long time-lapses played at 10 fps), but a clip that cannot be watched is a bug in the
    sweep rather than a choice about the scene.

    The bar is **seconds**, not frames, and it has to be: the old frame-count form read 90 as
    "three seconds" while silently assuming 30 fps, which the car driver has never used. Each
    count is divided by that driver's own `--fps` default (IG.13).
    """
    for scene, (script, _args, frames) in driver.SCENES.items():
        fps = _fps_default(SCRIPTS / script)
        assert fps is not None, f"{script} declares no --fps default"
        seconds = frames / fps
        assert seconds >= 3.0, (
            f"{scene} renders {frames} frames at {fps:g} fps -- a {seconds:.1f} s clip"
        )


def test_every_band_names_a_config_that_exists(driver):
    configs = SCRIPT.resolve().parents[1] / "configs" / "sensors"
    for band, (sensor, _args) in driver.BANDS.items():
        assert (configs / sensor).is_file(), f"{band} names a missing sensor config: {sensor}"


# --- IG.13: the sweep reaches every scene, and can actually invoke every driver ----------------


def test_every_scene_config_is_either_swept_or_accounted_for(driver):
    """The sweep covered three of eight scene configs, and nothing said which five it missed.

    A scene config that no sweep renders is a scene that has only ever been seen in the one band
    its own driver defaults to -- which for five of the eight, including the whole aerial
    point-target lane, was the case. The exclusion list is the honest half: a scene may sit
    outside the sweep, but it has to say why, in the file, where the next person will read it.
    """
    present = {p.name for p in SCENE_CONFIGS.glob("*.yaml")}
    assert present, "no scene configs found; the test is pointed at the wrong directory"

    filmed = {_filmed_scene(driver, entry) for entry in driver.SCENES.values()}
    assert None not in filmed, "a SCENES row names a driver with no resolvable --scene default"

    unaccounted = present - filmed - set(driver.UNSWEPT_SCENES)
    assert not unaccounted, (
        "these scene configs are rendered by no sweep and are not listed in UNSWEPT_SCENES: "
        + ", ".join(sorted(unaccounted))
    )


def test_the_exclusion_list_names_real_files_and_gives_a_reason(driver):
    """An exclusion list that drifts is worse than none: it reads as a decision that was made."""
    present = {p.name for p in SCENE_CONFIGS.glob("*.yaml")}
    for name, reason in driver.UNSWEPT_SCENES.items():
        assert name in present, f"UNSWEPT_SCENES names {name}, which no longer exists"
        assert name not in {_filmed_scene(driver, e) for e in driver.SCENES.values()}, (
            f"{name} is both swept and listed as unswept"
        )
        assert len(reason) > 60, f"{name} is excluded without a reason anyone can check"


def test_the_sweep_only_passes_flags_every_driver_accepts(driver, monkeypatch, tmp_path):
    """`render()` builds one command template for six different parsers.

    `--rt-subframes` was passed unconditionally and two drivers did not accept it, so adding
    either to `SCENES` would have made argparse reject every invocation -- twelve renders failing
    at once, for a reason visible only in a child process's stderr. This builds the real command
    for every scene and checks each flag against the target parser, without running anything.
    """
    commands = []

    def _capture(command, **_kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(subprocess, "run", _capture)
    for scene in driver.SCENES:
        for band in driver.BANDS:
            driver.render(scene, band, tmp_path, None, 4)

    assert len(commands) == len(driver.SCENES) * len(driver.BANDS)
    for command in commands:
        script = pathlib.Path(command[1])
        accepted = _option_strings(script)
        passed = {token for token in command[2:] if token.startswith("--")}
        assert passed <= accepted, (
            f"{script.name} does not accept "
            + ", ".join(sorted(passed - accepted))
            + " -- the sweep passes it to every driver"
        )


# --- IG.13: a driver the sweep runs in four bands must be able to run in four bands -----------


def _scene_from_file_calls(script: pathlib.Path):
    """Every ``Scene.from_file(...)`` in a driver, as AST nodes."""
    for node in ast.walk(ast.parse(script.read_text("utf-8"))):
        if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "from_file":
            yield node


def test_every_swept_driver_builds_its_scene_in_the_sensors_own_quantity(driver):
    """A sky model built in the wrong radiometric form refuses the render.

    `Scene.from_file` defaults to ``quantity="lb"`` -- band radiance, which is what a bolometer
    integrates. A photon FPA runs on ``lb_q``, and `PipelineConfig.from_sensor` compares the two
    and raises rather than mixing them. Three of the six drivers never passed the sensor's own
    quantity through, so each worked in the one band its `--sensor` default names and failed in
    the other three with `sky model built in the 'lb' form`. The aerial point-target scene --
    the lane the owner ranked first -- was among them, and the sweep found it by trying: 3 of 16
    renders died at startup.

    Checked by AST rather than by running the drivers, because every one of them boots Kit.
    """
    for scene, (script, _args, _frames) in driver.SCENES.items():
        path = SCRIPTS / script
        calls = list(_scene_from_file_calls(path))
        assert calls, f"{script} builds no Scene; the check is pointed at the wrong thing"
        for call in calls:
            named = {k.arg for k in call.keywords}
            assert "quantity" in named, (
                f"{scene}: {script} line {call.lineno} builds a Scene without the sensor's "
                "quantity, so it runs in one band and raises in the other three"
            )
