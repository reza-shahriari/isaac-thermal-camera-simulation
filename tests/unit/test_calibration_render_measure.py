"""SC.33's in-engine round trip, the engine-free half: the measurement and the target.

`scripts/measure_calibration_render.py` decides whether a rendered frame was distorted by the
camera's own calibration, by undistorting it with the same numbers and asking whether the rows of
board corners come out straight. That verdict is only worth something if the measurement can
tell a right lens from a wrong one, so it is run here on a board this repository distorts itself
(`irsim.optics.calibration.distort_image`, the CPU builder's lens): bowed by several pixels as
rendered, straight to a fraction of a pixel through the right coefficients, and still bowed
through the wrong ones. The driver's checkerboard preset is held to the scene it heats.
"""

from __future__ import annotations

import importlib.util
import pathlib
from types import ModuleType

import numpy as np
import pytest

from irsim.config.loader import load_sensor_config
from irsim.config.scene import load_scene_config
from irsim.optics.calibration import distort_image
from irsim.optics.projection import Intrinsics

REPO = pathlib.Path(__file__).resolve().parents[2]
SENSOR = REPO / "configs" / "sensors" / "example_boson_640_wide_calibrated.yaml"
SCENE = REPO / "configs" / "scenes" / "calibration_checkerboard.yaml"


def _script(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _board(intr: Intrinsics) -> np.ndarray:
    """The 13 x 11 board of 120 mm squares, square-on at 1.1 m, through an ideal pinhole."""
    v, u = np.mgrid[0:512, 0:640] + 0.5
    x = (u - intr.cx_px) / intr.fx_px * 1.1
    z = (v - intr.cy_px) / intr.fy_px * 1.1
    i, j = np.floor(x / 0.12 + 6.5), np.floor(z / 0.12 + 5.5)
    inside = (i >= 0) & (i < 13) & (j >= 0) & (j < 11)
    return np.where(inside, np.where((i + j) % 2 == 0, 305.0, 293.0), 285.0).astype(np.float32)


def test_the_measurement_straightens_the_right_lens_and_not_a_wrong_one() -> None:
    pytest.importorskip("cv2")
    measure = _script("measure_calibration_render")
    spec = load_sensor_config(SENSOR).sensor
    assert spec.calibration is not None and spec.calibration.geometric is not None
    g = spec.calibration.geometric
    intr = Intrinsics.from_sensor(spec, 1)
    rendered = measure._to_u8(distort_image(_board(intr), intr, spec.effective_distortion))

    k = np.array([[g.fx_px, 0, g.cx_px - 0.5], [0, g.fy_px, g.cy_px - 0.5], [0, 0, 1.0]])
    right = measure._measure(rendered, k, np.array([g.k1, g.k2, g.p1, g.p2, g.k3]))
    assert right["bow_px"] > 5.0, "the lens must visibly bow the rows or the test proves nothing"
    assert right["bow_undistorted_px"] < 0.3
    assert right["bow_undistorted_px"] < right["bow_px"] / 20.0
    # half the barrel is a wrong lens: the rows must stay visibly bent
    wrong = measure._measure(rendered, k, np.array([g.k1 / 2, g.k2 / 2, 0.0, 0.0, 0.0]))
    assert wrong["bow_undistorted_px"] is not None and wrong["bow_undistorted_px"] > 2.0


def test_the_preset_is_the_board_the_scene_heats() -> None:
    config = load_scene_config(SCENE)
    nodes = {t.name for t in config.scene.targets}
    # the driver boots Kit at import, so its preset table is read from source, not imported
    source = (REPO / "scripts" / "render_patch_scene.py").read_text(encoding="utf-8")
    namespace: dict[str, object] = {}
    start = source.index("def _checkerboard(")
    end = source.index("\nparser = argparse.ArgumentParser")
    exec(  # noqa: S102 - the preset table is plain data built by one helper
        "from typing import Any\n" + source[start:end], namespace
    )
    preset = namespace["PRESETS"]["calibration_checkerboard"]  # type: ignore[index]
    squares = preset["parts"]
    assert len(squares) == 13 * 11
    assert {p[5] for p in squares} == nodes == {"warm_squares", "cool_squares"}
    # alternating: every horizontal neighbour is on the other node
    by_pos = {(round(p[2][0], 3), round(p[2][2], 3)): p[5] for p in squares}
    for (x, z), node in by_pos.items():
        right = by_pos.get((round(x + 0.12, 3), z))
        assert right is None or right != node
    # tiles, not overlaps: the board spans exactly 13 x 11 squares
    xs = sorted({x for x, _ in by_pos})
    assert xs[-1] - xs[0] == pytest.approx(12 * 0.12)
    assert preset["sensor"] == "configs/sensors/example_boson_640_wide_calibrated.yaml"
