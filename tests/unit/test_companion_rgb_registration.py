"""IG.9: the companion RGB is refused unless it is the grid the G-buffer was read on.

The RGB/IR pair's whole claim is that it is registered by construction rather than by
calibration: the colour AOV and the geometry AOVs come off one render product, and the colour is
box-filtered by the supersample factor onto exactly the detector's pixels. That claim held only
by assumption. ``rgb`` is not a *required* channel, so the reader's shape guard never saw it, and
``_native_rgb`` cropped whatever arrived to a multiple of ``k`` and averaged it -- a colour buffer
at half the render product's resolution (which several colour AOVs on this build return) became a
frame of the right size in which every visible pixel sat on the wrong infrared pixel. Nothing
about the picture would say so.

These run without Isaac Sim: the method under test is pure NumPy on an array.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from irsim.pipeline.core import PipelineConfig
from irsim.scene import Scene
from irsim_isaac.pipeline.ir_camera import IrCamera

REPO = pathlib.Path(__file__).resolve().parents[2]
SCENE_YAML = REPO / "configs" / "scenes" / "sky_target_clear_day.yaml"


@pytest.fixture(scope="module")
def camera(tophat_lwir_lut, aerial_materials, aerial_sensor):  # type: ignore[no-untyped-def]
    # the shared aerial camera, rendered at 3x: a supersample of 1 makes the box the identity
    sensor = aerial_sensor.model_copy(deep=True)
    raw = sensor.model_dump()
    raw["sensor"]["optics"]["supersample_factor"] = 3
    sensor = type(aerial_sensor).model_validate(raw)
    scene = Scene.from_file(SCENE_YAML, {"lwir": tophat_lwir_lut})
    pipeline = PipelineConfig.from_sensor(sensor, aerial_materials, tophat_lwir_lut)
    return IrCamera(
        sensor,
        scene,
        pipeline=pipeline,
        prim_to_target={},
        resolutions=[],
        capture_rgb=True,
    )


def _product(camera: IrCamera, scale: float = 1.0) -> np.ndarray:
    """A colour buffer at ``scale`` x the render product, with a unique value per k x k block."""
    width, height = camera.optics.resolution
    h, w = int(height * scale), int(width * scale)
    rng = np.random.default_rng(9)
    return rng.integers(1, 255, (h, w, 4), dtype=np.uint8)


def test_a_full_resolution_buffer_lands_on_the_detector_grid(camera: IrCamera) -> None:
    k = camera.config.supersample
    assert k > 1, "the fixture must supersample, or the box is the identity and proves nothing"
    buf = _product(camera)
    out = camera._native_rgb(buf)
    fpa = camera.sensor.sensor.fpa
    assert out is not None and out.shape == (fpa.height, fpa.width, 4)
    # each native pixel is exactly the mean of its own k x k block of the product
    for y, x in ((0, 0), (fpa.height - 1, fpa.width - 1), (fpa.height // 2, 3)):
        block = buf[y * k : (y + 1) * k, x * k : (x + 1) * k].astype(np.float64)
        np.testing.assert_array_equal(out[y, x], np.rint(block.mean(axis=(0, 1))).astype(np.uint8))
    assert camera.rgb_problem is None


@pytest.mark.parametrize("scale", [0.5, 2.0])
def test_a_buffer_of_another_size_is_refused_not_resampled(camera: IrCamera, scale: float) -> None:
    """Half resolution is the measured failure; double is the same mistake the other way."""
    with pytest.raises(ValueError, match="IG.9"):
        camera._native_rgb(_product(camera, scale))


def test_one_row_short_is_refused_too(camera: IrCamera) -> None:
    """The old crop to a multiple of k hid exactly this: a buffer one row short still 'fit'."""
    with pytest.raises(ValueError, match="render product"):
        camera._native_rgb(_product(camera)[:-1])


def test_an_absent_or_black_buffer_is_still_a_reported_absence(camera: IrCamera) -> None:
    """A missing frame is not a wrong frame: it is reported, not raised, as before."""
    assert camera._native_rgb(None) is None
    assert camera.rgb_problem is not None and "no data" in camera.rgb_problem
    black = np.zeros_like(_product(camera))
    assert camera._native_rgb(black) is None
    assert camera.rgb_problem is not None and "all-zero" in camera.rgb_problem
