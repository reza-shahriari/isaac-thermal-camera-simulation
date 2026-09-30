"""IG.24: a time-lapse camera still integrates each frame for the detector's own frame.

A driver that captures once every N seconds of scene time (ADR 0074) sets ``frame_period_s`` to
N. Two things in `IrCamera` read that number as if it were the detector's window: IG.19's
companion exposure, which posed its sub-frames across the whole capture interval -- a Phantom 4
time-lapse at 17 s a frame would have averaged the aircraft over 17 s of flight -- and the
synthesised ``motion_px`` (IG.6), which is the displacement between two captures and was smeared
whole. The detector integrates for 1/60 s either way; `rotor_veils` already said so. These tests
hold both to the detector's frame, and hold a real-time camera to exactly what it was.
"""

from __future__ import annotations

import pathlib
from types import SimpleNamespace

import numpy as np
import pytest

from irsim.config.loader import load_sensor_config
from irsim.optics.exposure import exposure_window_s
from irsim_isaac.pipeline.ir_camera import IrCamera, capture_motion_scale, detector_period_s

REPO = pathlib.Path(__file__).resolve().parents[2]
BOSON = load_sensor_config(REPO / "configs" / "sensors" / "flir_boson_640_lwir.yaml")
INSB = load_sensor_config(REPO / "configs" / "sensors" / "example_mwir_insb_640.yaml")


def _offsets_posed(sensor, frame_period_s: float, n: int) -> list[float]:  # type: ignore[no-untyped-def]
    """Drive the real `_exposed_companion` on a stand-in camera and record where it posed."""
    posed: list[float] = []
    reader = SimpleNamespace(
        step=lambda frames, rt_subframes: None,
        read=lambda: SimpleNamespace(rgb=np.zeros((4, 4, 3), dtype=np.uint8)),
    )
    fake = SimpleNamespace(
        sensor=sensor,
        frame_period_s=frame_period_s,
        _reader=reader,
        _native_rgb=lambda rgb: rgb,
    )
    IrCamera._exposed_companion(fake, 1, n, posed.append)  # type: ignore[arg-type]
    assert posed[-1] == 0.0, "the pose is restored after the exposure"
    return posed[:-1]


@pytest.mark.parametrize("config", [BOSON, INSB], ids=["bolometer", "photon"])
def test_the_companion_is_exposed_over_the_detector_window_not_the_capture_interval(
    config,  # type: ignore[no-untyped-def]
) -> None:
    sensor = config.sensor
    detector = detector_period_s(sensor)
    window = exposure_window_s(sensor, detector)
    real_time = _offsets_posed(config, detector, 8)
    time_lapse = _offsets_posed(config, 17.35, 8)
    assert time_lapse == real_time, "a time-lapse must not widen the exposure"
    spread = max(real_time) - min(real_time)
    assert spread == pytest.approx(window * 7 / 8)  # midpoints of 8 slices of the window
    assert spread < 0.02  # 1/60 s at most, not seconds


def test_motion_between_captures_is_scaled_to_one_detector_frame() -> None:
    detector = detector_period_s(BOSON.sensor)
    assert detector == pytest.approx(1.0 / 60.0)
    assert capture_motion_scale(detector, detector) == 1.0, "a real-time camera is unchanged"
    # a 17 s time-lapse step is 1020 detector frames; the smear is one of them
    assert capture_motion_scale(17.0, detector) == pytest.approx(1.0 / 1020.0)
    assert capture_motion_scale(0.001, detector) == 1.0, "never scaled up"
    with pytest.raises(ValueError):
        capture_motion_scale(0.0, detector)
