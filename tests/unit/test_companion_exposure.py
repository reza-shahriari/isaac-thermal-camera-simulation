"""IG.19 -- the visible companion blurs over the same exposure as the infrared.

The infrared frame smears a moving edge over the detector's window with the SC.28 kernel
(`apply_motion_smear`); the companion is accumulated over the same window from sub-frames posed
along the motion (`irsim.optics.exposure`). On a step edge crossing the frame at three speeds,
under a bolometer's whole-period window and a photon FPA's short one, the two edges spread by
the same width to half a pixel; a static scene comes back bit-identical from either. ADR 0073,
ADR 0077, §8.3.
"""

from __future__ import annotations

import numpy as np
import pytest

from irsim.config.loader import load_sensor_config
from irsim.optics.exposure import accumulate_subframes, exposure_window_s, subframe_offsets_s
from irsim.optics.smear import apply_motion_smear

WIDTH, HEIGHT = 96, 8
EDGE_X = 40.0
LOW, HIGH = 0.2, 0.8


def _edge(shift_px: float) -> np.ndarray:
    """A vertical step edge at ``EDGE_X + shift_px``, antialiased by its sub-pixel coverage."""
    x = np.arange(WIDTH, dtype=np.float64)
    coverage = np.clip(EDGE_X + shift_px - x, 0.0, 1.0)  # fraction of pixel x left of the edge
    row = LOW + (HIGH - LOW) * (1.0 - coverage)
    return np.repeat(row[None, :], HEIGHT, axis=0).astype(np.float32)


def _width_10_90(profile: np.ndarray) -> float:
    """Distance between the 10 % and 90 % crossings of a rising profile, by interpolation."""
    p = (np.asarray(profile, dtype=np.float64) - LOW) / (HIGH - LOW)
    x = np.arange(p.size, dtype=np.float64)
    return float(np.interp(0.9, p, x) - np.interp(0.1, p, x))


def test_the_window_is_the_detector_s_and_the_offsets_are_centred() -> None:
    boson = load_sensor_config("boson640").sensor
    mwir = load_sensor_config("example_mwir_insb_640").sensor
    assert exposure_window_s(boson, 1.0 / 60.0) == pytest.approx(1.0 / 60.0)  # no shutter
    assert exposure_window_s(mwir, 1.0 / 30.0) == pytest.approx(mwir.fpa.integration_time_ms * 1e-3)
    assert exposure_window_s(mwir, 1e-6) == pytest.approx(1e-6), "never longer than the period"
    off = subframe_offsets_s(0.016, 8)
    assert off.size == 8 and off.mean() == pytest.approx(0.0, abs=1e-15)
    assert np.allclose(np.diff(off), 0.016 / 8) and off[0] == pytest.approx(-0.007)
    assert subframe_offsets_s(0.016, 1).tolist() == [0.0]
    assert np.all(subframe_offsets_s(0.0, 4) == 0.0)
    with pytest.raises(ValueError):
        subframe_offsets_s(0.016, 0)
    with pytest.raises(ValueError):
        exposure_window_s(boson, 0.0)


@pytest.mark.parametrize("duty", [1.0, 0.25])
@pytest.mark.parametrize("speed_px_per_frame", [3.0, 6.0, 12.0])
def test_a_moving_edge_spreads_the_same_in_both_bands(
    speed_px_per_frame: float, duty: float
) -> None:
    period = 1.0 / 30.0
    window = duty * period
    # infrared: the SC.28 kernel on the edge at the frame's instant
    motion = np.zeros((HEIGHT, WIDTH, 2))
    motion[..., 0] = speed_px_per_frame
    infrared = apply_motion_smear(_edge(0.0), motion, duty)
    # companion: sixteen sub-frames posed along the motion across the same window, averaged
    offsets = subframe_offsets_s(window, 16)
    companion = accumulate_subframes([_edge(speed_px_per_frame * dt / period) for dt in offsets])
    w_ir = _width_10_90(infrared[HEIGHT // 2])
    w_rgb = _width_10_90(companion[HEIGHT // 2])
    expected = 0.8 * speed_px_per_frame * duty  # a boxcar's 10-90 % width
    assert abs(w_rgb - w_ir) < 0.5, (speed_px_per_frame, duty, w_ir, w_rgb)
    if speed_px_per_frame * duty >= 3.0:  # a shorter smear is inside the edge's own width
        assert w_ir == pytest.approx(expected, abs=0.6) and w_rgb == pytest.approx(
            expected, abs=0.6
        )
    # the accumulated edge is centred on the frame's instant, as the kernel's is
    half_ir = float(np.interp(0.5, (infrared[HEIGHT // 2] - LOW) / (HIGH - LOW), np.arange(WIDTH)))
    half_rgb = float(
        np.interp(0.5, (companion[HEIGHT // 2] - LOW) / (HIGH - LOW), np.arange(WIDTH))
    )
    assert abs(half_ir - half_rgb) < 0.25, (half_ir, half_rgb)
    assert companion.dtype == np.float32


def test_a_static_scene_is_bit_identical_in_both_bands() -> None:
    still = _edge(0.0)
    motion = np.zeros((HEIGHT, WIDTH, 2))
    assert np.array_equal(apply_motion_smear(still, motion, 1.0), still.astype(np.float64))
    assert np.array_equal(accumulate_subframes([still] * 8), still)
    assert accumulate_subframes([still]) is not None
    rng = np.random.default_rng(3)
    noisy = rng.random((HEIGHT, WIDTH), dtype=np.float32)
    assert np.array_equal(accumulate_subframes([noisy, noisy.copy(), noisy]), noisy)
    with pytest.raises(TypeError, match="float16"):
        accumulate_subframes([still.astype(np.float16)] * 2)
    with pytest.raises(ValueError, match="shape"):
        accumulate_subframes([still, still[:, :10]])
