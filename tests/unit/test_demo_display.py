"""The demo clips show what the camera shows (`SC.24`, spec issue S52).

`render_phantom4.py` wrote two clips that were not the camera. Its `agc` clip rebuilt plateau
equalisation over 65 536 float bins of apparent temperature, none of which reached the plateau, so
the drone got 3 grey codes. Its `ir` ramp stopped at the target's 99th percentile, under the four
motor bells, so they clipped to one white. The fixes live in `irsim_eval.video`, shared with
`redisplay_planes.py`, and are pinned here: the span helper, the one number the AGC defect was
measured in, the readout margin, and a guard that no demo driver re-implements an AGC again.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from irsim_eval.video import (
    READOUT_MARGIN,
    apparent_target_span_k,
    interior,
    target_code_span,
    with_margin,
)

REPO = pathlib.Path(__file__).resolve().parents[2]


def _drone(seed: int = 0) -> np.ndarray:
    """Apparent temperatures of a phantom4-like target: a cold-reflecting shell and 4 hot bells.

    Shell −26 … +10 °C (ε < 1 reflecting a −45 °C sky), bells at +30 … +34 °C covering 0.5 % of
    the target -- the case where a percentile and a maximum disagree.
    """
    rng = np.random.default_rng(seed)
    shell = rng.uniform(247.0, 283.0, 9950)
    bells = np.array([303.15, 304.15, 306.15, 307.15]).repeat(12) + rng.normal(0, 0.05, 48)
    return np.concatenate([shell, bells, rng.uniform(283.0, 290.0, 2)]).astype(np.float32)


def test_the_span_top_is_the_hottest_pixel_so_the_bells_keep_their_order() -> None:
    pixels = _drone()
    lo, hi = apparent_target_span_k(pixels)
    assert hi == pytest.approx(float(pixels.max()))
    assert lo == pytest.approx(float(np.percentile(pixels, 1.0)))

    bells = [303.15, 304.15, 306.15, 307.15]
    old_hi = float(np.percentile(pixels, 99.0))  # the pre-SC.24 top
    assert old_hi < bells[0], "the red case: a 99th percentile sits under every bell"
    new_codes = [round(255 * (b - lo) / (hi - lo)) for b in bells]
    assert all(a < b for a, b in zip(new_codes, new_codes[1:], strict=False)), new_codes
    assert new_codes[-1] - new_codes[0] >= 10


def test_the_span_ignores_non_finite_pixels_and_refuses_an_empty_target() -> None:
    lo, hi = apparent_target_span_k(np.array([250.0, np.nan, 260.0, np.inf], np.float32))
    assert hi == pytest.approx(260.0)
    with pytest.raises(ValueError, match="finite"):
        apparent_target_span_k(np.array([np.nan], np.float32))
    assert apparent_target_span_k(np.full(10, 280.0, np.float32)) == (280.0, 281.0)


def test_target_code_span_reads_the_grey_channel_under_the_mask() -> None:
    frame = np.zeros((4, 5, 3), np.uint8)
    frame[1, 1] = 40
    frame[2, 3] = 43
    frame[0, 0] = 255  # sky: outside the mask
    mask = np.zeros((4, 5), bool)
    mask[1, 1] = mask[2, 3] = True
    assert target_code_span(frame, mask) == 3  # S52's "3 of 256"
    assert target_code_span(frame[..., 0], mask) == 3
    assert target_code_span(frame, np.zeros((4, 5), bool)) == 0


def test_the_interior_drops_the_rim_that_reads_the_sky_s_codes() -> None:
    """A 6x6 target at codes 100-105 whose rim is blended half with a code-10 sky: counted with
    its rim it spans ~50 codes, from its interior the 5 it really has."""
    frame = np.full((12, 12), 10, np.uint8)
    mask = np.zeros((12, 12), bool)
    mask[3:9, 3:9] = True
    frame[mask] = 100 + (np.arange(36) % 6).astype(np.uint8)
    rim = mask & ~interior(mask)
    frame[rim] = (frame[rim].astype(int) + 10) // 2
    assert interior(mask).sum() == 16
    assert target_code_span(frame, mask) > 40
    assert target_code_span(frame, interior(mask)) <= 5
    edge = np.zeros((4, 4), bool)
    edge[0:3, 0:3] = True
    assert not interior(edge)[0].any() and interior(edge)[1, 1]


def test_the_margin_leaves_the_frame_untouched_and_keeps_even_dimensions() -> None:
    rng = np.random.default_rng(1)
    frame = rng.integers(0, 256, (512, 640, 3), dtype=np.uint8)
    out = with_margin(frame)
    top, left = READOUT_MARGIN
    assert out.shape == (512 + top, 640 + left, 3)
    np.testing.assert_array_equal(out[top:, left:], frame)
    assert not out[:top].any() and not out[:, :left].any()
    assert out.shape[0] % 2 == 0 and out.shape[1] % 2 == 0  # h264 needs even dimensions
    grey = with_margin(frame[..., 0], (2, 4))
    assert grey.shape == (514, 644, 3)
    np.testing.assert_array_equal(grey[2:, 4:, 1], frame[..., 0])


@pytest.mark.parametrize("script", ["render_phantom4.py", "redisplay_planes.py"])
def test_no_demo_driver_rebuilds_an_agc(script: str) -> None:
    """A driver that builds its own equaliser is no camera (S52): the AGC clip is the camera's
    `display8`, or the camera's own `run_display_branch` on its DN -- never `irsim.isp.agc`."""
    source = (REPO / "scripts" / script).read_text(encoding="utf-8")
    assert "irsim.isp.agc" not in source
    assert "plateau_lut" not in source
