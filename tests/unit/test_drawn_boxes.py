"""EV.22: the box a person would draw, beside the truth's tight one.

A detector is scored on a real set's human boxes, and those are not the target's extent: on
Anti-UAV RGBT's sky-only test frames they are 1.28 times the visible target's width and 1.48
its height (``scripts/box_convention.py``), where irsim's were the mask's extent, 0.98 and 0.97.
:func:`irsim.io.labels.as_drawn` reproduces the convention from what a frame shows.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from irsim.io.labels import DRAWN_MARGIN, Box, FrameLabels, as_drawn, visible_extent

TRUTH = Box("quad", "drone", 90, 100, 150, 130, 1800, 1.0, [])


def _display(level: float = 220.0, sky: float = 60.0, sigma: float = 0.0) -> np.ndarray:
    frame = np.full((240, 320), sky)
    frame[TRUTH.y1 : TRUTH.y2, TRUTH.x1 : TRUTH.x2] = level
    return gaussian_filter(frame, sigma) if sigma else frame


def test_a_sharp_target_is_seen_to_its_own_edge() -> None:
    assert visible_extent(_display(), TRUTH) == (TRUTH.x1, TRUTH.y1, TRUTH.x2, TRUTH.y2)


def test_a_defocused_glow_counts_as_far_as_it_shows_and_a_dark_target_too() -> None:
    """Blurred, a step crosses half its contrast at the edge itself, so the visible extent is the
    truth to a pixel; a cold target against a warm sky is seen the same way."""
    for frame in (_display(sigma=3.0), _display(level=40.0, sky=200.0, sigma=3.0)):
        x1, y1, x2, y2 = visible_extent(frame, TRUTH)
        assert abs(x1 - TRUTH.x1) <= 1 and abs(x2 - TRUTH.x2) <= 1
        assert abs(y1 - TRUTH.y1) <= 1 and abs(y2 - TRUTH.y2) <= 1


def test_a_part_too_faint_to_see_is_not_boxed() -> None:
    """A tail at a fifth of the body's contrast lies inside the truth box and outside what a
    person sees: the visible extent stops at the body."""
    frame = _display()
    frame[TRUTH.y1 : TRUTH.y2, TRUTH.x1 : TRUTH.x1 + 20] = 60.0 + 0.2 * (220.0 - 60.0)
    assert visible_extent(frame, TRUTH)[0] == TRUTH.x1 + 20


def test_the_warm_sky_at_the_frame_edge_is_not_the_target() -> None:
    """EV.10: a frame aimed just above the horizon has a band as bright as the target at its
    bottom edge, inside the search window. Only what is connected to the target is boxed."""
    frame = _display()
    frame[150:, :] = 230.0  # the horizon's warm air, 20 px below the target's box
    assert visible_extent(frame, TRUTH) == (TRUTH.x1, TRUTH.y1, TRUTH.x2, TRUTH.y2)


def test_a_target_with_no_contrast_keeps_the_truth_box() -> None:
    assert visible_extent(np.full((240, 320), 80.0), TRUTH) == (90, 100, 150, 130)


def test_the_drawn_margins_are_the_measured_ones_and_the_truth_is_kept() -> None:
    """Over 2,000 draws the box grows by the real median margins (1.28 wide, 1.48 tall) about the
    visible centre; the mask, pixel count and visibility are the truth's."""
    labels = FrameLabels(320, 240, (TRUTH,), (), ("drone",))
    rng = np.random.default_rng(3)
    widths, heights = [], []
    for _ in range(2000):
        b = as_drawn(labels, _display(), rng).boxes[0]
        widths.append((b.x2 - b.x1) / 60.0)
        heights.append((b.y2 - b.y1) / 30.0)
        assert (b.pixels, b.visibility, b.mask_rle) == (1800, 1.0, [])
        assert abs(0.5 * (b.x1 + b.x2) - 120.0) <= 1.0 and abs(0.5 * (b.y1 + b.y2) - 115.0) <= 1.0
    assert np.median(widths) == pytest.approx(DRAWN_MARGIN["width"][0], rel=0.03)
    assert np.median(heights) == pytest.approx(DRAWN_MARGIN["height"][0], rel=0.04)


def test_a_drawn_box_stays_inside_the_frame() -> None:
    edge = Box("quad", "drone", 0, 0, 40, 20, 800, 1.0, [])
    frame = np.full((240, 320), 60.0)
    frame[0:20, 0:40] = 220.0
    b = as_drawn(FrameLabels(320, 240, (edge,), (), ("drone",)), frame, np.random.default_rng(0))
    assert b.boxes[0].x1 == 0 and b.boxes[0].y1 == 0
