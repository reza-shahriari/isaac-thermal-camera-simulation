"""EV.21: how soft a target's silhouette is, measured the same way on real and rendered frames.

``irsim.validation.targets.edge_width_px`` is the contrast over the steepest gradient in the box.
These tests hold it to the closed form it is read against -- a step blurred by a Gaussian of σ
reads √(2π σ² + 4) -- and to the property that lets it compare an 8-bit real frame with a render: a
linear display mapping does not move it.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from irsim.validation.targets import Box, blur_sigma_px, edge_width_px

BOX = Box(x=70, y=80, width=60, height=40)


def _target(sigma: float, level: float = 200.0, sky: float = 60.0) -> np.ndarray:
    frame = np.full((200, 200), sky)
    rows, cols = BOX.slice()
    frame[rows, cols] = level
    return gaussian_filter(frame, sigma) if sigma > 0 else frame


@pytest.mark.parametrize("sigma", [1.0, 2.0, 3.0, 4.0, 6.0])
def test_a_gaussian_blurred_target_reads_the_closed_form(sigma: float) -> None:
    """σ = 1-6 px read √(2π σ² + 4) to 4 %, and :func:`blur_sigma_px` gives σ back to 6 %: the
    edge's slope C / (σ√(2π)), widened by the central difference's two pixels."""
    width = edge_width_px(_target(sigma), BOX)
    assert width == pytest.approx(math.sqrt(2.0 * math.pi * sigma**2 + 4.0), rel=0.04)
    assert blur_sigma_px(width) == pytest.approx(sigma, rel=0.06)


def test_a_sharp_target_reads_the_operators_own_width() -> None:
    """An unblurred, pixel-aligned edge reads 2 px, and no blur: the floor the renders sat on."""
    assert edge_width_px(_target(0.0), BOX) == pytest.approx(2.0)
    assert blur_sigma_px(2.0) == 0.0


def test_a_linear_display_mapping_does_not_move_it() -> None:
    """An AGC's gain and offset cancel: the statistic can compare an 8-bit frame with a render."""
    frame = _target(2.5)
    assert edge_width_px(3.7 * frame + 41.0, BOX) == pytest.approx(edge_width_px(frame, BOX))


def test_a_dark_target_reads_like_a_bright_one() -> None:
    """Polarity is the contrast's sign, not the measure's: a cold drone against a warm sky."""
    bright = edge_width_px(_target(3.0, level=200.0, sky=60.0), BOX)
    dark = edge_width_px(_target(3.0, level=60.0, sky=200.0), BOX)
    assert dark == pytest.approx(bright, rel=0.02)


def test_a_box_with_nothing_in_it_is_refused() -> None:
    with pytest.raises(ValueError, match="no contrast"):
        edge_width_px(np.full((200, 200), 80.0), BOX)
