"""OC.3 -- the two-cube focus demo (docs/physics-model.md §8.3, ADR 0129).

The test that matters is the last one: the *measured* edge width of each cube tracks the blur
circle its own range earns. A demo that merely blurs something would pass an "output changed" test
and fail this one.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from focus_demo import (  # noqa: E402
    F_NUMBER,
    FOCAL_MM,
    PITCH_UM,
    apparent_temperature,
    render,
    two_cube_scene,
)

from irsim.optics.defocus import blur_circle_um  # noqa: E402

NEAR_M, FAR_M, SS = 3.0, 80.0, 2


@pytest.fixture(scope="module")
def layers():
    return two_cube_scene(80, 64, SS, NEAR_M, FAR_M)


def _edge_width_px(profile: np.ndarray) -> float:
    """10-90 % rise width of a monotone edge, in pixels, by linear interpolation."""
    lo, hi = float(profile.min()), float(profile.max())
    assert hi - lo > 1.0, "need a real edge to measure"
    p10, p90 = lo + 0.1 * (hi - lo), lo + 0.9 * (hi - lo)
    x = np.arange(profile.size, dtype=np.float64)
    rising = profile if profile[-1] > profile[0] else profile[::-1]
    return float(np.interp(p90, rising, x) - np.interp(p10, rising, x))


def _cube_edge_profile(t_k: np.ndarray, row_frac: float, col_slice: slice) -> np.ndarray:
    return t_k[int(row_frac * t_k.shape[0]), col_slice]


def test_focusing_on_one_cube_leaves_the_other_blurred(layers) -> None:
    near_t = apparent_temperature(render(layers, NEAR_M, SS)[0])
    far_t = apparent_temperature(render(layers, FAR_M, SS)[0])

    near_sharp = _edge_width_px(_cube_edge_profile(near_t, 0.65, slice(24, 44)))
    near_blurred = _edge_width_px(_cube_edge_profile(far_t, 0.65, slice(24, 44)))
    far_blurred = _edge_width_px(_cube_edge_profile(near_t, 0.30, slice(38, 58)))
    far_sharp = _edge_width_px(_cube_edge_profile(far_t, 0.30, slice(38, 58)))

    assert near_sharp < near_blurred, "the near cube must be sharper when focused on it"
    assert far_sharp < far_blurred, "and the far cube when focused on it"
    assert near_blurred / near_sharp > 1.5
    assert far_blurred / far_sharp > 1.5


def test_the_measured_edge_width_tracks_the_blur_circle(layers) -> None:
    """The quantitative half, and the one a merely-blurry demo fails.

    Edge widths add in **quadrature**, not linearly, so the defocus contribution is
    ``sqrt(w² − w_focus²)``. That contribution must be a fixed fraction of the blur circle,
    whatever the defocus -- the fraction being the 10-90 % width of the kernel's line-spread
    function, about 0.65 of the disk diameter once diffraction has rounded its edges. A blur that
    grew with defocus but not *as* the blur circle would break the constant, not the trend.
    """
    in_focus = _edge_width_px(
        _cube_edge_profile(apparent_temperature(render(layers, NEAR_M, SS)[0]), 0.65, slice(24, 44))
    )
    ratios = []
    for focus in (6.0, 12.0, FAR_M):
        t_k = apparent_temperature(render(layers, focus, SS)[0])
        width = _edge_width_px(_cube_edge_profile(t_k, 0.65, slice(24, 44)))
        blur_px = float(blur_circle_um(NEAR_M, FOCAL_MM, F_NUMBER, focus)) / PITCH_UM
        assert width > in_focus
        ratios.append(np.sqrt(width**2 - in_focus**2) / blur_px)
    assert np.ptp(ratios) < 0.05, f"the edge does not track the blur circle: {ratios}"
    assert 0.5 < float(np.mean(ratios)) < 0.8, f"unexpected line-spread width: {ratios}"


def test_a_uniform_scene_survives_any_focus(layers) -> None:
    """Blur conserves radiance: a flat field is the same flat field however badly focused."""
    flat = [type(layers[0])("sky", None, 290.0, np.ones_like(layers[0].mask))]
    base = render(flat, None, SS)[0]
    for focus in (1.0, 5.0, 50.0):
        assert np.allclose(render(flat, focus, SS)[0], base, rtol=1e-9)


def test_sky_stays_sharp_when_the_lens_is_focused_at_infinity(layers) -> None:
    blur = render(layers, None, SS)[1]
    assert blur["sky"] == 0.0
    assert blur["near cube"] > 10.0 * PITCH_UM / PITCH_UM


def test_blurring_happens_in_radiance_not_in_kelvin(layers) -> None:
    """Half coverage of a 250 K / 315 K edge must read hotter than 282.5 K.

    L(T) is convex, so by Jensen the temperature of the averaged radiance exceeds the average of
    the temperatures. Averaging in Kelvin would land exactly on 282.5 and this test is what says
    the pipeline did not (CLAUDE.md non-negotiable #3).
    """
    from focus_demo import radiance_table

    t_grid, l_grid = radiance_table()
    l_cold, l_hot = np.interp([250.0, 315.0], t_grid, l_grid)
    half_radiance = float(np.interp(0.5 * (l_cold + l_hot), l_grid, t_grid))
    assert half_radiance > 282.5 + 1.0, "radiance-space blending must be hotter than the K mean"

    # and the rendered edge really does pass through that value rather than through 282.5
    t_k = apparent_temperature(render(layers, FAR_M, SS)[0])
    profile = _cube_edge_profile(t_k, 0.65, slice(24, 44))
    assert profile.min() < 260.0 < 300.0 < profile.max(), "the edge must span both temperatures"
    assert np.any(np.abs(profile - half_radiance) < 2.0), (
        "no sample near the radiance-blended midpoint"
    )
