"""OC.6 -- depth-varying defocus by layered compositing (ADR 0129).

The headline test is `test_the_background_stays_sharp_behind_a_blurred_foreground`: one kernel for
the frame blurs a sharp background because the *foreground* was out of focus, and that is the
defect layering exists to remove. Everything else here guards a property that makes the composite
trustworthy -- ordering, sky, and the flat field.
"""

from __future__ import annotations

import numpy as np
import pytest

from irsim.optics.layered import DEFAULT_MAX_LAYERS, depth_layers, layered_defocus
from irsim.optics.psf import DefocusKernelBank, apply_psf

FOCAL_MM, F_NUMBER, PITCH_UM = 14.0, 1.0, 12.0
NEAR_M, FAR_M = 3.0, 200.0
SIZE = 96


@pytest.fixture(scope="module")
def bank() -> DefocusKernelBank:
    return DefocusKernelBank(10.5, F_NUMBER, 1.654, PITCH_UM, 1, "hopkins")


def _two_depth() -> tuple[np.ndarray, np.ndarray]:
    """A near slab on the left; a far background carrying a sharp edge of its own on the right."""
    distance = np.full((SIZE, SIZE), FAR_M)
    distance[:, :40] = NEAR_M
    radiance = np.full((SIZE, SIZE), 1.0)
    radiance[:, :40] = 8.0  # the near slab
    radiance[:, 70:] = 4.0  # a step in the BACKGROUND, far from the silhouette
    return radiance, distance


def _edge_width(row: np.ndarray, lo_i: int, hi_i: int) -> float:
    """10-90 % width of a monotone edge, in pixels. Falling edges are read reversed, because
    `np.interp` needs an increasing sample sequence and a silhouette can go either way."""
    seg = np.asarray(row[lo_i:hi_i], dtype=np.float64)
    if seg[-1] < seg[0]:
        seg = seg[::-1]
    lo, hi = float(seg.min()), float(seg.max())
    if hi - lo <= 0.0:
        return 0.0
    p10, p90 = lo + 0.1 * (hi - lo), lo + 0.9 * (hi - lo)
    x = np.arange(seg.size, dtype=np.float64)
    return float(np.interp(p90, seg, x) - np.interp(p10, seg, x))


def test_layers_run_farthest_first_with_sky_at_the_back(bank) -> None:
    distance = np.full((SIZE, SIZE), 50.0)
    distance[:, :30] = 5.0
    sky = np.zeros((SIZE, SIZE), bool)
    sky[:10, :] = True
    distance[sky] = 0.0  # the gbuffer convention
    layers = depth_layers(distance, FOCAL_MM, F_NUMBER, None, sky, DEFAULT_MAX_LAYERS)
    ranges = [layer.distance_m for layer in layers]
    assert ranges[0] == float("inf"), "sky is behind everything"
    assert ranges == sorted(ranges, reverse=True), f"layers must run far to near: {ranges}"
    assert sum(int(layer.mask.sum()) for layer in layers) == distance.size, "layers partition"


def test_a_flat_field_survives_any_focus(bank) -> None:
    """The normalisation property. Every kernel sums to one, so the quotient is the field --
    and if it were not, every defocused frame would carry a dark fringe at each depth step."""
    _, distance = _two_depth()
    flat = np.full((SIZE, SIZE), 3.0)
    for focus in (None, NEAR_M, 20.0):
        out = layered_defocus(flat, distance, bank, FOCAL_MM, F_NUMBER, focus)
        assert np.allclose(out, 3.0, rtol=1e-9), f"flat field broken at focus {focus}"


def test_the_background_stays_sharp_behind_a_blurred_foreground(bank) -> None:
    """The defect one global kernel has, stated as a measurement.

    Focus on the far background. The near slab is then badly defocused, and a single kernel chosen
    for the frame smears the background's own edge with the foreground's blur. Layering must not.
    """
    radiance, distance = _two_depth()
    layered = layered_defocus(radiance, distance, bank, FOCAL_MM, F_NUMBER, FAR_M)

    from irsim.optics.defocus import blur_circle_um, defocus_w020_um

    w020 = float(defocus_w020_um(blur_circle_um(NEAR_M, FOCAL_MM, F_NUMBER, FAR_M), F_NUMBER))
    globalised = apply_psf(radiance, bank.kernel_for(w020))

    row = SIZE // 2
    sharp = _edge_width(np.asarray(layered)[row], 64, 78)
    smeared = _edge_width(np.asarray(globalised)[row], 64, 78)
    assert sharp < smeared, "the in-focus background must survive layering"
    assert smeared / sharp > 1.8, f"and by a clear margin: {sharp:.2f} vs {smeared:.2f} px"

    # The other half, and the one whose absence let an earlier version of this test pass while the
    # binning had quietly collapsed to a single layer. A stage that simply left everything in focus
    # satisfies the assertions above, so the FOREGROUND must be shown to be blurred too -- against
    # the same scene focused on the foreground, which is the only like-for-like comparison.
    assert len(depth_layers(distance, FOCAL_MM, F_NUMBER, FAR_M)) >= 2, "the scene must layer"
    on_the_slab = layered_defocus(radiance, distance, bank, FOCAL_MM, F_NUMBER, NEAR_M)
    defocused = _edge_width(np.asarray(layered)[row], 30, 52)
    focused = _edge_width(np.asarray(on_the_slab)[row], 30, 52)
    # Widths add in quadrature (`OC.3`), so the defocus contribution is what must be substantial;
    # the in-focus width here is the diffraction PSF plus the box filter and is not the baseline.
    contribution = float(np.sqrt(max(defocused**2 - focused**2, 0.0)))
    assert contribution > 1.0, (
        f"the near slab must be blurred when focus is far: {focused:.2f} -> {defocused:.2f} px, "
        f"contribution {contribution:.2f} px"
    )


def test_the_defocused_silhouette_is_semi_transparent(bank) -> None:
    """A blurred foreground edge must hand over to the background gradually, not in one step."""
    radiance, distance = _two_depth()
    out = np.asarray(layered_defocus(radiance, distance, bank, FOCAL_MM, F_NUMBER, FAR_M))
    row = out[SIZE // 2, 30:52]
    between = (row > 1.2) & (row < 7.8)
    assert between.sum() >= 4, f"expected a graded silhouette, got {np.round(row, 2)}"
    assert row[0] == pytest.approx(8.0, abs=0.2) and row[-1] == pytest.approx(1.0, abs=0.2)


def test_sky_is_composited_at_infinity_and_not_at_zero_metres(bank) -> None:
    """Sky carries distance_m = 0. Put it in the nearest layer and it is blurred hardest of all."""
    distance = np.full((SIZE, SIZE), NEAR_M)
    sky = np.zeros((SIZE, SIZE), bool)
    sky[:, 48:] = True
    distance[sky] = 0.0
    radiance = np.where(sky, 1.0, 8.0).astype(np.float64)
    # focused at infinity: the sky half is in focus, the near half is not
    out = np.asarray(layered_defocus(radiance, distance, bank, FOCAL_MM, F_NUMBER, None, sky))
    deep_sky = out[SIZE // 2, 70:]
    assert np.allclose(deep_sky, 1.0, atol=1e-6), "sky away from the edge must be untouched"


def test_the_layer_count_is_bounded(bank) -> None:
    distance = np.linspace(2.0, 500.0, SIZE * SIZE).reshape(SIZE, SIZE)
    for cap in (1, 3, DEFAULT_MAX_LAYERS):
        assert len(depth_layers(distance, FOCAL_MM, F_NUMBER, None, None, cap)) <= cap


def test_a_frame_with_no_geometry_at_all_is_left_alone(bank) -> None:
    sky = np.ones((SIZE, SIZE), bool)
    radiance = np.full((SIZE, SIZE), 2.0)
    out = layered_defocus(radiance, np.zeros((SIZE, SIZE)), bank, FOCAL_MM, F_NUMBER, None, sky)
    assert np.allclose(out, 2.0, rtol=1e-9)


def test_float16_is_refused(bank) -> None:
    with pytest.raises(TypeError, match="float16"):
        layered_defocus(
            np.ones((8, 8), np.float16), np.full((8, 8), 10.0), bank, FOCAL_MM, F_NUMBER
        )
