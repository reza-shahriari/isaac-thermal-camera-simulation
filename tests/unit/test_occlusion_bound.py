"""OC.8 -- what partial occlusion actually costs, in kelvin (ADR 0129, ADR 0131).

The plan said the error was "bounded". This is the bound. Three fills are compared against a
two-layer reference on the same scene at three depth ratios:

* `OC.6` alone -- normalise against the layers that are there;
* `OC.8` -- push-pull the farthest layer into the gap;
* `OC.7` -- hand the stage the true background, which only sky and sea can do.

Reported in apparent temperature, because that is the unit the error matters in: a fringe of a few
tenths of a kelvin is under a Boson's NETD and a fringe of several kelvin is a feature a detector
will learn.
"""

from __future__ import annotations

import numpy as np
import pytest

from irsim.optics.defocus import blur_circle_um, defocus_w020_um
from irsim.optics.layered import estimate_background, layered_defocus
from irsim.optics.psf import DefocusKernelBank, apply_psf
from irsim.radiometry.planck import band_radiance_tophat

FOCAL_MM, F_NUMBER, PITCH_UM = 14.0, 1.0, 12.0
BAND_UM = (7.5, 13.5)
SIZE = 96
FOREGROUND_K, BACKGROUND_K, PATCH_K = 320.0, 280.0, 300.0
#: The band the error is confined to, in pixels either side of the silhouette: half a blur circle.
SILHOUETTE = slice(28, 52)


@pytest.fixture(scope="module")
def table() -> tuple[np.ndarray, np.ndarray]:
    t = np.arange(240.0, 360.0 + 1e-9, 0.05)
    return t, np.array([band_radiance_tophat(*BAND_UM, float(x)) for x in t])


@pytest.fixture(scope="module")
def bank() -> DefocusKernelBank:
    return DefocusKernelBank(10.5, F_NUMBER, 1.654, PITCH_UM, 1, "hopkins")


def _apparent_k(radiance: np.ndarray, table) -> np.ndarray:
    t, lb = table
    return np.asarray(np.interp(np.asarray(radiance, dtype=np.float64), lb, t))


def _scene(near_m: float, far_m: float, table) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t, lb = table
    l_fg, l_bg, l_patch = np.interp([FOREGROUND_K, BACKGROUND_K, PATCH_K], t, lb)
    distance = np.full((SIZE, SIZE), far_m)
    distance[:, :40] = near_m
    background = np.full((SIZE, SIZE), l_bg)
    # Structure that is entirely HIDDEN behind the near slab (which covers 0-39) and reaches the
    # silhouette. Put it where the visible background also shows it and push-pull copies it in for
    # free, which flatters the fill: an earlier version of this scene scored it a perfect 0.00 K
    # because the feature continued into the visible region.
    background[:, 32:40] = l_patch
    radiance = background.copy()
    radiance[:, :40] = l_fg
    return radiance, distance, background


def _reference(radiance, distance, background, bank, near_m, far_m, focus_m):
    near = (distance == near_m).astype(np.float64)
    k_near = bank.kernel_for(
        float(defocus_w020_um(blur_circle_um(near_m, FOCAL_MM, F_NUMBER, focus_m), F_NUMBER))
    )
    k_far = bank.kernel_for(
        float(defocus_w020_um(blur_circle_um(far_m, FOCAL_MM, F_NUMBER, focus_m), F_NUMBER))
    )
    alpha = np.clip(apply_psf(near, k_near), 0.0, 1.0)
    return apply_psf(radiance * near, k_near) + apply_psf(background, k_far) * (1.0 - alpha)


@pytest.mark.parametrize(("near_m", "far_m"), [(3.0, 30.0), (3.0, 200.0), (8.0, 400.0)])
def test_the_occlusion_error_is_measured_and_ordered(near_m, far_m, bank, table) -> None:
    """The three fills must rank: analytic exact, push-pull better than nothing, and every one of
    them confined to the silhouette band."""
    radiance, distance, background = _scene(near_m, far_m, table)
    reference = _reference(radiance, distance, background, bank, near_m, far_m, far_m)
    truth_k = _apparent_k(reference, table)

    def _error(plane) -> np.ndarray:
        return np.abs(_apparent_k(np.asarray(plane), table) - truth_k)

    normalised = _error(layered_defocus(radiance, distance, bank, FOCAL_MM, F_NUMBER, far_m))
    guessed_bg = estimate_background(radiance, distance, FOCAL_MM, F_NUMBER, far_m)
    inpainted = _error(
        layered_defocus(
            radiance, distance, bank, FOCAL_MM, F_NUMBER, far_m, background_radiance=guessed_bg
        )
    )
    exact = _error(
        layered_defocus(
            radiance, distance, bank, FOCAL_MM, F_NUMBER, far_m, background_radiance=background
        )
    )

    # 0.5 mK is the 0.05 K lookup table's own interpolation residual, not a modelling error: in
    # radiance the analytic fill matches the reference to 1e-12 (`test_background_fill.py`). It is
    # a hundredth of a Boson's NETD either way.
    assert exact.max() < 0.005, "the analytic background is not an approximation"
    assert inpainted.max() < normalised.max(), (
        f"push-pull must beat the normalisation: {inpainted.max():.2f} vs {normalised.max():.2f} K"
    )
    # ...but only just, and that is the finding. Where the hidden background has structure of its
    # own, no extrapolation from the visible part can recover it: push-pull buys 1-10 % here, not
    # an order of magnitude. The exact answer for clutter is a second rendered depth layer, which
    # ADR 0131 writes up and deliberately does not schedule.
    assert inpainted.max() > 0.5 * normalised.max(), (
        "push-pull is an extrapolation, not a solution; a large win means the scene was too easy"
    )
    assert normalised.max() > 1.0, "there must be a real error to improve on"

    # and both approximations stay inside the silhouette band
    for name, err in (("normalised", normalised), ("inpainted", inpainted)):
        outside = max(err[:, : SILHOUETTE.start].max(), err[:, SILHOUETTE.stop :].max())
        assert outside < 0.05 * err.max(), f"{name} error leaked out of the band"


def test_push_pull_leaves_the_known_region_alone(bank, table) -> None:
    radiance, distance, _ = _scene(3.0, 200.0, table)
    known = distance == 200.0
    filled = estimate_background(radiance, distance, FOCAL_MM, F_NUMBER, 200.0)
    assert np.allclose(filled[known], radiance[known]), "a fill must not edit what it was given"
    assert np.all(np.isfinite(filled))


def test_the_bound_is_reported_in_kelvin_for_the_record(bank, table) -> None:
    """The number `OC.8` exists to produce, pinned so a regression in any of the three fills is
    visible as a change in the published figure rather than as a silent drift."""
    radiance, distance, background = _scene(3.0, 200.0, table)
    reference = _reference(radiance, distance, background, bank, 3.0, 200.0, 200.0)
    truth_k = _apparent_k(reference, table)
    normalised = np.abs(
        _apparent_k(
            np.asarray(layered_defocus(radiance, distance, bank, FOCAL_MM, F_NUMBER, 200.0)), table
        )
        - truth_k
    )
    assert normalised.max() == pytest.approx(5.1, abs=0.4), (
        f"the published occlusion bound has moved: {normalised.max():.2f} K"
    )
