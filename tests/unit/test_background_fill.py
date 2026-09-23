"""OC.7 -- the analytic hidden layer (ADR 0129).

A defocused foreground silhouette lets the background show through, and a single-layer G-buffer has
no behind. `OC.6` fills the gap by normalising against the layers it *does* have, which is a guess.
Where the thing behind is sky or sea, it need not be: their radiance is a function of ray direction
this pipeline already evaluates, so the true background can be handed to the stage for every pixel
at no extra render cost. These tests hold that claim to an exact reference.
"""

from __future__ import annotations

import numpy as np
import pytest

from irsim.config.gbuffer import OPTIONAL_KEYS, PRECISION_CRITICAL_KEYS
from irsim.optics.defocus import blur_circle_um, defocus_w020_um
from irsim.optics.layered import layered_defocus
from irsim.optics.psf import DefocusKernelBank, apply_psf

FOCAL_MM, F_NUMBER = 14.0, 1.0
NEAR_M, FAR_M = 3.0, 200.0
SIZE = 96


@pytest.fixture(scope="module")
def bank() -> DefocusKernelBank:
    return DefocusKernelBank(10.5, F_NUMBER, 1.654, 12.0, 1, "hopkins")


def _scene() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(radiance seen, distance, true background). A hot near slab over a structured background."""
    distance = np.full((SIZE, SIZE), FAR_M)
    distance[:, :40] = NEAR_M
    background = np.full((SIZE, SIZE), 1.0)
    background[:, 60:] = 4.0  # the background has structure of its own, behind and beside the slab
    radiance = background.copy()
    radiance[:, :40] = 8.0
    return radiance, distance, background


def _exact_reference(radiance, distance, background, bank, focus_m) -> np.ndarray:
    """Two layers composited with the background KNOWN -- what a second rendered depth layer would
    give, built here by construction rather than by rendering one."""
    near = (distance == NEAR_M).astype(np.float64)
    k_near = bank.kernel_for(
        float(defocus_w020_um(blur_circle_um(NEAR_M, FOCAL_MM, F_NUMBER, focus_m), F_NUMBER))
    )
    k_far = bank.kernel_for(
        float(defocus_w020_um(blur_circle_um(FAR_M, FOCAL_MM, F_NUMBER, focus_m), F_NUMBER))
    )
    alpha = np.clip(apply_psf(near, k_near), 0.0, 1.0)
    return apply_psf(radiance * near, k_near) + apply_psf(background, k_far) * (1.0 - alpha)


def test_the_analytic_background_reproduces_the_two_layer_reference_exactly(bank) -> None:
    radiance, distance, background = _scene()
    got = layered_defocus(
        radiance, distance, bank, FOCAL_MM, F_NUMBER, FAR_M, background_radiance=background
    )
    want = _exact_reference(radiance, distance, background, bank, FAR_M)
    assert np.max(np.abs(np.asarray(got) - want)) < 1e-12, "the fill must be exact, not close"


def test_without_it_the_silhouette_carries_a_measurable_error(bank) -> None:
    """The size of the approximation `OC.6` alone makes -- the number `OC.8` has to live with."""
    radiance, distance, background = _scene()
    guess = np.asarray(layered_defocus(radiance, distance, bank, FOCAL_MM, F_NUMBER, FAR_M))
    want = _exact_reference(radiance, distance, background, bank, FAR_M)
    error = np.abs(guess - want)
    assert error.max() > 1.0, "the guess must actually be wrong, or this step buys nothing"

    # and the error is confined to the silhouette, not spread over the frame
    column_error = error.max(axis=0)
    silhouette = slice(32, 48)
    assert column_error[silhouette].max() == pytest.approx(error.max(), rel=1e-9)
    away = np.concatenate([column_error[:28], column_error[56:]])
    assert away.max() < 0.01 * error.max(), "the error must be local to the defocused edge"


def test_the_background_makes_the_normalisation_an_identity(bank) -> None:
    """With an opaque backmost layer the accumulated weight is 1 everywhere, so the division that
    `OC.6` needs stops doing anything -- which is why the result can be exact."""
    radiance, distance, background = _scene()
    got = np.asarray(
        layered_defocus(
            radiance, distance, bank, FOCAL_MM, F_NUMBER, FAR_M, background_radiance=background
        )
    )
    # energy: with the true background there is nothing to invent, so the frame mean sits between
    # the extremes of what was there rather than being pulled up by a renormalised deficit
    assert got.min() >= min(radiance.min(), background.min()) - 1e-9
    assert got.max() <= max(radiance.max(), background.max()) + 1e-9


def test_a_scalar_background_is_accepted(bank) -> None:
    radiance, distance, _ = _scene()
    plane = layered_defocus(
        radiance, distance, bank, FOCAL_MM, F_NUMBER, FAR_M, background_radiance=1.0
    )
    uniform = np.full((SIZE, SIZE), 1.0)
    assert np.allclose(
        plane,
        layered_defocus(
            radiance, distance, bank, FOCAL_MM, F_NUMBER, FAR_M, background_radiance=uniform
        ),
    )


def test_the_plane_is_part_of_the_gbuffer_contract_and_is_precision_critical() -> None:
    """It is a temperature, so float16 would cost 0.25 K at 300 K -- five NETDs (#2)."""
    assert "background_t_k" in OPTIONAL_KEYS
    assert "background_t_k" in PRECISION_CRITICAL_KEYS
