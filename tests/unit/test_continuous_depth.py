"""OC.11 -- a receding surface is one surface, not a stack of occluders (ADR 0134).

`OC.6` composites depth layers back to front with ``over``. That is right between a foreground and
the background behind it and **wrong between two slices of one surface that recedes through both**,
because those slices do not hide each other. The wrong version leaves a seam of
``alpha (1 - alpha) (L_surface - L_behind)`` -- a quarter of the contrast at its worst -- ruled
along every bin boundary, and *more* layers means more seams, so the error grew with the knob that
is supposed to reduce it.

The oracle is the old composite itself, reimplemented here in nine lines. A test that only
asserted "the seam is small" would pass just as well against a version that had quietly reverted,
so the reference is the defect: `_over_chain` is what `OC.6` shipped, and the fix has to beat it by
an order of magnitude on a scene made of one receding surface while matching it where two separate
objects genuinely occlude.

**What is left.** The interior is not exact, and the reason is worth knowing rather than
tightening a tolerance around. Adjacent layers carry different kernels, so
``sum_i K_i * cover_i`` -- the geometry's blurred coverage -- ripples by about +/-0.5 % across a
hard bin boundary instead of staying at one, and where it dips the background fills the difference.
That is 0.5 % of the surface-to-background contrast against the 25 % the ``over`` chain left, and
unlike the seam it falls as layers are added. Removing it needs fractional membership -- a pixel
belonging partly to two adjacent bins rather than wholly to one -- which is a change to how layers
are built, not to how they are composited.
"""

from __future__ import annotations

import numpy as np
import pytest

from irsim.optics.layered import _kernel_for_layer, depth_layers, layered_defocus, separated
from irsim.optics.psf import DefocusKernelBank, apply_psf

FOCAL_MM, F_NUMBER, PITCH_UM = 14.0, 1.0, 12.0
SIZE = 96
#: Radiance of the receding surface and of the background it sits against. The gap between them is
#: what the seam is made of, so it is made large on purpose.
SURFACE_L, BACKGROUND_L = 8.0, 1.0


@pytest.fixture(scope="module")
def bank() -> DefocusKernelBank:
    return DefocusKernelBank(10.5, F_NUMBER, 1.654, PITCH_UM, 1, "hopkins")


def _receding_slab() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """A slab whose range ramps 3 -> 9 m across the frame, on a sky background.

    The ramp is the point: it populates every W020 bin, so the binning slices one surface into as
    many layers as it is allowed, which is the case the composite used to get wrong.
    """
    distance = np.zeros((SIZE, SIZE), dtype=np.float64)
    slab = np.zeros((SIZE, SIZE), dtype=bool)
    slab[16:80, 16:80] = True
    ramp = np.linspace(3.0, 9.0, SIZE, dtype=np.float64)[None, :] * np.ones((SIZE, 1))
    distance[slab] = ramp[slab]
    radiance = np.full((SIZE, SIZE), BACKGROUND_L)
    radiance[slab] = SURFACE_L
    background = np.full((SIZE, SIZE), BACKGROUND_L)
    return radiance, distance, ~slab, background


def _interior(mask: np.ndarray, margin: int = 12) -> np.ndarray:
    """``mask`` eroded by ``margin``, so the silhouette -- where blur belongs -- is excluded."""
    out = mask.copy()
    for _ in range(margin):
        out &= np.roll(out, 1, 0) & np.roll(out, -1, 0) & np.roll(out, 1, 1) & np.roll(out, -1, 1)
    return out


def _over_chain(bank, radiance, distance, focus_m, sky, background, cap):
    """What `OC.6` shipped: ``over`` between every pair of layers, seeded with the background.

    Kept here rather than in the module because it is the defect, not an option. If a future
    change makes `layered_defocus` agree with this again, the tests below say so.
    """
    layers = depth_layers(distance, FOCAL_MM, F_NUMBER, focus_m, sky, cap)
    colour = np.asarray(
        apply_psf(
            np.broadcast_to(background, radiance.shape).astype(np.float64),
            _kernel_for_layer(bank, layers[0].distance_m, FOCAL_MM, F_NUMBER, focus_m),
        ),
        dtype=np.float64,
    )
    weight = np.ones(radiance.shape)
    for layer in layers[1:]:
        kernel = _kernel_for_layer(bank, layer.distance_m, FOCAL_MM, F_NUMBER, focus_m)
        cover = layer.mask.astype(np.float64)
        alpha = np.clip(apply_psf(cover, kernel), 0.0, 1.0)
        colour = apply_psf(radiance * cover, kernel) + colour * (1.0 - alpha)
        weight = alpha + weight * (1.0 - alpha)
    return np.divide(colour, weight, out=np.zeros_like(colour), where=weight > 1e-12)


@pytest.mark.parametrize("cap", [3, 5, 8])
def test_a_receding_surface_loses_its_seam(bank, cap: int) -> None:
    """The whole of `OC.11`, measured against the composite it replaces, at every layer count."""
    radiance, distance, sky, background = _receding_slab()
    inside = _interior(~sky)
    assert inside.sum() > 500, "need a real interior to measure"
    args = (radiance, distance, bank, FOCAL_MM, F_NUMBER, 5.0)
    fixed = layered_defocus(*args, sky_mask=sky, max_layers=cap, background_radiance=background)
    old = _over_chain(bank, radiance, distance, 5.0, sky, background, cap)
    now = float(np.abs(fixed[inside] - SURFACE_L).max())
    before = float(np.abs(old[inside] - SURFACE_L).max())
    contrast = SURFACE_L - BACKGROUND_L
    # The over chain's deficit is alpha (1 - alpha) (L_surface - L_behind); it reaches a quarter of
    # the contrast only where two slices meet exactly half and half, and 13 % of it on this ramp.
    assert before > 0.1 * contrast, f"the reference must show the seam: {before:.4f}"
    assert now < 0.15 * before, f"seam {now:.4f} against the over chain's {before:.4f}"
    # What is left is the coverage ripple, half a percent of contrast -- see the module docstring.
    assert now < 0.015 * contrast, f"{now:.4f} is more than the coverage ripple can account for"


def test_the_error_falls_with_more_layers_instead_of_rising(bank) -> None:
    """The knob has to point the right way: more layers must never mean a worse picture.

    This is the assertion the defect actually violated -- `max_layers=8`, the shipped default, was
    the *worst* setting for any scene holding a receding surface, because it cut the most seams.
    """
    radiance, distance, sky, background = _receding_slab()
    inside = _interior(~sky)

    def rms(plane) -> float:
        return float(np.sqrt(((plane[inside] - SURFACE_L) ** 2).mean()))

    # RMS, not peak: one seam is as deep as eight, so the peak saturates while the *area* spoiled
    # keeps growing. The over chain runs 0.21 -> 0.50 across these caps and the fix 0.013 -> 0.008.
    now, before = [], []
    for cap in (3, 5, 8):
        now.append(
            rms(
                layered_defocus(
                    radiance,
                    distance,
                    bank,
                    FOCAL_MM,
                    F_NUMBER,
                    5.0,
                    sky_mask=sky,
                    max_layers=cap,
                    background_radiance=background,
                )
            )
        )
        before.append(rms(_over_chain(bank, radiance, distance, 5.0, sky, background, cap)))
    assert before == sorted(before), f"the reference must get worse with the cap: {before}"
    assert before[-1] > 2.0 * before[0], f"and markedly so: {before}"
    assert now == sorted(now, reverse=True), f"the fix must get better instead: {now}"


def test_a_genuinely_separate_object_still_occludes(bank) -> None:
    """The other half: additivity must not be bought by letting a far object shine through a near.

    Two slabs with nothing between them in depth, so the binning leaves empty bins between their
    two layers and the composite must keep `over`. If it did not, the far slab's blur would raise
    the near slab's interior above its own radiance.
    """
    distance = np.zeros((SIZE, SIZE), dtype=np.float64)
    near = np.zeros((SIZE, SIZE), dtype=bool)
    near[24:72, 24:72] = True
    far = ~near
    distance[near] = 3.0
    distance[far] = 400.0  # far enough that the bins between them are empty
    radiance = np.where(near, SURFACE_L, 40.0)  # the FAR slab is the bright one
    out = layered_defocus(radiance, distance, bank, FOCAL_MM, F_NUMBER, 3.0, max_layers=8)
    inside = _interior(near, margin=10)
    assert inside.sum() > 200
    worst = float(np.abs(out[inside] - SURFACE_L).max())
    assert worst < 1e-6, f"the far slab bled {worst:.4f} into the near slab it is behind"


def test_the_span_says_which_layers_are_one_surface() -> None:
    """The record that carries the decision, checked on both cases it has to tell apart.

    The W020 bins themselves cannot answer this, and that is why the span exists: W020 is V-shaped
    about the focus distance, so sorting layers by range walks the bin indices up and back down
    again and two neighbours in the composite are routinely several bins apart.
    """
    _, ramp_d, ramp_sky, _ = _receding_slab()
    ramp = [layer for layer in depth_layers(ramp_d, FOCAL_MM, F_NUMBER, 5.0, ramp_sky)][1:]
    assert len(ramp) >= 3, "the ramp must actually split"
    assert not any(separated(a, b) for a, b in zip(ramp, ramp[1:], strict=False)), (
        "a smooth ramp is one surface"
    )

    split = np.zeros((SIZE, SIZE), dtype=np.float64)
    split[:, :48], split[:, 48:] = 3.0, 400.0
    pair = depth_layers(split, FOCAL_MM, F_NUMBER, 3.0)
    assert len(pair) == 2 and separated(pair[0], pair[1]), "two slabs with a void between them"


def test_sky_is_never_read_as_the_same_surface_as_the_geometry() -> None:
    """Sky spans infinity, so nothing can abut it and the gap test always separates it."""
    distance = np.full((SIZE, SIZE), 5.0)
    sky = np.zeros((SIZE, SIZE), dtype=bool)
    sky[:20] = True
    distance[sky] = 0.0
    layers = depth_layers(distance, FOCAL_MM, F_NUMBER, 5.0, sky)
    assert layers[0].span_m == (float("inf"), float("inf"))
    assert separated(layers[0], layers[1]), "geometry cannot be a slice of the sky"
