"""OC.11 and OC.13 -- a receding surface is one surface, not a stack of occluders (ADR 0134).

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

`OC.13` then removed what `OC.11` left. With hard bins the geometry's blurred coverage
``sum_b K_b * cover_b`` rippled by about +/-0.5 % across a bin boundary rather than staying at one
-- adjacent bins carry different kernels, the wider one spreads its coverage further than the
narrower one gathers it back -- and wherever it dipped the composite read the shortfall as sky
showing through and filled it with background. Making membership **fractional**, so a pixel is
shared between the two bins its W020 falls between, removes the step those mismatched kernels act
on: the ripple goes to +/-0.06 % and the interior error with it, 0.079 -> 0.0023 at three layers.

One assertion had to change rather than tighten. Under `OC.11` the error fell monotonically as
layers were added, and that was worth asserting because the defect had made it *rise*. It no longer
falls monotonically, and nothing is wrong: at 4e-4 on a contrast of 7 the residual is second order,
and which cap does best is decided by where the bin edges happen to land on this ramp rather than
by how many there are. What is asserted instead is that it stays small at every cap while the
reference keeps growing.
"""

from __future__ import annotations

import numpy as np
import pytest

from irsim.optics.defocus import blur_circle_um, defocus_w020_um
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


def _hard_bins(distance, focus_m, sky, cap):
    """The binning `OC.6` shipped: each pixel wholly in one equal-width W020 bin, farthest first.

    Reimplemented here rather than taken from `depth_layers`, because `OC.13` made membership
    fractional and a reference that borrowed the current binning would stop showing the defect the
    moment the binning changed -- which is exactly what happened when it did.
    """
    d = np.asarray(distance, dtype=np.float64)
    geometry = ~sky & np.isfinite(d) & (d > FOCAL_MM * 1e-3)
    w = np.zeros(d.shape)
    blur = blur_circle_um(d[geometry], FOCAL_MM, F_NUMBER, focus_m)
    w[geometry] = defocus_w020_um(blur, F_NUMBER)
    lo, hi = float(w[geometry].min()), float(w[geometry].max())
    n = cap - 1  # the sky layer takes one
    edges = np.linspace(lo, hi, n + 1)
    index = np.clip(np.digitize(w, edges[1:-1], right=False), 0, n - 1)
    out = []
    for b in range(n):
        mask = geometry & (index == b)
        if np.any(mask):
            out.append((mask, float(np.median(d[mask]))))
    out.sort(key=lambda item: item[1], reverse=True)
    return out


def _over_chain(bank, radiance, distance, focus_m, sky, background, cap):
    """What `OC.6` shipped: hard bins, ``over`` between every pair, seeded with the background.

    Kept here rather than in the module because it is the defect, not an option. If a future
    change makes `layered_defocus` agree with this again, the tests below say so.
    """
    colour = np.asarray(
        apply_psf(
            np.broadcast_to(background, radiance.shape).astype(np.float64),
            _kernel_for_layer(bank, float("inf"), FOCAL_MM, F_NUMBER, focus_m),
        ),
        dtype=np.float64,
    )
    weight = np.ones(radiance.shape)
    for mask, representative_m in _hard_bins(distance, focus_m, sky, cap):
        kernel = _kernel_for_layer(bank, representative_m, FOCAL_MM, F_NUMBER, focus_m)
        cover = mask.astype(np.float64)
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
    # What is left after `OC.13` is the residual coverage ripple, well under a tenth of a percent
    # of contrast. Before it this bound was 1.5 %, which hard bins needed and no longer do.
    assert now < 0.001 * contrast, f"{now:.5f} is more than the coverage ripple can account for"


def test_the_error_does_not_grow_with_the_layer_cap(bank) -> None:
    """The knob has to point the right way: more layers must never mean a worse picture.

    This is the assertion the defect actually violated -- `max_layers=8`, the shipped default, was
    the *worst* setting for any scene holding a receding surface, because it cut the most seams.
    `OC.13` does not restore monotonicity and is not asked to; see the module docstring.
    """
    radiance, distance, sky, background = _receding_slab()
    inside = _interior(~sky)

    def rms(plane) -> float:
        return float(np.sqrt(((plane[inside] - SURFACE_L) ** 2).mean()))

    # RMS, not peak: one seam is as deep as eight, so the peak saturates while the *area* spoiled
    # keeps growing. The over chain runs 0.21 -> 0.50 across these caps; the fix stays near 4e-4.
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
    # The fix need not *improve* with the cap, but it must stay far below the reference at every
    # one of them and must not drift upward the way the defect did.
    assert max(now) < 0.01 * min(before), f"the fix must stay two orders below: {now} vs {before}"
    assert now[-1] < 2.0 * now[0], f"and must not grow with the cap: {now}"


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


# ---------------------------------------------------------------------------------------------
# OC.13 -- fractional membership
# ---------------------------------------------------------------------------------------------


def test_membership_is_a_partition_of_unity() -> None:
    """Every geometry pixel is spread over the layers and adds up to exactly one of itself.

    This is what lets the composite stay additive. If the weights summed to less than one the
    background would show through a solid surface, and if they summed to more the surface would
    be brighter than it is; both would be invisible on a scene whose layers happen to align.
    """
    _, distance, sky, _ = _receding_slab()
    layers = depth_layers(distance, FOCAL_MM, F_NUMBER, 5.0, sky, 8)
    geometry = ~sky
    total = sum(layer.weight for layer in layers if layer.distance_m != float("inf"))
    assert np.allclose(total[geometry], 1.0, atol=1e-12), "geometry must be wholly accounted for"
    assert np.all(total[sky] == 0.0), "and sky must be in the sky layer, not shared with it"
    for layer in layers:
        assert np.array_equal(layer.mask, layer.weight > 0.0), "mask and weight must agree"
        assert layer.weight.min() >= 0.0 and layer.weight.max() <= 1.0


def test_a_surface_crossing_a_bin_boundary_ramps_rather_than_steps() -> None:
    """The point of `OC.13`: a boundary is a ramp, leaving no step for the kernels to act on."""
    _, distance, sky, _ = _receding_slab()
    layers = [
        layer
        for layer in depth_layers(distance, FOCAL_MM, F_NUMBER, 5.0, sky, 8)
        if layer.distance_m != float("inf")
    ]
    shared = [layer for layer in layers if np.any((layer.weight > 0.01) & (layer.weight < 0.99))]
    assert len(shared) >= 2, "a smooth ramp must leave pixels partly in one bin and partly the next"
    # A hard partition takes only the values 0 and 1, so the count of intermediate weights is
    # exactly what distinguishes this from what `OC.6` did.
    partial = sum(int(((layer.weight > 0.01) & (layer.weight < 0.99)).sum()) for layer in layers)
    assert partial > 1000, f"only {partial} shared pixels across the whole ramp"


def test_two_discrete_slabs_keep_hard_membership() -> None:
    """Where the scene has no continuum, the split degenerates to the one `OC.6` made.

    Worth asserting because it is why every earlier layered test still passes unchanged: a scene of
    flat slabs has no pixel between two bins, so there is nothing for fractional membership to do.
    """
    distance = np.zeros((SIZE, SIZE), dtype=np.float64)
    distance[:, :48], distance[:, 48:] = 3.0, 400.0
    layers = depth_layers(distance, FOCAL_MM, F_NUMBER, 3.0, None, 8)
    for layer in layers:
        assert np.array_equal(layer.weight, layer.mask.astype(np.float64)), "no pixel is shared"


def test_the_coverage_ripple_is_what_oc13_removed(bank) -> None:
    """The mechanism, measured: hard bins leave a ripple in the blurred coverage and ramps do not.

    ``sum_b K_b * cover_b`` is the fraction of the aperture bundle the geometry fills. On the
    interior of an opaque surface it must be one. Adjacent bins carry different kernels, so across
    a hard boundary it is not -- and `layered_defocus` spends the shortfall on background.
    """
    _, distance, sky, _ = _receding_slab()
    inside = _interior(~sky)

    def ripple(covers) -> float:
        total = np.zeros((SIZE, SIZE))
        for cover, representative_m in covers:
            kernel = _kernel_for_layer(bank, representative_m, FOCAL_MM, F_NUMBER, 5.0)
            total += np.clip(apply_psf(cover, kernel), 0.0, 1.0)
        return float(np.abs(total[inside] - 1.0).max())

    hard = ripple([(m.astype(np.float64), r) for m, r in _hard_bins(distance, 5.0, sky, 8)])
    soft = ripple(
        [
            (layer.weight, layer.distance_m)
            for layer in depth_layers(distance, FOCAL_MM, F_NUMBER, 5.0, sky, 8)
            if layer.distance_m != float("inf")
        ]
    )
    assert hard > 0.004, f"the hard binning must show the ripple: {hard:.5f}"
    assert soft < 0.2 * hard, f"ramping must flatten it: {soft:.5f} against {hard:.5f}"
