"""Depth-varying defocus by layered compositing (`OC.6`; docs/physics-model.md §8.3, ADR 0129).

One kernel for the frame (`OC.5`) is right when the scene's depth spread is narrow -- a target
against sky, open sea -- and wrong as soon as a frame holds both a near object and a far one,
because the near object's blur is then applied to the far one as well.

The fix is the standard one: split the frame into depth layers, blur each with **its own** kernel
together with its coverage, and composite back to front with the blurred coverage as alpha::

    out = premultiplied_layer + out * (1 - alpha)

Blurring the premultiplied radiance rather than gathering is what makes this a scatter: a
defocused layer spreads its own energy outward over whatever is behind it, and its edge becomes
genuinely semi-transparent instead of a hard cut.

**But that formula is only right between layers that occlude each other** (`OC.11`). Slicing one
receding surface into depth layers puts two slices of the *same* surface either side of a bin edge,
and they do not hide each other -- composited with ``over`` they leave a seam worth a quarter of
the surface-to-background contrast ruled along every bin boundary. :func:`layered_defocus` applies
``over`` only across a genuine depth gap and adds otherwise, over layers whose membership is
**fractional** (`OC.13`) so that a bin boundary is a ramp and not a step; the reasoning is there.

**What this does not fix.** Where a defocused foreground uncovers background, the background's
radiance behind it was never rendered -- a single-layer G-buffer has no behind -- so the alpha
opens a hole onto whatever the farther layers happened to put there. `OC.7` fills that hole
exactly for sky and sea, whose radiance is a function of ray direction this pipeline can evaluate;
`OC.8` measures what is left in ground clutter.

**Sky.** `irsim.config.gbuffer` writes ``distance_m = 0`` on sky pixels with ``sky_mask`` set. Sky
belongs in the farthest layer at the defocus of an object at infinity; reading its distance
literally would put the sky in the nearest layer and blur it hardest of all.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from irsim.optics.defocus import blur_circle_um, defocus_w020_um
from irsim.optics.psf import DefocusKernelBank, apply_psf

__all__ = [
    "DEFAULT_MAX_LAYERS",
    "DepthLayer",
    "depth_layers",
    "estimate_background",
    "layered_defocus",
    "push_pull_fill",
    "separated",
]

FloatArray = NDArray[np.float64]

#: Layers cost one FFT convolution each. Eight resolves a near/far scene well past what the box
#: filter can show and keeps the stage inside an order of magnitude of the single-kernel path.
DEFAULT_MAX_LAYERS = 8


class DepthLayer(NamedTuple):
    """One layer: where it is in frame, how far away, and **the depth range it spans**.

    ``span_m`` is not bookkeeping. Two layers whose spans abut are two slices of one surface that
    recedes through both; two layers with a clear gap between their spans are separate things with
    empty space in between. The composite has to tell those apart -- see :func:`layered_defocus`
    and :func:`separated` -- because only the second pair occludes.

    The span is the layer's own min and max range, **not** its bin edges. A bin is equal-width in
    W020 and W020 is V-shaped about the focus distance, so one bin can hold pixels from either
    side of focus and its span is then the whole scene. That is the correct reading: a surface
    reaching through focus is one surface.
    """

    mask: NDArray[np.bool_]
    distance_m: float
    span_m: tuple[float, float]
    #: Fractional membership, 0 to 1, same shape as ``mask`` and non-zero exactly where it is
    #: (`OC.13`). A pixel is shared between the two bins its W020 falls between rather than
    #: assigned wholly to one, and the weights of all the geometry layers sum to 1 on every
    #: geometry pixel. This is the plane the composite blurs; ``mask`` is for callers that need
    #: to ask *where* a layer is rather than *how much* of it is here.
    weight: NDArray[np.float64]


def separated(far: DepthLayer, near: DepthLayer) -> bool:
    """Is there empty space between these two layers, or are they one surface?

    True when the gap between their depth spans is wider than either layer is deep. The comparison
    is against the layers' own extents rather than a distance in metres, so it carries from a
    0.6 m cube to a 200 m ground plane without a constant to tune: a gap that is large *for this
    scene* is what separates two objects, and a surface sliced into layers leaves gaps of one depth
    sample, which is small for any scene.
    """
    gap_m = far.span_m[0] - near.span_m[1]  # the void between the near layer's back and the far
    if not np.isfinite(gap_m):  # sky is behind everything and touches nothing
        return True
    deepest_m = max(far.span_m[1] - far.span_m[0], near.span_m[1] - near.span_m[0])
    return bool(gap_m > deepest_m)


def depth_layers(  # noqa: PLR0913
    distance_m: object,
    focal_length_mm: float,
    f_number: float,
    focus_distance_m: float | None = None,
    sky_mask: object = None,
    max_layers: int = DEFAULT_MAX_LAYERS,
) -> list[DepthLayer]:
    """Split a distance plane into :class:`DepthLayer` records, **farthest first**.

    Bins are equal width in **W020**, not in distance and not in equal counts. W020 is what picks
    the kernel, so equal-width bins in it bound the blur error inside a layer directly: with eight
    layers the blur circle varies by an eighth of the frame's range within any one of them. Equal
    counts are the obvious choice and the wrong one -- a frame that is two thirds background and
    one third foreground puts the median in the background and collapses to a single layer, which
    silently turns this stage back into `OC.5`.

    Sky, if masked, is a layer of its own at infinity and sorts first because everything else is in
    front of it.

    Ordering is by **distance**, never by defocus: W020 is V-shaped about the focus distance, so
    two layers either side of focus can share a blur and must still composite in the right order.

    **Membership is fractional** (`OC.13`). A pixel is shared between the two bins its W020 falls
    between, in proportion, so a surface receding smoothly crosses a bin boundary as a ramp rather
    than a step; :attr:`DepthLayer.weight` carries the share and the weights sum to one on every
    geometry pixel. A scene of flat slabs has no pixel between two bins, so there the split
    degenerates to the hard partition this had before and nothing changes.
    """
    d = np.asarray(distance_m, dtype=np.float64)
    if max_layers < 1:
        raise ValueError("max_layers must be at least one")
    sky = np.zeros(d.shape, dtype=bool) if sky_mask is None else np.asarray(sky_mask, dtype=bool)
    geometry = ~sky & np.isfinite(d) & (d > focal_length_mm * 1e-3)

    layers: list[DepthLayer] = []
    if np.any(sky):
        layers.append(
            DepthLayer(sky, float("inf"), (float("inf"), float("inf")), sky.astype(np.float64))
        )
    if not np.any(geometry) or len(layers) >= max_layers:
        return layers

    w = np.zeros(d.shape, dtype=np.float64)
    w[geometry] = defocus_w020_um(
        blur_circle_um(d[geometry], focal_length_mm, f_number, focus_distance_m), f_number
    )
    lo, hi = float(w[geometry].min()), float(w[geometry].max())
    n = max_layers - len(layers)
    if n <= 1 or hi - lo <= 0.0:
        span = (float(d[geometry].min()), float(d[geometry].max()))
        return [
            *layers,
            DepthLayer(geometry, float(np.median(d[geometry])), span, geometry.astype(np.float64)),
        ]
    # `OC.13`: membership is fractional, not a hard partition. A pixel sits somewhere between two
    # bin centres in W020 and is split between them in proportion, so a surface receding smoothly
    # crosses a bin boundary as a ramp rather than a step. With hard masks the layers' blurred
    # coverage `sum_b K_b * cover_b` does not stay at one across such a boundary -- adjacent bins
    # carry different kernels, the wider one spreads its coverage further than the narrower one
    # gathers it back, and the sum ripples by about half a percent either way. Wherever it dips,
    # `layered_defocus` reads the shortfall as sky showing through and fills it with background.
    # Ramping the membership removes the step that the mismatched kernels act on.
    step = (hi - lo) / n
    centres = lo + step * (np.arange(n) + 0.5)
    #: Position in units of bin centres, clamped so the half-bin at either end is wholly owned.
    t = np.clip((w - centres[0]) / step, 0.0, n - 1.0)
    below = np.clip(np.floor(t), 0.0, n - 2.0)
    frac = t - below
    lower = below.astype(np.intp)
    bins = []
    for b in range(n):
        weight = np.where(lower == b, 1.0 - frac, 0.0) + np.where(lower + 1 == b, frac, 0.0)
        weight = np.where(geometry, weight, 0.0)
        mask = weight > 0.0
        if np.any(mask):
            span = (float(d[mask].min()), float(d[mask].max()))
            bins.append(DepthLayer(mask, _weighted_median(d[mask], weight[mask]), span, weight))
    bins.sort(key=lambda item: item.distance_m, reverse=True)  # farthest first
    return [*layers, *bins]


def _weighted_median(values: FloatArray, weights: FloatArray) -> float:
    """Median of ``values`` under ``weights`` -- the fractional analogue of `OC.5`'s median.

    The median rather than the mean for the reason `OC.5` gives: a layer straddling a depth edge,
    or a few pixels of mis-segmentation onto the background, should not drag the kernel. Weighting
    it keeps that property while letting a pixel count as the fraction of itself it really is.
    """
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    return float(values[order][np.searchsorted(cumulative, 0.5 * cumulative[-1])])


def layered_defocus(  # noqa: PLR0913
    radiance_ss: NDArray[np.floating],
    distance_m: object,
    bank: DefocusKernelBank,
    focal_length_mm: float,
    f_number: float,
    focus_distance_m: float | None = None,
    sky_mask: object = None,
    max_layers: int = DEFAULT_MAX_LAYERS,
    background_radiance: object = None,
) -> NDArray[np.floating]:
    """Depth-varying defocus of a supersampled radiance plane, composited back to front.

    Returns a plane of the same shape and dtype.

    The composite is ``over`` **between layers that occlude** and **additive between layers that
    do not**, normalised at the end by the accumulated alpha::

        keep   = (1 - alpha) if this layer occludes the accumulation else 1
        colour = premultiplied + colour * keep
        alpha_total = alpha    + alpha_total * keep
        out    = colour / alpha_total

    **Only some layers occlude, and using ``over`` for all of them is a real error** (`OC.11`).
    Two layers from *consecutive* W020 bins are two slices of one surface that recedes through
    both -- a ground plane, the flank of a vehicle, the receding face of any solid. Such slices do
    not hide each other: at a pixel on the boundary between them the aperture bundle lands partly
    on one slice and partly on the other, so their contributions **add**. Composite them with
    ``over`` instead and the far slice is multiplied by ``1 - alpha_near``, which leaves a deficit

        alpha (1 - alpha) (L_surface - L_behind)

    along every bin boundary -- peaking at a quarter of the surface-to-background contrast, at
    ``alpha = 1/2``. That is a **dark seam ruled across the surface at each bin edge**, and because
    more bins mean more edges it grows worse as ``max_layers`` rises, which inverts the knob it is
    meant to be. Measured on a 0.6 m cube at 3 m against sky: **8.6 K** peak, and RMS over the
    cube rising 1.31 -> 1.44 -> 2.32 K as the cap goes 3 -> 4 -> 8.

    Layers with an **empty bin between them** are a different matter: nothing in the scene lies at
    the depths in between, so they are separate objects with space between them, and the nearer
    one does occlude. That pair keeps ``over``. The test is the bin index rather than a tuned
    distance because it asks the question that matters -- *is there anything in between?* -- and
    reads the answer off a histogram the binning has already computed.

    Two layers that happen to land in adjacent bins while being genuinely separate objects are
    treated as one surface, which is harmless: adjacent bins differ by an eighth of the frame's
    W020 range, so their kernels, and therefore the two compositing rules, nearly agree.

    The **background** never gets this exemption. It is complete -- it has coverage at every pixel,
    including behind the geometry -- so everything geometric is genuinely in front of it, and it is
    composited under the accumulated geometry alpha.

    The normalisation is where the partial-occlusion approximation lives. Without a background
    plane, a defocused layer's alpha opens a gap that no farther layer fills, because what was
    behind it was never rendered; dividing by the accumulated weight fills that gap with the layers
    that *are* there, in proportion to how much of each is visible. That is a defensible guess and
    not the truth; `OC.7` replaces it with the exact answer wherever the thing behind is sky or
    sea, and `OC.8` measures what the guess costs in ground clutter.

    What each layer contributes is its **fractional coverage** (`OC.13`), not a hard mask. With a
    hard partition the layers' blurred coverage ``sum_b K_b * cover_b`` does not stay at one across
    a bin boundary -- adjacent bins carry different kernels, the wider one spreads its coverage
    further than the narrower one gathers it back -- and the composite spends the shortfall on
    background, which is a half-percent-of-contrast error along every boundary of a receding
    surface. Ramped membership removes the step those mismatched kernels act on and takes it to
    six hundredths of a percent.

    A uniform field survives any focus exactly under either rule: every kernel sums to one, so
    colour and weight are scaled alike and the quotient is the field.

    ``background_radiance`` (`OC.7`) removes the guess where the answer is known. It is the
    backmost layer **completed**: the radiance each pixel's ray would report with everything in
    front of that layer removed, so it carries both what the layer shows where it is visible and
    what it would show where a nearer layer hides it. It therefore **replaces** the backmost layer
    in the composite rather than sitting behind it. For a scene whose backdrop is sky or sea that is
    the radiance of each ray with all geometry removed -- for a sky or sea background that
    is an analytic function of ray direction this pipeline already evaluates, so it costs no second
    render pass and it is exact. Given it, the geometry is composited over a background that
    carries the truth everywhere, the accumulated weight stays 1, and the normalisation above
    becomes the identity. This is what makes the aerial and maritime lanes exact rather than
    bounded; `OC.8` is what is left for ground clutter, where no such function exists.
    """
    x = np.asarray(radiance_ss)
    if x.dtype == np.float16:
        raise TypeError("radiance plane is float16 (non-negotiable #2)")
    layers = depth_layers(
        distance_m, focal_length_mm, f_number, focus_distance_m, sky_mask, max_layers
    )
    if not layers:
        return x
    background: FloatArray | None = None
    if background_radiance is not None:
        # `OC.7`: the radiance of the ray with all geometry removed, known for every pixel and not
        # only where sky is visible. Seeding the composite with it as a fully opaque backmost layer
        # fills the occlusion gap with the truth, and the accumulated weight then stays 1 -- the
        # normalisation below becomes the identity rather than a guess.
        #
        # It is **blurred first**, with the backmost layer's kernel. The background is a scene
        # plane like any other and the lens defocuses it too; seeding it sharp leaves the hidden
        # region -- the only place the seed survives, since a layer covers it everywhere else --
        # carrying an unblurred background inside an otherwise blurred frame. The error is
        # invisible wherever the hidden background is locally uniform, which is why it took a scene
        # with structure hidden *entirely* behind the foreground to surface it (`OC.8`).
        plane = np.broadcast_to(np.asarray(background_radiance, dtype=np.float64), x.shape).astype(
            np.float64
        )
        background = np.asarray(
            apply_psf(
                plane,
                _kernel_for_layer(
                    bank, layers[0].distance_m, focal_length_mm, f_number, focus_distance_m
                ),
            ),
            dtype=np.float64,
        )
        # The background plane is the backmost layer, *completed* -- it already carries what that
        # layer shows where it is visible as well as what it would show where the foreground hides
        # it. Compositing the layer as well would count its visible part twice, which is a real
        # 1.5 K error and not a rounding one (`OC.8`).
        layers = layers[1:]

    colour = np.zeros(x.shape, dtype=np.float64)
    alpha_total = np.zeros(x.shape, dtype=np.float64)
    previous: DepthLayer | None = None
    for layer in layers:
        kernel = _kernel_for_layer(
            bank, layer.distance_m, focal_length_mm, f_number, focus_distance_m
        )
        cover = np.asarray(layer.weight, dtype=np.float64)
        premultiplied = np.asarray(
            apply_psf(np.asarray(x, dtype=np.float64) * cover, kernel), dtype=np.float64
        )
        alpha = np.clip(np.asarray(apply_psf(cover, kernel), dtype=np.float64), 0.0, 1.0)
        keep = 1.0 - alpha if previous is not None and separated(previous, layer) else 1.0
        colour = premultiplied + colour * keep
        alpha_total = alpha + alpha_total * keep
        previous = layer

    if background is None:
        weight = alpha_total
    else:
        fill = np.clip(1.0 - alpha_total, 0.0, 1.0)
        colour = colour + background * fill
        weight = alpha_total + fill
    out = np.divide(colour, weight, out=np.zeros_like(colour), where=weight > 1e-12)
    return np.asarray(out, dtype=x.dtype)


def _kernel_for_layer(
    bank: DefocusKernelBank,
    representative_m: float,
    focal_length_mm: float,
    f_number: float,
    focus_distance_m: float | None,
) -> FloatArray:
    """The kernel a layer at ``representative_m`` earns; ``inf`` is an object at infinity."""
    if np.isinf(representative_m):
        c_um = (
            0.0
            if focus_distance_m is None
            else float(blur_circle_um(1e9, focal_length_mm, f_number, focus_distance_m))
        )
    else:
        c_um = float(blur_circle_um(representative_m, focal_length_mm, f_number, focus_distance_m))
    return bank.kernel_for(float(defocus_w020_um(c_um, f_number)))


def push_pull_fill(
    values: NDArray[np.floating], known: NDArray[np.bool_], levels: int = 8
) -> FloatArray:
    """Fill ``~known`` by push-pull extrapolation from the known neighbourhood (`OC.8`).

    The classic pyramid fill: average value and coverage down by 2 until the hole is smaller than a
    cell (*push*), then walk back up substituting the coarser estimate wherever coverage is still
    missing (*pull*). It is smooth, needs no iteration count to converge, and costs O(N).

    This is the **fallback**, for where `OC.7`'s analytic background does not exist -- ground
    clutter, where the thing behind the foreground is another object nobody rendered. It is an
    estimate and is measured as one: `test_occlusion_bound.py` reports what it costs in apparent
    temperature against a two-layer reference, beside what `OC.6`'s normalisation costs and what
    `OC.7` costs where it applies.
    """
    v = np.asarray(values, dtype=np.float64)
    w = np.asarray(known, dtype=np.float64)
    if v.shape != w.shape:
        raise ValueError("values and known must have the same shape")
    pyramid = [(v * w, w)]
    for _ in range(max(0, levels - 1)):
        num, den = pyramid[-1]
        if min(num.shape) <= 2:
            break
        pyramid.append((_halve(num), _halve(den)))
    filled_num, filled_den = pyramid[-1]
    for num, den in reversed(pyramid[:-1]):
        up_num = _double(filled_num, num.shape)
        up_den = _double(filled_den, den.shape)
        gap = den <= 0.0
        filled_num = np.where(gap, up_num, num)
        filled_den = np.where(gap, up_den, den)
    out = np.divide(filled_num, filled_den, out=np.zeros_like(filled_num), where=filled_den > 1e-12)
    return np.asarray(np.where(known, v, out))


def _halve(a: FloatArray) -> FloatArray:
    """2x2 box decimation, padding an odd edge by replication so no sample is dropped."""
    h, w = a.shape
    a = a[: h - h % 2, : w - w % 2] if (h % 2 or w % 2) else a
    return np.asarray(a.reshape(a.shape[0] // 2, 2, a.shape[1] // 2, 2).mean(axis=(1, 3)))


def _double(a: FloatArray, shape: tuple[int, ...]) -> FloatArray:
    """Nearest-neighbour upsample to exactly ``shape``.

    ``_halve`` drops an odd last row or column, so doubling can land one short; the edge is
    replicated rather than left unfilled, which would otherwise put a zero-coverage stripe down the
    side of any frame with an odd dimension at some pyramid level.
    """
    out = np.repeat(np.repeat(a, 2, axis=0), 2, axis=1)
    out = out[: shape[0], : shape[1]]
    if out.shape[0] < shape[0]:
        out = np.vstack([out, np.repeat(out[-1:], shape[0] - out.shape[0], axis=0)])
    if out.shape[1] < shape[1]:
        out = np.hstack([out, np.repeat(out[:, -1:], shape[1] - out.shape[1], axis=1)])
    return np.asarray(out)


def estimate_background(  # noqa: PLR0913
    radiance_ss: NDArray[np.floating],
    distance_m: object,
    focal_length_mm: float,
    f_number: float,
    focus_distance_m: float | None = None,
    sky_mask: object = None,
    max_layers: int = DEFAULT_MAX_LAYERS,
) -> FloatArray:
    """A background plane guessed from the frame itself, for scenes with no analytic one (`OC.8`).

    Everything in the farthest layer is taken as known background and pushed into the region the
    nearer layers occupy. Where a scene *has* an analytic background (`OC.7`) this is strictly
    worse and should not be used; where it has none, it is better than letting `OC.6`'s
    normalisation invent the fill, and the difference is measured rather than asserted.
    """
    layers = depth_layers(
        distance_m, focal_length_mm, f_number, focus_distance_m, sky_mask, max_layers
    )
    if not layers:
        return np.asarray(radiance_ss, dtype=np.float64)
    known = layers[0].mask  # the farthest layer, sky included
    return push_pull_fill(np.asarray(radiance_ss, dtype=np.float64), np.asarray(known, dtype=bool))
