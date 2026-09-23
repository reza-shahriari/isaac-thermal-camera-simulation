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

import numpy as np
from numpy.typing import NDArray

from irsim.optics.defocus import blur_circle_um, defocus_w020_um
from irsim.optics.psf import DefocusKernelBank, apply_psf

__all__ = ["DEFAULT_MAX_LAYERS", "depth_layers", "layered_defocus"]

FloatArray = NDArray[np.float64]

#: Layers cost one FFT convolution each. Eight resolves a near/far scene well past what the box
#: filter can show and keeps the stage inside an order of magnitude of the single-kernel path.
DEFAULT_MAX_LAYERS = 8


def depth_layers(  # noqa: PLR0913
    distance_m: object,
    focal_length_mm: float,
    f_number: float,
    focus_distance_m: float | None = None,
    sky_mask: object = None,
    max_layers: int = DEFAULT_MAX_LAYERS,
) -> list[tuple[NDArray[np.bool_], float]]:
    """Split a distance plane into (mask, representative distance) layers, **farthest first**.

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
    """
    d = np.asarray(distance_m, dtype=np.float64)
    if max_layers < 1:
        raise ValueError("max_layers must be at least one")
    sky = np.zeros(d.shape, dtype=bool) if sky_mask is None else np.asarray(sky_mask, dtype=bool)
    geometry = ~sky & np.isfinite(d) & (d > focal_length_mm * 1e-3)

    layers: list[tuple[NDArray[np.bool_], float]] = []
    if np.any(sky):
        layers.append((sky, float("inf")))
    if not np.any(geometry) or len(layers) >= max_layers:
        return layers

    w = np.zeros(d.shape, dtype=np.float64)
    w[geometry] = defocus_w020_um(
        blur_circle_um(d[geometry], focal_length_mm, f_number, focus_distance_m), f_number
    )
    lo, hi = float(w[geometry].min()), float(w[geometry].max())
    n = max_layers - len(layers)
    if n <= 1 or hi - lo <= 0.0:
        return [*layers, (geometry, float(np.median(d[geometry])))]
    edges = np.linspace(lo, hi, n + 1)
    index = np.clip(np.digitize(w, edges[1:-1], right=False), 0, n - 1)
    bins = []
    for b in range(n):
        mask = geometry & (index == b)
        if np.any(mask):
            bins.append((mask, float(np.median(d[mask]))))
    bins.sort(key=lambda item: item[1], reverse=True)  # farthest first
    return [*layers, *bins]


def layered_defocus(  # noqa: PLR0913
    radiance_ss: NDArray[np.floating],
    distance_m: object,
    bank: DefocusKernelBank,
    focal_length_mm: float,
    f_number: float,
    focus_distance_m: float | None = None,
    sky_mask: object = None,
    max_layers: int = DEFAULT_MAX_LAYERS,
) -> NDArray[np.floating]:
    """Depth-varying defocus of a supersampled radiance plane, composited back to front.

    Returns a plane of the same shape and dtype.

    The composite is ``over`` **normalised by the accumulated alpha**::

        colour = premultiplied + colour * (1 - alpha)
        weight = alpha         + weight * (1 - alpha)
        out    = colour / weight

    The normalisation is not cosmetic and is worth understanding, because it is where the
    partial-occlusion approximation actually lives. A plain ``over`` leaves a deficit wherever a
    defocused layer's alpha has opened a gap that no farther layer fills -- the background behind
    it was never rendered -- and that deficit reads as a dark fringe along every out-of-focus
    silhouette. Dividing by the accumulated weight fills the gap with the layers that *are* there,
    in proportion to how much of each is visible. That is a defensible guess and not the truth;
    `OC.7` replaces it with the exact answer wherever the thing behind is sky or sea, and `OC.8`
    measures what the guess costs in ground clutter.

    A uniform field therefore survives any focus exactly: every kernel sums to one, so colour and
    weight are scaled alike and the quotient is the field.
    """
    x = np.asarray(radiance_ss)
    if x.dtype == np.float16:
        raise TypeError("radiance plane is float16 (non-negotiable #2)")
    layers = depth_layers(
        distance_m, focal_length_mm, f_number, focus_distance_m, sky_mask, max_layers
    )
    if not layers:
        return x
    colour = np.zeros(x.shape, dtype=np.float64)
    weight = np.zeros(x.shape, dtype=np.float64)
    for mask, representative_m in layers:
        if np.isinf(representative_m):
            c_um = (
                0.0
                if focus_distance_m is None
                else float(blur_circle_um(1e9, focal_length_mm, f_number, focus_distance_m))
            )
        else:
            c_um = float(
                blur_circle_um(representative_m, focal_length_mm, f_number, focus_distance_m)
            )
        kernel = bank.kernel_for(float(defocus_w020_um(c_um, f_number)))
        cover = mask.astype(np.float64)
        premultiplied = apply_psf(np.asarray(x, dtype=np.float64) * cover, kernel)
        alpha = np.clip(apply_psf(cover, kernel), 0.0, 1.0)
        colour = premultiplied + colour * (1.0 - alpha)
        weight = alpha + weight * (1.0 - alpha)
    out = np.divide(colour, weight, out=np.zeros_like(colour), where=weight > 1e-12)
    return np.asarray(out, dtype=x.dtype)
