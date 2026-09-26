"""Automatic gain control: 16-bit linear → [0, 1] display range (docs/physics-model.md §11.3).

The mapping from a 14-16 bit radiometric image to 8 bits is **part of the sensor model**: it
changes the image drastically, it is global (a hot exhaust entering the frame collapses the
contrast of everything else), and it must match between training and deployment (§15 Tier 5).
Two operators, both global and histogram-based on 2^bit_depth integer bins so that a GPU port
and a real core compute the same thing (ADR 0027, 0028):

* :func:`agc_linear` -- percentile clipping with in-bin linear interpolation of the CDF, then
  optional gamma: y = clip((x − x_lo)/(x_hi − x_lo), 0, 1)^(1/γ);
* :func:`agc_plateau` -- histogram equalisation with every bin's count clipped at
  P = plateau · N_pixels before the CDF is integrated. Small P → the rank map of the occupied
  bins (a min-max stretch on a dense histogram); P ≥ max count → full equalisation. This is the
  algorithm behind the characteristic thermal "look".

Both are global, and being global is the problem they have: one hot object sets the stretch for
every pixel in the frame. §11.3's two answers to that are here too (M9.10), and **neither is the
default** -- a real core ships the global operator and so does this one.

* **ROI weighting.** Every histogram here accepts per-pixel ``weights``, so the stretch can be
  chosen for the pixels that matter rather than for the ones that are numerous. Uniform weights
  reproduce the global operator bit for bit, which is the property that makes this safe to add.
* :func:`agc_plateau_local` -- the frame is tiled, each tile equalises its own histogram, and the
  four nearest tile mappings are blended **bilinearly** at every pixel (the CLAHE construction).
  One tile reproduces :func:`agc_plateau` exactly. This is what keeps the background readable while
  an exhaust plume is in frame, and the blend is what keeps the tile boundaries from showing.

Inputs are uint16 DN or float32 in [0, 2^bit_depth − 1]; float16 is refused; outputs float32.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "histogram_dn",
    "percentile_from_histogram",
    "agc_linear",
    "agc_plateau",
    "agc_plateau_local",
    "plateau_lut",
    "tile_bounds",
    "CONSTANT_FRAME_LEVEL",
    "DEFAULT_TILES",
]

Float32Array = NDArray[np.float32]

# A frame with no dynamic range (x_hi == x_lo) displays as mid-grey, as real cores do.
CONSTANT_FRAME_LEVEL = 0.5

#: Tiling for the local operator when a caller does not say. Eight across a 640-wide frame is an
#: 80 px tile -- large enough that a tile's histogram is not dominated by one object, small enough
#: that a plume occupies a few tiles rather than all of them.
DEFAULT_TILES = (8, 8)


def _as_dn(x: object, bit_depth: int) -> NDArray[np.float64]:
    if not 8 <= bit_depth <= 16:
        raise ValueError("bit_depth must be in 8..16")
    arr = np.asarray(x)
    if arr.dtype == np.float16:
        raise TypeError("AGC input is float16 (non-negotiable #2); use uint16 or float32")
    if arr.ndim != 2:
        raise ValueError("AGC works on one (H, W) frame")
    if not (np.issubdtype(arr.dtype, np.integer) or arr.dtype in (np.float32, np.float64)):
        raise TypeError(f"AGC input must be uint16 or float32, got {arr.dtype}")
    a = arr.astype(np.float64)
    top = float(2**bit_depth - 1)
    if not np.all(np.isfinite(a)):
        raise ValueError("AGC input contains NaN or inf")
    if a.min() < 0.0 or a.max() > top:
        raise ValueError(f"AGC input outside [0, {top:.0f}] for bit_depth {bit_depth}")
    return a


def histogram_dn(
    dn: NDArray[np.float64], bit_depth: int, weights: object = None
) -> NDArray[np.float64]:
    """Counts per integer DN bin (2^bit_depth bins); float inputs are floored to their bin.

    ``weights`` gives each pixel a say in the stretch proportional to its weight rather than one
    vote each -- a region-of-interest mask, or a soft weighting towards the centre of the field.
    Returned as float64 either way: with weights of exactly 1.0 the sums are integers and every
    downstream comparison lands identically, which is what
    ``test_uniform_weights_reproduce_the_global_operator_bit_for_bit`` holds.
    """
    n_bins = 2**bit_depth
    idx = np.floor(dn).astype(np.int64)
    if weights is None:
        return np.bincount(idx.ravel(), minlength=n_bins).astype(np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if w.shape != dn.shape:
        raise ValueError(f"weights {w.shape} do not match the frame {dn.shape}")
    if np.any(w < 0.0) or not np.all(np.isfinite(w)):
        raise ValueError("weights must be finite and non-negative")
    if w.sum() <= 0.0:
        raise ValueError("weights are all zero: no pixel would set the stretch")
    return np.asarray(np.bincount(idx.ravel(), weights=w.ravel(), minlength=n_bins))


def percentile_from_histogram(counts: object, p: float) -> float:
    """Value at fraction ``p`` of the pixels with linear interpolation inside the bin
    (the value ``v`` such that ``p·N`` pixels lie below it, pixels spread uniformly within
    their bin). For a ramp with one pixel per bin this equals p·N up to the in-bin offset."""
    if not 0.0 <= p <= 1.0:
        raise ValueError("percentile fraction must lie in [0, 1]")
    counts = np.asarray(counts, dtype=np.float64)
    n = float(counts.sum())
    if n == 0.0:
        raise ValueError("empty histogram")
    target = p * n
    cdf = np.cumsum(counts)
    b = int(np.searchsorted(cdf, target, side="left"))
    b = min(b, int(counts.size) - 1)
    below = float(cdf[b - 1]) if b > 0 else 0.0
    c = float(counts[b])
    frac = (target - below) / c if c > 0 else 0.0
    return b + min(max(frac, 0.0), 1.0)


def agc_linear(
    x: object,
    p_lo: float,
    p_hi: float,
    gamma: float = 1.0,
    bit_depth: int = 16,
    weights: object = None,
) -> Float32Array:
    """Linear AGC with percentile clipping and gamma, global over the frame (§11.3).

    ``weights`` moves the two percentiles onto the pixels a caller cares about; the mapping it
    produces still applies to the whole frame, so an ROI sets the stretch without cropping it.
    """
    if not 0.0 <= p_lo < p_hi <= 1.0:
        raise ValueError("need 0 <= p_lo < p_hi <= 1")
    if gamma <= 0.0:
        raise ValueError("gamma must be positive")
    dn = _as_dn(x, bit_depth)
    counts = histogram_dn(dn, bit_depth, weights)
    x_lo = percentile_from_histogram(counts, p_lo)
    x_hi = percentile_from_histogram(counts, p_hi)
    if x_hi <= x_lo or int(np.floor(x_hi)) == int(np.floor(x_lo)):
        # both percentiles fall in one integer bin: no dynamic range at DN resolution
        return np.full(dn.shape, CONSTANT_FRAME_LEVEL, dtype=np.float32)
    y = np.clip((dn - x_lo) / (x_hi - x_lo), 0.0, 1.0)
    if gamma != 1.0:
        y = y ** (1.0 / gamma)
    return np.asarray(y, dtype=np.float32)


def agc_plateau(
    x: object, plateau: float, bit_depth: int = 16, weights: object = None
) -> Float32Array:
    """Plateau-equalisation AGC (§11.3, ADR 0028): each of the 2^bit_depth bins' counts is
    clipped at P = plateau · N_pixels, the clipped counts are integrated into an exclusive CDF,
    and pixels map to (CDF(bin) − CDF(bin_min)) / (CDF(bin_max) − CDF(bin_min)), so the darkest
    occupied bin is 0 and the brightest is 1."""
    if plateau <= 0.0:
        raise ValueError("plateau must be positive (fraction of N_pixels per bin)")
    dn = _as_dn(x, bit_depth)
    counts = histogram_dn(dn, bit_depth, weights)
    lut = plateau_lut(counts, plateau)
    if lut is None:
        return np.full(dn.shape, CONSTANT_FRAME_LEVEL, dtype=np.float32)
    idx = np.floor(dn).astype(np.int64)
    return np.asarray(lut[idx], dtype=np.float32)


def plateau_lut(
    counts: object, plateau: float, clip_limit_low: float = 0.0
) -> NDArray[np.float64] | None:
    """The bin → [0, 1] mapping :func:`agc_plateau` applies, or ``None`` for a flat histogram.

    Factored out because the local operator needs one of these per tile and must blend four of
    them per pixel; sharing the function is what makes "one tile equals the global operator"
    an identity rather than a coincidence.

    ``clip_limit_low`` (SC.25, ADR 0149) is the Lepton-family control: ``clip_limit_low · N`` is
    added to every *occupied* bin after the high clip, so sparsely populated temperatures -- a
    small target's -- are guaranteed a floor of shades however few pixels hold them. Zero adds
    0.0 to every bin, which changes no bits.
    """
    if plateau <= 0.0:
        raise ValueError("plateau must be positive (fraction of N_pixels per bin)")
    if clip_limit_low < 0.0:
        raise ValueError("clip_limit_low must be non-negative (fraction of N_pixels per bin)")
    c = np.asarray(counts, dtype=np.float64)
    n = float(c.sum())
    clipped = np.minimum(c, plateau * n)
    if clip_limit_low > 0.0:
        clipped = clipped + np.where(c > 0.0, clip_limit_low * n, 0.0)
    cdf_excl = np.concatenate(([0.0], np.cumsum(clipped)[:-1]))
    occupied = np.flatnonzero(c)
    if occupied.size == 0:
        return None
    lo, hi = cdf_excl[occupied[0]], cdf_excl[occupied[-1]]
    if hi <= lo:
        return None
    return np.asarray(np.clip((cdf_excl - lo) / (hi - lo), 0.0, 1.0))


def tile_bounds(size: int, tiles: int) -> NDArray[np.int64]:
    """Tile edges along one axis: ``tiles + 1`` indices spanning ``size``.

    ``np.linspace`` rather than a fixed stride, so a frame whose height is not a multiple of the
    tile count still gets tiles that differ by at most one row instead of a ragged last one.
    """
    if tiles < 1:
        raise ValueError("tiles must be at least 1")
    if size < tiles:
        raise ValueError(f"cannot cut {size} pixels into {tiles} tiles")
    return np.linspace(0, size, tiles + 1).astype(np.int64)


def _blend_weights(
    size: int, bounds: NDArray[np.int64]
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """Per-pixel (lower tile index, fraction towards the next) along one axis.

    Tile *centres* carry the mapping and pixels between two centres interpolate; pixels outside
    the outermost centres take their own tile's mapping whole. That is the CLAHE construction, and
    the clamping at the edges is why the border rows do not darken.
    """
    n_tiles = int(bounds.size) - 1
    index = np.arange(size, dtype=np.float64)
    if n_tiles == 1:
        return np.zeros(size, dtype=np.int64), np.zeros(size, dtype=np.float64)
    centres = 0.5 * (bounds[:-1] + bounds[1:] - 1).astype(np.float64)
    t = np.interp(index, centres, np.arange(n_tiles, dtype=np.float64))
    lower = np.clip(np.floor(t).astype(np.int64), 0, n_tiles - 2)
    return lower, t - lower


def agc_plateau_local(
    x: object,
    plateau: float,
    tiles: tuple[int, int] = DEFAULT_TILES,
    bit_depth: int = 16,
    weights: object = None,
    clip_limit_low: float = 0.0,
) -> Float32Array:
    """Locally-adaptive plateau equalisation: per-tile CDFs blended bilinearly (§11.3, M9.10).

    Each tile equalises its **own** histogram, so a plume that dominates the frame's histogram
    dominates only the tiles it actually covers and the background keeps its contrast. The price
    is that display level no longer means one thing across the frame -- two pixels of equal DN in
    different tiles can display differently -- so this is a *display* choice and never touches the
    radiometric branch.

    The four tile mappings nearest each pixel are blended bilinearly. Without that blend each tile
    boundary is a visible step; with it, ``tiles=(1, 1)`` also reduces to :func:`agc_plateau`
    exactly, because the single tile's weight is 1.0 and multiplying by one changes no bits.
    """
    ty, tx = int(tiles[0]), int(tiles[1])
    dn = _as_dn(x, bit_depth)
    height, width = dn.shape
    rows, cols = tile_bounds(height, ty), tile_bounds(width, tx)
    w = None if weights is None else np.asarray(weights, dtype=np.float64)
    if w is not None and w.shape != dn.shape:
        raise ValueError(f"weights {w.shape} do not match the frame {dn.shape}")

    idx = np.floor(dn).astype(np.int64)
    row_lower, row_frac = _blend_weights(height, rows)
    col_lower, col_frac = _blend_weights(width, cols)
    out = np.zeros(dn.shape, dtype=np.float64)
    covered = np.zeros(dn.shape, dtype=np.float64)

    for i in range(ty):
        wy = np.where(row_lower == i, 1.0 - row_frac, 0.0)
        if ty > 1:
            wy += np.where(row_lower + 1 == i, row_frac, 0.0)
        if not wy.any():
            continue
        for j in range(tx):
            wx = np.where(col_lower == j, 1.0 - col_frac, 0.0)
            if tx > 1:
                wx += np.where(col_lower + 1 == j, col_frac, 0.0)
            if not wx.any():
                continue
            weight = wy[:, None] * wx[None, :]
            tile = dn[rows[i] : rows[i + 1], cols[j] : cols[j + 1]]
            tile_w = None if w is None else w[rows[i] : rows[i + 1], cols[j] : cols[j + 1]]
            if tile_w is not None and tile_w.sum() <= 0.0:
                tile_w = None  # an ROI that misses this tile entirely: fall back to every pixel
            lut = plateau_lut(histogram_dn(tile, bit_depth, tile_w), plateau, clip_limit_low)
            contribution = CONSTANT_FRAME_LEVEL if lut is None else lut[idx]
            out += weight * contribution
            covered += weight
    if not np.allclose(covered, 1.0, atol=1e-9):
        raise AssertionError("tile blend weights do not sum to one")  # pragma: no cover
    return np.asarray(np.clip(out, 0.0, 1.0), dtype=np.float32)
