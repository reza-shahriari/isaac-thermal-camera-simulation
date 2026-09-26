"""Information-based equalisation and Linear Percent (docs/physics-model.md §11.3, ADR 0147).

Global plateau equalisation clips a histogram bin only when one DN value holds more than
``plateau`` of the frame. A clear sky a few counts wide is such a spike; a sky of cumulus clutter
spread over thousands of DN is not, and on it the plateau never engages and the operator is full
histogram equalisation -- grey shades in proportion to pixel count. A drone covering 0.6 % of the
frame then gets 0.6 % of the ramp: three of 256 codes for parts spanning 15 → 38 °C (spec issue
S52). That is correct plateau behaviour. It is not what a Boson ships with.

A Boson's factory default is **Information-Based Equalization** blended by **Linear Percent**
[R51]. FLIR describes the operator qualitatively and publishes no equations, so this module is a
*flagged approximation* of it, built from the description:

1. **Split.** An edge-preserving low-pass ``x_LP`` (a 5×5 bilateral filter whose range sigma is
   FLIR's *Smoothing Factor*, read as DN of this core) and the high-pass ``x_HP = x − x_LP``. Steps
   much larger than the range sigma -- a drone against the sky -- stay in the low-pass; texture
   smaller than it goes to the high-pass.
2. **Histogram.** Plateau-clip the low-pass histogram at ``P·N`` as :func:`plateau_lut` does, then
   add an *information* histogram -- each bin's summed ``|x_HP|`` -- scaled so that at
   ``info_weight = 1`` it carries the same total mass as the clipped histogram. Bins holding detail
   earn shades by their detail rather than by their area. ``info_weight`` is not published; its
   default is a placeholder until a Tier 4 fit to public footage replaces it.
3. **Linear Percent.** The equalised transfer is blended with the min–max linear one,
   ``LUT = (1 − λ) LUT_eq + λ LUT_lin``, which restores *how much* hotter one object is than
   another at the cost of shades spent on empty DN.
4. **Detail Headroom and DDE.** The transfer is squeezed into ``[h, 1 − h]`` and the high-pass is
   added back at the transfer's own local slope, ``y = LUT(x_LP) + g · LUT'(x_LP) · x_HP``:
   detail is shown at the gain of its surroundings, ``g = 1`` reproduces the frame without
   sharpening, and the headroom is what keeps ``g · x_HP`` off the rails.

Identities the tests hold: with a vanishing range sigma the split is exact (``x_LP = x``), and then
``info_weight = 0``, ``linear_percent = 0``, ``headroom = 0``, ``g = 1`` is :func:`agc_plateau` bit
for bit; ``linear_percent = 1`` is the min–max linear map.

This is display only. The radiometric branch never passes through it, and it has no device (Warp)
implementation: the device AGC is a pure lookup table and this operator is spatial.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from irsim.isp.agc import CONSTANT_FRAME_LEVEL, _as_dn, histogram_dn, plateau_lut

__all__ = [
    "BILATERAL_RADIUS_PX",
    "BILATERAL_SIGMA_SPACE_PX",
    "bilateral_split",
    "linear_lut",
    "information_lut",
    "agc_information",
    "blend_linear",
    "equalise_image",
    "limit_gain",
]

Float32Array = NDArray[np.float32]

#: The bilateral window is 5×5: wide enough that a one-pixel texture is carried by its neighbours,
#: narrow enough to run on the CPU in a fraction of a second on a 640×512 frame.
BILATERAL_RADIUS_PX = 2
#: Spatial sigma of the bilateral weights, in pixels.
BILATERAL_SIGMA_SPACE_PX = 1.5


def bilateral_split(
    dn: NDArray[np.float64], sigma_range_dn: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """``(x_LP, x_HP)``: an edge-preserving 5×5 bilateral low-pass and its residual (§11.3).

    A step much larger than ``sigma_range_dn`` gets a weight of exp(−(Δ/σ_r)²/2) → 0 across it, so
    the step stays in the low-pass and the high-pass carries only the texture either side of it.
    As σ_r → 0 every off-centre weight underflows to exactly zero and ``x_LP == x`` bit for bit,
    which is the identity that ties this operator to :func:`agc_plateau`.
    """
    if sigma_range_dn <= 0.0:
        raise ValueError("the range sigma must be positive (DN)")
    r = BILATERAL_RADIUS_PX
    padded = np.pad(dn, r, mode="edge")
    acc = np.zeros_like(dn)
    wsum = np.zeros_like(dn)
    height, width = dn.shape
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            shifted = padded[r + dy : r + dy + height, r + dx : r + dx + width]
            spatial = np.exp(-(dy * dy + dx * dx) / (2.0 * BILATERAL_SIGMA_SPACE_PX**2))
            delta = shifted - dn
            w = spatial * np.exp(-0.5 * (delta / sigma_range_dn) ** 2)
            acc += w * delta
            wsum += w
    # Accumulated as a *deviation* from the centre, not as Σ w·s / Σ w: a neighbour equal to the
    # centre then contributes exactly zero, where the quotient form returns x − ε and a later
    # floor() drops the pixel one bin. The centre's own weight is 1, so wsum ≥ 1.
    low = dn + acc / wsum
    return low, dn - low


def linear_lut(counts: NDArray[np.float64]) -> NDArray[np.float64] | None:
    """The min–max linear transfer over the occupied bins, or ``None`` for a flat histogram."""
    occupied = np.flatnonzero(counts)
    if occupied.size == 0 or occupied[-1] == occupied[0]:
        return None
    lo, hi = float(occupied[0]), float(occupied[-1])
    return np.asarray(np.clip((np.arange(counts.size) - lo) / (hi - lo), 0.0, 1.0))


def information_lut(
    counts: NDArray[np.float64],
    plateau: float,
    info: NDArray[np.float64] | None = None,
    info_weight: float = 0.0,
    linear_percent: float = 0.0,
    clip_limit_low: float = 0.0,
) -> NDArray[np.float64] | None:
    """The bin → [0, 1] transfer of steps 2 and 3, or ``None`` when the frame has no range.

    With ``info_weight = 0`` and ``linear_percent = 0`` this *is* :func:`plateau_lut`: the same
    function is called and nothing is added to its result. ``clip_limit_low`` is passed to it
    unchanged (SC.25).
    """
    if info_weight < 0.0:
        raise ValueError("info_weight must be non-negative")
    if not 0.0 <= linear_percent <= 1.0:
        raise ValueError("linear_percent is a fraction in [0, 1]")
    c = np.asarray(counts, dtype=np.float64)
    if info is not None and info_weight > 0.0 and float(np.sum(info)) > 0.0:
        clipped = np.minimum(c, plateau * float(c.sum()))
        if clip_limit_low > 0.0:
            clipped = clipped + np.where(c > 0.0, clip_limit_low * float(c.sum()), 0.0)
        extra = info_weight * float(clipped.sum()) * np.asarray(info) / float(np.sum(info))
        # `plateau_lut` clips at P·N of what it is given; hand it the clipped counts plus the
        # information mass, with an N large enough that no bin is clipped a second time.
        combined = clipped + extra
        eq = _cdf_lut(combined, c)
    else:
        eq = plateau_lut(c, plateau, clip_limit_low)
    if eq is None:
        return None
    if linear_percent == 0.0:
        return eq
    lin = linear_lut(c)
    if lin is None:  # pragma: no cover - eq is None whenever lin is
        return eq
    return (1.0 - linear_percent) * eq + linear_percent * lin


def _cdf_lut(
    weights: NDArray[np.float64], counts: NDArray[np.float64]
) -> NDArray[np.float64] | None:
    """Exclusive-CDF transfer of already-clipped bin weights, normalised between the darkest and
    brightest *occupied* bins exactly as :func:`plateau_lut` normalises (ADR 0028)."""
    cdf_excl = np.concatenate(([0.0], np.cumsum(weights)[:-1]))
    occupied = np.flatnonzero(counts)
    if occupied.size == 0:
        return None
    lo, hi = cdf_excl[occupied[0]], cdf_excl[occupied[-1]]
    if hi <= lo:
        return None
    return np.asarray(np.clip((cdf_excl - lo) / (hi - lo), 0.0, 1.0))


def limit_gain(
    lut: NDArray[np.float64], counts: NDArray[np.float64], max_gain: float
) -> NDArray[np.float64]:
    """Cap the transfer's slope at ``max_gain`` display codes (of 255) per DN (SC.25, ADR 0149).

    Every equalising core has this control -- FLIR's *Max Gain*, Xenics' "maximal allowed
    stretching" -- because on a bland scene (a clear sky a few DN wide) equalisation stretches the
    whole ramp across the noise and shows the fixed pattern as the picture. Each bin's increment is
    capped, the transfer re-integrated, and when the cap bites the occupied range is centred on
    mid-grey, as a camera's mid-point default does. ``max_gain = 0`` means no cap; a cap that never
    binds returns ``lut`` itself.
    """
    if max_gain < 0.0:
        raise ValueError("max_gain must be non-negative (0 = no limit)")
    if max_gain == 0.0:
        return lut
    step = np.diff(lut)
    cap = max_gain / 255.0
    if float(step.max(initial=0.0)) <= cap:
        return lut
    cum = np.concatenate(([0.0], np.cumsum(np.minimum(step, cap))))
    occupied = np.flatnonzero(counts)
    lo, hi = occupied[0], occupied[-1]
    centred = 0.5 + cum - 0.5 * (cum[lo] + cum[hi])
    return np.asarray(np.clip(centred, 0.0, 1.0))


def equalise_image(
    x: object,
    plateau: float,
    bit_depth: int,
    *,
    linear_percent: float = 0.0,
    clip_limit_low: float = 0.0,
    max_gain: float = 0.0,
) -> Float32Array:
    """Global plateau equalisation with the cross-vendor controls (SC.21, SC.25).

    ``linear_percent`` (Boson), ``clip_limit_low`` (Lepton) and ``max_gain`` (both, and Xenics)
    on one transfer. All three at zero is :func:`agc_plateau` bit for bit.
    """
    dn = _as_dn(x, bit_depth)
    counts = histogram_dn(dn, bit_depth)
    lut = information_lut(counts, plateau, None, 0.0, linear_percent, clip_limit_low)
    if lut is None:
        return np.full(dn.shape, CONSTANT_FRAME_LEVEL, dtype=np.float32)
    lut = limit_gain(lut, counts, max_gain)
    return np.asarray(lut[np.floor(dn).astype(np.int64)], dtype=np.float32)


def blend_linear(y_eq: object, x: object, linear_percent: float, bit_depth: int) -> Float32Array:
    """Linear Percent for an operator that returns an image rather than a table (§11.3).

    ``(1 − λ) y_eq + λ · (floor(x) − min)/(max − min)``: the same blend as :func:`information_lut`,
    applied per pixel, because both terms are functions of the pixel's bin alone.
    """
    if not 0.0 <= linear_percent <= 1.0:
        raise ValueError("linear_percent is a fraction in [0, 1]")
    y = np.asarray(y_eq, dtype=np.float32)
    if linear_percent == 0.0:
        return y
    dn = _as_dn(x, bit_depth)
    lin = linear_lut(histogram_dn(dn, bit_depth))
    if lin is None:
        return y
    idx = np.floor(dn).astype(np.int64)
    out = (1.0 - linear_percent) * y.astype(np.float64) + linear_percent * lin[idx]
    return np.asarray(np.clip(out, 0.0, 1.0), dtype=np.float32)


def agc_information(
    x: object,
    plateau: float,
    bit_depth: int = 16,
    *,
    info_weight: float = 1.0,
    linear_percent: float = 0.0,
    detail_headroom: float = 0.0,
    detail_gain: float = 1.0,
    smoothing_sigma_dn: float = 1250.0,
    clip_limit_low: float = 0.0,
    max_gain: float = 0.0,
) -> Float32Array:
    """Information-based equalisation with Linear Percent, Detail Headroom and DDE (§11.3).

    ``detail_gain`` is the gain on the high-pass layer at the transfer's local slope: 1 shows the
    frame's own detail, above 1 sharpens (a Boson's DDE > 1), below 1 smooths.
    """
    if plateau <= 0.0:
        raise ValueError("plateau must be positive (fraction of N_pixels per bin)")
    if not 0.0 <= detail_headroom < 0.5:
        raise ValueError("detail_headroom is a fraction of the range at each end, in [0, 0.5)")
    if detail_gain < 0.0:
        raise ValueError("detail_gain must be non-negative")
    dn = _as_dn(x, bit_depth)
    low, high = bilateral_split(dn, smoothing_sigma_dn)
    low = np.clip(low, 0.0, float(2**bit_depth - 1))
    counts = histogram_dn(low, bit_depth)
    info = histogram_dn(low, bit_depth, np.abs(high)) if np.any(high) else None
    lut = information_lut(counts, plateau, info, info_weight, linear_percent, clip_limit_low)
    if lut is None:
        return np.full(dn.shape, CONSTANT_FRAME_LEVEL, dtype=np.float32)
    lut = limit_gain(lut, counts, max_gain)
    if detail_headroom > 0.0:
        lut = detail_headroom + (1.0 - 2.0 * detail_headroom) * lut
    idx = np.floor(low).astype(np.int64)
    y = lut[idx]
    if detail_gain > 0.0 and np.any(high):
        slope = np.gradient(lut)  # display units per DN
        y = y + detail_gain * slope[idx] * high
    return np.asarray(np.clip(y, 0.0, 1.0), dtype=np.float32)
