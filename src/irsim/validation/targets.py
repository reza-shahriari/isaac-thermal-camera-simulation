"""Target and sky statistics measured on a *display-domain* frame (ME.4, §15 T4).

ADR 0068's rule for this project: the public data that exists for sky targets is 8-bit,
post-AGC, codec-compressed video, so every comparison happens in DN8 and every statistic has to
survive that. Radiometric statistics would be better and there is nothing to compare them against.

What follows from that is the thing this module is built around: **a statistic that depends on the
absolute level is useless here**, because an unknown AGC has already rescaled it. So the sky
profile is normalised by its own horizon-to-zenith span and reported per degree in units of the
flat-sky σ; the clutter slope is a power-law exponent, which is scale-free by construction; and
the target statistic is SCR, a ratio.

The one statistic that is *not* scale-free is the most interesting: **edge asymmetry**. A moving
target on a bolometer trails, and the trail's length in pixels is a property of the camera and the
motion, not of the display mapping — so the shape survives an AGC even though the levels do not.
There is no closed-form inversion from it to τ (see :func:`asymmetry_vs_tau_curve` for why the
obvious one fails in both regimes), so a comparison inverts by interpolating against a curve built
for its own target size and window.

docs/physics-model.md §5.3, §8.3, §15 T4; ADR 0068
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "Box",
    "TargetStatistics",
    "target_statistics",
    "sky_profile",
    "clutter_slope",
    "edge_spread_function",
    "profile_centre_index",
    "edge_asymmetry",
    "asymmetry_vs_tau_curve",
    "size_from_range_px",
    "edge_width_px",
    "blur_sigma_px",
]


@dataclass(frozen=True)
class Box:
    """An axis-aligned target box in pixels: (x, y) of the top-left corner, then width/height."""

    x: int
    y: int
    width: int
    height: int

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("box dimensions must be positive")

    @property
    def area_px(self) -> int:
        return self.width * self.height

    def slice(self) -> tuple[slice, slice]:
        return slice(self.y, self.y + self.height), slice(self.x, self.x + self.width)

    def grown(self, margin: int) -> Box:
        return Box(
            self.x - margin, self.y - margin, self.width + 2 * margin, self.height + 2 * margin
        )


@dataclass(frozen=True)
class TargetStatistics:
    area_px: int
    mean_target: float
    mean_ring: float
    std_ring: float
    scr: float
    polarity: int  # +1 target brighter than its surround, -1 darker


def target_statistics(frame: Any, box: Box, ring_margin: int = 4) -> TargetStatistics:
    """Local SCR = (mean_target − mean_ring)/std_ring, and the polarity that comes with it.

    The ring is the annulus *around* the box, not the whole frame: a sky target's background is the
    sky right behind it, and a frame-wide background would mix in the horizon, the ground and any
    cloud on the other side of the picture.
    """
    image = np.asarray(frame, dtype=np.float64)
    if image.ndim != 2:
        raise ValueError(f"frame must be (H, W), got {image.shape}")
    rows, cols = box.slice()
    if rows.start < 0 or cols.start < 0 or rows.stop > image.shape[0] or cols.stop > image.shape[1]:
        raise ValueError("box lies outside the frame")
    outer = box.grown(ring_margin)
    o_rows = slice(max(0, outer.y), min(image.shape[0], outer.y + outer.height))
    o_cols = slice(max(0, outer.x), min(image.shape[1], outer.x + outer.width))
    mask = np.zeros(image.shape, dtype=bool)
    mask[o_rows, o_cols] = True
    mask[rows, cols] = False
    if not mask.any():
        raise ValueError("the ring is empty; widen ring_margin or move the box inside the frame")

    target = float(image[rows, cols].mean())
    ring = float(image[mask].mean())
    sigma = float(image[mask].std())
    scr = 0.0 if sigma == 0.0 else (target - ring) / sigma
    return TargetStatistics(
        area_px=box.area_px,
        mean_target=target,
        mean_ring=ring,
        std_ring=sigma,
        scr=scr,
        polarity=1 if target >= ring else -1,
    )


def sky_profile(
    frame: Any, horizon_row: int, zenith_row: int = 0, flat_sigma: float | None = None
) -> tuple[NDArray[np.float64], float]:
    """A **scale-free** sky column profile and its gradient in σ per degree-equivalent.

    Returns ``(shape, snr_per_row)``: the shape is the row means normalised so that the horizon
    is 1 and the zenith 0, which removes both the AGC's gain and its offset; the gradient is the
    DN8 change per row divided by the flat-sky σ, which says whether the gradient is *visible*
    rather than how many codes it spans.

    Both are necessary. An AGC that stretched the sky twice as hard would double the raw gradient
    and leave the shape and the SNR untouched.
    """
    image = np.asarray(frame, dtype=np.float64)
    if image.ndim != 2:
        raise ValueError(f"frame must be (H, W), got {image.shape}")
    lo, hi = sorted((int(zenith_row), int(horizon_row)))
    if not 0 <= lo < hi < image.shape[0]:
        raise ValueError("zenith and horizon rows must be distinct and inside the frame")
    rows = image[lo : hi + 1].mean(axis=1)
    span = (
        float(rows[-1] - rows[0])
        if int(horizon_row) > int(zenith_row)
        else float(rows[0] - rows[-1])
    )
    if span == 0.0:
        raise ValueError("the sky has no horizon-to-zenith span to normalise by")
    shape = (
        (rows - rows[0]) / span if int(horizon_row) > int(zenith_row) else (rows - rows[-1]) / span
    )
    sigma = float(np.std(image[lo : hi + 1] - rows[:, None])) if flat_sigma is None else flat_sigma
    gradient = float(np.mean(np.abs(np.diff(rows))))
    return np.asarray(shape), (0.0 if sigma == 0.0 else gradient / sigma)


def clutter_slope(patch: Any, min_cycles: int = 2) -> float:
    """β of a 1/f^β radially averaged power spectrum, fitted in log–log.

    Scale-free by construction: multiplying the patch by any constant shifts the fit's intercept
    and leaves β alone, which is what makes it usable on data an unknown AGC has already touched.
    """
    image = np.asarray(patch, dtype=np.float64)
    if image.ndim != 2 or min(image.shape) < 8:
        raise ValueError("clutter_slope needs a 2-D patch at least 8x8")
    centred = image - image.mean()
    power = np.abs(np.fft.fftshift(np.fft.fft2(centred))) ** 2
    rows, cols = np.indices(image.shape)
    radius = np.hypot(rows - image.shape[0] / 2.0, cols - image.shape[1] / 2.0)
    bins = np.arange(min_cycles, int(min(image.shape) / 2))
    if bins.size < 4:
        raise ValueError("patch too small for a slope fit")
    means = np.array([power[(radius >= r) & (radius < r + 1)].mean() for r in bins])
    good = means > 0.0
    if good.sum() < 4:
        raise ValueError("power spectrum has too many empty bins")
    slope = np.polyfit(np.log(bins[good]), np.log(means[good]), 1)[0]
    return float(-slope)


def edge_spread_function(
    frame: Any, box: Box, axis: int = 1, margin: int = 8
) -> NDArray[np.float64]:
    """The mean profile across a target, along ``axis``, extended by ``margin`` on each side."""
    image = np.asarray(frame, dtype=np.float64)
    grown = box.grown(margin)
    rows = slice(max(0, grown.y), min(image.shape[0], grown.y + grown.height))
    cols = slice(max(0, grown.x), min(image.shape[1], grown.x + grown.width))
    patch = image[rows, cols]
    return np.asarray(patch.mean(axis=0 if axis == 1 else 1))


def profile_centre_index(box: Box, margin: int, axis: int = 1) -> float:
    """Where the box's centre lands inside the profile :func:`edge_spread_function` returns.

    This exists so that a measurement and a calibration curve split their profiles at the *same*
    place. :func:`edge_asymmetry` will find a centroid for you, but the centroid is not a fixed
    landmark — it moves with the tail it is supposed to be measuring, and the resulting statistic
    is not monotone in τ (see :func:`asymmetry_vs_tau_curve`). The box centre is a landmark: it is
    where the detector says the target is, in both the simulated frame and the real one.

    ``margin`` is clipped at the frame edge by ``edge_spread_function``, so pass the same box and
    margin here and the index accounts for it.
    """
    lo = box.x if axis == 1 else box.y
    extent = box.width if axis == 1 else box.height
    start = max(0, lo - margin)
    return float(lo + (extent - 1) / 2.0 - start)


def edge_asymmetry(profile: Any, centre: float | None = None) -> float:
    """(trailing − leading) / (trailing + leading) energy about ``centre``.

    Zero for a symmetric target. Positive when the tail is on the high-index side. This is the
    **smear fingerprint**: a bolometer's membrane carries a moving target's history across frames,
    so the trailing side of its profile is longer than the leading one — and the effect is in the
    *shape*, which no AGC can rescale away.

    ``centre`` defaults to the profile's own centroid, which needs nothing but the profile. Prefer
    :func:`profile_centre_index` whenever a box is available: the centroid is dragged into the tail
    by the very smear being measured, which leaves the compact core on the *leading* side and gets
    the sign backwards. On a target whose trail demonstrably runs to high indices the centroid
    split reads −0.056, −0.117, −0.193, **−0.162**, −0.230 for τ of 0.5, 1, 2, 4 and 8 frames
    (σ = 3 px, 2 px/frame) — wrong in sign and not even monotone in magnitude.
    """
    values = np.asarray(profile, dtype=np.float64)
    if values.ndim != 1 or values.size < 5:
        raise ValueError("an edge profile needs at least five samples")
    excess = values - values.min()
    total = float(excess.sum())
    if total <= 0.0:
        raise ValueError("the profile is flat; there is no target in it")
    index = np.arange(values.size, dtype=np.float64)
    centroid = float((index * excess).sum() / total) if centre is None else float(centre)
    leading = float(excess[index < centroid].sum())
    trailing = float(excess[index > centroid].sum())
    if leading + trailing <= 0.0:
        return 0.0
    return (trailing - leading) / (trailing + leading)


def asymmetry_vs_tau_curve(
    tau_frames: Any,
    velocity_px_per_frame: float,
    target_sigma_px: float,
    margin_px: int = 120,
) -> NDArray[np.float64]:
    """Asymmetry for each τ, on a synthetic target of the caller's own size, speed and window.

    ⚠️ **There is no closed-form inversion from asymmetry to τ, and this replaces the one that was
    planned.** The obvious formula treats the target as a point and the trail as a geometric tail,
    and it is wrong for an extended one: at σ = 3 px the core carries far more energy than the
    tail, and a point-source inversion returns 0.45 frames for a true τ of 4. The asymmetry itself
    is a fine statistic — it just has to be read off a curve measured for the same target.

    Split about the box centre (:func:`profile_centre_index`), the curve is monotone in every
    regime tested: 0.083, 0.261, 0.496, 0.700, 0.836 for τ of 0.5, 1, 2, 4 and 8 frames at
    σ = 3 px and 2 px/frame, and still monotone for a point-like σ = 0.6 px (0.317 → 0.913) where
    a centroid split is not. Both the target size and the speed move the whole curve — halving the
    speed to 1 px/frame drops the same five points to 0.046 … 0.748 — so a curve is only valid for
    the geometry it was built with.

    The window is a real trap. ``margin_px`` truncates the tail, and a short window does not bias
    the curve so much as *compress* its top: the asymmetry gained between τ = 4 and τ = 16 falls
    from 0.212 at margin 120 to 0.110 at margin 6, so the same DN8 measurement error inverts to
    twice the spread in τ. Measure with the widest window the frame allows, and build the curve
    with that same window.
    """
    taus = np.atleast_1d(np.asarray(tau_frames, dtype=np.float64))
    if taus.ndim != 1 or taus.size == 0:
        raise ValueError("tau_frames must be a non-empty 1-D sequence")
    if np.any(taus <= 0.0):
        raise ValueError("tau_frames must be positive")
    if velocity_px_per_frame <= 0.0 or target_sigma_px <= 0.0:
        raise ValueError("velocity and target size must be positive")
    if margin_px < 5:
        raise ValueError("margin_px must leave at least five samples of window")
    out = []
    for tau in taus:
        # The target runs toward *low* columns, so its history — the trail — lies at high ones and
        # `edge_asymmetry`'s sign convention reads positive. Six time constants of history is
        # 0.25 % of the peak, below a DN8 code for any target that is not saturating.
        n_frames = int(math.ceil(6.0 * float(tau))) + 1
        trail_px = int(math.ceil(n_frames * velocity_px_per_frame)) + 5
        centre_col = float(margin_px + 5)
        # The frame is sized from the trail so the tail never runs off the *image*: a clipped image
        # would confound the window effect under study with an entirely separate one.
        width = int(centre_col + trail_px + margin_px + 10)
        height = int(max(24, 8 * target_sigma_px))
        rows, cols = np.mgrid[0:height, 0:width]
        ratio = math.exp(-1.0 / float(tau))
        frame = np.zeros((height, width), dtype=np.float64)
        for k in range(n_frames):
            offset = centre_col + k * velocity_px_per_frame
            frame += (ratio**k) * np.exp(
                -0.5 * (((rows - height / 2.0) ** 2 + (cols - offset) ** 2) / target_sigma_px**2)
            )
        box = Box(int(centre_col) - 4, int(height / 2) - 3, 9, 7)  # odd, so it straddles the target
        profile = edge_spread_function(frame, box, axis=1, margin=margin_px)
        out.append(edge_asymmetry(profile, centre=profile_centre_index(box, margin_px, axis=1)))
    return np.asarray(out)


def size_from_range_px(
    physical_size_m: float, range_m: Any, focal_length_mm: float, pitch_um: float
) -> NDArray[np.float64]:
    """Apparent size in pixels: f·s/(R·p). The relation a size-vs-range scatter must follow."""
    if physical_size_m <= 0.0 or focal_length_mm <= 0.0 or pitch_um <= 0.0:
        raise ValueError("size, focal length and pitch must be positive")
    r = np.asarray(range_m, dtype=np.float64)
    if np.any(r <= 0.0):
        raise ValueError("range must be positive")
    return np.asarray(focal_length_mm * 1e-3 * physical_size_m / (r * pitch_um * 1e-6))


def _sobel_magnitude(image: NDArray[np.float64]) -> NDArray[np.float64]:
    """|∇| by the 3 × 3 Sobel pair, scaled by 1/8 so a unit ramp reads 1 per pixel; edge-padded."""
    p = np.pad(image, 1, mode="edge")
    right = p[:-2, 2:] + 2.0 * p[1:-1, 2:] + p[2:, 2:]
    left = p[:-2, :-2] + 2.0 * p[1:-1, :-2] + p[2:, :-2]
    below = p[2:, :-2] + 2.0 * p[2:, 1:-1] + p[2:, 2:]
    above = p[:-2, :-2] + 2.0 * p[:-2, 1:-1] + p[:-2, 2:]
    gx, gy = (right - left) / 8.0, (below - above) / 8.0
    return np.asarray(np.hypot(gx, gy), dtype=np.float64)


def edge_width_px(frame: Any, box: Box, percentile: float = 99.0) -> float:
    """How soft the target's silhouette is, in pixels: its contrast over its steepest gradient.

    ``w = C / |∇|_p`` with ``C`` the larger of (99th percentile in the box − background) and
    (background − 1st percentile), the background the median of the frame within one box-size
    around it but outside the box, and ``|∇|_p`` the ``percentile``-th gradient magnitude inside
    the box. A step blurred by a Gaussian of σ pixels reads ``√(2π σ² + 4)`` to 4 %: 2.5 σ once σ
    passes a few pixels, and exactly 2 for a sharp edge, the central difference's own two-pixel
    span (:func:`blur_sigma_px` inverts it). It is a ratio of two
    display-domain quantities with the same units, so any linear display mapping cancels -- the
    property every statistic here needs (ADR 0068) -- and a clipped (saturated) core shortens it
    only as much as it shortens the contrast.

    EV.21 measured it on Anti-UAV RGBT's sky-only test frames (median 3.97 px, 10-90 % 2.9-6.6)
    against irsim's clip set (median 1.92 px, 1.6-2.1): the real targets carry a blur of about
    1.4 px σ, the renders none.
    """
    image = np.asarray(frame, dtype=np.float64)
    if image.ndim != 2:
        raise ValueError(f"frame must be (H, W), got {image.shape}")
    rows, cols = box.slice()
    if rows.start < 0 or cols.start < 0 or rows.stop > image.shape[0] or cols.stop > image.shape[1]:
        raise ValueError("box lies outside the frame")
    grow = max(box.width, box.height) // 2
    outer = box.grown(grow)
    o_rows = slice(max(0, outer.y), min(image.shape[0], outer.y + outer.height))
    o_cols = slice(max(0, outer.x), min(image.shape[1], outer.x + outer.width))
    patch = image[o_rows, o_cols]
    inside = np.zeros(patch.shape, dtype=bool)
    inside[rows.start - o_rows.start : rows.stop - o_rows.start,
           cols.start - o_cols.start : cols.stop - o_cols.start] = True  # fmt: skip
    if inside.all():
        raise ValueError("no background around the box: it fills the frame")
    background = float(np.median(patch[~inside]))
    target = patch[inside]
    contrast = max(
        float(np.percentile(target, 99.0)) - background,
        background - float(np.percentile(target, 1.0)),
    )
    steepest = float(np.percentile(_sobel_magnitude(patch)[inside], percentile))
    if contrast <= 0.0 or steepest <= 0.0:
        raise ValueError("the target has no contrast against its background")
    return contrast / steepest


def blur_sigma_px(edge_width: float) -> float:
    """The Gaussian σ, in pixels, that :func:`edge_width_px` reads as ``edge_width``.

    The inverse of ``w = √(2π σ² + 4)``; 0 at or below the sharp-edge floor of 2.
    """
    return math.sqrt(max(0.0, float(edge_width) ** 2 - 4.0) / (2.0 * math.pi))
