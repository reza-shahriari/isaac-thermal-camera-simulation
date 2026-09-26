"""Radial shading of a frame: the bowl a featureless sky shows (SC.20, §8.2, §11.2).

A clear sky gives the AGC no contrast of its own, so whatever smooth shading the camera leaves --
the housing seen through the field (§8.2, ADR 0145) and its drift since the last shutter event
(§11.2, ADR 0148) -- is stretched over the whole display range. This module measures that shading
the same way on a rendered frame and on a real one, so the two can be compared.

**The fit.** The frame is modelled as a plane plus a paraboloid about the principal point,

    I(x, y) ≈ a + b x + c y + k r²,      r² = (x² + y²) / r_corner²,

by least squares, with pixels more than five robust sigmas off the fit dropped and the fit
repeated. The plane absorbs the sky's elevation gradient, which a pitched-up camera always sees;
``k`` is then the bowl, in the image's own units, from the centre to the corner. ``k < 0`` is a
bright centre (a housing that cooled since the shutter event, or a black-hot display); ``k > 0``
a dark one. ``radial_share`` is how much of the variance left after the plane the paraboloid
explains -- near 1 on a frame that is all bowl, near 0 on one with no radial structure.

The fit is units-agnostic and never assumes an 8-bit range: it works on radiance, DN or display
grey, which is what lets one number describe a simulated and a real frame.

docs/physics-model.md §8.2, §11.2, §15 (Tier 4)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = ["RadialFit", "radial_fit"]

#: Pixels further than this many robust sigmas from the fit are outliers (a hot pixel, a bird).
OUTLIER_SIGMAS = 5.0


@dataclass(frozen=True)
class RadialFit:
    """The bowl of one frame. ``k`` is centre-to-corner in the image's units."""

    k: float
    plane: tuple[float, float, float]  # a, b (per unit x), c (per unit y); x, y in r_corner units
    radial_share: float
    relative_depth: float  # |k| over the frame's 1-99 percentile span
    profile_r: NDArray[np.float64]  # bin centres, fraction of the corner radius
    profile: NDArray[np.float64]  # plane-removed azimuthal mean per bin
    outliers: int

    @property
    def sign(self) -> int:
        """−1 bright centre, +1 dark centre, 0 when there is no bowl to speak of."""
        return int(np.sign(self.k)) if self.radial_share > 0.05 else 0

    @property
    def monotonic(self) -> bool:
        """True when the plane-removed profile moves one way from centre to corner."""
        d = np.diff(self.profile)
        return bool(np.all(d <= 0.0) or np.all(d >= 0.0))


def _coords(
    shape: tuple[int, int], centre: tuple[float, float] | None
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    h, w = shape
    cy, cx = (h / 2.0, w / 2.0) if centre is None else centre
    yy, xx = np.mgrid[:h, :w]
    x = xx + 0.5 - cx
    y = yy + 0.5 - cy
    r_corner = float(np.hypot(max(cx, w - cx), max(cy, h - cy)))
    return x / r_corner, y / r_corner, (x * x + y * y) / (r_corner * r_corner)


def radial_fit(
    image: object,
    *,
    centre: tuple[float, float] | None = None,
    n_bins: int = 12,
    mask: object | None = None,
) -> RadialFit:
    """Fit plane + paraboloid to a single-channel frame; see the module docstring.

    ``centre`` is (row, column) of the principal point, default the frame centre. ``mask`` (bool,
    True = use) excludes regions that are not sky -- a horizon, a target.
    """
    img = np.asarray(image)
    if img.dtype == np.float16:
        raise TypeError("image is float16 (non-negotiable #2)")
    if img.ndim == 3:
        img = img[..., :3].mean(axis=-1) if img.shape[-1] >= 3 else img[..., 0]
    if img.ndim != 2:
        raise ValueError(f"expected a single-channel (H, W) frame, got shape {img.shape}")
    z = img.astype(np.float64)
    shape = (int(z.shape[0]), int(z.shape[1]))
    x, y, r2 = _coords(shape, centre)
    use = np.isfinite(z)
    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        if m.shape != shape:
            raise ValueError(f"mask shape {m.shape} != image shape {shape}")
        use &= m
    if use.sum() < 16:
        raise ValueError("too few usable pixels to fit a bowl")

    design = np.stack([np.ones_like(x), x, y, r2], axis=-1)
    coef = np.zeros(4)
    for _ in range(3):
        coef, *_ = np.linalg.lstsq(design[use], z[use], rcond=None)
        resid = z - design @ coef
        mad = float(np.median(np.abs(resid[use] - np.median(resid[use]))))
        sigma = 1.4826 * mad if mad > 0.0 else float(np.std(resid[use]))
        keep = np.abs(resid) <= OUTLIER_SIGMAS * max(sigma, 1e-12)
        new_use = np.isfinite(z) & keep if mask is None else (np.isfinite(z) & keep & m)
        if np.array_equal(new_use, use):
            break
        use = new_use
    a, b, c, k = (float(v) for v in coef)

    detrended = z - (a + b * x + c * y)
    var_before = float(np.var(detrended[use]))
    var_after = float(np.var((detrended - k * r2)[use]))
    share = 0.0 if var_before <= 0.0 else max(0.0, 1.0 - var_after / var_before)

    r = np.sqrt(r2)
    edges = np.linspace(0.0, float(r[use].max()) + 1e-12, n_bins + 1)
    centres, means = [], []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        sel = use & (r >= lo) & (r < hi)
        if sel.any():
            centres.append(0.5 * (lo + hi))
            means.append(float(detrended[sel].mean()))
    lo_p, hi_p = np.percentile(z[use], [1.0, 99.0])
    span = float(hi_p - lo_p)
    return RadialFit(
        k=k,
        plane=(a, b, c),
        radial_share=share,
        relative_depth=abs(k) / span if span > 0.0 else 0.0,
        profile_r=np.asarray(centres),
        profile=np.asarray(means),
        outliers=int(np.count_nonzero(np.isfinite(z) & ~use)),
    )
