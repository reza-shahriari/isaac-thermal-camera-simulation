"""The lens for the engine-free G-buffer: a measured calibration applied to an ideal image (SC.33).

docs/physics-model.md §8.4; ADR 0015 (the engine applies distortion), ADR 0164. In Isaac Sim the
lens-distortion schema on the camera prim distorts every AOV at render time, so the infrared frame
and its visible companion arrive distorted together and nothing here runs. The engine-free builder
has no prim: it renders a pinhole G-buffer, and this module is its lens. The planes are resampled
from the ideal grid onto the distorted one with the same OpenCV model and the same intrinsics the
prim would have carried (:class:`~irsim.optics.projection.Intrinsics`), so the CPU frame and the
Isaac frame of one camera file agree about where a ray lands.

The map runs backwards, as a resampler must: the pixel at ``p_d`` of the distorted image shows the
ideal ray ``undistort(p_d)`` (:func:`~irsim.optics.projection.undistort_normalised`), and the ideal
image is sampled there -- bilinearly for a float plane (radiance, temperature, distance and
elevation all interpolate) and nearest-neighbour for an integer or boolean plane (an id or a mask
has no half-way value). A source outside the ideal image, which a barrel lens reaches in its
corners, is clamped to the ideal image's edge rather than filled with a zero that would read as
0 K; :func:`source_positions` says where each pixel came from for a caller that must know.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from numpy.typing import NDArray

from irsim.config.sensor import DistortionSpec
from irsim.optics.projection import Intrinsics, undistort_normalised

__all__ = ["distort_image", "distort_planes", "is_identity", "resample", "source_positions"]


def is_identity(distortion: DistortionSpec) -> bool:
    """True when every coefficient is zero: a pinhole, which no resampling should touch."""
    return not any(float(c) != 0.0 for c in distortion.coeffs)


def source_positions(
    intrinsics: Intrinsics, distortion: DistortionSpec
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Where in the ideal image each distorted pixel's centre came from, in continuous pixels.

    Pixel ``(i, j)`` spans ``[j, j+1] × [i, i+1]`` with its centre at ``(j + 0.5, i + 0.5)``, the
    convention of :class:`Intrinsics`. Returns ``(xs, ys)`` of shape ``(height, width)``. A point
    the inverse could not converge for (a lens strong enough to fold the field) is given its own
    position, so it shows the ideal pixel under it rather than NaN.
    """
    h, w = int(intrinsics.height), int(intrinsics.width)
    u = np.arange(w, dtype=np.float64) + 0.5
    v = np.arange(h, dtype=np.float64) + 0.5
    xd, yd = np.meshgrid(
        (u - intrinsics.cx_px) / intrinsics.fx_px, (v - intrinsics.cy_px) / intrinsics.fy_px
    )
    x, y = undistort_normalised(xd, yd, distortion)
    xs = x * intrinsics.fx_px + intrinsics.cx_px
    ys = y * intrinsics.fy_px + intrinsics.cy_px
    own_x, own_y = np.meshgrid(u, v)
    bad = ~(np.isfinite(xs) & np.isfinite(ys))
    return np.where(bad, own_x, xs), np.where(bad, own_y, ys)


def resample(plane: NDArray[Any], xs: NDArray[np.float64], ys: NDArray[np.float64]) -> NDArray[Any]:
    """``plane`` sampled at the continuous positions ``(xs, ys)``: a float plane bilinearly, an
    integer or boolean plane nearest-neighbour. The dtype is kept, so float32 stays float32 and
    never narrower (CLAUDE.md #2); float16 is refused outright."""
    p = np.asarray(plane)
    if p.ndim != 2:
        raise ValueError(f"a plane is 2-D, got shape {p.shape}")
    if xs.shape != ys.shape:
        raise ValueError("xs and ys must share a shape")
    h, w = p.shape
    if not np.issubdtype(p.dtype, np.floating):
        j = np.clip(np.floor(xs), 0, w - 1).astype(np.int64)
        i = np.clip(np.floor(ys), 0, h - 1).astype(np.int64)
        return np.asarray(p[i, j], dtype=p.dtype)
    if p.dtype == np.float16:
        raise TypeError("a float16 plane has no place in the radiance path (CLAUDE.md #2)")
    x = np.clip(xs - 0.5, 0.0, w - 1.0)
    y = np.clip(ys - 0.5, 0.0, h - 1.0)
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    fx = x - x0
    fy = y - y0
    q = p.astype(np.float64)
    out = (
        q[y0, x0] * (1.0 - fx) * (1.0 - fy)
        + q[y0, x1] * fx * (1.0 - fy)
        + q[y1, x0] * (1.0 - fx) * fy
        + q[y1, x1] * fx * fy
    )
    return np.asarray(out, dtype=p.dtype)


def distort_image(
    image: NDArray[Any], intrinsics: Intrinsics, distortion: DistortionSpec
) -> NDArray[Any]:
    """One plane of the grid's shape seen through the lens; the same object back for a pinhole."""
    img = np.asarray(image)
    if is_identity(distortion):
        return img
    if img.shape != (int(intrinsics.height), int(intrinsics.width)):
        raise ValueError(f"image {img.shape} is not the {intrinsics.resolution[::-1]} grid")
    xs, ys = source_positions(intrinsics, distortion)
    return resample(img, xs, ys)


def distort_planes(
    planes: Mapping[str, Any], intrinsics: Intrinsics, distortion: DistortionSpec
) -> dict[str, Any]:
    """Every 2-D plane of the grid's shape resampled through the lens, each by its dtype's rule;
    anything else (a scalar, a plane of another shape) passed through untouched."""
    out = dict(planes)
    if is_identity(distortion):
        return out
    shape = (int(intrinsics.height), int(intrinsics.width))
    xs, ys = source_positions(intrinsics, distortion)
    for key, value in planes.items():
        if isinstance(value, np.ndarray) and value.shape == shape:
            out[key] = resample(value, xs, ys)
    return out
