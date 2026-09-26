"""Field angles and cos⁴θ natural vignetting from undistorted pinhole geometry.

For a rectilinear lens of focal length f, a point on the focal plane at radius r from the principal
point sees the scene at field angle θ = atan(r / f), and the irradiance falls as cos⁴θ =
(f² / (f² + r²))² (docs/physics-model.md §2, §8.1). Mechanical vignetting of real IR optics is
measured, not derived, and enters as an optional multiplicative map on top.

Geometry is **undistorted**: distortion is applied by the engine on the rendered image (ADR 0015),
so the core computes field angles of the ideal pinhole at pixel centres (i + 0.5). cos⁴ is a
rectilinear-lens result and must not be used with fisheye models (`kannala_brandt`, `ftheta`);
the config schema refuses that combination (M3.4).

docs/physics-model.md §2, §8.1
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "cos4_at_radius",
    "field_radius_map_mm",
    "field_angle_map",
    "cos4_field",
    "load_vignetting_map",
]

Float32Array = NDArray[np.float32]


def cos4_at_radius(radius_mm: object, focal_length_mm: float) -> NDArray[np.float64]:
    """cos⁴θ for a focal-plane radius r: (f² / (f² + r²))². Scalar or array, float64."""
    if not focal_length_mm > 0.0:
        raise ValueError("focal_length_mm must be positive")
    r2 = np.asarray(radius_mm, dtype=np.float64) ** 2
    f2 = focal_length_mm * focal_length_mm
    return np.asarray((f2 / (f2 + r2)) ** 2, dtype=np.float64)


def field_radius_map_mm(
    width: int,
    height: int,
    pitch_um: float,
    principal_point_px: tuple[float, float] | None = None,
    supersample: int = 1,
) -> NDArray[np.float64]:
    """Radius (mm) from the principal point at every sample centre, shape (H·s, W·s).

    ``principal_point_px`` is in native-pixel units, default the format centre (W/2, H/2).
    With ``supersample`` = s the grid has s×s samples per native pixel and the same physical
    extent; sample centres sit at (k + 0.5) / s native pixels.
    """
    if width <= 0 or height <= 0 or supersample <= 0 or pitch_um <= 0:
        raise ValueError("width, height, supersample and pitch_um must be positive")
    cx, cy = principal_point_px if principal_point_px is not None else (width / 2.0, height / 2.0)
    s = supersample
    xs = (np.arange(width * s, dtype=np.float64) + 0.5) / s - cx
    ys = (np.arange(height * s, dtype=np.float64) + 0.5) / s - cy
    pitch_mm = pitch_um * 1e-3
    x_mm = xs[None, :] * pitch_mm
    y_mm = ys[:, None] * pitch_mm
    return np.asarray(np.sqrt(x_mm * x_mm + y_mm * y_mm), dtype=np.float64)


def field_angle_map(
    width: int,
    height: int,
    pitch_um: float,
    focal_length_mm: float,
    principal_point_px: tuple[float, float] | None = None,
    supersample: int = 1,
) -> Float32Array:
    """θ_ij = atan(r / f) at every sample centre, radians, float32 (H·s, W·s)."""
    if not focal_length_mm > 0.0:
        raise ValueError("focal_length_mm must be positive")
    r = field_radius_map_mm(width, height, pitch_um, principal_point_px, supersample)
    return np.asarray(np.arctan(r / focal_length_mm), dtype=np.float32)


def cos4_field(
    width: int,
    height: int,
    pitch_um: float,
    focal_length_mm: float,
    principal_point_px: tuple[float, float] | None = None,
    supersample: int = 1,
    enabled: bool = True,
    measured_map: NDArray[np.floating] | None = None,
) -> Float32Array:
    """The multiplicative irradiance field: cos⁴θ (or ones when ``enabled`` is False), times an
    optional measured vignetting map (same shape, values in (0, 1]). float32 (H·s, W·s)."""
    shape = (height * supersample, width * supersample)
    if enabled:
        r = field_radius_map_mm(width, height, pitch_um, principal_point_px, supersample)
        field = cos4_at_radius(r, focal_length_mm)
    else:
        if not focal_length_mm > 0.0:
            raise ValueError("focal_length_mm must be positive")
        field = np.ones(shape, dtype=np.float64)
    if measured_map is not None:
        m = np.asarray(measured_map, dtype=np.float64)
        if m.shape != shape:
            raise ValueError(f"measured vignetting map shape {m.shape} != field shape {shape}")
        if np.any(m <= 0.0) or np.any(m > 1.0) or not np.all(np.isfinite(m)):
            raise ValueError("measured vignetting map values must lie in (0, 1]")
        field = field * m
    return np.asarray(field, dtype=np.float32)


def load_vignetting_map(path: str, width: int, height: int) -> NDArray[np.float64]:
    """A measured mechanical-vignetting map from a ``.npy`` file at the native detector grid.

    ``optics.vignetting_map`` in a sensor config; the loader has already resolved the path against
    the data root. The map multiplies cos⁴ -- it is the part of the relative illumination a lens
    drawing cannot give (§2, §8.1) -- so it is normalised by the caller's measurement to 1 on axis,
    and values must lie in (0, 1]. float16 is refused (non-negotiable #2).
    """
    m = np.load(path, allow_pickle=False)
    if m.dtype == np.float16:
        raise TypeError(f"{path}: vignetting map is float16 (non-negotiable #2)")
    if m.shape != (height, width):
        grid = (height, width)
        raise ValueError(f"{path}: vignetting map shape {m.shape} != detector grid {grid}")
    return np.asarray(m, dtype=np.float64)


def format_corner_cos4(width: int, height: int, pitch_um: float, focal_length_mm: float) -> float:
    """cos⁴ at the extreme corner of the format (the *edge* of the corner pixel), for reports."""
    half_w = width * pitch_um * 1e-3 / 2.0
    half_h = height * pitch_um * 1e-3 / 2.0
    return float(cos4_at_radius(math.hypot(half_w, half_h), focal_length_mm))
