"""Defocus geometry: how far from the focal plane an object's image forms, and how wide the
resulting blur circle is (docs/physics-model.md §8.3).

§8.3 names ``MTF_defocus`` in the cascade and then declines to define it -- "in practice fit a
Gaussian ... rather than deriving it" -- so the spec gives the slot and no model. This module is
the geometry half of filling it: it produces the **one** parameter every defocus model is a
function of, and computes no kernels. `OC.2` turns that parameter into an OTF.

Two conventions fixed here:

* **W020** is the peak optical path difference at the edge of the exit pupil, in µm, and it is
  ``c / (8 F)`` for a blur circle of diameter ``c`` in air. It, not ``c``, is what decides which
  defocus model is valid, because the comparison is against the wavelength: geometric optics holds
  only for W020 > 2λ and Rayleigh's tolerance for "in focus" is W020 < λ/4.
* **A distance of zero is not a distance.** `irsim.config.gbuffer` gives sky pixels
  ``distance_m = 0`` (with ``sky_mask`` set) so that a consumer ignoring the mask still sees
  τ = 1. A depth-to-blur map that takes that literally reads the sky as the nearest thing in the
  scene and defocuses it hardest, which is the most likely way to ship a wrong-looking frame. Every
  entry point here **raises** on a non-positive distance rather than accepting one, so the caller
  has to route sky through ``focus_distance_m=None`` deliberately.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "blur_circle_um",
    "defocus_w020_um",
    "defocus_waves",
    "depth_of_field_m",
    "geometric_regime_blur_um",
    "hyperfocal_distance_m",
    "scene_defocus_um",
]

FloatArray = NDArray[np.float64]


def _checked_optics(focal_length_mm: float, f_number: float) -> tuple[float, float]:
    if focal_length_mm <= 0.0 or f_number <= 0.0:
        raise ValueError("focal length and f-number must be positive")
    return float(focal_length_mm), float(f_number)


def blur_circle_um(
    object_distance_m: object,
    focal_length_mm: float,
    f_number: float,
    focus_distance_m: float | None = None,
) -> FloatArray:
    """Blur-circle diameter (µm) for objects at ``object_distance_m``, lens focused at
    ``focus_distance_m`` (``None`` = infinity).

    Thin lens, exit pupil of diameter f/F, sensor fixed at the image of the focus distance::

        c = f² |s − s_f| / (F · s · (s_f − f))

    and its infinity limit ``c = f² / (F s)``. Zero exactly at ``s = s_f``, and monotone in
    ``|s − s_f|`` either side of it.

    Raises on a non-positive or sub-focal-length distance: a G-buffer's sky pixels carry
    ``distance_m = 0`` and must be routed to the infinity branch by the caller, not smeared.
    """
    f_mm, n = _checked_optics(focal_length_mm, f_number)
    s_m = np.asarray(object_distance_m, dtype=np.float64)
    f_m = f_mm * 1e-3
    if np.any(s_m <= f_m):
        raise ValueError(
            "object distance must exceed the focal length; sky pixels carry distance_m = 0 "
            "and must be routed through the infinity branch, not passed in here"
        )
    if focus_distance_m is None or np.isinf(focus_distance_m):
        c_m = f_m * f_m / (n * s_m)
    else:
        sf_m = float(focus_distance_m)
        if sf_m <= f_m:
            raise ValueError("focus distance must exceed the focal length")
        c_m = f_m * f_m * np.abs(s_m - sf_m) / (n * s_m * (sf_m - f_m))
    return np.asarray(c_m * 1e6, dtype=np.float64)


def defocus_w020_um(blur_circle_diameter_um: object, f_number: float) -> FloatArray:
    """W020 (µm of optical path difference at the pupil edge) for a blur circle of diameter ``c``.

    ``c = 8 F W020`` in air, so ``W020 = c / (8 F)``.
    """
    if f_number <= 0.0:
        raise ValueError("f-number must be positive")
    c = np.asarray(blur_circle_diameter_um, dtype=np.float64)
    if np.any(c < 0.0):
        raise ValueError("blur-circle diameter cannot be negative")
    return np.asarray(c / (8.0 * float(f_number)), dtype=np.float64)


def defocus_waves(
    blur_circle_diameter_um: object, f_number: float, wavelength_um: float
) -> FloatArray:
    """W020 in waves -- the number that decides which defocus model is valid.

    Below 0.25 the lens is "in focus" by Rayleigh's quarter-wave tolerance; above 2 the geometric
    disk is a fair approximation; between them only a diffraction model is right, and that band is
    where every camera in ``configs/sensors/`` spends its useful range in the long-wave bands.
    """
    if wavelength_um <= 0.0:
        raise ValueError("wavelength must be positive")
    return np.asarray(
        defocus_w020_um(blur_circle_diameter_um, f_number) / float(wavelength_um),
        dtype=np.float64,
    )


def geometric_regime_blur_um(f_number: float, wavelength_um: float) -> float:
    """The blur circle at which geometric optics becomes valid: ``c = 16 F λ`` (W020 = 2λ).

    At F/1.0 and 10.5 µm this is 168 µm -- fourteen pixels on a 12 µm pitch -- so the geometric
    disk model does not apply anywhere in a long-wave scene that still looks like an image.
    """
    _checked_optics(1.0, f_number)
    if wavelength_um <= 0.0:
        raise ValueError("wavelength must be positive")
    return 16.0 * float(f_number) * float(wavelength_um)


def hyperfocal_distance_m(focal_length_mm: float, f_number: float, coc_um: float) -> float:
    """``H = f²/(F c) + f``: focus here and everything from H/2 to infinity is within ``c``.

    ``coc_um`` is the *acceptable* circle of confusion and is a convention, not a measurement. One
    detector pitch is the usual choice for a sampled imager and is what a caller should pass unless
    it has a reason; the reason belongs in the config, not in a default here.
    """
    f_mm, n = _checked_optics(focal_length_mm, f_number)
    if coc_um <= 0.0:
        raise ValueError("acceptable circle of confusion must be positive")
    return float((f_mm * f_mm / (n * coc_um * 1e-3) + f_mm) * 1e-3)


def depth_of_field_m(
    focal_length_mm: float, f_number: float, coc_um: float, focus_distance_m: float | None = None
) -> tuple[float, float]:
    """Near and far limits (m) within which the blur circle stays under ``coc_um``.

    ``None`` focuses at infinity, whose near limit is the hyperfocal distance and whose far limit is
    infinite. Focusing at ``H`` returns ``(H/2, inf)``, which is the definition of hyperfocal.
    """
    f_mm, n = _checked_optics(focal_length_mm, f_number)
    h_m = hyperfocal_distance_m(f_mm, n, coc_um)
    f_m = f_mm * 1e-3
    if focus_distance_m is None or np.isinf(focus_distance_m):
        return (h_m, float("inf"))
    s_m = float(focus_distance_m)
    if s_m <= f_m:
        raise ValueError("focus distance must exceed the focal length")
    near = s_m * (h_m - f_m) / (h_m + s_m - 2.0 * f_m)
    far = float("inf") if s_m >= h_m else s_m * (h_m - f_m) / (h_m - s_m)
    return (float(near), float(far))


def scene_defocus_um(
    distance_m: object,
    focal_length_mm: float,
    f_number: float,
    focus_distance_m: float | None = None,
    sky_mask: object = None,
) -> float:
    """The single W020 (µm) one global kernel should carry for this frame (`OC.5`).

    The representative range is the **median** of the geometry in frame, not the mean: a frame that
    is nine-tenths sky and one-tenth foreground should be focused for the foreground, and a mean
    over a distance plane containing a horizon is dominated by whichever pixels happen to be far.

    Sky is excluded by ``sky_mask``, and — belt and braces — so is any non-positive or sub-focal
    distance, because `irsim.config.gbuffer` writes ``distance_m = 0`` on sky pixels and a median
    that includes them is a median of the wrong population. A frame with no geometry at all takes
    the defocus of an object at infinity, which is what it is looking at.
    """
    d = np.asarray(distance_m, dtype=np.float64)
    keep = np.isfinite(d) & (d > focal_length_mm * 1e-3)
    if sky_mask is not None:
        keep &= ~np.asarray(sky_mask, dtype=bool)
    representative = float(np.median(d[keep])) if np.any(keep) else 1e9
    return float(
        defocus_w020_um(
            blur_circle_um(representative, focal_length_mm, f_number, focus_distance_m), f_number
        )
    )
