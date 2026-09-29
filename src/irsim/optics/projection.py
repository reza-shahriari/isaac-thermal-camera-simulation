"""Lens projection: where a ray lands on the focal plane, for each §8.4 distortion model.

docs/physics-model.md §8.4 [R23]; roadmap M10.9a; ADR 0015 (the engine applies distortion, the
imaging path stays rectilinear) and its M10.9a addendum.

**This module does not warp anything.** ADR 0015 stands: the renderer owns the projection, and
``irsim.optics.vignetting`` keeps computing field angles from undistorted pinhole geometry, because
re-warping a G-buffer would interpolate ids and temperatures that must never be interpolated
(ADR 0014). What lives here is the *forward model* of the lens the engine is handed -- the oracle
that answers "given this ``optics.distortion`` block, where should a ray at this angle land?".

Two things need that answer and neither is a warp:

1. **Checking the engine got the lens we meant.** ``optics.distortion`` is written onto the camera
   prim as an ``OmniLensDistortion*API`` schema (M10.9a). Writing coefficients into USD and
   trusting the picture is how a lens silently ends up a few percent off; with a forward model,
   a rendered grid can be reprojected through the config and the disagreement measured in pixels.
2. **Point targets and validation.** Phase 1's targets are sub-pixel (MS.6) and the Tier 4
   statistics are size- and position-versus-range: both ask where a target of known geometry
   should appear, which is a projection question, not an imaging one.

**Frames.** USD cameras look down **−Z** with +Y up; OpenCV -- whose model the engine implements --
looks down **+Z** with +Y down. Every distortion formula here is in the OpenCV frame, and
:func:`usd_camera_to_opencv` is the one place the flip happens. Getting this wrong inverts the
tangential terms and flips the image vertically while leaving a radially symmetric lens looking
perfectly fine, so the conversion is explicit and tested rather than folded into a sign somewhere.

**Which model is which**, measured on the 6.1.0-rc.26 build (``scripts/probe_isaac_camera.py``):

===================  ==============================  ==================================
config ``model``     USD API schema                  coefficients
===================  ==============================  ==================================
``brown_conrady``    ``OmniLensDistortionOpenCv      ``[k1, k2, p1, p2, k3]`` -- the
                     PinholeAPI``                    first five of OpenCV's own 12-term
                                                     rational-polynomial + thin-prism
                                                     order, so the map is positional
``kannala_brandt``   ``OmniLensDistortionOpenCv      ``[k1, k2, k3, k4]`` on θ
                     FisheyeAPI``
``ftheta``           ``OmniLensDistortionFthetaAPI``  **refused** -- see below
===================  ==============================  ==================================

The first two are verified against the engine's own implementations in
``tests/integration/test_camera_distortion_isaac.py``.

**f-theta is refused, deliberately.** Its schema exposes ``k0..k4`` alongside ``nominalWidth``,
``nominalHeight`` and ``opticalCenter``, and two things about it are undetermined by anything this
build exposes: whether the polynomial returns a radius in *pixels of the nominal image* or in
normalised units, and whether ``k0`` is a constant term (making a one-coefficient block a constant
radius, which is not a lens) or the coefficient of θ¹. Guessing either way yields a projection
that renders a perfectly plausible fisheye image while disagreeing with the engine by tens of
pixels at the field edge -- the exact failure this module exists to catch. Both this model and the
USD writer therefore raise until the convention is *measured*: render a grid through an f-theta
camera with one coefficient set and fit r(θ) (M10.9b's renderer audit). No camera in
``configs/sensors/`` uses f-theta, so nothing is blocked meanwhile.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from irsim.config.sensor import DistortionSpec, SensorSpec

__all__ = [
    "Intrinsics",
    "OPENCV_PINHOLE_COEFFS",
    "usd_camera_to_opencv",
    "normalise",
    "distort_normalised",
    "undistort_normalised",
    "project",
    "project_usd",
    "opencv_pinhole_coeffs",
    "FTHETA_UNVERIFIED",
]

#: OpenCV's own coefficient order for the 12-term rational-polynomial + thin-prism model, which is
#: also the order of the ``omni:lensdistortion:opencvPinhole:*`` attributes (measured). A 5-term
#: ``brown_conrady`` block is the first five of these, so no reordering is needed anywhere.
OPENCV_PINHOLE_COEFFS: tuple[str, ...] = (
    "k1",
    "k2",
    "p1",
    "p2",
    "k3",
    "k4",
    "k5",
    "k6",
    "s1",
    "s2",
    "s3",
    "s4",
)

_UNDISTORT_ITERS = 20
_UNDISTORT_TOL = 1e-12


@dataclass(frozen=True)
class Intrinsics:
    """Pinhole intrinsics in pixels, at whatever grid they were built for.

    ``fx = f / pitch`` is exact for a square-pixel FPA: both are lengths on the focal plane, so
    the millimetres cancel. The principal point defaults to the **format centre** ``(W/2, H/2)``
    in pixel-edge coordinates, which is where a pixel *corner* sits -- pixel ``i`` spans
    ``[i, i+1]`` and its centre is at ``i + 0.5``, the same convention
    ``irsim.optics.vignetting.field_radius_map_mm`` uses. Mixing the two puts every field angle
    half a pixel out, which is 0.4 % of the Boson's corner radius and invisible in an image.
    """

    fx_px: float
    fy_px: float
    cx_px: float
    cy_px: float
    width: int
    height: int

    def __post_init__(self) -> None:
        if not (self.fx_px > 0.0 and self.fy_px > 0.0):
            raise ValueError("focal lengths in pixels must be positive")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("width and height must be positive")

    @classmethod
    def from_sensor(cls, spec: SensorSpec, supersample: int = 1) -> Intrinsics:
        """Intrinsics of ``spec`` on the ``supersample``× grid the renderer is asked for.

        Supersampling subdivides each detector pixel without changing the optics, so the focal
        length in *pixels* and the principal point both scale by ``s`` while the field of view
        does not. That is the whole reason the render product is created at ``s × native``
        (§8.3): the aliasing is then the detector's, not the renderer's.
        """
        if supersample <= 0:
            raise ValueError("supersample must be positive")
        f_px = spec.optics.focal_length_mm / (spec.fpa.pitch_um * 1e-3) * supersample
        width = spec.fpa.width * supersample
        height = spec.fpa.height * supersample
        geo = None if spec.calibration is None else spec.calibration.geometric
        if geo is not None:
            # SC.33 (ADR 0164): the measured pinhole, scaled onto the k× grid as the designed one
            return cls(
                fx_px=geo.fx_px * supersample,
                fy_px=geo.fy_px * supersample,
                cx_px=geo.cx_px * supersample,
                cy_px=geo.cy_px * supersample,
                width=width,
                height=height,
            )
        return cls(
            fx_px=f_px,
            fy_px=f_px,
            cx_px=width / 2.0,
            cy_px=height / 2.0,
            width=width,
            height=height,
        )

    @property
    def resolution(self) -> tuple[int, int]:
        """``(width, height)`` -- the order the renderer's render product wants."""
        return (self.width, self.height)


def opencv_pinhole_coeffs(spec: DistortionSpec) -> dict[str, float]:
    """``{k1: …, k2: …, p1: …}`` for the OpenCV pinhole schema, zero-padded to all 12 terms.

    Raises for a model that is not a rational-polynomial one -- a fisheye's coefficients are a
    polynomial in θ, not in r, and writing them into the pinhole schema would produce a lens that
    is wrong by tens of pixels at the field edge while still rendering a plausible picture.
    """
    if spec.model != "brown_conrady":
        raise ValueError(
            f"{spec.model!r} is not a rational-polynomial model; its coefficients are not "
            "OpenCV pinhole coefficients"
        )
    values = dict.fromkeys(OPENCV_PINHOLE_COEFFS, 0.0)
    for name, value in zip(OPENCV_PINHOLE_COEFFS, spec.coeffs, strict=False):
        values[name] = float(value)
    return values


def usd_camera_to_opencv(points: Any) -> NDArray[np.float64]:
    """USD camera space (+X right, +Y up, −Z forward) → OpenCV (+X right, +Y down, +Z forward).

    Applied to positions or to directions alike -- it is a pure axis flip, ``(X, −Y, −Z)``.
    """
    p = np.asarray(points, dtype=np.float64)
    if p.shape[-1] != 3:
        raise ValueError(f"points must have a trailing axis of 3, got {p.shape}")
    return np.stack([p[..., 0], -p[..., 1], -p[..., 2]], axis=-1)


def normalise(points_cv: Any) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Perspective division in the OpenCV frame: ``(x/z, y/z)`` for points in front of the camera.

    A point at or behind the camera (``z <= 0``) has no image under a projective model, so it
    comes back as NaN rather than as a plausible pixel on the wrong side of the frame.
    """
    p = np.asarray(points_cv, dtype=np.float64)
    if p.shape[-1] != 3:
        raise ValueError(f"points must have a trailing axis of 3, got {p.shape}")
    z = p[..., 2]
    ahead = z > 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        x = np.where(ahead, p[..., 0] / z, np.nan)
        y = np.where(ahead, p[..., 1] / z, np.nan)
    return np.asarray(x), np.asarray(y)


def _rational_polynomial(
    x: NDArray[np.float64], y: NDArray[np.float64], c: dict[str, float]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """OpenCV's rational-polynomial radial + tangential + thin-prism distortion (§8.4)."""
    r2 = x * x + y * y
    r4 = r2 * r2
    r6 = r4 * r2
    radial = (1.0 + c["k1"] * r2 + c["k2"] * r4 + c["k3"] * r6) / (
        1.0 + c["k4"] * r2 + c["k5"] * r4 + c["k6"] * r6
    )
    xd = (
        x * radial
        + 2.0 * c["p1"] * x * y
        + c["p2"] * (r2 + 2.0 * x * x)
        + c["s1"] * r2
        + c["s2"] * r4
    )
    yd = (
        y * radial
        + c["p1"] * (r2 + 2.0 * y * y)
        + 2.0 * c["p2"] * x * y
        + c["s3"] * r2
        + c["s4"] * r4
    )
    return xd, yd


def _theta_d(theta: NDArray[np.float64], coeffs: list[float]) -> NDArray[np.float64]:
    """Kannala–Brandt θ_d = θ + k1θ³ + k2θ⁵ + k3θ⁷ + k4θ⁹ (odd powers only, §8.4)."""
    out = theta.copy()
    power = theta
    for k in coeffs[:4]:
        power = power * theta * theta  # θ³, then θ⁵, θ⁷, θ⁹
        out = out + float(k) * power
    return np.asarray(out, dtype=np.float64)


#: Why ``ftheta`` is refused rather than approximated (see the module docstring).
FTHETA_UNVERIFIED = (
    "the f-theta polynomial's convention is not verified on this Isaac Sim build: neither the "
    "units of the radius it returns (pixels of the nominal image, or normalised) nor whether k0 "
    "is a constant term is determined by anything the build exposes, and either guess renders a "
    "plausible fisheye that disagrees with the engine by tens of pixels at the field edge. "
    "Measure it first (roadmap M10.9b: fit r(theta) from a rendered grid), then implement it."
)


def distort_normalised(
    x: Any, y: Any, spec: DistortionSpec
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Distorted normalised image coordinates for the two verified models.

    Defined for ``brown_conrady`` and ``kannala_brandt``, whose output is still normalised (it is
    multiplied by ``fx``, ``fy`` to reach pixels). ``ftheta`` is refused
    (:data:`FTHETA_UNVERIFIED`).
    """
    xn = np.asarray(x, dtype=np.float64)
    yn = np.asarray(y, dtype=np.float64)
    if spec.model == "brown_conrady":
        return _rational_polynomial(xn, yn, opencv_pinhole_coeffs(spec))
    if spec.model == "kannala_brandt":
        r = np.hypot(xn, yn)
        theta = np.arctan2(r, 1.0)
        theta_d = _theta_d(theta, list(spec.coeffs))
        # r → 0 is the optical axis, where the scale factor θ_d/r → 1 and the ratio is 0/0.
        scale = np.divide(theta_d, r, out=np.ones_like(r), where=r > 0.0)
        return xn * scale, yn * scale
    if spec.model == "ftheta":
        raise NotImplementedError(FTHETA_UNVERIFIED)
    raise ValueError(f"unknown distortion model {spec.model!r}")


def undistort_normalised(
    xd: Any, yd: Any, spec: DistortionSpec, *, iterations: int = _UNDISTORT_ITERS
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Invert :func:`distort_normalised` -- the ideal ray a distorted image point came from.

    Both models are inverted by fixed-point iteration from the distorted point, which is how
    OpenCV's own ``undistortPoints`` does it: the distortion is a small perturbation, so the
    iteration contracts. It is not guaranteed to converge for a lens strong enough to fold the
    field (``r_d`` non-monotonic), so the residual is checked and a non-converged point comes
    back as NaN rather than as a confident wrong answer.
    """
    xd_a = np.asarray(xd, dtype=np.float64)
    yd_a = np.asarray(yd, dtype=np.float64)
    if spec.model == "kannala_brandt":
        # θ is the only unknown: invert θ_d(θ) by Newton, then rescale the direction.
        r_d = np.hypot(xd_a, yd_a)
        theta = r_d.copy()
        for _ in range(iterations):
            f = _theta_d(theta, list(spec.coeffs)) - r_d
            # dθ_d/dθ = 1 + 3k1θ² + 5k2θ⁴ + 7k3θ⁶ + 9k4θ⁸
            d = np.ones_like(theta)
            for i, k in enumerate(list(spec.coeffs)[:4]):
                d = d + (2 * i + 3) * float(k) * theta ** (2 * i + 2)
            step = np.divide(f, d, out=np.zeros_like(f), where=np.abs(d) > 0.0)
            theta = theta - step
            if np.all(np.abs(step) < _UNDISTORT_TOL):
                break
        r = np.tan(theta)
        scale = np.divide(r, r_d, out=np.ones_like(r_d), where=r_d > 0.0)
        return xd_a * scale, yd_a * scale

    x = xd_a.copy()
    y = yd_a.copy()
    for _ in range(iterations):
        fx_, fy_ = distort_normalised(x, y, spec)
        dx = xd_a - fx_
        dy = yd_a - fy_
        x = x + dx
        y = y + dy
        if np.all(np.abs(dx) < _UNDISTORT_TOL) and np.all(np.abs(dy) < _UNDISTORT_TOL):
            break
    fx_, fy_ = distort_normalised(x, y, spec)
    bad = ~(np.isclose(fx_, xd_a, atol=1e-8) & np.isclose(fy_, yd_a, atol=1e-8))
    if np.any(bad):
        x = np.where(bad, np.nan, x)
        y = np.where(bad, np.nan, y)
    return x, y


def project(
    points_cv: Any, intrinsics: Intrinsics, spec: DistortionSpec
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Camera-space points (OpenCV frame) → pixel coordinates ``(u, v)``.

    ``u`` increases to the right and ``v`` downwards, both in pixel-edge units on the grid
    ``intrinsics`` was built for: the centre of pixel ``(row, col)`` is at ``(col + 0.5,
    row + 0.5)``. Points behind the camera come back as NaN.
    """
    p = np.asarray(points_cv, dtype=np.float64)
    if spec.model == "ftheta":
        raise NotImplementedError(FTHETA_UNVERIFIED)
    xn, yn = normalise(p)
    xd, yd = distort_normalised(xn, yn, spec)
    return (
        np.asarray(intrinsics.cx_px + intrinsics.fx_px * xd),
        np.asarray(intrinsics.cy_px + intrinsics.fy_px * yd),
    )


def project_usd(
    points_usd: Any, intrinsics: Intrinsics, spec: DistortionSpec
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """:func:`project` for points in **USD camera space** (+Y up, −Z forward)."""
    return project(usd_camera_to_opencv(points_usd), intrinsics, spec)
