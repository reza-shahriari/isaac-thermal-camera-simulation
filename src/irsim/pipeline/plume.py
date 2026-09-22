"""Stage 2d — an exhaust plume composited per pixel onto the k× radiance plane (roadmap PH.6).

`PH.4` built the operator for one ray through hot gas and `PH.5` filled its tables. This is where
a plume meets a frame: a truncated cone in camera space, a **chord through it per pixel**, and
the slab operator evaluated on that chord.

    L = τ_p · L_plane + (1 − τ_p) · B_gas,at-sensor,     τ_p = exp(−κ_b(T_g, X) · L_chord)

**That form is exact, not a blend chosen for convenience.** Write the plane's value at a pixel as
``L_plane = τ_atm(R_p) L_behind + (1 − τ_atm(R_p)) L_air``, with ``L_behind`` whatever reaches the
plume from beyond it; putting the slab in the way gives
``τ_atm(R_p)[τ_p L_behind + (1 − τ_p) B_gas] + (1 − τ_atm(R_p)) L_air``, and that rearranges to the
line above provided ``B_gas,at-sensor`` is the gas radiance run through **stage 2's own** path at
the plume's range -- which is how it is computed here, by the same
``apply_layered_gbuffer`` / ``apply_atmosphere`` the plane went through. So the plume and the
pixels under it cannot disagree about the atmosphere, and nothing has to decide what is behind.
`irsim.pipeline.rotor_veil` reaches the same conclusion for an opaque veil; a slab is the
semi-transparent case of the same argument.

**Occlusion is the depth plane, not a mask.** Each pixel's chord is clipped to the distance the
G-buffer already carries, so a plume that runs behind the car's own bumper is cut where the
bumper is, and one entirely behind an opaque surface contributes nothing. That is one comparison
per pixel and it removes the class of bug where a plume paints over the pipe it leaves.

**The gradient is along the plume, not along the ray.** ADR 0098 deferred a temperature gradient
*within* one slab and that deferral stands: each pixel gets one homogeneous slab, evaluated at the
axial station of its own chord's midpoint. What varies pixel to pixel is which station that is, so
a frame still shows a plume that cools with distance from the tip -- which is the feature -- while
each ray keeps the operator PH.4 verified.

**The tail reaches ambient, which is below the tables' floor.** `GasSlab` refuses a temperature
under 300 K because below that it is not a hot gas, and ambient air usually is under it. A plume
still has to mix out into whatever air there is, so the two temperatures are used differently:
the gas *radiance* ``B_b(T_g)`` is a quadrature and is evaluated at the true temperature, while
the absorption *coefficient* is read at the table's floor. The difference that hides is bounded
by how much κ moves over the last few kelvin of a table whose column is already the optically
thin one -- far inside the model's own envelope -- and it happens where the plume has nearly
stopped absorbing anything.

**Entrainment dilutes temperature and species by the same factor**, because both ride on the same
conserved scalar in a mixing jet: with ``φ = exp(−x / mixing_length)``, ``T_g = T_air + φ (T_tip −
T_air)`` and ``p_i = φ p_i,tip``. One authored length therefore sets how fast the plume both cools
and thins, and the two cannot be set inconsistently. No buoyancy, no bend, no puffing (ADR 0098).

docs/physics-model.md §6.6 (the exhaust rows), §8.1, §8.3, §13.4 stage 2; ADR 0098.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from irsim.atmosphere.beer_lambert import apply_atmosphere
from irsim.atmosphere.layered import LayeredAtmosphere
from irsim.atmosphere.model import Atmosphere
from irsim.config.sensor import DistortionSpec
from irsim.optics.projection import Intrinsics, project, undistort_normalised
from irsim.pipeline.atmosphere import apply_layered_gbuffer
from irsim.pipeline.gas_slab import (
    GAS_T_MAX_K,
    GAS_T_MIN_K,
    GasBandTables,
    SlabQuantity,
    gas_band_radiance,
    soot_band_kappa_per_m,
)
from irsim.radiometry.lut import BandLUT, Quantity
from irsim.radiometry.spectral_response import SpectralResponse

__all__ = [
    "FLAME_T_K",
    "PLUME_T_NODES",
    "ExhaustPlume",
    "PlumeCone",
    "WorldPlume",
    "flame_plume",
    "chord_through_cone",
    "inject_plumes",
    "plume_window",
]

#: How many temperatures the band quantities are evaluated at before being interpolated per pixel.
#: ``B_b(T)`` and soot's ``⟨1/λ⟩`` are Simpson quadratures over the response -- milliseconds each,
#: which is nothing once and impossible per pixel. Both are smooth and monotone in T over a
#: plume's few hundred kelvin, so 33 nodes hold the interpolation error far below the model's own.
PLUME_T_NODES = 33


@dataclass(frozen=True)
class PlumeCone:
    """A truncated cone in **camera space** (OpenCV frame: +x right, +y down, +z forward).

    ``apex_m`` is the pipe's exit plane centre, ``axis`` the unit direction the gas travels,
    ``length_m`` how far the plume is modelled, and the two radii the cone's mouth and its end.
    A cone rather than a cylinder because a jet spreads at a roughly fixed half-angle, and a
    truncated one because the mouth is the pipe, not a point.
    """

    apex_m: tuple[float, float, float]
    axis: tuple[float, float, float]
    length_m: float
    radius_tip_m: float
    radius_end_m: float

    def __post_init__(self) -> None:
        axis = np.asarray(self.axis, dtype=np.float64)
        norm = float(np.linalg.norm(axis))
        if norm <= 0.0:
            raise ValueError("the plume axis must have a direction")
        object.__setattr__(self, "axis", tuple(float(v) for v in axis / norm))
        if self.length_m <= 0.0:
            raise ValueError("a plume needs a positive length")
        if self.radius_tip_m <= 0.0 or self.radius_end_m <= 0.0:
            raise ValueError("both plume radii must be positive")


@dataclass(frozen=True)
class ExhaustPlume:
    """A cone of hot gas leaving a pipe, with the tip state the exhaust solver hands over.

    ``t_tip_k`` and the tip partial pressures are the gas as it leaves the tailpipe -- `TC.7`'s
    ``ExhaustSolver`` produces the first and a fuel's stoichiometry the others. ``t_air_k`` is
    what the plume mixes into, and ``mixing_length_m`` how fast: the excess over ambient falls by
    ``1/e`` every such length, and the species fall with it.
    """

    cone: PlumeCone
    t_tip_k: float
    t_air_k: float
    mixing_length_m: float
    p_co2_tip_atm: float = 0.0
    p_h2o_tip_atm: float = 0.0
    f_soot_tip: float = 0.0

    def __post_init__(self) -> None:
        if not GAS_T_MIN_K <= self.t_tip_k <= GAS_T_MAX_K:
            raise ValueError(
                f"tip temperature {self.t_tip_k} K is outside {GAS_T_MIN_K:g}-{GAS_T_MAX_K:g} K, "
                "which is the range every absorption table covers"
            )
        if self.t_air_k <= 0.0:
            raise ValueError("the air the plume mixes into needs a positive temperature")
        if self.t_air_k > self.t_tip_k:
            raise ValueError("a plume cools towards the air, so the tip cannot be the colder end")
        if self.mixing_length_m <= 0.0:
            raise ValueError("mixing_length_m must be positive")
        for name in ("p_co2_tip_atm", "p_h2o_tip_atm", "f_soot_tip"):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} cannot be negative")
        if self.p_co2_tip_atm == self.p_h2o_tip_atm == self.f_soot_tip == 0.0:
            raise ValueError("a plume with no CO2, no H2O and no soot is warm air, not a plume")

    def dilution(self, axial_m: NDArray[np.float64]) -> NDArray[np.float64]:
        """``φ(x)``: the conserved-scalar fraction still un-mixed ``x`` metres from the tip."""
        return np.exp(-np.asarray(axial_m, dtype=np.float64) / self.mixing_length_m)


def chord_through_cone(
    cone: PlumeCone,
    directions: NDArray[np.float64],
    depth_m: NDArray[np.float64] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Ray/truncated-cone intersection: ``(chord length, axial station of the midpoint, range)``.

    ``directions`` is ``(..., 3)`` of **unit** camera-space rays from the origin. A ray that
    misses, or that is blocked by ``depth_m`` before it reaches the cone, returns length 0.

    The cone is ``|w|² − (w·a)² ≤ (r₀ + k (w·a))²`` for ``0 ≤ w·a ≤ length``, with ``w`` the
    offset from the apex and ``k`` the radial growth per metre. Substituting ``w = w₀ + t d``
    gives an ordinary quadratic in ``t``; the near-degenerate case where the ray runs parallel to
    the cone's surface falls out as the linear one and is handled rather than divided by.
    """
    d = np.asarray(directions, dtype=np.float64)
    apex = np.asarray(cone.apex_m, dtype=np.float64)
    axis = np.asarray(cone.axis, dtype=np.float64)
    k = (cone.radius_end_m - cone.radius_tip_m) / cone.length_m
    eps = 1e-12

    w0 = -apex  # the camera sits at the origin, so w0 = origin - apex
    a0 = float(w0 @ axis)
    a1 = d @ axis
    r0 = cone.radius_tip_m + k * a0
    kk = k * a1

    qa = 1.0 - a1**2 - kk**2
    qb = 2.0 * ((d @ w0) - a0 * a1 - r0 * kk)
    qc = float(w0 @ w0) - a0**2 - r0**2

    # The two end caps bound t before anything else: 0 <= a0 + t a1 <= length. A ray perpendicular
    # to the axis has no cap crossing, so the cap interval is the whole line or nothing.
    big = 1.0e30
    with np.errstate(invalid="ignore", divide="ignore"):
        t_at_0 = np.where(np.abs(a1) > eps, -a0 / a1, np.where(a0 >= 0.0, -big, big))
        t_at_l = np.where(
            np.abs(a1) > eps, (cone.length_m - a0) / a1, np.where(a0 <= cone.length_m, big, -big)
        )
    slab_lo = np.maximum(np.minimum(t_at_0, t_at_l), 0.0)  # nothing behind the camera
    slab_hi = np.maximum(t_at_0, t_at_l)
    if depth_m is not None:
        slab_hi = np.minimum(slab_hi, np.asarray(depth_m, dtype=np.float64))

    disc = qb**2 - 4.0 * qa * qc
    with np.errstate(invalid="ignore", divide="ignore"):
        root = np.sqrt(np.maximum(disc, 0.0))
        t_first = (-qb - root) / (2.0 * qa)
        t_second = (-qb + root) / (2.0 * qa)
        t_linear = np.where(np.abs(qb) > eps, -qc / qb, 0.0)
    r_lo = np.minimum(t_first, t_second)
    r_hi = np.maximum(t_first, t_second)

    quadratic = np.abs(qa) > eps
    opens_up = quadratic & (qa > 0.0)
    opens_down = quadratic & (qa < 0.0)
    linear = ~quadratic

    # One interval in the ordinary case; two when the parabola opens downward, which is the
    # ray looking within the cone's own half-angle of its axis -- down the barrel of the plume.
    # There the inequality holds *outside* the roots, and the cap slab can leave a piece on each
    # side. Both count towards the optical depth, so their lengths add; the temperature is read
    # at the midpoint of the longer, which is the homogeneous-slab choice ADR 0098 already makes.
    lo_1 = np.where(opens_up, np.maximum(slab_lo, r_lo), slab_lo)
    hi_1 = np.where(opens_up, np.minimum(slab_hi, r_hi), slab_hi)
    lo_2 = np.array(slab_hi, dtype=np.float64, copy=True)
    hi_2 = np.array(slab_hi, dtype=np.float64, copy=True)

    both_roots = opens_down & (disc >= 0.0)
    hi_1 = np.where(both_roots, np.minimum(slab_hi, r_lo), hi_1)
    lo_2 = np.where(both_roots, np.maximum(slab_lo, r_hi), lo_2)
    hi_1 = np.where(opens_up & (disc < 0.0), lo_1, hi_1)  # no real roots: the ray misses entirely

    lo_1 = np.where(linear & (qb < -eps), np.maximum(slab_lo, t_linear), lo_1)
    hi_1 = np.where(linear & (qb > eps), np.minimum(slab_hi, t_linear), hi_1)
    hi_1 = np.where(linear & (np.abs(qb) <= eps) & (qc > 0.0), lo_1, hi_1)

    len_1 = np.maximum(hi_1 - lo_1, 0.0)
    len_2 = np.maximum(hi_2 - lo_2, 0.0)
    length = np.asarray(np.nan_to_num(len_1 + len_2, nan=0.0, posinf=0.0))
    longer = len_2 > len_1
    mid = np.where(longer, 0.5 * (lo_2 + hi_2), 0.5 * (lo_1 + hi_1))
    mid = np.where(length > 0.0, mid, 0.0)
    axial = np.clip(a0 + mid * a1, 0.0, cone.length_m)
    return (
        length,
        np.asarray(np.where(length > 0.0, axial, 0.0)),
        np.asarray(np.where(length > 0.0, mid, 0.0)),
    )


def plume_window(
    cone: PlumeCone, intrinsics: Intrinsics, spec: DistortionSpec
) -> tuple[int, int, int, int] | None:
    """Bounding box ``(y0, y1, x0, x1)`` of the cone on the grid, or ``None`` if it is off it.

    Built by projecting the two end discs' extreme points rather than the whole surface: for a
    convex body seen through a pinhole those bound the silhouette, and the box is then padded by
    one pixel so a chord that grazes an edge pixel is not clipped away by rounding.
    """
    apex = np.asarray(cone.apex_m, dtype=np.float64)
    axis = np.asarray(cone.axis, dtype=np.float64)
    helper = np.array([0.0, 0.0, 1.0]) if abs(axis[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(axis, helper)
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    points = []
    for centre, radius in (
        (apex, cone.radius_tip_m),
        (apex + axis * cone.length_m, cone.radius_end_m),
    ):
        for angle in np.linspace(0.0, 2.0 * math.pi, 16, endpoint=False):
            points.append(centre + radius * (math.cos(angle) * u + math.sin(angle) * v))
    pts = np.asarray(points)
    if not np.any(pts[:, 2] > 0.0):
        return None
    x, y = project(pts[pts[:, 2] > 0.0], intrinsics, spec)
    finite = np.isfinite(x) & np.isfinite(y)
    if not finite.any():
        return None
    x0 = max(0, int(math.floor(float(x[finite].min()))) - 1)
    x1 = min(intrinsics.width, int(math.ceil(float(x[finite].max()))) + 1)
    y0 = max(0, int(math.floor(float(y[finite].min()))) - 1)
    y1 = min(intrinsics.height, int(math.ceil(float(y[finite].max()))) + 1)
    if x0 >= x1 or y0 >= y1:
        return None
    return (y0, y1, x0, x1)


def _ray_directions(
    intrinsics: Intrinsics, spec: DistortionSpec, window: tuple[int, int, int, int]
) -> NDArray[np.float64]:
    """Unit camera-space rays through the centre of every pixel in ``window``."""
    y0, y1, x0, x1 = window
    us = np.arange(x0, x1, dtype=np.float64) + 0.5
    vs = np.arange(y0, y1, dtype=np.float64) + 0.5
    uu, vv = np.meshgrid(us, vs)
    xd = (uu - intrinsics.cx_px) / intrinsics.fx_px
    yd = (vv - intrinsics.cy_px) / intrinsics.fy_px
    xn, yn = undistort_normalised(xd, yd, spec)
    d = np.stack([xn, yn, np.ones_like(xn)], axis=-1)
    return np.asarray(d / np.linalg.norm(d, axis=-1, keepdims=True))


def _band_nodes(
    response: SpectralResponse, t_lo: float, t_hi: float, quantity: Quantity
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """``(T nodes, B_b(T), soot κ per unit f_soot)`` -- the two quadratures, done once each."""
    if quantity not in ("lb", "lb_q"):
        raise ValueError(
            f"a plume radiates in {quantity!r}, which is a derivative table, not a radiance; the "
            "slab operator needs 'lb' or 'lb_q'"
        )
    form: SlabQuantity = "lb" if quantity == "lb" else "lb_q"
    nodes = np.linspace(t_lo, t_hi, PLUME_T_NODES) if t_hi > t_lo else np.array([t_lo, t_lo + 1.0])
    radiance = np.array([gas_band_radiance(response, float(t), form) for t in nodes])
    soot = np.array([soot_band_kappa_per_m(response, 1.0, float(t)) for t in nodes])
    return nodes, radiance, soot


def _through_atmosphere(
    radiance: NDArray[np.float64],
    distance_m: NDArray[np.float64],
    atmosphere: Atmosphere | LayeredAtmosphere | None,
    band: str,
    t_s: float,
    lut: BandLUT,
    quantity: Quantity,
) -> NDArray[np.float64]:
    if atmosphere is None:
        return radiance
    if isinstance(atmosphere, LayeredAtmosphere):
        return np.asarray(
            apply_layered_gbuffer(atmosphere, band, t_s, radiance, distance_m, quantity)
        )
    state = atmosphere.state(t_s)
    l_air = float(lut.lookup(np.float64(state.t_air_k), quantity)[()])
    return np.asarray(apply_atmosphere(radiance, distance_m, state.gamma_per_m[band], l_air))


def inject_plumes(
    radiance_ss: NDArray[np.floating],
    plumes: Sequence[ExhaustPlume],
    intrinsics: Intrinsics,
    distortion: DistortionSpec,
    response: SpectralResponse,
    tables: GasBandTables | None,
    distance_m: NDArray[np.floating] | None,
    atmosphere: Atmosphere | LayeredAtmosphere | None,
    band: str,
    t_s: float,
    lut: BandLUT,
    quantity: Quantity = "lb",
) -> NDArray[np.floating]:
    """Composite every plume onto the post-stage-2 k× radiance plane.

    Returns the plane unchanged (the same object) when there are no plumes, so a scene without
    one pays nothing and its goldens are untouched.
    """
    if not plumes:
        return radiance_ss
    out: Any = np.array(radiance_ss, dtype=np.float64, copy=True)
    depth = None if distance_m is None else np.asarray(distance_m, dtype=np.float64)
    for plume in plumes:
        window = plume_window(plume.cone, intrinsics, distortion)
        if window is None:
            continue
        y0, y1, x0, x1 = window
        directions = _ray_directions(intrinsics, distortion, window)
        patch_depth = None if depth is None else depth[y0:y1, x0:x1]
        length, axial, range_m = chord_through_cone(plume.cone, directions, patch_depth)
        hit = length > 0.0
        if not hit.any():
            continue

        phi = plume.dilution(axial)
        t_gas = plume.t_air_k + phi * (plume.t_tip_k - plume.t_air_k)
        nodes, radiance_nodes, soot_nodes = _band_nodes(
            response, plume.t_air_k, plume.t_tip_k, quantity
        )
        t_table = np.clip(t_gas, GAS_T_MIN_K, GAS_T_MAX_K)
        kappa = np.zeros_like(length)
        if plume.f_soot_tip > 0.0:
            kappa += plume.f_soot_tip * phi * np.interp(t_gas, nodes, soot_nodes)
        if tables is not None:
            for name, tip in (("co2", plume.p_co2_tip_atm), ("h2o", plume.p_h2o_tip_atm)):
                if tip <= 0.0:
                    continue
                table = tables.species.get(name)
                if table is None:
                    raise ValueError(
                        f"the plume carries {name.upper()} but band {tables.band!r} has no table; "
                        "`PH.5` generates them (scripts/generate_gas_luts.py)"
                    )
                pressure = np.where(hit, tip * phi, tip)
                column = np.maximum(pressure * length, 1e-12)
                kappa += pressure * table.kappa(t_table, column)
        tau = np.where(hit, np.exp(-kappa * length), 1.0)

        b_gas = np.interp(t_gas, nodes, radiance_nodes)
        b_at_sensor = _through_atmosphere(
            b_gas, np.maximum(range_m, 1e-6), atmosphere, band, t_s, lut, quantity
        )
        patch = out[y0:y1, x0:x1]
        out[y0:y1, x0:x1] = np.where(hit, tau * patch + (1.0 - tau) * b_at_sensor, patch)
    return np.asarray(out, dtype=radiance_ss.dtype)


@dataclass(frozen=True)
class WorldPlume:
    """A plume as a **scene** authors it: world coordinates, with the gas state already resolved.

    :meth:`irsim.scene.Scene.plumes_at` produces these, taking the geometry and chemistry from the
    config's ``plume:`` block, the tip temperature from `TC.7`'s solved gas at the tailpipe exit
    and the ambient from the scene's weather. A driver then turns one into an
    :class:`ExhaustPlume` with its camera's pose, and nothing between the scene and the frame has
    to know both frames at once.
    """

    origin_m: tuple[float, float, float]
    direction: tuple[float, float, float]
    length_m: float
    radius_tip_m: float
    radius_end_m: float
    mixing_length_m: float
    t_tip_k: float
    t_air_k: float
    p_co2_atm: float = 0.0
    p_h2o_atm: float = 0.0
    f_soot: float = 0.0

    def in_camera(
        self, rotation: NDArray[np.float64], camera_position_m: NDArray[np.float64]
    ) -> ExhaustPlume:
        """This plume in camera space, given the camera's pose.

        ``rotation`` maps world directions into the **OpenCV camera frame** (+x right, +y down,
        +z forward): its rows are the camera's own three axes written in world coordinates, which
        is what a renderer's view matrix carries. ``camera_position_m`` is the camera's world
        position.
        """
        r = np.asarray(rotation, dtype=np.float64)
        if r.shape != (3, 3):
            raise ValueError("rotation must be a 3x3 world-to-camera matrix")
        eye = np.asarray(camera_position_m, dtype=np.float64).reshape(3)
        apex = r @ (np.asarray(self.origin_m, dtype=np.float64) - eye)
        axis = r @ np.asarray(self.direction, dtype=np.float64)
        return ExhaustPlume(
            cone=PlumeCone(
                apex_m=(float(apex[0]), float(apex[1]), float(apex[2])),
                axis=(float(axis[0]), float(axis[1]), float(axis[2])),
                length_m=self.length_m,
                radius_tip_m=self.radius_tip_m,
                radius_end_m=self.radius_end_m,
            ),
            t_tip_k=self.t_tip_k,
            t_air_k=self.t_air_k,
            mixing_length_m=self.mixing_length_m,
            p_co2_tip_atm=self.p_co2_atm,
            p_h2o_tip_atm=self.p_h2o_atm,
            f_soot_tip=self.f_soot,
        )


#: The soot slab a luminous hydrocarbon flame is read as, kelvin. 1150-1300 K across the pool
#: fires in the fire-protection literature; the midpoint is the default. It is what the *camera*
#: sees, and it is not the same number as the flame's radiative surface emissive power
#: (:mod:`irsim.thermal.fire`), which is what a *wall* sees -- 1200 K at ε = 1 is 118 kW/m², and
#: so is 1500 K at ε = 0.41.
FLAME_T_K = 1200.0


def flame_plume(
    fire: Any,
    t_air_k: float,
    *,
    t_flame_k: float = FLAME_T_K,
    f_soot: float = 2.0e-6,
    tip_radius_m: float | None = None,
) -> WorldPlume:
    """A :class:`~irsim.thermal.fire.PoolFire` as a soot slab a camera can see (`PH.7`).

    The cone stands on the pool and tapers to the mean flame height, and its **cooling is not
    authored**: the mixing length is solved so that the slab reaches exactly the centreline excess
    Heskestad's correlation gives at the flame tip. So the fire a camera sees and the air a
    thermometer above it would read come from one model, and a scene cannot set them apart.

    Soot only -- no CO2 or H2O table is consulted. A luminous flame's own emission is its soot,
    which is grey-ish rather than banded (``κ ∝ 1/λ``), and that is why a fire looks far more
    alike between MWIR and LWIR than an exhaust plume does. The band difference that remains is
    the ratio of the bands' wavelengths and nothing more.
    """
    from irsim.thermal.fire import centreline_rise_k

    height = float(fire.flame_height_m)
    tip_rise = float(centreline_rise_k(height, fire, t_air_k))
    excess = float(t_flame_k) - float(t_air_k)
    if excess <= tip_rise:
        raise ValueError(
            f"a {t_flame_k:g} K flame is only {excess:.0f} K above this air, which is at or below "
            f"the {tip_rise:.0f} K its own tip reaches by Heskestad's correlation: the slab would "
            "have to warm as it rises"
        )
    mixing = height / math.log(excess / tip_rise)
    radius = 0.5 * float(fire.diameter_m)
    base = np.asarray(fire.base_m, dtype=np.float64).reshape(3)
    axis = np.asarray(fire.axis, dtype=np.float64).reshape(3)
    return WorldPlume(
        origin_m=(float(base[0]), float(base[1]), float(base[2])),
        direction=(float(axis[0]), float(axis[1]), float(axis[2])),
        length_m=height,
        radius_tip_m=radius,
        radius_end_m=float(tip_radius_m) if tip_radius_m is not None else 0.35 * radius,
        mixing_length_m=mixing,
        t_tip_k=float(t_flame_k),
        t_air_k=float(t_air_k),
        f_soot=float(f_soot),
    )
