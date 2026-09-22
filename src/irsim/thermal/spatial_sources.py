"""Heat sources that vary **across** a surface: what makes a bonnet a gradient and not a patch.

docs/physics-model.md §6.1 (where this enters the balance), §6.6 (the source schedules);
ADR 0087 (the field this fills), ADR 0088 (this module's choices).

MP.1 gave a surface a grid of cells. This module is what makes their forcing differ. Two things in
a vehicle scene need it and they turn out to be the *same* geometry problem:

* an **engine bay** under a bonnet -- a hot cavity radiating up onto the skin above it, strongly
  where the skin is over the block and weakly out at the wings;
* a **warm underbody** over asphalt -- the same hot rectangle, now radiating down.

Both are "a differential surface element exchanging radiation with a parallel rectangle", which has
an exact closed-form configuration factor, so one kernel serves both and the falloff is *computed*
rather than authored as a Gaussian with a fitted width.

**The term this module exists to get right is the one that is easy to forget: a hot body that
stands over a surface also blocks the sky that surface was seeing.** Adding the car's radiation
without removing the sky it occludes is free energy, and it is wrong in both directions depending
on the weather. Under a **clear** night sky the occlusion *dominates*: a 290 K car over asphalt
replaces a ~250 K sky, and the warm car-shaped patch in a night parking-lot thermal image is mostly
that, not engine heat -- it is there before the engine starts. Under **overcast** the sky is close
to air temperature, the occlusion term nearly cancels, and what remains really is the engine. So
:func:`occluded_longwave_flux` takes the downwelling the cell would otherwise have seen and returns
the **net** change, never the source term alone.

**Parallel only.** The closed form is for a plane element and a rectangle parallel to it. A bonnet
over a block and asphalt under a sill are both that; a door over a kerb is not. Non-parallel
geometry raises rather than being silently evaluated with a formula that does not apply -- the
result would be a plausible number, which is the worst kind of wrong here.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from irsim.radiometry.constants import SIGMA_SB
from irsim.thermal.surface_field import PlanarPatch

__all__ = [
    "RadiantRectangle",
    "corner_view_factor",
    "view_factor_to_parallel_rectangle",
    "patch_view_factors",
    "clamp_view_factor_sum",
    "occluded_longwave_flux",
]


def corner_view_factor(x: Any, y: Any, c: Any) -> NDArray[np.float64]:
    """F from a plane element to a parallel rectangle with one **corner** over the element.

    The rectangle spans ``[0, x] x [0, y]`` at separation ``c``. Howell's catalogue C-11::

        F = 1/(2 pi) [ X/sqrt(1+X^2) atan(Y/sqrt(1+X^2)) + Y/sqrt(1+Y^2) atan(X/sqrt(1+Y^2)) ]

    with ``X = x/c`` and ``Y = y/c``. Extended **odd in each argument** so that
    :func:`view_factor_to_parallel_rectangle` can superpose four corners for a rectangle at an
    arbitrary offset; that is what the signs are for, and it is why the factor is not clipped to
    [0, 1] here -- an individual corner term is signed and only the sum is a view factor.

    The constant is 1/(2 pi) and was checked against a brute-force quadrature of
    ``c^2 / (pi (x^2 + y^2 + c^2)^2)`` rather than taken from memory: the quarter-infinite limit is
    1/4, and `test_spatial_sources` re-derives both.
    """
    xs = np.asarray(x, dtype=np.float64)
    ys = np.asarray(y, dtype=np.float64)
    cs = np.asarray(c, dtype=np.float64)
    if np.any(cs <= 0.0):
        raise ValueError("separation must be positive: a surface cannot radiate onto itself")
    sign = np.sign(xs) * np.sign(ys)
    big_x = np.abs(xs) / cs
    big_y = np.abs(ys) / cs
    rx = np.sqrt(1.0 + big_x**2)
    ry = np.sqrt(1.0 + big_y**2)
    return np.asarray(
        sign
        / (2.0 * np.pi)
        * (big_x / rx * np.arctan(big_y / rx) + big_y / ry * np.arctan(big_x / ry))
    )


@dataclass(frozen=True)
class RadiantRectangle:
    """A rectangular grey radiator -- an engine block seen from above, an underbody from below.

    ``emissivity`` is the *source* side of the exchange. An engine bay is close to a blackbody
    cavity (multiply-reflecting, cluttered, near-unity apparent emissivity), which is why its
    default is high; a painted or oily underbody is lower and should be authored.
    """

    centre_m: NDArray[np.float64]
    u_axis: NDArray[np.float64]
    v_axis: NDArray[np.float64]
    half_u_m: float
    half_v_m: float
    emissivity: float = 0.95
    #: A flame's **surface emissive power**, W/m² (`PH.7`). Set it and this rectangle radiates
    #: that, directly, instead of ``ε σ T⁴``. It is not a convenience: a flame's temperature and
    #: emissivity are not separately knowable -- 1200 K at ε = 1 and 1500 K at ε = 0.41 are the
    #: same 118 kW/m² -- so fire protection authors the product and nothing else.
    #: :mod:`irsim.thermal.fire` has the values and the flux term that consumes them.
    sep_w_m2: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "centre_m", np.asarray(self.centre_m, dtype=np.float64).reshape(3))
        for name in ("u_axis", "v_axis"):
            v = np.asarray(getattr(self, name), dtype=np.float64).reshape(3)
            norm = float(np.linalg.norm(v))
            if norm <= 0.0:
                raise ValueError(f"{name} has zero length")
            object.__setattr__(self, name, v / norm)
        if abs(float(np.dot(self.u_axis, self.v_axis))) > 1e-9:
            raise ValueError("u_axis and v_axis must be perpendicular")
        if self.half_u_m <= 0.0 or self.half_v_m <= 0.0:
            raise ValueError("half extents must be positive")
        if not 0.0 <= self.emissivity <= 1.0:
            raise ValueError("emissivity must lie in [0, 1]")
        if self.sep_w_m2 is not None and self.sep_w_m2 < 0.0:
            raise ValueError("surface emissive power cannot be negative")

    def emitted_flux_w_m2(self, temperature_k: float | None = None) -> float:
        """What leaves this rectangle's surface, W/m²: its authored SEP, else ``ε σ T⁴``.

        A radiator answers to one of the two and not both. Passing a temperature to a rectangle
        that carries a SEP is refused rather than ignored, because the two numbers disagree by
        design -- the SEP is what was measured and the temperature is whatever someone put beside
        it -- and silently preferring one would make which is used an implementation detail.
        """
        if self.sep_w_m2 is not None:
            if temperature_k is not None:
                raise ValueError(
                    "this rectangle carries an authored surface emissive power, so it has no "
                    "temperature to radiate at; pass one or the other (`PH.7`)"
                )
            return float(self.sep_w_m2)
        if temperature_k is None:
            raise ValueError("a rectangle with no authored SEP needs a temperature to radiate at")
        if temperature_k <= 0.0:
            raise ValueError("source temperature must be positive (kelvin)")
        return float(self.emissivity * SIGMA_SB * float(temperature_k) ** 4)

    @property
    def normal(self) -> NDArray[np.float64]:
        return np.asarray(np.cross(self.u_axis, self.v_axis))

    @property
    def area_m2(self) -> float:
        return float(4.0 * self.half_u_m * self.half_v_m)


def view_factor_to_parallel_rectangle(points: Any, rect: RadiantRectangle) -> NDArray[np.float64]:
    """F from each point to ``rect``, by superposing the four signed corner terms.

    ``points`` must lie on a plane parallel to the rectangle. Points level with the rectangle's
    plane raise -- the configuration factor is singular there and a clamped value would be a
    fiction.

    **The offsets are measured in the rectangle's own axes, and there is no option** (PT.4). This
    used to take an ``axes=`` argument "so the offsets are measured in a frame the caller chose",
    and :func:`patch_view_factors` passed the *receiver's* axes. That mixes frames: ``du`` came out
    along the patch's u while ``half_u_m`` is an extent along the rectangle's, so a receiver rotated
    in its own plane was evaluated as though the rectangle had turned with it.

    The parameter could not have been right, because the quantity does not depend on it. This is
    Howell C-11, a **differential element** to a parallel rectangle: the element contributes its
    position and its normal, and a plane element has no in-plane orientation for the answer to
    depend on. How the receiving surface happens to be parameterised is bookkeeping for the caller's
    array order, never an input to the configuration factor.
    """
    p = np.asarray(points, dtype=np.float64)
    if p.ndim == 0 or p.shape[-1] != 3:
        raise ValueError(f"points must have a trailing axis of 3, got shape {p.shape}")
    offset = p - rect.centre_m
    du = offset @ rect.u_axis
    dv = offset @ rect.v_axis
    c = np.abs(offset @ rect.normal)
    if np.any(c <= 0.0):
        raise ValueError(
            "a point lies in the radiator's own plane: the configuration factor is singular there"
        )
    # The rectangle spans [-half, +half] about the point's own footprint, so the four corners are
    # at (+-half - offset). Superposition: F = G(u1,v1) - G(u0,v1) - G(u1,v0) + G(u0,v0).
    u1 = rect.half_u_m - du
    u0 = -rect.half_u_m - du
    v1 = rect.half_v_m - dv
    v0 = -rect.half_v_m - dv
    total = (
        corner_view_factor(u1, v1, c)
        - corner_view_factor(u0, v1, c)
        - corner_view_factor(u1, v0, c)
        + corner_view_factor(u0, v0, c)
    )
    return np.asarray(np.clip(total, 0.0, 1.0))


def patch_view_factors(patch: PlanarPatch, rect: RadiantRectangle) -> NDArray[np.float64]:
    """``(n_cells,)`` view factors from a patch's cells to a parallel rectangle."""
    if abs(abs(float(np.dot(patch.normal, rect.normal))) - 1.0) > 1e-9:
        raise ValueError(
            "the closed form is for parallel surfaces only; this patch and radiator are not "
            "parallel, and evaluating it anyway would return a plausible wrong number"
        )
    # The rectangle's own axes, always -- see `view_factor_to_parallel_rectangle`. Passing the
    # patch's here is what PT.4 removed: it made the result depend on how the receiver happened to
    # be parameterised, which the configuration factor does not.
    return view_factor_to_parallel_rectangle(patch.cell_centres(), rect)


def clamp_view_factor_sum(
    views: Sequence[Any], *, tol: float = 1e-6
) -> tuple[NDArray[np.float64], ...]:
    """Rescale view factors from several radiators to the same patch so their sum never exceeds 1.

    docs/decisions/0090-clamped-view-factors-for-nested-radiators.md. A plane element's view
    factors to any set of surfaces filling disjoint solid angles sum to at most 1 (Howell). Each
    :func:`patch_view_factors` call is exact for *one* radiator, but ``car_demo.build_ground_field``
    calls it once each for ``underbody``, ``engine_bay`` and ``exhaust_pipe`` -- three
    :class:`RadiantRectangle` authored as *nested* regions of one physical floor pan (ADR 0088's
    "three rather than one because the picture they make differs"), not three independent bodies
    filling disjoint angles. Summing their individually-correct view factors therefore double
    (triple, under the engine bay) counts the solid angle the smaller rectangles share with
    ``underbody``. Measured peak 1.40 before this function existed.

    The fix rescales every radiator's factor at each cell by the same per-cell ratio wherever the
    sum would exceed 1, so the *relative* contribution between radiators -- the broad footprint,
    the brighter pool, the narrow stripe -- is unchanged and only the total is capped at the
    physical bound. This is an approximation, not a re-derivation of the true partitioned geometry;
    :func:`occluded_longwave_flux` is linear in its ``view_factors`` argument, so passing the
    clamped factors caps the occlusion term by the same ratio as the source term, keeping both
    internally consistent.

    Warns (loudly, not silently) whenever clamping actually changes a value, since the trigger is a
    modelling gap -- overlapping radiator footprints -- and not routine behaviour.
    """
    stacked = np.stack([np.asarray(v, dtype=np.float64) for v in views], axis=0)
    if np.any(stacked < 0.0) or np.any(stacked > 1.0 + tol):
        raise ValueError("each input view factor must already lie in [0, 1]")
    total = stacked.sum(axis=0)
    over = total > 1.0 + tol
    if np.any(over):
        warnings.warn(
            f"view factors summed to {float(total.max()):.4f} > 1 on {int(np.count_nonzero(over))} "
            "cell(s) -- radiator footprints overlap; rescaling proportionally to conserve energy "
            "(docs/decisions/0090-clamped-view-factors-for-nested-radiators.md)",
            stacklevel=2,
        )
    scale = np.where(over, 1.0 / total, 1.0)
    return tuple(np.asarray(stacked[i] * scale) for i in range(stacked.shape[0]))


def occluded_longwave_flux(
    view_factors: Any,
    source_temperature_k: float,
    surface_emissivity: Any,
    *,
    source_emissivity: float = 1.0,
    longwave_down_w_m2: Any = 0.0,
    sky_view: Any = 1.0,
) -> NDArray[np.float64]:
    """The **net** flux a hot body adds to a surface, as a ``q_internal_w_m2`` term.

    ``F eps_s (eps_r sigma T_r^4 - L_occluded)``, where ``L_occluded`` is the downwelling
    irradiance per unit sky view that the radiator now blocks. The subtraction is the point: a body
    standing over a surface both radiates onto it *and* takes away the sky it was seeing, and
    keeping only the first term invents energy. Under a clear night sky the second term is the
    larger one, which is why a parked car leaves a warm shadow before its engine has ever run.

    ``eps_s`` multiplies here because :meth:`~irsim.thermal.facets.FacetSolver.net_flux` adds
    ``q_internal`` **unweighted** -- it is an internal deposition, not an incident irradiance. A
    caller who leaves it out gets a result that is right only for a black surface.
    """
    f = np.asarray(view_factors, dtype=np.float64)
    eps_s = np.asarray(surface_emissivity, dtype=np.float64)
    if np.any(f < 0.0) or np.any(f > 1.0):
        raise ValueError("view factors must lie in [0, 1]")
    if source_temperature_k <= 0.0:
        raise ValueError("source temperature must be positive (kelvin)")
    if not 0.0 <= source_emissivity <= 1.0:
        raise ValueError("source_emissivity must lie in [0, 1]")
    l_down = np.asarray(longwave_down_w_m2, dtype=np.float64)
    v_s = np.asarray(sky_view, dtype=np.float64)
    # Per unit sky view, so that a wall already seeing half the sky loses half as much again.
    occluded = np.where(v_s > 0.0, l_down / np.where(v_s > 0.0, v_s, 1.0), 0.0)
    return np.asarray(
        f * eps_s * (source_emissivity * SIGMA_SB * source_temperature_k**4 - occluded)
    )
