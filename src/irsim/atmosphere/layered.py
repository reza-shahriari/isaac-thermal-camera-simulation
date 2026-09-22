"""Layered slant-path atmosphere with an exponential-sum band model (§7.1, §5.3; ADR 0071).

One grey γ_B per band is exact only for spectrally flat extinction; real bands mix opaque lines,
moderate lines and a transparent window, and the band transmittance is a *sum of exponentials*
(M8.7's curve of growth). Per band the model is

    τ_B(path) = Σ_k w_k exp(−∫ γ_k(h) ds),      γ_k(h) = γ_k,0 e^{−h/H_k} + γ_aer,B e^{−h/H_aer}

with one term per **spectral class** k -- a sub-band of the response (window, water-vapour lines,
band-edge lines, an opaque CO₂ core) whose weight w_k is its Planck-weighted share of the band
(300 K for the emissive bands, 5800 K for the reflective ones) and whose surface extinction is
a multiplier g_k times the preset's molecular γ_mol,B(w) ("water" classes, scale height H_w)
or an absolute value ("air" classes: CO₂, scale height H_air). The non-opaque classes are
rescaled by one factor s so that the *horizontal 200 m* transmittance equals the grey preset's
(M8.3's table anchor) at the current weather; opaque classes sit on top with fixed extinction
(ADR 0048 read the table for the transparent part of the band). Along a ray at elevation θ
(flat earth, h = s sin θ) the column integrals of the exponential profiles are analytic, so
transmittance is closed-form; path radiance L_path = Σ_k w_k ∫ γ_k(h) τ_k(s) L_B(T(h)) ds with
T(h) = T_air − Γ min(h, h_tropopause) is a quadrature, and the **sky radiance is the column
emission** L_sky,B(θ) = L_path,B(∞, θ) (space contributes nothing). At θ = 0 the model reduces
to horizontal Beer–Lambert per term and, with one class, to M8.1 exactly.

Calibration (ADR 0071): the window multipliers are set so the clear dry LWIR sky reads
−40 °C at 15° elevation and the MWIR sky warmer than +10 °C, the two R13 anchors (Tucson,
clear, low humidity). Everything else is spectroscopy (class edges) or the preset.

docs/physics-model.md §7.1 [R13], §7.4, §5.3, Appendix A #2
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq, least_squares

from irsim.atmosphere.extinction import gamma_aerosol_visible
from irsim.atmosphere.humidity import gamma_molecular
from irsim.config.atmosphere import AtmospherePreset
from irsim.config.bands import ANCHOR_BAND, nominal_range_for, regime_for
from irsim.radiometry.band_average import T_REF_K
from irsim.radiometry.band_integration import quadrature_grid, simpson
from irsim.radiometry.lut import BandLUT, Quantity
from irsim.radiometry.planck import spectral_radiance
from irsim.radiometry.spectral_response import SpectralResponse
from irsim.thermal.weather import WeatherSeries

__all__ = [
    "ANCHOR_DISTANCE_M",
    "SpectralClass",
    "ATMOSPHERE_LADDER",
    "LADDER_SPAN_UM",
    "band_span_um",
    "classes_for",
    "SOLAR_WEIGHT_T_K",
    "weight_reference_temperature",
    "class_weights",
    "ExponentialSum",
    "column_length",
    "LayeredAtmosphere",
    "fit_exponential_sum",
    "exponential_sum_from_piecewise",
    "T_SPACE_K",
]

ANCHOR_DISTANCE_M = 200.0  # the §7.2 table column the grey presets were fitted at
T_SPACE_K = 0.0  # the space term: L_B(2.7 K) is zero at every LUT precision
Kind = Literal["water", "air"]


@dataclass(frozen=True)
class SpectralClass:
    """A sub-band of the response with one extinction behaviour.

    ``multiplier`` is g_k (× γ_mol,B, then × the anchor scale s) for non-opaque water classes,
    an absolute γ (m⁻¹, × s) for non-opaque air classes, and an absolute γ for opaque classes
    (never rescaled, excluded from the 200 m anchor).
    """

    name: str
    edges_um: tuple[tuple[float, float], ...]
    kind: Kind
    multiplier: float
    opaque: bool = False


#: The spectroscopy, once, in wavelength order (`AT.10`, ADR 0113).
#:
#: Each rung is a contiguous interval of the spectrum with one extinction behaviour, and the
#: rungs are **gap-free** from `LADDER_SPAN_UM[0]` to `LADDER_SPAN_UM[1]`. A band's spectral
#: classes are *derived* from this by :func:`classes_for`, which is what makes a fifth band a
#: config change rather than a `src/` edit -- and what stops two cameras disagreeing about the
#: same air, which the per-band tables this replaces did: NIR resolved the 0.94 µm water band at
#: ×10 while SWIR's window swallowed 0.90-0.98 µm at ×0.5, a factor of twenty on one sky.
#:
#: H2O bands at 0.94, 1.1, 1.4, 1.9, 2.7, 6.3 µm and the rotational edges beyond 13 µm; CO2 ν3
#: at 4.3 µm (opaque within metres) and the CO2/N2O complex at 4.45-4.6 and 4.85-5.0 µm.
#: Multipliers are ESTIMATED; the window values are the R13 calibration (ADR 0071). The 1.9 µm
#: and 6.3 µm rungs fill holes the per-band tables left -- 1.80-2.00 and 6.00-7.00 µm belonged to
#: no class at all, so a response reaching either raised rather than absorbing.
ATMOSPHERE_LADDER: tuple[SpectralClass, ...] = (
    SpectralClass("window_sw", ((0.35, 0.90),), "water", 0.5),
    SpectralClass("h2o_0p94", ((0.90, 0.98),), "water", 10.0),
    SpectralClass("window_sw", ((0.98, 1.10),), "water", 0.5),
    SpectralClass("h2o_1p1", ((1.10, 1.17),), "water", 10.0),
    SpectralClass("window_sw", ((1.17, 1.33),), "water", 0.5),
    SpectralClass("h2o_1p4", ((1.33, 1.48),), "water", 0.05, opaque=True),
    SpectralClass("window_sw", ((1.48, 1.80),), "water", 0.5),
    SpectralClass("h2o_1p9", ((1.80, 2.00),), "water", 0.05, opaque=True),
    SpectralClass("window", ((2.00, 2.55),), "water", 0.3),
    SpectralClass("h2o_2p7", ((2.55, 2.95),), "water", 0.05, opaque=True),
    SpectralClass("h2o_wing", ((2.95, 3.35),), "water", 8.0),
    SpectralClass("window", ((3.35, 4.17),), "water", 0.3),
    SpectralClass("co2_4p3", ((4.17, 4.45),), "air", 0.5, opaque=True),
    SpectralClass("co2_n2o", ((4.45, 4.60),), "air", 2.0e-3),
    SpectralClass("window", ((4.60, 4.85),), "water", 0.3),
    SpectralClass("co2_n2o", ((4.85, 5.00),), "air", 2.0e-3),
    SpectralClass("h2o_wing", ((5.00, 6.00),), "water", 8.0),
    SpectralClass("h2o_6p3", ((6.00, 7.00),), "water", 0.05, opaque=True),
    SpectralClass("edges", ((7.00, 7.80),), "water", 40.0),
    SpectralClass("lines", ((7.80, 8.30),), "water", 5.0),
    SpectralClass("window", ((8.30, 12.50),), "water", 0.3),
    SpectralClass("lines", ((12.50, 13.20),), "water", 5.0),
    SpectralClass("edges", ((13.20, 14.50),), "water", 40.0),
)

#: What the ladder covers. A band whose response leaves it is refused rather than given a
#: nearest class: the numbers outside are spectroscopy nobody has written down here.
LADDER_SPAN_UM = (
    min(lo for rung in ATMOSPHERE_LADDER for lo, _ in rung.edges_um),
    max(hi for rung in ATMOSPHERE_LADDER for _, hi in rung.edges_um),
)


def _ladder_is_ordered_and_gap_free() -> None:
    """Checked at import, because a gap here is a band that raises at the far end of a render."""
    edge = LADDER_SPAN_UM[0]
    for rung in ATMOSPHERE_LADDER:
        ((lo, hi),) = rung.edges_um
        if lo != edge or hi <= lo:
            raise ValueError(f"atmosphere ladder is not ordered and gap-free at {rung.name}")
        edge = hi


_ladder_is_ordered_and_gap_free()


def band_span_um(band: str, response: SpectralResponse | None = None) -> tuple[float, float]:
    """The wavelengths a band's classes have to cover: its nominal range **and** its response.

    Both, because either alone has been wrong here. The nominal range alone was `AT.3`'s defect --
    the NIR classes stopped at 1.05 µm while `nir_si.csv` reaches 1.10, carrying 0.188 % of the
    band outside every class, which `class_weights` raises on. The response alone would let a
    camera with a narrow filter shrink the model of the air it looks through.
    """
    lo, hi = nominal_range_for(band)
    if response is not None:
        r_lo, r_hi = response.support_um
        lo, hi = min(lo, float(r_lo)), max(hi, float(r_hi))
    return float(lo), float(hi)


def classes_for(band: str, response: SpectralResponse | None = None) -> tuple[SpectralClass, ...]:
    """A band's spectral classes, derived from the one ladder by the band's own span.

    Every rung the span **overlaps** is taken whole -- not clipped to the span -- so the classes
    always reach a little past the detector, which is the margin the hand-written tables carried
    by eye. Rungs of the same name merge into one class with several intervals, in wavelength
    order, which is the shape :class:`SpectralClass` already had.

    Overlap is half-open, ``lo < span_hi and hi > span_lo``, so a band ending exactly on a rung
    boundary does not pick up the rung beyond it: LWIR's response stops at 7.00 µm and does not
    acquire the 6.3 µm water band, which carries none of its light.
    """
    lo, hi = band_span_um(band, response)
    if lo < LADDER_SPAN_UM[0] or hi > LADDER_SPAN_UM[1]:
        raise ValueError(
            f"band {band!r} spans {lo}-{hi} um, outside the atmosphere ladder "
            f"{LADDER_SPAN_UM[0]}-{LADDER_SPAN_UM[1]} um. Extend ATMOSPHERE_LADDER with the "
            "spectroscopy for that region rather than letting the nearest class stand in for it."
        )
    merged: dict[str, list[tuple[float, float]]] = {}
    seen: dict[str, SpectralClass] = {}
    for rung in ATMOSPHERE_LADDER:
        ((r_lo, r_hi),) = rung.edges_um
        if r_lo >= hi or r_hi <= lo:
            continue
        if rung.name in seen:
            other = seen[rung.name]
            if (other.kind, other.multiplier, other.opaque) != (
                rung.kind,
                rung.multiplier,
                rung.opaque,
            ):
                raise ValueError(f"ladder rung {rung.name!r} appears twice with different physics")
        else:
            seen[rung.name] = rung
            merged[rung.name] = []
        merged[rung.name].append((r_lo, r_hi))
    if not merged:
        raise ValueError(f"band {band!r} spans {lo}-{hi} um and overlaps no ladder rung")
    return tuple(
        SpectralClass(name, tuple(edges), seen[name].kind, seen[name].multiplier, seen[name].opaque)
        for name, edges in merged.items()
    )


#: Planck weighting temperature for a *reflective* band's spectral-class shares (§7.1): its in-band
#: radiance is borrowed sunlight, so how much of the band each class carries is set by the solar
#: spectrum rather than by the scene. ``irsim.radiometry.solar`` models the TOA spectrum itself at
#: 5778 K; the 22 K disagreement moves the shipped SWIR and NIR class shares by 1.2e-3 relative
#: (measured), which is below this fit's own uncertainty but is still two constants for one fact.
SOLAR_WEIGHT_T_K = 5800.0


def weight_reference_temperature(band: str) -> float:
    """The temperature ``class_weights`` weights a band's classes at, from the band's regime.

    This was a five-row dict keyed by band name -- the second band registry AT.4 exists to remove --
    and it had already drifted from the one it duplicated. It weighted MWIR at 300 K while
    ``DEFAULT_REGIME["mwir"]`` is ``"mixed"``, so the obvious derivation ("emissive keeps 300 K")
    would have moved MWIR to 5800 K and silently rescaled every MWIR class share, its anchor solve
    and therefore tau_MWIR at every range but the 200 m anchor. The rule that reproduces all five
    shipped rows exactly turns on *reflective*, not on emissive.
    """
    return SOLAR_WEIGHT_T_K if regime_for(band) == "reflective" else T_REF_K


def _nominal_response(band: str) -> SpectralResponse:
    lo, hi = nominal_range_for(band)
    return SpectralResponse(np.array([lo, hi]), np.array([1.0, 1.0]), f"<top-hat {band}>", "")


def class_weights(
    band: str, response: SpectralResponse | None = None, t_ref_k: float | None = None
) -> NDArray[np.float64]:
    """Planck-weighted share of the band per spectral class; sums to 1; every wavelength of the
    response must fall in some class (the class edges cover the nominal bands with margin)."""
    resp = response if response is not None else _nominal_response(band)
    classes = classes_for(band, response)
    t_ref = weight_reference_temperature(band) if t_ref_k is None else t_ref_k
    grid = quadrature_grid(resp)
    dl = float(grid[1] - grid[0])
    weight = resp.resampled(grid) * spectral_radiance(grid, np.asarray(t_ref, dtype=np.float64))
    total = float(simpson(weight, dl))
    out = np.zeros(len(classes))
    covered = np.zeros(grid.shape, dtype=bool)
    # The classes tile their span half-open, ``[lo, hi)``, so that a wavelength on a shared
    # boundary belongs to exactly one of them. The set's **own top edge** is closed instead: a
    # response whose support ends exactly there -- a top-hat does, at full height -- would
    # otherwise fall outside every class and raise, which is what the nominal MWIR band did the
    # moment its classes stopped being hand-written wider than the band (AT.10).
    top = max(hi for c in classes for _, hi in c.edges_um)
    for i, c in enumerate(classes):
        mask = np.zeros(grid.shape, dtype=bool)
        for lo, hi in c.edges_um:
            mask |= (grid >= lo) & ((grid <= hi) if hi == top else (grid < hi))
        covered |= mask
        out[i] = float(simpson(np.where(mask, weight, 0.0), dl)) / total
    if np.any(weight[~covered] > 1e-6 * weight.max()):
        lo, hi = grid[~covered][0], grid[~covered][-1]
        raise ValueError(f"band {band!r}: response has weight outside its classes ({lo}-{hi} um)")
    return np.asarray(out / out.sum(), dtype=np.float64)


def column_length(distance_m: Any, elevation_rad: Any, scale_height_m: float) -> Any:
    """∫₀^d e^{−s sinθ/H} ds on a flat-earth ray: d at θ = 0, else H/sinθ (1 − e^{−d sinθ/H}).

    ``elevation_rad`` broadcasts against ``distance_m`` (AT.1), so one call answers a whole frame
    whose pixels each look along their own ray. It used to be a scalar, and the pipeline passed
    **0.0 unconditionally** -- every resolved pixel got surface-density extinction over its whole
    slant range, while the unresolved point-target path beside it used the target's real elevation.

    A ray at or below the horizon keeps the horizontal form: the flat-earth column is then ``d``,
    and the descending branch is the sea's business (ADR 0078), not this model's.
    """
    d = np.asarray(distance_m, dtype=np.float64)
    st = np.sin(np.asarray(elevation_rad, dtype=np.float64))
    up = st > 0.0
    # `np.where` evaluates both branches, so the divide is guarded rather than masked afterwards:
    # a zero sine would raise and a negative one would return a negative column.
    safe = np.where(up, st, 1.0)
    with np.errstate(over="ignore", invalid="ignore"):
        slant = scale_height_m / safe * (1.0 - np.exp(-d * safe / scale_height_m))
    return np.asarray(np.where(up, slant, np.broadcast_to(d, np.shape(slant))))


def _scaled(gamma: float, column: NDArray[np.float64]) -> NDArray[np.float64]:
    if gamma == 0.0:
        return np.zeros_like(column)
    return np.asarray(gamma * column, dtype=np.float64)


@dataclass(frozen=True)
class ExponentialSum:
    """τ_B(path) = Σ_k w_k exp(−γ_k,0 C_k(path) − γ_aer C_aer(path)) for one band and weather."""

    weights: NDArray[np.float64]
    gamma_0: NDArray[np.float64]  # surface extinction per class, m^-1 (aerosol excluded)
    scale_heights_m: NDArray[np.float64]  # per class
    gamma_aerosol: float
    aerosol_scale_height_m: float
    names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if abs(float(self.weights.sum()) - 1.0) > 1e-9 or np.any(self.weights < 0.0):
            raise ValueError("class weights must be non-negative and sum to 1")
        if np.any(self.gamma_0 < 0.0) or self.gamma_aerosol < 0.0:
            raise ValueError("extinctions must be non-negative")

    @property
    def n_terms(self) -> int:
        return int(self.weights.size)

    def optical_depths(self, distance_m: Any, elevation_rad: Any = 0.0) -> NDArray[np.float64]:
        """Per class optical depth along the ray, shape ``(n_terms, *broadcast shape)``.

        ``elevation_rad`` broadcasts against ``distance_m``, so a frame is one call (AT.1).
        """
        d, el = np.broadcast_arrays(
            np.asarray(distance_m, dtype=np.float64), np.asarray(elevation_rad, dtype=np.float64)
        )
        col_aer = np.asarray(column_length(d, el, self.aerosol_scale_height_m), dtype=np.float64)
        # 0 * inf (a zero extinction on an infinite horizontal path) is 0, not NaN
        aer = _scaled(self.gamma_aerosol, col_aer)
        out = np.empty((self.n_terms, *d.shape))
        for k in range(self.n_terms):
            col = column_length(d, el, float(self.scale_heights_m[k]))
            out[k] = _scaled(float(self.gamma_0[k]), np.asarray(col, dtype=np.float64)) + aer
        return out

    def transmittance(self, distance_m: Any, elevation_rad: Any = 0.0) -> NDArray[np.float64]:
        od = self.optical_depths(distance_m, elevation_rad)
        with np.errstate(over="ignore", invalid="ignore"):
            tau = np.tensordot(self.weights, np.exp(-od), axes=1)
        d, el = np.broadcast_arrays(
            np.asarray(distance_m, dtype=np.float64), np.asarray(elevation_rad, dtype=np.float64)
        )
        # An infinite *horizontal* path is opaque; an infinite upward one is not, because the
        # column saturates at H/sin(theta). The mask is therefore per pixel, not a scalar branch.
        tau = np.where(np.isinf(d) & (np.sin(el) <= 0.0), 0.0, tau)
        return np.asarray(tau, dtype=np.float64)

    def gamma_at(self, height_m: Any) -> NDArray[np.float64]:
        """Per class extinction at height h (aerosol included), shape (n_terms, *h.shape)."""
        h = np.asarray(height_m, dtype=np.float64)
        aer = self.gamma_aerosol * np.exp(-h / self.aerosol_scale_height_m)
        return np.stack(
            [
                self.gamma_0[k] * np.exp(-h / self.scale_heights_m[k]) + aer
                for k in range(self.n_terms)
            ]
        )

    def effective_gamma(self) -> float:
        """d→0 slope Σ w_k (γ_k,0 + γ_aer): what a very short horizontal path sees."""
        return float(np.dot(self.weights, self.gamma_0 + self.gamma_aerosol))

    def path_radiance_per_class(
        self,
        distance_m: float,
        elevation_rad: float,
        lb_of_height: Callable[[NDArray[np.float64]], NDArray[np.float64]],
        n_steps: int = 4000,
        u_max: float = 40.0,
        s_max_m: float = 300e3,
    ) -> NDArray[np.float64]:
        """Per class ∫₀^d γ_k(h) τ_k(s) L_B(T(h)) ds (unweighted), shape (n_terms,).

        Each class is integrated in its own optical-depth coordinate u = od_k(s):
        ∫ L_B(T(h(s(u)))) e^{−u} du on a uniform u grid with s(u) from the analytic, monotone
        od_k(s) -- so an opaque class whose emission comes from the first metres is resolved as
        well as a window whose emission comes from kilometres up. Beyond u_max = 40 nothing is
        left (e^{−40}). At θ = 0 the ray stays at h = 0 and the result is closed-form.
        """
        st = math.sin(elevation_rad)
        lb0 = float(lb_of_height(np.zeros(1))[0])
        out = np.zeros(self.n_terms)
        if st <= 0.0:
            if math.isinf(distance_m):
                out[self.gamma_0 + self.gamma_aerosol > 0.0] = lb0
                return out
            tau_k = np.exp(-self.optical_depths(distance_m, 0.0))
            return np.asarray((1.0 - tau_k) * lb0, dtype=np.float64)
        upper = s_max_m if math.isinf(distance_m) else min(float(distance_m), s_max_m)
        if upper <= 0.0:
            return out
        s_dense = np.concatenate([[0.0], np.logspace(-3, math.log10(upper), 20000)])
        od_dense = self.optical_depths(s_dense, elevation_rad)  # (K, n)
        n = n_steps if n_steps % 2 == 0 else n_steps + 1
        for k in range(self.n_terms):
            od_end = float(od_dense[k, -1])
            if od_end <= 0.0:
                continue
            u_top = min(od_end, u_max)
            u = np.linspace(0.0, u_top, n + 1)
            s_of_u = np.interp(u, od_dense[k], s_dense)
            lb = lb_of_height(s_of_u * st)
            out[k] = float(simpson(lb * np.exp(-u), float(u[1] - u[0])))
        return out

    #: Elevation nodes for the per-pixel slant LUT, uniform in **sin θ** (AT.1). The flat-earth
    #: column saturates as ``H/sin θ``, so sin θ is the coordinate the quantity is smooth in;
    #: tabulating in the angle itself leaves a cusp at the horizon that refinement barely touches
    #: (the same trap `irsim.atmosphere.sea` records for the sea profile). Measured on the Boson
    #: LWIR band over 200 m-20 km and 0.05-90 deg against the 4000-step quadrature it replaces:
    #: 33 nodes leave 68 mK of apparent temperature, 65 leave 14 mK, and 129 leave **2.4 mK**
    #: against a 50 mK NETD. Beyond 20 km it degrades -- 65 mK at 100 km -- which is recorded
    #: rather than engineered away, since ADR 0071 bounds the model itself well inside that.
    SLANT_ELEVATION_NODES = 129
    #: The horizontal ray is **node zero**, not a clamped special case (AT.1). Near the horizon the
    #: flat-earth column ``H/sinθ (1 - e^{-d sinθ/H})`` expands to ``d (1 - d sinθ / 2H)``, which is
    #: *linear in sin θ* -- so a grid uniform in sin θ, anchored at the exact horizontal answer, is
    #: continuous at θ = 0 by construction. An earlier draft clamped below 0.25 deg instead and left
    #: a 209 mK step there, four times the NETD, right where long-range scene sits.
    HORIZONTAL_IS_NODE_ZERO = True

    def cumulative_path_table(
        self,
        elevation_rad: float,
        lb_of_height: Callable[[NDArray[np.float64]], NDArray[np.float64]],
        n_steps: int = 2000,
        u_max: float = 40.0,
        s_max_m: float = 300e3,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """``(w, cumulative)`` for one elevation: per class ∫₀^u L_B(T(h(s))) e^{-u'} du'.

        **The grid returned is w = 1 - e^{-u}, not u, and that is the whole design.** Substituting,
        the integral becomes ∫ L_B(T(h)) dw with no exponential left in it, so an isothermal path --
        the case every horizontal ray reduces to -- is *exactly linear* in w and a linear
        interpolation between nodes is exact. Handing back ``u`` instead would invite a caller to
        interpolate in the wrong coordinate: measured on an isothermal path, that costs 1.2e-4
        relative where interpolating in w costs 2e-7.

        The range dependence falls out of this for free, which is the point. :meth:`path_radiance`
        integrates to a *given* distance by choosing the upper limit of the same integral, so one
        cumulative pass answers every range at that elevation -- rather than a two-dimensional
        table over (elevation, range) costing a few thousand quadratures per frame.

        Returns ``w`` of shape ``(n + 1,)``, uniformly spaced, and ``cumulative`` of shape
        ``(n_terms, n + 1)``. A caller converts its own optical depth with ``w = 1 - exp(-od)``.
        """
        st = math.sin(elevation_rad)
        if st <= 0.0:
            raise ValueError(
                "the cumulative table is for upward rays; theta <= 0 has a closed form"
            )
        n = n_steps if n_steps % 2 == 0 else n_steps + 1
        s_dense = np.concatenate([[0.0], np.logspace(-3, math.log10(s_max_m), 20000)])
        od_dense = self.optical_depths(s_dense, elevation_rad)

        # The grid is uniform in **w = 1 - e^{-u}**, not in u. Substituting, the integral becomes
        # ∫ L_B(T(h)) dw with no exponential left in it, so an isothermal path -- the case every
        # horizontal ray reduces to -- is *exactly linear* in w and a trapezoid is exact. A uniform
        # u grid instead spends its points where e^{-u} has already killed the integrand: at 90 deg
        # the whole optical depth is a fraction of one grid step, and the error measured 77 mK
        # against a 50 mK NETD. In w it is three orders of magnitude smaller.
        # `1 - exp(-40)` rounds to exactly 1.0 in float64, and `log1p(-1)` is -inf, so the top of
        # the grid is held one ulp below 1. That still reaches u = 36.7, where e^{-u} is 1e-16.
        w_top = min(1.0 - math.exp(-u_max), float(np.nextafter(1.0, 0.0)))
        w = np.linspace(0.0, w_top, n + 1)
        u = -np.log1p(-w)
        cumulative = np.zeros((self.n_terms, u.size))
        for k in range(self.n_terms):
            if float(od_dense[k, -1]) <= 0.0:
                continue
            s_of_u = np.interp(u, od_dense[k], s_dense)
            lb = lb_of_height(s_of_u * st)
            # Cumulative trapezoid in w. Composite Simpson is only defined on an even number of
            # intervals, so a running Simpson would be exact at alternate nodes and interpolated
            # between them -- worse here than an exact-in-the-limit trapezoid.
            step = float(w[1] - w[0])
            cumulative[k, 1:] = np.cumsum(0.5 * step * (lb[1:] + lb[:-1]))
        return w, cumulative

    def path_radiance(
        self,
        distance_m: float,
        elevation_rad: float,
        lb_of_height: Callable[[NDArray[np.float64]], NDArray[np.float64]],
        n_steps: int = 4000,
        u_max: float = 40.0,
        s_max_m: float = 300e3,
    ) -> float:
        """Σ_k w_k ∫₀^d γ_k(h) τ_k(s) L_B(T(h)) ds  (d = ∞ → the column emission)."""
        per_class = self.path_radiance_per_class(
            distance_m, elevation_rad, lb_of_height, n_steps, u_max, s_max_m
        )
        return float(np.dot(self.weights, per_class))


def fit_exponential_sum(
    distances_m: NDArray[np.float64], tau: NDArray[np.float64], n_terms: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Least-squares (w_k, γ_k) with w ≥ 0, Σ w = 1, γ ≥ 0 to sampled τ(d): the tool for
    fitting a band's curve of growth from a spectral calculation (M8.7) or a MODTRAN run."""
    d = np.asarray(distances_m, dtype=np.float64)
    t = np.asarray(tau, dtype=np.float64)
    if n_terms < 1 or d.shape != t.shape or np.any(d < 0.0) or np.any((t <= 0.0) | (t > 1.0)):
        raise ValueError("need matching distances >= 0 and 0 < tau <= 1")
    if n_terms == 1:
        g_single = float(-np.sum(d * np.log(t)) / np.sum(d * d))
        return np.array([1.0]), np.array([g_single])
    g0 = float(-np.log(t[-1]) / d[-1])
    x0 = np.concatenate([np.zeros(n_terms - 1), np.log(g0 * np.logspace(-1, 1, n_terms))])

    def unpack(x: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        logits = np.concatenate([[0.0], x[: n_terms - 1]])
        w = np.exp(logits - logits.max())
        w = w / w.sum()
        return w, np.exp(x[n_terms - 1 :])

    def residual(x: NDArray[np.float64]) -> NDArray[np.float64]:
        w, g = unpack(x)
        model = np.exp(-np.outer(g, d)).T @ w
        return np.asarray(np.log(model) - np.log(t), dtype=np.float64)

    best = None
    for trial in range(6):
        start = x0 + (0.0 if trial == 0 else np.random.default_rng(trial).normal(0.0, 1.0, x0.size))
        res = least_squares(residual, start, method="lm" if d.size >= x0.size else "trf")
        if best is None or res.cost < best.cost:
            best = res
    assert best is not None
    w, g = unpack(best.x)
    order = np.argsort(g)
    return w[order], g[order]


def exponential_sum_from_piecewise(
    response: SpectralResponse,
    gamma_of_lambda: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    breakpoints_um: tuple[float, ...],
    t_ref_k: float = 300.0,
) -> ExponentialSum:
    """The exact exponential sum of a piecewise-constant γ(λ): one term per interval between
    breakpoints, weight = the interval's Planck-weighted share, γ = γ(λ) at its midpoint. Used
    to check the model against M8.7's spectral quadrature (horizontal paths, no profile)."""
    lo, hi = response.support_um
    edges = sorted({lo, hi, *[b for b in breakpoints_um if lo < b < hi]})
    grid = quadrature_grid(response)
    dl = float(grid[1] - grid[0])
    weight = response.resampled(grid) * spectral_radiance(
        grid, np.asarray(t_ref_k, dtype=np.float64)
    )
    total = float(simpson(weight, dl))
    w = []
    g = []
    for a, b in zip(edges[:-1], edges[1:], strict=True):
        mask = (grid >= a) & (grid < b) if b < hi else (grid >= a) & (grid <= b)
        w.append(float(simpson(np.where(mask, weight, 0.0), dl)) / total)
        g.append(float(gamma_of_lambda(np.array([0.5 * (a + b)]))[0]))
    w_arr = np.asarray(w) / sum(w)
    return ExponentialSum(
        weights=w_arr,
        gamma_0=np.asarray(g),
        scale_heights_m=np.full(len(g), 1e30),
        gamma_aerosol=0.0,
        aerosol_scale_height_m=1e30,
        names=tuple(f"{a}-{b}um" for a, b in zip(edges[:-1], edges[1:], strict=True)),
    )


class LayeredAtmosphere:
    """The MS.1 model bound to a preset and the shared WeatherSeries (CLAUDE.md #6)."""

    def __init__(
        self,
        preset: AtmospherePreset,
        weather: WeatherSeries,
        luts: Mapping[str, BandLUT] | None = None,
        responses: Mapping[str, SpectralResponse] | None = None,
    ) -> None:
        if not isinstance(weather, WeatherSeries):
            raise TypeError("LayeredAtmosphere takes the WeatherSeries object, never a path (#6)")
        self._preset = preset
        self._weather = weather
        self._luts = dict(luts or {})
        self._responses = dict(responses or {})
        self._weights: dict[str, NDArray[np.float64]] = {}
        #: AT.1's per-elevation cumulative tables, keyed by (band, quantity, exponential sum).
        #: Keyed on the extinction arrays and the surface air temperature, so it turns over
        #: exactly when the weather does and not once per frame.
        self._slant_cache: dict[tuple[Any, ...], tuple[Any, Any, float, int]] = {}

    #: How many slant tables to keep. A render walks one weather sample at a time, so a handful
    #: covers the reuse within a frame and across neighbouring frames without growing all day.
    _SLANT_CACHE_MAX = 8

    @property
    def preset(self) -> AtmospherePreset:
        return self._preset

    @property
    def weather(self) -> WeatherSeries:
        return self._weather

    def weights(self, band: str) -> NDArray[np.float64]:
        if band not in self._weights:
            response = self._responses.get(band)
            if response is None:
                # Loud, because the consequence is invisible in the output (AT.2). Without the
                # camera's R(λ) the band is split by a nominal top-hat, and on the shipped InSb
                # response that moves the MWIR `h2o_wing` weight from 0.0192 to 0.1303 -- a 6.8x
                # change in how much of the band is treated as a water wing -- while every frame
                # still looks exactly like a frame.
                warnings.warn(
                    f"LayeredAtmosphere has no spectral response for band {band!r}, so its "
                    "spectral-class weights come from a nominal top-hat rather than from the "
                    "camera -- and since AT.10 its spectral *classes* too, since they are derived "
                    "from the band's own span. Pass responses= (irsim.radiometry.lut_files."
                    "load_band_response_for_config) -- measured on the shipped InSb MWIR "
                    "response, this is a 6.8x error in the h2o_wing class weight.",
                    stacklevel=2,
                )
            self._weights[band] = class_weights(band, response)
        return self._weights[band]

    # -- the per-band exponential sum at time t ---------------------------------------
    def exponential_sum(self, band: str, t_s: float) -> ExponentialSum:
        sample = self._weather.at(t_s)
        w_h2o = sample.absolute_humidity_g_m3
        coeffs = self._preset.bands[band]
        gamma_mol = gamma_molecular(w_h2o, coeffs.gamma0_per_m, coeffs.beta_per_m_per_g_m3)
        vis = self._preset.bands[ANCHOR_BAND]
        gamma_mol_vis = gamma_molecular(w_h2o, vis.gamma0_per_m, vis.beta_per_m_per_g_m3)
        gamma_aer = coeffs.aerosol_ratio_to_visible * gamma_aerosol_visible(
            sample.visibility_m, gamma_mol_vis
        )
        gamma_grey = gamma_mol + gamma_aer
        classes = classes_for(band, self._responses.get(band))
        weights = self.weights(band)
        profile = self._preset.profile
        heights = np.array(
            [
                profile.water_vapour_scale_height_m
                if c.kind == "water"
                else profile.air_scale_height_m
                for c in classes
            ]
        )
        base = np.array(
            [
                (
                    c.multiplier
                    if c.opaque
                    else c.multiplier * (gamma_mol if c.kind == "water" else 1.0)
                )
                for c in classes
            ]
        )
        opaque = np.array([c.opaque for c in classes])
        free = ~opaque
        target = math.exp(-gamma_grey * ANCHOR_DISTANCE_M) * float(weights[free].sum())

        def anchored(scale: float) -> float:
            g = np.where(free, scale * base, base)
            return float(np.sum(weights[free] * np.exp(-(g[free] + gamma_aer) * ANCHOR_DISTANCE_M)))

        if not free.any() or base[free].max() <= 0.0:
            scale = 1.0
        else:
            hi = 1.0
            while anchored(hi) > target and hi < 1e8:
                hi *= 10.0
            if anchored(0.0) < target - 1e-12:
                raise ValueError(
                    f"band {band!r}: the aerosol alone makes tau(200 m) < the grey preset's; "
                    "class table and preset disagree"
                )
            scale = (
                0.0 if anchored(0.0) <= target else brentq(lambda x: anchored(x) - target, 0.0, hi)
            )
        gamma_0 = np.where(free, scale * base, base)
        return ExponentialSum(
            weights=weights,
            gamma_0=np.asarray(gamma_0, dtype=np.float64),
            scale_heights_m=heights,
            gamma_aerosol=gamma_aer,
            aerosol_scale_height_m=profile.aerosol_scale_height_m,
            names=tuple(c.name for c in classes),
        )

    # -- temperatures and radiances along the column ------------------------------------
    def air_temperature_at(self, t_s: float, height_m: Any) -> NDArray[np.float64]:
        t0 = self._weather.at(t_s).t_air_k
        p = self._preset.profile
        h = np.minimum(np.asarray(height_m, dtype=np.float64), p.tropopause_m)
        return np.asarray(t0 - p.lapse_rate_k_per_m * h, dtype=np.float64)

    def _slant_tables(
        self, band: str, t_s: float, quantity: Quantity
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], float, int]:
        """``(sin nodes, cumulative, w step, n)`` for the per-pixel slant path (AT.1), cached.

        Cached on ``(band, quantity, thermal tick)`` because the table depends on the temperature
        profile and nothing else that moves within a tick. Rebuilding it per frame would dominate
        the stage; rebuilding it per tick is what the rest of the thermal path already does.
        """
        es = self.exponential_sum(band, t_s)
        # Keyed on the *content* the table depends on, never on object identity. An earlier draft
        # used `id(es)`, which is wrong twice over: `exponential_sum` builds a fresh object on every
        # call so the cache never hit, and CPython reuses the ids of dead objects, so it could have
        # returned a table built for different weather. The extinction arrays and the surface air
        # temperature are what the table is a function of; equal values mean an identical table.
        key = (
            band,
            str(quantity),
            es.weights.tobytes(),
            es.gamma_0.tobytes(),
            es.scale_heights_m.tobytes(),
            float(es.gamma_aerosol),
            float(es.aerosol_scale_height_m),
            float(self._weather.at(t_s).t_air_k),
        )
        cached = self._slant_cache.get(key)
        if cached is not None:
            return cached
        lb_of_height = self._lb_of_height(band, t_s, quantity)
        sin_nodes = np.linspace(0.0, 1.0, ExponentialSum.SLANT_ELEVATION_NODES)
        tables = []
        w_grid = None
        for sin_theta in sin_nodes[1:]:
            w_grid, cumulative = es.cumulative_path_table(float(math.asin(sin_theta)), lb_of_height)
            tables.append(cumulative)
        assert w_grid is not None
        # Node zero is the horizontal ray, and it is *analytic*: the ray never leaves h = 0, so
        # L_B is constant along it and the cumulative integral is exactly `L_B(0) · w`. Anchoring
        # the grid on the closed form rather than on a near-horizontal quadrature is what makes the
        # join at θ = 0 continuous instead of a step.
        lb0 = float(lb_of_height(np.zeros(1))[0])
        tables.insert(0, np.broadcast_to(lb0 * w_grid, (es.n_terms, w_grid.size)).copy())
        stacked = np.stack(tables)  # (nodes, K, n + 1)
        n = w_grid.size - 1
        # The w grid is uniform, so a pixel's position along it is exact arithmetic rather than a
        # search: w = 1 - exp(-od), index = w / step.
        w_step = float(w_grid[1] - w_grid[0])
        out = (sin_nodes, stacked, w_step, n)
        # Bounded: a time-lapse render walks the weather and would otherwise grow one table per
        # frame. Oldest first, because a render moves forward through the day.
        if len(self._slant_cache) >= self._SLANT_CACHE_MAX:
            self._slant_cache.pop(next(iter(self._slant_cache)))
        self._slant_cache[key] = out
        return out

    def path_radiance_plane(
        self,
        band: str,
        t_s: float,
        distance_m: Any,
        elevation_rad: Any,
        quantity: Quantity = "lb",
    ) -> NDArray[np.float64]:
        """L_path per pixel, each along its own slant ray (AT.1).

        The stage used to pass elevation **0.0 unconditionally**, so every resolved pixel was given
        surface-density extinction and surface-temperature emission over its whole slant range,
        while the unresolved point-target path beside it used the target's real elevation and so did
        the sky behind it. Measured on `us_standard_clear` at 5 km, LWIR: L_path 18.27 W/m²/sr
        horizontal against 11.60 at 45°, a **37 %** over-estimate on every pixel of every sloping
        ray.

        Rays at or below the horizon take the exact horizontal closed form; everything above it
        goes through the table, whose **first node is that same closed form**, so the join at θ = 0
        is continuous by construction rather than by a tolerance.
        """
        d, el = np.broadcast_arrays(
            np.asarray(distance_m, dtype=np.float64), np.asarray(elevation_rad, dtype=np.float64)
        )
        es = self.exponential_sum(band, t_s)
        out = np.zeros(d.shape, dtype=np.float64)

        # Only rays that do not rise at all. Everything above the horizon goes through the table,
        # whose first node *is* the horizontal answer, so there is no clamp and no step.
        flat = np.sin(el) <= 0.0
        if flat.any():
            lb0 = self.air_radiance(band, t_s, quantity)
            with np.errstate(over="ignore", invalid="ignore"):
                tau_k = np.exp(-es.optical_depths(d, 0.0))
            horizontal = np.tensordot(es.weights, 1.0 - tau_k, axes=1) * lb0
            out = np.where(flat, horizontal, out)
        if flat.all():
            return out

        sin_nodes, tables, w_step, n = self._slant_tables(band, t_s, quantity)
        sin_el = np.clip(np.sin(el), 0.0, 1.0)
        # Which two elevation nodes bracket each pixel, and how far between them it sits.
        node = np.clip(np.searchsorted(sin_nodes, sin_el, side="right") - 1, 0, sin_nodes.size - 2)
        span = sin_nodes[node + 1] - sin_nodes[node]
        safe_span = np.where(span > 0.0, span, 1.0)
        blend = np.where(span > 0.0, (sin_el - sin_nodes[node]) / safe_span, 0.0)

        od = es.optical_depths(d, el)  # (K, ...)
        with np.errstate(over="ignore", invalid="ignore"):
            position = np.clip((1.0 - np.exp(-od)) / w_step, 0.0, float(n) - 1e-9)
        lower = position.astype(np.int64)
        frac = position - lower

        slant = np.zeros(d.shape, dtype=np.float64)
        for k in range(es.n_terms):
            i0, f = lower[k], frac[k]
            low = tables[node, k, i0] * (1.0 - f) + tables[node, k, i0 + 1] * f
            high = tables[node + 1, k, i0] * (1.0 - f) + tables[node + 1, k, i0 + 1] * f
            slant += es.weights[k] * (low * (1.0 - blend) + high * blend)
        return np.asarray(np.where(flat, out, slant))

    def _lb_of_height(
        self, band: str, t_s: float, quantity: Quantity
    ) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
        if band not in self._luts:
            raise KeyError(f"no LUT for band {band!r}; have {sorted(self._luts)}")
        lut = self._luts[band]

        def lb(height_m: NDArray[np.float64]) -> NDArray[np.float64]:
            t = self.air_temperature_at(t_s, height_m)
            return np.asarray(lut.lookup(t, quantity), dtype=np.float64)

        return lb

    def air_radiance(self, band: str, t_s: float, quantity: Quantity = "lb") -> float:
        """L_B(T_air) at the surface, from this model's own LUT: the level a horizontal ray's
        path radiance tends to, and what a fast path must use to stay bit-comparable (§7.1)."""
        return float(self._lb_of_height(band, t_s, quantity)(np.zeros(1))[0])

    def transmittance(
        self, band: str, t_s: float, distance_m: Any, elevation_rad: float = 0.0
    ) -> NDArray[np.float64]:
        return self.exponential_sum(band, t_s).transmittance(distance_m, elevation_rad)

    def path_radiance(
        self,
        band: str,
        t_s: float,
        distance_m: float,
        elevation_rad: float = 0.0,
        quantity: Quantity = "lb",
    ) -> float:
        es = self.exponential_sum(band, t_s)
        return es.path_radiance(distance_m, elevation_rad, self._lb_of_height(band, t_s, quantity))

    def class_transmittances(
        self, band: str, t_s: float, distance_m: float, elevation_rad: float = 0.0
    ) -> NDArray[np.float64]:
        """τ_k(d, θ) per spectral class (the band τ is Σ w_k τ_k)."""
        es = self.exponential_sum(band, t_s)
        return np.asarray(
            np.exp(-es.optical_depths(float(distance_m), elevation_rad)), dtype=np.float64
        )

    def sky_beyond_per_class(
        self,
        band: str,
        t_s: float,
        distance_m: float,
        elevation_rad: float,
        quantity: Quantity = "lb",
    ) -> NDArray[np.float64]:
        """Per class, the column emission beyond range R along the ray as seen *from R*:
        L_beyond,k = (L_sky,k − L_path,k(R)) / τ_k(R). A target at R occults exactly this."""
        es = self.exponential_sum(band, t_s)
        lb = self._lb_of_height(band, t_s, quantity)
        sky_k = es.path_radiance_per_class(math.inf, elevation_rad, lb)
        path_k = es.path_radiance_per_class(float(distance_m), elevation_rad, lb)
        tau_k = self.class_transmittances(band, t_s, distance_m, elevation_rad)
        with np.errstate(divide="ignore", invalid="ignore"):
            beyond = np.where(tau_k > 1e-300, (sky_k - path_k) / tau_k, 0.0)
        return np.asarray(np.maximum(beyond, 0.0), dtype=np.float64)

    def sky_beyond(
        self,
        band: str,
        t_s: float,
        distance_m: float,
        elevation_rad: float,
        quantity: Quantity = "lb",
    ) -> float:
        """The τ_k-weighted effective radiance beyond R (a target at it has zero excess)."""
        es = self.exponential_sum(band, t_s)
        tau_k = self.class_transmittances(band, t_s, distance_m, elevation_rad)
        beyond = self.sky_beyond_per_class(band, t_s, distance_m, elevation_rad, quantity)
        wt = es.weights * tau_k
        return float(np.dot(wt, beyond) / wt.sum()) if wt.sum() > 0.0 else 0.0

    def sky_radiance(
        self, band: str, t_s: float, elevation_rad: float, quantity: Quantity = "lb"
    ) -> float:
        """L_sky,B(θ) = L_path,B(∞, θ): the column's own emission (space adds nothing)."""
        return self.path_radiance(band, t_s, math.inf, elevation_rad, quantity)

    def apparent_sky_temperature_k(self, band: str, t_s: float, elevation_rad: float) -> float:
        lut = self._luts[band]
        return float(
            lut.apparent_temperature(np.asarray(self.sky_radiance(band, t_s, elevation_rad)))[()]
        )

    def apply(
        self,
        band: str,
        t_s: float,
        l_band: Any,
        distance_m: Any,
        elevation_rad: float = 0.0,
        quantity: Quantity = "lb",
    ) -> NDArray[np.floating]:
        """L' = τ L + L_path per pixel; horizontal paths use the closed form (any shape), a
        slant path with a scalar distance uses the quadrature."""
        lb = np.asarray(l_band)
        if lb.dtype == np.float16:
            raise TypeError("l_band is float16 (non-negotiable #2)")
        es = self.exponential_sum(band, t_s)
        d = np.asarray(distance_m, dtype=np.float64)
        tau = es.transmittance(d, elevation_rad)
        path: NDArray[np.float64] | float
        if math.sin(elevation_rad) <= 0.0:
            lb_air = self.air_radiance(band, t_s, quantity)
            tau_k = np.exp(-es.optical_depths(d, 0.0))
            path = np.asarray(
                np.tensordot(es.weights, 1.0 - tau_k, axes=1) * lb_air, dtype=np.float64
            )
        else:
            if d.shape != ():
                raise ValueError(
                    "slant-path apply takes a scalar distance (per-pixel slant paths: MS.8)"
                )
            path = es.path_radiance(
                float(d), elevation_rad, self._lb_of_height(band, t_s, quantity)
            )
        out = tau * lb.astype(np.float64) + path
        return np.asarray(
            out, dtype=lb.dtype if np.issubdtype(lb.dtype, np.floating) else np.float64
        )
