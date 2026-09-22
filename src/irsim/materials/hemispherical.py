"""Hemispherical emissivity: the number the thermal solver needs, which is not the one it sees.

An infrared camera measures ε(θ) along one ray. An energy balance radiates into the whole sky, so
it needs

    ε_hemi = 2 ∫₀^{π/2} ε(θ) cos θ sin θ dθ = 2 ∫₀¹ ε(μ) μ dμ

and the two differ by more than a rounding: water in the Boson band reads **0.990** at nadir and
**0.951** hemispherically, because the cos θ sin θ weight peaks at 45° and water's ε has already
begun to collapse there. Handing a solver the directional band value overstates its radiative
cooling by about 4 %, which under a clear sky is a fraction of a kelvin of surface temperature —
small, systematic, and in the same direction everywhere.

**The sign flips for metals.** Aluminium's ε *rises* with angle, so its ε_hemi is **above** its
normal value. A model that assumed ε_hemi ≤ ε(0) would be right for every dielectric and wrong for
every metal, which is the kind of assumption that survives a long time.

For the **total** (all-wavelength) form the honest difficulty is coverage, not quadrature: at 300 K
more than a third of Planck's weight lies beyond 13.5 µm, where no configured band says anything.
:func:`total_hemispherical_emissivity` extends the longest band's value outwards — the standard
assumption for a thermal solver — and **returns the fraction of the weight that came from that
extension**, so a caller can see how much of its answer is assumption (ADR 0043).

docs/physics-model.md §6.1, §6.2, §4.2; ADR 0043
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from irsim.radiometry.constants import C1L, C2

__all__ = [
    "GL_NODES",
    "NEGLIGIBLE_BAND_WEIGHT",
    "hemispherical_emissivity",
    "band_hemispherical_emissivity",
    "TotalHemispherical",
    "total_hemispherical_emissivity",
    "extrapolation_breakdown",
]

#: A band holding less than this fraction of Planck's weight at the working temperature is not
#: integrated. It cannot move the energy balance, and asking a material for an angular model in a
#: band the solver will never see turns an irrelevant §4.2 Level C violation into a hard failure.
NEGLIGIBLE_BAND_WEIGHT = 1e-4

#: Gauss–Legendre nodes for the µ integral. The integrand is ε(µ)·µ, smooth for every level, so
#: 32 nodes is far past convergence -- 16 and 64 agree to better than 1e-6 on every material
#: tested, which is what the test asserts rather than the node count being "enough".
GL_NODES = 32


def hemispherical_emissivity(
    epsilon_of_cos: Callable[[NDArray[np.float64]], Any], n_nodes: int = GL_NODES
) -> float:
    """2 ∫₀¹ ε(µ) µ dµ by Gauss–Legendre, for any callable ε(cos θ).

    Written in µ = cos θ rather than in θ: the substitution turns cos θ sin θ dθ into µ dµ, so the
    integrand has no trigonometry in it and the quadrature converges on a polynomial-like function
    instead of an oscillatory one.
    """
    if n_nodes < 2:
        raise ValueError("n_nodes must be at least 2")
    nodes, weights = np.polynomial.legendre.leggauss(n_nodes)
    mu = 0.5 * (nodes + 1.0)
    d_mu = 0.5 * weights
    epsilon = np.asarray(epsilon_of_cos(mu), dtype=np.float64)
    if epsilon.shape != mu.shape:
        raise ValueError("epsilon_of_cos must return one value per cos θ node")
    return float(2.0 * np.sum(epsilon * mu * d_mu))


def band_hemispherical_emissivity(
    material: Any,
    band: str,
    response: Any = None,
    n_nodes: int = GL_NODES,
    **kwargs: Any,
) -> float:
    """ε_hemi for one material in one band, through the M7.7 level dispatch."""
    from irsim.materials.directional import directional_emissivity

    return hemispherical_emissivity(
        lambda mu: np.asarray(
            directional_emissivity(material, band, mu, response, **kwargs), dtype=np.float64
        ),
        n_nodes,
    )


@dataclass(frozen=True)
class TotalHemispherical:
    """The total hemispherical emissivity, and how much of it rests on an assumption."""

    value: float
    temperature_k: float
    #: Fraction of Planck's weight at this temperature that fell outside every configured band
    #: and was filled by extending the nearest one. **Report this.** At 300 K over the four
    #: §12.1 bands it is large -- most of the thermal tail lies beyond 13.5 µm.
    extrapolated_fraction: float
    bands_used: tuple[str, ...]
    #: Where that extrapolated weight sits, summing to :attr:`extrapolated_fraction` (AT.7). The
    #: total alone is not actionable, because the three are assumptions of very different
    #: strength and which one dominates **flips with temperature**: at 300 K it is 0.508 red tail
    #: against 0.099 interior gaps, and at 800 K it is 0.071 against 0.394.
    red_tail_fraction: float = 0.0
    #: Weight in the holes *between* configured bands -- 1.7-3.0 µm and 5.0-7.5 µm for the §12.1
    #: set. "Extend the nearest band" is a far weaker claim here than in the tail: a real spectrum
    #: is bounded either side and can do anything in between, and a reststrahlen feature lives
    #: exactly there.
    interior_gap_fraction: float = 0.0
    #: Weight below the shortest configured band. Negligible at ambient; it is the one that would
    #: grow for a genuinely hot source.
    blue_end_fraction: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.value <= 1.0:
            raise ValueError(f"total hemispherical emissivity {self.value} outside [0, 1]")

    def summary(self) -> str:
        """One line for a report that quotes an absolute temperature (AT.7)."""
        return (
            f"epsilon_hemi = {self.value:.4f} at {self.temperature_k:.1f} K over "
            f"{'+'.join(self.bands_used)}; {self.extrapolated_fraction:.3f} of Planck's weight is "
            f"extrapolated ({self.red_tail_fraction:.3f} red tail, "
            f"{self.interior_gap_fraction:.3f} interior gaps, "
            f"{self.blue_end_fraction:.3f} blue end)"
        )


def _planck(lam_um: NDArray[np.float64], t_k: float) -> NDArray[np.float64]:
    return np.asarray(C1L / lam_um**5 / np.expm1(C2 / (lam_um * t_k)))


def total_hemispherical_emissivity(
    material: Any,
    temperature_k: float,
    bands: dict[str, tuple[float, float]] | None = None,
    responses: dict[str, Any] | None = None,
    lambda_min_um: float = 1.0,
    lambda_max_um: float = 100.0,
    n_lambda: int = 2001,
    n_nodes: int = GL_NODES,
    **kwargs: Any,
) -> TotalHemispherical:
    """Planck-weighted ε_hemi over all wavelengths at ``temperature_k`` (§6.1's ε for the balance).

    Each configured band contributes its own ε_hemi over its own wavelength interval; everything
    outside every band takes the value of the **nearest** band, and the Planck weight of that
    region is returned as ``extrapolated_fraction``. Beyond 13.5 µm that assumption is the usual
    one for a thermal solver — real dielectrics stay high and flat out there — but it is an
    assumption, and a solver that cannot see how much of its ε rests on one cannot report its own
    uncertainty.
    """
    from irsim.config.bands import NOMINAL_RANGES_UM

    ranges = dict(NOMINAL_RANGES_UM) if bands is None else dict(bands)
    if not ranges:
        raise ValueError("no bands to integrate over")
    lam = np.linspace(lambda_min_um, lambda_max_um, n_lambda)
    weight = _planck(lam, float(temperature_k))
    total_weight = float(np.trapezoid(weight, lam))

    # Skip a band that carries no Planck weight at this temperature. At 300 K, NIR and SWIR
    # together hold about 1e-9 of the exitance, so asking a material for its angular model there
    # is asking a question the answer to which cannot move the energy balance -- and it *can*
    # fail, since §4.2's Level C bound is about thermal-band roughness and a material may legally
    # be Level-C-invalid in a band the solver will never integrate.
    weights = {
        band: float(np.trapezoid(weight * ((lam >= lo) & (lam <= hi)).astype(np.float64), lam))
        / total_weight
        for band, (lo, hi) in ranges.items()
    }
    ranges = {b: r for b, r in ranges.items() if weights[b] >= NEGLIGIBLE_BAND_WEIGHT}
    if not ranges:
        raise ValueError(
            f"no configured band carries Planck weight at {temperature_k:.1f} K (weights {weights})"
        )
    per_band = {
        band: band_hemispherical_emissivity(
            material, band, (responses or {}).get(band), n_nodes, **kwargs
        )
        for band in ranges
    }
    epsilon = np.full_like(lam, np.nan)
    for band, (lo, hi) in ranges.items():
        epsilon[(lam >= lo) & (lam <= hi)] = per_band[band]

    outside = np.isnan(epsilon)
    centres = {band: 0.5 * (lo + hi) for band, (lo, hi) in ranges.items()}
    for index in np.flatnonzero(outside):
        nearest = min(centres, key=lambda b: abs(centres[b] - float(lam[index])))
        epsilon[index] = per_band[nearest]

    value = float(np.trapezoid(epsilon * weight, lam) / total_weight)
    # Integrate the mask against the full grid rather than over the selected points: the outside
    # region is not contiguous, and trapezoid over `lam[outside]` would bridge every band with one
    # wide trapezoid and report far more extrapolation than there is (0.958 against 0.639 here).
    extrapolated = float(np.trapezoid(weight * outside.astype(np.float64), lam) / total_weight)
    # AT.7: *where* that weight sits, on the same grid and the same mask, so the three parts add
    # up to `extrapolated` by construction rather than by a second integration that could drift.
    blue_edge = min(lo for lo, _ in ranges.values())
    red_edge = max(hi for _, hi in ranges.values())

    def _share(region: NDArray[np.bool_]) -> float:
        masked = (outside & region).astype(np.float64)
        return float(np.trapezoid(weight * masked, lam) / total_weight)

    return TotalHemispherical(
        value=value,
        temperature_k=float(temperature_k),
        extrapolated_fraction=extrapolated,
        bands_used=tuple(sorted(ranges)),
        red_tail_fraction=_share(lam > red_edge),
        interior_gap_fraction=_share((lam >= blue_edge) & (lam <= red_edge)),
        blue_end_fraction=_share(lam < blue_edge),
    )


def extrapolation_breakdown(
    temperature_k: float,
    bands: dict[str, tuple[float, float]] | None = None,
    lambda_min_um: float = 1.0,
    lambda_max_um: float = 100.0,
    n_lambda: int = 2001,
) -> TotalHemispherical:
    """The extrapolated fraction and its decomposition **without a material** (AT.7).

    Measured across the whole shipped library, the fraction is **bit-identical for every
    material** -- 0.606695073 at 300 K for all twenty -- because it is a property of the band set
    and the Planck weight, not of anything the material does. So a report that wants to say how
    much of a scene's emissivity is assumption does not need a material, and asking for one
    invites the reader to think the number varies with it.

    ``value`` is returned as :data:`math.nan`'s stand-in 0.0 here and carries no meaning; read
    :attr:`~TotalHemispherical.extrapolated_fraction` and the three shares.
    """
    from irsim.config.bands import NOMINAL_RANGES_UM

    ranges = dict(NOMINAL_RANGES_UM) if bands is None else dict(bands)
    if not ranges:
        raise ValueError("no bands to integrate over")
    lam = np.linspace(lambda_min_um, lambda_max_um, n_lambda)
    weight = _planck(lam, float(temperature_k))
    total_weight = float(np.trapezoid(weight, lam))
    shares = {
        band: float(np.trapezoid(weight * ((lam >= lo) & (lam <= hi)), lam)) / total_weight
        for band, (lo, hi) in ranges.items()
    }
    ranges = {b: r for b, r in ranges.items() if shares[b] >= NEGLIGIBLE_BAND_WEIGHT}
    if not ranges:
        raise ValueError(
            f"no configured band carries Planck weight at {temperature_k:.1f} K (weights {shares})"
        )
    inside = np.zeros_like(lam, dtype=bool)
    for lo, hi in ranges.values():
        inside |= (lam >= lo) & (lam <= hi)
    outside = ~inside
    blue_edge = min(lo for lo, _ in ranges.values())
    red_edge = max(hi for _, hi in ranges.values())

    def _share(region: NDArray[np.bool_]) -> float:
        return float(np.trapezoid(weight * (outside & region).astype(np.float64), lam)) / (
            total_weight
        )

    return TotalHemispherical(
        value=0.0,
        temperature_k=float(temperature_k),
        extrapolated_fraction=_share(np.ones_like(lam, dtype=bool)),
        bands_used=tuple(sorted(ranges)),
        red_tail_fraction=_share(lam > red_edge),
        interior_gap_fraction=_share((lam >= blue_edge) & (lam <= red_edge)),
        blue_end_fraction=_share(lam < blue_edge),
    )
