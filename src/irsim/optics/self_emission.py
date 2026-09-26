"""Optics self-emission: the warm lens and housing the detector also sees.

In LWIR the optics glow. For a single-lens uncooled core (docs/physics-model.md §8.2):

    Φ_self ≈ A_d · Ω_eff · (1 − τ_opt) · L_B(T_housing),     Ω_eff = π / (4F² + 1)

i.e. the lens is a grey body of emissivity 1 − τ_opt (ρ_lens = 0, ADR 0016) at the housing
temperature, filling the same cone as the scene. Because T_housing drifts, this term is the physical
origin of shutterless drift and the reason for periodic flat-field correction (§11.2).

The general N-element form for a window/lens/mirror stack, each element attenuated by everything
downstream of it (§8.2, [R19]):

    L_self = Σ_i ε_i L_B(T_i) Π_{j>i} τ_j,      ε_i = 1 − τ_i − ρ_i   (Kirchhoff, non-negotiable #4)

Everything here is in radiance/power space -- never kelvin (non-negotiable #3). The caller supplies
band radiances L_B(T) from the band LUT or the closed-form top-hat, so the module is band-agnostic.

Where that term lands on the array is :func:`housing_power_field`: a pixel off axis sees less of
the scene and more of the housing, so the non-scene power is A_d Ω_eff (1 − τ_opt RI_ij) L_B(T_h),
not one number per frame (§8.2 revised 2026-09-26, ADR 0145).

docs/physics-model.md §8.2, §2 (Φ_self), §11.2
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from irsim.optics.aperture import aperture_factor

__all__ = [
    "KIRCHHOFF_TOL",
    "OpticalElement",
    "self_emission_power",
    "housing_power_field",
    "housing_power_axis",
    "stack_transmittance",
    "stack_self_radiance",
]

KIRCHHOFF_TOL = 1e-6


def self_emission_power(
    active_area_m2: float, f_number: float, tau_opt: float, lb_housing: float
) -> float:
    """Φ_self = A_d Ω_eff (1 − τ_opt) L_B(T_housing), single-lens form (§8.2). Units follow L_B."""
    if not 0.0 < tau_opt <= 1.0:
        raise ValueError(f"tau_opt must be in (0, 1], got {tau_opt}")
    if not active_area_m2 > 0.0:
        raise ValueError("active_area_m2 must be positive")
    if lb_housing < 0.0:
        raise ValueError("lb_housing is a band radiance and cannot be negative")
    return active_area_m2 * aperture_factor(f_number) * (1.0 - tau_opt) * lb_housing


def housing_power_field(
    active_area_m2: float,
    f_number: float,
    tau_opt: float,
    lb_housing: float,
    relative_illumination: object,
) -> NDArray[np.float64]:
    """Everything a pixel receives that is not scene: A_d Ω_eff (1 − τ_opt RI_ij) L_B(T_housing).

    docs/physics-model.md §8.2 ("Where the housing radiation lands on the array"), ADR 0145. The
    pixel's hemisphere splits into the aperture cone, of projected solid angle Ω_eff RI_ij, and the
    inside of the camera. With lens and housing at one temperature the power on a uniform scene is

        Φ_ij = A_d π L_h + A_d Ω_eff τ_opt RI_ij (L_scene − L_h).

    The uniform pedestal A_d π L_h is balanced by the detector's own emission at T_FPA and is not
    carried (ADR 0145); referencing it to the optical axis instead leaves

        Φ_ij = A_d Ω_eff [L_h + τ_opt RI_ij (L_scene − L_h)],

    whose non-scene part is this function. On axis (RI = 1) it equals :func:`self_emission_power`
    exactly, so every on-axis number the project has measured is unchanged; off axis it grows,
    because the pixel sees less scene and more housing. Units follow L_B.
    """
    ri = np.asarray(relative_illumination, dtype=np.float64)
    if np.any(ri <= 0.0) or np.any(ri > 1.0 + 1e-12) or not np.all(np.isfinite(ri)):
        raise ValueError("relative illumination must lie in (0, 1]")
    axis_housing = housing_power_axis(active_area_m2, f_number, tau_opt, lb_housing)
    return np.asarray(axis_housing * (1.0 - tau_opt * ri), dtype=np.float64)


def housing_power_axis(
    active_area_m2: float, f_number: float, tau_opt: float, lb_housing: float
) -> float:
    """A_d Ω_eff L_B(T_housing): the housing power a pixel would receive with no scene in its cone.

    The two scalars of :func:`housing_power_field` a device kernel needs are this and L_h itself:
    Φ_ij = τ Ω_eff RI_ij A_d (L_scene − L_h) + this (§8.2, ADR 0145).
    """
    self_emission_power(active_area_m2, f_number, tau_opt, lb_housing)  # validates the scalars
    return active_area_m2 * aperture_factor(f_number) * lb_housing


@dataclass(frozen=True)
class OpticalElement:
    """One element of the optical train: transmittance τ, reflectance ρ, temperature T.

    Emissivity is derived, ε = 1 − τ − ρ, never authored (non-negotiable #4); τ + ρ > 1 raises.
    """

    tau: float
    temperature_k: float
    rho: float = 0.0
    name: str = ""

    def __post_init__(self) -> None:
        if not 0.0 <= self.tau <= 1.0 or not 0.0 <= self.rho <= 1.0:
            raise ValueError(f"{self.name or 'element'}: tau and rho must lie in [0, 1]")
        if self.tau + self.rho > 1.0 + KIRCHHOFF_TOL:
            raise ValueError(
                f"{self.name or 'element'}: tau + rho = {self.tau + self.rho:.6f} > 1 violates "
                "Kirchhoff closure (epsilon = 1 - tau - rho would be negative)"
            )
        if self.temperature_k <= 0.0:
            raise ValueError("temperature_k must be positive")

    @property
    def emissivity(self) -> float:
        return max(0.0, 1.0 - self.tau - self.rho)


def stack_transmittance(elements: Sequence[OpticalElement]) -> float:
    """Π τ_i over the whole train."""
    out = 1.0
    for e in elements:
        out *= e.tau
    return out


def stack_self_radiance(
    elements: Sequence[OpticalElement], lb_of_t: Callable[[float], float]
) -> float:
    """L_self = Σ_i ε_i L_B(T_i) Π_{j>i} τ_j, elements ordered scene → detector (§8.2).

    ``lb_of_t`` maps a temperature to band radiance (LUT lookup or closed form). Multiply by
    A_d Ω_eff for the power on a pixel; with one element of τ = τ_opt, ρ = 0 this reduces exactly to
    :func:`self_emission_power` / (A_d Ω_eff).
    """
    total = 0.0
    for i, e in enumerate(elements):
        downstream = 1.0
        for later in elements[i + 1 :]:
            downstream *= later.tau
        total += e.emissivity * lb_of_t(e.temperature_k) * downstream
    return total
