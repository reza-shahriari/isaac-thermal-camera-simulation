"""Band extinction from weather: molecular γ(w) plus Koschmieder aerosol scaled per band (§7.1–7.3).

    γ_aer,vis(V) = max(0, KOSCHMIEDER / V − γ_mol,vis)        KOSCHMIEDER = ln(1/0.02) = 3.912
    γ_B(w, V)    = γ₀,B + β_B w + r_B · γ_aer,vis(V)

Meteorological optical range V is *defined* on the total visible extinction (2 % contrast
threshold, WMO), so the aerosol part is the Koschmieder total minus the molecular (Rayleigh)
part of the visible band; V → ∞ recovers clear air and V above the Rayleigh limit (~330 km at
sea level) gives zero aerosol. r_B is the preset's per-band extinction ratio to the visible:
below 1 and falling with wavelength for small-particle haze, flatter for fog droplets
(ADR 0049). Everything here is scalar per band; the per-pixel work is Beer–Lambert on the
distance plane (irsim.atmosphere.beer_lambert).

docs/physics-model.md §7.1, §7.2, §7.3
"""

from __future__ import annotations

import math

from irsim.atmosphere.humidity import absolute_humidity_g_m3, gamma_molecular
from irsim.config.atmosphere import AtmosphereBandCoefficients, AtmospherePreset
from irsim.config.bands import ANCHOR_BAND
from irsim.radiometry.constants import KOSCHMIEDER

__all__ = [
    "airmass_kasten_young",
    "FOG_VISIBILITY_M",
    "gamma_aerosol_visible",
    "gamma_aerosol",
    "band_extinction",
    "extinction_per_band",
    "transmittance_per_band",
    "regime_for_visibility",
    "MAX_SOLAR_ZENITH_DEG",
    "airmass",
    "solar_transmittance",
]

# WMO: fog is visibility below 1 km (mist 1-5 km). Below this the droplet ratios apply.
FOG_VISIBILITY_M = 1000.0
# Plane-parallel airmass sec(theta) is within ~2 % of the refracted value to 80 deg and diverges
# at the horizon; beyond this the caller needs the spherical form (deferred to M11, ADR 0051).
MAX_SOLAR_ZENITH_DEG = 85.0


def gamma_aerosol_visible(visibility_m: float, gamma_mol_visible_per_m: float = 0.0) -> float:
    """Koschmieder: aerosol extinction in the visible, m⁻¹. V = ∞ → 0; V ≤ 0 raises."""
    if not visibility_m > 0.0:
        raise ValueError(f"visibility_m must be positive, got {visibility_m}")
    if gamma_mol_visible_per_m < 0.0:
        raise ValueError("gamma_mol_visible_per_m must be non-negative")
    if math.isinf(visibility_m):
        return 0.0
    return max(0.0, KOSCHMIEDER / visibility_m - gamma_mol_visible_per_m)


def gamma_aerosol(
    visibility_m: float, ratio_to_visible: float, gamma_mol_visible_per_m: float = 0.0
) -> float:
    """r_B · γ_aer,vis(V) (m⁻¹)."""
    if ratio_to_visible < 0.0:
        raise ValueError("ratio_to_visible must be non-negative")
    return ratio_to_visible * gamma_aerosol_visible(visibility_m, gamma_mol_visible_per_m)


def band_extinction(
    coeffs: AtmosphereBandCoefficients,
    w_g_m3: float,
    visibility_m: float,
    gamma_mol_visible_per_m: float,
) -> float:
    """γ_B = γ_mol,B(w) + r_B γ_aer,vis(V) for one band (m⁻¹)."""
    mol = gamma_molecular(
        w_g_m3, coeffs.gamma0_per_m, coeffs.beta_per_m_per_g_m3, coeffs.beta2_per_m_per_g2_m6
    )
    return mol + gamma_aerosol(
        visibility_m, coeffs.aerosol_ratio_to_visible, gamma_mol_visible_per_m
    )


def extinction_per_band(
    preset: AtmospherePreset, t_air_k: float, rh_fraction: float, visibility_m: float
) -> dict[str, float]:
    """γ_B (m⁻¹) for every band of the preset at the given weather."""
    w = absolute_humidity_g_m3(t_air_k, rh_fraction)
    vis = preset.bands[ANCHOR_BAND]
    gamma_mol_vis = gamma_molecular(
        w, vis.gamma0_per_m, vis.beta_per_m_per_g_m3, vis.beta2_per_m_per_g2_m6
    )
    return {
        band: band_extinction(coeffs, w, visibility_m, gamma_mol_vis)
        for band, coeffs in preset.bands.items()
    }


def transmittance_per_band(
    preset: AtmospherePreset,
    t_air_k: float,
    rh_fraction: float,
    visibility_m: float,
    distance_m: float,
) -> dict[str, float]:
    """τ_B(d) = exp(−γ_B d) per band -- the §7.2 table quantity at d = 200 m."""
    if distance_m < 0.0:
        raise ValueError("distance_m must be non-negative")
    gammas = extinction_per_band(preset, t_air_k, rh_fraction, visibility_m)
    return {band: math.exp(-g * distance_m) for band, g in gammas.items()}


def regime_for_visibility(visibility_m: float) -> str:
    """'droplet' below the WMO fog threshold (1 km), else 'aerosol': what the weather implies.
    A preset whose regime disagrees is being used outside the conditions it was fitted for."""
    if not visibility_m > 0.0:
        raise ValueError("visibility_m must be positive")
    return "droplet" if visibility_m < FOG_VISIBILITY_M else "aerosol"


def airmass(zenith_rad: float) -> float:
    """Plane-parallel airmass sec θ_zen (1 at zenith, 2 at 60°); refused beyond 85° (ADR 0051)."""
    deg = math.degrees(zenith_rad)
    if not -1e-9 <= deg <= MAX_SOLAR_ZENITH_DEG + 1e-9:
        raise ValueError(
            f"solar zenith angle {deg:.1f}° outside [0, {MAX_SOLAR_ZENITH_DEG}°]: the "
            "plane-parallel airmass diverges at the horizon (ADR 0051)"
        )
    return 1.0 / math.cos(zenith_rad)


def airmass_kasten_young(zenith_rad: float) -> float:
    """Kasten & Young (1989) relative air mass, finite all the way to the horizon.

    ``airmass`` above is the plane-parallel sec θ ADR 0051 chose for the band-averaged path, and
    it diverges at the horizon -- which is why it refuses beyond 85°. The *solar beam* needs an
    answer at low sun (M11.3): a scene at 09:00 is not an error condition, and a camera pointed
    at a sunlit target does not stop working because the sun is 8° up. This is the standard
    empirical fit,

        m = 1 / (cos θ + 0.50572 (96.07995 − θ_deg)^-1.6364)

    which reproduces the refracted air mass to better than 1 % from zenith to the horizon
    (m = 37.9 at θ = 90°) instead of going to infinity. It is used for the direct beam only; the
    band-averaged slant path keeps sec θ, so nothing that was measured against ADR 0051 moves.
    """
    deg = math.degrees(zenith_rad)
    if not -1e-9 <= deg <= 90.0 + 1e-9:
        raise ValueError(f"solar zenith angle {deg:.1f}° outside [0, 90°]")
    deg = min(max(deg, 0.0), 90.0)
    return float(1.0 / (math.cos(math.radians(deg)) + 0.50572 * (96.07995 - deg) ** -1.6364))


def solar_transmittance(preset: AtmospherePreset, band: str, zenith_rad: float) -> float:
    """τ_sun(θ_zen) = τ_zenith^{1/cos θ} for one band (§5.4 stub; ADR 0051).

    Bouguer's law for a plane-parallel atmosphere: the slant column is sec θ times the vertical
    one, so the transmittance is the zenith value raised to the airmass. τ_sun(60°) = τ_zenith²
    exactly.
    """
    if band not in preset.solar.zenith_transmittance:
        raise KeyError(f"preset has no solar transmittance for band {band!r}")
    return float(preset.solar.zenith_transmittance[band] ** airmass(zenith_rad))
