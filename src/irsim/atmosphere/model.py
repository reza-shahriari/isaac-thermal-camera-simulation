"""Atmosphere: a preset bound to the one shared WeatherSeries (§7.1–§7.3, CLAUDE.md #6).

    state(t) = { T_air(t), w(t), V(t), γ_B(t) per band, L_air,B(t) = L_B(T_air(t)) per band }

The constructor takes the ``WeatherSeries`` *object* -- never a path -- so the T_air used for
path radiance is, by identity, the T_air the thermal solver uses for convection; the Scene
(M6.17) asserts ``atmosphere.weather is solver.weather``. Extinction is
γ_B = γ₀,B + β_B w(T_air, RH) + r_B γ_aer,vis(V) (M8.3/M8.4), so the humid and fog crossovers
between bands follow from the weather alone with one preset. Per-pixel work stays in
:mod:`irsim.atmosphere.beer_lambert`; this class supplies the scalars the stage needs.

docs/physics-model.md §7.1, §7.2, §7.3, §13.4 stage 2
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from irsim.atmosphere.beer_lambert import apply_atmosphere, transmittance
from irsim.atmosphere.extinction import (
    band_extinction,
    regime_for_visibility,
)
from irsim.atmosphere.humidity import gamma_molecular
from irsim.config.atmosphere import AtmospherePreset
from irsim.config.bands import ANCHOR_BAND
from irsim.radiometry.lut import BandLUT, Quantity
from irsim.thermal.weather import WeatherSeries

__all__ = ["AtmosphereState", "Atmosphere"]


@dataclass(frozen=True)
class AtmosphereState:
    """The scalar atmosphere at one time: what stage 2 needs besides the distance plane."""

    t_s: float
    t_air_k: float
    w_g_m3: float
    visibility_m: float
    gamma_per_m: Mapping[str, float]  # every band of the preset
    l_air: Mapping[str, float]  # energy-form band radiance of the air, bands with a LUT
    regime_mismatch: (
        bool  # weather implies fog (V < 1 km) but the preset is not droplet, or vice versa
    )


class Atmosphere:
    """A preset + the shared weather (+ band LUTs for the air radiance)."""

    def __init__(
        self,
        preset: AtmospherePreset,
        weather: WeatherSeries,
        luts: Mapping[str, BandLUT] | None = None,
    ) -> None:
        if not isinstance(weather, WeatherSeries):
            raise TypeError(
                "Atmosphere takes the WeatherSeries object shared with the thermal solver, "
                f"not {type(weather).__name__} (CLAUDE.md #6: never a path)"
            )
        if not isinstance(preset, AtmospherePreset):
            raise TypeError("preset must be an AtmospherePreset (irsim.atmosphere.library)")
        luts = dict(luts or {})
        unknown = luts.keys() - preset.bands.keys()
        if unknown:
            raise ValueError(f"LUTs for bands {sorted(unknown)} the preset does not define")
        self._preset = preset
        self._weather = weather
        self._luts = luts

    @property
    def preset(self) -> AtmospherePreset:
        return self._preset

    @property
    def weather(self) -> WeatherSeries:
        return self._weather

    @property
    def bands(self) -> tuple[str, ...]:
        return tuple(sorted(self._preset.bands))

    @property
    def lut_bands(self) -> tuple[str, ...]:
        return tuple(sorted(self._luts))

    # -- scalars ---------------------------------------------------------------------------
    def state(self, t_s: float) -> AtmosphereState:
        sample = self._weather.at(t_s)
        w = sample.absolute_humidity_g_m3
        vis = self._preset.bands[ANCHOR_BAND]
        gamma_mol_vis = gamma_molecular(
            w, vis.gamma0_per_m, vis.beta_per_m_per_g_m3, vis.beta2_per_m_per_g2_m6
        )
        gammas = {
            band: band_extinction(coeffs, w, sample.visibility_m, gamma_mol_vis)
            for band, coeffs in self._preset.bands.items()
        }
        l_air = {
            band: float(lut.lookup(np.float64(sample.t_air_k), "lb")[()])
            for band, lut in self._luts.items()
        }
        implied = regime_for_visibility(sample.visibility_m)
        mismatch = (implied == "droplet") != (self._preset.aerosol_regime == "droplet")
        return AtmosphereState(
            t_s=float(t_s),
            t_air_k=sample.t_air_k,
            w_g_m3=w,
            visibility_m=sample.visibility_m,
            gamma_per_m=gammas,
            l_air=l_air,
            regime_mismatch=mismatch,
        )

    def gamma(self, band: str, t_s: float) -> float:
        return self.state(t_s).gamma_per_m[band]

    def transmittance(self, band: str, t_s: float, distance_m: Any) -> NDArray[np.floating]:
        """τ_B(d) at time t, per pixel or scalar (d = ∞ → 0)."""
        return transmittance(distance_m, self.gamma(band, t_s))

    def air_radiance(self, band: str, t_s: float, quantity: Quantity = "lb") -> float:
        """L_B(T_air(t)) in the requested LUT quantity (energy or photon form)."""
        if band not in self._luts:
            raise KeyError(f"no LUT for band {band!r}; Atmosphere has {self.lut_bands}")
        t_air = np.float64(self._weather.at(t_s).t_air_k)
        return float(self._luts[band].lookup(t_air, quantity)[()])

    def apply(
        self,
        band: str,
        t_s: float,
        l_band: Any,
        distance_m: Any,
        quantity: Quantity = "lb",
    ) -> NDArray[np.floating]:
        """L' = τ L + (1 − τ) L_B(T_air): the per-pixel stage-2 kernel at time t."""
        return apply_atmosphere(
            l_band, distance_m, self.gamma(band, t_s), self.air_radiance(band, t_s, quantity)
        )
