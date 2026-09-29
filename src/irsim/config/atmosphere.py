"""Atmosphere preset schema: per-band extinction coefficients, no weather (§7.2, §12.2).

A preset is an *atmosphere type* -- the coefficients of the grey band model (ADR 0048/0049)

    γ_B = γ₀,B + β_B · w  +  r_B · γ_aer,vis(V)

per band, an aerosol regime, a tropospheric profile shape for the slant-path model (MS.1) and
provenance. It deliberately carries **no weather**: T_air, relative humidity and visibility come
from the one ``WeatherSeries`` shared with the thermal solver (CLAUDE.md #6), and any weather-like
key anywhere in the file is rejected by name so a preset cannot smuggle a second climate in.

Band keys are the sensor band ids of :mod:`irsim.config.bands` plus ``visible`` (the Koschmieder
anchor, ratio 1 by definition); every preset must define the same key set so an ``Atmosphere``
can serve any camera in the sensor library.

docs/physics-model.md §7.2, §7.3, §12.2
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from irsim.config.bands import ANCHOR_BAND, BAND_KEYS

__all__ = [
    "SCHEMA_VERSION",
    "ATMOSPHERE_BAND_KEYS",
    "WEATHER_LIKE_KEYS",
    "AerosolRegime",
    "AtmosphereBandCoefficients",
    "AtmosphereProfile",
    "AtmosphereProvenance",
    "SolarPathSpec",
    "AtmospherePreset",
    "AtmosphereConfig",
]

SCHEMA_VERSION = 2  # v2: solar-path stub (M8.8, ADR 0051)
ATMOSPHERE_BAND_KEYS: frozenset[str] = frozenset(BAND_KEYS)
# Names that describe weather, not an atmosphere type. Rejected at every nesting level.
WEATHER_LIKE_KEYS: frozenset[str] = frozenset(
    {
        "air_temperature_k",
        "t_air_k",
        "temperature_k",
        "relative_humidity",
        "rh",
        "rh_fraction",
        "humidity",
        "absolute_humidity_g_m3",
        "visibility_m",
        "visibility_km",
        "wind_speed_m_s",
        "wind_speed",
        "cloud_fraction",
        "precipitation_mm_h",
        "solar_irradiance_w_m2",
        "weather",
        "weather_file",
    }
)
AerosolRegime = Literal["rural", "maritime", "urban", "droplet"]


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class AtmosphereBandCoefficients(_Frozen):
    """γ_B = gamma0 + beta · w + aerosol_ratio_to_visible · γ_aer,vis(V)  (m⁻¹, w in g m⁻³)."""

    gamma0_per_m: float = Field(ge=0.0)
    beta_per_m_per_g_m3: float = Field(ge=0.0)
    #: AT.27 (ADR 0160): the self-continuum square, m⁻¹ per (g m⁻³)². 0 -- the default, and
    #: every preset written before the field existed -- is §7.3's linear law bit for bit.
    beta2_per_m_per_g2_m6: float = Field(default=0.0, ge=0.0)
    aerosol_ratio_to_visible: float = Field(ge=0.0)


class AtmosphereProfile(_Frozen):
    """Shape of the lower troposphere for the slant-path model (MS.1): w(h) = w₀ e^{−h/H},
    T_air(h) = T_air − Γ h. Not used on horizontal paths."""

    water_vapour_scale_height_m: float = Field(gt=0.0)
    lapse_rate_k_per_m: float = Field(ge=0.0, le=0.02)
    aerosol_scale_height_m: float = Field(default=1200.0, gt=0.0)  # boundary-layer aerosol
    air_scale_height_m: float = Field(default=8000.0, gt=0.0)  # well-mixed gases (CO2)
    tropopause_m: float = Field(default=11000.0, gt=0.0)  # isothermal above (USSA 1976)


class AtmosphereProvenance(_Frozen):
    status: Literal["ESTIMATED", "MEASURED", "MODTRAN"]
    source: str
    note: str = ""


class SolarPathSpec(_Frozen):
    """Solar-path transmittance at zenith, per band (the M8.8 stub; ADR 0051).

    The slant solar path uses the plane-parallel Bouguer form τ_sun(θ_zen) = τ_zenith^{1/cos θ},
    i.e. an airmass of sec θ. Only the reflective bands consume it (§5.4, M11); the LWIR entry
    exists so every preset carries the same band-key set.
    """

    zenith_transmittance: dict[str, float]

    @field_validator("zenith_transmittance")
    @classmethod
    def _bands_complete(cls, values: dict[str, float]) -> dict[str, float]:
        missing = ATMOSPHERE_BAND_KEYS - values.keys()
        unknown = values.keys() - ATMOSPHERE_BAND_KEYS
        if missing or unknown:
            raise ValueError(
                f"solar.zenith_transmittance must name exactly {sorted(ATMOSPHERE_BAND_KEYS)}: "
                f"missing {sorted(missing)}, unknown {sorted(unknown)}"
            )
        for band, tau in values.items():
            if not 0.0 < tau <= 1.0:
                raise ValueError(f"solar.zenith_transmittance[{band!r}] = {tau} is not in (0, 1]")
        return values


class AtmospherePreset(_Frozen):
    name: str = Field(min_length=1)
    description: str = ""
    aerosol_regime: AerosolRegime
    valid_range_m: float = Field(gt=0.0, default=500.0)
    bands: dict[str, AtmosphereBandCoefficients]
    profile: AtmosphereProfile
    solar: SolarPathSpec
    provenance: AtmosphereProvenance

    @field_validator("bands")
    @classmethod
    def _bands_complete(
        cls, bands: dict[str, AtmosphereBandCoefficients]
    ) -> dict[str, AtmosphereBandCoefficients]:
        missing = ATMOSPHERE_BAND_KEYS - bands.keys()
        unknown = bands.keys() - ATMOSPHERE_BAND_KEYS
        if missing or unknown:
            raise ValueError(
                f"bands must be exactly {sorted(ATMOSPHERE_BAND_KEYS)}: "
                f"missing {sorted(missing)}, unknown {sorted(unknown)}"
            )
        if bands[ANCHOR_BAND].aerosol_ratio_to_visible != 1.0:
            raise ValueError(
                f"{ANCHOR_BAND}.aerosol_ratio_to_visible must be 1.0 (Koschmieder anchor)"
            )
        return bands


def _find_weather_keys(obj: Any, path: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            here = f"{path}.{key}" if path else str(key)
            if str(key) in WEATHER_LIKE_KEYS:
                found.append(here)
            found.extend(_find_weather_keys(value, here))
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            found.extend(_find_weather_keys(value, f"{path}[{i}]"))
    return found


class AtmosphereConfig(_Frozen):
    """Top level of an atmosphere YAML: ``schema_version`` + ``atmosphere``."""

    schema_version: int
    atmosphere: AtmospherePreset

    @model_validator(mode="before")
    @classmethod
    def _no_weather_anywhere(cls, data: Any) -> Any:
        if isinstance(data, dict):
            hits = _find_weather_keys(data)
            if hits:
                raise ValueError(
                    f"weather-like keys {hits} are not allowed in an atmosphere preset: T_air, "
                    "humidity and visibility come from the one WeatherSeries shared with the "
                    "thermal solver (CLAUDE.md #6)"
                )
        return data

    @field_validator("schema_version")
    @classmethod
    def _version(cls, v: int) -> int:
        if v != SCHEMA_VERSION:
            raise ValueError(f"atmosphere schema_version {v} != {SCHEMA_VERSION}")
        return v
