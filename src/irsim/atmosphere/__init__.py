"""Atmosphere: Beer–Lambert transmittance, path radiance, humidity, presets, the stage.

docs/physics-model.md §7
"""

from irsim.atmosphere.beer_lambert import (
    apply_atmosphere,
    apply_tau_override,
    path_radiance,
    transmittance,
)
from irsim.atmosphere.cloud import (
    CloudField,
    cloud_base_temperature_k,
    generate_cloud_field,
    lifting_condensation_level_m,
    psd_slope,
)
from irsim.atmosphere.extinction import (
    extinction_per_band,
    gamma_aerosol,
    gamma_aerosol_visible,
    regime_for_visibility,
    transmittance_per_band,
)
from irsim.atmosphere.humidity import (
    absolute_humidity_g_m3,
    gamma_molecular,
    saturation_vapour_pressure_hpa,
    vapour_pressure_hpa,
)
from irsim.atmosphere.layered import (
    ExponentialSum,
    LayeredAtmosphere,
    class_weights,
    exponential_sum_from_piecewise,
    fit_exponential_sum,
)
from irsim.atmosphere.library import available_presets, load_atmosphere_preset, preset_hash
from irsim.atmosphere.model import Atmosphere, AtmosphereState
from irsim.atmosphere.sea import (
    COX_MUNK,
    SeaModel,
    horizon_depression_rad,
    slant_range_m,
    slope_variance,
)
from irsim.atmosphere.sea_envelope import (
    VALIDATED_WIND_M_S,
    VALIDATED_ZENITH_DEG,
    EnvelopeReport,
    beyond_envelope,
    depression_at_zenith_rad,
    envelope_report,
    view_zenith_rad,
)
from irsim.atmosphere.sky import CosQFit, SkyModel
from irsim.atmosphere.spectral import (
    band_transmittance_spectral,
    effective_gamma,
    fit_grey_gamma,
    grey_fit_error,
)

__all__ = [
    "Atmosphere",
    "AtmosphereState",
    "ExponentialSum",
    "LayeredAtmosphere",
    "class_weights",
    "exponential_sum_from_piecewise",
    "fit_exponential_sum",
    "apply_atmosphere",
    "apply_tau_override",
    "path_radiance",
    "transmittance",
    "absolute_humidity_g_m3",
    "gamma_molecular",
    "saturation_vapour_pressure_hpa",
    "vapour_pressure_hpa",
    "CloudField",
    "cloud_base_temperature_k",
    "generate_cloud_field",
    "lifting_condensation_level_m",
    "psd_slope",
    "extinction_per_band",
    "gamma_aerosol",
    "gamma_aerosol_visible",
    "regime_for_visibility",
    "transmittance_per_band",
    "available_presets",
    "COX_MUNK",
    "SeaModel",
    "EnvelopeReport",
    "VALIDATED_WIND_M_S",
    "VALIDATED_ZENITH_DEG",
    "beyond_envelope",
    "depression_at_zenith_rad",
    "envelope_report",
    "view_zenith_rad",
    "horizon_depression_rad",
    "slant_range_m",
    "slope_variance",
    "load_atmosphere_preset",
    "preset_hash",
    "CosQFit",
    "SkyModel",
    "band_transmittance_spectral",
    "effective_gamma",
    "fit_grey_gamma",
    "grey_fit_error",
]
