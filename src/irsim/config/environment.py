"""Environment / illumination presets: sky depression, ground mode, solar, night terms (§5.3–§5.5).

    schema_version: 1
    environment:
      name: clear_dry
      regime: clear | humid | overcast
      sky:    {delta_t_clear_k: {lwir: 62, mwir: 15, swir: 0, nir: 0}, q: 0.75}
      ground: {mode: air | solver | fixed, fixed_temperature_k: null}
      solar:  {enabled: true, glint_model: specular | lambertian}
      night:  {airglow_irradiance_nw_cm2: 10   # or airglow_irradiance_w_m2 -- exactly one
               airglow_shape_file: null, k_cloud: 0.5, moon: {enabled: true, phase_fraction: 0.5}}

A preset is an *illumination model* setting, not weather: the cloud fraction, humidity and
temperature that modulate these terms come from the one ``WeatherSeries`` (CLAUDE.md #6), so
weather-like keys are refused here exactly as in the atmosphere presets. Ranges follow §5.3
(ΔT_clear 55–70 K at zenith for a clear dry LWIR sky, 5–10 K under thick overcast, q 0.5–1.0)
and §5.5 (airglow 3.5–39 nW cm⁻², default ~10). The airglow level carries its unit in the key
and is exposed in SI (10 nW cm⁻² = 1.0e-4 W m⁻²).

docs/physics-model.md §5.3(a), §5.4, §5.5, §12.2; ADR 0017 (schema-gap policy)
"""

from __future__ import annotations

import os
import pathlib
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from irsim.config.atmosphere import _find_weather_keys

__all__ = [
    "ENVIRONMENT_SCHEMA_VERSION",
    "ENVIRONMENT_DIR",
    "Regime",
    "DELTA_T_LWIR_RANGE_K",
    "AIRGLOW_RANGE_W_M2",
    "NW_CM2_TO_W_M2",
    "SkySpec",
    "GroundSpec",
    "SolarSpec",
    "MoonSpec",
    "NightSpec",
    "CloudSpec",
    "EnvironmentSpec",
    "EnvironmentConfig",
    "load_environment_preset",
    "available_environments",
]

# v2: optional clouds block (MS.3). `clouds.optical_depth` (ADR 0126) is additive within v2:
# a preset that does not author it keeps the `tau` model bit for bit, so nothing to bump.
ENVIRONMENT_SCHEMA_VERSION = 2
ENVIRONMENT_DIR = pathlib.Path(__file__).resolve().parents[3] / "configs" / "environments"
Regime = Literal["clear", "humid", "overcast"]
# §5.3: zenith clear-sky depression in the LWIR window by regime (kelvin)
DELTA_T_LWIR_RANGE_K: dict[str, tuple[float, float]] = {
    "clear": (55.0, 70.0),
    "humid": (15.0, 55.0),
    "overcast": (0.0, 15.0),
}
# §5.5: reported airglow irradiance 3.5-39 nW cm^-2
NW_CM2_TO_W_M2 = 1e-9 / 1e-4  # 1 nW cm^-2 = 1e-5 W m^-2
AIRGLOW_RANGE_W_M2 = (3.5 * NW_CM2_TO_W_M2, 39.0 * NW_CM2_TO_W_M2)


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class SkySpec(_Frozen):
    """T_sky(θ) = T_air − ΔT_clear (1 − cloud) cos^q θ_zen  (§5.3 a), ΔT per band."""

    delta_t_clear_k: dict[str, float]
    q: float = Field(ge=0.5, le=1.0)

    @field_validator("delta_t_clear_k")
    @classmethod
    def _bands(cls, v: dict[str, float]) -> dict[str, float]:
        if "lwir" not in v:
            raise ValueError("delta_t_clear_k must include the lwir band (§5.3 values are LWIR)")
        for band, dt in v.items():
            if not 0.0 <= dt <= 80.0:
                raise ValueError(f"delta_t_clear_k[{band!r}] = {dt} outside [0, 80] K")
        return v


class GroundSpec(_Frozen):
    """What lies below the horizon.

    ``air``, ``fixed`` and ``solver`` are all **one temperature for the whole ground**, which is
    all a sky-target scene ever claimed. ``sea`` is different in kind: a sea surface has no single
    apparent temperature, because water's emissivity runs from 0.99 looking straight down to under
    0.15 at the horizon and the rest of the signal is reflected sky (ADR 0078). It therefore names
    a *model* rather than a value -- :class:`irsim.atmosphere.sea.SeaModel` -- and the only scalar
    it carries is the bulk sea surface temperature that model starts from.
    """

    mode: Literal["air", "solver", "fixed", "sea"]
    fixed_temperature_k: float | None = Field(default=None, gt=150.0, lt=400.0)
    #: Bulk SST for ``mode: sea``. A measured SST is what a maritime scenario actually has; the
    #: skin temperature is derived from it (MM.4), never authored beside it.
    bulk_sst_k: float | None = Field(default=None, gt=250.0, lt=320.0)

    @model_validator(mode="after")
    def _mode_needs_its_value(self) -> GroundSpec:
        if self.mode == "fixed" and self.fixed_temperature_k is None:
            raise ValueError("ground.mode 'fixed' needs fixed_temperature_k")
        if self.mode != "fixed" and self.fixed_temperature_k is not None:
            raise ValueError("fixed_temperature_k only applies to ground.mode 'fixed'")
        if self.mode == "sea" and self.bulk_sst_k is None:
            raise ValueError("ground.mode 'sea' needs bulk_sst_k (the measured bulk SST)")
        if self.mode != "sea" and self.bulk_sst_k is not None:
            raise ValueError("bulk_sst_k only applies to ground.mode 'sea'")
        return self


class SolarSpec(_Frozen):
    enabled: bool = True
    glint_model: Literal["specular", "lambertian"] = "specular"  # §5.4: specular lobe


class MoonSpec(_Frozen):
    enabled: bool = True
    phase_fraction: float = Field(default=0.5, ge=0.0, le=1.0)  # 0 new, 1 full


class NightSpec(_Frozen):
    airglow_irradiance_w_m2: float | None = None
    airglow_irradiance_nw_cm2: float | None = None
    airglow_shape_file: str | None = None  # spectra/airglow_*.csv (M11); None = flat
    k_cloud: float = Field(default=0.5, ge=0.0, le=1.0)
    moon: MoonSpec = MoonSpec()

    @model_validator(mode="after")
    def _one_airglow_key(self) -> NightSpec:
        given = [
            k
            for k in ("airglow_irradiance_w_m2", "airglow_irradiance_nw_cm2")
            if getattr(self, k) is not None
        ]
        if len(given) != 1:
            raise ValueError(
                "give exactly one of airglow_irradiance_w_m2 / airglow_irradiance_nw_cm2"
            )
        lo, hi = AIRGLOW_RANGE_W_M2
        e = self.airglow_w_m2
        if not lo <= e <= hi:
            raise ValueError(
                f"airglow irradiance {e:.3e} W/m^2 outside the §5.5 range "
                f"[{lo:.2e}, {hi:.2e}] W/m^2 (3.5-39 nW/cm^2)"
            )
        return self

    @property
    def airglow_w_m2(self) -> float:
        if self.airglow_irradiance_w_m2 is not None:
            return float(self.airglow_irradiance_w_m2)
        assert self.airglow_irradiance_nw_cm2 is not None
        return float(self.airglow_irradiance_nw_cm2) * NW_CM2_TO_W_M2


class CloudSpec(_Frozen):
    """Cloud clutter (MS.3, ADR 0070): the 1/f^β spectral slope of the spatial structure, bounds
    on the LCL base height, and the cloud's opacity **one way or the other**. The cloud fraction
    itself is weather (WeatherSeries.cloud_fraction), never authored here.

    Two ways to say how opaque a cloud is, and a preset picks exactly one (ADR 0126):

    * ``optical_depth`` -- the cloud's **visible** optical depth at full depth. What a scene
      should author. The LWIR emissivity is derived from it by
      :func:`~irsim.atmosphere.cloud.cloud_emissivity` and varies along the ray, so a cloud thins
      toward its edges and low cloud is more opaque than the same cloud overhead. Fair-weather
      cumulus run roughly 5-20; stratocumulus higher; thin cirrus below 1. Anything past
      :data:`~irsim.atmosphere.cloud.OPAQUE_OPTICAL_DEPTH` is a blackbody and indistinguishable.
    * ``tau`` -- the original: one transmittance for the whole cloud, ε = 1 − τ. Every covered
      pixel then carries the same radiance, which renders as flat blobs. Kept so that every scene
      authored before ADR 0126 is bit-identical, not because it is the better model.

    Authoring both is refused rather than silently resolved: they are two answers to one question.
    """

    tau: float = Field(default=0.0, ge=0.0, le=0.99)
    #: Visible optical depth at full cloud depth. ``None`` selects the ``tau`` model above.
    optical_depth: float | None = Field(default=None, gt=0.0, le=200.0)
    beta: float = Field(default=1.8, ge=0.5, le=4.0)
    min_base_m: float = Field(default=0.0, ge=0.0)  # 0: saturated air puts the base at the surface
    max_base_m: float = Field(default=8000.0, gt=0.0)

    @model_validator(mode="after")
    def _one_opacity(self) -> CloudSpec:
        if self.optical_depth is not None and "tau" in self.model_fields_set:
            raise ValueError(
                "author clouds.optical_depth (visible OD, ADR 0126) or clouds.tau (one "
                "transmittance, ADR 0070), not both -- they are two answers to one question"
            )
        return self

    @property
    def emissivity(self) -> float:
        """ε = 1 − τ, the ``tau`` model's one emissivity.

        **Raises** when ``optical_depth`` is authored, rather than returning 1 − 0 = 1 and letting
        a caller render an opaque cloud from a preset that asked for a thin one. The emissivity is
        then not a property of the preset at all: it depends on the ray, so it belongs to
        :func:`~irsim.atmosphere.cloud.cloud_emissivity` and the caller must go there with an
        airmass. (Same shape as ``Scene.atmosphere`` after AT.5, and for the same reason.)
        """
        if self.optical_depth is not None:
            raise AttributeError(
                "this preset authors clouds.optical_depth, so the cloud has no single emissivity "
                "-- it depends on the slant path. Use irsim.atmosphere.cloud.cloud_emissivity("
                "optical_depth, path_factor) with cloud_airmass(elevation) for a ray, or the "
                "DIFFUSIVITY_FACTOR default for a flux."
            )
        return 1.0 - self.tau


class EnvironmentSpec(_Frozen):
    name: str = Field(min_length=1)
    description: str = ""
    regime: Regime
    sky: SkySpec
    ground: GroundSpec
    solar: SolarSpec = SolarSpec()
    night: NightSpec
    clouds: CloudSpec = CloudSpec()

    @model_validator(mode="after")
    def _delta_t_matches_regime(self) -> EnvironmentSpec:
        lo, hi = DELTA_T_LWIR_RANGE_K[self.regime]
        dt = self.sky.delta_t_clear_k["lwir"]
        if not lo <= dt <= hi:
            raise ValueError(
                f"sky.delta_t_clear_k.lwir = {dt} K outside the §5.3 range [{lo}, {hi}] K for a "
                f"{self.regime} sky"
            )
        return self


class EnvironmentConfig(_Frozen):
    schema_version: int
    environment: EnvironmentSpec

    @model_validator(mode="before")
    @classmethod
    def _no_weather(cls, data: Any) -> Any:
        if isinstance(data, dict):
            hits = _find_weather_keys(data)
            if hits:
                raise ValueError(
                    f"weather-like keys {hits} are not allowed in an environment preset: cloud, "
                    "humidity and temperature come from the WeatherSeries (CLAUDE.md #6)"
                )
        return data

    @field_validator("schema_version")
    @classmethod
    def _version(cls, v: int) -> int:
        if v != ENVIRONMENT_SCHEMA_VERSION:
            raise ValueError(f"environment schema_version {v} != {ENVIRONMENT_SCHEMA_VERSION}")
        return v


def available_environments(preset_dir: str | os.PathLike[str] | None = None) -> tuple[str, ...]:
    root = pathlib.Path(preset_dir) if preset_dir is not None else ENVIRONMENT_DIR
    return tuple(sorted(p.stem for p in root.glob("*.yaml")))


def load_environment_preset(
    name_or_path: str | os.PathLike[str], preset_dir: str | os.PathLike[str] | None = None
) -> EnvironmentSpec:
    root = pathlib.Path(preset_dir) if preset_dir is not None else ENVIRONMENT_DIR
    path = pathlib.Path(name_or_path)
    if path.suffix != ".yaml":
        path = root / f"{name_or_path}.yaml"
    if not path.is_file():
        raise FileNotFoundError(
            f"environment preset {name_or_path!r} not found; "
            f"available: {available_environments(root)}"
        )
    spec = EnvironmentConfig.model_validate(
        yaml.safe_load(path.read_text(encoding="utf-8"))
    ).environment
    if path.parent == root and spec.name != path.stem:
        raise ValueError(f"environment name {spec.name!r} does not match file name {path.stem!r}")
    return spec
