"""Environment presets (M7.11): the airglow unit conversion, §5.3 range guards, the regime enum
round trip, weather-key refusal, and the three committed presets."""

from __future__ import annotations

import copy

import pytest
import yaml

from irsim.config.environment import (
    AIRGLOW_RANGE_W_M2,
    ENVIRONMENT_DIR,
    EnvironmentConfig,
    available_environments,
    load_environment_preset,
)


def _raw() -> dict:  # type: ignore[type-arg]
    return yaml.safe_load((ENVIRONMENT_DIR / "clear_dry.yaml").read_text())


def test_presets_load_and_share_band_keys() -> None:
    # sea_clear_day (MM.5) differs from clear_dry only below the horizon: ground.mode 'sea';
    # scattered_cumulus (AT.11) differs from it only in the clouds block, which authors an
    # optical depth rather than a transmittance.
    assert available_environments() == (
        "clear_dry",
        "humid",
        "overcast",
        "scattered_cumulus",
        "sea_clear_day",
    )
    presets = [load_environment_preset(n) for n in available_environments()]
    keys = {frozenset(p.sky.delta_t_clear_k) for p in presets}
    assert keys == {frozenset({"lwir", "mwir", "swir", "nir"})}
    by_name = {p.name: p for p in presets}
    assert (
        by_name["clear_dry"].sky.delta_t_clear_k["lwir"]
        > by_name["humid"].sky.delta_t_clear_k["lwir"]
    )
    assert (
        by_name["humid"].sky.delta_t_clear_k["lwir"]
        > by_name["overcast"].sky.delta_t_clear_k["lwir"]
    )
    assert (
        not by_name["overcast"].solar.enabled
        and by_name["clear_dry"].solar.glint_model == "specular"
    )


def test_airglow_unit_key_converts_exactly() -> None:
    spec = load_environment_preset("clear_dry")
    assert spec.night.airglow_w_m2 == pytest.approx(1.0e-4, rel=1e-12)
    raw = _raw()
    raw["environment"]["night"] = {"airglow_irradiance_w_m2": 2.5e-4, "k_cloud": 0.4}
    assert EnvironmentConfig.model_validate(raw).environment.night.airglow_w_m2 == 2.5e-4
    raw["environment"]["night"] = {"airglow_irradiance_w_m2": 1e-4, "airglow_irradiance_nw_cm2": 10}
    with pytest.raises(ValueError, match="exactly one"):
        EnvironmentConfig.model_validate(raw)
    raw["environment"]["night"] = {"airglow_irradiance_nw_cm2": 100.0}
    with pytest.raises(ValueError, match="§5.5 range"):
        EnvironmentConfig.model_validate(raw)
    assert pytest.approx((3.5e-5, 3.9e-4), rel=1e-12) == AIRGLOW_RANGE_W_M2


def test_spec_ranges_by_regime_and_q() -> None:
    raw = _raw()
    raw["environment"]["sky"]["delta_t_clear_k"]["lwir"] = 80.0
    with pytest.raises(ValueError, match="clear sky"):
        EnvironmentConfig.model_validate(raw)
    raw = _raw()
    raw["environment"]["sky"]["delta_t_clear_k"]["lwir"] = 40.0
    with pytest.raises(ValueError, match="clear sky"):
        EnvironmentConfig.model_validate(raw)
    raw["environment"]["regime"] = "humid"
    assert EnvironmentConfig.model_validate(raw).environment.regime == "humid"
    raw = _raw()
    raw["environment"]["sky"]["q"] = 5.0
    with pytest.raises(ValueError):
        EnvironmentConfig.model_validate(raw)
    raw = _raw()
    del raw["environment"]["sky"]["delta_t_clear_k"]["lwir"]
    with pytest.raises(ValueError, match="lwir"):
        EnvironmentConfig.model_validate(raw)


def test_regime_enum_round_trips_and_ground_mode() -> None:
    for regime, dt in (("clear", 60.0), ("humid", 30.0), ("overcast", 6.0)):
        raw = _raw()
        raw["environment"]["regime"] = regime
        raw["environment"]["sky"]["delta_t_clear_k"]["lwir"] = dt
        spec = EnvironmentConfig.model_validate(raw).environment
        dumped = yaml.safe_load(
            yaml.safe_dump({"schema_version": 2, "environment": spec.model_dump(mode="json")})
        )
        assert EnvironmentConfig.model_validate(dumped).environment == spec
    raw = _raw()
    raw["environment"]["ground"] = {"mode": "fixed"}
    with pytest.raises(ValueError, match="fixed_temperature_k"):
        EnvironmentConfig.model_validate(raw)
    raw["environment"]["ground"] = {"mode": "fixed", "fixed_temperature_k": 285.0}
    assert EnvironmentConfig.model_validate(raw).environment.ground.fixed_temperature_k == 285.0
    raw["environment"]["ground"] = {"mode": "air", "fixed_temperature_k": 285.0}
    with pytest.raises(ValueError, match="only applies"):
        EnvironmentConfig.model_validate(raw)


def test_weather_like_keys_refused() -> None:
    raw = _raw()
    raw["environment"]["cloud_fraction"] = 0.3
    with pytest.raises(ValueError, match="WeatherSeries"):
        EnvironmentConfig.model_validate(raw)
    raw = copy.deepcopy(_raw())
    raw["environment"]["sky"]["air_temperature_k"] = 290.0
    with pytest.raises(ValueError, match="WeatherSeries"):
        EnvironmentConfig.model_validate(raw)
    with pytest.raises(FileNotFoundError, match="available"):
        load_environment_preset("mars")
