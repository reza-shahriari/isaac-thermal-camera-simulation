"""FpaParams (M3.5): the Boson known answers, discriminated validation, and the schema-v3
detector constants."""

from __future__ import annotations

import copy
import pathlib
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from irsim.config.sensor import SCHEMA_VERSION, SensorConfig
from irsim.detector import BolometerParams, PhotonParams, fpa_params_from_config

REPO = pathlib.Path(__file__).resolve().parents[2]
BOSON = yaml.safe_load((REPO / "configs" / "sensors" / "flir_boson_640_lwir.yaml").read_text())

PHOTON_FPA: dict[str, Any] = {
    "type": "photon",
    "width": 640,
    "height": 512,
    "pitch_um": 15.0,
    "fill_factor": 1.0,
    "frame_rate_hz": 30,
    "bit_depth": 14,
    "quantum_efficiency": 0.8,
    "well_capacity_e": 1.0e6,
    "integration_time_ms": 5.0,
    "dark_current_model": "arrhenius",
}


def _photon_cfg(**fpa_overrides: Any) -> dict[str, Any]:
    d = copy.deepcopy(BOSON)
    d["sensor"]["band"].update(lambda_min_um=3.0, lambda_max_um=5.0, regime="mixed")
    d["sensor"]["fpa"] = {**PHOTON_FPA, **fpa_overrides}
    return d


def test_boson_known_answers() -> None:
    p = fpa_params_from_config(SensorConfig.model_validate(BOSON))
    assert isinstance(p, BolometerParams) and p.type == "bolometer"
    assert p.active_area_m2 == pytest.approx(1.296e-10, rel=1e-12)
    assert p.frame_dt_s == pytest.approx(16.6667e-3, rel=1e-5)
    assert p.dn_max == 65535 and p.shape == (512, 640)
    assert p.thermal_time_constant_s == pytest.approx(8e-3)  # [R24]: nominally 8 msec (SC.3)
    # SC.34: the Boson authors VOx's G_th (2e-8 W/K, ESTIMATED); C_th follows as G_th * tau
    assert p.c_th_j_per_k == pytest.approx(8e-3 * 2e-8)
    assert (p.absorptance, p.g_th_w_per_k, p.bias_current_a, p.resistance_ohm) == (
        0.8,
        2e-8,
        50e-6,
        1e5,
    )
    assert p.fpa_temp_k is None and p.fpa_self_heating_k == 0.0
    assert SCHEMA_VERSION >= 5


def test_boson_yaml_without_photon_null_keys_loads() -> None:
    assert "quantum_efficiency" not in BOSON["sensor"]["fpa"]
    assert isinstance(fpa_params_from_config(SensorConfig.model_validate(BOSON)), BolometerParams)


def test_photon_params_and_dark_current_block() -> None:
    d = _photon_cfg(
        read_noise_e=30.0,
        dark_current={"i_ref_a_per_pixel": 1e-13, "t_ref_k": 80.0, "band_gap_ev": 0.23},
    )
    p = fpa_params_from_config(SensorConfig.model_validate(d))
    assert isinstance(p, PhotonParams) and p.type == "photon"
    assert p.integration_time_s == pytest.approx(5e-3) and p.quantum_efficiency == 0.8
    assert p.active_area_m2 == pytest.approx(2.25e-10, rel=1e-12)
    assert p.read_noise_e == 30.0 and p.dark_current_band_gap_ev == 0.23
    bare = fpa_params_from_config(SensorConfig.model_validate(_photon_cfg()))
    assert (
        isinstance(bare, PhotonParams)
        and bare.read_noise_e is None
        and bare.dark_current_i_ref_a is None
    )


def test_photon_validation_rules() -> None:
    d = _photon_cfg()
    del d["sensor"]["fpa"]["quantum_efficiency"]
    with pytest.raises(ValidationError, match="quantum_efficiency"):
        SensorConfig.model_validate(d)
    with pytest.raises(ValidationError, match="frame period"):
        SensorConfig.model_validate(_photon_cfg(integration_time_ms=40.0))
    with pytest.raises(ValidationError):
        SensorConfig.model_validate(_photon_cfg(read_noise_e=-1.0))
    with pytest.raises(ValidationError):
        SensorConfig.model_validate(
            _photon_cfg(
                dark_current={"i_ref_a_per_pixel": 1e-13, "t_ref_k": 0.0, "band_gap_ev": 0.23}
            )
        )


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("bit_depth", 7),
        ("bit_depth", 17),
        ("fill_factor", 0.0),
        ("fill_factor", 1.1),
        ("absorptance", 1.5),
        ("g_th_w_per_k", 0.0),
        ("fpa_temp_k", -5.0),
    ],
)
def test_bolometer_bounds(key: str, value: float) -> None:
    d = copy.deepcopy(BOSON)
    d["sensor"]["fpa"][key] = value
    with pytest.raises(ValidationError):
        SensorConfig.model_validate(d)


def test_fpa_thermal_node_fields() -> None:
    d = copy.deepcopy(BOSON)
    d["sensor"]["fpa"].update(fpa_temp_k=310.0, fpa_tau_s=600.0, fpa_self_heating_k=5.0)
    p = fpa_params_from_config(SensorConfig.model_validate(d))
    assert (p.fpa_temp_k, p.fpa_tau_s, p.fpa_self_heating_k) == (310.0, 600.0, 5.0)
