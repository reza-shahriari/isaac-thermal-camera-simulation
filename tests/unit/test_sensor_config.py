"""§12.2 sensor schema: the Boson reference file loads exactly, physics constraints are
enforced, derived quantities match the spec's worked numbers, and no aperture factor lives here.

The derived-quantity tests are known answers from docs/physics-model.md (§8.3 Nyquist 41.7,
§16.1 32° HFOV, §9.2 τ_th/Δt ≈ 0.6) -- a unit slip in pitch or focal length is a factor 1e3+.
"""

from __future__ import annotations

import ast
import copy
import pathlib
import re
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from irsim.config.sensor import BolometerFpa, PhotonFpa, SensorConfig

REPO = pathlib.Path(__file__).resolve().parents[2]
BOSON_YAML = REPO / "configs" / "sensors" / "flir_boson_640_lwir.yaml"


@pytest.fixture(scope="module")
def boson_dict() -> dict[str, Any]:
    with BOSON_YAML.open() as fh:
        return yaml.safe_load(fh)  # type: ignore[no-any-return]


@pytest.fixture
def boson(boson_dict: dict[str, Any]) -> SensorConfig:
    return SensorConfig.model_validate(boson_dict)


def _set(d: dict[str, Any], dotted: str, value: Any) -> dict[str, Any]:
    out = copy.deepcopy(d)
    node = out
    *path, last = dotted.split(".")
    for key in path:
        node = node[key]
    node[last] = value
    return out


def test_boson_matches_spec_16_1_exactly(boson: SensorConfig) -> None:
    s = boson.sensor
    assert s.name == "flir_boson_640_lwir"
    assert (s.band.lambda_min_um, s.band.lambda_max_um) == (7.5, 13.5)
    assert s.band.regime == "emissive"
    assert isinstance(s.fpa, BolometerFpa)
    assert (s.fpa.width, s.fpa.height, s.fpa.pitch_um) == (640, 512, 12.0)
    assert s.optics.f_number == 1.0 and s.optics.focal_length_mm == 14.0
    assert s.noise.netd_mk_at_300k == 50.0
    assert s.noise.ratios_3d.vh == 0.30 and s.noise.ratios_3d.t == 0.02
    assert s.isp.clip_percentiles == (0.005, 0.995)
    assert boson.schema_version == 10


def test_pixel_area_and_nyquist(boson: SensorConfig) -> None:
    s = boson.sensor
    assert s.pixel_area_m2 == pytest.approx(1.296e-10, rel=1e-12)  # (12 um)^2 * 0.90
    assert s.nyquist_cyc_per_mm == pytest.approx(41.67, rel=1e-3)  # §8.3 worked case


def test_hfov_documents_the_datasheet_discrepancy(boson: SensorConfig) -> None:
    """14 mm with 640 x 12 um gives 30.7 deg; §16.1 quotes 32 deg (open question 7)."""
    assert boson.sensor.hfov_deg == pytest.approx(30.7, abs=0.05)
    assert abs(boson.sensor.hfov_deg / 32.0 - 1.0) < 0.05


def test_frame_period_and_bolometer_smear_ratio(boson: SensorConfig) -> None:
    s = boson.sensor
    assert s.frame_period_s == pytest.approx(1.0 / 60.0, rel=1e-12)
    assert isinstance(s.fpa, BolometerFpa)
    # tau/frame period: 8 ms at 60 Hz ([R24]'s nominal figure, SC.3). It was 0.6 while the
    # config carried an ESTIMATED 10 ms, so the modelled detector was slower than the part.
    assert s.fpa.thermal_time_constant_ms * 1e-3 / s.frame_period_s == pytest.approx(0.48, rel=0.01)
    assert s.dn_max == 65535


def test_spec_example_with_null_photon_fields_loads(boson_dict: dict[str, Any]) -> None:
    d = copy.deepcopy(boson_dict)
    d["sensor"]["fpa"].update(
        quantum_efficiency=None,
        well_capacity_e=None,
        integration_time_ms=None,
        dark_current_model=None,
    )
    assert isinstance(SensorConfig.model_validate(d).sensor.fpa, BolometerFpa)


def test_photon_fpa_variant(boson_dict: dict[str, Any]) -> None:
    d = copy.deepcopy(boson_dict)
    d["sensor"]["band"].update(lambda_min_um=0.9, lambda_max_um=1.7, regime="reflective")
    d["sensor"]["fpa"] = {
        "type": "photon",
        "width": 640,
        "height": 512,
        "pitch_um": 15.0,
        "fill_factor": 1.0,
        "frame_rate_hz": 30,
        "bit_depth": 14,
        "quantum_efficiency": 0.8,
        "well_capacity_e": 1.0e6,
        "integration_time_ms": 10.0,
        "dark_current_model": "constant",
    }
    cfg = SensorConfig.model_validate(d)
    assert isinstance(cfg.sensor.fpa, PhotonFpa)
    with pytest.raises(ValidationError, match="frame period"):
        SensorConfig.model_validate(_set(d, "sensor.fpa.integration_time_ms", 50.0))
    with pytest.raises(ValidationError):  # bolometer field with a value on a photon FPA
        SensorConfig.model_validate(_set(d, "sensor.fpa.tcr_per_k", -0.02))


@pytest.mark.parametrize(
    ("dotted", "value", "why"),
    [
        ("sensor.optics.f_stop", 1.0, "typo must not silently default"),
        ("sensor.fpa.quantum_efficiency", 0.8, "photon field on a bolometer"),
        ("sensor.band.lambda_min_um", 7500.0, "nanometres"),
        ("sensor.band.lambda_max_um", 7.0, "max below min"),
        ("sensor.band.regime", "reflective", "reflective LWIR contradicts §12.1"),
        ("sensor.noise.ratios_3d.tvh", 0.9, "tvh is the unit"),
        ("sensor.isp.clip_percentiles", [0.99, 0.01], "unordered percentiles"),
        ("sensor.optics.transmittance", 1.2, "transmittance > 1"),
        ("sensor.optics.cold_shield_efficiency", -0.1, "efficiency < 0"),
        ("sensor.fpa.fill_factor", 0.0, "zero fill factor"),
        ("sensor.fpa.bit_depth", 32, "bit depth > 16"),
        ("sensor.noise.bad_pixel_fraction", 0.05, "5 % bad pixels is a broken camera"),
        ("sensor.optics.distortion.coeffs", [0.0, 0.0], "brown_conrady takes 5"),
        ("sensor.isp.palette", "viridis", "unknown palette"),
        ("schema_version", 99, "unsupported schema version"),
    ],
)
def test_invalid_values_rejected(
    boson_dict: dict[str, Any], dotted: str, value: Any, why: str
) -> None:
    with pytest.raises(ValidationError):
        SensorConfig.model_validate(_set(boson_dict, dotted, value))


def test_emissive_regime_rejected_in_swir(boson_dict: dict[str, Any]) -> None:
    d = _set(boson_dict, "sensor.band.lambda_min_um", 0.9)
    d = _set(d, "sensor.band.lambda_max_um", 1.7)
    with pytest.raises(ValidationError, match="emissive"):
        SensorConfig.model_validate(d)


def test_models_are_frozen(boson: SensorConfig) -> None:
    with pytest.raises(ValidationError):
        boson.sensor.optics.f_number = 2.0  # type: ignore[misc]


def test_no_aperture_factor_in_config_layer() -> None:
    """Non-negotiable #5: the aperture factor is defined once in irsim.optics, never here.

    Scans code only (docstrings and comments are dropped via the AST), so prose that
    *mentions* the factor is fine but an expression that computes it is not."""
    pattern = re.compile(r"f_number\s*\*\*\s*2|4(\.0)?\s*\*\s*\S*f_number|\bF\s*\*\*\s*2")
    # positive controls: the scanner must catch the forms someone would actually write
    for offending in ("math.pi / (4 * self.f_number**2)", "4.0 * optics.f_number ** 2 + 1"):
        assert pattern.search(offending), offending
    for path in (REPO / "src" / "irsim" / "config").rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef)):
                node.body = [
                    n
                    for n in node.body
                    if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant))
                ] or [ast.Pass()]
        code = ast.unparse(tree)
        assert not pattern.search(code), f"{path.name} contains an aperture expression"
