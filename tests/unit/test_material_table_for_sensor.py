"""A material table is packed for the camera that will use it (AT.33, ADR 0175).

`MaterialTable.from_library(library, "lwir")` averages every curve over the band's nominal
top-hat in energy form. `for_sensor` uses the camera's own R(λ) and its detector's weighting, so:
a curve material reads the Boson's value, not the top-hat's; a photon FPA averages under B_q; a
6-13 µm camera and a 7.5-13.5 µm one read a silicate-like curve differently; and a material
without a curve packs the same number whatever the camera.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest
import yaml

from irsim.config.loader import load_sensor_config
from irsim.config.sensor import SensorConfig
from irsim.materials.library import MaterialLibrary
from irsim.materials.table import MaterialTable
from irsim.radiometry.lut_files import load_band_response_for_config

REPO = pathlib.Path(__file__).resolve().parents[2]
BOSON = REPO / "configs" / "sensors" / "flir_boson_640_lwir.yaml"
INSB = REPO / "configs" / "sensors" / "example_mwir_insb_640.yaml"


def _material(name: str, optical: dict[str, object]) -> dict[str, object]:
    return {
        "schema_version": 2,
        "material": {
            "name": name,
            "source": "estimated",
            "surface_treatment": "as_manufactured",
            "thermal": {
                "density_kg_m3": 2200,
                "specific_heat_j_kgk": 750,
                "conductivity_w_mk": 1.4,
                "thickness_m": 0.005,
                "solar_absorptivity": 0.3,
            },
            "optical": optical,
        },
    }


def _silicate(lam: np.ndarray) -> np.ndarray:
    """ε 0.95 with a reststrahlen-like dip to 0.70 across 8.0-9.5 µm, and a slope in MWIR."""
    dip = 0.5 * (np.tanh((lam - 8.0) / 0.08) - np.tanh((lam - 9.5) / 0.08))
    mwir_slope = np.where(lam < 6.0, 0.80 + 0.03 * (lam - 3.0), 0.95)
    return np.where(lam < 6.0, mwir_slope, 0.95 - 0.25 * dip)


@pytest.fixture()
def library(tmp_path: pathlib.Path) -> MaterialLibrary:
    lam = np.linspace(0.3, 16.0, 3141)
    rows = "\n".join(f"{a:.6f},{b:.9f}" for a, b in zip(lam, _silicate(lam), strict=True))
    (tmp_path / "silicate.csv").write_text("# test curve\n" + rows + "\n")
    mats = tmp_path / "materials"
    mats.mkdir()
    for name, optical in {
        "silicate": {"spectral_emissivity": str(tmp_path / "silicate.csv")},
        "painted": {"emissivity_per_band": {"mwir": 0.88, "lwir": 0.92}},
        "grey": {"emissivity": 0.97},
    }.items():
        (mats / f"{name}.yaml").write_text(yaml.safe_dump(_material(name, optical)))
    return MaterialLibrary.load(mats)


def _custom_lwir(tmp_path: pathlib.Path, lo: float, hi: float) -> SensorConfig:
    """The Boson with its band moved to [lo, hi] and a matching top-hat response file."""
    doc = yaml.safe_load(BOSON.read_text())
    lam = np.round(np.arange(lo - 0.5, hi + 0.5001, 0.01), 4)
    r = ((lam >= lo) & (lam <= hi)).astype(float)
    path = tmp_path / f"r_{lo}_{hi}.csv"
    path.write_text(
        "# test top-hat\nwavelength_um,response\n"
        + "\n".join(f"{a},{b}" for a, b in zip(lam, r, strict=True))
        + "\n"
    )
    doc["sensor"]["band"].update(lambda_min_um=lo, lambda_max_um=hi, spectral_response=str(path))
    return SensorConfig.model_validate(doc)


def _eps(table: MaterialTable, name: str) -> float:
    return float(table.emissivity[table.id_for(name)])


def test_a_curve_packs_the_boson_s_value_not_the_top_hat_s(library: MaterialLibrary) -> None:
    boson = load_sensor_config(BOSON)
    packed = MaterialTable.for_sensor(library, boson)
    response = load_band_response_for_config(boson)
    expected = library["silicate"].band_properties("lwir", response, "energy").emissivity
    assert _eps(packed, "silicate") == pytest.approx(expected, abs=1e-6)  # float32 column
    top_hat = MaterialTable.from_library(library, "lwir")
    # Measured: the Boson is a top-hat at the same edges with ±0.25 µm raised-cosine skirts, and
    # both skirts sit on this curve's flat shoulder, so the two agree to 1.6e-5 (~1 mK). That is
    # why switching every driver to `for_sensor` barely moves a nominal-band render; the
    # difference lives in cameras whose range is not the band's (the next test).
    assert abs(_eps(packed, "silicate") - _eps(top_hat, "silicate")) < 1e-4


def test_a_photon_camera_averages_under_photon_weighting(library: MaterialLibrary) -> None:
    insb = load_sensor_config(INSB)
    assert insb.sensor.fpa.type == "photon"
    packed = MaterialTable.for_sensor(library, insb)
    response = load_band_response_for_config(insb)
    photon = library["silicate"].band_properties("mwir", response, "photon").emissivity
    energy = library["silicate"].band_properties("mwir", response, "energy").emissivity
    assert _eps(packed, "silicate") == pytest.approx(photon, abs=1e-6)
    # ε rises with λ across MWIR here, and photon weighting leans long-wave: higher, by a margin
    # the float32 column can see.
    assert photon - energy > 1e-4


def test_two_lwir_ranges_differ_on_a_curve_and_agree_without_one(
    library: MaterialLibrary, tmp_path: pathlib.Path
) -> None:
    wide = MaterialTable.for_sensor(library, _custom_lwir(tmp_path, 7.5, 13.5))
    shifted = MaterialTable.for_sensor(library, _custom_lwir(tmp_path, 6.0, 13.0))
    window = MaterialTable.for_sensor(library, _custom_lwir(tmp_path, 10.0, 13.0))
    assert wide.band_id == shifted.band_id == window.band_id == "lwir"
    # 6-8 µm is all shoulder, so the dip is a smaller share of a 6-13 µm band than of a
    # 7.5-13.5 µm one, which therefore reads lower; a 10-13 µm camera sees none of the dip.
    assert _eps(wide, "silicate") < _eps(shifted, "silicate") < _eps(window, "silicate")
    assert _eps(window, "silicate") == pytest.approx(0.95, abs=1e-4)
    for name in ("painted", "grey"):
        assert _eps(wide, name) == _eps(shifted, name) == _eps(window, name)
    assert _eps(wide, "painted") == pytest.approx(0.92, abs=1e-7)
    assert _eps(wide, "grey") == pytest.approx(0.97, abs=1e-7)


def test_a_response_that_contradicts_the_band_edges_is_refused(
    library: MaterialLibrary, tmp_path: pathlib.Path
) -> None:
    """The table goes through the same edge check as the LUT, so the two cannot disagree."""
    config = _custom_lwir(tmp_path, 7.5, 13.5)
    doc = config.model_dump(mode="json")
    doc["sensor"]["band"]["lambda_min_um"] = 6.0
    with pytest.raises(ValueError, match="half-power"):
        MaterialTable.for_sensor(library, SensorConfig.model_validate(doc))
