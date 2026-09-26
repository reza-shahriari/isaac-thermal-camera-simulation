"""SC.17: the housing is seen through the field, not added as one number (§8.2, ADR 0145).

A pixel at field angle θ sees the aperture as the projected solid angle Ω_eff RI_ij and the inside
of the camera over the rest of its hemisphere. With lens and housing at one temperature the power
on a uniform scene, referenced to the optical axis, is

    Φ_ij = A_d Ω_eff [L_h + τ_opt RI_ij (L_scene − L_h)]

so the relative illumination multiplies the scene *minus* the housing. Every test here would fail
on the pre-SC.17 form Φ = A_d Ω_eff [τ RI L_scene + (1 − τ) L_h], which darkens the corners of
every scene, including one at the housing temperature.
"""

from __future__ import annotations

import copy
import pathlib
from typing import Any

import numpy as np
import pytest
import yaml

from irsim.config.loader import config_hash, load_sensor_config
from irsim.config.sensor import SensorConfig, SensorSpec
from irsim.optics import (
    aperture_factor,
    apply_optics,
    housing_power_axis,
    housing_power_field,
    invert_optics,
    optics_field,
    self_emission_power,
)
from irsim.optics.vignetting import load_vignetting_map
from irsim.radiometry.planck import band_radiance_tophat

REPO = pathlib.Path(__file__).resolve().parents[2]
BOSON_YAML = REPO / "configs" / "sensors" / "flir_boson_640_lwir.yaml"
BOSON = yaml.safe_load(BOSON_YAML.read_text())


def _lb(t_k: float) -> float:
    return float(band_radiance_tophat(7.5, 13.5, t_k))


def _small(width: int = 64, height: int = 48, **optics: Any) -> SensorSpec:
    d = copy.deepcopy(BOSON)
    d["sensor"]["fpa"].update(width=width, height=height)
    # a short lens so a small array still spans a real field angle (corner cos⁴ ≈ 0.8)
    d["sensor"]["optics"].update({"focal_length_mm": 1.0, **optics})
    return SensorConfig.model_validate(d).sensor


def _corner_over_centre(plane: np.ndarray) -> float:
    h, w = plane.shape
    corners = np.mean([plane[0, 0], plane[0, -1], plane[-1, 0], plane[-1, -1]])
    centre = plane[h // 2 - 1 : h // 2 + 1, w // 2 - 1 : w // 2 + 1].mean()
    return float(corners / centre)


def _uniform_phi(s: SensorSpec, t_scene: float, t_housing: float) -> np.ndarray:
    rad = np.full(s.fpa_shape, _lb(t_scene), dtype=np.float64)
    return apply_optics(rad, s, _lb(t_housing), supersample=1).astype(np.float64)


# --- the sign of the shading ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("t_scene", "sign"),
    [(250.0, +1), (240.0, +1), (330.0, -1), (400.0, -1)],
)
def test_the_shading_has_the_sign_of_scene_minus_housing(t_scene: float, sign: int) -> None:
    """Colder than a 300 K housing (a clear sky), the corners are brighter; warmer, darker."""
    s = _small()
    ratio = _corner_over_centre(_uniform_phi(s, t_scene, 300.0))
    assert np.sign(ratio - 1.0) == sign, f"{t_scene} K: corner/centre {ratio:.5f}"


def test_a_scene_at_the_housing_temperature_is_not_shaded() -> None:
    """An isothermal enclosure: every pixel sees 300 K whichever way it looks."""
    s = _small()
    phi = _uniform_phi(s, 300.0, 300.0)
    assert np.ptp(phi) / phi.mean() < 1e-6


def test_the_shading_depth_is_the_closed_form() -> None:
    """Corner minus centre is A_d Ω_eff τ (RI_corner − RI_centre)(L_scene − L_h), exactly."""
    s = _small()
    ri = optics_field(s).astype(np.float64)
    a_d, f, tau = s.detector_active_area_m2, s.optics.f_number, s.optics.transmittance
    phi = _uniform_phi(s, 250.0, 300.0)
    expected = a_d * aperture_factor(f) * tau * (ri[0, 0] - ri.max()) * (_lb(250.0) - _lb(300.0))
    got = phi[0, 0] - phi[np.unravel_index(np.argmax(ri), ri.shape)]
    assert got == pytest.approx(expected, rel=1e-5)


# --- what did not move ----------------------------------------------------------------------


def test_on_axis_power_is_the_single_lens_form_exactly() -> None:
    """With RI = 1 (cos⁴ off, no map) the new term is the old A_d Ω_eff (1 − τ) L_h to 1e-12,
    so no on-axis number the project has measured -- the 87 mK/K housing sensitivity, NETD --
    moves."""
    a_d, f, tau, lb_h = 1.44e-10, 1.0, 0.92, _lb(305.0)
    assert float(housing_power_field(a_d, f, tau, lb_h, 1.0)) == pytest.approx(
        self_emission_power(a_d, f, tau, lb_h), rel=1e-12
    )
    s = _small(vignetting_cos4=False)
    phi = _uniform_phi(s, 280.0, 310.0)
    ad = s.detector_active_area_m2
    old = ad * aperture_factor(1.0) * (0.92 * _lb(280.0) + 0.08 * _lb(310.0))
    assert np.allclose(phi, old, rtol=1e-6)


def test_the_change_from_the_old_form_is_the_out_of_cone_housing_view() -> None:
    """New − old = A_d Ω_eff τ (1 − RI_ij) L_h: the housing a pixel sees in place of scene."""
    s = _small()
    ri = optics_field(s).astype(np.float64)
    a_d, f, tau = s.detector_active_area_m2, s.optics.f_number, s.optics.transmittance
    lb_s, lb_h = _lb(260.0), _lb(300.0)
    old = a_d * aperture_factor(f) * (tau * ri * lb_s + (1.0 - tau) * lb_h)
    new = _uniform_phi(s, 260.0, 300.0)
    np.testing.assert_allclose(
        new - old, a_d * aperture_factor(f) * tau * (1.0 - ri) * lb_h, rtol=2e-4, atol=0
    )
    assert housing_power_axis(a_d, f, tau, lb_h) == pytest.approx(a_d * aperture_factor(f) * lb_h)


# --- the inverse and a drifted housing ------------------------------------------------------


def test_round_trip_is_exact_everywhere_at_the_calibration_housing() -> None:
    s = _small()
    rng = np.random.default_rng(3)
    rad = _lb(250.0) + rng.random(s.fpa_shape) * (_lb(320.0) - _lb(250.0))
    back = invert_optics(apply_optics(rad, s, _lb(300.0), supersample=1), s, _lb(300.0))
    np.testing.assert_allclose(back, rad, rtol=2e-6)


def test_a_drifted_housing_reads_back_as_a_radial_shading() -> None:
    """The camera inverts with its *calibration* housing. A housing 1 K warmer adds
    A_d Ω_eff (1 − τ RI) ΔL_h, which reads back as ΔL_h (1 − τ RI)/(τ RI), deepest in the corners.
    The old scalar form gives (1 − τ)/(τ RI) instead: at a corner RI of 0.8 that is a third of the
    new value, so the closed form below separates the two."""
    s = _small()
    ri = optics_field(s).astype(np.float64)
    tau = s.optics.transmittance
    rad = np.full(s.fpa_shape, _lb(260.0))
    phi = apply_optics(rad, s, _lb(301.0), supersample=1)
    err = invert_optics(phi, s, _lb(300.0)).astype(np.float64) - rad
    expected = (_lb(301.0) - _lb(300.0)) * (1.0 - tau * ri) / (tau * ri)
    np.testing.assert_allclose(err, expected, rtol=5e-3)
    assert err[0, 0] > err[s.fpa.height // 2, s.fpa.width // 2]


# --- the measured map -----------------------------------------------------------------------


def _radial_map(h: int, w: int, corner: float) -> np.ndarray:
    y, x = np.mgrid[:h, :w]
    r2 = ((x + 0.5 - w / 2) / (w / 2)) ** 2 + ((y + 0.5 - h / 2) / (h / 2)) ** 2
    return 1.0 - (1.0 - corner) * r2 / 2.0


def test_a_measured_map_is_loaded_from_the_config_and_deepens_the_shading(
    tmp_path: pathlib.Path,
) -> None:
    """`optics.vignetting_map` was a schema key nothing read. It now multiplies cos⁴ into RI."""
    data = tmp_path / "data"
    (data / "spectra/responses").mkdir(parents=True)
    (data / "spectra/responses/boson_vox.csv").write_text(
        "wavelength_um,response\n7.0,0.0\n7.5,1.0\n13.5,1.0\n14.0,0.0\n"
    )
    (data / "optics").mkdir()
    m = _radial_map(48, 64, 0.7)
    np.save(data / "optics/map.npy", m)
    d = copy.deepcopy(BOSON)
    d["sensor"]["fpa"].update(width=64, height=48)
    d["sensor"]["optics"].update(focal_length_mm=1.0)
    plain_yaml, map_yaml = tmp_path / "plain.yaml", tmp_path / "map.yaml"
    plain_yaml.write_text(yaml.safe_dump(d))
    d["sensor"]["optics"]["vignetting_map"] = "optics/map.npy"
    map_yaml.write_text(yaml.safe_dump(d))

    plain = load_sensor_config(plain_yaml, data)
    mapped = load_sensor_config(map_yaml, data)
    assert pathlib.Path(mapped.sensor.optics.vignetting_map or "").is_absolute()
    np.testing.assert_allclose(
        optics_field(mapped.sensor), optics_field(plain.sensor) * m.astype(np.float32), rtol=1e-6
    )
    cold = [_corner_over_centre(_uniform_phi(c.sensor, 250.0, 300.0)) for c in (plain, mapped)]
    assert cold[1] > cold[0] > 1.0
    # the file is hashed by content, so an edited map is a different camera
    h0 = config_hash(mapped, data)
    np.save(data / "optics/map.npy", _radial_map(48, 64, 0.6))
    assert config_hash(mapped, data) != h0


def test_a_missing_map_file_is_an_error_naming_it(tmp_path: pathlib.Path) -> None:
    d = copy.deepcopy(BOSON)
    d["sensor"]["optics"]["vignetting_map"] = "optics/nope.npy"
    y = tmp_path / "s.yaml"
    y.write_text(yaml.safe_dump(d))
    with pytest.raises(FileNotFoundError, match="nope.npy"):
        load_sensor_config(y)


def test_the_map_loader_refuses_float16_and_the_wrong_grid(tmp_path: pathlib.Path) -> None:
    p16, pbad = tmp_path / "a.npy", tmp_path / "b.npy"
    np.save(p16, np.ones((4, 4), np.float16))
    np.save(pbad, np.ones((3, 4)))
    with pytest.raises(TypeError, match="float16"):
        load_vignetting_map(str(p16), 4, 4)
    with pytest.raises(ValueError, match="detector grid"):
        load_vignetting_map(str(pbad), 4, 4)


def test_relative_illumination_outside_0_1_is_refused() -> None:
    with pytest.raises(ValueError, match="relative illumination"):
        housing_power_field(1e-10, 1.0, 0.9, 1.0, np.array([0.5, 1.2]))
    with pytest.raises(ValueError, match="relative illumination"):
        housing_power_field(1e-10, 1.0, 0.9, 1.0, 0.0)
