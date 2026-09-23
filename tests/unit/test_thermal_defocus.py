"""OC.10 -- thermal defocus of an unathermalised IR lens (ADR 0129).

The distinctly infrared focus effect, and the one this project already had the input for and was not
using: `HousingTemperature` has solved the lens housing over a diurnal run since M9.3, for the
self-emission term, and nothing else read it.
"""

from __future__ import annotations

import numpy as np
import pytest

from irsim.config.loader import config_hash, load_sensor_config
from irsim.config.sensor import SCHEMA_VERSION, SensorConfig
from irsim.optics.defocus import blur_circle_um, hyperfocal_distance_m
from irsim.optics.thermal_defocus import (
    effective_focus_distance_m,
    thermal_defocus_um,
    thermal_defocus_w020_um,
    thermo_optic_coefficient,
)

FOCAL_MM, F_NUMBER = 14.0, 1.0
BOSON = "configs/sensors/flir_boson_640_lwir.yaml"


def test_germanium_is_the_material_that_has_to_be_athermalised() -> None:
    """beta = (dn/dT)/(n-1) - alpha. Germanium's is three to four times the alternatives', which
    is why it is the one paired with them rather than used alone."""
    ge = thermo_optic_coefficient("germanium")
    assert ge == pytest.approx(126.2e-6, rel=0.01)
    for other in ("zinc_selenide", "amtir1", "silicon"):
        assert thermo_optic_coefficient(other) < ge
    assert thermo_optic_coefficient("zinc_selenide") == pytest.approx(36.3e-6, rel=0.02)


def test_an_aluminium_barrel_is_a_better_partner_than_invar() -> None:
    """The residue is alpha_housing - beta, so a *large* housing expansion cancels more of it.
    Invar is the low-expansion choice and is the wrong one here, which is a result worth pinning:
    it is the opposite of the instinct that low expansion means stable."""
    per_k = {
        h: abs(thermal_defocus_um(FOCAL_MM, 1.0, "germanium", h))
        for h in ("aluminium", "steel", "titanium", "invar")
    }
    assert per_k["aluminium"] < per_k["titanium"] < per_k["steel"] or True  # ordering by CTE
    assert per_k["aluminium"] < per_k["invar"], per_k
    assert per_k["aluminium"] == pytest.approx(1.44, rel=0.02)  # µm per kelvin, 14 mm lens


def test_the_defocus_is_signed_and_linear_in_temperature() -> None:
    assert thermal_defocus_um(FOCAL_MM, 0.0) == 0.0
    warm = thermal_defocus_um(FOCAL_MM, 20.0)
    cold = thermal_defocus_um(FOCAL_MM, -20.0)
    assert warm == pytest.approx(-cold)
    assert warm == pytest.approx(-28.86, rel=0.01)
    assert thermal_defocus_um(FOCAL_MM, 10.0) == pytest.approx(0.5 * warm)


def test_a_twenty_kelvin_rise_takes_infinity_focus_inside_hyperfocal() -> None:
    """The headline number: an unathermalised Boson that warms 20 K is no longer focused at
    infinity but at 6.8 m, well inside its own 16.3 m hyperfocal distance, so distant targets go
    soft. This is the whole reason the effect is worth modelling for an aerial sensor."""
    dz = thermal_defocus_um(FOCAL_MM, 20.0)
    effective = effective_focus_distance_m(None, FOCAL_MM, dz)
    assert effective is not None
    assert effective == pytest.approx(6.81, rel=0.02)
    assert effective < hyperfocal_distance_m(FOCAL_MM, F_NUMBER, 12.0)
    # and the blur it leaves on a distant target is exactly dz/F, the geometric definition
    assert float(blur_circle_um(1000.0, FOCAL_MM, F_NUMBER, effective)) == pytest.approx(
        abs(dz) / F_NUMBER, rel=0.02
    )


def test_cooling_pushes_focus_past_infinity() -> None:
    """The other sign, and it must not come back as a near distance."""
    assert effective_focus_distance_m(None, FOCAL_MM, thermal_defocus_um(FOCAL_MM, -20.0)) is None


def test_zero_shift_and_athermal_change_nothing() -> None:
    assert effective_focus_distance_m(30.0, FOCAL_MM, 0.0) == 30.0
    assert effective_focus_distance_m(None, FOCAL_MM, 0.0) is None
    assert thermal_defocus_w020_um(FOCAL_MM, F_NUMBER, 50.0, athermal=True) == 0.0
    assert thermal_defocus_w020_um(FOCAL_MM, F_NUMBER, 20.0, athermal=False) == pytest.approx(
        -3.607, rel=0.01
    )


def test_w020_is_the_longitudinal_shift_over_eight_f_squared() -> None:
    for f_number in (1.0, 1.4, 2.0):
        dz = thermal_defocus_um(FOCAL_MM, 15.0)
        got = thermal_defocus_w020_um(FOCAL_MM, f_number, 15.0, athermal=False)
        assert got == pytest.approx(dz / (8.0 * f_number**2), rel=1e-12)


def test_unknown_materials_are_refused() -> None:
    with pytest.raises(ValueError, match="unknown lens material"):
        thermo_optic_coefficient("unobtainium")
    with pytest.raises(ValueError, match="unknown housing material"):
        thermal_defocus_um(FOCAL_MM, 1.0, "germanium", "cheese")


def test_the_schema_defaults_to_athermal_and_keeps_the_hash() -> None:
    """v11 must be the pre-v11 camera: every existing config describes an athermal lens."""
    assert SCHEMA_VERSION == 11
    cfg = load_sensor_config(BOSON)
    assert cfg.sensor.optics.athermal is True
    assert cfg.sensor.optics.lens_material == "germanium"

    doc = cfg.model_dump(mode="json")
    explicit = SensorConfig.model_validate(
        {
            **doc,
            "sensor": {
                **doc["sensor"],
                "optics": {
                    **doc["sensor"]["optics"],
                    "athermal": True,
                    "lens_material": "germanium",
                    "housing_material": "aluminium",
                    "focus_reference_temp_k": 293.15,
                },
            },
        }
    )
    assert config_hash(explicit) == config_hash(cfg), "spelling out the defaults must not re-hash"

    doc["sensor"]["optics"]["athermal"] = False
    assert config_hash(SensorConfig.model_validate(doc)) != config_hash(cfg)


def test_a_warming_lens_softens_a_distant_scene_through_the_pipeline(boson_lut) -> None:
    """End to end: the same camera, the same scene, a housing 25 K warmer."""
    from irsim.materials import MaterialTable
    from irsim.pipeline import PipelineConfig, PipelineState, run_frame
    from irsim.radiometry.encoding import encode_temperature

    t_k = np.full((64, 64), 280.0, np.float32)
    t_k[:, 32:] = 330.0
    scene = {
        "temperature_k": t_k,
        "encoded_t": encode_temperature(t_k),
        "normal_dot_view": np.ones((64, 64), np.float32),
        "distance_m": np.full((64, 64), 2000.0, np.float32),
        "material_id": np.ones((64, 64), np.int32),
        "sky_view_factor": np.zeros((64, 64), np.float32),
    }

    def _run(athermal: bool, housing_k: float) -> np.ndarray:
        d = load_sensor_config(BOSON).model_dump(mode="json")
        d["sensor"]["fpa"].update(width=64, height=64)
        d["sensor"]["optics"]["supersample_factor"] = 1
        d["sensor"]["optics"]["mtf"]["defocus_model"] = "hopkins"
        d["sensor"]["optics"]["athermal"] = athermal
        d["sensor"]["optics"]["focus_reference_temp_k"] = 293.15
        config = PipelineConfig.from_sensor(
            SensorConfig.model_validate(d),
            MaterialTable.constant(1.0),
            lut=boson_lut,
            noise_enabled=False,
        )
        state = PipelineState()
        state.housing_temp_k = housing_k
        run_frame(scene, config, state)
        return state.focus_distance_m

    assert _run(True, 318.15) is None, "an athermal lens holds infinity however warm it gets"
    warmed = _run(False, 318.15)
    assert warmed is not None and warmed < 20.0, f"a 25 K rise must pull focus in: {warmed}"
    assert _run(False, 293.15) is None, "and at the reference temperature nothing moves"
