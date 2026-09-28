"""AT.21 -- the per-pixel slant path reaches `run_frame`, not only the stage beside it.

`AT.1` taught `apply_layered_gbuffer` to take an ``elevation_rad`` plane and `atmosphere_stage`
passed it; `run_frame` -- the entry point every render driver and the Isaac camera actually call
-- did not. So every *resolved* pixel kept the horizontal column while the sky behind it and the
point targets in front of it took the slant one, which is the very handoff error `AT.1` was
written to remove. The Isaac adapter wrote the plane (`ir_camera.py`) and `run_frame` dropped it.

The test that carries weight is the first: the entry point must produce what the stage produces
on the same plane dict. The last records the size of what was wrong, in kelvin, so the fix is a
number rather than a claim. docs/physics-model.md §7.4, §5.3; ADR 0071; roadmap AT.1, AT.21.
"""

from __future__ import annotations

import copy
import pathlib

import numpy as np
import pytest
import yaml

from irsim.atmosphere.layered import LayeredAtmosphere
from irsim.atmosphere.library import load_atmosphere_preset
from irsim.config.sensor import SensorConfig
from irsim.materials import MaterialTable
from irsim.pipeline import PipelineConfig, PipelineState, run_frame
from irsim.pipeline.atmosphere import atmosphere_stage
from irsim.pipeline.radiance import band_radiance_stage
from irsim.radiometry.encoding import encode_temperature
from irsim.radiometry.lut import BandLUT
from irsim.radiometry.spectral_response import load_spectral_response
from irsim.thermal import WeatherSample, WeatherSeries

REPO = pathlib.Path(__file__).resolve().parents[2]
BOSON = yaml.safe_load((REPO / "configs" / "sensors" / "flir_boson_640_lwir.yaml").read_text())
SHAPE = (16, 32)


@pytest.fixture(scope="module")
def lut() -> BandLUT:
    response = load_spectral_response(REPO / "data" / "spectra" / "responses" / "boson_vox.csv")
    return BandLUT.build(response)


@pytest.fixture(scope="module")
def config(lut: BandLUT) -> PipelineConfig:
    response = load_spectral_response(REPO / "data" / "spectra" / "responses" / "boson_vox.csv")
    weather = WeatherSeries.constant(
        WeatherSample(288.15, 0.46, 1.0, 0.0, 0.0, 0.0, 23000.0, 0.0), 3600.0
    )
    atmosphere = LayeredAtmosphere(
        load_atmosphere_preset("us_standard_clear"), weather, {"lwir": lut}, {"lwir": response}
    )
    d = copy.deepcopy(BOSON)
    d["sensor"]["fpa"].update(width=SHAPE[1], height=SHAPE[0])
    d["sensor"]["optics"]["supersample_factor"] = 1
    return PipelineConfig.from_sensor(
        SensorConfig.model_validate(d),
        MaterialTable.constant(1.0),
        lut=lut,
        noise_enabled=False,
        psf_enabled=False,
        atmosphere=atmosphere,
    )


def _gbuffer(t_k: float, distance_m: float, elevation: np.ndarray | None) -> dict:
    temp = np.full(SHAPE, t_k, dtype=np.float32)
    planes = {
        "temperature_k": temp,
        "encoded_t": encode_temperature(temp),
        "normal_dot_view": np.ones(SHAPE, np.float32),
        "distance_m": np.full(SHAPE, distance_m, np.float32),
        "material_id": np.ones(SHAPE, np.int32),
        "sky_view_factor": np.full(SHAPE, 0.5, np.float32),
    }
    if elevation is not None:
        planes["elevation_rad"] = np.asarray(elevation, dtype=np.float32)
    return planes


def _varying_elevation() -> np.ndarray:
    """A horizon in the frame: 0° on the bottom row, 60° on the top."""
    rows = np.radians(np.linspace(60.0, 0.0, SHAPE[0]))[:, None]
    return rows * np.ones((1, SHAPE[1]))


def test_run_frame_matches_the_stage_on_a_varying_elevation_plane(
    config: PipelineConfig, lut: BandLUT
) -> None:
    """The entry point and the stage are the same physics on the same planes (red before AT.21:
    the entry point ignored the plane and every row came out at the horizon's transmittance)."""
    planes = _gbuffer(310.0, 5000.0, _varying_elevation())
    state = PipelineState(housing_temp_k=300.0)
    out = run_frame(planes, config, state)
    assert out.radiance is not None
    staged = dict(planes)
    staged.update(band_radiance_stage(staged, config, PipelineState(housing_temp_k=300.0)))
    staged.update(atmosphere_stage(staged, config, PipelineState(housing_temp_k=300.0)))
    expected = np.asarray(staged["radiance"], dtype=np.float64)
    # The radiometric branch inverts a noiseless ideal chain, so the residual is the LUT round
    # trip (< 0.1 mK), not the physics.
    t_got = lut.apparent_temperature(out.radiance.astype(np.float64), "lb")
    t_exp = lut.apparent_temperature(expected, "lb")
    assert np.max(np.abs(t_got - t_exp)) * 1000.0 < 10.0  # mK
    # And the frame actually varies with elevation: steeper rays attenuate the scene less.
    assert np.all(np.diff(out.radiance.mean(axis=1)) < 0.0)


def test_a_horizontal_plane_is_the_same_frame_as_no_plane(config: PipelineConfig) -> None:
    """Every scene that supplies no plane renders exactly as it did before AT.21."""
    without = run_frame(_gbuffer(310.0, 5000.0, None), config, PipelineState(housing_temp_k=300.0))
    flat = run_frame(
        _gbuffer(310.0, 5000.0, np.zeros(SHAPE)), config, PipelineState(housing_temp_k=300.0)
    )
    assert without.radiance is not None and flat.radiance is not None
    assert np.allclose(without.radiance, flat.radiance, rtol=1e-6, atol=0.0)


def test_the_defect_it_fixes_is_this_big(config: PipelineConfig) -> None:
    """A 310 K target at 5 km and 45° elevation read more than a kelvin low through the entry
    point before the fix (measured 1.14 K, with the path radiance 60 % high): the plane changes
    the answer by more than twenty NETDs, so a test that did not pass it could not have seen it."""
    state = PipelineState(housing_temp_k=300.0)
    slant = run_frame(_gbuffer(310.0, 5000.0, np.full(SHAPE, np.radians(45.0))), config, state)
    flat = run_frame(_gbuffer(310.0, 5000.0, None), config, PipelineState(housing_temp_k=300.0))
    assert slant.apparent_t is not None and flat.apparent_t is not None
    delta_k = float(np.mean(slant.apparent_t) - np.mean(flat.apparent_t))
    assert delta_k > 1.0, f"the slant path moves the target by only {delta_k:.3f} K"
