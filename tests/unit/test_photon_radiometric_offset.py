"""SC.30 -- the photon-FPA radiometric branch removes its own pedestal.

`PhotonDetector.electrons` adds the dark and cold-shield background electrons (§9.1, ADR 0066),
and `run_frame`'s inversion took the whole count as signal. Every uncooled camera in this
repository was unaffected (η_cs = 1, background 0, dark small against a 300 K LWIR scene), which
is why the existing photon round trip in `test_run_frame.py` passed: it stands an LWIR table in
for MWIR. On the MWIR InSb config as shipped -- η_cs = 0.90, 2 ms integration, 8.5 Me⁻ well --
the pedestal is 3.3e5 e⁻ against 3.3e6 of signal, and a 300 K blackbody read **303.5 K**.

A calibrated camera subtracts the pedestal's mean (a dark frame measures it); only its shot
noise survives, which the noise budget already carries. docs/physics-model.md §9.1, §11.5;
ADR 0021, ADR 0066; roadmap SC.1, SC.30.
"""

from __future__ import annotations

import copy
import pathlib

import numpy as np
import pytest
import yaml

from irsim.config.sensor import SensorConfig
from irsim.materials import MaterialTable
from irsim.pipeline import PipelineConfig, PipelineState, run_frame
from irsim.radiometry.encoding import encode_temperature
from irsim.radiometry.lut_files import load_band_lut_for_config

REPO = pathlib.Path(__file__).resolve().parents[2]
SHAPE = (16, 32)


@pytest.fixture(scope="module")
def config() -> PipelineConfig:
    raw = yaml.safe_load(
        (REPO / "configs" / "sensors" / "example_mwir_insb_640.yaml").read_text("utf-8")
    )
    d = copy.deepcopy(raw)
    d["sensor"]["fpa"].update(width=SHAPE[1], height=SHAPE[0])
    d["sensor"]["optics"]["supersample_factor"] = 1
    sensor = SensorConfig.model_validate(d)
    lut = load_band_lut_for_config(sensor, REPO / "data" / "lut")
    return PipelineConfig.from_sensor(
        sensor, MaterialTable.constant(1.0), lut=lut, noise_enabled=False, psf_enabled=False
    )


def _blackbody(t_k: float) -> dict[str, np.ndarray]:
    temp = np.full(SHAPE, t_k, dtype=np.float32)
    return {
        "temperature_k": temp,
        "encoded_t": encode_temperature(temp),
        "normal_dot_view": np.ones(SHAPE, np.float32),
        "distance_m": np.full(SHAPE, 50.0, np.float32),
        "material_id": np.ones(SHAPE, np.int32),
        "sky_view_factor": np.full(SHAPE, 0.5, np.float32),
    }


def test_the_pedestal_is_large_enough_to_matter(config: PipelineConfig) -> None:
    """The test below is only a test if the offset is not zero: on this camera it is a tenth of
    the 300 K signal, so an inversion that keeps it cannot land within 10 mK."""
    budget = config.detector.budget  # type: ignore[attr-defined]
    pedestal = float(budget.dark_electrons) + float(budget.background_electrons)
    assert pedestal > 1e5, pedestal


@pytest.mark.parametrize("t_k", [250.0, 300.0, 350.0])
def test_a_blackbody_reads_its_own_temperature_through_the_photon_chain(
    config: PipelineConfig, t_k: float
) -> None:
    """Round trip through the ideal chain: within 10 mK, where before the fix 300 K read 303.5 K."""
    out = run_frame(_blackbody(t_k), config, PipelineState(housing_temp_k=300.0))
    assert out.apparent_t is not None
    error_mk = abs(float(out.apparent_t[8, 16]) - t_k) * 1000.0
    assert error_mk < 10.0, f"{t_k} K reads {error_mk:.1f} mK off"
