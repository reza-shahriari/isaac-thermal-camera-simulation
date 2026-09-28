"""SC.28 -- within-frame motion smear reaches `run_frame`, not only the stage beside it.

`mtf_motion` has been in the cascade since M5; ADR 0077 wired it into `apply_optics` and
`optics_stage` passed the duty-scaled ``motion_px`` plane. `run_frame` -- what every render driver
and `IrCamera` call -- did not, so every frame this simulator has produced was sharp whatever
crossed it, while the Isaac adapter went to the trouble of synthesising the plane (IG.6). The
across-frame bolometer lag was applied; the within-frame smear, the only one a cooled photon
detector has, was not.

docs/physics-model.md §8.3, §9.2; ADR 0077; roadmap SC.28.
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
from irsim.pipeline.optics import optics_stage
from irsim.pipeline.radiance import band_radiance_stage
from irsim.radiometry.lut import BandLUT

REPO = pathlib.Path(__file__).resolve().parents[2]
BOSON = yaml.safe_load((REPO / "configs" / "sensors" / "flir_boson_640_lwir.yaml").read_text())


@pytest.fixture
def config(tophat_lwir_lut: BandLUT, step_edge_params: dict[str, float]) -> PipelineConfig:
    k = int(step_edge_params["supersample"])
    native = int(step_edge_params["size_px"]) // k
    d = copy.deepcopy(BOSON)
    d["sensor"]["fpa"].update(width=native, height=native)
    d["sensor"]["optics"]["supersample_factor"] = k
    return PipelineConfig.from_sensor(
        SensorConfig.model_validate(d),
        MaterialTable.constant(1.0, ids=(1, 2)),
        lut=tophat_lwir_lut,
        noise_enabled=False,
        psf_enabled=False,
    )


def test_run_frame_matches_the_optics_stage_on_a_moving_edge(
    config: PipelineConfig, gbuffer_moving_edge: list[dict[str, np.ndarray]]
) -> None:
    """The entry point and the stage are the same optics on the same planes (red before SC.28:
    the entry point ignored `motion_px` and rendered the edge sharp)."""
    planes = dict(gbuffer_moving_edge[0])
    out = run_frame(planes, config, PipelineState(housing_temp_k=300.0))
    staged = dict(planes)
    staged.update(band_radiance_stage(staged, config, PipelineState(housing_temp_k=300.0)))
    staged.update(optics_stage(staged, config, PipelineState(housing_temp_k=300.0)))
    expected = np.asarray(staged["flux"], dtype=np.float64)
    scale = float(np.abs(expected).max())
    assert float(np.abs(out.flux.astype(np.float64) - expected).max()) / scale < 1e-6


def test_the_smear_actually_changes_the_rendered_frame(
    config: PipelineConfig, gbuffer_moving_edge: list[dict[str, np.ndarray]]
) -> None:
    """A 2 px/frame edge on a bolometer (duty 1) must blur: the pixel-power difference at the edge
    is a few per cent of the step, which a sharp render cannot produce."""
    moving = dict(gbuffer_moving_edge[0])
    still = {key: value for key, value in moving.items() if key != "motion_px"}
    out_moving = run_frame(moving, config, PipelineState(housing_temp_k=300.0))
    out_still = run_frame(still, config, PipelineState(housing_temp_k=300.0))
    step = float(np.abs(out_still.flux).max() - np.abs(out_still.flux).min())
    difference = float(np.abs(out_moving.flux - out_still.flux).max())
    assert difference / step > 0.01, f"motion_px changed the flux by only {difference / step:.2%}"


def test_a_zero_motion_plane_is_the_same_frame_as_no_plane(
    config: PipelineConfig, gbuffer_moving_edge: list[dict[str, np.ndarray]]
) -> None:
    """Every scene without the plane renders exactly as it did before SC.28."""
    planes = dict(gbuffer_moving_edge[0])
    zero = dict(planes)
    zero["motion_px"] = np.zeros_like(planes["motion_px"])
    without = {key: value for key, value in planes.items() if key != "motion_px"}
    out_zero = run_frame(zero, config, PipelineState(housing_temp_k=300.0))
    out_without = run_frame(without, config, PipelineState(housing_temp_k=300.0))
    assert np.array_equal(out_zero.flux, out_without.flux)
