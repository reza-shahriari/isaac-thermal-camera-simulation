"""EV.23: the bowl a real tracking camera's clear sky shows, from a housing that drifted since its
shutter and a lens that vignettes.

Real Anti-UAV RGBT clear-sky frames are dark-centred in 71 % of cases and the bowl is 3.3 times
the sky's own vertical gradient; irsim's clips were rendered with a settled housing and a lens
whose only fall-off was cos^4 (0.98 at the corner of a 50 mm lens), so they had none. The
mechanism was already modelled -- an off-axis pixel sees ``1 - tau RI`` of housing (ADR 0145), and
the shutter only refreshes the offset (ADR 0148) -- and these tests hold the two knobs that feed
it: :func:`irsim.optics.vignetting.radial_vignetting_map` and ``SensorChain``'s housing start.
"""

from __future__ import annotations

import argparse
import importlib.util
import pathlib

import numpy as np
import pytest

from irsim.config.loader import config_hash, load_sensor_config, with_vignetting_map
from irsim.optics.self_emission import housing_power_field
from irsim.optics.vignetting import radial_vignetting_map
from irsim.pipeline.sensor_chain import SensorChain

REPO = pathlib.Path(__file__).resolve().parents[2]
TELEPHOTO = REPO / "configs" / "sensors" / "example_lwir_640_telephoto.yaml"


def test_the_map_is_one_on_axis_and_the_corner_value_at_the_corner() -> None:
    m = radial_vignetting_map(640, 512, 0.7)
    assert m.dtype == np.float32 and m.shape == (512, 640)
    assert float(m[256, 320]) == pytest.approx(1.0, abs=1e-5)
    assert float(m[0, 0]) == pytest.approx(0.7, abs=2e-3)
    assert np.all(np.diff(m[256, 320:]) < 0.0)  # falls monotonically outward
    with pytest.raises(ValueError):
        radial_vignetting_map(640, 512, 0.0)


@pytest.mark.parametrize("corner", [0.9, 0.75, 0.6])
def test_a_housing_drift_leaves_a_bowl_in_proportion_to_the_vignetting(corner: float) -> None:
    """The change in non-scene power between two housing radiances, corner minus centre, is
    ``Ω τ (1 - RI_corner) ΔL``: the bowl a flat field taken at the first leaves at the second."""
    ri = radial_vignetting_map(64, 48, corner).astype(np.float64)
    before = housing_power_field(1.0, 1.0, 0.92, 10.0, ri)
    after = housing_power_field(1.0, 1.0, 0.92, 11.0, ri)
    bowl = after - before
    omega = np.pi / (4.0 * 1.0**2 + 1.0)
    expected = omega * 0.92 * (1.0 - float(ri[0, 0])) * 1.0
    assert float(bowl[0, 0] - bowl[24, 32]) == pytest.approx(expected, rel=0.02)


def test_two_maps_are_two_cameras(tmp_path: pathlib.Path) -> None:
    sensor = load_sensor_config(TELEPHOTO)
    hashes = {config_hash(sensor)}
    for corner in (0.6, 0.8):
        path = tmp_path / f"v{corner}.npy"
        np.save(path, radial_vignetting_map(640, 512, corner))
        hashes.add(config_hash(with_vignetting_map(sensor, path)))
    assert len(hashes) == 3


def _chain(offset: float) -> SensorChain:
    sensor = load_sensor_config(TELEPHOTO).sensor
    return SensorChain.build(
        sensor, dn_per_k=30.0, sensor_seed=1, ambient_provider=lambda t: 290.0, t0_s=0.0,
        defects_enabled=False, residual_enabled=False, housing_start_offset_k=offset,
    )  # fmt: skip


def test_a_camera_just_switched_on_warms_toward_its_settled_housing() -> None:
    """Start 4 K below settled (290 K air + 4 K self-heating): the housing begins at 290 K and
    closes on 294 K with the configured 900 s lag -- 1 - 1/e of the way after one tau."""
    housing = _chain(-4.0).housing
    assert housing.temperature_k == pytest.approx(290.0)
    assert housing.at(900.0) == pytest.approx(294.0 - 4.0 * np.exp(-1.0), abs=0.05)
    assert _chain(0.0).housing.temperature_k == pytest.approx(294.0)


def test_an_offset_on_a_housing_with_no_state_is_refused() -> None:
    doc = load_sensor_config(TELEPHOTO).model_dump(mode="json")
    doc["sensor"]["optics"]["housing_temp_mode"] = "ambient"
    from irsim.config.sensor import SensorConfig

    sensor = SensorConfig.model_validate(doc).sensor
    with pytest.raises(ValueError, match="coupled"):
        SensorChain.build(
            sensor, dn_per_k=30.0, sensor_seed=1, ambient_provider=lambda t: 290.0,
            housing_start_offset_k=-2.0,
        )  # fmt: skip


def _planner():  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location(
        "generate_aerial_dataset", REPO / "scripts" / "generate_aerial_dataset.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_planner_draws_a_camera_unit_per_clip_and_nothing_else_moves() -> None:
    """60 clips: housing starts in -8..+3 K (about three quarters warming), corners in 0.6-0.9;
    fixing either leaves weather, hours, focus and seeds as drawn."""
    planner = _planner()
    args = argparse.Namespace(
        seed=20261004, clear_runs=50, cloud_runs=10, frames_per_run=300, cloud_frames_per_run=50,
        near_m=20.0, far_m=90.0, focus="drawn", scene="configs/scenes/phantom4_pointwise.yaml",
    )  # fmt: skip
    runs = planner.plan_runs(args)
    housing = np.array([r["housing_start_k"] for r in runs])
    corner = np.array([r["corner_illumination"] for r in runs])
    assert housing.min() >= -8.0 and housing.max() <= 3.0
    assert 0.6 < float((housing < 0.0).mean()) < 0.9
    assert corner.min() >= 0.6 and corner.max() <= 0.9
    fixed = planner.plan_runs(
        argparse.Namespace(**vars(args), housing_start="0", corner_illumination="sensor")
    )
    assert all(r["housing_start_k"] == 0.0 and r["corner_illumination"] is None for r in fixed)
    strip = ("housing_start_k", "corner_illumination")
    assert [{k: v for k, v in r.items() if k not in strip} for r in runs] == [
        {k: v for k, v in r.items() if k not in strip} for r in fixed
    ]
