"""The Boson runs its factory-default AGC (§11.3, spec issue S52, `SC.22`, ADR 0152).

`flir_boson_640_lwir.yaml` selected global plateau equalisation at a plateau FLIR does not ship,
which on a cluttered sky is full histogram equalisation: the drone got 2-3 grey codes. [R51] p. 5
publishes the factory values ("Current Camera Values": Information-Based on, Plateau 7, Linear
Percent 20, Max Gain 1.38, Detail Headroom 12, DDE 0.95, Smoothing Factor 1250). The config now
carries them, and these tests pin three things: that it does, what that buys a small target, and
that the switch touches the display branch only -- the raw DN and the radiometry are bit-identical.

`halmstad_boson_320.yaml` is deliberately not moved: its `linear` is the dataset recorder's
Y16 → 8-bit conversion, not a Boson AGC, and the test says so.
"""

from __future__ import annotations

import copy
import pathlib
from typing import Any

import numpy as np
import pytest
import yaml
from test_agc_information import _sky_with_target  # the phantom4-shaped synthetic frame

from irsim.config.loader import load_sensor_config
from irsim.config.sensor import IspSpec, SensorConfig
from irsim.isp.display import run_display_branch
from irsim.materials import MaterialTable
from irsim.pipeline import PipelineConfig, PipelineState, run_frame
from irsim.radiometry.encoding import encode_temperature
from irsim.radiometry.lut import BandLUT

REPO = pathlib.Path(__file__).resolve().parents[2]
BOSON_640 = REPO / "configs" / "sensors" / "flir_boson_640_lwir.yaml"
HALMSTAD = REPO / "configs" / "sensors" / "halmstad_boson_320.yaml"

#: The pre-SC.22 Boson ISP, so the red half can be reproduced from the config it replaced.
OLD_ISP: dict[str, Any] = {
    "agc": "plateau_equalization",
    "plateau": 0.012,
    "linear_percent": 0.0,
    "max_gain": 0.0,
    "detail_headroom": 0.0,
    "dde_gain": 0.35,
}


def _isp() -> IspSpec:
    return load_sensor_config(BOSON_640).sensor.isp


def _codes(isp: IspSpec, dn: np.ndarray) -> np.ndarray:
    return run_display_branch(dn, isp, 16).display8[..., 0].astype(np.int64)


def test_the_boson_640_carries_flir_s_published_defaults() -> None:
    isp = _isp()
    assert isp.agc == "information_based"
    assert isp.plateau == pytest.approx(0.07)  # Plateau Value 7
    assert isp.linear_percent == pytest.approx(0.20)  # Linear Percent 20
    assert isp.max_gain == pytest.approx(1.38)  # Max Gain 1.38
    assert isp.detail_headroom == pytest.approx(12.0 / 255.0, abs=5e-4)  # Detail Headroom 12
    assert isp.smoothing_sigma_dn == pytest.approx(1250.0)  # Smoothing Factor 1250
    # DDE 0.95 is FLIR's total high-pass gain; irsim's is 1 + dde_gain, so 0 is the nearest legal
    assert isp.dde_gain == 0.0


def test_the_halmstad_320_keeps_its_recorder_s_linear_map() -> None:
    assert load_sensor_config(HALMSTAD).sensor.isp.agc == "linear"


def test_the_default_gives_a_small_target_its_parts_where_the_old_config_gave_it_one_shade() -> (
    None
):
    """The phantom4-shaped synthetic frame: 2 codes → 37 measured, parts in temperature order."""
    dn, parts = _sky_with_target()
    target = np.logical_or.reduce(parts)
    new_isp = _isp()
    old = _codes(new_isp.model_copy(update=OLD_ISP), dn)[target]
    new_all = _codes(new_isp, dn)
    new = new_all[target]

    assert old.max() - old.min() <= 3
    assert new.max() - new.min() >= 30
    medians = [float(np.median(new_all[p])) for p in parts]  # coldest part first
    assert all(a < b for a, b in zip(medians, medians[1:], strict=False)), medians
    # Linear Percent keeps the sky a sky: most of the ramp still goes to it, not all
    sky = new_all[~target]
    assert 150 <= sky.max() - sky.min() < 255


@pytest.mark.slow
def test_the_switch_is_display_only(boson_lut: BandLUT) -> None:
    """dn16, radiance and apparent_t are bit-identical under either ISP; display8 is not."""
    d = copy.deepcopy(yaml.safe_load(BOSON_640.read_text()))
    d["sensor"]["fpa"].update(width=128, height=96)
    d["sensor"]["optics"]["supersample_factor"] = 1
    rng = np.random.default_rng(0)
    t = (250.0 + 40.0 * rng.random((96, 128))).astype(np.float32)
    planes = {
        "temperature_k": t,
        "encoded_t": encode_temperature(t),
        "normal_dot_view": np.ones(t.shape, np.float32),
        "distance_m": np.full(t.shape, 50.0, np.float32),
        "material_id": np.ones(t.shape, np.int32),
        "sky_view_factor": np.full(t.shape, 0.5, np.float32),
    }
    outs = []
    for isp in ({}, OLD_ISP):
        e = copy.deepcopy(d)
        e["sensor"]["isp"].update(isp)
        config = PipelineConfig.from_sensor(
            SensorConfig.model_validate(e),
            MaterialTable.constant(1.0),
            lut=boson_lut,
            sensor_seed=5,
        )
        outs.append(run_frame(planes, config, PipelineState()))
    new, old = outs
    for key in ("dn16", "radiance", "apparent_t"):
        np.testing.assert_array_equal(getattr(new, key), getattr(old, key), err_msg=key)
    assert not np.array_equal(new.display8, old.display8)
