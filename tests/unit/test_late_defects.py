"""SC.19: defects that appear after calibration are not replaced (§10.4).

A camera interpolates over the defect map it was shipped with. A pixel that fails afterwards is
not on it, so it reaches the output as an isolated stuck pixel -- the lone bright dot in a real
clear-sky frame. Before SC.19 the replacement ran on the exact defect mask and no stuck pixel ever
reached the image.
"""

from __future__ import annotations

import copy
import pathlib
from typing import Any

import numpy as np
import pytest
import yaml

from irsim.config.sensor import SensorConfig
from irsim.materials.table import MaterialTable
from irsim.noise.defects import (
    BadPixelMap,
    DefectKind,
    DefectState,
    active_defect_mask,
    generate_map,
    replacement_mask,
)
from irsim.pipeline import PipelineConfig, PipelineState, attach_sensor_chain, run_frame
from irsim.radiometry.lut import BandLUT

REPO = pathlib.Path(__file__).resolve().parents[2]
BOSON = yaml.safe_load((REPO / "configs" / "sensors" / "flir_boson_640_lwir.yaml").read_text())
SEED = 20260926
SHAPE = (64, 80)


def _noise(**over: Any):  # type: ignore[no-untyped-def]
    d = copy.deepcopy(BOSON)
    d["sensor"]["noise"].update(over)
    return SensorConfig.model_validate(d).sensor.noise


# --- the map -------------------------------------------------------------------------------


def test_the_late_fraction_does_not_move_a_single_defect() -> None:
    """Late flags are drawn after everything else, so positions and classes are bit-identical --
    turning the feature on does not silently give the camera a different focal plane."""
    none = generate_map((256, 256), _noise(bad_pixel_late_fraction=0.0), SEED)
    some = generate_map((256, 256), _noise(bad_pixel_late_fraction=0.3), SEED)
    assert np.array_equal(none.kind, some.kind)
    assert none.late is None and not none.late_mask.any()
    assert np.array_equal(none.factory_mask, none.mask)


def test_the_late_share_is_the_configured_fraction() -> None:
    m = generate_map((1024, 1024), _noise(bad_pixel_late_fraction=0.2), SEED)
    share = m.late_mask.sum() / m.count
    # binomial on ~1.5k defects: 3 sigma is ~0.03
    assert share == pytest.approx(0.2, abs=0.03), share
    assert not np.any(m.late_mask & ~m.mask)
    assert np.array_equal(m.factory_mask | m.late_mask, m.mask)


def test_replacement_leaves_out_exactly_the_late_defects() -> None:
    m = generate_map((256, 256), _noise(bad_pixel_late_fraction=0.5), SEED)
    state = DefectState(bad=np.ones(m.shape, dtype=bool))
    active = active_defect_mask(m, state)
    replaced = replacement_mask(m, state)
    assert np.array_equal(replaced, active & ~m.late_mask)
    assert np.any(active & m.late_mask), "the test needs a late active defect to mean anything"


def test_a_late_flag_on_a_good_pixel_is_refused() -> None:
    kind = np.zeros((4, 4), np.uint8)
    kind[1, 1] = int(DefectKind.HOT)
    late = np.zeros((4, 4), bool)
    late[2, 2] = True
    with pytest.raises(ValueError, match="good pixel"):
        BadPixelMap(kind=kind, late=late)


# --- end to end ----------------------------------------------------------------------------


def _config(lut: BandLUT, late: float) -> PipelineConfig:
    d = copy.deepcopy(BOSON)
    d["sensor"]["fpa"].update(width=SHAPE[1], height=SHAPE[0], frame_rate_hz=6.0)
    d["sensor"]["optics"].update(supersample_factor=1)
    d["sensor"]["noise"].update(
        netd_mk_at_300k=1e-6,
        bad_pixel_fraction=0.01,
        bad_pixel_late_fraction=late,
        bad_pixel_type_mix={"dead": 0.5, "hot": 0.5, "flickering": 0.0, "blinking": 0.0},
    )
    cfg = PipelineConfig.from_sensor(
        SensorConfig.model_validate(d), MaterialTable.constant(1.0), lut=lut, sensor_seed=SEED
    )
    return attach_sensor_chain(cfg, ambient_provider=lambda t: 293.15, residual_enabled=False)


def _frame(config: PipelineConfig):  # type: ignore[no-untyped-def]
    planes = {
        "temperature_k": np.full(SHAPE, 300.0, dtype=np.float32),
        "material_id": np.ones(SHAPE, dtype=np.int32),
        "distance_m": np.zeros(SHAPE, dtype=np.float32),
    }
    return run_frame(planes, config, PipelineState())


def test_a_late_hot_pixel_reaches_the_output_and_a_factory_one_does_not(
    tophat_lwir_lut: BandLUT,
) -> None:
    config = _config(tophat_lwir_lut, late=0.5)
    m = config.chain.bad_pixels
    hot = m.mask_of(DefectKind.HOT)
    late_hot, factory_hot = hot & m.late_mask, hot & ~m.late_mask
    assert late_hot.any() and factory_hot.any()

    out = _frame(config)
    dn = np.asarray(out.dn16, dtype=np.float64)
    dn_max = 2**config.sensor.sensor.fpa.bit_depth - 1
    median = float(np.median(dn))
    assert np.all(dn[late_hot] == dn_max), "a late hot pixel must stay stuck at full scale"
    assert np.all(dn[factory_hot] < median + 0.5 * (dn_max - median))

    # ...and in the 8-bit picture the late one is a white dot (unless a neighbour is also late)
    grey = np.asarray(out.display8)[..., 0].astype(np.float64)
    assert float(grey[late_hot].mean()) > float(np.percentile(grey, 99.0))


def test_with_no_late_defects_nothing_stuck_survives(tophat_lwir_lut: BandLUT) -> None:
    """The pre-SC.19 behaviour is still there: fraction 0 means every defect is replaced."""
    config = _config(tophat_lwir_lut, late=0.0)
    m = config.chain.bad_pixels
    dn = np.asarray(_frame(config).dn16)
    dn_max = 2**config.sensor.sensor.fpa.bit_depth - 1
    assert not np.any(dn[m.mask_of(DefectKind.HOT)] == dn_max)
    assert not np.any(dn[m.mask_of(DefectKind.DEAD)] == 0)


def test_no_late_defects_hashes_as_the_pre_v12_camera() -> None:
    """Spelling out `bad_pixel_late_fraction: 0` describes the camera every golden before SC.19
    was written against, so it must hash the same as leaving the key out."""
    from irsim.config.loader import config_hash

    d = copy.deepcopy(BOSON)
    d["sensor"]["noise"].pop("bad_pixel_late_fraction", None)
    implicit = SensorConfig.model_validate(d)
    d["sensor"]["noise"]["bad_pixel_late_fraction"] = 0.0
    assert config_hash(SensorConfig.model_validate(d)) == config_hash(implicit)
    d["sensor"]["noise"]["bad_pixel_late_fraction"] = 0.01
    assert config_hash(SensorConfig.model_validate(d)) != config_hash(implicit)
