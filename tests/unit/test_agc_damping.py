"""SC.10: the AGC's transfer carries over from frame to frame, as a real core's does (ADR 0170).

A camera core runs an IIR on its AGC transfer -- ``T_shown(t) = a T_shown(t-1) + (1-a) T_frame(t)``
with ``a`` the Boson's Damping Factor / 100 or the Lepton's HEQ Dampening Factor / 256 -- so a hot
target entering the field brightens or darkens the rest of the picture over a few frames. Every
operator here was a pure per-frame function, which gave that transient as a one-frame step.

These are identities, not tolerances: the transfer is linear in its history, so the displayed
value of an unchanged background pixel after ``k`` frames of the new scene is exactly
``a^k · old + (1 - a^k) · new``. What is checked:

  * damping off, or damping on for the first frame, is the per-frame operator **bit for bit** in
    every global mode (the hook adopts its first table);
  * a background pixel's displayed value follows the geometric series to 1e-6 in each mode;
  * the coefficient is per *native* frame and scales with the scene time that passed, so a
    six-second time-lapse is undamped and 60 Hz video is damped -- end to end through `run_frame`;
  * the Boson preset carries FLIR's factory Damping 85, the field is hash-neutral at 0, and the
    tiled mode damps its tables tile by tile.
"""

from __future__ import annotations

import copy
import pathlib
from typing import Any

import numpy as np
import pytest
import yaml

from irsim.config.loader import config_hash, load_sensor_config
from irsim.config.sensor import IspSpec, SensorConfig
from irsim.isp.damping import TRANSFER_KEY, TRANSFER_T_KEY, TransferDamper, damping_over
from irsim.isp.display import run_display_branch
from irsim.materials.table import MaterialTable
from irsim.pipeline import PipelineConfig, PipelineState, run_frame
from irsim.radiometry.lut import BandLUT

REPO = pathlib.Path(__file__).resolve().parents[2]
BOSON_PATH = REPO / "configs" / "sensors" / "flir_boson_640_lwir.yaml"
BOSON = yaml.safe_load(BOSON_PATH.read_text())
BITS = 16

#: FLIR's factory Damping Factor, [R51] p. 5, as the per-frame coefficient.
BOSON_DAMPING = 0.85


def _isp(**over: Any) -> IspSpec:
    d = dict(BOSON["sensor"]["isp"])
    d.update(over)
    return IspSpec.model_validate(d)


#: The global modes, each at settings that exercise its own transfer.
MODES = {
    "linear": {"agc": "linear", "clip_percentiles": [0.0, 1.0], "linear_percent": 0.0},
    "plateau": {
        "agc": "plateau_equalization",
        "linear_percent": 0.0,
        "max_gain": 0.0,
        "clip_limit_low": 0.0,
    },
    "equalise": {"agc": "plateau_equalization"},  # the Boson's linear percent and max gain
    "information": {},  # the Boson's own
    # one table per tile, damped as a stack; its SC.25 linear blend (off here) is not damped
    "local": {"agc": "plateau_local", "linear_percent": 0.0, "agc_tiles": [2, 3]},
}


def _scene(hot: bool, seed: int = 3) -> np.ndarray:
    """A 64 x 80 sky ramp at 1000-3000 DN with texture, and optionally a hot 6 x 6 target."""
    rng = np.random.default_rng(seed)
    ramp = np.linspace(1000.0, 3000.0, 64)[:, None] + rng.normal(0.0, 30.0, (64, 80))
    dn = np.clip(ramp, 0, 2**BITS - 1)
    if hot:
        dn[20:26, 30:36] = 12000.0
    return np.asarray(np.rint(dn), dtype=np.uint16)


# --------------------------------------------------------------------------------------------
# The operator, bit for bit and against the series
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("mode", sorted(MODES))
def test_an_adopting_damper_is_the_per_frame_operator_bit_for_bit(mode: str) -> None:
    isp = _isp(**MODES[mode])
    for hot in (False, True):
        dn = _scene(hot)
        plain = run_display_branch(dn, isp, BITS)
        hooked = run_display_branch(dn, isp, BITS, lut_hook=TransferDamper(BOSON_DAMPING))
        np.testing.assert_array_equal(hooked.y, plain.y)
        np.testing.assert_array_equal(hooked.display8, plain.display8)
        # and a zero coefficient never remembers anything, however long it runs
        zero = TransferDamper(0.0)
        run_display_branch(_scene(not hot), isp, BITS, lut_hook=zero)
        np.testing.assert_array_equal(run_display_branch(dn, isp, BITS, lut_hook=zero).y, plain.y)


@pytest.mark.parametrize("mode", ["linear", "plateau", "equalise", "local"])
def test_a_background_pixel_follows_the_geometric_series(mode: str) -> None:
    """y_k = a^k y_old + (1 - a^k) y_new on every pixel the step did not touch, to 1e-6."""
    isp = _isp(**MODES[mode])
    cold, hot = _scene(False), _scene(True)
    y_old = run_display_branch(cold, isp, BITS).y
    y_new = run_display_branch(hot, isp, BITS).y
    background = np.ones(cold.shape, dtype=bool)
    background[20:26, 30:36] = False
    # the step is visible: the hot target re-maps the sky. Pure plateau equalisation moves it
    # least -- an empty stretch of histogram between sky and target gets no codes -- so its bar
    # is lower, and the identity is still checked at 1e-6 against a step 4000 times that size.
    assert np.max(np.abs(y_new - y_old)[background]) > (0.004 if mode == "plateau" else 0.02)

    damper = TransferDamper(BOSON_DAMPING)
    run_display_branch(cold, isp, BITS, lut_hook=damper)
    for k in range(1, 25):
        y = run_display_branch(hot, isp, BITS, lut_hook=damper).y
        w = BOSON_DAMPING**k
        expected = w * y_old + (1.0 - w) * y_new
        np.testing.assert_allclose(y[background], expected[background], atol=1e-6)


def test_the_information_mode_s_base_layer_is_damped_and_its_detail_follows() -> None:
    """The Boson's own mode: the transfer is damped and the detail slope is taken from it."""
    isp = _isp()
    cold, hot = _scene(False), _scene(True)
    y_old = run_display_branch(cold, isp, BITS).y
    y_new = run_display_branch(hot, isp, BITS).y
    damper = TransferDamper(BOSON_DAMPING)
    run_display_branch(cold, isp, BITS, lut_hook=damper)
    step = float(np.mean(np.abs(y_new - y_old)))
    assert step > 0.02
    left = []
    for _ in range(40):
        y = run_display_branch(hot, isp, BITS, lut_hook=damper).y
        left.append(float(np.mean(np.abs(y - y_new))) / step)
    # the first frame shows 15 % of the step, not all of it, and the residual falls roughly as
    # 0.85 per frame (not exactly: the detail layer rides on the damped slope)
    assert 0.75 < left[0] < 0.95
    assert left[5] < 0.6 * left[0]
    assert left[-1] < 0.01


# --------------------------------------------------------------------------------------------
# The coefficient over an interval
# --------------------------------------------------------------------------------------------


def test_the_coefficient_is_per_native_frame_and_composes() -> None:
    assert damping_over(BOSON_DAMPING, 60.0, 1.0 / 60.0) == pytest.approx(BOSON_DAMPING)
    # a six-second time-lapse at 60 Hz: 360 filter steps have passed, nothing is left
    assert damping_over(BOSON_DAMPING, 60.0, 6.0) < 1e-20
    a = damping_over(BOSON_DAMPING, 60.0, 0.013)
    b = damping_over(BOSON_DAMPING, 60.0, 0.021)
    assert a * b == pytest.approx(damping_over(BOSON_DAMPING, 60.0, 0.034), rel=1e-12)
    assert damping_over(1.0, 60.0, 100.0) == 1.0  # df = 100: frozen
    assert damping_over(0.0, 60.0, 1e-4) == 0.0
    for bad in ((1.2, 60.0, 0.1), (0.5, 0.0, 0.1), (0.5, 60.0, 0.0)):
        with pytest.raises(ValueError):
            damping_over(*bad)


# --------------------------------------------------------------------------------------------
# Through `run_frame`
# --------------------------------------------------------------------------------------------

SHAPE = (24, 32)


def _config(lut: BandLUT, damping: float) -> PipelineConfig:
    d = copy.deepcopy(BOSON)
    d["sensor"]["fpa"].update(width=SHAPE[1], height=SHAPE[0])
    d["sensor"]["optics"].update(supersample_factor=1)
    d["sensor"]["isp"].update(agc_damping=damping)
    return PipelineConfig.from_sensor(
        SensorConfig.model_validate(d),
        MaterialTable.constant(1.0),
        lut=lut,
        noise_enabled=False,
        sensor_seed=20260929,
    )


def _planes(hot: bool) -> dict[str, np.ndarray]:
    t = np.linspace(290.0, 300.0, SHAPE[0], dtype=np.float32)[:, None] * np.ones(
        SHAPE, dtype=np.float32
    )
    if hot:
        t[8:12, 12:16] = 340.0
    return {
        "temperature_k": np.asarray(t, dtype=np.float32),
        "material_id": np.ones(SHAPE, dtype=np.int32),
        "distance_m": np.zeros(SHAPE, dtype=np.float32),
    }


def _sequence(config: PipelineConfig, dt_s: float, n_after: int) -> list[np.ndarray]:
    state = PipelineState(housing_temp_k=config.t_housing_cal_k)
    out = []
    for k in range(4 + n_after):
        state.t_s = k * dt_s
        frame = run_frame(_planes(hot=k >= 4), config, state)
        assert frame.display8 is not None
        out.append(np.asarray(frame.display8)[..., 0].astype(np.int64))
        if config.sensor.sensor.isp.agc_damping > 0.0:
            assert state.buffers[TRANSFER_T_KEY] == pytest.approx(state.t_s)
            assert state.buffers[TRANSFER_KEY].dtype == np.float64
    return out[4:]


def test_video_is_damped_and_a_time_lapse_is_not(tophat_lwir_lut: BandLUT) -> None:
    undamped = _sequence(_config(tophat_lwir_lut, 0.0), 1.0 / 60.0, 60)
    background = np.ones(SHAPE, dtype=bool)
    background[8:12, 12:16] = False

    video = _sequence(_config(tophat_lwir_lut, BOSON_DAMPING), 1.0 / 60.0, 60)
    first_gap = np.abs(video[0] - undamped[0])[background].mean()
    assert first_gap > 5.0, first_gap  # the first frame after the step still shows the old map
    last_gap = np.abs(video[-1] - undamped[-1])[background].max()
    assert last_gap <= 1, last_gap  # 0.85^60 = 6e-5: settled to a code

    lapse = _sequence(_config(tophat_lwir_lut, BOSON_DAMPING), 6.0, 3)
    lapse_ref = _sequence(_config(tophat_lwir_lut, 0.0), 6.0, 3)
    for got, ref in zip(lapse, lapse_ref, strict=True):
        np.testing.assert_array_equal(got, ref)


# --------------------------------------------------------------------------------------------
# The config
# --------------------------------------------------------------------------------------------


def test_the_boson_preset_carries_flir_s_factory_damping() -> None:
    assert load_sensor_config(BOSON_PATH).sensor.isp.agc_damping == BOSON_DAMPING


def test_damping_at_zero_is_hash_neutral() -> None:
    d = copy.deepcopy(BOSON)
    d["sensor"]["isp"].pop("agc_damping", None)
    without = SensorConfig.model_validate(d)
    d["sensor"]["isp"]["agc_damping"] = 0.0
    assert config_hash(SensorConfig.model_validate(d)) == config_hash(without)


def test_the_tiled_mode_damps_each_tile_on_its_own_history() -> None:
    """A tiled core's transfer is a table per tile: the damper keeps the whole stack."""
    isp = _isp(**MODES["local"])
    damper = TransferDamper(BOSON_DAMPING)
    run_display_branch(_scene(False), isp, BITS, lut_hook=damper)
    assert damper.previous is not None
    assert damper.previous.shape == (2, 3, 2**BITS)
