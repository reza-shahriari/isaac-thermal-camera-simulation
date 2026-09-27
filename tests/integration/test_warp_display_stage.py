"""Roadmap M10.8: Warp stage 6 -- device histogram, AGC, DDE and palette against the CPU branch.

Stage 6 is back to being (almost) exactly comparable: unlike stage 5 it draws no random numbers, so
the only differences between the paths are float32 association order and the single 8-bit
quantisation at the end. The roadmap's bar is therefore ±1 display code on ≥ 99.9 % of pixels, and
that is what is measured -- per AGC mode, on scenes chosen so each mode's failure would be visible:

* a ramp, where a broken CDF shows as banding;
* a hot exhaust patch, which is the case §11.3 and ADR 0028 exist for: linear AGC collapses the
  background contrast and plateau equalisation does not, and both behaviours must survive the port;
* a constant frame, the degenerate case where both AGCs must return the mid-grey level rather than
  divide by a zero span.

And the structural check that matters most: **DN16 is identical under either AGC**. The radiometric
and display branches fork after the ADC (§11.1), so a change of AGC mode must not move a single DN
code. An AGC that reached back into the linear output would be invisible in the picture and fatal
to the validation that consumes it.

Nothing here renders or needs Kit (ADR 0014 addendum).

    $PYTHON -m pytest tests/integration/test_warp_display_stage.py -m gpu
"""

from __future__ import annotations

import copy
import pathlib
from typing import Any

import numpy as np
import pytest
import yaml

REPO = pathlib.Path(__file__).resolve().parents[2]
BOSON_YAML = REPO / "configs" / "sensors" / "flir_boson_640_lwir.yaml"
DEVICES = ("cuda:0", "cpu")
SHAPE = (96, 128)
CODE_TOL = 1
PIXEL_FRACTION = 0.999
pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module")
def warp() -> Any:
    from irsim_isaac.env import ensure_warp_on_path

    ensure_warp_on_path()
    import warp as wp

    wp.init()
    if not any(d.is_cuda for d in wp.get_devices()):
        pytest.skip("no CUDA device for the cuda:0 half of the harness")
    return wp


def _config(lut, **isp: Any):  # type: ignore[no-untyped-def]
    """A Boson with the ISP block overridden; the LUT comes from the shared conftest fixture."""
    from irsim.config.sensor import SensorConfig
    from irsim.materials.table import MaterialTable
    from irsim.pipeline import PipelineConfig

    d = copy.deepcopy(yaml.safe_load(BOSON_YAML.read_text()))
    d["sensor"]["fpa"].update(width=SHAPE[1], height=SHAPE[0])
    d["sensor"]["optics"].update(supersample_factor=1)
    # The device AGC ports the table-shaped operators only (ADR 0147, 0149); since SC.22 the Boson
    # ships `information_based` with Linear Percent, Max Gain and Detail Headroom, which it
    # refuses. Pin the pre-SC.22 plateau ISP so these tests keep measuring the port they cover.
    d["sensor"]["isp"].update(
        agc="plateau_equalization",
        plateau=0.012,
        linear_percent=0.0,
        max_gain=0.0,
        detail_headroom=0.0,
        dde_gain=0.35,
    )
    d["sensor"]["isp"].update(isp)
    sensor = SensorConfig.model_validate(d)
    return PipelineConfig.from_sensor(sensor, MaterialTable.constant(1.0), lut=lut)


# -- scenes ------------------------------------------------------------------------------------


def _ramp_dn() -> np.ndarray:
    rows, cols = SHAPE
    ramp = np.linspace(2_000, 60_000, cols)
    return np.tile(ramp, (rows, 1)).astype(np.uint16)


def _exhaust_dn() -> np.ndarray:
    """A 5 % very hot patch on a low-contrast background: ADR 0028's collapse case."""
    rows, cols = SHAPE
    rng = np.random.default_rng(3)
    dn = (20_000 + rng.normal(0, 300, SHAPE)).astype(np.float64)
    h = max(1, int(round(rows * 0.22)))
    w = max(1, int(round(cols * 0.22)))
    dn[:h, :w] = 64_000
    return np.clip(dn, 0, 65_535).astype(np.uint16)


def _constant_dn() -> np.ndarray:
    return np.full(SHAPE, 30_000, dtype=np.uint16)


SCENES = {"ramp": _ramp_dn, "exhaust": _exhaust_dn, "constant": _constant_dn}
AGC_MODES = ("linear", "plateau_equalization", "none")


def _codes_within(a: np.ndarray, b: np.ndarray, tol: int) -> float:
    """Fraction of pixels whose display codes agree within ``tol``."""
    diff = np.abs(a.astype(np.int32) - b.astype(np.int32))
    return float(np.count_nonzero(diff <= tol) / diff.size)


# -- the comparison ------------------------------------------------------------------------------


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("agc", AGC_MODES)
@pytest.mark.parametrize("scene", sorted(SCENES))
def test_display_matches_the_cpu_branch_within_one_code(
    warp: Any, tophat_lwir_lut, device: str, agc: str, scene: str
) -> None:  # type: ignore[no-untyped-def]
    """≥ 99.9 % of pixels within ±1 display code, for every AGC mode on every scene."""
    from irsim.isp.display import run_display_branch
    from irsim.pipeline import PipelineState
    from irsim_isaac.pipeline.warp_stages import display_stage_warp

    config = _config(tophat_lwir_lut, agc=agc)
    dn = SCENES[scene]()
    bit_depth = config.sensor.sensor.fpa.bit_depth

    cpu = run_display_branch(dn, config.sensor.sensor.isp, bit_depth)
    gpu = display_stage_warp({"dn16": dn}, config, PipelineState(), device=device)

    fraction = _codes_within(gpu["display8"], cpu.display8, CODE_TOL)
    assert fraction >= PIXEL_FRACTION, f"{fraction:.5f} of pixels within +/-{CODE_TOL} code"
    # The float image before quantisation agrees far more tightly than one code.
    assert np.allclose(gpu["y"], cpu.y, atol=2e-3)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("agc", AGC_MODES)
def test_dn16_is_identical_under_either_agc(
    warp: Any, tophat_lwir_lut, device: str, agc: str
) -> None:  # type: ignore[no-untyped-def]
    """§11.1 forks the branches after the ADC: the AGC must not move a single DN code.

    An AGC that reached back into the linear output would be invisible in the picture and fatal
    to the validation that consumes it, which is why this is asserted rather than assumed.
    """
    from irsim.pipeline import PipelineState
    from irsim_isaac.pipeline.warp_stages import display_stage_warp

    dn = _exhaust_dn()
    out = display_stage_warp(
        {"dn16": dn}, _config(tophat_lwir_lut, agc=agc), PipelineState(), device=device
    )
    assert np.array_equal(out["dn16"], dn)
    assert out["dn16"].dtype == np.uint16


@pytest.mark.parametrize("device", DEVICES)
def test_the_exhaust_collapse_is_reproduced_on_device(
    warp: Any, tophat_lwir_lut, device: str
) -> None:  # type: ignore[no-untyped-def]
    """Linear AGC collapses the background contrast; plateau equalisation retains it (ADR 0028).

    The whole reason §11.3 calls the AGC "part of the sensor model, not a display detail". If the
    port lost this, the device images would look fine and would be useless for the one scene the
    behaviour was modelled for.
    """
    from irsim.pipeline import PipelineState
    from irsim_isaac.pipeline.warp_stages import display_stage_warp

    dn = _exhaust_dn()
    rows, cols = SHAPE
    background = np.ones(SHAPE, dtype=bool)
    background[: max(1, int(round(rows * 0.22))), : max(1, int(round(cols * 0.22)))] = False

    linear = display_stage_warp(
        {"dn16": dn}, _config(tophat_lwir_lut, agc="linear"), PipelineState(), device=device
    )
    plateau = display_stage_warp(
        {"dn16": dn},
        _config(tophat_lwir_lut, agc="plateau_equalization"),
        PipelineState(),
        device=device,
    )

    linear_contrast = float(linear["y"][background].std())
    plateau_contrast = float(plateau["y"][background].std())
    assert plateau_contrast > 3.0 * linear_contrast, (
        f"plateau {plateau_contrast:.4f} vs linear {linear_contrast:.4f}"
    )


@pytest.mark.parametrize("device", DEVICES)
def test_a_constant_frame_gives_the_mid_grey_level(warp: Any, tophat_lwir_lut, device: str) -> None:  # type: ignore[no-untyped-def]
    """The degenerate case: no dynamic range at DN resolution, so no division by a zero span."""
    from irsim.isp.agc import CONSTANT_FRAME_LEVEL
    from irsim.pipeline import PipelineState
    from irsim_isaac.pipeline.warp_stages import display_stage_warp

    dn = _constant_dn()
    for agc in ("linear", "plateau_equalization"):
        out = display_stage_warp(
            {"dn16": dn}, _config(tophat_lwir_lut, agc=agc), PipelineState(), device=device
        )
        assert np.allclose(out["y"], CONSTANT_FRAME_LEVEL, atol=1e-6)


# -- the individual stages -----------------------------------------------------------------------


@pytest.mark.parametrize("device", DEVICES)
def test_the_device_histogram_is_exact(warp: Any, device: str) -> None:
    """Atomics: no lost counts, no double counts, on a plane with heavy bin collisions."""
    import warp as wp

    from irsim_isaac.pipeline.warp_stages import launch_histogram

    rng = np.random.default_rng(7)
    dn = rng.integers(0, 4096, size=(128, 160), dtype=np.uint16)  # many pixels per bin
    src = wp.array2d(np.ascontiguousarray(dn), dtype=wp.uint16, device=device)
    counts = launch_histogram(src, 16, device).numpy()
    expected = np.bincount(dn.ravel(), minlength=counts.size)
    assert np.array_equal(counts, expected)
    assert int(counts.sum()) == dn.size


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("agc", AGC_MODES)
def test_the_agc_table_matches_the_cpu_mapping(
    warp: Any, tophat_lwir_lut, device: str, agc: str
) -> None:  # type: ignore[no-untyped-def]
    """The AGC is a monotone function of DN alone, so it is exactly a table on both paths.

    Comparing the *tables* rather than the images localises a disagreement to the AGC instead of
    leaving it somewhere in the DDE or the palette.
    """
    import warp as wp

    from irsim.isp.agc import agc_linear, agc_plateau
    from irsim.isp.display import agc_none
    from irsim_isaac.pipeline.warp_stages import agc_lut_warp

    config = _config(tophat_lwir_lut, agc=agc)
    isp = config.sensor.sensor.isp
    dn = _exhaust_dn()
    src = wp.array2d(np.ascontiguousarray(dn), dtype=wp.uint16, device=device)
    _, table = agc_lut_warp(src, isp, 16, device)

    if agc == "linear":
        expected = agc_linear(dn, *isp.clip_percentiles, 1.0, 16)
    elif agc == "plateau_equalization":
        expected = agc_plateau(dn, isp.plateau, 16)
    else:
        expected = agc_none(dn, 16)
    if isp.gamma != 1.0:
        expected = np.power(expected, np.float32(1.0 / isp.gamma), dtype=np.float32)

    applied = table[dn]
    assert np.allclose(applied, expected, atol=2e-6), (
        f"max |Δ| = {float(np.abs(applied - expected).max()):.3e}"
    )
    # And the table is monotone, which is what makes it an AGC rather than a scramble.
    assert np.all(np.diff(table) >= -1e-7)


@pytest.mark.parametrize("device", DEVICES)
def test_dde_matches_the_cpu_unsharp_mask(warp: Any, tophat_lwir_lut, device: str) -> None:  # type: ignore[no-untyped-def]
    """Including the edge clamping: zero padding would darken the border like a false vignette."""
    from irsim.isp.dde import dde
    from irsim.pipeline import PipelineState
    from irsim_isaac.pipeline.warp_stages import display_stage_warp

    config = _config(tophat_lwir_lut, agc="none", dde_gain=1.5, gamma=1.0)
    dn = _ramp_dn()
    got = display_stage_warp({"dn16": dn}, config, PipelineState(), device=device)

    from irsim.isp.display import agc_none

    expected = dde(agc_none(dn, 16), 1.5)
    assert np.allclose(got["y"], expected, atol=2e-6)


@pytest.mark.parametrize("device", DEVICES)
def test_black_hot_inverts_and_alpha_is_opaque(warp: Any, tophat_lwir_lut, device: str) -> None:  # type: ignore[no-untyped-def]
    from irsim.pipeline import PipelineState
    from irsim_isaac.pipeline.warp_stages import display_stage_warp

    dn = _ramp_dn()
    white = display_stage_warp(
        {"dn16": dn}, _config(tophat_lwir_lut, polarity="white_hot"), PipelineState(), device=device
    )["display8"]
    black = display_stage_warp(
        {"dn16": dn}, _config(tophat_lwir_lut, polarity="black_hot"), PipelineState(), device=device
    )["display8"]
    assert np.all(white[..., 3] == 255)
    assert np.array_equal(black[..., 0].astype(np.int16), 255 - white[..., 0].astype(np.int16))


@pytest.mark.parametrize("device", DEVICES)
def test_a_float_plane_is_refused(warp: Any, tophat_lwir_lut, device: str) -> None:  # type: ignore[no-untyped-def]
    """Stage 6 consumes the ADC's output; a float plane means the caller skipped the quantiser."""
    from irsim.pipeline import PipelineState
    from irsim_isaac.pipeline.warp_stages import display_stage_warp

    with pytest.raises(TypeError, match="uint16"):
        display_stage_warp(
            {"dn16": np.zeros(SHAPE, dtype=np.float32)},
            _config(tophat_lwir_lut),
            PipelineState(),
            device=device,
        )
