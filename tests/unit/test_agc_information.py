"""SC.21 — information-based equalisation and Linear Percent (§11.3, ADR 0147, spec issue S52).

The defect this step answers: on a sky of cloud clutter every 16-bit bin is occupied and none is
above ``plateau · N``, so global plateau equalisation is full histogram equalisation and a small
target gets the share of the ramp it has of the frame. The first test reproduces it on a synthetic
frame shaped like `phantom4_perpart` (a 60 K sky and a target at 0.6 % of the pixels) and holds the
operator a Boson ships with to a floor it must clear. The rest pin the identities that tie the new
operator to the old ones, so a regression in either shows up as a broken identity rather than as a
picture someone has to look at.
"""

from __future__ import annotations

import numpy as np
import pytest

from irsim.config.sensor import IspSpec
from irsim.isp.agc import agc_plateau
from irsim.isp.display import isp_config_hash, run_display_branch
from irsim.isp.information import agc_information, bilateral_split, blend_linear

#: The Boson's DN per kelvin near 300 K, measured on `phantom4_perpart` (≈ 171-173 DN/K).
DN_PER_K = 170.0
#: A range sigma small enough that every off-centre bilateral weight underflows to exactly zero
#: for a one-DN difference: exp(-0.5 · (1 / 1e-3)²) = 0.0 in float64.
EXACT_SPLIT_SIGMA = 1e-3
FRAME = (512, 640)


def _sky_with_target(seed: int = 0) -> tuple[np.ndarray, list[np.ndarray]]:
    """A smooth −47 … +14 °C sky and a 44×44 target (0.59 % of the frame) in four parts."""
    rng = np.random.default_rng(seed)
    height, width = FRAME
    yy, xx = np.mgrid[0:height, 0:width]
    t_c = -47.0 + 61.0 * (0.5 + 0.5 * np.sin(xx / 90.0) * np.cos(yy / 70.0))
    t_c = t_c + rng.normal(0.0, 0.05, t_c.shape)  # 50 mK of temporal noise
    parts = []
    for r0, c0, part_c in ((234, 298, 15.0), (234, 320, 20.0), (256, 298, 30.0), (256, 320, 38.0)):
        mask = np.zeros(FRAME, dtype=bool)
        mask[r0 : r0 + 22, c0 : c0 + 22] = True
        t_c[mask] = part_c + rng.normal(0.0, 0.05, int(mask.sum()))
        parts.append(mask)
    dn = np.clip(np.round((t_c + 60.0) * DN_PER_K), 0, 2**16 - 1).astype(np.uint16)
    return dn, parts


def _codes(y: np.ndarray) -> np.ndarray:
    return np.round(np.asarray(y, dtype=np.float64) * 255.0)


# --- the defect, and the floor the fix has to clear -----------------------------------------------


def test_global_plateau_starves_a_small_target_on_a_cluttered_sky() -> None:
    """Red half of S52: the four parts, 15 → 38 °C, share at most three grey codes."""
    dn, parts = _sky_with_target()
    target = np.logical_or.reduce(parts)
    assert 0.005 < target.mean() < 0.007
    for plateau in (0.012, 0.07):  # this repository's Boson value and FLIR's default
        codes = _codes(agc_plateau(dn, plateau, 16))[target]
        assert codes.max() - codes.min() <= 3


def test_information_based_with_linear_percent_spreads_the_target_and_keeps_its_order() -> None:
    """Green half: ≥ 25 codes across the target, and hotter parts are brighter, part by part."""
    dn, parts = _sky_with_target()
    target = np.logical_or.reduce(parts)
    codes = _codes(agc_information(dn, 0.07, 16, linear_percent=0.3))
    assert codes[target].max() - codes[target].min() >= 25
    medians = [float(np.median(codes[p])) for p in parts]  # parts are listed coldest first
    assert all(a < b for a, b in zip(medians, medians[1:], strict=False)), medians


def test_the_information_histogram_alone_widens_the_target() -> None:
    """Without Linear Percent the detail weighting on its own still buys the target ≥ 5× the
    codes the plateau gave it -- the part of the fix that is FLIR's IBE rather than the blend."""
    dn, parts = _sky_with_target()
    target = np.logical_or.reduce(parts)
    plateau_codes = _codes(agc_plateau(dn, 0.07, 16))[target]
    info_codes = _codes(agc_information(dn, 0.07, 16))[target]
    plateau_span = max(plateau_codes.max() - plateau_codes.min(), 1.0)
    assert info_codes.max() - info_codes.min() >= 5.0 * plateau_span


# --- identities -----------------------------------------------------------------------------------


def _random_dn(seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(1000, 9000, size=(64, 80)).astype(np.uint16)


def test_exact_split_no_information_no_blend_is_plateau_bit_for_bit() -> None:
    dn = _random_dn()
    got = agc_information(
        dn, 0.012, 14, info_weight=0.0, detail_gain=1.0, smoothing_sigma_dn=EXACT_SPLIT_SIGMA
    )
    np.testing.assert_array_equal(got, agc_plateau(dn, 0.012, 14))
    assert got.dtype == np.float32


def test_linear_percent_one_is_the_min_max_linear_map() -> None:
    dn = _random_dn()
    got = agc_information(
        dn, 0.012, 14, info_weight=0.0, linear_percent=1.0, smoothing_sigma_dn=EXACT_SPLIT_SIGMA
    )
    lin = (dn.astype(np.float64) - dn.min()) / (dn.max() - dn.min())
    np.testing.assert_allclose(got, lin, atol=0.5 / 255.0)


def test_blend_linear_at_zero_is_the_identity_bit_for_bit() -> None:
    dn = _random_dn()
    y = agc_plateau(dn, 0.012, 14)
    np.testing.assert_array_equal(blend_linear(y, dn, 0.0, 14), y)


def test_detail_headroom_reserves_both_ends() -> None:
    dn, _ = _sky_with_target()
    y = agc_information(dn, 0.07, 16, detail_headroom=0.1, detail_gain=0.0)
    assert float(y.min()) == pytest.approx(0.1, abs=1e-6)
    assert float(y.max()) == pytest.approx(0.9, abs=1e-6)


# --- the split ------------------------------------------------------------------------------------


def test_a_step_far_above_the_range_sigma_stays_in_the_low_pass() -> None:
    """A drone-against-sky edge (30 K ≈ 5100 DN, 4.1 σ_r at σ_r = 1250 DN): the cross-edge weight
    is exp(−8.3), so the high-pass at the edge is below 2e-4 of the step (≈ 1 DN of 5100) and the
    edge is equalised with the low-pass rather than boosted as detail."""
    dn = np.full((32, 32), 3000.0)
    dn[:, 16:] = 3000.0 + 30.0 * DN_PER_K
    low, high = bilateral_split(dn, 1250.0)
    assert float(np.abs(high).max()) < 2e-4 * 30.0 * DN_PER_K
    np.testing.assert_allclose(low + high, dn, rtol=0, atol=1e-9)


def test_texture_far_below_the_range_sigma_goes_to_the_high_pass() -> None:
    """A ±5 DN checkerboard (30 mK) with σ_r = 1250 DN is almost entirely high-pass."""
    yy, xx = np.mgrid[0:32, 0:32]
    dn = 5000.0 + 5.0 * np.where((yy + xx) % 2 == 0, 1.0, -1.0)
    low, high = bilateral_split(dn, 1250.0)
    assert float(np.abs(low - 5000.0)[4:-4, 4:-4].max()) < 1.0
    assert float(np.abs(high)[4:-4, 4:-4].min()) > 4.0


# --- refusals and wiring --------------------------------------------------------------------------


def test_float16_is_refused() -> None:
    with pytest.raises(TypeError, match="float16"):
        agc_information(np.zeros((8, 8), dtype=np.float16), 0.07, 16)


def _isp(**overrides: object) -> IspSpec:
    base: dict[str, object] = {
        "agc": "plateau_equalization",
        "plateau": 0.012,
        "clip_percentiles": (0.005, 0.995),
        "gamma": 1.0,
        "dde_gain": 0.35,
        "polarity": "white_hot",
        "palette": "gray",
    }
    base.update(overrides)
    return IspSpec.model_validate(base)


#: The `isp_hash` written into every `phantom4_perpart` frame sidecar, rendered with the Boson's
#: `isp` block before SC.21 existed. A hash recorded by a real run, not recomputed by the test.
PRE_SC21_BOSON_ISP_HASH = "bc94c25ef9fe1f3b5d3e14ca953d9aa71a5af4fd08eec898d4e5c92a50ec0c4e"


def test_a_config_written_before_sc21_keeps_its_isp_hash() -> None:
    """Every SC.21/SC.25 field at its default is left out of the hash, so the Boson's ISP hashes
    exactly as it did in a render made before the fields existed; any one set away from its
    default moves it."""
    assert isp_config_hash(_isp(), 16) == PRE_SC21_BOSON_ISP_HASH
    for field, value in (
        ("linear_percent", 0.3),
        ("info_weight", 2.0),
        ("clip_limit_low", 0.001),
        ("max_gain", 1.25),
    ):
        assert isp_config_hash(_isp(**{field: value}), 16) != PRE_SC21_BOSON_ISP_HASH, field


def test_the_display_branch_runs_information_based_to_rgba8() -> None:
    dn, parts = _sky_with_target()
    out = run_display_branch(
        dn, _isp(agc="information_based", plateau=0.07, linear_percent=0.3), 16
    )
    assert out.display8.dtype == np.uint8 and out.display8.shape == (*FRAME, 4)
    target = np.logical_or.reduce(parts)
    codes = out.display8[..., 0][target].astype(int)
    assert codes.max() - codes.min() >= 25
