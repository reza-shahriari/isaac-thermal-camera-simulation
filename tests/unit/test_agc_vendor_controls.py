"""SC.25 — the equalisation controls every thermal core has under some name (§11.3, ADR 0149).

`plateau_equalization` used to have one knob, the high clip. Real cores expose a small, shared set
of controls around the same histogram: the Lepton family's *low clip limit* (a floor of shades for
every occupied bin), and the *max gain* that FLIR's Boson and Xenics' SWIR cores both document (a
cap on how far a bland scene may be stretched). Each test below builds the scene the vendor
documentation describes the control *for*, and asserts that the control does that job, so the test
fails if the control is wired but does nothing, or does the wrong thing.
"""

from __future__ import annotations

import numpy as np
import pytest

from irsim.config.sensor import IspSpec
from irsim.isp.agc import agc_plateau, plateau_lut
from irsim.isp.display import run_display_branch
from irsim.isp.information import equalise_image, limit_gain

#: The Boson's DN per kelvin near 300 K, measured on `phantom4_perpart`.
DN_PER_K = 170.0
FRAME = (512, 640)


def _codes(y: np.ndarray) -> np.ndarray:
    return np.round(np.asarray(y, dtype=np.float64) * 255.0)


def _clear_sky_with_target() -> tuple[np.ndarray, list[np.ndarray]]:
    """A clear sky ~30 DN wide (σ 4 DN, ≈ 25 mK) and a 0.6 % four-part target 50-73 K above it."""
    rng = np.random.default_rng(4)
    dn = 20000.0 + rng.normal(0.0, 4.0, FRAME)
    parts = []
    for r0, c0, above_k in ((234, 298, 50.0), (234, 320, 55.0), (256, 298, 65.0), (256, 320, 73.0)):
        mask = np.zeros(FRAME, dtype=bool)
        mask[r0 : r0 + 22, c0 : c0 + 22] = True
        dn[mask] = 20000.0 + above_k * DN_PER_K + rng.normal(0.0, 4.0, int(mask.sum()))
        parts.append(mask)
    return np.round(dn).astype(np.uint16), parts


def _bland_sky() -> np.ndarray:
    """Nothing but a clear sky: 33 DN of noise, the scene max gain exists for."""
    rng = np.random.default_rng(3)
    return np.round(20000.0 + rng.normal(0.0, 4.0, (256, 320))).astype(np.uint16)


# --- the low clip limit (Lepton family) -----------------------------------------------------------


def test_the_low_clip_gives_a_small_target_on_a_clear_sky_its_shades_back() -> None:
    """A clear sky's ~30 bins hold 99.4 % of the pixels and none reaches a 7 % plateau, so plain
    equalisation leaves the target ≤ 3 codes. A low clip of 1e-3·N per occupied bin gives each of
    the target's ~120 occupied bins a floor: ≥ 20 codes, parts in temperature order."""
    dn, parts = _clear_sky_with_target()
    target = np.logical_or.reduce(parts)
    plain = _codes(agc_plateau(dn, 0.07, 16))[target]
    assert plain.max() - plain.min() <= 3
    codes = _codes(equalise_image(dn, 0.07, 16, clip_limit_low=1e-3))
    assert codes[target].max() - codes[target].min() >= 20
    medians = [float(np.median(codes[p])) for p in parts]
    assert all(a < b for a, b in zip(medians, medians[1:], strict=False)), medians


def test_a_zero_low_clip_is_the_old_table_bit_for_bit() -> None:
    dn, _ = _clear_sky_with_target()
    counts = np.bincount(dn.ravel(), minlength=2**16).astype(np.float64)
    np.testing.assert_array_equal(plateau_lut(counts, 0.07, 0.0), plateau_lut(counts, 0.07))


# --- max gain (Boson, Xenics) ---------------------------------------------------------------------


def test_max_gain_stops_a_bland_sky_being_stretched_across_its_noise() -> None:
    """Uncapped, 33 DN of sky noise fills all 256 codes; capped at 1.25 codes/DN it spans no more
    than 1.25 × 33 + 2 codes and sits on mid-grey."""
    dn = _bland_sky()
    wide = _codes(equalise_image(dn, 0.07, 16))
    assert wide.min() == 0.0 and wide.max() == 255.0
    capped = _codes(equalise_image(dn, 0.07, 16, max_gain=1.25))
    dn_range = float(dn.max()) - float(dn.min())
    assert capped.max() - capped.min() <= 1.25 * dn_range + 2.0
    assert abs(float(capped.mean()) - 127.5) < 3.0


def test_max_gain_that_never_binds_changes_nothing() -> None:
    """A scene spanning ~13 000 DN never asks for more than a fraction of a code per DN."""
    rng = np.random.default_rng(5)
    dn = rng.integers(1000, 14000, size=(128, 160)).astype(np.uint16)
    np.testing.assert_array_equal(
        equalise_image(dn, 0.07, 16, max_gain=1.25), agc_plateau(dn, 0.07, 16)
    )


def test_limit_gain_refuses_a_negative_cap() -> None:
    with pytest.raises(ValueError, match="max_gain"):
        limit_gain(np.linspace(0.0, 1.0, 16), np.ones(16), -1.0)


# --- wiring ---------------------------------------------------------------------------------------


def _isp(**overrides: object) -> IspSpec:
    base: dict[str, object] = {
        "agc": "plateau_equalization",
        "plateau": 0.07,
        "clip_percentiles": (0.005, 0.995),
        "gamma": 1.0,
        "dde_gain": 0.0,
        "polarity": "white_hot",
        "palette": "gray",
    }
    base.update(overrides)
    return IspSpec.model_validate(base)


@pytest.mark.parametrize(
    "mode", ["linear", "plateau_equalization", "plateau_local", "information_based", "none"]
)
def test_every_agc_mode_is_selectable_from_the_config(mode: str) -> None:
    """The user-facing selector is `isp.agc`: each mode runs to an RGBA8 frame, and the target is
    never darker than the clear sky around it (every mode is monotone in DN)."""
    dn, parts = _clear_sky_with_target()
    target = np.logical_or.reduce(parts)
    out = run_display_branch(dn, _isp(agc=mode, clip_limit_low=1e-3, max_gain=4.0), 16)
    assert out.display8.dtype == np.uint8 and out.display8.shape == (*FRAME, 4)
    grey = out.display8[..., 0].astype(float)
    assert float(np.median(grey[target])) >= float(np.median(grey[~target]))


def test_the_device_agc_refuses_what_it_has_no_port_of() -> None:
    """The Warp table has only the plain modes. A control it would silently drop is refused,
    before any device is touched, rather than rendered differently from the CPU oracle."""
    from irsim_isaac.pipeline.warp_stages import agc_lut_warp

    for isp in (_isp(max_gain=1.25), _isp(clip_limit_low=1e-3), _isp(agc="information_based")):
        with pytest.raises(NotImplementedError, match="CPU display branch"):
            agc_lut_warp(None, isp, 16, "cpu")
