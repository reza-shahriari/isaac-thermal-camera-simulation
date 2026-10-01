"""Glass gets its reststrahlen dip as a curve (XD.14, ADR 0175).

`scripts/derive_proxy_shape_curves.py` writes ε(λ) = 1 − s·R(λ) from the fused-silica proxy table,
with s fixing the LWIR nominal-band average at the authored 0.88. What has to hold: the committed
file is the script's output; the nominal band reads the authored value (so nothing that renders
the standard band moved); the curve has the Si-O dip where silicates have it; and cameras whose
ranges hold more or less of the dip read glass differently, in the right order.

docs/physics-model.md §4.1, §4.4, §12.3; ADR 0041, ADR 0175
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

from irsim.materials.library import MaterialLibrary
from irsim.radiometry.spectral_response import SpectralResponse

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
from derive_proxy_shape_curves import PROXIES, path_for, render  # noqa: E402


def _top_hat(lo: float, hi: float) -> SpectralResponse:
    return SpectralResponse(np.array([lo, hi]), np.array([1.0, 1.0]), f"<{lo}-{hi}>", "")


@pytest.fixture(scope="module")
def glass():  # type: ignore[no-untyped-def]
    return MaterialLibrary.load()["glass_windshield"]


@pytest.mark.parametrize("name", sorted(PROXIES))
def test_the_committed_curve_is_the_script_s_output(name: str) -> None:
    assert path_for(name).read_text(encoding="utf-8") == render(name)


def test_the_nominal_bands_read_what_was_authored(glass) -> None:  # type: ignore[no-untyped-def]
    authored = {"nir": 0.15, "swir": 0.22, "mwir": 0.85, "lwir": 0.88}
    for band, value in authored.items():
        props = glass.band_properties(band)
        assert props.emissivity == pytest.approx(value, abs=1e-6), band
        assert abs(props.emissivity + props.reflectance + props.transmittance - 1.0) < 1e-6
    assert glass.band_properties("lwir").curve_fraction == 1.0
    assert glass.band_properties("mwir").curve_fraction == 0.0


def test_the_curve_has_the_silicate_dip(glass) -> None:  # type: ignore[no-untyped-def]
    """Si-O stretching: reflectance peaks near 9 µm, so ε dips there and recovers either side."""
    lam = np.array([6.0, 9.0, 11.5])
    eps = glass.curve.values(lam)
    assert eps[1] < 0.75
    assert eps[0] > 0.95 and eps[2] > 0.93


def test_cameras_read_glass_by_how_much_of_the_dip_they_hold(glass) -> None:  # type: ignore[no-untyped-def]
    def eps(lo: float, hi: float) -> float:
        return float(glass.band_properties("lwir", _top_hat(lo, hi)).emissivity)

    dip, window, nominal, wide, long = (
        eps(8.0, 9.5),
        eps(8.0, 12.0),
        eps(7.5, 13.5),
        eps(6.0, 13.0),
        eps(10.0, 13.0),
    )
    assert dip < window < nominal < wide < long
    assert nominal - window > 0.02  # ~1.5 K apparent at 300 K: a difference a user would see
