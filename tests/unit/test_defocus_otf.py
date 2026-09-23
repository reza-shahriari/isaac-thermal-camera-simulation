"""OC.2 -- the defocus OTF models (docs/physics-model.md §8.3).

The load-bearing test is the first one: Hopkins reduces to the diffraction MTF at zero defocus.
That is what licenses it to *replace* the cascade's first factor rather than multiply onto it, and
it is a property no amount of plausible-looking blur would give you by accident.
"""

from __future__ import annotations

import numpy as np
import pytest

from irsim.optics.defocus import defocus_w020_um
from irsim.optics.mtf import (
    band_average_otf,
    bessel_j1,
    lens_otf,
    mtf_defocus_geometric,
    mtf_defocus_hopkins,
    mtf_diffraction,
)

LWIR_UM, F1 = 10.5, 1.0
XI_NYQUIST = 1.0 / (2.0 * 12e-3)  # 41.667 cyc/mm on a 12 µm pitch
XI = np.linspace(0.0, 95.0, 25)


def test_hopkins_at_zero_defocus_is_the_diffraction_mtf() -> None:
    """The oracle. Hopkins carries diffraction inside it, so it must reproduce the Airy MTF
    exactly when there is nothing to defocus -- checked at the limit, not at the special case."""
    assert np.allclose(mtf_defocus_hopkins(XI, LWIR_UM, F1, 0.0), mtf_diffraction(XI, LWIR_UM, F1))
    # 7.7e-9 is the Simpson floor at _QUAD_NODES, not a physical tolerance: it does not improve
    # as W020 shrinks further, which is how you can tell it is the quadrature and not the model.
    approaching = mtf_defocus_hopkins(XI, LWIR_UM, F1, 1e-4)
    assert np.max(np.abs(approaching - mtf_diffraction(XI, LWIR_UM, F1))) < 1e-8


@pytest.mark.parametrize(
    ("x", "want"),
    [(0.5, 0.2422684577), (1.0, 0.4400505857), (3.0, 0.3390589585), (5.0, -0.3275791376)],
)
def test_the_bessel_quadrature_is_j1(x: float, want: float) -> None:
    assert float(bessel_j1(x)) == pytest.approx(want, abs=1e-9)


def test_defocus_only_ever_costs_contrast() -> None:
    for w020 in (0.5, 1.0, 2.0, 5.0):
        otf = mtf_defocus_hopkins(XI, LWIR_UM, F1, w020)
        assert np.all(np.abs(otf) <= mtf_diffraction(XI, LWIR_UM, F1) + 1e-12)
    assert float(mtf_defocus_hopkins(0.0, LWIR_UM, F1, 3.0)) == pytest.approx(1.0, abs=1e-9)


def _max_model_gap(w020_um: float) -> float:
    """Worst absolute disagreement between Hopkins and diffraction × the geometric disk."""
    xi = np.linspace(1.0, 90.0, 90)
    return float(
        np.max(
            np.abs(
                mtf_defocus_hopkins(xi, LWIR_UM, F1, w020_um)
                - mtf_diffraction(xi, LWIR_UM, F1) * mtf_defocus_geometric(xi, 8.0 * F1 * w020_um)
            )
        )
    )


def test_the_geometric_disk_converges_to_hopkins_only_as_the_defocus_deepens() -> None:
    """Both halves asserted, so swapping one model for the other fails the test.

    Geometric optics is an asymptote, not a threshold: the gap peaks at 0.22 in the transition
    band, is still 0.07 at the W020 = 2λ rule of thumb `geometric_regime_blur_um` encodes, and only
    reaches 0.02 by sixteen waves. Anyone tempted to tighten that rule should read this test.
    """
    gaps = [_max_model_gap(m * LWIR_UM) for m in (0.5, 1.0, 2.0, 4.0, 8.0, 16.0)]
    assert all(b < a for a, b in zip(gaps, gaps[1:], strict=False)), f"not converging: {gaps}"
    assert gaps[0] > 0.20, "the models must disagree badly in the transition band"
    assert gaps[2] == pytest.approx(0.07, abs=0.02), "the 2λ rule of thumb is loose, not exact"
    assert gaps[-1] < 0.02, "and they must agree deep in the geometric regime"


def test_the_cheap_models_lose_half_the_contrast_where_this_project_operates() -> None:
    """A Boson focused at infinity looking at 10 m: c = 19.6 µm, W020 = 0.23λ. All three are the
    pure lens OTF with no aberration term, so the comparison is like for like."""
    shallow = float(defocus_w020_um(19.6, F1))
    hop_n = float(mtf_defocus_hopkins(XI_NYQUIST, LWIR_UM, F1, shallow))
    geo_n = float(lens_otf(XI_NYQUIST, LWIR_UM, F1, shallow, "geometric"))
    gau_n = float(lens_otf(XI_NYQUIST, LWIR_UM, F1, shallow, "gaussian"))
    focused = float(mtf_diffraction(XI_NYQUIST, LWIR_UM, F1))
    assert focused == pytest.approx(0.461, abs=0.002)
    assert hop_n == pytest.approx(0.356, abs=0.005)
    assert geo_n == pytest.approx(0.173, abs=0.005)
    assert gau_n == pytest.approx(0.203, abs=0.005)
    assert hop_n / geo_n > 2.0, "the geometric disk must lose about half the contrast here"


def test_hopkins_reverses_contrast_and_the_gaussian_cannot() -> None:
    """Spurious resolution is real and is why the Gaussian is an ablation, not a model."""
    w020 = 1.2 * LWIR_UM
    hop = mtf_defocus_hopkins(XI, LWIR_UM, F1, w020)
    assert np.any(hop < -1e-3), "a defocused lens must reverse contrast somewhere"
    gau = lens_otf(XI, LWIR_UM, F1, w020, "gaussian")
    assert np.all(gau >= 0.0), "a Gaussian can never reverse contrast"


def test_defocus_is_achromatic_and_band_averaging_only_reshapes_the_tail() -> None:
    """Measured, and it corrects the assumption `OC` was planned on.

    In ``a = 8π W020 s/λ`` with ``s = ξλF`` the wavelength **cancels**: a = 8π W020 ξ F. The blur
    circle c = 8 F W020 has no λ in it either, so geometric defocus is achromatic and band
    averaging cannot smear its zeros -- only diffraction is chromatic, through a cut-off that runs
    133 to 74 cyc/mm across 7.5-13.5 µm. The practical consequence is that averaging is a small
    correction here: under 0.3 % wherever the OTF still carries contrast, and dominant only in the
    far tail where the monochromatic curve is already near zero.
    """
    lam = np.linspace(7.5, 13.5, 25)
    w020 = 2.0 * LWIR_UM
    xi = np.linspace(1.0, 130.0, 500)
    mono = mtf_defocus_hopkins(xi, LWIR_UM, F1, w020)
    broad = band_average_otf(xi, lam, np.ones_like(lam), F1, w020, "hopkins")
    diff = np.abs(mono - broad)

    low = xi[:-1] < 40.0
    zeros_mono = xi[:-1][(np.diff(np.sign(mono)) != 0) & low]
    zeros_broad = xi[:-1][(np.diff(np.sign(broad)) != 0) & low]
    assert len(zeros_mono) >= 3, "the monochromatic curve must have zeros to compare"
    assert np.allclose(zeros_mono, zeros_broad[: len(zeros_mono)], atol=0.5), (
        "defocus zeros must not move with wavelength -- λ cancels out of Hopkins' a"
    )

    carries = np.abs(mono) >= 0.1
    assert np.max(diff[carries]) < 0.005, "the main lobe must barely move"
    tail = (xi >= 20.0) & (xi <= 74.0)
    assert np.max(diff[tail]) > 0.2 * np.max(np.abs(mono[tail])), "the tail must be reshaped"

    narrow = band_average_otf(xi, np.array([10.4999, 10.5001]), np.ones(2), F1, w020, "hopkins")
    assert np.allclose(narrow, mono, atol=2e-4), "a degenerate band is the monochromatic case"


def test_band_averaging_weights_by_the_response() -> None:
    """A response that only passes the short half must give the short half's OTF."""
    lam = np.linspace(7.5, 13.5, 25)
    short = np.where(lam <= 9.0, 1.0, 0.0)
    xi = np.linspace(1.0, 60.0, 60)
    got = band_average_otf(xi, lam, short, F1, 8.0, "hopkins")
    want = band_average_otf(xi, lam[lam <= 9.0], np.ones((lam <= 9.0).sum()), F1, 8.0, "hopkins")
    assert np.allclose(got, want, atol=5e-3)


def test_an_unknown_model_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown defocus model"):
        lens_otf(XI, LWIR_UM, F1, 1.0, "bokeh")
