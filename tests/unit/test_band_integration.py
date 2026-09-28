"""The quadrature oracle (M1.4) against the closed forms it must reproduce.

Top-hat files have an exact answer (band_radiance_tophat / band_photon_radiance_tophat), so
the Simpson path is checked against them in percent *and* in millikelvin -- the unit a NETD is
judged in. The in-band derivative is checked against a finite difference of the same integral,
and against the §9.4 anchor that a 39 mK NETD at 300 K reads 23 mK at 373 K (ratio 1.736).
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from irsim.config.loader import DEFAULT_DATA_DIR
from irsim.radiometry.band_integration import (
    FINE_GRID_FRACTION,
    band_photon_radiance,
    band_radiance,
    d_band_photon_radiance_dT,
    d_band_radiance_dT,
    quadrature_grid,
    simpson,
)
from irsim.radiometry.planck import (
    band_photon_radiance_tophat,
    band_radiance_tophat,
    spectral_radiance,
)
from irsim.radiometry.spectral_response import SpectralResponse, load_spectral_response

BOSON_CSV = DEFAULT_DATA_DIR / "spectra" / "responses" / "boson_vox.csv"


def _tophat_response(tmp_path: pathlib.Path, lo: float, hi: float) -> SpectralResponse:
    """A file whose only samples are the band edges: exact top-hat after edge-aligned resampling."""
    p = tmp_path / f"tophat_{lo}_{hi}.csv"
    p.write_text(f"# exact top-hat\n{lo},1.0\n{hi},1.0\n")
    return load_spectral_response(p)


def test_grid_is_odd_uniform_and_ends_on_both_edges(tmp_path: pathlib.Path) -> None:
    """AT.24: the last node is the last sample -- never a step beyond it -- and the spacing is
    at most 0.01 µm and at most FINE_GRID_FRACTION of the short edge."""
    for lo, hi in ((8.0, 12.0), (7.5, 13.5), (0.9, 1.7), (3.0, 5.005), (0.75, 1.0), (0.7, 1.1)):
        g = quadrature_grid(_tophat_response(tmp_path, lo, hi))
        assert g.size % 2 == 1 and g[0] == lo and g[-1] == pytest.approx(hi, abs=1e-12)
        step = float(np.diff(g).max())
        assert np.allclose(np.diff(g), step)
        assert step <= 0.01 + 1e-12 and step <= FINE_GRID_FRACTION * lo + 1e-12
        intervals = (hi - lo) / 0.01
        if lo >= 2.0 and abs(intervals - round(intervals)) < 1e-9 and round(intervals) % 2 == 0:
            assert step == pytest.approx(0.01, rel=1e-9), "an even 0.01 grid is unchanged"


def test_simpson_is_exact_for_cubics() -> None:
    x = np.linspace(0.0, 2.0, 21)
    assert simpson(x**3 - 2 * x**2 + 1, x[1] - x[0]) == pytest.approx(4.0 - 16 / 3 + 2, abs=1e-12)
    with pytest.raises(ValueError):
        simpson(x[:-1], 0.1)


@pytest.mark.parametrize(
    ("lo", "hi", "T"),
    [
        (7.5, 13.5, 300.0),
        (3.0, 5.0, 300.0),
        (3.0, 5.0, 500.0),
        (0.9, 1.7, 300.0),
        # AT.24: odd interval counts at 0.01 µm, which the padded grid got wrong by +33 % (300 K)
        # and +54 % (200 K) on the first of these
        (0.75, 1.0, 200.0),
        (0.75, 1.0, 300.0),
        (0.75, 1.0, 1000.0),
        (3.0, 5.005, 300.0),
    ],
)
def test_tophat_matches_closed_form_in_percent_and_mK(
    tmp_path: pathlib.Path, lo: float, hi: float, T: float
) -> None:
    sr = _tophat_response(tmp_path, lo, hi)
    quad = float(band_radiance(sr, T)[0])
    closed = band_radiance_tophat(lo, hi, T)
    rel = quad / closed - 1.0
    err_mk = abs(quad - closed) / float(d_band_radiance_dT(sr, T)[0]) * 1e3
    assert abs(rel) < 1e-3, f"{lo}-{hi} um at {T} K: {rel:.2e} ({err_mk:.3f} mK)"
    assert err_mk < 5.0, f"{err_mk:.3f} mK exceeds the 5 mK LUT budget"
    quad_q = float(band_photon_radiance(sr, T)[0])
    closed_q = band_photon_radiance_tophat(lo, hi, T)
    assert abs(quad_q / closed_q - 1.0) < 1e-3


def test_in_band_derivative_matches_finite_difference(tmp_path: pathlib.Path) -> None:
    sr = _tophat_response(tmp_path, 7.5, 13.5)
    temps = np.linspace(200.0, 1000.0, 33)
    d_t = 1e-3
    fd = (band_radiance(sr, temps + d_t) - band_radiance(sr, temps - d_t)) / (2 * d_t)
    np.testing.assert_allclose(d_band_radiance_dT(sr, temps), fd, rtol=1e-5)
    fd_q = (band_photon_radiance(sr, temps + d_t) - band_photon_radiance(sr, temps - d_t)) / (
        2 * d_t
    )
    np.testing.assert_allclose(d_band_photon_radiance_dT(sr, temps), fd_q, rtol=1e-5)


def test_known_answer_netd_scaling_300_to_373k(tmp_path: pathlib.Path) -> None:
    """dLb/dT(373)/dLb/dT(300) = 1.736 for 7.5-13.5 µm: the 39 -> 23 mK NETD anchor of [R9]."""
    sr = _tophat_response(tmp_path, 7.5, 13.5)
    ratio = float(d_band_radiance_dT(sr, 373.0)[0] / d_band_radiance_dT(sr, 300.0)[0])
    assert ratio == pytest.approx(1.736, rel=5e-3), ratio


def test_vectorised_over_temperature_and_monotone(tmp_path: pathlib.Path) -> None:
    sr = load_spectral_response(BOSON_CSV)
    temps = np.arange(200.0, 1000.0 + 1e-9, 5.0)
    lb = band_radiance(sr, temps)
    assert lb.shape == temps.shape and lb.dtype == np.float64
    assert np.all(np.diff(lb) > 0) and np.all(d_band_radiance_dT(sr, temps) > 0)
    assert np.all(np.diff(band_photon_radiance(sr, temps)) > 0)
    # scalar and array calls agree
    assert float(band_radiance(sr, 300.0)[0]) == lb[20]


def test_boson_estimate_is_close_to_its_tophat(tmp_path: pathlib.Path) -> None:
    """Soft edges centred on the edges: Lb within a few percent of the 7.5-13.5 top-hat."""
    boson = load_spectral_response(BOSON_CSV)
    lb = float(band_radiance(boson, 300.0)[0])
    assert abs(lb / band_radiance_tophat(7.5, 13.5, 300.0) - 1.0) < 0.03
    assert lb == pytest.approx(55.49, rel=0.03)  # W m^-2 sr^-1, the roadmap's top-hat anchor


RESPONSES = pathlib.Path(__file__).resolve().parents[2] / "data" / "spectra" / "responses"


def _converged(sr: SpectralResponse, temps: np.ndarray, dl_um: float = 0.0005) -> np.ndarray:
    """An independent quadrature: twenty times finer, same Simpson, same resampling."""
    lo, hi = sr.support_um
    n = int(np.ceil((hi - lo) / dl_um))
    n += n % 2
    grid = lo + (hi - lo) / n * np.arange(n + 1, dtype=np.float64)
    weights = sr.resampled(grid)
    spectral = spectral_radiance(grid[None, :], temps[:, None])
    return np.asarray(simpson(weights[None, :] * spectral, float(grid[1] - grid[0])))


@pytest.mark.parametrize("name", ["boson_vox", "insb", "ingaas", "nir_si"])
def test_every_shipped_response_is_within_3mK_of_a_converged_quadrature(name: str) -> None:
    """AT.24: the oracle the LUTs are built on, held to a grid twenty times finer than its own,
    in millikelvin through the band's own dL/dT. Measured: LWIR 0.53, MWIR 0.11, SWIR 0.21 and
    NIR 2.6 mK worst case over 200–1000 K; before the wavelength-scaled spacing NIR was 28 mK."""
    sr = load_spectral_response(RESPONSES / f"{name}.csv")
    temps = np.array([200.0, 250.0, 300.0, 400.0, 600.0, 1000.0])
    ref = _converged(sr, temps)
    got = band_radiance(sr, temps)
    err_mk = np.abs(got - ref) / d_band_radiance_dT(sr, temps) * 1e3
    assert float(err_mk.max()) < 3.0, f"{name}: {err_mk.max():.3f} mK"
