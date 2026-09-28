"""M7.3 — band-effective material properties: ε(λ) through a camera's own eyes (§12.3, §3.1).

A material's ε is a spectrum; a band is a number. Which number depends on the camera, and on
which *kind* of camera: a bolometer absorbs power and averages under B(λ, T), a photon detector
counts photons and averages under B_q(λ, T). The two differ by hc/λ inside the integral, and the
direction of the difference is fixed — photon weighting leans long, so a falling ε(λ) always
averages lower under it. That is not a convention to be chosen per call site; it follows from the
FPA type, and `weighting_for_fpa` is the one place that decides.

The failure this step is really about is quieter: **extrapolating a curve past its own support.**
For a material whose ε falls off a cliff at a band edge — glass, most paints — carrying the last
tabulated value across is a large error that looks like a small one, biased in whichever direction
the curve happened to be heading. So a response the curve does not cover is refused.

docs/physics-model.md §12.3, §3.1, Appendix A #3; ADR 0010
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from irsim.materials.spectra import (
    PropertySpectrum,
    band_effective,
    load_property_spectrum,
    weighting_for_fpa,
)
from irsim.radiometry.band_average import T_REF_K
from irsim.radiometry.band_integration import quadrature_grid
from irsim.radiometry.planck import band_radiance_tophat
from irsim.radiometry.spectral_response import SpectralResponse, load_spectral_response

REPO = pathlib.Path(__file__).resolve().parents[2]
DATA = REPO / "data"
LWIR_LO, LWIR_HI = 7.5, 13.5


def _tophat(lo: float, hi: float, pad: float = 1e-6) -> SpectralResponse:
    return SpectralResponse(
        wavelength_um=np.array([lo - pad, lo, hi, hi + pad]),
        response=np.array([0.0, 1.0, 1.0, 0.0]),
        source_path="<test>",
        sha256="",
    )


def _curve(wavelength_um: list[float], values: list[float]) -> PropertySpectrum:
    return PropertySpectrum(
        wavelength_um=np.asarray(wavelength_um, dtype=np.float64),
        values=np.asarray(values, dtype=np.float64),
        path=pathlib.Path("<test>"),
    )


# ---------------------------------------------------------------------------------------------
# identities
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("form", ["energy", "photon"])
@pytest.mark.parametrize("value", [0.0, 0.9, 1.0])
def test_a_flat_spectrum_averages_to_itself(form: str, value: float) -> None:
    """The one exact identity available: no weighting can move a constant."""
    curve = _curve([5.0, 20.0], [value, value])
    got = curve.band_effective(_tophat(LWIR_LO, LWIR_HI), form=form)  # type: ignore[arg-type]
    assert got == pytest.approx(value, abs=1e-12)


def test_a_step_spectrum_equals_the_closed_form_top_hat_ratio() -> None:
    """ε = 1 below 10 µm and 0 above, over a 7.5–13.5 µm top-hat: a ratio of closed-form band
    radiances, computed independently of the quadrature that produced it.

    Held to 3e-3, not to 1e-6, and the reason is the integrand rather than the code. A step
    discontinuity is exactly what composite Simpson handles worst: the cut falls inside one cell
    of the resampling grid and up to 4/3 of that cell (the weight on an interior node) is
    attributed to the wrong side, which over a 6 µm band at 0.01 µm is 2.2e-3 of the band at
    most. The earlier bound of 5e-4 (measured 2.5e-4) held only because the 0.01 µm grid happened
    to put a node 1e-6 µm from the cut; AT.24's grid ends exactly on the support, its nodes fall
    elsewhere, and the measured error is 2.3e-3 -- inside the analytic bound, not inside the
    lucky one. The 1e-6 identities this module really rests on are the flat spectrum and
    linearity, both below.
    """
    cut = 10.0
    curve = _curve([5.0, cut - 1e-9, cut + 1e-9, 20.0], [1.0, 1.0, 0.0, 0.0])
    response = _tophat(LWIR_LO, LWIR_HI)
    got = curve.band_effective(response, t_ref_k=T_REF_K, form="energy")
    expected = band_radiance_tophat(LWIR_LO, cut, T_REF_K) / band_radiance_tophat(
        LWIR_LO, LWIR_HI, T_REF_K
    )
    grid = quadrature_grid(response)
    one_cell_bound = 4.0 / 3.0 * float(grid[1] - grid[0]) / (LWIR_HI - LWIR_LO) * 1.5
    assert abs(got / expected - 1.0) < one_cell_bound, (got, expected, one_cell_bound)
    assert got == pytest.approx(expected, rel=3e-3), (got, expected)


@pytest.mark.parametrize("form", ["energy", "photon"])
def test_the_band_average_is_linear_in_the_spectrum(form: str) -> None:
    """be(α s₁ + β s₂) = α be(s₁) + β be(s₂) to 1e-12 — exact, and quadrature-independent.

    Linearity is the property a weighted average cannot fake: it fails the moment a
    normalisation is applied in the wrong place, or the weight depends on the spectrum.
    """
    response = _tophat(LWIR_LO, LWIR_HI)
    grid = [5.0, 9.0, 11.0, 20.0]
    s1 = _curve(grid, [1.0, 0.2, 0.9, 0.3])
    s2 = _curve(grid, [0.0, 0.7, 0.1, 0.8])
    a, b = 0.37, 0.41
    mixed = _curve(grid, [a * x + b * y for x, y in zip(s1.values, s2.values, strict=True)])
    assert mixed.band_effective(response, form=form) == pytest.approx(  # type: ignore[arg-type]
        a * s1.band_effective(response, form=form)  # type: ignore[arg-type]
        + b * s2.band_effective(response, form=form),  # type: ignore[arg-type]
        abs=1e-12,
    )


def test_photon_weighting_is_lower_than_energy_weighting_for_a_falling_spectrum() -> None:
    """hc/λ inside the integral leans the average towards the long-wave end, always."""
    falling = _curve([5.0, 20.0], [1.0, 0.0])
    response = _tophat(LWIR_LO, LWIR_HI)
    energy = falling.band_effective(response, form="energy")
    photon = falling.band_effective(response, form="photon")
    assert photon < energy
    # and the sign flips for a rising spectrum, so this is about the weighting and not the curve
    rising = _curve([5.0, 20.0], [0.0, 1.0])
    assert rising.band_effective(response, form="photon") > rising.band_effective(
        response, form="energy"
    )


def test_the_weighting_follows_the_detector_and_not_the_caller() -> None:
    assert weighting_for_fpa("bolometer") == "energy"
    assert weighting_for_fpa("photon") == "photon"
    with pytest.raises(ValueError, match="FPA type"):
        weighting_for_fpa("thermopile")


def test_the_reference_temperature_matters_and_is_explicit() -> None:
    """A band-effective value is defined *at a temperature*; the same curve gives a different
    number at 1000 K, because Planck has moved under it."""
    falling = _curve([5.0, 20.0], [1.0, 0.0])
    response = _tophat(LWIR_LO, LWIR_HI)
    cold = falling.band_effective(response, t_ref_k=250.0)
    hot = falling.band_effective(response, t_ref_k=1000.0)
    assert hot > cold, (cold, hot)


# ---------------------------------------------------------------------------------------------
# coverage
# ---------------------------------------------------------------------------------------------


def test_a_band_the_curve_does_not_cover_is_refused() -> None:
    narrow = _curve([9.0, 11.0], [0.9, 0.9])
    with pytest.raises(ValueError, match="extend the table"):
        narrow.band_effective(_tophat(LWIR_LO, LWIR_HI))
    assert not narrow.covers(_tophat(LWIR_LO, LWIR_HI))
    assert narrow.covers(_tophat(9.5, 10.5))


def test_coverage_is_judged_on_where_the_response_responds_not_on_the_file_extent() -> None:
    """A response padded with zeros beyond its band does not need ε(λ) out there."""
    padded = SpectralResponse(
        wavelength_um=np.array([5.0, 9.0, 9.5, 10.5, 11.0, 20.0]),
        response=np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0]),
        source_path="<test>",
        sha256="",
    )
    curve = _curve([9.0, 11.0], [0.9, 0.9])
    assert curve.covers(padded)
    assert curve.band_effective(padded) == pytest.approx(0.9, abs=1e-12)


# ---------------------------------------------------------------------------------------------
# the committed curve, through the committed cameras
# ---------------------------------------------------------------------------------------------


def test_the_committed_paint_curve_works_through_the_boson() -> None:
    curve = load_property_spectrum(DATA / "spectra" / "materials" / "car_paint_black.csv")
    boson = load_spectral_response(DATA / "spectra" / "responses" / "boson_vox.csv")
    value = band_effective(curve, boson, form="energy")
    assert 0.0 < value <= 1.0
    lo, hi = float(curve.values.min()), float(curve.values.max())
    assert lo - 1e-9 <= value <= hi + 1e-9, "a weighted average left the curve's own range"


@pytest.mark.parametrize("response_name", ["boson_vox", "ingaas", "insb"])
def test_the_committed_paint_curve_covers_every_configured_camera(response_name: str) -> None:
    """0.3–15 µm, which is the point: a material spectrum authored for one band is a trap the
    moment a second band is configured, and M11 configured two."""
    curve = load_property_spectrum(DATA / "spectra" / "materials" / "car_paint_black.csv")
    response = load_spectral_response(DATA / "spectra" / "responses" / f"{response_name}.csv")
    assert curve.covers(response)
    value = curve.band_effective(response, form="energy")
    assert float(curve.values.min()) - 1e-9 <= value <= float(curve.values.max()) + 1e-9


def test_a_thermal_only_curve_is_refused_by_a_swir_camera_rather_than_extrapolated() -> None:
    """The mistake this step exists to prevent: a table that stops at the band edge, and a camera
    that looks past it. Carrying the last value across is invisible and biased."""
    thermal_only = _curve([7.0, 14.0], [0.9, 0.9])
    ingaas = load_spectral_response(DATA / "spectra" / "responses" / "ingaas.csv")
    assert not thermal_only.covers(ingaas)
    with pytest.raises(ValueError, match="extend the table"):
        thermal_only.band_effective(ingaas)
