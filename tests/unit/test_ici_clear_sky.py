"""XD.6: the clear LWIR sky held to ARM's calibrated Infrared Cloud Imager (§7.1, §15 T3).

The ICI measured the downwelling sky in W/(m^2 sr) per pixel at the ARM SGP site in 2023; the
derived file holds the clear-pixel radiance of 92 of its images in eight elevation bands, with the
surface temperature and PWV the instrument's processing printed beside each. These tests hold the
layered sky (ADR 0071) to it where it can be held, and record where it cannot yet:

* with each night's water column pinned (AT.37: the ERA5 dew point for the surface humidity, the
  imager's own PWV for the depth), the cold, dry December night (2.5 C, 0.86 cm) is the
  measurement to 7 % at every elevation and the humid August night (27 C, 4.0 cm) to 10 % -- it
  was 24-28 % too bright before AT.37 halved the self-continuum's column and refit its square;
* the May night (20 C, 1.4 cm) is not: its real water sits in a moist layer under very dry air
  (ERA5: 61 % RH at 770 m, 3 % at 2.9 km), which one exponential cannot draw, and it reads 6-24 %
  bright. That is a ``strict`` xfail naming the step that owns it (AT.38, a tabulated profile).

The band is a stand-in (the ICI's response shape is not published; ADR 0183), which can scale the
skies but not make the misfit grow with humidity -- the last test shows the December agreement does
not hinge on it.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from irsim.atmosphere.library import load_atmosphere_preset
from irsim.radiometry.lut import BandLUT
from irsim.radiometry.spectral_response import SpectralResponse, load_spectral_response
from irsim.validation.ici import (
    load_ici_clear_sky,
    pinned_column,
    ratios,
    surface_rh_for_pwv,
)

REPO = pathlib.Path(__file__).resolve().parents[2]
CSV = REPO / "data" / "validation" / "ici_sgp2023_clear_sky.csv"
VOX = REPO / "data" / "spectra" / "responses" / "boson_vox.csv"


@pytest.fixture(scope="module")
def images():  # type: ignore[no-untyped-def]
    return load_ici_clear_sky(CSV)


@pytest.fixture(scope="module")
def band():  # type: ignore[no-untyped-def]
    response = load_spectral_response(VOX)
    return response, BandLUT.build(response, t0_k=150.0, t1_k=400.0, n=5001)


def _night(images, day: str):  # type: ignore[no-untyped-def]
    return [im for im in images if im.clear and im.utc.startswith(day)]


def test_the_derived_file_is_the_deployment_it_says_it_is(images) -> None:  # type: ignore[no-untyped-def]
    assert len(images) == 92
    assert {im.utc[:10] for im in images} == {
        "2023-05-21", "2023-07-14", "2023-07-20", "2023-08-06", "2023-12-11"
    }  # fmt: skip
    clear = [im for im in images if im.clear]
    assert len(clear) == 30
    for im in clear:
        sky = im.radiance[np.isfinite(im.radiance)]
        # a clear sky brightens monotonically toward the horizon (§5.3 a), in the measurement too
        assert np.all(np.diff(sky[::-1]) >= -0.5), im.utc
        assert 5.0 < sky[-1] < 30.0 and 263.0 < im.t_surface_k < 310.0
    # met read off the time-lapse: smooth minute to minute, as a misread digit would not be
    aug = _night(images, "2023-08-06")
    assert np.max(np.abs(np.diff([im.t_surface_k for im in aug]))) < 0.3
    assert np.max(np.abs(np.diff([im.pwv_cm for im in aug]))) < 0.05


def test_the_surface_humidity_holds_the_measured_column() -> None:
    rh, needed = surface_rh_for_pwv(300.15, 3.97, 2000.0)
    assert rh == needed == pytest.approx(19.85 / 25.6, rel=0.02)  # w0 = 19.85 g/m3 at 27 C
    rh_wet, needed_wet = surface_rh_for_pwv(300.15, 6.0, 2000.0)
    assert rh_wet == 1.0 and needed_wet > 1.0  # flagged, not silently capped


def test_every_clear_image_carries_its_hour_s_dew_point_and_the_column_round_trips(images) -> None:  # type: ignore[no-untyped-def]
    """AT.37: the pinned column is the measured one -- its surface humidity spread over the
    returned H_w gives back the imager's PWV -- and the depths it implies are the nights' real
    spread, not the preset's single 2 km."""
    from irsim.atmosphere.humidity import absolute_humidity_g_m3

    clear = [im for im in images if im.clear]
    heights = {}
    for im in clear:
        rh, h_w = pinned_column(im)  # type: ignore[misc]
        assert 0.0 < rh <= 1.0 and im.td_surface_k <= im.t_surface_k
        w0 = absolute_humidity_g_m3(im.t_surface_k, rh)
        assert w0 * h_w / 1.0e4 == pytest.approx(im.pwv_cm, rel=1e-12)
        heights.setdefault(im.utc[:10], []).append(h_w)
    mean = {day: float(np.mean(h)) for day, h in heights.items()}
    assert 1300.0 < mean["2023-05-21"] < 1700.0  # a shallow moist layer
    assert 2400.0 < mean["2023-08-06"] < 2900.0
    assert 2500.0 < mean["2023-12-11"] < 3100.0


def test_the_cold_dry_night_is_the_measurement_to_seven_percent(images, band) -> None:  # type: ignore[no-untyped-def]
    response, lut = band
    night = _night(images, "2023-12-11")
    assert len(night) == 5
    r = ratios(night, load_atmosphere_preset("us_standard_clear"), response, lut)
    assert np.all(np.abs(r - 1.0) < 0.07), np.round(r, 3)


def test_the_humid_night_is_the_measurement_to_ten_percent(images, band) -> None:  # type: ignore[no-untyped-def]
    """AT.37's criterion: 24-28 % too bright under ADR 0183's comparison; 0.98-1.09 with the
    column pinned and the self-continuum's square carried on its own (half) scale height."""
    response, lut = band
    night = _night(images, "2023-08-06")
    r = ratios(night, load_atmosphere_preset("us_standard_clear"), response, lut)
    assert np.all(np.abs(r - 1.0) < 0.10), np.round(r, 3)


@pytest.mark.xfail(
    strict=True,
    reason="AT.38: May's water is a moist layer under very dry air, which one exponential "
    "cannot draw; the zenith reads ~22 % bright with the column's total pinned",
)
def test_the_spring_night_is_the_measurement_to_ten_percent(images, band) -> None:  # type: ignore[no-untyped-def]
    response, lut = band
    night = _night(images, "2023-05-21")
    r = ratios(night, load_atmosphere_preset("us_standard_clear"), response, lut)
    assert np.all(np.abs(r - 1.0) < 0.10), np.round(r, 3)


def test_the_cold_night_agreement_does_not_hinge_on_the_stand_in_band(images) -> None:  # type: ignore[no-untyped-def]
    """A top-hat 7.5-13.5 um and the raised-cosine VOx curve put the zenith within 3 % of each
    other; the 8-14 um top-hat within 7 %. ADR 0183's humid misfit (25 %) was outside them all."""
    night = _night(images, "2023-12-11")[:1]
    preset = load_atmosphere_preset("us_standard_clear")
    zenith = []
    for lo, hi in ((7.5, 13.5), (8.0, 14.0)):
        hat = SpectralResponse(np.array([lo, hi]), np.array([1.0, 1.0]), f"<{lo}-{hi}>", "")
        lut = BandLUT.build(hat, t0_k=150.0, t1_k=400.0, n=5001)
        zenith.append(float(ratios(night, preset, hat, lut)[0, -1]))
    assert all(abs(z - 1.0) < 0.07 for z in zenith), zenith
