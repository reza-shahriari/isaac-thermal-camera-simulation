"""The optical PSF's second factor, and where its number comes from (roadmap SC.4).

`aberration_sigma_um` was 0.0 on every shipped camera while the schema docstring called it "the
Gaussian fitted from a measured slant edge". A zero there is not a neutral default: it renders a
**diffraction-limited** lens, which is the best lens physics allows and not the lens anyone sells.

The check that matters is not the system MTF. FLIR's 42 % at Nyquist cascaded with the ideal 12 µm
box (sinc = 0.637) predicts 0.267, and the roadmap's acceptance band is 0.27 ± 0.03 — but a
diffraction-only Boson gives **0.294**, which is *inside* that band. The band alone would have
passed the broken camera. What separates them is the lens factor against the datasheet: 0.420
authored against 0.461 diffraction-only, a 10 % error that the system band hides because the box
filter dilutes it.

So this file checks the **derivation**, not the outcome: that the number in the YAML is the
solution of the datasheet's own equation and not a value fitted to make a golden pass.

docs/physics-model.md §8.3; ADR 0059, ADR 0117.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from irsim.config.loader import load_sensor_config
from irsim.optics.mtf import (
    aberration_sigma_for_mtf,
    mtf_detector,
    mtf_diffraction,
    mtf_system,
    nyquist_frequency_cyc_per_mm,
)

REPO = pathlib.Path(__file__).resolve().parents[2]
SENSORS = REPO / "configs" / "sensors"

#: FLIR's Boson datasheet, "MTF at Nyquist, nominal, on-axis" for the f/1.0 lens (ADR 0091 is the
#: reading of that datasheet). The *whole lens*, diffraction included -- which is the fact the
#: solve turns on.
FLIR_LENS_MTF_AT_NYQUIST = 0.42

#: Cameras whose lens is left ideal on purpose, with the reason. A camera that is in neither this
#: set nor the "has a figure" branch below fails, so the next one added has to decide rather than
#: inherit a zero.
IDEAL_LENS = {
    "example_mwir_insb_640": (
        "cooled MWIR optics are specified diffraction-limited over their own field, and no "
        "measured slant edge for this generic part exists; a fitted sigma would be invention"
    ),
    "example_swir_ingaas_640": (
        "a generic InGaAs camera with no named lens: nothing to solve the datasheet equation "
        "against"
    ),
    "example_nir_si_1280": ("a generic silicon NIR camera with no named lens, as above"),
}


def _boson_configs() -> list[pathlib.Path]:
    return [SENSORS / "flir_boson_640_lwir.yaml", SENSORS / "halmstad_boson_320.yaml"]


# --- the solver ---------------------------------------------------------------------------------


@pytest.mark.parametrize("target", [0.20, 0.42, 0.45])
def test_the_solved_sigma_puts_the_lens_back_on_its_published_figure(target: float) -> None:
    """Round trip: solve for σ, cascade it with diffraction, land on the number asked for."""
    xi = nyquist_frequency_cyc_per_mm(12.0)
    sigma = aberration_sigma_for_mtf(target, xi, 10.5, 1.0)
    assert float(mtf_system(xi, 10.5, 1.0, sigma_mm=sigma * 1e-3)) == pytest.approx(
        target, abs=1e-9
    )


def test_the_solver_leaves_the_detector_out_of_it() -> None:
    """The box filter is the pipeline's (ADR 0059); folding it in here would count it twice.

    Stated as a test rather than a comment because the failure is invisible: a sigma that had
    absorbed the detector footprint would still produce a plausible image, just a softer one, and
    the system MTF would come out at 0.17 instead of 0.27.
    """
    xi = nyquist_frequency_cyc_per_mm(12.0)
    sigma = aberration_sigma_for_mtf(0.42, xi, 10.5, 1.0)
    lens = float(mtf_system(xi, 10.5, 1.0, sigma_mm=sigma * 1e-3))
    assert lens == pytest.approx(0.42, abs=1e-9)
    assert float(mtf_detector(xi, 12e-3)) == pytest.approx(2.0 / np.pi, abs=1e-3)
    assert lens * float(mtf_detector(xi, 12e-3)) == pytest.approx(0.267, abs=0.002)


def test_an_unreachable_target_is_refused_rather_than_solved() -> None:
    """No Gaussian raises an MTF, so a figure above diffraction is a datasheet to re-read."""
    xi = nyquist_frequency_cyc_per_mm(12.0)
    with pytest.raises(ValueError, match="no aberration Gaussian can raise"):
        aberration_sigma_for_mtf(0.90, xi, 10.5, 1.0)
    for bad in (0.0, 1.0, -0.1):
        with pytest.raises(ValueError, match="target_mtf"):
            aberration_sigma_for_mtf(bad, xi, 10.5, 1.0)
    with pytest.raises(ValueError, match="frequency must be positive"):
        aberration_sigma_for_mtf(0.42, 0.0, 10.5, 1.0)


# --- the shipped cameras --------------------------------------------------------------------------


@pytest.mark.parametrize("path", _boson_configs(), ids=lambda p: p.stem)
def test_the_shipped_boson_carries_the_sigma_its_datasheet_implies(path: pathlib.Path) -> None:
    """The YAML's number, re-derived. If it ever drifts, this says so before an image does."""
    spec = load_sensor_config(path).sensor
    xi = nyquist_frequency_cyc_per_mm(spec.fpa.pitch_um)
    lam, f_num = spec.reference_wavelength_um, spec.optics.f_number
    expected = aberration_sigma_for_mtf(FLIR_LENS_MTF_AT_NYQUIST, xi, lam, f_num)

    sigma = spec.optics.mtf.aberration_sigma_um
    assert sigma == pytest.approx(expected, abs=5e-3), (sigma, expected)
    lens = float(mtf_system(xi, lam, f_num, sigma_mm=sigma * 1e-3))
    assert lens == pytest.approx(FLIR_LENS_MTF_AT_NYQUIST, abs=2e-3), lens


@pytest.mark.parametrize("path", _boson_configs(), ids=lambda p: p.stem)
def test_a_diffraction_limited_boson_is_ten_percent_too_sharp(path: pathlib.Path) -> None:
    """The defect `SC.4` exists to fix, as the number the system band cannot see.

    Lens at Nyquist: 0.461 with no aberration against the 0.420 FLIR publishes. After the box
    filter those become 0.294 and 0.267, and the roadmap's 0.27 ± 0.03 contains both — which is
    exactly why the sharp check has to be on the lens factor.
    """
    spec = load_sensor_config(path).sensor
    xi = nyquist_frequency_cyc_per_mm(spec.fpa.pitch_um)
    lam, f_num = spec.reference_wavelength_um, spec.optics.f_number
    ideal = float(mtf_diffraction(xi, lam, f_num))
    box = float(mtf_detector(xi, spec.fpa.pitch_um * 1e-3))

    assert ideal == pytest.approx(0.461, abs=0.002)
    assert ideal / FLIR_LENS_MTF_AT_NYQUIST > 1.09
    # Both land inside the roadmap's system band, which is the trap.
    assert ideal * box == pytest.approx(0.294, abs=0.003)
    assert abs(ideal * box - 0.27) < 0.03
    assert abs(FLIR_LENS_MTF_AT_NYQUIST * box - 0.27) < 0.03


def test_every_shipped_camera_has_decided_about_its_lens() -> None:
    """A zero is a claim -- "diffraction-limited" -- so it has to be made on purpose.

    A camera in neither branch fails, which is the point: the next one added cannot inherit a
    zero by omission the way all five did.
    """
    for path in sorted(SENSORS.glob("*.yaml")):
        spec = load_sensor_config(path).sensor
        sigma = spec.optics.mtf.aberration_sigma_um
        if path.stem in IDEAL_LENS:
            assert sigma == 0.0, f"{path.stem} is listed as ideal but carries sigma {sigma}"
            assert len(IDEAL_LENS[path.stem]) > 40, f"{path.stem} is ideal without a reason"
        else:
            assert sigma > 0.0, (
                f"{path.stem} renders a diffraction-limited lens. Either solve its sigma from a "
                "published MTF figure, or add it to IDEAL_LENS with the reason (SC.4)."
            )
