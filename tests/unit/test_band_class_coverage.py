"""AT.2 / AT.3 — the camera's own R(λ) reaches the atmosphere, and every band's classes cover it.

`Scene.from_config` built `LayeredAtmosphere(preset, weather, luts)` and never passed the fourth
`responses` argument, so `class_weights` fell back to a **nominal top-hat** for every band. The
model then describes a different camera than the one being simulated, and says nothing about it.

Measured on the shipped InSb MWIR response: the `h2o_wing` class weight goes **0.0192 → 0.1303, a
6.8× change** in how much of the band is treated as a water wing (0.0238 before `AT.10` derived
the classes from the band's span rather than reading them off a hand-written table). LWIR barely
moves, because the Boson's response happens to be close to its nominal top-hat — which is exactly
why this stayed invisible: the one band the project renders most is the one the shortcut suits.

The two steps land together because AT.2 exposes AT.3. Supplying the real NIR response used to
**raise**: `BAND_CLASSES['nir']` stopped at 1.05 µm and `nir_si.csv` reaches 1.10, carrying 0.188 %
of the Planck-weighted band outside any class. SWIR and MWIR had the same shortfall and passed only
because their responses carry 0.000 % out there — luck, not coverage.

docs/physics-model.md §7.4; ADR 0071; roadmap AT.2, AT.3.
"""

from __future__ import annotations

import pathlib
import warnings

import numpy as np
import pytest

from irsim.atmosphere.layered import LayeredAtmosphere, class_weights, classes_for
from irsim.atmosphere.library import load_atmosphere_preset
from irsim.config.loader import load_sensor_config
from irsim.radiometry.lut_files import load_band_response_for_config
from irsim.radiometry.spectral_response import load_spectral_response
from irsim.thermal import WeatherSample, WeatherSeries

REPO = pathlib.Path(__file__).resolve().parents[2]
RESPONSES = REPO / "data" / "spectra" / "responses"
SENSORS = sorted((REPO / "configs" / "sensors").glob("*.yaml"))
BAND_FILES = {
    "nir": "nir_si.csv",
    "swir": "ingaas.csv",
    "mwir": "insb.csv",
    "lwir": "boson_vox.csv",
}


def _weather() -> WeatherSeries:
    return WeatherSeries.constant(
        WeatherSample(288.15, 0.46, 1.0, 0.0, 0.0, 0.0, 23000.0, 0.0), 3600.0
    )


# --- AT.3: every class set covers its own response ---------------------------------------------


def test_there_are_sensors_to_walk() -> None:
    assert len(SENSORS) >= 4, "the shipped sensor set has changed shape"


@pytest.mark.parametrize("path", SENSORS, ids=lambda p: p.stem)
def test_every_shipped_sensor_response_is_inside_its_bands_classes(path: pathlib.Path) -> None:
    """The guard AT.3 asks for: a response reaching past its classes must not be possible to ship.

    `class_weights` raises when weight falls outside every class, so this is the whole check — but
    it walks the *configs* rather than a hand-written list, so adding a camera cannot quietly add a
    band whose classes do not reach its detector's tail.
    """
    config = load_sensor_config(path)
    band = config.sensor.band.band_id
    response = load_band_response_for_config(config, REPO / "data")
    weights = class_weights(band, response)
    assert weights.shape == (len(classes_for(band, response)),)
    assert weights.sum() == pytest.approx(1.0)
    assert np.all(weights >= 0.0)


@pytest.mark.parametrize("band", sorted(BAND_FILES))
def test_the_class_edges_span_the_detector(band: str) -> None:
    """Coverage by construction, not by the response happening to be zero out there.

    Before AT.3, NIR was short by 0.05 µm carrying 0.188 % of the band — enough to raise — while
    SWIR (short both sides) and MWIR (short by 0.4 µm) carried 0.000 % and passed on luck. Since
    `AT.10` the classes are **derived** from the band's own span, which is its nominal range and
    its response together, so this holds for any detector inside the ladder rather than for the
    four that happen to ship.
    """
    response = load_spectral_response(RESPONSES / BAND_FILES[band])
    edges = [e for c in classes_for(band, response) for e in c.edges_um]
    low, high = min(e[0] for e in edges), max(e[1] for e in edges)
    lo_r, hi_r = response.support_um
    assert low <= lo_r, f"{band} classes start at {low}, response at {lo_r}"
    assert high >= hi_r, f"{band} classes end at {high}, response at {hi_r}"


# --- AT.2: the response changes the answer, and the fallback is loud ---------------------------


def test_the_mwir_water_wing_weight_moves_by_a_factor_of_six() -> None:
    """The measured defect, recorded as a number so a later change cannot quietly undo it.

    The nominal figure moved from 0.0238 to 0.0192 when `AT.10` derived the classes: a top-hat
    over MWIR's nominal 3.0-5.0 µm does not reach the 5.0-6.0 µm half of `h2o_wing` at all, where
    the hand-written table handed it to the band regardless of whether the camera could see there.
    The real response does reach it, which is the whole point.
    """
    response = load_spectral_response(RESPONSES / "insb.csv")
    nominal_names = [c.name for c in classes_for("mwir")]
    real_names = [c.name for c in classes_for("mwir", response)]
    nominal = class_weights("mwir")[nominal_names.index("h2o_wing")]
    real = class_weights("mwir", response)[real_names.index("h2o_wing")]
    assert nominal == pytest.approx(0.0192, abs=5e-4)
    assert real == pytest.approx(0.1303, abs=5e-4)
    assert real / nominal > 6.0


def test_lwir_barely_moves_which_is_why_this_hid() -> None:
    """The Boson's response is close to its nominal top-hat, so the band rendered most was fine."""
    nominal = class_weights("lwir")
    real = class_weights("lwir", load_spectral_response(RESPONSES / "boson_vox.csv"))
    assert np.max(np.abs(real - nominal)) < 5e-4


def test_a_layered_atmosphere_without_a_response_warns(recwarn: pytest.WarningsRecorder) -> None:
    """Silent was the problem: the output looks like an output either way."""
    atmosphere = LayeredAtmosphere(load_atmosphere_preset("us_standard_clear"), _weather())
    atmosphere.weights("mwir")
    assert any("nominal top-hat" in str(w.message) for w in recwarn)


def test_a_layered_atmosphere_with_a_response_is_quiet_and_different() -> None:
    """The two answers differ in the transmittance, not only in a weight vector.

    Since `AT.10` they can differ in *shape* too -- a nominal top-hat over 3.0-5.0 µm derives
    fewer classes than the InSb response's 2.5-6.0 µm does -- so the comparison is made on the
    quantity a render actually uses.
    """
    response = load_spectral_response(RESPONSES / "insb.csv")
    preset, weather = load_atmosphere_preset("us_standard_clear"), _weather()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        supplied = LayeredAtmosphere(preset, weather, None, {"mwir": response})
        tau_supplied = float(supplied.transmittance("mwir", 0.0, 5000.0, 0.0))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fallback = LayeredAtmosphere(preset, weather)
        tau_fallback = float(fallback.transmittance("mwir", 0.0, 5000.0, 0.0))
    assert supplied.weights("mwir").shape != fallback.weights("mwir").shape
    assert abs(tau_supplied - tau_fallback) > 0.01


def test_the_response_loader_and_the_lut_come_from_one_path() -> None:
    """They must describe one camera, so they are loaded from the same field of the same config."""
    config = load_sensor_config(REPO / "configs" / "sensors" / "example_mwir_insb_640.yaml")
    response = load_band_response_for_config(config, REPO / "data")
    named = load_spectral_response(REPO / "data" / config.sensor.band.spectral_response)
    assert response.support_um == named.support_um


# --- the scene actually passes it ----------------------------------------------------------------


def test_a_scene_threads_the_response_into_its_layered_atmosphere(tophat_lwir_lut) -> None:
    """The defect was a missing argument at exactly one call site; this is that argument."""
    from irsim.scene import Scene

    scene_yaml = REPO / "configs" / "scenes" / "sky_target_clear_day.yaml"
    response = load_spectral_response(RESPONSES / "boson_vox.csv")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        scene = Scene.from_file(scene_yaml, {"lwir": tophat_lwir_lut}, responses={"lwir": response})
        assert scene.layered is not None
        scene.layered.weights("lwir")


def test_a_scene_without_a_response_still_builds_but_warns(tophat_lwir_lut) -> None:
    """Existing callers keep working; they are told what they are getting."""
    from irsim.scene import Scene

    scene_yaml = REPO / "configs" / "scenes" / "sky_target_clear_day.yaml"
    scene = Scene.from_file(scene_yaml, {"lwir": tophat_lwir_lut})
    assert scene.layered is not None
    with pytest.warns(UserWarning, match="nominal top-hat"):
        scene.layered.weights("lwir")
