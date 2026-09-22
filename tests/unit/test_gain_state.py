"""Fire on the camera: the gain state, the rail and what the AGC does with them (roadmap PH.8).

Everything before this step measured radiometry. This one measures a *camera*, and the two
questions it answers are the ones a scene author cannot answer by looking at a frame.

**Is the ceiling in the right units?** For a single blackbody pixel, clipping the temperature at
500 °C and clipping the radiance at ``L_B(500 °C)`` are the same operation, which is exactly why
the mistake survives review. They part company the moment a pixel is a *scene*: what arrives is
``ε L_B(T) + (1 − ε) L_env``, which is not ``L_B`` of anything. Measured below on the shipped
Boson, for a 1273 K surface under the low-gain ceiling: at ε = 0.6 the radiance clip rails and a
kelvin clip **misses the rail**; at ε = 0.4 neither rails and a kelvin clip is **120 K wrong on a
pixel nowhere near saturation**. A frame with a saturated flame in it looks like a frame with a
saturated flame in it, so neither error would be questioned.

**What does the rail do to everything else?** FLIR's own note: a linear AGC collapses. Measured
below: the same person against the same room goes from **252 DN** of separation to **1.0 DN** when
a fire enters the frame, while plateau equalisation keeps **90 DN**. That is the whole argument for
the plateau mode, as a number.

docs/physics-model.md §9.2, §11.3; ADR 0091, ADR 0116.
"""

from __future__ import annotations

import copy
import pathlib
from typing import Any

import numpy as np
import pytest
import yaml

from irsim.config.sensor import SensorConfig
from irsim.detector.gain_state import (
    BOSON_GAIN_CEILING_K,
    clip_to_gain_ceiling,
    gain_ceiling_radiance,
)
from irsim.materials import MaterialTable
from irsim.pipeline import PipelineConfig, PipelineState, run_frame
from irsim.radiometry.lut import BandLUT
from irsim.radiometry.spectral_response import load_spectral_response

REPO = pathlib.Path(__file__).resolve().parents[2]
BOSON = yaml.safe_load((REPO / "configs" / "sensors" / "flir_boson_640_lwir.yaml").read_text())
ROOM_K = 303.0
PERSON_K = 310.0
FLAME_K = 1273.0
DN_TOP = 65535


@pytest.fixture(scope="module")
def boson_lut() -> BandLUT:
    return BandLUT.build(load_spectral_response(REPO / "data/spectra/responses/boson_vox.csv"))


def _config(
    lut: BandLUT,
    *,
    ceiling_k: float | None = None,
    agc: str = "linear",
    emissivity: float = 0.98,
    size: tuple[int, int] = (32, 24),
    display: bool = False,
) -> PipelineConfig:
    d = copy.deepcopy(BOSON)
    d["sensor"]["fpa"].update(width=size[0], height=size[1])
    d["sensor"]["optics"]["supersample_factor"] = 1
    d["sensor"]["isp"]["agc"] = agc
    if ceiling_k is not None:
        d["sensor"]["fpa"]["gain_ceiling_k"] = ceiling_k
    if display:
        d["sensor"]["outputs"]["display_8"] = True
    return PipelineConfig.from_sensor(
        SensorConfig.model_validate(d),
        MaterialTable.constant(emissivity),
        lut=lut,
        noise_enabled=False,
        psf_enabled=False,
    )


def _planes(config: PipelineConfig, hot_k: float, *, person: bool = False) -> dict[str, Any]:
    h, w = config.sensor.sensor.fpa_shape
    t = np.full((h, w), ROOM_K, dtype=np.float32)
    t[h // 3 : 2 * h // 3, w // 3 : 2 * w // 3] = hot_k
    if person:
        t[h // 8 : h // 8 + h // 4, 3 * w // 4 : 3 * w // 4 + w // 8] = PERSON_K
    return {
        "temperature_k": t,
        "material_id": np.ones((h, w), dtype=np.uint16),
        "distance_m": np.full((h, w), 6.0, dtype=np.float32),
    }


# --- the ceiling itself ------------------------------------------------------------------------


def test_the_bosons_two_states_are_the_datasheets() -> None:
    """Rev 340's "Scene Dynamic Range": −40 to +140 °C high gain, −40 to +500 °C low."""
    assert BOSON_GAIN_CEILING_K["high"] == pytest.approx(413.15)
    assert BOSON_GAIN_CEILING_K["low"] == pytest.approx(773.15)


def test_a_scene_under_the_ceiling_is_the_same_object(boson_lut: BandLUT) -> None:
    """Bit-identical, and the *same array*, so every golden written before `PH.8` still holds."""
    plane = np.linspace(1.0, 4.0, 48).reshape(6, 8)
    ceiling = gain_ceiling_radiance(BOSON_GAIN_CEILING_K["low"], boson_lut)
    assert clip_to_gain_ceiling(plane, ceiling) is plane
    hot = np.array([[ceiling * 2.0]])
    assert float(clip_to_gain_ceiling(hot, ceiling)[0, 0]) == pytest.approx(ceiling)


def test_the_ceiling_is_refused_in_float16_and_at_a_nonsense_level(boson_lut: BandLUT) -> None:
    with pytest.raises(TypeError, match="float16"):
        clip_to_gain_ceiling(np.ones((2, 2), dtype=np.float16), 1.0)
    with pytest.raises(ValueError, match="positive"):
        gain_ceiling_radiance(0.0, boson_lut)
    with pytest.raises(ValueError, match="positive"):
        clip_to_gain_ceiling(np.ones((2, 2)), 0.0)


def test_high_gain_rails_on_a_400_degree_object_and_low_gain_does_not(boson_lut: BandLUT) -> None:
    """The gain state is the transfer, so the clip lands exactly on the converter's top code.

    A 400 °C object sits between the Boson's two ceilings, which is what makes it the test: the
    same scene, the same camera, and the choice of state decides whether the frame is usable.
    """
    high = _config(boson_lut, ceiling_k=BOSON_GAIN_CEILING_K["high"])
    low = _config(boson_lut, ceiling_k=BOSON_GAIN_CEILING_K["low"])
    hot_k = 673.15  # 400 °C

    out_high = run_frame(_planes(high, hot_k), high, PipelineState())
    out_low = run_frame(_planes(low, hot_k), low, PipelineState())
    assert out_high.dn16 is not None and out_low.dn16 is not None

    assert int(out_high.dn16.max()) == DN_TOP
    assert int(out_low.dn16.max()) < DN_TOP
    # Railed, the camera reports its own ceiling and not the object.
    assert float(out_high.apparent_t.max()) == pytest.approx(BOSON_GAIN_CEILING_K["high"], abs=0.5)
    assert float(out_low.apparent_t.max()) == pytest.approx(hot_k, abs=8.0)


def test_a_state_whose_ceiling_is_below_its_floor_is_refused(boson_lut: BandLUT) -> None:
    with pytest.raises(ValueError, match="would hold no scene"):
        _config(boson_lut, ceiling_k=200.0)


# --- the units the ceiling is in ----------------------------------------------------------------


@pytest.mark.parametrize(
    "emissivity, radiance_rails, kelvin_rails",
    [(1.0, True, True), (0.6, True, False)],
)
def test_a_kelvin_ceiling_misses_a_rail_a_radiance_ceiling_catches(
    boson_lut: BandLUT, emissivity: float, radiance_rails: bool, kelvin_rails: bool
) -> None:
    """The first half of ADR 0116's argument, and it only shows on a non-black surface.

    At ε = 1 the two agree exactly, which is why the mistake survives: every test written against
    a blackbody passes either way. At ε = 0.6 a 1273 K surface still delivers more than the
    ceiling's blackbody radiance and rails — and a ceiling applied to its *temperature* caps it at
    500 °C first, so the rail never happens and the frame looks merely hot.
    """
    ceiling = BOSON_GAIN_CEILING_K["low"]
    config = _config(boson_lut, ceiling_k=ceiling, emissivity=emissivity, size=(16, 16))
    planes = _planes(config, FLAME_K)
    radiance = run_frame(planes, config, PipelineState())

    kelvin_planes = dict(planes)
    kelvin_planes["temperature_k"] = np.minimum(planes["temperature_k"], np.float32(ceiling))
    kelvin = run_frame(kelvin_planes, config, PipelineState())

    assert (int(radiance.dn16.max()) == DN_TOP) is radiance_rails
    assert (int(kelvin.dn16.max()) == DN_TOP) is kelvin_rails


def test_a_kelvin_ceiling_corrupts_pixels_that_never_reached_it(boson_lut: BandLUT) -> None:
    """The second half, and the worse one: it changes a pixel that is nowhere near saturation.

    A ε = 0.4 surface at 1273 K arrives as 663 K of apparent temperature — comfortably under the
    500 °C ceiling, so nothing should touch it. Clipping its *temperature* first drops it to 544 K.
    That is **119 K of error on an unsaturated pixel**, invented by a clip that should not have
    applied at all.
    """
    ceiling = BOSON_GAIN_CEILING_K["low"]
    config = _config(boson_lut, ceiling_k=ceiling, emissivity=0.4, size=(16, 16))
    planes = _planes(config, FLAME_K)
    radiance = run_frame(planes, config, PipelineState())

    kelvin_planes = dict(planes)
    kelvin_planes["temperature_k"] = np.minimum(planes["temperature_k"], np.float32(ceiling))
    kelvin = run_frame(kelvin_planes, config, PipelineState())

    assert int(radiance.dn16.max()) < DN_TOP, "the premise is an unsaturated pixel"
    true_k = float(radiance.apparent_t.max())
    wrong_k = float(kelvin.apparent_t.max())
    assert true_k == pytest.approx(663.0, abs=5.0), true_k
    assert true_k - wrong_k > 100.0, (true_k, wrong_k)


# --- what the rail does to the picture ------------------------------------------------------------


def _person_room_contrast(config: PipelineConfig, hot_k: float) -> float:
    out = run_frame(_planes(config, hot_k, person=True), config, PipelineState())
    assert out.display8 is not None
    y = out.display8[..., 0].astype(np.float64)
    h, w = config.sensor.sensor.fpa_shape
    person = y[h // 8 : h // 8 + h // 4, 3 * w // 4 : 3 * w // 4 + w // 8].mean()
    room = y[-6:, :6].mean()
    return float(abs(person - room))


def test_a_linear_agc_collapses_on_a_fire_and_plateau_equalisation_does_not(
    boson_lut: BandLUT,
) -> None:
    """`PH.8`'s acceptance, and FLIR's own observation, as four numbers.

    The comparison that makes it mean something is the *same frame without the fire*: a linear AGC
    is not a bad stretch, it is a perfectly good one being asked to span 1000 K. Plateau
    equalisation spends its output range on occupied bins instead of on the gap, so the two
    thousand pixels of room and person keep theirs.
    """
    ceiling = BOSON_GAIN_CEILING_K["low"]
    size = (64, 64)
    linear = _config(boson_lut, ceiling_k=ceiling, agc="linear", size=size, display=True)
    plateau = _config(
        boson_lut, ceiling_k=ceiling, agc="plateau_equalization", size=size, display=True
    )

    linear_calm = _person_room_contrast(linear, ROOM_K)
    linear_fire = _person_room_contrast(linear, FLAME_K)
    plateau_fire = _person_room_contrast(plateau, FLAME_K)

    assert linear_calm > 200.0, linear_calm  # the stretch is fine until the fire arrives
    assert linear_fire < 2.0, linear_fire  # FLIR: about 0.7 % of the range
    assert plateau_fire >= 10.0, plateau_fire
    assert plateau_fire > 20.0 * linear_fire, (plateau_fire, linear_fire)


def test_the_radiometric_branch_does_not_pass_through_the_agc(boson_lut: BandLUT) -> None:
    """Whatever the display does, the measurement is the same. The rail is the camera's only cap.

    Bit-for-bit across two AGC modes, because the radiometric branch inverts the ADC plane and the
    AGC is a display stretch downstream of it (ADR 0031). If this ever drifted, a scene's measured
    temperatures would depend on the palette someone chose.
    """
    ceiling = BOSON_GAIN_CEILING_K["low"]
    linear = _config(boson_lut, ceiling_k=ceiling, agc="linear", display=True)
    plateau = _config(boson_lut, ceiling_k=ceiling, agc="plateau_equalization", display=True)
    a = run_frame(_planes(linear, FLAME_K, person=True), linear, PipelineState())
    b = run_frame(_planes(plateau, FLAME_K, person=True), plateau, PipelineState())
    assert np.array_equal(a.apparent_t, b.apparent_t)
    assert not np.array_equal(a.display8, b.display8)


def test_a_low_gain_radiometric_config_gives_flame_3_s_histogram(boson_lut: BandLUT) -> None:
    """The shape FLAME 3's radiometric TIFFs have: an ambient mode and a tail piled at the rail.

    Built from FLAME 3's *published description* rather than its data — the dataset itself is
    `XD.5`'s, and until it is in hand this pins the shape a low-gain radiometric camera produces,
    not an agreement with the dataset. Two features, both of which a camera with no gain state
    would get wrong: a mode inside 0–25 °C, and a tail that stops dead at 500 °C instead of
    running on to the fire's real temperature.
    """
    ceiling = BOSON_GAIN_CEILING_K["low"]
    config = _config(boson_lut, ceiling_k=ceiling, size=(64, 64))
    h, w = config.sensor.sensor.fpa_shape
    rng = np.random.default_rng(20260922)
    t = np.float32(273.15 + rng.uniform(2.0, 22.0, size=(h, w)))  # the 0-25 C mode
    t[24:44, 20:48] = np.float32(rng.uniform(900.0, 1500.0, size=(20, 28)))  # the fire
    planes = {
        "temperature_k": t,
        "material_id": np.ones((h, w), dtype=np.uint16),
        "distance_m": np.full((h, w), 30.0, dtype=np.float32),
    }
    out = run_frame(planes, config, PipelineState())
    celsius = np.asarray(out.apparent_t, dtype=np.float64) - 273.15

    cool = celsius[celsius < 100.0]
    assert cool.size > 0.7 * celsius.size
    assert 0.0 < float(np.median(cool)) < 25.0, float(np.median(cool))

    tail = celsius[celsius > 100.0]
    assert tail.size > 0.1 * celsius.size
    assert float(tail.max()) == pytest.approx(500.0, abs=1.0)
    # Piled *against* the rail, not spread below it: that is what a ceiling looks like.
    assert float((tail > 495.0).mean()) > 0.9, float((tail > 495.0).mean())
