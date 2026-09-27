"""The ADC floor holds the coldest sky (docs/physics-model.md §11.1, spec issue S53, `SC.23`).

A bolometer's DN 0 used to sit at the datasheet's −40 °C (`RADIOMETRIC_RANGE_K`), so every scene
colder than that read one code: up to two thirds of a `phantom4_perpart` LWIR frame, and one
enormous bin for the AGC. ADR 0151 moves DN 0 to **zero scene radiance**. The roadmap row asked for
the band LUT's 200 K; the preset walk below is why that was not enough -- the layered model puts a
clear winter zenith sky well under L_B(200 K), in every preset.

Each test fails if the floor goes back up: to −40 °C (the red reproduction and the gain ratio), to
200 K (the preset walk), or anywhere a round trip or the ceiling would notice.
"""

from __future__ import annotations

import copy
import math
import pathlib
from typing import Any

import numpy as np
import pytest
import yaml

from irsim.atmosphere.layered import LayeredAtmosphere
from irsim.atmosphere.library import load_atmosphere_preset
from irsim.config.sensor import SensorConfig
from irsim.isp.radiometric import apparent_temperature_from_dn, dn_per_kelvin
from irsim.materials import MaterialTable
from irsim.pipeline import PipelineConfig
from irsim.pipeline.core import ADC_FLOOR_RADIANCE, RADIOMETRIC_RANGE_K
from irsim.radiometry.lut import BandLUT
from irsim.radiometry.spectral_response import SpectralResponse
from irsim.thermal import WeatherSample, WeatherSeries

REPO = pathlib.Path(__file__).resolve().parents[2]
H, W = 48, 64
CENTRE = (H // 2, W // 2)


def _sensor(name: str = "flir_boson_640_lwir") -> SensorConfig:
    d: dict[str, Any] = copy.deepcopy(
        yaml.safe_load((REPO / "configs" / "sensors" / f"{name}.yaml").read_text())
    )
    d["sensor"]["fpa"].update(width=W, height=H)
    d["sensor"]["optics"]["supersample_factor"] = 1
    return SensorConfig.model_validate(d)


def _config(lut: BandLUT, floor: float | None = ADC_FLOOR_RADIANCE) -> PipelineConfig:
    return PipelineConfig.from_sensor(
        _sensor(), MaterialTable.constant(1.0), lut=lut, adc_floor_radiance=floor
    )


def _dn(config: PipelineConfig, radiance: np.ndarray) -> np.ndarray:
    """Noiseless quantised DN of a scene radiance plane, through the camera's own transfer."""
    cal = config.calibration
    assert cal is not None
    signal = cal.signal_from_radiance(radiance.astype(np.float32))
    return cal.transfer.dn(cal.transfer.power_from_signal_w(signal))


def _sky_ramp(lut: BandLUT) -> tuple[np.ndarray, np.ndarray]:
    """A −68 … −41 °C sky down the rows: zenith to ~15° on a clear dry day (§11.1)."""
    t = np.repeat(np.linspace(205.0, 232.0, H)[:, None], W, axis=1)
    return t, np.asarray(lut.lookup(t), dtype=np.float32)


def test_the_floor_is_zero_radiance_and_the_datasheet_range_is_unchanged() -> None:
    assert ADC_FLOOR_RADIANCE == 0.0
    # the range still names the datasheet span: its ends are the flat field's blackbodies
    assert RADIOMETRIC_RANGE_K == (233.15, 473.15)


def test_a_minus_60_c_sky_quantises_to_one_code_only_under_the_old_floor(
    boson_lut: BandLUT,
) -> None:
    """Red reproduction: the pre-SC.23 transfer maps the whole cold sky to DN 0."""
    t, sky = _sky_ramp(boson_lut)
    old = _dn(_config(boson_lut, floor=None), sky)
    new = _dn(_config(boson_lut), sky)

    assert np.unique(old).tolist() == [0]
    assert not np.any(new == 0)
    column = new[:, W // 2].astype(np.int64)
    assert np.all(np.diff(column) > 0), "a colder row must read a lower code"
    # 27 K of sky at >= 36 DN/K (the 200 K slope) is ~1000 codes; the old transfer gave one
    assert column[-1] - column[0] > 900


def test_the_coldest_modelled_zenith_sky_stays_on_scale(
    boson_lut: BandLUT, boson_response: SpectralResponse
) -> None:
    """Walk every shipped atmosphere preset through a cold-to-warm, dry-to-humid weather grid.

    Two claims. The zero floor holds every zenith sky well above DN 0 -- by far more than the
    ~9 DN noise σ, so noise cannot clip it either. And the band LUT's 200 K would **not** have:
    some of these skies are below L_B(200 K), which is the reason the floor is zero.
    """
    config = _config(boson_lut)
    band = config.sensor.sensor.band.band_id
    l_200 = float(boson_lut.lookup(200.0)[()])
    coldest = math.inf
    below_lut = 0
    presets = sorted(p.stem for p in (REPO / "configs" / "atmospheres").glob("*.yaml"))
    assert len(presets) >= 7
    for name in presets:
        preset = load_atmosphere_preset(name)
        if band not in preset.bands:
            continue
        for t_air in (233.15, 253.15, 273.15, 303.15):
            for rh in (0.02, 0.5):
                weather = WeatherSeries.constant(
                    WeatherSample(t_air, rh, 2.0, 0.0, 0.0, 0.0, 23000.0, 0.0), 3600.0
                )
                atm = LayeredAtmosphere(preset, weather, {band: boson_lut}, {band: boson_response})
                radiance = float(atm.sky_radiance(band, 0.0, math.radians(90.0)))
                below_lut += radiance < l_200
                signal = _dn(config, np.full((H, W), radiance))[CENTRE]
                coldest = min(coldest, float(signal))
    assert below_lut > 0, "no preset sky fell below the LUT floor; ADR 0151's premise changed"
    assert coldest > 100.0


def test_zero_radiance_is_dn_zero_and_the_gain_moves_by_the_closed_form(
    boson_lut: BandLUT,
) -> None:
    """Φ is affine in scene radiance at the centre, so moving DN 0 from L_B(233.15 K) to zero
    rescales the gain by exactly (L_top − L_233) / (L_top − 0)."""
    new, old = _config(boson_lut), _config(boson_lut, floor=None)
    assert new.calibration is not None and old.calibration is not None
    zero = new.calibration.signal_from_radiance(np.zeros((H, W), np.float32))[CENTRE]
    assert abs(float(zero)) < 1e-2

    l_lo, l_hi = (float(boson_lut.lookup(t)[()]) for t in RADIOMETRIC_RANGE_K)
    ratio = new.calibration.transfer.gain_dn_per_w / old.calibration.transfer.gain_dn_per_w
    assert ratio == pytest.approx((l_hi - l_lo) / l_hi, rel=1e-6)
    # measured: 177.95 -> 170.33 DN/K at 300 K, 5.9 mK a code -- still far inside a 50 mK NETD
    assert 1000.0 / dn_per_kelvin(new.calibration, boson_lut, 300.0) < 10.0


def test_the_top_code_still_sits_at_the_top_of_the_range(boson_lut: BandLUT) -> None:
    config = _config(boson_lut)
    assert config.calibration is not None
    top = float(boson_lut.lookup(RADIOMETRIC_RANGE_K[1])[()])
    signal = config.calibration.signal_from_radiance(np.full((H, W), top, np.float32))[CENTRE]
    dn_max = 2**config.sensor.sensor.fpa.bit_depth - 1
    assert float(signal) == pytest.approx(dn_max, abs=0.05)


def test_dn_round_trips_to_apparent_temperature_inside_half_an_lsb(boson_lut: BandLUT) -> None:
    """Every scene the LUT holds: DN16 → T_app lands within half a code of the truth.

    Under the old floor everything under 233.15 K came back as 233.15 K -- 33 K out at 200 K.
    """
    config = _config(boson_lut)
    assert config.calibration is not None
    for t in np.linspace(200.3, 470.0, 55):
        radiance = np.full((H, W), float(boson_lut.lookup(t)[()]), np.float32)
        t_app = apparent_temperature_from_dn(_dn(config, radiance), config.calibration, boson_lut)
        half_lsb_k = 0.5 / dn_per_kelvin(config.calibration, boson_lut, float(t))
        assert abs(float(t_app[CENTRE]) - t) <= half_lsb_k + 1e-3, t
