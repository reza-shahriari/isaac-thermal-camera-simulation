"""AT.22 -- a sub-pixel target subtracts the pixel it sits in, not a clear sky.

`excess_radiance` under an atmosphere subtracted the clear column beyond R whatever the pixel
held. Over a clear-sky pixel that is the same number; over cloud, sea, terrain or another target
it is not, and a sub-pixel drone in front of cloud clutter (MS.3) came out about twice as bright
over its background as it should be. The fix is the blend `rotor_veil` already argues for:

    ΔL = φ [ τ_B(R) L_t + L_path(R) − L_pixel ].

docs/physics-model.md §5.1, §7.1; ADR 0071, ADR 0081; roadmap MS.6, AT.22.
"""

from __future__ import annotations

import copy
import math
import pathlib

import numpy as np
import pytest
import yaml

from irsim.atmosphere import Atmosphere
from irsim.atmosphere.layered import LayeredAtmosphere
from irsim.atmosphere.library import load_atmosphere_preset
from irsim.config.sensor import SensorConfig
from irsim.pipeline.point_target import (
    PointTarget,
    excess_radiance,
    fill_fraction,
    inject_point_targets,
)
from irsim.radiometry.lut import BandLUT
from irsim.thermal import WeatherSample, WeatherSeries

REPO = pathlib.Path(__file__).resolve().parents[2]
BOSON = yaml.safe_load((REPO / "configs" / "sensors" / "flir_boson_640_lwir.yaml").read_text())


def _sensor():  # type: ignore[no-untyped-def]
    d = copy.deepcopy(BOSON)
    d["sensor"]["fpa"].update(width=16, height=8)
    return SensorConfig.model_validate(d).sensor


def _weather() -> WeatherSeries:
    return WeatherSeries.constant(
        WeatherSample(288.15, 0.3, 1.0, 0.0, 0.0, 0.0, 23000.0, 0.0), 3600.0
    )


@pytest.fixture(scope="module")
def layered(tophat_lwir_lut: BandLUT) -> LayeredAtmosphere:
    return LayeredAtmosphere(
        load_atmosphere_preset("us_standard_clear"), _weather(), {"lwir": tophat_lwir_lut}
    )


def _area_for(sensor, phi: float, range_m: float) -> float:  # type: ignore[no-untyped-def]
    f = sensor.optics.focal_length_mm * 1e-3
    return phi * range_m**2 * sensor.pixel_area_m2 / f**2


def test_over_a_clear_sky_pixel_the_two_forms_agree(
    layered: LayeredAtmosphere, tophat_lwir_lut: BandLUT
) -> None:
    """The pixel-aware form reduces to the per-class clear-column form when the pixel holds the
    clear column: what every existing scene had, so their frames do not move."""
    sensor = _sensor()
    theta = math.radians(15.0)
    lt = float(tophat_lwir_lut.lookup(np.float64(320.0))[()])
    for r in (200.0, 700.0, 3000.0):
        t = PointTarget(_area_for(sensor, 0.278, r), r, lt, (4.5, 3.5), theta)
        old = excess_radiance(t, sensor, layered, "lwir", 0.0, tophat_lwir_lut)
        sky = layered.sky_radiance("lwir", 0.0, theta)
        new = excess_radiance(t, sensor, layered, "lwir", 0.0, tophat_lwir_lut, "lb", sky)
        assert new == pytest.approx(old, rel=1e-9)


def test_over_a_cloud_pixel_the_excess_is_what_replaces_the_pixel(
    layered: LayeredAtmosphere, tophat_lwir_lut: BandLUT
) -> None:
    """Measured before the fix: over a cloud pixel at 38.66 W/m²/sr with the clear sky at 17.92,
    a φ = 0.278 target at 700 m was 2× too bright (+5.8 W/m²/sr). Now the excess is the hand
    value, and it is smaller than the clear-sky excess by φ (L_cloud − L_clear)."""
    sensor = _sensor()
    theta = math.radians(15.0)
    r = 700.0
    phi = 0.278
    lt = float(tophat_lwir_lut.lookup(np.float64(320.0))[()])
    t = PointTarget(_area_for(sensor, phi, r), r, lt, (4.5, 3.5), theta)
    clear = layered.sky_radiance("lwir", 0.0, theta)
    cloud = clear + 20.0
    es = layered.exponential_sum("lwir", 0.0)
    tau_band = float(np.dot(es.weights, layered.class_transmittances("lwir", 0.0, r, theta)))
    path = layered.path_radiance("lwir", 0.0, r, theta)
    phi_exact = fill_fraction(
        t.area_m2, r, sensor.optics.focal_length_mm * 1e-3, sensor.pixel_area_m2
    )
    expect = phi_exact * (tau_band * lt + path - cloud)
    got = excess_radiance(t, sensor, layered, "lwir", 0.0, tophat_lwir_lut, "lb", cloud)
    assert got == pytest.approx(expect, rel=1e-12)
    over_clear = excess_radiance(t, sensor, layered, "lwir", 0.0, tophat_lwir_lut, "lb", clear)
    assert over_clear - got == pytest.approx(phi_exact * 20.0, rel=1e-9)
    # a target that looks exactly like the cloud through its own air adds nothing
    lt_same = (cloud - path) / tau_band
    same = PointTarget(t.area_m2, r, lt_same, (4.5, 3.5), theta)
    assert (
        abs(excess_radiance(same, sensor, layered, "lwir", 0.0, tophat_lwir_lut, "lb", cloud))
        < 1e-9 * cloud
    )


def test_the_grey_atmosphere_keeps_its_own_column(tophat_lwir_lut: BandLUT) -> None:
    """The grey L1 model's path radiance tends to L_B(T_air) while its sky pixels come from the
    `SkyModel` (ADR 0050): not one column, so the pixel cannot say what lies beyond R and the
    clear-column form stays, whatever the pixel holds (spec issue S28; `AT.5`)."""
    sensor = _sensor()
    grey = Atmosphere(
        load_atmosphere_preset("us_standard_clear"), _weather(), {"lwir": tophat_lwir_lut}
    )
    lt = float(tophat_lwir_lut.lookup(np.float64(320.0))[()])
    t = PointTarget(0.02, 1000.0, lt, (4.5, 3.5))
    old = excess_radiance(t, sensor, grey, "lwir", 0.0, tophat_lwir_lut)
    for pixel in (grey.air_radiance("lwir", 0.0), 20.0, 60.0):
        got = excess_radiance(t, sensor, grey, "lwir", 0.0, tophat_lwir_lut, "lb", pixel)
        assert got == pytest.approx(old, rel=1e-12)


def test_injection_reads_the_pixel_under_the_target(
    layered: LayeredAtmosphere, tophat_lwir_lut: BandLUT
) -> None:
    """On the plane, the excess over a bright pixel is smaller than over a dark one by exactly
    φ times the difference, and the frame sum is the excess (flux conserved)."""
    sensor = _sensor()
    theta = math.radians(15.0)
    r = 700.0
    lt = float(tophat_lwir_lut.lookup(np.float64(320.0))[()])
    t = PointTarget(_area_for(sensor, 0.278, r), r, lt, (4.5, 3.5), theta)
    clear = layered.sky_radiance("lwir", 0.0, theta)
    plane_clear = np.full((8, 16), clear, dtype=np.float32)
    plane_cloud = plane_clear.copy()
    plane_cloud[3, 4] = clear + 20.0
    out_clear = inject_point_targets(
        plane_clear, [t], sensor, layered, "lwir", 0.0, tophat_lwir_lut, "lb", 1
    )
    out_cloud = inject_point_targets(
        plane_cloud, [t], sensor, layered, "lwir", 0.0, tophat_lwir_lut, "lb", 1
    )
    phi = fill_fraction(t.area_m2, r, sensor.optics.focal_length_mm * 1e-3, sensor.pixel_area_m2)
    added_clear = float((out_clear.astype(np.float64) - plane_clear).sum())
    added_cloud = float((out_cloud.astype(np.float64) - plane_cloud).sum())
    assert added_clear - added_cloud == pytest.approx(phi * 20.0, rel=1e-5)
    hand = excess_radiance(t, sensor, layered, "lwir", 0.0, tophat_lwir_lut, "lb", clear + 20.0)
    assert added_cloud == pytest.approx(hand, rel=1e-5)
