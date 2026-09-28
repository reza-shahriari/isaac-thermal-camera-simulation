"""PT.28 -- a target has an altitude.

`site.altitude_m` reached nothing thermal and every aerial target read the surface T_air. A
target's `altitude_agl_m` now puts it in the column above the site: T_air(z) = T_air − Γ z with Γ
from the atmosphere preset, the speed of sound (and so the Mach number and the ram recovery) at
T(z), and the site's altitude thins the air the forced convection blows with. z = 0 is bit
for bit what it was. docs/physics-model.md §6.2, §6.5; ADR 0075; roadmap PT.28.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from irsim.radiometry.constants import (
    ISA_LAPSE_RATE_K_PER_M,
    R_SPECIFIC_AIR,
    STANDARD_GRAVITY_M_S2,
)
from irsim.scene import Scene
from irsim.thermal.aerial import (
    air_density_ratio,
    air_temperature_at_altitude_k,
    airframe_solver,
    heat_source_solver,
    ram_skin_solver,
    recovery_temperature_k,
    speed_of_sound_m_s,
)
from irsim.thermal.convection import DEFAULT_CONVECTION, forced_convection
from irsim.thermal.weather import WeatherSample, WeatherSeries

REPO = pathlib.Path(__file__).resolve().parents[2]
SCENE = REPO / "configs/scenes/sky_target_clear_day.yaml"


def _weather(t_air: float = 293.15) -> WeatherSeries:
    return WeatherSeries.constant(
        WeatherSample(t_air, 0.5, 2.0, 0.0, 0.0, 0.0, 23000.0, 0.0), 7200.0
    )


# --- the column -----------------------------------------------------------------------------------


def test_the_air_cools_with_the_presets_lapse_rate() -> None:
    assert air_temperature_at_altitude_k(293.15, 0.0, 0.0065) == 293.15
    assert air_temperature_at_altitude_k(293.15, 500.0, 0.0065) == pytest.approx(293.15 - 3.25)
    assert air_temperature_at_altitude_k(293.15, 3000.0, 0.0065) == pytest.approx(293.15 - 19.5)
    assert ISA_LAPSE_RATE_K_PER_M == 0.0065
    with pytest.raises(ValueError):
        air_temperature_at_altitude_k(293.15, -1.0, 0.0065)


def test_the_density_ratio_is_the_isa_column_to_one_percent() -> None:
    """ISA: 1.225 kg/m³ at sea level, 0.9093 at 3 km, 0.4135 at 10 km."""
    assert air_density_ratio(0.0) == 1.0
    assert air_density_ratio(3000.0) == pytest.approx(0.9093 / 1.225, rel=1e-2)
    assert air_density_ratio(10000.0) == pytest.approx(0.4135 / 1.225, rel=1e-2)
    # an isothermal column (no lapse) is the barometric exponential
    assert air_density_ratio(3000.0, lapse_rate_k_per_m=0.0) == pytest.approx(
        np.exp(-STANDARD_GRAVITY_M_S2 * 3000.0 / (R_SPECIFIC_AIR * 288.15)), rel=1e-9
    )


def test_forced_convection_blows_thinner_air_at_altitude_and_is_untouched_at_sea_level() -> None:
    v = 10.0
    sea = forced_convection(v, DEFAULT_CONVECTION)
    assert np.array_equal(forced_convection(v, DEFAULT_CONVECTION, density_ratio=1.0), sea)
    thin = forced_convection(v, DEFAULT_CONVECTION, density_ratio=0.742)
    expect = DEFAULT_CONVECTION.a + DEFAULT_CONVECTION.b * (0.742 * v) ** DEFAULT_CONVECTION.n
    assert thin == pytest.approx(expect, rel=1e-12) and thin < sea


# --- the targets ----------------------------------------------------------------------------------


def test_an_airframe_at_altitude_sits_at_the_air_up_there_and_at_zero_is_bit_identical() -> None:
    w = _weather()
    ground = airframe_solver(w)
    zero = airframe_solver(w, altitude_agl_m=0.0, lapse_rate_k_per_m=0.0065)
    high = airframe_solver(w, altitude_agl_m=500.0, lapse_rate_k_per_m=0.0065)
    for s in (ground, zero, high):
        s.advance(0.0, 600.0)
    assert zero.temperature() == ground.temperature()
    assert high.temperature() == pytest.approx(293.15 - 3.25, abs=2e-3)


def test_a_heat_source_at_altitude_rides_on_the_colder_air() -> None:
    from irsim.thermal.aerial import MOTOR

    w = _weather()
    low = heat_source_solver(MOTOR, w, [0.0, 3600.0], [1.0, 1.0])
    high = heat_source_solver(
        MOTOR, w, [0.0, 3600.0], [1.0, 1.0], altitude_agl_m=3000.0, lapse_rate_k_per_m=0.0065
    )
    low.advance(0.0, 600.0)
    high.advance(0.0, 600.0)
    assert low.temperature() == pytest.approx(293.15 + 45.0, abs=2e-3)
    assert high.temperature() == pytest.approx(293.15 - 19.5 + 45.0, abs=2e-3)


def test_the_ram_skin_recovers_at_the_speed_of_sound_of_its_own_altitude() -> None:
    w = _weather()
    v = 120.0
    low = ram_skin_solver(w, v)
    high = ram_skin_solver(w, v, altitude_agl_m=3000.0, lapse_rate_k_per_m=0.0065)
    low.advance(0.0, 600.0)
    high.advance(0.0, 600.0)
    t_z = 293.15 - 19.5
    mach_z = v / speed_of_sound_m_s(t_z)
    assert high.temperature() == pytest.approx(float(recovery_temperature_k(t_z, mach_z)), abs=2e-3)
    assert low.temperature() > high.temperature(), "colder air, colder skin, even at the same TAS"
    assert mach_z > v / speed_of_sound_m_s(293.15), "and a higher Mach number in the colder air"


# --- the scene ------------------------------------------------------------------------------------


def test_a_scene_target_takes_its_altitude_and_the_site_sets_the_density(  # type: ignore[no-untyped-def]
    tmp_path, tophat_lwir_lut
) -> None:
    """The sky-target scene sits at 520 m: its density ratio is 0.95 of sea level, and a target
    lifted 500 m above it reads the air 3 K colder by the preset's own lapse rate."""
    text = SCENE.read_text()
    assert text.count("{name: airframe, solver: newton") == 1
    lifted = text.replace(
        "{name: airframe, solver: newton",
        "{name: lifted, solver: airframe, offset_k: 0.0, altitude_agl_m: 500.0}\n"
        "    - {name: airframe, solver: newton",
    )
    path = tmp_path / "lifted.yaml"
    path.write_text(lifted)
    scene = Scene.from_file(path, {"lwir": tophat_lwir_lut})
    t_air = float(scene.weather.at(scene.t0_s).t_air_k)
    lapse = scene.lapse_rate_k_per_m
    assert lapse > 0.0
    assert scene.targets["lifted"].temperature() == pytest.approx(t_air - lapse * 500.0, abs=5e-3)
    assert scene.spec.site.altitude_m == 520.0
    expect = air_density_ratio(520.0, lapse_rate_k_per_m=lapse)
    assert scene.air_density_ratio == pytest.approx(expect) and 0.9 < expect < 1.0


def test_the_forcing_convects_less_in_thinner_air_and_the_same_at_sea_level() -> None:
    from irsim.thermal.scene_forcing import SceneSurfaceForcing, SurfaceOrientation

    def forcing(ratio: float) -> SceneSurfaceForcing:
        return SceneSurfaceForcing(
            weather=_weather(),
            latitude_deg=48.1,
            longitude_deg=11.6,
            orientations=(SurfaceOrientation(tilt_deg=0.0, azimuth_deg=0.0),),
            air_density_ratio=ratio,
        )

    sea = forcing(1.0)(600.0)
    default = SceneSurfaceForcing(
        weather=_weather(),
        latitude_deg=48.1,
        longitude_deg=11.6,
        orientations=(SurfaceOrientation(tilt_deg=0.0, azimuth_deg=0.0),),
    )(600.0)
    assert np.array_equal(np.asarray(sea.h_w_m2_k), np.asarray(default.h_w_m2_k))
    thin = forcing(air_density_ratio(3000.0))(600.0)
    v = 2.0  # the fixture's wind
    expect = DEFAULT_CONVECTION.a + DEFAULT_CONVECTION.b * (air_density_ratio(3000.0) * v) ** (
        DEFAULT_CONVECTION.n
    )
    assert float(np.asarray(thin.h_w_m2_k)[0]) == pytest.approx(expect, rel=1e-12)
    assert float(np.asarray(thin.h_w_m2_k)[0]) < float(np.asarray(sea.h_w_m2_k)[0])
