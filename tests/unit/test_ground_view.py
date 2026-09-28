"""PT.24 -- a down-facing surface sees the ground, not the air.

Every forcing path handed `longwave_down` the air temperature as the surround and `solar_loading`
had no albedo term, so a quadrotor's belly (tilt 180, V_s = 0) sat at T_air − 0.05 K through a
noon whose ground was 20–30 K warmer, and a ground camera looking up saw it 5–16 K too cold. The
forcing now carries the environment's ground temperature and the ground's shortwave albedo,
the latter through the tilted-surface model's (1 − cos β)/2 ground view.
docs/physics-model.md §6.1 (T_surround), §5.3; roadmap PT.24.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

from irsim.radiometry.constants import SIGMA_SB
from irsim.thermal import WeatherSample, WeatherSeries
from irsim.thermal.convection import DEFAULT_CONVECTION, forced_convection
from irsim.thermal.facets import FacetProperties, FacetSolver
from irsim.thermal.scene_forcing import SceneSurfaceForcing, SurfaceOrientation

T_AIR = 300.15
T_GROUND = T_AIR + 20.0


def _weather() -> WeatherSeries:
    # DNI 800, DHI 100, 1 m/s, dry, clear; the epoch puts t_s = 3600 near solar noon in June
    return WeatherSeries.constant(
        WeatherSample(T_AIR, 0.3, 1.0, 0.0, 800.0, 100.0, 23000.0, 0.0),
        6 * 3600.0,
        epoch_utc=datetime(2000, 6, 21, 10, 0, tzinfo=timezone.utc),
    )


def _forcing(tilt: float, ground: bool, albedo: float = 0.0) -> SceneSurfaceForcing:
    return SceneSurfaceForcing(
        weather=_weather(),
        latitude_deg=48.1,
        longitude_deg=11.6,
        orientations=(SurfaceOrientation(tilt_deg=tilt, azimuth_deg=0.0),),
        ground_temperature_k=(lambda t_s: T_GROUND) if ground else None,
        ground_albedo=albedo,
    )


def _properties() -> FacetProperties:
    return FacetProperties(
        heat_capacity_j_m2_k=np.array([1600.0 * 1050.0 * 0.002]),
        emissivity=np.array([0.887]),
        solar_absorptivity=np.array([0.9]),
    )


def _equilibrium(forcing: SceneSurfaceForcing, t_s: float = 3600.0) -> float:
    solver = FacetSolver(_properties(), T_AIR, guard=False)
    for _ in range(4000):
        solver.advance(forcing(t_s), 2.0)
    return float(solver.temperatures_k[0])


def test_an_up_facing_surface_is_bit_identical() -> None:
    """Tilt 0: V_s = 1 and (1 − cos β)/2 = 0, so neither term can reach it."""
    with_ground = _forcing(0.0, True, 0.25)(3600.0)
    without = _forcing(0.0, False)(3600.0)
    assert np.array_equal(with_ground.q_longwave_down_w_m2, without.q_longwave_down_w_m2)
    assert np.array_equal(with_ground.q_solar_w_m2, without.q_solar_w_m2)


def test_a_belly_over_warm_ground_is_warmer_by_the_hand_balance() -> None:
    """tilt 180 ((1 − cos β)/2 = 1): Q_LW↓ rises from σT_air⁴ to σT_ground⁴, +134 W/m² at
    300/320 K.
    The equilibrium moves by ε ΔQ / (h + 4εσT³) -- h = 9 at 1 m/s (above the free term), 4εσT³
    = 5.7, about +8.2 K -- and the solver agrees with that within 0.2 K. A belly sees neither the
    beam nor the sky, so the sun plays no part here."""
    forcing_air = _forcing(180.0, False)
    forcing_ground = _forcing(180.0, True)
    q_air = float(forcing_air(3600.0).q_longwave_down_w_m2[0])
    q_ground = float(forcing_ground(3600.0).q_longwave_down_w_m2[0])
    assert q_ground - q_air == pytest.approx(SIGMA_SB * (T_GROUND**4 - T_AIR**4), rel=1e-12)
    t_air_case = _equilibrium(forcing_air)
    t_ground_case = _equilibrium(forcing_ground)
    eps = 0.887
    t_mid = 0.5 * (t_air_case + t_ground_case)
    h_forced = float(forced_convection(1.0, DEFAULT_CONVECTION))
    h = max(h_forced, DEFAULT_CONVECTION.c * abs(t_mid - T_AIR) ** (1.0 / 3.0))
    delta_hand = eps * (q_ground - q_air) / (h + 4.0 * eps * SIGMA_SB * t_mid**3)
    assert t_ground_case - t_air_case == pytest.approx(delta_hand, abs=0.2)
    assert t_ground_case - t_air_case > 5.0


def test_reflected_sunlight_reaches_the_belly_only_through_the_ground() -> None:
    """(1 − cos β)/2 · albedo · GHI: for tilt 180 and albedo 0.25 that is 0.25·GHI, and nothing
    for a surface that sees no ground."""
    forcing = _forcing(180.0, True, 0.25)
    sun = forcing.solar_terms(3600.0)
    ghi = sun.dni_w_m2 * max(0.0, float(sun.direction_enu[2])) + sun.dhi_w_m2
    assert ghi > 300.0, "the fixture is a daytime one"
    belly = float(forcing(3600.0).q_solar_w_m2[0])
    belly_dark_ground = float(_forcing(180.0, True, 0.0)(3600.0).q_solar_w_m2[0])
    assert belly - belly_dark_ground == pytest.approx(0.25 * ghi, rel=1e-12)
    assert belly_dark_ground == pytest.approx(0.0, abs=1e-9), "a belly sees no sky and no beam"


def test_albedo_is_guarded() -> None:
    with pytest.raises(ValueError, match="ground_albedo"):
        _forcing(0.0, False, 1.5)
