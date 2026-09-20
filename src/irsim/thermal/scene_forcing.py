"""Turning a scene's weather and a surface's orientation into §6.1's forcing, at any instant.

This is the join between the parts: the shared `WeatherSeries` (CLAUDE.md #6), M6.4's NOAA sun,
M6.5's downwelling longwave and M6.6's convection, evaluated for a list of oriented facets. It
exists as an object rather than a closure so that it can carry ``.weather`` and be checked by the
`Scene`'s one-weather guard -- a forcing model quietly holding a second weather series is exactly
the failure that guard is for.

Three things it is careful about:

* **Shadow is a per-facet flag, not a global one.** §6.6's "shaded asphalt" is the same material
  as the sunlit asphalt beside it and differs only here, which is what makes the pair a usable
  test of the solver rather than of the material library.
* **Sky view follows the tilt**, V_s = (1 + cos β)/2, so a wall sees half the sky and half the
  ground -- the same relation stage 1 uses (M7.13), not a second convention.
* **Diffuse sunlight is not shadowed.** A shaded facet still sees the sky's diffuse component;
  zeroing all solar in shade makes shaded surfaces far too cold and is the usual shortcut.

docs/physics-model.md §6.1, §6.4, §6.6; ADR 0032, ADR 0037
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import numpy as np
from numpy.typing import NDArray

from irsim.thermal.convection import DEFAULT_CONVECTION, ConvectionParams, convection_coefficient
from irsim.thermal.facets import FacetForcing
from irsim.thermal.longwave import longwave_down
from irsim.thermal.solar import solar_loading, sun_direction, sun_position_utc
from irsim.thermal.weather import WeatherSeries

__all__ = ["SurfaceOrientation", "SceneSurfaceForcing", "CellForcing", "sky_view_for_tilt"]


def sky_view_for_tilt(tilt_deg: Any) -> NDArray[np.float64]:
    """V_s = (1 + cos β)/2 — 1 facing up, 1/2 for a wall, 0 facing down (M7.13's relation)."""
    tilt = np.asarray(tilt_deg, dtype=np.float64)
    if np.any(tilt < 0.0) or np.any(tilt > 180.0):
        raise ValueError("tilt must lie in [0, 180] degrees from the up axis")
    return np.asarray(0.5 * (1.0 + np.cos(np.deg2rad(tilt))))


@dataclass(frozen=True)
class SurfaceOrientation:
    """Where one facet faces, and whether anything stands between it and the sun."""

    tilt_deg: float = 0.0
    azimuth_deg: float = 180.0
    shaded: bool = False
    vehicle_speed_m_s: float = 0.0

    def normal_enu(self) -> NDArray[np.float64]:
        tilt = math.radians(self.tilt_deg)
        azimuth = math.radians(self.azimuth_deg)
        return np.array(
            [math.sin(tilt) * math.sin(azimuth), math.sin(tilt) * math.cos(azimuth), math.cos(tilt)]
        )


@dataclass
class SceneSurfaceForcing:
    """Callable ``t_s -> FacetForcing`` for a fixed set of oriented surfaces."""

    weather: WeatherSeries
    latitude_deg: float
    longitude_deg: float
    orientations: tuple[SurfaceOrientation, ...]
    convection: ConvectionParams = DEFAULT_CONVECTION
    surface_temperature_k: float = 300.0

    def __post_init__(self) -> None:
        if not self.orientations:
            raise ValueError("a forcing model needs at least one surface")

    @property
    def n_facets(self) -> int:
        return len(self.orientations)

    @property
    def sky_view(self) -> NDArray[np.float64]:
        return sky_view_for_tilt([o.tilt_deg for o in self.orientations])

    def __call__(self, t_s: float) -> FacetForcing:
        sample = self.weather.at(t_s)
        when = self.weather.epoch_utc + timedelta(seconds=float(t_s))
        sun = sun_position_utc(self.latitude_deg, self.longitude_deg, when)
        direction = sun_direction(sun.elevation_deg, sun.azimuth_deg)
        normals = np.stack([o.normal_enu() for o in self.orientations])
        v_s = self.sky_view
        # Shadow gates the DIRECT beam only: a shaded facet still sees diffuse sky, and zeroing
        # all solar in shade is the usual shortcut that renders shaded surfaces far too cold.
        shade = np.array([0.0 if o.shaded else 1.0 for o in self.orientations])
        dni = 0.0 if sun.elevation_deg <= 0.0 else sample.dni_w_m2
        q_solar = solar_loading(normals, direction, dni, sample.dhi_w_m2, v_s, shade)

        q_lw = longwave_down(
            sample.t_air_k,
            sample.vapour_pressure_hpa,
            sample.cloud_fraction,
            v_s,
            sample.t_air_k,
        )
        speeds = np.array([o.vehicle_speed_m_s for o in self.orientations])
        h = convection_coefficient(
            self.surface_temperature_k - sample.t_air_k,
            sample.wind_speed_m_s,
            speeds,
            self.convection,
        )
        return FacetForcing(
            t_air_k=sample.t_air_k,
            h_w_m2_k=np.asarray(h, dtype=np.float64),
            q_solar_w_m2=np.asarray(q_solar, dtype=np.float64),
            q_longwave_down_w_m2=np.asarray(q_lw, dtype=np.float64),
        )


@dataclass
class CellForcing:
    """One surface's forcing, handed to every cell of its patch (PT.17).

    The per-prim solve and the per-cell solve must agree wherever nothing varies across the
    surface, or a patch would change a surface's temperature merely by existing. So this does
    not compute anything: it evaluates the scene's :class:`SceneSurfaceForcing` -- the same call
    the per-prim field makes -- takes the one surface's entry, and broadcasts it. The arithmetic
    the cells then do is element-for-element the arithmetic the prim does, which is why
    `test_scene_surface_fields` can hold the two to bit-identity rather than to a tolerance.

    It is also the seam the spatial terms plug into: per-cell shadow (PT.18) and sky view (PT.21)
    replace the broadcast of ``q_solar`` and ``q_longwave_down`` here, and nothing else moves.

    docs/physics-model.md §6.1; ADR 0087
    """

    surfaces: SceneSurfaceForcing
    index: int
    n_cells: int

    def __post_init__(self) -> None:
        if not 0 <= self.index < self.surfaces.n_facets:
            raise IndexError(
                f"surface index {self.index} is outside the {self.surfaces.n_facets} surfaces"
            )
        if self.n_cells < 1:
            raise ValueError("a patch needs at least one cell")

    @property
    def weather(self) -> WeatherSeries:
        """The scene's one series, so the one-weather guard (CLAUDE.md #6) sees this too."""
        return self.surfaces.weather

    def __call__(self, t_s: float) -> FacetForcing:
        t_air, h, q_solar, q_lw, q_int = self.surfaces(t_s).arrays(self.surfaces.n_facets)
        i, n = self.index, self.n_cells
        return FacetForcing(
            t_air_k=float(t_air[i]),
            h_w_m2_k=np.full(n, h[i]),
            q_solar_w_m2=np.full(n, q_solar[i]),
            q_longwave_down_w_m2=np.full(n, q_lw[i]),
            q_internal_w_m2=np.full(n, q_int[i]),
        )
