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
from irsim.thermal.frames import WorldFrame
from irsim.thermal.longwave import longwave_down
from irsim.thermal.shadow import ShadowRectangle, cell_shadow
from irsim.thermal.solar import solar_loading, sun_direction, sun_position_utc
from irsim.thermal.surface_field import PlanarPatch
from irsim.thermal.weather import WeatherSeries

__all__ = [
    "SurfaceOrientation",
    "SceneSurfaceForcing",
    "SolarTerms",
    "CellForcing",
    "sky_view_for_tilt",
    "solar_terms_at",
]


def sky_view_for_tilt(tilt_deg: Any) -> NDArray[np.float64]:
    """V_s = (1 + cos β)/2 — 1 facing up, 1/2 for a wall, 0 facing down (M7.13's relation)."""
    tilt = np.asarray(tilt_deg, dtype=np.float64)
    if np.any(tilt < 0.0) or np.any(tilt > 180.0):
        raise ValueError("tilt must lie in [0, 180] degrees from the up axis")
    return np.asarray(0.5 * (1.0 + np.cos(np.deg2rad(tilt))))


@dataclass(frozen=True)
class SolarTerms:
    """The sun at one instant, as every solar term in a scene sees it (PT.18).

    ``direction_enu`` points toward the sun; ``dni_w_m2`` is already zero below the horizon, so a
    consumer never has to remember the night rule for itself.
    """

    direction_enu: NDArray[np.float64]
    elevation_deg: float
    dni_w_m2: float
    dhi_w_m2: float

    @property
    def above_horizon(self) -> bool:
        return self.elevation_deg > 0.0


def solar_terms_at(
    weather: WeatherSeries, latitude_deg: float, longitude_deg: float, t_s: float
) -> SolarTerms:
    """M6.4's sun and the weather's irradiance at ``t_s`` seconds after the series' epoch.

    One function, so that the per-prim balance, the per-cell shadow and a driver's own field all
    ask the same question and get the same numbers -- the beam that lands on a bonnet in
    `car_demo` is the beam the road beside it sees, or a frame could show a sunlit road under an
    unlit car.
    """
    sample = weather.at(t_s)
    when = weather.epoch_utc + timedelta(seconds=float(t_s))
    sun = sun_position_utc(latitude_deg, longitude_deg, when)
    dni = 0.0 if sun.elevation_deg <= 0.0 else float(sample.dni_w_m2)
    return SolarTerms(
        direction_enu=sun_direction(sun.elevation_deg, sun.azimuth_deg),
        elevation_deg=float(sun.elevation_deg),
        dni_w_m2=dni,
        dhi_w_m2=float(sample.dhi_w_m2),
    )


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

    def solar_terms(self, t_s: float) -> SolarTerms:
        """The sun this forcing uses at ``t_s`` -- shared with the per-cell terms (PT.18)."""
        return solar_terms_at(self.weather, self.latitude_deg, self.longitude_deg, t_s)

    def __call__(self, t_s: float) -> FacetForcing:
        sample = self.weather.at(t_s)
        sun = self.solar_terms(t_s)
        normals = np.stack([o.normal_enu() for o in self.orientations])
        v_s = self.sky_view
        # Shadow gates the DIRECT beam only: a shaded facet still sees diffuse sky, and zeroing
        # all solar in shade is the usual shortcut that renders shaded surfaces far too cold.
        shade = np.array([0.0 if o.shaded else 1.0 for o in self.orientations])
        q_solar = solar_loading(normals, sun.direction_enu, sun.dni_w_m2, sun.dhi_w_m2, v_s, shade)

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
        # PH.1: what the latent term needs from the weather, on every forcing. With no wetness
        # declared and no film the balance never evaluates it, so a dry scene is unchanged.
        from irsim.thermal.latent import bulk_conductance_kg_m2_s, specific_humidity_kg_kg

        return FacetForcing(
            t_air_k=sample.t_air_k,
            h_w_m2_k=np.asarray(h, dtype=np.float64),
            q_solar_w_m2=np.asarray(q_solar, dtype=np.float64),
            q_longwave_down_w_m2=np.asarray(q_lw, dtype=np.float64),
            q_air_kg_kg=float(specific_humidity_kg_kg(sample.t_air_k, sample.rh_fraction)),
            g_e_kg_m2_s=float(bulk_conductance_kg_m2_s(sample.wind_speed_m_s)),
            precip_kg_m2_s=float(sample.precip_mm_h) / 3600.0,
        )


@dataclass
class CellForcing:
    """One surface's forcing, handed to every cell of its patch (PT.17), with the beam gated per
    cell when the scene declares occluders (PT.18).

    The per-prim solve and the per-cell solve must agree wherever nothing varies across the
    surface, or a patch would change a surface's temperature merely by existing. So this
    evaluates the scene's :class:`SceneSurfaceForcing` -- the same call the per-prim field makes
    -- takes the one surface's entry, and broadcasts it. The arithmetic the cells then do is
    element-for-element the arithmetic the prim does, which is why `test_scene_surface_fields`
    can hold the two to bit-identity rather than to a tolerance.

    **With occluders** the one term that varies is the direct beam. Each cell gets its own
    visibility from :func:`~irsim.thermal.shadow.cell_shadow`, in the patch's frame, and
    `solar_loading` is evaluated with the *surface's* normal and sky view and the *cell's*
    shadow. Where no occluder hits, that is the per-prim expression on the same operands in the
    same order, so unshaded cells stay bit-identical to the prim; where one does, the diffuse
    term survives and the beam is gone. The scene's ``shaded`` flag is the other authority on the
    same question, and a surface may not hold both.

    The per-cell sky view (PT.21) replaces the broadcast of ``q_longwave_down`` here in the same
    way, and nothing else moves.

    docs/physics-model.md §6.1; ADR 0087, ADR 0095
    """

    surfaces: SceneSurfaceForcing
    index: int
    n_cells: int
    #: The cell geometry, needed only when there is something to shade it with.
    patch: PlanarPatch | None = None
    #: Occluders in the patch's frame. Empty leaves the surface's own ``shaded`` flag in charge.
    occluders: tuple[ShadowRectangle, ...] = ()
    #: How the patch's world frame relates to ENU; required with occluders, ignored without.
    frame: WorldFrame | None = None

    def __post_init__(self) -> None:
        if not 0 <= self.index < self.surfaces.n_facets:
            raise IndexError(
                f"surface index {self.index} is outside the {self.surfaces.n_facets} surfaces"
            )
        if self.n_cells < 1:
            raise ValueError("a patch needs at least one cell")
        self.occluders = tuple(self.occluders)
        if not self.occluders:
            return
        if self.patch is None or self.frame is None:
            raise ValueError("occluders need the patch they shade and the scene's world frame")
        if self.patch.n_cells != self.n_cells:
            raise ValueError(
                f"patch has {self.patch.n_cells} cells, forcing expects {self.n_cells}"
            )
        if self.patch.frame != "world":
            raise ValueError(
                f"patch frame {self.patch.frame!r} is not 'world': shadow in a moving frame needs "
                "the frame's pose, which the thermal core does not carry (PT.9, WM)"
            )
        orientation = self.surfaces.orientations[self.index]
        if orientation.shaded:
            raise ValueError(
                "a surface cannot be both `shaded: true` and shaded per cell by occluders: two "
                "shadow authorities. Drop the flag or the occluders."
            )
        # The patch's plane must be the surface's plane, or the shadow would be computed on
        # cells that lie somewhere the balance's normal does not face.
        patch_normal_enu = self.frame.to_enu(self.patch.normal)
        alignment = abs(float(np.dot(patch_normal_enu, orientation.normal_enu())))
        if alignment < 1.0 - 1e-6:
            raise ValueError(
                f"the patch's plane (normal {tuple(np.round(patch_normal_enu, 6))} in ENU) does "
                f"not match the surface's tilt {orientation.tilt_deg} / azimuth "
                f"{orientation.azimuth_deg} (normal {tuple(np.round(orientation.normal_enu(), 6))})"
            )

    @property
    def weather(self) -> WeatherSeries:
        """The scene's one series, so the one-weather guard (CLAUDE.md #6) sees this too."""
        return self.surfaces.weather

    def cell_visibility(self, t_s: float) -> NDArray[np.float64]:
        """``(n_cells,)`` direct-beam visibility at ``t_s``; all ones without occluders."""
        if not self.occluders or self.patch is None or self.frame is None:
            return np.ones(self.n_cells)
        sun = self.surfaces.solar_terms(t_s)
        if not sun.above_horizon:
            return np.ones(self.n_cells)
        return cell_shadow(self.patch, self.frame.to_world(sun.direction_enu), self.occluders)

    def __call__(self, t_s: float) -> FacetForcing:
        t_air, h, q_solar, q_lw, q_int = self.surfaces(t_s).arrays(self.surfaces.n_facets)
        i, n = self.index, self.n_cells
        if self.occluders:
            sun = self.surfaces.solar_terms(t_s)
            orientation = self.surfaces.orientations[i]
            q_solar_cells = solar_loading(
                orientation.normal_enu(),
                sun.direction_enu,
                sun.dni_w_m2,
                sun.dhi_w_m2,
                float(self.surfaces.sky_view[i]),
                self.cell_visibility(t_s),
            )
        else:
            q_solar_cells = np.full(n, q_solar[i])
        base = self.surfaces(t_s)
        return FacetForcing(
            t_air_k=float(t_air[i]),
            h_w_m2_k=np.full(n, h[i]),
            q_solar_w_m2=np.asarray(q_solar_cells, dtype=np.float64),
            q_longwave_down_w_m2=np.full(n, q_lw[i]),
            q_internal_w_m2=np.full(n, q_int[i]),
            q_air_kg_kg=float(base.q_air_kg_kg),
            g_e_kg_m2_s=float(base.g_e_kg_m2_s),
            precip_kg_m2_s=float(base.precip_kg_m2_s),
        )
