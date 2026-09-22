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
from irsim.thermal.frames import ENU, WorldFrame
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
    "MeshCellForcing",
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
    #: A speed that changes with the mission (PT.9): ``(times, speeds)`` on the weather's own
    #: axis, linearly interpolated and held flat outside. A drone on its pad cools by free
    #: convection and the same drone in a climb by forced -- the difference is tens of kelvin on
    #: a sunlit deck, and it is the flight that decides which, not the surface.
    speed_schedule: tuple[tuple[float, ...], tuple[float, ...]] | None = None

    def __post_init__(self) -> None:
        if self.speed_schedule is None:
            return
        times, speeds = self.speed_schedule
        if len(times) != len(speeds) or not times:
            raise ValueError("a speed schedule needs matching, non-empty times and speeds")
        if any(b <= a for a, b in zip(times[:-1], times[1:], strict=True)):
            raise ValueError("a speed schedule's times must be strictly increasing")
        if any(v < 0.0 for v in speeds):
            raise ValueError("a speed cannot be negative")
        if self.vehicle_speed_m_s:
            raise ValueError(
                "a surface has one speed authority: `vehicle_speed_m_s` or a schedule, not both"
            )

    def speed_at(self, t_s: float) -> float:
        """The speed this facet meets the air at, at ``t_s``."""
        if self.speed_schedule is None:
            return float(self.vehicle_speed_m_s)
        times, speeds = self.speed_schedule
        return float(np.interp(t_s, np.asarray(times), np.asarray(speeds)))

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
        speeds = np.array([o.speed_at(t_s) for o in self.orientations])
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
    #: Per-cell sky view factors (PT.21), ``None`` for the tilt's own on every cell. Scales the
    #: diffuse solar and the longwave down; ``sky_view_longwave=False`` is the negative control
    #: the roadmap names (a factor on solar alone), for tests.
    sky_view: Any = None
    sky_view_longwave: bool = True
    #: Rays across the sun's 0.53 deg disc (PT.22): 1 is PT.18's hard edge, bit for bit; 7, 19
    #: or 37 turn the beam's visibility into a sunlit fraction and the terminator into a ramp
    #: `d tan(0.53 deg)` wide. Costs one ray-occluder pass per ray per forcing evaluation.
    penumbra_rays: int = 1

    def __post_init__(self) -> None:
        if not 0 <= self.index < self.surfaces.n_facets:
            raise IndexError(
                f"surface index {self.index} is outside the {self.surfaces.n_facets} surfaces"
            )
        if self.n_cells < 1:
            raise ValueError("a patch needs at least one cell")
        self.occluders = tuple(self.occluders)
        if self.sky_view is not None:
            svf = np.asarray(self.sky_view, dtype=np.float64).reshape(-1)
            if svf.shape != (self.n_cells,):
                raise ValueError(f"sky_view has shape {svf.shape}, expected ({self.n_cells},)")
            if np.any((svf < 0.0) | (svf > 1.0)):
                raise ValueError("sky view factors must lie in [0, 1]")
            self.sky_view = svf
        if self.penumbra_rays < 1:
            raise ValueError("penumbra_rays must be at least 1 (1 is the hard-edged shadow)")
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
        """``(n_cells,)`` direct-beam visibility at ``t_s``; the surface's own ``shaded`` flag,
        broadcast, without occluders."""
        if not self.occluders or self.patch is None or self.frame is None:
            shaded = self.surfaces.orientations[self.index].shaded
            return np.full(self.n_cells, 0.0 if shaded else 1.0)
        sun = self.surfaces.solar_terms(t_s)
        if not sun.above_horizon:
            return np.ones(self.n_cells)
        toward_sun = self.frame.to_world(sun.direction_enu)
        if self.penumbra_rays > 1:  # PT.22: the sun as a disc, so the edge is a ramp
            from irsim.thermal.raycast import sunlit_fraction

            return sunlit_fraction(self.patch, toward_sun, self.occluders, self.penumbra_rays)
        return cell_shadow(self.patch, toward_sun, self.occluders)

    def __call__(self, t_s: float) -> FacetForcing:
        t_air, h, q_solar, q_lw, q_int = self.surfaces(t_s).arrays(self.surfaces.n_facets)
        i, n = self.index, self.n_cells
        v_s: Any = float(self.surfaces.sky_view[i]) if self.sky_view is None else self.sky_view
        if self.occluders or self.sky_view is not None:
            sun = self.surfaces.solar_terms(t_s)
            orientation = self.surfaces.orientations[i]
            q_solar_cells = solar_loading(
                orientation.normal_enu(),
                sun.direction_enu,
                sun.dni_w_m2,
                sun.dhi_w_m2,
                v_s,
                self.cell_visibility(t_s),
            )
        else:
            q_solar_cells = np.full(n, q_solar[i])
        if self.sky_view is not None and self.sky_view_longwave:
            # The longwave down through the cell's own share of the dome, the rest of the
            # hemisphere at the air temperature -- the per-prim call with the cell's factor.
            sample = self.surfaces.weather.at(t_s)
            q_lw_cells = np.asarray(
                longwave_down(
                    sample.t_air_k,
                    sample.vapour_pressure_hpa,
                    sample.cloud_fraction,
                    self.sky_view,
                    sample.t_air_k,
                ),
                dtype=np.float64,
            )
        else:
            q_lw_cells = np.full(n, q_lw[i])
        base = self.surfaces(t_s)
        return FacetForcing(
            t_air_k=float(t_air[i]),
            h_w_m2_k=np.full(n, h[i]),
            q_solar_w_m2=np.asarray(q_solar_cells, dtype=np.float64),
            q_longwave_down_w_m2=q_lw_cells,
            q_internal_w_m2=np.full(n, q_int[i]),
            q_air_kg_kg=float(base.q_air_kg_kg),
            g_e_kg_m2_s=float(base.g_e_kg_m2_s),
            precip_kg_m2_s=float(base.precip_kg_m2_s),
        )


@dataclass
class MeshCellForcing:
    """Per-cell forcing on a **mesh**: each cell's own face normal drives its solar term (WM.7).

    :class:`CellForcing` gives every cell of a planar patch the surface's one normal, which is
    right for a plane and is exactly what a curved surface cannot have. Here the normal comes
    from the cell's own triangle, so the sunlit side of a pipe and its shaded side are different
    cells of one solve -- which is the whole reason `WM.2`'s field exists.

    Two terms follow the normal and nothing else changes: the direct beam through
    :func:`~irsim.thermal.solar.solar_loading`'s ``max(0, n·s)``, and each cell's sky view
    ``V_s = (1 + n·up)/2``, which scales the diffuse solar and the longwave down together exactly
    as `PT.21` does for a patch.

    **With ``patch`` given, both stop being analytic (`WM.4`).** The beam is traced per cell
    against the scene's occluders and against the mesh's own triangles, so a cell another part
    of the object hides goes dark; and ``sky_view`` carries
    :func:`~irsim.thermal.mesh_geometry.mesh_sky_view`'s traced factor, so a cell tucked under an
    airframe loses the diffuse sun and the longwave it cannot see. Without a patch this is
    `WM.7`'s convex-geometry form -- the cosine, and the surface's one ``shaded`` flag -- which a
    convex mesh with nothing around it makes exactly right, and `mesh_geometry` skips the trace
    for that reason rather than for speed.
    """

    surfaces: SceneSurfaceForcing
    index: int
    #: ``(n_cells, 3)`` unit outward normals in the **scene's world frame**.
    normals_world: Any
    frame: WorldFrame = ENU
    sky_view_longwave: bool = True
    #: The cell geometry. Needed only to trace: without it the beam falls back to the surface's
    #: own ``shaded`` flag, as `WM.7` shipped it.
    patch: Any = None
    #: Scene occluders in the patch's frame. The mesh's own triangles are added by
    #: `mesh_geometry.cell_occluders` when the mesh is not convex, so an empty tuple here still
    #: traces a mesh that folds over itself.
    occluders: tuple[ShadowRectangle, ...] = ()
    #: Per-cell sky view factors (`WM.4`); ``None`` leaves each cell the tilt's own
    #: ``(1 + n·up)/2``, which is the unobstructed answer.
    sky_view: Any = None
    #: Rays across the sun's 0.53 deg disc (`PT.22`): 1 is the hard edge, 7/19/37 a ramp.
    penumbra_rays: int = 1

    def __post_init__(self) -> None:
        n = np.asarray(self.normals_world, dtype=np.float64)
        if n.ndim != 2 or n.shape[1] != 3 or n.shape[0] < 1:
            raise ValueError(f"normals_world must be (n_cells >= 1, 3), got {n.shape}")
        norms = np.linalg.norm(n, axis=-1)
        if np.any(norms <= 0.0):
            raise ValueError("a cell normal has zero length")
        if not 0 <= self.index < self.surfaces.n_facets:
            raise IndexError(
                f"surface index {self.index} is outside the {self.surfaces.n_facets} surfaces"
            )
        self.normals_world = n / norms[:, None]
        enu = np.asarray(self.frame.to_enu(self.normals_world), dtype=np.float64)
        object.__setattr__(self, "_normals_enu", enu)
        analytic = np.clip(0.5 * (1.0 + enu[:, 2]), 0.0, 1.0)
        if self.penumbra_rays < 1:
            raise ValueError("penumbra_rays must be at least 1 (1 is the hard-edged shadow)")
        if self.sky_view is not None:
            svf = np.asarray(self.sky_view, dtype=np.float64).reshape(-1)
            if svf.shape != (n.shape[0],):
                raise ValueError(f"sky_view has shape {svf.shape}, expected ({n.shape[0]},)")
            if np.any((svf < 0.0) | (svf > 1.0)):
                raise ValueError("sky view factors must lie in [0, 1]")
            analytic = svf
        #: After init this always holds the ``(n_cells,)`` factor the balance uses -- the traced
        #: one when the caller supplied it, the tilt's own otherwise.
        self.sky_view = analytic
        self.occluders = tuple(self.occluders)
        object.__setattr__(self, "_query", None)
        object.__setattr__(self, "_origins", None)
        if self.patch is None:
            if self.occluders:
                raise ValueError("occluders need the mesh patch they shade (WM.4)")
            return
        if self.patch.n_cells != n.shape[0]:
            raise ValueError(
                f"patch has {self.patch.n_cells} cells, forcing has {n.shape[0]} normals"
            )
        if self.patch.frame != "world":
            raise ValueError(
                f"patch frame {self.patch.frame!r} is not 'world': shadow in a moving frame needs "
                "the frame's pose, which the thermal core does not carry (PT.9, WM)"
            )
        if self.occluders and self.surfaces.orientations[self.index].shaded:
            raise ValueError(
                "a surface cannot be both `shaded: true` and shaded per cell by occluders: two "
                "shadow authorities. Drop the flag or the occluders."
            )
        from irsim.thermal.mesh_geometry import cell_occluders, cell_origins

        # Built once: the cells do not move and neither do the occluders. `None` means nothing
        # can stop a ray -- a convex mesh alone in the scene -- and the beam keeps WM.7's form.
        object.__setattr__(self, "_query", cell_occluders(self.patch, self.occluders))
        object.__setattr__(self, "_origins", cell_origins(self.patch))

    @property
    def n_cells(self) -> int:
        return int(np.shape(self.normals_world)[0])

    @property
    def weather(self) -> WeatherSeries:
        """The scene's one series, so the one-weather guard (CLAUDE.md #6) sees this too."""
        return self.surfaces.weather

    @property
    def normals_enu(self) -> NDArray[np.float64]:
        return np.asarray(self._normals_enu)  # type: ignore[attr-defined]

    def cell_visibility(self, t_s: float, sun: Any = None) -> NDArray[np.float64]:
        """``(n_cells,)`` direct-beam visibility at ``t_s``.

        Traced per cell when there is anything to trace against (`WM.4`); otherwise the surface's
        own ``shaded`` flag, broadcast, which is what a convex mesh alone in a scene needs.
        """
        if self._query is None:  # type: ignore[attr-defined]
            shaded = self.surfaces.orientations[self.index].shaded
            return np.full(self.n_cells, 0.0 if shaded else 1.0)
        if sun is None:
            sun = self.surfaces.solar_terms(t_s)
        if not sun.above_horizon:
            return np.ones(self.n_cells)
        from irsim.thermal.raycast import disc_visibility

        return np.asarray(
            disc_visibility(
                self._origins,  # type: ignore[attr-defined]
                self.frame.to_world(sun.direction_enu),
                self._query,  # type: ignore[attr-defined]
                self.penumbra_rays,
            )
        )

    def __call__(self, t_s: float) -> FacetForcing:
        t_air, h, _q_solar, q_lw, q_int = self.surfaces(t_s).arrays(self.surfaces.n_facets)
        i, n = self.index, self.n_cells
        sun = self.surfaces.solar_terms(t_s)
        q_solar_cells = solar_loading(
            self.normals_enu,
            sun.direction_enu,
            sun.dni_w_m2,
            sun.dhi_w_m2,
            self.sky_view,
            self.cell_visibility(t_s, sun),
        )
        if self.sky_view_longwave:
            sample = self.surfaces.weather.at(t_s)
            q_lw_cells = np.asarray(
                longwave_down(
                    sample.t_air_k,
                    sample.vapour_pressure_hpa,
                    sample.cloud_fraction,
                    self.sky_view,
                    sample.t_air_k,
                ),
                dtype=np.float64,
            )
        else:
            q_lw_cells = np.full(n, q_lw[i])
        base = self.surfaces(t_s)
        return FacetForcing(
            t_air_k=float(t_air[i]),
            h_w_m2_k=np.full(n, h[i]),
            q_solar_w_m2=np.asarray(q_solar_cells, dtype=np.float64),
            q_longwave_down_w_m2=q_lw_cells,
            q_internal_w_m2=np.full(n, q_int[i]),
            q_air_kg_kg=float(base.q_air_kg_kg),
            g_e_kg_m2_s=float(base.g_e_kg_m2_s),
            precip_kg_m2_s=float(base.precip_kg_m2_s),
        )
