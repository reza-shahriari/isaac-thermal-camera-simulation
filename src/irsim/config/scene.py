"""Scene config (phase 1): where, when, which weather file, which atmosphere, which targets.

The only place a weather *file* is named. ``Scene.from_config`` (irsim.scene) loads it once and
injects the resulting ``WeatherSeries`` into every consumer; no consumer takes a path
(CLAUDE.md #6, ADR 0032). Targets are the phase-1 solvers of M6.6 (sky targets first: aircraft,
drones, birds with scripted or relaxing temperatures); the environment solver and the §12.3
thermal block arrive with M6.12.

Four solver kinds. ``newton`` and ``prescribed`` are the raw M6.6 solvers. ``heat_source`` and
``airframe`` are the **aerial** model of §6.6 / ADR 0072 -- T = T_air(t) + ΔT_max u(t)^n for a
motor, speed controller or battery pack, and T = T_air(t) + offset for an unpowered skin. Those two
existed in :mod:`irsim.thermal.aerial` from M6.6 and were reachable only from tests: no scene could
ask for a motor. Naming a throttle profile here rather than a temperature schedule is the point --
a flight is authored as *what the pilot did*, and the temperatures follow from the model rather
than from numbers someone typed (ADR 0074).

docs/physics-model.md §6.5, §6.6, §12.2
"""

from __future__ import annotations

import math
import os
import pathlib
from datetime import datetime
from typing import Literal

import numpy as np
import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)

__all__ = [
    "PlumeSpec",
    "SCENE_SCHEMA_VERSION",
    "MIN_SCENE_SCHEMA_VERSION",
    "MeshSpec",
    "SurfaceSpec",
    "ThermalSceneSpec",
    "SiteSpec",
    "BackSpec",
    "WaterSpec",
    "CabinPanelSpec",
    "TemperatureMapSpec",
    "ParameterMapSpec",
    "CabinSpec",
    "TargetSpec",
    "SceneSpec",
    "SceneConfig",
    "load_scene_config",
]

SCENE_SCHEMA_VERSION = 17  # v17: a surface's `temperature_map:` /
# `parameter_maps:` (PT.13); v16 a surface mesh
# read from a prepared asset archive (ADR 0132,
# AI.2); v15 an
# exhaust target's `plume:` (PH.6); v14 a
# surface's `mesh:` (ADR 0110, WM.7); v13 its
# speed schedule (ADR 0109, PT.9); v12 its
# `water:` (ADR 0108, PH.3); v11
# `thermal.penumbra_rays:` (ADR 0107, PT.22); v10
# `thermal.cabin:` and a surface's `back:` (ADR 0106, PT.15);
# v9 `thermal.nodes/links/sources` (ADR 0097, TC.4); v8 `world_frame:` and
# `thermal.occluders:` (ADR 0095, PT.18); v7 the `patch:` block (ADR 0087)
#: The oldest version this loader still accepts. v5 added `thermal:` as an **optional** field, so
#: every v4 document is a valid v5 document and refusing one would be refusing it for a change
#: that cannot affect it. A range is the honest representation of a backwards-compatible change;
#: raise this floor only when a version genuinely stops being readable.
MIN_SCENE_SCHEMA_VERSION = 4  # v4: ram_skin aerodynamic-heating solver (ADR 0075)


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class SiteSpec(_Frozen):
    latitude_deg: float = Field(ge=-90.0, le=90.0)
    longitude_deg: float = Field(ge=-180.0, le=180.0)  # east positive
    altitude_m: float = Field(default=0.0, ge=-500.0, le=9000.0)


class PlumeSpec(_Frozen):
    """The gas cone leaving an exhaust, authored in **world** coordinates (schema v15, `PH.6`).

    Only an ``exhaust`` target may carry one, and that is the point: the plume's temperature is
    not authored here at all. It is the section's own solved gas temperature at the moment the
    frame is taken (`TC.7`), so a plume cannot drift out of step with the pipe it leaves -- the
    failure §6.6's independent schedules made easy.

    What *is* authored is geometry and chemistry. ``origin_m`` and ``direction`` place the pipe
    exit and point it; ``radius_tip_m`` is the pipe's own bore and ``radius_end_m`` the plume's
    spread at ``length_m``; ``mixing_length_m`` is the distance over which entrainment takes the
    excess temperature -- and the species with it -- down by 1/e. The partial pressures are the
    gas at the exit: roughly 0.11 atm CO2 and 0.12 atm H2O for petrol at stoichiometry, less for
    a lean diesel, and ``f_soot`` is a volume fraction (1e-7 to 1e-5 for a sooty flame; a modern
    car is far below that and a smoky diesel is not).
    """

    origin_m: tuple[float, float, float]
    direction: tuple[float, float, float]
    length_m: float = Field(gt=0.0)
    radius_tip_m: float = Field(gt=0.0)
    radius_end_m: float = Field(gt=0.0)
    mixing_length_m: float = Field(gt=0.0)
    p_co2_atm: float = Field(default=0.0, ge=0.0, le=1.0)
    p_h2o_atm: float = Field(default=0.0, ge=0.0, le=1.0)
    f_soot: float = Field(default=0.0, ge=0.0)

    @model_validator(mode="after")
    def _has_something_to_absorb(self) -> PlumeSpec:
        if self.p_co2_atm == self.p_h2o_atm == self.f_soot == 0.0:
            raise ValueError("a plume with no CO2, no H2O and no soot is warm air, not a plume")
        if sum(v * v for v in self.direction) <= 0.0:
            raise ValueError("a plume's direction needs a direction")
        return self


class TargetSpec(_Frozen):
    """One temperature node.

    * ``newton`` relaxes toward the weather's T_air from ``t0_k`` with ``tau_s``.
    * ``prescribed`` follows ``schedule_s`` (seconds after the scene start) → ``schedule_k``.
    * ``heat_source`` is §6.6's aerial dissipating node: ``source`` names one of
      :data:`irsim.thermal.aerial.AERIAL_HEAT_SOURCES` (``motor``, ``esc``, ``battery``) and
      ``throttle_s`` → ``throttle`` is the pilot's throttle fraction over time. The temperature is
      T_air(t) + ΔT_max u(t)^n, derived, never authored.
    * ``airframe`` is an unpowered skin at *multirotor* speed: T_air(t) + ``offset_k``.
    * ``vehicle_source`` is §6.6's **ground-vehicle** node: ``source`` names one of
      :data:`irsim.thermal.vehicle.VEHICLE_HEAT_SOURCES` (``engine_bay``, ``exhaust_manifold``,
      ``catalytic_converter``, ``exhaust_pipe``, ``exhaust_tip``, ``brake_disc``, ``tyre``) and
      ``load_s`` → ``load`` is the duty fraction over time -- 0 before the key turns, 1 at full
      load. Unlike ``heat_source`` it **has a time constant**, which for these rows is the whole
      character of the thing: an engine bay is 750 s of warm-up and half an hour of cool-down, and
      a steady-state law would show the full 65 K in the first frame after ignition.
    * ``ram_skin`` is an unpowered skin fast enough for aerodynamic heating to matter: the
      adiabatic wall temperature T_air (1 + r (gamma-1)/2 M^2) at a constant ``speed_m_s``, with
      the Mach number taken against the shared weather's own air temperature (ADR 0075). At 20 m/s
      it agrees with ``airframe`` to 0.2 K; at 250 m/s it is 28 K warmer, and using ``airframe``
      there understates the whole skin uniformly and plausibly.

    **The heat-source law is a steady-state relation** (ADR 0072): there is no thermal time
    constant in it, so the node follows the throttle instantaneously. It is only defensible where
    the throttle varies slowly against a real motor's time constant, which is minutes -- a profile
    that swings the throttle in seconds will produce a temperature swing no metal could follow.
    ADR 0074 records this and why the demo is filmed as a time-lapse because of it.
    """

    name: str = Field(min_length=1)
    solver: Literal[
        "newton",
        "prescribed",
        "heat_source",
        "airframe",
        "ram_skin",
        "vehicle_source",
        "engine",
        "exhaust",
    ]
    t0_k: float | None = Field(default=None, gt=0.0)
    tau_s: float | None = Field(default=None, gt=0.0)
    schedule_s: list[float] | None = None
    schedule_k: list[float] | None = None
    source: str | None = None
    throttle_s: list[float] | None = None
    throttle: list[float] | None = None
    offset_k: float | None = None
    load_s: list[float] | None = None
    load: list[float] | None = None
    speed_m_s: float | None = Field(default=None, ge=0.0)
    recovery_factor: float | None = Field(default=None, gt=0.0, le=1.0)
    #: ``exhaust`` only: which section's skin the target reports (TC.7, ADR 0105).
    section: str | None = None
    #: ``exhaust`` only: the gas cone it blows, in world coordinates (schema v15, `PH.6`).
    plume: PlumeSpec | None = None

    @model_validator(mode="after")
    def _fields_for_solver(self) -> TargetSpec:
        if self.section is not None and self.solver != "exhaust":
            raise ValueError(f"target {self.name!r}: only exhaust takes a `section`")
        if self.plume is not None and self.solver != "exhaust":
            raise ValueError(
                f"target {self.name!r}: only an exhaust target takes a `plume` -- its temperature "
                "is the section's own solved gas temperature, not an authored one (`PH.6`)"
            )
        if self.solver == "vehicle_source":
            return self._vehicle_fields()
        if self.solver in ("engine", "exhaust"):
            return self._engine_fields()
        if self.solver in ("heat_source", "airframe", "ram_skin"):
            return self._aerial_fields()
        if self.load is not None or self.load_s is not None:
            raise ValueError(
                f"target {self.name!r}: only vehicle_source and engine take a load profile"
            )
        if self.source is not None or self.throttle is not None or self.throttle_s is not None:
            raise ValueError(f"target {self.name!r}: {self.solver} takes no throttle profile")
        if self.offset_k is not None:
            raise ValueError(f"target {self.name!r}: only airframe takes offset_k")
        if self.speed_m_s is not None or self.recovery_factor is not None:
            raise ValueError(f"target {self.name!r}: only ram_skin takes an airspeed")
        if self.solver == "newton":
            if self.t0_k is None or self.tau_s is None:
                raise ValueError(f"target {self.name!r}: newton needs t0_k and tau_s")
            if self.schedule_s is not None or self.schedule_k is not None:
                raise ValueError(f"target {self.name!r}: newton takes no schedule")
        else:
            if not self.schedule_s or not self.schedule_k:
                raise ValueError(
                    f"target {self.name!r}: prescribed needs schedule_s and schedule_k"
                )
            if len(self.schedule_s) != len(self.schedule_k):
                raise ValueError(
                    f"target {self.name!r}: schedule_s and schedule_k differ in length"
                )
            if any(b <= a for a, b in zip(self.schedule_s[:-1], self.schedule_s[1:], strict=True)):
                raise ValueError(f"target {self.name!r}: schedule_s must be strictly increasing")
            if any(t <= 0.0 for t in self.schedule_k):
                raise ValueError(f"target {self.name!r}: schedule_k must be positive kelvin")
            if self.t0_k is not None or self.tau_s is not None:
                raise ValueError(f"target {self.name!r}: prescribed takes no t0_k/tau_s")
        return self

    def _vehicle_fields(self) -> TargetSpec:
        """Validate §6.6's ground-vehicle node. The source names are checked against the model."""
        from irsim.thermal.vehicle import VEHICLE_HEAT_SOURCES

        if self.t0_k is not None or self.tau_s is not None:
            raise ValueError(f"target {self.name!r}: vehicle_source takes no t0_k/tau_s")
        if self.schedule_s is not None or self.schedule_k is not None:
            raise ValueError(
                f"target {self.name!r}: vehicle_source derives its curve from §6.6's table; "
                "authoring temperatures directly is what this solver exists to replace"
            )
        if self.offset_k is not None:
            raise ValueError(f"target {self.name!r}: only airframe takes offset_k")
        if self.speed_m_s is not None or self.recovery_factor is not None:
            raise ValueError(f"target {self.name!r}: only ram_skin takes an airspeed")
        if self.throttle is not None or self.throttle_s is not None:
            raise ValueError(
                f"target {self.name!r}: vehicle_source takes load_s/load, not a throttle profile "
                "(throttle belongs to the aerial heat_source node, which has no time constant)"
            )
        if self.source not in VEHICLE_HEAT_SOURCES:
            raise ValueError(
                f"target {self.name!r}: source must be one of "
                f"{sorted(VEHICLE_HEAT_SOURCES)}, got {self.source!r}"
            )
        if not self.load_s or not self.load:
            raise ValueError(f"target {self.name!r}: vehicle_source needs load_s and load")
        if len(self.load_s) != len(self.load):
            raise ValueError(f"target {self.name!r}: load_s and load differ in length")
        if any(b <= a for a, b in zip(self.load_s[:-1], self.load_s[1:], strict=True)):
            raise ValueError(f"target {self.name!r}: load_s must be strictly increasing")
        if any(not 0.0 <= u <= 1.0 for u in self.load):
            raise ValueError(f"target {self.name!r}: load must lie in [0, 1]")
        return self

    def _engine_fields(self) -> TargetSpec:
        """``engine`` (TC.5, ADR 0100): block, bay air, mounts and subframe solved as a network from
        a load schedule -- no §6.6 row, no rise or time constant to author. ``exhaust`` (TC.7,
        ADR 0105) is the line downstream of it on the same kind of schedule, reporting the skin
        of ``section`` (default ``mid_pipe``)."""
        kind = self.solver
        if self.source is not None:
            raise ValueError(
                f"target {self.name!r}: {kind} takes no `source`; its heat is P_rated · load "
                "(the engine) or the gas that load makes (the exhaust), not a §6.6 row"
            )
        if self.t0_k is not None or self.tau_s is not None:
            raise ValueError(f"target {self.name!r}: {kind} takes no t0_k/tau_s")
        if self.schedule_s is not None or self.schedule_k is not None:
            raise ValueError(f"target {self.name!r}: {kind} takes no schedule; it is solved")
        if self.throttle is not None or self.throttle_s is not None:
            raise ValueError(f"target {self.name!r}: {kind} takes load_s/load, not a throttle")
        if (
            self.offset_k is not None
            or self.speed_m_s is not None
            or self.recovery_factor is not None
        ):
            raise ValueError(f"target {self.name!r}: {kind} takes no aerial fields")
        if not self.load_s or not self.load:
            raise ValueError(f"target {self.name!r}: {kind} needs load_s and load")
        if len(self.load_s) != len(self.load):
            raise ValueError(f"target {self.name!r}: load_s and load differ in length")
        if any(b <= a for a, b in zip(self.load_s[:-1], self.load_s[1:], strict=True)):
            raise ValueError(f"target {self.name!r}: load_s must be strictly increasing")
        if any(not 0.0 <= u <= 1.0 for u in self.load):
            raise ValueError(f"target {self.name!r}: load must lie in [0, 1]")
        return self

    def _aerial_fields(self) -> TargetSpec:
        """Validate the ADR 0072 solvers. The heat-source names are checked against the model."""
        from irsim.thermal.aerial import AERIAL_HEAT_SOURCES

        if self.t0_k is not None or self.tau_s is not None:
            raise ValueError(f"target {self.name!r}: {self.solver} takes no t0_k/tau_s")
        if self.load is not None or self.load_s is not None:
            raise ValueError(
                f"target {self.name!r}: only vehicle_source and engine take a load profile"
            )
        if self.schedule_s is not None or self.schedule_k is not None:
            raise ValueError(
                f"target {self.name!r}: {self.solver} derives its schedule from the model; "
                "authoring temperatures directly is what this solver exists to replace"
            )
        if self.solver in ("airframe", "ram_skin"):
            if self.source is not None or self.throttle is not None:
                raise ValueError(f"target {self.name!r}: {self.solver} takes no source/throttle")
            if self.solver == "airframe":
                if self.speed_m_s is not None or self.recovery_factor is not None:
                    raise ValueError(
                        f"target {self.name!r}: airframe has no airspeed -- it *assumes* a slow "
                        "one. Use ram_skin if the speed matters."
                    )
                return self
            if self.speed_m_s is None:
                raise ValueError(f"target {self.name!r}: ram_skin needs speed_m_s")
            if self.offset_k is not None:
                raise ValueError(f"target {self.name!r}: only airframe takes offset_k")
            return self

        if self.source not in AERIAL_HEAT_SOURCES:
            raise ValueError(
                f"target {self.name!r}: source must be one of "
                f"{sorted(AERIAL_HEAT_SOURCES)}, got {self.source!r}"
            )
        if self.offset_k is not None:
            raise ValueError(f"target {self.name!r}: only airframe takes offset_k")
        if self.speed_m_s is not None or self.recovery_factor is not None:
            raise ValueError(f"target {self.name!r}: only ram_skin takes an airspeed")
        if not self.throttle_s or not self.throttle:
            raise ValueError(f"target {self.name!r}: heat_source needs throttle_s and throttle")
        if len(self.throttle_s) != len(self.throttle):
            raise ValueError(f"target {self.name!r}: throttle_s and throttle differ in length")
        if any(b <= a for a, b in zip(self.throttle_s[:-1], self.throttle_s[1:], strict=True)):
            raise ValueError(f"target {self.name!r}: throttle_s must be strictly increasing")
        if any(not 0.0 <= u <= 1.0 for u in self.throttle):
            raise ValueError(f"target {self.name!r}: throttle must lie in [0, 1]")
        return self


class PatchSpec(_Frozen):
    """The spatial half of a surface: a rectangular grid of cells on a plane (ADR 0087, PT.2).

    Until now a point-wise surface could only be built in Python — `irsim_isaac.car_demo` hand-wrote
    its bonnet and road patches and `surface_fields=` was passed at exactly one call site. That
    makes the owner's own bar -- "a scene config plus one command produces frames" -- unmeetable
    for any new point-wise scene, which is the requirement point-wise temperature exists to serve.

    The fields are :class:`~irsim.thermal.surface_field.PlanarPatch`'s, declared rather than
    computed: cell ``(i_v, i_u)`` is centred at ``origin_m + (i_u + ½)·du·u + (i_v + ½)·dv·v``.

    ``frame`` is a *name*, not a transform — ``"world"`` for a road, a prim path for a panel that
    moves with its object — and `irsim_isaac.pipeline.point_bridge` checks it rather than assuming.
    ``prim_path`` is what binds the solved field to geometry at render time; a patch without one is
    still solvable, it simply never reaches a pixel.
    """

    origin_m: tuple[float, float, float]
    u_axis: tuple[float, float, float]
    v_axis: tuple[float, float, float]
    n_u: int = Field(ge=1)
    n_v: int = Field(ge=1)
    du_m: float = Field(gt=0.0)
    dv_m: float = Field(gt=0.0)
    #: A patch claims a **slab**, not a plane: a bonnet 0.9 m above a road projects into the road's
    #: own rectangle, and a patch testing only its in-plane extent would paint the car with the road
    #: and look reasonable doing it (ADR 0087).
    thickness_m: float = Field(default=0.25, gt=0.0)
    frame: str = Field(default="world", min_length=1)
    prim_path: str | None = None

    @field_validator("u_axis", "v_axis")
    @classmethod
    def _non_zero(cls, v: tuple[float, float, float], info: ValidationInfo) -> tuple[float, ...]:
        if math.isclose(sum(c * c for c in v), 0.0, abs_tol=1e-18):
            raise ValueError(f"{info.field_name} has zero length")
        return v

    @model_validator(mode="after")
    def _perpendicular(self) -> PatchSpec:
        """Checked here as well as in `PlanarPatch`, so a bad scene fails at load, not at render.

        Orthogonality is required rather than silently repaired by Gram-Schmidt: a caller who wrote
        two axes five degrees from perpendicular meant something, and a quietly squared-up grid
        would sample a surface nobody described.
        """
        u, v = np.asarray(self.u_axis, dtype=float), np.asarray(self.v_axis, dtype=float)
        u = u / float(np.linalg.norm(u))
        v = v / float(np.linalg.norm(v))
        if abs(float(np.dot(u, v))) > 1e-9:
            raise ValueError(
                f"u_axis {self.u_axis} and v_axis {self.v_axis} must be perpendicular; "
                f"they meet at {math.degrees(math.acos(min(1.0, abs(float(np.dot(u, v)))))):.3f} "
                "degrees from it"
            )
        return self


class MeshSpec(_Frozen):
    """The spatial half of a surface on **curved** geometry (WM.2/WM.3, ADR 0110).

    :class:`PatchSpec` projects the per-pixel world position onto a rectangle, which is exact for
    a road, a bonnet, a roof or a deck and wrong for everything that curves. This block builds a
    :class:`~irsim.thermal.mesh_field.TriangleMeshPatch` instead: cells live **on the mesh**, at a
    per-face level, and a closest-point query locates a pixel on it. A pipe, an arm, a mast, a
    tyre or a motor bell can then carry a temperature that varies around its circumference, which
    is what one shared patch normal cannot express.

    **Two sources, and neither imports an engine.** Either the author names a generated
    primitive (``shape:``) or names a prim in a **prepared asset archive** (``asset:`` +
    ``prim:``, ADR 0128, schema v16). The archive is an ``.npz`` of world-space triangles written
    by ``scripts/prep_asset.py`` running inside Blender, so `src/irsim/` still never imports `pxr`
    (CLAUDE.md #1) and a scene stays loadable and solvable with no renderer present. This is the
    follow-on `probe_warp_prim.py` measured the cost of; it landed on the Blender side rather than
    the Isaac side, because Blender's OpenUSD runs on the CPU and Kit's does not.

    **The archive is the *thermal* mesh, not the render mesh.** They need not match: the
    closest-point query locates a pixel on whichever mesh the field carries, so the renderer keeps
    the asset's full tessellation while the solver carries one sized to the physics. What bounds
    the divergence is the bridge's distance tolerance, not an equality.

    ``level`` is cells per face edge, so a face carries ``level²``; ``cell_m`` picks a level per
    face from a target cell size instead. On an imported prim ``cell_m`` is strongly preferred:
    a product-visualisation asset ships triangles far smaller than any thermal gradient, and
    ``level`` would multiply an already excessive face count. ``frame`` and ``prim_path`` mean
    exactly what they mean on a patch.
    """

    shape: Literal["cylinder", "sphere"] | None = None
    #: A prepared asset: a name resolving under ``data/assets/``, or a path to a ``.npz``.
    asset: str | None = None
    #: Which prim of that archive. Mutually exclusive with `shape`.
    prim: str | None = None
    centre_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    radius_m: float | None = Field(default=None, gt=0.0)
    #: Cylinders only; a sphere has none and refuses one.
    length_m: float | None = Field(default=None, gt=0.0)
    axis: tuple[float, float, float] = (0.0, 0.0, 1.0)
    #: Facets around the circumference, and along the axis (cylinder) or from pole to pole
    #: (sphere). These set the *geometry*; `level` and `cell_m` set the cells on it.
    segments: int = Field(default=24, ge=3, le=256)
    rings: int = Field(default=8, ge=1, le=256)
    capped: bool = True
    level: int = Field(default=1, ge=1, le=32)
    cell_m: float | None = Field(default=None, gt=0.0)
    #: Whether the mesh shades itself. ``None`` is "trace it unless it is convex", which is exact
    #: and is what every generated mesh uses. An **imported** asset usually has to say ``false``
    #: explicitly: self-occlusion tracing costs cells x faces, and a real asset's prim carries
    #: tens of thousands of both, so the trace that is free on a 16-facet tube does not finish on
    #: a 60,000-face shell. Saying ``false`` falls back to the analytic ``(1 + cos beta)/2`` per
    #: cell -- the same unoccluded form the in-sim path already uses (ADR 0045) -- and is honest
    #: for an airframe in free air, where the only occluder is the aircraft itself. A scene that
    #: leaves this ``None`` on a mesh over the budget is refused rather than left to run for hours.
    self_occluding: bool | None = None
    frame: str = Field(default="world", min_length=1)
    prim_path: str | None = None

    @model_validator(mode="after")
    def _shape_fields_match_the_shape(self) -> MeshSpec:
        generated, imported = self.shape is not None, self.asset is not None
        if generated == imported:
            raise ValueError("a mesh names exactly one of `shape:` (generated) or `asset:`")
        if imported:
            if self.prim is None:
                raise ValueError("an asset mesh needs `prim:` naming which prim to solve")
            for field in ("radius_m", "length_m"):
                if getattr(self, field) is not None:
                    raise ValueError(f"an asset mesh has no {field}; its geometry is in the file")
            if self.cell_m is not None and self.level != 1:
                raise ValueError("a mesh takes `level` or `cell_m`, not both")
            return self
        if self.prim is not None:
            raise ValueError("`prim:` names a prim in an asset archive; a generated mesh has none")
        if self.radius_m is None:
            raise ValueError("a generated mesh needs radius_m")
        if self.shape == "cylinder" and self.length_m is None:
            raise ValueError("a cylinder mesh needs length_m")
        if self.shape == "sphere" and self.length_m is not None:
            raise ValueError("a sphere mesh has no length_m")
        if self.cell_m is not None and self.level != 1:
            raise ValueError("a mesh takes `level` or `cell_m`, not both")
        if math.isclose(sum(c * c for c in self.axis), 0.0, abs_tol=1e-18):
            raise ValueError("a mesh's axis has zero length")
        return self


class WorldFrameSpec(_Frozen):
    """Which way the scene's world frame is up and which way is north (schema v8, PT.18).

    Patches and occluders are authored in world coordinates; the sun is computed in ENU. Until
    this block nothing related the two, and the car scenes -- Y-up, like the stage they author --
    could not have been given a shadow without one. The default is ENU itself, so every v4-v7
    scene reads exactly as before. East is derived (``north × up``), never declared.
    """

    up: tuple[float, float, float] = (0.0, 0.0, 1.0)
    north: tuple[float, float, float] = (0.0, 1.0, 0.0)

    @model_validator(mode="after")
    def _right_handed(self) -> WorldFrameSpec:
        from irsim.thermal.frames import WorldFrame

        WorldFrame(up=np.asarray(self.up), north=np.asarray(self.north))  # raises with the reason
        return self


class OccluderSpec(_Frozen):
    """A rectangle that casts a hard-edged shadow on every world-frame patch (schema v8, PT.18).

    The same primitive `irsim.thermal.shadow.ShadowRectangle` is: a slab, a parapet, a wing, a
    container. It shades the direct beam only; diffuse sky survives (§6.1). Occluders live at the
    scene level, not under a surface, because one slab shades the wall *and* the ground under it.
    """

    name: str = Field(min_length=1)
    centre_m: tuple[float, float, float]
    u_axis: tuple[float, float, float]
    v_axis: tuple[float, float, float]
    half_u_m: float = Field(gt=0.0)
    half_v_m: float = Field(gt=0.0)
    #: Only ``"world"`` for now: a moving occluder needs its frame's pose, which the thermal core
    #: does not carry (PT.9, WM). Declared rather than implied so the limit is visible.
    frame: str = "world"

    @field_validator("frame")
    @classmethod
    def _world_only(cls, v: str) -> str:
        if v != "world":
            raise ValueError(
                f"occluder frame {v!r} is not supported: shadow in a moving frame needs the "
                "frame's pose, which the thermal core does not carry (PT.9, WM)"
            )
        return v

    @model_validator(mode="after")
    def _well_formed(self) -> OccluderSpec:
        from irsim.thermal.shadow import ShadowRectangle

        ShadowRectangle(  # raises with the reason: zero-length or non-perpendicular axes
            centre_m=np.asarray(self.centre_m),
            u_axis=np.asarray(self.u_axis),
            v_axis=np.asarray(self.v_axis),
            half_u_m=self.half_u_m,
            half_v_m=self.half_v_m,
        )
        return self


class FilmSpec(_Frozen):
    """A water film on a patched surface at the scene start (PH.2, ADR 0101).

    ``depth_mm`` of water (1 mm is 1 kg m⁻²) on every cell, or only on the cells whose centre
    lies inside ``region_m`` -- ``[u0, u1, v0, v1]`` in the patch's own (u, v) metres -- so half
    a road can be wet and the other half dry on one prim. The film starts at t₀ (the road has
    just been watered, or the rain has just stopped); it is not spun up.
    """

    depth_mm: float = Field(gt=0.0, le=50.0)
    region_m: tuple[float, float, float, float] | None = None

    @model_validator(mode="after")
    def _region(self) -> FilmSpec:
        if self.region_m is not None:
            u0, u1, v0, v1 = self.region_m
            if not (u1 > u0 and v1 > v0):
                raise ValueError("film region_m must be [u0, u1, v0, v1] with u1 > u0, v1 > v0")
        return self


class TemperatureMapSpec(_Frozen):
    """A raster that *is* this surface's temperature -- DIRSIG's Map Temperature Solver (PT.13).

    ``path`` names a 2-D float ``.npy`` under the data root, row 0 at ``v = 0`` and column 0 at
    ``u = 0``, resampled bilinearly onto the patch's cells (exact and arithmetic-free when the
    raster is already ``n_v`` by ``n_u``).

    ``units`` has **no default**. It is the one fact about the file this code is not entitled to
    assume: published thermal frames are in Celsius far more often than in kelvin, both load as
    plain floats, and a 20 degC raster read as kelvin is 20 K -- not a cold surface but an
    impossible one, which renders as a uniform floor rather than as an error.

    A surface with a temperature map is **not solved**: the map is the answer, spin-up and energy
    balance do not run on it, and it is static in time (a map is a measurement of one moment).
    """

    path: str = Field(min_length=1)
    units: Literal["K", "degC"]


class ParameterMapSpec(_Frozen):
    """A raster driving one `FacetProperties` field per cell -- DIRSIG's MappedTherm (PT.13).

    The surface is still solved; only the parameter varies across it. This is how one prim
    carries a painted panel and a bare one without being two prims.
    """

    parameter: Literal["emissivity", "solar_absorptivity", "heat_capacity_j_m2_k"]
    path: str = Field(min_length=1)


class WaterSpec(_Frozen):
    """Standing water on part of a patched surface: a puddle, a pond, a flooded verge (PH.3).

    The cells inside ``region_m`` -- ``[u0, u1, v0, v1]`` in the patch's own (u, v) metres, or
    the whole patch when absent -- stop being road and become water: their areal heat capacity
    is ``rho c d`` of the mixed layer, their emissivity and solar absorptivity are the water
    material's, and they carry a full film of ``depth_mm`` so the latent term evaporates them
    (ADR 0101). Everything else about the surface is unchanged, which is the point: a puddle is
    a region of a road, on one prim, not a surface of its own.
    """

    depth_mm: float = Field(gt=0.0, le=1000.0)
    region_m: tuple[float, float, float, float] | None = None
    material: str = Field(default="water", min_length=1)

    @model_validator(mode="after")
    def _region(self) -> WaterSpec:
        if self.region_m is not None:
            u0, u1, v0, v1 = self.region_m
            if not (u1 > u0 and v1 > v0):
                raise ValueError("water region_m must be [u0, u1, v0, v1] with u1 > u0, v1 > v0")
        return self


class BackSpec(_Frozen):
    """The deep boundary under a layered surface (§6.4's R₂d and T_deep, ADR 0036)."""

    resistance_m2k_w: float = Field(gt=0.0)
    deep_temperature_k: float | Literal["ambient"]

    @model_validator(mode="after")
    def _positive(self) -> BackSpec:
        if isinstance(self.deep_temperature_k, float) and self.deep_temperature_k <= 0.0:
            raise ValueError("deep_temperature_k must be positive (kelvin)")
        return self


class CabinPanelSpec(_Frozen):
    """One patched surface bounding the cabin, and its inward resistance (ADR 0038)."""

    surface: str = Field(min_length=1)
    #: Conduction + interior-film resistance from the panel to the cabin air, m² K W⁻¹:
    #: ~0.13 for bare metal (the film dominates), ~0.5 with trim, ~0.17 for glass.
    inner_resistance_m2k_w: float = Field(default=0.13, gt=0.0)


class CabinSpec(_Frozen):
    """The cabin air as a lumped node behind its panels (PT.15, ADR 0038, ADR 0106).

    The panels named here are solved **together** with the cabin as one coupled field, so the
    roof loses heat into air the sun has already heated through the glass. The glazing must
    be one of the panels: it is both the solar inlet and one of the largest conduction paths,
    and a cabin given the inlet without the path runs ~22 K too hot (ADR 0038).
    """

    name: str = Field(default="cabin", min_length=1)
    panels: list[CabinPanelSpec] = Field(min_length=1)
    glazing_surface: str = Field(min_length=1)
    volume_m3: float = Field(default=3.0, gt=0.0)
    glazing_area_m2: float = Field(default=2.6, gt=0.0)
    glazing_transmittance: float = Field(default=0.55, ge=0.0, le=1.0)
    air_changes_per_hour: float = Field(default=2.0, ge=0.0)
    interior_mass_j_k: float = Field(default=25_000.0, gt=0.0)

    @model_validator(mode="after")
    def _glazing_is_a_panel(self) -> CabinSpec:
        names = [p.surface for p in self.panels]
        if len(set(names)) != len(names):
            raise ValueError(f"cabin {self.name!r}: panel surfaces must be unique: {names}")
        if self.glazing_surface not in names:
            raise ValueError(
                f"cabin {self.name!r}: glazing_surface must be one of its panels {names} -- the "
                "glass is both the solar inlet and a conduction path (ADR 0038)"
            )
        return self


class SurfaceSpec(_Frozen):
    """One thermally solved surface: a material, where it faces, and whether it is shaded.

    ``material`` names an entry in the material library. There is **no emissivity field**, here or
    in the §12.3 ``thermal:`` block: ε comes from the material's optical data through M7.8's
    hemispherical integral (ADR 0043), so a scene cannot radiate at one value while the camera
    sees another.
    """

    name: str = Field(min_length=1)
    material: str = Field(min_length=1)
    tilt_deg: float = Field(default=0.0, ge=0.0, le=180.0)  # 0 = facing up
    azimuth_deg: float = Field(default=180.0, ge=0.0, lt=360.0)
    shaded: bool = False
    vehicle_speed_m_s: float = Field(default=0.0, ge=0.0)
    #: A speed that changes with the mission (PT.9, ADR 0109): seconds after the scene start
    #: against metres per second, linearly interpolated and held flat outside. Use it instead of
    #: ``vehicle_speed_m_s`` when the platform is sometimes still and sometimes moving -- a
    #: quadrotor on its pad and the same quadrotor in a climb are free and forced convection,
    #: which is tens of kelvin on a sunlit deck.
    speed_s: list[float] | None = None
    speed_m_s: list[float] | None = None
    #: Present makes this surface point-wise: one temperature per cell instead of one for the
    #: whole surface. Absent leaves the surface exactly as it was, so every v4-v6 scene loads
    #: and solves unchanged (schema v7, PT.2).
    patch: PatchSpec | None = None
    #: The same, on **curved** geometry (WM.7, ADR 0110): cells on a generated mesh instead of a
    #: projected rectangle, so a pipe, an arm, a mast or a tyre carries a gradient around its
    #: circumference. Mutually exclusive with `patch`.
    mesh: MeshSpec | None = None
    #: A water film at the scene start, per cell (PH.2). Needs a patch; a per-prim surface has
    #: no cells to be half wet.
    film: FilmSpec | None = None
    #: In-plane conduction between a patch's cells from the material's k and thickness (PT.11,
    #: ADR 0102). On by default; off is the independent-column field for a comparison.
    lateral_conduction: bool = True
    #: Layers through the thickness of every cell (PT.12, ADR 0103): 1 is the single node the
    #: field always was; N cuts the material's thickness into N equal slices with §6.4's contact
    #: resistance between them and an adiabatic back. Needs a patch.
    layers: int = Field(default=1, ge=1, le=64)
    #: Standing water over part of the patch (PH.3, ADR 0108): those cells become water, with
    #: the mixed layer's capacity, water's optics and a film to evaporate. Needs a patch, and
    #: cannot sit on a layered surface or beside an authored film.
    water: WaterSpec | None = None
    #: What lies under the last layer (PT.15, ADR 0036's R₂d and T_deep): a resistance to a deep
    #: temperature, kelvin or ``"ambient"`` (the air at the scene start). Absent is adiabatic.
    #: Needs ``layers`` ≥ 2 -- a single layer with a deep boundary is `LumpedTwoNodeSolver`'s
    #: substrate-less case, which `layered_field` refuses.
    back: BackSpec | None = None
    #: PT.13: a raster that replaces the solve on this surface. Needs a `patch:`.
    temperature_map: TemperatureMapSpec | None = None
    #: PT.13: rasters that vary a material parameter across the cells the solver then runs on.
    parameter_maps: list[ParameterMapSpec] = Field(default_factory=list)

    @model_validator(mode="after")
    def _film_needs_a_patch(self) -> SurfaceSpec:
        if self.patch is not None and self.mesh is not None:
            raise ValueError(
                f"surface {self.name!r}: a surface takes `patch:` or `mesh:`, not both -- the two "
                "are different parameterisations of the same surface and a cell would belong to "
                "each of them"
            )
        if self.mesh is not None:
            # The patch-only extras are refused on a mesh rather than silently ignored. Each is a
            # rectangular-grid construction: a film and standing water are per-cell masses keyed
            # to a region in (u, v), layers cut a slab under a plane, and PT.11's lateral operator
            # is built on four-neighbour grid edges whose mesh equivalent is WM.6's.
            for field_name, owner in (
                ("film", "PH.2"),
                ("water", "PH.3"),
                ("back", "PT.15"),
            ):
                if getattr(self, field_name) is not None:
                    raise ValueError(
                        f"surface {self.name!r}: `{field_name}:` ({owner}) needs a `patch:`; it "
                        "has no meaning on a mesh yet"
                    )
            if self.layers != 1:
                raise ValueError(f"surface {self.name!r}: `layers:` (PT.12) needs a `patch:`")
        if (self.speed_s is None) != (self.speed_m_s is None):
            raise ValueError(f"surface {self.name!r}: a speed schedule needs speed_s and speed_m_s")
        if self.speed_s is not None:
            assert self.speed_m_s is not None
            if len(self.speed_s) != len(self.speed_m_s) or not self.speed_s:
                raise ValueError(f"surface {self.name!r}: speed_s and speed_m_s differ in length")
            if any(b <= a for a, b in zip(self.speed_s[:-1], self.speed_s[1:], strict=True)):
                raise ValueError(f"surface {self.name!r}: speed_s must be strictly increasing")
            if any(v < 0.0 for v in self.speed_m_s):
                raise ValueError(f"surface {self.name!r}: a speed cannot be negative")
            if self.vehicle_speed_m_s:
                raise ValueError(
                    f"surface {self.name!r}: one speed authority -- `vehicle_speed_m_s` or a "
                    "schedule, not both"
                )
        if self.water is not None:
            if self.patch is None:
                raise ValueError(
                    f"surface {self.name!r}: `water:` needs a `patch:` -- a puddle is a region of "
                    "cells, and a per-prim surface has no cells to be a region of"
                )
            if self.layers > 1:
                raise ValueError(
                    f"surface {self.name!r}: water on a layered surface is not supported -- the "
                    "puddle replaces the cell's one node (PH.3)"
                )
            if self.film is not None:
                raise ValueError(
                    f"surface {self.name!r}: `water:` already lays a film of its own depth; "
                    "authoring `film:` beside it is two authorities on how wet the cell is"
                )
        if self.back is not None and self.layers < 2:
            raise ValueError(
                f"surface {self.name!r}: a `back:` boundary needs `layers: 2` or more (the deep "
                "node sits under the last layer, not under the surface itself)"
            )
        # PT.13: both maps are rasters over a rectangular grid, so both need one.
        for field_name in ("temperature_map", "parameter_maps"):
            if getattr(self, field_name) and self.patch is None:
                raise ValueError(
                    f"surface {self.name!r}: `{field_name}:` (PT.13) needs a `patch:` -- a map is "
                    "resampled onto cells, and a per-prim surface has none"
                )
        if self.temperature_map is not None:
            # A prescribed surface is not solved, so anything that only means something inside a
            # solve is an authoring error rather than a term quietly ignored. Each of these would
            # otherwise be read by a person as having an effect it does not have.
            for field_name, owner in (
                ("parameter_maps", "PT.13"),
                ("film", "PH.2"),
                ("water", "PH.3"),
                ("back", "PT.15"),
            ):
                if getattr(self, field_name):
                    raise ValueError(
                        f"surface {self.name!r}: `{field_name}:` ({owner}) has no meaning beside "
                        "a `temperature_map:` -- the map *is* the temperature, so nothing that "
                        "feeds an energy balance on this surface is read"
                    )
            if self.layers != 1:
                raise ValueError(
                    f"surface {self.name!r}: `layers:` (PT.12) has no meaning beside a "
                    "`temperature_map:` -- a map prescribes the surface, not a slab under it"
                )
        if self.parameter_maps and self.layers > 1:
            raise ValueError(
                f"surface {self.name!r}: `parameter_maps:` (PT.13) with `layers:` (PT.12) is "
                "refused. A layered surface starts its solve from the per-prim spun state, one "
                "number for the whole surface, so a per-cell parameter would not be in the "
                "history the stack begins from -- it would creep in over the first ticks rather "
                "than being the surface's past."
            )
        seen = [m.parameter for m in self.parameter_maps]
        if len(set(seen)) != len(seen):
            raise ValueError(
                f"surface {self.name!r}: parameter_maps names {sorted(seen)} with a repeat; two "
                "rasters for one parameter is two authorities on the same number"
            )
        if self.film is not None and self.patch is None:
            raise ValueError(
                f"surface {self.name!r}: a film needs a `patch:` -- it is a per-cell state, and a "
                "per-prim surface has no cells to be half wet"
            )
        if self.layers > 1 and self.patch is None:
            raise ValueError(f"surface {self.name!r}: `layers` needs a `patch:`")
        if self.layers > 1 and self.film is not None:
            raise ValueError(
                f"surface {self.name!r}: a film on a layered surface is not supported yet -- "
                "the film lives on the single-node field (PH.1)"
            )
        return self


class NodeSpec(_Frozen):
    """One lumped part of the thermal network (schema v9, TC.4; `irsim.thermal.network`).

    Exactly one of: ``capacity_j_k``; ``mass_kg`` with ``specific_heat_j_kgk``; ``fixed`` (a
    kelvin value, or ``"ambient"`` for the weather's air temperature); or ``link_node`` -- a
    link with mass of its own, a rubber mount, between ``a`` and ``b`` with the series
    conductance ``g_w_k`` and this node's ``capacity_j_k``.
    """

    name: str = Field(min_length=1)
    capacity_j_k: float | None = Field(default=None, gt=0.0)
    mass_kg: float | None = Field(default=None, gt=0.0)
    specific_heat_j_kgk: float | None = Field(default=None, gt=0.0)
    fixed: float | Literal["ambient"] | None = None
    link_node: dict[str, str | float] | None = None
    #: A boundary that tracks a scene target's temperature (TC.6): ``{target: engine_bay}`` for
    #: the target's reported temperature, ``{target: engine_bay, node: block}`` for one node of
    #: a solved target. One-way: the target is not cooled by what hangs off it.
    follows: dict[str, str] | None = None
    initial_k: float | None = Field(default=None, gt=0.0)

    @model_validator(mode="after")
    def _one_kind(self) -> NodeSpec:
        forms = {
            "capacity_j_k": self.capacity_j_k is not None,
            "mass_kg": self.mass_kg is not None or self.specific_heat_j_kgk is not None,
            "fixed": self.fixed is not None,
            "link_node": self.link_node is not None,
            "follows": self.follows is not None,
        }
        chosen = [k for k, v in forms.items() if v]
        if len(chosen) != 1 and not (chosen == ["capacity_j_k", "link_node"]):
            raise ValueError(
                f"node {self.name!r} must be exactly one of capacity_j_k, mass_kg + "
                f"specific_heat_j_kgk, fixed, or link_node (+ capacity_j_k); got {chosen}"
            )
        if (self.mass_kg is None) != (self.specific_heat_j_kgk is None):
            raise ValueError(f"node {self.name!r}: mass_kg and specific_heat_j_kgk go together")
        if isinstance(self.fixed, float) and self.fixed <= 0.0:
            raise ValueError(f"node {self.name!r}: a fixed temperature must be positive kelvin")
        if (self.fixed is not None or self.follows is not None) and self.initial_k is not None:
            raise ValueError(f"node {self.name!r}: a boundary node has no initial temperature")
        if self.follows is not None and (
            "target" not in self.follows or not set(self.follows) <= {"target", "node"}
        ):
            raise ValueError(f"node {self.name!r}: follows takes `target` and optionally `node`")
        if self.link_node is not None:
            keys = set(self.link_node)
            if keys != {"a", "b", "g_w_k"} or self.capacity_j_k is None:
                raise ValueError(
                    f"node {self.name!r}: link_node needs a, b and g_w_k, plus capacity_j_k"
                )
            if float(self.link_node["g_w_k"]) <= 0.0:
                raise ValueError(f"node {self.name!r}: link_node g_w_k must be positive")
        return self

    @property
    def is_fixed(self) -> bool:
        return self.fixed is not None or self.follows is not None

    @property
    def capacity(self) -> float | None:
        if self.capacity_j_k is not None:
            return self.capacity_j_k
        if self.mass_kg is not None and self.specific_heat_j_kgk is not None:
            return self.mass_kg * self.specific_heat_j_kgk
        return None


class RadiationSpec(_Frozen):
    emissivity: float = Field(gt=0.0, le=1.0)
    area_m2: float = Field(gt=0.0)
    view_factor: float = Field(default=1.0, gt=0.0, le=1.0)


class LinkSpec(_Frozen):
    """A conductor between two nodes, in exactly one of the forms the literature reports.

    ``g_w_k`` (a total); ``joint`` from `configs/thermal/joints.yaml` with ``area_m2``;
    ``h_c_w_m2_k`` with ``area_m2`` (an inline contact conductance, range-checked like the
    table); ``fastener`` from the table with ``count``; ``h_w_m2_k`` with ``area_m2``
    (convection to a fluid node); or ``radiation``.
    """

    a: str = Field(min_length=1)
    b: str = Field(min_length=1)
    g_w_k: float | None = Field(default=None, ge=0.0)
    joint: str | None = None
    h_c_w_m2_k: float | None = None
    area_m2: float | None = Field(default=None, gt=0.0)
    fastener: str | None = None
    count: int | None = Field(default=None, ge=1)
    h_w_m2_k: float | None = Field(default=None, ge=0.0)
    radiation: RadiationSpec | None = None
    #: Forced → natural (TC.6): with ``switch`` naming an ``engine`` target, ``h_w_m2_k`` holds
    #: while that engine runs and ``off_h_w_m2_k`` when it is off (a fan that stops).
    switch: str | None = None
    off_h_w_m2_k: float | None = Field(default=None, ge=0.0)

    @model_validator(mode="after")
    def _one_form(self) -> LinkSpec:
        from irsim.config.joints import check_h_c

        if self.a == self.b:
            raise ValueError(f"a link cannot join {self.a!r} to itself")
        forms = {
            "g_w_k": self.g_w_k is not None,
            "joint": self.joint is not None,
            "h_c_w_m2_k": self.h_c_w_m2_k is not None,
            "fastener": self.fastener is not None,
            "h_w_m2_k": self.h_w_m2_k is not None,
            "radiation": self.radiation is not None,
        }
        chosen = [k for k, v in forms.items() if v]
        if len(chosen) != 1:
            raise ValueError(
                f"link {self.a!r}-{self.b!r} must take exactly one form (g_w_k, joint, "
                f"h_c_w_m2_k, fastener, h_w_m2_k or radiation); got {chosen}"
            )
        form = chosen[0]
        needs_area = form in ("joint", "h_c_w_m2_k", "h_w_m2_k")
        if needs_area != (self.area_m2 is not None):
            raise ValueError(
                f"link {self.a!r}-{self.b!r}: {form} "
                f"{'needs' if needs_area else 'takes no'} area_m2"
            )
        if (form == "fastener") != (self.count is not None):
            raise ValueError(
                f"link {self.a!r}-{self.b!r}: count goes with fastener, and only with it"
            )
        if self.h_c_w_m2_k is not None:
            check_h_c(self.h_c_w_m2_k, f"link {self.a!r}-{self.b!r}")
        if (self.switch is None) != (self.off_h_w_m2_k is None):
            raise ValueError(f"link {self.a!r}-{self.b!r}: switch and off_h_w_m2_k go together")
        if self.switch is not None and form != "h_w_m2_k":
            raise ValueError(f"link {self.a!r}-{self.b!r}: switch applies to an h_w_m2_k link")
        return self

    @property
    def form(self) -> str:
        for name in ("g_w_k", "joint", "h_c_w_m2_k", "fastener", "h_w_m2_k", "radiation"):
            if getattr(self, name) is not None:
                return name
        raise AssertionError("validated link without a form")


class SourceSpec(_Frozen):
    """Heat in watts on a node: a constant, or a piecewise-linear ``times_s`` → ``power_w``."""

    node: str = Field(min_length=1)
    power_w: float | list[float]
    times_s: list[float] | None = None

    @model_validator(mode="after")
    def _schedule(self) -> SourceSpec:
        if isinstance(self.power_w, list):
            if self.times_s is None or len(self.times_s) != len(self.power_w) or not self.power_w:
                raise ValueError(f"source on {self.node!r}: times_s and power_w must pair up")
            if any(b <= a for a, b in zip(self.times_s[:-1], self.times_s[1:], strict=True)):
                raise ValueError(f"source on {self.node!r}: times_s must be strictly increasing")
            if any(w < 0.0 for w in self.power_w):
                raise ValueError(f"source on {self.node!r}: power_w cannot be negative")
        else:
            if self.times_s is not None:
                raise ValueError(f"source on {self.node!r}: times_s needs a list of power_w")
            if self.power_w < 0.0:
                raise ValueError(f"source on {self.node!r}: power_w cannot be negative")
        return self


class ThermalSceneSpec(_Frozen):
    """§12.3's thermal block: how the surfaces are solved, not what they are."""

    surfaces: list[SurfaceSpec] = Field(default_factory=list)
    spin_up_hours: float = Field(default=48.0, gt=0.0, le=336.0)
    tick_s: float = Field(default=1.0, gt=0.0, le=3600.0)
    #: What casts shadows on the patched surfaces (schema v8, PT.18). Empty is the v7 scene.
    occluders: list[OccluderSpec] = Field(default_factory=list)
    #: The thermal network (schema v9, TC.4): parts, the joints between them, heat in watts.
    nodes: list[NodeSpec] = Field(default_factory=list)
    links: list[LinkSpec] = Field(default_factory=list)
    sources: list[SourceSpec] = Field(default_factory=list)
    #: The cabin behind a set of patched panels (PT.15). Absent leaves every back adiabatic.
    cabin: CabinSpec | None = None
    #: Rays across the sun's 0.53 deg disc when occluders shade a patch (PT.22, ADR 0107). 1 is
    #: the hard-edged shadow every scene has had; 7, 19 or 37 give a penumbra `d tan(0.53 deg)`
    #: wide -- 9.3 mm per metre of standoff, which is sub-cell unless the cells are small and
    #: the occluder close. Each extra ray is another occluder pass per forcing evaluation.
    penumbra_rays: int = Field(default=1, ge=1, le=37)

    @model_validator(mode="after")
    def _penumbra_is_a_ring_count(self) -> ThermalSceneSpec:
        allowed = {1, 7, 19, 37}
        if self.penumbra_rays not in allowed:
            raise ValueError(
                f"penumbra_rays must be one of {sorted(allowed)} (concentric rings on the disc), "
                f"got {self.penumbra_rays}"
            )
        if self.penumbra_rays > 1 and not self.occluders:
            raise ValueError("penumbra_rays sharpens nothing without `occluders:` to cast a shadow")
        return self

    @model_validator(mode="after")
    def _cabin_panels_are_plain_patches(self) -> ThermalSceneSpec:
        """A cabin's panels are patched, single-layer, dry surfaces, solved as one field."""
        if self.cabin is None:
            return self
        by_name = {s.name: s for s in self.surfaces}
        for panel in self.cabin.panels:
            surface = by_name.get(panel.surface)
            if surface is None:
                raise ValueError(
                    f"cabin {self.cabin.name!r}: panel {panel.surface!r} is not a surface; "
                    f"surfaces: {list(by_name)}"
                )
            if surface.patch is None:
                raise ValueError(
                    f"cabin {self.cabin.name!r}: panel {panel.surface!r} needs a `patch:` -- the "
                    "cabin is solved with its panels' cells"
                )
            if surface.layers > 1 or surface.film is not None:
                raise ValueError(
                    f"cabin {self.cabin.name!r}: panel {panel.surface!r} must be a single-layer "
                    "surface without a film (the cabin joins the panel's one node)"
                )
        if any(n.name == self.cabin.name for n in self.nodes):
            raise ValueError(f"cabin {self.cabin.name!r} shares its name with a network node")
        return self

    @field_validator("surfaces")
    @classmethod
    def _unique(cls, surfaces: list[SurfaceSpec]) -> list[SurfaceSpec]:
        names = [s.name for s in surfaces]
        if len(set(names)) != len(names):
            raise ValueError(f"surface names must be unique: {names}")
        return surfaces

    @field_validator("occluders")
    @classmethod
    def _unique_occluders(cls, occluders: list[OccluderSpec]) -> list[OccluderSpec]:
        names = [o.name for o in occluders]
        if len(set(names)) != len(names):
            raise ValueError(f"occluder names must be unique: {names}")
        return occluders

    @model_validator(mode="after")
    def _network_is_well_formed(self) -> ThermalSceneSpec:
        """Names resolve, exactly once; heat lands on a node that can hold it."""
        names = [n.name for n in self.nodes]
        if len(set(names)) != len(names):
            raise ValueError(f"node names must be unique: {names}")
        known = set(names)
        fixed = {n.name for n in self.nodes if n.is_fixed}
        for n in self.nodes:
            if n.link_node is not None:
                for end in (str(n.link_node["a"]), str(n.link_node["b"])):
                    if end not in known:
                        raise ValueError(f"link node {n.name!r} names unknown node {end!r}")
        for link in self.links:
            for end in (link.a, link.b):
                if end not in known:
                    raise ValueError(f"link {link.a!r}-{link.b!r} names unknown node {end!r}")
        for src in self.sources:
            if src.node not in known:
                raise ValueError(f"source names unknown node {src.node!r}")
            if src.node in fixed:
                raise ValueError(
                    f"source on fixed node {src.node!r} goes nowhere: its temperature is imposed"
                )
        if (self.links or self.sources) and not self.nodes:
            raise ValueError("links and sources need nodes")
        return self

    @model_validator(mode="after")
    def _one_shadow_authority(self) -> ThermalSceneSpec:
        """A patched surface under occluders may not also say ``shaded: true``.

        The flag zeroes the beam for the whole surface; the occluders decide per cell. A scene
        holding both would render whichever won the code path, and the author meant one of them.
        """
        if not self.occluders:
            return self
        both = [s.name for s in self.surfaces if s.patch is not None and s.shaded]
        if both:
            raise ValueError(
                f"surfaces {both} declare `shaded: true` and carry a patch while the scene "
                "declares occluders: two shadow authorities. Drop the flag (the occluders decide "
                "per cell) or the occluders."
            )
        return self


class SceneSpec(_Frozen):
    name: str = Field(min_length=1)
    description: str = ""
    weather_file: str = Field(min_length=1)  # relative to the data root (ADR 0008)
    atmosphere_preset: str = Field(min_length=1)  # name in configs/atmospheres
    environment_preset: str | None = None  # configs/environments (sky, ground, solar terms)
    site: SiteSpec
    start_utc: datetime
    targets: list[TargetSpec] = Field(default_factory=list)
    thermal: ThermalSceneSpec | None = None
    #: How world coordinates relate to east, north and up (schema v8). Absent means ENU.
    world_frame: WorldFrameSpec = Field(default_factory=WorldFrameSpec)

    @field_validator("start_utc")
    @classmethod
    def _aware(cls, v: datetime) -> datetime:
        if v.tzinfo is None or v.utcoffset() is None:
            raise ValueError("start_utc must carry a UTC offset (e.g. 2024-06-21T04:00:00Z)")
        return v

    @field_validator("targets")
    @classmethod
    def _unique_names(cls, targets: list[TargetSpec]) -> list[TargetSpec]:
        names = [t.name for t in targets]
        if len(set(names)) != len(names):
            raise ValueError(f"target names must be unique: {names}")
        return targets


class SceneConfig(_Frozen):
    schema_version: int
    scene: SceneSpec

    @field_validator("schema_version")
    @classmethod
    def _version(cls, v: int) -> int:
        if not MIN_SCENE_SCHEMA_VERSION <= v <= SCENE_SCHEMA_VERSION:
            raise ValueError(
                f"scene schema_version {v} outside the readable range "
                f"{MIN_SCENE_SCHEMA_VERSION}-{SCENE_SCHEMA_VERSION}"
            )
        return v


def load_scene_config(path: str | os.PathLike[str]) -> SceneConfig:
    raw = yaml.safe_load(pathlib.Path(path).read_text(encoding="utf-8"))
    return SceneConfig.model_validate(raw)
