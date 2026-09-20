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
    "SCENE_SCHEMA_VERSION",
    "MIN_SCENE_SCHEMA_VERSION",
    "SurfaceSpec",
    "ThermalSceneSpec",
    "SiteSpec",
    "TargetSpec",
    "SceneSpec",
    "SceneConfig",
    "load_scene_config",
]

SCENE_SCHEMA_VERSION = 8  # v8: `world_frame:` and `thermal.occluders:` (ADR 0095, PT.18);
# v7 added the `patch:` block (ADR 0087, PT.2)
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
    solver: Literal["newton", "prescribed", "heat_source", "airframe", "ram_skin", "vehicle_source"]
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

    @model_validator(mode="after")
    def _fields_for_solver(self) -> TargetSpec:
        if self.solver == "vehicle_source":
            return self._vehicle_fields()
        if self.solver in ("heat_source", "airframe", "ram_skin"):
            return self._aerial_fields()
        if self.load is not None or self.load_s is not None:
            raise ValueError(f"target {self.name!r}: only vehicle_source takes a load profile")
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

    def _aerial_fields(self) -> TargetSpec:
        """Validate the ADR 0072 solvers. The heat-source names are checked against the model."""
        from irsim.thermal.aerial import AERIAL_HEAT_SOURCES

        if self.t0_k is not None or self.tau_s is not None:
            raise ValueError(f"target {self.name!r}: {self.solver} takes no t0_k/tau_s")
        if self.load is not None or self.load_s is not None:
            raise ValueError(f"target {self.name!r}: only vehicle_source takes a load profile")
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
    #: Present makes this surface point-wise: one temperature per cell instead of one for the
    #: whole surface. Absent leaves the surface exactly as it was, so every v4-v6 scene loads
    #: and solves unchanged (schema v7, PT.2).
    patch: PatchSpec | None = None


class ThermalSceneSpec(_Frozen):
    """§12.3's thermal block: how the surfaces are solved, not what they are."""

    surfaces: list[SurfaceSpec] = Field(default_factory=list)
    spin_up_hours: float = Field(default=48.0, gt=0.0, le=336.0)
    tick_s: float = Field(default=1.0, gt=0.0, le=3600.0)
    #: What casts shadows on the patched surfaces (schema v8, PT.18). Empty is the v7 scene.
    occluders: list[OccluderSpec] = Field(default_factory=list)

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
