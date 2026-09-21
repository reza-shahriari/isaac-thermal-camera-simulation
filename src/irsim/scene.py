"""Scene: one WeatherSeries, loaded once, injected into every consumer (CLAUDE.md #6, ADR 0032).

``Scene.from_config`` is the only code path that turns a weather *file* into a ``WeatherSeries``
for a simulation. It builds the ``Atmosphere`` (M8.5) and the target solvers (M6.6) with that
one object and refuses, at construction, any consumer holding a different one. Phase-2
consumers -- the sky model (MS.2), the housing temperature (M9.3), the FPA thermal node (M9.2),
the environment solver (M6.12) -- register the same way: they take the object and expose it as
``.weather`` so the identity check below covers them without new code here.

Time base: consumers work in seconds on the weather's axis; ``t0_s`` is the scene start on
that axis, and ``t_rel_s`` in the helpers is seconds since the scene start.

docs/physics-model.md §6.4, §7.3, §12.2
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from irsim.atmosphere.layered import LayeredAtmosphere
from irsim.atmosphere.library import load_atmosphere_preset
from irsim.atmosphere.model import Atmosphere
from irsim.atmosphere.sky import SkyModel
from irsim.config.atmosphere import AtmospherePreset
from irsim.config.environment import EnvironmentSpec, load_environment_preset
from irsim.config.loader import resolve_data_dir
from irsim.config.scene import (
    PatchSpec,
    SceneConfig,
    SceneSpec,
    TargetSpec,
    load_scene_config,
)
from irsim.radiometry.lut import BandLUT, Quantity
from irsim.radiometry.spectral_response import SpectralResponse
from irsim.thermal.aerial import (
    AERIAL_HEAT_SOURCES,
    RECOVERY_FACTOR_TURBULENT,
    airframe_solver,
    heat_source_solver,
    ram_skin_solver,
)
from irsim.thermal.conduction import lateral_operator
from irsim.thermal.frames import ENU, WorldFrame
from irsim.thermal.network import ThermalNetwork
from irsim.thermal.shadow import ShadowRectangle
from irsim.thermal.solvers import NewtonCoolingSolver, PrescribedSolver, TemperatureSolver
from irsim.thermal.surface_field import PlanarPatch, PlanarThermalField
from irsim.thermal.vehicle import VEHICLE_HEAT_SOURCES, VehicleSourceSolver
from irsim.thermal.weather import WeatherSample, WeatherSeries
from irsim.thermal.weather_io import load_weather_csv

__all__ = [
    "Scene",
    "build_network",
    "build_occluder",
    "build_target",
    "build_world_frame",
    "wrap_into_weather",
]


def build_patch(spec: PatchSpec) -> PlanarPatch:
    """A declared `patch:` block as the grid the solver runs on (ADR 0087, PT.2).

    A straight hand-over: every field of :class:`~irsim.config.scene.PatchSpec` is a field of
    :class:`~irsim.thermal.surface_field.PlanarPatch`, and the validation lives in both — the schema
    so a bad scene fails at load rather than at render, the dataclass so a patch built in Python is
    held to the same rule. ``prim_path`` is not passed on: it binds the solved field to geometry at
    render time and is no business of the thermal grid.
    """
    return PlanarPatch(
        origin_m=np.asarray(spec.origin_m, dtype=np.float64),
        u_axis=np.asarray(spec.u_axis, dtype=np.float64),
        v_axis=np.asarray(spec.v_axis, dtype=np.float64),
        n_u=spec.n_u,
        n_v=spec.n_v,
        du_m=spec.du_m,
        dv_m=spec.dv_m,
        thickness_m=spec.thickness_m,
        frame=spec.frame,
    )


def build_mesh(spec: Any) -> Any:
    """A declared `mesh:` block as the triangle mesh the solver runs on (ADR 0110, WM.7).

    The geometry is generated rather than read from an asset, because the engine-free core may not
    import `pxr` (CLAUDE.md #1) and a scene has to be loadable and solvable with no renderer. The
    shapes are the ones ADR 0087 listed as Hard: a cylinder is a pipe, an arm, a mast or a motor
    bell, and a sphere is a bell housing or a dome.
    """
    from irsim.thermal.mesh_field import TriangleMeshPatch
    from irsim.thermal.raycast import cylinder_mesh, sphere_mesh

    if spec.shape == "cylinder":
        soup = cylinder_mesh(
            spec.centre_m,
            spec.radius_m,
            float(spec.length_m),
            spec.axis,
            spec.segments,
            spec.rings,
            capped=spec.capped,
        )
    else:
        soup = sphere_mesh(spec.centre_m, spec.radius_m, spec.rings, spec.segments)
    if spec.cell_m is not None:
        return TriangleMeshPatch.by_cell_size(
            soup.vertices, soup.faces, spec.cell_m, frame=spec.frame
        )
    return TriangleMeshPatch.from_soup(soup, spec.level, frame=spec.frame)


def build_world_frame(spec: SceneSpec) -> WorldFrame:
    """The scene's world frame (schema v8, PT.18); ENU for every scene that declares none."""
    wf = spec.world_frame
    return WorldFrame(
        up=np.asarray(wf.up, dtype=np.float64), north=np.asarray(wf.north, dtype=np.float64)
    )


def build_occluder(spec: Any) -> ShadowRectangle:
    """An `OccluderSpec` as the rectangle `cell_shadow` tests against (schema v8, PT.18)."""
    return ShadowRectangle(
        centre_m=np.asarray(spec.centre_m, dtype=np.float64),
        u_axis=np.asarray(spec.u_axis, dtype=np.float64),
        v_axis=np.asarray(spec.v_axis, dtype=np.float64),
        half_u_m=float(spec.half_u_m),
        half_v_m=float(spec.half_v_m),
    )


def _ambient_of(weather: WeatherSeries) -> Any:
    def at(t_s: float) -> float:
        return float(weather.at(t_s).t_air_k)

    return at


def _schedule_of(times_s: Any, power_w: Any) -> Any:
    times = np.asarray(times_s, dtype=np.float64)
    watts = np.asarray(power_w, dtype=np.float64)

    def at(t_s: float) -> float:
        return float(np.interp(t_s, times, watts))

    return at


def _follower(targets: Mapping[str, Any], target: str, node: str | None) -> Any:
    """A boundary reading a scene target's temperature (or one node of a solved target)."""
    solver = targets[target]
    if node is None:
        return lambda t_s: float(solver.temperature())
    if not hasattr(solver, "node_temperature_k"):
        raise ValueError(f"target {target!r} has no nodes to follow; drop `node`")
    solver.node_temperature_k(node)  # raises now if the node does not exist

    def at(t_s: float) -> float:
        return float(solver.node_temperature_k(node))

    return at


def _switched(targets: Mapping[str, Any], switch: str, on: float, off: float) -> Any:
    """``on`` while the named engine target runs, ``off`` otherwise."""
    solver = targets[switch]
    if not hasattr(solver, "load_at"):
        raise ValueError(f"link switch {switch!r} must name an `engine` target with a load")

    def at(t_s: float) -> float:
        return on if float(solver.load_at(t_s)) > 0.0 else off

    return at


def build_network(
    block: Any,
    weather: WeatherSeries,
    t0_s: float,
    joints: Any | None = None,
    targets: Mapping[str, Any] | None = None,
) -> ThermalNetwork | None:
    """The `nodes:` / `links:` / `sources:` blocks as a `ThermalNetwork` (schema v9, TC.4).

    ``None`` when the block declares no nodes, which is every scene before v9. An ``"ambient"``
    fixed node reads the scene's one weather series; a ``joint`` or ``fastener`` link reads
    `configs/thermal/joints.yaml` (or the table passed in), so the number and its provenance
    stay in one place. A source schedule is piecewise linear in time, held at its ends.
    """
    from irsim.config.joints import load_joint_table
    from irsim.thermal.network import (
        FixedNode,
        ImposedHeat,
        Link,
        LinkNode,
        Node,
        RadiationLink,
    )

    if block is None or not block.nodes:
        return None
    table = joints if joints is not None else load_joint_table()
    nodes: list[Node] = []
    fixed: list[FixedNode] = []
    link_nodes: list[LinkNode] = []
    initial: dict[str, float] = {}
    t_air_0 = float(weather.at(t0_s).t_air_k)
    known_targets: Mapping[str, Any] = targets or {}
    for n in block.nodes:
        if n.follows is not None:
            target = str(n.follows["target"])
            if target not in known_targets:
                raise ValueError(f"node {n.name!r} follows unknown target {target!r}")
            node = n.follows.get("node")
            fixed.append(FixedNode(n.name, _follower(known_targets, target, node)))
        elif n.fixed == "ambient":
            fixed.append(FixedNode(n.name, _ambient_of(weather)))
        elif n.fixed is not None:
            fixed.append(FixedNode(n.name, float(n.fixed)))
        elif n.link_node is not None:
            link_nodes.append(
                LinkNode(
                    n.name,
                    str(n.link_node["a"]),
                    str(n.link_node["b"]),
                    float(n.link_node["g_w_k"]),
                    float(n.capacity_j_k),
                )
            )
            initial[n.name] = t_air_0 if n.initial_k is None else float(n.initial_k)
        else:
            nodes.append(Node(n.name, float(n.capacity)))
            initial[n.name] = t_air_0 if n.initial_k is None else float(n.initial_k)
    links: list[Link] = []
    radiation: list[RadiationLink] = []
    for lk in block.links:
        form = lk.form
        if form == "g_w_k":
            links.append(Link(lk.a, lk.b, float(lk.g_w_k)))
        elif form == "joint":
            links.append(
                Link.from_contact(lk.a, lk.b, table.joint(lk.joint).h_c_w_m2_k, lk.area_m2)
            )
        elif form == "h_c_w_m2_k":
            links.append(Link.from_contact(lk.a, lk.b, float(lk.h_c_w_m2_k), lk.area_m2))
        elif form == "fastener":
            links.append(Link(lk.a, lk.b, table.fastener(lk.fastener).g_w_k * int(lk.count)))
        elif form == "h_w_m2_k":
            if lk.switch is not None:
                if lk.switch not in known_targets:
                    raise ValueError(
                        f"link {lk.a!r}-{lk.b!r} switches on unknown target {lk.switch!r}"
                    )
                h = _switched(known_targets, lk.switch, float(lk.h_w_m2_k), float(lk.off_h_w_m2_k))
                links.append(Link.convection(lk.a, lk.b, h, lk.area_m2))
            else:
                links.append(Link.convection(lk.a, lk.b, float(lk.h_w_m2_k), lk.area_m2))
        else:
            r = lk.radiation
            radiation.append(RadiationLink(lk.a, lk.b, r.emissivity, r.area_m2, r.view_factor))
    sources: list[ImposedHeat] = []
    for src in block.sources:
        if isinstance(src.power_w, list):
            sources.append(ImposedHeat(src.node, _schedule_of(src.times_s, src.power_w)))
        else:
            sources.append(ImposedHeat(src.node, float(src.power_w)))
    return ThermalNetwork(
        nodes=nodes,
        fixed=fixed,
        links=links,
        radiation=radiation,
        sources=sources,
        link_nodes=link_nodes,
        t0_s=t0_s,
        initial_k=initial,
        weather=weather,
    )


def build_target(spec: TargetSpec, weather: WeatherSeries, t0_s: float) -> TemperatureSolver:
    """A solver for one target spec, on the scene's one shared ``WeatherSeries`` (CLAUDE.md #6).

    ``heat_source``, ``airframe`` and ``ram_skin`` are the aerial nodes (ADR 0072, ADR 0075).
    They come back as ``PrescribedSolver``s like everything else -- the difference is only that
    their schedule is *derived*, from the throttle profile or the airspeed and the shared weather,
    refined until piecewise-linear interpolation reproduces the analytic law to 1 mK, rather than
    typed into the config.

    ``vehicle_source`` is the one that **cannot** be pre-derived (ADR 0089). §6.6's ground-vehicle
    rows carry a time constant, so the node's temperature depends on its own history and not only
    on the clock: the same load profile started from a cold engine and from one that parked ten
    minutes ago gives different curves. It therefore comes back as a stateful
    :class:`~irsim.thermal.vehicle.VehicleSourceSolver`, which primes itself at ``t0_s``.
    """
    if spec.solver == "newton":
        assert spec.t0_k is not None and spec.tau_s is not None
        return NewtonCoolingSolver(spec.t0_k, spec.tau_s, weather, t0_s=t0_s)
    if spec.solver == "engine":
        from irsim.thermal.engine import EngineSolver, EngineSpec

        assert spec.load_s is not None and spec.load is not None
        return EngineSolver(
            EngineSpec(),
            weather,
            t0_s + np.asarray(spec.load_s, dtype=np.float64),
            spec.load,
            t0_s,
        )
    if spec.solver == "exhaust":
        from irsim.thermal.exhaust_line import ExhaustSolver, stock_exhaust

        assert spec.load_s is not None and spec.load is not None
        # Parked: no ram air under the car, and no head to bolt the manifold to from here (the
        # engine target is a separate solver; the flange is a one-way boundary a scene cannot
        # yet wire -- ADR 0105).
        return ExhaustSolver(
            stock_exhaust(),
            weather,
            t0_s + np.asarray(spec.load_s, dtype=np.float64),
            spec.load,
            t0_s,
            section=spec.section or "mid_pipe",
        )
    if spec.solver == "vehicle_source":
        assert spec.source is not None and spec.load_s is not None and spec.load is not None
        # Written in seconds after the scene start; the solvers live on the weather's absolute
        # axis, so the offset happens here and nowhere else -- as it does for the throttle below.
        return VehicleSourceSolver(
            VEHICLE_HEAT_SOURCES[spec.source],
            weather,
            t0_s + np.asarray(spec.load_s, dtype=np.float64),
            spec.load,
            t0_s=t0_s,
        )
    if spec.solver in ("airframe", "heat_source", "ram_skin"):
        if spec.solver == "airframe":
            derived = airframe_solver(weather, offset_k=spec.offset_k or 0.0)
        elif spec.solver == "ram_skin":
            assert spec.speed_m_s is not None
            derived = ram_skin_solver(
                weather,
                spec.speed_m_s,
                recovery_factor=spec.recovery_factor or RECOVERY_FACTOR_TURBULENT,
            )
        else:
            assert spec.source is not None and spec.throttle_s is not None
            assert spec.throttle is not None
            # The profile is written in seconds after the *scene start*; the solvers live on the
            # weather's absolute axis, so it is offset here and nowhere else.
            derived = heat_source_solver(
                AERIAL_HEAT_SOURCES[spec.source],
                weather,
                t0_s + np.asarray(spec.throttle_s, dtype=np.float64),
                spec.throttle,
            )
        # Neither constructor takes t0_s, and an airframe node's schedule spans the whole weather
        # file, so without this its initial reading is the temperature at the *file's* start
        # rather than the scene's -- hours out, and perfectly plausible.
        derived.advance(t0_s, 0.0)
        return derived
    assert spec.schedule_s is not None and spec.schedule_k is not None
    times = t0_s + np.asarray(spec.schedule_s, dtype=np.float64)
    return PrescribedSolver(times, np.asarray(spec.schedule_k, dtype=np.float64), t0_s=t0_s)


@dataclass(frozen=True)
class Scene:
    spec: SceneSpec
    weather: WeatherSeries
    #: The grey L1 model (§8.5). **Reach it through :attr:`atmosphere` or, for radiative
    #: transfer, through :attr:`transfer_atmosphere` -- not by this name.** It is spelled out
    #: here because the one-weather guard and `consumers` need the object itself; AT.5 explains
    #: why reading it directly is a mistake once a layered model stands beside it.
    grey_atmosphere: Atmosphere
    targets: Mapping[str, TemperatureSolver]
    t0_s: float
    layered: LayeredAtmosphere | None = None  # MS.1 model on the same weather
    environment: EnvironmentSpec | None = None
    #: The M6.11 field, when the scene config carries a `thermal:` block (M6.12). It is
    #: registered as a consumer through its forcing model, so the one-weather guard sees it.
    thermal: Any = None  # ThermalField; Any avoids importing it into this module's signature
    thermal_surfaces: tuple[str, ...] = ()
    #: ``{surface name: patch}`` for every surface whose config declared one (schema v7, PT.2).
    #: Empty for a scene with no point-wise surface, which is every scene before v7.
    patches: Mapping[str, PlanarPatch] = field(default_factory=dict)
    #: ``{surface name: prim path}`` for the patches that named one, which is what binds a
    #: solved field to geometry at render time.
    patch_prims: Mapping[str, str] = field(default_factory=dict)
    #: ``{surface name: field}`` -- the patched surfaces solved **per cell** on their own library
    #: material and the scene's weather (PT.17). Until this existed a `patch:` block bought a grid
    #: and no solver: `_build_thermal_field` still solved the surface as one facet and only
    #: `car_demo.py` could put a temperature on a cell, in Python. Empty for a scene with no
    #: patch. The per-prim entry in :attr:`thermal` stays as well, so the flat path is unchanged.
    #: A `PlanarThermalField`, or for a surface with ``layers > 1`` the surface layer's
    #: `PatchView` over its stack (PT.12) -- the same ``patch`` / ``advance_to`` / ``sample_at``.
    surface_fields: Mapping[str, Any] = field(default_factory=dict)
    #: ``{surface name: TriangleMeshPatch}`` for every surface whose config declared a `mesh:`
    #: instead of a `patch:` (schema v14, WM.7). Empty for every scene before v14.
    meshes: Mapping[str, Any] = field(default_factory=dict)
    #: ``{surface name: prim path}`` for the meshes that named one.
    mesh_prims: Mapping[str, str] = field(default_factory=dict)
    #: ``{surface name: TriangleMeshField}`` -- the mesh surfaces solved **per cell**, each cell
    #: on its own face's normal (WM.7). The per-prim entry in :attr:`thermal` stays as well.
    mesh_fields: Mapping[str, Any] = field(default_factory=dict)
    #: How the scene's world coordinates relate to east, north and up (schema v8, PT.18). ENU
    #: unless the config says otherwise; the car scenes say Y-up.
    world_frame: WorldFrame = ENU
    #: ``{occluder name: rectangle}`` in world coordinates -- what casts shadows on the patched
    #: surfaces (PT.18). Empty for every scene before v8, and then nothing here changes.
    occluders: Mapping[str, ShadowRectangle] = field(default_factory=dict)
    #: ``{surface name: library material}`` for the §12.3 surfaces (PT.6): what a driver reads
    #: for the numbers the per-prim stack does not keep (k, δ) and the object ADR 0043 makes the
    #: single source of ε.
    surface_materials: Mapping[str, Any] = field(default_factory=dict)
    #: The `nodes:` / `links:` / `sources:` network (schema v9, TC.4), or ``None``. Stepped by
    #: :meth:`advance_targets` beside the targets; read through :meth:`node_temperature_k`.
    network: ThermalNetwork | None = None
    sky_models: Mapping[str, SkyModel] = field(default_factory=dict)  # per band (MS.2)
    extra_consumers: Mapping[str, Any] = field(
        default_factory=dict
    )  # phase-2 objects with .weather

    def __post_init__(self) -> None:
        for name, obj in self.consumers.items():
            w = getattr(obj, "weather", None)
            if w is None:
                continue
            if w is not self.weather:
                raise ValueError(
                    f"consumer {name!r} holds a different WeatherSeries than the scene "
                    "(CLAUDE.md #6: one weather object, injected everywhere)"
                )
        if not self.weather.time_s[0] <= self.t0_s <= self.weather.time_s[-1]:
            raise ValueError("scene start is outside the weather series")

    @property
    def atmosphere(self) -> Atmosphere:
        """The grey L1 model -- and a hard error once a layered model exists beside it (AT.5).

        docs/physics-model.md §8.5 (grey) vs §8.6 (layered exponential sum); ADR 0033.

        ``from_config`` builds **both** models whenever the scene names an environment preset,
        which every scene shipped in this repository does. They do not agree: the grey one takes a
        single band-averaged optical depth, the layered one sums the exponential-sum terms along
        the real slant path. Measured on the shipped aerial scene at 5 km and 20 degrees
        elevation: **tau 0.057 against 0.590**, path radiance 45.1 against 18.7 W/m^2/sr. That is
        the k-distribution rather than a bug in either -- exp(-mean tau) is not the mean of
        exp(-tau) across a band whose lines vary by orders of magnitude, which is what §8.6 exists
        to fix -- but it means the grey model is L1 only, and an attribute that reads as *the*
        scene atmosphere must not be the one handing it out.

        Nothing in the repository was taking the wrong one when this guard was written; both
        current readers want :attr:`atmosphere_preset`, which is the same object for both models.
        The point is that the next caller would have had no way to know, since the primary-looking
        name held the fallback while every render path used the other one.
        """
        if self.layered is not None:
            raise AttributeError(
                "this scene carries a layered atmosphere, so `scene.atmosphere` (the grey L1 "
                "model) is not the one it renders with and the two disagree materially. Use "
                "`scene.transfer_atmosphere` for radiative transfer, `scene.atmosphere_preset` "
                "for the preset, or `scene.grey_atmosphere` if you deliberately want L1 "
                "(docs/physics-model.md §8.5 vs §8.6; roadmap AT.5)"
            )
        return self.grey_atmosphere

    @property
    def transfer_atmosphere(self) -> Atmosphere | LayeredAtmosphere:
        """The model stage 2 must run: the layered one where it exists, else the grey fallback."""
        return self.grey_atmosphere if self.layered is None else self.layered

    @property
    def atmosphere_preset(self) -> AtmospherePreset:
        """The preset both models were built from -- one object, so it cannot disagree.

        This is what a caller almost always wants off a scene: the band coefficients, the aerosol
        term, the scale heights. Reading it through the grey model happened to work and made the
        grey model look load-bearing.
        """
        return self.grey_atmosphere.preset

    @property
    def consumers(self) -> dict[str, Any]:
        out: dict[str, Any] = {"atmosphere": self.grey_atmosphere}
        if self.layered is not None:
            out["layered"] = self.layered
        out.update({f"sky:{k}": v for k, v in self.sky_models.items()})
        out.update({f"target:{k}": v for k, v in self.targets.items()})
        if self.thermal is not None:
            out["thermal"] = self.thermal.forcing_at
        out.update({f"field:{k}": v.field.forcing_at for k, v in self.surface_fields.items()})
        if self.network is not None:
            out["network"] = self.network
        out.update(self.extra_consumers)
        return out

    def surface_bindings(self) -> tuple[tuple[str, Any], ...]:
        """``(prim path, field)`` for every patched surface that named a prim (PT.17).

        This is what a render driver hands to ``IrCamera`` -- through
        ``irsim_isaac.pipeline.point_bridge.bindings_from_scene``, which wraps each pair in a
        ``SurfaceBinding``; the core cannot import that class (CLAUDE.md #1), so the pairs are
        plain tuples here. A patch without a ``prim_path`` is solvable and never reaches a pixel,
        which is the schema's own rule.
        """
        return tuple(
            (self.patch_prims[name], fld)
            for name, fld in self.surface_fields.items()
            if name in self.patch_prims
        )

    def mesh_bindings(self) -> tuple[tuple[str, Any], ...]:
        """``(prim path, field)`` for every mesh surface that named a prim (WM.7).

        What a render driver hands to `irsim_isaac.pipeline.mesh_bridge.MeshPointBridge`, through
        a `MeshBinding`; the core cannot import that class (CLAUDE.md #1), so the pairs are plain
        tuples here, exactly as :meth:`surface_bindings` does for the planar path.
        """
        return tuple(
            (self.mesh_prims[name], fld)
            for name, fld in self.mesh_fields.items()
            if name in self.mesh_prims
        )

    def surface_properties(self, name: str) -> Any:
        """One §12.3 surface's `ThermalProperties`, as the scene solved it (PT.6, ADR 0043)."""
        from irsim.thermal.balance import ThermalProperties

        if self.thermal is None or name not in self.thermal_surfaces:
            raise KeyError(
                f"unknown surface {name!r}; the scene's thermal block declares "
                f"{list(self.thermal_surfaces)}"
            )
        i = self.thermal_surfaces.index(name)
        p = self.thermal.properties
        return ThermalProperties(
            heat_capacity_j_m2_k=float(p.heat_capacity_j_m2_k[i]),
            emissivity=float(p.emissivity[i]),
            solar_absorptivity=float(p.solar_absorptivity[i]),
        )

    def surface_temperature_k(self, name: str, t_s: float) -> float:
        """One named surface's temperature at a render time, float32-narrowed (M6.11)."""
        if self.thermal is None:
            raise ValueError("this scene has no thermal block")
        if name not in self.thermal_surfaces:
            raise KeyError(f"unknown surface {name!r}; scene has {list(self.thermal_surfaces)}")
        return float(self.thermal.temperature_at(t_s)[self.thermal_surfaces.index(name)])

    # -- construction ---------------------------------------------------------------------
    @classmethod
    def from_config(
        cls,
        config: SceneConfig | SceneSpec,
        luts: Mapping[str, BandLUT] | None = None,
        data_dir: str | os.PathLike[str] | None = None,
        quantity: Quantity = "lb",
        weather_override: WeatherSeries | None = None,
        skylights: Mapping[str, Any] | None = None,
        responses: Mapping[str, SpectralResponse] | None = None,
    ) -> Scene:
        """``weather_override`` replaces the file the scene names, for *variant* scenes.

        It exists for sensitivity studies -- the same scene under twice the wind, or under
        overcast -- where loading the file and then mutating it would leave two series in play and
        trip the one-weather guard for the wrong reason. The override is injected everywhere the
        loaded series would have been, so CLAUDE.md #6 holds exactly as before.
        """
        spec = config.scene if isinstance(config, SceneConfig) else config
        weather = (
            weather_override
            if weather_override is not None
            else load_weather_csv(resolve_data_dir(data_dir) / spec.weather_file)  # once
        )
        t0_s = weather.seconds_of(spec.start_utc)
        preset = load_atmosphere_preset(spec.atmosphere_preset)
        atmosphere = Atmosphere(preset, weather, luts)
        targets = {t.name: build_target(t, weather, t0_s) for t in spec.targets}
        layered = None
        environment = None
        sky_models: dict[str, SkyModel] = {}
        if spec.environment_preset is not None:
            environment = load_environment_preset(spec.environment_preset)
            # The camera's own R(λ) decides how much of each band falls in each spectral class
            # (AT.2). Without it `class_weights` falls back to a nominal top-hat and the model
            # describes a different camera: measured on the shipped InSb response, the MWIR
            # `h2o_wing` weight goes 0.0238 -> 0.1303, a **5.5x** change, silently.
            layered = LayeredAtmosphere(preset, weather, luts, responses)
            for band, lut in (luts or {}).items():
                # Scattered sunlight (M11.10, ADR 0086) is supplied by the caller, because
                # integrating it needs the *sensor's* R(λ) and this module does not take sensor
                # configs. `None` is the purely thermal sky this class has always been -- right
                # for an emissive band, where the scattered term is 1.6e-8 of the column's own
                # emission, and a black sky in a reflective one.
                sky_models[band] = SkyModel(
                    layered,
                    environment,
                    band,
                    lut,
                    quantity,
                    skylight=None if skylights is None else skylights.get(band),
                )
        build = _build_thermal_field(spec, weather, t0_s, data_dir)
        patches, patch_prims = {}, {}
        meshes, mesh_prims = {}, {}
        for surface in spec.thermal.surfaces if spec.thermal is not None else ():
            if surface.mesh is not None:
                meshes[surface.name] = build_mesh(surface.mesh)
                if surface.mesh.prim_path:
                    mesh_prims[surface.name] = surface.mesh.prim_path
            if surface.patch is None:
                continue
            patches[surface.name] = build_patch(surface.patch)
            if surface.patch.prim_path:
                patch_prims[surface.name] = surface.patch.prim_path
        world_frame = build_world_frame(spec)
        network = build_network(spec.thermal, weather, t0_s, targets=targets)
        occluders = {
            o.name: build_occluder(o)
            for o in (spec.thermal.occluders if spec.thermal is not None else ())
        }
        return cls(
            spec=spec,
            weather=weather,
            grey_atmosphere=atmosphere,
            targets=targets,
            t0_s=t0_s,
            layered=layered,
            environment=environment,
            sky_models=sky_models,
            thermal=build.field,
            thermal_surfaces=build.names,
            patches=patches,
            patch_prims=patch_prims,
            surface_fields=_build_surface_fields(
                spec, build, patches, world_frame, occluders, data_dir
            ),
            meshes=meshes,
            mesh_prims=mesh_prims,
            mesh_fields=_build_mesh_fields(spec, build, meshes, world_frame),
            world_frame=world_frame,
            occluders=occluders,
            network=network,
            surface_materials=dict(zip(build.names, build.materials, strict=True)),
        )

    @classmethod
    def from_file(
        cls,
        path: str | os.PathLike[str],
        luts: Mapping[str, BandLUT] | None = None,
        data_dir: str | os.PathLike[str] | None = None,
        quantity: Quantity = "lb",
        skylights: Mapping[str, Any] | None = None,
        responses: Mapping[str, SpectralResponse] | None = None,
    ) -> Scene:
        return cls.from_config(
            load_scene_config(path),
            luts,
            data_dir,
            quantity,
            skylights=skylights,
            responses=responses,
        )

    # -- time helpers ---------------------------------------------------------------------
    def solar_terms(self, t_abs_s: float) -> Any:
        """The sun and irradiance every solar term in this scene uses at ``t_abs_s`` (PT.18).

        One call for the per-prim balance, the per-cell shadow and any driver that builds its
        own field, so a frame cannot show a sunlit road under an unlit car.
        """
        from irsim.thermal.scene_forcing import solar_terms_at

        return solar_terms_at(
            self.weather, self.spec.site.latitude_deg, self.spec.site.longitude_deg, t_abs_s
        )

    def weather_at(self, t_rel_s: float) -> WeatherSample:
        return self.weather.at(self.t0_s + float(t_rel_s))

    def advance_targets(self, t_rel_s: float, dt_s: float) -> dict[str, float]:
        """Step every target -- and the network, if any -- from t_rel to t_rel + dt.

        Returns the new temperatures by name; network nodes are included under their own names.
        """
        t = self.t0_s + float(t_rel_s)
        out = {name: solver.advance(t, float(dt_s)) for name, solver in self.targets.items()}
        if self.network is not None:
            self.network.advance(t, float(dt_s))
            out.update(self.network.temperatures_k)
        return out

    def node_temperature_k(self, name: str) -> float:
        """One network node's current temperature (schema v9, TC.4)."""
        if self.network is None:
            raise ValueError("this scene declares no thermal network")
        return self.network.temperature(name)


def _identity(forcing: Any) -> Any:
    return forcing


def wrap_into_weather(weather: WeatherSeries, forcing: Any) -> Any:
    """``forcing`` evaluated inside the weather series for instants before it starts.

    A spin-up needs hours of weather *before* t₀ that a 48 h file usually does not have, so the
    spin-up wraps into the series it has: the synthetic files are a whole number of days long, a
    wrap lands at the same time of day, and the seam is a weather discontinuity rather than a
    clock one. Applied to the spin-up only -- the live forcing stays un-wrapped so a run past the
    end of the weather raises instead of quietly reading yesterday.
    """
    span = float(weather.time_s[-1] - weather.time_s[0])
    first = float(weather.time_s[0])

    def wrapped(t_s: float) -> Any:
        if t_s >= first:
            return forcing(t_s)
        return forcing(first + (t_s - first) % span)

    return wrapped


@dataclass(frozen=True)
class _ThermalBuild:
    """What `_build_thermal_field` made, kept together so the per-cell fields start from it."""

    field: Any = None  # ThermalField
    names: tuple[str, ...] = ()
    forcing: Any = None  # SceneSurfaceForcing
    properties: Any = None  # FacetProperties, one entry per surface
    spun_k: Any = None  # float64 (n_surfaces,) -- the state at t0, before any narrowing
    t0_s: float = 0.0
    #: The weather's content hash, the spin-up cache key every field of this scene shares.
    spin_up_hash: str = ""
    #: The library materials, one per surface, for what the per-prim stack does not keep (k, δ).
    materials: tuple[Any, ...] = ()
    #: Wraps a forcing callable into the weather series for a spin-up that starts before it.
    wrap: Any = _identity


def _build_thermal_field(
    spec: SceneSpec,
    weather: WeatherSeries,
    t0_s: float,
    data_dir: str | os.PathLike[str] | None,
) -> _ThermalBuild:
    """Build M6.11's field for the scene's `thermal:` block, spun up to the scene's own start.

    Returns an empty build when the scene has no thermal block, which is every scene written
    before M6.12 -- the phase-1 prescribed and Newton solvers in ``targets`` are untouched.

    The spin-up **ends at t0_s**, so the field's first query is the state the weather implies at
    the scene's start rather than a transient. That is the whole point of M6.10 and it is worth
    doing here rather than leaving to the caller, because a caller that forgets gets a scene that
    is wrong by kelvins for its first few hours and looks fine.
    """
    from irsim.materials.library import MaterialLibrary
    from irsim.thermal.balance import ThermalProperties
    from irsim.thermal.facets import FacetProperties, spin_up
    from irsim.thermal.field import ThermalField
    from irsim.thermal.scene_forcing import SceneSurfaceForcing, SurfaceOrientation

    block = spec.thermal
    if block is None or not block.surfaces:
        return _ThermalBuild()

    library = MaterialLibrary.load()
    materials = []
    for s in block.surfaces:
        try:
            materials.append(library[s.material])
        except KeyError as exc:
            raise ValueError(
                f"surface {s.name!r} names material {s.material!r}, which the library does not "
                f"have; known materials: {sorted(library.names)}"
            ) from exc
    properties = FacetProperties.stack(
        [ThermalProperties.from_material(m, 300.0, data_dir=data_dir) for m in materials]
    )
    forcing = SceneSurfaceForcing(
        weather=weather,
        latitude_deg=spec.site.latitude_deg,
        longitude_deg=spec.site.longitude_deg,
        orientations=tuple(
            SurfaceOrientation(
                tilt_deg=s.tilt_deg,
                azimuth_deg=s.azimuth_deg,
                shaded=s.shaded,
                vehicle_speed_m_s=s.vehicle_speed_m_s,
                speed_schedule=None
                if s.speed_s is None
                else (tuple(t0_s + t for t in s.speed_s), tuple(s.speed_m_s or ())),
            )
            for s in block.surfaces
        ),
    )

    # The spin-up needs `spin_up_hours` of weather *before* t0, and a 48 h file usually does not
    # have it -- see `wrap_into_weather`. The wrap is applied to the **spin-up only**.
    def wrap(inner: Any) -> Any:
        return wrap_into_weather(weather, inner)

    spun = spin_up(
        properties,
        wrap(forcing),
        weather.content_hash,
        t0_s,
        hours=block.spin_up_hours,
        dt_s=60.0,
    )
    field = ThermalField(properties, forcing, t0_s, spun.temperatures_k, block.tick_s)
    return _ThermalBuild(
        field=field,
        names=tuple(s.name for s in block.surfaces),
        forcing=forcing,
        properties=properties,
        spun_k=spun.temperatures_k,
        t0_s=t0_s,
        spin_up_hash=weather.content_hash,
        wrap=wrap,
        materials=tuple(materials),
    )


def _build_mesh_fields(
    spec: SceneSpec,
    build: _ThermalBuild,
    meshes: Mapping[str, Any],
    world_frame: WorldFrame = ENU,
) -> dict[str, Any]:
    """A per-cell field for every surface that declared a mesh (ADR 0110, WM.7).

    The twin of :func:`_build_surface_fields`, and the same contract: the surface's own facet
    repeated over its cells, on the same library material, the same spun-up starting state and the
    scene's tick. What differs is the parameterisation and one physical term -- each cell takes
    **its own face's normal** for the direct beam and for its sky view, through
    :class:`~irsim.thermal.scene_forcing.MeshCellForcing`, which is exactly the thing one shared
    patch normal cannot express.

    The spun-up state comes from the build in float64 for the same reason it does for a patch: a
    field started from the float32 read-back would sit microkelvins from the prim it claims to
    reproduce.
    """
    from irsim.thermal.facets import FacetProperties, spin_up
    from irsim.thermal.mesh_field import TriangleMeshField
    from irsim.thermal.scene_forcing import MeshCellForcing

    out: dict[str, Any] = {}
    if build.field is None or spec.thermal is None:
        return out
    for i, s in enumerate(spec.thermal.surfaces):
        mesh = meshes.get(s.name)
        if mesh is None:
            continue
        n = mesh.n_cells
        props = build.properties
        cells = FacetProperties(
            heat_capacity_j_m2_k=np.full(n, float(props.heat_capacity_j_m2_k[i])),
            emissivity=np.full(n, float(props.emissivity[i])),
            solar_absorptivity=np.full(n, float(props.solar_absorptivity[i])),
        )
        forcing = MeshCellForcing(
            surfaces=build.field.forcing_at,
            index=i,
            normals_world=mesh.cell_normal,
            frame=world_frame,
        )
        # **Always** spun up per cell, where a patch only bothers under occluders. Every cell of
        # a mesh has its own normal, so no two of them share a forcing history and the per-prim
        # spun-up value is wrong for all of them: a scene starting at 10:00 would open with a
        # uniform pipe and take an hour of its own to grow the gradient it should already have.
        spun = spin_up(
            cells,
            build.wrap(forcing),
            build.spin_up_hash,
            build.t0_s,
            hours=spec.thermal.spin_up_hours,
            dt_s=60.0,
        ).temperatures_k
        out[s.name] = TriangleMeshField(mesh, cells, forcing, build.t0_s, spun, spec.thermal.tick_s)
    return out


def _build_surface_fields(
    spec: SceneSpec,
    build: _ThermalBuild,
    patches: Mapping[str, PlanarPatch],
    world_frame: WorldFrame = ENU,
    occluders: Mapping[str, ShadowRectangle] | None = None,
    data_dir: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """A per-cell field for every surface that declared a patch (ADR 0087, PT.17).

    Each field is the surface's own facet repeated over its cells: the same library material,
    the same spun-up starting state, the same forcing through
    :class:`~irsim.thermal.scene_forcing.CellForcing`, on the scene's tick. With nothing varying
    across the surface the cells reproduce the per-prim value exactly, bit for bit. The terms
    that make a field a *field* enter through the forcing and the solve does not change: with
    ``occluders`` the direct beam is gated per cell (PT.18) -- the spin-up sees the shadow too,
    so a cell under an overhang starts the scene as cold as it has been all morning -- and the
    per-cell sky view (PT.21) and a hot part under the surface (TC.3) follow the same route.

    Occluders reach only the patches authored in the world frame. A patch in a moving frame
    under a scene with occluders is refused by `CellForcing`, not quietly left unshaded.

    The spun-up state is taken in float64 from the build, not read back through
    ``temperature_at`` (float32): a field that started from the narrowed value would sit up to
    15 µK from the prim it claims to reproduce, and the one-millikelvin contract would still
    hold while the bit-identity test did not. Narrowing happens once, at the query (CLAUDE.md #2).
    """
    from irsim.thermal.coupling import FieldMember
    from irsim.thermal.facets import FacetProperties, spin_up
    from irsim.thermal.scene_forcing import CellForcing

    out: dict[str, Any] = {}
    if build.field is None or spec.thermal is None:
        return out
    casters = tuple((occluders or {}).values())
    # PT.15: the surfaces that bound a cabin are solved *together* with it, so they are set
    # aside here and built as one `CoupledFields` after the loop (ADR 0106).
    cabin_spec = spec.thermal.cabin
    panel_order = [p.surface for p in cabin_spec.panels] if cabin_spec is not None else []
    panels: dict[str, Any] = {}
    for i, s in enumerate(spec.thermal.surfaces):
        patch = patches.get(s.name)
        if patch is None:
            continue
        n = patch.n_cells
        props = build.properties
        cells = FacetProperties(
            heat_capacity_j_m2_k=np.full(n, float(props.heat_capacity_j_m2_k[i])),
            emissivity=np.full(n, float(props.emissivity[i])),
            solar_absorptivity=np.full(n, float(props.solar_absorptivity[i])),
        )
        # PT.21: the sky each cell sees, through the same occluders that cast the beam. An
        # unobstructed cell keeps the tilt's own factor exactly; nothing to compute without
        # occluders, and a moving-frame patch is refused by `CellForcing` for the beam already.
        sky_view = None
        if casters and patch.frame == "world":
            from irsim.thermal.skyview import sky_view_factors

            orientation = build.forcing.orientations[i]
            sky_view = sky_view_factors(
                patch,
                orientation.normal_enu(),
                float(build.forcing.sky_view[i]),
                casters,
                world_frame.to_world,
            )
        forcing = CellForcing(
            build.forcing,
            i,
            n,
            patch=patch,
            occluders=casters,
            frame=world_frame,
            sky_view=sky_view,
            penumbra_rays=spec.thermal.penumbra_rays,
        )
        # PH.3: the cells that are standing water stop being road -- their capacity, their
        # optics and their film are the puddle's (ADR 0108).
        film = _film_for(patch, s.film)
        if s.water is not None:
            cells, film = _puddle(patch, s.water, cells, data_dir)
        # PT.11: in-plane conduction from the material's own k and thickness (ADR 0102).
        thermal = build.materials[i].spec.thermal
        conduction = None
        if s.lateral_conduction:
            conduction = lateral_operator(patch, thermal.conductivity_w_mk, thermal.thickness_m)
        if s.layers > 1:
            # PT.12: the material's thickness cut into N slices, one coupled solve (ADR 0103).
            from irsim.thermal.layers import LayerStack, layered_field

            deep: dict[str, Any] = {}
            if s.back is not None:  # PT.15: §6.4's R₂d and T_deep, from the config (ADR 0036)
                t_deep = s.back.deep_temperature_k
                deep = {
                    "back_resistance_m2k_w": s.back.resistance_m2k_w,
                    "deep_temperature_k": float(build.forcing.weather.at(build.t0_s).t_air_k)
                    if t_deep == "ambient"
                    else float(t_deep),
                }
            stack = LayerStack.uniform(
                s.layers,
                thermal.conductivity_w_mk,
                thermal.density_kg_m3,
                thermal.specific_heat_j_kgk,
                thermal.thickness_m,
                **deep,
            )
            optical = ThermalPropertiesOf(props, i)
            coupled = layered_field(
                s.name,
                patch,
                optical,
                stack,
                forcing,
                build.t0_s,
                float(build.spun_k[i]),
                spec.thermal.tick_s,
                lateral=s.lateral_conduction,
                spin_up_hours=spec.thermal.spin_up_hours,
                spin_up_hash=build.spin_up_hash,
                wrap=build.wrap,
            )
            out[s.name] = coupled.fields[s.name]
            continue
        if s.name in panel_order:
            # A cabin panel: its own cells and forcing, held for the coupled solve below. Its
            # start is the per-prim spun state; the whole system is then spun up together, so a
            # per-panel spin-up against an adiabatic back would only be thrown away.
            panels[s.name] = FieldMember(
                s.name, patch, cells, forcing, np.full(n, float(build.spun_k[i])), conduction
            )
            continue
        if casters:
            # The shadow is part of the surface's history, not a term switched on at t0: a cell
            # under an overhang has been under it all morning. So the field is spun up on its
            # own per-cell forcing, through the same wrap the per-prim spin-up uses.
            spun = spin_up(
                cells,
                build.wrap(forcing),
                build.spin_up_hash,
                build.t0_s,
                hours=spec.thermal.spin_up_hours,
                dt_s=60.0,
                conduction=conduction,
            ).temperatures_k
        else:
            spun = np.full(n, float(build.spun_k[i]))
        out[s.name] = PlanarThermalField(
            patch,
            cells,
            forcing,
            build.t0_s,
            spun,
            spec.thermal.tick_s,
            film_kg_m2=film,
            conduction=conduction,
        )
    if cabin_spec is not None:
        out.update(_build_cabin(spec, build, cabin_spec, panels))
    return out


def _build_cabin(
    spec: SceneSpec, build: _ThermalBuild, cabin_spec: Any, panels: Mapping[str, Any]
) -> dict[str, Any]:
    """The cabin and its panels as one coupled solve; every panel comes back as a `PatchView`."""
    from irsim.thermal.cabin import CabinNode, CabinPanel, cabin_field

    assert spec.thermal is not None
    order = [p.surface for p in cabin_spec.panels]
    missing = [name for name in order if name not in panels]
    if missing:  # pragma: no cover - the schema already checks the names and the patches
        raise ValueError(f"cabin {cabin_spec.name!r}: panels {missing} built no field")
    index = {name: i for i, name in enumerate(build.names)}
    node = CabinNode(
        [
            CabinPanel(
                p.surface,
                ThermalPropertiesOf(build.properties, index[p.surface]),
                float(panels[p.surface].patch.area_m2),
                p.inner_resistance_m2k_w,
            )
            for p in cabin_spec.panels
        ],
        volume_m3=cabin_spec.volume_m3,
        glazing_area_m2=cabin_spec.glazing_area_m2,
        glazing_transmittance=cabin_spec.glazing_transmittance,
        air_changes_per_hour=cabin_spec.air_changes_per_hour,
        interior_mass_j_k=cabin_spec.interior_mass_j_k,
        glazing_panel=cabin_spec.glazing_surface,
    )
    glazing = index[cabin_spec.glazing_surface]

    def glazing_solar_at(t_s: float) -> float:
        """The sun on the glazing's own tilt -- the per-prim term, which is what §6.6 transmits."""
        return float(np.atleast_1d(build.forcing(t_s).q_solar_w_m2)[glazing])

    coupled = cabin_field(
        node,
        [panels[name] for name in order],
        lambda t: float(build.forcing.weather.at(t).t_air_k),
        glazing_solar_at,
        build.t0_s,
        spec.thermal.tick_s,
        name=cabin_spec.name,
        spin_up_hours=spec.thermal.spin_up_hours,
        spin_up_hash=build.spin_up_hash,
        wrap=build.wrap,
    )
    # The panels as views, and the whole solve under the cabin's own name: that is where
    # `node_temperature_k` lives, and it is the only handle on a member with no cells.
    out: dict[str, Any] = {name: coupled.fields[name] for name in order}
    out[cabin_spec.name] = coupled
    return out


def ThermalPropertiesOf(props: Any, i: int) -> Any:  # noqa: N802 - reads as a constructor
    """One surface's scalar `ThermalProperties` out of the per-prim stack."""
    from irsim.thermal.balance import ThermalProperties

    return ThermalProperties(
        heat_capacity_j_m2_k=float(props.heat_capacity_j_m2_k[i]),
        emissivity=float(props.emissivity[i]),
        solar_absorptivity=float(props.solar_absorptivity[i]),
    )


def _puddle(
    patch: PlanarPatch, water: Any, cells: Any, data_dir: str | os.PathLike[str] | None
) -> tuple[Any, Any]:
    """``(properties, film)`` with the puddle's cells turned into water (PH.3, ADR 0108).

    A cell inside the region gets the mixed layer's areal capacity ``rho c d`` **added to the
    substrate's**, because the road under a puddle has not gone away and the two are coupled far
    more tightly than a day is long; the water material's own hemispherical emissivity and solar
    absorptivity (ADR 0043: never a second authored copy); and a film of the same depth so the
    latent term has something to evaporate. The cells outside keep the road's numbers, bit for
    bit. The film is laid at t0 like every other (ADR 0101), so the spin-up sees water's optics
    and mass but not its evaporation.
    """
    from irsim.materials.library import MaterialLibrary
    from irsim.thermal.balance import ThermalProperties
    from irsim.thermal.facets import FacetProperties
    from irsim.thermal.still_water import mixed_layer_capacity_j_m2_k

    library = MaterialLibrary.load()
    if water.material not in library.names:
        raise ValueError(
            f"a puddle names material {water.material!r}, which the library does not have; "
            f"known materials: {sorted(library.names)}"
        )
    material = library[water.material]
    optical = ThermalProperties.from_material(material, 300.0, data_dir=data_dir)
    depth_m = float(water.depth_mm) / 1000.0
    inside = np.ones(patch.n_cells, dtype=bool)
    if water.region_m is not None:
        u0, u1, v0, v1 = water.region_m
        uv = patch.local_coords(patch.cell_centres())
        inside = (uv[:, 0] >= u0) & (uv[:, 0] <= u1) & (uv[:, 1] >= v0) & (uv[:, 1] <= v1)
    wet = FacetProperties(
        heat_capacity_j_m2_k=np.where(
            inside,
            float(mixed_layer_capacity_j_m2_k(depth_m)) + cells.heat_capacity_j_m2_k,
            cells.heat_capacity_j_m2_k,
        ),
        emissivity=np.where(inside, float(optical.emissivity), cells.emissivity),
        solar_absorptivity=np.where(
            inside, float(optical.solar_absorptivity), cells.solar_absorptivity
        ),
    )
    return wet, np.where(inside, float(water.depth_mm), 0.0)  # 1 mm of water is 1 kg m^-2


def _film_for(patch: PlanarPatch, film: Any) -> Any:
    """``(n_cells,)`` kg m⁻² of water at t₀ from a `FilmSpec`, or ``None`` for a dry surface."""
    if film is None:
        return None
    depth = np.full(patch.n_cells, float(film.depth_mm))  # 1 mm of water is 1 kg m⁻²
    if film.region_m is not None:
        u0, u1, v0, v1 = film.region_m
        uv = patch.local_coords(patch.cell_centres())
        inside = (uv[:, 0] >= u0) & (uv[:, 0] <= u1) & (uv[:, 1] >= v0) & (uv[:, 1] <= v1)
        depth = np.where(inside, depth, 0.0)
    return depth
