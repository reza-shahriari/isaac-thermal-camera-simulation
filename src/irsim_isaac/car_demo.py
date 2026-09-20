"""A car on asphalt that starts its engine: the scene point-wise temperature exists for.

docs/physics-model.md §6.1, §6.6, §13; roadmap MP.4b; ADR 0087 (the field), ADR 0088 (the spatial
sources and the sky a body blocks), ADR 0089 (the `vehicle_source` solver).

Every frame this project rendered before ADR 0087 carried **one temperature per prim**, and this
stage is the cheapest scene that makes that visible as an error rather than as an opinion. The
bonnet is a single USD prim. By the end of the run it is several kelvin hotter over the engine
block than at its wings, and a per-prim bridge renders that as one flat value.

Three things happen after the key turns, and they are three different mechanisms:

* **The bonnet gains a gradient.** The engine bay is a hot cavity under the skin; ADR 0088's
  configuration factor decides how much of it each cell of the bonnet sees, so the falloff toward
  the wings is *computed from the bay's dimensions* rather than authored as a Gaussian.
* **The asphalt under the car warms.** Same kernel, pointing down: the underbody, the bay's own
  floor and the exhaust run radiate onto the road, and the exhaust's narrow rectangle is what makes
  a stripe rather than a blob. It is slow -- asphalt's time constant is about an hour -- so this is
  a few tenths of a kelvin, which at a 50 mK NETD is still several noise-equivalent steps.
* **The wheels do essentially nothing, and that is the physics.** §6.6 makes tyre heating flexing
  work and brake heating kinetic energy. A car idling in a car park is doing neither, so its tyres
  stay at ambient however long it idles; the front pair picks up a little radiation through the
  arch and nothing else. The readout prints the wheels' rise beside the bonnet's so the difference
  is stated rather than left to be noticed.

**What is modelled coarsely, and where that shows.** The engine bay is one isothermal rectangle at
§6.6's node temperature, standing in for a cluttered cavity that is also convecting hot air onto
the skin; treating it as a near-blackbody radiator is how that convective share is absorbed, and it
is why its emissivity is authored high. A real bay is hottest at the manifold and this one is not
hot anywhere in particular. The bonnet is a flat patch and a real bonnet has a crown, which ADR
0087 bounds: 60 mm of camber over a 1.2 m patch displaces a sample by under a third of a cell.

The stage frame is :mod:`irsim_isaac.airframe`'s: **+X right, +Y up, -Z forward**, so the car faces
-Z and its bonnet is at negative Z.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from irsim.radiometry.constants import SIGMA_SB
from irsim.scene import Scene, wrap_into_weather
from irsim.thermal.conduction import lateral_operator
from irsim.thermal.convection import DEFAULT_CONVECTION, convection_coefficient
from irsim.thermal.facets import FacetForcing, FacetProperties, SpinUpCache, spin_up
from irsim.thermal.longwave import longwave_down
from irsim.thermal.shadow import ShadowRectangle, box_faces, cell_shadow
from irsim.thermal.solar import solar_loading
from irsim.thermal.spatial_sources import (
    RadiantRectangle,
    clamp_view_factor_sum,
    occluded_longwave_flux,
    patch_view_factors,
)
from irsim.thermal.surface_field import PlanarPatch, PlanarThermalField
from irsim_isaac.airframe import Part

__all__ = [
    "CarGeometry",
    "CameraSetup",
    "CarDemoScene",
    "bonnet_patch",
    "ground_patch",
    "build_bonnet_field",
    "build_ground_field",
    "build_car_demo",
    "describe",
    "grids_source",
]

EX = np.array([1.0, 0.0, 0.0])
EY = np.array([0.0, 1.0, 0.0])
EZ = np.array([0.0, 0.0, 1.0])

#: Emissivity of the engine bay seen as a cavity. High on purpose: a cluttered, multiply
#: reflecting enclosure has a near-unity apparent emissivity, and treating it as one is also how
#: the convective share of the bay-to-skin coupling is absorbed into a radiative model (ADR 0088).
BAY_CAVITY_EMISSIVITY = 0.95
#: A dirty painted underbody. ESTIMATED.
UNDERBODY_EMISSIVITY = 0.88


@dataclass(frozen=True)
class CarGeometry:
    """A generic mid-size saloon, in metres. Every number ESTIMATED but ordinary.

    The bonnet is a **separate prim** from the shell because it is the surface that carries the
    field; the rest of the shell is one box and takes one temperature, which is the point of
    contrast. The engine block is not geometry at all -- it is never seen -- it is the radiating
    rectangle of ADR 0088, and giving it a prim would put a hot box in the picture that no camera
    looking at a closed car can see.
    """

    length_m: float = 4.35
    width_m: float = 1.80
    wheel_radius_m: float = 0.32
    wheel_width_m: float = 0.22
    body_height_m: float = 0.72
    cabin_length_m: float = 2.05
    cabin_width_m: float = 1.70
    cabin_height_m: float = 0.52
    bonnet_length_m: float = 1.30
    bonnet_inset_m: float = 0.08  # from each side of the shell
    bonnet_thickness_m: float = 0.06
    #: The bay cavity under the bonnet: how far below the skin, and how much of it there is.
    bay_drop_m: float = 0.20
    bay_half_width_m: float = 0.50
    bay_half_length_m: float = 0.45

    @property
    def floor_y_m(self) -> float:
        """Underside of the shell -- what faces the road."""
        return self.wheel_radius_m - 0.04

    @property
    def bonnet_y_m(self) -> float:
        """Top of the shell box, where the bonnet skin sits."""
        return self.floor_y_m + self.body_height_m

    @property
    def bonnet_centre_z_m(self) -> float:
        return -0.5 * self.length_m + 0.5 * self.bonnet_length_m + 0.25

    @property
    def bonnet_half_width_m(self) -> float:
        return 0.5 * self.width_m - self.bonnet_inset_m

    def parts(self) -> tuple[Part, ...]:
        """The renderable prims and the thermal node each falls back to."""
        half_len = 0.5 * self.length_m
        shell_y = self.floor_y_m + 0.5 * self.body_height_m
        cabin_y = self.bonnet_y_m + 0.5 * self.cabin_height_m
        wheel_z = half_len - 1.05
        wheel_x = 0.5 * self.width_m - 0.5 * self.wheel_width_m
        parts = [
            Part(
                name="shell",
                kind="box",
                centre_m=(0.0, shell_y, 0.0),
                size_m=(self.width_m, self.body_height_m, self.length_m),
                material="car_paint_black",
                thermal_node="shell",
            ),
            Part(
                name="bonnet",
                kind="box",
                centre_m=(
                    0.0,
                    self.bonnet_y_m + 0.5 * self.bonnet_thickness_m,
                    self.bonnet_centre_z_m,
                ),
                size_m=(
                    2.0 * self.bonnet_half_width_m,
                    self.bonnet_thickness_m,
                    self.bonnet_length_m,
                ),
                material="car_paint_black",
                thermal_node="bonnet_fallback",
            ),
            Part(
                name="cabin",
                kind="box",
                centre_m=(0.0, cabin_y, 0.45),
                size_m=(self.cabin_width_m, self.cabin_height_m, self.cabin_length_m),
                material="car_paint_black",
                thermal_node="shell",
            ),
            Part(
                name="windscreen",
                kind="box",
                centre_m=(0.0, cabin_y, 0.45 - 0.5 * self.cabin_length_m),
                size_m=(self.cabin_width_m - 0.06, self.cabin_height_m - 0.06, 0.05),
                material="glass_windshield",
                thermal_node="glass",
                rotate_xyz_deg=(28.0, 0.0, 0.0),
            ),
        ]
        for name, sx, sz in (
            ("wheel_fl", -1.0, -1.0),
            ("wheel_fr", 1.0, -1.0),
            ("wheel_rl", -1.0, 1.0),
            ("wheel_rr", 1.0, 1.0),
        ):
            parts.append(
                Part(
                    name=name,
                    kind="cylinder",
                    centre_m=(sx * wheel_x, self.wheel_radius_m, sz * wheel_z),
                    size_m=(
                        self.wheel_width_m,
                        2.0 * self.wheel_radius_m,
                        2.0 * self.wheel_radius_m,
                    ),
                    material="rubber_tyre",
                    thermal_node="tyre",
                    axis="X",
                )
            )
        return tuple(parts)

    # -- the radiators (ADR 0088) --------------------------------------------------------------

    def engine_bay(self) -> RadiantRectangle:
        """The hot cavity under the bonnet, facing **up** at the skin."""
        return RadiantRectangle(
            centre_m=np.array([0.0, self.bonnet_y_m - self.bay_drop_m, self.bonnet_centre_z_m]),
            u_axis=EX,
            v_axis=EZ,
            half_u_m=self.bay_half_width_m,
            half_v_m=self.bay_half_length_m,
            emissivity=BAY_CAVITY_EMISSIVITY,
        )

    def shadow_casters(self) -> tuple[ShadowRectangle, ...]:
        """The car as the sun sees it: the faces of its shell, bonnet and cabin boxes (PT.18).

        These are what put the car's own shadow on the road and, at a low sun from behind, the
        cabin's shadow on the bonnet. The windscreen sits inside the cabin's box and the wheels
        inside the shell's footprint, so neither adds an edge the boxes do not already cast.
        """
        faces: list[ShadowRectangle] = []
        for part in self.parts():
            if part.kind == "box" and part.name in ("shell", "bonnet", "cabin"):
                faces.extend(box_faces(part.centre_m, part.size_m))
        return tuple(faces)

    def ground_radiators(self) -> tuple[tuple[str, RadiantRectangle], ...]:
        """What the road sees, each paired with the scene target that gives its temperature.

        Three rather than one because the picture they make differs: a broad warm footprint from
        the underbody, a brighter pool under the engine, and a narrow **stripe** from the exhaust
        run, which is the feature that says "this car has been running" rather than "this car is
        warm".
        """
        y = self.floor_y_m
        return (
            (
                "underbody",
                RadiantRectangle(
                    centre_m=np.array([0.0, y, 0.15]),
                    u_axis=EX,
                    v_axis=EZ,
                    half_u_m=0.5 * self.width_m - 0.08,
                    half_v_m=0.5 * self.length_m - 0.25,
                    emissivity=UNDERBODY_EMISSIVITY,
                ),
            ),
            (
                "engine_bay",
                RadiantRectangle(
                    centre_m=np.array([0.0, y + 0.10, self.bonnet_centre_z_m]),
                    u_axis=EX,
                    v_axis=EZ,
                    half_u_m=0.36,
                    half_v_m=0.40,
                    emissivity=BAY_CAVITY_EMISSIVITY,
                ),
            ),
            (
                "exhaust_pipe",
                RadiantRectangle(
                    centre_m=np.array([0.26, y - 0.04, 0.55]),
                    u_axis=EX,
                    v_axis=EZ,
                    half_u_m=0.055,
                    half_v_m=1.30,
                    emissivity=0.80,  # oxidised steel pipe
                ),
            ),
        )


@dataclass(frozen=True)
class CameraSetup:
    """Where the camera stands. A depression angle and a range, not a position to be checked."""

    depression_deg: float = 45.0
    range_m: float = 27.0
    azimuth_deg: float = 22.0  # a few degrees off the nose, so the bonnet is not foreshortened away
    aim_height_m: float = 0.8

    def position_m(self) -> tuple[float, float, float]:
        d = math.radians(self.depression_deg)
        a = math.radians(self.azimuth_deg)
        horizontal = self.range_m * math.cos(d)
        return (
            horizontal * math.sin(a),
            self.aim_height_m + self.range_m * math.sin(d),
            horizontal * math.cos(a),
        )

    def basis(self) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """``(forward, right, up)`` for a camera at :meth:`position_m` aimed at the car."""
        eye = np.asarray(self.position_m(), dtype=np.float64)
        aim = np.array([0.0, self.aim_height_m, 0.0])
        forward = aim - eye
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, np.array([0.0, 1.0, 0.0]))
        right /= np.linalg.norm(right)
        return forward, right, np.asarray(np.cross(right, forward))

    def ground_footprint_m(
        self, vfov_deg: float, hfov_deg: float
    ) -> tuple[float, float, float, float]:
        """``(x_min, x_max, z_min, z_max)`` of the road the frame covers, at y = 0.

        Computed from the frustum's **corner rays**, not from a span about the boresight. An
        estimate along the axes is wrong as soon as the camera is off the nose -- the azimuth
        rotates the footprint, so its bounding box is wider than either field of view implies, and
        the corners are the part that escapes. A patch smaller than the frame makes
        `PointwiseTemperature` raise, which is the right failure and a tedious one to meet half an
        hour into a Kit session, so this is measured rather than guessed.

        Rays at or above the horizon are dropped; with none left the camera is not looking at the
        ground at all and that is an error, not an empty box.
        """
        eye = np.asarray(self.position_m(), dtype=np.float64)
        forward, right, up = self.basis()
        tan_h = math.tan(math.radians(0.5 * hfov_deg))
        tan_v = math.tan(math.radians(0.5 * vfov_deg))
        hits = []
        for sh in (-1.0, 1.0):
            for sv in (-1.0, 1.0):
                d = forward + sh * tan_h * right + sv * tan_v * up
                if d[1] >= -1e-9:
                    continue
                hits.append(eye + d * (-eye[1] / d[1]))
        if not hits:
            raise ValueError(
                f"no frustum corner reaches the ground from {self.depression_deg} deg "
                f"depression with a {vfov_deg} deg vertical field: this camera sees only sky"
            )
        pts = np.stack(hits)
        return (
            float(pts[:, 0].min()),
            float(pts[:, 0].max()),
            float(pts[:, 2].min()),
            float(pts[:, 2].max()),
        )


# ---------------------------------------------------------------------------------------------
# the patches
# ---------------------------------------------------------------------------------------------


def bonnet_patch(geom: CarGeometry, cell_m: float = 0.07) -> PlanarPatch:
    """A grid over the bonnet skin, in world space (the car does not move in this scene)."""
    n_u = max(2, int(round(2.0 * geom.bonnet_half_width_m / cell_m)))
    n_v = max(2, int(round(geom.bonnet_length_m / cell_m)))
    du = 2.0 * geom.bonnet_half_width_m / n_u
    dv = geom.bonnet_length_m / n_v
    return PlanarPatch(
        origin_m=np.array(
            [
                -geom.bonnet_half_width_m,
                geom.bonnet_y_m + geom.bonnet_thickness_m,
                geom.bonnet_centre_z_m - 0.5 * geom.bonnet_length_m,
            ]
        ),
        u_axis=EX,
        v_axis=EZ,
        n_u=n_u,
        n_v=n_v,
        du_m=du,
        dv_m=dv,
        # Enough to claim the skin's own thickness and a little camber, and far too little to
        # reach the cabin roof 0.5 m above or the road 1 m below (ADR 0087's slab rule).
        thickness_m=0.12,
    )


def ground_patch(bounds_m: tuple[float, float, float, float], cell_m: float = 0.30) -> PlanarPatch:
    """A grid over the road spanning ``(x_min, x_max, z_min, z_max)``.

    The bounds come from :meth:`CameraSetup.ground_footprint_m` widened to include the car, so the
    patch is placed where the *camera* looks rather than centred on the origin and hoped for.
    """
    x0, x1, z0, z1 = (float(v) for v in bounds_m)
    across = x1 - x0
    along = z1 - z0
    if across <= 0.0 or along <= 0.0:
        raise ValueError(f"road bounds must be increasing in each axis, got {bounds_m}")
    n_u = max(2, int(round(across / cell_m)))
    n_v = max(2, int(round(along / cell_m)))
    return PlanarPatch(
        origin_m=np.array([x0, 0.0, z0]),
        u_axis=EX,
        v_axis=EZ,
        n_u=n_u,
        n_v=n_v,
        du_m=across / n_u,
        dv_m=along / n_v,
        # The road is flat, and a thin slab is what stops it claiming the car standing on it.
        thickness_m=0.10,
    )


# ---------------------------------------------------------------------------------------------
# the fields
# ---------------------------------------------------------------------------------------------


def _weather_forcing(scene: Scene, sky_view: NDArray[np.float64], surface_t_guess_k: float) -> Any:
    """The environment half of §6.1, shared by both fields: air, convection, downwelling sky."""

    def at(t_abs_s: float) -> tuple[float, NDArray[np.float64], NDArray[np.float64]]:
        sample = scene.weather.at(t_abs_s)
        h = convection_coefficient(
            surface_t_guess_k - sample.t_air_k,
            sample.wind_speed_m_s,
            0.0,
            DEFAULT_CONVECTION,
        )
        q_lw = longwave_down(
            sample.t_air_k,
            sample.vapour_pressure_hpa,
            sample.cloud_fraction,
            sky_view,
            sample.t_air_k,
        )
        return float(sample.t_air_k), np.broadcast_to(h, sky_view.shape), np.asarray(q_lw)

    return at


#: Spun-up field states, keyed on the material, the weather, the geometry and the clock
#: (`_spin_up_key`), so the many builds of one scene in a test session integrate it once.
_SPIN_UP_CACHE = SpinUpCache()


def _spin_up_key(scene: Scene, geom: CarGeometry, patch: PlanarPatch, *parts: Any) -> str:
    """The weather-hash slot of `spin_up_key`, widened with what else decides the answer.

    `spin_up` keys on the material, the weather, t₀, the span and the step. A field spun up with
    the car present also depends on the car's geometry, the grid it lands on, what shades it
    and what radiates onto it; two scenes that differ only there would otherwise share a cache
    entry and one of them would start from the other's road.
    """
    import hashlib

    h = hashlib.sha256(scene.weather.content_hash.encode())
    h.update(repr(geom).encode())
    for arr in (patch.origin_m, patch.u_axis, patch.v_axis):
        h.update(np.ascontiguousarray(arr, dtype=np.float64).tobytes())
    h.update(np.array([patch.n_u, patch.n_v, patch.du_m, patch.dv_m, patch.thickness_m]).tobytes())
    for part in parts:
        h.update(repr(part).encode())
    return h.hexdigest()


def _spun_up_or_uniform(
    scene: Scene,
    key: str,
    properties: FacetProperties,
    forcing_at: Any,
    uniform_k: float,
    spin_up_hours: float | None,
) -> NDArray[np.float64]:
    """The field's starting state: spun up on its own forcing (PT.7), or the uniform value.

    ``spin_up_hours=None`` is the pre-PT.7 start -- the per-prim answer broadcast to every cell,
    the road as it would be with nothing standing on it -- kept so the two can be compared and
    so a caller who wants the old frame 0 can still ask for it, bit for bit.
    """
    n = properties.n_facets
    if spin_up_hours is None:
        return np.full(n, float(uniform_k))
    result = spin_up(
        properties,
        wrap_into_weather(scene.weather, forcing_at),
        key,
        scene.t0_s,
        hours=float(spin_up_hours),
        dt_s=60.0,
        cache=_SPIN_UP_CACHE,
    )
    return np.asarray(result.temperatures_k, dtype=np.float64)


def _emission_factor(
    views: Sequence[Any], source_emissivities: Sequence[float], surface_emissivity: float
) -> NDArray[np.float64]:
    """``1 − Σ_r F_r (1 − ε_r) ε_s`` per cell: one reflection off each grey body back at the cell.

    ADR 0088's kernel is single-bounce: a body radiates ``ε_r σ T_r⁴`` at the cell and blocks the
    sky, and the cell's own emission toward the body was treated as lost. A grey body reflects
    ``1 − ε_r`` of it back, and the cell re-absorbs ``ε_s`` of that. Without the term an ambient
    body of ε 0.88 over an ambient road under an overcast sky is a net *cooler*, which is not a
    thing. One reflection, not the infinite series: for ε ≥ 0.85 on both sides the second bounce
    is under 2 % of the first.
    """
    back = np.zeros_like(np.asarray(views[0], dtype=np.float64))
    for view, eps_r in zip(views, source_emissivities, strict=True):
        back = back + np.asarray(view, dtype=np.float64) * (1.0 - eps_r) * surface_emissivity
    return np.asarray(np.clip(1.0 - back, 0.0, 1.0))


def _node_temperature(scene: Scene, target: str, node: str) -> Any:
    """``t_abs_s -> K`` for one node of a solved target, with the spin-up rule of
    `_radiator_temperature`: before t₀ the node is the air of the hour plus its rise at t₀."""
    solver: Any = scene.targets[target]
    t_air_0 = float(scene.weather.at(scene.t0_s).t_air_k)

    def at(t_abs_s: float) -> float:
        now = float(solver.node_temperature_k(node))
        if t_abs_s >= scene.t0_s:
            return now
        return float(scene.weather.at(t_abs_s).t_air_k) + (now - t_air_0)

    return at


def _radiator_temperature(scene: Scene, name: str) -> Any:
    """``t_abs_s -> K`` for a radiating part: its solved node from t₀ on, and before t₀ -- the
    spin-up, when the node cannot be rewound -- the air of that instant plus the rise the node
    carries at t₀ (zero for a car that has stood cold, which is what a spin-up is)."""
    target = scene.targets[name]
    t_air_0 = float(scene.weather.at(scene.t0_s).t_air_k)

    def at(t_abs_s: float) -> float:
        if t_abs_s >= scene.t0_s:
            return float(target.temperature())
        return float(scene.weather.at(t_abs_s).t_air_k) + (float(target.temperature()) - t_air_0)

    return at


def _solar_forcing(
    scene: Scene,
    patch: PlanarPatch,
    sky_view: NDArray[np.float64],
    casters: tuple[ShadowRectangle, ...],
) -> Any:
    """The direct beam per cell and the diffuse sky per cell -- §6.1's Q_sol for a field (PT.18).

    Until this the car's fields had **no solar term at all**: a noon run rendered a sun-free
    bonnet. The sun is the scene's own (`Scene.solar_terms`, the same call the §12.3 balance
    makes), turned into the stage's Y-up frame by the scene's declared `world_frame`, and the
    beam is gated by `cell_shadow` against the car's own faces plus whatever the config declares.
    The cosine uses the patch's upward normal: both fields face straight up.
    """
    up_enu = np.array([0.0, 0.0, 1.0])

    def at(t_abs_s: float) -> NDArray[np.float64]:
        sun = scene.solar_terms(t_abs_s)
        if sun.above_horizon and casters:
            shade = cell_shadow(patch, scene.world_frame.to_world(sun.direction_enu), casters)
        else:
            shade = np.ones(patch.n_cells)
        return np.asarray(
            solar_loading(up_enu, sun.direction_enu, sun.dni_w_m2, sun.dhi_w_m2, sky_view, shade)
        )

    return at


def _checked_bonnet_patch(patch: PlanarPatch, geom: CarGeometry) -> PlanarPatch:
    """A declared bonnet grid must sit on *this* geometry's bonnet, or the declaration is stale.

    A scene config now carries the bonnet's patch (PT.17) while `CarGeometry` still owns where the
    bonnet is. The two can drift -- someone lengthens the bonnet in Python and the YAML keeps the
    old grid -- and the failure that would produce is quiet: a field solved over a rectangle that
    is partly cabin roof and partly air, rendered onto a prim it no longer covers, with the
    uncovered pixels raising only at render time under strict coverage. So the plane, the axes and
    the footprint are checked here, to a millimetre, and a mismatch names both numbers.
    """
    top_y = geom.bonnet_y_m + geom.bonnet_thickness_m
    span_u, span_v = patch.extent_m
    x0 = -geom.bonnet_half_width_m
    z0 = geom.bonnet_centre_z_m - 0.5 * geom.bonnet_length_m
    problems = []
    if patch.frame != "world":
        problems.append(f"frame is {patch.frame!r}, and this car does not move")
    if abs(float(patch.origin_m[1]) - top_y) > 1e-6:
        problems.append(f"plane y = {float(patch.origin_m[1])}, bonnet top is y = {top_y}")
    along_x = float(np.dot(patch.u_axis, EX)) >= 1.0 - 1e-9
    along_z = float(np.dot(patch.v_axis, EZ)) >= 1.0 - 1e-9
    if not (along_x and along_z):
        problems.append("axes must be +X (across) and +Z (along)")
    width = 2.0 * geom.bonnet_half_width_m
    if abs(float(patch.origin_m[0]) - x0) > 1e-3 or abs(span_u - width) > 1e-3:
        problems.append(
            f"across: origin {float(patch.origin_m[0]):.4f}, span {span_u:.4f} against the "
            f"bonnet's {x0:.4f} and {2.0 * geom.bonnet_half_width_m:.4f}"
        )
    if abs(float(patch.origin_m[2]) - z0) > 1e-3 or abs(span_v - geom.bonnet_length_m) > 1e-3:
        problems.append(
            f"along: origin {float(patch.origin_m[2]):.4f}, span {span_v:.4f} against the "
            f"bonnet's {z0:.4f} and {geom.bonnet_length_m:.4f}"
        )
    if problems:
        raise ValueError(
            "the scene config's bonnet patch does not sit on CarGeometry's bonnet: "
            + "; ".join(problems)
            + ". Update the `patch:` block under the `bonnet` surface to match the geometry."
        )
    return patch


def _checked_road_patch(
    patch: PlanarPatch, bounds_m: tuple[float, float, float, float]
) -> PlanarPatch:
    """A declared road grid must cover the camera's ground footprint (and the car under it).

    The hand-built road used to be sized from the frustum so that `PointwiseTemperature` could
    never raise half an hour into a Kit session; a declared road is whatever the author wrote, so
    the same guarantee is checked here instead, at build time, with the numbers in the message.
    """
    x0, x1, z0, z1 = bounds_m
    corners = np.array([[x0, 0.0, z0], [x1, 0.0, z0], [x0, 0.0, z1], [x1, 0.0, z1]])
    problems = []
    if patch.frame != "world":
        problems.append(f"frame is {patch.frame!r}; a road is authored in world")
    if abs(float(patch.origin_m[1])) > 1e-6:
        problems.append(f"plane y = {float(patch.origin_m[1])}, the road surface is y = 0")
    if not bool(np.all(patch.contains(corners))):
        u0, v0 = float(patch.origin_m[0]), float(patch.origin_m[2])
        span_u, span_v = patch.extent_m
        problems.append(
            f"it spans x {u0:.2f}..{u0 + span_u:.2f}, z {v0:.2f}..{v0 + span_v:.2f} m but the "
            f"camera's footprint plus the car and margin needs x {x0:.2f}..{x1:.2f}, "
            f"z {z0:.2f}..{z1:.2f} m"
        )
    if problems:
        raise ValueError(
            "the scene config's road patch cannot serve this camera: "
            + "; ".join(problems)
            + ". Widen the `patch:` block under the `asphalt` surface."
        )
    return patch


def build_bonnet_field(
    scene: Scene,
    geom: CarGeometry,
    *,
    emissivity: float,
    heat_capacity_j_m2_k: float = 8_000.0,
    solar_absorptivity: float = 0.90,
    cell_m: float = 0.07,
    tick_s: float = 10.0,
    patch: PlanarPatch | None = None,
    casters: tuple[ShadowRectangle, ...] = (),
    spin_up_hours: float | None = None,
    underside_h_w_m2_k: float = 5.0,
    conductivity_w_mk: float = 45.0,
    thickness_m: float = 0.0012,
) -> PlanarThermalField:
    """The bonnet skin: §6.1 per cell, with the bay's radiation weighted by ADR 0088's view factor.

    ``conductivity_w_mk`` and ``thickness_m`` are the skin's own (steel under paint, the
    `car_paint_black` substrate: 45 W/mK, 1.2 mm) and give the cells their in-plane conduction
    (PT.11); ``0`` for the independent-column field the demo had before.

    ``underside_h_w_m2_k`` is the skin's natural convection with the bay air below it (TC.6),
    used when the engine is a solved node with a bay-air node; ESTIMATED at 5 W m⁻² K⁻¹.

    **The bay's reference is the bay at ambient**, so a cold engine contributes exactly zero and
    the bonnet's temperature in frame 0 is set by its own top-side balance alone. Without that
    reference the underside would appear to exchange with a body at 0 K and the panel would start
    the film 30 K too cold, which is both wrong and, after AGC, not obviously wrong.

    ``patch`` is the grid the scene config declared (PT.17), checked against the geometry; when
    the scene declares none the grid is built here as before. The forcing stays in Python until
    `TC.3` gives the engine bay a place in the config: the scene's own field for the bonnet
    (`Scene.surface_fields`) is the same grid under the weather alone.
    """
    if patch is None:
        patch = bonnet_patch(geom, cell_m=cell_m)
    else:
        patch = _checked_bonnet_patch(patch, geom)
    view = patch_view_factors(patch, geom.engine_bay())
    sky_view = np.ones(patch.n_cells)  # a bonnet faces straight up
    environment = _weather_forcing(scene, sky_view, 290.0)
    solar = _solar_forcing(scene, patch, sky_view, casters)
    bay = geom.engine_bay()
    bay_temperature = _radiator_temperature(scene, "engine_bay")
    # TC.6: the skin's underside convects with the bay air when the engine is solved (the bay
    # air is what spikes at key-off, ADR 0100). Two fluids on one cell fold exactly into one
    # convective term: h_eff = h_top + h_under, T_eff = (h_top T_air + h_under T_bay) / h_eff.
    engine = scene.targets["engine_bay"]
    bay_air_at = (
        _node_temperature(scene, "engine_bay", "bay_air")
        if hasattr(engine, "node_temperature_k")
        else None
    )
    # No emission factor here, on purpose: the skin's one emission term is its **top** face
    # toward the sky, and the underside's exchange with the bay is the `reference` convention
    # below (zero net for a bay at ambient). The road's term is different -- its one face is the
    # one the car's grey underside faces -- and takes the factor (ADR 0088 addendum).

    def forcing_at(t_abs_s: float) -> FacetForcing:
        t_air, h, q_lw = environment(t_abs_s)
        t_bay = bay_temperature(t_abs_s)
        reference = bay.emissivity * SIGMA_SB * t_air**4
        q_int = occluded_longwave_flux(
            view,
            t_bay,
            emissivity,
            source_emissivity=bay.emissivity,
            longwave_down_w_m2=reference,
            sky_view=sky_view,
        )
        h_eff: Any = h
        t_eff: Any = t_air
        if bay_air_at is not None:
            h_under = underside_h_w_m2_k
            h_eff = h + h_under
            t_eff = (h * t_air + h_under * bay_air_at(t_abs_s)) / h_eff
        return FacetForcing(
            t_air_k=t_eff,
            h_w_m2_k=h_eff,
            q_solar_w_m2=solar(t_abs_s),
            q_longwave_down_w_m2=q_lw,
            q_internal_w_m2=q_int,
        )

    properties = FacetProperties(
        heat_capacity_j_m2_k=np.full(patch.n_cells, heat_capacity_j_m2_k),
        emissivity=np.full(patch.n_cells, emissivity),
        solar_absorptivity=np.full(patch.n_cells, solar_absorptivity),
    )
    # PT.7: spun up under its own sky and shadow, so a clear night's bonnet starts below the air
    # it has been radiating past all night rather than at it. `None` keeps the old start.
    initial = _spun_up_or_uniform(
        scene,
        _spin_up_key(scene, geom, patch, casters, emissivity, heat_capacity_j_m2_k),
        properties,
        forcing_at,
        float(scene.weather.at(scene.t0_s).t_air_k),
        spin_up_hours,
    )
    return PlanarThermalField(
        patch,
        properties,
        forcing_at,
        scene.t0_s,
        initial,
        tick_s,
        conduction=lateral_operator(patch, conductivity_w_mk, thickness_m),
    )


def build_ground_field(
    scene: Scene,
    geom: CarGeometry,
    patch: PlanarPatch,
    *,
    emissivity: float,
    heat_capacity_j_m2_k: float,
    solar_absorptivity: float,
    initial_k: float,
    tick_s: float = 30.0,
    casters: tuple[ShadowRectangle, ...] = (),
    spin_up_hours: float | None = None,
) -> PlanarThermalField:
    """The road: §12.3's solved asphalt per cell, plus what the car standing on it radiates down.

    ``spin_up_hours`` spins the field up **with the car on it** (PT.7): frame 0 then carries the
    patch a car that has stood there for hours has already made -- under a clear sky the dominant
    feature of a night parking-lot image -- instead of the uniform road no car has stood on.
    ``None`` is that uniform start (``initial_k`` broadcast), bit for bit as before.

    Here the occlusion term is the real one (ADR 0088): the car takes away the sky each cell was
    seeing. Under this scene's overcast that nearly cancels the car's own emission, which is why
    the patch is a few tenths of a kelvin rather than the several kelvin a clear night would give.

    ``underbody``, ``engine_bay`` and ``exhaust_pipe`` are nested regions of one floor pan, not
    three independent bodies (ADR 0090), so their view factors are clamped to sum to at most 1
    per cell before use -- otherwise a cell under the engine bay is credited with more than one
    sky's worth of occlusion and source.
    """
    sky_view = np.ones(patch.n_cells)
    environment = _weather_forcing(scene, sky_view, initial_k)
    solar = _solar_forcing(scene, patch, sky_view, casters)
    radiators = geom.ground_radiators()
    raw_views = [patch_view_factors(patch, rect) for _, rect in radiators]
    clamped_views = clamp_view_factor_sum(raw_views)
    views = tuple(
        (_radiator_temperature(scene, name), view, rect)
        for (name, rect), view in zip(radiators, clamped_views, strict=True)
    )
    # What the road emits at the car's grey underside comes partly back (ADR 0088 addendum). Without
    # this a cold car of ε 0.88 under an overcast sky "cooled" the road beneath its engine bay by
    # 2 K at equilibrium -- the missing reflection, which a spun-up frame 0 made visible.
    leaves = _emission_factor(clamped_views, [rect.emissivity for _, rect in radiators], emissivity)

    def forcing_at(t_abs_s: float) -> FacetForcing:
        t_air, h, q_lw = environment(t_abs_s)
        q_int = np.zeros(patch.n_cells)
        for temperature_of, view, rect in views:
            q_int = q_int + occluded_longwave_flux(
                view,
                temperature_of(t_abs_s),
                emissivity,
                source_emissivity=rect.emissivity,
                longwave_down_w_m2=q_lw,
                sky_view=sky_view,
            )
        return FacetForcing(
            t_air_k=t_air,
            h_w_m2_k=h,
            q_solar_w_m2=solar(t_abs_s),
            q_longwave_down_w_m2=q_lw,
            q_internal_w_m2=q_int,
            emission_factor=leaves,
        )

    properties = FacetProperties(
        heat_capacity_j_m2_k=np.full(patch.n_cells, heat_capacity_j_m2_k),
        emissivity=np.full(patch.n_cells, emissivity),
        solar_absorptivity=np.full(patch.n_cells, solar_absorptivity),
    )
    initial = _spun_up_or_uniform(
        scene,
        _spin_up_key(
            scene, geom, patch, casters, emissivity, heat_capacity_j_m2_k, solar_absorptivity
        ),
        properties,
        forcing_at,
        initial_k,
        spin_up_hours,
    )
    return PlanarThermalField(patch, properties, forcing_at, scene.t0_s, initial, tick_s)


@dataclass
class CarDemoScene:
    """What a render script needs: the prim map, the fields and the camera."""

    geometry: CarGeometry
    camera: CameraSetup
    prim_to_target: dict[str, str] = field(default_factory=dict)
    bonnet_field: PlanarThermalField | None = None
    ground_field: PlanarThermalField | None = None
    car_root: str = "/World/Car"
    road_path: str = "/World/Road"
    camera_path: str = "/World/IrCamera"
    up_axis: str = "Y"

    def surface_bindings(self) -> list[Any]:
        """The MP.3 bindings: which prim takes which field."""
        from irsim_isaac.pipeline.point_bridge import SurfaceBinding

        out = []
        if self.bonnet_field is not None:
            out.append(SurfaceBinding(f"{self.car_root}/bonnet", self.bonnet_field))
        if self.ground_field is not None:
            out.append(SurfaceBinding(self.road_path, self.ground_field))
        return out


# ---------------------------------------------------------------------------------------------
# the stage
# ---------------------------------------------------------------------------------------------


def build_car_demo(
    scene: Scene,
    *,
    geometry: CarGeometry | None = None,
    camera: CameraSetup | None = None,
    vfov_deg: float = 24.8,
    hfov_deg: float = 30.7,
    road_margin_m: float = 4.0,
    ground_cell_m: float = 0.30,
    bonnet_cell_m: float = 0.07,
    bonnet_emissivity: float = 0.92,
    asphalt_emissivity: float = 0.95,
    asphalt_capacity_j_m2_k: float = 60_000.0,
    asphalt_absorptivity: float = 0.88,
    stage: Any = None,
    author: bool = True,
    spin_up: bool = True,
) -> CarDemoScene:
    """Author the stage and build both fields. ``author=False`` builds the physics only.

    ``spin_up`` (PT.7) integrates both fields through the scene's ``spin_up_hours`` with the car
    present -- its shadow, its sky occlusion, its cold radiators -- so frame 0 is a car that has
    stood there, not one that has just arrived. ``False`` is the uniform start the demo had
    before, for comparison.

    **The grids come from the scene config when it declares them** (PT.17): a `patch:` block
    under the `bonnet` and `asphalt` surfaces. Each is checked -- the bonnet grid against this
    geometry's bonnet, the road grid against the camera's ground footprint -- because a declared
    grid can go stale in ways that render plausibly. A scene that declares neither gets the
    hand-built grids this function has always made, sized from the camera so a patch smaller
    than the frame cannot make `PointwiseTemperature` raise half an hour into a Kit session.
    The ``*_cell_m`` arguments apply only to those hand-built grids.
    """
    geom = geometry or CarGeometry()
    cam = camera or CameraSetup()
    if not np.allclose(scene.world_frame.up, EY):
        raise ValueError(
            "the car stage is +Y up (docstring: +X right, +Y up, -Z forward) but the scene's "
            f"`world_frame.up` is {tuple(scene.world_frame.up)}; declare "
            "`world_frame: {up: [0, 1, 0], north: [0, 0, -1]}` (schema v8) or the sun would "
            "come from the wrong side of the car"
        )
    # What shades the fields: the car's own faces and whatever the config declares (PT.18).
    casters = geom.shadow_casters() + tuple(scene.occluders.values())
    x0, x1, z0, z1 = cam.ground_footprint_m(vfov_deg, hfov_deg)
    # The union with the car's own footprint, so the patch holds the radiators even if the camera
    # is later moved somewhere that does not see them.
    half_w = 0.5 * geom.width_m
    half_l = 0.5 * geom.length_m
    bounds = (
        min(x0, -half_w) - road_margin_m,
        max(x1, half_w) + road_margin_m,
        min(z0, -half_l) - road_margin_m,
        max(z1, half_l) + road_margin_m,
    )
    declared_road = scene.patches.get("asphalt")
    road_grid = (
        ground_patch(bounds, cell_m=ground_cell_m)
        if declared_road is None
        else _checked_road_patch(declared_road, bounds)
    )
    demo = CarDemoScene(geometry=geom, camera=cam)

    # §12.3 solved the asphalt as one surface; the field starts from that spun-up state, so the
    # road does not begin the film at the air temperature having forgotten yesterday (M6.10).
    asphalt_k = scene.surface_temperature_k("asphalt", scene.t0_s)
    hours = None
    if spin_up:
        hours = scene.spec.thermal.spin_up_hours if scene.spec.thermal is not None else 48.0

    demo.bonnet_field = build_bonnet_field(
        scene,
        geom,
        emissivity=bonnet_emissivity,
        cell_m=bonnet_cell_m,
        patch=scene.patches.get("bonnet"),
        casters=casters,
        spin_up_hours=hours,
    )
    demo.ground_field = build_ground_field(
        scene,
        geom,
        road_grid,
        emissivity=asphalt_emissivity,
        heat_capacity_j_m2_k=asphalt_capacity_j_m2_k,
        solar_absorptivity=asphalt_absorptivity,
        initial_k=asphalt_k,
        casters=casters,
        spin_up_hours=hours,
    )

    demo.prim_to_target = {
        f"{demo.car_root}/{part.name}": part.thermal_node for part in geom.parts()
    }
    demo.prim_to_target[demo.road_path] = "asphalt"

    if author:
        from pxr import UsdGeom

        from irsim_isaac.airframe import author_parts
        from irsim_isaac.stage import bind_visible_look

        # +Y up, as every other stage in this repository and as `airframe.Part` assumes. It is not
        # cosmetic: Kit's default is +Z, and with it `azimuth_from_rays` gets an up vector parallel
        # to its own default forward and raises -- which is the loud version of the failure. The
        # quiet version is a scene whose geometry is on its side while the radiometry is not.
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
        author_parts(stage, demo.car_root, geom.parts(), look_binder=bind_visible_look)
        _author_road(stage, demo.road_path, road_grid)
        _author_camera(stage, demo.camera_path, cam)
    return demo


def _author_road(stage: Any, path: str, grid: PlanarPatch) -> None:
    """A thin slab whose **top face is y = 0**, which is the plane the ground patch lives on.

    A slab rather than a `UsdGeom.Plane` because a plane is single-sided and this build's normals
    AOV is read two-sided (`orient_to_viewer`); a box removes the question.
    """
    from pxr import Gf, Sdf, UsdGeom

    from irsim_isaac.stage import bind_visible_look

    thickness = 0.04
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    cube.CreateExtentAttr([Gf.Vec3f(-0.5, -0.5, -0.5), Gf.Vec3f(0.5, 0.5, 0.5)])
    xform = UsdGeom.Xformable(cube)
    xform.ClearXformOpOrder()
    across_m, along_m = grid.extent_m
    centre_x = float(grid.origin_m[0] + 0.5 * across_m)
    centre_z = float(grid.origin_m[2] + 0.5 * along_m)
    xform.AddTranslateOp().Set(Gf.Vec3d(centre_x, -0.5 * thickness, centre_z))
    xform.AddScaleOp().Set(Gf.Vec3f(float(across_m), thickness, float(along_m)))
    prim = cube.GetPrim()
    prim.CreateAttribute("thermal:material", Sdf.ValueTypeNames.String).Set("asphalt_dry")
    bind_visible_look(stage, cube, "asphalt_dry")


def _author_camera(stage: Any, path: str, cam: CameraSetup) -> None:
    """Place the camera and aim it at the car. `IrCamera` writes its optics, not its pose.

    The aim is an **azimuth/elevation** construction (`look_at_quaternion`) rather than the minimal
    rotation onto the boresight: the minimal rotation rolls the horizon, and this frame has a road
    running across it whose edge would tilt for no reason a viewer could name.
    """
    from pxr import Gf, UsdGeom

    from irsim_isaac.aircraft_pass import look_at_quaternion

    eye = np.asarray(cam.position_m(), dtype=np.float64)
    aim = np.array([0.0, cam.aim_height_m, 0.0])
    q = look_at_quaternion(aim - eye)
    camera = UsdGeom.Camera.Define(stage, path)
    xform = UsdGeom.Xformable(camera)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(*(float(v) for v in eye)))
    xform.AddOrientOp().Set(Gf.Quatf(float(q[0]), Gf.Vec3f(*(float(v) for v in q[1:]))))
    camera.GetClippingRangeAttr().Set(Gf.Vec2f(0.05, 1.0e5))


# ---------------------------------------------------------------------------------------------
# the readout
# ---------------------------------------------------------------------------------------------


def grids_source(scene: Scene) -> str:
    """Where the demo's grids came from, for the render script's readout."""
    declared = sorted(name for name in ("bonnet", "asphalt") if name in scene.patches)
    if not declared:
        return "grids sized from the camera in Python (the scene config declares no patch)"
    return f"grids declared in the scene config ({', '.join(declared)}), checked against geometry"


def describe(demo: CarDemoScene, scene: Scene, t_abs_s: float) -> dict[str, Any]:
    """What the scene is doing at one instant, in physical units, for the render script to print.

    It reports the **wheels beside the bonnet** on purpose. A viewer expects a running car's wheels
    to glow, and in a car park they do not: §6.6 makes tyre heating flexing work and brake heating
    kinetic energy, and a stationary vehicle is doing neither. Printing the number is how the model
    says so instead of leaving a viewer to conclude the wheels are broken.
    """
    bonnet = demo.bonnet_field
    ground = demo.ground_field
    if bonnet is None or ground is None:  # pragma: no cover - build_car_demo always sets both
        raise RuntimeError("describe needs a built demo")
    t_air = float(scene.weather.at(t_abs_s).t_air_k)
    b = np.asarray(bonnet.temperature_at(t_abs_s), dtype=np.float64)
    g = np.asarray(ground.temperature_at(t_abs_s), dtype=np.float64)

    # "Far" road: the outer tenth of the patch, which no radiator reaches.
    grid = ground.patch
    centres = grid.cell_centres()
    distance = np.linalg.norm(centres[:, [0, 2]] - np.array([0.0, 0.15]), axis=1)
    far = distance > 0.9 * distance.max()
    near = distance < 1.6

    return {
        "t_s": float(t_abs_s - scene.t0_s),
        "t_air_k": t_air,
        "engine_bay_k": float(scene.targets["engine_bay"].temperature()),
        "exhaust_pipe_k": float(scene.targets["exhaust_pipe"].temperature()),
        "underbody_k": float(scene.targets["underbody"].temperature()),
        "tyre_k": float(scene.targets["tyre"].temperature()),
        "tyre_rise_k": float(scene.targets["tyre"].temperature() - t_air),
        "bonnet_hot_k": float(b.max()),
        "bonnet_cold_k": float(b.min()),
        "bonnet_gradient_k": float(b.max() - b.min()),
        "bonnet_rise_k": float(b.max() - t_air),
        "road_near_k": float(g[near].max()) if near.any() else float("nan"),
        "road_far_k": float(g[far].mean()),
        "road_patch_k": float(g[near].max() - g[far].mean()) if near.any() else float("nan"),
    }
