"""Aerial thermal bridge: target solver temperatures -> the G-buffer's ``temperature_k`` plane.

docs/physics-model.md §13.1, §6.5; roadmap M10.18 (phase 1); ADR 0014 (why this is a table and
not an emissive colour), ADR 0060 (the coupling), CLAUDE.md non-negotiable #6 (one weather object).

The renderer cannot carry a temperature. Every colour AOV on this build is float16 and
exposure-scaled, which at 300 K quantises to about 100 mK against a 50 mK NETD (ADR 0014), so the
surface temperature reaches the kernels the other way round: the renderer transports an exact
integer **instance id** per pixel, and this module keeps a **float32 table indexed by that id**,
filled from the M6.6 solvers that the M6.17 :class:`~irsim.scene.Scene` built on the one shared
``WeatherSeries``. Temperature is then exact at facet granularity with no quantisation anywhere.

**Thermal time and render time are different clocks.** Surface temperature moves on a scale of
minutes; frames arrive every few tens of milliseconds. Advancing a solver per frame would be
wasted work and would make the result depend on the frame rate. The solvers therefore tick on a
coarse schedule (1 Hz by default) and the bridge linearly interpolates between the bracketing
ticks at render time. For the phase-1 aerial targets the interpolation error is negligible against
the 10 mK budget -- a Newton node with tau = 900 s curves by ~3 microkelvin across a 1 s tick --
and it is bounded and reportable rather than hidden: see :meth:`AerialThermalBridge.tick_error_k`.

**The sky is not a prim.** Phase 1 renders aerial targets against sky, and ADR 0044/MS.2 already
give the apparent sky temperature as a function of elevation. Background pixels (instance id 0,
``sky_mask`` set) therefore take ``T_sky(theta)`` evaluated from each pixel's own ray direction --
no emissive dome geometry, which would only reintroduce the fp16 colour path this design exists to
avoid, and would be wrong at the horizon where the elevation gradient is steepest. A background ray
pointing *below* the horizon is not sky at all; it takes the environment preset's ground
temperature instead (see :meth:`AerialThermalBridge.background_temperature_k`).

**Cloud, when a seed is given.** MS.3's structured cloud (ADR 0070) has existed since M7 and was
reachable only from the engine-free scene generator; a rendered frame got the clear-sky profile and
nothing else. That matters for this application specifically: against a sky background the dominant
false alarm is not sensor noise, it is cloud edge. Passing ``cloud_seed`` attaches a
:class:`~irsim.atmosphere.cloud.SkyFixedCloud` -- fixed to the *sky*, so a slewing mount sweeps
across it and a target crosses in front of it, where an image-plane field would travel with the
camera and never be crossed at all. Without a seed the background stays clear sky and every
existing frame is bit-identical.

Phase 2's full :mod:`thermal_bridge` (M10.3) replaces the per-prim solver map with a ThermalField
over a facet mesh; this module deliberately does less.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from irsim.atmosphere.cloud import SkyFixedCloud, generate_sky_cloud, sky_angles
from irsim.atmosphere.sea import SeaModel
from irsim.atmosphere.sky import SkyModel
from irsim.pipeline.environment import ground_temperature_k
from irsim.scene import Scene
from irsim_isaac.pipeline.material_ids import BACKGROUND_INSTANCE_ID, labels_to_paths

__all__ = [
    "DEFAULT_TICK_HZ",
    "TickBracket",
    "AerialThermalBridge",
    "elevation_from_rays",
    "azimuth_from_rays",
]

#: Thermal tick rate. Surface temperature is a minutes-scale quantity; 1 Hz is already far finer
#: than anything the energy balance resolves, and it decouples the result from the frame rate.
DEFAULT_TICK_HZ = 1.0


@dataclass(frozen=True)
class TickBracket:
    """The two solver ticks a render time falls between, per target."""

    t_prev_s: float
    t_next_s: float
    prev_k: dict[str, float]
    next_k: dict[str, float]

    def interpolate(self, t_rel_s: float) -> dict[str, float]:
        span = self.t_next_s - self.t_prev_s
        if span <= 0.0:
            return dict(self.next_k)
        frac = (float(t_rel_s) - self.t_prev_s) / span
        frac = min(max(frac, 0.0), 1.0)
        return {
            name: self.prev_k[name] + frac * (self.next_k[name] - self.prev_k[name])
            for name in self.next_k
        }


def azimuth_from_rays(
    ray_dirs: Any, up: Any = (0.0, 1.0, 0.0), forward: Any = (0.0, 0.0, -1.0)
) -> NDArray[np.float64]:
    """Azimuth of each ray about ``up``, radians in [0, 2pi), measured from ``forward``.

    Delegates to :func:`irsim.atmosphere.cloud.sky_angles` so that this and the visible dome
    (which bakes the same cloud field into a texture) cannot drift apart on a convention.
    """
    d = np.asarray(ray_dirs, dtype=np.float64)
    if d.ndim != 3 or d.shape[2] != 3:
        raise ValueError(f"ray_dirs must be (H, W, 3), got {d.shape}")
    return sky_angles(d, up, forward)[1]


def elevation_from_rays(ray_dirs: Any, up: Any = (0.0, 1.0, 0.0)) -> NDArray[np.float64]:
    """Elevation angle (radians) of each per-pixel ray above the horizon.

    ``ray_dirs`` are the unit vectors :func:`irsim_isaac.pipeline.gbuffer_isaac.ray_directions`
    produces -- pointing **away** from the camera -- so a ray aimed at the zenith gives +pi/2 and
    one aimed at the ground -pi/2. The sky model is a function of this angle (ADR 0044), which is
    why the sky is evaluated per pixel rather than as one number for the frame: across a 50 degree
    field the clear-sky apparent temperature changes by tens of kelvin.
    """
    d = np.asarray(ray_dirs, dtype=np.float64)
    if d.ndim != 3 or d.shape[2] != 3:
        raise ValueError(f"ray_dirs must be (H, W, 3), got {d.shape}")
    u = np.asarray(up, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(u))
    if norm == 0.0:
        raise ValueError("up must be a non-zero vector")
    return np.asarray(np.arcsin(np.clip(np.sum(d * (u / norm), axis=2), -1.0, 1.0)))


class AerialThermalBridge:
    """Per-prim solver temperatures and the sky model, assembled into ``temperature_k``.

    ``prim_to_target`` maps a USD prim path to the name of one of ``scene.targets``. Every prim
    that is not in the map keeps the sky/background treatment if it is background and otherwise
    raises when ``strict`` -- a rendered prim with no thermal node has no temperature at all, and
    guessing one is the silent failure this project exists to avoid.
    """

    def __init__(
        self,
        scene: Scene,
        prim_to_target: Mapping[str, str],
        *,
        band: str | None = None,
        sky: SkyModel | None = None,
        tick_hz: float = DEFAULT_TICK_HZ,
        cloud_seed: int | None = None,
        sea: SeaModel | None = None,
    ) -> None:
        # A prim's name resolves to a phase-1 *target solver* or to a phase-2 §12.3 *thermal
        # surface* (M10.3). Both are "a thing with a temperature" as far as the G-buffer is
        # concerned, and keeping them in one map is what lets a stage mix a solved facet with a
        # scripted drone without the caller sorting them. A name in **both** is refused rather
        # than resolved by precedence: whichever won, the other would be silently ignored.
        wanted = set(prim_to_target.values())
        surfaces = set(scene.thermal_surfaces)
        both = wanted & set(scene.targets) & surfaces
        if both:
            raise ValueError(
                f"ambiguous thermal names {sorted(both)}: each is both a target solver and a "
                "§12.3 thermal surface in this scene, and resolving it by precedence would "
                "silently ignore one of them. Rename one."
            )
        unknown = wanted - set(scene.targets) - surfaces
        if unknown:
            raise ValueError(
                f"prim_to_target names things the scene does not define: {sorted(unknown)}; "
                f"the scene has targets {sorted(scene.targets)} and thermal surfaces "
                f"{sorted(surfaces)}"
            )
        if tick_hz <= 0.0:
            raise ValueError("tick_hz must be positive")

        if sky is None and band is not None:
            sky = scene.sky_models.get(band)
        if sky is not None and sky.weather is not scene.weather:
            raise ValueError(
                "the sky model holds a different WeatherSeries than the scene "
                "(CLAUDE.md #6: one weather object, injected everywhere) -- a scene cannot run "
                "summer weather in the target solvers and winter weather in the sky"
            )

        if sea is not None:
            if sky is None:
                raise ValueError(
                    "a sea needs a sky model: the sea is mostly reflected sky, and the two must "
                    "come from one object or they will disagree about the weather (ADR 0078)"
                )
            if sea.sky is not sky:
                raise ValueError(
                    "the SeaModel reflects a different SkyModel than this bridge renders "
                    "(CLAUDE.md #6). The sea would show a sky that is not in the picture."
                )

        self.scene = scene
        self.prim_to_target = dict(prim_to_target)
        #: Which of the mapped names are §12.3 thermal surfaces rather than target solvers.
        #: A surface has no per-name bracket here: the scene's one `ThermalField` integrates every
        #: facet together (the facets share a forcing, and a two-node wall's back face is another
        #: facet's front), so the bridge advances the *field* and reads each surface out of it.
        self.surface_names = frozenset(wanted & surfaces)
        self.sky = sky
        self.sea = sea
        self.band = band
        self.tick_s = 1.0 / float(tick_hz)
        self._t_rel_s = 0.0
        # Built once and never regenerated: a cloud field that changed with time would flicker,
        # and the weather's cloud *fraction* moves on the hour, not on the frame.
        self.cloud: SkyFixedCloud | None = None
        if cloud_seed is not None:
            if sky is None:
                raise ValueError(
                    "cloud needs a sky model: the covered pixels read eps L_B(T_base) + tau "
                    "L_clear, and both terms come from it (MS.3, ADR 0070)"
                )
            environment = scene.environment
            beta = 1.8 if environment is None else environment.clouds.beta
            self.cloud = generate_sky_cloud(
                beta, float(scene.weather.at(scene.t0_s).cloud_fraction), int(cloud_seed)
            )

        # The first bracket has to name **everything a tick will report**, not just the targets.
        # `Scene.advance_targets` returns the network's nodes under their own names as well
        # (TC.4), so seeding this from `scene.targets` alone left the first tick's `next_k` six
        # names wider than its `prev_k`, and `interpolate` raised `KeyError` on the first frame of
        # any scene with a `nodes:` block. Every car render has failed that way since TC.4.
        initial = {name: float(s.temperature()) for name, s in scene.targets.items()}
        if scene.network is not None:
            initial.update(scene.network.temperatures_k)
        self._bracket = TickBracket(0.0, 0.0, dict(initial), dict(initial))
        self._advance_one_tick()
        self._advance_field_to(0.0)

    # -- the thermal clock ----------------------------------------------------------------

    @property
    def t_rel_s(self) -> float:
        """Current render time, seconds since the scene start."""
        return self._t_rel_s

    @property
    def bracket(self) -> TickBracket:
        return self._bracket

    def _advance_field_to(self, t_rel_s: float) -> None:
        """Push the §12.3 `ThermalField` up to a render time (M10.3).

        Separate from the target bracket because the field is a different kind of object: it
        refuses a query past its last tick rather than solving on demand, precisely so that a
        renderer asking many times per tick cannot change the answer by asking. So the advance is
        explicit, happens once per clock move, and is in **absolute** weather-axis time -- the
        field was spun up on that axis and a relative time would read a different hour of the day.
        """
        if not self.surface_names or self.scene.thermal is None:
            return
        self.scene.thermal.advance_to(self.scene.t0_s + float(t_rel_s))

    def _advance_one_tick(self) -> None:
        b = self._bracket
        t_next = b.t_next_s + self.tick_s
        new = self.scene.advance_targets(b.t_next_s, self.tick_s)
        self._bracket = TickBracket(b.t_next_s, t_next, dict(b.next_k), {**b.next_k, **new})

    def advance_to(self, t_rel_s: float) -> dict[str, float]:
        """Move render time to ``t_rel_s``, ticking the solvers only as far as needed.

        Time may not run backwards: the solvers are stateful (a Newton node integrates), so a
        rewind would silently produce a different history than a forward run of the same scene.
        """
        t = float(t_rel_s)
        if t < self._t_rel_s - 1e-9:
            raise ValueError(
                f"cannot rewind the thermal clock from {self._t_rel_s} s to {t} s: the solvers "
                "are stateful, so replaying a frame needs a fresh Scene"
            )
        while t > self._bracket.t_next_s + 1e-12:
            self._advance_one_tick()
        self._advance_field_to(t)
        self._t_rel_s = t
        return self.temperatures()

    def temperatures(self) -> dict[str, float]:
        """Every mapped node's temperature at the current render time.

        Targets come from the tick bracket; §12.3 thermal surfaces are read from the scene's
        ``ThermalField`` (M10.3). **The two use different time bases and that is the trap**: the
        bracket runs on time *relative* to the scene start, and `ThermalField.temperature_at`
        takes weather-axis *absolute* time, so the surface lookup adds ``t0_s``. Get it wrong and
        the render shows a different hour of the day with no other symptom.
        """
        out = dict(self._bracket.interpolate(self._t_rel_s))
        absolute = self.scene.t0_s + self._t_rel_s
        for name in self.surface_names:
            out[name] = self.scene.surface_temperature_k(name, absolute)
        return out

    def tick_error_k(self, name: str) -> float:
        """Bound on the interpolation error for one target across the current tick.

        Linear interpolation of a function with curvature ``f''`` over a step ``h`` is wrong by at
        most ``f'' h^2 / 8``. Estimated here from the bracket itself, so it is a reported number
        rather than an assumption; an exponential node with tau = 900 s and a 1 s tick gives a few
        microkelvin, which is why 1 Hz is enough for phase 1.
        """
        if name in self.surface_names:
            # A thermal surface is interpolated inside the ThermalField over its own solver step,
            # not across this bridge's tick, so this bridge contributes no error for it.
            return 0.0
        solver = self.scene.targets[name]
        tau = float(getattr(solver, "tau_s", 0.0) or 0.0)
        if tau <= 0.0:
            return 0.0
        delta = abs(self._bracket.next_k[name] - self._bracket.prev_k[name])
        # |f''| = |f'| / tau for an exponential, and |f'| ~ delta / tick
        return float(delta / tau * self.tick_s / 8.0)

    # -- the facet table ------------------------------------------------------------------

    def facet_temperatures(self) -> dict[str, float]:
        """Prim path -> temperature at the current render time."""
        temps = self.temperatures()
        return {path: temps[target] for path, target in self.prim_to_target.items()}

    def facet_table(
        self,
        id_to_labels: Mapping[Any, Any] | None,
        *,
        fill_k: float = 0.0,
        strict: bool = True,
    ) -> NDArray[np.float32]:
        """Float32 temperature indexed by instance id -- the array a Warp kernel binds.

        ``fill_k`` is what an id with no thermal node gets. It is 0 K rather than something
        plausible on purpose: if it ever reaches a radiance kernel the result is obviously,
        loudly wrong instead of a believable image of the wrong scene. With ``strict`` (the
        default) such an id raises here instead.
        """
        paths = labels_to_paths(id_to_labels)
        by_path = self.facet_temperatures()
        size = (max(paths) if paths else 0) + 1
        table = np.full(size, float(fill_k), dtype=np.float32)
        missing: list[str] = []
        for ident, path in paths.items():
            if ident == BACKGROUND_INSTANCE_ID:
                continue
            if path in by_path:
                table[ident] = np.float32(by_path[path])
            else:
                missing.append(path)
        if strict and missing:
            raise KeyError(
                f"rendered prims have no thermal node: {sorted(missing)}. Add them to "
                "prim_to_target or to the scene's targets -- a surface with no temperature "
                "cannot be given a plausible one."
            )
        return table

    # -- the plane ------------------------------------------------------------------------

    def temperature_plane(
        self,
        instance_ids: Any,
        id_to_labels: Mapping[Any, Any] | None,
        *,
        sky_mask: Any | None = None,
        elevation_rad: Any | None = None,
        azimuth_rad: Any | None = None,
        fill_k: float = 0.0,
        strict: bool = True,
    ) -> NDArray[np.float32]:
        """The G-buffer's ``temperature_k``: facet temperatures on geometry, T_sky on the sky.

        ``elevation_rad`` is the per-pixel ray elevation (:func:`elevation_from_rays`). Without a
        sky model, or without elevations, the masked pixels keep ``fill_k`` and the caller is
        asserting it will supply the sky itself.
        """
        ids = np.asarray(instance_ids)
        if not np.issubdtype(ids.dtype, np.integer):
            raise TypeError(f"instance_ids must be an integer plane, got {ids.dtype}")
        table = self.facet_table(id_to_labels, fill_k=fill_k, strict=strict)
        clipped = np.clip(ids, 0, table.size - 1)
        plane: NDArray[np.float32] = np.asarray(table[clipped], dtype=np.float32)

        mask = None if sky_mask is None else np.asarray(sky_mask, dtype=bool)
        if mask is None:
            mask = ids == BACKGROUND_INSTANCE_ID
        if mask.any() and self.sky is not None and elevation_rad is not None:
            plane = np.asarray(
                np.where(mask, self.background_temperature_k(elevation_rad, azimuth_rad), plane),
                dtype=np.float32,
            )
        return plane

    def background_temperature_k(
        self, elevation_rad: Any, azimuth_rad: Any = None
    ) -> NDArray[np.float64]:
        """Apparent temperature of a pixel that hit no geometry, split at the horizon.

        A ray with positive elevation that hits nothing is looking at sky, and takes MS.2's
        ``T_sky(theta)`` (ADR 0044). A ray with **negative** elevation that hits nothing is looking
        at *ground* beyond the scene, not at sky: the sky model is only defined on [0, 90] degrees
        and extrapolating it downwards would report a cold sky where the ground is, inverting the
        contrast of anything silhouetted against it. Those pixels take ``T_ground`` from the
        environment preset's ground mode, the same quantity ADR 0045's reflected term uses -- one
        temperature for the whole ground, which is all phase 1 claims.

        With a cloud field attached *and* ``azimuth_rad`` supplied, the above-horizon pixels go
        through MS.3's structured form instead. Azimuth is required rather than optional for that
        path: without it the field could only be sampled by elevation, which would band the sky in
        horizontal stripes -- a worse picture than no cloud at all, and one that looks deliberate.
        """
        if self.sky is None:
            raise ValueError("this bridge has no sky model; pass band= or sky= at construction")
        t_abs = self.scene.t0_s + self._t_rel_s
        elev = np.asarray(elevation_rad, dtype=np.float64)
        below = elev < 0.0
        if self.sea is not None:
            # Below the horizon is sea, and a sea has no single temperature: every pixel takes the
            # profile at its own depression angle (ADR 0078). Rays between the horizon and level --
            # there are some, because the geometric horizon is 0.14 degrees down at 20 m and a
            # pixel is 0.05 degrees -- never meet the water, so they keep the sky value.
            out = np.zeros(elev.shape, dtype=np.float64)
            horizon = self.sea.horizon_rad
            wet = below & (-elev >= horizon)
            if wet.any():
                out[wet] = self.sea.apparent_temperature_k(t_abs, -elev[wet])
            skim = below & ~wet
            if skim.any():
                out[skim] = self.sky.apparent_temperature_k(t_abs, np.zeros(int(skim.sum())))
        else:
            out = np.full(
                elev.shape, float(ground_temperature_k(self.sky, t_abs)), dtype=np.float64
            )
        above = ~below
        if not above.any():
            return out
        if self.cloud is not None and azimuth_rad is not None:
            azim = np.asarray(azimuth_rad, dtype=np.float64)
            coverage = self.cloud.sample(elev[above], azim[above])
            out[above] = self.sky.apparent_temperature_field(t_abs, elev[above], coverage)
        else:
            out[above] = self.sky.apparent_temperature_k(t_abs, elev[above])
        return out

    def sky_temperature(self, elevation_rad: Any) -> NDArray[np.float64]:
        """MS.2's apparent sky temperature; elevations must be above the horizon."""
        if self.sky is None:
            raise ValueError("this bridge has no sky model; pass band= or sky= at construction")
        t_abs = self.scene.t0_s + self._t_rel_s
        return np.asarray(
            self.sky.apparent_temperature_k(t_abs, np.asarray(elevation_rad, dtype=np.float64))
        )

    def __repr__(self) -> str:  # pragma: no cover - diagnostics
        return (
            f"AerialThermalBridge(t_rel={self._t_rel_s:.3f}s, tick={self.tick_s:.3f}s, "
            f"targets={sorted(self.scene.targets)}, prims={len(self.prim_to_target)}, "
            f"sky={'yes' if self.sky is not None else 'no'})"
        )
