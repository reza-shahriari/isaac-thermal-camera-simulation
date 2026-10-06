"""Author an `isaac-weather-fx` sky onto a stage, and hand the *same* cloud to the infrared band.

This is the engine-side half of the seam whose engine-free half is
:mod:`irsim.atmosphere.weather_fx`. Together they are what makes "one weather, both bands" a
structural property rather than a discipline:

* the **visible** companion frame gets its dome, its sun and its moon from weather-fx's own
  ``SkyEffect`` -- the same code its UI panel and its Python API drive, not a reimplementation;
* the **infrared** band gets the very same :class:`CloudField` object, wrapped in
  :class:`~irsim.atmosphere.weather_fx.WeatherFxDeck`, and does this project's radiometry on it.

Before this, the two bands ran two cloud models that agreed only as well as two implementations
ever do. They are now one array read twice.

**Why the effect and not a copy of it.** ``SkyEffect`` needs almost nothing from Kit -- a stage
and an up axis -- so a four-line shim context lets a headless render driver use the real thing.
Copying its authoring into a driver would have been quicker and would have started drifting the
same afternoon: the dome's exposure, the sun's airmass law and the moon's intensity are all
decisions that belong to weather-fx and should change there once.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from irsim.atmosphere.weather_fx import WeatherFxDeck, ensure_weather_fx_on_path

__all__ = [
    "CLOUD_TIERS",
    "StageOnlyContext",
    "WeatherFxSky",
    "author_weather_fx_sky",
    "VOLUME_EXTENSIONS",
    "cloud_render_path",
    "path_traced_volume_settings",
    "prepare_path_traced_volumes",
    "tier_occludes",
    "pixel_layer_settings",
    "weather_state",
]

#: AT.31 (ADR 0169): the two cloud tiers, one switch for both bands. ``path_traced`` draws the
#: cloud as weather-fx's 3-D volumes in the visible band and marches the infrared to every
#: pixel's own hit, so a target behind or inside a cloud is occluded in **both**; ``real_time``
#: keeps the visible cloud on the dome at infinity and skips the infrared march to the hit, so a
#: target is occluded in **neither**. The cloud itself is in both bands in both tiers: the
#: infrared sky pixels are marched on the same field from the same origin either way.
#:
#: WX.26: ``pixel`` is the third. The visible cloud is weather-fx's cloudscape (simulated patches
#: placed by a weather map) marched per camera pixel and drawn on a quad that rides with the
#: camera (ADR 0186), with the cloud in front of scene surfaces drawn over them; the infrared
#: marches the **same cloudscape** to every pixel's hit. A target is occluded in both bands.
CLOUD_TIERS = ("path_traced", "real_time", "pixel")


#: The extensions ADR 0144's probe enabled when a VDB cloud first rendered on this build. Without
#: them the density file reaches the material as a 1-D texture ("assigned an incompatible
#: texture ... (Type: '1D')", measured on the first AT.31 render) and the volume draws nothing.
VOLUME_EXTENSIONS = ("omni.volume", "omni.hydra.index", "omni.index")


def path_traced_volume_settings(
    volume_bounces: int | None = None,
) -> tuple[dict[str, Any], dict[str, int]]:
    """The render settings a path-traced VDB cloud needs, as ``(exact, floors)``.

    Read from weather-fx's own tables in ``backends/viewport/clouds_volume`` rather than copied:
    ``VOLUME_SETTINGS`` is set as given, ``VOLUME_FLOORS`` is a set of lower bounds, and every
    path in ``VOLUME_BOUNCE_SETTINGS`` gets ``volume_bounces`` -- weather-fx's
    ``clouds.volume_bounces`` (32) unless given. The renderer's own ``ptvol/maxBounces`` is 2,
    and a thick cloud whose paths end after two scatters renders grey: §7.5's fourth criterion
    for the visible companion's cloud (docs/physics-model.md).

    Pure, and importable without Kit, so that a submodule bump renaming a table fails the unit
    gate instead of the engine at render time. The bump to ``11bf843`` (WX.2) did rename one:
    ``MIN_PATH_BOUNCES`` became ``VOLUME_FLOORS``.
    """
    ensure_weather_fx_on_path()
    from weather_fx.backends.viewport.clouds_volume import (
        VOLUME_BOUNCE_SETTINGS,
        VOLUME_FLOORS,
        VOLUME_SETTINGS,
    )
    from weather_fx.core.state import WeatherState

    if volume_bounces is None:
        volume_bounces = int(WeatherState().clouds.volume_bounces)
    floors = dict(VOLUME_FLOORS)
    for path in VOLUME_BOUNCE_SETTINGS:
        floors[path] = max(int(floors.get(path, 0)), int(volume_bounces))
    return dict(VOLUME_SETTINGS), floors


def prepare_path_traced_volumes(volume_bounces: int | None = None) -> dict[str, str]:
    """Enable what a path-traced VDB cloud needs, and set its render settings unconditionally.

    weather-fx's ``VolumeSettingsLease`` sets its settings only where a setting already exists
    -- right inside the Kit app it was written for, where the renderer's defaults are
    registered, and a silent no-op in a headless app where ``/rtx/pathtracing/ptvol/*`` has not
    been read yet. ADR 0144's probe set them outright; so does this, from
    :func:`path_traced_volume_settings`: the exact values as given, the floors raised to and
    never lowered. Returns what each extension did, for the driver's log.
    """
    import carb.settings
    import omni.kit.app

    exact, floors = path_traced_volume_settings(volume_bounces)
    manager = omni.kit.app.get_app().get_extension_manager()
    report: dict[str, str] = {}
    for name in VOLUME_EXTENSIONS:
        if manager.is_extension_enabled(name):
            report[name] = "already on"
            continue
        try:
            manager.set_extension_enabled_immediate(name, True)
            report[name] = "enabled"
        except Exception as exc:  # noqa: BLE001 - a build without it should say so, not stop
            report[name] = f"enable failed: {type(exc).__name__}: {exc}"
    settings = carb.settings.get_settings()
    for path, value in exact.items():
        settings.set(path, value)
    for path, floor in floors.items():
        if int(settings.get(path) or 0) < int(floor):
            settings.set(path, int(floor))
    return report


def _check_tier(tier: str) -> str:
    if tier not in CLOUD_TIERS:
        raise ValueError(f"cloud tier must be one of {CLOUD_TIERS}, got {tier!r}")
    return tier


def cloud_render_path(tier: str) -> str:
    """weather-fx's ``clouds.render_path`` for a tier: ``"volume"``, ``"dome"`` or ``"pixel"``.

    Explicit rather than weather-fx's ``"auto"``, which reads the renderer's mode: a headless
    driver captures its colour frame under the path tracer whichever tier it asked for (it is the
    only mode that lights a colour AOV on this build), and ``auto`` would then pick volumes for a
    real-time tier whose infrared band is not occluding -- the disagreement ADR 0169 removes.
    """
    return {"path_traced": "volume", "real_time": "dome", "pixel": "pixel"}[_check_tier(tier)]


def tier_occludes(tier: str) -> bool:
    """Whether the infrared band marches the cloud to each geometry pixel's hit in this tier."""
    return _check_tier(tier) in ("path_traced", "pixel")


@dataclass
class StageOnlyContext:
    """The smallest thing ``SkyEffect`` will accept in place of a ``WeatherContext``.

    The effect asks for a stage, an up axis and the live state. A render driver has all three and
    none of the rest of a Kit application, which is the whole reason this is four lines.
    """

    _stage: Any
    _up_axis: int
    _state: Any
    #: Where the camera is, in stage units: what the dome is baked around and what the volumes
    #: follow (AT.30). The infrared march is started from the same point, so the two bands see
    #: the field from one place.
    anchor_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    #: The wind's accumulated drift of the cloud field, metres in stage axes. The extension's
    #: manager advances it on its clock; a headless render driver has no such clock and holds
    #: it at zero, so the dome, the volumes and the infrared march all read an undrifted field.
    cloud_drift_m: Any = None
    #: Published by the sky effect after a bake (the fog effect reads it); None until then.
    sky_horizon_rgb: Any = None
    #: Published by the sky effect after a bake too: the dome's exposure and white-balance gains,
    #: which the per-pixel cloud layer (`clouds.render_path = "pixel"`, ADR 0186) reads.
    sky_dome: Any = None
    #: The manager's clock, seconds, which the precipitation effect winds its gusts from. A
    #: headless driver authors one sky per run and has no such clock; zero.
    time: float = 0.0

    def __post_init__(self) -> None:
        if self.cloud_drift_m is None:
            self.cloud_drift_m = np.zeros(3)

    def stage(self) -> Any:
        return self._stage

    def up_axis(self) -> int:
        return self._up_axis

    def anchor(self) -> Any:
        return np.asarray(self.anchor_m, dtype=np.float64)

    @property
    def state(self) -> Any:
        return self._state

    def meters_per_unit(self) -> float:
        if self._stage is None:
            return 1.0
        from pxr import UsdGeom

        return float(UsdGeom.GetStageMetersPerUnit(self._stage))


@dataclass
class WeatherFxSky:
    """What a driver gets back: the resolved conditions, the deck for the infrared band, and a
    line it can print."""

    state: Any
    conditions: Any
    deck: WeatherFxDeck | None
    effect: Any
    #: AT.31: which tier this sky was authored for, and weather-fx's volume effect when it is the
    #: path-traced one and there is cloud to draw (``None`` otherwise).
    tier: str = "path_traced"
    volumes: Any = None
    #: WX.26: weather-fx's per-pixel cloud layer when the tier is ``pixel`` and there is cloud,
    #: and the app-update subscription that redraws it for the camera every frame.
    layer: Any = None
    layer_subscription: Any = None

    def visible_cloud_transmittance(self) -> Any:
        """The visible transmittance of the cloud the per-pixel layer last drew, per pixel of
        the camera it follows (WX.26) -- None in the other tiers or before the first draw. The
        infrared camera stores it beside its own march's transmittance so the bands' agreement
        is measured, not assumed (:mod:`irsim.validation.cloud_bands`)."""
        if self.layer is None:
            return None
        return self.layer.transmittance()

    @property
    def occludes(self) -> bool:
        """Whether the infrared band should march the cloud to each hit: the tier's answer."""
        return tier_occludes(self.tier)

    @property
    def companion_only_paths(self) -> tuple[str, ...]:
        """Prims the visible companion renders and the infrared G-buffer must not see (AT.31).

        weather-fx's volumes are mesh boxes marked as volumes; to the G-buffer they are geometry,
        so a drone inside the layer would read as the box around it. The infrared band has the
        same cloud as its march, so the camera hides these for the G-buffer render and shows
        them for the companion's own (``IrCamera(companion_only_prim_paths=...)``).

        The per-pixel layer's two quads (the cloud at the far end of the frustum and the veil in
        front of the camera, WX.26) are the same kind of thing: a picture of the cloud for the
        visible band, and geometry the G-buffer must not hit.
        """
        paths: list[str] = []
        if self.volumes is not None:
            from weather_fx.backends.viewport.clouds_volume import CLOUDS_ROOT

            paths.append(str(CLOUDS_ROOT))
        if self.layer is not None:
            from weather_fx.backends.viewport.clouds_pixel import LAYER_ROOT

            paths.append(str(LAYER_ROOT))
        return tuple(paths)

    def describe(self) -> str:
        return str(self.conditions.describe())

    def stats(self) -> dict[str, Any]:
        out = dict(self.effect.stats())
        out["cloud_tier"] = self.tier
        if self.volumes is not None:
            out["cloud_volumes"] = dict(self.volumes.stats())
        if self.layer is not None:
            out["cloud_layer"] = dict(self.layer.stats())
        return out


def weather_state(
    *,
    seed: int | None = None,
    regime: str | None = None,
    preset_json: str | None = None,
    overrides: dict[str, Any] | None = None,
    latitude_deg: float | None = None,
    longitude_deg: float | None = None,
    date_utc: str | None = None,
    hour_utc: float | None = None,
) -> Any:
    """Build a weather-fx ``WeatherState`` for a render.

    Four ways in, in increasing order of specificity, and they compose: a random draw, a named
    regime, a saved JSON preset, and explicit overrides. The site and the clock are applied last
    so a driver can film the *same* weather at a different hour or a different latitude, which is
    the comparison a sensor study usually wants.
    """
    ensure_weather_fx_on_path()
    from weather_fx.core.random_weather import random_state
    from weather_fx.core.state import WeatherState

    if seed is not None or regime is not None:
        state = random_state(seed, regime=regime)
    else:
        state = WeatherState()
    if preset_json:
        import json
        import pathlib

        state = WeatherState.from_dict(
            json.loads(pathlib.Path(preset_json).read_text(encoding="utf-8")), base=state
        )
    for section, values in (overrides or {}).items():
        state = state.with_updates(section, **values)

    clock: dict[str, Any] = {}
    if latitude_deg is not None:
        clock["latitude_deg"] = float(latitude_deg)
    if longitude_deg is not None:
        clock["longitude_deg"] = float(longitude_deg)
    if date_utc is not None:
        clock["date_utc"] = str(date_utc)
    if hour_utc is not None:
        clock["hour_utc"] = float(hour_utc)
    if clock:
        state = state.with_updates("sky", **clock)
    return state


#: WX.26: how the per-pixel layer is drawn for a headless infrared capture, as weather-fx
#: ``clouds`` settings (ADR 0191). Full resolution, so its transmittance lands on the G-buffer's
#: grid; one frame, because each capture is its own exposure and the layer's sixteen-frame
#: history (built for an interactive viewport) ghosted earlier poses into every frame; and the
#: march converged: a clear-air step that grows 0.1 % with range instead of 1.2 % (so a 10 m
#: wisp at 5 km, or a cirrus streak at 50 km, is integrated rather than sampled once or
#: skipped), a cap of 8192 samples, and a thin-cloud step of 0.02 optical depth instead of 0.8.
#: Measured engine-free against the infrared march of the same cloudscape on one shell: at the
#: viewport settings the two were 0.14-0.26 apart in band emissivity at the 95th percentile;
#: at these, 0.005-0.03 for cumulus and cirrus at 10 and 30 degrees. 60 ms at 640 x 512.
PIXEL_LAYER_SETTINGS: dict[str, float | int] = {
    "layer_scale": 1.0,
    "layer_accumulate": 1,
    "layer_step_m": 8.0,
    "layer_step_growth": 0.001,
    "layer_max_steps": 8192,
    "layer_thin_tau": 0.02,
}

#: The GPU deck's step cap and in-layer path cap in the pixel tier (ADR 0191). Against the
#: function's dense integral the shipped 512 steps are 0.05 at the 95th percentile in band
#: emissivity and 2048 are 0.005; the path cap of 12 km stopped the infrared short of what the
#: visible march (80 km of range) saw on rays shallower than asin(thickness / 12 km), 14.5
#: degrees for a 3 km cumulus layer. 40 km at 4096 steps keeps the 10 m sample and moves that
#: angle to 4.3 degrees, for about 0.4 s a frame. The CPU reference keeps 512 and 12 km: four
#: times its steps is four times twenty minutes.
PIXEL_TIER_GPU_STEPS = 4096
PIXEL_TIER_GPU_MAX_PATH_M = 40_000.0


def pixel_layer_settings() -> dict[str, float | int]:
    """The ``clouds`` settings a headless pixel-tier capture applies
    (:data:`PIXEL_LAYER_SETTINGS`)."""
    return dict(PIXEL_LAYER_SETTINGS)


def _make_deck(cloud: Any, gpu_march: bool | None) -> WeatherFxDeck:
    """The CPU deck, or its GPU twin for a cloudscape when asked for or available."""
    is_cloudscape = hasattr(cloud, "kernel_constants")
    if gpu_march is False or not is_cloudscape:
        return WeatherFxDeck(cloud)
    from irsim_isaac.cloud_march_gpu import GpuWeatherFxDeck, gpu_march_available

    if gpu_march is None and not gpu_march_available():
        return WeatherFxDeck(cloud)
    return GpuWeatherFxDeck(
        cloud, max_steps=PIXEL_TIER_GPU_STEPS, max_path_m=PIXEL_TIER_GPU_MAX_PATH_M
    )


def author_weather_fx_sky(
    stage: Any,
    state: Any,
    *,
    texture_dir: Any = None,
    up_axis: int | None = None,
    od_ratio: float | None = None,
    anchor_m: tuple[float, float, float] = (0.0, 0.0, 0.0),
    tier: str = "path_traced",
    camera_path: str | None = None,
    gpu_march: bool | None = None,
) -> WeatherFxSky:
    """Put weather-fx's sky on ``stage`` and return the cloud the infrared band should march.

    Everything authored lives under ``/WeatherFX`` in the **session layer**, so the caller's own
    stage is untouched and the effect's ``detach`` removes it cleanly.

    ``anchor_m`` is the camera's position in metres (AT.30): the dome is baked around it,
    and the returned deck's ``origin_m`` is the same point in the field's frame, so the infrared
    march and the visible dome read the cloud from one place.

    The returned ``deck`` is ``None`` for a cloudless state -- a clear sky needs no march, and
    passing a deck that covers nothing would cost a march per pixel to learn that.

    ``tier`` (AT.31, ADR 0169) is the one switch for both bands. ``path_traced`` sets the state's
    ``clouds.render_path`` to ``"volume"`` and attaches weather-fx's ``CloudVolumeEffect`` beside
    the sky; ``real_time`` sets it to ``"dome"`` and attaches no volumes. Either way the state's
    clock is set to ``"manual"``, so the dome and the volumes are built on the calling thread and
    are in the stage before the first frame -- a headless driver has no loop to collect them
    later. The caller reads :attr:`WeatherFxSky.occludes` for the infrared half, so the two
    cannot be chosen apart.

    ``pixel`` (WX.26) sets the render path to ``"pixel"`` and attaches weather-fx's
    ``CloudLayerEffect``, redrawn on every app update for the camera at ``camera_path`` (the
    prim may be created after this call). The deck it returns wraps the **cloudscape** that
    layer marches, not a ``CloudField``: one function of position read by both bands. The
    layer is drawn at the camera's full render resolution (``layer_scale`` 1), so its
    transmittance lands on the G-buffer's own grid.

    ``gpu_march`` (WX.26 follow-up): march the infrared on the GPU
    (:class:`irsim_isaac.cloud_march_gpu.GpuWeatherFxDeck`, the CPU deck's twin) when the cloud is
    a cloudscape; ``None`` does so whenever Warp and a CUDA device are present, ``False`` keeps
    the CPU reference. The visible ``density_scale`` reaches the deck either way.
    """
    ensure_weather_fx_on_path()
    state = state.with_updates("clouds", render_path=cloud_render_path(tier))
    # Both tiers bake synchronously. Under the default `wall` clock the sky installs a quick
    # cloudless first bake and finishes the real one in a worker that only the extension's own
    # `update` loop collects -- which a headless driver never runs, so the real-time tier's dome
    # stayed clear for the whole render while the infrared marched an overcast (measured, AT.31).
    state = state.with_updates("general", time_source="manual")
    from weather_fx.backends.viewport.sky import SkyEffect
    from weather_fx.core.sky import conditions_from_state

    if up_axis is None:
        from pxr import UsdGeom

        up_axis = 2 if UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.z else 1

    if tier == "pixel" and camera_path:
        state = state.with_updates("general", follow_prim=str(camera_path))
        state = state.with_updates("clouds", **pixel_layer_settings())
    # The pixel tier never reads the voxel field: its cloud is the cloudscape, built below.
    conditions = conditions_from_state(state, build_cloud=tier != "pixel")
    effect = SkyEffect(texture_dir=str(texture_dir) if texture_dir else None)
    # ``anchor_m`` arrives in metres; the effect asks the context for stage units.
    mpu = StageOnlyContext(stage, int(up_axis), state).meters_per_unit()
    anchor_units = tuple(float(v) / mpu for v in anchor_m)
    context = StageOnlyContext(stage, int(up_axis), state, anchor_m=anchor_units)  # type: ignore[arg-type]
    effect.attach(context)
    # `changed` names every section, because nothing has been applied to this stage yet and the
    # effect's own short-circuit would otherwise decide there was nothing to do.
    effect.apply_state(state, {"general", "sky", "clouds"})

    cloud = conditions.cloud
    if tier == "pixel":
        from weather_fx.core.cloudscape import cloudscape_from_state

        cloud = cloudscape_from_state(state)
    deck = None
    if cloud is not None:
        from weather_fx.core.clouds import stage_to_field

        deck = _make_deck(cloud, gpu_march)
        if od_ratio is not None:
            deck.od_ratio = float(od_ratio)
        # The visible layer applies `clouds.density_scale` to its own copy of the layer; the
        # infrared reads the field itself, so the scale is carried to it here or the two bands
        # march clouds of different depth (docs/weather/README.md §7).
        deck.density_scale = float(getattr(state.clouds, "density_scale", 1.0))
        # The dome's own origin (`SkyEffect`): the anchor in metres, less the drift, turned
        # into the field's Y-up frame. One expression on both sides, so they cannot part.
        origin = stage_to_field(
            np.asarray(anchor_m, dtype=np.float64)
            - np.asarray(context.cloud_drift_m, dtype=np.float64),
            int(up_axis),
        )
        deck.origin_m = (float(origin[0]), float(origin[1]), float(origin[2]))

    volumes = None
    if tier == "path_traced" and conditions.cloud is not None:
        from weather_fx.backends.viewport.clouds_volume import CloudVolumeEffect

        prepare_path_traced_volumes(int(state.clouds.volume_bounces))

        # The same context as the sky, so the volumes follow the same anchor and drift the dome
        # is baked around and the infrared deck marches from (AT.30).
        volumes = CloudVolumeEffect(directory=str(texture_dir) if texture_dir else None)
        volumes.attach(context)
        volumes.apply_state(state, {"general", "clouds"})
        volumes.update(0.0, 0.0)  # place the tiles around the anchor
    layer = None
    subscription = None
    if tier == "pixel" and cloud is not None:
        import omni.kit.app
        from weather_fx.backends.viewport.clouds_pixel import CloudLayerEffect

        # The same context as the sky: the layer reads the dome's exposure and gains from it and
        # marches from the camera's own position less the same drift the deck's origin carries.
        layer = CloudLayerEffect()
        layer.attach(context)
        layer.apply_state(state, {"general", "sky", "clouds"})

        def _redraw(_event: Any, layer: Any = layer) -> None:
            layer.update(1.0 / 60.0, 0.0)

        subscription = (
            omni.kit.app.get_app()
            .get_update_event_stream()
            .create_subscription_to_pop(_redraw, name="irsim weather-fx cloud layer")
        )
    return WeatherFxSky(
        state=state,
        conditions=conditions,
        deck=deck,
        effect=effect,
        tier=tier,
        volumes=volumes,
        layer=layer,
        layer_subscription=subscription,
    )
