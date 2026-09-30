"""AT.31: one switch picks the cloud tier for both bands (ADR 0169, physics-model §7.5).

The path-traced tier draws weather-fx's 3-D cloud volumes in the visible band and marches the
infrared to every pixel's own hit, so a target behind a cloud is dimmed in **both**; the real-time
tier keeps the visible cloud on the dome and skips that march, so it is dimmed in **neither**.
Before this, a headless render always put the visible cloud on the dome and always occluded in
the infrared -- the two bands disagreed about every target near a cloud, and the owner reads the
pair together.

What can be checked without the engine is that the two halves come from one switch and cannot be
chosen apart: the tier sets weather-fx's own ``clouds.render_path`` explicitly (weather-fx's
``auto`` reads the renderer's mode, and a headless driver runs the path tracer in both tiers), the
sky object reports the tier's occlusion answer, and the driver hands that answer to the camera
rather than deciding it twice. The in-engine half -- a target behind a cloud dimmed in both bands
or in neither, sky pixels bit-identical -- is measured by `render_phantom4.py --cloud-tier`.
"""

from __future__ import annotations

import pathlib
from typing import Any

import pytest

from irsim.atmosphere.weather_fx import ensure_weather_fx_on_path
from irsim_isaac.weather_fx_stage import (
    CLOUD_TIERS,
    WeatherFxSky,
    cloud_render_path,
    tier_occludes,
)

REPO = pathlib.Path(__file__).resolve().parents[2]
DRIVER = REPO / "scripts" / "render_phantom4.py"


def test_the_two_tiers_and_their_two_halves() -> None:
    assert CLOUD_TIERS == ("path_traced", "real_time")
    assert (cloud_render_path("path_traced"), tier_occludes("path_traced")) == ("volume", True)
    assert (cloud_render_path("real_time"), tier_occludes("real_time")) == ("dome", False)
    for bad in ("auto", "PathTracing", ""):
        with pytest.raises(ValueError, match="cloud tier"):
            cloud_render_path(bad)
        with pytest.raises(ValueError, match="cloud tier"):
            tier_occludes(bad)


def test_the_render_path_is_explicit_so_the_renderer_s_mode_cannot_override_it() -> None:
    """weather-fx's `auto` follows the renderer: a headless driver runs the path tracer in both
    tiers (the only mode that lights a colour AOV on this build), so `auto` would put volumes in
    a real-time tier whose infrared does not occlude. The explicit path wins over every mode."""
    ensure_weather_fx_on_path()
    from weather_fx.backends.viewport.render_mode import choose_cloud_path
    from weather_fx.core.state import WeatherState

    for tier in CLOUD_TIERS:
        state = WeatherState().with_updates("clouds", render_path=cloud_render_path(tier))
        for mode in ("PathTracing", "RaytracedLighting", None):
            assert choose_cloud_path(state.clouds.render_path, mode) == cloud_render_path(tier)
    # and `auto` is what the explicit path replaces: it would disagree with the real-time tier
    assert choose_cloud_path("auto", "PathTracing") == "volume"


class _Effect:
    def stats(self) -> dict[str, Any]:
        return {"sun_elevation_deg": 40.0}


class _Volumes:
    def stats(self) -> dict[str, Any]:
        return {"active": True, "tiles": 4}


def test_the_sky_reports_its_tier_s_occlusion_and_its_volumes() -> None:
    traced = WeatherFxSky(None, None, None, _Effect(), tier="path_traced", volumes=_Volumes())
    assert traced.occludes is True
    assert traced.stats()["cloud_tier"] == "path_traced"
    assert traced.stats()["cloud_volumes"] == {"active": True, "tiles": 4}
    fast = WeatherFxSky(None, None, None, _Effect(), tier="real_time")
    assert fast.occludes is False
    assert "cloud_volumes" not in fast.stats()


def test_the_volumes_are_for_the_companion_only() -> None:
    """weather-fx's volumes are mesh boxes to the renderer. Left in the G-buffer, a drone inside
    the layer would read as the box around it (measured: the first path-traced render stopped on
    `rendered prims have no thermal node: ['/WeatherFX/Clouds/Tile_0_0', ...]`). The sky names
    their root for the camera to hide from the infrared render; with no volumes it names none."""
    ensure_weather_fx_on_path()
    from weather_fx.backends.viewport.clouds_volume import CLOUDS_ROOT

    traced = WeatherFxSky(None, None, None, _Effect(), tier="path_traced", volumes=_Volumes())
    assert traced.companion_only_paths == (CLOUDS_ROOT,)
    assert WeatherFxSky(None, None, None, _Effect(), tier="real_time").companion_only_paths == ()
    # a path-traced sky with no cloud to draw has no volumes and so nothing to hide
    clear = WeatherFxSky(None, None, None, _Effect(), tier="path_traced")
    assert clear.companion_only_paths == ()


def test_the_driver_decides_both_halves_from_one_switch() -> None:
    """The visible half goes to `author_weather_fx_sky(tier=...)`; the infrared half is read
    back off the sky it returned, not taken from the argument a second time."""
    src = DRIVER.read_text(encoding="utf-8")
    assert '"--cloud-tier"' in src
    assert "tier=args.cloud_tier" in src
    assert "cloud_occlusion=True if weather_sky is None else weather_sky.occludes" in src
    assert (
        "companion_only_prim_paths=() if weather_sky is None else weather_sky.companion_only_paths"
        in src
    )
    assert src.count("args.cloud_tier") == 1, "the tier is read once, by the sky author"


def test_the_camera_s_default_is_the_occluding_tier() -> None:
    """Every driver that does not name a tier keeps what it had: the infrared marches to the hit."""
    import inspect

    from irsim_isaac.pipeline.ir_camera import IrCamera

    assert inspect.signature(IrCamera).parameters["cloud_occlusion"].default is True
    assert inspect.signature(IrCamera).parameters["companion_only_prim_paths"].default == ()
