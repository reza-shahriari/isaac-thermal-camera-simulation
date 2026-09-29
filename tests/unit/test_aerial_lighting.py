"""AT.23 -- the engine-free aerial builder lights and orients its surfaces.

`validation/aerial_scene.py` gave every surface V_s = 1 and n·v = 1 and wrote no `l_sun`,
`shadow_mask` or `sun_cos_incidence`, so a CPU render saw only sky reflection and no sun in any
band. Every surface now carries its orientation (a belly reflects the ground, not the sky) and,
with a solar bundle and a sky that knows its site, the sun. Without them the emissive frame is
what it was. Roadmap AT.23; ADR 0071, ADR 0153.
"""

from __future__ import annotations

import math
import pathlib

import numpy as np
import pytest

from irsim.atmosphere.layered import LayeredAtmosphere
from irsim.atmosphere.library import load_atmosphere_preset
from irsim.atmosphere.sky import SkyModel
from irsim.atmosphere.skylight import DiffuseSkylight
from irsim.config.environment import load_environment_preset
from irsim.config.loader import load_sensor_config
from irsim.pipeline import PipelineConfig, PipelineState, run_frame
from irsim.pipeline.solar import SolarIllumination
from irsim.radiometry.lut import BandLUT
from irsim.radiometry.lut_files import load_band_response_for_config
from irsim.radiometry.solar import load_solar_spectrum
from irsim.scene import ground_reflectance
from irsim.thermal.weather import WeatherSample, WeatherSeries
from irsim.validation.aerial_scene import SceneTarget, build_aerial_gbuffer

REPO = pathlib.Path(__file__).resolve().parents[2]
DATA = REPO / "data"
GROUND_SPECTRUM = DATA / "spectra" / "solar" / "direct_normal_am1p5.csv"
SITE = (45.0, 0.0)
NOON = 12.0 * 3600.0  # the constant weather's epoch is 2000-01-01T00:00Z; noon at 0 deg E


def _sunlit_sky(name: str, band: str, weather: WeatherSeries) -> tuple[SkyModel, BandLUT, object]:
    cfg = load_sensor_config(name)
    response = load_band_response_for_config(cfg)
    lut = BandLUT.build(response, n=1001)
    layered = LayeredAtmosphere(
        load_atmosphere_preset("us_standard_clear"), weather, {band: lut}, {band: response}
    )
    env = load_environment_preset("clear_dry")
    skylight = DiffuseSkylight.from_spectrum(
        load_solar_spectrum(GROUND_SPECTRUM), response, cfg.sensor.quantity, band
    )
    sky = SkyModel(
        layered,
        env,
        band,
        lut,
        cfg.sensor.quantity,
        skylight=skylight,
        site=SITE,
        ground_albedo=ground_reflectance(env, band),
    )
    return sky, lut, cfg


def _weather() -> WeatherSeries:
    return WeatherSeries.constant(
        WeatherSample(288.15, 0.4, 1.0, 0.0, 800.0, 100.0, 23000.0, 0.0), 24 * 3600.0
    )


# --- orientation -------------------------------------------------------------------------------


def test_the_ground_is_seen_along_its_ray_and_a_belly_faces_the_ground(
    aerial_sensor,
    aerial_sky,
    aerial_materials,  # type: ignore[no-untyped-def]
) -> None:
    deck = SceneTarget("painted_composite", 300.0, 2.0, 60.0, (20.0, 30.0), tilt_deg=0.0)
    belly = SceneTarget("painted_composite", 300.0, 2.0, 60.0, (44.0, 30.0), tilt_deg=180.0)
    scene = build_aerial_gbuffer(
        aerial_sensor.sensor,
        aerial_sky,
        aerial_materials,
        boresight_elevation_deg=10.0,
        targets=[deck, belly],
    )
    p = scene.planes
    ground = ~scene.sky_mask
    ndv = p["normal_dot_view"]
    assert ndv.dtype == np.float32 and np.all(ndv[scene.sky_mask] == 1.0)
    # the ground faces up and is seen at the ray's own elevation below the horizon
    (_, (i0, j0, i1, j1)), (_, (bi0, bj0, bi1, bj1)) = scene.resolved
    on_targets = np.zeros(ground.shape, dtype=bool)
    on_targets[i0:i1, j0:j1] = True
    on_targets[bi0:bi1, bj0:bj1] = True
    plain_ground = ground & ~on_targets
    expect = np.abs(np.sin(scene.elevation_rad))[plain_ground]
    assert np.allclose(ndv[plain_ground], expect, atol=1e-6)
    assert ndv[plain_ground].max() < 0.5, "a low camera sees the ground at grazing angles"
    # the deck looks up and sees the whole sky; the belly looks down and sees none of it
    assert np.all(p["sky_view_factor"][i0:i1, j0:j1] == 1.0)
    assert np.all(p["sky_view_factor"][bi0:bi1, bj0:bj1] == 0.0)
    # without a solar bundle there is no sun in the frame
    assert "l_sun" not in p and "shadow_mask" not in p and "sun_cos_incidence" not in p


def test_a_belly_reflects_the_ground_and_reads_warmer_than_the_same_deck(
    aerial_sensor,
    aerial_sky,
    aerial_materials,
    tophat_lwir_lut,  # type: ignore[no-untyped-def]
) -> None:
    """Under a cold clear sky the deck of a painted target reflects the sky and the belly the
    ground: on paint (ε ≈ 0.9) that is a few kelvin of apparent temperature in LWIR."""
    deck = SceneTarget("painted_composite", 290.0, 2.0, 60.0, (20.0, 30.0), tilt_deg=0.0)
    belly = SceneTarget("painted_composite", 290.0, 2.0, 60.0, (44.0, 30.0), tilt_deg=180.0)
    scene = build_aerial_gbuffer(
        aerial_sensor.sensor,
        aerial_sky,
        aerial_materials,
        boresight_elevation_deg=10.0,
        targets=[deck, belly],
    )
    cfg = PipelineConfig.from_sensor(
        aerial_sensor,
        aerial_materials,
        lut=tophat_lwir_lut,
        noise_enabled=False,
        psf_enabled=False,
        atmosphere=aerial_sky.atmosphere,
        sky=aerial_sky,
        sensor_seed=3,
    )
    out = run_frame(scene.planes, cfg, PipelineState(t_s=0.0))
    t_app = np.asarray(out.apparent_t, dtype=np.float64)
    (_, (i0, j0, i1, j1)), (_, (bi0, bj0, bi1, bj1)) = scene.resolved
    t_deck, t_belly = (
        float(np.median(t_app[i0:i1, j0:j1])),
        float(np.median(t_app[bi0:bi1, bj0:bj1])),
    )
    assert 1.0 < t_belly - t_deck < 8.0, (t_deck, t_belly)


# --- the sun -----------------------------------------------------------------------------------


@pytest.mark.parametrize("name,band", [("swir640", "swir"), ("boson640", "lwir")])
def test_with_a_solar_bundle_a_noon_deck_is_sunlit_and_a_belly_is_not(name: str, band: str) -> None:
    sky, lut, cfg = _sunlit_sky(name, band, _weather())
    from irsim.materials import MaterialTable
    from irsim.materials.library import MaterialLibrary

    materials = MaterialTable.from_library(MaterialLibrary.load(), band)
    solar = SolarIllumination.for_sensor(cfg, cfg.sensor.quantity, DATA)
    deck = SceneTarget("painted_composite", 300.0, 2.0, 60.0, (20.0, 30.0), tilt_deg=0.0)
    # 2 m at 60 m is ~40 px on either camera: keep the two blocks apart or the second one drawn
    # overwrites part of the first
    belly = SceneTarget("painted_composite", 300.0, 2.0, 60.0, (120.0, 30.0), tilt_deg=180.0)
    scene = build_aerial_gbuffer(
        cfg.sensor,
        sky,
        materials,
        t_s=NOON,
        boresight_elevation_deg=10.0,
        targets=[deck, belly],
        supersample=cfg.sensor.optics.supersample_factor,
        illumination=solar,
    )
    p = scene.planes
    assert {"l_sun", "shadow_mask", "sun_cos_incidence"} <= set(p)
    (_, (i0, j0, i1, j1)), (_, (bi0, bj0, bi1, bj1)) = scene.resolved
    assert np.all(p["shadow_mask"] == 1.0)
    assert np.all(p["sun_cos_incidence"][scene.sky_mask] == 0.0)
    # the weather's epoch is 1 January: noon at 45 N puts the sun 22° up, so a horizontal deck
    # sees it at cos = sin(22°) = 0.37 -- the builder's number is the sun's, not a guess
    expected = math.sin(math.radians(sky.solar_terms(NOON).elevation_deg))
    assert 0.3 < expected < 0.4, expected
    block = p["sun_cos_incidence"][i0:i1, j0:j1]
    assert block.min() == pytest.approx(expected, abs=1e-6) and block.max() == pytest.approx(
        expected, abs=1e-6
    )
    assert np.all(p["sun_cos_incidence"][bi0:bi1, bj0:bj1] == 0.0), "the belly never sees the sun"
    assert p["l_sun"][i0:i1, j0:j1].min() > 0.0 and np.all(p["l_sun"][bi0:bi1, bj0:bj1] == 0.0)
    if band == "swir":
        # the sunlit shell dominates its own emission in SWIR
        dark = build_aerial_gbuffer(
            cfg.sensor,
            sky,
            materials,
            t_s=NOON,
            boresight_elevation_deg=10.0,
            targets=[deck, belly],
            supersample=cfg.sensor.optics.supersample_factor,
        )
        lit_cfg = PipelineConfig.from_sensor(
            cfg,
            materials,
            lut=lut,
            noise_enabled=False,
            psf_enabled=False,
            atmosphere=sky.atmosphere,
            sky=sky,
            sensor_seed=3,
        )
        lit = np.asarray(run_frame(p, lit_cfg, PipelineState(t_s=NOON)).radiance, dtype=np.float64)
        unlit = np.asarray(
            run_frame(dark.planes, lit_cfg, PipelineState(t_s=NOON)).radiance, dtype=np.float64
        )
        # the radiance is on the native grid; the boxes are on the k× one (measured: 48× lit/dark)
        k = cfg.sensor.optics.supersample_factor
        deck_px = (slice(i0 // k, i1 // k), slice(j0 // k, j1 // k))
        assert lit[deck_px].mean() > 3.0 * unlit[deck_px].mean()


def test_a_sky_without_a_site_cannot_place_the_sun(
    aerial_sensor, aerial_sky, aerial_materials
) -> None:  # type: ignore[no-untyped-def]
    assert aerial_sky.site is None
    with pytest.raises(ValueError, match="no site"):
        aerial_sky.solar_terms(0.0)
    scene = build_aerial_gbuffer(
        aerial_sensor.sensor, aerial_sky, aerial_materials, illumination=object()
    )
    assert "l_sun" not in scene.planes, "no site, no sun -- and no silent guess"
    assert math.isclose(float(scene.planes["sky_view_factor"].max()), 1.0)
