"""HU.5 -- clothing on the body: six fabrics, a colour that moves only what colour moves, and a
garment surface solved on the skin beneath it.

What would be wrong and invisible: a dye that changed the long-wave emissivity (it does not,
Zhang 2009), a black shirt no warmer than a white one in the sun, a coat solved on a textbook
set point instead of the cold skin JOS-3 gives beneath it, a garment surface that ignored the
wind. Each is a case here.

docs/physics-model.md §4.2, §6.1, §16.2; roadmap HU.5; ADR 0122, ADR 0194, ADR 0195.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

pytest.importorskip("jos3")

from irsim.config.humans import JOS3_SEGMENTS, HumanSpec, load_body_schema  # noqa: E402
from irsim.materials.colour import (  # noqa: E402
    coloured_material,
    dress_asset_materials,
    dyed_reflectance,
    luminance,
    variant_name,
)
from irsim.materials.library import MaterialLibrary  # noqa: E402
from irsim.thermal.human import CLO_M2K_W, clothing_temperature_c  # noqa: E402
from irsim.thermal.human_body import HumanBodySolver  # noqa: E402
from irsim.thermal.weather import WeatherSample, WeatherSeries  # noqa: E402

WHITE, BLACK, BLUE, RED = (0.9, 0.9, 0.9), (0.03, 0.03, 0.03), (0.1, 0.2, 0.7), (0.7, 0.1, 0.1)
NEW_FABRICS = ("hair", "polyester_clothing", "denim", "wool_clothing", "leather", "shoe_rubber")


@pytest.fixture(scope="module")
def library() -> MaterialLibrary:
    return MaterialLibrary.load()


def weather(t_air_c: float, wind: float = 1.0, dni: float = 0.0, dhi: float = 0.0, rh: float = 0.5):
    sample = WeatherSample(
        t_air_k=273.15 + t_air_c, rh_fraction=rh, wind_speed_m_s=wind, cloud_fraction=0.0,
        dni_w_m2=dni, dhi_w_m2=dhi, visibility_m=20000.0, precip_mm_h=0.0,
    )  # fmt: skip
    return WeatherSeries.constant(sample, 48 * 3600.0, datetime(2024, 6, 21, tzinfo=timezone.utc))


# --- the six fabrics ---------------------------------------------------------------------------


def test_the_six_fabrics_are_in_the_library_and_close(library: MaterialLibrary) -> None:
    """The Kirchhoff walk (`test_materials_library.py`) already closes them; here, what each is."""
    for name in NEW_FABRICS:
        m = library[name]
        eps = m.spec.optical.emissivity_per_band
        assert eps is not None and 0.7 <= eps["lwir"] <= 0.98, name
        assert "ESTIMATED" in m.spec.reference or m.spec.source != "measured", name
    # Belliveau 2020: cotton, nylon and polyester 0.88 over 8-12 um; wool below them (Zhang 2009)
    assert library["polyester_clothing"].spec.optical.emissivity_per_band["lwir"] == 0.88
    assert library["denim"].spec.optical.emissivity_per_band["lwir"] == 0.88
    assert library["wool_clothing"].spec.optical.emissivity_per_band["lwir"] < 0.88
    assert (
        library["shoe_rubber"].spec.thermal.thickness_m > library["denim"].spec.thermal.thickness_m
    )


# --- colour --------------------------------------------------------------------------------------


def test_colour_moves_the_sun_and_the_nir_and_never_the_long_wave(library: MaterialLibrary) -> None:
    base = library["cotton_clothing"]
    white, black = coloured_material(base, WHITE), coloured_material(base, BLACK)
    for band in ("lwir", "mwir", "swir"):
        assert (
            white.spec.optical.emissivity_per_band[band]
            == base.spec.optical.emissivity_per_band[band]
        )
        assert (
            black.spec.optical.emissivity_per_band[band]
            == base.spec.optical.emissivity_per_band[band]
        )
    assert black.spec.thermal.solar_absorptivity - white.spec.thermal.solar_absorptivity >= 0.3
    assert (
        black.spec.optical.emissivity_per_band["nir"]
        > white.spec.optical.emissivity_per_band["nir"]
    )
    # the NIR rule reproduces what cotton_clothing.yaml quotes: white ~0.75, black ~0.15 reflected
    assert dyed_reflectance(luminance(WHITE)) == pytest.approx(0.69, abs=0.02)
    assert dyed_reflectance(luminance(BLACK)) == pytest.approx(0.168, abs=0.02)
    assert white.name == variant_name("cotton_clothing", WHITE) and white.name != base.name
    assert white.spec.thermal.density_kg_m3 == base.spec.thermal.density_kg_m3


def test_a_blue_and_a_red_shirt_are_the_same_cloth_in_lwir(library: MaterialLibrary) -> None:
    blue, red = (coloured_material(library["cotton_clothing"], c) for c in (BLUE, RED))
    assert (
        blue.spec.optical.emissivity_per_band["lwir"]
        == red.spec.optical.emissivity_per_band["lwir"]
    )
    assert blue.spec.thermal.solar_absorptivity != red.spec.thermal.solar_absorptivity


def test_dressing_an_asset_remaps_only_the_coloured_garments(library: MaterialLibrary) -> None:
    human = HumanSpec(
        phenotype={"sex": "male", "age_y": 35, "height_m": 1.72, "mass_kg": 68},
        garments={
            "torso": {"material": "cotton_clothing", "clo": 0.09, "colour_rgb": BLUE},
            "legs": {"material": "denim", "clo": 0.24},
        },
    )
    materials = {"skin": "human_skin", "garment_torso": "cotton_clothing", "garment_legs": "denim"}
    dressed, lib2 = dress_asset_materials(materials, human, library)
    assert dressed["garment_legs"] == "denim" and dressed["skin"] == "human_skin"
    assert dressed["garment_torso"] == variant_name("cotton_clothing", BLUE)
    assert dressed["garment_torso"] in lib2.names and dressed["garment_torso"] not in library.names
    assert len(lib2) == len(library) + 1


def test_a_colour_outside_reflectance_is_refused(library: MaterialLibrary) -> None:
    with pytest.raises(ValueError, match="linear reflectance"):
        coloured_material(library["cotton_clothing"], (1.2, 0.0, 0.0))


# --- the garment surface -----------------------------------------------------------------------


def test_with_the_defaults_the_clothing_balance_is_iso_7730s() -> None:
    assert clothing_temperature_c(0.0, 0.0, 1.0 * CLO_M2K_W) == pytest.approx(13.96, abs=0.05)


def test_a_black_shirt_runs_warmer_than_a_white_one_in_the_sun() -> None:
    kw = dict(t_air_c=25.0, t_radiant_c=25.0, insulation_m2k_w=0.09 * CLO_M2K_W, air_speed_m_s=1.0)
    shade = clothing_temperature_c(**kw)
    white = clothing_temperature_c(**kw, absorbed_solar_w_m2=0.3 * 250.0)
    black = clothing_temperature_c(**kw, absorbed_solar_w_m2=0.9 * 250.0)
    assert black > white > shade
    assert black - white > 0.5


def test_the_garment_sits_on_the_skin_jos3_gives_beneath_it() -> None:
    """A cold hand under a glove: the surface follows the 10 °C hand, not a 34 °C set point."""
    warm = clothing_temperature_c(0.0, -10.0, 0.5 * CLO_M2K_W, air_speed_m_s=1.0)
    cold = clothing_temperature_c(0.0, -10.0, 0.5 * CLO_M2K_W, air_speed_m_s=1.0, t_skin_c=10.0)
    assert cold < warm - 5.0
    with pytest.raises(ValueError, match="cannot be negative"):
        clothing_temperature_c(0.0, 0.0, 0.1, absorbed_solar_w_m2=-1.0)


def test_the_body_hands_out_a_garment_view_on_the_segments_it_covers() -> None:
    schema = load_body_schema()
    dressed = HumanSpec(
        phenotype={"sex": "male", "age_y": 35, "height_m": 1.72, "mass_kg": 68},
        garments={"torso": {"material": "wool_clothing", "clo": 1.0}},
    )
    from irsim.thermal.human_body import clo_by_segment

    body = HumanBodySolver(
        dressed, weather(0.0), 0.0,
        clo=clo_by_segment(dressed, schema.garment_slots),
        garment_segments={"torso": dressed.coverage(schema, "torso")},
        garment_absorptance={"torso": 0.72},
    )  # fmt: skip
    views = body.derived_targets("man")
    coat = views["man.garment_torso"]
    body.advance(0.0, 30.0)
    t_coat = coat.advance(0.0, 30.0) - 273.15
    skin = body.skin_k() - 273.15
    chest = skin[JOS3_SEGMENTS.index("Chest")]
    # the coat is colder than the skin under it and warmer than the air it stands in
    assert 0.0 < t_coat < chest, (t_coat, chest)
    # at 1 clo in a 0 °C, 1 m/s wind under a clear sky it stays within a few kelvin of the air
    assert t_coat - 0.0 < 8.0
    assert np.isclose(coat.state.temperature_k, t_coat + 273.15)
    with pytest.raises(KeyError):
        from irsim.thermal.human_body import GarmentView

        GarmentView(body, "hands")


# --- the committed dressed man ----------------------------------------------------------------


def test_the_dressed_man_loads_and_each_garment_is_a_target() -> None:
    import yaml

    from irsim.config.scene import SceneConfig
    from irsim.scene import Scene

    asset = __import__(
        "irsim.materials.mapping", fromlist=["load_asset_mapping"]
    ).load_asset_mapping("man_adult_dressed", known_materials=MaterialLibrary.load().names)
    assert asset.human is not None and set(asset.human.garments) == {"torso", "legs", "feet"}
    for slot in asset.human.garments:
        assert asset.materials[f"garment_{slot}"] == asset.human.garments[slot].material
    with open("configs/scenes/man_dressed_winter_night.yaml", encoding="utf-8") as fh:
        scene = Scene.from_config(SceneConfig.model_validate(yaml.safe_load(fh)))
    temps = scene.advance_targets(0.0, 30.0)
    air = scene.weather_at(0.0).t_air_k
    for slot in ("torso", "legs", "feet"):
        t_g = temps[f"man.garment_{slot}"]
        assert air - 1.0 < t_g < temps["man.skin_Head"], (slot, t_g - 273.15)
    # the winter street scene: the face is the brightest thing on him. The bare hands are not --
    # after two hours at -5 C JOS-3 lets them fall to within 3 K of the air (2.7 C here), colder
    # than the shirt over a 30 C chest; that is the model's extrapolation, recorded in ADR 0195.
    assert temps["man.skin_Head"] > temps["man.garment_torso"] + 5.0
    assert temps["man.skin_LHand"] < temps["man.skin_Head"] - 15.0


def test_the_default_garments_cover_every_slot_with_a_real_material(
    library: MaterialLibrary,
) -> None:
    """`prep_human.py` writes these into a new asset's `garments:`; a typo here is a config that
    fails to load the first time someone dresses a human."""
    from irsim.config.humans import DEFAULT_GARMENT_CLO, DEFAULT_GARMENT_MATERIAL

    slots = set(load_body_schema().garment_slots)
    assert set(DEFAULT_GARMENT_CLO) == slots and set(DEFAULT_GARMENT_MATERIAL) == slots
    assert set(DEFAULT_GARMENT_MATERIAL.values()) <= set(library.names)
    # ISO 9920's garment values: a T-shirt is under a tenth of a clo, trousers about a quarter
    assert DEFAULT_GARMENT_CLO["torso"] < 0.1 < DEFAULT_GARMENT_CLO["legs"] < 0.3
