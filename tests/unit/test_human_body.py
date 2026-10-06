"""HU.4 -- a person's seventeen skin temperatures from JOS-3, on the scene's one weather.

What would be wrong and invisible: a mean skin that is not a person's (JOS-3 misdriven), a hand
warmer than the head in the cold (segments mis-ordered or mis-indexed), a body stepped seventeen
times per tick (seventeen views each advancing it), a person forced by a weather the scene does
not have, or the sun making no difference. Each is a case here; the renderer's half of the row
(head − hand in the frame equals the solver's) is measured in the hero render and recorded in
the roadmap, not asserted here, because it needs Isaac Sim.

docs/physics-model.md §6.1, §16.2; roadmap HU.4; ADR 0122, ADR 0192, ADR 0194.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

pytest.importorskip("jos3", reason="JOS-3 (the `jos3` package) is a project dependency")

from irsim.config.humans import JOS3_SEGMENTS, HumanSpec, load_body_schema  # noqa: E402
from irsim.config.scene import TargetSpec  # noqa: E402
from irsim.thermal.human_body import (  # noqa: E402
    HEAD_LAYERS,
    HumanBodySolver,
    clo_by_segment,
    mean_radiant_temperature_k,
    projected_area_factor,
)
from irsim.thermal.weather import WeatherSample, WeatherSeries  # noqa: E402

MAN = HumanSpec(phenotype={"sex": "male", "age_y": 35, "height_m": 1.7245, "mass_kg": 68.4})


def weather(
    t_air_c: float, wind_m_s: float = 1.0, dni: float = 0.0, rh: float = 0.5
) -> WeatherSeries:
    sample = WeatherSample(
        t_air_k=273.15 + t_air_c,
        rh_fraction=rh,
        wind_speed_m_s=wind_m_s,
        cloud_fraction=0.0,
        dni_w_m2=dni,
        dhi_w_m2=0.0,
        visibility_m=20000.0,
        precip_mm_h=0.0,
    )
    return WeatherSeries.constant(
        sample, 48 * 3600.0, datetime(2024, 6, 21, 0, 0, tzinfo=timezone.utc)
    )


def index(seg: str) -> int:
    return JOS3_SEGMENTS.index(seg)


# --- the physiology ----------------------------------------------------------------------------


def test_a_neutral_seated_person_has_the_textbook_mean_skin() -> None:
    """Fanger's regression at 1 met gives 33.7 °C; JOS-3 at its own neutral (28.8 °C air, still
    air, seated, bare) must land within 0.5 K of it, or the model is being driven wrongly."""
    body = HumanBodySolver(
        MAN, weather(28.8, wind_m_s=0.1), 0.0, posture="sitting", activity_met=1.0
    )
    assert body.temperature() - 273.15 == pytest.approx(33.7, abs=0.5)
    # and the segments average, by JOS-3's own areas, to its reported mean
    assert abs(body.skin_k().mean() - body.temperature()) < 1.5


@pytest.mark.parametrize("t_air_c", [0.0, 10.0, 20.0])
def test_in_the_cold_the_head_stays_warmest_and_the_feet_go_first(t_air_c: float) -> None:
    body = HumanBodySolver(MAN, weather(t_air_c), 0.0)
    skin = body.skin_k()
    head, hand, foot = skin[index("Head")], skin[index("LHand")], skin[index("LFoot")]
    assert head > hand > foot, (head, hand, foot)
    # the extremities fall far more than the head does: that gradient is the picture
    assert head - foot > 5.0


def test_colder_air_gives_colder_skin_everywhere() -> None:
    cold = HumanBodySolver(MAN, weather(0.0), 0.0).skin_k()
    mild = HumanBodySolver(MAN, weather(20.0), 0.0).skin_k()
    assert np.all(cold < mild)


def test_the_radiant_temperature_is_the_scenes_sky_and_sun() -> None:
    """VDI 3787's outdoor balance: overcast and sunless it is the air; a clear night sits below
    the air; a summer noon lifts it 15–30 K above -- not SolarCal's +55 K behind a window."""
    overcast = (
        weather(10.0, rh=0.8)
        .at(0.0)
        .__class__(**{**weather(10.0, rh=0.8).at(0.0).__dict__, "cloud_fraction": 1.0})
    )
    assert mean_radiant_temperature_k(overcast, -10.0, 0.0, 0.0) == pytest.approx(283.15, abs=0.01)
    clear_night = weather(10.0, rh=0.5).at(0.0)
    night = mean_radiant_temperature_k(clear_night, -10.0, 0.0, 0.0)
    # Brunt's clear-sky emissivity at 6 hPa is 0.68: half sky, half ground gives −12 K
    assert 283.15 - 16.0 < night < 283.15 - 3.0, night
    noon = weather(26.2, rh=0.41, dni=823.0).at(0.0)
    sunlit = mean_radiant_temperature_k(noon, 61.1, 823.0, 106.5)
    assert 15.0 < sunlit - (273.15 + 26.2) < 30.0, sunlit
    # the beam lands on less of a standing body as the sun climbs
    assert projected_area_factor(0.0) == pytest.approx(0.308, abs=1e-6)
    assert projected_area_factor(0.0) > projected_area_factor(45.0) > projected_area_factor(89.0)


def test_the_sun_warms_the_skin_and_the_solver_reads_the_sites_sun() -> None:
    """Noon, 21 June, 48 °N: the solver reads the site's sun and the weather's beam."""
    noon = 12 * 3600.0
    kw = dict(site_latitude_deg=48.1, site_longitude_deg=11.6)
    shade = HumanBodySolver(MAN, weather(15.0, dni=0.0), noon, **kw)
    sun = HumanBodySolver(MAN, weather(15.0, dni=800.0), noon, **kw)
    assert sun.last_forcing is not None and shade.last_forcing is not None
    assert sun.last_forcing.tr_c > shade.last_forcing.tr_c + 10.0
    assert sun.temperature() > shade.temperature() + 0.5
    # the same beam at midnight is nothing, by the sun's elevation and not by a flag
    midnight = HumanBodySolver(MAN, weather(15.0, dni=800.0), 0.0, **kw)
    assert midnight.last_forcing is not None
    assert midnight.last_forcing.tr_c < 15.0, "a clear night sky is colder than the air"


def test_clothing_insulates_the_segments_it_covers() -> None:
    schema = load_body_schema()
    dressed = HumanSpec(
        phenotype=MAN.phenotype,
        garments={"torso": {"material": "cotton_clothing", "clo": 1.0}},
    )
    clo = clo_by_segment(dressed, schema.garment_slots)
    assert clo[index("Chest")] == 1.0 and clo[index("Back")] == 1.0 and clo[index("Pelvis")] == 1.0
    assert clo[index("LHand")] == 0.0 and clo[index("Head")] == 0.0
    bare = HumanBodySolver(MAN, weather(0.0), 0.0).skin_k()
    coat = HumanBodySolver(dressed, weather(0.0), 0.0, clo=clo).skin_k()
    assert coat[index("Chest")] > bare[index("Chest")] + 3.0
    with pytest.raises(ValueError, match="one non-negative value per segment"):
        HumanBodySolver(MAN, weather(0.0), 0.0, clo=np.full(16, 0.5))


# --- the solver contract -----------------------------------------------------------------------


def test_seventeen_views_step_the_body_once_per_tick() -> None:
    body = HumanBodySolver(MAN, weather(10.0), 0.0)
    views = body.derived_targets("man")
    assert set(views) == {f"man.skin_{s}" for s in JOS3_SEGMENTS} | {
        f"man.{layer}" for layer in HEAD_LAYERS
    }
    before = body.steps
    body.advance(0.0, 30.0)
    temps = {name: v.advance(0.0, 30.0) for name, v in views.items()}
    assert body.steps == before + 1, "the body took one step for the tick, not twenty-two"
    assert temps["man.skin_LHand"] == pytest.approx(body.segment_temperature_k("LHand"))
    assert temps["man.eyes"] == temps["man.skin_Head"], "the eyes render at the head's skin"
    assert body.state.t_s == 30.0 and views["man.skin_Head"].state.t_s == 30.0
    body.advance(30.0, 30.0)
    assert body.steps == before + 2


def test_a_mean_skin_target_is_a_temperature_solver() -> None:
    from irsim.thermal.solvers import TemperatureSolver

    body = HumanBodySolver(MAN, weather(10.0), 0.0)
    assert isinstance(body, TemperatureSolver)
    assert isinstance(body.derived_targets("p")["p.skin_Head"], TemperatureSolver)
    assert 273.15 + 15.0 < body.temperature() < 273.15 + 36.0


# --- the config --------------------------------------------------------------------------------


def test_a_human_target_names_its_asset_and_authors_nothing_else() -> None:
    TargetSpec(name="man", solver="human", asset="man_adult")
    TargetSpec(name="man", solver="human", asset="man_adult", posture="sitting", activity_met=1.0)
    with pytest.raises(ValueError, match="names its `kind: human` asset"):
        TargetSpec(name="man", solver="human")
    with pytest.raises(ValueError, match="takes no schedule_s"):
        TargetSpec(
            name="man", solver="human", asset="man_adult", schedule_s=[0.0], schedule_k=[300.0]
        )
    with pytest.raises(ValueError, match="only a human takes an asset"):
        TargetSpec(name="x", solver="airframe", asset="man_adult")


def test_the_scene_binds_a_human_to_its_one_weather(tmp_path) -> None:
    """One `WeatherSeries`, injected into physiology and atmosphere alike (CLAUDE.md #6)."""
    import yaml

    from irsim.scene import Scene

    with open("configs/scenes/man_standing_clear.yaml", encoding="utf-8") as fh:
        doc = yaml.safe_load(fh)
    scene = Scene.from_config(_spec(doc))
    assert scene.humans == ("man",)
    body = scene.targets["man"]
    assert body.weather is scene.weather, "the person is forced by the scene's own series"
    assert "target:man" in scene.consumers
    assert {f"man.skin_{s}" for s in JOS3_SEGMENTS} <= set(scene.targets)
    temps = scene.advance_targets(0.0, 30.0)
    skin = np.array([temps[f"man.skin_{s}"] for s in JOS3_SEGMENTS]) - 273.15
    assert np.all((skin > 28.0) & (skin < 37.5)), skin  # a living person at a summer noon
    assert skin.max() - skin.min() > 0.5, "seventeen segments, not one temperature"
    assert temps["man.eyes"] == temps["man.skin_Head"]


def _spec(doc):  # type: ignore[no-untyped-def]
    from irsim.config.scene import SceneConfig

    return SceneConfig.model_validate(doc)
