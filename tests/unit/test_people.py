"""HU.6 -- a woman and a child are phenotype changes, nothing else.

The claim of the lane is that "the human is human": a new person is new numbers under the same
labels, not new code. So the three dressed people's asset configs must differ only where a person
differs -- phenotype, garments, and the names those carry -- and the phenotype must reach the
physiology, not be decoration. The child is held to a published body: WHO 2007's median height
and BMI for a girl of eight.

What would be wrong and invisible: a child at an adult's stature with a child's age, a phenotype
the solver never reads (JOS-3 running its default 1.72 m, 74 kg man for everyone), a woman whose
config quietly changed the skin's material. And one thing that *is* wrong and is recorded rather
than hidden: JOS-3's basal metabolism is an adult equation, and for the girl it runs 17 % above
the child equation (Schofield, WHO/FAO/UNU 1985).

docs/physics-model.md §6.1; roadmap HU.6; ADR 0192, ADR 0194, ADR 0195.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

jos3 = pytest.importorskip("jos3")

from irsim.config.humans import JOS3_SEGMENTS  # noqa: E402
from irsim.materials.library import MaterialLibrary  # noqa: E402
from irsim.materials.mapping import AssetMapping, load_asset_mapping  # noqa: E402
from irsim.thermal.human_body import HumanBodySolver  # noqa: E402
from irsim.thermal.weather import WeatherSample, WeatherSeries  # noqa: E402

PEOPLE = ("man_adult_dressed", "woman_adult_dressed", "girl_child_dressed")

#: WHO Growth Reference 5-19 years (2007), girls, 96 months: height-for-age M = 126.5558 cm and
#: BMI-for-age M = 15.681 kg/m² (hfa-girls-perc-who2007-exp.xlsx, bmi-girls-perc-who2007-exp.xlsx).
WHO_GIRL_8Y_HEIGHT_M = 1.265558
WHO_GIRL_8Y_BMI = 15.681
#: Schofield's basal metabolic rate for girls aged 3-10 (WHO/FAO/UNU 1985, Technical Report
#: Series 724, Table 5): BMR = 22.5 W + 499 kcal/day, W in kg.
SCHOFIELD_GIRL_3_10 = (22.5, 499.0)
KCAL_PER_DAY_W = 4184.0 / 86400.0


@pytest.fixture(scope="module")
def people() -> dict[str, AssetMapping]:
    names = MaterialLibrary.load().names
    return {p: load_asset_mapping(p, known_materials=names) for p in PEOPLE}


def _identity(asset: AssetMapping) -> dict:
    """What must be the same for every person: the body, its rig, its skin, its parts."""
    assert asset.human is not None and asset.parts is not None
    return {
        "kind": asset.kind,
        "scale": asset.scale_to_metres,
        "rig": asset.human.rig,
        "forward": asset.human.forward_axis,
        "materials": {k: v for k, v in asset.materials.items() if not k.startswith("garment_")},
        "body_parts": sorted(
            p.name for p in asset.parts.parts if not p.name.startswith("garment_")
        ),
        "granularity": asset.parts.granularity,
    }


def test_three_people_differ_only_in_phenotype_and_garments(people) -> None:  # type: ignore[no-untyped-def]
    man, woman, girl = (_identity(people[p]) for p in PEOPLE)
    assert man == woman == girl
    assert {f"skin_{s}" for s in JOS3_SEGMENTS} <= set(man["body_parts"])
    phenotypes = {p: people[p].human.phenotype for p in PEOPLE}  # type: ignore[union-attr]
    assert len({(ph.sex, ph.age_y, ph.height_m) for ph in phenotypes.values()}) == 3


def test_the_girl_is_the_who_median_eight_year_old(people) -> None:  # type: ignore[no-untyped-def]
    ph = people["girl_child_dressed"].human.phenotype
    assert ph.sex == "female" and ph.age_y == pytest.approx(8.0, abs=0.05)
    # the row asks 5 %; make_human.py --stature-m solves the slider, so it lands within 0.1 %
    assert ph.height_m == pytest.approx(WHO_GIRL_8Y_HEIGHT_M, rel=0.05)
    assert ph.height_m == pytest.approx(WHO_GIRL_8Y_HEIGHT_M, rel=0.001)
    assert ph.mass_kg == pytest.approx(WHO_GIRL_8Y_BMI * ph.height_m**2, abs=0.06)


def test_the_girls_shorts_cover_her_thighs_only(people) -> None:  # type: ignore[no-untyped-def]
    from irsim.config.humans import load_body_schema

    h = people["girl_child_dressed"].human
    assert h.coverage(load_body_schema(), "legs") == {"LThigh", "RThigh"}


def _weather(t_air_c: float) -> WeatherSeries:
    s = WeatherSample(
        t_air_k=273.15 + t_air_c, rh_fraction=0.5, wind_speed_m_s=1.0, cloud_fraction=0.0,
        dni_w_m2=0.0, dhi_w_m2=0.0, visibility_m=2e4, precip_mm_h=0.0,
    )  # fmt: skip
    return WeatherSeries.constant(s, 48 * 3600.0, datetime(2024, 1, 15, tzinfo=timezone.utc))


def test_the_phenotype_reaches_jos3(people) -> None:  # type: ignore[no-untyped-def]
    """Our solver's skin equals JOS-3 driven by hand with the asset's own numbers, to 1e-9 K."""
    ph = people["girl_child_dressed"].human.phenotype
    ours = HumanBodySolver(
        people["girl_child_dressed"].human.model_copy(update={"garments": {}}), _weather(0.0), 0.0
    )
    ref = jos3.JOS3(height=ph.height_m, weight=ph.mass_kg, age=int(round(ph.age_y)), sex=ph.sex)
    ref.Ta, ref.Tr, ref.RH, ref.Va, ref.PAR, ref.posture = 0.0, 0.0, 50.0, 1.0, 1.25, "standing"
    ref.Icl = np.zeros(17)
    ref.simulate(times=120, dtime=60.0, output=False)
    assert np.allclose(ours.skin_k() - 273.15, np.asarray(ref.Tsk), atol=1e-9)
    assert float(np.sum(ref.BSA)) < 1.0, "a child's skin area, not the 1.87 m² default body"


def test_the_three_bodies_order_as_their_sizes_do(people) -> None:  # type: ignore[no-untyped-def]
    """Same weather; head > hand > foot for each, and skin area girl < woman < man."""
    bodies = {
        p: HumanBodySolver(people[p].human.model_copy(update={"garments": {}}), _weather(0.0), 0.0)
        for p in PEOPLE
    }
    for p, b in bodies.items():
        sk = b.skin_k()
        assert (
            sk[JOS3_SEGMENTS.index("Head")]
            > sk[JOS3_SEGMENTS.index("LHand")]
            > sk[JOS3_SEGMENTS.index("LFoot")]
        ), p
    area = {p: float(np.sum(b._model.BSA)) for p, b in bodies.items()}
    assert area["girl_child_dressed"] < area["woman_adult_dressed"] < area["man_adult_dressed"]


def test_jos3_overstates_the_girls_metabolism_and_this_test_says_by_how_much(people) -> None:  # type: ignore[no-untyped-def]
    """Recorded, not hidden: JOS-3's Harris-Benedict is an adult equation.

    For the WHO-median girl it gives 60.1 W of basal metabolism against Schofield's 51.5 W for a
    girl of 3-10 (+17 %). Per square metre of skin it is 63.6 W against the man's 44.2: even
    Schofield gives her 54.5, so she runs warmer than the man in the cold either way, but by more
    than she should. Bounded here so a change in JOS-3 or a fix in irsim moves this test, and the
    ADR and the roadmap row say where it goes next (HU.12).
    """
    ph = people["girl_child_dressed"].human.phenotype
    body = HumanBodySolver(
        people["girl_child_dressed"].human.model_copy(update={"garments": {}}), _weather(20.0), 0.0
    )
    jos3_w = float(body._model.BMR) * float(np.sum(body._model.BSA))
    a, b = SCHOFIELD_GIRL_3_10
    schofield_w = (a * ph.mass_kg + b) * KCAL_PER_DAY_W
    assert jos3_w / schofield_w == pytest.approx(1.17, abs=0.03)
