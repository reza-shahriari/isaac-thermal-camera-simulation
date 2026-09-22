"""People: skin and clothing are two temperatures on one body (PH.12).

The measurement, at 0 °C in still air with 1 clo: skin **34.07 °C**, clothing **13.96 °C**, a
**20.1 K** step across one person. Indoors at 22 °C in 0.5 clo the same person's step is **4.9 K**.
How much of a person stands out is a property of the weather, not of the person.

⚠️ **PH.12's third criterion does not hold at the condition it names, and the reason is worth
more than the criterion.** The row expects 1 clo at 0 °C to land the clothing 10-15 °C below skin;
ISO 7730's own equation gives **20.1 K** there. The row's band is what the same equation produces
at **10-15 °C** of air -- which is exactly ISO 7730's Annex A validity floor. The criterion was
written for a condition the standard does not cover, and the model is behaving correctly outside
it. `test_the_ten_to_fifteen_kelvin_band_belongs_to_the_standards_own_range` pins that rather than
loosening the band until 20.1 fits.

The fourth criterion -- agreement with `pythermalcomfort`'s two-node model to 0.5 K -- **is not
run here**: the package is not installed and is not a dependency of this project. The test that
would run it skips, loudly, and `test_the_two_node_oracle_is_declared_even_though_it_cannot_run`
asserts that the extra is declared so the skip is a recorded gap rather than an absence.

docs/physics-model.md §6.1, §16.2; roadmap PH.12; ADR 0122; ISO 7730:2005 §4
"""

from __future__ import annotations

import math
import pathlib

import pytest

from irsim.thermal.human import (
    CLO_M2K_W,
    MET_W_M2,
    SEATED_MET,
    WALKING_MET,
    clothing_area_factor,
    clothing_temperature_c,
    convective_coefficient_w_m2_k,
    human_surfaces,
    skin_temperature_c,
)

pytestmark = pytest.mark.slow

REPO = pathlib.Path(__file__).resolve().parents[2]
#: ISO 7730 Annex A's stated validity for operative temperature, °C.
ISO_7730_VALID_AIR_C = (10.0, 30.0)


# -- the skin set point --------------------------------------------------------------------------


def test_skin_is_authored_from_metabolism_and_falls_as_work_rises() -> None:
    """34.07 °C seated, 32.44 °C walking -- working *lowers* skin, which is counter-intuitive.

    More metabolic heat means more of it leaves by sweat and blood flow rather than by the
    surface, so the set point drops. A model that raised skin temperature with activity would
    look obviously sensible and be wrong.
    """
    seated = skin_temperature_c(SEATED_MET * MET_W_M2)
    walking = skin_temperature_c(WALKING_MET * MET_W_M2)
    assert seated == pytest.approx(34.07, abs=0.01)
    assert walking == pytest.approx(32.44, abs=0.01)
    assert walking < seated
    assert seated - walking == pytest.approx(0.028 * MET_W_M2, rel=1e-12)

    # external work is subtracted, so pushing a load raises skin back toward the seated value
    assert skin_temperature_c(2.0 * MET_W_M2, 0.5 * MET_W_M2) > walking
    with pytest.raises(ValueError):
        skin_temperature_c(0.0)
    with pytest.raises(ValueError):
        skin_temperature_c(100.0, 200.0)


# -- the clothing balance ------------------------------------------------------------------------


def test_no_clothing_gives_the_skin_temperature_exactly() -> None:
    """PH.12's first criterion, and it is exact rather than converged.

    ``I_cl = 0`` multiplies the whole flux term away, so the answer is the set point with no
    residual at all. Returning it from the bisection instead would leave a tolerance-sized error
    on the one case that has a closed answer.
    """
    for t_air in (-20.0, 0.0, 22.0, 40.0):
        t_cl = clothing_temperature_c(t_air, t_air, 0.0)
        assert t_cl == skin_temperature_c(MET_W_M2), "bit-exact, not approximately"
    assert human_surfaces(0.0, clo=0.0).step_k == 0.0


def test_one_clo_at_zero_celsius_is_twenty_kelvin_not_the_rows_ten_to_fifteen() -> None:
    """PH.12's third criterion measured at the condition it names, and it is 20.1 K.

    Not a failure of the model: ISO 7730's equation gives this, and the next test shows the row's
    band is what the same equation produces inside the standard's own validity range. Recorded at
    the measured value so nobody later "fixes" a correct implementation to hit a band written for
    a different condition.
    """
    surfaces = human_surfaces(0.0, clo=1.0)
    assert surfaces.skin_k - 273.15 == pytest.approx(34.07, abs=0.01)
    assert surfaces.clothing_k - 273.15 == pytest.approx(13.96, abs=0.05)
    assert surfaces.step_k == pytest.approx(20.11, abs=0.05)
    assert surfaces.step_k > 15.0, "the row's upper bound is exceeded at its own condition"


def test_the_ten_to_fifteen_kelvin_band_belongs_to_the_standards_own_range() -> None:
    """The row's band is the equation's answer at 10-15 °C -- ISO 7730's validity floor.

    Two ways to land inside it, both measured: keep 1 clo and warm the air into the standard's
    range, or keep 0 °C and halve the insulation. That the band coincides with the validity floor
    is the evidence that the criterion was written for an in-range condition and then quoted at
    an out-of-range one.
    """
    low, high = ISO_7730_VALID_AIR_C
    inside = {t: human_surfaces(t, clo=1.0).step_k for t in (0.0, 5.0, 10.0, 15.0, 20.0)}
    assert 10.0 <= inside[10.0] <= 15.0, inside
    assert 10.0 <= inside[15.0] <= 15.0, inside
    assert inside[0.0] > 15.0 and inside[20.0] < 10.0, inside
    # the two temperatures that land in the band are exactly the standard's floor region
    assert low == 10.0 and high == 30.0

    # or, at the row's 0 °C, half the insulation
    assert human_surfaces(0.0, clo=0.5).step_k == pytest.approx(14.41, abs=0.05)
    assert 10.0 <= human_surfaces(0.0, clo=0.3).step_k <= 15.0


def test_wind_lowers_the_clothing_surface(tmp_path: pathlib.Path) -> None:
    """PH.12's second criterion, monotone across two decades of air speed."""
    del tmp_path
    speeds = (0.1, 0.5, 1.0, 2.0, 5.0)
    temps = [human_surfaces(0.0, clo=1.0, air_speed_m_s=v).clothing_k for v in speeds]
    assert all(b < a for a, b in zip(temps, temps[1:], strict=False)), temps
    assert temps[0] - 273.15 == pytest.approx(13.96, abs=0.05)
    assert temps[-1] - 273.15 == pytest.approx(5.31, abs=0.05)
    # skin does not move: it is a set point, not a balance
    skins = {human_surfaces(0.0, clo=1.0, air_speed_m_s=v).skin_k for v in speeds}
    assert len(skins) == 1


def test_wind_drives_the_coat_toward_air_temperature_in_either_direction() -> None:
    """Wind does not always cool: a heavily insulated coat under a cold sky is **warmed** by it.

    "More wind lowers t_cl" holds only while the coat is above air temperature, and PH.12's own
    criterion is stated that way. It is the special case. The rule is that wind couples the
    surface to the **air**, so it moves it toward air temperature from whichever side it is on.

    Measured at −10 °C air under a −40 °C sky: at 1 and 2 clo the coat still sits above air and
    wind cools it, but by **3 clo** the radiative loss has dragged it *below* air (−10.31 °C) and
    the same wind pushes it **up**. Only a regime sweep finds this; a monotone-cooling test would
    pass at every insulation a person actually wears and hide the sign change beyond it.
    """
    speeds = (0.1, 1.0, 5.0)
    series = {
        clo: [
            clothing_temperature_c(-10.0, -40.0, clo * CLO_M2K_W, air_speed_m_s=v) for v in speeds
        ]
        for clo in (1.0, 2.0, 3.0, 4.0)
    }
    assert series[1.0][0] > series[1.0][-1], "above air, wind cools"
    assert series[2.0][0] > series[2.0][-1]
    assert series[4.0][0] < series[4.0][-1], "below air, the same wind warms"
    assert series[3.0][0] < -10.0 < series[1.0][0], "3 clo is where it crosses the air"

    # whichever side it starts on, wind takes it toward air temperature
    for values in series.values():
        assert abs(values[-1] + 10.0) < abs(values[0] + 10.0), values


def test_a_cold_sky_pulls_the_coat_down_further_than_the_air_does() -> None:
    """The mean radiant temperature is the term that makes an outdoor coat read cold.

    Indoors ``t_r ≈ t_a`` and the radiative term nearly vanishes. Under a clear winter sky at
    −40 °C the same person at the same air temperature reads **7.1 K** colder on the coat, which
    is why an outdoor scene must pass its own sky and not default to air.
    """
    same = human_surfaces(0.0, t_radiant_c=0.0, clo=1.0).clothing_k
    cold_sky = human_surfaces(0.0, t_radiant_c=-40.0, clo=1.0).clothing_k
    assert same - cold_sky == pytest.approx(7.1, abs=0.2)
    assert human_surfaces(0.0, clo=1.0).clothing_k == same, "t_r defaults to the air"


def test_the_solver_converges_where_a_fixed_point_iteration_diverges() -> None:
    """The regime that broke the textbook form: a coat in wind under a cold sky.

    An undamped ISO 7730 fixed point diverges here, and damping by a half only postpones it. The
    residual is strictly decreasing in t_cl, so a bracketed root cannot fail -- asserted by
    checking the sign change across the bracket rather than by the answer looking plausible.
    """
    for speed in (1.0, 5.0, 20.0):
        for clo in (0.5, 2.0, 4.0):
            t_cl = clothing_temperature_c(-10.0, -40.0, clo * CLO_M2K_W, air_speed_m_s=speed)
            assert math.isfinite(t_cl)
            t_sk = skin_temperature_c(MET_W_M2)
            assert -40.0 - 1.0 <= t_cl <= t_sk + 1e-9


# -- the pieces --------------------------------------------------------------------------------


def test_the_area_factor_never_shrinks_a_person() -> None:
    """f_cl ≥ 1 always, and the two branches meet at the 0.5 clo break."""
    assert clothing_area_factor(0.0) == pytest.approx(1.0)
    assert clothing_area_factor(0.078) == pytest.approx(1.0 + 1.290 * 0.078)
    assert clothing_area_factor(CLO_M2K_W) == pytest.approx(1.05 + 0.645 * CLO_M2K_W)
    for i_cl in (0.0, 0.05, 0.078, 0.1, 0.155, 0.5):
        assert clothing_area_factor(i_cl) >= 1.0
    # the branches are within a per cent of each other at the break, not a step
    below = 1.00 + 1.290 * 0.078
    above = 1.05 + 0.645 * 0.078
    assert abs(above - below) / below < 0.01
    with pytest.raises(ValueError):
        clothing_area_factor(-0.1)


def test_convection_takes_the_larger_of_free_and_forced() -> None:
    """Not their sum: one mechanism sets the boundary layer, which is ISO 7730's own choice."""
    still = convective_coefficient_w_m2_k(20.0, 0.0, 0.0)
    assert still == pytest.approx(2.38 * 20.0**0.25)
    windy = convective_coefficient_w_m2_k(20.0, 0.0, 4.0)
    assert windy == pytest.approx(12.1 * 2.0)
    assert windy > still
    # at the crossover the two are equal and the max is continuous
    crossover = (2.38 * 20.0**0.25 / 12.1) ** 2
    assert convective_coefficient_w_m2_k(20.0, 0.0, crossover) == pytest.approx(
        2.38 * 20.0**0.25, rel=1e-9
    )
    # an isothermal surface still convects if there is wind
    assert convective_coefficient_w_m2_k(0.0, 0.0, 1.0) == pytest.approx(12.1)
    assert convective_coefficient_w_m2_k(0.0, 0.0, 0.0) == 0.0
    with pytest.raises(ValueError):
        convective_coefficient_w_m2_k(20.0, 0.0, -1.0)


def test_the_indoor_and_outdoor_cases_differ_by_four_times() -> None:
    """A person indoors is a much weaker target than the same person outside.

    4.93 K step at 22 °C in 0.5 clo against 20.11 K at 0 °C in 1 clo. Both are the same body at
    the same metabolic rate; a single-temperature human would render both identically.
    """
    indoors = human_surfaces(22.0, clo=0.5)
    outdoors = human_surfaces(0.0, clo=1.0)
    assert indoors.step_k == pytest.approx(4.93, abs=0.05)
    assert outdoors.step_k == pytest.approx(20.11, abs=0.05)
    assert outdoors.step_k / indoors.step_k > 4.0
    assert indoors.skin_k == outdoors.skin_k, "same body, same set point"


# -- the oracle this cannot run --------------------------------------------------------------------


def test_the_two_node_oracle_is_declared_even_though_it_cannot_run() -> None:
    """PH.12's fourth criterion is **not met**, and the gap is declared rather than silent.

    `pythermalcomfort` is a dev-only oracle: it must never be imported by `src/irsim`, and it is
    not installed here, so the cross-check does not run. What this asserts is that the optional
    extra exists in `pyproject.toml` -- so the gap is a recorded, installable one rather than a
    criterion quietly dropped -- and that nothing under `src/` imports it.
    """
    pyproject = (REPO / "pyproject.toml").read_text()
    assert "pythermalcomfort" in pyproject, "the oracle must be declared as an optional extra"
    assert "comfort = [" in pyproject

    sources = list((REPO / "src" / "irsim").rglob("*.py"))
    assert sources
    offenders = [p for p in sources if "pythermalcomfort" in p.read_text()]
    assert not offenders, f"the physics core must not import the oracle: {offenders}"


def test_against_pythermalcomfort_when_it_is_installed() -> None:
    """The comparison PH.12 asked for, run only where the extra is installed.

    Skips loudly rather than passing vacuously. Install with `pip install -e '.[comfort]'`.
    """
    comfort = pytest.importorskip(
        "pythermalcomfort", reason="dev-only oracle; install with .[comfort] to run PH.12's check"
    )
    result = comfort.models.two_nodes(tdb=25.0, tr=25.0, v=0.1, rh=50.0, met=SEATED_MET, clo=0.5)
    ours = human_surfaces(25.0, t_radiant_c=25.0, clo=0.5, met=SEATED_MET, air_speed_m_s=0.1)
    assert float(result.t_skin) == pytest.approx(ours.skin_k - 273.15, abs=0.5)
