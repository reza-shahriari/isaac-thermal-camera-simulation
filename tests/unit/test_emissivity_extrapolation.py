"""How much of a scene's emissivity is an extension of the nearest band, and where it sits (AT.7).

`total_hemispherical_emissivity` has always computed `extrapolated_fraction`. Every scene build
then took `.value` and **threw the fraction away**, so 61 % of the weight that sets every surface
temperature was an assumption no scene author could see. This file pins what that number is, what
it is made of, and that a report quoting an absolute temperature now carries it.

Two things the roadmap row assumed that turned out not to hold, both measured here:

* **It does not vary across materials.** The row asks for "the worst fraction across its
  materials". The fraction is *bit-identical* for all twenty shipped materials --
  0.606695073 at 300 K for each -- because it is a property of the band set and the Planck
  weight, not of anything a material does. The worst across materials is any of them.
* **The single number hides which assumption is being made.** At 300 K the 0.607 is four fifths
  red tail beyond 13.5 µm, which is the standard and defensible thermal-solver assumption. At
  800 K it is 0.465 of which 0.394 is the *interior gaps* -- 1.7-3.0 µm and 5.0-7.5 µm -- where
  "extend the nearest band" is a far weaker claim, because a real spectrum is bounded on both
  sides and can do anything in between. PH.6 and PH.7 shipped 594 K plumes and 1200 K flames, so
  this is now the regime the project actually renders in.

docs/physics-model.md §6.1; roadmap AT.7; ADR 0043, ADR 0119
"""

from __future__ import annotations

import ast
import pathlib

import pytest

from irsim.config.bands import NOMINAL_RANGES_UM
from irsim.materials.hemispherical import (
    extrapolation_breakdown,
    total_hemispherical_emissivity,
)
from irsim.materials.library import MaterialLibrary
from irsim.scene import EPSILON_EVAL_K, Scene
from irsim.thermal.balance import ThermalProperties

pytestmark = pytest.mark.slow

REPO = pathlib.Path(__file__).resolve().parents[2]

#: Report scripts that print an absolute temperature a *surface emissivity* produced must carry
#: the extrapolated fraction. Everything else is listed with why it does not, so a report added
#: later has to decide rather than inherit silence. Mirrors SC.4's `IDEAL_LENS` registry.
CARRIES_THE_FRACTION = ("validate_thermal_diurnal.py",)
EXEMPT_REPORTS = {
    "validate_sky_r13.py": (
        "quotes an absolute apparent *sky* temperature, which comes from the atmosphere's column "
        "emission and not from any surface's eps_hemi -- there is no material to extrapolate"
    ),
    "validate_atmosphere_band_average.py": (
        "quotes transmittances and band averages, and no temperature"
    ),
    "validation_report.py": (
        "the Tier 4 acceptance report works entirely in DN8 by design (ADR 0068): there is no "
        "conversion from '2 K of apparent temperature' to '8 codes of histogram shift', so it "
        "quotes no absolute temperature to qualify"
    ),
    "fidelity_ablation.py": "reports deltas between configurations, not absolute levels",
    "reference_statistics.py": "reports statistics of public imagery, which carries no eps_hemi",
}


# -- what the number is made of --------------------------------------------------------------


@pytest.mark.parametrize("t_k", [250.0, 273.15, 300.0, 400.0, 600.0, 800.0, 1200.0, 1800.0])
def test_the_three_shares_account_for_the_whole_extrapolation(t_k: float) -> None:
    """Red tail + interior gaps + blue end = the total, to floating point.

    They are integrated from the same mask on the same grid, so this is a structural invariant
    rather than a numerical coincidence -- and it is asserted because a decomposition that did
    not close would let a report understate the assumption while looking complete.
    """
    b = extrapolation_breakdown(t_k)
    parts = b.red_tail_fraction + b.interior_gap_fraction + b.blue_end_fraction
    assert parts == pytest.approx(b.extrapolated_fraction, abs=1e-12)
    assert all(
        x >= 0.0 for x in (b.red_tail_fraction, b.interior_gap_fraction, b.blue_end_fraction)
    )
    assert 0.0 < b.extrapolated_fraction < 1.0


def test_the_fraction_is_identical_for_every_material_in_the_library() -> None:
    """The row's premise, measured and corrected: it does not vary across materials at all.

    Twenty materials with emissivities from 0.113 (bare aluminium) to 0.943 (cotton) return the
    same 0.606695073. A report that said "the worst material is X" would be inventing a ranking
    that does not exist, and a reader would draw the wrong conclusion about which material to
    improve. What actually moves the number is the **temperature**, and the next tests are that.
    """
    library = MaterialLibrary.load()
    fractions, values = {}, {}
    for name in sorted(library.names):
        result = total_hemispherical_emissivity(library[name], 300.0)
        fractions[name] = result.extrapolated_fraction
        values[name] = result.value

    assert len(fractions) >= 19, "the library has shrunk; re-check the claim"
    assert len(set(fractions.values())) == 1, sorted(set(fractions.values()))
    assert next(iter(fractions.values())) == pytest.approx(0.6067, abs=1e-4)
    # and the emissivities really do span the library, so the invariance is not a degenerate case
    assert max(values.values()) - min(values.values()) > 0.7, values


def test_the_material_free_breakdown_is_exactly_the_material_one() -> None:
    """So a report can ask for the fraction with no material and cannot get a different answer."""
    library = MaterialLibrary.load()
    for t_k in (273.15, 300.0, 400.0):
        free = extrapolation_breakdown(t_k)
        for name in ("concrete", "bare_aluminium"):
            bound = total_hemispherical_emissivity(library[name], t_k)
            assert bound.extrapolated_fraction == free.extrapolated_fraction
            assert bound.red_tail_fraction == free.red_tail_fraction
            assert bound.interior_gap_fraction == free.interior_gap_fraction
            assert bound.bands_used == free.bands_used


# -- what actually moves it --------------------------------------------------------------------


def test_which_assumption_dominates_flips_with_temperature() -> None:
    """At ambient it is the tail; by 800 K it is the interior gaps. Same number, different claim.

    This is why the total alone is not enough to act on. Holding ε flat beyond 13.5 µm is the
    standard thermal-solver assumption and real dielectrics do stay high and flat out there.
    Holding it flat across 5.0-7.5 µm is a much weaker claim, and it is the dominant one for
    anything as hot as an exhaust manifold.
    """
    ambient = extrapolation_breakdown(300.0)
    assert ambient.red_tail_fraction == pytest.approx(0.508, abs=0.005)
    assert ambient.interior_gap_fraction == pytest.approx(0.099, abs=0.005)
    assert ambient.red_tail_fraction > 4.0 * ambient.interior_gap_fraction

    hot = extrapolation_breakdown(800.0)
    assert hot.red_tail_fraction == pytest.approx(0.071, abs=0.005)
    assert hot.interior_gap_fraction == pytest.approx(0.394, abs=0.005)
    assert hot.interior_gap_fraction > 5.0 * hot.red_tail_fraction

    # the crossover is inside the range this project now renders (PH.6 plumes, PH.7 flames)
    crossings = [
        t
        for t in range(300, 900, 25)
        if extrapolation_breakdown(float(t)).interior_gap_fraction
        > extrapolation_breakdown(float(t)).red_tail_fraction
    ]
    assert crossings and 400 <= crossings[0] <= 600, crossings


def test_the_fraction_is_not_monotone_in_temperature() -> None:
    """It falls to a minimum near 800 K and rises again -- the SWIR-MWIR hole swallows the peak.

    A reader who assumed "hotter is better covered" would stop checking above a few hundred
    kelvin. The 1.7-3.0 µm gap sits right where a 1200 K Planck peak lands, so a flame is *less*
    well covered than a 500 K manifold.
    """
    curve = {
        t: extrapolation_breakdown(float(t)).extrapolated_fraction
        for t in (250, 300, 400, 600, 800, 1200, 1800)
    }
    assert curve[250] > curve[300] > curve[400] > curve[600] > curve[800], curve
    assert curve[1200] > curve[800], "the fraction turns back up as the peak enters the SWIR gap"
    assert curve[1200] == pytest.approx(0.509, abs=0.005)

    # and the turn is the gap, not the tail: the tail keeps shrinking through it
    assert (
        extrapolation_breakdown(1200.0).red_tail_fraction
        < extrapolation_breakdown(800.0).red_tail_fraction
    )
    assert (
        extrapolation_breakdown(1200.0).interior_gap_fraction
        > extrapolation_breakdown(800.0).interior_gap_fraction
    )


def test_the_gaps_are_the_ones_the_nominal_band_set_actually_leaves() -> None:
    """Named rather than implied, so the number cannot drift from the band table it describes."""
    assert NOMINAL_RANGES_UM["swir"][1] == 1.7 and NOMINAL_RANGES_UM["mwir"][0] == 3.0
    assert NOMINAL_RANGES_UM["mwir"][1] == 5.0 and NOMINAL_RANGES_UM["lwir"][0] == 7.5
    # at 300 K only lwir+mwir carry weight, so the only interior gap is 5.0-7.5 um
    assert extrapolation_breakdown(300.0).bands_used == ("lwir", "mwir")
    # widen the set by one contiguous band and the interior gap must shrink
    widened = extrapolation_breakdown(300.0, bands={"lwir": (7.5, 13.5), "gap": (5.0, 7.5)})
    assert widened.interior_gap_fraction < 1e-9
    assert widened.extrapolated_fraction < extrapolation_breakdown(300.0).extrapolated_fraction


# -- the scene reports it ------------------------------------------------------------------------


@pytest.mark.parametrize(
    "scene_name",
    [
        "thermal_facet_scene.yaml",
        "car_ignition_clear_night.yaml",
        "wall_half_in_sun.yaml",
    ],
)
def test_a_scene_build_reports_its_extrapolated_fraction(scene_name: str) -> None:
    """AT.7's first clause, on real shipped scene configs.

    Named rather than globbed, and with no skip guard: a scene that is renamed away should fail
    here instead of quietly passing on three of four.
    """
    path = REPO / "configs" / "scenes" / scene_name
    assert path.exists(), f"{scene_name} is no longer a shipped scene; update this list"
    scene = Scene.from_file(path)
    report = scene.emissivity_extrapolation()

    assert report.evaluated_at_k == EPSILON_EVAL_K
    assert report.fraction == pytest.approx(0.6067, abs=1e-4)
    assert report.materials, "a thermal scene names at least one material"
    assert set(report.materials) <= set(MaterialLibrary.load().names)
    assert "extrapolat" in report.summary()
    assert f"{report.fraction:.3f}" in report.summary()
    assert "identical for each" in report.summary()


def test_a_scene_with_no_library_material_says_so_instead_of_quoting_one() -> None:
    """`car_exhaust_plume.yaml` names none -- its targets are exhaust-line solvers.

    The fraction is a property of the band set and still computable, but a summary that said
    "0 material(s), same fraction for each" would read as though it had checked something. The
    branch is pinned so the empty case cannot quietly start claiming coverage.
    """
    scene = Scene.from_file(REPO / "configs" / "scenes" / "car_exhaust_plume.yaml")
    report = scene.emissivity_extrapolation()
    assert report.materials == ()
    assert report.fraction == pytest.approx(0.6067, abs=1e-4)
    assert "binds nothing here" in report.summary()
    assert "identical for each" not in report.summary()


def test_the_report_describes_the_emissivity_that_is_actually_in_the_solver() -> None:
    """The evaluation temperature the report quotes is the one the build used, not a second copy.

    Before AT.7 the 300 K appeared inline twice in `scene.py` and nowhere else; a report that
    hardcoded its own would have been right by luck. `EPSILON_EVAL_K` is now the single point,
    and this checks the solver's ε really is the one taken at it.
    """
    scene = Scene.from_file(REPO / "configs" / "scenes" / "thermal_facet_scene.yaml")
    library = MaterialLibrary.load()
    name = scene.material_names()[0]
    at_eval = ThermalProperties.from_material(library[name], EPSILON_EVAL_K).emissivity
    assert scene.emissivity_extrapolation().evaluated_at_k == EPSILON_EVAL_K
    # a 30 K move changes eps by far less than the material data is worth, which is *why* one
    # fixed evaluation point is defensible -- asserted so the justification stays true
    at_330 = ThermalProperties.from_material(library[name], 330.0).emissivity
    assert abs(at_330 - at_eval) < 2e-3, (at_eval, at_330)


def test_a_scene_can_be_asked_what_the_fraction_would_be_when_hot() -> None:
    scene = Scene.from_file(REPO / "configs" / "scenes" / "thermal_facet_scene.yaml")
    hot = scene.emissivity_extrapolation(800.0)
    assert hot.evaluated_at_k == 800.0
    assert hot.interior_gap_fraction > hot.red_tail_fraction
    assert hot.materials == scene.emissivity_extrapolation().materials


# -- the reports carry it -------------------------------------------------------------------------


def test_every_report_script_has_decided_whether_to_carry_the_fraction() -> None:
    """AT.7's second clause, as a registry: a new report must classify itself.

    The same discipline as SC.4's `IDEAL_LENS`. A report that quotes an absolute temperature a
    surface emissivity produced carries the fraction; anything else says why not, in writing.
    Adding `scripts/validate_something.py` without touching this file fails here.
    """
    reports = sorted(
        p.name
        for p in (REPO / "scripts").glob("*.py")
        if p.name.startswith("validate")
        or p.name in {"validation_report.py", "fidelity_ablation.py", "reference_statistics.py"}
    )
    classified = set(CARRIES_THE_FRACTION) | set(EXEMPT_REPORTS)
    assert set(reports) == classified, (
        f"unclassified report scripts: {sorted(set(reports) - classified)}; "
        f"stale entries: {sorted(classified - set(reports))}"
    )
    assert all(EXEMPT_REPORTS[name].strip() for name in EXEMPT_REPORTS), (
        "every exemption carries a written reason"
    )


@pytest.mark.parametrize("script", CARRIES_THE_FRACTION)
def test_the_reports_that_must_carry_it_call_for_it(script: str) -> None:
    """Checked in the AST, not by a substring: a mention in a comment is not a report."""
    tree = ast.parse((REPO / "scripts" / script).read_text())
    calls = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert "emissivity_extrapolation" in calls, f"{script} quotes absolute temperatures without it"
    assert "summary" in calls


def test_the_tier4_acceptance_report_really_does_quote_no_absolute_temperature() -> None:
    """Pins the premise of `validation_report.py`'s exemption rather than trusting the note.

    If a Tier 4 check ever starts reporting kelvin, this fails and the exemption has to be
    revisited -- which is the only way a written reason stays true.
    """
    from irsim.validation.compare import Tier4Targets

    fields = Tier4Targets().__dataclass_fields__
    assert not any(name.endswith("_k") or "kelvin" in name for name in fields), sorted(fields)
    source = (REPO / "scripts" / "validation_report.py").read_text()
    assert "273.15" not in source and "apparent_temperature" not in source
