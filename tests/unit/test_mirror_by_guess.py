"""AT.18 — a name glob may not decide that a surface is a mirror.

Emissivity is a property of the *surface state*, not of the substance (docs/physics-model.md §4.5),
and the two are not recoverable from each other: an asset whose material is called "Metal" may be
polished, anodised or painted, and the published spread for one aluminium runs 0.055–0.856. The
cost of choosing wrong is not symmetric. Reading a matte housing as polished replaces the part's
own temperature with the sky's, and the sky in LWIR is tens of kelvin colder than anything on an
airframe; reading a polished trim ring as matte costs a few tenths.

So the rule is about *who* is allowed to say it, not about which answer is likelier: a semantic
class or a name glob is an inference from what somebody called a thing, while a per-asset map or a
prim override is a statement about that asset. Only the second may reach a mirror.

This is a physics test, not a lint: the first case measures what the mistake costs through the band
LUT rather than asserting that a YAML field changed.
"""

from __future__ import annotations

import numpy as np
import pytest

from irsim.materials.library import MaterialLibrary
from irsim.materials.mapping import (
    MIRROR_EMISSIVITY,
    MappingRules,
    MaterialResolver,
    PatternRule,
    PrimRecord,
    audit,
    guessed_mirrors,
    load_mapping_rules,
)
from irsim.materials.table import MaterialTable

#: FLIR Boson 640, docs/physics-model.md §16.1. The gap below is quoted against it because "38 K"
#: means nothing without the instrument that would have to resolve it.
NETD_K = 0.050

T_SURFACE_K = 300.0
#: A clear LWIR sky at zenith sits 40–60 K below air temperature (§5.3); 250 K is the middle of
#: that for a 290 K day, and the test's claim is insensitive to the exact value.
T_SKY_K = 250.0


@pytest.fixture(scope="module")
def library() -> MaterialLibrary:
    return MaterialLibrary.load()


def _apparent_k(lut, emissivity: float) -> float:
    """T_app of a 300 K surface of this emissivity, reflecting a 250 K sky. §4.1, §5.1."""
    l_surface = float(lut.lookup(np.float64(T_SURFACE_K))[()])
    l_sky = float(lut.lookup(np.float64(T_SKY_K))[()])
    mixed = emissivity * l_surface + (1.0 - emissivity) * l_sky
    return float(lut.apparent_temperature(np.float32(mixed))[()])


def test_guessing_polished_metal_costs_thirty_eight_kelvin(library, tophat_lwir_lut) -> None:
    """The measurement AT.18 exists for.

    `bare_aluminium` at ε 0.09 and `aircraft_aluminium_painted` at ε 0.90 are the two answers the
    `*metal*` glob could give for the same motor housing. At ε 0.09 the surface is 91 % mirror, so
    what the camera integrates is nearly all sky: the housing reports a temperature near the sky's,
    not near its own. The gap is quoted in NETD because that is what decides whether a difference
    is visible, and at 760 × NETD this one is not a subtlety — it is the whole signal.
    """
    mirror = float(library["bare_aluminium"].band_properties("lwir").emissivity)
    matte = float(library["aircraft_aluminium_painted"].band_properties("lwir").emissivity)
    assert mirror < MIRROR_EMISSIVITY < matte

    t_mirror = _apparent_k(tophat_lwir_lut, mirror)
    t_matte = _apparent_k(tophat_lwir_lut, matte)

    # Direction first: the low-ε surface reports the *sky*, the high-ε one reports *itself*.
    assert abs(t_mirror - T_SKY_K) < abs(t_mirror - T_SURFACE_K)
    assert abs(t_matte - T_SURFACE_K) < abs(t_matte - T_SKY_K)

    gap = t_matte - t_mirror
    # 40.1 K over this fixture's exact 7.5-13.5 um top-hat. The same calculation over the Boson's
    # measured response gives 38.0 K, which is the figure the roadmap row and CHANGELOG quote --
    # the 2 K between them is band shape, not disagreement, and neither is near the NETD.
    assert gap == pytest.approx(40.1, abs=0.5)
    assert gap / NETD_K > 600.0


def test_no_guessing_route_in_the_shipped_rules_reaches_a_mirror(library) -> None:
    """The invariant, over the rules the project actually ships.

    Red before AT.18: `*chrome*`, `*alumin*`, `*metal*` and the `aircraft` semantic class all named
    `bare_aluminium`, so four of the commonest names on an imported airframe resolved to a mirror.
    """
    rules = load_mapping_rules(known_materials=library.names)
    guessed = set(rules.semantic.values()) | {r.material for r in rules.patterns}
    dark = {
        name
        for name in guessed
        if float(library[name].band_properties("lwir").emissivity) < MIRROR_EMISSIVITY
    }
    assert not dark, f"a glob or semantic class may not name a mirror: {sorted(dark)}"
    # And the material itself still exists, for an asset that wants to assert it.
    assert float(library["bare_aluminium"].band_properties("lwir").emissivity) < MIRROR_EMISSIVITY


def test_the_audit_fails_a_guessed_mirror_and_passes_an_asserted_one(library) -> None:
    """The gate, on rules built to break it — the audit must catch rather than merely pass.

    `passed` is not the coverage alone: one prim guessed into a mirror fails outright, because a
    single such prim is worth 38 K where a single unmapped prim out of a hundred is worth nothing.
    """
    names = MaterialTable.from_library(library, "lwir").names
    emissivity = {n: float(library[n].band_properties("lwir").emissivity) for n in library.names}
    bad_rules = MappingRules(
        coverage_threshold=0.5,
        semantic={},
        patterns=[PatternRule(match="*trim*", material="bare_aluminium")],
    )
    resolver = MaterialResolver(bad_rules, names)

    guessed = audit([PrimRecord("/W/a", "Chrome_Trim")], resolver, emissivity=emissivity)
    assert guessed.coverage == 1.0 and not guessed.passed
    assert "/W/a" in guessed.mirror_guesses
    assert "bare_aluminium" in guessed.render() and "FAIL" in guessed.render()

    asserted = audit(
        [PrimRecord("/W/a", "Chrome_Trim", override="bare_aluminium")],
        MaterialResolver(bad_rules, names),
        emissivity=emissivity,
    )
    assert asserted.passed and not asserted.mirror_guesses

    # Omitting `emissivity` keeps the pre-AT.18 coverage-only report, for a caller with no library.
    assert audit([PrimRecord("/W/a", "Chrome_Trim")], MaterialResolver(bad_rules, names)).passed


def test_guessed_mirrors_names_the_rule_that_fired(library) -> None:
    """The report has to say *why*, or the fix is a search rather than an edit."""
    names = MaterialTable.from_library(library, "lwir").names
    emissivity = {n: float(library[n].band_properties("lwir").emissivity) for n in library.names}
    rules = MappingRules(
        coverage_threshold=0.5,
        semantic={"hull": "bare_aluminium"},
        patterns=[],
    )
    resolver = MaterialResolver(rules, names)
    found = guessed_mirrors([resolver.resolve(PrimRecord("/W/h", None, "hull"))], emissivity)
    assert found == {"/W/h": "bare_aluminium via semantic 'hull'"}
