"""GT.9 — pigment colour moves the reflective bands and the sun, never the thermal bands.

A widely repeated rule of thumb says "the duller and blacker a material is, the higher its
emissivity". Half of it is real: roughness and oxidation raise emissivity, and measurement bears
that out. The other half is a visible-band intuition that does not survive into the thermal
infrared — paint emissivity is essentially independent of pigment colour over roughly 2–12 µm, and
a flat white coating routinely exceeds a gloss black one (docs/physics-model.md §4.5a).

This library already gets it right: `car_paint_black` and `car_paint_white` carry the same ε in
MWIR and LWIR and differ in `solar_absorptivity`, which is where colour genuinely acts. Nothing
held that, though — it was a property of the numbers somebody typed, so an author reading the folk
rule could have "corrected" it and made the model worse with every test still green.

The test is written so that it cannot pass by asserting that colour never matters: it also
requires the two paints to differ *strongly* in NIR, where colour is exactly what the band sees,
and it prices both mistakes in kelvin rather than asserting that a YAML field is unchanged.
"""

from __future__ import annotations

import numpy as np
import pytest

from irsim.materials.library import MaterialLibrary
from irsim.radiometry.constants import SIGMA_SB

#: FLIR Boson 640, §16.1 — the instrument the kelvin claims below are quoted against.
NETD_K = 0.050

#: Sprayed topcoats. `abs_plastic_white` is deliberately absent: it is a moulded gloss shell, a
#: different surface state, and §4.5 is explicit that the state is what sets ε.
PAINTS = ("car_paint_black", "car_paint_white", "aircraft_aluminium_painted")

#: The bands §4.5a's claim covers. NIR and most of SWIR sit below 2 µm, where pigment dominates.
THERMAL_BANDS = ("mwir", "lwir")


@pytest.fixture(scope="module")
def library() -> MaterialLibrary:
    return MaterialLibrary.load()


def test_every_paint_shares_one_thermal_emissivity(library: MaterialLibrary) -> None:
    """The invariant. A pigment may not move MWIR or LWIR emissivity at all."""
    for band in THERMAL_BANDS:
        values = {name: float(library[name].band_properties(band).emissivity) for name in PAINTS}
        spread = max(values.values()) - min(values.values())
        assert spread == pytest.approx(0.0, abs=1e-9), f"{band}: {values}"


def test_the_same_paints_differ_strongly_where_colour_is_what_the_band_sees(
    library: MaterialLibrary,
) -> None:
    """The control, without which the test above would pass on a library of identical materials.

    In NIR a black topcoat absorbs where a white one reflects, so ε goes the other way: 0.94
    against 0.30. If a future edit flattened the reflective bands too, this fails — the claim is
    that colour acts in *one* place, not that it acts nowhere.
    """
    black = float(library["car_paint_black"].band_properties("nir").emissivity)
    white = float(library["car_paint_white"].band_properties("nir").emissivity)
    assert black / white > 3.0


def test_colour_acts_through_solar_absorptivity_and_it_is_worth_tens_of_kelvin(
    library: MaterialLibrary,
) -> None:
    """Where the colour goes instead — and why the split is not cosmetic.

    Linearising the §6.1 balance about 320 K, a surface in steady state sits
    ΔT = Δα·Q / (h + 4εσT³) above its neighbour. At 800 W/m² of sun, h = 15 W/m²/K and the two
    paints' authored α_sol of 0.94 and 0.28, that is tens of kelvin of *real* temperature
    difference, which the camera then sees honestly through an identical ε.

    So the library's split is load-bearing in both directions: colour is worth ~0 K through
    emissivity and tens of kelvin through absorptivity, and an author who moved it to the other
    side would get a plausible-looking image that is wrong about both.
    """
    a_black = library["car_paint_black"].spec.thermal.solar_absorptivity
    a_white = library["car_paint_white"].spec.thermal.solar_absorptivity
    assert a_black / a_white > 3.0

    t_k, q_sol, h_c, eps = 320.0, 800.0, 15.0, 0.90
    delta_t = (a_black - a_white) * q_sol / (h_c + 4.0 * eps * SIGMA_SB * t_k**3)
    assert delta_t > 20.0
    assert delta_t / NETD_K > 400.0


def test_the_folk_rule_would_cost_kelvin_in_the_thermal_bands(
    library: MaterialLibrary, tophat_lwir_lut
) -> None:
    """What the invariant is protecting, priced.

    Suppose someone followed the rule and authored black at ε 0.95 against white at 0.85 — a
    modest split, well inside what the rule suggests. Two panels at the same 320 K under a 250 K
    sky would then report apparent temperatures several kelvin apart *purely because of their
    paint*, which no thermometer would confirm. The library's actual numbers give exactly zero.
    """

    def apparent(emissivity: float) -> float:
        l_surface = float(tophat_lwir_lut.lookup(np.float64(320.0))[()])
        l_sky = float(tophat_lwir_lut.lookup(np.float64(250.0))[()])
        mixed = emissivity * l_surface + (1.0 - emissivity) * l_sky
        return float(tophat_lwir_lut.apparent_temperature(np.float32(mixed))[()])

    folk_gap = apparent(0.95) - apparent(0.85)
    assert folk_gap > 4.0
    assert folk_gap / NETD_K > 80.0

    real = [float(library[n].band_properties("lwir").emissivity) for n in PAINTS]
    assert apparent(max(real)) - apparent(min(real)) == pytest.approx(0.0, abs=1e-6)
