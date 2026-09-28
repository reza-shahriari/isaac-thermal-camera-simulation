"""XD.13 -- materials built from published spectra, and the solar absorptance computed from them.

`scripts/import_material_spectra.py` writes a material's emissivity *curve* and its n/k table
from a published source, and computes its solar absorptivity from the curve. These tests hold the
three files to each other and to physics that does not come from the same source:

* the curve is the Fresnel emissivity of the table, recomputed here by the library's own Fresnel
  (a different piece of code from the script's);
* the solar absorptivity in the YAML is the curve averaged under the solar spectrum;
* a polished metal's long-wave emissivity sits where its DC resistivity puts it (Hagen-Rubens),
  which is independent of every optical measurement -- a swapped n/k column or a table in the
  wrong unit lands orders of magnitude away;
* and it rises toward grazing, as a metal's does and a dielectric's does not.

docs/physics-model.md §4.2, §4.5, §5.4, §12.3; ADR 0041; roadmap XD.13
"""

from __future__ import annotations

import math
import pathlib

import numpy as np
import pytest

from irsim.materials.directional import directional_emissivity
from irsim.materials.fresnel import fresnel_reflectance
from irsim.materials.library import MATERIAL_DIR, MaterialLibrary
from irsim.materials.nk import load_nk_table
from irsim.materials.spectra import (
    SOLAR_RANGE_UM,
    PropertySpectrum,
    load_property_spectrum,
    solar_absorptance,
)
from irsim.radiometry.constants import C_LIGHT, EPSILON_0
from irsim.radiometry.solar import AM15_DIRECT_FILE, load_solar_spectrum, solar_data_dir

WRITER = "scripts/import_material_spectra.py"

#: DC resistivity at 293 K, ohm m. CRC Handbook of Chemistry and Physics, "Electrical Resistivity
#: of Pure Metals". Test data for an independent check, not a constant the code uses.
RESISTIVITY_OHM_M = {
    "silver_polished": 1.587e-8,
    "copper_polished": 1.678e-8,
    "gold_polished": 2.214e-8,
    "magnesium_polished": 4.39e-8,
    "steel_polished": 9.61e-8,  # the table is pure iron's, so the resistivity is too
    "chrome_plated": 12.5e-8,
    "titanium_polished": 42.0e-8,
}


def imported() -> list[str]:
    return sorted(
        p.stem for p in MATERIAL_DIR.glob("*.yaml") if WRITER in p.read_text(encoding="utf-8")
    )


@pytest.fixture(scope="module")
def library() -> MaterialLibrary:
    return MaterialLibrary.load()


@pytest.fixture(scope="module")
def sun():
    return load_solar_spectrum(solar_data_dir().parent.parent / AM15_DIRECT_FILE)


def _curve(lam, values) -> PropertySpectrum:
    return PropertySpectrum(
        np.asarray(lam, dtype=np.float64),
        np.asarray(values, dtype=np.float64),
        pathlib.Path("synthetic.csv"),
    )


# --- the solar average itself ---------------------------------------------------------------


@pytest.mark.parametrize("value", [0.0, 0.3, 1.0])
def test_a_flat_absorptance_averages_to_itself(sun, value: float) -> None:
    assert solar_absorptance(_curve([0.2, 20.0], [value, value]), sun) == pytest.approx(value)


def test_a_step_absorbs_the_share_of_sunlight_on_its_dark_side(sun) -> None:
    """Black below 0.7 um and white above: the absorptance is the visible share of the energy."""
    step = _curve([0.2, 0.6999, 0.7001, 20.0], [1.0, 1.0, 0.0, 0.0])
    lo, hi = SOLAR_RANGE_UM
    lam = sun.wavelength_um[(sun.wavelength_um >= lo) & (sun.wavelength_um <= hi)]
    e = np.interp(lam, sun.wavelength_um, sun.values)
    visible = np.trapezoid(np.where(lam <= 0.7, e, 0.0), lam) / np.trapezoid(e, lam)
    assert solar_absorptance(step, sun) == pytest.approx(visible, abs=2e-3)
    # And it is a real share, not 0 or 1: the test would pass vacuously otherwise.
    assert 0.3 < visible < 0.6


def test_a_curve_that_stops_short_of_the_solar_band_is_refused(sun) -> None:
    with pytest.raises(ValueError, match="solar absorptance needs"):
        solar_absorptance(_curve([0.36, 16.0], [0.5, 0.5]), sun)


# --- the imported materials -------------------------------------------------------------------


def test_the_importer_has_written_materials() -> None:
    assert len(imported()) >= 7


@pytest.mark.parametrize("name", imported())
def test_the_curve_is_the_fresnel_emissivity_of_its_table(library, name: str) -> None:
    material = library[name]
    curve = material.spectrum
    table = load_nk_table(str(material.n_k_path))
    n, k = table.at(curve.wavelength_um)
    expected = 1.0 - fresnel_reflectance(n, k, np.ones_like(n))[0]
    # The curve is written to 6 significant figures.
    np.testing.assert_allclose(curve.values, expected, rtol=2e-5, atol=1e-6)


@pytest.mark.parametrize("name", imported())
def test_the_solar_absorptivity_is_computed_from_the_curve(library, sun, name: str) -> None:
    material = library[name]
    computed = solar_absorptance(load_property_spectrum(material.spectrum.path), sun)
    assert material.spec.thermal.solar_absorptivity == pytest.approx(computed, abs=6e-4)


@pytest.mark.parametrize("name", sorted(RESISTIVITY_OHM_M))
def test_a_polished_metal_sits_where_its_resistivity_puts_it(library, name: str) -> None:
    """Hagen-Rubens at 10 um: 1 - R = 2 sqrt(2 eps0 omega rho), from the DC resistivity alone.

    It is a low-frequency limit, and 10 um is not quite low enough for the best conductors
    (omega tau is a few for copper), where the measured value falls below it; hence a factor of
    two rather than a percentage. A table with n and k swapped, or in nanometres, misses by
    orders of magnitude.
    """
    omega = 2.0 * math.pi * C_LIGHT / 10e-6
    hagen_rubens = 2.0 * math.sqrt(2.0 * EPSILON_0 * omega * RESISTIVITY_OHM_M[name])
    curve = library[name].spectrum
    eps = float(np.interp(10.0, curve.wavelength_um, curve.values))
    assert 0.5 < eps / hagen_rubens < 2.0, f"{name}: {eps:.4f} against {hagen_rubens:.4f}"


def test_the_ranking_follows_the_resistivity(library) -> None:
    """The best conductors are the best mirrors: silver, copper and gold below the rest."""
    lwir = {n: library[n].band_properties("lwir").emissivity for n in RESISTIVITY_OHM_M}
    best = ("silver_polished", "copper_polished", "gold_polished")
    noble = max(lwir[n] for n in best)
    rest = min(v for n, v in lwir.items() if n not in best)
    assert noble < rest
    assert lwir["titanium_polished"] == max(lwir.values())


@pytest.mark.parametrize("name", imported())
def test_a_polished_metal_is_a_mirror_in_the_long_wave(library, name: str) -> None:
    assert library[name].band_properties("lwir").emissivity < 0.1


@pytest.mark.parametrize("name", imported())
def test_a_polished_metal_brightens_toward_grazing(library, name: str) -> None:
    """A metal's emissivity rises from normal toward grazing (it peaks near 85 deg); a
    dielectric's only falls. The angular model is the same table's Fresnel."""
    material = library[name]
    cos = np.cos(np.radians([0.0, 70.0]))
    normal, grazing = directional_emissivity(material, "lwir", cos)
    assert grazing > 1.5 * normal


def test_no_name_glob_reaches_an_imported_mirror() -> None:
    """AT.18: a polished metal is something an asset asserts, never something a name implies."""
    text = (MATERIAL_DIR / "mapping.yaml").read_text(encoding="utf-8")
    for name in imported():
        assert f"material: {name}" not in text
