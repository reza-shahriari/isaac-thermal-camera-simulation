"""XD.13 -- materials from SLUM, measured in both short and long wave on the same sample.

`scripts/import_slum.py` commits each sample's short-wave reflectance and long-wave emissivity
curves and computes the band values and the solar absorptivity from them. These tests re-derive
those numbers from the committed curves, and -- the independent check -- compare them with the
broadband albedo and emissivity the SLUM authors published for the same samples (Kotthaus et al.
2014, appendix table), which they computed with their own method and weighting.

docs/physics-model.md §4.1, §5.4; ADR 0041; roadmap XD.13
"""

from __future__ import annotations

import pathlib
import re
import sys

import pytest

from irsim.materials.library import MATERIAL_DIR, MaterialLibrary
from irsim.materials.spectra import PropertySpectrum, load_property_spectrum, solar_absorptance
from irsim.radiometry.solar import AM15_DIRECT_FILE, load_solar_spectrum, solar_data_dir

REPO = pathlib.Path(__file__).resolve().parents[2]
SLUM_DIR = REPO / "data" / "spectra" / "materials" / "slum"
WRITER = "scripts/import_slum.py"

#: Kotthaus et al. (2014), ISPRS J. Photogramm. Remote Sens. 94, appendix table: broadband albedo
#: and emissivity per sample, keyed by the column of the released CSV. The one entry whose key
#: differs is the slate shingle: the CSV's L001d column is the paper's L001u row (the file and the
#: paper label the two faces the other way round; the values settle which is which).
PAPER = {
    "A001": (0.21, 0.96), "A007": (0.07, 0.93), "A008": (0.13, 0.95), "B001": (0.30, 0.94),
    "B003": (0.09, 0.95), "B008": (0.43, 0.94), "B012": (0.56, 0.95), "B013": (0.32, 0.95),
    "C001": (0.29, 0.94), "C002": (0.21, 0.92), "C004": (0.37, 0.95), "C006": (0.42, 0.95),
    "G001": (0.48, 0.92), "G004": (0.54, 0.93), "L001d": (0.09, 0.90), "L002": (0.05, 0.94),
    "R001": (0.31, 0.93), "R006": (0.12, 0.94), "R007": (0.16, 0.92), "R008": (0.26, 0.95),
    "R010": (0.19, 0.95), "S001": (0.40, 0.90), "V001": (0.08, 0.94), "V002": (0.43, 0.93),
    "V004": (0.14, 0.94), "Z003": (0.11, 0.93), "Z004": (0.45, 0.94), "Z005": (0.12, 0.94),
    "Z006": (0.26, 0.81), "Z008": (0.05, 0.97),
}  # fmt: skip


def slum() -> list[str]:
    return sorted(
        p.stem for p in MATERIAL_DIR.glob("*.yaml") if WRITER in p.read_text(encoding="utf-8")
    )


def sample_of(name: str) -> str:
    text = (MATERIAL_DIR / f"{name}.yaml").read_text(encoding="utf-8")
    return re.search(r"SLUM sample (\S+):", text).group(1)


@pytest.fixture(scope="module")
def library() -> MaterialLibrary:
    return MaterialLibrary.load()


@pytest.fixture(scope="module")
def sun():
    return load_solar_spectrum(solar_data_dir().parent.parent / AM15_DIRECT_FILE)


def _band(lam, values, band: str) -> float:
    sys.path.insert(0, str(REPO / "scripts"))
    from import_paired_spectra import band_value

    return band_value(lam, values, band)[0]


def _curves(name: str) -> tuple[PropertySpectrum, PropertySpectrum]:
    sw = load_property_spectrum(SLUM_DIR / f"{name}_sw.csv")
    lw = load_property_spectrum(SLUM_DIR / f"{name}_lw.csv")
    return PropertySpectrum(sw.wavelength_um, 1.0 - sw.values, sw.path), lw


def test_the_importer_has_written_materials() -> None:
    assert len(slum()) == 30


@pytest.mark.parametrize("name", slum())
def test_three_bands_are_the_committed_curves(library, name: str) -> None:
    sw, lw = _curves(name)
    m = library[name]
    eps = {b: m.band_properties(b).emissivity for b in ("nir", "swir", "lwir")}
    assert eps["nir"] == pytest.approx(_band(sw.wavelength_um, sw.values, "nir"), abs=6e-4)
    assert eps["swir"] == pytest.approx(_band(sw.wavelength_um, sw.values, "swir"), abs=6e-4)
    assert eps["lwir"] == pytest.approx(_band(lw.wavelength_um, lw.values, "lwir"), abs=6e-4)
    # AT.34 (ADR 0175): NIR and SWIR are *read* from the curve, which covers both bands, so no
    # typed number survives beside it; LWIR keeps one, to fill the 7.5-8.0 um below the curve.
    table = m.spec.optical.emissivity_per_band or {}
    assert "nir" not in table and "swir" not in table and "lwir" in table
    assert m.band_properties("nir").curve_fraction == 1.0
    assert 0.85 < m.band_properties("lwir").curve_fraction < 1.0


@pytest.mark.parametrize("name", slum())
def test_the_solar_absorptivity_is_the_committed_curve(library, sun, name: str) -> None:
    sw, _ = _curves(name)
    text = (MATERIAL_DIR / f"{name}.yaml").read_text(encoding="utf-8")
    lo, hi = (float(x) for x in re.search(r"covers ([0-9.]+)-([0-9.]+) um \(", text).groups())
    expected = solar_absorptance(sw, sun, (lo, hi))
    assert library[name].spec.thermal.solar_absorptivity == pytest.approx(expected, abs=6e-4)


@pytest.mark.parametrize("name", slum())
def test_it_agrees_with_what_the_slum_authors_published(library, name: str) -> None:
    """Albedo and broadband emissivity from the paper, computed by its authors their own way.

    Our absorptivity is 1 - albedo under a different solar spectrum and our LWIR a Planck-
    weighted band average rather than their broadband value, and the paper rounds to two
    places; all thirty still agree within 0.012.
    """
    albedo, emissivity = PAPER[sample_of(name)]
    m = library[name]
    assert m.spec.thermal.solar_absorptivity == pytest.approx(1.0 - albedo, abs=0.012)
    assert m.spec.optical.emissivity_per_band["lwir"] == pytest.approx(emissivity, abs=0.012)


@pytest.mark.parametrize("name", slum())
def test_every_curve_keeps_slums_licence(name: str) -> None:
    for suffix in ("sw", "lw"):
        head = (SLUM_DIR / f"{name}_{suffix}.csv").read_text(encoding="utf-8")[:1500]
        assert "source: MEASURED" in head
        assert "Mozilla Public License 2.0" in head


@pytest.mark.parametrize("name", slum())
def test_the_band_slum_did_not_measure_names_its_source(name: str) -> None:
    text = (MATERIAL_DIR / f"{name}.yaml").read_text(encoding="utf-8")
    assert "MWIR" in text and ("UCSB" in text or "data/nk/" in text or "FLIR" in text)
