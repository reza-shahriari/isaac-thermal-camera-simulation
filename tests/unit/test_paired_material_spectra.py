"""XD.13 -- materials paired from a USGS short-wave curve and a long-wave measurement.

`scripts/import_paired_spectra.py` writes, for each material, the USGS reflectance curve it used
(public domain, committed) and band values computed from it and from a second source. These tests
re-derive what can be re-derived from checked-in data and hold the rest to physics that every one
of these surfaces must show whatever its source:

* NIR, SWIR and the solar absorptivity equal the library's own averages of 1 - R of the committed
  curve, and a polymer's MWIR/LWIR equal the Fresnel average of its committed n/k table;
* a leaf has the red edge (dark in the long wave, bright in the near infrared);
* snow is bright in NIR and black in SWIR, where ice absorbs;
* quartz sand is the least emissive natural surface in LWIR (the Si-O reststrahlen band);
* water is dark in every band;
* an unlicensed curve (UCSB) is cited, never committed.

docs/physics-model.md §4.1, §5.4, §12.3; ADR 0010, ADR 0041; roadmap XD.13
"""

from __future__ import annotations

import pathlib
import re
import sys

import numpy as np
import pytest

from irsim.materials.library import MATERIAL_DIR, MaterialLibrary, nominal_response
from irsim.materials.nk import load_nk_table
from irsim.materials.spectra import PropertySpectrum, load_property_spectrum, solar_absorptance
from irsim.radiometry.solar import AM15_DIRECT_FILE, load_solar_spectrum, solar_data_dir

REPO = pathlib.Path(__file__).resolve().parents[2]
WRITER = "scripts/import_paired_spectra.py"
USGS_DIR = REPO / "data" / "spectra" / "materials" / "usgs"


def paired() -> list[str]:
    return sorted(
        p.stem for p in MATERIAL_DIR.glob("*.yaml") if WRITER in p.read_text(encoding="utf-8")
    )


@pytest.fixture(scope="module")
def library() -> MaterialLibrary:
    return MaterialLibrary.load()


@pytest.fixture(scope="module")
def sun():
    return load_solar_spectrum(solar_data_dir().parent.parent / AM15_DIRECT_FILE)


def _eps_curve(name: str) -> PropertySpectrum:
    refl = load_property_spectrum(USGS_DIR / f"{name}.csv")
    return PropertySpectrum(refl.wavelength_um, 1.0 - refl.values, refl.path)


def _band(curve: PropertySpectrum, band: str) -> float:
    """The importer's reduction, restated: the nominal top-hat over the part the curve covers."""
    sys.path.insert(0, str(REPO / "scripts"))
    from import_paired_spectra import band_value

    return band_value(curve.wavelength_um, curve.values, band)[0]


def test_the_importer_has_written_materials() -> None:
    assert len(paired()) >= 15


@pytest.mark.parametrize("name", paired())
def test_the_short_wave_is_the_committed_usgs_curve(library, name: str) -> None:
    curve = _eps_curve(name)
    m = library[name]
    for band in ("nir", "swir"):
        props = m.band_properties(band)
        assert props.emissivity == pytest.approx(_band(curve, band), abs=6e-4), band
        # AT.34 (ADR 0175): read from the committed curve, no typed number beside it.
        assert props.curve_fraction == 1.0 and band not in (
            m.spec.optical.emissivity_per_band or {}
        )
    # And the nominal NIR/SWIR bands lie wholly inside the measurement: nothing extrapolated.
    for band in ("nir", "swir"):
        lo, hi = nominal_response(band).support_um
        assert curve.wavelength_um[0] <= lo and curve.wavelength_um[-1] >= hi


@pytest.mark.parametrize("name", paired())
def test_the_solar_absorptivity_is_the_committed_curve_under_the_sun(
    library, sun, name: str
) -> None:
    curve = _eps_curve(name)
    text = (MATERIAL_DIR / f"{name}.yaml").read_text(encoding="utf-8")
    lo, hi = (float(x) for x in re.search(r"over the ([0-9.]+)-([0-9.]+) um", text).groups())
    expected = solar_absorptance(curve, sun, (lo, hi))
    assert library[name].spec.thermal.solar_absorptivity == pytest.approx(expected, abs=6e-4)


@pytest.mark.parametrize("name", paired())
def test_every_usgs_curve_says_it_is_measured_and_public_domain(name: str) -> None:
    head = (USGS_DIR / f"{name}.csv").read_text(encoding="utf-8").split("\n# Columns")[0]
    assert "source: MEASURED" in head
    assert "public domain" in head
    assert "Record=" in head


@pytest.mark.parametrize(
    "name", [n for n in paired() if "data/nk/" in (MATERIAL_DIR / f"{n}.yaml").read_text()]
)
def test_a_plastics_long_wave_is_the_fresnel_average_of_its_table(library, name: str) -> None:
    text = (MATERIAL_DIR / f"{name}.yaml").read_text(encoding="utf-8")
    nk_name = re.search(r"data/nk/([a-z0-9_]+)\.csv", text).group(1)
    table = load_nk_table(nk_name)
    lam = table.wavelength_um[table.wavelength_um >= 2.5]
    n, k = table.at(lam)
    eps = 1.0 - ((n - 1.0) ** 2 + k**2) / ((n + 1.0) ** 2 + k**2)
    curve = PropertySpectrum(lam, eps, table.path)
    authored = library[name].spec.optical.emissivity_per_band
    for band in ("mwir", "lwir"):
        assert authored[band] == pytest.approx(_band(curve, band), abs=6e-4), band


def test_no_unlicensed_curve_is_committed() -> None:
    """UCSB states no licence: its numbers may be cited and extracted, its curves not copied."""
    for path in (REPO / "data").rglob("*.csv"):
        head = path.read_text(encoding="utf-8", errors="ignore")[:4000]
        assert "icess" not in head.lower(), path
    for name in paired():
        text = (MATERIAL_DIR / f"{name}.yaml").read_text(encoding="utf-8")
        if ".prn" in text:
            assert "Not committed" in text


# --- physics every one of these surfaces must show ------------------------------------------


def _eps(library, name: str) -> dict[str, float]:
    m = library[name]
    return {b: m.band_properties(b).emissivity for b in ("nir", "swir", "mwir", "lwir")}


@pytest.mark.parametrize("name", ["leaf_maple", "conifer_needles"])
def test_a_leaf_has_the_red_edge(library, name: str) -> None:
    """Chlorophyll-free mesophyll scatters the near infrared; water makes the thermal black."""
    e = _eps(library, name)
    assert e["nir"] < 0.5
    assert e["lwir"] > 0.95
    assert e["nir"] < e["swir"] < e["lwir"]


def test_snow_is_bright_in_the_near_infrared_and_black_in_the_short_wave(library) -> None:
    """Ice absorbs weakly below 1 um and strongly by 1.5 um: snow is white, then black."""
    e = _eps(library, "snow_melting")
    assert e["nir"] < 0.6
    assert e["swir"] > 0.9
    assert library["snow_melting"].spec.thermal.solar_absorptivity < 0.5


def test_quartz_sand_is_the_least_emissive_natural_surface_in_the_long_wave(library) -> None:
    """The Si-O reststrahlen band (8-9.5 um) reflects: sand sits below every other natural one."""
    natural = [n for n in paired() if library[n].spec.surface_treatment == "natural"]
    sand = _eps(library, "sand_beach")["lwir"]
    others = [_eps(library, n)["lwir"] for n in natural if n != "sand_beach"]
    assert sand < min(others)
    assert sand < 0.92


def test_water_is_dark_in_every_band(library) -> None:
    e = _eps(library, "seawater")
    assert min(e.values()) > 0.96


@pytest.mark.parametrize("name", paired())
def test_no_paired_surface_is_a_mirror(library, name: str) -> None:
    """Every one is a dielectric: rough, organic, mineral or painted. None is a bare metal."""
    assert library[name].band_properties("lwir").emissivity > 0.85


def test_the_values_are_the_measurements_and_not_the_estimates_they_sit_beside(library) -> None:
    """The measured old road differs from the estimated `asphalt_dry` in the band where the
    estimate had least to go on: NIR, where a worn road reflects about a seventh."""
    measured = _eps(library, "asphalt_road_aged")
    estimated = _eps(library, "asphalt_dry")
    assert abs(measured["nir"] - estimated["nir"]) > 0.03
    assert np.isclose(measured["lwir"], 0.972, atol=0.01)
