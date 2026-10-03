"""ECOSTRESS shapes: the last rung, the credit travels with them, the levels stay (XD.14, ADR 0179).

`scripts/import_ecostress.py` writes each material's LWIR curve as an ECOSTRESS sample's measured
*shape* at the library's authored *level*, and relinks the material. These tests hold the
conditions the owner's decision came with -- every committed file carries JPL's credit line and
both citations, and every material that reads one is listed in the importer -- and the physics
the importer promises: no standard-band value moved, all paints share one shape (colour does not
set thermal emissivity), and a camera with its own range sees the sample's structure.

docs/physics-model.md §4.1, §12.3, §16.2; ADR 0041, ADR 0175, ADR 0179
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

from irsim.materials.library import MATERIAL_DIR, MaterialLibrary
from irsim.radiometry.spectral_response import SpectralResponse

REPO = pathlib.Path(__file__).resolve().parents[2]
ECO = REPO / "data" / "spectra" / "materials" / "ecostress"
sys.path.insert(0, str(REPO / "scripts"))
from import_ecostress import CITE, CREDIT, ENTRIES, PAINT  # noqa: E402


def _top_hat(lo: float, hi: float) -> SpectralResponse:
    return SpectralResponse(np.array([lo, hi]), np.array([1.0, 1.0]), f"<{lo}-{hi}>", "")


@pytest.fixture(scope="module")
def library() -> MaterialLibrary:
    return MaterialLibrary.load()


def test_every_material_reading_ecostress_is_listed_in_the_importer() -> None:
    users = {
        p.stem
        for p in MATERIAL_DIR.glob("*.yaml")
        if "spectra/materials/ecostress/" in p.read_text(encoding="utf-8")
    }
    assert users == set(ENTRIES)


@pytest.mark.parametrize("name", sorted(ENTRIES))
def test_the_credit_line_and_citations_travel_with_the_data(name: str) -> None:
    head = (ECO / f"{name}_lw.csv").read_text(encoding="utf-8").split("\n# Columns")[0]
    flat = " ".join(line.lstrip("#").strip() for line in head.splitlines())
    assert CREDIT in flat
    assert CITE in flat
    assert "ECOSTRESS sample" in flat and "MEASURED SHAPE" in flat
    if "PROXY" in ENTRIES[name][2]:
        assert "CLASS PROXY" in flat


@pytest.mark.parametrize("name", sorted(ENTRIES))
def test_the_authored_lwir_level_is_kept_exactly(library, name: str) -> None:  # type: ignore[no-untyped-def]
    props = library[name].band_properties("lwir")
    assert props.emissivity == pytest.approx(ENTRIES[name][1], abs=1e-9)
    assert props.curve_fraction == 1.0
    for band in ("nir", "swir", "mwir"):
        assert library[name].band_properties(band).curve_fraction == 0.0, band


def test_all_paints_share_one_shape(library) -> None:  # type: ignore[no-untyped-def]
    """Colour does not set thermal emissivity: same sample, and paints with the same authored
    level read the same value to any camera."""
    paints = [n for n, e in ENTRIES.items() if e[0] == PAINT]
    assert {"car_paint_black", "car_paint_white", "aircraft_aluminium_painted"} <= set(paints)
    for lo, hi in ((8.0, 12.0), (6.0, 13.0), (10.0, 13.0)):
        values = {
            round(library[n].band_properties("lwir", _top_hat(lo, hi)).emissivity, 9)
            for n in ("car_paint_black", "car_paint_white", "aircraft_aluminium_painted")
        }
        assert len(values) == 1, (lo, hi, values)


def test_a_camera_s_range_now_moves_these_materials(library) -> None:  # type: ignore[no-untyped-def]
    """Concrete's quartz aggregate dips near 9 µm: an 8-12 µm camera reads it lower than a
    10-13 µm one, by more than the float32 table could hide."""
    concrete = library["concrete"]
    window = concrete.band_properties("lwir", _top_hat(8.0, 12.0)).emissivity
    long = concrete.band_properties("lwir", _top_hat(10.0, 13.0)).emissivity
    assert long - window > 0.02
