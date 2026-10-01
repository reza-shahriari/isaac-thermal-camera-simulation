#!/usr/bin/env python3
"""Library materials from the Spectral Library of impervious Urban Materials (SLUM).

    python scripts/import_slum.py                 # fetch SLUM from Zenodo, write every material
    python scripts/import_slum.py --only concrete_new

Roadmap XD.13, third part. SLUM (Kotthaus, Smith, Wooster & Grimmond 2014) measured 74 samples of
London's roads, roofs, walls and pavements **twice on the same sample**: short-wave reflectance
over 0.35-2.5 µm with an ASD spectroradiometer, and long-wave emissivity over 8-14 µm with a
portable FTIR, calibrated against blackbodies. So unlike ``import_paired_spectra.py`` there is no
pairing judgement for NIR, SWIR, LWIR and the solar absorptivity: one sample, two instruments.

**MWIR is the one band SLUM did not measure.** Each material takes it from the closest measured
source, and says which: a UCSB spectrum of the same class of surface (asphalt, concrete tile,
clay brick, clay tile, sandstone), the Fresnel emissivity of PVC's measured optical constants
(``data/nk/pvc.csv``), or, where neither exists, the FLIR emissivity table's 2-5 µm value for the
class (a compilation whose own note calls its values "recommendations only"; the entry names the
row and FLIR's reference number).

**Licence (ADR 0041).** SLUM is released on Zenodo (doi:10.5281/zenodo.4263842) under the Mozilla
Public License 2.0, which allows redistribution with the licence kept. The two curves each
material uses are committed under ``data/spectra/materials/slum/``, and each file keeps SLUM's
licence notice; the tests re-derive every SLUM-sourced value from them.

**Stated edges.** The long-wave curve starts at 8.0 µm, so LWIR covers 8.0-13.5 µm (92 % of the
nominal band); the solar absorptivity covers the 0.35-2.5 µm measured.

docs/physics-model.md §4.1, §5.4, §12.3; ADR 0010, ADR 0041; roadmap XD.13
"""

from __future__ import annotations

import argparse
import csv
import io
import pathlib
import sys
import urllib.request

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from import_paired_spectra import (  # noqa: E402
    BANDS,
    CLASSES,
    HT_A3,
    REPO,
    UCSB_REF,
    band_value,
    energy_share,
    fresnel_curve,
    optical_block,
    relink,
    ucsb_curve,
)

sys.path.insert(0, str(REPO / "src"))
from irsim.materials.nk import load_nk_table  # noqa: E402
from irsim.materials.spectra import PropertySpectrum, solar_absorptance  # noqa: E402
from irsim.radiometry.solar import AM15_DIRECT_FILE, load_solar_spectrum  # noqa: E402

ZENODO = "https://zenodo.org/api/records/4263842/files/{}/content"
SLUM_REF = (
    "SLUM, the Spectral Library of impervious Urban Materials: S. Kotthaus, T. E. L. Smith, "
    "M. J. Wooster, C. S. B. Grimmond (2014), Derivation of an urban materials spectral library "
    "through emittance and reflectance spectroscopy, ISPRS J. Photogramm. Remote Sens. 94, "
    "194-212, doi:10.1016/j.isprsjprs.2014.05.005; data doi:10.5281/zenodo.4263842 (MPL-2.0)"
)
FLIR_REF = "FLIR Systems, Emissivity tables, FLIR E95 user manual (T810190), section 37"
FLIR_REFS = {
    5: "Jones, Smith, Probert: External thermography of buildings, Proc. SPIE 110 (1977)",
    9: "Ohman, C.: Emittance measurements using AGEMA E-Box, technical report, AGEMA (1999)",
}

#: Thermal bodies for the classes SLUM adds to import_paired_spectra's.
SLUM_CLASSES: dict[str, dict] = {
    **CLASSES,
    "roof_tile": {
        "like": f"common brick, {HT_A3}, as a 12 mm tile",
        "thermal": (1920, 835, 0.72, 0.012),
        "treatment": "as_manufactured",
        "roughness": (0.80, 0.80, 0.75, 0.70),
        "angular": (0.05, 4.0),
    },
    "cement_tile": {
        "like": "concrete, as a 12 mm tile",
        "thermal": (2300, 880, 1.4, 0.012),
        "treatment": "as_manufactured",
        "roughness": (0.80, 0.80, 0.75, 0.70),
        "angular": (0.05, 4.0),
    },
    "slate": {
        "like": "ESTIMATED: a dense stone, between granite and sandstone in Table A.3, 5 mm",
        "thermal": (2700, 760, 2.0, 0.005),
        "treatment": "weathered",
        "roughness": (0.75, 0.75, 0.70, 0.65),
        "angular": (0.05, 4.0),
    },
    "stone_sand": {
        "like": f"sandstone, {HT_A3}",
        "thermal": (2160, 710, 2.90, 0.05),
        "treatment": "weathered",
        "roughness": (0.85, 0.85, 0.80, 0.75),
        "angular": (0.05, 4.0),
    },
    "stone_granite": {
        "like": f"granite, {HT_A3}",
        "thermal": (2630, 775, 2.79, 0.05),
        "treatment": "weathered",
        "roughness": (0.80, 0.80, 0.75, 0.70),
        "angular": (0.05, 4.0),
    },
    "painted_sheet": {
        "like": "plain carbon steel, Incropera Table A.1, a 1 mm roofing sheet",
        "thermal": (7854, 434, 60.5, 0.001),
        "treatment": "painted",
        "roughness": (0.25, 0.20, 0.12, 0.08),
        "angular": (0.75, 4.0),
    },
    "aluminium_sheet": {
        "like": "pure aluminium, Incropera Table A.1, a 1 mm sheet",
        "thermal": (2702, 903, 237, 0.001),
        "treatment": "weathered",
        "roughness": (0.60, 0.55, 0.45, 0.40),
        "angular": (0.15, 5.0),
    },
    "iron_sheet": {
        "like": "pure iron, Incropera Table A.1, a 3 mm plate",
        "thermal": (7870, 447, 80.2, 0.003),
        "treatment": "weathered",
        "roughness": (0.70, 0.65, 0.55, 0.50),
        "angular": (0.15, 5.0),
    },
    "pvc_membrane": {
        "like": "PVC, ESTIMATED as pvc_white's, a 1.5 mm membrane",
        "thermal": (1380, 1000, 0.17, 0.0015),
        "treatment": "as_manufactured",
        "roughness": (0.30, 0.28, 0.18, 0.12),
        "angular": (0.75, 4.0),
    },
}

#: MWIR sources. ``("ucsb", [prn])``, ``("nk", name)`` or ``("flir", value, row, reference)``.
UCSB_ASPHALT = ("ucsb", ["asphalt"])
UCSB_CONCRETE = ("ucsb", ["lfecontl"])
UCSB_BRICK = ("ucsb", ["clybrkcm"])
UCSB_CLAY_TILE = ("ucsb", ["clylintl"])
FLIR_PAINT = ("flir", 0.92, "Paint, 8 different colors, SW 0.88-0.96", 9)

#: One SLUM sample per material: its ID, class, MWIR source and what it is.
SLUM: dict[str, dict] = {
    "asphalt_road_stone": (
        "A001",
        "asphalt",
        UCSB_ASPHALT,
        "Weathered road asphalt with stone aggregate, black and grey",
    ),
    "tarmac_black": ("A008", "asphalt", UCSB_ASPHALT, "Weathered black tarmac"),
    "roofing_felt_grey": (
        "A007",
        "roofing",
        UCSB_ASPHALT,
        "New grey tarmac roofing paper (roofing felt)",
    ),
    "cement_weathered": ("C001", "concrete", UCSB_CONCRETE, "Weathered grey-ochre cement"),
    "concrete_new": ("C002", "concrete", UCSB_CONCRETE, "New grey-white concrete"),
    "concrete_weathered": ("C004", "concrete", UCSB_CONCRETE, "Weathered grey concrete"),
    "concrete_white": ("C006", "concrete", UCSB_CONCRETE, "Weathered white concrete"),
    "cement_brick_yellow": ("B001", "concrete", UCSB_CONCRETE, "New yellow cement brick"),
    "cement_brick_black": ("B003", "concrete", UCSB_CONCRETE, "New black cement brick"),
    "clay_brick_light_red": ("B008", "brick", UCSB_BRICK, "New light red clay brick"),
    "clay_brick_weathered": ("B013", "brick", UCSB_BRICK, "Weathered red clay brick"),
    "clay_brick_painted": (
        "B012",
        "brick",
        UCSB_BRICK,
        "Weathered clay brick, painted red and white",
    ),
    "slate_roof": (
        "L001d",
        "slate",
        ("ucsb", ["flagbuck"]),
        (
            "Weathered black slate roof shingle, upper face (with moss). This is the paper's "
            "L001u: the released CSV labels the two faces the other way round, and its L001d "
            "column reproduces the paper's L001u row (albedo 0.09, emissivity 0.90)"
        ),
    ),
    "fibre_cement_shingle": (
        "L002",
        "cement_tile",
        UCSB_CONCRETE,
        "Weathered black fibre-cement roof shingle",
    ),
    "roof_tile_ceramic_red": ("R001", "roof_tile", UCSB_CLAY_TILE, "New red ceramic roof tile"),
    "roof_tile_ceramic_black": ("R007", "roof_tile", UCSB_CLAY_TILE, "New black ceramic roof tile"),
    "roof_tile_ceramic_weathered": (
        "R010",
        "roof_tile",
        UCSB_CLAY_TILE,
        "Weathered red ceramic roof tile",
    ),
    "roof_tile_cement_grey": (
        "R006",
        "cement_tile",
        UCSB_CONCRETE,
        "New slate-grey cement roof tile",
    ),
    "roof_tile_cement_red": (
        "R008",
        "cement_tile",
        UCSB_CONCRETE,
        "New dull rustic-red cement roof tile",
    ),
    "metal_sheet_painted_green": (
        "Z003",
        "painted_sheet",
        FLIR_PAINT,
        "New green-painted metal sheet",
    ),
    "metal_sheet_painted_patina": (
        "Z004",
        "painted_sheet",
        FLIR_PAINT,
        "New metal sheet painted copper-patina green",
    ),
    "metal_sheet_painted_grey": (
        "Z005",
        "painted_sheet",
        FLIR_PAINT,
        "New slate-grey painted metal sheet",
    ),
    "aluminium_weathered": (
        "Z006",
        "aluminium_sheet",
        ("flir", 0.885, "Aluminum, weathered, heavily, SW 0.83-0.94", 5),
        "Weathered grey aluminium sheet",
    ),
    "iron_weathered": (
        "Z008",
        "iron_sheet",
        ("flir", 0.96, "Iron and steel, rusted, heavily, SW 0.96", 5),
        "Weathered black iron",
    ),
    "pvc_roofing_dark_grey": (
        "V001",
        "pvc_membrane",
        ("nk", "pvc"),
        "New lead-grey PVC roofing membrane",
    ),
    "pvc_roofing_light_grey": (
        "V002",
        "pvc_membrane",
        ("nk", "pvc"),
        "New light grey PVC roofing membrane",
    ),
    "pvc_roofing_blue": (
        "V004",
        "pvc_membrane",
        ("nk", "pvc"),
        "New azure-blue PVC roofing membrane",
    ),
    "sandstone_beige": ("S001", "stone_sand", ("ucsb", ["slisbsnd"]), "Used beige sandstone"),
    "granite_rough": (
        "G001",
        "stone_granite",
        ("flir", 0.96, "Granite, rough, 4 different samples, SW 0.95-0.97", 9),
        "New rough white-and-black granite",
    ),
    "granite_weathered": (
        "G004",
        "stone_granite",
        ("flir", 0.96, "Granite, rough, 4 different samples, SW 0.95-0.97", 9),
        "Weathered white, red and black granite",
    ),
}

LICENCE = (
    "#   Licence: Mozilla Public License 2.0, as SLUM is released; this file is a subset of it "
    "and stays under MPL-2.0 (https://mozilla.org/MPL/2.0/)."
)


def fetch(name: str, cache: pathlib.Path | None) -> str:
    if cache is not None and (cache / name).is_file():
        return (cache / name).read_text(encoding="utf-8")
    with urllib.request.urlopen(ZENODO.format(name), timeout=120) as response:  # noqa: S310
        return response.read().decode("utf-8")


def columns(text: str) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    rows = list(csv.reader(io.StringIO(text)))
    head = rows[0]
    data = np.array([[float(x) for x in r] for r in rows[1:] if r])
    return data[:, 0], {head[i]: data[:, i] for i in range(1, len(head))}


def write_curve(name: str, suffix: str, lam, values, what: str, sample: str) -> pathlib.Path:
    out = REPO / "data" / "spectra" / "materials" / "slum"
    out.mkdir(parents=True, exist_ok=True)
    head = [
        f"# {what} of {name.replace('_', ' ')}, {lam[0]:.4g}-{lam[-1]:.4g} um.",
        "#",
        f"# source: MEASURED. {SLUM_REF}. Sample {sample}.",
        "#   Copied by scripts/import_slum.py; wavelengths converted to micrometres and "
        "reflectance from percent to a fraction, otherwise unmodified.",
        LICENCE,
        "# Columns: wavelength in MICROMETRES, value.",
    ]
    rows = [f"{a:.6g},{b:.6g}" for a, b in zip(lam, values, strict=True)]
    path = out / f"{name}_{suffix}.csv"
    path.write_text("\n".join(head + rows) + "\n", encoding="utf-8")
    return path


def mwir_of(source) -> tuple[float, str]:
    if source[0] == "ucsb":
        curves = [ucsb_curve(p) for p in source[1]]
        lo = max(c[0][0] for c in curves)
        hi = min(c[0][-1] for c in curves)
        grid = np.linspace(lo, hi, 2000)
        mean = np.mean([np.interp(grid, c[0], c[1]) for c in curves], axis=0)
        value, cover = band_value(grid, mean, "mwir")
        titles = "; ".join(c[2] for c in curves)
        return value, (
            f"MWIR from the same class of surface, not this sample: {UCSB_REF}; "
            f"{', '.join(p + '.prn' for p in source[1])} ({titles}), over "
            f"{grid[0]:.2f}-5.0 um ({100 * cover:.0f} % of the band); cited, not committed"
        )
    if source[0] == "nk":
        table = load_nk_table(source[1])
        lam, eps = fresnel_curve(table.wavelength_um, table.n, table.k)
        return band_value(lam, eps, "mwir")[0], (
            f"MWIR: normal Fresnel emissivity of data/nk/{source[1]}.csv, the polymer's measured "
            "optical constants"
        )
    _, value, row, ref = source
    return value, (
        f"MWIR from the class, not this sample: {FLIR_REF}, row '{row}' (the middle of the "
        f"range), FLIR reference {ref}: {FLIR_REFS[ref]}. FLIR calls its values recommendations "
        "only"
    )


YAML = """\
# {title}: {description}.
#
# Written by scripts/import_slum.py (roadmap XD.13) -- edit the script, not this file.
# SLUM sample {sample}: NIR, SWIR, LWIR and the solar absorptivity are measured on this one sample
# (short-wave reflectance and long-wave emissivity, beside this material under
# data/spectra/materials/slum/, one curve in `optical`). Below 8.0 um LWIR uses its band value; the
# solar absorptivity covers {solar_lo:.4g}-{solar_hi:.4g} um ({share:.0f} % of the AM1.5 direct
# energy in 0.3-4 um). MWIR, which SLUM did not measure, comes from the source in `reference`.
schema_version: 2
material:
  name: {name}
  source: literature
  surface_treatment: {treatment}
  reference: "{slum_note}. {mwir_note}. Thermal: {thermal_note}"
  description: "{description}"
  thermal:
    density_kg_m3: {rho}
    specific_heat_j_kgk: {cp}
    conductivity_w_mk: {k}
    thickness_m: {thickness}   # ESTIMATED: a typical part or layer
    solar_absorptivity: {alpha:.3f}   # computed from the SLUM curve, not typed
  optical:
{optical}
    roughness_per_band: {roughness}   # ESTIMATED, as the class
    angular_model: {{type: empirical, a: {a}, p: {p}}}   # ESTIMATED, as the class
"""


SLUM_HEADER = (
    "# data/spectra/materials/slum/). LWIR covers the 8.0-13.5 um measured (92 % of the band); the",
    "# data/spectra/materials/slum/, one curve in `optical`). Below 8.0 um LWIR uses its band "
    "value; the",
)


def curves_of(name: str) -> list[tuple[str, str]]:
    """The two committed SLUM curves as one segmented curve (ADR 0175): SW as 1 - R, then LW."""
    return [
        (f"spectra/materials/slum/{name}_sw.csv", "reflectance"),
        (f"spectra/materials/slum/{name}_lw.csv", "emissivity"),
    ]


def build(name: str, sw: tuple, lw: tuple, sun) -> str:
    sample, cls_name, mwir_source, description = SLUM[name]
    cls = SLUM_CLASSES[cls_name]
    lam_sw = sw[0] / 1000.0
    refl = np.clip(sw[1][sample] / 100.0, 0.0, 1.0)
    lam_lw, eps_lw = lw[0], np.clip(lw[1][sample], 0.0, 1.0)
    write_curve(name, "sw", lam_sw, refl, "Short-wave reflectance (fraction)", sample)
    write_curve(name, "lw", lam_lw, eps_lw, "Long-wave emissivity", sample)
    eps_sw = 1.0 - refl
    nir = band_value(lam_sw, eps_sw, "nir")[0]
    swir = band_value(lam_sw, eps_sw, "swir")[0]
    lwir = band_value(lam_lw, eps_lw, "lwir")[0]
    mwir, mwir_note = mwir_of(mwir_source)
    solar_range = (max(0.35, float(lam_sw[0])), min(2.5, float(lam_sw[-1])))
    alpha = solar_absorptance(
        PropertySpectrum(lam_sw, eps_sw, pathlib.Path(name)), sun, solar_range
    )
    rho, cp, k, thickness = cls["thermal"]
    a, p = cls["angular"]
    roughness = ", ".join(f"{b}: {v}" for b, v in zip(BANDS, cls["roughness"], strict=True))
    text = YAML.format(
        title=name.replace("_", " ").capitalize(),
        description=description,
        sample=sample,
        solar_lo=solar_range[0],
        solar_hi=solar_range[1],
        share=100.0 * energy_share(sun, solar_range),
        name=name,
        treatment=cls["treatment"],
        slum_note=(
            f"NIR, SWIR, LWIR and solar absorptivity: {SLUM_REF}, sample {sample} "
            f"(data/spectra/materials/slum/{name}_sw.csv and _lw.csv)"
        ),
        mwir_note=mwir_note.replace('"', "'"),
        thermal_note=cls["like"],
        rho=rho,
        cp=cp,
        k=k,
        thickness=thickness,
        alpha=alpha,
        optical=optical_block(curves_of(name), (nir, swir, mwir, lwir)),
        roughness="{" + roughness + "}",
        a=a,
        p=p,
    )
    (REPO / "configs" / "materials" / f"{name}.yaml").write_text(text, encoding="utf-8")
    return (
        f"{name:28s} {sample:6s} nir {nir:.3f} swir {swir:.3f} mwir {mwir:.3f} lwir {lwir:.3f}"
        f"  alpha {alpha:.3f}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=sorted(SLUM))
    parser.add_argument("--cache", type=pathlib.Path, help="a folder already holding the CSVs")
    parser.add_argument(
        "--relink",
        action="store_true",
        help="rewrite the written materials to name their committed curves (AT.34); no fetch",
    )
    args = parser.parse_args()
    if args.relink:
        for name in [args.only] if args.only else list(SLUM):
            path = REPO / "configs" / "materials" / f"{name}.yaml"
            path.write_text(relink(path.read_text(encoding="utf-8"), curves_of(name), SLUM_HEADER))
            print(f"relinked {name}")
        return 0
    sw = columns(fetch("LUMA_SLUM_SW.csv", args.cache))
    lw = columns(fetch("LUMA_SLUM_IR.csv", args.cache))
    sun = load_solar_spectrum(REPO / "data" / AM15_DIRECT_FILE)
    for name in [args.only] if args.only else list(SLUM):
        print(build(name, sw, lw, sun))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
