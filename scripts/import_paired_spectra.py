#!/usr/bin/env python3
"""Library materials from two measurements: USGS for the short-wave, a second source for the long.

    python scripts/import_paired_spectra.py --usgs ASCIIdata_splib07a      # every material below
    python scripts/import_paired_spectra.py --usgs ASCIIdata_splib07a --only brick_red

Roadmap XD.13, second part. No single freely available source measures a brick, a road or a leaf
across all four bands:

* the **USGS Spectral Library v7** (Kokaly et al. 2017, public domain) measured hundreds of real,
  pigmented, weathered samples from 0.35 to 2.5 µm, but its artificial-materials chapter has almost
  no thermal-infrared data (7 of 290 spectra reach past 2.5 µm, none of them a building material);
* the **MODIS UCSB Emissivity Library** [R42] measured directional-hemispherical emissivity from
  3.3 to 14.5 µm with an integrating sphere, but nothing shorter.

So each material here pairs one USGS sample (NIR, SWIR and the solar absorptivity) with one
measurement of the same kind of surface in the thermal infrared: a UCSB spectrum, or for a
plastic the Fresnel emissivity of the polymer's own measured optical constants (Zhang et al.
2020, RefractiveIndex.INFO, CC0). Which sample stands for which is a judgement, and each YAML
says what it paired and why.

**What is committed, and what is not (ADR 0041).** USGS is public domain: its curve goes into
``data/spectra/materials/usgs/<material>.csv`` as measured, with its record title, and a test
re-derives NIR, SWIR and the solar absorptivity from it. The polymer tables are CC0 and go into
``data/nk/``. The UCSB library states no licence, so its curves are **not** committed: this
script fetches them, band-averages them, and writes only the resulting numbers with the file
names they came from -- cite and extract, the roadmap's default for unlicensed spectra.

**The band values are the library's own reduction.** Every number is a Planck-weighted band
average under the nominal top-hat of its band at 300 K (ADR 0010), the same call the library
makes for a spectral file -- so a USGS-derived NIR value here is exactly what the library would
return if the curve were the authored property. Two edges are stated, not hidden: UCSB starts at
3.34 µm, so MWIR is averaged over 3.34-5.0 µm (83 % of the nominal band); and the solar
absorptivity is averaged over the part of 0.35-2.5 µm each USGS curve measured, and each file
says what share of the AM1.5 direct energy that is (about 95 %).

**Emissivity is 1 − R.** Every surface here is opaque at the thickness it is given (a brick, a
road, a leaf layer, a plastic part, deep water), so Kirchhoff per wavelength gives ε = 1 − ρ.

docs/physics-model.md §4.1, §4.5, §5.4, §12.3, §16.2; ADR 0010, ADR 0041; roadmap XD.13
"""

from __future__ import annotations

import argparse
import datetime
import pathlib
import sys
import urllib.parse
import urllib.request

import numpy as np
from numpy.typing import NDArray

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from fetch_nk_tables import RII_RAW  # noqa: E402
from import_material_spectra import REPO, reference_of, write_nk  # noqa: E402

sys.path.insert(0, str(REPO / "src"))
from irsim.config.bands import NOMINAL_RANGES_UM  # noqa: E402
from irsim.materials.library import nominal_response  # noqa: E402
from irsim.materials.spectra import (  # noqa: E402
    PropertySpectrum,
    SpectralCurve,
    load_property_spectrum,
    solar_absorptance,
)
from irsim.radiometry.solar import AM15_DIRECT_FILE, load_solar_spectrum  # noqa: E402
from irsim.radiometry.spectral_response import SpectralResponse  # noqa: E402

UCSB_URL = "https://icess.eri.ucsb.edu/modis/EMIS/images/{}.prn"
UCSB_REF = (
    "MODIS UCSB Emissivity Library (Z. Wan, ICESS, UC Santa Barbara), "
    "https://icess.eri.ucsb.edu/modis/EMIS/html/em.html, directional-hemispherical emissivity "
    "measured with a MIDAC FTIR and an integrating sphere, 3.3-14.5 um [R42]"
)
USGS_REF = (
    "USGS Spectral Library Version 7 (Kokaly et al. 2017, U.S. Geological Survey Data Series "
    "1035, doi:10.3133/ds1035; data doi:10.5066/F7RR1WDJ), splib07a, public domain"
)
HT_A3 = "Incropera, DeWitt, Bergman & Lavine, Fundamentals of Heat and Mass Transfer, Table A.3"

#: USGS covers 0.35-2.5 um; the solar average runs over what it measured.
SOLAR_MEASURED_UM = (0.35, 2.5)
#: The four bands at the library's defaults: nominal top-hat, 300 K, energy form (ADR 0010).
BANDS = ("nir", "swir", "mwir", "lwir")

#: What each class of surface is made of and how it scatters, taken from the library's existing
#: material of that class wherever there is one (``like``), so a measured brick and an estimated
#: concrete differ in their optics and not in a thermal body someone retyped.
CLASSES: dict[str, dict] = {
    "asphalt": {
        "like": "asphalt_dry",
        "thermal": (2200, 920, 0.75, 0.05),
        "treatment": "weathered",
        "roughness": (0.80, 0.80, 0.75, 0.70),
        "angular": (0.05, 4.0),
    },
    "roofing": {
        "like": "asphalt_dry, in a 5 mm layer",
        "thermal": (2200, 920, 0.75, 0.005),
        "treatment": "weathered",
        "roughness": (0.80, 0.80, 0.75, 0.70),
        "angular": (0.05, 4.0),
    },
    "concrete": {
        "like": "concrete",
        "thermal": (2300, 880, 1.4, 0.10),
        "treatment": "weathered",
        "roughness": (0.85, 0.85, 0.80, 0.75),
        "angular": (0.05, 4.0),
    },
    "brick": {
        "like": f"common brick, {HT_A3}",
        "thermal": (1920, 835, 0.72, 0.10),
        "treatment": "as_manufactured",
        "roughness": (0.85, 0.85, 0.80, 0.75),
        "angular": (0.05, 4.0),
    },
    "softwood": {
        "like": f"softwoods (fir, pine), {HT_A3}",
        "thermal": (510, 1380, 0.12, 0.02),
        "treatment": "as_manufactured",
        "roughness": (0.70, 0.70, 0.60, 0.55),
        "angular": (0.05, 4.0),
    },
    "plywood": {
        "like": f"plywood, {HT_A3}",
        "thermal": (545, 1215, 0.12, 0.012),
        "treatment": "as_manufactured",
        "roughness": (0.70, 0.70, 0.60, 0.55),
        "angular": (0.05, 4.0),
    },
    "leaf": {
        "like": "vegetation_leaf",
        "thermal": (700, 3000, 0.30, 0.0003),
        "treatment": "natural",
        "roughness": (0.55, 0.55, 0.45, 0.40),
        "angular": (0.06, 4.0),
    },
    "dry_grass": {
        "like": "vegetation_leaf, dried: ESTIMATED, a cured blade holds less water",
        "thermal": (400, 1800, 0.10, 0.0005),
        "treatment": "natural",
        "roughness": (0.70, 0.70, 0.60, 0.55),
        "angular": (0.06, 4.0),
    },
    "sand": {
        "like": f"sand, {HT_A3}",
        "thermal": (1515, 800, 0.27, 0.10),
        "treatment": "natural",
        "roughness": (0.90, 0.90, 0.85, 0.80),
        "angular": (0.04, 4.0),
    },
    "soil": {
        "like": "soil_dry",
        "thermal": (1500, 800, 0.30, 0.10),
        "treatment": "natural",
        "roughness": (0.90, 0.90, 0.85, 0.80),
        "angular": (0.04, 4.0),
    },
    "seawater": {
        "like": "water",
        "thermal": (1025, 3990, 0.60, 0.10),
        "treatment": "natural",
        "roughness": (0.06, 0.05, 0.04, 0.03),
        "angular": "nk/water.csv",
    },
    "snow": {
        "like": "snow",
        "thermal": (300, 2100, 0.20, 0.10),
        "treatment": "natural",
        "roughness": (0.70, 0.70, 0.60, 0.55),
        "angular": (0.03, 4.0),
    },
    "plastic": {
        "like": "ESTIMATED: a typical value for the polymer; handbooks spread about 15 %",
        "thermal": None,  # per material
        "treatment": "as_manufactured",
        "roughness": (0.25, 0.22, 0.12, 0.08),
        "angular": (0.75, 4.0),
    },
    "painted_metal": {
        "like": "aircraft_aluminium_painted",
        "thermal": (2700, 900, 205, 0.0016),
        "treatment": "painted",
        "roughness": (0.25, 0.20, 0.12, 0.08),
        "angular": (0.75, 4.0),
    },
}

#: One material per entry: the USGS sample (chapter, file stem before the instrument token), the
#: thermal-infrared source -- ``("ucsb", [prn, ...])`` averaged, or ``("nk", rii_path, nk_name)``
#: for a polymer -- its class, and why this pairing stands for the surface named.
PAIRED: dict[str, dict] = {
    "asphalt_road_aged": {
        "usgs": ("ChapterA_ArtificialMaterials", "Asphalt_GDS376_Blck_Road_old"),
        "ir": ("ucsb", ["asphalt"]),
        "class": "asphalt",
        "description": "An old asphalt road surface, worn grey-black",
        "pairing": "an old black road (USGS) with a black asphalt sample (UCSB)",
    },
    "roof_tar_black": {
        "usgs": ("ChapterA_ArtificialMaterials", "Asphalt_Tar_GDS346_Blck_Roof"),
        "ir": ("ucsb", ["asphalt"]),
        "class": "roofing",
        "description": "A black tar roof",
        "pairing": "black roofing tar (USGS) with asphalt, which UCSB describes as bitumen",
    },
    "roof_shingle_dark": {
        "usgs": ("ChapterA_ArtificialMaterials", "Asphalt_Shingle_GDS367_DkGry"),
        "ir": ("ucsb", ["asphalt"]),
        "class": "roofing",
        "description": "Dark grey asphalt roof shingles",
        "pairing": (
            "dark grey shingles (USGS) with asphalt (UCSB): a PROXY in the long wave, since a "
            "shingle's mineral granules are not in the UCSB sample"
        ),
    },
    "brick_red": {
        "usgs": ("ChapterA_ArtificialMaterials", "Brick_GDS350_Dk_Red_Building"),
        "ir": ("ucsb", ["clybrkcm"]),
        "class": "brick",
        "description": "Red clay building brick",
        "pairing": "a dark red building brick (USGS) with a common clay brick (UCSB)",
    },
    "brick_tan": {
        "usgs": ("ChapterA_ArtificialMaterials", "Brick_GDS347_Paving_Tan"),
        "ir": ("ucsb", ["tanbrick"]),
        "class": "brick",
        "description": "Tan paving brick",
        "pairing": "a tan paving brick (USGS) with a tan brick (UCSB)",
    },
    "concrete_pavement": {
        "usgs": ("ChapterA_ArtificialMaterials", "Concrete_GDS375_Lt_Gry_Road"),
        "ir": ("ucsb", ["lfecontl"]),
        "class": "concrete",
        "description": "Light grey concrete road or pavement",
        "pairing": "a light grey concrete road (USGS) with a concrete tile (UCSB)",
    },
    "cinder_block": {
        "usgs": ("ChapterA_ArtificialMaterials", "Cinder_Block_GDS356_Lg_Grey"),
        "ir": ("ucsb", ["masonnat"]),
        "class": "concrete",
        "description": "Grey cinder (concrete) block",
        "pairing": "a grey cinder block (USGS) with a natural masonry cap (UCSB)",
    },
    "wood_pine": {
        "usgs": ("ChapterA_ArtificialMaterials", "Wood_Beam_GDS363_Nw_Pine_2X4"),
        "ir": ("ucsb", ["dougfir"]),
        "class": "softwood",
        "description": "New sawn pine timber",
        "pairing": "a new pine beam (USGS) with Douglas fir lumber (UCSB), both sawn softwood",
    },
    "plywood": {
        "usgs": ("ChapterA_ArtificialMaterials", "Plywood_GDS365_Fresh_Pine"),
        "ir": ("ucsb", ["dougfir"]),
        "class": "plywood",
        "description": "Fresh pine plywood",
        "pairing": "fresh pine plywood (USGS) with Douglas fir lumber (UCSB), both softwood faces",
    },
    "painted_aluminium_green": {
        "usgs": ("ChapterA_ArtificialMaterials", "Painted_Aluminum_GDS333_LgGr"),
        "ir": ("ucsb", ["alumdisk"]),
        "class": "painted_metal",
        "description": "Light green paint on aluminium",
        "pairing": (
            "light green painted aluminium (USGS) with black painted aluminium (UCSB): pigment "
            "colour stays out of the thermal bands (GT.9), the binder sets them"
        ),
    },
    "leaf_maple": {
        "usgs": ("ChapterV_Vegetation", "Maple_Leaves_DW92-1"),
        "ir": ("ucsb", ["mplrdstr"]),
        "class": "leaf",
        "description": "Green maple leaves: a broadleaf tree canopy",
        "pairing": "maple leaves (USGS) with a red-star maple leaf (UCSB)",
    },
    "conifer_needles": {
        "usgs": ("ChapterV_Vegetation", "Lodgepole-Pine_LP-Needles-1"),
        "ir": ("ucsb", ["pinenew"]),
        "class": "leaf",
        "description": "Pine needles: a conifer canopy",
        "pairing": "lodgepole pine needles (USGS) with new pine needles (UCSB)",
    },
    "grass_dry": {
        "usgs": ("ChapterV_Vegetation", "Grass_Golden_Dry_GDS480"),
        "ir": ("ucsb", ["drygras1", "drygras2", "drygras3"]),
        "class": "dry_grass",
        "description": "Golden dry grass, cured in summer",
        "pairing": "golden dry grass (USGS) with three dry-grass sets averaged (UCSB)",
    },
    "sand_beach": {
        "usgs": ("ChapterS_SoilsAndMixtures", "Sand_GrndIsle1_no_oil"),
        "ir": ("ucsb", ["goleta1", "goleta2"]),
        "class": "sand",
        "description": "Dry beach sand",
        "pairing": "Grand Isle beach sand (USGS) with Goleta beach sand, two samples (UCSB)",
    },
    "playa_dry_mud": {
        "usgs": ("ChapterS_SoilsAndMixtures", "Stonewall_Playa_Dry_Mud_2001"),
        "ir": ("ucsb", ["railpwd1", "railpwd2"]),
        "class": "soil",
        "description": "Dry desert playa mud: a flat, bright desert floor",
        "pairing": "Stonewall playa dry mud (USGS) with Railroad Valley playa soil (UCSB)",
    },
    "seawater": {
        "usgs": ("ChapterL_Liquids", "Seawater_Open_Ocean_SW2_lwch"),
        "ir": ("ucsb", ["seawat10"]),
        "class": "seawater",
        "description": "Open-ocean sea water, calm",
        "pairing": "open-ocean water (USGS) with a ten-set seawater average (UCSB)",
    },
    "snow_melting": {
        "usgs": ("ChapterL_Liquids", "Melting_snow_mSnw01a"),
        "ir": ("ucsb", ["snowmam01", "snowmam02"]),
        "class": "snow",
        "description": "Wet, melting snow",
        "pairing": "melting snow (USGS) with Mammoth Lakes ice-snow, two samples (UCSB)",
    },
    "pvc_white": {
        "usgs": ("ChapterA_ArtificialMaterials", "Plastic_PVC_GDS338_White"),
        "ir": ("nk", "(C2H3Cl)n - polyvinyl chloride/nk/Zhang.yml", "pvc"),
        "class": "plastic",
        "thermal": (1380, 1000, 0.17, 0.003),
        "description": "White PVC: pipe, window frames, siding",
        "pairing": (
            "white PVC (USGS) with the Fresnel emissivity of PVC's measured optical constants: "
            "the polymer, not its white pigment, which reflects a little in MWIR as well"
        ),
    },
    "pet_black": {
        "usgs": ("ChapterA_ArtificialMaterials", "Plastic_PETE_GDS382_Black"),
        "ir": ("nk", "(C10H8O4)n - polyethylene terephthalate/nk/Zhang.yml", "pet"),
        "class": "plastic",
        "thermal": (1380, 1100, 0.20, 0.002),
        "description": "Black PET plastic: moulded parts and packaging",
        "pairing": (
            "black PET (USGS) with the Fresnel emissivity of PET's measured optical constants"
        ),
    },
}


# --- reading ----------------------------------------------------------------------------------


def usgs_curve(root: pathlib.Path, chapter: str, stem: str) -> tuple[NDArray, NDArray, str, str]:
    """(wavelength um, reflectance, record title, file name) of one splib07a spectrum."""
    files = sorted((root / chapter).glob(f"splib07a_{stem}_*_*REF.txt"))
    files = [f for f in files if "AVIRIS" not in f.name]
    if len(files) != 1:
        raise FileNotFoundError(f"{stem}: expected one ASD/Beckman spectrum, found {files}")
    path = files[0]
    instrument = path.name[len(f"splib07a_{stem}_") :].split("_")[0]
    prefix = {"ASD": "ASD", "BEC": "BECK", "NIC": "NIC4"}[instrument[:3]]
    (wl_file,) = root.glob(f"splib07a_Wavelengths_{prefix}_*.txt")
    lines = path.read_text(encoding="utf-8").splitlines()
    title = " ".join(lines[0].split())
    values = np.array([float(x) for x in lines[1:] if x.strip()])
    wl = np.array([float(x) for x in wl_file.read_text().splitlines()[1:] if x.strip()])
    if len(wl) != len(values):
        raise ValueError(f"{path.name}: {len(values)} values against {len(wl)} wavelengths")
    good = (values > -1e30) & np.isfinite(values)
    wl, values = wl[good], np.clip(values[good], 0.0, 1.0)
    order = np.argsort(wl)
    wl, values = wl[order], values[order]
    keep = np.concatenate(([True], np.diff(wl) > 0))
    return wl[keep], values[keep], title, path.name


def ucsb_curve(prn: str) -> tuple[NDArray, NDArray, str]:
    """(wavelength um, emissivity, title) of one UCSB file, fetched."""
    url = UCSB_URL.format(prn)
    req = urllib.request.Request(url, headers={"User-Agent": "irsim-import"})
    with urllib.request.urlopen(req, timeout=60) as response:  # noqa: S310 - fixed https URL
        text = response.read().decode("latin-1")
    title = ""
    rows = []
    for line in text.splitlines():
        if line.startswith("#TITLE"):
            title = line.split("=", 1)[1].strip()
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split()
        if len(parts) >= 3:
            rows.append((float(parts[0]), float(parts[2])))
    arr = np.array(sorted(rows))
    keep = np.concatenate(([True], np.diff(arr[:, 0]) > 0))
    return arr[keep, 0], np.clip(arr[keep, 1], 0.0, 1.0), title


# --- reducing ---------------------------------------------------------------------------------


def band_value(lam: NDArray, eps: NDArray, band: str) -> tuple[float, float]:
    """The library's band average of ε(λ); and the fraction of the nominal band it covers."""
    nominal = nominal_response(band)
    lo, hi = nominal.support_um
    c_lo, c_hi = max(lo, float(lam[0])), min(hi, float(lam[-1]))
    if c_hi <= c_lo:
        raise ValueError(f"no data in band {band}")
    response = SpectralResponse(
        wavelength_um=np.array([c_lo, c_hi]),
        response=np.array([1.0, 1.0]),
        source_path=f"<nominal {band} over the measured {c_lo:.3g}-{c_hi:.3g} um>",
        sha256="",
    )
    curve = PropertySpectrum(lam, eps, pathlib.Path("<curve>"))
    return curve.band_effective(response, threshold=0.0), (c_hi - c_lo) / (hi - lo)


def energy_share(sun, range_um: tuple[float, float]) -> float:
    """The fraction of the AM1.5 direct energy in 0.3-4 um that falls inside ``range_um``."""
    lam = sun.wavelength_um
    e = sun.values
    whole = (lam >= 0.3) & (lam <= 4.0)
    part = (lam >= range_um[0]) & (lam <= range_um[1])
    return float(np.trapezoid(e[part], lam[part]) / np.trapezoid(e[whole], lam[whole]))


def fresnel_curve(lam: NDArray, n: NDArray, k: NDArray) -> tuple[NDArray, NDArray]:
    grid = lam[(lam >= 2.5)]
    ni, ki = np.interp(grid, lam, n), np.interp(grid, lam, k)
    return grid, 1.0 - ((ni - 1.0) ** 2 + ki**2) / ((ni + 1.0) ** 2 + ki**2)


# --- writing ----------------------------------------------------------------------------------


def write_usgs(name: str, lam: NDArray, refl: NDArray, title: str, filename: str) -> pathlib.Path:
    out = REPO / "data" / "spectra" / "materials" / "usgs"
    out.mkdir(parents=True, exist_ok=True)
    head = [
        f"# Spectral reflectance of {name.replace('_', ' ')}, {lam[0]:.3g}-{lam[-1]:.3g} um.",
        "#",
        f"# source: MEASURED. {USGS_REF}.",
        f"#   Record: {title} ({filename}); deleted channels dropped, otherwise unmodified.",
        "#   Copied by scripts/import_paired_spectra.py. Directional reflectance of the sample as",
        "#   measured; for an opaque surface the emissivity is 1 - reflectance (Kirchhoff).",
        "# Columns: wavelength in MICROMETRES, reflectance.",
    ]
    rows = [f"{a:.6g},{b:.6g}" for a, b in zip(lam, refl, strict=True)]
    path = out / f"{name}.csv"
    path.write_text("\n".join(head + rows) + "\n", encoding="utf-8")
    return path


def fmt(values) -> str:
    return "{" + ", ".join(f"{b}: {v:.3f}" for b, v in zip(BANDS, values, strict=True)) + "}"


def optical_block(curves: list[tuple[str, str]], values) -> str:
    """The ``optical:`` lines that name the committed curves (AT.34, ADR 0175).

    ``curves`` is ``[(path under data/, quantity), ...]`` in increasing wavelength; ``values`` the
    four computed band values. A band whose whole nominal range a curve covers is read from the
    curve by the library, so its number is **not** written -- writing it would be a second
    authoring the loader refuses. The rest stay as the per-band fill.
    """
    files = [load_property_spectrum(REPO / "data" / rel) for rel, _ in curves]
    curve = SpectralCurve(tuple(files), tuple(q != "emissivity" for _, q in curves))
    keep = [
        (b, v)
        for b, v in zip(BANDS, values, strict=True)
        if not curve.covers_interval(*NOMINAL_RANGES_UM[b])
    ]
    lines = ["    spectral_emissivity:   # the committed measured curves, see the header"]
    for rel, quantity in curves:
        if quantity == "emissivity":
            lines.append(f"      - {rel}")
        else:
            lines.append(f"      - {{file: {rel}, quantity: {quantity}}}   # opaque: 1 - R")
    if keep:
        table = "{" + ", ".join(f"{b}: {v:.3f}" for b, v in keep) + "}"
        lines.append(
            f"    emissivity_per_band: {table}   # computed; only where no curve is committed"
        )
    return "\n".join(lines)


def relink(text: str, curves: list[tuple[str, str]], header: tuple[str, str] | None) -> str:
    """Rewrite a material this script wrote to name its committed curves, changing nothing else.

    The four computed band values are read back from the file, so no source is re-fetched: what
    changes is only *where the library reads NIR and SWIR from* (ADR 0175). ``header`` is the
    one header sentence that described the old form, and its replacement.
    """
    import re

    match = re.search(
        r"^    emissivity_per_band: \{(.*)\}   # computed, see the header$", text, re.M
    )
    if match is None:
        if "spectral_emissivity:" in text:
            return text  # already relinked
        raise ValueError("no computed emissivity_per_band line to relink")
    values = dict(
        (k.strip(), float(v)) for k, v in (item.split(":") for item in match.group(1).split(","))
    )
    block = optical_block(curves, [values[b] for b in BANDS])
    text = text[: match.start()] + block + text[match.end() :]
    if header is not None:
        old, new = header
        if text.count(old) != 1:
            raise ValueError(f"header sentence not found once: {old!r}")
        text = text.replace(old, new)
    return text


PAIRED_HEADER = (
    "# NIR and SWIR are the library's band averages of 1 - R of the USGS curve beside this "
    "material.",
    "# NIR and SWIR are read from the USGS curve beside this material, as 1 - R (ADR 0175).",
)


def build(name: str, usgs_root: pathlib.Path, sun) -> dict:
    spec = PAIRED[name]
    cls = CLASSES[spec["class"]]
    lam, refl, title, filename = usgs_curve(usgs_root, *spec["usgs"])
    write_usgs(name, lam, refl, title, filename)
    eps_short = 1.0 - refl
    nir, _ = band_value(lam, eps_short, "nir")
    swir, _ = band_value(lam, eps_short, "swir")
    solar_range = (
        max(SOLAR_MEASURED_UM[0], float(lam[0])),
        min(SOLAR_MEASURED_UM[1], float(lam[-1])),
    )
    alpha = solar_absorptance(
        PropertySpectrum(lam, eps_short, pathlib.Path(filename)), sun, solar_range
    )
    share = energy_share(sun, solar_range)

    ir = spec["ir"]
    angular = cls["angular"]
    if ir[0] == "ucsb":
        curves = [ucsb_curve(p) for p in ir[1]]
        lo = max(c[0][0] for c in curves)
        hi = min(c[0][-1] for c in curves)
        grid = np.linspace(lo, hi, 2000)
        mean = np.mean([np.interp(grid, c[0], c[1]) for c in curves], axis=0)
        mwir, mwir_cover = band_value(grid, mean, "mwir")
        lwir, _ = band_value(grid, mean, "lwir")
        ir_note = (
            f"MWIR and LWIR: {UCSB_REF}; files {', '.join(p + '.prn' for p in ir[1])} "
            f"({'; '.join(c[2] for c in curves)}), averaged, retrieved "
            f"{datetime.date.today().isoformat()}. MWIR over the measured "
            f"{grid[0]:.2f}-5.0 um ({100 * mwir_cover:.0f} % of the band). Not committed: the "
            "library states no licence, so only these band values are extracted"
        )
    else:
        _, rii_path, nk_name = ir
        url = f"{RII_RAW}/organic/{urllib.parse.quote(rii_path)}"
        with urllib.request.urlopen(url, timeout=60) as response:  # noqa: S310 - fixed https URL
            text = response.read().decode("utf-8")
        nk_spec = {
            "nk": nk_name,
            "kind": (
                "MEASURED: the bulk polymer, unpigmented. For an optically thick part the "
                "normal emissivity is 1 - R from these constants; a pigment changes it "
                "mainly where the polymer is weakly absorbing."
            ),
        }
        _, nl, nn, kk = write_nk(nk_name, nk_spec, url, text, need_um=(2.5, 14.0))
        grid, eps_long = fresnel_curve(nl, nn, kk)
        mwir, _ = band_value(grid, eps_long, "mwir")
        lwir, _ = band_value(grid, eps_long, "lwir")
        ir_note = (
            f"MWIR and LWIR: normal-incidence Fresnel emissivity of data/nk/{nk_name}.csv "
            f"({reference_of(text)}), for an optically thick smooth part"
        )

    thermal = spec.get("thermal") or cls["thermal"]
    rho, cp, k, thickness = thermal
    return {
        "name": name,
        "title": name.replace("_", " ").capitalize(),
        "description": spec["description"],
        "pairing": spec["pairing"],
        "treatment": cls["treatment"],
        "usgs_note": (
            f"NIR, SWIR and solar absorptivity: {USGS_REF}, record '{title}' "
            f"(data/spectra/materials/usgs/{name}.csv)"
        ),
        "ir_note": ir_note,
        "thermal_note": cls["like"],
        "rho": rho,
        "cp": cp,
        "k": k,
        "thickness": thickness,
        "alpha": alpha,
        "solar_lo": solar_range[0],
        "solar_hi": solar_range[1],
        "share": 100.0 * share,
        "eps": (nir, swir, mwir, lwir),
        "roughness": cls["roughness"],
        "angular": angular,
    }


YAML = """\
# {title}: {description}.
#
# Written by scripts/import_paired_spectra.py (roadmap XD.13) -- edit the script, not this file.
# Paired measurements: {pairing}.
# NIR and SWIR are read from the USGS curve beside this material, as 1 - R (ADR 0175).
# The solar absorptivity is its AM1.5 average over the {solar_lo:.4g}-{solar_hi:.4g} um measured
# ({share:.0f} % of the direct energy in 0.3-4 um). MWIR and LWIR come from the second source
# named in `reference`.
schema_version: 2
material:
  name: {name}
  source: literature
  surface_treatment: {treatment}
  reference: "{usgs_note}. {ir_note}. Thermal: {thermal_note}"
  description: "{description}"
  thermal:
    density_kg_m3: {rho}
    specific_heat_j_kgk: {cp}
    conductivity_w_mk: {k}
    thickness_m: {thickness}   # ESTIMATED: a typical part or layer
    solar_absorptivity: {alpha:.3f}   # computed from the USGS curve, not typed
  optical:
{optical}
    roughness_per_band: {roughness}   # ESTIMATED, as the class's existing material
    angular_model: {angular}
"""


def write_yaml(m: dict) -> pathlib.Path:
    if isinstance(m["angular"], str):
        angular = f"{{type: fresnel, n_k_file: {m['angular']}}}"
    else:
        a, p = m["angular"]
        angular = f"{{type: empirical, a: {a}, p: {p}}}   # ESTIMATED, as the class"
    roughness = ", ".join(f"{b}: {v}" for b, v in zip(BANDS, m["roughness"], strict=True))
    plain = {
        k: (v.replace('"', "'") if isinstance(v, str) else v)
        for k, v in m.items()
        if k not in ("eps", "roughness", "angular")
    }
    optical = optical_block([(f"spectra/materials/usgs/{m['name']}.csv", "reflectance")], m["eps"])
    text = YAML.format(**plain, optical=optical, roughness="{" + roughness + "}", angular=angular)
    path = REPO / "configs" / "materials" / f"{m['name']}.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--usgs", help="the unzipped ASCIIdata_splib07a folder")
    parser.add_argument("--only", choices=sorted(PAIRED))
    parser.add_argument(
        "--relink",
        action="store_true",
        help="rewrite the written materials to name their committed curves (AT.34); no fetch",
    )
    args = parser.parse_args()
    names = [args.only] if args.only else list(PAIRED)
    if args.relink:
        for name in names:
            path = REPO / "configs" / "materials" / f"{name}.yaml"
            curves = [(f"spectra/materials/usgs/{name}.csv", "reflectance")]
            path.write_text(relink(path.read_text(encoding="utf-8"), curves, PAIRED_HEADER))
            print(f"relinked {name}")
        return 0
    if args.usgs is None:
        parser.error("--usgs is required unless --relink")
    root = pathlib.Path(args.usgs)
    sun = load_solar_spectrum(REPO / "data" / AM15_DIRECT_FILE)
    for name in names:
        m = build(name, root, sun)
        write_yaml(m)
        e = m["eps"]
        print(
            f"{name:24s} nir {e[0]:.3f} swir {e[1]:.3f} mwir {e[2]:.3f} lwir {e[3]:.3f}  "
            f"alpha {m['alpha']:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
