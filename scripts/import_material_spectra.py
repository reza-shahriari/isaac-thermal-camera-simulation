#!/usr/bin/env python3
"""Build library materials from published spectral data (roadmap XD.13).

    python scripts/import_material_spectra.py metals            # every metal below
    python scripts/import_material_spectra.py metals --only copper_polished

Each material in the library states its emissivity; until now all but one typed it per band, most of
it ESTIMATED. XD.13 asks for the opposite: a four-band source, imported by a script, with every
per-band value re-derived by the library's own band average rather than retyped. This script is
that importer. It writes three files per material and the provenance of each goes in the file:

* ``data/nk/<metal>.csv`` -- the complex refractive index, as published, with its ``# source:``
  block (ADR 0041, the same header ``scripts/fetch_nk_tables.py`` writes);
* ``data/spectra/materials/<material>.csv`` -- the spectral emissivity it implies;
* ``configs/materials/<material>.yaml`` -- the material, pointing at both, with its solar
  absorptivity **computed** from the curve (``irsim.materials.spectra.solar_absorptance``).

**Metals** come from their optical constants, because for a clean polished metal that is the
physics, not an approximation of it: the surface is an optically smooth interface, and at normal
incidence ε(λ) = 1 − |(ñ − 1)/(ñ + 1)|² with ñ = n + ik (§4.2 Level A). The same table then gives
the angular shape, through ``angular_model: fresnel``, so magnitude and shape agree by
construction. What it cannot give is a *real* surface: every metal in service is rougher and
more oxidised than a polished laboratory sample, and both raise ε (§4.5, [R47]). So these are
the **polished** states, the lower bound of each metal, and they are named so. An oxidised,
painted or weathered state is a different material with a different source.

All data comes from the RefractiveIndex.INFO database (public domain, CC0 1.0), and each file
quotes the paper it came from. Thermal properties are handbook values at 300 K: Incropera,
DeWitt, Bergman and Lavine, *Fundamentals of Heat and Mass Transfer*, Table A.1 (``[HT-A1]``).

docs/physics-model.md §4.2, §4.5, §5.4, §12.3; ADR 0041; roadmap XD.13
"""

from __future__ import annotations

import argparse
import datetime
import html
import pathlib
import re
import sys
import urllib.request

import numpy as np
from numpy.typing import NDArray

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from fetch_nk_tables import HEADER, RII_RAW, normal_emissivity, parse_rii_nk  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[1]
HT_A1 = (
    "Incropera, DeWitt, Bergman & Lavine, Fundamentals of Heat and Mass Transfer, Table A.1 (300 K)"
)

#: The range every table is cut to: the solar band (0.3-4 um) and every camera band up to the
#: 14 um edge of the widest LWIR response in data/spectra/responses/, with margin.
TRIM_UM = (0.3, 16.0)

#: One polished metal per entry. ``element``/``dataset`` name the RefractiveIndex.INFO file;
#: ``kind`` goes into the n/k file's source block (ADR 0041's three kinds). ``thermal`` is the
#: handbook body the surface sits on, with its thickness as a typical part.
METALS: dict[str, dict] = {
    "copper_polished": {
        "element": "Cu",
        "dataset": "Querry",
        "nk": "copper",
        "kind": "MEASURED: a copper ingot, polished.",
        "description": "Bare polished copper: bus bars, motor windings seen through a gap, pipe",
        "thermal": {"density_kg_m3": 8933, "specific_heat_j_kgk": 385, "conductivity_w_mk": 401},
        "thermal_note": f"pure copper, {HT_A1}",
        "thickness_m": 0.001,
    },
    "steel_polished": {
        "element": "Fe",
        "dataset": "Querry",
        "nk": "iron",
        "kind": (
            "MEASURED (iron) used as a PROXY for carbon steel: plain carbon steel is about 99 % "
            "iron, and the free electrons that set a metal's infrared reflectance are the iron's."
        ),
        "description": "Bright polished carbon steel: machined shafts, fresh-cut edges, tooling",
        "thermal": {"density_kg_m3": 7854, "specific_heat_j_kgk": 434, "conductivity_w_mk": 60.5},
        "thermal_note": f"plain carbon steel, {HT_A1}",
        "thickness_m": 0.002,
    },
    "chrome_plated": {
        "element": "Cr",
        "dataset": "Rakic-LD",
        "nk": "chromium",
        "kind": (
            "MODELLED: a Lorentz-Drude fit to measured chromium data from several sources, as "
            "published; not a single measurement."
        ),
        "description": (
            "Chrome plating on steel: bumpers, exhaust tips, motorcycle trim. The plating is "
            "micrometres thick, so the heat is the steel's and the optics are the chromium's"
        ),
        "thermal": {"density_kg_m3": 7854, "specific_heat_j_kgk": 434, "conductivity_w_mk": 60.5},
        "thermal_note": f"the steel under the plating: plain carbon steel, {HT_A1}",
        "thickness_m": 0.0015,
    },
    "titanium_polished": {
        "element": "Ti",
        "dataset": "Rakic-LD",
        "nk": "titanium",
        "kind": (
            "MODELLED: a Lorentz-Drude fit to measured titanium data from several sources, as "
            "published; not a single measurement."
        ),
        "description": "Polished titanium: airframe fittings, exhaust parts, fasteners",
        "thermal": {"density_kg_m3": 4500, "specific_heat_j_kgk": 522, "conductivity_w_mk": 21.9},
        "thermal_note": f"pure titanium, {HT_A1}",
        "thickness_m": 0.001,
    },
    "gold_polished": {
        "element": "Au",
        "dataset": "Olmon-ev",
        "nk": "gold",
        "kind": "MEASURED: an evaporated gold film, thick enough to be opaque.",
        "description": "Gold: plated contacts and gold-coated reflectors, the lowest-emissivity "
        "metal in the library",
        "thermal": {"density_kg_m3": 19300, "specific_heat_j_kgk": 129, "conductivity_w_mk": 317},
        "thermal_note": f"pure gold, {HT_A1}",
        "thickness_m": 0.0005,
    },
    "silver_polished": {
        "element": "Ag",
        "dataset": "Yang",
        "nk": "silver",
        "kind": "MEASURED: template-stripped silver, an ultra-smooth film.",
        "description": "Polished silver: mirrors' reflective layer, plated fittings",
        "thermal": {"density_kg_m3": 10500, "specific_heat_j_kgk": 235, "conductivity_w_mk": 429},
        "thermal_note": f"pure silver, {HT_A1}",
        "thickness_m": 0.0005,
    },
    "magnesium_polished": {
        "element": "Mg",
        "dataset": "Hagemann",
        "nk": "magnesium",
        "kind": (
            "MEASURED and compiled: magnesium's optical constants from the far infrared to the "
            "X-ray region, assembled from published measurements."
        ),
        "description": "Freshly machined magnesium: cast drone and camera housings before they "
        "oxidise or are painted",
        "thermal": {"density_kg_m3": 1740, "specific_heat_j_kgk": 1024, "conductivity_w_mk": 156},
        "thermal_note": f"pure magnesium, {HT_A1}",
        "thickness_m": 0.002,
    },
}


def fetch(element: str, dataset: str) -> tuple[str, str]:
    url = f"{RII_RAW}/main/{element}/nk/{dataset}.yml"
    with urllib.request.urlopen(url, timeout=120) as response:  # noqa: S310 - fixed https URL
        return url, response.read().decode("utf-8")


def reference_of(text: str) -> str:
    """The REFERENCES (and COMMENTS) block of a RefractiveIndex.INFO file, as plain text."""

    def block(key: str) -> str:
        m = re.search(rf"^{key}:\s*\|\n((?:[ \t]+.*\n)+)", text, flags=re.MULTILINE)
        if not m:
            return ""
        raw = " ".join(line.strip() for line in m.group(1).splitlines())
        raw = re.sub(r"\[<a [^>]*>[^<]*</a>\]", "", raw)  # links to calculation scripts
        return html.unescape(re.sub(r"<[^>]+>", "", raw)).strip()

    ref = block("REFERENCES")
    comment = block("COMMENTS")
    return f"{ref} ({comment.rstrip('.')}.)" if comment else ref


def emissivity_curve(
    lam: NDArray[np.float64], n: NDArray[np.float64], k: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Normal-incidence ε(λ) = 1 − R on 400 log-spaced wavelengths across the table.

    n and k are interpolated linearly in λ, as the loader interpolates the n/k file, so the curve
    written here is what Level A would compute from that file at the same wavelengths.
    """
    # Rounded to the 6 figures the file carries: on a steep edge (silver's plasma edge near
    # 0.32 um) a rounded wavelength would otherwise move the value by 2e-5.
    grid = np.array([float(f"{x:.6g}") for x in np.geomspace(lam[0], lam[-1], 400)])
    grid = np.clip(grid, lam[0], lam[-1])
    ni = np.interp(grid, lam, n)
    ki = np.interp(grid, lam, k)
    return grid, 1.0 - ((ni - 1.0) ** 2 + ki**2) / ((ni + 1.0) ** 2 + ki**2)


def write_nk(
    name: str, spec: dict, url: str, text: str, need_um: tuple[float, float] = (TRIM_UM[0], 14.0)
) -> tuple[pathlib.Path, NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Write ``data/nk/<spec['nk']>.csv``; refuse a table that does not span ``need_um``."""
    lam, n, k = parse_rii_nk(text)
    unique = np.concatenate(([True], np.diff(lam) > 0.0))
    lam, n, k = lam[unique], n[unique], k[unique]
    # Keep one row beyond each edge, so the table spans the trim range rather than stopping
    # a step inside it (the solar average and the bands both refuse an edge they cannot reach).
    first = max(int(np.searchsorted(lam, TRIM_UM[0], side="right")) - 1, 0)
    last = min(int(np.searchsorted(lam, TRIM_UM[1], side="left")), len(lam) - 1)
    lam, n, k = lam[first : last + 1], n[first : last + 1], k[first : last + 1]
    if lam[0] > need_um[0] or lam[-1] < need_um[1]:
        raise ValueError(f"{name}: table covers {lam[0]}-{lam[-1]} um, needs {need_um} um")

    def at(target: float) -> tuple[float, float]:
        return float(np.interp(target, lam, n)), float(np.interp(target, lam, k))

    n1, k1 = at(1.55)
    n2, k2 = at(10.0)
    header = HEADER.format(
        material=spec["nk"],
        reference=reference_of(text),
        proxy_note=spec["kind"],
        url=url,
        date=datetime.date.today().isoformat(),
        lo=f"{lam[0]:.3g}",
        hi=f"{lam[-1]:.3g}",
        q1=1.55,
        n1=n1,
        k1=k1,
        q2=10.0,
        n2=n2,
        k2=k2,
        e2=normal_emissivity(n2, k2),
    ).replace("scripts/fetch_nk_tables.py", "scripts/import_material_spectra.py")
    body = "\n".join(f"{a:.6g},{b:.6g},{c:.6g}" for a, b, c in zip(lam, n, k, strict=True))
    path = REPO / "data" / "nk" / f"{spec['nk']}.csv"
    path.write_text(header + body + "\n", encoding="utf-8")
    # The curve is computed from the table as written (6 significant figures), so that it is
    # exactly what Level A computes from the checked-in file.
    lam, n, k = (np.array([float(f"{x:.6g}") for x in a]) for a in (lam, n, k))
    return path, lam, n, k


def write_curve(name: str, spec: dict, lam, n, k) -> pathlib.Path:
    grid, eps = emissivity_curve(lam, n, k)
    lines = [
        f"# Spectral emissivity of {name.replace('_', ' ')}, normal incidence, "
        f"{grid[0]:.3g}-{grid[-1]:.3g} um.",
        "#",
        f"# source: COMPUTED from data/nk/{spec['nk']}.csv (see its source block) as the "
        "normal-incidence Fresnel",
        "#   emissivity of a smooth surface, eps = 1 - |(n + ik - 1)/(n + ik + 1)|^2, on 400 "
        "log-spaced wavelengths",
        "#   with n and k interpolated linearly in wavelength; scripts/import_material_spectra.py.",
        "#   A polished laboratory surface: real parts are rougher and more oxidised, and both "
        "raise eps (section 4.5).",
        "# Columns: wavelength in MICROMETRES, emissivity.",
    ]
    rows = [f"{a:.6g},{b:.6g}" for a, b in zip(grid, eps, strict=True)]
    path = REPO / "data" / "spectra" / "materials" / f"{name}.csv"
    path.write_text("\n".join(lines + rows) + "\n", encoding="utf-8")
    return path


def solar_absorptivity_of(curve_path: pathlib.Path) -> float:
    sys.path.insert(0, str(REPO / "src"))
    from irsim.materials.spectra import load_property_spectrum, solar_absorptance
    from irsim.radiometry.solar import AM15_DIRECT_FILE, load_solar_spectrum

    solar = load_solar_spectrum(REPO / "data" / AM15_DIRECT_FILE)
    return solar_absorptance(load_property_spectrum(curve_path), solar)


YAML = """\
# {title}: {description}.
#
# Written by scripts/import_material_spectra.py (roadmap XD.13) -- edit the script, not this file.
# The emissivity is a curve computed from published optical constants (data/nk/{nk}.csv, whose
# source block names the paper) and the solar absorptivity is computed from the same curve under
# the AM1.5 direct spectrum (irsim.materials.spectra.solar_absorptance). This is the POLISHED
# state: the lower bound of this metal's emissivity. A real part in service is rougher and more
# oxidised, and both raise it (section 4.5); that is a different material, not this one.
schema_version: 2
material:
  name: {name}
  source: literature
  surface_treatment: polished
  reference: "{reference}"
  description: "{description}"
  thermal:
    density_kg_m3: {density_kg_m3}
    specific_heat_j_kgk: {specific_heat_j_kgk}
    conductivity_w_mk: {conductivity_w_mk}
    thickness_m: {thickness_m}   # ESTIMATED: a typical part
    solar_absorptivity: {alpha:.3f}   # computed from the curve, not typed
  optical:
    spectral_emissivity: spectra/materials/{name}.csv
    roughness_per_band: {{nir: 0.02, swir: 0.02, mwir: 0.01, lwir: 0.01}}   # ESTIMATED, polished
    # Level A from the same table the curve came from: a metal's emissivity RISES toward grazing,
    # which only Fresnel on n and k represents (section 4.2).
    angular_model: {{type: fresnel, n_k_file: nk/{nk}.csv}}
"""


def build_metal(name: str) -> None:
    spec = METALS[name]
    url, text = fetch(spec["element"], spec["dataset"])
    nk_path, lam, n, k = write_nk(name, spec, url, text)
    curve = write_curve(name, spec, lam, n, k)
    alpha = solar_absorptivity_of(curve)
    reference = (
        f"Optical constants: {reference_of(text)} via RefractiveIndex.INFO (CC0), "
        f"data/nk/{spec['nk']}.csv. Thermal: {spec['thermal_note']}"
    ).replace('"', "'")
    path = REPO / "configs" / "materials" / f"{name}.yaml"
    path.write_text(
        YAML.format(
            title=name.replace("_", " ").capitalize(),
            name=name,
            nk=spec["nk"],
            reference=reference,
            description=spec["description"],
            thickness_m=spec["thickness_m"],
            alpha=alpha,
            **spec["thermal"],
        ),
        encoding="utf-8",
    )
    grid, eps = emissivity_curve(lam, n, k)
    at10 = float(np.interp(10.0, grid, eps))
    print(f"{name:22s} eps(10 um) {at10:.4f}  alpha_solar {alpha:.3f}  <- {nk_path.name}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="source", required=True)
    metals = sub.add_parser("metals", help="polished metals from their optical constants")
    metals.add_argument("--only", choices=sorted(METALS))
    args = parser.parse_args()
    if args.source == "metals":
        for name in [args.only] if args.only else list(METALS):
            build_metal(name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
