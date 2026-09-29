#!/usr/bin/env python3
"""Write the tables of ``docs/materials.md`` from the material library itself.

    python scripts/material_catalogue.py          # rewrite the tables
    python scripts/material_catalogue.py --check  # exit 1 if the page is out of date

The page explains where the library's numbers come from; the tables under it list every material
with the values the simulator actually uses. Written by hand they would drift the first time a
material changed, so they are generated, between marker comments, and
``tests/unit/test_material_catalogue.py`` fails when the page and the library disagree.

Each value is the library's own band-effective emissivity (``Material.band_properties``, nominal
band, 300 K), so the page shows what a render uses, whatever form the file authors it in.

docs/physics-model.md §4.1, §16.2; roadmap XD.13
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from irsim.materials.library import MATERIAL_DIR, MaterialLibrary  # noqa: E402

PAGE = REPO / "docs" / "materials.md"
BANDS = ("nir", "swir", "mwir", "lwir")
MARK = "<!-- {} table: written by scripts/material_catalogue.py -->"
END = "<!-- end of {} table -->"

#: Measured materials set beside the literature row of docs/physics-model.md §16.2 that describes
#: the same kind of surface: the independent comparison the page shows.
COMPARE = {
    "asphalt_road_aged": "Asphalt (dry)",
    "asphalt_road_stone": "Asphalt (dry)",
    "concrete_weathered": "Concrete",
    "concrete_pavement": "Concrete",
    "cinder_block": "Concrete",
    "leaf_maple": "Vegetation (leaf)",
    "conifer_needles": "Vegetation (leaf)",
    "playa_dry_mud": "Soil (dry)",
    "sand_beach": "Soil (dry)",
    "seawater": "Water",
    "snow_melting": "Snow",
}


def origin(name: str) -> str:
    """How a material's optics were obtained, from the script that wrote it (if any)."""
    text = (MATERIAL_DIR / f"{name}.yaml").read_text(encoding="utf-8")
    if "scripts/import_slum.py" in text:
        return "measured: SLUM (MWIR from its class)"
    if "scripts/import_paired_spectra.py" in text:
        second = "polymer n, k" if "data/nk/" in text else "UCSB"
        return f"measured: USGS + {second}"
    if "scripts/import_material_spectra.py" in text:
        return "measured optical constants"
    if "ESTIMATED" in text:
        return "literature, partly estimated"
    return "literature"


def catalogue_table(library: MaterialLibrary) -> str:
    rows = [
        "| material | surface | NIR | SWIR | MWIR | LWIR | sunlight absorbed "
        "| where the optics come from |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name in library.names:
        m = library[name]
        eps = [m.band_properties(b).emissivity for b in BANDS]
        rows.append(
            f"| `{name}` | {m.spec.surface_treatment.replace('_', ' ')} | "
            + " | ".join(f"{e:.2f}" for e in eps)
            + f" | {m.spec.thermal.solar_absorptivity:.2f} | {origin(name)} |"
        )
    return "\n".join(rows)


def spec_rows() -> dict[str, tuple[float, float, float]]:
    """§16.2's literature rows: name -> (LWIR, MWIR, solar absorptivity)."""
    text = (REPO / "docs" / "physics-model.md").read_text(encoding="utf-8")
    section = text.split("### 16.2", 1)[1].split("### 16.3", 1)[0]
    out = {}
    for line in section.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) == 7 and re.fullmatch(r"[0-9.]+", cells[1]):
            out[cells[0]] = (float(cells[1]), float(cells[2]), float(cells[3]))
    return out


def comparison_table(library: MaterialLibrary) -> str:
    spec = spec_rows()
    rows = [
        "| measured material | literature row (§16.2) | LWIR measured / literature "
        "| MWIR measured / literature | sunlight absorbed measured / literature |",
        "|---|---|---|---|---|",
    ]
    for name, row in COMPARE.items():
        m = library[name]
        lwir, mwir, alpha = spec[row]
        rows.append(
            f"| `{name}` | {row} | {m.band_properties('lwir').emissivity:.2f} / {lwir:.2f} "
            f"| {m.band_properties('mwir').emissivity:.2f} / {mwir:.2f} "
            f"| {m.spec.thermal.solar_absorptivity:.2f} / {alpha:.2f} |"
        )
    return "\n".join(rows)


def render(page: str, library: MaterialLibrary) -> str:
    for key, table in (
        ("catalogue", catalogue_table(library)),
        ("comparison", comparison_table(library)),
    ):
        start, end = MARK.format(key), END.format(key)
        head, _, rest = page.partition(start)
        _, _, tail = rest.partition(end)
        if not rest:
            raise ValueError(f"{PAGE.name} has no '{start}' marker")
        page = f"{head}{start}\n\n{table}\n\n{end}{tail}"
    return page


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    library = MaterialLibrary.load()
    current = PAGE.read_text(encoding="utf-8")
    wanted = render(current, library)
    if args.check:
        if wanted != current:
            print(f"{PAGE} is out of date: run scripts/material_catalogue.py")
            return 1
        return 0
    PAGE.write_text(wanted, encoding="utf-8")
    print(f"wrote {PAGE} ({len(library)} materials)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
