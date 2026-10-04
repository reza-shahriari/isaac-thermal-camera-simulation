#!/usr/bin/env python3
"""Long-wave curve SHAPES from the ECOSTRESS spectral library -- only where no freer source exists.

    python scripts/import_ecostress.py                  # fetch, write the curves, relink materials
    python scripts/import_ecostress.py --cache DIR      # read the spectra from DIR instead
    python scripts/import_ecostress.py --check          # exit 1 if a committed curve is stale

Roadmap XD.14, ADR 0179. ECOSTRESS (formerly the ASTER spectral library; JPL, JHU and USGS
measurements) has what the CC0 and public-domain sources do not: measured thermal-infrared spectra
of rubber, road asphalt, construction concrete and paint. It states no licence; it asks for a
citation and for the credit line below, under which other projects reproduce its spectra. The
owner accepted that on 2026-10-01 **for materials no other source covers**, so this importer is the
last rung: every material listed here has no CC0 n, k table, no SLUM or USGS curve of its own class,
and no curve derived from one.

**Shape from the sample, level from the library.** These materials are the §16.2 literature rows
(`tests/unit/test_library_v0.py` pins them), and one ECOSTRESS sample of a class -- roofing rubber
for a tyre, an enamel for a car -- is not a better *level* for the class than the literature value.
What the material lacked is a *curve*: where in 5-14 µm it absorbs and reflects, which is what a
camera with its own wavelength range reads. So each curve is

    ε(λ) = 1 − s · R(λ),   s = (1 − ε_B) / ⟨R⟩_B,

R the sample's measured hemispherical reflectance (opaque, so ε = 1 − R is Kirchhoff) and s fixing
the LWIR nominal-band average at the authored ε_B -- the rule `derive_proxy_shape_curves.py` applies
to optical-constant shapes. Every standard-band value is unchanged; NIR, SWIR and MWIR stay typed.

**All paints share one shape.** Paint's thermal-infrared emissivity is set by its binder, not its
pigment (`tests/unit/test_colour_does_not_set_emissivity.py`), so the four library paints take the
same gloss enamel sample; only its level differs where their authored LWIR does.

Each curve file carries JPL's credit line verbatim, both citations, the sample's name, number and
description, and says when the sample is a class proxy.

docs/physics-model.md §4.1, §12.3, §16.2; ADR 0041, ADR 0175, ADR 0179; roadmap XD.14
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys
import urllib.request

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from irsim.materials.library import nominal_response  # noqa: E402
from irsim.materials.spectra import PropertySpectrum  # noqa: E402

URL = "https://speclib.jpl.nasa.gov/ecospeclibdata/{}"
CREDIT = (
    "Reproduced from the ECOSTRESS Spectral Library through the courtesy of the Jet Propulsion "
    "Laboratory, California Institute of Technology, Pasadena, California. Copyright (c) 2017, "
    "California Institute of Technology. ALL RIGHTS RESERVED."
)
CITE = (
    "Meerdink, S. K., Hook, S. J., Roberts, D. A., & Abbott, E. A. (2019), The ECOSTRESS spectral "
    "library version 1.0, Remote Sensing of Environment 230, 111196; Baldridge, A. M., Hook, "
    "S. J., Grove, C. I., & Rivera, G. (2009), The ASTER spectral library version 2.0, "
    "Remote Sensing of Environment 113, 711-715"
)
OUT = REPO / "data" / "spectra" / "materials" / "ecostress"
#: From just past MWIR's nominal 3-5 µm edge, so its typed value is untouched, to past 14 µm, so a
#: Boson response file (which runs to 14.0) is covered.
RANGE_UM = (5.02, 14.1)

PAINT = "manmade.generalconstructionmaterial.paint.solid.tir.1247.jpl.nicolet.spectrum.txt"
RUBBER = "manmade.roofingmaterial.rubber.solid.all.0833uuurbr.jhu.becknic.spectrum.txt"

#: material -> (ECOSTRESS file, the authored LWIR value the curve keeps, why this sample).
ENTRIES: dict[str, tuple[str, float, str]] = {
    "concrete": (
        "manmade.concrete.pavingconcrete.solid.all.0424uuucnc.jhu.becknic.spectrum.txt",
        0.92,
        "grey construction concrete with quartz and limestone aggregate",
    ),
    "asphalt_dry": (
        "manmade.road.pavingasphalt.solid.all.0095uuuasp.jhu.becknic.spectrum.txt",
        0.94,
        "weathered grey-black road asphalt",
    ),
    "rubber_tyre": (RUBBER, 0.95, "CLASS PROXY: black weathered roofing rubber for a tyre"),
    "propeller_rubber": (RUBBER, 0.95, "CLASS PROXY: black weathered roofing rubber"),
    "car_paint_black": (PAINT, 0.90, "CLASS PROXY: gloss spray enamel for automotive paint"),
    "car_paint_white": (PAINT, 0.90, "CLASS PROXY: gloss spray enamel for automotive paint"),
    "aircraft_aluminium_painted": (PAINT, 0.90, "CLASS PROXY: gloss spray enamel for a topcoat"),
    "painted_composite": (PAINT, 0.92, "CLASS PROXY: gloss spray enamel for a painted fairing"),
}

#: car_paint_black authored a 7-row ESTIMATED sketch curve over every band (section 12.3's example);
#: its replacement curve covers LWIR only, so the other bands keep the sketch's own band values.
SKETCH_BANDS = {"car_paint_black": "{nir: 0.94, swir: 0.94, mwir: 0.88}"}


def fetch(name: str, cache: pathlib.Path | None) -> str:
    if cache is not None and (cache / name).is_file():
        return (cache / name).read_text(encoding="utf-8", errors="replace")
    req = urllib.request.Request(URL.format(name), headers={"User-Agent": "irsim-import"})
    with urllib.request.urlopen(req, timeout=60) as response:  # noqa: S310 - fixed https URL
        return response.read().decode("utf-8", errors="replace")


def parse(text: str) -> tuple[dict[str, str], np.ndarray, np.ndarray]:
    """(header fields, wavelength µm ascending, reflectance as a fraction) of one ECOSTRESS file."""
    head: dict[str, str] = {}
    rows = []
    for line in text.splitlines():
        m = re.match(r"^([A-Z][A-Za-z .]+?):\s*(.*)$", line)
        if m and not rows:
            head[m.group(1).strip()] = m.group(2).strip()
            continue
        parts = line.split()
        if len(parts) == 2:
            try:
                rows.append((float(parts[0]), float(parts[1])))
            except ValueError:
                continue
    if "percent" not in head.get("Y Units", "").lower():
        raise ValueError(f"expected reflectance in percent, got {head.get('Y Units')!r}")
    data = np.array(sorted(dict(rows).items()))
    return head, data[:, 0], np.clip(data[:, 1] / 100.0, 0.0, 1.0)


def curve_for(material: str, cache: pathlib.Path | None):  # type: ignore[no-untyped-def]
    """(header, λ, ε, s): the sample's shape at the material's authored LWIR level."""
    name, eps_b, _ = ENTRIES[material]
    head, lam, refl = parse(fetch(name, cache))
    sel = (lam >= RANGE_UM[0]) & (lam <= RANGE_UM[1])
    lam, refl = lam[sel], refl[sel]
    r_band = PropertySpectrum(lam, refl, pathlib.Path("<R>")).band_effective(
        nominal_response("lwir"), threshold=0.0
    )
    s = (1.0 - eps_b) / r_band
    eps = 1.0 - s * refl
    if np.any(eps < 0.0) or np.any(eps > 1.0):
        raise ValueError(f"{material}: scaled curve leaves [0, 1]")
    return head, lam, eps, float(s)


def render(material: str, cache: pathlib.Path | None) -> str:
    name, eps_b, why = ENTRIES[material]
    head, lam, eps, s = curve_for(material, cache)
    lines = [
        f"# Spectral emissivity of {material.replace('_', ' ')}, {lam[0]:.4g}-{lam[-1]:.4g} um.",
        "#",
        "# source: MODELLED, MEASURED SHAPE. eps = 1 - s R(lambda), R the measured reflectance of "
        f"ECOSTRESS sample {head.get('Sample No.', '?')} '{head.get('Name', '?')}' ({name}), "
        f"{head.get('Measurement', '').strip()}.",
        f"#   s = {s:.6f} holds the LWIR nominal-band average at the authored {eps_b} "
        f"(configs/materials/{material}.yaml): the SHAPE is the sample's, the LEVEL the library's.",
        f"#   {CREDIT}",
        f"#   Cite: {CITE}.",
        f"#   Sample: {head.get('Description', '').strip()}",
        f"#   Used here as: {why}.",
        "#   Written by scripts/import_ecostress.py (roadmap XD.14, ADR 0179); edit the script.",
        "# Columns: wavelength in MICROMETRES, emissivity.",
    ]
    rows = [f"{a:.8g},{b:.9f}" for a, b in zip(lam, eps, strict=True)]
    return "\n".join(lines + rows) + "\n"


def relink(material: str) -> str:
    """Point the material's LWIR at the curve; every other band keeps its typed value."""
    path = REPO / "configs" / "materials" / f"{material}.yaml"
    text = path.read_text(encoding="utf-8")
    rel = f"spectra/materials/ecostress/{material}_lw.csv"
    if rel in text:
        return "already linked"
    _, eps_b, why = ENTRIES[material]
    segment = (
        "    # LWIR is a curve (XD.14, ADR 0179): an ECOSTRESS sample's SHAPE at the authored\n"
        f"    # {eps_b} LEVEL; the curve file names the sample and carries JPL's credit line.\n"
        f"    spectral_emissivity: {rel}\n"
    )
    if material in SKETCH_BANDS:
        text, n = re.subn(
            r"^    spectral_emissivity: .*$",
            segment.rstrip("\n") + f"\n    emissivity_per_band: {SKETCH_BANDS[material]}"
            "   # the replaced sketch's own band values",
            text,
            count=1,
            flags=re.M,
        )
    else:
        m = re.search(r"^    emissivity_per_band: \{(.*)\}(.*)$", text, re.M)
        assert m is not None, material
        kept = ", ".join(i for i in m.group(1).split(", ") if not i.startswith("lwir:"))
        text = text.replace(
            m.group(0), segment + f"    emissivity_per_band: {{{kept}}}{m.group(2)}"
        )
        n = 1
    assert n == 1, material
    text = re.sub(
        r'^(  reference: ")',
        rf"\g<1>LWIR shape from ECOSTRESS ({why}); ",
        text,
        count=1,
        flags=re.M,
    )
    path.write_text(text, encoding="utf-8")
    return "linked"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=pathlib.Path, help="a folder already holding the spectra")
    parser.add_argument("--check", action="store_true", help="exit 1 if a committed curve is stale")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    stale = 0
    for material in ENTRIES:
        text = render(material, args.cache)
        out = OUT / f"{material}_lw.csv"
        if args.check:
            if not out.is_file() or out.read_text(encoding="utf-8") != text:
                print(f"stale: {out.relative_to(REPO)}")
                stale += 1
            continue
        out.write_text(text, encoding="utf-8")
        _, _, eps, s = curve_for(material, args.cache)
        print(
            f"{material:28s} s = {s:.3f}, eps {eps.min():.3f}-{eps.max():.3f}  {relink(material)}"
        )
    return 1 if stale else 0


if __name__ == "__main__":
    raise SystemExit(main())
