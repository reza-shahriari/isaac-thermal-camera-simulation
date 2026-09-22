#!/usr/bin/env python3
"""Fetch RadCal's tabulated narrow-band absorption data out of the FDS source (roadmap PH.5).

    python scripts/fetch_radcal_tables.py --out data/spectra/radcal

RadCal (Grosshandler, *RadCal: A Narrow-Band Model for Radiation Calculations in a Combustion
Environment*, NIST Technical Note 1402, 1993) is the fallback source for the hot-gas absorption
tables of `PH.5` -- open question 14 in docs/roadmap.md offered RADIS over HITEMP first and RadCal
if that was blocked. It was: RADIS is not installed in this project's interpreter and HITEMP is a
multi-gigabyte registered download, so the committed tables come from RadCal. ADR 0098's addendum
records the swap.

RadCal ships inside FDS as ``Source/rcal.f90``. FDS is a work of the US federal government and is
in the **public domain**, so the numbers can be checked in rather than fetched at run time --
which matters because `make ci` has no network and a radiometric result that depends on a live
download is not reproducible (the same rule `scripts/fetch_nk_tables.py` follows).

Two of RadCal's species are tabulated rather than computed, and those are what this script
extracts:

* **H2O** -- ``SD(6, 376)``, the narrow-band mean absorption coefficient on a 25 cm-1 grid from
  50 to 9300 cm-1 at 300, 600, 1000, 1500, 2000 and 2500 K. RadCal's ``H2O`` routine is nothing
  but a bilinear interpolation in this table, so the table *is* the model.
* **CO2 15 um** -- ``SD15(6, 80)``, a 5 cm-1 grid from 500 to 880 cm-1 at 300, 600, 1200, 1500,
  1800 and 2400 K. CO2's other bands (2.0, 2.7, 4.3 and 10 um) are computed from closed-form
  vibration-rotation expressions, which `scripts/radcal.py` ports directly; only this one needs
  data.

Units are RadCal's own: cm-1 per atm of partial pressure, referred to STP density (the
``SDWEAK*TEMP/273`` convention of NASA SP-3080 that the computed bands also apply).
"""

from __future__ import annotations

import argparse
import datetime
import pathlib
import re
import urllib.request

import numpy as np
from numpy.typing import NDArray

#: Pinned by commit, not by branch: ``master`` moves and a table whose provenance says "master"
#: cannot be reproduced. Bump deliberately and re-run, then diff the CSVs.
FDS_COMMIT = "61bd9cc6317ea4f47f46de3d52268e858f0ca8b0"
RCAL_URL = f"https://raw.githubusercontent.com/firemodels/fds/{FDS_COMMIT}/Source/rcal.f90"

REFERENCE = (
    "W. L. Grosshandler (1993), 'RadCal: A Narrow-Band Model for Radiation Calculations in a "
    "Combustion Environment', NIST Technical Note 1402; as implemented in FDS Source/rcal.f90"
)

#: ``name -> (rows, columns, temperatures K, first wavenumber, wavenumber step)``. The wavenumber
#: of column ``j`` (1-based, as Fortran indexes it) is ``first + step * (j - 1)``; both grids are
#: read straight out of the indexing arithmetic in ``rcal.f90``'s ``H2O`` and ``CO2`` routines.
TABLES: dict[str, dict[str, object]] = {
    "h2o_sd": {
        "array": "SD",
        "shape": (6, 376),
        "temperatures_k": (300.0, 600.0, 1000.0, 1500.0, 2000.0, 2500.0),
        "first_wavenumber_cm1": 50.0,
        "step_cm1": 25.0,
        "note": (
            "H2O narrow-band mean absorption coefficient. RadCal's H2O routine is a bilinear "
            "interpolation in this table over (temperature, wavenumber) and nothing else."
        ),
    },
    "co2_sd15": {
        "array": "SD15",
        "shape": (6, 80),
        "temperatures_k": (300.0, 600.0, 1200.0, 1500.0, 1800.0, 2400.0),
        "first_wavenumber_cm1": 500.0,
        "step_cm1": 5.0,
        "note": (
            "CO2 15 um band (500-880 cm-1). RadCal's other CO2 bands are closed-form and are "
            "ported in scripts/radcal.py; this one is tabulated. Note the temperature grid is "
            "not the H2O one, and RadCal clamps a query above 2400 K to the top row."
        ),
    },
}


def _fortran_source(url: str) -> str:
    with urllib.request.urlopen(url, timeout=120) as handle:  # noqa: S310 - pinned https URL
        return handle.read().decode("utf-8", errors="replace")


def parse_array(source: str, name: str, rows: int, columns: int) -> NDArray[np.float64]:
    """Read ``NAME(1:rows, lo:hi) = RESHAPE((/ ... /), (/rows, n/))`` blocks into one array.

    Fortran's ``RESHAPE`` fills column-major, and each source line carries exactly ``rows``
    values, so line *k* of a block is column ``lo + k - 1`` -- one wavenumber, every temperature.
    """
    out = np.zeros((rows, columns), dtype=np.float64)
    filled = np.zeros(columns, dtype=bool)
    pattern = re.compile(
        rf"^{name}\(1:{rows},(\d+):(\d+)\)\s*=\s*RESHAPE\s*\(\(/(.*?)/\),", re.S | re.M
    )
    for match in pattern.finditer(source):
        lo, hi = int(match.group(1)), int(match.group(2))
        body = re.sub(r"!.*", "", match.group(3))
        values = [float(v.replace("_EB", "")) for v in re.findall(r"[-+]?[.\d][\d.eE+-]*_EB", body)]
        width = hi - lo + 1
        if len(values) != rows * width:
            raise ValueError(
                f"{name}({lo}:{hi}) holds {len(values)} numbers, expected {rows * width}"
            )
        out[:, lo - 1 : hi] = np.array(values).reshape(width, rows).T
        filled[lo - 1 : hi] = True
    if not filled.all():
        missing = int((~filled).sum())
        raise ValueError(f"{name}: {missing} of {columns} columns were never assigned")
    if np.any(out < 0.0):
        raise ValueError(f"{name}: a narrow-band absorption coefficient came out negative")
    return out


def write_table(path: pathlib.Path, table: dict[str, object], data: NDArray[np.float64]) -> None:
    temperatures = tuple(float(t) for t in table["temperatures_k"])  # type: ignore[arg-type]
    first = float(table["first_wavenumber_cm1"])  # type: ignore[arg-type]
    step = float(table["step_cm1"])  # type: ignore[arg-type]
    wavenumbers = first + step * np.arange(data.shape[1], dtype=np.float64)
    today = datetime.date.today().isoformat()
    lines = [
        f"# RadCal {table['array']} array, extracted from FDS Source/rcal.f90.",
        f"# source: {RCAL_URL}",
        f"# retrieved: {today}",
        f"# reference: {REFERENCE}",
        "# licence: public domain (a work of the US federal government).",
        "# STATUS: REFERENCE DATA -- transcribed, not modelled and not measured here.",
        f"# note: {table['note']}",
        "# units: cm-1 per atm of partial pressure, referred to STP density (NASA SP-3080).",
        "wavenumber_cm-1," + ",".join(f"k_{t:g}K" for t in temperatures),
    ]
    for j, omega in enumerate(wavenumbers):
        lines.append(f"{omega:g}," + ",".join(f"{v:.6g}" for v in data[:, j]))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path("data/spectra/radcal"))
    parser.add_argument("--source", type=pathlib.Path, help="a local rcal.f90 instead of the URL")
    args = parser.parse_args()

    source = args.source.read_text(encoding="utf-8") if args.source else _fortran_source(RCAL_URL)
    args.out.mkdir(parents=True, exist_ok=True)
    for name, table in TABLES.items():
        rows, columns = table["shape"]  # type: ignore[misc]
        data = parse_array(source, str(table["array"]), int(rows), int(columns))
        path = args.out / f"{name}.csv"
        write_table(path, table, data)
        print(f"{path}: {data.shape[0]} temperatures x {data.shape[1]} wavenumbers")


if __name__ == "__main__":
    main()
