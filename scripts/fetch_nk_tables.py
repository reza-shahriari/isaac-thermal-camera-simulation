#!/usr/bin/env python3
"""Fetch the MEASURED complex-refractive-index tables from refractiveindex.info (roadmap M7.5).

    python scripts/fetch_nk_tables.py --out data/nk            # every table below
    python scripts/fetch_nk_tables.py --out data/nk --only glass

Companion to ``scripts/generate_nk_tables.py``, which produces the *modelled* tables. Measured data
is preferred wherever it is obtainable: a model reproduces the physics its author put into it, and
these two materials have published measurements covering every band the project uses.

The RefractiveIndex.INFO database is in the **public domain** (CC0 1.0), so the numbers can be
checked in here rather than fetched at run time -- which matters, because `make ci` has no network
and a radiometric result that depends on a live download is not reproducible. This script is how
the checked-in file is *regenerated*, and the URL and retrieval date go into the file's own
`# source:` header, which the loader treats as data: a table without provenance is refused
(ADR 0041).

**Both tables are proxies, and the header of each says so.** Neither substance has freely available
optical constants of its own across 0.75-13.5 um:

* **glass** -- fused silica standing in for soda-lime float glass. They share the Si-O network that
  produces the reststrahlen band dominating the LWIR, which is the feature the angular model needs;
  soda-lime's extra alkali and iron content shifts the magnitude, not the shape. The magnitude does
  not come from here anyway (see below).
* **paint_proxy** -- PMMA standing in for an automotive clearcoat. Clearcoat *is* an acrylic, so
  this is a close proxy for the binder; what it omits is the pigment loading underneath it.

Neither is used for its absolute emissivity. `directional_emissivity` takes the **shape** from the
table and the **magnitude** from the authored per-band value (the finding M7.5 recorded for
aluminium), so what a proxy has to get right is how emissivity falls with angle -- which is set by
the interface and the band's n and k, not by the pigment.
"""

from __future__ import annotations

import argparse
import datetime
import pathlib
import re
import urllib.request

import numpy as np
from numpy.typing import NDArray

RII_RAW = (
    "https://raw.githubusercontent.com/polyanskiy/refractiveindex.info-database/main/database/data"
)

#: One entry per checked-in measured table: where it comes from, what range to keep, and the
#: provenance prose that goes into the file. ``trim_um`` covers every §12.1 band (NIR 0.75-1.0
#: through LWIR 7.5-13.5) with margin, because the loader **refuses to extrapolate** -- a table
#: that stops inside a band makes that band raise rather than quietly invent a number.
SOURCES: dict[str, dict[str, object]] = {
    "glass": {
        "path": "main/SiO2/nk/Franta.yml",
        "trim_um": (0.6, 25.0),
        "material": "glass (fused silica, as a proxy for soda-lime float glass)",
        "reference": (
            "D. Franta, D. Necas, I. Ohlidal, A. Giglia (2016), 'Optical characterization of "
            "SiO2 thin films using universal dispersion model over wide spectral range', "
            "Proc. SPIE 9890, 989014. doi:10.1117/12.2227580. Fused silica PLATE, 0.405 mm, "
            "room temperature -- a bulk sample rather than a thin film, which is why this "
            "dataset was chosen over the thin-film alternatives in the same database."
        ),
        "proxy_note": (
            "PROXY: fused silica, not soda-lime. No freely available soda-lime dataset spans "
            "0.75-13.5 um. Both are silicate glasses and share the Si-O stretching resonance "
            "that dominates the LWIR; soda-lime's alkali and iron content changes the magnitude "
            "more than the angular shape, and the magnitude is taken from the material's "
            "authored per-band emissivity, not from this table (M7.5)."
        ),
    },
    "paint_proxy": {
        "path": ("organic/(C5H8O2)n%20-%20poly(methyl%20methacrylate)/nk/Zhang-Mitsubishi.yml"),
        "trim_um": (0.6, 19.9),
        "material": "paint proxy (PMMA, as a proxy for an automotive clearcoat)",
        "reference": (
            "X. Zhang, J. Qiu, X. Li, J. Zhao, L. Liu (2020), 'Complex refractive indices "
            "measurements of polymers in visible and near-infrared bands', Appl. Opt. 59, "
            "2337-2344. doi:10.1364/AO.383831. Mitsubishi PMMA sample."
        ),
        "proxy_note": (
            "PROXY: PMMA, not paint. A paint film is a pigmented basecoat under a polymer "
            "clearcoat, and the clearcoat is an acrylic -- so this is the binder, measured, "
            "without the pigment. It exists to give the *painted* material class a fitted "
            "Level B (a, p) instead of an estimated one (§4.2 asks for exactly that fit), and "
            "the emissivity magnitude still comes from each material's authored band value."
        ),
    },
    # XD.14: the shapes behind the curves `scripts/derive_proxy_shape_curves.py` writes (ADR 0175).
    "polycarbonate": {
        "path": "organic/(C16H14O3)n%20-%20polycarbonate/nk/Zhang.yml",
        "trim_um": (0.6, 19.9),
        "material": "polycarbonate (bulk, unpigmented)",
        "reference": (
            "X. Zhang, J. Qiu, J. Zhao, X. Li, L. Liu (2020), 'Complex refractive indices "
            "measurements of polymers in infrared bands', J. Quant. Spectrosc. Radiat. Transf. "
            "252, 107063. doi:10.1016/j.jqsrt.2020.107063; below 2 um the same authors, Appl. "
            "Opt. 59, 2337-2344 (2020), doi:10.1364/AO.383831."
        ),
        "proxy_note": (
            "MEASURED, the polymer itself but unpigmented: a grey or black part's pigment "
            "changes the level mainly where the polymer is weakly absorbing, so the curve built "
            "from this keeps its SHAPE and takes its LEVEL from the material's authored value."
        ),
    },
    "ice": {
        "path": "main/H2O/nk/Warren-2008.yml",
        "trim_um": (0.3, 25.0),
        "material": "water ice at -7 C",
        "reference": (
            "S. G. Warren and R. E. Brandt (2008), 'Optical constants of ice from the "
            "ultraviolet to the microwave: A revised compilation', J. Geophys. Res. 113, "
            "D14220. doi:10.1029/2007JD009744."
        ),
        "proxy_note": (
            "MEASURED, solid ice. Snow is granular ice, so its emissivity is higher and flatter "
            "than a smooth ice surface's; the snow curve built from this keeps the SHAPE (where "
            "ice's reflectance rises, 10-13 um) and takes its LEVEL from the authored value."
        ),
    },
}

#: The proxy note stays INSIDE the ``# source:`` block, with no blank comment line before it: the
#: loader ends the provenance it carries at the first empty comment, and "this is a proxy for
#: something else" is exactly the part of the provenance a later reader must not lose.
HEADER = """\
# Complex refractive index of {material}, n(lambda) + i k(lambda), {lo}-{hi} um.
#
# source: {reference}
#   Retrieved {date} from
#   {url}
#   (RefractiveIndex.INFO database, public domain / CC0 1.0), converted by
#   scripts/fetch_nk_tables.py and truncated to {lo}-{hi} um; values otherwise unmodified.
#   {proxy_note}
#
# Check values interpolated from this table: n({q1} um) = {n1:.6f}, k = {k1:.5g};
#   n({q2} um) = {n2:.6f}, k = {k2:.5g}. Normal-incidence Fresnel emissivity at {q2} um: {e2:.5f}.
# Columns: wavelength in MICROMETRES, then n, then k.
wavelength_um,n,k
"""

_ROW = re.compile(r"^\s+([0-9.eE+-]+)\s+([0-9.eE+-]+)\s+([0-9.eE+-]+)\s*$")


def parse_rii_nk(text: str) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Pull the ``tabulated nk`` block out of a RefractiveIndex.INFO YAML file.

    Deliberately not a YAML parse: the payload is a block scalar of whitespace-separated numbers,
    and the rows are the only three-number lines in the file. Anything that is not three numbers
    is prose, and prose is not data.
    """
    rows = [m.groups() for line in text.splitlines() if (m := _ROW.match(line))]
    if not rows:
        raise ValueError("no 'lambda n k' rows found; has the database format changed?")
    table = np.asarray(rows, dtype=np.float64)
    lam, n, k = table[:, 0], table[:, 1], table[:, 2]
    order = np.argsort(lam)
    return lam[order], n[order], k[order]


def normal_emissivity(n: float, k: float) -> float:
    """1 - R at normal incidence, for the header's check value."""
    r = ((n - 1.0) ** 2 + k**2) / ((n + 1.0) ** 2 + k**2)
    return 1.0 - r


def build(name: str, spec: dict[str, object], out_dir: pathlib.Path) -> pathlib.Path:
    url = f"{RII_RAW}/{spec['path']}"
    with urllib.request.urlopen(url, timeout=120) as response:  # noqa: S310 - fixed https URL
        text = response.read().decode("utf-8")
    lam, n, k = parse_rii_nk(text)

    lo, hi = spec["trim_um"]  # type: ignore[misc]
    keep = (lam >= lo) & (lam <= hi)
    if keep.sum() < 2:
        raise ValueError(f"{name}: only {keep.sum()} rows inside {lo}-{hi} um")
    lam, n, k = lam[keep], n[keep], k[keep]
    if np.any(np.diff(lam) <= 0.0):
        # The loader requires a strictly increasing axis; duplicates would be silently interpolated
        # across if it did not.
        unique = np.concatenate(([True], np.diff(lam) > 0.0))
        lam, n, k = lam[unique], n[unique], k[unique]

    def at(target: float) -> tuple[float, float]:
        """Interpolated, not nearest-row: the header must quote what the loader will return."""
        return float(np.interp(target, lam, n)), float(np.interp(target, lam, k))

    q1, q2 = 1.55, 10.0
    n1, k1 = at(q1)
    n2, k2 = at(q2)
    header = HEADER.format(
        material=spec["material"],
        reference=spec["reference"],
        proxy_note=spec["proxy_note"],
        url=url,
        date=datetime.date.today().isoformat(),
        lo=f"{lam[0]:.3g}",
        hi=f"{lam[-1]:.3g}",
        q1=q1,
        n1=n1,
        k1=k1,
        q2=q2,
        n2=n2,
        k2=k2,
        e2=normal_emissivity(n2, k2),
    )
    body = "\n".join(f"{a:.6g},{b:.6g},{c:.6g}" for a, b, c in zip(lam, n, k, strict=True))
    path = out_dir / f"{name}.csv"
    path.write_text(header + body + "\n", encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/nk")
    parser.add_argument("--only", default=None, choices=sorted(SOURCES))
    args = parser.parse_args()

    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    names = [args.only] if args.only else sorted(SOURCES)
    for name in names:
        path = build(name, SOURCES[name], out_dir)
        rows = sum(1 for line in path.read_text().splitlines() if not line.startswith("#"))
        print(f"wrote {path} ({rows - 1} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
