#!/usr/bin/env python3
"""Spectral curves for materials whose *shape* is known from a proxy and whose *level* is authored.

    python scripts/derive_proxy_shape_curves.py          # write every curve below
    python scripts/derive_proxy_shape_curves.py --check  # exit 1 if a committed curve is stale

Roadmap XD.14. A camera whose range is not its band's nominal one reads a material's value off the
material's curve (ADR 0175), and the material that most needs one is glass: its Si-O reststrahlen
band takes the emissivity of a window from ~0.95 down to ~0.7 across 8-10 µm, so how much of that
dip a camera's range holds decides what glass looks like to it. No freely licensed soda-lime
spectrum spans the band (ECOSTRESS's plate-glass measurement is "all rights reserved", ADR 0041),
but the library already commits a fused-silica n, k table as the glass's angular **shape** proxy
(`data/nk/glass.csv`, M7.5), with the **magnitude** taken from the authored band value.

This keeps that split, spectrally: the normal-incidence Fresnel reflectance R(λ) of the proxy,
scaled by one factor s so that the curve's nominal-band average is the authored value,

    ε(λ) = 1 − s · R(λ),   s = (1 − ε_B) / ⟨R⟩_B,

over the wavelengths where the part is opaque (τ through ``thermal.thickness_m`` below 1e-6), so
that ε = 1 − R is Kirchhoff for it. Scaling the reflectance rather than the emissivity keeps ε in
[0, 1] by construction. The file says MODELLED and PROXY in its own source block, and the material
keeps its authored per-band numbers for the bands the curve does not cover.

docs/physics-model.md §4.1, §4.4, §12.3; ADR 0010, ADR 0041, ADR 0175; roadmap XD.14
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from irsim.materials.library import nominal_response  # noqa: E402
from irsim.materials.nk import load_nk_table  # noqa: E402
from irsim.materials.spectra import PropertySpectrum  # noqa: E402

#: material -> (n/k table, band whose authored value sets the level, authored value,
#:              thickness m, wavelength range to write in µm, step µm)
PROXIES: dict[str, tuple[str, str, float, float, tuple[float, float], float]] = {
    # From 5.02 um: opaque there, and clear of MWIR's nominal 3-5 um edge (its 0.85 stays).
    "glass_windshield": ("glass", "lwir", 0.88, 0.005, (5.02, 14.0), 0.02),
}

OPAQUE_TAU = 1e-6


def curve_for(name: str) -> tuple[np.ndarray, np.ndarray, float, str]:
    """(λ, ε, s, the table's own source line) for one proxy material."""
    nk_name, band, eps_b, thickness, (lo, hi), step = PROXIES[name]
    table = load_nk_table(nk_name)
    lam = np.round(np.arange(lo, hi + step / 2, step), 6)
    n, k = table.at(lam)
    tau = np.exp(-4.0 * np.pi * k / (lam * 1e-6) * thickness)
    if np.any(tau > OPAQUE_TAU):
        bad = lam[tau > OPAQUE_TAU]
        raise ValueError(
            f"{name}: not opaque through {thickness} m at {bad[0]:.2f}-{bad[-1]:.2f} um, where "
            "eps = 1 - R would not be Kirchhoff; start the range later"
        )
    r = ((n - 1.0) ** 2 + k**2) / ((n + 1.0) ** 2 + k**2)
    r_band = PropertySpectrum(lam, r, pathlib.Path("<R>")).band_effective(
        nominal_response(band), threshold=0.0
    )
    s = (1.0 - eps_b) / r_band
    eps = 1.0 - s * r
    if np.any(eps < 0.0) or np.any(eps > 1.0):
        raise ValueError(f"{name}: scaled curve leaves [0, 1]")
    citation = table.source.split(". doi:")[0].split(" (20")[0] if table.source else nk_name
    return lam, eps, float(s), citation


def render(name: str) -> str:
    nk_name, band, eps_b, thickness, (lo, hi), _ = PROXIES[name]
    lam, eps, s, nk_source = curve_for(name)
    head = [
        f"# Spectral emissivity of {name.replace('_', ' ')}, normal incidence, {lo:g}-{hi:g} um.",
        "#",
        f"# source: MODELLED, PROXY SHAPE. eps = 1 - s R(lambda), R the normal-incidence Fresnel "
        f"reflectance of data/nk/{nk_name}.csv ({nk_source}; its own source block has the rest),",
        f"#   s = {s:.6f} chosen so the {band.upper()} nominal-band average is the authored "
        f"{eps_b} (configs/materials/{name}.yaml).",
        f"#   Opaque through the material's {thickness * 1000:g} mm over the whole range "
        "(tau < 1e-6), so eps = 1 - R is Kirchhoff here.",
        "#   The SHAPE (where the Si-O reststrahlen dip sits, how deep against the shoulders) is "
        "the proxy's; the LEVEL is the authored value's.",
        "#   Written by scripts/derive_proxy_shape_curves.py (roadmap XD.14, ADR 0175); edit the "
        "script, not this file.",
        "# Columns: wavelength in MICROMETRES, emissivity.",
    ]
    rows = [f"{a:.4f},{b:.6f}" for a, b in zip(lam, eps, strict=True)]
    return "\n".join(head + rows) + "\n"


def path_for(name: str) -> pathlib.Path:
    return REPO / "data" / "spectra" / "materials" / f"{name}_lw.csv"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="exit 1 if a curve is stale")
    args = parser.parse_args()
    stale = 0
    for name in PROXIES:
        text = render(name)
        path = path_for(name)
        if args.check:
            if not path.is_file() or path.read_text(encoding="utf-8") != text:
                print(f"stale: {path.relative_to(REPO)}")
                stale += 1
            continue
        path.write_text(text, encoding="utf-8")
        lam, eps, s, _ = curve_for(name)
        print(
            f"{name}: s = {s:.4f}, eps {eps.min():.3f}-{eps.max():.3f} -> {path.relative_to(REPO)}"
        )
    return 1 if stale else 0


if __name__ == "__main__":
    raise SystemExit(main())
