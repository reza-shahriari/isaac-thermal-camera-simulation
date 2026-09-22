#!/usr/bin/env python3
"""Generate the modelled detector spectral-response curves (roadmap M11.1, M11.5).

    python scripts/generate_response_curves.py --out data/spectra/responses

Writes one peak-normalised R(lambda) per detector family. Each file is ESTIMATED -- a shape built
from the physics of the part, not a measurement of a specific unit -- and says so in its own
header, like data/spectra/responses/boson_vox.csv does.

Keeping them in a script rather than hand-authoring them means the *reasoning* is in one place:
where a cut-on comes from (a substrate or a cold filter), where a cut-off comes from (the alloy's
band gap), and how wide the edges are. Rerun and diff to see that nothing drifted.
"""

from __future__ import annotations

import argparse
import math
import pathlib

CURVES: dict[str, dict[str, object]] = {}


def _raised_cosine_up(lam: float, lo: float, hi: float) -> float:
    return 0.5 * (1.0 - math.cos(math.pi * (lam - lo) / (hi - lo)))


def _raised_cosine_down(lam: float, lo: float, hi: float) -> float:
    return 0.5 * (1.0 + math.cos(math.pi * (lam - lo) / (hi - lo)))


# --- InGaAs (SWIR, M11.1) ---------------------------------------------------------------------

INGAAS = dict(
    edges=(0.80, 1.80),
    step=0.005,
    decimals=3,
    header="""\
# Generic 640x512 InGaAs SWIR camera, relative spectral response R(lambda), peak-normalised.
# STATUS: ESTIMATED -- not a measurement of any specific unit. Optics + detector folded together.
#
# Shape (docs/physics-model.md §12.1 SWIR row, roadmap M11.1):
#   * cut-on: raised cosine over 0.85-0.95 um. Back-illuminated lattice-matched In0.53Ga0.47As
#     is read through its InP substrate, whose own gap (1.35 eV, 0.92 um) sets the short-wave
#     edge; the 0.1 um width is the absorption edge broadened by the AR coat and window.
#   * plateau: 0.95 -> 1.00 linearly over 0.95-1.60 um. Quantum efficiency creeps up as the
#     absorption depth grows towards the band edge; datasheets show a few percent of tilt, not a
#     flat top.
#   * cut-off: raised cosine over 1.60-1.72 um, i.e. half response at 1.66 um. The alloy's gap is
#     0.75 eV -> 1.653 um; vendors quote "1.7 um cutoff" at the 10 % point, which this shape puts
#     at 1.70 um.
# Expected error vs a measured curve: the plateau level is a normalisation, so it cancels in a
# peak-normalised R(lambda); the cut-off placement is good to ~0.02 um, the cut-on to ~0.05 um.
# Columns: wavelength in MICROMETRES, dimensionless response in [0, 1]. Grid 0.005 um.
# Authored 2026-09-15 (roadmap M11.1).
wavelength_um,response
""",
)


def ingaas(lam: float) -> float:
    if lam <= 0.85 or lam >= 1.72:
        return 0.0
    if lam < 0.95:
        return _raised_cosine_up(lam, 0.85, 0.95) * 0.95
    if lam < 1.60:
        return 0.95 + 0.05 * (lam - 0.95) / (1.60 - 0.95)
    return _raised_cosine_down(lam, 1.60, 1.72)


# --- InSb (MWIR, M11.5) -----------------------------------------------------------------------

INSB = dict(
    edges=(2.50, 6.00),
    step=0.005,
    decimals=3,
    header="""\
# Generic 640x512 InSb MWIR camera at 77 K, relative spectral response R(lambda), peak-normalised.
# STATUS: ESTIMATED -- not a measurement of any specific unit. Cold filter + detector + optics
# folded together.
#
# Shape (docs/physics-model.md §12.1 MWIR row, roadmap M11.5):
#   * cut-on: raised cosine over 2.90-3.10 um. InSb itself responds from about 1 um; the short-
#     wave edge of a 3-5 um camera is the COLD FILTER, not the detector. That is why it is
#     steeper than the InGaAs substrate edge -- an interference filter at 77 K, not an
#     absorption edge.
#   * plateau: 0.92 -> 1.00 linearly over 3.10-4.90 um, the usual mild rise towards the band edge.
#   * cut-off: raised cosine over 4.90-5.25 um. Two mechanisms coincide here and the narrower one
#     wins: the cold filter's long-wave edge sits just inside InSb's own gap (0.23 eV at 77 K ->
#     5.39 um), which is why a warmer detector's cut-off creeps out and a filtered camera's does
#     not.
# NOTE: this is a plain 3-5 um shape. Many MWIR cameras ship a "blue" filter (about 3.6-4.2 um)
# to sit outside the 4.2-4.4 um CO2 absorption; that is a different file, not a different model,
# which is the point of keeping bands in data.
# Columns: wavelength in MICROMETRES, dimensionless response in [0, 1]. Grid 0.005 um.
# Authored 2026-09-15 (roadmap M11.5).
wavelength_um,response
""",
)


def insb(lam: float) -> float:
    if lam <= 2.90 or lam >= 5.25:
        return 0.0
    if lam < 3.10:
        return _raised_cosine_up(lam, 2.90, 3.10) * 0.92
    if lam < 4.90:
        return 0.92 + 0.08 * (lam - 3.10) / (4.90 - 3.10)
    return _raised_cosine_down(lam, 4.90, 5.25)


# --- Silicon CMOS (NIR, ME/M11 renders) -------------------------------------------------------

#: Silicon's absorption coefficient near its band edge, cm^-1 at 300 K, on a 50 nm grid.
#: STATUS: ESTIMATED. These are the standard published magnitudes for intrinsic silicon (the
#: shape everyone plots from Green 2008), reproduced to about 10 % rather than transcribed from a
#: table this project has in hand. They are used for a *shape* that is then peak-normalised, which
#: is what makes 10 % on each anchor tolerable: the ratio across the band is what matters, and it
#: is dominated by the four decades alpha falls over, not by any one point. Replace with a
#: tabulated source before quoting an absolute quantum efficiency.
SI_ALPHA_CM1 = {
    0.70: 1.90e3,
    0.75: 1.09e3,
    0.80: 8.50e2,
    0.85: 5.37e2,
    0.90: 3.06e2,
    0.95: 1.60e2,
    1.00: 6.40e1,
    1.05: 1.40e1,
    1.10: 6.00e-1,
}

#: Photosensitive silicon thickness. A front-illuminated CMOS imager collects from its epitaxial
#: layer, a few microns deep; this is what makes a silicon NIR camera fall off so hard towards
#: 1 um, and it is the one number that moves the shape.
SI_EPI_UM = 6.0

#: Where the IR-cut filter used to be. A NIR camera is a visible sensor with that filter replaced
#: by a long pass, so the short-wave edge is the *filter*, sharp, and not silicon's own response.
SI_LONGPASS_UM = (0.72, 0.78)

NIR_SI = dict(
    edges=(0.70, 1.10),
    step=0.005,
    decimals=3,
    header="""\
# Generic 1280x1024 silicon CMOS NIR camera, relative spectral response R(lambda), peak-normalised.
# STATUS: ESTIMATED -- not a measurement of any specific unit. Long-pass filter + silicon epi
# absorption + optics folded together.
#
# Shape (docs/physics-model.md §12.1; the NIR band is reflective, roadmap M11):
#   * cut-on: raised cosine over 0.72-0.78 um. This is the FILTER, not the detector. A NIR camera
#     is an ordinary visible imager with its IR-cut filter replaced by a long pass, so the
#     short-wave edge is an interference coating and is steep; silicon itself responds right
#     through the visible.
#   * body: QE(lambda) = 1 - exp(-alpha(lambda) . d) with d = 6 um of epitaxial silicon and
#     alpha from the band-edge table in the generator, peak-normalised on write (the absolute
#     peak is QE = 0.431 at 0.775 um; absolute efficiency belongs in fpa.quantum_efficiency,
#     ADR 0009). There is no plateau. Silicon becomes
#     transparent as the photon energy approaches its 1.12 eV indirect gap, so the response peaks
#     at 0.775 um -- where the filter has finished opening and silicon has not yet given up --
#     and then falls monotonically by 10.6x between 0.80 and 1.00 um (R = 0.926 -> 0.087). That
#     fall-off is the single most important fact about this band: it is the reason a NIR camera is
#     a *near*-infrared camera and not a SWIR one, and the reason a NIR scene is lit almost
#     entirely by the short end of its own band.
#   * cut-off: the same exponential, which reaches zero of its own accord at 1.107 um (1.12 eV).
#     No filter is needed or modelled at the long end.
# Expected error vs a measured curve: the alpha anchors are good to ~10 % and the epi thickness
# is a design choice that varies by part, so the fall-off rate is the uncertain quantity; the
# band edges are good to ~0.01 um (filter) and ~0.005 um (gap).
# Columns: wavelength in MICROMETRES, dimensionless response in [0, 1]. Grid 0.005 um.
# Authored 2026-09-15.
wavelength_um,response
""",
)


def _si_alpha_cm1(lam: float) -> float:
    """Log-linear interpolation of the band-edge anchors. Log, because alpha falls four decades
    across this band and a linear interpolation between two anchors would be wrong by more than
    the anchors themselves are."""
    keys = sorted(SI_ALPHA_CM1)
    if lam <= keys[0]:
        return SI_ALPHA_CM1[keys[0]]
    if lam >= keys[-1]:
        return 0.0
    for lo, hi in zip(keys, keys[1:], strict=False):
        if lo <= lam <= hi:
            f = (lam - lo) / (hi - lo)
            return math.exp((1.0 - f) * math.log(SI_ALPHA_CM1[lo]) + f * math.log(SI_ALPHA_CM1[hi]))
    return 0.0


def nir_si(lam: float) -> float:
    if lam <= SI_LONGPASS_UM[0] or lam >= 1.107:
        return 0.0
    alpha_um1 = _si_alpha_cm1(lam) / 1e4  # cm^-1 -> um^-1
    qe = 1.0 - math.exp(-alpha_um1 * SI_EPI_UM)
    if lam < SI_LONGPASS_UM[1]:
        qe *= _raised_cosine_up(lam, *SI_LONGPASS_UM)
    return qe


# --- InSb behind a through-flame bandpass (MWIR, PH.5) -----------------------------------------

#: The half-power edges of the bandpass, micrometres. Between the 2.7 um CO2/H2O complex and the
#: 4.3 um CO2 band there is a window in which hot combustion gas is nearly transparent, and a
#: filter that sits inside it lets an MWIR camera see the *soot and the surfaces* through a flame
#: instead of the flame's own gas emission. 3.80-4.05 um is the usual placement (the "flame
#: window" or "through-flame" filter sold for furnace and burner work).
FLAME_WINDOW_UM = (3.80, 4.05)
FLAME_EDGE_UM = 0.04  # half-width of each interference edge: 50 % at the quoted wavelength

INSB_FLAME_WINDOW = dict(
    edges=(3.60, 4.25),
    step=0.005,
    decimals=3,
    header="""\
# Generic InSb MWIR camera behind a 3.80-4.05 um through-flame bandpass, relative spectral
# response R(lambda), peak-normalised.
# STATUS: ESTIMATED -- not a measurement of any specific unit. Cold filter + bandpass + detector
# + optics folded together.
#
# Shape (roadmap PH.5; the INSB header in this generator predicted this file):
#   * the detector and its cold filter are exactly insb.csv's shape, multiplied by
#   * a bandpass with 50 % points at 3.80 and 4.05 um and raised-cosine edges 0.08 um wide, which
#     is what a 5-cavity interference filter at 77 K gives.
# WHY IT EXISTS: this passband sits in the gap between the 2.7 um CO2/H2O complex and the 4.3 um
# CO2 band. A plain 3-5 um camera looking at a flame sees the gas: hot CO2 at 4.3 um dominates
# the band and hides whatever is behind it. This one sees through the gas and reads the soot and
# the surfaces instead. The two cameras differ only by this file -- no kernel knows the
# difference, which is what "bands are data, not code" has to mean to be worth anything.
# The claim is checked, not asserted: tests/unit/test_gas_tables.py compares the committed
# kappa_b(T) table for this response against the one for 3-5 um.
# Columns: wavelength in MICROMETRES, dimensionless response in [0, 1]. Grid 0.005 um.
# Authored 2026-09-22 (roadmap PH.5).
wavelength_um,response
""",
)


def insb_flame_window(lam: float) -> float:
    lo, hi = FLAME_WINDOW_UM
    if lam <= lo - FLAME_EDGE_UM or lam >= hi + FLAME_EDGE_UM:
        return 0.0
    if lam < lo + FLAME_EDGE_UM:
        pass_band = _raised_cosine_up(lam, lo - FLAME_EDGE_UM, lo + FLAME_EDGE_UM)
    elif lam > hi - FLAME_EDGE_UM:
        pass_band = _raised_cosine_down(lam, hi - FLAME_EDGE_UM, hi + FLAME_EDGE_UM)
    else:
        pass_band = 1.0
    return insb(lam) * pass_band


CURVES["ingaas"] = {**INGAAS, "fn": ingaas}
CURVES["nir_si"] = {**NIR_SI, "fn": nir_si}
CURVES["insb"] = {**INSB, "fn": insb}
CURVES["insb_flame_window"] = {**INSB_FLAME_WINDOW, "fn": insb_flame_window}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/spectra/responses")
    parser.add_argument("--only", action="append", choices=sorted(CURVES))
    args = parser.parse_args(argv)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name in args.only or sorted(CURVES):
        spec = CURVES[name]
        lo, hi = spec["edges"]  # type: ignore[misc]
        step = float(spec["step"])  # type: ignore[arg-type]
        fn = spec["fn"]  # type: ignore[assignment]
        decimals = int(spec["decimals"])  # type: ignore[arg-type]
        n = int(round((hi - lo) / step)) + 1
        samples = [(lo + i * step, float(fn(lo + i * step))) for i in range(n)]  # type: ignore[operator]
        peak = max(value for _, value in samples)
        if peak <= 0.0:
            raise SystemExit(f"{name}: the curve is zero everywhere")
        # R(lambda) is a peak-normalised *shape* and the loader refuses anything else (ADR 0009:
        # absolute QE lives in fpa.quantum_efficiency, so that a response file and a detector's
        # efficiency cannot disagree). Normalising here rather than in each curve means a model
        # written in physical units -- silicon's QE = 1 - exp(-alpha d), say -- can stay in them.
        rows = [f"{lam:.{decimals}f},{value / peak:.6f}" for lam, value in samples]
        path = out / f"{name}.csv"
        path.write_text(str(spec["header"]) + "\n".join(rows) + "\n", encoding="utf-8")
        where = next(lam for lam, value in samples if value == peak)
        print(f"wrote {path} ({n} rows), peak {peak:.6f} at {where:.3f} um -> normalised to 1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
