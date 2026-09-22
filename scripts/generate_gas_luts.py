#!/usr/bin/env python3
"""Per-band hot-gas absorption tables, generated offline (roadmap PH.5).

    python scripts/generate_gas_luts.py                      # every band and every response curve
    python scripts/generate_gas_luts.py --only mwir --only insb_flame_window

Writes ``data/gas/<key>_{co2,h2o}.npy`` -- float32 ``kappa_b(T, X)`` in 1/(m.atm) -- and a
``data/gas/<key>_gas.json`` sidecar carrying the grids, the provenance and a hash of each array,
following the same shape as the band LUTs in ``data/lut``. `irsim.pipeline.gas_tables` reads them
and :class:`~irsim.pipeline.gas_slab.GasBandTables` consumes them; nothing in the physics core
evaluates a spectral model at render time (ADR 0098).

**The band mean, and why it has two axes.** ``L_b = tau_b L_behind + (1 - tau_b) B_b(T_g)`` wants
one coefficient per band, and the one worth having is the one that reproduces the band's own
transmittance at the path the ray actually takes:

    tau_bar(T, X) = INT R B exp(-kappa(lambda, T) X) dlambda / INT R B dlambda
    kappa_b(T, X) = -ln tau_bar(T, X) / X

with ``X = p.L`` the species' column density in atm.m. The obvious alternative -- the
emission-weighted mean of ``kappa(lambda)``, which is RadCal's Planck-mean coefficient restricted
to the response -- is *wrong for this model*, and not slightly. In a 3-5 um camera essentially all
of CO2's absorption sits in 4.2-4.45 um; the Planck mean of that spike is 200-360 1/(m.atm), and
putting it inside one exponential makes a 30 cm exhaust plume opaque when the measured band
transmittance is 0.8. It also falls with temperature, where the band's *absorptance* rises, so it
fails `PH.5`'s own acceptance check. The effective coefficient above gets both right: it is the
Planck mean in the optically thin limit and it saturates like the real band when the core does.

Weighted in energy, not photons, for the same reason ``soot_band_kappa_per_m`` is: the weight is
the emission the slab actually puts into the band, and the photon/energy distinction belongs to
the detector, one stage later.

**Where the numbers come from.** ``scripts/radcal.py`` -- a port of RadCal's ``CO2`` and ``H2O``
weak-line coefficients. Open question 14 in docs/roadmap.md offered RADIS over HITEMP first and
RadCal if that was blocked; it was blocked (RADIS is not installed in the project interpreter, and
HITEMP is a registered multi-gigabyte download into an interpreter shared with other work), so
these tables are RadCal's. ADR 0098's addendum records the swap and what it costs.

**A band the model cannot speak about gets no file.** RadCal truncates: CO2 above 5725 cm-1
(1.75 um) and H2O above 9300 cm-1 (1.08 um) are *set* to zero, which is right for fire heat
transfer and wrong for a SWIR or a NIR camera, where real overtone bands live. Writing zeros
there would look like a transparent gas rather than like missing data, so each key carries the
R.B-weighted fraction of its band that falls inside the model's support, and the loader refuses a
table below :data:`irsim.pipeline.gas_tables.MIN_SUPPORT_FRACTION`. A NIR flame is then an error
message that names HITEMP, not a silently invisible one.

docs/physics-model.md §8.1, §8.3; ADR 0098; roadmap PH.4, PH.5.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import pathlib
import sys

import numpy as np
from numpy.typing import NDArray

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import radcal  # noqa: E402

from irsim.config.bands import BAND_IDS, NOMINAL_RANGES_UM  # noqa: E402
from irsim.materials.library import nominal_response  # noqa: E402
from irsim.pipeline.gas_slab import GAS_T_MAX_K, GAS_T_MIN_K  # noqa: E402
from irsim.radiometry.band_integration import simpson  # noqa: E402
from irsim.radiometry.planck import spectral_radiance  # noqa: E402
from irsim.radiometry.spectral_response import load_spectral_response  # noqa: E402

#: 25 K is finer than the curvature of any band mean here, so ``SpeciesAbsorption``'s linear
#: interpolation between grid points costs far less than the model's own envelope.
TEMPERATURE_STEP_K = 25.0

#: The column-density axis, atm.m, log-spaced. The span covers a 1 mm flame sheet at the bottom
#: and a hundred-metre maritime plume at the top; five points per decade puts the interpolation
#: error well inside the model's own envelope, because the quantity interpolated is the optical
#: depth and it is smooth in log X.
COLUMN_MIN_ATM_M, COLUMN_MAX_ATM_M, COLUMNS_PER_DECADE = 1.0e-4, 10.0, 5

#: The quadrature grid, micrometres. 0.002 um is 1.1 cm-1 at 13.5 um and 12 cm-1 at 1.3 um --
#: finer than RadCal's own 5 cm-1 (CO2 15 um) and 25 cm-1 (H2O) data everywhere those tables have
#: anything to say, so the grid is never what limits the answer.
GRID_LO_UM, GRID_HI_UM, GRID_STEP_UM = 0.70, 13.60, 0.002

SPECIES = ("co2", "h2o")

RESPONSE_DIR = pathlib.Path("data/spectra/responses")
OUT_DIR = pathlib.Path("data/gas")


def quadrature_wavelengths() -> NDArray[np.float64]:
    n = int(round((GRID_HI_UM - GRID_LO_UM) / GRID_STEP_UM)) + 1
    if n % 2 == 0:  # Simpson needs an odd count
        n += 1
    return GRID_LO_UM + GRID_STEP_UM * np.arange(n, dtype=np.float64)


def spectral_kappa(
    wavelengths_um: NDArray[np.float64], temp_k: float
) -> dict[str, NDArray[np.float64]]:
    """``kappa(lambda, T)`` for both species, 1/(m.atm), on one wavelength grid."""
    omega = 1.0e4 / wavelengths_um  # um -> cm-1
    tables = radcal.load_radcal_tables()
    return {
        "co2": 100.0 * radcal.kappa_co2_cm_atm(omega, temp_k, tables),  # 1/(cm.atm) -> 1/(m.atm)
        "h2o": 100.0 * radcal.kappa_h2o_cm_atm(omega, temp_k, tables),
    }


def support_mask(wavelengths_um: NDArray[np.float64], species: str) -> NDArray[np.bool_]:
    lo, hi = radcal.CO2_SUPPORT_CM1 if species == "co2" else radcal.H2O_SUPPORT_CM1
    omega = 1.0e4 / wavelengths_um
    return (omega >= lo) & (omega < hi)


def effective_kappa(
    weights: NDArray[np.float64],
    kappa_lambda: NDArray[np.float64],
    columns: NDArray[np.float64],
    dx: float,
) -> NDArray[np.float64]:
    """``-ln <exp(-kappa X)>_RB / X`` at every column, 1/(m.atm)."""
    denominator = simpson(weights, dx)
    if denominator <= 0.0:
        raise ValueError("the response has no weight on the quadrature grid")
    out = np.empty(columns.shape[0], dtype=np.float64)
    for j, column in enumerate(columns):
        tau = float(simpson(weights * np.exp(-kappa_lambda * column), dx) / denominator)
        out[j] = 0.0 if tau >= 1.0 else -np.log(max(tau, 1e-300)) / column
    return out


def column_grid() -> NDArray[np.float64]:
    decades = np.log10(COLUMN_MAX_ATM_M / COLUMN_MIN_ATM_M)
    n = int(round(decades * COLUMNS_PER_DECADE)) + 1
    return COLUMN_MIN_ATM_M * 10.0 ** (np.arange(n, dtype=np.float64) / COLUMNS_PER_DECADE)


def sha256_of(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def keys_to_write() -> dict[str, dict[str, object]]:
    """``key -> {"response": ..., "provenance source": ...}`` for every band worth a table.

    Two kinds of key, and the distinction is the point of "bands are data": a **nominal** key is
    the registry's top-hat over a band's §12.1 range, which is what a scene gets when it names a
    band and no camera; a **response** key is one camera's own R(lambda). They differ by more
    than rounding -- ``insb_flame_window`` and ``mwir`` cover the same nominal band and disagree
    on CO2 by more than an order of magnitude, which is the whole reason the filter exists.
    """
    keys: dict[str, dict[str, object]] = {}
    for band in BAND_IDS:
        lo, hi = NOMINAL_RANGES_UM[band]
        keys[band] = {
            "response": nominal_response(band),
            "kind": "nominal",
            "source": f"registry top-hat {lo:g}-{hi:g} um",
        }
    for path in sorted(RESPONSE_DIR.glob("*.csv")):
        keys[path.stem] = {
            "response": load_spectral_response(path),
            "kind": "response",
            "source": str(path),
        }
    return keys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=pathlib.Path, default=OUT_DIR)
    parser.add_argument("--only", action="append", help="one key; repeatable")
    args = parser.parse_args(argv)

    n = int(round((GAS_T_MAX_K - GAS_T_MIN_K) / TEMPERATURE_STEP_K)) + 1
    temperatures = GAS_T_MIN_K + TEMPERATURE_STEP_K * np.arange(n, dtype=np.float64)
    columns = column_grid()

    radcal_dir = pathlib.Path("data/spectra/radcal")
    database = {str(path): sha256_of(path) for path in sorted(radcal_dir.glob("*.csv"))}
    if not database:
        raise SystemExit("no RadCal tables: run scripts/fetch_radcal_tables.py first")

    every = keys_to_write()
    wanted = args.only or sorted(every)
    for key in wanted:
        if key not in every:
            raise SystemExit(f"unknown key {key!r}; have {', '.join(sorted(every))}")

    grid = quadrature_wavelengths()
    dx = float(grid[1] - grid[0])
    shapes = {key: every[key]["response"].resampled(grid) for key in wanted}  # type: ignore[union-attr]
    masks = {name: support_mask(grid, name) for name in SPECIES}
    kappa = {
        key: {name: np.zeros((temperatures.shape[0], columns.shape[0])) for name in SPECIES}
        for key in wanted
    }
    support = {key: dict.fromkeys(SPECIES, 1.0) for key in wanted}

    # Temperature outermost: kappa(lambda, T) is the expensive part and every band shares it.
    for i, temp in enumerate(temperatures):
        spectral = spectral_kappa(grid, float(temp))
        planck = spectral_radiance(grid, np.float64(temp))
        for key in wanted:
            weights = shapes[key] * planck
            total = simpson(weights, dx)
            for name in SPECIES:
                kappa[key][name][i] = effective_kappa(weights, spectral[name], columns, dx)
                covered = float(simpson(np.where(masks[name], weights, 0.0), dx) / total)
                support[key][name] = min(support[key][name], covered)

    args.out.mkdir(parents=True, exist_ok=True)
    for key in wanted:
        entry = every[key]
        files = {}
        for name in SPECIES:
            table = np.ascontiguousarray(kappa[key][name], dtype=np.float32)
            path = args.out / f"{key}_{name}.npy"
            np.save(path, table, allow_pickle=False)
            files[name] = {"file": path.name, "sha256": sha256_of(path)}
        sidecar = {
            "band": key,
            "kind": entry["kind"],
            "columns_atm_m": [float(x) for x in columns],
            "database": database,
            "dtype": "float32",
            "files": files,
            "generated": datetime.date.today().isoformat(),
            "generator": "scripts/generate_gas_luts.py",
            "model": "RadCal weak-line coefficients (NIST TN 1402) via scripts/radcal.py",
            "quantity": "effective band absorption coefficient, -ln<exp(-kappa X)>_RB / X",
            "response_source": entry["source"],
            "support_fraction": support[key],
            "temperatures_k": [float(t) for t in temperatures],
            "units": "1/(m.atm) of partial pressure",
            "weighting": "energy (R . B_lambda)",
        }
        out = args.out / f"{key}_gas.json"
        out.write_text(json.dumps(sidecar, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        thin = {name: kappa[key][name][:, 0] for name in SPECIES}
        print(
            f"{out}: thin-limit CO2 {thin['co2'].min():.4g}-{thin['co2'].max():.4g}, "
            f"H2O {thin['h2o'].min():.4g}-{thin['h2o'].max():.4g} 1/(m.atm); "
            f"support CO2 {support[key]['co2']:.3f} H2O {support[key]['h2o']:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
