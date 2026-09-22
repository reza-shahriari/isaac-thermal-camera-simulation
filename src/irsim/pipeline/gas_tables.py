"""Reading the committed hot-gas absorption tables (roadmap PH.5).

`PH.4` fixed that :class:`~irsim.pipeline.gas_slab.GasBandTables` is *data a pipeline reads*, never
a spectral model it evaluates: a line-by-line result at flame temperature costs seconds per band
and belongs offline, and an ambient-temperature coefficient scaled up is wrong by the hot bands it
does not contain. This module is the reading half. `scripts/generate_gas_luts.py` is the writing
half, and its docstring carries the derivation.

A table is a pair of float32 arrays, ``data/gas/<key>_{co2,h2o}.npy`` of shape
``(n_temperature, n_column)``, plus the sidecar ``data/gas/<key>_gas.json`` that holds both grids,
the provenance, and a SHA-256 of each array. The layout deliberately mirrors ``data/lut`` -- the
band LUTs already established that an array on disk needs a sidecar that says what it is, and the
hash is what makes "regenerate and diff" a check rather than a hope.

**Two kinds of key.** A *nominal* key is a registry band (``lwir``, ``mwir``, ...) integrated
against the §12.1 top-hat; a *response* key is one camera's own R(λ), named after its response
file. :func:`gas_tables_for` prefers the camera's when the camera has one, because in this
particular corner of the physics the difference is not a rounding: ``insb`` and
``insb_flame_window`` are both MWIR and disagree on CO2 by twelve orders of magnitude, which is
exactly what a through-flame filter is bought for.

**A band the model cannot speak about is refused, not zeroed.** RadCal truncates -- CO2 above
5725 cm⁻¹ (1.75 µm) and H2O above 9300 cm⁻¹ (1.08 µm) are *set* to zero, which is sound for fire
heat transfer and unsound for a SWIR or NIR camera, where real overtone bands live. The generator
records the R·B-weighted fraction of each band that falls inside the model's support and this
loader refuses anything below :data:`MIN_SUPPORT_FRACTION`, so a NIR flame raises a message that
names HITEMP rather than rendering as a transparent one.

docs/physics-model.md §8.1, §8.3; ADR 0098.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
from typing import Any

import numpy as np

from irsim.config.loader import resolve_data_dir
from irsim.pipeline.gas_slab import GasBandTables, SpeciesAbsorption
from irsim.radiometry.spectral_response import SpectralResponse

__all__ = [
    "GAS_TABLE_DIRNAME",
    "MIN_SUPPORT_FRACTION",
    "SPECIES",
    "available_gas_tables",
    "gas_tables_for",
    "load_gas_tables",
]

GAS_TABLE_DIRNAME = "gas"
SPECIES = ("co2", "h2o")

#: How much of a band's emission-weighted response must fall inside the absorption model's own
#: spectral support before its table may be used. Just under 1: the generator's quadrature grid
#: puts a fraction of a percent of the weight on the wrong side of a support edge, and that is
#: arithmetic, not missing physics. A band that is genuinely outside -- NIR at 0.00, SWIR at 0.78
#: on the committed tables -- is refused.
MIN_SUPPORT_FRACTION = 0.99


def _root(data_dir: str | pathlib.Path | None) -> pathlib.Path:
    return resolve_data_dir(data_dir) / GAS_TABLE_DIRNAME


def available_gas_tables(data_dir: str | pathlib.Path | None = None) -> tuple[str, ...]:
    """Every key with a committed sidecar, sorted."""
    root = _root(data_dir)
    return tuple(sorted(p.name[: -len("_gas.json")] for p in root.glob("*_gas.json")))


def _read_sidecar(path: pathlib.Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(
            f"no hot-gas absorption table at {path}; generate one with "
            "`python scripts/generate_gas_luts.py` (roadmap PH.5)"
        )
    meta: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    for field in ("temperatures_k", "columns_atm_m", "files", "database", "support_fraction"):
        if field not in meta:
            raise ValueError(
                f"{path}: the sidecar has no {field!r}. A table without provenance is refused "
                "(ADR 0041): nobody can tell which line database produced it."
            )
    if not meta["database"]:
        raise ValueError(f"{path}: the sidecar names no source database")
    return meta


def load_gas_tables(
    key: str,
    data_dir: str | pathlib.Path | None = None,
    *,
    allow_partial_support: bool = False,
) -> GasBandTables:
    """Read ``data/gas/<key>_*.npy`` into a :class:`GasBandTables`, hashes and support checked."""
    root = _root(data_dir)
    meta = _read_sidecar(root / f"{key}_gas.json")
    temperatures = np.asarray(meta["temperatures_k"], dtype=np.float64)
    columns = np.asarray(meta["columns_atm_m"], dtype=np.float64)
    support = meta["support_fraction"]

    species: dict[str, SpeciesAbsorption] = {}
    for name in SPECIES:
        entry = meta["files"].get(name)
        if entry is None:
            continue
        fraction = float(support.get(name, 0.0))
        if fraction < MIN_SUPPORT_FRACTION and not allow_partial_support:
            continue
        path = root / entry["file"]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != entry["sha256"]:
            raise ValueError(
                f"{path}: sha256 {digest[:12]} does not match the sidecar's {entry['sha256'][:12]}."
                " Regenerate with scripts/generate_gas_luts.py rather than editing the array."
            )
        table = np.load(path, allow_pickle=False)
        if table.dtype == np.float16:
            raise TypeError(f"{path}: float16 absorption tables are refused (CLAUDE.md #2)")
        species[name] = SpeciesAbsorption(
            temperatures_k=temperatures,
            kappa_per_m_atm=np.asarray(table, dtype=np.float64),
            columns_atm_m=columns,
        )

    if not species:
        covered = ", ".join(
            f"{name.upper()} {float(support.get(name, 0.0)):.2f}" for name in SPECIES
        )
        raise ValueError(
            f"band {key!r} has no usable hot-gas table: the absorption model covers only "
            f"{covered} of its emission-weighted response, against a floor of "
            f"{MIN_SUPPORT_FRACTION:g}. RadCal stops at 1.75 µm for CO2 and 1.08 µm for H2O, so a "
            "short-wave band would be handed zeros that read as a transparent gas rather than as "
            "missing data. A line list that reaches there (HITEMP, roadmap open question 14) is "
            "what lifts this; pass allow_partial_support=True to accept the truncation knowingly."
        )
    return GasBandTables(band=key, species=species)


def gas_tables_for(
    band: str,
    response: SpectralResponse | None = None,
    data_dir: str | pathlib.Path | None = None,
    *,
    allow_partial_support: bool = False,
) -> GasBandTables:
    """The camera's own table if one was generated for its response, else the band's nominal one.

    ``response.source_path``'s stem is the key, which is why the generator writes one file per
    response curve: a camera and its table are matched by the file they already share, with no
    registry to keep in step.
    """
    keys = available_gas_tables(data_dir)
    if response is not None:
        stem = pathlib.Path(str(response.source_path)).stem
        if stem in keys:
            return load_gas_tables(stem, data_dir, allow_partial_support=allow_partial_support)
    return load_gas_tables(band, data_dir, allow_partial_support=allow_partial_support)
