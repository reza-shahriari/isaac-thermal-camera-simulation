"""Band LUT bundles on disk: float32 tables, raw binary export, and a hash sidecar.

A bundle for one band lives in ``<out_dir>/`` as

    <key>_{lb,lb_q,dlb_dt,dlb_q_dt}.npy   float32, shape (N,)     -- NumPy consumers
    <key>_{lb,lb_q,dlb_dt,dlb_q_dt}.f32   raw little-endian float32 -- Lua / SPG / Unreal
                                                                       (PF_R32_FLOAT) consumers
    <key>_lut.json                        sidecar

where ``<key>`` is the first 16 hex digits of the band hash (ADR 0008). The sidecar records the
grid (T0, T1, N, dt), dtype, the band block, the full band hash, the config hash, the SHA-256 of
the spectral-response file and the generator versions. Loading **recomputes** the band hash and
spectral SHA-256 from the current config and data and raises :class:`StaleLUTError` naming what
changed, so a table can never be silently reused after its inputs moved (§3.2 b, §13.5).

Policy (ADR 0012): bundles are **generated, not committed** (``make luts``; ``data/lut/`` is
gitignored); regeneration is deterministic; the committed regression is the golden LUT slice.

docs/physics-model.md §3.2 (b), §13.5, §14; CLAUDE.md non-negotiable #2
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
from dataclasses import dataclass
from typing import Any

import numpy as np

import irsim
from irsim.config.loader import band_hash as compute_band_hash
from irsim.config.loader import config_hash as compute_config_hash
from irsim.config.loader import file_sha256, resolve_data_dir
from irsim.config.sensor import SensorConfig
from irsim.radiometry.lut import QUANTITIES, BandLUT
from irsim.radiometry.spectral_response import SpectralResponse, load_spectral_response

__all__ = [
    "builder_sha256",
    "LUT_SCHEMA_VERSION",
    "StaleLUTError",
    "LUTPaths",
    "lut_paths",
    "save_band_lut",
    "load_band_lut",
    "build_band_lut_for_config",
    "load_band_lut_for_config",
    "load_band_response_for_config",
]

LUT_SCHEMA_VERSION = 1
KEY_LENGTH = 16


class StaleLUTError(RuntimeError):
    """The bundle on disk was built from inputs that differ from the current config/data."""


@dataclass(frozen=True)
class LUTPaths:
    npy: dict[str, pathlib.Path]
    f32: dict[str, pathlib.Path]
    sidecar: pathlib.Path


def lut_paths(band_hash: str, out_dir: str | os.PathLike[str]) -> LUTPaths:
    root = pathlib.Path(out_dir)
    key = band_hash[:KEY_LENGTH]
    return LUTPaths(
        npy={q: root / f"{key}_{q}.npy" for q in QUANTITIES},
        f32={q: root / f"{key}_{q}.f32" for q in QUANTITIES},
        sidecar=root / f"{key}_lut.json",
    )


def builder_sha256() -> str:
    """SHA-256 over the code that builds a table: the quadrature grid rule, the Planck forms and
    the constants (GT.11).

    `make luts` for AT.24 moved a gitignored MWIR bundle by a uniform 20 % although its grid was
    unchanged: an older builder had made it, the config and spectral hashes still matched, and a
    golden was recorded against it. A bundle now carries the hash of the modules whose source
    decides every number in it, and a bundle whose builder differs is refused with the remedy.
    Source text rather than a version: a version is bumped by hand, and forgotten.
    """
    import inspect

    from irsim.radiometry import band_integration, constants, planck

    digest = hashlib.sha256()
    for module in (band_integration, planck, constants):
        digest.update(inspect.getsource(module).encode("utf-8"))
    digest.update(f"lut_schema={LUT_SCHEMA_VERSION}".encode())
    return digest.hexdigest()


def save_band_lut(
    lut: BandLUT,
    out_dir: str | os.PathLike[str],
    *,
    band_hash: str,
    config_sha256: str,
    spectral_sha256: str,
    sensor_name: str,
    band: dict[str, Any],
) -> LUTPaths:
    """Write the four tables (``.npy`` and raw ``.f32``) and the sidecar. float32 asserted."""
    paths = lut_paths(band_hash, out_dir)
    paths.sidecar.parent.mkdir(parents=True, exist_ok=True)
    for q in QUANTITIES:
        table = lut.table(q)
        if table.dtype != np.float32:
            raise TypeError(f"{q} is {table.dtype}; LUT files are float32 (non-negotiable #2)")
        np.save(paths.npy[q], table, allow_pickle=False)
        paths.f32[q].write_bytes(table.astype("<f4").tobytes())
    sidecar = {
        "schema_version": LUT_SCHEMA_VERSION,
        "sensor_name": sensor_name,
        "band": band,
        "band_hash": band_hash,
        "config_sha256": config_sha256,
        "spectral_sha256": spectral_sha256,
        "t0_k": lut.t0_k,
        "t1_k": lut.t1_k,
        "n": lut.n,
        "dt_k": lut.dt_k,
        "dtype": "float32",
        "quantities": list(QUANTITIES),
        "files": {q: paths.npy[q].name for q in QUANTITIES},
        "raw_files": {q: paths.f32[q].name for q in QUANTITIES},
        "raw_layout": "little-endian float32, N entries, index i <-> T = t0_k + i * dt_k",
        "numpy_version": np.__version__,
        "irsim_version": irsim.__version__,
        "builder_sha256": builder_sha256(),
    }
    paths.sidecar.write_text(json.dumps(sidecar, indent=2, sort_keys=True) + "\n")
    return paths


def _read_sidecar(path: pathlib.Path) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(path.read_text())
    if data.get("schema_version") != LUT_SCHEMA_VERSION:
        raise StaleLUTError(
            f"{path}: sidecar schema_version {data.get('schema_version')} != {LUT_SCHEMA_VERSION}; "
            "regenerate with `make luts`"
        )
    built_by = data.get("builder_sha256")
    if built_by != builder_sha256():
        raise StaleLUTError(
            f"{path}: built by a different builder ({(built_by or 'unrecorded')[:12]}… against "
            f"{builder_sha256()[:12]}…): the quadrature grid, the Planck forms or the constants "
            "changed since, so its numbers are not today's. Regenerate with `make luts` (GT.11)"
        )
    return data


def load_band_lut(
    out_dir: str | os.PathLike[str],
    band_hash: str,
    *,
    spectral_sha256: str | None = None,
    config_sha256: str | None = None,
) -> BandLUT:
    """Load the bundle for ``band_hash``; verify the sidecar against the hashes given."""
    paths = lut_paths(band_hash, out_dir)
    if not paths.sidecar.is_file():
        raise FileNotFoundError(f"no LUT bundle {paths.sidecar}; generate it with `make luts`")
    meta = _read_sidecar(paths.sidecar)
    if meta["band_hash"] != band_hash:
        raise StaleLUTError(
            f"{paths.sidecar}: band_hash {meta['band_hash'][:12]}… != {band_hash[:12]}…"
        )
    if spectral_sha256 is not None and meta["spectral_sha256"] != spectral_sha256:
        raise StaleLUTError(
            f"{paths.sidecar}: the spectral response file changed since this LUT was built "
            f"(sha {meta['spectral_sha256'][:12]}… -> {spectral_sha256[:12]}…); run `make luts`"
        )
    if config_sha256 is not None and meta["config_sha256"] != config_sha256:
        # same band, different sensor config (e.g. a NETD grade): the LUT is still valid.
        pass
    tables: dict[str, Any] = {}
    for q in QUANTITIES:
        arr = np.load(paths.npy[q], allow_pickle=False)
        if arr.dtype != np.float32:
            raise TypeError(f"{paths.npy[q]}: dtype {arr.dtype}; LUT tables must be float32")
        if arr.shape != (meta["n"],):
            raise StaleLUTError(f"{paths.npy[q]}: shape {arr.shape} != sidecar n = {meta['n']}")
        tables[q] = arr
    return BandLUT(t0_k=meta["t0_k"], t1_k=meta["t1_k"], n=meta["n"], band_hash=band_hash, **tables)


def _band_block(config: SensorConfig) -> dict[str, Any]:
    b = config.sensor.band
    return {
        "id": b.band_id,
        "lambda_min_um": b.lambda_min_um,
        "lambda_max_um": b.lambda_max_um,
        "regime": b.regime,
        "spectral_response": pathlib.Path(b.spectral_response).name,
    }


def _spectral_path(config: SensorConfig, data_dir: str | os.PathLike[str] | None) -> pathlib.Path:
    p = pathlib.Path(config.sensor.band.spectral_response)
    return p if p.is_absolute() else resolve_data_dir(data_dir) / p


def load_band_response_for_config(
    config: SensorConfig, data_dir: str | os.PathLike[str] | None = None
) -> SpectralResponse:
    """The camera's own R(λ), from the same path the LUT was built against (AT.2).

    It lives beside :func:`load_band_lut_for_config` because the two must describe one camera. The
    layered atmosphere needs the response to split a band into spectral classes, and without it
    ``class_weights`` falls back to a nominal top-hat: measured on the shipped InSb response, the
    MWIR ``h2o_wing`` class weight goes 0.0238 to 0.1303, a **5.5x** change, with nothing said.
    """
    return load_spectral_response(_spectral_path(config, data_dir))


def build_band_lut_for_config(
    config: SensorConfig,
    out_dir: str | os.PathLike[str],
    data_dir: str | os.PathLike[str] | None = None,
) -> tuple[BandLUT, LUTPaths]:
    """Build the LUT for a validated config and write its bundle. Returns (lut, paths)."""
    from irsim.radiometry.band import Band

    band = Band.from_spec(config.sensor.band, _spectral_path(config, data_dir))
    bh = compute_band_hash(config, data_dir)
    lut = BandLUT.build(band.response, band_hash=bh)
    paths = save_band_lut(
        lut,
        out_dir,
        band_hash=bh,
        config_sha256=compute_config_hash(config, data_dir),
        spectral_sha256=band.response.sha256,
        sensor_name=config.sensor.name,
        band=_band_block(config),
    )
    return lut, paths


def load_band_lut_for_config(
    config: SensorConfig,
    out_dir: str | os.PathLike[str],
    data_dir: str | os.PathLike[str] | None = None,
) -> BandLUT:
    """Load the bundle a config needs, recomputing its hashes from the current config and data.

    If no bundle exists for the current band hash but one exists for this sensor name, the
    inputs moved since it was built: :class:`StaleLUTError` says which (band block vs spectral
    file). If none exists at all: ``FileNotFoundError`` with the ``make luts`` remedy.
    """
    bh = compute_band_hash(config, data_dir)
    spectral_sha = file_sha256(_spectral_path(config, data_dir))
    paths = lut_paths(bh, out_dir)
    if paths.sidecar.is_file():
        return load_band_lut(out_dir, bh, spectral_sha256=spectral_sha)
    root = pathlib.Path(out_dir)
    for sidecar in sorted(root.glob("*_lut.json")) if root.is_dir() else []:
        meta = json.loads(sidecar.read_text())
        if meta.get("sensor_name") != config.sensor.name:
            continue
        if meta.get("spectral_sha256") != spectral_sha:
            what = f"the spectral response file {_band_block(config)['spectral_response']} changed"
        else:
            what = "the band block of the sensor config changed"
        raise StaleLUTError(
            f"LUT bundle {sidecar.name} for {config.sensor.name!r} is stale: {what} since it was "
            f"built (band_hash {meta.get('band_hash', '')[:12]}… -> {bh[:12]}…). Run `make luts`."
        )
    raise FileNotFoundError(
        f"no LUT bundle for {config.sensor.name!r} (band_hash {bh[:12]}…) in {root}; "
        "run `make luts`"
    )
