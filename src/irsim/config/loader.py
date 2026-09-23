"""YAML loading, data-root path resolution and canonical hashing for sensor configs.

Two hashes key everything that is derived from a config (docs/physics-model.md §3.2 b; the
ir-radiometry and ir-sim-testing skills):

* ``config_hash`` -- SHA-256 of the canonical JSON of the *validated* model, with every data
  file path replaced by the SHA-256 of that file's bytes. Comments, key order, whitespace and
  ``60`` vs ``60.0`` do not change it; every numeric or enumerated leaf does. Goldens and
  pipeline outputs are keyed on this.
* ``band_hash`` -- the same over the ``band`` block and the spectral-response bytes only. This
  is what keys a band LUT, so the three NETD grades of one detector (§16.1) share one LUT.

Data files referenced by a config (``band.spectral_response``; later material n/k and
emissivity files) are relative to a data root: the ``data_dir`` argument, else
``$IRSIM_DATA_DIR``, else ``<repo>/data``. The canonical layout is ``data/spectra/responses/``
for R(λ) files (spec issue T4, fixed in §12.2 by M0.10). Loading stores the resolved absolute
path on the model and fails by name if the file is missing.

docs/physics-model.md §12.2, §3.2 (b)
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
from typing import Any

import yaml

from irsim.config.sensor import FULL_FIDELITY, FocusSpec, SensorConfig

__all__ = [
    "DEFAULT_DATA_DIR",
    "DATA_PATH_FIELDS",
    "resolve_data_dir",
    "load_sensor_config",
    "dump_sensor_config",
    "with_integration_time_ms",
    "config_hash",
    "band_hash",
    "file_sha256",
]

DEFAULT_DATA_DIR = pathlib.Path(__file__).resolve().parents[3] / "data"
# Dotted paths (under ``sensor``) of fields that name data files. Extend here when material or
# atmosphere configs add file references; the hashes pick them up automatically.
DATA_PATH_FIELDS: tuple[str, ...] = ("band.spectral_response",)


def resolve_data_dir(data_dir: str | os.PathLike[str] | None = None) -> pathlib.Path:
    """Argument, else ``$IRSIM_DATA_DIR``, else ``<repo>/data``."""
    if data_dir is not None:
        return pathlib.Path(data_dir).expanduser().resolve()
    env = os.environ.get("IRSIM_DATA_DIR")
    if env:
        return pathlib.Path(env).expanduser().resolve()
    return DEFAULT_DATA_DIR


def _get(d: dict[str, Any], dotted: str) -> Any:
    node: Any = d
    for key in dotted.split("."):
        node = node[key]
    return node


def _set(d: dict[str, Any], dotted: str, value: Any) -> None:
    *path, last = dotted.split(".")
    node = d
    for key in path:
        node = node[key]
    node[last] = value


def _resolve_paths(sensor: dict[str, Any], data_dir: pathlib.Path) -> None:
    for field in DATA_PATH_FIELDS:
        raw = _get(sensor, field)
        path = pathlib.Path(raw).expanduser()
        if not path.is_absolute():
            path = data_dir / path
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(
                f"sensor.{field} = {raw!r} resolves to {path}, which does not exist "
                f"(data root {data_dir}; override with data_dir= or $IRSIM_DATA_DIR)"
            )
        _set(sensor, field, str(path))


def load_sensor_config(
    path: str | os.PathLike[str], data_dir: str | os.PathLike[str] | None = None
) -> SensorConfig:
    """Read a §12.2 YAML file, resolve its data paths against the data root, validate."""
    path = pathlib.Path(path)
    with path.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict) or "sensor" not in raw:
        raise ValueError(f"{path}: a sensor config needs a top-level 'sensor:' block")
    if not isinstance(raw["sensor"], dict):
        raise ValueError(f"{path}: 'sensor:' must be a mapping")
    _resolve_paths(raw["sensor"], resolve_data_dir(data_dir))
    return SensorConfig.model_validate(raw)


def dump_sensor_config(config: SensorConfig) -> str:
    """YAML text that :func:`load_sensor_config` reads back to an equal model."""
    return yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False)


def with_integration_time_ms(config: SensorConfig, integration_ms: float) -> SensorConfig:
    """A copy of ``config`` whose photon FPA integrates for ``integration_ms`` milliseconds.

    docs/physics-model.md §8.1 (photon FPA), §12.2; ADR 0021.

    A camera has an exposure control and these configs carry one default each, chosen for the
    conditions the band is usually used in. A daylight reflective-band scene saturates a low-light
    exposure by around 120x -- measured: a 0.3-albedo surface in full sun puts 1.17e6
    photoelectrons into a NIR pixel in 16 ms against a 1e4 well -- so a sweep that films one scene
    in four bands has to be able to set it.

    It goes through the model rather than through the YAML so the change reaches
    :func:`config_hash`, which is the point: two renders at two exposures are two cameras, and a
    frame whose hash did not move would claim otherwise.

    A bolometer is refused rather than silently ignored. Its responsivity is thermal and it has no
    integration time to set (§8.2), so accepting the flag would hand back a config that reads as
    exposed and is not -- the failure this project's hash discipline exists to make impossible.
    """
    dumped = config.model_dump(mode="json")
    kind = dumped["sensor"]["fpa"]["type"]
    if kind != "photon":
        raise ValueError(
            f"integration time applies to a photon FPA; this is a {kind}. A bolometer's "
            "responsivity is thermal and it has no exposure to set (docs/physics-model.md §8.2)"
        )
    dumped["sensor"]["fpa"]["integration_time_ms"] = float(integration_ms)
    return SensorConfig.model_validate(dumped)


def file_sha256(path: str | os.PathLike[str]) -> str:
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)


#: `OC.4`: the focus block a pre-v10 document implies, dropped from the hash when it is unchanged.
DEFAULT_FOCUS = FocusSpec().model_dump(mode="json")


def _dump_with_file_hashes(
    config: SensorConfig, data_dir: str | os.PathLike[str] | None
) -> dict[str, Any]:
    """Model dump with each data-file path replaced by {"sha256": <content hash>}.

    A model built from a dictionary (not the loader) may still hold a relative path; it is
    resolved against the data root here so hashing does not depend on how the model was made.
    """
    dumped = config.model_dump(mode="json")
    # `schema_version` describes the *document format*, not the camera, so it is not part of the
    # hash: a v8 file and a v9 file that describe the same sensor are the same input and must
    # produce the same reference. The guard against a future version changing meaning without
    # changing fields is `MIN_SCHEMA_VERSION`, which refuses such a document outright rather than
    # letting it hash the same as a newer one.
    del dumped["schema_version"]
    sensor = dumped["sensor"]
    # ME.8: full fidelity is the identity, and the identity is not recorded. Dropping the default
    # `fidelity:` block keeps a v8 file and a v9 file that spells every switch out as `true`
    # hashing the same -- the hash tracks the *ablation*, not the notation -- and leaves every
    # golden written before ME.8 valid. Any switch turned off survives this and changes the hash,
    # which is the whole point of the block.
    if sensor.get("fidelity") == FULL_FIDELITY.model_dump(mode="json"):
        del sensor["fidelity"]
    # `OC.4`, same rule for the same reason: a v9 document and a v10 document that spell out
    # `focus: {mode: infinity}` and `defocus_model: none` describe one camera -- the perfectly
    # focused one every render before v10 used -- so they must hash the same, and every golden
    # written before `OC` stays valid. A focus distance or a named model survives this and changes
    # the hash, which is the point.
    optics = sensor.get("optics", {})
    if optics.get("focus") == DEFAULT_FOCUS:
        del optics["focus"]
    mtf = optics.get("mtf", {})
    for key, default in (("defocus_model", "none"), ("defocus_apply", "global")):
        if mtf.get(key) == default:
            mtf.pop(key, None)
    # `OC.10`, same rule again: an athermal lens is the pre-v11 camera, and the two material names
    # and the reference temperature describe nothing while it is athermal.
    v11_defaults: tuple[tuple[str, Any], ...] = (
        ("athermal", True),
        ("lens_material", "germanium"),
        ("housing_material", "aluminium"),
        ("focus_reference_temp_k", 293.15),
    )
    for key, default in v11_defaults:
        if optics.get(key) == default:
            optics.pop(key, None)
    root = resolve_data_dir(data_dir)
    for field in DATA_PATH_FIELDS:
        raw = _get(sensor, field)
        path = pathlib.Path(raw).expanduser()
        if not path.is_absolute():
            path = root / path
        if not path.is_file():
            raise FileNotFoundError(f"sensor.{field} = {raw!r}: {path} does not exist")
        _set(sensor, field, {"sha256": file_sha256(path)})
    return dumped


def config_hash(config: SensorConfig, data_dir: str | os.PathLike[str] | None = None) -> str:
    """SHA-256 over the whole validated config; data files by content, not by path."""
    return hashlib.sha256(_canonical(_dump_with_file_hashes(config, data_dir)).encode()).hexdigest()


def band_hash(config: SensorConfig, data_dir: str | os.PathLike[str] | None = None) -> str:
    """SHA-256 over ``sensor.band`` and the spectral-response bytes only -- the LUT key."""
    dumped = _dump_with_file_hashes(config, data_dir)
    return hashlib.sha256(_canonical(dumped["sensor"]["band"]).encode()).hexdigest()
