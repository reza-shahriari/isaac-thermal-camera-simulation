"""The camera catalogue: short names, a user directory first, and files that never go stale.

docs/physics-model.md §12.2 (the sensor file); ADR 0159 (this); SC.32.

A camera is something a user *adds*, not something they edit the code for. Three rules make
that true:

* **Short names.** ``--sensor boson640`` resolves through ``configs/sensor_catalogue.yaml``
  (an alias table, data not code) to ``flir_boson_640_lwir.yaml``; a file's own stem works
  too (``--sensor flir_boson_640_lwir``), and so does any path.
* **A user directory, searched first.** ``$IRSIM_SENSOR_DIR`` (or ``user_dir=``) holds the
  user's own cameras and shadows the shipped ones by name, so a private ``boson640.yaml``
  wins over the catalogue's alias without touching the repository.
* **Files never go stale.** A sensor file written against any schema version this loader
  still reads (``MIN_SCHEMA_VERSION``..``SCHEMA_VERSION``) loads to the same camera it always
  described: every field added since is optional with a default that *is* the pre-existing
  behaviour, ``schema_version`` moves only for a breaking rename and then ships a migrator
  registered in :data:`MIGRATIONS`, and an ``extensions:`` block carries whatever a user wants
  to keep beside the camera, untouched and unvalidated. A typo in a real field is still refused
  (``extra="forbid"``): forward compatibility is not silence.
"""

from __future__ import annotations

import copy
import os
import pathlib
from collections.abc import Callable, Mapping
from typing import Any

import yaml

from irsim.config.sensor import MIN_SCHEMA_VERSION, SCHEMA_VERSION

__all__ = [
    "CATALOGUE_PATH",
    "MIGRATIONS",
    "SENSORS_DIR",
    "USER_DIR_ENV",
    "list_sensors",
    "load_aliases",
    "migrate_sensor_document",
    "resolve_sensor_path",
    "sensor_search_dirs",
]

SENSORS_DIR = pathlib.Path(__file__).resolve().parents[3] / "configs" / "sensors"
CATALOGUE_PATH = SENSORS_DIR.parent / "sensor_catalogue.yaml"
USER_DIR_ENV = "IRSIM_SENSOR_DIR"

#: ``{from_version: migrator}``: a migrator takes the document written at ``from_version`` and
#: returns one valid at ``from_version + 1``. Empty while every version since 8 has only added
#: optional fields; a breaking rename registers its step here and nowhere else (ADR 0159).
MIGRATIONS: dict[int, Callable[[dict[str, Any]], dict[str, Any]]] = {}


def sensor_search_dirs(user_dir: str | os.PathLike[str] | None = None) -> tuple[pathlib.Path, ...]:
    """The user's directory (argument, else ``$IRSIM_SENSOR_DIR``) first, the shipped one second."""
    dirs: list[pathlib.Path] = []
    chosen = user_dir if user_dir is not None else os.environ.get(USER_DIR_ENV)
    if chosen:
        dirs.append(pathlib.Path(chosen).expanduser())
    dirs.append(SENSORS_DIR)
    return tuple(dirs)


def load_aliases(path: str | os.PathLike[str] | None = None) -> dict[str, str]:
    """``{short name: file stem}`` from the catalogue; an absent catalogue is no aliases."""
    p = pathlib.Path(path) if path is not None else CATALOGUE_PATH
    if not p.exists():
        return {}
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    aliases = raw.get("aliases", {})
    if not isinstance(aliases, Mapping) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in aliases.items()
    ):
        raise ValueError(f"{p}: `aliases:` must map short names to file stems")
    return {str(k): str(v) for k, v in aliases.items()}


def list_sensors(user_dir: str | os.PathLike[str] | None = None) -> dict[str, pathlib.Path]:
    """Every camera a name resolves to: stems in the search directories, plus the aliases."""
    out: dict[str, pathlib.Path] = {}
    for d in reversed(sensor_search_dirs(user_dir)):  # the user's directory wins
        if d.is_dir():
            for f in sorted(d.glob("*.yaml")):
                out[f.stem] = f
    for short, stem in load_aliases().items():
        if stem in out:
            out.setdefault(short, out[stem])
    return dict(sorted(out.items()))


def resolve_sensor_path(
    name_or_path: str | os.PathLike[str], user_dir: str | os.PathLike[str] | None = None
) -> pathlib.Path:
    """A path as given if it exists; else a stem or alias, user directory first.

    A path that does not exist is retried by its stem, so ``<repo>/boson640`` still finds the
    Boson; an unknown name lists what would have worked.
    """
    p = pathlib.Path(name_or_path)
    if p.is_file():
        return p
    stem = p.stem if p.suffix == ".yaml" else p.name
    aliases = load_aliases()
    candidates = [stem]
    if stem in aliases:
        candidates.append(aliases[stem])
    for d in sensor_search_dirs(user_dir):
        for c in candidates:
            f = d / f"{c}.yaml"
            if f.is_file():
                return f
    known = ", ".join(sorted(list_sensors(user_dir)))
    raise FileNotFoundError(
        f"no sensor file at {name_or_path!r} and no camera named {stem!r} in "
        f"{[str(d) for d in sensor_search_dirs(user_dir)]}; known: {known}"
    )


def migrate_sensor_document(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Bring a parsed sensor file from its ``schema_version`` up to today's, step by step.

    Every version since ``MIN_SCHEMA_VERSION`` has only added optional fields, so today the
    walk changes nothing but the version number; the day a field is renamed, its migrator goes
    into :data:`MIGRATIONS` and every older file keeps loading. A version below the floor or
    above today's is refused here, as the schema refuses it.
    """
    doc = copy.deepcopy(dict(raw))
    version = int(doc.get("schema_version", SCHEMA_VERSION))
    if not MIN_SCHEMA_VERSION <= version <= SCHEMA_VERSION:
        raise ValueError(
            f"schema_version {version} outside the readable range "
            f"{MIN_SCHEMA_VERSION}-{SCHEMA_VERSION}"
        )
    while version < SCHEMA_VERSION:
        step = MIGRATIONS.get(version)
        if step is not None:
            doc = step(doc)
        version += 1
        doc["schema_version"] = version
    return doc
