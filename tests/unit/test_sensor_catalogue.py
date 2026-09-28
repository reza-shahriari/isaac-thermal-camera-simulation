"""SC.32 -- a camera catalogue anyone can extend, whose files never go stale.

Short names resolve through `configs/sensor_catalogue.yaml`; a user directory
(`$IRSIM_SENSOR_DIR`) is searched first and shadows the shipped cameras by name; a frozen
fixture per past schema version loads to the camera today's file describes; `extensions:`
carries user data untouched; a typo in a real field is still refused. ADR 0159; roadmap SC.32.
"""

from __future__ import annotations

import pathlib

import pytest
import yaml
from pydantic import ValidationError

from irsim.config import catalogue
from irsim.config.catalogue import (
    MIGRATIONS,
    SENSORS_DIR,
    list_sensors,
    load_aliases,
    migrate_sensor_document,
    resolve_sensor_path,
    sensor_search_dirs,
)
from irsim.config.loader import dump_sensor_config, load_sensor_config
from irsim.config.sensor import MIN_SCHEMA_VERSION, SCHEMA_VERSION, SensorConfig

REPO = pathlib.Path(__file__).resolve().parents[2]
FIXTURES = REPO / "tests" / "fixtures" / "sensors"
BOSON = SENSORS_DIR / "flir_boson_640_lwir.yaml"


# --- names ---------------------------------------------------------------------------------------


def test_short_names_resolve_to_the_shipped_files(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("IRSIM_SENSOR_DIR", raising=False)
    aliases = load_aliases()
    assert aliases["boson640"] == "flir_boson_640_lwir"
    for short, stem in aliases.items():
        assert resolve_sensor_path(short) == SENSORS_DIR / f"{stem}.yaml", short
        assert resolve_sensor_path(stem) == SENSORS_DIR / f"{stem}.yaml"
    assert resolve_sensor_path(BOSON) == BOSON, "a path is a path"
    assert resolve_sensor_path(REPO / "boson640") == BOSON, "a missing path is retried by its stem"
    assert resolve_sensor_path("flir_boson_640_lwir.yaml") == BOSON
    known = list_sensors()
    assert "boson640" in known and "flir_boson_640_lwir" in known


def test_an_unknown_camera_names_the_ones_that_exist(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("IRSIM_SENSOR_DIR", raising=False)
    with pytest.raises(FileNotFoundError, match="known: .*boson640"):
        resolve_sensor_path("boson9000")
    with pytest.raises(FileNotFoundError):
        load_sensor_config("boson9000")


def test_the_user_directory_is_searched_first_and_shadows_by_name(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mine = tmp_path / "cameras"
    mine.mkdir()
    doc = yaml.safe_load(BOSON.read_text())
    doc["sensor"]["name"] = "my_boson"
    (mine / "boson640.yaml").write_text(yaml.safe_dump(doc))  # shadows the alias by file name
    (mine / "garage_cam.yaml").write_text(yaml.safe_dump(doc))
    monkeypatch.setenv("IRSIM_SENSOR_DIR", str(mine))
    assert sensor_search_dirs()[0] == mine and sensor_search_dirs()[-1] == SENSORS_DIR
    assert resolve_sensor_path("boson640") == mine / "boson640.yaml"
    assert resolve_sensor_path("garage_cam") == mine / "garage_cam.yaml"
    assert resolve_sensor_path("halmstad_boson_320") == SENSORS_DIR / "halmstad_boson_320.yaml"
    # and the loader takes the name straight: a user camera renders by name
    cfg = load_sensor_config("garage_cam")
    assert cfg.sensor.name == "my_boson"
    assert "garage_cam" in list_sensors()
    monkeypatch.delenv("IRSIM_SENSOR_DIR")
    assert resolve_sensor_path("boson640") == BOSON, "without the directory, the catalogue's"


# --- files never go stale -------------------------------------------------------------------------


@pytest.mark.parametrize("version", list(range(MIN_SCHEMA_VERSION, SCHEMA_VERSION)))
def test_a_frozen_file_from_every_past_schema_version_loads_unchanged(version: int) -> None:
    """The Boson as written at v8..v11 is the Boson today, minus the one field v12 added
    (`bad_pixel_late_fraction`, which defaults to the pre-v12 camera)."""
    frozen = load_sensor_config(FIXTURES / f"boson_v{version}.yaml")
    today = yaml.safe_load(BOSON.read_text())
    del today["sensor"]["noise"]["bad_pixel_late_fraction"]
    expect = SensorConfig.model_validate(_resolved(today))
    assert frozen.schema_version == SCHEMA_VERSION, "migrated up to today on load"
    assert frozen.sensor == expect.sensor


def _resolved(doc: dict) -> dict:  # type: ignore[type-arg]
    """The document with its data paths resolved the way the loader resolves them."""
    from irsim.config.loader import _resolve_paths, resolve_data_dir

    _resolve_paths(doc["sensor"], resolve_data_dir(None))
    return doc


def test_the_migrator_chain_walks_every_step_and_refuses_the_unreadable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    doc = {"schema_version": MIN_SCHEMA_VERSION, "sensor": {"old_name": 1}}
    walked = migrate_sensor_document(doc)
    assert walked["schema_version"] == SCHEMA_VERSION and walked["sensor"] == {"old_name": 1}
    assert doc["schema_version"] == MIN_SCHEMA_VERSION, "the input is not mutated"

    def rename(d: dict) -> dict:  # type: ignore[type-arg]
        d = dict(d)
        d["sensor"] = {"new_name": d["sensor"].pop("old_name")}
        return d

    monkeypatch.setitem(MIGRATIONS, MIN_SCHEMA_VERSION, rename)
    assert migrate_sensor_document(doc)["sensor"] == {"new_name": 1}
    later = {"schema_version": MIN_SCHEMA_VERSION + 1, "sensor": {"old_name": 1}}
    assert migrate_sensor_document(later)["sensor"] == {"old_name": 1}, "a step applies once"
    for bad in (MIN_SCHEMA_VERSION - 1, SCHEMA_VERSION + 1):
        with pytest.raises(ValueError, match="outside the readable range"):
            migrate_sensor_document({"schema_version": bad, "sensor": {}})


def test_extensions_carry_user_data_untouched_and_a_typo_is_still_refused(
    tmp_path: pathlib.Path,
) -> None:
    doc = yaml.safe_load(BOSON.read_text())
    doc["sensor"]["extensions"] = {"my_tool": {"serial": "B640-0042", "mount": [1, 2, 3]}}
    path = tmp_path / "with_extensions.yaml"
    path.write_text(yaml.safe_dump(doc))
    cfg = load_sensor_config(path)
    assert cfg.sensor.extensions == {"my_tool": {"serial": "B640-0042", "mount": [1, 2, 3]}}
    again = SensorConfig.model_validate(yaml.safe_load(dump_sensor_config(cfg)))
    assert again.sensor.extensions == cfg.sensor.extensions
    plain = load_sensor_config(BOSON)
    assert plain.sensor.extensions == {}
    typo = yaml.safe_load(BOSON.read_text())
    typo["sensor"]["optics"]["f_numbr"] = 1.0
    path.write_text(yaml.safe_dump(typo))
    with pytest.raises(ValidationError):
        load_sensor_config(path)


def test_the_catalogue_module_documents_the_rule() -> None:
    assert "schema_version" in (catalogue.__doc__ or "") and "extensions" in (
        catalogue.__doc__ or ""
    )


def test_extensions_do_not_move_the_config_hash(tmp_path: pathlib.Path) -> None:
    """User data beside the camera is not the camera: a serial number must stale no golden."""
    from irsim.config.loader import config_hash

    plain = load_sensor_config(BOSON)
    doc = yaml.safe_load(BOSON.read_text())
    doc["sensor"]["extensions"] = {"serial": "B640-0042"}
    path = tmp_path / "tagged.yaml"
    path.write_text(yaml.safe_dump(doc))
    tagged = load_sensor_config(path)
    assert tagged.sensor.extensions and config_hash(tagged) == config_hash(plain)
