"""Loader, data-root resolution and the two hashes (ADR 0008).

The hash tests are the important ones: a field the hash ignores is a field whose change would
silently reuse a stale LUT or golden. So every numeric leaf is perturbed and every enumerated
leaf flipped, and the hash must move each time.
"""

from __future__ import annotations

import copy
import pathlib
import shutil
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from irsim.config.loader import (
    DEFAULT_DATA_DIR,
    band_hash,
    config_hash,
    dump_sensor_config,
    load_sensor_config,
    resolve_data_dir,
    with_integration_time_ms,
)
from irsim.config.sensor import SensorConfig

REPO = pathlib.Path(__file__).resolve().parents[2]
BOSON_YAML = REPO / "configs" / "sensors" / "flir_boson_640_lwir.yaml"
RESPONSE_REL = pathlib.Path("spectra/responses/boson_vox.csv")

TOPHAT_CSV = (
    "# synthetic top-hat for loader tests\nwavelength_um,response\n"
    "7.0,0.0\n7.5,1.0\n13.5,1.0\n14.0,0.0\n"
)


@pytest.fixture
def data_dir(tmp_path: pathlib.Path) -> pathlib.Path:
    root = tmp_path / "data"
    (root / RESPONSE_REL).parent.mkdir(parents=True)
    (root / RESPONSE_REL).write_text(TOPHAT_CSV)
    return root


@pytest.fixture
def boson_text() -> str:
    return BOSON_YAML.read_text()


def _load_text(text: str, where: pathlib.Path, data_dir: pathlib.Path) -> SensorConfig:
    where.write_text(text)
    return load_sensor_config(where, data_dir)


def test_loads_repo_yaml_and_resolves_path(data_dir: pathlib.Path) -> None:
    cfg = load_sensor_config(BOSON_YAML, data_dir)
    resolved = pathlib.Path(cfg.sensor.band.spectral_response)
    assert resolved.is_absolute() and resolved == (data_dir / RESPONSE_REL).resolve()


def test_missing_spectral_file_errors_naming_path(tmp_path: pathlib.Path) -> None:
    empty = tmp_path / "nodata"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match=str(RESPONSE_REL)):
        load_sensor_config(BOSON_YAML, empty)


def test_repo_yaml_loads_against_the_default_data_root() -> None:
    """The committed YAML resolves to data/spectra/responses/boson_vox.csv (authored in M1.3)."""
    target = DEFAULT_DATA_DIR / RESPONSE_REL
    assert target.is_file(), f"{target} missing: the Boson response file is part of the repo"
    assert load_sensor_config(BOSON_YAML).sensor.band.spectral_response == str(target)


def test_data_dir_precedence(
    data_dir: pathlib.Path, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("IRSIM_DATA_DIR", raising=False)
    assert resolve_data_dir() == DEFAULT_DATA_DIR
    monkeypatch.setenv("IRSIM_DATA_DIR", str(data_dir))
    assert resolve_data_dir() == data_dir.resolve()
    assert load_sensor_config(BOSON_YAML).sensor.band.spectral_response.startswith(str(data_dir))
    other = tmp_path / "other"
    assert resolve_data_dir(other) == other.resolve()


def test_top_level_sensor_block_required(tmp_path: pathlib.Path, data_dir: pathlib.Path) -> None:
    with pytest.raises(ValueError, match="sensor:"):
        _load_text("name: x\n", tmp_path / "bad.yaml", data_dir)


def test_hash_invariant_to_comments_whitespace_key_order(
    boson_text: str, tmp_path: pathlib.Path, data_dir: pathlib.Path
) -> None:
    base = _load_text(boson_text, tmp_path / "a.yaml", data_dir)
    stripped = "\n".join(line.split("#")[0].rstrip() for line in boson_text.splitlines())
    reordered = yaml.safe_dump(yaml.safe_load(boson_text), sort_keys=True)
    ints_as_floats = boson_text.replace("frame_rate_hz: 60", "frame_rate_hz: 60.0").replace(
        "ffc_interval_s: 300", "ffc_interval_s: 300.00"
    )
    for text in (stripped, reordered, ints_as_floats):
        other = _load_text(text, tmp_path / "b.yaml", data_dir)
        assert config_hash(other) == config_hash(base)
        assert band_hash(other) == band_hash(base)


def _numeric_leaves(d: dict[str, Any], prefix: str = "") -> list[str]:
    out: list[str] = []
    for k, v in d.items():
        dotted = f"{prefix}{k}"
        if isinstance(v, dict):
            out += _numeric_leaves(v, dotted + ".")
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            out.append(dotted)
        elif isinstance(v, list) and v and isinstance(v[0], (int, float)):
            out += [f"{dotted}[{i}]" for i in range(len(v))]
    return out


def _perturb(d: dict[str, Any], dotted: str, factor: float) -> dict[str, Any]:
    out = copy.deepcopy(d)
    node: Any = out
    parts = dotted.split(".")
    for key in parts[:-1]:
        node = node[key]
    last = parts[-1]
    if "[" in last:
        name, idx = last[:-1].split("[")
        v = node[name][int(idx)]
        node[name][int(idx)] = v * factor if v else 0.01
    else:
        v = node[last]
        # a zero float scaled is still zero, which would test nothing (the Boson's dde_gain is 0.0
        # since SC.22), so a zero moves to 0.01 exactly as a zero list element does above
        if isinstance(v, int):
            node[last] = v + (1 if factor > 1 else -1)
        else:
            node[last] = v * factor if v else 0.01
    return out


BOSON_DICT = yaml.safe_load(BOSON_YAML.read_text())


@pytest.mark.parametrize("leaf", _numeric_leaves(BOSON_DICT))
def test_hash_changes_for_every_numeric_leaf(leaf: str, data_dir: pathlib.Path) -> None:
    base = SensorConfig.model_validate(BOSON_DICT)
    try:
        other = SensorConfig.model_validate(_perturb(BOSON_DICT, leaf, 1.01))
    except ValidationError:
        try:  # at an upper bound (1.0, 16 bits): perturb downward instead
            other = SensorConfig.model_validate(_perturb(BOSON_DICT, leaf, 0.99))
        except ValidationError:
            pytest.skip(f"{leaf} is pinned by the schema (e.g. tvh == 1.0), so it cannot vary")
    assert config_hash(other, data_dir) != config_hash(base, data_dir), leaf
    in_band = leaf.startswith("sensor.band.")
    assert (band_hash(other, data_dir) != band_hash(base, data_dir)) == in_band, leaf


@pytest.mark.parametrize(
    ("dotted", "value"),
    [
        ("sensor.band.regime", "mixed"),
        ("sensor.band.id", "lwir"),
        ("sensor.optics.housing_temp_mode", "fixed"),
        ("sensor.nuc.mode", "ideal"),
        ("sensor.isp.agc", "linear"),
        ("sensor.isp.polarity", "black_hot"),
        ("sensor.isp.palette", "ironbow"),
        ("sensor.optics.vignetting_cos4", False),
        ("sensor.outputs.display_8", False),
        ("sensor.optics.distortion.model", "kannala_brandt"),
    ],
)
def test_hash_changes_for_enumerated_leaves(
    dotted: str, value: Any, data_dir: pathlib.Path
) -> None:
    d = copy.deepcopy(BOSON_DICT)
    node: Any = d
    *path, last = dotted.split(".")
    for key in path:
        node = node[key]
    node[last] = value
    if value == "kannala_brandt":  # schema v2 rules: coefficient count, no cos4 with fisheye
        node["coeffs"] = [0.0, 0.0, 0.0, 0.0]
        d["sensor"]["optics"]["vignetting_cos4"] = False
    if value == "fixed":  # `fixed` needs a housing temperature (ADR 0017)
        d["sensor"]["optics"]["housing_temp_k"] = 300.0
    base = SensorConfig.model_validate(BOSON_DICT)
    other = SensorConfig.model_validate(d)
    assert config_hash(other, data_dir) != config_hash(base, data_dir)


def test_band_hash_ignores_non_band_fields_and_tracks_spectral_bytes(
    data_dir: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    base = SensorConfig.model_validate(BOSON_DICT)
    d = copy.deepcopy(BOSON_DICT)
    d["sensor"]["noise"]["netd_mk_at_300k"] = 40.0  # industrial grade, §16.1
    d["sensor"]["optics"]["f_number"] = 1.4
    graded = SensorConfig.model_validate(d)
    assert band_hash(graded, data_dir) == band_hash(base, data_dir)
    assert config_hash(graded, data_dir) != config_hash(base, data_dir)
    # one response sample moved by 1e-3: > 1 mK in Lb at 300 K, so the LUT must be rebuilt
    other_root = tmp_path / "data2"
    shutil.copytree(data_dir, other_root)
    csv = other_root / RESPONSE_REL
    csv.write_text(csv.read_text().replace("13.5,1.0", "13.5,0.999"))
    assert band_hash(base, other_root) != band_hash(base, data_dir)
    assert config_hash(base, other_root) != config_hash(base, data_dir)


def test_round_trip_load_dump_load(data_dir: pathlib.Path, tmp_path: pathlib.Path) -> None:
    cfg = load_sensor_config(BOSON_YAML, data_dir)
    text = dump_sensor_config(cfg)
    again = load_sensor_config(
        (tmp_path / "dump.yaml").write_text(text) and tmp_path / "dump.yaml", data_dir
    )
    assert again == cfg
    assert config_hash(again) == config_hash(cfg)


# --- the exposure override (IG.13) ------------------------------------------------------------


SWIR_YAML = REPO / "configs" / "sensors" / "example_swir_ingaas_640.yaml"


def test_overriding_the_integration_time_lands_and_moves_the_config_hash() -> None:
    """Two exposures are two cameras, and the hash has to say so.

    The multi-band sweep films one scene through four cameras, and the reflective bands need
    their own exposure -- a daylight scene saturates a low-light one by around 120x. If the
    override did not reach the hash, two renders that differ by a factor of a hundred in signal
    would carry the same provenance, and the LUT and golden machinery keyed off that hash would
    happily reuse one for the other.
    """
    base = load_sensor_config(SWIR_YAML)
    exposed = with_integration_time_ms(base, 0.08)

    assert exposed.sensor.fpa.integration_time_ms == pytest.approx(0.08)
    assert base.sensor.fpa.integration_time_ms != pytest.approx(0.08), "the fixture is a no-op"
    assert config_hash(exposed) != config_hash(base)

    # It is the *sensor*, not the band: R(lambda) has not moved, so a band LUT stays valid.
    assert band_hash(exposed) == band_hash(base)

    # And the original is untouched -- a driver holds both while it prints its summary.
    assert base.sensor.fpa.integration_time_ms != pytest.approx(0.08)


def test_a_bolometer_is_refused_an_exposure_rather_than_given_a_meaningless_one() -> None:
    """§8.2: a microbolometer's responsivity is thermal. There is no integration time to set.

    Ignoring the flag would be worse than refusing it: the returned config would read as exposed,
    hash as exposed, and behave exactly as it did before.
    """
    bolometer = load_sensor_config(BOSON_YAML)
    with pytest.raises(ValueError, match="photon FPA"):
        with_integration_time_ms(bolometer, 0.08)
