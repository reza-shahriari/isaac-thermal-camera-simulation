"""ME.8: the §15 Tier 5 ablation switches are config, and the config is in the hash.

§15's first caution is that a sim-to-real gap has to be attributed, which means running the same
scenario with one mechanism off. Every switch below already existed -- as a keyword argument of
`PipelineConfig.from_sensor` or of `attach_sensor_chain`. **A keyword argument is invisible to the
config hash**, so two ablation variants produced identical hashes and a stored result could not say
which of them it was. These tests pin both halves of the fix: the switch reaches the pipeline, and
it reaches the hash.

The second half is the one worth stating plainly, because it is what makes an ablation *study*
possible rather than just an ablation: a variant is identified by its config hash, so M12.3 can
store a result next to the hash of the configuration that produced it and never mix two up.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from irsim.config.loader import config_hash, load_sensor_config
from irsim.config.sensor import FULL_FIDELITY, MIN_SCHEMA_VERSION, SCHEMA_VERSION, SensorConfig
from irsim.materials import MaterialTable
from irsim.pipeline import PipelineConfig, PipelineState, run_frame
from irsim.pipeline.sensor_chain import attach_sensor_chain
from irsim.radiometry.encoding import encode_temperature

REPO = pathlib.Path(__file__).resolve().parents[2]
BOSON_YAML = REPO / "configs" / "sensors" / "flir_boson_640_lwir.yaml"

SWITCHES = ["noise", "optical_psf", "bad_pixels", "nuc_residual"]


def _boson_dict() -> dict:
    return load_sensor_config(BOSON_YAML).model_dump(mode="json")


def _small(**fidelity: bool) -> SensorConfig:
    """The Boson at 64x64 with supersampling off, so a frame is cheap."""
    d = _boson_dict()
    d["sensor"]["fpa"].update(width=64, height=64)
    d["sensor"]["optics"]["supersample_factor"] = 1
    if fidelity:
        d["sensor"]["fidelity"] = {**FULL_FIDELITY.model_dump(), **fidelity}
    return SensorConfig.model_validate(d)


def _scene() -> dict[str, np.ndarray]:
    t = np.full((64, 64), 295.0, np.float32)
    t[20:40, 20:40] = 320.0
    return {
        "temperature_k": t,
        "encoded_t": encode_temperature(t),
        "normal_dot_view": np.ones((64, 64), np.float32),
        "distance_m": np.full((64, 64), 100.0, np.float32),
        "material_id": np.ones((64, 64), np.int32),
        "sky_view_factor": np.zeros((64, 64), np.float32),
    }


def _run(config: PipelineConfig, n_frames: int = 3) -> list[np.ndarray]:
    state = PipelineState()
    scene = _scene()
    return [run_frame(scene, config, state).signal_dn.copy() for _ in range(n_frames)]


# --- the hash ---------------------------------------------------------------------------------


@pytest.mark.parametrize("switch", SWITCHES)
def test_every_switch_changes_the_config_hash(switch: str) -> None:
    base = config_hash(_small())
    assert config_hash(_small(**{switch: False})) != base, switch


def test_the_switches_are_independent_in_the_hash() -> None:
    """Four switches, sixteen distinct hashes -- no pair of variants can be confused."""
    hashes = set()
    for mask in range(2 ** len(SWITCHES)):
        flags = {s: not (mask >> i) & 1 for i, s in enumerate(SWITCHES)}
        hashes.add(config_hash(_small(**flags)))
    assert len(hashes) == 2 ** len(SWITCHES)


def test_full_fidelity_hashes_the_same_whether_it_is_written_or_left_out() -> None:
    """The hash tracks the ablation, not the notation.

    A `fidelity:` block with every switch `true` describes the same camera as no block at all, so
    it must not produce a second hash for the same configuration -- otherwise every result stored
    before ME.8 would read as a different variant. This is why the default block is dropped from
    the canonical dump.
    """
    assert config_hash(_small()) == config_hash(_small(**{s: True for s in SWITCHES}))


def test_a_v8_document_is_still_readable_and_hashes_the_same() -> None:
    """v9 added `fidelity:` as an optional block with a full-fidelity default, so a v8 document
    describes exactly the camera it described before -- and `schema_version` is deliberately not
    part of the hash, because it describes the document format and not the sensor.

    `OC.4` added v10 (`optics.focus` and the defocus switches) and `OC.10` v11 (the thermal-defocus
    materials) under the same rule, and `SC.19` v12 (late defects); all are covered here: every
    version of one camera must hash alike, or every golden array written before `OC` would read
    as a different sensor."""
    assert MIN_SCHEMA_VERSION == 8 and SCHEMA_VERSION == 12
    old = _boson_dict() | {"schema_version": 8}
    new = _boson_dict() | {"schema_version": 12}
    for between in (9, 10, 11):
        assert config_hash(SensorConfig.model_validate(old)) == config_hash(
            SensorConfig.model_validate(_boson_dict() | {"schema_version": between})
        )
    assert config_hash(SensorConfig.model_validate(old)) == config_hash(
        SensorConfig.model_validate(new)
    )
    for bad in (MIN_SCHEMA_VERSION - 1, SCHEMA_VERSION + 1):
        with pytest.raises(ValueError, match="readable range"):
            SensorConfig.model_validate(_boson_dict() | {"schema_version": bad})


def test_the_ablations_that_are_not_in_this_block_are_hashed_where_they_live() -> None:
    """§15 also names the AGC and the FFC. Both were already explicit config, and duplicating them
    here would raise the question of which copy wins -- so they stay where they are, and this is
    the test that they are in the hash all the same."""
    base = config_hash(_small())
    linear_agc = _boson_dict()
    linear_agc["sensor"]["fpa"].update(width=64, height=64)
    linear_agc["sensor"]["optics"]["supersample_factor"] = 1
    no_ffc = {**linear_agc, "sensor": {**linear_agc["sensor"]}}
    linear_agc["sensor"]["isp"] = {**linear_agc["sensor"]["isp"], "agc": "linear"}
    no_ffc["sensor"]["nuc"] = {**no_ffc["sensor"]["nuc"], "mode": "ideal"}
    assert config_hash(SensorConfig.model_validate(linear_agc)) != base
    assert config_hash(SensorConfig.model_validate(no_ffc)) != base


# --- the pipeline ------------------------------------------------------------------------------


def test_noise_off_is_bit_identical_to_the_argument(boson_lut) -> None:  # type: ignore[no-untyped-def]
    materials = MaterialTable.constant(1.0)
    from_config = PipelineConfig.from_sensor(
        _small(noise=False), materials, lut=boson_lut, sensor_seed=77
    )
    from_argument = PipelineConfig.from_sensor(
        _small(), materials, lut=boson_lut, sensor_seed=77, noise_enabled=False
    )
    assert from_config.noise_enabled is False
    for a, b in zip(_run(from_config), _run(from_argument), strict=True):
        assert np.array_equal(a, b)
    # ...and it is not the same as leaving it on, or the test above would pass on a dead switch.
    on = PipelineConfig.from_sensor(_small(), materials, lut=boson_lut, sensor_seed=77)
    assert not np.array_equal(_run(from_config)[0], _run(on)[0])


def test_the_optical_psf_switch_is_bit_identical_to_the_argument(boson_lut) -> None:  # type: ignore[no-untyped-def]
    materials = MaterialTable.constant(1.0)
    from_config = PipelineConfig.from_sensor(
        _small(optical_psf=False), materials, lut=boson_lut, sensor_seed=77
    )
    from_argument = PipelineConfig.from_sensor(
        _small(), materials, lut=boson_lut, sensor_seed=77, psf_enabled=False
    )
    assert from_config.psf is None and from_argument.psf is None
    for a, b in zip(_run(from_config), _run(from_argument), strict=True):
        assert np.array_equal(a, b)
    on = PipelineConfig.from_sensor(_small(), materials, lut=boson_lut, sensor_seed=77)
    assert on.psf is not None
    assert not np.array_equal(_run(from_config)[0], _run(on)[0])


@pytest.mark.parametrize(
    ("switch", "argument", "attribute"),
    [
        ("bad_pixels", "defects_enabled", "defects_enabled"),
        ("nuc_residual", "residual_enabled", "residual_enabled"),
    ],
)
def test_the_chain_switches_reach_the_chain(boson_lut, switch, argument, attribute) -> None:  # type: ignore[no-untyped-def]
    materials = MaterialTable.constant(1.0)
    ambient = {"ambient_provider": lambda t: 293.15}
    from_config = attach_sensor_chain(
        PipelineConfig.from_sensor(_small(**{switch: False}), materials, lut=boson_lut), **ambient
    )
    from_argument = attach_sensor_chain(
        PipelineConfig.from_sensor(_small(), materials, lut=boson_lut),
        **ambient,
        **{argument: False},
    )
    assert getattr(from_config.chain, attribute) is False
    assert getattr(from_argument.chain, attribute) is False
    default_on = attach_sensor_chain(
        PipelineConfig.from_sensor(_small(), materials, lut=boson_lut), **ambient
    )
    assert getattr(default_on.chain, attribute) is True


def test_an_explicit_argument_still_overrides_the_config(boson_lut) -> None:  # type: ignore[no-untyped-def]
    """The bench affordance, pinned so it is a decision rather than an accident: a bool wins over
    the config, which is how a bench isolates one mechanism without authoring a config file. A run
    whose provenance matters sets the config, because only the config reaches the hash."""
    materials = MaterialTable.constant(1.0)
    forced_on = PipelineConfig.from_sensor(
        _small(noise=False), materials, lut=boson_lut, noise_enabled=True
    )
    assert forced_on.noise_enabled is True
    forced_off = PipelineConfig.from_sensor(_small(), materials, lut=boson_lut, psf_enabled=False)
    assert forced_off.psf is None
