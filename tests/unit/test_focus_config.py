"""OC.4 -- the focus block and the two defocus switches (ADR 0129).

The test that protects everyone else's work is the hash one: adding a field to a schema every
golden array was keyed against is a good way to invalidate the whole regression suite silently.
"""

from __future__ import annotations

import copy

import pytest
from pydantic import ValidationError

from irsim.config.loader import config_hash, load_sensor_config
from irsim.config.sensor import SCHEMA_VERSION, FocusSpec, SensorConfig

BOSON = "configs/sensors/flir_boson_640_lwir.yaml"


@pytest.fixture(scope="module")
def boson_doc() -> dict:
    import yaml

    with open(BOSON, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _build(doc: dict, **optics) -> SensorConfig:
    d = copy.deepcopy(doc)
    for key, value in optics.items():
        if key == "focus":
            d["sensor"]["optics"]["focus"] = value
        elif key == "fidelity":
            d["sensor"]["fidelity"] = value
        else:
            d["sensor"]["optics"]["mtf"][key] = value
    return SensorConfig.model_validate(d)


def test_the_default_camera_is_the_one_every_render_before_the_lane_used() -> None:
    s = load_sensor_config(BOSON).sensor
    assert s.optics.focus.mode == "infinity"
    assert s.optics.mtf.defocus_model == "none"
    assert s.focus_distance_m is None
    assert s.defocus_enabled is False
    assert SCHEMA_VERSION == 11  # v10 added focus, v11 the thermal-defocus materials (OC.10)


def test_the_default_focus_block_is_dropped_from_the_config_hash(boson_doc) -> None:
    """A v9 document and a v10 document that spells out the defaults are one camera."""
    implicit = config_hash(load_sensor_config(BOSON))
    explicit = config_hash(
        _build(boson_doc, focus={"mode": "infinity"}, defocus_model="none", defocus_apply="global")
    )
    assert implicit == explicit, "spelling out the defaults must not change the hash"


def test_naming_a_model_or_a_focus_distance_does_change_the_hash(boson_doc) -> None:
    base = config_hash(load_sensor_config(BOSON))
    hopkins = config_hash(_build(boson_doc, defocus_model="hopkins"))
    focused = config_hash(_build(boson_doc, defocus_model="hopkins", focus={"mode": "hyperfocal"}))
    fixed = config_hash(
        _build(boson_doc, defocus_model="hopkins", focus={"mode": "fixed", "distance_m": 50.0})
    )
    off = config_hash(_build(boson_doc, defocus_model="hopkins", fidelity={"defocus": False}))
    assert len({base, hopkins, focused, fixed, off}) == 5, "every switch must be provenance"


def test_hyperfocal_resolves_against_the_pitch_and_says_so(boson_doc) -> None:
    """The acceptable circle of confusion is a convention, so it is resolved once, in the config."""
    s = _build(boson_doc, defocus_model="hopkins", focus={"mode": "hyperfocal"}).sensor
    assert s.focus_distance_m == pytest.approx(16.34733, abs=1e-4)  # c = one 12 µm pitch
    loose = _build(
        boson_doc, defocus_model="hopkins", focus={"mode": "hyperfocal", "coc_um": 24.0}
    ).sensor
    assert loose.focus_distance_m == pytest.approx(0.5 * 16.3333 + 0.014, abs=1e-3)
    assert loose.focus_distance_m < s.focus_distance_m, "a looser criterion focuses nearer"


def test_fixed_carries_its_distance_and_the_others_refuse_one() -> None:
    assert FocusSpec(mode="fixed", distance_m=50.0).distance_m == 50.0
    with pytest.raises(ValidationError, match="requires distance_m"):
        FocusSpec(mode="fixed")
    with pytest.raises(ValidationError, match="does not take distance_m"):
        FocusSpec(mode="infinity", distance_m=50.0)
    with pytest.raises(ValidationError, match="only meaningful for focus mode 'hyperfocal'"):
        FocusSpec(mode="infinity", coc_um=12.0)


def test_a_focus_distance_with_no_model_is_refused(boson_doc) -> None:
    """It would look like it did something and do nothing -- the failure mode worth refusing."""
    with pytest.raises(ValidationError, match="would change nothing"):
        _build(boson_doc, focus={"mode": "fixed", "distance_m": 20.0})
    # the reverse is legitimate: a lens focused past hyperfocal still defocuses everything near
    s = _build(boson_doc, defocus_model="hopkins").sensor
    assert s.focus_distance_m is None and s.defocus_enabled


def test_the_fidelity_switch_can_only_turn_defocus_off(boson_doc) -> None:
    on = _build(boson_doc, defocus_model="hopkins").sensor
    off = _build(boson_doc, defocus_model="hopkins", fidelity={"defocus": False}).sensor
    never = _build(boson_doc, fidelity={"defocus": True}).sensor
    assert on.defocus_enabled and not off.defocus_enabled
    assert not never.defocus_enabled, "fidelity cannot switch on a model that was never named"


@pytest.mark.parametrize("model", ["none", "gaussian", "geometric", "hopkins"])
def test_every_named_model_is_accepted(boson_doc, model: str) -> None:
    focus = {"mode": "infinity"}
    assert (
        _build(boson_doc, defocus_model=model, focus=focus).sensor.optics.mtf.defocus_model == model
    )


def test_an_unknown_model_or_application_is_refused(boson_doc) -> None:
    with pytest.raises(ValidationError):
        _build(boson_doc, defocus_model="bokeh")
    with pytest.raises(ValidationError):
        _build(boson_doc, defocus_apply="perpixel")
