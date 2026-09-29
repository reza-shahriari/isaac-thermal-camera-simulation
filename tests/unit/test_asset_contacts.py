"""AI.11 -- an asset's parts carry their contacts and hidden parts.

`configs/assets/<name>.yaml` gains `contacts:` (two parts, a joint from `joints.yaml`, an area)
and `hidden_parts:` (a component placed by a box, with overrides). One loader, engine-free;
refused: a contact naming a missing part, a joint not in the table, an area larger than either
part's, a hidden part without mass. A config round-trips through YAML and the Phantom 4 loads
with its own contacts. ADR 0128 lineage; roadmap AI.11.
"""

from __future__ import annotations

import pathlib

import pytest
import yaml
from test_asset_parts import CENTRE, quad_components  # the synthetic Phantom 4 of AI.5's tests

from irsim.config.joints import load_joint_table
from irsim.io.asset_parts import (
    ContactSpec,
    HiddenPartSpec,
    PartsConfig,
    PartSelector,
    PartSpec,
    assign_parts,
)
from irsim.materials.mapping import load_asset_mapping

REPO = pathlib.Path(__file__).resolve().parents[2]


def _config(**extra: object) -> PartsConfig:
    return PartsConfig(
        centre=CENTRE,
        parts=[
            PartSpec(
                name="motor_front_left",
                target="motor",
                select=PartSelector(
                    near_xy=(0.044, 0.314), within_m=0.03, materials=["Metal_Matte"]
                ),
            ),
            PartSpec(name="battery", target="battery", select=PartSelector(materials=["Black"])),
        ],
        **extra,  # type: ignore[arg-type]
    )


def test_a_config_with_contacts_and_hidden_parts_round_trips_through_yaml() -> None:
    config = _config(
        contacts=[ContactSpec(a="motor_front_left", b="esc", joint="dry_default", area_m2=2e-4)],
        hidden_parts=[
            HiddenPartSpec(
                name="esc",
                centre_m=(0.044, 0.314, 0.03),
                size_m=(0.03, 0.02, 0.006),
                component="esc",
                mass_kg=0.012,
                specific_heat_j_kgk=900.0,
                dissipation_w=2.0,
            )
        ],
    )
    text = yaml.safe_dump(config.model_dump(mode="json"))
    again = PartsConfig.model_validate(yaml.safe_load(text))
    assert again == config
    assert again.hidden("ESC").capacity_j_k == pytest.approx(10.8)
    assert again.hidden("esc").surface_area_m2 == pytest.approx(2 * (6e-4 + 1.2e-4 + 1.8e-4))


def test_an_asset_without_the_new_blocks_loads_unchanged() -> None:
    config = _config()
    assert config.contacts == [] and config.hidden_parts == [] and config.hidden_names == ()
    assert PartsConfig.model_validate(yaml.safe_load(yaml.safe_dump(config.model_dump()))) == config


# --- refusals -----------------------------------------------------------------------------------


def test_a_contact_naming_a_missing_part_is_refused() -> None:
    with pytest.raises(ValueError, match="neither a part nor a hidden part"):
        _config(
            contacts=[
                ContactSpec(a="motor_front_left", b="nobody", joint="dry_default", area_m2=1e-4)
            ]
        )


def test_a_part_cannot_touch_itself_and_hidden_names_cannot_shadow_parts() -> None:
    with pytest.raises(ValueError, match="touch itself"):
        ContactSpec(a="battery", b="Battery", joint="dry_default", area_m2=1e-4)
    with pytest.raises(ValueError, match="share a name"):
        _config(
            hidden_parts=[
                HiddenPartSpec(
                    name="Battery", centre_m=(0, 0, 0), size_m=(0.1, 0.1, 0.1), mass_kg=0.4
                )
            ]
        )


def test_a_joint_not_in_the_table_is_refused_by_the_loader(tmp_path: pathlib.Path) -> None:
    config = _config(
        contacts=[
            ContactSpec(a="motor_front_left", b="battery", joint="welded_unobtainium", area_m2=1e-4)
        ]
    )
    with pytest.raises(ValueError, match="unknown joint 'welded_unobtainium'"):
        config.check_joints(load_joint_table())
    doc = {
        "schema_version": 1,
        "asset": {
            "name": "bad_joint",
            "source_file": "x.fbx",
            "materials": {"Metal_Matte": "bare_aluminium"},
            "parts": config.model_dump(mode="json"),
        },
    }
    path = tmp_path / "bad_joint.yaml"
    path.write_text(yaml.safe_dump(doc))
    with pytest.raises(ValueError, match="welded_unobtainium"):
        load_asset_mapping(path)


def test_an_area_larger_than_either_part_is_refused_when_the_geometry_is_measured() -> None:
    """The motor can is 27 x 27 x 16 mm: about 0.0027 m2 of surface. A contact claiming a
    square decimetre on it is a typo, and the decomposition says so."""
    fine = _config(
        contacts=[ContactSpec(a="motor_front_left", b="battery", joint="dry_default", area_m2=1e-4)]
    )
    assign_parts(quad_components(), fine)
    wrong = _config(
        contacts=[ContactSpec(a="motor_front_left", b="battery", joint="dry_default", area_m2=1e-2)]
    )
    with pytest.raises(ValueError, match="larger than 'motor_front_left'"):
        assign_parts(quad_components(), wrong)
    # a hidden part's area is its box's
    boxed = _config(
        contacts=[ContactSpec(a="battery", b="esc", joint="dry_default", area_m2=5e-3)],
        hidden_parts=[
            HiddenPartSpec(
                name="esc", centre_m=(0.044, 0.314, 0.03), size_m=(0.03, 0.02, 0.006), mass_kg=0.012
            )
        ],
    )
    with pytest.raises(ValueError, match="larger than 'esc'"):
        assign_parts(quad_components(), boxed)


def test_a_hidden_part_without_mass_is_refused_unless_a_component_carries_it() -> None:
    with pytest.raises(ValueError, match="has no mass"):
        HiddenPartSpec(name="ghost", centre_m=(0, 0, 0), size_m=(0.01, 0.01, 0.01))
    with_component = HiddenPartSpec(
        name="esc", centre_m=(0, 0, 0), size_m=(0.01, 0.01, 0.01), component="esc"
    )
    assert with_component.capacity_j_k is None, "the component's, once AI.12 exists"
    with pytest.raises(ValueError, match="positive"):
        HiddenPartSpec(name="flat", centre_m=(0, 0, 0), size_m=(0.01, 0.0, 0.01), mass_kg=0.1)


# --- the Phantom 4 -------------------------------------------------------------------------------


def test_the_phantom4_carries_its_contacts_and_hidden_parts_and_still_loads() -> None:
    asset = load_asset_mapping("phantom4_parts")
    assert asset.parts is not None
    parts = asset.parts
    assert set(parts.hidden_names) == {
        "esc_front_left",
        "esc_front_right",
        "esc_rear_left",
        "esc_rear_right",
        "flight_controller",
        # TC.13: the sources themselves, inside the bells and the case
        "winding_front_left",
        "winding_front_right",
        "winding_rear_left",
        "winding_rear_right",
        "pack",
    }
    table = load_joint_table()
    parts.check_joints(table)
    known = {p.lower() for p in parts.names} | {h.lower() for h in parts.hidden_names}
    for c in parts.contacts:
        assert c.a.lower() in known and c.b.lower() in known
        for end in (c.a, c.b):
            if end in parts.hidden_names:
                assert c.area_m2 < parts.hidden(end).surface_area_m2
    # every station's motor is bolted to its mount and every ESC lies in the arm beside it
    pairs = {(c.a, c.b) for c in parts.contacts}
    for st in ("front_left", "front_right", "rear_left", "rear_right"):
        assert (f"motor_{st}", f"motor_mount_{st}") in pairs
        assert (f"esc_{st}", "arms") in pairs
    # every hidden part has a capacity and a dissipation: its own numbers, or (TC.13) the
    # component library's for the windings and the pack, which carry only `component:`
    from irsim.config.components import load_component_library

    components = load_component_library()
    for h in parts.hidden_parts:
        assert h.status == "ESTIMATED"
        assert h.resolved_capacity_j_k(components) > 0.0
        assert h.resolved_dissipation_w(1.0, components) > 0.0
        if h.component is None:
            assert h.capacity_j_k is not None and h.dissipation_w is not None
    # the older per-material config has no such blocks and loads exactly as before
    plain = load_asset_mapping("phantom4")
    assert plain.parts is None or plain.parts.contacts == []
