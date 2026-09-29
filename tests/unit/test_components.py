"""AI.12 -- a library of hidden components, aerial first.

`configs/components/` holds one file per component with mass, heat capacity, rated and idle
dissipation, the faces it heats, its case material and its provenance. The loader refuses an
entry whose rated dissipation disagrees with the efficiency it cites (P_in − P_shaft, 5 %), one
without mass or source, and one naming a material the library does not have; every hidden part
the Phantom 4 declares names a component that exists, and a hidden part with only a component
takes its capacity and dissipation from it. Roadmap AI.12; ADR 0072 (retired), AI.11, TC.11.
"""

from __future__ import annotations

import pathlib

import pytest
import yaml

from irsim.config.components import (
    COMPONENTS_DIR,
    ComponentSpec,
    load_component,
    load_component_library,
)
from irsim.materials.library import MaterialLibrary
from irsim.materials.mapping import load_asset_mapping

EXPECTED = {
    "brushless_motor",
    "esc",
    "lipo_pack",
    "flight_controller",
    "piston_engine",
    "small_turbine",
    "exhaust_line",
}
#: AI.13: the ground set -- what warms a car away from its engine.
GROUND = {
    "brake_disc": "brake",
    "tyre": "tyre",
    "tail_light": "light",
    "rear_window_heater": "heater",
    "differential": "gearbox",
    "ev_drive_motor": "motor",
    "ev_pack": "battery",
}


def _entry(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = dict(
        name="widget",
        kind="other",
        mass_kg=0.1,
        specific_heat_j_kgk=900.0,
        rated_power_in_w=100.0,
        efficiency=0.8,
        dissipation_rated_w=20.0,
        dissipation_idle_w=1.0,
        heats=["case"],
        material="bare_aluminium",
        status="ESTIMATED",
        source="an engineering estimate written down for the test, long enough to count",
    )
    base.update(overrides)
    return base


# --- the shipped library ------------------------------------------------------------------------


def test_the_library_ships_the_seven_components_the_row_names() -> None:
    lib = load_component_library()
    assert set(lib) == EXPECTED | set(GROUND)
    assert lib.path == COMPONENTS_DIR
    for c in lib.values():
        assert c.mass_kg > 0.0 and c.capacity_j_k > 0.0
        assert c.status in ("MEASURED", "ESTIMATED") and len(c.source) >= 20
        assert c.heats, f"{c.name} heats nothing"


def test_every_rated_dissipation_matches_its_cited_efficiency_to_five_percent() -> None:
    for c in load_component_library().values():
        expect = c.rated_power_in_w * (1.0 - c.efficiency)
        assert c.dissipation_rated_w == pytest.approx(expect, rel=0.05), c.name
        assert c.dissipation_w(1.0) == c.dissipation_rated_w
        assert c.dissipation_w(0.0) == c.dissipation_idle_w
        assert c.dissipation_idle_w <= c.dissipation_w(0.5) <= c.dissipation_rated_w


def test_every_material_a_component_names_is_in_the_material_library() -> None:
    lib = load_component_library()
    lib.check_materials(MaterialLibrary.load().keys())


def test_the_phantom4_hidden_parts_name_components_that_exist() -> None:
    lib = load_component_library()
    parts = load_asset_mapping("phantom4_parts").parts
    assert parts is not None
    named = {h.component for h in parts.hidden_parts if h.component is not None}
    assert named and named <= set(lib), named
    esc = lib["esc"]
    for h in parts.hidden_parts:
        if h.component == "esc":
            # the asset's own overrides sit within a factor of two of the library's figures
            assert h.mass_kg is not None and 0.5 < h.mass_kg / esc.mass_kg < 2.0
            assert h.dissipation_w is not None and h.dissipation_w <= esc.dissipation_rated_w


# --- refusals ------------------------------------------------------------------------------------


def test_an_entry_without_mass_or_source_is_refused() -> None:
    with pytest.raises(ValueError):
        ComponentSpec.model_validate(_entry(mass_kg=0.0))
    with pytest.raises(ValueError):
        ComponentSpec.model_validate({k: v for k, v in _entry().items() if k != "mass_kg"})
    with pytest.raises(ValueError):
        ComponentSpec.model_validate(_entry(source="short"))
    with pytest.raises(ValueError):
        ComponentSpec.model_validate({k: v for k, v in _entry().items() if k != "source"})


def test_a_dissipation_that_disagrees_with_its_efficiency_is_refused() -> None:
    with pytest.raises(ValueError, match="not rated_power_in_w"):
        ComponentSpec.model_validate(_entry(dissipation_rated_w=35.0))
    ok = ComponentSpec.model_validate(_entry(dissipation_rated_w=20.9))  # within 5 %
    assert ok.dissipation_rated_w == 20.9
    with pytest.raises(ValueError, match="idle dissipation exceeds rated"):
        ComponentSpec.model_validate(_entry(dissipation_idle_w=25.0))


def test_an_unknown_material_and_a_misnamed_file_are_refused(tmp_path: pathlib.Path) -> None:
    lib_dir = tmp_path / "components"
    lib_dir.mkdir()
    (lib_dir / "widget.yaml").write_text(
        yaml.safe_dump({"schema_version": 1, "component": _entry(material="unobtainium")})
    )
    lib = load_component_library(lib_dir)
    with pytest.raises(ValueError, match="unobtainium"):
        lib.check_materials(MaterialLibrary.load().keys())
    (lib_dir / "gadget.yaml").write_text(
        yaml.safe_dump({"schema_version": 1, "component": _entry()})
    )
    with pytest.raises(ValueError, match="named 'gadget' but the component 'widget'"):
        load_component(lib_dir / "gadget.yaml")
    with pytest.raises(KeyError, match="unknown component"):
        lib["nothing"]


# --- AI.13: the ground components ---------------------------------------------------------------


def test_the_ground_components_carry_their_kind_mass_and_cited_dissipation() -> None:
    """Seven entries a car needs away from its engine, each with a mass, a specific heat, a rated
    dissipation that its own efficiency reproduces to 5 %, at least one part it heats, a library
    material and a source that names where the numbers came from."""
    lib = load_component_library()
    for name, kind in GROUND.items():
        c = lib[name]
        assert c.kind == kind and c.mass_kg > 0.0 and c.specific_heat_j_kgk > 0.0, name
        assert c.dissipation_rated_w == pytest.approx(
            c.rated_power_in_w * (1.0 - c.efficiency), rel=0.05
        ), name
        assert c.heats and c.material is not None and len(c.source) > 80, name
    # a brake and a tyre are all heat, a lamp nearly so, a traction motor mostly not
    assert lib["brake_disc"].efficiency == 0.0 and lib["tyre"].efficiency == 0.0
    assert lib["tail_light"].dissipation_rated_w > 0.95 * lib["tail_light"].rated_power_in_w
    assert lib["ev_drive_motor"].efficiency >= 0.9 and lib["ev_pack"].efficiency >= 0.95
    # the pack's loss is I^2 R at its rating: (P / V)^2 R with 360 V and ~0.1 ohm
    assert lib["ev_pack"].dissipation_rated_w == pytest.approx(
        (40000.0 / 360.0) ** 2 * 0.1, rel=0.05
    )
    # capacities: the pack is a hundred times the lamp housing
    assert lib["ev_pack"].capacity_j_k > 100.0 * lib["tail_light"].capacity_j_k
