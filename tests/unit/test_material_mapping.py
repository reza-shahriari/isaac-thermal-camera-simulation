"""Material mapping resolver and audit (M7.17): precedence, the named cases, loud misses with
id 0, coverage and the CI exit code, ids stable and identical to the packed table."""

from __future__ import annotations

import importlib.util
import json
import pathlib

import pytest
import yaml

from irsim.materials import MaterialLibrary, MaterialTable
from irsim.materials.mapping import (
    MAPPING_PATH,
    MaterialResolver,
    PrimRecord,
    audit,
    load_mapping_rules,
)
from irsim.materials.table import UNMAPPED_MATERIAL_ID

REPO = pathlib.Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def resolver() -> MaterialResolver:
    library = MaterialLibrary.load()
    names = MaterialTable.from_library(library, "lwir").names
    return MaterialResolver(load_mapping_rules(known_materials=library.names), names)


def test_precedence_override_semantic_pattern_miss(resolver: MaterialResolver) -> None:
    r = resolver.resolve(PrimRecord("/W/a", "Rust_Trim", "car_body", "glass_windshield"))
    assert (r.material, r.rule) == ("glass_windshield", "override")
    r = resolver.resolve(PrimRecord("/W/b", "Rust_Trim", "car_body"))
    assert (r.material, r.rule, r.matched) == ("car_paint_black", "semantic", "car_body")
    r = resolver.resolve(PrimRecord("/W/c", "Rust_Trim", "unknown_class"))
    assert (r.material, r.rule, r.matched) == ("rusted_steel", "pattern", "*rust*")
    r = resolver.resolve(PrimRecord("/W/d", "Mystery_Mat_07", None))
    assert r.material is None and r.material_id == UNMAPPED_MATERIAL_ID and r.rule == "miss"
    assert r in resolver.misses and not r.mapped
    with pytest.raises(ValueError, match="not a library material"):
        resolver.resolve(PrimRecord("/W/e", override="unobtainium"))
    with pytest.raises(ValueError, match="not a library material"):
        resolver.resolve(PrimRecord("/W/f", override="UNMAPPED"))


def test_named_cases(resolver: MaterialResolver) -> None:
    assert resolver.resolve(PrimRecord("/W/g", "Windshield_Glass")).material == "glass_windshield"
    # AT.18: a name glob may not reach a mirror -- "Metal" is far more often matte than polished.
    assert resolver.resolve(PrimRecord("/W/h", "Metal_Matte")).material == (
        "aircraft_aluminium_painted"
    )
    assert resolver.resolve(PrimRecord("/W/i", "Car_Paint_Red")).material == "car_paint_black"
    assert resolver.resolve(PrimRecord("/W/j", "PAINT_WHITE_gloss")).material == "car_paint_white"
    miss = resolver.resolve(PrimRecord("/W/k", "Mystery_Mat_07"))
    assert miss.material_id == 0 and miss.material is None


def test_ids_stable_and_equal_to_the_packed_table(resolver: MaterialResolver) -> None:
    library = MaterialLibrary.load()
    table = MaterialTable.from_library(library, "mwir")  # a different band, same ids
    for name in library.names:
        assert resolver.id_for(name) == table.id_for(name)
    again = MaterialResolver(
        load_mapping_rules(), MaterialTable.from_library(MaterialLibrary.load(), "lwir").names
    )
    assert (
        again.resolve(PrimRecord("/x", "Metal_Housing")).material_id
        == resolver.resolve(PrimRecord("/x", "Metal_Housing")).material_id
    )


def _ten_prims() -> list[PrimRecord]:
    names = [
        "Car_Paint_Red",
        "Windshield_Glass",
        "Metal_Housing",
        "Road_Asphalt",
        "Skin_Face",
        "Paint_Blue",
        "Glass_Side",
    ]
    mapped = [PrimRecord(f"/W/{i}", n) for i, n in enumerate(names)]
    misses = [
        PrimRecord("/W/m1", "Mystery_Mat_07"),
        PrimRecord("/W/m2", "Mystery_Mat_07"),
        PrimRecord("/W/m3", None),
    ]
    return mapped + misses


def test_coverage_and_grouped_misses(resolver: MaterialResolver) -> None:
    report = audit(_ten_prims(), resolver, threshold=0.9)
    assert report.total == 10 and report.mapped == 7 and report.coverage == pytest.approx(0.7)
    assert not report.passed and report.misses_by_material_name == {
        "Mystery_Mat_07": 2,
        "<no material>": 1,
    }
    assert report.by_rule == {"pattern": 7, "miss": 3}
    assert "Mystery_Mat_07" in report.render() and "FAIL" in report.render()
    assert audit(_ten_prims(), resolver, threshold=0.5).passed
    assert audit([], resolver).coverage == 1.0


def test_audit_script_exit_code(tmp_path: pathlib.Path) -> None:
    spec = importlib.util.spec_from_file_location(
        "audit_materials", REPO / "scripts" / "audit_materials.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    dump = tmp_path / "prims.json"
    dump.write_text(json.dumps([p.__dict__ for p in _ten_prims()]))
    assert mod.main([str(dump), "--threshold", "0.9"]) == 1
    assert mod.main([str(dump), "--threshold", "0.7"]) == 0
    assert mod.main([str(dump)]) == 1, "the rules' 95 % threshold"


def test_rule_file_guards(tmp_path: pathlib.Path) -> None:
    raw = yaml.safe_load(MAPPING_PATH.read_text())
    assert raw["mapping"]["coverage_threshold"] == 0.95
    # A name that is deliberately not, and will not become, a material. `vegetation_leaf`
    # used to serve here and stopped working the day M7.9 authored it.
    raw["mapping"]["semantic"]["hull"] = "unobtanium_plating"
    p = tmp_path / "mapping.yaml"
    p.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="unknown materials"):
        load_mapping_rules(p, known_materials=MaterialLibrary.load().names)
    assert "unobtanium_plating" in load_mapping_rules(p).targets  # unchecked load is allowed
    raw = yaml.safe_load(MAPPING_PATH.read_text())
    raw["mapping"]["patterns"].append({"match": "*GLASS*", "material": "asphalt_dry"})
    p.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="duplicate pattern"):
        load_mapping_rules(p)
    with pytest.raises(ValueError, match="UNMAPPED"):
        MaterialResolver(load_mapping_rules(), ("car_paint_black",))
    with pytest.raises(ValueError, match="only known keys"):
        PrimRecord.from_dict({"path": "/x", "colour": "red"})
