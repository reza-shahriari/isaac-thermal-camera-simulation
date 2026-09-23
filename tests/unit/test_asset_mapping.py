"""Per-asset material mapping: precedence, the two corrections it exists for, and the real asset.

ADR 0128. The interesting tests here are not that the resolver returns a string -- they are that
the *global* rules and the *asset* rules disagree by an amount that matters radiometrically, and
in the direction the asset file claims. A mapping layer that resolved every name to something
plausible would pass a "does it map?" test and still ruin the image.

docs/physics-model.md §13.3; ADR 0047 (precedence, the coverage gate), ADR 0128 (this rung).
"""

from __future__ import annotations

import json
import pathlib

import pytest

from irsim.materials.library import MaterialLibrary
from irsim.materials.mapping import (
    ASSETS_DIR,
    AssetMapping,
    MappingRules,
    MaterialResolver,
    PatternRule,
    PrimRecord,
    audit,
    load_asset_mapping,
    load_mapping_rules,
)
from irsim.materials.table import MaterialTable

PHANTOM4_PRIMS = pathlib.Path(__file__).parent / "data" / "phantom4.prims.json"


@pytest.fixture(scope="module")
def library() -> MaterialLibrary:
    return MaterialLibrary.load()


@pytest.fixture(scope="module")
def names(library: MaterialLibrary) -> tuple[str, ...]:
    return MaterialTable.from_library(library, "lwir").names


@pytest.fixture(scope="module")
def phantom4(library: MaterialLibrary) -> AssetMapping:
    return load_asset_mapping("phantom4", known_materials=library.names)


@pytest.fixture(scope="module")
def phantom4_records() -> list[PrimRecord]:
    return [PrimRecord.from_dict(d) for d in json.loads(PHANTOM4_PRIMS.read_text(encoding="utf-8"))]


# ---------------------------------------------------------------------------------------------
# precedence
# ---------------------------------------------------------------------------------------------


def test_asset_beats_global_pattern(names: tuple[str, ...]) -> None:
    rules = MappingRules(patterns=[PatternRule(match="*white*", material="car_paint_white")])
    asset = AssetMapping(name="a", materials={"white_plastic_matte": "abs_plastic_white"})
    prim = PrimRecord(path="/p", material_name="white_plastic_matte")

    assert MaterialResolver(rules, names).resolve(prim).material == "car_paint_white"
    resolved = MaterialResolver(rules, names, asset=asset).resolve(prim)
    assert resolved.material == "abs_plastic_white"
    assert resolved.rule == "asset"


def test_prim_override_still_beats_the_asset_map(names: tuple[str, ...]) -> None:
    """The `thermal:material` attribute stays the top rung: a scene-prep script must win."""
    asset = AssetMapping(name="a", materials={"Copper": "abs_plastic_white"})
    prim = PrimRecord(path="/p", material_name="Copper", override="bare_aluminium")
    resolved = MaterialResolver(MappingRules(), names, asset=asset).resolve(prim)
    assert (resolved.material, resolved.rule) == ("bare_aluminium", "override")


def test_asset_beats_semantic(names: tuple[str, ...]) -> None:
    """An asset map is authored after looking at *this* asset; a semantic class is generic."""
    rules = MappingRules(semantic={"drone": "car_paint_black"})
    asset = AssetMapping(name="a", materials={"shell": "abs_plastic_white"})
    prim = PrimRecord(path="/p", material_name="shell", semantic_class="drone")
    assert MaterialResolver(rules, names).resolve(prim).rule == "semantic"
    assert MaterialResolver(rules, names, asset=asset).resolve(prim).rule == "asset"


def test_asset_lookup_is_case_insensitive_but_not_a_glob(names: tuple[str, ...]) -> None:
    asset = AssetMapping(name="a", materials={"Copper": "bare_aluminium"})
    resolver = MaterialResolver(MappingRules(), names, asset=asset)
    assert resolver.resolve(PrimRecord(path="/a", material_name="COPPER")).rule == "asset"
    # A near-match must miss. An asset map is authored knowledge, so a partial hit is a mistake.
    assert resolver.resolve(PrimRecord(path="/b", material_name="Copper_2")).rule == "miss"


def test_a_miss_is_still_loud_with_an_asset_map(names: tuple[str, ...]) -> None:
    asset = AssetMapping(name="a", materials={"known": "bare_aluminium"})
    resolver = MaterialResolver(MappingRules(), names, asset=asset)
    resolved = resolver.resolve(PrimRecord(path="/p", material_name="unknown"))
    assert resolved.material_id == 0
    assert resolver.misses == [resolved]


# ---------------------------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------------------------


def test_asset_targeting_an_unknown_material_is_refused(names: tuple[str, ...]) -> None:
    asset = AssetMapping(name="a", materials={"x": "unobtanium"})
    with pytest.raises(ValueError, match="not in the table"):
        MaterialResolver(MappingRules(), names, asset=asset)


def test_case_colliding_keys_are_refused() -> None:
    with pytest.raises(ValueError, match="differing only by case"):
        AssetMapping(name="a", materials={"Copper": "water", "copper": "snow"})


def test_scale_must_be_positive() -> None:
    with pytest.raises(ValueError):
        AssetMapping(name="a", scale_to_metres=0.0)


# ---------------------------------------------------------------------------------------------
# the two corrections the phantom4 map exists for -- stated radiometrically
# ---------------------------------------------------------------------------------------------


def test_the_shell_correction_halves_the_areal_heat_capacity(
    library: MaterialLibrary, names: tuple[str, ...], phantom4: AssetMapping
) -> None:
    """`*white*` -> car_paint_white is paint on steel; this shell is moulded ABS.

    Both are white dielectrics with eps ~ 0.9, so no emissivity test would catch the swap. What
    differs is the half that sets the temperature: areal heat capacity, and therefore how fast the
    skin follows its own shading and airspeed.
    """
    global_rules = load_mapping_rules(known_materials=library.names)
    prim = PrimRecord(path="/p", material_name="white_plastic_matte")

    without = MaterialResolver(global_rules, names).resolve(prim).material
    with_asset = MaterialResolver(global_rules, names, asset=phantom4).resolve(prim).material
    assert (without, with_asset) == ("car_paint_white", "abs_plastic_white")

    c_wrong = library[without].spec.thermal.heat_capacity_j_m2_k
    c_right = library[with_asset].spec.thermal.heat_capacity_j_m2_k
    # 4399 against 2205 J m^-2 K^-1: the global rule makes the shell twice as sluggish as it is.
    assert c_wrong / c_right == pytest.approx(2.0, rel=0.05)


def test_the_motor_correction_is_an_order_of_magnitude_in_emissivity(
    library: MaterialLibrary, names: tuple[str, ...], phantom4: AssetMapping
) -> None:
    """`*metal*` -> bare_aluminium puts eps = 0.09 on a matte housing.

    At eps 0.09 a surface is a mirror: 91 % of what the camera sees is reflected sky, not the
    part's own temperature. This is the trap `irsim_isaac/phantom3.py` already records for the
    same component, and it is the difference between a motor that reads hot and one that reads
    like the sky above it.
    """
    global_rules = load_mapping_rules(known_materials=library.names)
    prim = PrimRecord(path="/p", material_name="Metal_Matte")

    without = MaterialResolver(global_rules, names).resolve(prim).material
    with_asset = MaterialResolver(global_rules, names, asset=phantom4).resolve(prim).material
    assert (without, with_asset) == ("bare_aluminium", "aircraft_aluminium_painted")

    e_wrong = library[without].band_properties("lwir").emissivity
    e_right = library[with_asset].band_properties("lwir").emissivity
    assert e_wrong < 0.15 and e_right > 0.85
    assert e_right / e_wrong > 5.0


# ---------------------------------------------------------------------------------------------
# the committed asset, against the committed prim dump
# ---------------------------------------------------------------------------------------------


def test_phantom4_reaches_full_coverage_where_the_global_rules_fail(
    library: MaterialLibrary,
    names: tuple[str, ...],
    phantom4: AssetMapping,
    phantom4_records: list[PrimRecord],
) -> None:
    rules = load_mapping_rules(known_materials=library.names)
    assert len(phantom4_records) == 41

    bare = audit(phantom4_records, MaterialResolver(rules, names))
    assert not bare.passed
    assert bare.coverage == pytest.approx(20 / 41, abs=1e-9)

    mapped = audit(phantom4_records, MaterialResolver(rules, names, asset=phantom4))
    assert mapped.passed
    assert mapped.coverage == 1.0
    assert mapped.by_rule == {"asset": 41}


def test_phantom4_map_has_no_unused_entries(
    phantom4: AssetMapping, phantom4_records: list[PrimRecord]
) -> None:
    """Every key must correspond to a material the asset actually carries.

    A stale key is a statement about an asset that is no longer true, and it survives silently --
    the resolver simply never reaches it.
    """
    in_asset = {r.material_name.lower() for r in phantom4_records if r.material_name}
    in_map = {k.lower() for k in phantom4.materials}
    assert in_map == in_asset


#: The prepared asset's true axis-aligned extent, in source units, measured from its **vertices**
#: (not from `o.bound_box`, which is a local AABB and inflates under this asset's rotated parent
#: chain -- an earlier revision of this test quoted the inflated 60.21 and was wrong).
PHANTOM4_EXTENT_UNITS = (41.05, 46.37, 20.67)


def test_phantom4_scale_puts_the_aircraft_at_its_published_size(phantom4: AssetMapping) -> None:
    """The height is what pins the factor: 20.67 units against DJI's published 196 mm.

    Left at 1.0 this aircraft enters a scene 41 m wide, which still renders a plausible image and
    is the reason the factor is authored rather than guessed at import time. The neighbouring
    factors are not close calls -- millimetres would make this aircraft 2 cm tall and metres 20 m.
    """
    assert phantom4.scale_to_metres == 0.01
    height_m = PHANTOM4_EXTENT_UNITS[2] * phantom4.scale_to_metres
    assert height_m == pytest.approx(0.196, abs=0.015)

    # An X-quad's 350 mm diagonal puts each motor at 350/(2*sqrt 2) = 123.7 mm on both axes, and a
    # 239 mm propeller reaches 119.5 mm beyond that: ~486 mm subtended on each axis.
    for across in PHANTOM4_EXTENT_UNITS[:2]:
        assert 0.35 < across * phantom4.scale_to_metres < 0.55

    for wrong in (0.001, 1.0):
        assert not 0.15 < PHANTOM4_EXTENT_UNITS[2] * wrong < 0.25


def test_every_committed_asset_config_loads_and_targets_real_materials(
    library: MaterialLibrary,
) -> None:
    configs = sorted(ASSETS_DIR.glob("*.yaml"))
    assert configs, "configs/assets is empty"
    for path in configs:
        asset = load_asset_mapping(path, known_materials=library.names)
        assert asset.name == path.stem


# ---------------------------------------------------------------------------------------------
# the prep driver's Blender invocation, checked without Blender
# ---------------------------------------------------------------------------------------------


def _prep_module() -> object:
    import importlib.util

    path = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "prep_asset.py"
    spec = importlib.util.spec_from_file_location("prep_asset", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_blender_command_passes_the_scale_without_losing_precision() -> None:
    """The factor crosses a process boundary as text, so it must survive the round trip.

    `str(0.01)` is fine, but a factor like 1/3 formatted with `%s` at default precision is not,
    and a silently truncated scale is exactly the class of error this whole step exists to stop.
    """
    prep = _prep_module()
    cmd = prep.blender_command(  # type: ignore[attr-defined]
        "blender",
        pathlib.Path("a.fbx"),
        pathlib.Path("out.usdc"),
        pathlib.Path("out.json"),
        1.0 / 3.0,
    )
    assert "--background" in cmd and "--factory-startup" in cmd
    assert cmd[cmd.index("--") + 1 :][:2] == ["--source", "a.fbx"]
    assert float(cmd[cmd.index("--scale") + 1]) == 1.0 / 3.0


def test_every_importer_extension_is_lowercase_and_dotted() -> None:
    prep = _prep_module()
    for suffix in prep.IMPORTERS:  # type: ignore[attr-defined]
        assert suffix.startswith(".") and suffix == suffix.lower()
