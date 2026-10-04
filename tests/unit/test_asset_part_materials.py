"""A part may assert its material, and an asset may exclude geometry that is not it (AI.14).

Two facts about third-party models that the AI.5 decomposition could not state:

* **one atlas, many substances.** The Inspire 3's artist put its PC shell, carbon arms, glass-
  filled propellers and motor bells on one texture-atlas material called ``default``, so the
  source-material map can only say "all of it is the same thing". A part knows better: the arms
  are carbon fibre whatever the atlas is called.
* **not everything in the file is the aircraft.** The Matrice 300 RTK arrives standing on a display
  pole over its transport case. Leaving the case unclaimed drops it from the part split but fails
  the 95 % coverage gate, since the case is most of the file's area; claiming it as a part keeps
  a suitcase hanging under the aircraft in every render.

Every test here is engine-free and fails if the claiming rule, the precedence or the loader
check is reverted.
"""

from __future__ import annotations

import importlib.util
import pathlib

import numpy as np
import pytest
import yaml

from irsim.io.asset_parts import Component, PartsConfig, PartSelector, PartSpec, assign_parts
from irsim.materials.library import MaterialLibrary
from irsim.materials.mapping import MaterialResolver, PrimRecord, load_asset_mapping

REPO = pathlib.Path(__file__).resolve().parents[2]


def _box(index, centre, size, faces, area, material):
    half = tuple(s / 2.0 for s in size)
    return Component(
        index=index,
        faces=faces,
        area_m2=area,
        centroid=tuple(centre),
        lo=tuple(c - h for c, h in zip(centre, half, strict=True)),
        hi=tuple(c + h for c, h in zip(centre, half, strict=True)),
        material_name=material,
    )


def aircraft_on_a_case():
    """An airframe at z ~ 0.35 m standing on a pole over a case far larger than itself.

    The Matrice 300's proportions: the case holds about three quarters of the file's area.
    """
    return [
        _box(0, (0.0, 0.0, 0.35), (0.40, 0.30, 0.10), 5000, 0.30, "lambert19"),  # body
        _box(1, (0.3, 0.3, 0.33), (0.05, 0.05, 0.03), 800, 0.02, "lambert3"),  # a motor
        _box(2, (0.0, 0.0, 0.18), (0.02, 0.02, 0.20), 200, 0.01, "lambert25"),  # display pole
        _box(3, (0.0, 0.0, 0.05), (0.60, 0.45, 0.10), 9000, 0.95, "lambert32"),  # the case
    ]


def config_with_exclusion(exclude=True):
    return PartsConfig(
        parts=[
            PartSpec(name="motor", target="motor", select=PartSelector(materials=["lambert3"])),
            PartSpec(name="airframe", target="airframe", select=PartSelector()),
        ],
        exclude=[PartSelector(z_max_m=0.25)] if exclude else [],
    )


def test_an_excluded_case_leaves_the_coverage_totals_rather_than_failing_them():
    _, without = assign_parts(aircraft_on_a_case(), config_with_exclusion(exclude=False))
    _, with_ = assign_parts(aircraft_on_a_case(), config_with_exclusion())
    # without the field, a catch-all swallows the case -- 100 % "coverage" of a suitcase
    assert without.area_by_part["airframe"] == pytest.approx(0.30 + 0.01 + 0.95)
    assert with_.area_by_part["airframe"] == pytest.approx(0.30)
    assert with_.total_area_m2 == pytest.approx(0.32), "the case is not part of the asset's area"
    assert with_.excluded_area_m2 == pytest.approx(0.96)
    assert with_.excluded_faces == 9200
    assert with_.coverage == pytest.approx(1.0) and with_.passed


def test_exclusion_is_tried_before_the_first_part_so_no_catch_all_can_take_it_back():
    config = config_with_exclusion()
    case = aircraft_on_a_case()[3]
    assert config.excludes(case)
    assert config.claim(case) is None, "the trailing catch-all accepts everything else"
    assert config.claim(aircraft_on_a_case()[0]).name == "airframe"


def test_the_report_names_the_excluded_geometry():
    _, report = assign_parts(aircraft_on_a_case(), config_with_exclusion())
    assert "<excluded: not the asset>" in report.render()


def test_the_archive_split_drops_excluded_geometry():
    from irsim.io.asset_parts import split_by_part
    from irsim.io.assets import AssetMesh, AssetMeshes

    def cube(z):
        v = np.array(
            [
                [x, y, z + dz]
                for dz in (0.0, 0.1)
                for x, y in ((0, 0), (0.1, 0), (0.1, 0.1), (0, 0.1))
            ],
            dtype=np.float64,
        )
        f = np.array(
            [[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4],
             [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]],
            dtype=np.intp,
        )  # fmt: skip
        return v, f

    va, fa = cube(0.4)  # the aircraft
    vc, fc = cube(0.0)  # the case under it
    mesh = AssetMesh(
        name="Object_1",
        material_name="default",
        vertices_m=np.concatenate([va, vc]),
        faces=np.concatenate([fa, fc + len(va)]),
        area_m2=0.12,
        area_before_m2=0.12,
    )
    config = PartsConfig(
        parts=[PartSpec(name="airframe", select=PartSelector())],
        exclude=[PartSelector(z_max_m=0.25)],
    )
    parts, report = split_by_part(AssetMeshes(name="toy", meshes={"Object_1": mesh}), config)
    assert parts["airframe"].n_faces == 12, "only the aircraft's cube survives"
    assert report.excluded_faces == 12 and report.passed


def test_parts_list_the_materials_they_assert():
    config = PartsConfig(
        parts=[
            PartSpec(name="arms", material="carbon_fibre", select=PartSelector(r_min_m=0.1)),
            PartSpec(name="shell", select=PartSelector()),
        ]
    )
    assert config.materials == {"arms": "carbon_fibre"}


def _asset_doc(part_material: str) -> dict:
    return {
        "schema_version": 1,
        "asset": {
            "name": "atlas_quad",
            "source_file": "x.glb",
            "materials": {"default": "painted_composite"},
            "parts": {
                "parts": [
                    {"name": "arms", "material": part_material, "select": {"r_min_m": 0.1}},
                    {"name": "shell", "select": {}},
                ]
            },
        },
    }


def test_the_loader_refuses_a_part_material_the_library_does_not_have(tmp_path):
    names = MaterialLibrary.load().names
    good = tmp_path / "good.yaml"
    good.write_text(yaml.safe_dump(_asset_doc("carbon_fibre")))
    asset = load_asset_mapping(good, known_materials=names)
    assert "carbon_fibre" in asset.targets, "a part's material is one of the asset's targets"
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump(_asset_doc("unobtainium_weave")))
    with pytest.raises(ValueError, match="unobtainium_weave"):
        load_asset_mapping(bad, known_materials=names)


def test_the_stamped_override_outranks_the_atlas_on_every_face_group(tmp_path):
    """The part-split USD carries the asserted material as `thermal:material` on the mesh.

    A multi-material part is one record per subset, each taking the mesh's override, exactly as
    `materials_usd.walk_stage` reads it inside Kit -- so every face group of the arms resolves to
    carbon fibre although the atlas says `default`.
    """
    path = tmp_path / "a.yaml"
    path.write_text(yaml.safe_dump(_asset_doc("carbon_fibre")))
    library = MaterialLibrary.load()
    asset = load_asset_mapping(path, known_materials=library.names)
    from irsim.materials.mapping import load_mapping_rules
    from irsim.materials.table import MaterialTable

    rules = load_mapping_rules(known_materials=library.names)
    ids = MaterialTable.from_library(library, "lwir").names
    resolver = MaterialResolver(rules, ids, asset=asset)
    arms = resolver.resolve(
        PrimRecord(
            path="/World/arms/arms/default", material_name="default", override="carbon_fibre"
        )
    )
    shell = resolver.resolve(PrimRecord(path="/World/shell/shell", material_name="default"))
    assert (arms.material, arms.rule) == ("carbon_fibre", "override")
    assert (shell.material, shell.rule) == ("painted_composite", "asset")


@pytest.fixture(scope="module")
def prep():
    spec = importlib.util.spec_from_file_location("prep_asset", REPO / "scripts" / "prep_asset.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_blender_split_is_told_what_to_drop_and_what_to_stamp(prep):
    """The driver's plan is the worker's only instruction; an excluded component in it would
    come back as a part, and a missing `materials` key would stamp nothing, silently."""
    config = PartsConfig(
        parts=[
            PartSpec(name="motor", material="aircraft_aluminium_painted",
                     select=PartSelector(materials=["lambert3"])),
            PartSpec(name="airframe", select=PartSelector()),
        ],
        exclude=[PartSelector(z_max_m=0.25)],
    )  # fmt: skip
    stats = [
        {
            "index": c.index,
            "faces": c.faces,
            "area_m2": c.area_m2,
            "centroid": list(c.centroid),
            "lo": list(c.lo),
            "hi": list(c.hi),
            "material_name": c.material_name,
        }
        for c in aircraft_on_a_case()
    ]
    plan = prep.part_assignment(stats, config)
    assert plan["part_of_component"] == {"0": "airframe", "1": "motor"}
    assert plan["excluded_components"] == 2
    assert plan["materials"] == {"motor": "aircraft_aluminium_painted"}
    assert plan["parts"] == ["motor", "airframe"]


def test_component_areas_are_measured_in_the_stage_frame_not_the_mesh_frame(prep):
    """The component pass once summed Blender's local polygon areas under world-space bounds.

    A glTF at `scale_to_metres: 1.0` keeps its node transforms, so its local areas are in the
    artist's units: the Mini 3 Pro came out at 40,701 m2. Coverage is area-weighted, so that
    number decides which parts pass. A unit quad under a non-uniform (2, 3, 1) scale is 6 m2 in
    the stage, and no single factor applied to the local 1 m2 gets there for every orientation.
    """
    local = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float64)
    world = local * np.array([2.0, 3.0, 1.0])
    tris = np.array([[0, 1, 2], [0, 2, 3]])
    areas = prep.world_polygon_areas(world, tris, np.array([0, 0]), 1)
    assert areas.tolist() == pytest.approx([6.0])
    # two polygons, one triangle each, keep their own areas
    split = prep.world_polygon_areas(world, tris, np.array([0, 1]), 2)
    assert split.tolist() == pytest.approx([3.0, 3.0])


def test_every_blender_pass_fails_the_driver_when_the_worker_raises(prep):
    """Blender exits 0 after an uncaught Python exception unless told otherwise.

    Found while ingesting the Mini 3 Pro: the part split raised, the driver saw 0, and the run
    reported PASS over a part-split USD holding one prim of fifteen.
    """
    src = pathlib.Path("x.glb")
    out = pathlib.Path("data/assets/x/x.usdc")
    commands = [
        prep.blender_command("blender", src, out, out.with_suffix(".json"), 1.0),
        prep.blender_split_command("blender", src, 1.0, out, out, out),
        prep.blender_material_slots_command("blender", src, 1.0, out),
        prep.blender_material_split_command("blender", src, 1.0, out, out),
    ]
    for cmd in commands:
        i = cmd.index("--python-exit-code")
        assert cmd[i + 1] == "1" and i < cmd.index("--"), cmd
