"""A real asset's triangles as engine-free solvable geometry (AI.2, ADR 0128, ADR 0132).

The fixture is two prims lifted out of the prepared DJI Phantom 4 Pro — the full archive is
1.5 M triangles and generated, not in git. What is committed is small enough to read and real
enough to carry the asset's actual pathologies.

The tests that matter here are not "does it load". They are:

* **area survives**, because area sets both the radiated power and the convective load, and the
  decimation that produced this archive is only legitimate if it left area alone;
* **the manifest and the arrays agree**, because the prep tool's gate was applied to the manifest
  and a silent disagreement would mean the gate passed on geometry nobody is solving;
* **a mesh nobody can afford to trace is refused, not run**, because the failure mode it replaces
  is a scene that never finishes.

docs/physics-model.md §6.1; ADR 0110, ADR 0128.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest

from irsim.config.scene import MeshSpec
from irsim.io.assets import AssetMesh, load_asset_meshes

FIXTURE = pathlib.Path(__file__).parent / "data" / "phantom4_fixture.meshes.npz"
WHITE = "GeometryNode_455"
GREY = "GeometryNode_270"


@pytest.fixture(scope="module")
def meshes():  # type: ignore[no-untyped-def]
    return load_asset_meshes(FIXTURE)


# ---------------------------------------------------------------------------------------------
# the archive
# ---------------------------------------------------------------------------------------------


def test_the_manifest_and_the_arrays_agree_on_area(meshes) -> None:  # type: ignore[no-untyped-def]
    """The recorded area is what the prep tool's own gate was applied to."""
    for mesh in meshes.values():
        assert mesh.computed_area_m2() == pytest.approx(mesh.area_m2, rel=1e-9)


def test_a_manifest_that_disagrees_with_its_arrays_is_refused(tmp_path) -> None:  # type: ignore[no-untyped-def]
    with np.load(FIXTURE, allow_pickle=False) as data:
        manifest = json.loads(str(data["manifest"]))
        arrays = {k: data[k] for k in data.files if k != "manifest"}
    manifest[0]["area_m2"] = float(manifest[0]["area_m2"]) * 1.5  # a decimation nobody recorded
    bad = tmp_path / "bad.meshes.npz"
    np.savez_compressed(bad, manifest=json.dumps(manifest), **arrays)
    with pytest.raises(ValueError, match="disagree"):
        load_asset_meshes(bad)


def test_planar_dissolve_kept_the_area_it_decimated(meshes) -> None:  # type: ignore[no-untyped-def]
    """Collapse decimation removed 37 % of this asset's area; planar dissolve must remove none.

    The prep tool's whole justification for using the dissolve decimator is that it merges
    coplanar faces without moving a vertex. If that ever stops being true the archive is still
    loadable, still renders, and quietly under-radiates.
    """
    for mesh in meshes.values():
        assert mesh.area_before_m2 > 0.0
        assert mesh.area_m2 / mesh.area_before_m2 == pytest.approx(1.0, abs=0.02)


def test_no_degenerate_faces_survive(meshes) -> None:  # type: ignore[no-untyped-def]
    """A facet with no area has no normal, and §6.1 balances a facet on its normal."""
    for mesh in meshes.values():
        v, f = mesh.vertices_m, mesh.faces
        a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
        assert np.all(np.linalg.norm(np.cross(b - a, c - a), axis=-1) > 0.0)


def test_grouping_is_by_material_because_the_asset_is(meshes) -> None:  # type: ignore[no-untyped-def]
    assert [m.name for m in meshes.by_material("_dji_phantom_4_prowhite_plastic_matte")] == [WHITE]
    assert [m.name for m in meshes.by_material("metal_radial")] == [GREY]
    assert meshes.by_material("nothing_uses_this") == ()


def test_a_missing_prim_names_what_is_there(meshes) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(KeyError, match="GeometryNode_455"):
        meshes["shell_top"]


def test_a_missing_archive_says_how_to_make_one() -> None:
    with pytest.raises(FileNotFoundError, match="prep_asset.py"):
        load_asset_meshes(FIXTURE.parent / "no_such_asset.meshes.npz")


def test_a_face_indexing_a_vertex_that_does_not_exist_is_refused() -> None:
    with pytest.raises(ValueError, match="does not exist"):
        AssetMesh(
            name="x",
            material_name=None,
            vertices_m=np.zeros((3, 3)),
            faces=np.array([[0, 1, 7]], dtype=np.intp),
            area_m2=0.0,
            area_before_m2=0.0,
        )


# ---------------------------------------------------------------------------------------------
# the scene schema
# ---------------------------------------------------------------------------------------------


def test_a_mesh_names_one_source_or_the_other() -> None:
    MeshSpec(asset="phantom4", prim=WHITE, cell_m=0.01)
    MeshSpec(shape="sphere", centre_m=(0.0, 0.0, 0.0), radius_m=1.0)
    with pytest.raises(ValueError, match="exactly one"):
        MeshSpec(shape="sphere", radius_m=1.0, asset="phantom4", prim=WHITE)
    with pytest.raises(ValueError, match="exactly one"):
        MeshSpec()


def test_an_asset_mesh_carries_no_generated_geometry() -> None:
    with pytest.raises(ValueError, match="needs `prim:`"):
        MeshSpec(asset="phantom4")
    with pytest.raises(ValueError, match="no radius_m"):
        MeshSpec(asset="phantom4", prim=WHITE, radius_m=1.0)
    with pytest.raises(ValueError, match="asset archive"):
        MeshSpec(shape="sphere", radius_m=1.0, prim=WHITE)


def test_building_a_patch_from_an_asset_preserves_its_area() -> None:
    """The solver's geometry is the archive's geometry — nothing rescales on the way in.

    This is the assertion that would fail if `build_mesh` ever re-applied `scale_to_metres`: the
    prep tool already applied it, and a second application is a silent factor of 100.
    """
    from irsim.scene import build_mesh

    archive = load_asset_meshes(FIXTURE)
    patch = build_mesh(MeshSpec(asset=str(FIXTURE), prim=WHITE, cell_m=0.01))
    assert patch.area_m2 == pytest.approx(archive[WHITE].area_m2, rel=1e-9)
    assert patch.n_faces == archive[WHITE].n_faces
    assert patch.n_cells >= patch.n_faces  # ptex: at least one cell per face


def test_cells_never_fall_below_one_per_face() -> None:
    from irsim.scene import build_mesh

    coarse = build_mesh(MeshSpec(asset=str(FIXTURE), prim=WHITE, cell_m=10.0))
    assert coarse.n_cells == coarse.n_faces


# ---------------------------------------------------------------------------------------------
# the cost guard — the failure mode it replaces is a scene that never finishes
# ---------------------------------------------------------------------------------------------


def test_an_unaffordable_self_occlusion_trace_is_refused_with_both_ways_out() -> None:
    """Measured: `shell_upper` is 41,127 cells x 41,103 faces = 1.69e9 ray-triangle tests.

    Per sky patch, and the beam repeats it every tick. A 3,320-cell prim spent **248 s** in
    `disc_visibility` over one 6 h spin-up before the switch reached the beam; the same scene
    builds in 1.0 s with it. Without this guard a scene does not fail, it hangs.
    """
    import irsim.scene as scene_module

    patch = scene_module.build_mesh(MeshSpec(asset=str(FIXTURE), prim=WHITE, cell_m=0.01))
    assert patch.n_cells * patch.n_faces > 0

    with pytest.raises(ValueError, match="self_occluding"):
        scene_module._refuse_untraceable_mesh("shell", _Oversized(patch))

    # Under the budget it says nothing, which is what every generated mesh relies on. The grey
    # prim is 953 faces: 9.1e5, comfortably inside the 2e6 ceiling.
    small = scene_module.build_mesh(MeshSpec(asset=str(FIXTURE), prim=GREY, cell_m=10.0))
    assert small.n_cells * small.n_faces < scene_module.MESH_SELF_OCCLUSION_BUDGET
    scene_module._refuse_untraceable_mesh("small", small)


class _Oversized:
    """A stand-in with the real asset's measured face and cell counts, and its own vertices."""

    def __init__(self, patch: object) -> None:
        self._patch = patch
        self.n_cells = 41_127
        self.n_faces = 41_103

    def __getattr__(self, item: str) -> object:
        return getattr(self._patch, item)


def test_the_guard_lets_a_convex_mesh_through_however_big_it_is() -> None:
    """Tracing a convex mesh is exact *and* skippable, so it must never be refused for size."""
    import irsim.scene as scene_module
    from irsim.thermal.raycast import sphere_mesh

    soup = sphere_mesh((0.0, 0.0, 0.0), 1.0, 40, 80)
    from irsim.thermal.mesh_field import TriangleMeshPatch

    patch = TriangleMeshPatch.from_soup(soup, 3)
    assert patch.n_cells * patch.n_faces > scene_module.MESH_SELF_OCCLUSION_BUDGET
    scene_module._refuse_untraceable_mesh("dome", patch)  # convex: allowed


# ---------------------------------------------------------------------------------------------
# the physics the asset pipeline exists to deliver
# ---------------------------------------------------------------------------------------------


def _scene_text(material_a: str, material_b: str, self_occluding: str = "false") -> str:
    """A two-surface scene on the *same* fixture prim, differing only in material."""
    return f"""
schema_version: 17
scene:
  name: asset_alpha_contrast
  description: "One imported prim, solved twice, differing only in solar absorptivity"
  weather_file: weather/clear_midlat_summer_48h.csv
  atmosphere_preset: us_standard_clear
  environment_preset: clear_dry
  site: {{latitude_deg: 48.1, longitude_deg: 11.6, altitude_m: 520.0}}
  start_utc: "2024-06-21T10:00:00Z"
  world_frame: {{up: [0.0, 0.0, 1.0], north: [0.0, 1.0, 0.0]}}
  thermal:
    spin_up_hours: 6.0
    tick_s: 60.0
    surfaces:
      - name: pale
        material: {material_a}
        tilt_deg: 0.0
        mesh: {{asset: {FIXTURE}, prim: {WHITE}, cell_m: 0.02, self_occluding: {self_occluding}}}
      - name: dark
        material: {material_b}
        tilt_deg: 0.0
        mesh: {{asset: {FIXTURE}, prim: {WHITE}, cell_m: 0.02, self_occluding: {self_occluding}}}
"""


def test_solar_absorptivity_separates_two_materials_on_the_same_imported_prim(
    tmp_path,  # type: ignore[no-untyped-def]
) -> None:
    """The headline result of the asset lane, stated as physics rather than as plumbing.

    Identical geometry, identical forcing, identical spin-up — the only difference is
    `solar_absorptivity`, 0.25 for moulded white ABS against 0.94 for a black finish. In midday
    June sun that is the largest single contrast on a small aircraft and it has nothing to do with
    any heat source. If this ever collapses, either the per-cell forcing has stopped reading the
    material or the asset path has stopped carrying the geometry it claims to.

    Measured on the full asset with the same solver: the black mouldings reach **65.9 °C** while
    the white shell tops out at **33.8 °C**.
    """
    import numpy as np

    from irsim.scene import Scene

    path = tmp_path / "contrast.yaml"
    path.write_text(_scene_text("abs_plastic_white", "car_paint_black"), encoding="utf-8")
    scene = Scene.from_file(path)

    t0 = scene.t0_s
    fields = scene.mesh_fields
    for field in fields.values():
        field.advance_to(t0)
    pale = np.asarray(fields["pale"].temperature_at(t0), dtype=np.float64)
    dark = np.asarray(fields["dark"].temperature_at(t0), dtype=np.float64)

    # Same mesh, so the comparison is cell for cell.
    assert pale.shape == dark.shape
    assert float(dark.mean() - pale.mean()) > 5.0

    # And it is a *field*, not one value per object — the whole point of the lane.
    assert float(pale.max() - pale.min()) > 1.0
    assert float(dark.max() - dark.min()) > float(pale.max() - pale.min())


def test_the_scene_refuses_an_undecided_imported_prim(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """`self_occluding:` omitted on a prim over the budget must raise, not silently cost hours.

    The fixture prim's own product is 1.07e7 against a 2e6 ceiling, which is the case the budget
    was calibrated on: it is the mesh that spent 248 s tracing before the switch existed.
    """
    from irsim.scene import Scene

    text = _scene_text("abs_plastic_white", "car_paint_black").replace(
        ", self_occluding: false", ""
    )
    path = tmp_path / "undecided.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="self_occluding"):
        Scene.from_file(path)
