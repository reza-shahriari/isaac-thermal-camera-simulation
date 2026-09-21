"""WM.7 — a mesh temperature field, reachable from a scene config.

`WM.2` built the field and `WM.3` put it on pixels, and until this row nothing could ask for one:
no scene config could declare a mesh, so every shipped scene rendered its curved prims at a single
temperature and the whole lane was a capability with no user.

`configs/scenes/quad_flight_mesh.yaml` is the first scene that uses it. It is the aerial mission
of `quad_flight_pointwise.yaml` with the two arms as **tubes** instead of flat strips, which is
what an arm actually is — and the difference is the ~27 K between a sunlit carbon crown and an
underside sitting on air temperature, a spread a single patch normal cannot represent.

docs/physics-model.md §6.1, §13.1; ADR 0110 (the field on the mesh), ADR 0087, ADR 0109;
roadmap WM.7.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from irsim.config.scene import SCENE_SCHEMA_VERSION, MeshSpec, SurfaceSpec, load_scene_config
from irsim.scene import Scene, build_mesh
from irsim_isaac.pipeline.mesh_bridge import MeshBinding, MeshPointBridge

REPO = pathlib.Path(__file__).resolve().parents[2]
SCENE = REPO / "configs/scenes/quad_flight_mesh.yaml"
PLANAR = REPO / "configs/scenes/quad_flight_pointwise.yaml"

PAD_S = 0.0
CRUISE_S = 1002.0  # a multiple of the 6 s frame interval, as the driver steps it
UP = np.array([0.0, 0.0, 1.0])


@pytest.fixture(scope="module")
def scene() -> Scene:
    return Scene.from_file(SCENE)


def _crown_and_underside(field, t: float) -> tuple[float, float]:  # type: ignore[no-untyped-def]
    cells = np.asarray(field.temperature_at(t), dtype=np.float64)
    up = field.patch.cell_normal @ UP
    return float(cells[up > 0.7].mean()), float(cells[up < -0.7].mean())


# --- the bar: a tube is not a strip ---------------------------------------------------------


@pytest.mark.slow
def test_an_arm_carries_a_gradient_around_its_circumference(scene) -> None:  # type: ignore[no-untyped-def]
    """The measurement the row exists for. On the pad at a 61° sun the crown of a 30 mm carbon
    tube runs 26 K over the air its underside sits on, and both are cells of one solve on one
    prim. The patched scene beside it gives each arm one value across its whole width."""
    t = scene.t0_s + PAD_S
    air = float(scene.weather.at(t).t_air_k)
    assert set(scene.meshes) == {"arm_n", "arm_e"}
    assert set(scene.patches) == {"deck", "belly"}  # near-planar surfaces keep the cheaper path

    for name in ("arm_n", "arm_e"):
        field = scene.mesh_fields[name]
        field.advance_to(t)
        crown, underside = _crown_and_underside(field, t)
        assert crown - underside > 20.0, (name, crown, underside)
        assert abs(underside - air) < 1.5, (name, underside, air)
        cells = np.asarray(field.temperature_at(t), dtype=np.float64)
        assert float(np.ptp(cells)) > 25.0


@pytest.mark.slow
def test_the_crown_is_the_side_that_faces_the_sun_and_not_the_other_one() -> None:
    """The winding invariant, at the scene level. A mesh wound inside out shades correctly and
    solves to the same *span*, with the sunlit and shaded sides exchanged — so a test that only
    measured the spread would pass on an upside-down picture. This one names which side is hot."""
    scene = Scene.from_file(SCENE)
    t = scene.t0_s + PAD_S
    field = scene.mesh_fields["arm_n"]
    field.advance_to(t)
    cells = np.asarray(field.temperature_at(t), dtype=np.float64)
    up = field.patch.cell_normal @ UP
    hottest = field.patch.cell_normal[int(np.argmax(cells))]
    assert float(hottest @ UP) > 0.5, hottest  # the hottest cell faces up, toward a 61 deg sun
    # and around the tube temperature rises with how much sky a cell faces. Only the *side*
    # cells: the two end caps face along the arm, so their normals carry no information about
    # the circumference and sorting them in with the flank is comparing different surfaces.
    axis = np.array([0.0, 1.0, 0.0])  # arm_n runs north
    side = np.abs(field.patch.cell_normal @ axis) < 0.5
    # Binned rather than cell by cell: a quad of the tube is split into two triangles that are
    # not coplanar, so two cells at the same angular position have slightly different normals and
    # a strict per-cell ordering would be testing the tessellation, not the physics.
    edges = np.linspace(-1.0, 1.0, 9)
    means = [
        float(cells[side][(up[side] >= lo) & (up[side] < hi)].mean())
        for lo, hi in zip(edges[:-1], edges[1:], strict=True)
    ]
    assert np.all(np.diff(means) > 0.0), means
    assert means[-1] - means[0] > 25.0, means


@pytest.mark.slow
def test_the_planar_scene_gives_each_arm_one_value_across_its_width(scene) -> None:  # type: ignore[no-untyped-def]
    """The negative control: the same arm, modelled as ADR 0087's projection allows. Its cells
    vary along the arm where the pods shade it, and across the arm they cannot vary at all,
    because every cell of a patch shares the patch's one normal."""
    planar = Scene.from_file(PLANAR)
    t = planar.t0_s + PAD_S
    field = planar.surface_fields["arm_n"]
    field.advance_to(t)
    cells = np.asarray(field.temperature_at(t), dtype=np.float64).reshape(field.patch.shape)
    # n_u = 2 across the strip, and the width carries nothing: the columns differ by at most a
    # float32 ulp, which the pod shadow's edge falling between them accounts for. Against the
    # 27 K the mesh finds around the same tube, the strip's width is empty.
    assert cells.shape[1] == 2
    assert float(np.max(np.abs(cells[:, 0] - cells[:, 1]))) < 1e-3
    # There is no "underside" at all: one normal, so no cell of it faces away from the sun.
    normals = np.unique(np.round(field.patch.normal, 9).reshape(1, 3), axis=0)
    assert len(normals) == 1


@pytest.mark.slow
def test_the_mission_drives_the_tube_as_it_drives_the_deck() -> None:
    """ADR 0109's speed schedule reaches a mesh surface exactly as it reaches a patch: the crown's
    excess over air collapses in the climb and comes back on the ground.

    On its own scene, not the module fixture's: a field keeps two ticks and refuses to rewind, so
    a test that walks the mission forward would leave the shared one unable to answer at t0."""
    scene = Scene.from_file(SCENE)
    pad = scene.t0_s + PAD_S
    field = scene.mesh_fields["arm_n"]
    field.advance_to(pad)
    crown_pad = _crown_and_underside(field, pad)[0] - float(scene.weather.at(pad).t_air_k)

    cruise = scene.t0_s + CRUISE_S
    field.advance_to(cruise)
    crown_cruise = _crown_and_underside(field, cruise)[0] - float(scene.weather.at(cruise).t_air_k)
    assert crown_pad > 20.0, crown_pad
    assert crown_cruise < 15.0, crown_cruise


@pytest.mark.slow
def test_a_mesh_surface_opens_the_scene_with_its_gradient_already_grown() -> None:
    """A patch only spins up per cell when occluders make its cells differ. Every cell of a mesh
    has its own normal, so none of them share a history and the per-prim spun-up value is wrong
    for all of them: without a per-cell spin-up the scene would open with a uniform tube and take
    an hour of its own to reach the state it should already be in at 10:00."""
    scene = Scene.from_file(SCENE)
    t = scene.t0_s
    field = scene.mesh_fields["arm_n"]
    opening = np.asarray(field.temperature_at(t), dtype=np.float64)
    assert float(np.ptp(opening)) > 25.0, float(np.ptp(opening))


# --- the bar: it reaches pixels ----------------------------------------------------------------


@pytest.mark.slow
def test_the_scene_binds_its_meshes_to_prims_and_the_bridge_paints_them(scene) -> None:  # type: ignore[no-untyped-def]
    """`mesh_bindings()` is to `MeshPointBridge` what `surface_bindings()` is to the planar one,
    so a driver reads its curved fields off the scene instead of building them in Python."""
    assert dict(scene.mesh_prims) == {
        "arm_n": "/World/Quad/arm_n",
        "arm_e": "/World/Quad/arm_e",
    }
    bindings = scene.mesh_bindings()
    assert {path for path, _ in bindings} == set(scene.mesh_prims.values())

    t = scene.t0_s + PAD_S
    bridge = MeshPointBridge(
        [MeshBinding(path, fld) for path, fld in bindings],
        known_paths=list(scene.mesh_prims.values()),
    )
    patch = scene.mesh_fields["arm_n"].patch
    points = patch.cell_centres()
    ids = np.full(points.shape[0], 3, dtype=np.int32)
    plane = np.full(points.shape[0], 300.0, dtype=np.float32)
    out = bridge.apply(plane, ids, {"3": "/World/Quad/arm_n"}, points, t)
    assert float(np.ptp(plane)) == 0.0
    assert float(np.ptp(out)) > 25.0
    assert bridge.last_coverage["/World/Quad/arm_n"] == ids.size


# --- the schema --------------------------------------------------------------------------------


def test_the_schema_takes_a_mesh_and_refuses_what_has_no_meaning_on_one() -> None:
    assert SCENE_SCHEMA_VERSION == 14
    assert load_scene_config(SCENE).schema_version == SCENE_SCHEMA_VERSION

    cylinder = MeshSpec(
        shape="cylinder", centre_m=(0.0, 0.0, 0.0), radius_m=0.05, length_m=0.4, level=2
    )
    assert SurfaceSpec(name="a", material="concrete", mesh=cylinder).mesh is cylinder

    with pytest.raises(ValueError, match="a cylinder mesh needs length_m"):
        MeshSpec(shape="cylinder", centre_m=(0.0, 0.0, 0.0), radius_m=0.05)
    with pytest.raises(ValueError, match="a sphere mesh has no length_m"):
        MeshSpec(shape="sphere", centre_m=(0.0, 0.0, 0.0), radius_m=0.05, length_m=0.4)
    with pytest.raises(ValueError, match="`level` or `cell_m`, not both"):
        MeshSpec(shape="sphere", centre_m=(0.0, 0.0, 0.0), radius_m=0.05, level=2, cell_m=0.01)
    with pytest.raises(ValueError, match="axis has zero length"):
        MeshSpec(
            shape="cylinder",
            centre_m=(0.0, 0.0, 0.0),
            radius_m=0.05,
            length_m=0.4,
            axis=(0.0, 0.0, 0.0),
        )


def test_a_surface_takes_a_patch_or_a_mesh_and_not_both() -> None:
    """Two parameterisations of one surface would each claim the same cell."""
    from irsim.config.scene import PatchSpec

    mesh = MeshSpec(shape="sphere", centre_m=(0.0, 0.0, 0.0), radius_m=0.05)
    patch = PatchSpec(
        origin_m=(0.0, 0.0, 0.0),
        u_axis=(1.0, 0.0, 0.0),
        v_axis=(0.0, 1.0, 0.0),
        n_u=2,
        n_v=2,
        du_m=0.1,
        dv_m=0.1,
    )
    with pytest.raises(ValueError, match="`patch:` or `mesh:`, not both"):
        SurfaceSpec(name="a", material="concrete", patch=patch, mesh=mesh)

    # The patch-only extras are refused on a mesh rather than silently ignored: each is a
    # rectangular-grid construction with no mesh equivalent shipped yet.
    from irsim.config.scene import FilmSpec

    with pytest.raises(ValueError, match=r"`film:` \(PH.2\) needs a `patch:`"):
        SurfaceSpec(name="a", material="concrete", mesh=mesh, film=FilmSpec(depth_mm=0.5))
    with pytest.raises(ValueError, match=r"`layers:` \(PT.12\) needs a `patch:`"):
        SurfaceSpec(name="a", material="concrete", mesh=mesh, layers=3)


def test_build_mesh_makes_the_shape_the_author_named() -> None:
    cylinder = build_mesh(
        MeshSpec(
            shape="cylinder",
            centre_m=(1.0, 2.0, 3.0),
            radius_m=0.05,
            length_m=0.4,
            axis=(0.0, 1.0, 0.0),
            segments=16,
            rings=4,
            level=2,
        )
    )
    assert cylinder.n_faces == 2 * 16 * 4 + 2 * 16
    assert cylinder.n_cells == cylinder.n_faces * 4  # level 2 -> 4 cells a face
    lo, hi = cylinder.vertices_m.min(axis=0), cylinder.vertices_m.max(axis=0)
    assert hi[1] - lo[1] == pytest.approx(0.4)  # the length lies along the axis

    sphere = build_mesh(
        MeshSpec(shape="sphere", centre_m=(0.0, 0.0, 0.0), radius_m=0.25, rings=8, segments=16)
    )
    assert np.allclose(np.linalg.norm(sphere.vertices_m, axis=-1), 0.25)

    sized = build_mesh(
        MeshSpec(
            shape="cylinder",
            centre_m=(0.0, 0.0, 0.0),
            radius_m=0.05,
            length_m=0.4,
            segments=8,
            rings=2,
            cell_m=0.005,
        )
    )
    assert sized.n_cells > sized.n_faces  # `cell_m` bought subdivision, not one cell a face


@pytest.mark.slow
def test_a_mesh_cell_takes_its_sky_view_from_its_own_normal(scene) -> None:  # type: ignore[no-untyped-def]
    """`MeshCellForcing`'s other term. V_s = (1 + n·up)/2, so a cell on the crown sees the whole
    sky, one on the underside sees none of it, and one on the flank sees half — where every cell
    of a patch shares the surface's single tilt."""
    forcing = scene.mesh_fields["arm_n"].field.forcing_at
    sky = forcing.sky_view
    up = scene.mesh_fields["arm_n"].patch.cell_normal @ UP
    assert float(sky.max()) > 0.98 and float(sky.min()) < 0.02
    assert np.allclose(sky, 0.5 * (1.0 + up), atol=1e-12)
    assert float(np.ptp(sky)) > 0.9
