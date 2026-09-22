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
    tube runs 16 K over the air its underside sits on, and both are cells of one solve on one
    prim. The patched scene beside it gives each arm one value across its whole width.

    The numbers are smaller than `WM.7` first reported them and more correct: `WM.4` traces the
    motor pod's shadow onto the arm's outer end, and `WM.6` lets the tube conduct round itself,
    which the fin equation says must cost a quarter of an unconducted gradient."""
    t = scene.t0_s + PAD_S
    air = float(scene.weather.at(t).t_air_k)
    assert set(scene.meshes) == {"arm_n", "arm_e"}
    assert set(scene.patches) == {"deck", "belly"}  # near-planar surfaces keep the cheaper path

    for name in ("arm_n", "arm_e"):
        field = scene.mesh_fields[name]
        field.advance_to(t)
        crown, underside = _crown_and_underside(field, t)
        assert crown - underside > 14.0, (name, crown, underside)  # 15.4 K and 15.7 K measured
        assert abs(underside - air) < 2.0, (name, underside, air)
        cells = np.asarray(field.temperature_at(t), dtype=np.float64)
        assert float(np.ptp(cells)) > 22.0  # 24.3 K measured, crown to shaded end


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
    # Five bins, not nine: a 16-sided tube has 16 distinct normals, so finer bins are empty.
    edges = np.linspace(-1.0, 1.0, 6)
    means = [
        float(cells[side][(up[side] >= lo) & (up[side] < hi)].mean())
        for lo, hi in zip(edges[:-1], edges[1:], strict=True)
    ]
    assert np.all(np.diff(means) > 0.0), means
    # 15.3 K between the bin facing straight down and the bin facing straight up, measured. It
    # was 27 K before `WM.4` traced the motor pod's shadow onto the arm's outer end and `WM.6`
    # let the tube conduct round itself.
    assert means[-1] - means[0] > 13.0, means


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
    assert crown_pad > 14.0, crown_pad  # 16.4 K above air, measured
    assert crown_cruise < 12.0, crown_cruise  # 8.5 K above air, measured


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
    assert float(np.ptp(opening)) > 22.0, float(np.ptp(opening))  # 24.3 K measured


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
    assert float(np.ptp(out)) > 22.0
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
def test_a_mesh_cell_sees_the_sky_its_own_geometry_leaves_it(scene) -> None:  # type: ignore[no-untyped-def]
    """`MeshCellForcing`'s other term, traced (`WM.4`).

    The tilt's ``V_s = (1 + n·up)/2`` is the *unobstructed* sky, and the inner third of an arm is
    not unobstructed: it runs 25 mm under the body's roof. The traced factor can only take sky
    away, it takes a great deal of it from the sheltered cells, and it still spans nearly the
    whole range across the tube — which is the part a patch's single tilt cannot express at all.
    """
    field = scene.mesh_fields["arm_n"]
    sky = field.field.forcing_at.sky_view
    unobstructed = np.clip(0.5 * (1.0 + field.patch.cell_normal @ UP), 0.0, 1.0)
    assert np.all(sky <= unobstructed + 1e-12), "the trace added sky to a cell"
    lost = unobstructed - sky
    assert float(lost.max()) > 0.9, "no cell is sheltered; the body is not shading the arm"
    assert int((lost > 0.01).sum()) > 0.5 * field.patch.n_cells
    assert float(np.ptp(sky)) > 0.9


@pytest.mark.slow
def test_the_body_shadow_reaches_the_meshed_arm_as_it_reaches_the_patched_one(scene) -> None:  # type: ignore[no-untyped-def]
    """The consistency `WM.7` could not have: one set of occluders, two parameterisations.

    Both scenes declare the same body and pod rectangles. Before `WM.4` they shaded the patched
    arm and not the meshed one, so the same airframe at the same instant cast a shadow in one
    scene and none in the other. `arm_n` runs from y = 0.15 to y = 0.45 and the motor pod covers
    its last 60 mm, so at midday the crown goes dark there and is fully lit along the open span.
    """
    field = scene.mesh_fields["arm_n"]
    t = scene.t0_s + PAD_S
    lit = field.field.forcing_at.cell_visibility(t)
    sky = field.field.forcing_at.sky_view
    centres = field.patch.cell_centres()
    crown = field.patch.cell_normal @ UP > 0.7
    under_pod = crown & (centres[:, 1] > 0.42)
    open_span = crown & (centres[:, 1] > 0.20) & (centres[:, 1] < 0.38)
    assert under_pod.sum() > 0 and open_span.sum() > 0
    assert float(np.mean(lit[under_pod])) == 0.0
    assert float(np.mean(lit[open_span])) == 1.0
    # and the sky goes with the beam: 0.10 under the pod against 0.93 along the open span.
    assert float(np.mean(sky[under_pod])) < 0.2
    assert float(np.mean(sky[open_span])) > 0.9


def test_the_arms_are_cut_near_square_so_the_conduction_operator_is_right() -> None:
    """`ADR 0112`'s authoring rule, on the scene that has to obey it.

    The two-point flux between a mesh's cells is exact when a tube's quads are about 1.4 times
    longer along the axis than around the circumference, and degrades either side of that: long
    thin quads saturate at three quarters of the conductivity they should have, which renders a
    tube's gradient a quarter too strong. The arms shipped at 12.7 to 1 until `WM.6` measured it.

    Read off the config rather than the built mesh, because this is a statement about how the
    scene is **authored** -- it is the thing a person editing the YAML gets wrong.
    """
    spec = load_scene_config(SCENE).scene.thermal
    meshes = [s.mesh for s in spec.surfaces if s.mesh is not None]
    assert meshes, "the scene declares no meshes; the check is pointed at the wrong file"
    for mesh in meshes:
        assert mesh.shape == "cylinder"
        arc_m = 2.0 * np.pi * mesh.radius_m / mesh.segments
        ring_m = mesh.length_m / mesh.rings
        assert 1.0 < ring_m / arc_m < 2.0, (mesh.prim_path, ring_m / arc_m)
        # and the cells are no finer than carbon fibre's own smoothing length (ADR 0111): at
        # L = 8.8 mm a 5.9 mm arc is already past the point where extra cells buy resolution.
        assert arc_m > 0.004, (mesh.prim_path, arc_m)
