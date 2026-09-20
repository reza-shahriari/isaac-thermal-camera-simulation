"""MP.4b — the scene point-wise temperature exists for, checked without a renderer.

`test_the_bonnet_is_a_gradient_and_the_frame_0_panel_is_not` is the milestone's own acceptance:
one prim, flat at the start, several kelvin across by the end. Every frame this project rendered
before ADR 0087 would fail it by the whole gradient, because a per-prim bridge has one number to
give.

`test_the_wheels_do_not_warm` is here because it is the result a viewer will disbelieve. §6.6 makes
tyre heating flexing work and brake heating kinetic energy, so a car idling in a car park has cold
wheels however long it idles. Pinning it stops a later change from quietly "fixing" the wheels.

`test_the_road_patch_covers_the_camera_frame` guards the tedious failure: a patch smaller than the
frame makes `PointwiseTemperature` raise at render time, half an hour into a Kit session.

docs/physics-model.md §6.1, §6.6; ADR 0087, ADR 0088, ADR 0089
"""

from __future__ import annotations

import math
import pathlib

import numpy as np
import pytest

from irsim.scene import Scene
from irsim.thermal.spatial_sources import clamp_view_factor_sum, patch_view_factors
from irsim_isaac.car_demo import (
    CameraSetup,
    CarGeometry,
    build_car_demo,
    describe,
)

# GT.1: this whole module is the slow tier -- a validation bench or an end-to-end frame rather
# than a unit test. `make test` skips it; `make test-slow` and `make check` run it.
pytestmark = pytest.mark.slow

REPO = pathlib.Path(__file__).resolve().parents[2]
SCENE_YAML = REPO / "configs/scenes/car_ignition_overcast_night.yaml"
CLEAR_YAML = REPO / "configs/scenes/car_ignition_clear_night.yaml"
RUN_S = 1500.0
NETD_K = 0.050  # the Boson's, for expressing contrasts in units a camera can resolve


@pytest.fixture(scope="module")
def demo_at_end():  # type: ignore[no-untyped-def]
    """The scene marched to `RUN_S`, with the state at t=0 kept for comparison."""
    scene = Scene.from_file(SCENE_YAML)
    demo = build_car_demo(scene, author=False)
    start = describe(demo, scene, scene.t0_s)
    start_bonnet = np.asarray(demo.bonnet_field.temperature_at(scene.t0_s), dtype=np.float64)

    t_abs = scene.t0_s + RUN_S
    for solver in scene.targets.values():
        while solver.state.t_s < t_abs - 1e-6:
            solver.advance(solver.state.t_s, min(10.0, t_abs - solver.state.t_s))
    demo.bonnet_field.advance_to(t_abs)
    demo.ground_field.advance_to(t_abs)
    return scene, demo, start, start_bonnet, describe(demo, scene, t_abs)


# ---------------------------------------------------------------------------------------------
# the milestone's own acceptance
# ---------------------------------------------------------------------------------------------


def test_the_bonnet_is_a_gradient_and_the_frame_0_panel_is_not(demo_at_end) -> None:  # type: ignore[no-untyped-def]
    _scene, demo, start, start_bonnet, end = demo_at_end

    # Frame 0: one prim, one temperature, to well under a millikelvin.
    assert start["bonnet_gradient_k"] < 1e-3
    assert float(start_bonnet.max() - start_bonnet.min()) < 1e-3

    # By the end: several kelvin across the same prim, which is 100+ noise-equivalent steps.
    assert end["bonnet_gradient_k"] > 3.0
    assert end["bonnet_gradient_k"] / NETD_K > 60.0
    # And it is a rise over ambient, not a redistribution.
    assert end["bonnet_rise_k"] > 3.0


def test_the_bonnet_gradient_falls_away_from_the_engine_and_is_not_a_step(demo_at_end) -> None:  # type: ignore[no-untyped-def]
    """A step would mean the view factor was a mask; it is a configuration factor."""
    _scene, demo, _start, _sb, _end = demo_at_end
    patch = demo.bonnet_field.patch
    image = np.asarray(
        demo.bonnet_field.temperature_at(demo.bonnet_field.latest_t_s), dtype=np.float64
    ).reshape(patch.shape)

    row = image[patch.n_v // 2]
    peak = int(np.argmax(row))
    assert np.all(np.diff(row[peak:]) <= 1e-6), "temperature must not rise away from the engine"
    assert np.all(np.diff(row[: peak + 1]) >= -1e-6)
    # A step would put nearly all the drop in one or two cells; a configuration factor spreads it.
    drop = row.max() - row.min()
    biggest_step = float(np.abs(np.diff(row)).max())
    assert biggest_step < 0.35 * drop, f"one cell carries {biggest_step / drop:.0%} of the fall"


def test_the_road_under_the_car_warms_and_the_far_road_does_not(demo_at_end) -> None:  # type: ignore[no-untyped-def]
    _scene, demo, start, _sb, end = demo_at_end
    assert abs(start["road_patch_k"]) < 1e-3, "frame 0 road is uniform"
    assert end["road_patch_k"] > 0.4, "the car should leave a warm patch"
    assert end["road_patch_k"] / NETD_K > 8.0
    # The far road is still cooling on its own, so the patch is a *contrast*, not a scene warm-up.
    assert end["road_far_k"] < start["road_far_k"]


def test_the_road_patch_falls_with_distance_from_the_car(demo_at_end) -> None:  # type: ignore[no-untyped-def]
    """The view factor's falloff, seen on the surface it lands on."""
    _scene, demo, _start, _sb, _end = demo_at_end
    ground = demo.ground_field
    image = np.asarray(ground.temperature_at(ground.latest_t_s), dtype=np.float64)
    centres = ground.patch.cell_centres()
    distance = np.linalg.norm(centres[:, [0, 2]] - np.array([0.0, 0.15]), axis=1)

    bins = [(0.0, 1.5), (1.5, 3.0), (3.0, 6.0), (6.0, 10.0)]
    means = [float(image[(distance >= lo) & (distance < hi)].mean()) for lo, hi in bins]
    assert all(a > b for a, b in zip(means[:-1], means[1:], strict=True)), means


def test_the_wheels_do_not_warm(demo_at_end) -> None:  # type: ignore[no-untyped-def]
    """§6.6: tyre heating is flexing work and brake heating is kinetic energy. Neither is idling.

    A viewer expects a running car's wheels to glow. They do not, and the demo says so rather than
    staging a rise nobody computed.
    """
    _scene, _demo, _start, _sb, end = demo_at_end
    assert abs(end["tyre_rise_k"]) < 0.01
    assert end["bonnet_rise_k"] > 100.0 * max(abs(end["tyre_rise_k"]), 1e-6)


def test_the_engine_reaches_its_6_6_node_temperature(demo_at_end) -> None:  # type: ignore[no-untyped-def]
    """The whole chain rests on §6.6's own numbers, so state what they came out as."""
    from irsim.thermal.vehicle import VEHICLE_HEAT_SOURCES, first_order_rise

    _scene, _demo, _start, _sb, end = demo_at_end
    spec = VEHICLE_HEAT_SOURCES["engine_bay"]
    expected = float(first_order_rise(RUN_S - 30.0, 0.45 * spec.delta_t_max_k, spec.tau_rise_s))
    assert end["engine_bay_k"] - end["t_air_k"] == pytest.approx(expected, abs=0.05)


# ---------------------------------------------------------------------------------------------
# the geometry the render path depends on
# ---------------------------------------------------------------------------------------------


def test_the_road_patch_covers_the_camera_frame() -> None:
    """Every ground point the camera can see must land in a cell, or MP.3 raises mid-render."""
    scene = Scene.from_file(SCENE_YAML)
    demo = build_car_demo(scene, author=False)
    cam = demo.camera
    eye = np.array(cam.position_m())
    aim = np.array([0.0, cam.aim_height_m, 0.0])
    forward = (aim - eye) / np.linalg.norm(aim - eye)
    right = np.cross(forward, np.array([0.0, 1.0, 0.0]))
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)

    corners = []
    for sh in (-1.0, 1.0):
        for sv in (-1.0, 1.0):
            d = (
                forward
                + sh * math.tan(math.radians(0.5 * 30.7)) * right
                + sv * math.tan(math.radians(0.5 * 24.8)) * up
            )
            if d[1] >= -1e-6:
                continue  # looks at or above the horizon: no ground intersection
            corners.append(eye + d * (-eye[1] / d[1]))
    assert corners, "a 45-degree down-look must hit the ground somewhere"
    inside = demo.ground_field.patch.contains(np.stack(corners))
    assert bool(np.all(inside)), f"frame corners outside the road patch: {np.stack(corners)}"


def test_the_bonnet_patch_claims_the_bonnet_and_nothing_else() -> None:
    """ADR 0087's slab rule at scene scale: the road and the cabin roof project into it."""
    geom = CarGeometry()
    scene = Scene.from_file(SCENE_YAML)
    demo = build_car_demo(scene, author=False)
    patch = demo.bonnet_field.patch
    z = geom.bonnet_centre_z_m
    on_bonnet = np.array([[0.0, geom.bonnet_y_m + geom.bonnet_thickness_m, z]])
    on_road = np.array([[0.0, 0.0, z]])
    on_roof = np.array([[0.0, geom.bonnet_y_m + geom.cabin_height_m, z]])
    assert bool(patch.contains(on_bonnet)[0])
    assert not bool(patch.contains(on_road)[0]), "the bonnet patch claimed the road below it"
    assert not bool(patch.contains(on_roof)[0]), "the bonnet patch claimed the cabin roof"


def test_every_rendered_prim_has_a_thermal_node() -> None:
    """The bridge refuses a rendered prim with no temperature; catch it here, not in Kit."""
    scene = Scene.from_file(SCENE_YAML)
    demo = build_car_demo(scene, author=False)
    geom = demo.geometry
    for part in geom.parts():
        path = f"{demo.car_root}/{part.name}"
        assert path in demo.prim_to_target
        assert demo.prim_to_target[path] in scene.targets
    assert demo.prim_to_target[demo.road_path] == "asphalt"


def test_the_bonnet_sees_most_of_the_bay_and_the_wings_see_little() -> None:
    """The gradient's cause, isolated from the solve."""
    geom = CarGeometry()
    scene = Scene.from_file(SCENE_YAML)
    demo = build_car_demo(scene, author=False)
    view = patch_view_factors(demo.bonnet_field.patch, geom.engine_bay())
    grid = view.reshape(demo.bonnet_field.patch.shape)
    assert grid[grid.shape[0] // 2, grid.shape[1] // 2] > 0.7
    assert grid[0, 0] < 0.25
    assert float(view.max() / max(view.min(), 1e-9)) > 5.0


@pytest.mark.parametrize("yaml_path", [SCENE_YAML, CLEAR_YAML])
def test_ground_radiator_view_factors_never_exceed_one(yaml_path: pathlib.Path) -> None:
    """PT.3 / ADR 0090: `underbody`, `engine_bay` and `exhaust_pipe` are nested, not disjoint.

    Summing their individually-correct view factors used to reach 1.40 under the engine bay --
    more sky-and-source credit than a plane element can physically receive. `build_ground_field`
    now clamps them before use; this reproduces the raw (unclamped) sum the way it used to be
    computed, on both car scenes' actual geometry and patch, and asserts the physical bound.
    """
    geom = CarGeometry()
    scene = Scene.from_file(yaml_path)
    demo = build_car_demo(scene, author=False)
    patch = demo.ground_field.patch
    raw_total = sum(patch_view_factors(patch, rect) for _, rect in geom.ground_radiators())
    assert raw_total.max() > 1.0, "the nesting this test guards against should still be present"

    clamped = clamp_view_factor_sum(
        [patch_view_factors(patch, rect) for _, rect in geom.ground_radiators()]
    )
    clamped_total = sum(clamped)
    assert np.all(clamped_total <= 1.0 + 1e-6), float(clamped_total.max())


def test_the_camera_stands_where_it_says_it_does() -> None:
    cam = CameraSetup(depression_deg=45.0, range_m=27.0, azimuth_deg=0.0, aim_height_m=0.8)
    x, y, z = cam.position_m()
    assert x == pytest.approx(0.0, abs=1e-9)
    assert y - cam.aim_height_m == pytest.approx(27.0 * math.sin(math.radians(45.0)), abs=1e-9)
    assert z == pytest.approx(27.0 * math.cos(math.radians(45.0)), abs=1e-9)
    # The depression the geometry actually delivers, from the camera to the aim point.
    horizontal = math.hypot(x, z)
    assert math.degrees(math.atan2(y - cam.aim_height_m, horizontal)) == pytest.approx(45.0)


# ---------------------------------------------------------------------------------------------
# the pair: what the sky does, with the engine held identical
# ---------------------------------------------------------------------------------------------


def _patch_after(yaml_path: pathlib.Path, seconds: float) -> tuple[float, float]:
    """``(ground patch, bonnet gradient)`` after ``seconds`` on one scene."""
    scene = Scene.from_file(yaml_path)
    demo = build_car_demo(scene, author=False)
    t_abs = scene.t0_s + seconds
    for solver in scene.targets.values():
        while solver.state.t_s < t_abs - 1e-6:
            solver.advance(solver.state.t_s, min(20.0, t_abs - solver.state.t_s))
    demo.bonnet_field.advance_to(t_abs)
    demo.ground_field.advance_to(t_abs)
    row = describe(demo, scene, t_abs)
    return row["road_patch_k"], row["bonnet_gradient_k"]


def test_a_clear_sky_puts_far_more_of_the_patch_on_the_road_than_the_engine_does() -> None:
    """ADR 0088's occlusion term, isolated by holding the engine identical.

    The two scenes run the **same** §6.6 node on the **same** load profile, so the car's own
    emission is identical in both. The only difference is the sky, and it lands on the road: under
    a clear night the car blocks a 40-50 K sky depression, and the patch comes out nearly three
    times the overcast one. A model that added the car's emission without removing the sky it
    occludes would give the *same* number for both scenes.

    The bonnet moves the other way -- slightly smaller under the clear sky, because the skin is
    also radiating to it -- which is the cross-check that this is the sky and not a scale factor.
    """
    overcast_patch, overcast_bonnet = _patch_after(SCENE_YAML, 1800.0)
    clear_patch, clear_bonnet = _patch_after(CLEAR_YAML, 1800.0)

    assert overcast_patch > 0.5
    assert clear_patch > 2.0 * overcast_patch, f"clear {clear_patch}, overcast {overcast_patch}"
    assert clear_bonnet < overcast_bonnet, "a colder sky must cool the bonnet, not warm it"


def test_the_ground_patch_needs_time_because_the_field_starts_uniform() -> None:
    """A known limit, pinned so it is not mistaken for a result.

    §12.3 solves the asphalt as **one** surface, so the field inherits a uniform spun-up state --
    the road as it would be with no car on it. A car that has stood there for hours would already
    have its patch, and under a clear sky that is most of what a night thermal image of a car park
    shows. Spinning the field up with the car present is what would fix it; until then frame 0 is
    the moment the car arrived, and every patch here is one the run itself grew.
    """
    for yaml_path in (SCENE_YAML, CLEAR_YAML):
        patch_0, _ = _patch_after(yaml_path, 0.0)
        assert abs(patch_0) < 1e-3, f"{yaml_path.name} starts with a patch it did not earn"


def test_the_engine_is_identical_in_both_scenes() -> None:
    """What makes the comparison a comparison rather than two different cars."""
    from irsim.config.scene import load_scene_config

    a = {t.name: t for t in load_scene_config(SCENE_YAML).scene.targets}
    b = {t.name: t for t in load_scene_config(CLEAR_YAML).scene.targets}
    assert set(a) == set(b)
    for name in ("engine_bay", "exhaust_pipe", "underbody", "tyre"):
        assert a[name].solver == b[name].solver == "vehicle_source"
        assert a[name].source == b[name].source
        assert a[name].load == b[name].load
        assert a[name].load_s == b[name].load_s


# ---------------------------------------------------------------------------------------------
# PT.17: the grids are the scene config's own
# ---------------------------------------------------------------------------------------------


def test_the_demo_grids_are_the_scene_config_s_own() -> None:
    """The owner's bar, on this lane: the YAML declares the patch, Python only checks it."""
    scene = Scene.from_file(SCENE_YAML)
    demo = build_car_demo(scene, author=False)
    assert demo.bonnet_field.patch is scene.patches["bonnet"]
    assert demo.ground_field.patch is scene.patches["asphalt"]
    # And the scene solves the same grids on its own, under the weather alone.
    assert set(scene.surface_fields) == {"asphalt", "bonnet"}
    assert dict(scene.surface_bindings()) == {
        "/World/Road": scene.surface_fields["asphalt"],
        "/World/Car/bonnet": scene.surface_fields["bonnet"],
    }


def test_two_loads_of_the_config_give_bit_identical_fields() -> None:
    """Declared numbers are numbers: no camera-derived sizing can make two runs differ."""
    fields = []
    for _ in range(2):
        scene = Scene.from_file(SCENE_YAML)
        demo = build_car_demo(scene, author=False)
        t = scene.t0_s + 600.0
        demo.bonnet_field.advance_to(t)
        demo.ground_field.advance_to(t)
        fields.append((demo.bonnet_field.temperature_at(t), demo.ground_field.temperature_at(t)))
    assert np.array_equal(fields[0][0], fields[1][0])
    assert np.array_equal(fields[0][1], fields[1][1])


def _scene_with(tmp_path: pathlib.Path, old: str, new: str) -> Scene:
    text = SCENE_YAML.read_text()
    assert text.count(old) == 1, old
    path = tmp_path / "edited.yaml"
    path.write_text(text.replace(old, new, 1))
    return Scene.from_file(path)


def test_a_declared_bonnet_patch_off_the_bonnet_is_refused(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """A grid 24 cm above the skin is partly cabin roof and partly air; it must not solve."""
    scene = _scene_with(
        tmp_path, "origin_m: [-0.82, 1.06, -1.925]", "origin_m: [-0.82, 1.30, -1.925]"
    )
    with pytest.raises(ValueError, match=r"bonnet patch does not sit on CarGeometry's bonnet"):
        build_car_demo(scene, author=False)


def test_a_declared_road_patch_smaller_than_the_frame_is_refused(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """The tedious failure, caught at build time with the two footprints in the message."""
    scene = _scene_with(tmp_path, "          n_u: 102\n", "          n_u: 20\n")
    with pytest.raises(ValueError, match=r"road patch cannot serve this camera.*footprint"):
        build_car_demo(scene, author=False)


def test_a_scene_that_declares_no_grid_still_gets_the_hand_built_ones(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """The pre-PT.17 path is kept, so an older scene renders as it did."""
    text = SCENE_YAML.read_text()
    head, _, _ = text.partition("  thermal:")
    path = tmp_path / "bare.yaml"
    path.write_text(
        head.rstrip()
        + "\n  thermal:\n    spin_up_hours: 48.0\n    tick_s: 60.0\n    surfaces:\n"
        + "      - {name: asphalt, material: asphalt_dry, tilt_deg: 0.0}\n"
    )
    scene = Scene.from_file(path)
    demo = build_car_demo(scene, author=False)
    assert scene.patches == {}
    assert demo.bonnet_field.patch.n_cells == 23 * 19
    assert demo.ground_field.patch.n_cells > 0


# ---------------------------------------------------------------------------------------------
# PT.18: the sun reaches the fields, and the car shades the road
# ---------------------------------------------------------------------------------------------


def _noon_scene(tmp_path: pathlib.Path) -> Scene:
    """The clear scene at 13:00 local (11:00Z) on the same June day: a high sun from the south.

    The *clear* file: the overcast one carries no direct beam at any hour, by design.
    """
    text = CLEAR_YAML.read_text()
    old = 'start_utc: "2024-06-22T01:00:00Z"'
    assert text.count(old) == 1
    path = tmp_path / "noon.yaml"
    path.write_text(text.replace(old, 'start_utc: "2024-06-21T11:00:00Z"', 1))
    return Scene.from_file(path)


def test_the_night_scenes_carry_no_beam_and_the_noon_one_does(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Before PT.18 the fields had no solar term at all: a noon run rendered a sun-free bonnet."""
    night = build_car_demo(Scene.from_file(CLEAR_YAML), author=False)
    scene_n = Scene.from_file(CLEAR_YAML)
    assert not scene_n.solar_terms(scene_n.t0_s).above_horizon
    assert np.all(np.asarray(night.bonnet_field.field.forcing_at(scene_n.t0_s).q_solar_w_m2) == 0.0)

    scene = _noon_scene(tmp_path)
    demo = build_car_demo(scene, author=False)
    sun = scene.solar_terms(scene.t0_s)
    assert sun.elevation_deg > 60.0 and sun.dni_w_m2 > 800.0
    q_bonnet = np.asarray(demo.bonnet_field.field.forcing_at(scene.t0_s).q_solar_w_m2)
    lit = q_bonnet > 800.0
    assert lit.mean() > 0.7, "a bonnet facing straight up under a 65 degree sun"
    # The sun is behind the car (south is +Z on this stage), so the cabin's front face lays a
    # strip of shadow on the rear of the bonnet: those cells carry the diffuse sky alone.
    grid = lit.reshape(demo.bonnet_field.patch.shape)  # (n_v, n_u), v along +Z toward the cabin
    assert not lit.all() and np.allclose(q_bonnet[~lit], sun.dhi_w_m2)
    assert grid[0].all() and not grid[-1].any(), "the shadow is a strip at the cabin end"


def test_the_car_shades_the_road_beside_it_and_the_shaded_road_runs_colder(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """The car's own faces are occluders for the road field; the shadow lands north of the car.

    Cells the car shades but does not stand over (its radiators see them at zero) fall below
    the sunlit road around them by kelvins within 25 minutes: asphalt at 60 kJ m^-2 K^-1 losing
    ~500 W/m^2 of absorbed beam is 0.5 K a minute.
    """
    scene = _noon_scene(tmp_path)
    demo = build_car_demo(scene, author=False)
    road = demo.ground_field
    patch = road.patch
    t0 = scene.t0_s
    sun_world = scene.world_frame.to_world(scene.solar_terms(t0).direction_enu)
    assert sun_world[1] > 0.9, "the sun is high, and up is +Y on this stage"
    from irsim.thermal.shadow import cell_shadow

    lit = cell_shadow(patch, sun_world, demo.geometry.shadow_casters())
    shaded = lit == 0.0
    assert 0.005 < shaded.mean() < 0.05, shaded.mean()  # a car's shadow on a 30 m road
    # Shaded cells that lie outside the car's footprint, so no radiator term touches them.
    centres = patch.cell_centres()
    half_w, half_l = 0.5 * demo.geometry.width_m, 0.5 * demo.geometry.length_m
    outside = (np.abs(centres[:, 0]) > half_w) | (np.abs(centres[:, 2]) > half_l)
    beside = shaded & outside
    far = ~shaded & outside
    assert beside.sum() > 5 and far.sum() > 1000, (beside.sum(), far.sum())

    start = np.asarray(road.temperature_at(t0), dtype=np.float64)
    assert float(np.ptp(start)) == 0.0, "the field starts from the uniform spun-up asphalt"
    road.advance_to(t0 + RUN_S)
    end = np.asarray(road.temperature_at(t0 + RUN_S), dtype=np.float64)
    contrast = float(end[far].mean() - end[beside].mean())
    assert contrast > 2.0, f"the shaded road is only {contrast:.2f} K colder after {RUN_S:.0f} s"
    # The sunlit road warmed over the run; the shadowed road cooled or held.
    assert end[far].mean() > start[far].mean()


def test_a_scene_whose_world_frame_is_not_y_up_is_refused(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """The stage is +Y up; a scene that leaves the v8 frame at its ENU default would put the sun
    on the wrong side of the car, quietly."""
    scene = _scene_with(
        tmp_path, "  world_frame: {up: [0.0, 1.0, 0.0], north: [0.0, 0.0, -1.0]}\n", ""
    )
    with pytest.raises(ValueError, match=r"world_frame\.up"):
        build_car_demo(scene, author=False)
