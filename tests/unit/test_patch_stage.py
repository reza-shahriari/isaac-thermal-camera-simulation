"""IG.20: a patch scene becomes a stage -- the engine-free half (`irsim_isaac.patch_stage`).

The driver (`scripts/render_patch_scene.py`) rotates an ENU scene config into the Y-up stage
frame before anything is authored, because the Isaac glue and the point bridge both work in stage
coordinates. That is only legitimate if the rotation changes no physics, so the first test is the
one the approach stands on: the rotated road solves to the **same temperature in every cell**,
wet and dry, lit and shaded, as the config it came from. The rest hold the geometry to the
config: each quad is its patch exactly, a shared prim is split so each half keeps its material, a
water film is a visible-only sheet over exactly its region, and the per-prim fallback exists
without ever being a temperature anybody should see.
"""

from __future__ import annotations

import pathlib
from types import SimpleNamespace

import numpy as np
import pytest

from irsim.config.scene import load_scene_config
from irsim.scene import Scene
from irsim_isaac.patch_stage import (
    ENU_TO_STAGE,
    FALLBACK_NODE,
    FILM_LIFT_M,
    QUAD_INSET_M,
    enu_to_stage,
    patch_bindings,
    patch_prims,
    to_stage_frame,
    with_fallback_node,
)

REPO = pathlib.Path(__file__).resolve().parents[2]
ROAD = REPO / "configs" / "scenes" / "wet_road_noon.yaml"
WALL = REPO / "configs" / "scenes" / "wall_half_in_sun.yaml"
CAR = REPO / "configs" / "scenes" / "car_ignition_clear_night.yaml"


def test_the_rotation_is_proper_and_puts_up_and_north_where_the_glue_expects_them() -> None:
    assert np.linalg.det(ENU_TO_STAGE) == pytest.approx(1.0)
    np.testing.assert_allclose(ENU_TO_STAGE @ ENU_TO_STAGE.T, np.eye(3), atol=1e-12)
    assert enu_to_stage((0, 0, 1)) == (0.0, 1.0, 0.0)  # up is +Y
    assert enu_to_stage((0, 1, 0)) == (0.0, 0.0, -1.0)  # north is -Z: heading 0
    assert enu_to_stage((1, 0, 0)) == (1.0, 0.0, 0.0)  # east is +X


def test_the_rotated_road_solves_to_the_same_temperature_in_every_cell() -> None:
    """Wet and dry, lit and shaded, an hour into the drying: a rotation is not physics."""
    enu = Scene.from_config(load_scene_config(ROAD))
    stage = Scene.from_config(to_stage_frame(load_scene_config(ROAD)))
    for name in ("road", "south_wall"):
        a, b = enu.surface_fields[name], stage.surface_fields[name]
        for dt in (0.0, 1800.0, 3600.0):
            a.advance_to(enu.t0_s + dt)
            b.advance_to(stage.t0_s + dt)
            ta = np.asarray(a.temperature_at(enu.t0_s + dt), dtype=np.float64)
            tb = np.asarray(b.temperature_at(stage.t0_s + dt), dtype=np.float64)
            np.testing.assert_allclose(tb, ta, atol=1e-6, err_msg=f"{name} at +{dt:.0f} s")
        # and the cells sit where the rotation says they do
        ca = a.patch.cell_centres()
        np.testing.assert_allclose(b.patch.cell_centres(), ca @ ENU_TO_STAGE.T, atol=1e-9)
    # the shadow is the same shadow: the wall's hard edge falls on the same rows
    road_a, road_b = enu.surface_fields["road"], stage.surface_fields["road"]
    vis_a = road_a.field.forcing_at.cell_visibility(enu.t0_s)
    vis_b = road_b.field.forcing_at.cell_visibility(stage.t0_s)
    np.testing.assert_array_equal(vis_b, vis_a)
    assert 0.05 < 1.0 - float(np.mean(vis_a)) < 0.3


def test_a_config_already_in_a_stage_frame_is_left_alone() -> None:
    car = load_scene_config(CAR)
    assert tuple(car.scene.world_frame.up) != (0.0, 0.0, 1.0)
    assert to_stage_frame(car) is car


def test_every_quad_is_its_patch_and_a_shared_prim_is_split_by_material() -> None:
    config = to_stage_frame(load_scene_config(WALL))
    prims = {p.path: p for p in patch_prims(config)}
    assert config.scene.thermal is not None
    for srf in config.scene.thermal.surfaces:
        p = srf.patch
        assert p is not None
        path = f"/World/Building/west/{srf.name}" if srf.name.startswith("west_") else p.prim_path
        quad = prims[path]
        assert quad.material == srf.material and quad.surface == srf.name
        o, u, v = (np.asarray(x, dtype=np.float64) for x in (p.origin_m, p.u_axis, p.v_axis))
        lu, lv = p.n_u * p.du_m, p.n_v * p.dv_m
        expect = [o, o + u * lu, o + u * lu + v * lv]
        # the patch exactly, drawn QUAD_INSET_M inside it on every edge so no rendered point can
        # land a rounding error outside its field
        np.testing.assert_allclose(
            np.asarray(quad.corners_m[:3]), np.asarray(expect), atol=2.0 * QUAD_INSET_M
        )
        rel = (np.asarray(quad.corners_m) - o) @ np.stack([u, v]).T
        assert rel.min() >= QUAD_INSET_M - 1e-12
        assert (rel[:, 0].max(), rel[:, 1].max()) <= (lu - QUAD_INSET_M + 1e-12, lv)
    # the west face is one config prim with two materials: two quads, one per material
    west = [q for q in prims.values() if q.path.startswith("/World/Building/west/")]
    assert {q.material for q in west} == {"concrete", "etics_render"}
    assert not any(q.companion_only for q in prims.values())
    # and the bindings land on exactly those quads, one field each (a stand-in scene with the
    # same surface -> prim map: solving the building for this would cost half a minute)
    scene = SimpleNamespace(
        surface_fields={s.name: None for s in config.scene.thermal.surfaces},
        patch_prims={s.name: s.patch.prim_path for s in config.scene.thermal.surfaces if s.patch},
    )
    bound = [path for path, _ in patch_bindings(scene)]
    assert sorted(bound) == sorted(prims) and len(set(bound)) == len(bound)


def test_a_water_film_is_a_visible_only_sheet_over_exactly_its_region() -> None:
    config = to_stage_frame(load_scene_config(ROAD))
    prims = patch_prims(config)
    film = [p for p in prims if p.companion_only]
    road = next(p for p in prims if p.surface == "road")
    assert len(film) == 1 and film[0].surface is None and road.companion_only is False
    corners = np.asarray(film[0].corners_m)
    # the west half (u from 0 to 6 m of a 12 m road), full length, lifted along the road's normal
    assert np.ptp(corners[:, 0]) == pytest.approx(6.0 - 2 * QUAD_INSET_M)
    assert corners[:, 0].min() == pytest.approx(QUAD_INSET_M)
    assert np.ptp(corners[:, 2]) == pytest.approx(12.0 - 2 * QUAD_INSET_M)
    lift = corners[:, 1] - np.asarray(road.corners_m)[0, 1]
    np.testing.assert_allclose(np.abs(lift), FILM_LIFT_M, atol=1e-12)


def test_the_fallback_node_is_added_once_and_is_air_temperature() -> None:
    config = with_fallback_node(with_fallback_node(load_scene_config(ROAD)))
    names = [t.name for t in config.scene.targets]
    assert names.count(FALLBACK_NODE) == 1
    node = next(t for t in config.scene.targets if t.name == FALLBACK_NODE)
    assert node.solver == "airframe" and node.offset_k == 0.0
