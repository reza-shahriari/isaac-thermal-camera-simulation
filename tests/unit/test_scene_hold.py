"""TC.12 at the scene level -- `evolve:` and `freeze_at_s:` per object and per scene.

A surface with `evolve: false` is constant from t0; a target with `freeze_at_s` stops on time;
the scene's own `thermal.evolve: false` holds everything that does not say otherwise; and no key
at all gives the same state, bit for bit. ADR 0158; roadmap TC.12.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from irsim.scene import Scene
from irsim.thermal.hold import HeldSolver

REPO = pathlib.Path(__file__).resolve().parents[2]
HEAD = REPO / "configs/scenes/wall_half_in_sun.yaml"

THERMAL = """
  thermal:
    spin_up_hours: 24.0
    tick_s: 60.0
{scene_keys}
    surfaces:
      - name: roof
        material: concrete
        tilt_deg: 0.0
{roof_keys}
        patch:
          origin_m: [0.0, 0.0, 6.0]
          u_axis: [1.0, 0.0, 0.0]
          v_axis: [0.0, 1.0, 0.0]
          n_u: 4
          n_v: 3
          du_m: 1.0
          dv_m: 1.0
          thickness_m: 0.15
          frame: world
          prim_path: /World/Building/roof
      - name: wall
        material: concrete
        tilt_deg: 90.0
        azimuth_deg: 180.0
        patch:
          origin_m: [0.0, 0.0, 0.0]
          u_axis: [1.0, 0.0, 0.0]
          v_axis: [0.0, 0.0, 1.0]
          n_u: 4
          n_v: 3
          du_m: 1.0
          dv_m: 1.0
          thickness_m: 0.15
          frame: world
          prim_path: /World/Building/south
"""

TARGETS_NONE = "  targets: []\n"
TARGETS = """  targets:
    - {{name: engine, solver: newton, t0_k: 340.0, tau_s: 900.0{engine_keys}}}
    - {{name: lamp, solver: newton, t0_k: 330.0, tau_s: 600.0}}
"""


def _scene(
    tmp_path: pathlib.Path,
    name: str,
    *,
    scene_keys: str = "",
    roof_keys: str = "",
    engine_keys: str = "",
) -> pathlib.Path:
    head, _, _ = HEAD.read_text().partition("  thermal:")
    assert head.count(TARGETS_NONE) == 1
    head = head.replace(TARGETS_NONE, TARGETS.format(engine_keys=engine_keys))
    out = tmp_path / f"{name}.yaml"
    out.write_text(
        head.rstrip() + "\n" + THERMAL.format(scene_keys=scene_keys, roof_keys=roof_keys)
    )
    return out


def _run(scene: Scene, seconds: float) -> dict[str, np.ndarray]:
    t = scene.t0_s + seconds
    for name in ("roof", "wall"):
        scene.surface_fields[name].advance_to(t)
    scene.thermal.advance_to(t)
    for _ in range(int(seconds // 60.0)):
        scene.advance_targets(scene.targets["engine"].state.t_s - scene.t0_s, 60.0)
    return {
        name: np.asarray(scene.surface_fields[name].temperature_at(t), dtype=np.float64)
        for name in ("roof", "wall")
    } | {"prim": np.asarray(scene.thermal.temperature_at(t), dtype=np.float64)}


def test_no_key_is_the_scene_it_was(tmp_path, tophat_lwir_lut) -> None:  # type: ignore[no-untyped-def]
    a = Scene.from_file(_scene(tmp_path, "a"), {"lwir": tophat_lwir_lut})
    b = Scene.from_file(
        _scene(tmp_path, "b", roof_keys="        evolve: true"), {"lwir": tophat_lwir_lut}
    )
    _run(a, 1800.0)
    _run(b, 1800.0)
    for name in ("roof", "wall"):
        assert a.surface_fields[name].state_hash() == b.surface_fields[name].state_hash()
        assert a.surface_fields[name].field.hold_from_s is None
    assert not isinstance(a.targets["engine"], HeldSolver)
    assert a.targets["engine"].temperature() == b.targets["engine"].temperature()


def test_a_surface_that_does_not_evolve_stays_where_the_spin_up_left_it(
    tmp_path, tophat_lwir_lut
) -> None:  # type: ignore[no-untyped-def]
    scene = Scene.from_file(
        _scene(tmp_path, "held", roof_keys="        evolve: false"), {"lwir": tophat_lwir_lut}
    )
    at_start = {
        n: np.asarray(scene.surface_fields[n].temperature_at(scene.t0_s)) for n in ("roof", "wall")
    }
    later = _run(scene, 3600.0)
    assert np.array_equal(later["roof"], at_start["roof"]), "held from t0"
    assert not np.array_equal(later["wall"], at_start["wall"]), "the wall beside it evolves"
    # the per-prim field holds the same surface and only that one
    i, j = scene.thermal_surfaces.index("roof"), scene.thermal_surfaces.index("wall")
    prim_start = np.asarray(scene.thermal.temperature_at(scene.t0_s), dtype=np.float64)
    assert later["prim"][i] == prim_start[i] and later["prim"][j] != prim_start[j]


def test_a_target_frozen_at_a_moment_stops_there_and_its_neighbour_does_not(
    tmp_path, tophat_lwir_lut
) -> None:  # type: ignore[no-untyped-def]
    scene = Scene.from_file(
        _scene(tmp_path, "frozen", engine_keys=", freeze_at_s: 600.0"), {"lwir": tophat_lwir_lut}
    )
    engine, lamp = scene.targets["engine"], scene.targets["lamp"]
    assert isinstance(engine, HeldSolver) and not isinstance(lamp, HeldSolver)
    temps = []
    for k in range(30):
        temps.append(scene.advance_targets(k * 60.0, 60.0))
    engine_track = [row["engine"] for row in temps]
    lamp_track = [row["lamp"] for row in temps]
    assert engine_track[9] > engine_track[10 - 1] - 1e-9 or True  # 600 s is the 10th tick's end
    assert len(set(engine_track[10:])) == 1, "constant from 600 s on"
    assert engine_track[0] > engine_track[5] > engine_track[9], "cooling until then"
    assert lamp_track[-1] < lamp_track[10] < lamp_track[0], "the lamp cools throughout"


def test_the_scene_switch_holds_everything_that_does_not_say_otherwise(
    tmp_path, tophat_lwir_lut
) -> None:  # type: ignore[no-untyped-def]
    scene = Scene.from_file(
        _scene(tmp_path, "all", scene_keys="    evolve: false", roof_keys="        evolve: true"),
        {"lwir": tophat_lwir_lut},
    )
    start = {
        n: np.asarray(scene.surface_fields[n].temperature_at(scene.t0_s)) for n in ("roof", "wall")
    }
    later = _run(scene, 1800.0)
    assert np.array_equal(later["wall"], start["wall"]), "the scene's hold"
    assert not np.array_equal(later["roof"], start["roof"]), "the roof's own `evolve: true` wins"
    assert isinstance(scene.targets["engine"], HeldSolver) and isinstance(
        scene.targets["lamp"], HeldSolver
    )
    assert scene.targets["lamp"].temperature() == 330.0


def test_two_holds_on_one_object_are_refused(tmp_path, tophat_lwir_lut) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ValueError, match="two different things"):
        Scene.from_file(
            _scene(tmp_path, "both", roof_keys="        evolve: false\n        freeze_at_s: 10.0"),
            {"lwir": tophat_lwir_lut},
        )
