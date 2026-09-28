"""TC.10 at the scene level -- `thermal.object_exchange: true` is one switch.

Off (absent, or `false`), every field is the separate solve it was, bit for bit. On, the scene's
plain patches become one exchange group: a warm steel pan parked over asphalt on a clear night
leaves the road beneath it warmer than the open road, with no hand-placed rectangle anywhere in
the scene -- the shape on the asphalt is a consequence of the geometry. And the surfaces that
cannot join are refused by name rather than left out. ADR 0157; roadmap TC.10.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from irsim.scene import Scene
from irsim.thermal.object_exchange import ExchangedField, ObjectExchange

REPO = pathlib.Path(__file__).resolve().parents[2]
HEAD = REPO / "configs/scenes/wall_half_in_sun.yaml"

THERMAL = """
  thermal:
    spin_up_hours: 24.0
    tick_s: 30.0
{switch}
    surfaces:
      # 3 m x 2.4 m of asphalt, 0.2 m cells, facing up (u x v = +z), centred on the origin
      - name: asphalt
        material: asphalt_dry
        tilt_deg: 0.0
        patch:
          origin_m: [-1.5, -1.2, 0.0]
          u_axis: [1.0, 0.0, 0.0]
          v_axis: [0.0, 1.0, 0.0]
          n_u: 15
          n_v: 12
          du_m: 0.2
          dv_m: 0.2
          thickness_m: 0.1
          frame: world
          prim_path: /World/Road
      # a 1.6 m x 0.8 m steel pan 0.3 m over it, facing down (u x v = -z): a parked car's floor
      - name: pan
        material: rusted_steel
        tilt_deg: 180.0
        patch:
          origin_m: [-0.8, 0.4, 0.3]
          u_axis: [1.0, 0.0, 0.0]
          v_axis: [0.0, -1.0, 0.0]
          n_u: 8
          n_v: 4
          du_m: 0.2
          dv_m: 0.2
          thickness_m: 0.002
          frame: world
          prim_path: /World/Car/pan
{extra}
"""


def _scene_yaml(
    tmp_path: pathlib.Path, switch: str, extra: str = "", name: str = "s"
) -> pathlib.Path:
    """The wall scene's head (site, weather, presets), started at night, over a new block."""
    head, _, _ = HEAD.read_text().partition("  thermal:")
    head = head.replace('start_utc: "2024-06-21T16:00:00Z"', 'start_utc: "2024-06-21T23:00:00Z"')
    assert "T23:00:00Z" in head
    out = tmp_path / f"{name}.yaml"
    out.write_text(head.rstrip() + "\n" + THERMAL.format(switch=switch, extra=extra))
    return out


def _road_cells(scene: Scene, t_s: float) -> tuple[np.ndarray, np.ndarray]:
    fld = scene.surface_fields["asphalt"]
    fld.advance_to(t_s)
    temps = np.asarray(fld.temperature_at(t_s), dtype=np.float64)
    centres = fld.patch.cell_centres()
    under = (np.abs(centres[:, 0]) < 0.8) & (np.abs(centres[:, 1]) < 0.4)
    return temps, under


def test_absent_and_false_are_the_same_scene_bit_for_bit(tmp_path, tophat_lwir_lut) -> None:  # type: ignore[no-untyped-def]
    absent = Scene.from_file(_scene_yaml(tmp_path, "", name="absent"), {"lwir": tophat_lwir_lut})
    off = Scene.from_file(
        _scene_yaml(tmp_path, "    object_exchange: false", name="off"), {"lwir": tophat_lwir_lut}
    )
    assert absent.object_exchange is None and off.object_exchange is None
    t = absent.t0_s + 1800.0
    for name in ("asphalt", "pan"):
        absent.surface_fields[name].advance_to(t)
        off.surface_fields[name].advance_to(t)
        assert absent.surface_fields[name].state_hash() == off.surface_fields[name].state_hash()
        assert not isinstance(off.surface_fields[name], ExchangedField)


def test_the_switch_leaves_a_car_shaped_patch_on_the_road_at_night(
    tmp_path, tophat_lwir_lut
) -> None:  # type: ignore[no-untyped-def]
    """Clear night, 01:00 local: the open asphalt radiates to a cold sky; the cells under the
    steel pan see the pan instead. After an hour the difference is a patch the shape of the pan,
    and there is no rectangle anywhere in the scene that says so."""
    off = Scene.from_file(_scene_yaml(tmp_path, "", name="off"), {"lwir": tophat_lwir_lut})
    on = Scene.from_file(
        _scene_yaml(tmp_path, "    object_exchange: true", name="on"), {"lwir": tophat_lwir_lut}
    )
    assert isinstance(on.object_exchange, ObjectExchange)
    assert set(on.object_exchange.names) == {"asphalt", "pan"}
    for name in ("asphalt", "pan"):
        assert isinstance(on.surface_fields[name], ExchangedField)
    t = on.t0_s + 3600.0
    flat, under = _road_cells(off, t)
    assert np.ptp(flat) == 0.0, "no occluder, no exchange: one temperature"
    warm, under = _road_cells(on, t)
    assert on.surface_fields["pan"].latest_t_s == t, "the pan moved with the road"
    excess = warm[under].mean() - warm[~under].mean()
    # 0.25 K after one hour, measured: 0.1 m of asphalt has a time constant of hours, so the
    # patch grows slowly -- which is also why it is still there at dawn
    assert excess > 0.2, (warm[under].mean(), warm[~under].mean())
    assert warm[under].mean() > flat[0], "the road under the pan is warmer than without it"
    # the pan sees the road too: it cools less than the same pan over an unsolved surround
    pan_on = np.asarray(on.surface_fields["pan"].temperature_at(t), dtype=np.float64).mean()
    off.surface_fields["pan"].advance_to(t)
    pan_off = np.asarray(off.surface_fields["pan"].temperature_at(t), dtype=np.float64).mean()
    assert pan_on != pan_off
    # what the road gained of the pan's emission is what the pan sent toward it, to 1e-6
    temps = np.concatenate(
        [on.surface_fields[n].field.latest_state_k for n in on.object_exchange.names]
    )
    absorbed = on.object_exchange.absorbed_power_w(temps)
    emitted = on.object_exchange.emitted_toward_power_w(temps)
    i, j = on.object_exchange.index("asphalt"), on.object_exchange.index("pan")
    eps_road = float(on.object_exchange.cell_emissivity[on.object_exchange.cells_of(i)][0])
    assert absorbed[i, j] == pytest.approx(eps_road * emitted[j, i], rel=1e-6)


def test_the_switch_needs_two_bodies_and_refuses_what_cannot_join(
    tmp_path, tophat_lwir_lut
) -> None:  # type: ignore[no-untyped-def]
    layered = """      - name: slab
        material: concrete
        layers: 4
        patch:
          origin_m: [3.0, 0.0, 0.0]
          u_axis: [1.0, 0.0, 0.0]
          v_axis: [0.0, 1.0, 0.0]
          n_u: 2
          n_v: 2
          du_m: 0.5
          dv_m: 0.5
          thickness_m: 0.2
          frame: world
"""
    with pytest.raises(ValueError, match="slab.*coupled stack"):
        Scene.from_file(
            _scene_yaml(tmp_path, "    object_exchange: true", extra=layered, name="layered"),
            {"lwir": tophat_lwir_lut},
        )
    one_body = THERMAL.format(switch="    object_exchange: true", extra="").split(
        "      # a 1.6 m"
    )[0]
    head, _, _ = HEAD.read_text().partition("  thermal:")
    single = tmp_path / "single.yaml"
    single.write_text(head.rstrip() + "\n" + one_body)
    with pytest.raises(ValueError, match="at least two surfaces"):
        Scene.from_file(single, {"lwir": tophat_lwir_lut})
