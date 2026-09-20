"""PT.17 — a `patch:` block in a scene config buys a solved field, not just a grid.

Until this step a patched surface was still solved as **one facet**: `_build_thermal_field`
stacked one `ThermalProperties` per `SurfaceSpec` and never read `surface.patch`, `Scene.patches`
held geometry nobody consumed, and the only per-cell temperatures in the repository were built by
`car_demo.py` in Python. The docstrings said otherwise. This file is what makes them true.

The acceptance is deliberately an *identity*. With nothing varying across the surface the cells
must reproduce the per-prim value, and because `CellForcing` broadcasts the very numbers the
per-prim solve uses, the two are held to **bit-identity**, not to a tolerance: a field that
changed a surface's temperature merely by existing would be the quiet failure ADR 0087 warns
about, and one millikelvin of drift would hide inside any tolerance a camera could justify.

docs/physics-model.md §6.1, §6.4, §12.3; ADR 0087
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from irsim.config.scene import load_scene_config
from irsim.scene import Scene
from irsim.thermal.surface_field import PlanarThermalField

REPO = pathlib.Path(__file__).resolve().parents[2]
OVERCAST = REPO / "configs/scenes/car_ignition_overcast_night.yaml"
CLEAR = REPO / "configs/scenes/car_ignition_clear_night.yaml"
FACETS = REPO / "configs/scenes/thermal_facet_scene.yaml"

SMALL_PATCH = """
  thermal:
    spin_up_hours: 48.0
    tick_s: 60.0
    surfaces:
      - name: asphalt
        material: {material}
        tilt_deg: 0.0
        patch:
          origin_m: [-1.5, 0.0, -1.2]
          u_axis: [1.0, 0.0, 0.0]
          v_axis: [0.0, 0.0, 1.0]
          n_u: 10
          n_v: 8
          du_m: 0.3
          dv_m: 0.3
          thickness_m: 0.10
          frame: world
          prim_path: /World/Road
      - name: kerb
        material: concrete
        tilt_deg: 0.0
"""


def _scene_yaml(tmp_path: pathlib.Path, material: str = "asphalt_dry") -> pathlib.Path:
    """The committed overcast car scene with its thermal block replaced by a small patch."""
    head, _, _ = OVERCAST.read_text().partition("  thermal:")
    out = tmp_path / "small_patch.yaml"
    out.write_text(head.rstrip() + "\n" + SMALL_PATCH.format(material=material))
    return out


@pytest.fixture
def scene(tmp_path, tophat_lwir_lut):  # type: ignore[no-untyped-def]
    return Scene.from_file(_scene_yaml(tmp_path), {"lwir": tophat_lwir_lut})


# --- the field exists, and is the surface's own -------------------------------------------------


def test_a_patched_surface_becomes_a_field_of_its_own_cells(scene) -> None:  # type: ignore[no-untyped-def]
    fld = scene.surface_fields["asphalt"]
    assert isinstance(fld, PlanarThermalField)
    assert fld.patch.n_cells == 80
    assert fld.patch is scene.patches["asphalt"]
    # The unpatched surface beside it gets no field, and the per-prim path keeps both.
    assert set(scene.surface_fields) == {"asphalt"}
    assert scene.thermal_surfaces == ("asphalt", "kerb")


def test_the_field_starts_from_the_spun_up_prim_and_not_from_the_air(scene) -> None:  # type: ignore[no-untyped-def]
    """M6.10's point, carried over: frame 0 is the state the weather implies, per cell."""
    fld = scene.surface_fields["asphalt"]
    cells = fld.temperature_at(scene.t0_s)
    prim = scene.surface_temperature_k("asphalt", scene.t0_s)
    assert np.array_equal(cells, np.full(80, prim, dtype=np.float32))
    assert abs(prim - scene.weather.at(scene.t0_s).t_air_k) > 0.05, "spin-up did nothing"


def test_uniform_forcing_is_a_pure_redistribution(scene) -> None:  # type: ignore[no-untyped-def]
    """The identity this step is checked by: nothing varies, so nothing may differ."""
    fld = scene.surface_fields["asphalt"]
    t = scene.t0_s + 1500.0
    scene.thermal.advance_to(t)
    fld.advance_to(t)
    cells = fld.temperature_at(t)
    prim = scene.surface_temperature_k("asphalt", t)
    # The contract is 1 mK; the implementation gives bit-identity, and the stronger claim is
    # asserted so a drift inside the tolerance is still visible.
    assert float(np.abs(cells.astype(np.float64) - prim).max()) <= 1e-3
    assert np.array_equal(cells, np.full(80, prim, dtype=np.float32))
    assert cells.dtype == np.float32


def test_a_query_never_advances_the_field(scene) -> None:  # type: ignore[no-untyped-def]
    """The rule `ThermalField` was built on, re-asserted where a renderer will ask many times."""
    fld = scene.surface_fields["asphalt"]
    fld.advance_to(scene.t0_s + 120.0)
    before = fld.state_hash()
    for _ in range(5):
        fld.sample_at(scene.t0_s + 60.0, np.array([[0.0, 0.0, 0.0], [-1.4, 0.0, -1.1]]))
        fld.temperature_at(scene.t0_s + 90.0)
    assert fld.state_hash() == before


# --- the plumbing a render driver relies on -----------------------------------------------------


def test_surface_bindings_pair_each_field_with_its_prim(scene) -> None:  # type: ignore[no-untyped-def]
    bindings = scene.surface_bindings()
    assert bindings == (("/World/Road", scene.surface_fields["asphalt"]),)


def test_a_patch_without_a_prim_is_solved_but_never_bound(tmp_path, tophat_lwir_lut) -> None:  # type: ignore[no-untyped-def]
    text = _scene_yaml(tmp_path).read_text().replace("          prim_path: /World/Road\n", "")
    path = tmp_path / "unbound.yaml"
    path.write_text(text)
    scene = Scene.from_file(path, {"lwir": tophat_lwir_lut})
    assert "asphalt" in scene.surface_fields
    assert scene.surface_bindings() == ()


def test_the_field_is_a_weather_consumer_the_one_weather_guard_sees(scene) -> None:  # type: ignore[no-untyped-def]
    """CLAUDE.md #6, enforced at the object rather than by review."""
    consumer = scene.consumers["field:asphalt"]
    assert consumer.weather is scene.weather


def test_a_scene_without_patches_has_no_fields(tophat_lwir_lut) -> None:  # type: ignore[no-untyped-def]
    """Every scene before v7, and every v7 scene that does not want a field: bit-identical path."""
    scene = Scene.from_file(FACETS, {"lwir": tophat_lwir_lut})
    assert scene.surface_fields == {}
    assert scene.surface_bindings() == ()
    assert not any(k.startswith("field:") for k in scene.consumers)


def test_an_unknown_material_fails_at_load_naming_the_surface(tmp_path, tophat_lwir_lut) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ValueError, match=r"'asphalt'.*'unobtainium'"):
        Scene.from_file(_scene_yaml(tmp_path, material="unobtainium"), {"lwir": tophat_lwir_lut})


# --- the committed scenes actually use it ------------------------------------------------------


@pytest.mark.parametrize("path", [OVERCAST, CLEAR], ids=["overcast", "clear"])
def test_the_committed_car_scenes_declare_bonnet_and_road(path: pathlib.Path) -> None:
    """The owner's bar: the config, not a Python driver, says where the fields are."""
    config = load_scene_config(path)
    surfaces = {s.name: s for s in config.scene.thermal.surfaces}
    assert {"asphalt", "bonnet"} <= set(surfaces)
    assert surfaces["asphalt"].patch is not None and surfaces["bonnet"].patch is not None
    assert surfaces["asphalt"].patch.prim_path == "/World/Road"
    assert surfaces["bonnet"].patch.prim_path == "/World/Car/bonnet"
    assert surfaces["bonnet"].material == "car_paint_black"
