"""PT.2 — a point-wise surface is declarable in a scene config, not only in Python.

`SurfaceSpec` carried name, material, tilt, azimuth, shaded and vehicle_speed — nothing spatial. The
only point-wise scene that existed was `irsim_isaac.car_demo`, which hand-wrote its bonnet and road
patches, and `surface_fields=` was passed at exactly one call site. So the owner's own bar for a
lane — "a scene config plus one command produces frames" — was unmeetable for any *new* point-wise
scene, which is the requirement point-wise temperature exists to serve.

Schema v7 adds a `patch:` block. The tests that carry weight are the two that would catch a
transcription error: the car demo's hand-built patches must come back from a declaration with
**bit-identical cell centres**, and a field solved on each must agree **bit-identically after
1500 s** — because a patch that is nearly right produces a plausible gradient, which is the failure
mode this whole lane is about.

docs/physics-model.md §6.1, §12.3; ADR 0087; roadmap PT.2.
"""

from __future__ import annotations

import numpy as np
import pytest

from irsim.config.scene import SCENE_SCHEMA_VERSION, PatchSpec
from irsim.scene import build_patch
from irsim.thermal.facets import FacetForcing, FacetProperties
from irsim.thermal.surface_field import PlanarThermalField
from irsim_isaac.car_demo import CameraSetup, CarGeometry, bonnet_patch, ground_patch


def _spec_from(patch, **over) -> PatchSpec:
    """The declaration a scene author would write for an existing patch."""
    fields = {
        "origin_m": tuple(float(x) for x in patch.origin_m),
        "u_axis": tuple(float(x) for x in patch.u_axis),
        "v_axis": tuple(float(x) for x in patch.v_axis),
        "n_u": patch.n_u,
        "n_v": patch.n_v,
        "du_m": patch.du_m,
        "dv_m": patch.dv_m,
        "thickness_m": patch.thickness_m,
        "frame": patch.frame,
    }
    fields.update(over)
    return PatchSpec(**fields)


def _field(patch) -> PlanarThermalField:
    """The same solver on either patch: only the grid differs, so only the grid can be blamed."""
    n = patch.n_cells
    props = FacetProperties(
        heat_capacity_j_m2_k=np.full(n, 60_000.0),
        emissivity=np.full(n, 0.95),
        solar_absorptivity=np.full(n, 0.88),
    )
    # A ramp of internal flux across the cells, so the answer depends on *which* cell is where.
    q = np.linspace(0.0, 400.0, n)

    def forcing(_t: float) -> FacetForcing:
        return FacetForcing(
            t_air_k=288.0, h_w_m2_k=8.0, q_longwave_down_w_m2=300.0, q_internal_w_m2=q
        )

    return PlanarThermalField(patch, props, forcing, 0.0, np.full(n, 290.0))


# --- the acceptance: the car demo's own patches, from a declaration ---------------------------


@pytest.mark.parametrize("which", ["bonnet", "ground"])
def test_a_declared_patch_reproduces_the_hand_built_one(which: str) -> None:
    geom = CarGeometry()
    if which == "bonnet":
        built = bonnet_patch(geom)
    else:
        x0, x1, z0, z1 = CameraSetup().ground_footprint_m(24.8, 30.7)
        built = ground_patch((x0 - 4.0, x1 + 4.0, z0 - 4.0, z1 + 4.0))

    declared = build_patch(_spec_from(built))

    assert declared.shape == built.shape
    assert declared.n_cells == built.n_cells
    # Bit-identical, not close: a declaration is a transcription, and float64 arithmetic on the
    # same inputs is deterministic. `approx` would hide a typo in a cell size's last digit.
    assert np.array_equal(declared.cell_centres(), built.cell_centres())
    assert declared.thickness_m == built.thickness_m
    assert declared.frame == built.frame


@pytest.mark.parametrize("which", ["bonnet", "ground"])
def test_the_field_on_a_declared_patch_agrees_after_1500_seconds(which: str) -> None:
    """Grid equality is the premise; this is the consequence anyone would actually notice."""
    geom = CarGeometry()
    if which == "bonnet":
        built = bonnet_patch(geom)
    else:
        x0, x1, z0, z1 = CameraSetup().ground_footprint_m(24.8, 30.7)
        built = ground_patch((x0 - 4.0, x1 + 4.0, z0 - 4.0, z1 + 4.0))
    declared = build_patch(_spec_from(built))

    a, b = _field(built), _field(declared)
    a.advance_to(1500.0)
    b.advance_to(1500.0)
    got_a, got_b = a.temperature_at(1500.0), b.temperature_at(1500.0)
    assert np.array_equal(got_a, got_b)
    # And the run actually did something, so equality is not two copies of the initial state.
    assert float(np.ptp(got_a)) > 1.0


# --- what a bad declaration does ------------------------------------------------------------


def test_axes_that_are_not_perpendicular_are_refused() -> None:
    """At load, not at render: a scene that cannot be solved should not reach a Kit session."""
    with pytest.raises(ValueError, match="perpendicular"):
        PatchSpec(
            origin_m=(0.0, 0.0, 0.0),
            u_axis=(1.0, 0.0, 0.0),
            v_axis=(1.0, 0.0, 1.0),
            n_u=2,
            n_v=2,
            du_m=0.1,
            dv_m=0.1,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [("thickness_m", 0.0), ("du_m", 0.0), ("dv_m", -0.1), ("n_u", 0), ("n_v", 0)],
)
def test_a_degenerate_patch_is_refused(field: str, value: float) -> None:
    kwargs = {
        "origin_m": (0.0, 0.0, 0.0),
        "u_axis": (1.0, 0.0, 0.0),
        "v_axis": (0.0, 0.0, 1.0),
        "n_u": 2,
        "n_v": 2,
        "du_m": 0.1,
        "dv_m": 0.1,
        field: value,
    }
    with pytest.raises(ValueError):
        PatchSpec(**kwargs)


def test_a_zero_length_axis_is_refused() -> None:
    with pytest.raises(ValueError, match="zero length"):
        PatchSpec(
            origin_m=(0.0, 0.0, 0.0),
            u_axis=(0.0, 0.0, 0.0),
            v_axis=(0.0, 0.0, 1.0),
            n_u=2,
            n_v=2,
            du_m=0.1,
            dv_m=0.1,
        )


# --- the schema stays backwards compatible ------------------------------------------------------


def test_the_block_is_optional_and_the_version_moved() -> None:
    """Every v4-v6 scene must load and solve unchanged; `patch:` is additive.

    v8 (PT.18) added `world_frame:` and `occluders:` the same way -- both optional, both
    defaulting to what a v7 scene meant by omission, v9 the network, v10 the cabin and a
    layered surface's deep boundary, v11 the sun's disc, v12 standing water and v13 a
    surface's speed schedule.
    """
    from irsim.config.scene import SurfaceSpec

    assert SCENE_SCHEMA_VERSION == 13
    plain = SurfaceSpec(name="asphalt", material="asphalt_dry")
    assert plain.patch is None


def test_a_patch_carries_its_frame_and_prim_binding() -> None:
    """`frame` is a name the bridge checks; `prim_path` is what reaches a pixel."""
    spec = PatchSpec(
        origin_m=(0.0, 0.0, 0.0),
        u_axis=(1.0, 0.0, 0.0),
        v_axis=(0.0, 0.0, 1.0),
        n_u=2,
        n_v=2,
        du_m=0.1,
        dv_m=0.1,
        frame="world",
        prim_path="/World/Car/bonnet",
    )
    assert spec.prim_path == "/World/Car/bonnet"
    # The grid itself knows nothing about geometry binding.
    assert not hasattr(build_patch(spec), "prim_path")
    assert build_patch(spec).frame == "world"


# --- the whole point: it comes out of a YAML file ------------------------------------------------


PATCH_YAML = """
  thermal:
    spin_up_hours: 48.0
    tick_s: 60.0
    surfaces:
      - name: asphalt
        material: asphalt_dry
        tilt_deg: 0.0
        patch:
          origin_m: [-15.0, 0.0, -15.0]
          u_axis: [1.0, 0.0, 0.0]
          v_axis: [0.0, 0.0, 1.0]
          n_u: 100
          n_v: 104
          du_m: 0.3
          dv_m: 0.3
          thickness_m: 0.10
          frame: world
          prim_path: /World/Road
"""


def _scene_yaml_with_a_patch(tmp_path):
    import pathlib

    source = (
        pathlib.Path(__file__).resolve().parents[2]
        / "configs/scenes/car_ignition_overcast_night.yaml"
    )
    text = source.read_text()
    head, _, _ = text.partition("  thermal:")
    out = tmp_path / "patched_scene.yaml"
    out.write_text(head.rstrip() + "\n" + PATCH_YAML)
    return out


def test_a_patch_block_round_trips_through_yaml(tmp_path) -> None:
    """The claim in one assertion: a scene *file* can declare a point-wise surface."""
    from irsim.config.scene import load_scene_config

    config = load_scene_config(_scene_yaml_with_a_patch(tmp_path))
    assert config.schema_version == SCENE_SCHEMA_VERSION
    surface = config.scene.thermal.surfaces[0]
    assert surface.patch is not None
    assert surface.patch.n_u == 100 and surface.patch.n_v == 104
    assert surface.patch.prim_path == "/World/Road"

    patch = build_patch(surface.patch)
    assert patch.n_cells == 10_400
    assert patch.extent_m == pytest.approx((30.0, 31.2))


def test_a_scene_exposes_its_patches_by_surface_name(tmp_path, tophat_lwir_lut) -> None:
    """`Scene.patches` is what a render driver reads instead of hand-building a grid."""
    from irsim.scene import Scene

    scene = Scene.from_file(_scene_yaml_with_a_patch(tmp_path), {"lwir": tophat_lwir_lut})
    assert set(scene.patches) == {"asphalt"}
    assert scene.patches["asphalt"].n_cells == 10_400
    assert scene.patch_prims == {"asphalt": "/World/Road"}
    # The surface is still a solver node as well, so the flat path is unaffected.
    assert "asphalt" in scene.thermal_surfaces


def test_a_scene_without_patches_exposes_none(tophat_lwir_lut) -> None:
    """Every scene before v7, and every v7 scene that does not want a field.

    The car scenes stopped being that example in PT.17 -- they now declare their bonnet and road
    -- so the v4 facet scene stands in.
    """
    import pathlib

    from irsim.scene import Scene

    repo = pathlib.Path(__file__).resolve().parents[2]
    scene = Scene.from_file(
        repo / "configs/scenes/thermal_facet_scene.yaml", {"lwir": tophat_lwir_lut}
    )
    assert scene.patches == {}
    assert scene.patch_prims == {}
