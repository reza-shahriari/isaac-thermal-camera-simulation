"""TC.11 -- a scene's main object, fully solved from its asset.

A synthetic object stands in for the Phantom 4 (whose archive is generated, not in git): a
carbon arm tube, an aluminium motor can standing on its end, and a hidden winding inside the
can dissipating rated watts times the object's duty, joined by AI.11's contacts. `solve: full`
turns that into one coupled solve: no offsets, no instantaneous ΔT(u). The claims: the winding
warms the can before the arm, the can lags a throttle step by its own time constant and stays
hot after the throttle stops, the arm keeps warming after the can's source is off, and energy
closes. Roadmap TC.11; ADR 0099, TC.16, AI.11.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest
import yaml

from irsim.config.joints import load_joint_table
from irsim.io.asset_parts import ContactSpec, HiddenPartSpec, PartsConfig, PartSelector, PartSpec
from irsim.io.assets import AssetMesh, write_asset_meshes
from irsim.scene import Scene
from irsim.thermal.coupling import PatchView
from irsim.thermal.facets import FacetForcing, FacetProperties
from irsim.thermal.full_solve import build_full_solve
from irsim.thermal.mesh_conduction import mesh_lateral_operator
from irsim.thermal.mesh_field import TriangleMeshField, TriangleMeshPatch
from irsim.thermal.raycast import cylinder_mesh

REPO = pathlib.Path(__file__).resolve().parents[2]
HEAD = REPO / "configs/scenes/wall_half_in_sun.yaml"

ARM = cylinder_mesh(np.array([0.0, 0.0, 0.0]), 0.012, 0.24, axis=(1.0, 0.0, 0.0), n_phi=12, n_z=8)
CAN = cylinder_mesh(
    np.array([0.14, 0.0, 0.026]), 0.014, 0.020, axis=(0.0, 0.0, 1.0), n_phi=12, n_z=2
)
WINDING_CENTRE = (0.14, 0.0, 0.026)


def _parts() -> PartsConfig:
    return PartsConfig(
        parts=[
            PartSpec(name="arm", select=PartSelector(materials=["tube_mat"])),
            PartSpec(name="motor", select=PartSelector(materials=["can_mat"])),
        ],
        contacts=[
            ContactSpec(a="motor", b="arm", joint="dry_default", area_m2=2e-4),
            ContactSpec(a="winding", b="motor", joint="dry_default", area_m2=3e-4),
        ],
        hidden_parts=[
            HiddenPartSpec(
                name="winding",
                centre_m=WINDING_CENTRE,
                size_m=(0.02, 0.02, 0.012),
                mass_kg=0.03,
                specific_heat_j_kgk=400.0,
                dissipation_w=6.0,
            )
        ],
    )


def _props(n: int, c: float) -> FacetProperties:
    return FacetProperties(
        heat_capacity_j_m2_k=np.full(n, c), emissivity=np.zeros(n), solar_absorptivity=np.zeros(n)
    )


def _mesh_fields() -> dict[str, TriangleMeshField]:
    """The separate fields a scene would have built: adiabatic, so conduction is the physics."""
    arm = TriangleMeshPatch.by_cell_size(ARM.vertices, ARM.faces, 0.01)
    can = TriangleMeshPatch.by_cell_size(CAN.vertices, CAN.faces, 0.006)
    still = lambda t: FacetForcing(t_air_k=290.0, h_w_m2_k=0.0)  # noqa: E731
    return {
        "arm": TriangleMeshField(
            arm,
            _props(arm.n_cells, 2400.0),
            still,
            0.0,
            np.full(arm.n_cells, 290.0),
            5.0,
            conduction=mesh_lateral_operator(arm, 7.0, 0.0015),
        ),
        "motor": TriangleMeshField(
            can,
            _props(can.n_cells, 2700.0),
            still,
            0.0,
            np.full(can.n_cells, 290.0),
            5.0,
            conduction=mesh_lateral_operator(can, 200.0, 0.001),
        ),
    }


def _duty(t_s: float) -> float:
    return 1.0 if t_s < 600.0 else 0.0


# --- the builder -------------------------------------------------------------------------------


def test_the_object_is_one_coupled_solve_of_its_parts_hidden_parts_and_contacts() -> None:
    solve = build_full_solve(
        "quad",
        _parts(),
        _mesh_fields(),
        load_joint_table(),
        t0_s=0.0,
        tick_s=5.0,
        duty=_duty,
        gap_m=0.02,
    )
    assert solve.shown == ("arm", "motor") and solve.hidden == ("winding",)
    assert (
        isinstance(solve.field("arm"), PatchView)
        and solve.field("arm").field is solve.field("motor").field
    )
    assert set(solve.coupled.contactor_conductances) == {("motor", "arm"), ("motor", "winding")}
    k = solve.coupled.contactor_conductances[("motor", "arm")]
    assert k.sum() == pytest.approx(1000.0 * 2e-4, rel=1e-12)
    assert solve.coupled.contactor_conductances[("motor", "winding")].sum() == pytest.approx(
        0.3, rel=1e-12
    )


def test_the_winding_warms_the_can_before_the_arm_and_the_can_lags_its_throttle() -> None:
    """6 W into a 12 J/K winding bolted to a 2.7 kJ/m2/K can: the can's cells under the
    footprint rise first, the arm's cells at the far end last; the can's temperature climbs
    through the 600 s of throttle -- lagging the step, not jumping with it -- and goes on
    rising after the throttle stops while the winding's stored heat drains into it."""
    solve = build_full_solve(
        "quad",
        _parts(),
        _mesh_fields(),
        load_joint_table(),
        t0_s=0.0,
        tick_s=5.0,
        duty=_duty,
        gap_m=0.02,
    )
    can, arm = solve.field("motor"), solve.field("arm")
    far = np.asarray(arm.patch.cell_centres())[:, 0] < -0.08
    track = []
    for t in range(0, 1201, 5):
        solve.coupled.advance_to(float(t))
        c = float(np.asarray(can.temperature_at(float(t)), dtype=np.float64).mean())
        a = np.asarray(arm.temperature_at(float(t)), dtype=np.float64)
        track.append(
            (t, c, float(a[~far].mean()), float(a[far].mean()), solve.node_temperature_k("winding"))
        )
    t_can = next(t for t, c, _, _, _ in track if c > 290.1)
    t_near = next(t for t, _, n, _, _ in track if n > 290.1)
    t_far = next(t for t, _, _, f, _ in track if f > 290.1)
    assert t_can < t_near < t_far, (t_can, t_near, t_far)
    at = {t: row for t, *row in track}
    assert at[5][0] < at[60][0] < at[300][0] < at[600][0], "the can climbs, it does not jump"
    assert at[5][0] - 290.0 < 0.2 * (at[600][0] - 290.0), "no instantaneous ΔT(u)"
    rise = at[600][0] - 290.0
    assert at[650][0] - 290.0 > 0.9 * rise, "stays hot after the throttle stops: no jump down"
    assert at[1200][0] - 290.0 > 0.5 * rise, "and still hot at twice the run"
    assert at[1200][3] < at[600][3], "the winding itself is cooling into the can"
    # energy: 6 W for 600 s, nothing left the adiabatic object
    stored = solve.coupled.stored_energy_j()
    start = 12.0 * 290.0 + sum(
        float((f.field.properties.heat_capacity_j_m2_k * f.patch.cell_area_m2 * 290.0).sum())
        for f in _mesh_fields().values()
    )
    assert stored - start == pytest.approx(6.0 * 600.0, rel=1e-6)


def _only_component(component: str) -> PartsConfig:
    return _parts().model_copy(
        update={
            "hidden_parts": [
                HiddenPartSpec(
                    name="winding",
                    centre_m=WINDING_CENTRE,
                    size_m=(0.02, 0.02, 0.012),
                    component=component,
                )
            ]
        }
    )


def test_a_hidden_part_with_only_a_component_takes_its_numbers_from_the_library() -> None:
    """AI.12: `component: brushless_motor` and nothing else is a 46 J/K node dissipating the
    library's 60 W at full duty; without a library, or naming a component it lacks, refused."""
    from irsim.config.components import load_component_library

    lib = load_component_library()
    parts = _only_component("brushless_motor")
    solve = build_full_solve(
        "quad",
        parts,
        _mesh_fields(),
        load_joint_table(),
        t0_s=0.0,
        tick_s=5.0,
        gap_m=0.02,
        components=lib,
    )
    motor = lib["brushless_motor"]
    assert solve.coupled.field.properties.heat_capacity_j_m2_k[-1] == pytest.approx(
        motor.capacity_j_k
    )
    winding = solve.coupled.members[-1]
    assert float(winding.forcing_at(0.0).q_internal_w_m2) == pytest.approx(
        motor.dissipation_rated_w
    )
    with pytest.raises(ValueError, match="no mass_kg / specific_heat_j_kgk"):
        build_full_solve(
            "quad", parts, _mesh_fields(), load_joint_table(), t0_s=0.0, tick_s=5.0, gap_m=0.02
        )
    with pytest.raises(ValueError, match="unknown component"):
        build_full_solve(
            "quad",
            _only_component("warp_core"),
            _mesh_fields(),
            load_joint_table(),
            t0_s=0.0,
            tick_s=5.0,
            gap_m=0.02,
            components=lib,
        )


# --- the scene -----------------------------------------------------------------------------------


def _asset(
    tmp_path: pathlib.Path, part_materials: dict[str, str] | None = None
) -> tuple[pathlib.Path, pathlib.Path]:
    archive = write_asset_meshes(
        tmp_path / "toy.npz",
        {
            "arm": AssetMesh("arm", "tube_mat", ARM.vertices, ARM.faces, 0.0, 0.0),
            "motor": AssetMesh("motor", "can_mat", CAN.vertices, CAN.faces, 0.0, 0.0),
        },
    )
    doc = {
        "schema_version": 1,
        "asset": {
            "name": "toy",
            "source_file": "toy.fbx",
            "materials": {"tube_mat": "carbon_fibre", "can_mat": "bare_aluminium"},
            "parts": _parts().model_dump(mode="json"),
        },
    }
    for part in doc["asset"]["parts"]["parts"]:
        if part_materials and part["name"] in part_materials:
            part["material"] = part_materials[part["name"]]
    mapping = tmp_path / "toy.yaml"
    mapping.write_text(yaml.safe_dump(doc))
    return mapping, archive


THERMAL = """
  thermal:
    spin_up_hours: 6.0
    tick_s: 5.0
{switch}
    objects:
      - name: quad
        asset: {mapping}
        archive: {archive}
        solve: full
        cell_m: 0.01
        contact_gap_m: 0.02
        duty_s: [0.0, 600.0, 601.0]
        duty: [1.0, 1.0, 0.0]
"""


def _scene(
    tmp_path: pathlib.Path, switch: str = "", part_materials: dict[str, str] | None = None
) -> Scene:
    mapping, archive = _asset(tmp_path, part_materials)
    head, _, _ = HEAD.read_text().partition("  thermal:")
    head = head.replace('start_utc: "2024-06-21T16:00:00Z"', 'start_utc: "2024-06-21T23:00:00Z"')
    out = tmp_path / "toy_scene.yaml"
    out.write_text(
        head.rstrip() + "\n" + THERMAL.format(switch=switch, mapping=mapping, archive=archive)
    )
    return Scene.from_file(out)


def test_a_part_asserted_material_outranks_the_asset_map_in_the_scene(
    tmp_path: pathlib.Path,
) -> None:
    """AI.14: the motor's source material says bare aluminium; the part says painted.

    Without the rung the can keeps eps 0.09 and renders as reflected sky -- AT.18's trap.
    """
    scene = _scene(tmp_path, part_materials={"motor": "aircraft_aluminium_painted"})
    assert scene.surface_materials["quad.motor"].name == "aircraft_aluminium_painted"
    assert scene.surface_materials["quad.arm"].name == "carbon_fibre", "unasserted: the map"


def test_a_scene_solves_its_object_whole_and_binds_its_parts(tmp_path: pathlib.Path) -> None:
    scene = _scene(tmp_path)
    assert set(scene.objects) == {"quad"}
    obj = scene.objects["quad"]
    assert obj.part_surfaces == ("quad.arm", "quad.motor")
    for name in obj.part_surfaces:
        assert isinstance(scene.mesh_fields[name], PatchView)
        assert scene.mesh_fields[name].field is obj.coupled.field
    assert scene.surface_materials["quad.arm"].name == "carbon_fibre"
    assert scene.surface_materials["quad.motor"].name == "bare_aluminium"
    t = scene.t0_s + 900.0
    scene.mesh_fields["quad.arm"].advance_to(t)
    assert scene.mesh_fields["quad.motor"].latest_t_s == t, "one solve: both parts moved"
    can = np.asarray(scene.mesh_fields["quad.motor"].temperature_at(t), dtype=np.float64).mean()
    arm = np.asarray(scene.mesh_fields["quad.arm"].temperature_at(t), dtype=np.float64).mean()
    assert can > arm + 1.0, "the can is the hot part, warmed from inside"
    assert obj.node_temperature_k("winding") > can


def test_with_the_exchange_on_the_parts_radiate_inside_the_same_solve(
    tmp_path: pathlib.Path,
) -> None:
    scene = _scene(tmp_path, switch="    object_exchange: true")
    obj = scene.objects["quad"]
    assert obj.coupled.exchange is not None and obj.coupled.exchange_members == ("arm", "motor")
    assert scene.object_exchange is None, "nothing else in the scene to exchange with"
    t = scene.t0_s + 300.0
    scene.mesh_fields["quad.arm"].advance_to(t)
    assert scene.mesh_fields["quad.motor"].latest_t_s == t
