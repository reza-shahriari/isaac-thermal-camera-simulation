"""TC.13 -- the Phantom 4 on the solved network (`phantom4_solved.yaml`, ADR 0166).

`phantom4_perpart.yaml` gave the aircraft's parts their temperatures by prescription: ADR 0072's
instantaneous `T_air + dT_max u^2` node for a motor, ADR 0143's `T_air + offset` for a shell. This
scene hands the object to TC.11's full solve -- nineteen coarse mesh parts (ADR 0165), ten hidden
components, nineteen contacts, one implicit step -- and these tests pin what only a solve can say:

* the motor bells **lag** their throttle by their own time constant (a tenth of the rise in the
  first minute of a five-minute climb) and are still tens of kelvin over the air ten minutes
  after landing, where the prescribed node was 0.1 K over air the instant the throttle dropped;
* same-kind parts differ the way the flight makes them differ, through the solve: the rear pair
  leads in nose-down cruise, the outer pair in each orbit, and every winding sits above its bell;
* the pack warms the lower shell **from inside**, over its own footprint;
* energy closes: the real object, made adiabatic, stores exactly what its components dissipate.

The archive is generated (`scripts/prep_asset.py`), not in git, so the scene tests skip without
it; the schedule schema is exercised on its own.
"""

from __future__ import annotations

import importlib.util
import pathlib

import numpy as np
import pytest

from irsim.config.scene import ObjectSpec, ScheduleSpec, load_scene_config
from irsim.scene import Scene

REPO = pathlib.Path(__file__).resolve().parents[2]
SCENE = REPO / "configs" / "scenes" / "phantom4_solved.yaml"
ARCHIVE = REPO / "data" / "assets" / "phantom4_parts"
STATIONS = ("front_left", "front_right", "rear_left", "rear_right")
PACK_XY = np.array([-0.0078, 0.4915])

needs_archive = pytest.mark.skipif(
    not ARCHIVE.exists(), reason="the Phantom 4 archive is generated, not in git"
)


@pytest.fixture(scope="module")
def scene(tophat_lwir_lut):  # type: ignore[no-untyped-def]
    if not ARCHIVE.exists():
        pytest.skip("the Phantom 4 archive is generated, not in git")
    return Scene.from_config(load_scene_config(SCENE), {"lwir": tophat_lwir_lut})


def _bell_c(scene: Scene, station: str, t: float) -> float:
    field = scene.mesh_fields[f"phantom4.motor_{station}"]
    cells = np.asarray(field.temperature_at(t), dtype=np.float64)
    return float(field.patch.area_weighted_mean(cells)) - 273.15


def _flown(scene: Scene, rel_s: float) -> float:
    t = scene.t0_s + rel_s
    scene.objects["phantom4"].coupled.advance_to(t)
    return t


# --- the schedules ---------------------------------------------------------------------------


def test_a_schedule_is_a_schedule_and_an_object_names_only_parts_it_has() -> None:
    ScheduleSpec(at_s=[0.0, 10.0], values=[0.0, 1.0])
    with pytest.raises(ValueError, match="increase"):
        ScheduleSpec(at_s=[10.0, 0.0], values=[0.0, 1.0])
    with pytest.raises(ValueError, match="length"):
        ScheduleSpec(at_s=[0.0], values=[0.0, 1.0])
    with pytest.raises(ValueError, match="negative"):
        ScheduleSpec(at_s=[0.0], values=[-1.0])
    spec = ObjectSpec(
        name="q",
        asset="phantom4_parts",
        duties={"pack": ScheduleSpec(at_s=[0.0], values=[0.5])},
        speeds={"arms": ScheduleSpec(at_s=[0.0, 1.0], values=[2.0, 3.0])},
    )
    assert spec.duties["pack"].values == [0.5] and spec.speeds["arms"].values[-1] == 3.0
    cfg = load_scene_config(SCENE)
    obj = cfg.scene.thermal.objects[0]
    assert (
        obj.coarse
        and obj.cell_m == 0.02
        and set(obj.duties)
        == {
            *(f"winding_{s}" for s in STATIONS),
            *(f"esc_{s}" for s in STATIONS),
            "pack",
        }
    )
    assert all(f"propeller_{s}" in obj.speeds and f"motor_{s}" in obj.speeds for s in STATIONS)
    # a station's duty is 0.694 u^3 of the perpart scene's throttle: hover 0.6 -> 0.15 of rated
    hover = obj.duties["winding_front_left"]
    assert hover.values[hover.at_s.index(120.0)] == pytest.approx(0.694 * 0.55**3, abs=1e-3)


@needs_archive
def test_an_unknown_duty_or_speed_is_refused_by_name(tophat_lwir_lut) -> None:  # type: ignore[no-untyped-def]
    import yaml

    doc = yaml.safe_load(SCENE.read_text())
    obj = doc["scene"]["thermal"]["objects"][0]
    obj["duties"]["turbo"] = {"at_s": [0.0], "values": [1.0]}
    from irsim.config.scene import SceneConfig

    with pytest.raises(ValueError, match="duties name hidden parts the asset lacks: \\['turbo'\\]"):
        Scene.from_config(SceneConfig.model_validate(doc), {"lwir": tophat_lwir_lut})


# --- the solve --------------------------------------------------------------------------------


@pytest.mark.slow
@needs_archive
def test_the_object_is_solved_whole_on_coarse_cells(scene: Scene) -> None:
    from irsim.thermal.mesh_coarse import CoarseMeshPatch

    obj = scene.objects["phantom4"]
    assert len(obj.part_surfaces) == 19 and set(scene.targets) == set()
    assert set(obj.solve.hidden) == {
        *(f"winding_{s}" for s in STATIONS),
        "pack",
        *(f"esc_{s}" for s in STATIONS),
        "flight_controller",
    }
    n = obj.coupled.field.properties.n_facets
    assert 4000 < n < 7000, n  # 1.53 M faces, ~5,100 cells at 2 cm, plus ten nodes
    for name in obj.part_surfaces:
        assert isinstance(scene.mesh_fields[name].patch, CoarseMeshPatch)
    assert len(obj.coupled.contactor_conductances) >= 8, "motor-mount and mount-arm contacts"


@pytest.mark.slow
@needs_archive
def test_the_bells_lag_their_throttle_and_stay_hot_after_landing(scene: Scene) -> None:
    """Spool-up is complete by T+60 (u = 0.5); the bells have then risen a tenth of what they
    will by T+300 -- a time constant, not a jump. Landed and idle from T+1680 they are still
    15 K over the air ten minutes later, where ADR 0072's node read 0.1 K over air at once."""
    t0 = scene.t0_s
    air = float(scene.weather.at(t0).t_air_k) - 273.15
    rest = [_bell_c(scene, s, _flown(scene, 0.0)) for s in STATIONS]
    at_60 = [_bell_c(scene, s, _flown(scene, 60.0)) for s in STATIONS]
    at_300 = [_bell_c(scene, s, _flown(scene, 300.0)) for s in STATIONS]
    rise_60 = np.mean(at_60) - np.mean(rest)
    rise_300 = np.mean(at_300) - np.mean(rest)
    assert rise_300 > 8.0, rise_300
    assert rise_60 < 0.2 * rise_300, (rise_60, rise_300)
    _flown(scene, 1680.0)
    landed = [_bell_c(scene, s, _flown(scene, 1980.0)) for s in STATIONS]
    assert min(landed) - air > 12.0, (landed, air)
    later = [_bell_c(scene, s, _flown(scene, 2280.0)) for s in STATIONS]
    assert max(later) < min(landed), "and cooling"
    assert min(later) - air > 6.0


@pytest.mark.slow
@needs_archive
def test_same_kind_parts_differ_through_the_solve_and_the_winding_leads_its_bell(
    scene: Scene,
) -> None:
    obj = scene.objects["phantom4"]
    cruise = {s: _bell_c(scene, s, _flown(scene, 570.0)) for s in STATIONS}
    assert (
        min(cruise["rear_left"], cruise["rear_right"])
        - max(cruise["front_left"], cruise["front_right"])
        > 4.0
    ), cruise
    # in an orbit the outer pair carries more thrust; solved with their own inertia the two
    # motors of a pair no longer read alike, so the pairs are compared by their means
    right = {s: _bell_c(scene, s, _flown(scene, 870.0)) for s in STATIONS}
    outer = (right["front_left"] + right["rear_left"]) / 2.0
    inner = (right["front_right"] + right["rear_right"]) / 2.0
    assert outer - inner > 4.0, right
    left = {s: _bell_c(scene, s, _flown(scene, 1170.0)) for s in STATIONS}
    outer = (left["front_right"] + left["rear_right"]) / 2.0
    inner = (left["front_left"] + left["rear_left"]) / 2.0
    assert outer - inner > 4.0, left
    assert len({round(v, 1) for v in left.values()}) == 4, "four histories, four temperatures"
    t_left = scene.t0_s + 1170.0
    for s in STATIONS:
        winding = obj.node_temperature_at(f"winding_{s}", t_left) - 273.15
        assert winding > left[s] + 5.0, (s, winding, left[s])  # the source is hotter than its bell


@pytest.mark.slow
@needs_archive
def test_the_pack_warms_the_lower_shell_from_inside(scene: Scene) -> None:
    obj = scene.objects["phantom4"]
    t = _flown(scene, 1680.0)
    shell = scene.mesh_fields["phantom4.shell_lower"]
    cells = np.asarray(shell.temperature_at(t), dtype=np.float64)
    under = np.linalg.norm(shell.patch.cell_centres()[:, :2] - PACK_XY, axis=1) < 0.06
    assert under.sum() > 100
    assert cells[under].mean() - cells[~under].mean() > 1.0, "the footprint is warmer"
    case = scene.mesh_fields["phantom4.battery"]
    case_c = float(
        case.patch.area_weighted_mean(np.asarray(case.temperature_at(t), dtype=np.float64))
    )
    # the pack, its case and the shell under it are one thermal knot -- the sunlit black case
    # runs a little above the 1-2 W pack, so no ordering among the three is claimed -- and all
    # three sit above the rest of the shell
    pack_c = obj.node_temperature_k("pack") - 273.15
    knot = (pack_c, case_c - 273.15, float(cells[under].mean()) - 273.15)
    assert max(knot) - min(knot) < 1.0, knot
    assert min(knot) - (float(cells[~under].mean()) - 273.15) > 1.0, knot


@pytest.mark.slow
@needs_archive
def test_energy_closes_on_the_real_object(scene: Scene) -> None:
    """The scene's own coarse patches and properties, adiabatic, with the asset's components and
    contacts: after 600 s the object stores exactly the watts its components dissipated."""
    from irsim.config.components import load_component_library
    from irsim.config.joints import load_joint_table
    from irsim.thermal.facets import FacetForcing, FacetProperties
    from irsim.thermal.full_solve import build_full_solve
    from irsim.thermal.mesh_field import TriangleMeshField

    obj = scene.objects["phantom4"]
    shown = {p.name for p in obj.parts.parts}
    fields = {}
    for member in obj.coupled.members:
        if member.name not in shown:
            continue
        n = member.patch.n_cells
        fields[member.name] = TriangleMeshField(
            member.patch,
            FacetProperties(
                heat_capacity_j_m2_k=np.asarray(
                    member.properties.heat_capacity_j_m2_k, dtype=float
                ),
                emissivity=np.zeros(n),
                solar_absorptivity=np.zeros(n),
            ),
            lambda t: FacetForcing(t_air_k=290.0, h_w_m2_k=0.0),
            0.0,
            np.full(n, 290.0),
            30.0,
            conduction=None,
        )
    assert set(fields) == shown
    components = load_component_library()
    solve = build_full_solve(
        "phantom4",
        obj.parts,
        fields,
        load_joint_table(),
        t0_s=0.0,
        tick_s=30.0,
        duty={"*": lambda t: 0.5},
        gap_m=0.03,
        components=components,
    )
    e0 = solve.coupled.stored_energy_j()
    solve.coupled.advance_to(600.0)
    watts = sum(h.resolved_dissipation_w(0.5, components) for h in obj.parts.hidden_parts)
    assert watts > 20.0
    assert solve.coupled.stored_energy_j() - e0 == pytest.approx(watts * 600.0, rel=1e-6)


# --- the driver ---------------------------------------------------------------------------------


def test_the_driver_maps_every_part_to_the_object_s_own_surface() -> None:
    path = REPO / "scripts" / "render_phantom4.py"
    spec = importlib.util.spec_from_file_location("render_phantom4", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    defined = {"phantom4.motor_front_left", "phantom4.arms", "airframe"}
    assert (
        module.target_for_part("motor_front_left", defined, objects=("phantom4",))
        == "phantom4.motor_front_left"
    )
    assert (
        module.target_for_part("propeller_rear_left", defined, objects=("phantom4",)) == "airframe"
    )
    assert module.target_for_part("motor_front_left", {"motor"}) == "motor"
