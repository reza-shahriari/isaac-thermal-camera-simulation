"""PT.15 — the cabin and the two-node substrate, reachable from a scene.

`LumpedTwoNodeSolver` and `CabinNode` shipped in M6 and neither could be reached from a config:
every solved surface in every scene had an adiabatic back. Here the cabin becomes a lumped
member of its panels' coupled solve -- one implicit operator over panels *and* air, not an
alternating step -- and a layered surface gets §6.4's R₂d and T_deep from `back:`.

The bar is ADR 0038's own measurement, through the new path: the coupled member reproduces
`CabinNode`'s equilibrium to 0.1 K on ADR 0038's panels, where the roof stands 4.8 K above an
adiabatic back. A scene then shows the same mechanism on a schematic saloon (+1.6 K there,
because its only sun-facing panel is the roof), a cabin at the 60-80 C a sealed car reaches,
and roof and cabin both more than 2 K below the air on a clear night.

docs/physics-model.md §6.6, §6.4; ADR 0036, ADR 0038, ADR 0103, ADR 0106; roadmap PT.15.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from irsim.config.scene import SCENE_SCHEMA_VERSION, load_scene_config
from irsim.scene import Scene
from irsim.thermal import (
    CabinNode,
    CabinPanel,
    CoupledFields,
    FieldMember,
    LumpedTwoNodeSolver,
    cabin_coupling,
    cabin_field,
)
from irsim.thermal.balance import SurfaceForcing, ThermalProperties, steady_state_temperature
from irsim.thermal.facets import FacetForcing, FacetProperties
from irsim.thermal.surface_field import PlanarPatch

REPO = pathlib.Path(__file__).resolve().parents[2]
SCENE = REPO / "configs/scenes/parked_car_cabin.yaml"

PAINT = ThermalProperties(7800 * 470 * 0.0012, 0.90, 0.94)
GLASS = ThermalProperties(2500 * 840 * 0.005, 0.88, 0.10)
NOON = SurfaceForcing(t_air_k=298.15, h_w_m2_k=12.0, q_solar_w_m2=850.0, q_longwave_down_w_m2=340.0)


def _panels() -> list[CabinPanel]:
    """ADR 0038's own panel set, so its measured numbers are what this must reproduce."""
    return [
        CabinPanel("roof", PAINT, 2.2),
        CabinPanel("bonnet", PAINT, 1.6),
        CabinPanel("doors", PAINT, 3.2, 0.5),
        CabinPanel("glazing", GLASS, 2.6, 0.17),
    ]


def _square_patch(area_m2: float) -> PlanarPatch:
    """One cell of the given area: a lumped panel written as a field of one."""
    side = float(np.sqrt(area_m2))
    return PlanarPatch(
        origin_m=np.zeros(3),
        u_axis=np.array([1.0, 0.0, 0.0]),
        v_axis=np.array([0.0, 1.0, 0.0]),
        n_u=1,
        n_v=1,
        du_m=side,
        dv_m=side,
        thickness_m=0.01,
    )


def _member(panel: CabinPanel, start_k: float) -> FieldMember:
    def forcing(_t_s: float) -> FacetForcing:
        return FacetForcing(
            t_air_k=NOON.t_air_k,
            h_w_m2_k=NOON.h_w_m2_k,
            q_solar_w_m2=NOON.q_solar_w_m2,
            q_longwave_down_w_m2=NOON.q_longwave_down_w_m2,
        )

    props = FacetProperties(
        heat_capacity_j_m2_k=np.array([panel.properties.heat_capacity_j_m2_k]),
        emissivity=np.array([panel.properties.emissivity]),
        solar_absorptivity=np.array([panel.properties.solar_absorptivity]),
    )
    return FieldMember(
        panel.name, _square_patch(panel.area_m2), props, forcing, np.array([start_k])
    )


# --- the member is the node: ADR 0038's equilibrium, on the fields' own operator -------------


@pytest.mark.slow
def test_the_coupled_cabin_reproduces_cabin_node_s_equilibrium_to_a_tenth_of_a_kelvin() -> None:
    """Two integrators on one balance: `CabinNode`'s midpoint step on lumps, and the coupled
    field's IMEX step on cells joined to a lumped member. They must agree, or the scene and the
    module would be two copies of §6.6 drifting apart.

    The gap is the splitting error of ADR 0094's scheme, first order in the tick and therefore a
    property of the tick, not of the model: 0.105 K at 10 s, 0.021 K at 2 s, 0.005 K at 0.5 s.
    At the 2 s tick used here it is a fiftieth of the row's 0.1 K.
    """
    cabin = CabinNode(_panels(), glazing_panel="glazing")
    reference = cabin.equilibrium(NOON)
    coupled = cabin_field(
        cabin,
        [_member(p, NOON.t_air_k) for p in cabin.panels],
        lambda _t: NOON.t_air_k,
        lambda _t: NOON.q_solar_w_m2,
        0.0,
        2.0,
        initial_k=NOON.t_air_k,
    )
    coupled.advance_to(8.0 * 3600.0)  # the cabin's constant is minutes; 8 h is the steady state
    assert abs(coupled.node_temperature_k("cabin") - reference.cabin_k) < 0.1
    for i, panel in enumerate(cabin.panels):
        got = float(coupled.fields[panel.name].temperature_at(coupled.field.latest_t_s)[0])
        assert abs(got - float(reference.panels_k[i])) < 0.1, (panel.name, got)
    # And ADR 0038's headline, through the new path: the roof stands 4.8 K above an adiabatic
    # back, where §6.6 asks for more than 2 K.
    roof = float(coupled.fields["roof"].temperature_at(coupled.field.latest_t_s)[0])
    assert roof - steady_state_temperature(PAINT, NOON) == pytest.approx(4.8, abs=0.2)


def test_the_links_are_the_cabin_s_own_conductances_and_the_member_its_own_capacity() -> None:
    cabin = CabinNode(_panels(), glazing_panel="glazing")
    member, links = cabin_coupling(cabin, lambda _t: 290.0, lambda _t: 800.0, 290.0)
    assert member.name == "cabin" and member.capacity_j_k == cabin.capacity_j_k
    assert [link.field for link in links] == [p.name for p in cabin.panels]
    assert links[0].h_w_m2_k == pytest.approx(1.0 / 0.13)
    forcing = member.forcing_at(0.0)
    assert forcing.h_w_m2_k == cabin.infiltration_w_k  # infiltration as a conductance to the air
    assert forcing.q_internal_w_m2 == pytest.approx(0.55 * 2.6 * 800.0)  # the transmitted sun
    # The conductance a panel sees is 1/R times its own area, cell by cell.
    coupled = CoupledFields(
        [_member(p, 290.0) for p in cabin.panels],
        t0_s=0.0,
        lumped=[member],
        lumped_links=links,
    )
    k = coupled.contactor_conductances[("roof", "cabin")]
    assert float(k.sum()) == pytest.approx(2.2 / 0.13)
    assert coupled.node_temperature_k("cabin") == pytest.approx(290.0)
    with pytest.raises(KeyError, match="not a lumped member"):
        coupled.node_temperature_k("roof")
    with pytest.raises(ValueError, match="must be the cabin's own"):
        cabin_field(
            cabin, [_member(cabin.panels[0], 290.0)], lambda _t: 290.0, lambda _t: 0.0, 0.0, 10.0
        )


# --- the scene ---------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def scene() -> Scene:
    return Scene.from_file(SCENE)


@pytest.mark.slow
def test_the_cabin_makes_the_roof_hotter_than_the_same_paint_with_an_adiabatic_back(scene) -> None:  # type: ignore[no-untyped-def]
    """The bonnet is the control: the same material, the same tilt, the same forcing, an engine
    bay behind it instead of the cabin. Measured +1.60 K at local noon -- smaller than ADR 0038's
    4.8 K because this saloon's only sun-facing panel is the roof, where ADR 0038's set had the
    bonnet too; the mechanism and its sign are the same, and the magnitude is what the geometry
    gives.
    """
    t = scene.t0_s
    roof = float(np.mean(scene.surface_fields["roof"].temperature_at(t)))
    bonnet = float(np.mean(scene.surface_fields["bonnet"].temperature_at(t)))
    cabin = scene.surface_fields["cabin"].node_temperature_k("cabin")
    assert roof - bonnet == pytest.approx(1.6, abs=0.4), (roof, bonnet)
    assert cabin > roof, "the air behind the panel must be the hotter of the two"
    # A sealed car in strong sun really does reach 60-80 C.
    assert 60.0 < cabin - 273.15 < 80.0, cabin - 273.15
    # The glass transmits rather than absorbs (alpha_sol 0.10), so it sits far below the paint
    # while being the thing that heats the cabin.
    glazing = float(np.mean(scene.surface_fields["glazing"].temperature_at(t)))
    assert glazing < roof - 20.0, (glazing, roof)


@pytest.mark.slow
def test_on_a_clear_night_the_roof_and_the_cabin_are_both_below_the_air(scene) -> None:  # type: ignore[no-untyped-def]
    """Local midnight, 14 h on: the roof radiates to a cold sky and drags the cabin with it.
    Measured roof −3.8 K and cabin −3.0 K against the air, where the row asks for 2 K."""
    t = scene.t0_s + 14.0 * 3600.0
    scene.surface_fields["cabin"].advance_to(t)
    air = float(scene.weather.at(t).t_air_k)
    roof = float(np.mean(scene.surface_fields["roof"].temperature_at(t)))
    cabin = scene.surface_fields["cabin"].node_temperature_k("cabin")
    assert air - roof > 2.0 and air - cabin > 2.0, (air, roof, cabin)
    assert roof < cabin  # the panel leads, the air follows


@pytest.mark.slow
def test_without_the_cabin_block_the_panels_are_the_fields_they_always_were(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """The block is additive: delete it and the roof is bit-identical to the adiabatic bonnet."""
    text = SCENE.read_text()
    head, _, rest = text.partition("    cabin:\n")
    _, _, tail = rest.partition("    surfaces:\n")
    plain = tmp_path / "no_cabin.yaml"
    plain.write_text(head + "    surfaces:\n" + tail)
    built = Scene.from_file(plain)
    assert "cabin" not in built.surface_fields
    t = built.t0_s
    roof = np.asarray(built.surface_fields["roof"].temperature_at(t))
    bonnet = np.asarray(built.surface_fields["bonnet"].temperature_at(t))
    assert float(roof[0]) == float(bonnet[0])
    assert np.array_equal(roof, np.full(roof.shape, bonnet[0]))


def test_the_schema_refuses_a_cabin_it_cannot_build(tmp_path) -> None:  # type: ignore[no-untyped-def]
    text = SCENE.read_text()
    assert load_scene_config(SCENE).schema_version == SCENE_SCHEMA_VERSION

    # A floor with no patch, so a cabin can be pointed at a per-prim surface.
    with_floor = text + "      - {name: floor, material: car_paint_black, tilt_deg: 180.0}\n"

    def _load(replacement: tuple[str, str]) -> None:
        path = tmp_path / "bad.yaml"
        path.write_text(with_floor.replace(*replacement))
        load_scene_config(path)

    with pytest.raises(ValueError, match="glazing_surface must be one of"):
        _load(("glazing_surface: glazing", "glazing_surface: windscreen"))
    with pytest.raises(ValueError, match="is not a surface"):
        _load(("{surface: roof,      inner", "{surface: sunroof,   inner"))
    with pytest.raises(ValueError, match="needs a `patch:`"):
        # A per-prim surface has no cells for the cabin to join.
        _load(
            (
                "        - {surface: roof,      inner_resistance_m2k_w: 0.13}",
                "        - {surface: floor,     inner_resistance_m2k_w: 0.13}",
            )
        )


# --- the two-node substrate, from `back:` ------------------------------------------------------


def test_a_layered_surface_takes_r2d_and_t_deep_from_the_config(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """§6.4's deep boundary, which ADR 0036 defined and no scene could declare. With it the
    stack is `LumpedTwoNodeSolver`'s system (ADR 0103 holds the two to 1 mK); without it the
    back is adiabatic, which is what every scene had."""
    text = (REPO / "configs/scenes/wet_road_noon.yaml").read_text()
    head, _, _ = text.partition("        film:")
    layered = head.replace(
        "        material: asphalt_dry\n",
        "        material: asphalt_dry\n        layers: 4\n"
        "        back: {resistance_m2k_w: 0.5, deep_temperature_k: ambient}\n",
    )
    path = tmp_path / "deep.yaml"
    path.write_text(layered)
    spec = load_scene_config(path).scene
    surface = spec.thermal.surfaces[0]
    assert surface.back is not None and surface.back.deep_temperature_k == "ambient"
    built = Scene.from_file(path)
    view = built.surface_fields["road"]
    # The deep node is the air at the scene start, and the base layer is pulled toward it: a
    # noon road with an adiabatic back keeps its heat, one over cool soil bleeds it away.
    t = built.t0_s
    adiabatic = tmp_path / "adiabatic.yaml"
    adiabatic.write_text(
        head.replace(
            "        material: asphalt_dry\n", "        material: asphalt_dry\n        layers: 4\n"
        )
    )
    other = Scene.from_file(adiabatic)
    deep_base = float(np.mean(view.field.latest_state_k[-view.patch.n_cells :]))
    shut_base = float(
        np.mean(other.surface_fields["road"].field.latest_state_k[-view.patch.n_cells :])
    )
    assert deep_base < shut_base - 1.0, (deep_base, shut_base)
    assert float(np.mean(view.temperature_at(t))) < float(
        np.mean(other.surface_fields["road"].temperature_at(t))
    )


def test_a_back_boundary_without_layers_is_refused(tmp_path) -> None:  # type: ignore[no-untyped-def]
    text = (REPO / "configs/scenes/wet_road_noon.yaml").read_text()
    head, _, _ = text.partition("        film:")
    path = tmp_path / "bad.yaml"
    path.write_text(
        head.replace(
            "        material: asphalt_dry\n",
            "        material: asphalt_dry\n"
            "        back: {resistance_m2k_w: 0.5, deep_temperature_k: 285.0}\n",
        )
    )
    with pytest.raises(ValueError, match="needs `layers: 2`"):
        load_scene_config(path)


def test_the_two_node_solver_and_the_cabin_are_importable_from_the_package() -> None:
    """The other half of the row: both objects shipped in M6 and neither was reachable."""
    import irsim.thermal as thermal

    for name in (
        "LumpedTwoNodeSolver",
        "TwoNodeProperties",
        "NodeLayer",
        "CabinNode",
        "LayerStack",
    ):
        assert name in thermal.__all__ and hasattr(thermal, name)
    assert LumpedTwoNodeSolver is thermal.LumpedTwoNodeSolver
