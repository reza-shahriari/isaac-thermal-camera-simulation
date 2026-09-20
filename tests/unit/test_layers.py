"""PT.12 — an N-layer stack through the thickness of every cell.

A cell was one node with an adiabatic back. Here it becomes §6.4's two-node model generalised to
N: the two-layer stack reproduces `LumpedTwoNodeSolver` to 1 mK, a 1 mm steel skin and 0.3 m of
asphalt in one scene each show their own time constant ordered by areal capacity, and a
single-layer 0.3 m asphalt gets the night curve wrong by more than 2 K because its whole slab
holds the surface up. The scene declares it as ``layers: N`` on a patched surface.

docs/physics-model.md §6.4; ADR 0036, ADR 0094, ADR 0099, ADR 0103; roadmap PT.12.
"""

from __future__ import annotations

import math
import pathlib

import numpy as np
import pytest

from irsim.scene import Scene
from irsim.thermal.balance import SurfaceForcing, ThermalProperties
from irsim.thermal.coupling import PatchView
from irsim.thermal.facets import FacetForcing
from irsim.thermal.layers import LayerStack, layered_field
from irsim.thermal.surface_field import PlanarPatch
from irsim.thermal.two_node import LumpedTwoNodeSolver, NodeLayer, TwoNodeProperties, TwoNodeState

REPO = pathlib.Path(__file__).resolve().parents[2]
EX = np.array([1.0, 0.0, 0.0])
EY = np.array([0.0, 1.0, 0.0])

CONCRETE = dict(conductivity_w_mk=1.4, density_kg_m3=2300.0, specific_heat_j_kgk=880.0)
ASPHALT = dict(conductivity_w_mk=0.75, density_kg_m3=2200.0, specific_heat_j_kgk=920.0)
STEEL = dict(conductivity_w_mk=45.0, density_kg_m3=7800.0, specific_heat_j_kgk=470.0)


def _patch(n: int = 2) -> PlanarPatch:
    return PlanarPatch(
        origin_m=np.zeros(3),
        u_axis=EX,
        v_axis=EY,
        n_u=n,
        n_v=1,
        du_m=0.5,
        dv_m=0.5,
        thickness_m=0.3,
    )


def _diurnal(t_s: float) -> FacetForcing:
    hour = (t_s / 3600.0) % 24.0
    sun = max(0.0, 800.0 * math.sin(math.pi * (hour - 6.0) / 12.0))
    return FacetForcing(
        t_air_k=290.0 + 6.0 * math.sin(math.pi * (hour - 9.0) / 12.0),
        h_w_m2_k=12.0,
        q_solar_w_m2=sun,
        q_longwave_down_w_m2=330.0,
    )


# --- N = 2 is the two-node model -----------------------------------------------------------


def test_two_layers_reproduce_the_lumped_two_node_solver_to_a_millikelvin() -> None:
    """The same equations (§6.4's R₁₂ and R₂d) on two integrators, RK2 and IMEX, at a tick well
    inside both their accuracy: 6 h of diurnal forcing agree to 1 mK on both nodes."""
    surface = NodeLayer(0.05, **CONCRETE)
    substrate = NodeLayer(0.25, **CONCRETE)
    optical = ThermalProperties(surface.heat_capacity_j_m2_k, 0.92, 0.65)
    two_node = TwoNodeProperties(
        surface, substrate, optical, back_resistance_m2k_w=0.5, deep_temperature_k=288.0
    )
    stack = LayerStack((surface, substrate), back_resistance_m2k_w=0.5, deep_temperature_k=288.0)
    dt = 2.0  # first order in dt on the IMEX side: 10 s gave 1.7 mK, 2 s is inside 1 mK
    reference = LumpedTwoNodeSolver(two_node, dt, 12.0, TwoNodeState(300.0, 295.0))
    coupled = layered_field(
        "slab",
        _patch(1),
        optical,
        stack,
        _diurnal,
        0.0,
        np.array([300.0]),
        tick_s=dt,
        lateral=False,
    )
    base = coupled.fields["slab:layer1"]
    assert isinstance(base, PatchView)
    # The base starts at 295 K as the reference does: written into the solver's own state, a
    # test-only reach into the field for a like-for-like start.
    coupled.field._solver._state[1] = 295.0
    coupled.field._ticks[-1].temperatures_k[1] = 295.0
    for k in range(int(6 * 3600 / dt)):
        t = k * dt
        f = _diurnal(t)
        reference.advance(
            SurfaceForcing(
                t_air_k=float(np.asarray(f.t_air_k)),
                h_w_m2_k=float(np.asarray(f.h_w_m2_k)),
                q_solar_w_m2=float(np.asarray(f.q_solar_w_m2)),
                q_longwave_down_w_m2=float(np.asarray(f.q_longwave_down_w_m2)),
            )
        )
    coupled.advance_to(6 * 3600.0)
    state = coupled.field.latest_state_k
    assert abs(float(state[0]) - reference.state.surface_k) < 1e-3, (state[0], reference.state)
    assert abs(float(state[1]) - reference.state.substrate_k) < 1e-3, (state[1], reference.state)


def test_the_stack_s_contact_conductances_are_6_4_s_centre_to_centre_form() -> None:
    stack = LayerStack.uniform(3, **ASPHALT, thickness_m=0.3)
    assert stack.n_layers == 3 and stack.thickness_m == pytest.approx(0.3)
    dz = 0.1
    expected = 1.0 / (dz / (2 * 0.75) + dz / (2 * 0.75))  # k / dz
    assert stack.contact_conductances_w_m2_k() == pytest.approx((expected, expected))
    coupled = layered_field(
        "s", _patch(), ThermalProperties(1.0, 0.9, 0.9), stack, _diurnal, 0.0, 300.0
    )
    k = coupled.contactor_conductances[("s", "s:layer1")]
    assert float(k.sum()) == pytest.approx(expected * 2 * 0.25)  # two cells of 0.25 m²
    with pytest.raises(ValueError, match="at least one layer"):
        LayerStack(())
    with pytest.raises(ValueError, match="deep_temperature_k"):
        LayerStack((NodeLayer(0.1, **ASPHALT),), back_resistance_m2k_w=1.0)


# --- time constants, ordered by capacity ---------------------------------------------------


def test_a_steel_skin_and_thick_asphalt_each_show_their_own_time_constant() -> None:
    """Under one diurnal forcing the 1 mm skin peaks within minutes of the air, the asphalt
    surface an hour or more later: the lag orders with areal capacity."""
    dt = 60.0
    skin = layered_field(
        "skin",
        _patch(1),
        ThermalProperties(1.0, 0.85, 0.9),
        LayerStack.uniform(1, **STEEL, thickness_m=0.001),
        _diurnal,
        0.0,
        295.0,
        tick_s=dt,
        lateral=False,
    )
    road = layered_field(
        "road",
        _patch(1),
        ThermalProperties(1.0, 0.94, 0.9),
        LayerStack.uniform(6, **ASPHALT, thickness_m=0.3),
        _diurnal,
        0.0,
        295.0,
        tick_s=dt,
        lateral=False,
    )
    for c in (skin, road):
        c.advance_to(48 * 3600.0)  # settle into the cycle
    peaks = {}
    for name, c in (("skin", skin), ("road", road)):
        best, when = -1.0, 0.0
        for k in range(24 * 60):
            t = 48 * 3600.0 + 60.0 * k
            c.advance_to(t)
            temp = float(c.fields[name].temperature_at(t)[0])
            if temp > best:
                best, when = temp, (t / 3600.0) % 24.0
        peaks[name] = when
    # Measured: the skin peaks at 12:19, the road surface at 13:45.
    assert 12.0 <= peaks["skin"] <= 13.0, peaks  # the beam peaks at 12:00; a skin follows it
    assert peaks["road"] - peaks["skin"] > 1.0, peaks  # the road's surface lags by an hour+


def test_a_single_node_asphalt_slab_gets_the_night_wrong_by_more_than_two_kelvin() -> None:
    """Lumped, 0.3 m of asphalt holds its surface up all night; layered, the surface cools while
    the base keeps the day. The two differ by > 2 K at 04:00."""
    optical = ThermalProperties(1.0, 0.94, 0.9)
    lumped = layered_field(
        "a",
        _patch(1),
        optical,
        LayerStack.uniform(1, **ASPHALT, thickness_m=0.3),
        _diurnal,
        0.0,
        295.0,
        60.0,
        lateral=False,
    )
    layered = layered_field(
        "a",
        _patch(1),
        optical,
        LayerStack.uniform(6, **ASPHALT, thickness_m=0.3),
        _diurnal,
        0.0,
        295.0,
        60.0,
        lateral=False,
    )
    t_night = 48 * 3600.0 + 4 * 3600.0
    lumped.advance_to(t_night)
    layered.advance_to(t_night)
    one = float(lumped.fields["a"].temperature_at(t_night)[0])
    six = float(layered.fields["a"].temperature_at(t_night)[0])
    # Measured: lumped 292.3 K, layered 286.6 K -- 5.6 K apart at 04:00.
    assert one - six > 2.0, (one, six)
    # Energy check on the adiabatic stack: the layers hold what the surface took in, so the
    # deepest layer is warmer than the surface at night.
    deep = float(layered.fields["a:layer5"].temperature_at(t_night)[0])
    assert deep > six


# --- the scene -------------------------------------------------------------------------------


@pytest.mark.slow
def test_the_scene_declares_layers_and_one_layer_is_the_field_it_always_was(tmp_path) -> None:  # type: ignore[no-untyped-def]
    text = (REPO / "configs/scenes/wet_road_noon.yaml").read_text()
    head, _, _ = text.partition("        film:")
    plain = tmp_path / "one.yaml"
    plain.write_text(head)
    six = tmp_path / "six.yaml"
    six.write_text(
        head.replace(
            "        material: asphalt_dry\n", "        material: asphalt_dry\n        layers: 6\n"
        )
    )
    a = Scene.from_file(plain)
    b = Scene.from_file(six)
    view = b.surface_fields["road"]
    assert isinstance(view, PatchView) and view.patch.n_cells == 24 * 24
    assert view.field.properties.n_facets == 6 * 24 * 24
    assert not isinstance(a.surface_fields["road"], PatchView)
    # Both bind to the same prim and answer the bridge's interface.
    assert dict(b.surface_bindings())["/World/Road"] is view
    t = b.t0_s + 3600.0
    view.advance_to(t)
    a.surface_fields["road"].advance_to(t)
    assert view.sample_at(t, view.patch.cell_centres()).shape == (24 * 24,)
    # The layered road's surface at 15:00 is cooler than the lumped 5 cm slab's: the heat goes down.
    assert float(view.temperature_at(t).mean()) != float(
        a.surface_fields["road"].temperature_at(t).mean()
    )
    from irsim.config.scene import load_scene_config

    bad = tmp_path / "film.yaml"
    bad.write_text(
        text.replace(
            "        material: asphalt_dry\n", "        material: asphalt_dry\n        layers: 3\n"
        )
    )
    with pytest.raises(ValueError, match="layered surface"):
        load_scene_config(bad)
