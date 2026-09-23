"""PT.9 (engine-free half) — the aerial scenario, regenerated point-wise.

The quadrotor mission was four numbers: an airframe pinned to the air and three powered nodes.
Here the airframe is a field. One prim carries a gradient, the deck and the belly of the same
aircraft are 29 K apart, the deck's excess over air collapses when the mission takes off and
comes back when it lands, and every cell holds its own equilibrium to a millikelvin. The
per-prim scene beside it is untouched, and reports 0.000 K of gradient, which is the number
this row exists to replace.

The curved-prim binding a fuselage needs is `WM.3`'s and the rendered frame is `IG.2`'s;
`scripts/quad_flight_pointwise.py` writes the engine-free one from a synthetic G-buffer.

docs/physics-model.md §6.1, §6.6; ADR 0072, ADR 0087, ADR 0095, ADR 0109; roadmap PT.9.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from irsim.config.scene import SCENE_SCHEMA_VERSION, SurfaceSpec, load_scene_config
from irsim.scene import Scene
from irsim.thermal.balance import SurfaceForcing, steady_state_temperature
from irsim.thermal.scene_forcing import SurfaceOrientation

REPO = pathlib.Path(__file__).resolve().parents[2]
SCENE = REPO / "configs/scenes/quad_flight_pointwise.yaml"
PER_PRIM = REPO / "configs/scenes/quad_flight_clear_noon.yaml"

PAD_S = 0.0
CRUISE_S = 1000.0
LANDED_S = 1750.0


@pytest.fixture(scope="module")
def scene() -> Scene:
    return Scene.from_file(SCENE)


def _mean_c(scene: Scene, name: str, t: float) -> float:
    field = scene.surface_fields[name]
    field.advance_to(t)
    return float(np.mean(np.asarray(field.temperature_at(t), dtype=np.float64))) - 273.15


# --- one prim, many temperatures ----------------------------------------------------------------


@pytest.mark.slow
def test_one_airframe_carries_a_gradient_where_the_per_prim_scene_carries_none(scene) -> None:  # type: ignore[no-untyped-def]
    """The deck at 55.5 C and the belly at 26.2 C are the same aircraft at the same instant: a
    quadrotor seen from above is a hot carbon plate and seen from below it is air temperature.
    Within one arm the spread is 29 K, because the deck's own shadow and the motor pods fall
    across it. The per-prim scene's airframe is one number and its gradient is exactly zero."""
    t = scene.t0_s + PAD_S
    air = float(scene.weather.at(t).t_air_k) - 273.15
    deck, belly = _mean_c(scene, "deck", t), _mean_c(scene, "belly", t)
    assert deck - belly > 25.0, (deck, belly)
    assert abs(belly - air) < 1.0, (belly, air)  # tilt 180: no sky, no sun, pinned to the air
    for arm in ("arm_n", "arm_e"):
        field = scene.surface_fields[arm]
        field.advance_to(t)
        cells = np.asarray(field.temperature_at(t), dtype=np.float64)
        assert float(cells.max() - cells.min()) > 10.0, (arm, cells.min(), cells.max())
        lit = field.field.forcing_at.cell_visibility(t)
        assert 0.5 < lit.mean() < 1.0, lit.mean()  # the pods and the deck take a quarter of it

    # The scene this replaces: one airframe node, at the air temperature, with nothing to spread.
    per_prim = Scene.from_file(PER_PRIM)
    per_prim.advance_targets(PAD_S, 0.0)
    assert per_prim.targets["airframe"].temperature() == pytest.approx(
        float(per_prim.weather.at(per_prim.t0_s + PAD_S).t_air_k), abs=1e-6
    )
    assert not per_prim.surface_fields  # no patches at all: nothing could have a gradient


@pytest.mark.slow
def test_taking_off_collapses_the_deck_s_excess_and_landing_brings_it_back(scene) -> None:  # type: ignore[no-untyped-def]
    """The mission drives the skin's convection (ADR 0109). On the pad the deck runs 29 K over
    air; in the hard climb at 13 m/s forced convection has it down to 14 K; landed, it is 30 K
    again. The per-prim `airframe` solver asserted the flying value for the whole mission."""
    excess = {}
    for rel in (PAD_S, CRUISE_S, LANDED_S):
        t = scene.t0_s + rel
        excess[rel] = _mean_c(scene, "deck", t) - (float(scene.weather.at(t).t_air_k) - 273.15)
    assert excess[PAD_S] > 25.0, excess
    assert excess[CRUISE_S] < 16.0, excess
    assert excess[LANDED_S] > 25.0, excess
    assert excess[PAD_S] - excess[CRUISE_S] > 10.0, excess


@pytest.mark.slow
def test_holding_the_speed_at_its_pad_value_is_what_keeps_the_deck_hot(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """The negative control for the schedule: freeze the speed at the pad's 1 m/s and the deck
    never cools in the climb, which is how we know the collapse above is the mission and not the
    sun moving."""
    text = SCENE.read_text()
    frozen = tmp_path / "frozen.yaml"
    frozen.write_text(
        text.replace(
            "        speed_m_s: [1.0,   1.0,   8.0,  15.0,   10.0,    1.0,    1.0]",
            "        speed_m_s: [1.0,   1.0,   1.0,   1.0,    1.0,    1.0,    1.0]",
        )
    )
    still = Scene.from_file(frozen)
    t = still.t0_s + CRUISE_S
    air = float(still.weather.at(t).t_air_k) - 273.15
    assert _mean_c(still, "deck", t) - air > 25.0


# --- the oracle: every cell holds its own balance ------------------------------------------------


@pytest.mark.slow
def test_every_cell_holds_its_own_equilibrium_to_a_millikelvin(scene) -> None:  # type: ignore[no-untyped-def]
    """Freeze the forcing at the scene start and let the arm settle: each cell converges to the
    root `steady_state_temperature` gives for its **own** flux, to under 1 mK. A field that
    averaged or broadcast the forcing across the arm would miss by the 29 K the shadow spans."""
    from irsim.thermal.facets import FacetProperties
    from irsim.thermal.surface_field import PlanarThermalField

    t = scene.t0_s + PAD_S
    patch = scene.patches["arm_n"]
    properties = scene.surface_properties("arm_n")
    frozen = scene.surface_fields["arm_n"].field.forcing_at(t)
    t_air, h, q_solar, q_lw, q_int = frozen.arrays(patch.n_cells)
    cells = FacetProperties(
        heat_capacity_j_m2_k=np.full(patch.n_cells, properties.heat_capacity_j_m2_k),
        emissivity=np.full(patch.n_cells, properties.emissivity),
        solar_absorptivity=np.full(patch.n_cells, properties.solar_absorptivity),
    )
    field = PlanarThermalField(
        patch, cells, lambda _t: frozen, 0.0, np.full(patch.n_cells, 290.0), 1.0, keep_ticks=None
    )
    field.advance_to(4.0 * 3600.0)  # a 2 mm skin's constant is minutes; four hours is the root
    settled = np.asarray(field.temperature_at(4.0 * 3600.0), dtype=np.float64)
    expected = np.array(
        [
            steady_state_temperature(
                properties,
                SurfaceForcing(
                    t_air_k=float(t_air[i]),
                    h_w_m2_k=float(h[i]),
                    q_solar_w_m2=float(q_solar[i]),
                    q_longwave_down_w_m2=float(q_lw[i]),
                    q_internal_w_m2=float(q_int[i]),
                ),
            )
            for i in range(patch.n_cells)
        ]
    )
    assert float(np.max(np.abs(settled - expected))) < 1e-3, float(
        np.max(np.abs(settled - expected))
    )
    assert float(expected.max() - expected.min()) > 10.0  # the flux really does vary that much


# --- the schedule ---------------------------------------------------------------------------------


def test_a_speed_schedule_interpolates_and_holds_flat_outside_itself() -> None:
    orientation = SurfaceOrientation(speed_schedule=((0.0, 100.0, 200.0), (0.0, 0.0, 15.0)))
    assert [orientation.speed_at(t) for t in (-50.0, 0.0, 100.0, 150.0, 200.0, 5000.0)] == [
        0.0,
        0.0,
        0.0,
        7.5,
        15.0,
        15.0,
    ]
    assert SurfaceOrientation(vehicle_speed_m_s=5.0).speed_at(123.0) == 5.0
    assert SurfaceOrientation().speed_at(123.0) == 0.0
    with pytest.raises(ValueError, match="strictly increasing"):
        SurfaceOrientation(speed_schedule=((10.0, 0.0), (1.0, 2.0)))
    with pytest.raises(ValueError, match="cannot be negative"):
        SurfaceOrientation(speed_schedule=((0.0, 1.0), (1.0, -2.0)))
    with pytest.raises(ValueError, match="one speed authority"):
        SurfaceOrientation(vehicle_speed_m_s=3.0, speed_schedule=((0.0, 1.0), (1.0, 2.0)))


def test_the_schema_takes_a_schedule_or_a_constant_and_not_both(tmp_path) -> None:  # type: ignore[no-untyped-def]
    assert SCENE_SCHEMA_VERSION == 16
    assert load_scene_config(SCENE).schema_version == SCENE_SCHEMA_VERSION
    plain = SurfaceSpec(name="a", material="concrete")
    assert plain.speed_s is None and plain.speed_m_s is None and plain.vehicle_speed_m_s == 0.0
    with pytest.raises(ValueError, match="needs speed_s and speed_m_s"):
        SurfaceSpec(name="a", material="concrete", speed_s=[0.0, 1.0])
    with pytest.raises(ValueError, match="differ in length"):
        SurfaceSpec(name="a", material="concrete", speed_s=[0.0, 1.0], speed_m_s=[1.0])
    with pytest.raises(ValueError, match="strictly increasing"):
        SurfaceSpec(name="a", material="concrete", speed_s=[1.0, 0.0], speed_m_s=[1.0, 2.0])
    with pytest.raises(ValueError, match="one speed authority"):
        SurfaceSpec(
            name="a",
            material="concrete",
            vehicle_speed_m_s=3.0,
            speed_s=[0.0, 1.0],
            speed_m_s=[1.0, 2.0],
        )


@pytest.mark.slow
def test_the_mission_is_the_same_flight_as_the_per_prim_scene() -> None:
    """The regeneration changes how the airframe is solved, not what the pilot did: the throttle
    profile, the site, the weather file and the start time are the per-prim scene's own."""
    new = load_scene_config(SCENE).scene
    old = load_scene_config(PER_PRIM).scene
    assert (new.weather_file, new.start_utc, new.site) == (
        old.weather_file,
        old.start_utc,
        old.site,
    )
    for name in ("motor", "esc", "battery"):
        a = next(t for t in new.targets if t.name == name)
        b = next(t for t in old.targets if t.name == name)
        assert (a.solver, a.source, a.throttle_s, a.throttle) == (
            b.solver,
            b.source,
            b.throttle_s,
            b.throttle,
        )
    assert old.thermal is None or not old.thermal.surfaces
    assert [s.name for s in new.thermal.surfaces] == ["deck", "belly", "arm_n", "arm_e"]
