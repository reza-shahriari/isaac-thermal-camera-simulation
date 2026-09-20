"""TC.6 — the R2 reference scene: an engine warms the metal around it, from the scene config.

Both car scenes now declare the engine as a solved node (`solver: engine`) and the metal
around it as `nodes:` and `links:`: the block followed as a boundary, rubber mounts as a link
node to the subframe, a wing bracket bolted to the block through 25 cm² of a new ferrous joint
whose fan-driven convection stops at key-off, and the wing hung off it by two small bolts.

The row's checks: a bracket lags the block by its RC time and settles at G/(G + hA) of the
block's rise, to 1e-6 (held on a fixed block, where the closed form is exact); parts warm in
conductance order; the synthetic-G-buffer bonnet shows 10–40 K max–min with the engine on
(`test_car_demo` holds that on the shared fixture). Frames need `IG.2`.

docs/physics-model.md §6.4, §6.6; ADR 0096, ADR 0097, ADR 0100; roadmap TC.6.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from irsim.config.scene import load_scene_config
from irsim.scene import Scene

REPO = pathlib.Path(__file__).resolve().parents[2]
OVERCAST = REPO / "configs/scenes/car_ignition_overcast_night.yaml"
CLEAR = REPO / "configs/scenes/car_ignition_clear_night.yaml"

HEAD = """
schema_version: 9
scene:
  name: bracket_lag
  description: "a bracket bolted to a block held at its thermostat"
  weather_file: weather/clear_midlat_summer_48h.csv
  atmosphere_preset: us_standard_clear
  site: {latitude_deg: 45.0, longitude_deg: 30.0, altitude_m: 120.0}
  start_utc: "2024-06-21T12:00:00Z"
  targets: []
  thermal:
    tick_s: 60.0
    surfaces:
      - {name: kerb, material: concrete, tilt_deg: 0.0}
    nodes:
      - {name: block, fixed: 363.15}
      - {name: air, fixed: 300.0}
      - {name: bracket, mass_kg: 0.6, specific_heat_j_kgk: 470.0}
    links:
      - {a: block, b: bracket, joint: bolted_ferrous_new, area_m2: 25.0e-4}
      - {a: bracket, b: air, h_w_m2_k: 40.0, area_m2: 0.05}
"""


def test_a_bracket_lags_the_block_by_its_rc_time_and_settles_at_the_two_resistor_share(
    tmp_path,
) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "bracket.yaml"
    path.write_text(HEAD)
    scene = Scene.from_file(path)
    net = scene.network
    assert net is not None
    c = 0.6 * 470.0
    g, ha = 12000.0 * 25e-4, 40.0 * 0.05
    tau = c / (g + ha)  # 8.8 s: a bolted bracket follows the block within a minute
    share = g / (g + ha) * (363.15 - 300.0)
    # Fine ticks for the transient, then the settle.
    rise_at_tau = None
    while net.t_s < scene.t0_s + 5.0 * tau:
        net.advance(net.t_s, tau / 50.0)
        if rise_at_tau is None and net.t_s >= scene.t0_s + tau:
            rise_at_tau = net.temperature("bracket") - 300.0
    assert rise_at_tau is not None
    # 1 − e⁻¹ of the way there at t = τ, to the first-order step's own accuracy at dt = τ/50.
    assert abs(rise_at_tau / share - (1.0 - np.exp(-1.0))) < 0.03, rise_at_tau / share
    net.advance_to(scene.t0_s + 200.0 * tau, tau)
    assert net.temperature("bracket") - 300.0 == pytest.approx(share, rel=1e-6)


@pytest.mark.parametrize("yaml_path", [OVERCAST, CLEAR])
def test_the_car_scenes_declare_the_metal_around_the_engine(yaml_path: pathlib.Path) -> None:
    spec = load_scene_config(yaml_path).scene
    assert spec.thermal is not None
    names = {n.name for n in spec.thermal.nodes}
    assert {"block", "mounts", "subframe", "bracket", "wing", "air"} <= names
    block = next(n for n in spec.thermal.nodes if n.name == "block")
    assert block.follows == {"target": "engine_bay", "node": "block"}
    switched = [lk for lk in spec.thermal.links if lk.switch is not None]
    assert switched and switched[0].switch == "engine_bay" and switched[0].off_h_w_m2_k == 5.0
    engine = next(t for t in spec.targets if t.name == "engine_bay")
    assert engine.solver == "engine" and engine.load[2] == 0.10 and engine.load_s[3] == 1200.0


@pytest.mark.slow
def test_parts_warm_in_conductance_order_and_the_bracket_follows_the_block() -> None:
    """Block → bracket (a 30 W/K bolted joint) → mounts (12 W/K of rubber) → wing (two small
    bolts) → subframe (behind the rubber, with mass). Measured at 600 s: +51, +45, +27, +4 K."""
    scene = Scene.from_file(OVERCAST)
    for k in range(60):
        scene.advance_targets(10.0 * k, 10.0)
    t = scene.network.temperatures_k  # type: ignore[union-attr]
    air = t["air"]
    assert t["block"] > t["bracket"] > t["mounts"] > t["wing"] > air
    assert t["block"] == pytest.approx(scene.targets["engine_bay"].node_temperature_k("block"))
    assert t["bracket"] - air > 0.8 * (t["block"] - air), "a bolted bracket tracks the block"
    assert t["wing"] - air > 2.0, "and the wing behind two small bolts has warmed by kelvins"
    # Key-off at 1200 s: the bracket's fan-driven convection stops. Advance to 1500 s.
    for k in range(60, 150):
        scene.advance_targets(10.0 * k, 10.0)
    assert scene.targets["engine_bay"].load_at(scene.t0_s + 1500.0) == 0.0
    t = scene.network.temperatures_k  # type: ignore[union-attr]
    assert t["block"] > t["bracket"] > t["mounts"] > t["wing"] > t["air"]
