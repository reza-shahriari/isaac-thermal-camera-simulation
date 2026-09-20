"""TC.4 — nodes, links and joints in the scene schema, and the joint table with its provenance.

The network shipped in TC.2 could be built only in Python. This is what lets a scene config say
"a bracket bolted to the block through 25 cm² of a new ferrous joint", with the conductance and
its source read from `configs/thermal/joints.yaml` rather than typed into the scene.

The two checks the roadmap row names: the loader refuses a contact conductance outside
1e2–1e6 W m⁻² K⁻¹ (a per-K typo such as 1e-3 turns a bolted joint into an insulator, and the
scene would render plausibly with the bracket cold); and a bracket joined to a 400 °C part
through 25 cm² at 12 kW m⁻² K⁻¹ against 1 kW m⁻² K⁻¹ shows steady rises in the ratio the
two-resistor closed form gives, to 1e-6.

docs/physics-model.md §6.4; spec issue S42; ADR 0096, ADR 0097; roadmap TC.4.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest
import yaml

from irsim.config.joints import H_C_RANGE_W_M2_K, JOINTS_PATH, load_joint_table
from irsim.config.scene import load_scene_config
from irsim.scene import Scene

REPO = pathlib.Path(__file__).resolve().parents[2]
FACETS = REPO / "configs/scenes/thermal_facet_scene.yaml"

HEAD = """
schema_version: 9
scene:
  name: bracket_test
  description: "a bracket bolted to a hot block, from the scene config alone"
  weather_file: weather/clear_midlat_summer_48h.csv
  atmosphere_preset: us_standard_clear
  site: {{latitude_deg: 45.0, longitude_deg: 30.0, altitude_m: 120.0}}
  start_utc: "2024-06-21T12:00:00Z"
  targets: []
  thermal:
    tick_s: 60.0
    surfaces:
      - {{name: kerb, material: concrete, tilt_deg: 0.0}}
{network}
"""

BRACKET = """
    nodes:
      - {{name: block, fixed: 673.15}}
      - {{name: air, fixed: {air}}}
      - {{name: bracket, mass_kg: 0.6, specific_heat_j_kgk: 470.0}}
    links:
      - {{a: block, b: bracket, {joint}, area_m2: 25.0e-4}}
      - {{a: bracket, b: air, h_w_m2_k: 12.0, area_m2: 0.05}}
"""


def _write(tmp_path: pathlib.Path, network: str, name: str = "scene.yaml") -> pathlib.Path:
    out = tmp_path / name
    out.write_text(HEAD.format(network=network))
    return out


# --- the table ------------------------------------------------------------------------------------


def test_the_joint_table_carries_the_survey_s_numbers_with_provenance() -> None:
    table = load_joint_table()
    assert table.joint("bolted_ferrous_new").h_c_w_m2_k == 12000.0
    assert table.joint("bolted_ferrous_corroded").h_c_w_m2_k == 7000.0
    assert table.joint("bolted_ferrous_paste").h_c_w_m2_k == 59000.0
    assert table.joint("dry_default").h_c_w_m2_k == 1000.0
    assert table.fastener("small_bolt").g_w_k == 1.0
    for spec in (*table.joints.values(), *table.fasteners.values()):
        assert spec.status in ("MEASURED", "ESTIMATED")
        assert len(spec.source) > 20
    assert table.joint("bolted_ferrous_new").status == "MEASURED"
    assert "Voller" in table.joint("bolted_ferrous_new").source
    assert table.joint("dry_default").status == "ESTIMATED"
    assert table.fastener("small_bolt").status == "ESTIMATED"  # measured in vacuum, on aluminium
    with pytest.raises(KeyError, match="unknown joint"):
        table.joint("welded")


def test_the_loader_refuses_a_per_k_typo(tmp_path) -> None:  # type: ignore[no-untyped-def]
    raw = yaml.safe_load(JOINTS_PATH.read_text())
    lo, hi = H_C_RANGE_W_M2_K
    for bad in (1e-3, lo / 2.0, hi * 2.0):
        raw["joints"]["dry_default"]["h_c_w_m2_k"] = bad
        path = tmp_path / "joints.yaml"
        path.write_text(yaml.safe_dump(raw))
        with pytest.raises(ValueError, match="outside"):
            load_joint_table(path)
    raw["joints"]["dry_default"]["h_c_w_m2_k"] = 1000.0
    raw["fasteners"]["small_bolt"]["g_w_k"] = 5000.0
    (tmp_path / "joints.yaml").write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="outside"):
        load_joint_table(tmp_path / "joints.yaml")


def test_an_inline_h_c_is_range_checked_like_the_table(tmp_path) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ValueError, match="outside"):
        load_scene_config(
            _write(tmp_path, BRACKET.format(joint="h_c_w_m2_k: 1.0e-3", air="ambient"))
        )


# --- the closed form ------------------------------------------------------------------------------


def _steady_rise(tmp_path: pathlib.Path, joint: str) -> tuple[float, float, float]:
    """(bracket rise over air, G, hA) after the bracket has settled (τ well under an hour)."""
    # Fixed air: against a moving ambient the bracket lags by τ·dT_air/dt and 1e-6 is not a
    # statement about the network. `test_an_ambient_node_follows_the_scene_s_one_weather` has it.
    scene = Scene.from_file(
        _write(tmp_path, BRACKET.format(joint=joint, air=300.0), f"{joint[-6:]}.yaml")
    )
    assert scene.network is not None
    for _ in range(120):
        scene.advance_targets(scene.network.t_s - scene.t0_s, 60.0)
    net = scene.network
    t_air = net.temperature("air")
    k = net.conductances_w_k(net.t_s)
    i, j, a = net.names.index("bracket"), net.names.index("block"), net.names.index("air")
    return net.temperature("bracket") - t_air, float(k[i, j]), float(k[i, a])


@pytest.mark.slow
def test_a_new_joint_against_a_dry_one_follows_the_two_resistor_form(
    tmp_path,
) -> None:  # type: ignore[no-untyped-def]
    """ΔT_bracket = G/(G + hA) · (T_block − T_air), so the two rises' ratio is the closed form's."""
    rise_new, g_new, ha = _steady_rise(tmp_path, "joint: bolted_ferrous_new")
    rise_dry, g_dry, ha_2 = _steady_rise(tmp_path, "joint: dry_default")
    assert ha == ha_2 == pytest.approx(12.0 * 0.05)
    assert g_new == pytest.approx(12000.0 * 25e-4) and g_dry == pytest.approx(1000.0 * 25e-4)
    expected = (g_new / (g_new + ha)) / (g_dry / (g_dry + ha))
    assert rise_new / rise_dry == pytest.approx(expected, rel=1e-6)
    # And each rise on its own is the closed form against the block's 400 °C over the weather's air.
    assert rise_dry == pytest.approx(g_dry / (g_dry + ha) * (673.15 - 300.0), rel=1e-6)


# --- what the block builds ------------------------------------------------------------------------


def test_every_link_form_and_node_kind_builds(tmp_path) -> None:  # type: ignore[no-untyped-def]
    network = """
    nodes:
      - {name: block, capacity_j_k: 60000.0, initial_k: 350.0}
      - {name: bracket, mass_kg: 0.6, specific_heat_j_kgk: 470.0}
      - {name: wing, capacity_j_k: 3600.0}
      - {name: bay_air, capacity_j_k: 360.0}
      - {name: air, fixed: ambient}
      - {name: coolant, fixed: 363.0}
      - {name: mount, link_node: {a: block, b: wing, g_w_k: 2.0}, capacity_j_k: 900.0}
    links:
      - {a: block, b: bracket, joint: bolted_ferrous_new, area_m2: 25.0e-4}
      - {a: bracket, b: wing, fastener: small_bolt, count: 2}
      - {a: block, b: coolant, g_w_k: 40.0}
      - {a: block, b: bay_air, h_w_m2_k: 15.0, area_m2: 1.2}
      - {a: bay_air, b: air, h_c_w_m2_k: 1000.0, area_m2: 0.001}
      - {a: block, b: wing, radiation: {emissivity: 0.8, area_m2: 0.3, view_factor: 0.4}}
    sources:
      - node: block
        times_s: [0.0, 29.9, 30.0, 1200.0, 1200.1]
        power_w: [0.0, 0.0, 9000.0, 9000.0, 0.0]
      - {node: wing, power_w: 5.0}
"""
    scene = Scene.from_file(_write(tmp_path, network))
    net = scene.network
    assert net is not None
    assert set(net.names) == {"block", "bracket", "wing", "bay_air", "mount", "air", "coolant"}
    assert net.n_fixed == 2 and net.weather is scene.weather
    # Initial temperatures: the declared one, and the weather's air for the rest.
    t_air_0 = scene.weather.at(scene.t0_s).t_air_k
    assert net.temperature("block") == 350.0 and net.temperature("bracket") == t_air_0
    assert net.temperature("air") == t_air_0 and net.temperature("coolant") == 363.0
    k = net.conductances_w_k(net.t_s)
    idx = net.names.index
    assert k[idx("block"), idx("bracket")] == 12000.0 * 25e-4
    assert k[idx("bracket"), idx("wing")] == 2.0
    assert k[idx("block"), idx("mount")] == 4.0 and k[idx("mount"), idx("wing")] == 4.0
    assert k[idx("bay_air"), idx("air")] == 1.0
    assert k[idx("block"), idx("wing")] > 0.0  # the radiation link, linearised
    q = net.imposed_w
    assert q(0.0)[idx("block")] == 0.0 and q(600.0)[idx("block")] == 9000.0
    assert q(1200.05)[idx("block")] == pytest.approx(4500.0) and q(5000.0)[idx("block")] == 0.0
    assert q(0.0)[idx("wing")] == 5.0
    # `advance_targets` steps the network and reports its nodes.
    out = scene.advance_targets(0.0, 60.0)
    assert "bracket" in out and out["bracket"] == scene.node_temperature_k("bracket")
    assert net.t_s == scene.t0_s + 60.0


def test_an_ambient_node_follows_the_scene_s_one_weather(tmp_path) -> None:  # type: ignore[no-untyped-def]
    scene = Scene.from_file(
        _write(tmp_path, BRACKET.format(joint="joint: dry_default", air="ambient"))
    )
    net = scene.network
    assert net is not None
    for t_rel in (0.0, 1800.0, 7200.0):
        assert net.fixed[1].at(scene.t0_s + t_rel) == scene.weather.at(scene.t0_s + t_rel).t_air_k
    assert "network" in scene.consumers


def test_a_scene_without_nodes_has_no_network_and_is_unchanged() -> None:
    scene = Scene.from_file(FACETS)
    assert scene.network is None
    with pytest.raises(ValueError, match="no thermal network"):
        scene.node_temperature_k("block")


# --- refusals ------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("network", "message"),
    [
        (
            "    nodes:\n      - {name: a, capacity_j_k: 1.0}\n    links:\n"
            "      - {a: a, b: ghost, g_w_k: 1.0}\n",
            "unknown node",
        ),
        (
            "    nodes:\n      - {name: air, fixed: ambient}\n"
            "      - {name: a, capacity_j_k: 1.0}\n"
            "    sources:\n      - {node: air, power_w: 5.0}\n",
            "goes nowhere",
        ),
        (
            "    nodes:\n      - {name: a, capacity_j_k: 1.0}\n"
            "      - {name: b, capacity_j_k: 1.0}\n"
            "    links:\n      - {a: a, b: b, g_w_k: 1.0, joint: dry_default, area_m2: 0.1}\n",
            "exactly one form",
        ),
        (
            "    nodes:\n      - {name: a, capacity_j_k: 1.0}\n"
            "      - {name: b, capacity_j_k: 1.0}\n"
            "    links:\n      - {a: a, b: b, joint: dry_default}\n",
            "needs area_m2",
        ),
        (
            "    nodes:\n      - {name: a, capacity_j_k: 1.0, fixed: 300.0}\n",
            "exactly one of",
        ),
        (
            "    nodes:\n      - {name: a, capacity_j_k: 1.0}\n"
            "      - {name: a, capacity_j_k: 2.0}\n",
            "unique",
        ),
        (
            "    nodes:\n      - {name: a, capacity_j_k: 1.0}\n"
            "    sources:\n      - {node: a, times_s: [0.0, 1.0], power_w: [1.0]}\n",
            "pair up",
        ),
    ],
)
def test_malformed_networks_fail_at_load(tmp_path, network, message) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ValueError, match=message):
        load_scene_config(_write(tmp_path, network))


def test_an_unknown_joint_name_fails_at_build(tmp_path) -> None:  # type: ignore[no-untyped-def]
    path = _write(tmp_path, BRACKET.format(joint="joint: welded", air="ambient"))
    load_scene_config(path)  # the schema cannot know the table's names; the build does
    with pytest.raises(KeyError, match="unknown joint"):
        Scene.from_file(path)


def test_the_kerb_and_the_rest_of_the_thermal_block_are_untouched(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """A v9 network beside a v7 surface: the surface solves exactly as it did."""
    with_net = Scene.from_file(
        _write(tmp_path, BRACKET.format(joint="joint: dry_default", air="ambient"))
    )
    without = Scene.from_file(_write(tmp_path, "", "bare.yaml"))
    assert np.array_equal(
        with_net.thermal.temperature_at(with_net.t0_s), without.thermal.temperature_at(without.t0_s)
    )
