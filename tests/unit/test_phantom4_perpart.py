"""Every part of the Phantom 4 carries its own temperature history (`phantom4_perpart.yaml`).

`phantom4_parts.yaml` gives the decomposition of ADR 0138 four thermal nodes, so its four motors
read identically at every instant. A viewer cannot tell that from a simulator unable to do better,
which is what the per-part scene exists to refute. These tests pin the three claims the scene
makes, on the scene's own solvers, engine-free:

* **Nineteen parts, no two on the same node.** `render_phantom4.target_for_part` resolves each
  part of `configs/assets/phantom4_parts.yaml` against the scene's target names, and every
  part lands on a different node or on its own mesh surface -- the scene, not the driver's map,
  decides how fine the nodes are.
* **Same-kind parts differ, the way a flight makes them differ.** In the right-hand orbit the
  left pair runs hotter than the right pair by more than 4 K (eighty NETDs on a Boson); in the
  left-hand orbit the mirror; in cruise the rear pair leads the front.
* **They converge again.** Landed and idle, the four motors are within 0.1 K of each other: the
  differences were the mission, not four different models.
"""

from __future__ import annotations

import importlib.util
import pathlib

import pytest
import yaml

from irsim.config.scene import load_scene_config
from irsim.scene import Scene

REPO = pathlib.Path(__file__).resolve().parents[2]
SCENE = REPO / "configs" / "scenes" / "phantom4_perpart.yaml"
PARENT = REPO / "configs" / "scenes" / "phantom4_parts.yaml"
ASSET = REPO / "configs" / "assets" / "phantom4_parts.yaml"

STATIONS = ("front_left", "front_right", "rear_left", "rear_right")
#: A Boson 640's NETD is 50 mK; a motor-to-motor difference the demo relies on has to be many
#: times that, or the picture would show it only in the legend.
MIN_PAIR_SPLIT_K = 4.0
#: Landed and idle: the nodes are T_air + 45 K x 0.05^2 = 0.11 K over air, all four alike.
CONVERGED_K = 0.1


@pytest.fixture(scope="module")
def driver():  # type: ignore[no-untyped-def]
    path = REPO / "scripts" / "render_phantom4.py"
    spec = importlib.util.spec_from_file_location("render_phantom4", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def scene(tophat_lwir_lut):  # type: ignore[no-untyped-def]
    return Scene.from_config(load_scene_config(SCENE), {"lwir": tophat_lwir_lut})


def _parts() -> list[str]:
    raw = yaml.safe_load(ASSET.read_text())
    return [p["name"] for p in raw["asset"]["parts"]["parts"]]


def _motors_c(scene: Scene, t_rel_s: float) -> dict[str, float]:
    return {
        station: float(scene.targets[f"motor_{station}"].advance(scene.t0_s + t_rel_s, 0.0))
        - 273.15
        for station in STATIONS
    }


def test_every_part_of_the_asset_resolves_to_its_own_node_or_surface(driver, scene):  # type: ignore[no-untyped-def]
    parts = _parts()
    assert len(parts) == 19
    defined = set(scene.targets)
    surfaces = set(scene.thermal_surfaces)
    nodes = {part: driver.target_for_part(part, defined) for part in parts}
    # Every fallback the driver hands the bridge is a node the scene defines.
    assert set(nodes.values()) <= defined
    # A part has its own temperature if it has its own node, or its own mesh surface (whose cells
    # reach the pixels directly and whose fallback is only what an uncovered pixel gets).
    own = {p for p, n in nodes.items() if n == p} | {p for p in parts if p in surfaces}
    missing = sorted(set(parts) - own)
    assert missing == [], f"parts sharing a node with another part: {missing}"
    # And the same driver, on the four-node parent scene, still collapses to four nodes.
    parent = load_scene_config(PARENT)
    parent_defined = {t.name for t in parent.scene.targets}
    parent_nodes = {driver.target_for_part(part, parent_defined) for part in parts}
    assert parent_nodes == {"airframe", "motor", "esc", "battery"}


def test_a_part_the_scene_never_heard_of_falls_to_the_airframe(driver):  # type: ignore[no-untyped-def]
    assert driver.target_for_part("antenna", {"airframe", "motor"}) == "airframe"
    # The candidate order is honoured: a station ESC node wins over the generic one.
    assert (
        driver.target_for_part("motor_mount_rear_left", {"esc", "esc_rear_left"}) == "esc_rear_left"
    )
    assert driver.target_for_part("motor_mount_rear_left", {"esc"}) == "esc"


def test_same_kind_parts_differ_the_way_the_flight_makes_them_differ(scene):  # type: ignore[no-untyped-def]
    cruise = _motors_c(scene, 450.0)
    assert (
        min(cruise["rear_left"], cruise["rear_right"])
        - max(cruise["front_left"], cruise["front_right"])
        > MIN_PAIR_SPLIT_K
    ), f"nose-down cruise should load the rear pair: {cruise}"
    right = _motors_c(scene, 750.0)
    assert (
        min(right["front_left"], right["rear_left"])
        - max(right["front_right"], right["rear_right"])
        > MIN_PAIR_SPLIT_K
    ), f"a right-hand orbit should load the outer (left) pair: {right}"
    left = _motors_c(scene, 1050.0)
    assert (
        min(left["front_right"], left["rear_right"]) - max(left["front_left"], left["rear_left"])
        > MIN_PAIR_SPLIT_K
    ), f"a left-hand orbit should load the outer (right) pair: {left}"
    # And all four are distinct in the orbit, not two pairs: the outer-rear motor leads.
    assert len({round(v, 1) for v in right.values()}) == 4


def test_the_motors_converge_again_once_landed(scene):  # type: ignore[no-untyped-def]
    landed = _motors_c(scene, 1680.0)
    assert max(landed.values()) - min(landed.values()) < CONVERGED_K, landed
    hover = _motors_c(scene, 120.0)
    assert max(hover.values()) - min(hover.values()) < CONVERGED_K, hover
    assert min(hover.values()) - max(landed.values()) > 10.0


def test_each_propeller_follows_its_own_motor(scene):  # type: ignore[no-untyped-def]
    """The four propellers' speed schedules are not copies: each tracks its station's throttle."""
    spec = load_scene_config(SCENE).scene
    speeds = {
        s.name: tuple(s.speed_m_s or ()) for s in spec.thermal.surfaces if s.name.startswith("prop")
    }
    assert len(speeds) == 4
    assert len(set(speeds.values())) == 4, "propeller speed schedules should all differ"
    throttle = {t.name: t.throttle for t in spec.targets if t.name.startswith("motor_")}
    for station in STATIONS:
        u = throttle[f"motor_{station}"]
        v = speeds[f"propeller_{station}"]
        # Monotone in throttle wherever the floor is not active.
        for a, b, va, vb in zip(u[:-1], u[1:], v[:-1], v[1:], strict=True):
            if va > 2.0 and vb > 2.0:
                assert (b - a) * (vb - va) >= 0.0
