"""Engine-free tests for the aerial thermal bridge (roadmap M10.18, ADR 0014/0060).

The bridge is what makes surface temperature *exact* in a pipeline whose renderer cannot carry a
temperature at all. All of its arithmetic -- the tick/interpolate clock, the instance-id table, the
per-pixel sky substitution -- is plain NumPy over an id plane and a label table, so it is verified
here rather than inside a Kit boot; ``tests/integration/test_aerial_bridge_isaac.py`` then checks
only that a rendered frame's ids and rays feed it correctly.

The tolerances are the ones the physics needs, not round numbers: 10 mK is a fifth of the 50 mK
NETD the whole design exists to protect, and 0.5 K is the sky-model agreement the MS.2 elevation
LUT is quoted to.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from irsim.config.scene import SceneConfig, TargetSpec, load_scene_config
from irsim.radiometry.lut import BandLUT
from irsim.scene import Scene
from irsim_isaac.pipeline.aerial_bridge import (
    AerialThermalBridge,
    elevation_from_rays,
)

REPO = pathlib.Path(__file__).resolve().parents[2]
SCENE_YAML = REPO / "configs" / "scenes" / "sky_target_clear_day.yaml"

PRESCRIBED_K = 317.25
READBACK_TOL_K = 0.010
TRACKING_TOL_K = 0.010
SKY_TOL_K = 0.5

PRIMS = {"/World/Drone/Body": "airframe", "/World/Drone/Motor": "engine"}
LABELS = {"0": "BACKGROUND", "4": "/World/Drone/Body", "5": "/World/Drone/Motor"}


def _scene(lut: BandLUT, targets: list[TargetSpec] | None = None) -> Scene:
    config = load_scene_config(SCENE_YAML)
    if targets is not None:
        spec = config.scene.model_copy(update={"targets": targets})
        config = SceneConfig(schema_version=config.schema_version, scene=spec)
    return Scene.from_config(config, {"lwir": lut})


def _constant_targets() -> list[TargetSpec]:
    """An airframe that cools and a motor pinned at exactly PRESCRIBED_K."""
    return [
        TargetSpec(name="airframe", solver="newton", t0_k=296.15, tau_s=900.0),
        TargetSpec(
            name="engine",
            solver="prescribed",
            schedule_s=[0.0, 7200.0],
            schedule_k=[PRESCRIBED_K, PRESCRIBED_K],
        ),
    ]


# --- the facet table ------------------------------------------------------------------------


def test_a_prescribed_temperature_reads_back_exactly(tophat_lwir_lut: BandLUT) -> None:
    """317.25 K authored must come back as 317.25 K -- this is the whole point of the table.

    Through the fp16 emissive colour path ADR 0014 rejected, the same number would come back
    quantised to about 100 mK, ten times this tolerance and twice the sensor's NETD.
    """
    scene = _scene(tophat_lwir_lut, _constant_targets())
    bridge = AerialThermalBridge(scene, PRIMS, band="lwir")
    bridge.advance_to(3.5)

    table = bridge.facet_table(LABELS)
    assert table.dtype == np.float32
    assert abs(float(table[5]) - PRESCRIBED_K) < READBACK_TOL_K

    ids = np.array([[0, 4], [5, 5]], dtype=np.uint32)
    plane = bridge.temperature_plane(ids, LABELS)
    assert plane.dtype == np.float32
    assert np.all(np.abs(plane[1, :] - PRESCRIBED_K) < READBACK_TOL_K)


def test_the_table_resolves_a_50_mk_difference_that_float16_would_erase(
    tophat_lwir_lut: BandLUT,
) -> None:
    """Two targets 50 mK apart -- one NETD -- must stay distinguishable in the table.

    This is CLAUDE.md #2 stated as a measurement rather than a dtype assertion: at 300 K float16
    spaces 0.25 K, so 300.000 K and 300.050 K collapse onto the same value and the sensor's entire
    sensitivity disappears while the image still looks fine.
    """
    targets = [
        TargetSpec(
            name="airframe",
            solver="prescribed",
            schedule_s=[0.0, 7200.0],
            schedule_k=[300.0, 300.0],
        ),
        TargetSpec(
            name="engine",
            solver="prescribed",
            schedule_s=[0.0, 7200.0],
            schedule_k=[300.05, 300.05],
        ),
    ]
    scene = _scene(tophat_lwir_lut, targets)
    bridge = AerialThermalBridge(scene, PRIMS, band="lwir")
    bridge.advance_to(2.0)
    table = bridge.facet_table(LABELS)

    assert table.dtype == np.float32
    separation = float(table[5]) - float(table[4])
    assert abs(separation - 0.050) < 0.001, f"50 mK became {separation * 1e3:.1f} mK"
    # the same pair through the rejected fp16 path is indistinguishable
    as_half = table.astype(np.float16)
    assert float(as_half[5]) == float(as_half[4])


def test_a_rendered_prim_with_no_thermal_node_raises(tophat_lwir_lut: BandLUT) -> None:
    """A surface with no temperature must not be given a plausible one."""
    scene = _scene(tophat_lwir_lut, _constant_targets())
    bridge = AerialThermalBridge(scene, PRIMS, band="lwir")
    with pytest.raises(KeyError, match="no thermal node"):
        bridge.facet_table({**LABELS, "6": "/World/Drone/Antenna"})

    lenient = bridge.facet_table({**LABELS, "6": "/World/Drone/Antenna"}, strict=False)
    assert float(lenient[6]) == 0.0, "the non-strict fill must be obviously wrong, not plausible"


def test_a_target_name_the_scene_does_not_define_is_refused(tophat_lwir_lut: BandLUT) -> None:
    scene = _scene(tophat_lwir_lut, _constant_targets())
    with pytest.raises(ValueError, match="does not define"):
        AerialThermalBridge(scene, {"/World/X": "no_such_target"}, band="lwir")


# --- the thermal clock ----------------------------------------------------------------------


def test_a_newton_target_tracks_its_solver_over_120_frames(tophat_lwir_lut: BandLUT) -> None:
    """1 Hz ticks + linear interpolation must stay within 10 mK of the exact solver.

    The reference is a second, independent Scene whose solver is advanced at the frame rate, so
    the comparison is against the exponential itself rather than against a cached answer.
    """
    scene = _scene(tophat_lwir_lut, _constant_targets())
    reference = _scene(tophat_lwir_lut, _constant_targets())
    bridge = AerialThermalBridge(scene, PRIMS, band="lwir", tick_hz=1.0)

    dt = 1.0 / 30.0
    worst = 0.0
    for frame in range(1, 121):
        t = frame * dt
        got = bridge.advance_to(t)["airframe"]
        expected = reference.advance_targets(t - dt, dt)["airframe"]
        worst = max(worst, abs(got - expected))
    assert worst < TRACKING_TOL_K, f"worst tracking error {worst * 1e3:.3f} mK"
    assert worst > 0.0, "a perfect match would mean the reference is not independent"


def test_the_interpolation_error_bound_is_reported_and_tiny(tophat_lwir_lut: BandLUT) -> None:
    """tau = 900 s across a 1 s tick curves by microkelvin; the bound is computed, not assumed."""
    scene = _scene(tophat_lwir_lut, _constant_targets())
    bridge = AerialThermalBridge(scene, PRIMS, band="lwir")
    bridge.advance_to(10.0)
    assert bridge.tick_error_k("airframe") < 1e-4
    assert bridge.tick_error_k("engine") == 0.0  # a prescribed node has no time constant


def test_the_clock_cannot_rewind(tophat_lwir_lut: BandLUT) -> None:
    """Solvers are stateful, so replaying a frame would silently change the history."""
    scene = _scene(tophat_lwir_lut, _constant_targets())
    bridge = AerialThermalBridge(scene, PRIMS, band="lwir")
    bridge.advance_to(5.0)
    with pytest.raises(ValueError, match="rewind"):
        bridge.advance_to(4.0)


def test_ticking_is_independent_of_the_frame_rate(tophat_lwir_lut: BandLUT) -> None:
    """Two frame rates over the same interval must agree: the solver ticks on its own clock."""
    fast = AerialThermalBridge(_scene(tophat_lwir_lut, _constant_targets()), PRIMS, band="lwir")
    slow = AerialThermalBridge(_scene(tophat_lwir_lut, _constant_targets()), PRIMS, band="lwir")
    for i in range(1, 241):
        fast.advance_to(i / 60.0)
    for i in range(1, 41):
        slow.advance_to(i / 10.0)
    assert abs(fast.temperatures()["airframe"] - slow.temperatures()["airframe"]) < 1e-9


# --- one weather object (CLAUDE.md #6) --------------------------------------------------------


def test_a_sky_model_on_different_weather_is_refused(tophat_lwir_lut: BandLUT) -> None:
    """A scene may not run one weather in the targets and another in the sky."""
    scene = _scene(tophat_lwir_lut, _constant_targets())
    other = _scene(tophat_lwir_lut, _constant_targets())
    assert other.weather is not scene.weather
    with pytest.raises(ValueError, match="different WeatherSeries"):
        AerialThermalBridge(scene, PRIMS, sky=other.sky_models["lwir"])

    # the scene's own sky model is accepted
    AerialThermalBridge(scene, PRIMS, sky=scene.sky_models["lwir"])


# --- the sky ----------------------------------------------------------------------------------


def test_sky_pixels_take_t_sky_of_their_own_elevation(tophat_lwir_lut: BandLUT) -> None:
    """Background pixels must follow the MS.2 elevation profile, not one number per frame."""
    scene = _scene(tophat_lwir_lut, _constant_targets())
    bridge = AerialThermalBridge(scene, PRIMS, band="lwir")
    bridge.advance_to(2.0)

    elevation = np.deg2rad(np.array([[5.0, 30.0], [60.0, 89.0]]))
    ids = np.zeros((2, 2), dtype=np.uint32)
    plane = bridge.temperature_plane(ids, LABELS, elevation_rad=elevation)
    expected = bridge.background_temperature_k(elevation)

    assert np.max(np.abs(plane.astype(np.float64) - expected)) < SKY_TOL_K
    # the profile must actually vary: a constant sky would pass a per-pixel test trivially
    assert float(np.ptp(expected)) > 5.0, "the clear-sky elevation gradient should span > 5 K"
    assert expected[0, 0] > expected[1, 1], "the horizon is warmer than the zenith"


def test_below_the_horizon_the_background_is_ground_not_sky(tophat_lwir_lut: BandLUT) -> None:
    """A background ray aimed downwards sees ground; extrapolating the sky there inverts contrast.

    The sky model is only defined on [0, 90] degrees. Extending it below the horizon would report
    a 40-60 K cold sky where warm ground actually is, which flips the sign of the contrast of
    anything silhouetted against it -- and for a downward-looking aerial camera that is most of
    the frame.
    """
    from irsim.pipeline.environment import ground_temperature_k

    scene = _scene(tophat_lwir_lut, _constant_targets())
    bridge = AerialThermalBridge(scene, PRIMS, band="lwir")
    bridge.advance_to(1.0)

    elevation = np.deg2rad(np.array([[-30.0, -1.0], [1.0, 45.0]]))
    got = bridge.background_temperature_k(elevation)
    t_ground = ground_temperature_k(scene.sky_models["lwir"], scene.t0_s + 1.0)

    assert got[0, 0] == pytest.approx(t_ground)
    assert got[0, 1] == pytest.approx(t_ground)
    assert got[1, 0] < t_ground, "the sky just above the horizon is colder than the ground"
    assert got[1, 1] < got[1, 0], "and colder again higher up"
    # the sky model itself refuses the negative elevations, which is why the split exists
    with pytest.raises(ValueError, match=r"\[0, 90\]"):
        bridge.sky_temperature(elevation)


def test_geometry_pixels_are_untouched_by_the_sky(tophat_lwir_lut: BandLUT) -> None:
    scene = _scene(tophat_lwir_lut, _constant_targets())
    bridge = AerialThermalBridge(scene, PRIMS, band="lwir")
    bridge.advance_to(1.0)
    ids = np.array([[0, 5]], dtype=np.uint32)
    elevation = np.deg2rad(np.array([[20.0, 20.0]]))
    plane = bridge.temperature_plane(ids, LABELS, elevation_rad=elevation)
    assert abs(float(plane[0, 1]) - PRESCRIBED_K) < READBACK_TOL_K
    assert abs(float(plane[0, 0]) - PRESCRIBED_K) > 10.0, "the sky is not the target"


def test_without_elevations_the_sky_is_left_to_the_caller(tophat_lwir_lut: BandLUT) -> None:
    scene = _scene(tophat_lwir_lut, _constant_targets())
    bridge = AerialThermalBridge(scene, PRIMS, band="lwir")
    plane = bridge.temperature_plane(np.zeros((2, 2), dtype=np.uint32), LABELS)
    assert np.all(plane == 0.0)


# --- ray elevation -----------------------------------------------------------------------------


def test_elevation_from_rays_at_zenith_horizon_and_nadir() -> None:
    rays = np.array([[[0.0, 1.0, 0.0], [0.0, 0.0, -1.0], [0.0, -1.0, 0.0]]])
    elevation = elevation_from_rays(rays)
    assert elevation[0, 0] == pytest.approx(np.pi / 2)
    assert elevation[0, 1] == pytest.approx(0.0, abs=1e-12)
    assert elevation[0, 2] == pytest.approx(-np.pi / 2)


def test_elevation_uses_the_stage_up_axis() -> None:
    rays = np.array([[[0.0, 0.0, 1.0]]])
    assert elevation_from_rays(rays, up=(0.0, 0.0, 1.0))[0, 0] == pytest.approx(np.pi / 2)
    assert elevation_from_rays(rays, up=(0.0, 1.0, 0.0))[0, 0] == pytest.approx(0.0, abs=1e-12)
    with pytest.raises(ValueError, match="non-zero"):
        elevation_from_rays(rays, up=(0.0, 0.0, 0.0))
    with pytest.raises(ValueError, match=r"\(H, W, 3\)"):
        elevation_from_rays(np.zeros((2, 2)))


# --- the network's nodes are in the clock too ----------------------------------------------------

NETWORK_SCENE = REPO / "configs" / "scenes" / "car_ignition_clear_night.yaml"


def test_the_first_bracket_names_every_node_a_tick_will_report() -> None:
    """A scene with a `nodes:` block reports its network under its own names as well as its
    targets (`Scene.advance_targets`, TC.4). The bridge seeds its first bracket before any tick
    has run, so if it seeds from `scene.targets` alone the first tick widens `next_k` past
    `prev_k` and `interpolate` raises `KeyError` on frame zero -- which is what every car render
    did between TC.4 and this fix.

    The assertion is not merely that it does not raise: a bracket seeded with the wrong *values*
    would also not raise. Each network node must start at the network's own temperature.
    """
    scene = Scene.from_file(NETWORK_SCENE)
    expected = dict(scene.network.temperatures_k)
    assert expected, "this scene is supposed to declare a network"

    bridge = AerialThermalBridge(Scene.from_file(NETWORK_SCENE), {"/World/Car/shell": "shell"})
    bracket = bridge._bracket
    assert not set(bracket.next_k) - set(bracket.prev_k), sorted(
        set(bracket.next_k) - set(bracket.prev_k)
    )
    # Frame zero is the one that used to raise, and it lands inside the first tick.
    first = bridge.advance_to(0.0)
    for name, kelvin in expected.items():
        assert name in first, name
        assert first[name] == pytest.approx(kelvin, abs=1e-9), name
    # And the nodes really do move afterwards, so the seeding is a starting point and not a pin.
    later = bridge.advance_to(120.0)
    assert later["block"] - first["block"] > 1.0, (first["block"], later["block"])
