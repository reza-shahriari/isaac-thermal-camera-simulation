"""Roadmap M10.7b: device-side defects, the iterated replacement and the FFC hold.

These are the three pieces M10.7a left, and unlike the noise they are mostly **exact**, which is
why this file looks different from its neighbour. A defect map is a property of one physical focal
plane -- drawn once per sensor, never per frame -- so the device gets the CPU's map uploaded and
the two are bit-identical by construction rather than by tolerance. The replacement is arithmetic
with no randomness in it at all, so "matches the CPU" means every bit of every pixel. The freeze is
a buffer copy, so a held frame is the same bytes each time or the feature does not work.

The one statistical piece is the RTS chain, for the reason ADR 0022 gives: Warp's generator is not
NumPy's. It is seeded from the CPU's own starting realisation so that what is measured afterwards
is the two generators and not two unrelated defect populations.

Nothing here renders or needs Kit (ADR 0014 addendum).

    $PYTHON -m pytest tests/integration/test_warp_defects_stage.py -m gpu
"""

from __future__ import annotations

import copy
import pathlib
from typing import Any

import numpy as np
import pytest
import yaml

REPO = pathlib.Path(__file__).resolve().parents[2]
BOSON_YAML = REPO / "configs" / "sensors" / "flir_boson_640_lwir.yaml"
DEVICES = ("cuda:0", "cpu")
SHAPE = (48, 56)
#: The RTS statistics need a population, not a handful. At the small format above the cluster
#: process yields six flickering and blinking pixels between them, which is too few to say
#: anything about occupancy or dwell -- and a skipped statistical test proves nothing at all.
#: The defect *fraction* is capped at one percent by the schema, so the population comes from
#: area instead: 128 x 160 at one percent is ~205 defects, ~60 of them stateful.
RTS_SHAPE = (128, 160)
SEED = 90210
pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module")
def warp() -> Any:
    from irsim_isaac.env import ensure_warp_on_path

    ensure_warp_on_path()
    import warp as wp

    wp.init()
    return wp


@pytest.fixture(scope="module")
def sensor() -> Any:
    """The Boson config at a small format, with enough defects to make clusters likely."""
    from irsim.config.sensor import SensorConfig

    raw = copy.deepcopy(yaml.safe_load(BOSON_YAML.read_text()))
    raw["sensor"]["fpa"].update(width=SHAPE[1], height=SHAPE[0])
    raw["sensor"]["noise"]["bad_pixel_fraction"] = 0.01
    raw["sensor"]["noise"]["bad_pixel_cluster_lambda"] = 1.6
    return SensorConfig.model_validate(raw)


@pytest.fixture(scope="module")
def bad_map(sensor: Any) -> Any:
    from irsim.noise.defects import generate_map

    built = generate_map(SHAPE, sensor.sensor.noise, SEED)
    assert built.count > 0, "the fixture needs defects to say anything"
    return built


@pytest.fixture(scope="module")
def rts_sensor() -> Any:
    """A larger plane with more defects, so the chain's statistics have something to stand on."""
    from irsim.config.sensor import SensorConfig

    raw = copy.deepcopy(yaml.safe_load(BOSON_YAML.read_text()))
    raw["sensor"]["fpa"].update(width=RTS_SHAPE[1], height=RTS_SHAPE[0])
    # 0.01 is the schema's ceiling, and it is the right one: a focal plane with more than one
    # percent of its pixels dead is scrap, not a sensor. The population comes from the area.
    raw["sensor"]["noise"]["bad_pixel_fraction"] = 0.01
    raw["sensor"]["noise"]["bad_pixel_cluster_lambda"] = 1.6
    return SensorConfig.model_validate(raw)


@pytest.fixture(scope="module")
def rts_map(rts_sensor: Any) -> Any:
    from irsim.noise.defects import generate_map

    built = generate_map(RTS_SHAPE, rts_sensor.sensor.noise, SEED)
    assert int(built.stateful_mask.sum()) >= 50, "the RTS fixture needs a population"
    return built


def device_state(device: str) -> Any:
    from irsim_isaac.pipeline.warp_stages import WarpPipelineState

    return WarpPipelineState(device)


def ramp(shape: tuple[int, int] = SHAPE) -> np.ndarray:
    """A smooth float DN plane with sub-LSB structure, so rounding differences would show."""
    rows, cols = shape
    y, x = np.mgrid[0:rows, 0:cols]
    return (1000.0 + 7.25 * x + 3.5 * y + 0.125 * ((x * y) % 8)).astype(np.float32)


# --- the map ----------------------------------------------------------------------------------


@pytest.mark.parametrize("device", DEVICES)
def test_the_defect_map_reaches_the_device_bit_identically(
    warp: Any, bad_map: Any, device: str
) -> None:
    """Uploaded, not redrawn: two draws would be two cameras, not two implementations of one."""
    del warp
    state = device_state(device)
    kind, current, _ = state.defect_buffers(bad_map, np.zeros(SHAPE, np.uint8))
    assert np.array_equal(kind.numpy(), np.asarray(bad_map.kind, dtype=np.uint8))
    assert np.array_equal(current.numpy(), np.zeros(SHAPE, np.uint8))


@pytest.mark.parametrize("device", DEVICES)
def test_the_device_buffers_do_not_move_between_frames(
    warp: Any, sensor: Any, bad_map: Any, device: str
) -> None:
    """Cross-frame state stays put: a reallocation per frame would lose the RTS chain's history."""
    del warp
    from irsim_isaac.pipeline.warp_stages import defect_terms, launch_rts_step

    state = device_state(device)
    state.defect_buffers(bad_map, np.zeros(SHAPE, np.uint8))
    terms = defect_terms(sensor.sensor, 65535)
    before = set(state.defect_ptrs or ())
    for frame in range(5):
        launch_rts_step(state, terms, frame, SEED, device)
    assert set(state.defect_ptrs or ()) == before, "the RTS buffers were reallocated"


# --- the replacement --------------------------------------------------------------------------


def cpu_replace(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    from irsim.isp.bad_pixel import replace_bad_pixels

    return np.asarray(replace_bad_pixels(frame, mask), dtype=np.float32)


def device_replace(warp: Any, frame: np.ndarray, mask: np.ndarray, device: str) -> np.ndarray:
    from irsim_isaac.pipeline.warp_stages import launch_replacement

    src = warp.array(np.ascontiguousarray(frame, np.float32), dtype=warp.float32, device=device)
    active = warp.array(
        np.ascontiguousarray(mask.astype(np.uint8)), dtype=warp.uint8, device=device
    )
    out = warp.zeros(frame.shape, dtype=warp.float32, device=device)
    launch_replacement(src, active, out, device)
    return np.asarray(out.numpy(), dtype=np.float32)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("size", [1, 2, 3])
def test_replacement_matches_the_cpu_bit_for_bit_on_a_cluster(
    warp: Any, device: str, size: int
) -> None:
    """The roadmap's criterion, on the clusters that defeat a single 4-neighbour pass.

    Bit-for-bit and not to a tolerance, which is why the device kernel accumulates in float64 the
    way ``replace_bad_pixels`` does: a float32 mean of four neighbours differs in the last bit,
    and a tolerance would then be hiding an arithmetic difference rather than measuring one.
    """
    frame = ramp()
    mask = np.zeros(SHAPE, bool)
    mask[20 : 20 + size, 24 : 24 + size] = True
    cpu = cpu_replace(frame, mask)
    gpu = device_replace(warp, frame, mask, device)
    assert np.array_equal(gpu, cpu), f"{size}x{size} cluster: {np.abs(gpu - cpu).max()}"
    assert not np.array_equal(gpu[mask], frame[mask]), "the cluster should have been replaced"


@pytest.mark.parametrize("device", DEVICES)
def test_replacement_matches_the_cpu_on_the_real_defect_map(
    warp: Any, bad_map: Any, device: str
) -> None:
    """Not a hand-placed cluster: whatever ADR 0055's cluster process actually produced."""
    frame = ramp()
    mask = np.asarray(bad_map.mask)
    assert np.array_equal(device_replace(warp, frame, mask, device), cpu_replace(frame, mask))


@pytest.mark.parametrize("device", DEVICES)
def test_a_cluster_with_no_valid_neighbour_is_refused_on_device_too(warp: Any, device: str) -> None:
    """A whole masked row cannot be interpolated, and the device says so rather than filling it.

    The CPU raises here; a device path that quietly emitted the stuck values instead would put an
    unreplaced defect into the NUC, which is exactly the silent failure the CPU refuses to make.
    """
    frame = ramp()
    mask = np.zeros(SHAPE, bool)
    mask[:, :] = True
    with pytest.raises(ValueError, match="no valid neighbour"):
        device_replace(warp, frame, mask, device)


@pytest.mark.parametrize("device", DEVICES)
def test_an_isolated_defect_takes_one_pass_and_a_cluster_takes_more(warp: Any, device: str) -> None:
    """The pass count is the stencil's reach: a 3x3 centre has no valid neighbour at first."""
    from irsim_isaac.pipeline.warp_stages import launch_replacement

    for size, expected in ((1, 1), (3, 2)):
        mask = np.zeros(SHAPE, bool)
        mask[10 : 10 + size, 10 : 10 + size] = True
        src = warp.array(np.ascontiguousarray(ramp(), np.float32), warp.float32, device=device)
        active = warp.array(
            np.ascontiguousarray(mask.astype(np.uint8)), dtype=warp.uint8, device=device
        )
        out = warp.zeros(SHAPE, dtype=warp.float32, device=device)
        assert launch_replacement(src, active, out, device) == expected, f"{size}x{size}"


# --- the defects ------------------------------------------------------------------------------


@pytest.mark.parametrize("device", DEVICES)
def test_defect_injection_matches_the_cpu_for_a_given_rts_state(
    warp: Any, sensor: Any, bad_map: Any, device: str
) -> None:
    """With the same chain state on both sides the injection is arithmetic, so it is exact.

    This is the comparison that isolates the kernel from the generator: the RTS state is handed to
    both paths rather than drawn on each, so any difference here is the defect model and not the
    random numbers.
    """
    from irsim.noise.defects import DefectState, apply_defects, replacement_mask
    from irsim_isaac.pipeline.warp_stages import defect_terms, launch_defects

    rng = np.random.default_rng(7)
    stateful = np.asarray(bad_map.stateful_mask)
    bad = stateful & (rng.random(SHAPE) < 0.5)
    cpu_state = DefectState(bad=bad)
    dn_max = 65535
    terms = defect_terms(sensor.sensor, dn_max)

    frame = ramp()
    quantised = np.clip(np.floor(frame), 0, dn_max).astype(np.uint16)
    defective = apply_defects(
        quantised, bad_map, cpu_state, dn_max, sensor.sensor.noise.bad_pixel_rts_amplitude_dn
    )
    changed = defective != quantised
    expected = frame.copy()
    expected[changed] = defective[changed].astype(np.float32)
    expected_active = replacement_mask(bad_map, cpu_state)

    state = device_state(device)
    kind, _, _ = state.defect_buffers(bad_map, bad.astype(np.uint8))
    rts = warp.array(np.ascontiguousarray(bad.astype(np.uint8)), dtype=warp.uint8, device=device)
    src = warp.array(np.ascontiguousarray(frame, np.float32), dtype=warp.float32, device=device)
    out = warp.zeros(SHAPE, dtype=warp.float32, device=device)
    active = warp.zeros(SHAPE, dtype=warp.uint8, device=device)
    launch_defects(src, kind, rts, terms, out, active, device)

    assert np.array_equal(out.numpy(), expected)
    assert np.array_equal(active.numpy().astype(bool), expected_active)


@pytest.mark.parametrize("device", DEVICES)
def test_dead_and_hot_pixels_are_pinned_every_frame(
    warp: Any, sensor: Any, bad_map: Any, device: str
) -> None:
    """What makes a static defect findable by a calibration: it never moves."""
    from irsim.noise.defects import DefectKind
    from irsim_isaac.pipeline.warp_stages import defect_terms, launch_defects

    dn_max = 65535
    terms = defect_terms(sensor.sensor, dn_max)
    state = device_state(device)
    kind, _, _ = state.defect_buffers(bad_map, np.zeros(SHAPE, np.uint8))
    rts = warp.zeros(SHAPE, dtype=warp.uint8, device=device)
    src = warp.array(np.ascontiguousarray(ramp(), np.float32), dtype=warp.float32, device=device)
    out = warp.zeros(SHAPE, dtype=warp.float32, device=device)
    active = warp.zeros(SHAPE, dtype=warp.uint8, device=device)
    launch_defects(src, kind, rts, terms, out, active, device)

    result = out.numpy()
    dead = np.asarray(bad_map.mask_of(DefectKind.DEAD))
    hot = np.asarray(bad_map.mask_of(DefectKind.HOT))
    if dead.any():
        assert np.all(result[dead] == 0.0)
    if hot.any():
        assert np.all(result[hot] == float(dn_max))


@pytest.mark.parametrize("device", DEVICES)
def test_the_rts_chain_holds_its_occupancy_and_dwells_geometrically(
    warp: Any, rts_sensor: Any, rts_map: Any, device: str
) -> None:
    """The statistical half: the right fraction bad, and dwells that are not memoryless draws.

    A chain redrawn independently every frame would land on the same occupancy and have a mean
    dwell of one frame. Comparing the measured mean dwell against the configured one is what
    separates a random telegraph signal from a pixel that is merely noisy -- the distinction
    §10.4 is about.
    """
    del warp
    from irsim_isaac.pipeline.warp_stages import defect_terms, launch_rts_step

    noise = rts_sensor.sensor.noise
    stateful = np.asarray(rts_map.stateful_mask)

    state = device_state(device)
    state.defect_buffers(rts_map, np.zeros(RTS_SHAPE, np.uint8))
    terms = defect_terms(rts_sensor.sensor, 65535)

    frames = 2500
    history = np.zeros((frames, int(stateful.sum())), dtype=bool)
    for frame in range(frames):
        rts = launch_rts_step(state, terms, frame, SEED, device)
        history[frame] = rts.numpy().astype(bool)[stateful]

    occupancy = float(history.mean())
    assert occupancy == pytest.approx(float(noise.bad_pixel_rts_occupancy), rel=0.25)

    # Mean dwell in the bad state: total bad frames over the number of times it was entered.
    entered = int(np.count_nonzero(history[1:] & ~history[:-1]))
    assert entered > 20, "not enough transitions to estimate a dwell"
    mean_dwell = float(history[1:].sum()) / entered
    assert mean_dwell == pytest.approx(float(noise.bad_pixel_rts_dwell_frames), rel=0.35)
    assert mean_dwell > 2.0, "a memoryless redraw would give a mean dwell of about one frame"


@pytest.mark.parametrize("device", DEVICES)
def test_a_good_pixel_never_enters_the_chain(
    warp: Any, sensor: Any, bad_map: Any, device: str
) -> None:
    """The chain is launched over the whole plane; only the stateful classes may ever be bad."""
    del warp
    from irsim_isaac.pipeline.warp_stages import defect_terms, launch_rts_step

    state = device_state(device)
    state.defect_buffers(bad_map, np.zeros(SHAPE, np.uint8))
    terms = defect_terms(sensor.sensor, 65535)
    stateful = np.asarray(bad_map.stateful_mask)
    for frame in range(50):
        rts = launch_rts_step(state, terms, frame, SEED, device).numpy().astype(bool)
        assert not np.any(rts & ~stateful), f"frame {frame}: a non-stateful pixel went bad"


# --- the freeze -------------------------------------------------------------------------------


@pytest.mark.parametrize("device", DEVICES)
def test_a_frozen_frame_is_bit_identical_across_the_whole_freeze(warp: Any, device: str) -> None:
    """§11.2: the camera emits the *same* frame for the length of the shutter event.

    Bit-identical rather than similar, because that is the fingerprint ME.3 looks for in real
    video: a run of frames with no temporal noise at all is how an FFC is found in a clip whose
    metadata says nothing.
    """
    from irsim_isaac.pipeline.warp_stages import launch_ffc_hold

    state = device_state(device)
    good = ramp()
    src = warp.array(np.ascontiguousarray(good, np.float32), dtype=warp.float32, device=device)
    out = warp.zeros(SHAPE, dtype=warp.float32, device=device)

    assert launch_ffc_hold(src, state, False, out, device) is False
    assert np.array_equal(out.numpy(), good)

    emitted = []
    for step in range(6):
        moving = warp.array(
            np.ascontiguousarray(good + 50.0 * (step + 1), np.float32),
            dtype=warp.float32,
            device=device,
        )
        assert launch_ffc_hold(moving, state, True, out, device) is True
        emitted.append(out.numpy().copy())
    for frame in emitted:
        assert np.array_equal(frame, good), "a frozen frame must be the held one, unchanged"
    assert all(np.array_equal(frame, emitted[0]) for frame in emitted)


@pytest.mark.parametrize("device", DEVICES)
def test_a_freeze_before_anything_is_held_passes_the_frame_through(warp: Any, device: str) -> None:
    """An FFC on the first frame: emit what the camera saw, never a frame it never saw."""
    from irsim_isaac.pipeline.warp_stages import launch_ffc_hold

    state = device_state(device)
    first = ramp()
    src = warp.array(np.ascontiguousarray(first, np.float32), dtype=warp.float32, device=device)
    out = warp.zeros(SHAPE, dtype=warp.float32, device=device)
    assert launch_ffc_hold(src, state, True, out, device) is False
    assert np.array_equal(out.numpy(), first)


@pytest.mark.parametrize("device", DEVICES)
def test_the_hold_buffer_is_one_buffer(warp: Any, device: str) -> None:
    """Reallocating it per frame would quietly turn a freeze into a pass-through."""
    from irsim_isaac.pipeline.warp_stages import launch_ffc_hold

    state = device_state(device)
    src = warp.array(np.ascontiguousarray(ramp(), np.float32), dtype=warp.float32, device=device)
    out = warp.zeros(SHAPE, dtype=warp.float32, device=device)
    launch_ffc_hold(src, state, False, out, device)
    pointer = state.held_ptr
    for _ in range(4):
        launch_ffc_hold(src, state, False, out, device)
    assert state.held_ptr == pointer


@pytest.mark.parametrize("device", DEVICES)
def test_a_reset_drops_the_held_frame_and_the_defect_state(
    warp: Any, bad_map: Any, device: str
) -> None:
    """A cold start clears all of it together -- the reason there is one owner (ADR 0052)."""
    from irsim_isaac.pipeline.warp_stages import launch_ffc_hold

    state = device_state(device)
    state.defect_buffers(bad_map, np.zeros(SHAPE, np.uint8))
    src = warp.array(np.ascontiguousarray(ramp(), np.float32), dtype=warp.float32, device=device)
    out = warp.zeros(SHAPE, dtype=warp.float32, device=device)
    launch_ffc_hold(src, state, False, out, device)
    assert state.held_ptr is not None and state.defect_ptrs is not None

    state.reset()
    assert state.held_ptr is None and state.defect_ptrs is None


# --- the composition --------------------------------------------------------------------------


@pytest.mark.parametrize("device", DEVICES)
def test_defects_then_replacement_composes_exactly_like_the_cpu_chain(
    warp: Any, sensor: Any, device: str
) -> None:
    """The whole post-ADC half against ``SensorChain.finish_frame``, bit for bit.

    Testing the kernels one at a time is not enough, and M10.7a is why: there the two paths each
    had a correct noise stage and disagreed by a factor of ten because the *seam* between stage 4
    and stage 5 sat in different places. So this compares the composition -- inject, then replace,
    in that order, on the same frame and the same chain state -- rather than the parts.

    The residual is off and the FFC is not firing, so what is left is exactly the two pieces under
    test; the residual already has its own equivalence check in M10.7a.
    """
    from irsim.pipeline.sensor_chain import SensorChain
    from irsim.thermal.weather import WeatherSample, WeatherSeries
    from irsim_isaac.pipeline.warp_stages import (
        defect_terms,
        launch_defects,
        launch_replacement,
    )

    # The Boson's housing node is `coupled`, so the chain wants an ambient source (CLAUDE.md #6).
    # A constant series is the honest minimum here: the housing temperature is not what is under
    # test, and overriding the config's mode instead would test a camera nobody configured.
    weather = WeatherSeries.constant(
        WeatherSample(288.15, 0.4, 1.0, 0.0, 0.0, 0.0, 23000.0, 0.0), 3600.0
    )
    chain = SensorChain.build(
        sensor.sensor,
        dn_per_k=100.0,
        sensor_seed=SEED,
        weather=weather,
        residual_enabled=False,
    )
    if chain.bad_pixels.count == 0:
        pytest.skip("no defects in this chain's map")

    frame = ramp()
    dn_max = 65535
    cpu_out, _ = chain.finish_frame(frame, dn_max, 0, 300.0)

    terms = defect_terms(sensor.sensor, dn_max)
    state = device_state(device)
    kind, _, _ = state.defect_buffers(
        chain.bad_pixels, np.asarray(chain.defect_state.bad, dtype=np.uint8)
    )
    rts = warp.array(
        np.ascontiguousarray(np.asarray(chain.defect_state.bad, dtype=np.uint8)),
        dtype=warp.uint8,
        device=device,
    )
    src = warp.array(np.ascontiguousarray(frame, np.float32), dtype=warp.float32, device=device)
    defective = warp.zeros(SHAPE, dtype=warp.float32, device=device)
    active = warp.zeros(SHAPE, dtype=warp.uint8, device=device)
    launch_defects(src, kind, rts, terms, defective, active, device)
    replaced = warp.zeros(SHAPE, dtype=warp.float32, device=device)
    launch_replacement(defective, active, replaced, device)

    gpu_out = np.asarray(replaced.numpy(), dtype=np.float32)
    assert np.array_equal(gpu_out, np.asarray(cpu_out, dtype=np.float32)), (
        f"max difference {float(np.abs(gpu_out - np.asarray(cpu_out)).max())}"
    )
    assert not np.array_equal(gpu_out, frame), "the chain should have changed something"
