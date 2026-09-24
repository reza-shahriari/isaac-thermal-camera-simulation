"""Roadmap M10.4: the CPU-vs-GPU equivalence harness, parametrised over (stage, fixture, device).

Every Warp stage registers its (CPU oracle, GPU twin) pair in
`irsim_isaac.pipeline.warp_stages.EQUIVALENCE_STAGES`; this file runs both on the same synthetic
G-buffers from tests/conftest.py and demands agreement within the ir-sim-testing budget for a
GPU kernel against its CPU reference: ≤ 1e-4 relative, and ≤ 5 mK when the radiance error is
expressed through dL_B/dT at the pixel's temperature (one tenth of the tightest NETD modelled).
The CPU reference is the oracle (ADR 0018); a disagreement is a kernel bug until proven otherwise.

Nothing here renders, and nothing here needs Kit: `irsim_isaac.env.ensure_warp_on_path` puts the
`omni.warp.core` extension on `sys.path`, so the whole file runs from a bare Isaac Sim interpreter
in seconds (ADR 0014 addendum). `cuda:0` is the production device; Warp's `cpu` device runs the
same kernel source through the C++ backend and is included as a second, compiler-independent
check of the arithmetic.

    make test-all                      # with everything else
    $PYTHON -m pytest tests/integration/test_kernels_vs_reference.py -m gpu
"""

from __future__ import annotations

import pathlib
from typing import Any

import numpy as np
import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
FIXTURES = ("gbuffer_ramp", "gbuffer_sphere", "gbuffer_two_material")
DEVICES = ("cuda:0", "cpu")
#: (stage, fixture) pairs. Stages 1-2 are element-wise and run on any G-buffer; stage 3 consumes
#: the k-x supersampled grid of a specific detector format, so it takes the 4x step edge and the
#: sensor built to match it -- which is also the shape the PSF-then-box check wants (M10.5).
STAGE_FIXTURES = tuple(
    [("band_radiance", f) for f in FIXTURES]
    + [("atmosphere", f) for f in FIXTURES]
    + [("optics", "gbuffer_step_edge"), ("detector", "gbuffer_step_edge")]
)
REL_TOL = 1e-4
MK_TOL = 5.0
pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module")
def warp() -> Any:
    from irsim_isaac.env import ensure_warp_on_path

    ensure_warp_on_path()
    import warp as wp

    wp.init()
    if not any(d.is_cuda for d in wp.get_devices()):
        pytest.skip("no CUDA device for the cuda:0 half of the harness")
    return wp


def _sensor(**fpa: Any) -> Any:
    import yaml

    from irsim.config.sensor import SensorConfig

    raw = yaml.safe_load((REPO / "configs/sensors/flir_boson_640_lwir.yaml").read_text())
    raw["sensor"]["fpa"].update(fpa)
    return SensorConfig.model_validate(raw)


def _weather() -> Any:
    from irsim.thermal import WeatherSample, WeatherSeries

    return WeatherSeries.constant(
        WeatherSample(288.15, 0.46, 1.0, 0.0, 0.0, 0.0, 23000.0, 0.0), 3600.0
    )


@pytest.fixture(scope="module")
def config(tophat_lwir_lut: Any) -> Any:
    from irsim.materials import MaterialTable
    from irsim.pipeline import PipelineConfig

    return PipelineConfig.from_sensor(
        _sensor(), MaterialTable.from_mapping({1: 0.95, 2: 0.6}), lut=tophat_lwir_lut
    )


@pytest.fixture(scope="module")
def config_layered(tophat_lwir_lut: Any) -> Any:
    """Stage 2 with the MS.1 exponential sum -- three terms in LWIR, so the kernel's per-term
    loop is actually exercised rather than degenerating to the grey single-term case."""
    from irsim.atmosphere import load_atmosphere_preset
    from irsim.atmosphere.layered import LayeredAtmosphere
    from irsim.materials import MaterialTable
    from irsim.pipeline import PipelineConfig

    atmosphere = LayeredAtmosphere(
        load_atmosphere_preset("us_standard_clear"), _weather(), {"lwir": tophat_lwir_lut}
    )
    return PipelineConfig.from_sensor(
        _sensor(),
        MaterialTable.from_mapping({1: 0.95, 2: 0.6}),
        lut=tophat_lwir_lut,
        atmosphere=atmosphere,
    )


@pytest.fixture(scope="module")
def config_grey(tophat_lwir_lut: Any) -> Any:
    from irsim.atmosphere import Atmosphere, load_atmosphere_preset
    from irsim.materials import MaterialTable
    from irsim.pipeline import PipelineConfig

    return PipelineConfig.from_sensor(
        _sensor(),
        MaterialTable.from_mapping({1: 0.95, 2: 0.6}),
        lut=tophat_lwir_lut,
        atmosphere=Atmosphere(load_atmosphere_preset("us_standard_clear"), _weather()),
    )


@pytest.fixture(scope="module")
def config_edge(tophat_lwir_lut: Any) -> Any:
    """A 256x256 detector at 4x: the format `gbuffer_step_edge` renders (1024x1024)."""
    from irsim.materials import MaterialTable
    from irsim.pipeline import PipelineConfig

    return PipelineConfig.from_sensor(
        _sensor(width=256, height=256),
        MaterialTable.from_mapping({1: 0.95, 2: 0.6}),
        lut=tophat_lwir_lut,
    )


@pytest.fixture(scope="module")
def config_photon(tophat_mwir_lut: Any) -> Any:
    """A cooled MWIR photon FPA: the other half of stage 4, with no membrane and no state."""
    import copy

    import yaml

    from irsim.config.sensor import SensorConfig
    from irsim.materials import MaterialTable
    from irsim.pipeline import PipelineConfig

    raw = copy.deepcopy(
        yaml.safe_load((REPO / "configs/sensors/flir_boson_640_lwir.yaml").read_text())
    )
    raw["sensor"]["band"].update(lambda_min_um=3.0, lambda_max_um=5.0, regime="mixed")
    raw["sensor"]["fpa"] = {
        "type": "photon",
        "width": 32,
        "height": 32,
        "pitch_um": 15.0,
        "fill_factor": 1.0,
        "frame_rate_hz": 30,
        "bit_depth": 14,
        "quantum_efficiency": 0.8,
        "well_capacity_e": 1.0e6,
        "integration_time_ms": 5.0,
        "dark_current_model": "arrhenius",
    }
    return PipelineConfig.from_sensor(
        SensorConfig.model_validate(raw),
        MaterialTable.from_mapping({1: 0.95, 2: 0.6}),
        lut=tophat_mwir_lut,
    )


STAGE_CONFIG = {
    "band_radiance": "config",
    "atmosphere": "config_layered",
    "optics": "config_edge",
    "detector": "config_edge",
}


def _input_planes(stage: str, planes: dict[str, np.ndarray], config: Any) -> dict[str, np.ndarray]:
    """The planes a stage consumes: everything before it, run on the CPU oracle."""
    from irsim.pipeline import PipelineState
    from irsim.pipeline.atmosphere import atmosphere_stage
    from irsim.pipeline.optics import optics_stage
    from irsim.pipeline.radiance import band_radiance_stage

    out = dict(planes)
    if stage == "band_radiance":
        return out
    out.update(band_radiance_stage(out, config, PipelineState()))
    if stage == "atmosphere":
        return out
    out.update(atmosphere_stage(out, config, PipelineState()))
    if stage == "optics":
        return out
    out.update(optics_stage(out, config, PipelineState()))
    return out


def _error_mk(cpu: np.ndarray, gpu: np.ndarray, t: np.ndarray, lut: Any) -> np.ndarray:
    slope = lut.lookup(t, "dlb_dt").astype(np.float64)
    return np.abs(gpu.astype(np.float64) - cpu.astype(np.float64)) / slope * 1e3


def _scene_equivalent(plane: str, values: np.ndarray, config: Any, state: Any) -> np.ndarray:
    """Whatever a stage produces, expressed as scene band radiance, so one mK budget covers every
    stage. The oracle's own inverses are used: `BolometerTransfer.power_from_signal_w` back to
    pixel power and `invert_optics` back to the radiance the pixel saw."""
    if plane == "radiance":
        return values
    from irsim.optics.stage import invert_optics
    from irsim.pipeline.optics import housing_band_radiance

    phi = values
    if plane == "signal_dn":
        phi = config.detector.transfer.power_from_signal_w(values)
    return invert_optics(
        np.asarray(phi, np.float32), config.sensor.sensor, housing_band_radiance(config, state)
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(("stage", "fixture"), STAGE_FIXTURES)
def test_stage_matches_cpu_reference(
    warp: Any, request: pytest.FixtureRequest, stage: str, fixture: str, device: str
) -> None:
    from irsim.optics.sampling import box_downsample
    from irsim.pipeline import PipelineState
    from irsim_isaac.pipeline.warp_stages import EQUIVALENCE_OUTPUT, EQUIVALENCE_STAGES

    config = request.getfixturevalue(STAGE_CONFIG[stage])
    planes = _input_planes(stage, request.getfixturevalue(fixture), config)
    plane = EQUIVALENCE_OUTPUT[stage]
    cpu_stage, gpu_stage = EQUIVALENCE_STAGES[stage]
    state = PipelineState()
    ref = cpu_stage(planes, config, PipelineState())[plane]
    got = gpu_stage(planes, config, PipelineState(), device)[plane]
    assert got.dtype == np.float32 and got.shape == ref.shape

    rel = np.max(np.abs(got - ref) / np.abs(ref))
    # the temperature at the grid the stage's output lives on
    t = np.asarray(planes["temperature_k"])
    if t.shape != ref.shape:
        t = box_downsample(t, config.supersample)
    mk = np.max(
        _error_mk(
            _scene_equivalent(plane, ref, config, state),
            _scene_equivalent(plane, got, config, state),
            t,
            config.lut,
        )
    )
    assert rel <= REL_TOL, f"{stage}/{fixture}/{device}: {rel:.2e} relative"
    assert mk <= MK_TOL, f"{stage}/{fixture}/{device}: {mk:.3f} mK"


def test_sky_mask_and_environment_term_match(
    warp: Any, config: Any, gbuffer_ramp: dict[str, np.ndarray]
) -> None:
    """ε = 1 under the mask, ids ignored there, and the reflected term blended like the CPU."""
    from irsim.pipeline.radiance import band_radiance
    from irsim_isaac.pipeline.warp_stages import band_radiance_warp

    t = gbuffer_ramp["temperature_k"]
    ids = gbuffer_ramp["material_id"].copy()
    sky = np.zeros(t.shape, dtype=bool)
    sky[:, :32] = True
    ids[sky] = 0  # the renderer's background id: an error outside the mask, ignored under it
    l_env = (config.lut.lookup(np.full(t.shape, 280.0, np.float32)) * 0.7).astype(np.float32)
    ref = band_radiance(t, ids, config.materials, config.lut, sky_mask=sky, l_env=l_env)
    got = band_radiance_warp(t, ids, config.materials, config.lut, sky_mask=sky, l_env=l_env)
    assert np.max(np.abs(got - ref) / np.abs(ref)) <= REL_TOL
    assert np.array_equal(got[sky], got[sky])  # finite everywhere under the mask
    assert np.all(np.isfinite(got))


def test_isothermal_enclosure_identity_on_gpu(
    warp: Any, config: Any, gbuffer_two_material: dict[str, np.ndarray]
) -> None:
    """Kirchhoff on the device: with L_env = L_B(T) the radiance is L_B(T) for every ε (1e-5)."""
    from irsim_isaac.pipeline.warp_stages import band_radiance_warp

    t = gbuffer_two_material["temperature_k"]
    lb = config.lut.lookup(t)
    got = band_radiance_warp(
        t, gbuffer_two_material["material_id"], config.materials, config.lut, l_env=lb
    )
    assert np.max(np.abs(got - lb) / lb) <= 1e-5


def test_lut_device_pointer_is_stable_across_frames(
    warp: Any, config: Any, gbuffer_ramp: dict[str, np.ndarray]
) -> None:
    from irsim.pipeline import PipelineState
    from irsim_isaac.pipeline.warp_stages import band_radiance_stage_warp, device_tables

    tables = device_tables(config.lut, config.materials, config.quantity)
    ptr = tables.lut_ptr
    state = PipelineState()
    for _ in range(10):
        band_radiance_stage_warp(gbuffer_ramp, config, state)
        state.advance()
    again = device_tables(config.lut, config.materials, config.quantity)
    assert again is tables and again.lut_ptr == ptr


def test_gpu_refuses_what_the_oracle_refuses(warp: Any, config: Any) -> None:
    from irsim_isaac.pipeline.warp_stages import band_radiance_warp

    t = np.full((4, 4), 300.0, np.float32)
    with pytest.raises(ValueError, match="UNMAPPED"):
        band_radiance_warp(t, np.zeros((4, 4), np.int32), config.materials, config.lut)
    with pytest.raises(TypeError):
        band_radiance_warp(
            t.astype(np.float16), np.ones((4, 4), np.int32), config.materials, config.lut
        )


# ---- M10.5: stage 2 (atmosphere) and stage 3 (optics) ----------------------------------------


@pytest.mark.parametrize("device", DEVICES)
def test_sky_pixels_bypass_the_atmosphere_bit_identically(
    warp: Any, config_layered: Any, gbuffer_ramp: dict[str, np.ndarray], device: str
) -> None:
    """ADR 0050: a sky pixel already carries the whole column to space, so stage 2 must not
    attenuate it again. Bit-identical, not merely close -- the CPU returns the input array."""
    from irsim.pipeline import PipelineState
    from irsim_isaac.pipeline.warp_stages import atmosphere_stage_warp

    planes = _input_planes("atmosphere", gbuffer_ramp, config_layered)
    sky = np.zeros(planes["radiance"].shape, dtype=bool)
    sky[:, :32] = True
    distance = np.asarray(planes["distance_m"]).copy()
    distance[sky] = np.inf  # the DistanceToCamera sentinel a sky ray comes back with
    planes = {**planes, "sky_mask": sky, "distance_m": distance}
    got = atmosphere_stage_warp(planes, config_layered, PipelineState(), device)["radiance"]
    assert np.array_equal(got[sky], planes["radiance"][sky])
    assert np.all(got[~sky] != planes["radiance"][~sky]), "the rest must have been attenuated"


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("which", ["grey", "tau_override", "layered"])
def test_every_atmosphere_path_matches_its_oracle(
    warp: Any,
    request: pytest.FixtureRequest,
    gbuffer_ramp: dict[str, np.ndarray],
    which: str,
    device: str,
) -> None:
    """The three branches `atmosphere_stage` can take: one grey term, the constant-tau L1
    fallback, and MS.1's multi-term sum. A single kernel serves all three because Sum w_k = 1
    makes the per-term path radiance collapse to (1 - tau) L_air."""
    import dataclasses

    from irsim.pipeline import PipelineState
    from irsim.pipeline.atmosphere import atmosphere_stage
    from irsim_isaac.pipeline.warp_stages import atmosphere_stage_warp, atmosphere_terms

    config = request.getfixturevalue("config_layered" if which == "layered" else "config_grey")
    if which == "tau_override":
        config = dataclasses.replace(config, tau_override=0.7)
    planes = _input_planes("atmosphere", gbuffer_ramp, config)
    # a range spread that reaches optical depths where tau is small, plus the degenerate ends
    distance = np.tile(
        np.linspace(0.0, 5000.0, planes["radiance"].shape[1], dtype=np.float32),
        (planes["radiance"].shape[0], 1),
    )
    distance[0, 0] = np.inf
    planes = {**planes, "distance_m": distance}
    ref = atmosphere_stage(planes, config, PipelineState())["radiance"]
    got = atmosphere_stage_warp(planes, config, PipelineState(), device)["radiance"]
    assert np.max(np.abs(got - ref) / np.abs(ref)) <= REL_TOL
    terms = atmosphere_terms(config, PipelineState())
    assert terms is not None
    expected_terms = 3 if which == "layered" else 1
    assert terms.n_terms == expected_terms, f"{which}: {terms.n_terms} terms"


@pytest.mark.parametrize("device", DEVICES)
def test_step_edge_psf_then_box_matches_the_cpu_to_1e5(
    warp: Any, config_edge: Any, gbuffer_step_edge: dict[str, np.ndarray], device: str
) -> None:
    """M10.5's tight case: the optical PSF on the 4x grid *then* the block mean, in that order,
    across a 373/293 K edge where the blur has the most to do. A GPU path that downsampled first
    would still look plausible and would lose the sub-pixel edge profile MS.5 measures."""
    from irsim.pipeline import PipelineState
    from irsim.pipeline.optics import optics_stage
    from irsim_isaac.pipeline.warp_stages import optics_stage_warp

    assert config_edge.psf is not None and config_edge.psf.shape[0] >= 5
    planes = _input_planes("optics", gbuffer_step_edge, config_edge)
    ref = optics_stage(planes, config_edge, PipelineState())["flux"]
    got = optics_stage_warp(planes, config_edge, PipelineState(), device)["flux"]
    assert np.max(np.abs(got - ref) / np.abs(ref)) <= 1e-5


@pytest.mark.parametrize("device", DEVICES)
def test_vignetting_and_self_emission_come_from_the_host(
    warp: Any, config_edge: Any, device: str
) -> None:
    """cos^4 across the field and Phi_self at the corner, against `irsim.optics` directly: the
    kernel may scale and add, it may not own the aperture factor (non-negotiable #5)."""
    from irsim.optics.aperture import aperture_factor
    from irsim.optics.self_emission import self_emission_power
    from irsim.optics.stage import optics_field
    from irsim_isaac.pipeline.warp_stages import apply_optics_warp, optics_terms

    spec = config_edge.sensor.sensor
    lb_housing = float(config_edge.lut.lookup(300.0, config_edge.quantity)[()])
    terms = optics_terms(spec, lb_housing, config_edge.supersample, None)
    k, (h, w) = config_edge.supersample, spec.fpa_shape
    flux = apply_optics_warp(np.ones((h * k, w * k), np.float32), terms, device=device)

    f, tau, a_d = spec.optics.f_number, spec.optics.transmittance, spec.detector_active_area_m2
    phi_self = self_emission_power(a_d, f, tau, lb_housing)
    expected = aperture_factor(f) * tau * optics_field(spec) * a_d + phi_self
    assert np.max(np.abs(flux - expected) / expected) <= 1e-6
    corner = float(flux[0, 0] - phi_self) / float(flux[h // 2, w // 2] - phi_self)
    assert corner == pytest.approx(float(optics_field(spec)[0, 0]), rel=1e-6)


@pytest.mark.parametrize("device", DEVICES)
def test_a_warmer_housing_raises_apparent_temperature_exactly_as_the_cpu_does(
    warp: Any, config_edge: Any, device: str
) -> None:
    """Phi_self is why a shutterless camera drifts (§8.2, §11.2). A 1 K housing step must move
    the apparent temperature by the same amount on both paths, or the GPU drift is not the
    CPU's drift and every NUC/FFC test written against the oracle stops meaning anything."""
    from irsim.isp.radiometric import apparent_temperature
    from irsim.optics.stage import invert_optics
    from irsim.pipeline import PipelineState
    from irsim.pipeline.optics import housing_band_radiance, optics_stage
    from irsim_isaac.pipeline.warp_stages import optics_stage_warp

    spec = config_edge.sensor.sensor
    k, (h, w) = config_edge.supersample, spec.fpa_shape
    scene = np.full((h * k, w * k), float(config_edge.lut.lookup(300.0, config_edge.quantity)[()]))
    planes = {"radiance": scene.astype(np.float32)}
    cal = housing_band_radiance(config_edge, PipelineState(housing_temp_k=300.0))

    def t_app(stage: Any, t_housing: float) -> np.ndarray:
        state = PipelineState(housing_temp_k=t_housing)
        args = (
            (planes, config_edge, state)
            if stage is optics_stage
            else (
                planes,
                config_edge,
                state,
                device,
            )
        )
        flux = stage(*args)["flux"]
        return apparent_temperature(
            invert_optics(flux, spec, cal), config_edge.lut, config_edge.quantity
        )

    cpu_shift = t_app(optics_stage, 301.0) - t_app(optics_stage, 300.0)
    gpu_shift = t_app(optics_stage_warp, 301.0) - t_app(optics_stage_warp, 300.0)
    assert np.max(np.abs(gpu_shift - cpu_shift)) * 1e3 <= 1.0, "within 1 mK of the CPU drift"
    # M10.5's number, and a physics check rather than a tautology: with tau_opt = 0.92 the lens
    # is a grey body of emissivity 0.08 filling the same cone, so ~8 % of a housing step arrives
    # at the detector. A sign error or a missing (1 - tau_opt) would still pass the CPU-vs-GPU
    # comparison above and would fail here.
    assert float(np.median(gpu_shift)) * 1e3 == pytest.approx(87.0, abs=2.0)


# ---- M10.6: stage 4 (detector) and the cross-frame state --------------------------------------


def _step_flux(config: Any, t_cold: float, t_hot: float) -> tuple[np.ndarray, np.ndarray]:
    """Pixel power for two uniform scenes, through the oracle's own stage 3."""
    from irsim.optics.stage import apply_optics

    spec = config.sensor.sensor
    k, (h, w) = config.supersample, spec.fpa_shape
    lb_housing = float(config.lut.lookup(300.0, config.quantity)[()])

    def flux(t_k: float) -> np.ndarray:
        level = float(config.lut.lookup(np.float32(t_k), config.quantity)[()])
        plane = np.full((h * k, w * k), level, np.float32)
        return apply_optics(plane, spec, lb_housing, supersample=k, psf=None)

    return flux(t_cold), flux(t_hot)


@pytest.mark.parametrize("device", DEVICES)
def test_twenty_frame_step_tracks_the_cpu_iir(warp: Any, config_edge: Any, device: str) -> None:
    """The membrane lag is the one stage with memory, so an agreement measured on a single frame
    proves nothing: the CPU adopts its input on frame 1 and so does the GPU. Drive a flux step
    through twenty frames, where a wrong alpha, a stale state or a state reset each show up as a
    growing divergence rather than a constant offset."""
    from irsim.pipeline import PipelineState
    from irsim.pipeline.detector import detector_stage
    from irsim_isaac.pipeline.warp_stages import detector_stage_warp

    cold, hot = _step_flux(config_edge, 295.0, 320.0)
    cpu_state, gpu_state = PipelineState(), PipelineState()
    worst = 0.0
    for n in range(20):
        planes = {"flux": cold if n < 5 else hot}
        ref = detector_stage(planes, config_edge, cpu_state)["signal_dn"]
        got = detector_stage_warp(planes, config_edge, gpu_state, device)["signal_dn"]
        worst = max(worst, float(np.max(np.abs(got - ref) / np.abs(ref))))
    assert worst <= 1e-5, f"{worst:.2e} relative over 20 frames"


@pytest.mark.parametrize("device", DEVICES)
def test_the_first_frame_of_a_step_covers_alpha_of_it(
    warp: Any, config_edge: Any, device: str
) -> None:
    """1 - e^{-dt/tau} = 0.8755 at 60 Hz and tau_th = 8 ms. §9.2's "smears over roughly 0.6
    frames" is tau/dt, not the extent of the smear (spec issue S8) -- one frame already covers
    88 % of a step, and this pins the number the kernel actually applies.

    **The constant was 0.811 until IG.2 first ran this suite.** That is the same expression at
    tau = 10 ms, and `SC.3` corrected tau to 8 ms against [R24]'s published figure. The kernel
    followed; this assertion did not, because nothing had executed it. It is the clearest
    argument in the repository for running the in-sim suite on a schedule rather than when a
    step happens to need it."""
    from irsim.detector.lowpass import alpha_for
    from irsim.pipeline import PipelineState
    from irsim_isaac.pipeline.warp_stages import detector_stage_warp

    fpa = config_edge.fpa
    alpha = alpha_for(fpa.frame_dt_s, fpa.thermal_time_constant_s)
    assert alpha == pytest.approx(0.8755, abs=0.002)

    cold, hot = _step_flux(config_edge, 295.0, 320.0)
    state = PipelineState()
    settled = detector_stage_warp({"flux": cold}, config_edge, state, device)["signal_dn"]
    first = detector_stage_warp({"flux": hot}, config_edge, state, device)["signal_dn"]
    target = detector_stage_warp({"flux": hot}, config_edge, PipelineState(), device)["signal_dn"]
    fraction = float(np.mean((first - settled) / (target - settled)))
    assert fraction == pytest.approx(alpha, abs=0.002)


@pytest.mark.parametrize("device", DEVICES)
def test_the_iir_state_is_allocated_once_and_never_leaves_the_device(
    warp: Any, config_edge: Any, device: str
) -> None:
    """Re-allocating the state per frame would be a silent correctness bug (a fresh buffer adopts
    its input, so the lag would vanish and every frame would look ideal) as well as a performance
    one, and round-tripping it to the host would defeat the point of keeping the chain on device."""
    from irsim.pipeline import PipelineState
    from irsim_isaac.pipeline.warp_stages import (
        DEVICE_STATE_KEY,
        detector_stage_warp,
        warp_pipeline_state,
    )

    cold, hot = _step_flux(config_edge, 295.0, 320.0)
    state = PipelineState()
    pointers = set()
    for n in range(10):
        detector_stage_warp({"flux": cold if n < 3 else hot}, config_edge, state, device)
        pointers.add(warp_pipeline_state(state, device).iir_ptr)
    assert len(pointers) == 1 and None not in pointers, pointers

    device_state = state.buffers[DEVICE_STATE_KEY]
    assert device_state.device == device, "the pipeline state owns it (ADR 0052)"
    device_state.reset()
    assert device_state.iir_ptr is None, "reset() must give a genuine cold start"


@pytest.mark.parametrize("device", DEVICES)
def test_the_photon_path_has_no_memory_and_allocates_no_state(
    warp: Any, config_photon: Any, device: str
) -> None:
    """Cooled photon detectors are memoryless on these timescales (ADR 0052) -- the Tier 3
    phenomenology that LWIR smears and cooled MWIR does not. Two identical frames must give a
    bit-identical answer and no IIR buffer may exist at all."""
    from irsim.pipeline import PipelineState
    from irsim_isaac.pipeline.warp_stages import detector_stage_warp, warp_pipeline_state

    spec = config_photon.sensor.sensor
    h, w = spec.fpa_shape
    phi = np.full((h, w), 1.0e8, np.float32)
    state = PipelineState()
    first = detector_stage_warp({"flux": phi}, config_photon, state, device)["signal_dn"]
    second = detector_stage_warp({"flux": phi}, config_photon, state, device)["signal_dn"]
    assert np.array_equal(first, second)
    assert warp_pipeline_state(state, device).iir_ptr is None


@pytest.mark.parametrize("device", DEVICES)
def test_photon_dn_saturates_and_floors_exactly_like_the_cpu(
    warp: Any, config_photon: Any, device: str
) -> None:
    """DN = clip(floor(S), 0, 2^bits - 1). Floor rather than round is what makes the quantisation
    error uniform on [0, 1) LSB (ADR 0019), and saturation must clip, never wrap -- a wrapped
    well would turn the brightest part of the scene into the darkest."""
    from irsim.detector.quantise import quantise
    from irsim.pipeline import PipelineState
    from irsim.pipeline.detector import detector_stage
    from irsim_isaac.pipeline.warp_stages import detector_stage_warp, quantise_warp

    spec = config_photon.sensor.sensor
    fpa = config_photon.fpa
    h, w = spec.fpa_shape
    saturating = fpa.well_capacity_e / (fpa.quantum_efficiency * fpa.integration_time_s)
    phi = np.linspace(0.0, 2.0 * saturating, h * w, dtype=np.float32).reshape(h, w)

    ref = detector_stage({"flux": phi}, config_photon, PipelineState())["signal_dn"]
    got = detector_stage_warp({"flux": phi}, config_photon, PipelineState(), device)["signal_dn"]
    dn_ref = quantise(ref, fpa.bit_depth)
    dn_got = quantise_warp(got, fpa.bit_depth, device=device)
    assert np.array_equal(dn_ref, dn_got), "DN must match code for code across the whole sweep"
    assert dn_got.max() == fpa.dn_max and (dn_got == fpa.dn_max).sum() > h * w // 4
    assert dn_got.min() == 0 and dn_got.dtype == np.uint16
