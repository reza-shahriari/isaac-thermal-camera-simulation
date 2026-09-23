"""``run_frame``: the whole CPU reference chain for one frame (docs/physics-model.md §13.4).

    stage 1  band radiance     ε₀ L_B(T) + (1−ε₀) L_env on the k× G-buffer (irsim.pipeline.radiance)
    stage 2  atmosphere        τL + (1−τ)L_B(T_air) on the k× grid   (irsim.pipeline.atmosphere)
    stage 2e gain state        min(L, L_B(T_ceiling)) in radiance      (irsim.detector.gain_state)
    stage 3  optics            PSF, box ↓k, aperture·cos⁴·A_d, +Φ_self (irsim.optics.stage)
    stage 4  detector          Φ → signal in DN with per-pixel noise    (irsim.detector, ADR 0026)
    stage 5  noise             + correlated 3-D components              (irsim.noise.stage)
    ADC      quantise          floor + clip → uint16                    (irsim.detector.quantise)
    stage 6  ISP               radiometric branch → radiance, T_app     (irsim.isp.radiometric)
             display branch    AGC → gamma → DDE → polarity → palette  (irsim.isp.display)

Outputs follow §12.2 ``outputs``: ``radiance`` (float32, scene band radiance at the native grid),
``apparent_t`` (float32 K, from the float32 signal route), ``dn16`` (uint16), ``display8``
(RGBA8 through the isp block, ADR 0031). A flag set to ``false`` yields ``None`` -- never zeros.
The bolometer path uses the energy-form LUT table; a photon FPA runs the same chain on the
photon table with N_e = η t_int Φ_q (ADR 0021).

docs/physics-model.md §13.4, §12.2 outputs, §16.4 step 3
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from irsim.atmosphere.layered import LayeredAtmosphere
from irsim.detector.bolometer import MicrobolometerDetector
from irsim.detector.gain_state import clip_to_gain_ceiling, gain_ceiling_radiance
from irsim.detector.params import BolometerParams, PhotonParams
from irsim.detector.quantise import dn_max_for_bits, quantise
from irsim.isp.display import run_display_branch
from irsim.isp.radiometric import apparent_temperature
from irsim.optics.defocus import scene_defocus_um
from irsim.optics.projection import Intrinsics
from irsim.optics.stage import apply_optics, invert_optics
from irsim.pipeline.atmosphere import apply_atmosphere_gbuffer, apply_layered_gbuffer
from irsim.pipeline.core import PipelineConfig, PipelineState, Planes
from irsim.pipeline.detector import bolometer_lag, lag_interval_s
from irsim.pipeline.plume import ExhaustPlume, inject_plumes
from irsim.pipeline.point_target import PointTarget, inject_point_targets
from irsim.pipeline.radiance import band_radiance, stage_illumination
from irsim.pipeline.rotor_veil import RotorVeil, inject_rotor_veils

__all__ = ["Outputs", "run_frame"]


@dataclass(frozen=True)
class Outputs:
    radiance: NDArray[np.float32] | None
    apparent_t: NDArray[np.float32] | None
    dn16: NDArray[np.uint16] | None
    display8: NDArray[np.uint8] | None  # (H, W, 4) RGBA8
    signal_dn: NDArray[np.float32]  # un-quantised stage-4 signal, always kept for benches
    flux: NDArray[np.float32]  # stage-3 pixel power (W or photons/s), always kept
    isp_hash: str | None = None  # config hash of the isp block that produced display8
    report: Any = None  # FrameReport from the M9 chain (M9.8); None when no chain is attached


def _detector_signal(
    flux: NDArray[np.float32], config: PipelineConfig, state: PipelineState
) -> NDArray[np.float32]:
    """Stage 4 + 5: the membrane lag, the detector's per-pixel noise, then the correlated 3-D
    noise; the ideal chain when noise is disabled. Seeded by the sensor's own frame index
    (ADR 0022).

    **A bolometer's signal is lagged before any noise is added** (§9.2, ADR 0052). ADR 0052
    quantifies why the order matters: filtering *after* noise would cut the per-frame temporal
    variance by α/(2 − α) ≈ 0.68 and silently break the M4.6 NETD anchor. A photon detector has no
    membrane and takes the flux straight, which is §15 Tier 3's "lateral motion smears LWIR, not
    cooled MWIR" in its across-frame half.

    With an M9 chain attached the fixed pattern stage 5 adds is the *breathing* one (M9.4), so it
    is handed to the stage each frame rather than left to the stage's own static copy."""
    detector = config.detector
    if isinstance(config.fpa, BolometerParams):
        # The seam is the bolometer's alone and the protocol does not carry it: a photon
        # detector's noise is Poisson in electron space (CLAUDE.md #3), so there is no signal-DN
        # point to insert anything at, and it has no membrane to insert.
        assert isinstance(detector, MicrobolometerDetector), "a bolometer FPA needs its detector"
        signal = bolometer_lag(
            detector.noiseless_signal_dn(flux), config, state, lag_interval_s(config, state)
        )
        if not config.noise_enabled:
            return signal
        frame = detector.frame_from_signal(signal, state.frame_index, config.sensor_seed)
    else:
        if not config.noise_enabled:
            return detector.noiseless_signal_dn(flux)
        frame = detector.response(flux, state.frame_index, config.sensor_seed)
    fixed = None if config.chain is None else config.chain.fixed_pattern
    return config.noise.apply(
        frame.signal_dn, frame.sigma_dn, state.frame_index, fixed_override=fixed
    )


def _scene_radiance_from_signal(
    signal: NDArray[np.float32], config: PipelineConfig, lb_housing_cal: float
) -> NDArray[np.float32]:
    fpa = config.fpa
    if isinstance(fpa, BolometerParams):
        assert config.calibration is not None
        return config.calibration.radiance_from_signal(signal)
    assert isinstance(fpa, PhotonParams)
    n_e = signal.astype(np.float64) / 2**fpa.bit_depth * fpa.well_capacity_e
    phi_q = n_e / (fpa.quantum_efficiency * fpa.integration_time_s)
    return invert_optics(phi_q, config.sensor.sensor, lb_housing_cal)


def run_frame(
    planes: Planes,
    config: PipelineConfig,
    state: PipelineState,
    point_targets: Sequence[PointTarget] = (),
    rotor_veils: Sequence[RotorVeil] = (),
    plumes: Sequence[ExhaustPlume] = (),
) -> Outputs:
    """One frame through stages 1-6 (stage 2 is the identity without an Atmosphere).
    Advances ``state.frame_index``."""
    sensor = config.sensor.sensor
    lut = config.lut
    q = config.quantity
    h, w = sensor.fpa_shape
    k = config.supersample
    t = np.asarray(planes["temperature_k"])
    if t.shape != (h * k, w * k):
        raise ValueError(
            f"G-buffer {t.shape} is not the {k}x supersampled detector grid {(h * k, w * k)}; "
            "set optics.supersample_factor to match the render"
        )

    # stage 1 (k× grid): emission, always, plus whatever the band's regime lets illuminate it
    radiance_ss = band_radiance(
        t,
        planes["material_id"],
        config.materials,
        lut,
        q,
        sky_mask=planes.get("sky_mask"),
        l_behind=planes.get("radiance_behind"),
        illumination=stage_illumination(planes, config, state),
        normal_dot_view=planes.get("normal_dot_view"),
    )
    # stage 2 (k× grid): per-ray atmosphere; sky pixels pass through (ADR 0050)
    if isinstance(config.atmosphere, LayeredAtmosphere):
        radiance_ss = apply_layered_gbuffer(
            config.atmosphere,
            sensor.band.band_id,
            state.t_s,
            radiance_ss,
            np.asarray(planes["distance_m"]),
            q,
            sky_mask=planes.get("sky_mask"),
        )
    elif config.atmosphere is not None:
        atm_state = config.atmosphere.state(state.t_s)
        l_air = float(lut.lookup(np.float64(atm_state.t_air_k), q)[()])
        radiance_ss = apply_atmosphere_gbuffer(
            radiance_ss,
            planes["distance_m"],
            atm_state.gamma_per_m[sensor.band.band_id],
            l_air,
            sky_mask=planes.get("sky_mask"),
            tau_override=config.tau_override,
        )
    # stage 2b: analytic point targets below one native pixel (MS.6), before the PSF and box
    if point_targets:
        radiance_ss = inject_point_targets(
            radiance_ss,
            point_targets,
            sensor,
            config.atmosphere,
            sensor.band.band_id,
            state.t_s,
            lut,
            q,
            k,
        )
    # stage 2c: rotor discs as time-averaged veils (ADR 0081). After the point targets because a
    # blade can pass in front of one, and before the PSF for the same reason stage 2b is.
    if rotor_veils:
        radiance_ss = inject_rotor_veils(
            radiance_ss, rotor_veils, config.atmosphere, sensor.band.band_id, state.t_s, lut, q
        )
    # stage 2d: exhaust plumes as per-pixel gas slabs (`PH.6`). Last of the stage-2 overlays: a
    # plume is semi-transparent, so what it multiplies has to be everything already standing
    # behind it -- the scene, a point target through it, a rotor blade in it.
    if plumes:
        if config.response is None:
            raise ValueError(
                "a plume needs the camera's R(λ) to integrate B_b(T_g) above the LUT's 1000 K "
                "ceiling; build the config with PipelineConfig.from_sensor so it loads one"
            )
        radiance_ss = inject_plumes(
            radiance_ss,
            plumes,
            Intrinsics.from_sensor(sensor, k),
            sensor.optics.distortion,
            config.response,
            config.gas_tables,
            planes.get("distance_m"),
            config.atmosphere,
            sensor.band.band_id,
            state.t_s,
            lut,
            q,
        )
    # stage 2e: the gain state's intrascene ceiling (`PH.8`). Last thing in scene-radiance units
    # and the first place a fire stops being radiometry and starts being a camera problem. In
    # radiance and never in kelvin: what arrives is ε L_B(T) + (1 − ε) L_env plus the path, which
    # is not L_B of anything, so a ceiling on temperature would rail a low-emissivity flame that
    # does not rail and miss a cold reflector that does (ADR 0116).
    ceiling_k = sensor.fpa.gain_ceiling_k
    if ceiling_k is not None:
        radiance_ss = clip_to_gain_ceiling(radiance_ss, gain_ceiling_radiance(ceiling_k, lut, q))
    # The M9 chain owns the thermal nodes and the drift, so it clocks first: stage 3 needs the
    # housing temperature it produces (M9.3), and stage 5 needs the pattern it advanced (M9.4).
    t_fpa_k = state.housing_temp_k
    if config.chain is not None:
        housing_k, t_fpa_k = config.chain.begin_frame(state.t_s, state.frame_index)
        state.housing_temp_k = housing_k
    # stage 3
    lb_housing_now = float(lut.lookup(state.housing_temp_k, q)[()])
    # `OC.5`: one kernel for the frame, chosen from the median range of the geometry actually in
    # it. `config.defocus_bank` is None unless the camera names a defocus model, and then this is
    # `config.psf` exactly as before. The G-buffer's `distance_m` is already on the k× grid -- the
    # same grid the convolution runs on -- because the render product is created supersampled.
    psf = config.psf
    if config.defocus_bank is not None:
        state.defocus_w020_um = scene_defocus_um(
            planes["distance_m"],
            sensor.optics.focal_length_mm,
            sensor.optics.f_number,
            config.focus_distance_m,
            planes.get("sky_mask"),
        )
        psf = config.defocus_bank.kernel_for(state.defocus_w020_um)
    flux = apply_optics(radiance_ss, sensor, lb_housing_now, supersample=k, psf=psf)
    # stages 4-5 (detector noise, correlated noise)
    signal = _detector_signal(flux, config, state)
    # §11.1's post-ADC half, when a chain is attached: defects, replacement, the NUC residual, the
    # temporal filter (identity, ADR 0058) and the FFC freeze. Everything downstream -- including
    # the radiometric branch -- reads the corrected plane, so a frozen frame is stale in every
    # output at once rather than only in the picture.
    report = None
    if config.chain is not None:
        signal, report = config.chain.finish_frame(
            signal, dn_max_for_bits(sensor.fpa.bit_depth), state.frame_index, t_fpa_k
        )
    dn16 = quantise(signal, sensor.fpa.bit_depth)
    # stage 6: radiometric branch inverts with the *calibration* housing level (ADR 0021)
    lb_housing_cal = float(lut.lookup(config.t_housing_cal_k, q)[()])
    outputs = sensor.outputs
    radiance = apparent_t = None
    if outputs.radiance_linear or outputs.apparent_temperature:
        scene = _scene_radiance_from_signal(signal, config, lb_housing_cal)
        if outputs.radiance_linear:
            radiance = scene
        if outputs.apparent_temperature:
            apparent_t = apparent_temperature(scene, lut, q)
    display8 = None
    isp_hash = None
    if outputs.display_8:
        # §11.2's flat field, on the display branch alone (M9.12). The raw `dn16` above and the
        # radiometric outputs keep the un-corrected ADC plane, because `invert_optics` already
        # divides cos⁴ out per pixel -- correcting both would remove the same term twice. What a
        # viewer sees is what the camera's own ISP shows, which on any real core is flat-fielded.
        display_dn = dn16
        if config.flat_field is not None:
            display_dn = quantise(config.flat_field.apply(dn16), sensor.fpa.bit_depth)
        display = run_display_branch(display_dn, sensor.isp, sensor.fpa.bit_depth)
        display8, isp_hash = display.display8, display.isp_hash
    state.advance()
    return Outputs(
        radiance=radiance,
        apparent_t=apparent_t,
        dn16=dn16 if outputs.dn_16 else None,
        display8=display8,
        signal_dn=signal,
        flux=flux,
        isp_hash=isp_hash,
        report=report,
    )
