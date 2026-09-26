"""Warp stages on the device, op for op the CPU reference. Stage 1 (M10.4), 2-3 (M10.5), 4 (M10.6).

    stage 1   L  = ε₀[material] · L_B(T) + (1 − ε₀) · L_env,   ε₀ = 1 under the sky mask
    stage 2   L' = τ(d) L + (1 − τ(d)) L_B(T_air),             τ = Σ_k w_k exp(−γ_k d)
    stage 3   Φ  = box_k(PSF ∗ L') · Ω_eff τ_opt cos⁴θ A_d + Φ_self,   Ω_eff = π/(4F² + 1)
    stage 4   S_n = S_{n−1} + (S_ideal(Φ) − S_{n−1}) α,                α = 1 − e^{−Δt/τ_th}

`irsim.pipeline` is the oracle (ADR 0018); this module is the GPU path
that is *compared against it*, never a second definition of the physics. The kernel repeats
`BandLUT.lookup` exactly — float32 index, clamp to ``n − KERNEL_CLAMP_MARGIN``, linear
interpolation — so the only differences left are the compiler's fused multiply-adds and the
CPU's float64 blend, both far below the 1e-4 relative / 5 mK budget of the equivalence harness
(`tests/integration/test_kernels_vs_reference.py`, ir-sim-testing skill).

What stays on the host, deliberately: the reflected-environment plane ``l_env`` is built by
`irsim.pipeline.environment.environment_radiance` (a small tilt LUT over the sky-view factor plus
one ground radiance per frame) and uploaded; the emissivity table and the band LUT are uploaded
**once** per (table, device) and cached in :class:`DeviceTables`, so the LUT device pointer does
not change between frames (the M10.4 check that the table is not rebuilt); and every per-frame
scalar of stages 2 and 3 — the exponential sum's (w_k, γ_k), L_B(T_air), the aperture factor,
Φ_self — is evaluated by the model that owns it (:class:`AtmosphereTerms`, :class:`OpticsTerms`).
The kernels scale and add; they never re-derive a coefficient, which is how non-negotiable #5
survives having a second implementation of the imaging chain.

Warp is the ``omni.warp.core`` Kit extension, so the import is guarded: the module imports on any
machine, the kernels exist only when Warp does, and every entry point raises a clear error
otherwise. Kit is not needed to get at it — `irsim_isaac.env.ensure_warp_on_path` puts the
extension on ``sys.path`` and both the ``cpu`` and ``cuda:0`` devices then work from a bare Isaac
Sim interpreter (ADR 0014 addendum), which is what makes the equivalence harness a seconds-long
check rather than a Kit boot. No ``from __future__ import annotations`` here: Warp reads the
kernel's real type annotations.

ADR 0061 (Warp first; SPG only for stages proven stateless). docs/physics-model.md §13.5, §13.6.
"""

# mypy: disable-error-code="valid-type,no-untyped-def,untyped-decorator"
# (Warp kernel signatures are runtime-evaluated type constructors, e.g. wp.array2d(dtype=...))
import hashlib
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from irsim.atmosphere.beer_lambert import transmittance
from irsim.atmosphere.layered import LayeredAtmosphere
from irsim.config.gbuffer import UNMAPPED_MATERIAL_ID
from irsim.config.sensor import IspSpec, SensorSpec
from irsim.detector.bolometer import PHOTON_SCALE_GUARD, MicrobolometerDetector
from irsim.detector.lowpass import alpha_for
from irsim.detector.params import BolometerParams
from irsim.detector.photon import PhotonDetector
from irsim.detector.quantise import dn_max_for_bits
from irsim.isp.agc import CONSTANT_FRAME_LEVEL, percentile_from_histogram
from irsim.isp.bad_pixel import MAX_PASSES as REPLACEMENT_MAX_PASSES
from irsim.isp.palette import palette_table
from irsim.materials.table import MaterialTable
from irsim.noise.stage import NoiseStage
from irsim.noise.three_d import FixedPattern
from irsim.optics.aperture import aperture_factor
from irsim.optics.self_emission import housing_power_axis as housing_power_field_axis
from irsim.optics.stage import optics_field
from irsim.pipeline.atmosphere import atmosphere_stage
from irsim.pipeline.core import PipelineConfig, PipelineState, Planes, require_fp32_or_better
from irsim.pipeline.detector import detector_stage
from irsim.pipeline.environment import environment_radiance
from irsim.pipeline.optics import housing_band_radiance, optics_stage
from irsim.pipeline.radiance import band_radiance_stage
from irsim.radiometry.lut import KERNEL_CLAMP_MARGIN, BandLUT, Quantity
from irsim_isaac.env import ensure_warp_on_path

__all__ = [
    "DEFAULT_DEVICE",
    "AtmosphereTerms",
    "DEVICE_STATE_KEY",
    "DetectorTerms",
    "DeviceTables",
    "EQUIVALENCE_OUTPUT",
    "EQUIVALENCE_STAGES",
    "NoiseTerms",
    "OpticsTerms",
    "WarpAtmosphereStage",
    "WarpBandRadianceStage",
    "WarpDetectorStage",
    "WarpOpticsStage",
    "WarpPipelineState",
    "apply_atmosphere_warp",
    "apply_optics_warp",
    "atmosphere_stage_warp",
    "atmosphere_terms",
    "band_radiance_stage_warp",
    "band_radiance_warp",
    "detector_stage_warp",
    "detector_terms",
    "device_tables",
    "has_warp_module",
    "launch_atmosphere",
    "launch_band_radiance",
    "launch_detector",
    "launch_optics",
    "launch_noise",
    "launch_nuc_residual",
    "launch_ou_drift",
    "launch_defects",
    "launch_rts_step",
    "launch_replacement",
    "launch_ffc_hold",
    "DefectTerms",
    "defect_terms",
    "defects_stage_warp",
    "agc_lut_warp",
    "display_stage_warp",
    "launch_display",
    "launch_histogram",
    "noise_stage_warp",
    "noise_terms",
    "optics_stage_warp",
    "optics_terms",
    "quantise_warp",
    "tables_key",
    "validate_distance",
    "validate_flux",
    "validate_material_ids",
    "warp_pipeline_state",
]

DEFAULT_DEVICE = "cuda:0"


def _import_warp() -> Any:
    """Import Warp if it is available, adding the Kit extension cache to ``sys.path`` first.

    Warp is the ``omni.warp.core`` extension; Kit is normally what puts it on the path, but the
    extension is a plain package and `ensure_warp_on_path` finds it, so this module's kernels
    also run from a bare Isaac Sim interpreter (ADR 0014 addendum).
    """
    ensure_warp_on_path()
    try:
        import warp
    except ImportError:
        return None
    return warp


wp: Any = _import_warp()


def has_warp_module() -> bool:
    """True when Warp imported; the unit gate on a machine without it sees False and still
    imports this module, so the engine-free half of every stage stays testable."""
    return wp is not None


def _require() -> Any:
    if wp is None:
        raise RuntimeError(
            "irsim_isaac.pipeline.warp_stages needs NVIDIA Warp, which ships as the "
            "omni.warp.core Kit extension: run inside Isaac Sim (ADR 0014, 0061)."
        )
    return wp


if wp is not None:

    @wp.kernel
    def _band_radiance_kernel(
        temperature_k: wp.array2d(dtype=wp.float32),
        material_id: wp.array2d(dtype=wp.int32),
        sky_mask: wp.array2d(dtype=wp.uint8),
        l_env: wp.array2d(dtype=wp.float32),
        emissivity: wp.array(dtype=wp.float32),
        lut: wp.array(dtype=wp.float32),
        t0_k: wp.float32,
        span_k: wp.float32,
        n: wp.int32,
        margin: wp.float32,
        use_env: wp.int32,
        radiance: wp.array2d(dtype=wp.float32),
    ):
        i, j = wp.tid()
        t = temperature_k[i, j]
        # BandLUT.lookup, op for op: float32 index, clamp, linear interpolation (§13.5)
        u = (t - t0_k) / span_k * wp.float32(n - 1)
        u = wp.min(wp.max(u, wp.float32(0.0)), wp.float32(n) - margin)
        i0 = wp.int32(u)
        fr = u - wp.float32(i0)
        lb = lut[i0] * (wp.float32(1.0) - fr) + lut[i0 + 1] * fr
        eps = wp.float32(1.0)
        if sky_mask[i, j] == wp.uint8(0):
            eps = emissivity[material_id[i, j]]
        if use_env != 0:
            radiance[i, j] = eps * lb + (wp.float32(1.0) - eps) * l_env[i, j]
        else:
            radiance[i, j] = eps * lb

    @wp.kernel
    def _atmosphere_kernel(
        radiance: wp.array2d(dtype=wp.float32),
        distance_m: wp.array2d(dtype=wp.float32),
        sky_mask: wp.array2d(dtype=wp.uint8),
        weights: wp.array(dtype=wp.float32),
        gamma_per_m: wp.array(dtype=wp.float32),
        n_terms: wp.int32,
        l_air: wp.float32,
        tau_at_inf: wp.float32,
        tau_const: wp.float32,
        out: wp.array2d(dtype=wp.float32),
    ):
        """L' = τ(d) L + (1 − τ(d)) L_B(T_air), τ = Σ_k w_k exp(−γ_k d)  (§7.1, MS.1 per term).

        Σ w_k = 1, so the per-term path radiance Σ_k w_k (1 − τ_k) L_air collapses to
        (1 − τ) L_air exactly -- the same expression serves the grey M8.1 path (one term), the
        layered horizontal path (2-5 terms) and the constant-τ L1 fallback, and no term-weighted
        sum has to be carried separately. ``tau_const`` < 0 means "compute τ from the distance".
        """
        i, j = wp.tid()
        l_in = radiance[i, j]
        if sky_mask[i, j] != wp.uint8(0):
            out[i, j] = l_in  # ADR 0050: a sky pixel already carries the whole column
            return
        tau = tau_const
        if tau_const < wp.float32(0.0):
            d = distance_m[i, j]
            if wp.isinf(d):
                tau = tau_at_inf  # the host decides: 0, or 1 for a grey γ of exactly 0
            else:
                tau = wp.float32(0.0)
                for k in range(n_terms):
                    tau += weights[k] * wp.exp(-gamma_per_m[k] * d)
        out[i, j] = tau * l_in + (wp.float32(1.0) - tau) * l_air

    @wp.kernel
    def _psf_kernel(
        image: wp.array2d(dtype=wp.float32),
        psf: wp.array2d(dtype=wp.float32),
        radius_y: wp.int32,
        radius_x: wp.int32,
        out: wp.array2d(dtype=wp.float32),
    ):
        """Direct convolution with edge replication, at the supersampled pitch (§8.3, ADR 0059).

        The CPU oracle convolves by FFT over an edge-padded image and crops; because the padding
        is at least the kernel radius the circular convolution never wraps into the crop, so it
        equals this direct sum with clamped indices — same result, no FFT to reproduce. The index
        is ``i + r − a``, the convolution flip: the PSF is radially symmetric so it makes no
        difference today, and it makes the kernel correct if an asymmetric one is ever passed.
        """
        i, j = wp.tid()
        h = image.shape[0]
        w = image.shape[1]
        acc = wp.float32(0.0)
        for a in range(2 * radius_y + 1):
            y = wp.clamp(i + radius_y - a, 0, h - 1)
            for b in range(2 * radius_x + 1):
                x = wp.clamp(j + radius_x - b, 0, w - 1)
                acc += psf[a, b] * image[y, x]
        out[i, j] = acc

    @wp.kernel
    def _optics_kernel(
        radiance_ss: wp.array2d(dtype=wp.float32),
        cos4: wp.array2d(dtype=wp.float32),
        factor: wp.float32,
        area_m2: wp.float32,
        lb_housing: wp.float32,
        phi_housing: wp.float32,
        supersample: wp.int32,
        flux: wp.array2d(dtype=wp.float32),
    ):
        """Box-mean k× downsample, then Φ = A_d Ω τ RI (L − L_h) + A_d Ω L_h (§8.1, §8.2, §8.3).

        ``factor`` is Ω_eff τ_opt and ``phi_housing`` is A_d Ω_eff L_B(T_housing), both computed
        on the host by `irsim.optics` (ADR 0145) — the aperture factor π/(4F² + 1) is written once,
        in `irsim.optics.aperture`, and `tests/unit/test_aperture_guard.py` walks the AST of this
        file to keep it that way (non-negotiable #5). The box is the full pitch; the fill factor
        is already in A_d, not in the box (ADR 0020).
        """
        i, j = wp.tid()
        acc = wp.float32(0.0)
        for a in range(supersample):
            for b in range(supersample):
                acc += radiance_ss[i * supersample + a, j * supersample + b]
        mean = acc / (wp.float32(supersample) * wp.float32(supersample))
        flux[i, j] = (mean - lb_housing) * factor * cos4[i, j] * area_m2 + phi_housing

    @wp.kernel
    def _bolometer_signal_kernel(
        flux_w: wp.array2d(dtype=wp.float32),
        gain_dn_per_w: wp.float32,
        offset_w: wp.float32,
        signal_dn: wp.array2d(dtype=wp.float32),
    ):
        """S_ideal = gain · (Φ − offset), the ADR 0019 linear static transfer (§9.2).

        Written as (Φ − offset) · gain, not Φ·gain − offset·gain: Φ and the offset are both of
        order 1e-8 W and their difference spans the ADC, so the second form would subtract two
        large numbers to get a small one. In this form the float32 spacing at 1e-8 W is ~0.003 DN.
        """
        i, j = wp.tid()
        signal_dn[i, j] = (flux_w[i, j] - offset_w) * gain_dn_per_w

    @wp.kernel
    def _photon_signal_kernel(
        flux_q: wp.array2d(dtype=wp.float32),
        qe_t_int: wp.float32,
        offset_e: wp.float32,
        dn_per_electron: wp.float32,
        signal_dn: wp.array2d(dtype=wp.float32),
    ):
        """N_e = η t_int Φ_q + N_dark + N_bg, then S = N_e / N_well · 2^bits (§9.1).

        No lag: a cooled photon detector is memoryless on these timescales (ADR 0052), which is
        the Tier 3 check that LWIR smears and cooled MWIR does not.
        """
        i, j = wp.tid()
        signal_dn[i, j] = (flux_q[i, j] * qe_t_int + offset_e) * dn_per_electron

    @wp.kernel
    def _bolometer_lag_kernel(
        signal_dn: wp.array2d(dtype=wp.float32),
        alpha: wp.float32,
        adopt: wp.int32,
        state: wp.array2d(dtype=wp.float32),
        out: wp.array2d(dtype=wp.float32),
    ):
        """S_n = S_{n−1} + (S_ideal − S_{n−1}) α, in place on the persistent state (§9.2).

        ``adopt`` is the first frame: the membrane starts settled, not at zero, so a sequence does
        not open with a frame-long ramp no real core shows. ``state`` is the device buffer
        :class:`WarpPipelineState` owns; it is updated in place and never leaves the device, and
        ``out`` carries the frame onward so the next stage has something to read that the next
        frame will not overwrite.
        """
        i, j = wp.tid()
        s = signal_dn[i, j]
        if adopt != 0:
            state[i, j] = s
        else:
            state[i, j] = state[i, j] + (s - state[i, j]) * alpha
        out[i, j] = state[i, j]

    @wp.kernel
    def _quantise_kernel(
        signal_dn: wp.array2d(dtype=wp.float32),
        dn_max: wp.int32,
        dn: wp.array2d(dtype=wp.uint16),
    ):
        """DN = clip(floor(S), 0, 2^bits − 1) (§2 𝒬, §9.1).

        Floor, not round: an ideal ADC compares against thresholds, which is what makes the
        quantisation error uniform on [0, 1) LSB with 0.29 LSB rms — the figure the Tier 2 SITF
        bench is written against (ADR 0019). Saturation clips and never wraps.
        """
        i, j = wp.tid()
        code = wp.int32(wp.floor(signal_dn[i, j]))
        dn[i, j] = wp.uint16(wp.clamp(code, 0, dn_max))

    @wp.kernel
    def _noise_kernel(
        signal_dn: wp.array2d(dtype=wp.float32),
        v_fixed: wp.array(dtype=wp.float32),
        h_fixed: wp.array(dtype=wp.float32),
        vh_fixed: wp.array2d(dtype=wp.float32),
        sigma_tvh: wp.float32,
        sigma_t: wp.float32,
        sigma_tv: wp.float32,
        sigma_th: wp.float32,
        frame_index: wp.int32,
        seed: wp.int32,
        cols: wp.int32,
        out: wp.array2d(dtype=wp.float32),
    ):
        """Stage 5 on device: the seven §10.2 components added in DN space (non-negotiable #3).

        The fixed terms arrive as device arrays already scaled to sigma_TVH = 1, exactly as the
        CPU stage stores them, and are multiplied by this frame's sigma_TVH here -- so the *same*
        breathing pattern (M9.4) drives both paths and only the per-frame draws differ.

        Counters follow the roadmap's rule, ``frame * H * W + pixel``, with a separate offset per
        stream so that the row, column and frame terms cannot correlate with the per-pixel one.
        Warp's generator is not NumPy's, so this path is held to *statistical* equivalence with
        the CPU oracle rather than bit-equality (ADR 0022).
        """
        i, j = wp.tid()
        pixel = i * cols + j
        base = frame_index * 1000003 + seed

        # TVH: one independent normal per pixel per frame.
        state_tvh = wp.rand_init(base, pixel)
        n = signal_dn[i, j] + sigma_tvh * wp.randn(state_tvh)

        # T: one value per frame, so every pixel must draw the same one.
        state_t = wp.rand_init(base + 7, 0)
        n = n + sigma_t * wp.randn(state_t)
        # TV: one per row, constant along the row. TH: one per column.
        state_tv = wp.rand_init(base + 13, i)
        n = n + sigma_tv * wp.randn(state_tv)
        state_th = wp.rand_init(base + 29, j)
        n = n + sigma_th * wp.randn(state_th)

        # The fixed terms: per-row, per-column, per-pixel, scaled here.
        n = n + sigma_tvh * (v_fixed[i] + h_fixed[j] + vh_fixed[i, j])
        out[i, j] = n

    @wp.kernel
    def _ou_step_kernel_1d(
        x: wp.array(dtype=wp.float32),
        sigma: wp.float32,
        decay: wp.float32,
        innovation: wp.float32,
        epoch: wp.int32,
        seed: wp.int32,
        lane: wp.int32,
    ):
        """One exact OU step on a 1-D fixed term, in place (§10.3, ADR 0054).

        Same closed form as ``irsim.noise.ou_step``: decay and innovation are the exact solution
        over the interval, so the stationary variance is preserved at any step size on either
        path. ``sigma`` is in sigma_TVH = 1 units, like the buffer it updates.
        """
        i = wp.tid()
        state = wp.rand_init(epoch * 1000003 + seed + lane, i)
        x[i] = x[i] * decay + sigma * innovation * wp.randn(state)

    @wp.kernel
    def _ou_step_kernel_2d(
        x: wp.array2d(dtype=wp.float32),
        sigma: wp.float32,
        decay: wp.float32,
        innovation: wp.float32,
        epoch: wp.int32,
        seed: wp.int32,
        lane: wp.int32,
        cols: wp.int32,
    ):
        i, j = wp.tid()
        state = wp.rand_init(epoch * 1000003 + seed + lane, i * cols + j)
        x[i, j] = x[i, j] * decay + sigma * innovation * wp.randn(state)

    @wp.kernel
    def _nuc_residual_kernel(
        signal_dn: wp.array2d(dtype=wp.float32),
        gain_xi: wp.array2d(dtype=wp.float32),
        offset_xi: wp.array2d(dtype=wp.float32),
        reference: wp.array2d(dtype=wp.float32),
        gain_scale: wp.float32,
        offset_scale: wp.float32,
        out: wp.array2d(dtype=wp.float32),
    ):
        """ref + g_ij (x − ref) + o_ij, both scales already carrying ΔT_FPA (M9.6, §11.2).

        ``reference`` is the closed-shutter frame of the last FFC (SC.18), or zeros for the
        pre-SC.18 form g x + o.

        The scales are computed on the host from the same ``NucSpec`` and the same ∂DN/∂T the CPU
        path used, so the only thing that can differ between the two paths is the xi fields --
        and those are uploaded, not redrawn. At ΔT = 0 both scales are zero and this is exactly
        the identity, on either device.
        """
        i, j = wp.tid()
        g = wp.float32(1.0) + gain_scale * gain_xi[i, j]
        r = reference[i, j]
        out[i, j] = r + (signal_dn[i, j] - r) * g + offset_scale * offset_xi[i, j]

    @wp.kernel
    def _histogram_kernel(
        dn: wp.array2d(dtype=wp.uint16),
        counts: wp.array(dtype=wp.int32),
    ):
        """Atomic histogram of the DN plane into 2^bit_depth bins (§11.3).

        One atomic add per pixel. The alternative -- a per-block private histogram reduced
        afterwards -- would be faster on a 16-bit plane, and is not worth the extra state until a
        profile says the histogram is the bottleneck rather than the LUT application.
        """
        i, j = wp.tid()
        wp.atomic_add(counts, wp.int32(dn[i, j]), wp.int32(1))

    @wp.kernel
    def _clip_counts_kernel(
        counts: wp.array(dtype=wp.int32),
        plateau_count: wp.float32,
        clipped: wp.array(dtype=wp.float32),
    ):
        """min(count, P) per bin -- the plateau clip of ADR 0028, before the CDF."""
        i = wp.tid()
        c = wp.float32(counts[i])
        clipped[i] = wp.min(c, plateau_count)

    @wp.kernel
    def _apply_lut_kernel(
        dn: wp.array2d(dtype=wp.uint16),
        lut: wp.array(dtype=wp.float32),
        y: wp.array2d(dtype=wp.float32),
    ):
        """DN -> [0, 1] through the AGC's own lookup table.

        Both §11.3 modes are monotone functions of DN alone, so each *is* a table -- which is what
        makes them portable to a kernel, and why the port can be held to a display code.
        """
        i, j = wp.tid()
        y[i, j] = lut[wp.int32(dn[i, j])]

    @wp.kernel
    def _dde_kernel(
        y: wp.array2d(dtype=wp.float32),
        gain: wp.float32,
        rows: wp.int32,
        cols: wp.int32,
        out: wp.array2d(dtype=wp.float32),
    ):
        """Unsharp mask with a 3x3 box and edge clamping: y + gain (y - box3(y)), clipped (§11.4).

        Edge clamping, not zero padding, matching ``irsim.isp.dde``'s "edge" boundary: zero padding
        would darken the frame border by up to the gain, which reads as a vignette that the optics
        model did not put there.
        """
        i, j = wp.tid()
        total = wp.float32(0.0)
        for di in range(-1, 2):
            for dj in range(-1, 2):
                ii = wp.clamp(i + di, 0, rows - 1)
                jj = wp.clamp(j + dj, 0, cols - 1)
                total = total + y[ii, jj]
        box = total / wp.float32(9.0)
        v = y[i, j] + gain * (y[i, j] - box)
        out[i, j] = wp.clamp(v, wp.float32(0.0), wp.float32(1.0))

    @wp.kernel
    def _palette_kernel(
        y: wp.array2d(dtype=wp.float32),
        palette: wp.array2d(dtype=wp.uint8),
        invert: wp.int32,
        out: wp.array3d(dtype=wp.uint8),
    ):
        """[0, 1] -> RGBA8 through polarity and the palette table (§11.4, ADR 0030).

        The 8-bit quantisation mirrors ``irsim.isp.quantise_display`` exactly -- round-half-away
        then clamp, not floor -- because this is a *display* level rather than an ADC threshold,
        and the two rules differ by half a code on every pixel.
        """
        i, j = wp.tid()
        v = wp.clamp(y[i, j], wp.float32(0.0), wp.float32(1.0))
        code = wp.int32(wp.floor(v * wp.float32(255.0) + wp.float32(0.5)))
        code = wp.clamp(code, 0, 255)
        if invert != 0:
            code = 255 - code
        out[i, j, 0] = palette[code, 0]
        out[i, j, 1] = palette[code, 1]
        out[i, j, 2] = palette[code, 2]
        out[i, j, 3] = wp.uint8(255)

    @wp.kernel
    def _rts_step_kernel(
        kind: wp.array2d(dtype=wp.uint8),
        was_bad: wp.array2d(dtype=wp.uint8),
        p_enter: wp.float32,
        p_leave: wp.float32,
        frame_index: wp.int32,
        seed: wp.int32,
        cols: wp.int32,
        out: wp.array2d(dtype=wp.uint8),
    ):
        """One step of the two-state RTS chain of the flickering and blinking pixels (§10.4).

        Launched over the whole plane rather than over a compact list of the stateful pixels: the
        CPU restricts it because drawing 327k uniforms to move a few dozen bits is most of the
        noise chain's cost, and on a GPU that argument does not apply. Every other pixel takes the
        branch that writes zero.

        The chain, not a fresh draw: a pixel that was bad leaves with probability 1/dwell and one
        that was good enters with the balancing probability. Redrawing the state independently
        each frame would give white dwell statistics instead of geometric ones -- ordinary noise
        where the defining signature of RTS should be.
        """
        i, j = wp.tid()
        k = kind[i, j]
        if k != wp.uint8(3) and k != wp.uint8(4):  # FLICKERING, BLINKING
            out[i, j] = wp.uint8(0)
            return
        state = wp.rand_init(frame_index * 1000003 + seed + 101, i * cols + j)
        u = wp.randf(state)
        if was_bad[i, j] != wp.uint8(0):
            out[i, j] = wp.where(u >= p_leave, wp.uint8(1), wp.uint8(0))
        else:
            out[i, j] = wp.where(u < p_enter, wp.uint8(1), wp.uint8(0))

    @wp.kernel
    def _apply_defects_kernel(
        signal_dn: wp.array2d(dtype=wp.float32),
        kind: wp.array2d(dtype=wp.uint8),
        rts_bad: wp.array2d(dtype=wp.uint8),
        on_map: wp.array2d(dtype=wp.uint8),
        dn_max: wp.int32,
        amplitude_dn: wp.int32,
        out: wp.array2d(dtype=wp.float32),
        active: wp.array2d(dtype=wp.uint8),
    ):
        """§11.1's defect injection on the quantised plane, op for op with ``apply_defects``.

        Two details are load-bearing. The defect is applied to ``floor(signal)`` and carried back
        onto the float plane **only where it actually changed the code**, so an already-saturated
        hot pixel keeps its sub-LSB value exactly as on the CPU. And the replacement mask is the
        *active* mask -- dead and hot always, blinking and flickering only while their chain is in
        the bad state -- which is what makes an intermittent defect reach the image through a
        static factory map (§10.4). ``active`` is written as ``replacement_mask``: an active
        defect that is not ``on_map`` -- a late defect, SC.19 -- is left in the image.
        """
        i, j = wp.tid()
        s = signal_dn[i, j]
        q = wp.int32(wp.clamp(wp.floor(s), wp.float32(0.0), wp.float32(dn_max)))
        d = q
        k = kind[i, j]
        bad = rts_bad[i, j] != wp.uint8(0)
        is_active = wp.uint8(0)
        if k == wp.uint8(1):  # DEAD
            d = 0
            is_active = wp.uint8(1)
        elif k == wp.uint8(2):  # HOT
            d = dn_max
            is_active = wp.uint8(1)
        elif k == wp.uint8(4) and bad:  # BLINKING, currently bad
            d = 0
            is_active = wp.uint8(1)
        elif k == wp.uint8(3) and bad:  # FLICKERING, currently bad
            d = wp.clamp(q + amplitude_dn, 0, dn_max)
            is_active = wp.uint8(1)
        active[i, j] = is_active * on_map[i, j]
        if d != q:
            out[i, j] = wp.float32(d)
        else:
            out[i, j] = s

    @wp.kernel
    def _replace_init_kernel(
        signal_dn: wp.array2d(dtype=wp.float32),
        active: wp.array2d(dtype=wp.uint8),
        work: wp.array2d(dtype=wp.float64),
        valid: wp.array2d(dtype=wp.uint8),
    ):
        """Seed the replacement: float64 work plane and the validity mask ``replace_bad_pixels``
        iterates on. float64 because the CPU accumulates there, and a float32 mean of four
        neighbours differs from it in the last bit -- which is the difference between "matches
        exactly" and "matches to a tolerance nobody can interpret."""
        i, j = wp.tid()
        work[i, j] = wp.float64(signal_dn[i, j])
        valid[i, j] = wp.where(active[i, j] != wp.uint8(0), wp.uint8(0), wp.uint8(1))

    @wp.kernel
    def _replace_pass_kernel(
        work: wp.array2d(dtype=wp.float64),
        valid: wp.array2d(dtype=wp.uint8),
        rows: wp.int32,
        cols: wp.int32,
        work_out: wp.array2d(dtype=wp.float64),
        valid_out: wp.array2d(dtype=wp.uint8),
        remaining: wp.array(dtype=wp.int32),
        filled: wp.array(dtype=wp.int32),
    ):
        """One pass of the iterated 4-neighbour mean, from the state at the pass's start.

        Ping-pong rather than in place: the CPU builds its neighbour sums from ``work`` before it
        writes any of them, so a pixel filled earlier in the same pass must not feed its
        neighbour. In place on a GPU that ordering is not merely different, it is
        non-deterministic.
        """
        i, j = wp.tid()
        if valid[i, j] != wp.uint8(0):
            work_out[i, j] = work[i, j]
            valid_out[i, j] = wp.uint8(1)
            return
        total = wp.float64(0.0)
        count = wp.int32(0)
        if i > 0 and valid[i - 1, j] != wp.uint8(0):
            total = total + work[i - 1, j]
            count = count + 1
        if i + 1 < rows and valid[i + 1, j] != wp.uint8(0):
            total = total + work[i + 1, j]
            count = count + 1
        if j > 0 and valid[i, j - 1] != wp.uint8(0):
            total = total + work[i, j - 1]
            count = count + 1
        if j + 1 < cols and valid[i, j + 1] != wp.uint8(0):
            total = total + work[i, j + 1]
            count = count + 1
        if count > 0:
            work_out[i, j] = total / wp.float64(count)
            valid_out[i, j] = wp.uint8(1)
            wp.atomic_add(filled, 0, 1)
        else:
            work_out[i, j] = work[i, j]
            valid_out[i, j] = wp.uint8(0)
            wp.atomic_add(remaining, 0, 1)

    @wp.kernel
    def _replace_finish_kernel(
        work: wp.array2d(dtype=wp.float64), out: wp.array2d(dtype=wp.float32)
    ):
        i, j = wp.tid()
        out[i, j] = wp.float32(work[i, j])


# ---- device-resident tables ------------------------------------------------------------------


def tables_key(
    lut: BandLUT, materials: MaterialTable, quantity: Quantity, device: str
) -> tuple[Any, ...]:
    """Cache key: the LUT and the emissivity column by content, so a rebuilt table re-uploads
    and an unchanged one never does (identity alone would miss an in-place rebuild)."""
    lut_digest = hashlib.sha1(np.ascontiguousarray(lut.table(quantity)).tobytes()).hexdigest()
    eps_digest = hashlib.sha1(np.ascontiguousarray(materials.emissivity).tobytes()).hexdigest()
    return (lut_digest, float(lut.t0_k), float(lut.t1_k), int(lut.n), eps_digest, device)


@dataclass(frozen=True)
class DeviceTables:
    """The band LUT and the ε₀ column on one device, uploaded once."""

    key: tuple[Any, ...]
    device: str
    lut: Any  # wp.array(float32), n entries
    emissivity: Any  # wp.array(float32), index = material id (NaN where undefined)
    t0_k: float
    t1_k: float
    n: int

    @property
    def lut_ptr(self) -> int:
        return int(self.lut.ptr)


_TABLES: dict[tuple[Any, ...], DeviceTables] = {}


def device_tables(
    lut: BandLUT,
    materials: MaterialTable,
    quantity: Quantity = "lb",
    device: str = DEFAULT_DEVICE,
) -> DeviceTables:
    """Upload (or fetch the cached) LUT and emissivity tables for ``device``."""
    warp = _require()
    key = tables_key(lut, materials, quantity, device)
    cached = _TABLES.get(key)
    if cached is not None:
        return cached
    table = np.ascontiguousarray(lut.table(quantity), dtype=np.float32)
    eps = np.ascontiguousarray(materials.emissivity, dtype=np.float32)
    tables = DeviceTables(
        key=key,
        device=device,
        lut=warp.array(table, dtype=warp.float32, device=device),
        emissivity=warp.array(eps, dtype=warp.float32, device=device),
        t0_k=float(lut.t0_k),
        t1_k=float(lut.t1_k),
        n=int(lut.n),
    )
    _TABLES[key] = tables
    return tables


# ---- host-side guards (the CPU reference raises on these; the kernel cannot) -----------------


def validate_material_ids(
    material_id: NDArray[np.integer],
    materials: MaterialTable,
    sky_mask: NDArray[np.bool_] | None = None,
) -> None:
    """Refuse what `MaterialTable.emissivity_for` refuses, before anything is uploaded."""
    ids = np.asarray(material_id)
    if not np.issubdtype(ids.dtype, np.integer):
        raise TypeError(f"material_id must be an integer plane, got {ids.dtype}")
    if sky_mask is not None:
        sky = np.asarray(sky_mask)
        if sky.dtype != np.bool_ or sky.shape != ids.shape:
            raise ValueError("sky_mask must be a bool plane with the material_id shape")
        ids = ids[~sky]
    if ids.size == 0:
        return
    used = np.unique(ids)
    if used[0] == UNMAPPED_MATERIAL_ID:
        raise ValueError(
            "G-buffer contains material id 0 (UNMAPPED): an asset has no material mapping; "
            "fix the resolver rather than rendering a default emissivity"
        )
    if used[0] < 0 or used[-1] >= materials.emissivity.size:
        raise ValueError(
            f"material ids span {used[0]}..{used[-1]}, table has {materials.emissivity.size} rows"
        )
    if np.any(np.isnan(materials.emissivity[used])):
        bad = used[np.isnan(materials.emissivity[used])]
        raise ValueError(f"material ids {bad.tolist()} have no emissivity in this band")


# ---- the stage ------------------------------------------------------------------------------


def _as_device(warp: Any, value: Any, dtype: Any, device: str) -> Any:
    """A Warp array on ``device`` (uploaded from NumPy, or the caller's array unchanged)."""
    if isinstance(value, warp.array):
        return value
    return warp.array(np.ascontiguousarray(value), dtype=dtype, device=device)


def launch_band_radiance(
    temperature_k: Any,
    material_id: Any,
    sky_mask: Any,
    l_env: Any,
    tables: DeviceTables,
    radiance: Any,
) -> Any:
    """Launch the kernel on device arrays already resident on ``tables.device``.

    ``sky_mask`` and ``l_env`` may be ``None`` (no sky pixels / emission only); ``radiance`` is
    the float32 output array to fill. Returns it. This is the entry point the on-device chain
    (M10.9a) uses; :func:`band_radiance_warp` wraps it for NumPy callers.
    """
    warp = _require()
    h, w = temperature_k.shape
    device = tables.device
    if sky_mask is None:
        sky_mask = warp.zeros((h, w), dtype=warp.uint8, device=device)
    use_env = 0 if l_env is None else 1
    if l_env is None:
        l_env = warp.zeros((h, w), dtype=warp.float32, device=device)
    warp.launch(
        _band_radiance_kernel,
        dim=(h, w),
        inputs=[
            temperature_k,
            material_id,
            sky_mask,
            l_env,
            tables.emissivity,
            tables.lut,
            np.float32(tables.t0_k),
            np.float32(np.float32(tables.t1_k) - np.float32(tables.t0_k)),
            np.int32(tables.n),
            np.float32(KERNEL_CLAMP_MARGIN),
            np.int32(use_env),
        ],
        outputs=[radiance],
        device=device,
    )
    return radiance


def band_radiance_warp(
    temperature_k: NDArray[np.floating],
    material_id: NDArray[np.integer],
    materials: MaterialTable,
    lut: BandLUT,
    quantity: Quantity = "lb",
    sky_mask: NDArray[np.bool_] | None = None,
    l_env: NDArray[np.floating] | None = None,
    *,
    device: str = DEFAULT_DEVICE,
) -> NDArray[np.float32]:
    """NumPy in, NumPy out: the signature of `irsim.pipeline.radiance.band_radiance`."""
    warp = _require()
    t = require_fp32_or_better(np.asarray(temperature_k), "temperature_k").astype(np.float32)
    ids = np.asarray(material_id)
    if ids.shape != t.shape:
        raise ValueError(f"material_id shape {ids.shape} != temperature shape {t.shape}")
    validate_material_ids(ids, materials, sky_mask)
    env_np = None
    if l_env is not None:
        env_np = require_fp32_or_better(np.asarray(l_env), "l_env").astype(np.float32)
        if env_np.shape != t.shape:
            raise ValueError(f"l_env shape {env_np.shape} != temperature shape {t.shape}")
    tables = device_tables(lut, materials, quantity, device)
    out = warp.zeros(t.shape, dtype=warp.float32, device=device)
    launch_band_radiance(
        _as_device(warp, t, warp.float32, device),
        _as_device(warp, ids.astype(np.int32), warp.int32, device),
        None
        if sky_mask is None
        else _as_device(warp, np.asarray(sky_mask).astype(np.uint8), warp.uint8, device),
        None if env_np is None else _as_device(warp, env_np, warp.float32, device),
        tables,
        out,
    )
    return np.asarray(out.numpy(), dtype=np.float32)


def band_radiance_stage_warp(
    planes: Planes, config: PipelineConfig, state: PipelineState, device: str = DEFAULT_DEVICE
) -> Planes:
    """Stage-1 entry point on the plane dict, the GPU twin of `band_radiance_stage`."""
    l_env = None
    if config.sky is not None:
        l_env = environment_radiance(
            config.sky,
            config.lut,
            state.t_s,
            np.asarray(planes["sky_view_factor"]),
            config.quantity,
        )
    out = band_radiance_warp(
        planes["temperature_k"],
        planes["material_id"],
        config.materials,
        config.lut,
        config.quantity,
        sky_mask=planes.get("sky_mask"),
        l_env=l_env,
        device=device,
    )
    return {"radiance": out}


class WarpBandRadianceStage:
    """`irsim.pipeline.core.Stage` implementation running stage 1 on ``device``."""

    name = "band_radiance"

    def __init__(self, device: str = DEFAULT_DEVICE) -> None:
        self.device = device

    def __call__(self, planes: Planes, config: PipelineConfig, state: PipelineState) -> Planes:
        return band_radiance_stage_warp(planes, config, state, self.device)


# ---- stage 2: atmosphere ---------------------------------------------------------------------


@dataclass(frozen=True)
class AtmosphereTerms:
    """What stage 2 needs for one frame, read off the CPU model on the host (M10.5).

    The exponential sum is a property of the band and the weather, not of the pixel, so it is
    evaluated once per frame by the model that owns it and only ``(w_k, γ_k)``, ``L_B(T_air)``
    and the two sentinels cross to the device. Nothing here re-derives physics: ``tau_at_inf``
    is obtained by *asking the oracle* for τ(∞) rather than restating its rule, which is how the
    grey model's "γ = 0 means a transparent infinite path" and the layered model's "an infinite
    horizontal path is opaque" both come out right without a branch here.

    ``gamma_per_m`` is the total surface extinction per term including aerosol; on a horizontal
    ray every column integral is just ``d``, so the two contributions add (docs/physics-model.md
    §7.1, `irsim.atmosphere.layered.ExponentialSum.optical_depths`). Per-pixel *slant* paths are
    MS.8's job and are not in this form yet.
    """

    weights: NDArray[np.float32]
    gamma_per_m: NDArray[np.float32]
    l_air: float
    tau_at_inf: float
    tau_const: float = -1.0  # < 0: compute τ from the distance; >= 0: the L1 constant-τ fallback

    def __post_init__(self) -> None:
        if self.weights.shape != self.gamma_per_m.shape or self.weights.ndim != 1:
            raise ValueError("weights and gamma_per_m must be one matching 1-D array each")
        if abs(float(self.weights.sum()) - 1.0) > 1e-6:
            raise ValueError(f"class weights sum to {float(self.weights.sum())}, not 1")
        if np.any(self.gamma_per_m < 0.0) or not np.all(np.isfinite(self.gamma_per_m)):
            raise ValueError("extinction coefficients must be finite and non-negative")

    @property
    def n_terms(self) -> int:
        return int(self.weights.size)

    def transmittance(self, distance_m: NDArray[np.floating]) -> NDArray[np.float64]:
        """τ(d) as the kernel computes it, on the host — the check that the terms were read off
        the model correctly is `tests/unit/test_warp_atmosphere_terms.py`, no GPU involved."""
        d = np.asarray(distance_m, dtype=np.float64)
        if self.tau_const >= 0.0:
            return np.full(d.shape, self.tau_const, dtype=np.float64)
        od = self.gamma_per_m.astype(np.float64)[:, None] * d.reshape(1, -1)
        tau = (self.weights.astype(np.float64)[:, None] * np.exp(-od)).sum(axis=0)
        return np.asarray(np.where(np.isinf(d.ravel()), self.tau_at_inf, tau), dtype=np.float64)


def atmosphere_terms(config: PipelineConfig, state: PipelineState) -> AtmosphereTerms | None:
    """Read stage 2's per-frame terms off whichever atmosphere the config holds; None = identity.

    Mirrors `irsim.pipeline.atmosphere.atmosphere_stage` branch for branch: no atmosphere is the
    identity stage, a `LayeredAtmosphere` gives the MS.1 exponential sum at the frame's weather
    time, and a grey `Atmosphere` gives the single M8.1 term (with ``tau_override`` as the L1
    fallback). Both take L_B(T_air) from the LUT the oracle uses — the atmosphere's own for the
    layered model, the pipeline's for the grey one — so an isothermal scene stays invariant
    against stage 1 on the device as well (§7.1).
    """
    atmosphere = config.atmosphere
    if atmosphere is None:
        return None
    band = config.sensor.sensor.band.band_id
    if isinstance(atmosphere, LayeredAtmosphere):
        es = atmosphere.exponential_sum(band, state.t_s)
        return AtmosphereTerms(
            weights=es.weights.astype(np.float32),
            gamma_per_m=(es.gamma_0 + es.gamma_aerosol).astype(np.float32),
            l_air=atmosphere.air_radiance(band, state.t_s, config.quantity),
            tau_at_inf=float(es.transmittance(np.inf, 0.0)[()]),
        )
    atm_state = atmosphere.state(state.t_s)
    gamma = float(atm_state.gamma_per_m[band])
    return AtmosphereTerms(
        weights=np.ones(1, dtype=np.float32),
        gamma_per_m=np.asarray([gamma], dtype=np.float32),
        l_air=float(config.lut.lookup(np.float64(atm_state.t_air_k), config.quantity)[()]),
        tau_at_inf=float(transmittance(np.inf, gamma)[()]),
        tau_const=-1.0 if config.tau_override is None else float(config.tau_override),
    )


def launch_atmosphere(
    radiance: Any, distance_m: Any, sky_mask: Any, terms: AtmosphereTerms, out: Any, device: str
) -> Any:
    """Launch stage 2 on device arrays already resident on ``device``; fills and returns ``out``.

    ``sky_mask`` may be None (no sky pixels). This is the entry point the on-device chain
    (M10.9a) uses, so the radiance never returns to the host between stages 1 and 2.
    """
    warp = _require()
    h, w = radiance.shape
    if sky_mask is None:
        sky_mask = warp.zeros((h, w), dtype=warp.uint8, device=device)
    warp.launch(
        _atmosphere_kernel,
        dim=(h, w),
        inputs=[
            radiance,
            distance_m,
            sky_mask,
            _as_device(warp, terms.weights, warp.float32, device),
            _as_device(warp, terms.gamma_per_m, warp.float32, device),
            np.int32(terms.n_terms),
            np.float32(terms.l_air),
            np.float32(terms.tau_at_inf),
            np.float32(terms.tau_const),
        ],
        outputs=[out],
        device=device,
    )
    return out


def validate_distance(
    distance_m: NDArray[np.floating], shape: tuple[int, ...], sky_mask: Any = None
) -> None:
    """Refuse what `irsim.atmosphere.beer_lambert.transmittance` refuses, before any upload.

    A negative or NaN range is a broken G-buffer in either model; the grey path raises on it and
    the layered path would quietly return nonsense, so the guard is applied to both.
    """
    d = np.asarray(distance_m)
    if d.shape != shape:
        raise ValueError(f"distance_m shape {d.shape} != radiance shape {shape}")
    if np.any(d < 0.0) or np.any(np.isnan(d)):
        raise ValueError("distance_m must be non-negative (inf allowed)")
    if sky_mask is not None:
        sky = np.asarray(sky_mask)
        if sky.dtype != np.bool_ or sky.shape != shape:
            raise ValueError("sky_mask must be a bool plane with the radiance shape")


def apply_atmosphere_warp(
    radiance: NDArray[np.floating],
    distance_m: NDArray[np.floating],
    terms: AtmosphereTerms,
    sky_mask: NDArray[np.bool_] | None = None,
    *,
    device: str = DEFAULT_DEVICE,
) -> NDArray[np.float32]:
    """NumPy in, NumPy out: the GPU twin of `apply_atmosphere_gbuffer` / `apply_layered_gbuffer`."""
    warp = _require()
    l_in = require_fp32_or_better(np.asarray(radiance), "radiance").astype(np.float32)
    validate_distance(distance_m, l_in.shape, sky_mask)
    out = warp.zeros(l_in.shape, dtype=warp.float32, device=device)
    launch_atmosphere(
        _as_device(warp, l_in, warp.float32, device),
        _as_device(warp, np.asarray(distance_m, dtype=np.float32), warp.float32, device),
        None
        if sky_mask is None
        else _as_device(warp, np.asarray(sky_mask).astype(np.uint8), warp.uint8, device),
        terms,
        out,
        device,
    )
    return np.asarray(out.numpy(), dtype=np.float32)


def atmosphere_stage_warp(
    planes: Planes, config: PipelineConfig, state: PipelineState, device: str = DEFAULT_DEVICE
) -> Planes:
    """Stage-2 entry point on the plane dict, the GPU twin of `atmosphere_stage`."""
    radiance = np.asarray(planes["radiance"])
    terms = atmosphere_terms(config, state)
    if terms is None:
        return {"radiance": radiance}
    return {
        "radiance": apply_atmosphere_warp(
            radiance,
            np.asarray(planes["distance_m"]),
            terms,
            sky_mask=planes.get("sky_mask"),
            device=device,
        )
    }


class WarpAtmosphereStage:
    """`irsim.pipeline.core.Stage` implementation running stage 2 on ``device``."""

    name = "atmosphere"

    def __init__(self, device: str = DEFAULT_DEVICE) -> None:
        self.device = device

    def __call__(self, planes: Planes, config: PipelineConfig, state: PipelineState) -> Planes:
        return atmosphere_stage_warp(planes, config, state, self.device)


# ---- stage 3: optics -------------------------------------------------------------------------


@dataclass(frozen=True)
class OpticsTerms:
    """What stage 3 needs for one frame, all of it computed by `irsim.optics` on the host (M10.5).

    Every radiometric scalar crosses to the device already evaluated. That is not an optimisation:
    ``factor`` is Ω_eff τ_opt with Ω_eff = π/(4F² + 1) from `irsim.optics.aperture.aperture_factor`,
    and non-negotiable #5 says that expression exists once in the codebase. A kernel that took the
    f-number and squared it would be a second definition, 20 % wrong at F/1.0 the day someone
    typed the paraxial form (§2, §8.1).

    ``cos4`` is the native-grid field from `irsim.optics.stage.optics_field` (ones when vignetting
    is disabled) and ``psf`` the kernel from `irsim.optics.psf.optical_psf` at the k× pitch, or
    None for no optical blur. The PSF is cast to float32 for the device while the oracle convolves
    with float64 taps; the taps sum to 1 to ~1e-7 either way, which is a DC gain error two orders
    below the equivalence budget.
    """

    factor: float  # Omega_eff * tau_opt
    area_m2: float
    lb_housing: float
    phi_housing: float  # A_d * Omega_eff * L_B(T_housing), the housing power with no scene
    supersample: int
    cos4: NDArray[np.float32]
    psf: NDArray[np.float32] | None

    def __post_init__(self) -> None:
        if self.supersample < 1:
            raise ValueError("supersample factor must be >= 1")
        if self.psf is not None and (
            self.psf.ndim != 2 or self.psf.shape[0] % 2 == 0 or self.psf.shape[1] % 2 == 0
        ):
            raise ValueError("psf must be 2-D with odd sides")


def optics_terms(
    sensor: SensorSpec,
    lb_housing: float,
    supersample: int | None = None,
    psf: NDArray[np.floating] | None = None,
) -> OpticsTerms:
    """Stage 3's scalars and fields from the sensor spec, exactly as `apply_optics` takes them."""
    k = sensor.optics.supersample_factor if supersample is None else supersample
    f, tau = sensor.optics.f_number, sensor.optics.transmittance
    a_d = sensor.detector_active_area_m2
    return OpticsTerms(
        factor=aperture_factor(f) * tau,
        area_m2=a_d,
        lb_housing=float(lb_housing),
        # A_d Ω_eff L_h: housing_power_field at RI = 0, i.e. with no scene in the cone
        phi_housing=housing_power_field_axis(a_d, f, tau, lb_housing),
        supersample=k,
        cos4=np.ascontiguousarray(optics_field(sensor), dtype=np.float32),
        psf=None if psf is None else np.ascontiguousarray(psf, dtype=np.float32),
    )


def launch_optics(radiance_ss: Any, terms: OpticsTerms, flux: Any, device: str) -> Any:
    """Launch stage 3 on device arrays already resident on ``device``; fills and returns ``flux``.

    Allocates one intermediate only when there is a PSF, and the convolution is out-of-place
    because a pixel's neighbours must not have been overwritten before it is read.
    """
    warp = _require()
    h_ss, w_ss = radiance_ss.shape
    blurred = radiance_ss
    if terms.psf is not None:
        blurred = warp.zeros((h_ss, w_ss), dtype=warp.float32, device=device)
        warp.launch(
            _psf_kernel,
            dim=(h_ss, w_ss),
            inputs=[
                radiance_ss,
                _as_device(warp, terms.psf, warp.float32, device),
                np.int32(terms.psf.shape[0] // 2),
                np.int32(terms.psf.shape[1] // 2),
            ],
            outputs=[blurred],
            device=device,
        )
    warp.launch(
        _optics_kernel,
        dim=flux.shape,
        inputs=[
            blurred,
            _as_device(warp, terms.cos4, warp.float32, device),
            np.float32(terms.factor),
            np.float32(terms.area_m2),
            np.float32(terms.lb_housing),
            np.float32(terms.phi_housing),
            np.int32(terms.supersample),
        ],
        outputs=[flux],
        device=device,
    )
    return flux


def apply_optics_warp(
    radiance_ss: NDArray[np.floating],
    terms: OpticsTerms,
    *,
    device: str = DEFAULT_DEVICE,
) -> NDArray[np.float32]:
    """NumPy in, NumPy out: the GPU twin of `irsim.optics.stage.apply_optics`."""
    warp = _require()
    l_ss = require_fp32_or_better(np.asarray(radiance_ss), "radiance").astype(np.float32)
    if l_ss.ndim != 2:
        raise ValueError("stage 3 works on one (H, W) supersampled plane")
    k = terms.supersample
    h_ss, w_ss = l_ss.shape
    if h_ss % k or w_ss % k:
        raise ValueError(f"shape {(h_ss, w_ss)} is not divisible by the supersample factor {k}")
    native = (h_ss // k, w_ss // k)
    if terms.cos4.shape != native:
        raise ValueError(f"cos4 field {terms.cos4.shape} != detector grid {native}")
    flux = warp.zeros(native, dtype=warp.float32, device=device)
    launch_optics(_as_device(warp, l_ss, warp.float32, device), terms, flux, device)
    return np.asarray(flux.numpy(), dtype=np.float32)


def optics_stage_warp(
    planes: Planes, config: PipelineConfig, state: PipelineState, device: str = DEFAULT_DEVICE
) -> Planes:
    """Stage-3 entry point on the plane dict, the GPU twin of `optics_stage`."""
    terms = optics_terms(
        config.sensor.sensor,
        housing_band_radiance(config, state),
        config.supersample,
        config.psf,
    )
    return {
        "flux": apply_optics_warp(np.asarray(planes["radiance"]), terms, device=device),
    }


class WarpOpticsStage:
    """`irsim.pipeline.core.Stage` implementation running stage 3 on ``device``."""

    name = "optics"

    def __init__(self, device: str = DEFAULT_DEVICE) -> None:
        self.device = device

    def __call__(self, planes: Planes, config: PipelineConfig, state: PipelineState) -> Planes:
        return optics_stage_warp(planes, config, state, self.device)


# ---- stage 4: detector, and the cross-frame state that goes with it ---------------------------

#: Where the device-side companion of a `PipelineState` lives in its ``buffers`` (ADR 0052).
DEVICE_STATE_KEY = "warp_device_state"


class WarpPipelineState:
    """The device-resident cross-frame buffers of one camera on one device (ADR 0052).

    Today that is the bolometer's membrane IIR; M10.7's fixed-pattern drift, bad-pixel map and
    NUC residual join it. The point of a single owner is the reset path: a format change, a cold
    start or a new camera must clear *all* of it together, and a buffer that lives wherever it
    was first needed gets forgotten by exactly one of those.

    The IIR buffer is allocated once and updated in place; ``iir_ptr`` exists so a test can show
    the pointer does not move between frames and the state is never round-tripped to the host.
    """

    def __init__(self, device: str = DEFAULT_DEVICE) -> None:
        self.device = device
        self._iir: Any = None
        self._iir_shape: tuple[int, int] | None = None
        self._fixed: tuple[Any, Any, Any] | None = None
        self._fixed_shape: tuple[int, int] | None = None
        self._xi: tuple[Any, Any] | None = None
        self._xi_shape: tuple[int, int] | None = None
        self._defects: tuple[Any, Any, Any] | None = None
        self._defects_shape: tuple[int, int] | None = None
        self._factory: Any = None
        self._factory_shape: tuple[int, ...] | None = None
        self._held: Any = None
        self._held_shape: tuple[int, int] | None = None

    def iir_state(self, shape: tuple[int, int]) -> tuple[Any, bool]:
        """The IIR buffer for ``shape`` and whether this frame must adopt its input.

        A shape change reallocates and re-adopts rather than raising, because the only way to get
        here with a new shape is a deliberate reconfiguration; the CPU filter raises instead, and
        the difference is deliberate -- see the note in :func:`detector_stage_warp`.
        """
        warp = _require()
        if self._iir is None or self._iir_shape != shape:
            self._iir = warp.zeros(shape, dtype=warp.float32, device=self.device)
            self._iir_shape = shape
            return self._iir, True
        return self._iir, False

    @property
    def iir_ptr(self) -> int | None:
        return None if self._iir is None else int(self._iir.ptr)

    def fixed_pattern(
        self, shape: tuple[int, int], seed_from: FixedPattern
    ) -> tuple[Any, Any, Any]:
        """The device-resident V, H and VH fields, in sigma_TVH = 1 units (M9.4, M10.7a).

        Seeded on first use from the CPU stage's own unit pattern, so both paths start from the
        *same* realisation and only their per-frame draws differ. Statistical equivalence is then
        a measurement of the two generators, not of two unrelated cameras.
        """
        warp = _require()
        if self._fixed is None or self._fixed_shape != shape:
            rows, cols = shape
            if seed_from.shape != shape:
                raise ValueError(f"CPU fixed pattern {seed_from.shape} != device plane {shape}")
            self._fixed = (
                warp.array(
                    np.ascontiguousarray(seed_from.v, dtype=np.float32),
                    dtype=warp.float32,
                    device=self.device,
                ),
                warp.array(
                    np.ascontiguousarray(seed_from.h, dtype=np.float32),
                    dtype=warp.float32,
                    device=self.device,
                ),
                warp.array(
                    np.ascontiguousarray(seed_from.vh, dtype=np.float32),
                    dtype=warp.float32,
                    device=self.device,
                ),
            )
            self._fixed_shape = shape
            del rows, cols
        return self._fixed

    def residual_fields(self, gain_xi: Any, offset_xi: Any) -> tuple[Any, Any]:
        """The NUC residual's two xi fields on device, uploaded from the host model (M9.6).

        Uploaded rather than redrawn: the residual is reset by an FFC event, not by a frame, so
        redrawing it on device would mean reimplementing the epoch counter in a second place and
        having two things to keep in step across a shutter.
        """
        warp = _require()
        shape = (int(gain_xi.shape[0]), int(gain_xi.shape[1]))
        if self._xi is None or self._xi_shape != shape:
            self._xi = (
                warp.zeros(shape, dtype=warp.float32, device=self.device),
                warp.zeros(shape, dtype=warp.float32, device=self.device),
            )
            self._xi_shape = shape
        self._xi[0].assign(np.ascontiguousarray(gain_xi, dtype=np.float32))
        self._xi[1].assign(np.ascontiguousarray(offset_xi, dtype=np.float32))
        return self._xi

    @property
    def fixed_ptrs(self) -> tuple[int, int, int] | None:
        """Pointers to the three fixed buffers, so a test can show they do not move."""
        if self._fixed is None:
            return None
        return tuple(int(b.ptr) for b in self._fixed)  # type: ignore[return-value]

    def defect_buffers(self, bad_map: Any, seed_state: Any) -> tuple[Any, Any, Any]:
        """The defect ``kind`` plane and the two RTS state buffers (M10.7b).

        The map is **uploaded, not redrawn**, so it is bit-identical to the CPU's by construction.
        That is not a shortcut: a bad-pixel map is a property of one physical focal plane, drawn
        once per sensor and never per frame (§10.4), so two independent draws would be two
        different cameras rather than two implementations of one. The RTS state is uploaded once
        too, from the CPU's own starting realisation, and then advances on device -- the same rule
        M10.7a used for the fixed pattern, and for the same reason: what is under test afterwards
        is the two generators, not two unrelated defect populations.

        Two state buffers, ping-ponged, because the chain reads its previous state.
        """
        warp = _require()
        kind = np.ascontiguousarray(bad_map.kind, dtype=np.uint8)
        shape = (int(kind.shape[0]), int(kind.shape[1]))
        if self._defects is None or self._defects_shape != shape:
            self._defects = (
                warp.array(kind, dtype=warp.uint8, device=self.device),
                warp.zeros(shape, dtype=warp.uint8, device=self.device),
                warp.zeros(shape, dtype=warp.uint8, device=self.device),
            )
            self._defects_shape = shape
            self._defects[1].assign(np.ascontiguousarray(seed_state, dtype=np.uint8))
        return self._defects

    def factory_map_buffer(self, bad_map: Any) -> Any:
        """1 where a pixel is on the camera's factory map or is good, 0 for a late defect (SC.19).

        Uploaded once per map and cached, like the kind plane: the map is a property of the
        camera, and a per-frame upload is exactly the host round trip `IG.16` is removing.
        """
        warp = _require()
        on_map = np.ascontiguousarray(~np.asarray(bad_map.late_mask), dtype=np.uint8)
        if self._factory is None or self._factory_shape != on_map.shape:
            self._factory = warp.array(on_map, dtype=warp.uint8, device=self.device)
            self._factory_shape = on_map.shape
        return self._factory

    def swap_rts(self) -> None:
        """Make the freshly written RTS state the current one."""
        if self._defects is not None:
            kind, current, scratch = self._defects
            self._defects = (kind, scratch, current)

    @property
    def defect_ptrs(self) -> tuple[int, int, int] | None:
        if self._defects is None:
            return None
        return tuple(int(b.ptr) for b in self._defects)  # type: ignore[return-value]

    def held_frame(self, shape: tuple[int, int]) -> tuple[Any, bool]:
        """The FFC hold buffer and whether it is new (nothing held yet) -- M9.7 on device.

        One buffer, written on every unfrozen frame and read on every frozen one, so a freeze
        emits the *same bytes* for its whole length. Holding on the host instead would mean a
        round trip per frozen frame to send back a frame the device already had.
        """
        warp = _require()
        if self._held is None or self._held_shape != shape:
            self._held = warp.zeros(shape, dtype=warp.float32, device=self.device)
            self._held_shape = shape
            return self._held, True
        return self._held, False

    @property
    def held_ptr(self) -> int | None:
        return None if self._held is None else int(self._held.ptr)

    def reset(self) -> None:
        """Cold start: drop every device buffer, so the next frame adopts its input again."""
        self._iir = None
        self._iir_shape = None
        self._fixed = None
        self._fixed_shape = None
        self._xi = None
        self._xi_shape = None
        self._defects = None
        self._defects_shape = None
        self._factory = None
        self._factory_shape = None
        self._held = None
        self._held_shape = None


def warp_pipeline_state(state: PipelineState, device: str = DEFAULT_DEVICE) -> WarpPipelineState:
    """The device companion of ``state``, created on first use and kept in its ``buffers``."""
    existing = state.buffers.get(DEVICE_STATE_KEY)
    if isinstance(existing, WarpPipelineState) and existing.device == device:
        return existing
    fresh = WarpPipelineState(device)
    state.buffers[DEVICE_STATE_KEY] = fresh
    return fresh


@dataclass(frozen=True)
class DetectorTerms:
    """Stage 4's per-camera scalars, read off the detector the CPU path uses (M10.6).

    One dataclass for both FPA types because the kernel launch differs only in which scalars it
    carries; ``kind`` selects the path, from ``fpa.type``, never from a heuristic on the data.
    ``alpha`` is `irsim.detector.lowpass.alpha_for` at the configured frame interval -- the blend
    weight 1 − e^{−Δt/τ_th}, which is 0.811 at 60 Hz and τ_th = 10 ms, not the "0.6 frames" of
    §9.2 (spec issue S8).
    """

    kind: str
    dn_max: int
    gain_dn_per_w: float = 0.0
    offset_w: float = 0.0
    alpha: float = 0.0
    qe_t_int: float = 0.0
    offset_e: float = 0.0
    dn_per_electron: float = 0.0

    def __post_init__(self) -> None:
        if self.kind not in ("bolometer", "photon"):
            raise ValueError(f"unknown FPA type {self.kind!r}")
        if self.kind == "bolometer" and not 0.0 < self.alpha <= 1.0:
            raise ValueError(f"IIR blend weight must lie in (0, 1], got {self.alpha}")


def detector_terms(config: PipelineConfig) -> DetectorTerms:
    """Read stage 4's scalars off the configured detector; the transfer stays the oracle's."""
    fpa = config.fpa
    if isinstance(fpa, BolometerParams):
        detector = config.detector
        if not isinstance(detector, MicrobolometerDetector):
            raise TypeError("a bolometer FPA needs a MicrobolometerDetector")
        return DetectorTerms(
            kind="bolometer",
            dn_max=fpa.dn_max,
            gain_dn_per_w=detector.transfer.gain_dn_per_w,
            offset_w=detector.transfer.offset_w,
            alpha=alpha_for(fpa.frame_dt_s, fpa.thermal_time_constant_s),
        )
    detector_q = config.detector
    if not isinstance(detector_q, PhotonDetector):
        raise TypeError("a photon FPA needs a PhotonDetector")
    budget = detector_q.budget
    return DetectorTerms(
        kind="photon",
        dn_max=fpa.dn_max,
        qe_t_int=fpa.quantum_efficiency * fpa.integration_time_s,
        offset_e=budget.dark_electrons + budget.background_electrons,
        dn_per_electron=2**fpa.bit_depth / fpa.well_capacity_e,
    )


def validate_flux(flux: NDArray[np.floating], kind: str) -> NDArray[np.float32]:
    """Refuse what the CPU detector refuses, before anything is uploaded (§9.1, §9.2)."""
    phi = require_fp32_or_better(np.asarray(flux), "flux")
    if np.any(phi < 0.0):
        raise ValueError("pixel power cannot be negative")
    if kind == "bolometer" and np.any(phi > PHOTON_SCALE_GUARD):
        raise ValueError(
            f"pixel power {float(phi.max()):.3e} exceeds {PHOTON_SCALE_GUARD} W: this looks like "
            "a photon rate, not watts -- the bolometer takes energy-form band radiance (§9.2)"
        )
    return phi.astype(np.float32)


def launch_detector(
    flux: Any,
    terms: DetectorTerms,
    device_state: WarpPipelineState,
    signal_dn: Any,
    lagged: Any | None = None,
) -> Any:
    """Launch stage 4 on device arrays; returns the plane the next stage should read.

    ``signal_dn`` receives the ideal transfer and, for a bolometer, ``lagged`` receives the
    membrane's output while the persistent state is updated in place. The state never leaves the
    device: the IIR reads and writes it on the GPU and only ``lagged`` is ever downloaded.
    """
    warp = _require()
    shape = tuple(flux.shape)
    device = device_state.device
    if terms.kind == "bolometer":
        warp.launch(
            _bolometer_signal_kernel,
            dim=shape,
            inputs=[flux, np.float32(terms.gain_dn_per_w), np.float32(terms.offset_w)],
            outputs=[signal_dn],
            device=device,
        )
        state, adopt = device_state.iir_state((int(shape[0]), int(shape[1])))
        out = signal_dn if lagged is None else lagged
        warp.launch(
            _bolometer_lag_kernel,
            dim=shape,
            inputs=[signal_dn, np.float32(terms.alpha), np.int32(1 if adopt else 0), state],
            outputs=[out],
            device=device,
        )
        return out
    warp.launch(
        _photon_signal_kernel,
        dim=shape,
        inputs=[
            flux,
            np.float32(terms.qe_t_int),
            np.float32(terms.offset_e),
            np.float32(terms.dn_per_electron),
        ],
        outputs=[signal_dn],
        device=device,
    )
    return signal_dn


def detector_stage_warp(
    planes: Planes, config: PipelineConfig, state: PipelineState, device: str = DEFAULT_DEVICE
) -> Planes:
    """Stage-4 entry point on the plane dict, the GPU twin of `detector_stage`.

    One deliberate difference from the oracle: a frame whose shape does not match the IIR state
    reallocates and re-adopts here, where `BolometerLowPass` raises. On the CPU a shape change
    mid-sequence is a caller bug worth stopping on; on the device the buffer is owned by
    :class:`WarpPipelineState` and a reconfiguration legitimately replaces it. Feed a sequence of
    one shape and the two paths are identical, which is what the equivalence harness checks.
    """
    warp = _require()
    terms = detector_terms(config)
    phi = validate_flux(np.asarray(planes["flux"]), terms.kind)
    device_state = warp_pipeline_state(state, device)
    signal = warp.zeros(phi.shape, dtype=warp.float32, device=device)
    lagged = (
        warp.zeros(phi.shape, dtype=warp.float32, device=device)
        if terms.kind == "bolometer"
        else None
    )
    out = launch_detector(
        _as_device(warp, phi, warp.float32, device), terms, device_state, signal, lagged
    )
    return {"signal_dn": np.asarray(out.numpy(), dtype=np.float32)}


def quantise_warp(
    signal_dn: NDArray[np.floating], bit_depth: int, *, device: str = DEFAULT_DEVICE
) -> NDArray[np.uint16]:
    """DN = clip(floor(S), 0, 2^bits − 1) on the device, the twin of `irsim.detector.quantise`."""
    warp = _require()
    s = require_fp32_or_better(np.asarray(signal_dn), "signal_dn").astype(np.float32)
    if not np.all(np.isfinite(s)):
        raise ValueError("signal contains NaN or inf")
    dn = warp.zeros(s.shape, dtype=warp.uint16, device=device)
    warp.launch(
        _quantise_kernel,
        dim=s.shape,
        inputs=[_as_device(warp, s, warp.float32, device), np.int32(dn_max_for_bits(bit_depth))],
        outputs=[dn],
        device=device,
    )
    return np.asarray(dn.numpy(), dtype=np.uint16)


class WarpDetectorStage:
    """`irsim.pipeline.core.Stage` implementation running stage 4 on ``device``."""

    name = "detector"

    def __init__(self, device: str = DEFAULT_DEVICE) -> None:
        self.device = device

    def __call__(self, planes: Planes, config: PipelineConfig, state: PipelineState) -> Planes:
        return detector_stage_warp(planes, config, state, self.device)


#: (CPU oracle, GPU twin) per stage name — the equivalence harness parametrises over this, and
#: every later Warp stage registers here so it is compared the same way. The plane each stage
#: is compared on is `EQUIVALENCE_OUTPUT`, because stage 3 leaves ``flux``, not ``radiance``.
EQUIVALENCE_STAGES: dict[str, tuple[Any, Any]] = {
    "band_radiance": (band_radiance_stage, band_radiance_stage_warp),
    "atmosphere": (atmosphere_stage, atmosphere_stage_warp),
    "optics": (optics_stage, optics_stage_warp),
    "detector": (detector_stage, detector_stage_warp),
}

#: Which plane each registered stage produces.
EQUIVALENCE_OUTPUT: dict[str, str] = {
    "band_radiance": "radiance",
    "atmosphere": "radiance",
    "optics": "flux",
    "detector": "signal_dn",
}


# -- stage 5: the 3-D noise, the OU drift and the NUC residual on device (M10.7a) ------------


@dataclass(frozen=True)
class NoiseTerms:
    """Stage 5's per-camera scalars, read off the same `NoiseStage` the CPU path uses.

    Sigmas are in DN and are derived from the frame's own σ_TVH, exactly as
    ``irsim.noise.NoiseStage.apply`` derives them, so the two paths cannot drift apart through a
    second copy of the ratio vector. The fixed terms are not here: they are *buffers*, live on
    the device across frames, and belong to :class:`WarpPipelineState`.
    """

    sigma_tvh: float
    sigma_t: float
    sigma_tv: float
    sigma_th: float
    sigma_v: float
    sigma_h: float
    sigma_vh: float

    @classmethod
    def from_stage(cls, stage: NoiseStage, sigma_tvh: float) -> "NoiseTerms":
        # Quoted: this module has no `from __future__ import annotations` (Warp reads the
        # real annotation objects off its kernels), so a self-reference must be a string.
        sig = stage.sigmas(sigma_tvh)
        return cls(
            sigma_tvh=float(sigma_tvh),
            sigma_t=float(sig.t),
            sigma_tv=float(sig.tv),
            sigma_th=float(sig.th),
            sigma_v=float(sig.v),
            sigma_h=float(sig.h),
            sigma_vh=float(sig.vh),
        )


def noise_terms(config: PipelineConfig, sigma_tvh: float) -> "NoiseTerms":
    """Stage 5's scalars for this frame's σ_TVH; the ratio vector stays the config's."""
    if not np.isfinite(sigma_tvh) or sigma_tvh < 0.0:
        raise ValueError(f"sigma_tvh must be finite and non-negative, got {sigma_tvh}")
    return NoiseTerms.from_stage(config.noise, float(sigma_tvh))


def launch_noise(
    signal_dn: Any,
    fixed: tuple[Any, Any, Any],
    terms: "NoiseTerms",
    frame_index: int,
    sensor_seed: int,
    out: Any,
    device: str,
) -> Any:
    """Add the seven §10.2 components to a device signal plane."""
    warp = _require()
    rows, cols = signal_dn.shape
    warp.launch(
        _noise_kernel,
        dim=(rows, cols),
        inputs=[
            signal_dn,
            fixed[0],
            fixed[1],
            fixed[2],
            warp.float32(terms.sigma_tvh),
            warp.float32(terms.sigma_t),
            warp.float32(terms.sigma_tv),
            warp.float32(terms.sigma_th),
            warp.int32(int(frame_index)),
            warp.int32(int(sensor_seed)),
            warp.int32(int(cols)),
            out,
        ],
        device=device,
    )
    return out


def launch_ou_drift(
    fixed: tuple[Any, Any, Any],
    sigmas: tuple[float, float, float],
    dt_s: float,
    tau_s: float,
    epoch: int,
    sensor_seed: int,
    device: str,
) -> None:
    """Advance the device-resident V, H and VH fields by one exact OU step, in place (M9.4).

    ``tau_s`` of infinity freezes them and launches nothing, so a frozen pattern costs nothing and
    stays bit-identical -- the same control case the CPU path has.
    """
    warp = _require()
    if not math.isfinite(tau_s):
        return
    if not (dt_s > 0.0):
        return
    decay = math.exp(-float(dt_s) / float(tau_s))
    innovation = math.sqrt(max(0.0, 1.0 - decay * decay))
    v, h, vh = fixed
    for lane, (buf, sigma) in enumerate(((v, sigmas[0]), (h, sigmas[1]))):
        if sigma <= 0.0:
            continue
        warp.launch(
            _ou_step_kernel_1d,
            dim=buf.shape[0],
            inputs=[
                buf,
                warp.float32(sigma),
                warp.float32(decay),
                warp.float32(innovation),
                warp.int32(int(epoch)),
                warp.int32(int(sensor_seed)),
                warp.int32(lane * 101),
            ],
            device=device,
        )
    if sigmas[2] > 0.0:
        rows, cols = vh.shape
        warp.launch(
            _ou_step_kernel_2d,
            dim=(rows, cols),
            inputs=[
                vh,
                warp.float32(sigmas[2]),
                warp.float32(decay),
                warp.float32(innovation),
                warp.int32(int(epoch)),
                warp.int32(int(sensor_seed)),
                warp.int32(303),
                warp.int32(int(cols)),
            ],
            device=device,
        )


def launch_nuc_residual(
    signal_dn: Any,
    xi: tuple[Any, Any],
    gain_ppm_per_k: float,
    offset_mk_per_k: float,
    dn_per_k: float,
    delta_t_fpa_k: float,
    out: Any,
    device: str,
    reference: Any = None,
) -> Any:
    """Apply M9.6's residual on device; the scales are computed here, exactly as on the host.

    ``reference`` is the shutter frame (SC.18) as a device array, or None for zeros.
    """
    warp = _require()
    rows, cols = signal_dn.shape
    if reference is None:
        reference = warp.zeros((rows, cols), dtype=warp.float32, device=device)
    gain_scale = float(gain_ppm_per_k) * 1e-6 * float(delta_t_fpa_k)
    offset_scale = float(offset_mk_per_k) * 1e-3 * float(delta_t_fpa_k) * float(dn_per_k)
    warp.launch(
        _nuc_residual_kernel,
        dim=(rows, cols),
        inputs=[
            signal_dn,
            xi[0],
            xi[1],
            reference,
            warp.float32(gain_scale),
            warp.float32(offset_scale),
            out,
        ],
        device=device,
    )
    return out


def noise_stage_warp(
    planes: Planes, config: PipelineConfig, state: PipelineState, device: str = DEFAULT_DEVICE
) -> Planes:
    """Stage 5 on the plane dict: ``signal_dn`` in, ``signal_dn`` out (the Warp twin of M4.4).

    The fixed pattern comes from the device buffers of :class:`WarpPipelineState`, seeded on first
    use from the CPU stage's own unit pattern so the two paths start from the *same* realisation
    and only their per-frame draws differ. That is what makes "statistical equivalence" a
    measurement of the generator rather than of two unrelated cameras.
    """
    warp = _require()
    signal = require_fp32_or_better(np.asarray(planes["signal_dn"]), "signal_dn")
    shape = (int(signal.shape[0]), int(signal.shape[1]))
    sigma_tvh = float(np.asarray(planes["sigma_dn"], dtype=np.float64).mean())
    terms = noise_terms(config, sigma_tvh)

    device_state = warp_pipeline_state(state, device)
    fixed = device_state.fixed_pattern(shape, config.noise.unit_fixed)
    src = _as_device(warp, signal.astype(np.float32), warp.float32, device)
    out = warp.zeros(shape, dtype=warp.float32, device=device)
    launch_noise(src, fixed, terms, state.frame_index, config.sensor_seed, out, device)
    return {"signal_dn": np.asarray(out.numpy(), dtype=np.float32)}


# -- stage 6: the display branch on device (M10.8) -------------------------------------------


def launch_histogram(dn: Any, bit_depth: int, device: str) -> Any:
    """Atomic histogram of a device DN plane into 2^bit_depth int32 bins."""
    warp = _require()
    rows, cols = dn.shape
    counts = warp.zeros(dn_max_for_bits(bit_depth) + 1, dtype=warp.int32, device=device)
    warp.launch(_histogram_kernel, dim=(rows, cols), inputs=[dn, counts], device=device)
    return counts


def agc_lut_warp(
    dn: Any, isp: IspSpec, bit_depth: int, device: str
) -> tuple[Any, NDArray[np.float32]]:
    """The AGC as a 2^bit_depth lookup table, built on device from a device histogram (§11.3).

    Both §11.3 modes are *monotone functions of DN alone*, so each is exactly a table -- which is
    what makes them portable to a kernel at all, and is why the device path can be held to ±1
    display code rather than to a resemblance. The table is built once per frame and applied by
    one lookup per pixel.

    The histogram and the plateau clip run on device (atomics, then ``wp.utils.array_scan`` for the
    exclusive CDF), and the table is then **finished on the host**.

    **What that costs, stated accurately.** An earlier version of this docstring said "the last few
    scalars are read back". That is not what happens: ``counts.numpy()`` pulls the *whole*
    ``2**bit_depth`` histogram across the bus, plateau mode pulls a second array of the same size
    for the inclusive CDF, and the finished table goes back the other way -- three transfers of
    65536 entries per frame at 16 bits, not a few scalars. Each one is a **synchronisation point**,
    so the frame cannot overlap with anything.

    It is still a deliberate trade rather than an oversight -- the percentile search and the
    occupied-bin span are irregular reductions that would each be more code than the kernel they
    feed -- but it is a real per-frame stall and the honest justification is "not yet worth the
    kernels", not "only scalars move". The device-side version is a roadmap step; the loop in
    :func:`replace_bad_pixels_warp` is the worse offender and is named there too.
    """
    warp = _require()
    import warp.utils as wputils

    n_bins = dn_max_for_bits(bit_depth) + 1
    counts = launch_histogram(dn, bit_depth, device)
    host_counts = counts.numpy().astype(np.float64)
    n_pixels = float(host_counts.sum())

    if isp.agc == "none":
        # The top 8 bits over 255, not a linear rescale: `agc: none` is a bit shift (ADR 0031),
        # and the two differ by up to half a display code on every pixel.
        shift = bit_depth - 8
        table = (
            (np.arange(n_bins, dtype=np.uint32) >> np.uint32(shift)) / np.float32(255.0)
        ).astype(np.float32)
    elif isp.agc == "linear":
        lo, hi = isp.clip_percentiles
        x_lo = percentile_from_histogram(host_counts, lo)
        x_hi = percentile_from_histogram(host_counts, hi)
        if x_hi <= x_lo or int(np.floor(x_hi)) == int(np.floor(x_lo)):
            table = np.full(n_bins, CONSTANT_FRAME_LEVEL, dtype=np.float32)
        else:
            bins = np.arange(n_bins, dtype=np.float64)
            table = np.clip((bins - x_lo) / (x_hi - x_lo), 0.0, 1.0).astype(np.float32)
    elif isp.agc == "plateau_equalization":
        clipped = warp.zeros(n_bins, dtype=warp.float32, device=device)
        warp.launch(
            _clip_counts_kernel,
            dim=n_bins,
            inputs=[counts, warp.float32(isp.plateau * n_pixels), clipped],
            device=device,
        )
        inclusive = warp.zeros(n_bins, dtype=warp.float32, device=device)
        wputils.array_scan(clipped, inclusive, inclusive=True)
        cdf_incl = inclusive.numpy().astype(np.float64)
        cdf_excl = np.concatenate(([0.0], cdf_incl[:-1]))
        occupied = np.flatnonzero(host_counts)
        if occupied.size == 0:
            table = np.full(n_bins, CONSTANT_FRAME_LEVEL, dtype=np.float32)
        else:
            span_lo, span_hi = cdf_excl[occupied[0]], cdf_excl[occupied[-1]]
            if span_hi <= span_lo:
                table = np.full(n_bins, CONSTANT_FRAME_LEVEL, dtype=np.float32)
            else:
                table = np.clip((cdf_excl - span_lo) / (span_hi - span_lo), 0.0, 1.0).astype(
                    np.float32
                )
    else:  # pragma: no cover - the schema restricts the literal
        raise ValueError(f"unknown agc mode {isp.agc!r}")

    if isp.gamma != 1.0:
        table = np.power(table, np.float32(1.0 / isp.gamma), dtype=np.float32)
    device_table = warp.array(np.ascontiguousarray(table), dtype=warp.float32, device=device)
    return device_table, table


def launch_display(dn: Any, isp: IspSpec, bit_depth: int, device: str) -> tuple[Any, Any]:
    """DN16 on device → (y in [0, 1], RGBA8), in ADR 0031's fixed order."""
    warp = _require()
    rows, cols = dn.shape
    table, _ = agc_lut_warp(dn, isp, bit_depth, device)
    y = warp.zeros((rows, cols), dtype=warp.float32, device=device)
    warp.launch(_apply_lut_kernel, dim=(rows, cols), inputs=[dn, table, y], device=device)

    if isp.dde_gain > 0.0:
        sharpened = warp.zeros((rows, cols), dtype=warp.float32, device=device)
        warp.launch(
            _dde_kernel,
            dim=(rows, cols),
            inputs=[
                y,
                warp.float32(isp.dde_gain),
                warp.int32(rows),
                warp.int32(cols),
                sharpened,
            ],
            device=device,
        )
        y = sharpened

    lut = np.ascontiguousarray(palette_table(isp.palette), dtype=np.uint8)
    palette = warp.array2d(lut, dtype=warp.uint8, device=device)
    rgba = warp.zeros((rows, cols, 4), dtype=warp.uint8, device=device)
    warp.launch(
        _palette_kernel,
        dim=(rows, cols),
        inputs=[y, palette, warp.int32(1 if isp.polarity == "black_hot" else 0), rgba],
        device=device,
    )
    return y, rgba


def display_stage_warp(
    planes: Planes, config: PipelineConfig, state: PipelineState, device: str = DEFAULT_DEVICE
) -> Planes:
    """Stage 6 on the plane dict: ``dn16`` in, ``display8`` and ``y`` out (M10.8).

    ``dn16`` is preserved untouched. The radiometric branch and the display branch fork *after*
    the ADC (§11.1), so a change of AGC mode must not move a single DN code -- which is exactly
    what the equivalence test checks, because an AGC that reached back into the linear output
    would be invisible in the picture and fatal to the validation.
    """
    del state
    warp = _require()
    dn = np.asarray(planes["dn16"])
    if dn.dtype != np.uint16:
        raise TypeError(f"dn16 must be a uint16 plane, got {dn.dtype}")
    bit_depth = config.sensor.sensor.fpa.bit_depth
    src = warp.array2d(np.ascontiguousarray(dn), dtype=warp.uint16, device=device)
    y, rgba = launch_display(src, config.sensor.sensor.isp, bit_depth, device)
    return {
        "dn16": dn,
        "y": np.asarray(y.numpy(), dtype=np.float32),
        "display8": np.asarray(rgba.numpy(), dtype=np.uint8),
    }


# -- M10.7b: defects, the iterated replacement and the FFC hold on device ----------------------


@dataclass(frozen=True)
class DefectTerms:
    """The per-camera scalars the defect kernels need, read off the same spec the CPU uses."""

    dn_max: int
    amplitude_dn: int
    p_enter: float
    p_leave: float

    def __post_init__(self) -> None:
        for name in ("p_enter", "p_leave"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be a probability, got {value}")


def defect_terms(sensor: SensorSpec, dn_max: int) -> DefectTerms:
    """Derive the RTS switch probabilities exactly as ``irsim.noise.defects.advance_state`` does.

    Detailed balance, in one place: the stationary occupancy and the mean dwell are what the
    config authors, and the two switch probabilities follow. Computing them here rather than
    passing them in keeps the device path from acquiring its own idea of what the chain is.
    """
    noise = sensor.noise
    occupancy = float(noise.bad_pixel_rts_occupancy)
    dwell = float(noise.bad_pixel_rts_dwell_frames)
    p_leave = 1.0 / dwell
    p_enter = p_leave * occupancy / (1.0 - occupancy)
    if not 0.0 <= p_enter <= 1.0:
        raise ValueError(
            f"bad_pixel_rts_occupancy {occupancy} with dwell {dwell} frames implies a switch "
            f"probability of {p_enter:.3f}; lower the occupancy or raise the dwell"
        )
    return DefectTerms(
        dn_max=int(dn_max),
        amplitude_dn=int(round(float(noise.bad_pixel_rts_amplitude_dn))),
        p_enter=p_enter,
        p_leave=p_leave,
    )


def launch_rts_step(
    device_state: "WarpPipelineState",
    terms: DefectTerms,
    frame_index: int,
    sensor_seed: int,
    device: str = DEFAULT_DEVICE,
) -> Any:
    """Advance the RTS chain one frame on device and return the now-current state buffer."""
    warp = _require()
    assert device_state._defects is not None, "call defect_buffers() first"
    kind, current, scratch = device_state._defects
    rows, cols = int(kind.shape[0]), int(kind.shape[1])
    warp.launch(
        _rts_step_kernel,
        dim=(rows, cols),
        inputs=[
            kind,
            current,
            np.float32(terms.p_enter),
            np.float32(terms.p_leave),
            np.int32(frame_index),
            np.int32(sensor_seed),
            np.int32(cols),
        ],
        outputs=[scratch],
        device=device,
    )
    device_state.swap_rts()
    assert device_state._defects is not None
    return device_state._defects[1]


def launch_defects(
    signal: Any,
    kind: Any,
    rts_bad: Any,
    terms: DefectTerms,
    out: Any,
    active: Any,
    device: str = DEFAULT_DEVICE,
    on_map: Any = None,
) -> None:
    """Inject the defects and write the replacement mask the replacement will consume.

    ``on_map`` is `WarpPipelineState.factory_map_buffer`; None means every defect is on the map
    (no late defects), which is the pre-SC.19 behaviour.
    """
    warp = _require()
    rows, cols = int(signal.shape[0]), int(signal.shape[1])
    if on_map is None:
        on_map = warp.full((rows, cols), 1, dtype=warp.uint8, device=device)
    warp.launch(
        _apply_defects_kernel,
        dim=(rows, cols),
        inputs=[
            signal,
            kind,
            rts_bad,
            on_map,
            np.int32(terms.dn_max),
            np.int32(terms.amplitude_dn),
        ],
        outputs=[out, active],
        device=device,
    )


def launch_replacement(
    signal: Any,
    active: Any,
    out: Any,
    device: str = DEFAULT_DEVICE,
    max_passes: int = REPLACEMENT_MAX_PASSES,
) -> int:
    """Iterated 4-neighbour replacement on device; returns the number of passes it took.

    The loop lives on the host and reads back two counters per pass. That is two tiny transfers
    for a stencil that finishes in two or three passes on any map ADR 0055's cluster process can
    produce -- and the alternative, a fixed pass count, would either waste passes or silently
    leave a cluster unfilled, which is the failure `replace_bad_pixels` refuses to make quietly.
    """
    warp = _require()
    rows, cols = int(signal.shape[0]), int(signal.shape[1])
    work = warp.zeros((rows, cols), dtype=warp.float64, device=device)
    work_next = warp.zeros((rows, cols), dtype=warp.float64, device=device)
    valid = warp.zeros((rows, cols), dtype=warp.uint8, device=device)
    valid_next = warp.zeros((rows, cols), dtype=warp.uint8, device=device)
    counters = warp.zeros(2, dtype=warp.int32, device=device)

    warp.launch(
        _replace_init_kernel,
        dim=(rows, cols),
        inputs=[signal, active],
        outputs=[work, valid],
        device=device,
    )

    passes = 0
    for _ in range(int(max_passes)):
        counters.zero_()
        warp.launch(
            _replace_pass_kernel,
            dim=(rows, cols),
            inputs=[work, valid, np.int32(rows), np.int32(cols)],
            outputs=[work_next, valid_next, counters[0:1], counters[1:2]],
            device=device,
        )
        work, work_next = work_next, work
        valid, valid_next = valid_next, valid
        passes += 1
        # A full device synchronisation per pass, to read two integers. This is the sharper of the
        # module's two host readbacks: the histogram stalls once per frame, this stalls once per
        # *iteration*. The loop condition belongs in a kernel -- an unfilled-count flag the passes
        # write and a launch bound that does not depend on reading it back. Recorded as a roadmap
        # step rather than done here, because a device kernel cannot be verified without a GPU and
        # the CPU reference leads (ADR 0018).
        remaining, filled = (int(v) for v in counters.numpy())
        if remaining == 0:
            break
        if filled == 0:
            raise ValueError(
                "a masked region has no valid neighbour on any side and cannot be interpolated "
                "(a fully masked row, column or border block)"
            )
    else:
        raise ValueError(
            f"a defect cluster was still unfilled after {max_passes} passes; the bad-pixel map "
            "has a region far larger than the §10.4 cluster process should produce"
        )

    warp.launch(
        _replace_finish_kernel, dim=(rows, cols), inputs=[work], outputs=[out], device=device
    )
    return passes


def launch_ffc_hold(
    signal: Any,
    device_state: "WarpPipelineState",
    freezing: bool,
    out: Any,
    device: str = DEFAULT_DEVICE,
) -> bool:
    """§11.2's freeze on a device buffer; returns whether a held frame was emitted.

    A freeze that begins before anything has been held passes the frame through and starts
    holding it, exactly as `FfcController.process` does -- inventing a frame the camera never saw
    would put a synthetic first frame into every clip that opens on a shutter event.
    """
    warp = _require()
    shape = (int(signal.shape[0]), int(signal.shape[1]))
    held, fresh = device_state.held_frame(shape)
    if freezing and not fresh:
        warp.copy(out, held)
        return True
    warp.copy(held, signal)
    warp.copy(out, signal)
    return False


def defects_stage_warp(
    planes: Planes,
    config: PipelineConfig,
    state: PipelineState,
    device: str = DEFAULT_DEVICE,
    *,
    freezing: bool = False,
) -> Planes:
    """§11.1's post-ADC half on device: RTS step, defects, replacement, then the FFC hold.

    The chain's *schedule* stays on the host -- when the shutter fires is frame arithmetic, and
    duplicating `FfcController` on the device would be a second thing to keep in step. What moves
    here is the per-pixel work and the held buffer, which are the parts that would otherwise cost
    a round trip per frame.
    """
    warp = _require()
    chain = config.chain
    if chain is None:
        raise ValueError("defects_stage_warp needs a SensorChain attached (attach_sensor_chain)")
    signal = require_fp32_or_better(np.asarray(planes["signal_dn"]), "signal_dn")
    shape = (int(signal.shape[0]), int(signal.shape[1]))
    sensor = config.sensor.sensor
    terms = defect_terms(sensor, dn_max_for_bits(sensor.fpa.bit_depth))

    device_state = warp_pipeline_state(state, device)
    kind, _, _ = device_state.defect_buffers(
        chain.bad_pixels, np.asarray(chain.defect_state.bad, dtype=np.uint8)
    )
    rts_bad = launch_rts_step(device_state, terms, state.frame_index, config.sensor_seed, device)

    src = _as_device(warp, signal.astype(np.float32), warp.float32, device)
    defective = warp.zeros(shape, dtype=warp.float32, device=device)
    active = warp.zeros(shape, dtype=warp.uint8, device=device)
    on_map = device_state.factory_map_buffer(chain.bad_pixels)
    launch_defects(src, kind, rts_bad, terms, defective, active, device, on_map=on_map)

    replaced = warp.zeros(shape, dtype=warp.float32, device=device)
    passes = launch_replacement(defective, active, replaced, device)

    out = warp.zeros(shape, dtype=warp.float32, device=device)
    held = launch_ffc_hold(replaced, device_state, freezing, out, device)
    return {
        "signal_dn": np.asarray(out.numpy(), dtype=np.float32),
        "defects_active": np.asarray(active.numpy(), dtype=np.uint8),
        "replacement_passes": np.asarray([passes], dtype=np.int32),
        "ffc_held": np.asarray([held], dtype=np.bool_),
    }
