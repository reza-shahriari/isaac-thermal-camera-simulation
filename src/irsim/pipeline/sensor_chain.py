"""The M9 sensor chain: every piece of camera imperfection, in §11.1's order (M9.8).

docs/physics-model.md §11.1 (the ISP order), §11.2, §10.3, §10.4, §9.2, §8.2.
ADR 0058 (the temporal filter, and what is *not* applied here), ADR 0052, ADR 0053.

M9.1–M9.7 each built one mechanism and tested it alone. This assembles them, and the assembly is
where the real risk lives: every one of these mechanisms makes the image drift or speckle, and it
is easy to end up with two of them describing the same physics. The budget this chain is built to
satisfy is deliberately small — on a uniform scene, post-correction spatial noise is

    σ_spatial = √(σ_V² + σ_H² + σ_VH²  ⊕  σ_residual²)

and **nothing else**. Two mechanisms, not four. ``tests/unit/test_sensor_chain.py`` asserts exactly
that, because a chain that summed four plausible-looking contributions would still produce a
plausible-looking image.

Per-frame order, following §11.1 and the positions the ADRs fixed:

  1. ``T_housing`` (M9.3) → the optics self-emission of stage 3. Owned here, read by ``run_frame``.
  2. ``T_FPA`` (M9.2) → ΔT_FPA for the NUC residual, via the FFC controller which owns ΔT_eff.
  3. the breathing fixed pattern (M9.4) advances one frame.
  4. *(stages 3–5 happen in ``run_frame``: optics, detector + membrane IIR, 3-D noise, ADC)*
  5. defects injected on the raw DN plane (M9.5a).
  6. bad-pixel replacement (M9.5b) — §11.1's first ISP stage.
  7. the NUC residual (M9.6) on the corrected signal.
  8. the temporal filter — **identity** (ADR 0058).
  9. the FFC freeze (M9.7): if the shutter is closed, the whole frame is the held one.

**What is deliberately not applied: the raw gain/offset(T_FPA) polynomials (ADR 0053, ADR 0058).**
M9.2's ``gain_of_t``/``offset_of_t`` describe the *uncorrected* focal plane and are parts in 10³
per kelvin. In a chain that also applies a two-point NUC, they and their correction cancel to
within the residual — so computing both would mean forming a large number and subtracting almost
all of it back, in float32, to obtain something the residual model already gives directly. The
polynomials stay available for an uncorrected-FPA bench; they do not run here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from irsim.config.sensor import SensorSpec
from irsim.detector.fpa_thermal import FpaThermalModel
from irsim.isp.bad_pixel import replace_bad_pixels
from irsim.isp.ffc import FfcController, FfcEvent
from irsim.isp.nuc import TwoPointNuc
from irsim.noise.defects import (
    BadPixelMap,
    DefectState,
    active_defect_mask,
    advance_state,
    apply_defects,
    replacement_mask,
)
from irsim.noise.defects import generate_map as generate_defect_map
from irsim.noise.drift import DriftingPattern, FpnDrift, drift_rng
from irsim.noise.nuc_residual import NucResidual
from irsim.noise.three_d import FixedPattern, Sigmas7
from irsim.optics.housing import HousingTemperature

__all__ = ["SensorChain", "FrameReport", "attach_sensor_chain"]


@dataclass(frozen=True)
class FrameReport:
    """What the chain did to one frame; carried out for benches and for §15 Tier 3 tests."""

    t_housing_k: float
    t_fpa_k: float
    delta_t_fpa_k: float
    ffc: FfcEvent
    defects_active: int


@dataclass
class SensorChain:
    """All M9 state for one camera, advanced together so nothing runs on a stale clock.

    Construct with :meth:`build`. ``defects_enabled`` and ``residual_enabled`` exist so a bench can
    isolate one mechanism; both default on, because a camera has them.
    """

    sensor: SensorSpec
    housing: HousingTemperature
    fpa_node: FpaThermalModel | None
    ffc: FfcController
    residual: NucResidual
    bad_pixels: BadPixelMap
    defect_state: DefectState
    pattern: DriftingPattern
    sensor_seed: int
    defects_enabled: bool = True
    residual_enabled: bool = True
    #: The closed-shutter frame of the last FFC, noiseless DN (SC.18); None before the first.
    shutter_dn: NDArray[np.float32] | None = None
    #: The display flat field with its offset re-measured on that shutter; None = factory.
    display_nuc: TwoPointNuc | None = None
    _drift_rng: np.random.Generator = field(init=False, repr=False)
    _last_t_s: float | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self._drift_rng = drift_rng(self.sensor_seed)
        self.ffc.register(self.residual, self.pattern)

    # -- construction ---------------------------------------------------------------------
    @classmethod
    def build(
        cls,
        sensor: SensorSpec,
        dn_per_k: float,
        sensor_seed: int,
        weather: object | None = None,
        ambient_provider: object | None = None,
        t0_s: float = 0.0,
        defects_enabled: bool = True,
        residual_enabled: bool = True,
    ) -> SensorChain:
        """Assemble the chain a sensor config describes.

        ``weather`` is the scene's single ``WeatherSeries`` (CLAUDE.md #6); pass it whenever the
        housing or FPA node is not ``fixed``. ``dn_per_k`` is ∂DN/∂T at 300 K from
        ``irsim.isp.dn_per_kelvin`` — converted once, at the call site, as ADR 0056 requires.
        """
        shape = sensor.fpa_shape
        housing = HousingTemperature.from_optics(
            sensor.optics,
            weather=weather,  # type: ignore[arg-type]
            ambient_provider=ambient_provider,  # type: ignore[arg-type]
            t0_s=t0_s,
        )
        fpa_node = None
        if sensor.fpa.fpa_temp_mode is not None:
            fpa_node = FpaThermalModel(
                mode=sensor.fpa.fpa_temp_mode,
                tau_s=sensor.fpa.fpa_tau_s,
                self_heating_k=sensor.fpa.fpa_self_heating_k,
                fixed_temp_k=sensor.fpa.fpa_temp_k,
                weather=weather,  # type: ignore[arg-type]
                ambient_provider=ambient_provider,  # type: ignore[arg-type]
            )
            fpa_node.settle(t0_s)

        # Unit sigmas (sigma_TVH = 1), the same convention NoiseStage stores its static
        # pattern in: the stage scales by the frame's actual sigma_TVH at apply time, so the
        # drift must breathe in the same units or the ratios would be scaled twice.
        unit_sigmas = Sigmas7.from_ratios(1.0, sensor.noise.sigma_ratios())
        drift = FpnDrift(tau_s=sensor.noise.fpn_drift_tau_s, sigmas=unit_sigmas)
        bad_pixels = generate_defect_map(shape, sensor.noise, sensor_seed)
        return cls(
            sensor=sensor,
            housing=housing,
            fpa_node=fpa_node,
            ffc=FfcController(nuc=sensor.nuc, fps=sensor.fpa.frame_rate_hz),
            residual=NucResidual(
                nuc=sensor.nuc, shape=shape, dn_per_k=dn_per_k, sensor_seed=sensor_seed
            ),
            bad_pixels=bad_pixels,
            defect_state=DefectState.initial(bad_pixels, sensor.noise, sensor_seed),
            pattern=DriftingPattern.start(drift, shape, sensor_seed),
            sensor_seed=sensor_seed,
            defects_enabled=defects_enabled,
            residual_enabled=residual_enabled,
        )

    # -- the shutter (SC.18) ---------------------------------------------------------------
    def needs_shutter_frame(self, frame_index: int) -> bool:
        """True when this frame closes the shutter: at power-up and at every scheduled FFC.

        A shuttered core flat-fields when it starts, so the first frame takes a snapshot even
        though `FfcController.fires_on` never fires there (no freeze at frame 0, M9.7). The other
        two modes have no shutter to look at.
        """
        if not self.ffc.shutters:
            return False
        return self.shutter_dn is None or self.ffc.fires_on(frame_index)

    def record_shutter(
        self, shutter_dn: NDArray[np.floating], factory_nuc: TwoPointNuc | None
    ) -> None:
        """Keep the shutter frame for the residual and re-measure the display offset on it.

        §11.2 (SC.18): the offset snapshot. ``factory_nuc`` is the configured display flat field;
        without one the display has no gain map to refresh, and only the residual's reference
        changes.
        """
        sh = np.asarray(shutter_dn)
        if sh.dtype == np.float16:
            raise TypeError("shutter frame is float16 (non-negotiable #2)")
        self.shutter_dn = np.asarray(sh, dtype=np.float32)
        self.display_nuc = None if factory_nuc is None else factory_nuc.refreshed(self.shutter_dn)

    # -- the per-frame clock --------------------------------------------------------------
    def begin_frame(self, t_s: float, frame_index: int) -> tuple[float, float]:
        """Advance the thermal nodes and the drift to ``t_s``; return (T_housing, T_FPA).

        Called before stage 3, because the housing temperature is what the optics self-emission
        needs. The frame interval is measured from the previous call rather than taken from the
        frame rate, so a bench that steps at an irregular cadence gets the right drift.
        """
        dt_s = 0.0 if self._last_t_s is None else max(0.0, float(t_s) - self._last_t_s)
        self._last_t_s = float(t_s)

        t_housing = self.housing.at(t_s)
        t_fpa = t_housing
        if self.fpa_node is not None:
            t_fpa = self.fpa_node.step(t_s, dt_s) if dt_s > 0.0 else self.fpa_node.settle(t_s)

        if dt_s > 0.0:
            self.pattern.advance(dt_s, self._drift_rng)
            self.defect_state = advance_state(
                self.defect_state, self.bad_pixels, self.sensor.noise, self.sensor_seed, frame_index
            )
            self.ffc.update_delta_t(t_fpa, dt_s)
        return t_housing, t_fpa

    @property
    def fixed_pattern(self) -> FixedPattern:
        """The breathing pattern stage 5 adds this frame (M9.4)."""
        return self.pattern.pattern

    # -- the post-ADC half ----------------------------------------------------------------
    def finish_frame(
        self,
        signal_dn: NDArray[np.floating],
        dn_max: int,
        frame_index: int,
        t_fpa_k: float,
    ) -> tuple[NDArray[np.float32], FrameReport]:
        """Steps 5-9 above: defects, replacement, residual, temporal filter, freeze.

        Takes and returns the **un-quantised** signal in DN units, so the radiometric branch keeps
        its sub-LSB meaning; the defects themselves are applied on the quantised plane, which is
        where §11.1 puts them and where "stuck at the floor" is exact.
        """
        signal = np.asarray(signal_dn, dtype=np.float32)
        active = np.zeros(signal.shape, dtype=bool)

        if self.defects_enabled and self.bad_pixels.count:
            active = active_defect_mask(self.bad_pixels, self.defect_state)
            # the camera's map, not the truth: late defects stay in the image (SC.19, §10.4)
            replaced = replacement_mask(self.bad_pixels, self.defect_state)
            quantised = np.clip(np.floor(signal), 0, dn_max).astype(np.uint16)
            defective = apply_defects(
                quantised,
                self.bad_pixels,
                self.defect_state,
                dn_max,
                self.sensor.noise.bad_pixel_rts_amplitude_dn,
            )
            # Carry the defect back onto the float plane only where it actually bit, so the
            # un-quantised signal keeps full precision everywhere else.
            changed = defective != quantised
            signal = signal.copy()
            signal[changed] = defective[changed].astype(np.float32)
            if replaced.any():
                signal = replace_bad_pixels(signal, replaced).astype(np.float32)

        delta_t = self.ffc.delta_t_eff_k
        if self.residual_enabled:
            signal = self.residual.apply(signal, delta_t, self.shutter_dn)

        signal = self._temporal_filter(signal)

        out, event = self.ffc.process(signal, frame_index, t_fpa_k=t_fpa_k)
        return np.asarray(out, dtype=np.float32), FrameReport(
            t_housing_k=self.housing.temperature_k,
            t_fpa_k=float(t_fpa_k),
            delta_t_fpa_k=float(delta_t),
            ffc=event,
            defects_active=int(np.count_nonzero(active)),
        )

    def _temporal_filter(self, signal: NDArray[np.float32]) -> NDArray[np.float32]:
        """§11.1's unnamed "temporal filter": the identity, deliberately (ADR 0058).

        §11.1 lists a stage between the NUC and the AGC and never says what it does. Rather than
        invent a coefficient, this stays an identity until ME.5's temporal PSD on flat sky shows
        whether real cores low-pass their output at all. Inventing one would change σ_TVH -- the
        quantity NETD is measured from -- by an amount nobody could later distinguish from a
        detector-model error.
        """
        return signal


def attach_sensor_chain(
    config: Any,
    weather: object | None = None,
    ambient_provider: object | None = None,
    t0_s: float = 0.0,
    defects_enabled: bool | None = None,
    residual_enabled: bool | None = None,
) -> Any:
    """Return a copy of ``config`` with an M9 chain attached (M9.8).

    ``defects_enabled`` and ``residual_enabled`` default to ``None``, meaning *take the value from
    the sensor config's own* ``fidelity:`` *block* (ME.8, `bad_pixels` and `nuc_residual`), so an
    ablation variant has its own config hash instead of living in a caller's argument list. A bool
    overrides it, for a bench that wants one mechanism isolated without authoring a config.

    The ∂DN/∂T the NUC residual needs is taken from the camera's own radiometric calibration and
    evaluated **once**, here, at :data:`~irsim.noise.nuc_residual.RESIDUAL_REFERENCE_K` — ADR 0056's
    rule, enforced by there being exactly one place in the pipeline that does it.

    Attaching is explicit rather than automatic because ``chain=None`` is a meaningful
    configuration: it is the ideal camera that every M9 mechanism is measured against, and the one
    the radiometric goldens describe.
    """
    from dataclasses import replace

    from irsim.isp.radiometric import dn_per_kelvin
    from irsim.noise.nuc_residual import RESIDUAL_REFERENCE_K

    if config.calibration is None:
        raise ValueError(
            "the M9 chain needs a radiometric calibration to convert the NUC residual's "
            "millikelvin into DN (ADR 0056); this config has none (a photon FPA -- see M11.6)"
        )
    fidelity = config.sensor.sensor.fidelity
    if defects_enabled is None:
        defects_enabled = fidelity.bad_pixels
    if residual_enabled is None:
        residual_enabled = fidelity.nuc_residual
    dn_per_k = dn_per_kelvin(config.calibration, config.lut, RESIDUAL_REFERENCE_K)
    chain = SensorChain.build(
        config.sensor.sensor,
        dn_per_k=dn_per_k,
        sensor_seed=config.sensor_seed,
        weather=weather,
        ambient_provider=ambient_provider,
        t0_s=t0_s,
        defects_enabled=defects_enabled,
        residual_enabled=residual_enabled,
    )
    return replace(config, chain=chain)
