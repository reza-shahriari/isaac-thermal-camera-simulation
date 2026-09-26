"""NUC residual: what survives the two-point correction (M9.6).

docs/physics-model.md §11.2 ("Model the residual, not the ideal"), §2 (g_ij, o_ij), §12.2 ``nuc``.
ADR 0053 (the split), ADR 0056 (the mK → DN conversion reference).

A two-point correction is calibrated at one focal-plane temperature and applied at another. What
it leaves behind is this: a small per-pixel gain error and a small per-pixel offset error, both
proportional to ΔT_FPA = T_FPA − T_FPA^cal, both reset when the shutter closes.

    g_ij(ΔT) = 1 + ppm · 1e-6 · ΔT · ξ_g,      o_ij(ΔT) = mK/K · 1e-3 · ΔT · (∂DN/∂T) · ξ_o

with ξ_g, ξ_o ~ N(0, 1) drawn once per sensor and redrawn at each flat-field correction. At
ΔT = 0 — immediately after an FFC — the gain is exactly 1 and the offset exactly 0, by
construction rather than by cancellation.

**This is the only ΔT_FPA-driven mechanism (ADR 0053).** The detector's *raw* gain/offset(T_FPA)
polynomials (M9.2) describe the uncorrected FPA and are parts in 10³ per kelvin; they are what the
NUC removes. Applying both to a corrected image would count the same physics twice, and the result
would look like a plausible drift rate rather than a bug. Likewise M9.4's OU drift is *stationary*
and so contributes no growth between FFC events — all of that growth is here.

**The offset is authored in millikelvin and applied in DN (non-negotiable #3, ADR 0056).**
Datasheets and the literature quote residual non-uniformity in temperature units because that is
what a user sees, so the config keeps those units. But a residual that were *added* in kelvin would
be wrong everywhere except the temperature it was specified at: ∂L/∂T rises steeply with
temperature, so a flat 90 mK error at 300 K is a much smaller apparent-temperature error at 373 K.
The conversion therefore happens **once**, at construction, through ``∂DN/∂T`` evaluated at the
reference temperature — and from then on the residual is a fixed number of DN. Its
apparent-temperature equivalent then falls with scene temperature exactly as the derivative ratio
says it should, which is what ``test_residual_not_kelvin_flat`` pins.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from irsim.config.sensor import NucSpec
from irsim.noise.seeding import NoiseStream, field_normal, stream_key

__all__ = ["NucResidual", "RESIDUAL_REFERENCE_K"]

#: The temperature ``∂DN/∂T`` is evaluated at (ADR 0056). 300 K is where §9.4 anchors NETD and
#: where datasheet residual figures are quoted, so the config's millikelvin means the same thing
#: here as it does on the datasheet it came from.
RESIDUAL_REFERENCE_K = 300.0

Float32Array = NDArray[np.float32]


@dataclass
class NucResidual:
    """The per-pixel gain and offset a two-point NUC fails to remove (§11.2).

    ``dn_per_k`` is ``∂DN/∂T`` at :data:`RESIDUAL_REFERENCE_K`, supplied by the caller from the
    camera's radiometric calibration (``irsim.isp.dn_per_kelvin``). It is passed in rather than
    computed here so that ``irsim.noise`` does not depend on ``irsim.isp``, and so that the
    "converted once" rule is visible at the call site instead of buried.
    """

    nuc: NucSpec
    shape: tuple[int, int]
    dn_per_k: float
    sensor_seed: int
    ffc_index: int = 0
    _gain_field: Float32Array = field(init=False, repr=False)
    _offset_field: Float32Array = field(init=False, repr=False)

    def __post_init__(self) -> None:
        rows, cols = int(self.shape[0]), int(self.shape[1])
        if rows <= 0 or cols <= 0:
            raise ValueError(f"shape must be positive, got {self.shape}")
        self.shape = (rows, cols)
        if not np.isfinite(self.dn_per_k) or self.dn_per_k <= 0.0:
            raise ValueError(
                f"dn_per_k must be a positive finite ∂DN/∂T at {RESIDUAL_REFERENCE_K} K, "
                f"got {self.dn_per_k}"
            )
        self._redraw()

    # -- the two random fields ------------------------------------------------------------
    def _redraw(self) -> None:
        """Draw ξ_g and ξ_o for the current FFC epoch (independent streams, ADR 0022)."""
        self._gain_field = field_normal(
            stream_key(self.sensor_seed, self.ffc_index, NoiseStream.NUC_GAIN), self.shape
        )
        self._offset_field = field_normal(
            stream_key(self.sensor_seed, self.ffc_index, NoiseStream.NUC_OFFSET), self.shape
        )

    def ffc_reset(self) -> None:
        """The shutter closed: recalibrate. New ξ fields, and ΔT_FPA restarts from zero.

        A *new* draw, not a scaling of the old one: a fresh two-point calibration lands on a
        different realisation of the pixel-to-pixel spread, so the pattern after an FFC is
        uncorrelated with the pattern before it. That decorrelation is the visible signature of
        the event — the fixed pattern does not fade, it is replaced.
        """
        self.ffc_index += 1
        self._redraw()

    def reset(self, frame_index: int) -> None:
        """The :class:`~irsim.isp.ffc.Resettable` hook M9.7's controller calls on a shutter event.

        ``frame_index`` is accepted and ignored: the residual's epochs are counted by FFC, not by
        frame, so that a sequence replayed at a different frame rate gets the same patterns.
        """
        del frame_index
        self.ffc_reset()

    @property
    def gain_field(self) -> Float32Array:
        """ξ_g, the unit-variance per-pixel gain draw of the current FFC epoch."""
        return self._gain_field

    @property
    def offset_field(self) -> Float32Array:
        """ξ_o, the unit-variance per-pixel offset draw of the current FFC epoch."""
        return self._offset_field

    # -- the residual itself --------------------------------------------------------------
    def gain(self, delta_t_fpa_k: float) -> Float32Array:
        """g_ij(ΔT) = 1 + ppm·1e-6·ΔT·ξ_g. Exactly 1 everywhere at ΔT = 0."""
        scale = float(self.nuc.residual_gain_ppm_per_k) * 1e-6 * float(delta_t_fpa_k)
        return np.asarray(1.0 + scale * self._gain_field, dtype=np.float32)

    def offset_dn(self, delta_t_fpa_k: float) -> Float32Array:
        """o_ij(ΔT) in **DN**: mK/K → K/K → DN through ∂DN/∂T, converted once (ADR 0056)."""
        scale = (
            float(self.nuc.residual_offset_mk_per_k)
            * 1e-3
            * float(delta_t_fpa_k)
            * float(self.dn_per_k)
        )
        return np.asarray(scale * self._offset_field, dtype=np.float32)

    def offset_sigma_mk(self, delta_t_fpa_k: float) -> float:
        """The offset residual's standard deviation in millikelvin *at the reference temperature*.

        Useful for comparing against a datasheet figure, and for the tests. Away from
        :data:`RESIDUAL_REFERENCE_K` the same DN corresponds to a different number of millikelvin
        — which is the whole point of converting once.
        """
        return abs(float(self.nuc.residual_offset_mk_per_k) * float(delta_t_fpa_k))

    def apply(
        self,
        dn: NDArray[np.floating],
        delta_t_fpa_k: float,
        reference_dn: NDArray[np.floating] | None = None,
    ) -> Float32Array:
        """Apply the residual to a NUC-corrected signal plane (§2's g_ij, o_ij; §11.2).

        Operates on the un-quantised float signal in DN units, after the ideal two-point operator:
        what comes out is what a real camera's correction leaves. float16 is refused
        (non-negotiable #2) and the input is not modified.

        ``reference_dn`` is the closed-shutter frame of the last FFC (SC.18). The offset was
        re-measured on it, so a gain error acts on the signal *relative to the shutter*:
        ref + g (x − ref) + o. A scene at the shutter's radiance carries no gain residual at
        all, and a clear sky tens of kelvin below it carries the most. None keeps the
        pre-SC.18 form g x + o, which is the same thing with the shutter at zero signal.
        """
        arr = np.asarray(dn)
        if arr.dtype == np.float16:
            raise TypeError("signal is float16; apply the NUC residual in float32 or better")
        if not np.issubdtype(arr.dtype, np.floating):
            raise TypeError(f"expected a float signal plane in DN units, got {arr.dtype}")
        if arr.shape != self.shape:
            raise ValueError(f"signal shape {arr.shape} != residual shape {self.shape}")
        x = arr.astype(np.float32)
        if reference_dn is None:
            out = x * self.gain(delta_t_fpa_k) + self.offset_dn(delta_t_fpa_k)
            return np.asarray(out, dtype=np.float32)
        ref = np.asarray(reference_dn)
        if ref.dtype == np.float16:
            raise TypeError("reference_dn is float16 (non-negotiable #2)")
        if ref.shape != self.shape:
            raise ValueError(f"reference shape {ref.shape} != residual shape {self.shape}")
        r = ref.astype(np.float32)
        out = r + (x - r) * self.gain(delta_t_fpa_k) + self.offset_dn(delta_t_fpa_k)
        return np.asarray(out, dtype=np.float32)
