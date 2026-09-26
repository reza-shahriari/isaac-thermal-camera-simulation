"""Two-point non-uniformity correction (docs/physics-model.md §11.2).

    DN_corr = G_ij (DN_ij − O_ij)
    G_ij = (mean DN_H − mean DN_L) / (DN_H_ij − DN_L_ij),   O_ij = DN_L_ij

calibrated against two blackbodies. ``mode: ideal`` (§12.2) means exactly these coefficients
with no residual: a linear FPA is corrected perfectly, and the corrected cold blackbody reads
**0** -- the NUC level convention of ADR 0021, so the radiometric branch's DN ↔ L calibration and
this operator share a reference. The residual model (coefficients calibrated at one FPA
temperature and applied at another, growth between shutter events, the FFC freeze) is M9.

A pure per-frame operator: the coefficient tables are explicit state a wrapper keeps in
``PipelineState`` (ADR 0014: an SPG port cannot hold them). Float32 throughout; float16 refused.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = ["TwoPointNuc"]

Float32Array = NDArray[np.float32]


def _as_signal(x: object, what: str) -> NDArray[np.float64]:
    arr = np.asarray(x)
    if arr.dtype == np.float16:
        raise TypeError(f"{what} is float16 (non-negotiable #2)")
    if arr.ndim != 2:
        raise ValueError(f"{what} must be (H, W)")
    return arr.astype(np.float64)


@dataclass(frozen=True)
class TwoPointNuc:
    """Per-pixel gain and offset tables (float32)."""

    gain: Float32Array
    offset: Float32Array
    #: Level added back after the correction. Zero keeps §11.2's convention exactly -- the
    #: corrected cold blackbody reads 0, sharing a reference with the radiometric branch
    #: (ADR 0021). A *display* flat field sets it to the cold frame's mean instead, so the
    #: corrected plane keeps the DN range the AGC downstream was written for; without it a
    #: flat-fielded frame reads zero on a cold scene and the histogram moves for reasons that
    #: have nothing to do with the scene.
    pedestal: float = 0.0

    def __post_init__(self) -> None:
        if self.gain.dtype != np.float32 or self.offset.dtype != np.float32:
            raise TypeError("NUC coefficient tables must be float32")
        if self.gain.shape != self.offset.shape:
            raise ValueError("gain and offset must share a shape")

    @classmethod
    def calibrate(
        cls, dn_low: object, dn_high: object, *, restore_pedestal: bool = False
    ) -> TwoPointNuc:
        """§11.2: G_ij = (mean DN_H − mean DN_L)/(DN_H_ij − DN_L_ij), O_ij = DN_L_ij.

        ``restore_pedestal`` adds the cold frame's mean level back on application; see
        :attr:`pedestal`.
        """
        lo = _as_signal(dn_low, "dn_low")
        hi = _as_signal(dn_high, "dn_high")
        if lo.shape != hi.shape:
            raise ValueError("the two blackbody frames must share a shape")
        span = hi - lo
        if np.any(span <= 0.0):
            raise ValueError(
                "every pixel must respond more to the hot blackbody than to the cold one"
            )
        gain = (hi.mean() - lo.mean()) / span
        return cls(
            gain=gain.astype(np.float32),
            offset=lo.astype(np.float32),
            pedestal=float(lo.mean()) if restore_pedestal else 0.0,
        )

    @classmethod
    def identity(cls, shape: tuple[int, int]) -> TwoPointNuc:
        return cls(gain=np.ones(shape, np.float32), offset=np.zeros(shape, np.float32))

    def refreshed(self, shutter_dn: object) -> TwoPointNuc:
        """The same gain, with the offset re-measured on a closed shutter (§11.2, SC.18).

        A shutter event is a one-point offset update: whatever the correction leaves on a frame of
        the closed shutter, minus its mean, is taken out of the offset, so that frame reads
        uniform. The gain is the factory's, so what survives afterwards is the housing's drift
        since the event and any gain-map error scaled by scene minus shutter -- the two radial
        terms of §11.2. The mean level is untouched, so the AGC downstream sees the same range.
        """
        sh = _as_signal(shutter_dn, "shutter_dn")
        if sh.shape != self.gain.shape:
            raise ValueError(f"shutter frame {sh.shape} != coefficient shape {self.gain.shape}")
        corrected = self.apply(sh).astype(np.float64)
        excess = corrected - corrected.mean()
        offset = self.offset.astype(np.float64) + excess / self.gain.astype(np.float64)
        return TwoPointNuc(gain=self.gain, offset=offset.astype(np.float32), pedestal=self.pedestal)

    def apply(self, dn: object) -> Float32Array:
        """DN_corr = G (DN − O) + pedestal, float32; with pedestal 0 the cold blackbody reads 0."""
        x = _as_signal(dn, "dn")
        if x.shape != self.gain.shape:
            raise ValueError(f"frame shape {x.shape} != coefficient shape {self.gain.shape}")
        corrected = self.gain.astype(np.float64) * (x - self.offset.astype(np.float64))
        return np.asarray(corrected + float(self.pedestal), dtype=np.float32)
