"""AGC temporal damping: the transfer function carries over from frame to frame (SC.10).

Every AGC operator in :mod:`irsim.isp.agc` and :mod:`irsim.isp.information` is a pure function of
one frame, so a hot target entering the field re-maps the whole display in a single frame. A real
core does not: it runs an IIR filter on its transfer function,

    T_shown(t) = a · T_shown(t − 1) + (1 − a) · T_frame(t),

with ``a`` the vendor's damping control -- the Boson's Damping Factor ``df / 100`` (0 updates
immediately, 100 freezes the histogram and the transfer), the Lepton's HEQ Dampening Factor
``N / 256`` ("(N/256) · previous + ((256 − N)/256) · current", default 64), both [R51] and the
Lepton IDD. So a target entering frame gives a transient over a few frames rather than a
one-frame step in the brightness of everything else, which is what the owner's reference clips
show and the per-frame operators could not.

**The coefficient is per native frame, and the elapsed time decides how many of those passed.**
A camera filters at its own rate whether or not anybody records every frame, so a render that
captures one frame every six seconds of scene time (ADR 0074) has let a 60 Hz core run 360 filter
steps between captures: ``a_eff = a ** (Δt · f_frame)``, which at the Boson's 0.85 is 1e-26 --
the time-lapse is undamped, as it should be, and continuous video is damped at the camera's
rate. A caller that never advances its clock gets one native step per frame.

The state is the damped transfer itself, a float64 table over the 2^bit_depth DN bins, kept in
``PipelineState.buffers`` under :data:`TRANSFER_KEY` (ADR 0052: one owner, one reset path --
drop the key and the next frame adopts its own transfer, as a camera does after power-up).

docs/physics-model.md §11.3; ADR 0170
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "TRANSFER_KEY",
    "TRANSFER_T_KEY",
    "TransferDamper",
    "damping_over",
]

#: Where the damped transfer lives in ``PipelineState.buffers``.
TRANSFER_KEY = "agc_transfer"

#: Where the scene time of the frame that last updated it is kept.
TRANSFER_T_KEY = "agc_transfer_t_s"


def damping_over(factor: float, frame_rate_hz: float, dt_s: float) -> float:
    """The IIR coefficient over ``dt_s`` of a core that applies ``factor`` once per native frame.

    ``factor`` 1 is a frozen transfer however long the interval (the Boson's df = 100); 0 is no
    memory at all.
    """
    if not 0.0 <= factor <= 1.0:
        raise ValueError(f"damping factor must lie in [0, 1], got {factor}")
    if frame_rate_hz <= 0.0:
        raise ValueError("frame_rate_hz must be positive")
    if dt_s <= 0.0:
        raise ValueError("dt_s must be positive")
    if factor in (0.0, 1.0):
        return float(factor)
    return float(factor ** (dt_s * frame_rate_hz))


class TransferDamper:
    """The IIR on the transfer: call it with this frame's table, get the table to apply.

    Passed to an AGC operator as its ``lut_hook``; the operator indexes the frame with whatever
    comes back, so the damping reaches every pixel and, for the information-based mode, the
    slope the detail layer is added back at.

    The first call adopts its input: a core that has been looking at the scene is settled on it.
    """

    def __init__(self, coefficient: float, previous: Any = None) -> None:
        if not 0.0 <= coefficient <= 1.0:
            raise ValueError(f"coefficient must lie in [0, 1], got {coefficient}")
        self.coefficient = float(coefficient)
        self.previous: NDArray[np.float64] | None = (
            None if previous is None else np.asarray(previous, dtype=np.float64)
        )

    def __call__(self, lut: Any) -> NDArray[np.float64]:
        current: NDArray[np.float64] = np.asarray(lut, dtype=np.float64)
        out: NDArray[np.float64]
        if self.previous is None or self.previous.shape != current.shape:
            out = current.copy()
        else:
            a = self.coefficient
            out = np.asarray(a * self.previous + (1.0 - a) * current, dtype=np.float64)
        self.previous = out
        return out
