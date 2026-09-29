"""The visible companion's exposure: sub-frame accumulation over the infrared's own window (IG.19).

docs/physics-model.md §8.3 (MTF_motion), §9.2; ADR 0073 (the companion frame), ADR 0077 (the
within-frame smear). The infrared frame smears a moving edge over the detector's integration
window -- the whole frame period for a bolometer, ``integration_time_ms`` for a photon FPA
(:func:`~irsim.optics.smear.smear_duty`) -- and the companion RGB frame was rendered sharp, at
one instant. The owner reads the pair together, and a target blurred in one band and crisp in
the other is a defect of the pair, not a property of the scene.

The companion is therefore accumulated over the same window: the renderer poses the moving
geometry at ``n`` instants spread evenly across the window, centred on the frame's own instant,
and the frames are averaged. That is what a camera's exposure does -- integrate the scene while
it moves -- and it needs no motion vectors, no post-effect and no assumption about the motion
being linear; the pose callback the driver supplies is the trajectory, whatever it is. The
infrared planes are still taken at the frame's instant, and the SC.28 kernel does the smearing
there, so both bands describe one window. Everything here is engine-free: the Isaac glue only
loops, poses, renders and reads.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from irsim.config.sensor import SensorSpec
from irsim.optics.smear import smear_duty

__all__ = ["accumulate_subframes", "exposure_window_s", "subframe_offsets_s"]


def exposure_window_s(sensor: SensorSpec, frame_period_s: float) -> float:
    """How long the detector integrates within one frame period, seconds.

    ``smear_duty × frame_period``: the whole period for a bolometer (no shutter), the
    integration time for a photon FPA, never more than the period.
    """
    if frame_period_s <= 0.0:
        raise ValueError("frame_period_s must be positive")
    integration = (
        None
        if sensor.fpa.integration_time_ms is None
        else float(sensor.fpa.integration_time_ms) * 1e-3
    )
    return float(smear_duty(frame_period_s, integration)) * float(frame_period_s)


def subframe_offsets_s(window_s: float, n: int) -> NDArray[np.float64]:
    """``n`` instants across a window of ``window_s`` centred on 0: the midpoints of ``n`` equal
    slices, so the accumulation is the midpoint rule on the exposure and its mean instant is the
    frame's own. One sub-frame is the frame's instant itself; a zero window is ``n`` copies of it.
    """
    if n < 1:
        raise ValueError("n must be at least 1")
    if window_s < 0.0:
        raise ValueError("window_s cannot be negative")
    k = np.arange(n, dtype=np.float64)
    return np.asarray((k + 0.5) / n * window_s - 0.5 * window_s, dtype=np.float64)


def accumulate_subframes(frames: Sequence[Any]) -> NDArray[np.float32]:
    """The mean of the sub-frames, float32 (CLAUDE.md #2), summed in float64.

    A single frame comes back as itself, and identical frames come back bit-identical to one of
    them: a static scene must not move by a rounding error just because it was exposed in pieces.
    """
    if not frames:
        raise ValueError("no sub-frames to accumulate")
    arrays = [np.asarray(f) for f in frames]
    shape = arrays[0].shape
    for a in arrays:
        if a.shape != shape:
            raise ValueError(f"sub-frames differ in shape: {a.shape} vs {shape}")
        if a.dtype == np.float16:
            raise TypeError("a float16 sub-frame has no place in the radiance path (CLAUDE.md #2)")
    if len(arrays) == 1 or all(np.array_equal(a, arrays[0]) for a in arrays[1:]):
        return np.asarray(arrays[0], dtype=np.float32)
    total = np.zeros(shape, dtype=np.float64)
    for a in arrays:
        total += a.astype(np.float64)
    return np.asarray(total / len(arrays), dtype=np.float32)
