"""Within-frame motion smear: the scene moving across the detector while it integrates (§8.3, §9.2).

The MTF cascade has carried ``mtf_motion`` -- |sinc(v t_int xi)| for linear image-plane motion --
since M5, and nothing applied it. A rendered frame was therefore sharp no matter how fast the
scene crossed it, which is not a small omission: an aircraft at 250 m and 150 m/s sweeps the
boresight at 34 degrees per second, and a Boson pixel is 0.049 degrees, so the scene crosses **11
pixels in one frame period**. A detector trained on unsmeared synthetic imagery has never seen the
thing that most obviously distinguishes real thermal video of a moving target.

This is the *spatial* operator for that term. It is separate from, and additional to, the
bolometer's frame-to-frame lag: the membrane IIR (§9.2) carries a target's history across
successive frames, while this is the blur laid down **within** a single integration.

**The two detector families smear differently, and that is the physics, not a setting.** A
microbolometer has no shutter and no integration time -- `integration_time_ms` is `None` for one,
deliberately -- because it integrates continuously, so the scene smears over the *whole frame
period*. A cooled photon detector integrates for a short window inside the frame and is idle for
the rest, so it smears only over that window and is correspondingly sharper. This is the
mechanism behind the §16 checklist line "lateral motion smears LWIR, not cooled MWIR".

**A bolometer's smear is not a box (ADR 0077 amendment).** The membrane is a first-order system,
so the reading at the frame's instant weights the flux of ``s`` seconds earlier by
``e^(-s/tau_th) / tau_th``. The part older than one frame period is the frame-to-frame IIR
(:mod:`irsim.detector.lowpass`); the part *within* the frame is this operator, and it carries the
same exponential, truncated to ``[0, T]`` -- a bright head at the feature's current position and a
fading tail behind it, not a flat streak the length of the frame's travel. With
``decay_frames = tau_th / T`` this operator lays the within-frame part down; ``None`` keeps the
centred box, which is right for a photon detector's shutter window. §9.2 has always said the
bolometer smear *is* the tau_th response; the box spread the Boson's smear evenly over the whole
16.7 ms, about twice as wide at the half-energy point as the membrane.

**Spatially varying, necessarily.** A single convolution would be wrong for the scenes this
simulator is built for: under a tracking mount the target is stationary on the focal plane and the
sky sweeps past it, so one kernel cannot serve both. Each pixel is averaged along **its own**
motion vector, which costs a gather per tap.

docs/physics-model.md §8.3, §9.2
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "MAX_SMEAR_TAPS",
    "NEGLIGIBLE_SMEAR_PX",
    "apply_motion_smear",
    "smear_decay_frames",
    "smear_duty",
]

#: Below this much movement the smear is a sub-pixel effect the detector's own box filter already
#: covers, and the operator returns its input untouched rather than paying for a no-op gather.
NEGLIGIBLE_SMEAR_PX = 0.25

#: Ceiling on the number of samples along a smear segment. One tap per pixel of travel resolves
#: the boxcar exactly; beyond this the extra taps buy sub-percent accuracy for linear cost, and a
#: scene moving faster than this in one frame is past the point where a boxcar is the right model
#: anyway (the target has left the field).
MAX_SMEAR_TAPS = 65


def smear_duty(frame_period_s: float, integration_time_s: float | None) -> float:
    """Fraction of a frame period the detector is integrating, in [0, 1].

    ``None`` means a **bolometer**: no shutter, no integration window, continuously sensitive, so
    the answer is 1 and the scene smears over the whole frame. Treating a missing integration time
    as zero would silently make every uncooled camera in this repository sharper than it is, which
    is the wrong direction -- an unsmeared LWIR frame looks better and is less real.
    """
    if frame_period_s <= 0.0:
        raise ValueError("frame_period_s must be positive")
    if integration_time_s is None:
        return 1.0
    if integration_time_s < 0.0:
        raise ValueError("integration_time_s must be non-negative")
    return min(float(integration_time_s) / float(frame_period_s), 1.0)


def smear_decay_frames(
    frame_period_s: float,
    integration_time_s: float | None,
    thermal_time_constant_s: float | None,
) -> float | None:
    """``tau_th / T`` for a bolometer, ``None`` for a detector with a shutter window.

    A photon FPA (an integration time) smears as a box over its window, so it gets ``None``. A
    bolometer (no integration time) smears with its membrane's exponential, so it gets its time
    constant in frame periods -- 0.48 for the Boson's 8 ms at 60 Hz.
    """
    if frame_period_s <= 0.0:
        raise ValueError("frame_period_s must be positive")
    if integration_time_s is not None:
        return None
    if thermal_time_constant_s is None or not thermal_time_constant_s > 0.0:
        raise ValueError("a bolometer needs a positive thermal time constant to smear with")
    return float(thermal_time_constant_s) / float(frame_period_s)


def apply_motion_smear(
    image: Any,
    motion_px: Any,
    duty: float,
    *,
    max_taps: int = MAX_SMEAR_TAPS,
    decay_frames: float | None = None,
) -> NDArray[np.float64]:
    """Average each pixel along its own motion vector over the integration window.

    ``image`` is ``(H, W)`` and ``motion_px`` is ``(H, W, 2)`` in **pixels of this grid** per
    frame -- so on a supersampled grid it must be the supersampled displacement, which is what
    :func:`irsim.optics.motion.image_plane_motion` produces when given supersampled intrinsics.

    The segment is **centred** on the pixel, running from -s/2 to +s/2 along the motion direction
    with s = |v| * duty. Centred rather than trailing because the pixel's reading is the mean of
    what crossed it during the window, and the position the frame is labelled with is the middle
    of that window; a trailing segment would displace every moving feature by half its smear, a
    shift that looks like a timing error and is one.

    **With ``decay_frames`` it is the bolometer's kernel instead**: trailing, not centred, with
    tap weights ``e^(-u / decay_frames)`` integrated over each tap's slice ``u`` of the frame
    (``u = 0`` now, ``u = 1`` one frame ago) and normalised. A bolometer's reading really does lag
    its scene -- the mean delay is the membrane's, not a timing error -- so the head sits where the
    feature is and the tail lies behind it, along the direction it came from.

    Returns float64. The caller's dtype discipline applies at the stage boundary, not here.
    """
    img = np.asarray(image, dtype=np.float64)
    if img.ndim != 2:
        raise ValueError(f"image must be (H, W), got {img.shape}")
    motion = np.asarray(motion_px, dtype=np.float64)
    if motion.shape != (*img.shape, 2):
        raise ValueError(f"motion_px must be {(*img.shape, 2)}, got {motion.shape}")
    if not 0.0 <= duty <= 1.0:
        raise ValueError("duty must lie in [0, 1]")

    length = np.hypot(motion[..., 0], motion[..., 1]) * float(duty)
    longest = float(length.max(initial=0.0))
    if longest < NEGLIGIBLE_SMEAR_PX:
        return img

    from scipy.ndimage import map_coordinates

    taps = int(min(max(2, math.ceil(longest) + 1), max_taps))
    # Unit vectors where there is motion; zero elsewhere, so a still pixel samples itself at every
    # tap and comes back exactly unchanged rather than nearly so.
    with np.errstate(invalid="ignore", divide="ignore"):
        direction = np.where(
            length[..., None] > 0.0,
            motion / np.hypot(motion[..., 0], motion[..., 1])[..., None],
            0.0,
        )
    direction = np.nan_to_num(direction, nan=0.0, posinf=0.0, neginf=0.0)

    if decay_frames is not None and not decay_frames > 0.0:
        raise ValueError("decay_frames must be positive")
    rows, cols = np.indices(img.shape, dtype=np.float64)
    total = np.zeros_like(img)
    if decay_frames is not None:
        # Slice edges in frame fractions, the exact exponential mass of each slice, placed at the
        # slice's midpoint. `motion_px` is the displacement over the last frame, positive along
        # travel, so what this pixel saw u of a frame ago is now u * length further along.
        edges = np.arange(taps + 1, dtype=np.float64) / taps
        mass = np.exp(-edges[:-1] / decay_frames) - np.exp(-edges[1:] / decay_frames)
        mass = mass / mass.sum()
        for u, weight in zip((edges[:-1] + edges[1:]) / 2.0, mass, strict=True):
            shift = u * length
            total += weight * map_coordinates(
                img,
                [rows + shift * direction[..., 1], cols + shift * direction[..., 0]],
                order=1,
                mode="nearest",
                prefilter=False,
            )
        return np.asarray(total)
    # Midpoint rule: taps at the centres of `taps` equal sub-intervals, not at the segment's
    # endpoints. Endpoint sampling looks natural and is wrong -- N taps spanning length s sit a
    # distance s/(N-1) apart, so the comb implements a boxcar of length s + s/(N-1), and the
    # measured MTF comes out as the Dirichlet kernel of that longer smear rather than sinc(s f).
    # Measured before the fix: 0.7182 against an expected 0.7842 at 6 px and 1/16 cyc/px, which
    # is exactly Dirichlet_7 -- the operator was self-consistent and describing the wrong smear.
    for offset in (np.arange(taps) + 0.5) / taps - 0.5:
        shift = offset * length
        # motion_px is (vx, vy) with x across columns and y down rows, per the G-buffer contract.
        sample_rows = rows + shift * direction[..., 1]
        sample_cols = cols + shift * direction[..., 0]
        total += map_coordinates(
            img, [sample_rows, sample_cols], order=1, mode="nearest", prefilter=False
        )
    return np.asarray(total / taps)
