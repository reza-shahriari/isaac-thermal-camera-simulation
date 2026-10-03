"""Stage 3 on the plane dict — the adaptor `irsim.optics.stage.apply_optics` is reached through.

Stages 1 and 2 are `(planes, config, state) -> planes` functions (`irsim.pipeline.radiance`,
`irsim.pipeline.atmosphere`); stage 3 was only ever called positionally from `run_frame`, which
left the Warp twin (M10.5) with no oracle of the same shape to be compared against. This module
supplies one. It computes nothing: the physics is `apply_optics` -- PSF at the k× pitch, box-mean
downsample to the detector grid, × π τ_opt/(4F²+1) · cos⁴θ · A_d, + Φ_self, in that fixed order
(ADR 0020) -- and the only thing added here is where the housing radiance comes from, which is
the same lookup `run_frame` does: the LUT at ``state.housing_temp_k``, in the pipeline's quantity.

``radiance`` arrives on the supersampled grid and ``flux`` leaves on the detector grid, so this is
the one stage that changes the shape of the planes.

docs/physics-model.md §8.1-§8.3, §13.4 stage 3
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from irsim.config.sensor import SensorSpec
from irsim.optics.smear import smear_decay_frames, smear_duty
from irsim.optics.stage import apply_optics
from irsim.pipeline.core import PipelineConfig, PipelineState, Planes

__all__ = ["housing_band_radiance", "optics_stage", "OpticsStage"]


def housing_band_radiance(config: PipelineConfig, state: PipelineState) -> float:
    """L_B(T_housing) now, in the pipeline's quantity — the level Φ_self is built on (§8.2)."""
    return float(config.lut.lookup(state.housing_temp_k, config.quantity)[()])


def motion_for_integration(planes: Planes, sensor: SensorSpec) -> NDArray[np.float64] | None:
    """The G-buffer's ``motion_px`` scaled by how much of the frame the detector integrates.

    ``None`` when the plane is absent, which is every scene that has not asked for motion: the
    G-buffer treats ``motion_px`` as optional (M0.6) and a still scene should cost nothing.

    The duty is where the two detector families part. A bolometer has no integration window --
    ``integration_time_ms`` is ``None`` for one on purpose -- so it smears over the whole frame
    period; a cooled photon detector integrates briefly and is idle for the rest, so it smears
    less and comes out sharper. That is §16's "lateral motion smears LWIR, not cooled MWIR".
    """
    motion = planes.get("motion_px")
    if motion is None:
        return None
    duty = smear_duty(
        1.0 / float(sensor.fpa.frame_rate_hz),
        None
        if sensor.fpa.integration_time_ms is None
        else float(sensor.fpa.integration_time_ms) * 1e-3,
    )
    return np.asarray(motion, dtype=np.float64) * duty


def motion_decay(sensor: SensorSpec) -> float | None:
    """The bolometer's ``tau_th / T`` for the smear kernel; ``None`` for a photon FPA's box.

    The frame period is the detector's own, ``1 / frame_rate_hz``, for the reason
    :func:`motion_for_integration` gives: a time-lapse's capture step is not what it smears over.
    """
    tau_ms = getattr(sensor.fpa, "thermal_time_constant_ms", None)
    return smear_decay_frames(
        1.0 / float(sensor.fpa.frame_rate_hz),
        None
        if sensor.fpa.integration_time_ms is None
        else float(sensor.fpa.integration_time_ms) * 1e-3,
        None if tau_ms is None else float(tau_ms) * 1e-3,
    )


def optics_stage(planes: Planes, config: PipelineConfig, state: PipelineState) -> Planes:
    """Stage-3 entry point: supersampled ``radiance`` → ``flux`` on the detector grid."""
    return {
        "flux": apply_optics(
            np.asarray(planes["radiance"]),
            config.sensor.sensor,
            housing_band_radiance(config, state),
            supersample=config.supersample,
            psf=config.psf,
            motion_px=motion_for_integration(planes, config.sensor.sensor),
            motion_decay_frames=motion_decay(config.sensor.sensor),
        )
    }


class OpticsStage:
    name = "optics"

    def __call__(self, planes: Planes, config: PipelineConfig, state: PipelineState) -> Planes:
        return optics_stage(planes, config, state)
