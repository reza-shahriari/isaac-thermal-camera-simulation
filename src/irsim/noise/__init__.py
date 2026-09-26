"""Noise: seeded streams, the NVESD 3-D synthesiser, FPN drift, bad pixels, the noise stage.

docs/physics-model.md §10
"""

from irsim.noise.defects import (
    CLUSTER_RADIUS_PX,
    BadPixelMap,
    DefectKind,
    DefectState,
    active_defect_mask,
    advance_state,
    apply_defects,
    generate_map,
    replacement_mask,
)
from irsim.noise.drift import (
    DRIFTING_COMPONENTS,
    DriftingPattern,
    FpnDrift,
    drift_rng,
    ou_step,
)
from irsim.noise.nuc_residual import RESIDUAL_REFERENCE_K, NucResidual
from irsim.noise.seeding import (
    NoiseStream,
    field_normal,
    hash_normal,
    hash_u64,
    hash_uniform,
    noise_rng,
    sensor_rng,
    stream_key,
)
from irsim.noise.stage import NoiseStage, measure_from_uniform_scene
from irsim.noise.three_d import FixedPattern, Sigmas7, synthesize_frame

__all__ = [
    "BadPixelMap",
    "DefectKind",
    "DefectState",
    "CLUSTER_RADIUS_PX",
    "generate_map",
    "advance_state",
    "apply_defects",
    "active_defect_mask",
    "replacement_mask",
    "NucResidual",
    "RESIDUAL_REFERENCE_K",
    "DRIFTING_COMPONENTS",
    "FpnDrift",
    "DriftingPattern",
    "drift_rng",
    "ou_step",
    "NoiseStream",
    "field_normal",
    "hash_normal",
    "hash_u64",
    "hash_uniform",
    "noise_rng",
    "sensor_rng",
    "stream_key",
    "FixedPattern",
    "Sigmas7",
    "synthesize_frame",
    "NoiseStage",
    "measure_from_uniform_scene",
]
