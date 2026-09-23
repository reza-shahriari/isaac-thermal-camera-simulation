"""The G-buffer contract: what the renderer must hand the radiometry kernels.

A G-buffer is a dictionary of per-pixel physical quantities (docs/physics-model.md §13.3 AOV
table, §13.4). Isaac Sim's annotator adapter emits exactly these keys; the synthetic fixtures in
tests/conftest.py emit exactly these keys; the kernels consume exactly these keys. :class:`GBuffer`
is the one place the key set and dtypes are checked, so a float16 temperature cannot cross the
boundary (CLAUDE.md non-negotiable #2) and a kernel never sees a key it did not expect.

``encoded_t`` is optional and is *not* produced by the Isaac adapter (ADR 0014: the renderer
transports ids and geometry; temperature comes from a float32 table), so no kernel may require
it; when present it is checked against ``temperature_k``.

Precision policy (§13.3): temperature, encoded temperature, distance and any radiance key are
**float32 or better** -- float16 raises. Normals, sky-view factor and motion vectors may arrive
as float16 from the engine and are upcast to float32 here. Integer ids are cast to int32
(material) and uint32 (semantic). ``material_id`` 0 is the UNMAPPED sentinel (roadmap M7.18):
fixtures use ids >= 1.

``sky_mask`` (optional, bool) marks pixels where the renderer hit no geometry. The Isaac
renderer reports those with distance +inf and instance id 0 (measured, ADR 0014 lane); the
adapter translates that into ``sky_mask = True``, ``distance_m = 0`` (so a consumer that ignores
the mask sees τ = 1) and ``temperature_k`` = the apparent sky temperature T_sky(θ) of the sky
model (MS.2). Under the mask the kernels treat the pixel as blackbody-equivalent (ε = 1,
``material_id`` ignored, so id 0 there is *not* the UNMAPPED error) and stage 2 passes it through
untouched: its apparent temperature already includes the atmosphere to space. Without the plane
every pixel is geometry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from irsim.radiometry.encoding import decode_temperature

__all__ = [
    "GBuffer",
    "REQUIRED_KEYS",
    "OPTIONAL_KEYS",
    "PRECISION_CRITICAL_KEYS",
    "UNMAPPED_MATERIAL_ID",
    "ENCODED_T_CONSISTENCY_TOL_K",
    "BOOL_KEYS",
]

REQUIRED_KEYS: frozenset[str] = frozenset(
    {"temperature_k", "normal_dot_view", "distance_m", "material_id", "sky_view_factor"}
)
# ``shadow_mask`` and ``sun_cos_incidence`` are the solar pair (M11.3, §5.4). Both are
# optional, exactly as ``motion_px`` is: a scene that has not asked for sunlight costs
# nothing and renders identically. ``shadow_mask`` is S in [0, 1] with **1 = lit** -- the
# same polarity as ``irsim.thermal.solar.solar_loading``, because the two must never
# disagree about which pixels the sun reaches. ``sun_cos_incidence`` is n·ŝ, which an
# adapter forms from its normal AOV and the scene's own NOAA sun direction.
# ``background_t_k`` (`OC.7`) is the apparent temperature each pixel's ray would report **with all
# geometry removed** -- what is behind the thing in front. It exists because a defocused foreground
# silhouette lets the background show through, and a single-layer G-buffer has no behind: the
# layered defocus stage otherwise has to guess. For a sky or sea background it is not a guess and
# not a second render either, because sky and sea radiance are functions of ray direction the
# adapter already evaluates for the pixels where they are visible; evaluating them for every pixel
# costs nothing more. Optional: without it the stage falls back to `OC.6`'s normalisation.
# ``elevation_rad`` is each pixel's own ray elevation above the horizon, positive up (AT.1).
# Optional like the rest: without it the atmosphere keeps the horizontal path it has always used,
# so a scene that does not supply it renders bit-identically. With it, every resolved pixel gets
# the slant path the *unresolved* point-target path -- and the sky behind it -- were already using,
# which is the discontinuity AT.1 exists to close.
OPTIONAL_KEYS: frozenset[str] = frozenset(
    {
        "encoded_t",
        "motion_px",
        "semantic_id",
        "sky_mask",
        "shadow_mask",
        "sun_cos_incidence",
        "elevation_rad",
        "background_t_k",
    }
)
INTEGER_KEYS: dict[str, type] = {"material_id": np.int32, "semantic_id": np.uint32}
# Boolean planes: accept bool or an integer 0/1 plane (the adapter may hand a uint8 AOV); floats
# are refused because a fractional "sky" cannot be given a meaning.
BOOL_KEYS: frozenset[str] = frozenset({"sky_mask"})
# Keys where float16 is an error rather than something to upcast. Any key whose name starts with
# "radiance" is treated the same way, so later stages can add radiance planes without editing here.
PRECISION_CRITICAL_KEYS: frozenset[str] = frozenset(
    # `elevation_rad` joins these because it scales the optical depth of the whole slant path
    # (AT.1): fp16 spaces 1 degree at about 0.03 degrees near the horizon, where the airmass is
    # steepest and a tenth of a degree is metres of column.
    {"temperature_k", "encoded_t", "distance_m", "elevation_rad", "background_t_k"}
)
UNMAPPED_MATERIAL_ID = 0
# encoded_t must decode to temperature_k within this (the AOV round-trip budget, §13.3).
ENCODED_T_CONSISTENCY_TOL_K = 0.010

_PLANE_NDIM = {"motion_px": 3}


def _is_precision_critical(key: str) -> bool:
    return key in PRECISION_CRITICAL_KEYS or key.startswith("radiance")


@dataclass(frozen=True)
class GBuffer:
    """Validated per-pixel physical quantities for one frame. Build with :meth:`from_dict`."""

    temperature_k: NDArray[np.float32]
    normal_dot_view: NDArray[np.float32]
    distance_m: NDArray[np.float32]
    material_id: NDArray[np.int32]
    sky_view_factor: NDArray[np.float32]
    encoded_t: NDArray[np.float32] | None = None
    motion_px: NDArray[np.float32] | None = None
    semantic_id: NDArray[np.uint32] | None = None
    sky_mask: NDArray[np.bool_] | None = None
    shadow_mask: NDArray[np.float32] | None = None
    sun_cos_incidence: NDArray[np.float32] | None = None
    elevation_rad: NDArray[np.float32] | None = None
    extra: dict[str, NDArray[Any]] = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, int]:
        return (int(self.temperature_k.shape[0]), int(self.temperature_k.shape[1]))

    @classmethod
    def from_dict(cls, planes: dict[str, Any]) -> GBuffer:
        """Validate a plane dictionary (the adapter/fixture output) into a GBuffer.

        Raises ``KeyError`` for missing required keys, ``TypeError`` for a float16 plane on the
        precision-critical path or a non-integer id plane, ``ValueError`` for shape mismatches,
        non-finite values, or an ``encoded_t`` inconsistent with ``temperature_k``.
        """
        missing = REQUIRED_KEYS - planes.keys()
        if missing:
            raise KeyError(f"G-buffer missing required planes {sorted(missing)}")
        unknown = planes.keys() - REQUIRED_KEYS - OPTIONAL_KEYS
        unknown_bad = {k for k in unknown if not k.startswith("radiance")}
        if unknown_bad:
            raise KeyError(
                f"G-buffer has unexpected planes {sorted(unknown_bad)}; the key set is frozen in "
                "irsim.config.gbuffer (extend it there and in test_gbuffer_schema.py)"
            )

        converted: dict[str, NDArray[Any]] = {}
        for key, value in planes.items():
            arr = np.asarray(value)
            if key in BOOL_KEYS:
                is_bool = arr.dtype == np.bool_
                is_01 = np.issubdtype(arr.dtype, np.integer) and bool(np.isin(arr, (0, 1)).all())
                if not (is_bool or is_01):
                    raise TypeError(f"{key} must be a bool or 0/1 integer plane, got {arr.dtype}")
                converted[key] = arr.astype(np.bool_)
                continue
            if key in INTEGER_KEYS:
                if not np.issubdtype(arr.dtype, np.integer):
                    raise TypeError(f"{key} must be an integer plane, got {arr.dtype}")
                converted[key] = arr.astype(INTEGER_KEYS[key])
                continue
            if not np.issubdtype(arr.dtype, np.floating):
                raise TypeError(f"{key} must be a float plane, got {arr.dtype}")
            if arr.dtype == np.float16 and _is_precision_critical(key):
                raise TypeError(
                    f"{key} arrived as float16: 0.25 K spacing at 300 K, five times a 50 mK "
                    "NETD (CLAUDE.md non-negotiable #2). The AOV must be float32."
                )
            # §13.3: fp16 is acceptable for normals / AO / motion; everything is stored as float32
            # (float64 input is "float32 or better" and is narrowed here, once).
            arr = arr.astype(np.float32)
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"{key} contains non-finite values")
            converted[key] = arr

        h, w = converted["temperature_k"].shape[:2]
        for key, arr in converted.items():
            ndim = _PLANE_NDIM.get(key, 2)
            if arr.ndim != ndim or arr.shape[:2] != (h, w):
                raise ValueError(
                    f"{key} has shape {arr.shape}; expected ({h}, {w}{', 2' if ndim == 3 else ''})"
                )
        if "motion_px" in converted and converted["motion_px"].shape[2] != 2:
            raise ValueError("motion_px must be (H, W, 2): image-plane velocity in px/frame")

        if "encoded_t" in converted:
            decoded = decode_temperature(converted["encoded_t"]).astype(np.float64)
            err = np.max(np.abs(decoded - converted["temperature_k"].astype(np.float64)))
            if err > ENCODED_T_CONSISTENCY_TOL_K:
                raise ValueError(
                    f"encoded_t decodes to temperature_k with max error {err * 1e3:.1f} mK "
                    f"(> {ENCODED_T_CONSISTENCY_TOL_K * 1e3:.0f} mK): encoder and adapter disagree"
                )

        extra = {k: v for k, v in converted.items() if k.startswith("radiance")}
        named = {k: v for k, v in converted.items() if not k.startswith("radiance")}
        return cls(**named, extra=extra)

    def to_dict(self) -> dict[str, NDArray[Any]]:
        out: dict[str, NDArray[Any]] = {
            "temperature_k": self.temperature_k,
            "normal_dot_view": self.normal_dot_view,
            "distance_m": self.distance_m,
            "material_id": self.material_id,
            "sky_view_factor": self.sky_view_factor,
        }
        for key in (
            "encoded_t",
            "motion_px",
            "semantic_id",
            "sky_mask",
            "shadow_mask",
            "sun_cos_incidence",
        ):
            value = getattr(self, key)
            if value is not None:
                out[key] = value
        out.update(self.extra)
        return out
