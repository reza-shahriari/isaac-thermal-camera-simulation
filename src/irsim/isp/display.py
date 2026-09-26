"""Stage 6, display branch: DN16 → 8-bit RGBA through the §11.1 chain in the fixed order

    AGC (linear | plateau_equalization | none) → gamma → DDE → polarity → palette      (ADR 0031)

``information_based`` (SC.21, ADR 0147) carries its own DDE -- the high-pass is added back inside
the AGC at the transfer's local slope -- so the DDE step is skipped for it rather than run twice.

Every operator is a **pure per-frame function** with no hidden memory: an SPG port (which holds
no cross-frame state, ADR 0014) can implement the whole branch, and any future temporal
behaviour (AGC smoothing, FFC tables) must be explicit fields of ``PipelineState``.

Rounding points (for an exact GPU comparison, the port must round at the same places):

    R1  AGC: histogram and CDF are integer/float64 (exact); the output y is cast to float32.
    R2  gamma: y = y ** (1/γ) evaluated in float32.
    R3  DDE: box mean accumulated in float64 from float32 y, result cast to float32.
    R4  quantisation: DN8 = rint(255 · clip(y)) computed from the float32 y (half-to-even),
        then the uint8 palette table; ``agc: none`` is the exact shift DN16 >> (bits − 8).

The isp config hash (SHA-256 of the canonical JSON of the ``isp`` block plus the bit depth)
travels with the output so that a train/deploy AGC mismatch is detectable (§15 Tier 5).

docs/physics-model.md §11.1, §11.3, §11.4, §12.2 isp/outputs, §13.4
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from irsim.config.sensor import ISP_OPTIONAL_DEFAULTS, IspSpec
from irsim.isp.agc import agc_linear, agc_plateau, agc_plateau_local
from irsim.isp.dde import dde
from irsim.isp.information import agc_information, blend_linear, equalise_image
from irsim.isp.palette import to_display8

__all__ = ["DisplayOutputs", "isp_config_hash", "agc_none", "run_display_branch"]


@dataclass(frozen=True)
class DisplayOutputs:
    display8: NDArray[np.uint8]  # (H, W, 4) RGBA8
    y: NDArray[np.float32]  # the [0, 1] image before polarity/palette (for benches)
    isp_hash: str


def isp_config_hash(isp: IspSpec, bit_depth: int) -> str:
    dumped = isp.model_dump(mode="json")
    for key, default in ISP_OPTIONAL_DEFAULTS.items():  # SC.21: defaults leave the hash alone
        if dumped.get(key) == default:
            dumped.pop(key, None)
    payload = {"isp": dumped, "bit_depth": int(bit_depth)}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _check_dn16(dn16: object, bit_depth: int) -> NDArray[np.uint16]:
    arr = np.asarray(dn16)
    if arr.dtype != np.uint16:
        raise TypeError(f"the display branch takes uint16 DN, got {arr.dtype}")
    if arr.ndim != 2:
        raise ValueError("dn16 must be (H, W)")
    if not 8 <= bit_depth <= 16:
        raise ValueError("bit_depth must be in 8..16")
    if int(arr.max()) > 2**bit_depth - 1:
        raise ValueError(f"DN exceeds 2^{bit_depth} - 1")
    return arr


def agc_none(dn16: NDArray[np.uint16], bit_depth: int) -> NDArray[np.float32]:
    """``agc: none``: the top 8 bits, DN16 >> (bits − 8), as a [0, 1] float32 image (R4 exact)."""
    shift = bit_depth - 8
    top = (dn16.astype(np.uint32) >> np.uint32(shift)).astype(np.float32)
    return np.asarray(top / np.float32(255.0), dtype=np.float32)


def run_display_branch(dn16: object, isp: IspSpec, bit_depth: int) -> DisplayOutputs:
    """DN16 → RGBA8 in the ADR 0031 order; float32 at every intermediate; uint8 out."""
    dn = _check_dn16(dn16, bit_depth)
    if isp.agc == "linear":
        y = agc_linear(dn, isp.clip_percentiles[0], isp.clip_percentiles[1], 1.0, bit_depth)  # R1
    elif isp.agc == "plateau_equalization":
        if isp.linear_percent or isp.clip_limit_low or isp.max_gain:  # SC.21, SC.25
            y = equalise_image(
                dn,
                isp.plateau,
                bit_depth,
                linear_percent=isp.linear_percent,
                clip_limit_low=isp.clip_limit_low,
                max_gain=isp.max_gain,
            )
        else:
            y = agc_plateau(dn, isp.plateau, bit_depth)  # R1
    elif isp.agc == "information_based":
        # SC.21: DDE is part of this operator (the high-pass is added back at the transfer's
        # slope), so the R3 unsharp mask below is skipped for it rather than applied twice.
        y = agc_information(
            dn,
            isp.plateau,
            bit_depth,
            info_weight=isp.info_weight,
            linear_percent=isp.linear_percent,
            detail_headroom=isp.detail_headroom,
            detail_gain=1.0 + isp.dde_gain,
            smoothing_sigma_dn=isp.smoothing_sigma_dn,
            clip_limit_low=isp.clip_limit_low,
            max_gain=isp.max_gain,
        )
    elif isp.agc == "plateau_local":
        y = agc_plateau_local(
            dn, isp.plateau, isp.agc_tiles, bit_depth, clip_limit_low=isp.clip_limit_low
        )  # M9.10
        y = blend_linear(y, dn, isp.linear_percent, bit_depth)  # SC.25; λ = 0 is the identity
    elif isp.agc == "none":
        y = agc_none(dn, bit_depth)
    else:  # pragma: no cover - the schema restricts the literal
        raise ValueError(f"unknown agc mode {isp.agc!r}")
    if isp.gamma != 1.0:
        y = np.power(y, np.float32(1.0 / isp.gamma), dtype=np.float32)  # R2
    if isp.dde_gain > 0.0 and isp.agc != "information_based":
        y = dde(y, isp.dde_gain)  # R3
    display8 = to_display8(y, isp.polarity, isp.palette)  # R4
    return DisplayOutputs(
        display8=display8,
        y=np.asarray(y, dtype=np.float32),
        isp_hash=isp_config_hash(isp, bit_depth),
    )
