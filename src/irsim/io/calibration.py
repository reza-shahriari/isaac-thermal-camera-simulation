"""Bench-measured calibration planes beside a camera file (SC.33).

docs/physics-model.md §10.3, §10.4, §12.2; ADR 0164. ``calibration.radiometric`` in a sensor file
names ``.npy`` sidecars -- a gain map, an offset map, a bad-pixel map -- that the loader has
already resolved against the data root and hashed into the config hash. This module reads them
and refuses what would silently be wrong: a plane of another shape than the FPA, a gain or offset
narrower than float32 (CLAUDE.md #2), a bad-pixel map that is not boolean, a NaN anywhere.
"""

from __future__ import annotations

import os
import pathlib
from typing import Any

import numpy as np
from numpy.typing import NDArray

from irsim.noise.defects import BadPixelMap, DefectKind

__all__ = ["bad_pixel_map_from_mask", "load_bad_pixel_map", "load_gain_or_offset_map"]


def _load(path: str | os.PathLike[str], shape: tuple[int, int], what: str) -> NDArray[Any]:
    p = pathlib.Path(path).expanduser()
    if not p.is_file():
        raise FileNotFoundError(f"{what} {p} does not exist")
    arr = np.load(p, allow_pickle=False)
    if arr.ndim != 2 or tuple(arr.shape) != tuple(shape):
        raise ValueError(f"{what} {p} has shape {arr.shape}; the FPA is {tuple(shape)}")
    return np.asarray(arr)


def load_gain_or_offset_map(
    path: str | os.PathLike[str], shape: tuple[int, int]
) -> NDArray[np.float32]:
    """A per-pixel gain (about 1) or offset (DN) plane of the FPA's shape, float32 or better."""
    arr = _load(path, shape, "calibration map")
    if not np.issubdtype(arr.dtype, np.floating) or arr.dtype.itemsize < 4:
        raise TypeError(f"{path}: a gain or offset map is float32 or better, got {arr.dtype}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{path}: the map holds a NaN or an infinity")
    return np.asarray(arr, dtype=np.float32)


def load_bad_pixel_map(path: str | os.PathLike[str], shape: tuple[int, int]) -> BadPixelMap:
    """A boolean plane of the FPA's shape, True where the unit's factory map replaces the pixel."""
    arr = _load(path, shape, "bad-pixel map")
    if arr.dtype != np.bool_:
        raise TypeError(f"{path}: a bad-pixel map is boolean, got {arr.dtype}")
    return bad_pixel_map_from_mask(np.asarray(arr, dtype=bool))


def bad_pixel_map_from_mask(mask: NDArray[np.bool_]) -> BadPixelMap:
    """The unit's own map: every flagged pixel is on it (none is *late*, SC.19) and is classed
    DEAD, the one class a map that only says "bad" can support -- the camera replaces the pixel
    either way, so the class never shows."""
    m = np.asarray(mask, dtype=bool)
    kind = np.where(m, np.uint8(DefectKind.DEAD), np.uint8(DefectKind.GOOD)).astype(np.uint8)
    return BadPixelMap(kind=kind)
