"""Instance ids -> the G-buffer's ``material_id`` plane, and the magenta UNMAPPED overlay.

docs/physics-model.md §13.3, §13.5; roadmap M10.2; ADR 0014 (ids are the transport) and ADR 0047
(precedence, the UNMAPPED sentinel, magenta, the coverage gate).

The renderer knows nothing about emissivity. What it does transport exactly is an integer instance
id per pixel, plus an ``idToLabels`` table mapping that id to the prim path (measured in ADR 0014:
distinct ids at every quad centre, no blended edge pixels). The chain this module closes is

    instance id  --idToLabels-->  prim path  --M7.17 resolver-->  material id

so the ids in the G-buffer are the packed table's ids (M7.18) and the resolver, the table and the
kernel cannot disagree about what id 7 means.

**Ids must never be interpolated.** A pixel on the boundary between material 0 and material 7 is
one or the other; averaging them yields material 3 or 4, which is a different substance. That is
why the id path is nearest-neighbour everywhere and why anti-aliasing happens by supersampling the
ids and filtering **radiance** afterwards (ADR 0014), never by filtering the ids themselves.

**The miss is loud, in both branches.** An unresolved prim keeps id 0 (``UNMAPPED``) rather than
a plausible default emissivity, :func:`overlay_unmapped` paints those pixels magenta in the display
branch, and :func:`mark_unmapped_radiometry` writes NaN over them on the planes that claim physical
units (ADR 0047, `IG.17`). Until the second of those the marking was a *picture* only: `SE.2`
measured an undeclared sea reading **200.1 K** -- the band LUT's own floor, where an apparent
temperature lands when the radiance under it is nearly zero -- across 72 % of a frame, while the
mask beside it looked entirely correct. A number in kelvin and in range reads as cold water, not as
an absence of physics.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
from numpy.typing import NDArray

from irsim.materials.mapping import Resolution
from irsim.materials.table import UNMAPPED_MATERIAL_ID

__all__ = [
    "MAGENTA_RGBA",
    "BACKGROUND_INSTANCE_ID",
    "fold_mask_to_native",
    "mark_unmapped_radiometry",
    "labels_to_paths",
    "labels_from_payload",
    "material_id_plane",
    "unmapped_mask",
    "overlay_unmapped",
    "id_coverage",
]

#: The UNMAPPED debug colour (ADR 0047). Opaque, and deliberately a colour no palette produces.
MAGENTA_RGBA: tuple[int, int, int, int] = (255, 0, 255, 255)

#: Instance id 0 is "nothing was hit" -- the sky (ADR 0014). It is *not* an UNMAPPED material:
#: the kernels read those pixels as blackbody-equivalent from the sky model, so they must not be
#: painted magenta or counted as a mapping miss.
BACKGROUND_INSTANCE_ID = 0


def labels_to_paths(id_to_labels: Mapping[Any, Any] | None) -> dict[int, str]:
    """Normalise Replicator's ``idToLabels`` into ``{instance id: prim path}``.

    The payload varies by annotator: ``instance_segmentation`` gives the prim path as a plain
    string, while the semantic annotators give ``{"class": label}``. Keys arrive as strings.
    Entries that are neither are dropped rather than guessed at.
    """
    out: dict[int, str] = {}
    for key, value in (id_to_labels or {}).items():
        try:
            ident = int(key)
        except (TypeError, ValueError):
            continue
        if isinstance(value, str):
            label = value
        elif isinstance(value, Mapping):
            label = str(value.get("class", "")) or ""
        else:
            label = ""
        if label:
            out[ident] = label
    return out


def labels_from_payload(payload: Any) -> dict[int, str]:
    """Pull ``{instance id: prim path}`` out of a raw segmentation annotator payload.

    The segmentation annotators return ``{"data": array, "info": {"idToLabels": {...}}}`` rather
    than a bare array, and ``AovReader`` keeps that original payload in
    ``RawAovs.device_handles`` precisely so the label table is still reachable here.
    """
    info = payload.get("info", {}) if isinstance(payload, Mapping) else {}
    table = info.get("idToLabels") if isinstance(info, Mapping) else None
    return labels_to_paths(table if isinstance(table, Mapping) else None)


def material_id_plane(
    instance_ids: Any,
    id_to_labels: Mapping[Any, Any] | None,
    resolutions: Iterable[Resolution],
    *,
    background_id: int = BACKGROUND_INSTANCE_ID,
    strict: bool = False,
) -> NDArray[np.int32]:
    """Per-pixel int32 material id, by exact integer lookup -- never interpolation.

    ``resolutions`` are the M7.17 resolver's per-prim results. Any instance id that has no prim
    path, or whose prim did not resolve, becomes ``UNMAPPED_MATERIAL_ID`` (0); the background id
    becomes 0 too, which the kernels ignore because ``sky_mask`` is set there (the M0.6 contract
    says id 0 under the mask is *not* the UNMAPPED error).

    With ``strict``, an id present in the image but absent from ``idToLabels`` raises instead --
    use it in tests, where a silently unmapped prim is the bug being hunted.
    """
    ids = np.asarray(instance_ids)
    if not np.issubdtype(ids.dtype, np.integer):
        raise TypeError(f"instance_ids must be an integer plane, got {ids.dtype}")
    if ids.ndim != 2:
        raise ValueError(f"instance_ids must be (H, W), got {ids.shape}")

    by_path = {r.path: int(r.material_id) for r in resolutions}
    paths = labels_to_paths(id_to_labels)

    present = np.unique(ids)
    lookup: dict[int, int] = {}
    missing_labels: list[int] = []
    for ident in present.tolist():
        if ident == background_id:
            lookup[ident] = UNMAPPED_MATERIAL_ID
            continue
        path = paths.get(int(ident))
        if path is None:
            missing_labels.append(int(ident))
            lookup[ident] = UNMAPPED_MATERIAL_ID
            continue
        lookup[ident] = by_path.get(path, UNMAPPED_MATERIAL_ID)
    if strict and missing_labels:
        raise KeyError(
            f"instance ids {sorted(missing_labels)} appear in the image but not in idToLabels; "
            "every rendered prim must be resolvable or the pixel silently becomes UNMAPPED"
        )

    # Exact integer remap: build a small table over the ids actually present and index it, so no
    # arithmetic that could round or blend ever touches an id.
    out = np.zeros(ids.shape, dtype=np.int32)
    for ident, material in lookup.items():
        if material != UNMAPPED_MATERIAL_ID:
            out[ids == ident] = np.int32(material)
    return out


def unmapped_mask(material_id: Any, sky_mask: Any | None = None) -> NDArray[np.bool_]:
    """Pixels carrying the UNMAPPED sentinel on real geometry (sky pixels are excluded)."""
    mat = np.asarray(material_id)
    mask = mat == UNMAPPED_MATERIAL_ID
    if sky_mask is not None:
        mask &= ~np.asarray(sky_mask, dtype=bool)
    return np.asarray(mask, dtype=np.bool_)


def fold_mask_to_native(mask: Any, supersample: int) -> NDArray[np.bool_]:
    """A k x k G-buffer mask folded onto the detector grid, marking a pixel if **any** sample is.

    Ids are never interpolated, so a native pixel covering four samples covers up to four prims,
    and one unmapped sample is enough to make the pixel's radiance partly invented. Over-reporting
    a forgotten prim is the only safe direction: the alternative hides one behind three good
    samples, and a pixel that is three quarters physics and one quarter nothing is not a
    measurement of anything.
    """
    arr = np.asarray(mask, dtype=bool)
    k = int(supersample)
    if k <= 1:
        return np.asarray(arr, dtype=np.bool_)
    h, w = arr.shape[0] // k, arr.shape[1] // k
    folded = arr[: h * k, : w * k].reshape(h, k, w, k).any(axis=(1, 3))
    return np.asarray(folded, dtype=np.bool_)


def mark_unmapped_radiometry(plane: Any, mask: Any) -> NDArray[np.float32]:
    """``plane`` with NaN written over the masked pixels, as float32 (`IG.17`, ADR 0047).

    NaN rather than a sentinel kelvin. Every plausible sentinel -- 0 K, the LUT's 200 K floor,
    -999 -- is either a temperature something downstream will average or a value some consumer
    clips back into range, and this plane's whole job is to be *unusable* where the physics is
    absent. NaN is the one value that propagates through a mean, fails a comparison and shows up
    in a writer's finite check, which is exactly the behaviour wanted: a frame containing
    unmapped geometry is not a measurement of that geometry, and nothing should be able to
    average it into one by accident.

    A caller that wants the number anyway still has it: the mask is returned beside the frame, and
    ``debug_unmapped=False`` raises instead of marking.
    """
    out = np.array(plane, dtype=np.float32, copy=True)
    marked = np.asarray(mask, dtype=bool)
    if marked.shape != out.shape[: marked.ndim]:
        raise ValueError(f"mask {marked.shape} does not fit a plane of {out.shape}")
    out[marked] = np.float32("nan")
    return out


def overlay_unmapped(
    display_rgba: Any, mask: Any, colour: tuple[int, int, int, int] = MAGENTA_RGBA
) -> NDArray[np.uint8]:
    """Paint ``colour`` over the masked pixels of an RGBA8 display image, leaving the rest exact.

    The overlay is a *display* artefact only: DN16 and the radiance/apparent-temperature outputs
    are untouched, so a debug render stays radiometrically readable underneath the magenta
    (ADR 0047 -- and under the overlay the kernel uses eps = 1, never a "typical" emissivity that
    could pass as real).
    """
    rgba = np.asarray(display_rgba)
    if rgba.dtype != np.uint8:
        raise TypeError(f"display image must be uint8 RGBA, got {rgba.dtype}")
    if rgba.ndim != 3 or rgba.shape[2] != 4:
        raise ValueError(f"display image must be (H, W, 4), got {rgba.shape}")
    m = np.asarray(mask, dtype=bool)
    if m.shape != rgba.shape[:2]:
        raise ValueError(f"mask has shape {m.shape}, expected {rgba.shape[:2]}")
    out: NDArray[np.uint8] = rgba.copy().astype(np.uint8)
    out[m] = np.asarray(colour, dtype=np.uint8)
    return out


def id_coverage(material_id: Any, sky_mask: Any | None = None) -> float:
    """Fraction of geometry pixels that carry a real material. 1.0 when nothing is UNMAPPED."""
    mat = np.asarray(material_id)
    geometry = (
        np.ones(mat.shape, dtype=bool) if sky_mask is None else ~np.asarray(sky_mask, dtype=bool)
    )
    total = int(geometry.sum())
    if total == 0:
        return 1.0
    return float((mat[geometry] != UNMAPPED_MATERIAL_ID).sum() / total)
