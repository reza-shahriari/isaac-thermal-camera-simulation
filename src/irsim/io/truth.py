"""The scene truth behind a frame, as planes on the detector grid: what was *there* at each pixel.

docs/physics-model.md §12.2 ``outputs``, §13.3 (the G-buffer), ADR 0014 (ids are never filtered).

A frame's written outputs (:mod:`irsim.io.dataset`) are what the *camera* says: radiance,
apparent temperature, an ADC code. Debugging a render needs the other side of the comparison --
the surface temperature the radiometry was actually given, which part and which thermal node the
pixel landed on, what material, how far away. All of it is already in the G-buffer; this module
brings it to the detector grid and names the integer ids, so it can be written beside the camera
planes and read back at a clicked pixel (:mod:`irsim_viewer`).

**One sample per pixel, the one at its centre.** The G-buffer is rendered at ``k x`` the detector
grid (§8.3). Ids cannot be averaged -- a mean of two part ids is a third part that is not in the
scene -- so every plane here takes the sample nearest the pixel centre, temperature and distance
included, so all five describe the *same* surface point. That is the truth under the centre of the
pixel, not the pixel's value: on a silhouette the camera's pixel mixes surface and sky while this
reports only one of them. That is the point of having both.

``temperature_k`` under the sky mask is the apparent sky temperature the G-buffer contract puts
there (:mod:`irsim.config.gbuffer`), and ``distance_m`` there is written as NaN rather than the
contract's 0, so no reader mistakes the sky for a surface at the lens.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "CLOUD_LEGEND",
    "CLOUD_THRESHOLD",
    "TruthPlanes",
    "centre_sample",
    "truth_planes",
    "SKY_LABEL",
]

#: The legend entry for id 0 in ``part_id`` / ``node_id``: the ray hit no geometry.
SKY_LABEL = "(sky / no geometry)"

#: A pixel is *cloud* when the marched cloud lets less than this fraction of what lies behind it
#: through (AT.29): an emissivity above one half, the same line a cloud mask would draw in the
#: visible. Below it the pixel is what it hit, seen through thin cloud.
CLOUD_THRESHOLD = 0.5
CLOUD_LEGEND = {0: "clear", 1: "cloud"}


@dataclass(frozen=True)
class TruthPlanes:
    """Detector-grid truth planes, and the legends that name their integer ids."""

    planes: dict[str, NDArray[Any]] = field(default_factory=dict)
    legends: dict[str, dict[int, str]] = field(default_factory=dict)


def centre_sample(plane: Any, shape: tuple[int, int]) -> NDArray[Any]:
    """The sample nearest each detector pixel's centre, from a ``k x`` supersampled plane."""
    arr = np.asarray(plane)
    h, w = shape
    if arr.shape[:2] == (h, w):
        return arr
    k = arr.shape[0] // h
    if k < 1 or arr.shape[0] != k * h or arr.shape[1] != k * w:
        raise ValueError(f"plane {arr.shape[:2]} is not an integer supersample of {shape}")
    return np.ascontiguousarray(arr[k // 2 :: k, k // 2 :: k])


def _index(
    ids: NDArray[Any], names_of: Mapping[int, str], order: Sequence[str]
) -> tuple[NDArray[np.uint16], dict[int, str]]:
    """Recode raw ids to 1..N over ``order`` (0 = no name), with the legend for the codes."""
    code_of = {name: i for i, name in enumerate(order, start=1)}
    lut: dict[int, int] = {raw: code_of.get(name, 0) for raw, name in names_of.items()}
    out = np.zeros(ids.shape, dtype=np.uint16)
    for raw in np.unique(ids):
        out[ids == raw] = lut.get(int(raw), 0)
    return out, {0: SKY_LABEL, **{i: name for name, i in code_of.items()}}


def truth_planes(
    gbuffer: Mapping[str, Any],
    shape: tuple[int, int],
    *,
    instance_id: Any = None,
    labels: Mapping[int, str] | None = None,
    prim_to_node: Mapping[str, str] | None = None,
    material_names: Sequence[str] = (),
) -> TruthPlanes:
    """Bring a frame's G-buffer truth to the detector grid ``shape`` (rows, cols).

    ``instance_id`` and ``labels`` (renderer instance id -> prim path) give ``part_id``, named by
    the prim's leaf name; ``prim_to_node`` (prim path -> thermal node) gives ``node_id``, named by
    the node, so a viewer can join it with the frame's ``node_temperatures_k``. Each plane is
    written only when its inputs are present -- an absent plane is never faked with zeros.
    """
    planes: dict[str, NDArray[Any]] = {}
    legends: dict[str, dict[int, str]] = {}
    sky = gbuffer.get("sky_mask")
    sky_px = None if sky is None else centre_sample(sky, shape).astype(bool)

    if "temperature_k" in gbuffer:
        planes["temperature_k"] = centre_sample(gbuffer["temperature_k"], shape).astype(np.float32)
    if "distance_m" in gbuffer:
        distance = centre_sample(gbuffer["distance_m"], shape).astype(np.float32)
        if sky_px is not None:
            distance = np.where(sky_px, np.float32(np.nan), distance)
        planes["distance_m"] = distance
    if "material_id" in gbuffer:
        material = centre_sample(gbuffer["material_id"], shape).astype(np.uint16)
        planes["material_id"] = material
        if material_names:
            legends["material_id"] = dict(enumerate(material_names))

    if "cloud_transmittance" in gbuffer:
        # AT.29: the cloud between the camera and the hit, as the march found it -- a fraction,
        # the range its emission came from, and a name a label writer can use. Written for sky
        # pixels too: there the ray ran to infinity and the plane says how much of the sky was
        # cloud, which is what "unknown" became.
        tau = centre_sample(gbuffer["cloud_transmittance"], shape).astype(np.float32)
        planes["cloud_transmittance"] = tau
        if "cloud_range_m" in gbuffer:
            rng = centre_sample(gbuffer["cloud_range_m"], shape).astype(np.float32)
            planes["cloud_range_m"] = np.where(tau >= 1.0, np.float32(np.nan), rng)
        planes["cloud_id"] = (tau < CLOUD_THRESHOLD).astype(np.uint8)
        legends["cloud_id"] = dict(CLOUD_LEGEND)

    if instance_id is not None and labels:
        ids = centre_sample(instance_id, shape)
        leaves = {raw: path.rsplit("/", 1)[-1] for raw, path in labels.items()}
        planes["part_id"], legends["part_id"] = _index(ids, leaves, sorted(set(leaves.values())))
        if prim_to_node:
            nodes = {
                raw: prim_to_node[path] for raw, path in labels.items() if path in prim_to_node
            }
            planes["node_id"], legends["node_id"] = _index(
                ids, nodes, sorted(set(prim_to_node.values()))
            )
        if sky_px is not None:
            for key in ("part_id", "node_id"):
                if key in planes:
                    planes[key][sky_px] = 0

    return TruthPlanes(planes=planes, legends=legends)
