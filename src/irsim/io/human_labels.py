"""Labelling a skinned human mesh onto the body schema, as arithmetic (HU.3).

docs/physics-model.md §6.1, §16.2; roadmap HU.3; ADR 0192 (the skeleton is the labeller);
:mod:`irsim.config.humans` (the schema the labels come from).

``scripts/prep_human.py`` runs inside Blender, where neither pydantic nor this project's loaders
exist, so the arithmetic of the labelling lives here, NumPy only, and the Blender worker calls it
with plain arrays. That is also what makes it testable without Blender: a strip of faces under
two bones, a torso split by a forward axis, a pair of feet pointing the wrong way.

Four steps, four functions:

1. :func:`vertex_labels` -- the segment or region of the deforming bone with the largest weight
   on each vertex; a vertex with no deforming weight is ``-1``.
2. :func:`face_labels` -- each face takes the label with the most votes among its vertices, ties
   to the lowest label index so the result is deterministic; a face with no labelled vertex takes
   the label of the nearest labelled face, by centre.
3. :func:`split_region` -- faces on a region (``torso``) become its positive or negative segment by
   the sign of their centre's offset from the region's centre along the body's forward axis.
4. :func:`forward_check` -- a foot's centre lies forward of the shin's above it (toes point
   forward), so the displacement along the declared axis must be positive; a body facing the other
   way fails here instead of rendering with Chest and Back swapped.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "AXES",
    "face_labels",
    "forward_check",
    "split_region",
    "vertex_labels",
]

#: Named body-forward axes in the asset's frame (Blender's Z-up; a character facing the front
#: view faces -Y).
AXES: dict[str, tuple[float, float, float]] = {
    "+x": (1.0, 0.0, 0.0),
    "-x": (-1.0, 0.0, 0.0),
    "+y": (0.0, 1.0, 0.0),
    "-y": (0.0, -1.0, 0.0),
}


def vertex_labels(
    weights: NDArray[np.floating], group_label: NDArray[np.integer]
) -> NDArray[np.int64]:
    """The label of each vertex's heaviest deforming group.

    ``weights`` is ``(n_vertices, n_groups)``; ``group_label`` is ``(n_groups,)`` with the label
    index of each vertex group, or ``-1`` for a group that is not a deforming bone (ignored). A
    vertex with no positive weight on any deforming group is ``-1``.
    """
    w = np.asarray(weights, dtype=np.float64)
    gl = np.asarray(group_label, dtype=np.int64)
    if w.ndim != 2 or gl.shape != (w.shape[1],):
        raise ValueError("weights must be (n_vertices, n_groups) and group_label (n_groups,)")
    masked = np.where(gl[None, :] >= 0, w, -np.inf)
    best = np.argmax(masked, axis=1)
    out: NDArray[np.int64] = gl[best].astype(np.int64)
    out[~(np.max(masked, axis=1) > 0.0)] = -1
    return out


def face_labels(
    faces: list[list[int]] | NDArray[np.integer],
    vertex_label: NDArray[np.integer],
    face_centres: NDArray[np.floating],
) -> NDArray[np.int64]:
    """Each face's label by vertex vote; unlabelled faces take the nearest labelled face's label."""
    vl = np.asarray(vertex_label, dtype=np.int64)
    centres = np.asarray(face_centres, dtype=np.float64)
    n = len(faces)
    out = np.full(n, -1, dtype=np.int64)
    for i, f in enumerate(faces):
        labs = vl[np.asarray(f, dtype=np.int64)]
        labs = labs[labs >= 0]
        if labs.size:
            vals, counts = np.unique(labs, return_counts=True)
            out[i] = vals[np.argmax(counts)]  # np.unique sorts, so a tie goes to the lowest
    if np.any(out < 0):
        have = np.flatnonzero(out >= 0)
        if have.size == 0:
            raise ValueError("no face has a labelled vertex: the mesh carries no deforming weight")
        for i in np.flatnonzero(out < 0):
            d = np.sum((centres[have] - centres[i]) ** 2, axis=1)
            out[i] = out[have[np.argmin(d)]]
    return out


def split_region(
    face_label: NDArray[np.integer],
    face_centres: NDArray[np.floating],
    region: int,
    positive: int,
    negative: int,
    forward: tuple[float, float, float] | NDArray[np.floating],
) -> NDArray[np.int64]:
    """Resolve every face on ``region`` to ``positive`` or ``negative`` by the forward axis.

    The reference is the centre of the region's own faces, so a torso leaning forward or back
    still halves; the sign of ``(centre − region centre) · forward`` decides.
    """
    fl = np.array(face_label, dtype=np.int64, copy=True)
    centres = np.asarray(face_centres, dtype=np.float64)
    fwd = np.asarray(forward, dtype=np.float64)
    idx = np.flatnonzero(fl == region)
    if idx.size == 0:
        return fl
    ref = centres[idx].mean(axis=0)
    along = (centres[idx] - ref) @ fwd
    fl[idx] = np.where(along >= 0.0, positive, negative)
    return fl


def forward_check(
    face_label: NDArray[np.integer],
    face_centres: NDArray[np.floating],
    pairs: list[tuple[int, int]],
    forward: tuple[float, float, float] | NDArray[np.floating],
    up: tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> dict[int, float]:
    """Foot-minus-shin displacement along ``forward``, per (foot, shin) label pair, metres.

    The vertical component is removed first. Every value must be positive for the declared
    forward axis to be the body's; the caller decides what to do with a negative one.
    """
    fl = np.asarray(face_label, dtype=np.int64)
    centres = np.asarray(face_centres, dtype=np.float64)
    fwd = np.asarray(forward, dtype=np.float64)
    upv = np.asarray(up, dtype=np.float64)
    out: dict[int, float] = {}
    for foot, shin in pairs:
        cf, cs = centres[fl == foot], centres[fl == shin]
        if cf.size == 0 or cs.size == 0:
            continue
        d = cf.mean(axis=0) - cs.mean(axis=0)
        d = d - upv * (d @ upv)
        out[foot] = float(d @ fwd)
    return out
