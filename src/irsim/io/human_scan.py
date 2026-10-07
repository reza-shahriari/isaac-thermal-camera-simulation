"""Label a fused human scan from predicted bone weights, and score it against a known answer (HU.9).

docs/physics-model.md §6.1; roadmap HU.9; ADR 0192 (labelling by skeleton), ADR 0201.

A downloaded human scan arrives as one mesh with no skeleton. Make-It-Animatable (MIT, run by
``scripts/autorig_mia.py`` in its own environment) predicts Mixamo bone weights for every vertex;
this module turns those weights into the body schema's seventeen segments with the *same*
arithmetic ``prep_human.py`` applies to a rig the artist made (:mod:`irsim.io.human_labels`:
heaviest bone per vertex, a weighted vote per face, the torso split into Chest and Back by the
forward axis). Nothing here is new labelling logic -- only the bridge from an ``.npz`` to it.

It also scores the result, because "the rigger labelled 98 % of faces" says nothing about whether
the labels are right. ``scripts/fuse_human.py`` makes a test scan from a person the project already
labelled by its own skeleton, with the answer per triangle; :func:`score` compares the predicted
segment of every *skin* triangle with that answer, by area, per segment and in total. The answer is
a skeleton's labelling, not anatomy measured on a person: the score says how closely an auto-rig
reproduces a hand-made rig's segmentation, which is what the thermal model consumes.

NumPy only; tested without Blender, torch or the tool (``tests/unit/test_human_scan.py``).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from irsim.config.humans import BodySchema, segment_for_bone
from irsim.io.human_labels import face_labels, split_region, vertex_labels

__all__ = [
    "REVIEW_BAND",
    "SKIN_DELTA_E",
    "SMOOTH_RINGS",
    "ClassScore",
    "LabelScore",
    "classify_surfaces",
    "face_neighbours",
    "smooth_on_mesh",
    "score_classes",
    "srgb_to_lab",
    "forward_from_joints",
    "label_faces_from_weights",
    "nearest",
    "score",
]


def label_faces_from_weights(
    schema: BodySchema,
    rig: str,
    bones: Sequence[str],
    weights: NDArray[np.floating],
    vertices: NDArray[np.floating],
    faces: NDArray[np.integer],
    *,
    forward: NDArray[np.floating],
) -> list[str]:
    """The body segment of every face, from per-vertex bone weights (columns named ``bones``).

    ``forward`` is the direction the body faces in the vertices' frame (:func:`forward_from_joints`
    for a scan, whose facing nobody declared); it splits the torso into Chest and Back.
    """
    labels = sorted(set(schema.segment_names) | set(schema.regions))
    index_of = {lab: i for i, lab in enumerate(labels)}
    group_label = np.array(
        [index_of[t] if (t := segment_for_bone(schema, rig, b)) is not None else -1 for b in bones],
        dtype=np.int64,
    )
    w = np.asarray(weights, dtype=np.float64)
    if w.shape[1] != len(bones):
        raise ValueError(f"weights have {w.shape[1]} columns for {len(bones)} bones")
    vlab = vertex_labels(w, group_label)
    f = np.asarray(faces, dtype=np.int64)
    centres = np.asarray(vertices, dtype=np.float64)[f].mean(axis=1)
    flab = face_labels(f, vlab, centres)
    fwd = np.asarray(forward, dtype=np.float64)
    fwd = fwd / np.linalg.norm(fwd)
    for region, split in schema.regions.items():
        flab = split_region(
            flab,
            centres,
            index_of[region],
            index_of[split.positive],
            index_of[split.negative],
            fwd,
        )
    return [labels[i] for i in flab]


def forward_from_joints(
    bones: Sequence[str], joints_head: NDArray[np.floating], *, prefix: str = "mixamorig:"
) -> NDArray[np.float64]:
    """The way a Mixamo-rigged body faces, from its own joints, in their frame.

    Up runs from the hips to the head; the body's left runs from the right thigh's head to the
    left's. In a right-handed frame left = up x forward, so forward = left x up -- made orthogonal
    to up. A scan carries no declared facing, and a guessed one swaps Chest and Back silently.
    """
    at = {b: i for i, b in enumerate(bones)}
    j = np.asarray(joints_head, dtype=np.float64)
    up = j[at[f"{prefix}Head"]] - j[at[f"{prefix}Hips"]]
    up /= np.linalg.norm(up)
    left = j[at[f"{prefix}LeftUpLeg"]] - j[at[f"{prefix}RightUpLeg"]]
    fwd = np.cross(left, up)
    fwd -= up * float(fwd @ up)
    out: NDArray[np.float64] = fwd / np.linalg.norm(fwd)
    return out


def nearest(
    queries: NDArray[np.floating], points: NDArray[np.floating], *, chunk: int = 1024
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """Index of and distance to the nearest of ``points`` per query (brute force, no SciPy)."""
    q = np.asarray(queries, dtype=np.float64).reshape(-1, 3)
    p = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    pp = np.einsum("ij,ij->i", p, p)
    idx = np.empty(len(q), dtype=np.int64)
    dist = np.empty(len(q), dtype=np.float64)
    for i in range(0, len(q), chunk):
        a = q[i : i + chunk]
        d2 = np.einsum("ij,ij->i", a, a)[:, None] - 2.0 * a @ p.T + pp[None, :]
        j = np.argmin(d2, axis=1)
        idx[i : i + chunk] = j
        dist[i : i + chunk] = np.sqrt(np.maximum(d2[np.arange(len(a)), j], 0.0))
    return idx, dist


@dataclass(frozen=True)
class LabelScore:
    """Predicted segments against a skeleton-labelled answer, over the answer's skin faces."""

    agreement: float  # area fraction of skin faces whose predicted segment is the answer's
    per_segment: dict[str, float]  # the same, per answer segment
    confusions: list[tuple[str, str, float]] = field(default_factory=list)  # (truth, pred, area %)
    skin_area_m2: float = 0.0
    worst_match_m: float = 0.0  # the largest truth-to-prediction centre distance used

    def render(self) -> str:
        lines = [
            f"skin-face agreement (by area): {self.agreement:.4f} over {self.skin_area_m2:.3f} m2"
        ]
        for seg, v in sorted(self.per_segment.items(), key=lambda kv: kv[1]):
            lines.append(f"  {seg:10s} {v:.4f}")
        for t, p, a in self.confusions[:8]:
            lines.append(f"  confused {t} -> {p}: {a:.2f} % of skin area")
        lines.append(f"worst centre match: {self.worst_match_m * 1000:.3f} mm")
        return "\n".join(lines)


def score(
    truth_labels: Sequence[str],
    truth_centres: NDArray[np.floating],
    truth_areas: NDArray[np.floating],
    pred_labels: Sequence[str],
    pred_centres: NDArray[np.floating],
) -> LabelScore:
    """Score predicted segments on the answer's ``skin_<Segment>`` faces, matched by face centre."""
    t_lab = np.asarray(truth_labels)
    skin = np.char.startswith(t_lab.astype(str), "skin_")
    t_seg = np.char.replace(t_lab[skin].astype(str), "skin_", "")
    areas = np.asarray(truth_areas, dtype=np.float64)[skin]
    idx, dist = nearest(np.asarray(truth_centres)[skin], pred_centres)
    p_seg = np.asarray(pred_labels)[idx].astype(str)
    ok = t_seg == p_seg
    total = float(areas.sum())
    per = {
        s: float(areas[(t_seg == s) & ok].sum() / areas[t_seg == s].sum())
        for s in sorted(set(t_seg.tolist()))
    }
    conf: dict[tuple[str, str], float] = {}
    for t, p, a in zip(t_seg[~ok], p_seg[~ok], areas[~ok], strict=True):
        conf[(str(t), str(p))] = conf.get((str(t), str(p)), 0.0) + float(a)
    confusions = sorted(
        ((t, p, 100.0 * a / total) for (t, p), a in conf.items()), key=lambda r: -r[2]
    )
    return LabelScore(
        agreement=float(areas[ok].sum() / total),
        per_segment=per,
        confusions=confusions,
        skin_area_m2=total,
        worst_match_m=float(dist.max()) if len(dist) else 0.0,
    )


# ---------------------------------------------------------------------------------------------
# Skin, hair or garment: the person's own skin colour, not a skin-tone locus
# ---------------------------------------------------------------------------------------------

#: The CIE76 colour difference (Delta E*ab) from the person's skin reference above which a face is
#: not skin. Two colours about 2.3 apart are a just-noticeable difference (Sharma 2003, *Digital
#: Color Imaging Handbook*); a scan's skin texture (pores, baked shading, lips) spreads far wider.
#: Averaged over three rings of neighbours, skin's 99th percentile was 18-25 on all four test
#: people (``fuse_human.py``), so the threshold is that bound plus about two JNDs. Set on those four
#: MakeHuman people; a real scan is the held-out check (HU.9).
SKIN_DELTA_E = 30.0
#: Rings of vertex-sharing neighbours the colour difference is averaged over before it is
#: thresholded: a scan's skin texture carries pores, baked shading and lips, single triangles of
#: which differ from the face's median by more than leather does.
SMOOTH_RINGS = 3
#: A face whose smoothed difference lies within this many Delta E*ab of the threshold is decided
#: anyway but listed for a person (``review``): about 2.5 just-noticeable differences either side,
#: where skin's own tail and the nearest garments' (white cotton, tan leather) meet.
REVIEW_BAND = 6.0


def srgb_to_lab(rgb: NDArray[np.floating]) -> NDArray[np.float64]:
    """Display sRGB in 0..1 -> CIELAB (D65), by IEC 61966-2-1 and CIE 15:2004."""
    c = np.clip(np.asarray(rgb, dtype=np.float64), 0.0, 1.0)
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    m = np.array(
        [[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750],
         [0.0193339, 0.1191920, 0.9503041]]
    )  # fmt: skip
    xyz = lin @ m.T / np.array([0.95047, 1.0, 1.08883])
    eps, kappa = 216.0 / 24389.0, 24389.0 / 27.0
    f = np.where(xyz > eps, np.cbrt(xyz), (kappa * xyz + 16.0) / 116.0)
    out: NDArray[np.float64] = np.stack(
        [
            116.0 * f[..., 1] - 16.0,
            500.0 * (f[..., 0] - f[..., 1]),
            200.0 * (f[..., 1] - f[..., 2]),
        ],
        axis=-1,
    )
    return out


def face_neighbours(
    faces: NDArray[np.integer], n_vertices: int
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Pairs ``(i, j)`` of faces that share a vertex (both directions, no self pairs)."""
    f = np.asarray(faces, dtype=np.int64)
    vf_face = np.repeat(np.arange(len(f)), f.shape[1])
    vf_vert = f.ravel()
    order = np.argsort(vf_vert, kind="stable")
    v_sorted, f_sorted = vf_vert[order], vf_face[order]
    starts = np.searchsorted(v_sorted, np.arange(n_vertices + 1))
    rows, cols = [], []
    for v in range(n_vertices):
        fs = f_sorted[starts[v] : starts[v + 1]]
        if len(fs) > 1:
            a, b = np.meshgrid(fs, fs, indexing="ij")
            keep = a != b
            rows.append(a[keep])
            cols.append(b[keep])
    if not rows:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    pairs = np.unique(np.c_[np.concatenate(rows), np.concatenate(cols)], axis=0)
    return pairs[:, 0], pairs[:, 1]


def smooth_on_mesh(
    values: NDArray[np.floating],
    areas: NDArray[np.floating],
    neighbours: tuple[NDArray[np.int64], NDArray[np.int64]],
    rings: int,
) -> NDArray[np.float64]:
    """Area-weighted mean of a per-face value over ``rings`` rings of vertex-sharing neighbours.

    Only faces that share a vertex mix: a garment shell standing off the skin beneath it is a
    separate surface and never averages with it.
    """
    x = np.asarray(values, dtype=np.float64)
    w = np.asarray(areas, dtype=np.float64)
    i, j = neighbours
    for _ in range(rings):
        num = w * x + np.bincount(i, weights=w[j] * x[j], minlength=len(x))
        den = w + np.bincount(i, weights=w[j], minlength=len(x))
        x = num / den
    return x


def classify_surfaces(
    schema: BodySchema,
    segments: Sequence[str],
    rgb_srgb: NDArray[np.floating],
    centres: NDArray[np.floating],
    areas: NDArray[np.floating],
    forward: NDArray[np.floating],
    up: NDArray[np.floating],
    *,
    faces: NDArray[np.integer] | None = None,
    n_vertices: int | None = None,
    rings: int = SMOOTH_RINGS,
) -> tuple[list[str], dict[str, Any]]:
    """``skin_<Segment>``, ``garment_<slot>`` or ``hair`` for every face of a fused scan.

    The skin reference is the person's own: the area-weighted median CIELAB colour of the front,
    lower half of the head (the face -- the one region a clothed scan almost always shows bare,
    and where hair and hats are not). Each face's CIE76 difference from it is averaged over
    ``rings`` rings of vertex-sharing neighbours (when ``faces`` are given), and a face is skin
    when that lies under :data:`SKIN_DELTA_E`. (Otsu's threshold of the body's own histogram was
    tried: a body with little clothing has no second mode, and it split the bare man's skin at 17.)
    A fixed skin-tone locus would be biased by skin tone
    and is fooled by brown leather, which sits inside every published locus. Non-skin faces on the
    head above the eyes' height are ``hair``, below it the face's own (eyes, lips: ``skin_Head``);
    every other non-skin face is a garment on the slot
    whose default coverage includes its segment (``body_schema.yaml``). Returns the labels and a
    record: the reference (``L, a, b``), the ``threshold``, and ``review`` -- a boolean per face,
    true within :data:`REVIEW_BAND` of the threshold, where colour cannot decide and a person
    (or a segmentation tier, HU.9's plan) should.
    """
    seg = np.asarray(segments).astype(str)
    lab = srgb_to_lab(rgb_srgb)
    c = np.asarray(centres, dtype=np.float64)
    w = np.asarray(areas, dtype=np.float64)
    fwd = np.asarray(forward, dtype=np.float64) / np.linalg.norm(forward)
    u = np.asarray(up, dtype=np.float64) / np.linalg.norm(up)
    head = seg == "Head"
    if not head.any():
        raise ValueError("no Head faces: the skin reference is taken from the face")
    hc = np.average(c[head], axis=0, weights=w[head])
    h_up = (c - hc) @ u
    h_fwd = (c - hc) @ fwd
    face = head & (h_fwd > 0.0) & (h_up < 0.0)
    if face.sum() < 20:
        raise ValueError("too few face triangles for a skin reference")

    def wmedian(x: NDArray[np.float64], wt: NDArray[np.float64]) -> float:
        o = np.argsort(x)
        cw = np.cumsum(wt[o])
        return float(x[o][np.searchsorted(cw, 0.5 * cw[-1])])

    ref = np.array([wmedian(lab[face, k], w[face]) for k in range(3)])
    de = np.linalg.norm(lab - ref, axis=1)
    if faces is not None:
        nv = int(n_vertices if n_vertices is not None else np.asarray(faces).max() + 1)
        de = smooth_on_mesh(de, w, face_neighbours(faces, nv), rings)
    threshold = SKIN_DELTA_E
    skin = de <= threshold
    review = np.abs(de - threshold) < REVIEW_BAND
    # the eyes' height: the head's centre is near it on an adult head. Above it, non-skin colour on
    # the head is hair (or a hat, which a person confirms); below it, it is the face's own -- eyes,
    # lips, brows -- not a garment, since a head garment sits above the eyes
    hair = head & ~skin & (h_up > 0.0)
    skin = skin | (head & ~skin & (h_up <= 0.0))
    slot_of = {s: slot for slot, g in schema.garment_slots.items() for s in g.covers}
    slot_of.setdefault("Neck", "torso")
    out: list[str] = []
    for i in range(len(seg)):
        if skin[i]:
            out.append(f"skin_{seg[i]}")
        elif hair[i]:
            out.append("hair")
        else:
            out.append(f"garment_{slot_of.get(seg[i], 'torso')}")
    return out, {
        "L": float(ref[0]),
        "a": float(ref[1]),
        "b": float(ref[2]),
        "threshold": threshold,
        "review": review,
    }


#: The truth objects of a skeleton-labelled person, as surface classes.
_TRUTH_CLASS = {"eyes": "skin", "eyelashes": "hair", "eyebrows": "hair", "hair": "hair"}


def _surface_class(label: str) -> str:
    if label.startswith("skin_"):
        return "skin"
    if label.startswith(("garment_", "equipment_")):
        return "garment"
    return _TRUTH_CLASS.get(label, "other")


@dataclass(frozen=True)
class ClassScore:
    """Skin / hair / garment against the answer, by area."""

    accuracy: float
    per_class: dict[str, float]  # recall per truth class
    confusions: list[tuple[str, str, float]]  # (truth, predicted, % of all area)
    garment_slot_accuracy: float  # of truth garment faces called garment: the right slot
    confident_accuracy: float = 1.0  # accuracy over faces not listed for review
    review_fraction: float = 0.0  # area fraction listed for a person

    def render(self) -> str:
        lines = [f"surface-class accuracy (by area): {self.accuracy:.4f}"]
        lines += [f"  recall {k:8s} {v:.4f}" for k, v in sorted(self.per_class.items())]
        lines += [f"  {t} -> {p}: {a:.2f} %" for t, p, a in self.confusions[:6]]
        lines.append(f"garment slot accuracy: {self.garment_slot_accuracy:.4f}")
        lines.append(
            f"decided without a person: {1 - self.review_fraction:.4f} of the area, "
            f"accuracy there {self.confident_accuracy:.4f}"
        )
        return "\n".join(lines)


def score_classes(
    truth_labels: Sequence[str],
    areas: NDArray[np.floating],
    predicted: Sequence[str],
    review: NDArray[np.bool_] | None = None,
) -> ClassScore:
    """Score :func:`classify_surfaces` face by face (the rows must already correspond)."""
    t = [_surface_class(x) for x in truth_labels]
    p = [_surface_class(x) for x in predicted]
    w = np.asarray(areas, dtype=np.float64)
    tt, pp = np.array(t), np.array(p)
    ok = tt == pp
    per = {k: float(w[(tt == k) & ok].sum() / w[tt == k].sum()) for k in sorted(set(t))}
    conf: dict[tuple[str, str], float] = {}
    for a, b, x in zip(tt[~ok], pp[~ok], w[~ok], strict=True):
        conf[(str(a), str(b))] = conf.get((str(a), str(b)), 0.0) + float(x)
    total = float(w.sum())
    g = (tt == "garment") & (pp == "garment")
    tl, pl = np.asarray(truth_labels).astype(str), np.asarray(predicted).astype(str)
    slot_ok = float(w[g & (tl == pl)].sum() / w[g].sum()) if g.any() else 1.0
    rv = np.zeros(len(w), dtype=bool) if review is None else np.asarray(review, dtype=bool)
    sure = ~rv
    confident = float(w[sure & ok].sum() / w[sure].sum()) if sure.any() else 1.0
    return ClassScore(
        confident_accuracy=confident,
        review_fraction=float(w[rv].sum() / total),
        accuracy=float(w[ok].sum() / total),
        per_class=per,
        confusions=sorted(
            ((a, b, 100 * x / total) for (a, b), x in conf.items()), key=lambda r: -r[2]
        ),
        garment_slot_accuracy=slot_ok,
    )
