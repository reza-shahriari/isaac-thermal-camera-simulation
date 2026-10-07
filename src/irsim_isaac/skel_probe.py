"""HU.10's probe: does a skinned, walking body reach the IR pipeline's AOVs as it moves?

docs/physics-model.md §13.3 (AOVs), §6.1 (the person); roadmap HU.10; ADR 0199, ADR 0014.

Every human the project has rendered is static (ADR 0196: the labelled asset drops its rig).
A pedestrian walks, and the IR pipeline would follow a walking body only if three things held on
this build, none of them documented for UsdSkel:

1. **ids per prim.** ``instance_id_segmentation`` gives each skinned segment mesh its own id,
   mapped to its own prim path, and the id stays the same frame to frame. Temperature is looked up
   by prim (ADR 0014), so an id that changed with the pose would paint a hand with the foot's
   temperature.
2. **deformed positions.** The position AOV reports the *skinned* surface, not the rest pose. A
   renderer that skinned the colour but not the geometry AOVs would keep every pixel's position at
   rest while the image moved.
3. **float32 throughout.** No geometry plane arrives as float16 (CLAUDE.md non-negotiable 2).

The probe renders ``scripts/animate_human.py``'s walk frame by frame and holds every pixel of a
segment to **Blender's** skinned surface of that segment at that frame, never to the renderer's
own earlier output: a pixel's world position must lie on the deformed mesh (its distance to the
skinned triangles is at the depth buffer's precision, ``SURFACE_P95_M``). The control is the same
pixels against the *rest* pose: for a segment that moved, they must miss it by far more
(``CONTROL_MIN_M``), or the first test proved nothing. Centroids are reported too, but only as
context: a camera sees one side of a limb, and a leg swinging behind the other changes which side,
so a visible centroid is biased by up to a limb's radius in a way no tolerance separates from an
error (the first run of this probe failed on exactly that, ADR 0199).

The analysis (:func:`nearest_distances`, :func:`segment_centroids`, :func:`assess`) is plain
NumPy and unit-tested; only
:func:`run_probe` needs Kit.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "ABSOLUTE_TOL_M",
    "CONTROL_MIN_M",
    "SURFACE_P95_M",
    "nearest_distances",
    "surface_residuals",
    "DISPLACEMENT_TOL_M",
    "HAND_STRIDE_MIN_M",
    "FrameReading",
    "ProbeReport",
    "assess",
    "segment_centroids",
    "segment_of",
    "run_probe",
]

#: A visible centroid against a whole-mesh centroid: a limb seen from one side is offset toward
#: the camera by up to its radius (a thigh 8 cm). Displacements cancel that bias.
ABSOLUTE_TOL_M = 0.10
DISPLACEMENT_TOL_M = 0.03
#: The roadmap's criterion: a hand moves more than this over the clip.
HAND_STRIDE_MIN_M = 0.2
#: A segment needs this many pixels in a frame for its centroid to count.
MIN_PIXELS = 30
#: A pixel on the skinned surface lies on one of its triangles to the depth buffer's precision; the
#: margin covers the silhouette, where a pixel's position is interpolated across an edge.
SURFACE_P95_M = 0.005
#: The rest-pose control: a segment counts as having moved when its pixels' median distance to its
#: own rest-pose surface exceeds this, and then the deformed fit must beat the rest fit by 3x.
CONTROL_MIN_M = 0.03


def segment_of(prim_path: str) -> str | None:
    """The ``skin_<Segment>`` a prim path belongs to, or None.

    Blender's USD export nests a mesh under its object's Xform (``/root/skin_LHand/skin_LHand``),
    and Kit may label either; the first ``skin_`` component wins.
    """
    for part in prim_path.split("/"):
        if part.startswith("skin_"):
            return part
    return None


def segment_centroids(
    instance_id: NDArray[np.integer[Any]],
    id_to_path: Mapping[int, str],
    world: NDArray[np.floating[Any]],
    *,
    min_pixels: int = MIN_PIXELS,
) -> dict[str, tuple[NDArray[np.float64], int]]:
    """Mean world position and pixel count of every segment visible in one frame."""
    out: dict[str, tuple[NDArray[np.float64], int]] = {}
    ids = np.asarray(instance_id)
    pos = np.asarray(world, dtype=np.float64)
    groups: dict[str, list[int]] = {}
    for iid, path in id_to_path.items():
        seg = segment_of(path)
        if seg is not None:
            groups.setdefault(seg, []).append(int(iid))
    for seg, iids in groups.items():
        mask = np.isin(ids, iids) & np.all(np.isfinite(pos), axis=-1)
        n = int(mask.sum())
        if n >= min_pixels:
            out[seg] = (pos[mask].mean(axis=0), n)
    return out


def nearest_distances(
    points: NDArray[np.floating[Any]],
    vertices: NDArray[np.floating[Any]],
    faces: NDArray[np.integer[Any]] | None = None,
    *,
    chunk: int = 256,
) -> NDArray[np.float64]:
    """Distance from each point to a triangle mesh (or, with no ``faces``, to its nearest vertex).

    The closest point on each triangle by Ericson's region test (*Real-Time Collision Detection*,
    5.1.5), vectorised over points x triangles in chunks; no SciPy. Nearest-vertex distance is kept
    for a point cloud, but on a mesh it overstates the error by up to half a long edge.
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    vts = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
    out = np.empty(len(pts), dtype=np.float64)
    if faces is None:
        vv = np.einsum("ij,ij->i", vts, vts)
        for i in range(0, len(pts), chunk):
            p = pts[i : i + chunk]
            d2 = np.einsum("ij,ij->i", p, p)[:, None] - 2.0 * p @ vts.T + vv[None, :]
            out[i : i + chunk] = np.sqrt(np.maximum(d2.min(axis=1), 0.0))
        return out
    tri = vts[np.asarray(faces, dtype=np.int64)]
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    ab, ac = b - a, c - a
    for i in range(0, len(pts), chunk):
        p = pts[i : i + chunk, None, :]
        ap = p - a
        d1, d2 = (ap * ab).sum(-1), (ap * ac).sum(-1)
        bp = p - b
        d3, d4 = (bp * ab).sum(-1), (bp * ac).sum(-1)
        cp = p - c
        d5, d6 = (cp * ab).sum(-1), (cp * ac).sum(-1)
        va = d3 * d6 - d5 * d4
        vb = d5 * d2 - d1 * d6
        vc = d1 * d4 - d3 * d2
        with np.errstate(divide="ignore", invalid="ignore"):
            denom = 1.0 / (va + vb + vc)
            v = vb * denom
            w = vc * denom
            closest = a + ab * v[..., None] + ac * w[..., None]  # interior
            t_ab = d1 / (d1 - d3)
            t_ac = d2 / (d2 - d6)
            t_bc = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        # the edge and vertex regions, in Ericson's order (later assignments win)
        e_bc = (va <= 0) & (d4 - d3 >= 0) & (d5 - d6 >= 0)
        closest = np.where(e_bc[..., None], b + (c - b) * t_bc[..., None], closest)
        e_ac = (vb <= 0) & (d2 >= 0) & (d6 <= 0)
        closest = np.where(e_ac[..., None], a + ac * t_ac[..., None], closest)
        e_ab = (vc <= 0) & (d1 >= 0) & (d3 <= 0)
        closest = np.where(e_ab[..., None], a + ab * t_ab[..., None], closest)
        in_c = (d6 >= 0) & (d5 <= d6)
        closest = np.where(in_c[..., None], c, closest)
        in_b = (d3 >= 0) & (d4 <= d3)
        closest = np.where(in_b[..., None], b, closest)
        in_a = (d1 <= 0) & (d2 <= 0)
        closest = np.where(in_a[..., None], a, closest)
        dist = np.linalg.norm(p - closest, axis=-1)
        out[i : i + chunk] = np.nanmin(dist, axis=1)
    return out


def surface_residuals(
    points: NDArray[np.floating[Any]],
    posed: NDArray[np.floating[Any]],
    rest: NDArray[np.floating[Any]],
    faces: NDArray[np.integer[Any]] | None,
) -> tuple[float, float, float]:
    """One segment's rendered pixels against its skinned surface and its rest-pose surface.

    Returns (p95 distance to the posed surface, median to the rest surface, median to the posed
    surface), metres -- the triple :func:`assess` judges.
    """
    on_posed = nearest_distances(points, posed, faces)
    on_rest = nearest_distances(points, rest, faces)
    return (
        float(np.percentile(on_posed, 95)),
        float(np.median(on_rest)),
        float(np.median(on_posed)),
    )


@dataclass(frozen=True)
class FrameReading:
    """What the renderer reported for one frame."""

    frame: int
    id_by_segment: dict[str, tuple[int, ...]]
    centroids: dict[str, tuple[NDArray[np.float64], int]]
    dtypes: dict[str, str]
    #: per segment: (p95 distance to the frame's skinned surface, median to the rest pose's,
    #: median to the frame's), metres
    residuals: dict[str, tuple[float, float, float]] = field(default_factory=dict)
    #: per pixel, the index into ``irsim.config.humans.JOS3_SEGMENTS`` of the segment seen, -1
    #: elsewhere: what the IR pipeline's per-prim temperature lookup would paint
    segment_index: NDArray[np.int16] | None = None


@dataclass
class ProbeReport:
    frames: list[int]
    ids_stable: bool
    id_changes: list[str]
    float32: bool
    dtypes: dict[str, str]
    worst_absolute_m: float
    worst_displacement_m: float
    hand_travel_m: dict[str, float]
    per_segment: dict[str, dict[str, float]] = field(default_factory=dict)
    worst_surface_p95_m: float = math.inf
    moved_segments: list[str] = field(default_factory=list)
    control_failures: list[str] = field(default_factory=list)
    readings: list[FrameReading] = field(default_factory=list, repr=False)

    @property
    def passed(self) -> bool:
        """HU.10's criteria. The centroid errors are context, not criteria (see the module)."""
        return (
            self.ids_stable
            and self.float32
            and self.worst_surface_p95_m <= SURFACE_P95_M
            and bool(self.moved_segments)
            and not self.control_failures
            and min(self.hand_travel_m.values(), default=0.0) > HAND_STRIDE_MIN_M
        )

    def render(self) -> str:
        lines = [
            f"frames               : {self.frames}",
            f"ids stable per prim  : {self.ids_stable} {self.id_changes or ''}",
            f"float32 planes       : {self.float32} {self.dtypes}",
            f"on the skinned surface     : worst p95 {self.worst_surface_p95_m:.4f} m "
            f"(tol {SURFACE_P95_M})",
            f"moved (rest-pose control)  : {self.moved_segments}",
            f"control failures           : {self.control_failures or 'none'}",
            f"context: worst visible-centroid offset {self.worst_absolute_m:.4f} m, "
            f"displacement {self.worst_displacement_m:.4f} m",
            "hand travel (rendered)     : "
            + ", ".join(f"{k} {v:.3f} m" for k, v in self.hand_travel_m.items())
            + f" (> {HAND_STRIDE_MIN_M})",
        ]
        for seg, row in sorted(self.per_segment.items()):
            lines.append(
                f"  {seg:15s} surface p95 {row.get('surface_p95_m', math.nan):.4f}  "
                f"rest {row.get('rest_median_m', math.nan):.4f}  travel {row['travel_m']:.3f} m  "
                f"(centroid abs {row['abs_m']:.4f}, disp {row['disp_m']:.4f})"
            )
        lines.append(f"PASS: {self.passed}")
        return "\n".join(lines)


def assess(
    readings: Sequence[FrameReading],
    expected: Mapping[str, Sequence[Sequence[float]]],
    *,
    frame_start: int = 1,
) -> ProbeReport:
    """Hold the rendered frames to Blender's centroids (``expected[segment][frame - start]``)."""
    frames = [r.frame for r in readings]
    first: dict[str, tuple[int, ...]] = {}
    changes: list[str] = []
    for r in readings:
        for seg, ids in r.id_by_segment.items():
            if seg in first and first[seg] != ids:
                changes.append(f"{seg}@{r.frame}: {first[seg]} -> {ids}")
            first.setdefault(seg, ids)
    dtypes: dict[str, str] = {}
    for r in readings:
        dtypes.update(r.dtypes)
    float32 = all(d in ("float32", "float64") for d in dtypes.values())

    worst_abs = 0.0
    worst_disp = 0.0
    per: dict[str, dict[str, float]] = {}
    travel: dict[str, float] = {}
    segs = sorted({s for r in readings for s in r.centroids})
    for seg in segs:
        seen = [(r.frame, r.centroids[seg][0]) for r in readings if seg in r.centroids]
        if not seen or seg not in expected:
            continue
        want = {f: np.asarray(expected[seg][f - frame_start], dtype=np.float64) for f, _ in seen}
        abs_err = max(float(np.linalg.norm(c - want[f])) for f, c in seen)
        f0, c0 = seen[0]
        disp_err = max(
            (float(np.linalg.norm((c - c0) - (want[f] - want[f0]))) for f, c in seen[1:]),
            default=0.0,
        )
        span = max((float(np.linalg.norm(a - b)) for _, a in seen for _, b in seen), default=0.0)
        per[seg] = {"abs_m": abs_err, "disp_m": disp_err, "travel_m": span}
        worst_abs = max(worst_abs, abs_err)
        worst_disp = max(worst_disp, disp_err)
        if seg in ("skin_LHand", "skin_RHand"):
            travel[seg] = span
    worst_surface = 0.0
    moved: set[str] = set()
    control: list[str] = []
    for r in readings:
        for seg, (p95, rest_med, med) in r.residuals.items():
            worst_surface = max(worst_surface, p95)
            row = per.setdefault(seg, {"abs_m": math.nan, "disp_m": math.nan, "travel_m": 0.0})
            row["surface_p95_m"] = max(row.get("surface_p95_m", 0.0), p95)
            row["rest_median_m"] = max(row.get("rest_median_m", 0.0), rest_med)
            if rest_med > CONTROL_MIN_M:
                moved.add(seg)
                if rest_med < 3.0 * med:
                    control.append(f"{seg}@{r.frame}: rest {rest_med:.3f} vs posed {med:.3f}")
    if not any(r.residuals for r in readings):
        worst_surface = math.inf
    return ProbeReport(
        frames=frames,
        ids_stable=not changes,
        id_changes=changes,
        float32=float32,
        dtypes=dtypes,
        worst_absolute_m=worst_abs,
        worst_displacement_m=worst_disp,
        hand_travel_m=travel,
        per_segment=per,
        worst_surface_p95_m=worst_surface,
        moved_segments=sorted(moved),
        control_failures=control,
    )


def run_probe(
    usd_path: str,
    expected: Mapping[str, Any],
    *,
    vertices: Mapping[str, NDArray[Any]] | None = None,
    frames: Sequence[int],
    resolution: tuple[int, int] = (768, 768),
    camera_distance_m: float = 4.5,
) -> ProbeReport:  # pragma: no cover - needs Kit
    """Open the walk in Kit, render ``frames`` from in front, and :func:`assess` them."""
    import omni.replicator.core as rep
    import omni.timeline
    import omni.usd
    from pxr import Gf, UsdGeom

    from irsim.config.humans import JOS3_SEGMENTS
    from irsim_isaac.pipeline.gbuffer_isaac import AovReader, camera_pose
    from irsim_isaac.pipeline.material_ids import labels_from_payload
    from irsim_isaac.pipeline.point_bridge import world_positions

    ctx = omni.usd.get_context()
    ok = ctx.open_stage(usd_path)
    if not (ok[0] if isinstance(ok, tuple) else bool(ok)):
        raise RuntimeError(f"Kit could not open {usd_path}")
    stage = ctx.get_stage()
    if UsdGeom.GetStageUpAxis(stage) != UsdGeom.Tokens.z:
        raise RuntimeError("the walk stage must be Z-up, as Blender writes it")
    tcps = stage.GetTimeCodesPerSecond()

    # In front of the body (it faces -Y), level with the hips, looking along +Y.
    cam_path = "/IrsimProbeCamera"
    cam = UsdGeom.Camera.Define(stage, cam_path)
    cam.CreateFocalLengthAttr(24.0)
    cam.CreateHorizontalApertureAttr(20.955)
    cam.CreateVerticalApertureAttr(20.955)
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.05, 100.0))
    xf = UsdGeom.Xformable(cam.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(0.0, -camera_distance_m, 0.95))
    xf.AddRotateXYZOp().Set(Gf.Vec3f(90.0, 0.0, 0.0))

    width, height = resolution
    rp = rep.create.render_product(cam_path, (width, height))
    reader = AovReader(rp.path, expected_shape=(height, width)).attach(settle_frames=6)
    timeline = omni.timeline.get_timeline_interface()
    t_cam, r_cam = camera_pose(cam_path, stage=stage)

    readings: list[FrameReading] = []
    frame_start = int(expected.get("frame_start", 1))
    for frame in frames:
        timeline.set_current_time(frame / tcps)
        timeline.commit()
        for _ in range(3):
            rep.orchestrator.step(rt_subframes=4, delta_time=0.0)
        raw = reader.read()
        id_to_path = labels_from_payload(raw.device_handles.get("instance"))
        world = world_positions(
            raw.position, frame="camera", camera_position=t_cam, camera_to_world=r_cam
        )
        ids = raw.instance_id
        assert ids is not None, "the instance channel did not attach"
        by_seg: dict[str, list[int]] = {}
        present = set(np.unique(ids).tolist())
        for iid, path in id_to_path.items():
            seg = segment_of(path)
            if seg is not None and iid in present:
                by_seg.setdefault(seg, []).append(int(iid))
        residuals: dict[str, tuple[float, float, float]] = {}
        if vertices is not None:
            k = frame - frame_start
            for seg, iids in by_seg.items():
                if seg not in vertices:
                    continue
                mask = np.isin(ids, iids) & np.all(np.isfinite(world), axis=-1)
                if int(mask.sum()) < MIN_PIXELS:
                    continue
                pts = world[mask]
                residuals[seg] = surface_residuals(
                    pts, vertices[seg][k], vertices[seg][0], vertices.get(f"{seg}.faces")
                )
        seg_index = np.full(ids.shape[:2], -1, dtype=np.int16)
        for seg, iids in by_seg.items():
            name = seg[len("skin_") :]
            if name in JOS3_SEGMENTS:
                seg_index[np.isin(ids, iids)] = JOS3_SEGMENTS.index(name)
        readings.append(
            FrameReading(
                frame=frame,
                residuals=residuals,
                segment_index=seg_index,
                id_by_segment={s: tuple(sorted(v)) for s, v in by_seg.items()},
                centroids=segment_centroids(ids, id_to_path, world),
                dtypes={
                    "position": str(np.asarray(raw.position).dtype),
                    "distance": str(np.asarray(raw.distance_m).dtype),
                    "normal": str(np.asarray(raw.normal).dtype),
                },
            )
        )
    reader.detach()
    report = assess(readings, expected["centroids_m"], frame_start=frame_start)
    report.readings = readings
    if not math.isfinite(report.worst_absolute_m):
        raise RuntimeError("non-finite positions on segment pixels")
    return report
