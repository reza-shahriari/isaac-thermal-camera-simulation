"""HU.10 -- the skinned-body probe's arithmetic, without Kit.

The probe (``scripts/probe_isaac_skel.py``) holds every rendered pixel of a segment to Blender's
skinned surface of that segment at that frame, with the rest pose as the control. What would be
wrong and pass anyway: a distance that read zero for points off the mesh (every frame would pass),
a renderer that skinned the image but left the geometry AOVs at rest (it must fail), and an id that
moved from one prim to another between frames (a hand painted with a foot's temperature). Each is
a case here, on a two-triangle "segment" whose motion is known exactly.

docs/physics-model.md §13.3; roadmap HU.10; ADR 0199.
"""

from __future__ import annotations

import numpy as np
import pytest

from irsim_isaac.skel_probe import (
    FrameReading,
    assess,
    nearest_distances,
    segment_centroids,
    segment_of,
    surface_residuals,
)

#: A unit square in the z = 0 plane, as two triangles.
SQUARE = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float64)
FACES = np.array([[0, 1, 2], [0, 2, 3]])


def test_distance_to_a_triangle_is_exact_in_every_region() -> None:
    pts = np.array(
        [
            [0.25, 0.5, 0.0],  # on the face
            [0.25, 0.5, 0.3],  # above the face
            [0.5, -0.4, 0.0],  # beyond an edge
            [-0.3, -0.4, 0.0],  # beyond a corner
            [2.0, 2.0, 1.0],  # far off a corner
        ]
    )
    d = nearest_distances(pts, SQUARE, FACES)
    assert d == pytest.approx([0.0, 0.3, 0.4, 0.5, np.sqrt(3.0)], abs=1e-12)
    # nearest-vertex distance overstates a point inside a face, which is why the probe uses faces
    assert nearest_distances(pts[:1], SQUARE)[0] == pytest.approx(np.hypot(0.25, 0.5))


def test_a_renderer_that_skins_passes_and_one_that_leaves_positions_at_rest_fails() -> None:
    rng = np.random.default_rng(1)
    uv = rng.uniform(0.05, 0.95, size=(400, 2))
    rest_pts = np.c_[uv, np.zeros(len(uv))]
    shift = np.array([0.0, 0.3, 0.1])  # the "hand" moves 0.32 m by frame 2
    posed = SQUARE + shift
    good = surface_residuals(rest_pts + shift, posed, SQUARE, FACES)
    stuck = surface_residuals(rest_pts, posed, SQUARE, FACES)
    assert good[0] < 1e-9 and good[1] > 0.03
    assert stuck[0] > 0.05

    def reading(frame: int, res: tuple[float, float, float]) -> FrameReading:
        c = (np.zeros(3) if frame == 1 else shift) + 0.5
        return FrameReading(
            frame=frame,
            id_by_segment={"skin_LHand": (7,)},
            centroids={"skin_LHand": (c, 400)},
            dtypes={"position": "float32"},
            residuals={"skin_LHand": res},
        )

    expected = {"skin_LHand": [[0.5, 0.5, 0.5], list(shift + 0.5)]}
    r0 = surface_residuals(rest_pts, SQUARE, SQUARE, FACES)
    ok = assess([reading(1, r0), reading(2, good)], expected)
    assert ok.passed, ok.render()
    bad = assess([reading(1, r0), reading(2, stuck)], expected)
    assert not bad.passed


def test_an_id_that_changes_prim_between_frames_fails() -> None:
    base = dict(centroids={}, dtypes={"position": "float32"})
    readings = [
        FrameReading(frame=1, id_by_segment={"skin_LHand": (7,)}, **base),  # type: ignore[arg-type]
        FrameReading(frame=2, id_by_segment={"skin_LHand": (9,)}, **base),  # type: ignore[arg-type]
    ]
    report = assess(readings, {})
    assert not report.ids_stable and report.id_changes
    float16 = FrameReading(frame=1, id_by_segment={}, centroids={}, dtypes={"position": "float16"})
    assert not assess([float16], {}).float32


def test_segments_are_found_by_prim_path_and_averaged_over_their_pixels() -> None:
    assert segment_of("/root/skin_LHand/skin_LHand") == "skin_LHand"
    assert segment_of("/root/eyes/eyes") is None
    ids = np.zeros((10, 10), dtype=np.uint32)
    ids[:5] = 3
    ids[5:] = 4
    world = np.zeros((10, 10, 3))
    world[:5, :, 0] = 1.0
    world[5:, :, 0] = 2.0
    out = segment_centroids(ids, {3: "/r/skin_Head/m", 4: "/r/skin_Neck/m"}, world, min_pixels=10)
    assert out["skin_Head"][0][0] == pytest.approx(1.0) and out["skin_Head"][1] == 50
    assert out["skin_Neck"][0][0] == pytest.approx(2.0)
