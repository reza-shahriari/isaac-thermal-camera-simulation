"""HU.9 -- labelling a fused scan from predicted bone weights, and scoring it, without the tool.

What would be wrong and still look labelled: a facing guessed instead of measured (Chest and Back
swapped on every scan that faces the other way), Mixamo's ``mixamorig:`` names not reaching the rig
map (every face unlabelled, or one bone's label everywhere), and a score that counted garment or
eye faces as skin (inflating agreement with faces the segment labelling never claimed). Each is a
case here, on small meshes whose answer is known by construction.

docs/physics-model.md §6.1; roadmap HU.9; ADR 0192, ADR 0201.
"""

from __future__ import annotations

import numpy as np
import pytest

from irsim.config.humans import load_body_schema
from irsim.io.human_scan import forward_from_joints, label_faces_from_weights, nearest, score

P = "mixamorig:"
#: A Mixamo-ish standing body in the glTF frame: y up, facing +z, so its left is +x.
JOINTS = {
    "Hips": (0.0, 1.0, 0.0),
    "Head": (0.0, 1.6, 0.0),
    "LeftUpLeg": (0.1, 0.95, 0.0),
    "RightUpLeg": (-0.1, 0.95, 0.0),
}


def _joints(mirror: bool = False) -> tuple[list[str], np.ndarray]:
    names = [P + n for n in JOINTS]
    pts = np.array(list(JOINTS.values()))
    if mirror:  # the same body turned to face -z: its left is now -x
        pts[:, 0] *= -1.0
    return names, pts


def test_the_facing_comes_from_the_joints_and_turns_with_the_body() -> None:
    names, pts = _joints()
    assert forward_from_joints(names, pts) == pytest.approx([0.0, 0.0, 1.0])
    names, pts = _joints(mirror=True)
    assert forward_from_joints(names, pts) == pytest.approx([0.0, 0.0, -1.0])


def _patch(centre: tuple[float, float, float]) -> tuple[np.ndarray, np.ndarray]:
    c = np.array(centre)
    v = c + np.array([[0, 0, 0], [0.01, 0, 0], [0, 0.01, 0]])
    return v, np.array([[0, 1, 2]])


def test_weights_become_segments_and_the_torso_splits_by_the_measured_facing() -> None:
    schema = load_body_schema()
    bones = [P + "LeftForeArm", P + "Spine2", P + "Head"]
    parts = [  # (centre, heaviest bone column)
        ((0.4, 1.2, 0.0), 0),  # a forearm patch
        ((0.0, 1.3, 0.12), 1),  # torso, in front (+z)
        ((0.0, 1.3, -0.12), 1),  # torso, behind
        ((0.0, 1.65, 0.0), 2),  # the head
    ]
    verts, faces, w = [], [], []
    for k, (c, col) in enumerate(parts):
        v, f = _patch(c)
        verts.append(v)
        faces.append(f + 3 * k)
        row = np.zeros((3, len(bones)))
        row[:, col] = 1.0
        w.append(row)
    vertices, tris, weights = np.vstack(verts), np.vstack(faces), np.vstack(w)
    got = label_faces_from_weights(
        schema, "mixamo", bones, weights, vertices, tris, forward=np.array([0.0, 0.0, 1.0])
    )
    assert got == ["LArm", "Chest", "Back", "Head"]
    flipped = label_faces_from_weights(
        schema, "mixamo", bones, weights, vertices, tris, forward=np.array([0.0, 0.0, -1.0])
    )
    assert flipped[1:3] == ["Back", "Chest"], "the facing, not the geometry, decides Chest"


def test_the_score_counts_skin_by_area_and_ignores_garments_and_eyes() -> None:
    centres = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]], dtype=float)
    truth = ["skin_Head", "skin_LArm", "garment_torso", "eyes"]
    areas = np.array([3.0, 1.0, 100.0, 100.0])
    perfect = score(truth, centres, areas, ["Head", "LArm", "Chest", "Head"], centres + 1e-4)
    assert perfect.agreement == pytest.approx(1.0)
    wrong_arm = score(truth, centres, areas, ["Head", "RArm", "Head", "Head"], centres)
    assert wrong_arm.agreement == pytest.approx(0.75)  # 3 of the 4 m2 of skin
    assert wrong_arm.per_segment == {"Head": 1.0, "LArm": 0.0}
    assert wrong_arm.confusions[0][:2] == ("LArm", "RArm")


def test_nearest_is_exact() -> None:
    pts = np.random.default_rng(3).normal(size=(500, 3))
    q = pts[[7, 99, 401]] + 1e-6
    idx, d = nearest(q, pts)
    assert idx.tolist() == [7, 99, 401] and d.max() < 1e-5


def _body(skin_rgb, top_rgb, hair_rgb):  # type: ignore[no-untyped-def]
    """Face centres of a crude standing body in the glTF frame (y up, facing +z)."""
    rng = np.random.default_rng(0)
    rows = []  # (segment, centre, colour)
    for _ in range(200):  # the face: front, lower half of the head
        rows.append(("Head", (rng.uniform(-0.05, 0.05), rng.uniform(1.55, 1.6), 0.08), skin_rgb))
    for _ in range(100):  # the top of the head: hair
        rows.append(("Head", (rng.uniform(-0.05, 0.05), rng.uniform(1.66, 1.72), -0.02), hair_rgb))
    for _ in range(300):  # the chest, under a top
        rows.append(("Chest", (rng.uniform(-0.15, 0.15), rng.uniform(1.2, 1.4), 0.1), top_rgb))
    for _ in range(100):  # bare hands
        rows.append(("LHand", (0.4, rng.uniform(0.8, 0.9), 0.0), skin_rgb))
    seg = [r[0] for r in rows]
    centres = np.array([r[1] for r in rows], dtype=float)
    rgb = np.array([r[2] for r in rows], dtype=float)
    return seg, centres, rgb


def test_skin_is_this_persons_skin_whatever_its_tone() -> None:
    from irsim.io.human_scan import classify_surfaces

    schema = load_body_schema()
    up, fwd = np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])
    pale, dark = (0.85, 0.65, 0.55), (0.36, 0.23, 0.17)
    beige_top = (0.82, 0.63, 0.52)  # the colour of pale skin
    hair = (0.10, 0.07, 0.05)
    for skin, expect_top, expect_hair in (
        (pale, "skin_Chest", "hair"),
        # KNOWN LIMITATION (ADR 0201): black hair on dark skin is ~25 Delta E*ab from it, under
        # the threshold, so the colour tier calls it skin -- a bias by skin tone. In LWIR hair
        # takes the head's temperature either way and the RGB keeps the scan's own texture; the
        # fix is a segmentation tier, not a threshold. Pinned so a change to it is deliberate.
        (dark, "garment_torso", "skin_Head"),
    ):
        seg, c, rgb = _body(skin, beige_top, hair)
        got, rec = classify_surfaces(schema, seg, rgb, c, np.ones(len(seg)), fwd, up)
        assert set(got[:200]) == {"skin_Head"}
        assert set(got[200:300]) == {expect_hair}, skin
        assert set(got[300:600]) == {expect_top}, (skin, rec["L"])
        assert set(got[600:]) == {"skin_LHand"}
    # a pale person's beige top is a garment a person must confirm: it is not decided silently
    seg, c, rgb = _body(pale, (0.52, 0.40, 0.30), hair)  # 27 Delta E*ab from the skin
    got, rec = classify_surfaces(schema, seg, rgb, c, np.ones(len(seg)), fwd, up)
    assert rec["review"][300:600].all()
