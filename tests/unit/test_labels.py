"""EV.15 -- training labels from the planes that already exist.

Boxes equal the instance mask's extent to the pixel; a target under a pixel is labelled a point
with its phi; a run round-trips through a public COCO loader when one is installed. Roadmap EV.15.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest

from irsim.io.labels import (
    frame_labels,
    split_manifest,
    write_frame_labels,
    write_split_manifest,
)

LEGEND = {0: "sky", 1: "arm_n", 2: "motor_front_left", 3: "propeller_front_left", 4: "hull"}
TARGETS = {"quad": ["arm_n", "motor_front_left", "propeller_front_left"], "boat": ["hull"]}
CATEGORY = {"quad": "quadrotor", "boat": "vessel", "bird": "bird"}


def _plane() -> np.ndarray:
    ids = np.zeros((24, 32), dtype=np.uint16)
    ids[5:9, 10:14] = 1  # arm
    ids[3:5, 12:13] = 2  # motor above it
    ids[2:3, 11:14] = 3  # propeller
    ids[18:22, 2:30] = 4  # a long hull
    return ids


def test_boxes_equal_the_mask_extent_to_the_pixel() -> None:
    labels = frame_labels(_plane(), LEGEND, TARGETS, CATEGORY)
    by = {b.target: b for b in labels.boxes}
    quad = by["quad"]
    assert (quad.x1, quad.y1, quad.x2, quad.y2) == (10, 2, 14, 9)
    assert quad.pixels == 16 + 2 + 3 and quad.category == "quadrotor"
    assert quad.xywh == (10, 2, 4, 7)
    boat = by["boat"]
    assert (boat.x1, boat.y1, boat.x2, boat.y2) == (2, 18, 30, 22) and boat.pixels == 4 * 28
    cx, cy, w, h = boat.yolo(32, 24)
    assert (cx, cy, w, h) == pytest.approx((16 / 32, 20 / 24, 28 / 32, 4 / 24))
    assert labels.categories == ("quadrotor", "vessel")
    # the RLE decodes back to the mask, column-major as COCO wants it
    runs = np.array(quad.mask_rle)
    values = np.repeat(np.arange(runs.size) % 2, runs).astype(bool)
    decoded = values.reshape((24, 32), order="F")
    assert np.array_equal(decoded, np.isin(_plane(), [1, 2, 3]))


def test_a_target_absent_from_the_frame_has_no_box_and_a_sub_pixel_one_is_a_point() -> None:
    labels = frame_labels(
        _plane(),
        LEGEND,
        {"quad": TARGETS["quad"], "boat": ["hull"], "nothing": ["hull"]},
        {**CATEGORY, "nothing": "vessel"},
        points=[("bird", 20.5, 4.5, 0.3)],
    )
    assert {b.target for b in labels.boxes} == {"quad", "boat", "nothing"}
    empty = frame_labels(np.zeros((8, 8), dtype=np.int32), LEGEND, TARGETS, CATEGORY)
    assert empty.boxes == () and empty.categories == ("quadrotor", "vessel")
    assert labels.points[0].target == "bird" and labels.points[0].phi == 0.3
    doc = labels.coco(7, "f.png")
    point = [a for a in doc["annotations"] if a["attributes"].get("point")][0]
    assert point["bbox"] == [20.5, 4.5, 0.0, 0.0] and point["attributes"]["phi"] == 0.3
    with pytest.raises(ValueError, match="phi"):
        frame_labels(_plane(), LEGEND, TARGETS, CATEGORY, points=[("bird", 1, 1, 1.5)])
    with pytest.raises(KeyError, match="none of its parts"):
        frame_labels(_plane(), LEGEND, {"ghost": ["wing"]}, {"ghost": "bird"})


def test_visibility_is_the_share_of_the_mask_inside_the_frame() -> None:
    ids = _plane()
    labels = frame_labels(ids, LEGEND, TARGETS, CATEGORY, frame_shape=(20, 32))
    by = {b.target: b for b in labels.boxes}
    assert by["quad"].visibility == 1.0
    assert by["boat"].visibility == pytest.approx(2 * 28 / (4 * 28))
    assert (by["boat"].y1, by["boat"].y2) == (18, 20), "the box is the visible part"
    plain = frame_labels(ids, LEGEND, TARGETS, CATEGORY)
    assert plain.boxes[0].visibility is None, "not measured without a frame shape"


def test_the_files_round_trip_and_a_split_is_deterministic(tmp_path: pathlib.Path) -> None:
    labels = frame_labels(_plane(), LEGEND, TARGETS, CATEGORY, points=[("bird", 3.0, 3.0, 0.2)])
    coco, yolo = write_frame_labels(tmp_path, "frame_000007", labels, image_id=7)
    doc = json.loads(coco.read_text())
    assert doc["images"][0] == {"id": 7, "file_name": "frame_000007.png", "width": 32, "height": 24}
    assert {c["name"] for c in doc["categories"]} == {"quadrotor", "vessel", "bird"}
    assert len(doc["annotations"]) == 3
    lines = yolo.read_text().strip().split("\n")
    assert len(lines) == 2 and lines[0].startswith("0 ")
    cls, cx, cy, w, h = lines[0].split()
    assert (float(cx), float(w)) == pytest.approx((12 / 32, 4 / 32))
    stems = [f"frame_{i:06d}" for i in range(20)]
    split = split_manifest(stems, val_fraction=0.2, test_fraction=0.1, seed=3)
    assert split == split_manifest(stems, val_fraction=0.2, test_fraction=0.1, seed=3)
    assert len(split["val"]) == 4 and len(split["test"]) == 2 and len(split["train"]) == 14
    assert set(split["train"]) | set(split["val"]) | set(split["test"]) == set(stems)
    path = write_split_manifest(tmp_path, split, labels.categories)
    assert json.loads(path.read_text())["split"]["test"] == split["test"]
    pycocotools = pytest.importorskip("pycocotools.coco")
    coco_api = pycocotools.COCO(str(coco))
    assert coco_api.getAnnIds(imgIds=[7]) and len(coco_api.loadCats(coco_api.getCatIds())) == 3


def test_a_box_carries_the_cloud_in_front_of_it() -> None:
    """AT.29: with the frame's `cloud_transmittance` plane, every box reports the mean
    transmittance over its own mask -- the quad behind an opaque cloud reads 0, the boat in the
    clear reads 1 -- and the attribute reaches the COCO file. Without the plane it is None."""
    ids = _plane()
    tau = np.ones(ids.shape, dtype=np.float32)
    tau[:12] = 0.0  # the quad's rows are behind an opaque cloud; the hull's are clear
    labels = frame_labels(ids, LEGEND, TARGETS, CATEGORY, cloud_transmittance=tau)
    by = {b.target: b for b in labels.boxes}
    assert by["quad"].cloud_transmittance == pytest.approx(0.0)
    assert by["boat"].cloud_transmittance == pytest.approx(1.0)
    attrs = {
        a["attributes"]["target"]: a["attributes"]
        for a in labels.coco(1, "f.png")["annotations"]
        if "target" in a["attributes"]
    }
    assert attrs["quad"]["cloud_transmittance"] == pytest.approx(0.0)
    assert frame_labels(ids, LEGEND, TARGETS, CATEGORY).boxes[0].cloud_transmittance is None
    with pytest.raises(ValueError):
        frame_labels(ids, LEGEND, TARGETS, CATEGORY, cloud_transmittance=tau[:10])
