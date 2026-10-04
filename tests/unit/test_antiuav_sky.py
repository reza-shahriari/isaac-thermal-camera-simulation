"""The sky-only reduction of Anti-UAV RGBT: the rule that keeps a frame, and the box it carries.

The classifier is somebody's model and is measured by cross-validation in the build script; what
is pinned here is the arithmetic around it, because a wrong neighbour rule lets buildings into the
real test set and a wrong box conversion moves every label.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from irsim_eval.antiuav import (
    SPLITS,
    accept_sky_frames,
    labelled_frames,
    load_sky_labels,
    sampled_frames,
    yolo_line,
)

LABELS = Path(__file__).resolve().parents[2] / "data/validation/anti_uav_rgbt_sky_labels.csv"


def test_a_confident_frame_beside_clutter_is_dropped() -> None:
    # A building entering the frame: the classifier notices at index 3, and the frame before it
    # scores 0.995 -- above the threshold on its own, and it must still go.
    p = [1.0, 1.0, 0.995, 0.1, 0.0, 0.995, 1.0]
    keep = accept_sky_frames(p, threshold=0.99, neighbour_floor=0.5)
    assert keep.tolist() == [True, True, False, False, False, False, True]


def test_a_frame_below_threshold_is_dropped_but_does_not_condemn_its_neighbours() -> None:
    # 0.9 is not clutter (>= floor), merely not sure enough: it goes, its neighbours stay.
    keep = accept_sky_frames([1.0, 0.9, 1.0])
    assert keep.tolist() == [True, False, True]


def test_clip_ends_are_judged_on_their_one_neighbour() -> None:
    assert accept_sky_frames([1.0, 0.2]).tolist() == [False, False]
    assert accept_sky_frames([1.0]).tolist() == [True]
    assert accept_sky_frames([]).tolist() == []


@pytest.mark.parametrize("bad", [[1.2], [-0.1], [float("nan")]])
def test_scores_that_are_not_probabilities_are_refused(bad: list[float]) -> None:
    with pytest.raises(ValueError):
        accept_sky_frames(bad)


def test_box_conversion_is_exact_on_a_known_box() -> None:
    # 80x48 box with its corner at (216, 300) in a 640x512 frame: centre (256, 324).
    assert yolo_line([216, 300, 80, 48], 640, 512) == "0 0.400000 0.632812 0.125000 0.093750"


def test_box_running_off_the_frame_is_clipped_not_shifted() -> None:
    line = yolo_line([600, 500, 80, 48], 640, 512)
    assert line is not None
    _, cx, cy, w, h = (float(v) for v in line.split())
    assert cx + w / 2 == pytest.approx(1.0, abs=1e-6)
    assert cy + h / 2 == pytest.approx(1.0, abs=1e-6)
    assert w == pytest.approx(40 / 640, abs=1e-6)
    assert h == pytest.approx(12 / 512, abs=1e-6)


@pytest.mark.parametrize("rect", [[0, 0, 0, 0], [], [700, 10, 20, 20], [10, 10, 0.5, 20]])
def test_absent_or_vanished_target_has_no_box(rect: list[float]) -> None:
    assert yolo_line(rect, 640, 512) is None


def test_labelled_and_sampled_frames() -> None:
    assert labelled_frames(1000) == (0, 250, 500, 750, 998)
    assert sampled_frames(12, 5).tolist() == [0, 5, 10]
    with pytest.raises(ValueError):
        sampled_frames(10, 0)


def test_the_label_file_covers_every_split_with_both_classes() -> None:
    labels = load_sky_labels(LABELS)
    clips = {(lab.split, lab.clip) for lab in labels}
    # Five frames from each clip, and the publisher's 160 / 67 / 91 clips.
    assert len(labels) == 5 * len(clips)
    assert [sum(s == split for s, _ in clips) for split in SPLITS] == [160, 67, 91]
    for split in SPLITS:
        seen = {lab.label for lab in labels if lab.split == split}
        assert {"sky", "clutter"} <= seen
    # The classifier is only as good as the labels it is not told to ignore.
    unsure = np.mean([lab.label == "unsure" for lab in labels])
    assert unsure < 0.10
