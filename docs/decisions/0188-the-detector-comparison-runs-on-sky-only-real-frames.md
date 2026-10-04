# ADR 0188 — The detector comparison runs on sky-only real frames, filtered by a labelled classifier

**Status:** Accepted
**Date:** 2026-10-04

## Context

The validation phase asks the question the simulator exists for (docs/physics-model.md §15 T5):
does a detector trained on irsim renders find real drones, and do renders added to real frames
help? It trains one small detector three ways — real frames, renders, both — and scores all three
on the same real test frames.

Two facts shape it. The only real set on disk with a box on every frame and a train/val/test split
by clip is Anti-UAV RGBT (318 infrared sequences, 640×512, 8-bit display output), and about half
of its frames show tower blocks, cranes and wires behind the drone. irsim's aerial lane renders a
drone against sky and cloud and authors no terrain (ADR 0060). A detector's errors on a frame full
of buildings would be charged to a simulator that never drew a building.

The reference set of ADR 0068 (Halmstad) cannot be used: its labels are MATLAB objects with no
Python reader.

## Options considered

1. **Train and test on all frames.** No curation. The sim-to-real gap then mixes "the renders are
   wrong" with "the renders have no city in them", and the second term is large and not about the
   physics.
2. **Keep whole clips that are sky-only.** Simple, but the camera pans: buildings enter and leave
   within a clip, so clip-level selection either keeps cluttered frames or discards most of the set.
3. **Per-frame filter by a hand-tuned edge rule.** Tried: a straight-line (Hough) score and a
   sharp-edge fraction. Defocused buildings score like clean sky and textured cumulus scores like
   buildings; no threshold separates them.
4. **Per-frame filter by a small classifier trained on hand labels, biased to reject.**

## Decision

Option 4, applied to **train, val and test alike**.

- **Labels.** Five frames from each of the 318 clips (start, quarters, end) labelled `sky`,
  `clutter` or `unsure` by eye: `data/validation/anti_uav_rgbt_sky_labels.csv`, 1,590 rows. Clutter
  is anything but sky, cloud and the drone — a rooftop in a corner counts. Two rounds of
  cross-validated disagreement review corrected 40 labels, most of them small rooftops missed at
  thumbnail size.
- **Classifier.** ResNet-18 at the full 640×512 (at half resolution a corner rooftop disappears),
  in `scripts/build_anti_uav_sky.py`. It is a curation tool, not the detector under test, so the
  production model is trained on every label.
- **Rule.** Every fifth frame (4 Hz; neighbours at 20 Hz are near-duplicates) is kept when
  p(sky) ≥ 0.99 **and** neither sampled neighbour is below 0.5 (`irsim_eval.antiuav`). One-sided on
  purpose: a cluttered frame kept contaminates the measurement, a sky frame dropped costs only data.
- **Detector.** YOLO11n through `ultralytics` — the owner's choice, which settles roadmap open
  question 3 for local measurement. Every arm uses the same weights, image size, epochs, seed and
  augmentation; the checkpoint is selected on the **real** val split and the number reported is on
  the **real** test split, whatever the arm trained on (`scripts/train_detector.py`).
- **Renders.** Independent poses per frame, not clips (`ScatterTrack`), through a 50 mm example
  lens so a low boresight keeps the horizon out of frame.

## Consequences

- Measured, five folds split by clip: the classifier agrees with the labels on **99.4 %** of
  frames; at the 0.99 threshold **0 of 724** labelled clutter frames are accepted and **96.9 %** of
  sky frames are kept. Kept: 14,784 train / 6,728 val / 7,255 test frames from 213 of 318 clips.
  A random 140 kept frames, read by eye, were all sky.
- The result is a statement about **sky backgrounds only**. A detector that transfers here has not
  been shown to transfer to a drone in front of a building.
- Anti-UAV RGBT states **no licence** over its frames. The filtered set lives under the
  git-ignored `datasets/`; neither it nor a detector trained on it is published (ADR 0068). The
  label file names clips and frame indices only and is committed.
- The real frames carry things the renders do not, and they stay in: a burnt-in readout along the
  top, and a strong lens non-uniformity. They are part of the gap being measured, not removed
  from it.
- The labels are one reader's. `unsure` (7 %) is excluded from training and from the measurement.

## Revisit when

- Terrain or buildings are rendered: the filter then discards exactly the frames worth testing on.
- A real set with a stated licence and readable boxes is indexed.
