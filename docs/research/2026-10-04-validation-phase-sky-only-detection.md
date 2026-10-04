# The validation phase: does a detector trained on irsim find real drones? — first report, 2026-10-04

**One small detector is trained three ways — on real infrared frames, on irsim renders, and on both
— and all three are scored on the same real frames. This is the first day of that experiment: what
was built, what was measured, and what comes next.**

Produced for the owner's request of 2026-10-04: *"i want to have a validation phase … remove frames
something else exist, just sky and uav should be there … train a model (say yolo11n) … create a
dataset from uav in sky … train on simulated images, compare these two, and final phase, add the
created images to real images and have another validation in the end."* The decisions are recorded
in [ADR 0188](../decisions/0188-the-detector-comparison-runs-on-sky-only-real-frames.md); the
roadmap rows are `EV.17` (done), `EV.18` (done) and `EV.19` (open).

This is a **snapshot of an unfinished experiment**. The real side is complete. The synthetic side
has a 300-frame pilot and a first clear-sky clip set of 1,800 frames, and all three arms have been
run once on each. Every result is one seed, one airframe and clear sky only, so the numbers are a
first measurement, not a verdict.

---

## 1. The short version

1. **The real set had to be cleaned before it could judge the simulator.** Anti-UAV RGBT's infrared
   frames are about half city. irsim renders a drone against sky. A classifier trained on 1,590
   hand-labelled frames now removes every frame with a building, crane or wire from train, val and
   test alike, leaving **14,784 / 6,728 / 7,255** frames. Cross-validated by clip, it accepted
   **0 of 724** labelled clutter frames and kept **96.9 %** of the sky frames.
2. **Real → real is nearly saturated.** YOLO11n trained on the sky-only train split scores
   **mAP50 0.991, mAP50-95 0.568** on the sky-only test split. Finding a drone against sky is easy;
   boxing it tightly is not, so mAP50-95 is the number that separates the arms.
3. **Renders alone find under half the real drones; renders added to real frames score slightly
   higher than real frames alone.** Trained only on 1,800 rendered frames: **mAP50 0.494, mAP50-95
   0.211** (0.379 / 0.113 with the 300-frame pilot). Real plus the 1,800 renders: **0.991 /
   0.583**, against real-only 0.991 / 0.568. That +0.015 is one seed and has no spread beside it.
4. **The generator now films clips, not unrelated frames.** A continuous random flight
   (`--track wander`) varies range, elevation, bearing, heading and the aircraft's place in the
   frame while the mission clock runs, so the clip can be watched and the airframe is seen warming
   up. Each frame gets a box from the renderer's own truth plane.
5. **Building it surfaced a real camera-model interaction.** Independent poses drew a 2 K streak
   across the sky: the camera was smearing the jump between two unrelated poses as if the aircraft
   had flown it during the exposure. Fixed, and measured gone.

---

## 2. Why sky-only

The comparison only measures the simulator if the real frames contain what the simulator renders.
irsim's aerial lane renders a drone against sky and cloud and authors no terrain (ADR 0060). If the
real test set is half tower blocks, a detector's mistakes on those frames are charged to a
simulator that never drew a tower block, and the sim-to-real gap becomes mostly a statement about
scene content rather than about physics.

So the filter is applied to **all three splits**, not only to the test split. The result is a
statement about sky backgrounds and nothing else: a detector that transfers here has not been shown
to transfer to a drone in front of a building.

### 2.1 The real set

| | |
|---|---|
| Set | Anti-UAV RGBT (Jiang et al. 2021), infrared stream |
| Clips | 318, the publisher's split by clip: 160 train / 67 val / 91 test |
| Frames | 296,901 infrared frames, 293,209 with a target |
| Format | 640×512, 8-bit white-hot display output, MPEG-4, 20 Hz in the container |
| Labels | one box per frame, `[x, y, w, h]`, with an `exist` flag |
| Sensor and lens | undocumented |
| Licence | **none stated over the frames** |

Things the frames carry that matter later:

- **A burnt-in readout** along the top of every frame: the mount's azimuth and elevation and a
  timestamp. The elevation readout is the only viewing geometry the set carries; by eye it sits
  mostly between 0.2° and 10°.
- **A strong lens non-uniformity** — a dark centre and bright corners, different from clip to clip.
- **A panning camera.** Buildings enter and leave within a clip, so "sky-only" is a property of a
  frame, not of a clip.
- **Both day and night**, and both clear sky and heavy cumulus.

The reference set of ADR 0068 (Halmstad) could not be used: its labels are MATLAB objects with no
Python reader.

### 2.2 Two things that did not work

**A hand-tuned edge rule.** The first attempt scored each frame by the fraction of sharp-gradient
pixels and by the total length of straight line segments (Canny + Hough), outside the drone's box
and the readout. It does not separate the classes: defocused buildings at night score like clean
sky, textured cumulus scores like buildings, and a fixed artefact gave about fifty pixels of "line"
to perfectly clean frames.

**Clip-level selection.** Keeping whole clips that look sky-only at their middle frame either keeps
cluttered frames (the camera pans onto a rooftop) or throws most of the set away.

### 2.3 What did: hand labels and a small classifier

**Labels.** Five frames from each clip — the first, the quarters and the last — were read by eye
and labelled `sky`, `clutter` or `unsure`: 1,590 rows in
`data/validation/anti_uav_rgbt_sky_labels.csv`. Clutter is anything but sky, cloud and the drone; a
rooftop in a bottom corner counts, and so does a single wire.

| Label | Frames |
|---|---|
| sky | 747 |
| clutter | 724 |
| unsure | 119 (7 %, excluded from training and from the measurement) |

**The labels were wrong before the classifier was.** They were first made from 128-pixel
thumbnails, and two rounds of reviewing cross-validated disagreements at full size corrected 40 of
them. In most disagreements the classifier was right: a small rooftop along the bottom edge that
was invisible at thumbnail size.

**Classifier.** A ResNet-18 with one output, ImageNet-pretrained, trained for eight epochs on each
labelled frame plus four temporal neighbours (±0.2 s and ±0.4 s, where the scene has not changed
but the noise and the drone's position have), with horizontal flips and gain/offset jitter.

Resolution mattered more than anything else:

| Input | Agreement with labels (held-out clips) | Clutter accepted at p ≥ 0.99 |
|---|---|---|
| 320×256 | 97.9 % | 1 |
| 640×512, labels corrected | 99.0 % | 0 |
| 640×512, the committed script and seed | **99.4 %** | **0 of 724** |

At half resolution a corner rooftop is a few pixels and disappears.

**The keep rule is one-sided on purpose.** Every fifth frame is scored (4 Hz; neighbours at 20 Hz
are near-duplicates). A frame is kept when its own p(sky) ≥ 0.99 **and** neither sampled neighbour
is below 0.5. The second clause covers what a per-frame classifier cannot: a building enters over
several frames, and the frame on which the classifier first notices is later than the frame on
which the building first appears. A cluttered frame kept contaminates the measurement; a sky frame
dropped only costs data.

**Result.**

| Split | Clips | Clips with kept frames | Frames sampled | Frames kept | Kept with a drone |
|---|---|---|---|---|---|
| train | 160 | 107 | 29,924 | 14,784 | 14,611 |
| val | 67 | 44 | 12,407 | 6,728 | 6,507 |
| test | 91 | 62 | 17,082 | 7,255 | 7,100 |

A random 140 kept frames (70 train, 70 test) were read by eye afterwards: all sky, cloud and drone.

The classifier is a curation tool, not the detector under test, so the production model is trained
on every label. It can be run on anything:

```bash
python scripts/build_anti_uav_sky.py cv                      # the cross-validation above
python scripts/build_anti_uav_sky.py build                   # train, filter, export YOLO layout
python scripts/build_anti_uav_sky.py score frame.png clip.mp4   # p(sky) from the saved weights
```

---

## 3. The detector and the protocol

YOLO11n through `ultralytics`, from COCO-pretrained weights — the owner's choice of detector for
this phase. Every arm uses the same weights, image size (640), epochs (30), batch (64), seed and
default augmentation. **Whatever an arm trains on, its checkpoint is selected on the real val split
and the number reported is on the real test split** (`scripts/train_detector.py`). The arms differ
in their training images and in nothing else.

| Arm | Trains on | Question it answers |
|---|---|---|
| real → real | sky-only real train | what this detector and this much real data are worth at all |
| synthetic → real | irsim renders | the sim-to-real gap proper |
| mixed → real | both | whether the renders *help*, which the gap does not imply |

One consequence worth stating: selecting the synthetic arm's checkpoint on real val frames is
slightly generous to it. It is the same rule for every arm, and it is the rule a user with a small
real validation set would follow.

---

## 4. The synthetic side

### 4.1 The camera

The renders go through `configs/sensors/example_lwir_640_telephoto.yaml`: the project's Boson 640
core (12 µm, 640×512, F/1.0) behind a **50 mm** lens — 8.8° × 7.0°, 0.24 mrad per pixel. It is
**not** a model of the Anti-UAV camera, which is undocumented; it is the same class of instrument,
a long lens on a ground station. A long lens is also what lets the boresight sit low: this scene
has no terrain, so a frame that reaches the horizon would show analytic ground, and the driver
refuses to render one.

### 4.2 The aircraft

A DJI Phantom 4 Pro, imported from a third-party model (`--asset phantom4`), with its temperature
solved per cell on its own mesh (234,923 cells): the sun reaches each cell through its own beam,
the motors heat with throttle, and the mission clock runs through a 28-minute flight. The camera's
full chain is on — optics, detector noise, fixed pattern, bad pixels, flat field, and the camera's
own AGC producing the 8-bit white-hot display frame, which is what the public sets are.

### 4.3 Independent poses, and what they broke

The first generator (`--track scatter`) draws every frame's pose independently:

| Drawn per frame | Distribution | Why |
|---|---|---|
| Slant range | log-uniform | apparent size goes as 1/R; a uniform draw is mostly small targets |
| Elevation | uniform in a band | the lower edge is held above the horizon by the driver |
| Bearing | uniform, full circle | changes the sun's aspect on the airframe |
| Heading | uniform, full circle, independent of bearing | every aspect from head-on to tail-on |
| Place in frame | uniform within ±60 % of the half field | no real mount holds a target dead centre |

The published ablation this follows is the largest in the drone sim-to-real literature: mAP50 of
0.464 with a fixed camera pitch against 0.981 with a random one (roadmap `EV.10`).

**The streak.** The first six frames came back with a thin bright line across the sky in two of
them — about 2 K above the background, one to two pixels wide, in the radiance plane but not in the
true-temperature plane, and not on any labelled geometry. Re-rendering the same poses through the
wide 14 mm lens showed straight lines at a different angle in each frame. The cause was the
camera model working as designed: it synthesises exposure smear from the change in pose between
one capture and the next, and here the "motion" between captures was a jump to an unrelated
aircraft position under an unrelated boresight. The fix is `IrCamera.restart_motion()`, which
declares the next capture the first of a new sequence; with it, the count of stray bright pixels
away from the aircraft was zero in all six frames.

### 4.4 Continuous random flight

Independent frames give a detector variety and give a person nothing to watch. On the owner's
direction the default became a **clip**: `--track wander` is one continuous, seeded, non-repeating
flight. Log-range, elevation, bearing, heading and the mount's two pointing offsets are each a
constant plus three sinusoids with seeded frequencies and phases, so the aircraft closes and
recedes, climbs and sinks, circles the observer and turns through every aspect, smoothly. The clip
spans the mission, so the airframe is seen cold, warming and hot, and the exposure smear is live
again because the motion is real.

A 24-frame test clip measured: range 30–82 m, 31–85 px across, elevation 10–32°, aspect from
−159° to +174°, and the motors rising from 23.0 °C to 28.3 °C over the 1,666 s it spans.

### 4.5 Labels

Each frame's box is the extent of the aircraft's own pixels in the renderer's truth `part_id`
plane, written by `irsim.io.labels` (roadmap `EV.15`, now wired into a driver for the first time).
A target cut by the frame edge is boxed as far as it is in the picture; one that has left gets no
box. The box does **not** include the rotor discs, which the camera model composites separately.

### 4.6 Cost

Measured on the RTX A6000:

| Condition | Seconds per frame |
|---|---|
| Clear sky, GPU and CPU otherwise idle | 7.5–11 |
| Clear sky, sharing the machine with other jobs | 20–27 |
| Broken cumulus (cloud deck marched per pixel) | 63 |

Clouds render and move from frame to frame, but through a long lens they look soft and blobby
beside real cumulus, and they cost six times as much. On the owner's direction the set is
**clear sky only** until the cloud work is finished.

---

## 5. Results so far

All on the same 7,255 real sky-only test frames, YOLO11n, 30 epochs, one seed.

| Trained on | Training frames | mAP50 | mAP50-95 | Precision | Recall |
|---|---|---|---|---|---|
| Real sky-only | 14,784 | 0.991 | 0.568 | 0.989 | 0.987 |
| Synthetic pilot only | 300 | 0.379 | 0.113 | 0.413 | 0.454 |
| Real + synthetic pilot | 15,084 | 0.991 | 0.565 | 0.986 | 0.981 |
| **Synthetic clip set only** | 1,800 | **0.494** | **0.211** | 0.825 | 0.429 |
| **Real + synthetic clip set** | 16,584 | **0.991** | **0.583** | 0.985 | 0.979 |

**The pilot** is 300 independent-pose clear-sky frames (six runs of 50 at six hours of the day),
16–130 m, elevation 6–35°.

**The clip set** (`datasets/irsim_sky_v1`) is six continuous random-flight clips of 300 frames,
each spanning a 1,666 s mission:

| | |
|---|---|
| Slant range | 20–85 m |
| Apparent size | 30–128 px across |
| Elevation | 6–20° |
| Motors | 23 °C at take-off, about 41.5 °C at mid-mission, back to 22–23 °C after landing |
| Hours (UTC, drawn from the seed) | 09:45, 02:06, 22:12, 22:23, 22:49, 22:57 |
| Render cost | 7–13 s per frame |

**What can be said.**

- Renders alone transfer, partly. A detector that has never seen a real frame finds 43 % of the
  real drones, and when it fires it is usually right (precision 0.83). Six times more renders, as
  clips at a lower elevation, moved mAP50-95 from 0.113 to 0.211.
- Adding the clip set to the real frames did not hurt and scored 0.015 higher in mAP50-95
  (0.583 against 0.568) with mAP50 unchanged.

**What cannot be said yet.**

- That the renders *help*. One seed per arm; +0.015 could be seed noise. The 300-frame mix moved
  it by −0.003, which gives a feel for the size of that noise but is not a measurement of it.
- That the synthetic-only gap is 0.357 mAP50-95. The set is nine times smaller than the real one,
  one airframe, one sky condition.

**Two things the clip set got wrong, found when reading its own logs.**

- **Five of the six clips are at night.** The hours are drawn uniformly from the seed and this
  draw landed 22:12–22:57 four times and 02:06 once. Only one clip has the sun on the airframe.
- **The air is 22.0 °C at 09:45 and at 22:23.** The synthesised weather gives day and night the
  same air temperature, so the night clips differ from the day clip by the sun alone. A real
  night is cooler, and the sky behind the drone with it.

---

## 6. Where the gap probably comes from

None of these has been tested. They are the differences visible so far, in the order they would be
cheapest to check.

1. **Viewing geometry.** Real boxes are wide and flat — the drone seen nearly edge-on from a low
   elevation — with width/height reaching 3.3 at the 95th percentile. The pilot's reach 1.9,
   because its elevations start at 6° and run to 35°.

   | | Width px (5 / 50 / 95 %) | Height px (5 / 50 / 95 %) | Width ÷ height (5 / 50 / 95 %) |
   |---|---|---|---|
   | Real sky-only test | 31 / 50 / 93 | 21 / 30 / 51 | 1.12 / 1.53 / 3.33 |
   | Synthetic pilot | 15 / 40 / 105 | 9 / 26 / 69 | 1.29 / 1.55 / 1.92 |
   | Synthetic clip set | 24 / 41 / 73 | 14 / 24 / 42 | 1.52 / 1.74 / 2.00 |

   The clip set narrowed the band to 20–85 m and 6–20°, which removed the too-small targets but
   still stops at a ratio of 2.0. The real set's flattest views need elevations below 6°, which
   this lens cannot reach without the horizon entering the frame.
2. **Box convention.** The rendered box is tight on the airframe without its rotor discs; the real
   boxes are a human's and are looser. That costs mAP50-95 directly and mAP50 much less.
3. **Things in real frames that are not rendered**: the burnt-in readout, and the lens
   non-uniformity that makes a real clear sky a dark disc with bright corners where the render's is
   a smooth vertical gradient. The project already models the mechanism — the housing bowl an
   uncooled core shows between shutter events (`scripts/validate_sky_flat.py`) — but these renders
   are taken right after a flat field, where it is zero.
4. **One airframe.** The real set has several drone types; the renders have a Phantom 4 that yaws
   but never pitches or rolls.
5. **Target appearance.** Real drones are often a saturated blob with soft edges; the rendered one
   is sharper and shows more internal structure. Whether that is focus, the real camera's ISP or
   the render's contrast is exactly the kind of question the project's image statistics (roadmap
   `EV.6`) exist to answer.

---

## 7. What is done and what is next

- **Done:** the clip set and all three arms on it (`EV.19`), section 5. Nothing is running.
- **Next, by cost:** a second and third seed for every arm, so the mixed arm's +0.015 has a spread
  beside it; hours spread across the day instead of drawn; a night that is cooler than the day; a
  longer lens for elevations below 6°; rotor discs in the box; more airframes; and a mixed-ratio
  sweep (roadmap `EV.11`) instead of one mix.
- **Not planned for this phase:** clouds, until the cloud lane is done; buildings, which the
  simulator does not render.

---

## 8. Limits of this report

- **One seed for every number.** No arm has a spread.
- **One reader's labels.** The sky/clutter labels are one pass by one reader, corrected against a
  classifier; 7 % are `unsure`.
- **No licence on the real frames.** Anti-UAV RGBT states none. The filtered set, the classifier
  weights and every detector trained on them stay on the machine that made them; no real frame is
  reproduced here or on the project site. The label file names clips and frame indices only.
- **The real camera is unknown.** The 50 mm lens is a guess at its class, not a measurement.
- **Sky only.** Nothing here is evidence about drones in front of clutter.

---

## 9. Where everything is

| What | Where |
|---|---|
| The decision record | `docs/decisions/0188-the-detector-comparison-runs-on-sky-only-real-frames.md` |
| Hand labels | `data/validation/anti_uav_rgbt_sky_labels.csv` |
| Keep rule and box conversion | `src/irsim_eval/antiuav.py`, `tests/unit/test_antiuav_sky.py` |
| Classifier: train, cross-validate, build, score | `scripts/build_anti_uav_sky.py` |
| Classifier weights (local only) | `datasets/anti_uav_sky/sky_classifier.pt` |
| Sky-only real set (local only) | `datasets/anti_uav_sky/` |
| Detector training, three protocols | `scripts/train_detector.py` |
| Flight tracks | `src/irsim_isaac/asset_flight.py` (`WanderTrack`, `ScatterTrack`) |
| Render driver | `scripts/render_phantom4.py --track wander` |
| Dataset generator | `scripts/generate_aerial_dataset.py` |
| Long-lens camera | `configs/sensors/example_lwir_640_telephoto.yaml` |
| Runs and reports (local) | `outputs/validation_phase/*/report.json` |
