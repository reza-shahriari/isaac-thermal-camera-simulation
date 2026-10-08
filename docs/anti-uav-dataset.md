# A synthetic set for the Anti-UAV benchmark: the plan

The goal is a synthetic infrared set that a drone detector can be trained on alone and then
tested, zero-shot, on the real [Anti-UAV](https://anti-uav.github.io/) frames. This page lists
the steps that close the remaining gap between our renders and those frames. It is written so
that anyone can pick up the next step without the conversation that produced it.

Each step has a roadmap row (`docs/roadmap.md`, lane `EV`). The roadmap row is the record of
what was done and measured. This page is the plan and the reasons.

## Ground rules

- **Measure the real set first, then change the generator.** Each change is sized from a number
  read off the real Anti-UAV frames, with the same script on both sides. Nothing is tuned by eye.
- **Match the benchmark, not a guess about cameras in general.** A real-world effect is added
  only at the rate the real set shows it. If the real set shows none, it stays out of the
  benchmark set, and can still be offered as an option.
- **Effects are drawn per clip, not always on.** A lens or sensor effect that only some real
  cameras show appears in only that share of clips.
- **No text drawn onto frames.** The real set's burnt-in readouts are masked on the real side.
- **No data generation until it is asked for.** Each step is checked with a few test frames.
- Renders run on the A6000: `CUDA_DEVICE_ORDER=PCI_BUS_ID IRSIM_GPU=0`.

## Where things stand

| Step | What | Roadmap | Status |
|---|---|---|---|
| 1a | Empty frames: the camera loses the drone | `EV.26` | done |
| 1b | Birds and other things that are not drones | `EV.27` | planned |
| 2 | Lens shading only in the share of clips that have it, at the real depth | `EV.23` part 2 | planned |
| 3 | Elevation 0–15°, the drone seen from the side | `EV.28` | planned |
| 4 | Clouds behind the drone, so the camera's contrast moves | `EV.29` | planned |
| 5 | Camera shake, measured from the real tracks | `EV.30` | planned |
| 6 | Drone size and contrast matched to the real set | `EV.31` | planned |
| 7 | Cold drones and thermal crossover | `EV.32` | planned |
| 8 | Shutter freezes, only if the real set has them | `EV.33` | planned |
| 9 | Labels in Anti-UAV's own format, with a track ID | `EV.34` | planned |
| 10 | The real set's video codec applied before training | `EV.35` | planned |

Already done (roadmap `EV.18`–`EV.25`):

- clips that are continuous random flights;
- focus drawn per clip;
- boxes drawn the way a person draws them;
- hours spread over the day, with colder nights;
- seven airframes that pitch and roll;
- drones framed down to 1° above the horizon.

Also already in the simulator:

- sensor noise, fixed-pattern noise and dead pixels;
- the camera's damped automatic contrast;
- per-pixel masks;
- motors that heat the arms.

The real-set baseline, the synthetic-only score and the mixed score are in `EV.19`.

## Facts about the real set that the steps rely on

Measured on the 318 clips of `datasets/Anti-UAV-RGBT` (infrared side):

- **Video:** 640 × 512 at 20 fps, MPEG-4 Part 2 (not H.264), at about 0.47–0.96 Mbit/s per
  clip (read with `ffprobe` on 40 training clips).
- **Drone absent:** 3,692 of 296,901 frames (1.24 %), in 52 of the 318 clips. They come as 97
  runs: median 14 frames, 90th percentile 70 (`scripts/absent_frames.py`).
- **The sky-only split** (`EV.17`): 1.2 % of train frames, 3.3 % of val and 2.1 % of test have
  no drone.
- **Edge width, box margin, aspect and shading:** in `EV.21`–`EV.25` and ADRs 0204–0209.

## The steps

### 1a. Empty frames: the camera loses the drone (`EV.26`)

**Why.** A detector that has only seen frames with a drone in them learns that there is always
a drone. In the real set the operator sometimes loses the drone, and those frames have no box.

**What.** In a drawn share of clips the camera mount slides off the drone and back. The mount
swings left, right or up, never down toward the horizon, until the drone has drifted out of the
frame. It holds there for a run of frames, then comes back. Run lengths are drawn from the
real set's 97 measured runs. The total share of empty frames is the real 1.24 %, and a planner
option can raise it. An empty frame gets an empty label file, which is how YOLO marks a negative,
and is flagged as lost in the run's `summary.json` rows (`lost_lock`).

**Done.** The planner reaches the real share to the frame (22 of 1,800 frames at 6 clips, 373
of 30,000 at 100). In a 24-frame check clip with runs of 4 and 2, exactly those 6 frames have no
drone pixel and an empty label, and every other frame has a box. The swing takes one frame
(the set's frames are 5.6 s apart). Run lengths are matched in frames, not seconds (ADR 0210).

### 1b. Birds and other things that are not drones (`EV.27`)

**Why.** This is the most likely way a model trained on synthetic data fails in the field: it
has never seen anything warm in the sky that is not a drone. Anti-UAV has no bird labels, so
this will barely move the benchmark score. It matters for real use.

**What.**

- Import a licence-checked bird model through the normal asset import (`ingest-asset`). It
  needs a warm body and cooler feathers, its wings must flap, and it gets its own heat model.
- Birds fly in some clips beside the drone and in some clips alone.
- They get their own class (`bird`), or no box when the set is drone-only. They are never
  labelled as a drone.
- Later: distant aircraft and helicopters.

**Done when.** A rendered bird's surface temperatures are cited to a published thermal image of
the species or its size class. Its box statistics are reported beside the drone's.

### 2. Lens shading only where the real set has it (`EV.23` part 2)

**Why.** Today every clip gets both a lens falloff and a camera warm-up drift. Many real clips
show a dark centre and bright corners, but not all of them, and our bowl is a quarter of the
real depth.

**What.**

1. Measure the shading of each real clip. The sky brightens toward the horizon from top to
   bottom, but lens shading is circular, so the two can be separated.
2. Check whether each clip's shading stays fixed through the clip (the lens) or changes over time
   (the camera warming up).
3. From each clip's measured depth, work out its lens falloff, and draw clips from that spread.
   That spread includes clips with no shading at all.
4. If the shading is fixed through each clip, take the warm-up drift out of the default draws.

**Done when.** The share of clips with a bowl and the depth distribution both match the real
ones (depth p50 inside the real IQR).

### 3. Elevation 0–15°, seen from the side (`EV.28`)

**Why.** The real camera looks at the drone almost level: up to about 15° elevation, never from
underneath.

**What.**

- The planner band becomes 0–15° (it is 1–12° now). The lowest framable elevation, about 0.3°
  with the horizon held just below the frame, still applies.
- The tilt cap comes down so that a pitched drone does not show its belly. It is 25° today.

**Done when.** The rendered elevation spread and the box aspect are re-measured against the real
set.

### 4. Clouds behind the drone (`EV.29`)

**Why.** A real camera's automatic contrast shifts when the drone crosses from cold sky to cloud.
A clear-sky clip never makes it shift. The clouds are now finished in both bands (WX lane).

**What.** Cloudy clips join the set, with the drone passing in front of cloud edges. Their share
is measured from the sky-only split. A cloudy frame costs about 6× a clear one (63 s against
11 s).

**Done when.** The frame-to-frame change in the display's mean and spread matches the real
clips' distribution.

### 5. Camera shake, measured first (`EV.30`)

**Why.** The real footage comes from an operator or a pan-tilt mount. The drone jumps around the
frame from frame to frame, and fast pans blur it. Our mount follows smoothly.

**What.** Measure the real boxes' frame-to-frame motion (its spectrum and its size relative to the
box). Then add high-frequency shake, and the motion blur that goes with it, sized to that
measurement.

**Done when.** The rendered frame-to-frame box motion matches the real distribution.

### 6. Drone size and contrast matched to the real set (`EV.31`)

**Why.** Range is drawn uniformly within a band, so the box areas only roughly match the real
ones.

**What.** Measure the real distribution of box area and of drone-to-sky contrast (on the display,
as the detector sees it). Then draw each clip's range so that box areas reproduce the real
distribution.

**Done when.** Box area quantiles (p10/p50/p90) and the contrast median are within the real IQR.

### 7. Cold drones and thermal crossover (`EV.32`)

**Why.** The solved Phantom 4 always starts nine minutes into its flight, warm. A drone just
after takeoff can sit at the sky's own temperature and nearly vanish.

**What.** Draw each clip's start time in the mission from takeoff on, including a share of clips
that start cold.

**Done when.** The drone-to-sky contrast at takeoff is reported per weather case, and a share of
clips includes near-zero contrast.

### 8. Shutter freezes, only if the real set has them (`EV.33`)

**Why.** A real thermal camera sometimes freezes for a moment while it recalibrates. The
simulator already has this (`irsim.isp.ffc`), but it is off in the dataset.

**What.** Look for frozen frames in the real videos: a run of frames identical to the one before,
then a jump. Turn the freeze on in the dataset only at the rate found. If the real set shows
none, it stays off for the benchmark set.

**Done when.** The real rate is recorded, and the set's rate matches it.

### 9. Labels in Anti-UAV's own format (`EV.34`)

**What.** For each clip, write the file Anti-UAV itself uses: `exist` and `gt_rect` per frame. Add
a track ID per object, for multi-object tracking. YOLO labels and masks already exist.

**Done when.** The benchmark's own evaluation code reads a synthetic clip and scores the truth
as 1.0.

### 10. The real set's codec, before training (`EV.35`)

**Why.** Our renders are saved as lossless PNG; the real frames are compressed video. For a drone
30 pixels across, the codec's blocking and ringing are a large part of what it looks like. A
detector trained on clean frames meets them for the first time at test time. Expect a visible
gain in mAP50-95.

**What.**

1. The real videos are **MPEG-4 Part 2 at 0.47–0.96 Mbit/s, 20 fps**, not H.264. So the
   encoder and its settings must match that, not an H.264 CRF.
2. Measure the real frames' block artefacts: the step across 8 × 8 block borders compared with
   the step inside the blocks, plus the ringing around the drone.
3. Pick the encoder setting that reproduces that number on our clips.
4. Encode each synthetic clip with one `ffmpeg` call, decode it, and train on the decoded frames.
   The labels do not change. The lossless frames stay as the archive.

**Done when.** The blockiness measure on the decoded synthetic frames is inside the real IQR,
and a training run with and without the codec pass is reported (`EV.19`'s protocol).

## How to pick this up

- The roadmap row for each step says what was measured and what is open.
- `scripts/generate_aerial_dataset.py --dry-run` prints a set's plan (airframe, hour, focus,
  camera, empty-frame runs) without rendering.
- Measurement scripts, all run the same way on real and synthetic frames:
  - `scripts/target_sharpness.py`: edge width;
  - `scripts/box_convention.py`: drawn box against visible extent;
  - `scripts/sky_bowl.py`: shading;
  - `scripts/absent_frames.py`: empty frames.
