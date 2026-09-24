# ADR 0068 — Evaluation data: what the public sets are, and what may be claimed from them

**Status:** Accepted
**Date:** 2026-09-14

## Context

There is no IR camera on this project and there will not be one soon (ADR 0003), so every
comparison with reality is made against imagery somebody else published. That is workable, but only
if the *provenance* of each frame is treated as part of the measurement rather than as background
information. The same flat patch of sky supports a three-dimensional noise decomposition if it came
off a Y16 stream and supports nothing at all if it came out of an undocumented ISP through a lossy
codec; a box size converts to a range only if the pitch and focal length are known; a histogram
shape measures the recorder, not the camera, when the recorder did the 16→8-bit conversion.

The failure this ADR exists to prevent is quiet and one-directional: a number measured on a public
clip gets written into a table, the simulator is then tuned until it reproduces that number, and the
simulator has been fitted to a codec. Every decision below is a gate designed to make that harder
than doing it properly.

This ADR was opened at ME.1a (the index) and is extended by each ME step. ME.2b's codec floor is
recorded in ADR 0023's addendum because it is an extension of that estimator; the reading rules and
the Tier 4 acceptance targets live here. **The DN8 acceptance targets that replace §15's radiometric
ones are added to this ADR at ME.6**, when the comparison code that uses them exists.

## Options considered

1. **Index the sets informally in a README.** Cheapest, and what most projects do. Rejected: prose
   cannot gate an analyser, and the decision about whether a measurement is legitimate then gets
   made silently, differently, every time somebody writes a script.
2. **Machine-readable index with required provenance fields, and analysers that refuse.** A set that
   cannot state its licence, its signal path and the analysers it supports cannot be added; each
   analyser checks the property it depends on before producing a number.
3. **Take the data at face value and validate against it directly.** Rejected outright: three of the
   six indexed sets have an undocumented sensor, so there is nothing to convert a measured contrast
   or box size into.

## Decision

Option 2, in three layers.

**1. The set must say what it is (ME.1a).** `data/validation/datasets.yaml` + `irsim_eval.manifest`
make `licence`, `signal_path` and `analysers` required fields, with `excluded_analysers` beating
`analysers`. `licence: unstated` is a recorded fact, not a default — checked against each
publisher's own page on 2026-09-13, only the Halmstad set (CC0-1.0, Boson 320×256, Y16 → 8-bit →
mp4) states one, so it is the single `primary` and the other five are `unstated`.
`scripts/fetch_validation_data.py` refuses an `unstated` set unless explicitly asked, never
auto-downloads a `manual` one, and always hashes what is on disk.

**2. One canonical form, and 8 bits enforced (ME.1b).** Every converter writes
`irsim_eval.data.Sequence`; no analyser reads a publisher's layout. The reader refuses anything but
8-bit, because every indexed set *is* 8-bit and silently widening to float would invite a claim the
data cannot support.

**3. The analyser refuses what its data cannot support.** Three gates so far:

- **Clip length.** An FFC *interval* may only be reported from a clip of at least
  `MIN_INTERVAL_CLIP_S = 180 s` — the Boson's own schedule. Halmstad's clips are 10 s, so they can
  show a freeze *length* and can never show an interval; `freeze_intervals_s` raises rather than
  returning the spacing of two events that happened to be close. A real core also fires on ΔT_FPA,
  so the interval is a **distribution whose upper edge** is the configured schedule: a mean below
  180 s is the expected observation, not evidence against the schedule.
- **Frame-to-frame change.** Freeze detection compares each gap with the clip's own median gap.
  A codec that has removed the temporal noise leaves no median gap to compare against, and
  `find_freezes` raises instead of reporting the whole clip as one freeze. Measured through x264 on
  a 64×64 clip with a 42-frame freeze every 180 frames: at 4 codes of temporal noise the events
  survive CRF 18 **exactly** (start frame and length); at 1 code, CRF 23 flattens the clip and the
  detector refuses, and CRF 18 is worse than refusing — it invents an event at a frame the camera
  never froze on. This is the ME.2b codec floor in a second statistic.
- **Signal path.** Histogram-shape and edge-overshoot statistics measure an ISP, so they run only
  on sets whose frames *are* the camera's display output. (The gate itself lands with those
  extractors in ME.3b; the Halmstad entry already excludes `agc_signature` and `dde_overshoot` for
  this reason.)

**A freeze is a run of still frames; an FFC is a freeze followed by a new fixed pattern.** Counting
repeated frames alone cannot tell a shutter from a dropped chunk of recording, and the two would
contribute identically to an interval distribution. `find_freezes` therefore reports
`pattern_change` beside every run: the step in the time-averaged frame across the run, in units of
what the temporal noise alone would produce. A shutter lands far above 1 (8–20 on the synthetic
clips), a stall lands at 1, and neither is discarded — a stalled recording is a reason to distrust
every temporal statistic from that clip.

**What grows between shutter events grows from zero, and the factor of two is in the variance.**
After a shutter the correction is recalibrated, so the part of the pattern the shutter owns is zero
and relaxes back toward its stationary level (§11.2, §10.3). For an OU pattern of correlation time
τ started at zero the variance recovers as `1 − e^{−2t/τ}`, so `fit_pattern_growth` fits
`σ²(t) = floor + A(1 − e^{−2t/τ})` with `floor` absorbing the white noise and any pattern the
shutter does not recalibrate. Fitting the variance with the amplitude's law returns 2τ and looks
entirely reasonable, which is why the test asserts the true τ *and* rejects 2τ. The projection is
the column and row means: §10.3's stripe noise lives there, they average white noise down by the
width of the array, and a NUC residual moves them. Scene structure lives there too, so these run on
the flat windows ME.2b's finder returns, not on a whole frame with a horizon in it.

## Consequences

Measurements from this data come with a gate that can refuse, which means some questions have no
answer from the public sets and the reports will say so rather than fill the cell. Concretely:
Halmstad can never give an FFC interval; Anti-UAV410 and CST can never give a 3-D noise ratio (the
index already excludes it); no set gives a radiometric bias, so §15's "< 2 K per class" target is
declared untestable and reported as open in every Tier 4 report rather than quietly dropped.

The gates cost real coverage and that is the point: the alternative is a table of numbers whose
provenance nobody can reconstruct six months later. Everything measured here is a **lower bound or a
shape**, never an absolute, and ME.5/ME.6 have to carry it as one.

## Revisit when

A radiometric (16-bit, documented-sensor) public set appears, or a camera arrives — either makes the
absolute targets testable and turns most of these gates off. Also revisit if a licence status
changes: the fetch script's `unstated` refusal is the only thing standing between an experiment and
an unlicensed redistribution.

## Addendum (ME.3b, 2026-09-14): the display-output extractors and what each one can say

Three more extractors, and a third gate. The pattern is the same as above: the analyser is told what
the frames are and refuses what they cannot support.

**The signal-path gate is now code.** `require_display_output` takes the set's recorder conversion
(`display` / `linear` / `min_max` / `unknown`) as a **required argument** of `agc_signature` and
`edge_overshoot`, and raises on anything but `display`. Halmstad is the case it exists for: its
frames are a recorder's 16→8-bit conversion of a Y16 stream and the rule is itself unverified, so a
histogram measured there describes the recorder. The index already excludes `agc_signature` and
`dde_overshoot` on that set; this is the second lock on the same door, and it is the one that
travels with the function rather than with the dataset entry.

**The AGC's fingerprint is `cdf_deviation`, and it is a property of the scene as much as the ISP.**
Plateau equalisation integrates a clipped histogram into a CDF, so its output is near-uniform over
the codes it occupies; a linear stretch reproduces the scene's own distribution. The Kolmogorov
distance between the output's cumulative histogram and the uniform line separates them by more than
twenty times on every sky-like frame measured through the project's own `agc_plateau`/`agc_linear`
(plateau 0.003–0.007 against linear 0.15 flat sky, 0.15 gentle gradient, 0.19 with cloud and a
target, 0.50 across a horizon). **But a scene whose histogram is already uniform cannot be read**: a
strong ramp filling the frame gives 0.032 under a *linear* stretch, because a linear stretch of a
uniform scene is equalisation as far as any histogram statistic can tell. Hence three verdicts —
`equalised` below 0.02, `stretched` above 0.05, and `indeterminate` between — rather than a boolean
that would be confidently wrong on ramped scenes. `entropy_bits` and `occupied_fraction` are
reported alongside as corroboration (equalising a discrete histogram stretches sparse regions and
leaves output codes unused: 0.79 occupied against linear's 1.00 on flat sky).

**DDE's fingerprint is a known answer, not a fixture.** A 3×3 box at the pixel beside a step reads
two thirds of the way across it, so an unsharp mask of gain g overshoots by exactly **g/3** of the
step height, one pixel wide on each side. `edge_overshoot` measures that ratio and
`implied_box3_gain` inverts it, recovering 0.3/0.6/0.9 to 1e-6 on clean frames and to 10 % on a
noisy 8-bit one. One detail is load-bearing: the ringing either side of a step is *itself* a
difference above the edge threshold, so a candidate must be the largest difference in its own
neighbourhood — without that rule the detector measures from the overshoot and puts the real step
inside its own ring window.

**A replaced pixel is the mean of its four neighbours, so its Laplacian vanishes.** §10.4 asks for
the replacement to be simulated because the smoothed footprint is what a detector sees; the same
fact makes it findable. `replaced_pixel_map` scores each pixel by its median |4x − Σ neighbours|
over the clip, normalised by the array's own median so a smooth *scene* raises numerator and
denominator alike and produces no detections. Measured against `replace_bad_pixels` on 69 interior
defects: **every one found on float frames, 93 % after rounding to 8-bit codes, no false positives
in either case**; the handful that are not identically zero are cluster rims, filled while the
inside was still invalid, and they remain far smoother than a good pixel.

**And the codec takes this one away completely.** Through x264 at CRF 12, recall of that same
footprint falls from 1.0 to **0.0 with the false-positive rate still 0**: the block transform moves a
replaced pixel off the exact mean of its neighbours while flattening everyone else's Laplacian
toward it, so nothing stands out from the median. The estimator goes silent rather than wrong, which
is the behaviour to want — but it means an empty result on a lossy set is "not measurable here", and
ME.5 must print it beside that set's codec floor rather than as a bad-pixel count of zero.

## Addendum (XD.2, 2026-09-24): the 8-bit rule was the gate, and it became the obstacle

This ADR's "revisit when" said: *a radiometric (16-bit, documented-sensor) public set appears*. Three
have — MassMIND (16-bit LWIR maritime, FLIR ADK), LTIR (the only public source made of 16-bit
*sequences*) and the FLIR ADAS pre-AGC frames (14-bit-in-16, T-linear at 0.04 K per count, so an
11.5 mK quantisation floor under a 50 mK NETD). This is that revisit, and it changes one thing while
deliberately keeping the rest.

**What was right stays right.** Decision layer 2 refused anything but `uint8` because every indexed
set was 8-bit and a widened dtype would have invited a claim the data could not support. That
reasoning was never about `uint8`; it was about a claim being made by a dtype rather than by a
person. So the refusal did not go away, it moved: `Sequence` now carries `bit_depth` and refuses a
frame that contradicts it. The declaration is the **significant** width, not the container's, because
FLIR's 14-bit frames ride in 16-bit TIFFs and the two differ by a factor of four in the floor they
put under a noise figure. A frame holding a value its declared depth cannot represent is an error,
not a wider frame.

**`signal_path` became a value instead of prose.** It was accurate prose, and unreadable by anything;
a permission list sitting beside a paragraph nobody can parse is a permission list nobody checks. The
vocabulary is four words and closed — `display`, `recorder`, `radiometric`, `unknown` — and the prose
survives as `signal_path_note`, which is where per-set detail belongs. The distinction that pays for
the whole scheme is that **`unknown` is not a weaker `display`**: an undocumented path cannot be
assumed monotone, so it cannot say whether the picture is white-hot or black-hot, and the
measurements it supports are a strict subset.

**Each measurement declares its paths, in one table, and both locks turn from it.**
`irsim.validation.signal_path` holds every measurement name the index may use against the paths it
may use them on, with the reason written beside each one. The index will not load if a set lists a
measurement its own path cannot carry, and `measure_clip` records
`<measurement>_wrong_signal_path` in `skipped` rather than quietly omitting the statistic. The
physical content is one property: the paths that carry the ISP and the paths that carry the sensor
are disjoint, and no path can ever satisfy both — an AGC has rewritten the frame's statistics, or
there is no AGC in the frame to measure.

**Nothing indexed is radiometric yet, and the index says so.** `usable_for("noise_3d_kelvin")`
returns `[]` today, which is the honest state of the shelf and is pinned by a test that XD.3, XD.4
and XD.10 will each deliberately change. The alternative — adding the sets first and the vocabulary
afterwards — is how a permission gets granted by an entry nobody re-read.

**What this does not do.** A calibrated per-pixel-temperature path (FLAME 3's Celsius) is not in the
vocabulary. It is a different container and a different set of claims, and guessing at it in advance
would put a word in this table with no data behind it; it lands with XD.5, which needs it.
