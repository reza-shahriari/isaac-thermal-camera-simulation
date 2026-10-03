# ADR 0077 — Within-frame motion smear, spatially varying

**Status:** Accepted
**Date:** 2026-09-14

## Context

`irsim.optics.mtf.mtf_motion` — `|sinc(v t_int ξ)|` for linear image-plane motion during the
integration — has been in the MTF cascade since M5. **Nothing ever called it.** A grep for its name
outside its own definition and `__all__` returned the export list and nothing else, so every frame
this simulator has produced was sharp regardless of how fast the scene crossed it.

That is not a small omission for the scenes this project is built for. The aircraft stage (ADR
0075) sweeps the boresight at 34 °/s at closest approach and a Boson pixel is 0.049°, so the scene
crosses **eleven pixels in one frame period**. Motion blur is also one of the most obvious
signatures separating real thermal video of a moving target from synthetic video of one, which
puts it squarely in the sim-to-real path this whole lane exists to serve.

It is distinct from, and additional to, the bolometer's frame-to-frame lag. The membrane IIR (§9.2)
carries a target's history across successive frames; this is the blur laid down **within** a single
integration. Both are real and they compose.

## Options considered

**1. Frequency domain, folded into the PSF.** The cascade is already built that way, and it is what
`mtf_motion` is written for. It requires *one* velocity for the whole frame. That is exactly wrong
for the scenes here: under a tracking mount the target is stationary on the focal plane while the
sky sweeps past it, so a single kernel serves neither. Kept for the MTF bench, which measures a
uniform-motion figure of merit; not usable for a render.

**2. A spatially varying line average** (chosen). Each pixel is averaged along **its own** motion
vector, from −s/2 to +s/2 with s = |v|·duty. Costs one gather per tap and handles the tracked case
correctly by construction.

**3. Where it sits.** After the PSF, before the box filter. Both are convolutions laid down during
the integration and they commute, so the order between *them* is arbitrary; that both precede the
box filter is not, because the box filter is the detector sampling the result.

**4. Centred or trailing.** Centred. A trailing segment displaces every moving feature by half its
smear — a shift that looks like a timing error because it is one.

**5. The duty, which is where the two detector families part.** A microbolometer has no shutter and
no integration window — `integration_time_ms` is `None` for one in the schema, deliberately —
because it integrates continuously, so the scene smears over the **whole frame period**. A cooled
photon detector integrates briefly inside the frame and is idle for the rest, so it smears over
that fraction and comes out sharper. Reading a missing integration time as zero would have made
every uncooled camera in the repository sharper than it is, which is the flattering direction and
therefore the dangerous one. This is the mechanism behind §16's checklist line *"lateral motion
smears LWIR, not cooled MWIR"*.

## Decision

`irsim.optics.smear.apply_motion_smear` is the spatially varying operator; `smear_duty` is the
integration fraction, with `None` meaning a bolometer and therefore 1. `irsim.pipeline.optics`
scales the G-buffer's `motion_px` by the duty and hands it to `apply_optics`, which applies it
between the PSF and the box filter. `motion_px` stays optional in the G-buffer (M0.6), so a still
scene takes the previous path exactly and every committed golden is unchanged.

## Consequences

**It agrees with the cascade term it implements**, which is the only reason to trust it: measured
against `|sinc(s f)|` over smears of 3–20 px and frequencies of 1/48–1/12 cyc/px, worst deviation
**0.015**, mostly under 0.006. Two descriptions of one effect that disagreed would be worse than
one, because a picture cannot tell you which is wrong.

**The cross-check found a real error.** Taps placed at the segment's *endpoints* look natural and
are wrong: N taps spanning length s sit s/(N−1) apart, so the comb implements a boxcar of length
s + s/(N−1). Measured MTF 0.7182 where sinc said 0.7842 — exactly the Dirichlet kernel of the
longer smear. The operator was self-consistent and describing the wrong smear, which no amount of
"does it look blurred" would have caught. The taps are now at sub-interval midpoints.

**Cost.** One bilinear gather per tap, with one tap per pixel of travel up to 65. On a 4×
supersampled Boson frame with an 11 px native smear that is ~45 gathers of 5 M points. Scenes
without a `motion_px` plane pay nothing, and motion below a quarter pixel returns the input
untouched.

**What it does not model.** Rotation within the integration is approximated as translation at each
pixel's own velocity — correct to first order, and the error grows with the rotation rate times
the integration time. Occlusion is ignored: a smeared foreground edge blends with whatever the
background *currently* is rather than with what was actually behind it during the sweep, which is
the standard approximation for post-hoc motion blur and is visible only at high-contrast
silhouettes. There is no sub-frame *scene* motion — the target moves, the world does not deform.

## Revisit when

* A Warp port of stage 3 lands — this is a gather-heavy operator and is the natural candidate.
* Rotation rates get high enough that the per-pixel-translation approximation shows, most likely
  on a spinning propeller rather than a slewing mount.
* Propellers are modelled: a blade at flight rpm sweeps its whole disc within one integration, so
  the annulus it smears into is this operator's job and the reason ADR 0074 left props out.

## Amendment, 2026-10-03 — a bolometer smears with its membrane, not a box (EV.16)

**Option 5 was wrong for the bolometer, and the spec said so.** §9.2 states that a bolometer's motion
smear *is* its τ_th response, and `optics/mtf.py` and `optics/psf.py` both say `mtf_motion`'s box is
for photon detectors only. This ADR instead gave the bolometer a centred box over the whole frame
period. The membrane is a first-order system: the reading at the frame's instant weights the flux of
`s` seconds earlier by `e^(−s/τ)/τ`. The frame-to-frame IIR (§9.2, `detector/lowpass.py`) already
carries the part older than one frame period exactly; the part *within* the frame carries the same
exponential, truncated to `[0, T]`. The box had the right total and the wrong shape. It spread the
Boson's smear evenly over 16.7 ms where the 8 ms membrane puts half its weight in the most recent
4.6 ms.

Found on the first EV.16 render: a quadrotor crossing the frame edge at 10 px/frame drew a flat
10 px streak.

**Decision.** `apply_motion_smear(..., decay_frames=τ/T)` lays the bolometer's kernel down
**trailing**, not centred. Tap weights are the exponential's exact mass over each slice of the frame,
at the slice midpoints. `smear_decay_frames` gives `τ/T` for a detector with no integration time (0.48
for the Boson at 60 Hz) and `None` for a photon FPA, which keeps the centred box over its shutter
window. `apply_optics` takes `motion_decay_frames`; `optics_stage` and `run_frame` pass
`irsim.pipeline.optics.motion_decay(sensor)`.

**Trailing is not a timing error here.** Option 4's argument holds for a shutter: the window's
midpoint is the frame's instant. A bolometer genuinely lags, by its own time constant, so the head
sits at the feature's current position and the tail lies behind it.

**Measured** (`tests/unit/test_motion_smear.py`, `test_run_frame_motion_smear.py`,
`test_motion_wired.py`), on a line moving 11 px/frame:

| | box (before) | membrane (after) |
|---|---|---|
| energy within d px behind the head | `(d+½)/11` | `(1 − e^(−(d+½)/(11·0.48))) / (1 − e^(−1/0.48))`, within 0.05 |
| half-energy extent | ≥ 5 px | ≤ 3 px |
| brightest pixel | 0.09 | > 1.8 × the box's |
| 10–90 edge width | 8.8 px | 7.7 px |

`run_frame` on the Boson now equals `apply_optics` with the membrane kernel to 1e-6 and differs from
the box render.

**Not changed.** The visible companion's sub-frame exposure (`optics/exposure.py`, IG.19) still spans
the whole frame period for a bolometer. That is now a box beside an exponential, which is the
same total exposure in a different shape. No driver that films a bolometer requests sub-frames today;
revisit when one does.
