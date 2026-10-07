# ADR 0207 — A drone set's label is the box a person draws, measured, not the rotor discs

**Status:** Accepted (EV.22); amended by EV.10
**Date:** 2026-10-07

## Context

A detector trained on renders is scored on a real set's boxes (`EV.17`–`EV.19`). Those boxes
were drawn by people. The rendered boxes were the truth mask's extent (`EV.15`). The EV.19 report
blamed the difference on the rotor discs: the rendered box was tight on the airframe without the
discs its propellers sweep. That was a guess. `scripts/box_convention.py` measures how each box sits on what the frame
shows: the box over the extent of the pixels past half the target's contrast
(`irsim.io.labels.visible_extent`), per axis.

| Boxes ≥ 16 px wide | Box ÷ visible, width p10 / p50 / p90 | Height | Aspect p10 / p50 / p90 |
|---|---|---|---|
| Real, sky-only test (799) | 1.11 / 1.29 / 1.47 | 1.25 / 1.48 / 1.79 | 1.21 / 1.52 / 3.14 |
| irsim clip set (800) | 0.95 / 0.98 / 1.00 | 0.93 / 0.97 / 1.00 | 1.56 / 1.74 / 1.90 |

The real boxes carry a person's margin, larger in height than in width. The rendered ones carry
none. Whether a spinning disc is part of what a person boxes is already inside that margin: a
person boxes what they see.

## Options considered

1. **Project the rotor discs into the box.** This is physics, but it adds width, not height, at
   the low elevations these sets are filmed at. That is the wrong direction for the aspect, which
   was too wide.
2. **Draw the box as a person does:** what is visible, plus a margin drawn from the real set's
   measured distribution.

## Decision

Option 2.

- `irsim.io.labels.as_drawn(labels, display, rng)` replaces each box with `visible_extent` on the
  display frame. A defocus glow counts as far as it shows; a part too faint to see does not; a
  target with no contrast keeps its truth box.
- The box grows about its centre by a log-normal margin per axis, then is clipped to the frame.
  `DRAWN_MARGIN`: width median 1.28, σ_log 0.103; height median 1.48, σ_log 0.135. Both are fitted
  to the real 10–90 % ranges.
- Mask, pixel count, visibility and cloud transmittance stay the truth's.
- `render_phantom4.py` writes `labels_drawn/` beside `labels/` for every scatter or wander frame.
  `generate_aerial_dataset.py --boxes drawn` (the default) collects the drawn ones.

## Consequences

The old clip set re-labelled this way (`--as-drawn`, the same frames):

| | Box ÷ visible, width / height p50 | Aspect p10 / p50 / p90 | Width / height p50, px |
|---|---|---|---|
| As drawn | 1.28 / 1.50 | 1.17 / **1.47** / 1.85 | 54 / 37 |
| Real | 1.29 / 1.48 | 1.21 / 1.52 / 3.14 | 51 / 30 |

- The margins match by construction. The aspect is the independent check: it was not fitted, and
  its median moved from 1.74 to 1.47 against the real 1.52.
- The real set's flattest boxes (aspect p90 3.1) are drone views from below 6°. The 50 mm frame
  cannot reach those with the horizon out of shot; that is `EV.10`.
- The truth boxes stay in `labels/`. A set can be scored both ways, and mAP50-95 should be.
- The margin model is the one real set's. Another set's annotators would draw another.

## Revisit when

- A second real set's boxes are read (`XD.11`). Measure its margin before reusing this one.

## Amendment (EV.10)

At 1–3° elevation the warm sky near the horizon reaches past half the target's contrast at the
frame's bottom edge, inside `visible_extent`'s window, and took drawn boxes to 380 px.
`visible_extent` now keeps only the seen pixels connected to one inside the truth box. Re-measured
with it, the real margin is 1.29 (1.13–1.48) wide and 1.50 (1.26–1.79) tall, and `DRAWN_MARGIN`
is (1.29, 0.104) and (1.50, 0.136).

