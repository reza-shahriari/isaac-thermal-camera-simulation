# ADR 0117 — The optical PSF's second factor: derived from the datasheet, not pasted

**Status:** Accepted. Fills the gap [ADR 0059](0059-mtf-conventions.md) left
open: it fixed *where* the aberration Gaussian is applied and left *what it is* to the config,
where it stayed zero on every camera.
**Date:** 2026-09-22
Roadmap: SC.4 (§8.3)

## Context

`MtfSpec.aberration_sigma_um` is documented as "the Gaussian fitted from a measured slant edge",
defaults to 0.0, and no shipped sensor YAML carried an `mtf:` block. So every rendered camera in
this repository was **diffraction-limited** — the best lens physics allows, and not a lens anyone
sells. The Tier 2 MTF bench measured that camera and pinned its answer, so the defect had a passing
test on top of it.

A zero is not a neutral default here. It is a claim, and it is the strongest claim available.

## Options considered

**Leave it zero and say so.** Defensible for a camera with no published figure, and taken for
three of the five (below). Not defensible for the Boson, which is the camera the whole validation
lane is built around and the one whose datasheet publishes the number.

**Fit σ to a golden or to a plausible-looking edge.** Rejected. The point of the parameter is that
it carries external information; a value fitted to this repository's own output carries none, and
would make the bench a tautology.

**Paste the solved number into the YAML.** What a hurry looks like, and where this nearly landed.
Rejected because nobody could tell later whether 1.654 came from the datasheet or from a fit. See
the decision.

## Decision

**The number is solved, in code, from the published figure.**
`irsim.optics.mtf.aberration_sigma_for_mtf(target, ξ, λ, F)` inverts

    MTF_diff(ξ) · exp(−2π² σ² ξ²) = target

and `tests/unit/test_lens_mtf.py` re-runs the solve against what the YAML carries, so the config
and the datasheet cannot drift apart silently. For the Boson at ξ_N = 41.667 cyc/mm, λ = 10.5 µm,
F/1.0 and FLIR's 42 % nominal on-axis figure: **σ = 1.654 µm**.

**The detector footprint stays out of the solve.** The datasheet figure is the lens; the box filter
is the pipeline's (ADR 0059). Folding it in would count it twice and the result would still look
like an image — just a softer one, with the system MTF at 0.17 instead of 0.27. That is a test, not
a comment.

**The check is on the lens factor, not the system.** This is the part worth remembering. The
roadmap's acceptance is the system MTF at 0.27 ± 0.03, and a **diffraction-limited Boson gives
0.294 — inside that band**. The band would have passed the broken camera. What separates them is
the lens factor against the datasheet, 0.420 authored against 0.461 ideal: a 10 % error that the
box filter dilutes to 9 % of a number with an 11 % tolerance on it. Acceptance bands inherited from
a system measurement do not always constrain the component that is wrong.

**Two cameras get the figure; three keep an ideal lens, on the record.** Both Bosons share f/1.0,
a 12 µm pitch and the same band, so the Nyquist frequency and the diffraction term are identical
and only the field angle differs; the 9.03 mm variant's own figure was not found, so it takes the
family nominal and is marked ESTIMATED in a way the 640 is not. The MWIR InSb, SWIR InGaAs and
NIR silicon examples are generic parts with no named lens — cooled MWIR optics are specified
diffraction-limited over their own field, and for the other two there is nothing to solve against.
They are listed in `IDEAL_LENS` with their reasons and a test fails for any camera in neither
branch, so the next one added has to decide rather than inherit a zero the way all five did.

## Consequences

**The Boson's images changed, and in the shape an aberration should.** Twelve golden arrays moved
and seventeen did not, which is itself the check:

| golden | max |Δ| | what it is |
|---|---|---|
| `boson_ramp_apparent_t` | **7.8 mK** | a smooth gradient has almost no high-frequency content to lose |
| `boson_hot_patch_apparent_t` | **3.97 K** (rms 0.21 K) | hard edges, where a blur does its work |
| `boson_sitf_dn`, `boson_noise_cube` | **bit-identical** | uniform fields; no PSF can change one |
| MWIR / NIR / SWIR frames | **bit-identical** | only the Boson configs changed |

Edges move, flats do not, and cameras that were not touched did not move. Regenerated deliberately
with `make golden-update` after that comparison, not to silence a failure.

**The Tier 2 bench now measures the shipped camera**, reading σ from the YAML rather than typing a
0.0 beside it, so the bench and the renderer cannot disagree about which camera they describe. Its
Nyquist expectation moves from 0.31 to 0.27.

**Still estimated:** the figure is FLIR's *nominal on-axis* value, so the modelled lens is uniform
across the field where a real one softens toward the corners. One Gaussian also stands for
aberration and defocus together (ADR 0059) and cannot separate them.

## Revisit when

A measured slant edge for one of these cameras arrives — `tests/unit/test_tier2_mtf.py` already has
the comparison wired behind `measured_path("mtf", …)` and skips for want of the file. At that point
σ stops being solved from a nominal figure and becomes what the docstring always said it was.
