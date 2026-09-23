# ADR 0125 — A cloud is not a stencil, and a picture of a temperature needs its scale

**Status:** Accepted. Two fixes to what a rendered frame *shows*, both raised by looking at the
output rather than by a test: the cloud rendered as flat blobs in both bands, and a close-up of
an aircraft had no way to read a grey level back to a temperature.
**Date:** 2026-09-23
Roadmap: PT.9 (§5.3 a, §11.3)

## Context

ADR 0124 put cloud into a rendered scene for the first time. It looked wrong immediately, in both
bands, and the reason was one line: `SkyFixedCloud.sample` returns a **boolean**. Every consumer
took it as a stencil —

* the visible dome replaced each covered texel with one flat base value;
* the infrared background gave each covered ray the full cloud-base radiance.

So every cloud was a region of exactly uniform brightness with a one-sample cliff around it. No
thin edges, no internal structure, and in LWIR a step of tens of kelvin along every boundary —
which is precisely the edge statistic a sky-target detector keys on, so the error was not
cosmetic.

Separately, a close-up of the aircraft showed parts at plainly different greys and gave a viewer
no way to turn a grey back into a temperature. The burnt-in gauge says what each *named* part is;
it says nothing about the mapping, so "that part is brighter" remained an observation about the
display.

## Decision

**A cloud has a depth, not a membership.** `SkyFixedCloud.density` returns 0 to 1 along each ray:
a smoothstep on the underlying 1/f^β field's excess over its own threshold, measured in standard
deviations (the field is unit-variance by construction, so the softness parameter is directly in
σ). The radiance blend becomes `ε_eff = (1 − τ_cloud) d`, with the authored `τ_cloud` now meaning
the transmittance of cloud **at full depth**.

`density` crosses 0.5 exactly at the threshold, so `sample` and the authored coverage fraction
still mean what they did — this changes how a cloud looks, not how much sky it covers. And
`softness = 0` returns the hard mask exactly, so the old behaviour is a value of a parameter
rather than a deleted branch.

**A fixed-span frame carries the palette beside it.** `irsim_eval.video.palette_scale` draws the
colour bar from the *same lookup table the display branch indexed*, with Celsius ticks.

**A close-up is a different measurement, and the driver says so.** `--close-up` holds the aircraft
filling the frame for the whole mission, so the only thing changing is temperature.

## Consequences

### The change is provably confined to the pixels that were on the edge

`cloud_radiance` returns its two limits through `np.where` rather than through the blend, so a
boolean mask gives **bit-identical** radiance to the hard-edged version at any `τ_cloud`. Every
existing cloud result stands; what moved is the fringe, which did not exist before. Measured on
the rendered dome: at 0.45 coverage the fringe is 9.2 % of the map against 19.6 % at full depth,
and at 0.05 coverage it is 3.0 % against 2.4 % — **thin cloud is more edge than core**, which is
why the stencil looked worst exactly where cloud was sparse.

The mean-preservation property that makes the structured field compatible with every LUT and tilt
integral is unaffected: it is a statement about the average over the sky, and a symmetric ramp
about the threshold does not move it.

### The softness is estimated and says so

0.45 σ was chosen by looking at the result, not derived. This generator has always been a
**fixture for clutter statistics rather than cloud microphysics** (ADR 0070), and nothing
radiometric is claimed for the particular width. What can be claimed is that the previous value
was wrong: zero makes every cloud edge a step discontinuity at the sampling resolution, and a real
cumulus edge is not that.

### What the colour bar is not drawn beside

Under either §11.3 AGC mode the mapping is rebuilt from each frame's own histogram, so a bar drawn
once would be wrong for every frame but one. The AGC video therefore gets no scale — and that
difference between the two videos is the lesson rather than an omission, because it is exactly
what separates a picture of contrast from a measurement.

### Two spans, and the tight one is anchored on the cells

The motors run forty kelvin above a skin whose whole structure is four, so one linear span cannot
show both. The tight span is taken over the **solved cells across the whole mission**, half a
kelvin either side — not over the nodes, which the motors would set, and not relative to air,
because how far the skin sits from air is the measurement and a span defined from air would move
whenever the weather did.

### From below, the deck is still hidden

A close-up from a camera looking up sees the belly, the skids, the gimbal, the arms and the motor
bells — five parts at four distinct levels — and does **not** see the top shell, which is the
hottest part of the airframe. That is the same geometry ADR 0123 recorded and it has not changed.
Showing the deck to a ground camera needs either a view from above, whose background is ADR 0060's
analytic ground rather than sky, or an aircraft authored at a flying attitude so that its patches,
its occluders and its prims all pitch together. Neither is done here.
