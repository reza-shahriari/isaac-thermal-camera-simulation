# ADR 0134 — A receding surface is one surface, not a stack of occluders

**Status:** Accepted
**Date:** 2026-09-23

## Context

`OC.6` (ADR 0129) splits a frame into depth layers, blurs each with its own kernel and its own
coverage, and composites them back to front with the standard `over`:

    colour = premultiplied + colour * (1 - alpha)

That is right between a foreground and the background behind it, and it is what every layered
depth-of-field implementation does. `OC.6`'s tests measured it on a **near slab against a far
background** — two flat planes with empty space between them — and it passed, correctly.

Building `OC.12`'s cube-against-sky demo put a *continuously receding* surface in front of it for
the first time: an isometric cube whose three faces recede 0.69 m from the near corner. The binning
sliced the cube's own depth range into layers, and the render came back with **dark diagonal seams
ruled across the faces**, one at each bin boundary.

The seams are not a rendering curiosity. They are the `over` operator being asked a question it
cannot answer. Two slices of one surface do not hide each other: at a pixel on the boundary between
them the aperture bundle lands partly on one slice and partly on the other, so their contributions
**add**. Composited with `over` the farther slice is multiplied by `1 - alpha_near` instead, and the
deficit is exactly

    alpha (1 - alpha) (L_surface - L_behind)

which peaks at a quarter of the surface-to-background contrast where two slices meet half and half,
and is filled with whatever lies behind — the background. Hence a dark seam.

Measured on the demo's 0.6 m cube at 3 m against a 250 K sky: **8.6 K peak**, with RMS over the
cube's interior rising **1.31 → 1.44 → 2.32 K** as `max_layers` went 3 → 4 → 8. The default is 8.

Two things make this worse than a cosmetic defect. First, **the error grows with the layer cap**,
which inverts the only quality knob the stage has — the shipped default was its worst setting.
Second, continuously receding surfaces are not a corner case: a ground plane, the sea, the flank of
a vehicle and the receding face of any solid are all of them, so every lane past the aerial one
meets this.

## Options considered

**Leave it and cap the layers at two.** Removes the seam by removing the feature: a receding
surface gets one kernel, which is `OC.5` again and the thing `OC.6` exists to improve on.

**More layers, smaller seams.** Wrong direction, measurably — the RMS above rises with the cap. One
seam is as deep as eight; adding layers adds seams rather than shrinking them.

**Blend whole-frame blurs by per-pixel circle of confusion.** The classic games approach: blur the
whole frame at N radii and interpolate by each pixel's own CoC. Continuous in depth, so no seams at
all — but no occlusion either, and a defocused foreground failing to spread over its background is
precisely the defect `OC.6` was built to fix. It trades this artefact for the previous one.

**Composite `over` only where layers genuinely occlude. (chosen)** Keep `over` between layers with
empty space between them, and add between layers that abut.

## Decision

`layered_defocus` composites `over` **between layers separated by a depth gap** and **additively
between layers that abut**. `DepthLayer` carries the depth range each layer spans, and `separated`
decides: two layers are separate objects when the gap between their spans is wider than either
layer is itself deep.

The comparison is against the layers' own extents rather than a distance in metres, so it carries
from a 0.6 m cube to a 200 m ground plane with nothing to tune. A surface sliced into layers leaves
gaps of one depth sample, which is small for any scene; two objects with a void between them leave
a gap that is not.

The span is the layer's own min and max range and deliberately **not** its bin edges. Bins are
equal-width in W020 and W020 is V-shaped about the focus distance, so one bin can hold pixels from
either side of focus, and sorting layers by range walks the bin indices up and back down again —
two neighbours in the composite are routinely several bins apart. An earlier version of this fix
tested bin-index adjacency and was wrong for exactly that reason.

The **background** never gets the exemption. It is complete — it has coverage at every pixel,
including behind the geometry (`OC.7`) — so everything geometric is genuinely in front of it, and it
is composited under the accumulated geometry alpha.

## Consequences

The seam goes from 8.6 K to **0.16 K peak** on the demo cube, and the residual now **falls** with
the layer cap instead of rising. On the regression scene in `tests/unit/test_continuous_depth.py`,
against the `over` chain reimplemented there as the reference: peak **12–25× better**, and RMS
0.0125 → 0.0076 as the cap goes 3 → 8 where the reference runs 0.209 → 0.497.

`OC.7`'s exactness is untouched — the background path is bit-for-bit what it was, and its 1e-12
test still passes — and so are `OC.6`'s and `OC.8`'s measurements, because all three use scenes
whose layers are genuinely separated.

**What is left, and why it is left.** The interior is not exact. Adjacent layers carry different
kernels, so the geometry's blurred coverage `sum_i K_i * cover_i` ripples about **±0.5 %** across a
hard bin boundary instead of staying at one, and where it dips the background fills the difference.
That is 0.5 % of the surface-to-background contrast against the 25 % the `over` chain left, it is
below NETD in RMS, and unlike the seam it shrinks as layers are added. Removing it needs
**fractional membership** — a pixel belonging partly to two adjacent bins rather than wholly to one
— which changes how layers are *built* rather than how they are composited, and is recorded as
`OC.13` rather than taken here.

The one regression this accepts: two genuinely separate objects that happen to land in abutting
depth spans are read as one surface and composited additively. Their kernels then differ by an
eighth of the frame's W020 range at most, so the two rules nearly agree, and the case is bounded by
the same partial-occlusion argument ADR 0131 already records.

`depth_layers` now returns `DepthLayer` records rather than `(mask, distance)` tuples. The only
callers were inside `irsim.optics.layered` and its tests.
