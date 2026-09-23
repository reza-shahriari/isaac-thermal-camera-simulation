# ADR 0131 — Partial occlusion is exact for sky and sea, and measured elsewhere

**Status:** Accepted
**Date:** 2026-09-23

## Context

`OC.6` blurs each depth layer with its own kernel and composites back to front on the layer's blurred
coverage. A defocused foreground silhouette therefore becomes semi-transparent — which is correct —
and the background shows through it. But *what* shows through was never rendered: a single-layer
G-buffer has one surface per pixel and no behind. This is the standard partial-occlusion problem of
post-process depth of field, and the owner asked directly how to solve it or lower its impact.

The project's own priority order decides how much it matters. A target against sky, and a target
against open sea, are the first two lanes; ground clutter is third.

## Options considered

1. **Ignore it.** A plain `over` leaves a deficit wherever the alpha opened a gap no farther layer
   fills, and that deficit is a dark fringe along every out-of-focus silhouette.
2. **Normalise by the accumulated alpha** — fill the gap with the layers that *are* visible, in
   proportion to how much of each is. Free, and it makes a flat field survive exactly.
3. **Analytic background.** Where the thing behind is sky or sea, its radiance is a function of ray
   direction this pipeline already evaluates for the pixels where it is visible. Evaluating it for
   every pixel costs nothing more.
4. **Extrapolate from the visible background** (push-pull pyramid fill).
5. **Render a second depth layer** (depth peeling). Exact, and a second render pass per frame.

## Decision

**2 as the floor, 3 wherever it applies, 4 as the fallback, 5 written up and not scheduled.**

`layered_defocus` normalises by default (2). Given a `background_t_k` G-buffer plane it uses that
instead (3), and the plane is the backmost layer **completed** — what that layer shows where it is
visible *and* what it would show where a nearer layer hides it — so it replaces the backmost layer
rather than sitting behind it. `estimate_background` supplies (4) for scenes with no analytic
backdrop.

## Consequences

**Measured, in apparent temperature, against a two-layer reference** at three depth ratios, on a
scene with background structure hidden entirely behind the foreground:

| near / far | normalised (2) | push-pull (4) | analytic (3) |
|---|---|---|---|
| 3 m / 30 m | 4.93 K | 4.87 K | **0.0000 K** |
| 3 m / 200 m | 5.06 K | 5.03 K | **0.0000 K** |
| 8 m / 400 m | 2.87 K | 2.57 K | **0.0000 K** |

Three things follow, and the third is the uncomfortable one.

**The aerial and maritime lanes are exact, not bounded.** That is the whole reason this ADR ranks (3)
above everything else: the lanes the owner ranked first are the lanes where the hard problem does not
arise, and it costs no second render to make that true.

**The error is local.** Both approximations stay inside a band about the silhouette — under 5 % of
their own peak outside it — so the bound is a bound on a band and not on a frame.

**Push-pull buys 1–10 %, not an order of magnitude.** An extrapolation from the visible background
cannot recover structure that was never visible, and an earlier version of the measurement scored it
a perfect 0.00 K only because the hidden feature also continued into the visible region. It is worth
having because it is free and never worse, and it is not a solution. For ground clutter with real
depth complexity the honest answer is (5), a second rendered depth layer, and the reason it is not
scheduled is that it costs a render pass to serve the third lane. Revisit when a ground scene's
detection metric is actually measured against a real one.

**Two defects this measurement found in `OC.7` as first written**, both invisible on an easier scene
and both worth recording because they are the way this goes wrong quietly:

* The background seed was **not blurred**. The seed survives only in the hidden region, so a sharp
  seed put an unblurred background inside an otherwise blurred frame. Invisible wherever the hidden
  background is locally uniform, which is most test scenes. Worth 1.5 K here.
* The background was composited **behind** the backmost layer rather than replacing it, which counted
  that layer's visible part twice. Also worth about 1.5 K.

Neither would have been caught by a scene whose hidden background looks like its visible one, which
is the scene anybody writes first.
