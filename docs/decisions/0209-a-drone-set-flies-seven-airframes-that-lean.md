# ADR 0209 — A drone set flies seven airframes, and they lean

**Status:** Accepted (EV.25)
**Date:** 2026-10-07

## Context

The rendered drone sets flew one Phantom 4, level: it yawed but never pitched or rolled. The
real Anti-UAV RGBT set holds several drone types, and its flattest boxes (drawn-box aspect p90
3.14) are out of a Phantom 4's reach at any elevation. ADR 0208 measured 1.83 at 1–10°, because
its landing skids keep it tall. The library already holds six other DJI multirotors: Mini 3 Pro,
Avata 2, FPV, Inspire 3, Matrice 100 and Matrice 300 RTK. Each is part-split and flies through
`render_phantom4.py`, as `zoo_bundle.py` already does for the catalogue stills.

## Decision

- **Airframes.**
  - `generate_aerial_dataset.AIRFRAMES` lists the seven as (asset, scene, nose bearing, span).
  - Clips take them in a shuffled cycle, so n clips fly min(n, 7) different aircraft.
    `--airframes <name>` keeps one.
  - The Phantom 4 flies its solved per-cell scene. The others fly `zoo_hero.yaml`, whose generic
    heat sources serve any part-split asset: motors hot, the pack warm, the shell near air. That is
    the level a catalogue still claims, not a solved airframe.
  - Each clip's range band is the Phantom 4's scaled by the airframe's span, from 0.212 m (Avata 2)
    to 1.032 m (Matrice 300), so every airframe covers the apparent sizes the band was set for.
- **Attitude.**
  - `irsim_isaac.asset_flight.tilt_matrix(pitch, roll)` leans an aircraft about its own axes.
    `render_phantom4.py` composes `spin(heading) · tilt · spin(−nose) · mount`, which is the level
    mount exactly when the tilt is zero.
  - `ScatterTrack` draws a tilt per pose: half-normal in magnitude, uniform in direction, capped at
    `TILT_MAX_DEG` = 25°. `WanderTrack` leans smoothly on two more seeded channels. Both are on
    streams of their own, so no position, heading or earlier draw moves.
  - `--tilt-sigma-deg`: 0 by default, so existing renders are unchanged; the planner passes 8°.
    ESTIMATED: a camera multirotor cruises at 5–20° of tilt, and DJI quotes 25° for the Phantom 4
    in P-mode.

## Consequences

Measured on 6 tilted scatter frames per airframe at 1–10°, before the range scaling, with boxes as
drawn (`scripts/box_convention.py --as-drawn`):

| | Aspect p10 / p50 / p90 |
|---|---|
| Phantom 4 | 1.35 / 1.55 / 1.71 |
| Mini 3 Pro | 2.33 / 2.63 / 3.41 |
| Avata 2 | 2.14 / 2.73 / 3.08 |
| FPV | 1.62 / 2.06 / 2.27 |
| Inspire 3, Matrice 100, Matrice 300 | 1.38–1.62 / 1.63–1.76 / 1.91–2.24 |
| All seven (39 boxes) | 1.31 / 1.68 / **2.55** |
| Real sky-only test | 1.21 / 1.52 / **3.14** |

- **The flat tail is now reachable.** The Mini 3 Pro and the Avata 2 give it. The pooled median
  (1.68) sits above the real 1.52, because a uniform share of airframes is not the real set's
  share. That share is unknown: the set does not label drone types.
- **Before the scaling the big airframes were too large on screen**: the Matrice 300's boxes
  reached 262 px at p90 against the real 83. The span scaling is a pinhole identity; the first
  rendered set will measure its sizes.
- Six airframes fly with generic per-part heat, not a solved field. Their thermal detail is a
  catalogue still's, and the Phantom 4 is the only solved one.
- The FPV and Inspire 3 meshes carry their artists' forward-flight pitch (36° and about 20° nose
  down). The drawn tilt adds to it.

## Revisit when

- The real set's drone types can be told apart. Match their shares rather than a uniform cycle.
- A library airframe gets a solved scene (`phantom4_parts`-style). Fly it in that.
