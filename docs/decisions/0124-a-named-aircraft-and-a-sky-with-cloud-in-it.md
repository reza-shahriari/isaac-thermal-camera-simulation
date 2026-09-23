# ADR 0124 — A named aircraft, and a sky with cloud in it

**Status:** Accepted. First *real* airframe in this repository — an aircraft laid out from a
manufacturer's published specification rather than invented to be convenient — and the first
rendered scene whose sky carries structure in both bands.
**Date:** 2026-09-23
Roadmap: PT.9 (§6.6, §15 T3)

## Context

[ADR 0123](0123-the-camera-moves-so-the-aircraft-can-keep-its-shadows.md) rendered a point-wise
quadrotor against sky. Two things about that scene were the simulator's convenience rather than
the world's:

1. **The aircraft was invented.** `POINTWISE_QUAD` is a plausible heavy-lift frame in carbon
   fibre, sized so that its parts resolve. Nobody flies it. "What does a drone look like in the
   infrared" has no answer until you say *which* drone, because the answer is dominated by two
   properties that vary enormously across the class: how big it is, and what its skin is made of.
2. **The sky was empty.** The weather fixture carries `cloud_fraction` 0.05, and
   `generate_sky_cloud` lays coverage over the whole hemisphere, so a 31 × 25° field on a 0.05
   sky almost never contains a cloud. Every frame this project had produced against sky was a
   smooth gradient — the easiest background a sky-target detector will ever see.

## Decision

**A DJI Phantom 3, from DJI's published specification.** `irsim_isaac.phantom3` lays out the
aircraft from the four numbers the manufacturer publishes — 350 mm diagonal, 9450 propellers
(239 mm, 5.0 in pitch), 2312 motors, the battery in the rear of the shell — with the shell plan,
body depth, gimbal and skids **ESTIMATED from photographs scaled on that diagonal** and labelled
as such in the module. The alternative was importing a downloaded mesh, which needs asset-import
machinery, a per-prim material mapping and a binding from the point-wise field onto an arbitrary
asset's triangles (the follow-on to `WM.7`, unwritten) — and would have cost the per-point
gradient in the meantime.

**A new material, `abs_plastic_white`,** because a Phantom is not made of carbon and the
difference is the whole result. And **a new weather fixture,
`data/weather/scattered_cumulus_48h.csv`** at cloud 0.45 — SCT, 3–4 oktas — because the cloud
machinery of ADR 0076 was complete and had no scene that exercised it.

**Cloud is on by default in the outbound driver.** One `SkyFixedCloud`, seeded from the scene's
own weather, sampled per ray by the infrared background *and* baked into the visible dome, so the
two bands cannot disagree about where the cloud is.

## Consequences

### The Phantom is a much weaker daylight target, and that is the finding

| | heavy-lift, carbon | Phantom 3, white ABS |
|---|---|---|
| sunlit skin over air | **+24.4 K** | **+2.3 K** |
| solar absorptivity | 0.90 | **0.25** |
| motor-to-motor | 840 mm | **350 mm** |

A white consumer drone absorbs **under a third** of the short-wave flux a composite one does, so
its skin sits within a couple of kelvin of the air while a carbon deck runs twenty-five above it.
Against a 50 mK NETD both are visible — 2.3 K is forty-six NETD — but the margin is an order of
magnitude apart, and a detector tuned on one is not tuned on the other.

**It is not a thermal-mass effect, and the test says so.** At the same 1.5 mm the two skins carry
2205 and 2520 J m⁻² K⁻¹, 12 % apart. The first draft of this work asserted "half the heat, so
twice as responsive" in three places; it was measured, it was wrong, and it was removed. A second
cause invented for an effect that already has one is how a model becomes unfalsifiable. (The
factor of two *is* real against painted steel, which is the comparison the material file makes.)

**In LWIR the pigment does nothing.** White ABS is ε = 0.95 and carbon ε = 0.90; both are
near-blackbodies at 10 µm. The colour that makes one aircraft white changes how much sunlight it
absorbed on the way to its temperature, and nothing about how it radiates once there. A thermal
camera cannot see the paint; it sees the consequence of the paint.

### A span quoted beside a layout instead of derived from it

`quad_outbound.SPAN_M` was authored as `2 × 0.42 × √2` — the X-quad form every multirotor spec
sheet quotes. That frame is a **plus**: its arms run due N, E, S and W, so opposite motors are
`2 × 0.42` apart and the constant overstated the aircraft by **41 %**. It survived a review, a
test suite and a 300-frame render because nothing measured it; the burnt-in readout of that clip
says "span 115.5 px" where the motors span 82. Both spans are now derived from the authored motor
prims and a test measures them, which is a thing a constant cannot be wrong about.

### An arm's patch and a plate's patch inset in opposite directions

A plate wants its **prim inside its patch**; a solid arm wants its **patch outside its prim**.
Seen from below, an arm presents its end caps and both side faces, none of which lie on the plane
the grid is drawn on. Authoring the arms like the plates put **1204 pixels** outside every patch
on the first Phantom render. `test_rendered_airframes.py` now samples every face of every patched
prim, rotated by its own yaw, and asks the real `PlanarPatch.contains` — no formula restated, so
the test can disagree with the scene file.

### Cloud needs a grid finer than the survey default

`generate_sky_cloud` defaults to half a degree, which through this camera is **ten pixels**: the
coverage mask reads as blocks, not cloud. The driver asks for six cells per degree — about three
pixels, under the PSF — for 4.7 MB and a second.

### Not modelled

The cloud is a **fixed** field: it does not advect, so a long clip shows cloud that never moves
while the camera slews past it. Cloud base height, thickness and the LWIR temperature profile are
ADR 0076's single effective base; there is no cumulus vertical structure and no shadow cast by
cloud onto the aircraft. The Phantom's ESCs are inside the shell and present no surface, so the
scene declares no `esc` target — their heat would reach a camera by conducting into the skin, and
that route from a config is still `PT.9`'s remainder.
