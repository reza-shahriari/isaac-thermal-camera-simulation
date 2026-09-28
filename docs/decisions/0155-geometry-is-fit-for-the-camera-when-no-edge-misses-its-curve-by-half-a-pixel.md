# 0155 — Geometry is fit for the camera when no edge misses its curve by half a pixel

Date: 2026-09-28
**Status:** Accepted
Roadmap: AI.10

## Context

The asset library (ADR 0150) takes models from two worlds that fail in opposite directions.
Game and phone-game models approximate curves with a few flat facets. Mechanical models carry
exact CAD surfaces (STEP/IGES, which Blender cannot open) or a tessellation dense enough to cost
millions of triangles the camera cannot resolve. The owner wants one skill, `low2high`, that
moves a model either way, and is willing to go as far as redesigning parts through the Blender
MCP when smoothing is not enough (2026-09-28).

A facet matters more here than in a visible-only renderer. Directional emissivity and the
reflected-sky term follow the surface normal (§4.3, §8), so each facet of a coarse hub reports its
own apparent temperature and the hub renders as bands. That is a structure a detector trained on
these frames can learn, and real footage never shows it. At the silhouette a facet is a polygon
corner against the sky.

So both directions need one yardstick, and it must not be an impression.

## Decision

**The criterion is each edge's own silhouette error against half a pixel.** Two facets of width
`w` meeting at angle θ are chords of a circle of radius `w / (2 sin(θ/2))`, and each misses its arc
by `(w/2)·tan(θ/4)`. That is exactly `r(1 − cos(θ/2))`, and the test suite checks the identity
to 1e-9. An edge is faceted when that exceeds half the pixel footprint `R·IFOV/2` at the closest
range the scene flies the object, and it is below the crease angle (60° by default, a stated
heuristic: a hex nut and a 6-segment cylinder are the same polygon).

This replaced a first design that used one fixed angle (10°). On the Phantom 4 that design flagged
36 of 41 meshes of a 2.49 M-triangle product model, because an 8-sided 1 mm screw head has the
same angle as an 8-sided fuselage. The per-edge criterion needs no radius. It lets sub-pixel detail
stay as coarse as it likes, and it gives the right answer at both ends:

| model | faces | faceted at 2 m, 1.36 mrad |
|---|---|---|
| Phantom 4 as authored | 2,486,459 | **0 of 41** meshes (also 0 at 0.5 m) |
| the same, decimated to phone-game density | ~100 k | **39–40 of 41** |

The fixed angle survives only as a fallback when no scene is given, and it is labelled as a list of
candidates, not a verdict.

**It measures the render mesh, never the thermal archive.** Planar dissolve (ADR 0132) merges
near-flat faces on purpose and would read as faceted. The render geometry is measured from the
`.glb` or from `prep_asset.py --emit-mesh --dissolve-deg 0`. Smoothing up never lifts the thermal
budget (AI.3): the two meshes keep separate budgets.

**STEP/IGES go through OpenCascade, installed, not written.** `cadquery-ocp` (the kernel under
FreeCAD and CadQuery) is an optional `cad` extra. Only `scripts/tessellate_step.py` uses it, in
whatever interpreter it is installed into, and nothing in `src/irsim` imports it. Tolerances come
from the scene (chord = half a pixel at closest range), and the CAD part tree survives as glTF
node names, in metres. Measured: names preserved, extent right to 0.1 mm, and the requested angle
bounds every facet on a curved part (4 tests, skipped where the extra is absent).

**The upward ladder stops at the lowest rung that passes.** Normals, then subdivision with
creases, then displacement from the model's own maps, then redesign through the Blender MCP from
references of the real object. Redesigned geometry is recorded as `ESTIMATED — redesigned` with
its reference URLs, the same flag a guessed material carries.

## Consequences

* "Is it smooth enough?" has a number and a scene attached to it, in both directions, and
  "we subdivided it" is checkable against it.
* The verdict depends on the scene. An asset that passes for the aerial lane's 2 m may fail for a
  0.3 m close-up, and the report says so rather than averaging it away.
* The crease heuristic cannot tell a designed polygon from a coarse curve. A hex part below 60°
  can be flagged, and the skill confirms faceted parts by rendering them.
* The OpenCascade test does not run in the default gate. It was run in an environment with the
  extra installed; GT-lane CI work could add the extra to a job.
* The GLB reader measures per mesh without applying node transforms. That is exact under rigid
  motion and uniform scale, and wrong under a non-uniform node scale, which is rare in CAD and
  game exports and named in the reader's docstring.
