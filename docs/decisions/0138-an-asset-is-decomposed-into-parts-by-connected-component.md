# ADR 0138 — An imported asset is decomposed into functional parts by connected component, and the parts are data

**Status:** Accepted
**Date:** 2026-09-24
Extends [ADR 0128](0128-per-asset-material-mapping-and-the-blender-prep-path.md) (the asset
pipeline and the Blender prep path) and [ADR 0132](0132-the-thermal-mesh-is-not-the-render-mesh.md)
(the thermal mesh is not the render mesh). Uses [ADR 0137](0137-an-assets-mesh-is-budgeted-by-what-conduction-can-resolve.md)'s
component measurement. Roadmap: `AI.5`.

## Context

ADR 0128 gave an imported asset a per-asset **material** map, and that closed the question of what
an asset is *made of*: the Phantom 4 went from 48.8 % to 100 % material coverage, with the shell's
areal heat capacity halved and the motor housings' emissivity corrected from 0.09 to 0.90.

It did not touch the question of what an asset *is*, and that question turns out to be the one
almost every infrared sentence about an aircraft depends on. The source asset is grouped by
material, because that is what a renderer wants — one draw call per material. All forty-one of the
Phantom 4's prims are named `GeometryNode_<n>`, and one of them is "all the white plastic",
spread over 987 disconnected shells. So the model can say *0.176 m² of this aircraft is white ABS*
and cannot say *this is a propeller*.

The cost of that was concrete and it was silent. `configs/scenes/phantom4_pointwise.yaml` declared
a `battery` heat source with a full 28-minute throttle schedule, and bound it to **no geometry at
all** — because no prim is the battery. The scene solved a quadcopter with no battery in it, and
rendered a completely plausible aircraft. Nothing in the pipeline could have said otherwise: an
unbound target is indistinguishable from a target whose geometry happens to be cold.

`render_phantom4.py` had the same shape of problem one level up: `TARGET_BY_MATERIAL = {"copper":
"motor"}` and everything else `airframe`. Two thermal nodes for an aircraft with four motors on
four duty cycles, a battery, and propellers with almost no thermal inertia.

## Decision

**Parts are recovered from connected components, not from names or prims.**
`scripts/prep_asset.py --emit-components` splits the mesh into connected shells inside Blender
(CPU only, no Kit, no CUDA) and writes per-component statistics: faces, area, centroid, bounds and
dominant material. The Phantom 4's 41 prims become **31,068 components** over 2,486,459 faces.
This works because a propeller is a separate shell from the motor it bolts to even when both are
moulded from the same white ABS.

**The decomposition is data, in the asset's own config.** A part is a name, a thermal target, and
a predicate over component statistics — authored in `configs/assets/<asset>.yaml` beside the
material map and validated by `irsim.io.asset_parts.PartsConfig`. Nothing in the module knows what
a quadcopter is, which is the requirement that matters: the next asset is a boat.

**First match wins, in declaration order.** A YAML reads most-specific-first, and the catch-all
shells are terminal. This is what lets the battery be found *inside* the shell that encloses it.

**An empty part fails the report.** `PartReport.empty_parts` is the direct answer to the unbound
battery: a part that matched no geometry fails the decomposition even at 100 % area coverage,
because the asset can be fully covered while the one part a scene's heat source needs is missing.

## What the decomposition found

19 parts, 100 % of 0.294 m² covered, nothing over the 60 % greedy ceiling:

| part | area | faces | note |
|---|---|---|---|
| `battery` | 0.00570 m² | 401 | 88 × 83 × 28 mm against DJI's published 88 × 78 × 35 mm |
| `propeller_*` × 4 | 0.00740–0.00774 m² | ~2 500 each | spread 4.4 % |
| `motor_*` × 4 | 0.00728–0.00729 m² | ~255 000 each | spread **0.0 %** |
| `motor_mount_*` × 4 | 0.00287 m² | 12 224 each | — |
| `camera_lens`, `gimbal` | 0.00638, 0.0278 m² | — | the only τ > 0 surfaces |
| `landing_gear`, `arms` | 0.0260, 0.0322 m² | — | — |
| `shell_upper`, `shell_lower` | 0.100, 0.0248 m² | — | terminal catch-alls |

**Four-fold symmetry is the correctness check, and it needs no reference data.** A quadcopter's
four stations are identical hardware, so the four `propeller_*` parts must come out the same size
and so must the four `motor_*` parts. A mis-authored selector does not produce that agreement.

## Two measurements that fell out, and one defect

**The asset is modelled pitched 2.93° nose-down.** Fitting a plane through the four motor cans
gives `z = +0.0018x + 0.0512y + 0.0298` with *zero* residual — pitch 2.93°, roll 0.10°. The four
stations therefore sit at four different heights spanning 18 mm, and a single global `z` cut
cannot separate a propeller from the motor beneath it: at the front-left station the blades are at
z = 0.062 while at the rear-right station the motor *can* reaches z = 0.064. An earlier revision
of the selectors used one global cut and silently lost nine tenths of the front-left propeller.
Each station now carries its own measured height.

**`NOSE_IN_ASSET` was wrong by about 30°, and the reasoning behind it does not hold.**
`render_phantom4.py` declared the nose as `(0, -1, 0)`, derived from the gimbal camera sitting
"−53 mm" from the airframe centroid. That measurement took only the **y** component: the camera
body and its `Crystal` lens elements actually sit at offset (−32, −52) mm, bearing **−121°**, not
−90°. The four arms are at bearings −73.7°, −163.5°, +16.5° and +106.4°, and the camera falls
almost exactly midway between the first two (−118.6°) — the X configuration a Phantom 4 has, with
the camera forward between the two front arms.

The supporting argument fails independently: that docstring reasons from DJI putting red LEDs on
the front arms and green on the rear, but in this asset **all four arms carry the same red lamp**,
equal area at each station, and the `Green_light` it relied on is a 0.07 cm² speck at r = 77 mm
near the body rather than an arm LED.

Consequence: every Phantom 4 clip rendered before this ADR flew the aircraft roughly 30° crabbed.
`NOSE_IN_ASSET` is now the bisector of the two front arms, (−0.4789, −0.8779, 0), taken from the
four stations rather than from the camera because four points square to 0.23° and equidistant to
0.6 mm are far better conditioned than one small component's centroid; the camera is kept as the
cross-check, agreeing to 2.9°. **Regenerating the affected outputs is a deliberate act and is not
part of this ADR** — the clips on disk are still the crabbed ones until somebody re-renders them.

## Options considered

**Author parts by hand as prim lists.** Rejected: the prims are material groups, so no list of
them is a part. This is the state the scene was already in — six prims chosen by hand, described
in their own config as "not a part but 'all the white plastic in one region'".

**Cluster automatically, with a multirotor model in the code.** Rejected: it puts "what a
quadcopter is" into `irsim`, and the roadmap's next asset is maritime. The geometry *measurement*
is generic and belongs in the tool; the *interpretation* is per-asset and belongs in its config.

**Best-match scoring instead of first-match.** Rejected: the outcome would depend on a ranking
nobody authored, and the failure mode — a part quietly losing geometry to a better-scoring
sibling — is the one this ADR exists to eliminate.

**Split the mesh into per-part objects at prep time.** Rejected for now: it multiplies the archive
and forces a re-prep whenever a selector changes. Selectors are cheap to iterate against a
component dump; splitting can follow if a consumer needs real per-part prims.

## Consequences

* The `battery` heat source is bound to geometry for the first time, and the four rotors have four
  independent nodes rather than one lumped `motor`.
* Parts give the heat-transfer-between-connected-parts requirement something to act on: a battery
  can only warm the shell around it if the battery is a part that exists.
* `AssetMapping.parts` is optional, so every existing asset config is unaffected.
* The four `motor_*` parts carry ~255,000 faces each, over ADR 0137's 200,000-per-prim budget.
  They are parts rather than prims so the gate does not fire, but a per-part budget is the obvious
  follow-on.
* The `gimbal` part at 9.5 % of the asset's area is looser than the hardware it names; it is a
  region rather than a mechanism, and tightening it needs the gimbal's own sub-shells separated.

## Revisit when

A second asset is decomposed — a maritime hull has no four-fold symmetry, so the correctness check
used here does not transfer and something else will have to replace it.
