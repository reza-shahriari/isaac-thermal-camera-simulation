# ADR 0199 — A skinned body reaches the AOVs exactly as Blender skins it

**Status:** Accepted
**Date:** 2026-10-07
Roadmap: HU.10 (§13.3, §6.1). Builds on [ADR 0014](0014-isaac-sim-6-1-findings.md)
(ids and geometry from the renderer, temperature looked up by prim) and
[ADR 0192](0192-a-human-is-a-fixed-taxonomy-labelled-by-its-skeleton.md) (one prim per segment).
Revisits the static-asset choice recorded in ADR 0196.

## Context

Every person the project has rendered is static. HU.7 found MakeHuman's garments arriving with
no skin weights, and a USD that binds them to a skeleton makes the renderer collapse them. So
`prep_human.py` drops the rig, and every labelled human is a rest-pose statue. A pedestrian
walks, though, and the IR pipeline follows a walking body only if three things hold on this
build. None of them is documented for `UsdSkel`:

1. `instance_id_segmentation` gives each skinned segment mesh its own id, mapped to its own prim
   path, and the id is stable from frame to frame. Temperature is looked up by prim (ADR 0014).
2. The position AOV reports the skinned surface, not the rest pose.
3. No geometry plane is float16.

## Options considered

1. **Assume it works and build the walking render.** Rejected. A renderer that skinned the
   colour pass but not the geometry AOVs would produce a plausible walking image with every
   temperature at rest.
2. **Compare the renderer with itself across frames.** Rejected. It cannot tell a correct
   deformation from a consistent wrong one.
3. **Hold the renderer to Blender's skinning, pixel by pixel.** Chosen.

## Decision

- `prep_human.py --keep-rig` keeps the skin segments skinned to MPFB's game-engine rig. The
  default stays static.
- `scripts/animate_human.py` (Blender) keys a procedural walk, `gait_angles`, with Winter's
  normative sagittal ranges: two strides of 1.1 s at 30 fps, on the spot. It exports a skinned
  USD with one `Skeleton` and seventeen bound meshes. For every frame it also writes each
  segment's skinned world vertices and triangles, and its centroid. These are the answer.
- `scripts/probe_isaac_skel.py` with `irsim_isaac.skel_probe` renders the walk in Kit through
  the production `AovReader`. Its criteria:
  - every pixel of a segment lies on that segment's skinned triangles at that frame, p95 within
    5 mm;
  - for every segment that moved, the same pixels miss the rest pose by more than 3 cm, and by
    3× the posed fit (the control);
  - ids are stable per prim;
  - every plane is float32;
  - each hand travels more than 0.2 m.
- **The visible centroid is context, not a criterion.** The first run judged centroids. It
  failed at 0.16 m on the back, which a front camera barely sees, and at 4 cm on legs that
  swing behind one another. That is a visibility bias, not a renderer error, and no tolerance
  separates the two. Nearest-vertex distance failed the same way at 2.5 cm on the shins,
  half of MakeHuman's long faces. Point-to-triangle distance is exact.

**Measured** (bare `man_adult`, 67 frames, 768 × 768, A6000):

| quantity | value |
|---|---|
| worst p95 pixel-to-skinned-surface distance | 8.7 × 10⁻⁷ m |
| rest-pose control, feet | 0.50–0.56 m off |
| rest-pose control, hands | 0.17–0.22 m off |
| ids | stable on all 17 prims |
| position, distance and normal planes | float32 |
| hand travel, rendered (Blender 0.257 m) | 0.239 m |

Kit's skinning reproduces Blender's linear-blend skinning to float32 precision.

## Consequences

- **Per-segment temperature follows a walking body.** The id-to-prim lookup the IR camera uses
  is stable, and it lands on the right surface. The probe paints each frame's segments with
  JOS-3's skin temperatures (`walk_temperature.mp4`). That is the per-prim lookup made
  visible, not a camera render.
- **Not yet a walking IR render.** `render_phantom4.py` reads the static parts USD from
  `prep_asset.py`, which regroups prims and carries no skeleton. Rendering a walking person
  through the full camera chain needs that path to keep the `SkelRoot` and the time samples.
- **Garments do not walk yet.** MakeHuman's garments arrive unskinned (ADR 0196). Before a
  dressed person can walk, they need weights transferred from the body. Kit-generated pieces
  cut from the skin already carry the body's weights.
- **The walk is procedural.** It is on the spot, sagittal only, with no pelvis rotation or
  vertical bob. A captured clip (CMU, Mixamo) retargeted in Blender is the realism upgrade.
  Per the owner (HU.1), it stays local and is never committed.

## Revisit when

- A walking person is wanted in a camera render. Then the parts pipeline keeps the skeleton,
  and this probe becomes that render's integration test.
- A later Isaac build changes skinning (dual-quaternion, GPU skinning, or deformers). The
  probe's surface criterion then decides whether Blender's answer still holds.
