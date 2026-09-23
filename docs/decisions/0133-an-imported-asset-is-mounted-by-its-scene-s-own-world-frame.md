# ADR 0133 — An imported asset is mounted by its scene's own world frame, and it flies

**Status:** Accepted
**Date:** 2026-09-23
Extends [ADR 0128](0128-per-asset-material-mapping-and-the-blender-prep-path.md) (the asset
pipeline) and [ADR 0073](0073-visible-companion-environment-dome.md) (the
companion visible frame). Roadmap: `AI.2`.

## Context

ADR 0128 ended on a principle: *the asset's frame is the stage's frame*. The prepared USD comes
out of Blender Z-up at `metersPerUnit = 1`, so the first driver to film it authored a **Z-up
stage** and transformed nothing. The reasoning was sound as far as it went — a second scale or a
silent axis swap is exactly what that ADR exists to prevent — and the first still it produced was
wrong anyway. The aircraft came back apparently pitched over, rolled about thirty degrees, with
nothing in the physics to blame.

The cause was not the mount. `look_at_quaternion` builds an azimuth/elevation mount around a
**world up vector**, and its default is the stage convention `(0, 1, 0)`. Called on a Z-up stage
it therefore levelled the camera against an axis that is horizontal, which rolls the horizon by
whatever angle the geometry happens to give. Nothing raised: a rolled camera is a valid camera.

That was not the only thing the Z-up stage broke, only the visible one. Every piece of environment
machinery in this repository reads the stage's **+Y as up and its -Z as north**:
`visible_sky.stage_direction`, `visible_sky.latlong_directions`, and the `DistantLight` aimed by
`stage.author_environment`. A Z-up stage silently disagrees with all three, so the infrared frame
would have been lit and backed by a sky in one frame while the companion visible frame showed
another — and a registered pair that disagrees is worse than no pair, which is ADR 0073's own
argument turned against it.

## Options considered

1. **Keep the Z-up stage and teach the environment about up axes.** Correct in principle, and it
   is a change to `visible_sky` and `stage` — a shared, well-tested module with several callers —
   for the benefit of one driver. Deferred, not rejected: if a second imported asset arrives with
   a third convention, this is the answer.
2. **Write `rotateX 90` on the reference and move on.** What almost every pipeline does, and the
   sign is a coin flip. A mirrored or backwards mount produces a completely convincing aircraft
   facing the wrong way, and the render is not a test that would catch it.
3. **Derive the mount from the scene's own `world_frame:` block** (chosen).

## Decision

**1. One rotation, computed from the scene config.** `irsim_isaac.asset_flight.world_frame_to_stage`
turns a scene's `world_frame: {up, north}` into the 3×3 that takes its world coordinates into
stage axes, by construction satisfying `M·up = +Y`, `M·north = -Z`, `M·(north × up) = +X`. The
Phantom 4 scene declares ENU, so it mounts with that rotation; a scene already authored in stage
axes mounts with the identity, and pays nothing. East is derived rather than declared, exactly as
`WorldFrameSpec` derives it, and the function refuses a degenerate or left-handed frame instead of
guessing — a determinant of −1 there is a mirrored aircraft.

This is not a weakening of ADR 0128. The prohibition there is on an *undeclared* transform: a
second `scale_to_metres`, or an axis swap nobody wrote down. This rotation is declared, in the
scene config, by the same block the thermal solver already reads.

**2. The stage stays the renderer's Y-up.** The environment, the sun, the sky model and the
`look_at_quaternion` default all already agree on it. Bending one asset is cheaper and safer than
bending four modules, and the asset is the thing whose convention is foreign.

**3. The solver's frame stays the asset's.** The scene's patches are authored in the archive's own
Z-up coordinates. They are not bound to the camera in this driver — temperature is still per prim
— so the mount moves the rendered aircraft and touches nothing the solve depends on. **The day a
`MeshPointBridge` does reach the camera, that binding must carry this rotation**, or every
closest-point query will land on the wrong cell. Recorded here because it will not be obvious then.

**4. `FORWARD_AXIS_VECTOR` stays, and is now a fix without a user.** `IrCamera.planes` passed its
up axis to `azimuth_from_rays` without a matching forward, and the default `(0, 0, -1)` is
*parallel* to Z-up — so a Z-up stage raised "forward must not be parallel to up" on its first
frame. Fixed and tested when it was found, and the driver that provoked it is now Y-up. The fix is
kept because the next Z-up caller should meet a working camera, not this bug again.

**5. The aircraft flies a lemniscate, not an orbit.** `FigureEightTrack`: one circuit in front of
a ground observer, over the mission's full 28 minutes. An orbit at constant radius is easier and
shows a target that translates but never turns — the aspect angle is ±90° for the whole pass. A
figure of eight crosses its own track, so one circuit takes the line of sight through both
broadsides and tail-on while the slant range swings 2.2:1 and the aircraft grows from 46 px to
104 px across. The defaults are chosen against the sensor, not for looks, and the one hard
constraint is **elevation**: this scene authors no terrain, so an aircraft below the observer's
horizon is backed by sky-model radiance at near-air temperature instead of cold sky. The track
holds 10–53°, and there is a test that says so.

## Consequences

* A scene config's `world_frame:` block is now load-bearing for rendering, not only for the
  thermal solve. A scene that declares it wrongly gets a wrongly mounted aircraft — but it already
  got wrongly shaded patches, so this adds a symptom rather than a failure mode.
* The infrared frame and the companion visible frame come out of one capture on one pixel grid,
  which makes the companion usable as a check on the infrared rather than as decoration. The nose
  direction of this asset was determined from the archive (the gimbal, its lens, and DJI's green
  rear LEDs) and **confirmed in the companion frame**; there is no other way to check it, because
  the asset's prims are all named `GeometryNode_<n>`.
* Option 1 remains open and is the right answer for a *second* imported convention. What this ADR
  buys is that the driver does not have to be rewritten when it lands: the rotation is already a
  named function with tests, and making it the identity is all that changes.
