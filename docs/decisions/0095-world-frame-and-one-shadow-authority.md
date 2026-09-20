# ADR 0095 — The scene declares its world frame, and shadow has one authority

**Status:** Accepted
**Date:** 2026-09-20
Roadmap: PT.18 (§5.4, §6.1)

## Context

PT.1 shipped a per-cell shadow (`cell_shadow`, exact ray–rectangle) and PT.17 a per-cell solve
from the scene config, and between them nothing in a scene config could reach the shadow:
`cell_shadow` had no caller outside its own tests, a patched surface under the sun took one
`shaded` bool for all of its cells, and the car driver's two fields carried **no solar term at
all** — a noon run would have rendered a sun-free bonnet. The point-wise scenes shipped so far
are pre-dawn for exactly that reason.

Closing the gap raised two questions the code had been avoiding.

*Which way is up?* Every sun vector in the package is ENU (M6.4's NOAA position, `solar_loading`),
and every patch and occluder is authored in the scene's world frame. The car scenes are Y-up,
because the stage they author is (`car_demo` sets `UsdGeom` up to +Y and `airframe.Part` assumes
it); every ENU convention is Z-up. `cell_shadow`'s docstring says converting the sun into the
patch's frame "is the caller's job and is deliberately not guessed at here", and no caller had
the information to do it.

*Who decides a cell is shaded?* `SurfaceSpec.shaded` zeroes the beam for a whole surface; an
occluder decides per cell. A scene carrying both would render whichever the code path happened
to apply last, and the author meant one of them.

## Options considered

1. **Assume ENU everywhere and rotate the car scenes' geometry.** Every patch in the two car
   configs, `CarGeometry`, the radiators, the camera and the stage authoring are Y-up; the cost is
   the whole driver and it buys nothing a declaration does not.
2. **Infer the frame from the stage's up axis at render time.** The thermal core cannot see the
   stage (CLAUDE.md #1), and the spin-up needs the shadow long before a stage exists.
3. **Declare the frame in the scene config**, default ENU, and derive east so the triad is
   right-handed by construction. One block, read once, exact for the axis-aligned frames every
   shipped scene uses.
4. **Per-surface occluder lists.** A slab shades the wall *and* the ground under it; listing it
   twice is how the two drift apart.
5. **Scene-level occluders that reach every world-frame patch**, and a load-time refusal of a
   patched surface that also says `shaded: true` while occluders exist.

## Decision

Options 3 and 5.

* Schema **v8**. `scene.world_frame: {up, north}` (`WorldFrameSpec`, default ENU, so every v4–v7
  scene reads exactly as before) and `thermal.occluders:` (`OccluderSpec`: name, centre, two
  axes, half extents; `frame` must be `"world"` for now). Both are validated at load through the
  dataclasses they build.
* `irsim.thermal.frames.WorldFrame`: `east = north × up`, `world_from_enu` with the three as
  columns, `to_world` / `to_enu`. The car scenes declare `up: +Y, north: −Z`, so east is +X and
  the car faces north.
* `CellForcing` (PT.17's seam) gains `patch`, `occluders` and `frame`. With occluders it gates the
  beam per cell — `solar_loading` on the **surface's** normal and sky view and the **cell's**
  visibility — so an unshaded cell evaluates the per-prim expression on the per-prim operands and
  stays **bit-identical** to the prim, spin-up included (the per-cell field is spun up on its own
  forcing when occluders exist, because a cell under an overhang has been under it all morning).
  It refuses a patch in a moving frame, a surface whose patch plane disagrees with its tilt and
  azimuth, and a surface that is also `shaded: true`.
* `ThermalSceneSpec` refuses `shaded: true` on a patched surface when the scene declares
  occluders: *two shadow authorities*. Without occluders the flag is what it always was.
* `Scene.solar_terms(t)` is the one sun every solar term in a scene uses. `car_demo` builds its
  fields' `q_solar` from it, turned into the stage frame by the declared `world_frame`, gated by
  the car's own box faces (`CarGeometry.shadow_casters`, `shadow.box_faces`) plus the config's
  occluders; it refuses a scene whose frame is not Y-up.

## Consequences

**What a scene can now show:** a wall half in sun from YAML alone (measured 9.9 K across one
concrete wall at 16:00 local, a diurnal transient rather than PT.1's fixed-beam equilibrium), and
a car's shadow on the road beside it. PT.20's reference scene is a config, not code.

**What it costs:** a scene with occluders spins its patched surfaces up per cell instead of
copying the per-prim state, which is the per-prim spin-up multiplied by the cell count (64 cells
for a day: about a second). A scene without occluders is unchanged.

**Limits, recorded not hidden:** hard-edged shadow (no 0.53° penumbra, PT.22), no light bounced
off the occluder, no per-cell sky view (the diffuse term and the longwave down are still the
tilt's own, PT.21), no occluder in a moving frame (needs the frame's pose, PT.9 / WM), and no
shadow on a per-prim surface — a surface without a patch keeps its `shaded` flag, and the scene
does not warn that its occluders cannot reach it. Non-patched surfaces under occluders are the
one quiet case left, and it is quiet because the flag is still an honest authority there.

## Revisit when

PT.21 gives cells their own sky view, at which point `q_longwave_down` leaves the broadcast by
the same seam; or a moving occluder is needed (a wing over a fuselage), which means the thermal
core taking a pose per frame name and `cell_shadow` being evaluated per tick in that pose.
