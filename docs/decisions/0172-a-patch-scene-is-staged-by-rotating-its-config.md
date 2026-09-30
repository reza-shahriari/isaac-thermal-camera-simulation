# 0172 — A patch scene is staged by rotating its config, one quad per surface

Date: 2026-09-30

**Status:** Accepted (2026-09-30, IG.20). Builds on ADR 0087 (the patch field), ADR 0095
(occluders, world frame) and ADR 0171 (companion-only prims).

## Context

Phase P's reference scenes (`wall_half_in_sun.yaml`, `wet_road_noon.yaml`,
`car_exhaust_plume.yaml`) are thermal scenes. Each is a set of world-frame patches that names
the prim its field is drawn on, and each is solved and tested engine-free on a synthetic
G-buffer. None had a stage, so none had ever been rendered. Their roadmap rows all said "the
frame is IG.2's".

Two frames disagree. The scene configs are authored in ENU (x east, y north, z up). The Isaac
glue is Y-up with -Z forward: `sun_direction_stage`, the dome, and the camera helpers all assume
it. The point bridge then compares each pixel's *stage* position with the patch's own
coordinates.

## Decision

1. **Rotate the config, not the stage.** `irsim_isaac.patch_stage.to_stage_frame` rewrites
   every world coordinate through one proper rotation, `ENU_TO_STAGE`. That covers patch origins
   and axes, occluders, and plume origins and directions. It also declares
   `world_frame: {up: +Y, north: -Z}`, so the sun is still placed from the site and the clock.
   A config already in a stage frame is returned unchanged. A config with mesh surfaces or
   solved objects is refused rather than half-rotated.
2. **One quad per surface.** A shared `prim_path` (the west wall's concrete and render halves)
   is split into `<prim_path>/<surface>` children. Each quad carries its own
   `thermal:material`. Quads are drawn 1 mm inside their patch, so a rendered point cannot round
   outside its field under `strict_patch_coverage`.
3. **A per-prim fallback exists but is never shown.** Every quad is bound to a `patch_fallback`
   airframe node at air temperature, because `IrCamera` requires a node per prim. With strict
   coverage, a pixel that falls back raises an error rather than rendering.
4. **A water film is a visible-only sheet.** Latent cooling is already in the infrared field.
   The companion needs to see where the road is wet, so a dark, glossy sheet 2 mm above the
   film's region is passed to `IrCamera(companion_only_prim_paths=)`.
5. **The time-lapse runs on one fixed span.** The span is taken from the solved cells over the
   whole run, so any two frames are comparable. The camera's AGC clip is written beside it.

## Consequences

- A rotation changes no physics. `test_patch_stage.py` shows the rotated road solves every
  cell to the same temperature (1e-6 K) at 0, 30 and 60 min, and casts the same shadow cells.
  That equality is what makes the approach legitimate.
- A new patch scene needs only a camera preset in `scripts/render_patch_scene.py`.
- The wet cells keep the dry surface's emissivity: about 0.93 for asphalt, against about 0.96
  for water. This is a flagged approximation, of order 1 K apparent on the wet half against a
  clear sky, because a prim has one infrared material.
- Curved or meshed surfaces are out of scope. They are WM.3's and AI.2's.
