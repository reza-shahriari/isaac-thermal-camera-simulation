# ADR 0144 — A VDB cloud renders in the path tracer, as a volume box with the grid as its texture

**Status:** Accepted. Supersedes the conclusion of [ADR 0140](0140-openvdb-ships-with-isaac-sim-and-volumes-still-do-not-render.md)
("volumes still do not render"); its writer decision stands.
**Date:** 2026-09-26
Roadmap: AT.13 (reopened and closed), unblocks AT.14.

## Context

ADR 0140 concluded that nothing volumetric renders through a Replicator render product on this
build, on the strength of a control: a cube carrying `OmniVolumeDensity` and no file was "as
invisible as the grid". Re-run on 2026-09-26 with the A6000 available, that control turned out to
be wrong twice over, and the conclusion with it.

1. **The control stage contained the cube.** `--cube` built it into the "without" scene as well,
   so the probe differenced the cube against itself and read zero. The 24 September cube frame,
   opened today, shows a lit foggy block with the sun reflected on its face and its shadow on the
   backdrop. A procedural volume rendered fine; the probe could not see it.
2. **The material's inputs did not exist.** `author_cloud_volume` set `densityMultiplier`,
   `scattering_scale`, `absorption_scale` and `albedo`. `OmniVolumeDensity.mdl` in the Kit kernel
   declares `volume_density_texture`, `volume_albedo`, `volume_density_scale` and
   `directional_bias`. A `UsdShade` input the MDL does not declare is dropped without a warning,
   so the VDB file was never named to the material at all — the field relationship on the
   `UsdVol.Volume` is not what RTX reads.

## What was measured

Eleven path-traced runs, 640×480, 64 spp, each differenced against a control frame *and* a repeat
of the control (the noise floor). The columns that decide:

| stage under test | isVolume | ptvol on | bounces | texture | result |
|---|---|---|---|---|---|
| `UsdVol.Volume` + OpenVDB field (ADR 0140's) | — | no | 3 | — | nothing, at 12 or 200 settle steps |
| same, with the file also on the texture input | — | yes | 3 | file | nothing |
| NanoVDB variant | — | yes | 3 | — | nothing |
| unit cube × 4800, `OmniVolumeDensity`, no file | no | yes | 3 | — | glass block: writes depth on 81 % of the frame, sun highlight on its face |
| same | yes | yes | 3 | — | black, no depth |
| same | yes | yes | 32 | — | invisible (the interior is empty; the black was bounce starvation through two glass faces) |
| unit cube scaled to the grid bounds | yes | yes | 32 | file | whole frame darkens **40×** in linear HDR, backdrop below the cloud included; identical at 0.1× and 0.01× density |
| **mesh box at the grid bounds, no transform** | **yes** | **yes** | **32** | **file** | **the cloud**: lit tops, flat bases, self-shadowing, a shadow on the backdrop; HDR 99.9th percentile 2.12 against the control's 0.92; depth unchanged |

The scaled-cube row is the informative failure: the density texture is sampled in the prim's
**local** frame, so a unit cube scaled by 5200 magnifies the grid 5200-fold and the camera is
inside cloud everywhere. That is why no density scale could change it.

## Decision

1. **`author_cloud_volume` authors the recipe that renders**: a `UsdGeom.Mesh` box with its
   vertices at the grid's world bounds and no xform ops, `primvars:isVolume = true`,
   `OmniVolumeDensity` with the `.vdb` on `volume_density_texture`, the albedo on
   `volume_albedo`, the multiplier on `volume_density_scale`, the phase on `directional_bias`,
   and the material connected through `mdl:surface`, `mdl:displacement` and `mdl:volume`.
   The `UsdVol.Volume` form is gone from the authoring path; it never rendered here.
2. **The render settings are part of the recipe.** `/rtx/pathtracing/ptvol/enabled` must be on,
   and `/rtx/pathtracing/maxBounces` (default 3 on this app) raised — 32 was used; the UI allows
   63. `scripts/probe_cloud_volume.py` carries them as `VOLUME_SETTINGS` and `--pt-bounces`.
   A driver that turns `--cloud-volume` on has to set the same.
3. **The probe's control is a control.** `build_stage(None, …)` builds nothing under test,
   whatever mode was asked for; and the probe captures `HdrColor` with auto-exposure off, because
   a bright volume in one corner pulls the LDR exposure down over the whole frame and the
   difference then reports a darker backdrop where only the cloud changed.
4. **ADR 0140's writer decision stands.** OpenVDB from `omni.volume` is still the writer; only
   its rendering verdict is superseded.

## Consequences

* The visible band can have a cloud with an inside, parallax and a shadow on the scene, rendered
  by the path tracer from the same grid the infrared band marches. AT.14 is unblocked.
* **A volume writes no depth.** `distance_to_camera` behind the cloud is the backdrop's in every
  run. Occlusion of an aircraft by cloud in a band computed from AOVs therefore still comes from
  this project's own march, which is the split AT.14 has to keep.
* The path-traced cloud is path-traced: 64 spp at 640×480 is visibly noisy in the shadowed base,
  and a 1280-px frame at a useful sample count is a per-frame cost the real-time dome bake does
  not have. The dome stays the real-time path; the volume is the path-tracing one.
* The probe's global-fog setting (`globalVolumetricEffects/enabled`) is on in every run and
  shows as a haze with crepuscular rays under the cloud. It is a setting the scene can turn off;
  it is noted so the frame is not read as the cloud's own scattering.
* `scripts/render_quad_outbound.py --cloud-volume` still writes a NanoVDB and authors through
  this function; switching it to `write_openvdb` and setting the render settings is the next
  step, with a frame to prove it.

## Alternatives rejected

**Keep the `UsdVol.Volume` prim and search for the setting that makes it render.** Five variants
of it produced nothing, including one with the file also on the texture input. The Composer
recipe renders; the prim type is not worth the search until NVIDIA says it should work.

**Report the LDR difference as before.** It would have called the scaled-cube frame "loaded"
(99 % of pixels changed) and hidden that the change was exposure. The HDR means are what showed
the darkening reached the backdrop.
