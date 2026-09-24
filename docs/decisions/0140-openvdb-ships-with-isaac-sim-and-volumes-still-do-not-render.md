# ADR 0140 — OpenVDB ships with Isaac Sim, and volumes still do not render

**Status:** Accepted. The cloud deck is written with the OpenVDB that ships inside Isaac Sim
rather than with a hand-stamped NanoVDB. **Volume rendering still produces nothing** through
Replicator's render-product path on this build — and now it is known that this is true of *any*
volume, not of our file. ADR 0127's diagnosis is superseded.
**Date:** 2026-09-24
Roadmap: AT.13 (§5.3 a)

## Context

ADR 0127 put the cloud deck in the stage as a `UsdVol.Volume` so the path tracer could integrate
the same field the infrared band marches. It never rendered. Two causes were found and fixed — the
bundled Warp stamps NanoVDB **32.8** where the renderer's IndeX plugin wants **32.7**, and
`warp.Volume.save_to_nvdb` leaves `voxelCount`, `nodeCount` and `tileCount` at zero — and a third
was not, so `--cloud-volume` has been off by default since.

All of that rested on one sentence: *"There is no OpenVDB writer in this environment — no
`pyopenvdb` on any interpreter here."*

**That sentence is wrong.** Isaac Sim's `omni.volume` extension ships `openvdb.cpython-312-*.so`
(OpenVDB 12, file format 224), `nanovdb.cpython-312-*.so`, `libopenvdb.so.12` and a `carb.volume`
binding with `create_from_dense` and `save_volume`. They are simply not importable until someone
does what Kit would otherwise have done: preload the shared libraries the extension module names
but carries no RPATH for, and put the directory on `sys.path`. Both are three lines.

So the deck was being voxelised through a CUDA-only Warp path, written to a format by a library
that does not own it, patched byte-wise in two places, and refused — when the format's own writer
was on disk the whole time.

## Decision

**`irsim_isaac.env.ensure_openvdb_on_path` makes OpenVDB importable**, mirroring
`ensure_warp_on_path` (ADR 0014): `$IRSIM_OPENVDB_PATH` overrides, otherwise the extension cache
is globbed for `omni.volume`. Preloading opens shared libraries and imports nothing, so it starts
no Kit application and is safe on a machine with no GPU.

**`irsim_isaac.cloud_volume.write_openvdb` is the writer.** A named grid, fog-volume class, a
linear transform written as a matrix rather than composed from operations, `addStatsMetadata` for
the counts the renderer complained were zero, and a write tolerance so the empty air is *absent*
rather than stored as 1e-30. It needs no GPU and no Kit — where the NanoVDB path needed a CUDA
device to write a file — so it runs on the unit gate and has five tests there, including one that
checks a voxel's world position against the deck's own geometry (a volume with the wrong transform
renders beautifully, underground, and nothing in the frame says so) and one that samples the grid
against `CloudDeck.density_at`, which is ADR 0127's one-field-two-consumers rule made testable.

`write_nanovdb` stays. Its two fixes are correct and a `.nvdb` is still what Warp reads back
fastest. It is no longer what a renderer is handed.

**`scripts/probe_cloud_volume.py` is AT.13's probe.** A backdrop, a cloud in front of it, a camera
looking through, and **the scene rendered twice — once with the volume and once without**. A volume
that failed to load and a volume that is invisible to an annotator produce the same array as no
volume at all, and nothing in a single frame separates either from a cloud that is merely thin.

## What the renders said

Every combination, 640×480, 24 spp, backdrop confirmed 100 % in frame, cloud prim valid with a
non-empty world bound:

| `/rtx/rendermode` | volume | LdrColor changed | noise floor | verdict |
|---|---|---|---|---|
| `PathTracing` | OpenVDB grid | 0.000 | 0.000 | **nothing** |
| `PathTracing` | cube, procedural MDL, no file | 0.000 | 0.000 | **nothing** |
| `RaytracedLighting` | OpenVDB grid | 0.131 | 0.081 | below noise |
| `RaytracedLighting` | cube, procedural MDL, no file | 0.492 | 0.472 | below noise |

In `PathTracing` the render is deterministic — two captures of an identical stage are *byte*
identical — and the volume changes **nothing at all**. That rules out "it rendered faintly".

**The cube is the finding.** It carries `OmniVolumeDensity` at a constant density and no VDB file
of any kind, and it is as invisible as the grid is. So this was never about NanoVDB versions, file
metadata, the writer, or our cloud: **no volume renders here**, whatever it is made of.

Everything else checks out, which is what makes the result usable as a bug report:

* `OmniVolumeDensity.mdl` **resolves**, to
  `packman-repo/chk/kit-kernel/110.3.0*/mdl/core/Volume/OmniVolumeDensity.mdl`.
* `omni.volume` and `omni.index` are already enabled; `carb.volume.plugin` loads; `libusd_usdVol`
  and `libusd_usdVolImaging` load. (`omni.hydra.index` refuses to enable.)
* `/rtx/rendermode` reads back as set, and `/rtx/pathtracing/volumesAOV`,
  `/rtx/pathtracing/maxVolumeBounces` and `/rtx/raytracing/globalVolumetricEffects/enabled` are on.
* The volume prim has an authored `extent` — **added here**, and necessary: `UsdVol.Volume` is
  boundable with no points to fall back on, so without it there is no bounding box and the prim is
  culled. That is a second silent failure, and it was in the path before this.

## Consequences

* The visible band's cloud stays the dome baking the deck as a sheet. Nothing regressed; what
  changed is that the reason is now known and is not ours.
* **AT.14's premise is not yet testable.** Whether a volume reaches `distance_to_camera` — which
  decides whether cloud can occlude an aircraft in a band computed from AOVs — cannot be answered
  while no volume renders. `distance_to_camera` was unchanged in all four runs, but so was
  everything else.
* The question to put to NVIDIA is now sharp: *on Kit 110.3 / Isaac Sim 6.1, does a `UsdVol.Volume`
  or a procedural MDL volume render through an `omni.replicator.core` render product, in any
  rendermode?* It is reproducible in one command.
* Next worth trying, in order: the **interactive viewport's** own capture rather than a Replicator
  render product; `omni.hydra.index` (it refused to enable, and the RTX Scientific path is the one
  the docs put VDB under); and an app `.kit` that enables volume support at boot rather than a
  runtime `set_extension_enabled_immediate`.

## Alternatives rejected

**Keep chasing the NanoVDB header.** The cube settles it: a volume with no file at all behaves
identically, so no amount of work on the file was going to change the outcome. This is the value
of a control, and ADR 0127 did not have one.

**Call it "volumes are Path Tracing only" and move on.** The documentation does say that, and the
measurement says the path-traced frames are byte-identical with and without. Repeating the
documentation over the top of a measurement that contradicts it is how a wrong belief survives.

**Detect the volume by rendering once.** The first version of this probe did, with a threshold of
1e-3 of full scale, and reported that 43 % of the frame changed between two real-time captures of
a stage with no volume in it — denoiser churn, read as a cloud. The noise floor is now measured by
rendering the empty scene a second time, and a claim has to beat it by 3x.
