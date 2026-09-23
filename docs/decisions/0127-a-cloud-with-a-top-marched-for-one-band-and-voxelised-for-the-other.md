# ADR 0127 — A cloud with a top, marched for one band and voxelised for the other

**Status:** Accepted. Cloud becomes a **deck** — a horizontal map of column depth at the lifting
condensation level, each column given a top and an analytic vertical profile. The infrared band
ray-marches it; the visible band renders the same function as a NanoVDB volume in the stage.
**Date:** 2026-09-23
Roadmap: AT.12 (§5.3 a)

## Context

ADR 0126 gave a cloud a real optical depth and a real distance, and the rendered frame was still a
flat level: the cloud **core** spanned 1.25 K across a 640×512 frame. That is not a bug in ADR
0126, it is what a plane-parallel sheet *is*. Every ray that enters an infinite sheet stays in it,
so an optically thick column reads its base temperature and nothing else.

Looking straight up that is correct, and it is worth saying why: a cumulus base is flat because it
is the lifting condensation level, one height for the whole air mass. Looking at **20 degrees**,
which is where every clip in this lane aims, it is wrong. A real cumulus field is a field of
towers; the camera sees their flanks, their tops and the gaps between them, and a ray climbing
1.2 km at that elevation travels **3.3 km horizontally**, crossing several of them.

Separately, the visible companion's cloud was painted into the environment-dome texture (ADR
0073/0076). A dome light sits at infinity, so that cloud had no distance, no thickness and no
inside: a camera could not move relative to it, and the owner's request was explicitly for cloud
that is *not just a visual effect*.

## Decision

**One deck, two bands.** `irsim.atmosphere.cloud_deck.CloudDeck` holds a square horizontal map of
column depth `d ∈ [0, 1]` at the LCL, a geometric thickness, and the authored visible optical
depth. `CloudDeck.density_at` is the single definition of the cloud's shape; the infrared march
integrates it along rays and the voxeliser samples it on a grid, so the volume the path tracer
renders is the field the radiometry integrates rather than a second cloud that resembles it.

**The vertical profile is a normalised parabola,** `w(u) = 6u(1 − u)` on `u = h/H`, with
`H = thickness × d`. Its integral over the column is exactly `H`, so a **vertical ray accumulates
exactly `optical_depth × d`** — which is ADR 0126's model, to the bit. The third dimension is
therefore provably a generalisation: looking up, nothing moved. (Real cumulus liquid water rises
along a near-adiabat from the base and falls at the top, so a symmetric parabola is a stand-in;
the one anchored quantity, the column optical depth, is independent of the shape by construction.)

**The field is synthesised on the deck, in metres.** The first version projected the hemispherical
`SkyFixedCloud` down onto the deck and extruded each column. That is wrong and the preview frame
said so immediately: an angular field's features subtend a fixed number of *degrees*, so the
projection makes columns a few hundred metres wide and then extrudes each 1.2 km straight up — a
field of tall thin fins, along which an oblique ray runs for kilometres instead of crossing a
cloud. The frame came out as vertical streaks.

**`beta` is read as a transect slope, not a radial one.** `generate_cloud_field` applies the
authored `beta` as the *radial* exponent of a 2-D power spectrum, and on a 2-D field the variance
per octave goes as `f^(2−β)` — so `beta: 1.8` puts more variance at the smallest scale the grid
has than at the largest. Measured on the deck's own grid: the autocorrelation length at β = 1.8 is
**125 m**, which is texture, not a cumulus; at β = 2.8 it is **1250 m**, which is one. Published
cloud slopes near −5/3 are **transect** slopes, one less than the radial exponent of an isotropic
field, so the deck synthesises at `beta + 1`.

**Not changed here:** the hemispherical `SkyFixedCloud` still applies `beta` radially. Changing it
would move the cloud in every scene shipped since MS.3, and the question is about the preset's
authored number rather than about this module. It is recorded in `docs/spec-issues.md`.

**The step count is derived, not authored.** An oblique ray crosses the depth map horizontally, so
a fixed 48 steps put its samples 68 m apart across a 25 m grid, and the undersampling drew
horizontal bands along every cloud edge — visible at a glance in the preview. `adequate_steps`
sizes the march from `max(thickness/tan θ, thickness)` at two samples per cell, clamped to
[32, 256]. 256 is where the banding stopped being visible, at 4.2 s per 640×512 frame against
0.6 s at 48.

**The volume does not load on this build, and that is recorded rather than worked around.**
Isaac Sim 6.1's RTX renderer routes VDB assets through its IndeX plugin, and that plugin refuses
the grid. Two causes were found and fixed and a third was not:

1. Warp 1.16 stamps NanoVDB **32.8** into the grid header; the plugin accepts **32.7** and said
   so in the log. `INDEX_NANOVDB_VERSION` re-stamps it, and the log's version line then reports
   32.7 on both sides.
2. `warp.Volume.save_to_nvdb` leaves `voxelCount`, `nodeCount` and `tileCount` at **zero** in the
   file metadata — it fills only what it needs to read its own files back. `_fill_file_metadata`
   copies them out of the grid's own `TreeData`. A consumer sizing its storage from those fields
   would otherwise get nothing, which matches the reported "failed to generate VDB subset grid
   storage".
3. It still fails. Whatever remains is inside the plugin.

So `--cloud-volume` is **off by default** and `--cloud-deck` is the flag that works: the visible
dome bakes the deck's own column depth where each ray crosses the base — the deck seen as a sheet,
which puts the cloud in the right *places* at a cost the dome can afford, and keeps ADR 0076's
rule that the two bands cannot disagree. The volume code is kept, because both fixes are correct,
because the failure is loud (the companion frame comes back with no cloud in it), and because a
written `.nvdb` is the artefact `AT.13`'s AOV probe needs.

**NanoVDB, not OpenVDB.** There is no OpenVDB writer in this environment — no `pyopenvdb` on any
interpreter here — but Isaac Sim ships `omni.warp.core`, and `warp.Volume` round-trips a dense
NumPy array to `.nvdb`, which Omniverse documents as loadable beside `.vdb`. Warp's allocator
leaves the grid name empty and a `UsdVol.Volume` field relationship binds by name, so the name is
written into the grid header before the file is saved and the now-stale checksum is set to
NanoVDB's own "not computed" sentinel. Verified by round-trip: a 244×25×244 grid at 50 m comes
back from `Volume.load_from_nvdb` as `GridInfo(name='density', type_str='float', translation=…)`
with the right voxel size and world bounding box.

**The grid carries visible extinction per metre**, which is what `density_at` returns, so the
material's density multiplier is 1 and the optical depth a renderer integrates is the one the
environment preset authored. The dome's cloud is switched **off** when the volume carries it, or
the frame would show both, in the same places, which reads as haze rather than as a mistake.

## Consequences

Measured on the Phantom clip's own frame geometry, same scene, same seed, sheet against deck:

| | sheet (ADR 0126) | deck (this) |
|---|---|---|
| apparent T across the cloud (p1–p99) | 1.25 K | **19.7 K** |
| emission height above base | 0 m, always | **12 – 834 m** |
| largest single histogram bin | 55.4 % of the frame | **39.6 %** |
| cost per 640×512 frame | ~0 | 0.6 s at 48 steps, 4.2 s at 256 |
| cost per **rendered** frame, 300-frame clip | 7 s | 11 s |

The cloud now has an inside: a ray entering a tower low leaves it through the far flank, and the
level it emits from is where it ran out of cloud, not where it started.

**Known limits, stated rather than discovered later.**

* **The emission is linearised.** The march reports one `e^{−τ}`-weighted mean height and the
  radiance is evaluated there, which ignores Planck's curvature across the column. Measured
  against a per-step Planck integration: **64 mK at the worst pixel, 46 mK at the 99th
  percentile** — about one NETD on roughly one cloudy pixel in a hundred, landing on rays that
  graze a tower top. Fixing it means a band lookup inside the march, ~0.25 s per frame with a
  precomputed height/radiance table. Recorded, not done.
* **The deck has an edge.** Its footprint stops at `base / tan(10°)`; a ray below that leaves it
  and reads clear sky. Wrong in the same direction as the horizon itself, which is why these
  drivers keep the horizon out of frame.
* **No advection, no shadowing on the aircraft, no cloud–cloud shadowing.** The deck is static
  over a clip, and the volume is lit by the path tracer but casts nothing onto the target in the
  infrared path, which reads geometry from AOVs.
* **The volume does not load at all on this build** (see the Decision), so the visible band's
  cloud is the dome baking the deck as a sheet rather than participating geometry a camera
  can enter. The infrared half is unaffected: it marches the deck directly.
* **Whether the volume reaches the AOVs at all is untested** — the infrared band is computed from
  AOVs and a volume is not a surface, so cloud in front of the aircraft does not occlude it in
  LWIR. That is `AT.13` (the probe) and `AT.14` (the fix).

## Alternatives rejected

**Keep the plane-parallel sheet and add structure some other way.** There is no other way that is
honest: within a plane-parallel model an optically thick core *is* flat, and manufacturing
variation inside it would be drawing texture, not physics.

**Ray-march inside the renderer instead (AT.14 now).** The right long-term answer, and it is
blocked on a question nobody here has measured — whether Replicator exposes a volume to any AOV on
this build. `AT.13` is that probe, one cube and a dump. Shipping the engine-free march first also
leaves an oracle the in-engine path can be checked against, which is this project's usual shape.

**Marching every supersampled ray.** The camera's AOVs are rendered at four times native so
that *geometry* edges antialias, which is sixteen times the rays — and measured, that took the
render from 7 s to **107 s** a frame. The cloud behind the geometry has no edges at that scale, so
the march runs once per native pixel and is interpolated back (`resample_bilinear`), at 11 s a
frame. The cost of that approximation is measured against an unfiltered full march on a patch at
the real supersampled pitch: **median 0.05 K, 99th percentile under 1 K** on a field spanning 20 K,
with the worst pixels on cloud edges where the box filter over the same sixteen samples would have
smoothed anyway.

**A finer depth map instead of more march steps.** Backwards: the banding came from sampling the
map too coarsely *along the ray*, so a finer map makes it worse for the same step count.
