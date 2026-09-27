# ADR 0154 — A frame viewer reads the scene truth beside the camera, through a swappable source

**Status:** Accepted
**Date:** 2026-09-27

## Context

When a rendered frame looks wrong, the question is almost always about a point: *what is this
pixel?* Which part is it, what temperature did the solver give it, how far away is it, and what
did the camera report? Until now a frame's sidecar (`irsim.io.dataset`) carried only what the
camera said: `radiance`, `apparent_t`, `dn16`, `display8`. The truth behind them (surface
temperature, instance id, material id, range) lived in the G-buffer and was discarded after every
frame. Answering "why is the motor 3 K cold here" meant writing a one-off script against a live
Kit session.

There were three constraints:

* **No GUI toolkit in the environment.** The project interpreter is Isaac Sim's bundled Python.
  Installing Qt into it risks the Kit install, for a debugging tool.
* **The data layout will change.** Planes are `.npy` today. A Warp/CUDA path may write packed
  buffers, or serve planes straight from device memory. The viewer must not have to be rewritten
  when that happens.
* **A wrong pixel is worse than none.** A click that reads a neighbouring pixel's temperature
  looks exactly like a correct one.

## Options considered

1. **A Qt / Tk desktop app.** Needs a toolkit the Isaac interpreter does not have (Tk is absent
   from the source build). Rejected on the environment constraint.
2. **A Jupyter / matplotlib notebook.** Interactive clicking needs widget extensions, which are
   more dependencies. It is also awkward for stepping through a clip with markers held in place.
3. **A local web page served by the standard library.** `http.server` plus NumPy, one HTML file
   with no framework, opened in the browser the user already has. Nothing to install.

For the truth planes, there were two choices of what one pixel's value should be:

1. **The mean of the `k x k` supersamples.** Meaningless for ids: the mean of two part ids is a
   third part that is not in the scene (ADR 0014).
2. **The sample nearest the pixel centre, for every plane.** One surface point, described
   consistently by all five planes.

## Decision

* **`src/irsim_viewer/`**, a package separate from the physics core. It imports the core (for the
  PNG codec), never the reverse, and never the engine. It is started with `make viewer RUN=...`.
* **The data layer is the swap point** (`irsim_viewer.source`). The server and page talk only to a
  `FrameSource` protocol: `frames`, `metadata`, `planes`, `read_plane`. A new file container is one
  `register_plane_reader(suffix, fn)`. A new layout is one class plus
  `register_source_factory`. A test drives the server from an in-memory source with no files, to
  keep that promise honest.
* **Truth planes on the detector grid** (`irsim.io.truth`, `IrCamera.truth()`): `temperature_k`,
  `distance_m` (NaN for sky), `material_id`, `part_id` and `node_id`. Each is the **centre sample**
  of its supersample block, and ids are named in a new sidecar field, `legends`. All five render
  drivers that use `FrameWriter` write them.
* **Pixel mapping is explicit.** When a plane's grid differs from the image clicked, a grid that
  nests is mapped exactly, with a note. A grid that does not nest is read with a **warning** shown
  on the marker.

## Consequences

* A render now answers "what is this pixel" from disk. The viewer joins a pixel's `node_id` name
  with the frame's `node_temperatures_k`, so the per-pixel mesh-cell temperature and its node's
  lumped temperature sit side by side.
* **The error the centre sample introduces:** on a silhouette the camera's pixel mixes surface and
  sky, while the truth planes report only the surface under the centre. That is intended, since it
  is exactly the comparison a debugger needs, but it means `temperature_k` and `apparent_t` are not
  expected to agree at edges.
* Disk cost is about 4.6 MB more per frame at 640x512: two float32 planes, plus three uint16
  planes that compress well as PNG. `plane_stride` still thins it.
* `part_id` is named by the prim's **leaf** name. Two prims with the same leaf name under
  different parents share a label.

## Revisit when

* A render path writes planes somewhere other than per-frame files (a Warp buffer dump, a
  database). Add a `FrameSource`; the rest should not change.
* Someone needs the mean over the pixel instead of the centre truth. Add a second plane rather
  than changing this one.
