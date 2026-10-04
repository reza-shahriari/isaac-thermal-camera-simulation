# ADR 0185 — The cloud is a function marched per camera pixel on the GPU, not a picture on a dome

**Status:** Accepted
**Date:** 2026-10-03
Reopens the delivery of §7.5's cloud after the owner's review of 2026-10-03. Roadmap: the new rows
`WX.23`–`WX.27`, which replace the dome-resolution step `WX.21`. The prototype is in `isaac-weather-fx`
(`core/cloudscape.py`, `gpu/cloud_march.py`, `tools/render_cloudscape.py`); this repository takes
it as a submodule bump.

## Context

Four steps (`WX.2`–`WX.6`) corrected the cloud's edge resolution, lighting, pixel integration, air
in front of it and size statistics. Rendered in Isaac Sim at the shipped settings
(`outputs/cloud_audit/`), the sky was still not good enough in either mode, and the owner said so.

* **Real-time:** the cloud is marched on the CPU and painted into a 1024-row dome texture. The
  camera has about 4 pixels per dome texel and about 15 per cloud texel. The sky is blurred,
  texels show at sunset, the horizon smears, and a camera that turns 90° sees a different
  quality of sky, because a dome texture is not equally sharp in every direction. It also has
  no depth: a camera that moves sees a painting.
* **Path-traced:** the function is baked to a 60 m voxel grid. The clouds are popcorn and
  pancakes, their bases olive green from the ground's bounce, and there is no air between the
  camera and a cloud 30 km away, so the field looks like a miniature.
* **The two modes disagree** by a third in brightness and, at sunset, in colour.

The owner also fixed the structural requirement: whatever draws the cloud must be convertible to
the infrared, which rules out any picture-based sky (HDRI, painted dome, PNG clouds). Unreal
Engine's volumetric clouds were named as the look to reach.

## Options considered

1. **Keep tuning the dome.** The texel limit is a design limit; no tuning makes a dome sharp
   from every heading, and it never gains depth.
2. **Path tracer only, with finer voxels and air.** RTX renders volumes well, but the
   real-time viewport cannot, and the owner wants both. Voxels fine enough for a 15 m edge
   over 30 km do not fit in memory.
3. **A density function marched per camera pixel on the GPU (chosen).** This is what Unreal
   Engine and Horizon Zero Dawn (Schneider 2015, 2017) do: a 2-D weather map places the clouds,
   a per-type height profile shapes each one, and two 3-D noises erode the edge. Every pixel's
   ray evaluates the function at its own resolution, so the cloud is as sharp as the screen in
   every direction, the camera can move through it, and the same function read on the CPU with
   emission instead of scattering is the infrared cloud (§7.5's one-field rule).

## Decision

Option 3. The cloud layer is `weather_fx.core.cloudscape.Cloudscape`, a function of three
tiling textures and a dozen constants, with a numpy reference evaluation. The GPU kernel
`weather_fx.gpu.cloud_march` repeats that arithmetic line for line, and a test holds the two to
within the texture unit's 9-bit interpolation (max 0.05 density over 20 000 points, mean under
10⁻³).

The march lights each step by:

* the sun scattered once, `p(θ) e^{−τ_sun}`;
* the sun scattered many times, the δ-Eddington two-stream field along the sun's chord
  (ADR 0180), falling off with the optical depth from the lit surface so a lobe's own bumps
  shade its flank;
* the sky and the ground, mixed by the two-stream diffuse transmittance above and below the
  point, with weights summing to one (the furnace: a white cloud under a uniform sky vanishes
  to 2 %).

The step is coarse through clear air and a third of that inside and just past a cloud. Steps
integrate energy-conservingly, the result is premultiplied, and the sky model composes it through
its own aerial perspective (ADR 0182).

## Consequences

* **Measured on the A6000** at 1280 × 720: 6–13 ms per frame for the march. The clear sky and
  the air in front of the cloud are still computed on the CPU (about a second a frame) and are the
  next thing to move to the GPU.
* **The rotation problem is gone by construction**: four headings 90° apart render the same sky
  at the same sharpness (`captures/cloudscape/still4/sheet.png` upstream).
* **The look is not finished.** The lobes are smoother than a cumulus; the detail erosion and
  its frequencies are a tuning job on the new recipe, and the owner's eye is the test.
* **The live viewport is the open risk.** Isaac's real-time renderer has no hook for a custom
  volumetric shader. The cloud layer composites over a camera's output through depth for
  recorded frames; showing it live in the viewport needs a route that is not yet proved
  (`WX.24`).
* **The procedural `CloudField` and the dome stay** for lighting the scene (sky light, cloud
  shadows on the ground) and for the path tracer's volumes until `WX.25` exports the new function
  there. The infrared march still reads `CloudField`; `WX.26` moves it to the function.
* **`WX.21` (a finer dome) is withdrawn.** The dome is no longer what the camera looks at.

## Revisit when

* A renderer gives a per-pixel volumetric hook in real time. The function stays; only the march
  moves.
* Real simulated clouds (LES) are wanted as the shape source. They enter as a 3-D density
  texture in the same march.
