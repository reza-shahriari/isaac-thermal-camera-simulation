# ADR 0186 — The per-pixel cloud reaches the viewport as the emission of a quad that rides with the camera

**Status:** Accepted
**Date:** 2026-10-03
Closes the open risk of ADR 0185. Roadmap: `WX.24`. The change is in `isaac-weather-fx` (`f25c3a8`,
`backends/viewport/clouds_pixel.py`); this repository takes it as a submodule bump. The owner's
requirement, stated the same day: nothing is rendered outside Isaac Sim.

## Context

ADR 0185 made the cloud a function marched per camera pixel on the GPU, and proved the look with
a standalone tool. Isaac Sim's RTX Real-Time renderer has no hook for a custom volumetric shader,
so how that march reaches the live viewport was unproved.

## Options considered

1. **Composite after the renderer**, on the camera's output through its depth. It works for
   recorded frames, not for the viewport the owner looks at, and every consumer of a camera would
   need the composite step.
2. **Re-bake the dome every frame at a higher resolution.** A lat-long texture sharp enough for a
   60° camera is tens of megapixels, marched every frame for every direction, most of them off
   screen.
3. **A quad at the far end of the camera's frustum, emitting the march's frame (chosen).** The
   march's output stays on the GPU and becomes a dynamic texture
   (`omni.ui.DynamicTextureProvider.set_bytes_data_from_gpu`); the quad's material emits it. The
   renderer draws scene geometry in front of the quad by itself, and an emissive surface is the
   same thing to RTX Real-Time and to the path tracer.

## Decision

Option 3, measured before it was built (Isaac Sim 6, Kit 110, RTX A6000, headless):

* **The texture route works in both render modes** and updates every frame; the hand-over takes
  0.08 ms for 1280 × 720 RGBA32F.
* **Emission calibrates exactly against the dome**: OmniPBR at `emissive_intensity = π · I` renders
  as bright as a dome light at intensity `I` (0.00088 against 0.00028 × π in the HDR buffer, both
  modes). The layer therefore carries the same radiance, exposure and white balance as the dome it
  stands in front of.

The clear sky and the air in front of the cloud are tabulated once per quarter-degree of sun
movement (`core/layer_tables.py`) and sampled in the composite kernel, so the whole frame is made
on the GPU. The dome light stays, baked clear of cloud, to light the scene; the sun's intensity
follows the cloudscape's optical depth toward it. The path is opt-in, `clouds.render_path =
"pixel"`.

## Consequences

* **Measured** (`outputs/cloud_pixel_isaac/report.json`): 44–48 fps in RTX Real-Time at 1280 × 720
  with the camera turning every frame; the cloud layer is 15 ms of each 21–23 ms update. Four
  headings 90° apart render one sky at one sharpness, and at midday the real-time and path-traced
  frames show the same clouds pixel for pixel.
* **One camera.** The quad is aligned to the active viewport's camera (or `general.follow_prim`
  when that is a camera). A second camera in the same stage sees the first one's quad from the
  side. Several render products need a quad each and per-camera visibility; not built.
* **The dome is clear**, so reflections, ambient light and soft shadows do not see the clouds, and
  the ground has no cloud shadows beyond the sun's dimming at the camera (`WX.10`).
* **With the sun in frame the two modes expose differently**: real-time auto-exposes on the
  histogram, the path tracer does not. The clouds agree; the tonemapping does not.
* **Geometry beyond 0.9 of the camera's far clip is hidden** behind the quad.
* **The GPU must be CUDA device 0 and the renderer's own.** On a machine with two cards that needs
  `CUDA_DEVICE_ORDER=PCI_BUS_ID` alongside Kit's `active_gpu`.
* **The default stays `auto`** until the infrared march reads the cloudscape (`WX.26`); switching it
  now would put different clouds in the two bands.

## Revisit when

* More than one camera must show the layer in one stage.
* Kit exposes a per-pixel volumetric or post-process hook in real time; the march stays and the
  quad goes.
