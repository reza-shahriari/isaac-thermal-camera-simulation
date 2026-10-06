# Issue drafts awaiting review

<!-- Drafted by Claude during work; filed by the maintainer's account once approved. -->

## The visible companion does not hide an object behind cloud in headless captures
Labels: bug, isaac · Milestone: v0.1.0 · Found in: WX.26 (ADR 0191)

weather-fx's per-pixel cloud layer hides scene objects behind cloud with a veil quad fed by the
scene's depth (`distance_to_camera` annotator). In the Isaac viewport it works (WX.27). Under the
headless `render_phantom4.py --cloud-tier pixel` driver the layer's depth read never yields a
usable buffer (`cloud_layer.veil` is false in every run's `summary.json`), so the companion draws
the drone in front of a cloud the infrared march hides it behind. One run also crashed natively
inside the annotator's device-to-host copy during the camera's `open()`.

## Why it matters
The pixel tier promises occlusion in both bands (ADR 0169, 0190). In the six genus clips the
infrared hides the drone behind cloud and the visible does not, so the pair disagrees about the
target exactly where a detector's training label would be drawn. The drone's silhouette is also
part of the measured band disagreement (it is inside the 0.05–0.10 p95 of the twelve cumulus and
congestus frames outside the bar).

## Where
- `third_party/isaac-weather-fx/.../backends/viewport/clouds_pixel.py`, `_scene_depth`
- `src/irsim_isaac/weather_fx_stage.py`, `author_weather_fx_sky` (the redraw subscription)
- docs/physics-model.md §7.5 (both tiers' occlusion rule)
- Found while: WX.26, ADR 0191, `outputs/phantom4_cloudscape/*/summary.json` (`cloud_layer.veil`)

## Done when
- [ ] a headless pixel-tier capture reports `cloud_layer.veil` true when cloud lies in front of
      the target, and the companion frame shows the cloud over the drone
- [ ] `scripts/cloud_band_agreement.py` on such a run shows no error concentrated on the drone's
      silhouette (the `cloud_transmittance_vis` plane reads 1 on the drone's pixels when no cloud
      is in front of it, as `cloud_transmittance` does)
- [ ] the depth read cannot run inside the camera's `open()` (the crash seen once)
