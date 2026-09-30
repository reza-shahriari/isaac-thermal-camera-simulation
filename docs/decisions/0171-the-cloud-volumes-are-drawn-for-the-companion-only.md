# 0171 — The cloud tier is one switch; the cloud volumes are drawn for the companion only

Date: 2026-09-30

**Status:** Accepted (2026-09-30, AT.31). Implements ADR 0169; builds on ADR 0144 and ADR 0162.

## Context

ADR 0169 settled what the two cloud tiers are: path-traced occludes a target behind cloud in both
bands, real-time in neither, and the cloud itself is in both bands in both tiers. Before AT.31 a
headless render could be neither. weather-fx's `render_path = "auto"` follows the renderer's mode,
and a headless driver captures its colour frame under the path tracer in every case, because
PathTracing is the only mode that lights a colour AOV on this build. So `auto` picked "volume",
the dome dropped its cloud, and no volume effect was attached. The visible frame had **no cloud at
all**, while the infrared band marched and occluded on the full field.

Four things surfaced when the path-traced tier was first rendered:

1. **The volumes reach the G-buffer.** weather-fx draws each cloud tile as a mesh box marked
   `primvars:isVolume`. The instance AOV reports it, and the frame stopped with `rendered prims have
   no thermal node: ['/WeatherFX/Clouds/Tile_0_0', ...]`. A drone *inside* the layer would read as
   the box around it.
2. **The volume settings were never applied.** `CloudVolumeEffect` changes a path-tracer setting
   only if it already exists. That holds inside the Kit app it was written for, and silently does
   nothing in a headless app where `/rtx/pathtracing/ptvol/*` is unset.
3. **The real-time dome was cloudless.** Under the state's default `wall` clock the sky installs
   a quick cloudless first bake and finishes the real one in a worker. Only the extension's
   `update` loop collects that worker, and a headless driver never runs it.
4. **The demonstration preset has no cloud seed**, so each run draws a different field. Seed 0
   put no cloud in the frame at all.

## Decision

* **One switch.** `author_weather_fx_sky(tier=...)` sets `clouds.render_path` explicitly ("volume"
  or "dome", never `auto`). For the path-traced tier it attaches `CloudVolumeEffect` beside the sky.
  The returned `WeatherFxSky` answers `occludes` and `companion_only_paths` for the camera, so the
  infrared half is read off the sky that was built rather than decided a second time.
  `render_phantom4.py --cloud-tier path_traced|real_time`; the default is path-traced.
* **The volumes are companion-only.** `IrCamera(companion_only_prim_paths=...)` hides them for
  every G-buffer render and shows them for one extra companion render per frame, in the session
  layer. The infrared band already has this cloud, as the march on the same field. To the
  G-buffer, the box is only a box.
* **The camera's occlusion follows the tier.** `IrCamera(cloud_occlusion=False)` skips the march
  to the hit (the second pass over the frame), and sky pixels are marched either way.
* **Headless weather-fx is made deterministic.** Both tiers set `general.time_source = "manual"`,
  so the dome and the volumes are built on the calling thread. The path-traced tier also enables
  ADR 0144's volume extensions and sets `VOLUME_SETTINGS` unconditionally
  (`prepare_path_traced_volumes`).

## Consequences

* Measured on 12 frames of the stratus-fractus preset with its cloud seed pinned (621227):
  * **Infrared.** Target transmittance was 0.33–0.65 in the path-traced tier, and the target reads
    0.6–1.5 K cooler than in the real-time tier. The marched sky (`temperature_k`) is identical
    between tiers on every sky pixel. The output frame differs only within 10 px of the target
    on the first frame, and along its motion-smear trail after that.
  * **Visible, path-traced.** The companion is lit cloud (frame mean 200 of 255), and the drone's
    pixels read 150–209: it is inside the cloud.
  * **Visible, real-time.** The dome carries the 60 % overcast (frame mean 93–98), and the drone
    reads a dark 68–88 in front of it.
  * **Cost.** The path-traced tier costs 33 s a frame against 23 s, which is the extra companion
    render.
* The real-time tier's dome now carries the cloud: `cloud_cover_in_dome` is 0.60, where the
  `wall`-clock run baked 0.0. Before this, no headless render carried any visible cloud.
* The 1-D texture warning (`volume_density_texture ... (Type: '1D')`) still prints on the first
  bind of each tile, and the volumes render anyway.
* `configs/weather_fx/stratus_fractus_over_the_camera.json` still has no seed. A demonstration that
  must show a cloud in front of the drone pins one.
