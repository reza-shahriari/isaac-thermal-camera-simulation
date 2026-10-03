# Unreal Engine 5 as irsim's renderer — survey, 2026-10-01

**Can Unreal Engine replace Isaac Sim as the engine that renders irsim's frames, without losing the
float32 radiometry and the per-point surface temperature, and what would it take?**

Produced for the owner's request of 2026-10-01: *"create another plan file, a roadmap to add unreal
engine instead of having the isaacsim as render engine"*. The plan built on it is
[`unreal/PLAN.md`](../../unreal/PLAN.md). This is a **snapshot of what was readable on 2026-10-01**,
not a maintained document. It has three parts: an audit of this repository's engine boundary, an audit
of the owner's existing Unreal work on this machine, and a web survey of Epic's documentation, release
notes, engine source and prior art.

Epic's engine source is cited by file at tag `5.8.3-release` on `github.com/EpicGames/UnrealEngine`.
Those links need a GitHub account linked to an Epic account. A claim marked **UNVERIFIED** is inference
or rests on a weak source, and the plan treats it as something to measure.

---

## 1. The short version

1. **The port is smaller than "a new renderer" sounds, because irsim's renderer already carries no
   temperature.** On Isaac every colour AOV turned out to be float16 and exposure-scaled, so ADR 0014
   moved temperature out of the renderer entirely. The engine supplies ids and geometry; the host looks
   up and computes temperature per prim, per patch or per mesh cell (ADR 0060, 0087, 0110, 0111). An
   Unreal glue that delivers the same raw planes reuses every one of those tiers unchanged.
2. **Unreal can deliver float32 planes, but only on specific paths, and the obvious ones quietly drop
   to fp16.** These drop to fp16: `SCS_FinalColorHDR`, post-process materials, default SceneColor,
   GBuffer normals (10-bit) and Cosys-AirSim's float images. These keep float32: a dedicated
   SceneCapture writing unlit data materials into `RTF_RGBA32f` with `r.SceneColorFormat=5`, a
   plugin's own RDG compute writing `PF_R32_FLOAT`, and Movie Render Graph passes marked "High
   Precision (32 bit)". §3.
3. **Pre-exposure multiplies emissive output** unless the capture has no view state or is unlit. It is
   the Unreal twin of the exposure scale ADR 0014 measured on Isaac. §3.
4. **No prior Unreal thermal system carries float32 temperature end to end.** AirSim and Cosys-AirSim
   remap segmentation ids to 8-bit counts per object. The open-source UE 5.8 projects are per-object
   and fp16. The serious commercial claim (JRM's EO/IR plug-in) could not be inspected. §5.
5. **The owner already runs UE 5.4.3 from source** (`HunterSimulator` on Cosys-AirSim). They have
   also measured why AirSim's capture path is slow: a ~100 ms lock wait, flat across resolution
   (`~/ue-capture-lab/docs/findings.md` F1–F5). The plan adopts that finding as a design rule. §2.
6. **The current release is UE 5.8.3** (22 Sep 2026). Since 5.4 it has gained four things this work
   needs:
   - Vulkan ray tracing at parity on Linux;
   - a production path tracer on Linux;
   - custom render passes inside the main renderer with several outputs;
   - USD asset import through Interchange, now production.

   §4.

---

## 2. What exists today

### 2.1 This repository's engine boundary

| thing | where | ports how |
|---|---|---|
| G-buffer contract | `src/irsim/config/gbuffer.py` — `GBuffer` (:106), `REQUIRED_KEYS` (:49): `temperature_k`, `normal_dot_view`, `distance_m` (Euclidean ray length), `material_id` int32, `sky_view_factor`; float16 refused on precision-critical keys (:88) | unchanged, already engine-free |
| Frame chain | `irsim.pipeline.frame.run_frame` (`frame.py:210`), stages 1–6, refuses a G-buffer off the k× grid | unchanged |
| Raw engine planes | `RawAovs` (`gbuffer_isaac.py:223`): distance, world normal, camera-space position, instance id, semantic, rgb; `geometry_planes()` (:362) and `to_gbuffer()` (:449) are pure NumPy | move into the core; only `AovReader` (:515, Replicator) and `camera_pose` (:264, pxr) touch Isaac |
| Temperature tiers | T0 per prim, `aerial_bridge.py:391`; T1 planar patch, `point_bridge.py:199`; T2 mesh cells, `mesh_bridge.py` (closest-point within 7 mm) — all host-side, no omni/pxr imports | unchanged; need a float32 position plane and an id → part map |
| Materials | `material_ids.py`: instance id → prim path → `MaterialResolver` (ADR 0047: `thermal:material` attribute, then semantic class, then a glob on the bound material's name) → packed `MaterialTable` | resolver unchanged; the prim walk (`materials_usd.py`, UsdShade) can read the same USDC with `usd-core` |
| Motion | synthesised host-side from per-instance transforms (`motion_isaac.py`); the renderer's motion AOV is not used (ADR 0014 addendum) | unchanged, given transforms per frame |
| Rotors, sun, night light | `rotor_isaac.py`, `illumination_isaac.py` — no engine imports | unchanged |
| Warp kernels | `warp_stages.py` (2,288 lines) imports only `warp` and `irsim.*`; four stages paired with CPU oracles at ≤ 1e-4 relative / ≤ 5 mK (ADR 0061) | unchanged; usable on frames from any engine |
| Outputs | `irsim/io/dataset.py` float32 `.npy`/`.exr`, uint16 PNG, RGBA8 PNG, per-frame JSON sidecar with config hash; `irsim_viewer` reads through `FrameSource`; `ros2_bridge.py` builds messages engine-free | unchanged |
| Scenes | `configs/scenes/*.yaml` → `irsim.scene.Scene.from_config`, engine-free. There is **no generic config → stage builder**: each of six drivers authors its own USD stage, IG.14 open | rewrite: one builder, not six |
| Glue size | `src/irsim_isaac`: 16,157 lines in 38 files. Reusable pipeline 6,610; shared infrastructure 2,649; demo drivers 4,137; probes 2,759 | — |
| Layering guard | `tests/unit/test_layering.py:35` forbids `omni, pxr, isaacsim, warp, carb` in the core. It has **no `unreal`** | add `unreal` |

The project already anticipated a port:

- CLAUDE.md:13 says a port to Unreal follows later.
- `docs/physics-model.md` §14 maps every component onto Unreal.
- ADR 0012 writes the LUTs as raw `.f32` "for … Unreal (`PF_R32_FLOAT`)".
- ADR 0081 rejected renderer motion blur for rotors partly because "the Unreal port would have to reproduce it".
- The roadmap (revision 6, priority 4 and open question 2) holds Unreal at "no step spent" until the owner decides. `IU-29`, the engine-interface contract, is parked there.

### 2.2 The owner's Unreal work on this machine

| item | found |
|---|---|
| Engine | UE **5.4.3** source build at `~/Documents/UnrealEngine` (`Build.version`: changelist 34507850), 80 GB |
| Project | `~/Documents/Unreal Projects/HunterSimulator` (UE 5.4, plugins AirSim, MapCapture, MovieRenderPipeline). Content packs include cities, RuralAustralia, Cesium, BritishCity, LAKETOWN and Downtown_West |
| Its IR camera | Cosys-AirSim `ImageType::Infrared` renders into **`PF_B8G8R8A8`** (`Plugins/AirSim/Source/PIPCamera.cpp:84`) through `Content/HUDAssets/InfraredMaterial`. It is 8-bit and per-object: the defect that drove the owner to irsim (roadmap, *owner's priorities* 1) |
| Capture latency study | `~/ue-capture-lab` (UE 5.4 plugin `MinCapture`). Findings F1–F5: the cost is a ~100 ms lock wait that does not scale with resolution (render 15 ms vs wait 100 ms at 4096²). Request and delivery are one object, so removing the wait gives a black frame. Project AirSim fixed it by driving capture from tick and publishing from `OnRendered`, with zero blocking waits |
| GPUs as Vulkan sees them | `vulkaninfo --summary`: **GPU0 = RTX 5090, GPU1 = llvmpipe (CPU), GPU2 = RTX A6000**. This order differs from `nvidia-smi` (A6000 = 0) |
| Disk | `/home` 36 GB free of 985 GB (97 %). Other volumes: `/media/hunter/38502FBA502F7E2E` 650 GB free, `/media/hunter/4ADE65E378CDB57C` 311 GB free. Their filesystems were not checked; the volume names look like NTFS ids (UNVERIFIED) |

---

## 3. Getting float32 out of Unreal

### 3.1 What drops to fp16, and what does not

| route | precision | source |
|---|---|---|
| `r.SceneColorFormat` default 4 = `PF_FloatRGBA` | fp16 | `ConsoleManager.cpp`; `SceneTexturesConfig.cpp` |
| `r.SceneColorFormat=5` = `PF_A32B32G32R32F` | float32; Epic calls it "unreasonable but good for testing" | same |
| `SCS_FinalColorHDR` | **fp16 always**: the tonemapper forces `PF_FloatRGBA` | `PostProcessTonemap.cpp` |
| `SCS_SceneColorHDR` / `SCS_SceneColorHDRNoAlpha` | raw SceneColor copy with **no pre-exposure removal**. float32 with `r.SceneColorFormat=5` and an `RTF_RGBA32f` target (128-bit permutation exists) | `SceneCapturePixelShader.usf`; `SceneCaptureRendering.cpp`; [ESceneCaptureSource](https://dev.epicgames.com/documentation/unreal-engine/API/Runtime/Engine/ESceneCaptureSource) |
| `SCS_SceneDepth` | float32 view-space Z (planar, not radial) into `RTF_R32f` | same |
| `SCS_Normal` | GBuffer: **10:10:10 unorm** by default | `GBufferInfo.cpp` |
| Post-process material output | inherits input format → fp16 by default | `PostProcessMaterialInputs.h` |
| `r.PostProcessingColorFormat=1` | only consumers found are buffer visualisation (MRQ) and high-res screenshots; do not rely on it (UNVERIFIED that it is unused elsewhere) | `PostProcessVisualizeBuffer.cpp`; `UnrealClient.cpp` |
| Movie Render Graph post-process pass with "Use High Precision (32 bit) Output" | renders to `PF_A32B32G32R32F`, EXR channel `FLOAT`. **Inputs can still be fp16**, and samples are averaged | [MRG nodes](https://dev.epicgames.com/documentation/unreal-engine/movie-render-graph-nodes-in-unreal-engine?lang=en-US); `MoviePipelineEXROutput.cpp` |
| MRQ Object ID | editor-only (HitProxy), not in `-game` | [forum](https://forums.unrealengine.com/t/community-tutorial-command-line-rendering-with-unreal-engine-movie-render-queue/681764) |
| Custom stencil | 8-bit | — |
| Custom Primitive Data | float32, 36 floats per primitive (docs still say 32) | `SceneTypes.h`; [docs](https://dev.epicgames.com/documentation/en-us/unreal-engine/storing-custom-data-in-unreal-engine-materials-per-primitive) |
| Cosys-AirSim float images | `FFloat16Color` via `ReadFloat16Pixels`: **fp16 even for depth** | [RenderRequest.cpp](https://github.com/Cosys-Lab/Cosys-AirSim/blob/main/Unreal/Plugins/AirSim/Source/RenderRequest.cpp) |

### 3.2 Pre-exposure, Unreal's version of ADR 0014's exposure scale

- **What it does.** The base pass multiplies everything written to SceneColor, emissive included, by
  `View.PreExposure` (`BasePassPixelShader.usf`).
- **When it is 1.0.** When the capture has no view state (`bCaptureEveryFrame` and
  `bAlwaysPersistRenderingState` both false), or when the EyeAdaptation or Lighting show flag is off.
  Otherwise it is tint × exposure × local exposure (`PostProcessEyeAdaptation.cpp`).
- **Global override.** `r.EyeAdaptation.PreExposureOverride` exists, but it is global and would also
  change the RGB view.
- **Test.** Render a known emissive value and compare it with what comes back.

### 3.3 Things that blend values, and must be off for data planes

- **What blends.** TSR/TAA jitter and history, Lumen and volumetric-cloud temporal reconstruction, and
  Movie Render Queue's spatial and temporal sample averaging all blend values. Averaging temperature is
  not averaging radiance, and averaging ids is meaningless.
- **SceneCapture2D defaults help.** TAA and motion blur are off, and GI and reflections default to None
  (`SceneCaptureComponent.cpp`).
- **Main-view-mirroring captures** (5.5+) copy TAA jitter unless "Ignore Screen Percentage" is set
  ([5.5 notes](https://dev.epicgames.com/documentation/en-us/unreal-engine/unreal-engine-5-5-release-notes)).
- **For MRQ Object ID** Epic itself says to set AA to None and disable multisample effects
  ([docs](https://dev.epicgames.com/documentation/en-us/unreal-engine/cinematic-render-passes-in-unreal-engine)).

### 3.4 Renderer hooks a plugin can use without forking the engine

- **Scene View Extension hooks** (`SceneViewExtension.h`):
  - `PostRenderBasePassDeferred_RenderThread`, which receives the GBuffer;
  - `PrePostProcessPass_RenderThread`;
  - `SubscribeToPostProcessingPass`;
  - `PostTLASBuild_RenderThread`.

  RDG compute in such an extension can write `PF_R32_FLOAT` UAVs. Templates:
  [SceneViewExtensionTemplate](https://github.com/A57R4L/SceneViewExtensionTemplate) (5.6) and
  [itscai.us walkthrough](https://itscai.us/blog/post/ue-view-extensions/).
- **Per-capture extensions.** A SceneCapture can carry its own extensions
  (`SceneCaptureComponent2D::SceneViewExtensions`), so custom passes can be limited to the IR capture.
- **Custom mesh passes.** Epic staff recommend a custom `MeshPassProcessor` plus a Scene View Extension
  ([forum, Mar 2026](https://forums.unrealengine.com/t/custom-render-passes/2718793)). Whether that
  needs an engine change on 5.8 (the `EMeshPass` enum is fixed) is UNVERIFIED.
- **Custom render passes in the main renderer** (`bRenderInMainRenderer`):
  - 5.4.3: depth only.
  - 5.8: also BaseColor, Normal and SceneColor (emissive/unlit), plus `UserSceneTexture*` outputs, so
    one pass can produce several planes (`SceneCaptureComponent2D.h`).
  - Precision is UNVERIFIED.
- **Readback.** `FRHIGPUTextureReadback` with a fence is non-blocking: about 40 µs of CPU, at the cost
  of 3 frames of latency ([blog](https://nicholas477.github.io/blog/2023/reading-rt/)). That matches
  ue-capture-lab's conclusion: pipeline the captures and never block on one.

### 3.5 Temperature inside the engine (needed only for a live in-engine IR view)

| mechanism | precision |
|---|---|
| `UTexture2D::CreateTransient(W,H,PF_R32_FLOAT)` + `UpdateTextureRegions` (`Texture2D.h`) | float32 |
| per-object `RTF_R32f` render target written by compute | float32 |
| `UDynamicMeshComponent` UV channel (full-precision UVs, `MeshRenderBufferSet.h`) | float32, not Nanite |
| static-mesh UVs | fp16 default; float32 with "Use Full Precision UVs" |
| `UProceduralMeshComponent` UVs | fp16 ([forum](https://forums.unrealengine.com/t/can-someone-confirm-me-that-the-uv-on-the-uproceduralmeshcomponent-are-32-bits/449523)) |
| Nanite attributes | quantised, width undocumented (UNVERIFIED) |
| vertex colour, Runtime Virtual Texture | 8-bit / BC / at best R16_UNORM: inadequate |
| Material Parameter Collection | float32, global, 1024 scalars + 1024 vectors |

- **If a path must stay fp16,** encode Celsius or an offset, not Kelvin. fp16 spacing near 20 °C is
  about 0.016 K, against 0.25 K at 300 in Kelvin. That is acceptable only for near-ambient scenes.

---

## 4. Version, platform, automation, assets

- **Releases.**
  - 5.6: 3 Jun 2025 ([forum](https://forums.unrealengine.com/t/unreal-engine-5-6-released/2538952)).
  - 5.7: 12 Nov 2025 ([forum](https://forums.unrealengine.com/t/unreal-engine-5-7-released/2673913)).
  - 5.8: 17 Jun 2026 ([forum](https://forums.unrealengine.com/t/unreal-engine-5-8-released/2729274)).
  - 5.8.3: 22 Sep 2026 ([forum](https://forums.unrealengine.com/t/5-8-3-hotfix-released/2833315)).
  - UE6, which merges UE5 and UEFN, targets early access at the end of 2027
    ([GamesBeat](https://gamesbeat.com/unreal-engine-6-will-combine-ue5-and-uefn-into-a-unified-engine-state-of-unreal/)).
    "5.8 is the last planned 5.x" is UNVERIFIED.
- **Since 5.4.** Each item is from [5.5 notes](https://dev.epicgames.com/documentation/en-us/unreal-engine/unreal-engine-5-5-release-notes) or [5.8 notes](https://dev.epicgames.com/documentation/unreal-engine/unreal-engine-5-8-release-notes?lang=en-US):
  - 5.5: Vulkan ray tracing on by default at DX12 parity on Linux (driver 550+), and the path tracer production-ready with Linux support.
  - 5.7: Substrate production-ready, and Linux moved from SDL2 to SDL3.
  - 5.8: Movie Render Graph production-ready, and USD import through Interchange production-ready for assets (experimental for levels).
- **Cosys-AirSim builds.** v3.3 for UE 5.5; v3.4–3.5.0 for **UE 5.8**. There is no official 5.4 build
  ([releases](https://github.com/Cosys-Lab/Cosys-AirSim/releases)).
- **Linux and GPUs.**
  - `-graphicsadapter=N` picks a Vulkan device by **Vulkan's** index (`VulkanRHI.cpp`). It works with
    `-RenderOffscreen` ([forum](https://forums.unrealengine.com/t/unable-to-select-gpu-when-using-renderoffscreen/478045)).
  - VulkanRHI has no explicit multi-GPU: one GPU per process.
  - RTX 50-series needs NVIDIA's open kernel modules. There are 2026 reports of UE 5.7/5.8 Vulkan
    crashes on them ([Arch](https://bbs.archlinux.org/viewtopic.php?id=313526)), and a 5.8.0
    `VK_ERROR_DEVICE_LOST` targeted for 5.8.1
    ([forum](https://forums.unrealengine.com/t/ue-5-8-release-instant-vulkan-crash-vk-error-device-lost-on-linux-with-rtx-3090-ti-nvidia-driver/2729632)).
    The A6000 is the safer card.
- **Python.** Built-in Python is 3.11 and "only available in the Unreal Editor, not when your Project
  is running … Standalone Game, cooked executable"
  ([docs](https://dev.epicgames.com/documentation/en-us/unreal-engine/scripting-the-unreal-editor-using-python)).
  irsim's NumPy stack therefore stays in its own process.
- **Control and transport.**
  - Remote Control API: HTTP 30010 and WebSocket 30020; packaged builds need `-RCWebControlEnable`
    ([quick start](https://dev.epicgames.com/documentation/en-us/unreal-engine/remote-control-quick-start-for-unreal-engine)).
    It is JSON: fine for control, wrong for planes.
  - Named shared memory exists (`FPlatformMemory::MapNamedSharedMemoryRegion`). No off-the-shelf
    UE ↔ NumPy plugin was found.
  - Determinism: `-benchmark -fps=N`, `-deterministic`, or "Use Fixed Frame Rate"
    ([command-line docs](https://docs.unrealengine.com/4.27/en-US/ProductionPipelines/CommandLineArguments)).
- **USD.**
  - Interchange USD is production for assets in 5.8.
  - Runtime loading still works through the USD Stage Actor `SetRootLayer`
    ([docs](https://dev.epicgames.com/documentation/unreal-engine/universal-scene-description-in-unreal-engine)).
  - Multiple GeomSubset material slots have worked since 5.3.
  - A prim can bind an existing Unreal material through `info:unreal:sourceAsset`.
  - Arbitrary primvars → UV channels in 5.8 is UNVERIFIED.
  - How `purpose = "guide"` prims (irsim's hidden parts) are treated is UNVERIFIED.
- **Atmosphere and weather.** Sky Atmosphere, Exponential Height Fog and Volumetric Clouds are
  visible-band, and the clouds are fp16 and temporally reconstructed (`VolumetricCloudRendering.cpp`).
  Their rendered *output* therefore cannot be a data plane. Their *density fields and parameters* can
  be shared with the IR band, once they are driven from physical quantities rather than tuned by eye.
  The owner asked for exactly that on 2026-10-01: Unreal's weather is powerful, but it has to be
  changed to work as a simulation (plan phase `U4`). OpenVDB → Sparse Volume Texture is experimental
  ([docs](https://dev.epicgames.com/documentation/en-us/unreal-engine/sparse-volume-textures-in-unreal-engine)).
  The Water plugin is still experimental.
- **Custom rays.** [RayTraceShaders](https://github.com/A57R4L/RayTraceShaders) (5.7+) and
  [CustomRaytracingShader](https://github.com/historia-Inc/CustomRaytracingShader) (5.7, documents DX12
  only) trace custom rays against the scene TLAS from a plugin. Vulkan support is UNVERIFIED.
- **Licence.**
  - Free under US$1M annual gross revenue.
  - Above that, a non-game company needs the seat subscription at US$1,850 per seat per year.
  - Software shipped to third parties that runs UE code stays on the 5 % royalty model.
  - Sources: [CG Channel](https://www.cgchannel.com/2024/03/new-pricing-for-unreal-engine-twinmotion-and-realitycapture/) and [Engadget](https://www.engadget.com/epic-will-charge-non-game-developers-1850-per-seat-to-use-unreal-engine-162015997.html), both reporting Epic's 2024 announcement.

---

## 5. Prior art: how others put temperature into an Unreal image

| system | mechanism | precision |
|---|---|---|
| AirSim / Cosys-AirSim Infrared | a script turns temperature, emissivity and camera response into a count per object, then reassigns segmentation ids to it ([AirSim](https://microsoft.github.io/AirSim/InfraredCamera/), [Cosys](https://github.com/Cosys-Lab/Cosys-AirSim/blob/main/docs/InfraredCamera.md)) | per object, 8-bit |
| AirSim-W (Bondi et al., 2018) | per-class temperature table → band Planck with camera response → normalised by scene maximum to 8-bit; atmosphere ignored ([PDF](https://www.cais.usc.edu/wp-content/uploads/2018/05/bondi_camera_ready_airsim-w.pdf)) | per class, 8-bit, not radiometric |
| Sci. Rep. 2025, multiview UAV IR | AirSim with meshes split into regions by material and temperature ([paper](https://www.nature.com/articles/s41598-025-89585-x)) | per region |
| ThermalVision (UE 5.8, MIT) | Scene View Extension + RDG; custom stencil *is* °C; stored R16F ([repo](https://github.com/jlparreno/ThermalVision)) | per object, fp16 |
| ThermoForge (UE 5.6/5.7) | gameplay heat field, post-process vision, "not exact thermodynamics" ([repo](https://github.com/cem-akkaya/ThermoForge)) | not radiometric |
| IUPUI/Purdue (IMECE 2023, thesis) | post-process-material IR model on UE 5.1 + rclUE ([plugins](https://github.com/jevansiv/Unreal_Engine_ROS2_Sensor_Plugins)) | UNVERIFIED (full texts 403) |
| JRM EO/IR Sensor Plug-in (Fab) | SigSim library, 0.2–25 µm, "radiometrically-correct" ([Fab](https://www.fab.com/listings/7e781305-8e5e-4131-8af3-cc9b7e7a998f)) | UNVERIFIED (page 403) |
| MuSES 2023.1 | exports per-texel temperature results to "gaming engines" ([notes](https://support.thermoanalytics.com/hc/en-us/articles/15915299259923-MuSES-2023-1)) | offline, format unstated |
| Duality Falcon 5.3 (UE 5.5) | "each material is now automatically assigned a baseline IR value" ([blog](https://www.duality.ai/blog/falcon-5-3-with-robots)) | per material |
| CARLA, Project AirSim | no thermal sensor ([CARLA #6546](https://github.com/carla-simulator/carla/discussions/6546), [Project AirSim](https://iamaisim.github.io/ProjectAirSim/sensors/camera_capture_settings.html)) | — |

The pattern is consistent: Unreal thermal cameras put temperature *into* the renderer as a per-object
value and lose precision on the way. irsim's design, where the renderer carries geometry and ids and
the physics core owns temperature, is the reason the port is tractable.

---

## 6. Where the evidence is thin

- The cost of RGBA32F SceneColor.
- The precision of 5.8's `UserSceneTexture` outputs.
- Nanite UV quantisation.
- USD primvars and `purpose` on import.
- Vulkan support in the custom ray-tracing plugins.
- Whether a custom mesh pass needs an engine change.
- Whether the 5.8 Linux and Blackwell crashes are fully fixed in 5.8.3.
- The internals of JRM, Presagis and the Purdue work.

The plan's gate spike (`U0.3`) measures the first six on this machine before any glue code depends on
them, as ADR 0014 did for Isaac.
