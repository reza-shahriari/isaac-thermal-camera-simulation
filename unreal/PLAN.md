# irsim on Unreal Engine — the plan

**Goal:** Unreal Engine 5 renders irsim's frames instead of Isaac Sim. The physics core stays as it is.
Unreal takes over the job Isaac does today: it turns a scene config into geometry, ids, transforms and
an RGB companion frame, and hands them to the engine-free pipeline.

This plan lives **here, and only here**. The owner asked for it on 2026-10-01 as "another plan file",
following the pattern of [`blender_addon/PLAN.md`](../blender_addon/PLAN.md). The same day the owner
ruled that **`docs/roadmap.md` and `CLAUDE.md` are not changed at all** for this work, and no step below
changes them. Some steps do change project code (`src/irsim`, `tests/`); those are marked in the *needs
from the project* column and are tracked in this file, not in the roadmap.

The evidence behind every decision below, with sources, is in
[`docs/research/2026-10-01-unreal-engine-as-renderer.md`](../docs/research/2026-10-01-unreal-engine-as-renderer.md)
(cited as *survey §N*).

---

## Why the port is tractable

irsim's renderer already carries **no temperature**. On Isaac every colour AOV turned out to be
float16 and exposure-scaled (about 100 mK at 300 K, ten times the 10 mK budget). ADR 0014 therefore
moved temperature out of the renderer. The engine supplies:

- float32 distance;
- float32 position;
- normals;
- exact instance ids with an id → prim-path map;
- the RGB companion.

The host then computes surface temperature per prim, per planar patch or per mesh cell (ADR 0060,
0087, 0110, 0111). It also synthesises motion from transforms, composites rotors and evaluates
sun/night illumination itself. None of that code imports an engine. An Unreal glue that delivers the
same raw planes reuses it all. That covers the owner's headline requirement too: per-point temperature
across one object survives the switch without new physics.

Every Unreal thermal camera surveyed does the opposite: it puts a temperature or a count per object
*into* the renderer and loses precision on the way. That includes AirSim, Cosys-AirSim (the owner's
`HunterSimulator`, 8-bit `PF_B8G8R8A8`), AirSim-W and the open-source UE 5.8 projects (survey §5).
This plan does not repeat that design.

**Reused unchanged.**

- The whole of `src/irsim`: `GBuffer`, `run_frame`, the radiometry, atmosphere, optics, detector,
  noise and ISP stages, scenes, weather and solvers.
- The engine-free parts of `src/irsim_isaac`, which are ≈ 6,600 of its 16,157 lines:
  - `aerial_bridge`, `point_bridge` and `mesh_bridge`;
  - `rotor_isaac` and `illumination_isaac`;
  - `warp_stages`;
  - the NumPy halves of `gbuffer_isaac`;
  - `ros2_bridge`.
- `irsim.io` writers, `irsim_viewer`, `irsim_eval`, and every YAML config and `.f32` LUT.

**Rewritten.**

- The capture: Replicator `AovReader`, about 670 lines with its helpers.
- Camera authoring.
- The prim walk: `materials_usd` becomes a `usd-core` read of the same USDC.
- Transform reads for motion.
- `env.py` (launch, GPU pinning).
- Stage building. Today that is six hand-written drivers totalling ≈ 4,100 lines; under Unreal it
  becomes one builder.
- The companion sky and clouds.

---

## Decisions, and why

1. **Unreal carries ids and geometry, never temperature** — ADR 0014's design, kept on purpose.
   - The default SceneColor, `SCS_FinalColorHDR`, post-process materials, GBuffer normals and
     Cosys-AirSim's float images all drop to fp16 or below (survey §3.1).
   - Pre-exposure multiplies emissive output (survey §3.2).
   - A transport that carries no temperature is immune to all of it.
   - Temperature enters the engine only for the optional in-engine live view (`U6.2`), and only
     through a float32 path measured in `U0.3`.
2. **UE 5.8.3, pinned.** It is the current release (22 Sep 2026). Since the owner's 5.4.3 it has gained:
   - Vulkan ray tracing at parity on Linux;
   - a production path tracer on Linux (the full-fidelity cloud tier needs one);
   - custom render passes with several outputs;
   - production USD asset import;
   - official Cosys-AirSim builds (none exist for 5.4).

   UE6 is due in 2027, so the plugin is kept small enough to move. The owner's 5.4.3 build is the
   fallback if 5.8.3 fails on this machine's Linux and driver.
3. **A plugin of our own for capture, not Cosys-AirSim.** The owner's own measurements in
   `~/ue-capture-lab` (F1–F5) show AirSim's capture blocks for ~100 ms whatever the resolution,
   because request and delivery are one object. Its float images are fp16 and its infrared is 8-bit
   per object. The irsim plugin drives capture from tick, reads back asynchronously
   (`FRHIGPUTextureReadback`, ~3 frames of latency, no blocking wait) and pushes frames. It never
   answers a request with a render. Cosys-AirSim may still fly vehicles in `HunterSimulator` if the
   owner wants it to (open question 5).
4. **irsim stays outside Unreal's process.** Unreal's Python is 3.11 and exists only in the editor,
   never in a game or cooked build. irsim's NumPy stack runs in its own process:
   - It talks to the plugin over a control socket.
   - It receives float32 planes through named shared memory.
   - Unreal-side Python is used only for editor-time work: asset import and level building.
   - A side effect: Unreal work needs no Isaac interpreter. Any CPython ≥ 3.10 with `.[dev]` runs
     the client.
5. **The engine contract lands in the core first, and Isaac moves onto it first.** `RawAovs`,
   `geometry_planes` and `to_gbuffer` are already pure NumPy. They become an engine-neutral
   `RawFrame` plus an `EngineSession` protocol (this is the parked roadmap row `IU-29`). The Isaac
   adapter is ported onto that protocol before any Unreal code exists. Parity between the engines is
   then a plane-by-plane comparison on one scene config, not an argument.
6. **Unreal's own weather is the weather, changed to work as a simulation.** The owner, 2026-10-01:
   *"the unreal engine's own weather is so powerful but should be changed to be able to work as a
   simulation"*. Unreal's weather tools are richer than anything irsim would build:
   - Sky Atmosphere;
   - Volumetric Clouds;
   - height and volumetric fog;
   - Niagara rain and snow;
   - wet and snow material layers.

   As shipped, though, they are tools for a look. Their sliders are tuned by hand, they cover the
   visible band only, and they animate on wall-clock time. Cloud output is fp16 and temporally
   reconstructed (survey §4). Four changes turn them into a simulation (phase `U4`):
   - **Driven, not authored.** Every parameter is set each frame from the one `WeatherSeries`
     (CLAUDE.md #6): the NOAA sun, cloud cover, base, top and water content, visibility, rain rate,
     snowfall, wind and humidity. A slider with no physical quantity behind it is fixed, not exposed.
   - **Physical units, calibrated.**
     - Fog extinction comes from visibility.
     - Aerosol and Rayleigh coefficients come from the atmosphere preset.
     - Cloud extinction comes from water content and droplet size.
     - Particle counts come from rain rate and drop-size distribution.

     Each is checked against the contrast of targets at known distances.
   - **Deterministic.** Seeded, and animated on simulation time from the fixed step, so one config
     always renders the same weather.
   - **One field for both bands.** The density fields Unreal draws in RGB are the fields the IR band
     integrates. They cross to the core as float32 *density*, never as Unreal's rendered cloud or fog
     *output*, so the two bands agree about where cloud is (ADR 0076). The per-band extinction and
     emission stay in the core. A target behind cloud is attenuated in the IR as it is in RGB.

   The data capture still excludes all of it, so ids and geometry stay exact.
7. **CPU reference first.** The in-engine HLSL port of the chain (`U6.2`) is the last phase, and it
   needs the owner's word that the CPU pipeline is in good shape. Until then the frames run through
   `run_frame` on the host, as they do under Isaac.
8. **Isaac is frozen, not deleted, at the switch** (`U7.2`). Deleting `src/irsim_isaac` is a
   separate decision for the owner (open question 1).
9. **The plugin lives in this repository**, under `unreal/Plugins/IrsimCapture/`, beside a thin host
   project `unreal/IrsimHost/`. It implements this project's contract, so it versions with it.
   `ue-capture-lab` stays its own repository, and its findings are cited, not copied. Marketplace
   content and `HunterSimulator` levels never enter git: a scene names an Unreal level by path, as
   scenes name USD files today.
10. **Coordinates are converted in one engine-free module.** It handles ENU/USD (right-handed, metres)
    and Unreal (left-handed, Z-up, centimetres) both ways. A sign error here mirrors the frame and
    still looks plausible, so it is tested with an asymmetric target before anything renders.

---

## Steps

Phases run in order; within a phase, a step starts when its *deps* are done. Sizes are S (≤ 1 day),
M (a few days) and L (a week or more). Every step that produces a frame also produces a video (the
owner's standing rule) and checks the IR frame against its RGB companion before it is called done.

### U0 — Decisions and the gate (no project code)

| step | what | deps | gate | size | needs from the project | status |
|---|---|---|---|---|---|---|
| U0.1 | **The owner's choices for this plan.** These are the open questions below: Isaac retired or frozen, engine version, build location, licence, environments, distortion, and which way the cloud field flows | — | answers recorded in this file's open-question table | S | — | open |
| U0.2 | **Build UE 5.8.3 on Linux, and pin the A6000.** Build from source on an ext4 volume with room. `/home` has 36 GB free; a source build with intermediates and DDC needs well over 100 GB (UNVERIFIED estimate). Record engine, driver and Vulkan device list. Vulkan enumerates **5090 = 0, llvmpipe = 1, A6000 = 2**, unlike `nvidia-smi`. So `IRSIM_GPU` must map to a Vulkan index by device *name*, checked against Unreal's "Using device" log line | U0.1 | `UnrealEditor -RenderOffscreen -graphicsadapter=<A6000>` renders an empty level on the A6000, shown by `nvidia-smi` while it runs | M | — | open |
| U0.3 | **The ADR 0014 measurement, repeated on Unreal.** Use 64 quads at known positions with known ids, plus an emissive ramp 200–1000 K including 300.000/300.050/300.100 K. Measure five candidate transports: **C1** unlit data materials → `SCS_SceneColorHDRNoAlpha` → `RTF_RGBA32f` with `r.SceneColorFormat=5` and no view state; **C2** 5.8 custom render pass in the main renderer with `UserSceneTexture` outputs; **C3** `SCS_SceneDepth` → `RTF_R32f`; **C4** a Scene View Extension with RDG compute reading depth and the GBuffer; **C5** a custom mesh pass, only if C1 and C2 fail. For each, record dtype, full resolution or not, id exactness, position and distance error, normal error, and the pre-exposure gain on a known emissive value | U0.2 | the ADR 0014 bars: ids distinct and every 5×5 centre window one id; position error ≤ one pixel footprint; distance ≤ 1 mm at 2 m; |n·v| error worth ≤ 5 mK of apparent temperature at 80°; emissive read back to ≤ 1 ulp of float32; ramp round trip < 10 mK (needed only for `U6.2`) | M | an ADR, "Unreal transports ids and geometry, measured" | open |
| U0.4 | **The delivery spike.** Tick-driven capture, a queue of `FRHIGPUTextureReadback`s and named shared memory into a Python reader, at 2560×2048 (k = 4 over 640×512) float32 × 4 planes under a fixed time step. Builds on `ue-capture-lab` delivery mode B | U0.3 | no blocking wait anywhere; frame stamps strictly increase; zero dropped frames over 1,000 fixed steps; latency and throughput recorded as numbers | M | — | open |

### U1 — The engine contract in the core (engine-free; Isaac benefits first)

| step | what | deps | gate | size | needs from the project | status |
|---|---|---|---|---|---|---|
| U1.1 | **`RawFrame`.** Move `RawAovs`, `geometry_planes()` and `to_gbuffer()` out of `gbuffer_isaac.py` into the core. The frame carries distance, camera-space position, world normal, instance id, the id → part-path map, rgb, camera pose and per-instance transforms. Isaac's `AovReader` returns it | U0.1 | every existing unit test over the Isaac glue (`_FakeReader` and the rest) passes unchanged; `make check` green | M | `src/irsim`, `src/irsim_isaac` | open |
| U1.2 | **`EngineSession` protocol** (roadmap `IU-29`): `build(scene)`, `advance(t, poses)`, `capture() → RawFrame`, `transforms()`, `close()`. Write the Isaac adapter, a fake adapter, and a conformance suite any adapter must pass | U1.1 | the fake and Isaac adapters pass the same conformance suite; one demo driver runs through the protocol | M | `src/irsim`, `tests/unit` | open |
| U1.3 | **Frames and units.** One engine-free module for ENU ↔ USD ↔ Unreal (left-handed, Z-up, cm): positions, rotations, normals, camera axes, motion sign | U1.1 | an asymmetric target and a turning rotor survive a round trip through every pair of frames to 1e-9 m, and a test fails if any axis is flipped | S | `src/irsim` | open |
| U1.4 | **Layering.** Add `unreal` to `test_layering.py`'s forbidden engine set, and scan `src/irsim_unreal` as glue. The client package must not import `unreal`, which exists only inside the editor | U1.2 | the scanner fails on a planted `import unreal` in the core and in the client | S | `tests/unit/test_layering.py` | open |

### U2 — The first Unreal frame

| step | what | deps | gate | size | needs from the project | status |
|---|---|---|---|---|---|---|
| U2.1 | **Plugin and host project.** Set up `unreal/Plugins/IrsimCapture` (C++) and `unreal/IrsimHost`, with build instructions and UE Automation tests runnable headless (`UnrealEditor-Cmd … -ExecCmds="Automation RunTests Irsim" -RenderOffscreen`) | U0.4 | it builds on 5.8.3 Linux and the automation suite runs from one command | M | a `make test-unreal` target, outside `make check` | open |
| U2.2 | **The data capture**, using the transport `U0.3` chose. Per-frame asserts: dtype float32 at the readback boundary, pre-exposure = 1, no TAA/TSR, one sample, unlit, and no fog, atmosphere or cloud in the data view. A failed assert refuses the frame rather than writing it | U2.1 | the U0.3 bars re-run inside the plugin as automation tests | M | — | open |
| U2.3 | **`UnrealSession`** in `src/irsim_unreal/session.py`, implementing `EngineSession`. Control over a socket, planes through shared memory, lock-step fixed time step. The engine never blocks waiting on a request | U1.2, U2.2 | passes the `U1.2` conformance suite against a running editor (`@pytest.mark.unreal`, skipped by default) | M | a new pytest marker | open |
| U2.4 | **The camera.** Map the sensor YAML (focal length, pitch, k× supersample) to Unreal capture FOV and resolution, with an exact pinhole check. Distortion: ADR 0015 lets the engine own it, but Unreal's lens distortion acts on scene colour, not on data planes. Recommended: render pinhole and remap the raw planes host-side (ids by nearest neighbour), so both engines share one model | U2.3 | a grid target lands on the predicted pixels to ≤ 0.1 px; the distortion choice is recorded in an ADR | M | an ADR amending 0015 | open |
| U2.5 | **First frame: `PT.20`'s reference scene** (a block on a ground patch). It is the smallest scene that shows the owner's first requirement: a gradient across one prim, declared in the scene config. Write float32 planes and the sidecar, open them in `make viewer`, encode a video | U2.4, U3.1 | the same config renders through Isaac and Unreal. Interior pixels (≥ 2 px from any id edge) agree in apparent temperature to ≤ 10 mK and in id to 100 %; at least 99 % of all pixels agree in id; the sunlit and shaded faces differ as the solver says | M | `TECHNICAL_REPORT.md` status row | open |

### U3 — Assets, materials and scenes

| step | what | deps | gate | size | needs from the project | status |
|---|---|---|---|---|---|---|
| U3.1 | **Asset import.** Bring library USDC files in through Interchange USD (production for assets in 5.8). Every component carries its source prim path, which builds the id → part-path map. `thermal:material` is resolved host-side by reading the same USDC with `usd-core`, so the resolver (ADR 0047) is unchanged. Hidden parts (`purpose = "guide"`) must never reach a camera, and that has to be measured. FBX stays a courtesy; using it needs an ADR amending 0150 | U2.3 | the DJI Inspire 3 and Phantom 4 import with every part mapped; a guide-purpose box is absent from RGB and from every data plane; a slot-name audit runs after import | M | — | open |
| U3.2 | **Instance ids.** Write each component's id into Custom Primitive Data at load time (float32, exact to 2²⁴), and have the data material emit it. Unmapped components render magenta with NaN radiometry (ADR 0047, IG.17) | U3.1 | ids round-trip exactly for 10⁵ components; a deliberately unmapped part shows magenta | S | — | open |
| U3.3 | **One scene builder.** Engine-free code turns a scene config into a scene manifest (JSON: meshes, transforms, materials, sun, camera path). An editor-Python script in the plugin builds the Unreal level from it. This is the `IG.14` idea, applied once instead of six times | U3.1, U1.3 | the 24 scene configs in `configs/scenes/` build in Unreal with no per-scene Python | L | `src/irsim` (the manifest) | open |
| U3.4 | **Transforms and motion.** Stream per-instance transforms with every frame. `motion_px` is synthesised host-side as on Isaac, and rotor veils are composited host-side (ADR 0081). Unreal's motion blur stays off | U2.3 | the `test_motion_wired` scenarios give the same `motion_px` through both engines to 1e-3 px | S | — | open |
| U3.5 | **A real environment.** Map one owned marketplace environment (a city block from `HunterSimulator`) through `configs/materials/mapping.yaml` globs, with an `id_coverage` report (IG.10). The content stays out of git | U3.2 | at least 95 % of pixels mapped by area; every unmapped material listed by name | M | mapping rows | open |

### U4 — Unreal's weather as a simulation, and the RGB companion

| step | what | deps | gate | size | needs from the project | status |
|---|---|---|---|---|---|---|
| U4.1 | **One sun.** Aim Unreal's DirectionalLight and Sky Atmosphere from the scene's NOAA sun (the same vector that feeds `l_sun`) and from the atmosphere preset | U3.3 | sun direction agrees with the core to 0.01°; the data planes are bit-identical with all weather on and off | S | — | open |
| U4.2 | **RGB companion.** Same pose and intrinsics, k× then box-filtered onto the IR grid (IG.9). Two tiers: real-time (Lumen/TSR) and path-traced (Unreal's path tracer on Linux). Sub-frame exposure over the IR integration window (ADR 0167) | U4.1 | registration test: RGB and IR edges coincide to ≤ 0.5 px; both tiers render on the A6000 | M | — | open |
| U4.3 | **The weather driver.** List every parameter of Sky Atmosphere, Volumetric Cloud, height and volumetric fog, Niagara precipitation, and the wet and snow material layers. Map each to a quantity in `WeatherSeries` or the atmosphere preset, or fix it. A plugin component sets them from the scene manifest on every frame, seeded, and animated on simulation time | U4.1 | a table of every parameter, its physical quantity and its source; no parameter left on wall-clock time; two runs of one config give identical weather | M | — | open |
| U4.4 | **Calibrated in physical units.** Fog and haze: visibility → extinction, so black/white targets at known distances lose contrast as exp(−3.912·d/V) in the RGB render (fixed exposure, HDR read). Sky Atmosphere coefficients come from the preset. Precipitation particle density comes from rain rate and drop-size distribution ([`docs/research/2026-10-01-weather/`](../docs/research/2026-10-01-weather/)) | U4.3 | the contrast-vs-distance fit recovers the authored visibility within 10 % at three visibilities; particle density matches the rate | M | — | open |
| U4.5 | **Clouds: one field, both bands.** A spike decides which way the field flows. **(A)** A plugin compute pass evaluates Unreal's own cloud density (driven by `WeatherSeries`) on a float32 3D grid and hands it to the core, whose IR march (ADR 0162/0169) integrates it per band. **(B)** The core's `CloudField` drives Unreal's cloud material through a 3D texture or Sparse Volume Texture (experimental). Default (A): it keeps the cloud detail the owner wants from Unreal | U4.4 | cloud masks of the two bands agree to the ADR 0076 bar (it catches the 26 %/57 % defect); in the path-traced tier a target behind cloud is attenuated in both bands; in the real-time tier clouds still show in the IR | L | — | open |
| U4.6 | **Rain, snow, fog and wet surfaces.** Niagara precipitation and the wet and snow layers are driven by the same series and by the surface water and snow state the core solves. A wet road half then looks wet in RGB where it reads colder in IR. The IR extinction and emission of rain, snow and fog stay in the core, at the same rate. If this grows, it becomes its own project consumed as a submodule, as `isaac-weather-fx` is | U4.4, the `WX` and `PH` lanes' physics | wet and snow masks agree between bands; the IR loss over a path matches the core's rate model | L | — | open |
| U4.7 | **Water surface for the maritime lane.** Unreal's Water plugin is experimental, so use it for geometry and normals only, or render the core's own sea surface mesh. Emissivity and reflection stay in the core | U3.3 | the sea's apparent-temperature depression matches the golden fixture | M | — | open |

### U5 — Each lane reaches parity (owner's order: aerial, maritime, ground)

| step | what | deps | gate | size | needs from the project | status |
|---|---|---|---|---|---|---|
| U5.1 | **Aerial.** The aerial demo, quad flight, quad outbound, aircraft pass and asset flight, through `UnrealSession` | U4.5 | each scene config plus one command produces float32 planes, sidecars and a video; parity report against Isaac | L | — | open |
| U5.2 | **Maritime.** The maritime demo and vessel departure | U4.7, U5.1 | as U5.1 | M | — | open |
| U5.3 | **Ground.** Car ignition, plus the `TC.6` and `PT.20` reference scenes | U5.1 | as U5.1 | M | — | open |
| U5.4 | **Multiband and ROS 2.** Four bands on one scene, one GPU per process (Vulkan has no explicit multi-GPU), published through the existing engine-free `ros2_bridge` | U5.1 | four bands from one command; ROS 2 topics carry 32FC1 apparent temperature | M | — | open |

### U6 — Live IR inside Unreal

| step | what | deps | gate | size | needs from the project | status |
|---|---|---|---|---|---|---|
| U6.1 | **IR in the viewport.** Push the host's `display8` back as a texture into a viewport widget, the counterpart of the `omni.ui` route the owner asked for on Isaac. No GPU port is needed | U5.1 | an editor session shows the IR frame live beside the RGB view at the sensor's frame rate | M | — | open |
| U6.2 | **In-engine chain** (gated by "CPU reference first"). Port stages 1–6 to HLSL compute in a Scene View Extension with `PF_R32_FLOAT` throughout. The LUT ships as a `PF_R32_FLOAT` texture from ADR 0012's `.f32` files; IIR, FFC and noise use persistent render targets. Temperature enters the engine through the float32 path U0.3 measured (per-object R32F textures with full-precision UVs) | owner's word; U6.1 | against the NumPy oracle: ≤ 1e-4 relative and ≤ 5 mK per stage (ADR 0061's budget) on the synthetic G-buffer fixtures | L | an ADR | open |
| U6.3 | **IR reflections by ray tracing** (§13.7 option 4). Trace mirror rays for specular materials from a Scene View Extension (`PostTLASBuild`), fetching the hit's id and position. Spike first, because Vulkan support of custom ray-tracing shaders is unverified | U6.2 | a mirror plate shows the hot target it faces, at the angle-correct radiance | L | an ADR | open |

### U7 — Switch-over

| step | what | deps | gate | size | needs from the project | status |
|---|---|---|---|---|---|---|
| U7.1 | **Parity report.** Every lane's scene configs through both engines, with a table of plane differences, frame times and known gaps, published on the site | U5.4 | the owner reads it and decides | M | site page | open |
| U7.2 | **Switch.** Unreal becomes the default engine of the render commands. Add a project skill for Unreal (the counterpart of `isaac-sim-spg`) and update the Makefile, `TECHNICAL_REPORT.md` and README. `docs/roadmap.md` and `CLAUDE.md` stay as they are. `src/irsim_isaac` is frozen, not deleted | U7.1, owner's word | `make check` green; `make test-unreal` green on the A6000 | M | skills, Makefile | open |

---

## Risks

| risk | what goes wrong | mitigation |
|---|---|---|
| fp16 creeps in | a capture source, a post-process material, default SceneColor, GBuffer normals or a UV channel quietly halves precision while the image still looks fine | dtype and known-value asserts at every boundary on every frame (`U2.2`); refuse the frame |
| Pre-exposure | emissive output is scaled by eye adaptation | data capture without view state or unlit; a known-value check per run |
| Temporal blending | TSR/TAA, Lumen and cloud reconstruction, MRQ sample averaging | single-sample, unlit, no-AA data capture; MRQ is not used for data planes |
| Mirrored frames | left-handed Z-up centimetres against right-handed metres | `U1.3`, tested with an asymmetric target before any render |
| Wrong GPU | Vulkan lists the 5090 first; the 5090 also has 2026 crash reports with UE 5.7/5.8 on Linux | select by name, verify the log line, stay on the A6000 (`U0.2`) |
| Disk | 36 GB free on `/home` | build on a volume with room; check its filesystem is ext4, not NTFS (`U0.2`) |
| Data-material gaps | if C1 swaps materials, world-position offset (foliage wind) and opacity masks must be reproduced, or RGB and data geometry disagree | prefer C2/C4, which use the scene's own materials; a test with masked foliage |
| Version churn | 5.8 → UE6 in 2027; Substrate changed the GBuffer in 5.7 | pin 5.8.3; keep the plugin small; the contract is in Python, not in the plugin |
| Look-tuned weather | a weather parameter left on an artistic default or on wall-clock time makes frames irreproducible and lets RGB and IR disagree about the sky | `U4.3`'s parameter table, the determinism test, and density (never rendered output) crossing to the core |
| Two engines to maintain | every lane change lands twice until `U7.2` | the `EngineSession` conformance suite; Isaac frozen at the switch |
| Licence | seat pricing applies to a company over US$1M revenue | owner to confirm (open question 4) |

---

## Open questions for the owner

| # | question | default if unanswered |
|---|---|---|
| 1 | After parity, is Isaac retired (glue deleted) or kept as a second engine? | frozen: kept, tested, not extended |
| 2 | UE 5.8.3, or stay on the 5.4.3 that is already built and that `HunterSimulator` uses? | 5.8.3, with 5.4.3 as the fallback |
| 3 | Where does the engine build go? `/home` is 97 % full | the 650 GB volume, if it is ext4 |
| 4 | Licence: is the organisation above US$1M annual revenue? | assume not; record the answer |
| 5 | Reuse `HunterSimulator`'s environments and Cosys-AirSim flight for scenes? | environments yes (as unversioned content), Cosys-AirSim no |
| 6 | Distortion: host-side remap of raw planes, or Unreal's lens distortion? | host-side (`U2.4`) |
| 7 | Clouds: does Unreal's cloud field feed the IR band (A), or the core's field drive Unreal's clouds (B)? | (A), decided by the `U4.5` spike |

---

## Verification (when the steps exist)

```bash
make check                                   # the project gate, unchanged: no Unreal, no GPU
make test-unreal PYTHON=...                  # U2.1+: UE automation + @pytest.mark.unreal, on the A6000
python scripts/engine_parity.py configs/scenes/<scene>.yaml   # U2.5+: one config, both engines, a diff report
```
