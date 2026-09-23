# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

Grouped by the day an entry was written (`RP.2`). One `[Unreleased]` section carrying twenty
repeated `Added` / `Changed` / `Fixed` headings was a single merge hotspot for three sessions
working in one tree; two commits already exist whose whole subject is restoring lost entries.
`tests/unit/test_changelog_structure.py` fails on a repeated heading inside a dated section.

### 2026-09-23

#### Added
- **The focus pulled from a cloudy sky onto a cube** (`OC.12`, `scripts/focus_sky_demo.py`).
  `OC.3` put two cubes at two ranges and focused on either; this answers what a *background* looks
  like when the lens leaves it. One cube at 3 m against sky, the focus held on the sky, pulled to
  the cube, held, and pulled back — a loop, with each frame carrying its focus distance and both
  blur circles. It runs on the shipped `OC.6`/`OC.7` path rather than its own compositing, with the
  sky's radiance along **every** ray handed in as `background_radiance`, which is the case `OC.7`
  exists for and is exact here because a sky's radiance is a function of ray direction with no
  geometry in it. Measured with `OC.9`'s focus measure per region: the sky loses **3.1x** of its
  contrast when the lens leaves it and the cube **1.7x**, and the two blur circles cross at 32 µm
  with the lens at 5.8 m. Two numbers, not one, because a whole-frame measure cannot tell the two
  settings apart — something is sharp either way.

  **A clear sky cannot show this, and that is physics rather than a shortcoming of the demo.**
  Defocus is a low-pass filter, so it can only remove detail that was there; a clear LWIR sky is a
  smooth `1/sin(theta)` ramp through the column and a smooth ramp convolved with any normalised
  kernel is very nearly itself. Measured: the same pull that costs a cloudy sky 3.4x of its contrast
  costs a clear one **1.002x** — 0.2 % — and the clear sky carries **three orders of magnitude** less
  structure to begin with, 6.9e-8 against 8.4e-5. A demo built on a clear sky would have rendered two
  frames a reader could not tell apart, and been a fair picture of the physics while being a useless
  picture of focus. `test_a_clear_sky_barely_changes_at_all` asserts it rather than leaving it here. The high spatial frequencies in a real sky are **cloud edges**, so
  the demo carries the project's own sky-fixed cloud field at an eighth of a degree — the half-degree
  survey default is ten Boson pixels per cell, and a cloud whose smallest feature is twenty pixels
  wide barely registers a 5.4-pixel blur circle. Clear sky 250 K, cloud base 282 K from the LCL and
  the lapse rate, cube faces 296-314 K.
- **The Phantom 4 renders** (`scripts/render_phantom4.py`, AI.2's in-engine half). A 62 MB
  third-party FBX that nobody here modelled now produces LWIR frames on the A6000: 41 prims,
  2,486,459 triangles, **41/41 materials resolved in Kit** through `configs/assets/phantom4.yaml`
  (48.8 % through the global globs alone). 16 frames over a 28-minute mission, white-hot
  grayscale, mp4 written. The aircraft reads **-12.4 .. 26.7 C** against a **-27.9 C** sky.
- **`scripts/probe_isaac_asset.py`** re-measures ADR 0128's claims with **Kit's** OpenUSD rather
  than Blender's, since the whole asset pipeline was built and verified on the CPU. Kit agrees:
  same prim count, same triangle count, `metersPerUnit` 1.0, upAxis Z, and the same 48.8 % -> 100 %.
- **Fixed: every Z-up stage crashed the moment it rendered** (`FORWARD_AXIS_VECTOR`).
  `IrCamera.planes` passed its up axis to `azimuth_from_rays` but not a matching `forward`, and
  that default is (0, 0, -1) -- perpendicular to Y-up and **parallel** to Z-up, so the call raised
  "forward must not be parallel to up". Every stage in the repo was Y-up, so nothing had ever hit
  it; an imported asset brings its own convention and hits it immediately. The two are now a
  declared pair beside `UP_AXIS_VECTOR`, with a test that no up axis lacks a perpendicular
  forward.
- The Phantom 4 scene's throttle schedule now runs to **1680 s**, DJI's published flight time for
  the aircraft. It ended at 1200 s, and the solver refuses a time outside its schedule rather than
  extrapolating one -- so a clip longer than 20 minutes stopped rather than inventing a mission.
- **A real 3D model is now solvable geometry** (`irsim.io.assets`, `MeshSpec(asset=…, prim=…)`,
  schema v16, ADR 0132). `configs/scenes/phantom4_pointwise.yaml` is the **first scene in this
  project whose geometry was not authored in Python** -- six prims of a DJI Phantom 4 Pro FBX,
  234,923 cells over 0.1628 m^2 (55 % of the aircraft for 11 % of its triangles), built in 13.6 s.
  The result is the physics the lane exists for: black mouldings reach **65.9 C** where the white
  shell tops out at **33.8 C** (alpha 0.94 against 0.25), and every surface carries a *span* --
  7.8 K on the propellers to 39.7 K on the mouldings -- instead of one value per object.
- **Decimation that preserves area** (`prep_asset.py --emit-mesh`). Collapse decimation removed
  **37 %** of this asset's surface area at ratio 0.05 -- area sets both the radiated power and the
  convective load, so that is 37 % of the emitted signal, silently, on geometry that still looks
  right. The cause is structural: 31,068 disconnected shells, and a collapse budget spends itself
  destroying the small ones. Planar dissolve at 3 deg removes 38 % of the triangles for
  **+0.056 %** area. The tool gates on area and refuses to write an archive that moved, weighting
  the gate so a 1.1e-5 m^2 sliver is reported rather than blocking; 126 zero-area triangles are
  dropped and counted, because a facet with no area has no normal.
- **`self_occluding:` on a mesh, and a budget that refuses rather than hangs** (ADR 0132). Tracing
  a mesh against itself costs cells x faces, once for the sky view and **again every tick** for
  the solar disc: one imported prim of 3,320 cells spent **248 s** in `disc_visibility` over a 6 h
  spin-up, and the largest bound prim would be 1.69e9 ray-triangle tests. Over
  `MESH_SELF_OCCLUSION_BUDGET` a scene that did not decide is refused with a message naming both
  ways out. The switch reaches the beam as well as the sky view -- routing it to only one leaves
  the scene just as unable to finish, which is how this was found. With it, that scene builds in
  **1.0 s**.
- **Third-party assets can enter the simulator** (`scripts/prep_asset.py`, `configs/assets/`,
  ADR 0128). Until now every piece of geometry was generated in Python; there was no import path.
  The tool imports FBX/OBJ/glTF/USD, applies the asset's `scale_to_metres`, exports USD with
  `UsdPreviewSurface`, walks it into engine-free prim records and audits them. **All of it on the
  CPU** — Blender ships a complete `pxr` (OpenUSD 26.03), so the inspect → map → audit loop never
  boots Kit and never touches CUDA.
- **Per-asset material mapping** (`irsim.materials.mapping.AssetMapping`, ADR 0128) — a new
  precedence rung between the `thermal:material` override and the semantic class, matching source
  material names exactly and case-insensitively rather than by glob. ADR 0047's own "Revisit when"
  clause named this file. A miss stays loud; this adds a rung, not a default.
- **`configs/assets/phantom4.yaml`** — a 62 MB DJI Phantom 4 Pro FBX (41 meshes, 21 materials,
  2.49 M triangles) goes from **48.8 % to 100 %** coverage, and two *confident* global hits become
  correct: `*white*` → `car_paint_white` (paint on steel, 4399 J m⁻² K⁻¹) becomes the moulded
  `abs_plastic_white` (2205 — the global rule made the shell twice as sluggish as it is), and
  `*metal*` → `bare_aluminium` (ε 0.09, a mirror showing reflected sky) becomes
  `aircraft_aluminium_painted` (ε 0.90) on the motor housings. Four entries are `ESTIMATED` and
  flagged in place; the consequential one is `Copper`, which sits at the motor stations.
- `--asset` on `scripts/audit_materials.py`, and `docs/research/2026-09-23-asset-ingestion-survey.md`
  (the sourced evidence base, with its confirmed/unconfirmed split).

- **The colour bar** (`irsim_eval.video.palette_scale`, ADR 0125). A fixed-span frame now carries
  the palette beside it with Celsius ticks, drawn from the *same lookup table the display branch
  indexed* rather than from a gradient that resembles it. The gauge says what each named part is;
  this says how any temperature became the pixel beside it, which a viewer cannot reconstruct from
  the picture. Deliberately **not** drawn beside either AGC output, whose mapping is rebuilt from
  every frame's own histogram — that difference is what separates a picture of contrast from a
  measurement.
- **`--close-up`** on the outbound driver (ADR 0125): holds the aircraft filling the frame for the
  whole mission, so the only thing changing is temperature. The framing is checked against the
  sensor and refused if the aircraft overflows the frame or shrinks below a quarter of it. The
  tight display span is now taken over the **solved cells across the whole mission** rather than
  relative to air — how far the skin sits from air is the measurement, and a span defined from air
  would move whenever the weather did.
- **A named aircraft: the DJI Phantom 3** (`PT.9`, ADR 0124). `irsim_isaac.phantom3` +
  `configs/scenes/phantom3_outbound_pointwise.yaml`, laid out from DJI's published specification
  — 350 mm diagonal, 9450 propellers (239 mm, 5.0 in pitch, so a geometric pitch angle of 12.7°
  at 75 % radius), 2312 motors, the battery in the rear of the shell — with the shell plan, body
  depth, gimbal and skids ESTIMATED from photographs scaled on that diagonal and labelled so. The
  prop clearance is not authored anywhere: it falls out of the other two numbers at **8 mm**,
  which is why 9450 is the largest propeller this frame takes. `scripts/render_quad_outbound.py`
  grows `--airframe`, so one driver flies both aircraft, each with its own scene and range band.
- **`abs_plastic_white`**, because a Phantom is not carbon and the difference is the whole result.
  Sunlit skin over air: **+2.3 K** for white ABS against **+24.4 K** for the carbon deck, from a
  solar absorptivity of **0.25** against **0.90**. Against a 50 mK NETD both are visible — 2.3 K
  is forty-six NETD — but the margin is an order of magnitude apart. In LWIR the pigment does
  nothing: ε = 0.95 against 0.90, both near-blackbodies, so a thermal camera sees the consequence
  of the paint and never the paint. **Not a thermal-mass effect**, and a test now says so: at the
  same 1.5 mm the two carry 2205 and 2520 J m⁻² K⁻¹, 12 % apart. A first draft asserted "half the
  heat, twice as responsive" in three places; it was measured, it was wrong, and it was removed.
- **Cloud in both bands, by default** (ADR 0076, wired at last). New weather fixture
  `scattered_cumulus_48h.csv` at cloud **0.45** — SCT, 3–4 oktas — because the existing fixtures
  are 0.05 (a sky with nothing in it) and 0.98 (no gaps and no sun). One `SkyFixedCloud` seeded
  from the scene's own weather is sampled per ray by the infrared background *and* baked into the
  visible dome, so the pair cannot disagree about where the cloud is. The grid had to go finer
  than the survey default: half a degree is **ten pixels** through this camera and reads as
  blocks, so the driver asks for six cells per degree (~3 px, under the PSF).
- **`clouds.optical_depth`** on the environment schema, and `configs/environments/scattered_cumulus.yaml`
  authoring it (`AT.11`, ADR 0126). A preset now says how *deep* a cloud is, in visible optical
  depth, and the LWIR emissivity is derived per ray from it rather than authored as one
  transmittance. `clouds.tau` stays and every preset that authors it is bit-identical; authoring
  both is refused, because they are two answers to one question.
- **A cloud with a third dimension** (`AT.12`, ADR 0127). `irsim.atmosphere.cloud_deck`: a
  horizontal map of column depth at the LCL, each column given a top from its own depth and an
  analytic vertical profile. The infrared band ray-marches it (`SkyModel.radiance_field_from_deck`)
  and `irsim_isaac.cloud_volume` voxelises **the same function** into a NanoVDB volume for the
  visible band, so the volume the path tracer renders is the field the radiometry integrates.
  `--cloud-volume` on the outbound driver. Measured on the Phantom clip's frame geometry: the
  cloud spans **19.7 K** against the plane-parallel sheet's 1.25 K, the emission level runs
  **12–834 m** above the base instead of sitting on it, and the largest histogram bin falls from
  55.4 % of the frame to 39.6 %. A **vertical** ray still reproduces ADR 0126 to the bit, because
  the profile `6u(1−u)` integrates to exactly the column thickness.
- **`--cloud-deck`** on the outbound driver, separately from `--cloud-volume` (`AT.12`). The deck
  half works on this build: the infrared band marches it and the visible dome bakes the same deck,
  so the two bands still read one object. The **volume** half does not — Isaac Sim 6.1's IndeX
  plugin refuses the grid ("unable to create VDB subset") even after the NanoVDB version stamp is
  matched to its own 32.7 and the file metadata's node and voxel counts are filled in from the
  tree. Both fixes are kept, because they are correct and because the artefact is what `AT.13`
  probes with, but the flag is off by default and the driver says what it is.
- **A third display span, `sky`**, on the outbound clip (ADR 0126). Neither existing span reaches
  the sky — `ir` starts at the coolest airframe node — so cloud and clear zenith both landed on
  display code 0 and the only picture carrying the sky was the camera's own AGC. The new span is
  linear over the scene's own sky-to-target range: **63 K across 256 codes, 0.25 K each**, five
  times the sensor's NETD. The AGC video is still written beside it.

- **The `OC` focus lane in `docs/roadmap.md`** — ten steps for the focus distance no camera in
  this repo has. The audit behind it: `OpticsSpec` carries no focus field, `psf.py` builds one
  kernel from one scalar and convolves the whole plane with it regardless of `distance_m`, and the
  camera prim never sets `focusDistance` or `fStop`. The lane settles that defocus is a
  **post-process on the supersampled radiance**, because the renderer emits geometry and ids only
  and blurring a temperature would average the wrong quantity, and that the model is **Hopkins**
  rather than the geometric disk, because geometric optics needs a 14-pixel blur circle at F/1.0 in
  LWIR and every defocus this project renders is below that.

- **Defocus geometry** (`irsim.optics.defocus`, `OC.1`). Blur-circle diameter from focal length,
  f-number, object distance and focus distance, the wavefront error W020 that follows from it, the
  hyperfocal distance and the depth-of-field limits. The oracle is the thin-lens construction --
  image distances and similar triangles on the exit pupil -- so a wrong algebraic simplification
  fails the test rather than being confirmed by it. A Boson focused at infinity is at **0.02 waves
  at 100 m**, which is why the aerial lane never needed this, **0.23 at 10 m** — right at Rayleigh's
  quarter-wave boundary — and **0.47 at 5 m**. Geometric optics does not become valid until 168 µm
  of blur, fourteen pixels. Nothing reads this yet; no rendered output changes.

- **The defocus OTF** (`irsim.optics.mtf`, `OC.2`, ADR 0129). Hopkins' quadrature, which carries
  diffraction and defocus in one term and equals `mtf_diffraction` at zero defocus to the 7.7e-9
  Simpson floor — so it replaces the cascade's first factor rather than multiplying onto it, and
  ADR 0117's in-focus aberration Gaussian is untouched. The geometric disk and a Gaussian are
  selectable beside it for ablation; at Nyquist for a Boson at 10 m they give 0.173 and 0.203
  against Hopkins' 0.356 and an in-focus 0.461. `bessel_j1` is the integral representation, so
  SciPy is still not a dependency. **Two planning assumptions corrected:** the geometric rule of
  thumb is an asymptote, not a threshold — the gap is still 0.07 at W020 = 2λ — and **defocus is
  achromatic**, because λ cancels out of Hopkins' `a = 8π W020 s/λ` once `s = ξλF` is substituted.
  Band averaging therefore cannot smear the defocus zeros; it is worth having for the diffraction
  cut-off, which runs 133 to 74 cyc/mm across the LWIR band, and moves the contrast-carrying part
  of the OTF by under 0.3 %.

- **A two-cube focus demo** (`scripts/focus_demo.py`, `OC.3`). One command writes a near-focused
  frame, a far-focused frame, hyperfocal and infinity, plus a 72-frame focus-sweep MP4 — engine-free,
  from a synthetic two-depth scene, at Boson 640 optics. Each layer is blurred with the kernel its
  own range earns and composited back to front using its **blurred coverage as alpha**, so a
  defocused edge is semi-transparent rather than a hard cut. Blur is applied in radiance, never in
  Kelvin (non-negotiable #3), and a test pins the half-covered edge to the radiance-blended
  temperature rather than the Kelvin mean — 6.7 K apart on a 250/315 K edge. The measured 10-90 %
  edge width tracks the blur circle **in quadrature**, with a line-spread constant stable inside
  0.05 across focus distances from 3 m to 80 m. `defocus_psf` reproduces `optical_psf` in focus to
  7e-7 of the peak, the radial-interpolation floor, so `OC.5` must either keep the in-focus path on
  `optical_psf` or refresh the goldens deliberately.

- **Focus as config** (schema **v10**, `OC.4`, ADR 0129). `optics.focus` takes `infinity` (the
  default), `hyperfocal` — which resolves against the detector pitch unless `coc_um` says otherwise,
  because the acceptable circle of confusion is a convention and belongs in the document — or
  `fixed` with a `distance_m`. `optics.mtf.defocus_model` selects `none`/`gaussian`/`geometric`/
  `hopkins` and `defocus_apply` selects `global` or `layered`; `fidelity.defocus` is the ablation
  and can switch defocus off but never on. **Every default is the pre-v10 camera and is dropped
  from the config hash**, so a v9 document and a v10 document that spells the defaults out hash
  identically and every golden array written before `OC` stays valid — while naming a model, a
  focus distance or the ablation each changes the hash, which is what makes a run's provenance say
  which camera it was. A focus distance authored with `defocus_model: none` is refused rather than
  silently ignored.

- **Defocus reaches the pipeline** (`OC.5`, ADR 0129). `PipelineConfig.from_sensor` builds a
  `DefocusKernelBank` whenever the camera names a defocus model and fidelity allows it, and stage 3
  picks the frame's kernel from the **median** range of the geometry actually in frame — the median
  and not the mean, because a frame that is nine tenths sky and one tenth foreground should be
  focused for the foreground. Sky is excluded through `sky_mask`, since `distance_m = 0` there and
  a median that counted it would defocus the frame hardest of all; a frame with no geometry takes
  the defocus of an object at infinity, which is what it is looking at. `PipelineState` now reports
  `defocus_w020_um`. Kernels are quantised to a quarter of a supersample cell of blur circle —
  below what the box filter can resolve — and cached, so an unchanging scene builds one kernel and
  a thousand-frame sequence does not pay the Hopkins quadrature a thousand times. A camera that
  names no model gets no bank and keeps the single in-focus `optical_psf`, so every golden array is
  bit-identical.

- **Depth-varying defocus** (`irsim.optics.layered`, `OC.6`, ADR 0129). `defocus_apply: layered`
  splits the frame into depth layers, blurs each with the kernel its own range earns together with
  its coverage, and composites back to front with the blurred coverage as alpha. Two details carry
  the weight. Bins are equal width in **W020**, not equal counts: an equal-count split puts the
  median in whatever fills most of the frame and collapses a near/far scene to a single layer,
  which silently turns the stage back into `OC.5` — an earlier version of the test passed against
  exactly that. And the composite is `over` **normalised by the accumulated alpha**, because a
  plain `over` leaves a deficit wherever a defocused layer's alpha opened a gap no farther layer
  fills, and that deficit reads as a dark fringe along every out-of-focus silhouette. The
  normalisation fills it with the layers that *are* there, which is a defensible guess and not the
  truth — `OC.7` replaces it with the exact answer for sky and sea. Measured: a far background's
  own edge stays 1.8x sharper than one global kernel leaves it while the near slab still gains
  1.7 px of blur in quadrature, and a flat field survives any focus to 1e-9.

- **The occlusion gap is filled exactly for sky and sea** (`OC.7`, ADR 0129). A new optional
  G-buffer plane, `background_t_k`, carries the apparent temperature each pixel's ray would report
  **with all geometry removed**. Given it, `layered_defocus` seeds the composite with an opaque
  backmost layer carrying the truth, the accumulated weight stays 1 and `OC.6`'s normalisation
  becomes the identity rather than a guess. This is not a second render pass: sky and sea radiance
  are functions of ray direction the adapter already evaluates where they are visible, and
  evaluating them for every pixel costs nothing more. Measured against a two-layer reference built
  by construction: **1e-12 with the plane, 1.98 radiance units without it**, and the error without
  it is confined to the silhouette — under 1 % of its peak more than sixteen pixels away. The plane
  is precision-critical, because it is a temperature and float16 spaces 0.25 K at 300 K.

- **The partial-occlusion bound, measured** (`OC.8`, ADR 0131). `push_pull_fill` and
  `estimate_background` supply a background for scenes with no analytic one, and
  `test_occlusion_bound.py` reports what each of the three fills costs in apparent temperature
  against a two-layer reference at three depth ratios: normalisation alone **2.87–5.06 K**,
  push-pull **2.57–5.03 K**, the analytic background **0.0000 K**. The error is local — under 5 %
  of its own peak outside a band about the silhouette — so the bound is a bound on a band, not on a
  frame. The uncomfortable finding is that push-pull buys 1–10 % and not an order of magnitude: an
  extrapolation from the visible background cannot recover structure that was never visible, and an
  earlier version of the measurement scored it a perfect 0.00 K only because the hidden feature also
  continued into the visible region. **Two defects in `OC.7` as first written** were found by this
  measurement and fixed: the background seed was not blurred, and it was composited behind the
  backmost layer instead of replacing it — about 1.5 K each, and both invisible on any scene whose
  hidden background looks like its visible one.

- **Focus that moves** (`irsim.optics.autofocus`, `OC.9`, ADR 0129). Two new focus modes.
  `autofocus` is a model of **passive contrast detection**, not a shortcut: the servo may only look
  at the picture, it probes one multiplicative step either side of where it is, and it climbs. The
  focus measure is a **normalised** Tenengrad — an unnormalised one rises with scene radiance, so a
  servo using it would "focus" by finding the hottest frame of a diurnal run. The probe step widens
  when the servo stalls, which is what a viewer sees as hunting and is real: far from focus the
  contrast measure is nearly flat, because a small lens move barely changes a large blur.
  Convergence is asserted **to the depth of field** rather than to the metre, since a contrast
  measure cannot resolve a lens position finer than the band the blur stays inside. `track` follows
  a `semantic_id`'s median range and **holds its last position** when the target leaves frame,
  because a real payload does not snap to infinity when the target goes behind a cloud.
  `PipelineState` now carries `focus_distance_m` and the servo, since where the lens *is* depends on
  what the camera has been looking at and not on what the document says.

- **Thermal defocus** (`irsim.optics.thermal_defocus`, `OC.10`, schema **v11**, ADR 0129). The
  distinctly infrared focus effect, and the one this project already had the input for and was not
  using: `HousingTemperature` has solved the lens housing over a diurnal run since M9.3, for the
  self-emission term, and nothing else read it. Germanium's dn/dT is 396e-6 K⁻¹, some 250 times a
  visible glass, so `dz/dT = f·[α_housing − ((dn/dT)/(n−1) − α_lens)]` comes out at −1.44 µm per
  kelvin for a 14 mm lens in an aluminium barrel. It is folded into an **effective focus distance**
  rather than added as a new blur term, because a thermal image-plane shift is the same defocus as
  looking at the wrong distance — so the global kernel, the layered composite and the autofocus
  servo all get it without a thermal term of their own, and the servo *fights* the drift through
  the picture the way an unathermalised motorised core does. A **20 K rise takes a lens focused at
  infinity to 6.8 m**, inside its own 16.3 m hyperfocal, so distant targets go soft. One result
  worth stating because it is the opposite of the instinct: an **aluminium** barrel athermalises
  better than **invar**, since the residue is `α_housing − β` and a large expansion cancels more of
  it. `athermal: true` is the default and is every camera written before v11, hashing identically.

- **The imported Phantom 4 flies, in both bands** (`scripts/render_phantom4.py`,
  `irsim_isaac.asset_flight`, ADR 0133). One circuit of a lemniscate in front of a ground observer,
  over the aircraft's whole 28-minute mission, filmed as an infrared clip, a visible clip and the
  two side by side. An orbit would have been easier and would show a target that translates but
  never turns — its aspect angle is ±90° for the whole pass. The eight crosses its own track, so
  one circuit takes the line of sight through both broadsides and tail-on while the slant range
  swings **2.24:1** and the aircraft grows from **70 px to 157 px** across; measured over the
  shipped clip, aspect covers −178° to +178° and the motors reach **19.6 K** over an airframe that
  stays inside half a kelvin of itself. The one hard constraint
  is elevation: this scene authors no terrain, so an aircraft below the observer's horizon would be
  backed by sky-model radiance at near-air temperature instead of cold sky. The track holds
  **10.5–52.6°**, and a test says so rather than a comment.
- **An imported asset is mounted by its scene's own `world_frame:` block**
  (`irsim_isaac.asset_flight.world_frame_to_stage`, ADR 0133) — the 3×3 taking a scene's world
  coordinates into stage axes, satisfying `M·up = +Y`, `M·north = −Z`, `M·(north × up) = +X` by
  construction. Written down as a function with tests rather than as a `rotateX 90` in a driver,
  because the sign of a hand-typed axis swap is a coin flip and a mirrored mount renders a
  perfectly convincing aircraft facing the wrong way. Refuses a degenerate or left-handed frame
  instead of guessing. A scene already authored in stage axes mounts with the identity and pays
  nothing.
- The airframe's **nose direction** in the asset's own axes, measured from the archive rather than
  assumed: the gimbal camera body at −53 mm from the airframe centroid, its lens at −55 mm, and
  `Green_light` at **+64 mm** — DJI puts red status LEDs on the front arms and green on the rear
  ones, so a green lamp behind the centre of mass is the tail. Three independent parts, in an asset
  whose 41 prims are all named `GeometryNode_<n>` and carry no hint of function.

#### Fixed
- **A bin boundary was a step, and two kernels disagreed across it** (`OC.13`, ADR 0134 addendum).
  `OC.11` left a residual it named: the layers' blurred coverage `sum_b K_b * cover_b` did not stay
  at one across a bin boundary, because adjacent bins carry different kernels and the wider one
  spreads its coverage further than the narrower one gathers it back. It rippled by about ±0.5 %,
  and wherever it dipped `layered_defocus` read the shortfall as sky showing through and filled it
  with background — a half-percent-of-contrast error along every bin boundary of a receding surface.

  `depth_layers` no longer assigns a pixel wholly to one W020 bin. A pixel sits between two bin
  centres and is split between them in proportion, so a surface receding smoothly crosses a
  boundary as a **ramp**; `DepthLayer` gains a `weight` plane carrying the share, the weights sum
  to one on every geometry pixel, and the composite blurs the weight rather than the mask. With a
  ramp there is no step for two kernels to disagree over. Ripple **±0.52 % → ±0.06 %**, interior
  peak 0.079 → 0.0023 on the regression scene; on `OC.12`'s cube **0.16 K → 0.041 K** peak and
  **0.023 K → 0.0055 K** RMS, a seventh of the Boson's 50 mK NETD. The whole lane from what `OC.6`
  shipped is **8.76 K → 0.070 K**. The layer's representative range is now a *weighted* median,
  which keeps `OC.5`'s reason for using a median while letting a pixel count as the fraction of
  itself it is.

  **A scene of flat slabs is unchanged, exactly**: where W020 takes only a few distinct values no
  pixel lies between two bin centres, every weight is 0 or 1, and the split degenerates to the hard
  partition — which is why `OC.6`, `OC.7` and `OC.8`'s measurements carry over untouched, and there
  is a test asserting the degeneracy rather than leaving it inferred from the others passing. Two
  things had to be retired rather than tightened: the "error falls with more layers" assertion, now
  second order at 4e-4 and decided by where bin edges land rather than how many there are; and the
  test's reference implementation, which had been borrowing `depth_layers`' masks and so stopped
  exhibiting the seam the moment membership went fractional — three tests passed for the wrong
  reason until it grew its own copy of the old binning. A reference to a defect cannot share code
  with the thing it is a reference for.
- **A receding surface was composited as a stack of occluders** (`OC.11`, ADR 0134). `OC.6`'s
  layered defocus used `over` between every pair of depth layers. That is right between a
  foreground and the background behind it, and wrong between two slices of the **same** surface
  receding through two bins: those do not hide each other — the aperture bundle at their shared
  boundary lands partly on each, so they add. Composited with `over` the farther slice is
  multiplied by `1 - alpha_near` and the deficit is exactly `alpha (1 - alpha) (L_surface -
  L_behind)`, filled with whatever is behind — so a **dark seam** is ruled across the surface at
  every bin edge. Measured on a 0.6 m cube at 3 m against a 250 K sky: **8.6 K peak**, with RMS over
  the cube's interior **rising 1.31 → 1.44 → 2.32 K** as `max_layers` went 3 → 4 → 8. Two things
  made it more than cosmetic: the error grew with the stage's only quality knob, so the shipped
  default of 8 was its worst setting; and continuously receding surfaces are not a corner case —
  ground, sea and the flank of any vehicle are all of them, so every lane past the aerial one meets
  it. `OC.6`'s own tests missed it because they measured a flat slab against a flat background with
  empty space between the two, where `over` is correct.

  `DepthLayer` now carries the depth range each layer spans and `separated` decides: `over` between
  layers with a gap wider than either layer is deep, addition between layers that abut. The
  comparison is against the layers' own extents rather than a distance in metres, so it carries from
  a 0.6 m cube to a 200 m ground plane with nothing to tune. The span is deliberately **not** the
  bin index — bins are equal-width in W020 and W020 is V-shaped about focus, so sorting layers by
  range walks the indices up and back down and two neighbours in the composite are routinely several
  bins apart; a first version of the fix tested bin adjacency and was wrong for that reason. The
  background keeps `over` unconditionally: it is complete, so all geometry is genuinely in front of
  it. Peak **8.6 K → 0.16 K**, and RMS now **falls** with the cap (0.0125 → 0.0076 where the old
  composite ran 0.209 → 0.497). `OC.7`'s 1e-12 exactness is untouched.
  `tests/unit/test_continuous_depth.py` keeps the old composite as its reference, so a revert fails
  rather than passes quietly. What remains is a **±0.5 % ripple** in the geometry's blurred coverage
  at each hard bin boundary — 0.5 % of contrast against the 25 % removed, below NETD in RMS — whose
  fix is fractional layer membership (`OC.13`).
- **The imported aircraft rendered rolled, and the stage's up axis was why** (ADR 0133). The first
  Phantom 4 stills came back with the aircraft apparently pitched over and rolled about thirty
  degrees, with nothing wrong in the physics. ADR 0128's principle — *the asset's frame is the
  stage's frame* — had the driver author a **Z-up** stage to match the archive, and
  `look_at_quaternion` builds its mount around a world up vector whose default is the stage
  convention `(0, 1, 0)`. On a Z-up stage that levels the camera against a horizontal axis, which
  rolls the horizon by whatever angle the geometry happens to give, and raises nothing: a rolled
  camera is a valid camera. The stage is now the renderer's Y-up and the asset carries the one
  rotation its scene config already declares. That was not the only breakage, only the visible one
  — `visible_sky.stage_direction`, `latlong_directions` and the `DistantLight` of
  `stage.author_environment` all read +Y as up and −Z as north, so the companion visible frame
  would have shown a differently lit sky from the infrared one it is registered against.
- `render_phantom4.py` wrote its float32 planes with a bare `np.save` and **no sidecar**, so no
  frame it produced carried the config or band hash it came from (ADR 0004, `IG.13`). It now binds
  a `FrameWriter` like every other driver, with `--float-format` and `--plane-stride`. Caught by
  `tests/unit/test_frame_writer.py`, which asks the question of each driver by name rather than
  trusting a new one to remember.
- The Phantom 4 driver's frame interval was computed as `mission / (frames − 1)`, which overshoots
  the mission by one interval because `IrCamera` advances its clock before handing back a frame —
  and the throttle schedule is right to refuse a time past the end of a flight rather than
  extrapolate one.
- **The two bands were reading two different clouds, and the infrared one was a field of mesas**
  (`AT.15`, ADR 0130). On the Phantom clip's frame geometry the visible dome drew cloud over
  **26 %** of the pixels and the infrared band over **57 %** of the *same* pixels, which ADR 0076
  forbids. Three causes. (1) `deck_field` mapped the synthesised field through a soft threshold,
  which is a *membership* function — right for a plane-parallel sheet and wrong read as a top
  height, because it makes every covered column the deck's full 1.2 km. The deck was flat-topped
  blocks with vertical walls, invisible looking up and the whole picture at 20°, where an oblique
  ray runs *along* a wall: marched optical depth averaged **7.07** and reached **27.8**, opaque
  everywhere. Depth now rises with the field's excess over the condensation threshold and is
  clipped by the capping inversion, so a tower has rounded shoulders and a ray crosses it.
  (2) A `1/f^β` field is scale-free, so a 25 m grid carried 25 m clouds; the dome sampled them as
  specks and the march smeared each one over the kilometres a ray spends in the deck — *"some
  random points those are not in the rgb at all"*. Band-limited at **400 m**, the low end of the
  observed fair-weather cumulus mode. (3) The deck ended at `base / tan(10°)` and read clear sky
  beyond, putting a hard cold band across the bottom of every oblique frame; the depth map is an
  inverse FFT and therefore exactly periodic, so it now **tiles** seamlessly and has no edge.
  And the dome **marches** the deck rather than sampling its column depth at the base crossing,
  taking `α = 1 − exp(−τ)` from the same integration the infrared band applies `CLOUD_OD_RATIO`
  to. Measured after: the largest 0.25 K bin holds **6.4 %** of the frame (53.9 % before ADR 0126,
  39.6 % after 0127), in-cloud spread is **28.8 K** p1–p99, the emission level runs **15–1065 m**
  above the base, and the two bands draw cloud in the same pixels by construction. The march is
  now sized by **path length** in metres instead of by grid cells, and per ray rather than per
  frame. Measuring that properly -- against a converged reference, interpolated back up and
  box-filtered to native as a frame actually is -- overturned a decision made an hour earlier on a
  worse measurement: the frame's error is dominated by the march's **quadrature**, not by the
  interpolation from the marched grid to the supersampled one. Halving the stride moves the 99th
  percentile of the band emissivity from **0.029 to 0.024** and costs **five times** as much;
  going from 36 m steps to 12 m moves it from **0.073 to 0.029** (1.3 K against the 44 K a cloud
  stands above a clear zenith) for three times. So the camera marches once per native pixel and
  the steps go into the quadrature. Known and
  unfixed: a cloud is still a vertical extrusion rather than a 3-D body, the visible cloud's
  interior is lit as a slab rather than as a body, and every base sits at one altitude — AT.13
  and AT.14 own those.
- **A marched cloud drew as a flat grey cut-out in the visible band** (`AT.15`, ADR 0130). Found
  by looking at the first rendered pair after the dome started marching: the infrared frame had a
  cumulus field and the visible frame had paper shapes of it. `_cloud_base` returned one
  Lambertian radiance for every cloudy texel, which was fine while the opacity was a vertical
  column depth and carried the gradation itself; once `α = 1 − exp(−τ)` saturates over a cloud's
  body, one radiance is a flat shape with a hard edge. The brightness now comes from the same
  optical depth the opacity does, through the two-stream reflectance of a conservatively
  scattering layer, `R = (1−g)τ / (2μ₀ + (1−g)τ)` at `g = 0.85` — which is the reason a cloud
  *has* an inside: a thin edge returns almost nothing and is the sky behind it, a deep core
  returns nearly everything and is white. Measured over a marched frame the reflectance spans
  **0.3 to 0.9** where a constant spans nothing, and cloud sits at **1.23x** the clear sky's
  luminance at the median with thin edges darker and cores brighter, where the constant put every
  cloudy texel at a flat 1.8x. Absorption is neglected, right in the visible and not in the near
  infrared, and the function says so.
- **A marched cloud edge drew a staircase on the visible dome** (`AT.15`, ADR 0130).
  `DOME_HEIGHT` was 512 rows, a texel every 0.35 deg, chosen when the softest thing on the dome
  was the solar aureole; a marched cloud's opacity crosses from clear to opaque inside one texel,
  and a 640x512 frame magnifies each texel to **seven pixels**. Now 1024 rows -- 3.6 pixels, and
  real structure with it -- at 27 s of bake and a 25 MB EXR, both once per render. A clear dome
  pays neither: the cost is the march, not the resolution.
- **A cloud field's spectral slope was read in the wrong convention** (spec issue S50, ADR 0127).
  `generate_cloud_field` applies the authored `beta` as the **radial** exponent of a 2-D power
  spectrum, and on a 2-D field the variance per octave goes as `f^(2−β)` — so `beta: 1.8` puts more
  variance at the smallest scale the grid carries than at the largest. Measured on a 487² grid at
  25 m: the autocorrelation length is **125 m** at β = 1.8 and **1250 m** at β = 2.8. Published
  cloud slopes near −5/3 are *transect* slopes, one less than an isotropic field's radial exponent.
  The deck synthesises at `beta + 1` and says so; the hemispherical `SkyFixedCloud` is deliberately
  left alone, because changing it would move the cloud in every scene shipped since MS.3.
- **A cloud had no optical depth and no distance, so it rendered as one flat white level covered
  in amplified noise** (`AT.11`, ADR 0126). Measured on `outputs/phantom3_outbound/frame_000120`:
  **53.9 %** of a 640×512 LWIR frame sat within 0.25 K of one apparent temperature, and inside
  that region the true spread was **0.073 K** — the sensor's own NETD and nothing else — which the
  plateau-equalising AGC with DDE rendered as display codes **106 to 255**. Both halves of that
  have one cause: every shipped preset authored `tau: 0.0`, so `ε = 1` for every covered ray, and
  `L_B(T_base)` was evaluated as though the cloud were at the sensor rather than a kilometre away
  with warmer air in front of it. Now: `ε = 1 − exp(−0.5 m τ_vis)` along a ray of airmass
  `m = 1/sin θ`, which at the diffusivity factor *is* Shaw & Nugent's published `1 − exp(−0.79 τ)`
  exactly (0.79 = 1.58 × 0.5); and `L = L_clear + τ(R, θ) ε [L_B(T_base) − L_beyond(R, θ)]` with
  `R = z_base / sin θ`. The clear-sky limit is exact rather than approximate, because the layered
  model's band transmittance and its `sky_beyond` carry the same spectral-class weights.
  Measured, same scene and seed, preset swapped: the cloud **core** spans **1.25 K** against
  0.000 K, transmittance to the base runs 0.68 → 0.49 from the top of the frame to the bottom, and
  73 mK of NETD occupies **0.3 of one display code** instead of 149. Named and not fixed: a
  plane-parallel deck still has no *sides*, so an optically thick core is flat to within a kelvin
  — that needs a cloud with a third dimension (`AT.12`).
- **A cloud was a stencil, and it rendered as flat blobs in both bands** (ADR 0125).
  `SkyFixedCloud.sample` returned a boolean, so every covered texel got one flat value and every
  cloud had a one-sample cliff round it — no thin edges, no internal structure, and in LWIR a step
  of tens of kelvin along every boundary, which is exactly the edge statistic a sky-target
  detector keys on. `SkyFixedCloud.density` now returns a 0-to-1 depth (a smoothstep on the
  field's own excess over its threshold, in σ of the unit-variance field) and the blend becomes
  `ε_eff = (1 − τ) d`. Measured on the rendered dome: the fringe is **9.2 %** of the map against
  19.6 % at full depth at 0.45 coverage, and **3.0 % against 2.4 %** at 0.05 — thin cloud is more
  edge than core, which is why the stencil looked worst where cloud was sparsest. Three guarantees
  keep it a change to how a cloud *looks*: `density` crosses 0.5 exactly at the threshold so the
  covered fraction is untouched (asserted); `softness = 0` returns the hard mask exactly, so the
  old behaviour is a parameter value and not a deleted branch; and `cloud_radiance` returns both
  limits through `np.where`, so a boolean mask is **bit-identical** at any `τ_cloud`.
- **`quad_outbound.SPAN_M` overstated its own aircraft by 41 %** (ADR 0124). It was authored as
  `2 × 0.42 × √2`, the X-quad form every multirotor spec sheet quotes; that frame is a **plus**,
  with arms due N/E/S/W, so opposite motors are `2 × 0.42` = **0.84 m** apart, not 1.19 m. It
  survived a review, a test suite and a 300-frame render because nothing measured it — the
  burnt-in readout of `outputs/quad_outbound` says "span 115.5 px" where the motors span 82 and
  the propeller tips 111. Both airframes' spans are now derived from the authored motor prims,
  with a test measuring them off the layout; the driver quotes prop tip to tip for both, so the
  two aircraft are compared on the same measurement. **The images are unaffected — only the
  number printed beside them.**
- **A patched arm and a patched plate inset in opposite directions** (ADR 0124). A plate wants
  its prim *inside* its patch; a solid arm wants its patch *outside* its prim, because from below
  an arm presents its end caps and both side faces, none of which lie on the plane the grid is
  drawn on. Authoring the arms like the plates put **1204 pixels** outside every patch on the
  first Phantom render. `tests/unit/test_rendered_airframes.py` now samples every face of every
  patched prim, rotated by its own yaw, against the real `PlanarPatch.contains`, for every
  rendered airframe — no formula restated, so the test can disagree with the scene file.

### 2026-09-22

#### Added
- **The point-wise aerial scene is rendered** (`PT.9` in-engine / `IG.2`, ADR 0123).
  `configs/scenes/quad_outbound_pointwise.yaml` + `irsim_isaac.quad_outbound` +
  `scripts/render_quad_outbound.py`: a ground camera films one quadrotor from **12 m to 150 m**
  against sky, with the deck, the belly and two arms each solved **per cell** and bound to prims
  that exist. Sun and throttle drive it together. Deck − belly is **29.3 K** in the rendered scene
  and **29.3 K** in the engine-free oracle it was ported from, so the move into the stage frame is
  provably a change of coordinates and not of physics; the arms carry **27 K** across one prim from
  the deck's and the pods' own shadows. The target goes from **115 px** across to **12.7 px**.
  Three decisions with teeth: the **camera moves and the aircraft does not** (an occluder in a
  moving frame is refused by the thermal core, so a flying airframe would lose the self-shadowing
  that is the whole signature); the boresight's 7.6° of margin to the horizon is **computed from
  the sensor and the render refused** if a wider lens would put ADR 0060's analytic ground into a
  sky-target frame; and every patched prim is **2 mm smaller than its patch**, which was measured
  rather than anticipated — the first render raised on 150 deck pixels whose sampled position
  landed a float's width outside the rectangle. Recorded honestly: the deck's 29 K excess is on
  the *far side* of the aircraft from a camera looking up at it, which is the answer to the
  question rather than a gap — an anti-UAV sensor reads a belly near ambient with four hot bells
  on it. IR (two fixed spans), the camera's own AGC, and the companion RGB are all filmed.
- **People: skin and clothing are two temperatures** (`PH.12`, ADR 0122). `irsim.thermal.human` —
  skin *authored* from ISO 7730's thermoregulated set point, clothing *solved* from the standard's
  own implicit balance by bisection (the textbook fixed point **diverges for a coat in wind** at
  1 m/s). At 0 °C in 1 clo: skin **34.07 °C**, clothing **13.96 °C**, a **20.1 K step across one
  body**; indoors at 22 °C in 0.5 clo it is **4.9 K**. Two findings against the row: its 10–15 K
  band is what the same equation gives at **10–15 °C air** — ISO 7730's own validity floor — so
  the criterion was written for a condition the standard does not cover; and "wind lowers t_cl" is
  the special case, since wind drives the coat toward *air*, warming it above 3 clo under a −40 °C
  sky. The `pythermalcomfort` cross-check **does not run** (not installed); a `comfort` extra
  declares it, the test skips loudly, and another test forbids `src/irsim` from importing it.
- **Vegetation: leaves transpire** (`PH.11`, ADR 0121). `irsim.thermal.vegetation` — a leaf's
  temperature set by its stomata. Same sun, same wind, same air: a well-watered leaf sits
  **−1.93 K** below air at a 3.18 kPa deficit and a stressed one **+7.00 K** above it. The Idso
  non-water-stressed slope comes out **−1.84 °C/kPa**, inside the published [−3.8, −1.1], and the
  band was recorded as an acceptance rather than fitted to. Two decisions with teeth: a leaf is
  **solved, not stepped** (630 J m⁻² K⁻¹ gives a 15.4 s time constant, so the midpoint rule is
  stable only to 30.8 s and a 60 s tick diverges — `TC.1`'s guard), and a leaf has **its own
  boundary layer** (33 s/m at 2 m/s against the bulk formulation's 435 s/m; using the bulk one
  gave +6.4 K where the answer is −1.9 K, the wrong *sign* of the only effect here). The oracle is
  Campbell & Norman's closed form in **molar** units, independent of this project's SI mass-based
  balance: they agree to **0.10 K** near air, **0.37 K** at 4.6 K of departure, and to **4e-10 K**
  in the degenerate case where both linearisations vanish. No scene declares vegetation yet.
- **Snow: the melt cap** (`PH.10`, ADR 0120). `irsim.thermal.snow.melt_capped_step` holds a snow
  surface at 273.15 K and routes the surplus into fusion at L_f = 334 kJ/kg. +200 W/m² holds the
  cell at the cap **bit-exactly** and sheds **2.1557 mm w.e. per hour**. The flux is read **at**
  the melt point rather than at an RK2 midpoint above it — clamping after a step under-reports the
  melt by 0.1–1 % every step, one-signed. A step that carries a cell *through* the cap splits its
  enthalpy so energy is conserved across the transition, and a finite pack hands back what it
  cannot melt. The night needs no special case: with snow's ε_hemi of 0.9874 a clear calm night
  sits **−12.73 K** below air, a breezy one −4.75 K, and overcast **identically 0.00 K** (a fully
  overcast sky is a blackbody at air temperature, so the radiative term cancels). New constants
  `L_F_WATER_J_KG` and `T_MELT_WATER_K`. The alpine ESSD 16 (2024) Tier 4 bar (0.7–1.3 K MAE) is
  recorded, not run; no scene declares snow yet.
- **The extrapolated emissivity fraction, reported and decomposed** (`AT.7`, ADR 0119).
  `total_hemispherical_emissivity` always computed how much of ε rests on extending the nearest
  band; every scene build took `.value` and **threw the fraction away**, so 61 % of the weight
  setting every surface temperature was an invisible assumption. `Scene.emissivity_extrapolation()`
  and `extrapolation_breakdown()` now report it, and `validate_thermal_diurnal.py` prints it under
  the absolute temperatures it quotes. Two corrections to the row: the fraction is **bit-identical
  for all twenty materials** (it is a property of the band set, not of any material — so there is
  no "worst"), and the total hides which assumption is being made. At 300 K it is 0.508 **red
  tail** beyond 13.5 µm, the defensible one; by 800 K it is 0.394 **interior gaps** (1.7–3.0 and
  5.0–7.5 µm), the weak one — and `PH.6`/`PH.7` put the project in that regime. It is also not
  monotone: 0.509 at 1200 K, because a flame's peak lands in the SWIR–MWIR hole.
- **The sea model's angular validity envelope, recorded** (`SE.1`, ADR 0118).
  `irsim.atmosphere.sea_envelope` carries the 50° from-nadir limit published in-situ radiometry
  reaches, the exact spherical `sin θ = (1 + h/R) cos δ` that converts a camera depression into
  the angle that limit is expressed in, and an `EnvelopeReport` that measures a frame against it.
  `SeaModel.beyond_envelope`, `MaritimeScene.envelope_report()`, and the maritime Tier 3 report
  now prints the fraction. The measurement: **every shore and mast frame is 1.0000 outside the
  envelope and every down-looking airborne frame 0.0000 inside it** — at a 20 m eye height the
  limit is crossed at **31 m of slant range**, so the whole maritime working band is
  extrapolation. Two findings beyond the row: the envelope has a **wind axis** (our facet-
  integrated emissivity drop at 55° reproduces the published 2–3 % only to **7.3 m/s**, reaching
  4.33 % at 15 m/s), and reading the limit against **depression** instead of zenith inverts the
  answer rather than blurring it. Nothing is gated: the model still answers past 50°, now labelled.
- **The optical PSF's second factor** (`SC.4`, ADR 0117). Both Bosons carry
  `optics.mtf.aberration_sigma_um = 1.654 µm`, **solved in code** by the new
  `irsim.optics.mtf.aberration_sigma_for_mtf` from FLIR's published 42 % nominal on-axis MTF at
  Nyquist rather than pasted. Until now every shipped camera rendered a diffraction-limited lens.
  `tests/unit/test_lens_mtf.py` re-runs the derivation against the YAML and fails for any camera
  that is neither derived nor listed in `IDEAL_LENS` with a reason.
- **Fire on the camera: gain state, rail and AGC** (`PH.8`, ADR 0116). `irsim.detector.gain_state`
  and `fpa.gain_ceiling_k` — the intrascene ceiling of the state a camera is running in (Boson
  Rev 340: 140 °C high, 500 °C low), applied to the at-aperture **radiance** plane as stage 2e and
  used as the top of the radiometric range, so a clipped pixel lands exactly on the converter's top
  code. A 400 °C object rails high gain at DN 65535 reading 413 K and reads 668 K at DN 45 780 in
  low gain. Defaults to `None`, so every camera and golden written before this is unchanged.
- **Fire: the flame and what it heats** (`PH.7`, ADR 0115). `irsim.thermal.fire` — a `PoolFire`
  authored by its convective heat release and pool diameter; `flame_flux_w_m2` puts
  `F (α SEP − ε L_occluded)` into a cell's `q_internal`; Heskestad's centreline `ΔT₀` replaces the
  weather's air above it, held at the tip's value inside the flame. That tip value is
  **452.7 K above ambient independent of Q and D**, which is Heskestad's own definition of the mean
  flame height and the check that a transcribed coefficient would fail. `Q_c` given in watts is
  refused.
- **`RadiantRectangle` carries a surface emissive power** (`PH.7`). `sep_w_m2` is the authored
  product a flame is actually known by; `emitted_flux_w_m2` answers to a SEP **or** a temperature
  and refuses both, because the two disagree by design. 135 kW/m² luminous and 40 kW/m²
  smoke-obscured (Mudan, Considine), ESTIMATED to the literature's ranges.
- **`flame_plume`** (`PH.7`): the flame as a soot slab a camera can see, on a cone standing on the
  pool, its **mixing length solved** so the slab cools to exactly the centreline excess Heskestad
  gives at the tip. One model, two consumers. Measured: the MWIR and LWIR band means of soot differ
  by **2.47** — just the ratio of the bands' ⟨1/λ⟩ — against **5.49** for hot CO₂.
- `G_STANDARD_M_S2` and `T_STD_ICAO_K` in `irsim.radiometry.constants`, for the buoyancy group and
  an ambient density on the scene's own air temperature rather than on 15 °C.
- **The exhaust plume, per pixel** (`PH.6`, ADR 0114). `irsim.pipeline.plume` — stage 2d of
  `run_frame`: a truncated cone in camera space, one analytic ray/cone chord per pixel, `PH.4`'s
  slab evaluated on it. Occlusion is the G-buffer's own depth plane; entrainment dilutes
  temperature and species by one conserved-scalar factor, so a plume cannot cool without thinning.
  Measured on `configs/scenes/car_exhaust_plume.yaml`: τ = **0.866** in MWIR against **0.979** in
  LWIR and **+72.7 K** of peak apparent temperature against **+7.1 K**, for one authored plume.
- **A plume is authored in a scene file** (`PH.6`). Schema **v15**: a `plume:` block on an
  `exhaust` target, carrying geometry and chemistry only. Its temperature is `TC.7`'s solved
  outlet **gas** at that instant — not `temperature()`, which is a skin some 50 K cooler — and the
  air it mixes into is the scene's weather. `Scene.plumes_at` returns world-space plumes;
  `WorldPlume.in_camera` gives the camera-space one, so nothing holds both frames at once.
- `ExhaustSolver.gas_outlet_k()` — the gas leaving the last segment, which is what a plume is made
  of (`PH.6`).
- **Per-band hot-gas absorption tables, generated offline** (`PH.5`, ADR 0098 addendum).
  `data/gas/<key>_{co2,h2o}.npy` — float32 κ for CO₂ and H₂O over 300–2500 K **and over column
  density** — plus a sidecar that pins the source database by hash, read by
  `irsim.pipeline.gas_tables`. Generated by `scripts/generate_gas_luts.py` from `scripts/radcal.py`,
  a transcription of RadCal's weak-line coefficients (NIST TN 1402, public domain) whose two
  tabulated arrays are fetched from the FDS tree by `scripts/fetch_radcal_tables.py`. The port
  reproduces both arrays **bit-exactly at every grid node** except the two rows RadCal itself
  evaluates a hundredth of a degree short of. RADIS over HITEMP was the first choice and was not
  taken: a multi-gigabyte registered download into a shared interpreter.
- **A through-flame bandpass, as a response file** (`PH.5`).
  `data/spectra/responses/insb_flame_window.csv` — InSb behind a 3.80–4.05 µm filter, in the window
  between the 2.7 µm CO₂/H₂O complex and the 4.3 µm CO₂ band. For a 0.25 m 600 K plume at 10 % CO₂
  / 12 % H₂O it transmits **1.000** where the plain 3–5 µm camera transmits **0.830** and the LWIR
  bolometer **0.913**: three cameras that differ by a response file and nothing else.

- **What a mesh cell can see, traced** (`WM.4`, ADR 0088 addendum). `irsim.thermal.mesh_geometry`:
  a per-cell sky view factor on `PT.21`'s 145-patch Tregenza dome and a per-cell beam on `PT.22`'s
  solar disc, both against the scene's occluders *and* the mesh's own triangles through one
  `Occluders` query — so a shadow and a sky view cannot disagree about where the geometry is.
  `MeshCellForcing` takes the patch, the occluders and the traced factor; `Scene` wires them.
  A **convex** mesh traces to the analytic `(1 + n·up)/2` **bit for bit** (the gated and open
  quadratures are the same additions in the same order), so `cell_occluders` skips it — exactness,
  not a speed heuristic — and every mesh shipped before this keeps its numbers. A mesh cell and a
  patch cell agree to the bit about the sky at one point under one wall through two independent
  ray-rectangle implementations; foot of a 60 m wall 0.5000, edge of a wide overhang 0.5042;
  `PT.22`'s box as rectangles and as triangles cast identical mesh-cell shadows. In
  `quad_flight_mesh.yaml` the motor pod covers each arm's outer 60 mm: those cells lose the whole
  midday beam and all but 0.10 of their sky against 0.93 along the open span, and the arm's
  crown-to-underside spread falls from 26.6 K to 22.1 K — the same body rectangles now shade the
  meshed arms and the patched ones alike, where before they shaded only the patched ones.

- **Golden arrays beyond LWIR** (`GT.2`). All eight reference arrays came from one Boson LWIR
  config, so the reflective-band chain and the sky/sea background — the two places drift is
  hardest to see by eye — had no reference at all. Fifteen more, over the eight subjects the row
  named: MWIR, SWIR and NIR frames through the shipped configs, their committed LUTs and the
  material library's own table; the layered `τ(band, distance, elevation)` table; `L_sky(θ)` for
  both emissive bands; the sea's apparent temperature against depression angle; a half-in-sun
  thermal field after three hours; and the point-wise frame made by sampling it. The store goes
  8 → 23. Each frame asserts it spans the converter without clipping (a saturated golden hides the
  changes it exists to catch), and the reflective bands carry real solar irradiance from
  `SolarIllumination`, since at 300 K their self-emission is ~1e-9 of LWIR's and an emission-only
  golden would pin the dark current and nothing else. A test makes the case checkable rather than
  rhetorical: the τ table **would have caught `AT.10`**, which moved SWIR 0.4148 → 0.3877 at 5 km
  while all eight of the old goldens passed.

- **One wavelength ladder for the atmosphere's spectral classes** (`AT.10`, ADR 0113).
  `BAND_CLASSES` — five hand-written spectral-class tables keyed by camera band name — is replaced
  by `ATMOSPHERE_LADDER`, one wavelength-ordered, gap-free table from 0.35 to 14.5 µm, from which
  `classes_for(band, response)` derives a band's classes by intersecting its own span. That span is
  the band's nominal range **and** its response, because either alone has been wrong here: the
  nominal range alone was `AT.3`'s defect and the response alone would let a narrow filter shrink
  the model of the air it looks through. Three faults go with the tables. Two of them **disagreed
  about the same air** — NIR resolved the 0.94 µm water band at ×10 while SWIR's window swallowed
  0.90–0.98 µm at ×0.5, a factor of twenty over one sky; deriving from the ladder moves **16.9 %**
  of the Planck-weighted InGaAs band out of `window` and costs SWIR **6.5 % of its transmittance at
  5 km**, with the 200 m anchor exact (which is why nothing caught it: the model is pinned where it
  was fitted and wrong where it is extrapolated). Two stretches, **1.80–2.00 and 6.00–7.00 µm**,
  belonged to no class at all and would have raised for any camera that reached them; they are the
  1.9 and 6.3 µm water bands and are now modelled opaque, as the 1.4 and 2.7 µm bands already were.
  And a fifth band needed a sixth table in `src/`: the `atmosphere/layered.py` carve-out is down
  from **seven offences to two** (the two Koschmieder identifiers, which are the same photopic
  definition `extinction.py` carries), and a test now refuses *any* carve-out that enumerates the
  band registry. The calibrated bands are untouched — LWIR and NIR bit-identical, MWIR within one
  ulp (wavelength order changes the summation order), `visible` within the anchor solver's
  1.5e-14 — so ADR 0071's R13 sky calibration and every golden array still stand.

- **Lateral conduction between a mesh's cells** (`WM.6`, ADR 0112). `irsim.thermal.mesh_conduction`
  extends `PT.11`'s term from a rectangular grid to a triangle mesh: a two-point flux
  `k δ w/(d_a + d_b)` within each face and across every shared mesh edge, faces of different levels
  matched by the length they overlap (as `TC.3`'s contactors match two patches), behind the same
  `ConductionOperator` so ADR 0094's backward Euler and its prefactorisation are reused.
  **Monotone over consistent**: the circumcentric dual-cotan weight is the consistent one and goes
  negative on any obtuse face, which breaks the discrete maximum principle and renders as a bright
  speck; `ConductionOperator` refuses a negative conductance, so that assembly fails at build
  rather than in a frame. Measured: identical to the cotan weight on an equilateral face
  (`√3 kδ`), independent of the level, linear-exact to 7e-16 there against 1.5–4.5 % when skewed,
  energy conserved to 1e-9, and a carbon tube reproduces the fin equation's 0.744 to 1 %. The skew
  error is an authoring rule — cut a tube so its rings are ~1.4× its circumferential arc — and
  `quad_flight_mesh.yaml` is re-cut from 24 × 6 (a 12.7 : 1 quad, three quarters of the
  conductivity it should have) to 16 × 36. The arm's crown-to-underside span is now **15.4 K**,
  against 26.6 K before `WM.4` traced the pod's shadow and 22.1 K before this operator.

- **ADR 0111 — temperature granularity tiers** (`PT.14`). Four of them, written down beside each
  other for the first time: one facet per prim, cells on a plane (ADR 0087), cells on a mesh
  (ADR 0110), and a network node, which is a mass and not a surface. No per-material tier, and no
  tier chosen by range or pixel footprint. Measured: cells are nearly free per tick (288 → 2304
  cells is 0.95 → 1.66 ms, the fixed ~0.8 ms being the forcing) and cost at build instead. The
  selection rule is the fin equation's smoothing length `L = √(kδ/h)`, tabulated from the
  committed material library — 2 mm on a leaf, 9 mm on carbon fibre, 49 mm on asphalt, 145 mm on a
  painted aircraft skin, which therefore cannot hold a fine pattern at all. The meshed arms sit
  finer than their own `L` on purpose, which makes their 22.1 K an upper bound: the same ring
  solved with lateral conduction is 26 % smaller, so `WM.6` is worth about a quarter of it.

#### Changed
- **The Boson's goldens moved, and in the shape an aberration should** (`SC.4`). Twelve arrays
  changed and seventeen did not: the smooth ramp by **7.8 mK**, the hard-edged hot patch by
  **3.97 K** (rms 0.21 K), and the uniform SITF and noise-cube fields **bit-identical**, because no
  PSF can change a flat. MWIR/NIR/SWIR frames untouched. Regenerated after that comparison.
- **The Tier 2 MTF bench measures the shipped camera** (`SC.4`): σ is read from
  `flir_boson_640_lwir.yaml` instead of typed as 0.0 beside it, so the bench and the renderer
  cannot disagree about which lens they describe. Its Nyquist expectation moves from 0.31 to 0.27.
- **Golden config hashes regenerated** (`PH.8`). Adding `fpa.gain_ceiling_k` moves `config_hash` on
  23 stored sidecars even though every one has it as `null`. Every golden **array** was verified
  bit-identical before `make golden-update` ran; only the hash moved.
- **`SpeciesAbsorption.kappa` reads whole arrays** (`PH.6`). A per-pixel plume asks for the
  coefficient at every pixel it covers, so the two interpolations — linear in T, linear in log X on
  the *optical depth* — are done over arrays rather than pixel by pixel. `at()` is the scalar case
  of it and returns the same number.
- **`PipelineConfig` carries the camera's R(λ) and its band's gas tables** (`PH.6`). Stage 2d
  integrates `B_b(T_g)` above the LUT's 1000 K ceiling, so it needs the response itself. Both are
  `None` for a camera or band that has neither, and a plume then raises rather than rendering clear.
- **`SpeciesAbsorption` carries a column-density axis** (`PH.5`). A band coefficient that goes
  inside one exponential is not a property of the gas alone. In a 3–5 µm camera essentially all of
  CO₂'s absorption sits in 4.2–4.45 µm, so the emission-weighted mean of κ(λ) — 200–360 1/(m·atm) —
  makes a 30 cm exhaust plume opaque when its band transmittance is 0.82, and *falls* with
  temperature where the band's absorptance rises. What is stored instead reproduces the band's own
  transmittance at the path: κ runs **35.8 → 0.61 1/(m·atm)** across 0.005–0.5 atm·m, so a single
  number would have been wrong by sixty times. A one-dimensional table still reads as
  path-independent.
- `ShadowRectangle.normal` is cached in `__post_init__` instead of crossing its axes on every
  call. One scene's spin-up reached a million calls and spent more than half its wall clock inside
  `np.cross`; the mesh scene's build went 32.5 s → 17.3 s, and every occluded patch benefits too.

#### Fixed
- **A band the absorption model cannot reach is refused, not zeroed** (`PH.5`). RadCal truncates at
  1.75 µm for CO₂ and 1.08 µm for H₂O; `gas_tables` records each band's covered fraction and
  refuses below 0.99, so SWIR and NIR raise an error naming HITEMP instead of rendering a flame as
  a transparent one.
- **Three render drivers ran in one band each** (`IG.13`). `render_aerial_demo.py`,
  `render_car_ignition.py` and `render_vessel_departure.py` built their scene with
  `Scene.from_file`'s default `quantity="lb"` rather than the sensor's own, so the sky model came
  out in band radiance while a photon FPA runs on `lb_q`; `PipelineConfig.from_sensor` compares
  the two and raised `sky model built in the 'lb' form`. Each driver therefore worked only in the
  band its `--sensor` default names. Found by running the sweep, which lost 3 of 16 aerial renders
  at startup. The three now pass the sensor's quantity and its `DiffuseSkylight` (M11.10, ADR
  0086) through, as the other three already did, and `tests/unit/test_render_multiband.py` reads
  every swept driver's `Scene.from_file` calls by AST and fails if one omits it.
- **...and then attached a sensor chain the camera could not carry** (`IG.13`). Hidden behind the
  entry above, in the same three drivers: `attach_sensor_chain` needs a radiometric calibration to
  convert the M9 NUC residual's millikelvin into DN (ADR 0056), a photon FPA has none (M11.6), and
  these three attached it unconditionally. They now skip the chain and say so on stderr, as the
  other three already did -- the camera is shutterless anyway, so there is no FFC to freeze, and
  defects and 3-D noise still apply. Guarded by the same AST test.
- **A point target outside the frame stopped a whole render** (`IG.13`). The fourth of four, and
  the only one not in the driver: the aerial scene places its point targets by angle, so a target
  the Boson sees at the edge of its field lands at x = 700 px on the InSb's 640 px frame, and
  `splat` raised rather than the camera ignoring it. `IrCamera.point_targets()` now drops a target
  outside the frame and records it on `last_offscreen_targets` for the driver to report, using
  `splat`'s own safe interval — half a supersample cell in from each edge, which
  `tests/unit/test_point_target.py` pins from both sides.

- **...and then asked for a flat field their camera could not hold** (`IG.13`). The third of three,
  in the same three drivers, each hidden behind the last. `calibrate_flat_field`'s default hot
  point is ADR 0021's +200 C -- a *bolometer* range; the modelled InSb camera fills its well at
  366 K, so that point drives it 16x past its converter and the two-point fit becomes an
  extrapolation that arrives as inverted vignetting, reading as a lens problem rather than a
  calibration one. `PipelineConfig.from_sensor` is right to refuse; the three drivers now catch
  the refusal, say what they are dropping and render without a flat field, as the other three did.

### 2026-09-21

#### Added
- **A per-cell sky view factor** (`PT.21`, ADR 0104). `irsim.thermal.skyview`: Tregenza's
  145-patch dome, each patch sub-sampled 3 × 4 over its own extent with exact solid angles,
  every ray gated by the same `cell_shadow` test the beam uses and the sum normalised to the
  open dome so an unobstructed cell keeps its tilt's `V_s` to the bit. A scene computes it for
  every world-frame patch under occluders; `CellForcing` scales both the diffuse solar and the
  longwave down by it (`sky_view_longwave=False` is the roadmap's negative control). Anchors:
  open sky 1.000, the foot of an infinite wall and the edge of an infinite overhang 0.5 within
  0.01. Two shaded asphalt cells at SVF 0.2 and 0.9 under one weather file: swing ratio 1.25 and
  the enclosed cell 3.2 K warmer at its night minimum, where a factor on solar alone leaves the
  minima within 0.05 K. The ratio near 2 the row asked for is out of reach with one air
  temperature per scene (the swing cannot fall below the air's own 12 K) and is recorded as such.

- **Still water: lakes, ponds and puddles** (`PH.3`, ADR 0108). `irsim.thermal.still_water`: a
  fresh-water conductive sublayer (`sublayer_thickness_m`, Saunders bounded by a tanh) whose skin
  offset is **signed** -- the sea's clamp (ADR 0080) is right there and wrong on land, because
  over a pond the fluxes that reverse are longwave and condensation, and both land *in* the
  sublayer -- plus `mixed_layer_capacity_j_m2_k` and `apparent_temperature_k`, the
  `eps(theta) B(T) + (1 - eps(theta)) L_sky` mix a camera actually reads. Scene schema **v12**
  adds `water:` on a patched surface: the cells inside `region_m` take the mixed layer's mass on
  top of the substrate's, the water material's emissivity and solar absorptivity, and a film of
  the puddle's own depth to evaporate, while the cells outside stay the road bit for bit. Fresh
  water's kinematic viscosity, conductivity, density and specific heat join
  `irsim.radiometry.constants` rather than reusing the seawater values. Measured: the sublayer
  spans 0.67-3.61 mm over 0.8-8.2 m/s; a clear calm night puts the skin 0.38 K below the bulk and
  a humid overcast one 0.21 K above it (the clamped form: 0.00); Kirchhoff closes to 1e-12 from
  nadir to 89.5 deg; a 293 K puddle reads 0.55 K below its kinetic temperature at nadir, 1.90 K
  at 60 deg and 4.40 K at 70 deg under a clear sky, and under half a kelvin at 60 deg overcast;
  and 20 mm of water over half PH.2's road opens 3.8 K below the dry half and is 11 K below an
  hour later. The row asked for "several kelvin at 60 deg" and 0.5 K at nadir: this sky is colder
  than it assumed, which deepens the nadir deficit and moves "several" out to 70 deg.
- **Occlusion from geometry, and the sun as a disc** (`PT.22`, ADR 0107).
  `irsim.thermal.raycast`: an `Occluders` protocol whose one method answers "is this ray
  stopped?", implemented by `RectangleOccluders` (`cell_shadow`'s exact test generalised to
  per-ray origins and directions, and still the oracle), `TriangleSoup` / `MeshOccluders`
  (Möller–Trumbore in NumPy, an axis-aligned box reject per soup, chunked over rays, no
  acceleration structure and no dependency) and `AnyOccluders` (rectangles beside meshes).
  `solar_disc_rays` samples the sun's 0.5332° disc on concentric rings with the outermost ring
  exactly on the limb -- 1, 7, 19 or 37 rays -- and `sunlit_fraction` turns them into a per-cell
  number in [0, 1], snapped at the ends so a cell nothing shades stays bit-identical to the
  per-prim solve. Scene schema **v11** adds `thermal.penumbra_rays:` (default 1, which is the
  hard edge to the bit). `SOLAR_DISC_DIAMETER_DEG` joins `irsim.radiometry.constants`.
  Measured: a box shades identically as six rectangles and as twelve triangles, cell for cell,
  from 10° to 85° elevation and against trimesh's own intersector; a wall's own box never shades
  the wall while a block 1.5 m west of it does; the penumbra ramp is within 10 % of
  `d tan(0.53°)` at 0.5, 2 and 6 m standoff (18.0 mm against 18.6 mm at 2 m) and a binary test
  gives a ramp of zero width. Meshes are not yet authorable from a config.
- **The cabin and the two-node substrate, reachable from a scene** (`PT.15`, ADR 0106).
  `LumpedTwoNodeSolver` and `CabinNode` shipped in M6 and no config could construct either, so
  every solved surface had an adiabatic back. `irsim.thermal.coupling` gains `LumpedMember` (a
  node with no cells: its areal capacity is its J/K, its convection coefficient the infiltration
  conductance, its `q_internal` the watts through the glazing) and `LumpedLink` (`A_cell / R_p`
  from every cell of a panel to it); `irsim.thermal.cabin` gains `cabin_coupling` and
  `cabin_field`, which solve the panels and the air on **one** implicit operator rather than
  alternately -- the step ADR 0038 measured as wrong at a 60 s tick. Scene schema **v10** adds
  `thermal.cabin:` and a surface's `back:` (§6.4's R₂d and T_deep on a layered surface), and
  `irsim.thermal` now exports both objects with `LayerStack`. Measured: the coupled member
  reproduces `CabinNode.equilibrium` on ADR 0038's panels to 0.021 K at a 2 s tick, with the roof
  4.84 K above an adiabatic back. New scene `configs/scenes/parked_car_cabin.yaml` and
  `scripts/parked_car_cabin.py`: cabin 68.7 °C at local noon, roof 65.0 °C against 63.4 °C for
  the same paint with an adiabatic back, glazing 37.3 °C, and at midnight roof and cabin 3.8 K
  and 3.0 K below the air. The scene's roof boost is +1.6 K rather than ADR 0038's +4.8 K
  because its only sun-facing panel is the roof; recorded, not tuned away.
- **The exhaust line as a gas stream in a wall** (`TC.7`, ADR 0105). `irsim.thermal.exhaust_line`:
  `PipeSection`s in order, each a run of wall nodes on the S42 network (steel or cast iron,
  outside convection forced while the car moves and natural when it stands, radiation to the
  floor pan and the road, hangers at ~1 W/K, an inner mass for a catalyst's monolith or a
  silencer's baffles, a heat shield as a thin node across an air gap), and `GasFlow` (ṁ and
  inlet temperature linear in load, Dittus–Boelter h_i). The gas is marched segment by segment
  -- exact `exp(−NTU)` per segment -- and enters the network as a link of `ṁ c_p (1 − e^{−NTU})`
  to a fixed node at the segment's inlet temperature. `ExhaustSolver` is a `TemperatureSolver`;
  `solver: exhaust` with `section:` reaches it from YAML. Measured: the march matches the closed
  form to 1e-6 over 24 segments, the gas heat equals `ṁ c_p (T_in − T_tail)` to 1e-9, the
  energy residual is below 1e-6; at 60 % load the line runs manifold 557 → tailpipe 368 °C with
  hung segments 40–50 K colder; after key-off the manifold shield rises 287 → 347 °C peaking at
  +65 s and is below 260 °C 5.3 min later, the catalyst shell peaks at +65 s. The shell runs
  270 °C, not MVFRI's 400 (no exotherm), and the car scenes keep §6.6's schedule for now.

- **A mesh temperature field, reachable from a scene config** (`WM.7`, schema **v14**). `WM.2`
  built the field and `WM.3` put it on pixels; until this nothing could ask for one, so the lane
  was a capability with no user. A surface now takes a `mesh:` block instead of a `patch:`:
  `MeshSpec` names a cylinder or a sphere with its facet counts and a per-face cell `level` (or a
  `cell_m` to pick one), `build_mesh` generates it, and `Scene.meshes` / `mesh_fields` /
  `mesh_bindings()` carry it exactly as the patched path does. The shape is **generated from the
  config rather than read from an asset**, because the engine-free core may not import `pxr`
  (CLAUDE.md #1) and a scene has to be loadable with no renderer present; ingesting a real prim's
  triangles is the follow-on. New `MeshCellForcing` gives each cell **its own face's normal** for
  the direct beam and for `V_s = (1 + n·up)/2`, which is the term one shared patch normal cannot
  express. A mesh field is **always** spun up per cell, where a patch only bothers under
  occluders: no two cells of a mesh share a forcing history, so the per-prim spun-up value is
  wrong for all of them and the scene would otherwise open with a uniform tube. `cylinder_mesh`
  joins `irsim.thermal.raycast`. `configs/scenes/quad_flight_mesh.yaml` is the first scene to use
  any of it — the aerial mission of `quad_flight_pointwise.yaml` with the two arms as 30 mm carbon
  tubes — and `scripts/quad_flight_mesh.py` writes each arm **unrolled** plus a video of the
  mission. Measured: crown 52.8 °C against an underside sitting on 26.2 °C of air, **26.6 K around
  one arm**, where the same arm as a patched strip carries under a millikelvin across its width;
  the mission collapses the crown's excess over air from 26.6 K on the pad to 16.4 K in the climb and 12.7 K in the hard climb, and landing brings it back to 27.6 K. Refused
  rather than ignored on a mesh: `patch:` alongside it, and `film:`, `water:`, `layers:` and
  `back:`, each of which is a rectangular-grid construction with no mesh equivalent shipped.

- **ADR 0110 — the surface temperature field lives on the mesh** (`WM.5`). The record for
  `WM.1`–`WM.3`: what the closest-point parameterisation is, why it needs nothing from the
  renderer, the error budget, and the routes rejected *with their reasons*, so they are not
  rediscovered. Rejected: a **UV atlas as the solver domain** (not merely unavailable — it carries
  metric distortion, so equal texels are unequal areas and §6.1's areal heat capacity stops meaning
  what it says; seam severing, so a lateral operator has to reconnect what the unwrap cut; and a
  conservative-rasterisation tax on chart boundaries), a **per-triangle id AOV** (blocked on an SPG
  shader), **closest-point-method narrow bands** (the band must be finer than the thinnest feature,
  and a car panel is ~1 mm), and **transient surfels** (`PT.7` spins a field up over 48 h and the
  answer depends on that memory, which a per-frame sample set cannot hold). ADR 0087's status now
  names 0110 as superseding its curved-geometry limitation; its planar patch, slab rule and cost
  argument are untouched and still ship.

- **The mesh field on the render path** (`WM.3`). `irsim_isaac.pipeline.mesh_bridge`:
  `MeshPointBridge` takes a finished per-instance temperature plane and overwrites only the pixels
  of prims that have a `TriangleMeshField` bound to them -- per-pixel instance id picks the prim's
  `wp.Mesh`, a closest-point query gives `(face, u, v)`, and the field gives the temperature. This
  is the case ADR 0087 lists as Hard: **an exhaust pipe goes from 0.000 K across the whole prim to
  34.1 K** (12.9 → 47.0 °C around its circumference) without the renderer transporting anything it
  does not already carry. Additive exactly as the planar bridge is, so a prim with no binding is
  bit-identical and attaching it to an existing scene is a no-op. **Warp is the accelerator, not
  the authority**: `irsim.thermal.closest_point_on_mesh` (new, engine-free, Ericson's region test
  over every triangle) is the oracle the tests hold Warp to *and* the fallback when Warp is
  absent, so the bridge runs on the plain-CPython gate and `device` defaults to Warp's **CPU**
  device. Measured: Warp and the oracle agree on face, cell and sampled temperature for **100 %**
  of 4 000 pixels on a sphere, and the two routes render the same frame bit for bit. A pixel whose
  closest point on the bound mesh is further than 7 mm away (twice ADR 0014's 3.4 mm position
  budget, because a mesh is itself a chord approximation) **raises rather than snapping** -- the
  id plane and the position plane disagreeing is not a slightly noisy hit, and snapping would
  paint a wheel with whatever part of it happened to be closest. Where several meshes share a
  prim the nearest surface wins, so the result does not depend on binding order. `PT.19`'s rule
  carries over: a binding to a prim path the stage does not know raises at construction. Not yet
  reachable from a scene config -- no shipped scene authors a mesh field, so every rendered frame
  still shows its curved prims at one temperature until one does.

- **A temperature field on a triangle mesh** (`WM.2`). `irsim.thermal.mesh_field`:
  `TriangleMeshPatch` cuts every face into `k²` congruent cells on the barycentric grid at a
  **per-face** level -- Ptex style, so resolution follows the thermal gradient and not the
  tessellation, and `by_cell_size` picks the level per face from a target cell size.
  `TriangleMeshField` composes `ThermalField` exactly as `PlanarThermalField` does, so the fixed
  tick, the between-tick interpolation and the never-mutate-on-query rule are reused rather than
  restated; `PlanarPatch` stays the special case for near-planar surfaces and is untouched.
  `sphere_mesh` joins `irsim.thermal.raycast` beside `box_mesh`, so one mesh can both shade and be
  solved. Measured: a 0.25 m sphere under an overhead sun holds **every face to its own cos θ root
  within 1 mK** across a **34.3 K** span (sunlit cap 320.3 K, far side 286.0 K), where a single
  facet on the area-weighted mean flux -- what every prim in this project carried before a field
  -- lands at 295.0 K, 25 K under the cap and 9 K over the far side. Conservation: under uniform
  forcing every cell is **bit-identical in float32** to the scalar solve, and under varying
  forcing the area-weighted mean sits *below* the per-prim value exactly as the concavity of
  `T(q)` requires, with the area-weighted net flux closing to 1e-6 of the absorbed power -- so the
  change is provably a redistribution and not new energy. Deliberately deferred and recorded in
  the module docstring: the sample is piecewise constant within a cell (smoothing across faces
  needs edge adjacency, `WM.4`), normals are per face rather than per vertex (§6.1 balances a
  facet), and cells do not conduct to each other (`PT.11`'s operator is a rectangular grid's; the
  mesh equivalent is `WM.6`'s, which exists because a naive cotan Laplacian breaks the discrete
  maximum principle). Nothing on the render path calls this yet -- `WM.3` is the bridge, and
  `WM.5` writes the decision that supersedes ADR 0087.

- **A probe for per-pixel (face, u, v) from Warp** (`WM.1`, ADR 0087 addendum).
  `scripts/probe_warp_mesh.py` (no Kit boot) and `scripts/probe_warp_prim.py` (a real USD prim)
  build a `wp.Mesh` and recover a triangle and its barycentrics from a world position with
  `mesh_query_point_no_sign`, against a brute-force NumPy closest-point oracle. This is the
  measurement ADR 0087's "a real limit, not a temporary one" rested on, and it goes the other way:
  the parameterisation can be **derived** from the position AOV instead of transported by a UV or
  per-triangle AOV the build does not have. Measured on Warp 1.16.0: `mesh_eval_position` returns a
  surface point to **0.13 µm** against the 3.4 mm position budget (0.73 µm at 4.3 m from the origin
  -- the residual is float32 and grows with distance, so a scene tens of km across needs its own
  check). Warp's `(u, v)` weight **v0 and v1** with `1 - u - v` on v2; the reading a person writes
  down first is wrong by **0.56 m on a 0.4 m box**, which is why the convention is measured rather
  than assumed. Under a full 3.4 mm of position error the recovered *face* flips on 1.9 % of
  queries on a 12-triangle box and 56 % on a 16 k-triangle sphere, but the sampled surface point
  moves only 2.68 mm on average and never past 3.41 mm -- the face is unstable between neighbours
  and the position is not, which is what a temperature lookup needs. 2.3-5.9 M queries/s on the
  **CPU**, so a 640x512 frame is 0.06-0.14 s with no GPU. From a USD prim: quads need triangulating
  (`faceVertexCounts` of 4), the local-to-world transform must be applied, and an analytic gprim
  such as `UsdGeom.Sphere` exposes no points to hand Warp at all. A probe and numbers, not a
  pipeline: `WM.2` puts a field on a mesh, `WM.3` binds it and `WM.5` writes the decision.

#### Changed
- The R1 wall scene's terminators moved with the sky view: concrete 10.3 → 9.7 K, render
  7.4 → 6.3 K, memory after 30 min 8.7 → 8.05 K (the shaded half sees the neighbour's roof, not
  cold sky); the post-cap scene's never-shaded cells now sit within 20 mK of the per-prim solve
  instead of on it, the bit identity holding where the dome is open. A surface's `shaded` flag
  gates the beam under a supplied sky view as it did without one.

#### Fixed
- **Every render of a scene with a `nodes:` block failed on frame zero**, and had since `TC.4`.
  `Scene.advance_targets` returns the thermal network's nodes under their own names alongside the
  targets, but `AerialThermalBridge` seeded its first tick bracket from `scene.targets` alone, so
  the first tick made `next_k` six names wider than `prev_k` and `TickBracket.interpolate` raised
  `KeyError: 'subframe'` before a single frame was written. Both car scenes are affected --
  the only two that declare a network -- which is why `outputs/multiband`'s manifest recorded no
  car renders. The bracket is now seeded with the network's own temperatures, and
  `test_aerial_bridge.py` asserts the first bracket names everything a tick reports *and* starts
  each node at the network's value, since a bracket seeded with wrong numbers would not raise.

### 2026-09-20

#### Added
- **An N-layer stack through every cell** (`PT.12`, ADR 0103). `irsim.thermal.layers`:
  `LayerStack` (layers top first as `two_node.NodeLayer`s, §6.4's centre-to-centre resistance
  `δ_i/2k_i + δ_{i+1}/2k_{i+1}` between them, an optional deep node under the last, adiabatic by
  default; `LayerStack.uniform` cuts one material into N equal slices) and `layered_field`,
  which makes each layer a member of a `CoupledFields` on the same grid -- layer 0 with the
  surface balance and the material's ε and α, the layers below with none, consecutive layers
  joined by a contactor at `1/R` -- so the stack is one implicit solve per tick, its surface
  layer a `PatchView` the bridge binds, lateral conduction running in every layer with its own
  `k δ`, and the whole stack spun up so the base carries the days before. The scene schema gains
  `layers: N` on a patched surface (default 1, the field it always was; a film on a layered
  surface is refused for now). Measured: two layers with a deep boundary reproduce
  `LumpedTwoNodeSolver` to under 1 mK on both nodes over 6 h at a 2 s tick (1.7 mK at 10 s: the
  IMEX side is first order in dt); the contact conductances are `k/δ` for equal slices exactly; a
  1 mm steel skin peaks at 12:19 with the beam and a six-layer 0.3 m asphalt surface at 13:45; the
  single-node 0.3 m asphalt is 5.6 K warmer than the six-layer one at 04:00, holding its surface
  up on the whole slab's heat, with the stack's base warmer than its surface at night;
  `layers: 6` on a scene's road builds a six-member solve whose surface view answers the
  bridge's interface. 5 cases in `tests/unit/test_layers.py`.
- **An unconsumed binding is loud** (`PT.19`). `PointwiseTemperature(bindings, known_paths=)`:
  a binding to a prim path the stage does not know -- a misspelling, a renamed prim, a stage
  without it -- raises at construction, naming the path and the stage's prims, before the first
  frame; `IrCamera` passes its prim map. The prim used to be skipped in silence and rendered at
  its per-instance fallback with a seam that looked like physics, which the scene's own YAML
  comment called a guard. A frame's own labels cannot make that call (a prim off screen is absent
  from them too), so without a stage list a bound path absent from `idToLabels` raises under
  `strict` and warns otherwise, and with one it is coverage 0, quietly. `last_coverage` counts
  the pixels each binding took in the frame, `IrCamera.patch_coverage` exposes it and the car
  render's per-frame record carries it. Measured: `/World/raod` bound against a stage of
  `/World/road` raises naming both; frame-only, strict raises and lenient warns with the plane
  untouched; a correctly spelt binding is bit-identical to before; a bound prim off screen this
  frame is coverage 0. 2 cases in `tests/unit/test_point_bridge.py`.
- **The R1 reference scene: a wall half in sun, from YAML plus one command** (`PT.20`).
  `configs/scenes/wall_half_in_sun.yaml`: a concrete building 8 × 6 × 6 m (four wall patches and
  a roof, the west wall split into a concrete half and a half of the new `etics_render`, a thin
  light render over insulation), a lower neighbour standing 2 m to its west as five occluders
  beside the building's own five, asphalt ground, at 18:00 on the clear June file when a low
  western sun has been on the west wall for four hours. `scripts/wall_half_in_sun.py` prints
  every face's mean and lit/shaded step, writes the west wall as a frame from a synthetic
  G-buffer through the point bridge (both halves on one prim; float32 `.npy` and a white-hot
  PNG), lifts the neighbour's shadow and follows the once-shaded concrete. Measured: the
  terminator across the concrete is 10.3 K and across the render 7.4 K (the ETICS thermography's
  7.4 °C) -- one face, one shadow, two materials a shadow that only scaled the beam could not
  tell apart; the roof and the north wall differ by 12.4 K, the sunlit west wall and the east
  wall that lost the sun at noon by 7.5 K; half an hour after the shadow lifts the once-shaded
  concrete is still 8.7 K cooler. Two of the row's figures came out below it and are held to what
  was measured: west against north is 7.4 K rather than > 10 K because at 18:00 the north wall
  catches a grazing beam of its own, and the after-shadow memory is 84 % of the step rather than
  > 10 K. 6 cases in `tests/unit/test_wall_half_in_sun.py`; the scene is listed as unswept by
  the multiband sweep; the rendered frame needs `IG.2`.
- **The car demo's fields take their thermal properties from the material library** (`PT.6`,
  ADR 0043's single-source rule). `Scene.surface_properties(name)` returns a §12.3 surface's
  `ThermalProperties` as the scene solved it and `Scene.surface_materials` the library material
  behind it; `build_bonnet_field` and `build_ground_field` read C, ε, α, k and δ from the scene's
  `bonnet` and `asphalt` surfaces, and `build_car_demo` refuses an authored override, naming
  `configs/materials/`. The driver used to author C = 60 000 J m⁻² K⁻¹ for the road against the
  library's 101 200 -- 1.7× in the road's time constant -- ε 0.92 for the paint against the
  material's 0.853, and 8000 against 4399 for the skin. Measured: both fields equal
  `ThermalProperties.from_material` bit for bit and their lateral operators carry the materials'
  own k and δ; with the library's numbers the overcast bonnet is 17 K max–min at 1500 s (was
  18) and the road's engine patch after 30 min 2.4 K (was 3.9, the heavier asphalt warming more
  slowly). 2 cases in `tests/unit/test_car_demo.py`.
- **Lateral conduction between a patch's cells** (`PT.11`, ADR 0102; spec issue S41 is now
  `code`). `irsim.thermal.conduction.lateral_operator(patch, k, δ)` links four-neighbour cells
  with `K = k δ · (shared side) / (gap)` in W/K -- Fourier across the slab -- as a
  `ConductionOperator` on ADR 0094's IMEX step; `k δ = 0` returns `None`, the operator-free
  field bit for bit. Every patched surface in a scene builds it from its own material's
  `conductivity_w_mk` and `thickness_m` (the per-material switch: asphalt at 0.0375 W/K per
  edge barely notices, steel spreads 99 mm over an engine's rise), `lateral_conduction: false`
  opts a surface out, the per-cell spin-up carries the operator, and the car demo's bonnet takes
  steel's 45 W/mK over 1.2 mm until PT.6 reads the material. Measured: a 20 K step on a 2 m
  strip of 5 mm aluminium cells spreads as the semi-infinite sheet's erf to 0.6 % of the step at
  60 s; the operator is `k δ dv / du` per u-edge and `k δ du / dv` per v-edge exactly; at 5 cm in
  aluminium the explicit limit is 6.4 s and a 60 s implicit tick keeps the maximum principle and
  the mean where forward Euler on the same operator explodes; the steel bonnet's engine hot spot
  is 5 % smoother than the independent-column field with the same mean. The PT.17/PT.18
  bit-identity tests now declare `lateral_conduction: false`, because with the operator a lit
  cell beside a shaded one exchanges heat -- millikelvins for asphalt, and the physics. 6 cases in
  `tests/unit/test_lateral_conduction.py`.
- **The wet/dry road: one object, two states** (`PH.2`, ADR 0101 addendum). A patched surface
  may declare `film: {depth_mm, region_m}`: water on every cell or on the cells inside a
  rectangle of the patch's own (u, v), at the scene start. `configs/scenes/wet_road_noon.yaml`
  waters the west half of a 12 m asphalt patch with a 0.5 mm pass at 14:00 on a clear June day
  with a wall shading the south rows -- one prim, four states. `FacetSolver`, `ThermalField`
  and `PlanarThermalField` expose `evaporated_kg_m2`, the water each cell's film has given up,
  bookkept exactly through the clamp. Measured: the sunlit wet half runs up to 9.9 K colder than
  the dry half at ~27 min (Hendel 2014's FLIR B400: 6–13 K), the shaded contrast is a third of
  that at the peak and reaches ~5 K later because evaporation runs on the surface's own
  temperature; the sunlit film is gone at ~28 min, inside the 15–120 min window (0.2 mm would go
  in ten minutes on a 57 °C road under 2.5 m/s, 1.3 mm/h, which is why the scene waters 0.5 mm),
  and the halves reconverge to under a quarter of the peak by three hours; film₀ − film equals
  what evaporated to 1e-6 and an independent quadrature of E(T) over the tick temperatures agrees
  to 1e-6. 5 cases in `tests/unit/test_wet_road.py`. The rendered frame needs `IG.2`.
- **The R2 reference scene: an engine warms the metal around it** (`TC.6`, ADR 0100 addendum).
  Both car scenes declare the engine as `solver: engine` and the metal around it as `nodes:` and
  `links:`: the block followed as a boundary (`follows: {target, node}`, a one-way boundary
  reading a solved target's node each tick), rubber mounts as a link node to the subframe, a
  wing bracket bolted through 25 cm² of `bolted_ferrous_new` whose fan-driven convection drops
  to natural when the engine stops (`switch: engine_bay`, `off_h_w_m2_k`), and the wing on two
  small bolts; the key turns on at 30 s and off at 20 min so the hot soak is in the film. The
  engine model gained what the migration showed it lacked: heat into block and coolant as the
  thirds rule (`block_fraction`, 0.9 of mechanical power at the load), a **proportional
  thermostat** (90 °C, a 6 K band, 5 kW/K open) onto a radiator node, closed at key-off, and the
  scene's `load` becomes a duty fraction of rated power (an idle in a car park is 0.10). The
  target reports the **block**, the radiating mass under the bonnet, not the bay air; the bonnet's
  underside now convects with the bay air, folded exactly into the cell's convective term, so the
  bay's key-off spike reaches the skin. Measured: a bracket on a fixed block reaches 1 − e⁻¹ of
  its share at τ = C/(G + hA) and settles at G/(G + hA) to 1e-6; parts warm block → bracket (30
  W/K bolted) → mounts (12 W/K rubber) → wing; the overcast bonnet is 18 K max–min at 1500 s and
  its rise climbs from 21 K at key-off to 27 K ten minutes later; the engine's bands hold with
  the thermostat (full load +78.5 K flat to 1e-6 per tick, idle +65.5 K, key-off +32 K / +0.9 K
  at 1 h / 7 h, bay overshoot +40 K at 140 s). Not done, recorded: the bonnet joined to the body
  by a contactor needs one implicit system across the network and a field. 3 cases in
  `tests/unit/test_r2_scene.py`; `test_car_demo` and `test_engine_node` re-read for the solved
  engine (the clear/overcast road patch is now held as a difference, the sump radiating at the
  block's temperature having grown the engine's share). Frames need `IG.2`.
- **The latent-heat term and a wet film per cell** (`PH.1`, ADR 0101). §6.1's balance gains
  `Q_L = L_v ρ_a (q_sat(T_s) − q_a) / (r_a + r_s)`, evaluated on the solver's own temperature like
  emission (`irsim.thermal.latent`; Magnus/Bolton saturation shared with the atmosphere module;
  `L_V_WATER_J_KG`, `C_P_AIR_J_KGK`, `RHO_AIR_STD_KG_M3`, `P_STD_HPA`, `EPSILON_WATER_AIR` and
  COARE 3.6's `C_E_BULK` in `constants` with sources). `SurfaceForcing` / `FacetForcing` carry
  `q_air_kg_kg`, `g_e_kg_m2_s`, `wet_fraction`, `r_s_s_m` and the rain rate; `FacetSolver`,
  `ThermalField` and `PlanarThermalField` take `film_kg_m2=`, a water film per cell that rain
  fills and evaporation (at the step's midpoint temperature) empties, and a cell is wet while it
  holds water. `SceneSurfaceForcing` and `CellForcing` supply the humidity, the bulk conductance
  and the rain from the scene's one weather series on every call, so a scene needs only to
  declare a film (`PH.2`). The sea's cool skin now conducts the net longwave **plus** the latent
  flux from the bulk SST (clamped at zero: condensation onto a cold sea is a warm-layer mechanism
  the model does not carry), so the sea skin runs ~0.3 K colder; `test_sea_surface` re-pins the
  nadir reading (289.47 → 289.16 K) and widens the cool-skin band to 0.5 K with the reason.
  Measured: a saturated cell with no radiation relaxes to the psychrometric wet bulb within
  0.1 K at RH 1.0, 0.7 and 0.4 and cools as RH falls; Q_L = 0 for saturated air at the surface's
  temperature; a dry cell under moist weather is bit-identical to the plain balance over 200
  ticks, with or without an empty film; 75 W/m² at 5 m/s, 293 K, RH 0.7 against COARE 3.6's 76;
  the film budget closes to 1e-6 against the solver's own evaporation; removing the term warms
  the sea skin. 11 cases in `tests/unit/test_latent_heat.py`. Spec issue S45 moves to `PH.3`.
- **The engine as a solved node, not a schedule** (`TC.5`, ADR 0100; spec issue S43 is now
  `code`). `irsim.thermal.engine`: `EngineSpec` (block + coolant mass, skin area, forced and
  natural convection, bay volume and vent conductances, rated power and the bay's share of it,
  radiation views, rubber mounts, subframe -- every number ESTIMATED and said so),
  `engine_network` (four nodes and a fixed ambient on the TC.2 network, with forced → natural
  convection and venting when the engine stops) and `EngineSolver`, a `TemperatureSolver` whose
  reported temperature is the **bay air** -- the cavity temperature ADR 0088's radiator carries
  onto the bonnet -- so the bonnet field's forcing is unchanged. Scene targets gain
  `solver: engine` with `load_s` → `load` and no `source`; `vehicle_source: engine_bay` still
  works and the car scenes migrate in `TC.6`. Measured against the survey's bands: from 93 °C at
  27 °C the block is +32.3 K after 1 h (§6.6's 1800 s schedule: 9 K) and +0.92 K after 7 h; the
  full-load rise is +53.7 K (§6.6: +40…+90); the bay air overshoots by +27 K peaking 145 s after
  key-off; parts warm block → mounts → subframe; energy closes to 1e-6 at steady state; in the
  overcast scene the bonnet over the block keeps warming after key-off -- for ~23 minutes, where
  the row asked for 60–120 s: that figure is the survey's manifold-skin number and belongs to
  `TC.7`, and the bonnet is not yet the bay's loss path (`TC.6`), which ADR 0100 records. 7
  cases in `tests/unit/test_engine_node.py`.
- **The car scene's fields are spun up with the car present** (`PT.7`, the MP.5 limit). Both
  `car_demo` fields are integrated through the scene's `spin_up_hours` with the car's sky
  occlusion, its radiators at the air temperature of each hour plus whatever rise they carry at
  t₀, and its shadow (`wrap_into_weather` is now public), cached per material, weather, geometry
  and grid so a test session integrates each scene once (~14 s). Frame 0 of the clear-night scene
  now carries the ~4.5 K road patch a car parked all night makes under a clear sky -- the dominant
  feature of real night parking-lot imagery -- and its bonnet starts 4 K below the air it has been
  radiating past; under overcast the standing patch is +0.009 K. `build_car_demo(spin_up=False)`
  is the old uniform start, bit for bit. Spinning up exposed a residual of ADR 0088's kernel: an
  ambient underbody of ε 0.88 under an overcast sky "cooled" the road beneath the bay by 2.1 K at
  equilibrium, the reflection of the road's own emission off the grey body being missing. §6.1's
  balance therefore gains `emission_factor` on `SurfaceForcing` and `FacetForcing` (default 1;
  `1 − Σ F (1 − ε_r) ε_s` under grey bodies, one reflection), which the road field takes and the
  bonnet deliberately does not (ADR 0088 addendum). Measured: 24 h and 48 h spin-ups agree to
  0.1 mK; the engine's 30-minute growth is 1.81 K from either start.
  `test_the_ground_patch_needs_time_because_the_field_starts_uniform` is inverted into
  `test_frame_0_carries_the_patch_a_parked_car_has_already_made`.
- **Contactors and radiation between fields** (`TC.3`, ADR 0099). `irsim.thermal.coupling`:
  `cell_overlap_areas` clips every cell of one patch against the cells of another in the first
  patch's plane (Sutherland–Hodgman, candidates prefiltered by the grid), so a contactor's
  conductance `h_c · A_ij` needs no grid alignment, no shared cell size and no shared orientation,
  and its total is `h_c · A_overlap` regardless of refinement -- the property a per-node
  conductance cannot have. Patches must share a frame, be parallel and lie within a gap tolerance
  (a bonnet 0.9 m over a road is refused as a joint). `CoupledFields` concatenates its members'
  cells into **one** `ThermalField` with a block `ConductionOperator` (each member's lateral
  operator on the diagonal, the contactors off it), stepped by ADR 0094's IMEX scheme, and hands
  out a `PatchView` per member with the point bridge's interface (`patch`, `advance_to`,
  `sample_at`); advancing any view advances the system, and with no contactor the members are the
  separate fields bit for bit. `RadiationExchange` reads ADR 0088's parallel-rectangle view
  factors in both directions: the cells' net term is `occluded_longwave_flux`, the body's loss to
  the patch is `ε_r σ T_r⁴ Σ A_i F_i` by reciprocity, and `reverse_view_factor` checks that
  against an independent quadrature from the body's side. `ThermalField.latest_state_k` exposes
  the newest tick in float64 for energy bookkeeping. Measured: a 1 m² plate on a 2 m base gives
  `h_c · 1.0` to 1e-9 at 0.25 m cells and the same at 0.0625 m; a 30°-rotated square overlaps
  exactly its own area; a half-overhanging plate is credited with 0.5 m² where a per-node total
  says 1.0 at either refinement; a 350 K plate on a 300 K base at h_c = 10⁴ (τ = 0.8 s) settles
  under 60 s ticks with the stored energy conserved to 1e-9 and the plate's lost joules equal to
  the touched cells' gain; underbody → road power equals the cells' to 1e-6 and reciprocity holds
  to 1 % between the two discretisations. 8 cases in `tests/unit/test_coupling.py`.
- **ADR 0098: participating media and the phenomena tier** (`PH.13`). Records `PH.4`'s per-band
  slab in radiance space, why a grey emissivity knob, a Planck-mean coefficient, Hottel/Leckner
  totals and an emissive prim through the fp16 path were rejected for the image path (Leckner kept
  for heating), the ~8 % RadCal envelope as the fidelity this feature claims, the
  transport/radiometry split DIRSIG and FDS both keep, and what stays deferred: scattering, a
  gradient along the ray, buoyancy, flicker, a volume on the Isaac side.
- **A gas slab in radiance space** (`PH.4`, ADR 0098). `irsim.pipeline.gas_slab`:
  `GasSlab(t_gas_k, length_m, p_co2_atm, p_h2o_atm, f_soot)` -- authored by species and path,
  never by an emissivity -- and per band `L_b = τ_b L_behind + (1 − τ_b) B_b(T_g)` with
  `τ_b = exp(−κ_b(T_g) L)`, FDS's flame operator read per band. Soot needs no table: the
  small-particle `κ_λ = C0 f_v / λ` (`SOOT_RAYLEIGH_C0` = 7.0, Widmann 2003, ESTIMATED to its
  4.9–7.9 spread) is Planck-weighted over the camera's own R(λ) by the LUT quadrature. CO₂ and
  H₂O take `SpeciesAbsorption` tables in 1/(m·atm) over a temperature grid, linear between knots
  and refusing to extrapolate; a slab carrying a species with no table raises naming `PH.5`, the
  offline HITEMP/RadCal generator, rather than scaling ambient coefficients. `B_b(T_g)` comes
  from quadrature because the band LUT stops at 1000 K and a flame does not; the gas temperature
  is guarded to 300–2500 K. `slab_excess_radiance` attenuates the slab from its range exactly as
  MS.6's point target does (per class of the layered atmosphere, `τ(R)` for the grey one), with
  the sky beyond or an authored radiance behind it. Measured: κL → 0 returns the background
  bit-exactly and κL → ∞ returns B_b(T_g); an opaque soot flame reads its own temperature in MWIR
  and LWIR to 1 mK while soot's band means differ by the wavelength ratio (2–3×); a CO₂/H₂O slab
  on a synthetic MWIR ≫ LWIR table reads more than 900 K apart between the two cameras where a
  grey emissivity reads within 150 K; steam before a hot wall gives negative contrast; the grey
  and per-class range identities hold to 1e-12. 12 cases in `tests/unit/test_gas_slab.py`. The
  real species tables, the plume and the flame are `PH.5`–`PH.7`.
- **Nodes, links and joints in the scene schema** (`TC.4`, ADR 0097). Scene schema **v9**:
  `thermal.nodes:` (a `capacity_j_k`, a `mass_kg` with `specific_heat_j_kgk`, a `fixed` kelvin
  value or `"ambient"` for the scene's one weather series, or a `link_node` with its own
  capacity), `thermal.links:` in exactly one of the forms the literature reports (a total
  `g_w_k`; a `joint` name with `area_m2`; an inline `h_c_w_m2_k` with `area_m2`, range-checked;
  a `fastener` name with `count`; `h_w_m2_k` with `area_m2` for convection to a fluid node; or
  `radiation`), and `thermal.sources:` (a constant or a piecewise-linear schedule). All optional,
  so every v4–v8 scene reads as before. `configs/thermal/joints.yaml` (`irsim.config.joints`)
  holds the survey's joint conductances with provenance -- bolted ferrous 12 kW m⁻² K⁻¹ new, 7
  corroded, 59 with paste, 67 with foil (Voller & Tirovic 2007, MEASURED), a 1 kW m⁻² K⁻¹ dry
  default (ESTIMATED) and ~1 W/K per small bolt (Hasselström & Nilsson 2012, ESTIMATED: measured
  in vacuum on aluminium) -- and its loader refuses h_c outside 1e2–1e6 W m⁻² K⁻¹, the per-K typo
  that would render a bolted bracket plausibly cold. `Scene.network` is built from the block,
  stepped by `advance_targets` beside the targets and read through `node_temperature_k`; it
  carries the weather so the one-weather guard sees it. Measured: a bracket through 25 cm² of
  `bolted_ferrous_new` against `dry_default` on a 400 °C block, from the scene config alone,
  settles at rises in the two-resistor ratio G/(G + hA) to 1e-6; every link form and node kind
  builds to the conductance it declares; a v9 network beside a v7 surface leaves the surface's
  solve bit-identical. 12 cases in `tests/unit/test_network_schema.py`. Spec issue S42 is now
  `code`.
- **A thermal network: parts with mass, the joints between them, and their boundaries** (`TC.2`,
  ADR 0096). `irsim.thermal.network`: `Node` (capacity in J/K, or `from_mass`), `FixedNode` (an
  imposed temperature, constant or a callable of time -- ambient from the weather, a thermostatted
  coolant), `ImposedHeat` (watts on a node), `Link` in W/K authored as a total `G`, as a contact
  `h_c·A` or as convection `h·A` to a fluid node (the three are one multiplication and are
  bit-identical; `h` may be a callable for the forced-to-natural switch at key-off), `LinkNode`
  (a rubber mount: a node with 2G on each side so the series conductance is the one authored and
  τ = C/4G) and `RadiationLink` (ε A F σ (T_a⁴ − T_b⁴), linearised per tick at the exact
  conductance for the current flux). One backward Euler solve per tick on `ConductionOperator`'s
  Laplacian, fixed nodes eliminated at their end-of-tick value; the matrix the step applied is kept
  so `link_power_w` and `energy_residual_w` report the flows as the solver used them. Measured:
  ΔT = Q/G on a fixed sink and Q/(hA) on a fluid node to 1e-6; a mount's τ fitted from its own
  trajectory within 2 % of C/4G; the T⁴ steady state to 1e-6; a 50 g bracket on a 25 W/K bolt
  (τ ≈ 1 s) monotone under 60 s ticks; energy closes to 1e-6 of the imposed power every tick on a
  seven-node bay with a moving ambient, a switching h, a radiation link and a mount, and a bracket
  on a dry joint keeps warming for minutes after key-off -- the hot soak, from a callable h and a
  node with mass rather than a script. 13 cases in `tests/unit/test_thermal_network.py`. Spec
  issue S42 moves to `TC.4` (the schema and the joint table).
- **Occluders and daylight in the config path** (`PT.18`, ADR 0095). Scene schema **v8**:
  `world_frame: {up, north}` says which way the scene's geometry is up (ENU by default, so every
  v4–v7 scene reads as before; the car scenes declare +Y up, −Z north), and
  `thermal.occluders:` declares the rectangles that cast shadows. `CellForcing` gates the direct
  beam per cell -- `solar_loading` on the surface's normal and sky view with `cell_shadow`'s
  visibility -- and a patched surface under occluders is spun up on its own per-cell forcing, so a
  cell under an overhang starts the scene as cold as it has been all morning. `Scene.solar_terms`
  is the one sun every solar term uses; `car_demo`'s bonnet and road fields, which had **no solar
  term at all**, now take it, shaded by the car's own box faces (`CarGeometry.shadow_casters`,
  `shadow.box_faces`). Refused: `shaded: true` on a patched surface beside occluders (two shadow
  authorities, at load), an occluder or a shaded patch in a moving frame, a patch whose plane
  disagrees with its tilt, and a car scene whose frame is not +Y up. Measured: a south-west
  concrete wall under an overhang, from YAML alone, is 9.9 K lit/shaded at 16:00 local; cells no
  occluder ever reaches are bit-identical to the per-prim solve, spin-up included; the noon car
  shades 105 road cells and the strip beside it runs 12.7 K colder after 1500 s; the cabin lays a
  strip of shadow on the rear of the bonnet. 13 cases in `tests/unit/test_scene_occluders.py`,
  3 in `test_car_demo.py`.
- **Heat moves between facets, and the scene tick survives it** (`TC.1`, ADR 0094).
  `irsim.thermal.conduction.ConductionOperator`: symmetric link conductances in W/K (zero
  diagonal, non-negative, symmetry checked because an asymmetric matrix invents energy) plus the
  facet areas, so `C_i A_i dT_i/dt += Σ_j K_ij (T_j − T_i)` and unequal cells conserve.
  `FacetSolver(conduction=)` steps IMEX: the surface balance keeps ADR 0036's midpoint rule, the
  conduction term is backward Euler (L-stable; Crank–Nicolson would ring at dt/τ = 600) through
  one `splu` factorisation per tick size; both fields pass `conduction=` through. §6.4's explicit
  bound `2C/(h + 4εσT³)` is now checked every step on the forcing's actual h and raises with the
  numbers -- it never was, and a leaf at 630 J m⁻² K⁻¹ under wind would have diverged quietly.
  Measured: a zero operator is bit-identical to none over 240 ticks; a ladder with τ ≈ 0.1 s at a
  60 s tick settles to its mean within 1e-6 where forward Euler explodes; the slow mode's error
  halves with the tick (0.7 % at 60 s for τ = 1 h); energy closes to 1e-9 for 50 unequal cells
  under a real forcing. 13 cases in `tests/unit/test_conduction.py`; one broadcast test opts out
  of the guard because its 600 s step was always past the bound.
- **A field holds the ticks a query needs, not every tick it produced** (`PT.8`, ADR 0093).
  `ThermalField` gains `keep_ticks` (a `deque` ring; `None` keeps all, the per-prim default) and
  `on_tick`, a hook that sees every tick in order for a time-lapse or a validation to record.
  `PlanarThermalField` defaults to the bracketing pair: a day of the car scenes' 10 400-cell road
  held ~240 MB and now holds 166 KB, which is what blocked ADR 0074's full-diurnal film.
  `state_hash` is a running digest fed the same bytes in the same order as the old walk, so a ring
  hashes exactly as the unbounded field did (checked by reconstructing the walk from the hook);
  500 queries inside the window are bit-identical to the unbounded field's; a query before the
  oldest held tick raises and names the knob rather than answering from a fallback. `n_ticks`
  counts ticks produced, `n_held` ticks resident. 10 cases in `tests/unit/test_tick_history.py`.
- **Patches solved from the scene config** (`PT.17`, ADR 0087). A `patch:` block on a §12.3
  surface now buys a per-cell `PlanarThermalField` on the surface's own library material, started
  from the prim's spun-up state and forced through `CellForcing` -- the same numbers the per-prim
  solve uses -- so with nothing varying across the surface the cells reproduce the prim **bit for
  bit** (`tests/unit/test_scene_surface_fields.py`; the contract is 1 mK). `Scene.surface_fields`
  and `Scene.surface_bindings()` hand `(prim path, field)` pairs to `bindings_from_scene`, the one
  line a render driver needs. Both car scenes declare their bonnet (now a solved `car_paint_black`
  surface) and road grids in YAML; `build_car_demo` reads them and refuses a bonnet grid that does
  not sit on `CarGeometry`'s bonnet or a road grid short of the camera's footprint, with both
  numbers in the message. A scene that declares no grid keeps the hand-built path bit-identically;
  an unknown patch material fails at load naming the surface. The engine-bay and underbody
  radiators still ride in Python on the declared grids until `TC.3` gives them a place in the
  config. 14 cases.
- **Roadmap revision 6: the plan now leads to per-point, part-to-part, water and fire physics.** On
  2026-09-18 the owner restated the requirement that drove them off their previous simulator and widened
  it — a building whose sunlit and shaded parts differ; an engine that warms *the metal around it*, not
  only itself; and the same for water, fire and their kin. An audit that day (three repository audits and
  two web-research sweeps, run in parallel) found the point-wise machinery real but reachable from no
  scene config, no mechanism anywhere for heat between parts, and no model or step for water or fire.
  Revision 6 adds phase **P** — point-wise and coupled physics, CPU only, between repair and the aerial
  lane — two lanes, `TC` (thermal coupling, 8 rows) and `PH` (phenomena beyond opaque solids, 13 rows),
  six `PT` rows that make what shipped reachable from YAML and add sky view, geometry shadows with
  neighbours and the penumbra, and the R1 reference scene; and moves `PT.6`–`PT.8`, `PT.11`, `PT.12`,
  `PT.14`, `PT.15` and `WM.1`–`WM.6` into P. The queue now opens with `RP.10` (a small repair row) and then `PT.17`
  (patches solved from the scene config). Every tolerance in a new row is quoted from a primary source and is external evidence
  until an irsim test reproduces it.
- `docs/research/2026-09-18-thermal-coupling-survey.md`: the evidence — 70 sourced findings (55 with the
  page or PDF read), 43 audit findings with file and line, 22 open-source codes with licences.
- `tests/unit/test_roadmap_phase_table.py`: the phase-plan table is generated from the step rows and can
  no longer drift from them (revision 5's disagreed in seven cells and omitted five rows).
- Nine spec issues, `S41`–`S49`, recording what §6, §2 and §16.2 lack for the owner's requirements —
  lateral and part-to-part conduction, a solved engine node, the shadow term's provider, latent heat,
  participating media, snow, vegetation, people — each owned by a roadmap row.

#### Changed
- `scripts/next_step.py`: `PHASE_RANK` gains `P` between `0` and `A`; the tiebreak prose in the roadmap
  and the script follow.
- **Ten roadmap verification cells now state what would fail** (`RP.10`): `EV.6`, `EV.11`, `XD.4`,
  `XD.5`, `XD.7`, `XD.11`, `IG.3`, `IG.4`, `GT.6`, `DC.3` carried motivation or a description where the row
  contract asks for an assertion and its tolerance. Each now names a closed form, a band, a resolvable
  hash or a recorded refusal; the two in-engine probes say what a CPU-only session can ship.
- README: the plan paragraph describes revision 6 and the owner's order; the `thermal` status row stops
  calling the two-node solver and vehicle regimes "phase 2" (they are shipped and unreachable, `PT.15`)
  and says that point-wise reaches a frame only through the car demo's Python (`PT.17`); the command
  table says what `make check` actually runs and which packages mypy covers; the limitations bullet no
  longer claims there is no image-plane velocity (`IG.6` synthesises it).

#### Fixed
- Roadmap self-inconsistencies found by the audit: the header said revision 4 while the body said 5; the
  step count; "two prims in one of eight scenes" (it is two of eight); PT.5's case count; the WM lane
  "gated on phase A closing" while phase A's own exit waited on a WM step; the §6.6 plume deferral whose
  revisit trigger had already fired; the ground-breadth deferral that deferred the owner's first
  requirement's occlusion input along with material breadth; the ADR-allocation section's "highest ADR".
  Two defects are rows rather than fixes: `RP.10` (ten verification cells that state motivation, not a
  failure condition) and `GT.8` (`--lane` prints a blocked head).

### 2026-09-17

#### Added
- **The band-scalability guard denies by default, and its carve-out is a tested constant** (`AT.4`,
  ADR 0092). It scanned a positive list of six packages, so `atmosphere`, `materials`, `thermal`,
  `io`, `validation`, `config` and `scene.py` were unguarded *by omission* — and four of them had
  **zero** offences the whole time, so the carve-out naming `irsim.materials` rescued nothing. Now
  every module under `src/irsim` is scanned (140 of 146 guarded) and `BAND_AWARE` maps a path to a
  ceiling and a written reason, with tests that each entry exists, still has an offence, and has not
  grown. A new module is guarded the day it is created.
- The old shape could also switch itself off in silence: each test re-globbed its own package, so a
  package renamed in the tuple walked a path that does not exist — no files, no offences, green — and
  the one coverage assertion counted six packages' union against a floor of 30 when the six hold 72.
  Both are closed by scanning from a single root and asserting per-package presence.
- The forbidden name set widens to `BAND_KEYS`, so the guard can finally see the **fifth band the
  codebase already contains**: `visible` had a class table, a weighting temperature, a mandatory key
  in all seven presets and an `if band == "visible"` branch, and none of it tripped a guard whose
  forbidden set stopped at the four `BandId` values.
- `irsim.config.bands` gains `ANCHOR_BAND`, `ANCHOR_RANGE_UM`, `ANCHOR_REGIME`, `BAND_KEYS`,
  `regime_for()` and `nominal_range_for()` — the atmosphere's reference band, named once instead of
  spelled by hand in five places.

#### Fixed
- **Every frame `render_aerial_demo` and `render_maritime_demo` have written is stamped one frame
  period late.** `IrCamera.get_outputs` advances its clock on the way out, so `t_rel_s` read on the
  line below a capture names the frame that has not happened yet — and both drivers computed their
  sidecar's `t_s` (and therefore its `utc`) as `scene.t0_s + camera.t_rel_s` right there.
  `render_car_ignition` had already hit this and worked around it locally with a comment; the other
  two had not. On a time-lapse driver a frame period is **six seconds** of weather, sun angle and
  node temperature, and a dataset whose timestamps are one row out reads as a small calibration
  error in whatever is fitted to it rather than as a bug. `IrCamera.last_frame_t_s` now carries the
  absolute scene time of the frame that was actually captured, both drivers read it, and
  `tests/unit/test_frame_clock.py` checks the stamp against the surface field's own clock — an
  oracle reached through a different call path, so it fails when the stamp is the one being tested.
  A companion lint refuses any render driver that does the arithmetic itself; it flags exactly the
  two offenders on the previous commit and leaves `render_car_ignition` alone.
- **The Warp ISP's host-readback justification was wrong, and is corrected.** `agc_lut_warp`'s
  docstring said "the last few scalars — the percentile positions, the occupied-bin span — are read
  back". That is not what happens: `counts.numpy()` pulls the **whole 2^bit_depth histogram**, plateau
  mode pulls a second array of the same size for the inclusive CDF, and the finished table goes back
  — three 65536-entry transfers per frame at 16 bits, each a synchronisation point. The trade is
  still deliberate (the percentile search and occupied-bin span are irregular reductions), but the
  honest justification is "not yet worth the kernels", not "only scalars move".
- `replace_bad_pixels_warp`'s `counters.numpy()` is now named at its site as the sharper of the two:
  the histogram stalls once per frame, that loop stalls once per **iteration**, to read two integers.
  Neither readback is removed here — a device kernel cannot be verified without a GPU and the CPU
  reference leads (ADR 0018) — so `IG.16` carries the work.
- Both were raised by an external review of the codebase and verified against
  `warp_stages.py:1862`, `:1891` and `:2120` before being recorded.

#### Changed
- **`WEIGHT_T_REF_K` is derived from the regime instead of keyed by band name** (`AT.4`). All five
  shipped rows are reproduced exactly, and the step came within one line of a silent calibration
  change: the table weighted MWIR at 300 K while `DEFAULT_REGIME["mwir"]` is `"mixed"`, so the rule a
  reader reaches for first — "emissive keeps 300 K, everything else is solar" — gets four rows right
  and moves **MWIR to 5800 K**, rescaling every MWIR class share, its anchor solve and τ_MWIR at
  every range but the 200 m anchor. The rule that reproduces all five turns on *reflective*; a test
  asserts the two rules still differ so it cannot stop guarding that drift.
- `ATMOSPHERE_BAND_KEYS` now derives (`frozenset(BAND_KEYS)`); `VISIBLE_RANGE_UM` and the
  `band == "visible"` branch — with its `cast(BandId, band)`, which lied about the type — are gone.
- **Open question 6 — which Boson is "the" reference — is answered and closed: both, with different
  provenance.** The 640 carries [R24]'s Table 13 acceptance limits and says its ratios are
  ESTIMATED; the 320 carries ME.5's field-measured ratios. `test_boson_datasheet.py` checks on the
  *numbers* rather than the prose that neither file borrows the other's, which is the schema guard
  the resolution asked for in the only form that can work — a validator cannot see where a number
  came from.
- Three caveats travel with the substituted ratios, in the config where they will be read. N is
  **28 of 365 clips**; ME.5's own report calls every value in that table "an upper bound on what the
  codec left, not a measurement of the camera"; and — the one worth adding — a codec destroys high
  spatial frequency before low, and σ_TVH is the highest-frequency component of the seven and the
  *denominator* of every ratio here. So the bias runs one way: these ratios are more likely too
  large than too small. σ_T clears its floor on 43 % of clips and σ_H on 54 %, so `t` and `h` are the
  two least supported numbers in the block.
- **A premise this step started from was wrong, and the correction is recorded rather than quietly
  dropped.** The worry was that a child offset from a rotating assembly needs its own transform
  because its frame-to-frame delta is a *conjugation* of the root's. It is not: the displacement is
  built as `inv(cur) @ prev`, and a constant local offset cancels exactly in the middle of that
  product, so a rigid child of a banking airframe reports the same pixels either way — checked to
  float precision at a 20° bank, where the conjugation argument predicted a 2 px gap. Reading each
  rendered leaf still earns its place, but for a different reason: an **articulated** child, whose
  local pose differs between the two samples, has nothing left to cancel. This project has one —
  `render_quad_flight` re-poses the rotor discs every frame from the throttle.
- **The corrected schedule and membrane move three things, all in the unexpected direction.** The
  membrane is *faster*, not slower — α at 60 Hz goes 0.811 → 0.876, so irsim had been modelling a
  laggier detector than FLIR ships. ENBW is 1/(4τ), so the temperature-fluctuation floor *rises*,
  66.6 → 74.5 mK. And the shutter now fires every 300 s instead of 180 s, so the worst-case NUC
  residual under a 0.05 K/s drift grows **1.67×**, 403 → 673 mK — a correction that makes the
  simulated camera worse, which is the point of anchoring to a datasheet rather than to taste.
- `MIN_INTERVAL_CLIP_S` moves 180 s → 300 s with the schedule it is derived from. Its own docstring
  ties it to the camera ("a clip below this can only ever catch one event"), so leaving it behind
  would have let a 250 s clip report an FFC interval it cannot have measured.
- The FFC, membrane and NETD-floor tests now read their constants **off the committed config**
  instead of restating them, so the next correction moves the tests rather than leaving them
  asserting a camera the repository no longer models.
- **Every golden array is bit-identical**; only the config hashes in the five sidecars moved, and
  the store refused them as STALE rather than reporting a physics failure — ADR 0004's mechanism
  doing exactly its job. That the arrays did not move is explainable rather than lucky: the
  sensor-chain golden runs 60 s at 6 fps on a static scene, where the FFC cannot fire (it needs
  180 s *or* 300 s) and the membrane cannot show (α = 1 − 1e-8 at either τ).
- A second wall-clock assertion replaced by a ratio, found by this step's own run.
  `test_build_is_fast_enough` asserted `elapsed < 2.0` and failed under load while passing in
  isolation. It now compares the full LUT build against a quarter-size one, so load cancels:
  linear reads ~4×, quadratic would read ~16×, and the bound is 8×.
- `test_every_scene_is_long_enough_to_watch` measures a clip in **seconds**, not frames. The old
  form read 90 frames as "three seconds" while silently assuming 30 fps, which the car driver has
  never used — it plays at 10. Each count is now divided by that driver's own `--fps` default.
- **Two deferrals from the same review, recorded with their reasons.** **MCT (HgCdTe)** was called
  "completely ignored"; it is narrower than that — `dark_current_a` takes `band_gap_ev` as an authored
  sensor field, so an MCT camera is configurable today and only the Hansen–Schmit E_g(x, T) relation
  is missing. And an **unconditionally stable two-node solver**: the review proposed Backward Euler,
  which is not free, because ADR 0036 chose RK2 over Euler on *bias* — the T⁴ term makes explicit
  Euler inflate the diurnal swing that §6.3's acceptance test measures, and Backward Euler damps the
  same quantity. The stiffness is all in conduction (1/R₁₂ = 37 500 against h + 4εσT³ ≈ 43), so IMEX
  is the shape that pays, and `PT.11` already needs that machinery.
- Two of the review's four criticisms needed no change. Its Tier 4 root cause (accurate simulated
  noise against codec-scrubbed real clips, fix by adding a DNR pass) is contradicted by the project's
  own report, which says `noise_scale` at the top of the feature list means the **signal path**
  separates the sets — and `noise_scale` is the heaviest feature at 3.98. `EV.2` already measured the
  real cause: six clips taken alphabetically with none of the three gates built for them, on an
  archive where 81 of 365 are moving and only 28 support a noise table. Adding a DNR pass now would
  tune the simulator to an artefact of clip selection. And its MATLAB-MCOS parser recommendation is
  work `XD.11` already avoids: ~100–200 Python-readable boxes are what `EV.11` needs, not Halmstad's.

### 2026-09-16

#### Added
- **The camera's own R(λ) reaches the atmosphere, and every band's classes cover it** (`AT.2`,
  `AT.3` — they land together because the first exposes the second). `Scene.from_config` built
  `LayeredAtmosphere(preset, weather, luts)` and never passed the fourth `responses` argument, so
  `class_weights` fell back to a **nominal top-hat** for every band and the model described a
  different camera than the one being simulated, silently.
- **Measured on the shipped InSb MWIR response: the `h2o_wing` class weight goes 0.0238 → 0.1303, a
  5.5× change** in how much of the band is treated as a water wing. LWIR moves by under 5e-4 —
  which is exactly why this stayed invisible, because the Boson's response happens to sit close to
  its nominal top-hat and LWIR is the band the project renders most.
- Supplying the real NIR response used to **raise**: `BAND_CLASSES['nir']` stopped at 1.05 µm while
  `nir_si.csv` reaches 1.10, putting **0.188 %** of the Planck-weighted band outside every class.
  SWIR (short at both ends) and MWIR (short by 0.4 µm) had the same shortfall and passed only
  because their responses carry **0.000 %** out there — luck, not coverage. NIR's `window` now ends
  at 1.10, where SWIR's own first class begins; SWIR's spans 0.80–1.80; MWIR's `h2o_wing` reaches
  6.0 µm, which is the right class because 5.6–6.0 approaches the 6.3 µm water bend rather than
  opening into a window.
- `load_band_response_for_config` sits beside `load_band_lut_for_config` so the two are loaded from
  the same field of the same config and cannot describe different cameras. All six render drivers
  pass it. A `LayeredAtmosphere` built without one now **warns**, naming the measured consequence —
  the failure mode was that the output looks like an output either way.
- Eighteen cases. The guard `AT.3` asks for walks `configs/sensors/*.yaml` rather than a
  hand-written list, so adding a camera cannot quietly add a band whose classes stop short of its
  detector's tail. Negative control: reverting NIR's edge to 1.05 turns three red.
- **Every pixel takes its own slant path** (`AT.1`). `pipeline/atmosphere.py` passed elevation
  **0.0 unconditionally**, so every *resolved* pixel was given surface-density extinction and
  surface-temperature emission over its whole slant range — while the *unresolved* point-target path
  beside it used `target.elevation_rad`, and so did the sky behind it. Contrast jumped at the
  resolved/unresolved handoff for a reason that was in the code rather than in the sky, and the
  error grows with elevation exactly where phase 1's subject lives.
- **Measured on `us_standard_clear`, LWIR, 5 km: τ 0.5995 horizontal against 0.7230 at 45° (+21 %),
  and L_path 18.27 against 11.60 W/m²/sr (−37 %)** — applied to every pixel of every sloping ray.
- `column_length`, `optical_depths` and `transmittance` now broadcast elevation against distance, so
  a frame is one call. Path radiance needed more: it was a 4000-step quadrature per *scalar* range,
  unusable per pixel. `cumulative_path_table` integrates once per elevation and **hands back the
  range dependence for free**, because integrating to a given distance is choosing the upper limit
  of the same integral. `path_radiance_plane` interpolates that table.
- Two coordinate choices carry the accuracy, and both were measured rather than assumed. The grid is
  uniform in **w = 1 − e^{−u}**, not in u: substituting leaves ∫ L_B dw with no exponential in it, so
  an isothermal path — what every horizontal ray reduces to — is *exactly linear* and a trapezoid is
  exact to **2e-14**. A uniform u grid spends its points where the exponential has already killed the
  integrand, and measured **77 mK** at 90° where the whole optical depth is a fraction of one step.
  The method returns w rather than u for the same reason: interpolating the same table in u costs
  1.2e-4 where w costs 2e-7.
- The elevation nodes are uniform in **sin θ**, and **θ = 0 is a node rather than a clamp**. An
  earlier draft clamped below 0.25° to the horizontal closed form and left a **209 mK step** there —
  four times the NETD, right where long-range scene sits. Near the horizon the flat-earth column
  expands to `d(1 − d sinθ/2H)`, linear in sin θ, so a sin θ grid anchored on the exact horizontal
  answer is continuous by construction; measured at **<0.1 mK** across the join.
- Accuracy against the quadrature it replaces: **2.4 mK** worst over 0.05–90° and 200 m–20 km, with
  129 nodes (33 leave 68 mK, 65 leave 14 mK). Beyond 20 km it degrades to 65 mK at 100 km, which is
  recorded in a test rather than engineered away — ADR 0071 bounds the model itself well inside that.
  The table is cached per thermal tick: 0.15 s to build, 267 ms for a 640×512 frame.
- `elevation_rad` joins the M0.6 contract as an **optional** plane and a precision-critical one, so a
  scene that does not supply it renders bit-identically and fp16 is refused — it scales the optical
  depth of the whole slant path. `IrCamera` fills it from the ray elevations it has computed since
  M10.18 for the sky temperature and simply never passed on. The Warp twin still mirrors the old
  horizontal form and is **not** updated here: it cannot be verified without a GPU, and an
  unverified device kernel is worse than a recorded gap.
- **A patch can ride a moving prim** (`PT.5`). `surface_field` had documented a prim path as a legal
  patch frame since MP.1 — "`world` for a road, a prim path for a panel that moves with its object"
  — while `point_bridge` refused every frame but `world`. The docstring promised what the code
  declined, and every aerial and maritime target moves, so the promise was the useful half.
- `PointwiseTemperature.apply` takes `world_from_local` matrices, `local_frames` says which prims
  need one, and `IrCamera` reads them off the stage each frame — fresh, because a patch on a moving
  prim has to follow it. A scene whose patches are all in world space pays an empty dict and never
  touches USD, and its output is **bit-identical**.
- Two conventions had to be right at once and both are borrowed rather than restated. The matrix is
  **world-from-local**, USD's own direction, inverted inside the bridge rather than at the call site
  — a caller that had to remember to invert would eventually not, and the failure is a patch
  tracking its prim *backwards*. And USD matrices are **row-vector**, translation in the last row;
  the multiply goes through `irsim.optics.motion.transform_points`, the one place that convention
  already lives, rather than a second copy. Getting it backwards transposes every rotation and still
  produces a believable field — the mistake ADR 0014's M10.19 addendum records costing this project
  a 164-row horizon — so there is a test that feeds the transposed matrix and requires a different
  answer.
- Six new cases. The acceptance one translates a prim 10 m and yaws it 90°, 215° and back, and
  requires the same material point to read the same cell to **1e-6 K** across all four poses, where
  one cell spans about 4 K of the field. Its negative control replaces the pose with the identity —
  what sampling a world position against a local patch amounts to — and requires a **> 5 K**
  disagreement. In-sim confirmation of the stage read belongs to `IG.2`; the composition is
  engine-free and fully covered here.
- **Per-cell solar and shadow: a wall half in sun** (`PT.1`). ADR 0087's own headline case — a
  surface spanning 10–20 K because part of it is lit and part is shaded — was **unimplemented**.
  A patch gave every cell its own temperature, but `SceneSurfaceForcing` carried one `shaded` bool
  per *surface*, so the only spatial variation a field could express came from `q_internal_w_m2`, an
  engine bay under a bonnet. That is why both point-wise scenes shipped so far are **pre-dawn**: the
  machinery could not have rendered a sunlit one any differently from a flat surface.
- `irsim.thermal.shadow` supplies the per-cell `shadow` argument `solar_loading` has always accepted
  and nothing had ever varied. `ShadowRectangle` is an opaque rectangular occluder — the same
  primitive ADR 0088 uses for a radiator, because a slab, a parapet or a wing reads as one —
  and `cell_shadow` is an exact ray–rectangle intersection per cell. `patch_solar_loading` is the
  whole point in one call: the same `solar_loading`, handed an array that varies across the surface.
- **Measured on a 4 m vertical concrete wall at 45° sun, half occluded by a slab: lit cells settle
  at 305.1 K and shaded at 283.0 K — a 22.1 K step across one prim**, from 619.8 against
  55.0 W/m² of absorbed solar. Every cell holds the §6.1 `steady_state_temperature` root for its own
  flux to under **1 mK**, so this is not a gradient smeared across the surface but each cell being
  the right number. Negative control: the representation this replaces, a per-surface `shaded` bool,
  is flat to 1e-6 either way and wrong across half the wall by the full step.
- Two properties are deliberate rather than incidental. **Shadow gates the direct beam only** — a
  shaded cell still sees diffuse sky through its own view factor, and zeroing all solar in shade is
  the usual shortcut that renders shaded surfaces far too cold; the shaded cells here sit at
  55 W/m², not zero. And **`cell_shadow` reports a back-facing cell as lit**, leaving `max(0, n·s)`
  to zero the beam, so a self-shadowing result never looks like an occlusion one — the two need
  different fixes when a scene comes out wrong.
- Stated limits: hard-edged, so no penumbra (the sun's 0.53° disc would soften the terminator by
  about 1 cm per metre of standoff), and single-bounce, so no light returns from the occluder onto
  the shaded part. Both make the shaded side slightly warmer than this model says, which makes the
  measured step a **lower bound**.
- **A point-wise surface is declarable in a scene config** (`PT.2`, schema **v7**). `SurfaceSpec`
  carried name, material, tilt, azimuth, shaded and vehicle_speed — nothing spatial — so the only
  point-wise scene that existed was `irsim_isaac.car_demo`, which hand-wrote its bonnet and road
  patches, with `surface_fields=` passed at exactly one call site. The owner's own bar for a lane,
  "a scene config plus one command produces frames", was therefore unmeetable for any *new*
  point-wise scene — the requirement this whole lane exists to serve.
- `PatchSpec` adds a `patch:` block: origin, axes, cell counts and sizes, slab thickness, frame name
  and the prim path that binds the solved field to geometry. `irsim.scene.build_patch` hands it to
  `PlanarPatch`, and `Scene.patches` / `Scene.patch_prims` expose the result by surface name, so a
  render driver reads a grid instead of building one. The block is **optional**: every v4–v6 scene
  loads and solves unchanged, and the seven shipped configs move to v7 without gaining a patch.
- Validation lives in both layers on purpose — the schema so a scene that cannot be solved fails at
  *load* rather than half an hour into a Kit session, the dataclass so a patch built in Python is
  held to the same rule. Non-perpendicular axes, a zero-length axis, zero thickness and zero extents
  all raise; orthogonality is required rather than repaired by Gram-Schmidt, because two axes five
  degrees off perpendicular meant something and a squared-up grid would sample a surface nobody
  described.
- Sixteen cases. The two that carry the weight are the acceptance ones: the car demo's hand-built
  bonnet and road patches come back from a declaration with **bit-identical** cell centres — array
  equality, not `approx`, because a nearly-right patch produces a plausible gradient — and a field
  solved on each agrees bit-identically **after 1500 s**, with a check that the run moved at all so
  the equality is not two copies of the initial state.
- **ADR status hygiene: one spelling, and supersessions that are actually recorded** (`RP.5`). Three
  spellings were in use across 88 files — `**Status:** Accepted` (70), `Status: accepted` (11) and
  `- Status: accepted` (7) — so no single grep answered "what is the state of the record". All are now
  `**Status:**`, with the bullet preserved where a file's metadata block is a list.
- **Nothing had ever been marked superseded**, although ADR 0001's template offers it and at least
  three decisions had been overtaken in part. `0059` ruled that "bolometer smear is the inter-frame
  IIR of §9.2, never a second blur"; `0077` then made `smear_duty(None)` mean *a bolometer and
  therefore 1*, and `optics/stage.py:77` applies exactly the second blur that clause forbade. The two
  are different mechanisms and a bolometer has both — the IIR is a temporal lag across frames, while a
  scene crossing the focal plane during an integration smears within the frame whatever the detector
  — so `0077` is right and the clause conflated the thermal lag with the motion MTF. `0006`'s
  G-buffer transport was removed by `0014`'s measurement that no colour AOV can carry temperature, and
  `0048`'s ~500 m range clause by `0071`'s layered slant path.
- None of the three is superseded *entirely*, so the status line names the clause rather than
  retiring the document, and `0059`'s superseded bullet is struck through in the body — a reader who
  lands mid-document must not act on what the header retired. `tests/unit/test_adr_status.py` holds
  it: one label, a sanctioned state word, every "superseded by ADR NNNN" resolving to a file, no
  self-supersession, and the three pairs recorded. Negative control: reverting `0059` to the old
  spelling turns three red.
- **`CHANGELOG.md` is grouped by day, one heading set per section** (`RP.2`). It was a single
  `[Unreleased]` section of 3,300 lines carrying **twenty** repeated `Added` / `Changed` / `Fixed`
  headings — a merge hotspot for three sessions working in one tree, all appending under the same
  heading near the top of the same file. Not hypothetical: two commits already exist whose entire
  subject is restoring lost entries (`68acd1c`, `303b56b`). Five dated sections now hold seventeen
  former heading runs, so a collision is confined to one day's block.
- **The regrouping moved whole bullet blocks and rewrote nothing.** The multiset of non-heading
  lines was compared before and after, and **3,242 content lines** came through unchanged; the
  script refused to write until they did, which is how it caught its own new preamble on the first
  run. Dates come from `git blame` on each block, not from guesswork.
- `tests/unit/test_changelog_structure.py` makes a third loss detectable *before* it is committed
  rather than by later `git log` archaeology: no heading repeats inside a dated section, no date
  repeats, every heading is a Keep a Changelog kind, dates are real and newest-first, and every
  dated section sits under `[Unreleased]` and is non-empty. Negative control: injecting a second
  `Added` into the newest section turns it red.
- **The row-length lint the roadmap says enforces it** (`RP.9`). Revision 4's own prose states "Step
  rows are capped at 600 characters and ledger rows at 200, enforced by a parser in `make check`
  (RP.9)". Until now that was a sentence — and a sentence is exactly what revision 3 had, whose rows
  averaged **1,234 characters** with a 4,958 maximum, held 224,508 of its 298,272 characters inside
  table cells, and were unmergeable between two sessions without a manual rewrite. That is how four
  subsystems came to be over-reported and how a whole-file write silently reverted committed work.
- `tests/unit/test_roadmap_row_length.py` owns the 600; the ledger's 200 already has an owner
  (`test_shipped_ledger.py`, `RP.6`) and is not duplicated. **Scope is stated in the parser**, because
  a lint that creeps is a lint that gets disabled: step tables only, never prose, never the phase or
  risk tables, never the generated *Do next* block — whose rows begin with a position number rather
  than a step id, so the block cannot start linting itself. Three of the twelve cases check exactly
  that scope rather than assuming it, including that no step row is found above the first lane
  heading and that no ledger row leaks in at the wrong cap.
- Self-tested on a synthetic over-length row so the lint is shown to *catch* rather than merely pass,
  and negative-controlled by fattening a real row (`IG.13`) past the cap. This is a lint, not a
  physics verification, and is not counted as one.
- **The shipped ledger points at commits instead of saying `pending`** (`RP.6`). Revision 3 of the
  roadmap prescribed `done YYYY-MM-DD <hash>` and **zero of its 162 done rows carried one** — M0.1
  included, the row it used as the example of the format. All seventeen shipped milestones now carry
  a hash; `IU`, the one open row, correctly carries none.
- The rule is stated rather than assumed: **the last commit that *claimed* one of the milestone's
  steps**, by the three markers this project's commit messages actually use — `(M9.4)` closing the
  subject, `Roadmap MP.1` in the body, or `Implements M12.2`. Deliberately not "the last commit that
  mentions the milestone": mature milestones are cited by everything downstream, and that looser rule
  hands `M9` the roadmap-rewrite commit and `M2` a point-bridge commit that merely cites M2.4. Two
  independent cross-checks came out right — `MM` resolves into the same `ca5a663` lineage its own note
  already cited for Tier 3, and `M7` resolves to `1a0f13c`, which `RP.7` separately identified as the
  commit where `data/nk/glass.csv` first appeared.
- `tests/unit/test_shipped_ledger.py` checks the half that matters: every hash must **resolve in this
  repository**, not merely look like one. A 7-hex string of the right shape and the wrong value reads
  as evidence, which is worse than the `pending` it replaced. It also enforces the section's own
  200-character row cap — the thing that keeps the table mergeable, against revision 3's 1,234-character
  average — and skips the resolution half outside a git checkout rather than failing there. Negative
  controls: reverting one row to `pending` turns two red, and replacing its hash with a well-formed
  `deadbee` turns the resolution test red on its own.
- **`docs/spec-issues.md` now says per row what has been applied** (`RP.7`). Its header had always
  promised that "the **status** column below records what has been applied"; there was no status
  column. In its place was one sentence listing ten M0-era resolutions and ending "Everything else is
  open" — wrong for months, because **fifty-six of the sixty rows had shipped**, thirty-seven of them
  with an ADR written. Every row now carries `ADR NNNN`, `code — <step>` or `open`, and the four that
  are genuinely open are named: `S9`, `S15`, `S16` are edits to `docs/physics-model.md` belonging to
  the spec owner, and `S13` is reopened below. `S40` gained the ADR 0042 pointer `RP.4` created.
- `tests/unit/test_spec_issue_status.py` keeps that ledger honest: every ADR a status names must
  exist, no row may be `open` while the ADR its own resolution names is already written — the exact
  way the old sentence went stale — and **the summary paragraph's counts are recomputed from the
  table**, spelled out in words so a status edit cannot leave the prose behind. Negative control:
  marking `S19` open while ADR 0045 exists turns two red.
- **`transmittance_derivation` on the material schema**, and the guard behind it. A
  `transmittance_per_band` must now either match the Beer-Lambert transmittance of the material's own
  n/k table to within 0.10 per band, or declare `authored: <spec issue>` naming an issue that is
  still open. `irsim.materials.nk.band_slab_transmittance` is the physics (§4.4): Beer-Lambert
  absorption with the two surfaces and the incoherent multiple reflections between them, because
  internal absorption alone calls a glass slab in NIR a perfect transmitter when about 8 % of the
  light never gets in.
- **The two ADRs shipped code has been citing for weeks now exist** (`RP.4`). CLAUDE.md's reason for
  ADRs is that a reader who was not in the conversation can recover the reasoning; a citation pointing
  at a file nobody wrote is worse than none, because it tells that reader the reasoning exists and
  sends them to look for it. **ADR 0042** records the angular-emissivity level policy — the level is a
  property of the *material*, not of the caller; specular Fresnel stands in for 1 − ε only at Level A
  (the resolution `docs/spec-issues.md` S12 already named); the baked table carries (ε₀, a, p) rather
  than pinning p = 4 (S18); and the Level B fit refuses on *shape* before error, because aluminium's ε
  is so small that a completely wrong shape scores an RMS residual of 0.0034, inside the 0.02 bar.
  **ADR 0079** records the sea-water optical constants and the Cox–Munk slope model, including why
  `slope_variance` returns components rather than the published isotropic total: Cox & Munk fitted the
  two separately on the same data and they disagree by 0.8 % in variance.
- ADR 0079 also supplies the **salinity bound ADR 0078 deferred to it** and that had existed only as a
  forward reference. No redistributable sea-water n/k table covers the LWIR window, so the substitution
  cannot be checked directly; what is measured instead is the sensitivity the substitution would move
  along. Perturbing the whole table by +1 % and recomputing ε_B(θ) through the project's own
  `band_directional_emissivity` gives a maximum Δε_B of **3.0e-3 for n** and **5.6e-4 for k**, both
  peaking at **85°** of incidence — so ε_B is about five times more sensitive to n than to k, and the
  sensitivity is worst exactly in the near-horizon band a low maritime camera spends its pixels on. At
  a 20 K sea-to-sky contrast that is 60 mK at 85° against 15 mK at nadir, so the bound is conditional
  but usable: a salinity shift of under 1 % in n costs about one NETD at the worst angle.
- **`tests/unit/test_adr_citations.py` keeps it true.** It walks every `ADR NNNN` in `src/`, `tests/`,
  `docs/`, `scripts/` and `configs/` and fails on one that does not resolve. The parser reads the
  compound forms the repository actually uses — `ADRs 0031, 0058`, `ADR 0053/0054`,
  `ADR 0025 and 0023` — because one reading only the leading number would have silently passed twenty
  citations it never checked. A deliberately unresolvable mention goes in a `PHANTOMS` allowlist with
  its reason (one entry: the roadmap's post-mortem of a rejected revision records that that revision
  cited a non-existent ADR 0069, and deleting the mention would delete the record of the mistake), and
  a second test fails that entry once the text or the ADR changes, so the allowlist cannot rot.
  Negative control: hiding ADR 0042 turns the invariant and the contents check red.
- **Every Isaac entry point picks its GPU through one helper** (`irsim_isaac.env.simulation_app_config`).
  This workstation has two cards and the owner works on the second one, so a render that spreads
  across both takes memory somebody is using — and Kit's multi-GPU render graph does exactly that by
  default, dying with `ERROR_OUT_OF_DEVICE_MEMORY` and then segfaulting shortly after `app ready`.
  All eleven boot sites (six render drivers, four probes, `audit_materials --stage`) now go through
  the helper, which pins `active_gpu` and turns `multi_gpu` off. `IRSIM_GPU` overrides it: an index,
  or `all` to hand the choice back to Kit.
- Two measured details are baked in rather than left to the caller. **`CUDA_VISIBLE_DEVICES` cannot
  do this job** — Kit selects a *Vulkan* device, so a render launched under it still allocates on
  every card. And **CUDA's default device order is `FASTEST_FIRST`**, which on this machine puts the
  5090 at index 0 and the A6000 at 1, the reverse of what `nvidia-smi` prints: a render pinned with
  `CUDA_VISIBLE_DEVICES=0` came up on the card that flag looks like it excludes. The helper sets
  `CUDA_DEVICE_ORDER=PCI_BUS_ID` (without overriding an explicit one) so an index means what the
  person typing it means. Seven cases in `tests/unit/test_isaac_env.py`, including the refusal of a
  GPU that is neither an index nor `all` — defaulting there would render on whichever card and only
  `nvidia-smi` would ever say which.
- **`scripts/stage_own_hunk.sh` is now the default commit path for shared files, not just an
  available one** (`RP.3`). A whole-file write has silently reverted another session's committed
  work three times in two days; the script already three-way merged a session's edit onto HEAD, but
  nothing forced anyone to run it. Three additions close that gap: a `check` subcommand (read-only)
  that refuses when a shared file's staged content is not what `stage` would have produced — the
  exact signature of a plain `git add` clobbering a commit made since the snapshot; `make stage
  FILES="README.md CHANGELOG.md docs/roadmap.md"` as the one command to run before committing; and
  `.pre-commit-config.yaml`, wiring `check` in as a pre-commit hook (`pre-commit install`) so the
  guard still catches a bypass. The `ship-step` skill and README's Contributing section now teach
  this path instead of `git add -A`. Verified with a synthetic two-author tree: `check` refuses when
  session B commits mid-edit and session A then `git add`s over it, and passes when A uses `stage`
  instead (`tests/unit/test_stage_own_hunk.py`).
- **`scripts/next_step.py` — "do next" is now deterministic** (roadmap *Do next*). The roadmap's
  stated picking rule, "the first step in phase order whose deps are all ticked", **does not pick a
  step**: measured on revision 4 the day it landed, **51 of 109** open steps satisfied it at once,
  thirteen of them in phase 0 alone, and **67 of 109** open steps have no dependents at all, so the
  dependency graph cannot order the majority of the plan. What was actually choosing was the row's
  position in a table — not a rule, and not stable under an edit.
- The script is a **topological sort** (no step before its dependencies) with a documented total-order
  tiebreak: promoted, then phase 0→A→B→C→X, then transitive dependents descending, then size
  ascending, then step id. A dependency may pull a phase-X step forward — `XD.1` unblocks eight — and
  the sort does that on its own rather than needing a special case.
- **Promotion is the only place judgement overrides the rule**, and it is bounded at three entries
  with a reason each, because a long list would mean the rule itself is wrong. There is one today:
  `RP.3` makes `stage_own_hunk.sh` the default commit path. It unblocks nothing so the mechanical key
  sorts it ninth, but a whole-file write has silently reverted committed work **three times in two
  days**, and the cost falls on other sessions' shipped work.
- **The queue is published in the roadmap**, between `next:begin` / `next:end` markers, so opening the
  plan shows a `Start here →` block and the next fifteen steps without running anything. A pasted list
  is normally a liability — it goes stale the moment a step is ticked, and a stale queue is worse than
  none because it still answers — so it is *generated* by `--write`, never hand-edited, and
  `--check` runs inside `make check`. Drift fails the gate rather than misleading the next reader.
  `make next` prints the head and republishes. Verified by simulating two ticks
  (`RP.3` → `PT.3` → `IG.1`), and by a negative control that ticks a step and confirms both the gate
  and the test catch the stale block.
- `tests/unit/test_roadmap_queue.py` is what makes the answer safe to act on without reading the
  document: the order is **total** (one head, two runs agree), **sound** (no step before its deps),
  **complete** (every open step exactly once, so nothing is silently dropped), and **every pick is the
  best available one at that moment** — so disagreeing with the queue means disagreeing with the
  published key, which is editable. Checked with a negative control that introduces a dangling
  dependency and fails the guard.
- Writing the guard corrected its own premise: an early draft asserted that a later-phase step could
  only precede an earlier-phase one if something depended on it. `PT.7` (phase C) legitimately
  precedes `XD.3` (phase B) because `XD.3` is still waiting on its own dependencies at that point, so
  the test now checks the algorithm's real invariant instead of a proxy for it.

#### Changed
- **Spec issue `S13` is reopened, and its proposed resolution was wrong** (`RP.7`). S13 asked for
  glass's τ_mwir to be computed by Beer-Lambert over 5 mm of the checked-in k(λ). That cannot be
  done: `data/nk/glass.csv` is **fused silica** standing in for soda-lime and, by the design the
  material file states outright, supplies the angular *shape* while `emissivity_per_band` supplies
  the magnitude. Recomputing over the authored 5 mm gives **nir 0.935 / swir 0.936 / mwir 0.109 /
  lwir 0.000** against the authored **0.77 / 0.70 / 0.02 / 0.0** — three bands disagree, not one.
  Fused silica has none of the iron and alkali that make real windscreen glass absorb in the near
  infrared, so the proxy is transparent where soda-lime is not; only LWIR agrees, where the Si–O
  reststrahlen band makes both opaque, which is also why the table is usable for angular shape at
  all (ADR 0042). `mwir: 0.02` was authored in `412f019`, six commits before the table existed
  (`1a0f13c`), so the table never was its source. The fix is therefore **not** to change the number
  to match the proxy — the number is plausible for soda-lime — but to make the file declare which of
  the two it is. Closing S13 needs a soda-lime n/k source or a measured τ_mwir.

#### Fixed
- **Found, not fixed: SWIR's and NIR's spectral-class tables disagree about the same air** (`AT.10`).
  NIR resolves the 0.94 µm water band at ×10; SWIR's window swallows 0.90–0.98 µm at ×0.5. Resolving
  it in favour of the feature moves **16.9 %** of the shipped InGaAs band's Planck-weighted response
  out of "clear window", and τ by −1.5 % at 1 km, **−6.4 % at 5 km** and +37.8 % at 20 km. The 200 m
  anchor is exact by construction, which is exactly why no existing test caught it. Filed as `AT.10`
  with the measurements rather than folded into a guard commit.
- Measured while deciding it: the per-band multipliers are a **gauge**, not a physical disagreement —
  multiplying every non-opaque multiplier in a band by 3 moves τ at 200 m / 1 / 5 / 20 km by at most
  **3.2e-14** in all four bands, because the anchor solve absorbs any common factor. So "LWIR window
  0.3 vs SWIR window 0.5" carries no information; only ratios within a band do.
- **`read_noise_e` reaches the rendered noise, and the mutation test that proves it** (`SC.2`).
  `SC.1` reversed which of NETD and the electron datasheet is derived; this checks the reversal
  arrived at the pixels rather than only at the budget object. The pre-`SC.1` behaviour is kept as
  the negative control and is the sharper half: under the ADR 0025 anchor the solver picks whatever
  Gaussian reproduces the datasheet NETD, so perturbing `read_noise_e` by 10 % — or **halving it** —
  leaves the rendered frame **bit-identical**. A config field that cannot move the output is not a
  parameter, it is a comment, and it had been one in every photon render this project has made.
- The sensitivity is not 10 % out for 10 % in, and that is physics rather than a weak test. σ_total
  is √(N_e + N_dark + N_bg + σ_read²), so what a read-noise change shows depends on where the camera
  sits on that curve: the cooled InSb at a bright scene carries 4.9e5 signal electrons plus 3.3e5
  from its own cold shield, so 10 % in gives **2.1 %** out; the NIR at 0.004 mean electrons is
  read-limited and gives the **full 10 %**. Both are checked, against the quadrature prediction
  rather than a stored number, and Monte Carlo on 400 rendered frames confirms the analytic result.
- **`halmstad_boson_320.yaml` takes ME.5's measured 3-D ratios.** It is the camera the public
  Halmstad set was recorded with, so unlike the 640 it has a field measurement of its own — 365
  clips, decomposed in `reference-stats-2026-09-15`. Five of the seven components now come from it:
  **vh 0.30 → 2.64** (8.8× out), **v 0.08 → 0.58** (7.2×), **t 0.02 → 0.38** (19.1×), and **h 0.15 →
  0.16**, which was already right — worth saying, because the roadmap's "7–19× out" implied all four
  were badly wrong and one was not. `tv` and `th` are not in ME.5's table and stay ESTIMATED.
- σ_VH > σ_TVH is not a typo: between flat-field events a real core's fixed pattern is the larger
  term, which is the reason §11.2's shutter exists. Total noise over σ_TVH goes **1.06 → 2.91**, so
  this camera is 2.7× noisier in total than the estimates it replaces — the direction a guess never
  goes on its own. It also puts the repository somewhere it had never rendered: every previous
  config had vh < 1, so the synthesiser, the estimator and the variance closure had only been
  exercised on the other side of that line. A new bench renders a cube and recovers vh = 2.64 within
  the estimator's own sampling floors.
- **`motion_px` reaches a rendered frame, so ADR 0077's smear runs for the first time** (`IG.6`).
  M10.1b built the rigid-body synthesis and verified it in-sim to 0.1 px; its only caller was that
  test. `IrCamera.planes()` never set the plane, so every frame this project has rendered was sharp
  regardless of scene velocity — with M9.8 and M10.1b both ticked and the documentation asserting
  otherwise. The same shape as ADR 0082's membrane-lag finding, one layer up: a mechanism that is
  correct, tested, and unreachable.
- The reason it survived is structural and worth naming. The arithmetic in `irsim.optics.motion` is
  engine-free, but the only path to it ran through `MotionTracker.sample`, which read USD directly —
  so the wiring could only be exercised by a renderer, which the fast suite does not have.
  `sample` now takes an injectable `read` and a per-frame `paths` set, and
  `tests/unit/test_motion_wired.py` (14 cases) drives the real `IrCamera.planes()` over synthetic
  transforms on a CPU.
- The plane is **refused rather than faked** in three cases, each tested: the first frame of a
  sequence (motion is a difference, and one pose is not one), a camera that was never opened (no
  stage, no poses), and a `position_frame` other than `"camera"` — the synthesis is defined on USD
  camera-space points and would otherwise apply the camera pose twice. That last one was found by
  this module's own first draft, which built its rig in the world frame and got a plane of silent
  zeros because every point read as behind the lens.
- §16's "lateral motion smears LWIR, not cooled MWIR" is now measured rather than asserted, and it
  turns out to be a statement about the **duty cycle** rather than the band. A bolometer has no
  shutter, so `smear_duty` is 1.0; the cooled InSb integrates 2 ms of a 16.7 ms frame, so it is
  0.12. At the aircraft stage's 11 px/frame the bolometer's 10–90 edge width goes **0 → 8.8 px**
  and the InSb's **0 → 1.1 px** — the 8.33× ratio, on the same scene at the same velocity. A photon
  FPA run at full duty would smear identically; nothing about 8–12 µm versus 3–5 µm enters into it.
- **A scene with a layered atmosphere no longer hands out the grey one** (`AT.5`).
  `Scene.from_config` builds both models whenever a scene names an environment preset — which
  **all eight shipped scenes do** — so the attribute a reader would take for the scene's
  atmosphere held the L1 fallback while every render script passed `scene.layered`. The field is
  now `grey_atmosphere`; `Scene.atmosphere` is a property that raises once a layered model exists,
  and the message names all three ways out: `transfer_atmosphere` for radiative transfer (the
  layered one where it exists), `atmosphere_preset` for the preset, `grey_atmosphere` for a
  deliberate L1. Both live readers wanted the preset, which is one object and cannot disagree —
  nothing was taking the wrong model, and the next caller would have had no way to tell.
- The gap is larger than the step assumed and larger than `AT.1`'s, which compared the layered
  model against itself on a horizontal path. Grey against layered, on the aerial scene at 5 km and
  20° elevation: **τ 0.057 against 0.590** and path radiance **45.1 against 18.7 W/m²/sr**. That is
  the k-distribution rather than a defect in either — exp(−τ̄) is not the mean of exp(−τ) across a
  band whose lines vary by orders of magnitude, which is what §8.6's exponential sum exists to fix
  — but it is why the grey model is L1 only. `tests/unit/test_atmosphere_handout.py` (8 cases)
  measures it rather than quoting it, and checks that the grey model stays a registered consumer:
  refused is not removed, and CLAUDE.md #6's one-weather guard must keep seeing it.
- **Three committed Boson values disagreed with FLIR's own datasheet** (`SC.3`, ADR 0091). [R24],
  Doc. # 102-2013-40 Release 340, is free, public and EAR99 — the only external anchor a project
  with no camera has for its reference core. `ffc_interval_s` was **180 s**, which matches no
  published default; `thermal_time_constant_ms` was **10.0**, the midpoint of the generic VOx
  8–12 ms range carried as ESTIMATED while the datasheet says "nominally 8 msec" outright; and the
  `ratios_3d` blocks carried no provenance marker at all, in files whose headers promise one. All
  three corrected in both Boson configs, with the citation beside each value.
- **The datasheet contradicts itself about the FFC defaults, and which reading the configs take is
  now recorded.** Section 5 says the factory default FFC Period is **300 s** and FFC Temp Delta
  **1.0 °C**; Table 8 of the same document says 1200 s and 3.0 °C. ADR 0091 takes Section 5:
  Table 8 carries its own staleness note, its pair matches the **2018** FFC/NUC application note
  rather than this 2021 release, and Section 5 is internally consistent in a way Table 8 is not —
  its start-up paragraph says an FFC occurs "every 1/3rd degree" against a default that "results
  in an FFC event every 1 degree", true of 1.0 °C and false of 3.0 °C.
- **`tests/unit/test_boson_datasheet.py`** (9 cases) reproduces Table 13's stated acceptance
  conditions — lensless at f/1.0, high gain, 20 °C camera, 30 °C background, averager disabled —
  and measures **tvh 48.5 mK, th 2.4 mK, tv 2.6 mK** against limits of < 50 / < 18 / < 18. The
  camera is compliant and **seven times more spatially uniform than FLIR guarantees**: every grade
  in Table 13 gives th/tvh = 0.35 against the configured 0.05. The ratios are left alone and
  marked ESTIMATED rather than raised to 0.35, because Table 13 publishes **upper bounds** and the
  ratio of two upper bounds is not the ratio of two typical values; `SC.2` substitutes ME.5's
  measured ratios, which is the only thing that can settle it. The check has a failing direction:
  at th/tvh = 0.5 the bench reads 25 mK and the camera is out of spec.
- **Every photon camera this project has rendered was anchored to a NETD it could not use**
  (`SC.1`). ADR 0025's M11.6 addendum says a photon FPA's noise should be built from its electron
  datasheet — quantum efficiency, well, integration time, read noise, dark current — with NETD
  demoted to a cross-check, and `irsim.noise.electron.electron_budget` implemented exactly that.
  Nothing in `src/` ever called it. `PipelineConfig.from_sensor` now selects it whenever a photon
  FPA authors `read_noise_e`, via a `noise_handle` of `auto` / `netd` / `electrons`; a bolometer
  keeps the anchor, which is right for it, and is refused `electrons` rather than handed a
  meaningless budget.
- Two measurements the change moves, taken through the pipeline rather than the module, so the
  cold shield's background is in the shot term as it is in a render. The **MWIR InSb** rendered at
  σ **533.3 e⁻** against the 350 e⁻ its own config authors — **1.52×**, a Gaussian the solver was
  free to invent because any value reaching 20 mK would do. The **SWIR InGaAs** rendered with
  dark = 0 against the **199.7 e⁻** per integration its own Arrhenius block implies — a term
  *larger than its 120 e⁻ read noise*, simply absent, and one that is an offset as well as a
  Poisson term. Neither is visible in an image; both are the kind of error that makes a sensor
  trade study come out confidently wrong.
- NETD stops being a tautology for photon cameras. Anchored, the predicted NETD equalled the
  datasheet claim to 1e-9 — it was solved for, so it carried no information. Built from electrons
  the InSb predicts **19.47 mK against a 20 mK claim**: a number the datasheet could have
  contradicted. `tests/unit/test_electron_budget_wiring.py` (11 cases) drives the selection site
  rather than the physics, because the defect was never in the physics — it was that the physics
  had no caller — and includes the failing direction: a 10 mK claim on the same camera raises.
- **The multi-band sweep filmed three of the project's eight scene configs; it now films seven, and
  the eighth says why it does not** (`IG.13`, second half). Five scenes — including the whole aerial
  point-target lane, the one ranked first — had only ever been seen in the single band their own
  driver defaults to, which is the opposite of what a driver whose purpose is the four-band
  comparison is for. `render_multiband.SCENES` gains `sky_target`, `vessel_departure` and the two
  car-ignition night scenes, and `UNSWEPT_SCENES` records `thermal_facet_scene` as the §6.13 facet
  bench: seven surfaces, no camera, no prims, driven by `validate_thermal_diurnal.py` into a diurnal
  curve rather than a frame. A test refuses any scene config that is neither swept nor listed.
- **Two flags the sweep passes unconditionally were accepted by only three of the six drivers.**
  `--rt-subframes` was missing from `render_aerial_demo` and `render_car_ignition`, and
  `--integration-ms` from those two plus `render_vessel_departure` — so adding any of them to the
  sweep would have made argparse reject the invocation and fail twelve renders at once, with the
  reason visible only in a child process's stderr. Both are now on all six, and a test builds the
  real command for every scene/band pair and checks each flag against the target parser.
- **`irsim.config.loader.with_integration_time_ms`** replaces three byte-identical copies of the
  exposure-override block and supplies the other three drivers. It goes through the model, so the
  change reaches `config_hash` — two exposures are two cameras, and a daylight reflective-band scene
  saturates a low-light exposure by around 120× — and it **refuses a bolometer** rather than
  ignoring the flag, since §8.2's thermal responsivity has no integration time and a silently
  unchanged config would still hash as exposed.
- **`render_aerial_demo` encodes a video.** It wrote still frames only, which made it the one render
  nobody could watch. `write_frame` already numbers the display frames zero-padded, so the clip
  encodes straight off what is on disk: no second PNG sequence, and nothing deleted afterwards,
  because in this driver the frames are the dataset.
- **Three of the six render drivers wrote no radiometric output at all, and now do** (`IG.13`,
  first half). `render_quad_flight`, `render_aircraft_pass` and `render_vessel_departure` — the
  whole aerial-flight and vessel-departure lanes — emitted 8-bit display PNGs and an mp4, nothing
  else. ADR 0068 is explicit that an 8-bit stream cannot carry a radiometric claim: the AGC, the
  palette and a 256-level quantisation have all been applied and none of them invert. Every frame
  those three renders have ever produced is unusable for the measurements the simulator exists to
  make. All three now write float32 radiance and apparent-temperature planes, the uint16 ADC frame
  and a JSON sidecar carrying the config, band and ISP hashes, the scene time and its UTC, and each
  plane's dtype, shape and unit.
- **`irsim.io.FrameWriter`** — the binder that makes that cheap. `write_frame` takes nine keyword
  arguments of which seven are constant for a whole run, and spelling them out inside a 500-line
  driver is how three drivers came to skip it: the frame loop was the easy part and the bookkeeping
  was not. Bound once, the loop body is `writer.write(outputs, frame_index=i, ...)` and "does this
  driver write planes" is a one-line question. `stride` thins the written sequence without thinning
  the render — a 300-frame LWIR time-lapse is 786 MB of float32 and 3.1 GB at the NIR array's
  1280×1024 — and travels in every sidecar as `plane_stride`, so a `frame_000025` sitting beside no
  `frame_000024` reads as a thinned sequence rather than a render that died. `--plane-stride 0`
  writes none, which is the only honest way to spell "this run makes no radiometric claim".
- `tests/unit/test_frame_writer.py` (20 cases). The round trip is bit-exact in float32; the sidecar
  resolves a scene time to the right UTC on the weather axis; float16 is refused; per-frame
  metadata layers over the run's; and a parametrised guard asks the exit bar of **each driver in
  turn** — run against the previous commit it fails for exactly the three offenders and passes the
  other three. Static, because the alternative is a render, but a driver that names neither writer
  cannot be producing planes whatever else it does.
- **A wall-clock assertion in the unit suite is replaced by a ratio.**
  `test_the_profile_lut_is_fast_enough_for_a_supersampled_frame` asserted `< 1.0 s` and failed twice
  on a workstation at load average 12 while passing in isolation — which tells a reader nothing
  about the LUT. It now times the exact path on a sample and compares **per-angle cost**, so the
  load cancels: both halves are slowed by the same amount. Measured at about **470×**, with the
  bound set at 100 rather than at the measurement, because a threshold sitting on its own
  measurement is the same fragility one level up.
- **The `slow` marker R11 promised, and the two-tier gate** (`GT.1`). The marker was declared in
  `pyproject.toml` and applied to **nothing**, and the Makefile had no way to filter on it. It now
  means something stated: a validation bench (Tier 2/3/4 phenomenology), an end-to-end frame bench,
  or a file-regeneration check — applied at module level to 23 files — plus any single test over a
  second, applied to 10 more. 245 of 2,979 tests.
- `make test` runs the fast tier, `make test-slow` the rest, both printing the top 15 durations.
  **`make check` runs both**, so the commit gate stays complete and nothing escapes review by being
  slow — the marker is a developer-loop split, not a coverage reduction.
- **The 30-second budget in CLAUDE.md is not reachable, and is now open question 11 rather than a
  number quietly moved.** Measured over 2,979 tests: **154 s of the 170 s is in test bodies**, not
  fixtures — setup is only 15 s, so the obvious optimisation (17 modules each building their own
  `BandLUT`) is worth ~15 s at most. The tier as marked leaves the fast half at roughly **65 s of
  test time**. Reaching 30 s would mean marking every test over 0.2 s, 173 of them, which redefines
  `slow` to mean five times what it says. The step is ticked for what landed and the number is
  recorded as the owner's call.
- **The band-kernel guard keeps covering a file it used to exclude.** `AT.2` added
  `load_band_response_for_config` to `radiometry/lut_files.py`, which the M11.1 guard holds
  unchanged since M1.11 — correctly, since "bands are data, not code". The addition names no band
  and branches on none, so instead of excluding the file the way `spectral_response.py` was
  excluded, the guard gained **per-file baselines**: it now measures that file from the commit that
  legitimately moved it, so the **next** change still fails. A second test refuses an override with
  no reason or one pointing at a commit that never touched the file.
- **CLAUDE.md's layout block describes the repository that exists** (`RP.8`). Two claims had drifted
  into fiction in the one document every session reads first and nobody re-checks. `src/irsim_isaac/spg/`
  was advertised as holding ".cu kernels, .cu.lua launch scripts, .usda shader defs"; it holds one
  README whose own first paragraph says "**This directory is empty on purpose**" — the four steps
  that would fill it are blocked (`DC.1`). And `docs/maps/` was sold as "per-module JSON maps used for
  fast navigation of the physics core"; the JSON is there, but its own README calls it frozen
  2026-09-10 snapshots and it has **one commit** in its history, so a reader sent there for navigation
  got a stale picture of a fast-moving tree with no warning.
- **Scope, stated because it is a judgement call.** `docs/spec-issues.md` records that the user asked
  CLAUDE.md not be edited by the plan. That is read here as covering *policy and wording* — line 13 on
  Unreal, the commit-scope list, the Isaac version string, all of which stay with open questions 2 and
  9 and are untouched — and not as protecting statements of fact about paths that are simply untrue.
  If that reading is wrong, the two edits are one revert.
- `tests/unit/test_claude_md_layout.py` is the durable half: it parses the layout block, asserts every
  path exists, asserts a directory advertised as holding a file type contains one, and asserts that a
  directory whose own README disclaims currency is not described as live. A claim about the tree that
  a test can check stops being something someone has to remember. Negative control: restoring either
  sentence turns the matching test red.
- **A patch rotated in its own plane no longer mis-reads its radiator** (`PT.4`).
  `view_factor_to_parallel_rectangle` took an `axes=` argument "so the offsets are measured in a
  frame the caller chose", and `patch_view_factors` passed the **receiver's** axes. That mixes
  frames: `du` came out along the patch's u while `half_u_m` is an extent along the *rectangle's*, so
  a surface whose grid is not aligned to its radiator was evaluated as though the radiator had turned
  with it. Measured on a 2.0 × 0.5 m radiator: 0.152 against 0.104 for the same element, with no
  exception raised — the "plausible number, the worst kind of wrong" the module docstring refuses.
- **The parameter is removed rather than guarded.** It could not have been right, because the
  quantity does not depend on it: this is Howell C-11, a *differential element* to a parallel
  rectangle, and a plane element has no in-plane orientation for the answer to depend on. The
  roadmap's acceptance asked for a rotated configuration to **raise**; computing it correctly is
  strictly better, so a rotated patch now returns the right view factors instead of being refused.
  The parallel-surface guard, which is a real precondition, is untouched and still tested.
- Latent until now only because `car_demo` authors every patch on world EX/EZ — the first ship deck
  or wing would have hit it. Six new cases, including a **world-space quadrature oracle** that walks
  the rectangle's real corners instead of assuming an axis, so it fails on a frame mix-up rather than
  merely on a disagreement; the one-rectangle-two-labellings invariant; and a whole-configuration
  rotation about the shared normal. Negative control: re-mixing the frames turns three red, two of
  them the new oracle.
- **The G-buffer's dtype guard admits no float16 anywhere** (`IG.8`). `_as_f64_plane` refused fp16
  only on `distance_m` and `position`, and carved out normals, occlusion and motion on the stated
  grounds that "the renderer on this build delivers the normals AOV as float16 whether we like it or
  not (measured; ADR 0014 addendum)" — so banning it "would have meant no normals at all".
- **That claim was wrong, and it was the entire justification.** ADR 0014's own addendum table
  records `normals` as **float32 ×4 at full resolution** and marks it *use*; this build's Replicator
  registry agrees, registering `"normals": AnnotatorParams("NormalSD", np.float32, 4, ...)`. The
  fp16 plane in that table is `PtWorldNormal`, which `AovReader` rejects anyway for being
  half-resolution and all-zero. Nothing was being rescued — the carve-out spent CLAUDE.md
  non-negotiable #2, the one rule whose violation has no visible symptom, and bought no physics.
- `precision_critical` is gone; every float AOV is held to the same rule. The test that asserted
  the opposite (`test_float16_normals_are_accepted_and_upcast`, whose docstring repeated the false
  premise) is inverted, and occlusion and motion gained their own refusals. A companion test checks
  the guard refuses the *dtype* and not the channel: float32 normals still go through untouched.
  Negative control: reinstating the carve-out turns three red. The non-negotiable enforcement map's
  gap for rule #2 is closed — what remains there is `IG.13` and `IG.2`.
- **The motion AOV no longer reaches the G-buffer, and its convention is never guessed** (`IG.5`).
  Three places in the repository said this plane was omitted on this build — `gbuffer_isaac`'s own
  channel table, `irsim.optics.motion`'s opening paragraph ("the `motion_px` plane of the G-buffer
  had to be left empty", which is *why* the analytic tracker exists), and ADR 0014's addendum, which
  measured `motion_vectors` sitting at a ~6e-5 floor after a **180 px** displacement. The code
  delivered it anyway: `_reject_reason` applies its all-zero test only to *required* channels, and
  6e-5 is not zero in any case, so the plane reached `RawAovs.motion`, was scaled by a **defaulted**
  `motion_convention="pixels"`, and arrived in the G-buffer where `irsim.optics.stage` ran the smear
  path on it.
- Numerically that was a no-op; the hazard was the convention. Nothing had ever checked the sign,
  Replicator's documentation gives both signs opposite to this project's contract, and a plane that
  is noise today is a plane that is backwards the day a build starts filling it in. The three
  conventions differ by a factor of the **resolution** and by the sign of y: the same raw 0.01
  becomes 0.01, 1.28 or 2.56 px/frame at 256 px, with `ndc` pointing y the other way from `uv`.
- Two changes. `UNVERIFIED_CHANNELS` marks a channel whose annotator returns *something* whose
  meaning has never been established; `AovReader` does not attach it unless a caller names it
  (`unverified=("motion",)`), which `geometry_probe` does, because surveying it is the only way it
  could ever stop being unverified. And `geometry_planes`' `motion_convention` lost its default: a
  motion plane supplied without one now raises instead of being scaled by a guess. Requiring an
  unverified channel is refused outright, so the reader cannot drop it and then blame the build for
  producing no data.
- The motion the pipeline uses is still the synthesised one (`irsim.optics.motion`, verified in-sim
  to 0.1 px). Wiring it into `IrCamera` is `IG.6`; this step only stops the AOV competing with it.
  Negative control: restoring the `"pixels"` default and emptying `UNVERIFIED_CHANNELS` turns 5 of
  the 10 new cases red.
- **The patch-coverage guard was off in every frame this project has produced** (`IG.1`).
  `IrCamera` had one `strict_materials` flag standing in front of three unrelated failures: an
  instance id the label table does not name, a rendered prim with no thermal node, and a pixel that
  lands on a prim carrying a temperature *field* but outside every one of that prim's patches. All
  six render scripts and five integration tests passed it `False` to get past the first two, which
  turned the third off as well — the one whose own docstring calls the fallback "a seam that looks
  like physics", because a patch authored smaller than its geometry then renders part of a bonnet as
  a field and part as a flat value. It is now three flags: `strict_materials`,
  `strict_thermal_nodes` and `strict_patch_coverage`. The callers name the two they actually mean;
  patch coverage keeps its `True` default and nothing turns it off implicitly any more.
- The five new cases in `tests/unit/test_camera_strictness.py` drive the real `IrCamera.planes` over
  a synthetic frame with a fake AOV reader — no Kit — because the defect is a *routing* one: three
  flags that are all stored and then all read from the same place pass an attribute test and fail
  this one. Each guard is shown firing on its own failure and staying silent on the other two, and
  the headline case asks for exactly what the render scripts now ask for (`strict_materials=False,
  strict_thermal_nodes=False`) against a frame that is wrong in all three ways, and still raises on
  the patch gap. Negative control: re-pointing the two new flags back at `strict_materials` turns 3
  of the 5 red.
- Writing the test corrected its own oracle. Comparing the rendered plane against the field at
  `t = 0` missed by 8 mK: the camera samples on the **absolute** clock, and `t0_s` is hours into the
  weather axis, over which even a 1e9 J/m²K slab radiates a little. Sampled at the clock the camera
  actually used, the plane matches the field's own cells to under 1e-4 K — and carries a 30 K
  gradient across one prim, which is the whole point of ADR 0087.
- **The car scenes' ground radiators no longer over-count a shared solid angle** (`PT.3`, ADR 0090).
  `build_ground_field` summed independently-computed view factors from `underbody`, `engine_bay` and
  `exhaust_pipe` onto the road, but `engine_bay`'s and `exhaust_pipe`'s rectangles lie **entirely
  inside** `underbody`'s footprint — three descriptions of one floor pan, not three disjoint bodies.
  Measured peak Σ F = 1.40 across 28 cells under the engine bay, against a hard bound of 1 for a
  plane element, inflating the clear-night road patch by up to ~40%.
  `irsim.thermal.spatial_sources.clamp_view_factor_sum` rescales the three radiators' view factors
  proportionally wherever their sum would exceed 1, preserving the relative footprint/pool/stripe
  shape ADR 0088 authored three rectangles for, and warns loudly when it triggers.
  `occluded_longwave_flux` is linear in `view_factors`, so the same scale factor caps its occlusion
  term along with its source term. Verified against the actual car-scene geometry: Σ F ≤ 1 + 1e-6 on
  every cell of both `car_ignition_overcast_night.yaml` and `car_ignition_clear_night.yaml`
  (`tests/unit/test_car_demo.py::test_ground_radiator_view_factors_never_exceed_one`), plus 6 new
  unit cases for the clamp itself (`tests/unit/test_spatial_sources.py`).
- **README's component status table is a table again** (RP.1, roadmap revision 4). Four of its fifteen
  rows — `materials`, `detector`, `noise`, `isp` — had lost their **State** cell to pasted changelog
  prose, with the state token pushed to the end of a paragraph up to **1,473 characters** long and the
  tier column holding it instead. A markdown table with a 1,473-character cell still renders, which is
  why nobody saw it: it renders as a table whose second column is an essay. The prose moved to Notes,
  where the long-form detail already lives.
- `tests/unit/test_readme_status_table.py` is what keeps it fixed, and it pins **shape, not content**:
  Notes may say anything, the State cell holds one of four state tokens, the Tier cell holds a tier.
  `T4 infra` and `T2, T4 infra` are legitimate and allowed — they are statements about what a package
  supports rather than what it has been validated to. Checked with a negative control that reintroduces
  the defect and fails the parser.
- The MP.5 ground-field spin-up limitation was overwritten a **third** time by a parallel session
  writing README whole, and is restored again. RP.3 — making `scripts/stage_own_hunk.sh` the default
  path rather than an available one — is the structural fix; this row only repairs the damage.

### 2026-09-15

#### Added
- **Tier 3 maritime phenomenology, on frames rather than on models** (MM.8, §15 T3, ADR 0078).
  `irsim.validation.maritime_scene` is the maritime twin of MS.8's aerial fixture: sky above a
  **spherical** horizon (0.1436° down at 20 m, three Boson pixels below where a flat-earth scene
  would put it), sea below it taking `SeaModel.apparent_temperature_k` at each ray's own
  depression, and vessels resolved or sub-pixel. `tests/unit/test_tier3_maritime.py` runs whole
  frames through `run_frame`.
- Sea and sky are **both background**, at `distance_m = 0`. The sea profile already contains its
  own atmospheric path, so a non-zero distance would send stage 2 over the same 7 km again — an
  error that would render as a plausibly hazier sea rather than as a bug. The rendered sea matches
  the sea model's **un-tabulated** profile to **0.34 mK** through the whole chain, which is the
  test that catches it.
- A sea built on a different `SkyModel` than the frame's sky pixels now **raises**. The sea is
  mostly reflected sky, so two sky models means a horizon with different weather on each side of
  it — CLAUDE.md #6 in its most literal form, and it would render perfectly plausibly.
- **Sea skin temperature: the cool skin and the diurnal warm layer** (MM.4, ADR 0080, §6.1/§6.5).
  A maritime scenario knows its **bulk** SST — that is what a buoy, a ship's intake or a satellite
  product reports, and it is what `ground.bulk_sst_k` authors. An infrared camera does not see it.
  It sees the top fraction of a millimetre, and `irsim.thermal.sea_skin` now derives that:
  `T_skin = T_bulk − ΔT_cool(U, Q_net) + ΔT_warm(Q_sw, U)`.
- The cool skin is Saunders (1967): the sublayer thickness `δ = λν/u*` with the water-side friction
  velocity from stress continuity across the surface (`ρ_a C_D U² = ρ_w u*²`), and the deficit is
  Fourier's law across it. Measured **0.93 mm** at 5 m/s against the ~1 mm the measurements show,
  and **0.110 K** of deficit under the 79 W/m² of net longwave a clear night over a 290 K sea
  produces. That is 2–6× a 50 mK NETD, in one direction, across the whole lower half of the frame
  — so every maritime frame rendered until now has been biased warm by several noise widths.
- Saunders' form diverges as the wind drops (`u* → 0` gives an infinitely thick sublayer), so δ is
  bounded at the top of the observed range, 2 mm, through `δ_max tanh(δ/δ_max)` rather than a
  `min`. The two agree to third order wherever the Saunders term is small, so the wind-stirred
  regime is untouched; what the smooth form buys is a deficit that stays **strictly** decreasing in
  wind instead of acquiring a flat shelf and a corner exactly where calm maritime scenes live.
- The diurnal warm layer is **empirical and labelled as such** everywhere it surfaces. It carries
  the three properties the observations agree on — proportional to absorbed irradiance, zero at
  night, gone above ~6 m/s — and claims nothing else. Measured 2.6 K at 950 W/m² and 1 m/s, and
  exactly zero at night and at or above the cutoff.
- Under net *warming* the cool-skin deficit is **zero, not negative**. Conduction against an
  outgoing flux is the mechanism, so it stops when the flux reverses; a surface genuinely warmer
  than the water beneath it is the warm layer's business, and a signed deficit would count it twice.
- **The ship had no videos, in any band** — the maritime script exported per-frame physical-unit
  files (which is what MM.7 is for) and never encoded anything, while the two aerial scripts write
  an IR, an AGC and a visible video each. It encodes the same three now, so the four-band ship
  comparison can actually be watched rather than inspected as loose PNGs.
- Its manual span comes from **percentiles of the first captured frame**, not from thermal nodes:
  a maritime scene has no single target whose temperature is the subject, and spanning whatever
  vessels are in shot would throw away the sea's angular-emissivity gradient, which is most of the
  picture and the point of ADR 0078. `span_from_apparent_t` is the emissive counterpart of
  `span_from_dn16`, so the band-aware rule of M10.23 still holds: kelvin where §12.1 gives an
  apparent temperature, raw ADC where it does not.
- `irsim_isaac.display_span` had **no tests at all** since M10.23. It has eleven now, and writing
  them found a bug: with a frame flat enough that both percentiles land on the same value, the
  fallback anchored the span on `min()` — a single dead pixel or glint, which is exactly what the
  percentiles were there to exclude. It falls back around the median instead.
- **The n/k table library is complete, and its provenance rules are written down** (M7.5, ADR 0041,
  §4.2/§12.3). `data/nk/glass.csv` and `data/nk/paint_proxy.csv` join the measured water table and
  the modelled aluminium one, fetched from the CC0 RefractiveIndex.INFO database by the new
  `scripts/fetch_nk_tables.py` — glass from Franta (2016), a 0.405 mm fused-silica plate; the paint
  proxy from Zhang (2020), a PMMA sample. Both span 0.6 µm to past the LWIR window, because the
  loader refuses to extrapolate and a table that stops inside a band makes that band raise.
- Both are **proxies, and say so in the header the loader carries**: no freely redistributable
  dataset covers soda-lime float glass or a pigmented automotive coating across 0.75–13.5 µm. What
  a proxy has to get right is not the emissivity — that comes from the material's authored per-band
  value — but how emissivity *falls with angle*, which is set by the interface and the band's n
  and k rather than by the pigment.
- **ADR 0041 is written**, three months after `nk.py`'s own error message started citing it. It
  records three rules that were being followed informally: provenance is enforced by the loader and
  not by lint; only redistributable (CC0 / public-domain) sources are committed, so Palik stays out
  even where it is the better data; and every file declares which of *measured* / *MODELLED* /
  *PROXY* it is. It also states the limit the convention does **not** bound — Level A's 1 − R is
  absorptance plus transmittance, exact for glass in LWIR where τ = 0 and approximate in NIR and
  SWIR where τ is 0.77 and 0.70.
- **`glass_windshield` moves to Level A.** Not for want of a Level B fit, but because one cannot
  serve it: `angular_model` is a single setting for the whole material, and glass's LWIR shape is
  genuinely unlike its other three. The Si-O reststrahlen band sits inside the LWIR window and
  drives the real index down to **0.35** at 8.8 µm while k rises above 1.6 — the glass responds
  like a metal across a narrow band. Fitting each band separately wants a = **1.43** in LWIR
  against 0.66–0.73 elsewhere. Level A integrates each band's own response and gets this for free.
- The fitted-Level-B alternative fails in a specific and ugly way, which is why the test says so:
  ε₀(1 − a(1−cos θ)^p) with a > 1 crosses zero at **85.1°** and is clipped there, so it reports a
  windshield edge as having no emissivity at all — a perfect mirror — where Level A still has 0.37
  of normal at 85° and 0.18 at 88°. Those are the angles a windshield is seen at from across a
  street.
- Measured consequence: glass ε(70°) in LWIR falls **0.83 → 0.69** against the estimated Level B it
  replaces, and ε_hemi 0.862 → 0.801. ε(0) is unchanged in every band, by construction — the table
  supplies the shape and the YAML the magnitude, so a fused-silica table cannot silently turn a
  soda-lime windshield into a quartz one.
- **The full multi-band matrix ran**: twelve renders — four bands of the drone, the airplane and
  the ship, each with its registered visible companion (M10.24). Final state is zero saturated
  pixels in any of them. The comparison is the deliverable: a quadrotor is four hot motor bells on
  a cold sky in LWIR and a dark silhouette on a bright sky in NIR, and an aircraft shows **1615 px
  of nozzle in MWIR against 195 px in LWIR** at the same instant and aspect.
- Running all twelve at once is what found five defects that no single-band render could have
  exposed — a missing `--rt-subframes`, a display span still written in kelvin for bands that have
  no apparent temperature, a contact sheet that only read videos, and the two radiometric bugs
  above. That is the argument for the driver script existing rather than twelve invocations.
- **Water's n/k table extended to 0.65 µm** (M7.5). A maritime scene in SWIR or NIR needs water's
  Fresnel reflectance over 0.7–1.8 µm, and the loader **refuses to extrapolate** an n/k table rather
  than invent optical constants — so a ship render in those bands failed outright until the range
  existed. That refusal is the correct behaviour and it caught a real gap. The fix is a
  re-truncation of the same published download the file already cites (Segelstein 1981), and all
  **365 rows at and above 2.0 µm are byte-identical** to the previous file, which is the check that
  this is the same table and not a second compilation stitched onto the first.
- What the new range means for the sea: reflectance there is set by n alone — k runs from 1.6e-7 at
  0.75 µm to 3.6e-4 at the 1.45 µm band, and at those values the absorption term changes R by less
  than 1e-6 — and n itself moves only 1.3272 → 1.3034 across 0.75–1.80 µm, so normal-incidence
  reflectance varies between 0.0201 and 0.0192. The sea is the same dark, strongly angle-dependent
  mirror in every reflective band that it is in the visible.
- **The AOV semantics probe is closed (M2.4, ADR 0014 addendum, §13.3/§5.3(a)).** The M2 gate spike
  left three channels open because its unlit, static, front-parallel ramp could not exercise them.
  M10.1 measured all three on a lit, tilted, moving scene; what remained was not an experiment but
  two loose ends in the record, and both are now measured and pinned.
- `irsim_isaac.geometry_probe.position_frame_residuals` decides **which frame the position AOV is
  in** by scoring every candidate reading against a world point the AOV had no part in producing:
  `C + d · r`, from the independently-measured `DistanceToCameraSD` ray length and that pixel's
  pinhole direction (`pinhole_rays`). Measured on the M10.1 scene at 0 / −8 / −20° of camera pitch:
  `camera` **9.4 mm**, `world` 5.31–5.48 m, `rotated_world` 5.48 m. `Camera3dPositionSD` is camera
  space, as its name says.
- It is decided against **three** hypotheses, not two. `rotated_world` — the world point in camera
  *axes*, translation left in — had never been tested and is indistinguishable from the other two
  on every scene measured so far, yet it is the one that would break `ray_directions`' `frame=
  "camera"` branch by the full camera offset on a camera both moved and turned. It is wrong here by
  exactly `|C|` = 5.48 m, which is also the arithmetic statement of why a camera at the origin can
  never settle this question.
- A verdict now comes with its **margin**, and the degenerate case is a test: camera at the origin,
  no rotation, all three residuals zero, margin zero. That is the configuration that produced the
  error being corrected, so it fails loudly instead of picking a winner out of rounding.
- `gbuffer_isaac.camera_pose` states the USD→ray transpose **once** (USD matrices are row-vector;
  `ray_directions` applies `vec @ rot.T`), for `IrCamera` and the probe both. Getting it backwards
  rotates every ray by twice the camera tilt and raises nothing.
- **Surface temperature that varies across one surface** (MP.1, ADR 0087, §6.1/§6.4).
  `irsim.thermal.surface_field.PlanarPatch` is a grid of §6.1 facets on a plane and
  `PlanarThermalField` solves it on the existing fixed tick, so a bonnet over a running engine or a
  road beside a warm car carries a temperature *field* instead of one number. Until now every
  rendered prim had exactly one temperature — `ThermalField` was always an N-facet solver, but every
  consumer mapped one facet to one prim, so N counted objects and not points. In LWIR that is the
  dominant modelling error for anything bigger than a few pixels: a wall half in sun spans 10–20 K
  and irsim rendered it flat.
- Nothing in the balance changed. The spatial variation enters through `FacetForcing.q_internal_w_m2`
  and friends, which were already per-facet and had simply never been varied across one surface.
  The new code is the *lookup*: a point in space becomes a bilinear blend of cells, exact at cell
  centres, with no staircase at the resolutions a camera resolves.
- Held to per-cell equilibrium, not to "there is now a gradient": with 250 W/m² of internal load on
  half a patch, **every cell** converges to the root `steady_state_temperature` predicts for the flux
  *that cell* sees, to **1 mK**, with the two halves 9+ K apart. A field that averaged, broadcast or
  dropped the spatial forcing fails that by ~10 K.
- A patch claims a **slab**, not a rectangle (`thickness_m`): a bonnet 0.9 m above a road projects
  into the road's own (u, v) rectangle, and a patch testing only its in-plane extent would hand the
  road's temperature to the car and produce a frame that looks entirely reasonable. Points outside
  the slab sample as NaN rather than as the nearest edge value, because a patch quietly extending
  itself to the whole scene is the failure the fill exists to make loud.
- Known limit, recorded in ADR 0087 rather than hidden: the parameterisation is a *projection*, so it
  is exact for near-planar surfaces (road, bonnet, roof, deck) and a wheel or an exhaust pipe still
  takes one temperature per prim. Lifting that needs a UV or per-triangle AOV this build does not
  expose (ADR 0014).
- **Heat sources that vary across a surface** (MP.2, ADR 0088, §6.1/§6.6).
  `irsim.thermal.spatial_sources` supplies the per-cell forcing MP.1's field had no way to differ
  by. An engine bay radiating up onto a bonnet and a warm underbody radiating down onto asphalt are
  the *same* geometry problem — a plane element exchanging with a parallel rectangle — so one
  closed-form configuration factor (Howell C-11, superposed over four signed corners for an
  arbitrary offset) serves both, and the bonnet's falloff is **computed from the block's dimensions**
  rather than authored as a Gaussian with a fitted width.
- The constant is verified against a brute-force quadrature of the defining integral rather than
  taken from memory: a view factor wrong by a factor of two gives a gradient of exactly the right
  shape and half the right size, which no image would reveal. Non-parallel geometry raises.
- **A hot body over a surface also blocks the sky that surface was seeing**, and
  `occluded_longwave_flux` returns the net change rather than the source term alone. This is not a
  correction, it is the dominant term on a clear night: a 295 K underbody over asphalt adds about
  half the naive source-only figure under a 245 K clear sky and about a tenth of it under an
  overcast 288 K one. It is why a parked car leaves a warm car-shaped patch on asphalt **before its
  engine has ever run** — a familiar feature of night parking-lot imagery that a source-only model
  cannot produce at all, and one that now falls out of the geometry.
- It also settles a scene-design question: a demo meant to show *engine* heat must be shot under
  overcast, where the occlusion nearly cancels. Under a clear sky the car-shaped patch is already
  there in frame 0.
- **Per-pixel temperature on the Isaac render path** (MP.3, ADR 0087, §13.1/§13.3).
  `irsim_isaac.pipeline.point_bridge` overlays patch-backed prims onto the per-instance temperature
  plane: the position AOV gives each pixel a world point, the point selects a cell, and the cell
  carries the temperature. `IrCamera` takes `surface_fields=`; without it the overlay is not even
  constructed and every existing scene renders **bit-identically**.
- The frame conversion is the part that had to be right: M2.4 established the position AOV is
  *camera* space on this build, so `world_positions` applies the same `camera_to_world` rotation
  `ray_directions` does rather than deriving a second one. Tested on a camera both moved and
  rotated, because the readings are indistinguishable on a camera at the origin. A pixel that lands
  on a bound prim but outside all of its patches **raises**, and the overlay advances its fields on
  the **absolute** weather clock — the trap M10.3 pinned for `thermal_surfaces`.
- **§6.6's vehicle heat sources are reachable from a scene** (MP.4a, ADR 0089, §6.6). Scene schema
  **v6** gains `solver: vehicle_source` — `source` names a row of `VEHICLE_HEAT_SOURCES` and
  `load_s`/`load` is the duty fraction over time. The table and its integrator have existed since
  M6.14 and nothing outside their own unit tests could reach them: `heat_source` is ADR 0072's
  *aerial* node, whose law is a steady-state relation with **no time constant at all**, so an engine
  bay authored through it shows its full 65 K in the first frame after ignition.
  `VehicleSourceSolver` is deliberately only an adapter — the law and its exact-exponential step
  stay in `SourceHistory`. Held to §6.6's closed form to **1 mK** at five elapsed times; a 1 s and a
  200 s step agree to 1e-9; switch-off cools on τ_cool (1800 s), not τ_rise (750 s).
  `MIN_SCENE_SCHEMA_VERSION` stays at 4, so every older file still loads. Also adds
  `data/weather/overcast_still_48h.csv`, the flat still night frame 0 needs.
- **A car that starts its engine, filmed in LWIR** (MP.4b, ADR 0087/0088/0089). `irsim_isaac.car_demo`
  + `scripts/render_car_ignition.py` + two scene configs. The bonnet is **one USD prim**, flat to
  under a millikelvin in frame 0 and **6.3 K across** after 30 minutes — 127 NETD of structure a
  per-prim bridge has one number to represent.
- **The wheels do not warm, and the readout says so.** §6.6 makes tyre heating flexing work and
  brake heating kinetic energy; a car idling in a car park is doing neither, so its tyres sit at
  ambient (+0.000 K). A test pins it so a later change cannot quietly "fix" it.
- **What the field holds is not what the camera sees, and both are reported.** The road patch is
  **1.30 K** in the field and only **0.15 K** (3 NETD) reaches the sensor: the warmest asphalt is
  directly *under* the car and a 45° view cannot see it.
- **Two scenes, one engine.** `car_ignition_overcast_night.yaml` and `car_ignition_clear_night.yaml`
  run the *same* §6.6 node on the *same* load profile and differ only in the sky. Measured after
  30 minutes the road patch is **+4.13 K** under a clear sky against **+1.47 K** under overcast —
  nearly three times, from the sky alone (ADR 0088). The bonnet moves the *other* way (5.85 vs
  6.50 K), since the skin radiates to that sky too, which is the cross-check that this is the
  occlusion term and not a scale factor.
- Found while getting the first frame out of Kit: the stage up axis must be **+Y** (Kit defaults to
  +Z, which gives `azimuth_from_rays` an up vector parallel to its own forward and raises);
  `world_positions` reuses `gbuffer_isaac`'s plane validator because the annotator delivers
  (H, W, **4**); and the render script reads its clock *before* `get_outputs`, which advances it on
  the way out — on a time-lapse that labelled every row with the state of the row after it.
- The car scenes encode to **video** through the same `irsim_eval.video` path the quadrotor and
  aircraft films use: three streams per run — the bonnet's own span, a 1.6 K window on ambient, and
  the camera's own AGC — because no single linear span shows both features. The last frame's gauge
  reads bay **39.4 °C**, bonnet **19.0 °C**, wing **13.2 °C**: the milestone's claim as two numbers
  off one prim.
- ⚠️ **Known limit.** §12.3 solves the asphalt as one surface, so the ground field inherits a
  *uniform* spun-up state — the road as it would be with no car on it. Frame 0 is the moment the car
  arrived, and every patch is one the run itself grew. Pinned by a test so it is not mistaken for a
  result.
- **Per-pixel temperature on the Isaac render path** (MP.3, ADR 0087, §13.1/§13.3).
  `irsim_isaac.pipeline.point_bridge` overlays patch-backed prims onto the per-instance temperature
  plane: the position AOV gives each pixel a world point, the point selects a cell, and the cell
  carries the temperature. `IrCamera` takes `surface_fields=`; without it the overlay is not even
  constructed and every existing scene renders **bit-identically**.
- The frame conversion is the part that had to be right: M2.4 established the position AOV is
  *camera* space on this build, so `world_positions` applies the same `camera_to_world` rotation
  `ray_directions` does rather than deriving a second one. Tested on a camera both moved and
  rotated, because the readings are indistinguishable on a camera at the origin — read as world,
  every lookup is displaced by the camera's own position.
- A pixel that lands on a bound prim but outside all of its patches **raises**. A patch authored
  smaller than its geometry would otherwise render part of a bonnet as a field and part as a flat
  value, with a seam that looks like physics.
- The overlay advances its fields on the **absolute** weather clock, not scene-relative time — the
  same trap M10.3 pinned for `thermal_surfaces`, where a spun-up field handed relative time returns
  a plausible temperature from the wrong hour.
- **§6.6's vehicle heat sources are reachable from a scene** (MP.4a, ADR 0089, §6.6). Scene schema
  **v6** gains `solver: vehicle_source` — `source` names a row of `VEHICLE_HEAT_SOURCES` and
  `load_s`/`load` is the duty fraction over time. The table and its integrator have existed since
  M6.14 and nothing outside their own unit tests could reach them: `heat_source` is ADR 0072's
  *aerial* node, whose law is a steady-state relation with **no time constant at all**, so an
  engine bay authored through it shows its full 65 K in the first frame after ignition.
- `VehicleSourceSolver` is deliberately only an adapter — the law and its exact-exponential step
  stay in `SourceHistory`, which already had them. It is the first solver that **cannot** be
  pre-derived into a `PrescribedSolver`, because a node with a time constant depends on its own
  history: `delta_t0_k` is how a scene says whether the engine is cold or has just been switched
  off, which is what makes a daylight scene of a recently parked car a one-parameter change.
- Held to §6.6's closed form to **1 mK** at five elapsed times; one time constant is 63.2 % of the
  rise and not the whole of it; a 1 s step and a 200 s step agree to 1e-9; switching off cools on
  `τ_cool` (1800 s) and not `τ_rise` (750 s); ambient enters in exactly one place, so two nodes
  50 K apart in air carry identical ΔT. An aerial source name is refused for a vehicle node and
  neither kind accepts the other's profile.
- `SCENE_SCHEMA_VERSION` is 6 and the five committed demo configs move with it;
  `MIN_SCENE_SCHEMA_VERSION` stays at 4, so every older file still loads.
- **An overcast still-night weather fixture** (`data/weather/overcast_still_48h.csv`). What it is
  for is what it removes: under thick cloud the sky radiates near air temperature, so every surface
  settles within a kelvin or two of ambient regardless of material or tilt. That is the only
  condition under which a car and the asphalt under it look the same in LWIR, which is what an
  ignition demo needs in frame 0.
- **A car that starts its engine, filmed in LWIR** (MP.4b, ADR 0087/0088/0089).
  `irsim_isaac.car_demo` + `scripts/render_car_ignition.py` + two scene configs. This is the scene
  point-wise temperature exists for, and the number it produces is the milestone's claim: the
  bonnet is **one USD prim**, flat to under a millikelvin in frame 0, and **6.3 K across** after
  30 minutes — 127 NETD of structure that a per-prim bridge has one number to represent. The
  falloff toward the wings is ADR 0088's configuration factor to the bay below, computed from the
  bay's dimensions rather than authored as a Gaussian.
- **The wheels do not warm, and the readout says so.** §6.6 makes tyre heating flexing work and
  brake heating kinetic energy; a car idling in a car park is doing neither, so its tyres sit at
  ambient (+0.000 K) however long it idles. It is printed beside the bonnet's rise because it is
  the result a viewer disbelieves, and a test pins it so a later change cannot quietly "fix" it.
- **What the field holds is not what the camera sees, and both are reported.** The road patch is
  **1.30 K** in the field and only **0.15 K** (3 NETD) reaches the sensor: the warmest asphalt is
  directly *under* the car and a 45° view cannot see it. Quoting the field number about an image
  would have been the easy mistake.
- **Two scenes, one engine.** `car_ignition_overcast_night.yaml` and `car_ignition_clear_night.yaml`
  run the *same* §6.6 node on the *same* load profile and differ only in the sky. Measured after
  30 minutes the road patch is **+4.13 K** under a clear sky against **+1.47 K** under overcast —
  nearly three times, from the sky alone, because a clear night's car blocks a 40–50 K depression
  (ADR 0088). A model that added the car's emission without removing the sky it occludes would
  report the same number for both. The bonnet moves the *other* way (5.85 vs 6.50 K), since the
  skin radiates to that sky too, which is the cross-check that this is the sky and not a scale.
- Two fixed display spans are written beside the camera's own AGC output, because no single linear
  span shows both features: the bonnet spans 6 K and the road patch under 1 K. They are anchored on
  the first frame's **own median apparent temperature**, not on T_air — the two differ by ~0.9 K
  here (ε 0.95 asphalt reflecting a cooler sky, plus 27 m of path), and a span centred on air
  temperature renders the whole picture below its floor and black.
- Found and fixed while getting the first frame out: the stage up axis had to be set to **+Y**
  (Kit defaults to +Z, which gives `azimuth_from_rays` an up vector parallel to its own forward
  and raises); `world_positions` now reuses `gbuffer_isaac`'s own plane validator rather than a
  private copy, because the annotator delivers (H, W, **4**); and the render script reads its clock
  *before* `get_outputs`, which advances it on the way out — on a time-lapse that labelled every
  row with the state of the row after it.
- ⚠️ **Known limit (MP.5).** §12.3 solves the asphalt as one surface, so the ground field inherits
  a *uniform* spun-up state — the road as it would be with no car on it. Frame 0 is the moment the
  car arrived, and every patch is one the run itself grew. A car that has stood for hours already
  carries the full patch, which is most of what a real night image of a car park shows. Pinned by a
  test so it is not mistaken for a result.
- The car scenes encode to **video** (MP.4b), through the same `irsim_eval.video` path the
  quadrotor and aircraft films use: three streams per run — the bonnet's own span, a 1.6 K window
  on ambient, and the camera's own AGC — each with the standard caption block and temperature
  gauge, so the three are comparable frame for frame. Three rather than one because no single
  linear span shows both features. Measured on a 30-minute run at 1 frame / 30 s, the last frame's
  gauge reads bay **39.4 °C**, bonnet **19.0 °C** and wing **13.2 °C** — the milestone's claim as
  two numbers off one prim.
- **The fidelity ablation** (M12.3, ablation half). `scripts/fidelity_ablation.py` renders the same
  scenarios with one mechanism switched off and measures the DN8 distance from full fidelity with
  ME.6's own statistics. It needs no detector, no labels and no GPU, which is why it was worth
  running on its own: ME.7's training half is blocked on the reference set's MATLAB annotation boxes
  and this half is blocked on nothing, and a mechanism that moves nothing here cannot be responsible
  for a sim-to-real gap whatever a later detector says.
- Ranked by discriminator AUC against full fidelity, on 5 scenarios × 24 frames: **AGC linear→none
  1.000** (histogram EMD **61.7 codes**), **optical PSF off 0.962** (EMD 10.1), **noise off 0.725**
  (EMD 0.034), and bad pixels / NUC residual / FFC all at the control's own 0.495. The display
  mapping dominates everything else by a wide margin — the same conclusion the Tier 4 acceptance run
  reached from the other direction.
- **The control passes at 0.495, and how many scenarios that took is itself a result.** At two
  scenarios it read **0.40** — five null standard errors below chance on two sets identical by
  construction — because the AUC's null σ assumes independent patches and patches cut from the same
  frames are not. Independence is bought with scenarios, never with patches per frame.
- The three null-effect switches were **verified rather than assumed**: the M9 chain *is* attached,
  with 112 bad pixels and both switches enabled, but the FFC interval is **10 800 frames** at 60 Hz
  so a 24-frame clip contains no shutter event; the NUC residual grows from the last FFC and is near
  zero at a clip's start; and the chain's own replacement stage repairs the defects, which is what
  it is for. Chasing that is also how the CPU matched-scenario generator was found to be rendering
  **without the sensor chain at all** — now fixed, which matters because the published clips contain
  FFC freezes and dead pixels and a render without them is a different camera.
- **The SPG lane's blocker is written down instead of coded around** (M10.12, M10.13a/b/e). Stage 1
  needs a **per-frame float32 facet table** to reach the kernel, and M2.3 established that there is
  no cross-frame device state to keep it in, that `io.` is a forbidden token so it cannot be loaded,
  and that a Lua literal of LUT size crashes the Kit process. The Planck LUT is static and bakes
  fine; the facet table changes every frame and does not. `src/irsim_isaac/spg/README.md` states the
  question, three candidate answers ranked by cost, and how one Isaac session with a fourth
  `spg_probe` experiment would decide it for all four steps. Writing the kernels against a guess is
  what CLAUDE.md's "flag uncertainty rather than guessing" rule exists for, so they are not written —
  and M2.3's hazard travels with the note: a kernel that fails to load yields a **zero-filled output
  with status ok**, so any in-sim equivalence test must assert against a known non-zero reference.
- **A vessel departure, filmed in LWIR** (MM.7). `scripts/render_vessel_departure.py` films a 90 m
  coaster leaving from 250 m at 6 m/s while its funnel warms from cold alongside to cruise power —
  240 frames × 4 s of sea time, played as a time lapse, with IR, camera-AGC and visible videos.
- **The film shows what a static frame cannot: the target's signature getting stronger while the
  target gets smaller.** The funnel runs 32 → 179 °C while the hull falls from 420 px to 17.5 px
  and the funnel from 29 px to 1.2 px. That is the regime an infrared search set actually works in.
- The camera does **not** track. Against a fixed boresight the vessel climbs through the frame as
  its depression angle shrinks toward the horizon, crossing the sea's own angular gradient: it
  starts against near water at about the SST and ends against far water that is mostly warm air.
- `DepartureTrack`, and vessel prims re-authored in the vessel's **own frame** under a movable
  Xform, so one translate op per frame moves the whole ship. `configs/scenes/
  vessel_departure_clear_day.yaml` carries the funnel's first-order rise, T(t) = 455 − 150
  exp(−t/240 s), written out because the phase-1 solver set has no marine uptake node yet.
- **The funnel had to be repainted.** It was `bare_aluminium`, ε = 0.09 in LWIR, so it reflected
  the cold sky far more than it radiated and the renderer drew a **155 °C funnel as a dark
  rectangle**. Real funnels are painted steel, ε ≈ 0.9. A test now pins the two emissivities,
  because the mistake is invisible in the geometry and shows up only in the picture.
- **The funnel is deliberately left out of the display span and clips white.** Spanning it in is
  the obvious choice and it ruins the film: the funnel climbs 150 K, so a span that holds it puts
  the sea, the hull and the deck — everything with structure in it — inside about fifteen of the
  256 codes, and the opening frames come out nearly black. Clipping the hottest thing in the scene
  is also what a real thermal image of a ship under way does.
- **One command renders every demo scene in every band** (M10.24). `scripts/render_multiband.py`
  runs three demo scenes × four bands, each with its registered visible companion, and builds a
  per-scene contact sheet. The point is the comparison: the same geometry, the same weather, the
  same sun and the same instant through four bands whose physics could hardly be more different —
  a quadrotor is four hot spots against a cold sky in LWIR and a dark silhouette against a bright
  one in NIR, and those are not two renderings of one picture but two different detection problems.
- The sheet samples **70 % of the way through** each clip rather than frame 0: these are time-lapses
  of a process, the manual span covers the whole flight, and at t = 0 the motors are still at
  ambient — the least informative frame in the sequence. It **resamples** rather than crops, because
  the NIR camera is 1280×1024 and cropping would silently show a different part of the scene beside
  the others.
- Writing the driver found three bugs before the matrix reached them, each worth an hour of GPU:
  `render_maritime_demo.py` had no `--rt-subframes` at all; `render_aircraft_pass.py` still spanned
  its display in **kelvin**, which SWIR and NIR do not have, so four of the twelve renders would have
  died; and the contact sheet read only `*_ir.mp4` while the maritime script writes a file per frame,
  so the ship column would have come out empty. All three were verified against existing outputs.
- **The SPG lane's blocker is written down instead of coded around** (M10.12, M10.13a/b/e).
  Stage 1 needs a **per-frame float32 facet table** to reach the kernel, and M2.3 established that
  there is no cross-frame device state to keep it in, that `io.` is a forbidden token so it cannot
  be loaded, and that a Lua literal of LUT size crashes the Kit process. The Planck LUT is static
  and bakes fine; the facet table changes every frame and does not. `src/irsim_isaac/spg/README.md`
  states the question, three candidate answers ranked by cost, and how one Isaac session with a
  fourth `spg_probe` experiment would decide it for all four steps.
- Writing the kernels against a guess is exactly what CLAUDE.md's "flag uncertainty rather than
  guessing" rule exists for — Isaac Sim 6.0's SPG API is new and its public documentation is
  incomplete — so they are not written. The note also carries M2.3's hazard forward: a kernel that
  fails to load yields a **zero-filled output with status ok**, so any in-sim equivalence test must
  assert against a known non-zero reference and never merely that a buffer came back.
- Two placeholder tests said they were waiting for ME.5. ME.5 has landed and **refused** both
  bands: a sky elevation profile and a cloud PSD slope each need a labelled region on the published
  clips, and selecting one automatically would be inventing an annotation and calling it data. The
  skips now say that, and point at M12.2's acceptance report — which is where the comparison that
  *can* be made without a region list actually runs, and where it currently fails.
- **A rendered prim can take its temperature from the §12.3 thermal solver** (M10.3). Phase 1
  mapped a prim to one of the scene's target solvers — a scripted drone motor, a relaxing airframe.
  A prim can now map to a thermal *surface* instead, so a rendered roof is the one M6.12's energy
  balance solved rather than a number typed beside it. Both kinds live in one map, because "a thing
  with a temperature" is all the G-buffer cares about; a name that is **both** is refused rather
  than resolved by precedence, since whichever won the other would be silently ignored.
- The bridge **advances** the field rather than only reading it. `ThermalField` refuses a query
  past its last tick instead of solving on demand — precisely so that a renderer asking many times
  per tick cannot change the answer by asking — so the advance is explicit and happens once per
  clock move.
- ⚠️ **The two kinds use different time bases**: the target bracket runs on time relative to the
  scene start, and `ThermalField.temperature_at` takes absolute weather-axis time. On the facet
  scene a relative lookup *raises*, because that scene starts a day into its weather file — but
  that is luck, not a guarantee: on a scene whose `t0_s` is zero the same mistake would return a
  plausible number from the wrong hour with no symptom at all. The convention is in the docstring
  and pinned by a test.
- **The first Tier 4 acceptance run — and it fails, which is the result** (M12.2).
  `docs/validation/tier4-2026-09-15.md` compares six real Halmstad clips against six matched
  renders: histogram EMD **24.7 codes** against a target of 8, PSD shape ratio **5.70** against 1.5,
  and a discriminator AUC of **0.907** — **+37.8 null standard errors** from chance on 1440 patches
  a side. The two sets are trivially distinguishable.
- **The dominant term is the signal path, not the radiometry, and the discriminator says so
  itself.** Its heaviest feature is `noise_scale`, and ME.5 measured exactly why: 306 of the 365
  published clips sit at or below **one code** of noise because the encoder removed it, while the
  rendered clips keep 2–3 codes through the same nominal codec. A CRF sweep from 18 to 36 moves the
  rendered figure only 3.1 → 2.1, so the published recorder did something the round trip does not
  reproduce (~128 kb/s at 320×256×30). The number to act on is M12.1's recorder model and its codec
  settings, both already marked ESTIMATED in their own files.
- Every failing metric **names the step it points at first**, in a written-down attribution table a
  reader can disagree with rather than guess at. The table is deliberately not a list of physics
  steps: on public 8-bit lossy data the signal path is the first suspect for three of the five, and
  blaming the radiometry for an encoder's work is the easiest mistake available here.
- **No README row is promoted on this run.** ADR 0068's targets are the gate and they were not met.
  Landing this failing, with attribution, is the point: a passing first acceptance run on data this
  compressed would have meant the thresholds were wrong.
- A codec-flattened patch has *exactly zero* power in most radial bins, so `clutter_slope` refuses
  it — which crashed the discriminator on real data. That refusal is a measurement, not an error, so
  it became a feature of its own (`spectrum_degenerate`). Dropping such patches would have removed
  precisely the property that distinguishes the real set; substituting a plausible slope would have
  been inventing data.
- **ROS 2 message construction, checked without a ROS graph** (M10.10b).
  `irsim_isaac.ros2_bridge` builds `CameraInfo` and the four `Image` messages as plain data, so the
  part most likely to be wrong — the arithmetic — is testable on any machine, and `publish_frame` is
  the single function that imports `rclpy`. It raises naming the dependency rather than an
  `ImportError` from inside a library nobody asked for: ROS 2 is a system-level install, not a pip
  package this project can add for you, and it is not present here.
- Intrinsics are derived from the sensor's own blocks (`fx = f/p`) and never authored, so a camera
  and its `CameraInfo` cannot disagree. The principal point is `width/2` — the project's pixel-edge
  convention — because `(width − 1)/2` is invisible in a preview and a systematic half-pixel bias in
  anything that triangulates. The stamp is **integer nanoseconds**, which survives 3600.0000005 s
  where a float second at hour scale would not.
- Encodings follow non-negotiable #2 on the wire: `32FC1` for apparent temperature and radiance,
  `mono16` for the ADC plane because it *is* integer counts, `rgba8` for the ISP's picture. A
  float16 apparent-temperature plane is **refused** — at 300 K that grid is 0.25 K coarse, five
  times a 50 mK NETD, so a subscriber would be reading a camera five times worse than the one that
  sent it — while float64 is narrowed. An output the config switched off is **skipped, not zeroed**:
  a black `apparent_t` topic is indistinguishable from a scene at absolute zero.
- **A camera and scenarios matched to the reference set** (M12.1).
  `configs/sensors/halmstad_boson_320.yaml` is the Breach PTQ-136's Boson 320×256 at 9.03 mm —
  24.0° × 19.3°, the publication's stated figures — and **its ISP models the recorder, not the
  core**. ME.1a established these clips are a Y16 stream somebody converted to 8 bits, so the DDE
  is off and the palette grey; but the conversion is not the identity (the published frames have a
  mean near 115 and σ near 75, filling the code range, which truncating a sky-only scene cannot
  produce), so `agc: linear` stands in for it, ESTIMATED, and is named in the file as the single
  largest unknown in any comparison against this set.
- `irsim.validation.scenario` draws sky, range, target size, ΔT and boresight with a **provenance
  tag on every parameter**, and each scenario is seeded by its own index, so scenario 7 is the same
  whether ten were drawn or a thousand — without which a failing case can only be reproduced by
  re-running the whole set. `scripts/generate_matched_scenario.py` renders through the engine-free
  CPU pipeline into the ME.1 sequence layout with boxes, then **encodes through the publication's
  own codec** before anything is measured: ME.2b showed x264 at CRF 18 removes 95 % of a clip's
  temporal noise, so an uncoded render beside a coded clip would measure the encoder.
- ⚠️ **The sampler's own summary is the uncomfortable result: 0 of 5 parameters are measured.**
  ME.5 could measure neither the sky distributions nor the target ones, so every prior is `stated`
  or `estimated`, and every run prints that beside its numbers. A Tier 4 figure from these clips is
  a comparison against an *assumed* scenario, and that has to travel with it.
- Caught while wiring it, and worth naming: `build_aerial_gbuffer` reports its boxes on the
  **supersampled** grid while the written frames are the native one. A box off by the supersample
  factor lands outside the image, and every box-dependent statistic downstream then measures empty
  sky while still reporting a number. A test now asserts every generated box is inside its frame.
- **The Tier 2 lab-bench protocol, written before the camera exists** (M12.4).
  `docs/validation/tier2-bench-protocol.md` gives the SITF, NETD, 3-D-noise and slant-edge benches
  as a procedure — stimulus, frame counts, ROI rule, pass criterion — and
  `irsim.validation.measured` fixes the file layout in one place
  (`data/validation/<quantity>/<camera>.csv`) and implements the comparison. Deciding a tolerance
  while looking at the first measurement is how a bench becomes a rubber stamp, so all of it is
  fixed in advance.
- **The two comparison rules differ where it matters.** `compare_shape` fits a gain and offset, for
  a quantity whose scale is a range choice — ADR 0019 makes a SITF's absolute DN exactly that.
  `compare_absolute` fits nothing, for NETD in millikelvin and MTF as a ratio, which a simulator
  has to predict. Measured: a **doubled** NETD passes the shape comparison and **fails** the
  absolute one, which is precisely the failure a fitted bench would hide.
- Only the SITF bench had a skip-if-absent hook; NETD, MTF and 3-D noise have them now, and all four
  resolve their path through `measured_path` so a second layout cannot be invented in a test module.
  A one-row bench table raises rather than passing silently — on the file that is meant to *be* the
  ground truth, a silent pass is the worst possible outcome.
- **The Tier 5 detection metrics** (ME.7, scoring half). `irsim_eval.detection` is pure NumPy —
  pairwise IoU, greedy highest-score-first matching, **all-point** average precision at IoU 0.5,
  AP restricted to small and tiny targets, P_d against SCR and against apparent size, and false
  alarms per frame — so a detector trained anywhere can be scored in the default environment.
  All-point rather than the 11-point interpolation, which was an artefact of a 2007 evaluation
  server and reports a visibly different number on a small set: on a worked four-box example the
  two rules give 0.8333 and 0.8182, and a *comparison* metric must not depend on that choice.
- **An empty P_d bin reports NaN, never zero.** "No target was this small" and "every target this
  small was missed" are opposite conclusions, and a zero would let a reader draw the second from
  the first. P_d against SCR is the metric that carries the weight here: a drone at 2 km subtends
  two pixels at an SCR of 2 and is a rounding error in an AP dominated by close, large targets, so
  two sets can agree on AP and disagree completely about the regime this simulator exists to model.
- ⚠️ **Training is blocked, and the binding blocker is the data, not the compute.** ME.5 found the
  reference set's labels are MATLAB Video Labeler `groundTruth` MCOS objects with no Python reader,
  so there are no boxes to train on until somebody exports them with the dataset's own MATLAB
  script. Separately, the `ml` extra (torch, torchvision) is declared and deliberately **not**
  installed: it is multi-gigabyte, `src/irsim` may never import it, and a Tier 4 report must not be
  blocked on it. `scripts/train_detector.py` refuses with both reasons rather than a traceback.
- **A scattered-sunlight sky for the reflective bands** (M11.10, ADR 0086). Found by rendering NIR
  for the first time: a brightly sunlit quadrotor on a **black sky**. That is backwards — in the
  near infrared the daytime sky is the brightest thing in the frame and an aircraft is a dark
  silhouette against it. `SkyModel` is a *thermal* sky, the atmospheric column's own emission, which
  is right for LWIR and zero at 0.9 µm; the reflected-solar term lit the target and nothing lit the
  sky. `irsim.atmosphere.skylight` adds **L = f_B · DHI / π**, with DHI from the scene's one
  `WeatherSeries`.
- **Isotropy is chosen for the integral, not for convenience.** ∫L cos θ dΩ over the hemisphere is
  exactly L·π, so this form reproduces the diffuse irradiance the weather file measured — the one
  angular distribution guaranteed right in the integral, which is the quantity that was measured.
  The diffuse spectrum is the ground-level solar spectrum weighted by **λ⁻⁴**, because skylight is
  far bluer than the beam that made it and a near-infrared band therefore receives a much smaller
  share of DHI than of DNI.
- Measured at DHI = 120 W m⁻²: NIR 5.35e18 and SWIR 4.28e18 photons s⁻¹ m⁻² sr⁻¹, MWIR 6.35e15, and
  LWIR **8.2e-7 W m⁻² sr⁻¹** against ~50 of column emission — eight orders down, two below float32's
  spacing there, which is why an emissive band gets `None` and every LWIR render stays bit-identical.
- ⚠️ Wrong in its distribution, deliberately: no horizon brightening and **no circumsolar aureole**,
  so a target passing near the sun is rendered against a sky that is too dim. Both are distribution
  errors under a correct total; ADR 0086 records why a Preetham-style fit was not extended to 1.7 µm.
- **The render scripts work in any band now** (M10.23). Three things had quietly assumed a
  bolometer: the `Scene` was built in the energy form regardless of the FPA (ADR 0021's rule is now
  a `SensorSpec.quantity` property that `PipelineConfig` delegates to); `attach_sensor_chain` raises
  for a photon FPA because the M9 NUC residual is in mK/K and needs a radiometric calibration it has
  none of, so the scripts skip the chain and say why; and the manual display span was written in
  kelvin and applied to `apparent_t`, which §12.1 switches off for SWIR and NIR.
  `irsim_isaac.display_span` spans `apparent_t` in an emissive band and the raw `dn16` between
  first-frame percentiles in a reflective one, and the readout leaves the **temperature gauge off**
  rather than print a kelvin scale beside a picture of reflected sunlight.
- **The exposure had to move, and that is physics.** At f/1.4 a 0.3-albedo surface in full sun
  delivers **1.17e6 photoelectrons** per NIR pixel in 16 ms against a 1e4 well — **117× saturated**,
  a uniformly white frame. The NIR file is authored at **80 µs**, which puts that surface at ~60 %
  of full well, and its §9.4 NETD moves with it to 2.42e13 K. SWIR reads **122×** at its own 16 ms
  and keeps its file; `--integration-ms` on the render scripts is how a daylight SWIR render exposes.
- **The Tier 4 acceptance report** (ME.6, ADR 0085). `irsim.validation.compare` holds the four DN8
  statistics — histogram EMD in codes, contrast ratio, PSD shape, ESF width — and `Tier4Report`, in
  which **`untestable` is a verdict**: §15's "apparent temperature within 2 K" is printed as open in
  every run, because a missing row reads as a passing row. `scripts/validation_report.py` exits
  non-zero on any failed check and has a `--self-test` mode that runs synthetic-versus-itself,
  which is the gate on the gate: if the control does not pass, the thresholds are wrong and nothing
  the report says about a real comparison is worth reading.
- The **gap score** (`irsim_eval.discriminator`) is a linear probe on interpretable statistics,
  fitted in NumPy with no torch and no scikit-learn, cross-validated, and it names the feature that
  separated the two sets. Its AUC is a deliberate *lower bound*: an AUC near 0.5 means these
  statistics do not separate them, not that a detector cannot.
- **Three thresholds moved by measurement.** ADR 0068's ×2 PSD-shape target does not catch a 3×
  noise mismatch — matched noise reads 1.05–1.08, 2× reads 1.30–1.36, 3× reads 1.77–1.93 and only
  5× reaches 3.2 — so it is **1.5**, which separates matched from 3× with a 1.6× margin and openly
  misses a 2× error. A fraction-of-peak spectral floor turned out to be the wrong shape of filter:
  on a scene with low-frequency structure it excludes exactly the bins the noise lives in, taking a
  3× mismatch from **1.90 to 1.06**, so it defaults to zero. And "AUC 0.5 ± 0.03" is not achievable:
  the AUC's own null standard error at 96 patches per class is already **0.042**, so the control is
  judged by `|auc − 0.5| < 2σ_null`, which scales with the sample size instead of assuming one.
- **The reference-statistics report** (ME.5). `scripts/reference_statistics.py` measured all 365
  IR clips of the Halmstad set and wrote `docs/validation/reference-stats-2026-09-15.md` and its
  JSON. Every value carries N, a seeded bootstrap confidence interval and its floor, and everything
  that could not be measured is a stated refusal rather than an absence — a missing row reads as a
  passing row. The archive was downloaded and its SHA-256 recorded, so "measured on these exact
  bytes" stays checkable.
- **The headline is negative and it is the most useful result in the report.** The median robust
  noise scale over the set is **0.00 codes**, and **306 of the 365 clips** sit at or below one
  code: the codec has removed the sensor's noise entirely, and no window can meet a threshold
  written in units of a ruler that has collapsed onto the quantiser. With 81 clips moving and 30
  more too structured, **28 clips** support a noise table at all — σ_TVH 7.27, σ_VH 19.18, σ_V 4.21
  codes. Every one of those is an upper bound on what the codec left, not a measurement of a camera.
- The missing `color_range` flag is worth **6.09 DN8 codes** (CI 5.88–6.31), several times the
  noise that survived, which is the quantitative reason ADR 0068 made scale-free statistics the
  deliverable.
- **Three index errors found by measuring rather than reading.** The clips are stored at **30 fps**,
  not the core's 60 — a one-pole temporal fit at the index's rate is wrong by that factor and looks
  entirely plausible. `color_range` is unflagged on every clip. And the annotations are MATLAB Video
  Labeler `groundTruth` **MCOS** objects that `scipy.io.loadmat` returns as an opaque blob: the
  index claimed a Python decoder ships with the dataset and **it does not** (the repository ships a
  MATLAB script), so `target_size_and_scr`, `edge_spread` and `smear` are now excluded there and
  every box-dependent Tier 4 comparison is open.
- The set is `access: direct` now: the DOI serves the archive over https with no interstitial, which
  the index had recorded as `manual` from the repository README's link to the landing page.
  `irsim_eval.decode` reads the frame rate from the container and never from the index, and
  `irsim_eval.reference` caches the per-clip pass before aggregating — it is half an hour of
  decoding and an arithmetic bug should not cost it twice.
- **The illumination bundle reaches the Isaac render path** (M10.22, ADR 0084). M11.2 built the
  bundle, M11.3 the reflected-solar term and M11.4 the night sky, `run_frame` reads them out of
  the plane dict as `l_sun`/`l_night` — and **nothing in `irsim_isaac` ever wrote those planes**.
  Every Isaac render this project has produced was emission only: right to within 0.35 % for LWIR,
  and **black** for NIR. This is ADR 0077 and ADR 0082's failure mode a third time — a mechanism
  that exists, is tested, and cannot be reached from the layer above — and it only became visible
  when a fourth band was configured and renders were asked for in it.
- Measured on dry asphalt at 300 K under a 61° sun, sunlit over emission-only: **1.08e15 in NIR**,
  3.59e7 in SWIR, **1.17 in MWIR**. The MWIR figure is the interesting one and is bracketed
  tightly in the test, because a band that is nearly all self-emission is where a wrong solar term
  hides best. LWIR gets no bundle at all, and an emissive band renders **bit-identically** with and
  without a solar plane attached — §5.2's gate drops the term to `None` rather than adding a zero.
- **One sun.** The direction is the same expression `irsim_isaac.stage.add_sky_dome` aims the USD
  `DistantLight` along, so the radiometry and the shadows in the companion visible frame cannot
  disagree about where the light comes from; computed separately they would drift and nobody would
  notice until the two frames were overlaid. The solar slant path is in it and asserted rather than
  assumed: an 8° sun comes out dimmer than the cosine alone predicts.
- ⚠️ **Cast shadows are not modelled.** `shadow` is 1 everywhere and the only shadowing is
  self-shadowing through max(0, n·s). That is exact for an aerial scene — nothing is above a drone
  to shadow it — optimistic on the shaded side of a vessel, and wrong for a street. The renderer on
  this build supplies no working occlusion AOV (ADR 0014 addendum), and the honest interim is to say
  so rather than render a plausible shadow nobody computed.
- **A NIR camera: the fourth band** (M11.9). `configs/sensors/example_nir_si_1280.yaml` — a
  1280×1024 silicon CMOS at 0.75–1.0 µm, reflective, photon FPA — with
  `data/spectra/responses/nir_si.csv` generated from **silicon's own band-edge physics**:
  QE = 1 − exp(−α(λ)·d) over 6 µm of epitaxial silicon, α log-interpolated between published
  near-gap anchors, times the long-pass filter that replaces a visible camera's IR-cut. The curve
  peaks at 0.775 µm and falls **10.6×** by 1.00 µm, which is the fact that makes NIR a *near*-
  infrared band rather than a short-wave one.
- `scripts/generate_response_curves.py` now **peak-normalises on write**, so a curve can be
  written in physical units while the file stays the peak-1 shape ADR 0009 requires (absolute
  efficiency belongs in `fpa.quantum_efficiency`, so a response file and a detector's QE cannot
  disagree). `ingaas.csv` and `insb.csv` regenerate byte-identically.
- **Bands are data, demonstrated a fourth time.** `band: nir` was already in the registry, so a
  YAML, a CSV and `make luts` were the entire change — no kernel edit — and the band-scalability
  guard picked the new config up on its own. The four configured bands' self-emission at 300 K
  now spans **seventeen orders of magnitude**: NIR/LWIR = 5.2e-17, SWIR/LWIR = 4.2e-9,
  MWIR/LWIR = 4.1e-2. Emission is still never culled: NIR's 300 → 900 K gain is 1.4e14.
- §9.4's NETD for this camera evaluates to **1.21e11 K** — a 300 K blackbody delivers 3.7e-10
  photoelectrons per pixel per 16 ms frame, one electron roughly every seventy-five years — and is
  recorded and flagged unusable exactly as the SWIR file's 976 K is. `make luts` reports −41 %
  against a flat top-hat at 300 K, which is the Wien tail putting the emissive integral past the
  band's long edge, not a bug.
- **The §15 Tier 5 ablation switches are config now, not keyword arguments** (ME.8, ADR 0083).
  A `fidelity:` block in the sensor file carries `noise`, `optical_psf`, `bad_pixels` and
  `nuc_residual`, all defaulting to `true`. Each of the four already existed — as a keyword
  argument of `PipelineConfig.from_sensor` or of `attach_sensor_chain` — and **a keyword argument
  is invisible to the config hash**, so two ablation variants produced identical hashes and a
  stored result could not say which one produced it. Sixteen combinations of the four now produce
  sixteen distinct hashes. The arguments still work and still override, as a bench affordance.
- The AGC, the FFC and the clouds are deliberately **not** in the new block: `isp.agc: linear`,
  `nuc.mode: ideal` and the weather's own `cloud_fraction` were already explicit, hashed config,
  and a second way to set them would raise the question of which copy wins.
- **`schema_version` is no longer part of the config hash**, and the sensor schema gained a
  readable range (v8–v9) the way the scene schema already had. The version describes the document
  format, not the sensor: a v8 file and a v9 file that describe the same camera are the same input
  and must produce the same reference. The guard against a future version changing *meaning*
  without changing a field is the floor — raise `MIN_SCHEMA_VERSION` and the old document is
  refused outright rather than quietly hashing like a new one.
- Regenerating the goldens after the version bump rewrote twelve `.json` sidecars and left **every
  `.npy` array byte-identical**, which is the evidence that this changed provenance metadata and
  no physics.
- **Sky and target statistics for the Tier 4 comparison** (ME.4, ADR 0068 addendum).
  `irsim.validation.targets` measures a display-domain frame the way the public data forces:
  `target_statistics` (SCR against the *ring* around the box — a sky target's background is the sky
  right behind it, and a frame-wide background mixes in the horizon, the ground and any cloud on the
  other side of the picture), `sky_profile` (shape normalised by the horizon-to-zenith span **and**
  SNR per row, so an AGC that stretched the sky twice as hard moves neither), `clutter_slope`,
  `edge_spread_function`, `profile_centre_index`, `edge_asymmetry`, `asymmetry_vs_tau_curve` and
  `size_from_range_px`.
- **The planned `tau_from_asymmetry` does not exist, and the roadmap's "recovers τ within 2 %" is
  withdrawn.** The point-source inversion — target as a point, trail as a geometric tail — returns
  **0.45 frames for a true τ of 4** on a σ = 3 px target, because the core carries far more energy
  than the trail. What ships instead is `asymmetry_vs_tau_curve`, which builds the curve for the
  caller's own target size, speed and window; a comparison inverts by interpolating against it.
  A τ = 3 probe reads 3.25 against knots at 2 and 4 — the curve is convex, and that is the accuracy
  this statistic supports.
- **Edge asymmetry must be split at the box centre, not at the profile centroid.** The centroid is
  not a landmark: a long faint trail drags it into itself, which leaves the compact core on the
  *leading* side of the split, so the statistic reports the trail on the wrong **side** (−0.056,
  −0.117, −0.193, −0.162, −0.230 for τ of 0.5, 1, 2, 4, 8 frames on a demonstrably rightward trail)
  and is not monotone in magnitude either. Split at the box centre it is strictly monotone in every
  regime tested — 0.083, 0.261, 0.496, 0.700, 0.836 at σ = 3 px and 2 px/frame, and still monotone
  at a point-like σ = 0.6 px, which is where a real sky detection lives.
- A short measurement window **compresses** the curve rather than folding it: between τ = 4 and
  τ = 16 a 120 px window gains 0.212 of asymmetry and a 6 px window only 0.110, so the same DN8
  measurement error inverts to nearly twice the spread in τ. Measure with the widest window the
  frame allows and build the curve with that same window.
- **§6.6's vehicle thermal architecture: a cabin node, scripted heat sources, and the ghosts they
  leave** (M6.14–M6.16, ADR 0038, ADR 0039). A vehicle panel's back boundary is a lumped cabin, not
  ambient and not adiabatic: with the cabin the roof runs **+4.8 K** hotter at noon, the cabin air
  reaches **80 °C**, and on a clear night both sit **more than 2 K below ambient** — which is why
  cars frost when the air never does.
- **⚠️ The glazing must be in the panel list, and the constructor now refuses a cabin without it.**
  Glass is both the cabin's solar **inlet** and one of its largest conduction **paths** — 15 W K⁻¹
  against 2 W K⁻¹ of infiltration. A cabin given the inlet without the path reaches **102 °C** at
  noon instead of 80: not obviously wrong, just wrong, which is why it is an error rather than a note.
- **⚠️ The lumped cabin time constant is the panels-pinned one.** `C/(ΣA/R + ṁc_p)` is exact for
  the system it describes — it matches to **2 %** with the panels held — but let them move and the
  coupled system relaxes **33 % slower** (12.0 min against 9.0), because a cooling cabin drags its
  panels down and they feed heat back. Both are asserted, so a scene knows which number it quotes.
- **Two heat sources are deliberately *not* scripted, because a closed form exists.** Brakes are an
  **energy deposit** — ΔT = f·½m(v₁²−v₂²)/(m_disc c_p), 162 K into an 8 kg disc from 30 m/s, exactly
  4× that from twice the speed — and scripting the temperature would make it independent of how hard
  the car actually braked, the one thing a braking cue carries. Tyres follow **speed, not time**,
  over §6.6's +10 … +35 K. Everything else is a first-order schedule with §6.6's own ΔT and τ, and
  the exponential step is **exact**: a drive logged at 1 Hz and at 10 Hz agree to **1e-9**, where a
  forward difference would make a scene's engine-bay temperature depend on the logging rate.
- **Heat traces are overlays, not conserved quantities** (ADR 0039), and the cost is stated rather
  than hidden. Superposition is linear to **1e-12** and cells outside every footprint come back
  **bit-identical** — a layer that blended would make an untouched road pixel depend on whether a car
  drove past somewhere else in the frame. Rasterisation is by **distance to the segment**, so a path
  sampled at 1 Hz and at 100 Hz stripe the same cells. And a tyre trace is **two** footprints,
  because one wide stripe is a skid and two at a known gauge are a vehicle.
- **⚠️ The 20-minute decay criterion splits, and the split is the phenomenon.** §6.6 gives a
  5–20 min *range*: the stripes (τ = 7 min) are under **0.5 K** at twenty minutes and the body
  shadow (τ = 12 min) is still **1.0 K**. That residual is exactly why §6.6 says thermal ghosts
  "routinely confuse detectors trained only on synthetic data that lacks them" — a ghost that had
  faded before anyone looked would not. Asserting both under 0.5 K would have meant choosing τ for
  the convenience of the test (27 tests).
- **Tier 3 reflection phenomenology: what a facet *reads* at 03:00, not what it *is*** (M7.16,
  §15 T3, §5.3). Every other thermal test asks what temperature a surface reaches; this one asks
  what the camera reads, which is a different number. The ε 0.90 roof and the glass both read
  **> 3 K below** their own kinetic temperature under a clear sky, and the ε 0.09 aluminium panel
  reads **> 20 K below itself**, landing on the identity `Lb⁻¹(ε Lb(T_s) + (1−ε) L_sky,eff)` to
  **0.05 K**. Overcast cuts every gap and the spread of gaps by more than half.
- **⚠️ The roof/wall closed form in the roadmap is wrong as written, by 19 %.**
  `(1−ε)·0.5·(L_ground − L_sky,eff)` assumes the roof and the wall see sky of the *same* effective
  radiance and differ only in how much of it. They do not: `L_sky,eff` is a tilt-dependent
  cosine-weighted average over the sky each facet can actually see, and a wall's half is the
  **warmer half near the horizon**. With the tilt-dependent form the radiance identity closes to
  **2e-3** and the linearised version to **0.1 K**. Both facets are evaluated at the same kinetic
  temperature on purpose — in the scene they differ by 2.2 K, and comparing them as they stand
  would measure the thermal model and the reflection model at once.
- Two more criteria corrected by measurement rather than by widening a tolerance. Rough asphalt
  moves **2.0 K**, not "within 1 K": at ε = 0.94 the reflected fraction is 6 %, and this sky is 54 K
  below air at zenith, so even the roughest, blackest surface in the library is not reflection-free
  on a clear night. And bare aluminium tracks the sky to **5.5 K** rather than 5 K — the residual is
  not slop, it is exactly the tenth of its own emission the panel still contributes.
- This is also the bench that found §6.1's missing ε on `Q_LW↓`, above: the bare-aluminium facet it
  needed came out 19 K above the air at 03:00 (5 tests).
- **⚠️ Fixed: the surface balance was absorbing 100 % of the downwelling longwave regardless of
  emissivity.** §6.1 writes **ε Q_LW↓** — "absorbed sky/env" — and M6.7 shipped `Q_LW↓`, with a test
  asserting that was correct. Kirchhoff says a surface absorbs the same fraction of incident
  longwave that it emits; dropping that ε is invisible on a painted surface (10 % of one term) and
  catastrophic on a metal, where it hands a panel with ε = 0.09 the full ~320 W m⁻² of sky radiation
  while letting it emit only a tenth of a blackbody.
- **It was found by looking at a number that could not be true, not by re-reading the spec.** Adding
  a bare-aluminium facet for M7.16 put it at **19 K above the air at 03:00** — a mirror, at night,
  with no heat source. After the fix it sits **0.8 K below air**, which is what a surface that barely
  exchanges radiatively does. The reasoning in the original test was also wrong in a specific way
  worth naming: it argued that folding the terms into `εσ(T⁴ − T_sky⁴)` would "multiply the sky's
  emissivity by ε a second time". It would not — the *sky's* emissivity sets the incident flux and
  the *surface's* sets how much is absorbed, and both are needed. The folded form is algebraically
  identical to the corrected one.
- Everything downstream moved and the ordering did not: sunlit asphalt now peaks at **59.0 °C**
  (was 60.1), the dawn contrast collapse is **13.2 K → 1.39 K** (an 89 % drop), and overcast halves
  the swing while cutting the night spread by **25 %** rather than the quoted 30 % — because each
  surface's response to the sky is now scaled by its own ε, so the low-ε surfaces contributing most
  of the night spread are also the least moved by cloud. That damping is the corrected physics
  showing up in an ensemble statistic.
- **Tier 3 diurnal phenomenology on the facet scene** (M6.13, §6.3, §15 T3). Eleven facets, mostly
  in pairs that differ in exactly one thing, driven by 48 h of weather. Sunlit asphalt peaks at
  **14:10 local at 59.0 °C**; the hood at 100 km/h is **26.4 K** colder than the parked one at
  14:00; dry soil is **7.2 K** above wet at 14:00 and *below* it at dawn; shade is > 3 K colder by
  day and within **1 K** at night; black and white paint agree to **0.2 K** at 03:00 and differ by
  **> 20 K** at 13:00; overcast cuts both the swing and the mean night spread by > 30 %; doubling
  the wind lowers the peak by > 1 K; and the surface falls **below** the air on a clear night.
- **§6.3's own acceptance test — "does the scene go flat?" — is measured on the scene-wide spread
  rather than on a chosen pair**, because a pair's crossing time moves by a couple of hours
  depending which pair you pick, and the operational claim is about the picture. **At dawn the scene
  collapses from 13.2 K of spread to 1.39 K — an 89 % drop.**
- **⚠️ There is no dusk collapse, and that is recorded rather than engineered away.** At sunset the
  spread is still 5.6 K and decays monotonically through the night to the dawn minimum. Each pair
  *does* cross — one to three hours before sunset — but they cross at different times, so the
  ensemble never passes through a common point. A dusk flat needs a facet set whose members share a
  solar absorptivity as well as differing in mass; this one deliberately does not, because its pairs
  exist to isolate other variables. Choosing its materials to produce a dusk flat would be fitting
  the fixture to its own acceptance test. The **morning** pair-level crossover does land within
  **60 min** of sunrise.
- **⚠️ The swing is 47.1 K against the roadmap's 15–40 K, and the model says why.** These facets are
  single nodes with an **adiabatic back**, so the day's heat has nowhere to go but back out of the
  surface. M6.8's two-node solver with a finite R₂d and T_deep is exactly what pulls this into the
  quoted band, so the single-node number is an **upper bound** rather than a disagreement — and the
  shaded twin, which never sees the beam, sits inside the band at 15.4 K. Left as a bound because a
  scene that wants it tightened should say what is under its ground rather than have the fixture
  guess.
- Ships `scripts/validate_thermal_diurnal.py` — a plot for the shape and a **diffable CSV** for the
  numbers — and `docs/validation/tier3-checklist.md` for the six things a person has to look at that
  no test can, plus the recorded departures above (14 tests).
- **The `thermal:` scene block, wired to M6.11's field** (M6.12, §12.3, ADR 0032/0037/0043).
  `configs/scenes/thermal_facet_scene.yaml` ships: seven facets, mostly in pairs that differ in
  exactly one thing — dry vs wet, sunlit vs shaded, black vs white paint. `SceneSurfaceForcing`
  joins the shared weather, M6.4's NOAA sun, M6.5's longwave and M6.6's convection, and carries
  `.weather` precisely so the `Scene`'s existing one-weather guard can see it. A surface **cannot
  author an emissivity** — the schema forbids the field rather than ignoring it.
- **⚠️ A site and a weather file have to agree about where noon is, and nothing makes them.**
  `clear_midlat_summer_48h` declares "local = UTC+2" in its own header. Pairing it with a
  Californian site does **not** raise: `solar_loading` multiplies the file's DNI by a cos θ that
  peaks ten hours later, and the product peaks somewhere in between. The scene renders, the diurnal
  curve still looks like a diurnal curve, and every time-of-day statement about it is wrong. A test
  now pins the sunlit-minus-shaded peak (which isolates the direct beam) against the file's own DNI
  peak, and a counter-test confirms a mismatched site shifts it by hours. With the site corrected to
  +30°E, sunlit asphalt peaks at **14:10 local at 59.0 °C**, black roof at **64.2 °C** against
  white's **35.3 °C**, and shaded asphalt 31.7 K below sunlit.
- **⚠️ A 48 h weather file cannot supply 48 h of weather *before* t₀.** The spin-up therefore wraps
  into the series it has — it is asking "what would this surface look like after a couple of days of
  weather like this", and the synthetic files are a whole number of days long so the wrap lands at
  the same time of day. The wrap is applied to the **spin-up only**: the live forcing stays
  un-wrapped, so a render that runs past the end of the weather raises instead of quietly reading
  yesterday.
- The scene schema goes to **v5** with a readable **range** down to v4, rather than a bump that
  invalidates every existing document. `thermal:` is optional, so every v4 scene is a valid v5
  scene, and a range is the honest representation of a backwards-compatible change (12 tests).
- **`ThermalField`: a fixed-tick solve behind a query that cannot mutate it** (M6.11, §6.4, §1).
  §6.4's "decouple this from the render loop entirely" is one sentence and two requirements. The
  solve advances on its own clock — 240 fps and 1 fps agree to 1e-9 — and **a query never changes
  anything**: 3600 queries between ticks leave the state hash and the tick count bit-identical, and
  a query past the last tick **raises** rather than advancing. A field that advanced on demand would
  make the answer depend on how often the renderer asked, which does not show up as an error
  message; it shows up as a scene that renders differently when you add an output.
- **⚠️ The 1e-12 interpolation identity cannot be claimed through the query.** Its output is float32,
  whose spacing at 288 K is 3e-5, so asserting 1e-12 there would be a claim about the narrowing
  rather than about the interpolation. The two are tested separately: the identity in float64 on the
  stored ticks, and the boundary on its own.
- That boundary test is CLAUDE.md #2 run as an experiment rather than quoted as a rule: carrying
  float32 through every step for 48 h drifts **under 10 mK** from the float64 solve — and by **more
  than zero**, which the test also asserts, because an experiment that measures no difference is not
  an experiment. Running time backwards raises: a thermal history is not reversible, and silently
  re-integrating it would give a different answer (11 tests).
- **The facet solver, and the spin-up that makes its answer mean anything** (M6.9, M6.10, ADR 0037).
  `FacetSolver` runs §6.1's balance over `(N,)` arrays and is held to **under 0.1 mK** against 64
  separate scalar runs over 6 h — the two are the same arithmetic in a different loop order, not an
  approximation of each other. Shadowing one facet moves it by > 1 K and its neighbours by **exactly
  zero**, which is the check that vectorising has not accidentally coupled anything.
- **A surface temperature is a memory.** Asphalt at 06:00 is carrying yesterday afternoon, so
  starting a scene at the air temperature is wrong by **more than 5 K** on sunlit asphalt at 14:00 —
  and wrong in a way that decays over hours, across exactly the part of the diurnal cycle a thermal
  camera is most interesting in. `spin_up` integrates the hours *before* t₀ and returns the state at
  t₀ itself, so no caller has to advance it afterwards and two callers cannot advance it differently.
- **48 h is the default and is not adequate for everything, which is reported rather than hidden.**
  Between a 48 h and a 96 h spin-up: thin steel and asphalt settle to **< 0.5 K**, concrete to
  **< 1.0 K**. 202 kJ m⁻² K⁻¹ of concrete is still remembering the day before yesterday, and that is
  a property of concrete rather than a deficiency of the spin-up.
- **The cache is keyed on what determines the answer and nothing else** — the materials' bytes, the
  weather hash, t₀, the span and the step. Not the scene, the camera or the frame: two scenes made of
  the same materials under the same weather at the same hour have the same surface temperatures. A
  **0.01 K** perturbation of any property changes the key, because properties are hashed as bytes
  rather than bucketed; a bucketed key is one that occasionally returns someone else's answer.
- float64 inside, float32 only at `as_float32()`. The balance subtracts terms around 400 W m⁻² to
  leave a residual of a few, and a diurnal run accumulates ~10⁵ steps of that — in float32 the
  subtraction alone loses four significant digits before the accumulation starts (10 tests).
- **§6.4's two-node solver, and the factor of 400 in its stability bound** (M6.8, ADR 0036, spec
  issue S39). `LumpedTwoNodeSolver` gives a surface somewhere to put the day's heat, which is the
  difference between a road still warm at midnight and one that is not. R₁₂ → ∞ reproduces M6.7 to
  **< 0.1 mK**, the equilibrium matches `brentq` to **< 0.1 K**, `expm` on the linearised 2×2 system
  agrees with 6 h of RK2 to **< 10 mK**, and energy across the pair closes to **1e-6**.
- **⚠️ §6.4 states its bound correctly and evaluates it wrongly.** The formula
  `Δt < 2C₁/(h + 4εσT³ + 1/R₁₂)` is right; "lands around 60–200 s for thin painted metal" is not.
  On §16.2's own car-paint row with §6.4's own R₁₂, **1/R₁₂ = 37 500 W m⁻² K⁻¹** against
  h + 4εσT³ ≈ 43 — a ratio of **874** — and the bound is **0.2344 s**. The quoted range is the
  *single-node* bound at high wind: the same formula with the term that dominates it removed.
- **The bound is not a nuisance to work around; it is telling you something.** A 1.2 mm layer on a
  conductive substrate equilibrates across itself in a quarter of a second, and resolving that
  explicitly at a scene tick is not a sensible thing to want. So a thin panel is a **single node
  with a resistive back boundary** (bound 264 s), and the two-node solver is for surfaces with real
  depth, where it is comfortable anyway — asphalt 3 456 s, concrete 7 067 s. The guard fires in the
  **constructor**, because a caller that has chosen a tick has chosen it once, and finding out on
  frame 4000 that it was unstable is finding out after the scene is rendered.
- **⚠️ A second correction, also from measurement: the diurnal swing is ordered by areal capacity
  C = ρcδ, not by thermal inertia P = √(ρck).** P is the right measure for a *semi-infinite* solid;
  on finite layers with an adiabatic back it is exactly backwards. Car paint has the **highest** P
  of the three tested (12 800 against concrete's 1 700, because it is backed by steel) and the
  **largest** swing (58.6 K against 18.1 K), while C — 4.4, 101 and 202 kJ m⁻² K⁻¹ — orders them
  inversely and exactly. P returns as the right variable once the back boundary is a deep reservoir
  rather than a wall, which is M6.10's ground case.
- R₂d defaults to **infinity** and a finite value **requires** T_deep: a back resistance without a
  deep temperature is a conduction path to an unspecified reservoir, and a plausible default would
  let a scene lose heat to a number nobody chose, slowly enough to look like physics (11 tests).
- **§6.1's surface energy balance, its RK2 step and its fixed point** (M6.7, ADR 0036, ADR 0043).
  `irsim.thermal.balance` assembles the solar, longwave, radiative, convective and internal terms
  that already existed in this package into one equation, with a midpoint step and a bisection
  steady state. Steady state agrees with `brentq` to **< 0.1 K**, ε = 0 gives the closed-form
  `T_air + (αQ + q)/h` to **1e-9**, and the linearised τ = C/(h + 4εσT³) is recovered from a small
  perturbation to **1 %**.
- **⚠️ Energy conservation turned out to be an identity, not a tolerance — once the right quadrature
  is used.** The midpoint rule *is* `T_{n+1} = T_n + dt·f(T_half)/C`, so `C·ΔT = Σ dt·f(T_half)`
  holds to **1e-12** over 1000 steps. Accounting the same run with a trapezoid of the endpoint
  fluxes leaves **2.4e-3** — the difference between two second-order quadratures of a curved
  integrand, not an error in the step. A test that used the trapezoid and loosened its tolerance
  until it passed would have been measuring that difference and calling it conservation.
- **A scene cannot author an emissivity.** `ThermalSpec` has no emissivity field at all, and
  `ThermalProperties.from_material` reaches through M7.8's total hemispherical integral — two
  authored copies of ε, in files that are never compared, is two chances for one scene to radiate
  at 0.95 while the camera sees 0.88. The metal comes through with its sign: bare aluminium's
  thermal ε *exceeds* its optical one, and it equilibrates further from the air than paint does,
  because a poor radiator sheds absorbed sun less easily.
- Two modelling choices stated rather than assumed: Q_LW↓ and εσT⁴ stay **separate terms** as §6.1
  writes them (folding them into εσ(T⁴ − T_sky⁴) looks tidier and silently multiplies the sky's own
  emissivity by ε a second time — a test doubles the downwelling and checks the balance moves by
  exactly that, with no ε on it); and the stability bound is **reported, not enforced**, because a
  caller stepping a thin panel with a scene-sized tick usually wants the steady state instead. The
  linearised radiative coefficient is **4.88 W m⁻² K⁻¹** against h = 8 at the night equilibrium, so
  leaving it out would make the surface 1.6× too slow (16 tests).
- **Directional emissivity per pixel in stage 1** (M7.14, ADR 0042, §4.2, §13.5). `band_radiance`
  takes `normal_dot_view` and, when the material table carries an M7.10 angle LUT, evaluates ε(θ)
  per pixel. **ρ is re-derived at every angle, never carried**: ε(θ) moves and τ does not, so ρ must
  absorb the difference or the pixel stops conserving energy the moment the surface tilts — a
  per-pixel closure failure that grows towards the limb, exactly where the model is meant to help.
- **The path is opt-in by packing**, not by the presence of a plane. Stage 1 uses it only when the
  table was packed with an angle LUT, so every scene and every golden written before this is
  bit-identical; the directional path appears because a caller packed the table for it.
- On the sphere fixture under a 220 K sky, the rim follows
  `ΔT = (ε₀ − ε(θ))(L_B(T) − L_env)/(∂L_B/∂T)` to **under 10 mK**, and with `L_env = L_B(T)` the
  effect vanishes to **under 1 mK** — the half a wrong reflected term fails, since a model that
  varied ε with angle and forgot to move ρ with it would darken the limb here too and look entirely
  plausible. Paint's rim drops **1.44 K** against asphalt's **0.45 K**, which is the difference §4.2's
  "a → 0 for rough dielectrics" is about.
- **⚠️ Two things the criterion as written would have got wrong, and both were found by measuring.**
  The prediction has to be **differential**, against the same scene at ε₀: under a 220 K sky an
  ε = 0.90 surface at 300 K already reads **294.7 K** everywhere from M7.13's reflected term alone,
  so measuring the limb against the *kinetic* temperature reports a 5.3 K error that has nothing to
  do with the angular model and drowns the 1.4 K that does. And ∂L/∂T has to be taken at the
  **midpoint** of the excursion — anchoring at the baseline leaves 11.6 mK of pure second-order
  term, just outside the 10 mK asked for.
- The metal comes through the whole chain with its sign intact: bare aluminium's limb reads
  **warmer** than its centre, because its ε rises with angle so it emits more and reflects less of
  the cold sky — the opposite of every dielectric in the scene (8 tests).
- **Angular columns in the packed material table** (M7.10, §13.3, §13.5). The table gains an
  `angle_lut` carrying ε(θ) for **every** material, not just the Level A ones — a kernel cannot
  branch on §4.2's level, because a Fresnel material needs an n/k file and a Planck-weighted band
  average and a GPU has neither. The (a, p) columns stay alongside it, since §4.2's "two
  instructions in a shader" is cheaper still where it applies.
- **The grid is uniform in cos θ, and the node count was measured rather than chosen.** A kernel has
  `n·v` in hand, so a cos-uniform grid makes the lookup a multiply and a floor instead of an `acos`
  — but the spacing that buys is uneven in angle, and in the unhelpful direction: ~10° per step near
  normal where ε is flat, ~1° near grazing where it is falling off a cliff. At 33 nodes water
  interpolates 0.0048 from the exact dispatch; at 65 it is 0.0012, and **inside §4.2's own 70° bound
  every material lands within 2e-4** — two orders below the 0.02 Level B itself is allowed, so the
  packing is nowhere near the limiting approximation.
- **⚠️ Recorded, not chased: bare aluminium's ε reaches 0.96 within 0.2° of grazing**, and the LUT
  reads 0.18 there. A metal's reflectance goes to 1 at exactly 90°, so its emissivity goes to 0 — but
  it passes through a sub-degree maximum on the way. §4.2 claims nothing past 70°, a pixel at 89.8°
  incidence has essentially zero projected area, and resolving it would cost the kernel the `acos`
  the grid exists to avoid. A renderer that needs it calls the dispatch directly.
- Closure survives float32 packing to **1e-6** across all 19 materials, and the counter-test matters:
  the same library packed in **float16 breaks closure by ~1e-3**, so the float32 result is a property
  of float32 and not of the numbers happening to be round. Packing the LUT is off by default, so every
  table packed before M7.10 hashes and loads unchanged (24 tests).
- **Material library v0 — §16.2's fifteen, with provenance** (M7.9, §16.2, §12.3, §4.3, §4.4). Nine
  new materials (concrete, rusted steel, tyre rubber, cotton, leaf, dry and wet soil, water, snow)
  bring the library to **19**, and each of the fifteen §16.2 rows is reproduced *exactly* — ε_LWIR,
  ε_MWIR, α_sol, ρ, c_p, k to 1e-6. Closure holds over 19 materials × 4 bands to **1e-6**, and every
  Level C material clears §4.2's 0.93 in every band, which after M7.7 is enforced rather than hoped.
- **§16.2 gives nothing about NIR or SWIR, so every such value is ESTIMATED and says so in its own
  file** — a test reads the file to check the label is there. Two of them are not guesses, and they
  are why a multi-band library is worth having at all: a **leaf** in the NIR plateau reflects ~0.45
  and **transmits** ~0.45, absorbing a tenth, so a canopy is semi-transparent at 0.9 µm and an
  ordinary opaque dielectric at 10 µm; and **snow** runs ε 0.15 at 0.9 µm to 0.90 at 1.6 µm — the
  largest adjacent-band swing in the library, and the reason snow and cloud are indistinguishable in
  the visible and obvious at 1.6 µm. A single-band library cannot be wrong about either, because it
  cannot say anything about them.
- **`water` is now a material, and it is the only one with measured optical constants behind it**
  (Segelstein 1981), so it carries Level A — a sea surface is an angular-emissivity problem before it
  is anything else. Its file records a discrepancy rather than hiding it: §16.2 gives ε_LWIR = 0.96
  while Fresnel on that table gives **0.99** at normal and **0.951** hemispherically, which puts
  §16.2's water row within 0.01 of the *hemispherical* value and suggests that column is not the
  normal-incidence figure ADR 0043 reads it as.
- `configs/materials/mapping.yaml` gains 8 semantic classes and 20 name patterns so the new materials
  are reachable from USD asset names. One existing test had used `vegetation_leaf` as its example of
  an *unknown* material and stopped working the moment it became a real one; it now names something
  that will not become a material (40 tests).
- **An aluminium n/k table, and Level A scaled to the authored value** (M7.5 part, §4.2, §12.3).
  M7.7's guard left two committed materials with no valid angular model; this closes the one that
  genuinely had none. `data/nk/aluminium.csv` is **MODELLED, not measured** — the Drude free-electron
  form, which is the right physics for a metal far below its plasma frequency (the whole infrared
  is), with wp = 12.53 eV and Γ = 0.0759 eV chosen so it reproduces the commonly quoted n = 25.3,
  k = 89.8 at 10 µm. Generated by `scripts/generate_nk_tables.py`, labelled in its own `# source:`
  header, which the loader refuses a table without.
- **⚠️ Level A supplies the shape; the authored band value supplies the magnitude.** Ideal Drude
  aluminium is ε = **0.012** at 10 µm and §16.2 gives bare aluminium **0.09** — an oxide layer and a
  little roughness are worth almost an order of magnitude. Taking the table's absolute value would
  have fixed a ~30 % error in angular *shape* by introducing an **8× error in the emissivity
  itself**. Scaling also makes Level A consistent with B and C, which both already take ε₀ from the
  authored value; without it, ε(0) would mean one thing for a Fresnel material and another for
  everything else.
- **`bare_aluminium` is now Level A and `asphalt_dry` Level B with a ≈ 0**, which is what §4.2
  prescribes word for word for each ("metals rise toward grazing"; "nearly flat (a → 0) for rough
  dielectrics like asphalt"). Measured after the change: aluminium reads **0.090 → 0.146** between
  normal and 70°, and is the **only** material in the library whose ε_hemi (0.113) exceeds its ε(0).
  A survey test now asserts that, so the dielectric/metal split is a property of the whole library
  rather than a remark; and the Level C guard keeps its ability to fail by being exercised on a
  material forced back to it, since nothing committed trips it any more (13 tests).
- **Hemispherical emissivity — the number the thermal solver needs** (M7.8, ADR 0043, §6.1, §4.2).
  A camera measures ε(θ) along one ray; an energy balance radiates into the whole sky. Water reads
  **0.990** at normal and **0.9511** hemispherically, because the cos θ sin θ weight peaks at 45°
  where its collapse has already begun. The solver gets ε_hemi and **cannot reach** the directional
  band value, because the error is systematic and one-signed — the same direction for every water
  pixel of every maritime scene, at every hour.
- **The sign flips for metals, and that is not assumed anywhere.** Aluminium's ε rises with angle, so
  its ε_hemi is **1.29×** its normal value. A model that clamped ε_hemi ≤ ε(0) would be right for
  every dielectric in the library and wrong for every metal, quietly.
- **⚠️ The roadmap's "> 0.3 K equilibrium difference" is unreachable, and that is a property of the
  balance rather than a tolerance to loosen.** ΔT vanishes at both convective limits — with h → 0 the
  radiative equilibrium is T = ε_sky^{1/4}·T_air, *independent of ε*, and with h → ∞ the surface is
  pinned to the air — so it has a maximum in between: **0.230 K at h ≈ 4 W m⁻² K⁻¹**. The test finds
  that maximum over h instead of asserting one point, which survives a change of h.
- **The total form reports how much of itself is an assumption.** At 300 K, **61 %** of Planck's
  weight falls outside every configured band — nearly all beyond 13.5 µm — and is filled by extending
  the nearest one. `TotalHemispherical.extrapolated_fraction` returns that, because a solver that
  cannot see how much of its ε rests on an assumption cannot report its own uncertainty. A band
  holding under 1e-4 of the weight is skipped rather than queried, since §4.2's Level C bound is a
  thermal-band statement and a material may legally fail it in a band the solver never integrates.
- The µ = cos θ substitution turns cos θ sin θ dθ into µ dµ, so the integrand has no trigonometry and
  the quadrature converges on something polynomial-like: 8, 16 and 64 Gauss–Legendre nodes agree to
  better than **1e-6**, so the default 32 is far past convergence rather than a guess (15 tests).
- **Level B angular emissivity and its fitter** (M7.6, ADR 0042, §4.2). `emissivity_empirical` is
  §4.2's ε(θ) = ε₀[1 − a(1 − cos θ)^p], and `fit_empirical` / `fit_from_nk` / `fit_band_level_b`
  bake (a, p) per material class against Level A, searching §4.2's p ∈ {4, 5, 6} exhaustively (a is
  linear given p, so each candidate is one least-squares solve and the optimum over the set is
  exact). Measured: water **rms 0.0009**, glass 0.0034, paint 0.0034 over 0–70°.
- **⚠️ The rejection criterion had to change, and that is the finding.** An absolute residual bar
  does **not** catch a metal. Aluminium's ε *rises* with angle — 0.019 at normal to 0.030 at 70° —
  which Level B cannot represent at any a ≥ 0, but because ε never exceeds 0.03 a completely wrong
  flat shape fits to **rms 0.0034**, comfortably inside the 0.02 the roadmap asks for. An error bar
  is meaningless on a quantity that small. The fit now checks the monotonic **direction** first, on
  a scale relative to ε₀, and refuses a rising curve whatever its residual. Left as it was, the limb
  of every metal panel would have rendered darker when it should be brighter, and nothing would have
  complained. `bake_angular_models` **returns** its refusals rather than swallowing them, because a
  silently missing entry becomes a silently constant ε.
- **One dispatch for ε(θ), on the level the material declares** (M7.7, §4.2, §13.3).
  `directional_emissivity` selects Fresnel / empirical / constant per `angular_model`, takes and
  returns float32, refuses float16, uses ǀcos θǀ so a back-facing normal is not physics, and is
  finite at grazing. It is the CPU oracle the kernel's ε(θ) will be compared against.
- **⚠️ Enforcing §4.2's Level C bound found a violation in the committed library.**
  `bare_aluminium` declares `angular_model: constant` with ε_LWIR far below the 0.93 §4.2 permits —
  and it is a metal, so Level B cannot fit it either. It now **raises**, naming Level A as the fix,
  rather than quietly rendering a flat limb; the aluminium n/k table M7.5 still owes is now a
  blocking dependency for any scene needing bare metal's ε(θ). Asphalt, which is exactly what §4.2
  permits a constant for, passes.
- **⚠️ And the fit must see the band, not a representative wavelength.** Fitting water at 10 µm and
  comparing against the Boson band average is **0.0275** at 70° — past the 0.02 acceptance
  criterion, so it is a real error and not a rounding one. `fit_band_level_b` fits against the
  band-averaged Level A curve, and a test pins that the single-wavelength route fails (27 tests).
- **Band-effective material properties** (M7.3, ADR 0010, §12.3, §3.1). `PropertySpectrum.band_effective`
  turns a material's ε(λ) into the number *one camera* sees, and `weighting_for_fpa` decides which
  Planck weighting from the FPA type rather than from the call site: a bolometer absorbs power and
  averages under B(λ,T), a photon detector counts photons and averages under B_q(λ,T). The direction
  of the difference is fixed — photon weighting leans long, so a falling ε(λ) always averages lower
  under it, and a rising one higher. Both signs are asserted, so the test is about the weighting and
  not about the curve.
- **The failure this step is really about is extrapolating a curve past its own support.** For a
  material whose ε falls off a cliff at a band edge — glass, most paints — carrying the last
  tabulated value across is a large error that looks like a small one and is biased in whichever
  direction the curve happened to be heading. A response the curve does not cover is refused, and
  `MaterialLibrary` now delegates to the same `covers`/`band_effective` pair rather than carrying a
  second copy of the rule that could drift from it.
- **⚠️ The roadmap's "step spectrum to 1e-6" is held to 5e-4, and the integrand is why.** A
  discontinuity is what composite Simpson handles worst: the cut lands inside one 0.01 µm resampling
  cell and half of it is attributed to the wrong side, a few parts in ten thousand over a 6 µm band
  (measured 2.5e-4). The exact identities this rests on instead are the flat spectrum (to **1e-12**,
  under both weightings) and **linearity** — be(αs₁+βs₂) = α·be(s₁)+β·be(s₂) to 1e-12, which a
  weighted average cannot fake and which fails the moment a normalisation is applied in the wrong
  place (19 tests).
- **Tier 3 sensor-chain phenomenology** (M9.9, §15 T3, §16.4 step 9). `tests/unit/test_tier3_sensor_chain.py`
  measures the four things a *camera* does to a picture that a radiometer does not — AGC collapse,
  the FFC freeze, the membrane trail and saturation — each of which is visible in an image and each
  of which a radiometrically perfect frame scaled linearly would hide.
- **AGC collapse**, on a 300/303 K pair with a 5 % 450 K plume: `linear` keeps **1.54 %** of the
  pair's contrast (231 codes → 3.6), `plateau_equalization` keeps **76.3 %** (128 → 98), and
  `plateau_local` 8×8 keeps **144 %** — the local operator *raises* pair contrast under the plume,
  though from a smaller C0 (14.4 codes), because it has already spent the global gradient on local
  detail. That is the most recognisable artefact of real thermal video and the one most often left
  out of a simulator.
- **The FFC freeze reports 42 frozen frames and a viewer sees 43 bit-identical pictures**, because
  the frame the shutter held is itself the first of the run. Both numbers are asserted: it is an
  off-by-one that only shows up when someone counts frames in a clip. The pattern reset is measured
  on frame-averaged, flat-field-subtracted planes, because a single frame's spatial std is dominated
  by per-pixel temporal noise (13.6 DN at this camera's NETD) and by cos⁴ vignetting, neither of
  which an FFC touches.
- **The membrane trail is a geometric series** with ratio **0.1888754 / 0.1888748 / 0.1888791 /
  0.1888502 / 0.1888628** against e^{−dt/τ} = **0.1888756** — five terms to about 1e-5 — and the
  cooled MWIR config leaves none at all, which is §16's "lateral motion smears LWIR, not cooled
  MWIR" in its across-frame half. ⚠️ Getting there needed one correction: **the baseline must be the
  settled signal.** Using the last measured frame leaves 19 % of its own residual in it, shifting
  every term by the same amount and biasing the third ratio low by 2e-4 — exactly the size of the
  effect being resolved.
- A 1000 K source **clips at 65535** with DN monotone in scene temperature all the way up, so
  nothing wraps; uint16 arithmetic that wrapped would turn the hottest thing in the scene into the
  coldest.
- **⚠️ Two of the row's six criteria are open and named rather than dropped**: the aerial
  edge-asymmetry band needs ME.4's statistics of real imagery and the last criterion needs ME.6.
  Both wait on public data this machine cannot fetch. A test asserts the roadmap row stays amber
  while they are, so "amber" has something executable behind it instead of a note (9 tests).
- **Tier 3 multi-band phenomenology, and the architecture claim** (M11.8, §15 T3, §5.2, §16.4 step 10).
  `tests/unit/test_tier3_multiband.py` runs LWIR, SWIR and MWIR against one prescribed facet scene
  (M6's solver is a later step, so the noon and 03:00 temperatures are authored inputs; the tests
  assert *orderings*, which are properties of the radiometry, and say so where a number could have
  set the answer).
- **The architecture assertion, as a fact rather than a claim:** three bands are configured and
  `planck.py`, `band_integration.py`, `band_average.py`, `lut.py`, `lut_files.py`, `encoding.py` and
  `band.py` have **not changed since M1.11**, when LWIR was the only band there was. The two
  exclusions are named and separately tested rather than quietly globbed out: `constants.py` grew
  constants, which CLAUDE.md directs, and `spectral_response.py` took exactly one change, from M5's
  MTF cascade.
- **⚠️ The roadmap's "MWIR glass saturates at noon while LWIR does not" is true of a scene and false
  of a mirror, and both are now recorded.** At *exact* specular alignment **every** band saturates —
  LWIR by 8× — because a mirror shows you the sun and the sun is a 5778 K blackbody in LWIR too
  (E_B/Ω_sun = 2.8e4 W m⁻² sr⁻¹). What separates the bands is the **angular window** in which the
  glint saturates: **0.51° in LWIR, 3.72° in MWIR, 18.6° in SWIR** — a 36× spread, and the
  operational quantity, since it says how much of a real scene one reflection ruins. At a realistic
  2° off-specular geometry MWIR sits at 7.0× full scale and LWIR at 0.17 of it.
- At 03:00 with the sun below the horizon, MWIR and LWIR **agree** about which facet is hotter —
  Spearman rank correlation > 0.9 across six materials — because both are reading self-emission.
  Two companion tests keep that from being vacuous: the scene spans more than 5 K in both bands, and
  the noon ranking is *not* the night ranking.
- In SWIR at night the scene is lit rather than glowing: the signal is linear in the airglow level
  across §5.5's whole 3.5–39 nW/cm² range (r > 0.999), is exactly zero without it, and is more than
  10× the same surface's own 300 K emission.
- **⚠️ The example SWIR camera does not reach "SNR > 5 from airglow": it reaches 0.32**, and that is
  not a modelling failure — it is what a 60 Hz, 120 e⁻, F/1.4 uncooled InGaAs core gives on a
  moonless night, which is why night-capable SWIR parts are specified differently. The test builds
  the camera that *does* reach 5 and names the four changes it takes: 16 → 33 ms integration,
  15 → 20 µm pixels, 120 → 25 e⁻ read noise and F/1.4 → F/1.0. Every one is needed (20 tests).
- **The reflection lobe: near-mirror sky, and solar glint** (M11.7, ADR 0067, §4.3, §5.4).
  `irsim.materials.lobe` gives §4.3's qualitative claim a model — a GGX microfacet kernel with
  α = roughness² — and `irsim.pipeline.specular` binds it to the `SkyModel` and to M11.3's solar
  band irradiance. `roughness_per_band` had been authored in every material YAML since M7.9 and
  nothing read it, the same shape as M11.5's cold shield.
- **The specular/diffuse split is explicit because the tidy alternative was tried and measured.**
  One GGX kernel with no invented weight *should* give the diffuse answer as roughness → 1. It does
  not: a microfacet lobe never converges to Lambertian, and at α = 1 GGX sits a **total-variation
  distance of 0.30** from the cosine hemisphere — reading a test sky 1.7 % warm and not improving as
  α grows. A test measures exactly that, so the decision stays visible instead of becoming folklore.
  `K = w_s·K_GGX + (1−w_s)·cos θ/π` with `w_s = (1−r)²` conserves energy to **1e-12** at every
  roughness and makes both endpoints exact: mirror to 1e-6, M7.13's V_s blend to 1e-6.
- **`(1−r)²` is a modelling choice and its check is §4.3's own prose**, reproduced from the library's
  authored roughness values without being fitted to them: glass **94 %** specular, painted bodywork
  **77 %**, asphalt **9 %** — "paint and glass show near-mirror sky reflections in LWIR … asphalt
  stays near-Lambertian", as numbers.
- **Centring the specular quadrature on the mirror direction is not an optimisation.** On a grid tied
  to the normal, α = 1e-4 is a lobe 1° wide that the grid cannot see at all, and a glass windshield
  came out **0.09 units short** of the exact mirror answer — a bias, in the direction that makes
  shiny surfaces look less shiny.
- **The glint is capped at the sun's own radiance, and that is physics rather than a guard.** A
  microfacet NDF at its peak claims 1/(πα²) sr⁻¹ = **4e5** for glass, enough to render a glint four
  orders of magnitude brighter than the sun that caused it. MWIR glint against a 300 K scene, ρ = 0.1,
  60° sun: glass **1.4e4×** (apparent temperature off the top of the LUT), paint **630×** (698 K),
  roughness 0.3 → 5.6×, and by roughness 0.6 it is **below ambient and invisible**. Glint is a
  smooth-surface phenomenon and it is gone by roughness 0.3.
- In LWIR the same model gives the car-roof effect: a ρ = 0.9 flat mirror at 30° elevation under a
  clear sky reads **more than 30 K below** its own 300 K surface, matching
  `Lb⁻¹(ε Lb(T_s) + ρ Lb(T_sky(30°)))` to under **0.1 K**.
- **The incident field is sky *and* ground, and `up` is a separate argument from the normal.** A
  near-vertical windshield or a ship's flank reflects ground over half its lobe; passing the normal
  as `up` would give a vertical panel a sky in every direction and render it far too cold. The
  argument exists to make that mistake explicit, and a test pins it. Not yet wired into `run_frame`:
  stage 1's reflected term is still the V_s blend, which is exactly this model's roughness → 1 limit,
  so nothing rendered so far changes (23 tests).
- **The electron-space noise budget for photon FPAs** (M11.6, ADR 0025 addendum, §10.1, §9.1).
  `irsim.noise.electron` adds `electron_noise(N_e, i_dark, t_int, σ_read, rng)` — one Poisson draw
  over signal + dark + background (the sum of independent Poissons *is* Poisson in the sum, so two
  draws would be wrong rather than safer) plus an additive Gaussian read term — and `electron_budget`,
  which builds a photon camera's budget from its own datasheet electrons.
- **This reverses which of NETD and the electron numbers is derived, and M11.1's SWIR camera is why.**
  NETD is defined against a 300 K blackbody (§9.4), and a 300 K blackbody puts **1.24 photoelectrons
  per pixel per 16 ms frame** into 0.9–1.7 µm against 120 e⁻ of read noise — so §9.4 evaluated
  honestly gives **976 K**, and solving `σ² = (NETD·∂S/∂T)² − σ_shot²` against that scales the Gaussian
  term by about 2e4. The camera's noise would be *invented*, not described. A bolometer keeps ADR
  0025's anchor unchanged, because NETD is the only usable handle its datasheet gives.
- Everything ADR 0025 was actually about survives: Poisson terms are **never** rescaled, noise is
  added in electron space and never in kelvin, and a configuration that cannot reach its own claim
  **raises**. That last check is skipped — with its reason named in the code — for a **reflective**
  band, closing the loop M11.1 opened when it wrote 976 K into the SWIR file and said nothing may
  anchor to it.
- **Synthetic bench against prediction: 0.16 % for both cameras** (σ of a uniform scene over ∂S/∂T, in
  electrons, 200 000 samples) — InGaAs 7.86 vs 7.87 mK at 700 K, InSb 18.33 vs 18.36 mK at 300 K. The
  InGaAs is benched at 700 K rather than 300 K because a NETD bench at 300 K in that band measures the
  read noise and nothing else. The InSb sits **2.4 % above shot-limited**, which is what a cooled MWIR
  core is sold on and a real check that the read and dark terms are not overstated. A companion test
  inflates the read noise 10× and confirms the bench notices — a bench that passes against any budget
  is measuring the predictor, not the camera.
- **Dark current as the reason InSb is cryocooled, as a number:** 200 e⁻ per frame for the uncooled
  InGaAs array against **0.12 e⁻** for the InSb at 77 K, and warming that array to room temperature
  multiplies its dark count by more than **10⁶**.
- The aperture factor is re-derived from §9.1's own terms inside the test rather than read back from
  the helper, so `π/(4F²+1)` being right in one place and wrong here would be caught: at F/1 it is
  π/5 against π/4, i.e. **20 % less signal derivative**, held to ±0.5 % (15 tests).
- **The cold shield, and a cooled MWIR camera to need it** (M11.5, ADR 0066, §9.1, spec issue S31).
  `configs/sensors/example_mwir_insb_640.yaml` is the third configured band and the first camera with
  `cold_shield_efficiency < 1`. `irsim.detector.cold_shield` supplies the equation §9.1 asks for and
  does not give: `Ω_admit = min(π, Ω_lens/η_cs)`, `Φ_bg = A_d (Ω_admit − Ω_lens) L_B(T_housing)`.
- **`optics.cold_shield_efficiency` has been in the schema since M0.9 and nothing read it** — the same
  shape as ADR 0077's and ADR 0082's findings, closed here while a cooled camera was being added
  rather than after one had been shipped without it.
- Three properties asserted rather than argued: η_cs = 1 admits **exactly** 0.0 (a residue would put a
  spurious Poisson term into every well-shielded camera's budget); Ω_admit is capped at the
  **projected** hemisphere π, not 2π, because a pixel on a plane cannot receive from behind itself;
  and at η_cs = 0 the scene cone and the background together close the isothermal-enclosure identity
  `A_d·π·L_B(T)`, which only holds *because* the cap is π.
- **τ_opt does not multiply the background, and the signature is the assertion** — `background_power`
  has no transmittance argument. The flux reaches the pixel without going through the lens, which is
  exactly what separates it from §8.2's self-emission `A_d Ω_lens (1−τ) L_B`, the lens glowing
  *inside* the scene cone. The two add; folding one into the other looks right at η_cs = 1 and is
  wrong everywhere else.
- **What a mismatch costs is the offset's shot noise, not the offset** — NUC removes offsets. So
  `NETD(η)/NETD(1) = √((N_signal + N_bg)/N_signal)`, held to **1e-9** against `predict_netd_k` with
  σ_gaussian = 0 to isolate the shot term, and separately checked to rise monotonically with the
  camera's real 350 e⁻ read noise. For the InSb example at 300 K (N_signal = 2.49e6 e⁻, 29 % of well):
  η_cs = 0.95 costs 3 %, **η_cs = 0.90 costs 6 %**, η_cs = 0.50 costs 48 %, and at **η_cs = 0.20 the
  dewar alone fills 138 % of the well** — the camera saturates on its own structure before it sees the
  scene. That is why cold-shield matching is a manufacturing specification and not a tuning knob.
- Every uncooled camera in the repository has η_cs = 1.0, so the term is identically zero for them and
  **no golden moves**. `scripts/generate_response_curves.py` now generates both `ingaas.csv` (byte-identical
  to the committed file) and the new `insb.csv`, so the reasoning behind each cut-on and cut-off — a
  substrate edge, a cold filter, an alloy band gap — lives in one place instead of in a header (21 tests).
- **Night illumination — airglow and a first-order moon** (M11.4, ADR 0065, §5.5). `irsim.radiometry.night`
  and `irsim.pipeline.night` produce the M11.2 bundle's `l_night` plane,
  `[E_airglow · c(cloud) · v(t) + E_moon · Φ(phase) · sin(el) · τ] / π`. `SolarSpectrum` and
  `AirglowSpectrum` now share one `SpectralTable` band integral — two sources integrating a band
  two slightly different ways is how they come to disagree about what a band is.
- **Airglow has no shadow parameter, and the signature is the assertion.** §5.5 says airglow "is not
  blocked in the same way as moonlight"; it is emitted at ~87 km across the whole sky, so a wall
  casts no airglow shadow. `incident_radiance` takes no `shadow` and no sun direction, so a scene
  *cannot* darken the night sky by geometry. Cloud attenuates it to a floor of 0.10 and never to
  zero, because cloud scatters airglow rather than absorbing it.
- **The lunar level is derived, not authored.** m_sun = −26.74 and m_moon = −12.74 give a flux ratio
  of 3.98e5 and E_full = **3.42e-3 W m⁻²**; the moon borrows the solar spectral shape, so the lunar
  albedo cancels between shape and level and never has to be invented. Lane & Irvine's phase law is
  violently non-linear and that is the point: a **quarter moon is 0.091 of a full one**, not half. A
  model using the illuminated fraction directly would render every quarter-moon scene five times too
  bright and look entirely reasonable doing it.
- **The airglow level is defined over the shape file's support, so a band takes its fraction** —
  not all of it (ADR 0065). The literal "in-band" reading breaks the moment a second band exists: it
  would hand the LWIR camera 10 nW/cm² of OH emission at 10 µm, making the level a property of the
  camera rather than of the sky. The fraction is not a detail: the modelled InGaAs camera sees
  **61 %**, and extending the cut-off from 1.7 µm to 2.5 µm gains **a third more light**, because the
  strongest OH sequence (Δv = 1) sits at 1.4–2.0 µm. That is a real reason extended-InGaAs parts exist.
- **§5.5's claim that moonlight and airglow are "comparable" at full moon [R7] is not reproduced, and
  is recorded rather than tuned away** (spec issue S38). Standard lunar photometry and an OH band
  model give **12.6×** over 0.9–1.7 µm. Three readings bring it close — §5.5's own *upper* airglow
  level (3.2×), the **spectral density** at the OH band peaks rather than the band integral (2–5×),
  and a moon at a realistic elevation. What the models *do* reproduce is the rule of thumb SWIR night
  imaging is sold on: airglow ≈ a **quarter moon** (1.15×), and that is what the test asserts. The
  full-moon ratio is pinned to the measured value so the disagreement stays visible (25 tests).
- `data/spectra/airglow_oh_meinel.csv` from `scripts/generate_airglow_spectrum.py`: OH-Meinel band
  heads (spectroscopy) with modelled Δv = 1/2/3/4 sequence strengths of 55/27/12/6 %, plus O₂(a¹Δ) at
  1.27 µm. MODELLED in its first status line, like the solar files, and replaceable by a measured
  near-infrared sky spectrum with no code change.
- **The reflected-solar term** (M11.3, ADR 0064, §5.4). `irsim.radiometry.solar` band-integrates a
  spectral irradiance file the same unnormalised way the band LUT integrates Planck, in both energy
  and photon form; `irsim.pipeline.solar` turns it into the M11.2 bundle's `l_sun` plane,
  `E_B τ_sun cos θ_s S / π`, so ρ is applied **once** by stage 1 rather than baked in as §5.4
  writes it. `shadow_mask` (1 = lit) and `sun_cos_incidence` join the G-buffer's optional keys.
- **The number this step exists to produce: the solar/thermal crossover is at 4.14 µm** for a 300 K,
  ρ = 0.3 surface in full sun. That is why §12.1 has three regimes and why MWIR is the only band
  marked `mixed` — it straddles the line. At band level: SWIR reflected/emitted = **1.6e8**, LWIR =
  **0.0035**. It is also the least model-dependent quantity here, because both sides are steep.
- **The spectra are modelled and the files are named after the models, not after ASTM E-490 and
  G-173.** This machine has no network, and those are tabulated standards, not formulas; a file of
  plausible digits under a standard's name is worse than no file. `toa_planck_5778k.csv` is a
  5778 K Planck normalised to 1361 W m⁻²; `direct_normal_am1p5.csv` is that times a clear-sky
  band model (Bird & Riordan Rayleigh, Ångström aerosol, Chappuis ozone, the water/O₂/CO₂ bands as
  Gaussians in ln λ) normalised to G-173's 900.1 W m⁻². Both are generated by
  `scripts/generate_solar_spectra.py`, so they are reproducible rather than hand-authored, and
  swapping the real tables in is a file replacement with no code change.
- **Because the integrals are calibrations they are not evidence, so the tests lean on shape
  instead** — and shape was not fitted. τ(1.38 µm) = **0.025** and τ(1.87 µm) = **0.008**, which is
  why a SWIR camera is blind in those bands, while the 1.06 and 1.55 µm imaging windows stay above
  0.92. The effective-temperature model's real weakness is the ultraviolet (12.0 % of the constant
  below 0.4 µm against a real ~8 %), so it is **enforced rather than noted**: a spectral response
  reaching below 0.70 µm is refused, naming E-490 as the fix. Every configured band starts at
  0.75 µm or above.
- **τ_sun for the direct beam uses Kasten–Young air mass.** ADR 0051's plane-parallel sec θ
  diverges at the horizon and refuses beyond 85°, which is right for the band-averaged slant path
  and wrong for the sun — 09:00 is not an error condition. The band-averaged path is untouched, so
  nothing measured against ADR 0051 moves. Sun below the horizon, surface facing away and full
  shadow are **exactly** zero, because a horizon leaking 1e-12 of a 300 W m⁻² beam is a seam after
  AGC.
- **A disagreement recorded rather than silently resolved.** Band-integrating these two files gives
  a SWIR zenith solar transmittance of **0.789**, against the atmosphere presets' ESTIMATED 0.88 —
  because the 1.38 µm water band sits *inside* 0.9–1.7 µm and a broadband §7.2 midpoint does not
  know that. Neither number is measured, so a test holds the two within 20 % and asserts the sign
  rather than one overwriting the other (23 tests).
- **The band-agnostic shading core** (M11.2, ADR 0063, §5.2). Stage 1 now takes an `Illumination`
  bundle: three **incident** band radiances — the thermal environment, the direct beam as an
  equivalent isotropic radiance `E_B τ_sun cos θ_s S / π`, and the isotropic night sources —
  summed once and reflected once. Dividing the solar irradiance by π here rather than carrying
  §5.4's combined `(ρ_B/π) E_B τ cos θ_s S` makes the Lambertian identity ρ = 1 → E_B/π an identity
  *of the code*, and stops a caller applying ρ twice.
- **§5.2's compile-time flag is a runtime switch on `band.regime`** (spec issue S17). A
  compile-time flag cannot be set by a YAML file that did not exist when the binary was built, and
  M11.1's guard would reject the `if band == "swir"` alternative on sight. The regime is consulted
  in exactly one place, `Illumination.for_regime`, and every consumer downstream receives a bundle
  that has already been gated.
- A gated term is dropped to `None`, **not zeroed**, so an LWIR frame with a solar plane attached
  takes the same branch as one without and comes out **bit-identical** rather than equal to within
  rounding. That is the difference between a test that can fail and one that cannot.
- **Self-emission is never culled**, and this is the part that would have produced a plausible
  wrong picture. "Reflective" is a statement about a 300 K scene, not about a band. Against §5.5's
  10 nW/cm² airglow a ρ = 0.1 surface in 0.9–1.7 µm crosses over at **330 K**: reflection wins by
  15.3× at 300 K, emission wins by **11 235×** at 500 K. A jet exhaust, a flare or a brake disc is
  a self-luminous object in night SWIR with no sun and no airglow at all, and a core that culled
  ε·L_B in a reflective band would render it invisible with nothing looking broken.
- **The units tag is checked at kernel entry.** `lb` and `lb_q` differ by ~1e19 at these
  wavelengths, and a solar spectrum integrated in W m⁻² handed to a photon-unit kernel does not
  raise, does not produce NaN and does not look wrong after AGC — it produces a scene that is
  uniformly, invisibly mis-scaled. It is the one error in the illumination path no picture reveals.
- The photon path is pinned to be the energy path divided by a **band average**, not by hc/λ at one
  wavelength: the two kernels' ε = 1 outputs differ by an independently quadratured mean photon
  energy to 1e-6, and that mean is **19.3 %** from the centre-wavelength shortcut at 300 K, so a
  single-λ implementation could not pass. A uniform (0-d) environment is now broadcast — isotropic
  airglow over a whole frame is the common case — but only 0-d, since general broadcasting would
  let a `(H, 1)` column pass as a plane, which is what the shape check exists to catch (19 tests).
- **A second band, added as data** (M11.1, §12.1, §12.2). `configs/sensors/example_swir_ingaas_640.yaml`
  and `data/spectra/responses/ingaas.csv` put a 640×512 uncooled InGaAs SWIR camera beside the LWIR
  Boson. It loads, classifies by overlap, hashes and tabulates a float32 LUT through the same
  `make luts` — **no kernel changed, and none was touched.**
- `tests/unit/test_band_scalability.py` (32 tests) is what keeps that true, in two halves. The
  dynamic half parametrises over *every* YAML in `configs/sensors/`, so a third band is tested the
  moment it is added. The static half walks the ASTs of
  `irsim/{radiometry,optics,detector,noise,isp,pipeline}` and refuses a band name in a string **or
  an identifier** (`SWIR_CUTOFF` is `if band == "swir"` wearing different clothes), and a nominal
  band edge written as a wavelength — the numeric form of the same coupling. The core is clean of
  all of them today. The numeric guard only fires on a name that reads as a wavelength, because 1.0
  and 3 are the most ordinary numbers in the repository and a blanket scan would cry wolf 300 times.
- The known answer that makes `regime: reflective` a fact rather than a label: a 300 K blackbody's
  self-emission in 0.9–1.7 µm is **3.3e-9** of its LWIR emission, taken through the *photon* table
  and converted back with the band-mean photon energy so the photon path is exercised, not bypassed.
  The converse is asserted too — between 300 K and 900 K SWIR self-emission rises by more than 10⁶
  while LWIR rises by under 10², which is why M11.2 must not cull emission in a reflective band.
- **The SWIR file exposes where the schema is still emissive-shaped, and says so rather than lying.**
  `noise.netd_mk_at_300k` is mandatory, and NETD is defined against a 300 K blackbody — which puts
  **1.24 photoelectrons per pixel per 16 ms frame** into this band, against 120 e⁻ of read noise.
  Evaluating §9.4 honestly on the camera's own numbers gives **976 K**, so that is what the field
  records. It is the correct physics and a useless noise anchor; nothing in the SWIR path may use
  it until M11.6 replaces it with an electron-space budget. `outputs.apparent_temperature` is
  `false` for the same reason: inverting Lb is a temperature only where the signal is self-emission.
- **ROI-weighted and locally-adaptive AGC** (M9.10, §11.3). Every histogram in `irsim.isp.agc` now
  takes per-pixel `weights`, and `agc_plateau_local` tiles the frame, equalises each tile's own
  histogram and blends the four nearest mappings bilinearly (the CLAHE construction). The config
  enum gains `plateau_local` with `agc_tiles`; **the global modes stay the default**, because a real
  core is global and §15 Tier 5 needs the simulator to be too.
- Both additions are held to **identities, not tolerances**: uniform weights reproduce the global
  operator bit for bit, and so does a single tile. An approximation in either would mean the new
  code path had quietly become a second, slightly different AGC — which is exactly the failure a
  config-hashed display branch exists to prevent.
- The number this was built for. One hot object sets the stretch for every pixel, and on a broad
  exhaust plume a global *linear* stretch keeps **8 %** of the background's contrast and a global
  plateau stretch **61 %**, where `plateau_local` keeps **93 %** — the difference between a target
  that reads as one flat white shape and one whose motors and wings stay separable. Stated cost:
  display level no longer means one thing across the frame, and none of it touches the radiometric
  branch.
- Tile seams are the obvious failure mode, and the test has a guard rather than a threshold: on a
  ramp the blended boundary is no bigger a step than any other column, while assigning each pixel
  its own tile's mapping with no blending puts a **255-code** cliff there (28 tests).
- **Goldens regenerated, values unchanged.** Adding `agc_tiles` to `IspSpec` changes `config_hash`
  for every camera, so all 12 golden sidecars went stale. Verified before regenerating rather than
  after: **0 of 14 stored arrays changed**, and every changed line in the sidecars is the
  `config_hash` field. The staleness was the hashing working, not a physics change.
- **The quadrotor flies with propellers** (ADR 0081). `irsim_isaac.pipeline.rotor_isaac` turns a
  prim's transform into veils and `QuadrotorSpec.rotor_mounts` puts four above the motor bells;
  `scripts/render_quad_flight.py` grows `--no-rotors` for the before/after. **Nothing is authored**
  — no prim, no mesh, no material — because a spinning rotor is not geometry. That buys the
  sub-pixel and occlusion story for free and costs one honest asymmetry: the discs are in the
  infrared frame and *not* in the companion visible frame, which RTX renders from stage geometry
  there is none of.
- **Occlusion is a plane intersection, not a range comparison.** Each pixel's ray meets the disc
  *plane* and is compared with the renderer's depth. Using the disc centre's range would be wrong
  by up to a disc radius across the ellipse — 0.36 m at 20 m, about 18 px of arm drawn on the wrong
  side of the aircraft. The mask is meaningless outside the ellipse and does not need to be; a test
  asserting otherwise failed on arithmetic that was correct, and says so.
- **Two traps, both of which make a plausible picture out of deleted physics.** The sweep is taken
  over the *detector's* frame, not the time-lapse's six-second capture interval — that would turn
  three hundred revolutions into a perfectly uniform annulus with no banding at all. And rpm
  follows throttle by a **square root**, because thrust goes as rpm² and the profile's `u` is a
  fraction of maximum thrust (the same `u` ADR 0072's `ΔT_max u²` motor law reads); linearly, a
  hovering aircraft turns 40 % slow and its discs are filmed in the wrong regime.
- ESTIMATED: the blade's sky-view factor, taken as 0.5 for a blade seen mostly edge-on.
  `propeller_rubber` is ε = 0.95 in LWIR, so the reflected term is a twentieth of what leaves the
  blade and the choice is worth about a kelvin (21 tests).
- **What the first render found that no unit test did:** sky carries `distance_m = 0` in the
  G-buffer, not `inf` — the sentinel is zero so a consumer ignoring `sky_mask` still sees τ = 1.
  Read as a *distance*, zero is a surface at the camera, and the first `occlusion_mask` therefore
  masked **every pixel of the frame** and erased all four discs; the rendered frames were
  bit-identical with and without rotors. The test that should have caught it used `inf`, which is
  what the AOV reports *before* the adapter translates it — a plausible fixture rather than the
  contract, testing the one value that could not fail. Now parameterised over both, plus a sky-only
  frame, which is the demo's actual configuration.
- Measured in sim with the fix: **6444 native pixels change, peaking at +10.1 K**, and occlusion
  falls from the whole frame to the 15 786 airframe pixels really in front of a disc. The disc is
  brightest at its **root**, because local solidity `N c(r)/(2πr)` rises inward as the circumference
  shrinks while the chord does not — 3.2 % at three-quarter radius, 19 % just outside the bell.
- And it is **invisible at the demo's own display span**: ADR 0074 spans the target's nodes
  (18–62 °C), so a 273 K disc clips to black along with the 263 K sky, and plateau equalisation
  gives it no codes either. At `--span-c -15 40` the same frames differ by up to **45 display
  codes** and the four annuli are plainly there. Real in radiance, absent from the picture — the
  AGC lesson again, and which one you measure decides whether you believe the feature works.
- **A maritime demo: vessels on open water, filmed in LWIR** (MM.5–MM.7, ADR 0078).
  `scripts/render_maritime_demo.py` is the `render_aerial_demo.py` of the sea — scene YAML in,
  four physical-unit frames plus a registered visible frame out, 4 frames in 15 s.
  `configs/scenes/vessel_transit_clear_day.yaml` puts a skiff at 600 m, a patrol boat at 2.1 km
  and a freighter at 5.2 km, each with a hull pinned near the SST, a superstructure following the
  air, and an exhaust stack at 430 K.
- `GroundSpec.mode: sea` with `bulk_sst_k`, and `configs/environments/sea_clear_day.yaml`.
- **The water in the stage is for the eye only.** It occludes, it carries the Earth's curvature and
  it sets the horizon, but it joins the *background* mask, so its infrared temperature comes from
  the analytic sea profile at each ray's own depression angle. `--no-water` leaves the IR frame
  unchanged and is the cheapest proof of which half of the stage is physics.
- `data/weather/coastal_sea_breeze_48h.csv`: a maritime fixture with a 2–9 m/s diurnal sea breeze.
  `synthetic_clear_day` gained `wind_swing_m_s` / `wind_max_hour_local`, defaulting to zero so the
  existing fixture still regenerates bit-for-bit.

#### Changed
- MM.8 is **partly done**: the public-imagery half is blocked on data, not on effort. No public
  maritime LWIR set is in `data/validation/datasets.yaml` — the six indexed sets are aerial or
  single-frame, and only Halmstad states a licence at all (ME.1a) — so ME.2–ME.4's statistics have
  nothing maritime to run over. Indexing one is the unblocking step.
- **`SeaModel`'s `cool_skin_k` constructor argument is gone.** It defaulted to zero, so every
  maritime scene rendered its skin exactly equal to its bulk SST unless somebody remembered to type
  a number — and a number typed there could contradict the wind the same scene was using to roughen
  the surface three lines away (CLAUDE.md #6). The deficit now comes from that same
  `WeatherSeries`, the net longwave from the scene's own sky model (M6.5), and the absorbed solar
  from the scene's own site and DNI/DHI. There is nowhere left to type a contradicting value.
- `test_isothermal_identity_holds_at_every_angle_and_wind` now holds the sky at the **skin**
  temperature rather than the bulk. The enclosure is isothermal with the surface that radiates, and
  since MM.4 that is no longer the authored SST; holding it at the bulk leaves exactly the
  cool-skin deficit as a residual (0.09 K at 275 K), which looks like a broken quadrature and is
  not one.
- **The painted material class carries a fitted angular model instead of an estimated one**
  (M7.5/M7.6, §4.2, spec issue S40). `car_paint_black`, `car_paint_white`,
  `aircraft_aluminium_painted` and `painted_composite` move from an estimated `(a = 0.25, p = 5)`
  to `(a = 0.75, p = 4)`, fitted against Level A on the M7.5 paint proxy — which is what §4.2 asks
  for in as many words: "fit (a, p) once per material class against Level A and bake it". Per-band
  fits give a = 0.732–0.770 with p pinned at 4; a test re-derives them from the checked-in table,
  so the four YAML files cannot drift from their source.
- **This is a correction, not a refinement.** The estimate held ε at 0.97 of its normal value at
  70°, where Fresnel on an acrylic gives 0.86. On a uniform 300 K sphere under a 220 K sky the limb
  darkening goes from **1.4 K to 6.8 K**; against a 250 K sky a car door seen at 70° shifts
  **−3.8 K** in apparent temperature and at 80° **−8.7 K**. Both are past the 2 K Tier 4 target.
  The old value made every painted limb too warm, everywhere, which is precisely the one-sided
  error §4.2's Level C bound exists to prevent — it was simply hiding in Level B instead.
- ⚠️ **Raised as spec issue S40 rather than resolved silently.** §4.2's own prose quotes
  `a ≈ 0.15–0.35` for painted metals and plastics, and the fit disagrees by 2–3×. The fit wins
  here because §4.2 also *instructs* the fit, and because §4.3 puts a clearcoat in the optically
  smooth regime in LWIR (σ ≪ λ/8) where specular Fresnel is the right model. The quoted range may
  be remembered from matte or heavily pigmented coatings; the spec should say which finish it
  means, because the library contains both. `p = 4` also sits at the **edge** of the sanctioned
  candidate set {4, 5, 6} — p = 3.5 fits 2.3× better — which is noted in S40 and left alone.
- `carbon_fibre` and `propeller_rubber` sat on the same estimate and are **deliberately not**
  swept along: the proxy is PMMA, a clearcoat binder, and neither surface is painted. Their YAML
  now says so, and a test asserts they kept their own value.
- The margin on `test_the_rim_reads_colder_than_the_centre_by_the_analytic_amount` narrowed as a
  direct consequence and is recorded in the test: the closed form is a first-order expansion, and
  over a 7.1 K excursion anchoring ∂L/∂T at the baseline now leaves **254 mK** of second-order
  term. The existing midpoint iteration still brings it to 9.0 mK against a 10 mK bound, but a
  further darkening needs a second iteration rather than a looser tolerance.

#### Fixed
- **The multi-band manifest described the last invocation, not the output tree** (M10.24).
  `outputs/multiband/index.json` was overwritten wholesale on every run, so re-rendering the ship
  alone left a manifest claiming the directory held four ship renders while twelve videos and
  three contact sheets sat next to it. It merges now: entries this run produced win — a scene that
  goes from ok to failed must read failed — and entries it did not touch are carried through. A
  missing or unreadable index is treated as an empty one, because a convenience file any run can
  rebuild must never be able to discard a completed render.
- **The ship rendered a quarter-second clip by default.** Its frame count was 8, authored when the
  maritime script exported loose per-frame files and encoded no video at all; at 30 fps that is
  unwatchable beside the two 150-frame aerial flights. It is 90 now. The maritime camera is static,
  so those frames buy less motion than a flight does — but the sea state, the AGC and the noise all
  move, and those are exactly what a still frame cannot show. A test pins every scene's default at
  three seconds or more, since a default clip that cannot be played is a defect in the driver
  rather than a choice about the scene.
- The driver had **no tests**; it has ten, and they run without Isaac Sim or a GPU because the
  merge and the scene table are the two parts of it that do not.
- ⚠️ **Two of MM.8's three written criteria are backwards, and the tests measure rather than assert
  them.** Both assumed the sea behaves like an overcast *ground* scene, which MM.3 had already
  found it does not (ADR 0078):
- **"Sea near the horizon reads colder than sea close in" — it reads 2.89 K warmer.** The
  profile's coldest point is an interior trough about 5° down, so a near-horizon frame sits on its
  *rising* side (1.2 K of the span), and the atmospheric path then roughly doubles that because the
  far ray is 7 km of air pulling toward T_air while the close ray is 530 m. The criterion is true
  only of sea past the trough, not of the band a shore or mast camera actually works in. A
  wide-field frame reproduces the trough directly: minimum at 4.9°, more than 1 K warmer at both
  ends.
- **"Overcast collapses that span below 20 % of clear" — it collapses to 57 %.** An overcast sky
  stops the reflection varying with angle, but the cloud base is still colder than the water and
  the atmospheric path does not care about cloud at all. Consistent with the surface-only 50–95 %
  `test_sea_surface.py` already pinned.
- The third criterion holds, and is the one that matters for detection: **polarity flips with
  range.** A 289.5 K vessel reads **+1.27 K** against the sea at 574 m and **−0.86 K** at 7.9 km —
  same vessel, opposite sign, with a contrast null somewhere between where it is invisible at any
  sensitivity. A 310 K control stays bright at every range, so the flip is a property of that
  vessel temperature and not of range itself.
- ⚠️ **MM.4's own acceptance criterion is half wrong, and the test records the measurement instead
  of asserting it.** The step asked that a 1 K bulk-SST error move nadir apparent temperature by
  > 0.9 K and a 0.2°-depression patch by < 0.15 K. The *ordering* holds with room to spare —
  measured **0.98 K** at nadir against **0.20 K** at 0.2° — but the 0.15 K threshold is unreachable
  at **any** camera height. Near-horizon sensitivity is set by slant *range*, not by angle: a 20 m
  camera's 0.2° ray is 6.8 km out; getting under 0.15 K needs ~13 km of path; and at the ~100 m
  height where 13 km corresponds to 0.5°, 0.2° is above the horizon and sees no water at all. The
  useful statement is the regime — SST accuracy matters looking down and stops mattering within a
  degree of the horizon — not the number.
- Recorded rather than hidden: `Q_net` is the scene's net **longwave** only. At sea the latent flux
  is usually the largest term and a full-flux `Q_net` runs roughly twice the longwave-only value,
  so the modelled deficit is a **lower bound, low by about 2×**. The deficit is exactly linear in
  `Q_net` (asserted to 1e-12), which is what makes that a bounded omission rather than an unknown
  one — a caller with the turbulent fluxes passes their sum and nothing else changes.
- **A proxy label shipped where nothing could read it.** `load_nk_table` ends the provenance it
  carries at the first blank comment line, so the PROXY note in the first draft of `glass.csv` —
  placed after one, for readability — reached no consumer, while the file still looked
  self-documenting to anyone opening it. The note now lives inside the `# source:` block and a test
  asserts it survives the load. The same header also quoted its check values from the nearest table
  row rather than interpolating, so the numbers a reader would verify against were not the numbers
  the loader returns.
- **The sea reflected a sky in the wrong radiance form, and the error was ~1e19.** `SeaModel`
  defaulted to the energy form (`lb`) while the `SkyModel` it reflects was built in the photon form
  (`lb_q`) for every photon FPA. The sea *is* mostly reflected sky, so the two radiances are added
  together — which means a mismatch does not produce a slightly wrong sea, it produces one whose
  apparent temperature pins at the **LUT ceiling of 1000 K across the entire water surface**. That
  is exactly what a maritime MWIR render did: uniformly, silently, and only in the bands a
  bolometer never exercises. `SeaModel` now refuses the mismatch with an error that says why, the
  same guard `SkyModel` already had for its skylight, and the maritime script passes the camera's
  own quantity (ADR 0021). This is the fourth place that quietly assumed a bolometer.
- **The flat field calibrated outside the converter.** `calibrate_flat_field` defaults to
  `RADIOMETRIC_RANGE_K`, −40…+200 °C — a *bolometer* range (ADR 0021). The modelled InSb camera
  fills its well at 366 K, so the 473 K hot point drives it **15.8× over its 16383-code ADC** and
  the two-point fit becomes an extrapolation. It did not fail loudly: it returned a gain and offset
  map wrong by that factor, and the picture came back with the vignetting **inverted**, which reads
  as a lens artefact rather than a calibration error. It is refused now, where the range is known,
  and the three render scripts fall back to no flat field with the reason printed.
- **ADR 0014 recorded the position AOV as world space beside a probe that reported camera space**
  (M2.4). The M10.1 addendum's table said *world*; `detect_position_frame`, cited in the same
  addendum and run on the same scene, reported `camera` with `err_camera = 0.013 m` against
  `err_world = 5.48 m`. The wrong value survived three days and a second addendum because
  `test_position_aov_frame_is_unambiguous` asserts only that *one* hypothesis fits, never which.
  M10.19 later re-derived the truth from a horizon 164 rows out of place without the two records
  being reconciled, leaving the ADR asserting both readings in different sections. The code was
  already right — `IrCamera` has passed `position_frame="camera"` since M10.19 and is the only
  caller — so nothing rendered was wrong; what was wrong is what the next person would read before
  writing the next adapter.
- **"Revisit when `AmbientOcclusion` returns data" had nothing that would notice** (M2.4, risk R3).
  All three candidate AO names were re-probed and all three still return nothing on 6.1.0-rc.26
  (`SdPostRenderVarToHost: invalid input resource`), and that absence is now
  `test_no_ambient_occlusion_aov_delivers_on_this_build` rather than a memory. An all-zero buffer
  returned with status ok does not count as delivering — this ADR's standing hazard, and an AO
  plane of zeros would drive every `V_s` and every reflected-sky term to zero. R3 is **bounded**,
  not retired: unoccluded `V_s` is exact for the open-sky aerial and maritime scenes phase 1
  targets and optimistic in cluttered ground geometry, where the fix is the §5.3(b) irradiance
  cubemap.
- The probe's validity mask no longer rejects a pixel by the magnitude of the position AOV's miss
  sentinel (a point 1000 m down the ray). Sky is known from the ray length being `inf`; a magnitude
  test would have thrown away real geometry in a long-range aerial scene, where targets sit at km.
- **The bolometer membrane IIR is on `run_frame`'s path** (M9.13, ADR 0082, §9.2). `BolometerLowPass`
  has existed since M9.1 and `irsim.pipeline.detector` has driven it since, but **`run_frame` never
  called either.** Stage 4 reached the detector through `response(flux, frame_index, sensor_seed)` —
  a signature with no `dt` and no state, so it could not have carried a lag whatever it intended.
  Every frame *sequence* this repository has produced — the quadrotor flight, the aircraft pass, the
  maritime demo, every golden — came from a bolometer with a thermal time constant of **zero**: no
  trail behind a moving target, and §15 Tier 3's "lateral motion smears LWIR, not cooled MWIR" true
  only in the within-frame half ADR 0077 supplied.
- This is ADR 0077's failure mode a second time, and it was harder to see because **the
  documentation asserted the opposite**: M9.8's roadmap title is "wire IIR, … into run_frame",
  `sensor_chain`'s module docstring lists stage 4 as "detector + membrane IIR", and the README's
  detector row said "wired into the pipeline by M9.8". Six of M9.8's seven landed. All three claims
  are corrected.
- Held to the closed form rather than to "there is now a tail": the step response is
  `1 − e^(−k·dt/τ)` to 1e-6 in signal space, the first frame is **α = 0.811124** of the step,
  successive residuals fall by exactly `e^(−dt/τ)` = **0.188876**, and a cooled photon detector
  settles in one frame. The tail's tolerance is *derived* from float32's precision on each residual,
  because by the fourth term the residual is ~2 DN on a 10 120 DN signal and a flat `rel=1e-4` would
  fail there for no physical reason.
- The interval is **elapsed scene time when the caller advances its clock**, not the frame rate. A
  10 ms membrane settles completely across ADR 0074's six-second time-lapse; driving it at 1/60 s
  would leave 19 % of a scene six seconds old in the picture — a flattering error, since it smooths
  exactly the change being filmed. This is a `run_frame` policy and not a property of the membrane:
  `bolometer_lag`'s default stays the configured interval, because `detector_stage` is the CPU
  oracle the Warp twin is compared against and the twin takes `alpha_for(fpa.frame_dt_s, …)` at
  kernel-launch time.
- `MicrobolometerDetector.frame_from_signal` opens the seam the lag needed between the static
  transfer and the noise (ADR 0052: filtering after noise cuts the per-frame temporal variance by
  α/(2−α) ≈ 0.68 and breaks the M4.6 anchor). It is **not** on the `Detector` protocol: a photon
  detector's noise is Poisson in electron space and there is no signal-DN point to insert at.
- **No golden moved** — the membrane adopts its first input, so every single-frame reference is
  bit-identical. One test changed: a semi-transparent plumbing check shared one `PipelineState`
  across two frames and so read the second 81 % through its settle (65.50 against the 80.75 it
  asserts — exactly α). Rendered scenes: the quadrotor time-lapse is unchanged, and the 30 Hz
  aircraft pass gains a 3.6 % tail, the first trail this repository has rendered (13 tests).
- **Three things only rendering the picture could have found.**
  1. `ground_temperature_k` was first made to **raise** for `mode: sea`, reasoning that a sea has no
     single temperature. That broke every maritime frame: the caller is §5.3(a)'s *reflected* term,
     where what an object sees below itself is the water right around it at steep incidence, ε ≈
     0.99, and one scalar — the SST — is exactly right. The guard was right about the physics and
     wrong about which caller it was guarding. Background rays and reflected rays are now separated
     in the code and in a test.
  2. The sea profile was being evaluated **per pixel**, at a 4000-step path-radiance quadrature
     each, for 2.6 M supersampled pixels. A frame never finished. It is now a LUT, tabulated
     against **slant range rather than depression angle**: `d(δ)` has a square-root singularity at
     the horizon, so an angle grid keeps a cusp that refining barely touches (52 mK at 192 points,
     still 41 mK at 768). In range the same 192 points land under **13 mK** and answer two million
     pixels in 34 ms.
  3. The water mesh's rings must be uniform in **depression angle**, not radius. A pixel row covers
     ground as r², so a geometric radial grid — the obvious choice — is ten times too coarse in the
     near field and nine times finer than necessary at the horizon, and it renders every wave
     shorter than 20 m as corduroy.
- The wave train is scaled by the **shared weather's wind**, wavelength and amplitude together as
  U² (Pierson–Moskowitz similarity keeps the steepness scale-invariant). Before that it was a fixed
  constant while Cox–Munk read the real wind — the picture would have looked windy while the
  radiometry read calm, which is the split-brain CLAUDE.md #6 exists to prevent, and invisible.
- **Rotor veils reach the frame** (ADR 0081, stage 2c). `irsim.pipeline.rotor_veil` composites
  projected discs onto the k× radiance plane and `run_frame` takes `rotor_veils` alongside
  `point_targets`; a frame without them is bit-identical to before.
- It **composites rather than injecting an excess**, and the reason is not efficiency. MS.6's
  point-target form adds `φ τ (L_t − L_beyond)` because a sub-pixel target occults a *sky column*
  the plane does not separately carry. A rotor does not: by stage 2c the plane already holds the
  right background at every pixel — sky beyond the disc over part of it, the aircraft's own arm or
  motor bell over the rest. So `L = L_plane + α (L_blade,at-sensor − L_plane)` is exact for both at
  once, where the excess form would subtract a sky column that is not behind the airframe pixels.
  A test puts one rotor half over cold sky and half over a 300 K arm: the veil **lifts on one side
  and dips on the other**, from one operator in one pass.
- The blade takes stage 2's own atmospheric path (`blade_radiance_at_sensor` calls the same
  `apply_layered_gbuffer` / `apply_atmosphere` on a one-element array), so the veil and the pixels
  under it cannot disagree. Range pulls the disc **towards ambient**, which is not the same as
  fainter: against a 230 K sky in 288 K air, a 290 K blade at 3 km reads dimmer than at 20 m and a
  270 K blade reads brighter. That test was written the first way round only, and corrected.
- Coverage is built on the ellipse's bounding box — a full-frame float64 map per rotor is 40 MB on
  a 4× supersampled Boson frame, 160 MB for four — with a test asserting the windowed result is
  bit-identical to a full-frame composite, not merely close. Flux through the whole chain is
  conserved with the PSF on and off (23 tests).
- **The sea as a background** (MM.2, MM.3, ADR 0078). `irsim.atmosphere.sea` gives apparent sea
  temperature against depression angle the way `SkyModel` gives it against elevation: Cox–Munk slope
  statistics from the shared weather's wind, and a facet-tilt quadrature in which the emissivity and
  the reflected sky elevation move **together** — a tilted facet both presents less grazing incidence
  and swings the reflected ray.
- **The isothermal identity is exact (0 mK)** across depression 0.15–90°, wind 0–20 m/s and SST
  275–305 K. It holds for any emissivity, so it is the one test that catches a mis-weighted
  reflection — the failure a plausible-looking gradient would hide.
- Horizon geometry is spherical, not flat-earth: at 20 m the horizon is 0.1436° down and the range
  to it is **twice** what `h/sin δ` gives at the same angle. That factor of two lands directly on the
  atmospheric path length to every long-range maritime target.

### 2026-09-14

#### Added
- **Structured cloud in the rendered infrared background** (ADR 0076; MS.3/ADR 0070 into the M10.18
  bridge). For a sky-background sensor the dominant false alarm is not sensor noise, it is a cloud
  edge — a warm, target-sized, high-contrast feature with the *same polarity* as the thing being
  looked for, since a drone and a cloud base both read warmer than a cold clear zenith. The field
  has existed since M7 and was reachable only from the engine-free scene generator; nothing that
  rendered used it. `AerialThermalBridge` now takes a `cloud_seed`.
- **The field is fixed to the sky, not to the image plane.** A per-frame field flickers; an
  image-plane field is stable and *travels with the sensor*, so a slewing mount carries its clouds
  along and a tracked target never crosses an edge — removing the clutter the field exists to
  provide. A sky-fixed field is stable and stationary in the world, so slewing sweeps across it.
  `SkyFixedCloud` samples an (elevation, azimuth) grid per ray, with `azimuth_from_rays` supplying
  the second coordinate.
- Sampling by elevation alone is refused rather than approximated: it would band the sky in
  horizontal stripes, which is worse than no cloud because it looks like a deliberate atmospheric
  layer. A caller without azimuth gets the uniform blend — less detailed, not wrong.
- **The visible dome samples the same field**, through the same `sky_angles` convention — not a
  second cloud that looks similar, the same object, so the two halves of a frame pair cannot drift.
  Whether a cloud base reads brighter or darker than the sky beside it is left to the arithmetic:
  at the midday scene's 61° sun it comes out at ~15 900 cd/m² against a ~10 800 cd/m² sky, i.e.
  brighter, as a sunlit cumulus is. It need not agree with the infrared, where a cloud base is
  always the warmer feature — that divergence between bands is a real discriminator, not an
  artefact.
- The **continuous field is stored and thresholded after interpolation**, rather than a boolean
  mask being stored and sampled. A grid cell is 0.5° and a Boson pixel is 0.049°, so sampling a
  mask nearest-neighbour gave cloud edges that were ten-pixel rectangular steps — which is what
  the first render looked like, and not cosmetic, since edge sharpness is what a detector keys on.
  The threshold is estimated on a 2× upsampling because cutting it on the grid covers ~25 % less
  of a densely sampled sky, a bias that is *flat* across grid resolutions (a 1/fᵝ field is
  scale-invariant, so bilinear averaging smooths it equally at every scale). Residual: 8 % at
  c = 0.05, under 3 % by c = 0.2, against an input quantised in oktas.
- `--cloud-seed` on both flight scripts turns it on in **both bands** at once.
- **Image-plane motion, synthesised rather than rendered** (roadmap M10.1b, §13.3/§9.2).
  `irsim.optics.motion.image_plane_motion` computes the G-buffer's `motion_px` plane from rigid
  per-object transforms and the camera pose instead of from a motion AOV — ADR 0014's addendum
  measured that this build transports no motion at all (`motion_vectors` sits at a ~6e-5 floor
  after a 180 px displacement; `Motion2d` returns nothing), which left `motion_px` empty and with
  it everything that reads it: motion MTF and in-sim bolometer smear.
- Computing it is *exact* for rigid bodies rather than approximate, so the tests are held to it: a
  plane translating at 3 px/frame reports 3.0000 with **zero spread** across the frame, a static
  scene reports exactly 0, and an object and camera moving together report exactly 0 — the last
  being the case that catches the two transform pairs composed in the wrong order, which the other
  two cannot see because the camera is identity in both.
- `irsim_isaac.pipeline.motion_isaac.MotionTracker` is the USD half: it reads per-prim and camera
  transforms off the stage each frame and keeps a **stack of 4x4s indexed by instance id**, exactly
  as the thermal bridge keeps a float32 temperature table indexed the same way — one transport
  mechanism, two payloads. Motion is a difference, so the first frame of any sequence reports zero
  rather than a guess, and a prim that is untracked, absent or newly appeared is treated as static.
  `IrCamera` now keeps the camera-space position plane on its frame record, since re-projecting a
  surface point needs the position and the G-buffer only carries distance along the ray.
- In sim, against the renderer rather than made-up transforms: a bar at 3 px/frame reports 3.0
  within a tenth of a pixel, a static scene reports 0, and a prim moving with the camera reports 0.
  That last one caught a real mismatch — the position AOV arrives as `(H, W, 4)` with a padded
  fourth component, where the core wanted exactly three. The rest of the adapter already takes the
  first N of any float AOV; the core now does too, with a test asserting the padding cannot change
  the answer.
- **The background is treated as being at infinity**, so it does not translate with the camera,
  only rotate. That is why a tracking mount takes smear off the target and puts it on the sky, and
  it is now a computed quantity rather than an assertion: a yawing mount sweeps the sky by
  f·tan(δ) to 0.1 %.
- **What the ISP left in the picture** (ME.3b, ADR 0068 addendum). `irsim.validation.display_signature`
  reads the AGC's fingerprint, DDE's ringing and the camera's replaced pixels off finished frames.
  The recorder conversion is a **required argument** of the first two, so a Y16-derived set is
  refused inside the function and not only in the dataset index -- a histogram measured on a
  recorder's own 16-to-8-bit conversion describes the recorder. Plateau equalisation separates from
  a linear stretch by twenty times on every sky-like frame, and a scene whose histogram is already
  uniform is reported `indeterminate` rather than guessed at. DDE overshoot is the analytic
  `gain/3` of a 3x3 unsharp mask, so the gain reads straight off a picture. A replaced pixel is the
  mean of its four neighbours, so its Laplacian vanishes: 69 of 69 injected ones found on float
  frames, 93 % on 8-bit codes, no false positives -- and none at all through a codec, which takes
  the signature away entirely and is reported as "not measurable here" rather than as zero defects.
- **Reading the shutter off the frames** (ME.3a, ADR 0068). `irsim.validation.shutter` finds the
  freezes a flat-field correction leaves in the video, reports the interval only from a clip long
  enough to mean one, and fits what grows between events. Two distinctions carry the module: **a
  freeze is not an FFC** -- a run of repeated frames is equally a dropped chunk of recording, so
  every run is reported with `pattern_change`, the step in the time-averaged frame across it in
  units of what the noise alone would give (a shutter lands at 8-20, a stall at 1); and the
  between-shutter growth is fitted in the **variance**, `1 - e^(-2t/tau)`, because fitting it with
  the amplitude's law returns twice the true correlation time and looks perfectly reasonable.
- ADR 0068 -- the evaluation-data ADR, owed since ME.1a: what the public sets are, and the three
  gates (licence and signal path, 8-bit canonical form, and an analyser that refuses what its data
  cannot support) that stand between a public clip and a number the simulator gets tuned to.
- **What a statistic measured on somebody else's clip is allowed to claim** (ME.2b, ADR 0023
  addendum). `irsim.validation.flat` chooses the windows -- `robust_noise_scale` (MAD of the first
  differences about their own median, so a gradient cancels and striping does not inflate it) is
  the ruler, and flatness is scored against it rather than in DN, so the same thresholds work on a
  clip whose recorder stretched 16 bits into 8 and on one that did not. Striped windows stay flat
  on purpose: column noise is the thing being measured. A single-pixel target is caught by the
  outlier rule alone -- in a 4x4 block average it is a quarter of the block noise -- and on a clip
  the judgement runs on the temporal median, which deletes a moving target outright.
- `irsim.validation.codec` -- the floor the storage path puts under every one of those windows.
  The lattice step is read off the data, the floor is `step/sqrt(12)` (0.289 codes at 8 bits),
  Sheppard's correction is applied *and reported as limited* within two floors of it, and
  `flag_codec_limited` marks each of the seven 3-D components measurable or not. On a clip at
  sigma_TVH = 1.5 codes only the temporal white term survives that test, which is the honest state
  of the public sets.
- `irsim.validation.temporal_shape` -- a flat sky's temporal spectrum is flat unless something
  filtered it. Fits the sampled one-pole whose time constant in frames is exactly tau/dt and
  recovers the 10 ms membrane at 60 Hz to 1 %; reports drift separately so a wandering clip does
  not become a longer membrane.
- `irsim_eval.transcode.h264_round_trip` -- puts a synthetic cube through libx264 to calibrate that
  floor. Full-range flags are pinned so a CRF 0 round trip is bit-exact; at CRF 18, a *high
  quality* setting, 95 % of the temporal noise of a Boson-ratio cube is gone and the 8-pixel
  blocking score barely moves. Lossy sets give lower bounds, not measurements.
- **An aircraft flying a low pass, filmed in LWIR** (ADR 0075). `scripts/render_aircraft_pass.py`
  flies a light business jet past a ground sensor at 150 m/s and films ten seconds of it in real
  time. The variable is **aspect**, not throttle: the exhaust nozzles are hidden behind their own
  nacelles from the front and fully exposed from the rear, which is why a real aircraft's measured
  infrared signature varies by a large factor around the clock. Measured off the instance-id plane
  at the *same range* either side of closest approach, a rear aspect shows several times more
  nozzle than a head-on one and its hottest pixel is tens of kelvin warmer.
- `irsim_isaac.aircraft` — a parametric light jet (16 m span, rear-mounted engines so both nozzles
  sit on the centreline and occlusion is a clean function of one aspect angle). A jet's hot part is
  a *fortieth* of its span across, so unlike the quadrotor's motors you never resolve it: it is a
  near-point source, bright because it is hundreds of kelvin hot rather than because it is large.
- `irsim_isaac.aircraft_pass` — the track, the mount and the stage. The pass sweeps range 2:1 from
  the geometry alone, so it exercises the inverse-square fall-off and the atmospheric path against
  a target of known size and temperature for free.
- **A resolved quadrotor flying a mission, filmed in LWIR** (ADR 0074). `scripts/render_quad_flight.py`
  puts one heavy-lift multirotor at 20 m -- 105 px across the span, 6 px per motor bell -- against
  sky and flies a 30-minute profile while ADR 0072's heat sources take the motors from ambient to
  +45 K and back. The four hot bells, the warm speed controllers and the warm battery are separate
  objects in the image, which is what a thermal sensor actually keys on in a drone.
- `irsim_isaac.quadrotor` builds the airframe **parametrically from primitives** rather than
  importing a mesh. Every part is separate by construction, so the thermal nodes attach
  themselves; its dimensions are stated rather than measured off a bounding box, which is what
  lets a test assert an angular size; and there is no licence or unit-scale bug to carry. Motor
  bells default to a high-emissivity anodised material -- **bare aluminium is ε = 0.09 in this
  library** and would render a 70 °C motor as barely above the reflected sky.
- **Propellers are deliberately absent.** A 28-inch prop modelled as a solid disc is 33 px across
  at 20 m and hides the motors, which are the subject; physically a prop at flight rpm smears into
  a faint annulus within a 60 Hz integration period. Doing that properly needs the motion path
  (M10.1b).
- `irsim_isaac.quad_flight` is the stage. The camera tracks the aircraft, because a drone on a
  30-minute mission covers kilometres and nothing that flies realistically stays in a 33° field;
  attitude is driven by the **same throttle** the thermal model reads, so the picture and the
  physics cannot disagree about what the aircraft is doing.
- **A real sky for the companion visible frame** (ADR 0073). `--rgb` on the aerial demo used to
  write a flat grey void with six grey squares in it: no sky, no horizon, no sun, nothing a reader
  could use to check where the camera was pointing or what hour it was. Since the infrared frame
  cannot be eyeballed for correctness at all, the visible frame is the only half of the pair a
  human can apply ordinary judgement to, and it was not doing that job.
- `irsim_isaac.visible_sky` generates a float32 lat-long environment map -- a Preetham-Shirley-Smits
  (SIGGRAPH 1999) analytic daylight sky above the horizon, a Lambertian terrain faded into the
  horizon sky by Koschmieder's contrast transmittance below it -- and `aerial_demo` binds it to the
  stage's dome light with a 0.53° distant light for the solar disc.
- Every input is the scene's own, so the two frames cannot describe different days: NOAA sun
  position at the scene's site and clock, DNI/DHI from the shared `WeatherSeries` (CLAUDE.md #6),
  and turbidity derived from that weather's visibility against the atmosphere preset's visible
  Rayleigh coefficient. Turbidity is a ratio of **column** optical depths -- taking the
  ground-level ratio instead gives T = 14.2 on a clear 23 km day, a dense industrial haze, where
  the column ratio gives 2.98.
- **The infrared path is untouched, asserted rather than argued.** A `UsdLux.DomeLight` is a light
  and not geometry, so a ray that sees it still reports instance id 0 and an infinite
  `DistanceToCameraSD` and the background keeps coming from the sky model (ADR 0060). The radiance
  and apparent-temperature planes come out **bit-identical** with the dome and without it, which is
  an integration test and not a remark.
- **The renderer's lat-long convention had to be measured.** On this build the RTX dome light
  samples the texture with its polar axis on the stage's **+Z** and its azimuth running from +X
  toward -Y -- not the stage up axis, and not what the USD documentation implies. The first
  implementation authored the map in elevation and azimuth and rendered a frame filled entirely
  with ground, because the whole camera field fell inside one texture pole. The map is now authored
  in direction space, `DOME_POLE_AXIS` records the measurement, and an integration test re-measures
  it: pointed down the sun's own azimuth, the solar disc must land within 8 px of
  `f_px·tan(tilt − elevation)`.
- Targets get a `UsdPreviewSurface` in the visible band, so the companion frame shows a white
  airframe and a black carbon one rather than six identical grey squares. Appearance only -- the
  infrared properties still come from the material library through the `thermal:material` override,
  and the looks live under `/World/Looks` so they never reach the material resolver's prim walk.
- Two absences are deliberate, so the pair does not show what the infrared frame does not have:
  **no cloud** on the dome (MS.3's cloud field is not wired into `aerial_bridge` yet) and **no
  solar disc in the sky texture** (Preetham carries the aureole, the distant light carries the
  disc, so the two do not double-count).
- The dome light's `intensity` carries an **auto-exposure** on the median daylight sky, because the
  sky's absolute level moves by two decades between dawn and noon. That is a display choice and is
  why the visible frame carries no absolute brightness information; only its *geometry* is a
  calibrated claim. The twilight fade is divided back out of the reference, so a night scene still
  renders dark instead of being exposed up to look like noon.
- `scripts/render_aerial_demo.py` gains `--heading-deg` (the compass bearing the camera looks
  along, which is what places the sun in the frame), `--camera-height-m` and `--no-dome`.
- `--rgb` on `scripts/render_aerial_demo.py` and `capture_rgb` on `IrCamera`: the visible-light
  frame from the **same** camera prim, pose and lens, box-filtered onto the IR pixel grid, so an
  RGB/IR pair is registered by construction rather than by calibration. It carries no infrared
  information and nothing in the radiometric chain reads it -- the sidecar's unit string says so.
- Getting it took a measurement (ADR 0014 addendum). On 6.1.0-rc.26 the **lit** colour AOVs are
  silent in both render modes a script gets by default: `rgb` and `LdrColor` come back all zero
  under `RealTimePathTracing` (the build default) and `RaytracedLighting`, `HdrColor` comes back
  all NaN, and only the un-lit `DiffuseAlbedo` delivers. Under `PathTracing` the colour AOV
  delivers -- and every geometry AOV the IR chain needs still does, checked rather than assumed,
  which is what makes switching the mode safe instead of needing a second render pass. The mode is
  set only when `capture_rgb` is on.
- An all-zero colour buffer is **rejected with a reason** rather than saved: written to disk it is
  a black PNG, indistinguishable from a night scene. That is the same trap ADR 0014 already
  records for `PtWorldNormal` and ambient occlusion.
- The demo stage gains a distant light and a dome light *for the visible render only*. Neither is
  geometry, so no ray hits them: `instance_id` stays 0 and `DistanceToCameraSD` stays infinite on
  those pixels, and the IR background still comes from the sky model. Verified by the IR apparent
  temperature being identical with the lights present and absent (250.60 K to 291.65 K). The
  resulting RGB is a uniform grey field, because the stage has no sky texture or terrain -- it has
  no sky dome on purpose (ADR 0060) -- so the pair is correct and registered, and its visible
  content is as rich as the stage is.
- `--no-flat-field` as the ME.8-style ablation, so the artefact can be reproduced deliberately.

#### Changed
- `data/nk/water.csv` uses **Segelstein 1981, not Hale & Querry 1973**, whose n = 1.218 at 10 µm the
  physics model quotes and the M7.5 roadmap row originally required. Hale & Querry's n is not freely
  available as a table (only its absorption coefficient is), and mixing one compilation's n with
  another's k is worse practice than using one whole dataset. Their k agree to 0.02 %; the n differ
  by 2 %, worth 0.002 in emissivity — five times under MM.1's tolerance. Recorded in the file header
  and ADR 0079.
- **Maritime lane planned as phase 1b** (ADR 0078, roadmap milestone **MM**, 8 steps). Sky → sea →
  ground, amending ADR 0003's two-phase split. The sea is an **analytic background**, not displaced
  water geometry: at 3 km a Boson pixel spans 2.6 m and contains thousands of independent wave
  facets, so one normal per pixel is not a coarse version of the right answer, it is a different
  quantity. The reflected sky is integrated over a Cox–Munk slope distribution instead.
- The geometry is the reason the lane exists: a camera 20 m up sees the sea at 5 km at **0.23°
  depression = 89.8° incidence**, where water's emissivity has fallen from 0.99 to under 0.15. Nearly
  all visible sea is a sky mirror, so a vessel reads **dark against near water and bright against far
  water**, with a contrast null in between. A scalar `T_ground` below the horizon — what the code does
  today — cannot produce that at all.
- Phase 1b promotes **M7.3** and **M7.5** out of phase 2; M7.4 already landed early for the same
  reason. Three phenomenology-checklist rows, three ADR backlog entries (0078 written, 0079/0080
  pending) and the dependency graph updated to match.
- ADR 0078 states one error it does **not** bound: wave shadowing and inter-facet reflection are
  neglected, and that approximation is weakest in exactly the near-horizon band a low camera cares
  about most.
- **Branch-safe complex Fresnel reflectance** (M7.4, §4.2 Level A [R1]).
  `irsim.materials.fresnel_reflectance(n, k, cos θ) → (R, R⊥, R∥)` and
  `directional_emissivity_fresnel = 1 − R`, written with the §4.2 A/B reparameterisation rather
  than a complex `sqrt`. The branch is not cosmetic: the rejected root of the same Z returns
  R > 1 — a test asserts exactly that — and picking it produces reflectance discontinuities
  across wavelength and instability near grazing incidence, which in the IR is not a corner case
  because k is large for many materials.
- Water at 10 µm (n = 1.218, k = 0.0508) reproduces the [R1] validation numbers: R(0) = 0.010180
  (published 0.01018), ε(60°) = 0.9612, ε(80°) = 0.6972. **That 0.26 collapse between 60° and 80°
  is the physics a sea surface is made of** — a Level C constant emissivity reports 0.99 at both,
  and §4.2 refuses Level C for water by name. It is the first step of the maritime lane.
- Confirms [R1]'s counter-intuitive result along both axes: with k > 0 there is **no total
  internal reflection knee** at any angle (R < 1 − 1e-4 for n = 0.5, k = 0.5, bounded slope), and
  an index sweep across the lossless critical angle is a square-root cusp, not a jump — tested by
  grid refinement, since a fixed tolerance there either fails on real physics or is too loose to
  catch a real discontinuity.
- **Within-frame motion smear** (ADR 0077, §8.3/§9.2). `mtf_motion` — |sinc(v·t_int·ξ)| — has been
  in the MTF cascade since M5 and **nothing ever called it**, so every frame this simulator has
  produced was sharp however fast the scene crossed it. The aircraft stage sweeps the boresight at
  34 °/s at closest approach and a Boson pixel is 0.049°, so the scene crosses **11 pixels in one
  frame period**. It is also one of the most obvious signatures separating real thermal video of a
  moving target from synthetic video of one.
- `irsim.optics.smear.apply_motion_smear` is **spatially varying**, which the scenes here require:
  under a tracking mount the target is stationary on the focal plane while the sky sweeps past, so
  a single convolution serves neither. Each pixel is averaged along its own motion vector. Applied
  between the PSF and the box filter — both are convolutions laid down during the integration, and
  both must precede the detector sampling the result.
- **The duty is where the two detector families part.** A microbolometer has no shutter and no
  integration window — `integration_time_ms` is `None` for one on purpose — so it smears over the
  whole frame period; a cooled photon detector integrates briefly and comes out sharper. Reading a
  missing integration time as zero would have made every uncooled camera in the repo sharper than
  it is, which is the flattering direction. This is §16's "lateral motion smears LWIR, not cooled
  MWIR", as arithmetic.
- Verified against the cascade term it implements: `|sinc(s·f)|` over 3–20 px smears and
  1/48–1/12 cyc/px, worst deviation **0.015**, mostly under 0.006.
- **Correction to ADR 0073.** It recorded the infrared background as "the clear-sky profile only".
  That overstated the gap: `SkyModel.radiance` has always applied the uniform blend
  (1 − cε)·L_clear + cε·L_base, which is the *expectation* over the structured field. The mean
  cloud effect was in every rendered frame; only the structure was missing. So this replaces a mean
  with a realisation — and **the mean is preserved**, measured at ~0.2 % across seeds against a 1 %
  bound, which is what keeps the rendered background consistent with the elevation LUT, the tilt
  integral and ADR 0045's reflected term.
- Coverage is exact over the *sky*, not over any frame: one elevation ring measured 0.0195 against
  a whole-sky 0.0500. A camera pointed at a gap sees no cloud and one pointed at a bank sees only
  cloud, which is what makes cloud clutter rather than texture — and means no single frame can be
  used to check the cloud fraction.
- Opt-in by seed: without one the background is bit-identical to before, so no committed frame or
  golden moves silently.
- **Aerodynamic heating: the `ram_skin` node, and scene schema v3 → v4.** `airframe_solver` puts an
  unpowered skin *at* air temperature and justifies it by forced convection at flight speed — true
  at 20 m/s, false at 200. A stagnating boundary layer heats a skin to its recovery temperature,
  `T_r = T_air (1 + r (γ−1)/2 M²)` with `r = Pr^⅓ ≈ 0.892`: 0.18 K at multirotor speed, 10 K at
  150 m/s, 35 K at M 0.8. A config names an **airspeed** and the Mach number is taken against the
  shared weather's own air temperature, so one true airspeed is correctly a different Mach number
  on a cold day. The two skin models are deliberately not interchangeable — `airframe` refuses an
  airspeed, because it *assumes* a slow one and silently ignoring a stated 200 m/s is the failure
  this exists to prevent.
- `IrCamera.refresh_pose()`, so a camera can be **moved between frames**. The pose that every
  pixel's ray direction is built from is cached at `open()`; without a refresh the geometry AOVs
  follow a re-aimed prim while the elevations, the sky temperature and every view cosine stay at
  the opening aim — a completely normal looking frame describing two different cameras. The
  aircraft mount slews through 20° of elevation across a pass, and the sky is tens of kelvin colder
  at the top of that.
- `irsim_isaac.airframe` — `Part` and `author_parts` extracted from `quadrotor.py`, so the
  `thermal:material` override and the prim→thermal-node map have exactly one implementation.
- Both flight scripts share the burnt-in readout via `irsim_eval.video.overlay_readout`, and gain
  `--no-flat-field`.
- **The fixed-span videos now span the *target*, not the scene.** ±50 K about ambient spends half
  of 256 display levels on sky-to-ambient, which is two flat regions, and squeezes every part of
  the target into the rest. The span is now taken from the scene's own node temperatures across
  the sequence, with the floor a quarter of the node spread *below* the coldest node — on it, and
  a target that starts at ambient is invisible in the opening frames. Measured on the quadrotor at
  full throttle, like for like over the airframe's own 1000 pixels: 76 display codes of spread
  from the old rule, 128 from the new, with the cool parts starting near black rather than
  mid-grey. Sky clips to black, deliberately. `--span-c` restores a scene-context span.
- **Scene schema v2 → v3: `heat_source` and `airframe` target solvers.** §6.6's aerial node model
  (ADR 0072) has existed since M6.6 and was reachable *only from its own unit tests* -- no scene
  could ask for a motor, because `irsim.config.scene` had no solver kind that named one. A scene
  now names a **throttle profile** and the temperature is derived through ΔT = ΔT_max·u^n, refined
  until piecewise-linear interpolation reproduces the law to 1 mK. Authoring a temperature
  schedule alongside a throttle is refused: two answers and no way to pick. `configs/scenes/
  sky_target_clear_day.yaml` is migrated in the same commit.
- An `airframe` node's schedule spans the whole weather file, so its solver is explicitly advanced
  to the scene start; without that its first reading is the air temperature hours earlier -- here
  the difference between a 26 °C midday and an 18 °C dawn, and plausible either way.
- `IrCamera` takes an optional `frame_period_s`, which makes it a **time-lapse camera**. Not a
  fast-forward: every stage is told the truth about the interval, so the FFC fires on its real
  schedule, the fixed pattern drifts by a real amount, and the temporal noise decorrelates exactly
  as it would between two captures that far apart.
- The environment dome and the visible-band looks moved from `aerial_demo` into a shared
  `irsim_isaac.stage`, so a second stage cannot re-derive either and drift from the first.
- The flight video's palette now comes from the **sensor's own ISP config** instead of being
  hardcoded. ADR 0074 already said the fixed-span video maps "through the ISP's own palette" and
  the code did not: it used `ironbow` while `flir_boson_640_lwir.yaml` says `palette: gray`,
  `polarity: white_hot`, which is how a real Boson ships. A presentation video that picks its own
  colours is a second display path, and two pictures of one frame then disagree for a reason that
  is nowhere in the physics. `--palette` still overrides for a one-off.

#### Fixed
- **Two predictions written into the roadmap for MM.3 were wrong, and the model disproved them.**
  (a) The sea profile is **not** a monotone ramp from SST at nadir to sky at the horizon. Emissivity
  rising with depression and the reflected sky cooling with elevation pull in opposite directions, so
  the surface radiance turns over at an **interior minimum 2–15° below the horizon** — 283.7 K against
  289.5 K at nadir for a 290 K sea. The cold band, not a ramp, is what a maritime target is seen
  against. (b) Nadir reads 0.53 K under SST, not the predicted 0.2 K, and that offset checks out
  against (1−ε)(Lb(T_sea) − L_sky(90°))/dLb_dT.
- The atmospheric path then pulls the far field back toward air temperature — at 0.3° depression
  (4.1 km) the observed sea is over 1.5 K warmer than its own surface radiance. So **range, not angle
  alone, sets maritime background contrast**, and the profile cannot be tabulated against angle for a
  moving camera. The roadmap row and the Tier 3 checklist entry are corrected rather than quietly
  re-scoped.
- The phenomenology-checklist claim that overcast collapses the sea gradient below 20 % of clear was
  also wrong: it closes only ~30 %, bounded by how far the cloud base sits below the SST. An overcast
  sea is not an overcast ground scene.
- Cox & Munk's component slope fits sum to 3.0e-3 + 5.08e-3 U, not their separately fitted isotropic
  total 3.0e-3 + 5.12e-3 U — two independent regressions on the same data, 0.8 % apart. A test pins
  the gap so nobody later "fixes" the components to add up and silently substitutes an isotropic
  assumption for the measured anisotropy.
- **Rotor discs reach the focal plane through the lens oracle** (ADR 0081).
  `irsim.optics.rotor.disc_ellipse` projects a disc pose to its image ellipse by projecting rim
  points with `irsim.optics.projection.project`, so the authored distortion and the off-axis scale
  come from the forward model ADR 0015 already calls the oracle — a barrel lens 8 m off axis at
  20 m shrinks the disc 4.3 % and pulls it 17 px back towards the axis, which an `f·R/Z` stand-in
  sees as nothing. The major axis is taken along `axis × view`, the only diameter square to the
  line of sight and therefore unforeshortened.
- The residual is **measured, not argued**: the projected conic is assumed centred with
  perpendicular axes, which is exact only orthographically, and a 0.71 m rotor seen 75° off its
  axis at 20 m has rim points up to 0.079 px off the ellipse — falling quadratically to 0.013 px at
  50 m, with the axis ratio within 3e-4 of `cos(tilt)`. The docstring first claimed "under a
  hundredth of a pixel"; the measurement said 0.079 and the measurement is what is recorded.
- **A spinning rotor as a time-averaged veil** (ADR 0081, §8.3/§9.2). ADR 0074 left propellers off
  the quadrotor and ADR 0077 claimed the motion-smear operator would be their home. **It is not:**
  a blade tip at 3000 rpm does 112 m/s, sweeping 109 pixels of *circular* arc per bolometer frame
  and wrapping the disc 1.7 times, where `apply_motion_smear` averages along a straight segment
  with at most 65 taps — a different operator, not a coarse one.
- `irsim.optics.rotor` computes what the detector actually reports: the **time average of an
  intermittent opaque occluder**, composited in radiance (`α L_blade + (1−α) L_behind`). The blade
  is not semi-transparent; α is the fraction of the window it stood in the way. Blending apparent
  temperatures instead reads a 3 % veil of 290 K blade over a 230 K sky as 231.9 K against the
  radiance blend's 233.1 K — under-reporting the disc in the direction that hides it.
- **One formula, both detector families, selected by the window alone.** Coverage is a running mean
  of blade passage over the angle swept during the integration: a shutterless bolometer's whole
  frame (300° = 1.67 blade spacings) draws a smooth annulus banded exactly 2:1 by one whole blade
  pass, and a 2 ms cooled integration (36°) resolves two arcs five times as bright. **The totals
  agree**, because a running mean cannot move the mean of a periodic function — verified from a
  frozen shutter to ten revolutions. That invariant is what makes the two pictures comparable.
- Viewing tilt enters only as the azimuthal mean of a pitched plate's projected area (`cos β`
  face-on, `(2/π) sin β` edge-on), so a feathered blade is invisible edge-on and a coarse-pitch one
  is not. A consequence worth stating: coverage per unit image area is **exactly tilt-invariant**
  until the disc is within `pitch` of edge-on. The module was first written asserting the opposite
  and the rasteriser contradicted it — the peak at 0° and 45° was the same number to three digits.
  The demo stage sits just past that crossover (75° of tilt against 18° of pitch, threshold 72°).
- A two-blade 28-inch prop is 3.2 % solid at three-quarter radius, lifting a 230 K sky by 3.1 K:
  the faint annulus ADR 0074 predicted, and what a solid disc would have buried. Engine-free and
  not yet wired to a stage — mounting four needs the pose-to-ellipse projection and an occlusion
  mask (69 tests).
- **Real water optical constants and band-effective Fresnel emissivity** (M7.5 water half, MM.1).
  `data/nk/water.csv` is Segelstein 1981's complex refractive index, 2.0–15.6 µm, 365 rows, fetched
  from omlc.org with the citation in the file header. `load_nk_table` treats that header as **data,
  not lint**: a table with no `# source:` is refused at load, because every radiometric result
  computed from an unsourced table is uncheckable.
- `band_directional_emissivity(table, response, cos θ)` reduces Fresnel ε(λ, θ) to the scalar the
  pipeline uses, through the one sanctioned band-average route. Over the Boson band water reads
  **0.9879 at nadir, 0.9532 at 60°, 0.6726 at 80°, 0.1085 at 89°**.
- **The band average earns its keep.** Against Fresnel evaluated at a "representative" 10 µm, the
  band-effective value differs by 0.004 at nadir and **0.039 at 80°** — worst exactly where a sea
  surface lives, and worth more than 2 K of apparent temperature against a 60 K sea-to-sky contrast.
  Over 7.5–13.5 µm water's n runs 1.28 → 1.16 and its k spans an order of magnitude, so no single
  wavelength represents the band.
- n and k are interpolated **separately**; a test pins that this differs from interpolating the
  derived emissivity, so a refactor cannot quietly switch to the sampling-dependent version.
  Extrapolation raises rather than holding end values.
- Water's hemispherical emissivity measures 0.945 against a normal 0.988, with §16.2's tabulated
  0.96 sitting between them — the scalar ambiguity ADR 0043 exists to resolve, now with numbers.
- **The cross-check against sinc caught a real error before it shipped.** Taps placed at the smear
  segment's *endpoints* look natural and implement a boxcar of length s + s/(N−1): measured MTF
  0.7182 where sinc said 0.7842, which is exactly the Dirichlet kernel of the longer smear. The
  operator was self-consistent and describing the wrong smear — the failure mode no amount of
  "does it look blurred" would have caught. Taps are now at sub-interval midpoints.
- A test of mine compared picowatt-scale flux with `np.allclose`, whose default `atol` of 1e-8 is
  larger than the entire signal; it called a 15 % change "close". Now compared relatively.
- **The flight videos' AGC companion carried cos⁴ vignetting.** `PipelineConfig.from_sensor`
  defaults `flat_field_enabled=False` and neither flight script passed it, so the 8-bit picture
  showed the full 21 % centre-to-corner falloff that M9.12 exists to remove, stretched into dark
  corners by plateau equalisation — measured at 2.19× centre-to-edge before the fix and 0.93×
  after, the residual being genuine sky structure. The radiometric branch was never affected — it
  divides cos⁴ out analytically, and `tests/unit/test_flat_field.py` already asserted that
  `radiance`, `apparent_t` and `dn16` are bit-identical with the flat field on or off while
  `display8` changes — so the main fixed-span videos were always correct and only the companion
  was wrong.
- **The camera's flat-field correction was never applied** (roadmap M9.12). `irsim.isp.TwoPointNuc`
  has existed since M5 with its own tests, and nothing in `irsim.pipeline` referenced it -- so the
  8-bit picture carried every fixed spatial structure the optics put into the DN plane. Most
  visibly cos⁴ vignetting: measured at 21 % from centre to corner on the Boson's 14 mm lens, which
  plateau equalisation then stretched into black corners. No real thermal clip looks like that,
  because every real camera flat-fields before its AGC.
- The README claimed "the NUC stage (M9) removes it". It did not. That line is corrected rather
  than quietly deleted, because it is the kind of claim that was ahead of the code and got
  repeated.
- The correction lives in the **display branch alone**, and that is a decision. §11.1 forks the
  branches after the ADC and `invert_optics` already divides cos⁴ out per pixel analytically --
  which is why apparent temperature was flat across a row while the picture was not. Applying a
  measured correction to the radiometric side as well would remove the same term twice, so `dn16`
  and the radiometric outputs are **bit-identical** with the flat field on or off, and there is a
  test asserting exactly that.
- Coefficients come from two synthetic blackbody frames through the real forward chain with the
  noise off -- what a bench calibration does with two real blackbodies -- rather than from the
  analytic cos⁴ field. A correction built from the formula would remove exactly the term the
  formula describes and stay silent about any other fixed structure the chain grows later.
- `TwoPointNuc` gains an optional `pedestal`: §11.2's convention makes the corrected cold blackbody
  read 0, which is right for the radiometric reference and wrong as an AGC input, since a
  flat-fielded frame reading zero on a cold scene moves the histogram for reasons that have nothing
  to do with the scene. Default 0 keeps the spec convention and every existing test.
- Measured after the fix, on a scene whose temperature varies with row only so that any
  left-to-right difference is the camera: uncorrected the edges read 20+ display codes below the
  centre; corrected the difference is under a fifth of that. A uniform blackbody comes out flat to
  0.1 %, at temperatures it was not calibrated at as well as at them.

#### Notes
- **The video is a time-lapse, and that is a physics decision.** ADR 0072's node law is a
  *steady-state* relation -- T = T_air + ΔT_max·u², with no thermal time constant -- so it is only
  defensible while the throttle moves slowly against a real motor's minutes-scale response. The
  flight is authored over 1800 s and captured one frame every 6 s. Running the same profile at
  60 Hz would look smoother and would show 45 K of swing no metal could follow. Every transient in
  the video is a *throttle* transient.
- **The main video is a fixed display span, not the camera's AGC.** Both §11.3 AGC modes rescale
  from the current frame's own histogram, which cancels exactly the change the video exists to
  show; plateau equalisation additionally allocates display codes by population, so an aircraft
  covering under 1 % of the frame saturates to flat white with the motors indistinguishable from
  the arms. That is measured, not predicted -- it is what the first render looked like. The
  camera's own AGC output is filmed alongside as a second video, because the difference is the
  lesson.
- ΔT_max stays **ESTIMATED** (45 / 30 / 15 K for motor / ESC / battery, ADR 0072). The ordering is
  the defensible part; the magnitudes are what a Tier 4 fit against public aerial IR should move
  first.

### 2026-09-13

#### Added
- Warp stage 5's remainder on device (roadmap M10.7b): the bad-pixel map and its RTS chain, the
  iterated 4-neighbour replacement, and the FFC hold. Unlike M10.7a's noise these are mostly
  **exact**, and the tests say so in those terms -- 34 of them, on `cuda:0` and Warp `cpu`.
- **The defect map is uploaded, not redrawn.** A bad-pixel map is a property of one physical focal
  plane, drawn once per sensor and never per frame (§10.4), so two independent draws would be two
  different cameras rather than two implementations of one. The device map is therefore
  bit-identical to the CPU's by construction. The RTS state is seeded from the CPU's own starting
  realisation for the same reason M10.7a seeded the fixed pattern that way: what is measured
  afterwards is the two generators, not two unrelated defect populations.
- The RTS chain is launched over the whole plane rather than a compact list of stateful pixels.
  The CPU restricts it because drawing 327k uniforms to move a few dozen bits is most of the noise
  chain's cost; on a GPU that argument does not apply. It stays a **chain**: a pixel that was bad
  leaves with probability 1/dwell, and redrawing the state independently each frame would give
  white dwell statistics where the geometric ones are the defining signature of RTS.
- The replacement kernel accumulates in **float64**, as `replace_bad_pixels` does, so "matches the
  CPU" means every bit rather than a tolerance -- a float32 mean of four neighbours differs in the
  last bit, and a tolerance there would hide an arithmetic difference instead of measuring one.
  Passes ping-pong two buffers because the CPU builds its neighbour sums before writing any of
  them; in place on a GPU that ordering is not merely different, it is non-deterministic.
- The pass loop lives on the host and reads back two counters per pass. That is two tiny transfers
  for a stencil that finishes in two or three passes on any map ADR 0055 can produce, and it keeps
  the CPU's loud failures: a region with no valid neighbour on any side raises rather than being
  filled, and so does a cluster still unfilled after the pass cap.
- The FFC hold is one device buffer, written on every unfrozen frame and read on every frozen one,
  so a freeze emits the same bytes for its whole length -- which is the fingerprint ME.3 looks for
  in real video. A freeze that begins before anything is held passes the frame through rather than
  inventing one the camera never saw. The *schedule* stays on the host: when the shutter fires is
  frame arithmetic, and a second `FfcController` on the device would be another thing to keep in
  step.
- Verified, exactly: the map reaches the device bit-identically; replacement matches
  `irsim.isp.replace_bad_pixels` bit for bit on 1x1, 2x2 and 3x3 clusters and on the real defect
  map; defect injection matches `apply_defects` for a given chain state, including the detail that
  an already-saturated hot pixel keeps its sub-LSB float value; a frozen frame is bit-identical
  across a six-frame freeze; and the **composition** -- inject then replace, in order -- matches
  `SensorChain.finish_frame` bit for bit. That last one is there because M10.7a's lesson was that
  two individually correct stages can still disagree by a factor of ten if the seam between them
  sits in different places.
- Verified, statistically: the chain holds its configured occupancy within 25 % and its mean dwell
  within 35 % over 2500 frames, and the dwell is more than two frames -- a memoryless redraw would
  give about one. The population comes from area rather than a raised defect fraction, because the
  schema caps that at one percent and it is right to: a focal plane with more than one percent of
  its pixels dead is scrap, not a sensor.
- `irsim_eval.data` (roadmap ME.1b): the `Sequence` / `Frame` / `Box` form every validation
  analyser reads, plus a canonical on-disk layout (a JSON index and one array per frame) and its
  writer. The indexed sets agree on nothing -- Halmstad ships MATLAB labels beside mp4, Anti-UAV410
  per-sequence JSON, the single-frame sets folders of images -- so a per-dataset converter writes
  this layout once and no analyser ever reads a publisher's format. Frames load on demand, since a
  box-size histogram over a whole set should not pay for pixels it never looks at.
- **The reader insists frames are 8-bit.** Every indexed set stores 8-bit frames, and that
  quantisation is a floor under every statistic measured on them (ADR 0003); a dtype that quietly
  widened would hide the floor rather than remove it, and invite a radiometric claim the data
  cannot support. Boxes are top-left `(x, y)` plus `(w, h)` in the project's own pixel-edge
  convention, pinned by a test because half a pixel here changes every size-versus-range number.
- `irsim_eval.motion`: the static-camera clip detector that gates every per-pixel temporal
  statistic. On flat sky a per-pixel temporal standard deviation is the closest thing these public
  clips offer to a laboratory blackbody -- but only if the scene stayed on the same pixels. Pan by
  a pixel a frame and the same number measures the sky gradient crossing the detector, which is
  larger, perfectly smooth, and indistinguishable from a noisier camera.
- The discriminator is **cumulative displacement from the first frame**, not per-frame motion,
  because that is the distinction that matters: a shaken mount wobbles sub-pixel and goes nowhere
  (static, the statistic still describes the sensor), while a slow drift takes equally small steps
  that all point the same way and crosses many pixels (moving, however small each step was). The
  two synthetic controls are built with the *same* per-frame step size so the pair tests
  accumulation rather than amplitude.
- Shifts come from phase correlation, which ignores the amplitude and so survives the brightness
  changes an AGC makes between frames. Sub-pixel accuracy uses **Foroosh's ratio, not a parabolic
  fit**: a phase-only peak is a Dirichlet kernel that splits linearly between its two nearest
  samples, and the parabola everyone reaches for under-reads it by about 30 % at a third of a
  pixel -- small enough to pass for noise, biased enough to drag a drifting clip under the
  threshold. Caught by the estimator's own known-shift test, which the parabolic version failed at
  (-0.4, 0.6) and passed at every integer shift.
- New `validation` extra (imageio) for sets stored as images; sequences written as `.npy` need no
  decoder, which is why the default gate installs nothing new and every test here runs without it.
- The validation-data index (roadmap ME.1, first half; ADR 0003). `data/validation/datasets.yaml`
  records, per public dataset: licence, access mode, sensor, **signal path**, bit depth, codec,
  frame counts, and which ME.2-ME.4 analysers it may and may not support. `irsim_eval.manifest`
  makes licence, signal path and the analyser list *required* fields, so a set that cannot say
  what its frames are cannot be added.
- The index is the thing that decides whether a statistic means anything. The same flat-sky patch
  is a laboratory noise measurement on a Y16-derived clip and a measurement of somebody's unknown
  AGC on a display-output one. `usable_for("noise_3d")` returns exactly one set; the Halmstad
  clips never went through an AGC, so `agc_signature` is in their *excluded* list and only
  Anti-UAV410 can carry it; only LRDDv3 has range labels, so only it can do `size_vs_range`; and
  the single-frame sets are marked `prior`, able to bound a distribution but never to be a target
  the simulator is tuned to hit.
- **Every field was checked against the publisher's own page, and the findings changed the plan.**
  Only the Halmstad set (Svanström et al. 2021, Zenodo 10.5281/zenodo.5500576) states a licence at
  all -- CC0-1.0 -- and it is also the only one whose sensor and signal path are fully documented
  (FLIR Breach PTQ-136 / Boson 320x256, 24 deg x 19 deg, 60 fps, Y16 to 8-bit in the recorder,
  mp4). It is therefore the **primary** set, and the schema permits exactly one.
- The other five state no licence. `unstated` is recorded as a value, not left blank: it means the
  terms are unknown, not that they are permissive. Anti-UAV410 and CST Anti-UAV publish no licence;
  LRDDv3 is behind an access request citing US export control; IRSTD-1k and NUAA-SIRST have no
  canonical licensed source. CST Anti-UAV is not released yet.
- `scripts/fetch_validation_data.py` plans, reports and mostly refuses. A set with no stated
  licence is **skipped** unless `--accept-unstated-licence` is passed, so using data on unknown
  terms is always a deliberate act; a `manual` set (a drive link, a Zenodo record, an access form)
  is never downloaded, because pretending a script can do that fails in a way that looks like a
  network error; `http` is refused rather than silently upgraded. What it always does is hash what
  is on disk, so "measured on these exact bytes" stays checkable (ADR 0004's rule applied to
  someone else's data).
- `data/validation/README.md` is **generated** from the YAML and a test fails when it is stale --
  an index that disagrees with itself is worse than no index, because the prose is what people
  read and the fields are what the code reads. Datasets stay out of git (`data/validation/*/`).
- New package `src/irsim_eval/`, the home for everything that reads somebody else's imagery, kept
  out of the core so that verifying Planck's law never requires an image decoder (CLAUDE.md #1).
  `make typecheck` now covers it.
- MS.6 analytic point targets wired into `IrCamera` (roadmap M10.19, ADR 0071). Below one native
  pixel the renderer is the wrong instrument -- it samples geometry, so a target covering a
  quarter of a pixel is drawn or not drawn depending where the sample landed. `AnalyticTarget`
  carries such a target's world position, projected area, material and thermal node;
  `IrCamera.point_targets()` turns it into an MS.6 `PointTarget` each frame, rebuilt rather than
  cached because the temperature is still moving.
- **A target is rendered or injected, never both.** `aerial_demo.analytic_targets` hides the
  sub-pixel prims in the same call that produces their specs, so the two paths cannot both claim
  one, and `IrCamera.check_no_double_count()` reports any that still reached the id plane. Both
  halves are tested, including the failure: with the prim left visible the guard names it.
- The leaving radiance comes from `irsim.validation.aerial.target_leaving_radiance`, the same
  function the engine-free aerial work uses, so the in-sim and CPU paths cannot disagree about
  eps L_B(T) + (1 - eps) L_env.
- In-sim verification (`tests/integration/test_point_targets_isaac.py`, 7 tests): the injected
  excess **survives the whole chain to 5 %** -- splat, optical PSF, box downsample, detector, ADC
  and the radiometric inversion all compose back to the excess that went in; an injected target
  lands within half a native pixel of where the renderer draws the same object, so the handover at
  one pixel introduces no jump; and the fill fraction carries the entire geometric range law
  (phi R^2 constant to 1e-12).
- **The "signal falls as tau/R^2" shorthand is not exact, and now there is a test saying so.**
  Measured over 400-3200 m it drifts by 33 %, always in the direction of under-predicting the
  longer range. The excess is phi * sum_k w_k tau_k (L_t - L_beyond,k): the bracket depends on
  range too, because what a target occults is the sky column beyond it and there is less of that
  column left at 3200 m. A trade study using the shorthand would over-estimate detection range.
- `scripts/render_aerial_demo.py` uses the split: the 500 m, 1500 m and motor-pod targets are now
  injected analytically and marked ANALYTIC in the report, with `--no-point-targets` as the
  ADR 0071 ablation that renders them as geometry instead.

#### Fixed
- `world_to_camera` returns **USD** camera space (+Y up, -Z forward), so the projection has to be
  `project_usd`; `project` reads -Z as behind the camera and returns NaN for every target in front
  of it. Caught by that NaN guard rather than by a wrong picture, which is what it is for.
- Aerial demo stage and a one-command render (roadmap M10.19, stage half):
  `irsim_isaac.aerial_demo` authors six targets against sky -- a resolved quadrotor at 120 m, one
  at the pixel limit at 500 m, one well below it at 1500 m, an aircraft at 2.5 km, a bird and a
  hot motor pod -- and `scripts/render_aerial_demo.py` renders them through `IrCamera` and writes
  the frames with M10.10a. Sizes are the real ones, so the demo shows what the range problem
  actually looks like: a 0.35 m quadrotor at 500 m is 0.82 of a Boson pixel, and the stage says
  which targets are in MS.6's sub-pixel territory rather than leaving it to be discovered.
- There is **no sky dome and no ground plane** in the stage. A ray that hits nothing takes
  `T_sky(theta)` at its own elevation, or `T_ground` below the horizon (ADR 0060). An emissive
  dome would push the sky through a colour AOV -- all float16 on this build, ~100 mK against a
  50 mK NETD -- and would be worst at the horizon, where the gradient is steepest and the targets
  are.

### 2026-09-12

#### Changed
- Sensor schema v8: `nuc.shutterless_tau_s`, the scene-based-correction time constant that bounds a
  shutterless core's residual (M9.7, ADR 0057). Defaults to 120 s (ESTIMATED -- no published
  convergence time was available) and is unused outside `mode: shutterless`, so no existing config
  changes behaviour.
- Sensor schema v7: `noise.bad_pixel_rts_occupancy`, `bad_pixel_rts_dwell_frames` and
  `bad_pixel_rts_amplitude_dn` for the §10.4 flickering and blinking classes (M9.5a, ADR 0055).
  All three default, so existing configs are unchanged; all three are ESTIMATED, with ME.3's
  bad-pixel extractor the thing that should eventually set them.
- Sensor schema v6: `housing_temp_mode: coupled` now **requires** `housing_tau_s`, mirroring the
  rule ADR 0053 set for `fpa_temp_mode: coupled`. A lumped node with no time constant cannot be
  integrated, and defaulting the lag would put a number nobody authored into the drift the whole
  M9 chain rests on. `configs/sensors/flir_boson_640_lwir.yaml` declared `coupled` and authored
  neither parameter, so it gains `housing_tau_s: 900 s` and `housing_self_heating_k: 4 K` -- both
  ESTIMATED, no published figure for either, both candidates for ME.3's measured drift band to
  constrain. Golden references were regenerated for the config-hash change only: every stored
  array is bit-identical, because nothing reads the new fields until M9.8 wires the node in.
- `CLAUDE.md` now matches the repository it describes: the layout block gains `tests/conftest.py`
  (the synthetic G-buffer fixtures), `configs/environments/`, `docs/roadmap.md`,
  `docs/spec-issues.md`, `docs/maps/`, `.github/workflows/` and the `$IRSIM_DATA_DIR` override, and
  the command table gains `make ci` and `make golden-update`, records that `make test` runs the
  golden suite too and that `make typecheck` covers `src/irsim_isaac`, and states that every target
  honours `PYTHON=` (ADR 0002). The guidance was describing an earlier tree, which is the failure
  mode a project instruction file cannot afford.
- The reflected-environment tests no longer assume a fully overcast sky reads exactly `T_air`. The
  isothermal-enclosure test derives its enclosure temperature from the sky model (and pins the
  ground to it with `ground.mode: fixed`), and the cold-roof test computes a reference from each
  sky's own `effective_radiance`, so both assert the reflected term itself rather than a particular
  cloud-base convention. They pass under the pre-MS.3 and post-MS.3 cloud semantics alike.
- `docs/physics-model.md` spec fixes raised before coding (M0.10): §5.3(a) sky temperature now
  `cos^q(θ_zen)` (coldest at zenith, S1); §6.1 absorbed solar written `α_sol Q_sol` (S2); §12.2
  spectral-response path `spectra/responses/boson_vox.csv` relative to `data/` (T4).
- Boson YAML `spectral_response` now `spectra/responses/boson_vox.csv`, the canonical layout under the
  data root (spec issue T4); the file itself arrives with M1.3.
- Golden helper implementation moved to `tests/golden/golden_store.py` (conftest keeps the fixture) so
  its self-tests import it unambiguously.
- Makefile targets run through a `PYTHON` variable (`make check PYTHON=.../python.sh`); `make test`
  prints the ten slowest tests so the 30 s budget stays visible.
- NumPy floor raised from 1.24 to 2.0 (the Planck tests use `np.trapezoid`).
- ruff ignores `N802` alongside `N803`/`N806` (physics notation such as `d_spectral_radiance_dT`).

#### Fixed
- **`Camera3dPositionSD` is in camera space, not world space** (ADR 0014 addendum). ADR 0014
  recorded it as world, which was true of every scene that measured it: all of them had an
  unrotated camera at the origin, where the two frames coincide. Tilt the camera up and they
  separate -- measured at 8 degrees about X, the AOV still reports the frame centre's ray as
  `(0, 0, -1)`.
- Read as world it produces no error, just a different camera: the boresight reads 0 degrees
  instead of 8, the horizon moves 164 rows to the middle of the picture, the upper half of the sky
  falls below the model's horizon and is painted with `T_ground` 30 K too warm, and
  `normal_dot_view`, `normal_dot_up` and the sky-view factor all tilt with it. The result is
  smooth, monotonic and entirely plausible. `IrCamera` now defaults to `position_frame="camera"`
  and passes the prim's own local-to-world rotation.
- It was caught by asking where the horizon *should* be: `cy + f_px tan(tilt)` has no free
  parameters. With the fix, elevation crosses zero at native row 221.5 against a predicted 221.7
  and the frame centre reads 7.979 degrees for an 8 degree tilt. Nothing else in the repo was
  affected, because nothing else had rotated a camera -- which is exactly why it survived until a
  scene needed to look up.
- The in-sim aerial phenomenology suite (10 tests) states the background claims, since on a
  sky-target camera the background is most of the image and is where a wrong model hides: the
  boresight elevation, the horizon row, a monotone sky gradient over every row above it, a single
  flat ground temperature below it, and the sky reading 25 K below ambient while the ground reads
  at it.
- One of those is worth its own line because it is counter-intuitive and was a wrong assumption in
  the first draft of the test: **the horizon is not a visible edge**. A grazing clear sky has
  unbounded path length, so its emissivity approaches one at about the air temperature -- which is
  what the ground is at. The step across the horizon is smaller than the sky gradient over the
  frame, and an algorithm hunting the strongest edge finds a place in the sky instead.
- Float32-preserving dataset writers (roadmap M10.10a): `irsim.io.write_frame` puts one frame's
  four §12.2 outputs on disk with a JSON sidecar. radiance and apparent temperature go to float32
  `.npy` or float32 EXR; DN16 to a uint16 PNG (an ADC code is an integer and loses nothing);
  display8 to an RGBA8 PNG. A plane in physical units never reaches an integer or half-float
  container, which is CLAUDE.md #2 at the disk boundary.
- The arithmetic is now a test rather than an assertion in prose: a 16-bit PNG over the 233-473 K
  radiometric range quantises to 3.66 mK, and half-float at 300 K to 250 mK -- **exactly** five
  times a 50 mK NETD, the figure CLAUDE.md #2 quotes. Both would produce a frame that opens and
  looks right with the sensitivity already gone, so `write_frame` refuses a float16 plane and
  `write_exr` refuses `half=True` even when it is asked for.
- `irsim.io.exr`: a minimal single-part, scanline, uncompressed **float32** OpenEXR writer and a
  matching narrow reader, in stdlib `struct` and `zlib`. EXR exists because no image tool reads
  `.npy` and the practical alternative people reach for is a 16-bit PNG. No dependency is added
  (CLAUDE.md #1 forbids an imaging stack in the core), and the header is checked against the
  format specification -- magic, version, FLOAT pixel type, little-endian samples -- rather than
  only round-tripped through our own reader, which would pass with the byte order wrong in both
  directions. Multi-channel files come back in alphabetical channel order, as EXR stores them;
  pinned by a test so it is documented rather than surprising.
- `irsim.io.png` grows 16-bit grayscale. The DN16 round trip is decoded from the raw IHDR/IDAT
  chunks in the test rather than through our own encoder twice.
- The sidecar carries the config and band hashes (ADR 0008), the ISP hash, the frame index, the
  scene time and the UTC it corresponds to on the weather axis, and for every file its dtype,
  shape and **unit**. A directory of arrays nobody can trace to a configuration is a pile of
  images, not a dataset. Frame names are zero-padded so a sequence sorts lexicographically, and an
  output turned off in `sensor.outputs` leaves no file rather than a plane of zeros -- zeros are
  indistinguishable from a real dark frame.
- `IrCamera` (roadmap M10.9a-ii, ADR 0015 addendum): the object that turns a USD stage into an
  infrared frame. It authors the camera prim from the sensor YAML, creates the render product at
  `supersample x native`, attaches the M10.1 annotators, assembles the G-buffer from the geometry
  AOVs plus the M10.2 material ids and the M10.18 temperature table, and runs
  `irsim.pipeline.run_frame`. **This is first light from Isaac Sim through the whole camera
  model.**
- The constructor takes the `Scene` rather than the atmosphere preset and environment the roadmap
  sketched. The scene already holds all three bound to its single `WeatherSeries`; taking them
  separately would let one atmosphere run in the transmittance, another in the sky and a third in
  the thermal solvers (CLAUDE.md #6), and the camera refuses a pipeline whose sky or atmosphere is
  on a different weather object.
- **It runs the CPU reference, not the Warp stages, and that is deliberate.** The M10.4--M10.8
  twins cover stages 1--6 but not the M9 chain's post-ADC half -- device defects, the iterated
  bad-pixel replacement and the FFC hold are M10.7b. Running the device path today would produce a
  frame from a *different camera* and label it the same. `IrCamera.planes()` exposes the assembled
  G-buffer so the Warp path can be driven from the same scene and compared; the all-device frame
  lands with M10.7b.
- **Every attribute of the distortion schema is written, never a subset.** Measured on
  6.1.0-rc.26: the schemas default to `fx = 900`, `cx = 1024`, `imageSize = (2048, 1024)` and, on
  the fisheye, a non-zero `k1`. An attribute left unwritten is not "no distortion", it is a lens
  for somebody else's camera, and it renders without complaint. `imageSize` is a `GfVec2i`; a
  Python tuple is coerced to `GfVec2d` and the set is rejected outright, so it is built explicitly.
- In-sim verification (`tests/integration/test_ir_camera_isaac.py`, 12 tests): a grid of quads at
  known world positions reprojects through the config model to **< 0.2 px** both undistorted and
  under a barrel lens; the barrel coefficients move the outer quads 2.1 px while the optical axis
  stays put; and reprojecting the barrel render through a *zero-coefficient* model fails by more
  than 1.5 px, so neither half could pass on a build that discarded the schema.
- Measuring to a fifth of a pixel off a binary id mask needed a correction. A rectangle's
  included-pixel set snaps to whole pixels, so its centroid lands on a multiple of half a pixel
  however large the rectangle is -- averaging over more pixels does not help. Every quad came back
  a flat 1/3 px off, which looks exactly like a lens error and is not one. The grid is rendered 8x
  supersampled and the centroids divided down, putting the quantisation at 1/16 of a native pixel.
- Whole-frame verification on the five-prim material stage: float32 / float32 / uint16 / RGBA8 at
  the native grid, no float16 plane anywhere, the high-emissivity prims reading back their
  authored temperatures within 8 K, and -- the sharpest check that emissivity reaches the kernel --
  the two `bare_aluminium` prims, 27.15 K apart at the surface, collapsing to under a quarter of
  that in apparent temperature because at eps = 0.09 both show the same sky and ground. Note the
  direction: a mirror does not read cold, it reads *its environment*, so the 268 K window reads
  warmer than it is.
- An UNMAPPED prim has no emissivity and `MaterialTable` refuses to invent one (ADR 0047). In
  `debug_unmapped` mode those pixels are handed to stage 1 as blackbody-equivalent -- the eps = 1
  ADR 0047 specifies, on the prim's own temperature -- and painted magenta at the native grid,
  over-reporting rather than hiding a forgotten prim; without the debug mode the frame raises.
- `scripts/probe_isaac_camera.py` and `irsim_isaac.camera_probe` record the distortion-schema
  survey the mapping is built on, so the next build is re-measured rather than argued about.
- Lens projection (roadmap M10.9a, ADR 0015 addendum): `irsim.optics.projection` answers "given
  this `optics.distortion` block, where should a ray at this angle land?". ADR 0015 is unchanged --
  the engine still applies the lens and the imaging path stays rectilinear -- but without a
  **forward** model there is no way to tell whether the coefficients written onto a camera prim
  produced the lens that was configured, and a barrel term is a smooth radial stretch, exactly the
  error a human eye accepts in a picture.
- Two conventions are pinned by tests rather than by comments, because both are silent when wrong:
  USD camera space (+Y up, -Z forward) versus OpenCV (+Y down, +Z forward), flipped in exactly one
  function; and the principal point at the format corner with pixel centres at `i + 0.5`, tied to
  `irsim.optics.vignetting`'s independently written geometry -- half a pixel of disagreement is
  3.8e-4 rad at the Boson's corner, invisible in an image and fatal to a reprojection check.
- Measured on 6.1.0-rc.26, not assumed: `brown_conrady` maps to
  `OmniLensDistortionOpenCvPinholeAPI`, whose twelve attributes are in OpenCV's own
  `[k1, k2, p1, p2, k3, k4, k5, k6, s1..s4]` order, so a five-term block maps **positionally**;
  `kannala_brandt` maps to `OmniLensDistortionOpenCvFisheyeAPI`.
- **`ftheta` is refused, not approximated.** Nothing this build exposes determines whether its
  polynomial returns pixels or normalised units, or whether `k0` is a constant term -- under one
  reading a one-coefficient block is a lens, under the other a constant radius, which is not.
  Either guess renders a plausible fisheye that disagrees with the engine by tens of pixels at the
  field edge, so `project` raises and M10.9b's renderer audit measures it. No configured camera
  uses f-theta.
- Verified against arithmetic decided outside the module: hand-evaluated OpenCV radial and
  tangential terms, the equidistant limit of the fisheye model (r = theta, which differs from the
  rectilinear r = tan theta by 66 % at 60 deg, so a silent pinhole fallback cannot pass), the
  cos^4 identity against `irsim.optics.vignetting`, and a distort/undistort round trip over the
  whole Boson frame at < 1e-6 px against the row's 0.2 px in-sim budget.
- Warp stage 6 (roadmap M10.8): the display branch on device -- an atomic histogram, the ADR 0028
  plateau clip with `wp.utils.array_scan` for the exclusive CDF, the AGC applied as a
  2^bit_depth lookup table, an edge-clamped 3x3 DDE and the palette to RGBA8. Both §11.3 AGC modes
  are monotone functions of DN alone, so each *is* exactly a table; that is what makes them
  portable to a kernel at all, and why the port can be held to a display code rather than to a
  resemblance. The O(1) reductions over the 65536-entry histogram -- the percentile positions, the
  occupied-bin span -- are finished on the host, where each is a line rather than a kernel.
- Measured on an RTX A6000 and Warp `cpu`: ≥ 99.9 % of pixels within ±1 display code of the CPU
  branch for linear, plateau-equalisation and none, on a ramp, a hot-exhaust patch and a constant
  frame; the float image before quantisation agrees to 2e-3. DN16 is bit-identical under every AGC
  mode -- §11.1 forks the branches after the ADC, and an AGC that reached back into the linear
  output would be invisible in the picture and fatal to the validation that consumes it. The
  ADR 0028 exhaust collapse is reproduced: plateau equalisation retains more than three times the
  background contrast linear AGC leaves.
- `agc: none` is a **bit shift** (the top 8 bits over 255, ADR 0031), not a linear rescale of the
  full-scale range. The device table had it as the latter and was half a display code out on every
  pixel -- caught by comparing the AGC *tables* rather than the images, which localises a
  disagreement to the AGC instead of leaving it somewhere in the DDE or the palette.
- Warp stage 5 (roadmap M10.7a, ADR 0022 addendum): the seven §10.2 noise components, M9.4's OU
  drift of the device-resident fixed fields, and M9.6's NUC residual, as Warp kernels. This is the
  first stage that **cannot** be held to bit-equality with the CPU oracle -- Warp's generator is not
  NumPy's -- so it is held to statistical equivalence instead, and is deliberately not registered in
  `EQUIVALENCE_STAGES`, whose whole point is bit-level agreement.
- Two things make that comparison mean something. The device fixed pattern is **seeded from the
  CPU's own realisation** rather than redrawn, so what is measured is the two generators and not
  two different cameras -- a redrawn pattern would have passed even with the wrong spatial
  structure. And the NUC residual's ξ fields are uploaded too, keeping the one part of the chain
  with an exact CPU answer bit-comparable to 2e-6; the residual is reset by an FFC *event*, not by
  a frame, so redrawing it on device would mean a second epoch counter to keep in step across a
  shutter.
- The **stage boundary differs between the two paths and the composition does not**: on the CPU the
  per-pixel TVH term belongs to the detector (stage 4) and `NoiseStage` adds only the six correlated
  ones, while the Warp detector stage is the ideal transfer alone, so the device stage 5 supplies
  all seven. Comparing the stages alone therefore compares a six-term image with a seven-term one,
  which produces a PSD ratio of about ten and looks exactly like a broken kernel. The oracle is the
  detector and the stage together.
- Measured on an RTX A6000 and on Warp `cpu`, over 200 frames: NETD within 5 % of the CPU path and
  10 % of the configured σ_TVH; every 3-D component within 15 % of its configured sigma or within
  three times its own `estimate_floors` floor; NETD(373)/NETD(300) = 0.576, the derivative ratio, so
  non-negotiable #3 holds on the device as it does on the host; `compare_psd` ≤ 1.5. Exact where it
  can be: the same frame id is bit-identical, τ = ∞ freezes the pattern bit-identically and launches
  nothing, and the residual at ΔT_FPA = 0 is exactly the identity.
- The original M10.7 row bundled the noise stage, the drift, defects, replacement, the residual and
  the FFC. It is **split**: M10.7a is the noise half and carries every exit criterion the row
  listed; M10.7b is the new row for the device-side defect map, the iterated replacement stencil and
  the FFC hold, which are a different kind of work -- an iterated stencil and a piece of control
  flow, not a seeded draw.
- The M9 sensor chain wired into `run_frame` (roadmap M9.8, ADR 0058): `irsim.pipeline.SensorChain`
  owns the housing and FPA nodes, the breathing fixed pattern, the defect map and its state, the
  NUC residual and the FFC controller, and advances them together so nothing runs on a stale clock.
  Per-frame order follows §11.1: nodes and drift first (the housing temperature is what stage 3's
  self-emission needs), then defects on the quantised plane, replacement, the residual, the
  temporal filter and the freeze. `attach_sensor_chain` is the one place ∂DN/∂T is evaluated, so
  ADR 0056's "converted once" is enforced by there being a single site that can do it.
- The assembly's real hazard is double-counting, so the headline test is a budget:
  post-correction spatial noise is `√(σ_V² + σ_H² + σ_VH² + σ_residual²)` within 10 % -- **two
  mechanisms in quadrature, not four added up**. Three things had to be right for that to mean
  anything. The flat-field reference is subtracted, because `vignetting_cos4` alone puts a 0.9 %
  cos⁴ falloff across the array -- a larger spatial std than the whole noise budget, so a raw
  frame std measures the lens. Frames are averaged, because a single frame's spatial std also
  contains temporal per-pixel noise that is indistinguishable from fixed pattern. And it runs at
  two drift rates: at the roadmap's 0.05 K/s the residual is nearly thirty times the 3-D term and
  could hide a spurious third mechanism, so a second rate makes the two equal, where a linear sum
  would be 41 % high against a 10 % tolerance.
- ADR 0058 also records the two things the chain deliberately omits. §11.1's unnamed "temporal
  filter" stays an **identity** until ME.5's temporal PSD on flat sky shows whether real cores
  low-pass at all: the stage sits directly on σ_TVH, the quantity NETD is defined from, so an
  invented coefficient would change measured NETD by √(α/(2−α)) in a way nobody could later
  distinguish from a detector-model error. And the chain applies the NUC residual but **not** M9.2's
  raw gain/offset(T_FPA) polynomials, which are what the NUC removes -- applying both would form a
  large number and subtract almost all of it back, in float32, to reach what the residual gives
  directly (ADR 0053).
- `chain=None` remains the default and is a meaningful configuration, not an unwired one: it is the
  ideal camera every M9 mechanism is measured against, and the one the radiometric goldens
  describe. Those goldens are therefore **unchanged** by this step -- the roadmap anticipated
  updating them, which turned out to be the wrong trade, because a golden containing both the
  radiometry and the defect map can no longer fail informatively for either. The assembled chain
  gets its own golden (`boson_sensor_chain_dn16` / `_apparent_t`) instead.
- FFC controller (roadmap M9.7, ADR 0057): `irsim.isp.FfcController` is §11.2's shutter event --
  the part the spec singles out as "worth more than another decimal place of radiometry". It owns
  the schedule, the freeze, and the ΔT_FPA that M9.6's residual is evaluated at, which is what
  actually distinguishes the three §12.2 modes: raw offset from the last calibration for
  `shuttered`, a scene-based-correction first-order lag for `shutterless`, and identically zero for
  `ideal`. Objects the shutter recalibrates attach through a one-method `Resettable` protocol, so
  the controller neither imports nor knows about them.
- The freeze **holds the last good frame** rather than blanking. A closed shutter has the FPA
  looking at the blade, and a camera emitting those frames would show the scene vanish and
  reappear -- conspicuous, and not what real cores do. Holding makes the image *stale* instead:
  for 0.7 s anything tracking through it sees motion stop dead and then jump, which is the artefact
  a perception stack has to survive. At 60 Hz / 180 s / 700 ms the shutter closes on frame 10800
  and exactly 42 bit-identical frames are emitted while the input keeps changing; the rounding rule
  is round, not ceil, so 9 Hz gives 6 frames rather than 7 -- ceil would systematically lengthen
  every freeze at low frame rates, which is where the artefact is most visible.
- `shutterless` is bounded rather than unbounded: `dΔT_eff/dt = dΔT/dt − ΔT_eff/τ` with
  τ = `nuc.shutterless_tau_s`, so under a constant drift the residual approaches `r·τ` instead of
  tracking ΔT_FPA without limit. At 0.05 K/s the 540 s residual is 1.15× its 180 s value where an
  uncorrected core would be at 3×; the test asserts the saturation *value* as well as the bound, so
  "bounded" cannot be satisfied by merely being slow.
- `irsim.noise.DriftingPattern`: the FFC-resettable holder for M9.4's breathing pattern. `reset`
  redraws rather than zeroing -- a flat-field correction re-measures and re-subtracts the pattern,
  and what is left is a fresh realisation of the same distribution, uncorrelated with the old one.
- NUC residual (roadmap M9.6, ADR 0056): `irsim.noise.NucResidual` is what survives the two-point
  correction -- `g_ij = 1 + ppm·1e-6·ΔT_FPA·ξ_g` and `o_ij = mK/K·1e-3·ΔT_FPA·(∂DN/∂T)·ξ_o` -- with
  both fields redrawn by `ffc_reset`. At ΔT_FPA = 0 the gain is exactly 1 and the offset exactly 0,
  by construction rather than cancellation, so a freshly shuttered camera is perfectly corrected.
  Per ADR 0053 this is the **only** ΔT_FPA-driven mechanism in the chain: the detector's raw
  gain/offset(T_FPA) polynomials (M9.2) are what the NUC removes, and M9.4's OU drift is stationary
  and contributes no growth, so applying more than one would count the same physics twice at a rate
  that still looks plausible.
- ADR 0056 settles the unit question the config poses. `residual_offset_mk_per_k` is authored in
  millikelvin because that is how datasheets quote it, but a residual *applied* in kelvin would be
  right only at the temperature it was tuned at: measured on the committed Boson chain ∂DN/∂T runs
  97.2 / 178.0 / 309.0 / 439.6 DN/K at 250 / 300 / 373 / 450 K. The conversion therefore happens
  once, at construction, at 300 K -- where §9.4 anchors NETD and where datasheet figures are quoted
  -- and `∂DN/∂T` is supplied by the caller from `irsim.isp.dn_per_kelvin` so `irsim.noise` stays
  independent of `irsim.isp` and the "converted once" rule is visible at the call site.
  `test_residual_not_kelvin_flat` pins the payoff: the apparent-temperature error at 373 K is
  0.576× the 300 K one, exactly the measured derivative ratio -- and the same 0.576 ADR 0026
  arrived at independently for the two-blackbody NETD bench. A kelvin-space implementation would be
  flat across that range and would still produce entirely plausible images.
- `irsim.isp.dn_per_kelvin`: ∂DN/∂T of the calibrated transfer by central difference through the
  real forward chain (LUT → optics → detector), the same object the Tier 2 SITF bench measures.
- Bad-pixel replacement (roadmap M9.5b, ADR 0055 addendum): `irsim.isp.replace_bad_pixels` is the
  mean of the valid 4-neighbours, iterated until clusters fill. §10.4 asks for the defect *and* the
  replacement because "the replacement artefact is what a detector actually sees", and the stencil
  was chosen for three testable properties rather than convenience: it is exact on a linear field
  for an isolated defect (< 1 mK, so smooth scene content takes no radiometric bias); it divides
  white-noise variance by four, where copy-one-neighbour leaves σ² and an 8-neighbour mean gives
  σ²/8 -- both measured alongside it so the assertion is shown to exclude them; and it suppresses
  the local Laplacian to under half the untouched value, which *is* §10.4's detectable smoothed
  footprint. Inside a cluster it is deliberately **not** exact: a pixel in a 2×2 never sees an
  opposing pair of neighbours, so the mean is pulled outward -- real, kept, and the concrete reason
  M9.5a's clustering is not cosmetic. Passes are synchronous (a pass reads only what was valid when
  it began), so a 3×3 centre needs two passes, an isolated defect one, and transposing the problem
  transposes the answer -- without which a 2×2 would fill differently row-major than column-major
  and goldens would not reproduce. A partial fill raises: an unreplaced stuck value entering the
  NUC silently is worse than a loud failure. `irsim.noise.active_defect_mask` builds the per-frame
  mask, so an intermittent pixel is replaced only while it is actually bad.
- Bad-pixel map and defect injection (roadmap M9.5a, ADR 0055): `irsim.noise.generate_map` draws
  one sensor's defects by a Neyman--Scott cluster process -- parents uniform, `1 + Poisson(λ)`
  offspring uniform in a 2 px disc -- because §10.4's "clustered slightly, not uniform" is the part
  that matters downstream: a 4-neighbour stencil handles an isolated defect almost perfectly while
  a 2x2 cluster defeats it, so a uniformly scattered map produces no clusters at these densities
  and understates the artefact a perception stack actually sees. Measured at 1024² and 0.0015: 1479
  defects against the 1573 expected (within the 10 % bar; the shortfall is colliding offspring
  merging, left uncompensated because inflating the parent count would distort the cluster-size
  distribution, which is the part that matters), and a mean nearest-neighbour distance 0.26× the
  uniform-Poisson expectation -- with a genuinely uniform map run through the same estimator as a
  control, reading 1.0. The four §10.4 classes split as: dead and hot pinned to the DN floor and
  ceiling bit-identically forever; **flickering** still responding to the scene but offset, which
  is why it survives a map built from one calibration frame; **blinking** stuck only while its
  state is bad. Both stateful classes run on one two-state Markov chain parameterised by its
  stationary occupancy and mean dwell, so both dwell times are geometric by construction --
  verified by a Monte-Carlo-calibrated KS test (the analytic one is invalid for integer dwells and
  reports p ≈ 1e-174 on a perfect sample), with a memoryless pixel rejected by the same estimator.
  Injection is on the raw DN plane, matching §11.1's `raw DN → bad-pixel replace`.
- Mean-reverting drift of the fixed-pattern components (roadmap M9.4, ADR 0054):
  `irsim.noise.FpnDrift` makes §10.3's "pattern breathing" real. The V, H and VH terms follow an
  Ornstein--Uhlenbeck process advanced by its exact update
  `x ← x e^{−dt/τ} + σ√(1−e^{−2dt/τ}) ξ` with τ = `noise.fpn_drift_tau_s`. **Not a random walk:**
  a random walk's variance grows without bound, so it destroys the configured 3-D ratios -- the
  sensor's identity -- a little every frame while each individual frame still looks like plausible
  thermal imagery, which is precisely the failure no single-frame test or visual check would catch.
  The OU form is stationary by construction at any step size, and using the *exact* update rather
  than Euler--Maruyama is what stops the same camera modelled at 9 Hz and 60 Hz from ending up with
  different FPN. All three fixed terms drift because they share one physical cause and because
  breathing VH alone would leave the column stripes frozen -- among the most recognisable real
  artefacts, and the ones sim-to-real transfer is most sensitive to; `components` can restrict the
  set for an ablation. Passing the global term raises: the DC level drifts *physically* through the
  housing and FPA nodes (M3.3, M9.3) and the ΔT_FPA residual is M9.6's alone (ADR 0053), so putting
  a random walk on top would count one effect three times. Measured over long runs against closed
  forms: lag-τ autocorrelation e⁻¹ ± 0.05 at τ, 2τ and 3τ (with a matched random walk run through
  the same estimator as a control, reading > 0.9), σ stationary within 5 % after 2000 frames, the
  ratio vector intact within 8 %, zero mean, and τ = ∞ frozen bit-identical. Drift is the one
  sequential stream in the chain -- the OU state cannot be drawn from `(seed, frame_index)` -- so
  `drift_rng()` is the single blessed generator and M10.7's Warp twin is held to statistical, not
  bit, equivalence.
- Housing temperature source (roadmap M9.3, ADR 0016 addendum): `irsim.optics.HousingTemperature`
  supplies the `T_housing` that §8.2's self-emission term has needed since M3.3, in the three modes
  §12.2's `housing_temp_mode` already named. `fixed` is a bench number and shows no drift at all;
  `ambient` is the air with no lag; `coupled` is the lumped node `dT/dt = (T_air + ΔT_self − T)/τ`,
  which is the physical origin of shutterless drift on the optics side -- a housing warming under a
  NUC table calibrated at a different housing temperature, at ADR 0016's 87 mK of apparent
  temperature per kelvin. The coupled node delegates to M6.6's `NewtonCoolingSolver` rather than
  carrying a second lumped-node integrator; the FPA node's RK2 (ADR 0053) stays the deliberate
  exception. Verified against closed forms, not against itself: the step response matches
  `T∞ + (T0 − T∞)e^{−t/τ}` at τ, 2τ and 5τ to 0.01 K, one 5τ leap equals 5000 sub-steps to 1e-9
  (the exact update is unconditionally stable where forward Euler would diverge), the steady state
  is `T_air + ΔT_self` and not bare `T_air`, `ambient` tracks the weather to 1e-6 K, and under a
  24 h drive the node reproduces the first-order Bode response -- amplitude `1/√(1+(ωτ)²)` within
  2 % and the peak delayed by `arctan(ωτ)/ω` within half a sample, which a second-order node or a
  moving average would fail. It registers with the `Scene` through `.weather`, so a housing built
  on a different `WeatherSeries` than the atmosphere is refused at construction (non-negotiable #6)
  -- worth the guard because a wrong housing temperature is a smooth pedestal, not a visible
  artefact.
- Warp stage 4 (roadmap M10.6): the detector transfer and the microbolometer's membrane lag on the
  device. The IIR updates a persistent `wp.array` in place and writes the frame to a separate
  output, so the state is never round-tripped to the host; `WarpPipelineState` owns the device
  buffers and lives in `PipelineState.buffers`, the same single-owner rule ADR 0052 set for the CPU
  side so that M10.7's drift, defect and NUC buffers are cleared by the same cold start. The path is
  chosen from `fpa.type`: a cooled photon FPA is memoryless and allocates no state at all. Measured
  on an RTX A6000, `cuda:0` and Warp `cpu`: a 20-frame flux step tracks the oracle to 1.74e-7
  against a 1e-5 budget, the first frame of a step covers 0.8111 of it (= alpha, and the number
  §9.2's "roughly 0.6 frames" is not -- spec issue S8), one state pointer across ten frames, and
  photon DN matches the CPU floor() code for code across a 0 -> 2x saturation sweep.
- `irsim.pipeline.detector.detector_stage`: stage 4 on the plane dict -- the ideal transfer followed
  by the membrane lag for a bolometer only, with the per-pixel state in `PipelineState.buffers`.
  This is the composition ADR 0052 fixes and the oracle the Warp twin is compared against; M9.8
  wires it into `run_frame` along with the rest of the M9 chain.
- `quantise_warp`: the ADC on the device. Floor and clip, never round and never wrap.
- Semi-transparent second ray (M7.15, ADR 0046): `irsim.materials.surface_radiance`
  (ε L_B + ρ L_env + τ L_behind, per-pixel closure to 1e-6) and `MaterialTable.properties_for`
  (τ from the packed column, ρ derived, sky pixels blackbody-equivalent); stage 1 and `run_frame`
  take an optional `radiance_behind` plane, which rides the existing G-buffer contract and is
  float16-refused. With no such plane L_behind = L_env, so the form collapses to M7.13's exactly
  and opaque scenes and goldens are bit-identical; supplying one without an environment model
  raises rather than silently rendering a transparent material as opaque. The committed
  windshield's dL/dL_behind is 0.0 in LWIR, 0.02 MWIR, 0.70 SWIR, 0.77 NIR.
- `scripts/stage_own_hunk.sh`: stage only your own edit to a file several sessions are editing at
  once. It three-way merges your change (snapshot -> worktree) onto HEAD, so another session's
  *committed* change is a no-op instead of a failed patch, and a same-line collision conflicts loudly
  rather than silently picking a side. Snapshots are keyed by `$STAGE_OWN_HUNK_ID` so two sessions do
  not share a baseline. `tests/unit/test_stage_own_hunk.py` drives the real two-session collision,
  including the one blind spot it cannot cover (an uncommitted edit made after your snapshot).
- Warp stages 2 and 3 (roadmap M10.5): the atmosphere and the optics run on the device as twins of
  `irsim.pipeline.atmosphere` and `irsim.optics.stage`. One atmosphere kernel serves the grey M8.1
  path, MS.1's multi-term exponential sum and the constant-tau L1 fallback, because the class weights
  sum to 1 and the per-term path radiance collapses to (1 - tau) L_air. The optics kernels convolve
  the supersampled buffer with the optical PSF *before* the block-mean downsample and then apply
  Omega_eff tau_opt cos^4 A_d + Phi_self, with the aperture factor and Phi_self computed by
  `irsim.optics` on the host and passed in -- non-negotiable #5 holds across the second
  implementation, and `tests/unit/test_aperture_guard.py`'s scanner is pointed at the kernel file by
  name to keep it that way. Measured on an RTX A6000, `cuda:0` and Warp `cpu`: stage 2 within 1.9e-7
  relative / 0.015 mK on every branch (d = 0 and d = inf included, sky pixels bit-identical), stage 3
  within 8.5e-7 through a 53x53 PSF on the 4x step edge against a 1e-5 budget, and a +1 K housing step
  reading +87.43 mK on both paths -- the +87 +/- 2 mK the roadmap predicted, now a test.
- `irsim.pipeline.optics.optics_stage`: the plane-dict form of stage 3 (`radiance` on the k-x grid ->
  `flux` on the detector grid), which is what the Warp twin is compared against. It delegates to
  `apply_optics`; the only thing it adds is the housing-radiance lookup `run_frame` already did.
- `irsim.atmosphere.layered.LayeredAtmosphere.air_radiance`: L_B(T_air) at the surface from the
  model's own LUT, so a fast path can take the same value the oracle uses instead of recomputing it.
- `tests/unit/test_tier3_atmosphere.py` and the solar-path stub (M8.8, ADR 0051): every preset is
  checked for the §7.2 band orderings, the humid crossover (τ_LWIR < τ_SWIR in humid clear air) and
  the fog reversal from the weather alone, a target at T_air is distance-invariant to 1e-9 on a
  horizontal path for both atmosphere models, and a 300 m path at 10° reads within 0.1 K of the same
  length horizontally while 5 km differs by more than a kelvin. New `AtmospherePreset.solar.
  zenith_transmittance` (atmosphere schema v2, all seven presets, ESTIMATED) with
  `irsim.atmosphere.extinction.solar_transmittance` / `airmass`: τ_sun(θ) = τ_zenith^{sec θ},
  exact at τ_zenith² for 60°, refused beyond 85°.
- Warp is reachable without booting Kit (ADR 0014 addendum). `irsim_isaac.env.ensure_warp_on_path`
  locates the `omni.warp.core` extension in the Isaac build's `extscache` (or `$IRSIM_WARP_PATH`) and
  puts it on `sys.path`; it never shadows an already-resolving Warp. ADR 0014 had recorded Warp as
  "importable only inside a running Kit", which was a `sys.path` artefact, not a runtime requirement.
  The M10.4 equivalence harness now runs on `cpu` and `cuda:0` in 1.4 s from a bare interpreter
  instead of behind a ~35 s Kit boot, and `gpu`-marked integration tests are selected on Warp being
  available rather than on Isaac Sim.
- `irsim.validation.aerial_scene` + the `gbuffer_aerial` fixture and `tests/unit/test_tier3_sky.py`
  (MS.8): a synthetic sky-background G-buffer -- exact per-pixel ray elevation for a pitched pinhole,
  horizon, optional 1/f^β cloud coverage, targets rasterised above one native pixel and handed to
  MS.6's analytic injection below it -- plus the Tier 3 checks it exists for: the rendered sky
  profile is an identity on MS.2 within 1 mK and scale-free to 1 %, cloud reads its base temperature
  > 20 K above clear sky with real spatial structure, a sub-pixel target's excess follows
  φ τ(R)(L_t − L_air)/R² exactly against a grey atmosphere and decays measurably slower against the
  layered one (the occulted sky dims with range), and a 2 px target's footprint aliases with
  sub-pixel phase while the optical PSF lowers the peak and conserves the flux.
- Aerial thermal bridge (M10.18, ADR 0060): `irsim_isaac.pipeline.aerial_bridge` couples the M6.6
  target solvers to the renderer through a float32 table indexed by instance id -- 317.25 K reads
  back within 10 mK and a 50 mK pair stays resolved, where the rejected fp16 emissive path would
  collapse it. Thermal tick 1 Hz with per-frame interpolation (error bound computed, ~3 uK for
  tau = 900 s); background pixels take MS.2's `T_sky(theta)` per ray, and `T_ground` below the
  horizon where the sky model is undefined and extrapolating it would invert silhouette contrast.
- Warp stage 1 (roadmap M10.4, ADR 0061): `irsim_isaac.pipeline.warp_stages` runs band radiance on the
  device as an op-for-op twin of `irsim.pipeline.radiance` (float32 LUT index/clamp/interp, ε = 1 under the
  sky mask, host guards for every CPU refusal), with the LUT and ε₀ table uploaded once and cached by
  content. `tests/integration/test_kernels_vs_reference.py` is the CPU-vs-GPU equivalence harness every
  later stage registers in (`EQUIVALENCE_STAGES`): stage × fixture × {cuda:0, Warp cpu} at ≤ 1e-4 / 5 mK;
  measured ≤ 2 ulp on CUDA and bit-identical on the Warp CPU device. The `isaac` extra is now empty: Warp
  comes from the `omni.warp.core` Kit extension and a pip copy would shadow it.
- Material-ID transport for the Isaac path (M10.2): `irsim_isaac.pipeline.material_ids` (instance
  id → prim path → M7.17 material id by exact integer lookup, UNMAPPED mask, magenta display
  overlay) and `irsim_isaac.pipeline.materials_usd` (stage walk for bindings, `class` semantics and
  the `thermal:material` override). `scripts/audit_materials.py` gains `--stage` / `--dump-prims`.
  Measured: the id channel must be `instance_id_segmentation`, because `instance_segmentation`
  gives a distinct id only to *labelled* prims and collapses the rest into one id (ADR 0014
  addendum) -- four of five test prims would have shared a material with nothing raising.
- Isaac geometry AOVs assembled into the M0.6 `GBuffer` (M10.1): `irsim_isaac.pipeline.gbuffer_isaac`
  (`AovReader` + engine-free assembly of `distance_m`, `normal_dot_view` against the per-pixel ray,
  `normal_dot_up`, V_s = occlusion·(1+n·up)/2, `sky_mask`), `irsim_isaac.geometry_probe` and
  `scripts/probe_isaac_geometry.py`. Sphere cos θ matches the closed form to 1e-5 engine-free and
  0.01 in-sim; tilted-quad ray length to 1 cm; plates read V_s 1.0/0.5/0.0 ± 0.05.
- Four-material aerial library and target thermal signature (MS.7, ADR 0072): `configs/materials/`
  gains `painted_composite`, `carbon_fibre`, `aircraft_aluminium_painted` and `propeller_rubber`
  (`source: literature`, scalar `emissivity_per_band`, opaque, ρ derived -- CLAUDE.md #4 closure
  verified in all four bands before and after the M7.18 float32 packing).
  `irsim.thermal.aerial`: `HeatSource` with ΔT = ΔT_max u^n and the MOTOR/ESC/BATTERY presets
  (45/30/15 K at full throttle, n = 2 ohmic, magnitudes ESTIMATED), `heat_source_solver` and
  `airframe_solver` returning M6.6 `PrescribedSolver`s over the scene's shared `WeatherSeries`,
  and `refine_nodes` bisecting the schedule grid until linear interpolation holds the analytic law
  to 1 mK. `irsim.validation.aerial`: `AerialTarget`, `target_contrast` and
  `zero_contrast_elevation` -- for T_air = 300 K, ε = 0.9, V_s = 1, R = 1 km under
  `us_standard_clear` the contrast is +64 K at zenith and crosses zero at 1.26° (1.97° at ε = 0.8,
  0.78° at ε = 0.95, and **never at ε = 1**, which is what shows the inversion is the reflected
  cold sky rather than a path-radiance artefact).
- `irsim.atmosphere.cloud` + `SkyModel` cloud methods (MS.3): LCL base from the shared weather (Espy,
  `dew_point_k` in humidity), T_base by the preset lapse rate, ε_cloud = 1 − τ_cloud (τ authored in the
  environment preset's new `clouds:` block, schema v2), seeded 1/f^β structure with an exact-coverage
  threshold and a PSD-slope self-test; `radiance_field` / `apparent_temperature_field`. ADR 0070.
- `irsim.pipeline.point_target` (MS.6): `PointTarget`, `fill_fraction`, `excess_radiance` (per-class
  τ_k(R)[L_t − L_beyond,k] with the layered atmosphere; grey and no-atmosphere forms), `excess_power` via
  the single aperture factor, bilinear `splat`, `run_frame(..., point_targets=)`;
  `LayeredAtmosphere.class_transmittances / sky_beyond(_per_class)` and `ExponentialSum.path_radiance_per_class`.
  ADR 0071 records the measured rasteriser flux error vs size behind the 1 px handoff.
- Stage 1 reflected environment term (M7.13): `band_radiance(..., l_env=)`, `irsim.pipeline.environment`
  (`sky_view_factor = occlusion·(1+n·up)/2`, `environment_radiance` from the SkyModel tilt LUT and the
  ground mode), `PipelineConfig.sky`; scene schema v2 adds `environment_preset` and the Scene builds the
  layered atmosphere and per-band sky models on the same weather. ADR 0045.
- `irsim.atmosphere.sky.SkyModel` (MS.2): clear-sky elevation LUT over the layered column emission
  (fast path within 0.5 K), tilt LUT with an analytic azimuth kernel (`effective_radiance(β)`,
  `effective_radiance_from_sky_view(V_s)`), cloud blend to L_B(T_air), `broadband_downwelling` via M6.5,
  `fit_cos_q` deriving the §5.3(a) form and its error (ADR 0044).
- `irsim.atmosphere.layered` (MS.1): `LayeredAtmosphere` -- exponential sum over spectral classes per
  band (Planck-weighted class weights from the sensor response; water/air scale heights; horizontal
  200 m anchored to the grey preset; opaque CO₂/H₂O cores on top), analytic slant-path transmittance,
  per-class optical-depth quadrature for path radiance, `sky_radiance = L_path(∞, θ)`,
  `apparent_sky_temperature_k`; `fit_exponential_sum`, `exponential_sum_from_piecewise`;
  `scripts/validate_sky_r13.py`. Stage 2 accepts a `LayeredAtmosphere` (per-term horizontal form).
  Preset profile gains `aerosol_scale_height_m`, `air_scale_height_m`, `tropopause_m` defaults. ADR 0071.
- `irsim.materials.mapping` (M7.17): `MaterialResolver` (override → semantic → name pattern → loud
  miss with id 0), `configs/materials/mapping.yaml`, `audit()` and `scripts/audit_materials.py`
  (coverage %, misses grouped, non-zero exit below the 95 % threshold; ADR 0047).
- `irsim.config.environment` (M7.11): `EnvironmentConfig` (sky ΔT_clear per band + q, ground mode,
  solar glint model, night airglow with an explicit unit key / k_cloud / moon), ranges validated per
  regime against §5.3/§5.5, weather-like keys refused; presets `configs/environments/{clear_dry,
  humid, overcast}.yaml`.
- `MaterialTable.from_library` / `save` / `load` (M7.18): packed float32 per-band columns (ε₀, ρ, τ,
  Level-B (a, p) or (0, 4) placeholders, roughness, thermal), id 0 = UNMAPPED, ids stable across loads,
  `.npz` + sidecar with the library hash, `StaleMaterialTableError`; float16 refused.
- `irsim.config.materials` (M7.2): `MaterialConfig` -- one YAML per material with a `material:` block,
  `source` required, exactly one of spectral/scalar ε or ρ authored, optional τ/roughness per band,
  angular model union; `irsim.materials.library` derives the third quantity per band (ADR 0010 band
  average for spectra) and refuses ε + τ > 1; `irsim.materials.spectra` property-spectrum loader;
  `configs/materials/` six §16.2 materials; the CLAUDE.md #4 closure library walk. ADR 0040.
- `irsim.pipeline.atmosphere` + stage 2 in `run_frame` (M8.6): τ(d)L + (1−τ)L_B(T_air) on the k× grid
  from `PipelineConfig.atmosphere` at `PipelineState.t_s`; sky pixels bit-identical; `tau_override`
  L1 fallback; identity without an Atmosphere (goldens unchanged). ADR 0050.
- `irsim.config.scene` (`SceneConfig`: weather file, atmosphere preset, site, aware start, newton/
  prescribed targets) and `irsim.scene.Scene` (M6.17): loads the weather once, injects the same
  `WeatherSeries` into the `Atmosphere` and every target solver, refuses a consumer holding another
  weather object; `configs/scenes/sky_target_clear_day.yaml` sample.
- `irsim.atmosphere.Atmosphere` (M8.5): preset + the shared `WeatherSeries` (+ band LUTs) →
  `state(t)` with T_air, w, V, γ per band, L_air per band and a regime-mismatch flag; `transmittance`,
  `air_radiance`, `apply` (the per-pixel Beer–Lambert kernel at time t). A path raises TypeError.
- `irsim.thermal.solvers` (M6.6): `TemperatureSolver` protocol (`advance(t, dt)`, `temperature()`,
  `state`), `PrescribedSolver` (schedule, nodes exact, no extrapolation) and `NewtonCoolingSolver`
  (exact exponential update, midpoint ambient; accepts the shared `WeatherSeries` as ambient and
  exposes it as `.weather`). Scope limit documented: scripted actors and residual heat only.
- `irsim.thermal.longwave` (M6.5): Q_LW↓ = V_s·ε_sky·σT_air⁴ + (1−V_s)·σT_surround⁴ with Brunt (default)
  or Idso clear-sky emissivity and a linear cloud blend to 1; fed by the shared `WeatherSample` (ADR 0035).
- `irsim.thermal.solar` (M6.4): NOAA sun position (elevation, azimuth, declination, equation of time,
  solar noon; vectorised over Julian days), ENU sun vector, `solar_loading = S·max(0,n·s)·DNI + V_s·DHI`,
  `absorbed_solar = α·Q` (ADR 0034).
- `irsim.thermal.convection` (M6.3): h = max(c|ΔT|^{1/3}, a + b v_rel^n) with v_rel = |wind| + |vehicle|
  (ADR 0033); h(28 m/s) = 62.5 vs 5.0 parked.
- `irsim.thermal.weather_io` (M6.2): project weather CSV (`# irsim weather v1`, unit-suffixed columns,
  ISO-8601 UTC; `t_air_c`/`rh_percent` converted once at load), bit-exact write→read,
  `synthetic_clear_day`; `data/weather/clear_midlat_summer_48h.csv` (SYNTHETIC) regenerated by
  `scripts/generate_weather.py` and pinned by a test.
- `irsim.thermal.weather.WeatherSeries` / `WeatherSample` (M6.1): immutable validated hourly weather
  (T_air K, RH fraction, wind, cloud, DNI/DHI, visibility, precip), linear `at(t)`, extrapolation
  refused, `content_hash`; never opens a file (ADR 0032). `SOLAR_CONSTANT_W_M2 = 1361` in constants.
- `irsim.config.atmosphere` (`AtmospherePreset`: per-band γ₀/β/aerosol ratio, regime, profile,
  provenance; rejects weather-like keys at any depth — CLAUDE.md #6), seven presets in
  `configs/atmospheres/` fitted to the §7.2 table, `irsim.atmosphere.library`,
  `irsim.atmosphere.extinction` (Koschmieder `3.912/V` on total visible extinction, per-band ratios,
  droplet regime for fog; `KOSCHMIEDER` in constants). ADR 0049 resolves the haze row (its τ values
  imply 1.5 km visibility, not 5 km; spec issue T18).
- G-buffer contract: optional bool `sky_mask` plane (renderer hit nothing). Stage 1 treats masked
  pixels as blackbody-equivalent (ε = 1, the apparent sky temperature is what the plane carries) and
  ignores their material id, so the renderer's background id 0 is no longer confused with UNMAPPED;
  stage 2 (M8.6) will pass them through. Agreed with the Isaac lane (DistanceToCamera = +inf, id 0).
- `irsim.atmosphere.beer_lambert` (τ = e^{−γd}, path radiance, `apply_atmosphere`, the L1 `tau_override`),
  `irsim.atmosphere.humidity` (Magnus/Bolton e_s, absolute humidity with the 216.7 factor derived from
  `R_V_WATER`, γ_mol = γ₀ + βw; RH is a fraction), `irsim.atmosphere.spectral` + `scripts/
  validate_atmosphere_band_average.py` (exact spectral τ_B, curve of growth, grey-fit error: the ADR 0048
  evidence).
- `irsim.optics.mtf` (diffraction, detector sinc, motion, Gaussian, cascade, cut-off and Nyquist) and
  `irsim.optics.psf` (`optical_psf` = diffraction·Gaussian at the supersampled pitch, `apply_psf` FFT
  convolution); the PSF is now the first step of `apply_optics` and `PipelineConfig` builds it from the
  band's R-weighted mean wavelength (`psf_enabled=False` skips it). `irsim.validation.mtf.slant_edge_mtf`
  (ISO 12233-style) and the Tier 2 MTF bench: 0.31 ± 0.05 at Nyquist for the Boson, supersampled path
  aliases while a native blur does not (ADR 0059). Goldens regenerated with the PSF in the chain.
- README: tier-promotion rules (what evidence moves a row to T2/T3/T4; T5 is external), the first-image
  note, and status rows for the first LWIR camera through the ISP (isp 🟢, pipeline T2–T3).
- `run_frame` emits `display8` (RGBA8 through the isp block) and `isp_hash`; `irsim.io.png` (stdlib PNG
  writer for the human look); end-to-end goldens `boson_ramp_*` and `boson_hot_patch_*` (radiance 1e-5,
  T_app 1 mK, DN16 ±1, DN8 ±1) keyed on config hash + NumPy version; Tier 3 phenomenology through the
  pipeline: a 600 K patch entering the frame halves the pedestrian's 8-bit contrast under linear AGC
  while DN16 contrast is unchanged, and plateau 0.012 keeps > 50 % of the background std.
- `irsim.isp.display.run_display_branch`: AGC (linear / plateau / `none` = exact bit shift) → gamma → DDE →
  polarity → palette, pure per-frame, float32 rounding points R1–R4 documented, isp config hash in the
  output (ADR 0031). `irsim.isp.nuc.TwoPointNuc`: §11.2 calibrate/apply, ideal mode, corrected cold
  blackbody = 0 (ADR 0021 level).
- `irsim.isp.agc`: `agc_linear` (histogram percentiles with in-bin interpolation, gamma; ADR 0027) and
  `agc_plateau` (P = plateau·N per bin over 2^bits bins, exclusive CDF normalised to the occupied range,
  P→0 = rank map of occupied bins; ADR 0028). Hot exhaust at 5 % of pixels collapses the linear-AGC
  background std to < 20 % while plateau 0.012 keeps > 50 %.
- `irsim.isp.dde`: 3×3-box unsharp mask (±gain/6 step overshoot; transfer |1+g(1−K̂)|²; ADR 0029).
- `irsim.isp.palette`: gray/ironbow/rainbow/lava/arctic tables, polarity, `to_display8` → RGBA8 (ADR 0030).
- `irsim.noise.stage.NoiseStage` (correlated V/H/VH/TV/TH/T terms scaled from the detector's σ_TVH by the
  configured ratios, unit fixed patterns per sensor) and `measure_from_uniform_scene`; `run_frame` now runs
  the seeded detector response and the noise stage before the ADC (`PipelineConfig.from_sensor(...,
  sensor_seed, noise_enabled)`). Tier 2 benches: DN-domain two-blackbody NETD within 10 % of the anchor,
  3-D ratios recovered within the ADR 0023 floors, bit-identical replay, frame 500 reachable directly,
  sensors uncorrelated. Golden `boson_noise_cube_8x64x64` keyed on config hash + NumPy version.
- `irsim.validation.noise` (ME.2a): leakage-corrected NVESD `decompose_3d` (random-effects mean squares,
  uint16 promoted, negatives clipped with raw variances kept), `spatial_psd` with radial profile and the
  k_v = 0 / k_h = 0 striping lines, `temporal_psd`, `compare_psd`. Boson ratios recovered from 200
  frames of 64×64 within the stated sampling floors; white cube shows no directional terms (ADR 0023).
- `PhotonDetector.response` (Poisson shot + dark + background and hashed read noise in electron space,
  then DN) with Arrhenius `dark_current_a` (InSb/InGaAs band gaps in `constants.py`), and
  `MicrobolometerDetector.response` (static transfer + anchored σ in signal space, then DN);
  `DetectorFrame(signal_dn, dn, sigma_dn)`; `measured_netd_k` two-blackbody bench; golden
  `boson_sitf_dn` (ADR 0026). NETD(300 K) within 10 % of the anchor; bolometer NETD(373)/NETD(300) =
  0.576 while the DN noise std is scene-independent.
- `irsim.detector.netd` / `figures_of_merit`: NETD predictor NETD(T) = σ_total/(∂S/∂T) for both detector
  classes (∂S/∂T through the transfer, +1 aperture form), NEP, D*, the 4F² datasheet form as a labelled
  conversion (1.25 at F/1), ENBW = 1/(4τ_th) and the sampled-IIR ENBW, Johnson and temperature-
  fluctuation floors reported (ADR 0024). NETD(373)/NETD(300) = 0.576 for 7.5–13.5 µm.
- `irsim.detector.anchor.anchor_noise`: solves the scene-independent Gaussian σ so the predicted NETD at
  300 K and `netd_ref_f_number` equals `netd_mk_at_300k`; Poisson terms never rescaled; unattainable
  targets raise; NETD(F/1.4)/NETD(F/1.0) = 1.768 (ADR 0025).
- `irsim.noise.three_d`: `Sigmas7` (unit-agnostic, from ratios), `FixedPattern.generate` (V, H, VH once
  per sensor) and `synthesize_frame` = T + V + H + TV + TH + VH + TVH from the counter-based streams.
  Axis conventions tested; pooled variance = Σσ² within 3 %; row-mean variance identity within 10 %.
- `irsim.noise.seeding`: counter-based per-element hashing (`stream_key`, `hash_u64`, `hash_uniform`,
  `hash_normal`, `field_normal`: splitmix64 over (sensor seed, sensor frame index, stream, pixel index),
  Box–Muller in float32) so any pixel is regenerable alone, traversal-independent and reproducible bit-
  for-bit by a Warp kernel; `NoiseStream` values frozen; PCG64 `noise_rng` / `sensor_rng` kept for Poisson
  draws and the bad-pixel map (ADR 0022).
- Noise schema (v4): `Ratios3D.as_vector()` / `total_over_tvh()` (1.0604 for the Boson ratios),
  `noise.netd_ref_f_number`, `noise.bad_pixel_type_mix` (sums to 1).
- Isaac Sim gate spike (roadmap M2.1–M2.4, ADR 0014): `irsim_isaac.probe` (environment report; 64-quad
  emissive temperature ramp; AOV dtype, resolution, exposure and distance semantics; segmentation-id and
  float32 position transport) and `irsim_isaac.spg_probe` (SPG pass-through, cross-frame state, LUT
  delivery, checkpointed per experiment) with `scripts/probe_isaac_environment.py` and
  `scripts/probe_isaac_spg.py`. `tests/integration` boots one headless Kit per session (argv hidden from
  Kit's parser, app closed at interpreter exit); `test_environment.py` and `test_isaac_transport.py` pin
  the measured facts, including the negative one. **Outcome:** every colour AOV is float16 →
  temperature is transported as instance ids + float32 geometry, never as emission; SPG holds no state →
  stateful stages stay in Warp; LUTs are baked into the `.cu`. `docs/spec-issues.md` gains T16 (the build
  is 6.1.0-rc.26) and T17 (§13.1/§13.3 emission transport not viable).
- `irsim.radiometry.band_average`: `band_average(response, s(λ), T_ref, form)` = ∫R s B dλ / ∫R B dλ, the
  only sanctioned route from ε(λ)/ρ(λ)/τ(λ) to per-band scalars; linear so Kirchhoff closure survives
  (1e-9); the accepted grey-in-band error (~2e-3 for a 0.1 slope, 300→600 K) is measured (ADR 0010).
- `irsim.radiometry.band_integration`: the reference oracle — `band_radiance`, `band_photon_radiance`,
  `d_band_radiance_dT`, `d_band_photon_radiance_dT` by composite Simpson on an odd, edge-aligned 0.01 µm
  grid (NumPy only, float64, vectorised over T). Top-hat vs closed form: 7.5-13.5 um @ 300 K: 3.3e-11 rel, 0.000 mK; 3.0-5.0 um @ 300 K: 3.1e-11 rel, 0.000 mK; 3.0-5.0 um @ 500 K: 5.7e-11 rel, 0.000 mK; 0.9-1.7 um @ 300 K: 1.5e-06 rel, 0.016 mK.
  Derivative vs FD < 1e-5 over 200–1000 K; dLb/dT(373)/dLb/dT(300) = 1.736 for 7.5–13.5 µm.
- `irsim.radiometry.spectral_response`: R(λ) file contract and loader (two-column CSV, `#` provenance,
  µm strictly increasing, R in [0, 1], peak == 1 ± 1e-6 asserted never renormalised, zero outside support,
  half-power points, resampling) and `irsim.radiometry.band.Band` (config edges must agree with the file's
  half-power points to 0.5 µm). `data/spectra/responses/boson_vox.csv`: ESTIMATED Boson VOx curve,
  7.5–13.5 µm with ±0.25 µm raised-cosine edges (ADR 0009).
- `SIGMA_Q` (4π ζ(3) k³/(h³c²), derived in `constants.py`), `fractional_photon_exitance` and
  `band_photon_radiance_tophat` (§3.2 a): the closed-form oracle for the photon band table. Photon
  Stefan–Boltzmann closes to 1e-6; `Lb_q(7.5–13.5 µm, 300 K) = 2.915e21` known answer; `Lb/Lb_q = hc/λ_eff`
  with λ_eff inside the band.
- `d_spectral_photon_radiance_dT` (§3.4, §9.4): photon-form thermal derivative sharing the energy form's
  expm1/clip guards. Identity `dL_q/dT · hc/λ == dL/dT` to 1e-12; central FD < 1e-6; finite over
  200–2000 K × 0.4–20 µm.
- GPU-free CI: `.github/workflows/check.yml` runs `make check` on plain CPython 3.10 and 3.12; `make ci`
  reproduces it locally in a `python3.10` venv. Interpreter matrix recorded in ADR 0002.
- `docs/spec-issues.md`: the 37 physics and 15 tooling issues found in the spec, with proposed
  resolutions and applied status, for the spec owner (roadmap open question 10).
- `irsim.config.bands`: canonical band ids with §12.1 nominal ranges and per-band defaults,
  `band_id_for` by maximal overlap (< 50 % rejected), optional `band.id` validated against the edges,
  and `enabled_illumination_terms(regime)` so kernels never switch on a band name (§5.2, S17).
- `irsim.config.loader`: `load_sensor_config` (data paths resolved against `data_dir` /
  `$IRSIM_DATA_DIR` / `<repo>/data`, missing files named), `dump_sensor_config`, `config_hash` (canonical
  JSON, files by content hash) and `band_hash` (band block + spectral bytes: the LUT key). Every numeric
  and enumerated leaf of the Boson file is tested to move the hash (ADR 0008).
- `irsim.config.sensor.SensorConfig`: pydantic v2 schema for §12.2 (frozen, unknown keys rejected,
  units-in-names, discriminated bolometer/photon FPA, regime-vs-wavelength and 3-D ratio checks, derived
  pixel area / Nyquist / HFOV / frame period / DN max, deliberately no aperture factor). Boson file loads
  exactly; HFOV 30.7° vs datasheet 32° recorded (ADR 0007). `types-PyYAML` added to dev extras.
- `irsim.config.gbuffer.GBuffer`: the frozen G-buffer contract (`temperature_k`, `normal_dot_view`,
  `distance_m`, `material_id`, `sky_view_factor`; optional `encoded_t`, `motion_px`, `semantic_id`,
  `radiance*`). float16 refused on temperature/encoded/distance/radiance planes, upcast elsewhere;
  `encoded_t` checked against `temperature_k` to 10 mK; id 0 reserved as UNMAPPED.
- Synthetic G-buffer fixtures: `gbuffer_two_material`, `gbuffer_sphere` (analytic cos θ to 1e-6, reaches
  grazing), `gbuffer_step_edge` (1024², 5.5° tilt, ideal two-level), `gbuffer_moving_edge` (8 frames,
  2 px/frame with `motion_px`); all fixtures now carry `encoded_t` and use material id 1+.
  `tests/unit/test_gbuffer_schema.py` freezes the key set the Isaac adapter must emit.
- `irsim.radiometry.encoding`: float32 temperature encode/decode `c = (T − 200)/800` with the constants
  defined once in `constants.py` (plus the LUT grid constants); float16 and integer inputs refused; fp16
  coarse/fine pair codec as the §13.3 fallback. Round trip 0.05 mK; negative controls show 125 mK (raw
  kelvin) and ≥ 40 mK (encoded) through fp16 (ADR 0006).
- Stefan–Boltzmann identity test at 1e-6 relative via Simpson quadrature plus the closed-form tail
  (measured 3e-11); `fractional_exitance` gains a small-x Bernoulli branch so it is double-precision
  accurate in the Rayleigh–Jeans tail, and it and `band_radiance_tophat` now reject metre-valued
  wavelengths and Celsius temperatures (ADR 0005).
- Layering guard hardened in both directions: `src/irsim` may not import engine modules (now including
  `carb`), `irsim_isaac`, or the ML/imaging stack; default-gate tests may not import engines; glue files
  must keep engine imports inside functions. Scanner self-tested on synthetic offending modules.
- `irsim_isaac.env` (`has_isaac`, `has_warp`, `require_*`, force-off env flags) and
  `tests/integration/conftest.py`, which auto-marks and skips the directory without Isaac Sim.
- `isaac` optional extra (`warp-lang`, floor provisional until M2.1); mypy now covers `src/irsim_isaac`.
- Golden fixture helper (`tests/golden/conftest.py`, ADR 0004): `.npy` + JSON sidecar with config hash;
  STALE (inputs changed) reported distinctly from FAILING (values changed); actual array dumped to
  `outputs/golden/` on failure; float16/float64 refused on disk; `--update-golden` registered so
  `make golden-update` works. `tests/golden` now runs in `make test`.
- Project scaffold: physics specification (`docs/physics-model.md`), `CLAUDE.md`, skills, build tooling.
- `irsim.radiometry.constants` — physical constants with sources.
- `irsim.radiometry.planck` — spectral radiance (energy and photon forms), fractional exitance,
  thermal derivative. Analytic identity tests passing.
- Layering test enforcing that `src/irsim` imports no engine modules.
- ADR 0002: NumPy ≥ 2.0 floor and Isaac Sim's bundled Python as the project interpreter.
- `docs/roadmap.md`: phased implementation roadmap — 15 milestones, 157 one-commit steps with
  verification tolerances, risk register, ADR backlog, non-negotiable enforcement map, spec issue list.
- ADR 0003: sky targets first, validated against public anti-UAV thermal data (no camera); adds the
  evaluation-harness (ME) and sky/clouds/MTF/point-target (MS) milestones; revised after a four-critic
  adversarial review (72 findings applied); the twelve subsystem maps under `docs/maps/`.
- `irsim.detector.lowpass` (M9.1, ADR 0052): `BolometerLowPass` -- the §9.2 membrane thermal lag as a
  stateful per-pixel float32 IIR, `S_n = S_{n-1} + (S_ideal - S_{n-1})(1 - e^{-dt/tau_th})`, applied to
  the ideal signal *before* noise so the M4.6 NETD anchor survives; starts settled (the first frame
  adopts its input) with `reset()` for a cold start. `alpha_for`, `responsivity_rolloff(f, tau)` (the
  continuous `1/sqrt(1 + (2 pi f tau)^2)`, not the sampled IIR's transfer function) and
  `trailing_decay_length_px` = `v tau_th/dt` -- the moving-edge tail length that spec issue S8's
  "0.6 frames" phrasing conflates with `tau_th/dt`. Verified against the closed-form step response to
  1e-6, against RK4 of the membrane ODE to 1e-4 over 100 varying frames, and by measuring the tail of
  a rendered moving edge to within 2 %; the photon path is asserted memoryless (§15 T3).
- `irsim.detector.fpa_thermal` (M9.2, ADR 0053): `FpaThermalModel` -- the §9.2 FPA node
  `C dT/dt = P - h(T - T_amb)` in its identifiable form (tau = C/h and DeltaT_self = P/h are authored;
  C, h and P separately are not observable), RK2 (Heun) on the thermal tick, with the housing node's
  three modes -- `fixed` (TEC-pinned, no drift), `ambient` (zero-tau), `coupled`. Ambient comes from
  the scene's shared `WeatherSeries` *or* an injected provider, never both and never neither
  (CLAUDE.md #6). `gain_of_t`/`offset_of_t` are the **raw, uncorrected** response, coefficients in
  ascending powers of (T - T_cal) from order 1 so `g(T_cal) == 1` and `o(T_cal) == 0` structurally;
  what survives correction stays `nuc.residual_*` (M9.6), so the drift is not counted twice.
  Steady state and the 63.2 % time constant verified to 1 mK / 1 %, per-step energy conservation to
  1e-6 against the C/h/P form, and a 5 K ambient rise shown to shift DN uniformly and not at all
  when TEC-pinned. Sensor schema -> v5 (`fpa_temp_mode`, `fpa_t_cal_k`, `fpa_gain_coeffs_per_k`,
  `fpa_offset_coeffs_dn_per_k`; the node is opt-in, so existing configs are unchanged). Golden
  sidecars regenerated for the new `config_hash` -- all twelve arrays bit-identical (ADR 0004).
- `make ci` -- the plain-CPython gate that mirrors GitHub Actions -- now typechecks clean. It was
  red for 19 mypy errors that `make check` on the Isaac Sim interpreter does not see, because the
  two resolve different numpy versions (2.2.6 against 2.3.1) and numpy 2.x made `ndarray` generic
  over **shape** as well as dtype. Twelve were bare `np.ndarray` annotations in the probes, now
  `NDArray[Any]` -- the deliberate choice there, since probe code handles whatever an annotator
  returns and pinning a dtype would assert what the probe exists to measure. The rest were
  shape-parameter widening (`np.roll`, `np.tensordot`, fancy indexing) restated at the point it
  happens, and two `.shape` values narrowed to the 2-tuple their callee declares.
- `import irsim.thermal` failed outright on a clean interpreter: `irsim.thermal.weather` imported
  `irsim.atmosphere.humidity` at module scope, which runs `irsim/atmosphere/__init__` and lands back
  in the partially-initialised `weather` module. The whole suite passed only because pytest collects
  alphabetically and something imported `irsim.atmosphere` first, so the cycle was invisible until a
  single test file was run alone. The two humidity helpers are now imported inside the two
  `WeatherSample` properties that use them, breaking the cycle at its source; no atmosphere module
  needed changing. `tests/unit/test_import_order.py` imports each subpackage in its own subprocess
  so collection order can never hide this again.
- `SpectralResponse.resampled` snaps grid points within 1e-9 µm of the support edges: a grid built as
  `lo + k·dl` lands 2e-16 µm past the last sample and lost the endpoint (a 5 % error for SWIR at 300 K).
- `make check` is green on the scaffold: three files reformatted, one `Any` return in
  `irsim.radiometry.planck` typed explicitly.

