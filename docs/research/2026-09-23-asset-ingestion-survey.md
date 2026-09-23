# Asset ingestion survey, 2026-09-23

**What would it take to drop an arbitrary 3D model into a scene and have it render correctly in IR?**

Produced by a repository audit plus a web survey, in response to the owner's question about importing
FBX models whose parts are not separated — "the door, the windows and whole buildings is a single
object but many different materials are there" — and their hypothesis that a DCC such as Blender is
needed in the middle.

This is a **snapshot of what was readable on 2026-09-23**, not a maintained document. Every external
claim below is marked **[C]** (confirmed by reading the primary source) or **[U]** (not confirmed —
do not build on it without checking). Check a source before relying on it; prefer the primary link.

No roadmap row, ADR or code change is proposed here. This file is the evidence base for that
discussion, not the decision.

---

## 1. The short version

1. **There is no asset import in this project at all.** Every piece of geometry in every scene is
   generated in Python. This is not a gap in the import path; there is no import path.
2. **The material mapping layer (ADR 0047) is the right design and is not the blocker.** It resolves
   per *prim*, which is one level coarser than a single-mesh multi-material asset needs.
3. **A single-part building breaks three things, not one** — material, temperature and thermal
   parameters — because all three are bound per prim.
4. **A Blender-exported multi-material mesh fails silently, not loudly.** It defeats the UNMAPPED
   sentinel that ADR 0047 exists to provide. This is the single most important finding here.
5. **The mechanism that solves it already exists**, built for a different reason (ADR 0110), and it
   needs nothing from the renderer.
6. **Blender is genuinely useful in the middle, but not as a converter.** Conversion is a CPU-only
   CLI step. Blender's job is repair and authoring.

---

## 2. Where the repository stands

### 2.1 Geometry is authored, never imported

| evidence | file |
|---|---|
| ~18 `UsdGeom.Cube`/`Cylinder` prims built from DJI's published spec | [`phantom3.py:129`](../../src/irsim_isaac/phantom3.py) `phantom3_parts()` |
| The only code that writes an airframe to a stage; sets `thermal:material` per prim | [`airframe.py:70,125`](../../src/irsim_isaac/airframe.py) `author_parts()` |
| `MeshSpec.shape` is a two-value `Literal` — `cylinder` or `sphere`, both generated | [`scene.py:400`](../../src/irsim/config/scene.py) |

No scene config in `configs/scenes/` (16 files) references a `.usd`, `.fbx`, `.obj` or `.gltf`. The
scene schema (v15) has no asset-reference field. Nothing outside `scripts/audit_materials.py --stage`
calls `open_stage` or `AddReference`.

The gap is acknowledged in exactly one place, with no owner:

> **The shape is generated here, not read from USD.** … Ingesting a real asset's triangles is the
> Isaac side's job — `probe_warp_prim.py` measured what it takes … **and is the follow-on to this
> row.**
> — [`scene.py:410-415`](../../src/irsim/config/scene.py)

That follow-on has **no step id, no lane and no roadmap row**. `probe_warp_prim.py:6-20` records the
three concrete hazards it measured: quads need triangulating, points are in local space, and an
analytic gprim exposes no `points` at all.

### 2.2 What does exist, and is good

ADR 0047's mapping layer, shipped and wired into four render drivers:

- [`mapping.py`](../../src/irsim/materials/mapping.py) — engine-free resolver, precedence
  `thermal:material` override → semantic class → ordered case-insensitive glob → **loud miss**
  (`UNMAPPED_MATERIAL_ID = 0`, recorded, never a silent default).
- [`materials_usd.py`](../../src/irsim_isaac/pipeline/materials_usd.py) — the only place that opens a
  USD stage; produces engine-free `PrimRecord`s.
- [`audit_materials.py`](../../scripts/audit_materials.py) — coverage report, 95 % CI gate, runs
  engine-free in milliseconds from a JSON dump.
- `configs/materials/` — 21 materials, `mapping.yaml` with 17 semantic classes and 30 name globs.
- Magenta overlay for unmapped pixels, display-only.

This is a good layer. Sections 3 and 4 are about the one assumption in it.

---

## 3. The problem, stated precisely

The per-pixel chain today, verbatim from
[`material_ids.py:9-13`](../../src/irsim_isaac/pipeline/material_ids.py):

```
instance id  --idToLabels-->  prim path  --M7.17 resolver-->  material id
```

One prim → one material. A single-part building breaks **three** bindings:

| # | what breaks | why |
|---|---|---|
| 1 | **Material** | Door, window and wall collapse to one emissivity. ε spans 0.05 (bare metal) to 0.93 (paint) across that boundary. |
| 2 | **Temperature** | A thermal node binds to a prim, so the whole building becomes one temperature — precisely the per-object defect the project exists to fix. |
| 3 | **Thermal parameters** | Glass is ~4 mm of τ>0 dielectric; a rendered wall is ~300 mm of masonry with ~100× the areal heat capacity. `ρ·c_p·δ` is authored per *material*, so one prim cannot carry both. |

**[C]** In USD, a multi-material mesh is represented as `UsdGeomSubset` children with
`familyName="materialBind"`, `elementType=face`, `familyType=nonOverlapping`. This is what
`UsdShadeMaterialBindingAPI::CreateMaterialBindSubset()` produces, and it is what the FBX/OBJ
converters emit — one subset per source material.
([OpenUSD API](https://openusd.org/release/api/class_usd_shade_material_binding_a_p_i.html))

[`materials_usd.py:100`](../../src/irsim_isaac/pipeline/materials_usd.py) walks `UsdGeom.Gprim` and
calls `ComputeBoundMaterial()` on the **mesh**. It never descends into subsets.

---

## 4. The silent-failure hazard — read this before importing anything

**[C]** Blender's USD exporter documents:

> "When a mesh has multiple materials assigned, a geometry subset is created for each material.
> **The first material (if any) is always applied to the mesh itself as well** (regardless of the
> existence of geometry subsets), because the Hydra viewport does not support materials on subsets.
> See USD issue #542."
> — [Blender manual, Universal Scene Description](https://docs.blender.org/manual/en/latest/files/import_export/usd.html)

Confirmed in the exporter source, `USDGenericMeshWriter::assign_materials()` in
[`usd_writer_mesh.cc`](https://projects.blender.org/blender/blender/src/tag/v5.2.0/source/blender/io/usd/intern/usd_writer_mesh.cc):
the mesh prim gets `MaterialBindingAPI` applied and bound to the first non-null slot, *in addition to*
one `CreateMaterialBindSubset()` call per material.

**Consequence for this project.** A Blender-exported building does *not* arrive UNMAPPED and magenta.
`ComputeBoundMaterial()` returns the first material slot, the resolver maps it confidently, coverage
reads 100 %, and the entire building renders at one emissivity and one temperature. **That is the
exact failure mode ADR 0047 was written to prevent, arriving through a door ADR 0047 does not cover.**

Any subset-aware ingestion must therefore include a guard: *a mesh that has `materialBind` subsets
**and** a direct mesh-level binding must not resolve from the direct binding.*

Related exporter details, all read from the v5.2.0 source:

- **[C]** Subsets are written only when `export_materials` is true and there are ≥ 2 non-empty slots.
- **[C]** The subset prim name is the **USD material prim's name**, not the Blender slot index.
- **[C]** `doubleSided` is taken from the first non-empty slot only, for the whole mesh.
- **[C]** Subsets are **not time-sampled**, so they break if an animated mesh changes topology.

---

## 5. The mechanism that already exists

ADR 0110 solved point-wise temperature on curved geometry by rejecting the premise that a surface
parameterisation must be *transported* by the renderer. It is **derived** instead: the instance id
says which prim a pixel hit, the float32 world-position AOV says where, and a closest-point query
against that prim's own mesh returns **the triangle index and barycentric coordinates**.

That is [`closest_point_on_mesh`](../../src/irsim/thermal/mesh_field.py) — shipped, engine-free, and
already the oracle the Warp path is held to ([`mesh_bridge.py`](../../src/irsim_isaac/pipeline/mesh_bridge.py)).

A triangle index is exactly what a `GeomSubset` indexes. Therefore:

> **If the mesh carried in `irsim` also carries a `face_material_id` array, the same closest-point
> query that already produces per-pixel temperature produces per-pixel material — on the CPU, with
> zero new renderer capability.**

This reduces the multi-material single-mesh problem from "we need an AOV this build may not have" to
"we need one more integer array alongside `faces`". `TriangleMeshField` already derives `cell_face`,
`face_offsets`, `face_area_m2` and `face_normal`; a per-face material column is the same shape.

It also opens the texture route — see §6.5.

---

## 6. External findings

### 6.1 FBX → USD conversion is a CPU step

**[C]** `usd-convert-asset` is a standalone C++ library with Python bindings and a console script;
**"No GPU requirement for core conversion"**, and it does not need a running Kit application. Converts
FBX/OBJ/glTF → USD/USDA/USDC/USDZ. `--preview-surface` emits `UsdPreviewSurface` instead of MDL;
`--ignore-materials` strips materials. Windows 10 or Ubuntu 22.04, x86_64 or Linux aarch64; built
from source with Premake, then installed as a wheel.
([repo](https://github.com/NVIDIA-Omniverse/usd-convert-asset))

**[C]** The in-Kit `omni.kit.asset_converter` extension accepts **only .fbx, .obj, .gltf**
([Isaac Sim 6.0 formats](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/importer_exporter/formats.html)).

**This matters for this project specifically:** conversion does not touch the GPU, so it is compatible
with the standing constraint that Isaac Sim and CUDA workloads are not to be started on this
workstation.

### 6.2 Blender round-trips subsets in both directions

**[C]** Not just export. `USDMeshReader::assign_facesets_to_material_indices()` in
[`usd_reader_mesh.cc`](https://projects.blender.org/blender/blender/src/tag/v5.2.0/source/blender/io/usd/intern/usd_reader_mesh.cc)
calls `UsdGeomSubset::GetAllGeomSubsets()` — deliberately broader than `GetMaterialBindSubsets()`,
with a source comment noting that applications such as Houdini export subsets in other families — and
writes a real per-face `material_index` attribute on `AttrDomain::Face`. Non-`face` `elementType`
subsets are skipped with a warning.

**[C]** The historic crash on this path (Blender #125184, USD import crashing during material
assignment to geometry subsets) was fixed 2024-08-24.

**[C]** USD I/O is **not** marked experimental in Blender and never was; the only thing under the
manual's *Experimental* heading is the **Instancing** export option. Verified across the 2.93, 3.3,
3.6, 4.0, 4.2, 4.5, 5.0 and 5.2 manuals.

**[C]** Bundled OpenUSD versions: Blender 4.5 LTS → OpenUSD 25.02; Blender 5.2 LTS → OpenUSD 26.03.

### 6.3 Separate-by-material is scriptable headlessly

**[C]** `bpy.ops.mesh.separate(type='MATERIAL')`. Read from
[`editmesh_tools.cc`](https://projects.blender.org/blender/blender/src/tag/v5.2.0/source/blender/editors/mesh/editmesh_tools.cc):

- It **works in Object Mode** — `edbm_separate_exec` has an explicit object-mode branch iterating
  `selected_editable_bases`; only `type='SELECTED'` is rejected there. A headless script needs no
  Edit Mode toggle. It also runs multi-object.
- **Custom normals are preserved.** `mesh_separate_material()` brackets the operation with
  `BM_custom_loop_normals_to_vector_layer()` / `..._from_vector_layer()`. This contradicts the common
  claim that separate destroys custom normals — which matters here, because §8 lists normals as a
  silent-failure source.
- **Each result object ends with exactly one material slot** (`BKE_id_material_clear()` then
  `BKE_id_material_resize(..., 1, true)`), so the mesh-level USD binding and the single subset agree —
  the §4 hazard disappears.
- **Cost:** each separated object is a full object duplicate via `add_duplicate()`, carrying a copy of
  the entire modifier stack, constraints, parenting and custom properties. N materials → N objects
  each with a copy of the stack. Modifiers are *replicated*, not applied.
- If every face uses the same material, no new object is created.

### 6.4 Blender FBX import fidelity

**[C]** Blender 5.0 made the new **C++ FBX importer** (built on ufbx) the default:
`bpy.ops.wm.fbx_import`; the Python `bpy.ops.import_scene.fbx` is legacy.

- Material names are preserved directly — `BKE_material_add(bmain, fmat.name.data)` in
  `fbx_import_material.cc`. Standard Blender ID caveats apply (63-byte limit, `.001` suffix on
  collision).
- Per-face material assignment is preserved — `import_face_material_indices()` in
  `fbx_import_mesh.cc` writes the `material_index` face attribute from `fmesh->face_material`.
- **5.0 added a "Material Name Collision" option** with a *Reference Existing* mode, which reuses
  existing materials when names match. Relevant when importing many assets that share material names
  and `.001` suffixes would otherwise defeat the glob table.
- FBX **export** is still the Python add-on.

**[U]** How the C++ importer handles very old FBX 6.x ASCII files, and whether per-face assignment
survives files using "AllSame" mapping mode with node-level overrides. No tracker entry found either
way.

### 6.5 Prior art: how other IR simulators do this

- **DIRSIG** — geometry authored as OBJ, then material properties assigned **per facet** using the
  *Bulldozer* tool; emissivity supplied via `EMISSIVITY_PROP` in `SURFACE_PROPERTIES` of the material
  file. ([RIT DIRSIG](https://dirsig.cis.rit.edu/docs/lach.pdf))
- **OKTAL-SE SE-Workbench** — assigns physical materials to geometry **via textures**, mapping
  physical attributes (EO *and* RF) onto the 3D mock-up from a library of physical data.
  ([OKTAL-SE Materials](https://www.oktal-se.fr/materials/))
- **ThermoAnalytics MuSES** — states its model requirement as *a 3D surface mesh, thermal material
  properties, wavelength-dependent optical surface properties, and active heat sources*.
  ([MuSES](https://www.thermoanalytics.com/product/muses/))

Two things follow. First, MuSES's list is the honest scope statement: **three of those four are not
in the asset file**, which is §8's subject. Second, OKTAL-SE's texture route is directly reachable
here — with a face index and barycentrics from §5 you can interpolate the mesh's own UVs and sample a
material-ID texture, without any UV AOV. That matters because the painted-vs-bare-metal distinction
that dominates LWIR appearance often lives inside one albedo texture rather than in separate material
slots.

### 6.6 NVIDIA's own Blender path, and its availability

**[C]** The Omniverse Blender Connector is **discontinued**: NVIDIA stopped producing custom Blender
alpha builds, and from Blender 4.5 the USD work is integrated into official Blender LTS releases. The
old `NVIDIA-Omniverse/blender_omniverse_addons` repo returns HTTP 404. There is **no MDL exporter for
Blender any more** — `UsdPreviewSurface` is the path, and Isaac Sim 6.0 supports it alongside MDL.
([Omniverse Connect — Blender](https://docs-prod.omniverse.nvidia.com/connect/latest/blender.html))

**[C]** Its replacement is the **SimReady Blender Add-on**, release 2026.04.0 (published 2026-05-07),
Apache-2.0, targeting **Blender 5.1**, prerequisites **Windows 10/11 only — "Linux is coming soon"**.
([docs](https://nvidia.github.io/simready-blender-add-on/) ·
[repo](https://github.com/NVIDIA/simready-blender-add-on))

Directly relevant: its `MATERIAL_PT_simready_nonvisual` panel is described as *"non-visual sensor
attributes used by the NVIDIA RTX renderer to model how a material reads to non-visual sensors such as
radar, lidar, or **infrared (IR)**"*, managing Base, Coating and Attributes — e.g. base = aluminum,
coating = painted, attributes = emissive, visually-transparent, single-sided.

That is conceptually the same job as this project's `thermal:material` attribute, from the vendor, and
it is **not runnable on this workstation's platform today**. Worth tracking; not worth waiting for.
Note also that its taxonomy (base + coating) matches the `thermal-materials` skill's Rule 2 — the
metal question is paint, not alloy.

**[C]** The Omniverse Launcher was deprecated 2025-10-01, with connectors moved to the NGC Catalog.

**[C]** There is **no official Blender→Isaac asset-prep guide**. The Isaac Sim 6.0 documentation
search index — 3,658 indexed sections — contains **zero** occurrences of "blender" (against 4,933 for
"usd", 119 "mdl", 8 "fbx", 20 "onshape").

**[C]** NVIDIA staff guidance (forum, 2026-02-23) is that Blender armatures export as USD skeletal
animation, which is not PhysX articulation, and recommends Blender for geometry only with articulation
authored in URDF. For static and visual assets — this project's case — that restriction does not bite.
([thread](https://forums.developer.nvidia.com/t/blender-to-isaac-sim/361514))

---

## 7. Unconfirmed — do not build on these

| claim | status |
|---|---|
| A `perSubsetSegmentation` true/false mode on RTX semantic segmentation | **[U] Could not confirm.** Reported by one research pass, but a direct fetch of the cited Isaac Sim 6.0 RTX Sensor Annotators page found no occurrence of the term, and a targeted search returned no authoritative hit. |
| Camera annotators assign distinct instance ids to `GeomSubset` prims | **[U] Open.** This is roadmap `IG.4`. Replicator's `instance_id_segmentation` is documented to go "down to the leaf prim"; whether a `GeomSubset` child counts as one is untested on this build. |
| RTX `objId` / `StableIdMap` gives per-subset identity to the camera path | **[C] but not applicable.** The docs do say `StableIdMap` "registers … per-`GeomSubset` entries when an instance has more than one subset", with the submesh index in the high 32 bits of a 128-bit object id — but this is the **RTX lidar/radar** path, gated on `--/rtx-transient/stableIds/enabled=true`, and is stated not to be available for standard camera annotators. |

What *is* confirmed about subsets in the Isaac toolchain: Replicator's API exposes helpers for
fetching `UsdGeomSubset` prims, and the Scene Optimizer's mesh merge has a `considerMaterials` flag
that "preserves per-material GeomSubset partitioning on the merged mesh". So subsets are understood
by the toolchain; per-pixel camera identity for them is still the open question.

**Settling `IG.4` requires the engine and must not be run opportunistically** — it needs an explicit
request from the owner, per the standing constraint on starting Isaac Sim or GPU work.

---

## 8. What an asset does not contain

The material question is the visible half. These are the ones that produce plausible images and wrong
physics.

| gap | consequence |
|---|---|
| **Thickness** | An FBX is a zero-thickness shell. Areal heat capacity is `ρ·c_p·δ`. δ is in no asset file; today it comes from the *material*, which is why a window and a wall cannot share a prim (§3). |
| **Interior vs exterior** | If interior geometry survives import, the solver puts sunlight and sky on the inside of walls. |
| **Normals and winding** | §6.1 balances a facet. A flipped normal faces the ground instead of the sky; the sky view factor is wrong, silently, and the image still looks fine. |
| **Scale and units** | The solver works in metres. Blender's USD export exposes `convert_scene_units` / `meters_per_unit`. |
| **Heat sources** | An imported vehicle has no engine. The Phantom scene needed hand-authored motor and battery throttle schedules. |
| **Occluders** | `phantom3_outbound_pointwise.yaml` hand-lists 7 occluder rectangles duplicating prim faces by hand. Nothing derives these from geometry. |
| **Instancing** | `UsdGeomPointInstancer` and scenegraph instancing break `instance id → prim path`. Asset-store buildings use them for windows constantly. |
| **Glass** | τ>0 needs the second ray (ADR 0046). A window in an FBX is normally one zero-thickness quad — no pane thickness, no cavity, no interior. |
| **Material name quality** | Asset-store materials arrive as `Material.001`, `lambert1`, `Default_OBJ`. The 30 globs in `mapping.yaml` will match none of them. |
| **Semantic labels** | The semantic route reads Isaac labels; SimReady's taxonomy keys to WikiData Q-codes. Neither is present in a raw FBX. |
| **Provenance and licence** | ADR 0041 governs data provenance. Imported assets need the same treatment. |

**Cost is not the obstacle.** `GT.7` cites Fraunhofer running 1,313,410 triangles with a 10-layer
stack through five day–night cycles in **252 s** on one i7-8700 in MATLAB. A building is affordable.
The multiplier to watch is `TriangleMeshField`'s per-face cell subdivision, which is ours to choose.

---

## 9. Candidate pipeline

Sequenced, not scheduled. Nothing here is a committed step.

| # | step | engine needed? |
|---|---|---|
| 1 | **Convert.** `usd-convert-asset`, `--preview-surface`. FBX → USD. | No — CPU only |
| 2 | **Audit first.** Extend the prim dump to walk `UsdGeomSubset` children, one record per subset; run `audit_materials.py`. Add the §4 guard. Run this on a real asset *before* designing around it — the answer tells you how bad the naming is. | No |
| 3 | **Blender: repair and author.** Split by material *or* keep subsets; rename materials to hit the globs; write `thermal:material` as a custom property; fix scale, normals, interior geometry; decimate. | No |
| 4 | **Mapping to face granularity.** `PrimRecord` gains a subset path and face-index range; `Resolution` becomes per-subset. Precedence unchanged. | No |
| 5 | **Read real triangles.** The unowned follow-on at `scene.py:413`. Triangulate quads, apply local-to-world, handle analytic gprims. | Reading USD needs Kit; no render |
| 6 | **Per-pixel material.** Route A: `face_material_id` + closest-point query (§5) — works today. Route B: per-subset instance ids — needs `IG.4` settled. | A: no. B: yes |
| 7 | **Thermal authoring.** Thickness, interior/exterior, sky view factor, heat sources (§8). The real cost. | No |
| 8 | **Validate.** Coverage gate, plus a phenomenology check that the door, the windows and the wall actually differ. | Render |

`thermal:material` is deliberately a plain USD attribute rather than an applied schema *"so that any
DCC can write it without our schema installed"*
([`materials_usd.py:16-19`](../../src/irsim_isaac/pipeline/materials_usd.py)). That sentence was
written for exactly this use case, and step 3 is where it pays off.

---

## 10. The open forks

1. **Split-by-material, or GeomSubsets end to end?** Splitting ships an imported building with zero
   new code and removes the §4 hazard outright; its cost is object-count explosion and duplicated
   modifier stacks. Subsets are the cleaner long-term answer and cost steps 4–6.
2. **Route A or Route B for per-pixel material?** Route A needs no renderer capability and reuses
   machinery already held to an oracle. Route B is cheaper *if* `IG.4` comes back positive. Given how
   many of ADR 0014's measured negatives this build has produced, Route A is the safer default
   regardless of how `IG.4` resolves.
3. **Does asset ingestion become a lane?** It currently spans `PT`, `WM` and `IG` without belonging to
   any of them, and the one written acknowledgement of it names no owner.

---

## Sources

**Repository:** `src/irsim/materials/mapping.py` · `src/irsim/config/scene.py` ·
`src/irsim/thermal/mesh_field.py` · `src/irsim_isaac/pipeline/materials_usd.py` ·
`src/irsim_isaac/pipeline/material_ids.py` · `src/irsim_isaac/pipeline/mesh_bridge.py` ·
`src/irsim_isaac/phantom3.py` · `src/irsim_isaac/airframe.py` · `scripts/audit_materials.py` ·
`scripts/probe_warp_prim.py` · `configs/materials/mapping.yaml` ·
`configs/scenes/phantom3_outbound_pointwise.yaml` · ADR 0014, 0040–0047, 0087, 0110, 0112 ·
`docs/roadmap.md` (`IG.4`, `GT.7`) · the `thermal-materials` skill

**External:**
[Blender USD manual](https://docs.blender.org/manual/en/latest/files/import_export/usd.html) ·
[usd_writer_mesh.cc](https://projects.blender.org/blender/blender/src/tag/v5.2.0/source/blender/io/usd/intern/usd_writer_mesh.cc) ·
[usd_reader_mesh.cc](https://projects.blender.org/blender/blender/src/tag/v5.2.0/source/blender/io/usd/intern/usd_reader_mesh.cc) ·
[editmesh_tools.cc](https://projects.blender.org/blender/blender/src/tag/v5.2.0/source/blender/editors/mesh/editmesh_tools.cc) ·
[bpy.ops.wm](https://docs.blender.org/api/current/bpy.ops.wm.html) ·
[bpy.ops.mesh](https://docs.blender.org/api/current/bpy.ops.mesh.html) ·
[Separate manual](https://docs.blender.org/manual/en/latest/modeling/meshes/editing/mesh/separate.html) ·
[Blender release notes](https://developer.blender.org/docs/release_notes/) ·
[UsdShadeMaterialBindingAPI](https://openusd.org/release/api/class_usd_shade_material_binding_a_p_i.html) ·
[usd-convert-asset](https://github.com/NVIDIA-Omniverse/usd-convert-asset) ·
[Omniverse Asset Converter](https://docs.omniverse.nvidia.com/extensions/latest/ext_asset-converter.html) ·
[Isaac Sim 6.0 formats](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/importer_exporter/formats.html) ·
[Isaac Sim 6.0 reference architecture](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/introduction/reference_architecture.html) ·
[RTX Sensor Annotators](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/sensors/isaacsim_sensors_rtx_annotators.html) ·
[Replicator Standard Annotators](https://docs.omniverse.nvidia.com/py/replicator/1.11.16/source/extensions/omni.replicator.core/docs/annotators_details.html) ·
[Omniverse Connect — Blender](https://docs-prod.omniverse.nvidia.com/connect/latest/blender.html) ·
[SimReady Blender Add-on](https://nvidia.github.io/simready-blender-add-on/) ·
[SimReady Blender Add-on repo](https://github.com/NVIDIA/simready-blender-add-on) ·
[SimReady specification](https://docs.omniverse.nvidia.com/simready/latest/overview/simready-spec.html) ·
[Blender to Isaac Sim forum thread](https://forums.developer.nvidia.com/t/blender-to-isaac-sim/361514) ·
[Omniverse Launcher deprecation](https://forums.developer.nvidia.com/t/nvidia-omniverse-launcher-deprecation-on-oct-1st-what-you-need-to-know/321325) ·
[DIRSIG scene modeling](https://dirsig.cis.rit.edu/docs/lach.pdf) ·
[OKTAL-SE Materials](https://www.oktal-se.fr/materials/) ·
[ThermoAnalytics MuSES](https://www.thermoanalytics.com/product/muses/)
