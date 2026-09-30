---
name: ingest-asset
description: The full pipeline from a shared model link (Sketchfab) or a 3D file handed over directly, optionally with the object's real-life name ("DJI Mavic 3"), to a library entry — arrival through the licence gate, identify the real object and search its published specs, fix the scale, decompose into functional parts, research and map every part's real material, export .blend/.usdc/.fbx, audit, ship. Use this skill whenever the user shares a 3D-model link or a model file (.fbx/.glb/.gltf/.obj/.usd/.zip), names what a model really is, or asks to add, import, ingest or download a model/asset/drone/aircraft, whenever a new entry is wanted in the asset library, and whenever a downloaded mesh needs materials or parts assigned. The judgement steps (part naming, material research) are LLM work by design; do them with this checklist, not from memory.
---

# Ingesting a third-party asset: a link or a file becomes a library entry

The automation of what was done by hand for the Phantom 4. One asset per run. The pipeline itself
is CPU work — Blender is the toolchain (ADR 0128) and no step here needs Isaac Sim or the GPU.

## 0. What the owner hands you

Any of these, in any combination:

| input | example | goes to |
|---|---|---|
| a Sketchfab link | `https://sketchfab.com/3d-models/…-<uid>` | step 1a |
| a 3D file or a .zip | `~/Downloads/mavic.fbx`, `drone.glb`, `plane.zip` | step 1b |
| the object's **real-life name** | "DJI Mavic 3", "Cessna 172 Skyhawk", "white stork" | step 2 |
| a licence / author for a file | "it's CC-BY by some_artist", "I made it" | step 1b |

**The real name is the most valuable thing the owner can say.** It is what you search the web for
in step 2 — published dimensions, the real part list, what each part is made of — and it names the
asset (`dji_mavic_3`), outranking a page title like "free low poly drone". Pass it through with
`--real-name` so it is recorded in the provenance and the next reader of the material map knows
which product its `ESTIMATED` lines were researched against.

When the owner gives **no** real name, infer it (page title, tags, description, what the model
looks like), say which name you inferred and why in one line, and carry on — do not stop to ask.
When the inference is genuinely ambiguous (a generic "quadcopter", a bird of unknown species),
pick the most representative real product/species, flag every number that depends on it
`ESTIMATED`, and say so in the final report.

**What a finished entry is** (ADR 0150; the owner's 2026-09-26 format decision):

| where | what | in git? |
|---|---|---|
| `3d_models/<name>/` | downloaded glTF (`gltf/`) or handed-over file (`source/`), `ATTRIBUTION.md` | no (cloud drive) |
| `configs/assets/<name>.provenance.yaml` | real name, author, licence, gate verdict, sha256 | **yes** |
| `configs/assets/<name>.yaml` | `scale_to_metres`, material map, `parts:` block | **yes** |
| `data/assets/<name>*/` | generated `.usdc`, `.meshes.npz`, prim dump, textures | no (regenerable) |
| `data/assets/<name>/<name>.blend` + `.fbx` | master scene + interchange copy | no (cloud drive) |

Binaries travel by the owner's cloud-drive link, never by git. The two YAMLs are what git carries;
with them plus the binaries, every generated artefact reproduces from `prep_asset.py`.

## 1a. A link: fetch through the licence gate (AI.8)

```bash
python scripts/fetch_sketchfab.py <link> --dry-run   # metadata + gate verdict, no token needed
SKETCHFAB_API_TOKEN=…  python scripts/fetch_sketchfab.py <link> --real-name "DJI Mavic 3"
```

The gate runs **before** the download and fails towards quarantine: CC0 / CC-BY / CC-BY-SA land in
`3d_models/<name>/` with a committed provenance YAML; ND, NC, Standard, Editorial and anything
unrecognised land in `3d_models/quarantine/<name>/` — usable locally, **never** committed,
uploaded, bundled or published, and `QUARANTINED.md` in the directory says why. Do not argue with
the gate in code; widening the shareable set is an ADR-level decision (test-pinned, ADR 0150).

Known failure modes, all printed with what to do: view-only model (pick another — never scrape),
401 (token from sketchfab.com Settings → Password & API), 429 (automatic backoff), no glTF key
(rare; inspect by hand). The API **never serves the source FBX** — glTF in, FBX out is the design.

## 1b. A file: register it through the same gate

```bash
python scripts/register_local_asset.py <file> --real-name "DJI Mavic 3" \
    [--license by --author some_artist --origin-url <where it came from>]
```

Copies the file (or extracts the .zip) into `3d_models/<name>/source/`, pins its sha256, and
writes `ATTRIBUTION.md` + provenance exactly as a fetch does. A file carries no licence metadata,
so `--license` is the licence **the owner stated** (`cc0`, `by`, `by-sa`, …; their own work is
`cc0` or `by`). **No `--license` means quarantine** — the unknown-licence branch of the same gate,
with no special case for files. If the owner mentioned a licence in passing, pass it; if they did
not, register quarantined, keep going (local use is fine), and tell them in the final report that
one flag re-registers it as shareable. A `.blend` is refused: export it to `.glb` in Blender first,
so every asset enters through an importer that preserves material names. A **STEP/IGES** CAD file
is tessellated first (`scripts/tessellate_step.py`, the `low2high` skill §2) and the resulting
`.glb` registered.

## 2. Identify the real object and search its specs, before opening Blender

Start from the real name (step 0). Without one, the page's title/tags identify a real product more
reliably than anything inside the file. Web-search it now — manufacturer spec sheet first, then
teardowns, reviews and reference sites — and write down, with the source URL, in the config's
comments as you go:

* **one published dimension** (wingspan, rotor diameter, height) — this fixes the scale;
* **the real part list and what each part is made of** (a named quadcopter's shell is moulded
  ABS/polycarbonate, props nylon or ABS, arms sometimes CFRP) — this outranks every appearance cue;
* anything thermal (battery position, motor placement) for the `parts:` block later.

## 3. Author the config skeleton and fix the scale

Copy the shape of `configs/assets/phantom4.yaml` (schema: `irsim.materials.mapping.AssetConfig`).
First run only needs `asset.name`, `source_file` (`gltf/scene.gltf` for a link, `source/<file>`
for a file) and `scale_to_metres`. glTF is metres by spec, so expect **1.0**; an FBX declares no
unit and is often centimetres (the Phantom 4 was: 0.01). Verify either way: measure the
axis-aligned extent **from the vertices, never `bound_box`** (the phantom4.yaml header
records how the local-box shortcut inflated 41 cm to 60), compare against the published dimension,
and write the reasoning as a comment where the number lives. A wrong scale renders a perfectly
plausible image; that is the hazard.

## 4. First prep pass, and read the misses

```bash
python scripts/prep_asset.py --asset <name>
```

Import → rescale → USD → prim dump → audit. The audit prints every source material the global
rules miss or mis-map. glTF arrives triangulated with vertices split at every normal/UV seam, so
expect more shells than the artist made; the connected-component pass (step 6) merges by
proximity, but check its component count for sanity against the FBX-era numbers before trusting
selectors.

## 5. Map the materials — evidence, never names

Order of evidence, strongest first (the repo's own history: "Glass_Black" is opaque,
"metal_radial" is not metallic):

1. **step 2's product research** — what the real part is made of;
2. **the shader, dumped**: per material read alpha / metallic / base colour off the Principled
   BSDF (phantom4.yaml's header shows the table to write). In glTF's `metallicRoughness` texture
   the **B channel is metallic, G is roughness** — sampling the wrong channel inverts the reading;
3. **material names and texture filenames** as hints only; artist `metallic=1` is a style choice.

Then, per source material, one line in `materials:` mapping to a library entry:

* every guess is flagged `# ESTIMATED` with the reasoning, like phantom4.yaml does;
* ε < 0.2 (polished metal) may only come from an asset map or prim override, never a glob (AT.18);
* if the library has no entry for the real substance, **draft one** — `configs/materials/`,
  schema v2, `surface_treatment` required (ADR 0142), author exactly one of ε/ρ/τ (closure derives
  the rest; CLAUDE.md #4), cite a source, use the `thermal-materials` skill. The library walk test
  and the commit diff are the review gate (owner's decision 2026-09-26: drafting allowed, gated).

Re-run prep until the audit passes its ≥ 0.95 coverage gate with **zero wrong confident hits**.

Two name traps, both met on the DJI drones (AI.15):

* **USD sanitises names**: `black-paint` reaches the audit as `black_paint`, while the thermal archive
  and the component pass keep Blender's `black-paint`. Map **both spellings** of any name with a
  character outside `[A-Za-z0-9_]`, or one of the two paths misses.
* **One texture atlas for everything** (the Inspire 3's `default` covers 324 of 345 objects): map
  the atlas to what most of its area is, then let each part that is something else say so with
  `material:` (step 6). Evidence per part is the product research, not the atlas name.

## 6. Decompose into functional parts (ADR 0138)

A part is hardware a temperature can name — propeller, motor, battery, fuselage — never "the white
plastic". Get the evidence in this order:

1. `python scripts/prep_asset.py --asset <name> --emit-mesh` then `--emit-components` output:
   component sizes, centroids, materials;
2. **repetition is structure**: N identical components arranged symmetrically are the motors/props;
   verify candidate selectors by symmetry (four propellers within a few % area of each other —
   phantom4.yaml shows the check and why per-station heights were needed);
3. **look at it**: render turntable views and per-candidate highlight renders in Blender —
   CPU only (Cycles with `device='CPU'`, low samples, or the workbench engine; the Blender MCP's
   thumbnail tools also work) — and judge which functional part each candidate is;
4. object names last, and only as corroboration.

Author the `parts:` block (schema `irsim.io.asset_parts.PartsConfig`): geometric selectors,
first-match-wins, specific parts before the shells that enclose them, `coverage_threshold: 0.95`.
Part names obey `^[A-Za-z_][A-Za-z0-9_]*$`, ASCII, unique — they become USD prim names and the
keys every scene config binds by.

## 6b. Make the geometry fit for the camera — the `low2high` skill

Run `python scripts/mesh_facets.py <model> --range-m <closest> --ifov-mrad <pixel>` on the
render geometry. If any part is faceted (typical of game and phone-game models) or the model is
far over the render budget (typical of CAD), follow the `low2high` skill on the part-split scene
before exporting. A model that passes needs nothing.

Two optional fields (AI.14, ADR 0173):

* **`material:` on a part** asserts a library material for **every** face of that part, which is
  written as `thermal:material` on its prims in the part-split USD and outranks the source map.
  Use it only where the source material cannot tell substances apart. Split out anything the part
  must not paint first, e.g. a lens as its own part before the gimbal that holds it.
* **`exclude:`** takes selectors for geometry that is not the asset: a transport case, a display
  stand, a ground plane. It is tried before every part and leaves the coverage totals. Prefer a
  material selector when the foreign object has its own materials (the Matrice 300's case).

Measure selectors, don't eyeball them. Score a candidate against objects you identified in a
highlight render (capture %, plus what else it takes), then render every face coloured by its
assigned part before exporting. Symmetric stations must come out the same size: that is the
cheapest check there is. A tilted rotor station (the Inspire 3's arms are pitched 20°) makes
height cuts per-station. Check the **thermal archive's** part report as well as the render
split's. The archive is planar-dissolved, so a shell's **face count** and its vertex-mean
**centroid** differ between the two passes. Never select on `faces_*`, and don't put a radius
cut within a few millimetres of a shell you need; separate by area instead. On the Inspire 3, a
`faces_min` sent half of every blade into the motor in the archive only.

If the mesh emit refuses ("changed area by more than 2.0 %"), pass `--dissolve-deg 3` and record
it in the config's header. The number is part of the reproduction.

## 7. Export the library artefacts

```bash
python scripts/prep_asset.py --asset <name> --emit-mesh --emit-parts --save-blend --emit-fbx
```

This writes the part-split sibling (`<name>_parts`, one prim per part), the thermal archive, the
master `.blend` and the interchange `.fbx` beside the `.usdc`. The FBX is a courtesy for non-USD
consumers — scenes reference the USD, and nothing may depend on the FBX's axis/unit choices.
Respect the geometry budget's verdict (AI.3): over-budget means decimate or descope, not
`--allow-over-budget` by reflex.

## 8. Audit, then ship

* `make check` — the materials-library walk, the layering test and the asset tests are the gate;
* `python scripts/audit_materials.py` on the prim dump if anything changed since the last prep run;
* a contact-sheet render is welcome (CPU is enough for this); binding the asset into a scene
  and rendering it in Isaac Sim (`configs/scenes/phantom4_perpart.yaml` is the template) is the
  *next* step's work, not this skill's;
* finish with the `ship-step` skill: technical-report status, a README "What's new" line for the new asset, CHANGELOG, roadmap row, one commit. Remind the
  owner which directories to upload to the cloud drive (`3d_models/<name>/`,
  `data/assets/<name>*/`) — and that quarantined assets are excluded from that too.

## When to stop and ask

* the licence gate quarantines a model the user seems to want shared — say so, do not override
  (for a file, the fix is re-registering with the licence the owner states, not editing the gate);
* the real name the owner gave contradicts the model (a "Mavic 3" with fixed wings) — report it
  rather than researching the wrong product;
* no published dimension can be found and the glTF's own scale looks wrong — a guessed scale
  poisons everything downstream;
* the model is one merged blob with a texture atlas (no components to select) — per-face material
  maps are future work; say so rather than inventing parts;
* a part's material would need ε < 0.2 on a guess — AT.18 exists because that mistake rendered
  motors as reflected sky.
