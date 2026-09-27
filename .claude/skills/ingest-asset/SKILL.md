---
name: ingest-asset
description: The full pipeline from a shared model link (Sketchfab) to a library entry — download through the licence gate, identify the real object, fix the scale, decompose into functional parts, research and map every part's real material, export .blend/.usdc/.fbx, audit, ship. Use this skill whenever the user shares a 3D-model link or asks to add, import, ingest or download a model/asset/drone/aircraft, whenever a new entry is wanted in the asset library, and whenever a downloaded mesh needs materials or parts assigned. The judgement steps (part naming, material research) are LLM work by design; do them with this checklist, not from memory.
---

# Ingesting a third-party asset: a link becomes a library entry

The automation of what was done by hand for the Phantom 4. One asset per run. Everything here is
CPU-only — Blender is the toolchain (ADR 0128), no step boots Isaac Sim or touches the GPU.

**What a finished entry is** (ADR 0150; the owner's 2026-09-26 format decision):

| where | what | in git? |
|---|---|---|
| `3d_models/<name>/` | source glTF archive + zip, `ATTRIBUTION.md` | no (cloud drive) |
| `configs/assets/<name>.provenance.yaml` | author, licence, gate verdict, sha256 | **yes** |
| `configs/assets/<name>.yaml` | `scale_to_metres`, material map, `parts:` block | **yes** |
| `data/assets/<name>*/` | generated `.usdc`, `.meshes.npz`, prim dump, textures | no (regenerable) |
| `data/assets/<name>/<name>.blend` + `.fbx` | master scene + interchange copy | no (cloud drive) |

Binaries travel by the owner's cloud-drive link, never by git. The two YAMLs are what git carries;
with them plus the binaries, every generated artefact reproduces from `prep_asset.py`.

## 1. Fetch, through the licence gate (AI.8)

```bash
python scripts/fetch_sketchfab.py <link> --dry-run   # metadata + gate verdict, no token needed
SKETCHFAB_API_TOKEN=…  python scripts/fetch_sketchfab.py <link>
```

The gate runs **before** the download and fails towards quarantine: CC0 / CC-BY / CC-BY-SA land in
`3d_models/<name>/` with a committed provenance YAML; ND, NC, Standard, Editorial and anything
unrecognised land in `3d_models/quarantine/<name>/` — usable locally, **never** committed,
uploaded, bundled or published, and `QUARANTINED.md` in the directory says why. Do not argue with
the gate in code; widening the shareable set is an ADR-level decision (test-pinned, ADR 0150).

Known failure modes, all printed with what to do: view-only model (pick another — never scrape),
401 (token from sketchfab.com Settings → Password & API), 429 (automatic backoff), no glTF key
(rare; inspect by hand). The API **never serves the source FBX** — glTF in, FBX out is the design.

## 2. Identify the real object before opening Blender

The model page's title/tags identify a real product more reliably than anything inside the file.
Web-search it now and write down, in the config's comments as you go:

* **one published dimension** (wingspan, rotor diameter, height) — this fixes the scale;
* **the real part list and what each part is made of** (a named quadcopter's shell is moulded
  ABS/polycarbonate, props nylon or ABS, arms sometimes CFRP) — this outranks every appearance cue;
* anything thermal (battery position, motor placement) for the `parts:` block later.

## 3. Author the config skeleton and fix the scale

Copy the shape of `configs/assets/phantom4.yaml` (schema: `irsim.materials.mapping.AssetConfig`).
First run only needs `asset.name`, `source_file` (the extracted `gltf/scene.gltf`), and
`scale_to_metres`. glTF is metres by spec — so the expected factor is **1.0** — but verify anyway:
measure the axis-aligned extent **from the vertices, never `bound_box`** (the phantom4.yaml header
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
* a contact-sheet render is welcome but **only on the CPU**; never launch Isaac Sim for this —
  binding the asset into a scene (`configs/scenes/phantom4_perpart.yaml` is the template) is the
  *next* step's work, not this skill's;
* finish with the `ship-step` skill: README status, CHANGELOG, roadmap row, one commit. Remind the
  owner which directories to upload to the cloud drive (`3d_models/<name>/`,
  `data/assets/<name>*/`) — and that quarantined assets are excluded from that too.

## When to stop and ask

* the licence gate quarantines a model the user seems to want shared — say so, do not override;
* no published dimension can be found and the glTF's own scale looks wrong — a guessed scale
  poisons everything downstream;
* the model is one merged blob with a texture atlas (no components to select) — per-face material
  maps are future work; say so rather than inventing parts;
* a part's material would need ε < 0.2 on a guess — AT.18 exists because that mistake rendered
  motors as reflected sky.
