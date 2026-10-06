---
name: publish-to-zoo
description: Publish an asset-library model to the Thermal Model Zoo website (https://reza-shahriari.github.io/ThermalModelZoo/) — bundle it with scripts/zoo_bundle.py (renders, part table, 8-14 µm emissivities, packed files), then upload and list it with the zoo repo's tools/publish.py. Use whenever the user wants to publish, upload, share or add a model to the zoo / model zoo / the website / "the site with the models", wants a zoo entry updated or removed, asks where model files should be hosted (GitHub Releases vs Hugging Face), or asks to add a zoo category. Not for this repo's own project site (that is present-on-the-site).
---

# Publish a library model to the Thermal Model Zoo

The zoo is a **separate repository** — https://github.com/reza-shahriari/ThermalModelZoo — a static
GitHub Pages catalogue of part-split, material-mapped models. Nothing of it lives in this repo; it
is not `make site`. Two tools, one per side:

| Side | Tool | Does |
|---|---|---|
| irsim (here) | `scripts/zoo_bundle.py <asset>` | library asset → **bundle** folder |
| zoo repo | `tools/publish.py <bundle>` | bundle → web images, uploaded files, catalogue entry |

A bundle is `entry.json` + `images/` + `files/` + `ATTRIBUTION.md` (format in the zoo README,
"Add a model"). Anyone can hand-make one; `zoo_bundle.py` makes it from what `ingest-asset` produced.

## Preconditions — check, don't assume

1. The asset went through **ingest-asset** to the end: `configs/assets/<asset>.provenance.yaml`
   exists, and `data/assets/<asset>_parts/<asset>_parts.blend` exists (the part-split export is
   what the zoo publishes). No provenance → not publishable (phantom4 has none). No `_parts`
   .blend → finish ingestion first (liberty_ship, as of 2026-10-04).
2. Licence slug is `cc0`, `by` or `by-sa` and `shareable: true`. `zoo_bundle.py` refuses anything
   else — never work around it; NC/ND/editorial models stay out of the zoo.
3. Every source material maps to a library material (the script stops naming the one that does not).

## Steps

```bash
PY=~/IsaacSim/_build/linux-x86_64/release/python.sh       # has irsim, Pillow, yaml
# 1. bundle. --category is a zoo category id. --isaac renders the beauty + thermal pair in Isaac
#    Sim (GPU, ~4 min); --nose-deg is the nose bearing from the asset config (+x = 0; -y = -90).
IRSIM_GPU=0 IRSIM_ISAAC_PYTHON=$PY $PY scripts/zoo_bundle.py dji_mini_3_pro --category drones \
    --subcategory Multirotor --isaac --nose-deg -129 [--featured] [--summary "..."] \
    [--turn-deg 135 --elevation-deg 12 --range-m R] --out <scratch>/bundles
# 2. LOOK at <bundle>/images/*.png and entry.json before publishing (framing, part rows, values).
# 3. publish from a clone of the zoo repo
git clone https://github.com/reza-shahriari/ThermalModelZoo.git && cd ThermalModelZoo
$PY tools/publish.py <scratch>/bundles/dji-mini-3-pro --dry-run   # first, no upload
python3 -m http.server 8765    # check model.html?id=<id> (headless chrome --screenshot works)
$PY tools/publish.py <scratch>/bundles/dji-mini-3-pro             # uploads, writes the entry
git add data/models.json models/<id> && git commit && git push    # Pages deploys in ~30 s
```

Re-running either step replaces the bundle / the entry and the release files of the same name.
`tools/publish.py --remove <id>` takes an entry out (it leaves the release in place).
**Hand-edited entry fields survive only if you re-apply them.** A rebuilt bundle regenerates
`entry.json` from the configs; anything added by hand on the live entry (a part note, a longer
`changes` line — the DJI FPV has both) must be copied into the new `entry.json` before
`publish.py`, which replaces the entry whole. Diff the live entry against the new one first.

Uploading and pushing is publishing to the public: do it when the user asked for the model to go
up, not as a side effect of testing — `--dry-run` is the test.

## What the bundle contains, and its honest limits

- **Part rows**: one per functional part from the `_parts` .blend; mirror copies collapse
  (`motor_front_left` … → "Motor (x4)") with their material areas summed. Material text lists the
  library ids by area share; **emissivity is area-weighted over 8-14 µm** (top-hat; the registry's
  `lwir` is 7.5-13.5 µm, the zoo states 8-14). Source = the weakest of its materials' `source`
  (`literature` shows as "reference").
- **Images**: with `--isaac` (always, for the zoo) `beauty` + `thermal` are one camera's matched
  pair from Isaac Sim — the aircraft at cruise against a clear midday sky
  (`configs/scenes/zoo_hero.yaml`), path-traced RGB companion + white-hot LWIR, 1280×1024
  (`configs/sensors/example_lwir_1280.yaml`), via `render_phantom4.py --asset <asset>_parts`.
  Temperatures are **per part node** (motor / battery / airframe), not a per-point solve; say so,
  don't oversell. Frame 0's colour buffer is empty on this build, so it renders 3 frames and keeps
  the last. Raw renders land in `outputs/zoo_hero/<asset>/` (+ `.log`). Plus `wireframe` (one hue
  per kind of part) and `emissivity` (flat grey per surface, scale stretched to the model's range)
  from Blender. Without `--isaac` the beauty is a studio Eevee render and there is no thermal image
  unless you pass `--thermal`. Never fake one. Look at every pair before publishing: a dark
  aircraft seen from below in high sun reads dark in the RGB (the Avata 2), which is physical.
- **New pictures for a listed model**: `tools/publish.py <bundle> --images-only` replaces its
  images and thumbnail and keeps the live entry (part notes, `changes`, download links) and the
  release files as they are — the safe way to refresh images, since others hand-edit entries.
- **Files**: `<id>.blend` (textures packed), `<id>.usdc.zip`, `<id>.fbx.zip` (+ textures).

## Hosting: GitHub Releases vs Hugging Face

Default **GitHub Releases** (`gh` is logged in on this machine): one release `model-<id>` per
model, ≤ 2 GB per file, no setup. Use **`--host hf`** for files over 2 GB or large environments;
it needs `pip install huggingface_hub` and `hf auth login` by the user (no token here as of
2026-10-04). Release downloads lack CORS headers, so an in-browser 3D preview (`viewer_glb`)
must be served from Hugging Face or the repo, not from a release.

## Categories

Categories live in the zoo's `data/models.json` → `categories` (`id`, `label`, Material Symbols
`icon`). Add one there, insert as a single line to keep the file's layout; `validate.py` rejects
entries whose category is unknown. Current: drones, aircraft, ground, ships, animals,
infrastructure, vegetation, environments, props.

**Environments** use the same bundle: the scene zipped (USD stage + textures + referenced assets)
in `files/`, surface types (soil, grass, asphalt, foliage…) as `parts`, extent as `dimensions_m`;
prefer `--host hf` past a few hundred MB. `zoo_bundle.py` does not build environment bundles yet.
