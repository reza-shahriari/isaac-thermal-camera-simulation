# 0150 — Assets arrive from a link through a licence gate

Date: 2026-09-26
**Status:** Accepted
Roadmap: AI.8, AI.9

## Context

The owner's standing goal is a **library of ingested 3D objects**: share a model link with any
session, and a cleaned, part-decomposed asset with its material map comes out the other end —
the Phantom 4's hand-run route (fetch by hand, `prep_asset.py`, a hand-written
`configs/assets/phantom4.yaml`), made repeatable. Sky targets first, per the application order.

Three facts constrain the front door, none of them visible from inside the repo:

1. **Sketchfab's Download API serves glTF/GLB/USDZ only.** "Models are not currently available
   in their source formats such as FBX and OBJ through the API." The original FBX exists only
   behind the website's manual download, when the author enabled it. Store content moved to Fab,
   which has no public download API as of 2026-09.
2. **A cleaned asset is a derivative.** The pipeline rescales, regroups into functional parts,
   renames and re-exports. CC-BY-ND permits redistribution only "unchanged and in whole";
   Sketchfab's Standard licence forbids making the model available "as a stand-alone file";
   the NC variants forbid commercial use. All of these *download* without complaint — the
   defect appears only if the file is later committed, bundled or published, and no render
   would ever reveal it.
3. **The platform is not durable.** Sketchfab is on its second owner in two years and promises
   free-download support only until further notice. The library, not the link, has to be the
   artefact of record.

## Decision

**glTF in, FBX out.** The automated path ingests the Download API's glTF archive.
A "well-formatted FBX" is something `prep_asset.py` *exports* from the cleaned Blender scene,
alongside the master `.blend` and the `.usdc` that Isaac Sim references (the canonical pair);
the FBX is an interchange courtesy, never the file scenes load. Owner's decision 2026-09-26.

**The licence is gated before the download, and the gate fails towards quarantine.**
`irsim.io.sketchfab.gate_license` marks exactly **CC0, CC-BY and CC-BY-SA** shareable; ND, the
NC family, Standard, Editorial and *anything it does not recognise* go to
`3d_models/quarantine/<name>/`, usable locally and never committed, uploaded or published.
Widening the shareable set is an ADR-level change, and a unit test pins the set so a refactor
cannot widen it silently.

**Provenance is a first-class record.** Every fetch writes `ATTRIBUTION.md` beside the files
(creator + link, model + link, licence + link, indication of changes — what the Download API
guidelines require to follow the asset) and a `Provenance` YAML — for a shareable model at
`configs/assets/<name>.provenance.yaml`, committed, with the archive's sha256 and download date
pinning *which* upload it was. A quarantined model's record stays inside the quarantine
directory, so nothing committed ever points at it.

**Binaries do not travel by git.** `3d_models/` and `data/assets/` stay gitignored; the owner
distributes library binaries by cloud-drive link. What git carries per asset is the pair of
authored YAMLs (material map + provenance) — enough for anyone with the binaries to reproduce
every generated artefact with `prep_asset.py`.

**The network stays out of `irsim`.** `irsim.io.sketchfab` computes on URL strings and parsed
API responses; `scripts/fetch_sketchfab.py` owns HTTP (stdlib `urllib`, no new dependency),
downloads within the ~5-minute link expiry, retries 429s with backoff, and fetches one model
per invocation — a deliberate cadence, not a bulk scraper, matching the API guidelines.

## Consequences

* "Download the FBX from Sketchfab" — the phrasing the pipeline was first asked for — is
  recorded here as impossible via the API, so nobody re-litigates it against a 403.
* The gate makes full automation safe on the axis automation cannot see: the legal one. The
  price is that an ND- or NC-licensed model the owner wants anyway arrives quarantined; using
  it stays possible locally, sharing it does not, and the QUARANTINED.md in the directory says
  which and why.
* An unrecognised licence quarantines a possibly-shareable model until a human reads the terms
  — the cheap direction of wrong.
* A re-uploaded model under the same UID no longer matches its recorded sha256, which is
  detectable instead of silent.
* The `ingest-asset` skill (AI.9) is the checklist that drives this front door and the existing
  `prep_asset.py` machinery; nothing in it duplicates ADR 0128's toolchain.
