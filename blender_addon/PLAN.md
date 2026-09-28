# irsim Thermal Materials — the Blender add-on's plan

The add-on is planned **here, not in `docs/roadmap.md`**. The owner set that on 2026-09-28: the main
roadmap stays the physics project's, and this file tracks the add-on. What the add-on needs *from*
the physics project is project work, and it is recorded as rows in `docs/roadmap.md` (`AI.11`,
`AI.12`, `TC.9`–`TC.12`) and in `TECHNICAL_REPORT.md`'s limitations. Nothing under `src/`,
`scripts/` or `tests/` was changed to build the add-on.

How to use it: the tutorial [`docs/tutorials/blender-addon/`](../docs/tutorials/blender-addon/README.md),
one page per step, also on the project site under Start ▸ Tutorials.

## What it is for

A downloaded model arrives as a skin grouped by *look*: "white plastic" on the shell and the
propellers, no battery, no engine. irsim needs every part's real material, because in the thermal
bands that decides what the camera sees: the Phantom 4's motor housings were once mapped to polished
aluminium (ε 0.09), which makes a hot motor read as reflected sky (ADR 0128). The `ingest-asset` skill
does that research as LLM work. This add-on is the **human half**: a person clicks a part, sees what
it is made of, and fixes it, part by part, in Blender, where the geometry can also be edited.

## Decisions, and why

1. **The assignment lives on Blender materials** (`Material.irsim_material`), not on objects.
   Blender carries a face's material through join, separate, extrude and duplicate, so ordinary
   modelling cannot lose an assignment. `configs/assets/<name>.yaml` already maps a source-material
   name to a library material (ADR 0128), so the export is a direct read.
2. **A shared material is copied, never silently changed.** When the selected part shares its
   Blender material with parts that were not selected, it gets a copy with the same look and its own
   thermal material, named `<origin>__<thermal>`. Parts that were not selected never change. The
   same applies to linked duplicates: the selected one's slot moves to object level.
3. **The add-on has no physics of its own.** Blender's Python lacks pydantic and PyYAML. More
   importantly, a second copy of the material schema or of Kirchhoff closure (CLAUDE.md #4) would
   drift from the one the solver enforces. So every library read and every file written goes
   through `irsim_thermal/bridge.py`, which runs under the project interpreter and calls the
   project's own loaders.
4. **The new-material form authors ε and τ per band; ρ is shown as derived.** It cannot author
   both ε and ρ, and the bridge refuses anything the library walk would refuse. Values are
   rounded to six significant figures, because Blender stores the form in single precision.
5. **Size is never forced.** By default the size is shown in metres next to a familiar object.
   Guidance appears only when asked. A warning needs both of these: the size is outside the typical
   range for the kind of object, **and** a unit factor (10, 100, 1000, 2.54, 0.3048) brings it
   inside. A real 300 m ship is never flagged. A known dimension gives the exact scale, and names the
   unit when the scale is within 10 % of one (the Phantom 4 was 5.6 % off, and still centimetres).
6. **The export makes names agree once, visibly.** Material, part and mesh names that USD would
   change are renamed first. They then read the same in the `.blend`, the USD and the YAML. The
   written USD is read back, and every material bound in it is checked against the map.
7. **The export never overwrites what it did not write.** A hand-written `configs/assets/*.yaml`
   (no add-on marker on its first line) is never replaced. Neither is a file in `3d_models/<name>/`
   that no export wrote. An earlier export is replaced only when *Replace earlier export* is ticked.
   All three are checked **before** anything is written.
8. **Tests are the add-on's own.** They are not in `make check`, which gates the physics project.
9. **Licence: the repository's own.** `blender_manifest.toml` declares
   `LicenseRef-irsim-use-only`, meaning `LICENSE` at the repository root (use-only, non-commercial,
   committed on 2026-09-28). The add-on does not choose a licence of its own (open question 2).
10. **Connections are proposed by a finder and decided by a person.** The finder scatters points
    by area over each part (so a shell's one huge triangle beside many slivers is weighed
    correctly) and counts a point as touching another part when that part's surface is within
    reach, roughly parallel and straight across; the contact area is the smaller of the two sides'
    estimates. A facing pair is the area whose outward normal reaches another part first, which
    says *which* parts exchange heat by radiation, not how much: the view factor is the project's
    `TC.9`. A person confirms, rejects, re-joints or adds; a new search updates what it found and
    never overrides a verdict, so a rejected pair stays out.
11. **A hidden part is a box, not geometry the camera sees.** It is marked not to render, is left
    out of the USD and of the coverage figure, and moves with the part it sits inside. This
    answers open question 3 the reversible way: nothing about it reaches the RGB companion, and if
    the owner wants it as a guide prim instead, only the export changes.
12. **Until the asset format has a place for them (`AI.11`), connections and hidden parts go in
    `3d_models/<name>/<name>.structure.yaml`,** beside the USD and checked by the bridge first
    with the refusals `AI.11` specifies (a missing part, a joint not in `joints.yaml`, a contact
    larger than its parts, a hidden part without mass). Nothing in irsim reads that file yet; it
    and the `.blend` copy keep the work until `AI.11` moves it into the asset config.

## Steps

| step | what | status | needs from the project |
|---|---|---|---|
| B1 | Sidebar tab; parts list; library browser with ε/ρ/τ per band; assign to parts, to faces in Edit Mode, or to every part using a material; thermal view | ✅ 2026-09-28 | — |
| B2 | New-material form, checked by irsim's own schema and closure before it writes `configs/materials/<name>.yaml` | ✅ 2026-09-28 | — |
| B3 | Checklist (area-weighted coverage, mirror-like parts, default names); optional size guidance with *Apply scale*; export to USD + `.blend` + asset YAML; `prep_asset.py` audit in the background | ✅ 2026-09-28 | — |
| B4 | Tutorial | ✅ 2026-09-28: a folder of pages, `docs/tutorials/blender-addon/` (the owner: "should have a folder for that not just in the github but also in the github.io site"), with pictures from the Phantom 4 demo and one real screenshot | — |
| B5 | **Reopen from an asset config.** *Load materials from an asset* applies an existing map (the Phantom 4's hand-written one, or an earlier export's) to the Blender materials of the same names -- raw or USD-safe -- keeping existing assignments unless told otherwise | ✅ 2026-09-28 | — |
| B6 | **Hidden parts.** Place a box inside the shell for a part the model lacks (motor, ESC, battery, electronics, piston engine, turbine, exhaust, gearbox, other); set its material, mass, idle and full-load heat, and where the numbers came from. Exported to the structure file, never to the USD | ✅ 2026-09-28, with the person's own numbers. **Next:** ready-made components with cited numbers once the project has a component library | `AI.12` (library); `AI.11` for irsim to read them |
| B7 | **Connection finder.** Parts that touch (with contact area) and parts that face each other across a gap are found automatically, shown as a list and as crosses and lines in the viewport; the person confirms, rejects, changes the joint (from `configs/thermal/joints.yaml`) or the area, or adds one | ✅ 2026-09-28 | `AI.11` for irsim to read them; `TC.9` for the view factor itself |
| B8 | **Main-object flag.** If `TC.11` wants "solve this asset fully" to be a property of the asset rather than the scene, the add-on offers it. Evolve/freeze (`TC.12`) and the exchange switch (`TC.10`) are scene settings, not asset ones | open | `TC.10`–`TC.12` |
| B9 | The project site shows the tutorial | ✅ 2026-09-28: `docs/tutorials/` is a section of the site (`scripts/build_site.py`), every step linked to the next | — |
| B10 | Checked in an interactive Blender: the panels on screen, and screenshots for the tutorial | partly: the owner's own session (the Sketchfab Phantom 4) showed the panels drawing and gave the *Parts* screenshot, and two fixes (below). Screenshots of *Connections* and *Hidden parts* with content still to take | — |

## Verification

```bash
# Blender-free logic and the bridge (72 tests; the bridge writes only to temporary directories)
~/IsaacSim/_build/linux-x86_64/release/python.sh -m pytest blender_addon/tests -q

# End to end in a headless Blender, against a scratch copy of the repository (134 checks)
blender_addon/tests/run_blender_smoke.sh            # add --skip-audit to leave out prep_asset

# The real Phantom 4 (needs the git-ignored 3d_models/phantom4.fbx): a .blend to open, CPU
# pictures, the hidden battery and the connection finder, and an export + prep_asset audit of a
# 2.5-million-face asset
blender -b --factory-startup --python blender_addon/demo/make_phantom4_demo.py -- \
    --out outputs/blender_addon_demo --render --structure --export-scratch /tmp/irsim_phantom_export
```

What the end-to-end run proves, on a toy quadcopter modelled in centimetres:

- **The library.** It loads through the bridge with the band values the solver uses, and it
  closes in every band.
- **Assignment.** Assigning a part never changes an unselected part or a linked duplicate that
  shares its material. Assigning faces in Edit Mode changes exactly the selected faces.
- **Modelling.** Join and separate-by-material keep the area of every thermal material, to 1e-6.
- **Thermal view.** It changes only viewport colours and restores them exactly.
- **Size.** A 54.5-unit drone is recognised as centimetres, and *Apply scale* puts the scale into
  the mesh data.
- **New materials.** A new material is written. A duplicate name, and ε + τ > 1, are refused by
  irsim's own check, and nothing is written for them.
- **Export.** Every material bound in the exported USD is in the map. The export refuses to replace
  an earlier export unless asked, a hand-written config, or geometry it did not write.
- **Panels.** Every panel, list and form draws, with each layout call checked against Blender's
  own API.
- **The project's audit.** `scripts/prep_asset.py` passes the exported asset: 8/8 prims mapped,
  100 % coverage.
- **Connections, on shapes whose answers are exact.** A 0.5 m box standing on a 1 m block touches
  it over 0.25 m² (found 0.245) and is not also listed as facing it. Two 1 m plates 0.2 m apart
  face each other fully (1.000 and 1.000 m², gap 0.2000 m), a third plate behind one of them is
  hidden from the other, and a cube 10 m away is connected to nothing. A confirmed contact stays
  confirmed through a new search, with the joint the person chose, and a rejected pair stays out.
- **Hidden parts.** One is placed in the middle of the selected part at the size asked for, is
  never rendered, is not in the exported USD, and is flagged by the checklist until it has a mass.
- **The structure file.** The confirmed contact is written with its joint, the rejected pair is
  left out, an unreviewed pair is written as unreviewed, and a joint the table lacks is refused
  before anything is written.

## Open questions for the owner

1. *(Answered 2026-09-28: the tutorial is a folder, on GitHub and on the site.)*
2. **The licence.** The folder now carries the repository's use-only licence. The Blender
   Foundation's position is that add-ons importing `bpy` are GPL when distributed, so if the add-on
   is ever shared outside the repository, its licence is a decision for you, possibly with advice.
   Using it privately is unaffected.
3. **Hidden parts (B6).** Should a component's box appear in the exported USD at all? Decided for
   now (decision 11): no, it goes in the structure file only, so the RGB companion never shows it.
   A guide prim in the USD is the alternative, if the owner prefers it.

## Log

- **2026-09-28 (third commit)**: B6 hidden parts, B7 connection finder, B9 the tutorial as a folder
  on the site.
  - On the real Phantom 4, with the flight battery added inside the body (DJI's 468 g; heat
    estimated, and labelled so), the finder searches 2.5 million faces in 9.9 s and finds 103
    contacts and 87 facing pairs. The battery touches the middle shell over 29 cm² and faces
    64 cm² of it. Exported through the add-on with its structure file, the model still passes
    `prep_asset.py`, 41/41.
  - The first picture drew every contact as a line through both parts' centres, and 103 contacts
    made a web. Contacts are now small crosses where the parts touch; only the one selected in
    the list gets lines to its parts.
  - The first contact rule counted a strip beside a box standing on a block (within reach of the
    box's bottom edge): 0.27 m² for a 0.25 m² footprint. A point now counts only when the other
    surface is straight across as well as parallel: 0.245 m².
  - A ray starting on a contact starts inside the other part, and reported its far side as
    "facing". Points in contact are now left out of the facing search.
  - From the owner's own session on the Sketchfab Phantom 4: its 41 parts are all called
    `GeometryNode_<n>`, which the checklist did not flag (it now does, with 3ds Max's `Box001`
    and Maya's `polySurface12`), and the checklist's lines cut off the material names (they now
    come first).

- **2026-09-28 (second commit)**: B5, reopening from an asset config, plus the Phantom 4 demo.
  - The real FBX (2,486,459 faces in 41 prims) is measured in 0.35 s and read as centimetres. The
    scaled model is 0.4105 × 0.4637 × 0.2067 m, the extent ADR 0128 found by hand.
  - The hand-written map covers 100 %. The one mirror-like material flagged is the chrome trim
    that the map chose on purpose.
  - Exported through the add-on, the model passes `prep_asset.py`: 41/41 prims mapped.
  - The thermal view now makes brightness *on screen* equal ε (sRGB decoding), so ε 0.09 shows
    black rather than mid-grey. Rendered, the old `*metal*` rule's mistake (motor housings as
    bare aluminium) is four black motors.

- **2026-09-28**: B1–B4 shipped. The add-on is loaded in place from `blender_addon/` as a Blender
  local extension repository (checked headless with a throwaway user config). It found the
  repository and Isaac Sim's `python.sh` without configuration and loaded all 23 library materials.
  The end-to-end run and `prep_asset.py` pass. One bug was caught by the run and fixed before
  commit: the export used to refuse a hand-written config only *after* writing the geometry.
