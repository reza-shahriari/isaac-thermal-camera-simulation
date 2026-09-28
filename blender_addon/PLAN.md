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
11. **A hidden part is a real object that no camera renders.** The owner, 2026-09-28: "better to
    have the hidden files inside the usd as well, even not visible while rendering, its movable
    inside isaac", and "having a real cube would be better". So in Blender it is a solid, outlined
    and drawn in front of the shell (never a wireframe that is easy to lose), and in the USD it is
    a prim of its own marked `purpose = "guide"`. Tested in Isaac Sim 6.0 the same day: a guide
    cube, and a cube under a guide parent, are absent from both the RGB image and the depth, while
    an ordinary cube beside them renders. It stays out of the coverage figure and moves with the
    part it sits inside. It may be a placeholder from `irsim_thermal/components/` (a box, cylinder
    or cone at real size, 97 of them in twelve groups) that the owner later replaces with a
    detailed model.
12. **Until the asset format has a place for them (`AI.11`), connections and hidden parts go in
    `3d_models/<name>/<name>.structure.yaml`,** beside the USD and checked by the bridge first
    with the refusals `AI.11` specifies (a missing part, a joint not in `joints.yaml`, a contact
    larger than its parts, a hidden part without mass). Nothing in irsim reads that file yet; it
    and the `.blend` copy keep the work until `AI.11` moves it into the asset config. **For
    `AI.11`:** `scripts/prep_asset.py` reads an asset back through Blender's USD importer, which
    skips guide prims by default, so the prepared thermal mesh has no hidden parts in it; the
    project side will need to read them (`import_guide`, or from the USD directly).

## Steps

| step | what | status | needs from the project |
|---|---|---|---|
| B1 | Sidebar tab; parts list; library browser with ε/ρ/τ per band; assign to parts, to faces in Edit Mode, or to every part using a material; thermal view | ✅ 2026-09-28 | — |
| B2 | New-material form, checked by irsim's own schema and closure before it writes `configs/materials/<name>.yaml` | ✅ 2026-09-28 | — |
| B3 | Checklist (area-weighted coverage, mirror-like parts, default names); optional size guidance with *Apply scale*; export to USD + `.blend` + asset YAML; `prep_asset.py` audit in the background | ✅ 2026-09-28 | — |
| B4 | Tutorial | ✅ 2026-09-28: a folder of pages, `docs/tutorials/blender-addon/` (the owner: "should have a folder for that not just in the github but also in the github.io site"), with pictures from the Phantom 4 demo and one real screenshot | — |
| B5 | **Reopen from an asset config.** *Load materials from an asset* applies an existing map (the Phantom 4's hand-written one, or an earlier export's) to the Blender materials of the same names -- raw or USD-safe -- keeping existing assignments unless told otherwise | ✅ 2026-09-28 | — |
| B6 | **Hidden parts.** Put a part the model lacks inside the shell: from the placeholder library (97 components in twelve groups, drone to ship and people, real size) or as a box; set its material, mass, idle and max heat, and where the numbers came from. Solid in Blender; a guide prim in the USD; its numbers in the structure file | ✅ 2026-09-28, with the person's own numbers. **Next:** detailed models in place of the placeholders (the owner's, "a unique beautiful engine"), and cited numbers once the project has a component library | `AI.12` (library); `AI.11` for irsim to read them |
| B7 | **Connection finder.** Parts that touch (with contact area) and parts that face each other across a gap are found automatically, shown as a list and as crosses and lines in the viewport; the person confirms, rejects, changes the joint (from `configs/thermal/joints.yaml`) or the area, or adds one | ✅ 2026-09-28 | `AI.11` for irsim to read them; `TC.9` for the view factor itself |
| B8 | **Main-object flag.** If `TC.11` wants "solve this asset fully" to be a property of the asset rather than the scene, the add-on offers it. Evolve/freeze (`TC.12`) and the exchange switch (`TC.10`) are scene settings, not asset ones | open | `TC.10`–`TC.12` |
| B9 | The project site shows the tutorial | ✅ 2026-09-28: `docs/tutorials/` is a section of the site (`scripts/build_site.py`), every step linked to the next | — |
| B10 | Checked in an interactive Blender: the panels on screen, and screenshots for the tutorial | ✅ 2026-09-28, in the owner's own session through the Blender MCP (the Sketchfab Phantom 4, 48 parts with the hidden ones): three screenshots in the tutorial, and the fixes below | — |

## Verification

```bash
# Blender-free logic and the bridge (72 tests; the bridge writes only to temporary directories)
~/IsaacSim/_build/linux-x86_64/release/python.sh -m pytest blender_addon/tests -q

# End to end in a headless Blender, against a scratch copy of the repository (142 checks)
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
- **Hidden parts.** One is placed in the middle of the selected part at the size asked for; one
  from the library arrives at its real size (0.14 × 0.07 × 0.04 m for the drone battery). Both are
  solid and drawn in front, are in the exported USD as guide prims while the ordinary parts are
  not, and are flagged by the checklist until they have a mass. An old wireframe one is made
  solid and exportable again by *Show hidden parts*.
- **Many connections.** With a part selected, the list and the viewport show only its
  connections.
- **The structure file.** The confirmed contact is written with its joint, the rejected pair is
  left out, an unreviewed pair is written as unreviewed, and a joint the table lacks is refused
  before anything is written.

## Open questions for the owner

1. *(Answered 2026-09-28: the tutorial is a folder, on GitHub and on the site.)*
2. **The licence.** The folder now carries the repository's use-only licence. The Blender
   Foundation's position is that add-ons importing `bpy` are GPL when distributed, so if the add-on
   is ever shared outside the repository, its licence is a decision for you, possibly with advice.
   Using it privately is unaffected.
3. *(Answered 2026-09-28: hidden parts go in the USD, as guide prims; decision 11.)*
4. **Its own repository.** The add-on is outgrowing a folder of the physics repository. The
   proposal: a repository of its own, wired back in as a git submodule at `blender_addon/` (the
   way companion projects are kept), with the tutorial staying in `docs/tutorials/` so the site
   still shows it. Waiting for the owner: the repository's name, and whether it is public.

## Log

- **2026-09-28 (fourth commit)**: hidden parts in the USD, the placeholder library, and the panels
  checked in the owner's own Blender.
  - Isaac Sim probe (A6000): three cubes, one ordinary, one `guide`, one under a `guide` parent.
    The RGB image and the depth show only the ordinary one.
  - The Phantom 4 demo takes its battery from the library; exported, the battery is at
    `…/GeometryNode_205/battery` with purpose guide, and the audit still passes, 41/41.
  - In the owner's scene: the model scaled from centimetres (0.4105 × 0.4637 × 0.2067 m), seven
    hidden parts placed along the aircraft's real heading (the model sits turned by 119°), the
    hand-written map applied (100 %), and 108 contacts and 120 facing pairs found among 48 parts
    in 10.5 s. Safety copies of the scene were saved first.
  - **A crash, found by the smoke test:** the search's merge held Python references to connections
    while it removed and added others, and Blender may move a collection's items when it changes;
    writing through the old reference is a write into freed memory. Entries are now always looked
    up again by index, and existing ones are updated before any new one is added.
  - From the screenshots: 228 connections are unreadable at once, so the list and the viewport
    now show only the selected parts' connections by default, each row named by the other part;
    four cut-off labels were shortened; after a search the panel says which distances it used.
  - The owner asked for many more library components: 15 became 97, grouped in the menu under
    twelve headings (drone, fixed-wing drone, aircraft, helicopter, car, electric car, truck and
    bus, motorcycle, train, ship and boat, people and animals, buildings and equipment). Each
    entry's shape and size are in `catalog.json`; `tools/make_placeholders.py` builds the missing
    FBX files (never over an existing one without `--force`), imports each back to check its size
    and writes the README's table. The smoke test imports every one through the add-on. Sizes are
    estimates for a typical example, as the README says.

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
