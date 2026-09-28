# irsim Thermal Materials — the Blender add-on's plan

The add-on is planned **here, not in `docs/roadmap.md`**. The owner set that on 2026-09-28: the main
roadmap stays the physics project's, and this file tracks the add-on. What the add-on needs *from*
the physics project is project work, and it is recorded as rows in `docs/roadmap.md` (`AI.11`,
`AI.12`, `TC.9`–`TC.12`) and in `TECHNICAL_REPORT.md`'s limitations. Nothing under `src/`,
`scripts/` or `tests/` was changed to build the add-on.

How to use it: [`TUTORIAL.md`](TUTORIAL.md).

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

## Steps

| step | what | status | needs from the project |
|---|---|---|---|
| B1 | Sidebar tab; parts list; library browser with ε/ρ/τ per band; assign to parts, to faces in Edit Mode, or to every part using a material; thermal view | ✅ 2026-09-28 | — |
| B2 | New-material form, checked by irsim's own schema and closure before it writes `configs/materials/<name>.yaml` | ✅ 2026-09-28 | — |
| B3 | Checklist (area-weighted coverage, mirror-like parts, default names); optional size guidance with *Apply scale*; export to USD + `.blend` + asset YAML; `prep_asset.py` audit in the background | ✅ 2026-09-28 | — |
| B4 | Tutorial | ✅ text 2026-09-28; screenshots wait for B9 | — |
| B5 | **Hidden parts.** Pick a predefined component (motor, ESC, battery, piston or turbine engine, exhaust line; car and ship parts later), place it as a proxy box inside the shell, and edit its mass, idle and rated heat. The component data comes from the project library, not from the add-on | planned | `AI.11` (schema), `AI.12` (library) |
| B6 | **Connection finder.** Parts that touch (with contact area) and parts that face each other across a gap are found automatically. They are shown as a list and as lines in the viewport; the user confirms, deletes, or changes the joint type (from `configs/thermal/joints.yaml`) | planned | `AI.11` (contacts), `TC.9` (facing) |
| B7 | **Main-object flag.** If `TC.11` wants "solve this asset fully" to be a property of the asset rather than the scene, the add-on offers it. Evolve/freeze (`TC.12`) and the exchange switch (`TC.10`) are scene settings, not asset ones | open | `TC.10`–`TC.12` |
| B8 | The project site shows the tutorial | waiting for the owner: it needs a change to the site builder, a project file | — |
| B9 | Checked in an interactive Blender: the panels on screen, and screenshots for the tutorial | pending: no interactive Blender was running | — |

## Verification

```bash
# Blender-free logic and the bridge (48 tests; the bridge writes only to temporary directories)
~/IsaacSim/_build/linux-x86_64/release/python.sh -m pytest blender_addon/tests -q

# End to end in a headless Blender, against a scratch copy of the repository (66 checks)
blender_addon/tests/run_blender_smoke.sh            # add --skip-audit to leave out prep_asset
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

## Open questions for the owner

1. **The site (B8).** Should the tutorial be published on the project site? The site builder has a
   fixed document list, so this means editing `scripts/build_site.py` or `site/gallery.yaml`, which
   are project files.
2. **The licence.** The folder now carries the repository's use-only licence. The Blender
   Foundation's position is that add-ons importing `bpy` are GPL when distributed, so if the add-on
   is ever shared outside the repository, its licence is a decision for you, possibly with advice.
   Using it privately is unaffected.
3. **Hidden parts (B5).** Should a component's proxy box appear in the exported USD as an ordinary
   prim, or as a guide prim that the camera never renders? The physics needs its position and size;
   the RGB companion should not show a box inside the shell.

## Log

- **2026-09-28**: B1–B4 shipped. The add-on is loaded in place from `blender_addon/` as a Blender
  local extension repository (checked headless with a throwaway user config). It found the
  repository and Isaac Sim's `python.sh` without configuration and loaded all 23 library materials.
  The end-to-end run and `prep_asset.py` pass. One bug was caught by the run and fixed before
  commit: the export used to refuse a hand-written config only *after* writing the geometry.
