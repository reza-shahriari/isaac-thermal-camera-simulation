# Export it to irsim

1. Type an **Asset name** (lower case with underscores, for example `mavic3`). This is the name
   scenes will use.
2. Click **Export to irsim**. It writes:
   - `3d_models/<name>/<name>.usdc`: the parts, one USD prim per object. (`3d_models/` is not
     committed.) The hidden parts are in it too, marked `purpose = "guide"`: you can select and
     move them in Isaac Sim, and no camera renders them.
   - `3d_models/<name>/<name>.blend`: a copy of your file, so you can come back and edit.
   - `configs/assets/<name>.yaml`: the asset's material map. **Commit this file.**
   - `3d_models/<name>/<name>.structure.yaml`, if there are any: the contacts, facing pairs and
     hidden parts. Rejected connections are left out, and unreviewed ones are marked
     `reviewed: false`. It sits beside the model until the asset format has a place for it
     (roadmap row AI.11).

   Names that USD would change (spaces, dots) are renamed first, and the message lists them. That
   way the `.blend`, the USD and the YAML all say the same thing.
3. Click **Run irsim audit**. This runs the project's own `scripts/prep_asset.py` on the export in
   the background. Blender stays usable, and the result appears under the button and in a text
   block called `irsim_audit_<name>.log` (open it in Blender's Text Editor). *passed* means every
   part resolves to a library material.

Before anything is written, everything is checked: the coverage, the name, and every connection
and hidden part (a joint that is not in the table, a contact larger than its parts, a hidden part
without a mass). If anything is refused, nothing is written.

Exporting again under the same name needs **Replace earlier export**. The add-on never overwrites
an asset config that was written by hand, or files in `3d_models/` that it did not write. Choose
another name instead.

## What happens next

The asset config is what the rest of irsim reads.
`scripts/prep_asset.py --asset <name> --emit-mesh` prepares the thermal mesh archive that scenes
solve on, and [`configs/scenes/phantom4_parts.yaml`](../../../configs/scenes/phantom4_parts.yaml)
shows how a scene binds an asset's parts by name.
