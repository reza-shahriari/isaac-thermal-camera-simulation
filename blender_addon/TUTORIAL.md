# Giving a model its infrared materials in Blender

This tutorial shows how to take a 3D model into Blender, say what every part is made of, and
hand it to irsim with the **irsim Thermal Materials** add-on. You need no Python.

**Why it matters.** A thermal camera does not see colour. It sees how hot each surface is and how
much that surface *emits*. A matte black plastic emits well and shows its own temperature. Polished
aluminium hardly emits at all, so it mostly shows a reflection of whatever is around it, such as a
cold sky. Get the material wrong and a hot motor can read as cold sky. So every part needs its real
material, and a downloaded model almost never says what that is.

## What you need

- **Blender 5.2** (the version the irsim asset tools use).
- **This repository** checked out: the folder with `configs/`, `src/` and `blender_addon/` in it.
- **The project's Python**, which is Isaac Sim's `python.sh`
  (`~/IsaacSim/_build/linux-x86_64/release/python.sh`). Any Python 3.10 or newer that has irsim
  installed also works. The add-on uses it to read the material library and to check new materials
  with irsim's own rules.

## 1. Install the add-on (once)

The add-on is loaded straight from the repository, so a `git pull` also updates it.

1. In Blender: **Edit ▸ Preferences ▸ Get Extensions**.
2. Open the **Repositories** drop-down (top right) and click **+ ▸ Add Local Repository**.
3. Tick **Custom Directory** and choose the repository's `blender_addon` folder. Click
   **Create**.
4. Go to **Add-ons**, find **irsim Thermal Materials** and tick it.
5. Expand it. **irsim repository** and **irsim Python** are usually found on their own; the line
   under each says what is being used. If either says *not found*, set it by hand.
6. Click **Test connection**. You should see *OK: 23 materials in …/configs/materials* (the
   number grows as the library does).

## 2. Open your model

Import it as usual: **File ▸ Import** (FBX, glTF/GLB, OBJ or USD). Then press **N** in the 3D
viewport and pick the **irsim** tab in the sidebar. Click **Load library**.

The tab has three sections:

- **Parts**: every mesh object in the scene, each with its thermal status.
- **Material library**: every material irsim knows, with its numbers.
- **Check and export**: what is left to do, and the button that hands the model to irsim.

## 3. Make the parts real parts

irsim treats every **object** as a part, and the part's name is how a scene refers to it later
(`prim: battery`). Downloaded models are often grouped by look rather than by function: one object
holds all the white plastic, and another holds the four propellers together. Use Blender as you
normally would:

- **Separate** (`P` in Edit Mode) to split a propeller away from the arm.
- **Join** (`Ctrl J`) to merge pieces that are really one part.
- **Rename** (`F2`) to what the part really is: `battery`, `propeller_front_left`, `motor_rear`.

You can do this before or after assigning materials. **Assignments follow the faces**, so joining,
separating, extruding or duplicating keeps every face's material.

## 4. Assign materials

1. Click a part in the **Parts** list, or in the viewport. The box under the list shows its Blender
   materials and whether each one has an infrared material yet.
2. In the **Material library**, click the material the part is really made of. The box below it
   shows:
   - **ε / ρ / τ per band**: how much the surface **emits**, **reflects** and **transmits** in
     NIR, SWIR, MWIR and LWIR. The three always add up to 1.
   - **Solar absorptivity**: how strongly sunlight heats it.
   - **Heat capacity**: how slowly it warms up and cools down.
   - A red warning if LWIR ε is below 0.2 (mirror-like).
3. Click one of:
   - **Assign to selected parts**: every face of the selected objects.
   - **Assign to every part using '<material>'**: every object that uses the active Blender
     material. Useful when a download's "white plastic" really is one material everywhere.
   - In **Edit Mode**, **Assign to selected faces**: for a part made of two things, such as a wheel
     with a rubber tyre and a metal rim.

**If two parts shared a Blender material** and you give one of them a different infrared material,
that part gets a copy of the material: same look, own infrared material, named
`white_plastic__carbon_fibre`. The other part is not changed. Parts you did not select are never
changed by an assignment.

**Thermal view** (checkbox at the top) colours each part so that its brightness on screen is its
LWIR emissivity: white emits, black is a mirror. Anything still unassigned shows in **magenta**. It
changes only the viewport colours; untick it and the model looks as before. A part with no material
at all cannot be coloured and stays grey. The checklist lists it instead.

This view is how a wrong material shows up. On the Phantom 4, an old automatic rule once made the
motor housings polished aluminium (ε 0.09). In the thermal view they turn black: a hot motor would
have shown up as reflected sky. The demo below renders exactly that.

### Starting from an existing asset

If the model already has an asset config (the Phantom 4 has a hand-written one in
`configs/assets/phantom4.yaml`), click **From an existing asset…** under the library and pick it.
Every Blender material whose name is in that map gets its material, whether the name matches as
written or in the form USD uses. Materials you have already assigned are kept, unless you tick
**Replace existing assignments**. If the asset records a scale (the Phantom 4 was in centimetres),
the message says so; apply it from the size check.

A model you exported with this add-on reopens with everything in place: open
`3d_models/<name>/<name>.blend`.

## 5. When the material is not in the library

Click **New material…**, or **New from this…** to start from the selected material's values. Fill
in:

- **Name**: lower case with underscores, for example `nylon_black`.
- **Description**, **Source** (*estimated*, *literature* or *measured*) and **Reference**: where
  the numbers came from. Even "estimated from similar polymers" helps the next person.
- **Surface state**, for example *as manufactured*, *painted* or *anodised*. Emissivity belongs to
  the surface, not to the substance under it.
- **Heat storage**: density, specific heat, conductivity, the thickness of the skin that holds the
  heat, and solar absorptivity.
- **Per band**: emissivity ε and transmittance τ. **Reflectance ρ is shown, not typed in**: it is
  always 1 − ε − τ. That is how irsim keeps every material physically consistent.

When you click **Check and create**, irsim itself checks the material. If a band does not add up,
or the name is taken, nothing is written and the message says why. If the check passes, the file
appears as `configs/materials/<name>.yaml`; commit it like any other change. With *Assign it to the
selected parts* ticked, the new material is assigned to them straight away.

## 6. Check

Click **Check model**. You get:

- **A progress bar**: the share of the model's surface area that has an infrared material. irsim
  needs at least 95 %; aim for 100 %. **Select unassigned** selects what is left.
- **A list of things to look at**. Each line has a button that selects the part:
  - parts still unassigned;
  - parts assigned a mirror-like material, which is right only for genuinely polished metal;
  - parts with made-up names (`Cube.003`).
- **The size** in metres, next to a familiar object ("about the size of a small drone").

**Size guidance is optional.** Tick **Help me check the size** only if you want it:

- **What is it?** Pick the kind of object, and the add-on says whether the size is normal for it.
  It only warns when the model looks like a *units mistake*, for example a drone 46 m wide that
  would be 46 cm if the file was in centimetres. A 300 m ship is fine as a ship.
- **Real dimension.** If you know one (say DJI publishes a height of 196 mm), enter it and choose
  the axis. The add-on gives the exact scale, and says when it looks like a unit mix-up.
- **Apply scale** resizes the whole model and puts the scale into the mesh itself.

## 7. Export to irsim

1. Type an **Asset name** (lower case with underscores, for example `mavic3`). This is the name
   scenes will use.
2. Click **Export to irsim**. It writes:
   - `3d_models/<name>/<name>.usdc`: the parts, one USD prim per object. (`3d_models/` is not
     committed.)
   - `3d_models/<name>/<name>.blend`: a copy of your file, so you can come back and edit.
   - `configs/assets/<name>.yaml`: the asset's material map. **Commit this file.**

   Names that USD would change (spaces, dots) are renamed first, and the message lists them. That
   way the `.blend`, the USD and the YAML all say the same thing.
3. Click **Run irsim audit**. This runs the project's own `scripts/prep_asset.py` on the export in
   the background. Blender stays usable, and the result appears under the button and in a text
   block called `irsim_audit_<name>.log` (open it in Blender's Text Editor). *passed* means every
   part resolves to a library material.

Exporting again under the same name needs **Replace earlier export**. The add-on never overwrites
an asset config that was written by hand, or files in `3d_models/` that it did not write. Choose
another name instead.

## 8. What happens next

The asset config is what the rest of irsim reads.
`scripts/prep_asset.py --asset <name> --emit-mesh` prepares the thermal mesh archive that scenes
solve on. `configs/scenes/phantom4_parts.yaml` shows how a scene binds an asset's parts by name.

## See it on a real model

`blender_addon/demo/make_phantom4_demo.py` does all of the above, headless, on the DJI Phantom 4
(it needs the git-ignored `3d_models/phantom4.fbx`):

```bash
blender -b --factory-startup --python blender_addon/demo/make_phantom4_demo.py -- \
    --out outputs/blender_addon_demo --render
```

It writes `phantom4_thermal_demo.blend`, which opens in the thermal view with the sidebar out, plus
pictures of the model as it looks and of its emissivity in three states: nothing assigned, the old
automatic rule's mistake, and the hand-written map. The grey pictures show **emissivity, not
temperature**; they are the add-on's view, not a thermal image.

## Not yet in the add-on

The plan is in [`PLAN.md`](PLAN.md). Two features are next, and both wait on project work that is
on the main roadmap:

- **Hidden parts.** A downloaded car has no engine and a drone shell has no battery. You will be
  able to add them from a list of predefined components and set their heat output. This needs the
  asset format to carry them (`AI.11`) and a component library (`AI.12`).
- **Connections.** Which parts touch, and which face each other across a gap, will be found
  automatically and shown for you to confirm. This needs `AI.11` and view factors between arbitrary
  shapes (`TC.9`).

## Troubleshooting

- **Load library does nothing, or reports an error.**
  - Check **Preferences ▸ Add-ons ▸ irsim Thermal Materials ▸ Test connection**.
  - *No module named numpy*: the Python set there is not the project's. Use `python.sh`, not the
    `kit/python/bin/python3` inside it, because only `python.sh` sets up its environment.
- **Export is refused.** The message says why:
  - below 95 % coverage;
  - the name is taken by a hand-written asset;
  - an earlier export exists and *Replace earlier export* is not ticked;
  - the folder holds files the add-on did not write.
- **A part stays grey in the thermal view.** It has no material at all. Assign it one and it gets a
  plain material carrying the infrared one.
