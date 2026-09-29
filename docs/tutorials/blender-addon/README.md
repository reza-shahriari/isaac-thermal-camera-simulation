# Infrared materials in Blender

This tutorial takes a 3D model into Blender, says what every part is made of, and hands it to irsim
with the **irsim Thermal Materials** add-on. You need no Python.

![The Phantom 4 as it looks, and as the add-on shows it: each part's brightness is its LWIR emissivity](images/phantom4_look_and_emissivity.webp)

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

## Steps

1. [Install the add-on](01-install.md), once.
2. [Open the model and make its parts real parts](02-parts.md): separate, join and rename, as in
   any Blender work.
3. [Assign materials](03-assign.md), part by part or face by face, and see the result in the
   thermal view.
4. [Start from an existing asset](04-existing-asset.md), when the model already has a material map.
5. [Add a material the library lacks](05-new-material.md), checked by irsim before it is written.
6. [Find the connections between parts](06-connections.md): which parts touch, and which face each
   other across a gap.
7. [Add the parts you cannot see](07-hidden-parts.md): the engine inside a car shell, the battery
   inside a drone.
8. [Check the model and its size](08-check.md).
9. [Export it to irsim](09-export.md) and run the project's own audit.
10. [See it all on a real model](10-phantom4.md): the DJI Phantom 4, done headless by one script.
11. [Troubleshooting](11-troubleshooting.md).

What the add-on is, in one page: [the Blender add-on](../../blender-addon.md). Where every
material's numbers come from, and how to check them: [the material library](../../materials.md).

The add-on's own plan, including what is not built yet, is
[`blender_addon/PLAN.md`](../../../blender_addon/PLAN.md).
