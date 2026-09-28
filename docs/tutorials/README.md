# Tutorials

Step-by-step guides to getting things done with irsim, one folder per tutorial and one page per
step. They are written for someone using the simulator, not working on its code: the technical
record is [the technical report](../../TECHNICAL_REPORT.md), and the physics is
[the physics model](../physics-model.md).

| tutorial | what you end up with |
|---|---|
| [Infrared materials in Blender](blender-addon/README.md) | A downloaded 3D model whose every part has its real infrared material, exported as an irsim asset, using the **irsim Thermal Materials** add-on |

## Writing one

A tutorial is a folder here with a `README.md` (what it is for, what you need, and the list of
steps) and one page per step named `NN-short-name.md`, read in the order of the number. Its images
go in the folder's `images/`, web-sized. The project site publishes the folder as it is
(`make site`): each step gets arrows to the one before and after it. The build names any page that
its `README.md` does not link to, and a test fails on it.
