# Check the model and its size

Click **Check model**. You get:

- **A progress bar**: the share of the model's surface area that has an infrared material. irsim
  needs at least 95 %; aim for 100 %. **Select unassigned** selects what is left.
- **A list of things to look at**. Each line has a button that selects the part:
  - parts still unassigned, with the Blender materials that are;
  - parts assigned a mirror-like material, which is right only for genuinely polished metal;
  - parts with made-up names (`Cube.003`, `GeometryNode_57`);
  - hidden parts missing a material, a mass or their heat;
  - connections still to review, or ones that lost a part.
- **The size** in metres, next to a familiar object ("about the size of a small drone").

## Size guidance is optional

Tick **Help me check the size** only if you want it:

- **What is it?** Pick the kind of object, and the add-on says whether the size is normal for it.
  It only warns when the model looks like a *units mistake*, for example a drone 46 m wide that
  would be 46 cm if the file was in centimetres. A 300 m ship is fine as a ship.
- **Real dimension.** If you know one (say DJI publishes a height of 196 mm), enter it and choose
  the axis. The add-on gives the exact scale, and says when it looks like a unit mix-up.
- **Apply scale** resizes the whole model and puts the scale into the mesh itself.
