# Add the parts you cannot see

A downloaded model is a skin. A car has no engine, and a drone shell has no battery. But a thermal
camera sees exactly the heat those parts make, through the shell. A **hidden part** is a box put
where the real thing sits, which carries what the solver needs to know about it.

## Add one

1. Select the part it sits inside, for example the car body or the drone's shell.
2. In **Hidden parts**, click **Add hidden part…** and fill in:
   - **Name** and **What is it** (battery, electric motor, piston engine, turbine, exhaust…).
   - **Size**: it starts at a quarter of the selected part; change it here or later with `S`.
   - **Material**: what its outside is made of, from the library.
   - **Mass**, **Heat at idle** and **Heat at full load**. The heat is what it gives off, which
     for an engine is the fuel's power minus the useful work.
   - **Numbers are** *estimated*, *published* or *measured*, and a **Reference** saying where they
     came from, or what they were estimated from.
3. Click **Add**. The box appears in the middle of the selected part, drawn as an orange wireframe
   in front of everything, and it moves with that part.

Move it with `G`, turn it with `R` and resize it with `S`, like any object. Select it to edit its
numbers in the **Hidden parts** section.

The Phantom 4 demo adds its flight battery this way. The mass is DJI's published 468 g. The heat,
up to 25 W at full climb, is estimated from the current through the pack's internal resistance,
and the reference says so.

## What happens to it

- It is **not** exported as geometry, so the RGB companion never shows a box inside the shell,
  and it does not count towards the material coverage.
- It **is** a part for the connection finder. On the Phantom 4 the battery touches the middle
  shell over 29 cm² and faces 64 cm² of it across the gap. That is the path by which its heat
  reaches the outside.
- Its box, material and numbers are exported beside the model ([step 9](09-export.md)).
- The checklist asks for anything missing: a material, a mass, or full-load heat below idle.

A library of ready-made components with cited numbers is planned (roadmap row AI.12). Until then
the numbers are yours, which is why the add-on asks where they came from.
