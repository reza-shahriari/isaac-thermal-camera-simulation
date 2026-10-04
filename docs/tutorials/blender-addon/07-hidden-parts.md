# Add the parts you cannot see

A downloaded model is a skin. A car has no engine, and a drone shell has no battery. But a thermal
camera sees exactly the heat those parts make, through the shell. A **hidden part** is a real object
put where the real thing sits, which carries what the solver needs to know about it.

![The Phantom 4 with its battery, flight controller, video transmitter and four speed controllers added; the battery selected, taking irsim's lipo_pack component but keeping DJI's own mass](images/hidden_parts_panel.webp)

## Add one from the library

1. Select the part it sits inside, for example the car body or the drone's shell.
2. In **Hidden parts**, click **Add from library…** and pick a component. The menu has 97, in
   twelve groups: drones (batteries, motors, speed controllers, boards, an onboard computer),
   fixed-wing drones, aircraft (jet engines from a business jet's to a wide-body's, turboprop,
   piston engine, exhaust duct, APU, fuel tank, wheel brakes...), helicopters, cars, electric
   cars, trucks and buses, motorcycles, trains, ships and boats, **people and animals** (a driver
   or passenger is a heat source inside a vehicle too), and buildings and equipment. The dialog
   shows what the one you picked stands for and its size.
3. Give it a **material** from the library, its **Mass**, its **Idle heat** and **Max heat** (the
   heat it gives off, which for an engine is the fuel's power minus the useful work), whether those
   **Numbers are** *estimated*, *published* or *measured*, and a **Reference** saying where they came
   from.
4. Click **Add**. It arrives at its real size, in the middle of the selected part, and moves with
   that part.

For now each library component is a **placeholder**, a box, a cylinder or a cone at a typical
size. The files are in
[`blender_addon/irsim_thermal/components/`](../../../blender_addon/irsim_thermal/components/README.md),
with a table of every one: replace one with a detailed model of your own (keep its name, metres
and centre) and everything placed from it can use the real shape. To add a component the library
lacks, add one line to its `catalog.json` and run the script that README names; it builds the
placeholder and never overwrites a model you put in.

**Add a box…** does the same with a plain box of any size, for something the library does not have.

Move a hidden part with `G`, turn it with `R` and resize it with `S`, like any object. It is drawn
solid, with an outline, **in front of the shell**, so it is always visible and easy to grab.
Select it to edit its numbers in the **Hidden parts** section. If one ever seems to have
disappeared, **Show hidden parts** brings every one back into view.

The Phantom 4 in the picture has seven: the flight battery (DJI's published 468 g; up to 25 W
estimated from the current through the pack's internal resistance), the flight controller, the
video transmitter, and a speed controller in each arm, turned to lie along it. Every heat figure
there is an estimate, and each one says so.

## What happens to it

- **It is in the exported USD**, as a prim of its own that you can select and move in Isaac Sim.
  It is marked `purpose = "guide"`, which Isaac Sim's cameras skip: tested on 2026-09-28, a guide
  cube is absent from both the colour image and the depth, so neither the RGB companion nor the
  infrared camera shows a box inside the shell.
- It does not count towards the material coverage.
- It **is** a part for the connection finder ([step 6](06-connections.md)): the battery's contacts
  and facing pairs with the shell around it are the path by which its heat reaches the outside.
- It goes into the asset config's `hidden_parts:` ([step 9](09-export.md)), which irsim's thermal
  solve reads: its box, its material, and its component or its own mass and heat.
- The checklist asks for anything missing: a material, a mass (unless it names an irsim
  component), or a max heat below the idle heat.

## Take the numbers from irsim's component library

irsim keeps a library of components with **cited** numbers (`configs/components/`): a brushless
motor, a speed controller, a LiPo pack, a flight controller, a small piston engine, a micro
turbine, an exhaust line, an EV pack and drive motor, a brake disc, a differential and more. Each
says its mass, heat capacity, heat at idle and at full load, and where those came from.

Both dialogs have an **irsim component** field, and the **Hidden parts** panel a **Component**
field that is the same thing. Pick one and its numbers are shown under it; the part then takes them, and any number you type yourself overrides
that one alone (leave it at 0 to keep the component's). Library placeholders that match a
component pick it for you: the drone battery takes `lipo_pack`, the speed controller `esc`, the
motor `brushless_motor`, and so on. In the picture the battery takes `lipo_pack` but keeps DJI's
published 468 g in place of the component's 462 g. When nothing in the component library fits, the numbers are
yours, which is why the add-on asks where they came from.
