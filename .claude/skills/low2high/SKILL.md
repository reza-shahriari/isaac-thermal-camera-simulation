---
name: low2high
description: Make a 3D model's geometry fit for synthetic RGB–IR imagery, in whichever direction it needs — a low-poly game or phone-game model up to smooth, faithful geometry (normals, subdivision, displacement, and redesigning parts in Blender through the MCP when nothing less will do), or a heavy mechanical/CAD model (STEP, IGES, or dense OBJ/FBX) down to a tessellation the renderer can carry. Use this skill whenever a model looks faceted or blocky, is "low poly" or "game-ready", comes from CAD or mechanical engineering, arrives as .step/.stp/.iges/.igs, is too heavy to render, or when the facet report fails. It runs after the ingest-asset skill has the model's parts, and it measures before and after every change rather than judging by eye.
---

# low2high: geometry fit for the camera, in either direction

Models arrive at two extremes, and both image wrongly:

* **game / phone-game models** approximate curves with a handful of flat facets and fake the rest
  with a normal map. A facet is a step in the surface normal: a shading stripe in the RGB
  companion, and in the IR a step in directional emissivity and reflected sky — a smooth motor
  hub renders as bands of different apparent temperature, a structure a detector trained on our
  frames will learn and real footage never shows;
* **mechanical / CAD models** carry exact curves (STEP/IGES) or a tessellation of them dense
  enough to cost millions of triangles the camera cannot resolve.

The skill moves each part of a model up or down a ladder until it passes **one measurement**, the
same in both directions (ADR 0155):

```bash
python scripts/mesh_facets.py <model.glb | x.meshes.npz> --range-m <closest range> --ifov-mrad <pixel>
```

An edge is faceted when its own silhouette error — how far its facets stand off the curve they
approximate, `(w/2)·tan(θ/4)` — exceeds **half a pixel at the closest range the scene flies the
object**. No radius to guess, and sub-pixel detail may stay as coarse as it likes: measured on the
Phantom 4, the 2.49 M-triangle original has **0 of 41** meshes faceted at 2 m on a 1.36 mrad
sensor, and the same model decimated to phone-game density (~100 k faces) has **40 of 41**.
Without `--range-m/--ifov-mrad` the report falls back to an angle band that also flags harmless
small details: a list of candidates, not a verdict. Always give it the scene.

Take the range and IFOV from the scene config and sensor YAML the asset is for; with no scene
yet, use the closest range in the lane (aerial demos fly 2–6 m) and the finest sensor in
`configs/sensors/`.

**Two meshes, two budgets.** Everything here improves the **render** mesh. The thermal solver's
copy is still planar-dissolved to its own budget by `prep_asset.py` (ADR 0132, AI.3) — smooth
pictures must not make the heat solve slow. So measure the render geometry
(`prep_asset.py --emit-mesh --dissolve-deg 0`, or the `.glb` itself), never the default thermal
archive, whose deliberately merged faces read as faceted.

## 0. Where this sits

After `ingest-asset` has fetched or registered the model and decomposed it into named parts
(its steps 1–6), before the export (its step 7). Work on the part-split scene — **one object per
part, object name = part name, material slots untouched** — so the material map and every scene
binding still find their keys after the geometry changes. Record everything in a `geometry:`
comment block at the top of `configs/assets/<name>.yaml`, the way `phantom4.yaml` records its
scale reasoning: per part, the rung reached, the facet numbers before and after, and for any
redesign the references it was built from.

## 1. Measure, and look

1. Facet report with the scene's range and IFOV (above). Keep the output; it is the "before".
2. A before picture of the worst parts: Blender through the MCP (`render_viewport_to_path`,
   `get_screenshot_of_area_as_image`), or `blender --background` with Cycles on the CPU.
3. Decide the direction per part: **faceted → go up (§3)**, **over the render budget → go
   down (§2)**, both fine → leave it alone. A model can need both (a fine CAD body with a coarse
   imported fastener).

## 2. Down: CAD and over-dense meshes

**STEP / IGES** — Blender cannot open them. Tessellate with OpenCascade (the optional `cad`
extra: `pip install cadquery-ocp` into any Python ≥ 3.10):

```bash
python scripts/tessellate_step.py part.step --out part.glb --angle-deg <a> --chord-mm <c>
```

Choose the tolerances from the scene, not by habit: `--chord-mm` = half the pixel footprint at
the closest range (1.36 mrad × 2 m / 2 = 1.36 mm), and `--angle-deg` from
`irsim.io.mesh_facets.max_facet_angle_deg(r, range, ifov)` for the smallest radius that matters.
The CAD assembly tree's **part names come through as glTF node names** and the file's unit is
converted to metres, so the part decomposition and the scale are usually free — verify both
anyway (one published dimension, as ever). Then register the `.glb` with
`scripts/register_local_asset.py` and continue in `ingest-asset`.

**Dense OBJ / FBX from CAD** — check the unit first (CAD exports are usually millimetres:
`scale_to_metres: 0.001`). Reduce only what is over budget, and only in ways the facet report
confirms: planar dissolve first (removes coplanar faces, costs no silhouette), then a collapse
decimate per part at the ratio that keeps `mesh_facets` passing at the scene's range —
re-measure after every pass. Hidden internals (fasteners inside a housing) can leave the render
mesh, but keep any that carry heat as parts for the thermal side: a battery nobody sees still
warms the shell.

CAD rarely carries appearance — often one grey for everything — so materials still come from the
real-object research in `ingest-asset`, not from the file.

## 3. Up: the low-to-high ladder

Climb only as far as the report requires, part by part, re-measuring after each rung. Every rung
keeps the part's name, its material slots and its published dimensions.

**Rung 1 — normals.** Mark the real creases (designed edges: a flange, a chamfer) as sharp; smooth
everything else (smooth by angle, weighted normals). This removes the shading bands inside a
surface in both bands — the IR banding in particular — but not the polygon at the silhouette.
Cheap and always first.

**Rung 2 — subdivision.** Subdivision Surface with the rung-1 creases carried as crease weights
(or support loops) so designed edges stay sharp, level 1–2, then apply. Subdivision **shrinks**
convex shapes toward their control cage: re-measure the part's extent against the published
dimension and against the before, and fix a shrink with creases, not with a rescale that would
move the other parts. Keep the whole asset within the render budget.

**Rung 3 — displacement from the model's own maps.** Many game models carry their real detail in
a normal or height map. Turn the height map into geometry (Displace modifier on a subdivided
part, strength matched to the map's authored scale), or keep it as a normal map where the
geometry would exceed the budget. Detail from the asset's own maps is the artist's intent, not an
invention.

**Rung 4 — redesign, through the Blender MCP.** When the part is too crude for any of the above
(a propeller that is two flat quads, a hub with six sides and no bolt circle), rebuild it:

1. Gather references for the **real** object: manufacturer drawings and spec sheets, product
   photos from several views, teardown images. The real name from `ingest-asset` is the search
   key. Save the URLs.
2. Model the part in Blender through the MCP (`execute_blender_code` for bpy: curves, bevels,
   booleans, arrays, screw modifiers — prefer operators, inspect before changing, per the MCP's
   own rules), matching the published dimensions and the reference silhouettes. Build it in the
   part's local frame so it drops into the same position.
3. Replace the low-poly object with the rebuilt one **under the same object name, with the same
   material slots**, and delete the old one.
4. Compare: a render from the reference photo's viewpoint beside the photo
   (`render_viewport_to_path`, `get_screenshot_of_area_as_image`), and the facet report.
5. Record it: a redesigned part is **estimated geometry**. In the `geometry:` block write
   `ESTIMATED — redesigned`, what was rebuilt, and the reference URLs. Nothing downstream may
   treat redesigned geometry as the author's measurement.

Redesign only what the camera will resolve. A 3 mm bolt head is sub-pixel at every range the
aerial lane flies; do not spend a redesign on it.

## 4. Check the pair, then hand back

* `mesh_facets.py` passes at the scene's range and IFOV — paste the before/after lines into the
  `geometry:` block.
* The part-split asset still covers every part (`prep_asset.py --emit-parts` coverage ≥ 0.95),
  the material audit still passes, and the extents still match the published dimensions.
* **Look at the RGB–IR pair.** The RGB companion is the reference for the IR frame: any facet
  banding, shrink or seam visible in one band and not the other is a defect. Render the pair
  with the project's Isaac Sim driver when the A6000 is free (`nvidia-smi` first, `IRSIM_GPU=0`);
  a Blender CPU turntable is enough for the RGB side while iterating.
* Return to `ingest-asset` step 7 to export `.blend` + `.usdc` + `.fbx` from the refined scene,
  and ship with `ship-step`.

## When to stop and ask

* A redesign would need information no reference provides (an interior the photos never show) —
  say which part and what is missing rather than inventing it.
* Subdivision shrink or a redesign moves a published dimension by more than a few percent and
  creases do not recover it.
* The render budget cannot be met without the facet report failing at the scene's range — the
  scene may need a closer-range variant of the asset, which is a decision, not a tweak.
