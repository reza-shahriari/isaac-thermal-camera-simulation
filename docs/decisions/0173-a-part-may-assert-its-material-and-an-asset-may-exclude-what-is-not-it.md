# ADR 0173 — A part may assert its material, and an asset may exclude geometry that is not it

**Status:** Accepted
**Date:** 2026-09-30
Extends [ADR 0138](0138-an-asset-is-decomposed-into-parts-by-connected-component.md) (functional
parts by connected component) and [ADR 0128](0128-per-asset-material-mapping-and-the-blender-prep-path.md)
(the per-asset material map). Uses [ADR 0047](0047-material-mapping-defaults-and-coverage.md)'s
`thermal:material` override rung. Roadmap: `AI.14`; first used by `AI.15`.

## Context

ADR 0128 maps an asset's **source** material names onto the library, and ADR 0138 regroups its
geometry into **functional** parts. The two were independent on purpose: "the white plastic" is
four propellers and two shells, and "the battery" shares its moulding compound with half the
aircraft. Three DJI models handed over on 2026-09-30 found two things that pair cannot say.

1. **One material, many substances.** 324 of the DJI Inspire 3's 345 objects share a single
   texture-atlas material called `default`: the PC housing, the carbon-fibre arms, the
   glass-filled nylon propellers and the aluminium motors alike. The only per-name map possible
   is "all of it is one thing". The parts know better, since the arms are carbon fibre whatever
   the atlas is called, but a part had no way to say so.
2. **Not everything in the file is the asset.** The Matrice 300 RTK stands on its landing gear on
   its hard transport case, and the case is 74 % of the file's area. Leaving it unclaimed drops it
   from the part split but fails the 95 % coverage gate. Claiming it as a part keeps a suitcase
   hanging under the aircraft in every render. Lowering the gate to 0.26 would pass, but it would
   also pass a decomposition that had lost the fuselage.

## Decision

Two optional fields on the parts block, both engine-free data in the asset YAML:

* **`PartSpec.material`**: a library material asserted for the whole part. In the part-split USD,
  `prep_asset.py` writes it onto every mesh under the part as `thermal:material`, the override
  rung ADR 0047 already put at the top of the resolver. It is also written as a custom property
  on the part object in the master `.blend`. In a scene, it sits between the scene's own
  per-part override and the asset map (`scene._expand_objects`). `AssetMapping.targets` includes
  it, so the loader's known-material check covers it.
* **`PartsConfig.exclude`**: a list of selectors tried **before** any part. A component that
  matches is left out of the coverage totals, reported on its own line, and dropped from the
  split exactly like unclaimed geometry. The un-split USD still carries it, and still maps it.

`PartsConfig.claim` is the single claiming rule. The report, the engine-free archive split and the
driver's instructions to the Blender split all call it, so the three cannot disagree about where
a shell belongs.

## Consequences

* A part material covers **every face of the part**. A part that spans substances must be split
  first: the Inspire 3's lens glass is its own `camera_lens` part so the gimbal's painted metal
  does not paint it. Where the source materials are already distinct, as on the Matrice 300,
  nothing should be asserted: the map is per face and finer than any part.
* The override is written after Blender's USD export, with `pxr`. Blender's own custom-property
  export namespaces a property as `userProperties:…`, which the resolver does not read.
* `prep_asset.py` now audits the part-split USD's own prim records too, so an asserted material
  is **seen** to win (`by rule: override=N`) rather than assumed to.
* The un-split USD still maps the atlas to one material. Anything that renders it gets that one
  material; scenes that want per-part truth reference the part-split asset, as they already did
  for per-part temperature.

## Found on the way, fixed in the same step

These were latent in the pipeline since `AI.5`, and each was hidden by the Phantom 4 being an FBX
with a non-unit scale:

* **Component areas were in the mesh's local frame** while their bounds were in world space. A
  glTF at `scale_to_metres: 1.0` keeps its node transforms, and the DJI Mini 3 Pro summed to
  40,701 m² against a true 0.113. Coverage is area-weighted, so this decided which parts passed.
  Areas are now summed from world-space triangles.
* **The Blender split took the whole mesh on its first `separate`.** The glTF importer leaves every
  vertex selected, and edit mode rebuilds face selection from vertices, so fourteen of fifteen
  parts came out empty. Vertex and edge selection are now cleared and the split runs in
  face-select mode.
* **Parts landed under the source's node chain**, where neither the stamp nor a scene's
  `prim_root/<part>/<part>` binding finds them. Each part is now unparented, keeping its world
  transform.
* **Blender exits 0 when the worker raises**, so the driver reported success over a failed split.
  Every Blender command now passes `--python-exit-code 1`.
* **A glTF-sourced FBX lost its textures**, because the images were packed and the exporter
  copies from disk. They are now written out beside the FBX first.
