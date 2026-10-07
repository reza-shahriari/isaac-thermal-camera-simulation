#!/usr/bin/env python3
"""Make a scan-like test input with a known answer: one labelled person fused into one mesh (HU.9).

    blender -b --python scripts/fuse_human.py -- \\
        --source 3d_models/man_adult_dressed/source/man_adult_dressed_labelled.glb \\
        --out outputs/humans/scan_tests/man_adult_dressed_scan.glb

A downloaded scan is one mesh: skin, clothes and hair fused, no skeleton, no part names. The
any-human path (``autorig_mia.py`` -> ``rig_from_mia.py`` -> ``prep_human.py``) has to recover the
seventeen segments and the skin / hair / garment split from that alone. To *measure* how well it
does, the input must come with the answer, and a real scan does not. So this takes a person the
project already labelled by its own skeleton (``prep_human.py``: ``skin_<Segment>``,
``garment_<slot>``, ``hair``, ``eyes``, ...), joins every mesh into one object with no rig, and
writes

* ``<out>`` -- the fused mesh as glTF, materials and textures kept (a scan has a texture too);
* ``<out>.truth.npz`` -- for every triangle of it: the label it came from (``labels``), its
  centre in
  the **glTF frame** (Y up, which is what Make-It-Animatable's loader sees), its area, and the
  triangles themselves (``vertices_gltf``, ``faces``) for neighbourhoods.

The skeleton-labelled answer is itself a model (ADR 0192), not a measurement of a person: the test
says how well an auto-rigger reproduces a skeleton's labelling, which is what the pipeline needs.

docs/physics-model.md §6.1; roadmap HU.9; ADR 0201.
"""

from __future__ import annotations

import argparse
import pathlib
import sys


def main(argv: list[str]) -> None:
    import bpy
    import numpy as np

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--source", type=pathlib.Path, required=True, help="a prep_human.py output")
    ap.add_argument("--out", type=pathlib.Path, required=True, help="the fused .glb to write")
    a = ap.parse_args(argv)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(a.source))
    # the importer's bone display shape (an icosphere in `glTF_not_exported`) is not geometry
    shapes = bpy.data.collections.get("glTF_not_exported")
    if shapes is not None:
        for obj in list(shapes.objects):
            bpy.data.objects.remove(obj, do_unlink=True)
        bpy.data.collections.remove(shapes)
    for obj in [o for o in bpy.data.objects if o.type != "MESH"]:
        bpy.data.objects.remove(obj, do_unlink=True)
    meshes = sorted((o for o in bpy.data.objects if o.type == "MESH"), key=lambda o: o.name)
    if not meshes:
        raise SystemExit(f"no meshes in {a.source}")
    # every face remembers the object it came from, through the join
    names = [o.name.split(".")[0] for o in meshes]
    for i, obj in enumerate(meshes):
        attr = obj.data.attributes.new("irsim_truth", "INT", "FACE")
        attr.data.foreach_set("value", [i] * len(obj.data.polygons))
        for mod in list(obj.modifiers):
            obj.modifiers.remove(mod)
    for o in bpy.context.selected_objects:
        o.select_set(False)
    for obj in meshes:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    bpy.ops.object.join()
    scan = bpy.context.view_layer.objects.active
    scan.name = scan.data.name = "scan"
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    # triangles, as the exported file and every loader will have them: one truth row per triangle
    import bmesh

    bm = bmesh.new()
    bm.from_mesh(scan.data)
    bmesh.ops.triangulate(bm, faces=bm.faces[:])
    bm.to_mesh(scan.data)
    bm.free()

    me = scan.data
    truth = np.zeros(len(me.polygons), dtype=np.int64)
    me.attributes["irsim_truth"].data.foreach_get("value", truth)
    centres = np.array([tuple(p.center) for p in me.polygons], dtype=np.float64)
    areas = np.array([p.area for p in me.polygons], dtype=np.float64)
    me.attributes.remove(me.attributes["irsim_truth"])
    # Blender (x, y, z) -> glTF (x, z, -y): the frame the exported file and its loaders use
    gltf = np.c_[centres[:, 0], centres[:, 2], -centres[:, 1]]
    verts = np.array([tuple(v.co) for v in me.vertices], dtype=np.float64)
    verts_gltf = np.c_[verts[:, 0], verts[:, 2], -verts[:, 1]]
    tris = np.array([tuple(p.vertices) for p in me.polygons], dtype=np.int32)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.export_scene.gltf(
        filepath=str(a.out),
        export_format="GLB",
        use_selection=False,
        export_skins=False,
        export_yup=True,
        export_materials="EXPORT",
        export_image_format="AUTO",
    )
    np.savez_compressed(
        a.out.with_suffix(".truth.npz"),
        labels=np.array([names[i] for i in truth]),
        centres_gltf=gltf.astype(np.float32),
        areas=areas.astype(np.float32),
        vertices_gltf=verts_gltf.astype(np.float32),
        faces=tris,
        source=np.array(str(a.source)),
    )
    print(f"[fuse_human] {len(meshes)} objects -> 1 mesh, {len(truth)} faces -> {a.out}")


if __name__ == "__main__":
    main(sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else [])
