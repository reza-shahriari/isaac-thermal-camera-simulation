#!/usr/bin/env python3
"""Each triangle's surface colour on a fused human scan, from its texture (HU.9).

    blender -b --python scripts/scan_colours.py -- --source scan.glb --out scan.colours.npz

A scan's skin, hair and clothes are one mesh; the first thing that tells them apart is colour
(``irsim.io.human_scan.classify_surfaces``). This samples, for every triangle, its material's
base-colour texture at the triangle's three corners and centre (UV), averages them, and writes
``rgb`` (N, 3) as **display sRGB in 0..1** -- the space skin-colour models are fitted in -- with
``centres_gltf`` in the glTF frame, so the rows line up with ``autorig_mia.py``'s mesh by
position. A material with no texture contributes its flat base colour, converted from Blender's
linear value to sRGB.

Blender only (it reads the textures the way the file binds them); the classification itself is
NumPy and tested without it.

docs/physics-model.md §6.1; roadmap HU.9; ADR 0201.
"""

from __future__ import annotations

import argparse
import pathlib
import sys


def _linear_to_srgb(c: float) -> float:
    return 12.92 * c if c <= 0.0031308 else 1.055 * c ** (1.0 / 2.4) - 0.055


def main(argv: list[str]) -> None:
    import bpy
    import numpy as np

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--source", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    a = ap.parse_args(argv)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(a.source))
    shapes = bpy.data.collections.get("glTF_not_exported")
    if shapes is not None:
        for obj in list(shapes.objects):
            bpy.data.objects.remove(obj, do_unlink=True)

    def base_colour(mat):  # type: ignore[no-untyped-def]
        """(pixels HxWx4 sRGB, None) for a textured base colour, else (None, flat sRGB)."""
        if mat is None or not mat.use_nodes:
            return None, (0.8, 0.8, 0.8)
        bsdf = next((n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED"), None)
        if bsdf is None:
            return None, (0.8, 0.8, 0.8)
        sock = bsdf.inputs["Base Color"]
        stack = [lk.from_node for lk in sock.links]
        while stack:  # walk back through mix / multiply nodes to the first image
            node = stack.pop()
            if node.type == "TEX_IMAGE" and node.image is not None:
                img = node.image
                w, h = img.size
                px = np.empty(w * h * 4, dtype=np.float32)
                img.pixels.foreach_get(px)
                px = px.reshape(h, w, 4)
                if img.colorspace_settings.name != "sRGB":  # a linear texture: encode it
                    px[..., :3] = np.where(
                        px[..., :3] <= 0.0031308,
                        12.92 * px[..., :3],
                        1.055 * np.clip(px[..., :3], 0, None) ** (1 / 2.4) - 0.055,
                    )
                return px, None
            stack.extend(lk.from_node for inp in node.inputs for lk in inp.links)
        v = sock.default_value
        return None, tuple(_linear_to_srgb(float(v[i])) for i in range(3))

    rgb, centres, mats = [], [], []
    for obj in [o for o in bpy.data.objects if o.type == "MESH"]:
        me = obj.data
        me.calc_loop_triangles()
        uv = me.uv_layers.active.data if me.uv_layers.active else None
        cache = {i: base_colour(s.material) for i, s in enumerate(obj.material_slots)}
        mw = obj.matrix_world
        for tri in me.loop_triangles:
            px, flat = cache.get(tri.material_index, (None, (0.8, 0.8, 0.8)))
            if px is None or uv is None:
                rgb.append(flat)
            else:
                h, w = px.shape[:2]
                uvs = [tuple(uv[li].uv) for li in tri.loops]
                uvs.append(tuple(sum(c[k] for c in uvs) / 3.0 for k in range(2)))
                samples = [
                    px[int((v_ % 1.0) * (h - 1)), int((u_ % 1.0) * (w - 1)), :3] for u_, v_ in uvs
                ]
                rgb.append(tuple(float(x) for x in np.mean(samples, axis=0)))
            c = mw @ tri.center
            centres.append((c.x, c.z, -c.y))  # Blender -> glTF frame
            slot = obj.material_slots[tri.material_index] if obj.material_slots else None
            mats.append(slot.material.name if slot and slot.material else "")
    np.savez_compressed(
        a.out,
        rgb=np.clip(np.asarray(rgb, dtype=np.float32), 0.0, 1.0),
        centres_gltf=np.asarray(centres, dtype=np.float32),
        materials=np.array(mats),
    )
    print(f"[scan_colours] {len(rgb)} triangles -> {a.out}")


if __name__ == "__main__":
    main(sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else [])
