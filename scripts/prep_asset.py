#!/usr/bin/env python3
"""Prepare a third-party 3D asset for the simulator: rescale, convert to USD, audit its materials.

    python scripts/prep_asset.py --asset phantom4
    python scripts/prep_asset.py --asset phantom4 --source 3d_models/phantom4.fbx \
        --out-dir data/assets

Everything here runs on the **CPU**. Blender is the USD toolchain -- it ships a complete ``pxr``
(OpenUSD 26.03 on the 5.2 LTS build) -- so no part of this boots Kit, starts a renderer or touches
CUDA. That is the point: the inspect -> map -> audit loop is the loop an operator runs dozens of
times per asset, and paying a 15-35 s Kit boot for it (ADR 0014) made it a loop nobody would run.

**The file runs twice.** Invoked normally it is the *driver*: it reads the asset config (needing
pydantic and PyYAML, which Blender's interpreter does not have), then re-invokes itself inside
Blender for the geometry work, then audits the result. Invoked with ``bpy`` importable it is the
*worker*. The split exists because neither interpreter has the other's dependencies, not because
the work is naturally two pieces.

What the worker does, in order:

1. **Import** the source file. FBX goes through ``wm.fbx_import`` -- the C++/ufbx importer that is
   the default from Blender 5.0 -- which preserves material names and per-face material indices.
2. **Rescale** by the asset's ``scale_to_metres``. An FBX carries no unit, so this is the one fact
   about the asset that cannot be recovered from the file and must be authored. Unapplied, a
   Phantom 4 enters the stage 41 m wide and still renders a plausible image.
3. **Export** USD with ``UsdPreviewSurface`` (Isaac Sim supports it; Blender has had no MDL
   exporter since the Omniverse connector was discontinued).
4. **Walk** the stage into engine-free prim records -- the same ``{path, material_name,
   semantic_class, override}`` shape ``irsim_isaac.pipeline.materials_usd`` produces inside Kit,
   so the JSON is interchangeable and `irsim` still never imports USD.

The worker also refuses a mesh that carries ``materialBind`` subsets *and* a direct mesh-level
binding without reporting it: Blender binds the first material slot to the mesh as well as making
subsets (a documented Hydra workaround), so reading the mesh binding alone would map a whole
multi-material building to slot 0 -- silently, at 100 % coverage. See ADR 0128.

docs/physics-model.md §13.3; ADR 0047 (precedence, the UNMAPPED sentinel, the coverage gate),
ADR 0128 (per-asset mapping, the Blender prep path).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
from collections.abc import Sequence
from typing import Any

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

#: A prim carrying less than this fraction of the asset's area is reported but not gated on its
#: own relative area error -- see `emit_meshes`.
PRIM_AREA_FLOOR_FRACTION = 0.001

#: Integer face attribute carrying a part index. It survives both `join` and `separate`, which
#: is what lets `split_by_part_usd` regroup geometry without depending on face order (ADR 0138).
ATTR_PART = "irsim_part"

#: Source formats the worker knows how to import.
IMPORTERS: dict[str, str] = {
    ".fbx": "fbx",
    ".obj": "obj",
    ".gltf": "gltf",
    ".glb": "gltf",
    ".usd": "usd",
    ".usda": "usd",
    ".usdc": "usd",
}


# --------------------------------------------------------------------------------------------
# worker -- runs inside Blender
# --------------------------------------------------------------------------------------------


def _import_source(path: pathlib.Path) -> str:
    import bpy

    kind = IMPORTERS[path.suffix.lower()]
    if kind == "fbx":
        # The C++/ufbx importer (default from Blender 5.0). `use_existing` keeps two assets that
        # share a material name from becoming `Foo` and `Foo.001`, which would defeat the map.
        try:
            bpy.ops.wm.fbx_import(filepath=str(path))
            return "wm.fbx_import (C++/ufbx)"
        except AttributeError:
            bpy.ops.import_scene.fbx(filepath=str(path))
            return "import_scene.fbx (legacy Python)"
    if kind == "obj":
        bpy.ops.wm.obj_import(filepath=str(path))
        return "wm.obj_import"
    if kind == "gltf":
        bpy.ops.import_scene.gltf(filepath=str(path))
        return "import_scene.gltf"
    bpy.ops.wm.usd_import(filepath=str(path))
    return "wm.usd_import"


def _rescale(factor: float) -> None:
    """Scale every root object about the world origin, then apply it into the mesh data.

    Applied rather than left on the transform so the exported USD carries metres in its point
    data. A scale left on an Xform survives export but is then one more thing a downstream reader
    has to honour, and the mesh bridge (ADR 0110) queries points, not transforms.
    """
    import bpy

    if factor == 1.0:
        return
    for obj in bpy.data.objects:
        if obj.parent is None:
            obj.scale = tuple(s * factor for s in obj.scale)
            obj.location = tuple(c * factor for c in obj.location)
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.transform_apply(location=True, rotation=False, scale=True)
    bpy.ops.object.select_all(action="DESELECT")


def _prim_records(usd_path: pathlib.Path) -> list[dict[str, object]]:
    """Walk the exported stage into engine-free records, subsets included.

    Mirrors ``irsim_isaac.pipeline.materials_usd.prim_records``: visible ``UsdGeom.Gprim`` only,
    the bound material's *name* rather than its path (names survive re-parenting; paths do not),
    and the ``thermal:material`` override read straight off the prim.
    """
    from pxr import Usd, UsdGeom, UsdShade

    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise RuntimeError(f"could not open the stage this script just wrote: {usd_path}")
    records: list[dict[str, object]] = []
    shadowed: list[str] = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Gprim):
            continue
        imageable = UsdGeom.Imageable(prim)
        if imageable and imageable.ComputeVisibility() == UsdGeom.Tokens.invisible:
            continue
        binding = UsdShade.MaterialBindingAPI(prim)
        subsets = binding.GetMaterialBindSubsets()
        if subsets:
            # The multi-material case. Each subset is its own record; the mesh-level binding is
            # deliberately ignored, and reported, because Blender writes slot 0 there as well.
            if binding.ComputeBoundMaterial()[0]:
                shadowed.append(str(prim.GetPath()))
            for subset in subsets:
                material = UsdShade.MaterialBindingAPI(subset.GetPrim()).ComputeBoundMaterial()[0]
                records.append(_record(subset.GetPrim(), material))
            continue
        records.append(_record(prim, binding.ComputeBoundMaterial()[0]))
    if shadowed:
        print(
            f"note: {len(shadowed)} mesh(es) carry materialBind subsets AND a direct binding; "
            "the subsets win (ADR 0128). First: " + shadowed[0]
        )
    return records


def _record(prim: object, material: object) -> dict[str, object]:
    name = None
    if material is not None and material.GetPrim().IsValid():  # type: ignore[attr-defined]
        name = str(material.GetPrim().GetName())  # type: ignore[attr-defined]
    override = None
    attr = prim.GetAttribute("thermal:material")  # type: ignore[attr-defined]
    if attr and attr.IsValid():
        override = attr.Get()
    return {
        "path": str(prim.GetPath()),  # type: ignore[attr-defined]
        "material_name": name,
        "semantic_class": None,
        "override": None if override is None else str(override),
    }


def _triangles_of(obj: object) -> tuple[Any, Any]:
    """World-space vertices and triangle indices for one object, as plain arrays.

    The mesh is read *after* the scale has been applied, so the arrays are metres in the stage
    frame -- the frame `TriangleMeshPatch` queries and `mesh_bridge` hands it world positions in.
    """
    import numpy as np

    me = obj.data  # type: ignore[attr-defined]
    me.calc_loop_triangles()
    m = obj.matrix_world  # type: ignore[attr-defined]
    verts = np.empty((len(me.vertices), 3), dtype=np.float64)
    for i, v in enumerate(me.vertices):
        verts[i] = (m @ v.co)[:]
    faces = np.asarray([tuple(t.vertices) for t in me.loop_triangles], dtype=np.int32)
    return verts, faces


def _drop_degenerate(vertices: Any, faces: Any) -> tuple[Any, int]:
    """Remove zero-area triangles.

    Real assets carry them -- this Phantom 4 has 16 -- and they are not a rounding artefact to be
    tolerated: a facet with no area has no normal, so `TriangleMeshPatch` refuses the whole mesh
    rather than solve a cell whose orientation is undefined. Dropping them changes no area and no
    vertex, so it is a repair rather than a decimation, but it is reported all the same.
    """
    import numpy as np

    if len(faces) == 0:
        return faces, 0
    a, b, c = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    keep = np.linalg.norm(np.cross(b - a, c - a), axis=-1) > 0.0
    return faces[keep], int((~keep).sum())


def _area_of(vertices: Any, faces: Any) -> float:
    import numpy as np

    if len(faces) == 0:
        return 0.0
    a, b, c = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    return float(0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=-1).sum())


def _dissolve(obj: object, angle_deg: float) -> None:
    """Planar decimation: merge coplanar faces **without moving a vertex**.

    Deliberately not the collapse decimator. Surface area is a physical quantity here -- it sets
    both the radiated power and the convective load -- and collapse destroys it: measured on this
    asset, ratio 0.05 removed **37 %** of the area, because 31,068 of its shells are small enough
    to be collapsed away entirely. Planar dissolve at 5 deg removes half the triangles for
    **+0.2 %** area, because it only ever merges faces that were already in the same plane.
    """
    import bpy

    if angle_deg <= 0.0:
        return
    import math

    mod = obj.modifiers.new("irsim_dissolve", "DECIMATE")  # type: ignore[attr-defined]
    mod.decimate_type = "DISSOLVE"
    mod.angle_limit = math.radians(angle_deg)
    with bpy.context.temp_override(object=obj):
        bpy.ops.object.modifier_apply(modifier=mod.name)


def emit_meshes(
    out_path: pathlib.Path, dissolve_deg: float, max_area_error: float
) -> dict[str, object]:
    """Write one engine-free mesh archive for the whole asset.

    The archive is an ``.npz`` holding ``v<i>`` / ``f<i>`` per prim plus a JSON ``manifest``, so
    `irsim` can build a `TriangleMeshPatch` on a real asset without importing `pxr` or booting
    anything. Area before and after is recorded per prim and the whole emit is **refused** if any
    prim moves by more than ``max_area_error`` -- a decimation that quietly shrinks a surface is a
    decimation that quietly cools it.
    """
    import bpy
    import numpy as np

    meshes = [o for o in bpy.data.objects if o.type == "MESH"]
    manifest: list[dict[str, object]] = []
    arrays: dict[str, Any] = {}
    degenerate = 0
    for i, obj in enumerate(sorted(meshes, key=lambda o: o.name)):
        before_v, before_f = _triangles_of(obj)
        before = _area_of(before_v, before_f)
        _dissolve(obj, dissolve_deg)
        verts, faces = _triangles_of(obj)
        faces, dropped = _drop_degenerate(verts, faces)
        degenerate += dropped
        after = _area_of(verts, faces)
        material = (
            obj.data.materials[0].name if obj.data.materials and obj.data.materials[0] else None
        )
        manifest.append(
            {
                "index": i,
                "name": obj.name,
                "material_name": material,
                "n_faces": int(faces.shape[0]),
                "area_m2": after,
                "area_before_m2": before,
            }
        )
        arrays[f"v{i}"] = verts
        arrays[f"f{i}"] = faces

    # The gate is **area-weighted**, and deliberately not a bare relative error per prim.
    # A relative gate is the right guard for a prim that carries the signature and the wrong one
    # for a 1.1e-5 m^2 decorative sliver, where 3 % is 4e-7 m^2 and cannot move any temperature.
    # So: the whole asset is gated on relative area, and a prim is gated only once it carries at
    # least `PRIM_AREA_FLOOR_FRACTION` of the total. Everything else is reported, never silent.
    total_before = sum(float(m["area_before_m2"]) for m in manifest)
    total_after = sum(float(m["area_m2"]) for m in manifest)
    floor = PRIM_AREA_FLOOR_FRACTION * total_before
    offenders: list[str] = []
    ignored = 0
    for m in manifest:
        before, after = float(m["area_before_m2"]), float(m["area_m2"])
        if before == 0.0:
            continue
        error = abs(after / before - 1.0)
        if error <= max_area_error:
            continue
        if before < floor:
            ignored += 1
            continue
        offenders.append(f"{m['name']}: area {before:.6f} -> {after:.6f} m2 ({error:.2%})")
    total_error = 0.0 if total_before == 0.0 else abs(total_after / total_before - 1.0)
    if total_error > max_area_error:
        offenders.append(
            f"WHOLE ASSET: {total_before:.4f} -> {total_after:.4f} m2 ({total_error:.2%})"
        )
    if offenders:
        raise SystemExit(
            f"refusing to write {out_path.name}: {len(offenders)} prim(s) above the "
            f"{PRIM_AREA_FLOOR_FRACTION:.2%}-of-total floor changed area by more than "
            f"{max_area_error:.1%}. Lower --dissolve-deg.\n  " + "\n  ".join(offenders[:10])
        )
    if degenerate:
        print(f"note: dropped {degenerate} zero-area triangle(s); area unchanged")
    if ignored:
        print(
            f"note: {ignored} prim(s) below the {PRIM_AREA_FLOOR_FRACTION:.2%}-of-total area "
            "floor moved by more than the tolerance; too small to affect a temperature, kept"
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, manifest=json.dumps(manifest), **arrays)
    total_faces = sum(int(m["n_faces"]) for m in manifest)
    print(
        f"wrote {out_path} ({len(manifest)} prims, {total_faces:,} triangles, "
        f"{total_after:.4f} m2, {100 * total_error_signed(total_before, total_after):+.3f}% area)"
    )
    return {"prims": len(manifest), "faces": total_faces, "area_m2": total_after}


def _connected_roots(n_vertices: int, edges: Any) -> Any:
    """Union-find over an edge list: the shell id of every vertex.

    Lifted out of :func:`emit_components` rather than nested in its loop, because a closure over a
    loop variable is the classic way to have every iteration share one array.
    """
    import numpy as np

    parent = np.arange(n_vertices)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = int(parent[x])
        return int(x)

    for a, b in edges:
        ra, rb = find(int(a)), find(int(b))
        if ra != rb:
            parent[ra] = rb
    return np.array([find(i) for i in range(n_vertices)])


def emit_components(out_path: pathlib.Path) -> dict[str, object]:
    """Write per-connected-component statistics for the whole asset.

    A **component** is one connected shell of the mesh -- the unit a part decomposition is built
    from (:mod:`irsim.io.asset_parts`). The source asset groups by material, so a prim is not a
    part: the Phantom 4's 41 prims split into 31,068 components, and it is the components that
    line up with real hardware. A propeller is a separate shell from the motor it bolts to even
    when both are "white plastic", and that is the only reason parts are recoverable at all.

    Measured **before** the planar dissolve, so the statistics describe the asset as imported and
    do not shift when ``--dissolve-deg`` changes. Bounds come from the vertices under the object's
    world matrix, never from ``bound_box`` -- that is a *local* box, and re-taking min/max over its
    eight transformed corners inflates the Phantom 4 from 41 x 46 cm to 60 x 62.

    ``scale_to_metres`` has already been applied by :func:`_rescale`, so everything here is metres
    in the asset's own frame -- the frame the part selectors are authored in.
    """
    import bpy
    import numpy as np

    components: list[dict[str, object]] = []
    face_ids: dict[str, Any] = {}
    for obj in sorted((o for o in bpy.data.objects if o.type == "MESH"), key=lambda o: o.name):
        me = obj.data
        if not len(me.polygons):
            continue
        nv = len(me.vertices)
        co = np.empty(nv * 3, dtype=np.float64)
        me.vertices.foreach_get("co", co)
        co = co.reshape(nv, 3)
        world = np.asarray(obj.matrix_world)
        co = (world @ np.c_[co, np.ones(nv)].T).T[:, :3]

        ne = len(me.edges)
        ev = np.empty(ne * 2, dtype=np.int64)
        me.edges.foreach_get("vertices", ev)
        ev = ev.reshape(ne, 2)
        roots = _connected_roots(nv, ev)

        # Every vertex of a polygon is in the same shell, so one vertex per polygon identifies
        # it. Read through the loops rather than `polygons.foreach_get("vertices", ...)`, which
        # needs a flat buffer whose length depends on the polygon sizes -- n-gons and triangles
        # cannot share one call, and an asset may carry both.
        npoly = len(me.polygons)
        starts = np.empty(npoly, dtype=np.int64)
        me.polygons.foreach_get("loop_start", starts)
        loops = np.empty(len(me.loops), dtype=np.int64)
        me.loops.foreach_get("vertex_index", loops)
        first = loops[starts]

        areas = np.empty(npoly, dtype=np.float64)
        me.polygons.foreach_get("area", areas)
        mats = np.empty(npoly, dtype=np.int64)
        me.polygons.foreach_get("material_index", mats)
        slots = [ms.material.name if ms.material else None for ms in obj.material_slots]

        face_root = roots[first]
        vert_root = roots
        # global component index of every face of this object, filled in below
        this_object = np.full(npoly, -1, dtype=np.int64)
        face_ids[obj.name] = this_object
        for root in np.unique(face_root):
            fm = face_root == root
            pts = co[vert_root == root]
            if not len(pts):
                continue
            by_material: dict[str, float] = {}
            for idx, a in zip(mats[fm], areas[fm], strict=True):
                name = slots[idx] if 0 <= idx < len(slots) else None
                if name is None:
                    continue
                by_material[name] = by_material.get(name, 0.0) + float(a)
            dominant = max(by_material.items(), key=lambda kv: kv[1])[0] if by_material else None
            this_object[fm] = len(components)
            components.append(
                {
                    "index": len(components),
                    "faces": int(fm.sum()),
                    "area_m2": float(areas[fm].sum()),
                    "centroid": [float(v) for v in pts.mean(axis=0)],
                    "lo": [float(v) for v in pts.min(axis=0)],
                    "hi": [float(v) for v in pts.max(axis=0)],
                    "material_name": dominant,
                }
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(components), encoding="utf-8")
    # Per-face component ids, so `split_by_part_usd` can regroup geometry without recomputing the
    # union-find -- and, more importantly, without depending on two runs agreeing about it.
    np.savez_compressed(out_path.with_suffix(".faces.npz"), **face_ids)
    total_area = sum(float(c["area_m2"]) for c in components)
    total_faces = sum(int(c["faces"]) for c in components)
    print(
        f"wrote {out_path} ({len(components):,} components, "
        f"{total_faces:,} faces, {total_area:.5f} m2)"
    )
    return {"components": len(components), "faces": total_faces, "area_m2": total_area}


def split_by_part_usd(
    assignment_path: pathlib.Path, face_ids_path: pathlib.Path, out_usd: pathlib.Path
) -> dict[str, object]:
    """Rewrite the imported asset so that **one prim is one part**, then export it as USD.

    Why this pass exists. The renderer binds a temperature per *prim*, and in the source asset a
    prim is a material group -- so a prim is not a part, and no per-prim assignment can express
    one. Measured on the Phantom 4: only 25 of its 41 prims are more than 90 % a single part, and
    those 25 carry just 34.7 % of the asset's area. The prim holding the battery contains **nine**
    parts, of which the battery is 26 % -- painting it at the battery's temperature would be wrong
    about three quarters of it. So the geometry is regrouped rather than the assignment
    approximated.

    Faces are moved with Blender's own `separate` operator rather than rebuilt from arrays, because
    it carries material slots, per-face material indices and UV layers along with the faces;
    rebuilding would drop the textures the visible companion frame depends on.

    The part index rides on the mesh as an integer **face attribute**, which survives both `join`
    and `separate`. That is what makes this robust: nothing depends on face order being preserved
    across an operator, which it is not.

    The per-face component ids come from :func:`emit_components` in the same prepared run, so the
    union-find is never recomputed and two passes cannot disagree about where a shell begins.
    """
    import bpy
    import numpy as np

    assignment = json.loads(assignment_path.read_text(encoding="utf-8"))
    part_of_component: dict[str, str] = assignment["part_of_component"]
    names: list[str] = list(assignment["parts"])
    index_of = {n: i for i, n in enumerate(names)}
    with np.load(face_ids_path, allow_pickle=False) as data:
        face_ids = {k: np.asarray(data[k], dtype=np.int64) for k in data.files}

    # 1. tag every face with its part index (-1 = unassigned, which is dropped)
    tagged = 0
    for obj in [o for o in bpy.data.objects if o.type == "MESH"]:
        ids = face_ids.get(obj.name)
        me = obj.data
        npoly = len(me.polygons)
        values = np.full(npoly, -1, dtype=np.int32)
        if ids is not None and len(ids) == npoly:
            for face, component in enumerate(ids):
                part = part_of_component.get(str(int(component)))
                if part is not None:
                    values[face] = index_of[part]
        attribute = me.attributes.new(name=ATTR_PART, type="INT", domain="FACE")
        attribute.data.foreach_set("value", values)
        tagged += int((values >= 0).sum())
    print(f"tagged {tagged:,} faces with a part index")

    # 2. one object, so one separate per part is enough
    bpy.ops.object.select_all(action="SELECT")
    meshes = [o for o in bpy.data.objects if o.type == "MESH"]
    bpy.context.view_layer.objects.active = meshes[0]
    if len(meshes) > 1:
        bpy.ops.object.join()

    # 3. peel off one part at a time
    made: list[str] = []
    for name in names:
        target = index_of[name]
        current = bpy.context.view_layer.objects.active
        me = current.data
        values = np.empty(len(me.polygons), dtype=np.int32)
        me.attributes[ATTR_PART].data.foreach_get("value", values)
        selected = values == target
        if not selected.any():
            print(f"  part {name!r} has no faces in the import; skipped")
            continue
        if selected.all():
            current.name = current.data.name = name
            made.append(name)
            break
        bpy.ops.object.select_all(action="DESELECT")
        current.select_set(True)
        bpy.context.view_layer.objects.active = current
        me.polygons.foreach_set("select", selected.astype(np.int8))
        before = set(bpy.data.objects.keys())
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.separate(type="SELECTED")
        bpy.ops.object.mode_set(mode="OBJECT")
        fresh = [o.name for o in bpy.data.objects if o.name not in before]
        if not fresh:
            print(f"  part {name!r} did not separate; skipped")
            continue
        piece = bpy.data.objects[fresh[0]]
        piece.name = piece.data.name = name
        made.append(name)
        bpy.context.view_layer.objects.active = current

    # 4. anything still unassigned is geometry no part claimed; it must not reach the stage
    leftovers = [o for o in bpy.data.objects if o.type == "MESH" and o.name not in made]
    for o in leftovers:
        print(f"  dropping {o.name}: {len(o.data.polygons):,} faces no part claimed")
        bpy.data.objects.remove(o, do_unlink=True)

    out_usd.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.usd_export(
        filepath=str(out_usd),
        export_materials=True,
        generate_preview_surface=True,
        root_prim_path="/World",
    )
    print(f"wrote {out_usd} ({len(made)} part prims)")
    return {"parts": len(made), "names": made}


def total_error_signed(before: float, after: float) -> float:
    return 0.0 if before == 0.0 else after / before - 1.0


def run_worker(argv: Sequence[str]) -> int:
    import bpy

    ap = argparse.ArgumentParser(prog="prep_asset (worker)")
    ap.add_argument("--source", type=pathlib.Path, required=True)
    ap.add_argument("--out-usd", type=pathlib.Path, required=True)
    ap.add_argument("--out-prims", type=pathlib.Path, required=True)
    ap.add_argument("--scale", type=float, required=True)
    ap.add_argument("--out-meshes", type=pathlib.Path, default=None)
    ap.add_argument("--out-components", type=pathlib.Path, default=None)
    ap.add_argument("--split-assignment", type=pathlib.Path, default=None)
    ap.add_argument("--split-face-ids", type=pathlib.Path, default=None)
    ap.add_argument("--split-out-usd", type=pathlib.Path, default=None)
    ap.add_argument("--dissolve-deg", type=float, default=5.0)
    ap.add_argument("--max-area-error", type=float, default=0.02)
    args = ap.parse_args(list(argv))

    bpy.ops.wm.read_factory_settings(use_empty=True)
    how = _import_source(args.source)
    meshes = [o for o in bpy.data.objects if o.type == "MESH"]
    tris = sum(sum(len(p.vertices) - 2 for p in o.data.polygons) for o in meshes)
    print(f"imported {args.source.name} via {how}: {len(meshes)} meshes, {tris:,} triangles")

    _rescale(args.scale)
    if args.split_assignment is not None:
        # A split-only pass: exporting the un-split stage first would cost a 115 MB write of a
        # file nobody reads.
        split_by_part_usd(args.split_assignment, args.split_face_ids, args.split_out_usd)
        return 0
    args.out_usd.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.usd_export(
        filepath=str(args.out_usd),
        export_materials=True,
        generate_preview_surface=True,
        root_prim_path="/World",
    )
    records = _prim_records(args.out_usd)
    args.out_prims.parent.mkdir(parents=True, exist_ok=True)
    args.out_prims.write_text(json.dumps(records, indent=1), encoding="utf-8")
    print(f"wrote {args.out_usd}")
    print(f"wrote {args.out_prims} ({len(records)} prim records)")
    if args.out_components is not None:
        emit_components(args.out_components)
    if args.out_meshes is not None:
        emit_meshes(args.out_meshes, args.dissolve_deg, args.max_area_error)
    return 0


# --------------------------------------------------------------------------------------------
# driver -- runs under the project interpreter
# --------------------------------------------------------------------------------------------


def blender_split_command(
    blender: str,
    source: pathlib.Path,
    scale: float,
    components: pathlib.Path,
    assignment: pathlib.Path,
    out_usd: pathlib.Path,
) -> list[str]:
    """The argv for the part-splitting pass. Separate from :func:`blender_command` because it is a
    different job: that one prepares an asset, this one regroups a prepared one."""
    return [
        blender,
        "--background",
        "--factory-startup",
        "--python",
        str(pathlib.Path(__file__).resolve()),
        "--",
        "--source",
        str(source),
        "--out-usd",
        str(out_usd.with_name("unused.usdc")),
        "--out-prims",
        str(out_usd.with_name("unused.prims.json")),
        "--scale",
        repr(float(scale)),
        "--split-assignment",
        str(assignment),
        "--split-face-ids",
        str(components.with_suffix(".faces.npz")),
        "--split-out-usd",
        str(out_usd),
    ]


def blender_command(
    blender: str,
    source: pathlib.Path,
    out_usd: pathlib.Path,
    out_prims: pathlib.Path,
    scale: float,
    out_meshes: pathlib.Path | None = None,
    dissolve_deg: float = 5.0,
    max_area_error: float = 0.02,
) -> list[str]:
    """The exact argv the driver runs. Split out so a test can check it without Blender."""
    extra: list[str] = []
    if out_meshes is not None:
        extra = [
            "--out-meshes",
            str(out_meshes),
            "--dissolve-deg",
            repr(float(dissolve_deg)),
            "--max-area-error",
            repr(float(max_area_error)),
        ]
    return [
        blender,
        "--background",
        "--factory-startup",
        "--python",
        str(pathlib.Path(__file__).resolve()),
        "--",
        "--source",
        str(source),
        "--out-usd",
        str(out_usd),
        "--out-prims",
        str(out_prims),
        "--scale",
        repr(float(scale)),
        *extra,
    ]


def run_driver(argv: Sequence[str] | None = None) -> int:
    from irsim.materials.library import MaterialLibrary
    from irsim.materials.mapping import (
        MaterialResolver,
        PrimRecord,
        audit,
        load_asset_mapping,
        load_mapping_rules,
    )
    from irsim.materials.table import MaterialTable

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--asset", required=True, help="name in configs/assets, or a path to one")
    ap.add_argument(
        "--source", type=pathlib.Path, default=None, help="override the asset's source_file"
    )
    ap.add_argument(
        "--out-dir", type=pathlib.Path, default=None, help="default: data/assets/<name>"
    )
    ap.add_argument("--blender", default="blender", help="the Blender executable")
    ap.add_argument("--band", default="lwir", help="band whose packed table fixes the ids")
    ap.add_argument(
        "--emit-mesh",
        action="store_true",
        help="also write the engine-free thermal mesh archive (<name>.meshes.npz)",
    )
    ap.add_argument(
        "--dissolve-deg",
        type=float,
        default=5.0,
        help="planar dissolve angle; 0 disables. Never collapses, so area is preserved",
    )
    ap.add_argument(
        "--max-area-error",
        type=float,
        default=0.02,
        help="refuse the mesh emit if any prim's area moves by more than this",
    )
    ap.add_argument("--threshold", type=float, default=None, help="coverage gate override")
    ap.add_argument(
        "--emit-parts",
        action="store_true",
        help=(
            "regroup the prepared mesh archive by functional part (ADR 0138) and write it as a "
            "sibling asset `<name>_parts`. Engine-free: reads the archive, not the source file"
        ),
    )
    # AI.3. Nothing counted the geometry until this, so the first import too heavy to solve was
    # discovered by waiting for the solve. The budget is GT.7's measurement, not a guess.
    ap.add_argument(
        "--face-budget",
        type=int,
        default=None,
        help="total faces the thermal archive may carry (default: GT.7's 1,300,000)",
    )
    ap.add_argument(
        "--faces-per-prim",
        type=int,
        default=None,
        help="faces any one prim may carry (default 200,000)",
    )
    ap.add_argument(
        "--allow-over-budget",
        action="store_true",
        help="report the geometry budget but do not fail on it",
    )
    ap.add_argument(
        "--skip-convert",
        action="store_true",
        help="audit an existing prims.json in --out-dir without re-running Blender",
    )
    args = ap.parse_args(argv)

    library = MaterialLibrary.load()
    asset = load_asset_mapping(args.asset, known_materials=library.names)
    source = args.source or (None if asset.source_file is None else REPO_ROOT / asset.source_file)
    out_dir = args.out_dir or (REPO_ROOT / "data" / "assets" / asset.name)
    out_usd = out_dir / f"{asset.name}.usdc"
    out_prims = out_dir / f"{asset.name}.prims.json"
    out_meshes = out_dir / f"{asset.name}.meshes.npz" if args.emit_mesh else None

    if not args.skip_convert:
        if source is None:
            ap.error("no --source and the asset config has no source_file")
        if not source.exists():
            ap.error(f"source not found: {source}")
        if source.suffix.lower() not in IMPORTERS:
            ap.error(f"unsupported source {source.suffix!r}; know {sorted(IMPORTERS)}")
        cmd = blender_command(
            args.blender,
            source,
            out_usd,
            out_prims,
            asset.scale_to_metres,
            out_meshes,
            args.dissolve_deg,
            args.max_area_error,
        )
        print(f"$ {' '.join(cmd)}")
        try:
            done = subprocess.run(cmd, check=False)
        except FileNotFoundError:
            ap.error(f"{args.blender!r} not found; pass --blender /path/to/blender")
        if done.returncode != 0:
            print(f"blender exited {done.returncode}", file=sys.stderr)
            return done.returncode
    if not out_prims.exists():
        print(f"no prim records at {out_prims}", file=sys.stderr)
        return 2

    names = MaterialTable.from_library(library, args.band).names
    rules = load_mapping_rules(known_materials=library.names)
    records = [PrimRecord.from_dict(d) for d in json.loads(out_prims.read_text(encoding="utf-8"))]
    report = audit(records, MaterialResolver(rules, names, asset=asset), args.threshold)
    print(report.render())
    ok = report.passed

    # The geometry budget (AI.3), whenever the thermal archive exists -- emitted just now or by an
    # earlier run. It is read engine-free from the `.npz`, so this costs nothing and the answer is
    # about the mesh that will actually be solved rather than the one Blender imported.
    archive = out_dir / f"{asset.name}.meshes.npz"
    if archive.exists():
        from irsim.io.asset_budget import GeometryBudget, measure_geometry
        from irsim.io.assets import load_asset_meshes

        defaults = GeometryBudget()
        budget = GeometryBudget(
            total_faces=args.face_budget or defaults.total_faces,
            faces_per_prim=args.faces_per_prim or defaults.faces_per_prim,
        )
        geometry = measure_geometry(load_asset_meshes(archive), budget, asset=asset.name)
        print()
        print(geometry.render())
        if not geometry.passed and not args.allow_over_budget:
            ok = False
    elif args.emit_mesh:
        print(f"no mesh archive at {archive}: the geometry budget was not checked")

    # The functional part decomposition (AI.5, ADR 0138). Engine-free: it regroups the triangles
    # the archive already holds, so it needs neither Blender nor the source file, and a selector
    # can be re-tuned in seconds rather than by re-importing a 60 MB FBX.
    if args.emit_parts:
        if asset.parts is None:
            print(
                f"{asset.name}: no `parts:` block in its config; nothing to emit", file=sys.stderr
            )
            return 1
        if not archive.exists():
            print(f"no mesh archive at {archive}; run with --emit-mesh first", file=sys.stderr)
            return 1
        from irsim.io.asset_parts import split_by_part
        from irsim.io.assets import load_asset_meshes, write_asset_meshes

        parts, part_report = split_by_part(load_asset_meshes(archive), asset.parts)
        print()
        print(part_report.render())
        greedy = part_report.greedy_parts()
        if greedy:
            print(
                f"  NOTE: {', '.join(greedy)} hold(s) over 60 % of the asset -- check the selector"
            )
        if not part_report.passed:
            ok = False
        else:
            name = f"{asset.name}_parts"
            written = write_asset_meshes(out_dir.parent / name / f"{name}.meshes.npz", parts)
            print(f"wrote {written} ({len(parts)} parts)")

            # The renderer binds a temperature per prim, so the asset needs a USD whose prims ARE
            # parts. That regrouping is geometry work and happens in Blender -- on the CPU, in
            # background mode, booting no Kit (ADR 0128's whole point).
            components = out_dir / f"{asset.name}.components.json"
            if not components.exists() or not components.with_suffix(".faces.npz").exists():
                cmd = blender_command(
                    args.blender,
                    source,
                    out_dir / "unused.usdc",
                    out_dir / "unused.prims.json",
                    asset.scale_to_metres,
                ) + ["--out-components", str(components)]
                print("measuring components: " + " ".join(cmd[:5]) + " ...")
                if subprocess.run(cmd, check=False).returncode != 0:
                    print("the component pass failed", file=sys.stderr)
                    return 1

            stats = json.loads(components.read_text(encoding="utf-8"))
            from irsim.io.asset_parts import Component as _Component

            part_of: dict[str, str] = {}
            for entry in stats:
                component = _Component(
                    index=int(entry["index"]),
                    faces=int(entry["faces"]),
                    area_m2=float(entry["area_m2"]),
                    centroid=tuple(entry["centroid"]),
                    lo=tuple(entry["lo"]),
                    hi=tuple(entry["hi"]),
                    material_name=entry["material_name"],
                )
                for spec in asset.parts.parts:
                    if spec.select.accepts(component, asset.parts.centre):
                        part_of[str(component.index)] = spec.name
                        break
            assignment = out_dir / f"{asset.name}.part_assignment.json"
            assignment.write_text(
                json.dumps({"parts": list(asset.parts.names), "part_of_component": part_of}),
                encoding="utf-8",
            )
            print(f"assigned {len(part_of):,} of {len(stats):,} components to parts")

            cmd = blender_split_command(
                args.blender,
                source,
                asset.scale_to_metres,
                components,
                assignment,
                out_dir.parent / name / f"{name}.usdc",
            )
            print("splitting geometry by part: " + " ".join(cmd[:5]) + " ...")
            if subprocess.run(cmd, check=False).returncode != 0:
                print("the part-split pass failed", file=sys.stderr)
                return 1

    return 0 if ok else 1


def main(argv: Sequence[str] | None = None) -> int:
    try:
        import bpy  # noqa: F401
    except ImportError:
        return run_driver(argv)
    inner = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    return run_worker(inner)


if __name__ == "__main__":
    sys.exit(main())
