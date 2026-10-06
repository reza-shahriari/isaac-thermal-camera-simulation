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
from collections.abc import Mapping, Sequence
from typing import Any

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

#: A prim carrying less than this fraction of the asset's area is reported but not gated on its
#: own relative area error -- see `emit_meshes`.
PRIM_AREA_FLOOR_FRACTION = 0.001

#: Integer face attribute carrying a part index. It survives both `join` and `separate`, which
#: is what lets `split_by_part_usd` regroup geometry without depending on face order (ADR 0138).
ATTR_PART = "irsim_part"

#: The per-prim override the resolver reads first (ADR 0047); `irsim_isaac.pipeline.materials_usd`
#: reads the same name inside Kit.
THERMAL_MATERIAL_ATTR = "thermal:material"

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


#: The glTF importer's own collection for what it must never export: the icosphere it creates as
#: a custom display shape for a skinned file's bones (io_scene_gltf2 `blender/imp/node.py`). A
#: drone has no skin, so no asset met it before the first rigged human (HU.3); left in, it is an
#: 11.7 m² "part" that outweighs the body.
GLTF_NOT_EXPORTED_COLLECTION = "glTF_not_exported"


def _drop_gltf_bone_shapes() -> None:
    import bpy

    coll = bpy.data.collections.get(GLTF_NOT_EXPORTED_COLLECTION)
    if coll is None:
        return
    for obj in list(coll.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    bpy.data.collections.remove(coll)


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
        _drop_gltf_bone_shapes()
        return "import_scene.gltf"
    bpy.ops.wm.usd_import(filepath=str(path))
    return "wm.usd_import"


#: How far apart (per channel, linear 0..1) two colour-attribute samples may be and still count as
#: one colour: half an 8-bit step, so a tint quantised by the exporter still folds.
TINT_UNIFORM_TOL = 0.5 / 255.0


def uniform_colour(
    samples: Sequence[Sequence[float]], tol: float = TINT_UNIFORM_TOL
) -> tuple[float, ...] | None:
    """The one RGBA every sample shares (within ``tol`` per channel), or None if they differ.

    None also for no samples: a material no face uses has no colour to fold.
    """
    if not samples:
        return None
    lo = [min(s[c] for s in samples) for c in range(4)]
    hi = [max(s[c] for s in samples) for c in range(4)]
    if any(h - lo_c > tol for lo_c, h in zip(lo, hi, strict=True)):
        return None
    return tuple((lo_c + h) / 2.0 for lo_c, h in zip(lo, hi, strict=True))


def fold_multiply(tint: Sequence[float], factor: Sequence[float], fac: float = 1.0) -> list[float]:
    """Blender's Mix (MULTIPLY) of ``tint`` by ``factor`` at strength ``fac``, as one RGBA.

    Mix colour MULTIPLY is ``a * (1 - fac + fac * b)`` per RGB channel; alpha is ``a``'s.
    """
    rgb = [a * (1.0 - fac + fac * b) for a, b in zip(tint[:3], factor[:3], strict=True)]
    return [*rgb, tint[3] if len(tint) > 3 else 1.0]


def _fold_vertex_tints() -> None:
    """Fold a glTF vertex-colour tint into the material, so the USD carries the colour seen.

    Blender's glTF importer turns ``COLOR_0`` into a Color Attribute node multiplied into the
    base colour through a Mix node. The USD exporter cannot write that graph: it falls back to
    the socket's 0.8 grey default (an untextured base) or to the bare texture, so a black
    aircraft renders white in every RGB companion (the Matrice 100, AI.23). Where a material
    multiplies a colour attribute that is one colour on every face using it, the product is the
    material's colour: tint x constant becomes the base colour; white x texture hands the
    texture straight to the base. Anything else -- a tint varying across faces, a coloured tint
    over a texture -- is left as imported and reported: it needs a texture bake, not done here.
    """
    import bpy

    samples: dict[str, list[tuple[float, ...]]] = {}
    for obj in bpy.data.objects:
        if obj.type != "MESH" or not obj.data.color_attributes:
            continue
        mesh = obj.data
        attr = mesh.color_attributes.active_color or mesh.color_attributes[0]
        colours = [tuple(c.color) for c in attr.data]
        for poly in mesh.polygons:
            if poly.material_index >= len(obj.material_slots):
                continue
            mat = obj.material_slots[poly.material_index].material
            if mat is None:
                continue
            idx = poly.loop_indices[0] if attr.domain == "CORNER" else poly.vertices[0]
            samples.setdefault(mat.name, []).append(colours[idx])

    for mat in bpy.data.materials:
        if not mat.use_nodes or mat.name not in samples:
            continue
        bsdf = next((n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED"), None)
        if bsdf is None or not bsdf.inputs["Base Color"].is_linked:
            continue
        mix = bsdf.inputs["Base Color"].links[0].from_node
        if mix.type != "MIX" or mix.data_type != "RGBA" or mix.blend_type != "MULTIPLY":
            continue
        fac_in, a_in, b_in = mix.inputs["Factor"], mix.inputs[6], mix.inputs[7]
        if fac_in.is_linked:
            continue
        sides = [(a_in, b_in), (b_in, a_in)]
        tint_side = next(
            (
                (t, o)
                for t, o in sides
                if t.is_linked and t.links[0].from_node.type == "VERTEX_COLOR"
            ),
            None,
        )
        if tint_side is None:
            continue
        tint = uniform_colour(samples[mat.name])
        if tint is None:
            print(f"vertex tint on {mat.name!r} varies across faces; left as imported")
            continue
        _, other = tint_side
        links = mat.node_tree.links
        if not other.is_linked:
            colour = fold_multiply(tint, list(other.default_value), fac_in.default_value)
            links.remove(bsdf.inputs["Base Color"].links[0])
            bsdf.inputs["Base Color"].default_value = colour
            print(
                f"vertex tint on {mat.name!r} folded: base colour {[round(c, 4) for c in colour]}"
            )
        elif all(c >= 1.0 - TINT_UNIFORM_TOL for c in tint[:3]):
            # white x texture == texture: hand the texture straight to the BSDF
            links.new(other.links[0].from_socket, bsdf.inputs["Base Color"])
            print(f"vertex tint on {mat.name!r} is white; its texture is the base colour")
        else:
            print(f"vertex tint on {mat.name!r} multiplies a texture; left as imported")


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


def _rotate(degrees: Sequence[float]) -> None:
    """Turn every root object about the world origin by the asset's ``rotate_deg``, applied.

    XYZ Euler, degrees, in Blender's Z-up frame after import. Runs before :func:`_rescale` and
    identically in every pass, like the weld: the part split indexes faces by the order the
    component pass saw them in, and every selector is a position in the turned frame.
    """
    import math

    import bpy
    import mathutils

    if not any(degrees):
        return
    turn = mathutils.Euler([math.radians(d) for d in degrees], "XYZ").to_matrix().to_4x4()
    for obj in bpy.data.objects:
        if obj.parent is None:
            obj.matrix_world = turn @ obj.matrix_world
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=False)
    bpy.ops.object.select_all(action="DESELECT")
    print(f"rotated {tuple(degrees)} deg (XYZ) about the origin")


def _cut(cuts: Sequence[Mapping[str, Any]]) -> None:
    """Split chosen pieces along planes: the asset's ``cuts`` (:class:`CutSpec`).

    Runs after the turn, the rescale and the weld, identically in every pass: the part split
    indexes faces by the order the component pass saw them in. A piece is a set of faces of the
    named materials joined by shared edges; one that passes the area and centroid filters is
    bisected along the plane and its new edges split, so its two sides share no edge. The plane is
    in world metres and is carried into each object's local frame, because the rescale applies
    location and scale but not the importer's rotations.
    """
    import bmesh
    import bpy
    import mathutils

    total = 0
    for spec in cuts:
        materials = {m.lower() for m in spec.get("materials") or []}
        box = [(spec.get(f"{a}_min"), spec.get(f"{a}_max")) for a in "xyz"]
        point = mathutils.Vector(spec["point"])
        normal = mathutils.Vector(spec["normal"]).normalized()
        for obj in [o for o in bpy.data.objects if o.type == "MESH"]:
            world = obj.matrix_world
            slots = [(s.material.name.lower() if s.material else "") for s in obj.material_slots]
            bm = bmesh.new()
            bm.from_mesh(obj.data)
            bm.faces.ensure_lookup_table()

            def wanted(
                face: Any, slots: list[str] = slots, materials: set[str] = materials
            ) -> bool:
                if not materials:
                    return True
                i = face.material_index
                return i < len(slots) and slots[i] in materials

            seen: set[int] = set()
            targets: list[Any] = []
            for seed in bm.faces:
                if seed.index in seen or not wanted(seed):
                    continue
                piece, stack = [], [seed]
                seen.add(seed.index)
                while stack:
                    face = stack.pop()
                    piece.append(face)
                    for edge in face.edges:
                        for other in edge.link_faces:
                            if other.index not in seen and wanted(other):
                                seen.add(other.index)
                                stack.append(other)
                area, moment = 0.0, mathutils.Vector((0.0, 0.0, 0.0))
                for face in piece:
                    corners = [world @ v.co for v in face.verts]
                    for k in range(1, len(corners) - 1):
                        tri = (corners[k] - corners[0]).cross(corners[k + 1] - corners[0])
                        a = 0.5 * tri.length
                        area += a
                        moment += a * (corners[0] + corners[k] + corners[k + 1]) / 3.0
                if area <= 0.0 or area < float(spec.get("area_min_m2") or 0.0):
                    continue
                centre = moment / area
                if any(
                    (lo is not None and centre[i] < lo) or (hi is not None and centre[i] > hi)
                    for i, (lo, hi) in enumerate(box)
                ):
                    continue
                targets.extend(piece)
            if targets:
                local_point = world.inverted() @ point
                local_normal = (world.to_3x3().transposed() @ normal).normalized()
                verts = {v for f in targets for v in f.verts}
                edges = {e for f in targets for e in f.edges}
                done = bmesh.ops.bisect_plane(
                    bm,
                    geom=[*verts, *edges, *targets],
                    dist=1e-9,
                    plane_co=local_point,
                    plane_no=local_normal,
                )
                new_edges = [g for g in done["geom_cut"] if isinstance(g, bmesh.types.BMEdge)]
                if new_edges:
                    bmesh.ops.split_edges(bm, edges=new_edges)
                    total += len(new_edges)
                bm.to_mesh(obj.data)
            bm.free()
    if cuts:
        print(f"cut along {len(cuts)} plane(s): {total:,} edges split")


def _weld(distance_m: float) -> None:
    """Merge vertices closer than ``distance_m`` in every mesh: the asset's ``weld_m`` (AI.18).

    glTF splits a vertex at every UV and normal seam, so a surface arrives as shells that share no
    edge. Conduction crosses only shared edges (ADR 0112), and a part selector judges shells, so
    both need the seams closed. UVs and normals live on face corners, not vertices, so they
    survive. Runs after :func:`_rescale` (the distance is metres) and identically in every pass,
    because the part split indexes faces by the order the component pass saw them in.
    """
    import bmesh
    import bpy

    if distance_m <= 0.0:
        return
    before = after = 0
    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue
        bm = bmesh.new()
        bm.from_mesh(obj.data)
        before += len(bm.verts)
        bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=distance_m)
        after += len(bm.verts)
        bm.to_mesh(obj.data)
        bm.free()
    print(f"welded at {1000 * distance_m:g} mm: {before:,} -> {after:,} vertices")


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
            # `irsim_isaac.pipeline.materials_usd.walk_stage(expand_subsets=True)` is the same
            # reading inside Kit, and `tests/unit/data/subset_building.usda` is the fixture that
            # runs both (`AI.4`). The two must keep agreeing; nothing but that fixture says so.
            if binding.ComputeBoundMaterial()[0]:
                shadowed.append(str(prim.GetPath()))
            for subset in subsets:
                material = UsdShade.MaterialBindingAPI(subset.GetPrim()).ComputeBoundMaterial()[0]
                # the override is authored on the mesh and a subset inherits it, exactly as
                # `materials_usd.walk_stage` reads it inside Kit -- a part's asserted material
                # (AI.14) lands on a multi-material mesh and must reach every face group of it
                records.append(_record(subset.GetPrim(), material, override_from=prim))
            continue
        records.append(_record(prim, binding.ComputeBoundMaterial()[0]))
    if shadowed:
        print(
            f"note: {len(shadowed)} mesh(es) carry materialBind subsets AND a direct binding; "
            "the subsets win (ADR 0128). First: " + shadowed[0]
        )
    return records


def _record(
    prim: object, material: object, override_from: object | None = None
) -> dict[str, object]:
    name = None
    if material is not None and material.GetPrim().IsValid():  # type: ignore[attr-defined]
        name = str(material.GetPrim().GetName())  # type: ignore[attr-defined]
    override = None
    carrier = prim if override_from is None else override_from
    attr = carrier.GetAttribute(THERMAL_MATERIAL_ATTR)  # type: ignore[attr-defined]
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


def world_polygon_areas(
    vertices_world: Any, triangles: Any, triangle_polygon: Any, n_polygons: int
) -> Any:
    """Each polygon's area in the stage frame, from its triangulation's world-space vertices.

    Summing triangles rather than scaling Blender's per-polygon area is what makes this right
    under any transform, including a non-uniform one, where no single factor converts the two.
    """
    import numpy as np

    v = np.asarray(vertices_world, dtype=np.float64)
    t = np.asarray(triangles, dtype=np.int64)
    cross = np.cross(v[t[:, 1]] - v[t[:, 0]], v[t[:, 2]] - v[t[:, 0]])
    tri_area = 0.5 * np.linalg.norm(cross, axis=1)
    return np.bincount(
        np.asarray(triangle_polygon, dtype=np.int64), weights=tri_area, minlength=n_polygons
    )


def emit_components(
    out_path: pathlib.Path,
    weld_m: float = 0.0,
    rotate_deg: Sequence[float] = (0.0, 0.0, 0.0),
    cuts: Sequence[Mapping[str, Any]] = (),
) -> dict[str, object]:
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

        # World-space areas, never `polygons.foreach_get("area")`: that is the LOCAL area, and a
        # glTF imported at scale 1.0 keeps its node transforms (nothing is applied), so the
        # DJI Mini 3 Pro's local areas summed to 40,701 m2 against its true 0.113.
        me.calc_loop_triangles()
        nt = len(me.loop_triangles)
        tri = np.empty(nt * 3, dtype=np.int64)
        me.loop_triangles.foreach_get("vertices", tri)
        tri_poly = np.empty(nt, dtype=np.int64)
        me.loop_triangles.foreach_get("polygon_index", tri_poly)
        areas = world_polygon_areas(co, tri.reshape(nt, 3), tri_poly, npoly)
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
                    # the object it came from: `granularity: object` (AI.16) claims whole objects
                    "object": obj.name,
                    # the weld these shells were measured under (AI.18): a cache from another
                    # weld indexes different shells, so the driver re-measures rather than reuse it
                    "weld_m": weld_m,
                    # likewise the turn (rotate_deg): every centroid moves with it
                    "rotate_deg": [float(d) for d in rotate_deg],
                    # and the cuts: each one changes which shells exist
                    "cuts": list(cuts),
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


def emit_material_slots(out_path: pathlib.Path) -> dict[str, object]:
    """Write every mesh's per-face material slot and area, for `AI.6`'s split.

    The planning half is engine-free (:mod:`irsim.io.asset_material_split`) and cannot see a
    Blender scene, so the worker's job is to hand it two parallel per-face arrays and the slot
    table that indexes them. Areas are metres squared -- ``scale_to_metres`` has already been
    applied by :func:`_rescale` -- because the decision the planner makes is about area and not
    about face counts.

    A slot with no material assigned is reported as an empty name rather than dropped: a face
    painted with nothing is still a face, and silently merging it into slot 0 is the failure this
    whole pass exists to end.
    """
    import bpy
    import numpy as np

    arrays: dict[str, Any] = {}
    table: dict[str, list[str]] = {}
    for obj in sorted((o for o in bpy.data.objects if o.type == "MESH"), key=lambda o: o.name):
        me = obj.data
        npoly = len(me.polygons)
        if not npoly:
            continue
        slots = np.empty(npoly, dtype=np.int64)
        me.polygons.foreach_get("material_index", slots)
        areas = np.empty(npoly, dtype=np.float64)
        me.polygons.foreach_get("area", areas)
        names = [ms.material.name if ms.material else "" for ms in obj.material_slots] or [""]
        # A face may reference a slot the object does not have (an import artefact); clamp rather
        # than raise, and let the planner see it as the last real slot.
        np.clip(slots, 0, len(names) - 1, out=slots)
        arrays[f"{obj.name}::slot"] = slots
        arrays[f"{obj.name}::area"] = areas
        table[obj.name] = names

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(table, indent=1), encoding="utf-8")
    np.savez_compressed(out_path.with_suffix(".slots.npz"), **arrays)
    multi = sum(1 for k, v in arrays.items() if k.endswith("::slot") and len(set(v.tolist())) > 1)
    print(f"wrote {out_path} ({len(table)} mesh(es), {multi} carrying more than one material)")
    return {"meshes": len(table), "multi_material": multi}


def split_by_material_usd(plan_path: pathlib.Path, out_usd: pathlib.Path) -> dict[str, object]:
    """Rewrite the asset so that **one prim is one material**, then export it as USD (`AI.6`).

    The renderer transports one instance id per prim and indexes the material table by it (ADR
    0014), so a mesh carrying three materials renders as one of them however faithfully the asset
    authored its ``materialBind`` subsets. `AI.4` made the audit read those subsets; this makes
    the geometry match what the audit says.

    Blender's importers put a USD subset, an FBX material group and an OBJ ``usemtl`` on the same
    footing -- a per-face material index into the object's slot table -- so this pass is one
    mechanism for every source format, and reads no USD itself.

    Faces move with ``mesh.separate(type='MATERIAL')`` rather than being rebuilt from arrays,
    which carries the slots, the per-face indices and the UV layers with them; rebuilding would
    drop the textures the visible companion frame depends on (the same reason
    :func:`split_by_part_usd` uses the operator).

    Names come from the plan, not from the exporter. Blender names a separated piece after its
    parent (``Facade.001``), which is neither stable nor a legal USD identifier, and a scene
    config has to be authored against *some* prim path.
    """
    import bpy

    plan: dict[str, list[dict[str, object]]] = json.loads(plan_path.read_text(encoding="utf-8"))
    made: list[str] = []
    for name in sorted(plan):
        obj = bpy.data.objects.get(name)
        if obj is None or obj.type != "MESH":
            print(f"  {name!r} is not in the import; skipped")
            continue
        wanted = {str(s["material"]): str(s["piece"]) for s in plan[name]}
        if len(wanted) == 1:
            obj.name = obj.data.name = next(iter(wanted.values()))
            made.append(obj.name)
            continue
        before = set(bpy.data.objects.keys())
        bpy.ops.object.select_all(action="DESELECT")
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.mesh.separate(type="MATERIAL")
        bpy.ops.object.mode_set(mode="OBJECT")
        # `bpy.data.objects` iterates *objects*, not their names, so the difference is taken
        # over `.keys()` explicitly -- and materialised as a set first, because a comprehension
        # over `.keys()` is what ruff's SIM118 rewrites into the iteration that does not work.
        fresh = sorted(set(bpy.data.objects.keys()) - before)
        pieces = [obj] + [bpy.data.objects[k] for k in fresh]
        for piece in pieces:
            # After the separate each piece uses exactly one material, so its first face names it.
            slots = [ms.material.name if ms.material else "" for ms in piece.material_slots]
            index = piece.data.polygons[0].material_index if len(piece.data.polygons) else 0
            material = slots[index] if 0 <= index < len(slots) else ""
            target = wanted.get(material)
            if target is None:
                print(f"  {piece.name!r} came out as {material!r}, which the plan does not name")
                continue
            piece.name = piece.data.name = target
            made.append(target)

    out_usd.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.usd_export(
        filepath=str(out_usd),
        export_materials=True,
        generate_preview_surface=True,
        root_prim_path="/World",
    )
    print(f"wrote {out_usd} ({len(made)} material prims)")
    return {"prims": len(made), "names": made}


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
    asserted: dict[str, str] = dict(assignment.get("materials", {}))
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
    # The joined object keeps its first mesh's name, and a part may carry exactly that name (AI.18:
    # parts selected by object name, as the Blender add-on exports them). Peeling a piece off and
    # naming it after its part would then collide with the object it came from -- Blender would
    # call the piece `block.001` and the lookup below would miss. A neutral name avoids that.
    joined = bpy.context.view_layer.objects.active
    joined.name = joined.data.name = "__irsim_unsplit__"

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
        # Face selection must be the ONLY selection: the glTF importer leaves every vertex
        # selected, and in vertex-select mode edit mode rebuilds face selection from the vertices
        # -- so the first `separate` took the whole DJI Mini 3 Pro and fourteen parts came out
        # empty. The Phantom 4's FBX import happened to leave nothing selected.
        me.vertices.foreach_set("select", np.zeros(len(me.vertices), dtype=np.int8))
        me.edges.foreach_set("select", np.zeros(len(me.edges), dtype=np.int8))
        me.polygons.foreach_set("select", selected.astype(np.int8))
        bpy.context.tool_settings.mesh_select_mode = (False, False, True)
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

    # 4b. parts at the top of the stage. A glTF arrives under its own node chain (Sketchfab's
    #     `Sketchfab_model/root/GLTF_SceneRootNode/...`) and a separated piece keeps its parent,
    #     so the parts would land deep in the tree -- where neither a scene's `prim_root/<part>/
    #     <part>` binding nor the material stamp below finds them. Unparent keeping the world
    #     transform, then drop the empties that held the chain.
    for name in made:
        piece = bpy.data.objects[name]
        if piece.parent is not None:
            world = piece.matrix_world.copy()
            piece.parent = None
            piece.matrix_world = world
    for o in [o for o in bpy.data.objects if o.type == "EMPTY"]:
        bpy.data.objects.remove(o, do_unlink=True)

    # 4c. a separated piece keeps EVERY material slot of the joined mesh. Where a part's faces all
    #     use one slot that is not the first, Blender's USD export binds slot 0 to the mesh and
    #     writes no subset -- the Inspire 3's lens glass came out bound to the texture atlas, in
    #     the render as well as in the audit. Drop the slots a part does not use.
    for name in made:
        piece = bpy.data.objects[name]
        bpy.ops.object.select_all(action="DESELECT")
        piece.select_set(True)
        bpy.context.view_layer.objects.active = piece
        bpy.ops.object.material_slot_remove_unused()

    # 5. a part's asserted material (AI.14) rides on the object in the master .blend, where an
    #    operator reopening it can see it, and on the part's prims in the USD, where it matters
    for name in made:
        if name in asserted:
            bpy.data.objects[name][THERMAL_MATERIAL_ATTR] = asserted[name]

    out_usd.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.usd_export(
        filepath=str(out_usd),
        export_materials=True,
        generate_preview_surface=True,
        root_prim_path="/World",
    )
    stamped = _stamp_part_materials(out_usd, {n: asserted[n] for n in made if n in asserted})
    records = _prim_records(out_usd)
    prims_json = out_usd.with_suffix(".prims.json")
    prims_json.write_text(json.dumps(records, indent=1), encoding="utf-8")
    print(f"wrote {out_usd} ({len(made)} part prims, {stamped} carrying an asserted material)")
    print(f"wrote {prims_json} ({len(records)} prim records)")
    return {"parts": len(made), "names": made}


def _stamp_part_materials(usd_path: pathlib.Path, materials: Mapping[str, str]) -> int:
    """Author ``thermal:material`` on every mesh under each part that asserts one (AI.14).

    Written with ``pxr`` after the export rather than through Blender's custom-property export,
    which namespaces a property as ``userProperties:…`` -- a name the resolver does not read.
    Blender exports object ``<part>`` as ``/World/<part>`` holding mesh ``<part>``; every mesh
    under the part's Xform is stamped, so the rule does not depend on that nesting.
    """
    from pxr import Sdf, Usd, UsdGeom

    if not materials:
        return 0
    stage = Usd.Stage.Open(str(usd_path))
    stamped = 0
    for part, material in materials.items():
        root = stage.GetPrimAtPath(f"/World/{part}")
        if not root.IsValid():
            raise RuntimeError(f"part {part!r} is not at /World/{part} in {usd_path}")
        for prim in Usd.PrimRange(root):
            if prim.IsA(UsdGeom.Mesh):
                prim.CreateAttribute(THERMAL_MATERIAL_ATTR, Sdf.ValueTypeNames.String).Set(material)
                stamped += 1
    stage.GetRootLayer().Save()
    return stamped


def total_error_signed(before: float, after: float) -> float:
    return 0.0 if before == 0.0 else after / before - 1.0


def export_side_artefacts(save_blend: pathlib.Path | None, emit_fbx: pathlib.Path | None) -> None:
    """Save the master ``.blend`` and an interchange FBX of whatever scene the pass built.

    The library's per-asset formats (ADR 0150, the owner's 2026-09-26 decision): the ``.usdc``
    is what a scene references, the ``.blend`` is the master an operator reopens, and the FBX
    is a courtesy for non-USD consumers — Blender's FBX exporter re-bakes axis conventions and
    re-realises instancing, exactly the ambiguity the USD path avoids, so nothing downstream
    may depend on the FBX. Called at the end of a pass so all three artefacts describe one
    scene: after the rescale on a plain prep, after the regroup on a part or material split.
    """
    import bpy

    if save_blend is not None:
        save_blend.parent.mkdir(parents=True, exist_ok=True)
        bpy.ops.wm.save_as_mainfile(filepath=str(save_blend))
        print(f"wrote {save_blend}")
    if emit_fbx is not None:
        emit_fbx.parent.mkdir(parents=True, exist_ok=True)
        # A glTF's images arrive *packed* in the scene with no file behind them, and the FBX
        # exporter copies textures from disk -- so a glTF-sourced FBX came out untextured
        # ("missing .../textures/packed/Image_0, not copying"). Write them out beside it first.
        for img in bpy.data.images:
            if img.packed_file is None:
                continue
            ext = ".jpg" if img.file_format == "JPEG" else ".png"
            target = emit_fbx.parent / "textures" / f"{bpy.path.clean_name(img.name)}{ext}"
            target.parent.mkdir(parents=True, exist_ok=True)
            img.filepath_raw = str(target)
            img.save()
        bpy.ops.export_scene.fbx(filepath=str(emit_fbx), use_selection=False, path_mode="COPY")
        print(f"wrote {emit_fbx}")


def run_worker(argv: Sequence[str]) -> int:
    import bpy

    ap = argparse.ArgumentParser(prog="prep_asset (worker)")
    ap.add_argument("--source", type=pathlib.Path, required=True)
    ap.add_argument("--out-usd", type=pathlib.Path, required=True)
    ap.add_argument("--out-prims", type=pathlib.Path, required=True)
    ap.add_argument("--scale", type=float, required=True)
    ap.add_argument("--weld-m", type=float, default=0.0)
    ap.add_argument("--rotate-deg", type=float, nargs=3, default=(0.0, 0.0, 0.0))
    ap.add_argument("--cuts-json", default="[]", help="the asset's cuts, as JSON")
    ap.add_argument("--out-meshes", type=pathlib.Path, default=None)
    ap.add_argument("--out-components", type=pathlib.Path, default=None)
    ap.add_argument("--split-assignment", type=pathlib.Path, default=None)
    ap.add_argument("--split-face-ids", type=pathlib.Path, default=None)
    ap.add_argument("--split-out-usd", type=pathlib.Path, default=None)
    ap.add_argument("--out-material-slots", type=pathlib.Path, default=None)
    ap.add_argument("--split-material-plan", type=pathlib.Path, default=None)
    ap.add_argument("--dissolve-deg", type=float, default=5.0)
    ap.add_argument("--max-area-error", type=float, default=0.02)
    ap.add_argument("--save-blend", type=pathlib.Path, default=None)
    ap.add_argument("--emit-fbx", type=pathlib.Path, default=None)
    args = ap.parse_args(list(argv))

    bpy.ops.wm.read_factory_settings(use_empty=True)
    how = _import_source(args.source)
    _fold_vertex_tints()
    meshes = [o for o in bpy.data.objects if o.type == "MESH"]
    tris = sum(sum(len(p.vertices) - 2 for p in o.data.polygons) for o in meshes)
    print(f"imported {args.source.name} via {how}: {len(meshes)} meshes, {tris:,} triangles")

    _rotate(args.rotate_deg)
    _rescale(args.scale)
    _weld(args.weld_m)
    cuts = json.loads(args.cuts_json)
    _cut(cuts)
    if args.out_material_slots is not None:
        # A measure-only pass: the planner runs under the project interpreter, which Blender's
        # does not share, so the two halves talk through files exactly as the part split does.
        emit_material_slots(args.out_material_slots)
        return 0
    if args.split_material_plan is not None:
        split_by_material_usd(args.split_material_plan, args.split_out_usd)
        export_side_artefacts(args.save_blend, args.emit_fbx)
        return 0
    if args.split_assignment is not None:
        # A split-only pass: exporting the un-split stage first would cost a 115 MB write of a
        # file nobody reads.
        split_by_part_usd(args.split_assignment, args.split_face_ids, args.split_out_usd)
        export_side_artefacts(args.save_blend, args.emit_fbx)
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
        emit_components(args.out_components, args.weld_m, args.rotate_deg, cuts)
    if args.out_meshes is not None:
        emit_meshes(args.out_meshes, args.dissolve_deg, args.max_area_error)
    export_side_artefacts(args.save_blend, args.emit_fbx)
    return 0


# --------------------------------------------------------------------------------------------
# driver -- runs under the project interpreter
# --------------------------------------------------------------------------------------------


def _weld_args(weld_m: float) -> list[str]:
    """``--weld-m`` for every pass, or nothing at 0 so an unwelded asset's argv is unchanged.

    Every pass must weld alike: the part split indexes faces by the component pass's order.
    """
    return ["--weld-m", repr(float(weld_m))] if weld_m > 0.0 else []


def _rotate_args(rotate_deg: Sequence[float]) -> list[str]:
    """``--rotate-deg`` for every pass, or nothing for an unturned asset (its argv is unchanged)."""
    return ["--rotate-deg", *(repr(float(d)) for d in rotate_deg)] if any(rotate_deg) else []


def _cuts_args(cuts: Sequence[Mapping[str, Any]]) -> list[str]:
    """``--cuts-json`` for every pass, or nothing for an asset that cuts nothing."""
    return ["--cuts-json", json.dumps(list(cuts), sort_keys=True)] if cuts else []


def _cut_dicts(asset: Any) -> list[dict[str, Any]]:
    """An asset's cuts as the plain dicts the worker reads (and the component cache records)."""
    return [c.model_dump(mode="json") for c in asset.cuts]


def blender_split_command(
    blender: str,
    source: pathlib.Path,
    scale: float,
    components: pathlib.Path,
    assignment: pathlib.Path,
    out_usd: pathlib.Path,
    save_blend: pathlib.Path | None = None,
    emit_fbx: pathlib.Path | None = None,
    weld_m: float = 0.0,
    rotate_deg: Sequence[float] = (0.0, 0.0, 0.0),
    cuts: Sequence[Mapping[str, Any]] = (),
) -> list[str]:
    """The argv for the part-splitting pass. Separate from :func:`blender_command` because it is a
    different job: that one prepares an asset, this one regroups a prepared one."""
    side: list[str] = []
    if save_blend is not None:
        side += ["--save-blend", str(save_blend)]
    if emit_fbx is not None:
        side += ["--emit-fbx", str(emit_fbx)]
    return [
        blender,
        "--background",
        "--factory-startup",
        # without this Blender exits 0 when the worker raises, and the driver reports success
        "--python-exit-code",
        "1",
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
        *_weld_args(weld_m),
        *_rotate_args(rotate_deg),
        *_cuts_args(cuts),
        "--split-assignment",
        str(assignment),
        "--split-face-ids",
        str(components.with_suffix(".faces.npz")),
        "--split-out-usd",
        str(out_usd),
        *side,
    ]


def blender_material_slots_command(
    blender: str,
    source: pathlib.Path,
    scale: float,
    out_slots: pathlib.Path,
    weld_m: float = 0.0,
    rotate_deg: Sequence[float] = (0.0, 0.0, 0.0),
    cuts: Sequence[Mapping[str, Any]] = (),
) -> list[str]:
    """The argv for `AI.6`'s measuring pass: per-face material slot and area, no export."""
    return [
        blender,
        "--background",
        "--factory-startup",
        # without this Blender exits 0 when the worker raises, and the driver reports success
        "--python-exit-code",
        "1",
        "--python",
        str(pathlib.Path(__file__).resolve()),
        "--",
        "--source",
        str(source),
        "--out-usd",
        str(out_slots.with_name("unused.usdc")),
        "--out-prims",
        str(out_slots.with_name("unused.prims.json")),
        "--scale",
        repr(float(scale)),
        *_weld_args(weld_m),
        *_rotate_args(rotate_deg),
        *_cuts_args(cuts),
        "--out-material-slots",
        str(out_slots),
    ]


def blender_material_split_command(
    blender: str,
    source: pathlib.Path,
    scale: float,
    plan: pathlib.Path,
    out_usd: pathlib.Path,
    weld_m: float = 0.0,
    rotate_deg: Sequence[float] = (0.0, 0.0, 0.0),
    cuts: Sequence[Mapping[str, Any]] = (),
) -> list[str]:
    """The argv for `AI.6`'s splitting pass: one prim per material, exported as USD."""
    return [
        blender,
        "--background",
        "--factory-startup",
        # without this Blender exits 0 when the worker raises, and the driver reports success
        "--python-exit-code",
        "1",
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
        *_weld_args(weld_m),
        *_rotate_args(rotate_deg),
        *_cuts_args(cuts),
        "--split-material-plan",
        str(plan),
        "--split-out-usd",
        str(out_usd),
    ]


def part_assignment(stats: Sequence[Mapping[str, Any]], parts: Any) -> dict[str, Any]:
    """The Blender split's instructions: which part each render component joins (AI.5, AI.14).

    ``stats`` is the component pass's JSON and ``parts`` the asset's ``PartsConfig``. Claiming goes
    through ``PartsConfig.claim`` -- the rule the engine-free report and archive split use -- so an
    excluded component is in no part and the worker drops it with the other unclaimed geometry.
    ``materials`` carries each part's asserted library material for the worker to stamp onto the
    part's prims as ``thermal:material``.
    """
    from irsim.io.asset_parts import Component

    components = [
        Component(
            index=int(entry["index"]),
            faces=int(entry["faces"]),
            area_m2=float(entry["area_m2"]),
            centroid=tuple(entry["centroid"]),
            lo=tuple(entry["lo"]),
            hi=tuple(entry["hi"]),
            material_name=entry["material_name"],
            source=entry.get("object"),
        )
        for entry in stats
    ]
    claimed = parts.claim_all(components)
    part_of = {str(i): spec.name for i, spec in claimed.items() if spec is not None}
    return {
        "parts": list(parts.names),
        "part_of_component": part_of,
        "materials": dict(parts.materials),
        "excluded_components": len(parts.excluded_indices(components)),
    }


def plan_from_slots(slots_json: pathlib.Path) -> tuple[Any, dict[str, list[dict[str, str]]]]:
    """Turn the worker's per-face arrays into `AI.6`'s split plan.

    Returns the report (for printing and for the gate) and the plan the worker reads back. Kept
    out of :func:`run_driver` so a test can exercise it without Blender: the arrays on disk are
    the whole interface between the two interpreters.
    """
    import numpy as np

    from irsim.io.asset_material_split import MeshFaces, plan_material_split

    table: dict[str, list[str]] = json.loads(slots_json.read_text(encoding="utf-8"))
    with np.load(slots_json.with_suffix(".slots.npz"), allow_pickle=False) as data:
        meshes = [
            MeshFaces(
                name=name,
                slot_of_face=np.asarray(data[f"{name}::slot"], dtype=np.int64),
                area_of_face=np.asarray(data[f"{name}::area"], dtype=np.float64),
                materials=materials,
            )
            for name, materials in sorted(table.items())
        ]
    report = plan_material_split(meshes)
    plan = {
        mesh.mesh: [{"material": s.material, "piece": s.piece} for s in mesh.slots]
        for mesh in report.meshes
    }
    return report, plan


def blender_command(
    blender: str,
    source: pathlib.Path,
    out_usd: pathlib.Path,
    out_prims: pathlib.Path,
    scale: float,
    out_meshes: pathlib.Path | None = None,
    dissolve_deg: float = 5.0,
    max_area_error: float = 0.02,
    save_blend: pathlib.Path | None = None,
    emit_fbx: pathlib.Path | None = None,
    weld_m: float = 0.0,
    rotate_deg: Sequence[float] = (0.0, 0.0, 0.0),
    cuts: Sequence[Mapping[str, Any]] = (),
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
    if save_blend is not None:
        extra += ["--save-blend", str(save_blend)]
    if emit_fbx is not None:
        extra += ["--emit-fbx", str(emit_fbx)]
    return [
        blender,
        "--background",
        "--factory-startup",
        # without this Blender exits 0 when the worker raises, and the driver reports success
        "--python-exit-code",
        "1",
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
        *_weld_args(weld_m),
        *_rotate_args(rotate_deg),
        *_cuts_args(cuts),
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
    # AI.6. A prim is what the renderer can address, so a mesh carrying several materials is
    # rendered as one of them until it is split. This is opt-in rather than automatic because it
    # changes an asset's prim paths, and scene configs are authored against those.
    ap.add_argument(
        "--emit-material-split",
        action="store_true",
        help=(
            "write a sibling asset `<name>_materials` whose prims are one per material, so a "
            "multi-material mesh can be rendered as what it is (AI.6)"
        ),
    )
    ap.add_argument(
        "--skip-convert",
        action="store_true",
        help="audit an existing prims.json in --out-dir without re-running Blender",
    )
    # The library's side artefacts (ADR 0150): master .blend + interchange FBX beside the .usdc,
    # written by whichever Blender pass ran last so all three describe one scene. The FBX is a
    # courtesy for non-USD consumers -- a scene references the .usdc, never the FBX.
    ap.add_argument(
        "--save-blend",
        action="store_true",
        help="also save the master .blend beside the exported USD (ADR 0150)",
    )
    ap.add_argument(
        "--emit-fbx",
        action="store_true",
        help="also export an interchange .fbx beside the USD -- never what a scene references",
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
            save_blend=out_dir / f"{asset.name}.blend" if args.save_blend else None,
            emit_fbx=out_dir / f"{asset.name}.fbx" if args.emit_fbx else None,
            weld_m=asset.weld_m,
            rotate_deg=asset.rotate_deg,
            cuts=_cut_dicts(asset),
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
                stale = True
            else:
                # a measurement from before AI.16 has no source objects to group by
                head = json.loads(components.read_text(encoding="utf-8"))[:1]
                stale = bool(head) and (
                    (asset.parts.granularity == "object" and "object" not in head[0])
                    or float(head[0].get("weld_m", 0.0)) != asset.weld_m
                    or tuple(head[0].get("rotate_deg", (0.0, 0.0, 0.0))) != asset.rotate_deg
                    or head[0].get("cuts", []) != _cut_dicts(asset)
                )
            if stale:
                cmd = blender_command(
                    args.blender,
                    source,
                    out_dir / "unused.usdc",
                    out_dir / "unused.prims.json",
                    asset.scale_to_metres,
                    weld_m=asset.weld_m,
                    rotate_deg=asset.rotate_deg,
                    cuts=_cut_dicts(asset),
                ) + ["--out-components", str(components)]
                print("measuring components: " + " ".join(cmd[:5]) + " ...")
                if subprocess.run(cmd, check=False).returncode != 0:
                    print("the component pass failed", file=sys.stderr)
                    return 1

            stats = json.loads(components.read_text(encoding="utf-8"))
            plan = part_assignment(stats, asset.parts)
            assignment = out_dir / f"{asset.name}.part_assignment.json"
            assignment.write_text(json.dumps(plan), encoding="utf-8")
            print(
                f"assigned {len(plan['part_of_component']):,} of {len(stats):,} components to "
                f"parts; {plan['excluded_components']:,} excluded as not the asset"
            )

            cmd = blender_split_command(
                args.blender,
                source,
                asset.scale_to_metres,
                components,
                assignment,
                out_dir.parent / name / f"{name}.usdc",
                # The part-split scene IS the cleaned master when a parts: block exists, so the
                # library's .blend/.fbx of a decomposed asset come from this pass (ADR 0150).
                save_blend=out_dir.parent / name / f"{name}.blend" if args.save_blend else None,
                emit_fbx=out_dir.parent / name / f"{name}.fbx" if args.emit_fbx else None,
                weld_m=asset.weld_m,
                rotate_deg=asset.rotate_deg,
                cuts=_cut_dicts(asset),
            )
            print("splitting geometry by part: " + " ".join(cmd[:5]) + " ...")
            if subprocess.run(cmd, check=False).returncode != 0:
                print("the part-split pass failed", file=sys.stderr)
                return 1

            # The part-split USD is what a scene renders, so it is audited too -- this is where
            # a part's asserted material (AI.14) is seen to win over its source material.
            split_prims = out_dir.parent / name / f"{name}.prims.json"
            if split_prims.exists():
                split_records = [
                    PrimRecord.from_dict(d)
                    for d in json.loads(split_prims.read_text(encoding="utf-8"))
                ]
                split_report = audit(
                    split_records, MaterialResolver(rules, names, asset=asset), args.threshold
                )
                print()
                print(f"{name} (the part-split USD):")
                print(split_report.render())
                ok = ok and split_report.passed

    # One prim per material (AI.6). Two Blender passes, like the part split: measure, plan under
    # the project interpreter where `irsim` lives, then regroup. The plan is written to disk so
    # the split is reproducible from it without re-measuring.
    if args.emit_material_split:
        if source is None:
            print("no --source and the asset config has no source_file", file=sys.stderr)
            return 1
        slots_json = out_dir / f"{asset.name}.material_slots.json"
        if not slots_json.exists() or not slots_json.with_suffix(".slots.npz").exists():
            cmd = blender_material_slots_command(
                args.blender,
                source,
                asset.scale_to_metres,
                slots_json,
                asset.weld_m,
                asset.rotate_deg,
                _cut_dicts(asset),
            )
            print("measuring material slots: " + " ".join(cmd[:5]) + " ...")
            if subprocess.run(cmd, check=False).returncode != 0:
                print("the material-slot pass failed", file=sys.stderr)
                return 1
        split_report, plan = plan_from_slots(slots_json)
        print()
        print(split_report.render())
        plan_path = out_dir / f"{asset.name}.material_split.json"
        plan_path.write_text(json.dumps(plan, indent=1), encoding="utf-8")
        name = f"{asset.name}_materials"
        cmd = blender_material_split_command(
            args.blender,
            source,
            asset.scale_to_metres,
            plan_path,
            out_dir.parent / name / f"{name}.usdc",
            weld_m=asset.weld_m,
            rotate_deg=asset.rotate_deg,
            cuts=_cut_dicts(asset),
        )
        print("splitting geometry by material: " + " ".join(cmd[:5]) + " ...")
        if subprocess.run(cmd, check=False).returncode != 0:
            print("the material-split pass failed", file=sys.stderr)
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
