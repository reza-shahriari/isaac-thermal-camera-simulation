"""Build the hidden-part placeholder FBX files from ``components/catalog.json``.

Run inside Blender::

    blender -b --factory-startup --python blender_addon/tools/make_placeholders.py [-- --force]

Each catalog entry names a ``shape`` and a ``size_m`` (see the catalog's ``about``); this writes
``components/<file>.fbx`` for every entry that has no file yet, imports it back to check it
returns at that size with no rotation, and rewrites the table in ``components/README.md``.

An existing FBX is left alone unless ``--force`` is given, because it may be a real model someone
put in place of the placeholder.
"""

import json
import math
import pathlib
import sys

import bmesh
import bpy
from mathutils import Matrix, Vector

HERE = pathlib.Path(__file__).resolve().parent
LIBRARY = HERE.parent / "irsim_thermal" / "components"
TABLE_START = "<!-- table: written by tools/make_placeholders.py -->"
TABLE_END = "<!-- end of table -->"

#: Turns a Z-axis cylinder or cone onto the named axis (the +Z end goes to +axis).
TO_AXIS = {
    "X": Matrix.Rotation(math.pi / 2, 4, "Y"),
    "Y": Matrix.Rotation(-math.pi / 2, 4, "X"),
    "Z": Matrix.Identity(4),
}
SEGMENTS = 32


def expected_size(entry: dict) -> Vector:
    """The world size (x, y, z) the placeholder of ``entry`` has."""
    s = entry["size_m"]
    if entry["shape"] == "box":
        return Vector(s)
    length, width = s[0], max(s[1:])
    axis = entry.get("axis", "X")
    return Vector([length if a == axis else width for a in "XYZ"])


def build_mesh(entry: dict) -> bpy.types.Mesh:
    me = bpy.data.meshes.new(entry["file"])
    bm = bmesh.new()
    s = entry["size_m"]
    if entry["shape"] == "box":
        bmesh.ops.create_cube(bm, size=1.0, matrix=Matrix.Diagonal((*s, 1.0)))
    elif entry["shape"] in {"cylinder", "cone"}:
        length, d0 = s[0], s[1]
        d1 = s[2] if entry["shape"] == "cone" else d0
        bmesh.ops.create_cone(
            bm,
            cap_ends=True,
            segments=SEGMENTS,
            radius1=d0 / 2,
            radius2=d1 / 2,
            depth=length,
            matrix=TO_AXIS[entry.get("axis", "X")],
        )
    else:
        raise ValueError(f"{entry['file']}: unknown shape {entry['shape']!r}")
    bm.to_mesh(me)
    bm.free()
    return me


def clear_scene() -> None:
    for ob in list(bpy.data.objects):
        bpy.data.objects.remove(ob)
    for me in list(bpy.data.meshes):
        bpy.data.meshes.remove(me)


def write(entry: dict, path: pathlib.Path) -> None:
    clear_scene()
    ob = bpy.data.objects.new(entry["file"], build_mesh(entry))
    bpy.context.scene.collection.objects.link(ob)
    ob.select_set(True)
    bpy.context.view_layer.objects.active = ob
    bpy.ops.export_scene.fbx(
        filepath=str(path),
        use_selection=True,
        object_types={"MESH"},
        apply_scale_options="FBX_SCALE_UNITS",
        bake_space_transform=False,
        add_leaf_bones=False,
        bake_anim=False,
    )


def check(entry: dict, path: pathlib.Path) -> str | None:
    """Import ``path`` back; the problem with it, or None if it has the catalog's size."""
    clear_scene()
    bpy.ops.import_scene.fbx(filepath=str(path))
    meshes = [ob for ob in bpy.data.objects if ob.type == "MESH"]
    if not meshes:
        return "no mesh"
    bpy.context.view_layer.update()
    pts = [ob.matrix_world @ Vector(c) for ob in meshes for c in ob.bound_box]
    lo = Vector([min(p[i] for p in pts) for i in range(3)])
    hi = Vector([max(p[i] for p in pts) for i in range(3)])
    size, want = hi - lo, expected_size(entry)
    if max(abs(a - b) for a, b in zip(size, want, strict=True)) > 1e-4 * max(want):
        return f"size {tuple(round(x, 4) for x in size)}, catalog says {tuple(want)}"
    if (lo + hi).length / 2 > 1e-4 * max(want):
        return f"not centred: centre {tuple(round(x, 4) for x in (lo + hi) / 2)}"
    return None


def describe(entry: dict) -> tuple[str, str]:
    s = entry["size_m"]
    if entry["shape"] == "box":
        return "box", " × ".join(f"{x:g}" for x in s)
    axis = entry.get("axis", "X")
    if entry["shape"] == "cylinder":
        return "cylinder", f"{s[0]:g} along {axis}, ⌀ {s[1]:g}"
    return "cone", f"{s[0]:g} along {axis}, ⌀ {s[1]:g} → {s[2]:g}"


def table(entries: list[dict]) -> str:
    lines = [TABLE_START]
    category = None
    for e in entries:
        if e["category"] != category:
            category = e["category"]
            lines += ["", f"**{category}**", ""]
            lines += ["| file | shape | size (m) | stands for |", "|---|---|---|---|"]
        shape, size = describe(e)
        lines.append(f"| `{e['file']}.fbx` | {shape} | {size} | {e['label']}: {e['note']} |")
    lines += ["", TABLE_END]
    return "\n".join(lines)


def main() -> int:
    force = "--force" in sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else False
    entries = json.loads((LIBRARY / "catalog.json").read_text(encoding="utf-8"))["components"]
    files = [e["file"] for e in entries]
    if len(set(files)) != len(files):
        print("FAIL duplicate file names in catalog.json")
        return 1
    failed = 0
    for e in entries:
        path = LIBRARY / f"{e['file']}.fbx"
        made = force or not path.is_file()
        if made:
            write(e, path)
        problem = check(e, path)
        if problem and made:
            failed += 1
            state = "FAIL " + problem
        elif problem:  # someone's own model in its place: theirs to size
            state = "differs from the placeholder (a real model?): " + problem
        else:
            state = "ok"
        print(f"{'wrote' if made else 'kept '} {e['file']:32s} {state}")
    readme = LIBRARY / "README.md"
    text = readme.read_text(encoding="utf-8")
    head, _, rest = text.partition(TABLE_START)
    _, _, tail = rest.partition(TABLE_END)
    readme.write_text(head + table(entries) + tail, encoding="utf-8")
    print(f"{len(entries)} components, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    if code:
        sys.exit(code)
