"""Build a demonstration of the add-on on the real DJI Phantom 4 model, headless.

    blender -b --factory-startup --python-exit-code 1 \
        --python blender_addon/demo/make_phantom4_demo.py -- --out outputs/blender_addon_demo \
        [--render] [--export-scratch /tmp/irsim_phantom_export]

Needs ``3d_models/phantom4.fbx`` (git-ignored; see ``configs/assets/phantom4.yaml``). Does, with the
add-on's own operators, what a person would do by hand:

1. import the FBX and measure it: 46 units wide, which the size check reads as centimetres;
2. *Apply scale* x0.01;
3. *Load materials from an asset*: ``configs/assets/phantom4.yaml``, the hand-written map;
4. *Check model*: coverage, mirror-like parts;
5. with ``--structure``: add the flight battery the model lacks, as a hidden part inside the
   body shell, and run the connection finder over all 41 parts and the battery;
6. switch on the thermal view and save ``phantom4_thermal_demo.blend``, which opens with the
   sidebar showing and the viewport shaded by LWIR emissivity.

``--render`` adds CPU (Cycles) pictures, framed on the aircraft: the model as it looks, then the
thermal view's grey levels in three states -- nothing assigned (all magenta), the mapping the old
global ``*metal*`` rule produced (ADR 0128: motor housings as bare aluminium, ε 0.09, so they
show as mirrors), and the hand-written map. Those grey levels are **the LWIR emissivity of the
assigned material, not a thermal image**. ``--export-scratch`` exports through the add-on into a
scratch copy of the repository and runs ``scripts/prep_asset.py`` on the result, which checks the
add-on on a 2.5-million-face asset. With ``--structure --render``, one more picture shows the shell
see-through, the hidden battery inside it and every connection found, drawn as lines.
"""

import argparse
import json
import math
import os
import pathlib
import shutil
import sys
import time

import bpy
from mathutils import Vector

HERE = pathlib.Path(__file__).resolve()
ADDON_ROOT = HERE.parents[1]
REPO = ADDON_ROOT.parent
FBX = REPO / "3d_models" / "phantom4.fbx"

#: The flight battery the model lacks. Mass: DJI's published 468 g for the Phantom 4 Pro
#: Intelligent Flight Battery. Size and heat are estimates, and say so.
BATTERY = {
    "part_name": "battery",
    "kind": "BATTERY",
    "material": "abs_plastic_white",
    "mass_kg": 0.468,
    "heat_idle_w": 0.5,
    "heat_rated_w": 25.0,
    "values_source": "ESTIMATED",
    "reference": (
        "mass 468 g: DJI Phantom 4 Pro specification. Heat estimated as I^2 R in a 4S pack of "
        "about 40 mOhm: ~10 A in hover (150 W at 15.2 V) gives 4 W, 25 A at full climb 25 W"
    ),
}
BATTERY_SIZE_M = (0.14, 0.07, 0.04)  # estimated to fit the battery bay; long side fore-aft

#: What ADR 0128 measured the old global glob doing to this asset: `*metal*` -> bare_aluminium.
OLD_GLOB_METAL = ("Metal_Matte", "_DJI_Phantom_4_ProMetal_Matte", "metal_radial")


def import_fbx(path: pathlib.Path) -> None:
    try:
        bpy.ops.wm.fbx_import(filepath=str(path))
    except AttributeError:
        bpy.ops.import_scene.fbx(filepath=str(path))


def frame_camera(parts) -> None:
    """A camera looking down at the aircraft from the front quarter, the aircraft filling it."""
    lo = Vector((math.inf,) * 3)
    hi = Vector((-math.inf,) * 3)
    for ob in parts:
        for corner in ob.bound_box:
            p = ob.matrix_world @ Vector(corner)
            lo = Vector(map(min, lo, p))
            hi = Vector(map(max, hi, p))
    centre = (lo + hi) / 2
    radius = (hi - lo).length / 2
    cam_data = bpy.data.cameras.new("demo_camera")
    cam_data.lens = 50.0
    cam = bpy.data.objects.new("demo_camera", cam_data)
    bpy.context.scene.collection.objects.link(cam)
    direction = Vector((0.9, -1.0, 0.75)).normalized()
    half_fov = math.atan(cam_data.sensor_width / (2 * cam_data.lens))
    cam.location = centre + direction * (radius / math.sin(half_fov) * 0.72)
    cam.rotation_euler = (centre - cam.location).to_track_quat("-Z", "Y").to_euler()
    bpy.context.scene.camera = cam


def render(path: pathlib.Path, samples: int) -> None:
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = samples
    scene.cycles.use_denoising = True
    scene.render.resolution_x, scene.render.resolution_y = 1280, 900
    scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)


def _world_background(colour) -> None:
    world = bpy.context.scene.world or bpy.data.worlds.new("demo_world")
    bpy.context.scene.world = world
    world.color = colour[:3]
    if world.node_tree is None and hasattr(world, "use_nodes"):
        world.use_nodes = True
    bg = world.node_tree.nodes.get("Background") if world.node_tree else None
    if bg is not None:
        bg.inputs["Color"].default_value = colour
        bg.inputs["Strength"].default_value = 1.0


def light_for_look() -> None:
    _world_background((0.55, 0.62, 0.72, 1.0))
    sun = bpy.data.objects.new("demo_sun", bpy.data.lights.new("demo_sun", "SUN"))
    sun.data.energy = 3.0
    sun.rotation_euler = (math.radians(40), 0.0, math.radians(30))
    bpy.context.scene.collection.objects.link(sun)


def emissivity_shading(eps: dict[str, float]) -> None:
    """Every material becomes a flat emission shown at its LWIR ε (for the pictures only).

    The same mapping as the add-on's thermal view: brightness on screen equals ε, unassigned is
    magenta. With the Standard view transform and no lights, what is rendered is exactly that.
    """
    from irsim_thermal.thermal_view import display_grey

    for mat in bpy.data.materials:
        tree = mat.node_tree
        if tree is None:
            continue
        tree.nodes.clear()
        out = tree.nodes.new("ShaderNodeOutputMaterial")
        emit = tree.nodes.new("ShaderNodeEmission")
        thermal = mat.irsim_material
        if thermal in eps:
            v = display_grey(eps[thermal])
            emit.inputs["Color"].default_value = (v, v, v, 1.0)
        else:
            emit.inputs["Color"].default_value = (1.0, 0.0, 1.0, 1.0)
        tree.links.new(emit.outputs["Emission"], out.inputs["Surface"])
    _world_background((0.02, 0.02, 0.02, 1.0))
    for ob in bpy.context.scene.objects:
        if ob.type == "LIGHT":
            ob.hide_render = True
    bpy.context.scene.view_settings.view_transform = "Standard"


def open_viewports_on_the_thermal_view(parts) -> None:
    """So the saved file opens shaded by emissivity with the sidebar out, framed on the model."""
    centre = sum((ob.matrix_world.translation for ob in parts), Vector()) / max(1, len(parts))
    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type != "VIEW_3D":
                continue
            for space in area.spaces:
                if space.type != "VIEW_3D":
                    continue
                space.shading.type = "SOLID"
                space.shading.color_type = "MATERIAL"
                space.show_region_ui = True
                space.region_3d.view_location = centre
                space.region_3d.view_distance = 1.2


def add_structure(ctx, parts, summary: dict) -> None:
    """The battery inside the body, then the connection finder over everything."""
    from irsim_thermal import hidden_parts, scene_stats

    body = max(parts, key=lambda ob: scene_stats.part_stats(ob).total_area_m2)
    centre, _ = hidden_parts.world_box(body)
    # Fore-aft is towards the camera gimbal; the battery slides in from the back along it.
    gimbal = next(
        ob
        for ob in parts
        if any(s.material and "camera" in s.material.name for s in ob.material_slots)
    )
    ahead = hidden_parts.world_box(gimbal)[0] - centre
    long_x = abs(ahead.x) >= abs(ahead.y)
    summary["forward_axis"] = "x" if long_x else "y"
    box = BATTERY_SIZE_M if long_x else (BATTERY_SIZE_M[1], BATTERY_SIZE_M[0], BATTERY_SIZE_M[2])
    for ob in ctx.selected_objects:
        ob.select_set(False)
    body.select_set(True)
    ctx.view_layer.objects.active = body
    bpy.ops.irsim.add_hidden_part(size=box, **BATTERY)
    summary["battery_inside"] = body.name
    t0 = time.perf_counter()
    bpy.ops.irsim.find_connections()
    summary["find_connections_s"] = round(time.perf_counter() - t0, 1)
    found = list(ctx.scene.irsim_connections)
    summary["contacts"] = sum(1 for c in found if c.kind == "CONTACT")
    summary["facing_pairs"] = sum(1 for c in found if c.kind == "FACING")
    summary["battery_connections"] = sorted(
        f"{c.kind.lower()} {c.b.name if c.a.name == 'battery' else c.a.name} "
        f"{c.area_m2 * 1e4:.3g} cm2"
        for c in found
        if "battery" in (c.a.name, c.b.name)
    )
    ctx.scene.irsim.show_connections = True


def _emission(name: str, colour, strength: float = 1.0) -> bpy.types.Material:
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    tree = mat.node_tree
    tree.nodes.clear()
    out = tree.nodes.new("ShaderNodeOutputMaterial")
    emit = tree.nodes.new("ShaderNodeEmission")
    emit.inputs["Color"].default_value = colour
    emit.inputs["Strength"].default_value = strength
    tree.links.new(emit.outputs["Emission"], out.inputs["Surface"])
    return mat


def _ghost(mat: bpy.types.Material) -> None:
    """Make a material see-through, keeping a hint of its colour."""
    tree = mat.node_tree
    if tree is None:
        return
    tree.nodes.clear()
    out = tree.nodes.new("ShaderNodeOutputMaterial")
    mix = tree.nodes.new("ShaderNodeMixShader")
    clear = tree.nodes.new("ShaderNodeBsdfTransparent")
    diffuse = tree.nodes.new("ShaderNodeBsdfDiffuse")
    diffuse.inputs["Color"].default_value = (0.8, 0.8, 0.82, 1.0)
    mix.inputs["Fac"].default_value = 0.22
    tree.links.new(clear.outputs[0], mix.inputs[1])
    tree.links.new(diffuse.outputs[0], mix.inputs[2])
    tree.links.new(mix.outputs[0], out.inputs["Surface"])


def render_structure(ctx, out: pathlib.Path, samples: int) -> None:
    """The shell see-through, the battery solid orange, contacts orange and facing pairs blue."""
    from irsim_thermal import overlay

    battery = ctx.scene.objects["battery"]
    for mat in bpy.data.materials:
        _ghost(mat)
    solid = battery.copy()
    solid.data = battery.data.copy()
    solid.hide_render = False
    solid.irsim_component.is_hidden_part = False
    solid.data.materials.clear()
    solid.data.materials.append(_emission("demo_battery", (1.0, 0.35, 0.05, 1.0), 2.0))
    ctx.collection.objects.link(solid)
    radius = 0.0008
    # No connection is selected in the picture: every contact is its cross, none its lines.
    ctx.scene.irsim.active_connection_index = -1
    lines = []
    for points, colour, _ in overlay.segments(ctx):
        curve = bpy.data.curves.new("demo_line", "CURVE")
        curve.dimensions = "3D"
        curve.bevel_depth = radius
        for i in range(0, len(points), 2):
            spline = curve.splines.new("POLY")
            spline.points.add(1)
            for j, p in enumerate(points[i : i + 2]):
                spline.points[j].co = (*p, 1.0)
        curve.materials.append(_emission("demo_line", (*colour[:3], 1.0), 1.5))
        ob = bpy.data.objects.new("demo_line", curve)
        ctx.collection.objects.link(ob)
        lines.append(ob)
    render(out / "phantom4_structure.png", samples)
    for ob in [solid, *lines]:
        bpy.data.objects.remove(ob)


def scratch_repo(root: pathlib.Path) -> pathlib.Path:
    if root.exists():
        shutil.rmtree(root)
    (root / "configs").mkdir(parents=True)
    for name in ("src", "data", "scripts"):
        (root / name).symlink_to(REPO / name)
    shutil.copytree(REPO / "configs" / "materials", root / "configs" / "materials")
    shutil.copytree(REPO / "configs" / "assets", root / "configs" / "assets")
    return root


def main() -> None:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--samples", type=int, default=32)
    ap.add_argument("--export-scratch", type=pathlib.Path, default=None)
    ap.add_argument("--structure", action="store_true", help="add the battery, find connections")
    args = ap.parse_args(argv)
    if not FBX.exists():
        raise SystemExit(f"{FBX} is not here (it is git-ignored local data)")
    out = args.out if args.out.is_absolute() else REPO / args.out
    out.mkdir(parents=True, exist_ok=True)

    if args.export_scratch is not None:
        os.environ["IRSIM_REPO"] = str(scratch_repo(args.export_scratch / "repo"))
    sys.path.insert(0, str(ADDON_ROOT))
    bpy.ops.wm.read_factory_settings(use_empty=True)
    import irsim_thermal

    irsim_thermal.register()
    from irsim_thermal import library_state, scene_stats, sizing, thermal_view

    ctx = bpy.context
    summary: dict = {}
    t0 = time.perf_counter()
    import_fbx(FBX)
    summary["import_s"] = round(time.perf_counter() - t0, 2)
    library_state.load(ctx)
    parts = scene_stats.part_objects(ctx)
    summary["parts"] = len(parts)
    summary["faces"] = sum(len(ob.data.polygons) for ob in parts)

    t0 = time.perf_counter()
    scene_stats.refresh(ctx)
    summary["check_s"] = round(time.perf_counter() - t0, 3)
    s = ctx.scene.irsim
    size = (s.size_x_m, s.size_y_m, s.size_z_m)
    summary["size_as_imported"] = [round(v, 3) for v in size]
    diagnosis = sizing.diagnose(max(size), {r.key: r for r in sizing.load_ranges()}["multirotor"])
    summary["size_diagnosis"] = diagnosis.message
    if diagnosis.factor:
        bpy.ops.irsim.apply_scale(factor=diagnosis.factor)
    summary["size_m"] = [round(v, 4) for v in (s.size_x_m, s.size_y_m, s.size_z_m)]
    summary["size_phrase"] = sizing.comparison(max(s.size_x_m, s.size_y_m, s.size_z_m))

    bpy.ops.irsim.load_asset_map("EXEC_DEFAULT", asset="phantom4")
    result = scene_stats.refresh(ctx)
    summary["coverage_percent"] = round(100 * result.coverage, 3)
    summary["unassigned_parts"] = [p.name for p in result.unassigned_parts]
    summary["mirror_like"] = sorted({f"{t} (eps {e:.2f})" for _, t, e in result.mirror_parts})
    summary["mirror_like_parts"] = len(result.mirror_parts)
    summary["materials_assigned"] = sorted(
        {f"{m.name} -> {m.irsim_material}" for m in scene_stats.used_materials(parts)}
    )

    if args.structure:
        add_structure(ctx, parts, summary)

    ctx.scene.irsim.thermal_view = True
    thermal_view.apply(ctx)
    open_viewports_on_the_thermal_view(parts)
    blend = out / "phantom4_thermal_demo.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(blend), check_existing=False)
    summary["blend"] = str(blend.relative_to(REPO)) if blend.is_relative_to(REPO) else str(blend)

    if args.export_scratch is not None:
        from irsim_thermal import export
        from irsim_thermal.bridge_client import run_script

        ctx.scene.irsim.asset_name = "phantom4_addon_demo"
        t0 = time.perf_counter()
        bpy.ops.irsim.export_asset()
        summary["export_s"] = round(time.perf_counter() - t0, 2)
        repo = os.environ["IRSIM_REPO"]
        python = irsim_thermal.prefs.settings(ctx)[1]
        argv_audit = export.audit_argv(
            repo,
            "phantom4_addon_demo",
            export._blender_executable(),
            out_dir=str(args.export_scratch / "audit"),
        )
        status, text = run_script(python, repo, argv_audit, timeout_s=1800)
        summary["prep_asset_exit"] = status
        summary["prep_asset_audit"] = [ln for ln in text.splitlines() if "audit:" in ln][-2:]

    if args.render:
        # Pictures only, after the .blend is saved: the shading below replaces the materials.
        eps = library_state.emissivity_lwir(ctx)
        ctx.scene.irsim.thermal_view = False
        mapped = {m.name: m.irsim_material for m in bpy.data.materials}
        frame_camera(parts)
        light_for_look()
        t0 = time.perf_counter()
        render(out / "phantom4_look.png", args.samples)
        if args.structure:
            render_structure(ctx, out, args.samples)
        states = {
            "1_unassigned": {},
            "2_old_metal_glob": {
                **mapped,
                **{name: "bare_aluminium" for name in OLD_GLOB_METAL if name in mapped},
            },
            "3_hand_written_map": mapped,
        }
        for label, assignment in states.items():
            for mat in bpy.data.materials:
                mat.irsim_material = assignment.get(mat.name, "")
            emissivity_shading(eps)
            render(out / f"phantom4_emissivity_{label}.png", 4)
        summary["render_s"] = round(time.perf_counter() - t0, 1)

    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))


main()
