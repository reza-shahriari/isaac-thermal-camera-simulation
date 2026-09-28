"""End-to-end check of the add-on inside a real (headless) Blender, against a scratch repository.

Run with ``blender_addon/tests/run_blender_smoke.sh`` (or by hand)::

    blender -b --factory-startup --python-exit-code 1 \
        --python blender_addon/tests/blender_smoke.py -- --scratch /tmp/irsim_smoke

Nothing is written into the real repository: a scratch "repository" is assembled whose ``src``,
``data`` and ``scripts`` are links to the real ones and whose ``configs/materials`` and
``configs/assets`` are copies, and ``$IRSIM_REPO`` points the add-on at it.

What it proves, in the order a person would meet it:

1. the library loads through the bridge, with the values the solver uses;
2. assigning a part never changes an unselected part that shares its Blender material, and a
   part that shares its *mesh* with another (a linked duplicate) is handled the same way;
3. assigning faces in Edit Mode moves only those faces;
4. join and separate -- ordinary modelling -- keep every face's thermal material, measured as
   area per thermal material before and after;
5. the thermal view recolours only viewport colours and restores them exactly;
6. the size check and *Apply scale* turn a centimetre model into metres;
7. a new material is checked by irsim and written; a duplicate is refused;
8. the export writes USD + .blend + asset YAML whose names agree, refuses to replace an export
   unless asked and never replaces a hand-written config;
9. ``scripts/prep_asset.py`` -- the project's own asset audit -- passes the exported asset.
"""

import argparse
import math
import os
import pathlib
import shutil
import sys

import bpy

HERE = pathlib.Path(__file__).resolve()
ADDON_ROOT = HERE.parents[1]
REPO = ADDON_ROOT.parent
ISAAC = pathlib.Path("~/IsaacSim/_build/linux-x86_64/release/python.sh").expanduser()

failures: list[str] = []


def check(ok: bool, what: str) -> None:
    print(("PASS " if ok else "FAIL ") + what, flush=True)
    if not ok:
        failures.append(what)


def close(a: float, b: float, rel: float = 1e-6) -> bool:
    return abs(a - b) <= rel * max(1.0, abs(a), abs(b))


def scratch_repo(root: pathlib.Path) -> pathlib.Path:
    if root.exists():
        shutil.rmtree(root)
    (root / "configs").mkdir(parents=True)
    for name in ("src", "data", "scripts"):
        (root / name).symlink_to(REPO / name)
    shutil.copytree(REPO / "configs" / "materials", root / "configs" / "materials")
    (root / "configs" / "assets").mkdir()
    return root


def select_only(*objects) -> None:
    for ob in bpy.context.view_layer.objects:
        ob.select_set(False)
    for ob in objects:
        ob.select_set(True)
    if objects:
        bpy.context.view_layer.objects.active = objects[0]


def new_material(name: str, colour=(0.8, 0.8, 0.8, 1.0)) -> bpy.types.Material:
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = colour
    return mat


def add(kind: str, name: str, location=(0.0, 0.0, 0.0), **kw) -> bpy.types.Object:
    ops = {
        "cube": bpy.ops.mesh.primitive_cube_add,
        "plane": bpy.ops.mesh.primitive_plane_add,
        "cylinder": bpy.ops.mesh.primitive_cylinder_add,
        "sphere": bpy.ops.mesh.primitive_uv_sphere_add,
    }
    ops[kind](location=location, **kw)
    ob = bpy.context.active_object
    ob.name = name
    ob.data.name = name
    return ob


def by_thermal(stats_module) -> dict[str, float]:
    totals: dict[str, float] = {}
    for part in stats_module.summary(bpy.context).parts:
        for thermal, area in part.by_thermal_m2.items():
            totals[thermal] = totals.get(thermal, 0.0) + area
    return totals


UI_FUNCS = None


class FakeLayout:
    """Stands in for ``bpy.types.UILayout`` headless, checking every call against Blender's RNA.

    A panel's ``draw`` cannot run without a window, and a typo in it (a keyword that
    ``template_list`` does not take, a property that does not exist) only shows up as a red
    traceback in the sidebar. Each call here is checked against the parameter list Blender itself
    publishes for that function, each ``prop`` against the data's properties, and each operator's
    assigned properties against that operator.
    """

    CONTAINERS = {"row", "column", "box", "grid_flow", "split", "column_flow"}

    def __init__(self, calls: list) -> None:
        self.calls = calls
        self.alert = False
        self.enabled = True
        self.active = True

    def __getattr__(self, name: str):
        fn = UI_FUNCS.get(name)
        if fn is None:
            raise AttributeError(f"UILayout has no function {name!r}")
        params = [p.identifier for p in fn.parameters]

        def call(*args, **kwargs):
            unknown = set(kwargs) - set(params)
            if unknown:
                raise TypeError(f"UILayout.{name}() does not take {sorted(unknown)}")
            if len(args) > len(params):
                raise TypeError(f"UILayout.{name}() takes {len(params)} arguments")
            self.calls.append(name)
            if name == "prop":
                data, prop = args[0], args[1]
                if prop not in data.bl_rna.properties:
                    raise AttributeError(f"{type(data).__name__} has no property {prop!r}")
            if name == "template_list":
                for data, prop in ((args[2], args[3]), (args[4], args[5])):
                    if prop not in data.bl_rna.properties:
                        raise AttributeError(f"template_list: no property {prop!r}")
            if name in self.CONTAINERS:
                return FakeLayout(self.calls)
            if name == "operator":
                return FakeOperator(args[0] if args else kwargs["operator"])
            return None

        return call


class FakeOperator:
    def __init__(self, idname: str) -> None:
        module, name = idname.split(".")
        rna = getattr(getattr(bpy.ops, module), name).get_rna_type()
        object.__setattr__(self, "_props", {p.identifier for p in rna.properties})

    def __setattr__(self, key: str, value) -> None:
        if key not in self._props:
            raise AttributeError(f"operator has no property {key!r}")


def ui_checks(ctx) -> None:
    global UI_FUNCS
    import types

    from irsim_thermal import material_form, prefs, ui

    UI_FUNCS = {f.identifier: f for f in bpy.types.UILayout.bl_rna.functions}

    def fake(**extra):
        calls: list = []
        return types.SimpleNamespace(layout=FakeLayout(calls), **extra), calls

    panels = (ui.MainPanel, ui.PartsPanel, ui.LibraryPanel, ui.ExportPanel)
    helpers = {"_details": ui.LibraryPanel._details, "_size": ui.ExportPanel._size}
    s = ctx.scene.irsim
    s.size_help, s.size_kind, s.known_dimension_m = True, "multirotor", 0.0
    for state in ("loaded", "known dimension", "empty library"):
        if state == "known dimension":
            s.known_dimension_m = 0.35
        if state == "empty library":
            ctx.window_manager.irsim_library.clear()
        for panel in panels:
            try:
                me, calls = fake(**helpers)
                panel.draw(me, ctx)
                check(bool(calls), f"{panel.__name__} draws ({state}, {len(calls)} calls)")
            except Exception as exc:  # noqa: BLE001 -- report every panel, not the first
                check(False, f"{panel.__name__} draws ({state}): {type(exc).__name__}: {exc}")
    from irsim_thermal import library_state

    library_state.load(ctx)

    me, calls = fake()
    ui.PartsList.draw_item(me, ctx, me.layout, ctx.scene, ctx.scene.objects[0], 0, None, "", 0)
    lib = ctx.window_manager.irsim_library
    ui.LibraryList.draw_item(me, ctx, me.layout, ctx.window_manager, lib[0], 0, None, "", 0)
    if len(s.issues):
        me2, _ = fake(ICONS=ui.IssuesList.ICONS)
        ui.IssuesList.draw_item(me2, ctx, me2.layout, s, s.issues[0], 0, None, "", 0)
    check(True, "the three lists draw their rows")

    lister = types.SimpleNamespace(
        filter_name="", bitflag_filter_item=1 << 30, use_filter_sort_alpha=True
    )
    flags, order = ui.PartsList.filter_items(lister, ctx, ctx.scene, "objects")
    meshes = sum(1 for ob in ctx.scene.objects if ob.type == "MESH" and not ob.hide_render)
    check(
        sum(1 for f in flags if f) == meshes and len(order) == len(flags),
        "the parts list shows exactly the renderable meshes",
    )

    rna = bpy.ops.irsim.new_material.get_rna_type()
    props = {p.identifier: getattr(p, "default", None) for p in rna.properties}
    props["name"] = "Bad Name"
    props["bl_rna"] = rna
    me, calls = fake(**props)
    material_form.NewMaterial.draw(me, ctx)
    check("prop" in calls, f"the new-material form draws ({len(calls)} calls)")

    from irsim_thermal import asset_map

    rna = bpy.ops.irsim.load_asset_map.get_rna_type()
    me, calls = fake(asset="", overwrite=False, bl_rna=rna)
    asset_map.LoadAssetMap.draw(me, ctx)
    check("prop_search" in calls, "the load-from-asset dialog draws")

    me, calls = fake(repo_root="", python_path="", bl_rna=prefs.IrsimPreferences.bl_rna)
    prefs.IrsimPreferences.draw(me, ctx)
    check("operator" in calls, "the preferences draw")


def main() -> None:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--scratch", type=pathlib.Path, required=True)
    ap.add_argument("--skip-audit", action="store_true")
    args = ap.parse_args(argv)

    repo = scratch_repo(args.scratch / "repo")
    python = os.environ.get("IRSIM_PYTHON") or (str(ISAAC) if ISAAC.is_file() else "python3")
    os.environ["IRSIM_REPO"] = str(repo)
    os.environ["IRSIM_PYTHON"] = python

    sys.path.insert(0, str(ADDON_ROOT))
    bpy.ops.wm.read_factory_settings(use_empty=True)
    import irsim_thermal

    irsim_thermal.register()
    from irsim_thermal import export, library_state, prefs, scene_stats, sizing, thermal_view

    ctx = bpy.context
    check(prefs.settings(ctx) == (str(repo), python), "settings come from the environment")

    # --- 1. the library ---------------------------------------------------------------------
    n = library_state.load(ctx)
    cf = library_state.find(ctx, "carbon_fibre")
    alu = library_state.find(ctx, "bare_aluminium")
    check(n >= 20, f"library loaded ({n} materials)")
    check(cf is not None and close(cf.eps_lwir, 0.90), "carbon_fibre LWIR emissivity is 0.90")
    check(alu is not None and close(alu.eps_lwir, 0.09), "bare_aluminium LWIR emissivity is 0.09")
    check(
        all(
            close(
                cf.__getattribute__(f"eps_{b}")
                + cf.__getattribute__(f"rho_{b}")
                + cf.__getattribute__(f"tau_{b}"),
                1.0,
            )
            for b in ("nir", "swir", "mwir", "lwir")
        ),
        "the library as shown closes in every band",
    )

    # --- a toy quadcopter, modelled in centimetres like the Phantom 4 FBX ----------------------
    white = new_material("white_plastic", (0.9, 0.9, 0.9, 1.0))
    metal = new_material("Metal Matte", (0.3, 0.3, 0.3, 1.0))
    chrome = new_material("chrome", (0.95, 0.95, 0.95, 1.0))
    decal = new_material("decal", (0.2, 0.4, 0.8, 1.0))

    shell = add("cube", "shell", size=20.0)
    shell.data.materials.append(white)
    arm = add(
        "cylinder", "arm_left", location=(-25.0, 0.0, 0.0), radius=1.5, depth=30.0, vertices=16
    )
    arm.data.materials.append(white)
    arm.data.materials.append(metal)
    for poly in arm.data.polygons:
        poly.material_index = 1 if poly.normal.z > 0.5 else 0  # the top cap is metal
    prop_a = add("plane", "propeller_a", location=(0.0, 0.0, 12.0), size=25.0)
    prop_a.data.materials.append(white)
    prop_b = prop_a.copy()  # a linked duplicate: same mesh data
    prop_b.name = "propeller_b"
    prop_b.location = (0.0, 0.0, -12.0)
    ctx.scene.collection.objects.link(prop_b)
    bare = add(
        "sphere", "no_material", location=(25.0, 0.0, 0.0), radius=3.0, segments=16, ring_count=8
    )
    trim = add("cube", "trim_ring", location=(0.0, 12.0, 0.0), size=2.0)
    trim.data.materials.append(chrome)
    cube = add("cube", "Cube", location=(0.0, -12.0, 0.0), size=2.0)
    cube.data.materials.append(decal)
    bpy.context.view_layer.update()

    # --- 5a. thermal view on an unassigned model: everything magenta, then restored ---------------
    originals = {m.name: tuple(m.diffuse_color) for m in bpy.data.materials}
    ctx.scene.irsim.thermal_view = True
    check(
        all(tuple(m.diffuse_color) == thermal_view.UNASSIGNED for m in (white, metal, chrome)),
        "thermal view shows unassigned materials in magenta",
    )
    ctx.scene.irsim.thermal_view = False
    check(
        all(tuple(m.diffuse_color) == originals[m.name] for m in bpy.data.materials),
        "thermal view restores every viewport colour exactly",
    )

    # --- 2. parts ------------------------------------------------------------------------------
    select_only(prop_a)
    bpy.ops.irsim.assign(material="carbon_fibre", scope="PARTS")
    slot_a = prop_a.material_slots[0]
    check(
        slot_a.material.name == "white_plastic__carbon_fibre"
        and slot_a.material.irsim_material == "carbon_fibre"
        and slot_a.material.irsim_origin == "white_plastic",
        "a part sharing its material gets its own copy, named after both",
    )
    check(slot_a.link == "OBJECT", "a linked duplicate's slot is switched to object level")
    check(
        prop_b.material_slots[0].material == white and white.irsim_material == "",
        "the unselected linked duplicate is untouched",
    )
    check(shell.material_slots[0].material == white, "the unselected shell is untouched")

    select_only(shell)
    bpy.ops.irsim.assign(material="abs_plastic_white", scope="PARTS")
    check(
        shell.material_slots[0].material.name == "white_plastic__abs_plastic_white",
        "the shell gets its own copy while others still share the original",
    )

    select_only(prop_b, arm)
    bpy.ops.irsim.assign(material="abs_plastic_white", scope="PARTS")
    check(
        white.irsim_material == "abs_plastic_white",
        "when every user of a material is selected it is assigned in place",
    )
    check(metal.irsim_material == "abs_plastic_white", "every slot of a selected part is assigned")

    # --- 3. faces in Edit Mode ---------------------------------------------------------------
    import bmesh

    select_only(arm)
    bpy.ops.object.mode_set(mode="EDIT")
    bm = bmesh.from_edit_mesh(arm.data)
    for f in bm.faces:
        f.select = f.material_index == 1
    bmesh.update_edit_mesh(arm.data)
    bpy.ops.irsim.assign(material="aircraft_aluminium_painted", scope="FACES")
    check(
        metal.irsim_material == "aircraft_aluminium_painted",
        "a whole slot used by one part only is assigned in place",
    )
    bm = bmesh.from_edit_mesh(arm.data)
    side = [f for f in bm.faces if f.material_index == 0 and abs(f.normal.z) < 0.1]
    for f in bm.faces:
        f.select = False
    side[0].select = True
    bmesh.update_edit_mesh(arm.data)
    bpy.ops.irsim.assign(material="carbon_fibre", scope="FACES")
    bm = bmesh.from_edit_mesh(arm.data)
    moved = [
        f
        for f in bm.faces
        if arm.material_slots[f.material_index].material.irsim_material == "carbon_fibre"
    ]
    check(len(moved) == 1, "exactly the one selected face changed material")
    check(
        arm.material_slots[moved[0].material_index].material.name == "white_plastic__carbon_fibre",
        "the earlier copy is reused, not duplicated",
    )
    bpy.ops.object.mode_set(mode="OBJECT")

    select_only(bare)
    bpy.ops.irsim.assign(material="abs_plastic_white", scope="PARTS")
    check(
        bare.material_slots[0].material.name == "irsim_abs_plastic_white",
        "a part with no material gets a plain one carrying the thermal material",
    )
    select_only(trim)
    bpy.ops.irsim.assign(material="bare_aluminium", scope="PARTS")
    select_only(cube)
    bpy.ops.irsim.assign(material="painted_composite", scope="PARTS")

    summary = scene_stats.refresh(ctx)
    kinds = {(i.kind, i.name) for i in ctx.scene.irsim.issues}
    check(close(summary.coverage, 1.0), f"coverage is 100 % ({100 * summary.coverage:.2f} %)")
    check(("MIRROR", "trim_ring") in kinds, "the bare aluminium trim is flagged as mirror-like")
    check(("NAME", "Cube") in kinds, "a default-named part is flagged")
    check(not any(k == "UNASSIGNED" for k, _ in kinds), "no part is left unassigned")
    before = by_thermal(scene_stats)
    face_area = 2 * 1.5 * math.sin(math.pi / 16) * 30.0  # one side face: chord x depth
    expected_cf = 25.0 * 25.0 + face_area
    check(
        close(before["carbon_fibre"], expected_cf, 1e-5),
        f"carbon fibre area is one propeller plus one arm face ({before['carbon_fibre']:.2f} "
        f"vs {expected_cf:.2f})",
    )

    # --- 5b. thermal view on the assigned model -------------------------------------------------
    ctx.scene.irsim.thermal_view = True
    abs_eps = library_state.find(ctx, "abs_plastic_white").eps_lwir
    check(
        close(white.diffuse_color[0], thermal_view.display_grey(abs_eps), 1e-5),
        "thermal view grey level is LWIR ε",
    )
    check(
        thermal_view.display_grey(0.09) < 0.01 < 0.5 < thermal_view.display_grey(0.9),
        "a mirror-like ε displays dark, an emitter bright",
    )
    ctx.scene.irsim.thermal_view = False
    check(
        tuple(white.diffuse_color) == originals["white_plastic"],
        "and the look comes back afterwards",
    )

    # --- 4. join and separate keep every face's thermal material ----------------------------------
    select_only(arm, prop_b)
    bpy.ops.object.join()
    after_join = by_thermal(scene_stats)
    check(
        set(after_join) == set(before)
        and all(close(after_join[k], before[k], 1e-6) for k in before),
        "join keeps the area of every thermal material",
    )
    select_only(arm)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.separate(type="MATERIAL")
    bpy.ops.object.mode_set(mode="OBJECT")
    after_split = by_thermal(scene_stats)
    check(
        all(close(after_split[k], before[k], 1e-6) for k in before),
        "separate by material keeps the area of every thermal material",
    )
    check(close(scene_stats.summary(ctx).coverage, 1.0), "and coverage stays 100 %")

    # --- 6. size ---------------------------------------------------------------------------------
    scene_stats.refresh(ctx)
    s = ctx.scene.irsim
    largest = max(s.size_x_m, s.size_y_m, s.size_z_m)
    ranges = {r.key: r for r in sizing.load_ranges()}
    diagnosis = sizing.diagnose(largest, ranges["multirotor"])
    check(
        diagnosis.status == "unit" and close(diagnosis.factor, 0.01),
        f"a {largest:.1f}-unit drone is recognised as centimetres",
    )
    dims_before = (s.size_x_m, s.size_y_m, s.size_z_m)
    bpy.ops.irsim.apply_scale(factor=0.01)
    dims_after = (s.size_x_m, s.size_y_m, s.size_z_m)
    check(
        all(close(a, 0.01 * b, 1e-5) for a, b in zip(dims_after, dims_before, strict=True)),
        f"apply scale gives metres ({dims_after[0]:.3f} x {dims_after[1]:.3f} x "
        f"{dims_after[2]:.3f} m)",
    )
    check(
        all(tuple(ob.scale) == (1.0, 1.0, 1.0) for ob in scene_stats.part_objects(ctx)),
        "and the scale is applied into the mesh data",
    )

    # --- 7. a new material ----------------------------------------------------------------------
    result = bpy.ops.irsim.new_material(
        "EXEC_DEFAULT",
        name="nylon_black_smoke",
        about="Black moulded nylon, smoke test",
        reference="estimated from similar polymers",
        source="estimated",
        eps_lwir=0.93,
        tau_swir=0.0,
        assign_after=False,
    )
    written = repo / "configs" / "materials" / "nylon_black_smoke.yaml"
    check(result == {"FINISHED"} and written.exists(), "a new material is written")
    check(library_state.find(ctx, "nylon_black_smoke") is not None, "and appears in the library")
    check(
        (REPO / "configs" / "materials" / "nylon_black_smoke.yaml").exists() is False,
        "the real library is untouched",
    )
    try:
        bpy.ops.irsim.new_material("EXEC_DEFAULT", name="nylon_black_smoke", assign_after=False)
        check(False, "a duplicate name is refused")
    except RuntimeError as exc:
        check("already exists" in str(exc), "a duplicate name is refused")
    try:
        bpy.ops.irsim.new_material(
            "EXEC_DEFAULT", name="leaky_glass", eps_swir=0.5, tau_swir=0.6, assign_after=False
        )
        check(False, "ε + τ > 1 is refused")
    except RuntimeError as exc:
        check("closure" in str(exc).lower(), "ε + τ > 1 is refused by irsim's own check")
    check(
        not (repo / "configs" / "materials" / "leaky_glass.yaml").exists(),
        "and nothing was written for it",
    )

    # --- 8. export -------------------------------------------------------------------------------
    ctx.scene.irsim.asset_name = "smoke_quad"
    bpy.ops.irsim.export_asset()
    usd = repo / "3d_models" / "smoke_quad" / "smoke_quad.usdc"
    blend = repo / "3d_models" / "smoke_quad" / "smoke_quad.blend"
    yaml_path = repo / "configs" / "assets" / "smoke_quad.yaml"
    check(usd.exists() and blend.exists() and yaml_path.exists(), "USD, .blend and YAML written")
    check(
        bpy.data.materials.get("Metal_Matte") is not None
        and bpy.data.materials.get("Metal Matte") is None,
        "an unsafe material name was renamed for USD",
    )
    text = yaml_path.read_text(encoding="utf-8")
    check("Metal_Matte: aircraft_aluminium_painted" in text, "the map uses the renamed name")
    bound = export._bound_material_names(usd)
    keys = {
        line.split(":")[0].strip().lower() for line in text.splitlines() if line.startswith("    ")
    }
    check(
        bound and all(b.lower() in keys for b in bound),
        f"every material bound in the USD is in the map ({sorted(bound)})",
    )
    check(bpy.data.filepath == "", "saving the .blend copy did not change the open file")
    try:
        bpy.ops.irsim.export_asset()
        check(False, "a second export is refused without 'replace'")
    except RuntimeError as exc:
        check("exists" in str(exc), "a second export is refused without 'replace'")
    ctx.scene.irsim.replace_export = True
    check(bpy.ops.irsim.export_asset() == {"FINISHED"}, "and allowed with it")
    handmade = repo / "configs" / "assets" / "handmade.yaml"
    handmade.write_text("# a hand-written map, with its reasons\nschema_version: 1\n")
    ctx.scene.irsim.asset_name = "handmade"
    try:
        bpy.ops.irsim.export_asset()
        check(False, "a hand-written config is never replaced")
    except RuntimeError as exc:
        check("by hand" in str(exc), "a hand-written config is never replaced")
    check(handmade.read_text().startswith("# a hand-written map"), "and is unchanged")
    check(
        not (repo / "3d_models" / "handmade").exists(),
        "and no geometry was written for the refused export",
    )
    downloaded = repo / "3d_models" / "downloaded"
    downloaded.mkdir(parents=True)
    (downloaded / "downloaded.usdc").write_bytes(b"not ours")
    ctx.scene.irsim.asset_name = "downloaded"
    try:
        bpy.ops.irsim.export_asset()
        check(False, "files the add-on did not write are never overwritten")
    except RuntimeError as exc:
        check(
            "no export of this add-on wrote" in str(exc),
            "files the add-on did not write are never overwritten",
        )
    check((downloaded / "downloaded.usdc").read_bytes() == b"not ours", "and are unchanged")
    ctx.scene.irsim.asset_name = "smoke_quad"

    # --- 8b. reopen: an asset config's map puts the assignments back ------------------------------
    for mat in bpy.data.materials:
        mat.irsim_material = ""
    check(scene_stats.summary(ctx).coverage == 0.0, "every assignment cleared")
    chrome_mat = bpy.data.materials["chrome"]
    chrome_mat.irsim_material = "carbon_fibre"  # a deliberate local change
    bpy.ops.irsim.load_asset_map("EXEC_DEFAULT", asset="smoke_quad")
    check(
        chrome_mat.irsim_material == "carbon_fibre",
        "loading a map keeps an existing assignment by default",
    )
    check(close(scene_stats.summary(ctx).coverage, 1.0), "and restores everything else")
    bpy.ops.irsim.load_asset_map("EXEC_DEFAULT", asset="smoke_quad", overwrite=True)
    check(chrome_mat.irsim_material == "bare_aluminium", "'Replace' takes the map's value")
    try:
        bpy.ops.irsim.load_asset_map("EXEC_DEFAULT", asset="nothing_here")
        check(False, "a missing asset is reported")
    except RuntimeError as exc:
        check("no asset config" in str(exc), "a missing asset is reported")

    # --- 10. every panel draws, and every layout call is one Blender has -------------------------
    ui_checks(ctx)

    # --- 9. the project's own audit --------------------------------------------------------------
    if not args.skip_audit:
        from irsim_thermal.bridge_client import run_script

        out_dir = args.scratch / "audit"
        argv = export.audit_argv(
            str(repo), "smoke_quad", export._blender_executable(), out_dir=str(out_dir)
        )
        status, output = run_script(python, str(repo), argv, timeout_s=900)
        print("---- prep_asset output (tail) ----")
        print("\n".join(output.strip().splitlines()[-25:]))
        check(status == 0, f"scripts/prep_asset.py passes the exported asset (exit {status})")

    print(f"\n{len(failures)} failure(s)")
    if failures:
        raise SystemExit("FAILED: " + "; ".join(failures))


main()
