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
    for name in ("src", "scripts"):
        (root / name).symlink_to(REPO / name)
    # data/ is linked entry by entry, with a real spectra/materials/ underneath: a material made
    # from a picked curve copies that curve into it (B11), and the copy must land here, not in
    # the repository's own data.
    data = root / "data"
    data.mkdir()
    for entry in (REPO / "data").iterdir():
        if entry.name != "spectra":
            (data / entry.name).symlink_to(entry)
    (data / "spectra").mkdir()
    for entry in (REPO / "data" / "spectra").iterdir():
        if entry.name != "materials":
            (data / "spectra" / entry.name).symlink_to(entry)
    (data / "spectra" / "materials").mkdir()
    for entry in (REPO / "data" / "spectra" / "materials").iterdir():
        (data / "spectra" / "materials" / entry.name).symlink_to(entry)
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

    panels = (
        ui.MainPanel,
        ui.PartsPanel,
        ui.LibraryPanel,
        ui.HiddenPartsPanel,
        ui.ConnectionsPanel,
        ui.ExportPanel,
    )
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
    meshes = sum(
        1
        for ob in ctx.scene.objects
        if ob.type == "MESH" and not ob.hide_render and not ob.irsim_component.is_hidden_part
    )
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


def structure_checks(ctx, repo: pathlib.Path) -> None:
    """Connections and hidden parts, on shapes whose answers are known exactly.

    * ``block`` (1 m cube) with ``lid`` (0.5 m cube) standing on it: one contact of 0.25 m², and
      no facing pair between them (the lid's sides look sideways at nothing);
    * three 1 m plates stacked 0.2 m apart, facing: ``plate_a`` up, ``plate_b`` and ``plate_c``
      down. a and b face each other fully, b hides c from a, and c looks down onto b's back;
    * ``far``, a cube 10 m away, connected to nothing.
    """
    from irsim_thermal import export, hidden_parts, scene_stats, ui
    from mathutils import Vector

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    ctx.scene.irsim_connections.clear()
    ctx.scene.irsim.touch_gap_m = ctx.scene.irsim.facing_gap_m = 0.0
    block = add("cube", "block", size=1.0)
    lid = add("cube", "lid", location=(0.0, 0.0, 0.75), size=0.5)
    add("plane", "plate_a", location=(3.0, 0.0, 0.0), size=1.0)
    plate_b = add("plane", "plate_b", location=(3.0, 0.0, 0.2), size=1.0)
    plate_c = add("plane", "plate_c", location=(3.0, 0.0, 0.4), size=1.0)
    for plate in (plate_b, plate_c):
        plate.rotation_euler = (math.pi, 0.0, 0.0)
    add("cube", "far", location=(10.0, 0.0, 0.0), size=0.5)
    ctx.view_layer.update()

    check(bpy.ops.irsim.find_connections() == {"FINISHED"}, "the connection finder runs")
    found = {(frozenset((c.a.name, c.b.name)), c.kind): c for c in ctx.scene.irsim_connections}
    contact = found.get((frozenset(("block", "lid")), "CONTACT"))
    check(
        contact is not None and abs(contact.area_m2 - 0.25) < 0.0125,
        "the lid touches the block over 0.25 m² (found "
        f"{contact.area_m2 if contact else 0:.4f}, 5 % allowed)",
    )
    check(contact is not None and contact.joint == "dry_default", "a found contact starts dry")
    check(
        (frozenset(("block", "lid")), "FACING") not in found,
        "touching parts are not also listed as facing",
    )
    ab = found.get((frozenset(("plate_a", "plate_b")), "FACING"))
    check(
        ab is not None
        and abs(ab.area_m2 - 1.0) < 0.05
        and abs(ab.area_ba_m2 - 1.0) < 0.05
        and abs(ab.gap_m - 0.2) < 0.002,
        "plates a and b face each other fully across 0.2 m "
        f"({ab.area_m2 if ab else 0:.3f} / {ab.area_ba_m2 if ab else 0:.3f} m², "
        f"gap {ab.gap_m if ab else 0:.4f} m)",
    )
    check(
        (frozenset(("plate_a", "plate_c")), "FACING") not in found,
        "plate b hides plate c from plate a",
    )
    check(
        (frozenset(("plate_b", "plate_c")), "FACING") in found,
        "plate c looks down onto plate b's back",
    )
    check(not any("far" in key[0] for key in found), "the part 10 m away is connected to nothing")
    check(
        all(c.status == "FOUND" for c in ctx.scene.irsim_connections),
        "everything found waits for review",
    )

    # The person's verdicts survive a new search.
    connections_list = list(ctx.scene.irsim_connections)
    i_contact = connections_list.index(contact)
    i_ab = connections_list.index(ab)
    bpy.ops.irsim.set_connection_status(index=i_contact, status="CONFIRMED")
    bpy.ops.irsim.set_connection_status(index=i_ab, status="REJECTED")
    connection(ctx, "block", "lid", "CONTACT").joint = "bolted_ferrous_new"
    n_before = len(ctx.scene.irsim_connections)
    bpy.ops.irsim.find_connections()
    after = {(frozenset((c.a.name, c.b.name)), c.kind): c for c in ctx.scene.irsim_connections}
    check(len(after) == n_before, "a second search neither adds nor loses a connection")
    kept = after[(frozenset(("block", "lid")), "CONTACT")]
    check(
        kept.status == "CONFIRMED" and kept.joint == "bolted_ferrous_new",
        "a confirmed contact stays confirmed, with the joint the person chose",
    )
    check(
        after[(frozenset(("plate_a", "plate_b")), "FACING")].status == "REJECTED",
        "a rejected pair stays rejected",
    )

    # A connection added by hand.
    far = ctx.scene.objects["far"]
    select_only(far, block)
    check(bpy.ops.irsim.add_connection(kind="CONTACT") == {"FINISHED"}, "a contact is added")
    added = ctx.scene.irsim_connections[ctx.scene.irsim.active_connection_index]
    check(
        added.status == "ADDED" and added.area_m2 == 0.0,
        "by hand, with no area measured because the parts do not touch",
    )
    added.area_m2 = 0.01

    # A hidden part.
    select_only(block)
    result = bpy.ops.irsim.add_hidden_part(
        part_name="engine block",
        kind="PISTON_ENGINE",
        size=(0.3, 0.3, 0.3),
        material="rusted_steel",
        mass_kg=0.0,
        heat_idle_w=1000.0,
        heat_rated_w=5000.0,
    )
    engine = ctx.scene.objects.get("engine_block")
    check(result == {"FINISHED"} and engine is not None, "a hidden part is added, named safely")
    check(
        engine.irsim_component.is_hidden_part
        and engine.display_type == "SOLID"
        and engine.show_in_front
        and not engine.hide_render
        and engine.parent == block,
        "it is a solid drawn in front of the shell, and moves with the part it is inside",
    )
    centre, size = hidden_parts.world_box(engine)
    check(
        centre.length < 1e-6 and all(abs(v - 0.3) < 1e-6 for v in size),
        "it sits in the middle of the block, at the size asked for",
    )
    parts = scene_stats.part_objects(ctx)
    check(engine not in parts, "it is not a part the USD exports")
    # One from the component library: at its real size, in the middle of the part selected.
    select_only(lid)
    check(
        bpy.ops.irsim.add_component(
            component="drone_battery",
            part_name="battery pack",
            material="abs_plastic_white",
            mass_kg=0.468,
            heat_idle_w=0.5,
            heat_rated_w=25.0,
        )
        == {"FINISHED"},
        "a component is added from the library",
    )
    pack = ctx.scene.objects.get("battery_pack")
    pack_centre, pack_size = hidden_parts.world_box(pack)
    check(
        pack is not None
        and pack.irsim_component.is_hidden_part
        and pack.irsim_component.kind == "BATTERY"
        and pack.parent == lid
        and all(abs(a - b) < 1e-4 for a, b in zip(pack_size, (0.14, 0.07, 0.04), strict=True))
        and (pack_centre - Vector((0.0, 0.0, 0.75))).length < 1e-6,
        f"at its real size {tuple(round(v, 4) for v in pack_size)} m, inside the lid, a battery",
    )
    # The whole library: every entry has its file, the menu is grouped, and each one arrives at
    # its catalog size along its catalog axis.
    entries = hidden_parts.catalog()
    menu = hidden_parts._component_items(None, ctx)
    categories = list(dict.fromkeys(e["category"] for e in entries))
    check(
        len(entries) >= 90
        and [label for ident, label, _ in menu if not ident] == categories
        and len([i for i in menu if i[0]]) == len(entries),
        f"{len(entries)} library components with their files, under {len(categories)} headings",
    )
    wrong = []
    for e in entries:
        ob = hidden_parts.import_component(ctx, e["file"])
        ctx.view_layer.update()
        _, got = hidden_parts.world_box(ob)
        s = e["size_m"]
        axis = e.get("axis", "X")
        want = s if e["shape"] == "box" else [s[0] if a == axis else max(s[1:]) for a in "XYZ"]
        if any(abs(a - b) > 1e-4 * max(want) for a, b in zip(got, want, strict=True)):
            wrong.append(f"{e['file']} {tuple(round(v, 4) for v in got)}")
        bpy.data.objects.remove(ob)
    check(not wrong, "every library component arrives at its catalog size: " + ", ".join(wrong))
    # A hidden part made by the first version (a wireframe marked not to render) is brought back.
    pack.display_type, pack.hide_render = "WIRE", True
    bpy.ops.irsim.show_hidden_parts()
    check(
        pack.display_type == "SOLID" and not pack.hide_render,
        "'Show hidden parts' makes an old wireframe one solid and exportable again",
    )
    kinds = {(i.kind, i.name) for i in ctx.scene.irsim.issues}
    check(("HIDDEN", "engine_block") in kinds, "the checklist asks for its mass")
    engine.irsim_component.mass_kg = 80.0
    scene_stats.refresh(ctx)
    kinds = {(i.kind, i.name) for i in ctx.scene.irsim.issues}
    check(("HIDDEN", "engine_block") not in kinds, "and stops asking once it has one")
    check(
        ("CONNECTION", "Connections") in kinds,
        "the checklist says how many connections are unreviewed",
    )
    bpy.ops.irsim.find_connections()
    found = {(frozenset((c.a.name, c.b.name)), c.kind): c for c in ctx.scene.irsim_connections}
    check(
        (frozenset(("block", "engine_block")), "FACING") in found,
        "the finder sees the hidden engine facing the block around it",
    )

    # The viewport draws only the selected parts' connections, unless asked for all.
    from irsim_thermal import overlay

    ctx.scene.irsim.active_connection_index = -1
    select_only(ctx.scene.objects["plate_b"])
    with_b = sum(1 for c in ctx.scene.irsim_connections if "plate_b" in (c.a.name, c.b.name))
    shown = len(overlay.segments(ctx))
    ctx.scene.irsim.connections_of_selected = False
    everything = len(overlay.segments(ctx))
    check(
        shown == with_b and everything == len(ctx.scene.irsim_connections) and shown < everything,
        f"lines for the selected part only ({shown} of {everything}), or all when asked",
    )
    ctx.scene.irsim.connections_of_selected = True
    import types

    lister = types.SimpleNamespace(
        filter_name="", bitflag_filter_item=1 << 30, use_filter_sort_alpha=False
    )
    flags, _ = ui.ConnectionsList.filter_items(lister, ctx, ctx.scene, "irsim_connections")
    check(
        sum(1 for f in flags if f) == with_b,
        f"the list, too, narrows to the selected part's {with_b} connections",
    )

    # Every panel draws with all of it in place.
    ctx.view_layer.objects.active = engine
    ctx.scene.irsim.active_connection_index = 0
    ui_checks(ctx)
    me = types_ns(ui)
    for conn in ctx.scene.irsim_connections:
        ui.ConnectionsList.draw_item(me, ctx, me.layout, ctx.scene, conn, 0, None, "", 0)
    ui.HiddenList.draw_item(me, ctx, me.layout, ctx.scene, engine, 0, None, "", 0)
    check(True, "the connection and hidden-part rows draw")

    # Export: the structure file beside the model, checked by the bridge.
    select_only(*parts)
    bpy.ops.irsim.assign(material="carbon_fibre", scope="PARTS")
    ctx.scene.irsim.asset_name = "smoke_structure"
    check(bpy.ops.irsim.export_asset() == {"FINISHED"}, "the model exports with its structure")
    path = export.structure_path(repo, "smoke_structure")
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    check("joint: bolted_ferrous_new" in text, "the confirmed contact is written, with its joint")
    check("parts: [plate_a, plate_b]" not in text, "the rejected pair is left out")
    check("parts: [plate_b, plate_c]" in text, "an unreviewed pair is written...")
    check("reviewed: false" in text, "...marked as unreviewed")
    check("name: engine_block" in text and "mass_kg: 80.0" in text, "the hidden part is written")
    usd = repo / "3d_models" / "smoke_structure" / "smoke_structure.usdc"
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(str(usd))  # kept: a stage freed mid-traversal expires its prims
    purposes = {p.GetName(): UsdGeom.Imageable(p).ComputePurpose() for p in stage.Traverse()}
    del stage  # an open layer would stop the next export writing the same file
    check(
        purposes.get("engine_block") == UsdGeom.Tokens.guide
        and purposes.get("battery_pack") == UsdGeom.Tokens.guide,
        "the hidden parts are in the USD as guides, which no camera renders",
    )
    check(purposes.get("block") == UsdGeom.Tokens.default_, "the ordinary parts are not")

    # Looked up again: a reference held across changes to the list may point at freed memory.
    connection(ctx, "block", "lid", "CONTACT").joint = "welded_magic"
    ctx.scene.irsim.replace_export = True
    before = usd.stat().st_mtime_ns
    try:
        bpy.ops.irsim.export_asset()
        check(False, "a joint the table lacks is refused")
    except RuntimeError as exc:
        check("not in joints.yaml" in str(exc), "a joint the table lacks is refused")
    check(usd.stat().st_mtime_ns == before, "...before anything is written")


def connection(ctx, a: str, b: str, kind: str):
    """The connection between ``a`` and ``b``, looked up afresh."""
    for c in ctx.scene.irsim_connections:
        if c.kind == kind and {c.a.name, c.b.name} == {a, b}:
            return c
    raise KeyError((a, b, kind))


def types_ns(ui):
    import types

    return types.SimpleNamespace(
        layout=FakeLayout([]),
        KIND_ICONS=ui.ConnectionsList.KIND_ICONS,
        STATUS_ICONS=ui.ConnectionsList.STATUS_ICONS,
    )


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

    # --- 7b. the library's other two forms, and the plot (B11, ADR 0175) -----------------------
    result = bpy.ops.irsim.new_material(
        "EXEC_DEFAULT",
        name="grey_smoke",
        reference="estimated",
        optical_form="grey",
        eps_grey=0.95,
        assign_after=False,
    )
    grey = library_state.find(ctx, "grey_smoke")
    check(
        result == {"FINISHED"}
        and grey is not None
        and all(
            close(getattr(grey, f"eps_{b}"), 0.95, 1e-6) for b in ("nir", "swir", "mwir", "lwir")
        ),
        "a one-value material is written and reads 0.95 in every band",
    )
    check(grey is not None and grey.forms == "grey 0.95", "and the panel says it is grey")

    picked = args.scratch / "picked_lw.csv"
    picked.write_text(
        "# source: smoke test\n"
        + "\n".join(f"{7.0 + 0.05 * i:.2f},0.93" for i in range(141))
        + "\n",
        encoding="utf-8",
    )
    bpy.ops.irsim.check_curve("EXEC_DEFAULT", filepath=str(picked))
    check(
        "covers LWIR" in ctx.window_manager.irsim_curve_check,
        f"a picked curve is checked by irsim ({ctx.window_manager.irsim_curve_check})",
    )
    result = bpy.ops.irsim.new_material(
        "EXEC_DEFAULT",
        name="curve_smoke",
        reference="smoke test curve",
        optical_form="curve",
        curve_file=str(picked),
        curve_quantity="emissivity",
        eps_lwir=0.5,  # typed, but the curve covers LWIR: the bridge leaves it out
        assign_after=False,
    )
    curved = library_state.find(ctx, "curve_smoke")
    copied = repo / "data" / "spectra" / "materials" / "curve_smoke.csv"
    check(
        result == {"FINISHED"} and curved is not None and close(curved.eps_lwir, 0.93, 1e-6),
        "a material from a picked curve reads the curve in LWIR, not the typed 0.5",
    )
    check(curved is not None and close(curved.curve_lwir, 1.0, 1e-9), "and all of LWIR is curve")
    check(
        copied.is_file() and not copied.is_symlink(), "the curve is copied into the data directory"
    )
    check(
        not (REPO / "data" / "spectra" / "materials" / "curve_smoke.csv").exists(),
        "and the real data directory is untouched",
    )
    bpy.ops.irsim.plot_material("EXEC_DEFAULT", material="aluminium_weathered")
    wm = ctx.window_manager
    check(
        wm.irsim_plot_material == "aluminium_weathered"
        and bpy.data.images.get(wm.irsim_plot_image) is not None
        and bpy.data.images[wm.irsim_plot_image].size[0] > 0,
        "a material is plotted across the bands into a Blender image",
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

    # --- 11. connections and hidden parts --------------------------------------------------------
    structure_checks(ctx, repo)

    print(f"\n{len(failures)} failure(s)")
    if failures:
        raise SystemExit("FAILED: " + "; ".join(failures))


main()
