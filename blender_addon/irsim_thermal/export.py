"""Hand the model to irsim: USD + a ``.blend`` copy + ``configs/assets/<name>.yaml``, then audit.

What the export writes, all inside the repository the add-on is pointed at:

* ``3d_models/<name>/<name>.usdc`` -- the parts, one prim per Blender object, with their materials
  bound by name (Blender's own USD exporter, ``UsdPreviewSurface`` for the RGB companion). The
  hidden parts are in it too, each marked ``purpose = "guide"``: a prim that can be selected and
  moved in Isaac Sim but that no camera renders (:mod:`.hidden_parts`).
* ``3d_models/<name>/<name>.blend`` -- a copy of the working file, so the assignment can be
  reopened and edited later (the add-on's data lives on the materials inside it).
* ``configs/assets/<name>.yaml`` -- the asset map, written and re-loaded by the bridge with the
  project's own ``AssetConfig`` and the library check, ``scale_to_metres: 1.0`` because the model
  is exported in metres.
* ``3d_models/<name>/<name>.structure.yaml`` -- only when there are any: the contacts, facing
  pairs and hidden parts (:mod:`.connections`, :mod:`.hidden_parts`), checked by the bridge. The
  asset config has no place for them yet (roadmap AI.11). Rejected connections are left out;
  ones nobody reviewed are written with ``reviewed: false``.

``3d_models/`` is git-ignored, so the geometry never lands in a commit; the YAML is the part meant
to be reviewed and committed. Before anything is written, material and part names are made
identical in all three places (:mod:`.naming`), and after the USD is written it is read back and
every material bound in it is checked against the map, so a name the exporter changed on its own
is caught here rather than as an unmapped prim in a render.

*Run irsim audit* then runs ``scripts/prep_asset.py --asset <name>``: the same import, USD and
coverage audit every other asset in the library goes through (ADR 0128), in a second Blender
process, without blocking this one.
"""

import os
import pathlib
import re
import shutil
import subprocess
import tempfile

import bpy
from bpy.props import FloatProperty
from bpy.types import Operator

from . import naming, prefs, scene_stats, thermal_view
from .bridge import MARKER
from .bridge_client import BridgeError, child_env, run_bridge
from .coverage import COVERAGE_GATE

NAME = re.compile(r"^[a-z][a-z0-9_]*$")

#: The running audit, if any: ``{"proc", "log", "name", "text"}``.
_audit: dict = {}


def _bound_material_names(usd_path: pathlib.Path) -> set[str]:
    """Names of every material bound to a mesh or a face subset in the exported stage."""
    from pxr import Usd, UsdGeom, UsdShade

    stage = Usd.Stage.Open(str(usd_path))
    names: set[str] = set()
    for prim in stage.Traverse():
        if not (prim.IsA(UsdGeom.Mesh) or prim.IsA(UsdGeom.Subset)):
            continue
        material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
        if material:
            names.add(material.GetPrim().GetName())
    return names


def mark_guides(usd_path: pathlib.Path, names: set[str]) -> set[str]:
    """Mark the prims named ``names`` ``purpose = "guide"`` and save; returns the names found.

    Blender writes an object as an Xform holding its mesh, both under the object's name once the
    names are made safe; both are marked, and anything beneath inherits it.
    """
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(str(usd_path))
    found: set[str] = set()
    for prim in stage.Traverse():
        if prim.GetName() in names and prim.IsA(UsdGeom.Imageable):
            UsdGeom.Imageable(prim).CreatePurposeAttr(UsdGeom.Tokens.guide)
            found.add(prim.GetName())
    stage.GetRootLayer().Save()
    del stage  # an open layer would stop a later export writing the same file
    return found


def _rename(ids, renames: dict[str, str]) -> list[str]:
    done = []
    for old, new in renames.items():
        block = ids.get(old)
        if block is not None:
            block.name = new
            done.append(f"{old} -> {block.name}")
    return done


def make_names_safe(parts) -> list[str]:
    """Rename used materials, parts and their single-user meshes to USD-safe, map-safe names.

    ``parts`` includes the hidden parts: they are not in the USD, but the structure file names
    them, and a contact between ``engine`` and ``floor_pan`` must name both the same way.
    """
    used = scene_stats.used_materials(parts)
    notes = _rename(
        bpy.data.materials,
        naming.unique_safe_names(
            [m.name for m in used],
            taken=[m.name for m in bpy.data.materials if m not in used],
        ),
    )
    part_set = set(parts)
    notes += _rename(
        bpy.data.objects,
        naming.unique_safe_names(
            [ob.name for ob in parts],
            taken=[ob.name for ob in bpy.data.objects if ob not in part_set],
        ),
    )
    # The prim under each part's Xform is named after its mesh data; a part called `battery`
    # whose mesh is still `Cube.004` would export as /root/battery/Cube_004.
    for ob in parts:
        me = ob.data
        if me.users == 1 and me.name != ob.name:
            me.name = ob.name
    return notes


def structure_path(repo: pathlib.Path, name: str) -> pathlib.Path:
    return repo / "3d_models" / name / f"{name}.structure.yaml"


def preflight(
    repo: pathlib.Path,
    name: str,
    usd_path: pathlib.Path,
    blend_path: pathlib.Path,
    replace: bool,
) -> str:
    """Why this export must not start, or ``""``. Checked before a single file is written.

    The bridge refuses to overwrite a hand-written asset config too, but by the time it is asked
    the USD and the ``.blend`` are already on disk -- and an asset whose ``source_file`` happens
    to be ``3d_models/<name>/<name>.usdc`` would have had its geometry replaced. So the same rule
    runs here first: replace only what an earlier export of this add-on wrote, and only when asked.
    """
    config = repo / "configs" / "assets" / f"{name}.yaml"
    if config.exists():
        first = config.read_text(encoding="utf-8").splitlines()[:1]
        if not first or MARKER not in first[0]:
            return (
                f"configs/assets/{name}.yaml was written by hand, so '{name}' belongs to another "
                "asset. Export under another name."
            )
        if not replace:
            return (
                f"configs/assets/{name}.yaml exists (from an earlier export). Tick 'Replace "
                "earlier export' to overwrite it."
            )
        return ""
    clashes = [p.name for p in (usd_path, blend_path, structure_path(repo, name)) if p.exists()]
    if clashes:
        return (
            f"3d_models/{name}/ already holds {', '.join(clashes)}, which no export of this add-on "
            "wrote. Export under another name."
        )
    return ""


def default_asset_name(context: bpy.types.Context) -> str:
    """A name for an asset nobody named: the ``.blend`` file's, else the largest part's.

    A name some program made up (``Cube``, ``GeometryNode_57``) is taken only when there is
    nothing better, so an unsaved default cube still exports, as ``cube``.
    """
    stem = pathlib.Path(bpy.data.filepath).stem if bpy.data.filepath else ""
    candidates = [stem]
    parts = scene_stats.part_objects(context)
    if parts:
        candidates.append(max(parts, key=lambda ob: ob.dimensions.length).name)
    usable = [(t, naming.config_name(t)) for t in candidates if naming.config_name(t)]
    real = [name for text, name in usable if not naming.is_default_name(text)]
    return (real or [name for _, name in usable] or [""])[0]


class ExportAsset(Operator):
    """Write the parts as USD, a .blend copy and configs/assets/<name>.yaml into the repository"""

    bl_idname = "irsim.export_asset"
    bl_label = "Export to irsim"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return context.mode == "OBJECT"

    def execute(self, context):
        settings = context.scene.irsim
        repo, python = prefs.settings(context)
        if not settings.asset_name:
            settings.asset_name = default_asset_name(context)  # tidied by its update callback
            if settings.asset_name:
                self.report({"INFO"}, f"Asset name: {settings.asset_name}")
        name = settings.asset_name
        if not NAME.match(name):
            self.report(
                {"ERROR"},
                "Give the asset a name in Latin letters (Check and export > Asset name)",
            )
            return {"CANCELLED"}
        if not repo:
            self.report({"ERROR"}, "The irsim repository is not set (add-on preferences)")
            return {"CANCELLED"}
        parts = scene_stats.part_objects(context)
        if not parts:
            self.report({"ERROR"}, "No mesh objects to export")
            return {"CANCELLED"}
        summary = scene_stats.refresh(context)
        if summary.coverage < COVERAGE_GATE:
            self.report(
                {"ERROR"},
                f"Only {100 * summary.coverage:.1f} % of the area has a thermal material; irsim "
                f"needs {100 * COVERAGE_GATE:.0f} %. Use 'Select unassigned' to find the rest.",
            )
            return {"CANCELLED"}
        out_dir = pathlib.Path(repo) / "3d_models" / name
        usd_path = out_dir / f"{name}.usdc"
        blend_path = out_dir / f"{name}.blend"
        refusal = preflight(pathlib.Path(repo), name, usd_path, blend_path, settings.replace_export)
        if refusal:
            self.report({"ERROR"}, refusal)
            return {"CANCELLED"}

        view_was_on = settings.thermal_view
        if view_was_on:
            thermal_view.restore(context)
        hidden = scene_stats.hidden_parts(context)
        try:
            notes = make_names_safe(parts + hidden)
            exported = [ob for ob in parts if ob.visible_get()]
            structure = structure_payload(context, name, exported, hidden)
            if structure is not None:
                try:
                    run_bridge(python, repo, "write-structure", {**structure, "check_only": True})
                except BridgeError as exc:
                    self.report(
                        {"ERROR"},
                        f"Nothing was exported; fix the connections or hidden parts:\n{exc}",
                    )
                    return {"CANCELLED"}
            out_dir.mkdir(parents=True, exist_ok=True)

            previous = (list(context.selected_objects), context.view_layer.objects.active)
            written = [ob for ob in parts + hidden if ob.visible_get()]
            for ob in context.scene.objects:
                ob.select_set(ob in written)
            off = [ob.name for ob in parts + hidden if not ob.visible_get()]
            if off:
                self.report({"WARNING"}, f"Hidden in the viewport, not exported: {off}")
            bpy.ops.wm.usd_export(
                filepath=str(usd_path),
                check_existing=False,
                selected_objects_only=True,
                export_animation=False,
                export_materials=True,
                generate_preview_surface=True,
                export_lights=False,
                export_cameras=False,
                evaluation_mode="RENDER",
                convert_scene_units="METERS",
            )
            for ob in context.scene.objects:
                ob.select_set(ob in previous[0])
            context.view_layer.objects.active = previous[1]

            guides = {ob.name for ob in hidden if ob.visible_get()}
            missing = guides - mark_guides(usd_path, guides)
            if missing:
                self.report(
                    {"ERROR"},
                    f"Hidden parts missing from the USD, so not marked as guides: "
                    f"{sorted(missing)}. Nothing was written to configs/assets.",
                )
                return {"CANCELLED"}
            used = scene_stats.used_materials(written)
            mapping = {m.name: m.irsim_material for m in used if m.irsim_material}
            bound = _bound_material_names(usd_path)
            lowered = {k.lower() for k in mapping}
            unmapped = sorted(n for n in bound if n.lower() not in lowered)
            unassigned = {m.name for m in used if not m.irsim_material}
            surprising = [n for n in unmapped if n not in unassigned]
            if surprising:
                self.report(
                    {"ERROR"},
                    "The USD binds materials the map does not name (the exporter renamed "
                    f"them?): {surprising}. Nothing was written to configs/assets.",
                )
                return {"CANCELLED"}

            bpy.ops.wm.save_as_mainfile(filepath=str(blend_path), copy=True, check_existing=False)

            payload = {
                "name": name,
                "source_file": usd_path.relative_to(repo).as_posix(),
                "scale_to_metres": 1.0,
                "materials": dict(sorted(mapping.items())),
                "parts": sorted(ob.name for ob in parts if ob.visible_get()),
                "structure": structure or parts_only(exported),
                "overwrite": settings.replace_export,
                "notes": [
                    f"Coverage at export: {100 * summary.coverage:.1f} % of "
                    f"{summary.total_area_m2:.4g} m2 ({len(parts)} parts).",
                    *(["Renamed for USD: " + "; ".join(notes)] if notes else []),
                ],
            }
            try:
                result = run_bridge(python, repo, "write-asset", payload)
            except BridgeError as exc:
                if exc.kind == "exists":
                    self.report(
                        {"ERROR"},
                        f"{exc}. Tick 'Replace earlier export' to overwrite it.",
                    )
                else:
                    self.report({"ERROR"}, f"Asset config not written: {exc}")
                return {"CANCELLED"}
            written_structure = ""
            if structure is not None:
                structure["overwrite"] = True  # the asset config's own check above has decided
                try:
                    done = run_bridge(python, repo, "write-structure", structure)
                except BridgeError as exc:  # checked above, so only a failure to write lands here
                    self.report(
                        {"ERROR"},
                        f"The model and its materials were exported, but not its connections and "
                        f"hidden parts:\n{exc}",
                    )
                    return {"CANCELLED"}
                written_structure = (
                    f" ({result.get('parts', 0)} parts, {done['contacts']} contacts and "
                    f"{done['hidden_parts']} hidden parts in the asset config; "
                    f"{done['facing']} facing pairs beside the USD)"
                )
            else:
                stale = structure_path(pathlib.Path(repo), name)
                if stale.exists() and MARKER in stale.read_text(encoding="utf-8")[:200]:
                    stale.unlink()  # an earlier export's, and this one has none
        finally:
            if view_was_on:
                thermal_view.apply(context)

        settings.last_export = (
            f"{usd_path.relative_to(repo).as_posix()} + "
            f"{pathlib.Path(result['file']).relative_to(repo).as_posix()}{written_structure}"
        )
        if notes:
            self.report({"WARNING"}, f"Renamed for USD: {'; '.join(notes)}")
        self.report({"INFO"}, f"Exported {name}: {settings.last_export}")
        return {"FINISHED"}


def _area(ob: bpy.types.Object) -> float:
    return float(scene_stats.part_stats(ob).total_area_m2)


#: irsim's hidden-part status has two values; a published datasheet figure is a measurement, by
#: whoever published it, and the reference that says so stays in the structure file.
STATUS = {"MEASURED": "MEASURED", "PUBLISHED": "MEASURED", "ESTIMATED": "ESTIMATED"}


def _centre(objects) -> list[float]:
    if not objects:
        return [0.0, 0.0, 0.0]
    lo, hi = scene_stats.world_bounds(objects)
    return [float(x) for x in (lo + hi) / 2]


def parts_only(exported) -> dict:
    """The structure of an export with no connections or hidden parts: its named parts alone."""
    return {"part_areas": {ob.name: _area(ob) for ob in exported}, "centre": _centre(exported)}


def structure_payload(context, name: str, exported, hidden) -> dict | None:
    """The contacts, facing pairs and hidden parts to write, or ``None`` when there are none.

    One payload serves both files: the bridge writes the contacts and hidden parts into the asset
    config's ``parts:`` block (AI.11) and the rest -- facing pairs, idle heat, references, what
    nobody reviewed -- into the add-on's own structure file.
    """
    names = {ob.name for ob in exported} | {ob.name for ob in hidden}
    contacts, facing = [], []
    for c in context.scene.irsim_connections:
        if c.status == "REJECTED" or c.a is None or c.b is None:
            continue
        if c.a.name not in names or c.b.name not in names:
            continue
        record = {
            "parts": sorted((c.a.name, c.b.name)),
            "reviewed": c.status != "FOUND",
            "found_by": "person" if c.status == "ADDED" else "finder",
        }
        if c.kind == "CONTACT":
            record.update(joint=c.joint, area_m2=float(f"{c.area_m2:.6g}"))
            contacts.append(record)
        else:
            by_name = {c.a.name: c.area_m2, c.b.name: c.area_ba_m2}
            record.update(
                area_m2=[float(f"{by_name[n]:.6g}") for n in record["parts"]],
                gap_m=float(f"{c.gap_m:.6g}"),
            )
            facing.append(record)
    if not (contacts or facing or hidden):
        return None
    from . import library_state
    from .hidden_parts import box_record, world_box

    hidden_records = []
    for ob in hidden:
        comp = ob.irsim_component
        thermals = sorted(
            {s.material.irsim_material for s in ob.material_slots if s.material} - {""}
        )
        material = thermals[0] if thermals else ""
        centre, size = world_box(ob)
        c_p = comp.specific_heat_j_kgk
        if c_p <= 0.0 and not comp.component:  # nothing else to take it from: the material's
            item = library_state.find(context, material)
            c_p = item.specific_heat_j_kgk if item is not None else 0.0
        hidden_records.append(
            {
                "name": ob.name,
                "kind": comp.kind.lower(),
                "material": material,
                "inside": ob.parent.name if ob.parent is not None else None,
                "box": box_record(ob),
                # What the asset config carries (HiddenPartSpec): the box around the part in the
                # model's own axes, and only the numbers that override its component.
                "aabb": {
                    "centre_m": [float(x) for x in centre],
                    "size_m": [float(x) for x in size],
                },
                "component": comp.component or None,
                "specific_heat_j_kgk": float(f"{c_p:.6g}") if c_p > 0.0 else None,
                "dissipation_w": (
                    float(f"{comp.heat_rated_w:.6g}")
                    if comp.heat_rated_w > 0.0 or not comp.component
                    else None
                ),
                "status": STATUS[comp.values_source],
                "mass_kg": float(f"{comp.mass_kg:.6g}"),
                "heat_w": {
                    "idle": float(f"{comp.heat_idle_w:.6g}"),
                    "rated": float(f"{comp.heat_rated_w:.6g}"),
                },
                "values": {"source": comp.values_source.lower(), "reference": comp.reference},
                "usd": "a prim of its own in the exported USD, purpose guide: never rendered",
                "area_m2": _area(ob),
            }
        )
    return {
        "name": name,
        "centre": _centre(exported),
        "part_areas": {ob.name: _area(ob) for ob in exported},
        "contacts": sorted(contacts, key=lambda r: r["parts"]),
        "facing": sorted(facing, key=lambda r: r["parts"]),
        "hidden_parts": hidden_records,
    }


def _blender_executable() -> str:
    return shutil.which("blender") or bpy.app.binary_path


def audit_argv(repo: str, name: str, blender: str, out_dir: str | None = None) -> list[str]:
    """The ``scripts/prep_asset.py`` command line for an exported asset.

    The asset config and the source are passed as absolute paths inside ``repo``, so the audit
    reads exactly the files this export wrote even when ``repo`` is not the checkout the script
    itself lives in.
    """
    root = pathlib.Path(repo)
    argv = [
        "scripts/prep_asset.py",
        "--asset",
        str(root / "configs" / "assets" / f"{name}.yaml"),
        "--source",
        str(root / "3d_models" / name / f"{name}.usdc"),
        "--blender",
        blender,
    ]
    if out_dir:
        argv += ["--out-dir", out_dir]
    return argv


def _poll_audit():
    proc = _audit.get("proc")
    if proc is None:
        return None
    if proc.poll() is None:
        return 1.0
    log = pathlib.Path(_audit["log"]).read_text(encoding="utf-8", errors="replace")
    text = bpy.data.texts.get(_audit["text"]) or bpy.data.texts.new(_audit["text"])
    text.clear()
    text.write(log)
    status = "passed" if proc.returncode == 0 else f"FAILED (exit {proc.returncode})"
    for scene in bpy.data.scenes:
        scene.irsim.audit_status = f"irsim audit {status}: see text '{_audit['text']}'"
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            area.tag_redraw()
    _audit.clear()
    return None


class RunAudit(Operator):
    """Run scripts/prep_asset.py on the exported asset, in the background"""

    bl_idname = "irsim.run_audit"
    bl_label = "Run irsim audit"

    @classmethod
    def poll(cls, context):
        return not _audit and bool(context.scene.irsim.asset_name)

    def execute(self, context):
        repo, python = prefs.settings(context)
        name = context.scene.irsim.asset_name.strip()
        if not (pathlib.Path(repo) / "configs" / "assets" / f"{name}.yaml").exists():
            self.report({"ERROR"}, f"configs/assets/{name}.yaml does not exist: export first")
            return {"CANCELLED"}
        fd, log_name = tempfile.mkstemp(prefix="irsim_audit_", suffix=".log")
        try:
            with os.fdopen(fd, "wb") as log:
                proc = subprocess.Popen(
                    [python, *audit_argv(repo, name, _blender_executable())],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    cwd=repo,
                    env=child_env(repo),
                )
        except OSError as exc:
            self.report({"ERROR"}, f"Could not start the audit: {exc}")
            return {"CANCELLED"}
        _audit.update(proc=proc, log=log_name, name=name, text=f"irsim_audit_{name}.log")
        context.scene.irsim.audit_status = "irsim audit running..."
        bpy.app.timers.register(_poll_audit, first_interval=1.0)
        self.report({"INFO"}, "irsim audit started; the result appears in the panel")
        return {"FINISHED"}


class ApplyScale(Operator):
    """Scale the whole model by this factor and apply it to the mesh data"""

    bl_idname = "irsim.apply_scale"
    bl_label = "Apply scale"
    bl_options = {"REGISTER", "UNDO"}

    factor: FloatProperty(name="Factor", default=1.0, min=1e-6)

    @classmethod
    def poll(cls, context):
        return context.mode == "OBJECT"

    def execute(self, context):
        # The same two moves as scripts/prep_asset.py's `_rescale`: scale every root about the
        # world origin, then apply the scale into the mesh data so the exported points carry
        # metres. Locations are scaled but not applied, so each part keeps its own origin.
        roots = [ob for ob in context.scene.objects if ob.parent is None]
        for ob in roots:
            ob.scale = tuple(s * self.factor for s in ob.scale)
            ob.location = tuple(c * self.factor for c in ob.location)
        previous = (list(context.selected_objects), context.view_layer.objects.active)
        for ob in context.scene.objects:
            ob.select_set(ob.type == "MESH" and ob.visible_get())
        meshes = [ob for ob in context.scene.objects if ob.select_get()]
        if meshes:
            context.view_layer.objects.active = meshes[0]
            bpy.ops.object.transform_apply(
                location=False, rotation=False, scale=True, isolate_users=True
            )
        for ob in context.scene.objects:
            ob.select_set(ob in previous[0])
        context.view_layer.objects.active = previous[1]
        scene_stats.refresh(context)
        self.report({"INFO"}, f"Scaled by {self.factor:g}")
        return {"FINISHED"}


class CheckModel(Operator):
    """Measure coverage, size and names, and list what is left to do"""

    bl_idname = "irsim.check"
    bl_label = "Check model"

    @classmethod
    def poll(cls, context):
        return context.mode == "OBJECT"

    def execute(self, context):
        summary = scene_stats.refresh(context)
        self.report(
            {"INFO"},
            f"Coverage {100 * summary.coverage:.1f} %, "
            f"{len(summary.unassigned_parts)} part(s) still need a material",
        )
        return {"FINISHED"}


classes = (ExportAsset, RunAudit, ApplyScale, CheckModel)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    if _audit.get("proc") is not None and bpy.app.timers.is_registered(_poll_audit):
        bpy.app.timers.unregister(_poll_audit)
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
