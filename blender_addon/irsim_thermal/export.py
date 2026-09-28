"""Hand the model to irsim: USD + a ``.blend`` copy + ``configs/assets/<name>.yaml``, then audit.

What the export writes, all inside the repository the add-on is pointed at:

* ``3d_models/<name>/<name>.usdc`` -- the parts, one prim per Blender object, with their materials
  bound by name (Blender's own USD exporter, ``UsdPreviewSurface`` for the RGB companion).
* ``3d_models/<name>/<name>.blend`` -- a copy of the working file, so the assignment can be
  reopened and edited later (the add-on's data lives on the materials inside it).
* ``configs/assets/<name>.yaml`` -- the asset map, written and re-loaded by the bridge with the
  project's own ``AssetConfig`` and the library check, ``scale_to_metres: 1.0`` because the model
  is exported in metres.

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


def _rename(ids, renames: dict[str, str]) -> list[str]:
    done = []
    for old, new in renames.items():
        block = ids.get(old)
        if block is not None:
            block.name = new
            done.append(f"{old} -> {block.name}")
    return done


def make_names_safe(parts) -> list[str]:
    """Rename used materials, parts and their single-user meshes to USD-safe, map-safe names."""
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
    clashes = [p.name for p in (usd_path, blend_path) if p.exists()]
    if clashes:
        return (
            f"3d_models/{name}/ already holds {', '.join(clashes)}, which no export of this add-on "
            "wrote. Export under another name."
        )
    return ""


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
        name = settings.asset_name.strip()
        if not NAME.match(name):
            self.report({"ERROR"}, "Asset name: lower-case letters, digits and underscores")
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
        try:
            notes = make_names_safe(parts)
            out_dir.mkdir(parents=True, exist_ok=True)

            previous = (list(context.selected_objects), context.view_layer.objects.active)
            for ob in context.scene.objects:
                ob.select_set(ob in parts and ob.visible_get())
            hidden = [ob.name for ob in parts if not ob.visible_get()]
            if hidden:
                self.report({"WARNING"}, f"Hidden in the viewport, not exported: {hidden}")
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

            used = scene_stats.used_materials([ob for ob in parts if ob.visible_get()])
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
        finally:
            if view_was_on:
                thermal_view.apply(context)

        settings.last_export = (
            f"{usd_path.relative_to(repo).as_posix()} + "
            f"{pathlib.Path(result['file']).relative_to(repo).as_posix()}"
        )
        if notes:
            self.report({"WARNING"}, f"Renamed for USD: {'; '.join(notes)}")
        self.report({"INFO"}, f"Exported {name}: {settings.last_export}")
        return {"FINISHED"}


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
