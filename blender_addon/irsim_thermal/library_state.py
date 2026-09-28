"""The material library as the add-on last loaded it, through the bridge."""

import bpy
from bpy.types import Operator

from . import prefs
from .bridge_client import BridgeError, run_bridge
from .properties import BANDS


def library_items(context: bpy.types.Context):
    return context.window_manager.irsim_library


def find(context: bpy.types.Context, name: str):
    for item in library_items(context):
        if item.name == name:
            return item
    return None


def emissivity_lwir(context: bpy.types.Context) -> dict[str, float]:
    return {item.name: item.eps_lwir for item in library_items(context) if not item.error}


def load(context: bpy.types.Context) -> int:
    """Replace the cached library with a fresh read. Returns the number of materials."""
    repo, python = prefs.settings(context)
    result = run_bridge(python, repo, "library")
    items = library_items(context)
    active = context.scene.irsim.active_library_index if context.scene else 0
    previous = items[active].name if 0 <= active < len(items) else ""
    items.clear()
    for record in result["materials"]:
        item = items.add()
        item.name = record["name"]
        item.description = record["description"]
        item.reference = record["reference"]
        item.source = record["source"]
        item.surface_treatment = record["surface_treatment"]
        item.file = record["file"]
        item.spectral = bool(record["optical"]["spectral"])
        angular = record["optical"]["angular"]
        item.angular = angular.get("type", "")
        thermal = record["thermal"]
        for key in (
            "solar_absorptivity",
            "density_kg_m3",
            "specific_heat_j_kgk",
            "conductivity_w_mk",
            "thickness_m",
            "heat_capacity_j_m2_k",
        ):
            setattr(item, key, float(thermal[key]))
        errors = []
        for band in BANDS:
            values = record["bands"].get(band, {})
            if "error" in values:
                errors.append(f"{band}: {values['error']}")
                continue
            setattr(item, f"eps_{band}", float(values.get("emissivity", 0.0)))
            setattr(item, f"rho_{band}", float(values.get("reflectance", 0.0)))
            setattr(item, f"tau_{band}", float(values.get("transmittance", 0.0)))
        item.error = "; ".join(errors)
    wm = context.window_manager
    wm.irsim_joints.clear()
    for record in result.get("joints", []):
        joint = wm.irsim_joints.add()
        joint.name = record["name"]
        joint.h_c_w_m2_k = float(record["h_c_w_m2_k"])
        joint.status = record["status"]
        joint.source = record["source"]
    wm.irsim_library_hash = result.get("library_hash", "")
    wm.irsim_library_status = f"{len(items)} materials from {result['material_dir']}"
    if previous and context.scene:
        for i, item in enumerate(items):
            if item.name == previous:
                context.scene.irsim.active_library_index = i
                break
    return len(items)


class RefreshLibrary(Operator):
    """Read the material library from the irsim repository"""

    bl_idname = "irsim.refresh_library"
    bl_label = "Load library"

    def execute(self, context):
        try:
            n = load(context)
        except BridgeError as exc:
            context.window_manager.irsim_library_status = f"Not loaded: {exc}"
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        self.report({"INFO"}, f"Loaded {n} materials")
        return {"FINISHED"}


classes = (RefreshLibrary,)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
