"""Start from an existing asset config: apply its map to the Blender materials of the same names.

Reopening is the common case. An asset exported by this add-on comes back with its ``.blend``
copy, where the assignments already live. A hand-written map is different: the Phantom 4's,
for instance, records in its comments why each line is what it is. That map is applied here to
whichever Blender materials carry its source names, matched case-insensitively as the asset
audit matches them -- by its raw name or by the USD-safe form of it, since the map is keyed by what
the audit sees in the USD. By default a material that already has a thermal material keeps it.
"""

import bpy
from bpy.props import BoolProperty, StringProperty
from bpy.types import Operator, PropertyGroup

from . import prefs, scene_stats, thermal_view
from .bridge_client import BridgeError, run_bridge
from .naming import safe_identifier


class IrsimAssetName(PropertyGroup):
    """One entry of ``WindowManager.irsim_assets`` (``name`` is the asset name)."""

    by_addon: BoolProperty()


def apply_map(
    materials: dict[str, str], candidates, overwrite: bool
) -> tuple[list[str], list[str], list[str]]:
    """``(assigned, kept, unmatched)`` Blender material names after applying ``materials``."""
    lookup = {key.lower(): value for key, value in materials.items()}
    assigned, kept, unmatched = [], [], []
    for mat in sorted(candidates, key=lambda m: m.name):
        # A map's keys are names as the audit sees them in USD, so a Blender material still
        # carrying its raw FBX name (`[DJI_Phantom_4_Pro]Glass`) matches through its safe form.
        thermal = lookup.get(mat.name.lower(), lookup.get(safe_identifier(mat.name).lower()))
        if thermal is None:
            unmatched.append(mat.name)
        elif mat.irsim_material and not overwrite and mat.irsim_material != thermal:
            kept.append(mat.name)
        else:
            mat.irsim_material = thermal
            assigned.append(mat.name)
    return assigned, kept, unmatched


def refresh_asset_list(context: bpy.types.Context) -> int:
    repo, python = prefs.settings(context)
    result = run_bridge(python, repo, "list-assets")
    items = context.window_manager.irsim_assets
    items.clear()
    for record in result["assets"]:
        item = items.add()
        item.name = record["name"]
        item.by_addon = bool(record["by_addon"])
    return len(items)


class LoadAssetMap(Operator):
    """Assign materials from an existing asset config, matching Blender material names"""

    bl_idname = "irsim.load_asset_map"
    bl_label = "Load materials from an asset"
    bl_options = {"REGISTER", "UNDO"}

    asset: StringProperty(name="Asset", description="An asset config in configs/assets/")
    overwrite: BoolProperty(
        name="Replace existing assignments",
        description="Also change materials that already have a thermal material",
        default=False,
    )

    @classmethod
    def poll(cls, context):
        return context.mode == "OBJECT"

    def invoke(self, context, event):
        try:
            refresh_asset_list(context)
        except BridgeError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        return context.window_manager.invoke_props_dialog(
            self, width=420, title="Load materials from an asset", confirm_text="Load"
        )

    def draw(self, context):
        layout = self.layout
        layout.prop_search(self, "asset", context.window_manager, "irsim_assets")
        layout.prop(self, "overwrite")

    def execute(self, context):
        repo, python = prefs.settings(context)
        try:
            result = run_bridge(python, repo, "read-asset", {"name": self.asset})
        except BridgeError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        parts = scene_stats.part_objects(context)
        candidates = {slot.material for ob in parts for slot in ob.material_slots if slot.material}
        assigned, kept, unmatched = apply_map(result["materials"], candidates, self.overwrite)
        scene_stats.refresh(context)
        if context.scene.irsim.thermal_view:
            thermal_view.apply(context)
        if kept:
            self.report(
                {"WARNING"},
                f"Kept {len(kept)} existing assignment(s); tick 'Replace' to take the map's",
            )
        if result["scale_to_metres"] != 1.0:
            self.report(
                {"WARNING"},
                f"{self.asset} records scale_to_metres {result['scale_to_metres']:g}: if the "
                "model is still in its source units, apply that scale (Check and export > size)",
            )
        msg = f"{self.asset}: {len(assigned)} material(s) assigned"
        if unmatched:
            msg += f"; {len(unmatched)} not in its map: {', '.join(unmatched[:6])}"
            msg += "..." if len(unmatched) > 6 else ""
        self.report({"INFO"}, msg)
        return {"FINISHED"}


classes = (IrsimAssetName, LoadAssetMap)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.WindowManager.irsim_assets = bpy.props.CollectionProperty(type=IrsimAssetName)


def unregister():
    del bpy.types.WindowManager.irsim_assets
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
