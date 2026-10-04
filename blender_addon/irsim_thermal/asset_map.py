"""Start from an existing asset config: apply its map to the Blender materials of the same names.

Reopening is the common case. An asset exported by this add-on comes back with its ``.blend``
copy, where the assignments already live. A hand-written map is different: the Phantom 4's,
for instance, records in its comments why each line is what it is. That map is applied here to
whichever Blender materials carry its source names, matched case-insensitively as the asset
audit matches them -- by its raw name or by the USD-safe form of it, since the map is keyed by what
the audit sees in the USD. By default a material that already has a thermal material keeps it.

An asset config with a ``parts:`` block (AI.11) also brings back what the solver reads from it:
each hidden part becomes a hidden-part box where the config places it, with its component and its
own numbers, inside the part that encloses it; each contact between two parts the scene has
becomes a confirmed connection with its joint and area. Nothing already in the scene is replaced.
"""

from types import SimpleNamespace

import bpy
from bpy.props import BoolProperty, StringProperty
from bpy.types import Operator, PropertyGroup

from . import hidden_parts, library_state, prefs, scene_stats, thermal_view
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


#: irsim's component kinds (irsim.config.components.Kind) as the add-on's hidden-part kinds.
KINDS = {
    "motor": "MOTOR",
    "esc": "ESC",
    "battery": "BATTERY",
    "controller": "ELECTRONICS",
    "piston_engine": "PISTON_ENGINE",
    "turbine": "TURBINE",
    "exhaust": "EXHAUST",
    "gearbox": "GEARBOX",
}


def _host_of(centre, parts):
    """The smallest part whose box holds ``centre``: the part a hidden part sits in."""
    best, best_volume = None, None
    for ob in parts:
        lo, hi = scene_stats.world_bounds([ob])
        if all(lo[i] <= centre[i] <= hi[i] for i in range(3)):
            volume = float((hi - lo).prod())
            if best_volume is None or volume < best_volume:
                best, best_volume = ob, volume
    return best


def restore_structure(context, asset: str, result: dict) -> tuple[int, int]:
    """Bring back an asset config's hidden parts and contacts; ``(hidden, contacts)`` made."""
    objects = context.scene.objects
    parts = [ob for ob in scene_stats.part_objects(context) if ob.visible_get()]
    made_hidden = 0
    for h in result.get("hidden_parts", []):
        if h["name"] in objects:
            continue
        comp = library_state.find_component(context, h.get("component") or "")
        values = SimpleNamespace(
            kind=KINDS.get(comp.kind, "OTHER") if comp is not None else "OTHER",
            material=h.get("material") or "",
            irsim_component=h.get("component") or "",
            mass_kg=float(h.get("mass_kg") or 0.0),
            specific_heat_j_kgk=float(h.get("specific_heat_j_kgk") or 0.0),
            heat_idle_w=0.0,
            heat_rated_w=float(h.get("dissipation_w") or 0.0),
            values_source="MEASURED" if h.get("status") == "MEASURED" else "ESTIMATED",
            reference=f"configs/assets/{asset}.yaml",
        )
        ob = bpy.data.objects.new(h["name"], hidden_parts.make_box(h["name"], h["size_m"]))
        context.collection.objects.link(ob)
        ob.location = h["centre_m"]
        context.view_layer.update()
        hidden_parts.make_hidden(context, ob, _host_of(h["centre_m"], parts), values)
        made_hidden += 1
    connections = context.scene.irsim_connections
    made_contacts = 0
    for c in result.get("contacts", []):
        a, b = objects.get(c["a"]), objects.get(c["b"])
        if a is None or b is None:
            continue
        if any(x.kind == "CONTACT" and {x.a, x.b} == {a, b} for x in connections):
            continue
        connections.add()
        new = connections[len(connections) - 1]  # fetched again: a held item may move (B7)
        new.a, new.b, new.kind, new.status = a, b, "CONTACT", "CONFIRMED"
        new.joint, new.area_m2 = c["joint"], float(c["area_m2"])
        new.point_a = tuple(a.matrix_world.translation)
        new.point_b = tuple(b.matrix_world.translation)
        made_contacts += 1
    return made_hidden, made_contacts


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
    structure: BoolProperty(
        name="Hidden parts and contacts too",
        description=(
            "Also bring back the asset's hidden parts (as boxes) and its contacts between parts "
            "this scene has, from its parts block"
        ),
        default=True,
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
        layout.prop(self, "structure")

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
        if self.structure and (result.get("hidden_parts") or result.get("contacts")):
            made_hidden, made_contacts = restore_structure(context, self.asset, result)
            msg += f"; {made_hidden} hidden part(s) and {made_contacts} contact(s) brought back"
            scene_stats.refresh(context)
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
