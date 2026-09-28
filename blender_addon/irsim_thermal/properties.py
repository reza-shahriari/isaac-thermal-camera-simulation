"""What the add-on stores, and where.

* **On each Blender material**: ``irsim_material``, the library material its faces are made of.
  The assignment lives on the *material*, not on the object, because that is what survives
  modelling: Blender carries a face's material slot through join, separate, extrude and
  duplicate, and ``configs/assets/<name>.yaml`` already maps a source material name to a library
  material (ADR 0128). A part is "carbon fibre" because its faces use a Blender material whose
  ``irsim_material`` is ``carbon_fibre``. When two parts share one Blender material but are made
  of different things, the add-on gives one of them a copy (same look, own thermal material) and
  records where the copy came from in ``irsim_origin``.
* **On the scene** (``Scene.irsim``): the asset name, the size-guidance choices and the last
  checklist result -- things that belong to this model and should be saved with it.
* **On the window manager** (``WindowManager.irsim_library``): the library as the bridge last
  reported it. Never saved: the library lives in the repository, and a stale copy in a ``.blend``
  would be a second library.
"""

import bpy
from bpy.props import (
    BoolProperty,
    CollectionProperty,
    EnumProperty,
    FloatProperty,
    IntProperty,
    PointerProperty,
    StringProperty,
)
from bpy.types import PropertyGroup

from . import sizing

BANDS = ("nir", "swir", "mwir", "lwir")


def _size_items(self, context):
    items = [("NONE", "Not set", "No guidance: just show the size")]
    for r in sizing.load_ranges():
        items.append((r.key, r.label, f"{r.min_m:g} to {r.max_m:g} m -- {r.note}"))
    return items


def _select_active_part(self, context):
    """Clicking a row in the parts list selects that object, like clicking it in the viewport."""
    objects = context.scene.objects
    if not 0 <= self.active_part_index < len(objects):
        return
    ob = objects[self.active_part_index]
    if context.mode != "OBJECT" or not ob.visible_get():
        return
    for other in context.selected_objects:
        other.select_set(False)
    ob.select_set(True)
    context.view_layer.objects.active = ob


def _toggle_thermal_view(self, context):
    from . import thermal_view

    thermal_view.on_toggle(self, context)


class IrsimLibraryItem(PropertyGroup):
    """One library material as the bridge reported it (``name`` is the library name)."""

    description: StringProperty()
    reference: StringProperty()
    source: StringProperty()
    surface_treatment: StringProperty()
    file: StringProperty()
    angular: StringProperty()
    spectral: BoolProperty()
    error: StringProperty(description="Why a band could not be evaluated, if one could not")

    eps_nir: FloatProperty()
    eps_swir: FloatProperty()
    eps_mwir: FloatProperty()
    eps_lwir: FloatProperty()
    rho_nir: FloatProperty()
    rho_swir: FloatProperty()
    rho_mwir: FloatProperty()
    rho_lwir: FloatProperty()
    tau_nir: FloatProperty()
    tau_swir: FloatProperty()
    tau_mwir: FloatProperty()
    tau_lwir: FloatProperty()

    solar_absorptivity: FloatProperty()
    density_kg_m3: FloatProperty()
    specific_heat_j_kgk: FloatProperty()
    conductivity_w_mk: FloatProperty()
    thickness_m: FloatProperty()
    heat_capacity_j_m2_k: FloatProperty()


class IrsimIssue(PropertyGroup):
    """One line of the checklist (``name`` is the part or material it is about)."""

    kind: EnumProperty(
        items=[
            ("UNASSIGNED", "Unassigned", "Faces with no thermal material"),
            ("MIRROR", "Mirror-like", "LWIR emissivity below the mirror limit"),
            ("NAME", "Name", "A default or unsafe name"),
            ("INFO", "Info", "For information"),
        ]
    )
    detail: StringProperty()
    area_m2: FloatProperty()


class IrsimSceneSettings(PropertyGroup):
    asset_name: StringProperty(
        name="Asset name",
        description=(
            "The name scene configs will use for this model: lower-case letters, digits and "
            "underscores. Written to configs/assets/<name>.yaml"
        ),
    )
    active_part_index: IntProperty(update=_select_active_part)
    active_library_index: IntProperty()
    active_issue_index: IntProperty()

    thermal_view: BoolProperty(
        name="Thermal view",
        description=(
            "Shade every part by its LWIR emissivity (bright = high), unassigned parts in "
            "magenta. Changes only the viewport colours, and restores them when switched off"
        ),
        default=False,
        update=_toggle_thermal_view,
    )

    size_help: BoolProperty(name="Help me check the size", default=False)
    size_kind: EnumProperty(name="What is it?", items=_size_items)
    known_dimension_m: FloatProperty(
        name="Real dimension",
        description="A published dimension of the real object, if you know one (0 = unknown)",
        default=0.0,
        min=0.0,
        subtype="DISTANCE",
        unit="LENGTH",
    )
    known_axis: EnumProperty(
        name="Measured along",
        items=[
            ("LARGEST", "Largest", "The model's largest dimension"),
            ("X", "X", "Width along X"),
            ("Y", "Y", "Depth along Y"),
            ("Z", "Z", "Height along Z"),
        ],
        default="LARGEST",
    )

    checked: BoolProperty(default=False)
    coverage: FloatProperty(subtype="PERCENTAGE", min=0.0, max=100.0)
    total_area_m2: FloatProperty()
    part_count: IntProperty()
    unassigned_count: IntProperty()
    size_x_m: FloatProperty()
    size_y_m: FloatProperty()
    size_z_m: FloatProperty()
    issues: CollectionProperty(type=IrsimIssue)

    replace_export: BoolProperty(
        name="Replace earlier export",
        description="Overwrite configs/assets/<name>.yaml if an earlier export wrote it",
        default=False,
    )
    last_export: StringProperty()
    audit_status: StringProperty()


classes = (IrsimLibraryItem, IrsimIssue, IrsimSceneSettings)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Material.irsim_material = StringProperty(
        name="Thermal material",
        description="The irsim library material these faces are made of (empty = not assigned)",
        default="",
    )
    bpy.types.Material.irsim_origin = StringProperty(
        name="Copied from",
        description=(
            "Set on a per-part copy: the material it was copied from so that it could carry a "
            "different thermal material"
        ),
        default="",
    )
    bpy.types.Scene.irsim = PointerProperty(type=IrsimSceneSettings)
    bpy.types.WindowManager.irsim_library = CollectionProperty(type=IrsimLibraryItem)
    bpy.types.WindowManager.irsim_library_status = StringProperty(default="")
    bpy.types.WindowManager.irsim_library_hash = StringProperty(default="")


def unregister():
    del bpy.types.WindowManager.irsim_library_hash
    del bpy.types.WindowManager.irsim_library_status
    del bpy.types.WindowManager.irsim_library
    del bpy.types.Scene.irsim
    del bpy.types.Material.irsim_origin
    del bpy.types.Material.irsim_material
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
