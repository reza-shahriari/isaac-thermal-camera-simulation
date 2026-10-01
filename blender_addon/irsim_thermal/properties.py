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
* **On the scene** too: ``Scene.irsim_connections``, the contacts and facing pairs between parts,
  each with who proposed it (the finder or the person) and whether the person has reviewed it.
  A connection points at its two objects, so renaming a part keeps it; deleting one leaves a
  connection the checklist reports.
* **On each hidden part** (``Object.irsim_component``): what it is (motor, battery, engine...),
  its mass and heat output, and where those numbers came from. A hidden part is a box mesh the
  camera never sees -- the engine inside a downloaded car shell, the battery inside a drone.
* **On the window manager** (``WindowManager.irsim_library``, ``irsim_joints``): the library and
  the joint table as the bridge last reported them. Never saved: both live in the repository, and
  a stale copy in a ``.blend`` would be a second library.
"""

import bpy
from bpy.props import (
    BoolProperty,
    CollectionProperty,
    EnumProperty,
    FloatProperty,
    FloatVectorProperty,
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


def _select_active_part_hidden(self, context):
    """The hidden-parts list selects its row's object, like the parts list does."""
    objects = context.scene.objects
    index = self.active_hidden_index
    if 0 <= index < len(objects) and context.mode == "OBJECT":
        ob = objects[index]
        for other in context.selected_objects:
            other.select_set(False)
        if ob.visible_get():
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
    #: ADR 0175: which forms the material authors -- curve, per band, grey -- as one line.
    forms: StringProperty()
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
    #: The share of each band's value the curve supplied (`BandProperties.curve_fraction`).
    curve_nir: FloatProperty()
    curve_swir: FloatProperty()
    curve_mwir: FloatProperty()
    curve_lwir: FloatProperty()

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
            ("HIDDEN", "Hidden part", "A hidden part missing a material, a mass or its heat"),
            ("CONNECTION", "Connection", "A connection to review, or one whose part is gone"),
            ("INFO", "Info", "For information"),
        ]
    )
    detail: StringProperty()
    area_m2: FloatProperty()


def _redraw_overlay(self, context):
    from . import overlay

    overlay.on_toggle(self, context)


class IrsimJoint(PropertyGroup):
    """One entry of ``configs/thermal/joints.yaml`` (``name`` is the joint name)."""

    h_c_w_m2_k: FloatProperty()
    status: StringProperty()
    source: StringProperty()


CONNECTION_KINDS = [
    ("CONTACT", "Contact", "The two parts touch: heat is conducted through a joint"),
    ("FACING", "Facing", "The two parts look at each other across a gap: heat is radiated"),
]

CONNECTION_STATUS = [
    ("FOUND", "Found", "Proposed by the connection finder, not reviewed yet"),
    ("CONFIRMED", "Confirmed", "Reviewed and kept"),
    ("REJECTED", "Rejected", "Reviewed and ruled out: not exported, and a new search keeps it out"),
    ("ADDED", "Added by hand", "Added by a person"),
]


class IrsimConnection(PropertyGroup):
    """A heat path between two parts, found or added, and the person's verdict on it."""

    a: PointerProperty(type=bpy.types.Object, name="Part")
    b: PointerProperty(type=bpy.types.Object, name="Other part")
    kind: EnumProperty(name="Kind", items=CONNECTION_KINDS, default="CONTACT")
    status: EnumProperty(name="Status", items=CONNECTION_STATUS, default="FOUND")
    joint: StringProperty(
        name="Joint",
        description="How the two parts are joined: a contact conductance from "
        "configs/thermal/joints.yaml",
        default="",
    )
    area_m2: FloatProperty(
        name="Area",
        description=(
            "Contact: the area where the parts touch. Facing: the area of the first part that "
            "looks straight at the second"
        ),
        min=0.0,
        precision=6,
        unit="AREA",
    )
    area_ba_m2: FloatProperty(
        name="Area back",
        description="Facing: the area of the second part that looks straight at the first",
        min=0.0,
        precision=6,
        unit="AREA",
    )
    gap_m: FloatProperty(
        name="Gap", description="Facing: the mean distance across", min=0.0, unit="LENGTH"
    )
    point_a: FloatVectorProperty(size=3)
    point_b: FloatVectorProperty(size=3)


COMPONENT_KINDS = [
    ("MOTOR", "Electric motor", "A brushless motor"),
    ("ESC", "Speed controller", "An electronic speed controller"),
    ("BATTERY", "Battery", "A battery pack"),
    ("ELECTRONICS", "Electronics", "A flight controller, computer or radio"),
    ("PISTON_ENGINE", "Piston engine", "A car, boat or light-aircraft engine"),
    ("TURBINE", "Turbine", "A gas turbine or jet engine"),
    ("EXHAUST", "Exhaust", "An exhaust pipe, silencer or catalyst"),
    ("GEARBOX", "Gearbox", "A gearbox or drive train"),
    ("OTHER", "Other", "Anything else that makes or stores heat"),
]

VALUE_SOURCES = [
    ("ESTIMATED", "Estimated", "An engineering guess; say from what in the reference"),
    ("PUBLISHED", "Published", "From a datasheet or a paper; give it as the reference"),
    ("MEASURED", "Measured", "Measured on the real object"),
]


class IrsimComponent(PropertyGroup):
    """What a hidden part is and how much heat it makes. Set only on hidden parts."""

    is_hidden_part: BoolProperty(default=False)
    kind: EnumProperty(name="What is it", items=COMPONENT_KINDS, default="OTHER")
    mass_kg: FloatProperty(
        name="Mass", description="Its mass: it sets how slowly it warms up", min=0.0, unit="MASS"
    )
    heat_idle_w: FloatProperty(
        name="Idle heat",
        description="Heat it gives off when running lightly (W)",
        min=0.0,
        unit="POWER",
    )
    heat_rated_w: FloatProperty(
        name="Max heat",
        description="Heat it gives off at its rated load (W): input power minus useful work",
        min=0.0,
        unit="POWER",
    )
    values_source: EnumProperty(name="Numbers are", items=VALUE_SOURCES, default="ESTIMATED")
    reference: StringProperty(
        name="Reference", description="Where the numbers came from, or what they were guessed from"
    )


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

    active_connection_index: IntProperty()
    active_hidden_index: IntProperty(update=_select_active_part_hidden)
    touch_gap_m: FloatProperty(
        name="Touching within",
        description="Surfaces closer than this count as touching (0 = 0.2 % of the model)",
        default=0.0,
        min=0.0,
        unit="LENGTH",
    )
    facing_gap_m: FloatProperty(
        name="Facing within",
        description=(
            "Parts that look straight at each other across less than this count as facing "
            "(0 = 10 % of the model)"
        ),
        default=0.0,
        min=0.0,
        unit="LENGTH",
    )
    samples: IntProperty(
        name="Samples per part",
        description="Points scattered on each part; more is slower and more exact",
        default=3000,
        min=100,
        max=100000,
    )
    last_search: StringProperty(
        description="The distances the last connection search used, as the panel shows them"
    )
    connections_of_selected: BoolProperty(
        name="Only the selected parts'",
        description=(
            "List and draw only the connections of the selected parts (and draw the one picked "
            "in the list); with nothing selected, all of them. A model has hundreds"
        ),
        default=True,
        update=_redraw_overlay,
    )
    show_connections: BoolProperty(
        name="Show in viewport",
        description="Draw each connection as a line between its parts",
        default=False,
        update=_redraw_overlay,
    )


classes = (
    IrsimLibraryItem,
    IrsimIssue,
    IrsimJoint,
    IrsimConnection,
    IrsimComponent,
    IrsimSceneSettings,
)


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
    bpy.types.Scene.irsim_connections = CollectionProperty(type=IrsimConnection)
    bpy.types.Object.irsim_component = PointerProperty(type=IrsimComponent)
    bpy.types.WindowManager.irsim_joints = CollectionProperty(type=IrsimJoint)
    bpy.types.WindowManager.irsim_library = CollectionProperty(type=IrsimLibraryItem)
    bpy.types.WindowManager.irsim_library_status = StringProperty(default="")
    bpy.types.WindowManager.irsim_library_hash = StringProperty(default="")
    # The last plot drawn for the panel (B11): which material, in which Blender image.
    bpy.types.WindowManager.irsim_plot_material = StringProperty(default="")
    bpy.types.WindowManager.irsim_plot_image = StringProperty(default="")
    # The last picked curve the bridge checked, as one line for the new-material form.
    bpy.types.WindowManager.irsim_curve_check = StringProperty(default="")


def unregister():
    del bpy.types.WindowManager.irsim_curve_check
    del bpy.types.WindowManager.irsim_plot_image
    del bpy.types.WindowManager.irsim_plot_material
    del bpy.types.WindowManager.irsim_library_hash
    del bpy.types.WindowManager.irsim_library_status
    del bpy.types.WindowManager.irsim_library
    del bpy.types.WindowManager.irsim_joints
    del bpy.types.Object.irsim_component
    del bpy.types.Scene.irsim_connections
    del bpy.types.Scene.irsim
    del bpy.types.Material.irsim_origin
    del bpy.types.Material.irsim_material
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
