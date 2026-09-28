"""Parts the model does not have: the engine inside a downloaded car shell, the battery in a drone.

A downloaded model is a skin. What heats it from inside -- an engine, a battery, a motor's
windings -- is usually not modelled at all, and a thermal camera sees exactly that heat. A hidden
part is a **box** placed where the real thing sits, carrying what the solver needs to know about
it: what it is made of (a library material, assigned like any part's), its mass and the heat it
makes at idle and at full load, and where those numbers came from.

A hidden part is not geometry the camera renders: it is drawn as a wireframe in front of the
shell, is marked not to render, and is left out of the exported USD and of the coverage figure.
Its box, material and numbers go into ``3d_models/<name>/<name>.structure.yaml`` instead (the
asset config has no place for them yet, roadmap AI.11), and it takes part in the connection
finder like any other part, because an engine touching the floor pan is exactly the contact the
solver needs.

The numbers are the person's. A library of components with cited values (roadmap AI.12) will
offer them ready-made; until then *Numbers are* says whether they were estimated, published or
measured, and *Reference* says from where.
"""

import bmesh
import bpy
from bpy.props import EnumProperty, FloatProperty, FloatVectorProperty, StringProperty
from bpy.types import Operator
from mathutils import Matrix, Vector

from . import assign, naming, scene_stats
from .properties import COMPONENT_KINDS, VALUE_SOURCES

#: Wireframe colour of a hidden part in the viewport.
COLOUR = (1.0, 0.35, 0.1, 1.0)


def world_box(ob: bpy.types.Object) -> tuple[Vector, Vector]:
    """Centre and size (world metres, world axes) of an object's vertices."""
    lo, hi = scene_stats.world_bounds([ob])
    return Vector((lo + hi) / 2), Vector(hi - lo)


def box_record(ob: bpy.types.Object) -> dict:
    """The box as the structure file stores it: centre, size along its own axes, rotation."""
    _, rot, _ = ob.matrix_world.decompose()
    local_size = Vector(ob.dimensions)  # the box's own extent, scale included
    local_centre = sum((Vector(c) for c in ob.bound_box), Vector()) / 8.0
    centre = ob.matrix_world @ local_centre
    return {
        "centre_m": [round(float(x), 6) for x in centre],
        "size_m": [round(float(x), 6) for x in local_size],
        "rotation_wxyz": [round(float(x), 6) for x in rot],
    }


def make_box(name: str, size: Vector) -> bpy.types.Mesh:
    me = bpy.data.meshes.new(name)
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0, matrix=Matrix.Diagonal((*size, 1.0)))
    bm.to_mesh(me)
    bm.free()
    return me


class AddHiddenPart(Operator):
    """Add a part the model does not have, such as an engine inside a car shell, as a box"""

    bl_idname = "irsim.add_hidden_part"
    bl_label = "Add hidden part"
    bl_options = {"REGISTER", "UNDO"}

    part_name: StringProperty(name="Name", description="What the part is called, e.g. engine")
    kind: EnumProperty(name="What is it", items=COMPONENT_KINDS, default="OTHER")
    size: FloatVectorProperty(
        name="Size", size=3, subtype="XYZ_LENGTH", unit="LENGTH", default=(0.1, 0.1, 0.1), min=0.0
    )
    material: StringProperty(
        name="Material", description="The library material its outside is made of"
    )
    mass_kg: FloatProperty(name="Mass", min=0.0, unit="MASS")
    heat_idle_w: FloatProperty(name="Heat at idle", min=0.0, unit="POWER")
    heat_rated_w: FloatProperty(name="Heat at full load", min=0.0, unit="POWER")
    values_source: EnumProperty(name="Numbers are", items=VALUE_SOURCES, default="ESTIMATED")
    reference: StringProperty(name="Reference")

    @classmethod
    def poll(cls, context):
        return context.mode == "OBJECT"

    def invoke(self, context, event):
        # Start inside what is selected: a quarter of its size, at its middle.
        host = context.active_object if context.active_object in context.selected_objects else None
        if host is not None and host.type == "MESH":
            _, size = world_box(host)
            self.size = tuple(max(0.25 * s, 1e-3) for s in size)
        if not self.part_name:
            self.part_name = self.kind.lower()
        return context.window_manager.invoke_props_dialog(
            self, width=460, title="Add a hidden part", confirm_text="Add"
        )

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.prop(self, "part_name")
        layout.prop(self, "kind")
        layout.prop(self, "size")
        layout.prop_search(self, "material", context.window_manager, "irsim_library")
        layout.separator()
        layout.prop(self, "mass_kg")
        layout.prop(self, "heat_idle_w")
        layout.prop(self, "heat_rated_w")
        layout.prop(self, "values_source")
        layout.prop(self, "reference")
        host = context.active_object
        if host is not None and host.select_get():
            layout.label(text=f"Placed in the middle of {host.name}; moves with it", icon="INFO")
        else:
            layout.label(text="Placed at the 3D cursor", icon="INFO")

    def execute(self, context):
        if min(self.size) <= 0.0:
            self.report({"ERROR"}, "A hidden part needs a size")
            return {"CANCELLED"}
        taken = [ob.name for ob in bpy.data.objects]
        base = naming.safe_identifier(self.part_name.strip() or self.kind.lower())
        name = naming.unique_safe_names([base], taken=taken).get(base, base)
        host = context.active_object if context.active_object in context.selected_objects else None
        centre = world_box(host)[0] if host is not None and host.type == "MESH" else None
        if centre is None:
            centre = context.scene.cursor.location.copy()

        ob = bpy.data.objects.new(name, make_box(name, Vector(self.size)))
        context.collection.objects.link(ob)
        ob.location = centre
        if host is not None:
            ob.parent = host
            ob.matrix_parent_inverse = host.matrix_world.inverted()
        ob.display_type = "WIRE"
        ob.show_in_front = True
        ob.color = COLOUR
        ob.hide_render = True
        comp = ob.irsim_component
        comp.is_hidden_part = True
        comp.kind = self.kind
        comp.mass_kg = self.mass_kg
        comp.heat_idle_w = self.heat_idle_w
        comp.heat_rated_w = self.heat_rated_w
        comp.values_source = self.values_source
        comp.reference = self.reference
        if self.material:
            ob.data.materials.append(assign.plain_material(self.material))

        for other in context.selected_objects:
            other.select_set(False)
        ob.select_set(True)
        context.view_layer.objects.active = ob
        scene_stats.refresh(context)
        self.report({"INFO"}, f"Added {name}: move it with G and resize it with S, as any object")
        return {"FINISHED"}


classes = (AddHiddenPart,)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
