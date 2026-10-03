"""Parts the model does not have: the engine inside a downloaded car shell, the battery in a drone.

A downloaded model is a skin. What heats it from inside -- an engine, a battery, a motor's
windings -- is usually not modelled at all, and a thermal camera sees exactly that heat. A hidden
part is an object placed where the real thing sits, carrying what the solver needs to know about
it: what it is made of (a library material, assigned like any part's), its mass and the heat it
makes at idle and at full load, and where those numbers came from.

A hidden part is a real object: in Blender a solid, drawn in front of the shell so it can always be
seen and grabbed; in the exported USD a prim of its own, which can be selected and moved in Isaac
Sim. Only the cameras skip it: the export marks its prim ``purpose = "guide"``, which Isaac Sim's
RTX renderer leaves out of every render product (checked 2026-09-28: a guide cube, and a cube
under a guide parent, are absent from both the colour image and the depth). It does not count
towards the material coverage. Its material, mass and heat also go into
``3d_models/<name>/<name>.structure.yaml`` (the asset config has no place for them yet, roadmap
AI.11), and it takes part in the connection finder like any other part, because an engine
touching the floor pan is exactly the contact the solver needs.

Two ways to add one: a box of any size (:class:`AddHiddenPart`), or a placeholder from the
component library in ``components/`` at its real size (:class:`AddComponent`), which a person can
later replace with a detailed model of their own.

The numbers are the person's. A library of components with cited values (roadmap AI.12) will
offer them ready-made; until then *Numbers are* says whether they were estimated, published or
measured, and *Reference* says from where.
"""

import json
import pathlib
import textwrap

import bmesh
import bpy
from bpy.props import EnumProperty, FloatProperty, FloatVectorProperty, StringProperty
from bpy.types import Operator
from mathutils import Matrix, Vector

from . import assign, library_state, naming, scene_stats
from .properties import COMPONENT_KINDS, VALUE_SOURCES

#: Object colour of a hidden part (shown when the viewport colours by object).
COLOUR = (1.0, 0.35, 0.1, 1.0)

#: The placeholder library: one FBX per component, listed in ``catalog.json``.
LIBRARY = pathlib.Path(__file__).resolve().parent / "components"


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


def draw_numbers(layout, op, context) -> None:
    """The component and the numbers, shared by both dialogs: a component's values show beside
    the fields, which then override it one at a time (0 keeps the component's)."""
    wm = context.window_manager
    layout.prop_search(op, "irsim_component", wm, "irsim_components")
    comp = library_state.find_component(context, op.irsim_component)
    if comp is not None:
        box = layout.box()
        for i, line in enumerate(textwrap.wrap(comp.description, 70)[:3]):
            box.label(text=line, icon="INFO" if i == 0 else "BLANK1")
        box.label(
            text=f"{comp.mass_kg:g} kg, {comp.specific_heat_j_kgk:g} J/kg·K, "
            f"heat {comp.dissipation_idle_w:g}–{comp.dissipation_rated_w:g} W ({comp.status})"
        )
        box.label(text="Numbers below override it; 0 keeps the component's")
    elif op.irsim_component:
        layout.label(text=f"{op.irsim_component!r} is not in irsim's components", icon="ERROR")
    layout.prop_search(op, "material", wm, "irsim_library")
    layout.separator()
    layout.prop(op, "mass_kg")
    layout.prop(op, "specific_heat_j_kgk")
    layout.prop(op, "heat_idle_w")
    layout.prop(op, "heat_rated_w")
    layout.prop(op, "values_source")
    layout.prop(op, "reference")


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
    irsim_component: StringProperty(
        name="irsim component",
        description="A component from irsim's library whose cited numbers this part takes",
    )
    mass_kg: FloatProperty(name="Mass", min=0.0, unit="MASS")
    specific_heat_j_kgk: FloatProperty(name="Specific heat (J/kg·K)", min=0.0)
    heat_idle_w: FloatProperty(name="Idle heat", min=0.0, unit="POWER")
    heat_rated_w: FloatProperty(name="Max heat", min=0.0, unit="POWER")
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
        draw_numbers(layout, self, context)
        host = context.active_object
        if host is not None and host.select_get():
            layout.label(text=f"Placed in the middle of {host.name}; moves with it", icon="INFO")
        else:
            layout.label(text="Placed at the 3D cursor", icon="INFO")

    def execute(self, context):
        if min(self.size) <= 0.0:
            self.report({"ERROR"}, "A hidden part needs a size")
            return {"CANCELLED"}
        host = _host(context)
        name = _free_name(self.part_name.strip() or self.kind.lower())
        ob = bpy.data.objects.new(name, make_box(name, Vector(self.size)))
        context.collection.objects.link(ob)
        ob.location = _centre(context, host)
        make_hidden(context, ob, host, self)
        self.report({"INFO"}, f"Added {name}: move it with G and resize it with S, as any object")
        return {"FINISHED"}


def _host(context):
    """The part a new hidden part goes inside: the selected, active mesh, if there is one."""
    ob = context.active_object
    return ob if ob is not None and ob in context.selected_objects and ob.type == "MESH" else None


def _centre(context, host) -> Vector:
    return world_box(host)[0] if host is not None else context.scene.cursor.location.copy()


def _free_name(wanted: str) -> str:
    base = naming.safe_identifier(wanted)
    return naming.unique_safe_names([base], taken=[ob.name for ob in bpy.data.objects]).get(
        base, base
    )


def make_hidden(context, ob, host, values) -> None:
    """Make ``ob`` a hidden part: visible and solid here, skipped by the cameras once exported.

    ``values`` carries ``kind``, ``material``, ``mass_kg``, ``heat_idle_w``, ``heat_rated_w``,
    ``values_source`` and ``reference`` (an operator, or anything with those attributes).
    """
    if host is not None:
        ob.parent = host
        ob.matrix_parent_inverse = host.matrix_world.inverted()
    show_as_hidden_part(ob)
    comp = ob.irsim_component
    comp.is_hidden_part = True
    for key in ("kind", "mass_kg", "heat_idle_w", "heat_rated_w", "values_source", "reference"):
        setattr(comp, key, getattr(values, key))
    comp.component = getattr(values, "irsim_component", "") or ""
    comp.specific_heat_j_kgk = float(getattr(values, "specific_heat_j_kgk", 0.0) or 0.0)
    material = values.material
    if not material and comp.component:  # the component's own case material, if it names one
        found = library_state.find_component(context, comp.component)
        if found is not None and library_state.find(context, found.material) is not None:
            material = found.material
    if material:
        ob.data.materials.clear()
        ob.data.materials.append(assign.plain_material(material))
    for other in context.selected_objects:
        other.select_set(False)
    ob.select_set(True)
    context.view_layer.objects.active = ob
    scene_stats.refresh(context)


def show_as_hidden_part(ob) -> None:
    """Solid, outlined and drawn in front of the shell, so it is never lost inside it.

    Also undoes the first version's look (a wireframe marked not to render): a hidden part must be
    exported, and the export skips what does not render.
    """
    ob.display_type = "SOLID"
    ob.show_wire = True
    ob.show_in_front = True
    ob.color = COLOUR
    ob.hide_render = False


def catalog() -> list[dict]:
    """The placeholder library's entries whose FBX is present."""
    path = LIBRARY / "catalog.json"
    if not path.is_file():
        return []
    entries = json.loads(path.read_text(encoding="utf-8"))["components"]
    return [e for e in entries if (LIBRARY / f"{e['file']}.fbx").is_file()]


_ITEMS: list = []  # Blender keeps only pointers into an enum callback's strings: hold them here


def _component_items(self, context):
    """The catalog as a menu: one column heading per category (drone, car, ship...)."""
    items, category = [], None
    for e in catalog():
        if e.get("category") != category:
            category = e.get("category")
            items.append(("", category or "Other", ""))
        items.append((e["file"], e["label"], e["note"]))
    _ITEMS[:] = items or [("NONE", "(library missing)", str(LIBRARY))]
    return _ITEMS


def entry(file: str) -> dict | None:
    for e in catalog():
        if e["file"] == file:
            return e
    return None


def placeholder_size(e: dict) -> str:
    """The catalog's placeholder size in words, e.g. ``0.14 × 0.07 × 0.04 m``."""
    s = e.get("size_m")
    if not s:
        return ""
    if e.get("shape", "box") == "box":
        return " × ".join(f"{x:g}" for x in s) + " m"
    ends = f"⌀ {s[1]:g}" + (f" → {s[2]:g}" if e["shape"] == "cone" else "")
    return f"{s[0]:g} m along {e.get('axis', 'X')}, {ends} m"


def _take_component(self, context) -> None:
    """Choosing a placeholder preselects the irsim component its catalog entry names, if any."""
    e = entry(self.component)
    self.irsim_component = (e or {}).get("component", "")


def import_component(context, file: str) -> bpy.types.Object:
    """Import ``components/<file>.fbx`` as one mesh object at its own size, at the world origin.

    A placeholder is one mesh; a detailed model someone drops in later may be several, which are
    joined into one part, and any empties the FBX brings are removed with their transforms kept.
    """
    before = set(bpy.data.objects)
    bpy.ops.import_scene.fbx(filepath=str(LIBRARY / f"{file}.fbx"))
    new = [ob for ob in bpy.data.objects if ob not in before]
    meshes = [ob for ob in new if ob.type == "MESH"]
    if not meshes:
        for ob in new:
            bpy.data.objects.remove(ob)
        raise RuntimeError(f"{file}.fbx holds no mesh")
    for ob in meshes:
        matrix = ob.matrix_world.copy()
        ob.parent = None
        ob.matrix_world = matrix
    for ob in new:
        if ob.type != "MESH":
            bpy.data.objects.remove(ob)
    if len(meshes) > 1:
        with context.temp_override(
            active_object=meshes[0], selected_editable_objects=meshes, selected_objects=meshes
        ):
            bpy.ops.object.join()
    ob = meshes[0]
    ob.location = (0.0, 0.0, 0.0)
    return ob


class AddComponent(Operator):
    """Add a part the model lacks from the component library, at its real size"""

    bl_idname = "irsim.add_component"
    bl_label = "Add from library"
    bl_options = {"REGISTER", "UNDO"}

    component: EnumProperty(name="Component", items=_component_items, update=_take_component)
    part_name: StringProperty(name="Name", description="Empty: the component's own name")
    material: StringProperty(
        name="Material", description="The library material its outside is made of"
    )
    irsim_component: StringProperty(
        name="irsim component",
        description="A component from irsim's library whose cited numbers this part takes",
    )
    mass_kg: FloatProperty(name="Mass", min=0.0, unit="MASS")
    specific_heat_j_kgk: FloatProperty(name="Specific heat (J/kg·K)", min=0.0)
    heat_idle_w: FloatProperty(name="Idle heat", min=0.0, unit="POWER")
    heat_rated_w: FloatProperty(name="Max heat", min=0.0, unit="POWER")
    values_source: EnumProperty(name="Numbers are", items=VALUE_SOURCES, default="ESTIMATED")
    reference: StringProperty(name="Reference")

    @classmethod
    def poll(cls, context):
        return context.mode == "OBJECT"

    @property
    def kind(self) -> str:
        e = entry(self.component)
        return e["kind"] if e else "OTHER"

    def invoke(self, context, event):
        if entry(self.component) is None:  # the first menu item is a category heading
            entries = catalog()
            if entries:
                self.component = entries[0]["file"]
        _take_component(self, context)
        return context.window_manager.invoke_props_dialog(
            self, width=460, title="Add a hidden part from the library", confirm_text="Add"
        )

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.prop(self, "component")
        e = entry(self.component)
        if e is not None:
            layout.label(text=f"{e.get('category', '')}: {e['note']}")
            layout.label(text=f"Placeholder {e.get('shape', 'box')}, {placeholder_size(e)}")
        layout.prop(self, "part_name")
        draw_numbers(layout, self, context)
        host = _host(context)
        where = f"in the middle of {host.name}; moves with it" if host else "at the 3D cursor"
        layout.label(text=f"Real size, placed {where}", icon="INFO")

    def execute(self, context):
        if self.component not in {e["file"] for e in catalog()}:
            self.report({"ERROR"}, f"No component {self.component!r} in {LIBRARY}")
            return {"CANCELLED"}
        host = _host(context)
        centre = _centre(context, host)
        ob = import_component(context, self.component)
        name = _free_name(self.part_name.strip() or self.component)
        ob.name = name
        ob.data.name = name
        ob.location = centre
        make_hidden(context, ob, host, self)
        self.report({"INFO"}, f"Added {name} from the library, at its real size")
        return {"FINISHED"}


class ShowHiddenParts(Operator):
    """Draw every hidden part solid and in front again, and make sure the export includes it"""

    bl_idname = "irsim.show_hidden_parts"
    bl_label = "Show hidden parts"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        parts = scene_stats.hidden_parts(context)
        for ob in parts:
            show_as_hidden_part(ob)
        self.report({"INFO"}, f"{len(parts)} hidden part(s) shown")
        return {"FINISHED"}


classes = (AddHiddenPart, AddComponent, ShowHiddenParts)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
