"""Give a part, a set of faces, or a Blender material its infrared material.

The thermal material is stored on the Blender material (see :mod:`.properties` for why). The one
subtle case is a Blender material shared by parts that are made of different things -- a
download's "white_plastic" on the shell *and* the propellers, when the propellers are really
carbon fibre. Changing the shared material would silently re-assign the shell, so the selected
part is given a **copy** instead: same look, its own thermal material, named
``white_plastic__carbon_fibre`` and remembering its origin. Parts that were not selected are never
changed by an assignment to a part.
"""

from collections import Counter, defaultdict

import bmesh
import bpy
from bpy.props import EnumProperty, StringProperty
from bpy.types import Operator

from . import naming, scene_stats, thermal_view
from .library_state import find as find_library_item


def material_users() -> dict[bpy.types.Material, set[bpy.types.Object]]:
    """Every object in the file using each material, across all scenes."""
    users: dict[bpy.types.Material, set[bpy.types.Object]] = defaultdict(set)
    for ob in bpy.data.objects:
        for slot in getattr(ob, "material_slots", ()):
            if slot.material is not None:
                users[slot.material].add(ob)
    return users


def plain_material(thermal: str) -> bpy.types.Material:
    """A neutral grey material for faces that had none, carrying ``thermal``."""
    name = f"irsim_{thermal}"
    mat = bpy.data.materials.get(name)
    if mat is not None and mat.irsim_material == thermal:
        return mat
    mat = bpy.data.materials.new(name)
    mat.irsim_material = thermal
    mat.diffuse_color = (0.6, 0.6, 0.6, 1.0)
    return mat


def split_copy(mat: bpy.types.Material, thermal: str) -> bpy.types.Material:
    """The copy of ``mat`` that carries ``thermal``, reusing one made earlier."""
    origin = mat.irsim_origin or mat.name
    for other in bpy.data.materials:
        if other.irsim_material == thermal and (
            other.irsim_origin == origin or (other.name == origin and not other.irsim_origin)
        ):
            return other
    copy = mat.copy()
    copy.name = naming.split_name(origin, thermal)
    copy.irsim_origin = origin
    copy.irsim_material = thermal
    return copy


def _mesh_users(mesh: bpy.types.Mesh) -> set[bpy.types.Object]:
    return {ob for ob in bpy.data.objects if ob.data is mesh}


def _set_slot(ob: bpy.types.Object, index: int, mat, selection: set) -> None:
    """Put ``mat`` in one slot of ``ob`` without touching an unselected object sharing its mesh."""
    slot = ob.material_slots[index]
    if slot.link == "DATA" and not _mesh_users(ob.data) <= selection:
        slot.link = "OBJECT"
    slot.material = mat


def assign_to_parts(objects, thermal: str) -> dict[str, list[str]]:
    """Assign ``thermal`` to every face of every object in ``objects``. Returns what changed."""
    report: dict[str, list[str]] = {"set": [], "copied": [], "created": []}
    selection = {ob for ob in objects if ob.type == "MESH"}
    users = material_users()
    for ob in sorted(selection, key=lambda o: o.name):
        if not ob.material_slots:
            mat = plain_material(thermal)
            if _mesh_users(ob.data) <= selection:
                ob.data.materials.append(mat)
            else:
                ob.data.materials.append(None)
                ob.material_slots[-1].link = "OBJECT"
                ob.material_slots[-1].material = mat
            report["created"].append(f"{ob.name}: {mat.name}")
            continue
        for index, slot in enumerate(ob.material_slots):
            mat = slot.material
            if mat is None:
                new = plain_material(thermal)
                _set_slot(ob, index, new, selection)
                report["created"].append(f"{ob.name}: {new.name}")
                continue
            if mat.irsim_material == thermal:
                continue
            if users.get(mat, set()) <= selection:
                mat.irsim_material = thermal
                report["set"].append(mat.name)
            else:
                copy = split_copy(mat, thermal)
                _set_slot(ob, index, copy, selection)
                report["copied"].append(f"{mat.name} -> {copy.name}")
    return report


def assign_to_faces(objects, thermal: str) -> dict[str, list[str]]:
    """Assign ``thermal`` to the selected faces of objects in edit mode."""
    report: dict[str, list[str]] = {"set": [], "copied": [], "created": []}
    users = material_users()
    seen_meshes = set()
    for ob in objects:
        if ob.type != "MESH" or ob.data in seen_meshes:
            continue
        seen_meshes.add(ob.data)
        me = ob.data
        bm = bmesh.from_edit_mesh(me)
        selected = [f for f in bm.faces if f.select]
        if not selected:
            continue
        per_slot = Counter(f.material_index for f in bm.faces)
        groups: dict[int, list] = defaultdict(list)
        for f in selected:
            groups[f.material_index].append(f)
        sharers = _mesh_users(me)
        for index, faces in sorted(groups.items()):
            mat = ob.material_slots[index].material if index < len(ob.material_slots) else None
            if mat is not None and mat.irsim_material == thermal:
                continue
            whole_slot = len(faces) == per_slot[index]
            only_here = mat is not None and users.get(mat, set()) <= sharers
            in_one_slot = sum(1 for s in ob.material_slots if s.material == mat) == 1
            if mat is not None and whole_slot and only_here and in_one_slot:
                mat.irsim_material = thermal
                report["set"].append(mat.name)
                continue
            new = split_copy(mat, thermal) if mat is not None else plain_material(thermal)
            target = next((i for i, s in enumerate(ob.material_slots) if s.material == new), None)
            if target is None:
                me.materials.append(new)
                target = len(ob.material_slots) - 1
                report["created" if mat is None else "copied"].append(
                    f"{ob.name}: {mat.name if mat else '(none)'} -> {new.name}"
                )
            for f in faces:
                f.material_index = target
        bmesh.update_edit_mesh(me)
    return report


def _after_change(context: bpy.types.Context) -> None:
    if context.mode == "OBJECT":
        scene_stats.refresh(context)
    if context.scene.irsim.thermal_view:
        thermal_view.apply(context)


def _describe(report: dict[str, list[str]]) -> str:
    bits = []
    if report["set"]:
        bits.append(f"set on {len(report['set'])} material(s)")
    if report["copied"]:
        bits.append(f"{len(report['copied'])} per-part copy/copies made")
    if report["created"]:
        bits.append(f"{len(report['created'])} new material(s) for faces that had none")
    return ", ".join(bits) or "already assigned"


class AssignThermal(Operator):
    """Give the selection this infrared material"""

    bl_idname = "irsim.assign"
    bl_label = "Assign thermal material"
    bl_options = {"REGISTER", "UNDO"}

    material: StringProperty(name="Library material")
    scope: EnumProperty(
        name="Assign to",
        items=[
            ("PARTS", "Selected parts", "Every face of every selected object"),
            ("FACES", "Selected faces", "Only the faces selected in Edit Mode"),
            (
                "MATERIAL",
                "Every part using the active material",
                "The active object's active Blender material, wherever it is used",
            ),
        ],
        default="PARTS",
    )

    @classmethod
    def poll(cls, context):
        return context.mode in {"OBJECT", "EDIT_MESH"}

    def execute(self, context):
        if not self.material or find_library_item(context, self.material) is None:
            self.report({"ERROR"}, f"{self.material!r} is not in the loaded library")
            return {"CANCELLED"}
        if self.scope == "FACES":
            if context.mode != "EDIT_MESH":
                self.report({"ERROR"}, "Selected faces: enter Edit Mode and select faces first")
                return {"CANCELLED"}
            report = assign_to_faces(context.objects_in_mode_unique_data, self.material)
        elif self.scope == "MATERIAL":
            ob = context.active_object
            mat = ob.active_material if ob else None
            if mat is None:
                self.report({"ERROR"}, "The active object has no active material")
                return {"CANCELLED"}
            n_users = len(material_users().get(mat, ()))
            mat.irsim_material = self.material
            report = {"set": [mat.name], "copied": [], "created": []}
            self.report({"INFO"}, f"{mat.name} ({n_users} part(s)) is now {self.material}")
        else:
            parts = [ob for ob in context.selected_objects if ob.type == "MESH"]
            if not parts:
                self.report({"ERROR"}, "Select one or more parts (mesh objects) first")
                return {"CANCELLED"}
            report = assign_to_parts(parts, self.material)
        _after_change(context)
        if self.scope != "MATERIAL":
            self.report({"INFO"}, f"{self.material}: {_describe(report)}")
        return {"FINISHED"}


class ClearThermal(Operator):
    """Remove the infrared material from the selected parts' materials"""

    bl_idname = "irsim.clear"
    bl_label = "Clear thermal material"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return context.mode == "OBJECT" and bool(context.selected_objects)

    def execute(self, context):
        selection = {ob for ob in context.selected_objects if ob.type == "MESH"}
        users = material_users()
        cleared, others = set(), set()
        for ob in selection:
            for slot in ob.material_slots:
                if slot.material is not None and slot.material.irsim_material:
                    slot.material.irsim_material = ""
                    cleared.add(slot.material.name)
                    others |= {o.name for o in users.get(slot.material, set()) - selection}
        _after_change(context)
        msg = f"Cleared {len(cleared)} material(s)"
        if others:
            msg += f"; also cleared on {len(others)} unselected part(s) sharing them"
        self.report({"WARNING" if others else "INFO"}, msg)
        return {"FINISHED"}


class SelectParts(Operator):
    """Select parts by their thermal state"""

    bl_idname = "irsim.select_parts"
    bl_label = "Select parts"
    bl_options = {"REGISTER", "UNDO"}

    which: EnumProperty(
        items=[
            ("UNASSIGNED", "Unassigned", "Parts with any face that has no thermal material"),
            ("USING", "Using material", "Parts with any face made of the given material"),
        ]
    )
    material: StringProperty()

    @classmethod
    def poll(cls, context):
        return context.mode == "OBJECT"

    def execute(self, context):
        for ob in context.selected_objects:
            ob.select_set(False)
        chosen = []
        for ob in scene_stats.part_objects(context):
            stats = scene_stats.part_stats(ob)
            hit = (
                not stats.fully_assigned
                if self.which == "UNASSIGNED"
                else self.material in stats.by_thermal_m2
            )
            if hit and ob.visible_get():
                ob.select_set(True)
                chosen.append(ob)
        if chosen:
            context.view_layer.objects.active = chosen[0]
        self.report({"INFO"}, f"Selected {len(chosen)} part(s)")
        return {"FINISHED"}


class SelectIssue(Operator):
    """Select the part this checklist line is about"""

    bl_idname = "irsim.select_issue"
    bl_label = "Select"
    bl_options = {"REGISTER", "UNDO"}

    name: StringProperty()

    def execute(self, context):
        ob = context.scene.objects.get(self.name)
        if ob is None or not ob.visible_get():
            self.report({"WARNING"}, f"{self.name} is not a visible object in this scene")
            return {"CANCELLED"}
        if context.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
        for other in context.selected_objects:
            other.select_set(False)
        ob.select_set(True)
        context.view_layer.objects.active = ob
        return {"FINISHED"}


classes = (AssignThermal, ClearThermal, SelectParts, SelectIssue)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
