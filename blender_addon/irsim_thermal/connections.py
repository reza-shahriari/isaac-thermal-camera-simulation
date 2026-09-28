"""Find which parts touch and which face each other, and let the person review the result.

The finder proposes; the person decides. Every connection it finds is listed as *found*. The
person confirms it, rejects it, changes its joint or its area, or adds one the finder missed.
Running the finder again refreshes what it found but never overrides a verdict: a confirmed
connection stays confirmed, and a rejected one stays out.

How it looks (:mod:`.geometry` for the arithmetic):

* **Contact.** Points are scattered by area over the part of A that lies near B, and over the part
  of B near A. A point within the touching distance of the other part's surface, where the two
  surfaces lie roughly parallel (:data:`geometry.PARALLEL_COS`), is touching. The
  contact area is the smaller of the two estimates: both sides of one interface should agree, and
  the smaller one is the one not inflated by geometry that runs into the other part.
* **Facing.** From points scattered over the whole of A, a ray goes out along the surface normal.
  The part it reaches first within the facing distance, excluding A itself and anything closer
  than the touching distance, is what that bit of A looks at. That is how an engine block is found
  to face the bonnet above it and the floor pan below it, with nothing touching.

A part here is anything :func:`scene_stats.part_objects` exports, plus the hidden parts: a hidden
engine touching the floor pan is exactly the contact the solver needs.
"""

import time

import bpy
import numpy as np
from bpy.props import EnumProperty, IntProperty
from bpy.types import Operator
from mathutils import Vector
from mathutils.bvhtree import BVHTree

from . import geometry, scene_stats
from .properties import CONNECTION_KINDS

DEFAULT_JOINT = "dry_default"


def connectable_objects(context: bpy.types.Context) -> list[bpy.types.Object]:
    """The exported parts and the hidden parts."""
    parts = scene_stats.part_objects(context)
    return parts + [ob for ob in scene_stats.hidden_parts(context) if ob not in parts]


def world_surface(ob: bpy.types.Object, depsgraph) -> geometry.Surface:
    """The part as exported (modifiers applied), triangulated, in world metres."""
    ob_eval = ob.evaluated_get(depsgraph)
    me = ob_eval.to_mesh()
    try:
        me.calc_loop_triangles()
        co = np.empty(len(me.vertices) * 3, dtype=np.float32)
        me.vertices.foreach_get("co", co)
        tris = np.empty(len(me.loop_triangles) * 3, dtype=np.int32)
        me.loop_triangles.foreach_get("vertices", tris)
        m = np.array(ob_eval.matrix_world, dtype=np.float64)
    finally:
        ob_eval.to_mesh_clear()
    world = co.reshape(-1, 3).astype(np.float64) @ m[:3, :3].T + m[:3, 3]
    return geometry.Surface(world, tris.reshape(-1, 3))


def _tree(surface: geometry.Surface) -> BVHTree:
    return BVHTree.FromPolygons(surface.vertices.tolist(), surface.triangles.tolist())


def _near_region(surface: geometry.Surface, lo, hi, margin) -> geometry.Surface:
    """The triangles of ``surface`` whose own box reaches into ``[lo, hi]`` grown by ``margin``."""
    a, b, c = surface.corners()
    t_lo = np.minimum(np.minimum(a, b), c)
    t_hi = np.maximum(np.maximum(a, b), c)
    keep = np.all((t_hi >= lo - margin) & (t_lo <= hi + margin), axis=1)
    return geometry.Surface(surface.vertices, surface.triangles[keep])


class Finder:
    """One search over a set of parts: their surfaces, and each one's tree, built once if needed."""

    def __init__(self, objects, depsgraph, touch_m: float, facing_m: float, samples: int):
        self.objects = list(objects)
        self.touch = touch_m
        self.facing = max(facing_m, touch_m)
        self.samples = samples
        self.surfaces = {ob.name: world_surface(ob, depsgraph) for ob in self.objects}
        self.bounds = {name: s.bounds() for name, s in self.surfaces.items() if len(s.triangles)}
        self.areas = {name: s.area for name, s in self.surfaces.items()}
        self._trees: dict[str, BVHTree] = {}

    def tree(self, name: str) -> BVHTree:
        if name not in self._trees:
            self._trees[name] = _tree(self.surfaces[name])
        return self._trees[name]

    def neighbours(self, name: str, margin: float) -> list[str]:
        lo, hi = self.bounds[name]
        return [
            other
            for other, (olo, ohi) in self.bounds.items()
            if other != name and geometry.boxes_overlap(lo, hi, olo, ohi, margin)
        ]

    def one_side_contact(self, a: str, b: str, seed: int) -> tuple[float, np.ndarray | None]:
        """Area of A within the touching distance of B, and the mean point of that area."""
        lo, hi = self.bounds[b]
        region = _near_region(self.surfaces[a], lo, hi, self.touch)
        if not len(region.triangles):
            return 0.0, None
        pts = geometry.sample(region, self.samples, seed)
        tree = self.tree(b)
        found = np.zeros(len(pts), dtype=bool)
        other = np.zeros((len(pts), 3))
        offsets = np.zeros((len(pts), 3))
        for i, p in enumerate(pts.points):
            loc, normal, _, _ = tree.find_nearest(Vector(p), self.touch)
            if loc is not None:
                found[i] = True
                other[i] = normal
                offsets[i] = np.array(loc) - p
        near = geometry.touching(found, pts.normals, other, offsets, self.touch)
        if not near.any():
            return 0.0, None
        return geometry.contact_area(pts, near), pts.points[near].mean(axis=0)

    def contact(self, a: str, b: str) -> tuple[float, np.ndarray | None]:
        ab, point = self.one_side_contact(a, b, seed=1)
        if not ab:
            return 0.0, None
        ba, _ = self.one_side_contact(b, a, seed=2)
        return min(ab, ba), point

    def facing_from(self, a: str) -> dict[str, tuple[float, float, np.ndarray, np.ndarray]]:
        """``{b: (area of A looking at B, mean gap, mean start, mean hit)}`` for A's neighbours."""
        others = self.neighbours(a, self.facing)
        if not others:
            return {}
        pts = geometry.sample(self.surfaces[a], self.samples, seed=3)
        trees = [(name, self.tree(name)) for name in [a, *others]]
        offset = 1e-3 * self.touch
        hits: dict[str, list[tuple[float, np.ndarray, np.ndarray]]] = {}
        for p, n in zip(pts.points, pts.normals, strict=True):
            if not n.any():
                continue
            # A point in contact with another part conducts rather than radiates, and its ray
            # would start inside that part and report the far side of it as "facing".
            if any(
                tree.find_nearest(Vector(p), self.touch)[0] is not None
                for name, tree in trees
                if name != a
            ):
                continue
            origin = Vector(p + offset * n)
            direction = Vector(n)
            best = (self.facing, None, None)
            for name, tree in trees:
                loc, _, _, dist = tree.ray_cast(origin, direction, best[0])
                if loc is not None and dist < best[0]:
                    best = (dist, name, np.array(loc))
            dist, name, loc = best
            if name is None or name == a or dist < self.touch:
                continue
            hits.setdefault(name, []).append((dist, p, loc))
        out = {}
        for name, rows in hits.items():
            gaps = np.array([r[0] for r in rows])
            out[name] = (
                len(rows) * pts.area_each,
                float(gaps.mean()),
                np.mean([r[1] for r in rows], axis=0),
                np.mean([r[2] for r in rows], axis=0),
            )
        return out

    def run(self) -> list[dict]:
        """Every contact and facing pair worth showing, as plain records."""
        found: list[dict] = []
        names = sorted(self.bounds)
        for i, a in enumerate(names):
            for b in names[i + 1 :]:
                if b not in self.neighbours(a, self.touch):
                    continue
                area, point = self.contact(a, b)
                # At least a few samples' worth, so one stray point is not a contact.
                if point is not None and area >= self._floor(a, b):
                    found.append(
                        {"kind": "CONTACT", "a": a, "b": b, "area": area, "pa": point, "pb": point}
                    )
        facing = {a: self.facing_from(a) for a in names}
        for i, a in enumerate(names):
            for b in names[i + 1 :]:
                ab = facing[a].get(b)
                ba = facing[b].get(a)
                if ab is None and ba is None:
                    continue
                area_ab = ab[0] if ab else 0.0
                area_ba = ba[0] if ba else 0.0
                # 1 % of the smaller part: below that, the pair exchanges too little to list.
                if max(area_ab, area_ba) < max(
                    self._floor(a, b), 0.01 * min(self.areas[a], self.areas[b])
                ):
                    continue
                gaps = [g[1] for g in (ab, ba) if g]
                pa = ab[2] if ab else ba[3]
                pb = ab[3] if ab else ba[2]
                found.append(
                    {
                        "kind": "FACING",
                        "a": a,
                        "b": b,
                        "area": area_ab,
                        "area_ba": area_ba,
                        "gap": float(np.mean(gaps)),
                        "pa": pa,
                        "pb": pb,
                    }
                )
        return found

    def _floor(self, a: str, b: str) -> float:
        """Three samples' worth of the smaller part: the least area told apart from none."""
        return 3.0 * min(self.areas[a], self.areas[b]) / self.samples


def gaps_for(context: bpy.types.Context, objects) -> tuple[float, float]:
    """The touching and facing distances: as set on the scene, or the defaults for its size."""
    settings = context.scene.irsim
    largest = max(scene_stats.world_extent(objects)) if objects else 0.0
    touch, facing = geometry.default_gaps(largest)
    return settings.touch_gap_m or touch, settings.facing_gap_m or facing


def _key(a, b, kind: str) -> tuple:
    return (frozenset((a, b)), kind)


def merge(connections, found: list[dict], objects: dict[str, bpy.types.Object], joints) -> dict:
    """Apply a new search to the list, keeping every verdict the person has given.

    * a *found* entry that the new search did not find again is dropped;
    * a *confirmed*, *rejected* or *added* entry is never dropped or re-proposed, and a confirmed
      or rejected one gets the new measurement;
    * a new entry starts as *found*, a contact with the table's dry default joint.

    No Python reference to an entry is held across a ``remove`` or an ``add``: Blender may move a
    collection's items when it changes, and writing through an old reference then writes into
    freed memory (a crash, found by the smoke test). So entries are always looked up again by
    index, and every update to an existing entry happens before any new one is added.
    """
    for index in reversed(range(len(connections))):
        if connections[index].status == "FOUND":
            connections.remove(index)

    def key_of(c) -> tuple | None:
        return None if c.a is None or c.b is None else _key(c.a.name, c.b.name, c.kind)

    existing = {key_of(c): i for i, c in enumerate(connections) if key_of(c) is not None}
    new_records = []
    updated = 0
    for record in found:
        index = existing.get(_key(record["a"], record["b"], record["kind"]))
        if index is None:
            new_records.append(record)
        elif connections[index].status != "ADDED":
            _measure(connections[index], record)
            updated += 1
    for record in new_records:
        connections.add()
        c = connections[len(connections) - 1]
        c.a, c.b = objects[record["a"]], objects[record["b"]]
        c.kind = record["kind"]
        c.status = "FOUND"
        if c.kind == "CONTACT" and DEFAULT_JOINT in joints:
            c.joint = DEFAULT_JOINT
        _measure(c, record)
    return {"added": len(new_records), "updated": updated}


def _measure(c, record: dict) -> None:
    """Write a search's measurement onto a connection, whichever way round it stores its parts."""
    swapped = c.a.name != record["a"]
    c.area_m2 = record["area_ba"] if swapped and "area_ba" in record else record["area"]
    c.area_ba_m2 = record["area"] if swapped else record.get("area_ba", 0.0)
    c.gap_m = record.get("gap", 0.0)
    c.point_a = tuple(record["pb"] if swapped else record["pa"])
    c.point_b = tuple(record["pa"] if swapped else record["pb"])


def _after_change(context: bpy.types.Context) -> None:
    from . import overlay

    overlay.tag_redraw(context)


class FindConnections(Operator):
    """Find which parts touch and which face each other across a gap, for you to review"""

    bl_idname = "irsim.find_connections"
    bl_label = "Find connections"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return context.mode == "OBJECT"

    def execute(self, context):
        objects = connectable_objects(context)
        if len(objects) < 2:
            self.report({"ERROR"}, "Needs at least two parts")
            return {"CANCELLED"}
        touch, facing = gaps_for(context, objects)
        t0 = time.perf_counter()
        finder = Finder(
            objects, context.evaluated_depsgraph_get(), touch, facing, context.scene.irsim.samples
        )
        found = finder.run()
        counts = merge(
            context.scene.irsim_connections,
            found,
            {ob.name: ob for ob in objects},
            context.window_manager.irsim_joints,
        )
        context.scene.irsim.last_search = (
            f"Used: touch {1000 * touch:.2g} mm, face {100 * facing:.2g} cm"
        )
        scene_stats.refresh(context)
        _after_change(context)
        contacts = sum(1 for r in found if r["kind"] == "CONTACT")
        self.report(
            {"INFO"},
            f"{contacts} contacts and {len(found) - contacts} facing pairs among {len(objects)} "
            f"parts ({counts['added']} new) in {time.perf_counter() - t0:.1f} s; touching within "
            f"{1000 * touch:.3g} mm, facing within {facing:.3g} m",
        )
        return {"FINISHED"}


class AddConnection(Operator):
    """Connect the two selected parts yourself (a contact is measured; a facing pair is not)"""

    bl_idname = "irsim.add_connection"
    bl_label = "Connect selected"
    bl_options = {"REGISTER", "UNDO"}

    kind: EnumProperty(name="Kind", items=CONNECTION_KINDS, default="CONTACT")

    @classmethod
    def poll(cls, context):
        return context.mode == "OBJECT" and len(context.selected_objects) == 2

    def execute(self, context):
        a, b = sorted(context.selected_objects, key=lambda ob: ob.name)
        if a.type != "MESH" or b.type != "MESH":
            self.report({"ERROR"}, "Select two mesh parts")
            return {"CANCELLED"}
        connections = context.scene.irsim_connections
        for c in connections:
            if c.kind == self.kind and {c.a, c.b} == {a, b}:
                c.status = "ADDED"
                self.report({"INFO"}, f"{a.name} / {b.name} was already listed; marked as yours")
                return {"FINISHED"}
        connections.add()
        c = connections[len(connections) - 1]
        c.a, c.b, c.kind, c.status = a, b, self.kind, "ADDED"
        touch, facing = gaps_for(context, connectable_objects(context))
        a_centre = np.array([*a.matrix_world.translation])
        b_centre = np.array([*b.matrix_world.translation])
        c.point_a, c.point_b = tuple(a_centre), tuple(b_centre)
        if self.kind == "CONTACT":
            c.joint = DEFAULT_JOINT if DEFAULT_JOINT in context.window_manager.irsim_joints else ""
            depsgraph = context.evaluated_depsgraph_get()
            finder = Finder([a, b], depsgraph, touch, facing, context.scene.irsim.samples)
            area, point = finder.contact(a.name, b.name)
            c.area_m2 = area
            if point is not None:
                c.point_a = c.point_b = tuple(point)
            if not area:
                self.report(
                    {"WARNING"}, "They do not touch within the touching distance: type the area"
                )
        context.scene.irsim.active_connection_index = len(connections) - 1
        _after_change(context)
        return {"FINISHED"}


class SetConnectionStatus(Operator):
    """Confirm or reject a connection"""

    bl_idname = "irsim.set_connection_status"
    bl_label = "Review connection"
    bl_options = {"REGISTER", "UNDO"}

    index: IntProperty(default=-1)
    status: EnumProperty(
        items=[("CONFIRMED", "Confirm", "Keep it"), ("REJECTED", "Reject", "Rule it out")]
    )

    def execute(self, context):
        connections = context.scene.irsim_connections
        index = self.index if self.index >= 0 else context.scene.irsim.active_connection_index
        if not 0 <= index < len(connections):
            return {"CANCELLED"}
        c = connections[index]
        if c.status == "ADDED" and self.status == "CONFIRMED":
            return {"FINISHED"}
        c.status = self.status
        _after_change(context)
        return {"FINISHED"}


class ConfirmAllConnections(Operator):
    """Confirm every connection the finder proposed that you have not reviewed"""

    bl_idname = "irsim.confirm_all_connections"
    bl_label = "Confirm all found"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        n = 0
        for c in context.scene.irsim_connections:
            if c.status == "FOUND":
                c.status = "CONFIRMED"
                n += 1
        self.report({"INFO"}, f"Confirmed {n}")
        _after_change(context)
        return {"FINISHED"}


class RemoveConnection(Operator):
    """Delete a connection from the list (a new search may find it again; Reject keeps it out)"""

    bl_idname = "irsim.remove_connection"
    bl_label = "Delete connection"
    bl_options = {"REGISTER", "UNDO"}

    index: IntProperty(default=-1)

    def execute(self, context):
        connections = context.scene.irsim_connections
        index = self.index if self.index >= 0 else context.scene.irsim.active_connection_index
        if not 0 <= index < len(connections):
            return {"CANCELLED"}
        connections.remove(index)
        settings = context.scene.irsim
        settings.active_connection_index = min(index, len(connections) - 1)
        _after_change(context)
        return {"FINISHED"}


class SelectConnection(Operator):
    """Select the two parts of this connection"""

    bl_idname = "irsim.select_connection"
    bl_label = "Select its parts"

    index: IntProperty(default=-1)

    @classmethod
    def poll(cls, context):
        return context.mode == "OBJECT"

    def execute(self, context):
        connections = context.scene.irsim_connections
        index = self.index if self.index >= 0 else context.scene.irsim.active_connection_index
        if not 0 <= index < len(connections):
            return {"CANCELLED"}
        c = connections[index]
        context.scene.irsim.active_connection_index = index
        for ob in context.selected_objects:
            ob.select_set(False)
        for ob in (c.b, c.a):
            if ob is not None and ob.visible_get():
                ob.select_set(True)
                context.view_layer.objects.active = ob
        _after_change(context)
        return {"FINISHED"}


classes = (
    FindConnections,
    AddConnection,
    SetConnectionStatus,
    ConfirmAllConnections,
    RemoveConnection,
    SelectConnection,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
