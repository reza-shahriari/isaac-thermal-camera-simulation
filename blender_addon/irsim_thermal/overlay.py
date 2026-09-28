"""Connections drawn in the 3D viewport, so a wrong one is seen rather than read in a list.

Each connection is drawn where it was found: a contact as a small orange cross where the two parts
touch, a facing pair as a blue line from the middle of the area that looks across to the middle of
where it lands. A rejected connection is drawn faint. The one selected in the list is drawn white
and thicker, with lines to the middle of each of its two parts, so it is clear which two they are.
By default only the selected parts' connections are drawn (all of them when nothing is selected):
a real model has hundreds.
Nothing is drawn unless *Show in viewport* is ticked, and nothing is ever drawn in a render.
"""

import bpy
import gpu
from gpu_extras.batch import batch_for_shader

COLOURS = {
    "CONTACT": (1.0, 0.55, 0.1, 1.0),
    "FACING": (0.25, 0.65, 1.0, 1.0),
}
FAINT = 0.25
ACTIVE = (1.0, 1.0, 1.0, 1.0)

_handle = None


def _box(ob) -> tuple[tuple[float, float, float], float]:
    """Middle and diagonal of the part's box: cheap enough for every redraw of a large model."""
    from mathutils import Vector

    corners = [ob.matrix_world @ Vector(c) for c in ob.bound_box]
    lo = Vector(map(min, *corners))
    hi = Vector(map(max, *corners))
    return tuple((lo + hi) / 2), (hi - lo).length


def _cross(point, size: float) -> list:
    x, y, z = point
    h = size / 2
    return [
        (x - h, y, z), (x + h, y, z),
        (x, y - h, z), (x, y + h, z),
        (x, y, z - h), (x, y, z + h),
    ]  # fmt: skip


def segments(context) -> list[tuple[list, tuple, float]]:
    """``[(line-pair points, colour, width)]`` for every connection whose parts still exist.

    A contact is a small cross where the parts touch, sized to the smaller part; a facing pair is
    a line across the gap. Only the connection selected in the list also gets lines to the middle
    of each of its parts: drawn for all of them, a model with a hundred contacts is a web.
    """
    out = []
    settings = context.scene.irsim
    active = settings.active_connection_index
    selected = set(context.selected_objects) if settings.connections_of_selected else set()
    for index, c in enumerate(context.scene.irsim_connections):
        if c.a is None or c.b is None:
            continue
        if selected and index != active and c.a not in selected and c.b not in selected:
            continue
        colour = COLOURS[c.kind]
        width = 2.0
        if c.status == "REJECTED":
            colour = (*colour[:3], FAINT)
        (ca, da), (cb, db) = _box(c.a), _box(c.b)
        if c.kind == "CONTACT":
            points = _cross(tuple(c.point_a), 0.04 * min(da, db))
        else:
            points = [tuple(c.point_a), tuple(c.point_b)]
        if index == active:
            colour, width = ACTIVE, 4.0
            points += [ca, tuple(c.point_a), tuple(c.point_b), cb]
        out.append((points, colour, width))
    return out


def _draw():
    context = bpy.context
    if context.scene is None or not context.scene.irsim.show_connections:
        return
    shader = gpu.shader.from_builtin("POLYLINE_UNIFORM_COLOR")
    region = context.region
    gpu.state.blend_set("ALPHA")
    gpu.state.depth_test_set("NONE")
    for points, colour, width in segments(context):
        batch = batch_for_shader(shader, "LINES", {"pos": points})
        shader.uniform_float("viewportSize", (region.width, region.height))
        shader.uniform_float("lineWidth", width)
        shader.uniform_float("color", colour)
        batch.draw(shader)
    gpu.state.blend_set("NONE")


def tag_redraw(context) -> None:
    wm = getattr(context, "window_manager", None)
    for window in wm.windows if wm else ():
        for area in window.screen.areas:
            if area.type == "VIEW_3D":
                area.tag_redraw()


def on_toggle(settings, context) -> None:
    """``Scene.irsim.show_connections``'s update."""
    tag_redraw(context)


def register():
    # Added once and left in place: it draws nothing unless the open scene asks for it, which also
    # covers a saved file reopened with the lines switched on.
    global _handle
    if _handle is None and not bpy.app.background:
        _handle = bpy.types.SpaceView3D.draw_handler_add(_draw, (), "WINDOW", "POST_VIEW")


def unregister():
    global _handle
    if _handle is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_handle, "WINDOW")
        _handle = None
