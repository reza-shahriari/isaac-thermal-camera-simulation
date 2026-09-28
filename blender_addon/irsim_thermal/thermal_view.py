"""A viewport mode that shows the assignment instead of the texture.

Each material's viewport colour (Solid shading, *Material* colour) becomes a grey whose brightness
on screen is its thermal material's LWIR emissivity -- a bright part emits, a dark one mirrors
its surroundings -- and every material with no thermal material turns magenta, so what is left to
do is visible at a glance. Only ``Material.diffuse_color`` (the viewport display colour, which no
renderer uses) and the 3D views' shading mode change; both are restored when the mode is switched
off. The original colour is kept on the material itself, so it survives saving the file mid-way.

A part with **no material at all** cannot be coloured this way (Blender draws it in its default
grey); the checklist lists it instead, and assigning it a thermal material gives it one.
"""

import bpy

from .library_state import emissivity_lwir

SAVED = "irsim_saved_display_colour"
UNASSIGNED = (1.0, 0.0, 1.0, 1.0)
UNKNOWN = (0.5, 0.5, 0.5, 1.0)

#: ``{space pointer: (shading type, colour type)}`` for the views this mode changed.
_saved_shading: dict[int, tuple[str, str]] = {}


def display_grey(emissivity: float) -> float:
    """The linear colour value that *displays* as a grey of ``emissivity`` (sRGB decoding).

    Blender stores colours linear and encodes them for the screen, so writing ε straight in
    would show ε = 0.09 as a mid-grey 0.33 -- a polished part would not look dark at all. Decoding
    first makes the brightness on screen equal ε, which is the point of the view.
    """
    c = max(0.0, min(1.0, float(emissivity)))
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _view_spaces():
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == "VIEW_3D":
                for space in area.spaces:
                    if space.type == "VIEW_3D":
                        yield area, space


def apply(context: bpy.types.Context) -> None:
    eps = emissivity_lwir(context)
    seen = set()
    for ob in context.scene.objects:
        for slot in getattr(ob, "material_slots", ()):
            mat = slot.material
            if mat is None or mat in seen:
                continue
            seen.add(mat)
            if SAVED not in mat:
                mat[SAVED] = list(mat.diffuse_color)
            thermal = mat.irsim_material
            if not thermal:
                mat.diffuse_color = UNASSIGNED
            elif thermal in eps:
                v = display_grey(eps[thermal])
                mat.diffuse_color = (v, v, v, 1.0)
            else:
                mat.diffuse_color = UNKNOWN
    for area, space in _view_spaces():
        key = space.as_pointer()
        if key not in _saved_shading:
            _saved_shading[key] = (space.shading.type, space.shading.color_type)
        space.shading.type = "SOLID"
        space.shading.color_type = "MATERIAL"
        area.tag_redraw()


def restore(context: bpy.types.Context) -> None:
    for mat in bpy.data.materials:
        if SAVED in mat:
            mat.diffuse_color = tuple(mat[SAVED])
            del mat[SAVED]
    for area, space in _view_spaces():
        saved = _saved_shading.pop(space.as_pointer(), None)
        if saved is not None:
            space.shading.type, space.shading.color_type = saved
            area.tag_redraw()


def on_toggle(settings, context: bpy.types.Context) -> None:
    """``Scene.irsim.thermal_view``'s update callback."""
    if settings.thermal_view:
        apply(context)
    else:
        restore(context)
