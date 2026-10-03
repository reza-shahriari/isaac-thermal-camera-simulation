"""The sidebar tab (View3D > Sidebar > irsim): parts, library, hidden parts, connections, export."""

import bpy
from bpy.types import Panel, UIList

from . import library_state, sizing
from .properties import BANDS

MIRROR = 0.2


def _slot_summary(ob: bpy.types.Object) -> tuple[str, str]:
    """``(text, icon)`` for a part's row, from its slots (cheap enough to run on every redraw)."""
    thermals = [s.material.irsim_material if s.material else "" for s in ob.material_slots]
    if not thermals or not any(thermals):
        return "unassigned", "ERROR"
    if not all(thermals):
        return "partly assigned", "ERROR"
    distinct = sorted(set(thermals))
    if len(distinct) == 1:
        return distinct[0], "CHECKMARK"
    return f"{len(distinct)} materials", "CHECKMARK"


class PartsList(UIList):
    bl_idname = "IRSIM_UL_parts"

    def draw_item(self, context, layout, data, item, icon, active_data, active_prop, index=0):
        text, status = _slot_summary(item)
        row = layout.row(align=True)
        row.label(text=item.name, icon="MESH_DATA")
        row.label(text=text, icon=status)

    def filter_items(self, context, data, propname):
        objects = getattr(data, propname)
        helper = bpy.types.UI_UL_list
        if self.filter_name:
            flags = helper.filter_items_by_name(
                self.filter_name, self.bitflag_filter_item, objects, "name"
            )
        else:
            flags = [self.bitflag_filter_item] * len(objects)
        flags = [
            f
            if (ob.type == "MESH" and not ob.hide_render and not ob.irsim_component.is_hidden_part)
            else 0
            for f, ob in zip(flags, objects, strict=True)
        ]
        order = helper.sort_items_by_name(objects, "name") if self.use_filter_sort_alpha else []
        return flags, order


class LibraryList(UIList):
    bl_idname = "IRSIM_UL_library"

    def draw_item(self, context, layout, data, item, icon, active_data, active_prop, index=0):
        row = layout.row(align=True)
        mark = "ERROR" if item.error else ("QUESTION" if item.source == "estimated" else "NONE")
        row.label(text=item.name, icon=mark if mark != "NONE" else "MATERIAL")
        sub = row.row()
        sub.alert = item.eps_lwir < MIRROR
        sub.label(text=f"ε {item.eps_lwir:.2f}")


class IssuesList(UIList):
    bl_idname = "IRSIM_UL_issues"

    ICONS = {
        "UNASSIGNED": "ERROR",
        "MIRROR": "LIGHT_SUN",
        "NAME": "SORTALPHA",
        "HIDDEN": "MESH_CUBE",
        "CONNECTION": "LINKED",
        "INFO": "INFO",
    }

    def draw_item(self, context, layout, data, item, icon, active_data, active_prop, index=0):
        row = layout.row(align=True)
        row.label(text=item.name, icon=self.ICONS.get(item.kind, "INFO"))
        row.label(text=item.detail)
        op = row.operator("irsim.select_issue", text="", icon="RESTRICT_SELECT_OFF")
        op.name = item.name


class HiddenList(UIList):
    bl_idname = "IRSIM_UL_hidden"

    def draw_item(self, context, layout, data, item, icon, active_data, active_prop, index=0):
        comp = item.irsim_component
        row = layout.row(align=True)
        row.label(text=item.name, icon="MESH_CUBE")
        row.label(text=f"{comp.heat_rated_w:.4g} W, {comp.mass_kg:.3g} kg")

    def filter_items(self, context, data, propname):
        objects = getattr(data, propname)
        return [
            self.bitflag_filter_item
            if (ob.type == "MESH" and ob.irsim_component.is_hidden_part)
            else 0
            for ob in objects
        ], []


def _connection_text(c) -> str:
    a = c.a.name if c.a else "(gone)"
    b = c.b.name if c.b else "(gone)"
    return f"{a} – {b}"


def _area_text(m2: float) -> str:
    return f"{m2 * 1e4:.3g} cm²" if m2 < 0.1 else f"{m2:.3g} m²"


class ConnectionsList(UIList):
    bl_idname = "IRSIM_UL_connections"

    KIND_ICONS = {"CONTACT": "SNAP_FACE", "FACING": "LIGHT_SUN"}
    STATUS_ICONS = {
        "FOUND": "QUESTION",
        "CONFIRMED": "CHECKMARK",
        "REJECTED": "X",
        "ADDED": "USER",
    }

    def draw_item(self, context, layout, data, item, icon, active_data, active_prop, index=0):
        row = layout.row(align=True)
        row.active = item.status != "REJECTED"
        # With the list narrowed to the active part's connections, the other part is the news.
        ob = context.active_object
        if ob is not None and ob in (item.a, item.b) and _narrowed(context):
            other = item.b if item.a == ob else item.a
            text = other.name if other is not None else "(gone)"
        else:
            text = _connection_text(item)
        split = row.split(factor=0.66, align=True)
        split.label(text=text, icon=self.KIND_ICONS[item.kind])
        tail = split.row(align=True)
        # A facing pair's two areas differ (a small part sees much of a big one, not the reverse).
        tail.label(text=_area_text(max(item.area_m2, item.area_ba_m2)))
        tail.label(text="", icon=self.STATUS_ICONS[item.status])

    def filter_items(self, context, data, propname):
        items = getattr(data, propname)
        helper = bpy.types.UI_UL_list
        if self.filter_name:
            flags = [
                self.bitflag_filter_item
                if self.filter_name.lower() in _connection_text(c).lower()
                else 0
                for c in items
            ]
        else:
            flags = [self.bitflag_filter_item] * len(items)
        if _narrowed(context):
            selected = set(context.selected_objects)
            flags = [
                f if (c.a in selected or c.b in selected) else 0
                for f, c in zip(flags, items, strict=True)
            ]
        order = (
            helper.sort_items_helper(
                [(i, _connection_text(c)) for i, c in enumerate(items)], key=lambda x: x[1]
            )
            if self.use_filter_sort_alpha
            else []
        )
        return flags, order


def _narrowed(context) -> bool:
    """Whether lists and lines show only the selected parts' connections just now."""
    return context.scene.irsim.connections_of_selected and bool(context.selected_objects)


class _Base:
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "irsim"


class MainPanel(_Base, Panel):
    bl_idname = "IRSIM_PT_main"
    bl_label = "irsim Thermal"

    def draw(self, context):
        layout = self.layout
        wm = context.window_manager
        settings = context.scene.irsim
        row = layout.row(align=True)
        if len(wm.irsim_library):
            row.label(text=f"Library: {len(wm.irsim_library)} materials", icon="MATERIAL")
            row.operator("irsim.refresh_library", text="", icon="FILE_REFRESH")
        else:
            col = layout.column()
            col.operator("irsim.refresh_library", icon="IMPORT")
            if wm.irsim_library_status:
                col.label(text=wm.irsim_library_status[:120], icon="ERROR")
            col.label(text="Repository and Python: Edit > Preferences > Add-ons", icon="INFO")
        if settings.checked:
            text = (
                f"{settings.coverage:.0f} % assigned, "
                f"{settings.unassigned_count} of {settings.part_count} parts to go"
            )
            layout.progress(factor=settings.coverage / 100.0, type="BAR", text=text)
        layout.prop(settings, "thermal_view", icon="SHADING_SOLID")


class PartsPanel(_Base, Panel):
    bl_idname = "IRSIM_PT_parts"
    bl_parent_id = "IRSIM_PT_main"
    bl_label = "Parts"

    def draw(self, context):
        layout = self.layout
        settings = context.scene.irsim
        layout.template_list(
            "IRSIM_UL_parts",
            "",
            context.scene,
            "objects",
            settings,
            "active_part_index",
            rows=6,
        )
        row = layout.row(align=True)
        op = row.operator(
            "irsim.select_parts", text="Select unassigned", icon="RESTRICT_SELECT_OFF"
        )
        op.which = "UNASSIGNED"
        row.operator("irsim.check", text="Refresh", icon="FILE_REFRESH")

        ob = context.active_object
        if ob is None or ob.type != "MESH":
            return
        box = layout.box()
        box.label(text=f"{ob.name}: its materials", icon="OBJECT_DATA")
        if not ob.material_slots:
            box.label(text="No materials: assigning one creates it", icon="INFO")
        for slot in ob.material_slots:
            mat = slot.material
            row = box.row()
            if mat is None:
                row.label(text="(empty slot)", icon="BLANK1")
                continue
            row.label(text=mat.name, icon="MATERIAL")
            if mat.irsim_material:
                row.label(text=mat.irsim_material, icon="CHECKMARK")
            else:
                row.label(text="unassigned", icon="ERROR")


class LibraryPanel(_Base, Panel):
    bl_idname = "IRSIM_PT_library"
    bl_parent_id = "IRSIM_PT_main"
    bl_label = "Material library"

    def draw(self, context):
        layout = self.layout
        wm = context.window_manager
        settings = context.scene.irsim
        if not len(wm.irsim_library):
            layout.operator("irsim.refresh_library", icon="IMPORT")
            return
        layout.template_list(
            "IRSIM_UL_library",
            "",
            wm,
            "irsim_library",
            settings,
            "active_library_index",
            rows=6,
        )
        index = settings.active_library_index
        item = wm.irsim_library[index] if 0 <= index < len(wm.irsim_library) else None
        if item is None:
            return
        self._details(layout.box(), item, wm)

        col = layout.column(align=True)
        if context.mode == "EDIT_MESH":
            op = col.operator("irsim.assign", text="Assign to selected faces", icon="FACESEL")
            op.material, op.scope = item.name, "FACES"
        else:
            op = col.operator("irsim.assign", text="Assign to selected parts", icon="CHECKMARK")
            op.material, op.scope = item.name, "PARTS"
            ob = context.active_object
            mat = ob.active_material if ob else None
            if mat is not None:
                op = col.operator("irsim.assign", text=f"Assign to every part using '{mat.name}'")
                op.material, op.scope = item.name, "MATERIAL"
            op = col.operator(
                "irsim.select_parts", text="Select parts made of this", icon="RESTRICT_SELECT_OFF"
            )
            op.which, op.material = "USING", item.name
        row = layout.row(align=True)
        row.operator("irsim.new_material", text="New material...", icon="ADD")
        op = row.operator("irsim.new_material", text="New from this...", icon="DUPLICATE")
        op.start_from = item.name
        if context.mode == "OBJECT":
            layout.operator("irsim.load_asset_map", text="From an existing asset...", icon="IMPORT")

    @staticmethod
    def _details(box, item, wm) -> None:
        box.label(text=item.description[:90] or item.name)
        row = box.row()
        row.label(text=f"Source: {item.source}")
        row.label(text=f"Surface: {item.surface_treatment}")
        if item.forms:
            box.label(text=f"Authored as {item.forms}", icon="GRAPH" if item.spectral else "NONE")
        if item.error:
            box.label(text=item.error[:120], icon="ERROR")
        grid = box.grid_flow(row_major=True, columns=5, even_columns=True, align=True)
        grid.label(text="")
        for band in BANDS:
            grid.label(text=band.upper())
        for what, label in (("eps", "ε"), ("rho", "ρ"), ("tau", "τ")):
            grid.label(text=label)
            for band in BANDS:
                grid.label(text=f"{getattr(item, f'{what}_{band}'):.2f}")
        if item.spectral:
            # How much of each band the curve supplied: under 100 % the rest is a per-band or
            # grey fill, assumed flat (ADR 0175).
            grid.label(text="curve")
            for band in BANDS:
                grid.label(text=f"{100.0 * getattr(item, f'curve_{band}'):.0f} %")
        row = box.row()
        op = row.operator("irsim.plot_material", text="Plot across the bands", icon="GRAPH")
        op.material = item.name
        image = bpy.data.images.get(wm.irsim_plot_image) if wm.irsim_plot_image else None
        if image is not None and wm.irsim_plot_material == item.name:
            box.template_icon(icon_value=image.preview_ensure().icon_id, scale=12.0)
        col = box.column(align=True)
        col.label(text=f"Solar absorptivity {item.solar_absorptivity:.2f}")
        col.label(
            text=(
                f"Heat capacity {item.heat_capacity_j_m2_k:,.0f} J/m²K "
                f"({item.thickness_m * 1000:.3g} mm thick)"
            )
        )
        col.label(text=f"Conductivity {item.conductivity_w_mk:.3g} W/m·K")
        if item.eps_lwir < MIRROR:
            box.label(
                text="Mirror-like in LWIR: shows reflections, not its temperature", icon="ERROR"
            )


class HiddenPartsPanel(_Base, Panel):
    bl_idname = "IRSIM_PT_hidden"
    bl_parent_id = "IRSIM_PT_main"
    bl_label = "Hidden parts"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = self.layout
        settings = context.scene.irsim
        layout.label(text="Parts the model lacks", icon="INFO")
        layout.template_list(
            "IRSIM_UL_hidden", "", context.scene, "objects", settings, "active_hidden_index", rows=3
        )
        row = layout.row(align=True)
        row.operator("irsim.add_component", text="Add from library...", icon="ASSET_MANAGER")
        row.operator("irsim.add_hidden_part", text="Add a box...", icon="MESH_CUBE")
        layout.operator("irsim.show_hidden_parts", icon="HIDE_OFF")
        ob = context.active_object
        if ob is None or not ob.irsim_component.is_hidden_part:
            return
        comp = ob.irsim_component
        box = layout.box()
        box.use_property_split = True
        box.label(text=ob.name, icon="MESH_CUBE")
        box.prop(comp, "kind")
        box.prop(ob, "dimensions", text="Size")
        box.prop_search(
            comp, "component", context.window_manager, "irsim_components", text="Component"
        )
        found = library_state.find_component(context, comp.component)
        if found is not None:
            # short lines: the sidebar is narrow and a label is cut off, never wrapped
            col = box.column(align=True)
            col.label(text=f"Its numbers ({found.status.lower()}):", icon="INFO")
            col.label(text=f"{found.mass_kg:g} kg, {found.specific_heat_j_kgk:g} J/kg·K")
            col.label(text=f"heat {found.dissipation_idle_w:g}–{found.dissipation_rated_w:g} W")
            col.label(text="A number below overrides it;")
            col.label(text="0 keeps the component's.")
        elif comp.component:
            box.label(text=f"{comp.component!r} is not in irsim's components", icon="ERROR")
        box.prop(comp, "mass_kg")
        box.prop(comp, "specific_heat_j_kgk")
        box.prop(comp, "heat_idle_w")
        box.prop(comp, "heat_rated_w")
        box.prop(comp, "values_source")
        box.prop(comp, "reference")
        thermal = sorted(
            {s.material.irsim_material for s in ob.material_slots if s.material} - {""}
        )
        box.label(
            text=f"Material: {', '.join(thermal)}" if thermal else "Material: assign one below",
            icon="MATERIAL" if thermal else "ERROR",
        )
        if ob.parent is not None:
            box.label(text=f"Inside {ob.parent.name}", icon="LINKED")


class ConnectionsPanel(_Base, Panel):
    bl_idname = "IRSIM_PT_connections"
    bl_parent_id = "IRSIM_PT_main"
    bl_label = "Connections"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = self.layout
        settings = context.scene.irsim
        connections = context.scene.irsim_connections
        col = layout.column(align=True)
        col.prop(settings, "touch_gap_m", text="Touching within (0 = auto)")
        col.prop(settings, "facing_gap_m", text="Facing within (0 = auto)")
        col.prop(settings, "samples")
        row = layout.row(align=True)
        row.operator("irsim.find_connections", icon="VIEWZOOM")
        row.prop(settings, "show_connections", text="", icon="HIDE_OFF")
        layout.prop(settings, "connections_of_selected")
        if not len(connections):
            layout.label(text="Nothing found yet", icon="INFO")
            op = layout.operator("irsim.add_connection", text="Connect the two selected")
            op.kind = "CONTACT"
            return
        if settings.last_search:
            layout.label(text=settings.last_search, icon="INFO")
        found = sum(1 for c in connections if c.status == "FOUND")
        layout.label(
            text=f"{len(connections)} connections, {found} to review",
            icon="QUESTION" if found else "CHECKMARK",
        )
        layout.template_list(
            "IRSIM_UL_connections",
            "",
            context.scene,
            "irsim_connections",
            settings,
            "active_connection_index",
            rows=5,
        )
        row = layout.row(align=True)
        op = row.operator("irsim.add_connection", text="Connect", icon="ADD")
        op.kind = "CONTACT"
        if found:
            row.operator("irsim.confirm_all_connections", text="Confirm all", icon="CHECKMARK")
        index = settings.active_connection_index
        if not 0 <= index < len(connections):
            return
        c = connections[index]
        box = layout.box()
        box.label(text=_connection_text(c), icon=ConnectionsList.KIND_ICONS[c.kind])
        col = box.column()
        col.use_property_split = True
        col.prop(c, "kind")
        col.prop(c, "status")
        col.prop(c, "area_m2")
        if c.kind == "CONTACT":
            col.prop_search(c, "joint", context.window_manager, "irsim_joints")
            joint = context.window_manager.irsim_joints.get(c.joint)
            if joint is not None:
                box.label(
                    text=f"{joint.h_c_w_m2_k:,.0f} W/m²K → {joint.h_c_w_m2_k * c.area_m2:.3g} W/K",
                    icon="CHECKMARK" if joint.status == "MEASURED" else "QUESTION",
                )
        else:
            col.prop(c, "area_ba_m2")
            col.prop(c, "gap_m")
            box.label(text="Looking-at area, not a view factor (roadmap TC.9)", icon="INFO")
        row = box.row(align=True)
        op = row.operator("irsim.select_connection", text="Select", icon="RESTRICT_SELECT_OFF")
        op.index = index
        op = row.operator("irsim.set_connection_status", text="Confirm", icon="CHECKMARK")
        op.index, op.status = index, "CONFIRMED"
        op = row.operator("irsim.set_connection_status", text="Reject", icon="X")
        op.index, op.status = index, "REJECTED"
        op = row.operator("irsim.remove_connection", text="", icon="TRASH")
        op.index = index


class ExportPanel(_Base, Panel):
    bl_idname = "IRSIM_PT_export"
    bl_parent_id = "IRSIM_PT_main"
    bl_label = "Check and export"

    def draw(self, context):
        layout = self.layout
        settings = context.scene.irsim
        layout.operator("irsim.check", icon="VIEWZOOM")
        if settings.checked:
            self._size(layout.box(), settings)
            if len(settings.issues):
                layout.label(text="To look at:")
                layout.template_list(
                    "IRSIM_UL_issues",
                    "",
                    settings,
                    "issues",
                    settings,
                    "active_issue_index",
                    rows=4,
                )
            else:
                layout.label(text="Nothing left to fix", icon="CHECKMARK")

        box = layout.box()
        box.prop(settings, "asset_name")
        box.prop(settings, "replace_export")
        box.operator("irsim.export_asset", icon="EXPORT")
        if settings.last_export:
            box.label(text=settings.last_export, icon="FILE")
        box.operator("irsim.run_audit", icon="CHECKBOX_HLT")
        if settings.audit_status:
            box.label(text=settings.audit_status)

    @staticmethod
    def _size(box, settings) -> None:
        dims = (settings.size_x_m, settings.size_y_m, settings.size_z_m)
        largest = max(dims)
        box.label(
            text=f"Size {dims[0]:.3g} x {dims[1]:.3g} x {dims[2]:.3g} m", icon="FULLSCREEN_ENTER"
        )
        box.label(text=sizing.comparison(largest))
        box.prop(settings, "size_help")
        if not settings.size_help:
            return
        box.prop(settings, "size_kind")
        row = box.row()
        row.prop(settings, "known_dimension_m")
        row.prop(settings, "known_axis", text="")
        factor = None
        if settings.known_dimension_m > 0.0:
            axis = {"X": 0, "Y": 1, "Z": 2}.get(settings.known_axis)
            measured = largest if axis is None else dims[axis]
            if measured > 0.0:
                factor, why = sizing.scale_for_known(measured, settings.known_dimension_m)
                box.label(
                    text=f"Scale x{factor:.4g} makes it match" + (f" ({why})" if why else ""),
                    icon="INFO",
                )
        elif settings.size_kind != "NONE":
            ranges = {r.key: r for r in sizing.load_ranges()}
            diagnosis = sizing.diagnose(largest, ranges[settings.size_kind])
            icon = {"within": "CHECKMARK", "unit": "ERROR", "outside": "QUESTION"}
            for i, chunk in enumerate(_wrap(diagnosis.message, 60)):
                box.label(text=chunk, icon=icon[diagnosis.status] if i == 0 else "BLANK1")
            factor = diagnosis.factor
        if factor is not None and abs(factor - 1.0) > 1e-9:
            op = box.operator("irsim.apply_scale", text=f"Apply scale x{factor:.4g}")
            op.factor = factor


def _wrap(text: str, width: int) -> list[str]:
    words, lines, line = text.split(), [], ""
    for word in words:
        if line and len(line) + 1 + len(word) > width:
            lines.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    if line:
        lines.append(line)
    return lines


classes = (
    PartsList,
    LibraryList,
    IssuesList,
    HiddenList,
    ConnectionsList,
    MainPanel,
    PartsPanel,
    LibraryPanel,
    HiddenPartsPanel,
    ConnectionsPanel,
    ExportPanel,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
