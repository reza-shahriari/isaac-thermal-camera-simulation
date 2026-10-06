"""A form for a new library material, checked by irsim before anything is written.

The form asks for **emissivity** in one of the library's three forms (ADR 0175) -- by default a
measured curve picked as a CSV (up to two segments, a short-wave reflectance joined as 1 − R) with
per-band values filling only where the curve has no data; or, when nobody has measured the
surface, one grey value for every band or one value per band --
and **transmittance per band**, and shows reflectance as it will be derived, ρ = 1 − ε − τ. It
cannot ask for reflectance as well: authoring two of the three is the mistake CLAUDE.md #4 exists
to prevent, and ``irsim.config.materials`` refuses it. On *Create* the
bridge validates the draft with the project's own schema, loads it as the library would, checks
that every band closes, and only then writes ``configs/materials/<name>.yaml``.

The angular model is *constant* or *empirical* (ε(θ) = ε₀[1 − a(1 − cos θ)^p]). A Fresnel model
needs an n/k table under ``data/nk/``, which is a file to add by hand, not a number to type.
"""

import re

import bpy
from bpy.props import BoolProperty, EnumProperty, FloatProperty, StringProperty
from bpy.types import Operator

from . import library_state, naming, prefs
from .assign import assign_to_parts
from .bridge_client import BridgeError, run_bridge
from .properties import BANDS

SURFACE_TREATMENTS = (
    ("as_manufactured", "As manufactured", "Moulded plastic, woven fabric, cast rubber"),
    ("painted", "Painted", "Any paint or coating"),
    ("anodised", "Anodised", "Anodised aluminium"),
    ("polished", "Polished", "Mirror-like metal"),
    ("machined", "Machined", "Bare machined metal"),
    ("oxidised", "Oxidised", "Metal left to oxidise"),
    ("sandblasted", "Sandblasted", "Blasted metal"),
    ("weathered", "Weathered", "Aged by weather"),
    ("natural", "Natural", "Skin, soil, water, vegetation, snow"),
)
SOURCES = (
    ("estimated", "Estimated", "A judgement, not a measurement: flagged as such in the file"),
    ("literature", "Literature", "Taken from a publication: name it in Reference"),
    ("measured", "Measured", "Your own measurement of this surface"),
)
NAME = re.compile(r"^[a-z][a-z0-9_]*$")
#: Curve first and by default: the owner, 2026-10-01, "i want to have curves instead of specific
#: band selection". A measured curve is the material; band values only fill where it has no data.
#: *One value* and *Per band* remain for a surface nobody has measured, and say so.
OPTICAL_FORMS = (
    (
        "curve",
        "Curve",
        "A measured spectral curve (CSV), with per-band values only where it has no data",
    ),
    ("grey", "One value", "The same emissivity in every band (a grey body), when no curve exists"),
    ("per_band", "Per band (no curve)", "One emissivity for each band, when no curve exists"),
)
CURVE_QUANTITIES = (
    ("emissivity", "Emissivity", "The file tabulates emissivity"),
    ("reflectance", "Reflectance", "The file tabulates reflectance of an opaque surface (1 - R)"),
)


def _band_label(band: str) -> str:
    return band.upper()


def _tidy_name(self, context):
    """Whatever is typed becomes a library name: ``Nylon Black`` is ``nylon_black``."""
    tidy = naming.config_name(self.name)
    if tidy != self.name:  # assigning re-enters this callback once, with a tidy name
        self.name = tidy


class NewMaterial(Operator):
    """Create a new material in the irsim library (checked by irsim before it is written)"""

    bl_idname = "irsim.new_material"
    bl_label = "New thermal material"
    bl_options = {"REGISTER"}

    start_from: StringProperty(options={"SKIP_SAVE", "HIDDEN"})
    name: StringProperty(
        name="Name",
        description=(
            "The library name, written to configs/materials/<name>.yaml. Type it any way: it is "
            "kept in lower case with underscores (Nylon Black becomes nylon_black)"
        ),
        update=_tidy_name,
    )
    # Not `description`: that name is Operator.description, the dynamic-tooltip hook.
    about: StringProperty(name="Description", description="What surface this is")
    reference: StringProperty(
        name="Reference", description="Where the numbers came from (paper, datasheet, reasoning)"
    )
    source: EnumProperty(name="Source", items=SOURCES, default="estimated")
    surface_treatment: EnumProperty(
        name="Surface state", items=SURFACE_TREATMENTS, default="as_manufactured"
    )

    density_kg_m3: FloatProperty(name="Density (kg/m³)", default=1200.0, min=1.0)
    specific_heat_j_kgk: FloatProperty(name="Specific heat (J/kg·K)", default=1500.0, min=1.0)
    conductivity_w_mk: FloatProperty(name="Conductivity (W/m·K)", default=0.2, min=1e-4)
    thickness_mm: FloatProperty(
        name="Thickness (mm)",
        description="The thickness of the skin that stores heat (a panel, a shell wall)",
        default=2.0,
        min=0.01,
    )
    solar_absorptivity: FloatProperty(
        name="Solar absorptivity", default=0.5, min=0.0, max=1.0, precision=3
    )

    optical_form: EnumProperty(name="Emissivity as", items=OPTICAL_FORMS, default="curve")
    eps_grey: FloatProperty(name="ε (every band)", default=0.9, min=0.0, max=1.0, precision=3)
    curve_file: StringProperty(name="Curve", subtype="FILE_PATH", description="λ (µm), value CSV")
    curve_quantity: EnumProperty(name="Holds", items=CURVE_QUANTITIES, default="emissivity")
    curve_file_2: StringProperty(
        name="Second segment", subtype="FILE_PATH", description="Optional: another range"
    )
    curve_quantity_2: EnumProperty(name="Holds", items=CURVE_QUANTITIES, default="reflectance")

    eps_nir: FloatProperty(name="ε NIR", default=0.9, min=0.0, max=1.0, precision=3)
    eps_swir: FloatProperty(name="ε SWIR", default=0.9, min=0.0, max=1.0, precision=3)
    eps_mwir: FloatProperty(name="ε MWIR", default=0.9, min=0.0, max=1.0, precision=3)
    eps_lwir: FloatProperty(name="ε LWIR", default=0.9, min=0.0, max=1.0, precision=3)
    tau_nir: FloatProperty(name="τ NIR", default=0.0, min=0.0, max=1.0, precision=3)
    tau_swir: FloatProperty(name="τ SWIR", default=0.0, min=0.0, max=1.0, precision=3)
    tau_mwir: FloatProperty(name="τ MWIR", default=0.0, min=0.0, max=1.0, precision=3)
    tau_lwir: FloatProperty(name="τ LWIR", default=0.0, min=0.0, max=1.0, precision=3)
    rough_nir: FloatProperty(name="Roughness NIR", default=0.2, min=0.0, max=1.0, precision=2)
    rough_swir: FloatProperty(name="Roughness SWIR", default=0.2, min=0.0, max=1.0, precision=2)
    rough_mwir: FloatProperty(name="Roughness MWIR", default=0.1, min=0.0, max=1.0, precision=2)
    rough_lwir: FloatProperty(name="Roughness LWIR", default=0.1, min=0.0, max=1.0, precision=2)

    angular: EnumProperty(
        name="Angular model",
        items=[
            ("constant", "Constant", "Emissivity does not change with viewing angle"),
            ("empirical", "Empirical", "ε(θ) = ε₀ [1 − a (1 − cos θ)^p]: falls off at grazing"),
        ],
        default="empirical",
    )
    angular_a: FloatProperty(name="a", default=0.25, min=0.0, max=1.0)
    angular_p: FloatProperty(name="p", default=5.0, min=0.01, max=12.0)

    assign_after: BoolProperty(
        name="Assign it to the selected parts", default=True, options={"SKIP_SAVE"}
    )

    def _prefill(self, context) -> None:
        item = library_state.find(context, self.start_from)
        if item is None:
            return
        self.about = f"Based on {item.name}: {item.description}"[:250]
        self.reference = item.reference
        self.source = item.source if item.source in {s[0] for s in SOURCES} else "estimated"
        if item.surface_treatment in {s[0] for s in SURFACE_TREATMENTS}:
            self.surface_treatment = item.surface_treatment
        self.density_kg_m3 = item.density_kg_m3
        self.specific_heat_j_kgk = item.specific_heat_j_kgk
        self.conductivity_w_mk = item.conductivity_w_mk
        self.thickness_mm = item.thickness_m * 1000.0
        self.solar_absorptivity = item.solar_absorptivity
        for band in BANDS:
            setattr(self, f"eps_{band}", getattr(item, f"eps_{band}"))
            setattr(self, f"tau_{band}", getattr(item, f"tau_{band}"))
        if item.angular in {"constant", "empirical"}:
            self.angular = item.angular
        # The copy carries the original's band values, so it opens on them; picking a curve
        # (the default form for a new material) replaces them where the curve has data.
        self.optical_form = "per_band"
        self.name = ""

    def invoke(self, context, event):
        if self.start_from:
            self._prefill(context)
        return context.window_manager.invoke_props_dialog(
            self, width=620, title="New thermal material", confirm_text="Check and create"
        )

    def draw(self, context):
        layout = self.layout
        col = layout.column()
        col.prop(self, "name")
        if not self.name:
            col.label(text="Name it in Latin letters: it becomes a file name", icon="INFO")
        col.prop(self, "about")
        row = col.row()
        row.prop(self, "source")
        row.prop(self, "surface_treatment")
        col.prop(self, "reference")

        box = layout.box()
        box.label(text="Heat storage", icon="MOD_PHYSICS")
        grid = box.grid_flow(row_major=True, columns=2, even_columns=True)
        for key in (
            "density_kg_m3",
            "specific_heat_j_kgk",
            "conductivity_w_mk",
            "thickness_mm",
            "solar_absorptivity",
        ):
            grid.prop(self, key)
        capacity = self.density_kg_m3 * self.specific_heat_j_kgk * self.thickness_mm / 1000.0
        box.label(text=f"Heat capacity per area: {capacity:,.0f} J/m²·K")

        box = layout.box()
        box.prop(self, "optical_form", expand=True)
        if self.optical_form == "grey":
            box.prop(self, "eps_grey")
        elif self.optical_form == "curve":
            for file_key, q_key in (
                ("curve_file", "curve_quantity"),
                ("curve_file_2", "curve_quantity_2"),
            ):
                row = box.row()
                row.prop(self, file_key)
                row.prop(self, q_key, text="")
            row = box.row()
            op = row.operator("irsim.check_curve", text="Check curve", icon="CHECKMARK")
            op.filepath = self.curve_file
            if self.curve_file_2:
                op = row.operator("irsim.check_curve", text="Check second", icon="CHECKMARK")
                op.filepath = self.curve_file_2
            check = context.window_manager.irsim_curve_check
            if check:
                box.label(text=check[:110], icon="INFO")
            box.label(text="Per-band ε below fills only the bands the curve does not cover")
        header = box.row()
        header.label(text="")
        for band in BANDS:
            header.label(text=_band_label(band))
        rows = [("tau", "Transmittance τ")]
        if self.optical_form != "grey":
            rows.insert(0, ("eps", "Emissivity ε" if self.optical_form == "per_band" else "ε fill"))
        for what, label in rows:
            row = box.row()
            row.label(text=label)
            for band in BANDS:
                row.prop(self, f"{what}_{band}", text="")
        row = box.row()
        row.label(text="Reflectance ρ (derived)")
        bad = []
        for band in BANDS:
            eps = self.eps_grey if self.optical_form == "grey" else getattr(self, f"eps_{band}")
            rho = 1.0 - eps - getattr(self, f"tau_{band}")
            cell = row.row()
            cell.alert = rho < -1e-9
            cell.label(text=f"{rho:.3f}" if self.optical_form != "curve" else "from curve")
            if rho < -1e-9:
                bad.append(band.upper())
        if bad and self.optical_form != "curve":
            box.label(text=f"ε + τ > 1 in {', '.join(bad)}: not physical", icon="ERROR")
        row = box.row()
        row.label(text="Roughness")
        for band in BANDS:
            row.prop(self, f"rough_{band}", text="")
        lwir = self.eps_grey if self.optical_form == "grey" else self.eps_lwir
        if lwir < 0.2 and self.optical_form != "curve":
            box.label(
                text="LWIR ε below 0.2: it will show reflections, not its own temperature",
                icon="ERROR",
            )

        row = layout.row()
        row.prop(self, "angular")
        if self.angular == "empirical":
            row.prop(self, "angular_a")
            row.prop(self, "angular_p")
        layout.prop(self, "assign_after")

    def _draft(self) -> dict:
        # Blender keeps a FloatProperty in single precision, so a typed 0.9 arrives as
        # 0.8999999761581421. Six significant figures is what the form shows and more than any
        # of these quantities is known to; the file should say what the person typed.
        def v(x: float) -> float:
            return float(f"{x:.6g}")

        return {
            # The bridge drops the typed band values a picked curve already covers (ADR 0175).
            "prune_covered": self.optical_form == "curve",
            "material": {
                "name": self.name,
                "source": self.source,
                "surface_treatment": self.surface_treatment,
                "reference": self.reference,
                "description": self.about,
                "thermal": {
                    "density_kg_m3": v(self.density_kg_m3),
                    "specific_heat_j_kgk": v(self.specific_heat_j_kgk),
                    "conductivity_w_mk": v(self.conductivity_w_mk),
                    "thickness_m": v(self.thickness_mm / 1000.0),
                    "solar_absorptivity": v(self.solar_absorptivity),
                },
                "optical": {
                    **self._emissivity(v),
                    "transmittance_per_band": {b: v(getattr(self, f"tau_{b}")) for b in BANDS},
                    "roughness_per_band": {b: v(getattr(self, f"rough_{b}")) for b in BANDS},
                    "angular_model": (
                        {"type": "empirical", "a": v(self.angular_a), "p": v(self.angular_p)}
                        if self.angular == "empirical"
                        else {"type": "constant"}
                    ),
                },
            },
        }

    def _emissivity(self, v) -> dict:
        """The authored emissivity in the chosen form (ADR 0175)."""
        if self.optical_form == "grey":
            return {"emissivity": v(self.eps_grey)}
        per_band = {"emissivity_per_band": {b: v(getattr(self, f"eps_{b}")) for b in BANDS}}
        if self.optical_form == "per_band":
            return per_band
        segments = []
        for file_key, q_key in (
            ("curve_file", "curve_quantity"),
            ("curve_file_2", "curve_quantity_2"),
        ):
            path = getattr(self, file_key)
            if not path:
                continue
            path = bpy.path.abspath(path)
            quantity = getattr(self, q_key)
            segments.append(
                path if quantity == "emissivity" else {"file": path, "quantity": quantity}
            )
        return {"spectral_emissivity": segments, **per_band}

    def execute(self, context):
        if self.optical_form == "curve" and not self.curve_file:
            self.report({"ERROR"}, "Pick a curve file, or choose another form")
            return {"CANCELLED"}
        if not NAME.match(self.name):
            self.report({"ERROR"}, "Give the material a name in Latin letters")
            return {"CANCELLED"}
        repo, python = prefs.settings(context)
        try:
            result = run_bridge(python, repo, "write-material", self._draft())
        except BridgeError as exc:
            self.report({"ERROR"}, f"Not created: {exc}")
            return {"CANCELLED"}
        for warning in result.get("warnings", []):
            self.report({"WARNING"}, warning)
        try:
            library_state.load(context)
        except BridgeError as exc:
            self.report({"WARNING"}, f"Created, but the library did not reload: {exc}")
        items = context.window_manager.irsim_library
        for i, item in enumerate(items):
            if item.name == result["name"]:
                context.scene.irsim.active_library_index = i
        parts = [ob for ob in context.selected_objects if ob.type == "MESH"]
        if self.assign_after and parts and context.mode == "OBJECT":
            assign_to_parts(parts, result["name"])
            from .scene_stats import refresh

            refresh(context)
        self.report({"INFO"}, f"Created {result['file']}")
        return {"FINISHED"}


class CheckCurve(Operator):
    """Read the picked curve through irsim's own loader and say what it covers"""

    bl_idname = "irsim.check_curve"
    bl_label = "Check curve"

    filepath: StringProperty(options={"SKIP_SAVE"})

    def execute(self, context):
        wm = context.window_manager
        if not self.filepath:
            wm.irsim_curve_check = "No file picked"
            return {"CANCELLED"}
        repo, python = prefs.settings(context)
        try:
            r = run_bridge(python, repo, "check-curve", {"file": bpy.path.abspath(self.filepath)})
        except BridgeError as exc:
            wm.irsim_curve_check = f"Refused: {exc}"
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        covers = [b.upper() for b, ok in r["covers"].items() if ok] or ["no whole band"]
        wm.irsim_curve_check = (
            f"{r['lo_um']:.3g}-{r['hi_um']:.3g} µm, {r['rows']} rows, values "
            f"{r['min']:.2f}-{r['max']:.2f}; covers {', '.join(covers)}"
            + ("" if r["has_source"] else "; no '# source:' line -- name it in Reference")
        )
        return {"FINISHED"}


classes = (NewMaterial, CheckCurve)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
