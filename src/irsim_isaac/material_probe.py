"""A five-prim stage that exercises every branch of the ADR 0047 mapping precedence (M10.2).

Each prim is authored to make exactly one precedence rule fire, so a test can assert not just
*that* a prim resolved but *why*:

===================  ====================================  ===========================  ==========
prim                 authored                              expected material            rule
===================  ====================================  ===========================  ==========
``Body``             material ``Car_Paint_Red``             ``car_paint_black``          pattern
``Window``           material ``Glass_Clear`` **and** a     ``bare_aluminium``           override
                     ``thermal:material`` override
``Road``             semantic class ``road``, material      ``asphalt_dry``              semantic
                     name that matches no pattern
``Trim``             material ``Chrome_Trim`` **and** a     ``bare_aluminium``           override
                     ``thermal:material`` override
``Mystery``          nothing bound at all                   UNMAPPED (id 0)              miss
===================  ====================================  ===========================  ==========

``Window`` is the one that matters most. Its material is named ``Glass_Clear``, so the ``*glass*``
glob would claim it; the override must win, because that is what lets one wrong prim be fixed in
the asset without editing rules that apply to every scene. ``Mystery`` is the other: an asset
prim nobody remembered to map must stay a loud UNMAPPED, never acquire a plausible emissivity.

``Trim`` is a mirror **by assertion**, not by name (`AT.18`). Its material is still called
``Chrome_Trim``, and that name now resolves to nothing at all: no glob may reach a mirror, since
eps 0.09 makes a surface report the sky instead of itself and a name is weak evidence of a
polished finish. The prim therefore carries the same ``thermal:material`` override ``Window``
does, which is what an asset that really is polished has to do.

docs/physics-model.md §13.3; roadmap M10.2; ADR 0047.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = [
    "MaterialTarget",
    "MaterialScene",
    "build_material_scene",
    "expected_material_ids",
    "quad_half_width_px",
    "centre_ids",
    "EXPECTED",
]

CAMERA_POSITION: tuple[float, float, float] = (0.0, 0.0, 0.0)

#: (material name to author, semantic class, override, expected library material, expected rule).
EXPECTED: dict[str, tuple[str | None, str | None, str | None, str | None, str]] = {
    "Body": ("Car_Paint_Red", None, None, "car_paint_black", "pattern"),
    "Window": ("Glass_Clear", None, "bare_aluminium", "bare_aluminium", "override"),
    "Road": ("Unknown_Surface_7", "road", None, "asphalt_dry", "semantic"),
    # `AT.18`: `Trim` reached `bare_aluminium` through a `*chrome*` glob until that glob was
    # deleted, because a *name* is weak evidence of a polished finish -- artists call a matte
    # anodised housing "Metal", and at eps 0.09 a surface reports the sky rather than itself,
    # 38 K on a 300 K housing under a 250 K one. Polished metal is now something an asset
    # asserts, so this prim asserts it, and in doing so exercises that rule rather than the
    # rule AT.18 removed. `Body` still carries the pattern rung.
    "Trim": ("Chrome_Trim", None, "bare_aluminium", "bare_aluminium", "override"),
    "Mystery": (None, None, None, None, "miss"),
}


@dataclass(frozen=True)
class MaterialTarget:
    name: str
    prim_path: str
    centre: tuple[float, float]
    material_name: str | None
    semantic_class: str | None
    override: str | None
    expected_material: str | None
    expected_rule: str


@dataclass
class MaterialScene:
    camera_path: str
    camera_position: tuple[float, float, float]
    focal_length_mm: float
    aperture_mm: float
    resolution: int
    distance_m: float
    half_size_m: float
    targets: dict[str, MaterialTarget]
    errors: dict[str, str] = field(default_factory=dict)

    @property
    def focal_px(self) -> float:
        return self.focal_length_mm / self.aperture_mm * self.resolution

    def pixel_of(self, name: str) -> tuple[int, int]:
        """(column, row) of a target's centre. The camera is at the origin looking -Z."""
        x, y = self.targets[name].centre
        u = 0.5 * self.resolution + self.focal_px * x / self.distance_m
        v = 0.5 * self.resolution - self.focal_px * y / self.distance_m
        return int(round(u)), int(round(v))

    def patch_of(self, name: str, half_px: int = 3) -> tuple[slice, slice]:
        """A small window well inside a target, for asserting an id on many pixels at once."""
        col, row = self.pixel_of(name)
        return slice(row - half_px, row + half_px + 1), slice(col - half_px, col + half_px + 1)


def build_material_scene(
    *,
    resolution: int = 512,
    focal_length_mm: float = 24.0,
    aperture_mm: float = 20.955,
    distance_m: float = 4.0,
    half_size_m: float = 0.30,
    pitch_m: float = 0.75,
) -> MaterialScene:
    """Author the five quads. Non-emissive, lit, front-parallel: only the ids are under test."""
    import omni.usd
    from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade

    from irsim_isaac.probe import author_omnipbr_usdshade

    ctx = omni.usd.get_context()
    ctx.new_stage()
    stage = ctx.get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    stage.DefinePrim("/World", "Xform")
    stage.DefinePrim("/World/Looks", "Scope")
    stage.DefinePrim("/World/Targets", "Xform")
    errors: dict[str, str] = {}

    cam = UsdGeom.Camera.Define(stage, "/World/Camera")
    cam.GetFocalLengthAttr().Set(focal_length_mm)
    cam.GetHorizontalApertureAttr().Set(aperture_mm)
    cam.GetVerticalApertureAttr().Set(aperture_mm)
    cam.GetClippingRangeAttr().Set(Gf.Vec2f(0.01, 100.0))

    light = UsdLux.DistantLight.Define(stage, "/World/Light")
    light.CreateIntensityAttr(1000.0)

    names = list(EXPECTED)
    origin = -0.5 * pitch_m * (len(names) - 1)
    targets: dict[str, MaterialTarget] = {}
    for i, name in enumerate(names):
        material_name, semantic, override, expected, rule = EXPECTED[name]
        centre = (origin + i * pitch_m, 0.0)
        path = f"/World/Targets/{name}"

        mesh = UsdGeom.Mesh.Define(stage, path)
        x, y = centre
        z = -distance_m
        h = half_size_m
        mesh.GetPointsAttr().Set(
            [
                Gf.Vec3f(x - h, y - h, z),
                Gf.Vec3f(x + h, y - h, z),
                Gf.Vec3f(x + h, y + h, z),
                Gf.Vec3f(x - h, y + h, z),
            ]
        )
        mesh.GetFaceVertexCountsAttr().Set([4])
        mesh.GetFaceVertexIndicesAttr().Set([0, 1, 2, 3])
        mesh.GetNormalsAttr().Set([Gf.Vec3f(0, 0, 1)] * 4)
        mesh.GetDoubleSidedAttr().Set(True)
        mesh.GetSubdivisionSchemeAttr().Set("none")

        if material_name is not None:
            material = author_omnipbr_usdshade(
                stage,
                f"/World/Looks/{material_name}",
                emissive=None,
                emissive_intensity=0.0,
                diffuse=(0.5, 0.5, 0.5),
            )
            UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
        if override is not None:
            attr = mesh.GetPrim().CreateAttribute("thermal:material", Sdf.ValueTypeNames.String)
            attr.Set(override)
        if semantic is not None:
            try:
                from isaacsim.core.experimental.utils.semantics import add_labels

                add_labels(mesh.GetPrim(), labels=[semantic], taxonomy="class")
            except Exception as exc:  # noqa: BLE001
                errors[f"{name}:semantics"] = f"{type(exc).__name__}: {exc}"

        targets[name] = MaterialTarget(
            name=name,
            prim_path=path,
            centre=centre,
            material_name=material_name,
            semantic_class=semantic,
            override=override,
            expected_material=expected,
            expected_rule=rule,
        )

    return MaterialScene(
        camera_path="/World/Camera",
        camera_position=CAMERA_POSITION,
        focal_length_mm=focal_length_mm,
        aperture_mm=aperture_mm,
        resolution=resolution,
        distance_m=distance_m,
        half_size_m=half_size_m,
        targets=targets,
        errors=errors,
    )


def expected_material_ids(scene: MaterialScene, resolver: Any) -> dict[str, int]:
    """Library material id each target should end up with, from the resolver's own id order."""
    from irsim.materials.table import UNMAPPED_MATERIAL_ID

    out: dict[str, int] = {}
    for name, target in scene.targets.items():
        if target.expected_material is None:
            out[name] = UNMAPPED_MATERIAL_ID
        else:
            out[name] = int(resolver.id_for(target.expected_material))
    return out


def quad_half_width_px(scene: MaterialScene) -> float:
    """Half-width of one quad in pixels -- how far a patch may extend before leaving the target."""
    return float(scene.focal_px * scene.half_size_m / scene.distance_m)


def centre_ids(instance_ids: Any, scene: MaterialScene) -> dict[str, int]:
    """The raw instance id at each target's centre pixel."""
    ids = np.asarray(instance_ids)
    out: dict[str, int] = {}
    for name in scene.targets:
        col, row = scene.pixel_of(name)
        out[name] = int(ids[row, col])
    return out
