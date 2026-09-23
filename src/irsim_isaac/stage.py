"""Stage furniture shared by every demo stage: the environment dome and the visible looks.

Both live here rather than in one stage's module because they are the *same* two decisions for
any stage -- what the companion visible frame is rendered against (ADR 0073) and what each
material looks like in visible light -- and a second stage that re-derived either would be free to
drift from the first. Neither touches the infrared path: a ``UsdLux.DomeLight`` is a light, not
geometry, so a ray that sees it still reports instance id 0 and an infinite ``DistanceToCameraSD``
and the background keeps coming from the sky model (ADR 0060).

docs/physics-model.md §15 T3; ADR 0073 (the dome), ADR 0047 (material precedence)
"""

from __future__ import annotations

import math
import os
import pathlib
import tempfile
from typing import Any

from irsim.io.exr import write_exr
from irsim_isaac.visible_sky import (
    LUMINOUS_EFFICACY_DAYLIGHT_LM_W,
    DomeSpec,
    dome_intensity,
    environment_map,
)

__all__ = [
    "DOME_HEIGHT",
    "VISIBLE_ALBEDO",
    "author_environment",
    "bind_visible_look",
]

#: Rows in the generated environment map; the map is twice as wide.
#:
#: 512 was chosen when the softest thing on the dome was the solar aureole, and a texel every 0.35
#: degrees is finer than that. A **marched cloud** is not soft: its opacity is 1 - e^{-tau} on a
#: path that crosses a cloud boundary, so it goes from clear to opaque within one texel, and a
#: 640x512 frame magnifies each texel to seven pixels -- which drew a visible staircase along
#: every cloud edge in the companion frame (AT.15, ADR 0130). 1024 halves it to 3.6 pixels and
#: buys real structure with it, at 27 s of bake and a 25 MB uncompressed float32 EXR, both once
#: per render. A clear dome is unaffected in cost: the expense is the march, not the resolution.
DOME_HEIGHT = 1024

#: Visible-light appearance of each target material: (linear sRGB albedo, roughness, metallic).
#: This is *appearance only* -- the infrared properties come from the material library through the
#: ``thermal:material`` override, and nothing in the sensor chain reads a colour. It exists so the
#: companion frame shows a white airframe and a black carbon one instead of six identical grey
#: squares, which is the difference between a picture you can check and a picture you cannot.
VISIBLE_ALBEDO: dict[str, tuple[tuple[float, float, float], float, float]] = {
    "painted_composite": ((0.72, 0.73, 0.75), 0.45, 0.0),
    "carbon_fibre": ((0.05, 0.05, 0.06), 0.30, 0.0),
    "aircraft_aluminium_painted": ((0.82, 0.82, 0.80), 0.35, 0.0),
    "propeller_rubber": ((0.06, 0.05, 0.05), 0.75, 0.0),
    # Maritime (MM.6): a white superstructure, a bare metal stack, and a painted hull.
    "car_paint_white": ((0.78, 0.78, 0.76), 0.30, 0.0),
    "bare_aluminium": ((0.62, 0.63, 0.64), 0.28, 1.0),
}
_DEFAULT_LOOK: tuple[tuple[float, float, float], float, float] = ((0.35, 0.35, 0.35), 0.6, 0.0)


def author_environment(
    stage: Any,
    dome: DomeSpec | None,
    texture_path: str | os.PathLike[str] | None,
    height: int,
) -> None:
    """The dome light and the sun, for the companion visible frame only (ADR 0073).

    With no ``dome`` this is the untextured grey hemisphere the stage used to carry. With one, the
    map of :func:`irsim_isaac.visible_sky.environment_map` is written to an EXR in cd/m2 and bound
    as a latitude-longitude texture, and a ``DistantLight`` is aimed along the same NOAA sun
    direction so the targets cast shadows from the sun the thermal solver is using.

    The sun's **disc** is the distant light and only the distant light: Preetham's distribution
    carries the aureole around the sun but not the disc itself, so the two do not double-count.
    """
    from pxr import Gf, Sdf, UsdGeom, UsdLux

    sun = UsdLux.DistantLight.Define(stage, "/World/SunForRgb")
    sky = UsdLux.DomeLight.Define(stage, "/World/SkyForRgb")
    if dome is None:
        sun.CreateIntensityAttr(1200.0)
        sky.CreateIntensityAttr(900.0)
        return

    image = environment_map(dome, height)
    path = pathlib.Path(
        texture_path
        if texture_path is not None
        # Per-process, because a shared working tree runs more than one of these at a time.
        else pathlib.Path(tempfile.gettempdir()) / f"irsim_env_dome_{os.getpid()}.exr"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    write_exr(path, image)

    exposure = dome_intensity(dome, image)
    sky.CreateTextureFileAttr().Set(Sdf.AssetPath(str(path)))
    sky.CreateTextureFormatAttr().Set(UsdLux.Tokens.latlong)
    sky.CreateIntensityAttr(exposure)
    # RTX hides a dome light from primary rays unless told otherwise, which would give a black
    # background lit by a sky nobody can see -- the exact failure this whole module is fixing.
    sky.GetPrim().CreateAttribute("visibleInPrimaryRay", Sdf.ValueTypeNames.Bool).Set(True)

    # A distant light emits along its own -Z, so -Z has to point the way the sunlight travels,
    # which is *away* from the sun.
    elevation = math.radians(dome.sun_elevation_deg)
    azimuth = dome.sun_azimuth_in_stage_rad()
    toward_sun = Gf.Vec3d(
        math.sin(azimuth) * math.cos(elevation),
        math.sin(elevation),
        -math.cos(azimuth) * math.cos(elevation),
    )
    rotation = Gf.Rotation(Gf.Vec3d(0.0, 0.0, -1.0), -toward_sun)
    UsdGeom.Xformable(sun).AddOrientOp().Set(Gf.Quatf(rotation.GetQuat()))
    sun.CreateAngleAttr(0.53)  # the solar disc, so contact shadows have the right softness
    sun.CreateIntensityAttr(exposure * LUMINOUS_EFFICACY_DAYLIGHT_LM_W * max(dome.dni_w_m2, 0.0))


def bind_visible_look(stage: Any, mesh: Any, material_name: str) -> None:
    """Give a target a visible-band surface. Appearance only; the infrared path never reads it.

    Safe against the material mapping (ADR 0047): the ``thermal:material`` attribute authored on
    the same prim takes precedence over any shader name, and these looks are named for the same
    material anyway, so the two agree even where the name is what gets used.
    """
    from pxr import Gf, Sdf, UsdShade

    # Under /World/Looks, deliberately not under /World/Targets: `prim_records` walks the target
    # root, and a Material prim in there would arrive at the resolver as an unmapped surface.
    path = f"/World/Looks/{material_name}"
    material = UsdShade.Material.Get(stage, path)
    if not material:
        material = UsdShade.Material.Define(stage, path)
        shader = UsdShade.Shader.Define(stage, f"{path}/Surface")
        shader.CreateIdAttr("UsdPreviewSurface")
        albedo, roughness, metallic = VISIBLE_ALBEDO.get(material_name, _DEFAULT_LOOK)
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*albedo))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
