"""The maritime demo stage: vessels on open water, filmed against the sea (MM.6, ADR 0078).

The sea plays two different roles in one frame, and keeping them apart is the whole design.

**Infrared: the sea is computed, never rendered.** Pixels that land on water take their apparent
temperature from :class:`irsim.atmosphere.sea.SeaModel` at their own ray's depression angle. This is
the same decision ADR 0060 made for the sky and ADR 0078 restates for the sea, and the reason is
sampling, not laziness: at 3 km a Boson pixel spans 2.6 m and contains thousands of independent wave
facets, so one rendered normal per pixel is a different quantity from what the detector integrates,
not a coarse version of it. The water prims are therefore handed to ``IrCamera`` as **background**
geometry -- they are in the picture, they occlude, they set the horizon, and their temperature comes
from the analytic profile rather than from a material and a solver node.

**Visible: the sea is real geometry, because a picture of nothing teaches nothing.** The companion
frame gets an actual displaced water surface with wave structure and a dielectric response, so the
horizon, the sun glitter and the vessels' waterlines are where they should be and a person can look
at the frame and see whether the scene is what the code thinks it is.

The mesh carries the Earth's curvature (``r^2 / 2 R_e``), which is not a nicety at these ranges: a
flat plane's horizon is at 0 degrees of depression, the real one at 20 m of eye height is 0.1436
degrees down, and that difference is three Boson pixels of sea painted where the sky belongs.

Wave displacement is **tapered off where the mesh can no longer resolve it**. Beyond the ring
spacing that Nyquist allows, a displaced vertex is not a wave, it is an artefact the size of the
tessellation, so the amplitude goes to zero and the far water goes flat -- which is also what it
looks like, because out there the slope distribution is sub-pixel and shows up as roughness rather
than as shape.

docs/physics-model.md §15 T3; ADR 0078 (analytic sea), ADR 0073 (the visible dome), ADR 0047
(material precedence), ADR 0014 (ids are the transport)
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Any

from irsim_isaac.stage import DOME_HEIGHT, author_environment, bind_visible_look
from irsim_isaac.visible_sky import DomeSpec

__all__ = [
    "EARTH_RADIUS_M",
    "WAVE_COMPONENTS",
    "REFERENCE_WIND_M_S",
    "wave_train_for_wind",
    "Vessel",
    "DepartureTrack",
    "MaritimeDemoScene",
    "DEMO_VESSELS",
    "vessel_boxes",
    "build_maritime_demo",
    "describe",
]

EARTH_RADIUS_M = 6_371_000.0

#: Wave train for the visible surface at :data:`REFERENCE_WIND_M_S`: (wavelength m, amplitude m,
#: direction offset deg from the wind). A short ladder rather than a spectrum -- this is
#: appearance, and the radiometry takes its slope statistics from Cox-Munk (MM.2), not from these.
#: :func:`wave_train_for_wind` rescales it to the shared weather's wind.
WAVE_COMPONENTS: tuple[tuple[float, float, float, float], ...] = (
    # (wavelength m, amplitude m, direction offset deg from the wind, phase rad)
    # Nine components, not four. A handful of long sinusoids reads as corduroy however well it is
    # tessellated: the eye finds the periodicity immediately. What breaks it is (a) wavelengths in
    # a deliberately non-harmonic ratio, so no two components come back into phase across the
    # frame, (b) directional spreading of +/-70 degrees about the wind, which is roughly the
    # cos^2s spread a real wind sea has, and (c) an arbitrary fixed phase per component.
    (67.0, 0.255, 0.0, 0.00),
    (41.3, 0.191, -22.0, 1.73),
    (26.1, 0.140, 31.0, 4.12),
    (15.7, 0.101, -49.0, 2.55),
    (9.35, 0.070, 18.0, 5.61),
    (5.42, 0.047, 62.0, 0.94),
    (3.11, 0.030, -37.0, 3.38),
    (1.73, 0.018, 70.0, 5.02),
    (0.94, 0.010, -66.0, 1.21),
)

#: The vessels. Each is a hull at the waterline, a superstructure above it, and an exhaust stack --
#: three thermal nodes, because that is the maritime signature: a hull pinned near the sea by the
#: water it sits in, a superstructure near air temperature, and one small very hot thing.
#: (name, range_m, beam/length_m, lateral_offset_m, material, thermal node)
DEMO_VESSELS: tuple[tuple[str, float, float, float, str, str], ...] = (
    ("skiff", 600.0, 7.0, -60.0, "painted_composite", "hull"),
    ("patrol", 2100.0, 26.0, 140.0, "car_paint_white", "hull"),
    ("freighter", 5200.0, 95.0, -420.0, "painted_composite", "hull"),
)


#: Pierson-Moskowitz significant wave height for a fully developed sea, H_s = C U^2 (metres,
#: wind in m/s at 19.5 m). The ladder above is scaled to match it.
PM_WAVE_HEIGHT_COEFF = 0.0246


def _ladder_significant_height_m() -> float:
    """H_s = 4 sigma of the committed ladder, sigma^2 = sum(a^2)/2 for a sum of sinusoids."""
    variance = sum(a * a for _, a, _, _ in WAVE_COMPONENTS) / 2.0
    return 4.0 * math.sqrt(variance)


#: The wind the committed ladder corresponds to, from inverting Pierson-Moskowitz. Derived rather
#: than authored, so editing an amplitude above cannot leave the scaling silently inconsistent.
REFERENCE_WIND_M_S = math.sqrt(_ladder_significant_height_m() / PM_WAVE_HEIGHT_COEFF)


def wave_train_for_wind(wind_m_s: float) -> tuple[tuple[float, float, float, float], ...]:
    """The wave ladder scaled to a wind speed, wavelength and amplitude together.

    **Both** scale as U^2, which is the similarity a fully developed sea actually obeys: the
    Pierson-Moskowitz peak wavelength goes as U^2 and so does H_s, leaving the steepness -- and so
    the look -- scale invariant. Scaling amplitude alone, the obvious shortcut, makes a strong wind
    into a field of impossibly steep short waves.

    The wind is the shared ``WeatherSeries``'s, the same one Cox-Munk reads for the infrared sea's
    slope statistics (MM.2). A demo that let the visible surface and the radiometry disagree about
    how hard it is blowing would be exactly the split-brain CLAUDE.md #6 exists to prevent, and it
    would be invisible: the picture would look windy and read calm.
    """
    if wind_m_s < 0.0:
        raise ValueError("wind speed must be non-negative")
    scale = (wind_m_s / REFERENCE_WIND_M_S) ** 2
    return tuple(
        (wavelength * scale, amplitude * scale, offset, phase)
        for wavelength, amplitude, offset, phase in WAVE_COMPONENTS
    )


@dataclass(frozen=True)
class DepartureTrack:
    """A vessel leaving, straight away from a camera at the origin (MM.7).

    Three numbers with physical meanings rather than a waypoint list: it starts at
    ``start_range_m`` and steams directly away at ``speed_m_s``, so every quantity the film reports
    -- range, angular size, depression angle -- is available in closed form for a test to state
    what it should be.

    The camera does **not** track it. That is the point of filming a departure rather than a pass:
    against a fixed boresight the vessel climbs through the frame as it recedes, because its
    depression angle shrinks toward the horizon, and it crosses the sea's own angular gradient on
    the way. A tracking mount would hold it still and throw that away.
    """

    start_range_m: float = 150.0
    speed_m_s: float = 7.0
    camera_height_m: float = 20.0
    offset_m: float = 0.0

    def __post_init__(self) -> None:
        if self.start_range_m <= 0.0 or self.speed_m_s <= 0.0:
            raise ValueError("the vessel must start in front of the camera and be under way")

    def range_m(self, t_s: float) -> float:
        """Ground range at time ``t_s``; the slant range differs by under 1 % past 150 m."""
        return float(self.start_range_m + self.speed_m_s * float(t_s))

    def position_m(self, t_s: float) -> tuple[float, float, float]:
        """Where the waterline sits, with the Earth's curve dropping it as it goes."""
        r = self.range_m(t_s)
        drop = r * r / (2.0 * EARTH_RADIUS_M)
        return (float(self.offset_m), float(-self.camera_height_m - drop), float(-r))

    def depression_deg(self, t_s: float) -> float:
        r = self.range_m(t_s)
        return math.degrees(math.atan2(self.camera_height_m + r * r / (2.0 * EARTH_RADIUS_M), r))

    def pixels_across(self, t_s: float, size_m: float, ifov_mrad: float) -> float:
        return 1e3 * size_m / self.range_m(t_s) / ifov_mrad

    def horizon_time_s(self) -> float:
        """When the vessel reaches the geometric horizon and starts going hull-down."""
        far = math.sqrt(2.0 * self.camera_height_m * EARTH_RADIUS_M)
        return max(0.0, (far - self.start_range_m) / self.speed_m_s)


@dataclass(frozen=True)
class Vessel:
    """One vessel: where it floats, how big, and the three prims it is built from."""

    name: str
    range_m: float
    length_m: float
    offset_m: float
    material: str
    root_path: str
    hull_path: str
    superstructure_path: str
    stack_path: str

    def depression_deg(self, camera_height_m: float) -> float:
        """Depression of the waterline below the horizontal, allowing for the Earth's curve."""
        drop = self.range_m**2 / (2.0 * EARTH_RADIUS_M)
        return math.degrees(math.atan2(camera_height_m + drop, self.range_m))

    def angular_size_mrad(self) -> float:
        return 1e3 * self.length_m / self.range_m

    def pixels_across(self, ifov_mrad: float) -> float:
        return self.angular_size_mrad() / ifov_mrad


@dataclass
class MaritimeDemoScene:
    """The built stage and everything a caller needs to drive it."""

    camera_path: str
    camera_tilt_deg: float
    camera_height_m: float
    up_axis: str
    vessels: dict[str, Vessel]
    water_paths: tuple[str, ...]
    errors: dict[str, str] = field(default_factory=dict)

    @property
    def prim_to_target(self) -> dict[str, str]:
        """prim path -> thermal node, for every vessel prim. The water is deliberately absent:
        it is background, and its temperature comes from the sea model, not from a node."""
        out: dict[str, str] = {}
        for v in self.vessels.values():
            out[v.hull_path] = "hull"
            out[v.superstructure_path] = "superstructure"
            out[v.stack_path] = "stack"
        return out

    def horizon_elevation_deg(self) -> float:
        """Where the horizon sits relative to boresight, degrees (negative is below centre)."""
        horizon = -math.degrees(math.acos(EARTH_RADIUS_M / (EARTH_RADIUS_M + self.camera_height_m)))
        return horizon - self.camera_tilt_deg


def _taper(spacing_m: float, wavelength_m: float) -> float:
    """1 where the mesh resolves the wave, 0 where it cannot. Smooth in between.

    Two samples per wavelength is the hard limit; below four the shape is already a lie, so the
    fade runs between them. Without this the far rings, whose spacing is hundreds of metres, would
    carry displacement at the tessellation's own scale and read as a field of giant fake swells.
    """
    if spacing_m <= 0.0:
        return 1.0
    ratio = wavelength_m / spacing_m
    if ratio >= 4.0:
        return 1.0
    if ratio <= 2.0:
        return 0.0
    t = (ratio - 2.0) / 2.0
    return float(t * t * (3.0 - 2.0 * t))


def _author_water(
    stage: Any,
    camera_height_m: float,
    rings: int,
    sectors: int,
    half_angle_deg: float,
    wind_dir_deg: float,
    wind_speed_m_s: float,
    phase: float,
    max_depression_deg: float,
) -> str:
    """A curved, wave-displaced sea surface in front of the camera. Visible appearance only.

    **Rings are spaced uniformly in depression angle, not in radius**, which is the only spacing
    that matches what the camera samples. A pixel row is a fixed step in depression, so the ground
    distance it covers grows as r^2: at 20 m of eye height a Boson row spans 0.43 m of sea at 100 m
    and 4.3 km of sea at 10 km. A geometric radial grid -- the obvious choice, and the first one
    tried here -- gets that exponent wrong in both directions at once: ten times too coarse in the
    near field, where it flattens every wave shorter than 20 m into corduroy, and nine times finer
    than necessary out at the horizon, where it is paying for detail no pixel can see.

    The radius for a given depression comes from the curved surface, not a flat one:
    ``r^2 / 2 R_e + h = r tan(delta)``, whose near root is ``R_e tan(d) (1 - sqrt(1 - 2h/(R_e
    tan^2 d)))``. At the horizon the discriminant vanishes and it returns ``sqrt(2 h R_e)`` = 16 km
    for a 20 m eye height, where a flat plane would have said 8 km.

    Sectors cover a **wedge** around the view direction rather than the full circle. A 30.7 degree
    horizontal field does not need 360 degrees of water, and spending the vertex budget on the 92 %
    that is off-screen is what forced the near field to be coarse in the first place.
    """
    from pxr import Gf, Sdf, UsdGeom, UsdShade

    path = "/World/Water"
    mesh = UsdGeom.Mesh.Define(stage, path)
    wind = math.radians(wind_dir_deg)
    components = wave_train_for_wind(wind_speed_m_s)

    horizon = math.acos(EARTH_RADIUS_M / (EARTH_RADIUS_M + camera_height_m))

    def radius_at(delta: float) -> float:
        """Ground range whose waterline sits at depression ``delta`` on a curved surface."""
        t = math.tan(delta)
        disc = 1.0 - 2.0 * camera_height_m / (EARTH_RADIUS_M * t * t)
        if disc <= 0.0:
            return math.sqrt(2.0 * camera_height_m * EARTH_RADIUS_M)
        return EARTH_RADIUS_M * t * (1.0 - math.sqrt(disc))

    # Uniform in depression: the same coordinate the pixel rows are uniform in.
    deltas = [
        math.radians(max_depression_deg)
        + (horizon - math.radians(max_depression_deg)) * i / (rings - 1)
        for i in range(rings)
    ]
    radii = [radius_at(d) for d in deltas]

    # The wedge is centred on -Z, which is where a USD camera looks.
    centre_a = -0.5 * math.pi
    half = math.radians(half_angle_deg)

    points: list[Gf.Vec3f] = []
    normals: list[Gf.Vec3f] = []
    for ri, r in enumerate(radii):
        d_r = abs(radii[min(ri + 1, rings - 1)] - radii[max(ri - 1, 0)]) * 0.5
        spacing = max(d_r, 2.0 * half * r / sectors)
        drop = r * r / (2.0 * EARTH_RADIUS_M)
        for si in range(sectors):
            a = centre_a - half + 2.0 * half * si / (sectors - 1)
            x, z = r * math.cos(a), r * math.sin(a)
            h = 0.0
            nx = nz = 0.0
            for wavelength, amplitude, offset_deg, wave_phase in components:
                scale = _taper(spacing, wavelength)
                if scale <= 0.0:
                    continue
                k = 2.0 * math.pi / wavelength
                theta = wind + math.radians(offset_deg)
                kx, kz = k * math.cos(theta), k * math.sin(theta)
                arg = kx * x + kz * z + wave_phase + phase
                h += scale * amplitude * math.sin(arg)
                nx -= scale * amplitude * kx * math.cos(arg)
                nz -= scale * amplitude * kz * math.cos(arg)
            points.append(Gf.Vec3f(float(x), float(-camera_height_m + h - drop), float(z)))
            normals.append(Gf.Vec3f(float(nx), 1.0, float(nz)).GetNormalized())

    counts: list[int] = []
    indices: list[int] = []
    for ri in range(rings - 1):
        for si in range(sectors - 1):
            a = ri * sectors + si
            b = ri * sectors + si + 1
            c = (ri + 1) * sectors + si + 1
            d = (ri + 1) * sectors + si
            counts.append(4)
            indices.extend((a, b, c, d))

    mesh.GetPointsAttr().Set(points)
    mesh.GetNormalsAttr().Set(normals)
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    mesh.GetFaceVertexCountsAttr().Set(counts)
    mesh.GetFaceVertexIndicesAttr().Set(indices)
    mesh.GetSubdivisionSchemeAttr().Set("none")
    mesh.GetDoubleSidedAttr().Set(True)
    far = max(radii)
    mesh.CreateExtentAttr().Set(
        [
            Gf.Vec3f(-far, float(-camera_height_m - far * far / (2 * EARTH_RADIUS_M) - 2.0), -far),
            Gf.Vec3f(far, float(-camera_height_m + 2.0), far),
        ]
    )

    # A dielectric, not a grey Lambertian: water's look at range is almost entirely Fresnel
    # reflection of the sky, which `ior` gives for free and a diffuse albedo cannot imitate.
    look = "/World/Looks/sea_water"
    material = UsdShade.Material.Define(stage, look)
    shader = UsdShade.Shader.Define(stage, f"{look}/Surface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(0.008, 0.021, 0.032)
    )
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.055)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    shader.CreateInput("ior", Sdf.ValueTypeNames.Float).Set(1.333)
    shader.CreateInput("specular", Sdf.ValueTypeNames.Float).Set(0.62)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    return path


def vessel_boxes(length_m: float) -> dict[str, tuple[tuple[float, float, float], ...]]:
    """``part -> (centre, full extent)`` in the vessel's own frame, from its length alone.

    Centreline at x = 0, waterline at y = 0, amidships at z = 0, hull along X. The hull sits half
    in the water; the superstructure and the stack are stacked on it, with the stack small and
    tall -- it is the one part that is hundreds of kelvin above everything else, so its *size* is
    what decides whether it survives the PSF.

    Engine-free and separate from the authoring because a second maritime stage needs the same
    boxes: :mod:`irsim_isaac.vessel_pointwise` binds a temperature **field** to these faces, and a
    patch authored against a different set of numbers would sit off the prim and render part of a
    vessel as a field and part as a flat value (PT.10, and the failure ADR 0087 exists to remove).
    """
    beam = max(0.22 * length_m, 2.0)
    hull_h = max(0.10 * length_m, 1.2)
    return {
        "hull": ((0.0, 0.25 * hull_h, 0.0), (length_m, hull_h, beam)),
        "superstructure": (
            (0.0, hull_h + 0.30 * length_m * 0.18, 0.0),
            (0.30 * length_m, 0.36 * length_m * 0.18 * 2, 0.7 * beam),
        ),
        "stack": (
            (-0.06 * length_m, hull_h + 0.16 * length_m, 0.0),
            (0.07 * length_m, 0.10 * length_m, 0.22 * beam),
        ),
    }


def _box(
    stage: Any, path: str, centre: tuple[float, float, float], size: tuple[float, float, float]
) -> Any:
    from pxr import Gf, UsdGeom

    mesh = UsdGeom.Mesh.Define(stage, path)
    cx, cy, cz = centre
    hx, hy, hz = (0.5 * s for s in size)
    corners = [
        (cx - hx, cy - hy, cz - hz),
        (cx + hx, cy - hy, cz - hz),
        (cx + hx, cy + hy, cz - hz),
        (cx - hx, cy + hy, cz - hz),
        (cx - hx, cy - hy, cz + hz),
        (cx + hx, cy - hy, cz + hz),
        (cx + hx, cy + hy, cz + hz),
        (cx - hx, cy + hy, cz + hz),
    ]
    mesh.GetPointsAttr().Set([Gf.Vec3f(*c) for c in corners])
    mesh.GetFaceVertexCountsAttr().Set([4] * 6)
    mesh.GetFaceVertexIndicesAttr().Set(
        [0, 3, 2, 1, 4, 5, 6, 7, 0, 1, 5, 4, 2, 3, 7, 6, 0, 4, 7, 3, 1, 2, 6, 5]
    )
    mesh.GetSubdivisionSchemeAttr().Set("none")
    mesh.GetDoubleSidedAttr().Set(True)
    return mesh


def build_maritime_demo(
    *,
    camera_path: str = "/World/IrCamera",
    camera_tilt_deg: float = -5.0,
    camera_height_m: float = 20.0,
    vessels: tuple[tuple[str, float, float, float, str, str], ...] = DEMO_VESSELS,
    dome: DomeSpec | None = None,
    dome_texture_path: str | os.PathLike[str] | None = None,
    dome_height: int = DOME_HEIGHT,
    water: bool = True,
    water_rings: int = 420,
    water_sectors: int = 448,
    water_half_angle_deg: float = 26.0,
    water_max_depression_deg: float = 70.0,
    heading_deg: float = 0.0,
    wind_dir_deg: float = 35.0,
    wind_speed_m_s: float = REFERENCE_WIND_M_S,
    wave_phase: float = 0.0,
) -> MaritimeDemoScene:
    """Author the stage: a camera above the water tilted down, the sea, and the vessels on it.

    ``camera_tilt_deg`` is negative to look **down**. The default of -5 degrees puts the horizon
    about 40 % of the way up the frame, so most of the picture is the sea a maritime sensor spends
    its time staring at, with enough sky above it to show where the horizon actually is.

    ``water=False`` is the ablation: no water geometry at all. The infrared frame is **identical**,
    because the sea's radiometry never came from the mesh, and the visible frame loses its surface.
    Running both is the cheapest way to see which half of this stage is physics.
    """
    import omni.usd
    from pxr import Gf, Sdf, UsdGeom

    ctx = omni.usd.get_context()
    ctx.new_stage()
    stage = ctx.get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    stage.DefinePrim("/World", "Xform")
    stage.DefinePrim("/World/Targets", "Xform")
    stage.DefinePrim("/World/Looks", "Scope")

    errors: dict[str, str] = {}
    try:
        author_environment(stage, dome, dome_texture_path, dome_height)
    except Exception as exc:  # noqa: BLE001 - a dark companion frame must not stop the IR render
        errors["environment"] = f"{type(exc).__name__}: {exc}"

    water_paths: tuple[str, ...] = ()
    if water:
        try:
            water_paths = (
                _author_water(
                    stage,
                    camera_height_m,
                    water_rings,
                    water_sectors,
                    water_half_angle_deg,
                    wind_dir_deg,
                    wind_speed_m_s,
                    wave_phase,
                    water_max_depression_deg,
                ),
            )
        except Exception as exc:  # noqa: BLE001 - appearance only; the IR sea is analytic
            errors["water"] = f"{type(exc).__name__}: {exc}"

    built: dict[str, Vessel] = {}
    for name, range_m, length_m, offset_m, material, _node in vessels:
        root = f"/World/Targets/{name}"
        # The parts are authored in the vessel's OWN frame -- centreline at x = 0, waterline at
        # y = 0, amidships at z = 0 -- and the root Xform puts it on the water. That is what lets a
        # track move it: rewriting one translate op per frame moves the whole vessel, where prims
        # authored at absolute world positions would have to be rebuilt (MM.7).
        xform = UsdGeom.Xform.Define(stage, root)
        drop = range_m**2 / (2.0 * EARTH_RADIUS_M)
        xform.AddTranslateOp().Set(
            Gf.Vec3d(float(offset_m), float(-camera_height_m - drop), float(-range_m))
        )
        # Heading 0 lays the hull along +X, broadside to a camera looking down -Z; 90 turns it
        # bow-away, which is what a departing vessel shows.
        xform.AddRotateYOp().Set(float(heading_deg))

        parts = vessel_boxes(length_m)
        paths: dict[str, str] = {}
        for part, (centre, size) in parts.items():
            path = f"{root}/{part}"
            mesh = _box(stage, path, centre, size)
            paths[part] = path
            # The funnel is PAINTED, not bare metal. Bare aluminium is eps = 0.09 in LWIR, so a
            # bare-metal funnel reads cold however hot it is -- it reflects the sky instead of
            # radiating, and the engine signature this stage exists to show disappears. Real
            # funnels are painted steel, eps ~ 0.9. (Tried the other way round first; the funnel
            # was a dark rectangle at 155 C.)
            look = material if part != "stack" else "painted_composite"
            try:
                attr = mesh.GetPrim().CreateAttribute("thermal:material", Sdf.ValueTypeNames.String)
                attr.Set(look)
            except Exception as exc:  # noqa: BLE001 - a failed override is data, not a crash
                errors[f"{name}:{part}:override"] = f"{type(exc).__name__}: {exc}"
            try:
                bind_visible_look(stage, mesh, look)
            except Exception as exc:  # noqa: BLE001 - appearance only
                errors[f"{name}:{part}:look"] = f"{type(exc).__name__}: {exc}"

        built[name] = Vessel(
            name=name,
            range_m=float(range_m),
            length_m=float(length_m),
            offset_m=float(offset_m),
            material=material,
            root_path=root,
            hull_path=paths["hull"],
            superstructure_path=paths["superstructure"],
            stack_path=paths["stack"],
        )

    camera = UsdGeom.Camera.Define(stage, camera_path)
    camera.AddRotateXOp().Set(float(camera_tilt_deg))
    camera.GetClippingRangeAttr().Set(Gf.Vec2f(0.1, 1.0e6))

    return MaritimeDemoScene(
        camera_path=camera_path,
        camera_tilt_deg=float(camera_tilt_deg),
        camera_height_m=float(camera_height_m),
        up_axis="Y",
        vessels=built,
        water_paths=water_paths,
        errors=errors,
    )


def describe(scene: MaritimeDemoScene, ifov_mrad: float) -> list[dict[str, Any]]:
    """A row per vessel for a report: range, size, pixels across, and where its waterline sits."""
    return [
        {
            "name": v.name,
            "range_m": v.range_m,
            "length_m": v.length_m,
            "depression_deg": round(v.depression_deg(scene.camera_height_m), 4),
            "mrad": round(v.angular_size_mrad(), 3),
            "pixels": round(v.pixels_across(ifov_mrad), 2),
            "material": v.material,
        }
        for v in scene.vessels.values()
    ]
