"""The M10.1 geometry scene: what the renderer actually reports about shape and motion.

ADR 0014 settled the *transport* question (ids and geometry, never temperature in a colour AOV)
on an unlit, static, front-parallel ramp. On that scene the normals, ambient-occlusion and motion
annotators returned nothing at all, so the ADR explicitly left them to be re-probed here on a
scene that can actually exercise them: lit, with tilted and curved geometry, with surfaces facing
up / sideways / down, with something moving, and with the **camera away from the world origin** so
that a camera-space position AOV is distinguishable from a world-space one (on the ramp scene the
camera sat at the origin and the two frames coincided, which is why ``Camera3dPositionSD`` could
not be pinned down).

Everything the scene asserts is derived analytically from the authored transform, never from the
renderer, so a disagreement is a renderer finding rather than a tautology:

* **sphere** -- ``cos(theta) = sqrt(1 - (b/R)^2)`` for a ray of impact parameter ``b``. The
  cosine varies across the disc, so a pipeline that took the cosine against the optical axis
  instead of the per-pixel ray reads a bias well above the 0.01 tolerance.
* **tilted quad** -- Euclidean camera-to-centre distance, checked to 1 cm. ``DistanceToCameraSD``
  is a ray length; ``DistanceToImagePlaneSD`` is z-depth, and on a quad 24 degrees off axis the
  two differ by 9 %, far outside the tolerance.
* **three plates** -- normals +Y / +Z / -Y give ``V_s`` 1.0 / 0.5 / 0.0 (ADR 0045). They float in
  clear space so ambient occlusion is ~1 and the geometric correction is what is being measured.
* **bar** -- translated between steps by exactly the world distance that subtends 3 px at its
  depth, so the motion AOV should report 3.0 px/frame in whichever convention it uses.

docs/physics-model.md §13.3, §13.6; roadmap M10.1.
"""

from __future__ import annotations

import contextlib
import math
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from irsim_isaac.pipeline.point_bridge import world_positions

__all__ = [
    "GeometryScene",
    "Target",
    "build_geometry_scene",
    "probe_geometry_aovs",
    "translate_bar",
    "pitch_camera",
    "sphere_cos_theta",
    "detect_position_frame",
    "position_frame_residuals",
    "pinhole_rays",
    "POSITION_FRAME_HYPOTHESES",
    "configure_renderer",
    "RENDERER_SETTINGS",
    "survey_channels",
    "SURVEY_CANDIDATES",
]

#: Camera placed well away from the world origin: on the ADR 0014 ramp scene the camera sat at the
#: origin, so a world-space and a camera-space position AOV were indistinguishable.
CAMERA_POSITION: tuple[float, float, float] = (2.0, 1.0, 5.0)
MOTION_PX_PER_STEP = 3.0


@dataclass(frozen=True)
class Target:
    """One authored object and everything about it that can be predicted without rendering."""

    name: str
    prim_path: str
    centre: tuple[float, float, float]
    normal: tuple[float, float, float] | None = None
    distance_m: float = 0.0
    normal_dot_view: float | None = None
    normal_dot_up: float | None = None
    sky_view_factor: float | None = None
    radius_m: float | None = None
    semantic: str = ""


@dataclass
class GeometryScene:
    camera_path: str
    camera_position: tuple[float, float, float]
    focal_length_mm: float
    aperture_mm: float
    resolution: int
    up_axis: str
    targets: dict[str, Target]
    bar_step_m: float
    bar_depth_m: float
    errors: dict[str, str] = field(default_factory=dict)

    @property
    def focal_px(self) -> float:
        return self.focal_length_mm / self.aperture_mm * self.resolution

    def pixel_of(
        self, point: tuple[float, float, float] | NDArray[np.floating[Any]]
    ) -> tuple[int, int]:
        """Pinhole projection of a world point to (column, row). The camera looks -Z, unrotated."""
        p = np.asarray(point, dtype=np.float64) - np.asarray(self.camera_position, dtype=np.float64)
        depth = -float(p[2])
        if depth <= 0.0:
            raise ValueError(f"{point} is behind the camera")
        u = 0.5 * self.resolution + self.focal_px * float(p[0]) / depth
        v = 0.5 * self.resolution - self.focal_px * float(p[1]) / depth
        return int(round(u)), int(round(v))

    def distance_to(self, point: tuple[float, float, float]) -> float:
        p = np.asarray(point, dtype=np.float64) - np.asarray(self.camera_position, dtype=np.float64)
        return float(np.linalg.norm(p))


def sphere_cos_theta(
    scene: GeometryScene, columns: NDArray[np.integer[Any]], rows: NDArray[np.integer[Any]]
) -> NDArray[np.float64]:
    """Closed-form ``cos(theta)`` of the sphere target at the given pixels (NaN where it misses).

    For a ray whose perpendicular distance from the centre is ``b``, the near-side hit has
    ``sin(theta) = b / R`` between the surface normal and the view direction. Derived from the
    authored sphere, so it is an independent prediction of what the renderer should report.
    """
    sphere = scene.targets["sphere"]
    assert sphere.radius_m is not None
    centre = np.asarray(sphere.centre, dtype=np.float64) - np.asarray(
        scene.camera_position, dtype=np.float64
    )
    u = (np.asarray(columns, dtype=np.float64) - 0.5 * scene.resolution) / scene.focal_px
    v = (0.5 * scene.resolution - np.asarray(rows, dtype=np.float64)) / scene.focal_px
    direction = np.stack([u, v, -np.ones_like(u)], axis=-1)
    direction /= np.linalg.norm(direction, axis=-1, keepdims=True)

    along = direction @ centre
    impact_sq = float(centre @ centre) - along**2
    r = float(sphere.radius_m)
    cos_theta = np.sqrt(np.maximum(1.0 - impact_sq / (r * r), 0.0))
    return np.asarray(np.where(impact_sq < r * r, cos_theta, np.nan))


def detect_position_frame(
    scene: GeometryScene, position: NDArray[np.floating[Any]]
) -> dict[str, Any]:
    """World-space or camera-space position AOV? Decided against the authored sphere.

    The near surface point of the sphere on the ray through its centre is known exactly, so the
    two candidate frames predict two points a full camera-position apart. ADR 0014 could not make
    this call because its camera sat at the world origin.
    """
    sphere = scene.targets["sphere"]
    col, row = scene.pixel_of(sphere.centre)
    sample = np.asarray(position, dtype=np.float64)[row, col, :3]
    cam = np.asarray(scene.camera_position, dtype=np.float64)
    front = np.asarray(sphere.centre, dtype=np.float64) + np.array(
        [0.0, 0.0, float(sphere.radius_m or 0.0)]
    )
    err_world = float(np.linalg.norm(sample - front))
    err_camera = float(np.linalg.norm(sample - (front - cam)))
    return {
        "sample": [float(x) for x in sample],
        "expected_world": [float(x) for x in front],
        "expected_camera": [float(x) for x in (front - cam)],
        "err_world_m": err_world,
        "err_camera_m": err_camera,
        "verdict": "world" if err_world < err_camera else "camera",
    }


#: The three frames ``Camera3dPositionSD`` could plausibly be in, and what each predicts for a
#: point the distance AOV already measured. ADR 0014 recorded *world* (measured on a scene with a
#: translated but unrotated camera) and its M10.19 addendum recorded *camera* (measured on a scene
#: with a rotated camera at the origin). Both scenes are degenerate in one axis, and both readings
#: are also consistent with ``rotated_world`` -- the world point expressed in camera **axes** with
#: the translation left in -- which no scene so far could have distinguished from either.
POSITION_FRAME_HYPOTHESES: tuple[str, ...] = ("world", "camera", "rotated_world")


def pinhole_rays(resolution: int, focal_px: float, camera_to_world: Any) -> NDArray[np.float64]:
    """Unit world-axis ray direction for every pixel of a pinhole camera looking down camera -Z.

    The pixel convention matches :meth:`GeometryScene.pixel_of` (``u = W/2 + f x / z``, pixel
    centres on integers) so a ray and a projected point address the same pixel. ``camera_to_world``
    is the rotation in :func:`irsim_isaac.pipeline.gbuffer_isaac.ray_directions`' convention, so
    the camera-axis direction is taken to world as ``d @ rot.T``.
    """
    rot = np.asarray(camera_to_world, dtype=np.float64)
    if rot.shape == (4, 4):
        rot = rot[:3, :3]
    if rot.shape != (3, 3):
        raise ValueError(f"camera_to_world must be 3x3 or 4x4, got {rot.shape}")
    cols, rows = np.meshgrid(
        np.arange(resolution, dtype=np.float64), np.arange(resolution, dtype=np.float64)
    )
    x = cols - 0.5 * resolution
    y = 0.5 * resolution - rows
    d_cam = np.stack([x, y, np.full_like(x, -float(focal_px))], axis=2)
    d_cam /= np.linalg.norm(d_cam, axis=2, keepdims=True)
    return np.asarray(d_cam @ rot.T)


def position_frame_residuals(
    position: NDArray[np.floating[Any]],
    distance_m: NDArray[np.floating[Any]],
    ray_world: NDArray[np.floating[Any]],
    *,
    camera_position: Any,
    camera_to_world: Any,
) -> dict[str, Any]:
    """Score each :data:`POSITION_FRAME_HYPOTHESES` against a world point known per pixel.

    The oracle owes nothing to the position AOV: ``DistanceToCameraSD`` is an independently
    measured Euclidean ray length (M10.1: 1 cm on the tilted quad) and ``ray_world`` is the
    pinhole direction of that pixel, so the surface point is ``C + d * ray`` whatever the position
    AOV says. Each hypothesis decodes the AOV to a world point, and the residual is the distance
    between the two -- in metres, which is the unit the consequence is felt in.

    With ``C`` the camera translation and ``A`` its rotation in
    :func:`irsim_isaac.pipeline.gbuffer_isaac.ray_directions`' convention
    (``v_world = v_camera @ A.T``):

    ================  =====================  ===========================
    hypothesis        ``P`` holds            decodes to
    ================  =====================  ===========================
    ``world``         ``p``                  ``P``
    ``camera``        ``(p - C) @ A``        ``P @ A.T + C``
    ``rotated_world`` ``p @ A``              ``P @ A.T``
    ================  =====================  ===========================

    The three coincide only when ``C = 0`` **and** ``A = I``; one rotated, off-origin camera
    separates all three at once, ``camera`` and ``rotated_world`` by exactly ``|C|``. Returns the
    median residual per hypothesis, the winner, and the margin over the runner-up -- a margin near
    zero means the scene was degenerate and the verdict is not evidence, which is precisely how
    ADR 0014 and its M10.19 addendum came to disagree.

    docs/physics-model.md 13.3; roadmap M2.4.
    """
    pos = np.asarray(position, dtype=np.float64)[:, :, :3]
    d = np.asarray(distance_m, dtype=np.float64)
    ray = np.asarray(ray_world, dtype=np.float64)[:, :, :3]
    if pos.shape[:2] != d.shape or ray.shape[:2] != d.shape:
        raise ValueError(f"position {pos.shape[:2]}, distance {d.shape}, rays {ray.shape[:2]}")
    cam = np.asarray(camera_position, dtype=np.float64).reshape(3)
    rot = np.asarray(camera_to_world, dtype=np.float64)
    if rot.shape == (4, 4):
        rot = rot[:3, :3]
    if rot.shape != (3, 3):
        raise ValueError(f"camera_to_world must be 3x3 or 4x4, got {rot.shape}")

    # Sky pixels carry no geometry to score, and the ray length is how they are known: it is
    # `inf` there. That is also what excludes the position AOV's miss sentinel (measured: a point
    # 1000 m down the ray), so the sentinel value itself is never tested for -- a magnitude test
    # would throw away real geometry in a long-range aerial scene, where targets sit at km.
    valid = np.isfinite(d) & (d > 0.0) & np.isfinite(pos).all(axis=2)
    # `inf` distance on sky pixels would make `inf * 0` rays NaN; they are excluded anyway.
    truth = cam + np.where(valid, d, 0.0)[..., None] * ray
    # The decode is `world_positions`, not a second copy of it. This function is the one that
    # runs in-sim and that function is the one the render path calls, so a private `pos @ rot.T`
    # here would mean the arithmetic proved by a render and the arithmetic shipped in a frame
    # were two different expressions that merely happened to agree (roadmap IG.2). The
    # `rotated_world` hypothesis is the same decode with the translation left out, which is what
    # a zero camera position gives.
    decoded = {
        "world": world_positions(pos, frame="world"),
        "camera": world_positions(pos, frame="camera", camera_position=cam, camera_to_world=rot),
        "rotated_world": world_positions(
            pos, frame="camera", camera_position=np.zeros(3), camera_to_world=rot
        ),
    }
    residuals = {
        name: float(np.median(np.linalg.norm(p - truth, axis=2)[valid]))
        if valid.any()
        else float("inf")
        for name, p in decoded.items()
    }
    ranked = sorted(residuals, key=lambda name: residuals[name])
    return {
        "residual_m": residuals,
        "sample_pixels": int(valid.sum()),
        "verdict": ranked[0],
        "runner_up": ranked[1],
        "margin_m": residuals[ranked[1]] - residuals[ranked[0]],
    }


#: Renderer settings every M10.1 readback depends on. DLSS/TAA renders internally at a reduced
#: resolution and upscales -- ADR 0014 measured a 256 product coming back at 128, and a 384 product
#: silently became 192 here -- so every AOV read for measurement must run with AA off. Ambient
#: light is zeroed so the only illumination is the authored distant light.
RENDERER_SETTINGS: dict[str, Any] = {
    "/rtx/post/aa/op": 0,
    "/rtx/sceneDb/ambientLightIntensity": 0.0,
}


def configure_renderer(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Apply :data:`RENDERER_SETTINGS` and return what the settings store reads back.

    Call this **before** creating the render product. It is a function rather than a comment in one
    caller because forgetting it does not fail loudly: the AOVs simply arrive at half resolution
    and every pixel lookup silently addresses the wrong place.
    """
    import carb

    settings = carb.settings.get_settings()
    applied = {**RENDERER_SETTINGS, **(overrides or {})}
    for key, value in applied.items():
        settings.set(key, value)
    return {key: settings.get(key) for key in applied}


#: Every annotator worth trying per logical channel, widest first. ``survey_channels`` reports all
#: of them; :data:`irsim_isaac.pipeline.gbuffer_isaac.AOV_NAMES` holds the shortlist the adapter
#: actually uses, in preference order, once this survey has said which ones work.
SURVEY_CANDIDATES: dict[str, tuple[str, ...]] = {
    "distance": ("DistanceToCameraSD", "DistanceToImagePlaneSD", "distance_to_camera"),
    "position": ("Camera3dPositionSD", "PtWorldPos", "pointcloud"),
    "normal": (
        "PtWorldNormal",
        "NormalSD",
        "SmoothNormal",
        "BumpNormal",
        "normals",
        "PtSmoothNormal",
    ),
    "occlusion": ("AmbientOcclusion", "PtAmbientOcclusion", "ambient_occlusion"),
    "motion": ("Motion2d", "MotionVectorSD", "motion_vectors"),
    "instance": ("instance_segmentation", "instance_id_segmentation"),
    "semantic": ("semantic_segmentation",),
}


def survey_channels(
    scene: GeometryScene,
    *,
    candidates: dict[str, tuple[str, ...]] | None = None,
    settle_frames: int = 16,
    rt_subframes: int = 1,
    read_frames: int = 3,
) -> dict[str, Any]:
    """Attach **every** candidate annotator in turn and record what each one actually returns.

    :class:`AovReader` stops at the first name that answers, which is right in production and
    useless for a survey: ADR 0014's open questions are precisely *which* of several plausible
    names delivers, at what dtype, and -- the trap this scene exposed -- at what resolution. Some
    AOVs come back at the renderer's internal resolution rather than the render product's, which
    silently misaddresses every pixel lookup downstream, so the shape is recorded against the
    requested resolution for each one.
    """
    import omni.replicator.core as rep

    settings = configure_renderer()
    rp = rep.create.render_product(scene.camera_path, (scene.resolution, scene.resolution))
    rp_path = rp.path if hasattr(rp, "path") else str(rp)
    for _ in range(settle_frames):
        rep.orchestrator.step(rt_subframes=rt_subframes)

    report: dict[str, Any] = {
        "resolution": scene.resolution,
        "settings": settings,
        "registry": [],
        "channels": {},
    }
    with contextlib.suppress(Exception):
        report["registry"] = sorted(rep.AnnotatorRegistry.get_registered_annotators())

    for channel, names in (candidates or SURVEY_CANDIDATES).items():
        entries: dict[str, Any] = {}
        for name in names:
            entry: dict[str, Any] = {}
            entries[name] = entry
            try:
                anno = rep.AnnotatorRegistry.get_annotator(name, device="cpu")
                anno.attach(rp_path)
            except Exception as exc:  # noqa: BLE001
                entry["status"] = f"attach_error: {type(exc).__name__}: {exc}"
                continue
            try:
                for _ in range(read_frames):
                    rep.orchestrator.step(rt_subframes=rt_subframes)
                arr = _survey_array(anno.get_data())
            except Exception as exc:  # noqa: BLE001
                entry["status"] = f"read_error: {type(exc).__name__}: {exc}"
                arr = None
            with contextlib.suppress(Exception):
                anno.detach(rp_path)
            if arr is None:
                entry.setdefault("status", "no_data")
                continue
            entry["status"] = "ok"
            entry["dtype"] = str(arr.dtype)
            entry["shape"] = list(arr.shape)
            entry["full_resolution"] = bool(arr.shape[:2] == (scene.resolution, scene.resolution))
            if np.issubdtype(arr.dtype, np.floating):
                finite = arr[np.isfinite(arr)]
                entry["finite_fraction"] = float(np.isfinite(arr).mean())
                if finite.size:
                    entry["min_max"] = [float(finite.min()), float(finite.max())]
                entry["all_zero"] = bool(np.all(finite == 0.0)) if finite.size else True
            else:
                entry["unique_count"] = int(np.unique(arr).size)
        report["channels"][channel] = entries
    return report


def _survey_array(data: Any) -> NDArray[Any] | None:
    if data is None:
        return None
    if isinstance(data, dict):
        data = data.get("data", data)
    if hasattr(data, "numpy"):
        data = data.numpy()
    arr = np.asarray(data)
    return None if arr.size == 0 or arr.ndim < 2 else arr


# --- authoring ---------------------------------------------------------------------------------


def _quad(
    stage: Any,
    path: str,
    *,
    centre: tuple[float, float, float],
    half: float,
    rotate_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> Any:
    """A unit-normal quad in the XY plane (normal +Z), then rotated and translated into place."""
    from pxr import Gf, UsdGeom

    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.GetPointsAttr().Set(
        [
            Gf.Vec3f(-half, -half, 0.0),
            Gf.Vec3f(half, -half, 0.0),
            Gf.Vec3f(half, half, 0.0),
            Gf.Vec3f(-half, half, 0.0),
        ]
    )
    mesh.GetFaceVertexCountsAttr().Set([4])
    mesh.GetFaceVertexIndicesAttr().Set([0, 1, 2, 3])
    mesh.GetNormalsAttr().Set([Gf.Vec3f(0, 0, 1)] * 4)
    mesh.GetDoubleSidedAttr().Set(True)
    mesh.GetSubdivisionSchemeAttr().Set("none")
    xform = UsdGeom.Xformable(mesh)
    xform.AddTranslateOp().Set(Gf.Vec3d(*centre))
    xform.AddRotateXYZOp().Set(Gf.Vec3f(*rotate_xyz))
    return mesh


def _rotated_normal(rotate_xyz: tuple[float, float, float]) -> tuple[float, float, float]:
    """The +Z quad normal after USD's XYZ Euler rotation, so the test predicts it independently."""
    rx, ry, rz = (math.radians(a) for a in rotate_xyz)
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    mx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float64)
    my = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)
    mz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float64)
    n = (mz @ my @ mx) @ np.array([0.0, 0.0, 1.0])
    return (float(n[0]), float(n[1]), float(n[2]))


def _bind_grey(stage: Any, mesh: Any, mat_path: str, albedo: float = 0.5) -> None:
    from pxr import UsdShade

    from irsim_isaac.probe import author_omnipbr_usdshade

    material = author_omnipbr_usdshade(
        stage,
        mat_path,
        emissive=None,
        emissive_intensity=0.0,
        diffuse=(albedo, albedo, albedo),
    )
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)


def _label(prim: Any, label: str, errors: dict[str, str]) -> None:
    try:
        from isaacsim.core.experimental.utils.semantics import add_labels

        add_labels(prim, labels=[label], taxonomy="class")
    except Exception as exc:  # noqa: BLE001
        errors[f"{label}:add_labels"] = f"{type(exc).__name__}: {exc}"


def build_geometry_scene(
    *,
    resolution: int = 512,
    focal_length_mm: float = 24.0,
    aperture_mm: float = 20.955,
    sphere_radius_m: float = 1.2,
    plate_half_m: float = 1.0,
    export_path: str | None = None,
) -> GeometryScene:
    """Author the lit geometry scene described in the module docstring. Y is up."""
    import omni.usd
    from pxr import Gf, UsdGeom, UsdLux

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
    cam.GetClippingRangeAttr().Set(Gf.Vec2f(0.01, 1000.0))
    UsdGeom.Xformable(cam).AddTranslateOp().Set(Gf.Vec3d(*CAMERA_POSITION))

    # Lit, unlike the ADR 0014 ramp: several AOVs returned nothing on an unlit scene.
    light = UsdLux.DistantLight.Define(stage, "/World/Light")
    light.CreateIntensityAttr(1200.0)
    light.CreateAngleAttr(0.53)
    UsdGeom.Xformable(light).AddRotateXYZOp().Set(Gf.Vec3f(-45.0, 15.0, 0.0))

    cx, cy, cz = CAMERA_POSITION
    scene = GeometryScene(
        camera_path="/World/Camera",
        camera_position=CAMERA_POSITION,
        focal_length_mm=focal_length_mm,
        aperture_mm=aperture_mm,
        resolution=resolution,
        up_axis="Y",
        targets={},
        bar_step_m=0.0,
        bar_depth_m=0.0,
        errors=errors,
    )

    def add(target: Target) -> None:
        scene.targets[target.name] = target

    # -- sphere: cos(theta) varies across the disc ------------------------------------------
    sphere_centre = (cx, cy, cz - 9.0)
    sphere = UsdGeom.Sphere.Define(stage, "/World/Targets/Sphere")
    sphere.GetRadiusAttr().Set(sphere_radius_m)
    UsdGeom.Xformable(sphere).AddTranslateOp().Set(Gf.Vec3d(*sphere_centre))
    _bind_grey(stage, sphere, "/World/Looks/Sphere")
    _label(sphere.GetPrim(), "sphere", errors)
    add(
        Target(
            name="sphere",
            prim_path="/World/Targets/Sphere",
            centre=sphere_centre,
            radius_m=sphere_radius_m,
            distance_m=scene.distance_to(sphere_centre),
            semantic="sphere",
        )
    )

    # -- tilted quad: ray length vs z-depth --------------------------------------------------
    tilt = (0.0, -24.0, 0.0)
    tilted_centre = (cx + 1.7, cy + 0.9, cz - 5.8)
    tilted = _quad(
        stage, "/World/Targets/TiltedQuad", centre=tilted_centre, half=0.6, rotate_xyz=tilt
    )
    _bind_grey(stage, tilted, "/World/Looks/TiltedQuad")
    _label(tilted.GetPrim(), "tilted_quad", errors)
    n_tilt = _rotated_normal(tilt)
    to_cam = np.asarray(CAMERA_POSITION) - np.asarray(tilted_centre)
    to_cam = to_cam / np.linalg.norm(to_cam)
    add(
        Target(
            name="tilted_quad",
            prim_path="/World/Targets/TiltedQuad",
            centre=tilted_centre,
            normal=n_tilt,
            distance_m=scene.distance_to(tilted_centre),
            normal_dot_view=abs(float(np.dot(np.asarray(n_tilt), to_cam))),
            normal_dot_up=float(n_tilt[1]),
            sky_view_factor=0.5 * (1.0 + float(n_tilt[1])),
            semantic="tilted_quad",
        )
    )

    # -- three plates: V_s = 1.0 / 0.5 / 0.0 (ADR 0045) --------------------------------------
    # Rotations of the +Z quad: -90 deg about X faces +Y (up), 0 faces +Z, +90 faces -Y (down).
    plates = (
        ("plate_up", (-90.0, 0.0, 0.0), (cx - 0.1, cy - 1.9, cz - 6.0)),
        ("plate_vertical", (0.0, 0.0, 0.0), (cx - 1.7, cy + 0.15, cz - 6.0)),
        ("plate_under", (90.0, 0.0, 0.0), (cx - 0.1, cy + 1.9, cz - 6.0)),
    )
    for name, rot, centre in plates:
        mesh = _quad(
            stage, f"/World/Targets/{name}", centre=centre, half=plate_half_m, rotate_xyz=rot
        )
        _bind_grey(stage, mesh, f"/World/Looks/{name}")
        _label(mesh.GetPrim(), name, errors)
        n = _rotated_normal(rot)
        to_cam = np.asarray(CAMERA_POSITION) - np.asarray(centre)
        to_cam = to_cam / np.linalg.norm(to_cam)
        add(
            Target(
                name=name,
                prim_path=f"/World/Targets/{name}",
                centre=centre,
                normal=n,
                distance_m=scene.distance_to(centre),
                normal_dot_view=abs(float(np.dot(np.asarray(n), to_cam))),
                normal_dot_up=float(n[1]),
                sky_view_factor=0.5 * (1.0 + float(n[1])),
                semantic=name,
            )
        )

    # -- bar: exactly MOTION_PX_PER_STEP pixels of image motion per step ---------------------
    bar_depth = 7.0
    bar_centre = (cx + 1.4, cy - 1.3, cz - bar_depth)
    bar = _quad(
        stage, "/World/Targets/Bar", centre=bar_centre, half=0.45, rotate_xyz=(0.0, 0.0, 0.0)
    )
    _bind_grey(stage, bar, "/World/Looks/Bar", albedo=0.8)
    _label(bar.GetPrim(), "bar", errors)
    scene.bar_depth_m = bar_depth
    scene.bar_step_m = MOTION_PX_PER_STEP * bar_depth / scene.focal_px
    add(
        Target(
            name="bar",
            prim_path="/World/Targets/Bar",
            centre=bar_centre,
            normal=(0.0, 0.0, 1.0),
            distance_m=scene.distance_to(bar_centre),
            normal_dot_up=0.0,
            sky_view_factor=0.5,
            semantic="bar",
        )
    )

    if export_path:
        stage.GetRootLayer().Export(export_path)
    return scene


def pitch_camera(scene: GeometryScene, degrees: float) -> None:
    """Rotate the camera about the world X axis, to settle the frame of the normals AOV.

    With the camera axis-aligned, a world-space and a camera-space normal are numerically
    identical, so the measurement scene cannot tell them apart -- and getting it wrong silently
    corrupts every angular emissivity and sky-view factor downstream. A pitch breaks the
    degeneracy: the up-facing plate's normal stays exactly ``(0, 1, 0)`` in world space, while a
    camera-space normal would report ``cos(pitch)`` on the up axis instead.
    """
    import omni.usd
    from pxr import Gf, UsdGeom

    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(scene.camera_path)
    xform = UsdGeom.Xformable(prim)
    for op in xform.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeRotateXYZ:
            op.Set(Gf.Vec3f(float(degrees), 0.0, 0.0))
            return
    xform.AddRotateXYZOp().Set(Gf.Vec3f(float(degrees), 0.0, 0.0))


def translate_bar(scene: GeometryScene, steps: int) -> None:
    """Move the bar ``steps`` * ``bar_step_m`` to the right of where it was authored."""
    import omni.usd
    from pxr import Gf, UsdGeom

    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(scene.targets["bar"].prim_path)
    x, y, z = scene.targets["bar"].centre
    op = UsdGeom.Xformable(prim).GetOrderedXformOps()[0]
    op.Set(Gf.Vec3d(x + steps * scene.bar_step_m, y, z))


# --- readback ----------------------------------------------------------------------------------


def probe_geometry_aovs(
    scene: GeometryScene,
    *,
    settle_frames: int = 24,
    rt_subframes: int = 1,
    disable_aa: bool = True,
) -> dict[str, Any]:
    """Attach the M10.1 channels, render, and report what each one delivered.

    The report is the evidence for the ADR 0014 addendum: per channel, the annotator name that
    answered, its dtype and shape, and -- for the position channel -- whether the values are world
    or camera space, decided by comparing against the authored sphere centre in both frames.
    """
    import omni.replicator.core as rep

    from irsim_isaac.pipeline.gbuffer_isaac import (
        AOV_NAMES,
        UNVERIFIED_CHANNELS,
        AovReader,
    )

    report_settings = configure_renderer(None if disable_aa else {"/rtx/post/aa/op": 1})

    t0 = time.time()
    rp = rep.create.render_product(scene.camera_path, (scene.resolution, scene.resolution))
    rp_path = rp.path if hasattr(rp, "path") else str(rp)

    report: dict[str, Any] = {
        "resolution": scene.resolution,
        "camera_position": list(scene.camera_position),
        "settings": report_settings,
        "channels": {},
        "candidates": {k: list(v) for k, v in AOV_NAMES.items()},
        "errors": {},
    }
    # The survey attaches the unverified channels too -- establishing what they return is the
    # whole point of a probe, and is how one of them would ever stop being unverified (IG.5).
    reader = AovReader(
        rp_path, device="cpu", required=(), unverified=tuple(sorted(UNVERIFIED_CHANNELS))
    )
    try:
        reader.attach(settle_frames=settle_frames, rt_subframes=rt_subframes)
    except RuntimeError as exc:  # required=() makes this unreachable, kept for safety
        report["errors"]["attach"] = str(exc)
    report["resolved"] = dict(reader.resolved)
    report["failures"] = dict(reader.failures)

    try:
        aovs = reader.read()
    except RuntimeError as exc:
        report["errors"]["read"] = str(exc)
        reader.detach()
        return report

    for channel, name in reader.resolved.items():
        arr = getattr(
            aovs,
            {
                "distance": "distance_m",
                "position": "position",
                "normal": "normal",
                "occlusion": "occlusion",
                "motion": "motion",
                "instance": "instance_id",
                "semantic": "semantic_id",
            }[channel],
        )
        entry: dict[str, Any] = {"annotator": name}
        if arr is None:
            entry["status"] = "no_data"
        else:
            a = np.asarray(arr)
            entry.update(
                status="ok",
                dtype=str(a.dtype),
                shape=list(a.shape),
                finite_fraction=float(np.isfinite(a).mean()),
            )
            with contextlib.suppress(TypeError, ValueError):
                finite = a[np.isfinite(a)]
                if finite.size:
                    entry["min_max"] = [float(finite.min()), float(finite.max())]
        report["channels"][channel] = entry

    # Which frame is the position AOV in? ADR 0014's camera sat at the origin, so it could not say.
    if aovs.position is not None:
        report["position_frame"] = detect_position_frame(scene, np.asarray(aovs.position))

    report["timing_s"] = round(time.time() - t0, 2)
    reader.detach()
    return report
