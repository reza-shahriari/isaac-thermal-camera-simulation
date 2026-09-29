"""Rotor discs on a stage: from a prim's transform to the veils stage 2c composites (ADR 0081).

The physics is engine-free (:mod:`irsim.optics.rotor`, :mod:`irsim.pipeline.rotor_veil`); this is
the half that knows a rotor is bolted to a prim. It keeps the shape :mod:`motion_isaac` set: the
stage is an argument, nothing here imports ``omni`` or ``pxr`` at module scope, and every function
below is exercised without Isaac Sim.

**Nothing is authored.** ADR 0081's whole point is that a spinning rotor is not geometry, so a
mount adds no prim, no mesh and no material to the stage -- only a position, an axis and an rpm.
The disc appears in the infrared because the veil is composited, and appears nowhere in the
companion visible frame, which is the one honest asymmetry here and is noted in the ADR.

**Occlusion is a plane test, not a guess.** From below, a quadrotor's motor bell and arms stand in
front of the disc they carry, and a veil painted over them would put a blade on the wrong side of
the aircraft. Each pixel's ray is intersected with the disc *plane* and compared with the depth the
renderer reported: anything nearer occludes. Using the disc centre's range instead would be wrong
by up to the disc radius across the ellipse -- 0.36 m at 20 m, which is 18 pixels of arm.

docs/physics-model.md §13.3, §13.4 stage 2; ADR 0081
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from irsim.config.sensor import SensorSpec
from irsim.optics.motion import transform_points
from irsim.optics.projection import Intrinsics, usd_camera_to_opencv
from irsim.optics.rotor import RotorDisc, disc_ellipse, swept_angle_rad
from irsim.pipeline.rotor_veil import RotorVeil

__all__ = [
    "RotorMount",
    "disc_pose_world",
    "camera_ray_directions",
    "occlusion_mask",
    "build_rotor_veils",
]


@dataclass(frozen=True)
class RotorMount:
    """One rotor bolted to a prim: where it sits in that prim's frame, and how fast it turns.

    ``offset_m`` and ``axis`` are in the **prim's local frame**, so an airframe that pitches carries
    its rotors with it without anything here having to know it pitched. ``rpm`` is the caller's
    business frame by frame -- a multirotor's discs do not all turn at the same rate when it is
    manoeuvring -- and ``thermal_node`` names the solver whose temperature the blade takes.
    """

    disc: RotorDisc
    offset_m: tuple[float, float, float]
    axis: tuple[float, float, float] = (0.0, 1.0, 0.0)
    rpm: float = 3000.0
    thermal_node: str = "airframe"
    material: str = "propeller_rubber"
    phase_rad: float = 0.0
    #: ESTIMATED. A blade seen 75 degrees off its own axis shows mostly its edge and some of its
    #: underside, so neither 1 (all sky) nor 0 (all aircraft) is right. It matters little either
    #: way: ``propeller_rubber`` is eps = 0.95 in LWIR, so the reflected term is a twentieth of
    #: what leaves the blade, and the choice moves the disc by about a kelvin.
    sky_view_factor: float = 0.5

    def __post_init__(self) -> None:
        if float(np.linalg.norm(self.axis)) == 0.0:
            raise ValueError("axis must be a non-zero vector")


def disc_pose_world(
    local_to_world: Any, mount: RotorMount
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """``(centre, axis)`` of the disc in world space, from the prim's 4x4 (row-vector USD)."""
    matrix = np.asarray(local_to_world, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"expected a 4x4 transform, got {matrix.shape}")
    centre = transform_points(np.asarray(mount.offset_m, dtype=np.float64), matrix)
    # A direction takes the rotation only; normalising afterwards absorbs a uniform scale.
    axis = np.asarray(mount.axis, dtype=np.float64) @ matrix[:3, :3]
    norm = float(np.linalg.norm(axis))
    if norm == 0.0:
        raise ValueError("the prim's transform collapses the rotor axis")
    return np.asarray(centre), np.asarray(axis / norm)


def camera_ray_directions(intrinsics: Intrinsics) -> NDArray[np.float64]:
    """Unit ray directions in **OpenCV** camera space, one per pixel of ``intrinsics``' grid.

    Pinhole, ignoring distortion: this feeds a depth comparison whose answer changes over tens of
    centimetres, and the lens moves a ray by a fraction of a pixel. The projection of the disc
    itself does go through the distortion model (:func:`irsim.optics.rotor.disc_ellipse`).
    """
    rows, cols = np.indices((intrinsics.height, intrinsics.width), dtype=np.float64)
    x = (cols + 0.5 - intrinsics.cx_px) / intrinsics.fx_px
    y = (rows + 0.5 - intrinsics.cy_px) / intrinsics.fy_px
    rays = np.stack([x, y, np.ones_like(x)], axis=-1)
    return np.asarray(rays / np.linalg.norm(rays, axis=-1, keepdims=True))


def occlusion_mask(
    distance_m: Any,
    rays_cv: Any,
    centre_cv: Any,
    axis_cv: Any,
    *,
    sky_mask: Any = None,
    tolerance_m: float = 0.0,
) -> NDArray[np.bool_]:
    """``True`` where the renderer's depth puts a surface in **front** of the disc plane.

    The plane through ``centre_cv`` with normal ``axis_cv`` is hit at ``t = (n.c)/(n.d)`` along each
    unit ray, so a pixel is occluded when its reported distance is shorter than that. Rays parallel
    to the plane are never occluded, and neither are pixels with no geometry behind them.

    **Sky pixels carry ``distance_m = 0``, not infinity.** The G-buffer contract
    (:mod:`irsim.config.gbuffer`) stores the sentinel as zero so that a consumer ignoring
    ``sky_mask`` still sees tau = 1; read as a distance it is a surface at the camera, nearer than
    everything, and the first version of this function duly occluded **every pixel of the frame**
    and erased all four discs. A unit test written against ``inf`` -- what the AOV reports, not
    what the adapter stores -- agreed with it. So non-positive distances are treated as "no
    geometry", and ``sky_mask`` is accepted as the documented signal when the caller has it.

    ``tolerance_m`` pulls the plane towards the camera before the comparison; a small positive value
    keeps a surface that is *exactly* in the disc plane -- the motor bell whose top face the rotor
    sits on -- from flickering in and out of the mask with floating-point noise.

    **The mask means nothing outside the ellipse, and does not have to.** The plane is infinite, so
    a ray far off the disc can meet it kilometres away and be "occluded" by any ordinary surface;
    inside the disc, where :func:`irsim.optics.rotor.veil_radiance` reads it, the plane and the disc
    are the same thing. Bounding the plane to the disc would cost a second radius test per pixel to
    change an answer nothing consumes.
    """
    depth = np.asarray(distance_m, dtype=np.float64)
    rays = np.asarray(rays_cv, dtype=np.float64)
    if rays.shape[:2] != depth.shape[:2] or rays.shape[-1] != 3:
        raise ValueError(f"rays {rays.shape} do not match the depth plane {depth.shape}")
    normal = np.asarray(axis_cv, dtype=np.float64)
    centre = np.asarray(centre_cv, dtype=np.float64)
    denominator = rays @ normal
    with np.errstate(divide="ignore", invalid="ignore"):
        t_plane = float(normal @ centre) / denominator
    hit = np.isfinite(t_plane) & (t_plane > 0.0)
    has_geometry = np.isfinite(depth) & (depth > 0.0)
    if sky_mask is not None:
        has_geometry &= ~np.asarray(sky_mask, dtype=bool)
    return np.asarray(hit & has_geometry & (depth < t_plane - tolerance_m))


def build_rotor_veils(
    mounts: Sequence[RotorMount],
    local_to_world: Any,
    camera_position: Any,
    camera_to_world: Any,
    sensor: SensorSpec,
    supersample: int,
    blade_radiance: float,
    integration_s: float,
    *,
    distance_m: Any = None,
    sky_mask: Any = None,
    rays_cv: Any = None,
    occlusion_tolerance_m: float = 0.01,
) -> list[RotorVeil]:
    """Every mount on one prim, projected and ready for :func:`inject_rotor_veils`.

    ``local_to_world`` is that prim's 4x4 now; ``camera_position`` and ``camera_to_world`` are what
    :class:`~irsim_isaac.pipeline.ir_camera.IrCamera` already keeps. ``integration_s`` is how long
    the detector is actually sensitive -- a whole frame period for a bolometer, the integration
    time for a cooled photon detector -- and is what decides whether each disc draws an annulus or
    a set of arcs.

    ``distance_m`` is the G-buffer's depth plane on the **k× grid**; pass it to get occlusion, omit
    it to get veils that ignore what is in front of them. Pass ``sky_mask`` with it -- sky carries
    ``distance_m = 0`` by the G-buffer contract, and without the mask that zero reads as a surface
    at the camera. Discs behind the camera or off the frame
    are dropped here rather than by the compositor.
    """
    intrinsics = Intrinsics.from_sensor(sensor, supersample)
    distortion = sensor.effective_distortion
    if distance_m is not None and rays_cv is None:
        rays_cv = camera_ray_directions(intrinsics)

    out: list[RotorVeil] = []
    for mount in mounts:
        centre_world, axis_world = disc_pose_world(local_to_world, mount)
        centre_usd = (np.asarray(centre_world) - np.asarray(camera_position, dtype=np.float64)) @ (
            np.asarray(camera_to_world, dtype=np.float64)
        )
        axis_usd = np.asarray(axis_world) @ np.asarray(camera_to_world, dtype=np.float64)
        centre_cv = usd_camera_to_opencv(centre_usd)
        axis_cv = usd_camera_to_opencv(axis_usd)
        ellipse = disc_ellipse(centre_cv, axis_cv, mount.disc.radius_m, intrinsics, distortion)
        if ellipse is None:
            continue
        occluded = None
        if distance_m is not None:
            occluded = occlusion_mask(
                distance_m,
                rays_cv,
                centre_cv,
                axis_cv,
                sky_mask=sky_mask,
                tolerance_m=occlusion_tolerance_m,
            )
        out.append(
            RotorVeil(
                disc=mount.disc,
                ellipse=ellipse,
                swept_rad=swept_angle_rad(mount.rpm, integration_s),
                blade_radiance=blade_radiance,
                range_m=float(np.linalg.norm(centre_cv)),
                phase_rad=mount.phase_rad,
                occluded=occluded,
            )
        )
    return out
