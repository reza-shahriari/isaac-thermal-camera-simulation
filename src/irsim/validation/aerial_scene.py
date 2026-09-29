"""Synthetic aerial scene: a sky with an elevation gradient, a horizon, cloud, and targets.

The Tier 3 sky-phenomenology fixture (MS.8). It produces a G-buffer honouring the frozen
contract of :mod:`irsim.config.gbuffer` for a camera pitched up at a sky background, so the whole
chain -- MS.1's layered atmosphere, MS.2's sky model, MS.3's cloud, MS.6's point targets, M7.13's
reflected term -- can be exercised without launching Isaac Sim.

Geometry is the pinhole of §8.1 with the boresight raised by ``boresight_elevation_deg``. For
pixel (i, j) the camera-frame ray is (dx, −dy, f) with dx, dy the offsets from the principal point
in metres; rotating it about the x axis by the boresight elevation gives the world ray, and the
**elevation is its arcsine** -- exact, not a small-angle expansion, because the interesting part of
the sky profile is the first few degrees above the horizon where it is steepest.

Pixels above ``horizon_elevation_deg`` are sky: ``sky_mask`` true, ``distance_m`` 0, material id 0
(the UNMAPPED sentinel is not an error under the mask, ADR 0050) and ``temperature_k`` the
*apparent* sky temperature from the sky model, cloud included. Pixels below it are ground at the
environment preset's ground temperature. Targets whose projected extent reaches one native pixel
are rasterised into the G-buffer; smaller ones are returned as :class:`PointTarget` specs for
``run_frame`` to inject analytically (ADR 0071 -- a sub-pixel target must not be rasterised,
its flux would swing by tens of percent with sub-pixel position). A point target's radiance is
filled in here with :func:`irsim.validation.aerial.target_leaving_radiance`, so the sub-pixel and
resolved paths carry the same ε L_B(T) + (1 − ε) L_env the M7.13 stage would give them.

docs/physics-model.md §5.3, §8.1, §15 (Tier 3); ADR 0044, ADR 0050, ADR 0070, ADR 0071
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from irsim.atmosphere.sky import SkyModel
from irsim.config.sensor import SensorSpec
from irsim.materials.table import UNMAPPED_MATERIAL_ID, MaterialTable
from irsim.pipeline.environment import ground_temperature_k
from irsim.pipeline.point_target import PointTarget, fill_fraction
from irsim.radiometry.encoding import encode_temperature
from irsim.thermal.scene_forcing import SurfaceOrientation
from irsim.validation.aerial import AerialTarget, target_leaving_radiance

__all__ = [
    "SceneTarget",
    "AerialScene",
    "elevation_grid_rad",
    "build_aerial_gbuffer",
]

GROUND_DISTANCE_M = 5000.0  # a stand-in range for ground pixels (no terrain model in phase 1)


@dataclass(frozen=True)
class SceneTarget:
    """A target in the sky: what it is made of, how hot, how big and how far.

    ``size_m`` is the projected linear extent, so the projected area is ``size_m**2`` and the
    extent in native pixels is ``size_m f / (R p)``.
    """

    material: str
    temperature_k: float
    size_m: float
    range_m: float
    position_px: tuple[float, float]
    elevation_rad: float | None = None  # None: read from the pixel grid at its position
    #: AT.23: which way the surface faces. 0 is a deck looking up (sky view 1), 180 a belly
    #: looking down (sky view 0, it reflects the ground); the azimuth places the sun on it.
    tilt_deg: float = 0.0
    azimuth_deg: float = 180.0

    def __post_init__(self) -> None:
        if not self.temperature_k > 0.0 or not math.isfinite(self.temperature_k):
            raise ValueError(f"target {self.material!r}: temperature_k must be positive kelvin")
        if not self.size_m > 0.0 or not self.range_m > 0.0:
            raise ValueError(f"target {self.material!r}: size_m and range_m must be positive")

    def extent_px(self, sensor: SensorSpec) -> float:
        """Projected extent in native pixels."""
        f_m = sensor.optics.focal_length_mm * 1e-3
        return self.size_m * f_m / (self.range_m * sensor.fpa.pitch_um * 1e-6)


@dataclass(frozen=True)
class AerialScene:
    """The G-buffer plus everything a test needs to say what the right answer is."""

    planes: dict[str, NDArray[Any]]
    elevation_rad: NDArray[np.float64]  # per pixel, at the native grid
    sky_mask: NDArray[np.bool_]
    cloud_mask: NDArray[np.bool_]
    point_targets: tuple[PointTarget, ...] = ()
    resolved: tuple[tuple[SceneTarget, tuple[int, int, int, int]], ...] = ()  # (target, bbox)
    supersample: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, int]:
        h, w = self.sky_mask.shape
        return int(h), int(w)

    def column_elevation_deg(self, column: int | None = None) -> NDArray[np.float64]:
        """Elevation (degrees) down one image column -- the axis of the sky profile."""
        j = self.shape[1] // 2 if column is None else int(column)
        return np.degrees(self.elevation_rad[:, j])


def elevation_grid_rad(
    sensor: SensorSpec, boresight_elevation_deg: float, supersample: int = 1
) -> NDArray[np.float64]:
    """Ray elevation per pixel for a pinhole camera pitched up by ``boresight_elevation_deg``.

    Exact: the camera-frame ray is rotated about the x axis and the elevation is its arcsine.
    """
    k = int(supersample)
    if k < 1:
        raise ValueError("supersample must be >= 1")
    h, w = sensor.fpa.height * k, sensor.fpa.width * k
    pitch_m = sensor.fpa.pitch_um * 1e-6 / k
    f_m = sensor.optics.focal_length_mm * 1e-3
    dx = (np.arange(w, dtype=np.float64) + 0.5 - w / 2.0) * pitch_m
    dy = (np.arange(h, dtype=np.float64) + 0.5 - h / 2.0) * pitch_m
    ray = np.empty((h, w, 3), dtype=np.float64)
    ray[..., 0] = dx[None, :]
    ray[..., 1] = -dy[:, None]
    ray[..., 2] = f_m
    ray /= np.linalg.norm(ray, axis=-1, keepdims=True)
    el0 = math.radians(boresight_elevation_deg)
    y_world = ray[..., 1] * math.cos(el0) + ray[..., 2] * math.sin(el0)
    return np.asarray(np.arcsin(np.clip(y_world, -1.0, 1.0)), dtype=np.float64)


def build_aerial_gbuffer(
    sensor: SensorSpec,
    sky: SkyModel,
    materials: MaterialTable,
    t_s: float = 0.0,
    boresight_elevation_deg: float = 20.0,
    horizon_elevation_deg: float = 0.0,
    targets: Sequence[SceneTarget] = (),
    ground_material: str = "asphalt_dry",
    cloud_seed: int | None = None,
    supersample: int = 1,
    illumination: Any = None,
    boresight_azimuth_deg: float = 0.0,
) -> AerialScene:
    """Assemble the aerial G-buffer. ``cloud_seed`` None leaves the sky clear of structure.

    AT.23: every surface carries its orientation and its light. The ground faces up and is seen
    along each pixel's ray, so ``normal_dot_view`` is the ray's own sine of elevation, not 1; a
    target faces where its ``tilt_deg`` / ``azimuth_deg`` say, so a belly (tilt 180) has a sky
    view of 0 and reflects the ground, and its ``normal_dot_view`` is the cosine between its
    normal and the ray. With ``illumination`` (a `SolarIllumination` for this sensor) and a sky
    model that knows its site, the sun is placed from the weather's epoch and every lit surface
    gets ``l_sun`` through the preset's beam transmittance, ``sun_cos_incidence`` and a
    ``shadow_mask`` of ones (nothing occludes in this scene); without them the frame is the
    emissive-only picture it always was, bit for bit. Rays are taken along the boresight azimuth
    (``boresight_azimuth_deg``, degrees east of north) at each pixel's elevation.
    """
    k = int(supersample)
    elevation = elevation_grid_rad(sensor, boresight_elevation_deg, k)
    horizon = math.radians(horizon_elevation_deg)
    sky_mask = elevation > horizon
    if not sky_mask.any():
        raise ValueError(
            f"no sky in frame: boresight {boresight_elevation_deg}° with a "
            f"{sensor.hfov_deg:.1f}° horizontal field never rises above the horizon"
        )
    shape = (int(elevation.shape[0]), int(elevation.shape[1]))  # 2-D by construction

    # -- sky (cloud structure optional) -------------------------------------------------
    cloud_mask: NDArray[np.bool_] = np.zeros(shape, dtype=bool)
    if cloud_seed is not None:
        cloud_mask = np.asarray(
            sky.cloud_field(t_s, shape, cloud_seed).coverage & sky_mask, dtype=bool
        )
    el_for_sky = np.clip(elevation, 0.0, math.pi / 2)
    t_sky = np.asarray(sky.apparent_temperature_field(t_s, el_for_sky, cloud_mask))

    # -- ground -------------------------------------------------------------------------
    t_ground = ground_temperature_k(sky, t_s)
    temperature = np.where(sky_mask, t_sky, t_ground).astype(np.float32)
    distance = np.where(sky_mask, 0.0, GROUND_DISTANCE_M).astype(np.float32)
    material_id = np.where(sky_mask, UNMAPPED_MATERIAL_ID, materials.id_for(ground_material))
    material_id = material_id.astype(np.int32)
    sky_view = np.ones(shape, dtype=np.float32)  # sky pixels and flat ground both face up
    # AT.23: the ray through each pixel, and what faces it
    az = math.radians(boresight_azimuth_deg)
    ray = np.stack(
        [
            math.sin(az) * np.cos(elevation),
            math.cos(az) * np.cos(elevation),
            np.sin(elevation),
        ],
        axis=-1,
    )  # ENU, pointing away from the camera
    normal = np.zeros((*shape, 3), dtype=np.float64)
    normal[..., 2] = 1.0  # the ground faces up; sky pixels carry it too and never use it
    normal_dot_view = np.abs(np.sum(normal * -ray, axis=-1)).astype(np.float32)
    normal_dot_view[sky_mask] = 1.0

    # -- targets ------------------------------------------------------------------------
    point_targets: list[PointTarget] = []
    resolved: list[tuple[SceneTarget, tuple[int, int, int, int]]] = []
    for target in targets:
        extent = target.extent_px(sensor)
        x_px, y_px = target.position_px
        phi = fill_fraction(
            target.size_m**2,
            target.range_m,
            sensor.optics.focal_length_mm * 1e-3,
            sensor.pixel_area_m2,
        )
        if phi < 1.0:
            el = target.elevation_rad
            if el is None:
                i = min(max(int(y_px * k), 0), shape[0] - 1)
                j = min(max(int(x_px * k), 0), shape[1] - 1)
                el = float(max(elevation[i, j], 0.0))
            eps = float(materials.emissivity[materials.id_for(target.material)])
            leaving = target_leaving_radiance(
                AerialTarget(target.temperature_k, eps, target.range_m, sky_view_factor=1.0),
                sky,
                t_s,
            )
            point_targets.append(
                PointTarget(
                    area_m2=target.size_m**2,
                    range_m=target.range_m,
                    radiance=leaving,
                    position_px=(x_px, y_px),
                    elevation_rad=el,
                )
            )
            continue
        half = extent * k / 2.0
        cx, cy = x_px * k, y_px * k
        j0, j1 = int(round(cx - half)), int(round(cx + half))
        i0, i1 = int(round(cy - half)), int(round(cy + half))
        j0, j1 = max(j0, 0), min(j1, shape[1])
        i0, i1 = max(i0, 0), min(i1, shape[0])
        if j1 <= j0 or i1 <= i0:
            raise ValueError(f"target {target.material!r} falls outside the frame")
        temperature[i0:i1, j0:j1] = np.float32(target.temperature_k)
        distance[i0:i1, j0:j1] = np.float32(target.range_m)
        material_id[i0:i1, j0:j1] = materials.id_for(target.material)
        # AT.23: the target's own orientation, on every pixel it covers
        facet = SurfaceOrientation(tilt_deg=target.tilt_deg, azimuth_deg=target.azimuth_deg)
        n_t = np.asarray(facet.normal_enu(), dtype=np.float64)
        normal[i0:i1, j0:j1] = n_t
        sky_view[i0:i1, j0:j1] = np.float32(0.5 * (1.0 + math.cos(math.radians(target.tilt_deg))))
        block = -ray[i0:i1, j0:j1]
        normal_dot_view[i0:i1, j0:j1] = np.abs(block @ n_t).astype(np.float32)
        sky_mask[i0:i1, j0:j1] = False
        cloud_mask[i0:i1, j0:j1] = False
        resolved.append((target, (i0, j0, i1, j1)))

    planes: dict[str, NDArray[Any]] = {
        "temperature_k": temperature,
        "encoded_t": encode_temperature(temperature),
        "normal_dot_view": normal_dot_view,
        "distance_m": distance,
        "material_id": material_id,
        "sky_view_factor": sky_view,
        "sky_mask": sky_mask,
    }
    # AT.23: the sun on every lit surface, through the preset's beam transmittance
    if illumination is not None and sky.site is not None:
        from irsim.pipeline.solar import beam_transmittance

        terms = sky.solar_terms(t_s)
        el_sun = float(terms.elevation_deg)
        cos_sun = np.clip(np.sum(normal * np.asarray(terms.direction_enu), axis=-1), 0.0, 1.0)
        cos_sun[sky_mask] = 0.0
        tau_sun = float(beam_transmittance(sky.atmosphere.preset, sky.band, el_sun))
        shadow = np.ones(shape, dtype=np.float32)
        planes["sun_cos_incidence"] = cos_sun.astype(np.float32)
        planes["shadow_mask"] = shadow
        planes["l_sun"] = np.asarray(
            illumination.incident_radiance(
                cos_sun, elevation_deg=el_sun, tau_sun=tau_sun, shadow=shadow
            ),
            dtype=np.float64,
        )
    return AerialScene(
        planes=planes,
        elevation_rad=elevation,
        sky_mask=sky_mask,
        cloud_mask=cloud_mask,
        point_targets=tuple(point_targets),
        resolved=tuple(resolved),
        supersample=k,
        metadata={
            "t_s": float(t_s),
            "boresight_elevation_deg": float(boresight_elevation_deg),
            "horizon_elevation_deg": float(horizon_elevation_deg),
            "ground_temperature_k": float(t_ground),
            "cloud_seed": cloud_seed,
        },
    )
