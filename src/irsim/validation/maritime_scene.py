"""Synthetic maritime scene: sky above a horizon, sea below it, vessels at range (MM.8).

The maritime twin of :mod:`irsim.validation.aerial_scene`, and the Tier 3 maritime-phenomenology
fixture. It builds a G-buffer for a camera looking slightly **down** at open water, so the whole
maritime chain -- MM.2's Cox-Munk slopes, MM.3's depression profile, MM.4's skin temperature,
MS.2's sky above the horizon -- can be exercised on a frame without launching Isaac Sim.

**The sea is background, not geometry**, exactly as the rendered stage treats it (ADR 0078). A
pixel of sea at 5 km contains thousands of independent wave facets, so one normal per pixel is not
a coarse version of the right answer but a different quantity; what the detector integrates is the
*distribution*, which Cox & Munk give in closed form. So a water pixel carries the apparent
temperature :meth:`~irsim.atmosphere.sea.SeaModel.apparent_temperature_k` gives for its own ray's
depression, and it joins the background mask with ``distance_m = 0``.

That zero matters and is not a shortcut: the sea profile **already contains the atmospheric path**
(`SeaModel.radiance` applies tau and the path radiance over the slant range to each patch). Handing
stage 2 a non-zero distance as well would charge the same kilometres of air twice, which on a
12 km horizon ray is not a small error. The sky half of the frame has always worked this way
(ADR 0050); the sea simply joins it.

**Where the horizon is.** At the spherical horizon, ``horizon_depression_rad(camera_height_m)`` --
0.1436 degrees down for a 20 m eye height, three Boson pixels below where a flat-earth scene would
put it. Above that elevation the pixel is sky; below it, sea.

docs/physics-model.md §5.3, §8.1, §15 (Tier 3); ADR 0078 (analytic sea), ADR 0050, ADR 0080
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from irsim.atmosphere.sea import SeaModel
from irsim.atmosphere.sea_envelope import EnvelopeReport, envelope_report
from irsim.config.sensor import SensorSpec
from irsim.materials.table import UNMAPPED_MATERIAL_ID, MaterialTable
from irsim.pipeline.point_target import PointTarget, fill_fraction
from irsim.radiometry.encoding import encode_temperature
from irsim.validation.aerial import AerialTarget, target_leaving_radiance
from irsim.validation.aerial_scene import SceneTarget, elevation_grid_rad

__all__ = [
    "MaritimeScene",
    "build_maritime_gbuffer",
]


@dataclass(frozen=True)
class MaritimeScene:
    """The G-buffer plus what a test needs to state the right answer independently."""

    planes: dict[str, NDArray[Any]]
    elevation_rad: NDArray[np.float64]
    depression_rad: NDArray[np.float64]  # positive below the horizon, 0 above it
    sky_mask: NDArray[np.bool_]  # the *background* mask: sky and sea both
    sea_mask: NDArray[np.bool_]
    point_targets: tuple[PointTarget, ...] = ()
    resolved: tuple[tuple[SceneTarget, tuple[int, int, int, int]], ...] = ()
    supersample: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, int]:
        h, w = self.sky_mask.shape
        return int(h), int(w)

    def column_depression_deg(self, column: int | None = None) -> NDArray[np.float64]:
        """Depression (degrees) down one image column -- the axis of the sea profile."""
        j = self.shape[1] // 2 if column is None else int(column)
        return np.degrees(self.depression_rad[:, j])

    def envelope_report(self) -> EnvelopeReport:
        """How much of this frame's sea lies past the sea model's validated view angle (SE.1).

        Recomputed from the depression plane rather than stored, so it cannot drift from the
        geometry it describes; ``metadata["envelope"]`` holds the same object for a caller that
        only wants to print it.
        """
        return envelope_report(
            float(self.metadata["camera_height_m"]), self.depression_rad, self.sea_mask
        )

    def sea_rows(self, column: int | None = None) -> NDArray[np.intp]:
        """Row indices of the water pixels in one column, nearest the horizon first."""
        j = self.shape[1] // 2 if column is None else int(column)
        return np.asarray(np.flatnonzero(self.sea_mask[:, j]), dtype=np.intp)


def build_maritime_gbuffer(
    sensor: SensorSpec,
    sky: Any,
    sea: SeaModel,
    materials: MaterialTable,
    t_s: float = 0.0,
    boresight_elevation_deg: float = -1.0,
    targets: Sequence[SceneTarget] = (),
    cloud_seed: int | None = None,
    supersample: int = 1,
) -> MaritimeScene:
    """Assemble the maritime G-buffer. The boresight is normally **negative**: look down.

    ``sky`` is the same :class:`~irsim.atmosphere.sky.SkyModel` the ``sea`` was built on, and that
    is checked rather than trusted -- a scene whose sea reflects one sky while its sky pixels come
    from another is the CLAUDE.md #6 failure in its most literal form, and it would render as a
    perfectly plausible frame.
    """
    if sea.sky is not sky:
        raise ValueError(
            "the sea must reflect the same SkyModel the frame's sky pixels come from "
            "(CLAUDE.md #6): the sea is mostly reflected sky, so two sky models means a horizon "
            "with different weather on each side of it"
        )
    k = int(supersample)
    elevation = elevation_grid_rad(sensor, boresight_elevation_deg, k)
    horizon_el = -sea.horizon_rad
    sea_mask = elevation <= horizon_el
    if not sea_mask.any():
        raise ValueError(
            f"no sea in frame: boresight {boresight_elevation_deg}° with a "
            f"{sensor.hfov_deg:.1f}° horizontal field never dips below the horizon at "
            f"{math.degrees(sea.horizon_rad):.4f}° of depression"
        )
    shape = (int(elevation.shape[0]), int(elevation.shape[1]))
    depression = np.where(sea_mask, -elevation, 0.0)

    # -- sky above the horizon ----------------------------------------------------------
    cloud_mask: NDArray[np.bool_] = np.zeros(shape, dtype=bool)
    above = ~sea_mask
    if cloud_seed is not None:
        cloud_mask = np.asarray(
            sky.cloud_field(t_s, shape, cloud_seed).coverage & above, dtype=bool
        )
    el_for_sky = np.clip(elevation, 0.0, math.pi / 2)
    t_sky = np.asarray(sky.apparent_temperature_field(t_s, el_for_sky, cloud_mask))

    # -- sea below it -------------------------------------------------------------------
    # One interpolation over the whole plane rather than a masked one: the profile LUT is the
    # cheap part and the mask costs a copy either way.
    t_sea = np.asarray(sea.apparent_temperature_k(t_s, np.maximum(depression, sea.horizon_rad)))

    temperature = np.where(sea_mask, t_sea, t_sky).astype(np.float32)
    # Both halves are background: their apparent temperatures already carry their own path.
    background = np.ones(shape, dtype=bool)
    distance = np.zeros(shape, dtype=np.float32)
    material_id = np.full(shape, UNMAPPED_MATERIAL_ID, dtype=np.int32)
    sky_view = np.ones(shape, dtype=np.float32)

    # -- vessels ------------------------------------------------------------------------
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
                el = float(elevation[i, j])
            eps = float(materials.emissivity[materials.id_for(target.material)])
            # A vessel sits *on* the water with its sides facing outward, so what it reflects is
            # overwhelmingly sky, not sea -- the same environment the aerial path assumes. A hull
            # reflecting its own wake is a second-order effect this fixture does not claim.
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
                    elevation_rad=max(el, 0.0),
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
        background[i0:i1, j0:j1] = False
        sea_mask[i0:i1, j0:j1] = False
        cloud_mask[i0:i1, j0:j1] = False
        resolved.append((target, (i0, j0, i1, j1)))

    planes: dict[str, NDArray[Any]] = {
        "temperature_k": temperature,
        "encoded_t": encode_temperature(temperature),
        "normal_dot_view": np.ones(shape, dtype=np.float32),
        "distance_m": distance,
        "material_id": material_id,
        "sky_view_factor": sky_view,
        "sky_mask": background,
    }
    return MaritimeScene(
        planes=planes,
        elevation_rad=elevation,
        depression_rad=depression,
        sky_mask=background,
        sea_mask=sea_mask,
        point_targets=tuple(point_targets),
        resolved=tuple(resolved),
        supersample=k,
        metadata={
            "t_s": float(t_s),
            "boresight_elevation_deg": float(boresight_elevation_deg),
            "horizon_depression_deg": math.degrees(sea.horizon_rad),
            "camera_height_m": sea.camera_height_m,
            "bulk_sst_k": sea.bulk_sst_k,
            "skin_temperature_k": sea.skin_temperature_k(t_s),
            "cloud_seed": cloud_seed,
            # SE.1: the frame's own answer to "is any of this backed by a measurement?"
            "envelope": envelope_report(sea.camera_height_m, depression, sea_mask),
        },
    )
