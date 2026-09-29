"""Stage 2 — atmosphere on the G-buffer: L' = τ(d) L + (1 − τ(d)) L_B(T_air), sky pixels untouched.

Per-pixel Beer–Lambert (irsim.atmosphere.beer_lambert) with the scalars the ``Atmosphere``
(M8.5) supplies for the frame time: one γ_B and one L_B(T_air) per band, both evaluated with the
pipeline's own LUT so the isothermal invariance holds bit-for-bit against stage 1. Runs on the
k× supersampled grid before the PSF (the atmosphere is a property of each ray, the blur of the
optics). Sky-pixel policy (ADR 0050): pixels under ``sky_mask`` carry the apparent sky
temperature, which already includes the atmosphere to space, and are returned untouched --
bit-identical -- in both the Beer–Lambert and the constant-τ (L1) paths.

docs/physics-model.md §13.4 stage 2, §7.1, §3.3
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from irsim.atmosphere.beer_lambert import apply_atmosphere, apply_tau_override
from irsim.atmosphere.layered import LayeredAtmosphere
from irsim.pipeline.core import PipelineConfig, PipelineState, Planes

__all__ = [
    "CloudOcclusion",
    "apply_atmosphere_gbuffer",
    "apply_layered_gbuffer",
    "atmosphere_stage",
    "AtmosphereStage",
    "cloud_from_planes",
    "compose_cloud",
]


@dataclass(frozen=True)
class CloudOcclusion:
    """The cloud between the camera and each pixel's hit, as three planes (AT.14).

    Produced by the deck march (:meth:`irsim.atmosphere.sky.SkyModel.cloud_occlusion`) and
    carried in the plane dict as ``cloud_transmittance``, ``cloud_radiance`` and
    ``cloud_range_m`` -- outside the M0.6 G-buffer contract, the way ``radiance_behind`` is,
    because they are stage-2 inputs rather than geometry. Sky pixels are ignored: their cloud is
    already in the sky temperature the bridge marched.
    """

    #: The band's transmittance through the cloud short of the hit; 1 where there is none.
    transmittance: NDArray[np.floating]
    #: Cloud emission toward the camera, before the air in front of the cloud; 0 without cloud.
    radiance: NDArray[np.floating]
    #: Range at which that emission is centred, metres, so the air in front can attenuate it.
    range_m: NDArray[np.floating]


def cloud_from_planes(planes: Planes) -> CloudOcclusion | None:
    """The three cloud planes, or ``None`` when the frame carries no cloud march."""
    if "cloud_transmittance" not in planes:
        return None
    return CloudOcclusion(
        np.asarray(planes["cloud_transmittance"]),
        np.asarray(planes["cloud_radiance"]),
        np.asarray(planes["cloud_range_m"]),
    )


def compose_cloud(
    radiance: NDArray[np.floating],
    cloud: CloudOcclusion,
    air_transmittance_to_cloud: NDArray[np.floating],
    sky_mask: NDArray[np.bool_] | None = None,
) -> NDArray[np.floating]:
    """``τ_c · L + τ_air(R_c) · L_cloud`` on every non-sky pixel; dtype preserved.

    ``radiance`` is the pixel *after* its own air (``τ_air(R) L_hit + L_path,air(R)``): the
    cloud sits between the camera and the hit, so the whole of that is seen through it, and
    the cloud's own emission reaches the camera through the air in front of the cloud only.
    The air *inside* the cloud path is counted twice at most over the cloud's own thickness,
    which its opacity makes invisible (docs/clouds-in-the-infrared.md).
    """
    l_in = np.asarray(radiance)
    if l_in.dtype == np.float16:
        raise TypeError("radiance is float16 (non-negotiable #2)")
    tau_c = np.asarray(cloud.transmittance, dtype=np.float64)
    if tau_c.shape != l_in.shape:
        raise ValueError(f"cloud planes {tau_c.shape} do not match the radiance {l_in.shape}")
    out = tau_c * l_in.astype(np.float64) + np.asarray(
        air_transmittance_to_cloud, dtype=np.float64
    ) * np.asarray(cloud.radiance, dtype=np.float64)
    if sky_mask is not None:
        out = np.where(np.asarray(sky_mask, dtype=bool), l_in, out)
    return np.asarray(
        out, dtype=l_in.dtype if np.issubdtype(l_in.dtype, np.floating) else np.float64
    )


def apply_atmosphere_gbuffer(
    radiance: NDArray[np.floating],
    distance_m: NDArray[np.floating],
    gamma_per_m: float,
    l_air: float,
    sky_mask: NDArray[np.bool_] | None = None,
    tau_override: float | None = None,
    cloud: CloudOcclusion | None = None,
) -> NDArray[np.floating]:
    """Per-pixel M8.1 on the radiance plane; dtype preserved (float16 refused).

    ``cloud`` (AT.14) lays the marched cloud over the result: the grey atmosphere's own
    ``exp(−γ R_c)`` attenuates the cloud's emission.
    """
    l_in = np.asarray(radiance)
    d = np.asarray(distance_m)
    if d.shape != l_in.shape:
        raise ValueError(f"distance_m shape {d.shape} != radiance shape {l_in.shape}")
    if tau_override is None:
        out = apply_atmosphere(l_in, d, gamma_per_m, l_air)
    else:
        out = apply_tau_override(l_in, tau_override, l_air)
    if sky_mask is not None:
        sky = np.asarray(sky_mask)
        if sky.dtype != np.bool_ or sky.shape != l_in.shape:
            raise ValueError("sky_mask must be a bool plane with the radiance shape")
        out = np.where(sky, l_in, out).astype(l_in.dtype, copy=False)
    if cloud is not None:
        to_cloud = np.exp(-float(gamma_per_m) * np.asarray(cloud.range_m, dtype=np.float64))
        out = compose_cloud(out, cloud, to_cloud, sky_mask)
    return out


def apply_layered_gbuffer(
    atmosphere: LayeredAtmosphere,
    band: str,
    t_s: float,
    radiance: NDArray[np.floating],
    distance_m: NDArray[np.floating],
    quantity: str,
    sky_mask: NDArray[np.bool_] | None = None,
    elevation_rad: NDArray[np.floating] | None = None,
    observer_height_m: float = 0.0,
    cloud: CloudOcclusion | None = None,
) -> NDArray[np.floating]:
    """MS.1 on the radiance plane, each pixel along its own slant ray when one is given (AT.1).

    ``cloud`` (AT.14) composes the marched cloud over every non-sky pixel afterwards, its
    emission attenuated by the same exponential sum over the range the emission came from.

    ``observer_height_m`` (AT.28) starts every column at the camera's height; 0 is the surface
    camera every scene had, bit for bit.

    ``elevation_rad`` is optional and its absence is the old behaviour exactly: a horizontal path
    for every pixel. That used to be the *only* behaviour, with ``0.0`` passed unconditionally --
    so every **resolved** pixel got surface-density extinction and surface-temperature emission
    over its whole slant range, while the **unresolved** point-target path beside it used the
    target's real elevation, and so did the sky behind it. Contrast therefore jumped at the
    resolved/unresolved handoff for no physical reason.
    """
    l_in = np.asarray(radiance)
    if l_in.dtype == np.float16:
        raise TypeError("radiance is float16 (non-negotiable #2)")
    d = np.asarray(distance_m)
    if d.shape != l_in.shape:
        raise ValueError(f"distance_m shape {d.shape} != radiance shape {l_in.shape}")
    if elevation_rad is None:
        out = atmosphere.apply(band, t_s, l_in, d, 0.0, quantity)  # type: ignore[arg-type]
    else:
        el = np.asarray(elevation_rad)
        if el.dtype == np.float16:
            raise TypeError("elevation_rad is float16 (non-negotiable #2)")
        if el.shape != l_in.shape:
            raise ValueError(f"elevation_rad shape {el.shape} != radiance shape {l_in.shape}")
        z0 = float(observer_height_m)
        tau = atmosphere.exponential_sum(band, t_s).transmittance(d, el, z0)
        path = atmosphere.path_radiance_plane(band, t_s, d, el, quantity, z0)  # type: ignore[arg-type]
        out = np.asarray(
            tau * l_in.astype(np.float64) + path,
            dtype=l_in.dtype if np.issubdtype(l_in.dtype, np.floating) else np.float64,
        )
    if sky_mask is not None:
        sky = np.asarray(sky_mask)
        if sky.dtype != np.bool_ or sky.shape != l_in.shape:
            raise ValueError("sky_mask must be a bool plane with the radiance shape")
        out = np.where(sky, l_in, out).astype(l_in.dtype, copy=False)
    if cloud is not None:
        el_c = 0.0 if elevation_rad is None else np.asarray(elevation_rad, dtype=np.float64)
        to_cloud = atmosphere.exponential_sum(band, t_s).transmittance(
            np.asarray(cloud.range_m, dtype=np.float64), el_c, float(observer_height_m)
        )
        out = compose_cloud(out, cloud, to_cloud, sky_mask)
    return out


def atmosphere_stage(planes: Planes, config: PipelineConfig, state: PipelineState) -> Planes:
    """Stage-2 entry point on the plane dict: replaces ``radiance``; identity without an
    Atmosphere."""
    radiance = np.asarray(planes["radiance"])
    if config.atmosphere is None:
        return {"radiance": radiance}
    if isinstance(config.atmosphere, LayeredAtmosphere):
        return {
            "radiance": apply_layered_gbuffer(
                config.atmosphere,
                config.sensor.sensor.band.band_id,
                state.t_s,
                radiance,
                np.asarray(planes["distance_m"]),
                config.quantity,
                sky_mask=planes.get("sky_mask"),
                elevation_rad=planes.get("elevation_rad"),
                observer_height_m=float(planes.get("observer_height_m", 0.0)),
                cloud=cloud_from_planes(planes),
            )
        }
    atm_state = config.atmosphere.state(state.t_s)
    band = config.sensor.sensor.band.band_id
    l_air = float(config.lut.lookup(np.float64(atm_state.t_air_k), config.quantity)[()])
    out = apply_atmosphere_gbuffer(
        radiance,
        np.asarray(planes["distance_m"]),
        atm_state.gamma_per_m[band],
        l_air,
        sky_mask=planes.get("sky_mask"),
        tau_override=config.tau_override,
        cloud=cloud_from_planes(planes),
    )
    return {"radiance": out}


class AtmosphereStage:
    name = "atmosphere"

    def __call__(self, planes: Planes, config: PipelineConfig, state: PipelineState) -> Planes:
        return atmosphere_stage(planes, config, state)
