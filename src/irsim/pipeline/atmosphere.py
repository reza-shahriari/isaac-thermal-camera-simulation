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

import numpy as np
from numpy.typing import NDArray

from irsim.atmosphere.beer_lambert import apply_atmosphere, apply_tau_override
from irsim.atmosphere.layered import LayeredAtmosphere
from irsim.pipeline.core import PipelineConfig, PipelineState, Planes

__all__ = [
    "apply_atmosphere_gbuffer",
    "apply_layered_gbuffer",
    "atmosphere_stage",
    "AtmosphereStage",
]


def apply_atmosphere_gbuffer(
    radiance: NDArray[np.floating],
    distance_m: NDArray[np.floating],
    gamma_per_m: float,
    l_air: float,
    sky_mask: NDArray[np.bool_] | None = None,
    tau_override: float | None = None,
) -> NDArray[np.floating]:
    """Per-pixel M8.1 on the radiance plane; dtype preserved (float16 refused)."""
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
) -> NDArray[np.floating]:
    """MS.1 on the radiance plane, each pixel along its own slant ray when one is given (AT.1).

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
    )
    return {"radiance": out}


class AtmosphereStage:
    name = "atmosphere"

    def __call__(self, planes: Planes, config: PipelineConfig, state: PipelineState) -> Planes:
        return atmosphere_stage(planes, config, state)
