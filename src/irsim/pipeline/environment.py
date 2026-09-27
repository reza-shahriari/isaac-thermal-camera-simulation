"""The reflected environment term of stage 1 (§5.3 a, §4.1, ADR 0045).

    L = ε L_B(T_s) + (1 − ε) L_env,        L_env = V_s L_sky,eff + (1 − V_s) L_ground

In a reflective or mixed band the ground is also lit (AT.20, ADR 0153, spec issue S55):
L_ground = L_B(T_ground) + ρ_B,ground E_B / π, where E_B is the weather's own DNI and DHI through
the band's two solar fractions. Without it a downward-facing surface (V_s → 0) saw only the
ground's *thermal* emission, which is ~0 in NIR and SWIR, and a white underside rendered black.

The reflected part uses (1 − ε) = ρ + τ: a transmitting material passes the environment behind
it (L_behind = L_env until a second ray exists, ADR 0046). L_sky,eff comes from the SkyModel's tilt
LUT indexed by the pixel's sky-view factor (the unoccluded relation V_s = (1 + cos β)/2);
L_ground = L_B(T_ground) with T_ground from the environment preset's ground mode (``air``: the
shared weather's T_air; ``fixed``: the authored value; ``sea``: the bulk SST, because what an
object reflects from below is the water right around it at steep incidence, where ε ≈ 0.99 --
*not* the angular sea profile a background ray sees, which is ADR 0078's SeaModel; ``solver``:
M6.12).

``sky_view_factor(n·up, occlusion) = occlusion · (1 + n·up)/2`` is what an adapter or fixture
should write into the G-buffer: the cosine-weighted fraction of the sky a plane of tilt β sees
(1 facing up, 1/2 for a wall, 0 facing down), times an occlusion factor -- raw ambient
occlusion alone is wrong for walls, which see half the sky unoccluded.

docs/physics-model.md §5.3(a), §4.1, §13.7 option 1
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from irsim.atmosphere.sky import SkyModel
from irsim.radiometry.lut import BandLUT, Quantity

__all__ = ["sky_view_factor", "ground_temperature_k", "environment_radiance"]


def sky_view_factor(normal_dot_up: Any, occlusion: Any = 1.0) -> NDArray[np.float64]:
    """V_s = occlusion (1 + n·up)/2: the cosine-weighted sky fraction of a tilted plane."""
    n_up = np.asarray(normal_dot_up, dtype=np.float64)
    occ = np.asarray(occlusion, dtype=np.float64)
    if np.any((n_up < -1.0 - 1e-9) | (n_up > 1.0 + 1e-9)):
        raise ValueError("normal_dot_up must lie in [-1, 1]")
    if np.any((occ < 0.0) | (occ > 1.0)):
        raise ValueError("occlusion must lie in [0, 1] (1 = unoccluded)")
    return np.asarray(occ * 0.5 * (1.0 + np.clip(n_up, -1.0, 1.0)), dtype=np.float64)


def ground_temperature_k(sky: SkyModel, t_s: float) -> float:
    """T_ground per the environment preset's ground mode."""
    ground = sky.environment.ground
    if ground.mode == "air":
        return float(sky.weather.at(t_s).t_air_k)
    if ground.mode == "fixed":
        assert ground.fixed_temperature_k is not None
        return float(ground.fixed_temperature_k)
    if ground.mode == "sea":
        # The **reflected-environment** value, and only that. What a surface in the scene sees
        # below its own horizon is the water immediately around it, seen at steep angles where
        # water's emissivity is ~0.99 and it radiates essentially at its own temperature -- so one
        # scalar is right here, and it is the SST. The *background* sea, which a ray travels
        # kilometres across at grazing incidence, is a completely different quantity and comes
        # from irsim.atmosphere.sea.SeaModel, which needs the ray's depression angle (ADR 0078).
        assert ground.bulk_sst_k is not None
        return float(ground.bulk_sst_k)
    raise ValueError("ground.mode 'solver' needs the environment solver (M6.12)")


def environment_radiance(
    sky: SkyModel,
    lut: BandLUT,
    t_s: float,
    sky_view: NDArray[np.floating],
    quantity: Quantity = "lb",
) -> NDArray[np.float32]:
    """L_env per pixel = V_s L_sky,eff(V_s) + (1 − V_s) L_ground, float32.

    ``L_ground = L_B(T_ground) + sky.ground_shine(t_s)``; the second term is exactly 0.0 in an
    emissive band, so an LWIR frame is bit-identical to what it was before AT.20.
    """
    v = np.asarray(sky_view, dtype=np.float64)
    if v.dtype == np.float16:  # pragma: no cover - asarray above widens; kept for clarity
        raise TypeError("sky_view_factor is float16")
    if np.any((v < 0.0) | (v > 1.0)):
        raise ValueError("sky_view_factor must lie in [0, 1]")
    l_sky = sky.effective_radiance_from_sky_view(t_s, v)
    l_ground = float(lut.lookup(np.float64(ground_temperature_k(sky, t_s)), quantity)[()])
    l_ground += sky.ground_shine(t_s)
    return np.asarray(v * l_sky + (1.0 - v) * l_ground, dtype=np.float32)
