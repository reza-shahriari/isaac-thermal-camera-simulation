"""Per-cell sky view factor from a 145-patch sky, through the occluders that cast the beam (PT.21).

docs/physics-model.md §5.4 (the diffuse term), §6.1 (where V_s scales both diffuse solar and
the longwave down); ADR 0095 (occluders in the scene), ADR 0104.

PT.18 gave each cell its own direct beam. The other half of a shadow is the *sky* a cell cannot
see: under an overhang, at the foot of a wall, in a passage, a cell receives less diffuse sun
and less longwave from the sky and more from the warm surfaces around it -- and that, not the
beam, is what keeps a shaded facade's diurnal swing small (a hot-dry-climate study measured
walls at SVF ≈ 0.2 swinging 1.97 °C where open walls swung 4.21 °C). The per-prim forcing uses
the tilt's isotropic factor ``V_s = (1 + cos β)/2`` for every cell; this module replaces it per
cell with the fraction of the dome the cell actually sees.

**The dome.** Tregenza's 145-patch subdivision (8 altitude bands of 30, 30, 24, 24, 18, 12, 6
patches plus the zenith cap), each patch a direction and a solid angle. A cell's factor is the
cosine-weighted, visibility-gated sum over the patches above its plane,

    SVF = V_s · Σ_i ω_i cos θ_i vis_i / Σ_i ω_i cos θ_i,

**normalised** to the unobstructed quadrature of the same dome: an open cell keeps *exactly*
the ``V_s`` the per-prim balance used (so PT.17's identity holds bit for bit), and an obstructed
one keeps the fraction the dome says. ``vis_i`` is :func:`~irsim.thermal.shadow.cell_shadow`
-- the same exact ray-rectangle test the beam uses, so a shadow and a sky view cannot disagree
about where the occluders are -- and it is a *fraction*: each patch is sub-sampled on a
``SUB_ALTITUDES × SUB_AZIMUTHS`` grid of its own extent (12 rays, each with its own exact solid
angle), because a patch that straddles a wall's edge is neither seen nor hidden, and a one-ray
patch put the foot of a wall at 0.536 where the analytic answer is 0.5. 1740 rays per cell,
computed once: the occluders do not move.

**What the factor scales.** Both the diffuse solar (``V_s · DHI``) and the longwave down
(`longwave_down`'s sky share, the rest of the hemisphere seen at the air temperature) -- both,
because a factor on solar alone leaves the night unchanged and the diurnal amplitude wrong. The
Perez split of the diffuse into dome, circumsolar and horizon bands is **not** implemented:
the dome is isotropic here, and ADR 0104 records the deferral.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from irsim.thermal.shadow import ShadowRectangle, cell_shadow
from irsim.thermal.surface_field import PlanarPatch

__all__ = ["TREGENZA_BANDS", "sky_view_factors", "sub_rays", "tregenza_patches"]

#: (centre altitude in degrees, number of patches) per band, plus the zenith cap: 145 patches.
TREGENZA_BANDS: tuple[tuple[float, int], ...] = (
    (6.0, 30),
    (18.0, 30),
    (30.0, 24),
    (42.0, 24),
    (54.0, 18),
    (66.0, 12),
    (78.0, 6),
    (90.0, 1),
)


#: Sub-rays per patch, in altitude and azimuth. The azimuth offsets (2m+1)/8 of a patch never
#: land on a cardinal direction for any band count, so a wall along N-S or E-W never sees a
#: ray exactly in its own plane (which `cell_shadow` must let pass).
SUB_ALTITUDES = 3
SUB_AZIMUTHS = 4
_BAND_HALF_DEG = 6.0


def tregenza_patches() -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """``(directions (145, 3) in ENU, solid angles (145,))``; the solid angles sum to 2π."""
    directions: list[NDArray[np.float64]] = []
    omegas: list[float] = []
    for altitude_deg, count in TREGENZA_BANDS:
        lo, hi = _band_edges_rad(altitude_deg, count)
        band_omega = 2.0 * np.pi * (np.sin(hi) - np.sin(lo)) / count
        alt = np.deg2rad(altitude_deg) if count > 1 else 0.5 * (lo + hi)
        for k in range(count):
            azimuth = 2.0 * np.pi * (k + 0.5) / count  # from north, clockwise (ENU convention)
            directions.append(_direction(alt, azimuth) if count > 1 else np.array([0.0, 0.0, 1.0]))
            omegas.append(float(band_omega))
    return np.asarray(directions), np.asarray(omegas)


def sub_rays() -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int64]]:
    """The quadrature actually traced: ``(directions (1740, 3), solid angles, patch index)``.

    Every Tregenza patch is cut into ``SUB_ALTITUDES × SUB_AZIMUTHS`` cells of its own
    altitude-azimuth extent, each with its exact solid angle; the solid angles still sum to 2π
    and sum per patch to the patch's own.
    """
    directions: list[NDArray[np.float64]] = []
    omegas: list[float] = []
    index: list[int] = []
    patch = 0
    for altitude_deg, count in TREGENZA_BANDS:
        lo, hi = _band_edges_rad(altitude_deg, count)
        alt_edges = np.linspace(lo, hi, SUB_ALTITUDES + 1)
        for k in range(count):
            for j in range(SUB_ALTITUDES):
                a_lo, a_hi = alt_edges[j], alt_edges[j + 1]
                omega = 2.0 * np.pi * (np.sin(a_hi) - np.sin(a_lo)) / (count * SUB_AZIMUTHS)
                for m in range(SUB_AZIMUTHS):
                    azimuth = 2.0 * np.pi * (k + (2 * m + 1) / (2 * SUB_AZIMUTHS)) / count
                    directions.append(_direction(0.5 * (a_lo + a_hi), azimuth))
                    omegas.append(float(omega))
                    index.append(patch)
            patch += 1
    return np.asarray(directions), np.asarray(omegas), np.asarray(index)


def _band_edges_rad(altitude_deg: float, count: int) -> tuple[float, float]:
    if count == 1:  # the zenith cap, 84°-90°
        return float(np.deg2rad(90.0 - _BAND_HALF_DEG)), float(np.deg2rad(90.0))
    return (
        float(np.deg2rad(altitude_deg - _BAND_HALF_DEG)),
        float(np.deg2rad(altitude_deg + _BAND_HALF_DEG)),
    )


def _direction(altitude_rad: float, azimuth_rad: float) -> NDArray[np.float64]:
    return np.array(
        [
            np.cos(altitude_rad) * np.sin(azimuth_rad),
            np.cos(altitude_rad) * np.cos(azimuth_rad),
            np.sin(altitude_rad),
        ]
    )


def sky_view_factors(
    patch: PlanarPatch,
    normal_enu: Any,
    tilt_view_factor: float,
    occluders: Sequence[ShadowRectangle],
    to_world: Any,
) -> NDArray[np.float64]:
    """``(n_cells,)`` sky view factors: ``V_s`` for an open cell, less where the dome is hidden.

    ``normal_enu`` is the surface's outward normal (the balance's authority on where it faces);
    ``to_world`` maps an ENU direction into the patch's frame for the occluder test.
    """
    n = np.asarray(normal_enu, dtype=np.float64).reshape(3)
    n = n / float(np.linalg.norm(n))
    directions, omegas, _ = sub_rays()
    cosines = directions @ n
    above = cosines > 0.0
    weights = omegas[above] * cosines[above]
    if not np.any(above):  # a downward-facing surface sees no sky at all
        return np.zeros(patch.n_cells)
    if not occluders:
        return np.full(patch.n_cells, float(tilt_view_factor))
    # The open quadrature is accumulated in the same order as the gated one, so an unobstructed
    # cell's ratio is exactly 1 and its factor exactly V_s (PT.17's bit identity survives).
    seen = np.zeros(patch.n_cells)
    total = 0.0
    for direction, weight in zip(directions[above], weights, strict=True):
        seen += weight * cell_shadow(patch, to_world(direction), occluders)
        total += weight
    return np.asarray(float(tilt_view_factor) * (seen / total))
