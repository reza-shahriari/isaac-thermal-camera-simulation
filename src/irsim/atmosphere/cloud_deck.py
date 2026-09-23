"""A cloud with a third dimension: one deck, marched for the infrared and voxelised for the
visible band (AT.12, ADR 0127).

AT.11 gave a cloud an optical depth and a distance and it is still, geometrically, a
**plane-parallel sheet**: every ray that enters it stays in it, so an optically thick core reads
one level to within a kelvin. Looking straight up that is right -- a cumulus base really is flat,
which is what the lifting condensation level *means*. Looking at 20 degrees it is not, because a
real cumulus field is a field of towers and the camera sees their flanks, their tops and the gaps
between them.

This module gives the deck a top. A column of the sky-fixed field has depth ``d`` in 0..1; that
column's cloud reaches ``thickness_m * d`` above the base, and its liquid water is distributed
through that height by an analytic profile. A ray then **crosses columns**: at 20 degrees
elevation a ray climbing 1200 m travels 3.3 km horizontally, so it samples a long line of towers
and gaps, and where it stops emitting depends on which of them it met. That is where the several
kelvin of structure in an oblique frame comes from, and no plane-parallel model has it.

**One object, two bands.** The same deck is ray-marched by the infrared background
(:meth:`CloudDeck.march`) and voxelised into a NanoVDB volume for the visible one
(:meth:`CloudDeck.voxels`, written by :mod:`irsim_isaac.cloud_volume`). They are not two models
that resemble each other -- the visible volume is the field the infrared band integrates, sampled
on a grid. ADR 0076's rule that the two bands cannot disagree about where a cloud is therefore
survives the move from a painted dome to real geometry.

**The vertical profile is a normalised parabola**, ``w(u) = 6 u (1 - u)`` on ``u = h / H``. Two
reasons, and the second is the load-bearing one:

* it is zero at the base and at the top and smooth in between, so a rendered volume has no slab
  edges and a marched ray has no step;
* its integral over the column is **exactly H**, so a *vertical* ray through a column of depth
  ``d`` accumulates exactly ``optical_depth * d`` -- which is AT.11's model, to the bit. The
  third dimension is therefore provably a generalisation and not a different cloud: looking up,
  nothing moved.

Real cumulus liquid water content rises with height above the base along a near-adiabat and falls
off at the top, so a symmetric parabola is a **stand-in**, not a microphysical profile. It is
labelled ESTIMATED wherever it is authored, and the one quantity that is anchored -- the column
optical depth -- is independent of the shape by construction.

docs/physics-model.md has no cloud section (§5.3(a) only says T_sky -> T_air under overcast);
ADR 0126, ADR 0127.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from irsim.atmosphere.cloud import (
    CLOUD_AIRMASS_FLOOR_DEG,
    CLOUD_OD_RATIO,
    DEFAULT_EDGE_SOFTNESS,
    generate_cloud_field,
)

__all__ = [
    "DEFAULT_THICKNESS_M",
    "DEFAULT_CELL_M",
    "DEFAULT_MIN_ELEVATION_DEG",
    "MAX_MARCH_STEPS",
    "MIN_MARCH_STEPS",
    "MARCH_SAMPLES_PER_CELL",
    "CloudDeck",
    "MarchResult",
    "TRANSECT_TO_RADIAL_SLOPE",
    "vertical_profile",
    "deck_field",
    "generate_cloud_deck",
]

#: Geometric thickness of a column at full depth, metres. ESTIMATED: continental fair-weather
#: cumulus run roughly 0.5-2 km from base to top, and a scene's own optical depth is authored
#: separately, so this sets the *aspect* of a tower rather than how opaque it is.
DEFAULT_THICKNESS_M = 1200.0

#: Horizontal cell of the depth map, metres. At a 1 km base and 20 degrees of elevation a cell
#: this size subtends ~0.3 degrees, which is under the angular cell of the field it is built from
#: (the drivers ask for six cells per degree), so the map does not invent structure the field
#: does not have and does not throw away structure it does.
DEFAULT_CELL_M = 25.0

#: Lowest elevation the deck is built out to. The horizontal extent is ``base / tan(this)``, which
#: diverges at the horizon; below it a ray leaves the deck's footprint and reads clear sky.
DEFAULT_MIN_ELEVATION_DEG = 10.0

#: Cap on the samples per ray, and the floor. A march has to resolve the depth map along the
#: ray, and an oblique ray crosses the map **horizontally**: at 20 degrees a ray climbing 1.2 km
#: travels 3.3 km sideways, so 48 steps put its samples 68 m apart across a 25 m grid and the
#: undersampling shows up as horizontal banding at every cloud edge -- visible immediately in a
#: rendered frame. :meth:`CloudDeck.adequate_steps` sizes the march from the geometry instead and
#: clamps it here. 256 is where the banding stopped being visible in the preview frames, at
#: 4.2 s for a 640x512 frame against 0.6 s at 48.
MAX_MARCH_STEPS = 256
MIN_MARCH_STEPS = 32

#: Horizontal samples per depth-map cell the step count aims for.
MARCH_SAMPLES_PER_CELL = 2.0


def vertical_profile(u: Any) -> NDArray[np.float64]:
    """``w(u) = 6 u (1 - u)`` on ``u`` in [0, 1], zero outside. ``∫₀¹ w du = 1`` exactly.

    So a column of geometric thickness H carrying this profile has ``∫ w(h/H) dh = H``, and a
    vertical ray through it accumulates exactly the column's authored optical depth. See the
    module docstring for why that identity is the point.
    """
    x = np.asarray(u, dtype=np.float64)
    inside = (x >= 0.0) & (x <= 1.0)
    return np.asarray(np.where(inside, 6.0 * x * (1.0 - x), 0.0), dtype=np.float64)


@dataclass(frozen=True)
class MarchResult:
    """What a ray took from the deck: how much cloud, and where in it the emission came from."""

    #: Visible optical depth accumulated along the ray (not the LWIR one; see `emissivity`).
    optical_depth: NDArray[np.float64]
    #: Height above the deck's base, in metres, weighted by ``e^{-τ} dτ`` -- the level a LWIR
    #: photon leaving along this ray most likely came from. Zero where the ray met no cloud.
    emission_height_m: NDArray[np.float64]

    def emissivity(self, od_ratio: float = CLOUD_OD_RATIO) -> NDArray[np.float64]:
        """``1 - exp(-od_ratio * τ_vis)`` -- the same Beer-Lambert law AT.11 applies, except that
        the optical depth was integrated along the real path rather than scaled by an airmass."""
        return np.asarray(1.0 - np.exp(-float(od_ratio) * self.optical_depth), dtype=np.float64)


@dataclass(frozen=True)
class CloudDeck:
    """A slab of cumulus at the LCL: a horizontal map of column depth, plus an analytic profile.

    ``depth`` is the sky-fixed field's 0-to-1 depth, resampled onto a square horizontal grid at
    the base altitude. Cell ``(i, j)`` covers stage X and Z, in that order, and the grid is
    centred on the camera's own column, so index ``n // 2`` is overhead.
    """

    depth: NDArray[np.float64]
    cell_m: float
    base_m: float
    thickness_m: float
    optical_depth: float

    def __post_init__(self) -> None:
        if self.depth.ndim != 2 or self.depth.shape[0] != self.depth.shape[1]:
            raise ValueError(f"the depth map must be a square grid, got {self.depth.shape}")
        if self.depth.shape[0] % 2 == 0:
            raise ValueError("the depth map needs an odd size so one cell sits overhead")
        if min(self.cell_m, self.thickness_m, self.optical_depth) <= 0.0 or self.base_m < 0.0:
            raise ValueError("cell, thickness and optical depth must be positive")

    @property
    def half_extent_m(self) -> float:
        return float(0.5 * (self.depth.shape[0] - 1) * self.cell_m)

    @property
    def extinction_per_m(self) -> float:
        """Visible volume extinction of cloud at the profile's mean, per metre.

        ``optical_depth / thickness``: constant, because a deeper column is deeper rather than
        denser, which is what makes the vertical column depth come out as ``optical_depth * d``.
        """
        return self.optical_depth / self.thickness_m

    @property
    def top_m(self) -> float:
        return self.base_m + self.thickness_m

    def adequate_steps(self, elevation_rad: Any) -> int:
        """Samples per ray that resolve the depth map along the shallowest ray in the array.

        A ray crossing the deck at elevation theta travels ``thickness / tan(theta)`` horizontally
        and ``thickness`` vertically; the march has to sample whichever is longer at
        :data:`MARCH_SAMPLES_PER_CELL` per cell, clamped to
        [:data:`MIN_MARCH_STEPS`, :data:`MAX_MARCH_STEPS`].

        The **minimum** elevation in the array sets it, not the mean: the step count is one number
        for the whole march, and sizing it on a typical ray leaves the shallow ones aliased --
        which is where the banding is worst, because those are the rays that cross the most cells.
        """
        el = np.abs(np.asarray(elevation_rad, dtype=np.float64))
        lowest = float(np.min(el[el > 0.0])) if np.any(el > 0.0) else 0.5 * math.pi
        lowest = max(lowest, math.radians(CLOUD_AIRMASS_FLOOR_DEG))
        horizontal = self.thickness_m / math.tan(lowest)
        need = MARCH_SAMPLES_PER_CELL * max(horizontal, self.thickness_m) / self.cell_m
        return int(min(MAX_MARCH_STEPS, max(MIN_MARCH_STEPS, math.ceil(need))))

    def column_depth(self, x_m: Any, z_m: Any) -> NDArray[np.float64]:
        """Bilinear sample of the depth map at stage (X, Z); zero outside the deck's footprint."""
        n = self.depth.shape[0]
        centre = 0.5 * (n - 1)
        fx = np.asarray(x_m, dtype=np.float64) / self.cell_m + centre
        fz = np.asarray(z_m, dtype=np.float64) / self.cell_m + centre
        inside = (fx >= 0.0) & (fx <= n - 1) & (fz >= 0.0) & (fz <= n - 1)
        cx = np.clip(fx, 0.0, n - 1.001)
        cz = np.clip(fz, 0.0, n - 1.001)
        i0 = cx.astype(np.int64)
        j0 = cz.astype(np.int64)
        tx = cx - i0
        tz = cz - j0
        i1 = np.minimum(i0 + 1, n - 1)
        j1 = np.minimum(j0 + 1, n - 1)
        top = self.depth[i0, j0] * (1.0 - tz) + self.depth[i0, j1] * tz
        bottom = self.depth[i1, j0] * (1.0 - tz) + self.depth[i1, j1] * tz
        return np.asarray(np.where(inside, top * (1.0 - tx) + bottom * tx, 0.0))

    def density_at(self, x_m: Any, y_m: Any, z_m: Any) -> NDArray[np.float64]:
        """Visible extinction per metre at a stage point. The one definition of the cloud's shape.

        Both consumers go through here -- the ray march samples it along a path and the voxeliser
        evaluates it on a grid -- so the volume the path tracer renders and the field the infrared
        band integrates cannot be different clouds.
        """
        d = self.column_depth(x_m, z_m)
        height = np.asarray(y_m, dtype=np.float64) - self.base_m
        with np.errstate(divide="ignore", invalid="ignore"):
            u = np.where(d > 0.0, height / np.where(d > 0.0, d * self.thickness_m, 1.0), -1.0)
        return np.asarray(self.extinction_per_m * vertical_profile(u), dtype=np.float64)

    # -- the infrared path ------------------------------------------------------------------
    def march(
        self,
        elevation_rad: Any,
        azimuth_rad: Any,
        *,
        origin_m: tuple[float, float, float] = (0.0, 0.0, 0.0),
        steps: int | None = None,
        up: Any = (0.0, 1.0, 0.0),
        forward: Any = (0.0, 0.0, -1.0),
    ) -> MarchResult:
        """Integrate the deck along each ray: how much cloud, and the level it emits from.

        The emission height is the ``e^{-τ} dτ`` weighted mean of the height above base -- the
        level a photon leaving along this ray came from, which is near the base for a ray that
        enters an opaque tower from below and high up for one that grazes a top. The caller turns
        it into a temperature with the environmental lapse rate and into a radiance with the band
        LUT, because this module knows no radiometry.

        Using one mean height rather than integrating Planck through the column is a
        **linearisation**: over the 8 K a 1.2 km deck spans it is worth well under 0.1 K, and
        `tests/unit/test_cloud_deck.py` bounds it against a per-step Planck oracle rather than
        asserting it.

        Rays that never reach the deck (pointing down, or outside the footprint) come back with
        zero optical depth, which the radiance blend turns into clear sky to the bit.
        """
        el = np.asarray(elevation_rad, dtype=np.float64)
        az = np.asarray(azimuth_rad, dtype=np.float64)
        if el.shape != az.shape:
            raise ValueError(f"elevation {el.shape} and azimuth {az.shape} must match")
        # `None` sizes the march from the geometry; see `adequate_steps`. A caller passing a
        # number is usually a test pinning the quadrature, and gets exactly what it asked for.
        steps = self.adequate_steps(el) if steps is None else int(steps)
        if steps < 2:
            raise ValueError("a march needs at least two steps")
        direction = _direction(el, az, up, forward)
        oy = float(origin_m[1])
        sin_el = direction[..., 1]
        # A ray must be going up to reach a deck above it; a horizontal one never arrives.
        rising = sin_el > 1e-6
        safe = np.where(rising, sin_el, 1.0)
        t_in = np.where(rising, (self.base_m - oy) / safe, 0.0)
        t_out = np.where(rising, (self.top_m - oy) / safe, 0.0)
        t_in = np.maximum(t_in, 0.0)
        ds = np.where(rising, (t_out - t_in) / steps, 0.0)

        tau = np.zeros(el.shape, dtype=np.float64)
        weighted = np.zeros(el.shape, dtype=np.float64)
        weight = np.zeros(el.shape, dtype=np.float64)
        ratio = float(CLOUD_OD_RATIO)
        for k in range(steps):
            t = t_in + (k + 0.5) * ds
            px = origin_m[0] + t * direction[..., 0]
            py = oy + t * direction[..., 1]
            pz = origin_m[2] + t * direction[..., 2]
            d_tau = self.density_at(px, py, pz) * ds
            # Emission weight in the band that will read it: e^{-tau_band} d(tau_band). Using the
            # authored (0.55 um) depth here would put the emitting level low by the OD ratio.
            transmitted = np.exp(-ratio * tau)
            w = transmitted * (1.0 - np.exp(-ratio * d_tau))
            weighted += w * (py - self.base_m)
            weight += w
            tau += d_tau
        height = np.where(weight > 0.0, weighted / np.where(weight > 0.0, weight, 1.0), 0.0)
        return MarchResult(optical_depth=tau, emission_height_m=height)

    # -- the visible path -------------------------------------------------------------------
    def voxels(
        self, voxel_m: float = DEFAULT_CELL_M
    ) -> tuple[NDArray[np.float32], tuple[float, float, float]]:
        """``(density, min_world)``: the deck on a dense (X, Y, Z) grid, visible extinction per
        metre, float32, ready for :mod:`irsim_isaac.cloud_volume` to write as NanoVDB.

        Evaluated through :meth:`density_at`, so it is a *sampling* of the same function the
        infrared march integrates and not a second authoring of the cloud.
        """
        if voxel_m <= 0.0:
            raise ValueError("voxel size must be positive")
        half = self.half_extent_m
        n_h = int(round(2.0 * half / voxel_m)) + 1
        n_v = int(round(self.thickness_m / voxel_m)) + 1
        axis = np.linspace(-half, half, n_h)
        height = self.base_m + np.linspace(0.0, self.thickness_m, n_v)
        x = axis[:, None, None]
        y = height[None, :, None]
        z = axis[None, None, :]
        grid = self.density_at(np.broadcast_to(x, (n_h, n_v, n_h)), y, z)
        return np.ascontiguousarray(grid, dtype=np.float32), (-half, float(height[0]), -half)


def _direction(el: Any, az: Any, up: Any, forward: Any) -> NDArray[np.float64]:
    """The inverse of :func:`~irsim.atmosphere.cloud.sky_angles`, in the same convention.

    Written as the inverse rather than restated, because the field is sampled through
    `sky_angles` when the deck is built and marched through this when it is read; if the two
    disagreed by a sign the cloud would be in a different place in each band, plausibly.
    """
    u = np.asarray(up, dtype=np.float64).reshape(3)
    u = u / np.linalg.norm(u)
    f = np.asarray(forward, dtype=np.float64).reshape(3)
    f = f - np.dot(f, u) * u
    f = f / np.linalg.norm(f)
    right = np.cross(f, u)
    e = np.asarray(el, dtype=np.float64)
    a = np.asarray(az, dtype=np.float64)
    return np.asarray(
        np.cos(e)[..., None] * (np.cos(a)[..., None] * f + np.sin(a)[..., None] * right)
        + np.sin(e)[..., None] * u
    )


#: A 1-D transect of an isotropic 2-D field whose *radial* power spectrum goes as f^-b has a
#: spectral slope of b - 1, because the transect integrates the 2-D spectrum over the other axis.
#: Published cloud-field slopes -- the classic -5/3 of satellite radiance and lidar transects --
#: are **transect** slopes, so they correspond to a radial exponent of 1 + 5/3 = 2.67.
#:
#: This matters because `CloudSpec.beta` is authored at 1.8 and
#: :func:`~irsim.atmosphere.cloud.generate_cloud_field` applies it as the *radial* exponent. On a
#: 2-D field the variance per octave goes as f^(2-b), so b = 1.8 puts **more** variance at small
#: scales than at large ones: measured on the deck's own grid, the autocorrelation length of a
#: b = 1.8 field is 125 m, which is not a cumulus, it is texture. At b = 2.8 it is 1250 m, which
#: is. The deck therefore synthesises at ``beta + 1``, reading the authored number as the transect
#: slope the literature quotes.
#:
#: The hemispherical :class:`~irsim.atmosphere.cloud.SkyFixedCloud` still applies `beta` radially
#: and is **not** changed here -- that would move the cloud in every scene shipped since MS.3, and
#: it is a question about the preset rather than about this module. Recorded in
#: `docs/spec-issues.md` and ADR 0127 instead.
TRANSECT_TO_RADIAL_SLOPE = 1.0


def deck_field(
    n: int,
    *,
    beta: float,
    cloud_fraction: float,
    seed: int,
    softness: float = DEFAULT_EDGE_SOFTNESS,
) -> NDArray[np.float64]:
    """A 0-to-1 column-depth map on the deck plane: 1/f^beta structure, thresholded at ``1 - c``.

    Built **in metres on the deck**, not by projecting the hemispherical field down onto it. That
    distinction is the whole geometry of a cumulus field and getting it wrong is visible
    immediately: the angular field's features subtend a fixed number of degrees, so projecting
    them onto a deck makes columns a few hundred metres wide and then extrudes each one 1.2 km
    straight up. The result is a field of tall thin fins, and an oblique ray runs *along* one for
    kilometres instead of crossing a cloud -- which renders as vertical streaks, which is what the
    first version of this module produced and what the rendered preview showed.

    Synthesised here, a cloud is about as wide as it is tall, which is what a fair-weather cumulus
    is, and an oblique ray enters one side and leaves the other.

    ``beta`` is the **radial** exponent this function applies; see
    :data:`TRANSECT_TO_RADIAL_SLOPE` for why a caller holding a published transect slope adds one
    to it first. The threshold is taken at the ``1 - cloud_fraction`` quantile exactly as the
    hemispherical field's is, so the authored coverage still means what it did -- over the deck,
    which is what a camera near the zenith sees and slightly less than what an oblique one does,
    since a tilted view foreshortens the gaps between towers.
    """
    if n < 8 or n % 2 == 0:
        raise ValueError("the deck grid must be odd and at least 8 cells across")
    field = generate_cloud_field((n, n), beta, cloud_fraction, seed).field
    if cloud_fraction <= 0.0:
        return np.zeros((n, n), dtype=np.float64)
    threshold = float(np.quantile(field, 1.0 - cloud_fraction))
    if softness <= 0.0:
        return np.asarray(field >= threshold, dtype=np.float64)
    # The same smoothstep on the field's own excess in sigma that `SkyFixedCloud.density` uses
    # (ADR 0125), so a cloud thins toward its edge here for the same reason and by the same law.
    x = np.clip((field - threshold) / softness + 0.5, 0.0, 1.0)
    return np.asarray(x * x * (3.0 - 2.0 * x), dtype=np.float64)


def generate_cloud_deck(
    *,
    beta: float,
    cloud_fraction: float,
    seed: int,
    base_m: float,
    optical_depth: float,
    thickness_m: float = DEFAULT_THICKNESS_M,
    cell_m: float = DEFAULT_CELL_M,
    min_elevation_deg: float = DEFAULT_MIN_ELEVATION_DEG,
) -> CloudDeck:
    """A cumulus deck over the camera: seeded, reproducible, and the one cloud both bands read.

    The footprint runs out to ``base / tan(min_elevation_deg)``, because that expression diverges
    at the horizon. A ray below it leaves the deck and reads clear sky -- wrong in the same
    direction as the horizon itself, which is why the drivers keep the horizon out of frame.
    """
    if base_m <= 0.0:
        raise ValueError("a deck needs a base above the camera; a base at the surface is fog")
    half = base_m / math.tan(math.radians(float(min_elevation_deg)))
    n = int(round(2.0 * half / cell_m))
    n += 1 if n % 2 == 0 else 2  # odd, so one cell sits exactly overhead
    return CloudDeck(
        depth=deck_field(n, beta=beta, cloud_fraction=cloud_fraction, seed=seed),
        cell_m=float(cell_m),
        base_m=float(base_m),
        thickness_m=float(thickness_m),
        optical_depth=float(optical_depth),
    )
