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
    generate_cloud_field,
)

__all__ = [
    "DEFAULT_THICKNESS_M",
    "DEFAULT_CELL_M",
    "DEFAULT_MIN_ELEVATION_DEG",
    "DEFAULT_SMALLEST_CLOUD_M",
    "DEFAULT_DEPTH_SCALE_SIGMA",
    "MAX_MARCH_STEPS",
    "MIN_MARCH_STEPS",
    "MARCH_STEP_M",
    "CloudDeck",
    "MarchResult",
    "TRANSECT_TO_RADIAL_SLOPE",
    "vertical_profile",
    "deck_field",
    "generate_cloud_deck",
    "resample_bilinear",
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

#: Lowest elevation the deck's *tile* is sized for. The tile's half-period is ``base / tan(this)``,
#: so the shallowest ray of interest crosses one whole tile before it repeats. The deck itself has
#: no edge: the depth map is periodic (see :meth:`CloudDeck.column_depth`).
DEFAULT_MIN_ELEVATION_DEG = 10.0

#: Field excess, in standard deviations above the condensation threshold, at which a column
#: reaches the capping inversion and stops growing.
#:
#: This is what makes the deck a field of *towers* rather than a field of mesas. A cumulus grows
#: as far above its base as the thermal that made it can carry it, and the strength of a thermal
#: in a field of them is distributed about a mean -- so cloud-top height rises with the synthesised
#: field's excess over the threshold rather than jumping to full thickness the moment the threshold
#: is crossed. The inversion clips the strongest towers, which is why a fair-weather cumulus field
#: has a common top and rounded shoulders below it rather than one flat lid.
#:
#: At 1.5 sigma and the 0.45 coverage the presets author, the mean cloudy column comes out about
#: half the deck's thickness and 12% of cloudy columns reach the inversion -- a 600 m mean depth at
#: a 1.2 km cap, which is fair-weather cumulus. ESTIMATED: the value is chosen for that outcome,
#: not measured. The quantity that is anchored is the *column* optical depth, which is independent
#: of it.
DEFAULT_DEPTH_SCALE_SIGMA = 1.5

#: Diameter of the smallest cumulus the deck carries, metres.
#:
#: A ``1/f^beta`` field is *scale-free*: it has structure at every size down to the grid, so a 25 m
#: cell gets 25 m clouds. A cumulus field does not work that way. Its size distribution has a
#: mode near half a kilometre and falls off sharply below it, because a parcel that small entrains
#: dry air faster than it can condense and evaporates -- there is a smallest cloud, and it is
#: hundreds of metres across, not tens.
#:
#: Leaving the small scales in is not a cosmetic error in an infrared frame, because the two bands
#: read the deck differently: the dome samples the depth map at the point each ray crosses the
#: base, so a 25 m speck is a 25 m speck, while the infrared band *integrates along the ray*, so
#: the same speck smears over the kilometres the ray spends in the deck. That is how a frame ends
#: up covered in bright marks with nothing in the visible image to match them.
#:
#: Applied as a Gaussian roll-off at ``smallest_cloud_m / 4``, which leaves ~2% of the variance
#: below ``smallest_cloud_m``. ESTIMATED: 400 m is the low end of the observed fair-weather
#: cumulus mode, not a measurement of this scene.
DEFAULT_SMALLEST_CLOUD_M = 400.0

#: Cap on the samples per ray, and the floor. A march has to resolve the depth map along the
#: ray, and an oblique ray crosses the map **horizontally**: at 20 degrees a ray climbing 1.2 km
#: travels 3.3 km sideways, so 48 steps put its samples 68 m apart and the undersampling shows up
#: as horizontal banding at every cloud edge -- visible immediately in a rendered frame.
#: :meth:`CloudDeck.adequate_steps` sizes the march from the geometry instead and clamps it here.
MAX_MARCH_STEPS = 512
MIN_MARCH_STEPS = 32

#: Sample spacing along the ray the step count aims for, metres.
#:
#: Chosen by measuring the error a *frame* keeps -- marched, interpolated back up and box-filtered
#: to native, against a converged reference -- rather than the march in isolation. At the shipped
#: settings the 99th percentile of the band emissivity error runs 0.073 at 36 m, 0.039 at 18 m and
#: 0.029 at 12 m; 0.029 is 1.3 K against the 44 K a cloud stands above a clear zenith, and 12 m
#: costs 6.5 s of a frame against 2.2 s at 36 m.
#:
#: That measurement also settled something it was not aimed at. The frame's error is dominated by
#: **quadrature**, not by the interpolation from the marched grid up to the supersampled one:
#: halving the march stride changes the 99th percentile from 0.029 to 0.024 and costs five times
#: as much. So the march runs once per native pixel and the steps are spent here instead.
MARCH_STEP_M = 12.0


def vertical_profile(u: Any) -> NDArray[np.float64]:
    """``w(u) = 6 u (1 - u)`` on ``u`` in [0, 1], zero outside. ``∫₀¹ w du = 1`` exactly.

    So a column of geometric thickness H carrying this profile has ``∫ w(h/H) dh = H``, and a
    vertical ray through it accumulates exactly the column's authored optical depth. See the
    module docstring for why that identity is the point.
    """
    x = np.asarray(u, dtype=np.float64)
    inside = (x >= 0.0) & (x <= 1.0)
    # Clamped before the product, not after: a column with a very small depth puts `u` in the
    # millions, and 6u(1-u) on that overflows to -inf before `where` gets to discard it.
    clamped = np.where(inside, x, 0.0)
    return np.asarray(np.where(inside, 6.0 * clamped * (1.0 - clamped), 0.0), dtype=np.float64)


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
        """Steps that keep the march's sample spacing to :data:`MARCH_STEP_M` along the shallowest
        ray in the array, clamped to [:data:`MIN_MARCH_STEPS`, :data:`MAX_MARCH_STEPS`].

        **Path length is the criterion, not cells per cloud.** The march is a midpoint rule over an
        integrand with a jump in it -- the cloud's own boundary -- so it converges at first order
        and the error is set by how precisely a step lands on that boundary, which is a distance
        in metres and has nothing to do with the grid.

        This is the array-wide answer, which is what a caller needs to size a cost or to pin a
        test. :meth:`march` does better: it gives each ray the count its own path needs, so the
        top of an oblique frame is not marched four times more finely than it can use.

        The **minimum** elevation in the array sets this one, not the mean, for the same reason
        the march is per-ray at all -- the shallow rays are where the aliasing shows, because they
        are the ones that spend longest in the deck.
        """
        el = np.abs(np.asarray(elevation_rad, dtype=np.float64))
        lowest = float(np.min(el[el > 0.0])) if np.any(el > 0.0) else 0.5 * math.pi
        lowest = max(lowest, math.radians(CLOUD_AIRMASS_FLOOR_DEG))
        need = self.thickness_m / math.sin(lowest) / MARCH_STEP_M
        return int(min(MAX_MARCH_STEPS, max(MIN_MARCH_STEPS, math.ceil(need))))

    def column_depth(self, x_m: Any, z_m: Any) -> NDArray[np.float64]:
        """Bilinear sample of the depth map at stage (X, Z). The deck **tiles**: it has no edge.

        The depth map is an inverse FFT of a synthesised spectrum, so it is exactly periodic in
        both axes -- sample ``j`` and sample ``j + n`` are the same number, not merely similar
        ones. Wrapping the lookup is therefore seamless to the bit, and it removes the artefact a
        finite footprint has: a ray shallower than :data:`DEFAULT_MIN_ELEVATION_DEG` used to leave
        the map and read clear sky, which put a hard, straight, cold band across the bottom of
        every oblique frame where a real cumulus field puts its densest wall.

        The tile is ``n * cell_m`` across -- 12 km at the shipped settings, against the 8 km the
        shallowest ray in a 25-degree frame reaches -- so a frame sees at most one repeat.
        """
        n = self.depth.shape[0]
        centre = 0.5 * (n - 1)
        fx = np.mod(np.asarray(x_m, dtype=np.float64) / self.cell_m + centre, n)
        fz = np.mod(np.asarray(z_m, dtype=np.float64) / self.cell_m + centre, n)
        i0 = fx.astype(np.int64) % n
        j0 = fz.astype(np.int64) % n
        tx = fx - i0
        tz = fz - j0
        i1 = (i0 + 1) % n
        j1 = (j0 + 1) % n
        top = self.depth[i0, j0] * (1.0 - tz) + self.depth[i0, j1] * tz
        bottom = self.depth[i1, j0] * (1.0 - tz) + self.depth[i1, j1] * tz
        return np.asarray(top * (1.0 - tx) + bottom * tx)

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
        # A caller pinning the quadrature gets exactly what it asked for; `None` means each ray
        # gets the count its own path needs, computed once the geometry is known.
        pinned = None if steps is None else int(steps)
        if pinned is not None and pinned < 2:
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
        # **Each ray gets the steps its own path needs**, not the shallowest ray's. `MARCH_STEP_M`
        # is a spacing in metres, so the count is the path over it, and the path is four times
        # longer at 8 degrees than at 33 -- one number for the array means the top of an oblique
        # frame is marched four times more finely than it can use. Over a 25 degree field that is
        # a third of the work for nothing. `adequate_steps` remains the array-wide answer, because
        # it is what a caller needs to size a cost; the march does better than it.
        if pinned is not None:
            per_ray = np.full(el.shape, pinned, dtype=np.int64)
        else:
            need = np.ceil((t_out - t_in) / MARCH_STEP_M)
            per_ray = np.clip(need, MIN_MARCH_STEPS, MAX_MARCH_STEPS).astype(np.int64)
        longest = int(per_ray.max(initial=MIN_MARCH_STEPS))
        ds = np.where(rising, (t_out - t_in) / per_ray, 0.0)

        tau = np.zeros(el.shape, dtype=np.float64)
        weighted = np.zeros(el.shape, dtype=np.float64)
        weight = np.zeros(el.shape, dtype=np.float64)
        ratio = float(CLOUD_OD_RATIO)
        # **Only the rays still on their own path.** The loop runs to the longest ray's step
        # count, so a compacted working set is what makes per-ray step counts pay: at 33 degrees
        # a ray is done after 60 steps while the 8 degree ray below it has 250 to go.
        #
        # Deliberately *not* also dropping rays whose optical depth has saturated. It is tempting
        # -- most of a near-horizon ray's path sits behind e^{-12} of transmittance and moves
        # neither the emission height nor the infrared emissivity -- but the visible band reads
        # the same optical depth through a two-stream reflectance that is still climbing at tau =
        # 24 (0.70) and does not approach white until several times that. Truncating there caps
        # how bright a deep cloud can be, in one band only, for a 20 % saving. Measured, then
        # removed.
        live = np.flatnonzero(np.asarray(rising).reshape(-1))
        flat_tin = np.asarray(t_in).reshape(-1)
        flat_ds = np.asarray(ds).reshape(-1)
        flat_steps = np.asarray(per_ray).reshape(-1)
        flat_dir = direction.reshape(-1, 3)
        flat_tau = tau.reshape(-1)
        flat_weighted = weighted.reshape(-1)
        flat_weight = weight.reshape(-1)
        for k in range(longest):
            if live.size == 0:
                break
            t = flat_tin[live] + (k + 0.5) * flat_ds[live]
            step = flat_ds[live]
            px = origin_m[0] + t * flat_dir[live, 0]
            py = oy + t * flat_dir[live, 1]
            pz = origin_m[2] + t * flat_dir[live, 2]
            d_tau = self.density_at(px, py, pz) * step
            # Emission weight in the band that will read it: e^{-tau_band} d(tau_band). Using the
            # authored (0.55 um) depth here would put the emitting level low by the OD ratio.
            running = flat_tau[live]
            w = np.exp(-ratio * running) * (1.0 - np.exp(-ratio * d_tau))
            flat_weighted[live] += w * (py - self.base_m)
            flat_weight[live] += w
            flat_tau[live] = running + d_tau
            live = live[flat_steps[live] > k + 1]
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


def resample_bilinear(values: Any, shape: tuple[int, int]) -> NDArray[np.float64]:
    """Bilinear resize of a 2-D array onto ``shape``, corners aligned.

    Here so that a marched sky can be computed on a coarse grid and put back on a fine one. That
    is worth doing because the two grids exist for different reasons: an infrared camera's AOVs
    are rendered **supersampled** so that *geometry* edges antialias, and at this project's
    factor of four that is sixteen times as many rays. The cloud field behind the geometry has no
    edges at that scale -- its finest feature is a depth-map cell, which subtends about a native
    pixel at the ranges these scenes work at -- so marching every subpixel spends sixteen times
    the work to recover a value a box filter would have averaged back anyway.

    Not used for anything with an edge in it. The aircraft's own pixels never come from here;
    they are masked out before this result is applied.
    """
    v = np.asarray(values, dtype=np.float64)
    if v.ndim != 2:
        raise ValueError(f"resample_bilinear takes a 2-D array, got {v.shape}")
    rows, cols = int(shape[0]), int(shape[1])
    if rows < 1 or cols < 1:
        raise ValueError("the target shape must be positive")
    out = v
    for axis, (have, want) in enumerate(((v.shape[0], rows), (v.shape[1], cols))):
        if have == want:
            continue
        if have == 1:
            out = np.repeat(out, want, axis=axis)
            continue
        pos = np.linspace(0.0, have - 1.0, want)
        i0 = np.floor(pos).astype(np.int64)
        i1 = np.minimum(i0 + 1, have - 1)
        frac = (pos - i0).reshape((-1, 1) if axis == 0 else (1, -1))
        lo = np.take(out, i0, axis=axis)
        hi = np.take(out, i1, axis=axis)
        out = lo * (1.0 - frac) + hi * frac
    return np.asarray(out, dtype=np.float64)


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


def _low_pass(field: NDArray[np.float64], sigma_cells: float) -> NDArray[np.float64]:
    """Circular Gaussian blur at ``sigma_cells``, renormalised to zero mean and unit variance.

    Done in the Fourier domain both because the field was synthesised there and because a
    circular convolution keeps it exactly periodic, which is what lets the deck tile
    (:meth:`CloudDeck.column_depth`). The renormalisation matters: the threshold and the depth
    ramp are both expressed in standard deviations of the field, and a blur removes variance, so
    without it a smoothed field would come out systematically shallower as well as smoother.
    """
    n = field.shape[0]
    fy = np.fft.fftfreq(n)[:, None]
    fx = np.fft.fftfreq(n)[None, :]
    f2 = fx * fx + fy * fy
    kernel = np.exp(-2.0 * (math.pi * sigma_cells) ** 2 * f2)
    out = np.real(np.fft.ifft2(np.fft.fft2(field) * kernel))
    sd = float(out.std())
    return np.asarray((out - out.mean()) / (sd if sd > 0.0 else 1.0), dtype=np.float64)


def deck_field(
    n: int,
    *,
    beta: float,
    cloud_fraction: float,
    seed: int,
    depth_scale: float = DEFAULT_DEPTH_SCALE_SIGMA,
    smooth_cells: float = 0.0,
) -> NDArray[np.float64]:
    """A 0-to-1 column-depth map on the deck plane: 1/f^beta structure, thresholded at ``1 - c``
    and then **grown with height** rather than switched on.

    Built **in metres on the deck**, not by projecting the hemispherical field down onto it. That
    distinction is the whole geometry of a cumulus field and getting it wrong is visible
    immediately: the angular field's features subtend a fixed number of degrees, so projecting
    them onto a deck makes columns a few hundred metres wide and then extrudes each one 1.2 km
    straight up. The result is a field of tall thin fins, and an oblique ray runs *along* one for
    kilometres instead of crossing a cloud -- which renders as vertical streaks, which is what the
    first version of this module produced and what the rendered preview showed.

    Synthesised here, a cloud is about as wide as it is tall, which is what a fair-weather cumulus
    is, and an oblique ray enters one side and leaves the other.

    **The height is a ramp, not a switch**, and that is the load-bearing part. A threshold with a
    soft edge -- which is what a plane-parallel *membership* map wants, and what this function
    used to return -- makes every covered column the deck's full thickness, so the deck is a field
    of flat-topped mesas with vertical walls. Straight up that is invisible, because a vertical ray
    through a mesa and a vertical ray through a tower of the same column depth read the same. At 20
    degrees it is the whole picture: an oblique ray runs *along* a 1.2 km wall for kilometres, so
    the marched optical depth smears into vertical streaks and saturates -- tau averaged 7 over a
    rendered frame, which is opaque everywhere, which is the flat white the analytic fix of AT.11
    was meant to remove. Growing the top with the field's excess instead gives a tower rounded
    shoulders, and a ray crosses it rather than grazing a wall.

    ``depth = smoothstep(min(1, (field - threshold) / depth_scale))``: zero at the threshold,
    rising with the excess, clipped by the capping inversion at :data:`DEFAULT_DEPTH_SCALE_SIGMA`
    sigma above it. The smoothstep rounds both ends -- no cusp where a tower meets the inversion
    and no crease where it meets the clear air.

    ``beta`` is the **radial** exponent this function applies; see
    :data:`TRANSECT_TO_RADIAL_SLOPE` for why a caller holding a published transect slope adds one
    to it first. The threshold is taken at the ``1 - cloud_fraction`` quantile exactly as the
    hemispherical field's is, and because the ramp is zero exactly at the threshold the authored
    coverage comes out *exact* -- the fraction of columns with any cloud in them is ``c`` to within
    a cell, where the old smoothstep spread cloud half its softness past the threshold.
    """
    if n < 8 or n % 2 == 0:
        raise ValueError("the deck grid must be odd and at least 8 cells across")
    if depth_scale <= 0.0:
        raise ValueError("depth_scale must be positive: it is an excess in sigma, not a flag")
    field = generate_cloud_field((n, n), beta, cloud_fraction, seed).field
    if cloud_fraction <= 0.0:
        return np.zeros((n, n), dtype=np.float64)
    if smooth_cells > 0.0:
        field = _low_pass(field, smooth_cells)
    threshold = float(np.quantile(field, 1.0 - cloud_fraction))
    x = np.clip((field - threshold) / depth_scale, 0.0, 1.0)
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
    smallest_cloud_m: float = DEFAULT_SMALLEST_CLOUD_M,
) -> CloudDeck:
    """A cumulus deck over the camera: seeded, reproducible, and the one cloud both bands read.

    The tile runs out to ``base / tan(min_elevation_deg)``, so the shallowest ray a driver aims at
    crosses one whole period before the map repeats. The deck has no edge -- see
    :meth:`CloudDeck.column_depth`.
    """
    if base_m <= 0.0:
        raise ValueError("a deck needs a base above the camera; a base at the surface is fog")
    half = base_m / math.tan(math.radians(float(min_elevation_deg)))
    n = int(round(2.0 * half / cell_m))
    n += 1 if n % 2 == 0 else 2  # odd, so one cell sits exactly overhead
    return CloudDeck(
        depth=deck_field(
            n,
            beta=beta,
            cloud_fraction=cloud_fraction,
            seed=seed,
            smooth_cells=max(0.0, float(smallest_cloud_m)) / 4.0 / float(cell_m),
        ),
        cell_m=float(cell_m),
        base_m=float(base_m),
        thickness_m=float(thickness_m),
        optical_depth=float(optical_depth),
    )
