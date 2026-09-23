"""Cloud clutter: LCL base from the shared weather, ε = 1 − τ, 1/f^β structure (MS.3, ADR 0070).

No cloud section exists in the spec; §5.3(a) only says T_sky → T_air under overcast. This model:

* **Base height** from the lifting condensation level of the same WeatherSeries the atmosphere
  and the thermal solver read: z_LCL ≈ 125 m K⁻¹ (T_air − T_dew) (Espy's rule; Lawrence 2005,
  BAMS 86:225, accurate to ~2 % for RH > 50 %), clamped to the preset's bounds; saturated air
  (RH = 1) puts the base at the surface, so a thick overcast reads T_air (§5.3 a).
* **Base temperature** by the environmental lapse rate of the atmosphere preset,
  T_base = T_air − Γ z_LCL (the cloud is in equilibrium with its surroundings).
* **Radiance** of a cloud pixel: ε_cloud L_B(T_base) + τ_cloud L_clear(θ), with τ_cloud authored
  in the environment preset and ε_cloud = 1 − τ_cloud derived (thick → ε = 1; the clear column
  emission stays behind thin cloud). Per band through the LUT.
* **Spatial structure** from a seeded Gaussian 1/f^β field (power spectral density ∝ f^{−β}) on
  the image plane, thresholded at the (1 − c) quantile so exactly the weather's cloud fraction c
  is covered. The generator is a fixture for clutter statistics, not a cloud simulation; its PSD
  slope is verified by a self-test and the ME.5 display-domain bands are the only physical bound
  (deferred until the evaluation lane lands them).

docs/physics-model.md §5.3(a); ADR 0070
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from irsim.atmosphere.humidity import dew_point_k

__all__ = [
    "ESPY_M_PER_K",
    "DEFAULT_EDGE_SOFTNESS",
    "SkyFixedCloud",
    "generate_sky_cloud",
    "sky_angles",
    "lifting_condensation_level_m",
    "cloud_base_temperature_k",
    "CloudField",
    "generate_cloud_field",
    "psd_slope",
    "cloud_radiance",
]

ESPY_M_PER_K = 125.0  # z_LCL / (T - T_dew), Espy's rule (Lawrence 2005)

#: Default width of a cloud's edge ramp, in standard deviations of the 1/f^beta field
#: (:meth:`SkyFixedCloud.density`). ESTIMATED, and chosen by looking: at 0.45 coverage this puts
#: roughly a sixth of the sky in the fringe between clear and full depth, which is about what a
#: fair-weather cumulus field looks like. It is a **fixture for clutter statistics, not cloud
#: microphysics**, and nothing radiometric is claimed for the particular value -- but zero, which
#: is what this was before, is the one value that is definitely wrong: it makes every cloud edge
#: a step discontinuity at the sampling resolution, in both bands.
DEFAULT_EDGE_SOFTNESS = 0.45


def lifting_condensation_level_m(
    t_air_k: float, rh_fraction: float, min_m: float = 0.0, max_m: float = 8000.0
) -> float:
    """z_LCL = 125 (T − T_d) m, clamped to [min_m, max_m]; RH = 1 gives 0 (base at the surface)."""
    if not 0.0 < rh_fraction <= 1.0:
        raise ValueError("RH must be a fraction in (0, 1]")
    z = ESPY_M_PER_K * (t_air_k - dew_point_k(t_air_k, rh_fraction))
    return float(min(max(z, min_m), max_m))


def cloud_base_temperature_k(t_air_k: float, base_m: float, lapse_rate_k_per_m: float) -> float:
    """T_base = T_air − Γ_env z_base."""
    if base_m < 0.0 or lapse_rate_k_per_m < 0.0:
        raise ValueError("base height and lapse rate must be non-negative")
    return float(t_air_k - lapse_rate_k_per_m * base_m)


@dataclass(frozen=True)
class CloudField:
    """A unit-variance 1/f^β field and the coverage mask at the requested fraction."""

    field: NDArray[np.float64]
    coverage: NDArray[np.bool_]
    beta: float
    fraction: float
    seed: int


def generate_cloud_field(
    shape: tuple[int, int], beta: float, cloud_fraction: float, seed: int
) -> CloudField:
    """Seeded Gaussian field with PSD ∝ f^{−β} (FFT synthesis), thresholded at the (1 − c)
    quantile so exactly round(c N) pixels are covered. β = 0 is white noise."""
    h, w = int(shape[0]), int(shape[1])
    if h < 2 or w < 2:
        raise ValueError("cloud field needs at least 2x2 pixels")
    if not 0.0 <= cloud_fraction <= 1.0:
        raise ValueError("cloud_fraction must lie in [0, 1]")
    if beta < 0.0:
        raise ValueError("beta must be non-negative")
    rng = np.random.default_rng(int(seed))
    white = rng.standard_normal((h, w))
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    f = np.sqrt(fx * fx + fy * fy)
    amplitude = np.zeros_like(f)
    nonzero = f > 0.0
    amplitude[nonzero] = f[nonzero] ** (-beta / 2.0)
    spectrum = np.fft.fft2(white) * amplitude
    field = np.real(np.fft.ifft2(spectrum))
    field = (field - field.mean()) / (field.std() if field.std() > 0 else 1.0)
    n_cover = int(round(cloud_fraction * h * w))
    coverage = np.zeros((h, w), dtype=bool)
    if n_cover >= h * w:
        coverage[:] = True
    elif n_cover > 0:
        order = np.argsort(field, axis=None)[::-1][:n_cover]
        coverage.flat[order] = True
    return CloudField(
        field=field,
        coverage=coverage,
        beta=float(beta),
        fraction=float(cloud_fraction),
        seed=int(seed),
    )


def psd_slope(field: NDArray[np.floating], f_min: float = 0.02, f_max: float = 0.3) -> float:
    """Slope −β of the radially averaged power spectral density in log–log over [f_min, f_max]
    cycles per pixel (the generator's self-test; also usable on a real sky/cloud patch)."""
    x = np.asarray(field, dtype=np.float64)
    x = x - x.mean()
    power = np.abs(np.fft.fft2(x)) ** 2
    fy = np.fft.fftfreq(x.shape[0])[:, None]
    fx = np.fft.fftfreq(x.shape[1])[None, :]
    f = np.sqrt(fx * fx + fy * fy).ravel()
    p = power.ravel()
    sel = (f >= f_min) & (f <= f_max)
    bins = np.logspace(np.log10(f_min), np.log10(f_max), 16)
    idx = np.digitize(f[sel], bins)
    centres = []
    means = []
    for b in range(1, bins.size):
        m = idx == b
        if m.sum() >= 4:
            centres.append(np.sqrt(bins[b - 1] * bins[b]))
            means.append(p[sel][m].mean())
    slope, _ = np.polyfit(np.log(centres), np.log(means), 1)
    return float(slope)


def cloud_radiance(
    l_clear: Any, l_base: float, tau_cloud: float, density: Any
) -> NDArray[np.float64]:
    """Per pixel: ``ε_eff L_B(T_base) + (1 − ε_eff) L_clear`` with ``ε_eff = (1 − τ) d``.

    ``density`` is :meth:`SkyFixedCloud.density`'s 0-to-1 depth along the ray, or a boolean
    coverage mask, which is the same thing at its two ends. The authored ``tau_cloud`` is the
    transmittance of cloud **at full depth**, so a ray through the middle of a bank reads exactly
    what it read before this generalisation, and a ray through the fringe reads mostly sky.

    Both limits are returned by ``np.where`` rather than by the blend, so a boolean mask gives
    **bit-identical** results to the hard-edged version for any ``tau_cloud`` -- the change is
    provably confined to the pixels that used to be one or the other and are now in between.
    """
    if not 0.0 <= tau_cloud < 1.0:
        raise ValueError("tau_cloud must lie in [0, 1)")
    clear = np.asarray(l_clear, dtype=np.float64)
    d = np.asarray(density, dtype=np.float64)
    if d.shape != clear.shape:
        raise ValueError("the density field and the clear radiance must have the same shape")
    if np.any(d < 0.0) or np.any(d > 1.0):
        raise ValueError("cloud density must lie in [0, 1]")
    cloudy = (1.0 - tau_cloud) * l_base + tau_cloud * clear
    emissivity = (1.0 - tau_cloud) * d
    blend = emissivity * l_base + (1.0 - emissivity) * clear
    return np.asarray(
        np.where(d >= 1.0, cloudy, np.where(d <= 0.0, clear, blend)), dtype=np.float64
    )


def sky_angles(
    direction: Any,
    up: Any = (0.0, 1.0, 0.0),
    forward: Any = (0.0, 0.0, -1.0),
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """``(elevation, azimuth)`` in radians for ``(..., 3)`` unit directions in stage axes.

    The one definition of these two angles, because **two consumers have to agree exactly**: the
    infrared background samples the cloud field along each pixel's ray, and the visible dome bakes
    the same field into a texture. If their conventions differed by so much as a sign the two
    halves of one frame pair would show cloud in different parts of the sky, each internally
    consistent and plausible.

    Elevation is measured from the horizon toward ``up``; azimuth runs from ``forward`` toward
    ``cross(forward, up)`` and wraps into [0, 2pi). Only stability matters for the sampling -- the
    same world direction must give the same pair every time -- not that zero lands on any
    particular bearing.
    """
    d = np.asarray(direction, dtype=np.float64)
    if d.shape[-1] != 3:
        raise ValueError(f"directions must be (..., 3), got {d.shape}")
    u = np.asarray(up, dtype=np.float64).reshape(3)
    u = u / np.linalg.norm(u)
    f = np.asarray(forward, dtype=np.float64).reshape(3)
    f = f - np.dot(f, u) * u
    norm = float(np.linalg.norm(f))
    if norm == 0.0:
        raise ValueError("forward must not be parallel to up")
    f = f / norm
    right = np.cross(f, u)
    elevation = np.arcsin(np.clip(np.sum(d * u, axis=-1), -1.0, 1.0))
    azimuth = np.mod(np.arctan2(np.sum(d * right, axis=-1), np.sum(d * f, axis=-1)), 2.0 * np.pi)
    return np.asarray(elevation), np.asarray(azimuth)


@dataclass(frozen=True)
class SkyFixedCloud:
    """Cloud coverage attached to the **sky**, sampled per ray, not painted on the image plane.

    Which frame the field lives in is the whole of the physics here, and only one of the three
    obvious choices behaves:

    * A field regenerated per frame **flickers** -- every frame is a different sky.
    * A field fixed to the **image plane** is stable, and moves with the sensor: a slewing mount
      carries its clouds along with it, so a tracked target never passes in front of one and the
      background never changes. Exactly backwards.
    * A field fixed to the **sky** (this one) is stable *and* stationary in the world, so slewing
      the mount sweeps the camera across it and a target crosses cloud edges. That is the geometry
      that makes cloud a clutter source rather than a texture.

    **The continuous field is kept and thresholded after interpolation**, rather than the boolean
    mask being stored and sampled. A grid cell here is half a degree and a Boson pixel is 0.049,
    so nearest-neighbour sampling of a mask gives cloud edges that are ten-pixel rectangular
    steps -- visibly wrong, and wrong in the way that matters most, since edge sharpness is
    precisely what a detector keys on. Interpolating the underlying 1/f^beta field and thresholding
    afterwards gives an edge resolved at the *sampling* resolution instead, at no extra storage.

    The grid is equirectangular in (elevation, azimuth), which stretches structure azimuthally as
    the zenith is approached -- an ``n_azimuth``-wide row spans 360 degrees at every elevation.
    For a sky-target sensor working at low to moderate elevation the distortion is small; looking
    near the zenith it is not, and a proper treatment would generate on the sphere.

    The field does not move. Wind advection is a rotation of the azimuth axis over time and is not
    modelled: over the seconds a flypast lasts, cloud drift is far below a pixel.
    """

    field: NDArray[np.float64]
    threshold: float
    beta: float
    fraction: float
    seed: int
    #: Width of the edge ramp, in standard deviations of the (unit-variance) field. See
    #: :meth:`density`. Zero restores the hard-edged binary cloud this class had first.
    softness: float = DEFAULT_EDGE_SOFTNESS

    def value(self, elevation_rad: Any, azimuth_rad: Any) -> NDArray[np.float64]:
        """The continuous field along each ray, bilinear over the grid.

        Azimuth wraps, because the grid is a full circle and the last column's neighbour is the
        first; elevation clamps, because the hemisphere ends. Getting the wrap wrong would leave a
        seam of discontinuous cloud down one bearing, which reads as a real feature.
        """
        n_el, n_az = self.field.shape
        el = np.asarray(elevation_rad, dtype=np.float64)
        az = np.asarray(azimuth_rad, dtype=np.float64)
        if el.shape != az.shape:
            raise ValueError(f"elevation {el.shape} and azimuth {az.shape} must match")
        row = np.clip(el / (0.5 * math.pi) * n_el - 0.5, 0.0, n_el - 1.0)
        col = np.mod(az / (2.0 * math.pi) * n_az - 0.5, n_az)
        r0 = np.floor(row).astype(np.int64)
        c0 = np.floor(col).astype(np.int64)
        fr = row - r0
        fc = col - c0
        r1 = np.minimum(r0 + 1, n_el - 1)
        c1 = np.mod(c0 + 1, n_az)
        top = self.field[r0, c0] * (1.0 - fc) + self.field[r0, c1] * fc
        bottom = self.field[r1, c0] * (1.0 - fc) + self.field[r1, c1] * fc
        return np.asarray(top * (1.0 - fr) + bottom * fr)

    def sample(self, elevation_rad: Any, azimuth_rad: Any) -> NDArray[np.bool_]:
        """Coverage along each ray: the interpolated field above its own threshold.

        This is the **definition of the covered fraction** and is what the ``fraction`` this
        object was built for refers to. :meth:`density` crosses 0.5 at exactly this threshold, so
        adding the edge ramp did not move the coverage statistic.
        """
        return np.asarray(self.value(elevation_rad, azimuth_rad) >= self.threshold)

    def density(self, elevation_rad: Any, azimuth_rad: Any) -> NDArray[np.float64]:
        """How much cloud is along each ray, 0 (clear) to 1 (full depth), smooth across the edge.

        **A cloud is not a stencil.** The first version of this class returned only
        :meth:`sample`'s boolean, so every covered ray got the cloud base's radiance in full and
        every other ray got none -- which renders as flat blobs with a one-pixel cliff round each
        one, in both bands. Real cumulus thins toward its edges: the liquid water path falls to
        zero over some distance, so the optical depth does too and the sky behind shows through.
        That is what this returns, and it is the quantity the radiance blend actually wants.

        The ramp is a smoothstep on the field's own excess over its threshold, measured in
        standard deviations -- the field is unit-variance by construction, so ``softness`` is
        directly in sigma. ``d = 0.5`` sits exactly at the threshold, which is what keeps
        :meth:`sample` and the authored coverage fraction meaning what they did.

        ``softness = 0`` gives back the hard mask exactly, so the old behaviour is a value of a
        parameter rather than a deleted branch.
        """
        value = self.value(elevation_rad, azimuth_rad)
        if self.softness <= 0.0:
            return np.asarray(value >= self.threshold, dtype=np.float64)
        x = np.clip((value - self.threshold) / self.softness + 0.5, 0.0, 1.0)
        # Smoothstep rather than a linear ramp: a linear one leaves a visible crease where it
        # meets the flat core, and the crease is a straight line in a picture of a cloud.
        return np.asarray(x * x * (3.0 - 2.0 * x), dtype=np.float64)


def generate_sky_cloud(
    beta: float,
    cloud_fraction: float,
    seed: int,
    *,
    n_elevation: int = 180,
    n_azimuth: int = 720,
) -> SkyFixedCloud:
    """A :class:`SkyFixedCloud` over the whole visible hemisphere at half-degree resolution.

    The coverage fraction is exact over the *grid*, which is the sky, not over any one frame --
    a camera pointed at a gap sees no cloud and one pointed at a bank sees only cloud, which is
    what a real sensor does and what makes cloud a false-alarm source worth simulating.
    """
    generated = generate_cloud_field((n_elevation, n_azimuth), beta, cloud_fraction, seed)
    # The threshold is taken on the **interpolated** field, not on the grid, and that is not a
    # detail. `generate_cloud_field` cuts at the level covering exactly c of the grid *cells*;
    # sampling between cells averages neighbours, which pulls values toward the mean, so the same
    # level covers noticeably less of a densely sampled sky -- measured at 25 % low. It does not
    # improve with a finer grid, because a 1/f^beta field is scale-invariant and bilinear
    # averaging therefore smooths it by the same relative amount at every scale. Estimating the
    # quantile on a 2x upsampling instead -- which contains both the unsmoothed cell centres and
    # the most-smoothed diagonal midpoints -- puts the realised coverage back on c.
    field = generated.field
    upsampled = np.concatenate(
        [
            field.ravel(),
            (0.5 * (field + np.roll(field, -1, axis=1))).ravel(),
            (0.5 * (field + np.roll(field, -1, axis=0))).ravel(),
            (
                0.25
                * (
                    field
                    + np.roll(field, -1, axis=0)
                    + np.roll(field, -1, axis=1)
                    + np.roll(np.roll(field, -1, axis=0), -1, axis=1)
                )
            ).ravel(),
        ]
    )
    threshold = (
        float(np.quantile(upsampled, 1.0 - cloud_fraction))
        if cloud_fraction > 0.0
        else float(upsampled.max()) + 1.0
    )
    return SkyFixedCloud(
        field=generated.field,
        threshold=threshold,
        beta=generated.beta,
        fraction=generated.fraction,
        seed=generated.seed,
    )
