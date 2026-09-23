"""A physically-generated environment dome for the **companion visible frame** (ADR 0073).

The infrared path never reads a colour AOV and nothing here changes one infrared pixel: a
``UsdLux.DomeLight`` is a light, not geometry, so a ray that sees it still reports instance id 0
and an infinite ``DistanceToCameraSD``, and the background keeps taking ``T_sky(theta)`` from the
sky model exactly as ADR 0060 specifies. This module exists because the alternative -- the
untextured grey dome the stage used to carry -- rendered the visible companion as a flat grey void
with a few grey squares in it, which tells a reader nothing about where the camera is pointing,
what time of day it is, or where the horizon falls.

What is generated, per direction:

* **Sky**, from the Preetham-Shirley-Smits analytic daylight model (SIGGRAPH 1999, "A Practical
  Analytic Model for Daylight"): a Perez sky-luminance distribution scaled to an absolute zenith
  luminance, with the CIE xy chromaticity distributed by the same functional form. Its two inputs
  are the sun's zenith angle and the atmospheric **turbidity**.
* **Ground**, below the horizon: a Lambertian terrain of a given albedo lit by the direct and
  diffuse irradiance the shared ``WeatherSeries`` already reports, faded into the horizon sky by
  Koschmieder's contrast transmittance over the same visibility.

Both inputs come from the scene rather than from taste. The sun position is the NOAA geometry of
:mod:`irsim.thermal.solar` at the scene's own site and start time; the turbidity is derived from
the shared weather's visibility against the atmosphere preset's Rayleigh coefficient
(:func:`turbidity_from_visibility`); the irradiance is the weather's DNI and DHI. So the visible
frame shows the same sun, at the same hour, through the same air as the infrared frame -- which is
the only reason it is worth looking at.

**What this is not.** The visible frame is a legibility aid. Only its *geometry* is a calibrated
claim (it is box-filtered onto the infrared pixel grid, so the pair is registered by construction).
Its photometry is a daylight-appearance model tone-mapped by the RTX path tracer and is not
traceable to a radiometric unit; no part of the sensor chain reads it. Nothing in
``docs/physics-model.md`` is implemented here, and nothing here may be cited as though it were.

Two known absences, both deliberate, both so the pair does not lie to the reader:

* **Cloud, when the scene has a field.** ADR 0073 originally left cloud off for a good reason --
  the infrared background did not have it, and a pair that disagrees is worse than a pair that is
  plain. ADR 0076 put the structured field into the infrared background, so the reason expired and
  the dome now samples **the same** :class:`~irsim.atmosphere.cloud.SkyFixedCloud`, through the
  same :func:`~irsim.atmosphere.cloud.sky_angles` convention. Not a second cloud that looks
  similar: the same object, so the two halves of a frame pair cannot drift.
* **No sun disc.** The disc is a ``UsdLux.DistantLight`` at the same NOAA direction, so that it
  casts shadows. Preetham's distribution carries the aureole around the sun but not the disc
  itself, so the two do not double-count.

Pure NumPy, no engine imports: the generator is unit-testable without Isaac Sim, and the USD side
of it lives in :mod:`irsim_isaac.aerial_demo`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from datetime import timedelta
from typing import Any

import numpy as np
from numpy.typing import NDArray

from irsim.atmosphere.cloud import SkyFixedCloud, sky_angles
from irsim.atmosphere.cloud_deck import CloudDeck
from irsim.scene import Scene
from irsim.thermal.solar import sun_position_utc

__all__ = [
    "KOSCHMIEDER_CONTRAST",
    "AEROSOL_SCALE_HEIGHT_M",
    "RAYLEIGH_SCALE_HEIGHT_M",
    "LUMINOUS_EFFICACY_DAYLIGHT_LM_W",
    "TURBIDITY_RANGE",
    "DOME_POLE_AXIS",
    "DomeSpec",
    "turbidity_from_visibility",
    "perez",
    "luminance_coefficients",
    "zenith_luminance_cd_m2",
    "zenith_chromaticity",
    "sky_luminance_xyy",
    "xyy_to_linear_rgb",
    "latlong_directions",
    "stage_direction",
    "environment_map",
    "dome_spec_from_scene",
    "dome_intensity",
    "DOME_EXPOSURE_TARGET",
]

#: Koschmieder's constant: meteorological visibility V is the range at which a black target falls
#: to the 2 % contrast threshold, so the extinction coefficient is ln(1/0.02) / V = 3.912 / V.
#: (Koschmieder 1924; the 2 % threshold is the WMO convention.)
KOSCHMIEDER_CONTRAST = 3.912

#: Scale heights used to turn ground-level extinction into column optical depth. Rayleigh follows
#: the density profile of the whole atmosphere; the continental aerosol layer is much shallower.
#: 8 km is the standard density scale height; 1.2 km is the usual continental-aerosol value
#: (Elterman 1968 / the LOWTRAN rural profile, which falls by 1/e over roughly 1-1.5 km).
RAYLEIGH_SCALE_HEIGHT_M = 8000.0
AEROSOL_SCALE_HEIGHT_M = 1200.0

#: Luminous efficacy of global daylight, lm/W. Used only to put the terrain's radiance on the same
#: photometric scale as Preetham's sky, which is authored in cd/m2. Littlefair (1985) reports
#: 100-120 lm/W for global daylight over a wide range of solar altitudes.
LUMINOUS_EFFICACY_DAYLIGHT_LM_W = 110.0

#: Preetham's model is fitted over this turbidity range; outside it the zenith-luminance
#: polynomial leaves the data it was regressed on, so the derived turbidity is clamped here.
TURBIDITY_RANGE = (1.8, 10.0)

# Perez distribution coefficients as (slope on turbidity, intercept), for luminance and for the
# two CIE chromaticity coordinates. Preetham et al. 1999, appendix A2.
_PEREZ_LUMINANCE = (
    (0.1787, -1.4630),
    (-0.3554, 0.4275),
    (-0.0227, 5.3251),
    (0.1206, -2.5771),
    (-0.0670, 0.3703),
)
_PEREZ_X = (
    (-0.0193, -0.2592),
    (-0.0665, 0.0008),
    (-0.0004, 0.2125),
    (-0.0641, -0.8989),
    (-0.0033, 0.0452),
)
_PEREZ_Y = (
    (-0.0167, -0.2608),
    (-0.0950, 0.0092),
    (-0.0079, 0.2102),
    (-0.0441, -1.6537),
    (-0.0109, 0.0529),
)

# Zenith chromaticity: a quadratic in turbidity whose coefficients are cubics in the solar zenith
# angle (radians). Rows are the T^2, T^1 and T^0 terms; columns the theta^3..theta^0 terms.
_ZENITH_X = (
    (0.00166, -0.00375, 0.00209, 0.0),
    (-0.02903, 0.06377, -0.03202, 0.00394),
    (0.11693, -0.21196, 0.06052, 0.25886),
)
_ZENITH_Y = (
    (0.00275, -0.00610, 0.00317, 0.0),
    (-0.04214, 0.08970, -0.04153, 0.00516),
    (0.15346, -0.26756, 0.06670, 0.26688),
)

# CIE XYZ (D65) to linear sRGB. IEC 61966-2-1.
_XYZ_TO_RGB = np.array(
    [
        [3.2406, -1.5372, -0.4986],
        [-0.9689, 1.8758, 0.0415],
        [0.0557, -0.2040, 1.0570],
    ],
    dtype=np.float64,
)

#: Renderer-side level the *median daylight sky* is exposed to, in the units the RTX dome light's
#: `intensity` multiplies its texture by. **Measured** on this build by sweeping the intensity on
#: the demo stage: intensity x median-sky-luminance near 100 renders the sky as mid-grey, near 400
#: it begins to wash out, and 10 is deep twilight. 300 puts a clear daytime sky where a camera's
#: own auto-exposure would put it.
DOME_EXPOSURE_TARGET = 300.0

#: Stage axis the RTX dome light puts the lat-long texture's pole on, with the azimuth running
#: from +X toward -Y. **Measured**, on Isaac Sim 6.1.0 with `omni:rtx:domeLight:mode = infinite`,
#: by rendering a coded environment map from the six axis directions: looking along +Z returned
#: the texture's theta < 90 hemisphere and -Z its theta > 90 one, +X the phi = 0 meridian and -Y
#: the phi = 90 one. It is *not* the USD documented behaviour of a Y-up stage, and Isaac Sim 6.0
#: is new enough that the public documentation does not settle it, so
#: `tests/integration/test_aerial_demo_isaac.py` re-measures it against the renderer rather than
#: trusting this constant.
DOME_POLE_AXIS = (0.0, 0.0, 1.0)

#: Below this solar elevation Preetham's fit is outside its data (it was regressed on daylight).
#: The map is evaluated at the limit and scaled down smoothly rather than going negative; see
#: :func:`_twilight_scale`.
_TWILIGHT_LIMIT_DEG = 3.0
#: Depression below the limit at which the dome reaches its night floor.
_TWILIGHT_SPAN_DEG = 9.0
_NIGHT_FLOOR = 0.002


@dataclass(frozen=True)
class DomeSpec:
    """Everything the environment map needs, all of it read off the scene.

    ``heading_deg`` is the compass bearing the stage's -Z axis points along, which is what ties the
    stage's arbitrary world frame to the sun's azimuth. 0 means the camera looks north.
    """

    sun_elevation_deg: float
    sun_azimuth_deg: float
    turbidity: float = 2.5
    heading_deg: float = 0.0
    dni_w_m2: float = 0.0
    dhi_w_m2: float = 0.0
    visibility_m: float = 23000.0
    ground_albedo: tuple[float, float, float] = (0.16, 0.17, 0.12)
    camera_height_m: float = 2.0
    #: The scene's cloud field, shared with the infrared background (ADR 0076). ``None`` leaves
    #: the dome clear, which is what it was before and what a clear-sky scene wants anyway.
    cloud: SkyFixedCloud | None = None
    #: The cloud **deck**, when the infrared band is marching one (AT.12). Takes precedence over
    #: ``cloud``, and must: a deck in the infrared band beside the hemispherical field on the dome
    #: would put cloud in different parts of the sky in the two halves of a frame pair, which is
    #: precisely what ADR 0076 exists to prevent. The dome **marches** it, exactly as the infrared
    #: background does, and turns the same optical depth into this band's opacity -- at a cost the
    #: dome can afford: it is baked once, where a frame is marched three hundred times.
    deck: CloudDeck | None = None
    #: Effective reflectance of a cloud **base**, which is the side a ground sensor sees. Not the
    #: 0.7-0.9 of a sunlit cloud top: a base is lit by light that has already been through the
    #: cloud, and whether it ends up brighter or darker than the sky beside it is left to the
    #: arithmetic rather than asserted here. ESTIMATED.
    cloud_base_albedo: float = 0.55

    def sun_zenith_rad(self) -> float:
        return math.radians(90.0 - self.sun_elevation_deg)

    def sun_azimuth_in_stage_rad(self) -> float:
        """Sun azimuth measured from the stage's -Z axis, positive toward +X."""
        return math.radians(self.sun_azimuth_deg - self.heading_deg)


def turbidity_from_visibility(
    visibility_m: float,
    rayleigh_extinction_per_m: float,
    *,
    aerosol_scale_height_m: float = AEROSOL_SCALE_HEIGHT_M,
    rayleigh_scale_height_m: float = RAYLEIGH_SCALE_HEIGHT_M,
) -> float:
    """Linke turbidity from the shared weather's visibility (dimensionless).

    Turbidity is defined on *column* optical depths -- T = (tau_molecular + tau_aerosol) /
    tau_molecular -- but visibility is a *ground-level* extinction coefficient, so the two layers
    have to be given their own scale heights before the ratio means anything. Taking the ground
    ratio instead would report T ~ 14 for a clear 23 km day, which is a dense haze.

    ``rayleigh_extinction_per_m`` is the atmosphere preset's visible ``gamma0_per_m``, which is
    Rayleigh scattering at 0.55 um; at the preset's 1.2e-5 /m it gives a column depth of 0.096,
    against the textbook 0.0973 at 550 nm, so the two really are the same quantity.
    """
    if visibility_m <= 0.0:
        raise ValueError("visibility must be positive")
    if rayleigh_extinction_per_m <= 0.0:
        raise ValueError("the Rayleigh extinction coefficient must be positive")
    total_per_m = KOSCHMIEDER_CONTRAST / float(visibility_m)
    aerosol_per_m = max(total_per_m - float(rayleigh_extinction_per_m), 0.0)
    tau_molecular = float(rayleigh_extinction_per_m) * rayleigh_scale_height_m
    tau_aerosol = aerosol_per_m * aerosol_scale_height_m
    turbidity = 1.0 + tau_aerosol / tau_molecular
    return float(np.clip(turbidity, *TURBIDITY_RANGE))


def _coefficients(
    table: tuple[tuple[float, float], ...], turbidity: float
) -> tuple[float, float, float, float, float]:
    a, b, c, d, e = (slope * turbidity + intercept for slope, intercept in table)
    return a, b, c, d, e


def luminance_coefficients(turbidity: float) -> tuple[float, float, float, float, float]:
    """The five Perez luminance coefficients A..E at a turbidity (Preetham et al. appendix A2)."""
    return _coefficients(_PEREZ_LUMINANCE, float(turbidity))


def perez(
    theta: Any,
    gamma: Any,
    coefficients: tuple[float, float, float, float, float],
) -> NDArray[np.float64]:
    """Perez sky distribution: (1 + A e^(B/cos theta)) (1 + C e^(D gamma) + E cos^2 gamma).

    ``theta`` is the zenith angle of the direction being evaluated and ``gamma`` its angle from the
    sun. B is negative for every turbidity, so the horizon term stays finite as cos theta -> 0;
    the cosine is floored anyway so the array form never divides by zero.
    """
    a, b, c, d, e = coefficients
    cos_theta = np.maximum(np.cos(np.asarray(theta, dtype=np.float64)), 1e-4)
    g = np.asarray(gamma, dtype=np.float64)
    return np.asarray(
        (1.0 + a * np.exp(b / cos_theta)) * (1.0 + c * np.exp(d * g) + e * np.cos(g) ** 2)
    )


def zenith_luminance_cd_m2(turbidity: float, sun_zenith_rad: float) -> float:
    """Absolute zenith luminance in cd/m2 (Preetham et al. 1999, eq. A.2)."""
    chi = (4.0 / 9.0 - turbidity / 120.0) * (math.pi - 2.0 * sun_zenith_rad)
    kcd = (4.0453 * turbidity - 4.9710) * math.tan(chi) - 0.2155 * turbidity + 2.4192
    return 1000.0 * max(kcd, 0.0)


def _zenith_poly(
    table: tuple[tuple[float, float, float, float], ...], turbidity: float, theta: float
) -> float:
    powers = (theta**3, theta**2, theta, 1.0)
    weights = (turbidity * turbidity, turbidity, 1.0)
    total = 0.0
    for weight, row in zip(weights, table, strict=True):
        total += weight * sum(c * p for c, p in zip(row, powers, strict=True))
    return total


def zenith_chromaticity(turbidity: float, sun_zenith_rad: float) -> tuple[float, float]:
    """CIE (x, y) at the zenith (Preetham et al. 1999, eq. A.1)."""
    return (
        _zenith_poly(_ZENITH_X, turbidity, sun_zenith_rad),
        _zenith_poly(_ZENITH_Y, turbidity, sun_zenith_rad),
    )


def sky_luminance_xyy(
    theta: Any,
    gamma: Any,
    turbidity: float,
    sun_zenith_rad: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Luminance (cd/m2) and CIE chromaticity of the clear sky in the given directions.

    Each of the three quantities is its zenith value times the ratio of the Perez distribution at
    the direction to the same distribution at the zenith, which is what normalises the shape
    function into an absolute quantity.
    """
    t = float(turbidity)
    zenith_gamma = float(sun_zenith_rad)
    zenith_x, zenith_y = zenith_chromaticity(t, zenith_gamma)
    out: list[NDArray[np.float64]] = []
    for table, zenith_value in (
        (_PEREZ_LUMINANCE, zenith_luminance_cd_m2(t, zenith_gamma)),
        (_PEREZ_X, zenith_x),
        (_PEREZ_Y, zenith_y),
    ):
        coeff = _coefficients(table, t)
        normal = float(perez(0.0, zenith_gamma, coeff))
        out.append(np.asarray(zenith_value * perez(theta, gamma, coeff) / normal))
    return out[0], out[1], out[2]


def xyy_to_linear_rgb(luminance: Any, x: Any, y: Any) -> NDArray[np.float64]:
    """CIE xyY -> linear sRGB (D65), stacked on a trailing axis of 3 and clipped at zero."""
    big_y = np.asarray(luminance, dtype=np.float64)
    cx = np.asarray(x, dtype=np.float64)
    cy = np.maximum(np.asarray(y, dtype=np.float64), 1e-6)
    big_x = cx / cy * big_y
    big_z = (1.0 - cx - cy) / cy * big_y
    xyz = np.stack([big_x, big_y, big_z], axis=-1)
    return np.asarray(np.maximum(xyz @ _XYZ_TO_RGB.T, 0.0))


def _twilight_scale(sun_elevation_deg: float) -> float:
    """Smooth fade applied below Preetham's validity, so a night scene is dark, not negative."""
    if sun_elevation_deg >= _TWILIGHT_LIMIT_DEG:
        return 1.0
    frac = (_TWILIGHT_LIMIT_DEG - sun_elevation_deg) / _TWILIGHT_SPAN_DEG
    return float(_NIGHT_FLOOR + (1.0 - _NIGHT_FLOOR) * math.exp(-3.0 * max(frac, 0.0) ** 2))


def latlong_directions(height: int) -> NDArray[np.float64]:
    """Unit direction, in **stage** axes, of every texel of a lat-long environment map.

    Returns ``(height, 2 * height, 3)``. Pixel centres, not edges, so neither pole is sampled
    exactly and the horizon falls on a row boundary for an even ``height``.

    The layout is not a choice -- it is the renderer's, and it was measured rather than assumed.
    See :data:`DOME_POLE_AXIS`: the texture's polar angle is taken from the stage's **+Z** axis
    and its azimuth runs from **+X toward -Y**, which on a Y-up stage puts the texture's poles on
    the horizon and its equator through the zenith. Building the map in elevation/azimuth and
    hoping is what produced a frame filled entirely with ground: at that layout the whole visible
    field fell inside one texture pole.
    """
    if height < 8 or height % 2:
        raise ValueError("height must be an even number of rows, at least 8")
    width = 2 * height
    theta = (np.arange(height, dtype=np.float64) + 0.5) * (math.pi / height)
    phi = (np.arange(width, dtype=np.float64) + 0.5) * (2.0 * math.pi / width)
    t, p = np.meshgrid(theta, phi, indexing="ij")
    return np.stack([np.sin(t) * np.cos(p), -np.sin(t) * np.sin(p), np.cos(t)], axis=-1)


def stage_direction(elevation_deg: Any, azimuth_deg: Any) -> NDArray[np.float64]:
    """Unit vector in stage axes for an elevation above the horizon and an azimuth off boresight.

    Stage convention, the one :func:`camera_space_position` already uses: +X right, +Y up, -Z the
    direction the untilted camera looks. Azimuth is measured from -Z toward +X.
    """
    el = np.deg2rad(np.asarray(elevation_deg, dtype=np.float64))
    az = np.deg2rad(np.asarray(azimuth_deg, dtype=np.float64))
    return np.stack([np.cos(el) * np.sin(az), np.sin(el), -np.cos(el) * np.cos(az)], axis=-1)


def environment_map(spec: DomeSpec, height: int = 512) -> NDArray[np.float32]:
    """The latitude-longitude environment map: ``(height, 2 * height, 3)`` float32, linear RGB.

    Authored in *direction* space -- every texel is turned into a stage-space unit vector by
    :func:`latlong_directions` and the sky is evaluated there -- so the texture layout is the
    renderer's business and the physics never has to know it.

    Float32 and never float16 -- not because a nit is a physical unit here, but because the
    encode/decode discipline of CLAUDE.md #2 applies to every buffer this repo writes, and a
    half-float sky is exactly the habit that ends up in a temperature path.
    """
    direction = latlong_directions(height)
    up = direction[..., 1]
    sun = stage_direction(spec.sun_elevation_deg, spec.sun_azimuth_deg - spec.heading_deg)

    # The sky is only defined on the upper hemisphere. A direction below the horizon is evaluated
    # *at* the horizon, which gives `_terrain` the colour its haze has to converge to.
    sky_theta = np.arccos(np.clip(up, 0.0, 1.0))
    gamma = np.arccos(np.clip(np.sum(direction * sun, axis=-1), -1.0, 1.0))

    # Preetham is a daylight model: evaluate it no lower than its validity limit and fade the
    # result, rather than extrapolating a fit into a regime it never saw.
    clamped = max(spec.sun_elevation_deg, _TWILIGHT_LIMIT_DEG)
    luminance, cx, cy = sky_luminance_xyy(
        sky_theta, gamma, spec.turbidity, math.radians(90.0 - clamped)
    )
    scale = _twilight_scale(spec.sun_elevation_deg)
    image = xyy_to_linear_rgb(luminance * scale, cx, cy)

    if spec.deck is not None:
        # **Marched, not sampled.** The infrared band integrates the deck along each ray, so the
        # visible band has to as well or the two disagree about where the cloud is -- which ADR
        # 0076 forbids and which a viewer spots instantly: taking the column depth where the ray
        # crosses the base reads a *vertical* thickness for an *oblique* look, so the dome showed
        # a few thin wisps over the same sky in which the infrared frame showed a wall of cumulus.
        # Over a rendered frame the two coverages were 26% and 57% of the same pixels.
        alpha = np.zeros(up.shape, dtype=np.float64)
        above = up > 0.0
        if above.any():
            el = np.arcsin(np.clip(up[above], -1.0, 1.0))
            az = np.arctan2(direction[..., 0][above], -direction[..., 2][above])
            tau = spec.deck.march(el, az).optical_depth
            # The authored optical depth is the visible one, so this band's opacity is Beer's law
            # on it directly, where the infrared band applies `CLOUD_OD_RATIO` first.
            alpha[above] = 1.0 - np.exp(-tau)
        lit = alpha > 0.0
        if lit.any():
            base = _cloud_base(spec, image[lit], scale)
            a = alpha[lit][..., None]
            image[lit] = (1.0 - a) * image[lit] + a * base
    elif spec.cloud is not None:
        elevation, azimuth = sky_angles(direction)
        above = up > 0.0
        # The same 0-to-1 depth the infrared background samples (ADR 0125), not a stencil. A
        # boolean mask replaced every covered pixel with one flat value, so the visible cloud was
        # a field of uniform grey blobs with a hard cliff round each one -- no thin edges, no
        # internal structure, and nothing that looked like a sky.
        depth = np.zeros(up.shape, dtype=np.float64)
        depth[above] = spec.cloud.density(elevation[above], azimuth[above])
        lit = depth > 0.0
        if lit.any():
            base = _cloud_base(spec, image[lit], scale)
            alpha = depth[lit][..., None]
            image[lit] = (1.0 - alpha) * image[lit] + alpha * base

    below = up < 0.0
    if below.any():
        image[below] = _terrain(spec, -up[below], image[below], scale)
    return np.ascontiguousarray(image, dtype=np.float32)


def _cloud_base(
    spec: DomeSpec, sky_behind: NDArray[np.float64], scale: float
) -> NDArray[np.float64]:
    """A Lambertian cloud base under the downwelling irradiance, blended by the cloud's opacity.

    The same shape as :func:`_terrain` and for the same reason: a surface of stated reflectance
    lit by the irradiance the shared weather already reports, rather than a colour chosen to look
    like cloud. Neutral grey, because cloud droplets scatter without much spectral preference --
    which is itself the visible signature, since the sky behind is strongly blue.

    Whether the base comes out brighter or darker than the sky beside it is **not** asserted here;
    it falls out of the irradiance and the reflectance, and it goes both ways. Measured on the
    midday scene -- a 61 degree sun, 820 W/m2 of downwelling -- the base is about 15 900 cd/m2
    against a 10 800 cd/m2 sky, so it is *brighter*, which is what a sunlit cumulus against a blue
    zenith looks like. At a low sun, or under a base thick enough to cut the transmitted light,
    the same expression makes it darker.

    None of which need agree with the infrared. In LWIR a cloud base is always *warmer* than a
    cold clear zenith and therefore always the brighter feature; in the visible it depends on the
    hour. A frame pair where cloud is bright in one band and dark in the other is not a bug, and
    that divergence is a real discriminator rather than an artefact.
    """
    irradiance = max(
        spec.dni_w_m2 * math.sin(math.radians(max(spec.sun_elevation_deg, 0.0))) + spec.dhi_w_m2,
        0.0,
    )
    base = spec.cloud_base_albedo * irradiance / math.pi * LUMINOUS_EFFICACY_DAYLIGHT_LM_W * scale
    return np.asarray(np.full_like(sky_behind, base))


def _terrain(
    spec: DomeSpec,
    sin_depression: NDArray[np.float64],
    horizon_sky: NDArray[np.float64],
    scale: float,
) -> NDArray[np.float64]:
    """Lambertian terrain faded into the horizon sky by Koschmieder's contrast transmittance.

    ``horizon_sky`` is the sky already evaluated at the horizon along the same direction, which is
    what the haze converges to, so the two hemispheres meet continuously instead of at a painted
    line.

    The dome sits at infinity and the terrain does not, so the fade needs a distance: for a camera
    ``camera_height_m`` above a flat plane, a ray at depression ``delta`` meets the ground at
    ``h / tan(delta)``. Looking straight down that is a few metres and unhazed; at the horizon it
    diverges and the terrain becomes the sky. This is a stand-in for terrain, not terrain: being
    painted on the dome it has no parallax, so a target at 120 m does not move against it.
    """
    irradiance = max(
        spec.dni_w_m2 * math.sin(math.radians(max(spec.sun_elevation_deg, 0.0))) + spec.dhi_w_m2,
        0.0,
    )
    albedo = np.asarray(spec.ground_albedo, dtype=np.float64)
    ground = albedo * irradiance / math.pi * LUMINOUS_EFFICACY_DAYLIGHT_LM_W * scale

    sin_d = np.clip(sin_depression, 1e-9, 1.0)
    tan_d = sin_d / np.sqrt(1.0 - np.minimum(sin_d, 1.0 - 1e-12) ** 2)
    distance_m = spec.camera_height_m / tan_d
    transmittance = np.exp(-KOSCHMIEDER_CONTRAST * distance_m / spec.visibility_m)[..., None]
    return np.asarray(transmittance * ground + (1.0 - transmittance) * horizon_sky)


def dome_spec_from_scene(
    scene: Scene,
    *,
    t_rel_s: float = 0.0,
    heading_deg: float = 0.0,
    camera_height_m: float = 2.0,
    ground_albedo: tuple[float, float, float] | None = None,
    cloud: SkyFixedCloud | None = None,
    deck: CloudDeck | None = None,
) -> DomeSpec:
    """Read the dome's inputs off the scene, so the visible frame cannot describe a different day.

    Every number here already exists and already drives the infrared side: the sun position is the
    NOAA geometry (:mod:`irsim.thermal.solar`) at the scene's own site and clock, the irradiance
    and the visibility are samples of the one ``WeatherSeries`` (CLAUDE.md #6), and the turbidity
    comes from that visibility against the atmosphere preset's visible Rayleigh coefficient. Left
    to a caller these would be four more knobs to keep in sync by hand, and the first time they
    drifted the companion frame would show a clear noon over an infrared dawn.
    """
    site = scene.spec.site
    when = scene.spec.start_utc + timedelta(seconds=float(t_rel_s))
    sun = sun_position_utc(site.latitude_deg, site.longitude_deg, when)
    weather = scene.weather_at(t_rel_s)
    rayleigh = scene.atmosphere_preset.bands["visible"].gamma0_per_m
    spec = DomeSpec(
        sun_elevation_deg=float(sun.elevation_deg),
        sun_azimuth_deg=float(sun.azimuth_deg),
        turbidity=turbidity_from_visibility(weather.visibility_m, rayleigh),
        heading_deg=float(heading_deg),
        dni_w_m2=weather.dni_w_m2,
        dhi_w_m2=weather.dhi_w_m2,
        visibility_m=weather.visibility_m,
        camera_height_m=float(camera_height_m),
        cloud=cloud,
        deck=deck,
    )
    return spec if ground_albedo is None else replace(spec, ground_albedo=ground_albedo)


def dome_intensity(spec: DomeSpec, image: Any) -> float:
    """The dome light's ``intensity`` for a generated map: an auto-exposure on the median sky.

    The texture stays in honest cd/m2 on disk and the *intensity* carries the exposure, because
    the sky's absolute level moves by two decades between dawn and noon and any fixed intensity
    blacks out one end of that. This is an auto-exposure and is meant to be: the companion frame
    exists to be looked at, and a frame that is correctly, uselessly black is not worth writing.
    It also means **the visible frame carries no absolute brightness information** -- brightness
    there is a display choice, not a measurement, and only the infrared outputs are in units.

    Night is the exception that has to be handled on purpose: normalising on the median sky alone
    would expose a moonless sky up to look like noon. The twilight fade is divided back out first,
    so the reference is the *daylight* sky the scene would have had and a night scene stays dark.
    """
    arr = np.asarray(image, dtype=np.float64)
    above = latlong_directions(arr.shape[0])[..., 1] > 0.0
    median = float(np.median(arr[above])) if above.any() else float(np.median(arr))
    reference = median / max(_twilight_scale(spec.sun_elevation_deg), 1e-9)
    return DOME_EXPOSURE_TARGET / max(reference, 1e-9)
