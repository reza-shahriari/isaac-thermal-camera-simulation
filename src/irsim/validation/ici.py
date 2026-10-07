"""The clear LWIR sky against a calibrated full-sky imager (XD.6; §7.1, §15 T3).

Every other comparison against reality in this project is made on 8-bit, post-AGC imagery, where
nothing radiometric survives (ADR 0068). ARM's Infrared Cloud Imager is the exception: an
upward-looking LWIR microbolometer imager at the Southern Great Plains site (May-Dec 2023,
doi:10.5439/3001561) whose files carry **calibrated sky radiance in W/(m^2 sr)** per pixel, held to
ARM's AERI spectrometer to 0.17 W/(m^2 sr) by the deployment's intercomparison report. That makes
it the first measurement the layered sky (ADR 0071, calibrated on one dry Tucson anchor) can be
held to across temperature and humidity.

What is compared, and what stands in for what we do not have:

* **The measurement** is ``data/validation/ici_sgp2023_clear_sky.csv``: per image, the median
  radiance of the *clear* pixels (the instrument's own cloud mask) in eight elevation bands, the
  cloud fraction, and the surface air temperature and PWV the instrument's processing printed on
  the deployment time-lapse (``scripts/ici_extract_profiles.py``,
  ``scripts/ici_met_from_video.py``).
* **The weather** is that one surface sample, held constant: ``T_air`` is the surface temperature,
  and the relative humidity is the one whose surface absolute humidity, spread over the preset's
  water-vapour scale height ``H_w``, holds the measured column: ``w0 = PWV / H_w``
  (:func:`surface_rh_for_pwv`). A humid night can need more than saturation at ``H_w`` = 2 km --
  the real column is deeper than the preset's -- and is then clamped to RH = 1 and flagged.
  **AT.37 pins the column instead** wherever the CSV carries the hour's surface dew point (ERA5,
  ``td_era5_c``): the relative humidity is then the measured one, and the water scale height is
  the one the measured PWV implies, ``H_w = PWV / w0`` (:func:`pinned_column`) -- 1.5 km on the
  May night, 2.6 km in August, 2.8 km in December. Fitting the continuum against a column the
  preset's fixed 2 km got wrong would have baked a profile error into it (ADR 0160); against a
  measured one it does not (ADR 0200).
* **The band** is the one thing the data cannot supply. The ICI is a 7.3-14 um microbolometer
  whose response *shape* is not published (the intercomparison report names it the main
  limitation of its own calibration). The comparison uses a microbolometer-shaped response --
  the project's ESTIMATED VOx curve by default -- and the result is sensitive to it: the band
  edges sit in near-opaque water and CO2 lines that carry ~40 % of a dry sky's radiance. A band
  change scales every sky alike, though; a *humidity-dependent* misfit cannot come from it.

docs/physics-model.md §7.1, §5.3 (a), §15 T3; ADR 0071, ADR 0183
"""

from __future__ import annotations

import csv
import math
import pathlib
import warnings
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from irsim.atmosphere.humidity import absolute_humidity_g_m3, saturation_vapour_pressure_hpa
from irsim.atmosphere.layered import LayeredAtmosphere
from irsim.config.atmosphere import AtmospherePreset
from irsim.config.bands import band_id_for
from irsim.radiometry.lut import BandLUT
from irsim.radiometry.spectral_response import SpectralResponse
from irsim.thermal.weather import WeatherSample, WeatherSeries

__all__ = [
    "IciImage",
    "load_ici_clear_sky",
    "surface_rh_for_pwv",
    "pinned_column",
    "model_clear_sky",
    "ratios",
    "IciCloudMinute",
    "load_ici_low_cloud",
    "model_opaque_cloud",
    "CEILING_W_M2_SR",
    "CLEAR_FRACTION_MAX",
    "ICI_BAND_UM",
]

#: The ICI's nominal band (the deployment's intercomparison report), um. Its *shape* is unpublished.
ICI_BAND_UM = (7.3, 14.0)
#: An image counts as clear when at most this share of its sky pixels is masked as cloud.
CLEAR_FRACTION_MAX = 0.005
#: g/m^3 x m -> cm of precipitable water (1 g/cm^2 of water is 1 cm).
_G_M2_PER_CM = 1.0e4


@dataclass(frozen=True)
class IciImage:
    """One ICI image reduced to its clear-pixel radiance profile and its surface met."""

    utc: str
    t_surface_k: float
    pwv_cm: float
    cloud_fraction: float
    elevations_deg: NDArray[np.float64]
    radiance: NDArray[np.float64]  # W/(m^2 sr), NaN where a band had too few clear pixels
    ici_model: NDArray[np.float64]  # the instrument's own clear-sky model, same bands
    #: The hour's surface dew point (ERA5), K; NaN where the CSV has none (AT.37).
    td_surface_k: float = math.nan

    @property
    def clear(self) -> bool:
        return self.cloud_fraction <= CLEAR_FRACTION_MAX


def load_ici_clear_sky(path: str | pathlib.Path) -> list[IciImage]:
    """Read the derived CSV (``#`` lines are its provenance header)."""
    text = pathlib.Path(path).read_text(encoding="utf-8").splitlines()
    rows = list(csv.DictReader(line for line in text if not line.startswith("#")))
    if not rows:
        raise ValueError(f"{path}: no images")
    elevations = sorted(int(k[4:]) for k in rows[0] if k.startswith("l_el"))
    out = []
    for r in rows:

        def num(key: str, row: dict[str, str] = r) -> float:
            return float(row[key]) if row[key] != "" else math.nan

        out.append(
            IciImage(
                utc=r["utc"],
                t_surface_k=float(r["t_surface_c"]) + 273.15,
                pwv_cm=float(r["pwv_cm"]),
                cloud_fraction=float(r["cloud_fraction"]),
                elevations_deg=np.asarray(elevations, dtype=np.float64),
                radiance=np.asarray([num(f"l_el{e}") for e in elevations], dtype=np.float64),
                ici_model=np.asarray([num(f"ici_model_el{e}") for e in elevations]),
                td_surface_k=num("td_era5_c") + 273.15 if "td_era5_c" in r else math.nan,
            )
        )
    return out


def surface_rh_for_pwv(t_k: float, pwv_cm: float, scale_height_m: float) -> tuple[float, float]:
    """``(RH used, RH needed)`` for a surface humidity that holds ``pwv_cm`` over ``H_w``.

    The layered model's water falls off as ``w0 e^{-h/H_w}``, whose column is ``w0 H_w``; so
    ``w0 = PWV / H_w``. The second value is unclamped: above 1 the preset's ``H_w`` is too shallow
    for that column, and the first value (clamped to 1) under-supplies it.
    """
    if pwv_cm < 0.0 or scale_height_m <= 0.0:
        raise ValueError("pwv_cm must be >= 0 and scale_height_m > 0")
    w0 = pwv_cm * _G_M2_PER_CM / scale_height_m
    needed = w0 / absolute_humidity_g_m3(t_k, 1.0)
    return min(needed, 1.0), needed


def pinned_column(image: IciImage) -> tuple[float, float] | None:
    """``(RH, H_w)`` measured for the image's night (AT.37), or None without a dew point.

    The surface relative humidity is the dew point's (Magnus, the project's coefficients), and
    the water-vapour scale height is the one that holds the measured column over that surface
    humidity: ``H_w = PWV / w0``, the layered model's own exponential column read backwards.
    """
    if not math.isfinite(image.td_surface_k):
        return None
    t_c, td_c = image.t_surface_k - 273.15, image.td_surface_k - 273.15
    rh = min(saturation_vapour_pressure_hpa(td_c) / saturation_vapour_pressure_hpa(t_c), 1.0)
    w0 = absolute_humidity_g_m3(image.t_surface_k, rh)
    return rh, image.pwv_cm * _G_M2_PER_CM / w0


def model_clear_sky(
    image: IciImage,
    preset: AtmospherePreset,
    response: SpectralResponse,
    lut: BandLUT,
    band: str | None = None,
    *,
    pin: bool = True,
) -> NDArray[np.float64]:
    """The layered model's clear sky at the image's elevations, W/(m^2 sr) in ``response``.

    The weather is the image's surface sample held constant -- wind 1 m/s, no sun (the ICI
    images here are night and the LWIR sky has no scattered term worth the name; ADR 0086).
    With ``pin`` (the default) and a dew point on the image, the column is the measured one
    (:func:`pinned_column`); otherwise the humidity is solved to hold the PWV over the preset's
    own ``H_w`` (:func:`surface_rh_for_pwv`), as part 1 did.
    """
    band = band or band_id_for(*ICI_BAND_UM)  # the registry names it, not this module
    column = pinned_column(image) if pin else None
    if column is not None:
        rh, scale_height = column
        profile = preset.profile.model_copy(update={"water_vapour_scale_height_m": scale_height})
        preset = preset.model_copy(update={"profile": profile})
    else:
        scale_height = float(preset.profile.water_vapour_scale_height_m)
        rh, _ = surface_rh_for_pwv(image.t_surface_k, image.pwv_cm, scale_height)
    weather = WeatherSeries.constant(
        WeatherSample(image.t_surface_k, rh, 1.0, 0.0, 0.0, 0.0, 23000.0, 0.0), 3600.0
    )
    atmosphere = LayeredAtmosphere(preset, weather, {band: lut}, {band: response})
    with warnings.catch_warnings():
        # the anchor solve warns when a narrow camera sheds classes; not this comparison's concern
        warnings.simplefilter("ignore")
        return np.asarray(
            [
                atmosphere.sky_radiance(band, 0.0, math.radians(float(e)))
                for e in image.elevations_deg
            ],
            dtype=np.float64,
        )


def ratios(
    images: Sequence[IciImage],
    preset: AtmospherePreset,
    response: SpectralResponse,
    lut: BandLUT,
) -> NDArray[np.float64]:
    """Model over measurement, ``(n_images, n_elevations)``, NaN where the measurement is."""
    return np.asarray(
        [model_clear_sky(im, preset, response, lut) / im.radiance for im in images],
        dtype=np.float64,
    )


# -- XD.6 part 2: low cloud --------------------------------------------------------------------

#: The deployment time-lapse draws radiance on a fixed -5..35 W/(m^2 sr) scale; at and above this
#: its colour is the bar's top, so a pixel reads "at least this" (scripts/ici_timelapse_clouds.py).
CEILING_W_M2_SR = 34.5


@dataclass(frozen=True)
class IciCloudMinute:
    """One low-cloud minute of the deployment, read off its time-lapse (XD.6 part 2)."""

    utc: str
    t_surface_k: float
    pwv_cm: float
    td_surface_k: float
    cloud_fraction: float
    is_day: bool
    #: The two nearest airports' ceilometer base (m above the site) and the air temperature there
    #: (K); NaN where the stations did not agree (``scripts/ici_low_cloud_bases.py``).
    base_m: float
    t_base_k: float
    elevations_deg: NDArray[np.float64]
    clear: NDArray[np.float64]  # clear pixels' median, NaN where too few
    cloud_p99: NDArray[np.float64]  # the brightest cloud pixels, NaN where too few
    cloud_saturated: NDArray[np.float64]  # share of cloud pixels at the scale's ceiling


def load_ici_low_cloud(path: str | pathlib.Path) -> list[IciCloudMinute]:
    """Read ``data/validation/ici_sgp2023_low_cloud.csv``."""
    text = pathlib.Path(path).read_text(encoding="utf-8").splitlines()
    rows = list(csv.DictReader(line for line in text if not line.startswith("#")))
    if not rows:
        raise ValueError(f"{path}: no minutes")
    elevations = sorted(int(k[len("clear_el") :]) for k in rows[0] if k.startswith("clear_el"))
    out = []
    for r in rows:

        def num(key: str, row: dict[str, str] = r) -> float:
            return float(row[key]) if row.get(key, "") != "" else math.nan

        out.append(
            IciCloudMinute(
                utc=r["utc"],
                t_surface_k=float(r["t_surface_c"]) + 273.15,
                pwv_cm=float(r["pwv_cm"]),
                td_surface_k=float(r["era5_td_c"]) + 273.15,
                cloud_fraction=num("cloud_fraction"),
                is_day=r["is_day"] == "1",
                base_m=num("obs_base_m"),
                t_base_k=num("t_base_c") + 273.15,
                elevations_deg=np.asarray(elevations, dtype=np.float64),
                clear=np.asarray([num(f"clear_el{e}") for e in elevations]),
                cloud_p99=np.asarray([num(f"cloud_p99_el{e}") for e in elevations]),
                cloud_saturated=np.asarray([num(f"cloud_sat_el{e}") for e in elevations]),
            )
        )
    return out


def model_opaque_cloud(
    minute: IciCloudMinute,
    preset: AtmospherePreset,
    response: SpectralResponse,
    lut: BandLUT,
    band: str | None = None,
) -> NDArray[np.float64]:
    """An opaque cloud at the minute's observed base, seen through its clear air, W/(m^2 sr).

    ``L = L_clear − τ(0 → R) L_beyond(R) + τ(0 → R) L_B(T_base)`` at the slant range ``R`` to a
    plane at the base: the identity the sky model's own cloud blend is built on (ADR 0126),
    with ``T_base`` the reanalysis air at the ceilometer's height rather than a lifted parcel's.
    The column is pinned as the clear-sky comparison's is (:func:`pinned_column`). It is the
    *brightest* a cloud there can be; a real cloud's brightest pixels reaching it says the cores
    are opaque at that temperature.
    """
    band = band or band_id_for(*ICI_BAND_UM)
    if not (math.isfinite(minute.base_m) and math.isfinite(minute.t_base_k)):
        return np.full(minute.elevations_deg.shape, math.nan)
    t_c, td_c = minute.t_surface_k - 273.15, minute.td_surface_k - 273.15
    rh = min(saturation_vapour_pressure_hpa(td_c) / saturation_vapour_pressure_hpa(t_c), 1.0)
    w0 = absolute_humidity_g_m3(minute.t_surface_k, rh)
    profile = preset.profile.model_copy(
        update={"water_vapour_scale_height_m": minute.pwv_cm * _G_M2_PER_CM / w0}
    )
    weather = WeatherSeries.constant(
        WeatherSample(minute.t_surface_k, rh, 1.0, 0.0, 0.0, 0.0, 23000.0, 0.0), 3600.0
    )
    atmosphere = LayeredAtmosphere(
        preset.model_copy(update={"profile": profile}), weather, {band: lut}, {band: response}
    )
    l_base = float(np.asarray(lut.lookup(np.float64(minute.t_base_k)))[()])
    out = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for e in minute.elevations_deg:
            el = math.radians(float(e))
            slant = max(minute.base_m, 1.0) / math.sin(el)
            tau = float(atmosphere.transmittance(band, 0.0, slant, el)[()])
            beyond = atmosphere.sky_beyond(band, 0.0, slant, el)
            clear = atmosphere.sky_radiance(band, 0.0, el)
            out.append(clear - tau * beyond + tau * l_base)
    return np.asarray(out, dtype=np.float64)
