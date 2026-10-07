"""XD.6 part 2: real low cloud in ARM's calibrated imager, against an opaque cloud at its base.

The 92 ICI images of part 1 hold almost no low cloud, so the low-cloud minutes are read off the
deployment's own time-lapse. Its frames draw calibrated radiance on a fixed -5..35 W/(m^2 sr)
scale, and the colour inversion agrees with the NetCDF files to a median 0.13-0.24 W/(m^2 sr)
(``scripts/ici_timelapse_clouds.py``). ERA5 picks hours whose cloud is low only. The two
nearest airports' ceilometers give the base, and the reanalysis gives that height's air
temperature (``scripts/ici_low_cloud_bases.py``).

What is measured: an opaque cloud at the observed base, seen through the pinned clear air, is the
brightest the cloud can read (:func:`irsim.validation.ici.model_opaque_cloud`). Where that is
within 10 K of the scale's ceiling, a real cloud with opaque cores at the base's temperature would
saturate its brightest pixels. In 490 such cases they never do. Where the model is more than 10 K
above the ceiling, about 60 % do. Under the ESTIMATED VOx stand-in band, then, real low-cloud
cores read about 10 K colder than an opaque base at the ceilometer height's air.

What is not concluded: the ICI's band shape is unpublished, and it alone moves a cloud's
brightness temperature by several kelvin (ADR 0183). The ceilometers are 37 km away. So this
pins a measurement and does not change the cloud's physics; a spectral reference at the same
minutes (ARM's AERI) would decide it.
"""

from __future__ import annotations

import math
import pathlib

import numpy as np
import pytest

from irsim.atmosphere.library import load_atmosphere_preset
from irsim.radiometry.lut import BandLUT
from irsim.radiometry.spectral_response import load_spectral_response
from irsim.validation.ici import CEILING_W_M2_SR, load_ici_low_cloud, model_opaque_cloud

REPO = pathlib.Path(__file__).resolve().parents[2]
CSV = REPO / "data" / "validation" / "ici_sgp2023_low_cloud.csv"
VOX = REPO / "data" / "spectra" / "responses" / "boson_vox.csv"


@pytest.fixture(scope="module")
def minutes():  # type: ignore[no-untyped-def]
    return load_ici_low_cloud(CSV)


def test_the_derived_file_is_the_deployment_it_says_it_is(minutes) -> None:  # type: ignore[no-untyped-def]
    """461 low-cloud minutes from June to December 2023, 200 of them with a base both airports
    agree on, the bases those of low cloud and the radiances inside the time-lapse's scale."""
    assert len(minutes) == 461
    assert {m.utc[:7] for m in minutes} >= {"2023-06", "2023-10", "2023-12"}
    based = [m for m in minutes if math.isfinite(m.base_m)]
    assert len(based) == 200
    bases = np.array([m.base_m for m in based])
    assert np.percentile(bases, 5) > 100.0 and np.percentile(bases, 95) < 2500.0
    for m in minutes:
        values = np.concatenate([m.clear, m.cloud_p99])
        values = values[np.isfinite(values)]
        assert np.all(values >= -5.0) and np.all(values <= 35.0), m.utc
        assert m.elevations_deg.tolist() == [15.0, 20.0, 30.0, 45.0, 60.0, 75.0, 88.0]


@pytest.mark.slow
def test_real_low_cloud_cores_never_reach_an_opaque_base_within_ten_kelvin_of_the_ceiling(
    minutes,  # type: ignore[no-untyped-def]
) -> None:
    """The measurement, recorded: with the model's opaque base 0-10 K above the scale's
    ceiling (VOx band, 30 deg and up, minutes at least 10 % cloudy), the brightest real cloud
    pixels saturate in none of 490 cases; more than 10 K above, in 40-80 %."""
    response = load_spectral_response(VOX)
    lut = BandLUT.build(response, t0_k=150.0, t1_k=400.0, n=5001)
    preset = load_atmosphere_preset("us_standard_clear")

    def bt(radiance: float) -> float:
        return float(lut.apparent_temperature(np.float32(radiance)))

    ceiling = bt(CEILING_W_M2_SR)
    near, far = [], []
    for m in minutes:
        if not math.isfinite(m.base_m) or m.cloud_fraction < 0.1:
            continue
        model = model_opaque_cloud(m, preset, response, lut)
        for k, e in enumerate(m.elevations_deg):
            if e < 30.0 or not math.isfinite(m.cloud_p99[k]):
                continue
            above = bt(model[k]) - ceiling
            saturated = m.cloud_p99[k] >= CEILING_W_M2_SR
            if 0.0 <= above < 10.0:
                near.append(saturated)
            elif above >= 10.0:
                far.append(saturated)
    assert len(near) == 490 and not any(near)
    assert len(far) > 300 and 0.4 < float(np.mean(far)) < 0.8, float(np.mean(far))
