"""The sea model's angular validity envelope, and what fraction of a frame lies outside it (SE.1).

`test_sea_surface.py` checks that the sea model is self-consistent. This file checks something it
cannot: **whether anyone has measured the sea at the angles we ask the model about.**

The finding, in one line: **every shore and mast frame this project renders is 100 % outside the
validated envelope, and every down-looking airborne frame is 100 % inside it.** At a 20 m eye
height the 50° limit is crossed at 31 m of slant range, so the whole maritime working band --
hundreds of metres to the horizon -- runs on an extrapolation.

⚠️ **The isothermal identity is deliberately not the test here.** It is the load-bearing check in
`test_sea_surface.py` precisely because it holds for *any* emissivity, which is the same reason it
cannot see this: a sea model with a badly wrong angular emissivity satisfies it exactly.
:func:`test_the_isothermal_identity_cannot_see_the_envelope` demonstrates that rather than asserting
it, by holding the identity while the emissivity is replaced with a constant.

What does discriminate, in rising order of sharpness:

* the **axis**: the envelope is indexed by view zenith at the *surface*, not by camera depression,
  and the two differ by the horizon dip -- exactly at the grazing end the envelope is about;
* the **geometry**: shore 1.0000 beyond, airborne 0.0000, so the flag separates two real
  deployments instead of firing everywhere;
* the **magnitude**: our facet-integrated emissivity falls 2.07 % by 55° in a calm sea, inside the
  published 2-3 %, and crosses 3 % at 7.3 m/s of wind -- so the envelope turns out to have a wind
  axis the roadmap row did not know about.

docs/physics-model.md §5.3, §15; roadmap SE.1; ADR 0078, ADR 0079, ADR 0118
"""

from __future__ import annotations

import copy
import itertools
import math
import pathlib

import numpy as np
import pytest
import yaml

from irsim.atmosphere import LayeredAtmosphere, SkyModel, load_atmosphere_preset
from irsim.atmosphere.sea import SeaModel, horizon_depression_rad, slant_range_m
from irsim.atmosphere.sea_envelope import (
    PUBLISHED_DROP_BY_55_DEG,
    VALIDATED_WIND_M_S,
    VALIDATED_ZENITH_DEG,
    beyond_envelope,
    depression_at_zenith_rad,
    envelope_report,
    view_zenith_rad,
)
from irsim.config.environment import load_environment_preset
from irsim.config.sensor import SensorConfig
from irsim.materials.library import MaterialLibrary
from irsim.materials.nk import load_nk_table
from irsim.materials.table import MaterialTable
from irsim.radiometry.lut import BandLUT
from irsim.radiometry.spectral_response import load_spectral_response
from irsim.thermal.weather import WeatherSample, WeatherSeries
from irsim.validation.maritime_scene import build_maritime_gbuffer

# GT.1: an end-to-end frame and a band-average sweep, not a unit test.
pytestmark = pytest.mark.slow

REPO = pathlib.Path(__file__).resolve().parents[2]
BOSON_RESPONSE = REPO / "data" / "spectra" / "responses" / "boson_vox.csv"
SST_K = 290.0
T_AIR_K = 291.0
SHORE_HEIGHT_M = 20.0


@pytest.fixture(scope="module")
def rig():  # type: ignore[no-untyped-def]
    response = load_spectral_response(BOSON_RESPONSE)
    return response, BandLUT.build(response), load_nk_table("water")


def _sea(rig, wind=5.0, height=SHORE_HEIGHT_M, sst=SST_K):  # type: ignore[no-untyped-def]
    response, lut, table = rig
    weather = WeatherSeries.constant(
        WeatherSample(T_AIR_K, 0.5, wind, 0.0, 0.0, 0.0, 23000.0, 0.0), 3600.0
    )
    atm = LayeredAtmosphere(load_atmosphere_preset("us_standard_clear"), weather, {"lwir": lut})
    sky = SkyModel(atm, load_environment_preset("clear_dry"), "lwir", lut)
    return sky, SeaModel(sky, table, response, bulk_sst_k=sst, camera_height_m=height)


def _sensor(width=640, height=512, focal_mm=None):  # type: ignore[no-untyped-def]
    raw = yaml.safe_load((REPO / "configs" / "sensors" / "flir_boson_640_lwir.yaml").read_text())
    d = copy.deepcopy(raw)
    d["sensor"]["fpa"].update(width=width, height=height)
    d["sensor"]["optics"]["supersample_factor"] = 1
    if focal_mm is not None:
        d["sensor"]["optics"]["focal_length_mm"] = focal_mm
    return SensorConfig.model_validate(d)


# -- the axis: view zenith at the surface, not camera depression ----------------------------


@pytest.mark.parametrize("height_m", [2.0, 20.0, 100.0, 2000.0, 10000.0])
def test_the_view_zenith_is_exactly_grazing_at_the_horizon(height_m: float) -> None:
    """θ = 90° at the geometric horizon and 0° at nadir, for every camera height.

    These two anchors are what fix the conversion, and the horizon one is the reason it cannot be
    the flat-earth ``90° − δ``: that form reads 89.856° at a 20 m horizon and 86.79° at a 10 km
    one, so a 10 km camera would be told its most grazing ray is 3.2° short of grazing. The
    envelope is a statement about how close to grazing the model is being pushed, so an error that
    grows with height in exactly that quantity is not survivable.
    """
    horizon = horizon_depression_rad(height_m)
    assert math.degrees(float(view_zenith_rad(height_m, horizon))) == pytest.approx(90.0, abs=1e-6)
    nadir_deg = math.degrees(float(view_zenith_rad(height_m, math.pi / 2)))
    assert nadir_deg == pytest.approx(0.0, abs=1e-9)

    # and the flat-earth form is short by exactly the horizon dip, which is height-dependent
    flat = 90.0 - math.degrees(horizon)
    assert 90.0 - flat == pytest.approx(math.degrees(horizon), abs=1e-9)


def test_the_conversion_is_the_flat_earth_one_plus_a_curvature_term() -> None:
    """Away from grazing the two agree to millidegrees; at grazing they part by the dip.

    Stated as a bound rather than a fact about one angle, so the test says where the spherical
    form is actually earning its keep.
    """
    for height_m in (20.0, 2000.0):
        horizon = horizon_depression_rad(height_m)
        steep = np.radians(np.array([30.0, 45.0, 60.0, 89.0]))
        spherical = np.degrees(view_zenith_rad(height_m, steep))
        flat = 90.0 - np.degrees(steep)
        assert np.max(np.abs(spherical - flat)) < 0.2, (height_m, spherical - flat)

        at_horizon = float(np.degrees(view_zenith_rad(height_m, horizon)))
        assert at_horizon - (90.0 - math.degrees(horizon)) == pytest.approx(
            math.degrees(horizon), abs=1e-6
        )


def test_a_ray_above_the_horizon_has_no_incidence_angle_on_the_sea() -> None:
    with pytest.raises(ValueError, match="never hits the sea"):
        view_zenith_rad(20.0, math.radians(0.10))


# -- the flag --------------------------------------------------------------------------------


def test_the_flag_turns_over_exactly_at_the_published_limit() -> None:
    """Inside at 49.99°, outside at 50.01°, and the crossing depression inverts the conversion."""
    height = SHORE_HEIGHT_M
    crossing = depression_at_zenith_rad(height)
    assert math.degrees(crossing) == pytest.approx(40.0002, abs=1e-4)
    assert math.degrees(float(view_zenith_rad(height, crossing))) == pytest.approx(
        VALIDATED_ZENITH_DEG, abs=1e-9
    )

    inside = depression_at_zenith_rad(height, math.radians(VALIDATED_ZENITH_DEG - 0.01))
    outside = depression_at_zenith_rad(height, math.radians(VALIDATED_ZENITH_DEG + 0.01))
    assert not bool(beyond_envelope(height, inside))
    assert bool(beyond_envelope(height, outside))
    assert inside > outside, "a steeper look is a smaller zenith, so the inverse is decreasing"


def test_the_envelope_ends_31_m_from_a_shore_camera() -> None:
    """The number that makes the risk legible: everything past a boat length is extrapolated.

    31 m at 20 m of eye height, and it scales with the height rather than with anything about the
    sea -- 3.1 km from a 2 km UAV. So the envelope is not a small correction at long range, it is
    a statement that low cameras and high cameras are in different validation regimes.
    """
    for height, expected in ((20.0, 31.1), (100.0, 155.6), (2000.0, 3110.8)):
        crossing = depression_at_zenith_rad(height)
        assert float(slant_range_m(height, crossing)) == pytest.approx(expected, rel=1e-3)
        # independent oracle: for a surface this close the earth is flat, so the ray from a
        # camera at `height` meeting the water at 50° from the vertical is `height / cos 50°`
        assert float(slant_range_m(height, crossing)) == pytest.approx(
            height / math.cos(math.radians(VALIDATED_ZENITH_DEG)), rel=2e-3
        )


# -- the discriminating measurement ----------------------------------------------------------


@pytest.mark.parametrize(
    ("label", "boresight_deg", "height_m", "expect_beyond"),
    [
        ("shore camera, just below the horizon", -1.0, 20.0, 1.0),
        ("shore camera, well down", -20.0, 20.0, 1.0),
        ("mast camera", -5.0, 100.0, 1.0),
        ("UAV at 2 km, oblique", -60.0, 2000.0, 0.0),
        ("UAV at 2 km, near nadir", -90.0, 2000.0, 0.0),
    ],
)
def test_the_envelope_separates_shore_cameras_from_airborne_ones(
    rig, label: str, boresight_deg: float, height_m: float, expect_beyond: float
) -> None:  # type: ignore[no-untyped-def]
    """The headline: shore frames are **entirely** outside, airborne frames **entirely** inside.

    This is the test the step exists for, and it is sharp in both directions. A flag that fired on
    every frame would be a constant dressed as a measurement; one that fired on none would be
    dead code. Getting 1.0000 and 0.0000 out of the same function on real sensor geometry is what
    says the envelope is being evaluated rather than assumed.
    """
    materials = MaterialTable.from_library(MaterialLibrary.load(), "lwir")
    sky, sea = _sea(rig, height=height_m)
    scene = build_maritime_gbuffer(
        _sensor().sensor, sky, sea, materials, boresight_elevation_deg=boresight_deg
    )
    report = scene.envelope_report()
    print(f"{label}: {report.summary()}")

    assert report.n_samples == int(scene.sea_mask.sum())
    assert report.fraction_beyond == pytest.approx(expect_beyond, abs=1e-9)
    assert report.inside is (expect_beyond == 0.0)
    assert report.camera_height_m == pytest.approx(height_m)
    # the report's own bounds must bracket what the flag decided
    assert (report.zenith_max_deg > VALIDATED_ZENITH_DEG) is (expect_beyond > 0.0)


def test_recording_the_envelope_against_depression_inverts_the_answer(rig) -> None:  # type: ignore[no-untyped-def]
    """The negative control, and it fails **backwards**, which is the dangerous direction.

    A plausible-looking implementation records "50 degrees" against the camera's depression angle,
    because that is the number a scene config carries. It does not merely blur the distinction --
    it reverses it. The shore frame (0.19-2.15° of depression) reads **0.000 beyond**, i.e. fully
    validated, when in truth none of it is; the oblique airborne frame (45.5-72.3° down) reads
    **0.867 beyond** when in truth all of it is inside the envelope. So the wrong axis would
    certify the one deployment that is entirely extrapolated and warn about the one that is not.

    That is why this file pins the axis with its own test rather than trusting the conversion:
    both axes produce a plausible fraction, and only one of them is about the sea.
    """
    materials = MaterialTable.from_library(MaterialLibrary.load(), "lwir")
    fractions_zenith, fractions_depression = [], []
    for boresight_deg, height_m in ((-1.0, 20.0), (-60.0, 2000.0)):
        sky, sea = _sea(rig, height=height_m)
        scene = build_maritime_gbuffer(
            _sensor().sensor, sky, sea, materials, boresight_elevation_deg=boresight_deg
        )
        depression = scene.depression_rad[scene.sea_mask]
        fractions_zenith.append(scene.envelope_report().fraction_beyond)
        # the wrong axis: "is the camera looking down by more than 50 degrees?"
        fractions_depression.append(float(np.mean(np.degrees(depression) > VALIDATED_ZENITH_DEG)))

    shore_z, air_z = fractions_zenith
    shore_d, air_d = fractions_depression
    assert (shore_z, air_z) == (pytest.approx(1.0), pytest.approx(0.0))
    assert shore_d == pytest.approx(0.0), "the wrong axis calls the shore frame fully validated"
    assert air_d == pytest.approx(0.867, abs=0.01), "and the airborne frame mostly not"
    # the orderings are opposite: that is the inversion, stated as the relation and not as a pair
    assert shore_z > air_z and shore_d < air_d


def test_a_frame_with_no_sea_reports_no_samples_rather_than_raising() -> None:
    """An empty mask is an answer, not an error -- the caller asked what fraction is outside."""
    report = envelope_report(20.0, np.zeros((4, 4)), np.zeros((4, 4), dtype=bool))
    assert report.n_samples == 0
    assert report.n_beyond == 0
    assert report.fraction_beyond == 0.0
    assert report.inside is True
    assert math.isnan(report.zenith_min_deg) and math.isnan(report.zenith_max_deg)
    assert "0.0000 of 0 sea samples" in report.summary()


# -- the magnitude, against the published figure ---------------------------------------------


def test_the_emissivity_drop_by_55_degrees_matches_published_in_situ_radiometry(rig) -> None:  # type: ignore[no-untyped-def]
    """2.07 % in a calm sea, inside the published 2-3 %: an external check the identity cannot
    give.

    This is the one place the angular emissivity is compared against something outside this repo.
    It is the *facet-integrated* value, not the flat-Fresnel one, because the published figure is
    a measurement of a sea and a sea has waves; the flat-facet number is 2.01 %, so the two happen
    to agree here, but they diverge fast with wind and the next test is about that.
    """
    _, sea = _sea(rig, wind=0.0)
    alpha, weight = sea.facet_quadrature(0.0)

    def integrated(theta_deg: float) -> float:
        cos_i = np.abs(np.cos(np.radians(theta_deg) - alpha))
        return float(np.sum(sea.emissivity(cos_i) * weight))

    nadir = integrated(0.0)
    assert nadir == pytest.approx(0.988, abs=0.002), nadir
    drop = (nadir - integrated(55.0)) / nadir
    lo, hi = PUBLISHED_DROP_BY_55_DEG
    assert lo <= drop <= hi, f"drop {drop:.4f} outside the published {lo}-{hi}"
    assert drop == pytest.approx(0.0207, abs=2e-4)

    # the limit itself sits where the effect is still small: the envelope ends because the
    # *evidence* does, not because the physics turns over
    assert (nadir - integrated(VALIDATED_ZENITH_DEG)) / nadir < lo


def test_the_published_drop_is_only_reproduced_below_a_wind(rig) -> None:  # type: ignore[no-untyped-def]
    """The envelope has a second axis: past 7.3 m/s our drop at 55° exceeds the published 3 %.

    The roadmap row records the envelope as an angle. It is not only an angle -- the published
    figure is a measurement of a sea, so it carries that sea's roughness, and our facet spread
    widens with wind until more of the tilt distribution sits in the steep part of the Fresnel
    curve. At 15 m/s the drop is 4.3 %, half again outside the published range, at an angle the
    published range is supposed to cover. A contributor who reads only the angle will over-trust
    a rough-sea frame at 45°.
    """
    _, hi = PUBLISHED_DROP_BY_55_DEG

    def drop_at(wind: float) -> float:
        _, sea = _sea(rig, wind=wind)
        alpha, weight = sea.facet_quadrature(0.0)

        def integrated(theta_deg: float) -> float:
            cos_i = np.abs(np.cos(np.radians(theta_deg) - alpha))
            return float(np.sum(sea.emissivity(cos_i) * weight))

        nadir = integrated(0.0)
        return (nadir - integrated(55.0)) / nadir

    below = drop_at(VALIDATED_WIND_M_S - 0.5)
    above = drop_at(VALIDATED_WIND_M_S + 0.5)
    assert below <= hi < above, (below, above)
    assert drop_at(15.0) == pytest.approx(0.0433, abs=5e-4)
    # monotone in wind, so a single crossing speed is meaningful
    drops = [drop_at(u) for u in (0.0, 2.0, 5.0, 10.0, 15.0)]
    assert all(b > a for a, b in itertools.pairwise(drops)), drops


# -- what the identity cannot see -------------------------------------------------------------


def test_the_isothermal_identity_cannot_see_the_envelope(rig) -> None:  # type: ignore[no-untyped-def]
    """Demonstrated, not asserted: the identity holds with the emissivity replaced by a constant.

    `test_sea_surface.py` leans on ε L + (1−ε) L = L, and that is exactly why it is blind here --
    the angular emissivity cancels. Swapping the real ε(θ) for a flat 0.5, which is wrong at every
    angle by between 0.49 and 0.02, leaves the identity satisfied to the same tolerance. So SE.1's
    acceptance is right to say the identity is preserved but is not the test.
    """
    _, lut, _ = rig
    sky, sea = _sea(rig)
    skin = sea.skin_temperature_k(0.0)
    constant = float(lut.lookup(skin, "lb"))
    sky.radiance = lambda t_s, elevation: np.full(np.shape(elevation), constant)  # type: ignore[method-assign]

    depression = np.radians(np.array([0.15, 0.5, 1.0, 5.0, 30.0, 60.0, 90.0]))
    truthful = np.asarray(lut.apparent_temperature(sea.surface_radiance(0.0, depression), "lb"))
    assert np.max(np.abs(truthful - skin)) < 1e-3

    real_eps = sea.emissivity(np.cos(np.radians(np.array([0.0, 55.0, 89.0]))))
    sea.emissivity = lambda cos_i: np.full(np.shape(cos_i), 0.5)  # type: ignore[method-assign]
    wrecked = np.asarray(lut.apparent_temperature(sea.surface_radiance(0.0, depression), "lb"))
    assert np.max(np.abs(wrecked - skin)) < 1e-3, "the identity survives a badly wrong emissivity"
    assert np.max(np.abs(real_eps - 0.5)) > 0.4, "and the emissivity really was badly wrong"


# -- the frame carries its own answer ---------------------------------------------------------


def test_the_maritime_scene_carries_the_report_in_its_metadata(rig) -> None:  # type: ignore[no-untyped-def]
    """So a Tier 3 report can print it without knowing the geometry, and it cannot drift."""
    materials = MaterialTable.from_library(MaterialLibrary.load(), "lwir")
    sky, sea = _sea(rig)
    scene = build_maritime_gbuffer(
        _sensor(width=64, height=48).sensor, sky, sea, materials, boresight_elevation_deg=-1.0
    )
    stored = scene.metadata["envelope"]
    assert stored == scene.envelope_report(), "the stored report must equal the recomputed one"
    assert stored.n_samples == int(scene.sea_mask.sum()) > 0
    assert stored.fraction_beyond == 1.0
    assert stored.envelope_range_m == pytest.approx(31.1, rel=1e-2)
    assert f"{VALIDATED_ZENITH_DEG:.0f} deg from nadir" in stored.summary()

    # the mask excludes vessels, so a report never describes a ray that hit a hull
    assert not np.any(scene.sea_mask & ~np.asarray(scene.planes["sky_mask"], dtype=bool))


def test_the_sea_model_flags_its_own_angles(rig) -> None:  # type: ignore[no-untyped-def]
    """`SeaModel.beyond_envelope` is the flag on the model, not only on a frame."""
    _, sea = _sea(rig)
    depression = np.radians(np.array([0.15, 1.0, 20.0, 40.0, 41.0, 60.0, 90.0]))
    flagged = sea.beyond_envelope(depression)
    assert list(map(bool, flagged)) == [True, True, True, True, False, False, False]
    assert np.allclose(
        np.asarray(sea.view_zenith_rad(depression)),
        np.asarray(view_zenith_rad(sea.camera_height_m, depression)),
    )
    report = sea.envelope_report(depression)
    assert report.n_samples == 7
    assert report.n_beyond == 4
