"""M11.8 — Tier 3 multi-band phenomenology, and the architecture claim that makes it possible.

Three bands are configured, and the point of this bench is that they behave *differently* for
reasons that come out of the data rather than out of branches in the code:

* at noon, a glass facet's MWIR **glint** saturates a 14-bit ADC by four orders of magnitude while
  the same facet in LWIR sits comfortably inside the radiometric range;
* at 03:00 with no sun anywhere, MWIR and LWIR **agree** about which facet is hotter — both are
  reading self-emission, and the rank correlation across the scene is the evidence;
* in SWIR at night the scene is *lit*, not glowing, so the signal is linear in the airglow level
  and vanishes with it.

And the architecture assertion: **the band kernel has not changed since the LWIR band was the only
one there was.** Adding SWIR and MWIR touched YAML, CSV and new source modules; `planck.py`,
`band_integration.py`, `band_average.py`, `lut.py`, `lut_files.py`, `encoding.py` and `band.py` are
byte-for-byte what M1.11 left.

The facet scene is **prescribed**, not solved: M6's surface energy balance is a later step, so the
noon and 03:00 temperatures below are authored inputs chosen to be representative. The *orderings*
tested here are properties of the radiometry and the illumination, not of those numbers, and the
tests say so where a number could have set the answer.

docs/physics-model.md §15 T3, §5.2, §12.1, §16.4 step 10
"""

from __future__ import annotations

import math
import pathlib
import subprocess

import numpy as np
import pytest

from irsim.atmosphere.layered import LayeredAtmosphere
from irsim.atmosphere.library import load_atmosphere_preset
from irsim.atmosphere.sky import SkyModel
from irsim.config.environment import load_environment_preset
from irsim.config.loader import load_sensor_config
from irsim.detector.netd import mean_signal
from irsim.detector.params import fpa_params_from_config
from irsim.materials.library import MaterialLibrary
from irsim.materials.lobe import mirror_direction, specular_weight
from irsim.materials.table import MaterialTable
from irsim.noise.electron import electron_budget
from irsim.pipeline.night import NightIllumination
from irsim.pipeline.solar import SolarIllumination, beam_transmittance
from irsim.pipeline.specular import specular_glint_radiance, specular_reflected_radiance
from irsim.radiometry.lut_files import load_band_lut_for_config
from irsim.thermal.weather import WeatherSample, WeatherSeries

# GT.1: this whole module is the slow tier -- a validation bench or an end-to-end frame rather
# than a unit test. `make test` skips it; `make test-slow` and `make check` run it.
pytestmark = pytest.mark.slow

REPO = pathlib.Path(__file__).resolve().parents[2]
DATA = REPO / "data"
UP = np.array([0.0, 0.0, 1.0])

BANDS = {
    "lwir": "flir_boson_640_lwir",
    "swir": "example_swir_ingaas_640",
    "mwir": "example_mwir_insb_640",
}

#: (material, T at 12:00, T at 03:00). PRESCRIBED -- M6's solver is a later step. Representative
#: of a car in a car park on a clear day: dark paint runs hottest at noon and coldest at night,
#: glass tracks the cabin, asphalt holds heat, skin is regulated.
FACETS: tuple[tuple[str, float, float], ...] = (
    ("car_paint_black", 335.0, 281.0),
    ("car_paint_white", 313.0, 282.0),
    ("glass_windshield", 322.0, 285.0),
    ("asphalt_dry", 331.0, 289.0),
    ("bare_aluminium", 318.0, 283.0),
    ("human_skin", 307.0, 306.0),
)

WEATHER = WeatherSample(
    t_air_k=298.0,
    rh_fraction=0.35,
    wind_speed_m_s=1.5,
    cloud_fraction=0.0,
    dni_w_m2=0.0,
    dhi_w_m2=0.0,
    visibility_m=23000.0,
    precip_mm_h=0.0,
)
NOON_SUN_ELEVATION_DEG = 65.0


@pytest.fixture(scope="module")
def library() -> MaterialLibrary:
    return MaterialLibrary.load(REPO / "configs" / "materials")


@pytest.fixture(scope="module")
def bands():  # type: ignore[no-untyped-def]
    out = {}
    for band, name in BANDS.items():
        cfg = load_sensor_config(REPO / "configs/sensors" / f"{name}.yaml", DATA)
        out[band] = (cfg, load_band_lut_for_config(cfg, DATA / "lut", DATA))
    return out


@pytest.fixture(scope="module")
def tables(library, bands):  # type: ignore[no-untyped-def]
    return {band: MaterialTable.from_library(library, band, form="energy") for band in bands}


def _sky(band: str, lut):  # type: ignore[no-untyped-def]
    weather = WeatherSeries.constant(WEATHER, 3600.0)
    atmosphere = LayeredAtmosphere(
        load_atmosphere_preset("us_standard_clear"), weather, {band: lut}
    )
    return SkyModel(atmosphere, load_environment_preset("clear_dry"), band, lut)


def _facet_radiance(
    band: str, cfg, lut, table: MaterialTable, name: str, temperature_k: float, sun_elevation: float
) -> float:
    """L leaving one upward-facing facet: ε L_B(T) + the lobe's reflected sky + the glint."""
    ids = np.array([[table.id_for(name)]], dtype=np.int32)
    eps, rho, _tau = table.properties_for(ids)
    roughness = float(np.asarray(table.roughness)[table.id_for(name)])
    sun = np.array(
        [math.cos(math.radians(sun_elevation)), 0.0, math.sin(math.radians(sun_elevation))]
    )
    view = mirror_direction(sun, UP) if sun_elevation > 0.0 else np.array([0.0, 0.0, 1.0])
    sky = _sky(band, lut)
    emitted = float(eps[0, 0]) * float(lut.lookup(np.float64(temperature_k))[()])
    reflected = specular_reflected_radiance(
        float(rho[0, 0]), sky, 0.0, view, UP, roughness, n_theta=24, n_phi=48
    )
    glint = 0.0
    if sun_elevation > 0.0:
        solar = SolarIllumination.for_sensor(cfg, "lb", DATA)
        tau_sun = float(
            beam_transmittance(load_atmosphere_preset("us_standard_clear"), band, sun_elevation)
        )
        glint = specular_glint_radiance(
            float(rho[0, 0]),
            view,
            UP,
            sun,
            roughness,
            solar.e_band,
            tau_sun=tau_sun,
            n_theta=24,
            n_phi=48,
        )
    return emitted + reflected + glint


def _dn_fraction(cfg, lut, radiance: float) -> float:
    """Where this radiance lands on the ADC, as a fraction of full scale."""
    params = fpa_params_from_config(cfg)
    spec = cfg.sensor
    from irsim.optics.aperture import aperture_factor

    geometry = (
        params.active_area_m2 * aperture_factor(spec.optics.f_number) * spec.optics.transmittance
    )
    if params.type == "photon":
        photon_energy = 1.986445857e-25 / (
            0.5 * (spec.band.lambda_min_um + spec.band.lambda_max_um) * 1e-6
        )
        n_e = (
            params.quantum_efficiency
            * params.integration_time_s
            * geometry
            * (radiance / photon_energy)
        )
        return n_e / params.well_capacity_e
    # a bolometer: where the radiance sits in the ADR 0021 radiometric span
    lo = float(lut.lookup(233.15)[()])
    hi = float(lut.lookup(473.15)[()])
    return (radiance - lo) / (hi - lo)


# ---------------------------------------------------------------------------------------------
# noon: the MWIR glint saturates and the LWIR does not
# ---------------------------------------------------------------------------------------------


#: How far off the exact specular direction the camera sits. A perfect alignment is a measure-zero
#: coincidence; two degrees is "near enough that a driver would see the reflection", and it is the
#: geometry that separates the bands. At **exact** alignment every band saturates, because a mirror
#: shows you the sun and the sun is 5778 K in all of them -- that is recorded below, because the
#: roadmap's wording ("MWIR saturates while LWIR does not") is true of a scene and not of a mirror.
OFF_SPECULAR_DEG = 2.0


def _glass_facet_fraction(band: str, bands, tables, off_specular_deg: float) -> float:  # type: ignore[no-untyped-def]
    """Full-scale fraction for a glass facet at noon, viewed this far off the specular direction."""
    cfg, lut = bands[band]
    table = tables[band]
    index = table.id_for("glass_windshield")
    ids = np.array([[index]], dtype=np.int32)
    eps, rho, _ = table.properties_for(ids)
    roughness = float(np.asarray(table.roughness)[index])
    sun = np.array(
        [
            math.cos(math.radians(NOON_SUN_ELEVATION_DEG)),
            0.0,
            math.sin(math.radians(NOON_SUN_ELEVATION_DEG)),
        ]
    )
    specular_el = math.degrees(math.asin(float(mirror_direction(sun, UP)[2])))
    el = math.radians(specular_el + off_specular_deg)
    view = np.array([-math.cos(el), 0.0, math.sin(el)])
    tau_sun = float(
        beam_transmittance(
            load_atmosphere_preset("us_standard_clear"), band, NOON_SUN_ELEVATION_DEG
        )
    )
    solar = SolarIllumination.for_sensor(cfg, "lb", DATA)
    emitted = float(eps[0, 0]) * float(lut.lookup(np.float64(322.0))[()])
    glint = specular_glint_radiance(
        float(rho[0, 0]),
        view,
        UP,
        sun,
        roughness,
        solar.e_band,
        tau_sun=tau_sun,
        n_theta=24,
        n_phi=48,
    )
    return _dn_fraction(cfg, lut, emitted + glint)


def test_the_mwir_glass_facet_saturates_at_noon_while_lwir_does_not(bands, tables) -> None:  # type: ignore[no-untyped-def]
    """Two degrees off specular: MWIR is seven times full scale, LWIR a sixth of it."""
    mwir = _glass_facet_fraction("mwir", bands, tables, OFF_SPECULAR_DEG)
    lwir = _glass_facet_fraction("lwir", bands, tables, OFF_SPECULAR_DEG)
    swir = _glass_facet_fraction("swir", bands, tables, OFF_SPECULAR_DEG)
    assert mwir > 1.0, f"MWIR glass did not saturate ({mwir:.3f})"
    assert lwir < 1.0, f"LWIR glass saturated ({lwir:.3f})"
    assert swir > mwir, "the shorter band should be further over"
    assert mwir == pytest.approx(6.97, rel=0.1)
    assert lwir == pytest.approx(0.168, rel=0.1)


def test_every_band_saturates_on_an_exact_mirror_and_that_is_correct(bands, tables) -> None:  # type: ignore[no-untyped-def]
    """The roadmap's wording is about a scene, not about a mirror, and the difference is physics.

    At **exact** specular alignment even LWIR goes 8x over full scale -- because a mirror shows
    you the sun, and the sun is a 5778 K blackbody in LWIR too (E_B/Omega_sun = 2.8e4 W/m2/sr,
    an apparent temperature off the top of the LUT). What separates the bands is not whether the
    glint saturates but **how wide the angular window is** in which it does.
    """
    for band in ("lwir", "mwir", "swir"):
        assert _glass_facet_fraction(band, bands, tables, 0.0) > 1.0, band


def test_the_saturation_window_widens_as_the_band_shortens(bands, tables) -> None:  # type: ignore[no-untyped-def]
    """The measurement that actually distinguishes the three bands, in degrees.

    A wider window means more pixels of a real scene are ruined by one glint, which is the
    operational consequence and the thing a detector trained on this imagery has to cope with.
    """
    widths = {}
    for band in ("lwir", "mwir", "swir"):
        # bisect rather than sweep: the fraction falls monotonically with the offset (the GGX
        # tail is monotone in the half-angle), so 24 evaluations beat 97 and are more precise.
        lo, hi = 0.0, 24.0
        if _glass_facet_fraction(band, bands, tables, hi) > 1.0:
            widths[band] = hi
            continue
        for _ in range(24):
            mid = 0.5 * (lo + hi)
            if _glass_facet_fraction(band, bands, tables, mid) > 1.0:
                lo = mid
            else:
                hi = mid
        widths[band] = lo
    assert widths["lwir"] < widths["mwir"] < widths["swir"], widths
    assert widths["lwir"] == pytest.approx(0.51, abs=0.15), widths
    assert widths["mwir"] == pytest.approx(3.72, abs=0.40), widths
    assert widths["swir"] == pytest.approx(18.6, abs=2.0), widths
    assert widths["swir"] / widths["lwir"] > 20.0, widths


def test_the_saturation_is_the_glint_and_not_the_temperature(bands, tables) -> None:  # type: ignore[no-untyped-def]
    """The same facet at the same temperature, out of the specular direction, is unremarkable."""
    cfg, lut = bands["mwir"]
    ids = np.array([[tables["mwir"].id_for("glass_windshield")]], dtype=np.int32)
    eps, rho, _ = tables["mwir"].properties_for(ids)
    sky = _sky("mwir", lut)
    off_axis_view = np.array([0.0, 0.0, 1.0])  # straight down at the facet
    quiet = float(eps[0, 0]) * float(lut.lookup(322.0)[()]) + specular_reflected_radiance(
        float(rho[0, 0]), sky, 0.0, off_axis_view, UP, 0.04, n_theta=24, n_phi=48
    )
    assert _dn_fraction(cfg, lut, quiet) < 1.0


def test_a_rough_facet_does_not_glint_even_at_noon(bands, tables) -> None:  # type: ignore[no-untyped-def]
    """Asphalt in the specular direction at noon: no glint. §4.3 says so, and (1−r)² agrees."""
    cfg, lut = bands["mwir"]
    rough = _facet_radiance(
        "mwir", cfg, lut, tables["mwir"], "asphalt_dry", 331.0, NOON_SUN_ELEVATION_DEG
    )
    smooth = _facet_radiance(
        "mwir", cfg, lut, tables["mwir"], "glass_windshield", 322.0, NOON_SUN_ELEVATION_DEG
    )
    assert _dn_fraction(cfg, lut, rough) < 1.0
    assert smooth / rough > 1e3
    assert float(specular_weight(0.75)) < 0.1


# ---------------------------------------------------------------------------------------------
# 03:00: MWIR and LWIR agree about what is hot
# ---------------------------------------------------------------------------------------------


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(np.argsort(values))
    return order.astype(np.float64)


def test_mwir_and_lwir_rank_the_night_scene_the_same_way(bands, tables) -> None:  # type: ignore[no-untyped-def]
    """Spearman > 0.9 with the sun below the horizon: both bands are reading self-emission."""
    readings = {}
    for band in ("lwir", "mwir"):
        cfg, lut = bands[band]
        temps = []
        for name, _noon, night in FACETS:
            radiance = (
                _facet_radiance("", cfg, lut, tables[band], name, night, -10.0)
                if False
                else _facet_radiance(band, cfg, lut, tables[band], name, night, -10.0)
            )
            temps.append(float(lut.apparent_temperature(np.asarray(radiance))[()]))
        readings[band] = np.asarray(temps)

    a, b = _rank(readings["lwir"]), _rank(readings["mwir"])
    rho = float(np.corrcoef(a, b)[0, 1])
    assert rho > 0.9, f"rank correlation {rho:.3f}\n{readings}"
    # and the agreement is not vacuous: the scene really does span a range in both bands
    for band, values in readings.items():
        assert float(values.max() - values.min()) > 5.0, (band, values)


def test_the_night_scene_disagrees_with_the_noon_scene(bands, tables) -> None:  # type: ignore[no-untyped-def]
    """The rank test above would pass on any two identical lists; this shows the ranks can move."""
    cfg, lut = bands["lwir"]
    noon = np.asarray(
        [
            _facet_radiance("lwir", cfg, lut, tables["lwir"], n, hot, NOON_SUN_ELEVATION_DEG)
            for n, hot, _ in FACETS
        ]
    )
    night = np.asarray(
        [_facet_radiance("lwir", cfg, lut, tables["lwir"], n, cold, -10.0) for n, _, cold in FACETS]
    )
    assert not np.array_equal(_rank(noon), _rank(night))


# ---------------------------------------------------------------------------------------------
# SWIR at night is lit, not glowing
# ---------------------------------------------------------------------------------------------


def _swir_night_signal_e(cfg, lut, level_w_m2: float, reflectance: float = 0.3) -> float:
    preset = load_environment_preset("clear_dry")
    night_spec = preset.night.model_copy(
        update={"airglow_irradiance_w_m2": level_w_m2, "airglow_irradiance_nw_cm2": None}
    )
    env = preset.model_copy(update={"night": night_spec})
    night = NightIllumination.for_sensor(cfg, env, "lb_q", DATA)
    l_night_q = float(night.incident_radiance(0.0, moon_elevation_deg=-10.0))
    params = fpa_params_from_config(cfg)
    from irsim.optics.aperture import aperture_factor

    geometry = (
        params.active_area_m2
        * aperture_factor(cfg.sensor.optics.f_number)
        * cfg.sensor.optics.transmittance
    )
    return (
        params.quantum_efficiency * params.integration_time_s * geometry * reflectance * l_night_q
    )


def test_the_swir_night_signal_is_linear_in_the_airglow_level(bands) -> None:  # type: ignore[no-untyped-def]
    """§5.5's range, 3.5 → 39 nW/cm², as a straight line through the origin."""
    cfg, lut = bands["swir"]
    levels = np.linspace(3.5e-5, 3.9e-4, 12)
    signal = np.asarray([_swir_night_signal_e(cfg, lut, float(x)) for x in levels])
    r = float(np.corrcoef(levels, signal)[0, 1])
    assert r > 0.999, r
    assert signal[0] > 0.0
    assert _swir_night_signal_e(cfg, lut, 0.0) == 0.0


def test_the_swir_night_snr_and_what_it_takes_to_reach_five(bands) -> None:  # type: ignore[no-untyped-def]
    """⚠️ The example camera does **not** reach SNR 5 from airglow: it reaches 0.32.

    That is not a modelling failure, it is what a 60 Hz, 120 e⁻ uncooled InGaAs core gives on a
    moonless night, and it is why night-capable SWIR parts are specified differently. The second
    half of this test builds the camera that *does* reach 5, and names the four changes it takes:
    a longer integration (16 -> 33 ms), a bigger pixel (15 -> 20 um), lower read noise
    (120 -> 25 e-) and a faster lens (F/1.4 -> F/1.0). Every one of them is needed; drop any and
    it falls back under 5. See ADR 0065 and the M11.8 roadmap row.
    """
    cfg, lut = bands["swir"]
    budget = electron_budget(cfg.sensor, lut)
    signal = _swir_night_signal_e(cfg, lut, 1.0e-4)
    sigma = math.sqrt(signal + budget.dark_electrons + budget.sigma_gaussian**2)
    snr = signal / sigma
    assert snr == pytest.approx(0.32, rel=0.25), f"example-camera SNR {snr:.3f}"
    assert snr < 1.0

    # the night-capable variant: 30 Hz / 33 ms, 20 um pixels, 25 e- read noise, cooled, F/1.0
    night_cfg = cfg.model_copy(
        update={
            "sensor": cfg.sensor.model_copy(
                update={
                    "optics": cfg.sensor.optics.model_copy(update={"f_number": 1.0}),
                    "fpa": cfg.sensor.fpa.model_copy(
                        update={
                            "frame_rate_hz": 30.0,
                            "integration_time_ms": 33.0,
                            "pitch_um": 20.0,
                            "read_noise_e": 25.0,
                            "fpa_temp_mode": "fixed",
                            "fpa_temp_k": 240.0,
                        }
                    ),
                }
            )
        }
    )
    night_budget = electron_budget(night_cfg.sensor, lut)
    night_signal = _swir_night_signal_e(night_cfg, lut, 1.0e-4)
    night_sigma = math.sqrt(
        night_signal + night_budget.dark_electrons + night_budget.sigma_gaussian**2
    )
    assert night_signal / night_sigma > 5.0, f"night-capable SNR {night_signal / night_sigma:.2f}"


def test_swir_at_night_is_reflection_and_not_emission(bands) -> None:  # type: ignore[no-untyped-def]
    """The regime claim, at the temperatures this scene actually has."""
    cfg, lut = bands["swir"]
    reflected = _swir_night_signal_e(cfg, lut, 1.0e-4)
    emitted = mean_signal(300.0, cfg.sensor, lut) * 0.7
    assert reflected / emitted > 10.0, f"reflected/emitted = {reflected / emitted:.2f}"


# ---------------------------------------------------------------------------------------------
# the architecture assertion
# ---------------------------------------------------------------------------------------------

M1_11_COMMIT = "3b8172b"  # test(radiometry): golden LUT slice -- the last M1 step
#: The band machinery itself. `constants.py` is excluded because CLAUDE.md explicitly directs new
#: physical constants there, and `spectral_response.py` because M5's MTF cascade needed one
#: addition to it (31b1f66); both are named here rather than quietly left out of the glob.
BAND_KERNEL_MODULES = (
    "planck.py",
    "band_integration.py",
    "band_average.py",
    "band.py",
    "lut.py",
    "lut_files.py",
    "encoding.py",
    "__init__.py",
)

#: Modules whose band kernel legitimately moved, with the commit that moved it and why. The guard
#: then measures from *that* commit, so the file stays covered against the **next** change -- which
#: is what an exclusion gives up. `constants.py` and `spectral_response.py` keep their own
#: treatment below for historical reasons; new exceptions belong here.
BASELINE_OVERRIDES: dict[str, tuple[str, str]] = {
    "band_integration.py": (
        "c974e25",
        "AT.24 changed `quadrature_grid`, the one function every band's integral goes through: "
        "the grid now ends on the response file's last sample with an even interval count, and "
        "steps no wider than 0.5 % of the band's short edge, because the padded odd-count grid "
        "integrated an exact 0.75-1.0 um top-hat 33 % high and the 0.01 um step sat 28 mK off a "
        "converged NIR quadrature. The kernel still names no band and branches on none; adding a "
        "band remains a config and a response file, which is the property the guard is for.",
    ),
    "lut_files.py": (
        "4de2305",
        "GT.11 added `builder_sha256` -- a hash of the source of `band_integration`, `planck` and "
        "`constants` written into every LUT sidecar and checked on load, so a bundle an older "
        "builder made is refused with the `make luts` remedy instead of feeding a golden. It "
        "names no band and branches on none (AT.2's `load_band_response_for_config`, the previous "
        "baseline, likewise read only the response path a config declares). Adding a band still "
        "needs no edit to this file, which is the property the guard is for.",
    ),
}


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout


@pytest.mark.parametrize("module", BAND_KERNEL_MODULES)
def test_the_band_kernel_has_not_changed_since_there_was_one_band(module: str) -> None:
    """Three bands are configured and `irsim.radiometry`'s kernel is what M1.11 left.

    Skipped outside a git checkout. The M11.1 AST guard is the version that always runs; this is
    the stronger statement it cannot make.
    """
    if not (REPO / ".git").exists():
        pytest.skip("not a git checkout")
    path = f"src/irsim/radiometry/{module}"
    baseline, _ = BASELINE_OVERRIDES.get(module, (M1_11_COMMIT, ""))
    log = _git("log", "--oneline", f"{baseline}..HEAD", "--", path).strip()
    since = "M1.11" if baseline == M1_11_COMMIT else f"its recorded baseline {baseline}"
    assert log == "", f"{path} changed after {since}:\n{log}"


def test_every_baseline_override_is_justified_and_real() -> None:
    """An override with no reason, or one pointing at a commit that never touched the file, is a
    silently widened guard -- the failure mode the exclusions below were already written against."""
    if not (REPO / ".git").exists():
        pytest.skip("not a git checkout")
    for module, (commit, reason) in BASELINE_OVERRIDES.items():
        assert module in BAND_KERNEL_MODULES, f"{module} is not a guarded module"
        assert len(reason) > 80, f"{module}'s override needs a reason, not a note"
        touched = _git(
            "log", "--oneline", f"{commit}~1..{commit}", "--", f"src/irsim/radiometry/{module}"
        ).strip()
        assert touched, f"{commit} does not touch {module}; the override is stale"


def test_the_two_excluded_modules_are_excluded_for_a_stated_reason() -> None:
    """A guard whose exclusions are unexplained is a guard that will be widened silently."""
    if not (REPO / ".git").exists():
        pytest.skip("not a git checkout")
    constants = _git(
        "log", "--oneline", f"{M1_11_COMMIT}..HEAD", "--", "src/irsim/radiometry/constants.py"
    ).strip()
    response = _git(
        "log",
        "--oneline",
        f"{M1_11_COMMIT}..HEAD",
        "--",
        "src/irsim/radiometry/spectral_response.py",
    ).strip()
    assert constants, "constants.py was expected to have grown"
    assert len(response.splitlines()) == 1, response


def test_three_bands_are_configured_and_classify_distinctly(bands) -> None:  # type: ignore[no-untyped-def]
    ids = {band: cfg.sensor.band.band_id for band, (cfg, _) in bands.items()}
    assert ids == {"lwir": "lwir", "swir": "swir", "mwir": "mwir"}
    regimes = {band: cfg.sensor.band.regime for band, (cfg, _) in bands.items()}
    assert regimes == {"lwir": "emissive", "swir": "reflective", "mwir": "mixed"}
