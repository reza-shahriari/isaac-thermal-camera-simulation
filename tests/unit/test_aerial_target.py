"""MS.7 -- the four-material aerial library and the target thermal signature.

Covers the CLAUDE.md #4 library walk over the aerial materials, the prescribed motor / ESC /
battery / airframe schedules against their analytic law, the warm-belly / cold-top ordering under
a clear sky, and the fully specified horizon test: where a 300 K drone's contrast against the sky
changes sign, by two independent routes.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from irsim.atmosphere import LayeredAtmosphere, SkyModel, load_atmosphere_preset
from irsim.config.bands import BAND_IDS
from irsim.config.environment import load_environment_preset
from irsim.materials import MaterialTable
from irsim.materials.library import CLOSURE_TOL, MaterialLibrary
from irsim.pipeline.environment import environment_radiance
from irsim.pipeline.radiance import band_radiance
from irsim.thermal import WeatherSample, WeatherSeries
from irsim.thermal.aerial import (
    AERIAL_HEAT_SOURCES,
    BATTERY,
    ESC,
    MOTOR,
    HeatSource,
    airframe_solver,
    heat_source_solver,
    node_temperature,
    refine_nodes,
    throttle_profile,
)
from irsim.validation.aerial import (
    AerialTarget,
    apparent_target_radiance,
    background_radiance,
    target_contrast,
    zero_contrast_elevation,
)

AERIAL_MATERIALS = (
    "painted_composite",
    "carbon_fibre",
    "aircraft_aluminium_painted",
    "propeller_rubber",
)

# The horizon test of the roadmap's MS.7 row, fully specified.
T_AIR_K = 300.0
TARGET_EPS = 0.9
TARGET_RANGE_M = 1000.0
TARGET_T_K = 300.0


def _weather(t_air: float = T_AIR_K, cloud: float = 0.0, rh: float = 0.3) -> WeatherSeries:
    return WeatherSeries.constant(
        WeatherSample(t_air, rh, 1.0, cloud, 0.0, 0.0, 23000.0, 0.0), 3600.0
    )


def _sky(lut, weather: WeatherSeries, env: str = "clear_dry") -> SkyModel:  # type: ignore[no-untyped-def]
    atm = LayeredAtmosphere(load_atmosphere_preset("us_standard_clear"), weather, {"lwir": lut})
    return SkyModel(atm, load_environment_preset(env), "lwir", lut)


# -- the library -------------------------------------------------------------------------


def test_aerial_library_closes_in_every_band() -> None:
    """CLAUDE.md #4 over the four aerial materials: ε + ρ + τ = 1 to 1e-6, every band."""
    lib = MaterialLibrary.load()
    assert set(lib.names) >= set(AERIAL_MATERIALS)
    for name in AERIAL_MATERIALS:
        material = lib[name]
        assert material.spec.source == "literature"
        for band in BAND_IDS:
            props = material.band_properties(band)
            closure = props.emissivity + props.reflectance + props.transmittance
            assert abs(closure - 1.0) < CLOSURE_TOL, (name, band, closure)
            assert props.authored == "emissivity", (name, "author ε, derive ρ (CLAUDE.md #4)")
            assert props.transmittance == 0.0, (name, band, "opaque airframe")
            assert 0.0 <= props.reflectance <= 1.0


def test_aerial_library_phenomenology_is_ordered() -> None:
    """The orderings that make these four worth having, not just their numbers."""
    lib = MaterialLibrary.load()
    lwir = {n: lib[n].band_properties("lwir").emissivity for n in AERIAL_MATERIALS}
    # Paint and rubber are near-blackbodies; the same aluminium unpainted is a mirror.
    assert lwir["propeller_rubber"] > lwir["painted_composite"] > 0.9
    # 1e-9, not 1e-12: since XD.14 this LWIR value is a curve's band integral held at the
    # authored 0.90 (ADR 0179), exact to quadrature and nine written decimals, not a typed number.
    assert lib["aircraft_aluminium_painted"].band_properties("lwir").emissivity == pytest.approx(
        0.90, abs=1e-9
    )
    assert lib["bare_aluminium"].band_properties("lwir").emissivity == pytest.approx(
        0.09, abs=1e-12
    )
    # Carbon fibre is the short-wave absorber -- it is the part that runs hot in sunlight.
    alpha = {n: lib[n].spec.thermal.solar_absorptivity for n in AERIAL_MATERIALS}
    assert alpha["carbon_fibre"] > alpha["painted_composite"]
    assert alpha["carbon_fibre"] > alpha["aircraft_aluminium_painted"]
    # Thermally the painted skin is a heat spreader; the composite shell is an insulator.
    k = {n: lib[n].spec.thermal.conductivity_w_mk for n in AERIAL_MATERIALS}
    assert k["aircraft_aluminium_painted"] > 100.0 > k["painted_composite"]


def test_aerial_materials_pack_into_a_material_table(tophat_lwir_lut) -> None:  # type: ignore[no-untyped-def]
    """M7.18: the four pack to float32 ε/ρ/τ by id, closed, with their thermal columns."""
    lib = MaterialLibrary.load()
    table = MaterialTable.from_library(lib, "lwir")
    assert table.emissivity.dtype == np.float32
    for name in AERIAL_MATERIALS:
        i = table.id_for(name)
        props = lib[name].band_properties("lwir")
        assert float(table.emissivity[i]) == pytest.approx(props.emissivity, abs=1e-6), name
        assert float(table.reflectance[i]) == pytest.approx(props.reflectance, abs=1e-6), name
        assert float(table.transmittance[i]) == 0.0, name
        # closure must survive the float32 packing, not only the float64 library (#4)
        closure = float(table.emissivity[i] + table.reflectance[i] + table.transmittance[i])
        assert abs(closure - 1.0) < CLOSURE_TOL, (name, closure)
        assert float(table.thermal["density_kg_m3"][i]) == pytest.approx(
            lib[name].spec.thermal.density_kg_m3, rel=1e-6
        )


# -- heat schedules ----------------------------------------------------------------------


def test_throttle_law_is_quadratic_and_ordered() -> None:
    """ΔT ∝ u²: dissipation is ohmic with current ∝ throttle (ADR 0072, ESTIMATED)."""
    for source in (MOTOR, ESC, BATTERY):
        assert float(source.delta_t_k(0.0)) == 0.0
        assert float(source.delta_t_k(1.0)) == pytest.approx(source.delta_t_max_k, rel=1e-12)
        # a quadratic is exactly a quarter of its full-scale value at half throttle
        assert float(source.delta_t_k(0.5)) == pytest.approx(0.25 * source.delta_t_max_k, rel=1e-12)
    assert MOTOR.delta_t_max_k > ESC.delta_t_max_k > BATTERY.delta_t_max_k
    assert set(AERIAL_HEAT_SOURCES) == {"motor", "esc", "battery"}
    with pytest.raises(ValueError):
        MOTOR.delta_t_k(1.5)


@pytest.mark.parametrize("source", [MOTOR, ESC, BATTERY])
def test_prescribed_schedule_matches_the_analytic_law_to_1mk(source: HeatSource) -> None:
    """The MS.7 criterion: the prescribed schedule *is* T_air + ΔT_max u², to 1 mK, everywhere
    between its nodes as well as on them."""
    weather = _weather()
    times = [0.0, 60.0, 120.0, 300.0, 600.0]
    throttle = [0.0, 0.35, 0.95, 0.60, 0.10]
    solver = heat_source_solver(source, weather, times, throttle, tolerance_k=1e-3)
    law = node_temperature(source, weather, throttle_profile(times, throttle))

    probe = np.linspace(times[0], times[-1], 4001)
    got = np.array([solver.advance(float(t), 0.0) for t in probe])
    err_mk = np.abs(got - law(probe)) * 1e3
    assert err_mk.max() < 1.0, f"{source.name}: {err_mk.max():.3f} mK exceeds the 1 mK schedule tol"
    # and it is a real signature, not a flat line
    assert got.max() - got.min() > 0.5 * source.delta_t_max_k


def test_refinement_is_driven_by_the_tolerance() -> None:
    """A tighter tolerance buys more nodes and a smaller error -- the grid is derived, not typed."""
    weather = _weather()
    times = [0.0, 600.0]
    throttle = [0.0, 1.0]
    law = node_temperature(MOTOR, weather, throttle_profile(times, throttle))
    coarse = refine_nodes(law, times, tolerance_k=1.0)
    fine = refine_nodes(law, times, tolerance_k=1e-3)
    assert fine.size > coarse.size > 2

    def worst(nodes: np.ndarray) -> float:
        probe = np.linspace(times[0], times[-1], 5000)
        return float(np.abs(np.interp(probe, nodes, law(nodes)) - law(probe)).max())

    assert worst(coarse) < 1.0
    assert worst(fine) < 1e-3 < worst(coarse)


def test_airframe_follows_the_shared_weather() -> None:
    """Airframe near T_air (CLAUDE.md #6: T_air comes from the injected series, not a file)."""
    weather = WeatherSeries.from_arrays(
        _weather().epoch_utc,
        np.array([0.0, 1800.0, 3600.0]),
        t_air_k=np.array([295.0, 300.0, 291.0]),
        rh_fraction=np.full(3, 0.3),
        wind_speed_m_s=np.full(3, 1.0),
        cloud_fraction=np.zeros(3),
        dni_w_m2=np.zeros(3),
        dhi_w_m2=np.zeros(3),
        visibility_m=np.full(3, 23000.0),
        precip_mm_h=np.zeros(3),
    )
    solver = airframe_solver(weather)
    for t in (0.0, 900.0, 1800.0, 2700.0, 3600.0):
        expected = float(weather.at(t).t_air_k)
        assert abs(solver.advance(t, 0.0) - expected) * 1e3 < 1.0

    warm = airframe_solver(weather, offset_k=2.5)
    assert warm.advance(900.0, 0.0) - solver.advance(900.0, 0.0) == pytest.approx(2.5, abs=1e-9)


# -- reflected environment: belly vs top ---------------------------------------------------


def test_belly_reads_warmer_than_top_under_a_clear_sky(tophat_lwir_lut) -> None:  # type: ignore[no-untyped-def]
    """Same skin temperature, opposite sky view: the belly reflects warm ground, the top reflects
    a cold sky, so the belly is brighter -- and the gap widens as emissivity falls."""
    lut = tophat_lwir_lut
    sky = _sky(lut, _weather())
    lib = MaterialLibrary.load()
    t_skin = np.full((1, 2), T_AIR_K, np.float32)
    v_s = np.array([[0.0, 1.0]], np.float64)  # belly sees ground, top sees sky
    l_env = environment_radiance(sky, lut, 0.0, v_s, sky.quantity)
    assert l_env[0, 0] > l_env[0, 1], "ground is warmer than a clear sky in LWIR"

    gaps = {}
    for name in (*AERIAL_MATERIALS, "bare_aluminium"):
        eps = lib[name].band_properties("lwir").emissivity
        table = MaterialTable.from_mapping({1: eps}, "lwir")
        radiance = band_radiance(
            t_skin, np.ones((1, 2), np.int32), table, lut, sky.quantity, l_env=l_env
        )
        t_app = lut.apparent_temperature(radiance.astype(np.float64))
        assert t_app[0, 0] > t_app[0, 1], f"{name}: belly should read warmer than the top"
        gaps[name] = float(t_app[0, 0] - t_app[0, 1])

    # a mirror shows the environment; a near-blackbody shows itself
    assert gaps["bare_aluminium"] > gaps["painted_composite"] > gaps["propeller_rubber"]
    assert gaps["propeller_rubber"] > 0.0


# -- the horizon test ---------------------------------------------------------------------


def _rendered_contrast(sky: SkyModel, lut, elevation_rad: float) -> float:  # type: ignore[no-untyped-def]
    """Route B: contrast assembled by the pipeline stages rather than the closed form.

    Stage 1 (``band_radiance`` with the M7.13 reflected term) then stage 2 (the layered
    atmosphere's ``apply`` over the slant path) on a two-pixel G-buffer: target, then sky. The
    sky pixel carries the apparent sky temperature and gets no atmosphere -- L_sky is already the
    whole column (ADR 0050).
    """
    table = MaterialTable.from_mapping({1: TARGET_EPS}, "lwir")
    v_s = np.array([[1.0]], np.float64)
    l_env = environment_radiance(sky, lut, 0.0, v_s, sky.quantity)
    leaving = band_radiance(
        np.full((1, 1), TARGET_T_K, np.float32),
        np.ones((1, 1), np.int32),
        table,
        lut,
        sky.quantity,
        l_env=l_env,
    )
    at_camera = sky.atmosphere.apply(
        "lwir", 0.0, leaving.astype(np.float64), TARGET_RANGE_M, elevation_rad, sky.quantity
    )
    return float(at_camera[0, 0]) - background_radiance(sky, 0.0, elevation_rad)


def test_drone_is_bright_at_zenith_and_inverts_near_the_horizon(tophat_lwir_lut) -> None:  # type: ignore[no-untyped-def]
    """T_air = 300 K, ε = 0.9, V_s = 1, R = 1 km, the clear preset."""
    lut = tophat_lwir_lut
    sky = _sky(lut, _weather())
    target = AerialTarget(TARGET_T_K, TARGET_EPS, TARGET_RANGE_M, sky_view_factor=1.0)

    zenith = target_contrast(target, sky, 0.0, math.radians(90.0))
    assert zenith > 0.0, "a 300 K drone must be brighter than a clear zenith sky"
    # in apparent temperature that is a large, unmissable signal
    t_app_target = float(
        lut.apparent_temperature(
            np.float64(apparent_target_radiance(target, sky, 0.0, math.radians(90.0)))
        )[()]
    )
    t_app_sky = float(
        lut.apparent_temperature(np.float64(background_radiance(sky, 0.0, math.radians(90.0))))[()]
    )
    assert t_app_target - t_app_sky > 50.0

    assert target_contrast(target, sky, 0.0, math.radians(0.5)) < 0.0, (
        "near the horizon the reflected cold sky must invert the contrast"
    )
    # exactly one crossing on a 1-degree grid
    grid = np.arange(0.5, 90.01, 1.0)
    signs = np.sign([target_contrast(target, sky, 0.0, math.radians(d)) for d in grid])
    assert int(np.count_nonzero(np.diff(signs) != 0)) == 1


def test_zero_contrast_elevation_matches_the_rendered_root_within_one_degree(  # type: ignore[no-untyped-def]
    tophat_lwir_lut,
) -> None:
    """The MS.7 criterion. Route A is the analytic root of
    τ(R,θ)[ε L_B(300) + (1−ε) L_env] + L_path(R,θ) = L_sky(θ); route B bisects the contrast that
    the stage-1 + stage-2 pipeline actually renders. They must agree within 1°."""
    lut = tophat_lwir_lut
    sky = _sky(lut, _weather())
    target = AerialTarget(TARGET_T_K, TARGET_EPS, TARGET_RANGE_M, sky_view_factor=1.0)

    analytic_deg = zero_contrast_elevation(target, sky, 0.0)
    assert analytic_deg is not None, "the specified case must have a crossing"

    lo, hi = 0.5, 90.0
    assert (
        _rendered_contrast(sky, lut, math.radians(lo))
        < 0.0
        < _rendered_contrast(sky, lut, math.radians(hi))
    )
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if _rendered_contrast(sky, lut, math.radians(mid)) < 0.0:
            lo = mid
        else:
            hi = mid
    rendered_deg = 0.5 * (lo + hi)

    assert abs(analytic_deg - rendered_deg) < 1.0, (
        f"analytic root {analytic_deg:.3f}° vs rendered {rendered_deg:.3f}° "
        "-- the closed form and the pipeline disagree"
    )
    # at the root the target is radiometrically indistinguishable from the sky behind it
    residual = target_contrast(target, sky, 0.0, math.radians(analytic_deg))
    t_sky = float(
        lut.apparent_temperature(
            np.float64(background_radiance(sky, 0.0, math.radians(analytic_deg)))
        )[()]
    )
    t_tgt = float(
        lut.apparent_temperature(
            np.float64(apparent_target_radiance(target, sky, 0.0, math.radians(analytic_deg)))
        )[()]
    )
    assert abs(t_tgt - t_sky) * 1e3 < 1.0, f"{residual:g} W/m2/sr residual at the root"


def test_the_inversion_is_the_reflected_term(tophat_lwir_lut) -> None:  # type: ignore[no-untyped-def]
    """A blackbody target at air temperature never inverts: with ε = 1 there is no reflected sky
    to drag it below the horizon sky, so the crossing must disappear. This is what distinguishes
    the modelled mechanism from a path-radiance artefact, and it fixes the ordering: the lower the
    emissivity, the higher the elevation at which the target vanishes."""
    lut = tophat_lwir_lut
    sky = _sky(lut, _weather())
    roots = {}
    for eps in (0.80, 0.90, 0.95):
        root = zero_contrast_elevation(AerialTarget(TARGET_T_K, eps, TARGET_RANGE_M, 1.0), sky, 0.0)
        assert root is not None
        roots[eps] = root
    assert roots[0.80] > roots[0.90] > roots[0.95] > 0.0

    black = AerialTarget(TARGET_T_K, 1.0, TARGET_RANGE_M, 1.0)
    assert zero_contrast_elevation(black, sky, 0.0) is None
    for deg in (0.5, 1.0, 5.0, 90.0):
        assert target_contrast(black, sky, 0.0, math.radians(deg)) > 0.0


def test_a_hot_target_never_crosses(tophat_lwir_lut) -> None:  # type: ignore[no-untyped-def]
    """A motor-warm target is brighter than the sky everywhere -- the helper says so with None
    rather than inventing a root."""
    lut = tophat_lwir_lut
    sky = _sky(lut, _weather())
    hot = AerialTarget(T_AIR_K + float(MOTOR.delta_t_k(1.0)), TARGET_EPS, TARGET_RANGE_M, 1.0)
    assert zero_contrast_elevation(hot, sky, 0.0) is None
    assert target_contrast(hot, sky, 0.0, math.radians(0.5)) > 0.0
