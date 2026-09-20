"""M6.9 / M6.10 — the same balance over many facets, and where their initial condition comes from.

A scene has thousands of surfaces and the scalar solver is the oracle, not the implementation. So
the vectorised form is held to **0.1 mK** against N separate scalar runs — not to "close enough",
because the two are meant to be the same arithmetic in a different loop order.

The second half is the part that is easy to skip and expensive to get wrong. A surface temperature
is a **memory**: asphalt at 06:00 is carrying yesterday afternoon. Starting a scene at the air
temperature is starting it wrong by several kelvin and staying wrong for hours — through exactly
the part of the diurnal cycle a thermal camera is most interesting in.

docs/physics-model.md §6.4, §15 T1; ADR 0036, ADR 0037
"""

from __future__ import annotations

import math
import pathlib

import numpy as np
import pytest

from irsim.thermal.balance import (
    SurfaceForcing,
    ThermalProperties,
    rk2_step,
    steady_state_temperature,
)
from irsim.thermal.facets import (
    DEFAULT_SPIN_UP_HOURS,
    FacetForcing,
    FacetProperties,
    FacetSolver,
    SpinUpCache,
    spin_up,
    spin_up_key,
)

REPO = pathlib.Path(__file__).resolve().parents[2]
DT_S = 30.0


def _random_facets(n: int, seed: int = 7) -> tuple[FacetProperties, list[ThermalProperties]]:
    rng = np.random.default_rng(seed)
    scalars = [
        ThermalProperties(
            heat_capacity_j_m2_k=float(rng.uniform(2_000.0, 200_000.0)),
            emissivity=float(rng.uniform(0.05, 0.99)),
            solar_absorptivity=float(rng.uniform(0.1, 0.95)),
        )
        for _ in range(n)
    ]
    return FacetProperties.stack(scalars), scalars


def _forcing(hour: float, shadow: np.ndarray | float = 1.0) -> FacetForcing:
    sun = max(0.0, 900.0 * math.sin(math.pi * (hour - 6.0) / 12.0))
    return FacetForcing(
        t_air_k=288.0 + 6.0 * math.sin(math.pi * (hour - 8.0) / 12.0),
        h_w_m2_k=10.0,
        q_solar_w_m2=sun * np.asarray(shadow),
        q_longwave_down_w_m2=320.0,
    )


# ---------------------------------------------------------------------------------------------
# M6.9: the vectorised solver is the scalar one
# ---------------------------------------------------------------------------------------------


def test_sixty_four_facets_equal_sixty_four_scalar_runs() -> None:
    """Over 6 h, to **0.1 mK**. The two are the same arithmetic in a different loop order."""
    packed, scalars = _random_facets(64)
    solver = FacetSolver(packed, 288.0)
    scalar_state = [288.0] * len(scalars)
    steps = int(6 * 3600 / DT_S)
    for i in range(steps):
        hour = 6.0 + i * DT_S / 3600.0
        forcing = _forcing(hour)
        solver.advance(forcing, DT_S)
        scalar_forcing = SurfaceForcing(
            t_air_k=float(np.asarray(forcing.t_air_k)),
            h_w_m2_k=float(np.asarray(forcing.h_w_m2_k)),
            q_solar_w_m2=float(np.asarray(forcing.q_solar_w_m2)),
            q_longwave_down_w_m2=float(np.asarray(forcing.q_longwave_down_w_m2)),
        )
        scalar_state = [
            float(rk2_step(t, DT_S, p, scalar_forcing))
            for t, p in zip(scalar_state, scalars, strict=True)
        ]
    assert float(np.max(np.abs(solver.temperatures_k - np.asarray(scalar_state)))) < 1e-4


def test_shadowing_one_facet_changes_only_that_facet() -> None:
    """The check that the vectorisation has not accidentally coupled anything."""
    packed, _ = _random_facets(16, seed=3)
    shadow = np.ones(16)
    shadow[5] = 0.0
    lit = FacetSolver(packed, 290.0)
    shaded = FacetSolver(packed, 290.0)
    for i in range(int(3 * 3600 / DT_S)):
        hour = 9.0 + i * DT_S / 3600.0
        lit.advance(_forcing(hour), DT_S)
        shaded.advance(_forcing(hour, shadow), DT_S)
    difference = np.abs(lit.temperatures_k - shaded.temperatures_k)
    assert difference[5] > 1.0, "the shadowed facet did not cool"
    assert float(np.max(np.delete(difference, 5))) == 0.0, "a neighbour moved"


def test_float16_is_refused_and_float32_is_only_the_boundary() -> None:
    packed, _ = _random_facets(4)
    with pytest.raises(TypeError, match="float16"):
        FacetSolver(packed, np.full(4, 290.0, dtype=np.float16))
    solver = FacetSolver(packed, 290.0)
    assert solver.temperatures_k.dtype == np.float64
    assert solver.as_float32().dtype == np.float32
    with pytest.raises(TypeError, match="float16"):
        solver.advance(FacetForcing(t_air_k=np.float16(288.0), h_w_m2_k=10.0), DT_S)


def test_shape_and_range_errors_are_refused() -> None:
    packed, _ = _random_facets(4)
    with pytest.raises(ValueError, match="expected"):
        FacetSolver(packed, np.full(3, 290.0))
    solver = FacetSolver(packed, 290.0)
    with pytest.raises(ValueError, match="expected"):
        solver.advance(FacetForcing(t_air_k=np.full(3, 288.0), h_w_m2_k=10.0), DT_S)
    with pytest.raises(ValueError, match="dt_s must be positive"):
        solver.advance(FacetForcing(t_air_k=288.0, h_w_m2_k=10.0), 0.0)
    with pytest.raises(ValueError, match="disagree on shape"):
        FacetProperties(
            heat_capacity_j_m2_k=np.ones(3),
            emissivity=np.ones(4) * 0.9,
            solar_absorptivity=np.ones(3) * 0.5,
        )


def test_scalars_broadcast_and_arrays_are_per_facet() -> None:
    packed, _ = _random_facets(8)
    # A 600 s step at h = 50 is past §6.4's explicit bound for these thin facets (TC.1 now
    # refuses it); broadcasting is what is under test here, so the guard is switched off.
    solver = FacetSolver(packed, 290.0, guard=False)
    per_facet = np.linspace(280.0, 300.0, 8)
    solver.advance(FacetForcing(t_air_k=per_facet, h_w_m2_k=50.0), 600.0)
    warmed = solver.temperatures_k
    assert warmed[-1] > warmed[0], "a hotter air temperature must warm its own facet more"


# ---------------------------------------------------------------------------------------------
# M6.10: spin-up
# ---------------------------------------------------------------------------------------------


def test_constant_weather_spins_up_to_the_analytic_equilibrium() -> None:
    """The one case with an independent answer: < 0.1 K against M6.7's root."""
    packed, scalars = _random_facets(12, seed=11)
    forcing = FacetForcing(
        t_air_k=295.0, h_w_m2_k=10.0, q_solar_w_m2=500.0, q_longwave_down_w_m2=330.0
    )
    result = spin_up(packed, lambda _t: forcing, "constant", 0.0, hours=48.0, dt_s=60.0)
    expected = [
        steady_state_temperature(
            p,
            SurfaceForcing(
                t_air_k=295.0, h_w_m2_k=10.0, q_solar_w_m2=500.0, q_longwave_down_w_m2=330.0
            ),
        )
        for p in scalars
    ]
    assert float(np.max(np.abs(result.temperatures_k - np.asarray(expected)))) < 0.1


def test_forty_eight_hours_is_enough_for_the_materials_that_matter() -> None:
    """⚠️ Measured per material class, and they do not all pass the same bar.

    Asphalt and thin steel settle to under **0.5 K** between a 48 h and a 96 h spin-up. Concrete
    does not and is not expected to — 0.1 m of it at 202 kJ m⁻² K⁻¹ is still remembering the day
    before yesterday — so its figure is *reported* rather than asserted at the same threshold,
    which is what the roadmap asks for and what ADR 0037 records.
    """
    cases = {
        "asphalt": ThermalProperties(2200 * 920 * 0.1 / 20.0, 0.94, 0.90),
        "thin_steel": ThermalProperties(7800 * 470 * 0.002, 0.85, 0.80),
        "concrete": ThermalProperties(2300 * 880 * 0.1, 0.92, 0.65),
    }
    gaps = {}
    for name, props in cases.items():
        packed = FacetProperties.stack([props])
        short = spin_up(packed, lambda t: _forcing((t / 3600.0) % 24.0), "diurnal", 0.0, 48.0, 60.0)
        long = spin_up(packed, lambda t: _forcing((t / 3600.0) % 24.0), "diurnal", 0.0, 96.0, 60.0)
        gaps[name] = float(abs(short.temperatures_k[0] - long.temperatures_k[0]))
    assert gaps["asphalt"] < 0.5, gaps
    assert gaps["thin_steel"] < 0.5, gaps
    assert gaps["concrete"] < 1.0, gaps
    assert gaps["concrete"] > gaps["thin_steel"], "the slowest material should settle slowest"
    assert DEFAULT_SPIN_UP_HOURS == 48.0


def test_starting_at_the_air_temperature_is_wrong_by_kelvins() -> None:
    """The reason spin-up exists, as a number rather than an argument."""
    packed = FacetProperties.stack([ThermalProperties(2200 * 920 * 0.005, 0.94, 0.90)])
    spun = spin_up(
        packed, lambda t: _forcing((t / 3600.0) % 24.0), "diurnal", 14.0 * 3600.0, 48.0, 60.0
    )
    naive = float(np.asarray(_forcing(14.0).t_air_k))
    assert abs(float(spun.temperatures_k[0]) - naive) > 5.0


def test_the_cache_is_keyed_on_what_determines_the_answer() -> None:
    """A second call with identical keys performs **zero** steps; 0.01 K invalidates."""
    packed, _ = _random_facets(6, seed=5)
    cache = SpinUpCache()
    first = spin_up(
        packed, lambda t: _forcing((t / 3600.0) % 24.0), "w1", 0.0, 6.0, 60.0, cache=cache
    )
    assert first.steps > 0 and not first.from_cache
    second = spin_up(
        packed, lambda t: _forcing((t / 3600.0) % 24.0), "w1", 0.0, 6.0, 60.0, cache=cache
    )
    assert second.steps == 0 and second.from_cache
    assert np.array_equal(first.temperatures_k, second.temperatures_k)
    assert cache.hits == 1 and cache.misses == 1

    # a different weather, a different t0, or a different material all miss
    for kwargs in ({"weather_hash": "w2"}, {"t0_s": 3600.0}):
        spin_up(
            packed,
            lambda t: _forcing((t / 3600.0) % 24.0),
            kwargs.get("weather_hash", "w1"),
            kwargs.get("t0_s", 0.0),
            6.0,
            60.0,
            cache=cache,
        )
    assert cache.misses == 3

    perturbed = FacetProperties(
        heat_capacity_j_m2_k=packed.heat_capacity_j_m2_k + 0.01,
        emissivity=packed.emissivity,
        solar_absorptivity=packed.solar_absorptivity,
    )
    assert spin_up_key(perturbed, "w1", 0.0, 6.0, 60.0) != spin_up_key(packed, "w1", 0.0, 6.0, 60.0)


def test_the_spin_up_ends_where_the_scene_begins() -> None:
    """It integrates the hours *before* t₀, so what comes back is the state at t₀ itself — not a
    state at some earlier time the caller then has to advance."""
    packed = FacetProperties.stack([ThermalProperties(50_000.0, 0.94, 0.90)])
    seen: list[float] = []

    def forcing_at(t_s: float) -> FacetForcing:
        seen.append(t_s)
        return _forcing((t_s / 3600.0) % 24.0)

    t0 = 12.0 * 3600.0
    spin_up(packed, forcing_at, "w", t0, hours=6.0, dt_s=600.0)
    assert min(seen) == pytest.approx(t0 - 6.0 * 3600.0)
    assert max(seen) < t0
    assert max(seen) >= t0 - 600.0
