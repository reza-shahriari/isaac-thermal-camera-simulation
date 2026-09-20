"""PT.8 — a field holds the ticks a query needs, not every tick it ever produced.

`ThermalField` appended every tick and pruned none, while a query only ever read the two ticks
bracketing it. Measured before this step: a 24 h run of the car scenes' 10 400-cell road patch at
a 30 s tick held 2 880 × 10 400 × 8 B ≈ 240 MB, and the 10⁵-cell fields the coupling lane brings
would be gigabytes — which is what stopped ADR 0074's full-diurnal time-lapse.

Three claims, each of which fails if the mechanism is wrong rather than merely absent: a bounded
field's memory is bounded (a leak of one tick per step fails the first test at 1 440 ticks); the
state hash of a ring equals the hash a walk over the full history gives, byte for byte (a hash
that forgot pruned ticks, or fed them in another order, fails); and every query inside the window
is bit-identical to the unbounded field's answer (an off-by-one in the ring's index fails).

docs/physics-model.md §6.4; ADR 0093
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from irsim.thermal.facets import FacetForcing, FacetProperties
from irsim.thermal.field import ThermalField
from irsim.thermal.surface_field import DEFAULT_KEEP_TICKS, PlanarPatch, PlanarThermalField

TICK_S = 60.0
DAY_S = 24 * 3600.0


def _forcing(t_s: float) -> FacetForcing:
    hour = (t_s / 3600.0) % 24.0
    return FacetForcing(
        t_air_k=288.0 + 6.0 * np.sin(np.pi * (hour - 8.0) / 12.0),
        h_w_m2_k=10.0,
        q_solar_w_m2=max(0.0, 900.0 * np.sin(np.pi * (hour - 6.0) / 12.0)),
        q_longwave_down_w_m2=320.0,
    )


def _road(keep_ticks: int | None, on_tick=None) -> PlanarThermalField:  # type: ignore[no-untyped-def]
    """The car scenes' road: 100 × 104 cells of asphalt, as MP.4b shipped it."""
    patch = PlanarPatch(
        origin_m=np.array([-15.0, 0.0, -15.0]),
        u_axis=np.array([1.0, 0.0, 0.0]),
        v_axis=np.array([0.0, 0.0, 1.0]),
        n_u=100,
        n_v=104,
        du_m=0.3,
        dv_m=0.3,
        thickness_m=0.10,
    )
    n = patch.n_cells
    props = FacetProperties(
        heat_capacity_j_m2_k=np.full(n, 101_200.0),
        emissivity=np.full(n, 0.94),
        solar_absorptivity=np.full(n, 0.90),
    )
    return PlanarThermalField(
        patch,
        props,
        _forcing,
        0.0,
        np.full(n, 288.0),
        TICK_S,
        keep_ticks=keep_ticks,
        on_tick=on_tick,
    )


def _small(keep_ticks: int | None, on_tick=None) -> ThermalField:  # type: ignore[no-untyped-def]
    props = FacetProperties(
        heat_capacity_j_m2_k=np.array([101_200.0, 4_399.0]),
        emissivity=np.array([0.94, 0.90]),
        solar_absorptivity=np.array([0.90, 0.94]),
    )
    return ThermalField(
        props,
        _forcing,
        0.0,
        np.array([288.0, 288.0]),
        TICK_S,
        keep_ticks=keep_ticks,
        on_tick=on_tick,
    )


# ---------------------------------------------------------------------------------------------
# memory
# ---------------------------------------------------------------------------------------------


def test_a_day_of_the_road_patch_holds_under_sixteen_megabytes() -> None:
    """The row's own acceptance. Before PT.8 this same run held ~240 MB of ticks."""
    road = _road(DEFAULT_KEEP_TICKS)
    road.advance_to(DAY_S)
    held = sum(t.temperatures_k.nbytes for t in road.field._ticks)  # noqa: SLF001
    assert road.n_ticks == int(DAY_S / TICK_S) + 1  # every tick was produced ...
    assert road.n_held == 2  # ... and only the bracketing pair is resident
    assert held < 16 * 1024 * 1024, held
    assert held == 2 * 10_400 * 8


def test_the_default_for_a_spatial_field_is_the_ring_and_for_a_facet_field_it_is_not() -> None:
    """A per-prim field is a few facets and the diurnal validations sweep its whole day."""
    assert DEFAULT_KEEP_TICKS == 2
    small = _small(None)
    small.advance_to(10 * TICK_S)
    assert small.n_held == small.n_ticks == 11
    assert small.keep_ticks is None


@pytest.mark.parametrize("bad", [0, 1, -3])
def test_a_ring_shorter_than_the_bracketing_pair_is_refused(bad: int) -> None:
    with pytest.raises(ValueError, match="at least 2"):
        _small(bad)


# ---------------------------------------------------------------------------------------------
# the hash and the hook
# ---------------------------------------------------------------------------------------------


def test_the_ring_s_hash_is_the_hash_of_the_whole_history() -> None:
    """The old `state_hash` walked every held tick; the running digest must equal that walk."""
    recorded: list[tuple[float, np.ndarray]] = []
    ring = _small(2, on_tick=lambda t, temps: recorded.append((t, temps.copy())))
    ring.advance_to(200 * TICK_S)
    walk = hashlib.sha256()
    for t, temps in recorded:
        walk.update(np.float64(t).tobytes())
        walk.update(np.ascontiguousarray(temps, dtype=np.float64).tobytes())
    assert ring.state_hash() == walk.hexdigest()
    # And it equals the unbounded field's, which held every one of those ticks.
    full = _small(None)
    full.advance_to(200 * TICK_S)
    assert full.state_hash() == ring.state_hash()
    assert len(recorded) == ring.n_ticks == 201


def test_a_query_leaves_the_ring_s_hash_and_count_untouched() -> None:
    ring = _road(2)
    ring.advance_to(30 * TICK_S)
    before, produced, held = ring.state_hash(), ring.n_ticks, ring.n_held
    for i in range(500):
        ring.temperature_at(29 * TICK_S + (i % 60))
        ring.sample_at(30 * TICK_S, np.array([[0.0, 0.0, 0.0]]))
    assert (ring.state_hash(), ring.n_ticks, ring.n_held) == (before, produced, held)


def test_the_hook_sees_every_tick_in_order_including_the_start() -> None:
    times: list[float] = []
    small = _small(2, on_tick=lambda t, _temps: times.append(t))
    small.advance_to(5 * TICK_S)
    assert times == [0.0, 60.0, 120.0, 180.0, 240.0, 300.0]


# ---------------------------------------------------------------------------------------------
# the answer inside the window is the same answer
# ---------------------------------------------------------------------------------------------


def test_five_hundred_queries_inside_the_window_are_bit_identical_to_the_unbounded_field() -> None:
    ring, full = _small(2), _small(None)
    ring.advance_to(100 * TICK_S)
    full.advance_to(100 * TICK_S)
    for t in np.linspace(99 * TICK_S, 100 * TICK_S, 500):
        assert np.array_equal(ring.temperature_at(float(t)), full.temperature_at(float(t)))
    assert ring.earliest_t_s == 99 * TICK_S


def test_a_query_before_the_window_raises_and_names_the_knob() -> None:
    """Answering from a fallback would tell a time-lapse a different story than was solved."""
    ring = _small(3)
    ring.advance_to(10 * TICK_S)
    ring.temperature_at(8 * TICK_S)  # the oldest held tick: fine
    with pytest.raises(ValueError, match="keep_ticks.*on_tick"):
        ring.temperature_at(7 * TICK_S)
    with pytest.raises(ValueError, match="before the field's start"):
        ring.temperature_at(-1.0)


def test_a_wider_ring_answers_a_wider_window() -> None:
    ring, full = _small(5), _small(None)
    ring.advance_to(20 * TICK_S)
    full.advance_to(20 * TICK_S)
    assert ring.earliest_t_s == 16 * TICK_S
    assert np.array_equal(ring.temperature_at(16.5 * TICK_S), full.temperature_at(16.5 * TICK_S))
