"""GT.7 -- the cell and prim budget, pinned so nobody coarsens a patch for speed.

Fraunhofer ran 1,313,410 triangles with a 10-layer stack through five day-night cycles in 252 s,
in MATLAB, on one i7-8700. That is the number this lane's sizing rests on: 10^5-10^6 cells is
affordable, so a patch authored coarsely to "keep the solve cheap" is trading accuracy for nothing.
This file measures whether that is still true here.

**Timing tests on a shared workstation flake.** Two things keep these honest. The budgets are set
roughly an order of magnitude above the measured value, because the regression worth catching is an
accidental O(n^2) or a per-cell Python loop, not a 20 % drift -- and a tight budget would be red
whenever somebody else's render is running. And the one assertion that can be made *structurally*
is: `test_sample_computes_the_local_coordinates_once` counts calls instead of seconds, so the
`GT.7` optimisation is guarded by something that cannot flake at all.

Measured on this workstation while otherwise loaded, at the values the budgets are set from:

===========================================  ==========  ===========
what                                         measured    budget here
===========================================  ==========  ===========
102,400-cell field, 48 h spin-up at 60 s      11.5 s      90 s
``PointwiseTemperature.apply``, 640x512       32.7 ms     250 ms
===========================================  ==========  ===========
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from irsim.thermal.balance import ThermalProperties
from irsim.thermal.facets import FacetForcing, FacetProperties
from irsim.thermal.surface_field import PlanarPatch, PlanarThermalField
from irsim_isaac.pipeline.point_bridge import PointwiseTemperature, SurfaceBinding

PANEL = ThermalProperties(heat_capacity_j_m2_k=8_000.0, emissivity=0.92, solar_absorptivity=0.85)
#: 320 x 320 = 102,400 cells -- the 10^5 the row asks for, on a 16 m x 16 m surface at 5 cm.
SPIN_UP_CELLS_BUDGET_S = 90.0
APPLY_BUDGET_MS = 250.0
WIDTH, HEIGHT = 640, 512


def _patch(n_u: int, n_v: int, du_m: float = 0.05, thickness_m: float = 0.25) -> PlanarPatch:
    return PlanarPatch(
        origin_m=np.zeros(3),
        u_axis=np.array([1.0, 0.0, 0.0]),
        v_axis=np.array([0.0, 1.0, 0.0]),
        n_u=n_u,
        n_v=n_v,
        du_m=du_m,
        dv_m=du_m,
        thickness_m=thickness_m,
    )


def _diurnal(t_s: float) -> FacetForcing:
    """A real day, so the solver is not stepping a constant and short-cutting the work."""
    day = float(np.sin(2.0 * np.pi * t_s / 86400.0))
    return FacetForcing(
        t_air_k=290.0 + 8.0 * day,
        h_w_m2_k=10.0,
        q_longwave_down_w_m2=300.0 + 40.0 * day,
        q_solar_w_m2=max(0.0, 800.0 * day),
    )


def _best(fn, reps: int = 5) -> float:
    """Fastest of several runs: the minimum is the one least interrupted by other load."""
    fn()
    times = []
    for _ in range(reps):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return min(times)


# --- the structural one, which cannot flake ---------------------------------------------------


def test_sample_computes_the_local_coordinates_once() -> None:
    """`GT.7`: `sample` needed both the patch coordinates and the inside test, and computed the
    coordinates twice to get them -- once itself and once inside `contains`.

    Counted rather than timed, because this is the assertion that has to survive a loaded machine.
    ``local_coords`` is three dot products over every pixel of the frame and measured **5.3 ms of
    a 23.5 ms sample** at 640x512; doing it twice was 22 % of the sampling cost for nothing.
    """
    patch = _patch(16, 16)
    values = np.linspace(280.0, 320.0, patch.n_cells)
    points = np.random.default_rng(0).uniform(0.0, 0.8, size=(4096, 3))

    calls = 0
    original = PlanarPatch.local_coords

    def counting(self, pts):  # noqa: ANN001, ANN202
        nonlocal calls
        calls += 1
        return original(self, pts)

    PlanarPatch.local_coords = counting  # type: ignore[method-assign]
    try:
        sampled = patch.sample(values, points)
    finally:
        PlanarPatch.local_coords = original  # type: ignore[method-assign]
    assert calls == 1, f"sample recomputed the patch coordinates {calls} times"
    # and it must still be the same answer, not a fast wrong one
    inside = patch.contains(points)
    assert np.isfinite(sampled[inside]).all()
    assert np.isnan(sampled[~inside]).all()


def test_contains_and_contains_local_agree() -> None:
    """The split must not have moved the slab test, only the coordinates it reads."""
    patch = _patch(8, 8, thickness_m=0.1)
    points = np.random.default_rng(1).uniform(-0.2, 0.6, size=(2000, 3))
    assert np.array_equal(patch.contains(points), patch.contains_local(patch.local_coords(points)))


# --- the budgets ------------------------------------------------------------------------------


@pytest.mark.slow
def test_a_hundred_thousand_cells_spin_up_inside_the_budget() -> None:
    """The row's premise, measured: 10^5 cells over 48 h is affordable, so do not coarsen a patch.

    A whole day-night cycle twice over, at a 60 s tick, driven by a varying forcing so the solver
    cannot short-cut a constant. Measured at **11.5 s** -- 39 ns per cell per tick -- against a
    budget of 90 s. Fraunhofer's MATLAB reference works out at 38 us per cell-day against this
    path's 56 us, the same order for a model that carried ten layers where this carries one.
    """
    patch = _patch(320, 320)
    assert patch.n_cells >= 100_000, f"the budget is for 10^5 cells, this is {patch.n_cells}"
    properties = FacetProperties.stack([PANEL] * patch.n_cells)

    start = time.perf_counter()
    field = PlanarThermalField(patch, properties, _diurnal, 0.0, 290.0, 60.0)
    field.advance_to(48.0 * 3600.0)
    elapsed = time.perf_counter() - start

    temperatures = field.temperature_at(48.0 * 3600.0)
    assert np.isfinite(temperatures).all(), "a spin-up that diverged is not a timing result"
    assert 200.0 < float(temperatures.min()) < 400.0
    assert elapsed < SPIN_UP_CELLS_BUDGET_S, (
        f"{patch.n_cells:,} cells over 48 h took {elapsed:.1f} s against a "
        f"{SPIN_UP_CELLS_BUDGET_S:.0f} s budget -- roughly eight times the measured 11.5 s, so "
        "this is a real regression rather than a busy machine: look for a per-cell Python loop."
    )


@pytest.mark.slow
def test_a_bound_prim_costs_a_fraction_of_a_frame() -> None:
    """The render half: what one patched prim costs per frame at a real sensor's resolution.

    The row recorded 79 ms per frame at 640x512 for one bound prim. It is **32.7 ms** here, part
    of which is `GT.7`'s own duplicated-pass fix and part of which is that the row's figure was
    taken on another machine -- so the budget is set from what is measured here, not from the
    difference, which is not this test's to claim.
    """
    patch = _patch(200, 200, du_m=0.01, thickness_m=0.5)
    field = PlanarThermalField(
        patch,
        FacetProperties.stack([PANEL] * patch.n_cells),
        lambda _t: FacetForcing(290.0, 10.0, 300.0),
        0.0,
        290.0,
        60.0,
    )
    bridge = PointwiseTemperature([SurfaceBinding("/World/car", field)])
    ids = np.full((HEIGHT, WIDTH), 7, dtype=np.uint32)
    points = np.zeros((HEIGHT, WIDTH, 3))
    points[..., 0], points[..., 1] = np.meshgrid(
        np.linspace(0.0, 1.9, WIDTH), np.linspace(0.0, 1.9, HEIGHT)
    )
    plane = np.full((HEIGHT, WIDTH), 300.0, dtype=np.float32)
    labels = {7: {"class": "/World/car"}}

    out = bridge.apply(plane, ids, labels, points, 0.0)
    assert out.dtype == np.float32 and not np.array_equal(out, plane), "it must do the work"

    elapsed_ms = _best(lambda: bridge.apply(plane, ids, labels, points, 0.0)) * 1e3
    assert elapsed_ms < APPLY_BUDGET_MS, (
        f"one bound prim cost {elapsed_ms:.1f} ms at {WIDTH}x{HEIGHT} against a "
        f"{APPLY_BUDGET_MS:.0f} ms budget, about eight times the measured 32.7 ms."
    )


@pytest.mark.slow
def test_the_solve_scales_with_the_cell_count_and_not_worse() -> None:
    """A ratio, not a clock: sixteen times the cells must not cost far more than sixteen times.

    This is the shape of the regression the budgets exist for -- an accidental O(n^2) shows up here
    as a ratio well above the cell ratio, and it says so on a machine of any speed, which an
    absolute budget cannot.
    """
    ticks_to = 3600.0
    costs = {}
    for side in (80, 320):
        patch = _patch(side, side)
        properties = FacetProperties.stack([PANEL] * patch.n_cells)

        def run(patch=patch, properties=properties) -> None:
            PlanarThermalField(patch, properties, _diurnal, 0.0, 290.0, 60.0).advance_to(ticks_to)

        costs[patch.n_cells] = _best(run, reps=2)
    small, large = sorted(costs)
    ratio = costs[large] / costs[small]
    cells = large / small
    assert ratio < 2.5 * cells, (
        f"{large:,} cells cost {ratio:.1f}x what {small:,} did, for {cells:.0f}x the work -- "
        "superlinear, which is the regression this file exists to catch"
    )
