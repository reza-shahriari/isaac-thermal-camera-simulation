"""Fire: what a flame radiates onto a surface, and how hot the air above it is (roadmap PH.7).

The correlations here are published, so most of these tests are not "does it compute" but "does
it still say what its source says". Three in particular earn their place.

**The flame tip falls out at 453 K whatever the fire is.** Heskestad's mean flame height is
*defined* as the height at which the centreline excess has dropped to roughly 500 K, so the
constant has to reappear when the correlation is evaluated at that height — and it does,
independent of Q and D, for a one-bar gas ring and a burning tanker alike. If a coefficient were
transcribed wrong, or the virtual origin used with the wrong heat release, this number would move
and nothing else in the module would notice.

**A flame's SEP is not its σT⁴.** The flux term deliberately takes the surface's absorptivity for
the flame's spectrum and its own emissivity for the sky it blocks, which are the same number only
for a grey surface. The steady-state test pins the whole chain through `FacetSolver`.

**The slab a camera sees and the air a thermometer reads come from one model.** `flame_plume`
solves the mixing length rather than taking one, so the two cannot be authored apart.

docs/physics-model.md §6.1, §6.6; ADR 0088, ADR 0090, ADR 0114.
"""

from __future__ import annotations

import numpy as np
import pytest

from irsim.pipeline.plume import FLAME_T_K, flame_plume
from irsim.radiometry.constants import SIGMA_SB
from irsim.thermal.facets import FacetForcing, FacetProperties, FacetSolver
from irsim.thermal.fire import (
    PLUME_SPREAD,
    SEP_SMOKE_OBSCURED_W_M2,
    SEP_UNOBSCURED_W_M2,
    PoolFire,
    centreline_rise_k,
    flame_flux_w_m2,
    flame_height_m,
    plume_air_temperature_k,
    virtual_origin_m,
)
from irsim.thermal.spatial_sources import RadiantRectangle, clamp_view_factor_sum

T_AIR = 293.0

#: A one-metre pool releasing 1 MW convectively -- a car fire's order, and the size most of the
#: fire-protection literature is written about.
FIRE = PoolFire(diameter_m=1.0, q_c_kw=1000.0)


# --- Heskestad -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "diameter_m, q_c_kw",
    [(0.3, 50.0), (1.0, 1000.0), (2.0, 5000.0), (6.0, 40000.0)],
)
def test_the_flame_tip_is_the_same_temperature_whatever_the_fire(
    diameter_m: float, q_c_kw: float
) -> None:
    """Heskestad's flame height *means* "where ΔT₀ ≈ 500 K", so the constant has to reappear.

    Two decades of fire size and twenty times the pool diameter, and the answer does not move:
    453 K above ambient at the tip. It is algebraically exact — the Q and D dependence cancels
    between ``Q_c^(2/3)`` and ``(L − z₀)^(−5/3)`` — which is what makes it a transcription check
    rather than a curve fit.
    """
    fire = PoolFire(diameter_m=diameter_m, q_c_kw=q_c_kw)
    rise = float(centreline_rise_k(fire.flame_height_m, fire, T_AIR))
    assert rise == pytest.approx(452.7, abs=0.5), rise


def test_the_profile_is_continuous_at_the_tip_monotone_above_it_and_vanishes() -> None:
    """The three things the roadmap asks of ΔT₀, in one sweep."""
    tip = FIRE.flame_height_m
    # Below the tip the value is *held*, so the one-sided limit is bit-for-bit the tip's.
    assert float(centreline_rise_k(tip * (1.0 - 1e-12), FIRE, T_AIR)) == float(
        centreline_rise_k(tip, FIRE, T_AIR)
    )
    # From above it is the correlation, and a micrometre's step moves it by a thousandth of a
    # kelvin -- continuous, not merely close.
    above = float(centreline_rise_k(tip + 1e-6, FIRE, T_AIR))
    assert abs(above - float(centreline_rise_k(tip, FIRE, T_AIR))) < 1e-3

    z = np.concatenate([np.linspace(0.0, tip, 32), np.geomspace(tip, 500.0, 96)])
    rise = centreline_rise_k(z, FIRE, T_AIR)
    assert np.all(np.diff(rise) <= 1e-9), "the plume warms as it rises somewhere"
    assert np.all(rise[z < tip] == pytest.approx(float(rise[0])))  # held inside the flame
    assert float(rise[-1]) < 0.1, float(rise[-1])


def test_the_correlations_are_the_published_ones() -> None:
    """L and z₀ against their closed forms, and the ordering that makes the model well posed."""
    q = FIRE.q_kw
    assert FIRE.flame_height_m == pytest.approx(0.235 * q**0.4 - 1.02 * 1.0)
    assert FIRE.virtual_origin_m == pytest.approx(0.083 * q**0.4 - 1.02 * 1.0)
    # L − z₀ = 0.152 Q^(2/5) whatever D is: the reach that sets the tip temperature.
    assert FIRE.flame_height_m - FIRE.virtual_origin_m == pytest.approx(0.152 * q**0.4)
    assert flame_height_m(q, 1.0) > virtual_origin_m(q, 1.0)


def test_a_heat_release_written_in_watts_is_refused() -> None:
    """1 MW is 1000, not 1 000 000, and the difference is a plume 100 times too strong."""
    with pytest.raises(ValueError, match="looks like watts"):
        PoolFire(diameter_m=1.0, q_c_kw=1.0e6)
    PoolFire(diameter_m=1.0, q_c_kw=1.0e3)  # the same fire, correctly


def test_a_fire_that_does_not_clear_its_own_rim_is_refused() -> None:
    """Below the correlation's range the flame height goes negative, which is not a small fire."""
    with pytest.raises(ValueError, match="does not clear the pool rim"):
        PoolFire(diameter_m=6.0, q_c_kw=40.0)


def test_the_plume_is_hottest_on_the_axis_and_cools_outward() -> None:
    """The Gaussian is the one estimated number here, so what is tested is its limits."""
    z = 6.0
    axis = plume_air_temperature_k(np.array([0.0, 0.0, z]), FIRE, T_AIR)
    assert float(axis) == pytest.approx(T_AIR + float(centreline_rise_k(z, FIRE, T_AIR)))

    radii = np.array([0.0, 0.25, 0.5, 1.0, 2.0, 8.0])
    points = np.stack([radii, np.zeros_like(radii), np.full_like(radii, z)], axis=-1)
    temps = plume_air_temperature_k(points, FIRE, T_AIR)
    assert np.all(np.diff(temps) < 0.0)
    assert float(temps[-1]) == pytest.approx(T_AIR, abs=1e-6)
    # Half-width where the correlation says: PLUME_SPREAD above the virtual origin.
    b = PLUME_SPREAD * (z - FIRE.virtual_origin_m)
    at_b = plume_air_temperature_k(np.array([b, 0.0, z]), FIRE, T_AIR)
    assert (float(at_b) - T_AIR) / (float(axis) - T_AIR) == pytest.approx(np.exp(-1.0), rel=1e-9)


def test_a_surface_beside_or_below_the_fire_is_at_ambient() -> None:
    """The plume rises; a wall next to a fire is heated by radiation, not by this term."""
    points = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, -2.0], [3.0, 0.0, -1.0]])
    assert np.allclose(plume_air_temperature_k(points, FIRE, T_AIR), T_AIR)


# --- the radiative half ----------------------------------------------------------------------


def test_a_flame_rectangle_radiates_its_sep_and_refuses_a_temperature() -> None:
    """The two ways of saying what a radiator emits, and the refusal to hold both at once."""
    engine = RadiantRectangle(
        centre_m=(0.0, 0.0, 0.0),
        u_axis=(1.0, 0.0, 0.0),
        v_axis=(0.0, 1.0, 0.0),
        half_u_m=0.5,
        half_v_m=0.4,
        emissivity=0.9,
    )
    assert engine.emitted_flux_w_m2(400.0) == pytest.approx(0.9 * SIGMA_SB * 400.0**4)

    flame = RadiantRectangle(
        centre_m=(0.0, 0.0, 0.0),
        u_axis=(1.0, 0.0, 0.0),
        v_axis=(0.0, 1.0, 0.0),
        half_u_m=0.5,
        half_v_m=1.6,
        sep_w_m2=SEP_UNOBSCURED_W_M2,
    )
    assert flame.emitted_flux_w_m2() == pytest.approx(SEP_UNOBSCURED_W_M2)
    with pytest.raises(ValueError, match="no temperature to radiate at"):
        flame.emitted_flux_w_m2(1200.0)
    with pytest.raises(ValueError, match="needs a temperature"):
        engine.emitted_flux_w_m2()


def test_smoke_takes_most_of_a_large_fires_radiation_away() -> None:
    """Why a bigger pool is not proportionally worse to stand beside: its flame is shrouded."""
    assert SEP_SMOKE_OBSCURED_W_M2 < 0.4 * SEP_UNOBSCURED_W_M2


def test_the_flux_uses_absorptivity_for_the_flame_and_emissivity_for_the_sky() -> None:
    """The distinction the term exists for. A grey surface is the special case, not the rule."""
    view = np.array([0.10])
    sep = 1.2e5
    assert float(flame_flux_w_m2(view, sep, 0.85)) == pytest.approx(0.10 * 0.85 * sep)
    # The sky a flame standing over a surface blocks is weighted by ε, not by α.
    with_sky = flame_flux_w_m2(view, sep, 0.85, surface_emissivity=0.93, longwave_down_w_m2=300.0)
    assert float(with_sky) == pytest.approx(0.10 * (0.85 * sep - 0.93 * 300.0))
    # A cell already seeing half the sky loses half as much again.
    half = flame_flux_w_m2(
        view, sep, 0.85, surface_emissivity=0.93, longwave_down_w_m2=150.0, sky_view=0.5
    )
    assert float(half) == pytest.approx(float(with_sky))


def test_the_pinned_steady_state_is_reproduced_through_the_solver() -> None:
    """`PH.7`'s own acceptance: F = 0.10 facing 120 kW/m², ε = 0, h > 0.

    With no radiative loss the balance is h (T − T_air) = q_int, so the cell must sit at exactly
    ``T_air + α F SEP / h`` — and nothing else in the chain may add a term. Run to steady state
    through the real `FacetSolver`, not the closed form, so the test would fail if `q_internal`
    were weighted by ε on the way in (it is a deposition, not an irradiance).
    """
    alpha, view, sep, h = 0.85, 0.10, 1.2e5, 40.0
    q = float(flame_flux_w_m2(np.array([view]), sep, alpha)[0])
    assert q == pytest.approx(alpha * 12.0e3)

    props = FacetProperties(
        heat_capacity_j_m2_k=np.array([2.0e4]),
        emissivity=np.array([0.0]),
        solar_absorptivity=np.array([0.0]),
    )
    solver = FacetSolver(props, np.array([T_AIR]))
    forcing = FacetForcing(t_air_k=T_AIR, h_w_m2_k=h, q_internal_w_m2=q)
    for _ in range(4000):
        solver.advance(forcing, 30.0)
    assert float(solver.temperatures_k[0]) == pytest.approx(T_AIR + q / h, abs=1e-9)


def test_overlapping_radiators_are_clamped_to_a_unit_solid_angle() -> None:
    """ADR 0090 still holds when one of the radiators is a flame: Σ F ≤ 1 per cell."""
    flame = np.array([0.55, 0.30, 0.05])
    pool = np.array([0.60, 0.20, 0.02])
    a, b = clamp_view_factor_sum([flame, pool])
    assert np.all(a + b <= 1.0 + 1e-9)
    # The relative split between radiators is untouched where the clamp bites.
    assert float(a[0] / b[0]) == pytest.approx(float(flame[0] / pool[0]))
    assert np.allclose(a[2], flame[2]) and np.allclose(b[2], pool[2])


# --- the flame a camera sees -------------------------------------------------------------------


def test_the_slab_cools_to_exactly_the_tip_heskestad_gives() -> None:
    """One model, two consumers. The mixing length is solved, not authored, so they agree."""
    world = flame_plume(FIRE, T_AIR)
    assert world.t_tip_k == FLAME_T_K
    assert world.length_m == pytest.approx(FIRE.flame_height_m)
    assert world.radius_tip_m == pytest.approx(0.5 * FIRE.diameter_m)
    assert world.f_soot > 0.0 and world.p_co2_atm == 0.0 and world.p_h2o_atm == 0.0

    phi = np.exp(-world.length_m / world.mixing_length_m)
    slab_tip_k = T_AIR + phi * (world.t_tip_k - T_AIR)
    heskestad_tip_k = T_AIR + float(centreline_rise_k(FIRE.flame_height_m, FIRE, T_AIR))
    assert slab_tip_k == pytest.approx(heskestad_tip_k, rel=1e-9)


def test_a_flame_colder_than_its_own_tip_is_refused() -> None:
    """The slab would have to warm as it rises, which is not a cooling model with a bad number."""
    with pytest.raises(ValueError, match="warm as it rises"):
        flame_plume(FIRE, T_AIR, t_flame_k=700.0)


def test_soot_is_greyish_where_the_exhaust_gas_is_banded() -> None:
    """Why a fire looks alike in both bands and a plume does not, as a number rather than a claim.

    Rayleigh soot absorbs as ``κ ∝ 1/λ``, so its band means differ only by the ratio of the bands'
    emission-weighted ``⟨1/λ⟩``: **2.47** between 3-5 µm and 7.5-13.5 µm, which is just the ratio
    of the two wavelengths. Hot CO2 differs by **5.49** over the same pair, because its absorption
    is *in lines* and the lines are in one band. So a camera that cannot see an exhaust plume can
    still see a fire, which is the operational fact underneath both steps — and it is why `PH.7`
    needs no gas table while `PH.6` could not exist without one.
    """
    from irsim.materials.library import nominal_response
    from irsim.pipeline.gas_slab import soot_band_kappa_per_m
    from irsim.pipeline.gas_tables import load_gas_tables

    mwir, lwir = nominal_response("mwir"), nominal_response("lwir")
    soot_ratio = soot_band_kappa_per_m(mwir, 1e-6, FLAME_T_K) / soot_band_kappa_per_m(
        lwir, 1e-6, FLAME_T_K
    )
    assert 2.0 < soot_ratio < 3.2, soot_ratio

    column = 0.05
    gas_ratio = load_gas_tables("mwir").species["co2"].at(600.0, column) / load_gas_tables(
        "lwir"
    ).species["co2"].at(600.0, column)
    assert gas_ratio > 2.0 * soot_ratio, (gas_ratio, soot_ratio)
