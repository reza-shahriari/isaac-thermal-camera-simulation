"""AT.1 — every pixel takes its own slant path, not a horizontal one.

`pipeline/atmosphere.py` passed elevation **0.0 unconditionally**, so every *resolved* pixel was
given surface-density extinction and surface-temperature emission over its whole slant range —
while the *unresolved* point-target path beside it used `target.elevation_rad`, and so did the sky
behind it. Contrast therefore jumped at the resolved/unresolved handoff for no physical reason, and
the error grows with elevation exactly where phase 1's subject lives: a target against the sky.

The tests that carry weight are the three at the end. `test_the_defect_it_fixes_is_this_big`
records the size of the error as a number rather than a claim;
`test_contrast_does_not_jump_across_the_handoff` is the acceptance criterion; and
`test_the_lut_reproduces_the_exact_quadrature` is what lets the LUT be trusted at all, since a
table that is smooth and wrong looks exactly like a table that is right.

docs/physics-model.md §7.4, §5.3; ADR 0071; roadmap AT.1.
"""

from __future__ import annotations

import math
import pathlib

import numpy as np
import pytest

from irsim.atmosphere.layered import LayeredAtmosphere, column_length
from irsim.atmosphere.library import load_atmosphere_preset
from irsim.pipeline.atmosphere import apply_layered_gbuffer
from irsim.radiometry.lut import BandLUT
from irsim.radiometry.spectral_response import load_spectral_response
from irsim.thermal import WeatherSample, WeatherSeries

REPO = pathlib.Path(__file__).resolve().parents[2]
#: dL/dT for the Boson LWIR band at 288 K, so a radiance error can be quoted in millikelvin
#: against the 50 mK NETD the whole project is budgeted around.
DL_DT_W_M2_SR_K = 0.78171


@pytest.fixture(scope="module")
def atmosphere() -> LayeredAtmosphere:
    response = load_spectral_response(REPO / "data" / "spectra" / "responses" / "boson_vox.csv")
    lut = BandLUT.build(response)
    weather = WeatherSeries.constant(
        WeatherSample(288.15, 0.46, 1.0, 0.0, 0.0, 0.0, 23000.0, 0.0), 3600.0
    )
    return LayeredAtmosphere(
        load_atmosphere_preset("us_standard_clear"), weather, {"lwir": lut}, {"lwir": response}
    )


# --- the geometry vectorises ------------------------------------------------------------------


def test_column_length_broadcasts_elevation_against_distance() -> None:
    """It took a scalar elevation, which is why the stage could only pass one for a whole frame."""
    d = np.array([[1000.0, 2000.0], [3000.0, 4000.0]])
    el = np.radians(np.array([[0.0, 30.0], [60.0, 90.0]]))
    got = column_length(d, el, 2000.0)
    assert got.shape == d.shape
    for i in range(2):
        for j in range(2):
            assert got[i, j] == pytest.approx(column_length(d[i, j], el[i, j], 2000.0))


def test_a_horizontal_ray_keeps_the_flat_column() -> None:
    d = np.array([100.0, 5000.0])
    assert np.array_equal(column_length(d, 0.0, 2000.0), d)
    assert np.array_equal(column_length(d, -0.3, 2000.0), d)


def test_an_upward_column_saturates_at_the_scale_height() -> None:
    """H/sinθ is the point: a vertical ray leaves the atmosphere, a horizontal one does not."""
    assert float(column_length(np.inf, math.pi / 2, 2000.0)) == pytest.approx(2000.0)
    assert float(column_length(np.inf, math.radians(30.0), 2000.0)) == pytest.approx(4000.0)
    assert math.isinf(float(column_length(np.inf, 0.0, 2000.0)))


def test_transmittance_varies_across_a_plane(atmosphere: LayeredAtmosphere) -> None:
    es = atmosphere.exponential_sum("lwir", 0.0)
    d = np.full(5, 5000.0)
    el = np.radians(np.array([0.0, 10.0, 30.0, 60.0, 90.0]))
    tau = es.transmittance(d, el)
    assert tau.shape == d.shape
    assert np.all(np.diff(tau) > 0.0), "a steeper ray must transmit more, not less"


# --- the LUT is worth trusting ------------------------------------------------------------------


@pytest.mark.parametrize("elevation_deg", [0.5, 2.0, 5.0, 20.0, 45.0, 70.0, 90.0])
def test_the_lut_reproduces_the_exact_quadrature(
    atmosphere: LayeredAtmosphere, elevation_deg: float
) -> None:
    """Against the 4000-step Simpson quadrature it replaces, in millikelvin.

    A LUT that is smooth and wrong is indistinguishable from one that is right by looking at a
    frame, so the budget is stated and measured: under 10 mK, against a 50 mK NETD.
    """
    es = atmosphere.exponential_sum("lwir", 0.0)
    lb_of_height = atmosphere._lb_of_height("lwir", 0.0, "lb")
    el = math.radians(elevation_deg)
    ranges = np.array([200.0, 1000.0, 5000.0, 20000.0])
    got = atmosphere.path_radiance_plane("lwir", 0.0, ranges, np.full(ranges.shape, el))
    for distance, value in zip(ranges, got, strict=True):
        exact = es.path_radiance(float(distance), el, lb_of_height)
        assert abs(value - exact) * 1000.0 / DL_DT_W_M2_SR_K < 5.0


def test_the_lut_degrades_beyond_the_ranges_this_project_renders(
    atmosphere: LayeredAtmosphere,
) -> None:
    """The stated envelope, measured rather than assumed.

    Inside 20 km the LUT is under 5 mK. At **100 km** it reaches 65 mK, past the 50 mK NETD — the
    elevation nodes are spaced for scenes, and at a hundred kilometres a node's worth of elevation
    is a very different column. No scene here renders that far (ADR 0071 bounds the model itself at
    a few km on a slant path), so the limit is recorded rather than engineered away; refining the
    node count is the fix if a long-range case ever arrives.
    """
    es = atmosphere.exponential_sum("lwir", 0.0)
    lb_of_height = atmosphere._lb_of_height("lwir", 0.0, "lb")
    el = math.radians(5.0)
    far = 100_000.0
    got = float(atmosphere.path_radiance_plane("lwir", 0.0, np.array([far]), np.array([el]))[0])
    error_mk = abs(got - es.path_radiance(far, el, lb_of_height)) * 1000.0 / DL_DT_W_M2_SR_K
    assert 10.0 < error_mk < 200.0, f"the recorded far-field error moved: {error_mk:.1f} mK"


def test_the_grid_is_uniform_in_transmittance_not_optical_depth(
    atmosphere: LayeredAtmosphere,
) -> None:
    """The coordinate choice, pinned, because it is worth 500x.

    `w = 1 - e^{-u}` makes an isothermal path exactly linear, so a trapezoid is exact in the limit
    every horizontal ray reduces to. A uniform `u` grid instead spends its points where the
    exponential has already killed the integrand: measured at 90 deg, where the whole optical depth
    is a fraction of one step, that was 77 mK against a 50 mK NETD.
    """
    es = atmosphere.exponential_sum("lwir", 0.0)
    w, cumulative = es.cumulative_path_table(
        math.pi / 2, atmosphere._lb_of_height("lwir", 0.0, "lb")
    )
    assert np.isfinite(w).all() and np.isfinite(cumulative).all()
    assert np.allclose(np.diff(w), np.diff(w)[0], rtol=1e-9), "the grid is not uniform in w"
    assert w[0] == 0.0 and 0.0 < w[-1] < 1.0, "the top node must stay strictly below 1"


def test_an_isothermal_path_is_exact(atmosphere: LayeredAtmosphere) -> None:
    """The closed form the coordinate was chosen for: constant L_B gives (1 - tau) L_B exactly."""
    es = atmosphere.exponential_sum("lwir", 0.0)
    flat = 42.0

    def lb_of_height(h: np.ndarray) -> np.ndarray:
        return np.full(np.shape(h), flat)

    el = math.radians(30.0)
    w, cumulative = es.cumulative_path_table(el, lb_of_height)
    for distance in (500.0, 5000.0, 50_000.0):
        od = es.optical_depths(np.asarray(distance), el)
        approx = float(
            np.dot(
                es.weights,
                [np.interp(1.0 - np.exp(-od[k]), w, cumulative[k]) for k in range(es.n_terms)],
            )
        )
        expected = float(np.dot(es.weights, 1.0 - np.exp(-od))) * flat
        # Machine precision, not a tolerance: this is the identity the w coordinate was chosen
        # for. Interpolating the same table in `u` instead costs 1.2e-4, which is why the method
        # hands back `w`.
        assert approx == pytest.approx(expected, rel=1e-12)


def test_a_downward_ray_is_refused_by_the_table(atmosphere: LayeredAtmosphere) -> None:
    es = atmosphere.exponential_sum("lwir", 0.0)
    with pytest.raises(ValueError, match="upward"):
        es.cumulative_path_table(-0.1, atmosphere._lb_of_height("lwir", 0.0, "lb"))


# --- the defect, and the acceptance ---------------------------------------------------------------


def test_the_defect_it_fixes_is_this_big(atmosphere: LayeredAtmosphere) -> None:
    """The number, recorded rather than claimed: us_standard_clear, LWIR, 5 km."""
    es = atmosphere.exponential_sum("lwir", 0.0)
    d = np.array([5000.0])
    flat = float(es.transmittance(d, np.zeros(1))[0])
    slant = float(es.transmittance(d, np.full(1, math.radians(45.0)))[0])
    assert flat == pytest.approx(0.5995, abs=5e-4)
    # 0.7230 before AT.37 gave the self-continuum its own half-depth column (ADR 0200); the
    # horizontal path, which never leaves the surface, is unchanged
    assert slant == pytest.approx(0.7243, abs=5e-4)
    assert (slant - flat) / flat > 0.20

    path_flat = float(atmosphere.path_radiance_plane("lwir", 0.0, d, np.zeros(1))[0])
    path_slant = float(
        atmosphere.path_radiance_plane("lwir", 0.0, d, np.full(1, math.radians(45.0)))[0]
    )
    assert path_flat == pytest.approx(18.27, abs=0.05)
    assert path_slant == pytest.approx(11.60, abs=0.05)
    assert (path_flat - path_slant) / path_slant > 0.35


def test_contrast_does_not_jump_across_the_handoff(atmosphere: LayeredAtmosphere) -> None:
    """The acceptance criterion: a resolved pixel and the unresolved path must agree.

    The point-target path has always used `target.elevation_rad`. A resolved pixel at the same
    range and elevation must now see the same transmittance and the same path radiance — otherwise
    a target crossing the resolved/unresolved boundary changes apparent temperature for a reason
    that is in the code rather than in the sky.
    """
    es = atmosphere.exponential_sum("lwir", 0.0)
    lb_of_height = atmosphere._lb_of_height("lwir", 0.0, "lb")
    for elevation_deg in (5.0, 20.0, 45.0):
        el = math.radians(elevation_deg)
        distance = 4000.0
        # what the unresolved path uses
        unresolved_tau = float(es.transmittance(np.asarray(distance), el))
        unresolved_path = es.path_radiance(distance, el, lb_of_height)
        # what a resolved pixel now gets
        plane = np.full((1, 1), distance)
        resolved_tau = float(es.transmittance(plane, np.full((1, 1), el))[0, 0])
        resolved_path = float(
            atmosphere.path_radiance_plane("lwir", 0.0, plane, np.full((1, 1), el))[0, 0]
        )
        assert resolved_tau == pytest.approx(unresolved_tau, rel=1e-12)
        assert abs(resolved_path - unresolved_path) * 1000.0 / DL_DT_W_M2_SR_K < 10.0


def test_the_stage_without_an_elevation_plane_is_bit_identical(
    atmosphere: LayeredAtmosphere,
) -> None:
    """Every scene that does not supply the plane renders exactly as it did."""
    radiance = np.full((4, 4), 60.0, dtype=np.float32)
    distance = np.full((4, 4), 3000.0, dtype=np.float32)
    without = apply_layered_gbuffer(atmosphere, "lwir", 0.0, radiance, distance, "lb")
    flat = apply_layered_gbuffer(
        atmosphere, "lwir", 0.0, radiance, distance, "lb", elevation_rad=np.zeros((4, 4))
    )
    assert np.allclose(without, flat, rtol=1e-6)


def test_the_stage_varies_across_a_frame(atmosphere: LayeredAtmosphere) -> None:
    """A frame with a horizon in it must not come out with one transmittance."""
    radiance = np.full((8, 8), 60.0, dtype=np.float32)
    distance = np.full((8, 8), 6000.0, dtype=np.float32)
    elevation = np.radians(np.linspace(0.0, 60.0, 8))[:, None] * np.ones((1, 8))
    out = apply_layered_gbuffer(
        atmosphere, "lwir", 0.0, radiance, distance, "lb", elevation_rad=elevation
    )
    rows = out.mean(axis=1)
    assert np.all(np.diff(rows) > 0.0), "a steeper ray must attenuate the scene less"
    assert float(rows[-1] - rows[0]) > 1.0


def test_a_float16_elevation_plane_is_refused(atmosphere: LayeredAtmosphere) -> None:
    """Non-negotiable #2: elevation scales the optical depth of the whole slant path."""
    radiance = np.full((2, 2), 60.0, dtype=np.float32)
    distance = np.full((2, 2), 3000.0, dtype=np.float32)
    with pytest.raises(TypeError, match="float16"):
        apply_layered_gbuffer(
            atmosphere,
            "lwir",
            0.0,
            radiance,
            distance,
            "lb",
            elevation_rad=np.zeros((2, 2), dtype=np.float16),
        )


def test_the_join_at_the_horizon_is_continuous(atmosphere: LayeredAtmosphere) -> None:
    """θ = 0 is a node, not a clamp, so the join is exact rather than within a tolerance.

    An earlier draft clamped everything below 0.25° to the horizontal closed form and left a
    **209 mK** step there — four times the NETD, right where long-range scene sits. Near the
    horizon the flat-earth column expands to `d (1 - d sinθ / 2H)`, which is *linear in sin θ*, so
    a grid uniform in sin θ anchored on the exact horizontal answer is continuous by construction.
    """
    d = np.array([10_000.0])
    horizontal = float(atmosphere.path_radiance_plane("lwir", 0.0, d, np.array([0.0]))[0])
    just_above = float(atmosphere.path_radiance_plane("lwir", 0.0, d, np.array([1e-9]))[0])
    assert abs(horizontal - just_above) * 1000.0 / DL_DT_W_M2_SR_K < 0.1

    # And a ray that does rise really does differ, so continuity is not flatness.
    risen = float(atmosphere.path_radiance_plane("lwir", 0.0, d, np.radians([0.75]))[0])
    assert abs(risen - horizontal) * 1000.0 / DL_DT_W_M2_SR_K > 50.0


def test_a_downward_ray_takes_the_horizontal_form(atmosphere: LayeredAtmosphere) -> None:
    """Below the horizon is the sea's business (ADR 0078), not this model's."""
    d = np.array([4000.0])
    level = float(atmosphere.path_radiance_plane("lwir", 0.0, d, np.array([0.0]))[0])
    down = float(atmosphere.path_radiance_plane("lwir", 0.0, d, np.radians([-10.0]))[0])
    assert down == pytest.approx(level, rel=1e-12)
