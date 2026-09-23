"""A cloud with a top: the deck both bands read (AT.12, ADR 0127).

AT.11 gave a cloud a real optical depth and a real distance, and it is still a plane-parallel
sheet: every ray that enters it stays in it, so an optically thick core reads one level to within
a kelvin. Looking straight up that is right — a cumulus base is flat, which is what the lifting
condensation level means. Looking at 20 degrees, where these clips aim, it is not: a real cumulus
field is towers, and a ray climbing 1.2 km at that elevation travels **3.3 km horizontally**, so
it crosses a line of towers and gaps and stops emitting wherever it happened to run out of cloud.

Two properties make this a generalisation rather than a replacement, and they are the two tests to
read first:

* `test_a_vertical_ray_reproduces_the_plane_parallel_model` — the vertical profile integrates to
  exactly the column's thickness, so looking up, **nothing moved**;
* `test_the_voxel_grid_is_the_marched_function_sampled` — the volume the path tracer renders is
  the field the infrared band integrates, evaluated on a grid. An axis transposition there is the
  classic VDB bug and would put the visible cloud somewhere the infrared cloud is not.
"""

from __future__ import annotations

import math
import pathlib

import numpy as np
import pytest

from irsim.atmosphere.cloud import CLOUD_OD_RATIO, generate_sky_cloud, psd_slope, sky_angles
from irsim.atmosphere.cloud_deck import (
    DEFAULT_MIN_ELEVATION_DEG,
    DEFAULT_SMALLEST_CLOUD_M,
    MARCH_STEP_M,
    MAX_MARCH_STEPS,
    MIN_MARCH_STEPS,
    TRANSECT_TO_RADIAL_SLOPE,
    CloudDeck,
    _direction,
    deck_field,
    generate_cloud_deck,
    resample_bilinear,
    vertical_profile,
)
from irsim.config.loader import load_sensor_config
from irsim.radiometry.lut_files import load_band_lut_for_config, load_band_response_for_config
from irsim.scene import Scene

REPO = pathlib.Path(__file__).resolve().parents[2]
SENSOR_YAML = REPO / "configs" / "sensors" / "flir_boson_640_lwir.yaml"
PHANTOM_SCENE = REPO / "configs" / "scenes" / "phantom3_outbound_pointwise.yaml"
BAND = "lwir"
AIM_ELEVATION_DEG = 20.0
VFOV_DEG = 24.752


@pytest.fixture(scope="module")
def scene() -> Scene:
    sensor = load_sensor_config(SENSOR_YAML)
    return Scene.from_file(
        PHANTOM_SCENE,
        {BAND: load_band_lut_for_config(sensor, REPO / "data" / "lut")},
        responses={BAND: load_band_response_for_config(sensor, REPO / "data")},
        quantity=sensor.sensor.quantity,
    )


@pytest.fixture(scope="module")
def deck(scene: Scene) -> CloudDeck:
    return scene.sky_models[BAND].cloud_deck(scene.t0_s, 7)


def _frame_angles(rows: int = 128, cols: int = 160) -> tuple[np.ndarray, np.ndarray]:
    half = 0.5 * VFOV_DEG
    el = np.radians(np.linspace(AIM_ELEVATION_DEG - half, AIM_ELEVATION_DEG + half, rows))
    az = np.radians(np.linspace(-15.5, 15.5, cols))
    return el[:, None] * np.ones((1, cols)), az[None, :] * np.ones((rows, 1))


# -- the profile, and the identity it exists for -------------------------------------------


def test_the_vertical_profile_integrates_to_exactly_one() -> None:
    """`∫₀¹ 6u(1-u) du = 1`. Everything below rests on this, so it is checked first and tightly."""
    u = np.linspace(0.0, 1.0, 200_001)
    assert np.trapezoid(vertical_profile(u), u) == pytest.approx(1.0, abs=1e-9)
    assert vertical_profile(np.array([-0.1, 0.0, 0.5, 1.0, 1.1])).tolist() == [
        0.0,
        0.0,
        1.5,
        0.0,
        0.0,
    ]


def test_a_vertical_ray_reproduces_the_plane_parallel_model(deck: CloudDeck) -> None:
    """Straight up, the third dimension changes nothing: τ = optical_depth × depth.

    This is what makes AT.12 a generalisation of AT.11 and not a second cloud. The tolerance is
    the march's own quadrature error (midpoint rule on a parabola), not a fudge: it falls as the
    step count rises, which the second half of this test demonstrates rather than assumes.
    """
    az = np.zeros(1)
    el = np.full(1, 0.5 * math.pi)
    # A deep column, found rather than assumed: since the depth map became a *height* ramp the
    # deepest columns are a small minority, and the cell directly overhead is usually not one.
    n = deck.depth.shape[0]
    i, j = np.unravel_index(int(np.argmax(deck.depth)), deck.depth.shape)
    centre = 0.5 * (n - 1)
    origin = ((i - centre) * deck.cell_m, 0.0, (j - centre) * deck.cell_m)
    column = float(deck.depth[i, j])
    assert column > 0.9, "the deepest column should reach the capping inversion"
    expected = deck.optical_depth * column
    coarse = float(deck.march(el, az, origin_m=origin, steps=8).optical_depth[0])
    fine = float(deck.march(el, az, origin_m=origin, steps=256).optical_depth[0])
    assert fine == pytest.approx(expected, rel=1e-4)
    assert abs(fine - expected) < abs(coarse - expected), "the error must fall with more steps"


def test_every_column_is_its_own_authored_optical_depth(deck: CloudDeck) -> None:
    """Not only the one overhead: a vertical ray anywhere reads that column's depth."""
    offsets = np.array([-0.3, -0.1, 0.0, 0.1, 0.3]) * deck.half_extent_m
    x, z = np.meshgrid(offsets, offsets, indexing="ij")
    el = np.full(x.shape, 0.5 * math.pi)
    az = np.zeros(x.shape)
    marched = deck.march(el, az, origin_m=(0.0, 0.0, 0.0), steps=256)
    # Marching from the origin straight up only samples the centre column, so move the origin.
    for i in range(x.size):
        ox, oz = float(x.flat[i]), float(z.flat[i])
        one = deck.march(np.full(1, 0.5 * math.pi), np.zeros(1), origin_m=(ox, 0.0, oz), steps=256)
        depth = float(deck.column_depth(ox, oz)[()])
        # A column thinner than a few steps is not a physics case, it is an empty quadrature: the
        # march spaces 256 steps over the deck's *full* 1.2 km, so a column filling 1% of it has
        # two samples in it and a column filling 0.01% has none. Checked where the march can see.
        if depth < 0.05:
            continue
        expected = deck.optical_depth * depth
        # 3e-3 rather than the 1e-4 the deepest column reaches, and the difference is quadrature
        # rather than physics: a column that only fills a third of the deck is resolved by a third
        # of the steps, and the profile's truncation at the column top lands between two of them.
        assert float(one.optical_depth[0]) == pytest.approx(expected, rel=3e-3, abs=1e-6)
    assert marched.optical_depth.shape == x.shape  # the vectorised path has the right shape


# -- what the third dimension actually buys ------------------------------------------------


def test_an_oblique_ray_crosses_columns_instead_of_staying_in_one(deck: CloudDeck) -> None:
    """The whole point. A plane-parallel sheet gives τ = OD·d(entry)·airmass and nothing else.

    Measured against that: the marched depth must differ from the sheet's answer, because a ray
    at 20 degrees travels kilometres horizontally and meets other columns. If it agreed, the deck
    would be a sheet with extra arithmetic.
    """
    el, az = _frame_angles()
    marched = deck.march(el, az, steps=64).optical_depth
    # The sheet's answer for the same rays: the column the ray enters, times its airmass.
    entry_t = deck.base_m / np.sin(el)
    direction = _direction(el, az, (0.0, 1.0, 0.0), (0.0, 0.0, -1.0))
    sheet = (
        deck.optical_depth
        * deck.column_depth(entry_t * direction[..., 0], entry_t * direction[..., 2])
        / np.sin(el)
    )
    lit = (marched > 0.1) | (sheet > 0.1)
    assert lit.mean() > 0.2, "the test frame must contain cloud"
    disagreement = np.abs(marched[lit] - sheet[lit]) / np.maximum(sheet[lit], 1e-6)
    assert float(np.median(disagreement)) > 0.1


def test_the_emission_level_varies_across_a_frame(deck: CloudDeck) -> None:
    """Where a ray stops emitting is the structure a sheet cannot have, so it must actually vary.

    Bounded by the deck as well as spread: an emission height outside [0, thickness] would mean
    the march is integrating outside the cloud, which a pure spread test would not catch.
    """
    el, az = _frame_angles()
    march = deck.march(el, az, steps=64)
    lit = march.optical_depth > 0.5
    heights = march.emission_height_m[lit]
    assert lit.mean() > 0.2
    assert heights.min() >= 0.0 and heights.max() <= deck.thickness_m
    assert float(heights.max() - heights.min()) > 200.0


def test_a_ray_that_never_reaches_the_deck_takes_nothing_from_it(deck: CloudDeck) -> None:
    """Pointing down or along the horizon: zero, so the blend falls back to clear sky exactly."""
    below = deck.march(np.radians(np.array([-30.0, -1.0, 0.0])), np.zeros(3))
    assert np.array_equal(below.optical_depth, np.zeros(3))
    assert np.array_equal(below.emission_height_m, np.zeros(3))


def test_the_deck_has_no_edge_because_it_tiles(deck: CloudDeck) -> None:
    """A ray shallower than the tile's own elevation still finds cloud, and finds it seamlessly.

    The deck used to stop at ``base / tan(10 deg)`` and read clear sky beyond, which put a hard
    straight cold band across the bottom of every oblique frame -- exactly where a real cumulus
    field puts its densest wall, because that is the direction with the longest slant path. The
    depth map is an inverse FFT and therefore exactly periodic, so wrapping the lookup costs
    nothing and is continuous to the bit.
    """
    grazing = deck.march(
        np.radians(np.full(1, 0.5 * DEFAULT_MIN_ELEVATION_DEG)), np.zeros(1), steps=64
    )
    assert float(grazing.optical_depth[0]) > 1.0, "a 5 degree ray runs the length of the deck"
    # Seamless: one period apart is the same column, not merely a similar one.
    period = deck.depth.shape[0] * deck.cell_m
    x = np.array([-311.0, 0.0, 517.0, 2400.0])
    z = np.array([77.0, -1290.0, 640.0, -33.0])
    assert np.allclose(deck.column_depth(x, z), deck.column_depth(x + period, z - period), atol=0.0)
    # And continuous across the seam: the step over the wrap is no larger than a typical step.
    edge = 0.5 * period
    walk = np.linspace(edge - 5 * deck.cell_m, edge + 5 * deck.cell_m, 401)
    jump = np.abs(np.diff(deck.column_depth(walk, np.zeros_like(walk))))
    inland = np.abs(np.diff(deck.column_depth(walk - 0.25 * period, np.zeros_like(walk))))
    assert jump.max() <= 4.0 * max(float(inland.max()), 1e-6)


def test_the_emissivity_is_the_same_beer_lambert_law_as_the_sheet(deck: CloudDeck) -> None:
    """Only the optical depth changed; the law that turns it into an emissivity did not."""
    march = deck.march(*_frame_angles(64, 80), steps=32)
    assert np.allclose(
        march.emissivity(), 1.0 - np.exp(-CLOUD_OD_RATIO * march.optical_depth), atol=0.0
    )


# -- the two bands read one object ----------------------------------------------------------


def test_the_voxel_grid_is_the_marched_function_sampled(deck: CloudDeck) -> None:
    """The axis order and the origin, which is where a VDB goes wrong and looks plausible.

    A transposed grid still renders a cloud; it renders it somewhere the infrared band's cloud is
    not, and the pair would disagree in a way no single frame reveals. So the check walks a few
    voxels by index, turns each into a world point through the returned `min_world` and the voxel
    size, and asks `density_at` — the function the march integrates — for the same number.
    """
    voxel = 50.0
    grid, origin = deck.voxels(voxel)
    assert grid.dtype == np.float32
    assert grid.ndim == 3 and grid.shape[0] == grid.shape[2]
    rng = np.random.default_rng(11)
    idx = np.stack([rng.integers(0, n, 400) for n in grid.shape], axis=-1)
    world = origin + idx * voxel
    expected = deck.density_at(world[:, 0], world[:, 1], world[:, 2])
    assert np.allclose(grid[idx[:, 0], idx[:, 1], idx[:, 2]], expected, rtol=1e-6, atol=1e-12)
    assert float(grid.max()) > 0.0, "an all-zero volume renders as clear sky and passes everything"


def test_the_deck_field_is_cloud_shaped_rather_than_one_bank(deck: CloudDeck) -> None:
    """Cloud-scale structure with the largest scales taken out, and the authored coverage kept.

    The taper is the parameter that makes a cloud a cloud: a pure 1/f^beta field puts most of its
    variance at the largest scale available, so without it the whole twelve-kilometre footprint
    comes out as a single bank and every ray in a frame reads the same thing. This measures the
    autocorrelation length of the depth map along one axis and asks it to be cumulus-sized rather
    than footprint-sized.
    """
    d = deck.depth - deck.depth.mean()
    line = d[d.shape[0] // 2]
    auto = np.correlate(line, line, mode="full")[line.size - 1 :]
    auto = auto / auto[0]
    # First lag where the correlation falls through 1/e, in metres.
    below = np.nonzero(auto < math.exp(-1.0))[0]
    assert below.size, "the depth map never decorrelates: the field is one bank"
    length_m = float(below[0]) * deck.cell_m
    assert 500.0 < length_m < 4000.0, f"{length_m:.0f} m is not a cumulus"
    # And a cloud is about as wide as it is tall, which is what stops an oblique ray running
    # along a fin for kilometres instead of crossing a cloud.
    assert 0.2 < deck.thickness_m / length_m < 5.0


def test_the_deck_keeps_the_authored_coverage_exactly() -> None:
    """The threshold still means coverage -- and now means it *exactly*, which it did not before.

    The depth map is a ramp that is zero at the threshold and positive above it, so the set of
    columns carrying any cloud at all is the set above the quantile, to within the tie-breaking
    of a single cell. The old smoothstep spread cloud half its softness *past* the threshold, so
    an authored 0.45 came out as covered area somewhere above it.
    """
    for fraction in (0.2, 0.45, 0.7):
        depth = deck_field(257, beta=2.8, cloud_fraction=fraction, seed=3)
        assert float((depth > 0.0).mean()) == pytest.approx(fraction, abs=2e-3)
        # A ramp, not a switch: covered columns take a range of depths rather than one.
        covered = depth[depth > 0.0]
        assert float(np.percentile(covered, 90) - np.percentile(covered, 10)) > 0.4
    assert psd_slope(deck_field(257, beta=2.8, cloud_fraction=0.99, seed=3)) < -1.5


def test_the_deck_carries_no_cloud_smaller_than_a_cloud() -> None:
    """The band limit, which is what stopped the infrared frame filling with marks.

    The two bands read the deck differently -- the dome samples it where a ray crosses the base,
    the infrared band integrates it along the ray -- so structure far below the cloud scale does
    not merely look wrong, it looks wrong *differently in each band*: a 25 m speck stays a speck
    on the dome and smears over kilometres in the march. Measured as the power left above the
    band limit's own frequency, which is where the roll-off is defined, not as an eyeball.
    """
    cell_m = 25.0
    smooth = deck_field(
        487,
        beta=2.8,
        cloud_fraction=0.45,
        seed=7,
        smooth_cells=DEFAULT_SMALLEST_CLOUD_M / 4 / cell_m,
    )
    raw = deck_field(487, beta=2.8, cloud_fraction=0.45, seed=7)

    def power_between(field: np.ndarray, small_m: float, large_m: float) -> float:
        """Absolute power in the band of spatial scales [small_m, large_m], per unit area."""
        p = np.abs(np.fft.fft2(field - field.mean())) ** 2 / field.size**2
        fy = np.fft.fftfreq(field.shape[0])[:, None]
        fx = np.fft.fftfreq(field.shape[1])[None, :]
        f = np.sqrt(fx * fx + fy * fy)
        band = (f >= cell_m / large_m) & (f < cell_m / small_m)
        return float(p[band].sum())

    below = DEFAULT_SMALLEST_CLOUD_M
    loose = power_between(raw, 0.0 + 2 * cell_m, below)
    tight = power_between(smooth, 0.0 + 2 * cell_m, below)
    assert loose > 0.0, "the premise of the limit is gone: the raw field has no small scales"
    assert tight < 0.1 * loose, "the sub-cloud scales must be gone, not merely reduced"
    # And the limit must not eat the clouds themselves: the kilometre band survives.
    assert power_between(smooth, 1000.0, 4000.0) == pytest.approx(
        power_between(raw, 1000.0, 4000.0), rel=0.15
    )


def test_reading_beta_as_a_transect_slope_is_what_makes_a_cloud_a_cloud() -> None:
    """The correction, measured rather than asserted, because it is the one judgement call here.

    `generate_cloud_field` applies `beta` as the *radial* exponent, and on a 2-D field the
    variance per octave goes as f^(2 - beta) -- so the preset's authored 1.8 puts most of the
    variance at the smallest scale the grid has and synthesises texture, not cumulus. Published
    cloud slopes near -5/3 are transect slopes, one less than the radial exponent, so the deck
    adds one. This measures both and demands they differ by an order of magnitude.
    """

    def correlation_m(beta: float) -> float:
        field = deck_field(487, beta=beta, cloud_fraction=0.45, seed=7)
        line = field[243] - field.mean()
        auto = np.correlate(line, line, mode="full")[line.size - 1 :]
        below = np.nonzero(auto / auto[0] < math.exp(-1.0))[0]
        return float(below[0]) * 25.0 if below.size else float("inf")

    as_authored = correlation_m(1.8)
    corrected = correlation_m(1.8 + TRANSECT_TO_RADIAL_SLOPE)
    assert as_authored < 300.0, f"{as_authored:.0f} m -- the premise of the correction is gone"
    assert corrected > 3.0 * as_authored


def test_a_deck_is_reproducible_from_its_seed() -> None:
    a = generate_cloud_deck(beta=1.8, cloud_fraction=0.45, seed=7, base_m=1000.0, optical_depth=9.0)
    b = generate_cloud_deck(beta=1.8, cloud_fraction=0.45, seed=7, base_m=1000.0, optical_depth=9.0)
    c = generate_cloud_deck(beta=1.8, cloud_fraction=0.45, seed=8, base_m=1000.0, optical_depth=9.0)
    assert np.array_equal(a.depth, b.depth)
    assert not np.array_equal(a.depth, c.depth)
    assert psd_slope(a.depth) < 0.0


def test_the_direction_helper_inverts_the_angle_helper() -> None:
    """The march and the deck's construction use opposite halves of one convention."""
    rng = np.random.default_rng(5)
    el = np.radians(rng.uniform(1.0, 89.0, 256))
    az = rng.uniform(0.0, 2.0 * math.pi, 256)
    back_el, back_az = sky_angles(_direction(el, az, (0.0, 1.0, 0.0), (0.0, 0.0, -1.0)))
    assert np.allclose(back_el, el, atol=1e-12)
    assert np.allclose(np.mod(back_az, 2 * math.pi), np.mod(az, 2 * math.pi), atol=1e-12)


def test_a_deck_refuses_what_it_cannot_represent() -> None:
    depth = np.zeros((9, 9))
    with pytest.raises(ValueError, match="square grid"):
        CloudDeck(np.zeros((9, 8)), 25.0, 1000.0, 1200.0, 12.0)
    with pytest.raises(ValueError, match="odd size"):
        CloudDeck(np.zeros((8, 8)), 25.0, 1000.0, 1200.0, 12.0)
    with pytest.raises(ValueError, match="must be positive"):
        CloudDeck(depth, 25.0, 1000.0, 1200.0, 0.0)
    with pytest.raises(ValueError, match="fog"):
        generate_cloud_deck(beta=1.8, cloud_fraction=0.4, seed=1, base_m=0.0, optical_depth=5.0)
    with pytest.raises(ValueError, match="odd and at least"):
        deck_field(64, beta=1.8, cloud_fraction=0.4, seed=1)


# -- into the radiance -----------------------------------------------------------------------


def test_the_frame_gains_real_structure_where_the_sheet_had_a_kelvin(
    scene: Scene, deck: CloudDeck
) -> None:
    """The deliverable, measured on the clip's own frame geometry.

    AT.11's sheet spans 1.25 K across the cloud core of this frame; the roadmap row asks the deck
    for at least 5 K, because that is the difference between a flat blob and a cloud you can see
    the shape of. Both are computed here from the same field and the same scene, so nothing but
    the geometry can account for the gap.
    """
    sky = scene.sky_models[BAND]
    el, az = _frame_angles(256, 320)
    env = scene.environment
    assert env is not None
    field = generate_sky_cloud(
        env.clouds.beta, float(scene.weather.at(scene.t0_s).cloud_fraction), 7
    )
    sheet = np.asarray(sky.apparent_temperature_field(scene.t0_s, el, field.density(el, az)))
    marched = np.asarray(
        sky.apparent_temperature_field_from_deck(scene.t0_s, el, az, deck, steps=64)
    )
    # **Each model's own cloudy pixels.** The two do not agree on which rays are covered, and
    # that disagreement is the result, not an error: a sheet asks only which column a ray enters,
    # while the deck asks what the ray met on the way through. Measuring the sheet over the deck's
    # mask would score it on pixels it calls clear sky and report a 42 K "spread" that is the
    # sky-to-cloud step, not cloud structure.
    sheet_cloudy = field.density(el, az) >= 1.0
    deck_cloudy = deck.march(el, az, steps=64).optical_depth > 1.0
    assert sheet_cloudy.mean() > 0.2 and deck_cloudy.mean() > 0.2
    spread = lambda a, m: float(np.percentile(a[m], 99) - np.percentile(a[m], 1))  # noqa: E731
    sheet_spread = spread(sheet, sheet_cloudy)
    deck_spread = spread(marched, deck_cloudy)
    assert sheet_spread < 3.0, f"the sheet was supposed to be flat: {sheet_spread:.2f} K"
    assert deck_spread > 5.0, f"the deck's cloud spans only {deck_spread:.2f} K"
    # And it is colder on average, not merely different: emission from higher in a tower is
    # emission from colder air, so the deck can only move the cloud one way.
    assert float(np.mean(marched[deck_cloudy])) < float(np.mean(sheet[sheet_cloudy]))


def test_the_mean_emission_height_is_a_fair_stand_in_for_the_full_integral(
    scene: Scene, deck: CloudDeck
) -> None:
    """The march reports one height and the radiance is evaluated there; Planck is convex in T.

    Bounded rather than asserted, and the bound is not negligible: against a per-step Planck
    integration the error is **64 mK at the worst pixel and 46 mK at the 99th percentile**, which
    is about one NETD on roughly one cloudy pixel in a hundred. It lands on rays that graze a
    tower top, where the emission is spread over the widest range of heights and Planck's
    curvature has the most room to bite.

    Kept, rather than fixed, because the alternative is a band lookup inside the march -- 48 per
    frame instead of one, about 0.25 s per frame with a precomputed height/radiance table -- and
    the quantity this deck exists to produce is 20 K of cloud structure. The number is recorded
    here so that a later step that needs sub-NETD sky can find the cost already measured.
    """
    sky = scene.sky_models[BAND]
    el, az = _frame_angles(48, 64)
    lapse = sky.atmosphere.preset.profile.lapse_rate_k_per_m
    t_base = sky.cloud_base_temperature_k(scene.t0_s)
    steps = 64

    # Oracle: accumulate emission per step in radiance, which is what the linearisation skips.
    direction = _direction(el, az, (0.0, 1.0, 0.0), (0.0, 0.0, -1.0))
    t_in = deck.base_m / np.maximum(direction[..., 1], 1e-9)
    ds = (deck.thickness_m / np.maximum(direction[..., 1], 1e-9)) / steps
    tau = np.zeros(el.shape)
    emitted = np.zeros(el.shape)
    for k in range(steps):
        t = t_in + (k + 0.5) * ds
        py = t * direction[..., 1]
        d_tau = deck.density_at(t * direction[..., 0], py, t * direction[..., 2]) * ds
        w = np.exp(-CLOUD_OD_RATIO * tau) * (1.0 - np.exp(-CLOUD_OD_RATIO * d_tau))
        emitted += w * np.asarray(sky.lut.lookup(t_base - lapse * (py - deck.base_m), sky.quantity))
        tau += d_tau
    march = deck.march(el, az, steps=steps)
    eps = march.emissivity()
    lin = eps * np.asarray(sky.lut.lookup(t_base - lapse * march.emission_height_m, sky.quantity))
    lit = eps > 0.05
    err_k = np.abs(
        np.asarray(sky.lut.apparent_temperature(emitted[lit] / eps[lit], sky.quantity))
        - np.asarray(sky.lut.apparent_temperature(lin[lit] / eps[lit], sky.quantity))
    )
    assert float(np.percentile(err_k, 99)) < 0.05, "the 99th percentile must stay under a NETD"
    assert float(err_k.max()) < 0.10, f"linearisation costs {err_k.max():.4f} K at worst"


def test_the_march_sizes_itself_from_the_geometry(deck: CloudDeck) -> None:
    """A fixed step count is right for one elevation and aliases at every other one.

    An oblique ray crosses the depth map horizontally -- at 20 degrees a ray climbing 1.2 km
    travels 3.3 km sideways -- so it needs far more samples than a vertical one. Undersampling it
    put horizontal bands along every cloud edge in the preview frames, which is why this is
    derived rather than authored.
    """
    zenith = deck.adequate_steps(np.array([0.5 * math.pi]))
    oblique = deck.adequate_steps(np.radians(np.array([20.0])))
    shallow = deck.adequate_steps(np.radians(np.array([5.0])))
    assert MIN_MARCH_STEPS <= zenith < oblique <= MAX_MARCH_STEPS
    assert shallow == MAX_MARCH_STEPS  # the cap, which is a cost decision and is stated as one
    # Vertically the path *is* the thickness, so the count is that over the step length.
    assert zenith == pytest.approx(deck.thickness_m / MARCH_STEP_M, rel=0.05)
    # The lowest ray in the array sets it, because the step count is one number for the march.
    el, _ = _frame_angles(64, 64)
    assert deck.adequate_steps(el) == deck.adequate_steps(np.array([el.min()]))


def test_a_finer_march_changes_the_answer_less_and_less(deck: CloudDeck) -> None:
    """Convergence, so the cap is a measured cost and not a hope."""
    el, az = _frame_angles(64, 80)
    coarse = deck.march(el, az, steps=48).optical_depth
    mid = deck.march(el, az, steps=128).optical_depth
    fine = deck.march(el, az, steps=MAX_MARCH_STEPS).optical_depth
    assert float(np.abs(mid - fine).mean()) < float(np.abs(coarse - fine).mean())


def test_the_dome_bakes_the_deck_when_the_infrared_band_is_marching_one() -> None:
    """One object in both bands, still (ADR 0076), now that the object has a top.

    A deck in the infrared band beside the hemispherical field on the dome would put cloud in
    different parts of the sky in the two halves of a frame pair -- each internally consistent and
    plausible, which is the worst kind of wrong. The dome takes the deck's own column depth where
    each ray crosses the base.
    """
    from irsim_isaac.visible_sky import DomeSpec, environment_map, latlong_directions

    deck = generate_cloud_deck(
        beta=2.8, cloud_fraction=0.45, seed=7, base_m=1000.0, optical_depth=12.0
    )
    base = DomeSpec(sun_elevation_deg=40.0, sun_azimuth_deg=150.0, dni_w_m2=800.0, dhi_w_m2=180.0)
    clear = environment_map(base, height=64)
    clouded = environment_map(
        DomeSpec(**{**base.__dict__, "deck": deck}),
        height=64,
    )
    up = latlong_directions(64)[..., 1]
    moved = np.any(np.abs(clouded - clear) > 1e-6, axis=-1)
    assert moved[up > 0.0].mean() > 0.1, "the dome drew no cloud at all"
    # Below the horizon is terrain and must be untouched: a deck is above the camera.
    assert not moved[up < 0.0].any()
    # And the cloud lands exactly where the *infrared* band's cloud is -- which is the whole
    # point of ADR 0076 and the thing that was wrong: the dome used to sample the column depth
    # where the ray crosses the base, a **vertical** thickness read for an **oblique** look, so
    # it drew a few thin wisps over the same sky in which the march found a wall of cumulus.
    directions = latlong_directions(64)
    # Low in the sky, which is where the two readings diverge and where these clips point: a
    # vertical thickness and a slant path through a 1.2 km deck agree near the zenith by
    # construction and disagree by a factor of several at 10 degrees.
    elevated = (directions[..., 1] > 0.05) & (directions[..., 1] < 0.35)
    el = np.arcsin(np.clip(directions[..., 1][elevated], -1.0, 1.0))
    az = np.arctan2(directions[..., 0][elevated], -directions[..., 2][elevated])
    marched = deck.march(el, az).optical_depth
    assert np.array_equal(moved[elevated], marched > 0.0)
    # Not a repeat of the same arithmetic: the sampled answer really is a different cloud, so a
    # dome that agreed with *it* would be the bug this asserts against.
    t = deck.base_m / directions[..., 1][elevated]
    sampled = deck.column_depth(t * directions[..., 0][elevated], t * directions[..., 2][elevated])
    assert float((marched > 0.0).mean()) > 1.5 * float((sampled > 0.0).mean())


def test_marching_once_per_native_pixel_and_interpolating_is_the_same_picture(
    scene: Scene, deck: CloudDeck
) -> None:
    """The saving that makes a supersampled frame affordable, and the bound on what it costs.

    An infrared camera's AOVs are rendered at four times native so that *geometry* edges
    antialias: sixteen times the rays. The cloud behind that geometry has no geometry, so the
    march runs once per native pixel and is interpolated back. This measures the difference
    against marching every ray -- on the sky, which is all this path ever supplies.

    The bound is loose on purpose, and the reason is the measurement that set `MARCH_STEP_M`: over
    a whole frame, compared against a converged reference rather than against another march, the
    error is dominated by the march's own **quadrature** and not by this interpolation. Halving
    the stride moves the 99th percentile of the band emissivity from 0.029 to 0.024 and costs five
    times as much, so the steps are spent on the quadrature instead. What this test is for is
    catching the interpolation going *wrong* -- a transposed axis, an off-by-one in the corner
    alignment -- which shows up as a gross error, not as a tenth of a kelvin.
    """
    sky = scene.sky_models[BAND]
    # A patch of the real frame at the real **supersampled** pitch -- the sensor's 0.857 mrad IFOV
    # over four samples -- rather than a coarse stand-in, because the whole question is how much
    # the field moves between adjacent supersamples.
    pitch = 0.8571e-3 / 4.0
    rows, cols = 512, 640
    el = math.radians(AIM_ELEVATION_DEG) + (np.arange(rows) - 0.5 * rows)[
        :, None
    ] * pitch * np.ones((1, cols))
    az = (np.arange(cols) - 0.5 * cols)[None, :] * pitch * np.ones((rows, 1))
    full = np.asarray(sky.apparent_temperature_field_from_deck(scene.t0_s, el, az, deck))
    coarse = np.asarray(
        sky.apparent_temperature_field_from_deck(scene.t0_s, el[::4, ::4], az[::4, ::4], deck)
    )
    upsampled = resample_bilinear(coarse, el.shape)
    err = np.abs(upsampled - full)
    # Stated rather than hidden: this is an approximation and it has a size, and the size is set
    # by the cloud edges, where 40 K crosses a pixel. The median is the load-bearing half -- it
    # says the interpolation is right almost everywhere, which is what a transposed axis or a
    # misaligned corner would break.
    assert float(np.percentile(err, 99)) < 2.0
    assert float(np.median(err)) < 0.1


def test_resampling_leaves_an_unchanged_shape_alone_and_keeps_the_corners() -> None:
    a = np.arange(12.0).reshape(3, 4)
    assert np.array_equal(resample_bilinear(a, (3, 4)), a)
    big = resample_bilinear(a, (9, 16))
    assert big.shape == (9, 16)
    for (i, j), (bi, bj) in (
        ((0, 0), (0, 0)),
        ((0, 3), (0, 15)),
        ((2, 0), (8, 0)),
        ((2, 3), (8, 15)),
    ):
        assert big[bi, bj] == pytest.approx(a[i, j], abs=1e-12)
    assert np.all(np.diff(big, axis=1) >= -1e-12)  # monotone in, monotone out
    with pytest.raises(ValueError, match="2-D array"):
        resample_bilinear(np.zeros(4), (2, 2))
