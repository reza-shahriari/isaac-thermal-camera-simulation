"""Structured cloud in the rendered background (MS.3/ADR 0070 into the M10.18 bridge).

`SkyModel.radiance` has always carried cloud, but as its **expectation**: the uniform blend
(1 - c eps) L_clear + c eps L_base, which is the mean over a structured field and is exactly right
for a LUT or a tilt integral. What a rendered frame needs is the field itself, because against a
sky background the dominant false alarm is not sensor noise, it is a cloud edge -- and an edge has
no mean.

So the property that makes this an upgrade rather than a different model is that **the mean is
preserved**: averaged over the whole sky the structured field returns the uniform blend it
replaced. If that failed, every existing LUT, tilt integral and reflected term would silently
disagree with the rendered background.

The second property is that the field is fixed to the **sky**. A field fixed to the image plane is
equally stable frame to frame and travels with the sensor, so a slewing mount never sweeps across
cloud and a tracked target never crosses one -- which removes the very clutter the field is for.
"""

from __future__ import annotations

import math
import pathlib

import numpy as np
import pytest

from irsim.atmosphere.cloud import generate_sky_cloud
from irsim.config.loader import load_sensor_config
from irsim.radiometry.lut_files import load_band_lut_for_config
from irsim.scene import Scene
from irsim_isaac.pipeline.aerial_bridge import (
    AerialThermalBridge,
    azimuth_from_rays,
    elevation_from_rays,
)

REPO = pathlib.Path(__file__).resolve().parents[2]
SCENE_YAML = REPO / "configs" / "scenes" / "quad_flight_clear_noon.yaml"
SENSOR_YAML = REPO / "configs" / "sensors" / "flir_boson_640_lwir.yaml"

#: The structured field's mean must agree with the uniform blend to better than this, over the
#: whole sky. It is not exact: coverage is clumped and the clear radiance varies with elevation,
#: so which elevations a realisation happens to cover moves the mean slightly. 1 % is far inside
#: the ~0.2 % observed across seeds and far outside anything a wrong model would achieve.
MEAN_PRESERVATION_TOL = 0.01


@pytest.fixture(scope="module")
def scene() -> Scene:
    sensor = load_sensor_config(SENSOR_YAML)
    lut = load_band_lut_for_config(sensor, REPO / "data" / "lut")
    return Scene.from_file(SCENE_YAML, {"lwir": lut})


def whole_sky(n_el: int = 180, n_az: int = 720) -> tuple[np.ndarray, np.ndarray]:
    el = np.radians((np.arange(n_el) + 0.5) * 90.0 / n_el)[:, None].repeat(n_az, 1)
    az = np.radians((np.arange(n_az) + 0.5) * 360.0 / n_az)[None, :].repeat(n_el, 0)
    return el, az


# --- the field --------------------------------------------------------------------------------


@pytest.mark.parametrize("fraction", [0.05, 0.2, 0.5])
def test_the_sky_coverage_is_the_cloud_fraction_it_was_asked_for(fraction: float) -> None:
    """Within 15 % relative, on a densely sampled sky, from the shared weather's own number.

    Not exact, and the reason is recorded rather than tuned away. The field is thresholded at a
    level estimated on a 2x upsampling, because sampling *between* grid cells averages neighbours
    and a level cut on the grid itself covers about 25 % less of a densely sampled sky -- a bias
    that does not improve with resolution, since a 1/f^beta field is scale-invariant and bilinear
    averaging smooths it equally at every scale. The estimate leaves a residual that is largest
    deep in the tail: about 8 % at c = 0.05 and under 3 % by c = 0.2.

    15 % of 0.05 is 0.0075 of sky. The input is a weather file's cloud fraction, conventionally
    reported in oktas -- eighths, so 0.125 of granularity. The residual is an order of magnitude
    inside the precision of the number being matched, and tightening it would be fitting to a
    figure that was never that sharp.
    """
    cloud = generate_sky_cloud(1.8, fraction, seed=11)
    el, az = whole_sky(600, 2400)
    assert float(cloud.sample(el, az).mean()) == pytest.approx(fraction, rel=0.15)


def test_a_clear_sky_has_no_cloud_at_all(scene: Scene) -> None:
    """c = 0 must give exactly nothing, not a few stray pixels from a tail estimate."""
    cloud = generate_sky_cloud(scene.environment.clouds.beta, 0.0, seed=11)
    el, az = whole_sky(120, 480)
    assert not cloud.sample(el, az).any()


def test_cloud_edges_are_resolved_at_the_sampling_scale_not_the_grid(scene: Scene) -> None:
    """Interpolating the field before thresholding is what stops edges being grid-sized blocks.

    A Boson pixel is 0.049 degrees and a grid cell is half a degree, so sampling a stored *mask*
    nearest-neighbour gives cloud edges that are ten-pixel rectangular steps -- which is what the
    first render of this looked like. Edge sharpness is exactly what a detector keys on, so a
    blocky edge is not a cosmetic problem.

    Measured as the number of distinct transitions along a finely sampled arc: a mask sampled at
    grid resolution can only change where a cell boundary falls, so its transitions land on a
    coarse lattice. The interpolated field crosses its threshold wherever it likes.
    """
    cloud = generate_sky_cloud(scene.environment.clouds.beta, 0.3, seed=5)
    az = np.radians(np.linspace(0.0, 30.0, 4000))
    el = np.radians(np.full(4000, 25.0))
    covered = cloud.sample(el, az)
    edges = np.flatnonzero(np.diff(covered.astype(np.int8)) != 0)
    assert edges.size > 2, "need several edges along this arc for the check to mean anything"
    # Grid cells are 0.5 deg; over a 30 deg arc of 4000 samples that is one cell per 66.7 samples.
    # If edges only ever fell on cell boundaries their spacings would all be multiples of that.
    spacings = np.diff(edges)
    off_lattice = np.mod(spacings, 66.7)
    assert np.any((off_lattice > 5.0) & (off_lattice < 61.7)), (
        "every edge fell on a grid-cell boundary; the mask is being sampled, not the field"
    )


def test_one_elevation_ring_is_not_the_sky_average(scene: Scene) -> None:
    """Cloud clumps: a single ring departs from c, which is the point of having structure at all.

    A field whose every ring carried exactly c would be a haze, not cloud, and would never give a
    detector an edge to false-alarm on.
    """
    fraction = float(scene.weather.at(scene.t0_s).cloud_fraction)
    cloud = generate_sky_cloud(scene.environment.clouds.beta, fraction, seed=11)
    rings = [
        float(cloud.sample(np.radians(np.full(720, e)), np.radians(np.arange(720) * 0.5)).mean())
        for e in (10.0, 25.0, 40.0, 60.0)
    ]
    assert max(rings) > min(rings) + 0.01, f"rings {rings} are suspiciously uniform"


def test_the_structured_field_preserves_the_uniform_blend_it_replaces(scene: Scene) -> None:
    """Averaged over the sky, the field returns the expectation every other consumer uses.

    This is the compatibility property. `SkyModel.radiance` feeds the elevation LUT, the tilt
    integral and ADR 0045's reflected term; if the rendered background's mean drifted from it, the
    same sky would be two different skies depending on which code path asked.
    """
    sky = scene.sky_models["lwir"]
    t = scene.t0_s
    fraction = float(scene.weather.at(t).cloud_fraction)
    el, az = whole_sky()
    for seed in (1, 2, 3, 11):
        coverage = generate_sky_cloud(scene.environment.clouds.beta, fraction, seed).sample(el, az)
        structured = sky.radiance_field(t, el.ravel(), coverage.ravel())
        # Against the blend at the coverage this realisation *actually* has, not at the fraction
        # it was asked for. Those differ by a few percent in the tail (see the coverage test), and
        # folding that in here would be testing two properties at once and pinning neither: the
        # question here is whether structure and mean agree, not how accurately the threshold
        # hits a quantile.
        realised = float(coverage.mean())
        blend = (
            sky.radiance_field(t, el.ravel(), np.zeros(el.size, dtype=bool)) * (1.0 - realised)
            + sky.radiance_field(t, el.ravel(), np.ones(el.size, dtype=bool)) * realised
        )
        relative = abs(float(structured.mean() - blend.mean()) / float(blend.mean()))
        assert relative < MEAN_PRESERVATION_TOL, f"seed {seed}: mean drifted by {relative:.2%}"


def test_covered_pixels_read_warmer_than_clear_ones(scene: Scene) -> None:
    """Cloud base sits far above a cold clear zenith, so cloud is a *positive* contrast feature.

    Worth pinning: a drone is also warmer than the clear sky, so cloud and target have the same
    polarity and cloud edge is a genuine confuser rather than something a sign test removes.
    """
    sky = scene.sky_models["lwir"]
    t = scene.t0_s
    el = np.radians(np.full(4000, 45.0))
    covered = np.ones(4000, dtype=bool)
    clear = np.zeros(4000, dtype=bool)
    warm = sky.apparent_temperature_field(t, el, covered)
    cold = sky.apparent_temperature_field(t, el, clear)
    assert float(warm.mean()) > float(cold.mean()) + 10.0


# --- the bridge -------------------------------------------------------------------------------


def bridge(scene: Scene, **kwargs: object) -> AerialThermalBridge:
    return AerialThermalBridge(scene, {}, band="lwir", **kwargs)  # type: ignore[arg-type]


def test_without_a_seed_the_background_is_unchanged(scene: Scene) -> None:
    """Bit-identical to the uniform blend: every existing frame and golden stays exactly as it was.

    The structured path is opt-in for this reason. Turning it on by default would have moved every
    committed background by a few kelvin with no commit saying so.
    """
    el, az = whole_sky(40, 80)
    plain = bridge(scene)
    assert plain.cloud is None
    assert np.array_equal(
        plain.background_temperature_k(el, az), plain.background_temperature_k(el, None)
    )


def test_a_seed_puts_structure_in_the_background(scene: Scene) -> None:
    el, az = whole_sky(60, 120)
    plain = bridge(scene).background_temperature_k(el, az)
    seeded = bridge(scene, cloud_seed=11).background_temperature_k(el, az)
    assert not np.allclose(plain, seeded)
    # Cloud is warmer than the blend where it is and colder where the gap is -- per pixel. The
    # two frames' *maxima* are no longer the comparison: both sit at the horizon, where a base
    # read through the air in front of it and the clear column meet (ADR 0126 on the blend,
    # ADR 0146), so the tail has to be looked for against the same pixel.
    difference = seeded - plain
    assert float(difference.max()) > 1.0, "cloud should add a warm tail"
    assert float(difference.min()) < -1.0, "and the gaps between clouds a cold one"


def test_the_field_is_fixed_to_the_sky_not_to_the_frame(scene: Scene) -> None:
    """The same world direction gives the same cloud however the camera happens to be pointed.

    Sampled here as two overlapping windows of the sky: where they overlap they must agree. An
    image-plane field would instead give whatever its own pixel grid said, so the overlap would
    disagree and a slewing mount would drag its clouds along with it.
    """
    cloudy = bridge(scene, cloud_seed=11)
    el = np.radians(np.linspace(10.0, 50.0, 64))[:, None].repeat(64, 1)
    # Two windows offset by a whole number of *samples*, so the overlapping columns ask about
    # identical world directions. Offsetting by an arbitrary angle instead would compare
    # neighbouring directions and fail on the field's own resolution rather than on its frame.
    step_deg = 0.5
    columns = np.arange(64) * step_deg
    az_a = np.radians(columns)[None, :].repeat(64, 0)
    az_b = np.radians(columns + 32 * step_deg)[None, :].repeat(64, 0)
    first = cloudy.background_temperature_k(el, az_a)
    second = cloudy.background_temperature_k(el, az_b)
    assert np.allclose(first[:, 32:], second[:, :32], atol=1e-9)
    # ...and the windows are not trivially identical: they overlap by half, not wholly.
    assert not np.allclose(first, second, atol=1e-9)


def test_cloud_without_a_sky_model_is_refused(scene: Scene) -> None:
    """Both terms of a covered pixel come from the sky model; a seed alone cannot make cloud."""
    with pytest.raises(ValueError, match="cloud needs a sky model"):
        AerialThermalBridge(scene, {}, cloud_seed=11)


def test_cloud_without_azimuth_falls_back_rather_than_banding(scene: Scene) -> None:
    """Sampling the field by elevation alone would stripe the sky horizontally. It does not.

    A caller that has not got azimuth gets the uniform blend, which is merely less detailed. The
    failure this avoids is the one that looks deliberate: horizontal bands across the sky that a
    reader would take for a real atmospheric layer.
    """
    el, az = whole_sky(40, 80)
    cloudy = bridge(scene, cloud_seed=11)
    no_azimuth = cloudy.background_temperature_k(el, None)
    plain = bridge(scene).background_temperature_k(el, None)
    assert np.array_equal(no_azimuth, plain)


# --- the azimuth helper -----------------------------------------------------------------------


def test_azimuth_from_rays_is_measured_from_forward_about_up() -> None:
    rays = np.array([[[0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]]])
    azimuth = np.degrees(azimuth_from_rays(rays))
    assert np.allclose(azimuth, [0.0, 90.0, 180.0, 270.0], atol=1e-9)


def test_azimuth_ignores_the_component_along_up() -> None:
    """A ray climbing in elevation keeps its bearing, which is what makes the sampling stable."""
    level = np.array([[[1.0, 0.0, -1.0]]]) / math.sqrt(2.0)
    steep = np.array([[[1.0, 8.0, -1.0]]]) / math.sqrt(66.0)
    assert float(azimuth_from_rays(level)[0, 0]) == pytest.approx(
        float(azimuth_from_rays(steep)[0, 0]), abs=1e-9
    )


def test_azimuth_and_elevation_agree_on_their_conventions() -> None:
    """Straight up is elevation 90; the azimuth there is degenerate but must stay finite."""
    up = np.array([[[0.0, 1.0, 0.0]]])
    assert float(np.degrees(elevation_from_rays(up)[0, 0])) == pytest.approx(90.0)
    assert np.all(np.isfinite(azimuth_from_rays(up)))


def test_azimuth_refuses_a_forward_parallel_to_up() -> None:
    with pytest.raises(ValueError, match="forward must not be parallel"):
        azimuth_from_rays(np.zeros((1, 1, 3)), up=(0.0, 1.0, 0.0), forward=(0.0, 1.0, 0.0))


# --- the two bands must agree on where the cloud is -------------------------------------------


def test_the_dome_and_the_background_sample_the_same_cloud(scene: Scene) -> None:
    """The visible dome and the infrared background must put cloud in the same part of the sky.

    This is the whole reason ADR 0073 kept cloud off the dome until ADR 0076 put it in the
    infrared: a pair that disagrees is worse than a pair that is plain, because each half is
    internally consistent and plausible. They share one field and one angle convention, and this
    checks the sharing rather than trusting it — a sign flip in either path would still produce a
    convincing sky, just a different one.
    """
    from irsim.atmosphere.cloud import sky_angles
    from irsim_isaac.visible_sky import latlong_directions

    fraction = float(scene.weather.at(scene.t0_s).cloud_fraction)
    cloud = generate_sky_cloud(scene.environment.clouds.beta, fraction, seed=11)

    # The dome's own texel directions, which is the hardest case: they are laid out in the
    # renderer's lat-long frame, not in elevation/azimuth at all.
    direction = latlong_directions(64)
    above = direction[..., 1] > 0.0
    dome_elevation, dome_azimuth = sky_angles(direction)
    dome_covered = cloud.sample(dome_elevation[above], dome_azimuth[above])

    # The infrared path reaches the same directions as per-pixel rays.
    rays = direction[above].reshape(1, -1, 3)
    ir_elevation = elevation_from_rays(rays)
    ir_azimuth = azimuth_from_rays(rays)
    ir_covered = cloud.sample(ir_elevation, ir_azimuth).ravel()

    assert np.array_equal(dome_covered, ir_covered)
    assert 0.0 < float(dome_covered.mean()) < 1.0, (
        "a mix of covered and clear, or this proves little"
    )


def test_the_dome_paints_cloud_by_depth_and_leaves_clear_sky_alone(scene: Scene) -> None:
    """The three regimes of a soft-edged cloud, asserted separately (ADR 0125).

    Before the edge ramp this test asked only that *covered* texels changed and *uncovered* ones
    did not, which a stencil satisfies. A stencil is what made the rendered sky a field of flat
    grey blobs. The invariant that replaces it has to distinguish three populations, because the
    middle one is the whole point:

    * ``d = 0`` -- outside the cloud entirely. Untouched, bit for bit.
    * ``d = 1`` -- at full depth. The flat base value, which is what the hard mask gave.
    * ``0 < d < 1`` -- the fringe. **Strictly between** the two, monotonically in ``d``.
    """
    from irsim.atmosphere.cloud import sky_angles
    from irsim_isaac.visible_sky import dome_spec_from_scene, environment_map, latlong_directions

    fraction = float(scene.weather.at(scene.t0_s).cloud_fraction)
    cloud = generate_sky_cloud(scene.environment.clouds.beta, fraction, seed=11)
    clear = environment_map(dome_spec_from_scene(scene), 128)
    cloudy = environment_map(dome_spec_from_scene(scene, cloud=cloud), 128)

    direction = latlong_directions(128)
    up = direction[..., 1]
    elevation, azimuth = sky_angles(direction)
    depth = np.zeros(up.shape, dtype=np.float64)
    depth[up > 0.0] = cloud.density(elevation[up > 0.0], azimuth[up > 0.0])

    untouched = depth <= 0.0
    assert np.array_equal(clear[untouched], cloudy[untouched]), "clear sky must be untouched"

    full = depth >= 1.0
    assert full.any(), "no texel reached full depth; the ramp has swallowed the cloud"
    assert not np.allclose(clear[full], cloudy[full]), "full-depth sky must have changed"
    # At full depth the base is one flat value, so every such texel of the cloudy map agrees.
    assert np.allclose(cloudy[full], cloudy[full][0], rtol=1e-6)

    fringe = (depth > 0.0) & (depth < 1.0)
    # Against the full-depth area rather than against the whole map, so the bound does not move
    # with the scene's cloud fraction. Thin cloud is nearly all edge: measured here, fringe is
    # 1.23x the full-depth area at c = 0.05 and 0.47x at c = 0.45, because a high threshold sits
    # far out in the tail where the field spends little area above it.
    assert fringe.mean() > 0.4 * full.mean(), (
        f"fringe is {fringe.mean():.1%} of the map against {full.mean():.1%} at full depth; "
        "the cloud is still nearly a stencil"
    )
    # Brightness in the fringe lies between the clear sky and the base, and follows the depth.
    base = float(cloudy[full].reshape(-1, cloudy.shape[-1])[0, 0])
    sky = clear[fringe][..., 0]
    lit = cloudy[fringe][..., 0]
    between = (lit - sky) / np.where(np.abs(base - sky) > 1e-12, base - sky, 1.0)
    assert np.all(between >= -1e-6) and np.all(between <= 1.0 + 1e-6)
    order = np.argsort(depth[fringe])
    assert np.corrcoef(depth[fringe][order], between[order])[0, 1] > 0.99


def test_the_edge_ramp_leaves_the_coverage_fraction_exactly_where_it_was() -> None:
    """The ramp crosses 0.5 at the threshold, so it cannot have moved the covered fraction.

    This is what lets ADR 0125 be a change to how a cloud *looks* rather than to how much sky it
    covers -- every statistic taken on `sample` before it still means the same thing.
    """
    cloud = generate_sky_cloud(1.8, 0.45, seed=3, n_elevation=270, n_azimuth=1080)
    rng = np.random.default_rng(5)
    el = rng.uniform(0.05, 1.5, 50_000)
    az = rng.uniform(0.0, 2.0 * np.pi, 50_000)
    mask = cloud.sample(el, az)
    depth = cloud.density(el, az)
    assert np.array_equal(mask, depth >= 0.5)


def test_switching_the_ramp_off_restores_the_hard_mask_exactly() -> None:
    """The old behaviour is a value of a parameter, not a deleted branch."""
    import dataclasses

    soft = generate_sky_cloud(1.8, 0.45, seed=3)
    hard = dataclasses.replace(soft, softness=0.0)
    rng = np.random.default_rng(9)
    el = rng.uniform(0.05, 1.5, 20_000)
    az = rng.uniform(0.0, 2.0 * np.pi, 20_000)
    assert np.array_equal(hard.density(el, az), soft.sample(el, az).astype(float))


def test_a_boolean_mask_gives_the_radiance_it_always_did() -> None:
    """`cloud_radiance` generalised without moving any pixel that was already decided."""
    from irsim.atmosphere.cloud import cloud_radiance

    clear = np.linspace(1.0, 2.0, 501)
    mask = np.zeros(501, dtype=bool)
    mask[::3] = True
    for tau in (0.0, 0.25, 0.6):
        expected = np.where(mask, (1.0 - tau) * 5.0 + tau * clear, clear)
        assert np.array_equal(cloud_radiance(clear, 5.0, tau, mask), expected)
    # And a half-depth ray sits halfway between them, which is the new behaviour.
    half = cloud_radiance(clear, 5.0, 0.0, np.full(501, 0.5))
    assert np.allclose(half, 0.5 * 5.0 + 0.5 * clear)


def test_a_dome_without_a_field_is_unchanged(scene: Scene) -> None:
    """Bit-identical to before ADR 0076, so a clear-sky scene renders exactly as it did."""
    from irsim_isaac.visible_sky import dome_spec_from_scene, environment_map

    spec = dome_spec_from_scene(scene)
    assert spec.cloud is None
    assert np.array_equal(environment_map(spec, 64), environment_map(spec, 64))
