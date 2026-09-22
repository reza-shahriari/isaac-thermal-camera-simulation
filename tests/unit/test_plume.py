"""Stage 2d: an exhaust plume composited per pixel (roadmap PH.6).

Three things have to be true and are each measured here rather than argued.

**The geometry is exact.** A chord through a cone has a closed form at the stations where one
exists -- broadside through a cylinder is `2r`, broadside through a cone at station `s` is
`2r(s)`, straight down the axis is the cone's own length -- and the intersection is checked
against all three. A ray-marched or sampled version would pass a loose tolerance and hide a
half-pixel bias; these are exact to floating point.

**The two bands disagree, and a grey slab cannot make them.** This is the feature `PH.4` exists
for and the one a scene author would never notice was missing: a CO2/H2O plume absorbs a fifth of
the MWIR band and almost none of the LWIR one. The negative control builds a *grey* table from
the same machinery and shows it gives the two bands the same transmittance, so the test that
passes is measuring the tables and not the plumbing.

**A frame with no plume is untouched.** Bit for bit, the same object, so every golden written
before this step still describes what the pipeline does.

docs/physics-model.md §6.6, §8.1, §8.3, §13.4 stage 2; ADR 0098.
"""

from __future__ import annotations

import copy
import pathlib
from dataclasses import replace
from typing import Any

import numpy as np
import pytest
import yaml
from numpy.typing import NDArray

from irsim.config.sensor import SensorConfig
from irsim.materials import MaterialTable
from irsim.optics.projection import Intrinsics
from irsim.pipeline import PipelineConfig, PipelineState, run_frame
from irsim.pipeline.gas_slab import GasBandTables, SpeciesAbsorption, gas_band_radiance
from irsim.pipeline.plume import ExhaustPlume, PlumeCone, chord_through_cone, inject_plumes
from irsim.radiometry.lut import BandLUT
from irsim.radiometry.spectral_response import load_spectral_response

REPO = pathlib.Path(__file__).resolve().parents[2]
CONFIGS = REPO / "configs" / "sensors"
RESPONSES = REPO / "data" / "spectra" / "responses"

#: A tailpipe plume: 6 cm mouth widening to 24 cm over 1.2 m, 700 K at the tip into 290 K air,
#: mixing out over 0.6 m. Mole fractions are a warm petrol exhaust's.
TIP_K, AIR_K = 700.0, 290.0


def _plume(apex=(0.0, 0.0, 4.0), axis=(1.0, 0.0, 0.0), **kw: Any) -> ExhaustPlume:
    defaults = dict(
        t_tip_k=TIP_K,
        t_air_k=AIR_K,
        mixing_length_m=0.6,
        p_co2_tip_atm=0.12,
        p_h2o_tip_atm=0.12,
    )
    defaults.update(kw)
    cone = PlumeCone(apex_m=apex, axis=axis, length_m=1.2, radius_tip_m=0.03, radius_end_m=0.12)
    return ExhaustPlume(cone=cone, **defaults)  # type: ignore[arg-type]


# --- the geometry ---------------------------------------------------------------------------------


def test_a_broadside_chord_through_a_cylinder_is_its_diameter() -> None:
    cone = PlumeCone((-1.0, 0.0, 10.0), (1.0, 0.0, 0.0), 2.0, 0.2, 0.2)
    length, axial, range_m = chord_through_cone(cone, np.array([[0.0, 0.0, 1.0]]))
    assert length[0] == pytest.approx(0.4, abs=1e-9)
    assert axial[0] == pytest.approx(1.0, abs=1e-9)
    assert range_m[0] == pytest.approx(10.0, abs=1e-9)


def test_a_broadside_chord_through_a_cone_follows_its_radius() -> None:
    """The radius grows linearly along the axis, so the chord has to as well."""
    for station in (0.25, 1.5, 2.75):
        cone = PlumeCone((-station, 0.0, 10.0), (1.0, 0.0, 0.0), 3.0, 0.1, 0.4)
        length, _, _ = chord_through_cone(cone, np.array([[0.0, 0.0, 1.0]]))
        expected = 2.0 * (0.1 + (0.4 - 0.1) * station / 3.0)
        assert length[0] == pytest.approx(expected, abs=1e-9), station


def test_a_ray_down_the_axis_traverses_the_whole_plume() -> None:
    """The case that breaks a naive quadratic: within the cone's half-angle of its own axis the
    inequality holds *outside* the roots, not between them, and a solver that misses it renders a
    plume seen end-on -- the commonest view of a tailpipe -- as empty."""
    cone = PlumeCone((0.0, 0.0, 5.0), (0.0, 0.0, 1.0), 3.0, 0.1, 0.4)
    length, axial, range_m = chord_through_cone(cone, np.array([[0.0, 0.0, 1.0]]))
    assert length[0] == pytest.approx(3.0, abs=1e-9)
    assert axial[0] == pytest.approx(1.5, abs=1e-9)
    assert range_m[0] == pytest.approx(6.5, abs=1e-9)


def test_a_ray_that_misses_gets_nothing() -> None:
    cone = PlumeCone((-1.0, 0.0, 10.0), (1.0, 0.0, 0.0), 2.0, 0.2, 0.2)
    for direction in ([0.0, 1.0, 0.0], [0.0, 0.0, -1.0], [0.0, 0.5, 0.866]):
        length, _, _ = chord_through_cone(cone, np.array([direction]))
        assert length[0] == 0.0, direction


def test_the_depth_plane_cuts_the_chord_where_the_geometry_is() -> None:
    """Occlusion without a mask: the G-buffer already knows what stands in front."""
    cone = PlumeCone((-1.0, 0.0, 10.0), (1.0, 0.0, 0.0), 2.0, 0.2, 0.2)
    ray = np.array([[0.0, 0.0, 1.0]])
    assert chord_through_cone(cone, ray, np.array([9.9]))[0][0] == pytest.approx(0.1, abs=1e-9)
    assert chord_through_cone(cone, ray, np.array([9.5]))[0][0] == 0.0  # wholly behind a surface
    assert chord_through_cone(cone, ray, np.array([50.0]))[0][0] == pytest.approx(0.4, abs=1e-9)


def test_entrainment_dilutes_temperature_and_species_by_the_same_factor() -> None:
    """One authored length, one conserved scalar: they cannot be set inconsistently."""
    plume = _plume(mixing_length_m=0.6)
    phi = plume.dilution(np.array([0.0, 0.6, 1.2]))
    assert phi[0] == pytest.approx(1.0)
    assert phi[1] == pytest.approx(np.exp(-1.0))
    t_gas = plume.t_air_k + phi * (plume.t_tip_k - plume.t_air_k)
    assert (t_gas[1] - AIR_K) / (TIP_K - AIR_K) == pytest.approx(np.exp(-1.0))


@pytest.mark.parametrize(
    "kw, match",
    [
        ({"t_tip_k": 250.0}, "outside"),
        ({"t_air_k": 800.0}, "colder end"),
        ({"t_air_k": -1.0}, "positive temperature"),
        ({"mixing_length_m": 0.0}, "must be positive"),
        ({"p_co2_tip_atm": 0.0, "p_h2o_tip_atm": 0.0}, "warm air, not a plume"),
    ],
)
def test_a_plume_that_is_not_one_is_refused(kw: dict[str, Any], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        _plume(**kw)


# --- the radiometry -------------------------------------------------------------------------------

#: A tailpipe plume at cruise: 4.5 cm mouth widening to 22 cm over 1.5 m, 480 K at the exit
#: (`TC.7`'s tip node, MVFRI's post-stop series) into 290 K air, mixing out over 0.7 m, six metres
#: from the camera. Mole fractions are a warm petrol exhaust's.
BENCH_DISTANCE_M = 6.0


def _bench_plume() -> ExhaustPlume:
    cone = PlumeCone((0.0, 0.0, BENCH_DISTANCE_M), (1.0, 0.0, 0.0), 1.5, 0.045, 0.22)
    return ExhaustPlume(
        cone=cone,
        t_tip_k=480.0,
        t_air_k=290.0,
        mixing_length_m=0.7,
        p_co2_tip_atm=0.11,
        p_h2o_tip_atm=0.12,
    )


CAMERAS = (("mwir", "example_mwir_insb_640.yaml"), ("lwir", "flir_boson_640_lwir.yaml"))


def _config(name: str, **kw: Any) -> PipelineConfig:
    """A small crop of a shipped camera, at a field of view a plume bench would actually use."""
    spec = copy.deepcopy(yaml.safe_load((CONFIGS / name).read_text()))
    spec["sensor"]["fpa"].update(width=48, height=32)
    spec["sensor"]["optics"]["supersample_factor"] = 1
    spec["sensor"]["optics"]["focal_length_mm"] = 4.0
    sensor = SensorConfig.model_validate(spec)
    stem = pathlib.Path(sensor.sensor.band.spectral_response).name
    return PipelineConfig.from_sensor(
        sensor,
        MaterialTable.constant(0.95),
        lut=BandLUT.build(load_spectral_response(RESPONSES / stem)),
        noise_enabled=False,
        psf_enabled=False,
        **kw,
    )


def _planes(config: PipelineConfig, t_background_k: float, distance_m: float) -> dict[str, Any]:
    h, w = config.sensor.sensor.fpa_shape
    return {
        "temperature_k": np.full((h, w), t_background_k, dtype=np.float32),
        "material_id": np.ones((h, w), dtype=np.uint16),
        "distance_m": np.full((h, w), distance_m, dtype=np.float32),
    }


def _transmittance(config: PipelineConfig, plume: ExhaustPlume) -> NDArray[np.float64]:
    """The per-pixel band transmittance the injector applied, recovered from two planes.

    Rather than reaching into the injector: the composite is affine in the plane, so running it on
    a plane of zeros and on a plane of ``V`` and dividing the difference by ``V`` returns τ
    exactly. ``V`` is the gas's own band radiance and not 1.0 -- a photon FPA's radiances are
    around 1e18, and a unit plane would be lost under the gas term before the subtraction.
    """
    sensor = config.sensor.sensor
    h, w = sensor.fpa_shape
    scale = gas_band_radiance(
        config.response, plume.t_tip_k, "lb" if config.quantity == "lb" else "lb_q"
    )
    args = (
        [plume],
        Intrinsics.from_sensor(sensor, 1),
        sensor.optics.distortion,
        config.response,
        config.gas_tables,
        np.full((h, w), 1000.0),
        None,
        sensor.band.band_id,
        0.0,
        config.lut,
        config.quantity,
    )
    lit = inject_plumes(np.full((h, w), scale), *args)
    dark = inject_plumes(np.zeros((h, w)), *args)
    return np.asarray((lit - dark) / scale)


def test_the_plume_absorbs_far_more_of_mwir_than_of_lwir() -> None:
    """`PH.6`'s acceptance bands, from NIRATAM's plume fits.

    Measured on the committed `PH.5` tables for the bench plume above: **τ_MWIR = 0.866** and
    **τ_LWIR = 0.979**. A grey emissivity cannot produce a difference like that, and the negative
    control below shows this test can tell.
    """
    plume = _bench_plume()
    taus = {}
    for band, name in CAMERAS:
        tau = _transmittance(_config(name), plume)
        covered = tau < 1.0 - 1e-9
        assert covered.any(), f"{band}: the plume covered no pixels"
        taus[band] = float(tau[covered].min())
    assert 0.65 <= taus["mwir"] <= 1.0, taus
    assert taus["lwir"] >= 0.90, taus
    assert taus["lwir"] - taus["mwir"] > 0.05, taus


def test_a_grey_table_gives_the_two_bands_the_same_answer() -> None:
    """The negative control, and the measurement it controls, in one test.

    Swap the `PH.5` tables for a single coefficient used in both bands and the two cameras agree
    to **0.02 %** -- the residue is that their pixel grids sample slightly different chords, not
    any band physics. With the committed tables the same two cameras differ by **11 %**. A grey
    emissivity knob is the first thing a scene author reaches for, and this is the number that
    says it cannot do the job.
    """
    grey = {
        name: SpeciesAbsorption(np.array([300.0, 2500.0]), np.array([3.0, 3.0]))
        for name in ("co2", "h2o")
    }
    plume = _bench_plume()
    flat, real = [], []
    for band, name in CAMERAS:
        config = _config(name)
        for tables, into in (
            (GasBandTables(band=band, species=grey), flat),
            (config.gas_tables, real),
        ):
            tau = _transmittance(replace(config, gas_tables=tables), plume)
            into.append(float(tau[tau < 1.0 - 1e-9].min()))
    assert abs(flat[0] - flat[1]) < 1e-3, flat
    assert abs(real[0] - real[1]) > 0.05, real


def test_transmittance_falls_towards_the_exit() -> None:
    """Hottest and densest at the pipe; the plume clears as it mixes out.

    `PH.6`'s row asks for τ "falling toward the exit", and it is the *shape* a plume model has to
    get right -- a constant-property cylinder would be flat whatever its coefficient.
    """
    tau = _transmittance(_config("example_mwir_insb_640.yaml"), _bench_plume())
    covered = tau < 1.0 - 1e-9
    columns = np.where(covered.any(axis=0))[0]
    assert columns.size >= 4
    profile = [float(tau[covered[:, c], c].min()) for c in columns]
    # The apex sits at +x, which projects to the low-column end: that is the exit.
    assert all(a < b for a, b in zip(profile, profile[1:], strict=False)), profile
    # Only 1.5 % of transmittance over the visible span, and that is the physics rather than a
    # weak effect: downstream the gas is cooler and thinner but the cone is wider, so a longer
    # chord through weaker gas nearly cancels a short chord through strong gas. Monotone is the
    # claim that has teeth here; a constant-property cylinder would be flat.
    assert profile[-1] - profile[0] > 0.01, profile


def test_a_frame_with_no_plume_is_the_same_object() -> None:
    plane = np.linspace(1.0, 2.0, 48 * 32).reshape(32, 48)
    config = _config("example_mwir_insb_640.yaml")
    sensor = config.sensor.sensor
    out = inject_plumes(
        plane,
        (),
        Intrinsics.from_sensor(sensor, 1),
        sensor.optics.distortion,
        config.response,
        config.gas_tables,
        None,
        None,
        "mwir",
        0.0,
        config.lut,
        config.quantity,
    )
    assert out is plane


def test_the_plume_dominates_the_mwir_frame_and_barely_marks_the_lwir_one() -> None:
    """Through the whole chain, in apparent temperature -- what the two cameras actually show.

    Measured: **+72.7 K** peak on the InSb camera against **+7.1 K** on the Boson, for the same
    gas at the same moment. That ten-to-one split is `PH.6`'s claim, and it comes entirely from
    the tables: the plume's geometry, temperature and species are identical in both frames.
    """
    plume = _bench_plume()
    contrast = {}
    for band, name in CAMERAS:
        config = _config(name)
        planes = _planes(config, 300.0, 1000.0)
        clean = run_frame(planes, config, PipelineState())
        hot = run_frame(planes, config, PipelineState(), plumes=[plume])
        assert clean.apparent_t is not None and hot.apparent_t is not None
        contrast[band] = float(np.max(hot.apparent_t - clean.apparent_t))
    assert contrast["mwir"] > 50.0, contrast
    assert contrast["lwir"] < 10.0, contrast
    assert contrast["mwir"] > 8.0 * contrast["lwir"], contrast


def test_an_occluded_plume_leaves_the_frame_alone() -> None:
    """A wall between the camera and the plume: the depth plane alone has to be enough."""
    config = _config("example_mwir_insb_640.yaml")
    plume = _bench_plume()
    sensor = config.sensor.sensor
    h, w = sensor.fpa_shape
    plane = np.full((h, w), 1.0e17)
    args: tuple[Any, ...] = (
        [plume],
        Intrinsics.from_sensor(sensor, 1),
        sensor.optics.distortion,
        config.response,
        config.gas_tables,
        np.full((h, w), BENCH_DISTANCE_M - 1.0),
        None,
        "mwir",
        0.0,
        config.lut,
        config.quantity,
    )
    assert np.array_equal(inject_plumes(plane, *args), plane)


def test_a_band_with_no_table_refuses_a_gas_plume() -> None:
    """SWIR has no CO2 table (`PH.5`), so a plume there is an error, not a clear plume."""
    config = _config("example_swir_ingaas_640.yaml")
    assert config.gas_tables is None
    sensor = config.sensor.sensor
    h, w = sensor.fpa_shape
    with pytest.raises(ValueError, match="no table"):
        inject_plumes(
            np.ones((h, w)),
            [_bench_plume()],
            Intrinsics.from_sensor(sensor, 1),
            sensor.optics.distortion,
            config.response,
            GasBandTables(band="swir", species={}),
            None,
            None,
            "swir",
            0.0,
            config.lut,
            config.quantity,
        )


# --- one scene file, two bands -----------------------------------------------------------------

SCENE = REPO / "configs" / "scenes" / "car_exhaust_plume.yaml"

#: Broadside, eight metres north of the plume, looking north. Rows of the world-to-camera
#: rotation are the camera's own axes in world coordinates: right = east, down = −up, forward =
#: north.
BROADSIDE = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
EYE_M = np.array([-2.85, -8.0, 0.30])


def _scene_plume(minutes: float = 20.0) -> Any:
    from irsim.scene import Scene

    scene = Scene.from_file(SCENE)
    t = 0.0
    for _ in range(int(minutes)):
        scene.advance_targets(t, 60.0)
        t += 60.0
    return scene, scene.plumes_at(t)


def test_the_scene_takes_the_plume_temperature_from_the_solved_gas_not_the_skin() -> None:
    """`PH.6`'s "T and species from the exhaust node", and the trap inside it.

    An exhaust target reports a *skin* -- that is what a camera under the car sees. A plume is
    made of what is inside the pipe, and at load the two differ by about 50 K. Authoring the
    plume from the number the target reports would be the easy mistake and would cool every
    plume in the repository by that much.
    """
    scene, plumes = _scene_plume()
    assert len(plumes) == 1
    skin_k = scene.targets["tailpipe"].temperature()
    assert plumes[0].t_tip_k > skin_k + 25.0, (plumes[0].t_tip_k, skin_k)
    assert plumes[0].t_air_k == pytest.approx(scene.weather_at(1200.0).t_air_k)


def test_one_scene_file_gives_the_two_bands_completely_different_pictures() -> None:
    """The step's own acceptance: one authored plume, two cameras, nothing else changed.

    Neither band appears anywhere in `car_exhaust_plume.yaml`. The split comes out of
    `data/gas/*.npy` and the camera's own R(λ), which is what "bands are data, not code" buys.
    """
    _, plumes = _scene_plume()
    plume = plumes[0].in_camera(BROADSIDE, EYE_M)
    contrast = {}
    for band, name in CAMERAS:
        config = _config(name)
        planes = _planes(config, 300.0, 1000.0)
        clean = run_frame(planes, config, PipelineState())
        hot = run_frame(planes, config, PipelineState(), plumes=[plume])
        assert clean.apparent_t is not None and hot.apparent_t is not None
        contrast[band] = float(np.max(hot.apparent_t - clean.apparent_t))
    assert contrast["mwir"] > 60.0, contrast
    assert contrast["mwir"] > 3.0 * contrast["lwir"], contrast


def test_a_plume_authored_on_anything_but_an_exhaust_is_refused() -> None:
    """The temperature has to come from a solved pipe; there is no way to author one."""
    import yaml as _yaml

    from irsim.config.scene import SceneSpec

    doc = _yaml.safe_load(SCENE.read_text())["scene"]
    target = doc["targets"][1]
    target["solver"] = "newton"
    target.update(t0_k=400.0, tau_s=600.0)
    for gone in ("load_s", "load", "section"):
        target.pop(gone, None)
    with pytest.raises(ValueError, match="only an exhaust target takes a `plume`"):
        SceneSpec.model_validate(doc)
