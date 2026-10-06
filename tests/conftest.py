"""Shared fixtures.

Synthetic G-buffer fixtures live here so kernels can be tested without launching Isaac Sim
(ir-sim-testing skill). Every fixture emits exactly the key set frozen in
``irsim.config.gbuffer`` -- the contract the Isaac annotator adapter must also honour -- and
``tests/unit/test_gbuffer_schema.py`` checks each one against it. ``material_id`` 0 is the
UNMAPPED sentinel, so fixtures use ids >= 1.
"""

from __future__ import annotations

import numpy as np
import pytest

from irsim.radiometry.encoding import encode_temperature


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register ``--update-golden`` here (the root conftest) so it exists whichever
    directory pytest is pointed at. The helper that consumes it lives in
    tests/golden/conftest.py."""
    parser.addoption(
        "--update-golden",
        action="store_true",
        default=False,
        help="Regenerate golden reference arrays instead of comparing against them "
        "(make golden-update). Never do this in the same commit as a behaviour change "
        "without saying so in the commit body.",
    )
    # ADR 0193: which data files each test module reads, for `make check`'s selection.
    parser.addoption(
        "--track-reads",
        action="store_true",
        default=False,
        help="Record the data files each test module reads into .testreads.json "
        "(tests/read_tracker.py; consumed by scripts/affected_tests.py).",
    )
    parser.addoption(
        "--track-reads-reset",
        action="store_true",
        default=False,
        help="With --track-reads: replace .testreads.json instead of merging (full-suite runs).",
    )


def pytest_configure(config: pytest.Config) -> None:
    if config.getoption("--track-reads"):
        from read_tracker import ReadTracker

        config.pluginmanager.register(ReadTracker(config), "irsim-read-tracker")


@pytest.hookimpl(hookwrapper=True)
def pytest_collection_modifyitems(items: list[pytest.Item]):  # type: ignore[no-untyped-def]
    """Keep tests in collection order whatever a plugin selects (ADR 0193).

    pytest-testmon sorts what it selects by recorded duration. Some modules share a module-level
    RNG or a module-scoped scene that each test steps forward in time, so running a module's
    tests out of file order changes their results (a scene asked for t0 after a sibling moved
    it 14 h on). Selection is kept; order is restored.
    """
    order = {id(item): i for i, item in enumerate(items)}
    yield
    items.sort(key=lambda item: order.get(id(item), len(order)))


@pytest.fixture(scope="session")
def boson_response():  # type: ignore[no-untyped-def]
    """The committed (estimated) Boson VOx response."""
    from irsim.config.loader import DEFAULT_DATA_DIR
    from irsim.radiometry.spectral_response import load_spectral_response

    return load_spectral_response(DEFAULT_DATA_DIR / "spectra" / "responses" / "boson_vox.csv")


@pytest.fixture(scope="session")
def boson_lut(boson_response):  # type: ignore[no-untyped-def]
    """Full 200-1000 K, 0.05 K BandLUT for the Boson response (built once per session, ~1 s)."""
    from irsim.radiometry.lut import BandLUT

    return BandLUT.build(boson_response)


@pytest.fixture(scope="session")
def tophat_lwir_lut(tmp_path_factory: pytest.TempPathFactory):  # type: ignore[no-untyped-def]
    """BandLUT for an exact 7.5-13.5 um top-hat: every entry has a closed-form answer."""
    from irsim.radiometry.lut import BandLUT
    from irsim.radiometry.spectral_response import load_spectral_response

    p = tmp_path_factory.mktemp("lut") / "tophat_7p5_13p5.csv"
    p.write_text("# exact top-hat\n7.5,1.0\n13.5,1.0\n")
    return BandLUT.build(load_spectral_response(p))


@pytest.fixture(scope="session")
def tophat_mwir_lut(tmp_path_factory: pytest.TempPathFactory):  # type: ignore[no-untyped-def]
    """BandLUT for an exact 3-5 um top-hat (photon-detector tests)."""
    from irsim.radiometry.lut import BandLUT
    from irsim.radiometry.spectral_response import load_spectral_response

    p = tmp_path_factory.mktemp("lut") / "tophat_3_5.csv"
    p.write_text("# exact top-hat\n3.0,1.0\n5.0,1.0\n")
    return BandLUT.build(load_spectral_response(p))


@pytest.fixture
def rng() -> np.random.Generator:
    """Seeded generator. Every test that uses randomness must take this."""
    return np.random.default_rng(20260910)


def _planes(
    temperature_k: np.ndarray,
    *,
    normal_dot_view: np.ndarray | float = 1.0,
    distance_m: np.ndarray | float,
    material_id: np.ndarray | int,
    sky_view_factor: np.ndarray | float,
) -> dict[str, np.ndarray]:
    """Assemble a G-buffer dict with the contract dtypes and the derived ``encoded_t`` plane."""
    t = np.asarray(temperature_k, dtype=np.float32)
    shape = t.shape
    return {
        "temperature_k": t,
        "encoded_t": encode_temperature(t),
        "normal_dot_view": np.broadcast_to(np.float32(normal_dot_view), shape).astype(np.float32),
        "distance_m": np.broadcast_to(np.float32(distance_m), shape).astype(np.float32),
        "material_id": np.broadcast_to(np.int32(material_id), shape).astype(np.int32),
        "sky_view_factor": np.broadcast_to(np.float32(sky_view_factor), shape).astype(np.float32),
    }


@pytest.fixture
def gbuffer_ramp() -> dict[str, np.ndarray]:
    """256x256 G-buffer: linear 250-450 K temperature ramp along x, camera-facing normals,
    constant 50 m range, single material. The workhorse fixture for kernel tests."""
    h = w = 256
    t = np.tile(np.linspace(250.0, 450.0, w, dtype=np.float32), (h, 1))
    return _planes(t, distance_m=50.0, material_id=1, sky_view_factor=0.5)


@pytest.fixture
def gbuffer_uniform() -> dict[str, np.ndarray]:
    """64x64 G-buffer at a uniform 300 K. Use for noise-decomposition tests."""
    t = np.full((64, 64), 300.0, dtype=np.float32)
    return _planes(t, distance_m=10.0, material_id=1, sky_view_factor=0.0)


@pytest.fixture
def gbuffer_two_material() -> dict[str, np.ndarray]:
    """64x64 uniform 300 K; left half material 1, right half material 2.

    Any radiance difference between the halves is due to the material table alone."""
    h = w = 64
    t = np.full((h, w), 300.0, dtype=np.float32)
    mat = np.ones((h, w), dtype=np.int32)
    mat[:, w // 2 :] = 2
    return _planes(t, distance_m=20.0, material_id=mat, sky_view_factor=0.5)


SPHERE_RADIUS_PX = 56
SPHERE_T_K = 320.0
SPHERE_BACKGROUND_T_K = 280.0


@pytest.fixture
def sphere_params() -> dict[str, float]:
    """Geometry of ``gbuffer_sphere`` for tests that recompute the analytic normal."""
    return {
        "radius_px": SPHERE_RADIUS_PX,
        "t_k": SPHERE_T_K,
        "background_t_k": SPHERE_BACKGROUND_T_K,
    }


@pytest.fixture
def gbuffer_sphere() -> dict[str, np.ndarray]:
    """128x128: a sphere of radius 56 px (320 K, 20 m, material 1) on a flat 280 K wall
    (100 m, material 2). Inside the disc ``normal_dot_view = sqrt(1 - r^2/R^2)`` exactly, so
    the fixture runs from 1 at the centre to 0 at the four axis rim pixels -- the grazing
    coverage the angular-emissivity tests need (docs/physics-model.md §4.2)."""
    h = w = 128
    cy = cx = 64
    yy, xx = np.mgrid[0:h, 0:w]
    r2 = ((yy - cy) ** 2 + (xx - cx) ** 2).astype(np.float64)
    inside = r2 <= SPHERE_RADIUS_PX**2
    cos_theta = np.ones((h, w), dtype=np.float64)
    cos_theta[inside] = np.sqrt(1.0 - r2[inside] / SPHERE_RADIUS_PX**2)
    t = np.where(inside, SPHERE_T_K, SPHERE_BACKGROUND_T_K).astype(np.float32)
    dist = np.where(inside, 20.0, 100.0).astype(np.float32)
    mat = np.where(inside, 1, 2).astype(np.int32)
    svf = np.where(inside, 0.5, 0.3).astype(np.float32)
    return _planes(
        t,
        normal_dot_view=cos_theta.astype(np.float32),
        distance_m=dist,
        material_id=mat,
        sky_view_factor=svf,
    )


STEP_EDGE_SUPERSAMPLE = 4
STEP_EDGE_ANGLE_DEG = 5.5
MOVING_EDGE_FRAMES = 8
MOVING_EDGE_VELOCITY_PX = 2.0  # px/frame at supersampled resolution, along +x
STEP_EDGE_HOT_K = 373.0  # hotplate behind ...
STEP_EDGE_COLD_K = 293.0  # ... an aluminium sheet (the ISO 12233-style slant-edge target)


def _step_edge_planes(x_edge_at_top: float, size: int) -> dict[str, np.ndarray]:
    yy, xx = np.mgrid[0:size, 0:size]
    edge_x = x_edge_at_top + yy * np.tan(np.radians(STEP_EDGE_ANGLE_DEG))
    hot = (xx + 0.5) > edge_x  # pixel centre right of the edge sees the hotplate
    t = np.where(hot, STEP_EDGE_HOT_K, STEP_EDGE_COLD_K).astype(np.float32)
    mat = np.where(hot, 1, 2).astype(np.int32)
    return _planes(t, distance_m=5.0, material_id=mat, sky_view_factor=0.4)


@pytest.fixture
def step_edge_params() -> dict[str, float]:
    """Geometry of the step-edge fixtures, for tests that fit or shift the edge."""
    return {
        "supersample": STEP_EDGE_SUPERSAMPLE,
        "angle_deg": STEP_EDGE_ANGLE_DEG,
        "size_px": 256 * STEP_EDGE_SUPERSAMPLE,
        "x_edge_at_top": 512.0,
        "hot_k": STEP_EDGE_HOT_K,
        "cold_k": STEP_EDGE_COLD_K,
        "velocity_px": MOVING_EDGE_VELOCITY_PX,
        "n_frames": MOVING_EDGE_FRAMES,
    }


@pytest.fixture
def gbuffer_step_edge(step_edge_params: dict[str, float]) -> dict[str, np.ndarray]:
    """1024x1024 (a 256x256 frame at 4x supersample): 373 K hotplate on the right of an edge
    tilted 5.5 deg from vertical, 293 K sheet on the left. Ideal -- exactly two temperatures and
    one transition per row -- so any edge blur measured downstream is attributable to the
    optics/box-filter stage under test, not to the fixture (§8, slant-edge MTF)."""
    return _step_edge_planes(step_edge_params["x_edge_at_top"], int(step_edge_params["size_px"]))


@pytest.fixture
def gbuffer_moving_edge(step_edge_params: dict[str, float]) -> list[dict[str, np.ndarray]]:
    """Sequence of 8 step-edge frames with the edge translating +2 px/frame, each carrying a
    ``motion_px`` plane (H, W, 2) = (vx, vy) in px/frame. For motion-MTF and bolometer-smear
    tests (§8.3, §9.4)."""
    size = int(step_edge_params["size_px"])
    frames = []
    for k in range(MOVING_EDGE_FRAMES):
        planes = _step_edge_planes(
            step_edge_params["x_edge_at_top"] + k * MOVING_EDGE_VELOCITY_PX, size
        )
        motion = np.zeros((size, size, 2), dtype=np.float32)
        motion[..., 0] = MOVING_EDGE_VELOCITY_PX
        planes["motion_px"] = motion
        frames.append(planes)
    return frames


# --- aerial scene (MS.8) ---------------------------------------------------------------
# A sky-background scene needs a sky model, which needs an atmosphere, a weather series and a
# band LUT, so the expensive pieces are session-scoped: MS.2's elevation LUT costs ~0.4 s to
# build and is cached per weather time inside the SkyModel.

AERIAL_T_AIR_K = 288.15
AERIAL_RH = 0.4
AERIAL_BORESIGHT_DEG = 20.0


@pytest.fixture(scope="session")
def aerial_sensor():  # type: ignore[no-untyped-def]
    """A wide-field 64x48 LWIR camera: the Boson's pitch and f/1 optics on a 1 mm lens, so a
    small frame still spans 42 deg horizontally and the sky gradient is visible across it."""
    import copy
    import pathlib as _pathlib

    import yaml

    from irsim.config.sensor import SensorConfig

    repo = _pathlib.Path(__file__).resolve().parent.parent
    raw = yaml.safe_load((repo / "configs" / "sensors" / "flir_boson_640_lwir.yaml").read_text())
    d = copy.deepcopy(raw)
    d["sensor"]["fpa"].update(width=64, height=48)
    d["sensor"]["optics"]["focal_length_mm"] = 1.0
    d["sensor"]["optics"]["supersample_factor"] = 1
    return SensorConfig.model_validate(d)


def _aerial_sky(lut, cloud_fraction: float, rh: float = AERIAL_RH):  # type: ignore[no-untyped-def]
    from irsim.atmosphere.layered import LayeredAtmosphere
    from irsim.atmosphere.library import load_atmosphere_preset
    from irsim.atmosphere.sky import SkyModel
    from irsim.config.environment import load_environment_preset
    from irsim.thermal.weather import WeatherSample, WeatherSeries

    weather = WeatherSeries.constant(
        WeatherSample(AERIAL_T_AIR_K, rh, 1.0, cloud_fraction, 0.0, 0.0, 23000.0, 0.0), 3600.0
    )
    atmosphere = LayeredAtmosphere(
        load_atmosphere_preset("us_standard_clear"), weather, {"lwir": lut}
    )
    return SkyModel(atmosphere, load_environment_preset("clear_dry"), "lwir", lut)


@pytest.fixture(scope="session")
def aerial_sky(tophat_lwir_lut):  # type: ignore[no-untyped-def]
    """Clear-sky LWIR SkyModel over the US Standard atmosphere at 288.15 K, RH 0.4."""
    return _aerial_sky(tophat_lwir_lut, cloud_fraction=0.0)


@pytest.fixture(scope="session")
def aerial_cloudy_sky(tophat_lwir_lut):  # type: ignore[no-untyped-def]
    """The same sky at cloud fraction 0.4 and RH 0.6 (a cloud base around 800 m)."""
    return _aerial_sky(tophat_lwir_lut, cloud_fraction=0.4, rh=0.6)


@pytest.fixture(scope="session")
def aerial_materials():  # type: ignore[no-untyped-def]
    """The committed material library packed for the LWIR band (ids 1..N, 0 = UNMAPPED)."""
    from irsim.materials.library import MaterialLibrary
    from irsim.materials.table import MaterialTable

    return MaterialTable.from_library(MaterialLibrary.load(), "lwir")


@pytest.fixture
def aerial_scene(aerial_sensor, aerial_sky, aerial_materials):  # type: ignore[no-untyped-def]
    """Clear sky from ~4 deg to ~36 deg elevation with one resolved 2 px target.

    Returns the :class:`irsim.validation.aerial_scene.AerialScene`, which carries the per-pixel
    elevation grid and the target bookkeeping a phenomenology test needs to state the right
    answer; ``gbuffer_aerial`` is just its plane dict.
    """
    from irsim.validation.aerial_scene import SceneTarget, build_aerial_gbuffer

    sensor = aerial_sensor.sensor
    range_m = 700.0
    size_m = 2.0 * range_m * sensor.fpa.pitch_um * 1e-6 / (sensor.optics.focal_length_mm * 1e-3)
    return build_aerial_gbuffer(
        sensor,
        aerial_sky,
        aerial_materials,
        boresight_elevation_deg=AERIAL_BORESIGHT_DEG,
        targets=[SceneTarget("painted_composite", 310.0, size_m, range_m, (20.5, 20.5))],
    )


@pytest.fixture
def gbuffer_aerial(aerial_scene) -> dict[str, np.ndarray]:  # type: ignore[no-untyped-def]
    """64x48 aerial G-buffer: sky with an elevation gradient, a 2 px target, ``sky_mask`` set.

    Unlike the ground fixtures above this one *does* carry ``material_id`` 0 -- under the sky
    mask, where it is the agreed contract rather than an unmapped asset (ADR 0050) -- so it is
    deliberately not in ``test_gbuffer_schema.py``'s sentinel-free list; ``test_tier3_sky.py``
    checks the sky-aware form of that rule instead.
    """
    return dict(aerial_scene.planes)
