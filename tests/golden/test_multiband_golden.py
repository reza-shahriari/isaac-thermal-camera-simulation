"""Golden frames beyond LWIR (GT.2): one MWIR, one SWIR and one NIR.

Every golden array in this store came from one Boson LWIR config. That left the two subsystems
where drift is hardest to see by eye -- the reflective-band chain and the sky/sea background --
with no reference array at all, and `SC.1` is the proof it matters: a 1.52x sigma and a zero dark
current sat in the MWIR configuration until someone read the numbers, and a single MWIR golden
would have caught both.

Each frame runs the **real** shipped config, its committed band LUT and the material library's own
table for that band, on a synthetic G-buffer whose scene is chosen to span the ADC without
clipping -- a saturated golden hides exactly the changes it exists to catch. The reflective bands
carry an `l_sun` plane scaled from `SolarIllumination`'s own band irradiance, because at 300 K
their self-emission is ~1e-9 of LWIR's and an emission-only frame would be a picture of the noise.

docs/physics-model.md §5.2, §9, §10, §11; ADR 0004 (the golden policy), ADR 0084 (illumination).
"""

from __future__ import annotations

import hashlib
import pathlib

import numpy as np
import pytest
from golden_store import GoldenStore

from irsim.config.loader import config_hash, load_sensor_config
from irsim.config.sensor import SensorConfig
from irsim.materials import MaterialTable
from irsim.materials.library import MaterialLibrary
from irsim.pipeline import PipelineConfig, PipelineState, run_frame
from irsim.pipeline.solar import SolarIllumination
from irsim.radiometry.encoding import encode_temperature
from irsim.radiometry.lut_files import load_band_lut_for_config

REPO = pathlib.Path(__file__).resolve().parents[2]
DATA = REPO / "data"
SIZE = 128

#: Sensor per band, and the fraction of that band's own solar irradiance the scene is lit by.
#: `None` leaves the band on self-emission alone. The fractions are **chosen so the frame spans
#: the converter without clipping** -- SWIR reaches 15 % saturation by 2e-3 and NIR needs 0.6 of
#: full sun to reach two thirds of a 12-bit scale, which is itself the phenomenology: the two
#: bands are three orders of magnitude apart in how much sun it takes to make a picture.
SCENES: dict[str, tuple[str, float | None]] = {
    "mwir": ("example_mwir_insb_640", None),
    "swir": ("example_swir_ingaas_640", 1.0e-3),
    "nir": ("example_nir_si_1280", 0.6),
}


@pytest.fixture(scope="module")
def library() -> MaterialLibrary:
    return MaterialLibrary.load(REPO / "configs" / "materials")


def _planes(table: MaterialTable, sun_band: float | None) -> dict[str, np.ndarray]:
    """A ramp of illumination across a scene with two temperature steps in it.

    Two steps and one ramp, so the frame exercises the emissive term, the reflective term and
    their sum in one array rather than three.
    """
    t = np.full((SIZE, SIZE), 285.0, np.float32)
    t[40:80, 20:60] = 310.0
    t[10:25, 90:120] = 330.0
    planes: dict[str, np.ndarray] = {
        "temperature_k": t,
        "encoded_t": encode_temperature(t),
        "normal_dot_view": np.ones((SIZE, SIZE), np.float32),
        "distance_m": np.full((SIZE, SIZE), 50.0, np.float32),
        "material_id": np.full((SIZE, SIZE), table.id_for("concrete"), np.int32),
        "sky_view_factor": np.full((SIZE, SIZE), 0.5, np.float32),
    }
    if sun_band is not None:
        ramp = np.linspace(0.05 * sun_band, sun_band, SIZE, dtype=np.float32)
        planes["l_sun"] = np.ascontiguousarray(np.broadcast_to(ramp, (SIZE, SIZE)), np.float32)
    return planes


@pytest.mark.parametrize("band", sorted(SCENES))
def test_golden_frame_for_every_band_beyond_lwir(
    golden: GoldenStore, library: MaterialLibrary, band: str
) -> None:
    name, sun_fraction = SCENES[band]
    cfg = load_sensor_config(REPO / "configs" / "sensors" / f"{name}.yaml", DATA)
    lut = load_band_lut_for_config(cfg, DATA / "lut", DATA)
    table = MaterialTable.from_library(library, band)

    dumped = cfg.model_dump(mode="json")
    dumped["sensor"]["fpa"].update(width=SIZE, height=SIZE)
    dumped["sensor"]["optics"]["supersample_factor"] = 1
    small = SensorConfig.model_validate(dumped)
    pipeline = PipelineConfig.from_sensor(small, table, lut=lut, sensor_seed=77)

    sun_band = None
    if sun_fraction is not None:
        e_band = float(SolarIllumination.for_sensor(cfg, pipeline.quantity, DATA).e_band)
        sun_band = sun_fraction * e_band
    planes = _planes(table, sun_band)
    out = run_frame(planes, pipeline, PipelineState(housing_temp_k=pipeline.t_housing_cal_k))

    key = hashlib.sha256(
        (config_hash(cfg) + f"|{SIZE}x{SIZE}|sun={sun_fraction}|numpy=" + np.__version__).encode()
    ).hexdigest()
    assert out.radiance is not None and out.dn16 is not None and out.display8 is not None
    golden.check(f"{band}_frame_radiance", out.radiance, config_hash=key, rtol=1e-5, units="band")
    golden.check(f"{band}_frame_dn16", out.dn16, config_hash=key, atol=1.0, units="DN16")
    golden.check(
        f"{band}_frame_display8",
        out.display8[..., 0],
        config_hash=key,
        atol=1.0,
        units="DN8 (R channel)",
    )
    # The frame has to be worth comparing: a golden that sits at one end of the converter would
    # pass through a change that moved everything below it.
    assert float((out.dn16 >= 65535).mean()) == 0.0
    assert int(np.ptp(out.dn16)) > 100


def test_the_reflective_bands_really_are_running_on_reflected_sunlight(
    library: MaterialLibrary,
) -> None:
    """Why the `l_sun` plane is in the scene at all, asserted rather than assumed.

    Remove it and a SWIR frame is a picture of the dark current, because a 310 K concrete slab
    emits ~1e-9 of what it does in LWIR. A golden of that would pin the noise model and nothing
    else, and would pass unchanged through any error in the reflective chain.
    """
    cfg = load_sensor_config(REPO / "configs" / "sensors" / "example_swir_ingaas_640.yaml", DATA)
    lut = load_band_lut_for_config(cfg, DATA / "lut", DATA)
    table = MaterialTable.from_library(library, "swir")
    dumped = cfg.model_dump(mode="json")
    dumped["sensor"]["fpa"].update(width=32, height=32)
    dumped["sensor"]["optics"]["supersample_factor"] = 1
    pipeline = PipelineConfig.from_sensor(
        SensorConfig.model_validate(dumped), table, lut=lut, sensor_seed=77, noise_enabled=False
    )
    e_band = float(SolarIllumination.for_sensor(cfg, pipeline.quantity, DATA).e_band)
    state = PipelineState(housing_temp_k=pipeline.t_housing_cal_k)

    lit = _planes(table, 1.0e-3 * e_band)
    dark = _planes(table, None)
    lit_out = run_frame({k: v[:32, :32] for k, v in lit.items()}, pipeline, state)
    dark_out = run_frame({k: v[:32, :32] for k, v in dark.items()}, pipeline, state)
    assert lit_out.dn16 is not None and dark_out.dn16 is not None
    assert int(dark_out.dn16.max()) < 32
    assert int(lit_out.dn16.max()) > 1000
