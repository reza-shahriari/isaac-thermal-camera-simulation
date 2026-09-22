"""Golden arrays for a surface temperature *field* and the frame made from one (GT.2).

`PT.17`'s point-wise path is the project's headline capability and the only thing guarding it was
a set of property tests -- each cell converges to its own root, an unshaded cell reproduces the
per-prim value. Those catch a field that is wrong in a *stated* way. They do not catch a field that
drifts: a solver whose tick changes shape, a forcing term that moves by a percent, an interpolation
that starts leaning. Two arrays here: the cells themselves, and the frame that sampling them makes.

The field is built from a literal forcing rather than from a scene config on purpose. A scene takes
twenty seconds to load and pulls in the weather file, the material library, the spin-up cache and
the occluders, so a change in any of them would read as a change in the field. What is being pinned
is the solver.

docs/physics-model.md §6.1, §6.4, §12.3; ADR 0004, ADR 0087, ADR 0093, ADR 0102.
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
from irsim.pipeline import PipelineConfig, PipelineState, run_frame
from irsim.radiometry.encoding import encode_temperature
from irsim.radiometry.lut import BandLUT
from irsim.thermal.conduction import lateral_operator
from irsim.thermal.facets import FacetForcing, FacetProperties
from irsim.thermal.surface_field import PlanarPatch, PlanarThermalField

REPO = pathlib.Path(__file__).resolve().parents[2]
BOSON_YAML = REPO / "configs" / "sensors" / "flir_boson_640_lwir.yaml"

N_U, N_V = 32, 32
TICK_S = 60.0
STEPS = 180  # three hours of scene time: long enough for the two halves to separate
T0_K = 288.0
#: Concrete, from `configs/materials/concrete.yaml`, stated here so the golden's inputs are all
#: in this file: a change in the library is a change to the *scene*, not to the solver.
CAPACITY_J_M2_K = 1.4e5
EMISSIVITY = 0.92
ABSORPTIVITY = 0.6
CONDUCTIVITY_W_MK = 1.4
THICKNESS_M = 0.10


def _patch() -> PlanarPatch:
    return PlanarPatch(
        origin_m=np.array([0.0, 0.0, 0.0]),
        u_axis=np.array([1.0, 0.0, 0.0]),
        v_axis=np.array([0.0, 1.0, 0.0]),
        n_u=N_U,
        n_v=N_V,
        du_m=0.25,
        dv_m=0.25,
    )


def _forcing(n_cells: int):  # type: ignore[no-untyped-def]
    """Half the patch in sun, the other half in shadow, with the terminator down the middle.

    This is `PT.20`'s wall in miniature and it is the case a field exists for: the two halves
    settle 10 K apart and lateral conduction smears the step between them, so the array carries
    both the per-cell balance and the operator that couples the cells.
    """
    lit = np.zeros(n_cells)
    lit.reshape(N_V, N_U)[:, N_U // 2 :] = 1.0

    def at(t_s: float) -> FacetForcing:
        return FacetForcing(
            t_air_k=290.0,
            h_w_m2_k=10.0,
            q_solar_w_m2=ABSORPTIVITY * 800.0 * lit,
            q_longwave_down_w_m2=EMISSIVITY * 300.0,
        )

    return at


@pytest.fixture(scope="module")
def field() -> PlanarThermalField:
    patch = _patch()
    n = patch.n_cells
    properties = FacetProperties(
        heat_capacity_j_m2_k=np.full(n, CAPACITY_J_M2_K),
        emissivity=np.full(n, EMISSIVITY),
        solar_absorptivity=np.full(n, ABSORPTIVITY),
    )
    return PlanarThermalField(
        patch,
        properties,
        _forcing(n),
        0.0,
        np.full(n, T0_K),
        TICK_S,
        conduction=lateral_operator(patch, CONDUCTIVITY_W_MK, THICKNESS_M),
    )


def _key(*parts: object) -> str:
    return hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()


SOLVER_KEY = _key(
    N_U,
    N_V,
    TICK_S,
    STEPS,
    T0_K,
    CAPACITY_J_M2_K,
    EMISSIVITY,
    ABSORPTIVITY,
    CONDUCTIVITY_W_MK,
    THICKNESS_M,
    np.__version__,
)


@pytest.fixture(scope="module")
def cells(field: PlanarThermalField) -> np.ndarray:
    """The field walked forward once, in a single pass -- `keep_ticks` refuses a rewind."""
    t_end = STEPS * TICK_S
    field.advance_to(t_end)
    return np.asarray(field.temperature_at(t_end), dtype=np.float32).reshape(N_V, N_U)


def test_golden_thermal_field_cells(golden: GoldenStore, cells: np.ndarray) -> None:
    golden.check(
        "thermal_field_half_in_sun_cells",
        cells,
        config_hash=SOLVER_KEY,
        atol=1e-3,
        units="K (1 mK)",
    )
    lit, shaded = float(cells[:, N_U // 2 + 4 :].mean()), float(cells[:, : N_U // 2 - 4].mean())
    assert lit - shaded > 8.0, (lit, shaded)
    # the terminator is smeared rather than a step: that is the conduction operator, in the array
    edge = cells[N_V // 2, N_U // 2 - 1 : N_U // 2 + 1]
    assert 0.0 < float(np.diff(edge)[0]) < lit - shaded


def test_golden_point_wise_frame(
    golden: GoldenStore, cells: np.ndarray, boson_lut: BandLUT
) -> None:
    """A frame whose every pixel takes the field's own value, which is the whole of `PT.17`.

    The temperature plane *is* the field, sampled at one pixel per cell -- the same array
    `point_bridge` puts on pixels in a render, without the renderer. A per-prim frame would be
    one number here; this one carries the terminator.
    """
    cfg = load_sensor_config(BOSON_YAML)
    dumped = cfg.model_dump(mode="json")
    dumped["sensor"]["fpa"].update(width=N_U, height=N_V)
    dumped["sensor"]["optics"]["supersample_factor"] = 1
    pipeline = PipelineConfig.from_sensor(
        SensorConfig.model_validate(dumped),
        MaterialTable.constant(EMISSIVITY),
        lut=boson_lut,
        sensor_seed=77,
    )
    planes = {
        "temperature_k": cells,
        "encoded_t": encode_temperature(cells),
        "normal_dot_view": np.ones((N_V, N_U), np.float32),
        "distance_m": np.full((N_V, N_U), 12.0, np.float32),
        "material_id": np.ones((N_V, N_U), np.int32),
        "sky_view_factor": np.full((N_V, N_U), 0.5, np.float32),
    }
    out = run_frame(planes, pipeline, PipelineState(housing_temp_k=pipeline.t_housing_cal_k))
    key = _key(SOLVER_KEY, config_hash(cfg), f"{N_U}x{N_V}")
    assert out.apparent_t is not None and out.dn16 is not None
    golden.check(
        "pointwise_frame_apparent_t", out.apparent_t, config_hash=key, atol=1e-3, units="K"
    )
    golden.check("pointwise_frame_dn16", out.dn16, config_hash=key, atol=1.0, units="DN16")
    # The frame is a picture of the field and not of one number: the two halves are apart in DN.
    assert int(out.dn16[:, -4:].mean()) - int(out.dn16[:, :4].mean()) > 100
