"""ADR 0187: a heat source bolted to a solved surface -- a motor's heat into the end of its arm.

Three layers, each checked where it can fail:

  * **the boundary** (`ConductionOperator.boundary_w_k`): a cell bolted to a body at T_b relaxes to
    it at exactly ``g / (C A)``, one backward-Euler step balances to rounding, a step two hundred
    time constants long lands on T_b without overshooting, and a zero boundary is no boundary;
  * **the mount** (`irsim.thermal.mounts`): the footprint conducts ``h_c · area`` in total on a
    tube as on a grid, and the motor is off -- the boundary is the air -- outside its mission,
    which is what keeps the spin-up from running yesterday's flight through the weather wrap;
  * **the arm** (`aerial_clear_exit.yaml`, the east arm alone): with the mount the tube is warmest
    under the motor, crown and underside both, and never above the motor; without it the same cells
    sit in the motor's shadow, colder than the open span -- the profile the first EV.16 render
    showed and the one a real airframe does not.
"""

from __future__ import annotations

import math
import pathlib

import numpy as np
import pytest
import scipy.sparse as sp
import yaml

from irsim.config.scene import load_scene_config
from irsim.scene import Scene
from irsim.thermal.conduction import ConductionOperator
from irsim.thermal.facets import FacetForcing, FacetProperties, FacetSolver
from irsim.thermal.mounts import Mount, MountedForcing, mount_conductances

REPO = pathlib.Path(__file__).resolve().parents[2]
SCENE = REPO / "configs" / "scenes" / "aerial_clear_exit.yaml"

#: One cell of 1.5 mm carbon: its areal capacity, a 10 cm^2 face, a 1000 W m^-2 K^-1 joint over it.
C_J_M2_K = 2520.0
AREA_M2 = 1.0e-3
G_W_K = 1000.0 * AREA_M2
TAU_S = C_J_M2_K * AREA_M2 / G_W_K  # 2.52 s


def _solver(n: int = 1, boundary: float | None = G_W_K, links: float = 0.0) -> FacetSolver:
    """Facets that neither convect, radiate nor see the sun: conduction and the boundary only."""
    props = FacetProperties(
        heat_capacity_j_m2_k=np.full(n, C_J_M2_K),
        emissivity=np.zeros(n),
        solar_absorptivity=np.zeros(n),
    )
    k = sp.diags([links], [1], shape=(n, n)) if n > 1 else sp.csr_matrix((n, n))
    op = ConductionOperator(k + k.T, np.full(n, AREA_M2))
    if boundary is not None:
        g = np.zeros(n)
        g[0] = boundary
        op = op.with_boundary(g)
    return FacetSolver(props, np.full(n, 300.0), conduction=op)


def _still(boundary_k: float | None = 350.0) -> FacetForcing:
    return FacetForcing(t_air_k=300.0, h_w_m2_k=0.0, boundary_k=boundary_k)


# -- the boundary ---------------------------------------------------------------------------------


def test_a_bolted_cell_relaxes_to_the_body_at_g_over_c_a() -> None:
    """T(t) = T_b + (T0 - T_b) e^(-t/tau), tau = C A / g, to backward Euler's first-order error."""
    solver = _solver()
    dt = TAU_S / 400.0
    for _ in range(400):
        solver.advance(_still(), dt)
    expected = 350.0 - 50.0 * math.exp(-1.0)
    assert float(solver.temperatures_k[0]) == pytest.approx(expected, abs=0.05)


def test_one_step_balances_to_rounding_and_a_long_one_does_not_overshoot() -> None:
    """The implicit step's own energy statement, C A (T1 - T0) / dt = g (T_b - T1); and at
    dt = 238 tau (a 600 s tick on a 2.5 s joint) the cell lands within 50/239 K of T_b from below,
    where an explicit step would have flung it hundreds of kelvin past."""
    solver = _solver()
    t1 = float(solver.advance(_still(), 1.0)[0])
    assert C_J_M2_K * AREA_M2 * (t1 - 300.0) / 1.0 == pytest.approx(G_W_K * (350.0 - t1), rel=1e-9)
    long = _solver()
    t = float(long.advance(_still(), 600.0)[0])
    assert 350.0 - 50.0 / 239.0 - 1e-9 <= t <= 350.0


def test_a_zero_boundary_is_no_boundary_bit_for_bit() -> None:
    plain, zero = _solver(3, None, links=0.05), _solver(3, 0.0, links=0.05)
    for _ in range(20):
        plain.advance(FacetForcing(t_air_k=300.0, h_w_m2_k=0.0, q_internal_w_m2=[50.0, 0, 0]), 5.0)
        zero.advance(
            FacetForcing(t_air_k=300.0, h_w_m2_k=0.0, q_internal_w_m2=[50.0, 0, 0], boundary_k=1.0),
            5.0,
        )
    assert np.array_equal(plain.temperatures_k, zero.temperatures_k)


def test_a_boundary_with_no_temperature_is_refused_not_read_as_zero() -> None:
    with pytest.raises(ValueError, match="no boundary_k"):
        _solver().advance(_still(None), 1.0)
    with pytest.raises(ValueError, match="not negative"):
        ConductionOperator(sp.csr_matrix((1, 1)), [1.0], boundary_w_k=[-1.0])


# -- the mount ------------------------------------------------------------------------------------


def _tube() -> object:
    from irsim.scene import build_mesh

    spec = load_scene_config(SCENE).scene
    assert spec.thermal is not None
    arm = next(s for s in spec.thermal.surfaces if s.name == "arm_e")
    assert arm.mesh is not None
    return build_mesh(arm.mesh)


def _mount(covers: tuple[float, float] = (0.0, 1e12), temperature_k: float = 320.0) -> Mount:
    return Mount(
        name="motor",
        centre_m=(0.42, 0.335, 0.0),
        area_m2=0.002,
        h_c_w_m2_k=1000.0,
        temperature_at=lambda t: temperature_k,
        covers=lambda t: covers[0] <= t <= covers[1],
    )


def test_the_footprint_is_h_c_times_its_area_on_the_tube_and_sits_under_the_motor() -> None:
    tube = _tube()
    g = mount_conductances(tube, [_mount()])
    assert float(g.sum()) == pytest.approx(1000.0 * 0.002, rel=1e-12)
    centres = np.asarray(tube.cell_centres())  # type: ignore[attr-defined]
    touched = centres[g > 0.0]
    # within the motor's 60 mm along the tube, and wrapping down from the crown no further than
    # a clamp does: above the lower quarter of the 15 mm tube, centred on the top
    assert float(np.abs(touched[:, 0] - 0.42).max()) < 0.045
    assert float(touched[:, 1].min()) > 0.32 - 0.5 * 0.015
    assert float(touched[:, 1].mean()) > 0.32 + 0.5 * 0.015


def test_the_motor_is_off_outside_its_mission() -> None:
    """Outside the schedule the boundary is the air; inside, the owned cells read the motor.
    The mount reads the instant it is *asked*, so a spin-up through the weather wrap -- which hands
    the inner forcing yesterday's clock -- cannot run the motors through yesterday's flight."""
    tube = _tube()
    asked: list[float] = []

    def inner(t: float) -> FacetForcing:
        asked.append(t)
        return FacetForcing(t_air_k=290.0, h_w_m2_k=10.0)

    forcing = MountedForcing(inner, [_mount(covers=(100.0, 200.0))], tube)
    owned = mount_conductances(tube, [_mount()]) > 0.0
    before, during = forcing(50.0).boundary_k, forcing(150.0).boundary_k
    assert np.all(before == 290.0)
    assert np.all(during[owned] == 320.0) and np.all(during[~owned] == 290.0)
    assert asked == [50.0, 150.0]


# -- the arm --------------------------------------------------------------------------------------


def _one_arm(tmp_path: pathlib.Path, mounted: bool) -> object:
    """The scene's east arm alone, at cruise, with or without its motor bolted on."""
    raw = yaml.safe_load(SCENE.read_text())
    thermal = raw["scene"]["thermal"]
    thermal["surfaces"] = [s for s in thermal["surfaces"] if s["name"] == "arm_e"]
    thermal["mounts"] = [m for m in thermal["mounts"] if m["surface"] == "arm_e"] if mounted else []
    thermal["spin_up_hours"] = 12.0  # the arm's memory is minutes; both variants share it
    path = tmp_path / f"arm_e_{mounted}.yaml"
    path.write_text(yaml.safe_dump(raw))
    scene = Scene.from_file(path)
    field = scene.mesh_fields["arm_e"]
    t = scene.t0_s + 900.0
    field.advance_to(t)
    motor = scene.advance_targets(900.0, 0.0)["motor"]
    return field, np.asarray(field.temperature_at(t), dtype=np.float64), float(motor)


@pytest.fixture(scope="module")
def arms(tmp_path_factory: pytest.TempPathFactory) -> dict[bool, object]:
    tmp = tmp_path_factory.mktemp("arms")
    return {flag: _one_arm(tmp, flag) for flag in (True, False)}


def _regions(field: object) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    patch = field.patch  # type: ignore[attr-defined]
    centres = np.asarray(patch.cell_centres())
    up = np.asarray(patch.cell_normal)[:, 1]
    under_motor = np.abs(centres[:, 0] - 0.42) < 0.025
    open_span = (centres[:, 0] > 0.22) & (centres[:, 0] < 0.32)
    return under_motor, open_span, up > 0.7, up < -0.7


def test_with_the_mount_the_arm_is_warmest_under_its_motor(arms) -> None:  # type: ignore[no-untyped-def]
    field, temps, motor = arms[True]
    under_motor, open_span, crown, underside = _regions(field)
    for face in (crown, underside):
        assert temps[under_motor & face].mean() > temps[open_span & face].mean() + 2.0
    # heat flows down the gradient: nothing on the arm is hotter than the motor driving it
    assert float(temps.max()) < motor


def test_without_it_the_same_cells_sit_in_the_motor_s_shadow(arms) -> None:  # type: ignore[no-untyped-def]
    """The control: the profile the first EV.16 render showed. Same arm, same sun, same air."""
    field, temps, _ = arms[False]
    under_motor, open_span, crown, _ = _regions(field)
    assert temps[under_motor & crown].mean() < temps[open_span & crown].mean()
    mounted = arms[True][1]
    assert mounted[under_motor].mean() > temps[under_motor].mean() + 5.0


# -- the schema -----------------------------------------------------------------------------------


def _raw_with(**mount: object) -> dict:  # type: ignore[type-arg]
    raw = yaml.safe_load(SCENE.read_text())
    raw["scene"]["thermal"]["mounts"] = [mount]
    return raw


@pytest.mark.parametrize(
    ("mount", "message"),
    [
        (
            {
                "target": "motor",
                "surface": "nowhere",
                "centre_m": [0, 0, 0],
                "area_m2": 0.002,
                "joint": "dry_default",
            },
            "must exist",
        ),
        (
            {
                "target": "airframe",
                "surface": "arm_e",
                "centre_m": [0, 0, 0],
                "area_m2": 0.002,
                "joint": "dry_default",
            },
            "heat_source",
        ),
        (
            {
                "target": "motor",
                "surface": "arm_e",
                "centre_m": [0, 0, 0],
                "area_m2": 0.002,
                "joint": "dry_default",
                "h_c_w_m2_k": 1000.0,
            },
            "exactly one",
        ),
    ],
)
def test_the_schema_refuses_a_mount_it_cannot_honour(mount, message) -> None:  # type: ignore[no-untyped-def]
    from irsim.config.scene import SceneConfig

    with pytest.raises(ValueError, match=message):
        SceneConfig.model_validate(_raw_with(**mount))


def test_a_mount_beside_the_object_exchange_is_refused() -> None:
    from irsim.config.scene import SceneConfig

    raw = yaml.safe_load(SCENE.read_text())
    raw["scene"]["thermal"]["object_exchange"] = True
    with pytest.raises(ValueError, match="object_exchange"):
        SceneConfig.model_validate(raw)
