"""PT.11 — lateral conduction between a patch's cells.

Cells on one steel bonnet were independent columns, so a metal panel rendered sharper than
aluminium can be. The operator is held to the analytic spreading of a step in a semi-infinite
sheet to 1 % of the step; the k → 0 limit is the operator-free field bit for bit; the shipped
60 s tick stands (backward Euler) where forward Euler at the same tick diverges; and the scene
builds the operator per material from its own k and thickness, with a per-surface switch.

docs/physics-model.md §6.1, §6.4; ADR 0094, ADR 0102; roadmap PT.11.
"""

from __future__ import annotations

import math
import pathlib

import numpy as np
import pytest
from scipy.special import erf

from irsim.scene import Scene
from irsim.thermal.conduction import lateral_operator
from irsim.thermal.facets import FacetForcing, FacetProperties
from irsim.thermal.surface_field import PlanarPatch, PlanarThermalField

REPO = pathlib.Path(__file__).resolve().parents[2]
EX = np.array([1.0, 0.0, 0.0])
EY = np.array([0.0, 1.0, 0.0])

#: 2 mm aluminium sheet: k = 237 W/mK, ρ c = 2700 × 900.
K_AL, DELTA_AL, C_AL = 237.0, 0.002, 2700.0 * 900.0 * 0.002
ALPHA_AL = K_AL * DELTA_AL / C_AL  # 9.75e-5 m²/s


def _strip(n: int, du: float) -> PlanarPatch:
    return PlanarPatch(
        origin_m=np.zeros(3), u_axis=EX, v_axis=EY, n_u=n, n_v=1, du_m=du, dv_m=du, thickness_m=0.01
    )


def _adiabatic(_t: float) -> FacetForcing:
    return FacetForcing(t_air_k=300.0, h_w_m2_k=0.0)


def _props(n: int, c: float = C_AL) -> FacetProperties:
    return FacetProperties(
        heat_capacity_j_m2_k=np.full(n, c), emissivity=np.zeros(n), solar_absorptivity=np.zeros(n)
    )


# --- the analytic anchor ------------------------------------------------------------------------


def test_a_step_spreads_as_the_semi_infinite_sheet_s_error_function_to_one_percent() -> None:
    """T(x, t) = T̄ + ΔT/2 · erf(x / 2√(αt)) while √(αt) ≪ the strip: 1 % of ΔT at t = 60 s."""
    n, du = 400, 0.005  # a 2 m strip of 5 mm cells
    patch = _strip(n, du)
    x = patch.cell_centres()[:, 0] - 0.5 * n * du
    initial = np.where(x < 0.0, 290.0, 310.0)
    field = PlanarThermalField(
        patch,
        _props(n),
        _adiabatic,
        0.0,
        initial,
        tick_s=1.0,  # fine ticks: this is a test of the operator, not of the tick's error
        conduction=lateral_operator(patch, K_AL, DELTA_AL),
    )
    t = 60.0
    field.advance_to(t)
    got = np.asarray(field.temperature_at(t), dtype=np.float64)
    expected = 300.0 + 10.0 * erf(x / (2.0 * math.sqrt(ALPHA_AL * t)))
    assert math.sqrt(ALPHA_AL * t) < 0.1 * (0.5 * n * du)  # the sheet is still semi-infinite
    assert float(np.max(np.abs(got - expected))) < 0.01 * 20.0, float(
        np.max(np.abs(got - expected))
    )
    # Energy is conserved: no losses, and the mean is the mean.
    assert float(got.mean()) == pytest.approx(300.0, abs=1e-6)


def test_the_operator_is_k_delta_times_the_aspect_ratio_per_edge() -> None:
    patch = PlanarPatch(
        origin_m=np.zeros(3),
        u_axis=EX,
        v_axis=EY,
        n_u=3,
        n_v=2,
        du_m=0.1,
        dv_m=0.25,
        thickness_m=0.01,
    )
    op = lateral_operator(patch, 45.0, 0.0012)
    k = op.conductance_w_k.toarray()
    assert np.array_equal(k, k.T) and not k.diagonal().any()
    assert k[0, 1] == pytest.approx(45.0 * 0.0012 * 0.25 / 0.1)  # along u: shared side dv, gap du
    assert k[0, 3] == pytest.approx(45.0 * 0.0012 * 0.1 / 0.25)  # along v
    assert k[0, 2] == 0.0 and k[0, 4] == 0.0  # not neighbours
    assert k[1, 4] > 0.0 and (k > 0).sum() == 2 * (2 * 2 + 3 * 1)


def test_k_to_zero_is_the_operator_free_field_bit_for_bit() -> None:
    assert lateral_operator(_strip(8, 0.05), 0.0, 0.002) is None
    assert lateral_operator(_strip(8, 0.05), 45.0, 0.0) is None
    patch = _strip(16, 0.05)
    x = patch.cell_centres()[:, 0]
    initial = 300.0 + 5.0 * np.sin(x / 0.3)
    forcing = FacetForcing(t_air_k=295.0, h_w_m2_k=8.0, q_longwave_down_w_m2=300.0)
    plain = PlanarThermalField(patch, _props(16), lambda t: forcing, 0.0, initial, 60.0)
    with_none = PlanarThermalField(
        patch,
        _props(16),
        lambda t: forcing,
        0.0,
        initial,
        60.0,
        conduction=lateral_operator(patch, 0.0, 1.0),
    )
    for t in (600.0, 3600.0):
        plain.advance_to(t)
        with_none.advance_to(t)
        assert np.array_equal(plain.temperature_at(t), with_none.temperature_at(t))
    with pytest.raises(ValueError, match="negative"):
        lateral_operator(patch, -1.0, 0.002)


# --- the tick --------------------------------------------------------------------------------


def test_the_shipped_tick_stands_where_forward_euler_diverges() -> None:
    """5 cm of aluminium: the explicit limit is du² C / (4 k δ) ≈ 6 s. At 60 s the implicit
    field obeys the maximum principle and forward Euler on the same operator explodes."""
    n, du = 40, 0.05
    patch = _strip(n, du)
    op = lateral_operator(patch, K_AL, DELTA_AL)
    limit = du * du * C_AL / (4.0 * K_AL * DELTA_AL)
    assert 5.0 < limit < 9.0, limit
    x = patch.cell_centres()[:, 0]
    initial = np.where(x < 1.0, 290.0, 310.0)
    field = PlanarThermalField(patch, _props(n), _adiabatic, 0.0, initial, 60.0, conduction=op)
    field.advance_to(3600.0)
    got = np.asarray(field.temperature_at(3600.0), dtype=np.float64)
    assert got.min() >= 290.0 - 1e-9 and got.max() <= 310.0 + 1e-9
    assert float(got.mean()) == pytest.approx(300.0, abs=1e-6)
    assert float(np.ptp(got)) < 20.0 and float(np.ptp(got)) > 1.0  # spreading, not flat yet

    lap = op.laplacian.toarray()
    explicit = initial.astype(np.float64).copy()
    for _ in range(60):
        explicit = explicit - 60.0 / (C_AL * patch.cell_area_m2) * (lap @ explicit)
    assert not np.all(np.isfinite(explicit)) or float(np.ptp(explicit)) > 1e3


# --- the scene's per-material switch -------------------------------------------------------


@pytest.mark.slow
def test_the_scene_builds_the_operator_from_the_material_and_the_switch_turns_it_off(
    tmp_path,
) -> None:  # type: ignore[no-untyped-def]
    scene = Scene.from_file(REPO / "configs/scenes/wet_road_noon.yaml")
    op = scene.surface_fields["road"].field.conduction
    assert op is not None
    k = op.conductance_w_k
    # asphalt_dry: k = 0.75 W/mK, δ = 0.05 m, square 0.5 m cells → 0.0375 W/K per edge.
    assert float(k.max()) == pytest.approx(0.75 * 0.05)
    assert k.nnz == 2 * (24 * 23 + 23 * 24)
    text = (REPO / "configs/scenes/wet_road_noon.yaml").read_text()
    path = tmp_path / "columns.yaml"
    path.write_text(
        text.replace(
            "        material: asphalt_dry\n",
            "        material: asphalt_dry\n        lateral_conduction: false\n",
        )
    )
    assert Scene.from_file(path).surface_fields["road"].field.conduction is None


@pytest.mark.slow
def test_the_car_bonnet_conducts_laterally_and_its_gradient_softens() -> None:
    """Steel under paint: the engine's hot spot spreads along the skin (99 mm over 750 s), so
    the coupled bonnet is smoother than the independent-column one, with the same mean."""
    from irsim_isaac.car_demo import build_bonnet_field, build_car_demo

    scene = Scene.from_file(REPO / "configs/scenes/car_ignition_overcast_night.yaml")
    demo = build_car_demo(scene, author=False, spin_up=False)
    columns = build_bonnet_field(
        scene, demo.geometry, emissivity=0.92, patch=demo.bonnet_field.patch, conductivity_w_mk=0.0
    )
    assert demo.bonnet_field.field.conduction is not None and columns.field.conduction is None
    t = scene.t0_s
    for k in range(120):
        scene.advance_targets(10.0 * k, 10.0)
    demo.bonnet_field.advance_to(t + 1200.0)
    columns.advance_to(t + 1200.0)
    smooth = np.asarray(demo.bonnet_field.temperature_image(t + 1200.0), dtype=np.float64)
    sharp = np.asarray(columns.temperature_image(t + 1200.0), dtype=np.float64)

    def roughness(img: np.ndarray) -> float:
        return float(np.abs(np.diff(img, axis=1)).mean() + np.abs(np.diff(img, axis=0)).mean())

    # 5 % on this already-smooth view-factor profile: 99 mm over 750 s is one 70 mm cell of
    # spreading in steel. Aluminium (270 mm) would show far more; the material is PT.6's.
    assert roughness(smooth) < 0.97 * roughness(sharp), (roughness(smooth), roughness(sharp))
    assert float(np.ptp(smooth)) < float(np.ptp(sharp))
    assert float(smooth.mean()) == pytest.approx(float(sharp.mean()), abs=0.5)
