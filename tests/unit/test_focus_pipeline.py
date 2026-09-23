"""OC.5 -- the global defocus kernel, through `run_frame` (ADR 0129).

`OC.3` showed the physics on a synthetic plane. This is the same physics reached the way a render
reaches it: config -> `PipelineConfig.from_sensor` -> `run_frame`, with the kernel chosen from the
G-buffer's own `distance_m`.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from irsim.config.loader import load_sensor_config
from irsim.config.sensor import FULL_FIDELITY, SensorConfig
from irsim.materials import MaterialTable
from irsim.optics.defocus import blur_circle_um, defocus_w020_um, scene_defocus_um
from irsim.pipeline import PipelineConfig, PipelineState, run_frame
from irsim.radiometry.encoding import encode_temperature

REPO = pathlib.Path(__file__).resolve().parents[2]
BOSON_YAML = REPO / "configs" / "sensors" / "flir_boson_640_lwir.yaml"
SCENE_RANGE_M = 4.0


def _camera(model: str = "none", focus: dict | None = None, **fidelity: bool) -> SensorConfig:
    d = load_sensor_config(BOSON_YAML).model_dump(mode="json")
    d["sensor"]["fpa"].update(width=64, height=64)
    d["sensor"]["optics"]["supersample_factor"] = 1
    d["sensor"]["optics"]["mtf"]["defocus_model"] = model
    if focus is not None:
        d["sensor"]["optics"]["focus"] = focus
    if fidelity:
        d["sensor"]["fidelity"] = {**FULL_FIDELITY.model_dump(), **fidelity}
    return SensorConfig.model_validate(d)


def _scene(distance_m: float = SCENE_RANGE_M, sky: bool = False) -> dict[str, np.ndarray]:
    t = np.full((64, 64), 280.0, np.float32)
    t[:, 32:] = 330.0  # a vertical edge, so an edge width is measurable
    planes = {
        "temperature_k": t,
        "encoded_t": encode_temperature(t),
        "normal_dot_view": np.ones((64, 64), np.float32),
        "distance_m": np.full((64, 64), distance_m, np.float32),
        "material_id": np.ones((64, 64), np.int32),
        "sky_view_factor": np.zeros((64, 64), np.float32),
    }
    if sky:  # the gbuffer convention: sky is distance 0 with the mask set
        planes["sky_mask"] = np.zeros((64, 64), bool)
        planes["sky_mask"][:, :48] = True
        planes["distance_m"] = np.where(planes["sky_mask"], 0.0, distance_m).astype(np.float32)
    return planes


@pytest.fixture(scope="module")
def materials() -> MaterialTable:
    return MaterialTable.constant(1.0)


def _config(sensor: SensorConfig, materials: MaterialTable, boson_lut) -> PipelineConfig:
    return PipelineConfig.from_sensor(sensor, materials, lut=boson_lut, noise_enabled=False)


def _edge_width_px(row: np.ndarray) -> float:
    lo, hi = float(row.min()), float(row.max())
    p10, p90 = lo + 0.1 * (hi - lo), lo + 0.9 * (hi - lo)
    x = np.arange(row.size, dtype=np.float64)
    return float(np.interp(p90, row, x) - np.interp(p10, row, x))


def test_a_camera_that_names_no_model_gets_no_bank(materials, boson_lut) -> None:
    """The pre-v10 path, untouched: `config.psf` is used exactly as it was."""
    config = _config(_camera(), materials, boson_lut)
    assert config.defocus_bank is None
    assert config.focus_distance_m is None
    state = PipelineState()
    run_frame(_scene(), config, state)
    assert state.defocus_w020_um == 0.0


def test_focusing_on_the_scene_sharpens_the_edge(materials, boson_lut) -> None:
    focused = _config(
        _camera("hopkins", {"mode": "fixed", "distance_m": SCENE_RANGE_M}), materials, boson_lut
    )
    at_infinity = _config(_camera("hopkins"), materials, boson_lut)
    rows = []
    for config in (focused, at_infinity):
        out = run_frame(_scene(), config, PipelineState())
        rows.append(_edge_width_px(np.asarray(out.signal_dn, dtype=np.float64)[32, :]))
    sharp, blurred = rows
    assert sharp < blurred, "focusing on the scene must sharpen it"
    assert blurred / sharp > 1.5, f"and by a visible amount: {sharp:.2f} -> {blurred:.2f} px"


def test_the_frame_records_the_defocus_it_used(materials, boson_lut) -> None:
    """The reported W020 is the analytic one for the frame's own median range, not an estimate."""
    config = _config(_camera("hopkins"), materials, boson_lut)
    state = PipelineState()
    run_frame(_scene(), config, state)
    want = float(defocus_w020_um(blur_circle_um(SCENE_RANGE_M, 14.0, 1.0, None), 1.0))
    assert state.defocus_w020_um == pytest.approx(want, rel=1e-9)
    assert want == pytest.approx(6.125, abs=0.01)  # c = 49 µm at 4 m, over 8F


def test_sky_is_not_read_as_the_nearest_thing_in_the_scene(materials, boson_lut) -> None:
    """Sky carries distance_m = 0. A median that counted it would defocus the frame hardest of
    all -- and three quarters of this scene is sky."""
    config = _config(_camera("hopkins"), materials, boson_lut)
    state = PipelineState()
    run_frame(_scene(sky=True), config, state)
    geometry_only = float(defocus_w020_um(blur_circle_um(SCENE_RANGE_M, 14.0, 1.0, None), 1.0))
    assert state.defocus_w020_um == pytest.approx(geometry_only, rel=1e-9)

    # and a frame with nothing but sky takes the defocus of an object at infinity, i.e. none
    all_sky = {**_scene(), "sky_mask": np.ones((64, 64), bool)}
    all_sky["distance_m"] = np.zeros((64, 64), np.float32)
    assert scene_defocus_um(all_sky["distance_m"], 14.0, 1.0, None, all_sky["sky_mask"]) < 1e-6


def test_the_bank_quantises_below_what_the_box_filter_can_resolve(materials, boson_lut) -> None:
    config = _config(_camera("hopkins"), materials, boson_lut)
    bank = config.defocus_bank
    assert bank is not None
    # a quarter of a supersample cell of blur circle, expressed in W020: c = 8 F W020
    assert 8.0 * bank.f_number * bank.step_um == pytest.approx(12.0 / 4.0, rel=1e-12)
    state = PipelineState()
    for _ in range(4):
        run_frame(_scene(), config, state)
    assert len(bank) == 1, "an unchanging scene must not rebuild the kernel"
    run_frame(_scene(distance_m=40.0), config, state)
    assert len(bank) == 2, "a different range must get a different kernel"


def test_the_ablation_switch_takes_the_defocus_away(materials, boson_lut) -> None:
    on = _config(_camera("hopkins"), materials, boson_lut)
    off = _config(_camera("hopkins", defocus=False), materials, boson_lut)
    assert on.defocus_bank is not None and off.defocus_bank is None
    a = run_frame(_scene(), on, PipelineState()).signal_dn
    b = run_frame(_scene(), off, PipelineState()).signal_dn
    assert not np.array_equal(a, b), "the ablation must actually change the picture"


def test_the_layered_switch_reaches_the_frame(materials, boson_lut) -> None:
    """`OC.6` wiring, on a scene where it can be told apart from `OC.5`.

    The near slab is the **majority** of the frame, so the median range `OC.5` picks is the near
    one and its single kernel smears the far background's own edge. Layering must not. A scene
    where the median lands on the background would show nothing, because the global kernel would
    then already be the in-focus one -- which is the trap this test was written into once.
    """
    t_k = np.full((64, 64), 280.0, np.float32)
    t_k[:, :40] = 350.0  # the near slab
    t_k[:, 52:] = 330.0  # a step in the far background, clear of the silhouette
    distance = np.full((64, 64), 200.0, np.float32)
    distance[:, :40] = 3.0
    scene = {
        "temperature_k": t_k,
        "encoded_t": encode_temperature(t_k),
        "normal_dot_view": np.ones((64, 64), np.float32),
        "distance_m": distance,
        "material_id": np.ones((64, 64), np.int32),
        "sky_view_factor": np.zeros((64, 64), np.float32),
    }

    def _run(how: str) -> np.ndarray:
        d = load_sensor_config(BOSON_YAML).model_dump(mode="json")
        d["sensor"]["fpa"].update(width=64, height=64)
        d["sensor"]["optics"]["supersample_factor"] = 1
        d["sensor"]["optics"]["mtf"].update(defocus_model="hopkins", defocus_apply=how)
        config = _config(SensorConfig.model_validate(d), materials, boson_lut)
        return np.asarray(run_frame(scene, config, PipelineState()).signal_dn, dtype=np.float64)

    glob, layered = _run("global"), _run("layered")
    assert not np.array_equal(glob, layered), "the switch must reach the frame"
    sharp = _edge_width_px(layered[32, 46:60])
    smeared = _edge_width_px(glob[32, 46:60])
    assert sharp < smeared, (
        f"the far background must survive layering: {sharp:.2f} vs {smeared:.2f}"
    )


def test_the_background_plane_reaches_the_layered_stage(materials, boson_lut) -> None:
    """`OC.7` wiring: a `background_t_k` plane must change the defocused silhouette, and must do
    so only there -- it is the answer to what is behind the foreground, not a global offset."""
    t_k = np.full((64, 64), 280.0, np.float32)
    t_k[:, :40] = 350.0
    distance = np.full((64, 64), 200.0, np.float32)
    distance[:, :40] = 3.0
    scene = {
        "temperature_k": t_k,
        "encoded_t": encode_temperature(t_k),
        "normal_dot_view": np.ones((64, 64), np.float32),
        "distance_m": distance,
        "material_id": np.ones((64, 64), np.int32),
        "sky_view_factor": np.zeros((64, 64), np.float32),
    }
    d = load_sensor_config(BOSON_YAML).model_dump(mode="json")
    d["sensor"]["fpa"].update(width=64, height=64)
    d["sensor"]["optics"]["supersample_factor"] = 1
    d["sensor"]["optics"]["mtf"].update(defocus_model="hopkins", defocus_apply="layered")
    config = _config(SensorConfig.model_validate(d), materials, boson_lut)

    without = np.asarray(run_frame(scene, config, PipelineState()).signal_dn, dtype=np.float64)
    with_bg = np.asarray(
        run_frame(
            {**scene, "background_t_k": np.full((64, 64), 280.0, np.float32)},
            config,
            PipelineState(),
        ).signal_dn,
        dtype=np.float64,
    )
    delta = np.abs(with_bg - without)
    assert delta.max() > 1.0, "the background must reach the composite"
    assert delta[:, :20].max() < 0.02 * delta.max(), "deep inside the slab nothing should change"
    assert delta[:, 55:].max() < 0.02 * delta.max(), "and nothing in the open background either"
