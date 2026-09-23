"""Freeze the G-buffer contract and verify the synthetic fixtures are what they claim.

The key set and dtypes asserted here are what the Isaac annotator adapter must emit; changing
them is an interface change for every kernel and for the engine glue, so it must be deliberate.
The fixture-geometry tests matter because a fixture that never reaches grazing incidence, or a
step edge that is already blurred, would let a wrong angular-emissivity or MTF model pass.
"""

from __future__ import annotations

import numpy as np
import pytest

from irsim.config.gbuffer import (
    OPTIONAL_KEYS,
    REQUIRED_KEYS,
    UNMAPPED_MATERIAL_ID,
    GBuffer,
)
from irsim.radiometry.encoding import encode_temperature

FROZEN_REQUIRED = {
    "temperature_k",
    "normal_dot_view",
    "distance_m",
    "material_id",
    "sky_view_factor",
}
# `shadow_mask` and `sun_cos_incidence` joined the contract with the reflected-solar term
# (M11.3, §5.4). Both are optional, so every fixture and adapter written before them is
# unchanged; a scene that has not asked for sunlight renders identically.
FROZEN_OPTIONAL = {
    "encoded_t",
    "motion_px",
    "semantic_id",
    "sky_mask",
    "shadow_mask",
    "sun_cos_incidence",
    # Added deliberately by AT.1, which is what this frozen set is for: the per-pixel ray
    # elevation, so the atmosphere can take each pixel's own slant path instead of the
    # horizontal one it used for every resolved pixel.
    "elevation_rad",
    # Added deliberately by `OC.7`: the apparent temperature each ray would report with all
    # geometry removed. A defocused foreground silhouette lets the background through and a
    # single-layer G-buffer has no behind, so without this plane the layered defocus stage has to
    # guess. For sky and sea it is not a guess and not a second render either.
    "background_t_k",
}
FROZEN_DTYPES = {
    "temperature_k": np.float32,
    "encoded_t": np.float32,
    "normal_dot_view": np.float32,
    "distance_m": np.float32,
    "elevation_rad": np.float32,
    "material_id": np.int32,
    "sky_view_factor": np.float32,
    "motion_px": np.float32,
    "semantic_id": np.uint32,
    "sky_mask": np.bool_,
    "shadow_mask": np.float32,
    "sun_cos_incidence": np.float32,
}
SINGLE_FRAME_FIXTURES = [
    "gbuffer_ramp",
    "gbuffer_uniform",
    "gbuffer_two_material",
    "gbuffer_sphere",
    "gbuffer_step_edge",
]


def test_contract_key_set_is_frozen() -> None:
    assert set(REQUIRED_KEYS) == FROZEN_REQUIRED
    assert set(OPTIONAL_KEYS) == FROZEN_OPTIONAL


@pytest.mark.parametrize("name", SINGLE_FRAME_FIXTURES)
def test_fixture_honours_contract(name: str, request: pytest.FixtureRequest) -> None:
    planes = request.getfixturevalue(name)
    assert FROZEN_REQUIRED <= planes.keys() <= FROZEN_REQUIRED | FROZEN_OPTIONAL
    for key, arr in planes.items():
        assert arr.dtype == FROZEN_DTYPES[key], f"{name}[{key}] is {arr.dtype}"
    # encoded_t is OPTIONAL in the contract: the Isaac adapter never emits it (ADR 0014 -- the
    # renderer transports ids + geometry and temperature comes from a float32 table). Fixtures
    # carry it only to exercise the encode/decode consistency check.
    assert "encoded_t" in planes
    assert np.all(planes["material_id"] != UNMAPPED_MATERIAL_ID), "id 0 is the UNMAPPED sentinel"
    gb = GBuffer.from_dict(planes)
    assert gb.shape == planes["temperature_k"].shape
    assert gb.to_dict().keys() == planes.keys()


def test_moving_edge_frames_honour_contract(
    gbuffer_moving_edge: list[dict[str, np.ndarray]],
) -> None:
    for planes in gbuffer_moving_edge:
        gb = GBuffer.from_dict(planes)
        assert gb.motion_px is not None and gb.motion_px.shape == (*gb.shape, 2)


# --- boundary assertions (non-negotiable #2) ---------------------------------------------


@pytest.mark.parametrize("key", ["temperature_k", "encoded_t", "distance_m"])
def test_float16_on_precision_path_raises(gbuffer_uniform: dict[str, np.ndarray], key: str) -> None:
    planes = dict(gbuffer_uniform)
    planes[key] = planes[key].astype(np.float16)
    with pytest.raises(TypeError, match="float16"):
        GBuffer.from_dict(planes)


def test_float16_radiance_plane_raises(gbuffer_uniform: dict[str, np.ndarray]) -> None:
    planes = dict(gbuffer_uniform)
    planes["radiance_lwir"] = np.ones(planes["temperature_k"].shape, dtype=np.float16)
    with pytest.raises(TypeError, match="float16"):
        GBuffer.from_dict(planes)


@pytest.mark.parametrize("key", ["normal_dot_view", "sky_view_factor"])
def test_float16_ancillary_planes_are_upcast(
    gbuffer_uniform: dict[str, np.ndarray], key: str
) -> None:
    planes = dict(gbuffer_uniform)
    planes[key] = planes[key].astype(np.float16)
    gb = GBuffer.from_dict(planes)
    assert getattr(gb, key).dtype == np.float32


def test_float64_temperature_is_narrowed_to_float32(gbuffer_uniform: dict[str, np.ndarray]) -> None:
    planes = dict(gbuffer_uniform)
    planes["temperature_k"] = planes["temperature_k"].astype(np.float64)
    assert GBuffer.from_dict(planes).temperature_k.dtype == np.float32


def test_inconsistent_encoded_t_raises(gbuffer_uniform: dict[str, np.ndarray]) -> None:
    planes = dict(gbuffer_uniform)
    planes["encoded_t"] = encode_temperature(planes["temperature_k"] + np.float32(0.5))
    with pytest.raises(ValueError, match="encoded_t"):
        GBuffer.from_dict(planes)


def test_missing_unknown_and_misshapen_planes_raise(gbuffer_uniform: dict[str, np.ndarray]) -> None:
    planes = dict(gbuffer_uniform)
    del planes["distance_m"]
    with pytest.raises(KeyError, match="distance_m"):
        GBuffer.from_dict(planes)
    planes = dict(gbuffer_uniform)
    planes["albedo"] = planes["sky_view_factor"]
    with pytest.raises(KeyError, match="albedo"):
        GBuffer.from_dict(planes)
    planes = dict(gbuffer_uniform)
    planes["distance_m"] = planes["distance_m"][:32]
    with pytest.raises(ValueError, match="shape"):
        GBuffer.from_dict(planes)
    planes = dict(gbuffer_uniform)
    planes["material_id"] = planes["material_id"].astype(np.float32)
    with pytest.raises(TypeError, match="integer"):
        GBuffer.from_dict(planes)


# --- fixture geometry -----------------------------------------------------------------------


def test_sphere_fixture_matches_analytic_cos_theta(
    gbuffer_sphere: dict[str, np.ndarray], sphere_params: dict[str, float]
) -> None:
    SPHERE_RADIUS_PX = sphere_params["radius_px"]
    h, w = gbuffer_sphere["temperature_k"].shape
    yy, xx = np.mgrid[0:h, 0:w]
    r2 = (yy - h // 2) ** 2 + (xx - w // 2) ** 2
    inside = gbuffer_sphere["material_id"] == 1
    assert np.array_equal(inside, r2 <= SPHERE_RADIUS_PX**2)
    expected = np.sqrt(1.0 - r2[inside] / SPHERE_RADIUS_PX**2)
    got = gbuffer_sphere["normal_dot_view"][inside].astype(np.float64)
    assert np.max(np.abs(got - expected)) < 1e-6
    assert got.max() == 1.0 and got.min() < 0.02, "fixture must reach grazing incidence"
    assert np.mean(got < 0.2) > 0.01, "too few grazing pixels to test epsilon(theta)"
    assert np.all(gbuffer_sphere["normal_dot_view"][~inside] == 1.0)


def test_step_edge_is_ideal(
    gbuffer_step_edge: dict[str, np.ndarray], step_edge_params: dict[str, float]
) -> None:
    t = gbuffer_step_edge["temperature_k"]
    assert t.shape == (step_edge_params["size_px"],) * 2
    assert set(np.unique(t).tolist()) == {step_edge_params["cold_k"], step_edge_params["hot_k"]}
    transitions = np.count_nonzero(np.diff(t, axis=1), axis=1)
    assert np.all(transitions == 1), "exactly one cold->hot transition per row"
    # fit the edge: first hot column per row vs row index
    first_hot = np.argmax(t == step_edge_params["hot_k"], axis=1).astype(np.float64)
    rows = np.arange(t.shape[0], dtype=np.float64)
    slope = np.polyfit(rows, first_hot, 1)[0]
    angle = np.degrees(np.arctan(slope))
    assert abs(angle - step_edge_params["angle_deg"]) < 0.1, f"fitted {angle:.3f} deg"


def test_moving_edge_translates_at_stated_velocity(
    gbuffer_moving_edge: list[dict[str, np.ndarray]], step_edge_params: dict[str, float]
) -> None:
    MOVING_EDGE_VELOCITY_PX = step_edge_params["velocity_px"]
    assert len(gbuffer_moving_edge) == step_edge_params["n_frames"]
    positions = []
    for planes in gbuffer_moving_edge:
        hot = planes["temperature_k"] == planes["temperature_k"].max()
        positions.append(float(np.mean(np.argmax(hot, axis=1))))
        assert np.all(planes["motion_px"][..., 0] == MOVING_EDGE_VELOCITY_PX)
        assert np.all(planes["motion_px"][..., 1] == 0.0)
    steps = np.diff(positions)
    assert np.all(np.abs(steps - MOVING_EDGE_VELOCITY_PX) < 0.05), steps


def test_sky_mask_plane_contract(gbuffer_uniform: dict[str, np.ndarray]) -> None:
    """sky_mask: bool or 0/1 integer in, bool out; floats and other integers refused; shape
    checked; absent means every pixel is geometry (sky_mask is None)."""
    assert GBuffer.from_dict(dict(gbuffer_uniform)).sky_mask is None
    h, w = gbuffer_uniform["temperature_k"].shape
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[: h // 2] = 1
    g = GBuffer.from_dict({**gbuffer_uniform, "sky_mask": mask})
    assert g.sky_mask is not None and g.sky_mask.dtype == np.bool_
    assert g.sky_mask[0, 0] and not g.sky_mask[-1, 0]
    assert g.to_dict()["sky_mask"].dtype == np.bool_
    g2 = GBuffer.from_dict({**gbuffer_uniform, "sky_mask": mask.astype(bool)})
    assert g2.sky_mask is not None and np.array_equal(g2.sky_mask, g.sky_mask)
    with pytest.raises(TypeError, match="bool"):
        GBuffer.from_dict({**gbuffer_uniform, "sky_mask": mask.astype(np.float32)})
    with pytest.raises(TypeError, match="bool"):
        GBuffer.from_dict({**gbuffer_uniform, "sky_mask": mask * 2})
    with pytest.raises(ValueError, match="shape"):
        GBuffer.from_dict({**gbuffer_uniform, "sky_mask": mask[:, : w // 2]})
