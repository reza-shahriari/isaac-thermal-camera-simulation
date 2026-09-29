"""IG.18 -- `GBuffer` matches its own contract.

`config/gbuffer.py` listed `background_t_k` as optional but the dataclass had no such field, so
`from_dict` raised on a plane the contract allowed; and `to_dict` omitted `elevation_rad`, so a
validated G-buffer lost its slant plane and the adapter re-added it by hand. The key set, the
dataclass and the round trip are one thing now. Roadmap IG.18; ADR 0014 (the contract).
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from irsim.config.gbuffer import OPTIONAL_KEYS, REQUIRED_KEYS, GBuffer


def _planes(h: int = 3, w: int = 4) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(7)
    planes: dict[str, np.ndarray] = {
        "temperature_k": rng.uniform(280.0, 310.0, (h, w)).astype(np.float32),
        "normal_dot_view": rng.uniform(0.1, 1.0, (h, w)).astype(np.float32),
        "distance_m": rng.uniform(10.0, 500.0, (h, w)).astype(np.float32),
        "material_id": rng.integers(1, 5, (h, w)).astype(np.int32),
        "sky_view_factor": rng.uniform(0.5, 1.0, (h, w)).astype(np.float32),
        "motion_px": rng.uniform(-1.0, 1.0, (h, w, 2)).astype(np.float32),
        "semantic_id": rng.integers(0, 9, (h, w)).astype(np.uint32),
        "sky_mask": rng.integers(0, 2, (h, w)).astype(bool),
        "shadow_mask": rng.uniform(0.0, 1.0, (h, w)).astype(np.float32),
        "sun_cos_incidence": rng.uniform(0.0, 1.0, (h, w)).astype(np.float32),
        "elevation_rad": rng.uniform(-0.2, 0.8, (h, w)).astype(np.float32),
        "background_t_k": rng.uniform(230.0, 300.0, (h, w)).astype(np.float32),
    }
    return planes


def test_every_optional_key_is_a_field_and_every_field_is_a_key() -> None:
    fields = {f.name for f in dataclasses.fields(GBuffer)} - {"extra"}
    assert fields == REQUIRED_KEYS | OPTIONAL_KEYS


def test_from_dict_of_to_dict_round_trips_every_optional_plane_bit_exactly() -> None:
    planes = _planes()
    g = GBuffer.from_dict(planes)
    assert g.background_t_k is not None and g.elevation_rad is not None
    again = g.to_dict()
    assert set(again) == set(planes), "the two planes that used to be dropped are back"
    for key, value in planes.items():
        assert np.array_equal(again[key], value), key
        assert again[key].dtype == GBuffer.from_dict(again).to_dict()[key].dtype
    twice = GBuffer.from_dict(again)
    for key in planes:
        assert np.array_equal(getattr(twice, key), getattr(g, key)), key


def test_the_required_planes_alone_still_round_trip_and_extras_ride_along() -> None:
    planes = {k: v for k, v in _planes().items() if k in REQUIRED_KEYS}
    planes["radiance_behind"] = np.ones((3, 4), dtype=np.float32)
    g = GBuffer.from_dict(planes)
    out = g.to_dict()
    assert set(out) == set(planes)
    assert np.array_equal(out["radiance_behind"], planes["radiance_behind"])
    with pytest.raises(KeyError, match="unexpected planes"):
        GBuffer.from_dict({**planes, "not_a_plane": np.zeros((3, 4), np.float32)})
