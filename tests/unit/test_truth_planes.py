"""Scene-truth planes (irsim.io.truth): the right surface under each pixel, named correctly.

What would go wrong unnoticed: a supersampled plane folded at the wrong offset (every truth
reading one sample off, which on an edge is the wrong part), a legend whose codes do not match
the plane, or the sky reported as a surface at 0 m. Each test pins one of those.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest

from irsim.io.dataset import PLANE_UNITS, write_frame
from irsim.io.truth import SKY_LABEL, centre_sample, truth_planes

K = 3  # supersample factor
H, W = 4, 5  # detector grid


def _gbuffer() -> dict[str, np.ndarray]:
    """A (H*K, W*K) G-buffer whose every sample encodes its own (row, col) so offsets show."""
    rows, cols = np.mgrid[0 : H * K, 0 : W * K]
    sky = np.zeros((H * K, W * K), bool)
    sky[:, : 2 * K] = True  # detector columns 0..1 are sky
    return {
        "temperature_k": (250.0 + rows + cols / 100.0).astype(np.float32),
        "distance_m": np.where(sky, 0.0, 10.0 + cols).astype(np.float32),
        "material_id": np.where(sky, 0, 1 + (cols >= 4 * K)).astype(np.int32),
        "sky_mask": sky,
    }


def test_the_centre_sample_is_taken_not_a_corner() -> None:
    plane = _gbuffer()["temperature_k"]
    out = centre_sample(plane, (H, W))
    # detector pixel (1, 2) covers samples rows 3..5, cols 6..8; its centre is (4, 7)
    assert out[1, 2] == pytest.approx(250.0 + 4 + 7 / 100.0)
    with pytest.raises(ValueError, match="integer supersample"):
        centre_sample(plane[:-1], (H, W))


def test_sky_has_no_distance_and_no_part() -> None:
    iid = np.where(_gbuffer()["sky_mask"], 0, 7).astype(np.uint32)
    t = truth_planes(
        _gbuffer(),
        (H, W),
        instance_id=iid,
        labels={7: "/World/drone/motor_front_left"},
        prim_to_node={"/World/drone/motor_front_left": "motors"},
    )
    assert np.isnan(t.planes["distance_m"][:, :2]).all()
    assert np.isfinite(t.planes["distance_m"][:, 2:]).all()
    assert (t.planes["part_id"][:, :2] == 0).all() and t.legends["part_id"][0] == SKY_LABEL
    assert t.planes["temperature_k"].dtype == np.float32


def test_legends_name_the_codes_the_planes_actually_hold() -> None:
    iid = np.zeros((H * K, W * K), np.uint32)
    iid[:, 2 * K : 4 * K] = 11
    iid[:, 4 * K :] = 42
    labels = {11: "/World/drone/arm_left", 42: "/World/drone/motor_front_left", 99: "/World/x"}
    t = truth_planes(
        _gbuffer(),
        (H, W),
        instance_id=iid,
        labels=labels,
        prim_to_node={labels[11]: "airframe", labels[42]: "motor_front_left"},
        material_names=("UNMAPPED", "paint", "aluminium"),
    )
    part, node = t.planes["part_id"], t.planes["node_id"]
    assert t.legends["part_id"][int(part[0, 2])] == "arm_left"
    assert t.legends["part_id"][int(part[0, 4])] == "motor_front_left"
    assert t.legends["node_id"][int(node[0, 4])] == "motor_front_left"
    assert t.legends["node_id"][int(node[0, 3])] == "airframe"
    assert t.legends["material_id"][int(t.planes["material_id"][0, 4])] == "aluminium"


def test_a_plane_without_its_inputs_is_not_written() -> None:
    t = truth_planes({"temperature_k": _gbuffer()["temperature_k"]}, (H, W))
    assert set(t.planes) == {"temperature_k"} and t.legends == {}


def test_the_sidecar_carries_units_and_legends(tmp_path: pathlib.Path) -> None:
    iid = np.full((H * K, W * K), 5, np.uint32)
    t = truth_planes(_gbuffer(), (H, W), instance_id=iid, labels={5: "/World/a/battery"})

    class Out:
        radiance = np.ones((H, W), np.float32)

    record = write_frame(tmp_path, Out(), extra_planes=t.planes, legends=t.legends)
    doc = json.loads(record.sidecar.read_text())
    for key in ("temperature_k", "distance_m", "part_id"):
        assert doc["planes"][key]["unit"] == PLANE_UNITS[key]
    assert doc["legends"]["part_id"]["1"] == "battery"
    back = np.load(tmp_path / doc["planes"]["temperature_k"]["file"])
    assert back.dtype == np.float32 and np.array_equal(back, t.planes["temperature_k"])


def test_the_cloud_planes_reach_the_truth_and_name_a_cloud_pixel() -> None:
    """AT.29: the march's transmittance to each hit is written beside the truth, the emission
    range is NaN where there was no cloud, and `cloud_id` names a pixel *cloud* once less than
    half of what lies behind the cloud gets through -- so a label reader stops calling it
    unknown. Absent from the G-buffer, none of the three is written."""
    from irsim.io.truth import CLOUD_LEGEND, CLOUD_THRESHOLD

    g = _gbuffer()
    assert "cloud_id" not in truth_planes(g, (H, W)).planes
    rows, cols = np.mgrid[0 : H * K, 0 : W * K]
    tau = np.clip(cols / (W * K - 1.0), 0.0, 1.0).astype(np.float32)  # clear on the right
    g["cloud_transmittance"] = tau
    g["cloud_range_m"] = np.full(tau.shape, 250.0, np.float32)
    t = truth_planes(g, (H, W))
    assert t.planes["cloud_transmittance"].dtype == np.float32
    assert np.array_equal(t.planes["cloud_transmittance"], centre_sample(tau, (H, W)))
    cloud = t.planes["cloud_id"]
    assert cloud.dtype == np.uint8 and t.legends["cloud_id"] == CLOUD_LEGEND
    assert np.array_equal(cloud, (centre_sample(tau, (H, W)) < CLOUD_THRESHOLD).astype(np.uint8))
    assert cloud[:, 0].all() and not cloud[:, -1].any()
    rng = t.planes["cloud_range_m"]
    assert np.isnan(rng[centre_sample(tau, (H, W)) >= 1.0]).all()
    assert (rng[centre_sample(tau, (H, W)) < 1.0] == 250.0).all()
    for key in ("cloud_transmittance", "cloud_range_m", "cloud_id"):
        assert key in PLANE_UNITS
