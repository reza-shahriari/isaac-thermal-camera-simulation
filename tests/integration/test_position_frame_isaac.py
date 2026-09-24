"""Roadmap IG.2: the position decode and the point bridge, against real rendered geometry.

The arithmetic is covered engine-free in ``tests/unit/test_point_bridge.py``, whose oracle is a
camera pose written out as three named world-space axes. What needs a renderer is the other end of
the same claim: that ``Camera3dPositionSD`` arrives from *this build* in the frame the pipeline
assumes, as float32, and that decoding it with ``world_positions`` puts each pixel back on the
surface the stage says it is on.

The oracle here owes nothing to the renderer and nothing to the decode. ``plate_vertical`` is
authored on the world plane ``z = cz - 6`` and ``plate_up`` on ``y = cy - 1.9``; those numbers come
from ``build_geometry_scene``'s USD calls. A decode that transposed the camera rotation still
produces a plane -- tilted and displaced -- so *flatness at the authored depth* is the
discriminator, and it is the same one that would have caught ADR 0014's M10.19 horizon, 164 rows
out. The existing frame test in ``test_gbuffer_isaac.py`` scores the three hypotheses against
``C + d * ray``; this one asks the stage instead, so the two do not share an assumption.

The camera is **pitched and off the origin**. With either alone the three candidate frames are
degenerate in one axis, which is how ADR 0014 and its addendum came to disagree.

docs/physics-model.md §13.1, §13.3; ADR 0014 (and its M10.19 addendum), ADR 0087.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

RESOLUTION = 256
PITCH_DEG = 20.0

#: The AOV is documented at 3.4 mm; a plate read 1 cm off its authored plane is a real finding.
PLANE_TOL_M = 0.01
#: A transposed rotation displaces a surface by order the camera's own offset, not by rounding.
FLIP_MIN_M = 0.10


@pytest.fixture(scope="module")
def scene(simulation_app: Any) -> Any:
    del simulation_app
    from irsim_isaac.geometry_probe import build_geometry_scene, configure_renderer, pitch_camera

    configure_renderer()
    built = build_geometry_scene(resolution=RESOLUTION)
    pitch_camera(built, -PITCH_DEG)
    return built


@pytest.fixture(scope="module")
def frame(scene: Any) -> Any:
    """One settled render: the position and instance planes, and the camera's pose off the stage."""
    import omni.replicator.core as rep

    from irsim_isaac.pipeline.gbuffer_isaac import AovReader, camera_pose
    from irsim_isaac.pipeline.material_ids import labels_from_payload

    rp = rep.create.render_product(scene.camera_path, (RESOLUTION, RESOLUTION))
    rp_path = rp.path if hasattr(rp, "path") else str(rp)
    reader = AovReader(
        rp_path,
        device="cpu",
        required=("instance", "position"),
        expected_shape=(RESOLUTION, RESOLUTION),
    ).attach(settle_frames=24)
    aovs = reader.read()
    cam_position, cam_to_world = camera_pose(scene.camera_path)
    labels = {str(k): v for k, v in labels_from_payload(aovs.device_handles["instance"]).items()}
    yield {
        "aovs": aovs,
        "labels": labels,
        "camera_position": cam_position,
        "camera_to_world": cam_to_world,
    }
    reader.detach()


def _mask_of(frame: Any, prim_path: str) -> tuple[np.ndarray, np.ndarray]:
    """``(covered, interior)``: the renderer's own pixels for ``prim_path``, and those less the rim.

    Edge pixels of a quad straddle the geometry and the background, and a straddled position is
    neither surface. One pixel of erosion is enough and is applied to the mask, not to the
    tolerance, so the tolerance stays a physical claim.

    **Both are returned because they answer different questions, and conflating them is a bug this
    file had until it was first run (IG.2).** A claim about the *values* on the plate must use the
    interior, or a straddled position pollutes it. A claim about what lies *off* the plate must use
    the renderer's own coverage: the rim pixels carry the plate's instance id, so the bridge is
    right to write the plate's temperature there, and testing `out[~interior]` called that
    correct behaviour a leak. The un-eroded complement still catches a real leak -- a pixel the
    renderer says is not the plate holding the plate's field -- which is the claim intended.
    """
    from irsim_isaac.pipeline.material_ids import labels_to_paths

    paths = labels_to_paths(frame["labels"])
    wanted = {int(i) for i, path in paths.items() if path == prim_path}
    assert wanted, f"{prim_path} is not on screen: {sorted(set(paths.values()))}"
    ids = np.asarray(frame["aovs"].instance_id)
    mask = np.isin(ids, list(wanted))
    inner = mask.copy()
    inner[:-1, :] &= mask[1:, :]
    inner[1:, :] &= mask[:-1, :]
    inner[:, :-1] &= mask[:, 1:]
    inner[:, 1:] &= mask[:, :-1]
    assert inner.sum() > 200, f"{prim_path} covers too few interior pixels: {int(inner.sum())}"
    return mask, inner


def test_the_position_plane_arrives_as_float32(frame: Any) -> None:
    """CLAUDE.md non-negotiable #2 at the one boundary that carries metres, not kelvin.

    A float16 position at 10 m spaces its samples ~5 mm apart, which is the width of the cells a
    point-wise field is gathered from -- so the plane would still look like geometry while every
    lookup quantised onto the wrong cell.
    """
    position = np.asarray(frame["aovs"].position)
    assert position.dtype == np.float32, position.dtype
    assert np.isfinite(position).any()


@pytest.mark.parametrize(
    ("target", "axis", "offset"),
    [("plate_vertical", 2, -6.0), ("plate_up", 1, -1.9)],
)
def test_a_rendered_plate_decodes_onto_the_plane_it_was_authored_on(
    scene: Any, frame: Any, target: str, axis: int, offset: float
) -> None:
    """The independent oracle: the stage says where the plate is, and the decode must agree."""
    from irsim_isaac.pipeline.point_bridge import world_positions

    rot = frame["camera_to_world"]
    expected = float(scene.camera_position[axis]) + offset
    # The interior: this is a claim about decoded *positions*, which a straddled edge pixel
    # would pollute. Unchanged by IG.2's fix -- only the off-prim claim needed the other one.
    _, mask = _mask_of(frame, scene.targets[target].prim_path)

    world = world_positions(
        frame["aovs"].position,
        frame="camera",
        camera_position=frame["camera_position"],
        camera_to_world=rot,
    )
    residual = np.abs(world[..., axis][mask] - expected)
    assert float(residual.max()) < PLANE_TOL_M, (target, float(residual.max()), expected)

    flipped = world_positions(
        frame["aovs"].position,
        frame="camera",
        camera_position=frame["camera_position"],
        camera_to_world=rot.T,
    )
    off = np.abs(flipped[..., axis][mask] - expected)
    assert float(off.max()) > FLIP_MIN_M, (target, float(off.max()))


def test_the_point_bridge_gathers_a_field_across_one_rendered_prim(scene: Any, frame: Any) -> None:
    """The point-wise claim itself, in-sim: one prim, many temperatures, in the authored order.

    A patch is authored over ``plate_vertical`` with a ramp along world +X. If the decode were in
    the wrong frame the pixels would fall outside the patch and the bridge would raise; if the
    gather were still per-instance the plate would come back flat. Neither shows up in a rendered
    image, which is why this is the check the car and wall demos wait on.
    """
    from irsim.thermal.facets import FacetForcing, FacetProperties
    from irsim.thermal.surface_field import PlanarPatch, PlanarThermalField
    from irsim_isaac.pipeline.point_bridge import (
        PointwiseTemperature,
        SurfaceBinding,
        world_positions,
    )

    plate = scene.targets["plate_vertical"]
    half = 1.05  # the authored half-size plus a margin: a silhouette pixel is not a seam
    n = 16
    patch = PlanarPatch(
        origin_m=np.array(plate.centre) - np.array([half, half, 0.0]),
        u_axis=np.array([1.0, 0.0, 0.0]),
        v_axis=np.array([0.0, 1.0, 0.0]),
        n_u=n,
        n_v=n,
        du_m=2.0 * half / n,
        dv_m=2.0 * half / n,
        thickness_m=0.05,
    )
    # A ramp along +u, frozen: a huge heat capacity makes the initial state the answer.
    ramp = np.tile(np.linspace(290.0, 320.0, n), n)
    field = PlanarThermalField(
        patch,
        FacetProperties(
            heat_capacity_j_m2_k=np.full(patch.n_cells, 1e9),
            emissivity=np.full(patch.n_cells, 0.95),
            solar_absorptivity=np.full(patch.n_cells, 0.9),
        ),
        lambda _t: FacetForcing(t_air_k=290.0, h_w_m2_k=0.0),
        0.0,
        ramp,
    )
    field.advance_to(1.0)

    world = world_positions(
        frame["aovs"].position,
        frame="camera",
        camera_position=frame["camera_position"],
        camera_to_world=frame["camera_to_world"],
    )
    flat = np.full((RESOLUTION, RESOLUTION), 300.0, dtype=np.float32)
    ids = np.asarray(frame["aovs"].instance_id)
    bridge = PointwiseTemperature([SurfaceBinding(plate.prim_path, field)])
    out = bridge.apply(flat, ids, frame["labels"], world, 1.0, strict=False)

    covered, interior = _mask_of(frame, plate.prim_path)
    on_plate = out[interior].astype(np.float64)
    assert float(on_plate.max() - on_plate.min()) > 20.0, "the plate came back flat"
    # Off the prim as the *renderer* draws it, not as the erosion trims it: the rim carries the
    # plate's instance id, so the plate's temperature belongs there.
    assert np.all(out[~covered] == np.float32(300.0)), "the gather leaked off the bound prim"
    # And the rim is not silently background either -- if erosion were hiding a real failure, the
    # pixels it removes would have kept the untouched 300 K.
    rim = covered & ~interior
    assert rim.any(), "the silhouette has no rim, so the erosion proves nothing"
    assert np.any(out[rim] != np.float32(300.0)), "the rim was not written; the gather stops short"

    # The ramp runs along world +X, so the temperature must rise with the decoded x, not with the
    # pixel column: the camera is pitched, and a frame error would still give a smooth gradient.
    x = world[..., 0][interior]
    lo, hi = np.percentile(x, 20.0), np.percentile(x, 80.0)
    assert float(on_plate[x > hi].mean() - on_plate[x < lo].mean()) > 15.0
