"""The cloud deck written as a volume the renderer can load (AT.13, ADR 0140).

These are the checks that do not need a renderer. What a written grid *looks like* to RTX is not
knowable here and is `scripts/probe_cloud_volume.py`'s job; what is knowable here is that the
file is a valid OpenVDB, that it carries the field the infrared band marches, and that a voxel
lands where the deck says it does -- a volume in the wrong place renders perfectly and is wrong.
"""

from __future__ import annotations

import numpy as np
import pytest

from irsim.atmosphere.cloud_deck import generate_cloud_deck
from irsim_isaac import env
from irsim_isaac.cloud_volume import DENSITY_GRID_NAME, write_openvdb

openvdb_only = pytest.mark.skipif(
    not env.has_openvdb(), reason="needs the OpenVDB that ships inside Isaac Sim's omni.volume"
)


def deck():
    return generate_cloud_deck(
        beta=2.8, cloud_fraction=0.35, seed=7, base_m=1200.0, optical_depth=20.0
    )


@openvdb_only
def test_the_written_file_is_an_openvdb_a_stranger_can_read(tmp_path):
    """Not "Warp reads back what Warp wrote". The format's own library, in a fresh read, with the
    name a `UsdVol.Volume` field relationship binds by."""
    import openvdb

    volume = write_openvdb(deck(), tmp_path / "cloud.vdb", voxel_m=100.0)
    assert volume.path.read_bytes()[:4] == b" BDV"  # OpenVDB's magic, little-endian "VDB "

    grid = openvdb.read(str(volume.path), DENSITY_GRID_NAME)
    assert grid.name == DENSITY_GRID_NAME
    assert grid.activeVoxelCount() == volume.active_voxels > 0
    low, high = grid.evalMinMax()
    assert high == pytest.approx(volume.max_density_per_m, rel=1e-5)
    assert low >= 0.0


@openvdb_only
def test_the_volume_is_sparse_which_is_the_whole_reason_for_the_format(tmp_path):
    """A cumulus deck is mostly clear air. If the empty space were stored the grid would be seven
    times the size and a finer voxel would stop being affordable -- which is the thing that makes
    a good-looking cloud reachable at all."""
    volume = write_openvdb(deck(), tmp_path / "cloud.vdb", voxel_m=100.0)
    assert 0.02 < volume.occupancy < 0.5
    dense_bytes = np.prod(volume.shape) * 4
    assert volume.file_bytes < 0.6 * dense_bytes


@openvdb_only
def test_a_voxel_lands_where_the_deck_puts_it(tmp_path):
    """**The failure this exists for is silent.** A volume with the wrong transform renders
    beautifully, in the wrong place -- underground, or a kilometre to one side -- and nothing in
    the frame says which. So the grid's own index-to-world map is checked against the deck's
    geometry rather than against the array that was handed to it."""
    import openvdb

    d = deck()
    voxel_m = 100.0
    volume = write_openvdb(d, tmp_path / "cloud.vdb", voxel_m=voxel_m)
    grid = openvdb.read(str(volume.path), DENSITY_GRID_NAME)

    origin = grid.transform.indexToWorld((0.0, 0.0, 0.0))
    assert origin == pytest.approx(volume.min_world_m, abs=1e-6)
    # The cloud's base is the lifting condensation level, and nothing may sit below it.
    assert origin[1] == pytest.approx(d.base_m, abs=1e-6)

    far = grid.transform.indexToWorld(
        (volume.shape[0] - 1.0, volume.shape[1] - 1.0, volume.shape[2] - 1.0)
    )
    assert far[1] == pytest.approx(d.top_m, abs=voxel_m)
    assert far[0] == pytest.approx(d.half_extent_m, abs=voxel_m)


@openvdb_only
def test_the_grid_carries_the_field_the_infrared_band_marches(tmp_path):
    """One deck, two consumers (ADR 0127). The volume the path tracer renders has to be a
    *sampling* of `density_at`, not a second cloud that resembles it -- if these two ever disagree
    the bands disagree about where the cloud is, which is the defect the whole design exists to
    make impossible."""
    import openvdb

    d = deck()
    voxel_m = 100.0
    volume = write_openvdb(d, tmp_path / "cloud.vdb", voxel_m=voxel_m)
    grid = openvdb.read(str(volume.path), DENSITY_GRID_NAME)
    accessor = grid.getConstAccessor()

    rng = np.random.default_rng(0)
    checked = 0
    for _ in range(400):
        i = (
            int(rng.integers(0, volume.shape[0])),
            int(rng.integers(0, volume.shape[1])),
            int(rng.integers(0, volume.shape[2])),
        )
        x, y, z = grid.transform.indexToWorld(tuple(float(v) for v in i))
        expected = float(d.density_at(np.array(x), np.array(y), np.array(z)))
        stored = float(accessor.getValue(i))
        if expected <= 1e-4 and stored == 0.0:
            continue  # background, dropped by the write tolerance
        assert stored == pytest.approx(expected, abs=2e-4, rel=1e-3)
        checked += 1
    assert checked > 20, "the sample never landed in cloud; the test proved nothing"


@openvdb_only
def test_a_clear_sky_writes_an_empty_grid_rather_than_failing(tmp_path):
    d = generate_cloud_deck(
        beta=2.8, cloud_fraction=0.0, seed=7, base_m=1200.0, optical_depth=20.0
    )
    volume = write_openvdb(d, tmp_path / "clear.vdb", voxel_m=200.0)
    assert volume.active_voxels == 0
    assert volume.max_density_per_m == 0.0
