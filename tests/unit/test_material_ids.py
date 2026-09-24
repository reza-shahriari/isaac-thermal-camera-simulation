"""Engine-free tests for instance-id → material-id transport (roadmap M10.2, ADR 0047).

The property that matters is that ids are *decoded*, never *interpolated*. A pixel on the boundary
between material 0 and material 7 is one or the other; a blended 3 or 4 is a different substance
with a different emissivity, and the resulting image looks entirely plausible. These tests assert
that no value outside the authored set can ever appear, that a miss keeps the loud UNMAPPED
sentinel instead of a convenient default, and that the magenta overlay touches display pixels only.
"""

from __future__ import annotations

import numpy as np
import pytest

from irsim.materials.mapping import Resolution
from irsim.materials.table import UNMAPPED_MATERIAL_ID
from irsim_isaac.pipeline.material_ids import (
    MAGENTA_RGBA,
    fold_mask_to_native,
    id_coverage,
    labels_to_paths,
    mark_unmapped_radiometry,
    material_id_plane,
    overlay_unmapped,
    unmapped_mask,
)

# Three ids spanning the range the packed table can hold, including the sentinel's neighbour.
IDS = {1: ("/World/A", 7), 2: ("/World/B", 200), 3: ("/World/C", 1)}


def _resolutions() -> list[Resolution]:
    return [Resolution(path, f"mat_{mid}", mid, "pattern", "*x*") for path, mid in IDS.values()]


def _labels() -> dict[str, str]:
    return {str(k): v[0] for k, v in IDS.items()}


def test_ids_decode_exactly_on_every_pixel() -> None:
    """Every pixel gets the authored id; nothing in between ever appears."""
    rng = np.random.default_rng(0)
    instance = rng.choice(np.array([0, 1, 2, 3], dtype=np.uint32), size=(64, 64))
    plane = material_id_plane(instance, _labels(), _resolutions())

    assert plane.dtype == np.int32
    assert set(np.unique(plane).tolist()) <= {0, 1, 7, 200}
    for ident, (_, material) in IDS.items():
        where = instance == ident
        assert np.all(plane[where] == material), ident
    assert np.all(plane[instance == 0] == UNMAPPED_MATERIAL_ID)


def test_an_edge_between_two_materials_never_blends() -> None:
    """The 0/7 boundary must not produce 3 or 4 -- the failure this transport exists to avoid."""
    instance = np.zeros((16, 16), dtype=np.uint32)
    instance[:, 8:] = 1  # left half background, right half material 7
    plane = material_id_plane(instance, _labels(), _resolutions())

    values = set(np.unique(plane).tolist())
    assert values == {0, 7}
    assert not values & {3, 4}
    assert np.all(plane[:, :8] == 0) and np.all(plane[:, 8:] == 7)


def test_supersampled_ids_downsample_by_nearest_not_by_mean() -> None:
    """A 4x supersampled id block decimates to one of its members, never to their average."""
    block = np.zeros((8, 8), dtype=np.uint32)
    block[:4] = 1  # material 7
    block[4:] = 2  # material 200
    plane = material_id_plane(block, _labels(), _resolutions())
    decimated = plane[::4, ::4]
    assert set(np.unique(decimated).tolist()) <= {7, 200}
    assert abs(float(plane.mean()) - 103.5) < 1e-6  # the mean IS 103.5 -- and is never an id
    assert 103 not in np.unique(plane).tolist()


def test_an_unresolved_prim_keeps_the_unmapped_sentinel() -> None:
    """A prim the resolver missed must not acquire a plausible default emissivity."""
    resolutions = [
        Resolution("/World/A", "mat_7", 7, "pattern", "*x*"),
        Resolution("/World/B", None, UNMAPPED_MATERIAL_ID, "miss", None),
    ]
    instance = np.array([[1, 2], [1, 2]], dtype=np.uint32)
    plane = material_id_plane(instance, {"1": "/World/A", "2": "/World/B"}, resolutions)
    assert np.all(plane[:, 0] == 7)
    assert np.all(plane[:, 1] == UNMAPPED_MATERIAL_ID)


def test_an_id_with_no_label_is_unmapped_and_strict_mode_raises() -> None:
    instance = np.array([[1, 9]], dtype=np.uint32)
    plane = material_id_plane(instance, _labels(), _resolutions())
    assert plane[0, 1] == UNMAPPED_MATERIAL_ID
    with pytest.raises(KeyError, match="idToLabels"):
        material_id_plane(instance, _labels(), _resolutions(), strict=True)


def test_background_is_unmapped_but_is_not_a_mapping_miss() -> None:
    """Sky pixels carry id 0 and must not be counted as a forgotten prim (M0.6 contract)."""
    instance = np.array([[0, 0], [1, 1]], dtype=np.uint32)
    sky = instance == 0
    plane = material_id_plane(instance, _labels(), _resolutions())
    mask = unmapped_mask(plane, sky_mask=sky)
    assert not mask.any()
    assert id_coverage(plane, sky_mask=sky) == pytest.approx(1.0)
    # without the sky mask the same frame looks half unmapped, which is why the mask exists
    assert id_coverage(plane) == pytest.approx(0.5)


def test_labels_to_paths_handles_both_payload_shapes() -> None:
    paths = labels_to_paths({"1": "/World/A", "2": {"class": "car_body"}, "3": None, "x": "/W"})
    assert paths == {1: "/World/A", 2: "car_body"}


# --- the magenta overlay -----------------------------------------------------------------------


def test_magenta_paints_the_mask_and_nothing_else() -> None:
    rgba = np.full((4, 4, 4), 30, dtype=np.uint8)
    mask = np.zeros((4, 4), dtype=bool)
    mask[1, 2] = True
    out = overlay_unmapped(rgba, mask)

    assert tuple(out[1, 2]) == MAGENTA_RGBA
    untouched = np.ones((4, 4), dtype=bool)
    untouched[1, 2] = False
    assert np.all(out[untouched] == 30)
    assert np.all(rgba == 30), "the input image must not be modified in place"


def test_overlay_refuses_a_non_rgba8_image() -> None:
    mask = np.zeros((2, 2), dtype=bool)
    with pytest.raises(TypeError, match="uint8"):
        overlay_unmapped(np.zeros((2, 2, 4), dtype=np.float32), mask)
    with pytest.raises(ValueError, match=r"\(H, W, 4\)"):
        overlay_unmapped(np.zeros((2, 2, 3), dtype=np.uint8), mask)
    with pytest.raises(ValueError, match="mask"):
        overlay_unmapped(np.zeros((2, 2, 4), dtype=np.uint8), np.zeros((3, 3), dtype=bool))


def test_material_id_plane_refuses_a_float_id_plane() -> None:
    with pytest.raises(TypeError, match="integer"):
        material_id_plane(np.zeros((2, 2), dtype=np.float32), _labels(), _resolutions())


# --- IG.17: the radiometric half of the marking -------------------------------------------------
#
# Until this, an unmapped prim was marked in the *picture* and not in the numbers. `SE.2` measured
# an undeclared sea at 200.1 K -- the band LUT's floor, where an apparent temperature lands when
# the radiance under it is nearly zero -- over 72 % of a frame, with a mask beside it that looked
# entirely correct. These are the engine-free half of the repair; the rendered half is
# `tests/integration/test_maritime_isaac.py`.

LUT_FLOOR_K = 200.0


def test_an_unmapped_pixel_carries_nan_rather_than_a_number_in_range() -> None:
    """The failure being repaired is a *plausible* value, not an obviously broken one.

    200.1 K is in the LUT's domain, is in kelvin, and reads as very cold water. Anything that
    averages a frame -- an AGC, a report, a dataset label -- consumes it without complaint.
    """
    plane = np.full((2, 3), LUT_FLOOR_K + 0.1, dtype=np.float32)
    mask = np.zeros((2, 3), dtype=bool)
    mask[1, 2] = True
    out = mark_unmapped_radiometry(plane, mask)
    assert np.isnan(out[1, 2])
    assert np.count_nonzero(np.isnan(out)) == 1
    assert out[0, 0] == np.float32(LUT_FLOOR_K + 0.1)
    assert not np.isnan(plane).any(), "the input plane must not be marked in place"


def test_the_marking_is_the_one_value_a_mean_cannot_swallow() -> None:
    """A sentinel kelvin would be averaged into a believable frame temperature; NaN is not.

    The negative control is the whole argument for NaN: with the floor left in place the frame's
    mean is a number somebody could quote.
    """
    plane = np.full((10, 10), 290.0, dtype=np.float32)
    mask = np.zeros((10, 10), dtype=bool)
    mask[:, :7] = True  # 70 % of the frame, as SE.2 measured
    sentinel = plane.copy()
    sentinel[mask] = np.float32(LUT_FLOOR_K)
    assert float(sentinel.mean()) == pytest.approx(227.0)  # believable, and wrong
    assert np.isnan(mark_unmapped_radiometry(plane, mask).mean())


def test_one_bad_sample_marks_the_whole_native_pixel() -> None:
    """Ids are never interpolated, so a 2x2 block may cover two prims.

    Over-reporting is the only safe direction: a pixel that is three quarters physics and one
    quarter nothing is not a measurement of anything, and hiding it behind the three good samples
    is how a forgotten prim survives review.
    """
    mask = np.zeros((4, 4), dtype=bool)
    mask[0, 0] = True  # one sample of the top-left 2x2 block
    folded = fold_mask_to_native(mask, 2)
    assert folded.shape == (2, 2)
    assert folded[0, 0] and folded.sum() == 1


def test_an_unsupersampled_mask_passes_through_unchanged() -> None:
    mask = np.array([[True, False], [False, True]])
    assert np.array_equal(fold_mask_to_native(mask, 1), mask)


def test_a_mask_that_does_not_fit_its_plane_is_refused() -> None:
    """Silently marking the wrong pixels would be worse than the bug being fixed."""
    with pytest.raises(ValueError, match="does not fit"):
        mark_unmapped_radiometry(np.zeros((2, 2), dtype=np.float32), np.zeros((3, 3), dtype=bool))
