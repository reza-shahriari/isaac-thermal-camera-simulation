"""The point-wise vessel, engine-free (PT.10; ADR 0087, ADR 0095, ADR 0123).

The maritime lane carried three temperatures for three prims -- hull, deckhouse, funnel -- and a
deck that was one number. This scene makes the deck and the deckhouse fields, and what has to be
checked is not that a field exists but that it is **the right field in the right place**:

  * every patch cell and every occluder face in ``vessel_pointwise_clear_day.yaml`` lies on the
    prim :mod:`irsim_isaac.vessel_pointwise` authors for it. A patch that drifts off its prim does
    not fail -- ``PointwiseTemperature`` leaves those pixels at the prim's per-prim fallback and
    the frame looks entirely normal, with part of a deck at one temperature. That is the failure
    ADR 0087 exists to remove, so it is asserted cell by cell.
  * the shadow is where the **sun** puts it, not merely somewhere. A field with a plausible-looking
    gradient in the wrong place is the most expensive kind of wrong here, because nothing about the
    picture says so.
  * the change is provably a **redistribution**, not an addition: take the occluders away and the
    point-wise deck has to collapse onto the single number it replaced.

The hour is not chosen for the sun alone. This coast's sea breeze climbs from 3 m/s at dawn to
9 m/s by mid-afternoon, and h(U) decides how far a sunlit plate gets from the air, so the largest
sun is not the largest signature. Swept hour by hour the deck's own span peaks at 7.7 K at local
noon while the deckhouse's sunlit-to-shaded step peaks at 5.3 K in mid-morning; the scene sits at
09:00Z, where both are above 5 K at once.

docs/physics-model.md §6; roadmap PT.10.
"""

from __future__ import annotations

import math
import pathlib

import numpy as np
import pytest

from irsim.config.scene import load_scene_config
from irsim.scene import Scene
from irsim_isaac.maritime_demo import vessel_boxes
from irsim_isaac.vessel_pointwise import (
    POINTWISE_VESSEL,
    VESSEL_LENGTH_M,
    VESSEL_ROOT,
    deck_plane_y_m,
)

REPO = pathlib.Path(__file__).resolve().parents[2]
SCENE_PATH = REPO / "configs" / "scenes" / "vessel_pointwise_clear_day.yaml"

#: The sun at 09:00Z on 2024-06-21 from 36.8 N, 5.4 W, from `irsim.thermal.solar`.
SUN_ELEVATION_DEG = 44.6
SUN_AZIMUTH_DEG = 92.3

#: A cell centre may sit exactly on the face it describes; a millimetre is the slack for that.
ON_FACE_TOL_M = 1e-3


@pytest.fixture(scope="module")
def config() -> object:
    assert SCENE_PATH.exists(), f"{SCENE_PATH} is gone; this test names it deliberately"
    return load_scene_config(SCENE_PATH)


@pytest.fixture(scope="module")
def scene(config: object) -> Scene:
    return Scene.from_config(config)


def _boxes() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Each prim's axis-aligned bounds in the stage frame. No part here is rotated."""
    out = {}
    for part in POINTWISE_VESSEL:
        assert part.rotate_xyz_deg == (0.0, 0.0, 0.0), (
            f"{part.name} is rotated, so a bounding box is no longer its extent and the "
            "containment checks below would silently pass on the wrong volume"
        )
        centre = np.asarray(part.centre_m, dtype=np.float64)
        half = 0.5 * np.asarray(part.size_m, dtype=np.float64)
        out[f"{VESSEL_ROOT}/{part.name}"] = (centre - half, centre + half)
    return out


def _field(scene: Scene, name: str) -> np.ndarray:
    fld = scene.surface_fields[name]
    fld.advance_to(scene.t0_s)
    return np.asarray(fld.temperature_at(scene.t0_s), dtype=np.float64)


# --- the two files agree about the boat -----------------------------------------------------


def test_the_pointwise_vessel_and_the_demo_stage_describe_one_boat() -> None:
    """The boxes come from `maritime_demo`, not from numbers retyped beside it.

    If this module ever grew its own copy of the layout, the demo stage and the point-wise stage
    would be two different vessels with one name, and a patch authored against either would sit
    off the other's prim.
    """
    shared = vessel_boxes(VESSEL_LENGTH_M)
    parts = {p.name: p for p in POINTWISE_VESSEL}
    for name in ("hull", "superstructure", "stack"):
        assert parts[name].centre_m == shared[name][0], name
        assert parts[name].size_m == shared[name][1], name
    # The deck is the one part the demo stage has no use for, and it must be flush with the
    # deckhouse's base -- a deck below it would let the sun under the house onto the deck.
    deck = parts["weather_deck"]
    top = deck.centre_m[1] + 0.5 * deck.size_m[1]
    assert abs(top - deck_plane_y_m()) < 1e-12
    house_bottom = shared["superstructure"][0][1] - 0.5 * shared["superstructure"][1][1]
    assert abs(top - house_bottom) < 1e-12, (top, house_bottom)


def test_every_patch_binds_to_a_prim_that_exists(scene: Scene) -> None:
    boxes = _boxes()
    assert scene.patch_prims, "the scene declares no patch prims at all"
    for name, path in scene.patch_prims.items():
        assert path in boxes, f"{name} binds to {path}, which no part of the vessel authors"


def test_every_patch_cell_lies_inside_its_own_prim(scene: Scene) -> None:
    """Cell by cell, because one cell off the prim is one wrong patch of deck in every frame."""
    boxes = _boxes()
    for name, path in scene.patch_prims.items():
        lo, hi = boxes[path]
        centres = scene.surface_fields[name].patch.cell_centres()
        outside = ~np.all((centres >= lo - ON_FACE_TOL_M) & (centres <= hi + ON_FACE_TOL_M), axis=1)
        assert not outside.any(), (
            f"{name}: {int(outside.sum())} of {len(centres)} cells are off {path}; "
            f"first is {centres[outside][0]}, prim spans {lo} to {hi}"
        )


def test_every_patched_prims_visible_face_is_inside_its_patch(scene: Scene) -> None:
    """The other direction: a pixel at the edge of the prim must still find a cell.

    A patch smaller than the face it covers leaves a rim of pixels that raise under
    ``strict_patch_coverage`` -- and rendered without it, a rim at the per-prim fallback.
    """
    faces = {
        # (surface, the prim face it covers as (fixed axis, value))
        "weather_deck": (1, deck_plane_y_m()),
        "house_west": (0, -0.5 * POINTWISE_VESSEL[2].size_m[0]),
        "house_east": (0, 0.5 * POINTWISE_VESSEL[2].size_m[0]),
    }
    boxes = _boxes()
    for name, (axis, value) in faces.items():
        lo, hi = boxes[scene.patch_prims[name]]
        corners = []
        free = [i for i in range(3) if i != axis]
        for a in (lo[free[0]], hi[free[0]]):
            for b in (lo[free[1]], hi[free[1]]):
                point = np.empty(3)
                point[axis] = value
                point[free[0]], point[free[1]] = a, b
                corners.append(point)
        inside = scene.surface_fields[name].patch.contains(np.asarray(corners))
        assert inside.all(), (
            f"{name}: corners {np.asarray(corners)[~inside]} fall outside the patch"
        )


def test_a_displaced_patch_is_caught(scene: Scene) -> None:
    """The containment check above has teeth: move a patch a metre and it must go red."""
    boxes = _boxes()
    lo, hi = boxes[scene.patch_prims["weather_deck"]]
    centres = scene.surface_fields["weather_deck"].patch.cell_centres() + np.array([0.0, 1.0, 0.0])
    assert not np.all(
        (centres >= lo - ON_FACE_TOL_M) & (centres <= hi + ON_FACE_TOL_M), axis=1
    ).all()


def test_every_occluder_face_sits_on_the_deckhouse(scene: Scene) -> None:
    """An occluder adrift shades the wrong metres of deck and nothing says so."""
    lo, hi = _boxes()[f"{VESSEL_ROOT}/superstructure"]
    assert scene.spec.thermal.occluders, "the scene declares no occluders; the deck would be flat"
    for occ in scene.spec.thermal.occluders:
        centre = np.asarray(occ.centre_m, dtype=np.float64)
        u = np.asarray(occ.u_axis, dtype=np.float64) * occ.half_u_m
        v = np.asarray(occ.v_axis, dtype=np.float64) * occ.half_v_m
        corners = np.asarray([centre + su * u + sv * v for su in (-1, 1) for sv in (-1, 1)])
        ok = np.all((corners >= lo - ON_FACE_TOL_M) & (corners <= hi + ON_FACE_TOL_M), axis=1)
        assert ok.all(), f"{occ.name}: {corners[~ok]} is not on the deckhouse ({lo} to {hi})"


# --- the physics ----------------------------------------------------------------------------


def test_the_deckhouse_lays_a_step_across_the_weather_deck(scene: Scene) -> None:
    """PT.10's headline: one deck prim, two temperatures, and the step is the deckhouse's."""
    t = _field(scene, "weather_deck")
    assert t.max() - t.min() > 5.0, f"the deck spans only {t.max() - t.min():.2f} K"


def test_the_shadow_is_where_the_sun_puts_it(scene: Scene) -> None:
    """A gradient in the wrong place looks exactly like a gradient in the right one.

    The sun bears 92 degrees -- due east -- at 44.6 degrees of elevation, so the deckhouse throws
    its shadow **forward**, to the west, reaching h / tan(elevation) beyond its own west face.
    The cells inside that strip must be the cold ones and the cells well clear of it the warm
    ones, which no amount of solver noise can arrange by accident.
    """
    fld = scene.surface_fields["weather_deck"]
    t = _field(scene, "weather_deck").reshape(fld.patch.shape)
    centres = fld.patch.cell_centres().reshape(*fld.patch.shape, 3)
    x, z = centres[..., 0], centres[..., 2]

    house = {p.name: p for p in POINTWISE_VESSEL}["superstructure"]
    west_face = house.centre_m[0] - 0.5 * house.size_m[0]
    half_beam = 0.5 * house.size_m[2]
    height = house.centre_m[1] + 0.5 * house.size_m[1] - deck_plane_y_m()
    reach = height / math.tan(math.radians(SUN_ELEVATION_DEG))
    assert 90.0 < SUN_AZIMUTH_DEG < 180.0, "a sun east of north throws its shadow to the west"

    # The shadow is a rectangle in both axes: the deckhouse is 4.0 m across on a 5.7 m deck, so
    # the strakes outboard of it are in full sun at every station. A test that took the whole beam
    # would mix the two and see no step at all -- which is how it first failed.
    inboard = np.abs(z) < half_beam
    cast = inboard & (x < west_face) & (x > west_face - reach)
    clear = inboard & (x < west_face - reach - 1.0)
    outboard = (np.abs(z) > half_beam) & (x < west_face) & (x > west_face - reach)
    assert cast.sum() > 5 and clear.sum() > 5 and outboard.sum() > 3, (
        int(cast.sum()),
        int(clear.sum()),
        int(outboard.sum()),
    )
    assert float(t[clear].min() - t[cast].max()) > 4.0, (
        f"cast shadow {t[cast].mean():.2f} K, deck clear of it {t[clear].mean():.2f} K"
    )
    # Same stations, outboard of the deckhouse: in sun, and within a few tenths of the open deck.
    # This is what makes the step above a *shadow* rather than a gradient along the hull.
    assert float(t[outboard].min() - t[cast].max()) > 4.0, (
        f"outboard {t[outboard].mean():.2f} K, in shadow {t[cast].mean():.2f} K"
    )


def test_the_deck_under_the_house_is_warmer_than_the_deck_in_its_shadow(scene: Scene) -> None:
    """Both have no beam at all, and they are not the same temperature.

    A shaded cell is not simply a cell with the sun switched off. The deck beneath the deckhouse
    is roofed, so it loses very little to the cold sky; the deck in the cast shadow a few metres
    forward is wide open to it. The measured difference is 1.3 K, in the direction the sky-view
    factor says (ADR 0104) -- and it is the difference a model that treated shadow as a single
    "shaded" flag would get exactly backwards in sign for free.
    """
    fld = scene.surface_fields["weather_deck"]
    t = _field(scene, "weather_deck").reshape(fld.patch.shape)
    centres = fld.patch.cell_centres().reshape(*fld.patch.shape, 3)
    x, z = centres[..., 0], centres[..., 2]

    house = {p.name: p for p in POINTWISE_VESSEL}["superstructure"]
    west_face = house.centre_m[0] - 0.5 * house.size_m[0]
    inboard = np.abs(z) < 0.5 * house.size_m[2]
    height = house.centre_m[1] + 0.5 * house.size_m[1] - deck_plane_y_m()
    reach = height / math.tan(math.radians(SUN_ELEVATION_DEG))

    roofed = inboard & (x > west_face) & (x < -west_face)
    cast = inboard & (x < west_face) & (x > west_face - reach)
    assert roofed.sum() > 20 and cast.sum() > 5
    assert float(t[roofed].mean() - t[cast].mean()) > 0.5, (
        f"roofed {t[roofed].mean():.2f} K, in the open shadow {t[cast].mean():.2f} K"
    )


def test_one_deckhouse_has_a_sunlit_face_and_a_shaded_face(scene: Scene) -> None:
    """Two patches on the SAME prim, differing only in which way they point."""
    east, west = _field(scene, "house_east"), _field(scene, "house_west")
    assert scene.patch_prims["house_east"] == scene.patch_prims["house_west"]
    assert float(east.mean() - west.mean()) > 5.0, (float(east.mean()), float(west.mean()))
    # The shaded face is the control: it has no beam at all, so it must sit at or below the air.
    air = float(scene.weather.at(scene.t0_s).t_air_k)
    assert west.max() < air, (float(west.max()), air)


def test_taking_the_shadow_away_collapses_the_field_onto_what_it_replaced(config: object) -> None:
    """The conservation bar (WM.2's): the change must be provably a redistribution.

    With no occluders every cell of the deck sees the same sun, the same sky and the same wind,
    so the field has to reproduce the single number the surface solved to before it was patched --
    not approximately, but to the tolerance a scalar solve of the same forcing has. If it did not,
    the field would be adding energy rather than moving it, and the measured step above would be
    an artefact of the discretisation rather than of the deckhouse.
    """
    thermal = config.scene.thermal.model_copy(update={"occluders": []})
    bare = Scene.from_config(config.scene.model_copy(update={"thermal": thermal}))
    field = _field(bare, "weather_deck")
    scalar = bare.surface_temperature_k("weather_deck", bare.t0_s)

    assert float(field.max() - field.min()) < 1e-3, "an unshaded deck is not uniform"
    assert abs(float(field.mean()) - scalar) < 0.01, (float(field.mean()), scalar)
