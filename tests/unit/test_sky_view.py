"""PT.21 — per-cell sky view factor and diffuse shadowing.

The 145-patch dome, gated by the same ray-rectangle test the beam uses, is held to three
analytic anchors within 0.01 -- open sky 1.000, the foot of an infinite wall 0.5, under an
infinite overhang 0.5 -- and to the phenomenon it exists for: two identical cells at SVF 0.2 and
0.9 under one weather file swing with a diurnal-amplitude ratio near 2 (a hot-dry-climate study:
1.97 against 4.21 °C), and a factor applied to the solar term alone cannot reproduce it.

docs/physics-model.md §5.4, §6.1; ADR 0095, ADR 0104; roadmap PT.21.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from irsim.scene import Scene
from irsim.thermal.scene_forcing import CellForcing
from irsim.thermal.shadow import ShadowRectangle
from irsim.thermal.skyview import TREGENZA_BANDS, sky_view_factors, sub_rays, tregenza_patches
from irsim.thermal.surface_field import PlanarPatch, PlanarThermalField

REPO = pathlib.Path(__file__).resolve().parents[2]
EX = np.array([1.0, 0.0, 0.0])
EY = np.array([0.0, 1.0, 0.0])
EZ = np.array([0.0, 0.0, 1.0])
UP = EZ


def _identity(v: np.ndarray) -> np.ndarray:
    return v


def _ground(
    n: int = 4, cell: float = 0.5, origin: tuple[float, float, float] = (0.0, 0.0, 0.0)
) -> PlanarPatch:
    return PlanarPatch(
        origin_m=np.asarray(origin, dtype=float),
        u_axis=EX,
        v_axis=EY,
        n_u=n,
        n_v=n,
        du_m=cell,
        dv_m=cell,
        thickness_m=0.05,
    )


# --- the dome ------------------------------------------------------------------------------------


def test_the_dome_has_145_patches_that_tile_the_hemisphere() -> None:
    directions, omegas = tregenza_patches()
    assert directions.shape == (145, 3) and sum(n for _, n in TREGENZA_BANDS) == 145
    assert np.allclose(np.linalg.norm(directions, axis=1), 1.0)
    assert float(omegas.sum()) == pytest.approx(2.0 * np.pi, rel=1e-12)
    assert np.all(directions[:, 2] > 0.0)
    # The cosine-weighted quadrature of an isotropic dome over a horizontal cell is π within 1 %.
    assert float((omegas * directions[:, 2]).sum()) == pytest.approx(np.pi, rel=0.01)
    # The traced sub-rays tile each patch exactly and never lie on a cardinal direction.
    subs, sub_omegas, index = sub_rays()
    assert subs.shape == (145 * 12, 3)
    assert np.allclose(np.bincount(index, weights=sub_omegas), omegas)
    assert np.all(np.abs(subs[:, 0]) > 1e-9) and np.all(np.abs(subs[:, 1]) > 1e-9)


def test_the_analytic_anchors_hold_within_0_01() -> None:
    ground = _ground()
    open_sky = sky_view_factors(ground, UP, 1.0, [], _identity)
    assert np.array_equal(open_sky, np.ones(ground.n_cells))  # exactly V_s, not 0.99-something

    # At the foot of an infinite wall standing along the north edge of the cells (a wall in the
    # x-z plane at y = 2, 1 km tall and long): half the dome is gone.
    wall = ShadowRectangle(
        centre_m=np.array([1.0, 2.0, 500.0]), u_axis=EX, v_axis=EZ, half_u_m=1e6, half_v_m=500.0
    )
    foot = _ground(n=2, cell=0.01, origin=(0.99, 1.98, 0.0))  # cells a centimetre from the wall
    at_foot = sky_view_factors(foot, UP, 1.0, [wall], _identity)
    assert np.allclose(at_foot, 0.5, atol=0.01), at_foot

    # Under an infinite overhang: a horizontal slab 1 m above, covering the half-space y > 1.
    overhang = ShadowRectangle(
        centre_m=np.array([0.0, 1.0 + 5e5, 1.0]), u_axis=EX, v_axis=EY, half_u_m=1e6, half_v_m=5e5
    )
    edge = _ground(n=2, cell=0.01, origin=(0.0, 0.99, 0.0))  # cells at the overhang's edge
    under_edge = sky_view_factors(edge, UP, 1.0, [overhang], _identity)
    assert np.allclose(under_edge, 0.5, atol=0.01), under_edge
    # Deep under it, nothing; far from it, everything.
    deep = _ground(n=1, cell=0.01, origin=(0.0, 100.0, 0.0))
    assert float(sky_view_factors(deep, UP, 1.0, [overhang], _identity)[0]) == 0.0
    far = _ground(n=1, cell=0.01, origin=(0.0, -1000.0, 0.0))
    assert float(sky_view_factors(far, UP, 1.0, [overhang], _identity)[0]) == pytest.approx(
        1.0, abs=0.01
    )


def test_a_tilted_open_cell_keeps_its_tilt_s_factor_exactly_and_a_downward_one_sees_nothing() -> (
    None
):
    wall_patch = PlanarPatch(
        origin_m=np.zeros(3),
        u_axis=EZ,
        v_axis=EX,
        n_u=2,
        n_v=2,
        du_m=0.5,
        dv_m=0.5,
        thickness_m=0.05,
    )
    south = np.array([0.0, -1.0, 0.0])
    assert np.array_equal(sky_view_factors(wall_patch, south, 0.5, [], _identity), np.full(4, 0.5))
    assert np.array_equal(sky_view_factors(wall_patch, -UP, 0.0, [], _identity), np.zeros(4))


# --- what the factor does to the balance -------------------------------------------------------


@pytest.mark.slow
def test_cells_at_svf_0_2_and_0_9_swing_less_and_keep_their_night_warmth() -> None:
    """Two identical shaded asphalt cells under one weather file, one seeing a fifth of the sky
    and one nine tenths. The roadmap's anchor is a hot-dry-climate passage swinging 1.97 °C
    against 4.21 °C in the open (ratio near 2); with the hidden hemisphere at the *scene's one*
    air temperature the swing cannot fall below the air's own 12 K, and the measured ratio is
    1.25 (11.9 K against 14.9 K). What the factor does get -- and what a factor on the solar
    term alone cannot -- is the night: the enclosed cell loses less to the sky and stays 3.2 K
    warmer at its minimum, where the solar-only control leaves the two minima within 0.05 K."""
    scene = Scene.from_file(REPO / "configs/scenes/thermal_facet_scene.yaml")
    surfaces = scene.thermal.forcing_at
    i = scene.thermal_surfaces.index("asphalt_shade")  # no beam: the diffuse and the sky only
    patch = _ground(n=2, cell=0.5)
    props = scene.thermal.properties
    from irsim.thermal.facets import FacetProperties

    cells = FacetProperties(
        heat_capacity_j_m2_k=np.full(4, float(props.heat_capacity_j_m2_k[i])),
        emissivity=np.full(4, float(props.emissivity[i])),
        solar_absorptivity=np.full(4, float(props.solar_absorptivity[i])),
    )
    svf = np.array([0.2, 0.2, 0.9, 0.9])

    def day(longwave: bool) -> tuple[float, float, float]:
        forcing = CellForcing(surfaces, i, 4, sky_view=svf, sky_view_longwave=longwave)
        # The 48 h weather file starts a day before the scene: that day settles, the next counts.
        field = PlanarThermalField(
            patch, cells, forcing, 0.0, np.full(4, 290.0), 300.0, keep_ticks=None
        )
        field.advance_to(scene.t0_s + 24.0 * 3600.0)
        window = np.arange(scene.t0_s, scene.t0_s + 24.0 * 3600.0, 300.0)
        temps = np.stack([np.asarray(field.temperature_at(t), dtype=np.float64) for t in window])
        swing = temps.max(axis=0) - temps.min(axis=0)
        low, high = temps.min(axis=0)[0], temps.min(axis=0)[2]
        return float(swing[0] / swing[2]), float(low), float(high)

    ratio, low_min, high_min = day(longwave=True)
    assert 0.7 < ratio < 0.9, ratio  # measured 0.80: the enclosed cell swings less
    assert low_min - high_min > 2.0, (low_min, high_min)  # and keeps its night warmth
    control_ratio, control_low, control_high = day(longwave=False)
    assert abs(control_low - control_high) < 0.1, (control_low, control_high)  # the control cannot
    assert control_ratio < ratio  # (it damps the day a little more, having no night at all)


@pytest.mark.slow
def test_the_scene_gives_cells_their_own_sky_and_open_cells_the_tilt_s_factor() -> None:
    scene = Scene.from_file(REPO / "configs/scenes/wall_half_in_sun.yaml")
    ground = scene.surface_fields["ground"].field.forcing_at
    north = scene.surface_fields["north"].field.forcing_at
    assert ground.sky_view is not None and north.sky_view is not None
    # Ground cells far from both blocks see the whole sky; those between the blocks do not.
    uv = scene.patches["ground"].local_coords(scene.patches["ground"].cell_centres())
    corner = (uv[:, 0] < 1.0) & (uv[:, 1] > 22.0)  # the far south-west... north-west corner
    between = (uv[:, 0] > 10.5) & (uv[:, 0] < 11.5) & (uv[:, 1] > 13.0) & (uv[:, 1] < 17.0)
    # (a 6 m block 12 m away still hides a sliver of the dome: 0.97 at the corner)
    assert ground.sky_view[corner].min() > 0.95 and ground.sky_view[corner].max() < 1.0
    assert ground.sky_view[between].max() < 0.8
    # A wall sees half the sky at most, and exactly that where nothing stands in front of it; at
    # its foot the lower block to the west takes a sliver (0.478 measured).
    assert north.sky_view.max() == 0.5 and north.sky_view.min() < 0.49
    with pytest.raises(ValueError, match="sky_view has shape"):
        CellForcing(scene.thermal.forcing_at, 0, 4, sky_view=np.ones(3))
