"""PH.2 — the wet/dry road: one object, two states, from the scene config alone.

`configs/scenes/wet_road_noon.yaml` waters the west half of a 12 m asphalt patch with a 0.5 mm
pass at 14:00 on a clear June day, with a wall shading its south rows. The checks are Hendel
et al. 2014's (Paris pavement watering, FLIR B400): the wet half runs 6–13 K colder than the dry
half in the sun and a few kelvin in the shade; the film is gone within the 15–120 min window
and the halves reconverge -- the drying spike the camera saw; and the film's mass budget closes
to 1e-6 against what evaporated. A latent term scaled wrong dries outside the window.

docs/physics-model.md §6.1, §15 Tier 3; spec issue S45; ADR 0101; roadmap PH.2.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from irsim.config.scene import load_scene_config
from irsim.scene import Scene
from irsim.thermal.latent import (
    bulk_conductance_kg_m2_s,
    evaporation_kg_m2_s,
    specific_humidity_kg_kg,
)

REPO = pathlib.Path(__file__).resolve().parents[2]
SCENE = REPO / "configs/scenes/wet_road_noon.yaml"

# GT.1: a three-hour march of a 576-cell field is a validation bench, not a unit test.
pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def road():  # type: ignore[no-untyped-def]
    """The scene marched three hours in its own 60 s ticks, with the contrasts logged."""
    scene = Scene.from_file(SCENE)
    fld = scene.surface_fields["road"]
    wet0 = fld.film_kg_m2 > 0.0
    lit = fld.field.forcing_at.cell_visibility(scene.t0_s) == 1.0
    log = []
    sun_dry_at = None
    for minute in range(0, 181):
        t = scene.t0_s + 60.0 * minute
        fld.advance_to(t)
        temps = np.asarray(fld.temperature_at(t), dtype=np.float64)
        film = fld.film_kg_m2
        log.append(
            (
                minute,
                float(temps[~wet0 & lit].mean() - temps[wet0 & lit].mean()),
                float(temps[~wet0 & ~lit].mean() - temps[wet0 & ~lit].mean()),
                float(film[wet0 & lit].max()),
            )
        )
        if sun_dry_at is None and np.all(film[wet0 & lit] == 0.0):
            sun_dry_at = minute
    return scene, fld, wet0, lit, log, sun_dry_at


def test_the_scene_is_one_prim_half_wet_and_part_shaded(road) -> None:  # type: ignore[no-untyped-def]
    scene, fld, wet0, lit, _, _ = road
    spec = load_scene_config(SCENE).scene
    assert [s.name for s in spec.thermal.surfaces] == ["road"]  # type: ignore[union-attr]
    assert spec.thermal.surfaces[0].film.depth_mm == 0.5  # type: ignore[union-attr]
    assert wet0.sum() == fld.patch.n_cells // 2  # the west half, cell for cell
    assert 0.05 < 1.0 - lit.mean() < 0.3  # the wall shades the south rows
    assert (wet0 & lit).sum() > 100 and (wet0 & ~lit).sum() > 10
    uv = fld.patch.local_coords(fld.patch.cell_centres())
    assert uv[wet0, 0].max() < 6.0 < uv[~wet0, 0].min()  # the watered half is the west half


def test_the_wet_half_runs_6_to_13_k_colder_in_the_sun_and_less_in_the_shade(road) -> None:  # type: ignore[no-untyped-def]
    _scene, _fld, _wet0, _lit, log, _ = road
    peak = max(row[1] for row in log)
    assert 6.0 <= peak <= 13.0, peak
    # Measured 9.9 K at ~27 min, the moment the sunlit film is nearly gone.
    t_peak = next(row[0] for row in log if row[1] == peak)
    assert 15 <= t_peak <= 60, t_peak
    # In the shade the contrast is smaller at the sunlit peak: evaporation runs on the surface's
    # own temperature, and a shaded cell is cooler to begin with.
    shade_at_peak = next(row[2] for row in log if row[0] == t_peak)
    assert 0.5 < shade_at_peak < peak / 2.0, shade_at_peak
    shade_peak = max(row[2] for row in log)
    assert 2.0 <= shade_peak <= 6.0, shade_peak


def test_the_film_dries_in_the_window_and_the_halves_reconverge(road) -> None:  # type: ignore[no-untyped-def]
    _scene, _fld, _wet0, _lit, log, sun_dry_at = road
    assert sun_dry_at is not None and 15 <= sun_dry_at <= 120, sun_dry_at
    peak = max(row[1] for row in log)
    at_end = log[-1][1]
    assert at_end < 0.25 * peak, (peak, at_end)  # 3 h on: the wet half is nearly the dry half
    assert all(
        a >= b
        for a, b in zip(
            [r[1] for r in log[sun_dry_at + 5 :]][:-1],
            [r[1] for r in log[sun_dry_at + 5 :]][1:],
            strict=True,
        )
    ), "monotone reconvergence after drying"


def test_the_film_mass_budget_closes_against_what_evaporated(road) -> None:  # type: ignore[no-untyped-def]
    """film₀ − film + rain = evaporated, exactly; and the solver's evaporation over the sunlit
    film agrees with an independent quadrature of E(T) at the tick temperatures to a few %."""
    scene, fld, wet0, lit, _, _ = road
    evaporated = fld.evaporated_kg_m2
    gone = 0.5 - fld.film_kg_m2
    assert np.allclose(gone[wet0], evaporated[wet0], rtol=0.0, atol=1e-6 * 0.5)
    assert np.all(evaporated[~wet0] == 0.0), "a dry cell evaporated nothing"
    # The independent quadrature: re-walk the sunlit wet cells with a fresh field, integrating
    # the bulk formula on the tick temperatures until each cell's film is spent.
    fresh = Scene.from_file(SCENE).surface_fields["road"]
    cells = np.flatnonzero(wet0 & lit)
    integral = np.zeros(cells.size)
    remaining = np.full(cells.size, 0.5)
    for minute in range(0, 60):
        t = scene.t0_s + 60.0 * minute
        fresh.advance_to(t + 60.0)
        temps = np.asarray(fresh.temperature_at(t + 30.0), dtype=np.float64)[cells]
        sample = scene.weather.at(t + 30.0)
        rate = evaporation_kg_m2_s(
            temps,
            float(specific_humidity_kg_kg(sample.t_air_k, sample.rh_fraction)),
            float(bulk_conductance_kg_m2_s(sample.wind_speed_m_s)),
        )
        step = np.minimum(remaining, np.maximum(0.0, rate) * 60.0)
        integral += step
        remaining -= step
    assert np.allclose(integral, 0.5, rtol=1e-6)  # every sunlit film is spent within the hour
    assert np.allclose(evaporated[cells], 0.5, rtol=1e-6)


def test_a_film_needs_a_patch_and_a_sensible_depth(tmp_path) -> None:  # type: ignore[no-untyped-def]
    text = SCENE.read_text()
    bad = text.replace("          depth_mm: 0.5\n", "          depth_mm: 80.0\n")
    path = tmp_path / "deep.yaml"
    path.write_text(bad)
    with pytest.raises(ValueError, match="depth_mm"):
        load_scene_config(path)
    head, _, _ = text.partition("        patch:")
    path = tmp_path / "no_patch.yaml"
    path.write_text(head + "        film: {depth_mm: 0.5}\n")
    with pytest.raises(ValueError, match="needs a `patch:`"):
        load_scene_config(path)
