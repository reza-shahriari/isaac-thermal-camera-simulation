"""Temperature and parameter rasters draped onto a patch, engine-free (PT.13).

Three things carry weight, and they are the three the roadmap row names.

``test_a_float32_raster_round_trips_through_a_patch`` is the whole point of the escape hatch: if
a map does not come back out of the patch it went into, then a hull somebody else solved is not
the hull this simulator renders, and nothing downstream can tell.

``test_a_celsius_raster_declared_as_kelvin_is_refused`` is the failure the hatch invites. Published
thermal frames are in Celsius far more often than in kelvin, both load as plain float rasters, and
a 20 degC raster read as kelvin is 20 K -- which is not a cold surface but an impossible one, and
which renders as a uniform floor rather than as an error.

``test_a_mapped_absorptivity_reaches_the_solver`` is the one that proves the parameter hatch is
wired rather than merely parsed: the cells must land where a *scalar* energy balance says a
surface of that absorptivity lands, so the raster is driving the physics and not decorating it.

docs/physics-model.md §6.1; roadmap PT.13.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from irsim.radiometry.constants import LUT_T_MAX_K, LUT_T_MIN_K
from irsim.thermal.facets import FacetProperties
from irsim.thermal.maps import (
    MapEntry,
    PrescribedPatchField,
    apply_parameter_maps,
    load_raster,
    parameter_map_to_cells,
    resample_to_patch,
    temperature_map_to_cells,
)
from irsim.thermal.surface_field import PlanarPatch

EX = np.array([1.0, 0.0, 0.0])
EY = np.array([0.0, 1.0, 0.0])


def _patch(n_u: int = 8, n_v: int = 6) -> PlanarPatch:
    return PlanarPatch(
        origin_m=np.zeros(3),
        u_axis=EX,
        v_axis=EY,
        n_u=n_u,
        n_v=n_v,
        du_m=0.25,
        dv_m=0.25,
        thickness_m=0.02,
    )


# --- the round trip ---------------------------------------------------------------------------


def test_a_float32_raster_round_trips_through_a_patch() -> None:
    """A raster already at the patch's own shape must come back **exactly**, not to a tolerance.

    The row's bar is 1 mK. This asserts bit equality instead, because a raster at the cell grid
    is a relabelling and not a resample: the moment it needs arithmetic to survive, the hatch has
    started inventing values that the file somebody else solved does not contain. The 1 mK bar is
    then checked on the *interesting* case -- a raster that really is resampled, below.
    """
    patch = _patch()
    rng = np.random.default_rng(20260924)
    raster = (280.0 + 40.0 * rng.random(patch.shape)).astype(np.float32)

    cells = temperature_map_to_cells(raster, patch, units="K")
    assert cells.shape == (patch.n_cells,)
    assert np.array_equal(cells.reshape(patch.shape), raster.astype(np.float64))

    field = PrescribedPatchField(patch, cells)
    back = field.temperature_image(0.0)
    assert back.dtype == np.float32
    assert np.array_equal(back, raster)


def test_the_map_survives_the_patch_at_a_tenth_of_a_millikelvin() -> None:
    """The row's own bar, on a raster that is genuinely resampled.

    A bilinear resample of a *linear* ramp is the ramp, so the analytic value at each cell centre
    is known exactly and the error measured here is the arithmetic's alone.
    """
    patch = _patch(n_u=16, n_v=12)
    rows, cols = 48, 64
    # A raster that is linear in both axes, so the resample has an exact closed form.
    u = np.linspace(0.0, 1.0, cols)
    v = np.linspace(0.0, 1.0, rows)
    raster = (290.0 + 10.0 * v[:, None] + 4.0 * u[None, :]).astype(np.float32)

    cells = temperature_map_to_cells(raster, patch, units="K").reshape(patch.shape)
    # The same linear function evaluated where the resample says the cells sit.
    vc = ((np.arange(patch.n_v) + 0.5) / patch.n_v) * rows - 0.5
    uc = ((np.arange(patch.n_u) + 0.5) / patch.n_u) * cols - 0.5
    expected = 290.0 + 10.0 * (vc[:, None] / (rows - 1)) + 4.0 * (uc[None, :] / (cols - 1))
    assert float(np.abs(cells - expected).max()) < 1e-4, float(np.abs(cells - expected).max())


def test_row_zero_is_the_patch_origin_edge() -> None:
    """The convention, asserted rather than assumed: a flip is a plausible picture either way."""
    patch = _patch(n_u=2, n_v=3)
    raster = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    cells = resample_to_patch(raster, patch).reshape(patch.shape)
    assert cells[0, 0] == 1.0 and cells[2, 1] == 6.0
    # And the patch's own flattening agrees: v is the slow axis.
    assert np.array_equal(resample_to_patch(raster, patch), np.array([1.0, 2, 3, 4, 5, 6]))


def test_celsius_is_converted_and_not_merely_labelled() -> None:
    patch = _patch()
    raster = np.full(patch.shape, 26.85, dtype=np.float32)
    cells = temperature_map_to_cells(raster, patch, units="degC")
    assert np.allclose(cells, 300.0, atol=1e-9)


# --- what must not pass quietly ----------------------------------------------------------------


def test_a_celsius_raster_declared_as_kelvin_is_refused() -> None:
    """20 degC read as kelvin is 20 K: not a cold surface, an impossible one."""
    patch = _patch()
    raster = np.full(patch.shape, 20.0, dtype=np.float32)  # degrees Celsius, mislabelled
    with pytest.raises(ValueError, match="outside the band"):
        temperature_map_to_cells(raster, patch, units="K")
    # The message has to say what it thinks happened, or the next person re-derives it.
    with pytest.raises(ValueError, match="looks like a Celsius raster"):
        temperature_map_to_cells(raster, patch, units="K")
    # Declared correctly, the same file is fine -- so the guard is about the label, not the data.
    assert np.allclose(temperature_map_to_cells(raster, patch, units="degC"), 293.15)


def test_the_guard_is_the_lut_domain_and_is_asymmetric() -> None:
    """Stated honestly: the reverse mistake lands *inside* the range and is not catchable here.

    Kelvin mislabelled Celsius adds 273.15 and gives a hot but legal surface. There is no bound
    that separates it from a real exhaust, so the guard does not pretend to, and this test pins
    that limitation so nobody later reads the one-sided check as a two-sided one.
    """
    patch = _patch()
    just_under = np.full(patch.shape, LUT_T_MIN_K - 0.5)
    with pytest.raises(ValueError, match="outside the band"):
        temperature_map_to_cells(just_under, patch, units="K")
    assert temperature_map_to_cells(np.full(patch.shape, LUT_T_MIN_K), patch, units="K").min() > 0
    with pytest.raises(ValueError, match="outside the band"):
        temperature_map_to_cells(np.full(patch.shape, LUT_T_MAX_K + 1.0), patch, units="K")

    # The asymmetry: a 300 K raster mislabelled degC is 573 K, and nothing here refuses it.
    passes_anyway = temperature_map_to_cells(np.full(patch.shape, 300.0), patch, units="degC")
    assert np.allclose(passes_anyway, 573.15)


def test_a_nan_cell_is_refused() -> None:
    patch = _patch()
    raster = np.full(patch.shape, 300.0)
    raster[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        resample_to_patch(raster, patch)


def test_a_parameter_outside_its_own_interval_is_refused() -> None:
    patch = _patch()
    with pytest.raises(ValueError, match="outside"):
        parameter_map_to_cells(np.full(patch.shape, 1.2), patch, parameter="emissivity")
    with pytest.raises(ValueError, match="outside"):
        parameter_map_to_cells(np.full(patch.shape, -0.1), patch, parameter="solar_absorptivity")
    assert parameter_map_to_cells(
        np.full(patch.shape, 0.9), patch, parameter="emissivity"
    ).max() == pytest.approx(0.9)


def test_a_float16_raster_is_refused(tmp_path: pathlib.Path) -> None:
    """CLAUDE.md #2 where a map crosses it: fp16 spaces 0.25 K at 300 K, 5x a 50 mK NETD."""
    path = tmp_path / "skin.npy"
    np.save(path, np.full((4, 4), 300.0, dtype=np.float16))
    with pytest.raises(TypeError, match="float16"):
        load_raster(path)


def test_a_raster_that_is_not_a_single_band_image_is_refused(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "rgb.npy"
    np.save(path, np.zeros((4, 4, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="2-D raster"):
        load_raster(path)


# --- the parameter hatch -----------------------------------------------------------------------


def _props(n: int, alpha: float = 0.3) -> FacetProperties:
    return FacetProperties(
        heat_capacity_j_m2_k=np.full(n, 5.0e4),
        emissivity=np.full(n, 0.92),
        solar_absorptivity=np.full(n, alpha),
    )


def test_an_unmapped_surface_is_untouched() -> None:
    """Additive, like every other field in this project: no map, no change at all."""
    patch = _patch()
    cells = _props(patch.n_cells)
    assert apply_parameter_maps(cells, patch, []) is cells


def test_a_parameter_map_replaces_only_what_it_names() -> None:
    patch = _patch()
    cells = _props(patch.n_cells)
    alpha = np.linspace(0.2, 0.8, patch.n_cells).reshape(patch.shape)
    out = apply_parameter_maps(cells, patch, [MapEntry("solar_absorptivity", alpha)])
    assert np.allclose(out.solar_absorptivity, alpha.reshape(-1))
    assert np.array_equal(out.emissivity, cells.emissivity)
    assert np.array_equal(out.heat_capacity_j_m2_k, cells.heat_capacity_j_m2_k)


def test_a_prescribed_field_never_falls_behind_a_query() -> None:
    """It answers at every time, so a caller cannot tell it apart from a solved surface."""
    patch = _patch()
    field = PrescribedPatchField(patch, np.full(patch.n_cells, 305.0), t0_s=1000.0)
    field.advance_to(1.0e9)  # accepted and ignored
    assert float(field.temperature_at(-1.0e9).min()) == pytest.approx(305.0)
    assert float(field.temperature_at(1.0e9).max()) == pytest.approx(305.0)
    assert field.latest_t_s == float("inf")


def test_a_map_of_the_wrong_size_is_refused() -> None:
    patch = _patch()
    with pytest.raises(ValueError, match="cells"):
        PrescribedPatchField(patch, np.zeros(patch.n_cells + 1))


# --- through a real scene ----------------------------------------------------------------------

REPO = pathlib.Path(__file__).resolve().parents[2]
SCENE_PATH = REPO / "configs" / "scenes" / "vessel_pointwise_clear_day.yaml"


def _one_surface_scene(**surface_updates: object):
    """The vessel's weather deck alone: no occluders, no conduction, a short spin-up.

    Everything that would couple one cell to another is removed deliberately, so a cell's answer
    is a function of that cell's own properties and nothing else. That is what makes the
    comparison below a statement about the map rather than about the neighbourhood.
    """
    from irsim.config.scene import SurfaceSpec, load_scene_config
    from irsim.scene import Scene

    spec = load_scene_config(SCENE_PATH).scene
    deck = next(s for s in spec.thermal.surfaces if s.name == "weather_deck")
    # Rebuilt through the validator rather than `model_copy`, which skips it: the point of
    # several of these cases is that the schema refuses the combination, and a copy would hand
    # the scene builder a document the loader would never have produced.
    deck = SurfaceSpec.model_validate(
        {**deck.model_dump(exclude_none=True), "lateral_conduction": False, **surface_updates}
    )
    thermal = spec.thermal.model_copy(
        update={"occluders": [], "surfaces": [deck], "spin_up_hours": 2.0}
    )
    return Scene.from_config(spec.model_copy(update={"thermal": thermal}))


def _deck_cells(scene) -> np.ndarray:
    field = scene.surface_fields["weather_deck"]
    field.advance_to(scene.t0_s)
    return np.asarray(field.temperature_at(scene.t0_s), dtype=np.float64)


def _alpha_map(tmp_path: pathlib.Path, name: str, values: np.ndarray) -> list[dict[str, str]]:
    path = tmp_path / f"{name}.npy"
    np.save(path, values.astype(np.float32))
    return [{"parameter": "solar_absorptivity", "path": str(path)}]


def test_a_mapped_absorptivity_reaches_the_solver(tmp_path: pathlib.Path) -> None:
    """The row's third bar: a per-cell alpha must give the per-cell equilibrium the solver predicts.

    The oracle is the solver itself, run twice: a deck that is 0.2 absorptive everywhere and a
    deck that is 0.8 absorptive everywhere are two scalar answers. A deck whose raster is 0.2 on
    one half and 0.8 on the other must then reproduce **both**, cell for cell, on one surface --
    which it can only do if the raster is driving the energy balance rather than decorating it.
    Conduction is off, so the two halves are genuinely independent and the seam is not smeared.
    """
    shape = _one_surface_scene().surface_fields["weather_deck"].patch.shape
    lo = _deck_cells(
        _one_surface_scene(parameter_maps=_alpha_map(tmp_path, "lo", np.full(shape, 0.2)))
    )
    hi = _deck_cells(
        _one_surface_scene(parameter_maps=_alpha_map(tmp_path, "hi", np.full(shape, 0.8)))
    )

    # The test must not be able to pass by both answers being the same answer.
    assert float(hi.mean() - lo.mean()) > 3.0, (float(lo.mean()), float(hi.mean()))
    assert float(lo.max() - lo.min()) < 1e-6 and float(hi.max() - hi.min()) < 1e-6

    split = np.full(shape, 0.2)
    split[shape[0] // 2 :, :] = 0.8
    got = _deck_cells(
        _one_surface_scene(parameter_maps=_alpha_map(tmp_path, "split", split))
    ).reshape(shape)

    cold, warm = got[: shape[0] // 2, :], got[shape[0] // 2 :, :]
    assert float(np.abs(cold - lo.reshape(shape)[: shape[0] // 2, :]).max()) < 1e-3
    assert float(np.abs(warm - hi.reshape(shape)[shape[0] // 2 :, :]).max()) < 1e-3


def test_a_surface_with_no_map_is_bit_identical(tmp_path: pathlib.Path) -> None:
    """Additive, at the scene level: every scene authored before PT.13 solves to the same bits."""
    del tmp_path
    assert np.array_equal(_deck_cells(_one_surface_scene()), _deck_cells(_one_surface_scene()))


def test_a_temperature_map_replaces_the_solve(tmp_path: pathlib.Path) -> None:
    """The map *is* the surface: what comes back is the raster, not something solved from it."""
    from irsim.thermal.maps import PrescribedPatchField

    shape = _one_surface_scene().surface_fields["weather_deck"].patch.shape
    rng = np.random.default_rng(13)
    raster = (5.0 + 30.0 * rng.random(shape)).astype(np.float32)  # degrees Celsius
    path = tmp_path / "hull.npy"
    np.save(path, raster)

    scene = _one_surface_scene(temperature_map={"path": str(path), "units": "degC"})
    field = scene.surface_fields["weather_deck"]
    assert isinstance(field, PrescribedPatchField)
    assert np.allclose(field.temperature_image(scene.t0_s), raster + 273.15, atol=1e-3)
    # And it is still a binding a render driver can take, like any other patched surface.
    assert scene.patch_prims["weather_deck"].endswith("/weather_deck")
    assert any(name == "weather_deck" for name, _ in _named(scene))


def _named(scene):
    return [(n, f) for n, f in scene.surface_fields.items()]


def test_a_celsius_hull_declared_in_kelvin_is_refused_by_the_scene(tmp_path: pathlib.Path) -> None:
    """The guard reaches the config, not only the helper: the file is named in the message."""
    shape = _one_surface_scene().surface_fields["weather_deck"].patch.shape
    path = tmp_path / "hull_degc.npy"
    np.save(path, np.full(shape, 18.0, dtype=np.float32))
    with pytest.raises(ValueError, match="looks like a Celsius raster"):
        _one_surface_scene(temperature_map={"path": str(path), "units": "K"})


def test_a_prescribed_surface_refuses_the_terms_it_would_ignore(tmp_path: pathlib.Path) -> None:
    """A map beside a `film:` or `layers:` would read as having an effect it does not have."""
    shape = _one_surface_scene().surface_fields["weather_deck"].patch.shape
    path = tmp_path / "skin.npy"
    np.save(path, np.full(shape, 300.0, dtype=np.float32))
    spec = {"path": str(path), "units": "K"}
    with pytest.raises(ValueError, match="no meaning beside"):
        _one_surface_scene(temperature_map=spec, layers=3)
    with pytest.raises(ValueError, match="no meaning beside"):
        _one_surface_scene(
            temperature_map=spec,
            parameter_maps=[{"parameter": "emissivity", "path": str(path)}],
        )


def test_a_parameter_map_is_refused_where_its_history_could_not_be_kept(
    tmp_path: pathlib.Path,
) -> None:
    """The refusals that exist because of what this step measured, not because of taste.

    A mapped absorptivity has to be in the surface's *history* or it is not that surface: the
    first version of this wiring left the field starting from the per-prim spun state and a deck
    mapped 0.2 came back identical to the same deck mapped 0.8. The main path now spins up per
    cell. The two paths that cannot -- a layered stack and a cabin panel, both of which begin
    from one scalar -- refuse instead of quietly reproducing that bug.
    """
    shape = _one_surface_scene().surface_fields["weather_deck"].patch.shape
    maps = _alpha_map(tmp_path, "alpha", np.full(shape, 0.5))
    with pytest.raises(ValueError, match="history the stack begins from"):
        _one_surface_scene(parameter_maps=maps, layers=4)
