"""M7.10 — the angular columns in the packed table: (a, p, roughness) and a Level-A angle LUT.

The point of packing is that a kernel should not branch on which of §4.2's three levels a material
declares. It cannot: a Fresnel material needs an n/k file and a Planck-weighted band average, and
a GPU has neither. So `angle_lut` carries ε(θ) for **every** material on one cos θ grid, sampled
through the M7.7 dispatch, and the kernel does one lookup. The (a, p) columns stay alongside it
because §4.2's "two instructions in a shader" is cheaper still where it applies.

The grid is uniform in **cos θ**, not in θ, because a kernel has `n·v` in hand and would otherwise
have to call acos to index it. The spacing that buys is uneven in angle — one step is ~10° near
normal and ~1° near grazing — so the node count is set by measuring the interpolation error
against the exact dispatch rather than by taste: 33 nodes leaves 0.0048 on water, 65 leaves
0.0012, an order of magnitude below the 0.02 §4.2 allows Level B itself.

docs/physics-model.md §13.3, §13.5, §14; ADR 0042
"""

from __future__ import annotations

import math
import pathlib

import numpy as np
import pytest

from irsim.materials.directional import angular_level, directional_emissivity
from irsim.materials.library import MaterialLibrary
from irsim.materials.table import (
    ANGLE_LUT_COS,
    UNMAPPED_MATERIAL_ID,
    MaterialTable,
    StaleMaterialTableError,
)
from irsim.radiometry.spectral_response import load_spectral_response

REPO = pathlib.Path(__file__).resolve().parents[2]
DATA = REPO / "data"


@pytest.fixture(scope="module")
def library() -> MaterialLibrary:
    return MaterialLibrary.load(REPO / "configs" / "materials")


@pytest.fixture(scope="module")
def boson():  # type: ignore[no-untyped-def]
    return load_spectral_response(DATA / "spectra" / "responses" / "boson_vox.csv")


@pytest.fixture(scope="module")
def packed(library, boson):  # type: ignore[no-untyped-def]
    return MaterialTable.from_library(library, "lwir", boson, angle_lut=True, data_dir=DATA)


# ---------------------------------------------------------------------------------------------
# the columns
# ---------------------------------------------------------------------------------------------


def test_the_angular_columns_are_populated_for_every_level_b_material(packed, library) -> None:  # type: ignore[no-untyped-def]
    assert packed.angular_a is not None and packed.angular_p is not None
    for name in library.names:
        index = packed.id_for(name)
        a, p = float(packed.angular_a[index]), float(packed.angular_p[index])
        if angular_level(library[name]) == "B":
            model = library[name].spec.optical.angular_model
            assert a == pytest.approx(model.a, abs=1e-6), name
            assert p == pytest.approx(model.p, abs=1e-6), name
        else:
            assert np.isnan(a) or a == 0.0, f"{name} is not Level B but carries a = {a}"


def test_roughness_is_packed_per_band(packed, library) -> None:  # type: ignore[no-untyped-def]
    assert packed.roughness is not None
    glass = packed.roughness[packed.id_for("glass_windshield")]
    asphalt = packed.roughness[packed.id_for("asphalt_dry")]
    assert float(glass) < float(asphalt), "§4.3: glass is near-mirror, asphalt near-Lambertian"


def test_the_unmapped_sentinel_stays_undefined(packed) -> None:
    """Id 0 must not acquire a plausible emissivity just because the table grew a column."""
    assert np.isnan(float(packed.emissivity[UNMAPPED_MATERIAL_ID]))
    assert packed.angle_lut is not None
    assert np.all(np.isnan(packed.angle_lut[UNMAPPED_MATERIAL_ID]))


# ---------------------------------------------------------------------------------------------
# the angle LUT
# ---------------------------------------------------------------------------------------------


def test_the_lut_grid_is_uniform_in_cosine(packed) -> None:
    assert ANGLE_LUT_COS.dtype == np.float32
    assert float(ANGLE_LUT_COS[0]) == 1.0 and float(ANGLE_LUT_COS[-1]) == 0.0
    spacing = np.diff(ANGLE_LUT_COS.astype(np.float64))
    assert np.allclose(spacing, spacing[0], atol=1e-7), "the grid must be uniform in cos θ"
    assert packed.angle_lut is not None
    assert packed.angle_lut.shape == (packed.n_ids, ANGLE_LUT_COS.size)


def test_the_lut_reproduces_the_exact_dispatch_at_its_own_nodes(packed, library, boson) -> None:  # type: ignore[no-untyped-def]
    """Packing must not change the answer — only where it is available."""
    for name in library.names:
        exact = directional_emissivity(library[name], "lwir", ANGLE_LUT_COS, boson, data_dir=DATA)
        assert packed.angle_lut is not None
        packed_row = packed.angle_lut[packed.id_for(name)]
        assert np.allclose(packed_row, exact, atol=0.0, rtol=0.0), name


LUT_CHECK_MATERIALS = [
    "car_paint_black",
    "glass_windshield",
    "bare_aluminium",
    "water",
    "asphalt_dry",
]
COS_70 = math.cos(math.radians(70.0))


@pytest.mark.parametrize("name", LUT_CHECK_MATERIALS)
def test_interpolating_the_lut_is_exact_where_section_4_2_makes_a_claim(
    packed, library, boson, name: str
) -> None:
    """Inside 70°, every material interpolates to **2e-4** — two orders below Level B's own 0.02.

    70° is where §4.2 stops claiming anything, and where M7.6's fits stop. Inside it the packing
    is nowhere near the limiting approximation in the chain.
    """
    cos_theta = np.linspace(1.0, COS_70, 361, dtype=np.float32)
    exact = np.asarray(
        directional_emissivity(library[name], "lwir", cos_theta, boson, data_dir=DATA),
        dtype=np.float64,
    )
    ids = np.full(cos_theta.shape, packed.id_for(name), dtype=np.int32)
    interpolated = np.asarray(packed.epsilon_at(ids, cos_theta), dtype=np.float64)
    assert float(np.max(np.abs(exact - interpolated))) < 2e-4, name


@pytest.mark.parametrize("name", LUT_CHECK_MATERIALS)
def test_the_lut_stays_usable_out_to_eighty_seven_degrees(
    packed, library, boson, name: str
) -> None:
    """Past §4.2's bound the error grows, and the worst case is the metal: 0.010 against water's
    0.0009. That is the cos-uniform grid running out of resolution where ε is steepest."""
    cos_theta = np.linspace(1.0, 0.05, 361, dtype=np.float32)
    exact = np.asarray(
        directional_emissivity(library[name], "lwir", cos_theta, boson, data_dir=DATA),
        dtype=np.float64,
    )
    ids = np.full(cos_theta.shape, packed.id_for(name), dtype=np.int32)
    interpolated = np.asarray(packed.epsilon_at(ids, cos_theta), dtype=np.float64)
    assert float(np.max(np.abs(exact - interpolated))) < 0.011, name


def test_the_metal_s_grazing_spike_is_gone_and_the_packed_table_follows_the_dispatch(
    packed, library, boson
) -> None:
    """AT.25 (ADR 0163). Before, bare aluminium's ε reached 0.96 within 0.2° of grazing -- the
    clean metal's Fresnel shape scaled by the authored magnitude and clipped to 1 -- and this
    test *recorded* that the cos-uniform LUT could not resolve the spike (it read 0.18 there).
    With the effective n, k the angular law is a conductor's own: it peaks near 0.22 at about
    85° and goes to 0 at 90°, there is no spike to resolve, and the packed table follows the
    exact dispatch to a few hundredths everywhere, including the last degree."""
    cos_theta = np.linspace(1.0, 0.0, 721, dtype=np.float32)
    aluminium = library["bare_aluminium"]
    exact = np.asarray(
        directional_emissivity(aluminium, "lwir", cos_theta, boson, data_dir=DATA),
        dtype=np.float64,
    )
    ids = np.full(cos_theta.shape, packed.id_for("bare_aluminium"), dtype=np.int32)
    interpolated = np.asarray(packed.epsilon_at(ids, cos_theta), dtype=np.float64)
    assert exact.max() < 0.3 and exact[-1] < 1e-3, (exact.max(), exact[-1])
    theta = np.degrees(np.arccos(cos_theta.astype(np.float64)))
    assert 80.0 < theta[int(np.argmax(exact))] < 88.0
    assert np.abs(exact - interpolated).max() < 0.05, np.abs(exact - interpolated).max()


def test_the_lut_lookup_uses_the_absolute_cosine(packed) -> None:
    index = np.int32(packed.id_for("car_paint_black"))
    assert float(packed.epsilon_at(index, np.float32(-0.4))) == float(
        packed.epsilon_at(index, np.float32(0.4))
    )


def test_a_table_without_a_lut_says_so_rather_than_guessing(library, boson) -> None:  # type: ignore[no-untyped-def]
    plain = MaterialTable.from_library(library, "lwir", boson)
    assert plain.angle_lut is None
    with pytest.raises(ValueError, match="without an angle LUT"):
        plain.epsilon_at(np.int32(1), np.float32(1.0))


def test_packing_the_lut_is_off_by_default_so_older_tables_are_unchanged(library, boson) -> None:  # type: ignore[no-untyped-def]
    """Every table packed before M7.10 must still hash and load the same way."""
    plain = MaterialTable.from_library(library, "lwir", boson)
    assert plain.angle_lut is None
    assert (
        plain.library_hash
        == MaterialTable.from_library(
            library, "lwir", boson, angle_lut=True, data_dir=DATA
        ).library_hash
    )


# ---------------------------------------------------------------------------------------------
# packing, round trip, staleness
# ---------------------------------------------------------------------------------------------


def test_closure_survives_float32_packing(packed, library, boson) -> None:  # type: ignore[no-untyped-def]
    """ε + ρ + τ = 1 to 1e-6 *after* the float64 authored values are narrowed to float32."""
    assert packed.reflectance is not None and packed.transmittance is not None
    for name in library.names:
        index = packed.id_for(name)
        closure = (
            float(packed.emissivity[index])
            + float(packed.reflectance[index])
            + float(packed.transmittance[index])
        )
        assert abs(closure - 1.0) < 1e-6, (name, closure)


def test_a_deliberate_float16_pack_fails_the_same_closure_test(packed, library) -> None:  # type: ignore[no-untyped-def]
    """The counter-test, because "float32 is enough" is only meaningful if float16 is not.

    Packed in half precision, the same library's closure error reaches ~1e-3 -- a thousand times
    the tolerance -- so the float32 result above is a property of float32 and not of the numbers
    happening to be round.
    """
    assert packed.reflectance is not None and packed.transmittance is not None
    worst = 0.0
    for name in library.names:
        index = packed.id_for(name)
        halves = [
            np.float16(packed.emissivity[index]),
            np.float16(packed.reflectance[index]),
            np.float16(packed.transmittance[index]),
        ]
        worst = max(worst, abs(float(sum(np.float64(h) for h in halves)) - 1.0))
    assert worst > 1e-6, "float16 packing was expected to break closure and did not"
    with pytest.raises(TypeError, match="float16"):
        MaterialTable(emissivity=packed.emissivity.astype(np.float16))


def test_the_lut_round_trips_through_disk_by_name(packed, library, tmp_path: pathlib.Path) -> None:  # type: ignore[no-untyped-def]
    npz, side = packed.save(tmp_path / "lwir_table")
    assert npz.is_file() and side.is_file()
    reloaded = MaterialTable.load(tmp_path / "lwir_table", library)
    assert reloaded.angle_lut is not None
    for name in library.names:
        assert reloaded.id_for(name) == packed.id_for(name)
        before = packed.angle_lut[packed.id_for(name)]  # type: ignore[index]
        after = reloaded.angle_lut[reloaded.id_for(name)]
        assert np.allclose(before, after, atol=1e-6, rtol=0.0), name


def test_a_stale_table_is_rejected(packed, library, tmp_path: pathlib.Path) -> None:  # type: ignore[no-untyped-def]
    import dataclasses

    stale = dataclasses.replace(packed, library_hash="0" * 64)
    stale.save(tmp_path / "stale")
    with pytest.raises(StaleMaterialTableError, match="regenerate"):
        MaterialTable.load(tmp_path / "stale", library)
    # and without a library to check against, it loads -- staleness is the caller's question
    assert MaterialTable.load(tmp_path / "stale").band_id == packed.band_id


def test_a_float16_table_on_disk_is_refused(packed, tmp_path: pathlib.Path) -> None:
    npz, side = packed.save(tmp_path / "half")
    with np.load(npz) as data:
        arrays = {k: np.asarray(data[k]) for k in data.files}
    arrays["emissivity"] = arrays["emissivity"].astype(np.float16)
    np.savez(str(npz), **arrays)
    with pytest.raises(TypeError, match="float16"):
        MaterialTable.load(tmp_path / "half")
