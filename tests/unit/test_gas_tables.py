"""The committed hot-gas absorption tables, and the contract for reading them (roadmap PH.5).

Two halves, and the second is the one that matters.

The **loader** half is ordinary: provenance is mandatory, the sidecar's hash is checked, a band
the absorption model does not reach is refused rather than zeroed.

The **data** half asserts physics about the numbers actually checked in, so that regenerating
from a different line database -- HITEMP through RADIS, one day -- has to reproduce the
phenomenology or fail here. Each of these would pass trivially against a grey coefficient and
does not: the point of a per-band table is that a band camera sees a different gas in each band,
and if that stops being true the tables have stopped being worth their disk.

docs/physics-model.md §8.1, §8.3; ADR 0098.
"""

from __future__ import annotations

import json
import pathlib
import shutil

import numpy as np
import pytest

from irsim.materials.library import nominal_response
from irsim.pipeline.gas_slab import GasSlab, SpeciesAbsorption, slab_transmittance
from irsim.pipeline.gas_tables import (
    MIN_SUPPORT_FRACTION,
    SPECIES,
    available_gas_tables,
    gas_tables_for,
    load_gas_tables,
)
from irsim.radiometry.spectral_response import load_spectral_response

DATA = pathlib.Path(__file__).resolve().parents[2] / "data"
GAS = DATA / "gas"
RESPONSES = DATA / "spectra" / "responses"

#: A car exhaust plume the way `PH.6` will author one: 25 cm across, 600 K, the usual
#: stoichiometric-ish mole fractions. Small enough that nothing is saturated, which is the regime
#: where a band model is easiest to get wrong. Measured on the committed tables, this plume walks
#: out of `PH.6`'s LWIR bound (tau >= 0.90) at about 0.30 m, so a scene that wants a thicker one
#: has to cool it or dilute it -- worth knowing before authoring, which is why it is written here.
PLUME = GasSlab(t_gas_k=600.0, length_m=0.25, p_co2_atm=0.10, p_h2o_atm=0.12)


def _response(key: str):  # type: ignore[no-untyped-def]
    path = RESPONSES / f"{key}.csv"
    return load_spectral_response(path) if path.is_file() else nominal_response(key)


# --- the loader's contract ----------------------------------------------------------------------


def test_every_committed_table_pins_the_database_that_made_it() -> None:
    """The roadmap's own acceptance item: a table whose source cannot be identified is useless.

    Not just *a* hash -- the hash of every file the generator read, so that a RadCal table edited
    by hand and a RadCal table refetched from a different FDS commit are distinguishable.
    """
    keys = available_gas_tables()
    assert keys, "no gas tables are committed"
    for key in keys:
        meta = json.loads((GAS / f"{key}_gas.json").read_text())
        assert meta["database"], f"{key}: no source database named"
        for source, digest in meta["database"].items():
            assert len(digest) == 64, f"{key}: {source} has no sha256"
            assert digest == _sha256(pathlib.Path(source)), f"{key}: {source} has changed"
        assert meta["model"].startswith("RadCal"), key
        assert meta["dtype"] == "float32"


def _sha256(path: pathlib.Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_an_edited_array_is_refused(tmp_path: pathlib.Path) -> None:
    shutil.copytree(GAS, tmp_path / "gas")
    array = tmp_path / "gas" / "mwir_co2.npy"
    table = np.load(array)
    table[0, 0] *= 1.5
    np.save(array, table, allow_pickle=False)
    with pytest.raises(ValueError, match="does not match the sidecar"):
        load_gas_tables("mwir", tmp_path)


def test_a_sidecar_without_provenance_is_refused(tmp_path: pathlib.Path) -> None:
    shutil.copytree(GAS, tmp_path / "gas")
    sidecar = tmp_path / "gas" / "mwir_gas.json"
    meta = json.loads(sidecar.read_text())
    del meta["database"]
    sidecar.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="no 'database'"):
        load_gas_tables("mwir", tmp_path)


def test_a_missing_table_names_the_generator() -> None:
    with pytest.raises(FileNotFoundError, match="generate_gas_luts"):
        load_gas_tables("no_such_band")


def test_a_band_outside_the_models_reach_is_refused_not_zeroed() -> None:
    """NIR and SWIR would otherwise read as a perfectly transparent flame.

    RadCal stops at 1.75 µm for CO2 and 1.08 µm for H2O. Those are truncations for fire heat
    transfer, not statements that a short-wave camera sees nothing, and the difference is exactly
    the kind that never surfaces if the table quietly holds zeros.
    """
    for band in ("nir", "swir"):
        with pytest.raises(ValueError, match="no usable hot-gas table"):
            load_gas_tables(band)
        meta = json.loads((GAS / f"{band}_gas.json").read_text())
        assert meta["support_fraction"]["co2"] < MIN_SUPPORT_FRACTION
    # The refusal is a floor, not a wall -- but what comes back past it is the trap, stated
    # plainly: SWIR's H2O table is real over most of the band, and its CO2 table is identically
    # zero, which would render a flame as a clear one. That is what the caller opts into.
    partial = load_gas_tables("swir", allow_partial_support=True)
    assert set(partial.species) == set(SPECIES)
    assert np.all(partial.species["co2"].kappa_per_m_atm == 0.0)
    assert partial.species["h2o"].at(1500.0, 0.05) > 0.0


def test_the_camera_gets_its_own_table_when_it_has_one() -> None:
    plain = gas_tables_for("mwir", _response("insb"))
    filtered = gas_tables_for("mwir", _response("insb_flame_window"))
    assert (plain.band, filtered.band) == ("insb", "insb_flame_window")
    assert gas_tables_for("mwir").band == "mwir"  # no response -> the registry top-hat


def test_the_arrays_are_float32_on_disk_and_float64_in_memory() -> None:
    for key in available_gas_tables():
        for species in SPECIES:
            assert np.load(GAS / f"{key}_{species}.npy").dtype == np.float32
    tables = load_gas_tables("mwir")
    assert tables.species["co2"].kappa_per_m_atm.dtype == np.float64


# --- the second axis ------------------------------------------------------------------------------


def test_a_two_dimensional_table_needs_a_column_and_the_limits_are_the_right_ones() -> None:
    temperatures = np.array([300.0, 2500.0])
    columns = np.array([0.01, 1.0])
    kappa = np.array([[10.0, 0.2], [10.0, 0.2]])
    table = SpeciesAbsorption(temperatures, kappa, columns)

    with pytest.raises(ValueError, match="column density"):
        table.at(1000.0)
    # Below the grid the medium is optically thin and the coefficient stops depending on path.
    assert table.at(1000.0, 1e-6) == pytest.approx(10.0)
    # Above it the optical depth has saturated, so kappa falls as 1/X and tau* stays put.
    assert table.at(1000.0, 100.0) * 100.0 == pytest.approx(0.2 * 1.0)
    # In between, the optical depth interpolates monotonically.
    depths = [table.at(1000.0, x) * x for x in (0.01, 0.1, 1.0)]
    assert depths[0] < depths[1] < depths[2]


def test_a_one_dimensional_table_still_reads_as_path_independent() -> None:
    table = SpeciesAbsorption(np.array([300.0, 2500.0]), np.array([4.0, 4.0]))
    assert table.at(1000.0) == pytest.approx(4.0)
    assert table.at(1000.0, 7.5) == pytest.approx(4.0)


def test_the_column_axis_is_not_decoration() -> None:
    """A single coefficient would be wrong by a large factor across a plume's own path lengths.

    This is why the table has two axes at all, so it is worth stating as a number: if this ratio
    ever collapses towards 1, CO2's 4.3 µm band has stopped dominating MWIR and something is
    wrong with the generator, not with this test.
    """
    co2 = load_gas_tables("mwir").species["co2"]
    thin = co2.at(1500.0, 0.005)
    thick = co2.at(1500.0, 0.5)
    assert thin / thick > 20.0, f"only {thin / thick:.1f}x across 0.005-0.5 atm.m"
    # Saturation, not error: more gas is never less opaque.
    depths = [co2.at(1500.0, x) * x for x in (0.005, 0.05, 0.5)]
    assert depths[0] < depths[1] < depths[2]


# --- the physics the committed numbers have to carry ----------------------------------------------


def test_hot_co2_absorbs_more_of_the_mwir_band_than_cold_co2() -> None:
    """`PH.5`'s headline check: the 4.3 µm band mean at 1500 K exceeds the ambient value.

    It is not obvious, and it is the reason the table stores an *effective* coefficient rather
    than the emission-weighted mean of κ(λ). The peak of the band gets *lower* with temperature
    as the population spreads; what grows is the band's width, as hot bands open on both sides of
    the fundamental. A camera integrates over the band, so it sees the width.
    """
    co2 = load_gas_tables("mwir").species["co2"]
    # 300 K, not the 296 K the roadmap row wrote: `GasSlab` refuses anything below 300 K as "not
    # a hot gas", so the table starts there and is not extrapolated down to a HITRAN reference.
    for column in (0.005, 0.05, 0.5):
        cold = co2.at(300.0, column)
        hot = co2.at(1500.0, column)
        assert hot > cold, f"X = {column}: 1500 K gives {hot:.3g}, 300 K gives {cold:.3g}"
    assert co2.at(1500.0, 0.05) / co2.at(300.0, 0.05) > 1.5


def test_a_plume_is_far_more_opaque_in_mwir_than_in_lwir() -> None:
    """The whole point of `PH.4`: a grey slab cannot do this, and a band camera must.

    Numbers are what `PH.6` will hold the plume to (NIRATAM's ship-plume fits): the LWIR
    bolometer sees a nearly clear plume, the MWIR camera a distinctly absorbing one.
    """
    insb = _response("insb")
    tau_mwir = slab_transmittance(PLUME, insb, gas_tables_for("mwir", insb))
    tau_lwir = slab_transmittance(
        PLUME, _response("boson_vox"), gas_tables_for("lwir", _response("boson_vox"))
    )
    assert 0.65 <= tau_mwir <= 1.0, tau_mwir
    assert tau_lwir >= 0.90, tau_lwir
    assert tau_lwir - tau_mwir > 0.05, "the two bands agree, so the table has gone grey"


def test_the_through_flame_filter_sees_almost_no_co2() -> None:
    """Why `insb_flame_window.csv` exists, checked rather than asserted in a comment.

    3.80-4.05 µm sits in the window between the 2.7 µm CO2/H2O complex and the 4.3 µm CO2 band.
    Two cameras, one band id, one file apart: no kernel knows the difference, which is what
    "bands are data, not code" has to mean to be worth anything (CLAUDE.md).
    """
    hot = GasSlab(t_gas_k=1500.0, length_m=0.5, p_co2_atm=0.12, p_h2o_atm=0.10)
    window = gas_tables_for("mwir", _response("insb_flame_window"))
    plain = gas_tables_for("mwir", _response("insb"))
    assert window.species["co2"].at(1500.0, 0.06) < 1e-6
    assert plain.species["co2"].at(1500.0, 0.06) > 1.0

    co2_only = GasSlab(t_gas_k=1500.0, length_m=0.5, p_co2_atm=0.12)
    tau_window = slab_transmittance(co2_only, _response("insb_flame_window"), window)
    tau_plain = slab_transmittance(co2_only, _response("insb"), plain)
    assert tau_window > 0.999, tau_window
    assert tau_plain < 0.85, tau_plain
    # With H2O back in, the window still transmits most of the band -- it is a gas window, not a
    # CO2 filter, which is the claim a furnace camera is sold on.
    assert slab_transmittance(hot, _response("insb_flame_window"), window) > 0.80


def test_hot_steam_is_a_strong_lwir_absorber_and_cold_steam_is_not() -> None:
    """H2O's rotation band walks up into the LWIR as it heats; this is the steam plume of `PH.9`."""
    h2o = load_gas_tables("lwir").species["h2o"]
    cold = h2o.at(300.0, 0.05)
    warm = h2o.at(400.0, 0.05)
    hot = h2o.at(1200.0, 0.05)
    assert cold < warm < hot
    assert hot / max(cold, 1e-12) > 20.0


def test_every_table_is_monotone_in_optical_depth_and_never_negative() -> None:
    for key in available_gas_tables():
        meta = json.loads((GAS / f"{key}_gas.json").read_text())
        columns = np.asarray(meta["columns_atm_m"])
        for species in SPECIES:
            table = np.load(GAS / f"{key}_{species}.npy").astype(np.float64)
            assert np.all(table >= 0.0), f"{key}/{species}: negative coefficient"
            depth = table * columns  # broadcast over the column axis
            assert np.all(np.diff(depth, axis=1) >= -1e-9), f"{key}/{species}: depth decreases"
            assert np.all(np.diff(table, axis=1) <= 1e-9), f"{key}/{species}: kappa rises with X"
