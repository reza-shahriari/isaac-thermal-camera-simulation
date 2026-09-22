"""The RadCal transcription in `scripts/radcal.py`, checked against what it was transcribed from.

`scripts/generate_gas_luts.py` turns this port into the committed tables, and
`tests/unit/test_gas_tables.py` checks those. This file checks the step in between, because the
failure mode of transcribing four hundred lines of Fortran is not a wrong physical constant -- it
is an index off by one, and an index off by one moves a band by 25 cm-1 and changes nothing that
looks obviously wrong downstream.

Two kinds of check, then. The **grid-node** tests read a value straight out of the checked-in
RadCal array and demand the ported interpolation return it exactly, which is what catches an
index error. The **band-structure** tests demand the closed-form CO2 bands sit where CO2's bands
are, which is what catches an algebra error.

The module lives under `scripts/` because nothing in `src/irsim` may evaluate a spectral model at
render time (ADR 0098), so it is loaded here by path rather than imported.

Roadmap PH.5.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]


def _load_radcal():  # type: ignore[no-untyped-def]
    path = ROOT / "scripts" / "radcal.py"
    spec = importlib.util.spec_from_file_location("irsim_test_radcal", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


radcal = _load_radcal()
TABLES = radcal.load_radcal_tables()


def _sd_row(path: pathlib.Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = [
        [float(v) for v in line.split(",")]
        for line in path.read_text().splitlines()
        if line and not line.startswith("#") and not line.startswith("wavenumber")
    ]
    body = np.array(rows)
    header = next(line for line in path.read_text().splitlines() if line.startswith("wavenumber"))
    columns = header.split(",")[1:]
    temperatures = np.array([float(c.strip().lstrip("k_").rstrip("K")) for c in columns])
    return body[:, 0], temperatures, body[:, 1:]


# --- the interpolations land on their own grid nodes ---------------------------------------------


def _nodes_match(sampled: np.ndarray, expected: np.ndarray, *, tol: float) -> None:
    worst = float(np.max(np.abs(sampled - expected)))
    assert worst <= tol, f"worst node error {worst:.3g}"


def test_h2o_reproduces_its_table_exactly_at_every_grid_node() -> None:
    """Both axes at once: a shifted wavenumber index or a shifted temperature index fails here.

    Bit-exact, not approximate, for every temperature but the last -- RadCal evaluates its own
    top row at 2499.99 K rather than 2500 K, so the port lands a part in fifty thousand short of
    it there. That clamp is the reason the tolerance is not zero, and it is asserted rather than
    tolerated so that a *different* discrepancy cannot hide inside it.
    """
    wavenumbers, temperatures, table = _sd_row(ROOT / "data/spectra/radcal/h2o_sd.csv")
    # Past 9300 cm-1 the array is RadCal's padding, not data: the routine returns zero there.
    inside = wavenumbers < 9300.0 - 25.0
    for t_index, temp in enumerate(temperatures):
        sampled = radcal.kappa_h2o_cm_atm(wavenumbers[inside], float(temp), TABLES)
        tol = 0.0 if temp < temperatures[-1] else 1e-4 * float(table[inside, t_index].max())
        _nodes_match(sampled, table[inside, t_index], tol=tol)


def test_co2_reproduces_its_15um_table_exactly_at_every_grid_node() -> None:
    """Same check for the one CO2 band RadCal tabulates; its clamp sits at 2399.99 K."""
    wavenumbers, temperatures, table = _sd_row(ROOT / "data/spectra/radcal/co2_sd15.csv")
    inside = (wavenumbers > 500.0) & (wavenumbers < 875.0)
    for t_index, temp in enumerate(temperatures):
        sampled = radcal.kappa_co2_cm_atm(wavenumbers[inside], float(temp), TABLES)
        tol = 0.0 if temp < temperatures[-1] else 1e-4 * float(table[inside, t_index].max())
        _nodes_match(sampled, table[inside, t_index], tol=tol)


def test_a_query_outside_the_temperature_grid_is_clamped_as_radcal_clamps_it() -> None:
    """RadCal pins T to its own range rather than extrapolating; the port must do the same."""
    omega = np.array([1600.0])
    assert radcal.kappa_h2o_cm_atm(omega, 200.0, TABLES) == pytest.approx(
        radcal.kappa_h2o_cm_atm(omega, 300.0, TABLES)
    )
    assert radcal.kappa_h2o_cm_atm(omega, 4000.0, TABLES) == pytest.approx(
        radcal.kappa_h2o_cm_atm(omega, 2499.99, TABLES), rel=1e-9
    )


# --- the closed-form CO2 bands sit where CO2's bands sit ------------------------------------------


def test_the_co2_bands_are_where_spectroscopy_puts_them() -> None:
    """Peaks at 4.26 and 14.99 um, transparent between 5 and 9 um, nothing beyond 1.75 um.

    The 4.3 um band is the nu3 fundamental at 2349 cm-1 and it is by far the strongest; 15 um is
    the nu2 bend at 667 cm-1; 10.4 um is a weak difference band. Between 1100 and 1975 cm-1 CO2
    has nothing, which is why a LWIR camera sees through air that an MWIR camera does not.
    """
    grid = np.arange(400.0, 6000.0, 1.0)
    kappa = radcal.kappa_co2_cm_atm(grid, 1000.0, TABLES)
    peak = float(grid[int(np.argmax(kappa))])
    assert 2300.0 <= peak <= 2400.0, f"strongest CO2 feature at {peak} cm-1"

    window = (grid > 1150.0) & (grid < 1950.0)
    assert np.all(kappa[window] == 0.0), "CO2 absorbs in its own 5-9 um window"

    bend = (grid > 600.0) & (grid < 750.0)
    assert kappa[bend].max() > 1.0, "the 15 um bend band has gone missing"

    assert np.all(radcal.kappa_co2_cm_atm(np.array([5726.0, 8000.0]), 1000.0, TABLES) == 0.0)


def test_the_hot_wings_of_the_43um_band_open_with_temperature() -> None:
    """The reason a hot plume is visible off the cold band, and the reason the band mean rises.

    2200 cm-1 (4.55 um) is outside the fundamental's own contour at room temperature and inside
    it once the vibrationally excited states are populated. If this ever came out flat, the table
    would have lost the one temperature dependence a plume camera lives on.
    """
    wing = np.array([2200.0])
    sweep = (300.0, 1000.0, 1500.0, 2000.0)
    values = [float(radcal.kappa_co2_cm_atm(wing, t, TABLES)[0]) for t in sweep]
    assert values[0] < 1e-6, values[0]
    assert values[1] < values[2] < values[3]
    assert values[3] > 5.0, values

    core = np.array([2350.0])
    hot = [float(radcal.kappa_co2_cm_atm(core, t, TABLES)[0]) for t in (300.0, 1000.0, 2000.0)]
    assert hot[1] > hot[0] > hot[2], "the band core should peak in the middle and fall when hot"


def test_the_coefficients_are_never_negative_over_the_whole_domain() -> None:
    grid = np.arange(60.0, 9250.0, 5.0)
    for temp in (300.0, 900.0, 1800.0, 2500.0):
        assert np.all(radcal.kappa_co2_cm_atm(grid, temp, TABLES) >= 0.0)
        assert np.all(radcal.kappa_h2o_cm_atm(grid, temp, TABLES) >= 0.0)
