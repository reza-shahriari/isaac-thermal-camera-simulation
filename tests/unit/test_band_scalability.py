"""M11.1 — "bands are data, not code", asserted rather than asserted-in-prose.

Two halves, and the second is the one that will fail first:

* a **dynamic** half parametrised over every YAML in ``configs/sensors/``: each one loads,
  classifies into a canonical band by overlap, hashes reproducibly and distinctly, and tabulates
  a band LUT -- all through the same functions, with no branch anywhere on which band it is.
  Adding ``example_swir_ingaas_640.yaml`` next to the LWIR Boson exercised exactly this path and
  needed no code change; this test is what keeps that true.
* a **static** half over **every** module under ``src/irsim``: none may contain a band name, in a
  string or in an identifier, nor a nominal band edge written as a wavelength. The registry in
  ``irsim.config.bands`` is the single place that knows the bands; a kernel that learns one has
  broken the abstraction CLAUDE.md calls the main scalability requirement of the project, and it
  breaks it silently -- the code still works, for the bands that were thought of.

  The static half used to name the six packages it scanned (``CORE_PACKAGES``), which made the
  exemption an **omission**: ``atmosphere``, ``materials``, ``thermal``, ``io``, ``validation``,
  ``config`` and ``scene.py`` were unguarded because nobody had listed them, not because anybody
  had decided they should be. Four of those were already clean and had been for their whole lives.
  It also had two ways to switch itself off in silence -- a package renamed in the tuple made
  ``rglob`` walk a path that does not exist, which yields no files and therefore no offences, and
  the one coverage assertion counted the six packages' union against a floor of 30, which survives
  dropping the largest of them. AT.4 inverts it: scan everything, and carry the carve-out as
  :data:`BAND_AWARE`, a file-by-file constant with a reason and a ceiling for each entry.

docs/physics-model.md §12.1, §12.2 line 900, §16.4 step 10; ADR 0092
"""

from __future__ import annotations

import ast
import json
import pathlib

import numpy as np
import pytest

from irsim.config.bands import (
    ANCHOR_BAND,
    BAND_IDS,
    BAND_KEYS,
    NOMINAL_RANGES_UM,
    band_id_for,
    overlap_fraction,
)
from irsim.config.loader import band_hash, config_hash, load_sensor_config
from irsim.config.sensor import BolometerFpa, PhotonFpa
from irsim.radiometry.constants import C_LIGHT, H_PLANCK
from irsim.radiometry.lut import BandLUT
from irsim.radiometry.spectral_response import load_spectral_response

REPO = pathlib.Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO / "configs" / "sensors"
DATA_DIR = REPO / "data"
SRC = REPO / "src" / "irsim"

# A coarse grid: this half of the suite is about *whether* a band tabulates, not about quadrature
# accuracy, which tests/unit/test_planck.py owns at the full 0.05 K spacing.
COARSE_N = 161

SENSOR_CONFIGS = sorted(CONFIG_DIR.glob("*.yaml"))
CONFIG_IDS = [p.stem for p in SENSOR_CONFIGS]


@pytest.fixture(scope="module")
def configs():  # type: ignore[no-untyped-def]
    return {p.stem: load_sensor_config(p, DATA_DIR) for p in SENSOR_CONFIGS}


def test_the_config_directory_is_not_empty() -> None:
    """Guards the guard: every parametrised test below would vacuously pass on an empty glob."""
    assert SENSOR_CONFIGS, f"no sensor YAML under {CONFIG_DIR}"


def test_more_than_one_band_is_configured(configs) -> None:  # type: ignore[no-untyped-def]
    """A scalability claim tested on one band is not tested at all."""
    ids = {c.sensor.band.band_id for c in configs.values()}
    assert len(ids) >= 2, f"only band {ids} is configured; M11.1 adds SWIR beside LWIR"


# ---------------------------------------------------------------------------------------------
# dynamic half: every configured band goes through the same code
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", CONFIG_IDS)
def test_every_config_loads_and_classifies_by_overlap(name, configs) -> None:  # type: ignore[no-untyped-def]
    band = configs[name].sensor.band
    derived = band_id_for(band.lambda_min_um, band.lambda_max_um)
    assert derived in BAND_IDS
    assert band.band_id == derived
    # Classification is by *maximal* overlap, so the winner must beat every other band, not
    # merely clear the 50 % floor.
    others = {b: overlap_fraction(band.lambda_min_um, band.lambda_max_um, b) for b in BAND_IDS}
    assert others[derived] == max(others.values())
    assert others[derived] > max(v for b, v in others.items() if b != derived)


@pytest.mark.parametrize("name", CONFIG_IDS)
def test_every_config_hashes_reproducibly(name, configs) -> None:  # type: ignore[no-untyped-def]
    cfg = configs[name]
    assert config_hash(cfg, DATA_DIR) == config_hash(
        load_sensor_config(CONFIG_DIR / f"{name}.yaml", DATA_DIR), DATA_DIR
    )
    assert len(band_hash(cfg, DATA_DIR)) == 64


def test_the_configs_hash_distinctly(configs) -> None:  # type: ignore[no-untyped-def]
    """Distinct *cameras*, and a band hash that collides **exactly when the bands are the same**.

    The second half used to assert every band hash was distinct, which held only while every
    config was a different band. It is not the invariant, and M12.1's Halmstad camera -- a Boson
    320x256 with the same LWIR response as the 640 -- is the case that shows why: two cameras in
    one band *must* share a band hash, because that is what lets `make luts` build one LUT bundle
    for both instead of two identical ones. The invariant is the iff.
    """
    hashes = {name: config_hash(c, DATA_DIR) for name, c in configs.items()}
    assert len(set(hashes.values())) == len(hashes), hashes

    bands = {name: band_hash(c, DATA_DIR) for name, c in configs.items()}
    blocks = {
        name: json.dumps(c.model_dump(mode="json")["sensor"]["band"], sort_keys=True)
        for name, c in configs.items()
    }
    for a in configs:
        for b in configs:
            same_block = blocks[a] == blocks[b]
            same_hash = bands[a] == bands[b]
            assert same_hash == same_block, (a, b, bands[a], bands[b])


@pytest.mark.parametrize("name", CONFIG_IDS)
def test_every_config_tabulates_a_lut_with_no_code_change(name, configs) -> None:  # type: ignore[no-untyped-def]
    """The same ``BandLUT.build`` over the same four quantities, whatever the band."""
    response = load_spectral_response(configs[name].sensor.band.spectral_response)
    lut = BandLUT.build(response, n=COARSE_N)
    assert lut.lb.dtype == np.float32 and lut.lb_q.dtype == np.float32
    assert np.all(np.diff(lut.lb) > 0.0), "Lb must rise with temperature in every band"
    assert np.all(np.diff(lut.lb_q) > 0.0)
    assert np.all(lut.dlb_dt > 0.0) and np.all(lut.dlb_q_dt > 0.0)
    # A band with no signal at all would pass the monotonicity checks as a table of zeros.
    assert float(lut.lb[0]) > 0.0


@pytest.mark.parametrize("name", CONFIG_IDS)
def test_the_response_curve_covers_the_configured_band(name, configs) -> None:  # type: ignore[no-untyped-def]
    band = configs[name].sensor.band
    lo, hi = load_spectral_response(band.spectral_response).support_um
    assert lo <= band.lambda_min_um and hi >= band.lambda_max_um


@pytest.mark.parametrize("name", CONFIG_IDS)
def test_the_declared_regime_is_physically_possible_for_the_band(name, configs) -> None:  # type: ignore[no-untyped-def]
    """§12.1: no self-emission below ~2.5 um, no reflected sun beyond ~3 um."""
    from irsim.config.sensor import EMISSIVE_MIN_LAMBDA_MAX_UM, REFLECTIVE_MAX_LAMBDA_MIN_UM

    band = configs[name].sensor.band
    if band.regime == "emissive":
        assert band.lambda_max_um > EMISSIVE_MIN_LAMBDA_MAX_UM
    if band.regime == "reflective":
        assert band.lambda_min_um < REFLECTIVE_MAX_LAMBDA_MIN_UM


# ---------------------------------------------------------------------------------------------
# the SWIR config itself
# ---------------------------------------------------------------------------------------------


def test_the_swir_config_is_a_photon_fpa_with_no_bolometer_fields(configs) -> None:  # type: ignore[no-untyped-def]
    fpa = configs["example_swir_ingaas_640"].sensor.fpa
    assert isinstance(fpa, PhotonFpa) and not isinstance(fpa, BolometerFpa)
    assert fpa.thermal_time_constant_ms is None and fpa.tcr_per_k is None
    assert fpa.quantum_efficiency > 0 and fpa.well_capacity_e > 0
    assert fpa.integration_time_ms <= 1000.0 / fpa.frame_rate_hz


def test_the_swir_config_is_reflective_and_claims_no_apparent_temperature(configs) -> None:  # type: ignore[no-untyped-def]
    """Inverting Lb gives a temperature only where the signal *is* self-emission."""
    sensor = configs["example_swir_ingaas_640"].sensor
    assert sensor.band.regime == "reflective"
    assert sensor.outputs.apparent_temperature is False


def test_a_300_k_scene_is_invisible_in_swir(configs) -> None:  # type: ignore[no-untyped-def]
    """The known answer that makes "reflective" a fact rather than a label.

    Compares the two configured bands' self-emission from the same 300 K blackbody, with the
    SWIR side converted out of photon units by its own band-mean photon energy -- so the ratio
    is energy over energy and the photon path is exercised, not bypassed.
    """
    swir = configs["example_swir_ingaas_640"].sensor.band
    lwir = configs["flir_boson_640_lwir"].sensor.band
    r_swir = load_spectral_response(swir.spectral_response)
    r_lwir = load_spectral_response(lwir.spectral_response)
    lut_swir = BandLUT.build(r_swir, n=COARSE_N)
    lut_lwir = BandLUT.build(r_lwir, n=COARSE_N)

    photon_energy_j = H_PLANCK * C_LIGHT / (r_swir.mean_wavelength_um() * 1e-6)
    lb_swir_from_photons = float(lut_swir.lookup(300.0, "lb_q")[()]) * photon_energy_j
    ratio = lb_swir_from_photons / float(lut_lwir.lookup(300.0)[()])
    assert ratio < 1e-6, f"300 K self-emission ratio SWIR/LWIR = {ratio:.3e}"

    # And the photon-unit table agrees with the energy-unit one to the band-mean photon energy:
    # a band this narrow in hc/lambda is well described by a single mean.
    assert lb_swir_from_photons == pytest.approx(float(lut_swir.lookup(300.0)[()]), rel=0.3)


def test_swir_self_emission_becomes_significant_at_exhaust_temperatures(configs) -> None:  # type: ignore[no-untyped-def]
    """The other side of the same coin: "reflective" is a statement about 300 K, not about the band.

    Between 300 K and 900 K the SWIR band's own emission rises by many orders of magnitude while
    the LWIR band's rises by about one -- which is why M11.2 refuses to cull emission in a
    reflective band.
    """
    lut_swir = BandLUT.build(
        load_spectral_response(configs["example_swir_ingaas_640"].sensor.band.spectral_response),
        n=COARSE_N,
    )
    lut_lwir = BandLUT.build(
        load_spectral_response(configs["flir_boson_640_lwir"].sensor.band.spectral_response),
        n=COARSE_N,
    )
    swir_gain = float(lut_swir.lookup(900.0)[()]) / float(lut_swir.lookup(300.0)[()])
    lwir_gain = float(lut_lwir.lookup(900.0)[()]) / float(lut_lwir.lookup(300.0)[()])
    assert swir_gain > 1e6
    assert lwir_gain < 100.0


def test_the_nir_config_is_a_photon_fpa_that_claims_no_apparent_temperature(configs) -> None:  # type: ignore[no-untyped-def]
    nir = configs["example_nir_si_1280"].sensor
    assert isinstance(nir.fpa, PhotonFpa) and not isinstance(nir.fpa, BolometerFpa)
    assert nir.fpa.thermal_time_constant_ms is None and nir.fpa.tcr_per_k is None
    assert nir.band.regime == "reflective"
    assert nir.outputs.apparent_temperature is False


def test_nir_is_the_extreme_case_of_reflective(configs) -> None:  # type: ignore[no-untyped-def]
    """The fourth band, and the one that makes the point hardest.

    "Reflective" is a statement about how much of the signal is the scene's own emission, and it
    is quantitative. At 300 K the SWIR band's self-emission is already 1e-9 of the LWIR band's;
    the NIR band's is smaller again by several more orders, because Planck's Wien tail falls
    another factor of e^(hc/kT (1/0.9 - 1/1.3)) between the two. A camera whose own band cannot
    see a 300 K object at all is one for which *every* photon is borrowed -- sun, moon or
    airglow -- which is what makes the M11.3/M11.4 illumination terms load-bearing rather than a
    refinement.
    """
    bands = {
        name: load_spectral_response(configs[name].sensor.band.spectral_response)
        for name in ("example_nir_si_1280", "example_swir_ingaas_640", "flir_boson_640_lwir")
    }
    luts = {name: BandLUT.build(response, n=COARSE_N) for name, response in bands.items()}

    def energy_at(name: str, temp: float) -> float:
        photon_j = H_PLANCK * C_LIGHT / (bands[name].mean_wavelength_um() * 1e-6)
        return float(luts[name].lookup(temp, "lb_q")[()]) * photon_j

    lwir_300 = float(luts["flir_boson_640_lwir"].lookup(300.0)[()])
    nir_300 = energy_at("example_nir_si_1280", 300.0)
    swir_300 = energy_at("example_swir_ingaas_640", 300.0)
    assert nir_300 / lwir_300 < 1e-12, f"NIR/LWIR at 300 K = {nir_300 / lwir_300:.3e}"
    assert nir_300 < swir_300 / 1e3, "NIR must be far deeper into the Wien tail than SWIR"

    # ...and, as with SWIR, emission is not culled: the same band lights up at flame temperature,
    # which is why the regime gates the *illumination* terms and never the emissive one.
    nir_gain = float(luts["example_nir_si_1280"].lookup(900.0)[()]) / float(
        luts["example_nir_si_1280"].lookup(300.0)[()]
    )
    assert nir_gain > 1e9, f"NIR 300 -> 900 K gain = {nir_gain:.3e}"


# ---------------------------------------------------------------------------------------------
# static half: no module under src/irsim may know a band's name or its edges
# ---------------------------------------------------------------------------------------------

#: Names whose value is a wavelength. A nominal band edge is only evidence of a hard-coded band
#: when it is *used as a wavelength*; 1.0 and 3 are otherwise the most ordinary numbers there are.
WAVELENGTH_NAME_MARKERS = ("lambda", "wavelength", "_um", "micron")

#: The four camera bands' nominal edges -- and deliberately **not** the anchor band's (0.4, 0.7).
#: Those two are photopic definition points, and adding them buys one true positive against one
#: false one: ``irsim.radiometry.solar.TRUSTED_MIN_UM = 0.7`` is the wavelength below which the
#: 5778 K solar model departs from the real spectrum, which is a fact about the sun rather than a
#: band limit, and exempting a clean radiometry module to catch it would cost more coverage than
#: it wins. The *name* half of the guard does cover the anchor (see :data:`BAND_NAMES`).
NOMINAL_EDGES_UM = frozenset(e for pair in NOMINAL_RANGES_UM.values() for e in pair)

#: Every band key the physics may be asked about, the atmosphere's anchor included. The guard used
#: to hunt only ``BAND_IDS``, which left it blind to the fifth band the codebase already contains:
#: ``visible`` has a spectral-class table, a Planck weighting temperature, a mandatory key in all
#: seven atmosphere presets and a branch that tested for it by name, and not one of those tripped a
#: guard whose forbidden set stopped at the four ``BandId`` values.
BAND_NAMES = frozenset(BAND_KEYS)

#: The carve-out, as a constant with a reason -- the thing AT.4's exit bar asks for in place of an
#: omission. Each value is ``(ceiling, why)``: the ceiling is a **ratchet**, not a target. It may
#: fall (and a stale entry is caught by ``test_every_exemption_is_still_load_bearing``); it may rise
#: only by someone editing this table, which is the deliberate friction. Paths are relative to
#: ``src/irsim``.
BAND_AWARE: dict[str, tuple[int, str]] = {
    "config/bands.py": (
        25,
        "The registry. This is the one module whose job is to know the band names and their "
        "nominal edges; every other entry below is a place that has not yet been routed through "
        "it. Adding a fifth band raises this count, and tripping the ceiling is the intended "
        "moment to re-read the rest of this table.",
    ),
    "config/environment.py": (
        4,
        "DELTA_T_LWIR_RANGE_K and the SkySpec key it validates. §5.3's 55-70 K clear-sky zenith "
        "depression is a property of the 8-14 um window and means nothing in SWIR, so the band "
        "name is carrying real physics rather than a convention. Laundering it into a derived "
        "lookup would make the constant claim to generalise when it does not. AT.9 owns the "
        "question of what the other bands' depressions are.",
    ),
    "config/sensor.py": (
        1,
        "REFLECTIVE_MAX_LAMBDA_MIN_UM = 3.0 is the solar/thermal crossover -- beyond it reflected "
        "sunlight is negligible -- and it happens to equal NOMINAL_RANGES_UM['mwir'][0]. A false "
        "positive of the edge half: the number is physics that the classifier's nominal table "
        "coincides with, not a band limit copied into the schema.",
    ),
    "config/atmosphere.py": (
        2,
        "AtmosphereBandCoefficients.aerosol_ratio_to_visible. The anchor band's name is inside a "
        "pydantic *field* name, so it is a key in all seven preset YAMLs on disk; renaming it is a "
        "breaking schema change for a quantity whose definition is the anchor. The band keys "
        "themselves now derive from the registry (ATMOSPHERE_BAND_KEYS = frozenset(BAND_KEYS)).",
    ),
    "atmosphere/extinction.py": (
        13,
        "The Koschmieder machinery: meteorological optical range is *defined* photopically, so "
        "gamma_mol_visible_per_m and aerosol_ratio_to_visible name the reference the ratio is "
        "taken against. Same field-name constraint as config/atmosphere.py. The band *keys* are "
        "gone -- the anchor is read as bands[ANCHOR_BAND] from the registry.",
    ),
    "atmosphere/layered.py": (
        2,
        "The two Koschmieder identifiers, `gamma_aerosol_visible` and `aerosol_ratio_to_visible`: "
        "the same physics `atmosphere/extinction.py`'s entry states, since meteorological optical "
        "range is *defined* photopically and the ratio has to name what it is taken against. "
        "BAND_CLASSES is gone (AT.10, ADR 0113) -- the five per-band spectral-class tables it held "
        "are derived from one wavelength-ordered ladder by intersecting a band's own span, so a "
        "fifth band needs no edit here. That was the one carve-out of the six that was neither "
        "physics nor a schema constraint, and the one ADR 0092 recorded as debt.",
    ),
}


def _guarded_sources() -> list[pathlib.Path]:
    """Every module under ``src/irsim`` that is not in the carve-out."""
    return [p for p in sorted(SRC.rglob("*.py")) if _rel(p) not in BAND_AWARE]


def _rel(path: pathlib.Path) -> str:
    return path.relative_to(SRC).as_posix()


def _is_wavelength_name(name: str) -> bool:
    lowered = name.lower()
    return any(marker in lowered for marker in WAVELENGTH_NAME_MARKERS)


def band_offences(source: str, filename: str = "<scan>") -> list[str]:
    """Every band name and nominal band edge ``source`` states outright.

    One function for both halves, so a file's carve-out ceiling counts the same thing the guard
    forbids. ``if band == "lwir"`` and ``SWIR_CUTOFF`` are the same mistake wearing different
    clothes, and ``lambda_cut_um = 13.5`` is the third outfit.
    """
    offences: list[str] = []
    tree = ast.parse(source, filename)
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and node.value.strip().lower() in BAND_NAMES
        ):
            offences.append(f"{filename}:{node.lineno} string {node.value!r}")
        names: list[str] = []
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names = [node.name]
        elif isinstance(node, ast.Name):
            names = [node.id]
        elif isinstance(node, ast.arg):
            names = [node.arg]
        elif isinstance(node, ast.Attribute):
            names = [node.attr]
        for name in names:
            parts = name.lower().replace("-", "_").split("_")
            if any(part in BAND_NAMES for part in parts):
                offences.append(f"{filename}:{node.lineno} identifier {name!r}")
        targets: list[tuple[str, ast.expr]] = []
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    targets.append((target.id, node.value))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.value is not None:
                targets.append((node.target.id, node.value))
        elif isinstance(node, ast.Compare) and isinstance(node.left, ast.Name):
            for comparator in node.comparators:
                targets.append((node.left.id, comparator))
        for name, value in targets:
            if not _is_wavelength_name(name):
                continue
            for constant in ast.walk(value):
                if not isinstance(constant, ast.Constant):
                    continue
                number = constant.value
                if isinstance(number, bool) or not isinstance(number, (int, float)):
                    continue
                if float(number) in NOMINAL_EDGES_UM:
                    offences.append(f"{filename}:{node.lineno} edge {name} = {number}")
    return sorted(set(offences))


def _offences_of(path: pathlib.Path) -> list[str]:
    return band_offences(path.read_text(encoding="utf-8"), _rel(path))


# -- the guard ---------------------------------------------------------------------------------


@pytest.mark.parametrize("path", _guarded_sources(), ids=_rel)
def test_no_guarded_module_names_a_band(path: pathlib.Path) -> None:
    """Deny by default: a module is guarded the day it is created, not the day it is listed."""
    offences = _offences_of(path)
    assert not offences, (
        f"{_rel(path)} names a band; the registry in irsim.config.bands is the only place that "
        f"may. If this file genuinely must, add it to BAND_AWARE with its reason:\n  "
        + "\n  ".join(offences)
    )


# -- the guard's own coverage ------------------------------------------------------------------


def test_the_scan_reaches_every_source_file() -> None:
    """The failure the old shape could not see: a guard scanning nothing passes.

    ``CORE_PACKAGES`` named directories and re-globbed each one, so a package renamed or moved in
    that tuple walked a path that does not exist -- no files, no offences, green. Scanning from a
    single root and asserting the root is populated removes the whole class.
    """
    found = sorted(SRC.rglob("*.py"))
    assert len(found) > 100, f"only {len(found)} modules found under {SRC}"
    assert len(_guarded_sources()) == len(found) - len(BAND_AWARE)


def test_every_top_level_package_is_covered() -> None:
    """No package may be silently absent -- the second half of the same failure.

    The old coverage assertion counted six packages' union against a floor of 30 files; the six
    hold 72, so dropping even the largest of them still passed. This asserts presence per package
    instead of a total, so a package cannot fall out of the scan unnoticed.
    """
    packages = {p.name for p in SRC.iterdir() if p.is_dir() and not p.name.startswith("__")}
    assert packages >= {
        "radiometry",
        "optics",
        "detector",
        "noise",
        "isp",
        "pipeline",
        "atmosphere",
        "materials",
        "thermal",
        "config",
        "io",
        "validation",
    }, packages
    scanned = {_rel(p).split("/")[0] for p in sorted(SRC.rglob("*.py"))}
    for package in packages:
        assert package in scanned, f"{package} contributed no module to the scan"
    assert "scene.py" in {_rel(p) for p in sorted(SRC.rglob("*.py"))}


# -- the carve-out, kept honest ----------------------------------------------------------------


def test_every_exemption_names_a_file_that_exists() -> None:
    for rel in BAND_AWARE:
        assert (SRC / rel).is_file(), f"BAND_AWARE names {rel}, which is not under {SRC}"


@pytest.mark.parametrize("rel", sorted(BAND_AWARE))
def test_every_exemption_is_still_load_bearing(rel: str) -> None:
    """The ratchet that removes a carve-out once it stops rescuing anything.

    ``irsim.materials`` was exempt for its whole life and had **zero** offences the entire time --
    the same shape as the float16 carve-out IG.8 removed. An exemption nobody can justify by a
    failing line is an exemption that should be deleted, and only a test notices.
    """
    offences = _offences_of(SRC / rel)
    assert offences, (
        f"{rel} is in BAND_AWARE but names no band: the carve-out rescues nothing. Delete its "
        "row and let the guard cover it."
    )


@pytest.mark.parametrize("rel", sorted(BAND_AWARE))
def test_no_exemption_has_grown(rel: str) -> None:
    """A carve-out may shrink freely and may not grow by accident."""
    ceiling, why = BAND_AWARE[rel]
    offences = _offences_of(SRC / rel)
    assert len(offences) <= ceiling, (
        f"{rel} now names a band in {len(offences)} places, above its ceiling of {ceiling}. Its "
        f"carve-out exists because: {why}\nIf the new ones belong to that reason, raise the "
        f"ceiling deliberately; otherwise route them through irsim.config.bands.\n  "
        + "\n  ".join(offences)
    )


@pytest.mark.parametrize("rel", sorted(BAND_AWARE))
def test_every_exemption_states_a_reason(rel: str) -> None:
    """ "Recorded with its reason" is the half of AT.4's exit bar a bare path list would miss."""
    _, why = BAND_AWARE[rel]
    assert len(why) > 80, f"{rel}'s carve-out reason is too short to be one: {why!r}"


def test_no_carve_out_is_a_band_registry_any_more() -> None:
    """The debt AT.4 recorded and ADR 0092 named is paid, and this is what stops it returning.

    Five of the six entries were always a fact about the world (an LWIR-only sky depression, the
    solar/thermal crossover) or a field name already written into YAML on disk. The sixth was
    neither: ``BAND_CLASSES``, a spectroscopy table filed under camera names, five keys that a
    fifth band would have had to grow a sixth of. `AT.10` replaced it with one wavelength-ordered
    ladder that bands are *derived* from, so `atmosphere/layered.py` is down to the two Koschmieder
    identifiers -- the same photopic definition `atmosphere/extinction.py` carries.

    The check is that **no carve-out enumerates the band registry**: an entry whose offences
    include every band name is a second registry, whatever it is called.
    """
    for rel in BAND_AWARE:
        if rel == "config/bands.py":
            continue  # the registry itself: naming every band is its whole job
        offences = _offences_of(REPO / "src" / "irsim" / rel)
        named = {b for b in BAND_KEYS if any(f"'{b}" in o or f'"{b}' in o for o in offences)}
        assert named != set(BAND_KEYS), (
            f"{rel} names every band in the registry, which makes it a second one; "
            "derive from irsim.config.bands instead"
        )
    from irsim.atmosphere.layered import ATMOSPHERE_LADDER, classes_for

    # and the thing that replaced it is keyed by wavelength, not by camera.
    assert all(len(rung.edges_um) == 1 for rung in ATMOSPHERE_LADDER)
    for band in BAND_KEYS:
        assert classes_for(band), f"{band} derives no spectral classes from the ladder"


# -- the guard's ability to fail ---------------------------------------------------------------


def test_the_guard_would_catch_a_planted_band_name() -> None:
    """The guard is worth only as much as its ability to fail."""
    assert band_offences('if band == "lwir":\n    pass\n')
    assert band_offences("SWIR_CUTOFF = 1\n")
    assert band_offences("def mwir_gain():\n    return 1\n")
    assert not band_offences('"""A docstring may say lwir as often as it likes."""\n')
    assert not band_offences("broadband = 1\n"), "'broadband' must not split into a band name"


def test_the_guard_would_catch_the_anchor_band() -> None:
    """The fifth band that already exists, and that the old forbidden set could not see."""
    assert ANCHOR_BAND not in BAND_IDS
    assert ANCHOR_BAND in BAND_NAMES
    assert band_offences(f'anchor = "{ANCHOR_BAND}"\n')
    assert band_offences("VISIBLE_RANGE_UM = (0.4, 0.7)\n")


def test_the_edge_guard_would_catch_a_planted_band_edge() -> None:
    assert band_offences("lambda_cut_um = 13.5\n")
    assert band_offences("if wavelength_um > 7.5:\n    pass\n")
    # Not every 13.5 is a band edge -- only one used as a wavelength.
    assert not band_offences("gain_db = 13.5\n")


def test_the_illumination_switch_is_on_the_regime_not_the_band() -> None:
    """§5.2's "compile-time flag" as a runtime function of the regime alone (ADR 0063)."""
    from irsim.config.bands import DEFAULT_REGIME, enabled_illumination_terms

    for band in BAND_IDS:
        terms = enabled_illumination_terms(DEFAULT_REGIME[band])
        assert terms, band
    # Two bands that share a regime share a term set exactly -- there is no band-specific tail.
    assert enabled_illumination_terms("reflective") == enabled_illumination_terms("reflective")
    assert enabled_illumination_terms(DEFAULT_REGIME["nir"]) == enabled_illumination_terms(
        DEFAULT_REGIME["swir"]
    )


def test_the_weighting_temperature_is_derived_from_the_regime_not_the_band_name() -> None:
    """The five rows ``WEIGHT_T_REF_K`` held, and the drift that made the obvious rule wrong.

    ``irsim.atmosphere.layered`` kept a Planck weighting temperature keyed by band name. It is the
    registry restated -- except it had already drifted from it: MWIR was weighted at 300 K while
    ``DEFAULT_REGIME["mwir"]`` is ``"mixed"``. So the reading a reader reaches for first,
    "emissive keeps 300 K and everything else is solar", reproduces four rows and silently moves
    MWIR to 5800 K, rescaling every MWIR class share, its anchor solve, and therefore tau_MWIR at
    every range except the 200 m anchor. The rule that reproduces all five turns on *reflective*.
    """
    from irsim.atmosphere.layered import SOLAR_WEIGHT_T_K, weight_reference_temperature
    from irsim.config.bands import DEFAULT_REGIME, regime_for
    from irsim.radiometry.band_average import T_REF_K

    shipped = {"lwir": 300.0, "mwir": 300.0, "swir": 5800.0, "nir": 5800.0, "visible": 5800.0}
    for band, expected in shipped.items():
        assert weight_reference_temperature(band) == expected, band

    naive = {b: T_REF_K if DEFAULT_REGIME[b] == "emissive" else SOLAR_WEIGHT_T_K for b in BAND_IDS}
    assert naive["mwir"] != shipped["mwir"], (
        "the emissive-keyed rule no longer differs from the shipped table, so this test has "
        "stopped guarding the drift it was written for"
    )
    assert regime_for("mwir") == "mixed" and shipped["mwir"] == T_REF_K

    # And the anchor band, which no BandId names, still answers.
    assert regime_for(ANCHOR_BAND) == "reflective"
    assert weight_reference_temperature(ANCHOR_BAND) == SOLAR_WEIGHT_T_K
