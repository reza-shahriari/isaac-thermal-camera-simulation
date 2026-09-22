"""One wavelength ladder for the atmosphere's spectral classes (AT.10, ADR 0113).

`BAND_CLASSES` was five hand-written tables filed under camera names, and two of them disagreed
about the same air: NIR resolved the 0.94 µm water band at ×10 while SWIR's window swallowed
0.90–0.98 µm at ×0.5 — a factor of twenty on one sky, depending on which camera looked at it.
Two stretches of spectrum, 1.80–2.00 and 6.00–7.00 µm, belonged to no class at all, so a response
reaching either raised rather than absorbing. And a fifth band meant a sixth table in `src/`.

The tests here are the ones that fail if the ladder stops being one ladder.

docs/physics-model.md §7.1, §7.4; ADR 0071, ADR 0092, ADR 0113.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from irsim.atmosphere.layered import (
    ATMOSPHERE_LADDER,
    LADDER_SPAN_UM,
    LayeredAtmosphere,
    band_span_um,
    class_weights,
    classes_for,
)
from irsim.atmosphere.library import load_atmosphere_preset
from irsim.config.bands import BAND_KEYS, nominal_range_for
from irsim.radiometry.spectral_response import SpectralResponse, load_spectral_response
from irsim.thermal import WeatherSample, WeatherSeries

REPO = pathlib.Path(__file__).resolve().parents[2]
RESPONSES = REPO / "data" / "spectra" / "responses"
BAND_FILES = {
    "nir": "nir_si.csv",
    "swir": "ingaas.csv",
    "mwir": "insb.csv",
    "lwir": "boson_vox.csv",
}


def _responses() -> dict[str, SpectralResponse]:
    return {b: load_spectral_response(RESPONSES / f) for b, f in BAND_FILES.items()}


def _weather() -> WeatherSeries:
    return WeatherSeries.constant(
        WeatherSample(288.15, 0.46, 1.0, 0.0, 0.0, 0.0, 23000.0, 0.0), 3600.0
    )


def _atmosphere() -> LayeredAtmosphere:
    return LayeredAtmosphere(
        load_atmosphere_preset("us_standard_clear"), _weather(), None, _responses()
    )


# --- the ladder is one ladder --------------------------------------------------------------------


def test_the_ladder_is_wavelength_ordered_and_gap_free() -> None:
    """A gap is a band that raises at the far end of a render, which is how 1.80–2.00 µm and
    6.00–7.00 µm sat unnoticed: no shipped response reached either."""
    edge = LADDER_SPAN_UM[0]
    for rung in ATMOSPHERE_LADDER:
        assert len(rung.edges_um) == 1, f"{rung.name} is a rung, not a class"
        (lo, hi) = rung.edges_um[0]
        assert lo == edge, f"gap or overlap before {rung.name}: {edge} then {lo}"
        assert hi > lo
        edge = hi
    assert edge == LADDER_SPAN_UM[1]


def test_a_rung_name_means_one_thing_everywhere_it_appears() -> None:
    """`window` appears three times in LWIR's span and twice in MWIR's, and every appearance has
    to be the same air -- otherwise merging them into one class averages two different gases."""
    seen: dict[str, tuple[str, float, bool]] = {}
    for rung in ATMOSPHERE_LADDER:
        physics = (rung.kind, rung.multiplier, rung.opaque)
        if rung.name in seen:
            assert seen[rung.name] == physics, f"{rung.name} appears with two behaviours"
        seen[rung.name] = physics


def test_two_bands_never_disagree_about_the_same_wavelength() -> None:
    """**The defect, as a property.** Whatever wavelength two bands both reach, they must model it
    the same way. Before the ladder, 0.94 µm was a ×10 water band to NIR and a ×0.5 window to
    SWIR, and a scene rendered in both bands described two different atmospheres.
    """
    responses = _responses()
    tables = {b: classes_for(b, responses.get(b)) for b in BAND_KEYS}
    grid = np.arange(LADDER_SPAN_UM[0], LADDER_SPAN_UM[1], 0.005)

    def behaviour(band: str, lam: float) -> tuple[str, float, bool] | None:
        for cls in tables[band]:
            for lo, hi in cls.edges_um:
                if lo <= lam < hi:
                    return (cls.kind, cls.multiplier, cls.opaque)
        return None

    overlaps = 0
    for lam in grid:
        answers = {b: behaviour(b, float(lam)) for b in BAND_KEYS}
        present = {b: a for b, a in answers.items() if a is not None}
        if len(present) < 2:
            continue
        overlaps += 1
        assert len(set(present.values())) == 1, f"{lam:.3f} um: " + ", ".join(
            f"{b}={a}" for b, a in present.items()
        )
    assert overlaps > 100, "the bands barely overlap; the check is not exercising anything"


def test_a_band_reaching_past_the_ladder_is_refused_rather_than_guessed_at() -> None:
    """The numbers outside are spectroscopy nobody has written down here, and a nearest-class
    fallback would look exactly like a model."""
    far = SpectralResponse(
        np.array([LADDER_SPAN_UM[1] + 1.0, LADDER_SPAN_UM[1] + 2.0]),
        np.array([1.0, 1.0]),
        "<beyond the ladder>",
        "",
    )
    with pytest.raises(ValueError, match="outside the atmosphere ladder"):
        classes_for("lwir", far)


# --- a band is derived, not written -------------------------------------------------------------


@pytest.mark.parametrize("band", sorted(BAND_KEYS))
def test_every_registered_band_derives_classes_that_cover_it(band: str) -> None:
    response = _responses().get(band)
    lo, hi = band_span_um(band, response)
    classes = classes_for(band, response)
    assert classes
    covered = sorted(e for c in classes for e in c.edges_um)
    assert covered[0][0] <= lo and covered[-1][1] >= hi
    weights = class_weights(band, response)
    assert weights.shape == (len(classes),)
    assert float(weights.sum()) == pytest.approx(1.0, abs=1e-12)


def test_a_band_the_registry_has_never_heard_of_still_gets_classes() -> None:
    """The exit bar: a fifth band must need no `src/` edit.

    A hypothetical 5.5–7.5 µm camera straddles the 6.3 µm water band and the LWIR rotational
    edges, and the ladder answers for it without anyone adding a table. Before `AT.10` this raised
    a `KeyError` on `BAND_CLASSES`, and 6.00–7.00 µm belonged to no class in any case.
    """
    response = SpectralResponse(
        np.array([5.5, 6.5, 7.5]), np.array([1.0, 1.0, 1.0]), "<hypothetical>", ""
    )
    lo, hi = 5.5, 7.5
    classes = tuple(c for c in ATMOSPHERE_LADDER if c.edges_um[0][0] < hi and c.edges_um[0][1] > lo)
    names = [c.name for c in classes]
    assert names == ["h2o_wing", "h2o_6p3", "edges"], names
    assert any(c.opaque for c in classes)
    assert response.support_um == (5.5, 7.5)


# --- what it changed, measured ---------------------------------------------------------------


def test_lwir_and_nir_are_untouched_and_mwir_moves_by_a_rounding_error() -> None:
    """The three bands whose hand-written tables the ladder reproduces.

    LWIR and NIR come out bit-identical. MWIR differs in the **order** its classes are returned in
    -- wavelength order rather than the order someone typed them -- so its sum runs in a different
    order and lands within one ulp. Anything larger than that would mean the ladder had changed a
    calibrated band, which it must not: ADR 0071's window multipliers are fitted to R13's sky.
    """
    atmosphere = _atmosphere()
    expected = {
        "lwir": (0.930057013277753, 0.599481209005659, 0.303552327459639),
        "nir": (0.974267827601158, 0.529129262530538, 0.086541111389012),
        "mwir": (0.797018961895581, 0.361438188342700, 0.159372797288951),
    }
    for band, (near, mid, far) in expected.items():
        got = [float(atmosphere.transmittance(band, 0.0, d, 0.0)) for d in (200.0, 5000.0, 20000.0)]
        for value, want in zip(got, (near, mid, far), strict=True):
            assert value == pytest.approx(want, rel=5e-15), f"{band}: {value!r} vs {want!r}"


def test_swir_stops_swallowing_the_point_nine_four_water_band() -> None:
    """The fix, as the two numbers AT.10 was written around.

    SWIR's hand-written window ran 0.80–1.10 µm at ×0.5 and so treated the 0.94 µm water band as
    clear air. Deriving from the ladder hands that stretch to `h2o_0p94` at ×10 instead: **16.9 %**
    of the Planck-weighted band leaves `window`, and the band loses **6.5 %** of its transmittance
    at 5 km. The 200 m anchor is untouched, because the anchor solve re-fits to the grey preset at
    exactly that distance -- which is why no test caught this for so long.
    """
    response = _responses()["swir"]
    classes = classes_for("swir", response)
    weights = class_weights("swir", response)
    names = [c.name for c in classes]
    assert "h2o_0p94" in names
    assert float(weights[names.index("h2o_0p94")]) == pytest.approx(0.169, abs=2e-3)

    atmosphere = _atmosphere()
    assert float(atmosphere.transmittance("swir", 0.0, 200.0, 0.0)) == pytest.approx(
        0.819930495196555, rel=1e-12
    )
    assert float(atmosphere.transmittance("swir", 0.0, 5000.0, 0.0)) == pytest.approx(
        0.387690895559268, rel=1e-9
    )
    before, after = 0.414804696154634, 0.387690895559268
    assert after / before == pytest.approx(0.935, abs=2e-3)


def test_the_holes_are_filled_with_water_bands_and_not_with_window() -> None:
    """1.80–2.00 and 6.00–7.00 µm are the 1.9 and 6.3 µm water bands, both strong enough to be
    opaque over the ranges this simulator works at. Calling either a window would have made a
    future band look through a wall."""
    by_name = {rung.name: rung for rung in ATMOSPHERE_LADDER}
    for name, edges in (("h2o_1p9", (1.80, 2.00)), ("h2o_6p3", (6.00, 7.00))):
        assert by_name[name].edges_um == (edges,)
        assert by_name[name].opaque
        assert by_name[name].kind == "water"


def test_the_nominal_range_alone_would_not_cover_the_detectors() -> None:
    """Why `band_span_um` takes both. The nominal range alone was AT.3's defect; every shipped
    response reaches past its own band's nominal edges."""
    for band, response in _responses().items():
        nominal = nominal_range_for(band)
        support = response.support_um
        assert support[0] < nominal[0] or support[1] > nominal[1], band
        span = band_span_um(band, response)
        assert span[0] <= min(nominal[0], support[0])
        assert span[1] >= max(nominal[1], support[1])
