"""A material's optical quantity in three forms -- a curve, per band, one grey value -- layered
under the camera's own response (ADR 0175, docs/physics-model.md §4.1, §12.3).

The tests that matter are the camera-dependent ones: a curve with a reststrahlen-like dip at
8-9.5 µm must read lower to a camera that sees more of the dip, and a material with no curve must
read the same to every camera in the band. Every expected value comes from an independent dense
trapezoid of R(λ)·B(λ, 300 K)·ε(λ), not from the library's own Simpson grid.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest
import yaml
from pydantic import ValidationError

from irsim.config.materials import MaterialConfig
from irsim.materials.library import CLOSURE_TOL, load_material
from irsim.radiometry.planck import spectral_radiance
from irsim.radiometry.spectral_response import SpectralResponse

T_REF = 300.0


def _material(optical: dict[str, object], name: str = "m") -> dict[str, object]:
    return {
        "schema_version": 2,
        "material": {
            "name": name,
            "source": "estimated",
            "surface_treatment": "as_manufactured",
            "thermal": {
                "density_kg_m3": 1000,
                "specific_heat_j_kgk": 1000,
                "conductivity_w_mk": 1,
                "thickness_m": 0.01,
                "solar_absorptivity": 0.5,
            },
            "optical": optical,
        },
    }


def _table(path: pathlib.Path, lam: np.ndarray, val: np.ndarray) -> None:
    rows = "\n".join(f"{a:.6f},{b:.9f}" for a, b in zip(lam, val, strict=True))
    path.write_text("# test curve\n" + rows + "\n")


def _load(tmp_path: pathlib.Path, optical: dict[str, object]):  # type: ignore[no-untyped-def]
    p = tmp_path / "m.yaml"
    p.write_text(yaml.safe_dump(_material(optical), sort_keys=False))
    return load_material(p, data_dir=tmp_path)


def _top_hat(lo: float, hi: float) -> SpectralResponse:
    return SpectralResponse(np.array([lo, hi]), np.array([1.0, 1.0]), f"<{lo}-{hi}>", "")


def _oracle(lo: float, hi: float, eps: object) -> float:
    """∫ B ε dλ / ∫ B dλ over a top-hat, on a 20 001-point trapezoid grid."""
    lam = np.linspace(lo, hi, 20001)
    w = spectral_radiance(lam, np.asarray(T_REF))
    e = np.asarray(eps(lam) if callable(eps) else np.full_like(lam, float(eps)))  # type: ignore[arg-type]
    return float(np.trapezoid(w * e, lam) / np.trapezoid(w, lam))


def _quartz_like(lam: np.ndarray) -> np.ndarray:
    """ε 0.95 with a reststrahlen-like dip to 0.70 across 8.0-9.5 µm (smooth 0.25 µm edges)."""
    dip = 0.5 * (np.tanh((lam - 8.0) / 0.08) - np.tanh((lam - 9.5) / 0.08))
    return 0.95 - 0.25 * dip


@pytest.fixture()
def quartz(tmp_path: pathlib.Path):  # type: ignore[no-untyped-def]
    lam = np.linspace(0.3, 16.0, 3141)
    _table(tmp_path / "quartz.csv", lam, _quartz_like(lam))
    return _load(tmp_path, {"spectral_emissivity": "quartz.csv"})


# -- the three forms, each alone -------------------------------------------------------------


def test_a_grey_value_alone_is_the_same_in_every_band_and_any_camera(
    tmp_path: pathlib.Path,
) -> None:
    m = _load(tmp_path, {"emissivity": 0.93})
    for band in ("nir", "swir", "mwir", "lwir", "lwir_custom"):
        props = m.band_properties(band)
        assert props.emissivity == 0.93 and props.curve_fraction == 0.0
        assert abs(props.emissivity + props.reflectance + props.transmittance - 1) < CLOSURE_TOL
    assert m.band_properties("lwir", _top_hat(6.0, 13.0)).emissivity == 0.93


def test_a_per_band_table_reads_its_band_whatever_the_camera(tmp_path: pathlib.Path) -> None:
    """No curve: a 6-13 and a 7.5-13.5 camera read the one LWIR number (the old behaviour)."""
    m = _load(tmp_path, {"emissivity_per_band": {"mwir": 0.80, "lwir": 0.90}})
    assert m.band_properties("lwir", _top_hat(6.0, 13.0)).emissivity == 0.90
    assert m.band_properties("lwir", _top_hat(7.5, 13.5)).emissivity == 0.90
    with pytest.raises(KeyError, match="no emissivity authored"):
        m.band_properties("swir")


def test_a_curve_is_averaged_under_the_camera_s_own_response(quartz) -> None:  # type: ignore[no-untyped-def]
    """The headline: the same curve is a different number to cameras with different ranges."""
    cameras = {(7.5, 13.5): None, (6.0, 13.0): None, (8.0, 9.5): None, (10.0, 13.0): None}
    for lo, hi in cameras:
        props = quartz.band_properties("lwir", _top_hat(lo, hi))
        cameras[(lo, hi)] = props.emissivity
        assert props.curve_fraction == 1.0
        assert props.emissivity == pytest.approx(_oracle(lo, hi, _quartz_like), abs=2e-4)
    # More of the dip in the band -> lower ε; none of it -> the 0.95 shoulder.
    assert cameras[(8.0, 9.5)] < cameras[(6.0, 13.0)] < cameras[(10.0, 13.0)]
    assert cameras[(10.0, 13.0)] == pytest.approx(0.95, abs=1e-4)
    assert abs(cameras[(7.5, 13.5)] - cameras[(6.0, 13.0)]) > 5e-3


# -- layering ------------------------------------------------------------------------------


def test_the_band_value_fills_where_the_curve_has_no_data(tmp_path: pathlib.Path) -> None:
    """SLUM's shape: a long-wave curve from 8 µm, a camera from 7. The 7-8 µm slice takes the
    band's value, and ``curve_fraction`` is that slice's Planck share, measured independently."""
    lam = np.linspace(8.0, 14.0, 601)
    _table(tmp_path / "lw.csv", lam, np.full_like(lam, 0.95))
    m = _load(tmp_path, {"spectral_emissivity": "lw.csv", "emissivity_per_band": {"lwir": 0.50}})
    props = m.band_properties("lwir", _top_hat(7.0, 14.0))

    def piecewise(x: np.ndarray) -> np.ndarray:
        return np.where(x < 8.0, 0.50, 0.95)

    # The fill boundary is a step in ε, and Simpson across a step on the library's 0.01 µm grid
    # errs by O(jump · dλ): 2.2e-4 here for a 0.45 jump, ~15 mK apparent at 300 K.
    assert props.emissivity == pytest.approx(_oracle(7.0, 14.0, piecewise), abs=5e-4)
    lam_o = np.linspace(7.0, 14.0, 20001)
    w = spectral_radiance(lam_o, np.asarray(T_REF))
    share = float(np.trapezoid(w * (lam_o >= 8.0), lam_o) / np.trapezoid(w, lam_o))
    assert props.curve_fraction == pytest.approx(share, abs=2e-3)
    # A camera inside the curve never touches the fill.
    inside = m.band_properties("lwir", _top_hat(8.5, 13.0))
    assert inside.emissivity == pytest.approx(0.95, abs=1e-9) and inside.curve_fraction == 1.0


def test_the_grey_value_fills_when_the_band_has_no_value(tmp_path: pathlib.Path) -> None:
    lam = np.linspace(8.0, 14.0, 601)
    _table(tmp_path / "lw.csv", lam, np.full_like(lam, 0.95))
    m = _load(tmp_path, {"spectral_emissivity": "lw.csv", "emissivity": 0.60})
    got = m.band_properties("lwir", _top_hat(6.0, 13.0)).emissivity
    oracle = _oracle(6.0, 13.0, lambda x: np.where(x < 8, 0.6, 0.95))
    assert got == pytest.approx(oracle, abs=5e-4)  # a step again, as above
    assert m.band_properties("mwir").emissivity == pytest.approx(0.60, abs=1e-12)


def test_a_segmented_curve_joins_short_wave_reflectance_to_long_wave_emission(
    tmp_path: pathlib.Path,
) -> None:
    """ε = 1 − ρ on the reflectance segment (opaque), the emission segment as it stands, and a
    per-band MWIR value in the hole between them."""
    sw = np.linspace(0.35, 2.5, 216)
    lw = np.linspace(8.0, 14.0, 601)
    _table(tmp_path / "sw.csv", sw, np.full_like(sw, 0.20))
    _table(tmp_path / "lw.csv", lw, np.full_like(lw, 0.91))
    m = _load(
        tmp_path,
        {
            "spectral_emissivity": [{"file": "sw.csv", "quantity": "reflectance"}, "lw.csv"],
            "emissivity_per_band": {"mwir": 0.85, "lwir": 0.90},
        },
    )
    assert m.band_properties("nir").emissivity == pytest.approx(0.80, abs=1e-12)
    assert m.band_properties("swir").emissivity == pytest.approx(0.80, abs=1e-12)
    assert m.band_properties("mwir").emissivity == pytest.approx(0.85, abs=1e-12)
    lwir = m.band_properties("lwir")  # nominal 7.5-13.5: 7.5-8 from the 0.90 fill
    assert 0.90 < lwir.emissivity < 0.91 and 0.9 < lwir.curve_fraction < 1.0
    assert m.spectrum is None  # not one plain file
    assert len(m.content_hash()) == 64


# -- refusals ------------------------------------------------------------------------------


def test_a_gap_with_nothing_to_fill_it_is_refused(tmp_path: pathlib.Path) -> None:
    lam = np.linspace(8.0, 14.0, 601)
    _table(tmp_path / "lw.csv", lam, np.full_like(lam, 0.95))
    m = _load(tmp_path, {"spectral_emissivity": "lw.csv"})
    with pytest.raises(ValueError, match="covers 8-14"):
        m.band_properties("lwir", _top_hat(7.0, 13.0))


def test_a_band_value_the_curve_already_covers_is_authored_twice(quartz, tmp_path) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ValueError, match="authored twice"):
        _load(tmp_path, {"spectral_emissivity": "quartz.csv", "emissivity_per_band": {"mwir": 0.9}})
    # ...while a band key with no nominal range is allowed: nothing says the curve covers it.
    m = _load(tmp_path, {"spectral_emissivity": "quartz.csv", "emissivity_per_band": {"x": 0.9}})
    assert m.band_properties("x").emissivity == 0.9


def test_overlapping_segments_and_a_transmitting_complement_are_refused(
    tmp_path: pathlib.Path,
) -> None:
    a = np.linspace(0.3, 3.0, 28)
    b = np.linspace(2.0, 14.0, 121)
    _table(tmp_path / "a.csv", a, np.full_like(a, 0.2))
    _table(tmp_path / "b.csv", b, np.full_like(b, 0.9))
    with pytest.raises(ValueError, match="overlap"):
        _load(tmp_path, {"spectral_emissivity": ["a.csv", "b.csv"]})
    with pytest.raises(ValueError, match="opaque complement"):
        _load(
            tmp_path,
            {
                "spectral_emissivity": [{"file": "a.csv", "quantity": "reflectance"}],
                "emissivity_per_band": {"lwir": 0.9},
                "transmittance_per_band": {"swir": 0.5},
            },
        )


def test_both_quantities_or_neither_is_refused_by_the_schema() -> None:
    with pytest.raises(ValidationError, match="exactly one"):
        MaterialConfig.model_validate(
            _material({"emissivity": 0.9, "reflectance_per_band": {"x": 0.1}})
        )
    with pytest.raises(ValidationError, match="exactly one"):
        MaterialConfig.model_validate(_material({"roughness_per_band": {"lwir": 0.1}}))
    with pytest.raises(ValidationError, match="Kirchhoff"):
        MaterialConfig.model_validate(
            _material({"emissivity": 0.9, "transmittance_per_band": {"lwir": 0.2}})
        )
    # The table's own value is the one closure is checked against, not the grey one.
    MaterialConfig.model_validate(
        _material(
            {
                "emissivity": 0.9,
                "emissivity_per_band": {"lwir": 0.7},
                "transmittance_per_band": {"lwir": 0.2},
            }
        )
    )
