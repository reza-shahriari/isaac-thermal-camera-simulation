"""OC.1 -- defocus geometry (docs/physics-model.md §8.3).

The oracle throughout is the thin-lens construction itself: image distances from 1/v = 1/f − 1/s
and similar triangles on the exit pupil. The module ships the algebraically simplified closed form,
so testing it against the construction catches a wrong simplification, which is the failure mode an
"it returns a number" test would sail past.
"""

from __future__ import annotations

import numpy as np
import pytest

from irsim.optics.defocus import (
    blur_circle_um,
    defocus_w020_um,
    defocus_waves,
    depth_of_field_m,
    geometric_regime_blur_um,
    hyperfocal_distance_m,
)

#: FLIR Boson 640, configs/sensors/flir_boson_640_lwir.yaml.
BOSON = {"focal_length_mm": 14.0, "f_number": 1.0}
PITCH_UM = 12.0
LWIR_UM = 10.5


def _blur_by_construction(s_m: float, sf_m: float, f_mm: float, n: float) -> float:
    """Blur circle (µm) from image distances and similar triangles -- no algebra shared with the
    module under test."""
    f_m = f_mm * 1e-3
    v_s = f_m * s_m / (s_m - f_m)  # where the object's image forms
    v_f = f_m * sf_m / (sf_m - f_m)  # where the sensor sits
    diameter_m = f_m / n  # exit pupil
    return float(diameter_m * abs(v_s - v_f) / v_s * 1e6)


@pytest.mark.parametrize("s_m", [2.0, 5.0, 20.0, 100.0, 1000.0])
@pytest.mark.parametrize("sf_m", [3.0, 10.0, 50.0])
def test_the_closed_form_is_the_thin_lens_construction(s_m: float, sf_m: float) -> None:
    got = float(blur_circle_um(s_m, focus_distance_m=sf_m, **BOSON))
    want = _blur_by_construction(s_m, sf_m, BOSON["focal_length_mm"], BOSON["f_number"])
    assert got == pytest.approx(want, abs=1e-9), "closed form diverges from the construction"


def test_focus_is_the_only_place_the_blur_vanishes_and_it_grows_either_side() -> None:
    sf = 20.0
    assert float(blur_circle_um(sf, focus_distance_m=sf, **BOSON)) == 0.0
    nearer = blur_circle_um([19.0, 15.0, 10.0, 5.0], focus_distance_m=sf, **BOSON)
    farther = blur_circle_um([21.0, 30.0, 60.0, 500.0], focus_distance_m=sf, **BOSON)
    assert np.all(np.diff(nearer) > 0.0), "blur must grow as the object comes nearer than focus"
    assert np.all(np.diff(farther) > 0.0), "blur must grow as the object goes beyond focus"


def test_the_infinity_branch_is_the_limit_of_the_finite_one() -> None:
    """Two code paths, one physics. The finite form approaches the infinite one as 1 − s/s_f, so
    the test asserts that *rate* rather than a bare closeness -- a branch that converged to the
    wrong constant, or at the wrong order, fails."""
    s = np.array([5.0, 50.0, 500.0])
    inf = blur_circle_um(s, focus_distance_m=None, **BOSON)
    for sf in (1e5, 1e7, 1e9):
        rel = np.abs(blur_circle_um(s, focus_distance_m=sf, **BOSON) - inf) / inf
        assert np.all(rel <= 2.0 * s / sf), f"convergence slower than s/s_f at s_f = {sf:g}"
        assert np.all(rel >= 0.25 * s / sf), f"convergence faster than s/s_f at s_f = {sf:g}"


def test_hyperfocal_is_the_distance_that_holds_infinity_inside_the_circle() -> None:
    """The definition, tested as a definition: focus at H and a star blurs to exactly one pitch."""
    h_m = hyperfocal_distance_m(coc_um=PITCH_UM, **BOSON)
    assert h_m == pytest.approx(16.34733, abs=1e-4)  # (14²/12e-3 + 14) mm
    at_infinity = float(blur_circle_um(1e12, focus_distance_m=h_m, **BOSON))
    assert at_infinity == pytest.approx(PITCH_UM, rel=1e-6)


def test_the_depth_of_field_limits_are_where_the_blur_reaches_the_circle() -> None:
    """The strong form: evaluate the blur AT the returned limits and it must equal the criterion."""
    for sf_m in (5.0, 20.0, 100.0):
        near, far = depth_of_field_m(coc_um=PITCH_UM, focus_distance_m=sf_m, **BOSON)
        assert float(blur_circle_um(near, focus_distance_m=sf_m, **BOSON)) == pytest.approx(
            PITCH_UM, rel=1e-6
        )
        if np.isfinite(far):
            assert float(blur_circle_um(far, focus_distance_m=sf_m, **BOSON)) == pytest.approx(
                PITCH_UM, rel=1e-6
            )


def test_focusing_at_hyperfocal_reaches_from_half_of_it_to_infinity() -> None:
    h_m = hyperfocal_distance_m(coc_um=PITCH_UM, **BOSON)
    near, far = depth_of_field_m(coc_um=PITCH_UM, focus_distance_m=h_m, **BOSON)
    assert near == pytest.approx(h_m / 2.0, rel=1e-6)
    assert far == float("inf")
    # and focusing at infinity starts being acceptable at H, not at H/2
    near_inf, far_inf = depth_of_field_m(coc_um=PITCH_UM, focus_distance_m=None, **BOSON)
    assert near_inf == pytest.approx(h_m, rel=1e-12)
    assert far_inf == float("inf")


def test_w020_is_the_blur_circle_over_eight_f_numbers() -> None:
    assert float(defocus_w020_um(80.0, 2.0)) == pytest.approx(5.0)
    # and the waves conversion is that over lambda
    assert float(defocus_waves(80.0, 2.0, 5.0)) == pytest.approx(1.0)


def test_the_geometric_threshold_is_two_waves_of_path_error() -> None:
    """16 F λ is only meaningful if it really is W020 = 2λ -- check the two agree."""
    for n, lam in ((1.0, 10.5), (2.0, 4.0), (1.4, 1.3)):
        c = geometric_regime_blur_um(n, lam)
        assert float(defocus_waves(c, n, lam)) == pytest.approx(2.0, rel=1e-12)
    # the headline number the roadmap and README quote for a Boson at F/1.0
    assert geometric_regime_blur_um(1.0, LWIR_UM) == pytest.approx(168.0)
    assert geometric_regime_blur_um(1.0, LWIR_UM) / PITCH_UM == pytest.approx(14.0)


def test_the_aerial_lane_is_diffraction_limited_and_the_near_field_is_not() -> None:
    """Why focus was safe to omit until now, as a number. Rayleigh calls a lens diffraction
    limited below a quarter wave, and a Boson focused at infinity sits at 0.02 waves at 100 m --
    which is why the aerial lane never needed this. It reaches the quarter wave at about 10 m and
    is clearly past it by 5 m, which is why the ground and close maritime lanes do."""
    waves = defocus_waves(
        blur_circle_um([5.0, 10.0, 100.0], focus_distance_m=None, **BOSON),
        BOSON["f_number"],
        LWIR_UM,
    )
    assert waves[2] == pytest.approx(0.02, abs=0.005), "100 m must be well inside the quarter wave"
    assert waves[1] == pytest.approx(0.23, abs=0.01), "10 m sits right at Rayleigh's boundary"
    assert waves[0] == pytest.approx(0.47, abs=0.01), "5 m must be clearly past it"
    assert waves[0] > 0.25 > waves[2]


def test_a_sky_pixel_cannot_be_mistaken_for_the_nearest_thing_in_the_scene() -> None:
    """G-buffer sky carries distance_m = 0; accepting it would defocus the sky hardest of all."""
    with pytest.raises(ValueError, match="sky pixels"):
        blur_circle_um(0.0, focus_distance_m=50.0, **BOSON)
    with pytest.raises(ValueError, match="sky pixels"):
        blur_circle_um([100.0, 0.0], focus_distance_m=None, **BOSON)
    with pytest.raises(ValueError, match="sky pixels"):
        blur_circle_um(0.010, focus_distance_m=None, **BOSON)  # inside the focal length


@pytest.mark.parametrize(
    ("bad", "kwargs"),
    [
        ("focal length", {"focal_length_mm": 0.0, "f_number": 1.0}),
        ("focal length", {"focal_length_mm": 14.0, "f_number": -1.0}),
    ],
)
def test_impossible_optics_are_refused(bad: str, kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError, match=bad):
        blur_circle_um(10.0, **kwargs)
