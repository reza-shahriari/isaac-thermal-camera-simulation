"""The add-on's Blender-free logic: names, the size check, area coverage, the material picker.

Each test is built on a case that has actually happened or that the owner raised: the Phantom 4
arriving in centimetres (ADR 0128), a real 300 m ship that must not be flagged (2026-09-28), a
Blender material shared by parts made of different things, a shell triangulated into one huge
triangle beside many small ones (the connection finder's sampling).
"""

import math
import re

import numpy as np
import pytest
from irsim_thermal import coverage, geometry, naming, picker, sizing

SAFE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


# --- names ---------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "safe"),
    [
        ("white plastic.001", "white_plastic_001"),
        ("Metal Matte", "Metal_Matte"),
        ("3M_tape", "_3M_tape"),
        ("[DJI]Glass-Black", "_DJI_Glass_Black"),
        ("already_fine", "already_fine"),
    ],
)
def test_a_name_comes_out_as_usd_writes_it(raw: str, safe: str) -> None:
    assert naming.safe_identifier(raw) == safe
    assert SAFE.match(naming.safe_identifier(raw))


def test_non_ascii_names_become_safe_identifiers() -> None:
    out = naming.safe_identifier("ماده فلزی")
    assert SAFE.match(out), out


def test_two_names_differing_only_by_case_cannot_both_reach_the_asset_map() -> None:
    # AssetMapping refuses keys that differ only by case, so one of them must move.
    renames = naming.unique_safe_names(["Metal", "metal"])
    final = {n: renames.get(n, n) for n in ["Metal", "metal"]}
    assert len({v.lower() for v in final.values()}) == 2
    assert all(SAFE.match(v) for v in final.values())


def test_a_rename_never_lands_on_a_name_another_material_already_has() -> None:
    renames = naming.unique_safe_names(["a.b"], taken=["a_b"])
    assert renames == {"a.b": "a_b_2"}


def test_renaming_reaches_a_fixed_point() -> None:
    names = ["white plastic.001", "white_plastic_001", "Metal", "metal", "ok", "3d"]
    renames = naming.unique_safe_names(names)
    after = [renames.get(n, n) for n in names]
    assert naming.unique_safe_names(after) == {}
    assert len({a.lower() for a in after}) == len(after)


def test_a_split_copy_keeps_the_thermal_name_whole_within_blenders_limit() -> None:
    long_origin = "x" * 80
    name = naming.split_name(long_origin, "carbon_fibre")
    assert name.endswith("__carbon_fibre")
    assert len(name.encode("utf-8")) <= naming.MAX_NAME_BYTES
    multibyte = naming.split_name("ž" * 40, "abs_plastic_white")
    assert multibyte.endswith("__abs_plastic_white")
    assert len(multibyte.encode("utf-8")) <= naming.MAX_NAME_BYTES
    multibyte.encode("utf-8").decode("utf-8")  # no character cut in half


@pytest.mark.parametrize(
    ("name", "default"),
    [
        ("Cube", True),
        ("Cube.003", True),
        ("Cylinder_12", True),
        ("GeometryNode_57", True),  # the Sketchfab Phantom 4, all 41 parts
        ("Box001", True),
        ("polySurface12", True),
        ("battery", False),
        ("motor_front_left", False),
        ("Boxer_engine", False),
    ],
)
def test_default_names_are_recognised(name: str, default: bool) -> None:
    assert naming.is_default_name(name) is default


# --- size ----------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ranges() -> dict[str, sizing.SizeRange]:
    return {r.key: r for r in sizing.load_ranges()}


def test_the_phantom_4_in_centimetres_is_recognised_as_centimetres(ranges) -> None:
    # ADR 0128: the FBX's extent is 41.05 x 46.37 x 20.67 source units, read as metres.
    d = sizing.diagnose(46.37, ranges["multirotor"])
    assert d.status == "unit"
    assert d.factor == pytest.approx(0.01)
    assert d.corrected_m == pytest.approx(0.4637)


def test_a_real_ship_is_never_flagged_for_being_big(ranges) -> None:
    for length in (15.0, 120.0, 300.0, 400.0):
        assert sizing.diagnose(length, ranges["ship"]).status == "within"


def test_the_same_300_m_as_a_car_is_a_units_mistake(ranges) -> None:
    d = sizing.diagnose(300.0, ranges["car"])
    assert d.status == "unit"
    assert d.corrected_m == pytest.approx(3.0)


def test_an_odd_size_no_unit_explains_is_reported_without_a_factor(ranges) -> None:
    # 8 m is outside 2.5-6.5 m, and no unit factor lands it inside: 8 x 0.3048 = 2.44 m.
    d = sizing.diagnose(8.0, ranges["car"])
    assert d.status == "outside"
    assert d.factor is None


def test_a_known_dimension_gives_the_exact_scale_and_names_the_unit() -> None:
    # Phantom 4: 20.67 source units tall, DJI publishes 196 mm. 5.6 % off, still centimetres.
    factor, why = sizing.scale_for_known(20.67, 0.196)
    assert factor == pytest.approx(0.196 / 20.67)
    assert why == "centimetres read as metres"
    factor, why = sizing.scale_for_known(1.0, 1.5)
    assert why is None


def test_no_two_unit_factors_are_within_the_matching_tolerance_of_each_other() -> None:
    # Otherwise a known dimension could be explained by two different units.
    factors = sorted(f for f, _ in sizing.UNIT_FACTORS)
    for a, b in zip(factors, factors[1:], strict=False):
        assert b / a > (1 + sizing.UNIT_MATCH_TOLERANCE) ** 2


def test_every_size_range_is_well_formed(ranges) -> None:
    assert len(ranges) >= 10
    for r in ranges.values():
        assert 0 < r.min_m < r.max_m
        assert r.min_m < r.centre_m < r.max_m


@pytest.mark.parametrize(
    ("metres", "phrase"),
    [(1.8, "a person"), (0.46, "a small drone"), (300.0, "a large ship"), (4.2, "a car")],
)
def test_the_size_is_compared_with_something_familiar(metres: float, phrase: str) -> None:
    assert phrase in sizing.comparison(metres)


def test_comparison_picks_the_nearest_in_log_space() -> None:
    # 20 m sits between a bus (12) and an airliner's wingspan (40); log-nearest is the bus.
    assert math.log(20 / 12) < math.log(40 / 20)
    assert "a bus" in sizing.comparison(20.0)


# --- coverage ------------------------------------------------------------------------------------


def test_area_is_attributed_per_slot_and_a_face_past_the_last_slot_is_unassigned() -> None:
    stats = coverage.part_stats(
        "arm",
        np.array([1.0, 2.0, 3.0, 4.0]),
        np.array([0, 0, 1, 5]),
        ["white_plastic", "Metal_Matte"],
        ["abs_plastic_white", ""],
    )
    assert stats.total_area_m2 == pytest.approx(10.0)
    assert stats.by_thermal_m2 == {"abs_plastic_white": pytest.approx(3.0)}
    assert stats.unassigned_by_material_m2 == {
        "Metal_Matte": pytest.approx(3.0),
        "<no material>": pytest.approx(4.0),
    }
    assert stats.unassigned_area_m2 == pytest.approx(7.0)
    assert not stats.fully_assigned


def test_coverage_is_area_weighted_not_part_counted() -> None:
    wing = coverage.part_stats("wing", np.array([9.0]), np.array([0]), ["m"], ["carbon_fibre"])
    rivet = coverage.part_stats("rivet", np.array([1.0]), np.array([0]), ["r"], [None])
    s = coverage.summarize([wing, rivet], {"carbon_fibre": 0.9})
    assert s.coverage == pytest.approx(0.9)
    assert [p.name for p in s.unassigned_parts] == ["rivet"]
    assert not s.passes_gate  # 90 % < ADR 0047's 95 %


def test_unassigned_parts_are_listed_largest_gap_first() -> None:
    parts = [
        coverage.part_stats(n, np.array([a]), np.array([0]), ["m"], [None])
        for n, a in (("small", 1.0), ("big", 5.0), ("mid", 3.0))
    ]
    s = coverage.summarize(parts, {})
    assert [p.name for p in s.unassigned_parts] == ["big", "mid", "small"]


def test_mirror_like_assignments_are_flagged_below_the_limit_only() -> None:
    trim = coverage.part_stats("trim", np.array([1.0]), np.array([0]), ["c"], ["bare_aluminium"])
    edge = coverage.part_stats("edge", np.array([1.0]), np.array([0]), ["e"], ["at_limit"])
    s = coverage.summarize([trim, edge], {"bare_aluminium": 0.09, "at_limit": 0.2})
    assert s.mirror_parts == [("trim", "bare_aluminium", 0.09)]


def test_the_gate_is_the_pipelines_95_percent() -> None:
    a = coverage.part_stats("a", np.array([95.0, 5.0]), np.array([0, 1]), ["x", "y"], ["m", ""])
    assert coverage.summarize([a], {}).passes_gate
    b = coverage.part_stats("b", np.array([94.9, 5.1]), np.array([0, 1]), ["x", "y"], ["m", ""])
    assert not coverage.summarize([b], {}).passes_gate


# --- the connection finder's sampling --------------------------------------------------------


def unit_square_uneven() -> geometry.Surface:
    """The unit square as two triangles of 0.49 m² above a strip of 100 slivers (y < 0.02)."""
    xs = np.linspace(0.0, 1.0, 51)
    verts = [(x, 0.0, 0.0) for x in xs] + [(x, 0.02, 0.0) for x in xs]
    verts += [(0.0, 1.0, 0.0), (1.0, 1.0, 0.0)]  # 102, 103
    tris = []
    for i in range(50):
        tris += [(i, i + 1, 52 + i), (i, 52 + i, 51 + i)]
    tris += [(51, 101, 103), (51, 103, 102)]
    return geometry.Surface(np.array(verts, dtype=float), np.array(tris))


def test_the_uneven_square_really_is_a_unit_square() -> None:
    s = unit_square_uneven()
    areas, normals = s.areas_normals()
    assert s.area == pytest.approx(1.0)
    assert np.allclose(normals, [0.0, 0.0, 1.0])
    assert areas.max() / areas.min() > 1000  # two huge triangles beside many slivers


def test_samples_are_spread_by_area_not_by_triangle() -> None:
    """100 of the 102 triangles hold 2 % of the area between them, so they get 2 % of the points."""
    pts = geometry.sample(unit_square_uneven(), 20000, seed=7)
    assert pts.area_each * len(pts) == pytest.approx(1.0)
    in_strip = np.mean(pts.points[:, 1] < 0.02)
    assert in_strip == pytest.approx(0.02, abs=0.004)
    left_half = np.mean(pts.points[:, 0] < 0.5)
    assert left_half == pytest.approx(0.5, abs=0.015)
    assert np.all((pts.points >= -1e-12) & (pts.points <= 1.0 + 1e-12))


def test_the_same_seed_finds_the_same_points() -> None:
    a = geometry.sample(unit_square_uneven(), 100, seed=3)
    b = geometry.sample(unit_square_uneven(), 100, seed=3)
    assert np.array_equal(a.points, b.points)


def test_a_contact_is_counted_only_parallel_and_straight_across() -> None:
    up = np.array([[0.0, 0.0, 1.0]] * 5)
    other = np.array([[0, 0, -1], [0, 0, -1], [1, 0, 0], [0, 0, -1], [0, 0, 1]], dtype=float)
    offsets = np.array([[0, 0, 0.001], [0.001, 0, 0], [0, 0, 0.001], [0, 0, 0], [0, 0, 0.001]])
    found = np.array([True, True, True, True, False])
    # straight across a 1 mm gap: yes; the same distance sideways, to the edge of a box standing
    # beside it: no; a wall at right angles: no; exactly on the other surface: yes; out of
    # reach: no.
    assert geometry.touching(found, up, other, offsets, 0.002).tolist() == [
        True,
        False,
        False,
        True,
        False,
    ]


def test_default_gaps_follow_the_model_size_with_a_floor() -> None:
    assert geometry.default_gaps(0.46) == pytest.approx((9.2e-4, 0.046))  # the Phantom 4
    assert geometry.default_gaps(0.05)[0] == pytest.approx(5e-4)  # never below half a millimetre
    assert geometry.default_gaps(4.5)[1] == pytest.approx(0.45)  # a car: its bonnet gap and more


# --- the material picker (B13) -------------------------------------------------------------------


def _material(name, lwir, description="", surface="as_manufactured", spectral=False, error=""):
    from types import SimpleNamespace

    return SimpleNamespace(
        name=name,
        description=description,
        surface_treatment=surface,
        spectral=spectral,
        error=error,
        eps_nir=0.1,
        eps_swir=0.2,
        eps_mwir=lwir - 0.05,
        eps_lwir=lwir,
    )


LIBRARY = [
    _material("abs_plastic_white", 0.95, "White ABS, as on a drone shell"),
    _material("aluminium_polished", 0.04, "Mirror-finished aluminium", "polished", spectral=True),
    _material("bare_aluminium", 0.09, "Unpainted aluminium sheet", "oxidised"),
    _material("car_paint_black", 0.93, "Automotive paint, gloss black", "painted"),
    _material("carbon_fibre", 0.90, "Carbon-fibre reinforced polymer"),
    _material("broken_curve", 0.0, "A curve that failed to load", error="no data in band"),
]


def _names(keep):
    return [m.name for m, k in zip(LIBRARY, keep, strict=True) if k]


def test_search_finds_every_word_in_any_order() -> None:
    # the way a person types: part of a word, the words reversed, a description word
    assert _names(picker.shown(LIBRARY, query="polished al")) == ["aluminium_polished"]
    assert _names(picker.shown(LIBRARY, query="al polished")) == ["aluminium_polished"]
    assert _names(picker.shown(LIBRARY, query="alu")) == ["aluminium_polished", "bare_aluminium"]
    assert _names(picker.shown(LIBRARY, query="drone")) == ["abs_plastic_white"]
    assert _names(picker.shown(LIBRARY, query="PAINT black")) == ["car_paint_black"]
    assert _names(picker.shown(LIBRARY, query="carbonfibre")) == ["carbon_fibre"]
    # its surface; and a word matches from its start: not bare aluminium's "Unpainted"
    assert _names(picker.shown(LIBRARY, query="painted")) == ["car_paint_black"]
    assert sum(picker.shown(LIBRARY, query="  ")) == len(LIBRARY)
    assert not any(picker.shown(LIBRARY, query="titanium"))


def test_the_band_filter_finds_the_mirror_like_materials() -> None:
    # "what in the library is mirror-like in LWIR?": the case behind the Phantom 4's motors
    keep = picker.shown(LIBRARY, band="lwir", eps_max=0.2)
    assert _names(keep) == ["aluminium_polished", "bare_aluminium"]
    # a material whose band could not be evaluated is not passed off as a value in the range
    assert "broken_curve" not in _names(keep)
    assert _names(picker.shown(LIBRARY, band="lwir", eps_min=0.92)) == [
        "abs_plastic_white",
        "car_paint_black",
    ]
    # the band matters: in MWIR (0.05 lower here) car paint falls under 0.9
    assert "car_paint_black" not in _names(picker.shown(LIBRARY, band="mwir", eps_min=0.9))
    # without a range, the band does not hide anything
    assert sum(picker.shown(LIBRARY, band="nir")) == len(LIBRARY)


def test_favourites_and_curves_filters() -> None:
    fav = ["carbon_fibre", "abs_plastic_white"]
    assert _names(picker.shown(LIBRARY, favourites=fav, only_favourites=True)) == [
        "abs_plastic_white",
        "carbon_fibre",
    ]
    assert _names(picker.shown(LIBRARY, only_curves=True)) == ["aluminium_polished"]
    # filters combine
    assert not any(picker.shown(LIBRARY, favourites=fav, only_favourites=True, only_curves=True))


def test_favourites_come_first_then_the_chosen_order() -> None:
    names = [m.name for m in LIBRARY]
    by_name = [names[i] for i in picker.order(LIBRARY)]
    assert by_name == sorted(names)
    fav = ["carbon_fibre"]
    assert [names[i] for i in picker.order(LIBRARY, favourites=fav)][0] == "carbon_fibre"
    high = [names[i] for i in picker.order(LIBRARY, sort="EPS_HIGH")]
    assert high[:2] == ["abs_plastic_white", "car_paint_black"]
    assert high[-1] == "broken_curve"  # no value: last either way
    low = [names[i] for i in picker.order(LIBRARY, sort="EPS_LOW")]
    assert low[:2] == ["aluminium_polished", "bare_aluminium"]
    assert low[-1] == "broken_curve"
    # positions is the inverse UIList.filter_items wants
    seq = picker.order(LIBRARY, sort="EPS_HIGH")
    pos = picker.positions(seq)
    assert all(seq[pos[i]] == i for i in range(len(LIBRARY)))


def test_favourite_and_recent_lists() -> None:
    stored = picker.join_names(["carbon_fibre", "abs_plastic_white"])
    assert picker.parse_names(stored) == ["carbon_fibre", "abs_plastic_white"]
    assert picker.parse_names(" a, ,b,a ,") == ["a", "b"]  # blanks and repeats dropped
    assert picker.parse_names("") == []
    assert picker.toggle(["a", "b"], "a") == ["b"]
    assert picker.toggle(["b"], "a") == ["b", "a"]
    recent: list[str] = []
    for name in ["a", "b", "c", "a", "d", "e", "f", "g"]:
        recent = picker.remember(recent, name)
    assert recent == ["g", "f", "e", "d", "a", "c"]  # most recent first, one of each, at most 6
    assert len(recent) == picker.RECENT


def test_the_search_menu_line_says_the_band() -> None:
    assert picker.label(LIBRARY[0], "lwir") == "abs_plastic_white   ε 0.95 LWIR"
    assert "not available" in picker.label(LIBRARY[-1], "lwir")


# --- names irsim's configs use: typed any way, written lower-case (the owner, 2026-10-03) --------


@pytest.mark.parametrize(
    ("typed", "name"),
    [
        ("My Drone-2", "my_drone_2"),
        ("  DJI  Mavic 3 Pro ", "dji_mavic_3_pro"),
        ("Nylon__Black", "nylon_black"),
        ("already_fine", "already_fine"),
        ("Cube.001", "cube_001"),
        ("3DR Solo", "m_3dr_solo"),  # a config name starts with a letter
        ("ماده فلزی", ""),  # no transliteration guessed: the caller picks another name
        ("--", ""),
    ],
)
def test_a_typed_name_becomes_a_config_name(typed: str, name: str) -> None:
    assert naming.config_name(typed) == name
    assert name == "" or re.fullmatch(r"[a-z][a-z0-9_]*", name)
    assert naming.config_name(name) == name  # tidying a tidy name changes nothing


def test_a_long_name_is_cut_without_a_trailing_underscore() -> None:
    out = naming.config_name("a" * 63 + " b")
    assert len(out) <= 64 and not out.endswith("_")
