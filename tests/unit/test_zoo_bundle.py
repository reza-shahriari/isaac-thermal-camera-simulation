"""The Thermal Model Zoo bundle's pure rules: licence gate, part grouping, emissivity weighting.

`scripts/zoo_bundle.py` publishes a library asset to a public catalogue, so the two mistakes that
matter are publishing what may not be redistributed and stating a wrong emissivity. The Blender
passes are not testable here; these rules are, and each test fails if its rule is broken.
"""

from __future__ import annotations

import importlib.util
import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def zb():
    spec = importlib.util.spec_from_file_location("zoo_bundle", REPO / "scripts" / "zoo_bundle.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "slug,zoo", [("by", "CC-BY-4.0"), ("BY-SA", "CC-BY-SA-4.0"), ("cc0", "CC0-1.0")]
)
def test_redistributable_licences_map_to_the_zoo_ids(zb, slug, zoo):
    assert zb.zoo_licence(slug) == zoo


@pytest.mark.parametrize("slug", ["by-nc", "by-nd", "by-nc-sa", "standard", "editorial", ""])
def test_any_other_licence_stops_the_bundle(zb, slug):
    with pytest.raises(SystemExit, match="not one the zoo accepts"):
        zb.zoo_licence(slug)


def test_part_emissivity_is_area_weighted_not_a_plain_mean(zb):
    # 3 m2 of paint at 0.9 and 1 m2 of bare metal at 0.1: 0.7, where an unweighted mean says 0.5.
    eps = zb.area_weighted({"paint": 0.9, "metal": 0.1}, {"paint": 3.0, "metal": 1.0})
    assert eps == pytest.approx(0.7, abs=1e-12)


def test_a_part_without_area_is_an_error_not_a_zero(zb):
    with pytest.raises(ValueError):
        zb.area_weighted({"paint": 0.9}, {"paint": 0.0})


@pytest.mark.parametrize(
    "name,base",
    [
        ("motor_front_left", "motor"),
        ("arm_rear_right", "arm"),
        ("propeller_2", "propeller"),
        ("fuselage", "fuselage"),
        ("gimbal_camera", "gimbal_camera"),
        ("hull_port_bow", "hull"),
    ],
)
def test_side_and_index_tokens_are_stripped_from_part_names(zb, name, base):
    assert zb.part_base(name) == base


def test_mirror_parts_collapse_and_their_material_areas_add(zb):
    # One motor of four carries the status LED: still one row, and the LED's area is kept.
    rows = [
        {"name": f"motor_{s}", "areas": {"alu": 1.0}}
        for s in ("front_left", "front_right", "rear_left")
    ]
    rows.append({"name": "motor_rear_right", "areas": {"alu": 1.0, "led": 0.5}})
    rows.append({"name": "fuselage", "areas": {"pc": 2.0}})
    groups = {g["name"]: g for g in zb.group_parts(rows)}
    assert set(groups) == {"Motor (x4)", "Fuselage"}
    assert groups["Motor (x4)"]["areas"] == pytest.approx({"alu": 4.0, "led": 0.5})


def test_weakest_source_wins(zb):
    assert zb.weakest(["measured", "estimated", "reference"]) == "estimated"
    assert zb.weakest(["measured", "reference"]) == "reference"


def test_emissivity_scale_floor_sits_below_every_value(zb):
    assert zb.eps_scale_floor([0.88, 0.94]) == pytest.approx(0.7)
    assert zb.eps_scale_floor([0.03, 0.9]) == 0.0  # never below zero
    for vals in ([0.845, 0.95], [0.5], [0.11, 0.99]):
        assert zb.eps_scale_floor(vals) <= min(vals) - 0.1 + 1e-12


@pytest.mark.parametrize("box", [(500, 400, 1900, 1300), (0, 0, 2400, 300), (1000, 0, 1100, 1800)])
def test_crop_window_has_the_stage_aspect_and_holds_the_model(zb, box):
    size, aspect = (2400, 1800), 1200 / 820
    x0, y0, x1, y1 = zb.crop_window(box, size, aspect)
    assert (x1 - x0) / (y1 - y0) == pytest.approx(aspect, rel=0.01)
    assert x0 <= box[0] and y0 <= box[1] and x1 >= box[2] and y1 >= box[3]


def test_zoo_ids_are_url_slugs(zb):
    assert zb.zoo_id("dji_mini_3_pro") == "dji-mini-3-pro"
    assert zb.zoo_id("Liberty_Ship") == "liberty-ship"


def _lab(rgb):
    def lin(c):
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    r, g, b = (lin(c) for c in rgb)
    x = (0.4124 * r + 0.3576 * g + 0.1805 * b) / 0.9505
    y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    z = (0.0193 * r + 0.1192 * g + 0.9505 * b) / 1.089

    def f(t):
        return t ** (1 / 3) if t > 0.008856 else 7.787 * t + 16 / 116

    return (116 * f(y) - 16, 500 * (f(x) - f(y)), 200 * (f(y) - f(z)))


def test_every_part_gets_its_own_colour(zb):
    import colorsys
    import itertools
    import math

    sides = ("front_left", "front_right", "rear_left", "rear_right")
    kinds = ("propeller", "motor", "arm", "duct")
    names = [f"{k}_{s}" for k in kinds for s in sides]
    names += ["battery", "camera_lens", "chassis", "frame", "fuselage", "gimbal_camera"]
    lab = {n: _lab(c) for n, c in zb.part_colours(names).items()}  # the Avata 2's 22 parts
    worst = min(math.dist(lab[a], lab[b]) for a, b in itertools.combinations(names, 2))
    # CIELAB dE 10 is plainly visible. Hue per kind gave the four propellers dE ~10 shades of
    # one colour; a golden-angle hue per index put each propeller 20 deg from its duct.
    assert worst > 15
    # the pairs a reader must tell apart: the four copies of a kind, and the parts of one station
    hue = {n: colorsys.rgb_to_hsv(*c)[0] for n, c in zb.part_colours(names).items()}
    for a, b in itertools.combinations(names, 2):
        same_kind = zb.part_base(a) == zb.part_base(b)
        same_station = a.split("_", 1)[-1] == b.split("_", 1)[-1] and a.split("_", 1)[-1] in sides
        if same_kind or same_station:
            gap = abs(hue[a] - hue[b])
            assert min(gap, 1 - gap) * 360 > 45, (a, b)


def test_a_part_row_is_named_for_its_part(zb):
    assert zb.part_label("propeller_front_left") == "Propeller front left"
    assert zb.part_label("fuselage") == "Fuselage"


def test_hero_range_scales_with_the_larger_plan_side_not_the_height(zb):
    # The Mini 3 Pro's measured framing: 0.298 m wide at 0.95 m filled 913 of 1280 px.
    assert zb.hero_range_m([0.282, 0.298, 0.085]) == pytest.approx(0.954, abs=1e-3)
    # Twice the span, twice the range; a tall mast does not push the camera back.
    assert zb.hero_range_m([0.6, 0.2, 0.1]) == pytest.approx(2 * zb.hero_range_m([0.3, 0.1, 0.1]))
    assert zb.hero_range_m([0.3, 0.3, 5.0]) == zb.hero_range_m([0.3, 0.3, 0.1])
