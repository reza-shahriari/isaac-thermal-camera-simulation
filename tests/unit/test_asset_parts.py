"""Part decomposition of an imported asset: `irsim.io.asset_parts`.

The fixtures here are a synthetic quadcopter, not the Phantom 4 archive, because the archive is
generated and not in git. The numbers are the Phantom 4's measured ones -- rotor stations at
r = 185 mm, a 88 x 83 x 28 mm battery block at the body centre, propeller blades at z = 80 mm --
so a selector that works here is authored against the real geometry's scale.

Roadmap `AI.5`; ADR 0128.
"""

from __future__ import annotations

import math

import pytest

from irsim.io.asset_parts import Component, PartsConfig, PartSelector, PartSpec, assign_parts

# The Phantom 4's own centre: the asset does not sit at its origin, and every radius in this file
# is relative to this. A decomposition that ignored it would place every station at r ~ 0.49.
CENTRE = (-0.0078, 0.4915, 0.0)

# Measured bearings of the four rotor stations, 90 deg apart at r = 185 mm.
STATIONS = {
    "front_left": (0.044, 0.314),
    "front_right": (-0.185, 0.439),
    "rear_left": (0.170, 0.544),
    "rear_right": (-0.060, 0.669),
}


def _box(index, centre, size, faces, area, material):
    """A component with an axis-aligned box's bounds about ``centre``."""
    half = tuple(s / 2.0 for s in size)
    lo = tuple(c - h for c, h in zip(centre, half, strict=True))
    hi = tuple(c + h for c, h in zip(centre, half, strict=True))
    return Component(
        index=index,
        faces=faces,
        area_m2=area,
        centroid=tuple(centre),
        lo=lo,
        hi=hi,
        material_name=material,
    )


def quad_components():
    """A synthetic quadcopter at the Phantom 4's measured dimensions."""
    out = []
    i = 0
    for _, (sx, sy) in STATIONS.items():
        # motor can: 27 x 27 x 16 mm of matte metal at z = 64 mm
        out.append(_box(i, (sx, sy, 0.064), (0.027, 0.027, 0.016), 9216, 0.00175, "Metal_Matte"))
        i += 1
        # two propeller blades, offset +/-12 mm in y, at z = 80 mm, thin
        for dy in (-0.012, 0.012):
            out.append(
                _box(i, (sx, sy + dy, 0.080), (0.055, 0.046, 0.011), 2000, 0.00078, "white_plastic")
            )
            i += 1
    # the battery: a 401-face block inside the fuselage, at the body centre
    out.append(_box(i, (-0.011, 0.496, -0.005), (0.088, 0.083, 0.028), 401, 0.00570, "Black_matte"))
    i += 1
    # the shell: one huge component spanning the arms, above and below
    out.append(
        _box(i, (-0.006, 0.494, 0.039), (0.404, 0.404, 0.041), 72894, 0.04561, "white_plastic")
    )
    i += 1
    out.append(
        _box(i, (-0.001, 0.491, 0.023), (0.404, 0.403, 0.054), 65454, 0.02374, "white_plastic")
    )
    i += 1
    # landing gear: a 95 mm strut and a 10 mm skid foot, same plastic, same radius
    out.append(
        _box(i, (0.099, 0.523, -0.057), (0.042, 0.076, 0.095), 900, 0.00091, "white_plastic")
    )
    i += 1
    out.append(
        _box(i, (0.068, 0.480, -0.104), (0.075, 0.135, 0.010), 563, 0.00110, "white_plastic")
    )
    return out


def quad_config(parts):
    return PartsConfig(centre=CENTRE, parts=parts, coverage_threshold=0.95)


def rotor_parts():
    """Per-rotor parts: four propellers and four motors, each selected at its own station."""
    specs = []
    for name, (sx, sy) in STATIONS.items():
        specs.append(
            PartSpec(
                name=f"propeller_{name}",
                target="propeller",
                select=PartSelector(near_xy=(sx, sy), within_m=0.06, z_min_m=0.070),
            )
        )
        specs.append(
            PartSpec(
                name=f"motor_{name}",
                target="motor",
                select=PartSelector(near_xy=(sx, sy), within_m=0.06, materials=["metal_matte"]),
            )
        )
    return specs


def test_each_rotor_station_is_its_own_part():
    """Four rotors must be four parts, not one.

    A lumped `motor` node gives four motors one duty history. This is the test that would fail if
    the station selectors overlapped -- `within_m` is 60 mm and the stations are 262 mm apart, so
    a selector that leaked would claim a neighbour's blades.
    """
    assignments, report = assign_parts(quad_components(), quad_config(rotor_parts()))
    by_part = {}
    for a in assignments:
        by_part.setdefault(a.part, []).append(a.index)

    for name in STATIONS:
        assert len(by_part[f"propeller_{name}"]) == 2, f"{name}: expected two blades"
        assert len(by_part[f"motor_{name}"]) == 1, f"{name}: expected one motor can"
    assert report.empty_parts == ()


def test_stations_are_far_enough_apart_that_the_selectors_cannot_overlap():
    """The geometry behind the previous test: a 60 mm radius on stations 262 mm apart."""
    pts = list(STATIONS.values())
    gaps = [math.dist(a, b) for idx, a in enumerate(pts) for b in pts[idx + 1 :]]
    assert min(gaps) > 2 * 0.06, "station selectors would overlap at this radius"


def test_the_battery_is_found_inside_the_shell_that_encloses_it():
    """The defect this module exists for.

    The battery block sits at the body centre, wholly inside the shell's bounds. Declaration order
    is what separates them: the battery is authored before the shell, so it is claimed first.
    """
    parts = [
        PartSpec(
            name="battery",
            target="battery",
            select=PartSelector(materials=["black_matte"], r_max_m=0.05, faces_max=2000),
        ),
        PartSpec(name="airframe", target="airframe", select=PartSelector()),
    ]
    assignments, report = assign_parts(quad_components(), quad_config(parts))
    battery = [a for a in assignments if a.part == "battery"]
    assert len(battery) == 1
    assert battery[0].target == "battery"
    assert report.faces_by_part["battery"] == 401
    # Position alone cannot separate them: the battery's centroid lies inside the shell's own
    # footprint, and both sit within 50 mm of the asset centre. Only the material and the face
    # count -- a 401-face block against a 72,894-face shell -- distinguish the two.
    comps = {c.index: c for c in quad_components()}
    shell = max(comps.values(), key=lambda c: c.area_m2)
    b = comps[battery[0].index]
    assert shell.lo[0] <= b.centroid[0] <= shell.hi[0]
    assert shell.lo[1] <= b.centroid[1] <= shell.hi[1]
    assert b.radius_m(CENTRE) < 0.05 and shell.radius_m(CENTRE) < 0.05


def test_a_part_that_matches_nothing_is_reported_rather_than_silently_empty():
    """The Phantom 4 scene declared a battery target and bound no geometry to it for months.

    An empty part must fail the report even when coverage is perfect, because the whole asset can
    be covered while the one part the scene's heat source needs is missing.
    """
    parts = [
        PartSpec(name="fuel_tank", target="fuel", select=PartSelector(materials=["kerosene"])),
        PartSpec(name="airframe", target="airframe", select=PartSelector()),
    ]
    _, report = assign_parts(quad_components(), quad_config(parts))
    assert report.coverage == pytest.approx(1.0)
    assert report.empty_parts == ("fuel_tank",)
    assert not report.passed, "perfect coverage must not excuse a part with no geometry"


def test_coverage_is_area_weighted_not_component_counted():
    """Most components are specks; area is what a radiometric image cares about.

    Claiming the twelve small rotor components and missing both shells is 12 of 17 components --
    71 % -- and under a fifth of the area. A count-based coverage would call that a pass at any
    gate below 70 %; the area-weighted one correctly refuses it.
    """
    assignments, report = assign_parts(quad_components(), quad_config(rotor_parts()))
    covered = sum(report.area_by_part.values())
    claimed = sum(1 for a in assignments if a.assigned)
    assert claimed / len(assignments) > 0.70
    assert covered / report.total_area_m2 < 0.20
    assert report.coverage == pytest.approx(covered / report.total_area_m2)
    assert not report.passed


def test_radius_is_measured_from_the_asset_centre_not_the_origin():
    """The Phantom 4 sits at y = +0.49 m in its own frame.

    Measured from the origin every station is at r ~ 0.49 and indistinguishable; measured from the
    asset's centre they are all at 185 mm. A selector banding on radius depends on this entirely.
    """
    comps = quad_components()
    cans = [c for c in comps if c.material_name == "Metal_Matte"]
    assert len(cans) == 4
    for c in cans:
        assert c.radius_m(CENTRE) == pytest.approx(0.185, abs=0.002)
        assert c.radius_m((0.0, 0.0, 0.0)) > 0.30


def test_extent_separates_a_gear_strut_from_the_skid_foot_it_stands_on():
    """Same material, same radius, same region -- only the shell's own height tells them apart."""
    parts = [
        PartSpec(
            name="landing_gear_strut",
            target="airframe",
            select=PartSelector(z_max_m=0.0, extent_z_min_m=0.05),
        ),
        PartSpec(
            name="landing_gear_foot",
            target="airframe",
            select=PartSelector(z_max_m=0.0, extent_z_max_m=0.02),
        ),
        PartSpec(name="airframe", target="airframe", select=PartSelector()),
    ]
    assignments, _ = assign_parts(quad_components(), quad_config(parts))
    comps = {c.index: c for c in quad_components()}
    struts = [comps[a.index] for a in assignments if a.part == "landing_gear_strut"]
    feet = [comps[a.index] for a in assignments if a.part == "landing_gear_foot"]
    assert [round(c.extent[2], 3) for c in struts] == [0.095]
    assert [round(c.extent[2], 3) for c in feet] == [0.010]


def test_first_match_wins_in_declaration_order():
    """Reversing the order moves the battery into the catch-all, and nothing else changes."""
    battery = PartSpec(
        name="battery",
        target="battery",
        select=PartSelector(materials=["black_matte"], r_max_m=0.05, faces_max=2000),
    )
    catchall = PartSpec(name="airframe", target="airframe", select=PartSelector())
    _, specific_first = assign_parts(quad_components(), quad_config([battery, catchall]))
    _, catchall_first = assign_parts(quad_components(), quad_config([catchall, battery]))
    assert specific_first.faces_by_part["battery"] == 401
    assert catchall_first.faces_by_part["battery"] == 0
    assert catchall_first.empty_parts == ("battery",)


def test_unassigned_geometry_is_returned_not_dropped():
    parts = [PartSpec(name="motors", select=PartSelector(materials=["metal_matte"]))]
    assignments, report = assign_parts(quad_components(), quad_config(parts))
    unassigned = [a for a in assignments if not a.assigned]
    assert len(unassigned) == len(quad_components()) - 4
    assert report.unassigned_faces == sum(a.faces for a in unassigned)
    assert report.total_faces == sum(a.faces for a in assignments)


def test_greedy_part_is_flagged_without_failing_the_report():
    parts = [PartSpec(name="everything", target="airframe", select=PartSelector())]
    _, report = assign_parts(quad_components(), quad_config(parts))
    assert report.passed, "a single catch-all still covers the asset"
    assert report.greedy_parts(0.60) == ("everything",)


def test_near_xy_without_a_radius_is_an_error_not_a_silent_pass():
    sel = PartSelector(near_xy=(0.0, 0.0))
    with pytest.raises(ValueError, match="together"):
        sel.accepts(quad_components()[0], CENTRE)


def test_duplicate_part_names_are_rejected():
    with pytest.raises(ValueError, match="duplicate part name"):
        PartsConfig(
            parts=[PartSpec(name="motor"), PartSpec(name="Motor")],
        )


def test_degenerate_components_are_rejected():
    with pytest.raises(ValueError, match="no faces"):
        Component(index=0, faces=0, area_m2=1.0, centroid=(0, 0, 0), lo=(0, 0, 0), hi=(1, 1, 1))
    with pytest.raises(ValueError, match="hi is below lo"):
        Component(index=0, faces=1, area_m2=1.0, centroid=(0, 0, 0), lo=(1, 1, 1), hi=(0, 0, 0))


def test_report_renders_every_named_part_and_the_remainder():
    parts = [
        PartSpec(name="motors", target="motor", select=PartSelector(materials=["metal_matte"])),
        PartSpec(name="ghost", target="ghost", select=PartSelector(materials=["unobtainium"])),
    ]
    _, report = assign_parts(quad_components(), quad_config(parts))
    text = report.render()
    assert "motors" in text
    assert "<unassigned>" in text
    assert "EMPTY: part 'ghost'" in text


# ------------------------------------------------------------------------------------------------
# The committed Phantom 4 decomposition. These need no archive: they check the authored config.
# ------------------------------------------------------------------------------------------------


def phantom4_parts():
    from irsim.materials.mapping import load_asset_mapping

    parts = load_asset_mapping("configs/assets/phantom4.yaml").parts
    assert parts is not None, "the Phantom 4 asset config must declare a part decomposition"
    return parts


def test_the_phantom4_declares_the_parts_its_scene_needs():
    """The `battery` heat source had no geometry. This is the test that keeps it bound."""
    parts = phantom4_parts()
    assert "battery" in parts.names
    assert "battery" in parts.targets
    assert {"propeller", "motor", "battery", "airframe"} <= parts.targets


def test_the_phantom4_has_four_of_each_rotor_part():
    parts = phantom4_parts()
    for prefix in ("propeller_", "motor_mount_"):
        assert sum(n.startswith(prefix) for n in parts.names) == 4, prefix
    motors = [n for n in parts.names if n.startswith("motor_") and "mount" not in n]
    assert len(motors) == 4
    assert {n.split("_", 1)[1] for n in motors} == set(STATIONS)


def test_the_phantom4_station_selectors_cannot_claim_each_other():
    """Four stations 262 mm apart, each claiming a 75 mm radius: no component is in two."""
    parts = phantom4_parts()
    discs = [
        (p.select.near_xy, p.select.within_m) for p in parts.parts if p.select.near_xy is not None
    ]
    assert len(discs) == 12
    for idx, (xy_a, r_a) in enumerate(discs):
        for xy_b, r_b in discs[idx + 1 :]:
            if xy_a == xy_b:
                continue  # same station: propeller, motor and mount deliberately overlap
            assert math.dist(xy_a, xy_b) > r_a + r_b, f"{xy_a} and {xy_b} overlap"


def test_the_phantom4_propeller_cut_clears_the_motor_under_it_at_every_station():
    """The asset is pitched 2.93 deg nose-down, so the stations sit at four different heights.

    A single global z cut would put the rear-right motor can (z = 0.064) above the front-left
    propeller blades (z = 0.062) and mis-assign both, which is exactly what an earlier revision
    did -- it lost nine tenths of the front-left propeller.

    The two selectors' height bands do overlap by a few millimetres at each station, and that is
    safe only because their **material lists are disjoint**: a propeller is white ABS and a motor
    is matte metal or copper. Both properties are asserted, because the height band alone does not
    separate them and the material list alone would claim the propeller's whole column.
    """
    parts = phantom4_parts()
    by_name = {p.name: p for p in parts.parts}
    floors = []
    for station in STATIONS:
        prop = by_name[f"propeller_{station}"].select
        motor = by_name[f"motor_{station}"].select
        assert prop.z_min_m is not None and motor.z_max_m is not None
        # the propeller floor clears the motor *can* it sits on, if not the band's ceiling
        assert prop.z_min_m > motor.z_min_m, station
        assert not set(prop.materials) & set(motor.materials), station
        floors.append(prop.z_min_m)
    # and the four floors really do differ -- if they were equal, the pitch would have been missed
    assert max(floors) - min(floors) == pytest.approx(0.018, abs=1e-6)


def test_the_phantom4_shells_are_last_so_they_cannot_swallow_a_part():
    """Catch-alls must be terminal: an empty selector earlier would claim the whole aircraft."""
    parts = phantom4_parts()
    empty = {"materials": []}
    catchalls = [
        i for i, p in enumerate(parts.parts) if p.select.model_dump(exclude_none=True) == empty
    ]
    assert catchalls, "expected a terminal catch-all"
    assert max(catchalls) == len(parts.parts) - 1
    assert min(catchalls) >= len(parts.parts) - 2
