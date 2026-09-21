"""PT.18 — occluders and daylight in the config path: a wall half in sun, from YAML alone.

PT.1 shipped the per-cell shadow and PT.17 the per-cell solve, and between them no scene config
could reach either: `cell_shadow` had no caller outside its own tests, a patched surface under
the sun took one `shaded` bool for all of its cells, and nothing said which way a scene's world
frame was up. This file is the scene-config half of ADR 0087's headline case.

The load-bearing test is `test_a_wall_half_in_sun_from_yaml_alone`. Its companion
`test_unshaded_cells_stay_bit_identical_to_the_per_prim_solve` is the guard that the beam term
did not move anything it should not have: a cell no occluder ever reaches must reproduce the
per-prim value exactly, spin-up included, because the two are the same arithmetic on the same
operands.

docs/physics-model.md §5.4, §6.1; ADR 0087, ADR 0095; roadmap PT.18.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from irsim.config.scene import load_scene_config
from irsim.scene import Scene
from irsim.thermal.frames import ENU, WorldFrame
from irsim.thermal.shadow import ShadowRectangle, cell_shadow
from irsim.thermal.solar import solar_loading

REPO = pathlib.Path(__file__).resolve().parents[2]
FACETS = REPO / "configs/scenes/thermal_facet_scene.yaml"

HEAD = """
schema_version: 8
scene:
  name: wall_test
  description: "a south-facing concrete wall under an overhang, 14:00 local on a clear June day"
  weather_file: weather/clear_midlat_summer_48h.csv
  atmosphere_preset: us_standard_clear
  site: {{latitude_deg: 45.0, longitude_deg: 30.0, altitude_m: 120.0}}
  start_utc: "{start}"
{frame}
  targets: []
  thermal:
    spin_up_hours: 24.0
    tick_s: 60.0
{occluders}
    surfaces:
{surfaces}
"""

#: A 4 m x 4 m wall in the east-up plane at y = 0, facing **south** (tilt 90, azimuth 180): the
#: side the afternoon sun is on. `u x v = +north`; the sign is irrelevant to the shadow test and
#: the balance takes its normal from the tilt, so only the plane has to agree.
WALL_ENU = """
      - name: wall
        material: concrete
        tilt_deg: 90.0
        azimuth_deg: 180.0
        patch:
          origin_m: [-2.0, 0.0, 0.0]
          u_axis: [0.0, 0.0, 1.0]
          v_axis: [1.0, 0.0, 0.0]
          n_u: 8
          n_v: 8
          du_m: 0.5
          dv_m: 0.5
          thickness_m: 0.15
          frame: world
          prim_path: /World/Wall
"""
#: A horizontal slab standing off the wall on its sunny side, 2 m up: the overhang.
SLAB_ENU = """
    occluders:
      - name: overhang
        centre_m: [-1.0, -1.5, 2.0]
        u_axis: [1.0, 0.0, 0.0]
        v_axis: [0.0, 1.0, 0.0]
        half_u_m: 2.0
        half_v_m: 1.5
"""
#: The same wall and slab authored in a Y-up world (up = +Y, north = -Z, so ENU (e, n, u) is
#: world (e, u, -n)) -- the car stage's convention.
FRAME_Y_UP = "  world_frame: {up: [0.0, 1.0, 0.0], north: [0.0, 0.0, -1.0]}"
WALL_Y_UP = WALL_ENU.replace("u_axis: [0.0, 0.0, 1.0]", "u_axis: [0.0, 1.0, 0.0]")
SLAB_Y_UP = SLAB_ENU.replace("centre_m: [-1.0, -1.5, 2.0]", "centre_m: [-1.0, 2.0, 1.5]").replace(
    "v_axis: [0.0, 1.0, 0.0]", "v_axis: [0.0, 0.0, -1.0]"
)

#: The headline wall faces **south-west** (azimuth 225) and the scene starts at 16:00 local
#: (14:00Z): a June sun at 45 N is high, and a south wall never sees more than ~300 W/m^2 of beam
#: at noon (step 6.5 K on concrete); the SW wall at 16:00 gets 0.66 x DNI and the step passes
#: 10 K -- Morrison 2021's terminator figure and PT.20's bar. The slab is rotated with it.
R2 = 0.7071067811865476
WALL_SW = WALL_ENU.replace("azimuth_deg: 180.0", "azimuth_deg: 225.0").replace(
    "v_axis: [1.0, 0.0, 0.0]", f"v_axis: [{R2}, {-R2}, 0.0]"
)
SLAB_SW = (
    SLAB_ENU.replace("centre_m: [-1.0, -1.5, 2.0]", "centre_m: [-1.65, -2.47, 2.0]")
    .replace("u_axis: [1.0, 0.0, 0.0]", f"u_axis: [{R2}, {-R2}, 0.0]")
    .replace("v_axis: [0.0, 1.0, 0.0]", f"v_axis: [{-R2}, {-R2}, 0.0]")
)
START_16H = "2024-06-21T14:00:00Z"

#: `lateral_conduction: false` (PT.11): the bit-identity below is a property of the shadow term
#: alone. With asphalt's own in-plane conduction on, a lit cell beside a shaded one exchanges
#: heat with it -- millikelvins for asphalt, but not bits.
ROAD = """
      - name: road
        material: asphalt_dry
        tilt_deg: 0.0
        lateral_conduction: false
        patch:
          origin_m: [-1.5, -1.2, 0.0]
          u_axis: [1.0, 0.0, 0.0]
          v_axis: [0.0, 1.0, 0.0]
          n_u: 10
          n_v: 8
          du_m: 0.3
          dv_m: 0.3
          thickness_m: 0.10
          frame: world
          prim_path: /World/Road
"""
#: A 1 m square 2 m above the road's south-west quarter, so its shadow crosses part of the patch
#: around midday and misses the north-east corner all day.
POST_CAP = """
    occluders:
      - name: cap
        centre_m: [-0.8, -0.6, 2.0]
        u_axis: [1.0, 0.0, 0.0]
        v_axis: [0.0, 1.0, 0.0]
        half_u_m: 0.5
        half_v_m: 0.5
"""


def _write(
    tmp_path: pathlib.Path,
    surfaces: str,
    occluders: str = "",
    frame: str = "",
    start: str = "2024-06-21T12:00:00Z",
) -> pathlib.Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    out = tmp_path / "scene.yaml"
    out.write_text(HEAD.format(frame=frame, occluders=occluders, surfaces=surfaces, start=start))
    return out


# --- the headline case --------------------------------------------------------------------------


@pytest.mark.slow  # a 24 h spin-up of 64 cells: about a second
def test_a_wall_half_in_sun_from_yaml_alone(tmp_path, tophat_lwir_lut) -> None:  # type: ignore[no-untyped-def]
    """The scene config declares the wall and the overhang; the field carries the step.

    Nothing here is built in Python: the patch, the occluder and the world frame all come from
    the YAML, the spin-up sees the shadow, and the terminator across one prim is > 8 K.

    Measured 9.9 K on concrete at 16:00 local -- a transient under a real diurnal sun, not
    PT.1's 22.1 K equilibrium under a fixed 850 W/m^2 beam 20 degrees off the normal.
    """
    scene = Scene.from_file(
        _write(tmp_path, WALL_SW, SLAB_SW, start=START_16H), {"lwir": tophat_lwir_lut}
    )
    assert set(scene.occluders) == {"overhang"}
    fld = scene.surface_fields["wall"]
    lit = fld.field.forcing_at.cell_visibility(scene.t0_s)
    assert 0.2 < lit.mean() < 0.8, (
        f"the overhang should shade part of the wall, not {lit.mean():.2f}"
    )

    temperature = np.asarray(fld.temperature_at(scene.t0_s), dtype=np.float64)
    step = float(temperature[lit == 1.0].mean() - temperature[lit == 0.0].mean())
    assert step >= 8.0, f"lit/shaded step is only {step:.2f} K at 16:00 on a clear June day"
    # And the per-prim entry for the same surface is one number: the representation this replaces.
    assert float(np.ptp(temperature)) > 8.0
    assert scene.thermal.temperature_at(scene.t0_s).shape == (1,)


def test_the_per_cell_beam_is_pt1_s_shadow_on_the_scene_s_own_sun(
    tmp_path, tophat_lwir_lut
) -> None:  # type: ignore[no-untyped-def]
    """The forcing's q_solar equals `solar_loading` on `cell_shadow`, exactly, cell for cell."""
    scene = Scene.from_file(_write(tmp_path, WALL_ENU, SLAB_ENU), {"lwir": tophat_lwir_lut})
    forcing = scene.surface_fields["wall"].field.forcing_at
    t = scene.t0_s
    sun = scene.solar_terms(t)
    assert sun.above_horizon and sun.dni_w_m2 > 700.0
    patch = scene.patches["wall"]
    shade = cell_shadow(patch, sun.direction_enu, [scene.occluders["overhang"]])
    # PT.21: the diffuse term carries each cell's own sky view -- the wall's half-dome where
    # nothing hides it, less under the overhang.
    svf = forcing.sky_view
    assert svf is not None and svf.max() == 0.5 and svf.min() < 0.5
    expected = solar_loading(
        np.array([0.0, -1.0, 0.0]), sun.direction_enu, sun.dni_w_m2, sun.dhi_w_m2, svf, shade
    )
    got = np.asarray(forcing(t).q_solar_w_m2)
    assert np.array_equal(got, expected)
    # Shade gates the beam only: the shaded cells still carry their share of the diffuse sky.
    assert np.allclose(got[shade == 0.0], svf[shade == 0.0] * sun.dhi_w_m2)
    assert float(got[shade == 0.0].min()) > 0.0


@pytest.mark.slow
def test_unshaded_cells_stay_bit_identical_to_the_per_prim_solve(tmp_path, tophat_lwir_lut) -> None:  # type: ignore[no-untyped-def]
    """The verification cell's "1 mK where unshaded".

    A cell the cap never shades -- at any instant the spin-up or the run evaluates -- sees the
    per-prim expression on the per-prim operands except for one: since PT.21 the cap also hides
    a sliver of its sky (SVF 0.93-0.99 here), so it is the prim's to within 50 mK rather than
    to the bit, warmer by less sky to lose to, and closest to the prim where it sees most sky.
    The bit identity itself lives where the dome is open (`test_sky_view`, PT.17's scene). The
    cells the cap does reach are colder, by kelvins, at noon.
    """
    scene = Scene.from_file(_write(tmp_path, ROAD, POST_CAP), {"lwir": tophat_lwir_lut})
    fld = scene.surface_fields["road"]
    forcing = fld.field.forcing_at
    t0, end = scene.t0_s, scene.t0_s + 2.0 * 3600.0
    # Every instant the integrator asks the forcing about, spin-up included (60 s steps, midpoint
    # evaluations), lies on a 30 s grid from 24 h before t0.
    grid = np.arange(t0 - 24.0 * 3600.0, end + 30.0, 30.0)
    first, span = float(scene.weather.time_s[0]), float(np.ptp(scene.weather.time_s))
    always_lit = np.ones(fld.patch.n_cells, dtype=bool)
    for t in grid:
        # The spin-up wraps instants before the weather file into it (`_build_thermal_field`).
        wrapped = float(t) if t >= first else first + (float(t) - first) % span
        always_lit &= forcing.cell_visibility(wrapped) == 1.0
    shaded_now = forcing.cell_visibility(t0) == 0.0
    assert always_lit.any() and shaded_now.any(), (always_lit.sum(), shaded_now.sum())

    fld.advance_to(end)
    scene.thermal.advance_to(end)
    cells = np.asarray(fld.temperature_at(end), dtype=np.float64)
    prim = float(np.asarray(scene.thermal.temperature_at(end))[0])
    svf = forcing.sky_view
    assert svf is not None and svf[always_lit].min() > 0.9 and svf.max() < 1.0
    gap = cells[always_lit] - prim
    assert np.all(gap >= 0.0) and gap.max() < 0.05, (gap.min(), gap.max())  # measured 20 mK
    assert gap[np.argmax(svf[always_lit])] == gap.min()
    assert float(prim - cells[shaded_now].mean()) > 1.0


def test_y_up_and_enu_authoring_of_one_wall_shade_identically(tmp_path, tophat_lwir_lut) -> None:  # type: ignore[no-untyped-def]
    """The car stage is Y-up; the frame block is what lets its patches be shaded at all.

    The rotation between the two axis-aligned frames is exact, so the visibility and the beam
    are equal to the bit, not to a tolerance.
    """
    enu = Scene.from_file(_write(tmp_path / "enu", WALL_ENU, SLAB_ENU), {"lwir": tophat_lwir_lut})
    y_up = Scene.from_file(
        _write(tmp_path / "y", WALL_Y_UP, SLAB_Y_UP, FRAME_Y_UP), {"lwir": tophat_lwir_lut}
    )
    assert enu.world_frame.is_enu and not y_up.world_frame.is_enu
    a, b = enu.surface_fields["wall"].field.forcing_at, y_up.surface_fields["wall"].field.forcing_at
    for t in (enu.t0_s, enu.t0_s + 1800.0, enu.t0_s + 3600.0):
        assert np.array_equal(a.cell_visibility(t), b.cell_visibility(t))
        assert np.array_equal(a(t).q_solar_w_m2, b(t).q_solar_w_m2)
    assert 0.2 < a.cell_visibility(enu.t0_s).mean() < 0.8


# --- what is refused ----------------------------------------------------------------------------


def test_a_shaded_flag_beside_occluders_is_two_authorities(tmp_path) -> None:  # type: ignore[no-untyped-def]
    wall = WALL_ENU.replace("azimuth_deg: 180.0\n", "azimuth_deg: 180.0\n        shaded: true\n")
    with pytest.raises(ValueError, match="two shadow authorities"):
        load_scene_config(_write(tmp_path, wall, SLAB_ENU))


def test_a_shaded_flag_without_occluders_is_still_the_v7_scene(tmp_path, tophat_lwir_lut) -> None:  # type: ignore[no-untyped-def]
    wall = WALL_ENU.replace("azimuth_deg: 180.0\n", "azimuth_deg: 180.0\n        shaded: true\n")
    scene = Scene.from_file(_write(tmp_path, wall), {"lwir": tophat_lwir_lut})
    q = np.asarray(scene.surface_fields["wall"].field.forcing_at(scene.t0_s).q_solar_w_m2)
    sun = scene.solar_terms(scene.t0_s)
    assert np.allclose(q, 0.5 * sun.dhi_w_m2)  # the whole wall in the flag's shade


def test_an_occluder_in_a_moving_frame_is_refused_at_load(tmp_path) -> None:  # type: ignore[no-untyped-def]
    slab = SLAB_ENU.replace("half_v_m: 1.5\n", "half_v_m: 1.5\n        frame: /World/Car\n")
    with pytest.raises(ValueError, match="moving frame"):
        load_scene_config(_write(tmp_path, WALL_ENU, slab))


def test_a_patch_in_a_moving_frame_under_occluders_is_refused_at_build(
    tmp_path, tophat_lwir_lut
) -> None:  # type: ignore[no-untyped-def]
    wall = WALL_ENU.replace("frame: world", "frame: /World/Wall")
    with pytest.raises(ValueError, match="moving frame"):
        Scene.from_file(_write(tmp_path, wall, SLAB_ENU), {"lwir": tophat_lwir_lut})


def test_a_patch_whose_plane_disagrees_with_its_tilt_is_refused(tmp_path, tophat_lwir_lut) -> None:  # type: ignore[no-untyped-def]
    """A vertical grid on a surface that says it faces up: the shadow would be cast on cells the
    balance's normal never faces."""
    wall = WALL_ENU.replace("tilt_deg: 90.0", "tilt_deg: 0.0")
    with pytest.raises(ValueError, match="does not match the surface's tilt"):
        Scene.from_file(_write(tmp_path, wall, SLAB_ENU), {"lwir": tophat_lwir_lut})


def test_malformed_frames_and_occluders_fail_at_load(tmp_path) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ValueError, match="perpendicular"):
        load_scene_config(
            _write(tmp_path, WALL_ENU, SLAB_ENU, "  world_frame: {up: [0, 0, 1], north: [0, 1, 1]}")
        )
    with pytest.raises(ValueError, match="perpendicular"):
        load_scene_config(
            _write(
                tmp_path,
                WALL_ENU,
                SLAB_ENU.replace("v_axis: [0.0, 1.0, 0.0]", "v_axis: [1.0, 1.0, 0.0]"),
            )
        )
    with pytest.raises(ValueError, match="unique"):
        load_scene_config(_write(tmp_path, WALL_ENU, SLAB_ENU + SLAB_ENU.split("occluders:\n")[1]))


# --- the frame itself ---------------------------------------------------------------------------


def test_the_world_frame_is_right_handed_and_exact() -> None:
    y_up = WorldFrame(up=np.array([0.0, 1.0, 0.0]), north=np.array([0.0, 0.0, -1.0]))
    assert np.array_equal(y_up.east, [1.0, 0.0, 0.0])
    r = y_up.world_from_enu
    assert np.array_equal(r @ r.T, np.eye(3)) and np.linalg.det(r) == pytest.approx(1.0)
    up_enu = np.array([0.0, 0.0, 1.0])
    assert np.array_equal(y_up.to_world(up_enu), [0.0, 1.0, 0.0])
    assert np.array_equal(y_up.to_enu(y_up.to_world([0.3, -0.4, 0.5])), [0.3, -0.4, 0.5])
    assert ENU.is_enu and np.array_equal(ENU.world_from_enu, np.eye(3))
    with pytest.raises(ValueError, match="zero length"):
        WorldFrame(up=np.zeros(3), north=np.array([0.0, 1.0, 0.0]))


def test_a_v7_scene_reads_as_enu_with_nothing_to_shade(tophat_lwir_lut) -> None:  # type: ignore[no-untyped-def]
    scene = Scene.from_file(FACETS, {"lwir": tophat_lwir_lut})
    assert scene.world_frame.is_enu and scene.occluders == {}


def test_a_box_casts_from_all_six_faces() -> None:
    from irsim.thermal.shadow import box_faces
    from irsim.thermal.surface_field import PlanarPatch

    faces = box_faces([0.0, 0.0, 1.0], [2.0, 2.0, 2.0])
    assert len(faces) == 6 and all(isinstance(f, ShadowRectangle) for f in faces)
    ground = PlanarPatch(
        origin_m=np.array([-5.0, -5.0, 0.0]),
        u_axis=np.array([1.0, 0.0, 0.0]),
        v_axis=np.array([0.0, 1.0, 0.0]),
        n_u=40,
        n_v=40,
        du_m=0.25,
        dv_m=0.25,
    )
    # A low sun from the east: the shadow is long, and the box's east *side* casts the near part
    # of it -- the top face alone leaves the cells beside the box lit.
    low = np.array([np.cos(np.radians(30.0)), 0.0, np.sin(np.radians(30.0))])
    from_top_only = cell_shadow(ground, low, [faces[5]])  # +z face alone
    from_box = cell_shadow(ground, low, faces)
    assert from_box.mean() < from_top_only.mean() < 1.0
    with pytest.raises(ValueError, match="positive extents"):
        box_faces([0.0, 0.0, 0.0], [1.0, 0.0, 1.0])
