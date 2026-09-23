"""Every rendered airframe against the scene config that describes it (PT.9, ADR 0123/0124).

One driver now flies two aircraft: `irsim_isaac.quad_outbound`'s invented heavy-lift frame and
`irsim_isaac.phantom3`'s DJI Phantom 3. Each travels with its own scene config, and the agreement
between the two files is not a property either one can check on its own.

**The invariant, and why it is sampled rather than reasoned about.** Every point on the surface of
a patched prim must fall inside the patch bound to it. `PointwiseTemperature` refuses a frame
where one does not, which is the right failure -- but it refuses it thirty seconds into a Kit
session, and the arithmetic that gets it wrong is easy to get wrong in the *direction* that looks
right. That happened here: a plate wants its prim **inside** its patch by a couple of millimetres,
and an arm wants its patch **outside** the prim, because an arm seen from below presents its end
caps and both side faces while a plate presents only its plane. Authoring the arms like the plates
put 1204 pixels outside every patch on the first render of the Phantom.

So the check samples the real prim surface -- every face, rotated by the part's own yaw -- and
asks the real `PlanarPatch.contains`. No formula is restated, which is what makes it able to
disagree with the one in the scene file.
"""

from __future__ import annotations

import functools
import math
import pathlib

import numpy as np
import pytest

from irsim.scene import Scene
from irsim_isaac.airframe import Part
from irsim_isaac.phantom3 import PHANTOM_3
from irsim_isaac.quad_outbound import POINTWISE_QUAD

REPO = pathlib.Path(__file__).resolve().parents[2]
SCENES = REPO / "configs" / "scenes"
QUAD_ROOT = "/World/Targets/quad"

#: ``(label, scene file, the parts the stage authors)``. A new rendered airframe joins here, and
#: every check below runs on it.
AIRFRAMES: tuple[tuple[str, str, tuple[Part, ...]], ...] = (
    ("heavy_lift", "quad_outbound_pointwise.yaml", POINTWISE_QUAD),
    ("phantom3", "phantom3_outbound_pointwise.yaml", PHANTOM_3),
)


@functools.cache
def _scene(name: str) -> Scene:
    """Built once per scene file. Each build spins the fields up 48 h, which is seconds."""
    path = SCENES / name
    assert path.exists(), f"{path} is gone; this test names it deliberately"
    return Scene.from_file(path)


def _surface_points(part: Part, per_axis: int = 11) -> np.ndarray:
    """Points over every face of a part, in the stage frame, honouring its own yaw.

    A cylinder is sampled on its bounding box, which is conservative: the box contains the
    cylinder, so a patch that holds the box holds the cylinder too.
    """
    yaw = math.radians(part.rotate_xyz_deg[1])
    c, s = math.cos(yaw), math.sin(yaw)
    # Rows are the part's local axes expressed in the stage frame, so `local @ rot` is the
    # world offset -- USD's row-vector convention, the same one `irsim.optics.motion` uses.
    rot = np.array([[c, 0.0, -s], [0.0, 1.0, 0.0], [s, 0.0, c]])
    assert part.rotate_xyz_deg[0] == 0.0 and part.rotate_xyz_deg[2] == 0.0, (
        f"{part.name} is rotated about X or Z; this sampler only carries yaw"
    )
    centre = np.asarray(part.centre_m, dtype=np.float64)
    half = 0.5 * np.asarray(part.size_m, dtype=np.float64)
    grid = np.linspace(-1.0, 1.0, per_axis)
    points = []
    for a in grid:
        for b in grid:
            for axis in range(3):
                for sign in (-1.0, 1.0):
                    local = np.zeros(3)
                    local[axis] = sign * half[axis]
                    others = [i for i in range(3) if i != axis]
                    local[others[0]] = a * half[others[0]]
                    local[others[1]] = b * half[others[1]]
                    points.append(centre + local @ rot)
    return np.asarray(points)


@pytest.mark.parametrize(("label", "scene_file", "parts"), AIRFRAMES, ids=[a[0] for a in AIRFRAMES])
def test_every_patch_binds_to_a_prim_the_stage_authors(label, scene_file, parts) -> None:
    scene = _scene(scene_file)
    authored = {f"{QUAD_ROOT}/{p.name}" for p in parts}
    assert scene.patch_prims, f"{label}: the scene declares no patch prims at all"
    for name, path in scene.patch_prims.items():
        assert path in authored, (
            f"{label}: surface {name!r} binds to {path!r}, which this stage does not author. "
            f"authored: {sorted(authored)}"
        )


@pytest.mark.parametrize(("label", "scene_file", "parts"), AIRFRAMES, ids=[a[0] for a in AIRFRAMES])
def test_every_point_of_a_patched_prim_lies_inside_its_patch(label, scene_file, parts) -> None:
    """The one that matters: a prim point outside its patch is a refused frame at render time."""
    scene = _scene(scene_file)
    by_name = {p.name: p for p in parts}
    for name, path in scene.patch_prims.items():
        part = by_name[path.rsplit("/", 1)[1]]
        points = _surface_points(part)
        inside = np.asarray(scene.patches[name].contains(points))
        missed = int((~inside).sum())
        assert missed == 0, (
            f"{label}: {missed} of {inside.size} surface points of {path} fall outside patch "
            f"{name!r}. A plate wants its prim inside its patch; a solid arm wants its patch "
            "outside its prim, because from below it presents its end caps and its sides."
        )


@pytest.mark.parametrize(("label", "scene_file", "parts"), AIRFRAMES, ids=[a[0] for a in AIRFRAMES])
def test_shrinking_a_patch_is_caught(label, scene_file, parts) -> None:
    """The negative control: the check above is not vacuous on either airframe."""
    from irsim.thermal.surface_field import PlanarPatch

    scene = _scene(scene_file)
    name, path = next(iter(scene.patch_prims.items()))
    part = {p.name: p for p in parts}[path.rsplit("/", 1)[1]]
    wide = scene.patches[name]
    narrow = PlanarPatch(
        origin_m=wide.origin_m,
        u_axis=wide.u_axis,
        v_axis=wide.v_axis,
        n_u=max(1, wide.n_u // 3),
        n_v=max(1, wide.n_v // 3),
        du_m=wide.du_m,
        dv_m=wide.dv_m,
        thickness_m=wide.thickness_m,
        frame=wide.frame,
    )
    assert not np.asarray(narrow.contains(_surface_points(part))).all()


@pytest.mark.parametrize(("label", "scene_file", "parts"), AIRFRAMES, ids=[a[0] for a in AIRFRAMES])
def test_every_occluder_face_sits_on_a_prim(label, scene_file, parts) -> None:
    """A shadow caster that is not a face of the aircraft shades from where nothing is."""
    scene = _scene(scene_file)
    assert scene.spec.thermal.occluders, f"{label}: the scene declares no occluders"
    boxes = []
    for part in parts:
        centre = np.asarray(part.centre_m, dtype=np.float64)
        half = 0.5 * np.asarray(part.size_m, dtype=np.float64)
        # Yawed parts get the radius of their own half-extent, which is conservative and enough:
        # no occluder in either scene sits on a rotated part.
        if any(part.rotate_xyz_deg):
            half = np.full(3, float(np.linalg.norm(half)))
        boxes.append((part.name, centre - half, centre + half))
    for occ in scene.spec.thermal.occluders:
        centre = np.asarray(occ.centre_m, dtype=np.float64)
        gaps = {
            n: float(np.linalg.norm(np.maximum(np.maximum(lo - centre, centre - hi), 0.0)))
            for n, lo, hi in boxes
        }
        nearest = min(gaps, key=lambda k: gaps[k])
        assert gaps[nearest] <= 0.005, (
            f"{label}: occluder {occ.name!r} at {centre} is {gaps[nearest] * 1e3:.1f} mm from "
            f"the nearest prim ({nearest})"
        )


@pytest.mark.parametrize(("label", "scene_file", "parts"), AIRFRAMES, ids=[a[0] for a in AIRFRAMES])
def test_the_scene_solves_every_node_the_stage_names(label, scene_file, parts) -> None:
    scene = _scene(scene_file)
    needed = {p.thermal_node for p in parts}
    assert needed <= set(scene.targets), (
        f"{label}: no solver for {sorted(needed - set(scene.targets))}"
    )
    # And nothing the other way: a target no prim carries is a node that never reaches a pixel.
    spare = set(scene.targets) - needed
    assert not spare, f"{label}: the scene declares {sorted(spare)}, which no prim uses"


# ---------------------------------------------------------------------------------------------
# the Phantom, as a named real aircraft
# ---------------------------------------------------------------------------------------------


def test_the_phantom_matches_djis_published_dimensions() -> None:
    """The four numbers DJI publishes, and the clearance they imply.

    These are MEASURED (published) rather than chosen, and they are what makes this airframe a
    Phantom 3 instead of a quadrotor-shaped object. The clearance is the interesting one: it is
    not authored anywhere, it falls out of the other two, and it comes out at 8 mm -- which is
    why 9450 is the largest propeller this frame takes.
    """
    from irsim_isaac import phantom3

    # Bound to locals because ruff reads `CONSTANT == approx(...)` as a Yoda condition.
    diagonal, prop = phantom3.DIAGONAL_M, phantom3.PROP_DIAMETER_M
    offset, tip_to_tip = phantom3.MOTOR_OFFSET_M, phantom3.TIP_TO_TIP_M
    assert diagonal == pytest.approx(0.350)
    assert prop == pytest.approx(0.2394)
    assert offset == pytest.approx(0.350 / (2.0 * math.sqrt(2.0)))
    assert tip_to_tip == pytest.approx(0.5894)

    by_name = {p.name: p for p in PHANTOM_3}
    fr = np.asarray(by_name["motor_fr"].centre_m)
    fl = np.asarray(by_name["motor_fl"].centre_m)
    rr = np.asarray(by_name["motor_rr"].centre_m)
    adjacent = float(np.linalg.norm(fr - fl))
    assert adjacent == pytest.approx(0.350 / math.sqrt(2.0), rel=1e-9), "adjacent motor spacing"
    clearance = adjacent - phantom3.PROP_DIAMETER_M
    assert 0.005 < clearance < 0.012, f"prop clearance came out {clearance * 1e3:.1f} mm"
    # Diagonally opposite, which is the published number itself.
    assert float(np.linalg.norm(fr - np.asarray(by_name["motor_rl"].centre_m))) == pytest.approx(
        0.350, rel=1e-9
    )
    assert float(np.linalg.norm(fl - rr)) == pytest.approx(0.350, rel=1e-9)


def test_each_airframes_span_is_the_one_its_own_motors_measure() -> None:
    """A span quoted beside a layout instead of derived from it is how a pixel count goes wrong.

    The heavy-lift frame is a **plus** -- arms due N, E, S, W -- so its opposite motors are
    ``2 x 0.42`` apart. It was first authored with the X-quad's ``2 x 0.42 x sqrt 2``, which is
    the form every multirotor spec sheet quotes and which overstated this aircraft by 41 %. Both
    spans are now measured off the authored motor prims, which is a thing a constant cannot be
    wrong about.
    """
    import itertools

    from irsim_isaac import phantom3
    from irsim_isaac.quad_outbound import QUAD_ROTOR
    from irsim_isaac.quad_outbound import SPAN_M as HEAVY_SPAN_M
    from irsim_isaac.quad_outbound import TIP_TO_TIP_M as HEAVY_TIPS_M

    def widest(parts: tuple[Part, ...]) -> float:
        motors = [
            np.asarray(p.centre_m, dtype=np.float64) for p in parts if p.name.startswith("motor_")
        ]
        return max(float(np.linalg.norm(a - b)) for a, b in itertools.combinations(motors, 2))

    heavy, phantom = widest(POINTWISE_QUAD), widest(PHANTOM_3)
    assert heavy == pytest.approx(HEAVY_SPAN_M, rel=1e-12), "the constant lies about the layout"
    assert phantom == pytest.approx(phantom3.DIAGONAL_M, rel=1e-12)
    heavy_tips = HEAVY_TIPS_M
    assert heavy_tips == pytest.approx(heavy + 2.0 * QUAD_ROTOR.radius_m, rel=1e-12)

    assert heavy / phantom == pytest.approx(2.4, abs=0.05), "across the motors"
    tips = heavy_tips / phantom3.TIP_TO_TIP_M
    assert tips == pytest.approx(1.94, abs=0.05), "across the props"


def test_white_abs_absorbs_a_quarter_of_what_carbon_does() -> None:
    """The material fact behind the Phantom's weak daylight signature.

    Asserted on the library rather than on a render, because it is a property of the substance:
    a white TiO2-pigmented shell takes 0.25 of the short-wave flux where a carbon laminate takes
    0.90. Everything the two scenes measure downstream of this is downstream of this.
    """
    from irsim.materials.library import MaterialLibrary

    lib = MaterialLibrary.load()
    white = lib.get("abs_plastic_white")
    carbon = lib.get("carbon_fibre")
    alpha_white = float(white.spec.thermal.solar_absorptivity)
    alpha_carbon = float(carbon.spec.thermal.solar_absorptivity)
    assert alpha_white == pytest.approx(0.25)
    assert alpha_carbon == pytest.approx(0.90)
    assert alpha_white / alpha_carbon < 0.3

    # And yet both are near-blackbodies in LWIR: the pigment that makes the shell white does
    # nothing at 10 um. A thermal camera cannot tell them apart by emissivity -- only by the
    # temperature each one reached, which is the whole point.
    assert float(white.band_properties("lwir").emissivity) > 0.9
    assert float(carbon.band_properties("lwir").emissivity) >= 0.9

    # **And the thermal mass is NOT the story**, which is worth pinning so nobody later explains
    # the Phantom's weak signature with a second cause it does not have. At the same 1.5 mm the
    # ABS shell carries 2205 J m^-2 K^-1 against the carbon laminate's 2520 -- 12 % less, not
    # half. The measured difference downstream is the absorptivity above and almost nothing else.
    def capacity(m: object) -> float:
        t = m.spec.thermal  # type: ignore[attr-defined]
        return float(t.density_kg_m3 * t.specific_heat_j_kgk * t.thickness_m)

    assert capacity(white) == pytest.approx(2205.0)
    assert capacity(carbon) == pytest.approx(2520.0)
    assert 0.85 < capacity(white) / capacity(carbon) < 1.0, "same order, not half"
    # Against a painted steel panel it *is* about half, which is the comparison the material
    # file makes and the one that does not carry over to another composite.
    assert capacity(white) < 0.6 * capacity(lib.get("car_paint_white"))


def test_the_phantoms_sunlit_skin_sits_far_closer_to_air_than_the_carbon_decks() -> None:
    """Measured on both scenes, and the ordering is the claim -- not the ratio.

    The two scenes differ in more than material: the Phantom flies at 09:30 under broken cumulus
    with the sun at 39 degrees, the heavy-lift at noon under a clear sky at 61. So this records
    what each one actually reads and asserts that a white consumer drone is the weaker daylight
    target by a wide margin, without pretending the whole gap is the absorptivity.
    """
    excess = {}
    for label, scene_file, _parts in AIRFRAMES:
        scene = _scene(scene_file)
        # The *largest* up-facing surface, not the alphabetically first: on the heavy-lift
        # scene an arm sorts before the deck, and an arm is half in its own shadow, so naming
        # it here would quietly compare a shaded strip against a whole sunlit plate.
        up_facing = [s.name for s in scene.spec.thermal.surfaces if float(s.tilt_deg) < 90.0]
        widest_name = max(up_facing, key=lambda n: scene.patches[n].n_u * scene.patches[n].n_v)
        field = scene.surface_fields[widest_name]
        field.advance_to(scene.t0_s)
        skin = float(np.mean(np.asarray(field.temperature_at(scene.t0_s))))
        excess[label] = skin - float(scene.weather.at(scene.t0_s).t_air_k)

    assert excess["heavy_lift"] > 25.0, excess
    assert 1.5 < excess["phantom3"] < 6.0, excess
    assert excess["phantom3"] < 0.25 * excess["heavy_lift"], excess


def test_the_phantom_scene_flies_under_broken_cloud_and_the_other_does_not() -> None:
    """Cloud has to be in the weather before a render can put it in either band (ADR 0076).

    `generate_sky_cloud` lays coverage over the whole hemisphere, so a 31 x 25 degree field on a
    0.05 sky almost never contains one. The claim "cloud in both bands" is only testable on a
    scene whose weather actually carries cloud, and this pins which scene that is.
    """
    phantom = _scene("phantom3_outbound_pointwise.yaml")
    heavy = _scene("quad_outbound_pointwise.yaml")
    sct = float(phantom.weather.at(phantom.t0_s).cloud_fraction)
    clear = float(heavy.weather.at(heavy.t0_s).cloud_fraction)
    # SCT is 3-4 oktas in the meteorological sense: 0.375 to 0.5.
    assert 0.375 <= sct <= 0.5, f"the Phantom scene's sky is {sct:.2f}, which is not SCT"
    assert clear < 0.1, f"the heavy-lift scene's sky is {clear:.2f}, which is no longer clear"
    # The sun has to survive it, or the sunlit-versus-shaded contrast has nothing to come from.
    noon = phantom.weather.at(phantom.t0_s + 4.5 * 3600.0)
    assert float(noon.dni_w_m2) > 600.0, "broken cloud, not overcast: the direct beam survives"
