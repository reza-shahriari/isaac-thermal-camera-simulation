"""A driving vehicle's wheels: the brake deposit and the tyre relation, driven from a trace.

Roadmap `TC.8`; docs/physics-model.md §6.6; ADR 0038 (brakes are an energy deposit), ADR 0089
(the vehicle source solver and its "revisit when"), ADR 0088 (a radiator facing a panel).

`brake_temperature_rise_k` and `tyre_delta_t_k` have been in `irsim.thermal.vehicle` since M6.14
with **no caller outside their own unit tests**, so no frame this project has rendered has ever
contained a warm brake. These tests drive both from a `VehicleState` trace and then check the one
thing a picture depends on: that the arch above a tyre ends up measurably warmer than the door
beside it, which is the feature a thermal camera actually reads on a car that has been driven.
"""

from __future__ import annotations

import numpy as np
import pytest

from irsim.thermal.coupling import RadiationExchange
from irsim.thermal.drive_cycle import (
    DriveCycle,
    WheelHistory,
    WheelSpec,
    brake_energy_share,
    passenger_car_wheels,
)
from irsim.thermal.spatial_sources import RadiantRectangle
from irsim.thermal.surface_field import PlanarPatch
from irsim.thermal.vehicle import VEHICLE_HEAT_SOURCES, VehicleState, brake_temperature_rise_k

EX = np.array([1.0, 0.0, 0.0])
EY = np.array([0.0, 1.0, 0.0])
EZ = np.array([0.0, 0.0, 1.0])

#: ADR 0038's worked example: 1600 kg, an 8 kg disc, 90 % of the energy into the discs.
CAR_MASS_KG = 1600.0
DISC_MASS_KG = 8.0

T_AIR_K = 293.15


def _stop(mass: float, v_from: float, v_to: float = 0.0) -> float:
    """The whole car's rise into one 8 kg disc, which is ADR 0038's own framing."""
    return brake_temperature_rise_k(mass, v_from, v_to, DISC_MASS_KG)


def test_a_stop_deposits_adr_0038s_number_and_twice_the_speed_deposits_four_times_it() -> None:
    """The reason brakes are arithmetic and not a schedule.

    A scripted brake temperature is independent of how hard the car braked, which is the one thing
    a braking cue carries. 162 K into an 8 kg disc from 30 m/s, and exactly four times that from
    60 m/s, because the energy goes as v².
    """
    assert _stop(CAR_MASS_KG, 30.0) == pytest.approx(162.0, abs=0.5)
    assert _stop(CAR_MASS_KG, 60.0) / _stop(CAR_MASS_KG, 30.0) == pytest.approx(4.0)


def test_the_trace_puts_that_energy_into_the_discs_split_across_the_axles() -> None:
    """One stop, four corners: the front discs take roughly twice the rear ones.

    A model that split the energy evenly would render four identical wheels, which is not what a
    camera sees behind a car that has just come off a motorway.
    """
    cycle = DriveCycle(CAR_MASS_KG, passenger_car_wheels())
    cycle.step(VehicleState(t_s=0.0, speed_m_s=30.0))
    seen = cycle.step(VehicleState(t_s=3.0, speed_m_s=0.0, braking=True))

    front = seen["wheel_fl"].disc_delta_t_k
    rear = seen["wheel_rl"].disc_delta_t_k
    assert front / rear == pytest.approx(0.65 / 0.35, rel=1e-6)
    assert seen["wheel_fl"].disc_delta_t_k == pytest.approx(seen["wheel_fr"].disc_delta_t_k)
    # The four corners together hold the whole deposit, exactly. The three seconds of cooling
    # apply to what the discs held *before* the stop, which was nothing: the deposit is the work
    # done arriving at this sample, so it cannot lose part of itself to a constant it has not yet
    # spent. That ordering is the model's, and asserting the cooled figure here would encode the
    # opposite one.
    total = sum(w.disc_delta_t_k for w in seen.values())
    assert total == pytest.approx(_stop(CAR_MASS_KG, 30.0), rel=1e-9)


def test_lifting_off_is_not_braking() -> None:
    """A trace that slows without the brake puts nothing in the discs.

    Engine braking, a hill, or simply coasting all reduce speed. Depositing kinetic energy into
    the discs for any of them would paint a hot brake on a car that never touched the pedal.
    """
    history = WheelHistory(CAR_MASS_KG, WheelSpec("wheel_fl", share=0.325))
    history.step(VehicleState(t_s=0.0, speed_m_s=30.0))
    coast = history.step(VehicleState(t_s=10.0, speed_m_s=18.0, braking=False))
    assert coast.disc_delta_t_k == 0.0
    braked = history.step(VehicleState(t_s=20.0, speed_m_s=6.0, braking=True))
    assert braked.disc_delta_t_k == pytest.approx(0.325 * _stop(CAR_MASS_KG, 18.0, 6.0), rel=1e-9)
    assert braked.disc_delta_t_k > 15.0


def test_a_disc_forgets_a_stop_over_its_own_time_constant() -> None:
    """Only the cool-down is a time constant (ADR 0038), and it is the brake row's 300 s."""
    history = WheelHistory(CAR_MASS_KG, WheelSpec("wheel_fl", share=0.325))
    history.step(VehicleState(t_s=0.0, speed_m_s=30.0))
    hot = history.step(VehicleState(t_s=2.0, speed_m_s=0.0, braking=True)).disc_delta_t_k
    tau = VEHICLE_HEAT_SOURCES["brake_disc"].tau_cool_s
    cool = history.step(VehicleState(t_s=2.0 + tau, speed_m_s=0.0)).disc_delta_t_k
    assert cool / hot == pytest.approx(np.exp(-1.0), rel=1e-9)


def test_a_parked_car_has_cold_tyres_however_long_it_waits() -> None:
    """§6.6's +10 K floor describes a tyre rolling slowly, not one standing still.

    Tyre heating is flexing work. Reading the relation literally at v = 0 would give every car in
    every car park a +10 K wheel, which is a feature a detector would learn and no camera sees.
    The tyre still *forgets* over its own 1800 s, so a car that has just pulled up is warm.
    """
    history = WheelHistory(CAR_MASS_KG, WheelSpec("wheel_fl", share=0.325))
    history.step(VehicleState(t_s=0.0, speed_m_s=30.0))
    driven = history.step(VehicleState(t_s=3600.0, speed_m_s=30.0)).tyre_delta_t_k
    assert driven > 25.0, "an hour at 30 m/s should reach the top of §6.6's range"

    just_stopped = history.step(VehicleState(t_s=3660.0, speed_m_s=0.0)).tyre_delta_t_k
    assert just_stopped > 0.9 * driven, "a minute after stopping the tyre is still hot"
    two_hours = VehicleState(t_s=3660.0 + 4.0 * 1800.0, speed_m_s=0.0)
    long_parked = history.step(two_hours).tyre_delta_t_k
    assert long_parked < 1.0, f"a car parked two hours still shows {long_parked:.2f} K"


def test_the_tyre_follows_its_history_and_not_its_current_speed() -> None:
    """A 20-30 minute constant means a frame shows where the car has *been*.

    The negative control is the steady-state reading: a car that has just accelerated to 30 m/s
    would show the full rise immediately if the relation were applied without its time constant.
    """
    history = WheelHistory(CAR_MASS_KG, WheelSpec("wheel_fl", share=0.325))
    history.step(VehicleState(t_s=0.0, speed_m_s=0.0))
    after_a_minute = history.step(VehicleState(t_s=60.0, speed_m_s=30.0)).tyre_delta_t_k
    steady = float(np.asarray(VEHICLE_HEAT_SOURCES["tyre"].delta_t_max_k))
    assert after_a_minute < 0.2 * steady, "a minute is nothing against a 20-minute constant"


@pytest.mark.parametrize(
    ("corner", "expected"),
    [("wheel_fl", 0.325), ("wheel_fr", 0.325), ("wheel_rl", 0.175), ("wheel_rr", 0.175)],
)
def test_the_axle_split_comes_from_the_corners_own_name(corner: str, expected: float) -> None:
    assert brake_energy_share(corner) == pytest.approx(expected)


def test_a_corner_whose_axle_cannot_be_read_is_refused() -> None:
    """Guessing would silently put the front axle's energy on a rear disc."""
    with pytest.raises(ValueError, match="which axle"):
        brake_energy_share("nearside_wheel")


def test_corners_may_not_claim_more_than_the_whole_cars_braking_energy() -> None:
    with pytest.raises(ValueError, match="which is > 1"):
        DriveCycle(CAR_MASS_KG, [WheelSpec("wheel_fl", 0.6), WheelSpec("wheel_fr", 0.6)])


# --- the feature a camera reads ---------------------------------------------------------------


def _arch_patch(centre_z: float) -> PlanarPatch:
    """A 0.5 x 0.3 m panel 0.12 m above the tread, facing down at it -- the arch liner."""
    return PlanarPatch(
        origin_m=np.array([-0.25 + 0.79, 0.74, centre_z - 0.15]),
        u_axis=EX,
        v_axis=EZ,
        n_u=10,
        n_v=6,
        du_m=0.05,
        dv_m=0.05,
        thickness_m=0.002,
    )


def test_after_a_drive_the_arch_above_the_tyre_is_warmer_than_the_door_beside_it() -> None:
    """`TC.8`'s criterion, and the reason the wheel sources are worth wiring at all.

    The arch liner sits 0.12 m above the tread and sees a large solid angle of it; the door is
    the same panel moved 1.2 m along the car, where the tyre is a small object far away. The
    difference is the cells' net flux from the tyre, converted to a temperature through the
    panel's own conductance -- so the claim is about geometry and §6.6's relation together, and
    it fails if either the view factors or the tyre model is wrong.
    """
    cycle = DriveCycle(CAR_MASS_KG, passenger_car_wheels())
    cycle.step(VehicleState(t_s=0.0, speed_m_s=0.0))
    for minute in range(1, 41):
        seen = cycle.step(VehicleState(t_s=60.0 * minute, speed_m_s=27.0))
    tyre_k = seen["wheel_fl"].tyre_k(T_AIR_K)
    assert tyre_k - T_AIR_K > 15.0, "forty minutes at 27 m/s should warm the tyre well"

    tread = RadiantRectangle(
        centre_m=np.array([0.79, 0.62, -1.125]),
        u_axis=EX,
        v_axis=EZ,
        half_u_m=0.11,
        half_v_m=0.14,
        emissivity=0.94,
    )
    arch = _arch_patch(-1.125)
    door = _arch_patch(0.075)  # the same panel, 1.2 m back along the car

    #: A painted steel panel: what it does with the flux it receives.
    conductance_w_m2_k = 12.0
    rise = {}
    for name, patch in (("arch", arch), ("door", door)):
        flux = RadiationExchange(tread, patch, surface_emissivity=0.92).cell_flux_w_m2(tyre_k)
        rise[name] = float(np.max(flux) / conductance_w_m2_k)
    assert rise["arch"] - rise["door"] > 2.0, (
        f"the arch rises {rise['arch']:.2f} K and the door {rise['door']:.2f} K above what they "
        "would be with no wheel; TC.8 asks for more than 2 K between them"
    )


def test_the_demo_cars_wheel_radiators_pair_with_the_drive_cycles_corners_by_name() -> None:
    """Pairing by index would put the front-left disc's temperature on another corner.

    `car_demo` authors its wheel prims in one order and `passenger_car_wheels` its corners in
    another the day somebody edits either, and the two lists are joined by a driver that has no
    way to notice. The names are the join.
    """
    from irsim_isaac.car_demo import CarGeometry

    radiators = dict(CarGeometry().wheel_arch_radiators())
    corners = {w.name for w in passenger_car_wheels()}
    assert {name.rsplit("_", 1)[0] for name in radiators} == corners
    assert set(radiators) == {f"{c}_{part}" for c in corners for part in ("tyre", "disc")}


def test_a_tread_radiator_faces_up_into_its_arch_and_a_disc_faces_outboard() -> None:
    """Geometry, because a rectangle facing the wrong way heats the road instead of the car.

    `view_factor_to_parallel_rectangle` takes the plane it is given: a tread authored facing down
    would produce a perfectly plausible number for a panel that is not there.
    """
    from irsim_isaac.car_demo import CarGeometry

    car = CarGeometry()
    radiators = dict(car.wheel_arch_radiators())
    for corner in ("wheel_fl", "wheel_fr", "wheel_rl", "wheel_rr"):
        tread = radiators[f"{corner}_tyre"]
        assert abs(float(np.dot(tread.normal, EY))) == pytest.approx(1.0)
        assert tread.centre_m[1] > car.wheel_radius_m, "the tread crown is above the axle"
        disc = radiators[f"{corner}_disc"]
        assert abs(float(np.dot(disc.normal, EX))) == pytest.approx(1.0)
        assert disc.centre_m[1] == pytest.approx(car.wheel_radius_m), "a disc is on the axle"
    # Left and right discs sit on opposite sides of the car, so one camera sees one pair.
    assert radiators["wheel_fl_disc"].centre_m[0] < 0.0
    assert radiators["wheel_fr_disc"].centre_m[0] > 0.0
