"""TC.10 -- heat exchange between objects, as one exchange group.

A road under a parked car exchanged longwave with the sky and an air-temperature surround, never
with the car, unless a scene author hand-placed one of ADR 0088's parallel rectangles.
`irsim.thermal.object_exchange` joins solved fields into a group that trades radiation by TC.9's
traced factors, net of the sky each body hides. The oracles: ADR 0088's closed form for the one
geometry it covers, energy that one body loses and the other gains, lockstep stepping, and the
night-time patch a warm pan leaves on the road beneath it.
docs/physics-model.md §6.1, §6.6; ADR 0088, ADR 0156, ADR 0157; roadmap TC.10.
"""

from __future__ import annotations

import numpy as np
import pytest

from irsim.thermal.facets import FacetForcing, FacetProperties
from irsim.thermal.object_exchange import ExchangeBody, ExchangedField, ObjectExchange
from irsim.thermal.spatial_sources import (
    RadiantRectangle,
    occluded_longwave_flux,
    patch_view_factors,
)
from irsim.thermal.surface_field import PlanarPatch, PlanarThermalField

EX, EY, EZ = np.eye(3)


def road(n: int = 8, size_m: float = 1.0) -> PlanarPatch:
    """A horizontal ``n × n`` patch of side ``size_m`` centred on the origin, facing up."""
    d = size_m / n
    return PlanarPatch(
        origin_m=np.array([-0.5 * size_m, -0.5 * size_m, 0.0]),
        u_axis=EX,
        v_axis=EY,
        n_u=n,
        n_v=n,
        du_m=d,
        dv_m=d,
        thickness_m=0.05,
    )


def pan(height_m: float, half_m: float = 0.25, n: int = 4) -> PlanarPatch:
    """A horizontal ``2 half × 2 half`` patch at ``height_m``, facing down (``u × v = −z``)."""
    d = 2.0 * half_m / n
    return PlanarPatch(
        origin_m=np.array([-half_m, half_m, height_m]),
        u_axis=EX,
        v_axis=-EY,
        n_u=n,
        n_v=n,
        du_m=d,
        dv_m=d,
        thickness_m=0.002,
    )


def _props(n: int, c: float, eps: float) -> FacetProperties:
    return FacetProperties(
        heat_capacity_j_m2_k=np.full(n, c),
        emissivity=np.full(n, eps),
        solar_absorptivity=np.full(n, 0.9),
    )


def _night(t_air_k: float = 280.0, q_lw: float = 250.0, h: float = 8.0) -> FacetForcing:
    return FacetForcing(t_air_k=t_air_k, h_w_m2_k=h, q_longwave_down_w_m2=q_lw)


# --- bodies ------------------------------------------------------------------------------------


def test_a_planar_body_is_its_cells_as_triangles() -> None:
    patch = road(4)
    body = ExchangeBody.from_planar("road", patch, 0.95)
    assert body.n_cells == 16 and body.n_faces == 32
    areas = body.soup.vertices  # noqa: F841 - the soup is well formed if this is reachable
    from irsim.thermal.view_factors import RadiantMesh

    mesh = RadiantMesh("road", body.soup)
    assert mesh.area_m2 == pytest.approx(patch.area_m2, rel=1e-12)
    assert np.allclose(mesh.face_normals, patch.normal)
    # the two triangles of cell c are faces 2c and 2c + 1, and the maps say so
    assert np.array_equal(np.flatnonzero(body.receive[5]), [10, 11])
    assert np.array_equal(np.flatnonzero(body.emit[:, 5]), [10, 11])
    centres = patch.cell_centres()
    tri = body.soup.vertices[body.soup.faces[10]]
    assert np.allclose(tri.mean(axis=0)[:2] - centres[5][:2], 0.0, atol=patch.du_m)


# --- the physics against ADR 0088 --------------------------------------------------------------


def test_parallel_plates_reproduce_adr_0088_to_one_percent() -> None:
    """A hot pan over a road is the one geometry the closed form covers: the exchange's net
    flux per road cell is `occluded_longwave_flux` with C-11's factors, to 1 % in total and
    within 3 % of the peak cell by cell (the fan's resolution)."""
    ground, hot = road(8), pan(0.3)
    exchange = ObjectExchange(
        [ExchangeBody.from_planar("road", ground, 0.95), ExchangeBody.from_planar("pan", hot, 0.9)]
    )
    t_road, t_pan, q_lw = 285.0, 350.0, 250.0
    temps = np.concatenate([np.full(ground.n_cells, t_road), np.full(hot.n_cells, t_pan)])
    flux = exchange.flux_w_m2(temps, q_lw)[exchange.cells_of(0)]
    rect = RadiantRectangle(
        centre_m=np.array([0.0, 0.0, 0.3]),
        u_axis=EX,
        v_axis=EY,
        half_u_m=0.25,
        half_v_m=0.25,
        emissivity=0.9,
    )
    expect = occluded_longwave_flux(
        patch_view_factors(ground, rect),
        t_pan,
        0.95,
        source_emissivity=0.9,
        longwave_down_w_m2=q_lw,
        sky_view=1.0,
    )
    assert np.sum(flux) == pytest.approx(np.sum(expect), rel=1e-2)
    assert np.max(np.abs(flux - expect)) < 0.03 * np.max(expect)
    assert flux.max() > 50.0, "the pan is hot enough to matter"


def test_what_one_body_loses_the_other_gains_to_1e6() -> None:
    """Black bodies: the power the road absorbs of the pan's emission is the power the pan
    emits toward the road, and vice versa -- exact because the traced factors are symmetrised
    into A_i F_ij = A_j F_ji before use."""
    ground, hot = road(6), pan(0.2, half_m=0.35, n=3)
    exchange = ObjectExchange(
        [ExchangeBody.from_planar("road", ground, 1.0), ExchangeBody.from_planar("pan", hot, 1.0)]
    )
    rng = np.random.default_rng(3)
    temps = np.concatenate(
        [rng.uniform(280.0, 290.0, ground.n_cells), rng.uniform(320.0, 360.0, hot.n_cells)]
    )
    absorbed = exchange.absorbed_power_w(temps)
    emitted = exchange.emitted_toward_power_w(temps)
    assert absorbed[0, 1] == pytest.approx(emitted[1, 0], rel=1e-6)
    assert absorbed[1, 0] == pytest.approx(emitted[0, 1], rel=1e-6)
    assert absorbed[0, 1] > 0.0 and absorbed[1, 0] > 0.0
    # and the per-cell flux integrates to the same power, less what the sky loses
    flux = exchange.flux_w_m2(temps, 0.0)
    power_in = np.sum(flux[exchange.cells_of(0)] * exchange.cell_areas_m2[exchange.cells_of(0)])
    assert power_in == pytest.approx(absorbed[0, 1], rel=1e-9)


def test_a_grey_body_reflects_what_it_does_not_absorb() -> None:
    ground, hot = road(4), pan(0.3, n=2)
    exchange = ObjectExchange(
        [ExchangeBody.from_planar("road", ground, 0.5), ExchangeBody.from_planar("pan", hot, 1.0)]
    )
    temps = np.concatenate([np.full(ground.n_cells, 285.0), np.full(hot.n_cells, 350.0)])
    absorbed = exchange.absorbed_power_w(temps)
    emitted = exchange.emitted_toward_power_w(temps)
    assert absorbed[0, 1] == pytest.approx(0.5 * emitted[1, 0], rel=1e-6)


# --- the lockstep --------------------------------------------------------------------------------


def _fields(
    exchange: bool, t_end: float = 3600.0
) -> tuple[dict[str, object], ObjectExchange | None]:
    ground, hot = road(8, 2.0), pan(0.25, half_m=0.6, n=3)
    n_r, n_p = ground.n_cells, hot.n_cells
    asphalt = PlanarThermalField(
        ground, _props(n_r, 2.0e5, 0.95), lambda t: _night(), 0.0, np.full(n_r, 281.0), 10.0
    )
    steel = PlanarThermalField(
        hot, _props(n_p, 5.0e6, 0.9), lambda t: _night(), 0.0, np.full(n_p, 310.0), 10.0
    )
    if not exchange:
        return {"road": asphalt, "pan": steel}, None
    group = ObjectExchange(
        [ExchangeBody.from_planar("road", ground, 0.95), ExchangeBody.from_planar("pan", hot, 0.9)]
    )
    return {"road": group.register("road", asphalt), "pan": group.register("pan", steel)}, group


def test_the_group_steps_every_member_together() -> None:
    fields, group = _fields(True)
    assert group is not None
    assert isinstance(fields["road"], ExchangedField)
    fields["road"].advance_to(600.0)  # through the proxy: the pan moves too
    assert fields["pan"].latest_t_s == 600.0 and fields["road"].latest_t_s == 600.0
    group.advance_to(1200.0)
    assert fields["pan"].latest_t_s == 1200.0
    # a member stepped on its own is refused before it can drift from its partners
    with pytest.raises(RuntimeError, match="through the group"):
        fields["pan"].inner.advance_to(1300.0)


def test_a_warm_pan_leaves_a_patch_on_the_road_beneath_it_and_only_with_the_exchange() -> None:
    """Under a clear night sky the road cools everywhere; with the exchange on, the cells under
    the warm pan see 310 K steel where the open cells see a 250 W/m² sky, and after an hour they
    stand more than a kelvin warmer. Off, every road cell is the same temperature -- the shape
    of a car on the asphalt is a consequence of the geometry, not something anyone authored."""
    off, _ = _fields(False)
    off["road"].advance_to(3600.0)
    flat = np.asarray(off["road"].temperature_at(3600.0), dtype=np.float64)
    assert np.ptp(flat) == 0.0

    on, group = _fields(True)
    on["road"].advance_to(3600.0)
    warm = np.asarray(on["road"].temperature_at(3600.0), dtype=np.float64)
    centres = on["road"].patch.cell_centres()
    under = (np.abs(centres[:, 0]) < 0.6) & (np.abs(centres[:, 1]) < 0.6)
    assert under.sum() >= 9 and (~under).sum() >= 20
    assert warm[under].mean() - warm[~under].mean() > 1.0, (warm[under].mean(), warm[~under].mean())
    # the pan's underside is seen at a grazing angle from beyond its footprint too, so the open
    # road is a little warmer than the road with no pan; at the far corners it is the same road
    corner = (np.abs(centres[:, 0]) > 0.9) & (np.abs(centres[:, 1]) > 0.9)
    assert np.all(warm[corner] < flat[0] + 0.3) and np.all(warm[~under] > flat[0] - 1e-9)
    assert group is not None and group.last_flux_w_m2["road"][under].mean() > 20.0


# --- the spin-up -------------------------------------------------------------------------------


def _engine_night(t_s: float) -> FacetForcing:
    """The night, plus 150 W/m² from an engine behind the pan: it stays warm all night."""
    return FacetForcing(
        t_air_k=280.0, h_w_m2_k=8.0, q_longwave_down_w_m2=250.0, q_internal_w_m2=150.0
    )


def _group_at(t0_s: float, init_k: float | None = None) -> tuple[dict[str, object], ObjectExchange]:
    """The road and a heated pan as one group starting at ``t0_s`` (from 280 K unless given)."""
    ground, hot = road(8, 2.0), pan(0.25, half_m=0.6, n=3)
    n_r, n_p = ground.n_cells, hot.n_cells
    start = 280.0 if init_k is None else init_k
    asphalt = PlanarThermalField(
        ground, _props(n_r, 2.0e5, 0.95), lambda t: _night(), t0_s, np.full(n_r, start), 10.0
    )
    steel = PlanarThermalField(
        hot, _props(n_p, 2.0e4, 0.9), _engine_night, t0_s, np.full(n_p, start), 10.0
    )
    group = ObjectExchange(
        [ExchangeBody.from_planar("road", ground, 0.95), ExchangeBody.from_planar("pan", hot, 0.9)]
    )
    return {"road": group.register("road", asphalt), "pan": group.register("pan", steel)}, group


def test_the_group_spin_up_is_the_exchanged_run_that_ends_at_t0() -> None:
    """The oracle is the lockstep itself: spinning the group up for two hours on a 10 s step
    gives, bit for bit, the state of the same group started two hours earlier from the air
    temperature and advanced to t₀. So the spin-up is the exchange, not an approximation of it."""
    spun, group = _group_at(0.0, init_k=300.0)  # the start it is given is thrown away
    states = group.spin_up(2.0, dt_s=10.0)
    ran, _ = _group_at(-7200.0)
    ran["road"].advance_to(0.0)
    for name in ("road", "pan"):
        expect = ran[name].field.latest_state_k
        assert np.array_equal(spun[name].field.latest_state_k, expect), name
        assert np.array_equal(states[name], expect)
        assert spun[name].latest_t_s == 0.0 and spun[name].n_ticks == 1


def test_a_spun_up_road_opens_the_scene_with_the_pan_already_on_it() -> None:
    """ADR 0157's limitation, closed: at t₀, before a single tick, the cells under the heated pan
    already stand warmer than the open road -- the night they spent under it is in their state."""
    fields, group = _group_at(0.0)
    group.spin_up(6.0, dt_s=60.0)
    t0 = np.asarray(fields["road"].temperature_at(0.0), dtype=np.float64)
    centres = fields["road"].patch.cell_centres()
    under = (np.abs(centres[:, 0]) < 0.6) & (np.abs(centres[:, 1]) < 0.6)
    assert t0[under].mean() - t0[~under].mean() > 1.0, (t0[under].mean(), t0[~under].mean())
    # and the run continues from there without a seam: the next tick moves by millikelvin
    fields["road"].advance_to(10.0)
    t1 = np.asarray(fields["road"].temperature_at(10.0), dtype=np.float64)
    assert np.max(np.abs(t1 - t0)) < 0.05


def test_a_spin_up_after_the_run_began_is_refused() -> None:
    fields, group = _group_at(0.0)
    fields["road"].advance_to(10.0)
    with pytest.raises(RuntimeError, match="already ticked"):
        group.spin_up(1.0)
    with pytest.raises(RuntimeError, match="only at t0"):
        fields["pan"].field.restart(np.full(9, 300.0))
    with pytest.raises(ValueError, match="positive"):
        _group_at(0.0)[1].spin_up(0.0)


# --- refusals ----------------------------------------------------------------------------------


def test_the_group_is_checked() -> None:
    ground = road(2)
    body = ExchangeBody.from_planar("road", ground, 0.95)
    with pytest.raises(ValueError, match="at least two"):
        ObjectExchange([body])
    with pytest.raises(ValueError, match="unique"):
        ObjectExchange([body, body])
    other = ExchangeBody.from_planar("pan", pan(0.3, n=1), 0.9)
    group = ObjectExchange([body, other])
    with pytest.raises(KeyError, match="not a body"):
        group.index("wall")
    wrong = PlanarThermalField(
        road(3), _props(9, 1e5, 0.9), lambda t: _night(), 0.0, np.full(9, 280.0), 10.0
    )
    with pytest.raises(ValueError, match="cells"):
        group.register("road", wrong)
    with pytest.raises(ValueError, match="before every body"):
        group.advance_to(10.0)
