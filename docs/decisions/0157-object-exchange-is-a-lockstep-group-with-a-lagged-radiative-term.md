# 0157 — Heat exchange between objects is a lockstep group with a one-tick-lagged radiative term

Date: 2026-09-28
**Status:** Accepted
Roadmap: TC.10 (§6.1, §6.6); builds on ADR 0088 (the net rule), ADR 0099 (coupled fields), ADR 0156 (traced factors)

## Context

A road under a parked car exchanged longwave with the sky and an air-temperature surround and
never with the car, unless a scene author hand-placed one of ADR 0088's parallel rectangles in
Python (`car_demo.ground_radiators`). The owner's requirement is one scene switch --
`thermal.object_exchange: true` -- after which every solved surface trades radiation with every
other by TC.9's traced factors, net of the sky each hides, for any shape; and off, every scene is
bit-identical to today (`thermal-realism-switches-are-requirements`).

## Options considered

1. **Merge every exchanged surface into one `ThermalField`**, ADR 0099's pattern for contactors
   and the cabin. Exact and implicit, but the radiation term is nonlinear (`T⁴` across bodies)
   and would either need a Newton step inside the IMEX scheme or a linearised conductance matrix
   rebuilt every tick; it would also force meshes and planar patches -- two field classes with
   different cell geometry -- into one solver object, which nothing else needs.
2. **Each field advances on its own and reads its partners' latest state.** No orchestration,
   but the answer depends on the order a driver calls the fields, and a member advanced ahead of
   its partners reads a future it should not have.
3. **A lockstep group with an explicit, one-tick-lagged exchange.** The group snapshots every
   member at the start of a tick, computes every exchange flux from that one snapshot, and steps
   each member exactly one tick with the flux added to its `q_internal`. The members keep their
   own solvers and their own IMEX steps.

## Decision

Option 3, as `irsim.thermal.object_exchange.ObjectExchange`, because the coupling is soft:
radiation's conductance `4 ε σ T³ · cover` is a few W m⁻² K⁻¹ against areal heat capacities from
10³ (a skin) to 10⁶ J m⁻² K⁻¹ (a road), so the explicit term's time constant is minutes to days
against a 1-10 s tick, and §6.4's explicit guard already bounds it. A bolted contact (ADR 0099)
is stiff and needed the merged solver; a facing surface is not and does not.

The flux per cell is

    q_c = ε_c · [ Σ_j G_cj ε_j σ T̄_j⁴  −  cover_c · q_lw,c ]

with `G` the traced cell-to-face factors, **symmetrised** first (`A_i F_ij = A_j F_ji` exactly,
by averaging the power matrix with its transpose) so that what one body loses the other gains to
rounding, not to the fan's resolution. The second term is ADR 0088's rule -- a body that radiates
onto a cell also hides what the cell saw there -- with the cell's own hemisphere-mean incoming
longwave standing for what each covered steradian replaces; for a road cell under an open sky
that is exactly the closed form's `L_down / V_s`.

Three further choices:

* **A member cannot be stepped past its partners.** The scene hands out an `ExchangedField`
  proxy whose `advance_to` moves the whole group, and a member's forcing asked for a tick the
  group has not snapshotted *raises* rather than returning a forcing with no exchange in it.
  A silently un-exchanged tick would be the worst kind of wrong: a plausible number.
* **The exchange begins at t₀.** Spin-up runs per field, as before, without the group; the first
  tick of the scene is the first exchanged one. A road that has stood under a car all night
  therefore opens the scene without the patch and grows it over the first hour. This is a known
  limitation, recorded in the report; the fix is a group-aware spin-up, which TC.12's freeze
  semantics will need anyway.
* **Single bounce, opaque faces.** What a face reflects of another's emission is lost from the
  pair (as in ADR 0088); a face's own body can shadow it; layered surfaces, cabin panels and
  prescribed maps are not members and a scene that switches the exchange on with one of them is
  refused with the surface's name, not silently left out.

**Verification** (`tests/unit/test_object_exchange.py`, `tests/unit/test_scene_object_exchange.py`):
a hot pan over a road reproduces `occluded_longwave_flux` with C-11's factors to 1 % in total and
3 % of the peak per cell; for black bodies the power the road absorbs of the pan's emission
equals the power the pan emits toward the road to 1e-6, and a grey body absorbs its ε share; a
proxy's `advance_to` moves every member and a member stepped alone is refused; under a clear
night sky a 310 K pan leaves the road beneath it more than a kelvin warmer than the open road
after an hour, and with the switch off every road cell is one temperature. At the scene level,
`object_exchange` absent and `false` give the same state hash, bit for bit.

## Consequences

* One key turns the effect on for a whole scene; nothing needs a hand-placed rectangle, and the
  shape a car leaves on asphalt is a consequence of its geometry.
* The first tick's exchange uses the spun-up state; a group-aware spin-up is owed (above).
* `car_demo`'s Python-authored radiators are untouched; they remain the exact answer for the
  parallel-rectangle sources they model, and the scenes that use them do not set the switch.
* Cost is TC.9's trace, once per scene at build (a 300-face pair in a minute), plus one
  matrix-vector product per tick.

## Amendment, 2026-10-03 — the group is spun up together

The "exchange begins at t₀" limitation is closed. `ObjectExchange.spin_up` runs §6.4's spin-up
for the whole group: the same integration as `facets.spin_up` (`spin_up_hours` of weather ending
at t₀, from the air temperature at the start, 60 s step, through the scene's weather wrap), with
the members in lockstep and each step's exchange flux, from one snapshot, added to every member's
`q_internal`. Each member's `ThermalField` is then restarted at t₀ from the result
(`ThermalField.restart`, legal only before the first tick). The scene does this whenever
`object_exchange: true`; with the switch off nothing changes, bit for bit.

Why not a separate switch: a group spun up apart and joined at t₀ is not a cheaper version of the
exchange. It is a state that no history could have produced, and the first hour of the scene is
a transient from it. With the switch on, the spin-up is part of what the switch means.

Not cached: the per-field spin-up cache is keyed on one material and the weather. A group key
would also need the geometry and the traced factors. The cost is one matrix-vector product per
60 s step, about 1,440 for a 24 h spin-up, which is small beside the build's trace.

Measured: the group spin-up equals, bit for bit, the same group started `hours` earlier from air
temperature and advanced to t₀ on the same step (`tests/unit/test_object_exchange.py`). In the
night scene of `tests/unit/test_scene_object_exchange.py` the road under the pan opens 0.58 K
warmer than the open road; before this amendment it opened at 0 K. After an hour the difference
is 0.54 K, against 0.25 K before.

Still open from this ADR: layered surfaces, cabin panels and prescribed maps are refused, not
joined. TC.11's full-object solve also starts from each part's separate spin-up, and so its
contacts and internal exchange begin at t₀. That limitation is the "per-part spin-up equalises
its contacts on the first tick" already recorded for TC.13.
