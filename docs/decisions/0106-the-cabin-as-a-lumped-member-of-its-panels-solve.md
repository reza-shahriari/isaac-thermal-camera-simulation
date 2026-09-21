# ADR 0106 — The cabin as a lumped member of its panels' coupled solve

**Status:** Accepted
**Date:** 2026-09-21
Roadmap: PT.15 (§6.6, §6.4)

## Context

ADR 0036 built `LumpedTwoNodeSolver` and ADR 0038 built `CabinNode`, and neither was reachable
from a scene: they were not importable from `irsim.thermal`, neither was a `solver:` kind, and
no config or demo constructed either. Every solved surface in every scene therefore had an
**adiabatic back** — for a road that is nearly right, and for a car roof it is not. ADR 0038
measured the difference at +4.8 K at noon, which is the difference between a parked car reading
like a parked car and reading like a painted slab lying on the ground.

`CabinNode` steps its panels itself, as lumps, with a midpoint step on the coupled system. A
scene's panels are **fields**: cells with their own shadows, their own sky view and their own
lateral conduction, stepped by the IMEX scheme of ADR 0094. Those are two different integrators
over the same balance, and a scene has to use one of them.

## Options considered

1. **Alternate: step the panel fields, then step `CabinNode` on their means.** ADR 0038 already
   measured what that costs — stable at 1 s and wrong at 60 s, in the direction that
   under-predicts the greenhouse, which is the effect the node exists to produce. Rejected for
   the reason ADR 0038 rejected it inside the module.
2. **Make the cabin a node of the `ThermalNetwork` (ADR 0096) and couple it to the fields.** The
   network and the fields are two solvers with two tick contracts; joining them needs exactly
   the lagged exchange option 1 was rejected for. (The engine does this deliberately, one way:
   a block that does not feel the bracket bolted to it.)
3. **Make the cabin a member of the panels' `CoupledFields` (ADR 0099).** A `LumpedMember` is
   one cell of unit area whose areal capacity *is* its J/K, whose convection coefficient *is*
   the infiltration conductance ṁ c_p to ambient, whose `q_internal` *is* the watts transmitted
   through the glazing, and whose ε = α = 0 so it neither radiates nor sees the sun. A
   `LumpedLink` joins every cell of a panel to it at `A_cell / R_p`. One sparse implicit
   operator then steps panels and air together, which is what ADR 0038 required. Chosen.

## Decision

Option 3. `irsim.thermal.coupling` gains `LumpedMember` and `LumpedLink`; `irsim.thermal.cabin`
gains `cabin_coupling` (the member and its links from a `CabinNode`) and `cabin_field` (the
panels and the cabin as one `CoupledFields`, spun up together so the cabin carries the days
before). Scene schema v10 adds `thermal.cabin:` — the panels by surface name with their inward
resistances, the glazing among them — and a surface's `back:`, which is §6.4's R₂d and T_deep
for a layered surface, the two-node substrate ADR 0036 defined and no config could declare.
`LumpedTwoNodeSolver`, `CabinNode`, `LayerStack` and their companions are exported from
`irsim.thermal`.

The glazing's solar inlet uses the **per-prim** irradiance on the glazing's own tilt, not its
cells' mean: it is the term §6.6 transmits, it is what `CabinNode` takes, and it keeps the
scene's cabin comparable to the module's to the tolerance below.

## Consequences

**Measured:** the coupled member reproduces `CabinNode.equilibrium` on ADR 0038's own panels to
0.021 K at a 2 s tick (0.105 K at 10 s, 0.005 K at 0.5 s — the splitting error of ADR 0094's
IMEX scheme, first order in the tick), and the roof there stands **4.84 K** above an adiabatic
back, which is ADR 0038's 4.8 K through the new path. `configs/scenes/parked_car_cabin.yaml`
then shows the mechanism on a schematic saloon: cabin 68.7 °C at local noon (a sealed car really
reaches 60–80 °C), roof 65.0 °C against 63.4 °C for the same paint at the same tilt with an
adiabatic back, glazing 37.3 °C (α_sol 0.10 transmits rather than absorbs), and at local
midnight the roof 3.8 K and the cabin 3.0 K below the air. Deleting the `cabin:` block leaves
the roof bit-identical to that adiabatic control, so the schema stays additive.

**Deviation:** the scene's roof gains **+1.6 K**, not ADR 0038's +4.8 K. The size of the boost
is set by how much of the cabin's boundary is in the sun: ADR 0038's panel set had both a roof
and a bonnet facing the sun, and this saloon's only sun-facing panel is the roof, with two
vertical doors and a windscreen draining the air. Same mechanism, same sign, a magnitude the
geometry chooses — recorded here rather than tuned away by adding panels the car does not have.

**What it costs:** a cabin's panels are solved as one system, so they share a tick and a
spin-up and cannot be advanced independently; a panel of a cabin may not be layered or carry a
film (the cabin joins the panel's one node), and the schema refuses both. The lumped member is
a cell in the field's arrays, so `temperature_at` over a coupled solve includes it —
`node_temperature_k` is the way to read it, and the scene exposes the whole solve under the
cabin's own name beside the panels' `PatchView`s.

## Revisit when

A cabin needs a floor over warm ground, a firewall to the engine bay (the engine network's block
is right there, one way), forced ventilation, or a panel that is both layered and a cabin wall —
each is a link or a member on this same operator rather than a new solver.
