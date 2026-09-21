# ADR 0105 — The exhaust line as a quasi-1-D gas stream in a wall of network nodes

**Status:** Accepted
**Date:** 2026-09-21
Roadmap: TC.7 (§6.6, §6.4; spec issue S43)

## Context

§6.6 prescribes the manifold, the pipe and the tip as three schedules with their own ΔT_max and
τ, and the car scenes drive one of them (`exhaust_pipe`, τ 360 s) to put a stripe on the road.
A camera under a car sees what three schedules cannot make: a gradient along the line, hottest
at the manifold, cold at every hanger, moving with load; and after key-off a shielded manifold
and a catalyst shell that *warm* for a minute or two before they cool (MVFRI R04-13 measured
post-stop exhaust skins peaking 60–120 s after the stop and falling from ~400 °C to below
260 °C in 3–12 min). TC.5 (ADR 0100) replaced the bay's schedule with a solved network; this
does the same for the line, on the same network.

## Options considered

1. **Keep §6.6's schedules, add a fourth for the catalyst.** Cannot produce the gradient, the
   hangers, or the post-stop rise; the rise is the one signature every measurement has.
2. **A 1-D wall conduction solve along the pipe with the gas as a boundary.** Axial conduction
   in a 1.5 mm steel wall is ~0.01 W/K per segment against ~1 W/K of gas coupling: it does not
   set the profile, the gas does. Deferred as an addition (a link between neighbouring wall
   nodes), not a different model.
3. **The gas as a quasi-1-D stream marched segment by segment, each segment's wall a network
   node.** ``ṁ c_p dT_g/dx = −h_i π D (T_g − T_w)`` is exact over a segment with a uniform
   wall, ``T_out − T_w = (T_in − T_w) e^{−NTU}``, and the heat it leaves,
   ``ṁ c_p (T_in − T_out) = G_eff (T_in − T_w)`` with ``G_eff = ṁ c_p (1 − e^{−NTU})``, is a
   link the network already knows how to step: from the wall to a fixed node at the segment's
   inlet gas temperature, implicit in the wall and lagged one tick in the upstream gas. Chosen.

## Decision

Option 3, `irsim.thermal.exhaust_line`: `PipeSection`s in order (each ``n_segments`` wall
nodes of ``ρ c π D δ Δx``, outside convection forced while the car *moves* and natural when it
stands -- underbody air is ram air, not fan air -- radiation to the floor pan and the road,
hangers at ~1 W/K each), `GasFlow` (ṁ and inlet temperature linear in load between idle and
rated, Dittus–Boelter for h_i), an optional inner mass per section (a monolith or a silencer's
baffles: the gas touches it over its own area and the wall through the mat, plus the cones
where the gas still meets the shell) and an optional heat shield (a thin node across an air gap
by conduction and radiation; the camera sees the shield). `ExhaustLine` marches the gas on the
walls as they stand, then steps the network; `ExhaustSolver` is the `TemperatureSolver` the
scene's ``targets`` take as ``solver: exhaust`` with ``section:`` naming the reported skin;
`stock_exhaust()` is a mid-size petrol car's line in 22 segments, ESTIMATED throughout.

The flange to the head is a one-way boundary (the head's temperature is read; the engine does
not feel the line), and from a scene it is not wired at all: the engine is a separate solver
and targets cannot yet read each other. The runners' own cast section (~0.3 W/K each) rules
that conductance, which is why a manifold can run red while the head sits at coolant
temperature -- a 15 W/K gasket alone pinned the manifold at 185 °C.

## Consequences

**Measured:** the march reproduces ``exp(−NTU x/L)`` to 1e-6 over 24 segments; the gas leaves
at the wall temperature as h_i → ∞ and the wall goes to the inlet gas as ṁ c_p → ∞ too (with a
finite ṁ c_p it sits losses/ṁc_p · ΔT below, 15 K in the test); the network's energy residual is
below 1e-6 of the gas heat and the gas heat equals ṁ c_p (T_in − T_tail) to 1e-9. At 60 % load
moving: manifold wall 557 °C, downpipe 424, mid pipe 397, tailpipe 368 (section means), hung
segments 40–50 K colder than their neighbours, 13.7 kW left in the line. After key-off from that
run the manifold shield rises 287 → 347 °C peaking at +65 s and is below 260 °C at +380 s
(5.3 min from its peak); the catalyst shell rises 268 → 278 °C peaking at +65 s. At idle in a
car park nothing peaks after key-off (the outside air was already still).

**Not reached:** MVFRI's 400 °C shell. The stock converter's shell runs 270 °C at 60 % load and
370 °C at full load with no light-off exotherm and the gas on the shell only at the cones. The
fall is therefore timed from the shell's own peak; the manifold shield carries the 3–12 min
band. The ~100 °C clamp-probe spread MVFRI reports is the tolerance behind every band here.

**What it changes downstream:** the R2 car scenes keep §6.6's ``exhaust_pipe`` schedule. Switching
them to ``solver: exhaust`` puts the idling mid pipe at ~236 °C where the schedule had ambient
+ 48 K, a +150 K change to the road stripe that the car-demo tests and the R2 numbers were
measured on; that switch is its own step.

## Revisit when

A scene needs the flange wired to the engine target (targets reading targets, or the line as
nodes of the scene's own network), axial conduction in a thick manifold, the catalyst's
light-off exotherm, or the plume (PH.6) reading ``gas_out_k`` at the tip -- the tip's gas
temperature is already the line's last outlet.
