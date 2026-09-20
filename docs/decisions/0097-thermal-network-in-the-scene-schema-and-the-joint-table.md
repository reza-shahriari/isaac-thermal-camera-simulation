# ADR 0097 — The thermal network in the scene schema, and a joint table with provenance

**Status:** Accepted
**Date:** 2026-09-20
Roadmap: TC.4 (§6.4; spec issue S42)

## Context

TC.2 shipped `irsim.thermal.network` (ADR 0096), and it could be built only in Python. The
owner's bar for a lane is "a scene config plus one command", so the parts, the joints between
them and the heat they receive have to be declarable in the scene file — and the joint
conductances have to come from somewhere a reader can check. The survey found the numbers in
three places with three units: Voller & Tirovic's per-area contact conductances for bolted
ferrous joints (12 kW m⁻² K⁻¹ new, 7 corroded, 59 with paste), an engineering default of
1 kW m⁻² K⁻¹ for a dry bolted interface, and Hasselström & Nilsson's ~1 W/K per small bolt. A
scene author who types any of them by hand types the exponent wrong sooner or later — and a
joint at 1e-3 W m⁻² K⁻¹ renders as a plausibly cold bracket, not as an error.

## Options considered

1. **A `network:` block of its own beside `thermal:`.** The network shares the weather, the
   tick and the surfaces' names with the thermal block (a contactor in TC.3 joins a node to a
   patched surface), so a second block would duplicate `tick_s` and split the one place a
   reader looks.
2. **Typed link kinds** (`kind: contact | convection | radiation | ...`) with a per-kind
   parameter set. Correct but verbose, and the literature does not think in kinds: it reports a
   conductance per area or a conductance per fastener.
3. **One `LinkSpec` that takes exactly one of the forms the literature reports**, validated as
   such: a total `g_w_k`; `joint` (a table name) with `area_m2`; an inline `h_c_w_m2_k` with
   `area_m2`; `fastener` (a table name) with `count`; `h_w_m2_k` with `area_m2` for convection
   to a fluid node; or `radiation`. The joint and fastener names resolve against
   `configs/thermal/joints.yaml`.
4. **Joint values inline only, no table.** Every scene would restate Voller's 12 000 with no
   source beside it, and the range check would be the only defence.

## Decision

Options 3 and a table. Scene schema **v9**: `thermal.nodes:`, `thermal.links:` and
`thermal.sources:`, all optional, so every v4–v8 scene reads exactly as before.

* `NodeSpec` is exactly one of `capacity_j_k`, `mass_kg` + `specific_heat_j_kgk`, `fixed`
  (kelvin, or `"ambient"` for the scene's one weather series) or `link_node` (`a`, `b`,
  `g_w_k`, plus its own `capacity_j_k`). `initial_k` is optional and defaults to the weather's
  air at the scene start; a fixed node has none.
* `LinkSpec` takes exactly one form (above). An inline `h_c_w_m2_k` is range-checked exactly as
  the table is.
* `SourceSpec` is a constant `power_w` or a piecewise-linear schedule `times_s` → `power_w`,
  held at its ends, the same shape as `load_s` → `load` on a `vehicle_source`.
* Names must resolve at load; a source on a fixed node is refused at load ("goes nowhere").
* `configs/thermal/joints.yaml` (`irsim.config.joints`, schema 1): `joints` (h_c) and
  `fasteners` (G), each entry with `status: MEASURED | ESTIMATED` and a `source` of at least ten
  characters. The loader refuses h_c outside **1e2–1e6 W m⁻² K⁻¹** and G outside 1e-3–1e3 W/K.
  The small-bolt fastener is ESTIMATED on purpose: the measurement was in vacuum on aluminium.
* `Scene.network` is built by `build_network`; `Scene.advance_targets` steps it beside the
  targets and reports its nodes; `Scene.node_temperature_k` reads one. The network carries the
  weather so the one-weather guard (CLAUDE.md #6) sees it.

## Consequences

**What follows:** TC.5's engine is a `nodes:` entry with a `sources:` schedule and a coolant
fixed node; TC.6 migrates the car scenes' `vehicle_source` targets to nodes and links; TC.3's
contactor is one more link form (a node to a patched surface). PT.14's granularity tiers now
have three declarable kinds in one block: a surface, a patch, a node.

**What it costs:** a scene author must know the table's names (an unknown one fails at build,
naming the table's entries); the schedule form is linear between knots, so a step needs two
knots 0.1 s apart, as `load_s` already does.

**Measured:** the table's five joints and one fastener load with provenance; h_c of 1e-3, 50 and
2e6 are refused by the table loader and 1e-3 by the scene schema; a bracket through 25 cm² of
`bolted_ferrous_new` against `dry_default` on a 400 °C block, from the scene config alone,
settles at rises whose ratio matches the two-resistor closed form G/(G + hA) to 1e-6; every
link form and node kind builds to the conductance it declares; a v9 network beside a v7 surface
leaves the surface's solve bit-identical.

## Revisit when

TC.3 adds the contactor form (a link whose end is a patched surface), or a scene needs a
temperature-dependent joint (Voller's h_c rises with temperature), which is a callable
conductance the network already accepts and the schema does not yet spell.
