# ADR 0093 — Tick history: a ring for spatial fields, a hook for whoever wants the history

**Status:** Accepted
**Date:** 2026-09-20
Roadmap: PT.8 (§6.4)

## Context

`ThermalField` (M6.11) solves on a fixed tick and answers render-time queries by blending the two
ticks that bracket the query. It appended every tick to a list and pruned none. For the per-prim
field that is a few facets times a few thousand ticks and it never mattered. ADR 0087 then put the
same object under a `PlanarPatch`, and the car scenes' road patch — 100 × 104 cells — held
2 880 × 10 400 × 8 B ≈ **240 MB** after a day at a 30 s tick. The coupling lane (`TC`) brings fields
of 10⁵ cells, at which point a diurnal run is gigabytes, and ADR 0074's full-diurnal time-lapse
was blocked on exactly this.

The history was also load-bearing in two places that are easy to miss: `state_hash`, the guard
that a query never mutates the solve, walked every held tick; and the diurnal validations
(`scripts/validate_thermal_diurnal.py`, `test_tier3_thermal`, `test_thermal_scene`) advance the
per-prim field to the end of a day and then read it back hour by hour.

## Options considered

1. **Prune nothing; tell callers to make smaller fields.** Leaves the time-lapse blocked and makes
   the cell count a memory decision rather than a thermal-gradient one, which ADR 0087 says it must
   not be.
2. **A two-tick ring for every field.** Bounded, and breaks the diurnal validations, which would
   each need rewriting around a recording hook for no physics gain.
3. **A ring whose length is a constructor choice, defaulting by field kind**, with a running hash
   and a recording hook. Spatial fields default to the bracketing pair; per-prim fields keep
   everything as before; anyone who wants a history records it.

## Decision

Option 3.

* `ThermalField(..., keep_ticks=None, on_tick=None)`. `keep_ticks` bounds the held ticks to a
  `deque(maxlen=keep_ticks)`; `None` keeps all, and is the per-prim default, so every existing
  scene and validation is unchanged. Fewer than two is refused: a query is a blend of two ticks.
* `PlanarThermalField` defaults to `keep_ticks=DEFAULT_KEEP_TICKS = 2`. A renderer advances to a
  tick and asks for that tick — many times, per band and per pass — and never for last hour.
* `state_hash` is a **running SHA-256** fed at each tick with the same bytes in the same order the
  old walk used, so its digest is byte-identical to the unbounded field's and does not depend on
  how many ticks are held. `test_tick_history` reconstructs the walk from the hook and checks.
* `on_tick(t_s, temperatures_k)` fires for every tick produced, including t₀, in order. It is how
  a time-lapse writer or a validation gets the history without the field holding it.
* `n_ticks` now counts ticks **produced**; `n_held` counts ticks resident. They are equal only for
  an unbounded field. The one existing use of `n_ticks` ("unchanged by a query") reads the same.
* A query before the oldest held tick **raises**, naming `keep_ticks` and `on_tick`. Answering
  from the oldest tick or a fallback would hand a time-lapse a temperature the solve never held at
  that time, and it would look like physics.

## Consequences

**Easy:** the road patch after a day holds 166 KB; a 10⁵-cell field holds 1.6 MB; ADR 0074's
time-lapse can run a whole day. The interpolation is unchanged — 500 queries inside the window are
bit-identical to the unbounded field's answers.

**Hard:** a spatial field can no longer be advanced to the end and read back in time. That was
never done anywhere in the repository, but it is the natural way to write a Tier 3 sweep over a
patch, and the first author to try it will hit the error. The message says what to do.

**Error introduced:** none in the physics; the solve, the tick and the blend are untouched.

## Revisit when

A consumer needs random access into a spatial field's history — a replay tool, or a Tier 4
comparison against a timestamped public sequence — at which point `on_tick` writing to a
memory-mapped float32 store is the shape, not a larger ring.
