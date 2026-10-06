# ADR 0193 — The commit gate runs the tests a change affects, and CI runs them all

**Status:** Accepted
**Date:** 2026-10-06

## Context

`make check` was lint + mypy + the whole unit and golden suite in both tiers, one test at a time.
On 2026-10-06 the suite was 5,699 tests. The *fast* tier alone (5,262 tests, `-m "not slow"`)
took **14 min 20 s** on the 32-core workstation, against the 30-second budget the `slow`
marker (GT.1) was written to protect. The slow tier came on top of that. Lint (0.03 s) and
mypy with a warm cache (0.5 s) were negligible; nearly all the time was pytest, on one core.
The 15 slowest "fast" tests (end-to-end thermal solves, stage builds in session fixtures) took
about 410 s between them.

Several sessions commit to this repository each day, each paying that cost per commit. Most of
those tests cannot be affected by a given commit: a change under `configs/humans/` cannot alter
the car-on-road exchange solve. The one property that makes a full run worthwhile here is that
the physics core is shared. A change to Planck, a LUT, a material or the solver can break a test
far from the edit, so whatever replaces the full run must follow real dependencies, not file
names.

## Options considered

1. **Parallelism only** (pytest-xdist). It changes nothing about which tests run. The full suite
   (both tiers) drops to about 5 min 45 s, bounded by its longest single test (about 150 s
   under load). That is still minutes per commit.
2. **Select by file name or import graph.** Cheap, but it misses dependencies through data. A
   test that reads `configs/materials/*.yaml` does not import it.
3. **Select by coverage (pytest-testmon) plus a record of data-file reads, full suite in CI.**
   testmon reruns the tests whose executed Python lines changed. It is blind to non-Python
   inputs, so a small plugin (`tests/read_tracker.py`) records, through `sys.addaudithook`,
   every repository file each test module opens and every directory it lists. It attributes a
   read in a non-function-scoped fixture to that fixture, and links each module to the
   fixtures it requests. `scripts/affected_tests.py` compares those records with the tree.
4. **Mark more tests `slow` and keep a full run per commit.** This is worth doing anyway, but
   the slow tier still runs per commit, so the commit cost stays.

## Decision

Option 3, with option 1 underneath every target.

* `make check` is lint + mypy + `test-affected`. That target runs the testmon selection, then the
  modules `scripts/affected_tests.py` names (data changed under them), each with `-n auto`. The
  testmon run uses `--testmon-forceselect`, because plain `--testmon` switches its selection off
  whenever `-m` is given, and `pyproject.toml`'s `addopts` always gives one. The first version
  missed this and ran every test.
  It runs the full suite instead when there is no record yet, or when something every test
  depends on changed: `pyproject.toml`, or a file read outside any test (at import or
  collection).
* `make check-full` is lint + mypy + every test, and rebuilds both records. It is the gate for a
  push, and the thing to run after a pull or rebase or whenever the selection is in doubt. CI
  (`.github/workflows/check.yml`) and `make ci` run it.
* Every pytest target runs on all cores (`-n auto --dist loadfile`) with one BLAS thread per
  worker. `loadfile`, not `worksteal`: some modules share a module-level RNG
  (`RNG = np.random.default_rng(...)`) or a module-scoped scene that each test steps forward in
  time. Splitting such a module across workers changes the draws, or asks a scene for a time it
  has already moved past. Under `worksteal`, four tests failed this way
  (`test_target_statistics`, `test_tier4_comparison`, `test_cabin_scene`, `test_quad_pointwise`).
  `loadfile` runs each module in file order, as a serial run does. testmon also sorts what it
  selects by duration, so `tests/conftest.py` restores collection order after it.
* Records are per machine and gitignored: `.testmondata` and `.testreads.json`. A module with a
  failing test is not marked as passed against its inputs, so it stays selected until it passes.

## Consequences

* Measured on 2026-10-06 (32 cores, other sessions' jobs running):

  | Run | Before | Now |
  |---|---|---|
  | Full suite, both tiers (`make check-full`) | more than 14 min 20 s (fast tier alone) | 7 min 10 s |
  | `make check`, nothing changed since the last pass | same | 3.8 s (reruns only the two tests that last failed) |
  | `make check` after an edit to `isp/dde.py` | same | 8 s, 22 tests |
  | `make check` after an edit to one material YAML | same | 5 min 47 s, 92 modules (1,351 tests) |

  The last row is the selection working as intended. The material library feeds most scene
  solves, so a material edit really does reach them. The first `make check` in a fresh clone is
  a full run.
* **Error introduced: a commit can pass `make check` while breaking a test the selection did
  not pick.** Known ways it can happen:
  * A value cached at module level in `src/` and served to a second test module from that cache
    in the same worker. The tracker attributes the read only to the first module.
    `irsim.atmosphere.droplets._water_table` is the one such cache today.
  * A test that reaches a file through a subprocess or a C extension that opens the path itself.
  * Behaviour that depends on something outside the repository (installed packages, the
    environment). testmon tracks package versions; the environment is not tracked.
  * Nondeterminism.

  None of these can be bounded in advance. The backstop is that `make check-full` runs on every
  push and PR, so such a break is caught before anyone else pulls it, not after.
* Tests must not share files across modules. Within a module, file order still holds
  (`loadfile`), but a selected subset skips earlier tests, so a test must not need an earlier
  test to have run. The 2026-10-06 runs passed except for failures that also occur serially
  (other sessions' in-flight work, and a localhost request that this machine's HTTP proxy
  intercepts).
* The rule that the fast tier stays under 30 s still holds for `make test`, and a test over a
  second is still marked `slow`. Selection does not excuse a slow test.
* The audit hook is installed only with `--track-reads`, and does nothing outside repository
  paths.

## Revisit when

* A break that the selection missed reaches CI. Note how it got through. If the cause is a new
  class (not one listed above), widen the tracker or narrow the selection.
* testmon gains native tracking of data files. The read tracker then becomes redundant.
* The full suite gets back under a minute on all cores. Selection is not worth its complexity
  below that.
