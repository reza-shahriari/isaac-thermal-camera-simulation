---
name: ship-step
description: The completion checklist for finishing any unit of work in this repo — running the check gate, updating the technical report (TECHNICAL_REPORT.md) and CHANGELOG, adding a one-line feature note to the README only when the step is user-visible, writing an ADR when a decision was made, and committing with the project's Conventional Commits format. Use this skill EVERY time a piece of work is finished, before committing, and whenever the user says "commit this", "that's done", "next step", "ship it", or asks to wrap up. Do not reconstruct the process from memory — the technical-report and CHANGELOG updates are the parts most often skipped and they are what keep a multi-person project legible.
---

# Shipping a step

A step is not done when the code works. It is done when someone who was not in the conversation can tell
what happened and why.

## The gate

Run in order. Do not proceed past a failure.

```bash
make check        # lint + typecheck + the tests this change can affect (ADR 0193)
```

`make check` selects tests: pytest-testmon reruns those whose Python code changed, and
`scripts/affected_tests.py` adds the modules that read a changed data file (config YAML, CSV,
golden `.npy`, a document a test checks). It falls back to the whole suite when it has no
record or `pyproject.toml` changed. `python scripts/affected_tests.py --why` shows what the data
side picked and why.

Run **`make check-full`** (every test, all cores, a few minutes) instead when the step touches the
test machinery itself (`tests/conftest.py`, `tests/read_tracker.py`, `pyproject.toml`, the
Makefile), after a pull or rebase, and before any push. CI runs `make check-full` regardless.

If either fails, fix it. Never commit with `--no-verify`. Never disable a lint rule to get past
it without saying so in the commit body. Never deselect, skip or `-k` away a failing test to turn
`make check` green — a red test the selection found is a real result.

## The checklist

**1. Code**
- [ ] Implements one coherent thing. If it implements three, split the commit.
- [ ] Docstrings cite the physics spec where relevant: `# docs/physics-model.md §3.2`
- [ ] No engine imports in `src/irsim/` (there is a test; it should have caught this)
- [ ] No float16 anywhere temperature or radiance flows
- [ ] No new physical constants defined inline — they belong in `irsim.radiometry.constants`

**2. Tests**
- [ ] New physics has a test that would fail if the physics were wrong (see the `ir-sim-testing` skill)
- [ ] Tolerances expressed in physically meaningful units (mK, K, relative)
- [ ] Randomness seeded
- [ ] Any new test over a second is marked `slow` (the fast tier, `make test`, stays fast)

**3. Technical report**

All technical detail goes into `TECHNICAL_REPORT.md` — never into `README.md`. Update:
- the **status table** (which components exist, and at what validation tier), keeping each Notes
  cell to what the component does and how it is verified
- anything the change makes untrue (commands, layout, requirements, supported bands)
- the **Current limitations** section if the change added or removed one

This is the step most often skipped. A report that lags the code by three weeks is worse than none,
because people trust it.

**3b. README — only for a user-visible feature**

`README.md` is the public front page: a hero GIF, a few highlights, a short **What's new** list, a
quick start and links. It is written for someone deciding whether to use the project, not for
someone working on it. Do **not** put status rows, ADR numbers, step IDs, measurements, caveats or
implementation detail there.

If — and only if — the step adds something a user would notice (a new capability, band, scene
type, camera preset, tool), add **one plain-language bullet** at the top of **What's new**, e.g.
`- **Sunlit NIR and SWIR** — the ground reflects the sun and clouds scatter it.` Keep the list to
about five bullets; drop the oldest. If the feature has a good render, add a web-sized image or GIF
(a few MB at most) under `docs/media/` and show it in the relevant **Highlights** block. Refactors,
fixes, tests, docs and internal plumbing do not touch the README at all.

**4. CHANGELOG**

Add an entry under `## [Unreleased]`, in the appropriate subsection (`Added`, `Changed`, `Fixed`,
`Removed`). One line, written for a teammate, not for a compiler:

```markdown
### Added
- Band-integrated Planck LUT with inverse lookup (200-1000 K, 0.05 K, float32). Round-trip < 1 mK.
```

**5. ADR, if a decision was made**

Write `docs/decisions/NNNN-short-title.md` when the change involved a choice that outlives the commit:

- a solver, encoding, or algorithm chosen over alternatives
- a fidelity level deliberately skipped
- a tolerance loosened
- a dependency added
- an approximation whose error you cannot currently bound

Use the template in `docs/decisions/0001-record-architecture-decisions.md`. Keep it to a page. This
project involves a lot of "we chose the cheaper approximation on purpose", and without the reasoning
recorded, someone will later mistake a deliberate choice for a bug — or worse, mistake a bug for a
deliberate choice.

**6. The site**

Run `make site` and read its output. Most of what you just wrote reaches the project site on its own
— a module, a test file, an ADR, a config and a script are all published from their own docstrings,
so a missing docstring is now a blank cell on a public page. A **render or a clip is not
automatic**: name it in `site/gallery.yaml`, with captions whose numbers come from that run's own
`summary.json` or off its frames.

The build reports its page count, its media budget, any manifest entry this checkout could not
supply, and any dead link in the documents. `tests/unit/test_site_build.py` fails the gate on a dead
internal link, so a broken site shows up in step 1, not after publishing.

Do not push `gh-pages` unless the user asks. Full detail: the `present-on-the-site` skill.

**6b. Issues**

Follow the `issue-log` skill:
- If this step resolves an open issue, add `Closes #N` (or `Refs #N` for partial work) to the
  commit body.
- If this step found a bug it did not fix, added a limitation, or deferred a follow-up, draft an
  issue for it and show the drafts to the maintainer before the commit. Anything not approved yet
  goes into `docs/issues-queue.md`.
- Only real work, only the maintainer's account.

**7. Commit**

Conventional Commits. Scope from: `radiometry`, `materials`, `thermal`, `atmosphere`, `optics`,
`detector`, `noise`, `isp`, `config`, `isaac`, `spg`, `build`, `docs`.

```
feat(radiometry): add band-integrated Planck LUT with inverse lookup

Implements docs/physics-model.md §3.2 (b) and §3.3. LUT covers 200-1000 K at
0.05 K spacing, float32, linear interpolation. Inverse by binary search.

Round-trip error < 1 mK across the full range; agrees with direct Simpson
quadrature to < 2 mK equivalent (tests/unit/test_planck.py).

Refs: ADR 0004
```

Body should say **what physics it implements** and **what the verification showed**. "Added LUT" tells a
reviewer nothing they could not read from the diff.

Several sessions share this working tree. Never `git add -A`: a blanket add stages whoever else has
`TECHNICAL_REPORT.md`, `README.md`, `CHANGELOG.md` or `docs/roadmap.md` open, and their prose lands in your commit under your
message — this has actually happened, more than once. Stage named paths instead:

```bash
git add path/to/your_module.py tests/unit/test_your_module.py   # files only you touched
make stage FILES="TECHNICAL_REPORT.md CHANGELOG.md docs/roadmap.md"   # the shared files, if you
                                                                  # touched them and snapshotted first
git commit -F <message-file>     # or -m with a heredoc; keep the body
```

`make stage` runs `scripts/stage_own_hunk.sh stage` (three-way merges your edit onto current HEAD, so
a change someone else *committed* meanwhile is a no-op instead of getting overwritten) followed by
`scripts/stage_own_hunk.sh check` (refuses and explains if the staged content still isn't safe to
commit). It only works if you snapshotted the files *before* editing them:

```bash
scripts/stage_own_hunk.sh snapshot TECHNICAL_REPORT.md CHANGELOG.md docs/roadmap.md   # before you edit
```

The same check runs as a pre-commit hook (`.pre-commit-config.yaml`, install once with
`pre-commit install`) so a plain `git add` on a shared file is refused even if this step is skipped.
Full detail: TECHNICAL_REPORT.md's Contributing section.

Do not `git push` unless asked — the user may want to review or amend first.

## When to stop instead of shipping

Say so in the conversation rather than committing, if:

- the step turned out larger than expected and is half-tested → propose a split
- a test is failing and the fix is not obvious → do not loosen the tolerance to go green
- something about the Isaac Sim API turned out ambiguous → propose an experiment, do not guess
- the change contradicts `docs/physics-model.md` → raise the disagreement first; the spec may be wrong,
  but silently diverging from it is what makes a spec worthless

## Quick version

```
make check  →  TECHNICAL_REPORT status  →  CHANGELOG entry  →  README one-liner (user-visible only)
           →  ADR if a decision  →  make site  →  commit
```
