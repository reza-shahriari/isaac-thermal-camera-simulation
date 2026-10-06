---
name: issue-log
description: Keep the GitHub issue tracker as the honest, public log of real work on this repo — draft an issue whenever work turns up a bug, a limitation, a deferred follow-up or the next roadmap step; file it under the maintainer's own account only after they approve; and close it from the commit or PR that does the work ("Closes #N"). Use this skill whenever a bug or wrong number is found and not fixed in the same step, a TODO / "later" / "out of scope" / "follow-up" is said or written, a known limitation is added to TECHNICAL_REPORT.md, a roadmap step is started or finished, a step is being shipped (with ship-step), or the user says "make an issue", "log this", "track this", "open issues for the roadmap", or asks what is open. Never invent work to fill the tracker and never post as anyone but the maintainer.
---

# Issue log

The issue tracker is the public record of what this project is doing and why. A visitor should be
able to open it and see real work: what was found, what is planned, and which commit closed what.
That is only worth anything if every issue is **real** and **posted by who it says**.

## The two rules

1. **Only real work.** An issue exists because something actually happened in the work: a failing
   test, a wrong number, a limitation written down, a step deferred, a roadmap item. Never create an
   issue to make the tracker look busy, and never split one piece of work into many issues for
   volume.
2. **Only the maintainer's account.** File issues as the signed-in maintainer (or a clearly named
   bot the maintainer set up). Never help create, script or impersonate other accounts, and never
   write an issue phrased as if an outside user reported it. Inauthentic activity breaks GitHub's
   terms and can get the account and repository suspended. If asked to, decline and offer this
   workflow instead.

## When to draft an issue

| Trigger in the work | Issue type | Labels |
|---|---|---|
| A bug or wrong number found and **not** fixed in this step | Bug | `bug` + area |
| A test skipped, quarantined or failing for a reason outside this step | Bug | `bug`, `tests` |
| A limitation added to `TECHNICAL_REPORT.md` → Current limitations | Limitation | `limitation` + area |
| "Later", "out of scope", "follow-up", a TODO left in code | Follow-up | `enhancement` + area |
| A value marked `ESTIMATED` that real data could replace | Validation | `validation`, `help wanted` |
| A `docs/roadmap.md` step about to start | Roadmap | `roadmap` + area, milestone |
| A `docs/spec-issues.md` row left `open` | Physics | `physics`, `spec` |

Do **not** draft an issue for work finished in the same step; the commit is its own record.

Area labels follow the commit scopes in CLAUDE.md: `radiometry`, `materials`, `thermal`,
`atmosphere`, `optics`, `detector`, `noise`, `isp`, `config`, `isaac`, `spg`, `build`, `docs`.

Add `good first issue` only when the task is small, self-contained, needs no Isaac Sim, and the
issue body tells a newcomer exactly which files to touch and which test proves it. While the licence
does not accept pull requests (LICENSE §2, §4), still label them — they are ready for the day it
changes.

## Writing the issue

Write for a reader who was not in the conversation, the same standard as an ADR.

```markdown
**Title:** <imperative, specific: "Clamp sky emissivity above 1 at low elevation in humid presets">

## What
One or two sentences: what is wrong or missing.

## Why it matters
The physical consequence, in units (mK, %, W m⁻² sr⁻¹), not "it looks off".

## Where
- `src/irsim/...` (file and function)
- docs/physics-model.md §N.N
- Found while: <step id / commit hash / test name>

## Done when
- [ ] <checkable outcome>
- [ ] <the test that would fail if this regressed>
```

Rules: no secrets, tokens, hostnames, local paths under a home directory, or personal email
addresses in the body. Quote numbers from the actual run, not from memory. One issue, one problem.

## Approval, then filing

Never file without the maintainer's go-ahead in this conversation. The flow:

1. **Collect** drafts during the work. Do not interrupt a step to file them.
2. **Show** them at the end of the step (or when asked), as a short numbered list: title, labels,
   one-line reason. Offer the full bodies.
3. **On approval**, file each approved draft:
   - GitHub MCP connector (cloud sessions): `issue_write` with title, body, labels, milestone.
   - Locally: `gh issue create --title "..." --body-file <file> --label bug,optics --milestone v0.1.0`
4. **Not approved yet** → append the drafts to `docs/issues-queue.md` (create it if missing, see
   below) so nothing is lost between sessions. When a queued draft is later filed, delete it from
   the queue in the same commit.
5. **Report** the filed issue numbers and links back to the maintainer.

Before filing, search open issues for a duplicate (`gh issue list --search "<keywords>"` or the
connector's search). If one exists, add a comment with the new evidence instead of a new issue.

### `docs/issues-queue.md` format

```markdown
# Issue drafts awaiting review

<!-- Drafted by Claude during work; filed by the maintainer's account once approved. -->

## <title>
Labels: bug, optics · Milestone: v0.1.0 · Found in: <commit / step>

<body as above>
```

## Closing issues with the work

- The commit or PR that resolves an issue says so in its body: `Closes #12` (or `Fixes #12`).
  GitHub closes the issue when that commit reaches `main` and links the two both ways.
- Partial progress uses `Refs #12`, which links without closing.
- Put the keyword in the commit **body**, under the ship-step message, never as the whole subject.
- If work shows an issue was wrong or is no longer needed, close it with a one-line comment saying
  why (`not planned` / `duplicate of #N`). Never close an issue silently, and never close one the
  work did not actually resolve.

## Roadmap milestones

When the maintainer asks to "put the roadmap on GitHub" or plans a release:

1. Create the milestone (`gh api repos/{owner}/{repo}/milestones -f title=v0.1.0`) if missing.
2. Draft one issue per roadmap step in scope, titled with the step id
   (`AT.32 — Slant-path sky radiance for elevated cameras`), body linking the roadmap line.
3. Show the list for approval; file as above; put the issue number back into `docs/roadmap.md`
   next to the step so the two stay linked.

## Labels to create once

If a label is missing, list the missing ones and ask before creating them:

```bash
gh label create limitation  --color C5DEF5 --description "Known limitation of the current model"
gh label create validation  --color 0E8A16 --description "Needs comparison with measured data"
gh label create roadmap     --color 5319E7 --description "A step from docs/roadmap.md"
gh label create physics     --color D93F0B --description "Physics model or spec question"
gh label create spec        --color FBCA04 --description "docs/spec-issues.md row"
gh label create tests       --color BFD4F2 --description "Test suite or golden data"
```

Plus one per area scope above, if the maintainer wants them.
