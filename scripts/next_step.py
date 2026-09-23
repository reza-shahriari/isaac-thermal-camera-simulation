#!/usr/bin/env python3
"""What to do next, computed from `docs/roadmap.md` rather than written into it.

    python scripts/next_step.py              # the one step to start now
    python scripts/next_step.py --queue 20   # the next 20, in order
    python scripts/next_step.py --lane PT    # restrict to one lane
    python scripts/next_step.py --write      # publish the queue into docs/roadmap.md
    python scripts/next_step.py --check      # fail if what is published has drifted
    python scripts/next_step.py --json       # the whole order, for tooling

The roadmap's picking rule -- "the first step in phase order whose deps are all ticked" -- does not
pick a step. Measured on revision 4 the day it landed: **51 of 109** open steps satisfied it at
once, 13 of them in phase 0 alone, and **67 of 109** open steps have no dependents at all, so the
dependency graph cannot order the majority of the plan. What was actually choosing the next step
was its row's position in the table, which is not a rule and is not stable under an edit.

This script is the rule. It is a **topological sort**, so a step never comes before something it
depends on, with ties broken by a total order:

1. **Promoted** (see :data:`PROMOTED`) -- the documented exceptions, and the only place judgement
   enters. The mechanical key counts dependents; it cannot see that a step prevents a recurring
   loss.
2. **Phase**, 0 → P → A → B → C → X. This is where the owner's ordering lives: repair, then the
   physics the owner's requirements need (per-point temperature from a config, heat between parts,
   water and fire — revision 6), then aerial, then maritime, then ground, with the cross-cutting
   lane last. A *dependency* may still pull an
   X step earlier -- `XD.1` unblocks eight -- and the topological sort does that on its own rather
   than needing a special case.
3. **Dependents, descending.** Transitive, not immediate: a step that unblocks ten outranks one
   that unblocks two.
4. **Size, ascending.** Between two steps of equal leverage, the small one first.
5. **Step id.** A total order, so two runs on the same file always agree.

The queue **is** published in `docs/roadmap.md`, between the `next:begin` / `next:end` markers, so
a reader opens the plan and sees what to start without running anything. The usual objection to a
pasted list is that it goes stale the moment a step is ticked, and a stale queue is worse than none
because it still answers — so it is *generated*, never hand-edited, and `--check` runs inside
`make check`. Drift fails the gate rather than misleading the next reader.

docs/roadmap.md ("Do next"); `tests/unit/test_roadmap_queue.py` is the guard.
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import re
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
ROADMAP = REPO / "docs" / "roadmap.md"

#: Phase order. `P` (revision 6) is the engine-free physics every scene lane needs and sorts right
#: after repair; `X` is the cross-cutting lane and sorts last; a dependency can still pull one of
#: its steps forward, which the topological sort handles without a rule of its own.
PHASE_RANK = {"0": 0, "P": 1, "A": 2, "B": 3, "C": 4, "X": 5}
SIZE_RANK = {"S": 0, "M": 1, "L": 2}

#: The documented exceptions, and the ONLY place judgement overrides the mechanical key. Each entry
#: needs a reason a reader can disagree with. Keep this list short: every entry is a claim that the
#: rule is wrong, and a long list means the rule is wrong.
PROMOTED: dict[str, str] = {
    "OC.3": (
        "Carries the focus lane the owner asked for on 2026-09-23, having found that no camera in "
        "this repo has a focus distance at all. The mechanical key sorts the lane last -- it is "
        "phase X and each step unblocks only its own lane -- but every frame the project has ever "
        "rendered is in perfect focus at every range, which is a statement about the camera that "
        "is not true of any camera. Move this entry to the lane's next open step as it advances."
    ),
    "RP.3": (
        "Makes stage_own_hunk.sh the default commit path. It unblocks nothing, so the mechanical "
        "key sorts it ninth -- but three sessions share this tree and a whole-file write has "
        "silently reverted committed work three times in two days (see the two `docs: restore ...` "
        "commits). Every session pays for it until it is fixed, and the cost is other people's "
        "shipped work."
    ),
}

STEP_ROW = re.compile(
    r"^\| ([A-Z]{2}\.\d+) \| (.*?) \| (.*?) \| ([^|]*) \| ([SML]) \| ([0-9A-Z]+) \|\s*$"
)
LANE_HEAD = re.compile(r"^## ([A-Z]{2}) — (.*)")


class Step:
    __slots__ = ("id", "lane", "lane_name", "what", "deps", "size", "phase", "done")

    def __init__(self, sid, lane, lane_name, what, deps, size, phase, done):  # noqa: PLR0913
        self.id, self.lane, self.lane_name = sid, lane, lane_name
        self.what, self.deps, self.size, self.phase, self.done = what, deps, size, phase, done


def load(path: pathlib.Path = ROADMAP) -> dict[str, Step]:
    """Parse the roadmap's step tables. The document is the source of truth, not a sidecar file."""
    steps: dict[str, Step] = {}
    lane = lane_name = None
    for line in path.read_text(encoding="utf-8").splitlines():
        head = LANE_HEAD.match(line)
        if head:
            lane, lane_name = head.group(1), head.group(2)
            continue
        row = STEP_ROW.match(line)
        if not row or lane is None:
            continue
        deps = tuple(d.strip(" `") for d in row.group(4).split(",") if d.strip(" `—-"))
        steps[row.group(1)] = Step(
            row.group(1),
            lane,
            lane_name,
            row.group(2).strip(),
            deps,
            row.group(5),
            row.group(6),
            row.group(2).startswith("✅"),
        )
    if not steps:
        raise SystemExit(f"no step rows found in {path}; has the table format changed?")
    return steps


def dependents(steps: dict[str, Step]) -> dict[str, set[str]]:
    """Immediate dependents, one entry per step."""
    out: dict[str, set[str]] = collections.defaultdict(set)
    for step in steps.values():
        for dep in step.deps:
            out[dep].add(step.id)
    return out


def leverage(steps: dict[str, Step]) -> dict[str, int]:
    """How many steps each one unblocks, transitively. The ordering signal that is not position."""
    children = dependents(steps)
    scores = {}
    for sid in steps:
        seen: set[str] = set()
        stack = [sid]
        while stack:
            for child in children[stack.pop()]:
                if child not in seen:
                    seen.add(child)
                    stack.append(child)
        scores[sid] = len(seen)
    return scores


def order(steps: dict[str, Step]) -> list[str]:
    """The queue: a topological sort with the documented total-order tiebreak.

    Raises on a cycle rather than emitting a partial queue -- a queue that silently omits steps is
    the failure this script exists to prevent.
    """
    score = leverage(steps)
    children = dependents(steps)
    pending = {sid for sid, s in steps.items() if not s.done}

    def key(sid: str) -> tuple:
        return (
            0 if sid in PROMOTED else 1,
            PHASE_RANK.get(steps[sid].phase, 9),
            -score[sid],
            SIZE_RANK[steps[sid].size],
            sid,
        )

    remaining = {sid: len([d for d in steps[sid].deps if d in pending]) for sid in pending}
    ready = sorted([s for s, n in remaining.items() if n == 0], key=key)
    queue: list[str] = []
    while ready:
        head = ready.pop(0)
        queue.append(head)
        for child in children[head]:
            if child in remaining:
                remaining[child] -= 1
                if remaining[child] == 0:
                    ready.append(child)
        ready.sort(key=key)
    if len(queue) != len(pending):
        stuck = sorted(pending - set(queue))
        raise SystemExit(f"dependency cycle among {len(stuck)} steps: {', '.join(stuck[:8])}")
    return queue


BEGIN = "<!-- next:begin -->"
END = "<!-- next:end -->"
PUBLISHED = 15


def render_block(steps: dict[str, Step], queue: list[str], score: dict[str, int]) -> str:
    """The generated queue, as it appears in the roadmap. Deterministic: same input, same bytes."""
    shown = queue[:PUBLISHED]
    head = steps[shown[0]]
    lines = [
        BEGIN,
        "<!-- generated by `python scripts/next_step.py --write`; do not edit by hand -->",
        "",
        f"### Start here → `{shown[0]}` · {head.lane_name.split('(')[0].strip()}",
        "",
        f"> {head.what[:300].rstrip()}",
        "",
        f"`{shown[0]}` is phase {head.phase}, size {head.size}, and unblocks "
        f"{score[shown[0]]} other step(s).",
    ]
    if shown[0] in PROMOTED:
        lines += ["", f"**Promoted over the mechanical order.** {PROMOTED[shown[0]]}"]
    lines += [
        "",
        f"#### Then, in order — {len(queue)} open steps",
        "",
        "| # | step | lane | phase | size | unblocks | waiting on |",
        "|---|---|---|---|---|---|---|",
    ]
    position = {sid: i for i, sid in enumerate(queue)}
    for i, sid in enumerate(shown, 1):
        st = steps[sid]
        blockers = [d for d in st.deps if d in position and position[d] < position[sid]]
        waiting = ", ".join(f"`{b}`" for b in blockers) if blockers else "ready"
        lines.append(
            f"| {i} | **`{sid}`** | {st.lane} | {st.phase} | {st.size} | "
            f"{score[sid] or '—'} | {waiting} |"
        )
    if len(queue) > PUBLISHED:
        lines += [
            "",
            f"…and {len(queue) - PUBLISHED} more — `python scripts/next_step.py --queue 40`.",
        ]
    lines += ["", END]
    return "\n".join(lines)


def publish(block: str, path: pathlib.Path) -> str:
    """Splice the generated block between the markers. Raises if they are missing or malformed."""
    text = path.read_text(encoding="utf-8")
    start, stop = text.find(BEGIN), text.find(END)
    if start < 0 or stop < 0 or stop < start:
        raise SystemExit(
            f"{path.name} has no intact {BEGIN} / {END} pair; the queue cannot be published "
            "without them, and guessing where it goes would be worse than failing"
        )
    return text[:start] + block + text[stop + len(END) :]


def _line(steps, score, sid, index=None):
    s = steps[sid]
    prefix = f"{index:3}. " if index else "     "
    deps = ", ".join(s.deps) if s.deps else "—"
    return (
        f"{prefix}{sid:<7} phase {s.phase}  {s.size}  unblocks {score[sid]:>2}  "
        f"[{s.lane}] deps {deps}"
    )


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--queue", type=int, nargs="?", const=15, metavar="N", help="show the next N")
    ap.add_argument("--lane", help="restrict to one lane, e.g. PT")
    ap.add_argument("--write", action="store_true", help="publish the queue into the roadmap")
    ap.add_argument("--check", action="store_true", help="fail if the published queue drifted")
    ap.add_argument("--json", action="store_true", help="the whole order as JSON")
    ap.add_argument("--roadmap", type=pathlib.Path, default=ROADMAP)
    args = ap.parse_args()

    steps = load(args.roadmap)
    score = leverage(steps)
    queue = order(steps)
    if args.lane:
        queue = [s for s in queue if steps[s].lane == args.lane.upper()]
        if not queue:
            print(f"no open steps in lane {args.lane.upper()}", file=sys.stderr)
            return 1

    if args.write or args.check:
        block = render_block(steps, queue, score)
        updated = publish(block, args.roadmap)
        if args.check:
            if updated != args.roadmap.read_text(encoding="utf-8"):
                print(
                    f"{args.roadmap.name}'s published queue has drifted from the roadmap's own "
                    "tables.\n  Run: python scripts/next_step.py --write",
                    file=sys.stderr,
                )
                return 1
            print(f"{args.roadmap.name}: published queue is current (head {queue[0]})")
            return 0
        args.roadmap.write_text(updated, encoding="utf-8")
        print(
            f"wrote {min(PUBLISHED, len(queue))} of {len(queue)} steps into {args.roadmap.name} "
            f"(head {queue[0]})"
        )
        return 0

    if args.json:
        print(
            json.dumps(
                [
                    {
                        "id": s,
                        "lane": steps[s].lane,
                        "phase": steps[s].phase,
                        "size": steps[s].size,
                        "unblocks": score[s],
                        "deps": list(steps[s].deps),
                        "promoted": PROMOTED.get(s),
                    }
                    for s in queue
                ],
                indent=1,
            )
        )
        return 0

    done = sum(1 for s in steps.values() if s.done)
    if args.queue:
        print(
            f"next {min(args.queue, len(queue))} of {len(queue)} open steps "
            f"({done} shipped), from {args.roadmap.name}\n"
        )
        for i, sid in enumerate(queue[: args.queue], 1):
            print(_line(steps, score, sid, i))
            if sid in PROMOTED:
                print(f"       promoted: {PROMOTED[sid]}")
        return 0

    head = queue[0]
    s = steps[head]
    print(f"next: {head}  [{s.lane} — {s.lane_name}]")
    print(f"  phase {s.phase}, size {s.size}, unblocks {score[head]} step(s)")
    print(f"  deps: {', '.join(s.deps) if s.deps else 'none'}")
    if head in PROMOTED:
        print(f"  promoted over the mechanical order: {PROMOTED[head]}")
    print(f"\n  {s.what[:400]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
