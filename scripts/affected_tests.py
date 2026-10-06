"""Which test modules must rerun because a *data* file they read has changed (ADR 0193).

``make check`` runs two selections, and the union is what a commit has to pass:

1. ``pytest --testmon`` -- the tests whose Python code (source or test) changed since they last
   passed. pytest-testmon does this from line-level coverage.
2. this script -- the tests that read a non-Python file (config YAML, spectral CSV, golden
   ``.npy``, a document a test checks) that changed, or listed a directory a file was added to
   or removed from, according to ``.testreads.json`` (written by ``tests/read_tracker.py``).

Prints one test-module path per line, or the single word ``ALL`` when only the full suite will
do: there is no record yet, or something every test depends on changed (``pyproject.toml``, a
file read at import/collection time).

    python scripts/affected_tests.py            # the list, or ALL
    python scripts/affected_tests.py --why      # ... and which file put each module on it
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))

from read_tracker import GLOBAL, STORE, dir_fingerprint, same_file  # noqa: E402


def stale(entry: dict[str, dict[str, object]]) -> list[str]:
    """The files and directories this owner read that are no longer as it saw them."""
    out = [f for f, fp in entry.get("files", {}).items() if not same_file(f, fp)]  # type: ignore[arg-type]
    out += [d + "/" for d, fp in entry.get("dirs", {}).items() if dir_fingerprint(d) != fp]
    return out


def affected() -> tuple[bool, dict[str, list[str]]]:
    """(needs_full_suite, {test module: [reasons]})."""
    if not STORE.exists():
        return True, {"(no .testreads.json yet)": []}
    try:
        store = json.loads(STORE.read_text())
        owners = store["owners"]
        module_fixtures = store["module_fixtures"]
    except (OSError, ValueError, KeyError):
        return True, {"(unreadable .testreads.json)": []}

    changed = {owner: why for owner, entry in owners.items() if (why := stale(entry))}
    if GLOBAL in changed:
        return True, {"(read outside any test)": changed[GLOBAL]}

    out: dict[str, list[str]] = {}
    for owner, why in changed.items():
        if owner.startswith("module:"):
            out.setdefault(owner[len("module:") :], []).extend(why)
    for module, fixtures in module_fixtures.items():
        for fx in fixtures:
            if fx in changed:
                out.setdefault(module, []).extend(
                    f"{w} (via {fx.split('::')[-1]})" for w in changed[fx]
                )
    return False, {m: sorted(set(w)) for m, w in sorted(out.items()) if (ROOT / m).exists()}


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--why", action="store_true", help="say which changed file selected each module"
    )
    args = ap.parse_args()
    full, modules = affected()
    if full:
        print("ALL")
        if args.why:
            for reason, files in modules.items():
                print(f"  # {reason}: {', '.join(files[:5])}", file=sys.stderr)
        return 0
    for module, why in modules.items():
        print(module)
        if args.why:
            print(f"  # {', '.join(why[:5])}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
