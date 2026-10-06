"""Record which data files each test module reads, so ``make check`` can rerun exactly those
tests when a data file changes (ADR 0193).

pytest-testmon selects tests by the *Python* code they execute. It cannot see that
``tests/unit/test_materials.py`` depends on ``configs/materials/asphalt.yaml`` or that a golden
test depends on its ``.npy``. This plugin closes that gap: with ``--track-reads`` it installs a
``sys.addaudithook`` that sees every ``open()`` of a file inside the repository (anything that
is not Python source) and every directory listing (``os.listdir`` / ``os.scandir``, which
``glob``, ``os.walk`` and ``Path.iterdir`` go through), and attributes each one to

* the test module whose test is running, for reads during a test or a function-scoped fixture;
* a ``module``/``class``/``package``/``session``-scoped fixture, for reads during its setup --
  the fixture is cached, so later modules that use it never trigger the read themselves, and
  are linked to it through the fixtures they request;
* the test module being collected, for reads its own top-level code makes on import;
* ``GLOBAL`` for reads made by a ``src/`` module's top-level code (once per process, so no
  single test owns them) and anything else outside a test. A change to a ``GLOBAL`` file makes
  ``scripts/affected_tests.py`` ask for the full suite. Directory listings importlib makes
  while finding modules are ignored.

At the end of a passing session the reads are merged into ``.testreads.json``: for every owner,
the fingerprint of each file it read (size, mtime, sha1) and of each directory it listed (its
sorted names), as they were when it passed. ``scripts/affected_tests.py`` compares those with
the tree. A module with a failing test is not updated, so its data changes stay "changed" until
it passes.

Known gap: a value cached at *module level in src/* and read by a second test module in the same
worker is attributed only to the first (``irsim.atmosphere.droplets._water_table`` is the one such
cache today). Union over runs narrows it; the full suite in CI (``make check-full``) is the
backstop.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parent.parent
STORE = ROOT / ".testreads.json"
SHARDS = ROOT / ".testreads-shards"
GLOBAL = "GLOBAL"
# Files outside any test's reach that still change what every test does.
ALWAYS_GLOBAL = ("pyproject.toml",)
_SKIP_PARTS = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".testreads-shards",
}
_SKIP_SUFFIXES = {".py", ".pyc", ".pyi", ".pth"}
_SKIP_NAMES = {
    ".testmondata",
    ".testmondata-shm",
    ".testmondata-wal",
    ".testreads.json",
    ".testreads.tmp",
}

_ROOT_STR = str(ROOT) + os.sep
_owner: str = GLOBAL
_active = False
# owner -> {"files": set[str], "dirs": set[str]}
_reads: dict[str, dict[str, set[str]]] = {}


def _rel(path: Any) -> str | None:
    """Repo-relative path of a tracked target, or None if it is not ours to track."""
    if isinstance(path, int):
        return None
    try:
        p = os.fsdecode(path)
    except TypeError:
        return None
    p = os.path.abspath(p)
    if not p.startswith(_ROOT_STR):
        return None
    rel = p[len(_ROOT_STR) :]
    parts = rel.split(os.sep)
    if _SKIP_PARTS.intersection(parts) or parts[-1] in _SKIP_NAMES:
        return None
    return rel.replace(os.sep, "/")


def _is_read(mode: Any, flags: Any) -> bool:
    if isinstance(mode, str):
        return "r" in mode or "+" in mode
    if isinstance(flags, int):
        return (flags & os.O_ACCMODE) != os.O_WRONLY
    return True


_SRC_STR = str(ROOT / "src") + os.sep


def _owner_of_this_read() -> str | None:
    """Who a read belongs to, from the call stack; None if it is the import system's own.

    A read made by a ``src/`` module's top-level code runs once per process, at whichever test
    happened to import it first, so it belongs to no single test: it is ``GLOBAL``. A directory
    listing made by importlib looking for a module is not a dependency at all.
    """
    f = sys._getframe(2)
    while f is not None:
        name = f.f_code.co_filename
        if name.startswith("<frozen importlib"):
            return None
        if f.f_code.co_name == "<module>" and name.startswith(_SRC_STR):
            return GLOBAL
        f = f.f_back
    return _owner


def _hook(event: str, args: tuple[Any, ...]) -> None:
    if not _active:
        return
    if event == "open":
        path, mode, flags = (args + (None, None, None))[:3]
        if not _is_read(mode, flags):
            return
        rel = _rel(path)
        if rel is None or os.path.splitext(rel)[1] in _SKIP_SUFFIXES or not os.path.isfile(path):
            return
        owner = _owner_of_this_read()
        if owner is not None:
            _reads.setdefault(owner, {"files": set(), "dirs": set()})["files"].add(rel)
    elif event in ("os.listdir", "os.scandir"):
        path = args[0] if args else "."
        rel = _rel(path if path is not None else ".")
        owner = _owner_of_this_read() if rel is not None else None
        # Outside any test this is pytest scanning for test files; a new test file is new to
        # testmon, which runs it, so the listing is not a reason to run everything.
        if rel is not None and owner is not None and owner != GLOBAL:
            _reads.setdefault(owner, {"files": set(), "dirs": set()})["dirs"].add(rel)


_sha_cache: dict[tuple[str, int, int], str] = {}


def file_fingerprint(rel: str) -> list[Any] | None:
    """[size, mtime_ns, sha1]; the hash is what is compared, size+mtime only skip re-hashing."""
    p = ROOT / rel
    try:
        st = p.stat()
        key = (rel, st.st_size, st.st_mtime_ns)
        if key not in _sha_cache:
            _sha_cache[key] = hashlib.sha1(p.read_bytes()).hexdigest()
        return [st.st_size, st.st_mtime_ns, _sha_cache[key]]
    except OSError:
        return None


def same_file(rel: str, recorded: list[Any] | None) -> bool:
    p = ROOT / rel
    try:
        st = p.stat()
    except OSError:
        return recorded is None
    if recorded is None:
        return False
    if [st.st_size, st.st_mtime_ns] == recorded[:2]:
        return True
    fp = file_fingerprint(rel)
    return fp is not None and fp[2] == recorded[2]


def dir_fingerprint(rel: str) -> str | None:
    p = ROOT / rel if rel else ROOT
    try:
        names = sorted(os.listdir(p))
    except OSError:
        return None
    return hashlib.sha1("\0".join(names).encode()).hexdigest()


class ReadTracker:
    """Registered by tests/conftest.py when ``--track-reads`` is given."""

    def __init__(self, config: pytest.Config) -> None:
        self.config = config
        self.worker = getattr(config, "workerinput", {}).get("workerid")
        # test module -> broad-scope fixture keys it requests
        self.module_fixtures: dict[str, set[str]] = {}
        self.failed_modules: set[str] = set()

    def pytest_configure(self) -> None:
        global _active
        sys.addaudithook(_hook)  # cannot be removed; it is inert while _active is False
        _active = True

    @pytest.hookimpl(hookwrapper=True)
    def pytest_make_collect_report(self, collector: pytest.Collector):  # type: ignore[no-untyped-def]
        """Importing a test module runs its top level: a table it loads there is its own read."""
        global _owner
        if not isinstance(collector, pytest.Module):
            yield
            return
        prev, _owner = _owner, f"module:{collector.nodeid}"
        try:
            yield
        finally:
            _owner = prev

    @pytest.hookimpl(hookwrapper=True)
    def pytest_fixture_setup(self, fixturedef: Any, request: Any):  # type: ignore[no-untyped-def]
        global _owner
        if fixturedef.scope == "function":
            yield
            return
        prev, _owner = _owner, f"fixture:{fixturedef.baseid}::{fixturedef.argname}"
        try:
            yield
        finally:
            _owner = prev

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_protocol(self, item: pytest.Item, nextitem: Any):  # type: ignore[no-untyped-def]
        global _owner
        module = item.nodeid.split("::", 1)[0]
        fixtures = self.module_fixtures.setdefault(module, set())
        for name, defs in item._fixtureinfo.name2fixturedefs.items():
            for d in defs:
                if d.scope != "function":
                    fixtures.add(f"fixture:{d.baseid}::{name}")
        prev, _owner = _owner, f"module:{module}"
        try:
            yield
        finally:
            _owner = prev

    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        if report.failed:
            self.failed_modules.add(report.nodeid.split("::", 1)[0])

    def _snapshot(self) -> dict[str, Any]:
        return {
            "failed": sorted(self.failed_modules),
            "reads": {o: {k: sorted(v) for k, v in r.items()} for o, r in _reads.items()},
            "module_fixtures": {m: sorted(f) for m, f in self.module_fixtures.items()},
        }

    def pytest_sessionfinish(self, session: pytest.Session, exitstatus: int) -> None:
        global _active
        _active = False
        if self.worker is not None:
            SHARDS.mkdir(exist_ok=True)
            (SHARDS / f"{self.worker}-{os.getpid()}.json").write_text(json.dumps(self._snapshot()))
            return
        shards = [self._snapshot()]
        for f in sorted(SHARDS.glob("*.json")) if SHARDS.exists() else []:
            shards.append(json.loads(f.read_text()))
            f.unlink()
        if exitstatus not in (
            pytest.ExitCode.OK,
            pytest.ExitCode.TESTS_FAILED,
            pytest.ExitCode.NO_TESTS_COLLECTED,
        ):
            return  # interrupted or broken: record nothing
        failed = {m for shard in shards for m in shard["failed"]}
        merge_into_store(
            shards, failed, reset=self.config.getoption("--track-reads-reset") and not failed
        )


def merge_into_store(shards: list[dict[str, Any]], failed: set[str], reset: bool) -> None:
    """Fingerprints are kept per owner -- what *that* owner last passed against -- so a file
    changed under two modules stays "changed" for the one that has not been rerun yet.

    A module with a failing test is not updated, so it stays selected until it passes, and
    neither is a shared fixture any failing module used (its read may be what broke it).
    ``GLOBAL`` reads (a ``src/`` module's import) do not depend on outcomes."""
    store: dict[str, Any] = {"owners": {}, "module_fixtures": {}}
    if not reset and STORE.exists():
        with contextlib.suppress(OSError, ValueError):
            store = json.loads(STORE.read_text())
    tainted = {
        fx
        for shard in shards
        for m, fxs in shard["module_fixtures"].items()
        if m in failed
        for fx in fxs
    }
    for shard in shards:
        for owner, r in shard["reads"].items():
            if owner.startswith("module:") and owner[len("module:") :] in failed:
                continue
            if owner in tainted:
                continue
            cur = store["owners"].setdefault(owner, {"files": {}, "dirs": {}})
            for f in r.get("files", []):
                cur["files"][f] = file_fingerprint(f)
            for d in r.get("dirs", []):
                cur["dirs"][d] = dir_fingerprint(d)
        for module, fx in shard["module_fixtures"].items():
            old = store["module_fixtures"].get(module, [])
            store["module_fixtures"][module] = sorted(set(old) | set(fx))
    g = store["owners"].setdefault(GLOBAL, {"files": {}, "dirs": {}})
    for rel in ALWAYS_GLOBAL:
        g["files"][rel] = file_fingerprint(rel)
    tmp = STORE.with_suffix(".tmp")
    tmp.write_text(json.dumps(store, sort_keys=True))
    tmp.replace(STORE)
