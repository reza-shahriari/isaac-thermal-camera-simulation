"""Integration tests need a running Isaac Sim. Without it every test here is skipped, not errored.

Every test collected from this directory is marked `isaac` automatically (so the default
`-m 'not isaac and not gpu'` in pyproject deselects it) and, when selected with `make test-all`
on a machine without Isaac Sim, is skipped with an explicit reason.

The `simulation_app` fixture boots one headless Kit application for the whole session (~35 s on
an RTX A6000, ADR 0014). Engine modules (`omni.*`, `pxr`) are importable only after that boot, so
tests must import them inside the test body, never at module level, and must request the fixture.

**It boots through `simulation_app_config()`, so the suite obeys `$IRSIM_GPU` like every render
driver does.** It used to pass a bare `{"headless": True}`, which leaves Kit's multi-GPU render
graph on, and `irsim_isaac.env` records what that costs (measured 2026-09-16): a run that reached
`app ready` and then segfaulted with `ERROR_OUT_OF_DEVICE_MEMORY` because the second card held
26 GB of somebody else's job. This machine's second card is the one the owner works on, so an
unpinned integration run is not merely fragile, it reaches onto a GPU somebody is using.

NVIDIA Warp is the exception. It is provided by the `omni.warp.core` extension, but the extension
is a plain Python package and `irsim_isaac.env.ensure_warp_on_path` adds it to `sys.path`, so
`import warp` and both its `cpu` and `cuda:0` devices work with no Kit running (ADR 0014
addendum). A `gpu`-marked test therefore needs Warp and a CUDA device, not Isaac Sim, and is
selected on that basis; everything else here needs Isaac Sim and is skipped without it.

Shutdown: `SimulationApp.close()` ends in `os._exit`, which would kill pytest before it prints its
summary and would replace its exit status with 0. The fixture therefore does not close the app in
its teardown; it registers an `atexit` handler that closes with the status pytest reported in
`pytest_sessionfinish`. Handlers run last-in-first-out, so ours runs before the one SimulationApp
registered at construction.

Command line: `SimulationApp` forwards every argument it does not recognise to Kit, and Kit's own
parser aborts the process on pytest's options (`-m ""` is an "Ill formed parameter" → segfault in
`_start_app`), so the fixture hides `sys.argv` while the app boots.
"""

from __future__ import annotations

import atexit
import sys
from collections.abc import Iterator
from typing import Any

import pytest

from irsim_isaac.env import ensure_warp_on_path, has_isaac, has_warp, simulation_app_config

_session_exit = {"status": 0}


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    del config
    ensure_warp_on_path()
    no_isaac = pytest.mark.skip(reason="Isaac Sim not available (irsim_isaac.env.has_isaac())")
    no_warp = pytest.mark.skip(reason="NVIDIA Warp not available (irsim_isaac.env.has_warp())")
    warp_ok, isaac_ok = has_warp(), has_isaac()
    for item in items:
        item.add_marker(pytest.mark.isaac)  # keeps the default `-m 'not isaac'` deselect
        # A `gpu` test runs a Warp kernel and needs no Kit; everything else needs Isaac Sim.
        if item.get_closest_marker("gpu") is not None:
            if not warp_ok:
                item.add_marker(no_warp)
        elif not isaac_ok:
            item.add_marker(no_isaac)


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    del session
    _session_exit["status"] = int(exitstatus)


@pytest.fixture(scope="session")
def simulation_app() -> Iterator[Any]:
    """One headless Isaac Sim application per test session (closed at interpreter exit)."""
    if not has_isaac():
        pytest.skip("Isaac Sim not available")
    from isaacsim import SimulationApp

    argv = sys.argv
    sys.argv = argv[:1]  # Kit must not see pytest's options
    try:
        app = SimulationApp(simulation_app_config())
    finally:
        sys.argv = argv
    atexit.register(lambda: app.close(exit_code=_session_exit["status"]))
    yield app
