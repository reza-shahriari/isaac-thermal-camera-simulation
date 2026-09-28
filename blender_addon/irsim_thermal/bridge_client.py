"""Run ``bridge.py`` under the irsim project interpreter and read its answer.

No ``bpy`` here: the add-on passes in the two settings it resolved (repository and interpreter),
and this module owns only the subprocess, its environment and the JSON protocol.
"""

import json
import os
import pathlib
import subprocess
import tempfile
from typing import Any

__all__ = ["BRIDGE", "BridgeError", "child_env", "run_bridge", "run_script"]

BRIDGE = pathlib.Path(__file__).resolve().with_name("bridge.py")


class BridgeError(RuntimeError):
    """The bridge refused (``kind`` says why) or could not be run at all (``kind == "setup"``)."""

    def __init__(self, message: str, kind: str = "setup") -> None:
        super().__init__(message)
        self.kind = kind


def child_env(repo: str) -> dict[str, str]:
    """The environment for the project interpreter, cleaned of what Blender leaked into it.

    Blender's own ``PYTHONHOME``/``PYTHONPATH`` would point the child at Blender's Python, and a
    snap-packaged Blender prepends its private library directories to ``LD_LIBRARY_PATH``; either
    breaks an unrelated interpreter in ways that surface as an import error far from the cause.
    """
    env = dict(os.environ)
    for key in ("PYTHONHOME", "PYTHONPATH", "PYTHONSTARTUP", "PYTHONNOUSERSITE"):
        env.pop(key, None)
    ld = env.get("LD_LIBRARY_PATH")
    if ld is not None:
        kept = [p for p in ld.split(os.pathsep) if p and "/snap/" not in p]
        if kept:
            env["LD_LIBRARY_PATH"] = os.pathsep.join(kept)
        else:
            env.pop("LD_LIBRARY_PATH")
    env["PYTHONPATH"] = str(pathlib.Path(repo) / "src")
    return env


def _check_setup(python: str, repo: str) -> None:
    if not repo or not (pathlib.Path(repo) / "configs" / "materials").is_dir():
        raise BridgeError(
            "The irsim repository is not set (or has no configs/materials). Set it in the "
            "add-on's preferences."
        )
    if not python:
        raise BridgeError(
            "No irsim Python interpreter is set. Set it in the add-on's preferences: the "
            "project's interpreter (Isaac Sim's python.sh), or any Python >= 3.10 with irsim "
            "installed."
        )
    exe = pathlib.Path(python).expanduser()
    if exe.is_absolute() and not exe.exists():
        raise BridgeError(f"The irsim Python interpreter {python} does not exist.")


def run_bridge(
    python: str,
    repo: str,
    command: str,
    payload: Any = None,
    extra_args: tuple[str, ...] = (),
    timeout_s: float = 180.0,
) -> dict[str, Any]:
    """One bridge call. Returns the result dict when ``ok``; raises :class:`BridgeError` if not."""
    _check_setup(python, repo)
    with tempfile.TemporaryDirectory(prefix="irsim_bridge_") as tmp:
        out = pathlib.Path(tmp) / "result.json"
        argv = [python, str(BRIDGE), command, "--repo", repo, "--out", str(out), *extra_args]
        if payload is not None:
            inp = pathlib.Path(tmp) / "payload.json"
            inp.write_text(json.dumps(payload), encoding="utf-8")
            argv += ["--in", str(inp)]
        try:
            proc = subprocess.run(
                argv,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                env=child_env(repo),
                cwd=repo,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise BridgeError(f"Could not run the irsim interpreter {python}: {exc}") from exc
        if not out.exists():
            tail = "\n".join((proc.stderr or proc.stdout or "").strip().splitlines()[-12:])
            raise BridgeError(
                f"The irsim interpreter exited with status {proc.returncode} and no answer."
                + (f"\n{tail}" if tail else "")
            )
        result: dict[str, Any] = json.loads(out.read_text(encoding="utf-8"))
    if not result.get("ok"):
        raise BridgeError(str(result.get("error", "unknown error")), str(result.get("kind", "")))
    return result


def run_script(
    python: str, repo: str, argv: list[str], timeout_s: float = 1800.0
) -> tuple[int, str]:
    """Run a repository script (``scripts/prep_asset.py``) and return ``(status, output)``."""
    _check_setup(python, repo)
    try:
        proc = subprocess.run(
            [python, *argv],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env=child_env(repo),
            cwd=repo,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise BridgeError(f"Could not run {argv[0]}: {exc}") from exc
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")
