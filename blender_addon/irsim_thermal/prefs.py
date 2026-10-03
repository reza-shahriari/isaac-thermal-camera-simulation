"""Where the irsim repository is, and which Python can import it.

Both are machine settings, so they live in the add-on's preferences rather than in the ``.blend``.
Either may be left empty: the repository is then ``$IRSIM_REPO``, or found by walking up from this
file (which works when Blender loads the add-on straight out of the repository, as the tutorial
recommends), and the interpreter is ``$IRSIM_PYTHON``, the repository's ``.venv``, the project's
documented Isaac Sim interpreter (ADR 0002), or ``python3`` on the path.

The material picker's favourites and recently used materials (B13) are kept here too, so they
follow the person from file to file. When the add-on runs without being installed (the headless
smoke test) they live for the session only, on the window manager's picker settings.
"""

import os
import pathlib
import shutil

import bpy
from bpy.props import StringProperty
from bpy.types import AddonPreferences, Operator

from . import picker
from .bridge_client import BridgeError, run_bridge

PACKAGE = __package__ or "irsim_thermal"

#: ADR 0002: the project interpreter is Isaac Sim's bundled python.sh from a source build.
ISAAC_PYTHON_GUESS = "~/IsaacSim/_build/linux-x86_64/release/python.sh"


def _is_repo(path: pathlib.Path) -> bool:
    return (path / "configs" / "materials").is_dir() and (path / "src" / "irsim").is_dir()


def guess_repo_root() -> str:
    """``$IRSIM_REPO`` if it names a repository, else the checkout this file sits in, if any."""
    env = os.environ.get("IRSIM_REPO", "")
    if env and _is_repo(pathlib.Path(env).expanduser()):
        return str(pathlib.Path(env).expanduser())
    for parent in pathlib.Path(__file__).resolve().parents:
        if _is_repo(parent):
            return str(parent)
    return ""


def guess_python(repo: str) -> str:
    candidates = [os.environ.get("IRSIM_PYTHON", "")]
    if repo:
        candidates.append(str(pathlib.Path(repo) / ".venv" / "bin" / "python"))
    candidates.append(str(pathlib.Path(ISAAC_PYTHON_GUESS).expanduser()))
    for c in candidates:
        if c and pathlib.Path(c).expanduser().is_file():
            return str(pathlib.Path(c).expanduser())
    return shutil.which("python3") or ""


def _addon_prefs(context: bpy.types.Context) -> "IrsimPreferences | None":
    addon = context.preferences.addons.get(PACKAGE)
    return None if addon is None else addon.preferences


def settings(context: bpy.types.Context) -> tuple[str, str]:
    """``(repository, interpreter)``: the preferences where set, otherwise the guesses."""
    prefs = _addon_prefs(context)
    repo = bpy.path.abspath(prefs.repo_root) if prefs and prefs.repo_root else guess_repo_root()
    repo = repo.rstrip("/") if repo not in ("", "/") else repo
    python = (
        bpy.path.abspath(prefs.python_path) if prefs and prefs.python_path else guess_python(repo)
    )
    return repo, python


def _picks_store(context: bpy.types.Context):
    return _addon_prefs(context) or context.window_manager.irsim_picker


def picked(context: bpy.types.Context, which: str) -> list[str]:
    """The material names in ``which`` (``"favourites"`` or ``"recent"``)."""
    return picker.parse_names(getattr(_picks_store(context), which))


def set_picked(context: bpy.types.Context, which: str, names: list[str]) -> None:
    setattr(_picks_store(context), which, picker.join_names(names))
    if _addon_prefs(context) is not None:
        context.preferences.is_dirty = True  # saved with the preferences, like any other setting


class TestConnection(Operator):
    """Ask the irsim interpreter for the material library, to prove both settings work"""

    bl_idname = "irsim.test_connection"
    bl_label = "Test connection"

    def execute(self, context):
        repo, python = settings(context)
        try:
            result = run_bridge(python, repo, "library")
        except BridgeError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        self.report(
            {"INFO"}, f"OK: {len(result['materials'])} materials in {result['material_dir']}"
        )
        return {"FINISHED"}


class IrsimPreferences(AddonPreferences):
    bl_idname = PACKAGE

    repo_root: StringProperty(
        name="irsim repository",
        description="The irsim-scaffold checkout (the folder with configs/ and src/)",
        subtype="DIR_PATH",
        default="",
    )
    python_path: StringProperty(
        name="irsim Python",
        description=(
            "An interpreter that can import irsim: the project's (Isaac Sim's python.sh), "
            "or any Python >= 3.10 with the repository installed"
        ),
        subtype="FILE_PATH",
        default="",
    )
    favourites: StringProperty(
        name="Favourite materials",
        description="Library materials starred in the picker, comma-separated",
        default="",
    )
    recent: StringProperty(
        name="Recently used materials",
        description="The materials last assigned, most recent first, comma-separated",
        default="",
    )

    def draw(self, context):
        layout = self.layout
        repo, python = settings(context)
        col = layout.column()
        col.prop(self, "repo_root")
        if not self.repo_root:
            col.label(text=f"Using: {repo or 'not found'}", icon="INFO")
        col.prop(self, "python_path")
        if not self.python_path:
            col.label(text=f"Using: {python or 'not found'}", icon="INFO")
        col.operator("irsim.test_connection", icon="CHECKMARK")
        col.separator()
        col.prop(self, "favourites")


classes = (TestConnection, IrsimPreferences)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
