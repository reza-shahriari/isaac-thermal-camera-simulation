"""Make ``irsim_thermal`` importable without Blender (its ``__init__`` defers every ``bpy`` import).

These tests are the add-on's own and are not part of ``make check``: the add-on is planned and
gated separately from the physics project (``blender_addon/PLAN.md``). Run them with the project
interpreter, which also has ``irsim`` for the bridge tests::

    <python> -m pytest blender_addon/tests -q
"""

import pathlib
import sys

ADDON_ROOT = pathlib.Path(__file__).resolve().parents[1]
REPO = ADDON_ROOT.parent
if str(ADDON_ROOT) not in sys.path:
    sys.path.insert(0, str(ADDON_ROOT))
