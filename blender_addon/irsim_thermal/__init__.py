"""irsim Thermal Materials: give every part of a model its infrared material, then hand it to irsim.

A Blender add-on (extension) for the human half of asset ingestion: open a downloaded model, pick
a part, choose what it is made of from the irsim material library -- or create a new material,
checked by irsim before it is written -- and export the model with its asset map. Plan:
``blender_addon/PLAN.md``; how to use it: ``blender_addon/TUTORIAL.md``.

Submodules are imported inside :func:`register` so that the pure-Python ones (``naming``,
``sizing``, ``coverage``, ``bridge_client``) can be imported and tested without Blender.
"""

_MODULES = (
    "prefs",
    "properties",
    "library_state",
    "assign",
    "material_form",
    "export",
    "ui",
)


def _modules():
    import importlib

    return [importlib.import_module(f"{__name__}.{name}") for name in _MODULES]


def register():
    for module in _modules():
        module.register()


def unregister():
    for module in reversed(_modules()):
        module.unregister()
