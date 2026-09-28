"""SC.29 -- every render driver is shutter-referenced, and retries without the flat field.

`render_phantom4.py` built its `PipelineConfig` bare while the other eight drivers enabled the
two-point flat field and caught the `ValueError` a photon camera raises when the hot calibration
point is outside its ADC. The Phantom 4 clips therefore showed the housing bowl §8.2 predicts and
a Boson's FFC removes: on frame 96 the dn16 sky rose 3235 → 4465 DN from centre to corner while
`apparent_t` stayed flat, and the AGC stretched that into 90 display codes of vignette.

`test_render_multiband.py` guards the drivers `render_multiband` sweeps; this walks **every**
``scripts/render_*.py`` so a new driver cannot repeat the omission. docs/physics-model.md §8.2,
§11.2; ADR 0053; roadmap M9.12, SC.29.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

SCRIPTS = pathlib.Path(__file__).resolve().parents[2] / "scripts"
#: Every driver that builds a pipeline. `render_multiband.py` sweeps the others and builds none.
DRIVERS = sorted(s for s in SCRIPTS.glob("render_*.py") if "from_sensor(" in s.read_text("utf-8"))


def _from_sensor_calls(tree: ast.AST) -> list[ast.Call]:
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "from_sensor"
    ]


@pytest.mark.parametrize("script", DRIVERS, ids=[s.name for s in DRIVERS])
def test_every_render_driver_enables_the_flat_field(script: pathlib.Path) -> None:
    """A `from_sensor` call with no `flat_field_enabled` keyword is the un-shuttered picture."""
    tree = ast.parse(script.read_text("utf-8"))
    calls = _from_sensor_calls(tree)
    assert calls, f"{script.name} builds no PipelineConfig"
    enabled = [
        call
        for call in calls
        for kw in call.keywords
        if kw.arg == "flat_field_enabled"
        and not (isinstance(kw.value, ast.Constant) and kw.value.value is False)
    ]
    assert enabled, f"{script.name}: no from_sensor call enables the flat field (SC.29)"


@pytest.mark.parametrize("script", DRIVERS, ids=[s.name for s in DRIVERS])
def test_every_render_driver_retries_without_it(script: pathlib.Path) -> None:
    """The refusal a photon camera raises must produce a frame with no flat field, not no frame."""
    tree = ast.parse(script.read_text("utf-8"))
    disabled = [
        call
        for call in _from_sensor_calls(tree)
        for kw in call.keywords
        if kw.arg == "flat_field_enabled"
        and isinstance(kw.value, ast.Constant)
        and kw.value.value is False
    ]
    handlers = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ExceptHandler)
        and getattr(node.type, "id", "") == "ValueError"
        and any(call in ast.walk(node) for call in disabled)
    ]
    assert handlers, f"{script.name}: no ValueError retry without the flat field"


def test_the_phantom4_driver_is_among_them() -> None:
    """The one that was missing."""
    assert any(s.name == "render_phantom4.py" for s in DRIVERS)
