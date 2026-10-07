"""HU.10 -- the procedural walk's joint curves have a normal adult walk's ranges.

``scripts/animate_human.py`` keys the probe's walk from :func:`gait_angles`. The probe's verdict
does not depend on how natural the walk is, but its hand-travel criterion does: arms that barely
swing would fail it for the wrong reason, and a knee that bent backwards would deform the mesh in
a way no person does. The ranges are Winter's normative sagittal-plane gait (2009, ch. 3): hip
about -10 to +25 degrees, knee 5 to about 60 with the swing peak near 70 % of the cycle.

roadmap HU.10; ADR 0199.
"""

from __future__ import annotations

import importlib.util
import pathlib

import numpy as np
import pytest

_PATH = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "animate_human.py"
_spec = importlib.util.spec_from_file_location("animate_human", _PATH)
assert _spec is not None and _spec.loader is not None
animate_human = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(animate_human)


def _curves() -> dict[str, np.ndarray]:
    phases = np.linspace(0.0, 1.0, 401)
    rows = [animate_human.gait_angles(float(p)) for p in phases]
    return {k: np.array([r[k] for r in rows]) for k in rows[0]}


def test_the_walk_has_normal_ranges_and_never_bends_a_knee_backwards() -> None:
    c = _curves()
    assert c["hip"].min() == pytest.approx(-10.0, abs=0.5)
    assert c["hip"].max() == pytest.approx(25.0, abs=0.5)
    assert c["knee"].min() >= 4.0, "a knee never hyperextends"
    assert 55.0 <= c["knee"].max() <= 65.0
    peak = np.linspace(0.0, 1.0, 401)[int(np.argmax(c["knee"]))]
    assert 0.65 <= peak <= 0.78, "the knee peaks in swing, not in stance"
    assert c["elbow"].min() > 0.0


def test_the_cycle_closes_and_the_arm_swings_against_its_leg() -> None:
    c = _curves()
    for k, v in c.items():
        assert v[0] == pytest.approx(v[-1], abs=1e-9), k  # periodic: no jump at the loop
    # at heel strike the hip is flexed forward and the same side's arm is back
    a0 = animate_human.gait_angles(0.0)
    assert a0["hip"] > 20.0 and a0["shoulder"] < -10.0
