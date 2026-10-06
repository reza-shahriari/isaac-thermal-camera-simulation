"""A glTF vertex-colour tint reaches the USD as the colour it makes, or is reported untouched.

Blender's glTF importer multiplies ``COLOR_0`` into the base colour through a Mix node, which its
USD exporter cannot write: it falls back to the 0.8 grey socket default, and the DJI Matrice 100
(near-black, AI.23) rendered white in the Isaac RGB companion. ``prep_asset`` folds a tint that
is one colour across a material's faces into that material. The fold itself runs inside Blender;
what is pinned here engine-free is the rule it applies -- Blender's MULTIPLY formula and when a
set of samples counts as one colour -- since a wrong rule paints every asset the wrong colour
without failing anything else.
"""

from __future__ import annotations

import importlib.util
import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def prep():
    spec = importlib.util.spec_from_file_location("prep_asset", REPO / "scripts" / "prep_asset.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_matrice_100_body_folds_to_near_black(prep) -> None:
    # material_1 as imported: tint (0.79, 0.82, 0.93) x constant (0.0, 0.001, 0.003)
    rgba = prep.fold_multiply((0.79, 0.82, 0.93, 1.0), (0.0, 0.001, 0.003, 1.0))
    assert rgba == pytest.approx([0.0, 0.00082, 0.00279, 1.0], abs=1e-6)
    # what the exporter wrote without the fold: the default 0.8 grey, ~300x brighter
    assert max(rgba[:3]) < 0.8 / 100


def test_multiply_honours_the_mix_strength(prep) -> None:
    # Blender: a * (1 - fac + fac * b); fac 0 leaves the tint, fac 0.5 meets halfway
    assert prep.fold_multiply((0.5, 0.5, 0.5, 1.0), (0.0, 0.0, 0.0, 1.0), fac=0.0)[:3] == [0.5] * 3
    assert prep.fold_multiply((0.5, 0.5, 0.5, 1.0), (0.0, 0.0, 0.0, 1.0), fac=0.5)[:3] == [0.25] * 3


def test_alpha_is_the_tints_not_the_product(prep) -> None:
    assert prep.fold_multiply((1.0, 1.0, 1.0, 0.4), (0.5, 0.5, 0.5, 0.0))[3] == 0.4


def test_one_colour_within_half_an_8_bit_step_folds(prep) -> None:
    step = 1.0 / 255.0
    samples = [(0.3, 0.3, 0.3, 1.0), (0.3 + 0.4 * step, 0.3, 0.3, 1.0)]
    got = prep.uniform_colour(samples)
    assert got is not None and got[0] == pytest.approx(0.3 + 0.2 * step)


def test_a_tint_that_varies_does_not_fold(prep) -> None:
    # two 8-bit steps apart: a real per-face colour, which a single base colour would erase
    step = 1.0 / 255.0
    assert prep.uniform_colour([(0.3, 0.3, 0.3, 1.0), (0.3, 0.3 + 2 * step, 0.3, 1.0)]) is None
    assert prep.uniform_colour([(1, 0, 0, 1), (0, 0, 1, 1)]) is None


def test_no_samples_is_no_colour(prep) -> None:
    assert prep.uniform_colour([]) is None
