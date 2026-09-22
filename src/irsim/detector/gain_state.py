"""The intrascene ceiling: what a camera can hold in one frame before it rails (roadmap PH.8).

docs/physics-model.md §9.2, §11.3; ADR 0091 (the Boson's published figures), ADR 0116 (this).

A thermal camera does not have one dynamic range, it has a **gain state**. FLIR's Boson (Rev 340)
publishes two: *high gain* holds a scene up to about 140 °C and *low gain* up to about 500 °C,
with the same 16-bit converter behind both -- so high gain buys resolution and low gain buys
headroom, and switching is a decision the camera or its operator makes about what the scene
contains. Every camera in this repository has run, in effect, in a state with no ceiling at all,
which is fine until a scene contains fire. `PH.7` put fire in a scene.

**The ceiling is a radiance, and applying it in kelvin is a different operation.** For one pixel
in isolation the two agree -- ``L_B`` is monotone, so clipping ``T`` at 500 °C and clipping ``L``
at ``L_B(500 °C)`` pick the same pixels. They stop agreeing the moment the plane is a *scene*
rather than a temperature map, because what reaches the detector is
``ε L_B(T) + (1 − ε) L_env`` plus whatever the path adds, and that is not ``L_B`` of anything:

* a **ε = 0.4** flame at 1273 K delivers well under half a blackbody's radiance and does **not**
  rail a low-gain core, where a ceiling applied to its temperature would cap it at 500 °C and
  invent a rail that is not there;
* a **cold, shiny** surface reflecting a flame can rail without ever being hot, which a ceiling
  applied to temperature cannot express at all.

So the clip lives on the at-aperture radiance plane and never on a temperature. The test for it is
the first case, because it fails loudly in the direction a scene author would not question -- an
image with a saturated flame in it looks exactly like an image with a saturated flame in it.

**What it does not cap.** The ISP's AGC is a *display* stretch and the radiometric branch does not
pass through it (ADR 0031), so a railed frame still yields the camera's own radiometric estimate
up to the rail -- which is the shape FLAME 3's histogram has, a room-temperature mode and a tail
piled against the ceiling. What the rail *does* destroy is scene contrast under a linear AGC, and
that is `PH.8`'s other half.
"""

from __future__ import annotations

from typing import Final

import numpy as np
from numpy.typing import NDArray

from irsim.radiometry.lut import BandLUT, Quantity

__all__ = [
    "BOSON_GAIN_CEILING_K",
    "clip_to_gain_ceiling",
    "gain_ceiling_radiance",
]

#: FLIR Boson Rev 340, "Scene Dynamic Range": high gain −40 to +140 °C, low gain −40 to +500 °C.
#: The datasheet quotes the *scene* temperature, which is why the ceiling is authored in kelvin
#: and converted here rather than being authored as a radiance nobody could check.
BOSON_GAIN_CEILING_K: Final[dict[str, float]] = {
    "high": 273.15 + 140.0,
    "low": 273.15 + 500.0,
}


def gain_ceiling_radiance(ceiling_k: float, lut: BandLUT, quantity: Quantity = "lb") -> float:
    """The band radiance a blackbody at the ceiling puts out -- the plane's clip level.

    The ceiling is defined against a *blackbody* scene, as the datasheet's "scene dynamic range"
    is: it is the hottest thing the core can be pointed at, not the hottest surface a scene may
    contain. A grey or reflective surface reaching the same radiance rails at the same place.
    """
    if ceiling_k <= 0.0:
        raise ValueError("a gain ceiling must be a positive temperature in kelvin")
    return float(lut.lookup(np.float64(ceiling_k), quantity)[()])


def clip_to_gain_ceiling(radiance: NDArray[np.floating], ceiling_lb: float) -> NDArray[np.floating]:
    """``min(L, L_ceiling)`` on the at-aperture plane, dtype and shape preserved.

    Returns the plane **unchanged (the same object)** when nothing reaches the ceiling, so a scene
    that never saturates is bit-identical to one rendered without a gain state and every golden
    written before `PH.8` still describes what the pipeline does.
    """
    if ceiling_lb <= 0.0:
        raise ValueError("the ceiling radiance must be positive")
    if radiance.dtype == np.float16:
        raise TypeError("float16 radiance is refused (CLAUDE.md #2)")
    if not np.any(radiance > ceiling_lb):
        return radiance
    return np.asarray(np.minimum(radiance, ceiling_lb), dtype=radiance.dtype)
