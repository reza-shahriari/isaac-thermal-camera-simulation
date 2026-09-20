"""The scene's world frame: how its geometry's axes relate to east, north and up (PT.18).

docs/physics-model.md §5.4, §6.1 (the shadow term S); ADR 0095.

Every sun vector in this package is ENU -- east, north, up -- because that is what M6.4's NOAA
position gives and what `solar_loading` consumes. Every patch and occluder in a scene config is
authored in the scene's **world** frame, and nothing said which way that frame was up: the car
scenes are Y-up, as the stage they author (`car_demo.build_car_demo` sets it), while every ENU
convention is Z-up. A shadow test needs both in one frame, so the scene now declares the relation
once and this module carries it.

``WorldFrame(up, north)`` names two world-frame vectors; ``east = north × up`` completes the
right-handed triad, so a scene cannot declare a left-handed one by accident. The rotation is
exact for the axis-aligned frames every shipped scene uses (its entries are 0 and ±1, and a
product by either is exact in floating point), which is what lets the Y-up and Z-up authoring of
one wall be held to **bit-identical** shadow rather than to a tolerance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = ["ENU", "WorldFrame"]


def _unit(vector: Any, what: str) -> NDArray[np.float64]:
    v = np.asarray(vector, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(v))
    if norm <= 0.0:
        raise ValueError(f"world frame {what} has zero length")
    return np.asarray(v / norm)


@dataclass(frozen=True)
class WorldFrame:
    """Which world-frame direction is up and which is north; east follows."""

    up: NDArray[np.float64]
    north: NDArray[np.float64]

    def __post_init__(self) -> None:
        up = _unit(self.up, "up")
        north = _unit(self.north, "north")
        if abs(float(np.dot(up, north))) > 1e-9:
            raise ValueError(
                f"world frame up {tuple(up)} and north {tuple(north)} must be perpendicular"
            )
        object.__setattr__(self, "up", up)
        object.__setattr__(self, "north", north)

    @property
    def east(self) -> NDArray[np.float64]:
        """``north × up``, so that ``east × north = up`` as in ENU."""
        return np.asarray(np.cross(self.north, self.up))

    @property
    def world_from_enu(self) -> NDArray[np.float64]:
        """The 3×3 rotation whose columns are east, north and up in world coordinates."""
        return np.stack([self.east, self.north, self.up], axis=1)

    def to_world(self, vector_enu: Any) -> NDArray[np.float64]:
        """An ENU direction (..., 3) expressed in the world frame."""
        v = np.asarray(vector_enu, dtype=np.float64)
        return np.asarray(v @ self.world_from_enu.T)

    def to_enu(self, vector_world: Any) -> NDArray[np.float64]:
        """A world-frame direction (..., 3) expressed in ENU."""
        v = np.asarray(vector_world, dtype=np.float64)
        return np.asarray(v @ self.world_from_enu)

    @property
    def is_enu(self) -> bool:
        return bool(np.array_equal(self.world_from_enu, np.eye(3)))


#: The default: world *is* ENU, which every scene written before schema v8 assumed by omission.
ENU = WorldFrame(up=np.array([0.0, 0.0, 1.0]), north=np.array([0.0, 1.0, 0.0]))
