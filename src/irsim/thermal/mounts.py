"""A heat source bolted to a solved surface: a motor's heat into the end of its arm (EV.16).

docs/physics-model.md §6.1 (conduction into a facet), §6.6 (heat sources); ADR 0072 (the motor's
throttle law), ADR 0097 (the joint table), ADR 0187 (this module).

A quadrotor's motors are the hottest parts of it at cruise, 46-48 °C under 26 °C air, and they
are bolted to the ends of carbon arms. Until this module nothing carried that heat anywhere:
ADR 0072's motor is a temperature with no route into the skin, so the arm under a 47 °C motor
sat in the motor's own shadow at 28 °C -- the coolest point of an arm whose hottest point, on a
real airframe, is the mount.

**The mount is a boundary, not a member.** The motor's temperature is *prescribed* by its
throttle law; it is not solved, so it cannot be a lumped node the arm cools. It enters the arm's
cells as ``g_i (T_motor − T_i)``, with ``g_i`` the joint's contact conductance over cell *i*'s
share of the mount's footprint (:func:`irsim.thermal.coupling.footprint_conductances`: nearest
cells first, ``h_c · area`` in total, the same rule the asset path uses to bolt a hidden part to
its skin). That is one-way, which is right for a prescribed temperature -- the same rule TC.6's
``follows:`` nodes keep -- and it is stiff: 1000 W m⁻² K⁻¹ on carbon's areal capacity is a 2.5 s
time constant against a 10 s tick. So it rides the implicit conduction step
(:class:`~irsim.thermal.conduction.ConductionOperator`'s ``boundary_w_k``), never the explicit
surface balance, whose §6.4 bound it would break.

**Before take-off and after landing the motor is at air.** The motor's schedule covers the
mission and nothing else. The spin-up asks for the days before, and through the weather wrap the
same time of day *yesterday* falls inside the mission window, so a mount that read the schedule
at the wrapped instant would run the motors through yesterday's flight too. :class:`MountedForcing`
is therefore applied **outside** the wrap: it reads the true instant, and outside the schedule
the boundary is the cell's own air temperature -- a motor that is off.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from irsim.thermal.coupling import footprint_conductances

__all__ = ["Mount", "MountedForcing", "mount_conductances"]


@dataclass(frozen=True)
class Mount:
    """One body bolted to a surface: where, over how much area, how well, and how hot.

    ``temperature_at(t)`` answers inside ``covers(t)`` (the motor's mission schedule); outside it
    the motor is off and the boundary is the air.
    """

    name: str
    centre_m: tuple[float, float, float]
    area_m2: float
    h_c_w_m2_k: float
    temperature_at: Callable[[float], float]
    covers: Callable[[float], bool]

    def __post_init__(self) -> None:
        if not self.area_m2 > 0.0:
            raise ValueError(f"mount {self.name!r}: area_m2 must be positive")
        if not self.h_c_w_m2_k > 0.0:
            raise ValueError(f"mount {self.name!r}: h_c_w_m2_k must be positive")


def mount_conductances(patch: Any, mounts: Sequence[Mount]) -> NDArray[np.float64]:
    """``(n_cells,)`` W/K from the surface's cells to their mounts, summed over the mounts.

    Each mount takes the cells nearest its centre until their area reaches its own; the total is
    ``h_c · area`` per mount to rounding, whatever the grid.
    """
    g = np.zeros(int(patch.n_cells))
    for m in mounts:
        g = g + footprint_conductances(patch, m.centre_m, m.area_m2, m.h_c_w_m2_k)
    return g


class MountedForcing:
    """A surface's forcing with each cell's boundary temperature filled in from its mount.

    ``inner`` is the cell forcing (wrapped into the weather for a spin-up, or not); the mount is
    evaluated at the **true** instant ``t_s``. Cells under no mount carry the air temperature,
    which their zero conductance ignores.
    """

    def __init__(self, inner: Callable[[float], Any], mounts: Sequence[Mount], patch: Any) -> None:
        self.inner = inner
        self.mounts = tuple(mounts)
        n = int(patch.n_cells)
        # Which mount owns each cell: the one that conducts to it most. Two mounts never share a
        # cell on an airframe, and if they did the larger conductance is the better guess.
        shares = np.stack(
            [footprint_conductances(patch, m.centre_m, m.area_m2, m.h_c_w_m2_k) for m in mounts]
        )
        self._owner = np.where(shares.max(axis=0) > 0.0, shares.argmax(axis=0), -1)
        self._n = n

    def __call__(self, t_s: float) -> Any:
        f = self.inner(t_s)
        t_air = np.broadcast_to(np.asarray(f.t_air_k, dtype=np.float64), (self._n,))
        boundary = np.array(t_air, dtype=np.float64)
        for k, m in enumerate(self.mounts):
            if m.covers(t_s):
                boundary[self._owner == k] = float(m.temperature_at(t_s))
        return dataclasses.replace(f, boundary_k=boundary)
